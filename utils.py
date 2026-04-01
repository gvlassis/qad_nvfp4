import os
from itertools import chain

from transformers import AutoTokenizer
from datasets import load_dataset

from torch import inference_mode, tensor, int64, AdamW
from torch.utils.data import DataLoader
from torch.nn import functional as F

from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

import modelopt.torch.quantization as mtq


DEFAULT_EVAL_TASKS = ["arc_easy", "hellaswag", "piqa"]
MAP_BATCH_SIZE = 10_000 # default: 1000
MAP_NUM_PROC = os.cpu_count()//2


def load_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensures batched generation works correctly
    tokenizer.padding_side = "left"

    return tokenizer


def get_model_device(model):
    device = next(model.parameters()).device

    return device


@inference_mode()
def evaluate(tokenizer, model, tasks=DEFAULT_EVAL_TASKS, limit=100):
    hflm = HFLM(tokenizer=tokenizer, pretrained=model)

    results = simple_evaluate(
        model = hflm,
        tasks = tasks,
        num_fewshot = 0,
        batch_size = "auto",
        device = get_model_device(model), # Defaults to cuda
        use_cache = None, # Do NOT cache results
        rewrite_requests_cache = True, # Dataset requests cache
        limit = limit, # Total samples. Be careful, some datasets are very noisy and/or not even shuffled
        bootstrap_iters = 100, # Default=100_000, with bootstrap_iters=min(bootstrap_iters, 100)
        log_samples = False,
        verbosity = "ERROR",
        random_seed = None,
        numpy_random_seed = None,
        torch_random_seed = None,
        fewshot_random_seed = None
    )

    results = results["results"]

    print(results)


def get_dataloader(tokenizer, dataset, context_size=1024, batch_size=32):
    """
        Practical and clean sequence packing. No truncation or padding. Non-overlapping sequences are loaded sequentially and deterministically. For convenience, a tiny number of tail tokens are dropped.

        However, caching does not work in practise (opt for manual Dataset.save_to_disk()), and multiple context sizes are not supported (rather save as 1D).
    """
    
    dataset = load_dataset(dataset, split="train")
    
    def tokenize(batch):
        """batch = {"text": []}"""
        
        batch = tokenizer(
            batch["text"],
            add_special_tokens=False, # DANGER
            padding=False, # Dynamic padding more efficient (not used here)
            truncation=False,
            stride=0,
            return_overflowing_tokens=False,
            verbose=False
        )

        batch = batch["input_ids"]

        batch = [sample+[tokenizer.eos_token_id] for sample in batch]

        return {"input_ids": batch}

    def pack(batch):
        """batch = {"input_ids": []}"""

        batch = batch["input_ids"]
        batch = chain.from_iterable(batch)
        batch = list(batch)
        
        # Drop tail (otherwise I cannot use map()). Negligible impact for large batch size.
        samples = len(batch) // context_size

        batch = [batch[i*context_size : (i+1)*context_size] for i in range(samples)]

        return {"input_ids": batch}

    def tokenize_and_pack(batch):
        """batch = {"input_ids": []}"""

        batch = tokenize(batch)
        batch = pack(batch)

        return batch
    
    # map() enables caching, memory mapping, multiprocessing and returns Dataset
    dataset = dataset.map(
        tokenize_and_pack,
        batched=True,
        batch_size=MAP_BATCH_SIZE,
        remove_columns=dataset.column_names,
        num_proc=MAP_NUM_PROC
    )

    def collate_fn(batch):
        """batch = [{"input_ids": []}]"""
        
        batch = [sample["input_ids"] for sample in batch]
        batch = tensor(batch, dtype=int64)

        return {"input_ids": batch, "labels": batch.clone()} # Models shift labels internally

    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=False, # Reshuffling at the START of every epoch
        sampler=None,
        num_workers=0, # For LLMs, keep at 0 (loading from main thread/synchronous)
        collate_fn=collate_fn,
        pin_memory=False, # Page-locked RAM
        drop_last=True,
        persistent_workers=False
    )
    
    return dataloader


def quantize(tokenizer, model, dataset, context_size=1024, batches=128, batch_size=32):
    dataloader = get_dataloader(tokenizer, dataset, context_size, batch_size)
    device = get_model_device(teacher)

    optimizer = AdamW(student.parameters(), lr=1e-5)

    for batch_idx, batch in zip(range(batches), dataloader):

        with inference_mode():
            teacher_logits = teacher(batch["input_ids"].to(device)).logits # (batch_size, context_size, vocab_size)
            teacher_logits = teacher_logits[:, :-1, :] # Drop last token (unaligned)
            teacher_logits = teacher_logits.flatten(0, -2) # (batch_size*context_size, vocab_size)

        student_logits = student(batch["input_ids"].to(device)).logits
        student_logits = student_logits[:, :-1, :]
        student_logits = student_logits.flatten(0, -2)
        
        # BE CAREFUL OF THE ORDERING
        loss = kl_div(student_logits, teacher_logits, temperature)

        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
