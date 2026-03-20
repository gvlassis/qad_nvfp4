from transformers import AutoTokenizer
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate
from datasets import load_dataset
import modelopt.torch.quantization as mtq

DEFAULT_EVAL_TASKS = ["arc_easy", "hellaswag", "piqa"]

def load_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Ensures batched generation works correctly
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_model_device(model):
    device = next(model.parameters()).device

    return device

@torch.inference_mode()
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

def get_ptq_dataset(dataset, keep_in_memory=False, samples=128):
    ptq_dataset = load_dataset(dataset, split="train", keep_in_memory=keep_in_memory)
    ptq_dataset = ptq_dataset[:samples]["text"]

    return ptq_dataset

def quantize(tokenizer, model, dataset, keep_in_memory=False, samples=128, batch_size=32):
    dataset = get_ptq_dataset(dataset, keep_in_memory, samples)
    batches = samples//batch_size
    device = get_model_device(model)
    
    @torch.inference_mode()
    def forward_loop(model):
        for batch_idx in range(batches):
            batch = dataset[batch_idx*batch_size:(batch_idx+1)*batch_size]

            batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            
            model(batch["input_ids"].to(device), batch["attention_mask"].to(device))

    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)

    return model
