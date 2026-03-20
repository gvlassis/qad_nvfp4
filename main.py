from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transformers import AutoModelForCausalLM
from utils import DEFAULT_EVAL_TASKS, load_tokenizer, evaluate, quantize

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", help="Model to apply QAD on. Used both as the teacher and the student.", default="Qwen/Qwen2.5-0.5B")

parser.add_argument("--eval_tasks", help="Tasks to evaluate unquantized teacher and quantized student after PTQ and after QAD", nargs="+", default=DEFAULT_EVAL_TASKS)
parser.add_argument("--eval_limit", help="Total samples used for every evaluation task", type=int, default=100)

parser.add_argument("--ptq_dataset", help="Dataset used for NVFP4 calibration", default="gvlassis/ClimbMix10M")
parser.add_argument("--ptq_keep", help="Keep calibration dataset in RAM", action="store_true")
parser.add_argument("--ptq_samples", help="Total documents used for calibration", type=int, default=128)
parser.add_argument("--ptq_batch_size", help="Documents in every calibration batch", type=int, default=32)

parser.add_argument("--device_index", help="GPU index (NVIDIA support only)", type=int, default=0)

args = parser.parse_args()


device_type = "cuda"
device = f"{device_type}:{args.device_index}"


tokenizer = load_tokenizer(args.model)

teacher = AutoModelForCausalLM.from_pretrained(args.model)
teacher = teacher.to(device)

evaluate(tokenizer, teacher, args.eval_tasks, args.eval_limit)

student = AutoModelForCausalLM.from_pretrained(args.model)
student = student.to(device)
student = quantize(tokenizer, student, args.ptq_dataset, args.ptq_keep, args.ptq_samples, args.ptq_batch_size)

evaluate(tokenizer, student, args.eval_tasks, args.eval_limit)

# distil(teacher, student)
#
# evaluate(tokenizer, student, args.eval_tasks, args.eval_limit)
