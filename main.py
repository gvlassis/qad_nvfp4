from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transformers import AutoModelForCausalLM
from utils import DEFAULT_EVAL_TASKS, load_tokenizer, evaluate, quantize, distil


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", help="Model to apply QAD on. Used both as the teacher and the student.", default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--context_size", help="Overrides model sequence length (calibration+distillation)", type=int, default=1024)

parser.add_argument("--eval_tasks", help="Tasks to evaluate unquantized teacher and quantized student after PTQ and after QAD", nargs="+", default=DEFAULT_EVAL_TASKS)
parser.add_argument("--eval_limit", help="Total samples used for every evaluation task", type=int, default=100)

parser.add_argument("--ptq_dataset", help="Dataset used for NVFP4 calibration (forward only)", default="JeanKaddour/minipile")
parser.add_argument("--ptq_batches", help="Number of batches used for calibration. We do sequence packing.", type=int, default=128)
parser.add_argument("--ptq_batch_size", help="Size of batches used for calibration", type=int, default=32)

parser.add_argument("--qad_dataset", help="Dataset used for QAD (forward+backward)", default="JeanKaddour/minipile")
parser.add_argument("--qad_batches", help="Number of batches used for distillation", type=int, default=128)
parser.add_argument("--qad_batch_size", help="Size of batches used for distillation", type=int, default=32)
parser.add_argument("--temperature", help="Temperature used for distillation", type=float, default=1)

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
student = quantize(tokenizer, student, args.ptq_dataset, args.context_size, args.ptq_batches, args.ptq_batch_size)

evaluate(tokenizer, student, args.eval_tasks, args.eval_limit)

distil(tokenizer, teacher, student, args.qad_dataset, args.context_size, args.qad_batches, args.qad_batch_size, args.temperature)

evaluate(tokenizer, student, args.eval_tasks, args.eval_limit)
