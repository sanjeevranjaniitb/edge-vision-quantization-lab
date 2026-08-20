import argparse, sys, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from edgevision.benchmark import benchmark_callable, file_size_mb
from edgevision.sam_model import load_sam2, model_summary, select_device
from edgevision.evaluation import append_csv

VARIANTS = {
    "sam2.1_hiera_small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
    "sam2.1_hiera_large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model-variant", choices=VARIANTS, default="sam2.1_hiera_small")
parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
parser.add_argument("--input-size", type=int, default=1024)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iterations", type=int, default=30)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
config, ckpt = VARIANTS[args.model_variant]
ckpt_path = root / ckpt

device = select_device()
model = load_sam2(config, str(ckpt_path), device=device, precision=args.precision)
summary = model_summary(model)

h = w = args.input_size
with torch.inference_mode():
    def forward():
        x = torch.randn(1, 3, h, w, device=device)
        if args.precision == "fp16":
            x = x.half()
        return model.image_encoder(x)

result = benchmark_callable(forward, args.warmup, args.iterations)
result.update({
    "variant": f"{args.model_variant.split('_')[-1]}_{args.precision}",
    "backend": "pytorch",
    "device": device,
    "input_size": args.input_size,
    "model_size_mb": round(file_size_mb(ckpt_path), 1),
    "parameters": summary["parameters"],
    "warmup": args.warmup,
    "iterations": args.iterations,
    "machine": platform.machine(),
    "platform": platform.platform(),
})
append_csv(result)
print(result)
