from __future__ import annotations
from pathlib import Path
import torch

def select_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"

def load_sam2(config, checkpoint, device=None, precision="fp32"):
    from sam2.build_sam import build_sam2
    device = device or select_device()
    model = build_sam2(config, checkpoint, device=device)
    if precision == "fp16":
        model.half()
    model.eval()
    return model

def model_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters": total, "trainable_parameters": trainable}
