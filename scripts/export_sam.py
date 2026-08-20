import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import coremltools as ct
from coremltools.converters.mil import register_torch_op, Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights,
)
from edgevision.sam_model import load_sam2


# coremltools 9 has no MIL translation for upsample_bicubic2d.
# Replace with bilinear — positional embeddings are not sensitive to this.
@register_torch_op
def upsample_bicubic2d(context, node):
    inputs = _get_inputs(context, node, expected=4)
    x = inputs[0]
    out_size = inputs[1].val
    in_h = float(x.shape[2])
    in_w = float(x.shape[3])
    result = mb.upsample_bilinear(
        x=x,
        scale_factor_height=float(out_size[0]) / in_h,
        scale_factor_width=float(out_size[1]) / in_w,
        align_corners=bool(inputs[2].val),
        name=node.name,
    )
    context.add(result)


VARIANTS = {
    "sam2.1_hiera_small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
    "sam2.1_hiera_large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["fp16", "int8"], required=True)
parser.add_argument("--variant", choices=VARIANTS, default="sam2.1_hiera_small")
parser.add_argument("--input-size", type=int, default=1024)
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
config, ckpt = VARIANTS[args.variant]
ckpt_path = root / ckpt
out_dir = root / "checkpoints"

print(f"Loading {args.variant} ...")
model = load_sam2(config, str(ckpt_path), device="cpu", precision="fp32")
encoder = model.image_encoder.eval()

h = w = args.input_size
example = torch.zeros(1, 3, h, w)


class EncoderWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x):
        out = self.enc(x)
        return (
            out["vision_features"],
            out["backbone_fpn"][0],
            out["backbone_fpn"][1],
            out["backbone_fpn"][2],
            out["vision_pos_enc"][0],
            out["vision_pos_enc"][1],
            out["vision_pos_enc"][2],
        )


print("Tracing image encoder ...")
with torch.no_grad():
    traced = torch.jit.trace(EncoderWrapper(encoder), example)

print("Converting to Core ML ...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="image", shape=example.shape)],
    minimum_deployment_target=ct.target.macOS13,
    compute_precision=ct.precision.FLOAT16,
)

if args.precision == "int8":
    print("Applying INT8 weight quantization ...")
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
    config_obj = OptimizationConfig(global_config=op_config)
    mlmodel = linear_quantize_weights(mlmodel, config_obj)

out_path = out_dir / f"{args.variant}_encoder_{args.precision}.mlpackage"
mlmodel.save(str(out_path))

size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6
print(f"Saved: {out_path}")
print(f"Size:  {size_mb:.1f} MB")
