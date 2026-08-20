# Edge Vision Quantization - SAM 2.1 on Apple Silicon

Hands on model compression trade-offs for a modern vision foundation model targeting Apple Silicon edge deployment.

The experiment is not a demonstration that INT8 is smaller. It is a measurement of the actual Pareto frontier across **segmentation quality × inference latency × memory footprint × model size** on real hardware with a real runtime.

---

## Research question

> How much can a vision foundation model be compressed before its practical perception quality becomes unacceptable for edge deployment — and does quantization actually improve runtime performance on Apple Silicon ?

The target model is **SAM 2.1 (Segment Anything Model 2.1)** by Meta AI, evaluated on its image segmentation path. SAM 2.1 is a promptable segmentation foundation model with a hierarchical image encoder (Hiera), a prompt encoder, and a mask decoder. It presents a meaningful compression challenge: the encoder alone accounts for the majority of compute and memory, and its multi-scale feature pyramid output makes naive tracing non-trivial.

---

## Key findings

Benchmarked on Apple Silicon (M-series, macOS 15.7, arm64) using Core ML with `minimum_deployment_target=macOS13`.

| Variant | Backend | Input | p50 latency | FPS | Model size | Process RSS |
|---|---|---|---|---|---|---|
| small FP32 | PyTorch MPS | 256² | 21.9 ms | 45.6 | 176 MB (.pt) | 748 MB |
| small FP16 | PyTorch MPS | 256² | 22.2 ms | 44.8 | 176 MB (.pt) | 748 MB |
| small FP16 | Core ML | 1024² | 165.9 ms | 6.0 | 125.5 MB | 910 MB |
| small INT8 | Core ML | 1024² | 208.6 ms | 4.8 | 85.1 MB | 1045 MB |
| large FP16 | Core ML | 1024² | 893.6 ms | 1.1 | 488.9 MB | 1404 MB |
| large INT8 | Core ML | 1024² | 1013.7 ms | 1.0 | 267.7 MB | 855 MB |

**Observations:**

- **INT8 is slower than FP16 on Apple Silicon for this model.** The ANE and GPU dequantize INT8 weights to FP16 at runtime before computation. For a model of this depth and channel width, the dequantization overhead exceeds any bandwidth savings from the smaller weight representation. This is the expected result for activation-compute-bound workloads on unified memory architectures.
- **INT8 does win on model size** — 32% reduction for small (125.5 → 85.1 MB), 45% for large (488.9 → 267.7 MB). For over-the-air delivery or storage-constrained deployment, this matters.
- **INT8 wins on RSS for the large model** — 855 MB vs 1404 MB. When the model approaches the memory budget of the device, INT8 becomes the only viable option regardless of latency.
- **FP16 is not faster than FP32 on MPS** for this model at 256² input. Apple Silicon's unified memory architecture means FP32 and FP16 share the same memory bus; the cast overhead at load time is negligible but the compute savings are also negligible at this resolution.
- **The PyTorch MPS and Core ML numbers are not directly comparable** — the MPS baseline used 256² synthetic input (smoke test), while Core ML ran the full 1024² production resolution. A fair comparison requires matching input resolution.

---

## Reproducing the experiment

### Prerequisites

Python 3.10–3.12 is required. coremltools 9.x does not ship native extensions for Python 3.13+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Known compatibility issue — coremltools 9.0 + PyTorch 2.13:**

Two patches are required before export will succeed:

1. `upsample_bicubic2d` is not implemented in the coremltools MIL frontend. `export_sam.py` registers a bilinear fallback automatically — no manual action needed.

2. The `_cast` / `aten::Int` handler in `coremltools/converters/mil/frontend/torch/ops.py` fails on multi-element numpy arrays produced by `aten::size`. Patch line ~3037:

```python
# Before
int(val)

# After
int(val.flat[0]) if hasattr(val, "flat") else int(val)
```

This is a one-line edit to the installed package source. It will be unnecessary once coremltools ships official PyTorch 2.x support.

### Phase 1 — download checkpoints

```bash
python scripts/download_sam.py --variant sam2.1_hiera_small
python scripts/download_sam.py --variant sam2.1_hiera_large
```

### Phase 2 — PyTorch MPS baseline

```bash
python scripts/benchmark_sam.py --model-variant sam2.1_hiera_small --precision fp32
python scripts/benchmark_sam.py --model-variant sam2.1_hiera_small --precision fp16
```

### Phase 3 — export to Core ML

```bash
python scripts/export_sam.py --variant sam2.1_hiera_small --precision fp16
python scripts/export_sam.py --variant sam2.1_hiera_small --precision int8
python scripts/export_sam.py --variant sam2.1_hiera_large --precision fp16
python scripts/export_sam.py --variant sam2.1_hiera_large --precision int8
```

### Phase 4 — Core ML benchmark

```bash
python scripts/benchmark_coreml.py --warmup 5 --iterations 20
```

Results are appended to `benchmarks/results.csv`.

---

## Implementation notes

### Encoder tracing

SAM 2.1's image encoder returns a dict with mixed `Tensor` / `List[Tensor]` outputs:

```python
{
    "vision_features":  Tensor,           # [1, C, H/16, W/16]
    "backbone_fpn":     List[Tensor x3],  # multi-scale FPN features
    "vision_pos_enc":   List[Tensor x3],  # positional encodings per scale
}
```

`torch.jit.trace` cannot handle dict outputs with heterogeneous value types. `EncoderWrapper` in `export_sam.py` flattens this to a 7-tuple, which traces cleanly and maps to named Core ML outputs.

### Precision allocation

All Core ML models are converted with `compute_precision=FLOAT16`. INT8 applies post-conversion weight-only quantization (`linear_symmetric`, `per_channel`) via `coremltools.optimize.coreml.linear_quantize_weights`. Activations remain FP16 in both cases — this is weight-only quantization, not full INT8 inference.

### Why the encoder only

The image segmentation path of SAM 2.1 separates cleanly into encoder (Hiera backbone + FPN neck) and decoder (prompt encoder + mask decoder). The encoder accounts for ~95% of FLOPs and ~90% of parameters. Benchmarking the encoder in isolation gives a clean signal on compression sensitivity before introducing the prompt-conditioned decoder path.

---

All benchmark runs record: variant, backend, device, input resolution, warmup iterations, measured iterations, p50/p95 latency, throughput, process RSS, model artifact size, macOS version, and chip architecture. No numbers are hard-coded in source. All results live in `benchmarks/results.csv`.

Hardware: Apple Silicon (arm64), macOS 15.7.7.
