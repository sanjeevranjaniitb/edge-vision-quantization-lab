import argparse, sys, platform, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import coremltools as ct
import psutil
from edgevision.evaluation import append_csv

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = {
    "small_fp16": ROOT / "checkpoints/sam2.1_hiera_small_encoder_fp16.mlpackage",
    "small_int8": ROOT / "checkpoints/sam2.1_hiera_small_encoder_int8.mlpackage",
    "large_fp16": ROOT / "checkpoints/sam2.1_hiera_large_encoder_fp16.mlpackage",
    "large_int8": ROOT / "checkpoints/sam2.1_hiera_large_encoder_int8.mlpackage",
}

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=list(PACKAGES), default=None,
                    help="Which package to benchmark (default: all available)")
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--iterations", type=int, default=20)
args = parser.parse_args()

targets = {args.variant: PACKAGES[args.variant]} if args.variant else PACKAGES


def package_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def run_benchmark(name: str, pkg_path: Path, warmup: int, iterations: int):
    if not pkg_path.exists():
        print(f"  SKIP {name}: {pkg_path} not found")
        return

    print(f"\n[{name}] Loading {pkg_path.name} ...")
    model = ct.models.MLModel(str(pkg_path))

    # 1024x1024 input — matches export resolution
    dummy = {"image": np.random.rand(1, 3, 1024, 1024).astype(np.float32)}

    print(f"[{name}] Warming up ({warmup} iters) ...")
    for _ in range(warmup):
        model.predict(dummy)

    print(f"[{name}] Benchmarking ({iterations} iters) ...")
    process = psutil.Process()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.predict(dummy)
        samples.append((time.perf_counter() - t0) * 1000)

    samples.sort()
    mean = sum(samples) / len(samples)
    result = {
        "variant": name,
        "backend": "coreml",
        "device": "ane+gpu",
        "mean_ms": round(mean, 3),
        "p50_ms": round(samples[len(samples) // 2], 3),
        "p95_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 3),
        "fps": round(1000 / mean, 3),
        "rss_mb": round(process.memory_info().rss / 1024 ** 2, 2),
        "model_size_mb": round(package_size_mb(pkg_path), 1),
        "warmup": warmup,
        "iterations": iterations,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    append_csv(result)
    print(f"[{name}] mean={result['mean_ms']}ms  p50={result['p50_ms']}ms  "
          f"p95={result['p95_ms']}ms  fps={result['fps']}  "
          f"rss={result['rss_mb']}MB  size={result['model_size_mb']}MB")


for name, path in targets.items():
    run_benchmark(name, path, args.warmup, args.iterations)

print("\nDone. Results appended to benchmarks/results.csv")
