from __future__ import annotations
import platform, time
from pathlib import Path
import psutil

def system_info():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }

def file_size_mb(path):
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / (1024**2)
    return sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / (1024**2)

def benchmark_callable(fn, warmup=10, iterations=30):
    for _ in range(warmup):
        fn()
    samples = []
    process = psutil.Process()
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    mean = sum(samples) / len(samples)
    return {
        "mean_ms": round(mean, 3),
        "p50_ms": round(samples[len(samples)//2], 3),
        "p95_ms": round(samples[min(len(samples)-1, int(len(samples)*.95))], 3),
        "fps": round(1000 / mean, 3),
        "rss_mb": round(process.memory_info().rss / 1024**2, 2),
    }
