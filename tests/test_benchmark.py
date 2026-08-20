import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgevision.benchmark import benchmark_callable

def test_benchmark():
    r = benchmark_callable(lambda: sum(range(100)), warmup=1, iterations=3)
    assert r["mean_ms"] >= 0
    assert r["fps"] > 0
    assert r["p95_ms"] >= r["p50_ms"]
