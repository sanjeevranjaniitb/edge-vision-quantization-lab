"""
Generate a LinkedIn-ready benchmark figure from benchmarks/results.csv.
Saves to benchmarks/sam21_apple_silicon_benchmark.jpg
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── load data ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
rows = []
with open(ROOT / "benchmarks/results.csv", newline="") as f:
    rows = list(csv.DictReader(f))

def get(rows, variant, col):
    for r in rows:
        if r["variant"] == variant:
            v = r[col]
            return float(v) if v else None
    return None

# Core ML rows only (comparable resolution)
COREML = ["small_fp16", "small_int8", "large_fp16", "large_int8"]
labels  = ["Small\nFP16", "Small\nINT8", "Large\nFP16", "Large\nINT8"]

p50      = [get(rows, v, "p50_ms")      for v in COREML]
p95      = [get(rows, v, "p95_ms")      for v in COREML]
rss      = [get(rows, v, "rss_mb")      for v in COREML]
size     = [get(rows, v, "model_size_mb") or 0.0 for v in COREML]
fps_vals = [get(rows, v, "fps")         for v in COREML]

# error bars: p95 - p50
yerr = [p95[i] - p50[i] for i in range(4)]

# ── style ─────────────────────────────────────────────────────────────────────
FP16_COLOR  = "#4C9BE8"   # blue
INT8_COLOR  = "#E8834C"   # orange
GRID_COLOR  = "#E8E8E8"
BG_COLOR    = "#FAFAFA"
TEXT_COLOR  = "#1A1A2E"

colors = [FP16_COLOR, INT8_COLOR, FP16_COLOR, INT8_COLOR]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

x = np.arange(4)
BAR_W = 0.55

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.patch.set_facecolor("white")
fig.suptitle(
    "SAM 2.1 Image Encoder — Quantization on Apple Silicon (M-series, macOS 15.7)",
    fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.98
)

def style_ax(ax, title, ylabel):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)

def bar_labels(ax, vals, fmt="{:.0f}"):
    for rect, v in zip(ax.patches, vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + rect.get_height() * 0.02,
            fmt.format(v),
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=TEXT_COLOR
        )

# ── panel 1: p50 latency with p95 error bar ──────────────────────────────────
ax = axes[0, 0]
bars = ax.bar(x, p50, width=BAR_W, color=colors, zorder=3, edgecolor="white", linewidth=0.5)
ax.errorbar(x, p50, yerr=yerr, fmt="none", color="#555", capsize=4, linewidth=1.2, zorder=4)
style_ax(ax, "Inference Latency  (p50, Core ML, 1024²)", "milliseconds")
bar_labels(ax, p50, "{:.0f} ms")
ax.set_ylim(0, max(p50) * 1.25)

# ── panel 2: throughput ───────────────────────────────────────────────────────
ax = axes[0, 1]
ax.bar(x, fps_vals, width=BAR_W, color=colors, zorder=3, edgecolor="white", linewidth=0.5)
style_ax(ax, "Throughput  (frames per second)", "FPS")
bar_labels(ax, fps_vals, "{:.2f} fps")
ax.set_ylim(0, max(fps_vals) * 1.3)

# ── panel 3: model size ───────────────────────────────────────────────────────
ax = axes[1, 0]
ax.bar(x, size, width=BAR_W, color=colors, zorder=3, edgecolor="white", linewidth=0.5)
style_ax(ax, "Model Size  (.mlpackage on disk)", "MB")
bar_labels(ax, size, "{:.0f} MB")
# annotate compression ratios
for s_fp16, s_int8, xi in [(size[0], size[1], 0.5), (size[2], size[3], 2.5)]:
    if s_fp16 > 0 and s_int8 > 0:
        ratio = (1 - s_int8 / s_fp16) * 100
        ax.annotate(
            f"−{ratio:.0f}%",
            xy=(xi, max(s_fp16, s_int8) * 1.05),
            ha="center", fontsize=8.5, color="#C0392B", fontweight="bold"
        )
ax.set_ylim(0, max(size) * 1.3)

# ── panel 4: process RSS ──────────────────────────────────────────────────────
ax = axes[1, 1]
ax.bar(x, rss, width=BAR_W, color=colors, zorder=3, edgecolor="white", linewidth=0.5)
style_ax(ax, "Process Memory  (RSS at inference)", "MB")
bar_labels(ax, rss, "{:.0f} MB")
ax.set_ylim(0, max(rss) * 1.25)

# ── shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=FP16_COLOR, label="FP16  (weight-only, activations FP16)"),
    mpatches.Patch(color=INT8_COLOR, label="INT8  (weight-only, activations FP16)"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center", ncol=2,
    fontsize=9, frameon=False,
    bbox_to_anchor=(0.5, 0.01)
)

# ── footnote ──────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.045,
    "All Core ML runs: 1024² input · 5 warmup · 20 measured iterations · "
    "coremltools 9.0 · PyTorch 2.13 · arm64",
    ha="center", fontsize=7.5, color="#888888"
)

plt.tight_layout(rect=[0, 0.07, 1, 0.97])

out = ROOT / "benchmarks/sam21_apple_silicon_benchmark.jpg"
fig.savefig(str(out), dpi=180, format="jpeg",
            bbox_inches="tight", pil_kwargs={"quality": 95})
print(f"Saved: {out}")
