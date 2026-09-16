"""Figure for the K/V tensor-layout measurement (scripts/kv_split.py).

    .venv/bin/python scripts/plot_kv_split.py \
        results/kv_split/summary.csv figures/kv_split.png

Left: all-bank row-hit rate. Right: effective PIM bandwidth, with the
PIM-aware ÷ paged gain above each pair. Palette follows pimkv.plots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
COLOR = {"paged": "#2a78d6", "pim-aware": "#eb6834"}
LABEL = {"paged": "paged (vLLM port)", "pim-aware": "KV-PIMple"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11.5,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def panel(ax, df, col, ylabel, title, fmt, gain_fmt=None, ymax=None):
    layouts = list(dict.fromkeys(df.layout))
    w = 0.34
    for j, al in enumerate(("paged", "pim-aware")):
        ys = [float(df[(df.layout == L) & (df.allocator == al)][col].iloc[0])
              for L in layouts]
        b = ax.bar([i + (j - 0.5) * w for i in range(len(layouts))], ys, w,
                   color=COLOR[al], label=LABEL[al], zorder=3)
        ax.bar_label(b, fmt=fmt, fontsize=10.5, color=INK, padding=2)
    if gain_fmt:
        for i, L in enumerate(layouts):
            p = float(df[(df.layout == L) & (df.allocator == "paged")][col].iloc[0])
            x = float(df[(df.layout == L) & (df.allocator == "pim-aware")][col].iloc[0])
            ax.text(i, max(p, x) * 1.14, gain_fmt.format(x / p), ha="center",
                    va="bottom", fontsize=13, color=INK, fontweight="bold")
    ticks = []
    for L in layouts:
        r = df[df.layout == L].iloc[0]
        ticks.append(f"{L}\n{r.block_kib:.0f} KiB block · "
                     f"{r.blocks_per_region:.0f} blocks/region")
    ax.set_xticks(range(len(layouts)), ticks, fontsize=10.5)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.set_title(title, loc="left", fontsize=12.5, color=INK, pad=10)
    _style(ax)


def main(argv) -> int:
    df = pd.read_csv(Path(argv[0]))
    out = Path(argv[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    panel(axes[0], df, "m1_mean", "all-bank row-hit rate",
          "Row-hit rate", "%.3f", gain_fmt=None, ymax=1.13)
    axes[0].axhline(0.969, color=MUTED, lw=1, ls=":")
    axes[0].text(0.02, 0.975, "ceiling 0.969", ha="left", va="bottom",
                 transform=axes[0].get_yaxis_transform(), fontsize=9.5,
                 color=MUTED)
    panel(axes[1], df, "m3_gbps_mean", "effective PIM bandwidth (GB/s)",
          "Bandwidth", "%.0f", gain_fmt=None, ymax=4150)
    axes[1].axhline(3810, color=MUTED, lw=1, ls=":")
    axes[1].text(0.02, 3830, "ideal all-bank 3,810", ha="left", va="bottom",
                 fontsize=9.5, color=MUTED,
                 transform=axes[1].get_yaxis_transform())
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=11, ncol=2, loc="upper right",
               bbox_to_anchor=(0.985, 0.925))
    fig.suptitle("Separate K and V tensors halve the block per tensor",
                 fontsize=14, color=INK, x=0.055, ha="left", y=0.965)
    fig.text(0.055, 0.02, "Steady workload, 1,000 requests, 16-token blocks, "
             "HBM3-PIM, host-cacheline map. Combined K+V is the layout we "
             "report; split is what vLLM v0 allocates.",
             fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.055, 1, 0.87))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
