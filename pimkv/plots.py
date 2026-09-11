"""Figure generation (Phase 2/4 — AGENT_BRIEF §9).

    python -m pimkv.plots headline  results/headline/summary.csv figures/headline.png
    python -m pimkv.plots bandwidth results/headline/summary.csv figures/bandwidth.png
    python -m pimkv.plots m1-hist   <run.csv> <out.png>

Every plot reads CSVs produced by pimkv.run/pimkv.sweep — nothing is
fabricated. Style follows the dataviz reference palette: color identifies the
ALLOCATOR (fixed slot order: paged=blue, pim-aware=orange, contiguous=aqua —
a validated all-pairs categorical trio), line style identifies the workload,
grids recessive, text in ink colors, fonts sized for a compressed video.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dataviz reference palette (light mode)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
ALLOC_COLOR = {          # fixed categorical slot order — never re-assigned
    "paged": "#2a78d6",
    "pim-aware": "#eb6834",
    "contiguous": "#1baf7a",
}
ALLOC_LABEL = {"paged": "paged (vLLM port)", "pim-aware": "PIM-aware",
               "contiguous": "contiguous (oracle)"}
WL_STYLE = {"steady": "-", "spec": "--"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 13,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _style_axis(ax):
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def headline(summary_csv: Path, out_path: Path) -> None:
    """M1 (top) and M3 (bottom) vs block size; three allocators; steady
    solid, spec dashed. The money shot."""
    df = pd.read_csv(summary_csv)
    df = df[df.workload.isin(("steady", "spec"))]
    ideal = None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for wl in ("steady", "spec"):
        for al in ("paged", "pim-aware", "contiguous"):
            d = (df[(df.workload == wl) & (df.allocator == al)]
                 .sort_values("block_tokens"))
            if d.empty:
                continue
            for ax, col in ((ax1, "m1_mean"), (ax2, "m3_gbps_mean")):
                ax.plot(d.block_tokens, d[col], WL_STYLE[wl],
                        color=ALLOC_COLOR[al], lw=2, marker="o", ms=5,
                        zorder=3,
                        label=(f"{ALLOC_LABEL[al]}, {wl}"
                               if (wl == "steady" or al == "paged")
                               else None))
    for ax in (ax1, ax2):
        _style_axis(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df.block_tokens.unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        for x, txt in ((16, "vLLM default"), (128, "derived\nPIM-natural")):
            if x in df.block_tokens.values:
                ax.axvline(x, color=MUTED, lw=1, ls=":", zorder=1)
        ax1.set_ylim(0, 1.02)
    for x, txt in ((16, "vLLM default (16)"), (128, "derived PIM-natural (128)")):
        if x in df.block_tokens.values:
            ax1.annotate(txt, (x, 0.04), textcoords="offset points",
                         xytext=(6, 0), fontsize=11, color=MUTED)
    ax1.set_ylabel("M1 — all-bank row-hit rate")
    ax2.set_ylabel("M3 — effective PIM bandwidth (GB/s)")
    ax2.set_xlabel("KV block size (tokens)")
    ax2.set_ylim(bottom=0)
    ax1.set_title("Paged for capacity, punished by rows — "
                  "HBM3-PIM, realistic host address map",
                  color=INK, fontsize=15, pad=12)
    ax1.legend(loc="center right", fontsize=11, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def bandwidth(summary_csv: Path, out_path: Path, ideal_gbps: float) -> None:
    """M3 per workload at the serving-standard block size (16), grouped by
    allocator, ideal all-bank bandwidth as reference line."""
    df = pd.read_csv(summary_csv)
    df = df[df.block_tokens == 16]
    wls = [w for w in ("steady", "longctx", "prefix", "spec")
           if w in set(df.workload)]
    allocs = ["paged", "pim-aware", "contiguous"]
    x = np.arange(len(wls))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _style_axis(ax)
    for k, al in enumerate(allocs):
        vals = [float(df[(df.workload == w) & (df.allocator == al)]
                      .m3_gbps_mean.mean()) for w in wls]
        bars = ax.bar(x + (k - 1) * width, vals, width * 0.92,
                      color=ALLOC_COLOR[al], label=ALLOC_LABEL[al],
                      edgecolor=SURFACE, linewidth=2, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=10, color=INK,
                        xytext=(0, 2), textcoords="offset points")
    ax.axhline(ideal_gbps, color=MUTED, lw=1.4, ls="--", zorder=2)
    ax.annotate(f"ideal all-bank  {ideal_gbps:,.0f} GB/s",
                (len(wls) - 0.55, ideal_gbps), va="bottom", ha="right",
                fontsize=11, color=MUTED)
    ax.set_xticks(x, wls)
    ax.set_ylabel("M3 — effective PIM bandwidth (GB/s)")
    ax.set_ylim(0, ideal_gbps * 1.08)
    ax.set_title("Effective PIM bandwidth by workload — 16-token blocks, "
                 "HBM3-PIM, realistic host map", color=INK, fontsize=14,
                 pad=12)
    ax.legend(fontsize=11, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# sequential blue ramp, light -> dark (dataviz reference palette)
SEQ_BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
             "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281", "#0d366b"]


def heatmap(summary_csv: Path, out_path: Path) -> None:
    """M1 over (block size x context length), paged allocator — sequential
    single-hue ramp, every cell direct-labeled."""
    from matplotlib.colors import LinearSegmentedColormap
    df = pd.read_csv(summary_csv)
    df = df[(df.allocator == "paged") & df.prompt_len.notna()]
    bts = sorted(df.block_tokens.unique())
    pls = sorted(df.prompt_len.unique())
    grid = np.full((len(pls), len(bts)), np.nan)
    for i, pl in enumerate(pls):
        for j, bt in enumerate(bts):
            d = df[(df.prompt_len == pl) & (df.block_tokens == bt)]
            if len(d):
                grid[i, j] = float(d.m1_mean.mean())
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUES)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin="lower",
                   aspect="auto")
    for i in range(len(pls)):
        for j in range(len(bts)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=11,
                        color=INK if grid[i, j] < 0.55 else "#ffffff")
    ax.set_xticks(range(len(bts)), [str(b) for b in bts])
    ax.set_yticks(range(len(pls)), [f"{int(p):,}" for p in pls])
    ax.set_xlabel("KV block size (tokens)")
    ax.set_ylabel("context length (prompt tokens)")
    ax.set_title("M1 row-hit rate — paged allocator, HBM3-PIM, "
                 "realistic host map", color=INK, fontsize=14, pad=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("M1 — all-bank row-hit rate")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def m1_hist(csv_path: Path, out_path: Path) -> None:
    """Histogram of per-decode-step M1 from one run CSV (Phase 0 check)."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _style_axis(ax)
    ax.hist(df.m1, bins=50, color="#2a78d6", edgecolor=SURFACE, zorder=3)
    ax.set_xlabel("M1 — all-bank row-hit rate", fontsize=14)
    ax.set_ylabel("sampled decode steps", fontsize=14)
    ax.set_xlim(0, 1)
    ax.axvline(df.m1.mean(), color="#eb6834", lw=2,
               label=f"mean = {df.m1.mean():.3f}")
    ax.legend(fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) == 3 and args[0] == "m1-hist":
        m1_hist(Path(args[1]), Path(args[2]))
        return 0
    if len(args) == 3 and args[0] == "heatmap":
        heatmap(Path(args[1]), Path(args[2]))
        return 0
    if len(args) == 3 and args[0] == "headline":
        headline(Path(args[1]), Path(args[2]))
        return 0
    if len(args) == 3 and args[0] == "bandwidth":
        from .config import DEFAULT_TIMING, HBM3_PIM
        g, t = HBM3_PIM, DEFAULT_TIMING
        ideal = g.channels * g.banks_per_channel * g.burst_bytes / t.tccd_ab_ns
        bandwidth(Path(args[1]), Path(args[2]), ideal)
        return 0
    print("usage: python -m pimkv.plots {headline|bandwidth|heatmap} "
          "<summary.csv> <out.png>\n"
          "       python -m pimkv.plots m1-hist <run.csv> <out.png>",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
