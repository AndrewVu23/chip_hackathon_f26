"""Figure generation (Phase 2/4 — AGENT_BRIEF §9).

    python -m pimkv.plots headline  results/headline/summary.csv figures/headline.png
    python -m pimkv.plots bandwidth results/headline/summary.csv figures/bandwidth.png
    python -m pimkv.plots m1-hist   <run.csv> <out.png>
    python -m pimkv.plots validation results/phaseE/summary.csv \
                                     results/phaseE2/summary.csv figures/validation.png

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
               "contiguous": "contiguous (knows lengths)"}
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
                        zorder=3)
    for ax in (ax1, ax2):
        _style_axis(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df.block_tokens.unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        for x in (16, 128):
            if x in df.block_tokens.values:
                ax.axvline(x, color=MUTED, lw=1.4, ls=(0, (2, 3)), zorder=1)
    ax1.set_ylim(0, 1.02)
    for x, txt in ((16, "vLLM default (16)"), (128, "derived PIM-natural (128)")):
        if x in df.block_tokens.values:
            ax1.annotate(txt, (x, 0.04), textcoords="offset points",
                         xytext=(6, 0), fontsize=11, color=MUTED)
    # legend: color = allocator, line style = workload
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ALLOC_COLOR[al], lw=2.5,
                      label=ALLOC_LABEL[al])
               for al in ("paged", "pim-aware", "contiguous")]
    handles += [Line2D([], [], color=INK, lw=1.8, ls=WL_STYLE[wl],
                       label=f"{wl} workload") for wl in ("steady", "spec")]
    ax1.legend(handles=handles, loc="center right", fontsize=11,
               framealpha=0.9)
    ax1.set_ylabel("all-bank row-hit rate")
    ax2.set_ylabel("effective PIM bandwidth (GB/s)")
    ax2.set_xlabel("KV block size (tokens)")
    ax2.set_ylim(bottom=0)
    ax1.set_title("Paged for capacity, punished by rows — "
                  "HBM3-PIM, realistic host address map",
                  color=INK, fontsize=15, pad=12)
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
    ax.set_ylabel("effective PIM bandwidth (GB/s)")
    ax.set_ylim(0, ideal_gbps * 1.08)
    ax.set_title("Effective PIM bandwidth by workload\n"
                 "16-token blocks, HBM3-PIM, realistic host map",
                 color=INK, fontsize=14, pad=10)
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
    ax.set_title("All-bank row-hit rate — paged allocator, HBM3-PIM, "
                 "realistic host map", color=INK, fontsize=14, pad=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("all-bank row-hit rate")
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
    ax.set_xlabel("all-bank row-hit rate", fontsize=14)
    ax.set_ylabel("sampled decode steps", fontsize=14)
    ax.set_xlim(0, 1)
    ax.axvline(df.m1.mean(), color="#eb6834", lw=2,
               label=f"mean = {df.m1.mean():.3f}")
    ax.legend(fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def kvheads(summary_csv: Path, out_path: Path) -> None:
    """Phase K: M1 by KV-head count (model shape) x allocator, steady, bt16."""
    df = pd.read_csv(summary_csv)
    df = df[df.workload == "steady"]
    order = [("mqa-1kv", "MQA\n1 KV head\n512 B/token"),
             ("llama-gqa-8kv", "GQA\n8 KV heads\n4 KB/token"),
             ("mha-32kv", "MHA\n32 KV heads\n16 KB/token")]
    allocs = ["paged", "pim-aware", "contiguous"]
    x = np.arange(len(order)); width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.2))
    _style_axis(ax)
    for k, al in enumerate(allocs):
        vals = [float(df[(df.model == m) & (df.allocator == al)].m1_mean.mean())
                for m, _ in order]
        bars = ax.bar(x + (k - 1) * width, vals, width * 0.92,
                      color=ALLOC_COLOR[al], label=ALLOC_LABEL[al],
                      edgecolor=SURFACE, linewidth=2, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=10, color=INK,
                        xytext=(0, 2), textcoords="offset points")
    labels = []
    for i, (m, lbl) in enumerate(order):
        r = (df[(df.model == m) & (df.allocator == "pim-aware")].m3_gbps_mean.mean()
             / df[(df.model == m) & (df.allocator == "paged")].m3_gbps_mean.mean())
        labels.append(f"{lbl}\nPIM-aware gains {r:.1f}×")
    ax.set_xticks(x, labels)
    ax.set_ylabel("all-bank row-hit rate")
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("The penalty grows as models shed KV heads — 16-token blocks",
                 color=INK, fontsize=14, pad=34)
    ax.legend(fontsize=10.5, framealpha=0, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), ncol=3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def validation(ram_csv: Path, attacc_csv: Path, out_path: Path) -> None:
    """Phase E/E2 cross-validation, stated as two plain questions.

    Left  — does the model get PLACEMENT right? our M1 against stock
            Ramulator 2's own measured row-hit fraction, same sampled decode
            steps, one dot each; dots on the diagonal means agreement.
    Right — does the SPEEDUP we claim survive a cycle-level PIM simulator?
            speedup over the paged baseline on the same three sequences,
            as our M3 predicts it and as AttAcc's PIM_MAC_AB cycles measure
            it. The shortfall is ACT overlap, which our timing omits.
    """
    r = pd.read_csv(ram_csv)
    a = pd.read_csv(attacc_csv)
    allocs = ["paged", "pim-aware", "contiguous"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2),
                                   width_ratios=[3, 2.2])
    for ax in (ax1, ax2):
        _style_axis(ax)

    # -- left: agreement on row-hit rate -------------------------------
    lo = min(r.m1.min(), r.ram_hit_frac.min()) - 0.03
    hi = max(r.m1.max(), r.ram_hit_frac.max()) + 0.02
    ax1.plot([lo, hi], [lo, hi], ls="--", lw=1.4, color=MUTED, zorder=2)
    for al in ("paged", "contiguous", "pim-aware"):   # aligned pair on top
        g = r[r.allocator == al]
        ax1.scatter(g.m1, g.ram_hit_frac, s=120, color=ALLOC_COLOR[al],
                    edgecolor=SURFACE, linewidth=1.8, zorder=4,
                    label=ALLOC_LABEL[al])
    d = (r.m1 - r.ram_hit_frac).abs()
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("our analytical model — row-hit rate")
    ax1.set_ylabel("stock Ramulator 2 — measured row-hit fraction")
    # every number lives in the title; nothing is drawn inside the axes
    ax1.set_title("Placement: the model agrees\n"
                  f"{len(r)} samples · mean |\u0394| {d.mean():.4f} · "
                  f"worst {d.max():.4f}", color=INK, fontsize=11.5, pad=10)
    from matplotlib.lines import Line2D
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([], [], ls="--", lw=1.4, color=MUTED))
    labels.append("perfect agreement")
    ax1.legend(handles, labels, fontsize=10, framealpha=0.92, loc="upper left")

    # -- right: claimed speedup vs measured speedup ---------------------
    base = a[a.allocator == "paged"].set_index("seq_id")
    tgt = [al for al in allocs if al != "paged"]
    model = [float((a[a.allocator == al].set_index("seq_id").m3_gbps
                    / base.m3_gbps).mean()) for al in tgt]
    meas = [float((base.cycles_per_kb
                   / a[a.allocator == al].set_index("seq_id").cycles_per_kb
                   ).mean()) for al in tgt]
    x = np.arange(len(tgt)); w = 0.34
    b1 = ax2.bar(x - w / 2, model, w * 0.94,
                 color=[ALLOC_COLOR[al] for al in tgt], edgecolor=SURFACE,
                 linewidth=2, zorder=3, label="speedup we claim (our model)")
    b2 = ax2.bar(x + w / 2, meas, w * 0.94,
                 color=[ALLOC_COLOR[al] for al in tgt], edgecolor=INK,
                 linewidth=1.2, hatch="//", alpha=0.55, zorder=3,
                 label="speedup AttAcc measures (PIM cycles)")
    for bars in (b1, b2):
        for bb in bars:
            ax2.annotate(f"{bb.get_height():.2f}×",
                         (bb.get_x() + bb.get_width() / 2, bb.get_height()),
                         ha="center", va="bottom", fontsize=11, color=INK,
                         xytext=(0, 2), textcoords="offset points")
    ax2.axhline(1.0, color=MUTED, lw=1.4, ls="--", zorder=2)
    ax2.annotate("paged baseline", (len(tgt) - 0.52, 1.0), va="bottom",
                 ha="right", fontsize=10.5, color=MUTED)
    over = np.mean([m / v for m, v in zip(model, meas)])
    ax2.set_xticks(x, [ALLOC_LABEL[al] for al in tgt])
    ax2.set_xlabel(f"we are {over:.1f}× optimistic: our timing charges a full\n"
                   "row cycle per miss, AttAcc overlaps activation",
                   fontsize=10.5, color=MUTED)
    ax2.set_ylim(0, max(model) * 1.45)
    ax2.set_ylabel("speedup over paged (same three sequences)")
    ax2.set_title("Speedup: claimed vs measured", color=INK, fontsize=13, pad=8)
    ax2.legend(fontsize=9.5, framealpha=0.92, loc="upper right")

    fig.suptitle("Cross-validation: placement confirmed, timing overstated by "
                 f"{over:.1f}×", color=INK, fontsize=14.5, y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def pressure(d_csv: Path, specfix_csv: Path, out_path: Path) -> None:
    """Phase D: M1 vs pool headroom under vLLM admission + preemption."""
    d = pd.read_csv(d_csv); f = pd.read_csv(specfix_csv)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    _style_axis(ax)
    v = d[d.admission == "vllm"]
    series = [("paged", "steady", v[(v.allocator == "paged") & (v.workload == "steady")], "-"),
              ("pim-aware", "steady", v[(v.allocator == "pim-aware") & (v.workload == "steady")], "-"),
              ("paged", "spec", v[(v.allocator == "paged") & (v.workload == "spec")], "--"),
              ("pim-aware", "spec", f[f.admission == "vllm"], "--")]
    for al, wl, s, ls in series:
        s = s.sort_values("headroom")
        ax.plot(s.headroom, s.m1_mean, ls, color=ALLOC_COLOR[al], lw=2,
                marker="o", ms=5, zorder=3)
    pre = v[(v.allocator == "pim-aware") & (v.workload == "steady")].sort_values("headroom")
    hs = list(pre.headroom)
    for h, m, n in zip(pre.headroom, pre.m1_mean, pre.preemptions):
        # keep the end labels inside the axes
        ha = "left" if h == hs[0] else ("right" if h == hs[-1] else "center")
        dx = 6 if ha == "left" else (-6 if ha == "right" else 0)
        ax.annotate(f"{int(n)} preemptions", (h, m), textcoords="offset points",
                    xytext=(dx, -17), ha=ha, fontsize=9.5, color=MUTED)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ALLOC_COLOR[a], lw=2.5, label=ALLOC_LABEL[a])
               for a in ("paged", "pim-aware")]
    handles += [Line2D([], [], color=INK, lw=1.8, ls=s, label=f"{w} workload")
                for w, s in (("steady", "-"), ("spec", "--"))]
    ax.legend(handles=handles, fontsize=10.5, framealpha=0.9, loc="center right")
    ax.axvline(1.0, color=MUTED, lw=1.2, ls=(0, (2, 3)), zorder=1)
    ax.annotate("pool = mean demand", (1.0, 0.71), fontsize=10, color=MUTED,
                xytext=(6, 0), textcoords="offset points")
    ax.set_xlabel("KV pool size ÷ mean demand (headroom)")
    ax.set_ylabel("all-bank row-hit rate")
    ax.set_ylim(0.7, 1.0)
    ax.set_xlim(0.55, 1.38)
    ax.set_title("Under real preemption, the benefit shrinks when the pool is "
                 "oversubscribed", color=INK, fontsize=13.5, pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) == 3 and args[0] == "m1-hist":
        m1_hist(Path(args[1]), Path(args[2]))
        return 0
    if len(args) == 3 and args[0] == "kvheads":
        kvheads(Path(args[1]), Path(args[2])); return 0
    if len(args) == 4 and args[0] == "validation":
        validation(Path(args[1]), Path(args[2]), Path(args[3])); return 0
    if len(args) == 4 and args[0] == "pressure":
        pressure(Path(args[1]), Path(args[2]), Path(args[3])); return 0
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


