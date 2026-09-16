"""Figures for the allocator CPU-cost measurement.

    .venv/bin/python scripts/plot_bench_alloc.py \
        results/bench_alloc/summary.csv results/headline/summary.csv \
        figures/alloc_cost.png

Reads only CSVs written by scripts/bench_alloc.py and pimkv.sweep — nothing
is fabricated. Timing-harness overhead (measured in the same run) is
subtracted here; the raw columns stay in the CSV. Palette follows
pimkv.plots.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
COLOR = {"paged": "#2a78d6", "pim-aware": "#eb6834", "contiguous": "#1baf7a"}
LABEL = {"paged": "paged (vLLM port)", "pim-aware": "KV-PIMple",
         "contiguous": "contiguous (knows lengths)"}
LAYERS = 32                     # Llama-3-8B-class model
# mean concurrent sequences = total generated tokens / decode steps, per
# workload (so the comparison uses each workload's real occupancy, not 64)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def corrected(bench_csv: Path) -> pd.DataFrame:
    """Subtract the measured per-call harness overhead."""
    df = pd.read_csv(bench_csv)
    oh_ms = df.harness_overhead_ns.iloc[0] / 1e6
    s = df[df.part == "in-situ"].copy()
    s["cpu_ms"] = s.alloc_ms_total - s.calls * oh_ms
    s["us_per_step"] = s.cpu_ms * 1000 / s.decode_steps
    s["ns_per_alloc"] = s.cpu_ms * 1e6 / s.blocks_allocated
    return s, df[df.part == "worst-case"].copy(), df.harness_overhead_ns.iloc[0]


def panel_per_step(ax, s):
    cases = [("steady", "oracle", "steady\n(1.3× pool)"),
             ("steady", "vllm", "oversubscribed\n(0.6× pool)"),
             ("spec", "oracle", "spec decode\n(14× allocations)")]
    allocs = ["paged", "pim-aware", "contiguous"]
    w = 0.26
    for j, al in enumerate(allocs):
        xs, ys = [], []
        for i, (wl, adm, _) in enumerate(cases):
            r = s[(s.workload == wl) & (s.admission == adm) & (s.allocator == al)]
            if len(r):
                xs.append(i + (j - 1) * w)
                ys.append(float(r.us_per_step.iloc[0]))
        b = ax.bar(xs, ys, w, color=COLOR[al], label=LABEL[al], zorder=3)
        ax.bar_label(b, fmt="%.0f", fontsize=9.5, color=INK, padding=2)
    ax.set_xticks(range(len(cases)), [c[2] for c in cases], fontsize=10)
    ax.set_ylabel("allocator CPU per decode step (µs)")
    ax.set_title("What the allocator costs the CPU", loc="left", fontsize=12.5,
                 color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style(ax)


def workload_trade(s, extra, head_csv: Path):
    """Per workload: attention ms/step for each allocator (from the headline
    sweep, scaled by that workload's mean concurrency) and the measured
    allocator CPU. Returns a list of dicts."""
    from pimkv.workload import WORKLOADS
    h = pd.read_csv(head_csv)
    h = h[(h.block_tokens == 16) & (h.addrmap == "host-cacheline")]

    def cpu(wl, al):
        r = s[(s.workload == wl) & (s.admission == "oracle")
              & (s.allocator == al)]
        if len(r):
            return float(r.us_per_step.iloc[0])
        r = extra[(extra.workload == wl) & (extra.allocator == al)]
        return float(r.us_per_step.iloc[0])

    out = []
    for wl, nreq in (("steady", 1000), ("spec", 1000), ("prefix", 300),
                     ("longctx", 300)):
        hh = h[h.workload == wl]
        if not len(hh):
            continue
        steps = float(hh[hh.allocator == "paged"].decode_steps.iloc[0])
        active = sum(r.output_len for r in WORKLOADS[wl](nreq, 0)) / steps
        att = {al: float(hh[hh.allocator == al].m4_ns_mean.iloc[0])
               * LAYERS * active / 1e6 for al in ("paged", "pim-aware")}
        out.append(dict(workload=wl, active=active, att=att,
                        d_cpu_us=cpu(wl, "pim-aware") - cpu(wl, "paged"),
                        saved_ms=att["paged"] - att["pim-aware"]))
    return out


def panel_budget(ax, trade, s):
    tr = next(t for t in trade if t["workload"] == "steady")
    rows = []
    for al in ("paged", "pim-aware"):
        cpu = float(s[(s.workload == "steady") & (s.admission == "oracle")
                      & (s.allocator == al)].us_per_step.iloc[0]) / 1e3
        rows.append((al, tr["att"][al], cpu))
    for i, (al, att, cpu) in enumerate(rows):
        ax.barh(i, att, color=COLOR[al], zorder=3)
        ax.barh(i, cpu, left=att, color=INK, zorder=3)
        ax.text(att + 0.06, i, f"{att:.2f} ms attention   +{cpu * 1e3:.0f} µs allocator",
                va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)), [LABEL[r[0]] for r in rows], fontsize=10)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.65)
    ax.set_xlabel(f"time per decode step (ms) — steady workload, "
                  f"{tr['active']:.0f} concurrent sequences, {LAYERS} layers")
    ax.set_title(f"Steady: +{(rows[1][2] - rows[0][2]) * 1e3:.0f} µs of CPU, "
                 f"−{tr['saved_ms']:.1f} ms of memory time",
                 loc="left", fontsize=12.5, color=INK)
    ax.invert_yaxis()
    _style(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.grid(False, axis="y")


def panel_worst(ax, w):
    ops = [("ns_per_block_alloc", "grow a sequence\n(append_block)"),
           ("ns_per_admit", "admit a request\n(+ region planning)")]
    wdt = 0.34
    for j, al in enumerate(("paged", "pim-aware")):
        r = w[w.allocator == al]
        ys = [float(r[c].iloc[0]) for c, _ in ops]
        b = ax.bar([i + (j - 0.5) * wdt for i in range(len(ops))], ys, wdt,
                   color=COLOR[al], label=LABEL[al], zorder=3)
        ax.bar_label(b, fmt="%.0f", fontsize=9.5, color=INK, padding=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ops)), [o[1] for o in ops], fontsize=10)
    ax.set_ylabel("ns per operation (log)")
    ax.set_title("Worst case: pool full, no empty region left",
                 loc="left", fontsize=12.5, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style(ax)


def panel_ratio(ax, trade):
    names = {"steady": "steady", "spec": "spec\ndecode",
             "prefix": "prefix\nsharing", "longctx": "long context\n(8k–32k)"}
    xs = range(len(trade))
    ratios = [t["saved_ms"] * 1000 / t["d_cpu_us"] for t in trade]
    b = ax.bar(xs, ratios, 0.55, color=COLOR["pim-aware"], zorder=3)
    ax.bar_label(b, labels=[f"{r:.0f}×" for r in ratios], fontsize=10,
                 color=INK, padding=2)
    for x, tr, r in zip(xs, trade, ratios):
        ax.text(x, r * 1.45, f"saves {tr['saved_ms']:.2f} ms\n"
                f"costs {tr['d_cpu_us']:.0f} µs", ha="center", va="bottom",
                fontsize=8.5, color=MUTED)
    ax.axhline(1, color=INK, lw=1)
    ax.set_yscale("log")
    ax.set_ylim(1, max(ratios) * 6)
    ax.set_xticks(list(xs), [names[t["workload"]] for t in trade], fontsize=10)
    ax.set_ylabel("memory time saved ÷ CPU spent (log)")
    ax.set_title("The trade is workload-dependent", loc="left", fontsize=12.5,
                 color=INK)
    _style(ax)


def main(argv) -> int:
    bench, head, out = (Path(argv[0]), Path(argv[1]), Path(argv[2]))
    s, w, oh = corrected(bench)
    extra = pd.read_csv(Path(argv[0]).parent / "extra_workloads.csv")
    trade = workload_trade(s, extra, Path(argv[1]))
    fig = plt.figure(figsize=(13, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.46,
                          wspace=0.3, left=0.115, right=0.97, top=0.9,
                          bottom=0.105)
    panel_per_step(fig.add_subplot(gs[0, 0]), s)
    panel_worst(fig.add_subplot(gs[0, 1]), w)
    panel_budget(fig.add_subplot(gs[1, 0]), trade, s)
    panel_ratio(fig.add_subplot(gs[1, 1]), trade)
    fig.suptitle("Cost of KV-PIMple allocation — measured, CPython, "
                 f"harness overhead ({oh:.0f} ns/call) subtracted",
                 fontsize=14, color=INK, x=0.06, ha="left", y=0.965)
    fig.text(0.115, 0.014, "attention time = measured decode latency × 32 layers "
             "× that workload's own mean concurrency (not a fixed batch)",
             fontsize=8.5, color=MUTED, ha="left", va="bottom")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
