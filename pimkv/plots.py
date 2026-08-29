"""Figure generation (Phase 2/4 — AGENT_BRIEF §9).

Planned figures:
  headline.png      row-hit rate vs block size, three allocator policies,
                    spec workload curve overlaid (the money shot)
  bandwidth.png     effective PIM bandwidth: baseline vs pim-aware vs ideal,
                    per workload
  sweep_heatmap.png M1 over (block size x context length)

Only a minimal M1-distribution plot exists so far; the real figures land in
Phase 2 once the pim-aware allocator and the sweep runner exist. Nothing in
this module fabricates data: every plot reads a results CSV produced by
pimkv.run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def m1_hist(csv_path: Path, out_path: Path) -> None:
    """Histogram of per-decode-step M1 from one run CSV (Phase 0 check)."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df.m1, bins=50, color="#4477aa", edgecolor="white")
    ax.set_xlabel("M1 — all-bank row-hit rate", fontsize=14)
    ax.set_ylabel("sampled decode steps", fontsize=14)
    ax.set_xlim(0, 1)
    ax.axvline(df.m1.mean(), color="#cc3311", lw=2,
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
    print("usage: python -m pimkv.plots m1-hist <run.csv> <out.png>\n"
          "(headline/bandwidth/heatmap figures are Phase 2)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
