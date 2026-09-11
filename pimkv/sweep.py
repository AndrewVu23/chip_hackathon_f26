"""Sweep runner: run a YAML-defined grid of configurations, one CSV +
meta.json per cell, plus an aggregated ``summary.csv`` for the figures.

    python -m pimkv.sweep --config configs/headline.yaml --out results/headline/ [--jobs 4]

Config format (see configs/headline.yaml): ``common`` holds defaults, each
entry in ``cells`` is a cartesian product over its list-valued keys
(workloads x allocators x block_tokens x seeds) with scalar overrides of
``common``. Every run is independently seeded and written to its own file,
so results are byte-identical regardless of --jobs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import yaml

from .addrmap import AddrMap
from .allocator import make_allocator
from .config import DEFAULT_TIMING, DRAM_PRESETS, MODEL_PRESETS
from .run import frame_blocks_for
from .sim import SpecParams, auto_pool_blocks, simulate
from .workload import WORKLOADS


def _run_cell(job: dict) -> dict:
    """Worker: one (workload, allocator, block_tokens, seed) cell."""
    out_dir = Path(job["out_dir"])
    wkw = job.get("workload_kwargs") or {}
    tag = "".join(f"_{k[:2]}{v}" for k, v in sorted(wkw.items()))
    name = (f"{job['workload']}_{job['allocator']}_{job['addrmap']}"
            f"_bt{job['block_tokens']}{tag}_s{job['seed']}")
    csv_path = out_dir / f"{name}.csv"
    geom = DRAM_PRESETS[job["dram"]]
    shape = MODEL_PRESETS[job["model"]]
    bt = job["block_tokens"]
    am = AddrMap(geom, job["addrmap"])
    requests = WORKLOADS[job["workload"]](job["requests"], job["seed"], **wkw)
    pool = job.get("pool_blocks") or auto_pool_blocks(requests, bt,
                                                      job["max_batch"])
    fb = (frame_blocks_for(geom, shape, bt)
          if job["allocator"] == "pim-aware" else 1)
    alloc = make_allocator(job["allocator"], pool, job["seed"],
                           frame_blocks=fb)
    spec = SpecParams() if job["workload"] == "spec" else None
    res = simulate(requests, alloc, geom, shape, am, block_tokens=bt,
                   max_batch=job["max_batch"],
                   sample_every=job["sample_every"],
                   sample_seqs=job["sample_seqs"], window=job["window"],
                   mode=job["coalesce"], timing=DEFAULT_TIMING,
                   seed=job["seed"], spec=spec)
    res.df.to_csv(csv_path, index=False, float_format="%.8g")
    meta = dict(config={k: v for k, v in job.items() if k != "out_dir"},
                summary={k: v for k, v in res.summary.items()
                         if k != "runtime_s"})
    csv_path.with_suffix(".csv.meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n")
    row = dict(workload=job["workload"], allocator=job["allocator"],
               addrmap=job["addrmap"], block_tokens=bt, seed=job["seed"],
               csv=str(csv_path), **{k: v for k, v in wkw.items()})
    row.update({k: v for k, v in res.summary.items() if k != "runtime_s"})
    print(f"  done {name}: M1 {res.summary['m1_mean']:.3f}  "
          f"M3 {res.summary['m3_gbps_mean']:.0f} GB/s  "
          f"({res.summary['runtime_s']:.0f}s)", flush=True)
    return row


DEFAULTS = dict(dram="hbm3-pim", addrmap="host-cacheline",
                model="llama-gqa-8kv", requests=1000, max_batch=64,
                sample_every=16, sample_seqs=8, window=64,
                coalesce="window", pool_blocks=0)


def expand(config: dict, out_dir: Path) -> list[dict]:
    common = dict(DEFAULTS, **config.get("common", {}))
    jobs = []
    for cell in config["cells"]:
        base = dict(common, **{k: v for k, v in cell.items()
                               if k not in ("workloads", "allocators",
                                            "block_tokens", "seeds",
                                            "prompt_lens")})
        for wl, al, bt, seed, pl in itertools.product(
                cell["workloads"], cell["allocators"],
                cell.get("block_tokens", [16]), cell.get("seeds", [0]),
                cell.get("prompt_lens", [None])):
            job = dict(base, workload=wl, allocator=al,
                       block_tokens=bt, seed=seed, out_dir=str(out_dir))
            if pl is not None:
                job["workload_kwargs"] = {"prompt_len": pl}
            jobs.append(job)
    return jobs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m pimkv.sweep")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = expand(config, args.out)
    print(f"sweep: {len(jobs)} runs -> {args.out} (jobs={args.jobs})",
          flush=True)
    ctx = get_context("spawn")
    with ctx.Pool(args.jobs) as pool:
        rows = pool.map(_run_cell, jobs)
    df = pd.DataFrame(rows).sort_values(
        ["workload", "allocator", "addrmap", "block_tokens", "seed"])
    df.to_csv(args.out / "summary.csv", index=False, float_format="%.8g")
    print(f"summary: {args.out / 'summary.csv'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
