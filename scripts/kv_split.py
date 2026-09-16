"""K/V tensor layout sensitivity: combined vs split KV tensors.

    .venv/bin/python scripts/kv_split.py --out results/kv_split

We model one combined K+V tensor (4 KB/token -> 64 KB block -> 8 blocks per
region). Real vLLM v0 keeps K and V in SEPARATE tensors, so each tensor sees
half the bytes per token (2 KB/token -> 32 KB block -> 16 blocks per region).
This runs the headline configuration under both layouts; the combined rows
reproduce the reported headline numbers, which is the control.

Measurement only — imports pimkv, changes nothing in it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pimkv.addrmap import AddrMap
from pimkv.allocator import make_allocator
from pimkv.config import DRAM_PRESETS, ModelShape
from pimkv.run import frame_blocks_for
from pimkv.sim import auto_pool_blocks, simulate
from pimkv.workload import WORKLOADS

LAYOUTS = [("combined K+V", "one tensor, 4 KB/token", 8),
           ("split K | V", "per tensor, 2 KB/token", 4)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kv_split")
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-tokens", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("results/kv_split"))
    args = p.parse_args(argv)

    geom = DRAM_PRESETS["hbm3-pim"]
    am = AddrMap(geom, "host-cacheline")
    rows = []
    for layout, detail, kv_heads in LAYOUTS:
        shape = ModelShape(layout, kv_heads=kv_heads, head_dim=128)
        fb = frame_blocks_for(geom, shape, args.block_tokens)
        for al in ("paged", "pim-aware"):
            reqs = WORKLOADS["steady"](args.requests, args.seed)
            pool = auto_pool_blocks(reqs, args.block_tokens, 64)
            alloc = make_allocator(al, pool, args.seed,
                                   frame_blocks=fb if al == "pim-aware" else 1)
            res = simulate(reqs, alloc, geom, shape, am,
                           block_tokens=args.block_tokens, max_batch=64,
                           seed=args.seed, frame_blocks=fb)
            s = res.summary
            rows.append(dict(layout=layout, detail=detail, allocator=al,
                             bytes_per_token=shape.kv_bytes_per_token,
                             block_kib=args.block_tokens
                             * shape.kv_bytes_per_token // 1024,
                             blocks_per_region=fb,
                             m1_mean=s["m1_mean"], m2_mean=s["m2_mean"],
                             m3_gbps_mean=s["m3_gbps_mean"],
                             frame_spread_mean=s["frame_spread_mean"]))
            print(f"{layout:13s} {al:10s} block {rows[-1]['block_kib']:3d} KiB  "
                  f"{fb:2d} blocks/region  row-hit {s['m1_mean']:.4f}  "
                  f"BW {s['m3_gbps_mean']:7.1f} GB/s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False,
                              float_format="%.6g")
    print("wrote", args.out / "summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
