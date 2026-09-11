"""CLI entry point. Every reported number is reproducible with one command.

Phase 0 kill test (AGENT_BRIEF §7):

    python -m pimkv.run --workload steady --allocator paged --dram hbm3-pim \
        --addrmap host-centric --block-tokens 16 --requests 2000 --seed 0 \
        --out results/steady_paged.csv

Writes the per-decode-step CSV and a ``<out>.meta.json`` sidecar with the
full configuration and summary; prints the summary to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .addrmap import AddrMap, SCHEMES
from .allocator import ALLOCATORS, make_allocator
from .config import (DEFAULT_TIMING, DRAM_PRESETS, MODEL_PRESETS,
                     derive_block_tokens, shard_kv)
from .sim import SpecParams, auto_pool_blocks, simulate
from .workload import WORKLOADS


def frame_blocks_for(geom, shape, block_tokens: int) -> int:
    """Blocks per pim-aware alignment frame. A frame's linear extent is one
    row index across every bank of every channel (rowgroup_bytes * channels
    — the all-bank alignment quantum); derived, never hardcoded."""
    frame_bytes = geom.rowgroup_bytes * geom.channels
    block_bytes = block_tokens * shape.kv_bytes_per_token
    if block_bytes >= frame_bytes:
        if block_bytes % frame_bytes:
            raise ValueError("block size must be a multiple of the alignment "
                             f"frame ({frame_bytes} B) when larger than it")
        return 1
    if frame_bytes % block_bytes:
        raise ValueError(f"alignment frame ({frame_bytes} B) must be a "
                         f"multiple of block size ({block_bytes} B)")
    return frame_bytes // block_bytes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m pimkv.run",
                                description=__doc__.splitlines()[0])
    p.add_argument("--workload", choices=sorted(WORKLOADS), default="steady")
    p.add_argument("--allocator", choices=sorted(ALLOCATORS), default="paged")
    p.add_argument("--dram", choices=sorted(DRAM_PRESETS), default="hbm3-pim")
    p.add_argument("--addrmap", choices=sorted(SCHEMES), default="host-centric")
    p.add_argument("--model", choices=sorted(MODEL_PRESETS),
                   default="llama-gqa-8kv")
    p.add_argument("--block-tokens", type=int, default=16,
                   help="KV block size in tokens (vLLM default: 16); "
                        "pass 0 to use the PIM-derived size "
                        "(config.derive_block_tokens)")
    p.add_argument("--requests", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--pool-blocks", type=int, default=0,
                   help="KV pool size in blocks (0 = auto-size to ~1.3x "
                        "mean demand at full batch)")
    p.add_argument("--sample-every", type=int, default=16,
                   help="measure metrics every N decode steps")
    p.add_argument("--sample-seqs", type=int, default=8,
                   help="sequences measured per sampled step")
    p.add_argument("--kv-shards", type=int, default=1,
                   help="model KV-head sharding: split the cache into N "
                        "independent all-bank domains, each owning "
                        "channels/N channels and kv_heads/N heads "
                        "(config.shard_kv). 1 = fully interleaved")
    p.add_argument("--spec-width", type=int, default=4,
                   help="spec workload: draft branches per round (W)")
    p.add_argument("--spec-depth", type=int, default=3,
                   help="spec workload: draft tokens per branch (D)")
    p.add_argument("--spec-accept", type=float, default=0.8,
                   help="spec workload: draft i accepted with p**i")
    p.add_argument("--spec-adopt", choices=("auto", "splice", "copyback"),
                   default="auto",
                   help="spec: splice = vLLM pointer-swap of the winning "
                        "branch tail; copyback = copy accepted tokens into "
                        "the sequence's own tail, free all branches")
    p.add_argument("--pim-scratch", choices=("separate", "shared"),
                   default="separate",
                   help="pim-aware: draw spec branch tails from a shared "
                        "scratch domain (Phase B2 fix) or from the "
                        "sequence's own frame (Phase 2 behaviour)")
    p.add_argument("--pim-plan", choices=("bestfit", "greedy"),
                   default="bestfit",
                   help="pim-aware: plan the whole admission into the "
                        "fewest frames (bestfit, Phase D2) or take one "
                        "frame at a time (greedy, pre-fix)")
    p.add_argument("--pim-compact", type=float, default=0.0,
                   help="pim-aware: compact frames whose occupancy "
                        "falls below this fraction (0 = off); cost is "
                        "reported as blocks_copied")
    p.add_argument("--admission", choices=("oracle", "vllm"),
                   default="oracle",
                   help="oracle = reserve final footprint, no preemption "
                        "(Phases 0-2); vllm = v0.2.7 scheduler: watermark "
                        "admission + preempt-by-recompute (Phase D)")
    p.add_argument("--coalesce", choices=("window", "inorder"),
                   default="window",
                   help="PIM controller model: reorder within a window "
                        "(default) or strictly in-order")
    p.add_argument("--window", type=int, default=64,
                   help="controller reorder window, bursts per channel")
    p.add_argument("--out", type=Path, default=None,
                   help="CSV output path (metrics, one row per sampled "
                        "decode step per sequence)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    geom, shape = shard_kv(DRAM_PRESETS[args.dram],
                           MODEL_PRESETS[args.model], args.kv_shards)
    block_tokens = args.block_tokens or derive_block_tokens(geom, shape)
    am = AddrMap(geom, args.addrmap)
    requests = WORKLOADS[args.workload](args.requests, args.seed)
    pool_blocks = args.pool_blocks or auto_pool_blocks(
        requests, block_tokens, args.max_batch)
    # geometry property: used for placement only by pim-aware, but every
    # policy is diagnosed against it
    fb = frame_blocks_for(geom, shape, block_tokens)
    alloc = make_allocator(args.allocator, pool_blocks, args.seed,
                           frame_blocks=fb if args.allocator == "pim-aware"
                           else 1, pim_scratch=args.pim_scratch,
                           pim_plan=args.pim_plan,
                           pim_compact=args.pim_compact)
    adopt = args.spec_adopt
    if adopt == "auto":   # B2: pim-aware needs copy-back; paged stays vLLM-faithful
        adopt = "copyback" if args.allocator == "pim-aware" else "splice"
    spec = (SpecParams(width=args.spec_width, depth=args.spec_depth,
                       accept=args.spec_accept, adopt=adopt)
            if args.workload == "spec" else None)

    res = simulate(requests, alloc, geom, shape, am,
                   block_tokens=block_tokens, max_batch=args.max_batch,
                   sample_every=args.sample_every,
                   sample_seqs=args.sample_seqs, window=args.window,
                   mode=args.coalesce, timing=DEFAULT_TIMING, seed=args.seed,
                   frame_blocks=fb, spec=spec, admission=args.admission)

    config = dict(vars(args), out=str(args.out) if args.out else None,
                  block_tokens_effective=block_tokens,
                  pool_blocks_effective=alloc.num_blocks,
                  dram_geometry=geom.name, rowgroup_bytes=geom.rowgroup_bytes,
                  kv_bytes_per_token=shape.kv_bytes_per_token,
                  timing_tccd_ab_ns=DEFAULT_TIMING.tccd_ab_ns,
                  timing_trc_ns=DEFAULT_TIMING.trc_ns)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        res.df.to_csv(args.out, index=False, float_format="%.8g")
        meta = dict(config=config,
                    summary={k: v for k, v in res.summary.items()
                             if k != "runtime_s"})
        args.out.with_suffix(args.out.suffix + ".meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n")

    s = res.summary
    print(f"# pimkv {args.workload}/{args.allocator}/{args.dram}/"
          f"{args.addrmap} block_tokens={block_tokens} "
          f"requests={args.requests} seed={args.seed}")
    print(f"completed={s['completed']}  dropped={s['dropped']}  "
          f"frag_failures={s['frag_failures']}  "
          f"decode_steps={s['decode_steps']}  samples={s['samples']}  "
          f"pool={s['pool_blocks']} blocks "
          f"(peak util {s['pool_utilization_peak']:.2f})")
    print(f"M1 row-hit rate      mean={s['m1_mean']:.4f}  "
          f"p05={s['m1_p05']:.4f}  p95={s['m1_p95']:.4f}")
    print(f"M2 bank parallelism  mean={s['m2_mean']:.4f}  "
          f"p05={s['m2_p05']:.4f}  p95={s['m2_p95']:.4f}")
    print(f"M3 effective BW      mean={s['m3_gbps_mean']:.1f} GB/s")
    if s["preemptions"]:
        print(f"preemption           preemptions={s['preemptions']}  "
              f"self={s['self_preemptions']}  dropped={s['dropped']}")
    if s["spec_stalls"] or s["copied_bytes"] or s["cow_copies"]:
        print(f"spec/CoW             cow_copies={s['cow_copies']}  "
              f"spec_stalls={s['spec_stalls']}  "
              f"copied={s['copied_bytes']/1e6:.1f} MB  "
              f"phys_allocs={s['blocks_allocated_total']}")
    print(f"placement            blk_adj={s['blk_adj_mean']:.3f}  "
          f"same_frame={s['same_frame_mean']:.3f}  "
          f"frame_spread={s['frame_spread_mean']:.2f}  (frame={fb} blocks)")
    print(f"M4 decode latency    mean={s['m4_ns_mean']:.0f} ns  "
          f"p95={s['m4_ns_p95']:.0f} ns")
    print(f"runtime {s['runtime_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
