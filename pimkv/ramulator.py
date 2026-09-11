"""Phase E — Ramulator 2 cross-validation of the analytical model.

    python -m pimkv.ramulator --workload steady --block-tokens 16 \
        --allocators paged,pim-aware,contiguous --samples 3 --out results/phaseE/

What is validated and what is not
---------------------------------
Stock Ramulator 2 has no all-bank PIM command; it is a conventional DRAM
timing simulator. So this cross-validates the ROW-LOCALITY half of the
model (M1 and the tRC-vs-tCCD timing factor in M3) against a cycle-level
controller with its own FR-FCFS reordering, on the SAME per-burst address
streams the analytical model scores. It cannot validate M2's lockstep
all-bank semantics. Comparison is of direction and rough magnitude
(AGENT_BRIEF §5.5): if Ramulator's cycle deltas between allocators disagree
in SIGN with M3's, the analytical model is wrong.

Mechanics
---------
1. Run the pimkv simulator with a sampling hook and capture real block
   tables (sequence, table) at a chosen decode step.
2. Enumerate the sequence's KV bursts exactly as pimmodel does and map
   them through pimkv.addrmap. Emit Ramulator's ReadWriteTrace format with
   PRE-DECODED address vectors (``R ch,pc,sid,bg,ba,row,col``) paired with
   PassThroughAddrMapper/PassThroughChannelMapper, so Ramulator sees our
   address map bit-for-bit — no mapper reconciliation. Our 32 all-bank
   domains = 16 HBM3 channels x 2 pseudo-channels (org HBM3_4Gb: 4 bank
   groups x 4 banks, 16384 rows, 1 KB row per PC, 32 B per BL8 access —
   identical geometry to the hbm3-pim preset). Column is in HBM3 column-
   address units (8 per 32 B burst).
3. Run Ramulator through its Python bindings (third_party/ramulator2/
   python, nanobind module built by CMake) and collect row hits / misses /
   conflicts and cycles across controllers.

Build note: third_party/ramulator2 needed a one-line patch for Apple
clang 21 (``config[name].template as<T>()`` in src/ramulator/base/param.h)
and must be configured with -DPython_EXECUTABLE=<this venv's python>; see
docs/NOTES.md 2026-09-11.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .addrmap import AddrMap, MappedAddrs
from .allocator import make_allocator
from .config import DEFAULT_TIMING, DRAM_PRESETS, MODEL_PRESETS
from .run import frame_blocks_for
from .sim import SpecParams, auto_pool_blocks, measure_seq, simulate
from .workload import WORKLOADS

RAMULATOR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "ramulator2"
HBM3_PSEUDOCHANNELS = 2          # per HBM3 channel (org preset HBM3_4Gb)
HBM3_COLS_PER_BURST = 8          # BL8: column address units per 32 B access


def import_ramulator():
    sys.path.insert(0, str(RAMULATOR_ROOT / "python"))
    try:
        import ramulator                       # noqa: F401
        from ramulator import _ramulator       # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"Ramulator 2 Python bindings not importable ({e}). Build them:\n"
            f"  cd {RAMULATOR_ROOT} && mkdir -p build-py && cd build-py && "
            f"env -u CXXFLAGS -u CFLAGS -u LDFLAGS cmake .. "
            f"-DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE={sys.executable} "
            f"&& make -j6") from e
    return ramulator


def enumerate_bursts(table: list[int], length: int, geom, shape,
                     block_tokens: int, am: AddrMap) -> MappedAddrs:
    """Same enumeration pimmodel scores (sim.measure_seq)."""
    bpt = shape.kv_bytes_per_token // geom.burst_bytes
    bpb = block_tokens * bpt
    lb = np.arange(length * bpt, dtype=np.int64)
    blocks = np.asarray(table, dtype=np.int64)
    return am.map(blocks[lb // bpb] * bpb + (lb % bpb))


def emit_trace(m: MappedAddrs, path: Path) -> int:
    """Write a ReadWriteTrace file; returns the number of requests."""
    ch = m.ch // HBM3_PSEUDOCHANNELS
    pc = m.ch % HBM3_PSEUDOCHANNELS
    col = m.co * HBM3_COLS_PER_BURST
    with open(path, "w") as f:
        for a, b, g, k, r, c in zip(ch.tolist(), pc.tolist(), m.bg.tolist(),
                                    m.ba.tolist(), m.ro.tolist(), col.tolist()):
            f.write(f"R {a},{b},0,{g},{k},{r},{c}\n")
    return int(m.ch.size)


def _collect(stats, keys=("row_hits", "row_misses", "row_conflicts",
                          "cycles", "num_read_reqs")) -> dict:
    """Sum row stats across controllers (max for cycles); tolerant of
    either a single controller dict or a list."""
    out = {k: 0 for k in keys}
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for k in keys:
                if k in node and isinstance(node[k], (int, float)):
                    found.add(k)
                    if k == "cycles":
                        out[k] = max(out[k], node[k])
                    else:
                        out[k] += node[k]
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
    walk(stats)
    out["_found"] = sorted(found)
    return out


def run_ramulator(trace_path: Path, n_channels: int,
                  mem_clock_ratio: int = 64) -> dict:
    """``mem_clock_ratio``: frontend clock ticks per memory clock tick.
    In Ramulator 2 a component's ``clock_ratio`` is its frequency relative
    to the base clock (LARGER = FASTER; the stock example runs the CPU
    frontend at 8 and DRAM at 3). The trace frontend issues at most ONE
    request per frontend tick, so at 1:1 sixteen channels never saturate
    and 'cycles' just counts issues (cycles/burst == 1.0, observed; with
    the ratio applied the wrong way round, == 64.0, also observed). With
    the frontend ``mem_clock_ratio`` times faster than the memory, up to
    that many requests arrive per memory cycle, controller queues fill,
    send() back-pressures, and the memory cycle count becomes
    DRAM-throughput-limited — the quantity to compare across allocators."""
    rm = import_ramulator()
    fe = rm.frontend.ReadWriteTrace(clock_ratio=mem_clock_ratio,
                                    path=str(trace_path))
    ctrls = []
    for _ in range(n_channels):
        dram = rm.dram.HBM3(org_preset="HBM3_4Gb",
                            timing_preset="HBM3_6400Mbps")
        ctrls.append(rm.controller.HBM34(
            dram=dram, scheduler=rm.scheduler.FRFCFS(),
            refresh_manager=rm.refresh_manager.NoRefresh(),
            row_policy=rm.row_policy.Open(),
            addr_mapper=rm.addr_mapper.PassThroughAddrMapper()))
    mem = rm.memory_system.GenericDRAM(
        clock_ratio=1, controllers=ctrls,
        channel_mapper=rm.channel_mapper.PassThroughChannelMapper())
    sim = rm.Simulation(fe, mem)
    t0 = time.perf_counter()
    sim.run()
    stats = sim.stats
    out = _collect(stats)
    out["wall_s"] = time.perf_counter() - t0
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m pimkv.ramulator")
    p.add_argument("--workload", default="steady")
    p.add_argument("--allocators", default="paged,pim-aware,contiguous")
    p.add_argument("--addrmap", default="host-cacheline")
    p.add_argument("--block-tokens", type=int, default=16)
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--samples", type=int, default=3,
                   help="sequences captured per allocator")
    p.add_argument("--min-step", type=int, default=400,
                   help="capture at the first sampled step >= this (lets "
                        "the free list churn first)")
    p.add_argument("--spec-adopt", default="copyback")
    p.add_argument("--mem-clock-ratio", type=int, default=64,
                   help="memory ticks per frontend tick; >16 makes the run "
                        "DRAM-bound so cycles measure throughput")
    p.add_argument("--out", type=Path, default=Path("results/phaseE"))
    args = p.parse_args(argv)

    geom = DRAM_PRESETS["hbm3-pim"]
    shape = MODEL_PRESETS["llama-gqa-8kv"]
    am = AddrMap(geom, args.addrmap)
    bt = args.block_tokens
    args.out.mkdir(parents=True, exist_ok=True)
    n_channels = geom.channels // HBM3_PSEUDOCHANNELS
    rows = []
    for al in args.allocators.split(","):
        requests = WORKLOADS[args.workload](args.requests, args.seed)
        pool = auto_pool_blocks(requests, bt, args.max_batch)
        fb = frame_blocks_for(geom, shape, bt)
        alloc = make_allocator(al, pool, args.seed,
                               frame_blocks=fb if al == "pim-aware" else 1)
        spec = (SpecParams(adopt=args.spec_adopt)
                if args.workload == "spec" else None)
        captured: list[tuple[int, int, int, list[int]]] = []

        def hook(step, s, table, _c=captured):
            if step >= args.min_step and len(_c) < args.samples:
                _c.append((step, s.rid, s.length, table))

        simulate(requests, alloc, geom, shape, am, block_tokens=bt,
                 max_batch=args.max_batch, seed=args.seed, frame_blocks=fb,
                 spec=spec, on_sample=hook)
        for step, rid, length, table in captured:
            m = enumerate_bursts(table, length, geom, shape, bt, am)
            trace = args.out / f"{args.workload}_{al}_bt{bt}_seq{rid}_t{step}.trace"
            n = emit_trace(m, trace)
            ours = measure_seq(type("S", (), dict(rid=rid, length=length))(),
                               table, am, geom, shape, DEFAULT_TIMING, bt,
                               64, "window", fb)
            r = run_ramulator(trace, n_channels, args.mem_clock_ratio)
            acc = r["row_hits"] + r["row_misses"] + r["row_conflicts"]
            row = dict(workload=args.workload, allocator=al, block_tokens=bt,
                       seq_id=rid, step=step, seq_len=length, n_bursts=n,
                       m1=ours["m1"], m2=ours["m2"], m3_gbps=ours["m3_gbps"],
                       ram_row_hits=r["row_hits"], ram_row_misses=r["row_misses"],
                       ram_row_conflicts=r["row_conflicts"],
                       ram_hit_frac=(r["row_hits"] / acc if acc else float("nan")),
                       ram_cycles=r["cycles"],
                       ram_cycles_per_burst=(r["cycles"] / n if n else float("nan")),
                       ram_stats_found=";".join(r["_found"]), wall_s=r["wall_s"],
                       mem_clock_ratio=args.mem_clock_ratio)
            rows.append(row)
            print(f"  {al:11s} seq {rid:5d} len {length:5d}  ours M1 {ours['m1']:.3f}"
                  f"  ram hit {row['ram_hit_frac']:.3f}  cyc/burst "
                  f"{row['ram_cycles_per_burst']:.2f}  ({r['wall_s']:.0f}s)",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "summary.csv", index=False, float_format="%.8g")
    print(df.groupby("allocator")[["m1", "ram_hit_frac", "m3_gbps",
                                   "ram_cycles_per_burst"]].mean()
          .to_string(float_format=lambda x: f"{x:,.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
