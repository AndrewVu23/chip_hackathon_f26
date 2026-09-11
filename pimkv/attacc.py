"""Phase E2 — all-bank (M2) validation against AttAcc's PIM Ramulator 2.

    python -m pimkv.attacc --workload steady --block-tokens 16 \
        --allocators paged,pim-aware,contiguous --samples 3 --out results/phaseE2/

Why this exists
---------------
Phase E used *stock* Ramulator 2, which has no all-bank PIM command: its 16
banks run independently under FR-FCFS, so it can validate the row-locality
half of the model (M1) but not M2, the lockstep bank-parallelism term.

AttAcc (ASPLOS'24, scale-snu/attacc_simulator) ships a Ramulator 2 extension
that DOES implement all-bank PIM MAC — ``HBM3-PIM.cpp``, a PIM controller,
a PIM scheduler and the ``PIM_MAC_AB`` request type. Its geometry with
``channel: 16`` is identical to our ``hbm3-pim`` preset: 16 channels x 2
pseudo-channels = 32 all-bank domains, 4 bank groups x 4 banks = 16 banks
each, 16384 rows, 32 columns x 32 B = 1 KB rows.

So we replay OUR allocator's placements as real all-bank commands:

1. capture real block tables from the simulator;
2. enumerate the sequence's KV bursts and map them through pimkv.addrmap;
3. run pimmodel's coalescer in *stream* mode to get the actual all-bank
   command sequence it would issue (row + column per command, per channel);
4. emit one ``PIM_MAC_AB`` per command, round-robin across channels so the
   32 per-channel controllers run in parallel as they would in hardware;
5. run AttAcc's ramulator2 and read ``memory_system_cycles``.

Cycle count then reflects BOTH effects at once: a scattered placement needs
MORE all-bank commands for the same KV bytes (that is M2) and pays ACT/PRE
on row changes between them (that is M1). Comparing cycles across
allocators therefore tests M3 end-to-end.

Address layout (from hbm3_pim_linear_mappers.cpp, HBM3PIMMap): the byte
address is shifted right by log2(tx_bytes)=5 and then sliced from the LSB
as Co(5) | Ro(14) | Ba(2) | BG(2) | Ra(1) | Pch(1) | Ch(4). Our channel
index c maps to (Ch=c>>1, Pch=c&1) and our bank index b to (BG=b>>2,
Ba=b&3); rank is always 0. Banks are immaterial to an all-bank command but
are encoded faithfully anyway.

Build: third_party/attacc_simulator (pinned c600051) + Ramulator 2 at
b7c7027 with AttAcc's 21 patches; needs the same Apple-clang dependent
template fix as the modern tree. See third_party/README.md.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .addrmap import AddrMap
from .allocator import make_allocator
from .config import DEFAULT_TIMING, DRAM_PRESETS, MODEL_PRESETS
from .pimmodel import coalesce_channel_stream
from .ramulator import enumerate_bursts
from .run import frame_blocks_for
from .sim import SpecParams, auto_pool_blocks, measure_seq, simulate
from .workload import WORKLOADS

ATTACC_ROOT = (Path(__file__).resolve().parents[1] / "third_party"
               / "attacc_simulator" / "ramulator2")
BIN = ATTACC_ROOT / "build" / "ramulator2"

# HBM3_8Gb_2R with channel:16 -> level counts (Ch, Pch, Ra, BG, Ba, Ro, Co)
BITS = dict(ch=4, pch=1, ra=1, bg=2, ba=2, ro=14, co=5)
SH_CO = 0
SH_RO = BITS["co"]
SH_BA = SH_RO + BITS["ro"]
SH_BG = SH_BA + BITS["ba"]
SH_RA = SH_BG + BITS["bg"]
SH_PCH = SH_RA + BITS["ra"]
SH_CH = SH_PCH + BITS["pch"]
TX_OFFSET = 5                      # log2(prefetch 8 * channel_width 32 / 8)

CONFIG = """Frontend:
  impl: PIMLoadStoreTrace
  path: {trace}
  clock_ratio: {fe_ratio}

  Translation:
    impl: NoTranslation
    max_addr: 68719476736

MemorySystem:
  impl: PIMDRAM
  clock_ratio: 1
  DRAM:
    impl: HBM3-PIM
    org:
      preset: HBM3_8Gb_2R
      channel: 16
    timing:
      preset: HBM3_5.2Gbps

  Controller:
    impl: HBM3-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: No

  AddrMapper:
    impl: HBM3-PIM
"""


def pim_address(ch: int, bank: int, row: int, col: int) -> int:
    """Encode one of our (channel, bank, row, column) coordinates into an
    AttAcc HBM3-PIM byte address."""
    return (((ch >> 1) << SH_CH) | ((ch & 1) << SH_PCH) | (0 << SH_RA)
            | ((bank >> 2) << SH_BG) | ((bank & 3) << SH_BA)
            | (row << SH_RO) | (col << SH_CO)) << TX_OFFSET


def command_stream(table: list[int], length: int, geom, shape,
                   block_tokens: int, am: AddrMap, *, window: int = 64,
                   mode: str = "window") -> list[tuple[int, int, int]]:
    """The all-bank command sequence for one sequence's decode sweep, as
    (channel, row, column), interleaved round-robin across channels."""
    m = enumerate_bursts(table, length, geom, shape, block_tokens, am)
    per_ch: dict[int, list[tuple[int, int]]] = {}
    for c in np.unique(m.ch):
        sel = m.ch == c
        per_ch[int(c)] = coalesce_channel_stream(
            m.bank[sel], m.ro[sel], m.co[sel],
            banks_per_channel=geom.banks_per_channel, window=window,
            mode=mode)
    out: list[tuple[int, int, int]] = []
    chans = sorted(per_ch)
    for i in range(max((len(v) for v in per_ch.values()), default=0)):
        for c in chans:
            if i < len(per_ch[c]):
                r, col = per_ch[c][i]
                out.append((c, r, col))
    return out


def emit_trace(cmds: list[tuple[int, int, int]], path: Path) -> int:
    with open(path, "w") as f:
        for c, r, col in cmds:
            f.write(f"PIM_MAC_AB 0x{pim_address(c, 0, r, col):08x}\n")
    return len(cmds)


def run_attacc(trace: Path, fe_ratio: int, workdir: Path) -> dict:
    if not BIN.exists():
        raise SystemExit(
            f"AttAcc PIM Ramulator not built at {BIN}.\n"
            "Build it per third_party/README.md (clone ramulator2 at "
            "b7c7027 into third_party/attacc_simulator/ramulator2, run "
            "set_pim_ramulator.sh, apply the Apple-clang param.h fix, make).")
    # the simulator runs with cwd=ATTACC_ROOT, so every path we hand it
    # must be absolute
    cfg = (workdir / f"{trace.stem}.yaml").resolve()
    cfg.write_text(CONFIG.format(trace=trace.resolve(), fe_ratio=fe_ratio))
    t0 = time.perf_counter()
    p = subprocess.run([str(BIN), "-f", str(cfg)], capture_output=True,
                       text=True, cwd=str(ATTACC_ROOT))
    if p.returncode != 0:
        raise SystemExit(f"ramulator2 failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    out = {}
    for key in ("memory_system_cycles", "total_num_pim_mac_all_bank_requests"):
        mm = re.search(rf"{key}:\s*(\d+)", p.stdout)
        if mm:
            out[key] = int(mm.group(1))
    out["wall_s"] = time.perf_counter() - t0
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m pimkv.attacc")
    p.add_argument("--workload", default="steady")
    p.add_argument("--allocators", default="paged,pim-aware,contiguous")
    p.add_argument("--addrmap", default="host-cacheline")
    p.add_argument("--block-tokens", type=int, default=16)
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--min-step", type=int, default=400)
    p.add_argument("--fe-ratio", type=int, default=64,
                   help="frontend clock ratio (frontend faster than memory "
                        "so the DRAM, not the trace reader, is the limit)")
    p.add_argument("--out", type=Path, default=Path("results/phaseE2"))
    args = p.parse_args(argv)

    geom = DRAM_PRESETS["hbm3-pim"]
    shape = MODEL_PRESETS["llama-gqa-8kv"]
    am = AddrMap(geom, args.addrmap)
    bt = args.block_tokens
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for al in args.allocators.split(","):
        requests = WORKLOADS[args.workload](args.requests, args.seed)
        pool = auto_pool_blocks(requests, bt, args.max_batch)
        fb = frame_blocks_for(geom, shape, bt)
        alloc = make_allocator(al, pool, args.seed,
                               frame_blocks=fb if al == "pim-aware" else 1)
        spec = SpecParams(adopt="copyback") if args.workload == "spec" else None
        captured: list[tuple[int, int, int, list[int]]] = []

        def hook(step, s, table, _c=captured):
            if step >= args.min_step and len(_c) < args.samples:
                _c.append((step, s.rid, s.length, table))

        simulate(requests, alloc, geom, shape, am, block_tokens=bt,
                 max_batch=args.max_batch, seed=args.seed, frame_blocks=fb,
                 spec=spec, on_sample=hook)
        for step, rid, length, table in captured:
            cmds = command_stream(table, length, geom, shape, bt, am)
            trace = args.out / f"{args.workload}_{al}_bt{bt}_seq{rid}_t{step}.trace"
            n = emit_trace(cmds, trace)
            ours = measure_seq(type("S", (), dict(rid=rid, length=length))(),
                               table, am, geom, shape, DEFAULT_TIMING, bt,
                               64, "window", fb)
            r = run_attacc(trace, args.fe_ratio, args.out)
            cyc = r.get("memory_system_cycles", float("nan"))
            kv = length * shape.kv_bytes_per_token
            rows.append(dict(
                workload=args.workload, allocator=al, block_tokens=bt,
                seq_id=rid, step=step, seq_len=length, kv_bytes=kv,
                n_cmds=n, m1=ours["m1"], m2=ours["m2"], m3_gbps=ours["m3_gbps"],
                attacc_cycles=cyc,
                attacc_mac_ab=r.get("total_num_pim_mac_all_bank_requests"),
                cycles_per_kb=cyc / (kv / 1024) if kv else float("nan"),
                wall_s=r["wall_s"]))
            print(f"  {al:11s} seq {rid:5d} len {length:5d}  cmds {n:7d}  "
                  f"M1 {ours['m1']:.3f} M2 {ours['m2']:.3f}  "
                  f"cycles {cyc}  ({r['wall_s']:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "summary.csv", index=False, float_format="%.8g")
    g = df.groupby("allocator")[["m1", "m2", "m3_gbps", "n_cmds",
                                 "attacc_cycles", "cycles_per_kb"]].mean()
    # normalize both models to the best allocator: do they agree on ratios?
    g["m3_rel"] = g.m3_gbps / g.m3_gbps.max()
    g["attacc_rel"] = g.cycles_per_kb.min() / g.cycles_per_kb
    print(g.to_string(float_format=lambda x: f"{x:,.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
