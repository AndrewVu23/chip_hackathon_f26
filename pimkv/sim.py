"""Continuous-batching simulator: workload -> allocator -> addrmap -> metrics.

One iteration of the serving engine's decode loop == one step. Per step:

  1. Admission (FCFS, head-of-line blocking, like vLLM's scheduler): a
     request is admitted when the batch has room AND the pool can hold its
     final footprint on top of the growth already committed to running
     sequences. This oracle admission gate (final lengths are known to the
     simulator) removes preemption/swapping from scope — identical for every
     policy, so comparisons are fair; recorded in LIMITATIONS.
  2. Decode: every running sequence appends one token, allocating a new
     block when it crosses a block boundary.
  3. Sampling: every ``sample_every`` steps, up to ``sample_seqs`` running
     sequences are measured: the sequence's whole KV cache is enumerated as
     burst addresses (block table -> linear -> addrmap) and pushed through
     the PIM command model (M1..M4). One CSV row per (step, sequence).
  4. Finished sequences release their blocks.

Determinism: everything is driven by the workload seed and ``seed`` (metric
sampling); same seeds -> byte-identical CSV (validation gate 4).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .addrmap import AddrMap
from .allocator import AdmissionFailure, KVAllocator
from .config import DramGeometry, ModelShape, PimTiming, DEFAULT_TIMING
from .pimmodel import (decode_latency_ns, effective_bandwidth_gbps,
                       sequence_metrics)
from .workload import Request


@dataclass
class SeqState:
    rid: int
    length: int          # tokens currently in the KV cache
    total_len: int       # prompt + output
    total_blocks: int
    blocks: list[int] = field(default_factory=list)


@dataclass
class SimResult:
    df: pd.DataFrame
    summary: dict


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def auto_pool_blocks(requests: list[Request], block_tokens: int,
                     max_batch: int, headroom: float = 1.3) -> int:
    """Size the pool to ~headroom x the mean concurrent demand at full batch."""
    mean_total = float(np.mean([r.prompt_len + r.output_len
                                for r in requests]))
    return int(headroom * max_batch * _ceil_div(int(mean_total) + 1,
                                                block_tokens))


def measure_seq(seq: SeqState, am: AddrMap, geom: DramGeometry,
                shape: ModelShape, timing: PimTiming, block_tokens: int,
                window: int, mode: str) -> dict:
    bpt = shape.kv_bytes_per_token // geom.burst_bytes
    bpb = block_tokens * bpt
    n_bursts = seq.length * bpt
    lb = np.arange(n_bursts, dtype=np.int64)
    blocks = np.asarray(seq.blocks, dtype=np.int64)
    phys = blocks[lb // bpb] * bpb + (lb % bpb)
    m = am.map(phys)
    s = sequence_metrics(m.ch, m.bank, m.ro, geom, window=window, mode=mode)
    kv_bytes = seq.length * shape.kv_bytes_per_token
    m3 = effective_bandwidth_gbps(s.m1, s.m2, geom, timing)
    return dict(seq_id=seq.rid, seq_len=seq.length, kv_bytes=kv_bytes,
                n_bursts=s.n_bursts, n_cmds=s.n_cmds, n_hits=s.n_hits,
                m1=s.m1, m2=s.m2, m3_gbps=m3,
                m4_ns=decode_latency_ns(kv_bytes, m3))


def simulate(requests: list[Request], alloc: KVAllocator, geom: DramGeometry,
             shape: ModelShape, am: AddrMap, *, block_tokens: int,
             max_batch: int = 64, sample_every: int = 16,
             sample_seqs: int = 8, window: int = 64, mode: str = "window",
             timing: PimTiming = DEFAULT_TIMING, seed: int = 0,
             check_every: int = 64) -> SimResult:
    if shape.kv_bytes_per_token % geom.burst_bytes:
        raise ValueError("kv_bytes_per_token must be a multiple of burst_bytes")
    bpb = block_tokens * (shape.kv_bytes_per_token // geom.burst_bytes)
    if alloc.num_blocks * bpb > geom.total_bursts:
        raise ValueError(
            f"pool ({alloc.num_blocks} blocks x {bpb} bursts) exceeds "
            f"geometry capacity {geom.total_bursts} bursts")

    rng = np.random.default_rng(seed + 1)
    t0 = time.perf_counter()
    reqs = sorted(requests, key=lambda r: (r.arrival_step, r.rid))
    active: dict[int, SeqState] = {}
    rows: list[dict] = []
    i = 0                      # next pending request
    t = 0
    completed = 0
    dropped: list[int] = []
    frag_failures = 0
    decode_steps = 0
    peak_live = 0

    while i < len(reqs) or active:
        # -- 1. admission ---------------------------------------------------
        while (i < len(reqs) and reqs[i].arrival_step <= t
               and len(active) < max_batch):
            r = reqs[i]
            total_blocks = _ceil_div(r.prompt_len + r.output_len, block_tokens)
            prompt_blocks = _ceil_div(r.prompt_len, block_tokens)
            committed = sum(s.total_blocks - len(s.blocks)
                            for s in active.values())
            if alloc.num_free - committed < total_blocks:
                break                      # wait for space (head-of-line)
            try:
                blocks = alloc.admit(r.rid, prompt_blocks, total_blocks)
            except AdmissionFailure:
                frag_failures += 1
                if not active:
                    dropped.append(r.rid)  # can never be placed; skip it
                    i += 1
                    continue
                break                      # retry once something frees
            active[r.rid] = SeqState(rid=r.rid, length=r.prompt_len,
                                     total_len=r.prompt_len + r.output_len,
                                     total_blocks=total_blocks, blocks=blocks)
            i += 1

        if not active:
            if i < len(reqs):
                t = max(t + 1, reqs[i].arrival_step)
                continue
            break

        # -- 2. decode: one token per running sequence ----------------------
        finished = []
        for s in active.values():
            s.length += 1
            if s.length > len(s.blocks) * block_tokens:
                s.blocks.append(alloc.append_block(s.rid))
            if s.length >= s.total_len:
                finished.append(s.rid)
        decode_steps += 1
        peak_live = max(peak_live, alloc.num_live)

        # -- 3. sampling ----------------------------------------------------
        if t % sample_every == 0:
            ids = sorted(active)
            pick = rng.choice(len(ids), size=min(sample_seqs, len(ids)),
                              replace=False)
            for j in sorted(pick.tolist()):
                row = measure_seq(active[ids[j]], am, geom, shape, timing,
                                  block_tokens, window, mode)
                row.update(step=t, live_blocks=alloc.num_live,
                           free_blocks=alloc.num_free)
                rows.append(row)

        # -- 4. completion --------------------------------------------------
        for rid in finished:
            alloc.release(rid)
            del active[rid]
            completed += 1

        if check_every and t % check_every == 0:
            alloc.assert_conservation()
        t += 1

    alloc.assert_conservation()
    cols = ["step", "seq_id", "seq_len", "kv_bytes", "n_bursts", "n_cmds",
            "n_hits", "m1", "m2", "m3_gbps", "m4_ns", "live_blocks",
            "free_blocks"]
    df = pd.DataFrame(rows, columns=cols)
    q = (lambda c, p: float(np.percentile(df[c], p)) if len(df) else float("nan"))
    summary = dict(
        n_requests=len(reqs), completed=completed, dropped=len(dropped),
        frag_failures=frag_failures, decode_steps=decode_steps,
        samples=len(df), peak_live_blocks=peak_live,
        pool_blocks=alloc.num_blocks,
        pool_utilization_peak=peak_live / alloc.num_blocks,
        m1_mean=float(df.m1.mean()) if len(df) else float("nan"),
        m1_p05=q("m1", 5), m1_p95=q("m1", 95),
        m2_mean=float(df.m2.mean()) if len(df) else float("nan"),
        m2_p05=q("m2", 5), m2_p95=q("m2", 95),
        m3_gbps_mean=float(df.m3_gbps.mean()) if len(df) else float("nan"),
        m4_ns_mean=float(df.m4_ns.mean()) if len(df) else float("nan"),
        m4_ns_p95=q("m4_ns", 95),
        runtime_s=time.perf_counter() - t0,
    )
    return SimResult(df=df, summary=summary)
