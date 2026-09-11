"""Continuous-batching simulator: workload -> allocator -> addrmap -> metrics.

One iteration of the serving engine's decode loop == one step. Per step:

  1. Admission (FCFS, head-of-line blocking): a request is admitted when the
     batch has room AND the pool can hold its final footprint on top of the
     growth already committed to running sequences (oracle admission — final
     lengths known; removes preemption/swap from scope, identical for every
     policy). Prefix-workload requests attach to a pinned prefix cache
     (block sharing, ref-counted) when the allocator supports it.
  2. Decode. Normal workloads: every running sequence appends one token,
     allocating a block at each boundary. Spec workload: every sequence
     runs one speculation round (see _spec_round).
  3. Sampling: every ``sample_every`` steps, up to ``sample_seqs`` running
     sequences are measured: block table -> linear burst addresses ->
     addrmap -> all-bank command model (M1..M4). One CSV row per
     (step, sequence).
  4. Finished sequences release their blocks.

Speculation model (--workload spec): per round, the sequence forks W draft
branches of depth D (brief §5.1: draft tree W=4 D=3; we model W independent
depth-D paths — a mid-density tree). Each branch owns a private tail: a
copy-on-write duplicate of the partial last block (vLLM append_slot
semantics) plus overflow blocks. Draft i (1-indexed) is accepted with
probability accept^i, sequentially until the first rejection, plus one
bonus token; the adopted branch's tail is spliced into the block table and
every other branch is freed — allocate/free churn several times per
committed token. If the pool cannot hold W branch tails, the round degrades
to ordinary 1-token decode (counted as spec_stalls — real engines disable
speculation under memory pressure too). The contiguous oracle instead uses
a per-sequence reserved scratch region (zero churn) and pays a copy of the
accepted tokens into its span (copied_bytes — the price of contiguity).

Determinism: workload seed + ``seed`` (sampling: seed+1, speculation:
seed+2) -> byte-identical CSV (validation gate 4).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .addrmap import AddrMap
from .allocator import AdmissionFailure, KVAllocator
from .config import DramGeometry, ModelShape, PimTiming, DEFAULT_TIMING
from .pimmodel import (decode_latency_ns, effective_bandwidth_gbps,
                       sequence_metrics)
from .workload import Request


@dataclass(frozen=True)
class SpecParams:
    width: int = 4      # W draft branches per round
    depth: int = 3      # D draft tokens per branch
    accept: float = 0.8  # draft i accepted with accept**i
    adopt: str = "splice"  # "splice": vLLM pointer-swap of the winning
    #                        branch's tail into the block table (no copy);
    #                        "copyback": accepted tokens are copied into the
    #                        sequence's own tail (cost counted in
    #                        copied_bytes) and ALL branch blocks are freed —
    #                        committed blocks then never live in scratch.

    def scratch_blocks(self, block_tokens: int) -> int:
        """Worst-case tail blocks per branch: partial block + D drafts +
        the bonus token emitted on full acceptance."""
        return -(-(block_tokens - 1 + self.depth + 1) // block_tokens)


@dataclass
class SeqState:
    rid: int
    length: int          # tokens currently committed to the KV cache
    total_len: int       # prompt + output
    total_blocks: int    # final footprint incl. contiguous spec scratch
    scratch: list[int] = field(default_factory=list)
    prefix_id: int | None = None
    prefix_len: int = 0
    preempts: int = 0


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


def placement_diagnostics(table: list[int], frame_blocks: int) -> dict:
    """Structural quality of a block table, independent of the DRAM map.

    ``frame_blocks`` is the alignment quantum in blocks (one row index across
    every bank of every channel) and is a property of the GEOMETRY, not of
    the allocator — so every policy is measured on the same yardstick.

    blk_adj      fraction of consecutive logical block pairs that are also
                 physically adjacent (b[i+1] == b[i] + 1).
    same_frame   fraction of consecutive pairs inside one alignment frame.
    frame_spread frames actually touched / minimum frames needed. 1.0 means
                 perfect packing; 2.0 means the sequence is smeared over
                 twice as many frames as its size requires.
    """
    b = np.asarray(table, dtype=np.int64)
    if b.size < 2:
        return dict(blk_adj=float("nan"), same_frame=float("nan"),
                    frame_spread=float("nan"))
    d = np.diff(b)
    out = dict(blk_adj=float(np.mean(d == 1)),
               same_frame=float("nan"), frame_spread=float("nan"))
    if frame_blocks > 1:
        fr = b // frame_blocks
        out["same_frame"] = float(np.mean(np.diff(fr) == 0))
        out["frame_spread"] = (len(np.unique(fr))
                               / _ceil_div(b.size, frame_blocks))
    return out


def measure_seq(seq: SeqState, table: list[int], am: AddrMap,
                geom: DramGeometry, shape: ModelShape, timing: PimTiming,
                block_tokens: int, window: int, mode: str,
                frame_blocks: int = 1) -> dict:
    bpt = shape.kv_bytes_per_token // geom.burst_bytes
    bpb = block_tokens * bpt
    n_bursts = seq.length * bpt
    lb = np.arange(n_bursts, dtype=np.int64)
    blocks = np.asarray(table, dtype=np.int64)
    phys = blocks[lb // bpb] * bpb + (lb % bpb)
    m = am.map(phys)
    s = sequence_metrics(m.ch, m.bank, m.ro, geom, window=window, mode=mode)
    kv_bytes = seq.length * shape.kv_bytes_per_token
    m3 = effective_bandwidth_gbps(s.m1, s.m2, geom, timing)
    row = dict(seq_id=seq.rid, seq_len=seq.length, kv_bytes=kv_bytes,
               n_bursts=s.n_bursts, n_cmds=s.n_cmds, n_hits=s.n_hits,
               m1=s.m1, m2=s.m2, m3_gbps=m3,
               m4_ns=decode_latency_ns(kv_bytes, m3))
    row.update(placement_diagnostics(table, frame_blocks))
    return row


def simulate(requests: list[Request], alloc: KVAllocator, geom: DramGeometry,
             shape: ModelShape, am: AddrMap, *, block_tokens: int,
             max_batch: int = 64, sample_every: int = 16,
             sample_seqs: int = 8, window: int = 64, mode: str = "window",
             timing: PimTiming = DEFAULT_TIMING, seed: int = 0,
             check_every: int = 64, frame_blocks: int = 1,
             spec: SpecParams | None = None, admission: str = "oracle",
             watermark: float = 0.01, on_sample=None) -> SimResult:
    """``admission``: "oracle" (Phases 0-2: reserve the final footprint, no
    preemption) or "vllm" (Phase D: vLLM v0.2.7 scheduler semantics —
    admit on prompt blocks + a 1% watermark, and when a running sequence
    cannot get a block, preempt the most recently admitted other sequence
    by RECOMPUTE: free all its blocks, keep its generated tokens as prompt,
    re-queue it at the front). Under "vllm" the pool is genuinely
    oversubscribed, so the free list churns the way it does in production.
    """
    if shape.kv_bytes_per_token % geom.burst_bytes:
        raise ValueError("kv_bytes_per_token must be a multiple of burst_bytes")
    bpb = block_tokens * (shape.kv_bytes_per_token // geom.burst_bytes)
    if alloc.num_blocks * bpb > geom.total_bursts:
        raise ValueError(
            f"pool ({alloc.num_blocks} blocks x {bpb} bursts) exceeds "
            f"geometry capacity {geom.total_bursts} bursts")

    bt = block_tokens
    rng = np.random.default_rng(seed + 1)       # metric sampling
    rng_spec = np.random.default_rng(seed + 2)  # speculation decisions
    t0 = time.perf_counter()
    reqs = sorted(requests, key=lambda r: (r.arrival_step, r.rid))
    active: dict[int, SeqState] = {}
    prefix_cache: dict[int, list[int]] = {}     # prefix_id -> full blocks
    rows: list[dict] = []
    i = 0
    t = 0
    completed = 0
    dropped: list[int] = []
    frag_failures = 0
    decode_steps = 0
    spec_stalls = 0
    copied_tokens = 0
    peak_live = 0
    scratch_per_seq = (spec.scratch_blocks(bt) * spec.width
                       if (spec and alloc.persistent_scratch) else 0)

    if admission not in ("oracle", "vllm"):
        raise ValueError(f"unknown admission mode {admission!r}")
    if admission == "vllm" and alloc.persistent_scratch:
        raise ValueError("the contiguous oracle reserves whole spans up "
                         "front; it only makes sense under oracle admission")
    watermark_blocks = int(watermark * alloc.num_blocks)
    waiting: deque[SeqState] = deque()   # preempted; recompute on re-admit
    preemptions = 0
    self_preemptions = 0

    def committed() -> int:
        return sum(s.total_blocks - len(alloc.get_table(s.rid))
                   - len(s.scratch) for s in active.values())

    # ---------------------------------------------------------------- admit
    def try_admit(st: SeqState) -> bool:
        """Admit (or re-admit after preemption) ``st``. Its current
        ``length`` is the prompt to prefill — recompute keeps the tokens
        generated before preemption, exactly as vLLM does."""
        nonlocal frag_failures
        total_blocks = _ceil_div(st.total_len, bt)
        prompt_blocks = _ceil_div(st.length, bt)
        total_with_scratch = total_blocks + scratch_per_seq
        shared: list[int] = []
        if (st.prefix_id is not None and alloc.supports_sharing
                and st.prefix_len >= bt):
            if st.prefix_id not in prefix_cache:
                n_shared = st.prefix_len // bt
                cache_id = -1000 - st.prefix_id
                if admission == "oracle":
                    ok = (alloc.num_free - committed()
                          >= n_shared + total_with_scratch)
                else:
                    ok = alloc.num_free - prompt_blocks >= watermark_blocks
                if not ok:
                    return False
                try:
                    prefix_cache[st.prefix_id] = alloc.admit(
                        cache_id, n_shared, n_shared)
                except AdmissionFailure:
                    frag_failures += 1
                    return False
            shared = prefix_cache[st.prefix_id]
        n_new = prompt_blocks - len(shared)
        if admission == "oracle":
            if alloc.num_free - committed() < total_with_scratch - len(shared):
                return False
        else:
            # vLLM v0.2.7 BlockSpaceManager.can_allocate (lines 103-120):
            # free - prompt blocks >= watermark; future growth unreserved
            if alloc.num_free - n_new < watermark_blocks:
                return False
        try:
            if shared:
                alloc.admit_shared(st.rid, shared, n_new, total_with_scratch)
            else:
                alloc.admit(st.rid, prompt_blocks, total_with_scratch)
        except AdmissionFailure:
            frag_failures += 1
            return False
        st.total_blocks = total_with_scratch
        if scratch_per_seq:
            st.scratch = alloc.take_scratch(st.rid, scratch_per_seq)
        active[st.rid] = st
        return True

    def preempt(rid: int) -> None:
        """vLLM v0.2.7 Scheduler._preempt_by_recompute: free every block,
        keep generated tokens as prompt, re-queue at the FRONT of waiting."""
        nonlocal preemptions
        st = active.pop(rid)
        if st.scratch:
            alloc.unref_blocks(st.scratch)
            st.scratch = []
        alloc.release(rid)
        st.preempts += 1
        preemptions += 1
        waiting.appendleft(st)

    def ensure_free(s: SeqState, n: int) -> bool:
        """vLLM mode: make room for ``n`` blocks for ``s`` by preempting the
        most recently admitted OTHER sequences (Scheduler._schedule pops
        victims from the tail of the running list). With no one else left,
        ``s`` preempts itself and its step is abandoned (returns False).
        Oracle mode never needs this (its admission gate guarantees room)."""
        nonlocal self_preemptions
        if admission == "oracle" or alloc.num_free >= n:
            return True
        while alloc.num_free < n:
            others = [rid for rid in active if rid != s.rid]
            if not others:
                self_preemptions += 1
                preempt(s.rid)
                return False
            preempt(others[-1])
        return True

    # ------------------------------------------------------------ spec round
    def spec_round(s: SeqState) -> None:
        nonlocal spec_stalls, copied_tokens
        remaining = s.total_len - s.length
        # acceptance: draft i (1-indexed) survives with accept**i
        k = 0
        while k < spec.depth and rng_spec.random() < spec.accept ** (k + 1):
            k += 1
        adv = min(k + 1, remaining)
        w_star = int(rng_spec.integers(spec.width))

        if alloc.persistent_scratch:
            # contiguous oracle: branches live in the fixed scratch region;
            # accepted tokens are copied into the span
            grow = _ceil_div(s.length + adv, bt) - _ceil_div(s.length, bt)
            if not ensure_free(s, grow):
                return
            for _ in range(grow):
                alloc.append_block(s.rid)
            copied_tokens += adv
            s.length += adv
            return

        r = s.length % bt
        # a branch tail must hold the partial block, D drafts, and the
        # bonus token (adv can reach D+1)
        d1 = spec.depth + 1
        n_br = _ceil_div(r + d1, bt) if r else _ceil_div(d1, bt)
        if alloc.num_free < spec.width * n_br:
            # pool too tight for branch tails: degrade to plain decode
            spec_stalls += 1
            need = (s.length + 1) > len(alloc.get_table(s.rid)) * bt
            if need and not ensure_free(s, 1):
                return
            s.length += 1
            if need:
                alloc.append_block(s.rid)
            return
        branches = [alloc.alloc_scratch(s.rid, n_br)
                    for _ in range(spec.width)]
        if r:
            copied_tokens += spec.width * r   # each branch CoWs the tail
        if spec.adopt == "copyback":
            grow = _ceil_div(s.length + adv, bt) - _ceil_div(s.length, bt)
            if grow and not ensure_free(s, grow):
                for br in branches:
                    alloc.unref_blocks(br)
                return
            for _ in range(grow):
                alloc.append_block(s.rid)
            copied_tokens += adv
            for br in branches:
                alloc.unref_blocks(br)
            s.length += adv
            return
        keep = _ceil_div(r + adv, bt) if r else _ceil_div(adv, bt)
        alloc.spec_round_adopt(s.rid, replace_tail=bool(r),
                               chosen=branches[w_star], keep=keep,
                               branches=branches)
        s.length += adv

    # ------------------------------------------------------------- main loop
    while i < len(reqs) or active or waiting:
        # preempted sequences first (vLLM re-queues them at the front),
        # then new arrivals; FCFS with head-of-line blocking throughout
        blocked = False
        while waiting and len(active) < max_batch:
            st = waiting[0]
            if (st.preempts > 1000
                    or _ceil_div(st.length + 1, bt) > alloc.num_blocks):
                waiting.popleft()
                dropped.append(st.rid)          # can never fit / livelock
                continue
            if try_admit(st):
                waiting.popleft()
            elif not active:
                waiting.popleft()
                dropped.append(st.rid)
            else:
                blocked = True
                break
        while (not blocked and i < len(reqs) and reqs[i].arrival_step <= t
               and len(active) < max_batch):
            r = reqs[i]
            st = SeqState(rid=r.rid, length=r.prompt_len,
                          total_len=r.prompt_len + r.output_len,
                          total_blocks=0, prefix_id=r.prefix_id,
                          prefix_len=r.prefix_len)
            if try_admit(st):
                i += 1
            elif not active:
                dropped.append(r.rid)   # can never be placed; skip it
                i += 1
            else:
                break                   # head-of-line: wait for space

        if not active:
            if i < len(reqs):
                t = max(t + 1, reqs[i].arrival_step)
                continue
            break

        finished = []
        for s in list(active.values()):
            if s.rid not in active:     # preempted earlier this step
                continue
            if spec is not None:
                spec_round(s)
            else:
                need = (s.length + 1) > len(alloc.get_table(s.rid)) * bt
                if need and not ensure_free(s, 1):
                    continue
                s.length += 1
                if need:
                    alloc.append_block(s.rid)
            if s.rid in active and s.length >= s.total_len:
                finished.append(s.rid)
        decode_steps += 1
        peak_live = max(peak_live, alloc.num_live)

        if t % sample_every == 0:
            ids = sorted(active)
            pick = rng.choice(len(ids), size=min(sample_seqs, len(ids)),
                              replace=False)
            for j in sorted(pick.tolist()):
                s = active[ids[j]]
                row = measure_seq(s, alloc.get_table(s.rid), am, geom, shape,
                                  timing, bt, window, mode, frame_blocks)
                row.update(step=t, live_blocks=alloc.num_live,
                           free_blocks=alloc.num_free)
                rows.append(row)
                if on_sample is not None:   # Phase E trace capture hook
                    on_sample(t, s, list(alloc.get_table(s.rid)))

        for rid in finished:
            s = active.pop(rid)
            if s.scratch:
                alloc.unref_blocks(s.scratch)
            alloc.release(rid)
            completed += 1

        if check_every and t % check_every == 0:
            alloc.assert_conservation()
        t += 1

    for pid in sorted(prefix_cache):
        alloc.release(-1000 - pid)
    alloc.assert_conservation()

    cols = ["step", "seq_id", "seq_len", "kv_bytes", "n_bursts", "n_cmds",
            "n_hits", "m1", "m2", "m3_gbps", "m4_ns", "blk_adj",
            "same_frame", "frame_spread", "live_blocks", "free_blocks"]
    df = pd.DataFrame(rows, columns=cols)
    q = (lambda c, p: float(np.percentile(df[c], p)) if len(df) else float("nan"))
    summary = dict(
        n_requests=len(reqs), completed=completed, dropped=len(dropped),
        frag_failures=frag_failures, decode_steps=decode_steps,
        samples=len(df), peak_live_blocks=peak_live,
        pool_blocks=alloc.num_blocks,
        pool_utilization_peak=peak_live / alloc.num_blocks,
        blocks_allocated_total=alloc.allocated_total,
        cow_copies=alloc.cow_copies,
        blocks_copied=getattr(alloc, "blocks_copied", 0),
        compactions=getattr(alloc, "compactions", 0),
        spec_stalls=spec_stalls,
        copied_bytes=copied_tokens * shape.kv_bytes_per_token,
        admission=admission, preemptions=preemptions,
        self_preemptions=self_preemptions,
        m1_mean=float(df.m1.mean()) if len(df) else float("nan"),
        m1_p05=q("m1", 5), m1_p95=q("m1", 95),
        m2_mean=float(df.m2.mean()) if len(df) else float("nan"),
        m2_p05=q("m2", 5), m2_p95=q("m2", 95),
        m3_gbps_mean=float(df.m3_gbps.mean()) if len(df) else float("nan"),
        m4_ns_mean=float(df.m4_ns.mean()) if len(df) else float("nan"),
        m4_ns_p95=q("m4_ns", 95),
        blk_adj_mean=float(df.blk_adj.mean()) if len(df) else float("nan"),
        same_frame_mean=(float(df.same_frame.mean()) if len(df)
                         else float("nan")),
        frame_spread_mean=(float(df.frame_spread.mean()) if len(df)
                           else float("nan")),
        runtime_s=time.perf_counter() - t0,
    )
    return SimResult(df=df, summary=summary)
