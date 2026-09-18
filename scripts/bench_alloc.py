"""Allocator CPU cost measurement. Measurement only — imports pimkv, changes nothing in it.

    .venv/bin/python scripts/bench_alloc.py --out results/bench_alloc

Part 1 (in situ): wrap a real allocator instance in a timing proxy and run
the normal simulator, so the operation mix, pool pressure and fallback
frequency are exactly those of the reported runs. Metric sampling is turned
off (it does not touch the allocator) so the run is fast.

Part 2 (worst case): fill the pool until no region is fully empty, then time
allocations one at a time — this is PimAware._pick_frame's linear fallback
scan, the only super-constant path in the allocator.

Timing harness overhead is measured and reported alongside; all numbers are
CPython on this machine, which is the right comparison because vLLM's block
manager is Python too.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pimkv.addrmap import AddrMap
from pimkv.allocator import ALLOCATORS, make_allocator
from pimkv.config import DRAM_PRESETS, MODEL_PRESETS
from pimkv.run import frame_blocks_for
from pimkv.sim import SpecParams, auto_pool_blocks, simulate
from pimkv.workload import WORKLOADS

# public allocator API the simulator drives (assert_conservation is a
# validation harness, not production cost, so it is excluded)
TIMED_METHODS = ("admit", "admit_shared", "append_block", "alloc_scratch",
                 "unref_blocks", "spec_round_adopt", "fork", "release",
                 "take_scratch", "get_table")


class _TimedMixin:
    """Times every outermost allocator call. Nested calls (e.g. num_free
    inside admit) are counted but their time is already inside the outer
    measurement, so it is not added twice."""

    def _init_timing(self) -> None:
        self.t_ns: dict[str, int] = {}
        self.calls: dict[str, int] = {}
        self._depth = 0

    def _rec(self, name: str, dt: int) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        if self._depth == 0:
            self.t_ns[name] = self.t_ns.get(name, 0) + dt

    @property
    def num_free(self):
        t0 = time.perf_counter_ns()
        self._depth += 1
        try:
            v = super().num_free
        finally:
            self._depth -= 1
        self._rec("num_free", time.perf_counter_ns() - t0)
        return v

    @property
    def num_live(self):
        t0 = time.perf_counter_ns()
        self._depth += 1
        try:
            v = super().num_live
        finally:
            self._depth -= 1
        self._rec("num_live", time.perf_counter_ns() - t0)
        return v


def _wrap_method(name: str):
    def f(self, *a, **k):
        t0 = time.perf_counter_ns()
        self._depth += 1
        try:
            r = getattr(super(type(self), self), name)(*a, **k)
        finally:
            self._depth -= 1
        self._rec(name, time.perf_counter_ns() - t0)
        return r
    f.__name__ = name
    return f


def timed_allocator(alloc):
    """Return ``alloc`` re-created as an instance of a timing subclass."""
    base = type(alloc)
    ns = {n: _wrap_method(n) for n in TIMED_METHODS if hasattr(base, n)}
    cls = type(f"Timed{base.__name__}", (_TimedMixin, base), ns)
    alloc.__class__ = cls
    alloc._init_timing()
    return alloc


def harness_overhead_ns(reps: int = 200_000) -> float:
    """Cost of the timing wrapper itself: two clock reads + bookkeeping."""
    d: dict[str, int] = {}
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        a = time.perf_counter_ns()
        b = time.perf_counter_ns()
        d["x"] = d.get("x", 0) + (b - a)
    return (time.perf_counter_ns() - t0) / reps


def deep_size(obj, seen=None) -> int:
    """Recursive sys.getsizeof over the allocator's own containers."""
    import sys as _sys
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = _sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen)
                    for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(deep_size(x, seen) for x in obj)
    return size


def struct_bytes(alloc) -> int:
    """Bytes held by placement bookkeeping (not the block tables, which
    every allocator has)."""
    fields = {"paged": ["free_blocks"], "random": ["free_blocks"],
              "pim-aware": ["frame_free", "empty_frames", "affinity", "plans"],
              "contiguous": ["extents", "reserved", "spans"]}
    return sum(deep_size(getattr(alloc, f))
               for f in fields.get(alloc.name, []) if hasattr(alloc, f))


# ---------------------------------------------------------------- part 1
def run_in_situ(name: str, workload: str, admission: str, headroom: float,
                requests: int, seed: int, block_tokens: int) -> dict:
    geom = DRAM_PRESETS["hbm3-pim"]
    shape = MODEL_PRESETS["llama-gqa-8kv"]
    am = AddrMap(geom, "host-cacheline")
    reqs = WORKLOADS[workload](requests, seed)
    pool = auto_pool_blocks(reqs, block_tokens, 64, headroom=headroom)
    fb = frame_blocks_for(geom, shape, block_tokens)
    alloc = make_allocator(name, pool, seed,
                           frame_blocks=fb if name == "pim-aware" else 1)
    spec = (SpecParams(adopt="copyback" if name == "pim-aware" else "splice")
            if workload == "spec" else None)
    alloc = timed_allocator(alloc)
    t0 = time.perf_counter()
    res = simulate(reqs, alloc, geom, shape, am, block_tokens=block_tokens,
                   max_batch=64, sample_every=10 ** 9, sample_seqs=1,
                   seed=seed, frame_blocks=fb, spec=spec,
                   admission=admission)
    wall = time.perf_counter() - t0
    total_ns = sum(alloc.t_ns.values())
    calls = sum(alloc.calls.values())
    s = res.summary
    row = dict(part="in-situ", allocator=name, workload=workload,
               admission=admission, headroom=headroom,
               pool_blocks=alloc.num_blocks, decode_steps=s["decode_steps"],
               blocks_allocated=s["blocks_allocated_total"],
               preemptions=s["preemptions"], calls=calls,
               alloc_ms_total=total_ns / 1e6,
               ns_per_call=total_ns / calls if calls else float("nan"),
               us_per_decode_step=total_ns / 1e3 / s["decode_steps"],
               ns_per_block_alloc=(total_ns / s["blocks_allocated_total"]
                                   if s["blocks_allocated_total"] else float("nan")),
               struct_kib=struct_bytes(alloc) / 1024, sim_wall_s=wall)
    row["breakdown"] = "; ".join(
        f"{k}:{alloc.calls[k]}x/{alloc.t_ns.get(k, 0) / 1e6:.1f}ms"
        for k in sorted(alloc.t_ns, key=lambda k: -alloc.t_ns[k])[:5])
    return row


# ---------------------------------------------------------------- part 2
def run_worst_case(name: str, pool: int, fb: int, reps: int = 2000,
                   seed: int = 0) -> dict:
    """No fully-empty region left: PimAware falls back to scanning every
    region for the emptiest one. Paged's pop() is unaffected — the same
    measurement is run on it as the control."""
    alloc = make_allocator(name, pool, seed,
                           frame_blocks=fb if name == "pim-aware" else 1)
    n = alloc.num_blocks
    owner: dict[int, int] = {}
    for sid in range(n):                       # fill the pool completely
        owner[alloc.admit(sid, 1, 1)[0]] = sid
    # free fb-1 slots in EVERY region: no region is empty, so pim-aware can
    # never take the fast path, yet there is room for `reps` allocations
    for r in range(n // fb):
        for off in range(fb - 1):
            alloc.release(owner[r * fb + off])
    if name == "pim-aware":
        empty = sum(1 for f in alloc.frame_free if len(f) == alloc.G)
        assert empty == 0, f"expected no empty region, found {empty}"
    assert alloc.num_free >= reps, f"{alloc.num_free} free < {reps} reps"

    # growth path: append_block only (fallback region scan, no planning)
    grower = n + 10 ** 6
    alloc.admit(grower, 1, 1)
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        alloc.append_block(grower)
    dt_grow = time.perf_counter_ns() - t0
    # admission path: admit() also runs best-fit planning over all regions
    t1 = time.perf_counter_ns()
    for sid in range(n + 1, n + 1 + reps // 10):
        alloc.admit(sid, 1, 1)
    dt_admit = time.perf_counter_ns() - t1
    return dict(part="worst-case", allocator=name, pool_blocks=n,
                regions=(n // fb if name == "pim-aware" else n),
                calls=reps, ns_per_block_alloc=dt_grow / reps,
                ns_per_admit=dt_admit / (reps // 10),
                alloc_ms_total=dt_grow / 1e6)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench_alloc")
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-tokens", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("results/bench_alloc"))
    args = p.parse_args(argv)

    oh = harness_overhead_ns()
    print(f"# timing harness overhead: {oh:.0f} ns per wrapped call "
          f"(subtract this from ns_per_call)\n")

    cases = [("steady", "oracle", 1.3), ("spec", "oracle", 1.3),
             ("steady", "vllm", 0.6)]
    rows = []
    for workload, admission, headroom in cases:
        for name in ("paged", "pim-aware", "contiguous"):
            if admission == "vllm" and name == "contiguous":
                continue          # contiguous requires oracle admission
            r = run_in_situ(name, workload, admission, headroom,
                            args.requests, args.seed, args.block_tokens)
            rows.append(r)
            print(f"{workload:6s} {admission:6s} h{headroom} {name:11s} "
                  f"{r['calls']:8d} calls  {r['alloc_ms_total']:8.1f} ms  "
                  f"{r['ns_per_call']:7.0f} ns/call  "
                  f"{r['us_per_decode_step']:7.1f} us/step  "
                  f"{r['struct_kib']:7.1f} KiB", flush=True)

    fb = frame_blocks_for(DRAM_PRESETS["hbm3-pim"],
                          MODEL_PRESETS["llama-gqa-8kv"], args.block_tokens)
    print()
    for name in ("paged", "pim-aware"):
        r = run_worst_case(name, 3168, fb)
        rows.append(r)
        print(f"worst-case {name:11s} {r['ns_per_block_alloc']:8.0f} ns/append  "
              f"{r['ns_per_admit']:8.0f} ns/admit  "
              f"({r['regions']} regions)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "harness_overhead_ns", round(oh, 1))
    df.to_csv(args.out / "summary.csv", index=False, float_format="%.6g")
    print(f"\nwrote {args.out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
