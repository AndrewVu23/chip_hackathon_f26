"""End-to-end simulator tests: validation gates 3-5 (AGENT_BRIEF §6)."""
import numpy as np
import pytest

from pimkv.addrmap import AddrMap
from pimkv.allocator import ContiguousOracle, PagedFirstFit
from pimkv.config import HBM3_PIM, LLAMA_GQA_8KV
from pimkv.sim import auto_pool_blocks, simulate
from pimkv.workload import steady

GEOM = HBM3_PIM
SHAPE = LLAMA_GQA_8KV


def _run(alloc_cls, scheme, block_tokens, n_req=60, seed=0, **kw):
    reqs = steady(n_req, seed=seed)
    pool = auto_pool_blocks(reqs, block_tokens, kw.get("max_batch", 32))
    alloc = alloc_cls(pool, seed=seed)
    am = AddrMap(GEOM, scheme)
    kw.setdefault("max_batch", 32)
    kw.setdefault("sample_every", 8)
    kw.setdefault("sample_seqs", 4)
    return simulate(reqs, alloc, GEOM, SHAPE, am, block_tokens=block_tokens,
                    seed=seed, **kw)


def test_gate3_monotonic_in_block_size_for_contiguous():
    """Gate 3: M1 non-decreasing in block size for the contiguous allocator
    (a physically contiguous span is contiguous at any block granularity, so
    a decrease flags an aliasing bug in the address map)."""
    m1s = []
    for bt in (4, 8, 16, 32, 64, 128):
        res = _run(ContiguousOracle, "pim-friendly", bt, n_req=40)
        m1s.append(res.summary["m1_mean"])
    for lo, hi in zip(m1s, m1s[1:]):
        assert hi >= lo - 0.005, f"M1 decreased with block size: {m1s}"


def test_gate4_determinism():
    """Gate 4: same seed -> byte-identical results CSV."""
    csvs = []
    for _ in range(2):
        res = _run(PagedFirstFit, "host-centric", 16, n_req=80, seed=3)
        csvs.append(res.df.to_csv(index=False, float_format="%.8g"))
    assert csvs[0] == csvs[1]


def test_gate5_conservation_every_step():
    """Gate 5: allocated - freed == live at every step (checked inside the
    allocator each step via check_every=1; raises on violation)."""
    res = _run(PagedFirstFit, "host-centric", 16, n_req=120, check_every=1)
    s = res.summary
    assert s["completed"] + s["dropped"] == 120


def test_all_requests_accounted_for():
    res = _run(ContiguousOracle, "pim-friendly", 16, n_req=60, check_every=1)
    s = res.summary
    assert s["completed"] + s["dropped"] == 60


def test_metrics_in_range_and_sampled():
    res = _run(PagedFirstFit, "host-cacheline", 16, n_req=60)
    df = res.df
    assert len(df) > 0
    assert df.m1.between(0, 1).all()
    assert df.m2.between(0, 1).all()
    assert (df.n_hits <= df.n_cmds).all()
    assert (df.m3_gbps > 0).all()
    # every burst of a sampled sequence was enumerated
    bpt = SHAPE.kv_bytes_per_token // GEOM.burst_bytes
    assert (df.n_bursts == df.seq_len * bpt).all()


def test_inorder_mode_runs():
    res = _run(PagedFirstFit, "host-centric", 16, n_req=20,
               mode="inorder", sample_every=64, sample_seqs=2)
    assert 0.0 <= res.summary["m1_mean"] <= 1.0
