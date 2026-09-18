"""End-to-end simulator tests: validation gates 3-5."""
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


# ------------------------------------------------------- spec / prefix

from pimkv.allocator import make_allocator
from pimkv.sim import SpecParams
from pimkv.workload import prefix as prefix_wl, spec as spec_wl


def _run_wl(reqs, alloc_name, scheme, bt=16, seed=0, spec=None, fb=1, **kw):
    pool = auto_pool_blocks(reqs, bt, kw.get("max_batch", 32))
    alloc = make_allocator(alloc_name, pool, seed, frame_blocks=fb)
    am = AddrMap(GEOM, scheme)
    kw.setdefault("max_batch", 32)
    kw.setdefault("sample_every", 8)
    kw.setdefault("sample_seqs", 4)
    return simulate(reqs, alloc, GEOM, SHAPE, am, block_tokens=bt,
                    seed=seed, spec=spec, **kw)


def test_spec_conservation_and_churn():
    reqs = spec_wl(80, seed=0)
    res = _run_wl(reqs, "paged", "host-cacheline", spec=SpecParams(),
                  check_every=1)
    s = res.summary
    assert s["completed"] + s["dropped"] == 80
    # speculation must allocate far more physical blocks than the final
    # footprints need (fork/rewind churn is the point of this workload)
    assert s["blocks_allocated_total"] > 3 * s["peak_live_blocks"]
    assert s["cow_copies"] > 0


def test_spec_determinism():
    outs = []
    for _ in range(2):
        res = _run_wl(spec_wl(60, seed=3), "paged", "host-cacheline",
                      spec=SpecParams(), seed=3)
        outs.append(res.df.to_csv(index=False, float_format="%.8g"))
    assert outs[0] == outs[1]


def test_spec_contiguous_uses_scratch_not_churn():
    reqs = spec_wl(60, seed=0)
    res = _run_wl(reqs, "contiguous", "host-cacheline", spec=SpecParams(),
                  check_every=1)
    s = res.summary
    assert s["completed"] + s["dropped"] == 60
    assert s["cow_copies"] == 0
    assert s["copied_bytes"] > 0          # pays copies instead of pointers


def test_prefix_sharing_conservation():
    reqs = prefix_wl(100, seed=0)
    res = _run_wl(reqs, "paged", "host-cacheline", check_every=1)
    s = res.summary
    assert s["completed"] + s["dropped"] == 100


def test_prefix_sharing_saves_memory_vs_contiguous():
    reqs = prefix_wl(100, seed=0)
    paged = _run_wl(reqs, "paged", "host-cacheline")
    contig = _run_wl(reqs, "contiguous", "host-cacheline")
    # contiguous cannot share the system prompt: strictly larger peak
    assert contig.summary["peak_live_blocks"] > paged.summary["peak_live_blocks"]


# --------------------------------------------------- PimAware recovery

def test_pim_aware_recovers_alignment_under_host_cacheline():
    """The contribution: with bt=16 (2 KB per-channel slice, 1/8 row-group)
    under the realistic host map, paged M1 ~ 0.76; frame-aligned placement
    must recover close to the contiguous oracle (~0.96)."""
    reqs = steady(60, seed=0)
    fb = (GEOM.rowgroup_bytes * GEOM.channels) // (16 * SHAPE.kv_bytes_per_token)
    pa = _run_wl(reqs, "pim-aware", "host-cacheline", fb=fb, check_every=1)
    pg = _run_wl(reqs, "paged", "host-cacheline")
    assert pa.summary["m1_mean"] > pg.summary["m1_mean"] + 0.1
    assert pa.summary["m1_mean"] >= 0.90


def test_pim_aware_spec_still_aligned():
    reqs = spec_wl(60, seed=0)
    fb = (GEOM.rowgroup_bytes * GEOM.channels) // (16 * SHAPE.kv_bytes_per_token)
    pa = _run_wl(reqs, "pim-aware", "host-cacheline", fb=fb,
                 spec=SpecParams(), check_every=1)
    pg = _run_wl(reqs, "paged", "host-cacheline", spec=SpecParams())
    assert pa.summary["m1_mean"] > pg.summary["m1_mean"]


# ------------------------------- Phase B: placement diagnostics

from pimkv.sim import placement_diagnostics


def test_placement_diagnostics_perfect_and_scattered():
    perfect = list(range(16))          # one frame of 8, then the next
    d = placement_diagnostics(perfect, frame_blocks=8)
    assert d["blk_adj"] == 1.0
    assert d["same_frame"] == pytest.approx(14 / 15)   # one frame crossing
    assert d["frame_spread"] == 1.0

    # same 16 blocks smeared one-per-frame: worst case
    smeared = [i * 8 for i in range(16)]
    d = placement_diagnostics(smeared, frame_blocks=8)
    assert d["blk_adj"] == 0.0
    assert d["same_frame"] == 0.0
    assert d["frame_spread"] == 8.0                    # 16 frames vs 2 needed


def test_placement_diagnostics_degenerate():
    d = placement_diagnostics([5], frame_blocks=8)
    assert np.isnan(d["blk_adj"])
    d = placement_diagnostics([1, 2, 3], frame_blocks=1)
    assert d["blk_adj"] == 1.0 and np.isnan(d["same_frame"])


# ------------------------------- Phase C: KV-head sharding

from pimkv.config import shard_kv


def test_shard_kv_preserves_per_channel_slice():
    """Proportional sharding scales rowgroup bytes and kv bytes/token by the
    same factor, so the alignment ratio a block sees is invariant."""
    g1, s1 = shard_kv(GEOM, SHAPE, 1)
    ratio1 = (g1.rowgroup_bytes * g1.channels) / (16 * s1.kv_bytes_per_token)
    for n in (2, 4, 8):
        g, s = shard_kv(GEOM, SHAPE, n)
        assert g.channels == GEOM.channels // n
        assert s.kv_heads == SHAPE.kv_heads // n
        ratio = (g.rowgroup_bytes * g.channels) / (16 * s.kv_bytes_per_token)
        assert ratio == pytest.approx(ratio1)


def test_shard_kv_rejects_indivisible():
    with pytest.raises(ValueError):
        shard_kv(GEOM, SHAPE, 16)      # 8 kv_heads cannot split 16 ways


# ------------------------------- Phase D: vLLM admission + preemption

from pimkv.allocator import ContiguousOracle as _CO


def _run_adm(alloc_name, headroom, n=120, seed=0, wl=None, **kw):
    reqs = (wl or steady)(n, seed=seed)
    pool = auto_pool_blocks(reqs, 16, 32, headroom=headroom)
    alloc = make_allocator(alloc_name, pool, seed, frame_blocks=8)
    am = AddrMap(GEOM, "host-cacheline")
    kw.setdefault("sample_every", 8); kw.setdefault("sample_seqs", 4)
    return simulate(reqs, alloc, GEOM, SHAPE, am, block_tokens=16,
                    max_batch=32, seed=seed, admission="vllm",
                    frame_blocks=8, **kw)


def test_vllm_admission_preempts_under_pressure_and_conserves():
    res = _run_adm("paged", headroom=0.6, check_every=1)
    s = res.summary
    assert s["preemptions"] > 0
    assert s["completed"] + s["dropped"] == 120
    assert s["dropped"] == 0


def test_vllm_admission_no_preemption_when_roomy():
    res = _run_adm("paged", headroom=3.0, check_every=1)
    assert res.summary["preemptions"] == 0
    assert res.summary["completed"] == 120


def test_vllm_admission_deterministic():
    a = _run_adm("pim-aware", headroom=0.7, seed=2).df
    b = _run_adm("pim-aware", headroom=0.7, seed=2).df
    assert a.to_csv(index=False) == b.to_csv(index=False)


def test_vllm_admission_spec_and_prefix_conserve():
    r1 = _run_adm("paged", headroom=0.7, wl=spec_wl, spec=SpecParams(),
                  check_every=1)
    assert r1.summary["completed"] + r1.summary["dropped"] == 120
    r2 = _run_adm("pim-aware", headroom=0.7, wl=prefix_wl, check_every=1)
    assert r2.summary["completed"] + r2.summary["dropped"] == 120


def test_vllm_admission_rejects_contiguous():
    reqs = steady(10, seed=0)
    with pytest.raises(ValueError):
        simulate(reqs, _CO(500), GEOM, SHAPE, AddrMap(GEOM, "host-cacheline"),
                 block_tokens=16, admission="vllm")


# ------------------------------- Phase E: trace emission (no Ramulator needed)

def test_ramulator_trace_format(tmp_path):
    from pimkv.ramulator import emit_trace, enumerate_bursts
    am = AddrMap(GEOM, "host-cacheline")
    m = enumerate_bursts([5, 9], 20, GEOM, SHAPE, 16, am)
    n = emit_trace(m, tmp_path / "t.trace")
    lines = (tmp_path / "t.trace").read_text().splitlines()
    assert n == len(lines) == 20 * SHAPE.kv_bytes_per_token // GEOM.burst_bytes
    op, vec = lines[0].split(" ")
    f = [int(x) for x in vec.split(",")]
    assert op == "R" and len(f) == 7
    assert 0 <= f[0] < GEOM.channels // 2 and f[1] in (0, 1) and f[2] == 0
    assert f[3] < GEOM.bank_groups and f[4] < GEOM.banks_per_group
    assert f[5] < GEOM.rows_per_bank and f[6] % 8 == 0 and f[6] < 256
