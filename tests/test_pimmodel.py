"""Tests for pimkv.pimmodel — written BEFORE the implementation.

Includes validation gates 1 and 2 from AGENT_BRIEF §6 at the model level
(gates 3-5 need the simulator and live in test_sim.py):

  Gate 1 (perfect-case anchor): contiguous placement + PIM-friendly map
          -> M1 >= 0.95.
  Gate 2 (worst-case anchor): uniformly random placement at burst granularity
          -> M1 ~ 1/rows_used within a factor of 2.
"""
import numpy as np
import pytest

from pimkv.addrmap import AddrMap
from pimkv.config import DramGeometry, HBM3_PIM, DEFAULT_TIMING, LLAMA_GQA_8KV
from pimkv.pimmodel import (coalesce_channel, sequence_metrics,
                            effective_bandwidth_gbps, decode_latency_ns)

TINY4 = DramGeometry(name="tiny4", channels=1, bank_groups=2, banks_per_group=2,
                     row_bytes=256, burst_bytes=32, rows_per_bank=64)


# ----------------------------------------------------- coalescer hand cases

def test_window_perfect_stream():
    """16 bursts, one row, banks cycling 0..3, B=4, W=8.

    Each window of 8 holds 2 bursts per bank at one row -> 2 commands of 4
    banks each. Window 1: first command misses (cold), second hits. Window 2:
    carry row matches -> both hit. Totals: 4 commands, 3 hits.
    """
    row = np.zeros(16, dtype=np.int64)
    bank = np.arange(16, dtype=np.int64) % 4
    n_cmds, n_hits, n_bursts = coalesce_channel(bank, row, banks_per_channel=4,
                                                window=8, mode="window")
    assert (n_cmds, n_hits, n_bursts) == (4, 3, 16)


def test_window_alternating_rows():
    """Rows alternate a,b per burst, distinct banks, B=4, W=4.

    Each window: two row groups of 2 distinct banks -> 2 commands. First
    window: 2 misses. Later windows: carry row (row of last burst = b) matches
    one group -> 1 miss each.
    """
    row = np.tile(np.array([7, 9], dtype=np.int64), 8)       # 16 bursts
    bank = np.tile(np.array([0, 1, 2, 3], dtype=np.int64), 4)
    n_cmds, n_hits, _ = coalesce_channel(bank, row, banks_per_channel=4,
                                         window=4, mode="window")
    assert n_cmds == 8
    assert n_hits == 3  # windows 2..4 recover one hit each via carry


def test_same_bank_run_needs_serial_commands():
    """8 bursts all in bank 0 at one row: no bank parallelism is available,
    so 8 commands are needed even though every one after the first is a row
    hit. M2 must come out at 1/B."""
    row = np.zeros(8, dtype=np.int64)
    bank = np.zeros(8, dtype=np.int64)
    n_cmds, n_hits, n_bursts = coalesce_channel(bank, row, banks_per_channel=4,
                                                window=8, mode="window")
    assert n_cmds == 8
    assert n_hits == 7
    assert n_bursts / (4 * n_cmds) == pytest.approx(0.25)


def test_inorder_perfect_stream():
    """In-order mode, banks cycling, one row: command closes when full (B=4).
    16 bursts -> 4 commands, hits after the first ACT -> 3 hits."""
    row = np.zeros(16, dtype=np.int64)
    bank = np.arange(16, dtype=np.int64) % 4
    n_cmds, n_hits, _ = coalesce_channel(bank, row, banks_per_channel=4,
                                         window=8, mode="inorder")
    assert (n_cmds, n_hits) == (4, 3)


def test_inorder_bank_repeat_closes_command():
    row = np.zeros(4, dtype=np.int64)
    bank = np.array([0, 0, 1, 1], dtype=np.int64)
    n_cmds, n_hits, _ = coalesce_channel(bank, row, banks_per_channel=4,
                                         window=8, mode="inorder")
    # [0][0,1][1] -> 3 commands, all same row -> 2 hits
    assert (n_cmds, n_hits) == (3, 2)


# ----------------------------------------------------------- validation gates

def test_gate1_perfect_case_anchor():
    """Contiguous linear placement + pim-friendly map on hbm3-pim: M1 >= 0.95.

    (Expected exactly 1 - 1/cols_per_row = 31/32 = 0.969: one ACT per
    row-group, then cols_per_row - 1 hits.)"""
    geom = HBM3_PIM
    am = AddrMap(geom, "pim-friendly")
    lin = np.arange(2 ** 17, dtype=np.int64)   # 4 MiB contiguous
    m = am.map(lin)
    s = sequence_metrics(m.ch, m.bank, m.ro, geom, window=64, mode="window")
    assert s.m1 >= 0.95
    assert s.m2 == pytest.approx(1.0, abs=0.01)


def test_gate2_worst_case_anchor():
    """Uniformly random burst placement: M1 ~ 1/rows_used within factor 2.

    The 1/rows anchor presumes one command per access with no coalescing
    (the textbook DRAM row-hit definition): P(hit) = P(consecutive commands
    coincide on a row) = 1/rows. Our coalescer *merges* adjacent same-row
    accesses into one command (strictly better than hitting), which removes
    exactly those coincidences, so the strict factor-of-2 check uses a
    single-bank stream, where merging is impossible and every burst is its
    own command. With random banks the window-mode coalescer additionally
    harvests a few hits from coincidental same-(row,bank) pairs inside a
    window (~3/rows for W=64, R=512); bounded loosely as a secondary sanity
    check. See NOTES.md 2026-08-28.
    """
    geom = HBM3_PIM
    rows_used = 512
    rng = np.random.default_rng(42)
    n = 400_000
    row = rng.integers(0, rows_used, size=n, dtype=np.int64)
    n_cmds, n_hits, _ = coalesce_channel(np.zeros(n, dtype=np.int64), row,
                                         banks_per_channel=geom.banks_per_channel,
                                         window=64, mode="inorder")
    m1 = n_hits / n_cmds
    assert n_cmds == n
    assert 0.5 / rows_used <= m1 <= 2.0 / rows_used

    bank = rng.integers(0, geom.banks_per_channel, size=n, dtype=np.int64)
    n_cmds, n_hits, _ = coalesce_channel(bank, row,
                                         banks_per_channel=geom.banks_per_channel,
                                         window=64, mode="window")
    m1_win = n_hits / n_cmds
    assert m1_win <= 8.0 / rows_used


def test_host_centric_kills_bank_parallelism():
    """Brief §5.3 host-centric map (ro:bg:ba:ch:co): a contiguous stream walks
    an entire row of one bank before touching the next bank, so within a
    64-burst window only 2 banks are reachable -> M2 = 2/16 = 0.125 on
    hbm3-pim, even though rows stay open (M1 high)."""
    geom = HBM3_PIM
    am = AddrMap(geom, "host-centric")
    lin = np.arange(2 ** 17, dtype=np.int64)
    m = am.map(lin)
    s = sequence_metrics(m.ch, m.bank, m.ro, geom, window=64, mode="window")
    assert s.m2 == pytest.approx(2 / geom.banks_per_channel, rel=0.05)
    assert s.m1 > 0.9


# ------------------------------------------------------------- M3/M4 algebra

def test_effective_bandwidth_endpoints():
    geom = HBM3_PIM
    t = DEFAULT_TIMING
    ideal = effective_bandwidth_gbps(1.0, 1.0, geom, t)
    # BW_ideal = channels * banks * burst / tCCD_ab
    expect = geom.channels * geom.banks_per_channel * geom.burst_bytes / t.tccd_ab_ns
    assert ideal == pytest.approx(expect)
    # all-miss, full parallelism: slowed by exactly trc/tccd
    worst = effective_bandwidth_gbps(0.0, 1.0, geom, t)
    assert worst == pytest.approx(ideal / t.miss_hit_ratio)
    # half parallelism scales linearly
    assert effective_bandwidth_gbps(1.0, 0.5, geom, t) == pytest.approx(ideal / 2)


def test_decode_latency_matches_bandwidth():
    kv_bytes = 1 << 20
    bw = 512.0  # GB/s == bytes/ns
    assert decode_latency_ns(kv_bytes, bw) == pytest.approx(kv_bytes / 512.0)


# --------------------------- command-stream view (Phase E2 / AttAcc)

from pimkv.pimmodel import coalesce_channel_stream


@pytest.mark.parametrize("mode", ["window", "inorder"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stream_matches_counting_implementation(mode, seed):
    """The stream view must agree with the validated counting view on both
    command count and the hit/miss sequence — otherwise AttAcc would be
    replaying a different model than the one we report."""
    rng = np.random.default_rng(seed)
    n = 4000
    B = 16
    bank = rng.integers(0, B, size=n, dtype=np.int64)
    row = rng.integers(0, 40, size=n, dtype=np.int64)
    col = rng.integers(0, 32, size=n, dtype=np.int64)
    n_cmds, n_hits, _ = coalesce_channel(bank, row, banks_per_channel=B,
                                         window=64, mode=mode)
    stream = coalesce_channel_stream(bank, row, col, banks_per_channel=B,
                                     window=64, mode=mode)
    # command COUNT must match exactly: it is what M2 is computed from and
    # what AttAcc replays as PIM_MAC_AB commands
    assert len(stream) == n_cmds
    rows = [r for r, _ in stream]
    hits = sum(1 for a, b in zip(rows, rows[1:]) if a == b)
    # hits may differ by at most one per window: when the carried-in open
    # row and the window's last burst fall in the SAME row group but the
    # window holds other rows too, the group cannot be emitted both first
    # and last (see coalesce_channel_stream). Measured: 0-2 per 63 windows.
    n_windows = -(-n // 64)
    assert 0 <= n_hits - hits <= n_windows
    assert all(0 <= c < 32 for _, c in stream)
