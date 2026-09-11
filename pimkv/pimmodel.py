"""Analytical PIM execution model: metrics M1-M4 (AGENT_BRIEF §5.4).

Execution model
---------------
A decode step streams a sequence's entire KV cache through the PIM units.
Per channel, the controller walks the sequence's KV bursts in logical token
order and groups them into *all-bank commands*. One all-bank command:

  * targets ONE row index, activated simultaneously in every bank of the
    channel (all-bank ACT), and
  * reads one burst from each participating bank.

Two bursts can share a command iff they are in the same channel, at the same
row index, and in *different* banks. We deliberately do NOT require the same
column index (real all-bank designs broadcast one column; allowing per-bank
columns within an open row is optimistic for the PAGED BASELINE, i.e.
conservative for this project's claim — see NOTES.md).

Reordering is limited to a sliding window of ``window`` bursts per channel
(the PIM controller's buffering). ``mode="inorder"`` (window of 1 command)
is the pessimistic sensitivity case.

Metrics (definitions fixed by AGENT_BRIEF — do not change)
----------------------------------------------------------
M1  row-hit rate      hits / commands, where a command is a hit iff its row
                      is already open. All-bank semantics: the channel has a
                      single open-row state; a command at a different row
                      re-activates ALL banks (partial hits are misses).
M2  bank parallelism  mean over commands of participating_banks / banks_per
                      _channel == total_bursts / (banks_per_channel * total
                      _commands).
M3  effective BW      BW_ideal * M2 * timing_factor, with
                      BW_ideal = channels * banks_per_channel * burst_bytes
                                 / tCCD_ab                    [bytes/ns = GB/s]
                      timing_factor = tCCD_ab /
                                      (M1 * tCCD_ab + (1 - M1) * tRC)
                      i.e. hits cost tCCD_ab, misses cost the full row cycle.
M4  decode latency    kv_bytes_touched / M3                   [ns]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DramGeometry, PimTiming


@dataclass
class SeqMetrics:
    n_bursts: int
    n_cmds: int
    n_hits: int
    m1: float
    m2: float


def coalesce_channel(bank: np.ndarray, row: np.ndarray, *,
                     banks_per_channel: int, window: int = 64,
                     mode: str = "window") -> tuple[int, int, int]:
    """Coalesce one channel's burst stream into all-bank commands.

    Returns (n_cmds, n_hits, n_bursts).

    window mode: within each consecutive window of ``window`` bursts, bursts
    are grouped by row; a row group with at most m bursts per bank needs m
    commands (same-bank bursts serialize; distinct banks ride along).
    Commands for the same row issue back-to-back, so a row group incurs one
    potential miss; the group matching the carried-in open row is scheduled
    first and starts with a hit. The open row carried out is the row of the
    window's last burst (approximation, documented in NOTES.md).

    inorder mode: no reordering; a command accumulates bursts until the row
    changes, a bank repeats, or the command is full.
    """
    bank = np.asarray(bank, dtype=np.int64)
    row = np.asarray(row, dtype=np.int64)
    n = int(row.size)
    if n == 0:
        return 0, 0, 0
    if mode == "inorder":
        return _coalesce_inorder(bank, row, banks_per_channel)
    if mode != "window":
        raise ValueError(f"unknown coalesce mode {mode!r}")

    B = banks_per_channel
    n_cmds = 0
    n_hits = 0
    carry = np.int64(-1)
    for s in range(0, n, window):
        rw = row[s:s + window]
        bw = bank[s:s + window]
        key = rw * B + bw
        ukey, cnt = np.unique(key, return_counts=True)
        urow = ukey // B                       # sorted, grouped by row
        seg = np.flatnonzero(np.r_[True, np.diff(urow) != 0])
        rows_in_win = urow[seg]                # distinct rows, ascending
        m = np.maximum.reduceat(cnt, seg)      # commands needed per row group
        cmds_w = int(m.sum())
        d = len(rows_in_win)
        i = np.searchsorted(rows_in_win, carry)
        carry_present = i < d and rows_in_win[i] == carry
        misses = d - (1 if carry_present else 0)
        n_cmds += cmds_w
        n_hits += cmds_w - misses
        carry = rw[-1]
    return n_cmds, n_hits, n


def _coalesce_inorder(bank: np.ndarray, row: np.ndarray,
                      banks_per_channel: int) -> tuple[int, int, int]:
    n_cmds = 0
    n_hits = 0
    open_row = -1
    cur_row = -1
    cur_banks: set[int] = set()
    for b, r in zip(bank.tolist(), row.tolist()):
        if (r != cur_row or b in cur_banks
                or len(cur_banks) == banks_per_channel):
            if cur_row != -1:                  # issue current command
                n_cmds += 1
                n_hits += cur_row == open_row
                open_row = cur_row
            cur_row = r
            cur_banks = {b}
        else:
            cur_banks.add(b)
    if cur_row != -1:
        n_cmds += 1
        n_hits += cur_row == open_row
    return n_cmds, n_hits, int(row.size)


def sequence_metrics(ch: np.ndarray, bank: np.ndarray, row: np.ndarray,
                     geom: DramGeometry, *, window: int = 64,
                     mode: str = "window") -> SeqMetrics:
    """Aggregate M1/M2 for one sequence's decode-step KV sweep.

    Streams are split per channel (each channel's PIM controller is
    independent and holds its own open-row state); order within a channel is
    preserved. Open-row state starts cold for the sequence (conservative:
    costs at most one extra miss per channel).
    """
    ch = np.asarray(ch, dtype=np.int64)
    n_cmds = n_hits = n_bursts = 0
    for c in np.unique(ch):
        sel = ch == c
        cc, hh, bb = coalesce_channel(bank[sel], row[sel],
                                      banks_per_channel=geom.banks_per_channel,
                                      window=window, mode=mode)
        n_cmds += cc
        n_hits += hh
        n_bursts += bb
    m1 = n_hits / n_cmds if n_cmds else 0.0
    m2 = n_bursts / (geom.banks_per_channel * n_cmds) if n_cmds else 0.0
    return SeqMetrics(n_bursts=n_bursts, n_cmds=n_cmds, n_hits=n_hits,
                      m1=m1, m2=m2)


def effective_bandwidth_gbps(m1: float, m2: float, geom: DramGeometry,
                             timing: PimTiming) -> float:
    """M3. See module docstring for the formula. bytes/ns == GB/s."""
    bw_ideal = (geom.channels * geom.banks_per_channel * geom.burst_bytes
                / timing.tccd_ab_ns)
    t_cmd = m1 * timing.tccd_ab_ns + (1.0 - m1) * timing.trc_ns
    return bw_ideal * m2 * (timing.tccd_ab_ns / t_cmd)


def decode_latency_ns(kv_bytes: float, bw_gbps: float) -> float:
    """M4: attention-portion decode-step latency in ns."""
    return kv_bytes / bw_gbps


def coalesce_channel_stream(bank: np.ndarray, row: np.ndarray,
                            col: np.ndarray, *, banks_per_channel: int,
                            window: int = 64,
                            mode: str = "window") -> list[tuple[int, int]]:
    """Same coalescing as :func:`coalesce_channel`, but returning the
    all-bank command stream as (row, column) pairs in issue order instead
    of only counting it. Used by pimkv.attacc to replay our placements as
    real PIM_MAC_AB commands in AttAcc's Ramulator 2 extension.

    Consistency with the counting implementation (identical command count
    and identical row-transition sequence) is asserted in tests — the
    counting version stays the validated one for all reported metrics.
    """
    bank = np.asarray(bank, dtype=np.int64)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    n = int(row.size)
    out: list[tuple[int, int]] = []
    if n == 0:
        return out
    B = banks_per_channel
    if mode == "inorder":
        open_row = -1
        cur_row = -1
        cur_col = 0
        cur_banks: set[int] = set()
        for b, r, c in zip(bank.tolist(), row.tolist(), col.tolist()):
            if (r != cur_row or b in cur_banks or len(cur_banks) == B):
                if cur_row != -1:
                    out.append((cur_row, cur_col))
                cur_row, cur_col, cur_banks = r, c, {b}
            else:
                cur_banks.add(b)
        if cur_row != -1:
            out.append((cur_row, cur_col))
        return out
    if mode != "window":
        raise ValueError(f"unknown coalesce mode {mode!r}")

    carry = np.int64(-1)
    for s in range(0, n, window):
        rw, bw, cw = row[s:s + window], bank[s:s + window], col[s:s + window]
        key = rw * B + bw
        order = np.argsort(key, kind="stable")
        ukey, idx, cnt = np.unique(key[order], return_index=True,
                                   return_counts=True)
        urow = ukey // B
        seg = np.flatnonzero(np.r_[True, np.diff(urow) != 0])
        rows_in_win = urow[seg]
        m = np.maximum.reduceat(cnt, seg)
        # representative column per row group: first burst of that row
        cols = [int(cw[order[idx[i]]]) for i in seg]
        groups = list(zip(rows_in_win.tolist(), m.tolist(), cols))
        # Emission order must reproduce the counting model's bookkeeping:
        #  * the group matching the carried-IN open row issues first (it is
        #    the one that starts on a hit);
        #  * the group holding the window's LAST burst issues last, so the
        #    carried-OUT open row really is rw[-1] as the counting model
        #    assumes. When those are the same group and the window holds
        #    other rows, both cannot hold; first wins, and the next window
        #    loses at most one hit (bounded by one per window, asserted in
        #    tests).
        last_row = int(rw[-1])
        j = next((k for k, g in enumerate(groups) if g[0] == last_row), None)
        if j is not None and len(groups) > 1:
            groups.append(groups.pop(j))
        i = next((k for k, g in enumerate(groups) if g[0] == int(carry)), None)
        if i is not None:
            groups.insert(0, groups.pop(i))
        for r, mm, c in groups:
            out.extend([(int(r), int(c))] * int(mm))
        carry = rw[-1]
    return out
