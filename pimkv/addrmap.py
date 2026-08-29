"""Physical address mapping: linear KV burst address -> (ch, bg, ba, ro, co).

The KV pool is one linear physical region (the serving framework's
preallocated KV tensor). The memory controller scatters it across DRAM
according to an interleaving scheme, expressed here as an ordered list of
bit-fields from LSB to MSB, in units of one burst (``geom.burst_bytes``).
A field may appear twice (split column bits).

Schemes
-------
``host-centric``   ro:bg:ba:ch:co (MSB->LSB) — the conventional map named in
                   AGENT_BRIEF §5.3. All column bits are low, so consecutive
                   addresses in one channel sweep an entire row of ONE bank
                   before touching the next bank: rows stay open but all-bank
                   parallelism collapses (M2 -> ~2/banks for a 64-burst
                   window).
``host-cacheline`` ro:co_hi:bg:ba:ch:co_lo — a realistic host map that
                   interleaves cachelines across channels and puts bank bits
                   just above them, as real memory controllers do to maximize
                   bank-level parallelism for random host traffic.
``pim-friendly``   ch:ro:co:bg:ba — bank bits lowest: banks_per_channel
                   consecutive bursts land in distinct banks at the same
                   (row, col), i.e. exactly one all-bank command. Each channel
                   owns a large contiguous slab (what PIM-attention papers
                   assume).

All sizes are powers of two; mapping is pure bit slicing, so it is exactly
invertible (tested).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DramGeometry

_FIELDS = ("ch", "bg", "ba", "ro", "co")


def _bits(n: int) -> int:
    return int(n).bit_length() - 1  # n is a power of two


def _scheme_fields(geom: DramGeometry, scheme: str) -> list[tuple[str, int]]:
    """Return the bit layout as (field, n_bits) from LSB to MSB."""
    ch, bg, ba = _bits(geom.channels), _bits(geom.bank_groups), _bits(geom.banks_per_group)
    ro, co = _bits(geom.rows_per_bank), _bits(geom.cols_per_row)
    if scheme == "host-centric":
        return [("co", co), ("ch", ch), ("ba", ba), ("bg", bg), ("ro", ro)]
    if scheme == "host-cacheline":
        co_lo = _bits(geom.cacheline_bytes // geom.burst_bytes)
        co_lo = min(co_lo, co)
        return [("co", co_lo), ("ch", ch), ("ba", ba), ("bg", bg),
                ("co", co - co_lo), ("ro", ro)]
    if scheme == "pim-friendly":
        return [("ba", ba), ("bg", bg), ("co", co), ("ro", ro), ("ch", ch)]
    raise KeyError(f"unknown address-map scheme: {scheme!r} "
                   f"(available: {sorted(SCHEMES)})")


SCHEMES = ("host-centric", "host-cacheline", "pim-friendly")


@dataclass
class MappedAddrs:
    """Per-burst DRAM coordinates (all int64 arrays of equal length)."""
    ch: np.ndarray
    bg: np.ndarray
    ba: np.ndarray
    ro: np.ndarray
    co: np.ndarray

    @property
    def bank(self) -> np.ndarray:
        """Flat bank index within the channel (bank group folded in).
        All-bank operation spans all banks_per_channel of these."""
        return self.bg * self._banks_per_group + self.ba

    # set by AddrMap.map
    _banks_per_group: int = 0


class AddrMap:
    def __init__(self, geom: DramGeometry, scheme: str) -> None:
        self.geom = geom
        self.scheme = scheme
        self.fields = _scheme_fields(geom, scheme)
        self.total_bits = sum(b for _, b in self.fields)
        assert 1 << self.total_bits == geom.total_bursts

    def map(self, lin: np.ndarray) -> MappedAddrs:
        """Map linear burst indices (int64 array) to DRAM coordinates."""
        lin = np.asarray(lin, dtype=np.int64)
        if lin.size and (lin.min() < 0 or lin.max() >= self.geom.total_bursts):
            raise ValueError("linear burst index out of range for geometry "
                             f"{self.geom.name} (0..{self.geom.total_bursts - 1})")
        out = {f: np.zeros_like(lin) for f in _FIELDS}
        in_field_shift = {f: 0 for f in _FIELDS}
        shift = 0
        for f, nb in self.fields:
            if nb:
                chunk = (lin >> shift) & ((1 << nb) - 1)
                out[f] |= chunk << in_field_shift[f]
                in_field_shift[f] += nb
                shift += nb
        m = MappedAddrs(**out)
        m._banks_per_group = self.geom.banks_per_group
        return m

    def inverse(self, *, ch, bg, ba, ro, co) -> np.ndarray:
        """Inverse of map(): DRAM coordinates -> linear burst index."""
        vals = {"ch": np.asarray(ch, dtype=np.int64),
                "bg": np.asarray(bg, dtype=np.int64),
                "ba": np.asarray(ba, dtype=np.int64),
                "ro": np.asarray(ro, dtype=np.int64),
                "co": np.asarray(co, dtype=np.int64)}
        lin = np.zeros_like(vals["ch"])
        in_field_shift = {f: 0 for f in _FIELDS}
        shift = 0
        for f, nb in self.fields:
            if nb:
                chunk = (vals[f] >> in_field_shift[f]) & ((1 << nb) - 1)
                lin |= chunk << shift
                in_field_shift[f] += nb
                shift += nb
        return lin
