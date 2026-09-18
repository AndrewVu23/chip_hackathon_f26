"""Tests for pimkv.addrmap.

Uses a deliberately tiny geometry so every expected value below is
hand-computed from the bit layout:

  tiny: channels=2, bank_groups=2, banks_per_group=2, row_bytes=256,
        burst_bytes=32, rows_per_bank=16, cacheline_bytes=64
  -> cols_per_row = 8 (3 bits), 4 banks/channel, total bursts = 1024 (10 bits)

Bit layouts (LSB -> MSB), in burst units:
  pim-friendly   : ba(1) bg(1) co(3) ro(4) ch(1)
  host-centric   : co(3) ch(1) ba(1) bg(1) ro(4)      [ro:bg:ba:ch:co]
  host-cacheline : co(1) ch(1) ba(1) bg(1) co(2) ro(4) [cacheline-interleaved]
"""
import numpy as np
import pytest

from pimkv.addrmap import AddrMap, SCHEMES
from pimkv.config import DramGeometry, HBM3_PIM, GDDR6_PIM

TINY = DramGeometry(name="tiny", channels=2, bank_groups=2, banks_per_group=2,
                    row_bytes=256, burst_bytes=32, rows_per_bank=16,
                    cacheline_bytes=64)


def _map1(am: AddrMap, lin: int):
    m = am.map(np.array([lin], dtype=np.int64))
    return {k: int(getattr(m, k)[0]) for k in ("ch", "bg", "ba", "ro", "co")}


# ---------------------------------------------------------------- spot checks

def test_pim_friendly_spot():
    am = AddrMap(TINY, "pim-friendly")
    assert _map1(am, 0) == dict(ch=0, bg=0, ba=0, ro=0, co=0)
    assert _map1(am, 1) == dict(ch=0, bg=0, ba=1, ro=0, co=0)
    assert _map1(am, 2) == dict(ch=0, bg=1, ba=0, ro=0, co=0)
    assert _map1(am, 4) == dict(ch=0, bg=0, ba=0, ro=0, co=1)
    assert _map1(am, 32) == dict(ch=0, bg=0, ba=0, ro=1, co=0)
    assert _map1(am, 512) == dict(ch=1, bg=0, ba=0, ro=0, co=0)


def test_pim_friendly_first_stripe_is_one_allbank_command():
    """The first banks_per_channel consecutive bursts must land in distinct
    banks at the SAME (channel, row, col) — the definition of PIM-friendliness."""
    am = AddrMap(TINY, "pim-friendly")
    m = am.map(np.arange(TINY.banks_per_channel, dtype=np.int64))
    assert len(set(m.bank.tolist())) == TINY.banks_per_channel
    assert np.all(m.ch == 0) and np.all(m.ro == 0) and np.all(m.co == 0)


def test_host_centric_spot():
    am = AddrMap(TINY, "host-centric")
    assert _map1(am, 0) == dict(ch=0, bg=0, ba=0, ro=0, co=0)
    assert _map1(am, 1) == dict(ch=0, bg=0, ba=0, ro=0, co=1)
    assert _map1(am, 8) == dict(ch=1, bg=0, ba=0, ro=0, co=0)
    assert _map1(am, 16) == dict(ch=0, bg=0, ba=1, ro=0, co=0)
    assert _map1(am, 32) == dict(ch=0, bg=1, ba=0, ro=0, co=0)
    assert _map1(am, 64) == dict(ch=0, bg=0, ba=0, ro=1, co=0)


def test_host_cacheline_spot():
    am = AddrMap(TINY, "host-cacheline")
    assert _map1(am, 0) == dict(ch=0, bg=0, ba=0, ro=0, co=0)
    assert _map1(am, 1) == dict(ch=0, bg=0, ba=0, ro=0, co=1)   # co low chunk
    assert _map1(am, 2) == dict(ch=1, bg=0, ba=0, ro=0, co=0)
    assert _map1(am, 4) == dict(ch=0, bg=0, ba=1, ro=0, co=0)
    assert _map1(am, 8) == dict(ch=0, bg=1, ba=0, ro=0, co=0)
    assert _map1(am, 16) == dict(ch=0, bg=0, ba=0, ro=0, co=2)  # co high chunk
    assert _map1(am, 64) == dict(ch=0, bg=0, ba=0, ro=1, co=0)


# ------------------------------------------------------------- global checks

@pytest.mark.parametrize("geom", [TINY, HBM3_PIM, GDDR6_PIM])
@pytest.mark.parametrize("scheme", sorted(SCHEMES))
def test_bijective(geom, scheme):
    am = AddrMap(geom, scheme)
    rng = np.random.default_rng(0)
    lin = rng.integers(0, geom.total_bursts, size=5000, dtype=np.int64)
    m = am.map(lin)
    back = am.inverse(ch=m.ch, bg=m.bg, ba=m.ba, ro=m.ro, co=m.co)
    np.testing.assert_array_equal(back, lin)


@pytest.mark.parametrize("geom", [TINY, HBM3_PIM])
@pytest.mark.parametrize("scheme", sorted(SCHEMES))
def test_field_bounds(geom, scheme):
    am = AddrMap(geom, scheme)
    rng = np.random.default_rng(1)
    lin = rng.integers(0, geom.total_bursts, size=5000, dtype=np.int64)
    m = am.map(lin)
    assert m.ch.max() < geom.channels and m.ch.min() >= 0
    assert m.bg.max() < geom.bank_groups
    assert m.ba.max() < geom.banks_per_group
    assert m.ro.max() < geom.rows_per_bank
    assert m.co.max() < geom.cols_per_row
    np.testing.assert_array_equal(m.bank, m.bg * geom.banks_per_group + m.ba)


def test_out_of_range_rejected():
    am = AddrMap(TINY, "pim-friendly")
    with pytest.raises(ValueError):
        am.map(np.array([TINY.total_bursts], dtype=np.int64))
    with pytest.raises(ValueError):
        am.map(np.array([-1], dtype=np.int64))


def test_unknown_scheme_rejected():
    with pytest.raises(KeyError):
        AddrMap(TINY, "no-such-scheme")
