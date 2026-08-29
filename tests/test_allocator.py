"""Tests for pimkv.allocator, including the vLLM-port semantics."""
import pytest

from pimkv.allocator import (AdmissionFailure, ContiguousOracle,
                             PagedFirstFit, RandomAlloc)


def test_paged_fresh_pool_hands_out_descending_ids():
    """vLLM v0.2.7 BlockAllocator.allocate() pops from the END of a free list
    initialized in ascending order — a fresh pool yields descending ids."""
    a = PagedFirstFit(8)
    got = a.admit(1, 3, 3)
    assert got == [7, 6, 5]


def test_paged_lifo_reuse():
    a = PagedFirstFit(8)
    a.admit(1, 2, 2)          # [7, 6]
    a.admit(2, 2, 2)          # [5, 4]
    a.release(1)              # free list: [0,1,2,3, 7,6]
    got = a.admit(3, 2, 2)    # LIFO: most recently freed first
    assert got == [6, 7]
    a.assert_conservation()


def test_paged_append_and_conservation():
    a = PagedFirstFit(16)
    a.admit(1, 4, 8)
    for _ in range(4):
        a.append_block(1)
    assert len(a.tables[1]) == 8
    a.assert_conservation()
    a.release(1)
    a.assert_conservation()
    assert a.num_free == 16


def test_contiguous_span_is_contiguous_and_lowest():
    a = ContiguousOracle(32)
    got = a.admit(1, 3, 10)
    assert got == [0, 1, 2]
    for _ in range(7):
        a.append_block(1)
    assert a.tables[1] == list(range(10))
    got2 = a.admit(2, 2, 4)
    assert got2 == [10, 11]
    a.assert_conservation()


def test_contiguous_release_merges_extents():
    a = ContiguousOracle(30)
    a.admit(1, 10, 10)
    a.admit(2, 10, 10)
    a.admit(3, 10, 10)
    a.release(2)
    with pytest.raises(AdmissionFailure):
        a.admit(4, 1, 11)     # only a 10-hole exists
    a.release(1)              # merges [0,10)+[10,10) -> [0,20)
    got = a.admit(5, 1, 20)
    assert got == [0]
    a.assert_conservation()


def test_contiguous_fragmentation_fails_loudly():
    a = ContiguousOracle(10)
    a.admit(1, 1, 4)
    a.admit(2, 1, 4)
    a.release(1)              # holes: [0,4) and [8,10)
    with pytest.raises(AdmissionFailure):
        a.admit(3, 1, 5)


def test_random_is_deterministic_per_seed():
    a = RandomAlloc(64, seed=3)
    b = RandomAlloc(64, seed=3)
    assert a.admit(1, 10, 10) == b.admit(1, 10, 10)
    a.assert_conservation()


def test_double_admit_rejected():
    a = PagedFirstFit(8)
    a.admit(1, 1, 1)
    with pytest.raises(ValueError):
        a.admit(1, 1, 1)
