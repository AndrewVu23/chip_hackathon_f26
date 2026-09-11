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


# --------------------------------------------------- Phase 1: fork/CoW/refs

def test_fork_shares_blocks_and_release_keeps_them():
    a = PagedFirstFit(16)
    a.admit(1, 3, 3)
    a.fork(1, 2)
    assert a.tables[2] == a.tables[1]
    assert all(a.refcount[b] == 2 for b in a.tables[1])
    live_before = a.num_live
    a.release(1)                  # child still holds refs
    assert a.num_live == live_before
    a.assert_conservation()
    a.release(2)
    assert a.num_live == 0
    a.assert_conservation()


def test_admit_shared_prefix():
    a = PagedFirstFit(32)
    cache = a.admit(-1, 4, 4)     # pinned prefix cache
    t1 = a.admit_shared(1, cache, 2, 8)
    t2 = a.admit_shared(2, cache, 2, 8)
    assert t1[:4] == cache and t2[:4] == cache
    assert all(a.refcount[b] == 3 for b in cache)
    assert a.num_live == 4 + 2 + 2   # shared counted once
    a.release(1); a.release(2); a.release(-1)
    assert a.num_live == 0
    a.assert_conservation()


def test_spec_round_adopt_accounting():
    a = PagedFirstFit(64)
    a.admit(1, 4, 8)              # partial-tail situation is the caller's call
    branches = [a.alloc_scratch(1, 2) for _ in range(3)]
    a.spec_round_adopt(1, replace_tail=True, chosen=branches[1], keep=1,
                       branches=branches)
    # old tail freed, one branch block adopted, 5 branch blocks freed
    assert len(a.tables[1]) == 4
    assert a.num_live == 4
    assert a.cow_copies == 1
    a.assert_conservation()


# ---------------------------------------------------- Phase 2: PimAware

from pimkv.allocator import PimAware, make_allocator


def test_pim_aware_fills_frames_ascending_and_sequence_pure():
    a = PimAware(64, blocks_per_frame=8)
    t1 = a.admit(1, 12, 12)       # 1.5 frames
    assert t1 == list(range(0, 12))     # frame 0 asc, then frame 1
    t2 = a.admit(2, 4, 4)
    # seq 2 must NOT squat in seq 1's half-used frame while empty frames exist
    assert t2 == list(range(16, 20))
    a.assert_conservation()


def test_pim_aware_alignment_survives_reuse():
    a = PimAware(32, blocks_per_frame=8)
    a.admit(1, 8, 8)              # frame 0
    a.admit(2, 8, 8)              # frame 1
    a.release(1)                  # frame 0 empties
    t3 = a.admit(3, 8, 8)
    assert t3 == list(range(0, 8))      # refilled from aligned offsets
    a.assert_conservation()


def test_pim_aware_fallback_to_partial_frames():
    a = PimAware(16, blocks_per_frame=8)
    a.admit(1, 6, 6)
    a.admit(2, 6, 6)              # takes frame 1
    t3 = a.admit(3, 4, 4)         # no empty frame left: spills into partials
    assert len(t3) == 4
    a.assert_conservation()


def test_make_allocator_pads_pool_to_frames():
    a = make_allocator("pim-aware", 30, 0, frame_blocks=8)
    assert a.num_blocks == 32


def test_pim_aware_scratch_segregation_keeps_frames_pure():
    a = PimAware(64, blocks_per_frame=8, segregate_scratch=True)
    a.admit(1, 4, 4)                       # frame 0: offsets 0..3
    br = a.alloc_scratch(1, 4)             # must NOT land in frame 0
    assert all(b // 8 != 0 for b in br)
    a.unref_blocks(br)
    assert a.append_block(1) == 4          # sequence continues contiguously
    a.assert_conservation()
    b = PimAware(64, blocks_per_frame=8, segregate_scratch=False)
    b.admit(1, 4, 4)
    assert all(x // 8 == 0 for x in b.alloc_scratch(1, 4))   # old behaviour
