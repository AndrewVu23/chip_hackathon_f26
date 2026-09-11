"""KV block allocation policies (AGENT_BRIEF §5.2).

Sequence-level interface driven by the simulator:

    admit(seq_id, n_prompt_blocks, n_total_blocks) -> prompt block ids
    admit_shared(seq_id, shared, n_new, n_total)   -> prefix sharing (ref++)
    append_block(seq_id) -> new block id (decode-time growth)
    alloc_scratch(seq_id, n) -> ref-counted blocks outside any table
                                (speculative branch tails)
    spec_round_adopt(...)     -> commit one branch, free the rest
    fork(parent_id, child_id) -> share the whole table copy-on-write style
    release(seq_id)           -> drop the sequence's refs (frees refcount-0)

Reference counting / copy-on-write semantics are ported from vLLM v0.2.7,
commit 2e0b6e775756345aa1d39f772c186e00f8c29e92, vllm/core/block_manager.py:
``free()`` decrements and only returns a block at refcount 0 (lines 46-51),
``fork()`` shares the parent's table and increments every block (182-188),
and a write to a shared last block allocates a fresh block and drops one
ref on the old (append_slot, 168-180) — in this harness that copy shows up
as a speculative branch's private tail copy. Every allocator keeps
conservation counters for validation gate 5.
"""
from __future__ import annotations

import heapq

import numpy as np


class AdmissionFailure(Exception):
    """Raised when a policy cannot place the request right now (e.g. no
    contiguous extent large enough). The simulator retries later."""


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class KVAllocator:
    """Bookkeeping base class: block tables + reference counts. Subclasses
    implement _alloc_one/_free_one (placement policy), or override the
    span-based paths (ContiguousOracle)."""

    name = "base"
    supports_sharing = True           # prefix-cache block sharing
    persistent_scratch = False        # spec branches live in a fixed region

    def __init__(self, num_blocks: int, seed: int = 0) -> None:
        self.num_blocks = num_blocks
        self.tables: dict[int, list[int]] = {}
        self.refcount: dict[int, int] = {}
        self.allocated_total = 0      # physical block allocations
        self.freed_total = 0          # physical frees (refcount hit 0)
        self.cow_copies = 0           # branch-tail copies performed

    # -- policy hooks -------------------------------------------------------
    def _alloc_one(self, seq_id: int) -> int:
        raise NotImplementedError

    def _free_one(self, block: int) -> None:
        raise NotImplementedError

    @property
    def num_free(self) -> int:
        raise NotImplementedError

    # -- ref-counted primitives --------------------------------------------
    def _alloc(self, seq_id: int) -> int:
        b = self._alloc_one(seq_id)
        self.refcount[b] = 1
        self.allocated_total += 1
        return b

    def _unref(self, block: int) -> None:
        rc = self.refcount[block] - 1
        if rc == 0:
            del self.refcount[block]
            self._free_one(block)
            self.freed_total += 1
        else:
            self.refcount[block] = rc

    # -- sequence-level interface ------------------------------------------
    def admit(self, seq_id: int, n_prompt_blocks: int,
              n_total_blocks: int) -> list[int]:
        if seq_id in self.tables:
            raise ValueError(f"seq {seq_id} already admitted")
        if self.num_free < n_prompt_blocks:
            raise AdmissionFailure
        blocks = [self._alloc(seq_id) for _ in range(n_prompt_blocks)]
        self.tables[seq_id] = blocks
        return list(blocks)

    def admit_shared(self, seq_id: int, shared: list[int], n_new: int,
                     n_total_blocks: int) -> list[int]:
        """Admit with a shared full-block prefix (prefix cache hit):
        the shared blocks get one more reference, only the unique remainder
        is allocated. Policies that cannot share (contiguous) override."""
        if seq_id in self.tables:
            raise ValueError(f"seq {seq_id} already admitted")
        if self.num_free < n_new:
            raise AdmissionFailure
        for b in shared:
            self.refcount[b] += 1
        blocks = list(shared) + [self._alloc(seq_id) for _ in range(n_new)]
        self.tables[seq_id] = blocks
        return list(blocks)

    def append_block(self, seq_id: int) -> int:
        if self.num_free < 1:
            raise AdmissionFailure
        b = self._alloc(seq_id)
        self.tables[seq_id].append(b)
        return b

    def fork(self, parent_id: int, child_id: int) -> None:
        """vLLM fork: child shares the parent's whole table (no allocation)."""
        table = self.tables[parent_id]
        for b in table:
            self.refcount[b] += 1
        self.tables[child_id] = list(table)

    # -- speculative decoding support --------------------------------------
    def alloc_scratch(self, seq_id: int, n: int) -> list[int]:
        """Ref-counted blocks owned by the caller, outside any table
        (a draft branch's private tail: CoW copy + overflow blocks)."""
        if self.num_free < n:
            raise AdmissionFailure
        return [self._alloc(seq_id) for _ in range(n)]

    def unref_blocks(self, blocks: list[int]) -> None:
        for b in blocks:
            self._unref(b)

    def spec_round_adopt(self, seq_id: int, replace_tail: bool,
                         chosen: list[int], keep: int,
                         branches: list[list[int]]) -> None:
        """Commit one draft branch: splice ``chosen[:keep]`` onto the table
        (dropping the old shared tail block if the branch copied it), free
        everything else (losing branches and the chosen branch's unused
        overflow)."""
        t = self.tables[seq_id]
        if replace_tail:
            self._unref(t.pop())
            self.cow_copies += 1
        t.extend(chosen[:keep])
        self.unref_blocks(chosen[keep:])
        for br in branches:
            if br is not chosen:
                self.unref_blocks(br)

    # -- teardown / inspection ---------------------------------------------
    def release(self, seq_id: int) -> None:
        for b in self.tables.pop(seq_id):
            self._unref(b)

    def get_table(self, seq_id: int) -> list[int]:
        return self.tables[seq_id]

    # -- validation gate 5 --------------------------------------------------
    @property
    def num_live(self) -> int:
        """Physical blocks currently allocated (shared blocks count once)."""
        return len(self.refcount)

    def assert_conservation(self) -> None:
        live = self.num_live
        assert self.allocated_total - self.freed_total == live, (
            f"{self.name}: allocated {self.allocated_total} - freed "
            f"{self.freed_total} != live {live}")
        assert live + self.num_free == self.num_blocks, (
            f"{self.name}: live {live} + free {self.num_free} != "
            f"pool {self.num_blocks}")


class PagedFirstFit(KVAllocator):
    """vLLM-semantics baseline.

    Ported from vLLM v0.2.7, commit 2e0b6e775756345aa1d39f772c186e00f8c29e92,
    vllm/core/block_manager.py::BlockAllocator:

      * free list initialized with all blocks in ascending id order
        (__init__, lines 32-37);
      * allocate() = ``self.free_blocks.pop()`` — LIFO from the tail
        (line 42); a fresh pool hands out DESCENDING ids;
      * free() appends back (line 51), at refcount 0 only (46-50).

    Fidelity note: real vLLM frees a finished sequence via
    ``_free_block_table`` iterating ``set(block_table)`` (line 266,
    unordered); we free in block-table order for determinism (gate 4).
    """

    name = "paged"

    def __init__(self, num_blocks: int, seed: int = 0) -> None:
        super().__init__(num_blocks, seed)
        self.free_blocks: list[int] = list(range(num_blocks))

    def _alloc_one(self, seq_id: int) -> int:
        return self.free_blocks.pop()

    def _free_one(self, block: int) -> None:
        self.free_blocks.append(block)

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)


class RandomAlloc(KVAllocator):
    """Uniformly random placement — validation-gate-2 reference, not a real
    policy. Deterministic per seed."""

    name = "random"

    def __init__(self, num_blocks: int, seed: int = 0) -> None:
        super().__init__(num_blocks, seed)
        self.free_blocks: list[int] = list(range(num_blocks))
        self.rng = np.random.default_rng(seed)

    def _alloc_one(self, seq_id: int) -> int:
        i = int(self.rng.integers(len(self.free_blocks)))
        self.free_blocks[i], self.free_blocks[-1] = (self.free_blocks[-1],
                                                     self.free_blocks[i])
        return self.free_blocks.pop()

    def _free_one(self, block: int) -> None:
        self.free_blocks.append(block)

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)


class PimAware(KVAllocator):
    """The contribution (AGENT_BRIEF §5.2): alignment-preserving placement.

    The pool is carved into fixed *frames* of ``blocks_per_frame`` blocks,
    where one frame's linear extent is exactly one row index across every
    bank of every channel (``rowgroup_bytes * channels`` — the all-bank
    alignment quantum; computed by the caller from geometry, never
    hardcoded). Policy:

      * a sequence keeps *frame affinity*: consecutive blocks fill its
        current frame at ascending offsets, so per-channel slices
        concatenate into whole row-groups;
      * when the frame fills, the lowest-indexed EMPTY frame is claimed
        (keeps frames sequence-pure); only when no empty frame exists does
        it fall back to the partial frame with most free slots;
      * a freed block returns to its originating frame (frames are fixed
        regions), so alignment survives reuse — a later sequence refills
        the frame from aligned offsets.

    Opportunistic compaction (brief: optional, behind a flag) is not
    implemented; ``cow_copies``/copied-bytes accounting exists so its cost
    could be measured. Derived block size lives in
    ``config.derive_block_tokens``.
    """

    name = "pim-aware"

    def __init__(self, num_blocks: int, seed: int = 0, *,
                 blocks_per_frame: int = 1) -> None:
        super().__init__(num_blocks, seed)
        if num_blocks % blocks_per_frame:
            raise ValueError("num_blocks must be a multiple of blocks_per_frame")
        self.G = blocks_per_frame
        self.n_frames = num_blocks // blocks_per_frame
        # per-frame min-heap of free offsets (ascending allocation)
        self.frame_free: list[list[int]] = [list(range(self.G))
                                            for _ in range(self.n_frames)]
        self.empty_frames: list[int] = list(range(self.n_frames))  # min-heap
        heapq.heapify(self.empty_frames)
        self.affinity: dict[int, int] = {}
        self._nfree = num_blocks

    @property
    def num_free(self) -> int:
        return self._nfree

    def _pick_frame(self, seq_id: int) -> int:
        f = self.affinity.get(seq_id)
        if f is not None and self.frame_free[f]:
            return f
        while self.empty_frames:
            f = heapq.heappop(self.empty_frames)
            if len(self.frame_free[f]) == self.G:   # still actually empty
                self.affinity[seq_id] = f
                return f
        # fallback: partial frame with the most free slots (lowest index ties)
        best, best_free = -1, 0
        for f in range(self.n_frames):
            n = len(self.frame_free[f])
            if n > best_free:
                best, best_free = f, n
        if best < 0:
            raise AdmissionFailure("pim-aware: pool exhausted")
        self.affinity[seq_id] = best
        return best

    def _alloc_one(self, seq_id: int) -> int:
        f = self._pick_frame(seq_id)
        off = heapq.heappop(self.frame_free[f])
        self._nfree -= 1
        return f * self.G + off

    def _free_one(self, block: int) -> None:
        f, off = divmod(block, self.G)
        heapq.heappush(self.frame_free[f], off)
        self._nfree += 1
        if len(self.frame_free[f]) == self.G:
            heapq.heappush(self.empty_frames, f)

    def release(self, seq_id: int) -> None:
        super().release(seq_id)
        self.affinity.pop(seq_id, None)


class ContiguousOracle(KVAllocator):
    """Upper-bound reference: each sequence occupies one physically
    contiguous span, reserved in full at admission (oracle knowledge of the
    final length). First-fit lowest-address over free extents; extents merge
    on release.

    Cannot share prefix blocks (a span belongs to one sequence) —
    ``admit_shared`` falls back to a full private copy, which is reported as
    this policy's memory cost, not hidden. Speculative branches use a
    per-sequence reserved scratch region at the end of the span
    (``persistent_scratch``): zero placement churn, accepted tokens are
    "copied" into the span (copy bytes counted by the simulator) — the
    idealized upper bound.

    Expected to fail under fragmentation (AdmissionFailure, counted): that
    is a finding, not a bug — it is why paging exists.
    """

    name = "contiguous"
    supports_sharing = False
    persistent_scratch = True

    def __init__(self, num_blocks: int, seed: int = 0) -> None:
        super().__init__(num_blocks, seed)
        self.extents: list[list[int]] = [[0, num_blocks]]  # [start, size]
        self.reserved: dict[int, list[int]] = {}           # unconsumed tail
        self.spans: dict[int, tuple[int, int]] = {}

    @property
    def num_free(self) -> int:
        return (sum(sz for _, sz in self.extents)
                + sum(len(r) for r in self.reserved.values()))

    def admit(self, seq_id: int, n_prompt_blocks: int,
              n_total_blocks: int) -> list[int]:
        if seq_id in self.tables:
            raise ValueError(f"seq {seq_id} already admitted")
        for ext in self.extents:
            if ext[1] >= n_total_blocks:
                start = ext[0]
                ext[0] += n_total_blocks
                ext[1] -= n_total_blocks
                if ext[1] == 0:
                    self.extents.remove(ext)
                span = list(range(start, start + n_total_blocks))
                self.spans[seq_id] = (start, n_total_blocks)
                head = span[:n_prompt_blocks]
                for b in head:
                    self.refcount[b] = 1
                self.allocated_total += n_prompt_blocks
                self.tables[seq_id] = head
                self.reserved[seq_id] = span[n_prompt_blocks:]
                return list(head)
        raise AdmissionFailure(f"no contiguous extent of {n_total_blocks}")

    def admit_shared(self, seq_id: int, shared: list[int], n_new: int,
                     n_total_blocks: int) -> list[int]:
        # no sharing possible: allocate the full footprint privately
        n_prompt = len(shared) + n_new
        return self.admit(seq_id, n_prompt, n_total_blocks)

    def append_block(self, seq_id: int) -> int:
        if not self.reserved[seq_id]:
            raise AdmissionFailure(f"seq {seq_id} outgrew its reservation")
        b = self.reserved[seq_id].pop(0)
        self.refcount[b] = 1
        self.allocated_total += 1
        self.tables[seq_id].append(b)
        return b

    def take_scratch(self, seq_id: int, n: int) -> list[int]:
        """Claim the LAST n reserved blocks as the permanent spec-scratch
        region (highest addresses of the span)."""
        r = self.reserved[seq_id]
        if len(r) < n:
            raise AdmissionFailure("reservation too small for scratch")
        scratch = r[len(r) - n:]
        del r[len(r) - n:]
        for b in scratch:
            self.refcount[b] = 1
        self.allocated_total += n
        return scratch

    def release(self, seq_id: int) -> None:
        for b in self.tables.pop(seq_id):
            self._unref_span_block(b)
        self.reserved.pop(seq_id)
        start, size = self.spans.pop(seq_id)
        self._return_extent(start, size)

    def unref_blocks(self, blocks: list[int]) -> None:
        for b in blocks:
            self._unref_span_block(b)

    def _return_extent(self, start: int, size: int) -> None:
        self.extents.append([start, size])
        self.extents.sort()
        merged: list[list[int]] = []
        for s, sz in self.extents:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1][1] += sz
            else:
                merged.append([s, sz])
        self.extents = merged

    def _unref_span_block(self, b: int) -> None:
        # span blocks are never shared; the physical space returns via the
        # extent merge in release(), so only counters/refcount update here
        rc = self.refcount[b] - 1
        if rc == 0:
            del self.refcount[b]
            self.freed_total += 1
        else:  # pragma: no cover - spans are unshared
            self.refcount[b] = rc

    def assert_conservation(self) -> None:
        live = self.num_live
        assert self.allocated_total - self.freed_total == live
        # released spans return whole extents; scratch/table blocks of live
        # seqs are the refcounted set
        assert live + self.num_free == self.num_blocks


ALLOCATORS = ("paged", "random", "contiguous", "pim-aware")


def make_allocator(name: str, num_blocks: int, seed: int, *,
                   frame_blocks: int = 1) -> KVAllocator:
    """Factory. ``frame_blocks`` (pim-aware only) = blocks per alignment
    frame; the caller derives it from geometry (see run.py) and pads
    num_blocks to a multiple."""
    if name == "paged":
        return PagedFirstFit(num_blocks, seed)
    if name == "random":
        return RandomAlloc(num_blocks, seed)
    if name == "contiguous":
        return ContiguousOracle(num_blocks, seed)
    if name == "pim-aware":
        num_blocks = _ceil_div(num_blocks, frame_blocks) * frame_blocks
        return PimAware(num_blocks, seed, blocks_per_frame=frame_blocks)
    raise KeyError(name)
