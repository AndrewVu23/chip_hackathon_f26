"""KV block allocation policies (AGENT_BRIEF §5.2).

All allocators share one sequence-level interface driven by the simulator:

    admit(seq_id, n_prompt_blocks, n_total_blocks) -> list of prompt block ids
    append_block(seq_id) -> one new block id (decode-time growth)
    release(seq_id)      -> free everything the sequence holds
    fork(...)            -> Phase 1 (prefix sharing / speculative trees)

``n_total_blocks`` is an admission-control hint (the simulator knows final
lengths); only ContiguousOracle uses it for placement. Every allocator keeps
conservation counters for validation gate 5.
"""
from __future__ import annotations

import numpy as np


class AdmissionFailure(Exception):
    """Raised when a policy cannot place the request right now (e.g. no
    contiguous extent large enough). The simulator retries later."""


class KVAllocator:
    """Bookkeeping base class. Subclasses implement _alloc_one/_free_one,
    or override admit/release for span-based policies."""

    name = "base"

    def __init__(self, num_blocks: int, seed: int = 0) -> None:
        self.num_blocks = num_blocks
        self.tables: dict[int, list[int]] = {}
        self.allocated_total = 0
        self.freed_total = 0

    # -- policy hooks -------------------------------------------------------
    def _alloc_one(self) -> int:
        raise NotImplementedError

    def _free_one(self, block: int) -> None:
        raise NotImplementedError

    @property
    def num_free(self) -> int:
        raise NotImplementedError

    # -- sequence-level interface ------------------------------------------
    def admit(self, seq_id: int, n_prompt_blocks: int,
              n_total_blocks: int) -> list[int]:
        if seq_id in self.tables:
            raise ValueError(f"seq {seq_id} already admitted")
        if self.num_free < n_prompt_blocks:
            raise AdmissionFailure
        blocks = [self._alloc_one() for _ in range(n_prompt_blocks)]
        self.allocated_total += n_prompt_blocks
        self.tables[seq_id] = blocks
        return list(blocks)

    def append_block(self, seq_id: int) -> int:
        if self.num_free < 1:
            raise AdmissionFailure
        b = self._alloc_one()
        self.allocated_total += 1
        self.tables[seq_id].append(b)
        return b

    def release(self, seq_id: int) -> None:
        blocks = self.tables.pop(seq_id)
        for b in blocks:                    # freed in block-table order
            self._free_one(b)
        self.freed_total += len(blocks)

    # -- validation gate 5 --------------------------------------------------
    @property
    def num_live(self) -> int:
        return sum(len(t) for t in self.tables.values())

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

      * the free list is initialized with all physical blocks in ascending
        id order (``__init__``, lines 32-37);
      * ``allocate()`` takes ``self.free_blocks.pop()`` — the LAST element,
        i.e. a LIFO stack (line 42). Note: on a fresh pool this hands out
        DESCENDING block ids;
      * ``free()`` appends the block back (line 51).

    Fidelity note: real vLLM frees a finished sequence via
    ``_free_block_table``, which iterates ``set(block_table)`` (line 266) —
    an unordered set. We free in block-table (logical) order instead, for
    determinism; recorded in NOTES.md / LIMITATIONS. Ref-count / CoW
    semantics arrive with fork() in Phase 1.
    """

    name = "paged"

    def __init__(self, num_blocks: int, seed: int = 0) -> None:
        super().__init__(num_blocks, seed)
        self.free_blocks: list[int] = list(range(num_blocks))

    def _alloc_one(self) -> int:
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

    def _alloc_one(self) -> int:
        i = int(self.rng.integers(len(self.free_blocks)))
        self.free_blocks[i], self.free_blocks[-1] = (self.free_blocks[-1],
                                                     self.free_blocks[i])
        return self.free_blocks.pop()

    def _free_one(self, block: int) -> None:
        self.free_blocks.append(block)

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)


class ContiguousOracle(KVAllocator):
    """Upper-bound reference: each sequence occupies one physically
    contiguous span, reserved in full at admission (the simulator's oracle
    knowledge of the final length stands in for a reservation-based serving
    policy). First-fit lowest-address over a free-extent list; extents merge
    on release.

    Expected to fail under fragmentation/load — that is a finding, not a bug
    (it is why paging exists); failures surface as AdmissionFailure and are
    counted by the simulator.
    """

    name = "contiguous"

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
                self.tables[seq_id] = span[:n_prompt_blocks]
                self.reserved[seq_id] = span[n_prompt_blocks:]
                self.allocated_total += n_prompt_blocks
                return span[:n_prompt_blocks]
        raise AdmissionFailure(f"no contiguous extent of {n_total_blocks}")

    def append_block(self, seq_id: int) -> int:
        if not self.reserved[seq_id]:
            raise AdmissionFailure(f"seq {seq_id} outgrew its reservation")
        b = self.reserved[seq_id].pop(0)
        self.allocated_total += 1
        self.tables[seq_id].append(b)
        return b

    def release(self, seq_id: int) -> None:
        blocks = self.tables.pop(seq_id)
        self.reserved.pop(seq_id)
        start, size = self.spans.pop(seq_id)
        self.freed_total += len(blocks)
        self._return_extent(start, size)

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

    def assert_conservation(self) -> None:
        live = self.num_live
        assert self.allocated_total - self.freed_total == live
        assert live + self.num_free == self.num_blocks


class PimAware(KVAllocator):
    """Phase 2 — the contribution. Design constraints (AGENT_BRIEF §5.2):

    * blocks come from bank-group-aligned buddy pools, so a block's physical
      extent covers the same row index across all banks of a group;
    * block size in tokens is DERIVED (config.derive_block_tokens), never
      hardcoded;
    * freed blocks return to their originating aligned pool;
    * optional opportunistic compaction behind a flag, cost measured in
      block-copy bytes.
    """

    name = "pim-aware"

    def __init__(self, num_blocks: int, seed: int = 0) -> None:
        raise NotImplementedError("PimAware allocator is Phase 2")


ALLOCATORS: dict[str, type[KVAllocator]] = {
    "paged": PagedFirstFit,
    "random": RandomAlloc,
    "contiguous": ContiguousOracle,
    "pim-aware": PimAware,
}
