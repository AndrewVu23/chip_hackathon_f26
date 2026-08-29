# NOTES — design decisions and surprising numbers

Chronological log, one dated entry per decision or finding, kept for the
writeup. Anything that diverges from real hardware or real vLLM also gets a
line in README.md § Limitations.

## 2026-08-28 — toolchain

- Homebrew `python@3.14` on this Mac is broken (its `pyexpat` links a newer
  libexpat symbol than `/usr/lib/libexpat.1.dylib` provides, so `ensurepip`
  dies). Using **python3.13 via `uv venv`** instead; `make install`.
- The shell profile exports a self-referencing `CXXFLAGS`, which breaks any
  CMake/Make build ("Recursive variable ... references itself"). All C++
  builds (Ramulator 2) run with `env -u CXXFLAGS -u CFLAGS -u LDFLAGS`.
- Pinned deps of record: numpy 2.5.2, pandas 3.0.5 (see uv lock behavior;
  exact versions matter only for CSV byte-determinism across machines).
- Cloned + built `third_party/ramulator2` (commit `9ac28d3`, Phase 3) and
  cloned `third_party/attacc_simulator` (commit `c600051`, reference only).

## 2026-08-28 — vLLM port

- Ported from **vLLM v0.2.7, commit 2e0b6e775756345aa1d39f772c186e00f8c29e92,
  `vllm/core/block_manager.py`** (vendored in `third_party/vllm_ref/`).
  `BlockAllocator`: free list initialized ascending, `allocate()` = `pop()`
  from the tail (**LIFO**), `free()` appends. Consequence worth remembering:
  a fresh pool hands out **descending** block ids, so even the very first
  sequences are reverse-ordered in physical space.
- Fidelity gap (logged): real vLLM frees a finished sequence by iterating
  `set(block_table)` — unordered. We free in block-table order for
  determinism (gate 4). This slightly *tidies* the baseline's free list, so
  if anything it flatters the paged baseline.
- CoW/fork/prefix-hash semantics are in the source we vendored but land in
  Phase 1 together with the `prefix`/`spec` workloads.

## 2026-08-28 — modeling decisions (the load-bearing ones)

1. **One representative layer.** vLLM keeps one KV tensor per layer but a
   single block table per sequence, so every layer sees the same inter-block
   placement. Metrics from one layer generalize; absolute bytes scale by
   layer count. (Limitation.)
2. **Within-block layout is the accelerator's problem; the allocator owns
   inter-block placement.** The PIM stack writes K/V at decode time and can
   arrange bytes inside a block optimally, so we model a block's bytes as
   linearly contiguous and measure only what the allocator controls: which
   physical blocks. This is the correct abstraction boundary for an
   allocator study. (Limitation: co-designed within-block layouts could
   shift absolute numbers, not the comparison.)
3. **Column-relaxed all-bank commands.** A command = one row index, distinct
   banks, same channel; we do NOT require one broadcast column (real HBM-PIM
   broadcasts one column). Relaxing columns can only help the scattered
   baseline, i.e. it is conservative for our claim.
4. **Controller reorder window** (default 64 bursts/channel) models PIM-side
   buffering; `--coalesce inorder` is the pessimistic end. This is a
   sensitivity axis to sweep in Phase 2, not a tuned constant.
5. **Open-row state starts cold per sampled sequence** (batch interleaving
   between sequences is not modeled). Costs ≤1 extra miss per channel per
   sample — negligible for the sequence lengths involved; direction is
   against the good cases, so it is conservative for the fix's measured
   *recovery*, and against the baseline by the same tiny amount.
6. **Admission control is an oracle** (final lengths known; a request is
   admitted only if its full footprint fits on top of committed growth).
   This removes vLLM's watermark/preemption/swap machinery from scope, and
   is identical across policies, so comparisons stay fair. (Limitation:
   real preemption would *churn the free list harder* — our baseline is the
   gentle version.)
7. **Timing:** only the tRC/tCCD_ab ratio matters to M3; set 45/4.3 ≈ 10.5
   per the published AttAcc/Ramulator2 measurement (nRC 10–11× nCCDAB).
   Absolute ns are indicative until Phase 3 cross-validation.
8. **HBM3-PIM preset:** 32 pseudo-channels × 16 banks × 1 KB rows × 32 B
   bursts, all-bank domain = one pseudo-channel. Model shape: GQA 8 KV
   heads × 128 dim × fp16 → 4096 B/token/layer → a 16-token block is 64 KB,
   which interleaves to a 2 KB slice per pseudo-channel = **1/8 of a
   row-group**. That fraction is the geometric root of the whole problem.

## 2026-08-28 — validation-gate interpretations (flagged to owner)

- **Gate 2 (random ⇒ M1 ≈ 1/rows, factor 2).** Two subtleties found while
  making it pass honestly:
  (a) a coalescing controller *merges* adjacent same-row accesses into one
  command — removing exactly the coincidences the 1/rows heuristic counts as
  hits — so the strict factor-2 check is run on a single-bank stream, where
  merging is impossible and every burst is one command (measured: within
  factor 2 of 1/512 ✓);
  (b) with random banks and a 64-burst window, coincidental same-(row,bank)
  pairs legitimately yield ~3/rows — bounded as a looser secondary check
  (≤ 8/rows). Test docstring documents both. **Owner sign-off wanted** since
  the brief says not to reinterpret gates silently.
- **Gate 3 (monotonic in block size, contiguous).** For a contiguous
  allocator M1 is essentially block-size-invariant (a contiguous span is
  contiguous at any granularity), so the gate degenerates to "no aliasing
  bug makes it *decrease*". Passes.
- Gate 1 passes at 0.969 = 1 − 1/cols_per_row (one ACT per row-group,
  cols−1 hits — the theoretical ceiling for this geometry, so ≥0.95 is the
  right bar for 1 KB rows / 32 B bursts).

## 2026-08-28 — early observations (pre-kill-test, from gate runs)

- The brief's `host-centric` map (`ro:bg:ba:ch:co`) fails PIM differently
  than expected: a contiguous stream sweeps a whole row of ONE bank before
  switching banks, so rows stay open (M1 high) but **M2 collapses to
  ~2/16 = 0.125** with a 64-burst window. The realistic `host-cacheline`
  map fails the other way (banks interleave fine, rows fragment). Lesson:
  M1 alone can look healthy while effective bandwidth is destroyed — M3
  (= f(M1, M2)) is the honest headline metric, with M1 and M2 reported
  separately. The kill-test go/no-go in the brief is phrased on M1; judge
  it per-map with M2 alongside.
