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

## 2026-08-28 — PHASE 0 KILL-TEST RESULT (decision point — owner input needed)

All numbers: steady workload, 2000 requests, seed 0, hbm3-pim, block_tokens
16, max_batch 64, window-64 coalescing. Ideal all-bank BW = 3810 GB/s.
Commands in `Makefile:kill-test`; raw CSVs in `results/phase0/`.

| allocator / addrmap            | M1 mean | M2 mean | M3 GB/s | % ideal |
|--------------------------------|--------:|--------:|--------:|--------:|
| paged / host-centric (brief's baseline) | 0.970 | 0.124 | 368 | 9.7% |
| paged / host-cacheline (realistic host) | 0.759 | 0.990 | 1173 | 31% |
| contig / host-cacheline                 | 0.958 | 0.990 | 2719 | 71% |
| paged / pim-friendly                    | 0.969 | 1.000 | 2938 | 77% |
| contig / pim-friendly (upper anchor)    | 0.969 | 1.000 | 2938 | 77% |

Block-size probes (paged / host-cacheline, 500 req, seed 0; CSVs
`results/phase0/probe_bt{4,128}_paged_hostcacheline.csv`): bt=4 → M1 0.571,
M2 0.566, M3 508 GB/s (13.3%); bt=128 (the derived PIM-natural size) →
M1 0.963, M2 0.989, M3 2795 GB/s (73.4%). (At bt=4 the per-channel slice is
512 B, so even bank spread inside a window degrades — M2 falls too.)

**Verdict per the brief's stop rule (M1 > 0.7 ⇒ stop and tell): STOPPED.**
Baseline M1 is 0.97 on the brief's host-centric map and 0.76 on the
realistic map — neither is "under 0.3". Reported as-is, no rationalizing.
What the runs actually show:

1. The brief's host-centric map loses its 10× not through row misses but
   through **M2 = 0.124** (whole-row-per-bank sweeps serialize banks):
   9.7% of ideal BW with a *healthy* M1. The M1-only kill criterion was
   aimed at the wrong metric for this map.
2. Under the **pim-friendly channel-slab map, paged ≡ contiguous ≡ 77%**:
   a 16-token block there is 4 whole row-groups in one channel, so scatter
   is harmless. No allocator story exists in that regime.
3. The recoverable allocator gap lives exactly where the block's
   **per-channel slice < row-group** (bt=16 ⇒ 2 KB slice vs 16 KB
   row-group under channel-interleaved maps): paged 0.759/1173 GB/s vs
   contiguous 0.958/2719 GB/s — **placement alone is worth 2.3×**, and
   block size 16→128 recovers similarly (2805 GB/s). This is a real,
   measured, recoverable gap — just narrower than the proposal's framing
   (it requires a channel-interleaved map, which is what real hosts and
   sharded PIM layouts use).

Open question for the owner before Phase 1: keep the original framing with
the honest numbers (2.3–2.6× recoverable, spec/prefix workloads may widen
it), or reframe the headline around the block-size × address-map geometry
(the derived-128 recommendation) with the allocator as the second lever.
Note `spec` (tree speculation) is expected to churn placement much harder
than steady — the baseline may yet drop well below 0.76; that is the next
measurement either way.

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

## 2026-09-10 — Phase 1+2 design decisions (owner said proceed)

Owner resumed the project ("proceed with next steps") after the Phase 0 stop;
framing question resolved by measuring both: the headline figure carries M1
AND M3 vs block size (allocator recovery + the derived-128 recommendation in
one plot), spec overlaid.

**Speculation model** (`sim.py::spec_round`, --workload spec): per decode
round the sequence forks W=4 draft branches of depth D=3; each branch owns a
private tail = CoW copy of the partial last block (vLLM append_slot
semantics) + overflow blocks, sized to hold D drafts **plus the bonus token**
(first implementation missed the bonus and under-allocated — caught by the
conservation/enumeration checks, not by eye). Draft i accepted w.p. 0.8^i
sequentially; adopted branch spliced in, all other branches freed. Deviations
logged: W independent depth-D paths rather than a literal tree (mid-density);
drafting-phase KV reads not measured (we score the committed cache each
round); if the pool cannot hold W tails the round degrades to 1-token decode
(spec_stalls — real engines also disable speculation under pressure).
Contiguous runs spec against a per-seq reserved scratch region: zero churn,
copies counted (copied_bytes) — the honest price of contiguity.

**Prefix model**: one system prompt (500-1500 tok, per-seed) shared by ~70%
of requests, shared at FULL-block granularity through a pinned cache
(refcount+1 per member, vLLM fork semantics); the partial boundary block is
private. Contiguous cannot share and allocates private copies — its
peak_live_blocks is strictly higher (tested), reported as the capacity cost
of contiguity rather than hidden.

**PimAware allocator**: pool carved into frames whose linear extent is one
row index across every bank of every channel (rowgroup_bytes x channels =
512 KB on hbm3-pim — derived in run.py::frame_blocks_for, never hardcoded).
Sequences keep frame affinity and fill frames at ascending offsets;
empty-frame-first keeps frames sequence-pure; freed blocks return to their
frame so alignment survives reuse. No compaction (the brief's optional flag
is left unimplemented; copied-bytes accounting exists if we add it).
Longctx arrival rate raised 0.02->0.2 before any measurement so the pool
actually churns (at near-empty batch every allocator looks fresh-pool clean).

Gate tests re-verified after the refactor (64 tests): conservation holds
each step under sharing + spec churn; spec runs byte-deterministic; pim-aware
M1 >= 0.90 vs paged ~0.76 under host-cacheline in the small smoke runs.
Full-size sweep numbers land below when the 48-run sweep finishes.

## 2026-09-10 — PHASE 2 RESULTS (headline sweep, results/headline/)

All: hbm3-pim, host-cacheline map, 1000 requests, seed 0, max_batch 64.
Ideal all-bank BW = 3810 GB/s. `make sweep` reproduces (48 runs, ~25 min).

**M1 / M3 vs block size** (steady | spec):

| bt | paged | pim-aware | contiguous | paged-spec | pim-aware-spec |
|---:|------:|----------:|-----------:|-----------:|---------------:|
| 4   | 0.542 / 432  | 0.963 / 2800 | 0.959 / 2648 | 0.538 / 405  | 0.956 / 2312 |
| 8   | 0.552 / 777  | 0.963 / 2801 | 0.958 / 2710 | 0.533 / 723  | 0.929 / 2303 |
| 16  | 0.769 / 1226 | 0.963 / 2805 | 0.958 / 2712 | 0.756 / 1153 | 0.879 / 1849 |
| 32  | 0.879 / 1777 | 0.963 / 2799 | 0.959 / 2727 | 0.873 / 1715 | 0.912 / 2110 |
| 64  | 0.935 / 2337 | 0.963 / 2799 | 0.960 / 2753 | 0.933 / 2304 | 0.943 / 2471 |
| 128 | 0.963 / 2803 | 0.963 / 2803 | 0.963 / 2805 | 0.963 / 2795 | 0.963 / 2795 |
| 256 | 0.963 / 2802 | 0.963 / 2802 | 0.963 / 2804 | 0.963 / 2795 | 0.963 / 2795 |

**bt=16 across workloads (M3 GB/s)**: steady 1226/2805/2712,
longctx 2823/2938/2936, prefix 2042/2867/2865, spec 1153/1849/2710
(paged/pim-aware/contiguous).

Findings, in order of importance:

1. **The fix works, and makes block size irrelevant on steady traffic.**
   PIM-aware hits 0.963 M1 (the 1−1/cols geometric ceiling) at EVERY block
   size, including 4 — 2.3× bandwidth over paged at bt=16, 6.5× at bt=4. It
   even edges the contiguous oracle (ascending aligned frames vs vLLM's
   descending spans), with none of contiguous's costs: contiguous suffered
   3343 admission stalls on steady = 8.9% longer makespan (6485 vs 5955
   steps) and cannot prefix-share (higher peak footprint, tested).
2. **Two levers, same destination**: at bt=128 (the derived PIM-natural
   size) every allocator converges to 0.963. So the design recommendation
   has two forms: use 128-token blocks, OR keep 16 and allocate
   alignment-aware. The second preserves paging's granularity benefits.
3. **spec is adversarial exactly as hypothesized — and it dents the fix.**
   Paged drops slightly (0.756), but pim-aware degrades to 0.879 at bt=16
   (seed-stable: 0.877 at seed 1): branch-tail churn punches holes in
   frames, adopted CoW copies land at hole offsets in *other* frames, so
   consecutive logical blocks straddle frames more often. The dip is
   centered at bt=16 (0.956 at bt=4, 0.929 at bt=8, 0.943 at bt=64) —
   at small bt the tail region is a tiny fraction of the sequence; at large
   bt rounds rarely cross block boundaries; bt=16 maximizes
   churn-per-KV-byte with W=4/D=3. Hypothesis logged, not yet
   root-cause-verified block-by-block. Spec CoW traffic itself: 12.6 GB
   copied over the run (both paged and pim-aware); contiguous instead
   copies accepted tokens (1.08 GB) — cheaper in bytes but needs the
   8.9%-makespan-class reservation regime.
4. **longctx barely suffers under ANY allocator** (paged 0.963/2823):
   a 20k-token prompt allocates hundreds of blocks in one prefill burst,
   which even a LIFO free list serves in long runs; decode growth is a
   rounding error of the context. The paged penalty is a *churn* phenomenon
   (steady/spec), not a length phenomenon.
5. Prefix sharing helps the paged baseline (0.893 vs 0.769 steady): the
   shared prompt is allocated once, early, compactly, and reused by 70% of
   requests.

Remaining gap to ideal (2805 vs 3810 GB/s = 74%): the 1−1/cols_per_row
row-ACT floor (0.969 ceiling → timing factor ~0.77 at nRC/nCCD=10.5), not
placement. A PIM controller with cross-row-group pipelining would close it;
out of scope, noted for the writeup.

**Heatmap (results/heatmap/, paged, fixedlen prompts, 128-token outputs):**
M1 rises monotonically with both block size and context length — 0.74 at
(bt=4, 512 tok) up to 0.97 at (any bt, 16k tok). Confirms finding 4: the
paged penalty is concentrated where decode-time growth is a large fraction
of the cache (short contexts, small blocks); long prompts are laid down in
one prefill burst and stay compact. Together with the headline figure this
is the design envelope: PIM-aware allocation matters most for
short/medium-context, high-churn serving — which is the common case.

## 2026-09-11 — PHASE B: spec dip diagnosed (design omission, not artifact)

Added `sim.placement_diagnostics`: blk_adj (consecutive logical blocks that
are also physically adjacent), same_frame (consecutive pairs inside one
alignment frame), frame_spread (frames touched / frames needed; 1.0 =
perfect packing). These are STRUCTURAL — no DRAM model involved — and every
allocator is diagnosed against the same geometric frame, so paged and
pim-aware are on one yardstick. Config: configs/phaseB.yaml, 400 requests,
host-cacheline, results/phaseB/.

| wl | alloc | bt | M1 | blk_adj | same_frame | frame_spread |
|---|---|--:|--:|--:|--:|--:|
| steady | pim-aware | 4  | 0.963 | 0.980 | 0.975 | 1.000 |
| steady | pim-aware | 16 | 0.963 | 0.921 | 0.898 | 1.000 |
| steady | pim-aware | 64 | 0.963 | 0.648 | 0.561 | 1.000 |
| spec   | pim-aware | 4  | 0.956 | 0.572 | 0.960 | 1.410 |
| spec   | pim-aware | 16 | 0.881 | 0.487 | 0.556 | 3.269 |
| spec   | pim-aware | 64 | 0.943 | 0.235 | 0.200 | 1.540 |
| steady | paged     | 16 | 0.794 | 0.069 | 0.194 | 5.413 |
| spec   | paged     | 16 | 0.765 | 0.026 | 0.075 | 5.752 |
| steady | paged     | 4  | 0.585 | 0.072 | 0.229 | 16.344 |

Read same_frame relative to the frame size G = 512 KB / (bt x 4096 B):
G = 32/8/2 blocks at bt = 4/16/64, so the structural ceiling on same_frame
is (G-1)/G = 0.969/0.875/0.500. **pim-aware on steady sits at that ceiling
for every block size, with frame_spread exactly 1.000** — the allocator is
doing precisely what it was designed to do.

**Diagnosis of the spec dip.** `PimAware.alloc_scratch` inherits the base
implementation, which allocates from the SEQUENCE'S OWN frame via
`_pick_frame(seq_id)`. There is no notion that speculative scratch is
transient. Per round at bt=16: 4 branch tails x 1 block = 4 blocks claimed
from an 8-block frame, 3 freed on adoption. Frames therefore never present
as empty, `_pick_frame` falls through to the global "partial frame with the
most free slots" path, sequences get smeared AND intermixed, and
frame_spread goes 1.000 -> 3.269.

Why bt=16 is the worst point (the dip is centered, not monotonic): at bt=4,
G=32 so 4 blocks of churn is a small fraction of a frame (spread 1.41); at
bt=64 a block is 256 KB = half a row-group and self-aligns regardless
(spread 1.54, M1 0.943 despite same_frame 0.200); bt=16 is the crossover
where per-round churn is half the frame capacity while blocks are still too
small to align on their own.

So: **a design omission with a known fix**, not a measurement artifact and
not a bug in the fallback path (the fallback is the symptom; steady never
reaches it, spread 1.000). Proposed Phase B2 (~40 lines, NOT implemented —
owner decision): give speculative scratch its own frame pool keyed
separately from the sequence, so drafts never fragment committed frames.
Prediction: spec/bt16 M1 recovers from 0.881 toward ~0.96.

**Independent bonus finding.** The paged baseline's blk_adj is 0.03-0.07:
vLLM's LIFO free list yields almost NO physically adjacent consecutive
blocks under churn, at frame_spread 5.4-16.3. This confirms the project's
core thesis structurally, without relying on the analytical DRAM model at
all — a stronger form of the argument than M1 alone.

## 2026-09-11 — PHASE A: credibility sweeps (114 runs, configs/phaseA.yaml)

### A1 — seed variance: PASSES, decisively
Three seeds over {steady,spec} x {paged,pim-aware,contiguous} x bt{4,16,64,128}.
**Worst max-min spread of M1 across all 24 cells: 0.0079**; typical 0.0003.
Every Phase 2 curve stands as reported. Single-seed results were safe.

### A2 — controller reorder window: THE REAL CAVEAT
Same grid with `--coalesce inorder` (zero reordering, the pessimistic
endpoint) vs the default 64-burst window.

PIM-aware advantage (M3 pim-aware / M3 paged), steady:

| bt | window | inorder |
|---:|---:|---:|
| 4   | 6.48x | 2.01x |
| 16  | 2.29x | 1.24x |
| 64  | 1.20x | 1.04x |
| 128 | 1.00x | 1.00x |

**The sign and the shape are invariant — pim-aware >= paged everywhere,
small blocks always worst, convergence at bt=128 always — but the
MAGNITUDE is assumption-dependent.** The headline "2.3x" holds only with a
reorder window; with none it is 1.24x. Absolute M3 under in-order caps at
~454 GB/s (12% of ideal) for EVERY allocator including contiguous, i.e. in
that regime the controller, not the allocator, is the bottleneck — which is
why zero-reordering is an unrealistically pessimistic straw man for a real
PIM controller with request queues. Correct framing for the writeup: quote
the window result as the headline, quote 1.24x as the floor, and state that
the true controller lies between.

### A3 — contiguous's makespan penalty is a MEMORY penalty (steady, bt16)

| headroom | contig frag_failures | contig makespan | paged/pim-aware makespan |
|---:|---:|---:|---:|
| 1.1 | 5472 | 7406 | 6486 / 6478 |
| 1.3 | 3343 | 6485 | 5959 / 5955 |
| 1.6 |  682 | 5909 | 5836 / 5836 |
| 2.0 |    0 | 5836 | 5836 / 5836 |

The 8.9% penalty reported in Phase 2 is real *at 1.3x headroom* and
**disappears entirely at 2.0x**. So contiguous placement is not impossible,
it is purchasable: it costs ~2x the KV pool. That is precisely the capacity
argument that motivated paging, and it is a cleaner way to state the
tradeoff than "the oracle stalls". paged and pim-aware take ZERO admission
failures at every headroom, and M1 is headroom-invariant for all three.

### A4 — offered load: NULL RESULT, experiment was mis-designed
Arrival rate 0.25 -> 0.5 -> 1.0 moved the pim-aware/paged ratio only
2.29 -> 2.32 -> 2.34, and peak live blocks not at all (~2500). Reason:
`max_batch=64` already saturates at rate 0.25, so raising arrivals just
lengthens the queue without adding memory pressure. The experiment tested
nothing. The memory-pressure evidence actually comes from A3's headroom-1.1
row, where the pool is tight and paged/pim-aware still show zero admission
failures and unchanged M1 — so pool pressure does not alter the allocator
comparison. To test load properly would require raising max_batch, not the
arrival rate; logged for whoever picks this up.

## 2026-09-11 — PHASE C: KV-head sharding (configs/phaseC.yaml, 24 runs)

`config.shard_kv(geom, shape, S)` models S independent all-bank domains,
each owning channels/S channels and kv_heads/S heads — the objection that
Phases 0-2 byte-interleaved the KV cache over all 32 channels whereas real
PIM-attention designs assign heads to channel groups.

Result across S = 1, 2, 4, 8 (steady and spec, all three allocators):
**M1, M2, same_frame and frame_spread are IDENTICAL to three decimals**;
M3 scales exactly 1/S (steady pim-aware 2801 -> 1400 -> 700 -> 350 GB/s)
because a shard is proportionally less hardware. The pim-aware/paged M3
ratio is constant: 2.14x steady, 1.53x spec, at every S.

So the finding is scale-invariant, and it is invariant *for a structural
reason*: proportional sharding divides rowgroup_bytes and
kv_bytes_per_token by the same S, holding fixed the only quantity that
matters — the block's per-channel slice as a fraction of a row-group. The
result is not an artifact of "32 channels".

**Honest limit of this test.** Because proportional sharding holds that
ratio fixed by construction, this experiment can only return invariance;
it answers "is 32 channels special?" (no) but NOT the sharper form of the
objection, which is "what if a design's bytes-per-token-per-channel ratio
differs?" That ratio is
    slice_fraction = bt * kv_bytes_per_token / (channels * rowgroup_bytes)
and on hbm3-pim at bt=16 it evaluates to:

| model shape | KV B/token | slice as fraction of a row-group |
|---|---:|---:|
| MQA, 1 KV head        |    512 | 1/64 (far worse) |
| GQA, 8 KV heads (ours)|   4096 | 1/8              |
| MHA, 32 KV heads      |  16384 | 1/2 (nearly self-aligned) |

i.e. **the problem intensifies as models move toward FEWER KV heads**,
which is exactly the direction GQA/MQA have taken the field — the same
"direction of travel" argument the proposal makes for speculative decoding.
Untested (would need model presets + ~6 runs, ~10 min); this is the
sharpest remaining gap and should be either measured or stated plainly.

## 2026-09-11 — PHASE K: KV-head count (configs/phaseK.yaml, 18 runs, bt16)

The non-proportional axis Phase C could not test. Per-channel block slice
= bt x kv_bytes_per_token / (channels x rowgroup_bytes): 1/64, 1/8, 1/2 of
a row-group for MQA-1KV / GQA-8KV / MHA-32KV.

| model | steady paged M1 | pim-aware | pa/paged M3 | spec paged | pim-aware | pa/paged |
|---|--:|--:|--:|--:|--:|--:|
| mqa-1kv  (512 B/tok)  | 0.561 | 0.918 | **6.04x** | 0.562 | 0.927 | 7.04x |
| gqa-8kv  (4096 B/tok) | 0.785 | 0.963 | 2.14x | 0.760 | 0.873 | 1.53x |
| mha-32kv (16384 B/tok)| 0.941 | 0.967 | 1.18x | 0.938 | 0.951 | 1.09x |

Prediction confirmed: **the paged penalty grows as KV heads shrink** — the
direction GQA/MQA have taken the field. At MHA the block is already half a
row-group and mostly self-aligns; at MQA paged loses 6x. Note the pim-aware
and contiguous ceilings drop to ~0.92/0.89 on MQA: a 400-token MQA
sequence is only 6.4 KB per channel, under one row-group, so the
per-sequence cold-start ACT (modeling decision 5, NOTES 2026-08-28) is
amortized over ~12 commands instead of 32 — affects every allocator
equally and is the conservative direction.

## 2026-09-11 — PHASE B2: the spec fix works (configs/phaseB2.yaml)

2x2 on pim-aware/spec: scratch domain {shared (Phase 2), separate} x
adoption {splice (vLLM pointer swap), copyback (copy accepted tokens into
the sequence's own tail, free all branches)}. 500 requests, host-cacheline.

| bt | shared+splice (Phase 2) | shared+copyback | separate+splice | **separate+copyback** |
|---:|--:|--:|--:|--:|
| 4  | 0.954 / 1.49 | 0.959 / 1.23 | 0.700 / 12.9 | **0.963 / 1.000** |
| 16 | 0.873 / 3.49 | 0.871 / 3.56 | 0.850 / 4.08 | **0.963 / 1.000** |
| 64 | 0.942 / 1.57 | 0.948 / 1.43 | 0.942 / 1.57 | **0.963 / 1.000** |
(M1 / frame_spread)

**Both halves are necessary.** Copyback alone changes nothing (branches
still churn the sequence's frame). Segregation alone is *harmful*: with
splice the adopted branch block lives permanently in a scratch frame, so
committed blocks end up smeared across scratch frames (spread 12.9 at
bt=4). Together they make spec indistinguishable from steady: M1 0.963,
frame_spread exactly 1.000 at every block size; bt16 M3 1796 -> 2795 GB/s.
Cost: copied bytes 6858 -> 7443 MB per run (+8.5%, the accepted tokens),
on top of the CoW copies every spec engine already pays. Copyback on the
paged baseline: 0.760 -> 0.767, i.e. nothing — the fix is specific to
frame-aligned placement. Steady is unchanged (0.963 / 1.000).

Default flipped: pim-aware now uses separate scratch (`--pim-scratch`),
spec adoption stays `splice` by default so the baseline remains
vLLM-faithful; `--spec-adopt copyback` is the recommended pim-aware
configuration and the headline figure should be regenerated with it.
