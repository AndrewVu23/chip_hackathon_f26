# AGENT BRIEF — PIM-Aware KV-Cache Allocation

You are helping build a two-week hackathon project. Read this whole file before writing any code. Ask me before deviating from the metric definitions or the validation gates — everything else you can decide yourself.

---

## 0. The one-sentence version

Production LLM serving allocates the KV cache in scattered fixed-size pages; near-memory (PIM) attention accelerators require the KV cache to be laid out in aligned DRAM rows across banks. These two things are incompatible, nobody has measured the cost, and a smarter allocator can recover most of it.

## 1. Background you need

**Decode is memory-bound.** Generating one token requires reading the whole KV cache and multiplying it by a single query vector. Low arithmetic intensity, bandwidth-limited.

**How PIM wins.** A PIM-enabled DRAM puts a small MAC unit next to each bank. The speedup mechanism is *all-bank operation*: activate the same row index in every bank of a channel, broadcast one column command, and every bank's MAC unit computes in parallel. With 16 banks per channel this is roughly a 4–8× effective bandwidth gain in practice. The gain is entirely conditional on all banks being able to serve from the *same row index* and on the operand data actually living in those rows.

**The cost when it fails.** If consecutive all-bank commands need different rows, each one pays a full ACT/PRE cycle (nRC). Published measurement: for decode GEMV, nRC is 10–11× larger than nCCDAB, and host-centric address interleaving forces every all-bank MAC into a different row. That was measured under *static* weight placement.

**What paging does.** vLLM's PagedAttention splits each sequence's KV cache into fixed-size blocks (commonly 16 tokens), maps logical positions to physical blocks through a per-sequence block table, allocates on demand as sequences grow, frees blocks when requests finish, and shares blocks across common prefixes via copy-on-write. Blocks land wherever the free list happens to hand them out.

**The hypothesis.** Under a realistic arrival pattern, the physical block placement produced by a paged allocator destroys row alignment, so a PIM attention unit achieves a small fraction of its theoretical bank parallelism. Tree-drafted speculative decoding makes it worse, because it forks and rewinds the cache several times per token instead of appending once every 16.

---

## 2. Goals and non-goals

**Goals**
1. Measure PIM row-hit rate and achieved bank parallelism under realistic paged allocation.
2. Show how it degrades with speculative decoding, prefix sharing, and long context.
3. Build a PIM-aware allocator that recovers the gap.
4. Produce one publication-quality headline figure and a reproducible harness.

**Non-goals — do not build these**
- No GPU code. No CUDA. No NVBit. Nothing that requires an NVIDIA card.
- No actual LLM inference. We do not need model weights or real logits; we need *allocation event streams*.
- No RTL, no synthesis, no EDA flow. Out of scope.
- No attempt to install vLLM as a running server. See §4.
- No new attention kernel.

---

## 3. Environment

- Primary dev machine: **macOS (Apple Silicon MacBook)**. Everything must build and run here.
- Secondary: Purdue ECN Linux box `eceprog4`, **no sudo**. Anything installed there must be user-local (conda/venv, `--prefix=$HOME`).
- Python 3.11+, numpy, pandas, matplotlib, pytest. No heavyweight ML deps.
- Ramulator 2 (C++20, CMake) — builds on macOS with Apple clang. Used in Phase 3 only.
- Everything runs single-threaded on a laptop in minutes. If a run takes more than ~10 minutes, the design is wrong; tell me.

---

## 4. Critical decision: do not run vLLM

Installing and running vLLM on macOS is a time sink and we don't need inference. Instead:

**Port the allocator, not the engine.** Read vLLM's block manager source (`vllm/core/block_manager.py`, `vllm/core/block/*`) and reimplement its *allocation policy* — free-list discipline, block table structure, prefix-cache hashing, copy-on-write on fork, eviction order — in a clean Python module. Cite the exact source file and commit hash you ported from in a docstring.

**Fidelity caveat must be stated in the writeup.** We are modeling vLLM's allocation policy, not running vLLM. If we get access to a Linux box with a GPU later, we validate by dumping real block IDs. Do not overclaim in any generated text.

---

## 5. Architecture

Build five modules. Keep them independent and testable.

```
pimkv/
  workload.py     # request arrival + length generator
  allocator.py    # allocation policies (baseline, contiguous, pim-aware)
  addrmap.py      # block id -> (channel, bank_group, bank, row, col)
  pimmodel.py     # row-hit / bank-parallelism / effective-bandwidth metrics
  ramulator.py    # Phase 3: emit DRAM command trace, run Ramulator 2, parse
  plots.py
tests/
scripts/
results/
```

### 5.1 `workload.py`
Generate request streams. Each request: arrival time, prompt length, output length. Four workload profiles:

- `steady` — Poisson arrivals, ShareGPT-like lognormal prompt/output lengths (prompt median ~200 tok, output median ~200 tok, long tail).
- `longctx` — prompt lengths 8k–32k, short outputs.
- `prefix` — 60–80% of requests share a common system prompt of 500–1500 tokens (exercises copy-on-write).
- `spec` — the adversarial case. Tree speculative decoding: each decode step forks the sequence into a draft tree of width W and depth D (default W=4, D=3), allocates blocks along every branch, then accepts one path and frees the rest. Acceptance length drawn from a decaying distribution (accept token *i* with probability ~0.8^i).

Deterministic: every generator takes a seed. Same seed, same stream, always.

### 5.2 `allocator.py`
A common interface: `alloc(n_blocks) -> [block_ids]`, `free(block_ids)`, `fork(seq) -> seq'`.

Three policies:
1. `PagedFirstFit` — the vLLM-semantics baseline. LIFO free list, first available block, no placement awareness.
2. `Contiguous` — upper-bound reference. Each sequence gets a physically contiguous span. (Will fail under fragmentation; that's fine and worth reporting — it's why paging exists.)
3. `PimAware` — the contribution. Design constraints:
   - Blocks are allocated from **bank-group-aligned buddy pools** so that a block's physical extent covers the same row index across all banks of a group.
   - Block size in tokens is derived, not chosen: `block_tokens = f(row_bytes, banks_per_pim_unit, head_dim, dtype_bytes, kv_heads)`. Compute this explicitly in code with the formula visible; do not hardcode 16.
   - On free, blocks return to their originating aligned pool so alignment survives reuse.
   - Optional: opportunistic compaction when a pool falls below an occupancy threshold. Implement behind a flag; measure its cost in block-copy bytes.

### 5.3 `addrmap.py`
Configurable mapping from a linear physical KV address to `(channel, bank_group, bank, row, column)`.

- Support at least two interleaving schemes: a conventional host-centric one (`ro:bg:ba:ch:co`, column-interleaved low bits) and a PIM-friendly one.
- Parameterize DRAM geometry: `row_bytes`, `banks_per_channel`, `bank_groups`, `channels`, `burst_bytes`.
- Ship two presets: **HBM3-PIM** (16 banks/channel, 1KB rows) and **GDDR-PIM**.
- This module is where subtle bugs hide. Write its tests first.

### 5.4 `pimmodel.py` — metric definitions (do not change these without asking me)

Given a decode step for one sequence, the PIM unit issues a sequence of all-bank MAC commands, each covering `banks_per_channel × burst_bytes` of KV data.

- **M1 — Row-hit rate.** Fraction of all-bank MAC commands whose target row index is already open in every participating bank. A command counts as a hit only if *all* banks hit; partial hits count as misses (that's what all-bank operation means).
- **M2 — Bank-parallelism utilization.** For each all-bank command, the number of banks that can participate (i.e. hold operand data at the same row index) divided by `banks_per_channel`.
- **M3 — Effective PIM bandwidth.** `ideal_allbank_bandwidth × M2 × (row_hit_weighted_timing_factor)`, where the timing factor amortizes nRC over hits and pays it in full on misses. Show the formula in the docstring.
- **M4 — Estimated decode-step latency** for the attention portion, in ns, from M3 and the KV bytes touched.

Report mean and p95 across decode steps, not just mean.

### 5.5 `ramulator.py` (Phase 3, only after Phase 2 is green)
Emit the DRAM command stream to a Ramulator 2 trace, run it, parse cycle counts. Purpose is **cross-validation of direction and rough magnitude**, not precision. If Ramulator's cycle deltas disagree in sign with M3, stop and tell me — the analytical model is wrong.

---

## 6. Validation gates

Do not report any result until these pass. Write them as pytest tests.

1. **Perfect-case anchor.** `Contiguous` allocator + block size exactly `row_bytes × banks_per_channel` + PIM-friendly address map ⇒ M1 ≥ 0.95. If not, `addrmap.py` is wrong. Fix before proceeding.
2. **Worst-case anchor.** Uniformly random block placement ⇒ M1 ≈ `1/rows_per_bank` within a factor of 2.
3. **Monotonicity.** M1 must be non-decreasing in block size for the contiguous allocator. A non-monotonic curve means an aliasing bug in the address map.
4. **Determinism.** Same seed ⇒ byte-identical results CSV. Run this in CI.
5. **Conservation.** Blocks allocated minus freed equals blocks live, at every step, for every policy.

---

## 7. Timeline

**Phase 0 — the kill test (do this first, before anything else, target: 2 days).**
Minimum viable path: `workload.steady` + `allocator.PagedFirstFit` + `addrmap` HBM3 preset + M1 only. Produce a single number: baseline PIM row-hit rate under realistic paged allocation.

- If M1 is low (say under 0.3), the project premise holds. Proceed, and this number is the motivating figure.
- If M1 is already high (over ~0.7), **stop and tell me immediately.** The premise is dead and we pivot. Do not rationalize the number or go looking for a workload that makes it look bad.

**Phase 1 (days 1–4).** All four workloads, M1–M4, baseline vs contiguous. Validation gates green.

**Phase 2 (days 5–10).** `PimAware` allocator. Sweep block size, batch size, context length, address-map scheme. Produce the headline figure.

**Phase 3 (days 11–13).** Ramulator 2 cross-validation. Sensitivity analysis. If Phase 2 ran long, cut this — it is the first thing to drop.

**Phase 4 (days 12–13, parallel).** Figures, README, demo assets.

---

## 8. How to run

Build the CLI so every result in the video is reproducible with one visible command.

```bash
python -m pimkv.run \
  --workload steady \
  --allocator paged \
  --dram hbm3-pim \
  --addrmap host-centric \
  --block-tokens 16 \
  --requests 2000 \
  --seed 0 \
  --out results/steady_paged.csv

# sweep everything for the headline figure
python -m pimkv.sweep --config configs/headline.yaml --out results/headline/

python -m pimkv.plots headline --in results/headline/ --out figures/
```

Also provide `make reproduce` that regenerates every figure from scratch.

---

## 9. Output artifacts

- `figures/headline.png` — row-hit rate vs block size, three allocator policies, with the `spec` workload curve overlaid. This is the money shot.
- `figures/bandwidth.png` — effective PIM bandwidth, baseline vs PIM-aware vs ideal, per workload.
- `figures/sweep_heatmap.png` — M1 over (block size × context length).
- `results/*.csv` — raw, one row per decode step.
- `README.md` — problem, method, how to reproduce, honest limitations section.

---

## 10. Rules for you

- **Never fabricate a number.** If a run hasn't been executed, say so. No placeholder results that look real, no "example output" in the README that isn't generated.
- **Report bad news immediately.** If the fix doesn't help, that's a finding, not a failure. Do not tune the workload until the result looks good.
- **Write the test before the module** for `addrmap.py` and `pimmodel.py`.
- **Commit after each working increment** with a message describing what was measured, not just what was coded.
- **Keep a `NOTES.md`** logging every design decision and every surprising number, dated. I need this for the writeup and for explaining the work to a professor.
- **Flag fidelity gaps out loud.** Anywhere the model diverges from real hardware or real vLLM, add a line to a `LIMITATIONS` section rather than quietly approximating.
- Ask before adding any dependency beyond numpy/pandas/matplotlib/pytest/pyyaml.

---

## 11. Demo video shot list (build toward this)

Final deliverable is a single recorded video, no repo or report required. Structure it as:

1. The problem in one slide: paged KV layout vs all-bank PIM requirement, side by side.
2. Live terminal: run the baseline command, show M1 print out low.
3. The headline figure appearing, with the speculative-decoding curve highlighted.
4. Live terminal: same command with `--allocator pim-aware`, show M1 recover.
5. The design recommendation: the block size a PIM-backed serving system should use, and why 16 is wrong.
6. Limitations, stated plainly, for 10 seconds.

Keep figure fonts large enough to read on a laptop screen in a compressed video.
