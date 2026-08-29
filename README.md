# pimkv — Paged for Capacity, Punished by Rows

**PIM-aware KV-cache allocation for LLM decode: measuring (and recovering)
the DRAM row alignment that paged serving allocators destroy.**

## The problem in three sentences

Near-memory (PIM) attention accelerators get their speedup from *all-bank
operation*: activate the same row index in every bank of a DRAM channel and
broadcast one column command, multiplying effective bandwidth by roughly the
bank count. That only works if the KV cache physically sits at aligned row
indices across banks. Production serving (vLLM's PagedAttention) instead
scatters the KV cache into fixed-size blocks handed out by a runtime free
list — and nobody has measured what that costs a PIM attention unit. This
repo measures it and builds an allocator that recovers it.

We simulate the *allocation event stream*, not the LLM: no GPU, no model
weights, no vLLM server. Everything runs on a laptop in minutes and every
number is reproducible from one command.

## Quickstart

```bash
make install          # uv venv (python3.13) + numpy/pandas/matplotlib/pytest/pyyaml
make test             # 49 tests incl. the 5 validation gates
make kill-test        # Phase 0: baseline M1 under paged allocation
make reproduce        # all Phase 0 runs (results/phase0/*.csv + .meta.json)
```

Single runs (this exact command produced the baseline number):

```bash
.venv/bin/python -m pimkv.run \
  --workload steady --allocator paged --dram hbm3-pim \
  --addrmap host-centric --block-tokens 16 --requests 2000 --seed 0 \
  --out results/phase0/steady_paged_hostcentric.csv
```

`--allocator {paged,contiguous,random,pim-aware}`,
`--addrmap {host-centric,host-cacheline,pim-friendly}`,
`--dram {hbm3-pim,gddr6-pim}`, `--coalesce {window,inorder}`,
`--block-tokens 0` selects the PIM-derived block size. See
`python -m pimkv.run --help` for the rest.

## How the pipeline works

```
workload.py      allocator.py        addrmap.py         pimmodel.py
requests   --->  block tables  --->  (ch,bg,ba,ro,co) ---> M1..M4 metrics
(steady,         (paged = vLLM       per KV burst,        per decode step
 longctx,         v0.2.7 port,       3 interleaving       (row-hit rate,
 prefix*,         contiguous         schemes, 2 DRAM      bank parallelism,
 spec*)           oracle, random,    presets              eff. bandwidth,
                  pim-aware*)                             latency)
                        ^  driven per decode step by sim.py (continuous
                           batching: admit -> decode/grow -> sample -> free)
                                                          (* = Phase 1/2)
```

1. **workload.py** generates deterministic request streams (arrival step,
   prompt length, output length). `steady` = Poisson arrivals, lognormal
   ShareGPT-like lengths (median ~200/~200).
2. **sim.py** replays them through a continuous-batching loop: FCFS
   admission, one token per sequence per step, a new KV block whenever a
   sequence crosses a block boundary, blocks freed at completion.
3. **allocator.py** decides *which physical block ids* each sequence gets.
   `paged` is a faithful port of vLLM v0.2.7's `BlockAllocator`
   (commit `2e0b6e7`, LIFO free list — see the class docstring for
   line-level citations).
4. **addrmap.py** maps each block's bytes to DRAM coordinates under a
   configurable interleaving scheme (bit-sliced, exactly invertible).
5. **pimmodel.py** replays each sampled sequence's KV sweep as *all-bank PIM
   commands* (same row across distinct banks of a channel; controller
   reordering limited to a configurable window) and scores:

   | Metric | Meaning |
   |---|---|
   | **M1** | all-bank row-hit rate (a partial hit is a miss) |
   | **M2** | participating banks per command / banks per channel |
   | **M3** | effective PIM bandwidth = ideal × M2 × tCCD/(M1·tCCD+(1−M1)·tRC) |
   | **M4** | attention decode-step latency = KV bytes / M3 |

   Definitions are fixed by AGENT_BRIEF §5.4; formulas live in the
   `pimkv/pimmodel.py` module docstring.

Outputs: one CSV row per sampled (decode step, sequence) plus a
`.meta.json` sidecar carrying the full configuration and summary, so every
figure is traceable to the exact command that made it.

## Validation gates (all enforced in `tests/`)

1. **Perfect-case anchor** — contiguous placement + PIM-friendly map ⇒
   M1 ≥ 0.95 (measures 0.969 = 1 − 1/cols_per_row). `test_gate1_*`
2. **Worst-case anchor** — uniformly random placement ⇒ M1 ≈ 1/rows within
   2× (see NOTES.md 2026-08-28 for the coalescing subtlety). `test_gate2_*`
3. **Monotonicity** — M1 non-decreasing in block size for the contiguous
   allocator. `test_gate3_*`
4. **Determinism** — same seed ⇒ byte-identical CSV. `test_gate4_*`
5. **Conservation** — allocated − freed == live at every step. `test_gate5_*`

## Repo layout

```
pimkv/            the package (config, addrmap, workload, allocator,
                  pimmodel, sim, run, plots, ramulator[Phase 3 stub])
tests/            pytest suite; validation gates are named test_gateN_*
results/          run outputs (gitignored; regenerate with make reproduce)
figures/          generated figures (gitignored)
third_party/      pinned external sims + vendored vLLM reference source
                  (see third_party/README.md for pins and build notes)
AGENT_BRIEF.md    the project brief (metric definitions live here)
proposal.md       hackathon proposal
NOTES.md          dated log of every design decision and surprising number
```

## Environment notes

- Primary: macOS / Apple Silicon. **Known machine quirks** (both handled by
  the Makefile, documented in NOTES.md): Homebrew python@3.14's pyexpat is
  broken against the system libexpat → we pin python3.13 via `uv`; a
  self-referential `CXXFLAGS` in the shell profile breaks CMake → the
  `ramulator` target builds with `env -u CXXFLAGS -u CFLAGS -u LDFLAGS`.
- Secondary (eceprog4, no sudo): `python3 -m venv .venv && .venv/bin/pip
  install -e '.[dev]'` — everything is user-local.
- Ramulator 2 is a Phase 3 dependency only; the analytical model has no
  C++ dependency.

## Honest limitations (kept current — additions go here, not under the rug)

- **We model vLLM's allocation policy, not vLLM.** Ported from
  `vllm/core/block_manager.py` @ v0.2.7 (`2e0b6e7`). If a GPU box becomes
  available, validation = dump real block IDs and replay them. One known
  divergence: vLLM frees finished sequences in `set()` order; we free in
  block-table order for determinism.
- **One representative transformer layer.** Layers share the block table in
  vLLM, so placement statistics are layer-invariant; absolute bandwidth/
  latency numbers are per-layer quantities.
- **Within-block layout is assumed PIM-optimal** — the allocator study is
  about inter-block placement. Per-head striding inside a block is a static
  layout problem PIM designs already solve.
- **Column-relaxed all-bank commands** (same row, distinct banks): optimistic
  for the paged baseline, so the reported gap is a lower bound in that
  respect.
- **Oracle admission control** (final lengths known): removes preemption/
  swap from scope; identical for all policies.
- **No batch-interleaving of open-row state** between sequences within a
  step; sampled sequences start cold (≤1 extra miss per channel; also
  conservative — real interleaving hurts scattered layouts more).
- **Timing**: only the tRC/tCCD_ab ≈ 10.5 ratio is evidence-backed; absolute
  ns pending Phase 3 Ramulator 2 cross-validation.
- **Channel-level load balance is out of scope**: M2 scores bank
  parallelism within a channel; a PIM stack's cross-channel scheduling of
  heads/sequences is a different (orthogonal) problem.

## Project status

- [x] Phase 0 — toolchain, skeleton, validation gates, kill test
- [ ] Phase 1 — all four workloads (prefix/spec need fork/CoW), M1–M4,
      baseline vs contiguous
- [ ] Phase 2 — `PimAware` allocator + sweeps + headline figure
- [ ] Phase 3 — Ramulator 2 cross-validation (first thing to cut)
- [ ] Phase 4 — figures, demo video
