# Paged for Capacity, Punished by Rows

**PIM-aware KV-cache allocation for LLM decode.** Production LLM serving
allocates the KV cache in scattered fixed-size pages (vLLM PagedAttention);
near-memory (PIM) attention accelerators need the KV cache laid out in
aligned DRAM rows across banks (*all-bank operation*: activate the same row
index in every bank of a channel, broadcast one column command, get
~bank-count × effective bandwidth). This project **measures the cost of that
mismatch** under realistic serving workloads and **recovers it in the
allocator** — no GPU, no inference, no RTL: a CPU-side simulation harness
that replays allocation event streams through a DRAM address map and an
analytical all-bank PIM command model.

See `docs/proposal.md` for motivation and `docs/AGENT_BRIEF.md` for the full
spec this repo implements. `docs/NOTES.md` is the dated decision log.

## Quickstart (macOS / Linux, no sudo)

```bash
make install          # uv venv (python3.13) + editable install
make test             # 49 tests incl. the 5 validation gates
make kill-test        # Phase 0: baseline + bracketing runs -> results/phase0/
```

One run, one visible command (all results in the repo are reproducible this
way; `--seed` makes output byte-identical):

```bash
.venv/bin/python -m pimkv.run \
  --workload steady --allocator paged --dram hbm3-pim \
  --addrmap host-centric --block-tokens 16 --requests 2000 --seed 0 \
  --out results/steady_paged.csv
```

Toolchain notes for THIS machine (see docs/NOTES.md 2026-08-28): Homebrew
python3.14 is broken (pyexpat/libexpat mismatch) — use python3.13 via `uv`;
C++ builds need `env -u CXXFLAGS -u CFLAGS -u LDFLAGS` because the shell
profile self-references `CXXFLAGS`. On Purdue `eceprog4` everything is
user-local (venv + `third_party/` clones); nothing needs sudo.

## What gets simulated (and what deliberately not)

```
workload.py   request streams: steady | longctx | prefix* | spec*   (*Phase 1)
   |              Poisson arrivals, deterministic per seed
   v
sim.py        continuous-batching decode loop (1 step = 1 engine iteration):
   |          FCFS oracle admission -> +1 token/seq/step -> block growth ->
   |          release on completion; metrics sampled every N steps
   v
allocator.py  paged   = vLLM v0.2.7 BlockAllocator port (LIFO free list;
   |                    cited to file+commit+lines in the docstring)
   |          contiguous = oracle upper bound (whole-span reservation)
   |          random  = gate-2 reference        pim-aware = Phase 2
   v
addrmap.py    linear KV address -> (channel, bank-group, bank, row, col)
   |          host-centric (ro:bg:ba:ch:co) | host-cacheline | pim-friendly
   v
pimmodel.py   all-bank command coalescing + M1..M4
```

**Deliberate non-goals** (AGENT_BRIEF §2): no GPU/CUDA, no real inference,
no RTL/EDA, no running vLLM server (we port its *allocation policy* — the
engine is irrelevant to placement), no new attention kernel.

## Metrics (definitions frozen — AGENT_BRIEF §5.4)

| | name | definition |
|---|---|---|
| M1 | row-hit rate | fraction of all-bank commands whose row is already open in **every** participating bank (partial hits are misses) |
| M2 | bank-parallelism utilization | participating banks per command / banks_per_channel |
| M3 | effective PIM bandwidth | `BW_ideal × M2 × tCCD_ab / (M1·tCCD_ab + (1−M1)·tRC)` |
| M4 | decode-step attention latency | KV bytes touched / M3 |

Mean **and** p95 are reported (CSVs carry one row per sampled decode step
per sequence). Lesson already learned (NOTES 2026-08-28): M1 alone can look
healthy while M2 is destroyed — the brief's host-centric map keeps rows open
but serializes banks. Judge configurations on (M1, M2, M3) together.

The PIM execution model: per channel, bursts coalesce into all-bank commands
(same row, distinct banks; column-relaxed — conservative toward the paged
baseline) within a controller reorder window (`--window`, default 64;
`--coalesce inorder` = pessimistic no-reordering sensitivity case).

## Validation gates (pytest, all green before any result is reported)

1. **Perfect-case anchor** — contiguous + pim-friendly map ⇒ M1 ≥ 0.95
   (measures 0.969 = 1 − 1/cols_per_row, the geometric ceiling).
2. **Worst-case anchor** — random placement ⇒ M1 ≈ 1/rows within 2×
   (run on a single-bank stream because a coalescing controller *merges*
   the coincidences the heuristic counts as hits; see test docstring).
3. **Monotonicity** — M1 non-decreasing in block size, contiguous allocator.
4. **Determinism** — same seed ⇒ byte-identical CSV.
5. **Conservation** — allocated − freed == live, every step, every policy.

## Repo layout

```
pimkv/          the package (config, addrmap, workload, allocator, pimmodel,
                sim, run, plots, ramulator[Phase 3 stub])
tests/          49 tests incl. the validation gates
configs/        sweep configs (headline.yaml = Phase 2 placeholder)
results/        run outputs: <name>.csv + <name>.csv.meta.json (gitignored)
third_party/    vllm_ref (committed, cited) + ramulator2/attacc clones
                (pinned, re-fetch per third_party/README.md)
docs/           AGENT_BRIEF.md (spec), proposal.md, NOTES.md (decision log)
blog/           write-ups (primer, kill-test post)
```

## Key derived quantity

`config.derive_block_tokens()` computes the PIM-natural block size from
geometry, in code, instead of hardcoding 16: one row-group per channel
(hbm3-pim × llama-gqa-8kv: 1024 B row × 16 banks × 32 channels / 4096 B
per token = **128 tokens**). A 16-token block is a 2 KB per-channel slice =
1/8 row-group — the geometric root of the misalignment this project
measures.

## Limitations (fidelity gaps, stated rather than approximated silently)

- **Modeling vLLM's allocation policy, not running vLLM** (port of v0.2.7
  `BlockAllocator`, cited). Free order is block-table order, not vLLM's
  unordered `set()` iteration (slightly flatters the baseline). Validation
  against real dumped block IDs needs a GPU box; planned if one appears.
- **One representative layer**; block tables are shared across layers so
  placement patterns are identical per layer.
- **Within-block layout assumed PIM-optimal**; the allocator owns only
  inter-block placement (the correct boundary for an allocator study).
- **Oracle admission** (final lengths known, full-footprint reservation,
  FCFS, no preemption/swap). Identical across policies; real preemption
  would churn the free list harder, so the baseline shown is the gentle one.
- **Open-row state cold per sampled sequence**; batch interleaving between
  sequences' reads is not modeled (≤1 extra miss/channel/sample).
- **Timing**: only tRC/tCCD_ab ≈ 10.5 is load-bearing (published
  measurement); absolute ns pending Phase 3 Ramulator 2 cross-validation.
- **Channel-level load balance is out of scope**: M1-M3 rate per-command
  quality; they do not measure whether all channels stay busy.

## Headline results (docs/NOTES.md 2026-09-10 has the full tables)

All on hbm3-pim with the realistic host-cacheline map, 1000 requests,
seed 0; ideal all-bank bandwidth 3810 GB/s; `make sweep && make reproduce`
regenerates everything.

- **The PIM-aware allocator recovers the whole steady-state gap at every
  block size**: M1 0.963 (the 1−1/cols geometric ceiling) from 4-token to
  256-token blocks — 2.3× effective bandwidth over the vLLM-port baseline
  at the standard 16-token block (2805 vs 1226 GB/s), 6.5× at 4 tokens —
  while the contiguous oracle pays 8.9% longer makespan in admission stalls
  and cannot prefix-share.
- **Two levers, same destination**: at the derived 128-token block size all
  three allocators converge to 0.963. Use 128-token blocks, or keep 16 and
  place alignment-aware.
- **Tree speculative decoding is adversarial as hypothesized**: branch
  fork/free churn drags the paged baseline to 0.756 and dents even the
  PIM-aware allocator to 0.879 at bt=16 (seed-stable) — the honest
  limitation of frame-affinity placement, analyzed in NOTES.
- **longctx is nearly immune under any allocator** (paged 0.963): prefill
  allocates hundreds of blocks in one burst that even a LIFO free list
  serves in long runs. The paged penalty is a churn phenomenon, not a
  length phenomenon.

## Phase status

- **Phase 0 (kill test): DONE** — stopped at the brief's gate, owner
  resumed. Baseline M1 was high on the brief's host-centric map (0.97) but
  that map destroys M2 (0.124 → 9.7% of ideal BW) for every allocator; the
  allocator-recoverable gap lives on channel-interleaved maps
  (docs/NOTES.md 2026-08-28).
- **Phase 1: DONE** — all four workloads (spec = fork/CoW churn per vLLM
  semantics; prefix = pinned shared system prompt), M1–M4, 64 tests green.
- **Phase 2: DONE** — PimAware frame-aligned allocator, 48-run headline
  sweep, figures/headline.png + bandwidth.png + sweep_heatmap.png.
- Phase 3: Ramulator 2 cross-validation (binary builds) — the designated
  cut if the deadline arrives first; not started.
- Phase 4: demo assets — blog/ write-ups exist; video per brief §11.
