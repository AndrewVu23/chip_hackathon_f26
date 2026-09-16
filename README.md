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
make test             # 85 tests incl. the 5 validation gates
make kill-test        # Phase 0: baseline + bracketing runs -> results/phase0/
```

Everything above runs on the analytical model alone. The Phase 3
cross-validation additionally needs two external DRAM simulators, which one
script clones at their pinned commits and builds (no sudo, nothing outside
`third_party/`, safe to re-run):

```bash
./scripts/setup_third_party.sh          # venv + Ramulator 2 + AttAcc PIM
./scripts/setup_third_party.sh --check  # report what is present, build nothing
make ramulator && make attacc           # then run the cross-validation
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
workload.py   request streams: steady | longctx | prefix | spec | fixedlen
   |              Poisson arrivals, deterministic per seed
   v
sim.py        continuous-batching decode loop (1 step = 1 engine iteration):
   |          FCFS oracle admission -> +1 token/seq/step -> block growth ->
   |          release on completion; metrics sampled every N steps
   v
allocator.py  paged   = vLLM v0.2.7 BlockAllocator port (LIFO free list;
   |                    cited to file+commit+lines in the docstring)
   |          contiguous = oracle upper bound (whole-span reservation)
   |          random  = gate-2 reference        pim-aware = frame-aligned
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
                sim, run, sweep, plots, ramulator[Phase 3 harness])
tests/          85 tests incl. the validation gates
configs/        sweep configs (headline, heatmap, phaseA-D/K credibility sweeps)
results/        run outputs: <name>.csv + <name>.csv.meta.json (gitignored)
third_party/    vllm_ref (committed, cited) + ramulator2/attacc clones
                (pinned, re-fetch per third_party/README.md)
docs/           AGENT_BRIEF.md (spec), proposal.md, NOTES.md (decision log)
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
- **Oracle admission by default** (final lengths known, full-footprint
  reservation, no preemption). `--admission vllm` ports v0.2.7 watermark
  admission + preempt-by-recompute; swap-to-CPU is not modeled. Under heavy
  preemption the PIM-aware advantage shrinks (see results).
- **Open-row state cold per sampled sequence**; batch interleaving between
  sequences' reads is not modeled (≤1 extra miss/channel/sample).
- **Timing**: M3 assumes a row miss costs a full tRC with no overlap
  (ratio 10.5 from the published nRC/nCCDAB figure). AttAcc's PIM
  controller overlaps activation, giving an effective ratio ~4.4, so M3
  ratios are optimistic by ~1.4× — measured, not estimated (NOTES
  2026-09-11). M1 and M2 are both simulator-confirmed; absolute GB/s
  remain analytical.
- **Channel-level load balance is out of scope**: M1-M3 rate per-command
  quality; they do not measure whether all channels stay busy.

## Headline results (docs/NOTES.md has every table, dated)

All on hbm3-pim with the realistic host-cacheline map; ideal all-bank
bandwidth 3810 GB/s; `make sweep && make reproduce` regenerates everything.

- **The PIM-aware allocator recovers the whole gap at every block size.**
  M1 0.963 (the 1−1/cols geometric ceiling) from 4- to 256-token blocks on
  steady traffic — 2.3× the vLLM-port baseline's effective bandwidth at the
  standard 16-token block (2805 vs 1226 GB/s), 6.5× at 4 tokens. With the
  speculative-decoding fix below it holds the same 0.963 on the `spec`
  workload (2.4× over paged).
- **Two levers, same destination**: at the derived 128-token block size all
  three allocators converge to 0.963. Use 128-token blocks, or keep 16 and
  place alignment-aware.
- **Structural confirmation, no DRAM model needed**: vLLM's LIFO free list
  leaves only 3–7% of a sequence's consecutive blocks physically adjacent
  and smears sequences over 5–16× more alignment frames than they need
  (`placement_diagnostics`).
- **Speculative decoding needs two changes to frame-aligned placement**:
  draft branch tails must come from a segregated scratch domain AND the
  accepted tokens must be copied back into the sequence's own tail (+8.5%
  copy bytes). Either alone fails; together spec is indistinguishable from
  steady. Diagnosed via frame_spread (1.0 → 3.3 → 1.0).
- **The problem grows as models shed KV heads**: at 16-token blocks the
  paged penalty is 1.2× (MHA, 32 KV heads), 2.1× (GQA, 8), 6.0× (MQA, 1) —
  the direction the field is moving.
- **Honest limits found by the credibility sweeps**: (1) the 2.3× assumes a
  64-burst controller reorder window; with zero reordering it is 1.24×
  (direction and shape invariant). (2) Under heavy vLLM-style preemption
  (0.6× pool headroom, ~200 recompute preemptions) the steady advantage
  shrinks to 1.38× — re-admission bursts land in fragmented frames. **Two
  fixes were implemented and both failed**: best-fit frame planning is a
  no-op (the old greedy path already chose the same frames), and
  opportunistic compaction is actively harmful (it scatters a sequence's
  own blocks to free donor frames, costing up to 10k block copies for
  *worse* M1). Reported as negative results; a correct compaction would
  relocate whole sequences, not blocks. (3) The contiguous oracle's 8.9%
  makespan penalty is purchasable: it vanishes at 2× pool headroom. (4)
  Long-context traffic is nearly immune under any allocator — the penalty
  is churn, not length. (5) Batch size at a fixed pool does not change the
  picture (advantage 2.43×→2.14× from batch 16→256).
- **Cross-validated against two DRAM simulators.** Stock Ramulator 2: the
  row-hit fraction under its own FR-FCFS controller matches M1 within 0.003
  on every sample. AttAcc's all-bank PIM extension (`python -m pimkv.attacc`,
  real `PIM_MAC_AB` commands): **M2 confirmed quantitatively** — our model
  predicts 8.0x more all-bank commands when bank parallelism collapses
  (0.988 -> 0.124), AttAcc measures 7.3x more cycles.
- **Known overstatement, measured**: M3's row-miss penalty uses the
  published nRC/nCCDAB ratio (10.5) with no ACT overlap; AttAcc's PIM
  controller overlaps activation with useful work, giving an *effective*
  ratio near 4.4. So M3 **ratios are optimistic by ~1.4x** on the M1-driven
  component (the M2-driven component is accurate): the 2.3x steady headline
  is ~1.6x on AttAcc timing. Direction, ordering, and every block-size /
  address-map conclusion are unaffected.
- Seeds: worst max−min spread of M1 across 24 cells × 3 seeds is 0.0079.

## Phase status

- **Phase 0 (kill test): DONE** — stopped at the brief's gate, owner
  resumed. Baseline M1 was high on the brief's host-centric map (0.97) but
  that map destroys M2 (0.124 → 9.7% of ideal BW) for every allocator; the
  allocator-recoverable gap lives on channel-interleaved maps.
- **Phase 1: DONE** — four workloads (spec = fork/CoW churn per vLLM
  semantics; prefix = pinned shared system prompt), M1–M4.
- **Phase 2: DONE** — PimAware frame-aligned allocator (+ B2 spec fix),
  headline/bandwidth/heatmap figures.
- **Credibility sweeps (A–D, K): DONE** — seeds, reorder window, pool
  headroom, KV-head sharding (proportional: invariant; head count: 6×→1.2×),
  vLLM admission + preempt-by-recompute (`--admission vllm`).
- **Phase 3 (DRAM cross-validation): DONE, both halves** — `python -m
  pimkv.ramulator` (stock Ramulator 2: M1) and `python -m pimkv.attacc`
  (AttAcc all-bank PIM extension: M2). Build recipes in
  third_party/README.md (the 08-28 claim that Ramulator "already built"
  was wrong; corrected in NOTES).
- Phase 4: demo assets — figures/ and the talk deck.
- Tests: `make test` (85 tests incl. the five validation gates).
