# KV-PIMple — PIM-aware KV-cache allocation for LLM decode

**Problem.** Processing-in-memory attention is fast because of the *all-bank*
operation: activate the same row index in every bank of a channel, broadcast
one column command, compute in lockstep. That only works if a sequence's KV
sits at matching row indices across banks. Every PIM paper places the KV cache
statically, so their layout is correct by construction. Real serving engines
don't: vLLM's PagedAttention hands out whichever 16-token block was freed last.

**Fix.** KV-PIMple keeps vLLM's 16-token blocks and adds one rule: a sequence
fills its current *region* (the blocks covering one row index across every bank
and channel) before claiming another.

**Result**, steady traffic, 16-token blocks, HBM3-PIM:

| | row-hit (M1) | effective PIM bandwidth (M3) |
|---|---|---|
| paged (vLLM v0.2.7 port) | 0.77 | 1,226 GB/s |
| **KV-PIMple** | **0.96** | **2,805 GB/s (2.3×)** |

2.3× on published DRAM timing, ~1.6× on a cycle-level PIM controller (AttAcc).
No extra memory, no hardware change, no new PIM commands.

---

## Design choices

- **Software only, inside the block manager.** The DRAM row a block lands in is
  `block ID ÷ blocks_per_region`, so choosing a block number *is* choosing a
  row. The allocator never touches an address, and the PIM command set is
  unchanged.
- **The region size is derived, not tuned.** `config.derive_block_tokens()`
  computes it from geometry and model shape: 1 KB row × 16 banks × 32 channels
  ÷ 4 KB per token = 128 tokens = 8 blocks. Change the model or the DRAM and it
  recomputes (512 B/token → 64 blocks per region, 16 KB/token → 2).
- **Keep the 16-token block.** Growing blocks to a full row also fixes
  alignment, and is what prior PIM work does, but it gives up the fine-grained
  packing paging exists for. We get alignment *and* keep the packing.
- **Degrade, don't fail.** With no empty region available, allocation falls back
  to the emptiest partial region, which is plain paging behaviour.
- **Analytical model for breadth, cycle-level simulators for truth.** Sweeping
  hundreds of configurations through AttAcc would take days, so the model does
  the sweeps and both external simulators check it on sampled sequences. The
  model's optimism is measured (~1.4×) and reported, not hidden.
- **The baseline is a cited port**, not a strawman: vLLM v0.2.7
  `BlockAllocator` (LIFO free list), reproduced to file, commit and line.

## Quickstart

Python 3.11+ (numpy, pandas, matplotlib, pyyaml).

```bash
make install        # uv venv + editable install   (no uv? use: make install-pip)
make test           # 85 tests incl. 5 validation gates, ~3-5 min
```

Same workload, same pool, same block size — only the allocator differs (~45 s
each):

```bash
.venv/bin/python -m pimkv.run --workload steady --allocator paged \
  --addrmap host-cacheline --block-tokens 16 --requests 1000 --seed 0

.venv/bin/python -m pimkv.run --workload steady --allocator pim-aware \
  --addrmap host-cacheline --block-tokens 16 --requests 1000 --seed 0
```

Expect `0.7695 / 1226.3 GB/s` then `0.9633 / 2805.4 GB/s`, plus `frame_spread`
(regions touched ÷ regions needed) of 6.10 vs 1.00. Same seed ⇒ byte-identical
output; `--out run.csv` also writes a `.meta.json` with the full config.

```bash
# any figure, rebuilt from the shipped summary CSV in ~1 s
.venv/bin/python -m pimkv.plots headline results/headline/summary.csv figures/headline.png
# allocator CPU cost, ~1 s
.venv/bin/python scripts/bench_alloc.py --requests 200 --out /tmp/bench/
# cross-validation (needs the external simulators, see below)
./scripts/setup_third_party.sh && make ramulator && make attacc
```

`setup_third_party.sh` clones Ramulator 2 and AttAcc at pinned commits and
builds them locally: no sudo, nothing outside `third_party/`, safe to re-run,
`--check` reports status without building.

## How it works

```
workload.py   steady | longctx | prefix | spec — Poisson arrivals, seeded
   v
sim.py        continuous-batching decode loop; admission, block growth,
   |          release, optional vLLM preempt-by-recompute
   v
allocator.py  paged (vLLM port) | pim-aware | contiguous | random
   v
addrmap.py    linear KV address -> (channel, bank group, bank, row, column)
   v
pimmodel.py   all-bank command coalescing -> M1..M4
```

| | metric | definition |
|---|---|---|
| M1 | row-hit rate | commands whose row is already open in **every** participating bank |
| M2 | bank parallelism | participating banks per command ÷ banks per channel |
| M3 | effective bandwidth | `BW_ideal × M2 × tCCD_ab / (M1·tCCD_ab + (1−M1)·tRC)` |
| M4 | attention latency | KV bytes touched ÷ M3 |

M1 alone can look healthy while M2 is destroyed, so configurations are judged
on all three.

## Results

- **Every block size.** M1 0.963 (the 1 − 1/32 geometric ceiling) from 4 to 256
  tokens; 6.5× bandwidth gain at 4-token blocks, 2.4× under speculative
  decoding. All three allocators converge at 128 tokens, where a block *is* a
  region.
- **Worse as models shed KV heads**: 1.2× (MHA-32), 2.1× (GQA-8), 6.0× (MQA-1).
- **Cost of the fix, measured**: +8 µs CPU per decode step against 1.72 ms of
  attention time saved; bookkeeping 70 KiB vs the paged free list's 113 KiB.
  Benefit-to-cost 226× (steady) down to 5× (long context).
- **Cross-validation**: Ramulator 2's own row-hit fraction matches M1 within
  0.0023 mean (0.0065 worst) over 9 samples; AttAcc confirms M2 (model predicts
  8.0× more commands when bank parallelism collapses, AttAcc measures 7.3×
  more cycles).
- **Structural, no DRAM model needed**: vLLM's free list leaves 3–7% of a
  sequence's blocks physically adjacent and smears it over 5–16× more regions
  than it needs.

## Where it weakens (measured, not hand-waved)

- Under vLLM preemption at 0.6× pool headroom the gain drops 2.13× → 1.38×.
- Long-context traffic is nearly immune under any allocator (1.04×).
- With no controller reordering the gain is 1.24× instead of 2.3×.
- M3 charges a full row cycle per miss; AttAcc overlaps activation, so ratios
  are ~1.4× optimistic. Both numbers are reported everywhere.
- Two fixes were tried and failed, reported as negative results: best-fit
  region planning is a no-op, and opportunistic compaction is harmful.

**Not modelled**: running vLLM itself (its allocation policy is ported; the
engine doesn't affect placement), multiple layers (block tables are shared, so
placement repeats), the KV write path and K/V asymmetry, within-block layout,
swap-to-CPU.

## Layout

```
pimkv/      config, addrmap, workload, allocator, pimmodel, sim, run, sweep,
            plots, ramulator + attacc (cross-validation harnesses)
tests/      85 tests including the validation gates
scripts/    third-party setup, allocator cost benchmark, extra plots
configs/    sweep definitions consumed by pimkv.sweep
results/    summary.csv per sweep + per-run .meta.json configs
figures/    every figure, regenerable from the summaries
third_party/  vllm_ref (the cited allocator source) + a Ramulator build patch
```

`make sweep` regenerates the headline sweeps (~15 min, 4 workers),
`make credibility` the full sensitivity set (~2 h, 8 workers),
`make reproduce` the main figures.
