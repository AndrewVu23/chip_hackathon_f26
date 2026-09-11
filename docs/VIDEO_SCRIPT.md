# Demo video script — "Paged for Capacity, Punished by Rows"

**Target length:** 6:30 (range 5:45–7:00)
**Format:** slides + two live terminal segments
**Figures:** all in `figures/`, already generated. Paths given per slide.

A note before you record: the live demo uses `--requests 300` so each command
finishes in ~15 seconds on camera. The figures in the slides come from
1,000-request runs, where more churn has accumulated, so the baseline there is
a bit worse (0.77 vs 0.81) and the gap a bit wider (2.3× vs 2.0×). If anyone
asks, that difference is exactly the finding: **more turnover, more damage.**

---

## Slide 1 — Title (0:00–0:20)

**Figure:** none. Title card.
Title: *Paged for Capacity, Punished by Rows*
Subtitle: *PIM-aware KV-cache allocation for LLM decode*
Your name · Purdue ECE · Theme 4

> Generating one token from a language model means reading the model's entire
> memory of the conversation and doing almost no math with it. Decoding isn't
> compute-bound. It's bandwidth-bound. So the interesting question is how you
> feed it.

---

## Slide 2 — The two ideas (0:20–1:20)

**Figure:** rebuild the SVG from the report as a slide — the aligned row on top
(orange, all four banks at one row index), the scattered version below (blue,
four different rows). Screenshot it from the published report if that's easier:
it's the boxed diagram in section 02.

> One answer is to stop moving the data at all — put small multiplier units
> inside the DRAM and compute where the data already lives. Several published
> accelerators do this, and they all lean on the same trick: open the *same row
> number* in all sixteen banks at once, broadcast a single command, and every
> bank computes in parallel. Sixteen banks' worth of work, one command.
>
> [point at the top row] That's what it needs. Same row, every bank.
>
> [point at the bottom] And this is what a modern LLM server actually produces.
> vLLM's PagedAttention splits the cache into fixed 16-token blocks and hands
> them out from a free list, wherever one is free. It was designed to stop
> wasting memory, and it works — it's the right design, for capacity. But
> blocks land wherever the list points. After a few minutes of real traffic,
> the physical layout is confetti.
>
> Every PIM paper assumes the layout is fine. Every production server
> guarantees it isn't. Nobody had measured the cost.

---

## Slide 3 — The arithmetic (1:20–1:50)

**Figure:** none — three lines of big text, built one at a time.

```
one row-group  (same row, all 16 banks)   16 KB
one 16-token block, per channel            2 KB
                                        = 1/8
```

> Here's the number that sets up the whole problem. On the HBM3-class geometry
> I model, one row index across all sixteen banks holds 16 kilobytes. A
> 16-token block, spread across the 32 channels, contributes just 2 kilobytes
> to each one. **One-eighth** of what the hardware wants aligned.
>
> So eight consecutive blocks have to land side by side before the accelerator
> gets one clean all-bank operation. Nothing in the allocator arranges that —
> it has no idea DRAM rows exist.

---

## Slide 4 — What I built (1:50–2:25)

**Figure:** simple 4-box pipeline diagram (make in PowerPoint):
`workload → vLLM allocator policy → DRAM address map → all-bank model`
Caption underneath: *no GPU · no model weights · runs on a laptop in minutes*

> I deliberately didn't run a language model. Nothing here depends on what the
> model *says* — only on when blocks get allocated and freed. So I built a
> harness that isolates exactly that.
>
> Realistic request streams go in. They run through a faithful port of vLLM's
> allocation policy — the free list, the block table, copy-on-write on fork.
> The resulting block addresses get mapped onto real DRAM coordinates. Then I
> replay each sequence as all-bank commands and measure what the accelerator
> would actually achieve.
>
> Five checks run as tests on every change, two of them anchors with known
> answers — a perfect layout has to hit the theoretical ceiling, a random one
> the floor. Eighty-five tests total. Nothing gets reported unless they pass.

---

## Slide 5 — LIVE TERMINAL: the baseline (2:25–3:10)

**Figure:** none — full-screen terminal, large font.

Run this:

```bash
python -m pimkv.run --workload steady --allocator paged \
  --addrmap host-cacheline --block-tokens 16 --requests 300 --seed 0
```

Expected output (~15 s):

```
M1 row-hit rate      mean=0.8066  p05=0.7431  p95=0.9354
M2 bank parallelism  mean=0.9896
M3 effective BW      mean=1418.2 GB/s
placement            blk_adj=0.078  same_frame=0.248  frame_spread=5.06
```

> This is today's allocator on steady chat-like traffic. Ideal bandwidth for
> this hardware is 3,810 gigabytes per second. We're getting 1,418.
>
> But look at the bottom line, because that one needs no DRAM model at all.
> `blk_adj` says **eight percent** — only eight percent of a sequence's
> neighbouring blocks are physically next to each other. And `frame_spread`
> says each sequence is smeared across **five times** more alignment regions
> than its own size requires.
>
> That's not a modelling result you can argue with. That's just where the
> blocks are.

---

## Slide 6 — Headline result (3:10–4:05)

**Figure:** `figures/headline.png` — full slide.

> So I built an allocator that knows rows exist. It carves the pool into
> aligned regions, gives each sequence its own, fills it in order, and returns
> freed blocks to the region they came from.
>
> Blue is today's allocator. Orange is the row-aware one. Green is a
> contiguous-span oracle. Solid is steady traffic; dashed is speculative
> decoding.
>
> [point left] At small blocks the baseline falls apart — row-hit rate down at
> 0.54. At the standard 16 tokens it's 0.77. Orange sits flat at the ceiling,
> 0.963, at *every* block size and on *both* workloads.
>
> [point right] But look where they meet. At **128 tokens** everyone is the
> same. That number isn't tuned — it falls straight out of the hardware: row
> size, times banks, times channels, divided by bytes per token. A 128-token
> block *is* one aligned region, so where you put it stops mattering.
>
> Which gives two ways to fix this. Use 128-token blocks and the allocator
> becomes irrelevant — but you give up the fine-grained packing that made
> paging worth adopting. Or keep 16-token blocks and make the allocator
> row-aware. Same result, and you keep the packing.

---

## Slide 7 — LIVE TERMINAL: the fix (4:05–4:35)

**Figure:** none — same terminal. Ideally keep Slide 5's output visible above.

Run this — **one word changed**:

```bash
python -m pimkv.run --workload steady --allocator pim-aware \
  --addrmap host-cacheline --block-tokens 16 --requests 300 --seed 0
```

```
M1 row-hit rate      mean=0.9633
M3 effective BW      mean=2805.4 GB/s
placement            blk_adj=0.923  same_frame=0.897  frame_spread=1.00
```

> Same workload, same block size, same amount of memory. One word changed.
>
> Row-hit rate 0.81 to 0.96. Bandwidth 1,418 to 2,805 — **twice** the
> throughput. And the structural numbers invert: adjacency from 8% to 92%,
> frame spread from 5.06 down to exactly 1.00 — the minimum possible.

*(Optional, if you have 10 seconds spare — the same run with `--block-tokens 0`
derives the 128-token size and gets 0.9633 out of the **plain** allocator,
which demonstrates the other lever live.)*

---

## Slide 8 — Does the model hold up? (4:35–5:20)

**Figure:** `figures/validation.png` — full slide.

> Everything so far comes from my own model of all-bank execution. That model
> could be wrong. So I checked it against two independent cycle-level DRAM
> simulators, replaying the *same* block layouts my allocators actually
> produced.
>
> Ramulator 2 — a standard DRAM simulator with its own controller and its own
> request reordering — lands within **0.003** of my predicted row-hit rate on
> every sample.
>
> [point right panel] But stock simulators have no all-bank command, so they
> can't test the other half. AttAcc, an in-memory attention accelerator from
> ASPLOS 2024, publishes a Ramulator extension that *does* — a real all-bank
> multiply-accumulate instruction, on DRAM geometry identical to mine. I
> replayed my command streams through it as genuine all-bank instructions.
>
> My model predicts a crippled layout needs **8× more** all-bank commands.
> AttAcc measures **7.4× and 7.3× more cycles** for the two well-aligned
> allocators. The bank-parallelism term is real.

**If asked about the leftmost blue bar (5.3×):** paged's starting layout is
already degraded by row misses, so it has less left to lose — the ratio
compresses. That's why the chart shows all three rather than an average.

---

## Slide 9 — Where it doesn't help (5:20–5:55)

**Figure:** `figures/pressure.png` on the right half; four bullets on the left.

- **Oversubscribed memory** → 2.1× falls to **1.38×**
- **No controller reordering** → 2.3× falls to **1.24×**
- **Long contexts** → baseline is already fine (0.963)
- **Many KV heads** → only **1.2×** on older architectures

> A result is only useful if you know its edges, so here are mine — measured,
> not guessed.
>
> Real servers evict and recompute under memory pressure. When I model that
> faithfully and squeeze the pool to 60% of demand, re-admitted sequences land
> in whatever fragments are left and the advantage drops to 1.38×. It never
> goes *below* the baseline — but the benefit clearly needs room to work in.
>
> My headline also assumes the memory controller can reorder requests, which
> real ones do. Strip that out entirely and the gain narrows to 1.24×. The
> shape of every curve is identical; only the magnitude moves. The truth is
> between those two bounds.
>
> And long-context traffic is nearly immune under *any* allocator — a
> 20,000-token prompt is allocated in one burst and is already well laid out.
> This is a busy-server problem, not a long-context one.

---

## Slide 10 — Why this gets worse (5:55–6:15)

**Figure:** `figures/kvheads.png` — full slide.

> One last thing, and it's the reason I think this matters going forward.
>
> How much a block covers depends on how many bytes each token stores, which
> depends on the number of key/value heads. Architectures have been cutting
> that to save memory — 32 heads, down to 8, down to 1.
>
> Every cut makes this *worse*. On an older 32-head model the fix buys 1.2×.
> On today's 8-head models, 2.1×. On a 1-head model, **6×**. The direction the
> field is already moving makes the mismatch more expensive, not less.

---

## Slide 11 — Close (6:15–6:35)

**Figure:** none — three lines of text.

```
Use 128-token blocks   — the size the hardware implies
   or
Keep 16 and align      — same result, no extra memory
```

> The in-memory computing literature assumes a good data layout. The serving
> literature assumes memory is flat and placement is free. Each treats the
> other's hard part as somebody else's problem.
>
> The cheapest place to reconcile them is the allocator. A few hundred lines,
> no hardware change, no extra memory, and none of the packing efficiency that
> made paging worth adopting in the first place.

End card: report URL + `github.com/<you>/chip_hackathon_f26`

---

## Figure checklist

| Slide | Asset | Source |
|---|---|---|
| 2 | Aligned-vs-scattered diagram | SVG in report §02 (screenshot), or rebuild |
| 4 | Four-box pipeline | build in PowerPoint |
| 6 | `figures/headline.png` | generated |
| 8 | `figures/validation.png` | generated |
| 9 | `figures/pressure.png` | generated |
| 10 | `figures/kvheads.png` | generated |

**Not used in the video, keep as backup for Q&A:**
`figures/bandwidth.png` (per-workload bandwidth — good if asked "does this hold
across workloads?") and `figures/sweep_heatmap.png` (block size × context
length — good if asked "when does this *not* matter?").

---

## Recording notes

- Terminal at ~18pt minimum; the figures already use large fonts, but check
  legibility after compression.
- Rehearse the two live commands once so the 15-second waits don't feel dead —
  fill them by reading the next sentence, or trim the pause in the edit.
- Slides 5 and 7 are the spine of the demo. If you run long, cut Slide 10 down
  to one sentence over the figure; don't cut Slide 9.
- Have `docs/NOTES.md` open in another tab during Q&A — every number quoted
  above has a dated entry with the exact command that produced it.

## Likely questions

**"Did you run vLLM?"** No — it needs an NVIDIA GPU, and running real inference
would turn ~350 experiments from minutes into days. I ported its allocation
policy instead, since block placement is the only thing that matters here. The
honest gap: I reproduce the policy, not the running system. Validating that
needs one instrumented run on a GPU box.

**"Why does PIM-aware beat the contiguous oracle?"** Contiguous means each
sequence gets an unbroken span, but first-fit puts that span at whatever
address is free — so it usually straddles row-group boundaries (spread 1.145).
PIM-aware starts each sequence *at* a boundary (spread 1.000). Contiguous isn't
the same as aligned.

**"Is 2.3× the number, or 1.6×?"** 2.3× under my model, which assumes a row
miss costs a full row cycle with no overlap. AttAcc's controller overlaps
activation with useful work, and on its measured timing the same result is
about 1.6×. Every ordering and crossover point is unchanged. Quote whichever
you prefer — but state the assumption.
