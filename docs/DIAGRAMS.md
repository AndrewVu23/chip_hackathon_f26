# Diagram drafts — for Excalidraw / draw.io / Figma

Six diagrams. Each has a layout sketch, exact label text, and notes on what to
emphasize. Redraw them by hand; the ASCII is a blueprint, not the deliverable.

**Shared palette** (matches the generated figures, so slides and charts agree):

| Role | Hex | Used for |
|---|---|---|
| Paged / "the problem" | `#2A78D6` blue | today's allocator, scattered layouts |
| PIM-aware / "the fix" | `#D4541F` orange | aligned layouts, the contribution |
| Contiguous oracle | `#0F8F63` green | upper-bound reference |
| Ink | `#0E1116` | titles, primary labels |
| Muted | `#78828F` | annotations, units, secondary text |
| Hairline | `#DCE2EA` | box borders, grid lines |
| Ground | `#FFFFFF` / `#F7F8FA` | fills |

Use **one** accent per diagram where possible. Keep fills white with hairline
borders; let color carry meaning, not decoration.

---

## D1 — The core conflict (most important; video slide 2)

This is the diagram that makes or breaks comprehension. Everything else is
detail. Canvas ≈ 1200 × 620.

```
  WHAT THE ACCELERATOR NEEDS                        [title, ink, 20pt bold]
  One row index, every bank — one command, 16 banks compute at once
                                                    [subtitle, muted, 13pt]

        bank 0      bank 1      bank 2      bank 3   ... (fade out at 15)
      ┌─────────┬─────────┬─────────┬─────────┐
row 0 │         │         │         │         │
row 1 │         │         │         │         │
row 2 │█████████│█████████│█████████│█████████│  <- ORANGE, solid, aligned
row 3 │         │         │         │         │
      └─────────┴─────────┴─────────┴─────────┘
                                          ← one all-bank command  [orange]

  WHAT THE PAGED ALLOCATOR PRODUCES
  Four different rows — four separate activations, no parallelism

        bank 0      bank 1      bank 2      bank 3
      ┌─────────┬─────────┬─────────┬─────────┐
row 0 │         │         │█████████│         │   <- BLUE, scattered
row 1 │█████████│         │         │         │
row 2 │         │         │         │█████████│
row 3 │         │█████████│         │         │
      └─────────┴─────────┴─────────┴─────────┘
                                          ← four commands, one per row [blue]
```

**Drawing notes**
- Draw only 4 banks, not 16. Add "… ×16 banks" in muted text at the right edge.
- The two grids must be **identical in size and position** so the eye compares
  rows directly. Same cell size, same spacing, vertically stacked.
- Top block: one unbroken orange bar spanning all four cells at row 2. It
  should read as *one object*, not four — consider a single rounded rect drawn
  over the grid rather than four separate fills.
- Bottom: four separate blue rects, deliberately staggered. Keep them the same
  size as the orange cells so the difference is clearly *position*, not amount.
- Optional: a small clock/stopwatch icon next to each right-hand caption —
  "1×" beside the orange, "4×" beside the blue.

**One-line takeaway to put underneath:**
> Same amount of data. Sixteen times the work, or one.

---

## D2 — The size mismatch (video slide 3)

Small, arithmetic-driven. Canvas ≈ 900 × 380. This is the "aha" for anyone
who asks *why* the allocator can't just get lucky.

```
  ONE ROW-GROUP                                      16 KB
  (one row index × 16 banks, within one channel)
  ┌────────────────────────────────────────────────────────────┐
  │                                                            │  [grey outline]
  └────────────────────────────────────────────────────────────┘

  ONE 16-TOKEN BLOCK, per channel                     2 KB
  ┌───────┐
  │▓▓▓▓▓▓▓│                                                       [blue fill]
  └───────┘
  └──────────────────────── 1/8 ───────────────────────────────┘

  → Eight consecutive blocks must land side by side
    before the accelerator gets one clean operation.
    Nothing in the free list arranges that.
```

**Drawing notes**
- The 2 KB bar must be *visually* one-eighth of the 16 KB bar. Measure it.
- Consider showing 8 slots inside the big bar with dashed dividers, then
  filling only 1 — makes the fraction self-evident without reading numbers.
- A second row underneath, in orange, showing all 8 slots filled and labeled
  "128-token block = exactly one row-group", previews the recommendation.

**Numbers (verified):** 1 KB row × 16 banks = 16 KB row-group.
16 tokens × 4 KB/token ÷ 32 channels = 2 KB per channel. Ratio 1/8.

---

## D3 — Harness architecture (video slide 4)

The system diagram. Canvas ≈ 1200 × 480. Left-to-right pipeline, four stages.

```
 ┌──────────────┐   ┌──────────────────┐   ┌───────────────┐   ┌──────────────┐
 │  WORKLOAD    │   │   ALLOCATOR      │   │  ADDRESS MAP  │   │  ALL-BANK    │
 │              │──▶│                  │──▶│               │──▶│  PIM MODEL   │
 │ steady       │   │ paged  (vLLM     │   │ block id      │   │              │
 │ long-context │   │         policy)  │   │   ↓           │   │ M1 row hits  │
 │ prefix-share │   │ pim-aware ★      │   │ channel /     │   │ M2 bank par. │
 │ speculative  │   │ contiguous       │   │ bank / row /  │   │ M3 bandwidth │
 │              │   │ (oracle)         │   │ column        │   │ M4 latency   │
 └──────────────┘   └──────────────────┘   └───────────────┘   └──────────────┘
   request            WHEN blocks are        WHERE those          what the
   streams            allocated & freed      blocks land          hardware gets

        ↑                    ↑                                        │
        │                    │                                        ▼
   deterministic      ┌──────────────┐                      ┌──────────────────┐
   per seed           │ SIMULATOR    │                      │  CROSS-CHECK     │
                      │ continuous   │                      │ Ramulator 2 (M1) │
                      │ batching     │                      │ AttAcc PIM  (M2) │
                      └──────────────┘                      └──────────────────┘

              no GPU · no model weights · laptop, minutes
```

**Drawing notes**
- Four main boxes equal width, evenly spaced, same style. The caption strip
  under each ("request streams", "WHEN blocks are…") is what makes this
  readable — don't drop it.
- Star or orange-outline the `pim-aware` line inside box 2; it's the
  contribution and should be findable at a glance.
- The two lower boxes hang *below* the main line — they're supporting, not
  sequential. Use a lighter border or grey fill.
- Bottom strapline in muted italic. It preempts "did you run a real model?"

---

## D4 — How the PIM-aware allocator works

The mechanism. Canvas ≈ 1100 × 560. Two panels, before/after.

```
  TODAY: free list                        PIM-AWARE: aligned frames
  ┌────────────────────────────┐          ┌────────────────────────────┐
  │ free: [7][3][19][2][11]... │          │ frame 0  ████████ seq A    │
  │        ↑ pop from the tail │          │ frame 1  ████░░░░ seq B    │
  └────────────────────────────┘          │ frame 2  ░░░░░░░░ empty    │
                                          │ frame 3  ██████░░ seq C    │
  seq A gets: 7, 3, 19, 2, 11             └────────────────────────────┘
             └── wherever they were ──┘    seq A gets: 0,1,2,3,4,5,6,7
                                                      └─ one frame, in order ─┘
  ┌──────────────────────────────┐
  │ blk_adj      0.069           │        ┌──────────────────────────────┐
  │ frame spread 5.41×           │        │ blk_adj      0.921           │
  └──────────────────────────────┘        │ frame spread 1.00×  ← minimum│
                                          └──────────────────────────────┘
```

**The three rules to label on the right panel:**
1. **Frame = one row index across every bank and channel** (512 KB here) —
   derived from geometry, not chosen.
2. **One sequence fills its own frame in order** before touching another.
3. **Freed blocks return to their own frame**, so alignment survives reuse.

**Drawing notes**
- Left panel should look messy — scattered numbers, arrows crossing. Right
  panel should look orderly. The visual contrast *is* the argument.
- Use blue for left, orange for right.
- The two metric boxes at the bottom are the punchline; give them equal
  weight and align them horizontally so the numbers sit side by side.

---

## D5 — DRAM hierarchy primer (backup slide / appendix)

Only include if your audience needs it. Canvas ≈ 900 × 420.

```
  HBM3 stack
  └── 32 channels                       ← independent, run in parallel
       └── 16 banks per channel         ← all-bank command targets ALL of these
            └── 16,384 rows per bank    ← only ONE can be open at a time  ⚠
                 └── 1 KB per row       ← 32 columns × 32 B bursts

  ┌──────────────────────────────────────────────────────────┐
  │  ACTIVATE row  ~45 ns   ← slow: copies the row to a buffer│  [blue]
  │  read column   ~4.3 ns  ← fast: row already open          │  [orange]
  └──────────────────────────────────────────────────────────┘
        A row miss costs ~10× a row hit. That ratio is the whole game.
```

**Drawing notes**
- Indented tree, each level a nested box or just indentation with connectors.
- Put a warning icon on "only ONE can be open at a time" — that constraint is
  the root of everything.
- The timing box: two bars drawn *to scale* (45 vs 4.3) lands harder than the
  numbers alone.

---

## D6 — Validation strategy (video slide 8 companion)

Explains *how* we checked the model without re-deriving it. Canvas ≈ 1000 × 420.

```
              real block layouts from the simulator
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌───────────────────┐       ┌────────────────────┐
    │ OUR MODEL         │       │ CYCLE-LEVEL SIMS   │
    │ analytical        │       │                    │
    │ M1 row hits       │       │ Ramulator 2        │
    │ M2 bank parallel  │       │   → row behaviour  │
    │ M3 bandwidth      │       │ AttAcc PIM ext.    │
    └───────────────────┘       │   → all-bank cmds  │
              │                 └────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
                    ┌───────────────┐
                    │   AGREEMENT   │
                    ├───────────────┤
                    │ row hits      │  within 0.003
                    │ bank parallel │  8.0× predicted / 7.4× measured
                    │ ordering      │  identical
                    └───────────────┘
```

**Drawing notes**
- Emphasize that the *same* layouts feed both sides — that's what makes it a
  validation rather than two separate experiments. A single source node at the
  top with two arrows says this better than any caption.
- The agreement box is the payload. Make it the visually heaviest element.

---

## Priority, if you only draw some

1. **D1** — without it nobody follows the talk
2. **D3** — answers "what did you actually build?"
3. **D2** — the arithmetic that makes D1 inevitable
4. **D4** — the contribution's mechanism
5. **D6** — credibility
6. **D5** — only if the audience is new to DRAM

## Reusing the existing one

The report's section 02 already contains a rendered version of D1 as inline
SVG. Open the published page, screenshot it, or lift the `<svg>` block from
`docs/report.html` — it pastes into Figma and most vector tools directly, and
Excalidraw will import it as editable shapes.
