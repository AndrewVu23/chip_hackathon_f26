# Paged for Capacity, Punished by Rows: PIM-Aware KV-Cache Allocation for LLM Decode

**Kiet Vu** — Computer Engineering, Purdue University
**Theme:** 4 (In-Memory Computing and Compute-Near-Memory), cross-cutting with 2 (System-Level AI) and 3 (Memory-Hierarchy Optimization)

## Problem and Motivation

LLM decode is memory-bound. Each generated token streams the entire KV cache through a low-arithmetic-intensity GEMV, and a growing body of near-memory work — AttAcc, NeuPIMs, IANUS, Duplex, LoL-PIM, AttenPIM — attacks this by computing inside DRAM banks. All of it rests on one hardware mechanism: activating the same row across every bank in a channel and broadcasting a single column command, which multiplies effective bandwidth by roughly the bank count.

That mechanism requires the KV cache to be physically laid out in a specific way. Every published PIM-attention design assumes it is. Every production serving system guarantees it is not. PagedAttention splits the KV cache into fixed-size blocks placed non-contiguously and reached through a block table, allocated and freed at runtime, returned out of order, and shared across request prefixes. The abstraction that made serving memory-efficient is the abstraction that breaks bank-parallel execution.

There is already evidence this matters under the *easy* conditions. A recent Ramulator 2 extension of the AttAcc simulator found that for decode GEMV the DRAM row cycle time nRC is 10–11× larger than the nCCDAB power constraint prior work optimized against, and traced the cause to host-centric address interleaving forcing every all-bank MAC command into a different row. That result assumes static, architect-chosen placement. A runtime paged allocator is strictly worse. Nobody has measured how much worse, and the serving and PIM literatures each assume the other side's problem away.

## Technical Approach

I will quantify the mismatch and then fix it in the allocator, where it is cheapest to fix.

**1. Allocator trace capture.** Port vLLM's block-manager allocation semantics (first-fit over a LIFO free list, block table, prefix-sharing copy-on-write) into a standalone harness, and replay realistic request arrivals: steady-state continuous batching over ShareGPT-like length distributions, long-context, prefix-shared, and — as the adversarial case — tree-drafted speculative decoding, which forks and rewinds the cache several times per token.

**2. Physical mapping.** Push the resulting block-ID stream through a configurable DRAM address map (channel / bank-group / bank / row / column interleaving) parameterized for HBM3-PIM and GDDR-PIM geometries.

**3. Measurement.** Report the metric no PIM-attention paper reports: the fraction of all-bank PIM MAC commands that land in an already-open row, plus achieved bank-parallelism and the resulting effective PIM bandwidth. Cross-validate the analytical model against a Ramulator 2 command-stream replay.

**4. The fix.** A PIM-aware KV block allocator: bank-group-aligned block placement, block size derived from row size and banks-per-PIM-unit rather than chosen as a serving-side knob, and alignment-preserving reuse on free.

## Prototype and Evaluation

The deliverable is a working simulation harness plus the allocator, entirely CPU-side — no GPU required. Success is a measured recovery in PIM row-hit rate and effective bandwidth from the paged baseline toward the contiguous-placement upper bound, reported across all four workloads and swept over block size, batch size, and context length. Two sanity anchors bracket every result: contiguous placement with row-sized blocks must approach a 1.0 row-hit rate, and random placement must approach 1/num_rows.

The headline figure is a single plot — row-hit rate versus block size for paged, PIM-aware, and contiguous allocation, with the speculative-decoding curve showing the gap widening in the direction the field is moving. The demo video walks through the measurement, the fix, and the resulting design recommendation: what block size a PIM-backed serving system should actually use, and why the value the serving community converged on is the wrong one.

**What would falsify this:** if a real allocator's free-list reuse already produces near-sequential physical placement under steady-state load, the gap does not exist. I am running that check before submission, and the measured baseline number goes in this proposal's motivating figure either way.
