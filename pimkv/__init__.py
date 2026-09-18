"""pimkv — PIM-aware KV-cache allocation study.

Measures the DRAM row-alignment cost that paged LLM-serving allocators
(vLLM-style PagedAttention block managers) impose on all-bank PIM attention
execution, and evaluates a PIM-aware allocator that recovers it.

Modules
-------
config     : DRAM geometry / PIM timing / model-shape dataclasses and presets.
addrmap    : linear physical KV address -> (channel, bank_group, bank, row, col).
workload   : deterministic request-stream generators (steady, longctx, prefix, spec).
allocator  : allocation policies (paged vLLM-port baseline, contiguous oracle,
             random, pim-aware).
pimmodel   : analytical PIM metrics M1..M4 (row-hit rate, bank parallelism,
             effective bandwidth, decode latency).
sim        : continuous-batching event loop tying the above together.
run        : CLI entry point (python -m pimkv.run).
ramulator  : Ramulator 2 trace emission / cross-validation.
"""

__version__ = "0.1.0"
