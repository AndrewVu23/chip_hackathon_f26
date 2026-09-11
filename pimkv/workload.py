"""Deterministic request-stream generators (AGENT_BRIEF §5.1).

Each request: arrival time (in decode-step units), prompt length, output
length. Every generator takes a seed and is fully deterministic: same seed,
same stream, always (validation gate 4 depends on this).

Time units: one decode iteration of the serving engine == 1 step. Poisson
arrivals are therefore expressed in requests/step; with an output median of
~200 tokens (~200 steps of residency), rate * mean_output ~ steady-state
batch occupancy.

Profiles
--------
steady   Poisson arrivals; ShareGPT-like lognormal prompt/output lengths
         (median ~200 each, long tail).
longctx  8k-32k prompts, short outputs.
prefix   60-80% of requests share a common system prompt (Phase 1 —
         exercises copy-on-write).
spec     tree speculative decoding: fork W wide, D deep per step, accept a
         decaying-length path, free the rest (Phase 1 — the adversarial
         case).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Request:
    rid: int
    arrival_step: int
    prompt_len: int
    output_len: int
    prefix_id: int | None = None   # shared-prefix group (prefix workload)
    prefix_len: int = 0


def _lognormal_lengths(rng: np.random.Generator, n: int, median: float,
                       sigma: float, lo: int, hi: int) -> np.ndarray:
    x = rng.lognormal(mean=np.log(median), sigma=sigma, size=n)
    return np.clip(np.round(x), lo, hi).astype(np.int64)


def _poisson_arrivals(rng: np.random.Generator, n: int,
                      rate_per_step: float) -> np.ndarray:
    gaps = rng.exponential(scale=1.0 / rate_per_step, size=n)
    return np.floor(np.cumsum(gaps)).astype(np.int64)


def steady(n_requests: int, seed: int, *, arrival_rate: float = 0.25,
           prompt_median: float = 200.0, prompt_sigma: float = 1.0,
           output_median: float = 200.0, output_sigma: float = 0.8,
           ) -> list[Request]:
    """Steady-state continuous batching over ShareGPT-like lengths.

    Defaults: prompt median 200 (clip 8..8192), output median 200
    (clip 4..2048), 0.25 arrivals/step -> offered steady-state batch
    ~ 0.25 * E[output] ~ 68 concurrent sequences (saturates a max_batch=64
    server, which is the realistic operating point).
    """
    rng = np.random.default_rng(seed)
    arrive = _poisson_arrivals(rng, n_requests, arrival_rate)
    prompts = _lognormal_lengths(rng, n_requests, prompt_median, prompt_sigma,
                                 8, 8192)
    outputs = _lognormal_lengths(rng, n_requests, output_median, output_sigma,
                                 4, 2048)
    return [Request(rid=i, arrival_step=int(arrive[i]),
                    prompt_len=int(prompts[i]), output_len=int(outputs[i]))
            for i in range(n_requests)]


def longctx(n_requests: int, seed: int, *, arrival_rate: float = 0.2,
            ) -> list[Request]:
    """Long-context: prompts uniform 8k-32k tokens, outputs 32-256.

    Default arrival rate targets ~0.2 * E[output] ~ 30 concurrent sequences
    so the pool actually churns; at very low load any allocator looks
    fresh-pool clean."""
    rng = np.random.default_rng(seed)
    arrive = _poisson_arrivals(rng, n_requests, arrival_rate)
    prompts = rng.integers(8192, 32768 + 1, size=n_requests)
    outputs = rng.integers(32, 256 + 1, size=n_requests)
    return [Request(rid=i, arrival_step=int(arrive[i]),
                    prompt_len=int(prompts[i]), output_len=int(outputs[i]))
            for i in range(n_requests)]


def prefix(n_requests: int, seed: int, *, arrival_rate: float = 0.25,
           share_frac: float = 0.7, unique_median: float = 150.0,
           output_median: float = 200.0) -> list[Request]:
    """Prefix sharing: ``share_frac`` of requests start with one common
    system prompt of 500-1500 tokens (drawn once per seed); their private
    remainder is lognormal (median ~150). The rest are plain steady-style
    requests. Exercises the prefix cache + block sharing path (contiguous
    cannot share and pays a full private copy — reported, not hidden)."""
    rng = np.random.default_rng(seed)
    prefix_len = int(rng.integers(500, 1500 + 1))
    arrive = _poisson_arrivals(rng, n_requests, arrival_rate)
    shared = rng.random(n_requests) < share_frac
    uniq = _lognormal_lengths(rng, n_requests, unique_median, 0.8, 8, 4096)
    plain = _lognormal_lengths(rng, n_requests, 200.0, 1.0, 8, 8192)
    outputs = _lognormal_lengths(rng, n_requests, output_median, 0.8, 4, 2048)
    reqs = []
    for i in range(n_requests):
        if shared[i]:
            reqs.append(Request(rid=i, arrival_step=int(arrive[i]),
                                prompt_len=prefix_len + int(uniq[i]),
                                output_len=int(outputs[i]),
                                prefix_id=0, prefix_len=prefix_len))
        else:
            reqs.append(Request(rid=i, arrival_step=int(arrive[i]),
                                prompt_len=int(plain[i]),
                                output_len=int(outputs[i])))
    return reqs


def spec(n_requests: int, seed: int, **kw) -> list[Request]:
    """Tree speculative decoding — the adversarial case. Same arrival/length
    distributions as ``steady``; the speculation itself (fork W draft
    branches of depth D per decode round, adopt one, free the rest) is
    executed by the simulator (sim.py, --spec-width/--spec-depth/
    --spec-accept), because it is an allocation-event pattern, not a
    request-stream property."""
    return steady(n_requests, seed, **kw)


def fixedlen(n_requests: int, seed: int, *, prompt_len: int = 2048,
             output_len: int = 128, arrival_rate: float = 0.3,
             ) -> list[Request]:
    """Fixed prompt/output lengths — the controlled-context axis for the
    (block size x context length) sweep heatmap. Poisson arrivals keep the
    pool churning (batch ~ arrival_rate * output_len)."""
    rng = np.random.default_rng(seed)
    arrive = _poisson_arrivals(rng, n_requests, arrival_rate)
    return [Request(rid=i, arrival_step=int(arrive[i]),
                    prompt_len=prompt_len, output_len=output_len)
            for i in range(n_requests)]


WORKLOADS = {
    "steady": steady,
    "longctx": longctx,
    "prefix": prefix,
    "spec": spec,
    "fixedlen": fixedlen,
}
