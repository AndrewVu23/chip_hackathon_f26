"""Tests for pimkv.workload."""
import numpy as np
import pytest

from pimkv.workload import steady, longctx, prefix, spec


def test_deterministic():
    a = steady(500, seed=7)
    b = steady(500, seed=7)
    assert a == b
    c = steady(500, seed=8)
    assert a != c


def test_steady_shapes():
    reqs = steady(4000, seed=0)
    arr = np.array([r.arrival_step for r in reqs])
    assert np.all(np.diff(arr) >= 0), "arrivals must be non-decreasing"
    prompts = np.array([r.prompt_len for r in reqs])
    outputs = np.array([r.output_len for r in reqs])
    assert 150 < np.median(prompts) < 260
    assert 150 < np.median(outputs) < 260
    assert prompts.min() >= 8 and prompts.max() <= 8192
    assert outputs.min() >= 4 and outputs.max() <= 2048
    # long tail exists
    assert np.percentile(prompts, 99) > 3 * np.median(prompts)


def test_longctx_shapes():
    reqs = longctx(200, seed=0)
    prompts = np.array([r.prompt_len for r in reqs])
    outputs = np.array([r.output_len for r in reqs])
    assert prompts.min() >= 8192 and prompts.max() <= 32768
    assert outputs.max() <= 256


def test_prefix_shapes():
    reqs = prefix(1000, seed=0)
    shared = [r for r in reqs if r.prefix_id is not None]
    assert 0.6 < len(shared) / len(reqs) < 0.8
    plens = {r.prefix_len for r in shared}
    assert len(plens) == 1                    # one common system prompt
    pl = plens.pop()
    assert 500 <= pl <= 1500
    assert all(r.prompt_len > r.prefix_len for r in shared)
    assert prefix(1000, seed=0) == reqs       # deterministic


def test_spec_is_steady_lengths():
    # speculation is an allocation pattern executed by the simulator;
    # the request stream itself matches steady
    assert spec(100, seed=5) == __import__("pimkv.workload",
                                           fromlist=["steady"]).steady(100, 5)
