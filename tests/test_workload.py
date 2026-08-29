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


def test_phase1_workloads_are_explicit_stubs():
    with pytest.raises(NotImplementedError):
        prefix(10, seed=0)
    with pytest.raises(NotImplementedError):
        spec(10, seed=0)
