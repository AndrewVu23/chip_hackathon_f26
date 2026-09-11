# third_party

External code this project depends on or ports from. The two simulator
clones are **not** committed (see `.gitignore`); re-fetch them with the
commands below. The vLLM reference files **are** committed (Apache-2.0,
attribution below) because the allocator port cites them line-by-line.

## vllm_ref/ (committed)

`block_manager.py` and `block.py` from **vLLM v0.2.7**, commit
`2e0b6e775756345aa1d39f772c186e00f8c29e92`, path `vllm/core/block_manager.py`
and `vllm/block.py`. Apache License 2.0, © vLLM contributors.
`pimkv/allocator.py::PagedFirstFit` is a semantics port of
`BlockAllocator` in these files; its docstring cites the exact lines.

## ramulator2/ (clone, not committed)

```bash
git clone https://github.com/CMU-SAFARI/ramulator2.git third_party/ramulator2
git -C third_party/ramulator2 checkout 9ac28d3f60564c86a1eeb53a373929b17569360b
# Apple clang 21 rejects a dependent-template call in the pinned source;
# one-line fix, kept as a patch so the pin stays exact:
git -C third_party/ramulator2 apply ../patches/ramulator2-apple-clang21-param-template.patch
cd third_party/ramulator2 && mkdir -p build-py && cd build-py
env -u CXXFLAGS -u CFLAGS -u LDFLAGS cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=$(git rev-parse --show-toplevel)/.venv/bin/python
env -u CXXFLAGS -u CFLAGS -u LDFLAGS make -j6
# -> python/ramulator/_ramulator.cpython-313-darwin.so (nanobind module);
#    pimkv.ramulator adds third_party/ramulator2/python to sys.path.
```

**Correction (2026-09-11):** the 2026-08-28 note that the binary "builds
on macOS" was wrong — that build had failed at 44% (`make: *** [all]
Error 2`) and the success message was echoed by a shell chain whose output
was never read. This pinned commit builds a shared library plus Python
bindings, not a standalone binary; the recipe above is the verified one.

Used in Phase 3 only (`pimkv/ramulator.py`) for cross-validating the
analytical model's direction/magnitude. The `env -u` clearing is needed on
machines whose shell profile exports self-referencing `CXXFLAGS` (Make
rejects the recursive variable). Verified to build on macOS arm64 /
Apple clang 21.

## attacc_simulator/ (clone, not committed)

```bash
git clone https://github.com/scale-snu/attacc_simulator.git third_party/attacc_simulator
git -C third_party/attacc_simulator checkout c60005143a6b492d7ef83231723386478b59a506
```

Reference only (AttAcc PIM-attention simulator, HPCA'24): consulted for
PIM command semantics and timing sanity; no code is imported from it.
