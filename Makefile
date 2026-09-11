PY := .venv/bin/python

.PHONY: venv install test kill-test sweep credibility ramulator reproduce clean

venv:
	uv venv --python python3.13 .venv

install: venv
	uv pip install -p .venv/bin/python -e . pytest

test:
	$(PY) -m pytest tests/

# Phase 0: baseline PIM row-hit rate under realistic paged allocation,
# plus the three bracketing reference runs.
kill-test:
	$(PY) -m pimkv.run --workload steady --allocator paged --dram hbm3-pim \
	  --addrmap host-centric --block-tokens 16 --requests 2000 --seed 0 \
	  --out results/phase0/steady_paged_hostcentric.csv
	$(PY) -m pimkv.run --workload steady --allocator paged --dram hbm3-pim \
	  --addrmap pim-friendly --block-tokens 16 --requests 2000 --seed 0 \
	  --out results/phase0/steady_paged_pimfriendly.csv
	$(PY) -m pimkv.run --workload steady --allocator contiguous --dram hbm3-pim \
	  --addrmap pim-friendly --block-tokens 16 --requests 2000 --seed 0 \
	  --out results/phase0/steady_contig_pimfriendly.csv
	$(PY) -m pimkv.run --workload steady --allocator paged --dram hbm3-pim \
	  --addrmap host-cacheline --block-tokens 16 --requests 2000 --seed 0 \
	  --out results/phase0/steady_paged_hostcacheline.csv

# Phase 2 headline sweep: 48 runs (~15 min with 4 workers).
sweep:
	$(PY) -m pimkv.sweep --config configs/headline.yaml \
	  --out results/headline/ --jobs 4
	$(PY) -m pimkv.sweep --config configs/heatmap.yaml \
	  --out results/heatmap/ --jobs 4

# Regenerates every figure from scratch (grows as phases land).
reproduce: kill-test sweep
	$(PY) -m pimkv.plots m1-hist results/phase0/steady_paged_hostcentric.csv \
	  figures/phase0_m1_hist.png
	$(PY) -m pimkv.plots headline results/headline/summary.csv \
	  figures/headline.png
	$(PY) -m pimkv.plots bandwidth results/headline/summary.csv \
	  figures/bandwidth.png
	$(PY) -m pimkv.plots heatmap results/heatmap/summary.csv \
	  figures/sweep_heatmap.png

clean:
	rm -rf results figures .pytest_cache

# Credibility sweeps (seeds, reorder window, headroom, sharding, KV heads,
# preemption, spec fix). ~2 h total at 8 workers; each writes summary.csv.
credibility:
	for c in phaseA phaseB phaseC phaseK phaseB2 phaseD phaseD_specfix headline_specfix; do \
	  $(PY) -m pimkv.sweep --config configs/$$c.yaml --out results/$$c/ --jobs 8; done

# Phase 3: Ramulator 2 row-locality cross-validation (needs the bindings;
# build recipe in third_party/README.md).
ramulator:
	$(PY) -m pimkv.ramulator --workload steady --block-tokens 16 --samples 3 \
	  --out results/phaseE/
