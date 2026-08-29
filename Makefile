PY := .venv/bin/python

.PHONY: venv install test kill-test reproduce clean

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

# Regenerates every figure from scratch (grows as phases land).
reproduce: kill-test
	$(PY) -m pimkv.plots m1-hist results/phase0/steady_paged_hostcentric.csv \
	  figures/phase0_m1_hist.png

clean:
	rm -rf results figures .pytest_cache
