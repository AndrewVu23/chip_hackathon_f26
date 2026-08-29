"""Phase 3 — Ramulator 2 cross-validation (stub; do not start before Phase 2
is green, per AGENT_BRIEF §5.5/§7).

Plan:
  1. Reuse the simulator's per-sequence burst enumeration to emit a DRAM
     request trace in Ramulator 2's load-store-trace format (one memory read
     per burst, physical byte addresses = linear KV address; Ramulator's own
     address mapper is configured to match pimkv.addrmap's scheme).
  2. Run third_party/ramulator2/build/ramulator2 with a YAML config generated
     from the DramGeometry preset.
  3. Parse cycle counts / row-conflict stats from the plugin output and
     compare DIRECTION and rough magnitude against M3.

Purpose is cross-validation, not precision. HARD RULE from the brief: if
Ramulator's cycle deltas disagree in SIGN with M3 across two configurations,
stop and report — the analytical model is wrong.

Build note (macOS): third_party/ramulator2 builds with Apple clang via
  env -u CXXFLAGS -u CFLAGS -u LDFLAGS cmake .. -DCMAKE_BUILD_TYPE=Release
(the env vars must be cleared if the shell profile self-references them).
Pinned commit: see third_party/README.md.
"""
from __future__ import annotations

RAMULATOR_BIN = "third_party/ramulator2/build/ramulator2"


def emit_trace(*args, **kwargs):
    raise NotImplementedError("Phase 3 — see module docstring for the plan")


def run_ramulator(*args, **kwargs):
    raise NotImplementedError("Phase 3 — see module docstring for the plan")
