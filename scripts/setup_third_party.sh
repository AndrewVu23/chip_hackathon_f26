#!/usr/bin/env bash
# Fetch and build the external DRAM simulators this project cross-validates
# against. Everything lands under third_party/ and nothing is installed
# system-wide; no sudo is needed anywhere.
#
#   ./scripts/setup_third_party.sh              # venv + both simulators + NeuPIMs source
#   ./scripts/setup_third_party.sh ramulator    # stock Ramulator 2 only (M1)
#   ./scripts/setup_third_party.sh attacc       # AttAcc all-bank PIM only (M2)
#   ./scripts/setup_third_party.sh neupims      # NeuPIMs source only (read, not built)
#   ./scripts/setup_third_party.sh --check      # report what is present, build nothing
#
# Safe to re-run: each step is skipped when its output already exists.
# Pass --force to rebuild a step from scratch.
#
# The pins and the reasoning behind them live in third_party/README.md.
set -euo pipefail

RAMULATOR_URL="https://github.com/CMU-SAFARI/ramulator2.git"
RAMULATOR_SHA="9ac28d3f60564c86a1eeb53a373929b17569360b"
ATTACC_URL="https://github.com/scale-snu/attacc_simulator.git"
ATTACC_SHA="c60005143a6b492d7ef83231723386478b59a506"
# AttAcc's PIM extension targets a Ramulator commit that is not reachable
# from any branch (upstream restructured the tree), so it is fetched by SHA.
ATTACC_RAM_SHA="b7c70275f04126c647edb989270cc429776955d1"
# NeuPIMs (ASPLOS '24) is cloned to read its KV allocator and scheduler, not
# built: it needs gcc 8.3 + conan 1.57 (their Docker image), and running it
# unmodified adds nothing — its allocator is row-aligned by construction.
NEUPIMS_URL="https://github.com/casys-kaist/NeuPIMs.git"
NEUPIMS_SHA="f299af3fc8f20077e816f2e48313294cddb4bd7c"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"
VENV_PY="$ROOT/.venv/bin/python"

bold=$'\033[1m'; dim=$'\033[2m'; red=$'\033[31m'; grn=$'\033[32m'; off=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$bold" "$off" "$*"; }
skip() { printf '%s    skip%s %s\n' "$dim" "$off" "$*"; }
ok()   { printf '%s    ok%s   %s\n' "$grn" "$off" "$*"; }
die()  { printf '%s    error%s %s\n' "$red" "$off" "$*" >&2; exit 1; }

# Shell profiles that export a self-referencing CXXFLAGS make cmake and make
# fail with a recursive-variable error; clear them for every C++ invocation.
cxx() { env -u CXXFLAGS -u CFLAGS -u LDFLAGS "$@"; }

jobs_n() {
  if command -v nproc >/dev/null 2>&1; then nproc
  elif command -v sysctl >/dev/null 2>&1; then sysctl -n hw.ncpu
  else echo 4; fi
}

# GNU sed wants -i, BSD/macOS sed wants -i ''.
sed_i() { if sed --version >/dev/null 2>&1; then sed -i "$@"; else sed -i '' "$@"; fi; }

# Clone at an exact commit, fetching it explicitly when it is unreachable.
pin_clone() {
  local url="$1" sha="$2" dest="$3"
  if [ -d "$dest/.git" ]; then
    if [ "$(git -C "$dest" rev-parse HEAD)" = "$sha" ]; then
      skip "$(basename "$dest") already at ${sha:0:8}"; return 0
    fi
    say "re-pinning $(basename "$dest") to ${sha:0:8}"
  else
    say "cloning $(basename "$dest")"
    git clone --quiet "$url" "$dest"
  fi
  git -C "$dest" fetch --quiet origin "$sha" 2>/dev/null || true
  git -C "$dest" -c advice.detachedHead=false checkout --quiet "$sha"
  ok "$(basename "$dest") at ${sha:0:8}"
}

require() {
  for t in "$@"; do
    command -v "$t" >/dev/null 2>&1 || die "missing '$t' — install it and re-run"
  done
}

# ---------------------------------------------------------------- python env
VENV_DONE=0
setup_venv() {
  [ "$VENV_DONE" = 1 ] && return 0
  VENV_DONE=1
  if [ -x "$VENV_PY" ] && "$VENV_PY" -c 'import pimkv' 2>/dev/null; then
    skip "python env ready ($("$VENV_PY" --version))"; return 0
  fi
  say "creating the python environment"
  command -v uv >/dev/null 2>&1 || die \
    "uv not found. Install it (https://docs.astral.sh/uv/) or create the venv
    manually: python3.13 -m venv .venv && .venv/bin/pip install -e . pytest"
  ( cd "$ROOT" && uv venv --python python3.13 .venv \
      && uv pip install -p .venv/bin/python -e . pytest )
  ok "python env ready"
}

# ------------------------------------------------ stock Ramulator 2 (M1 check)
setup_ramulator() {
  require git cmake make
  [ -x "$VENV_PY" ] || die "python env missing — run this script with no arguments first"
  local dir="$TP/ramulator2"
  local so_glob="$dir/python/ramulator/_ramulator*.so"

  if [ "$FORCE" = 0 ] && compgen -G "$so_glob" >/dev/null; then
    skip "ramulator2 bindings already built"; return 0
  fi
  pin_clone "$RAMULATOR_URL" "$RAMULATOR_SHA" "$dir"

  # Apple clang 21 rejects a dependent-template call in the pinned source.
  # Kept as a patch so the pin stays exact; harmless to skip if applied.
  if git -C "$dir" apply --check "$TP/patches/ramulator2-apple-clang21-param-template.patch" 2>/dev/null; then
    say "applying the dependent-template fix"
    git -C "$dir" apply "$TP/patches/ramulator2-apple-clang21-param-template.patch"
  else
    skip "dependent-template fix already applied (or not needed)"
  fi

  say "building ramulator2 python bindings (a few minutes)"
  mkdir -p "$dir/build-py"
  ( cd "$dir/build-py" \
    && cxx cmake .. -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$VENV_PY" >/dev/null \
    && cxx make -j"$(jobs_n)" >/dev/null )
  compgen -G "$so_glob" >/dev/null || die "build finished but no _ramulator*.so was produced"
  ok "ramulator2 bindings built — try: $VENV_PY -m pimkv.ramulator --help"
}

# ------------------------------------------- AttAcc all-bank PIM (M2 check)
setup_attacc() {
  require git cmake make
  local dir="$TP/attacc_simulator"
  local bin="$dir/ramulator2/build/ramulator2"

  if [ "$FORCE" = 0 ] && [ -x "$bin" ]; then
    skip "attacc PIM ramulator already built"; return 0
  fi
  pin_clone "$ATTACC_URL" "$ATTACC_SHA" "$dir"

  if [ "$FORCE" = 1 ] || [ ! -d "$dir/ramulator2/.git" ]; then
    say "fetching the ramulator2 base AttAcc patches against"
    rm -rf "$dir/ramulator2"
    git clone --quiet "$RAMULATOR_URL" "$dir/ramulator2"
    git -C "$dir/ramulator2" fetch --quiet origin "$ATTACC_RAM_SHA"
    git -C "$dir/ramulator2" -c advice.detachedHead=false checkout --quiet "$ATTACC_RAM_SHA"

    # copies the PIM sources in and applies AttAcc's 21 patches
    say "applying AttAcc's PIM_MAC_AB extension"
    ( cd "$dir" && bash set_pim_ramulator.sh >/dev/null )
    sed_i 's/return _config\[_name\]\.as<T>();/return _config[_name].template as<T>();/' \
      "$dir/ramulator2/src/base/param.h"
  else
    skip "attacc ramulator sources already patched"
  fi

  say "building the attacc PIM ramulator (a few minutes)"
  mkdir -p "$dir/ramulator2/build"
  ( cd "$dir/ramulator2/build" \
    && cxx cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null \
    && cxx make -j"$(jobs_n)" >/dev/null )
  [ -x "$bin" ] || die "build finished but $bin is missing"
  ok "attacc PIM ramulator built — try: $VENV_PY -m pimkv.attacc --help"
}

# ---------------------------------------------- NeuPIMs source (reference)
# Submodules (booksim, FlameGraph) are only needed to build, so they are not
# fetched; the in-house DRAM simulator (extern/NewtonSim) is in the main repo.
setup_neupims() {
  require git
  pin_clone "$NEUPIMS_URL" "$NEUPIMS_SHA" "$TP/neupims"
}

# --------------------------------------------------------------------- report
report() {
  printf '\n%sstatus%s\n' "$bold" "$off"
  local py="missing"; [ -x "$VENV_PY" ] && py="$("$VENV_PY" --version 2>&1)"
  printf '  python env           %s\n' "$py"
  if compgen -G "$TP/ramulator2/python/ramulator/_ramulator*.so" >/dev/null
    then printf '  ramulator2 (M1)      built\n'
    else printf '  ramulator2 (M1)      not built\n'; fi
  if [ -x "$TP/attacc_simulator/ramulator2/build/ramulator2" ]
    then printf '  attacc PIM (M2)      built\n'
    else printf '  attacc PIM (M2)      not built\n'; fi
  if [ -d "$TP/neupims/.git" ]
    then printf '  neupims (source)     cloned at %s\n' "$(git -C "$TP/neupims" rev-parse --short=8 HEAD)"
    else printf '  neupims (source)     not cloned\n'; fi
  printf '  vllm_ref             committed in-tree (no fetch needed)\n\n'
}

FORCE=0
TARGETS=()
for arg in "$@"; do
  case "$arg" in
    --force)  FORCE=1 ;;
    --check)  report; exit 0 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
                 "${BASH_SOURCE[0]}"; exit 0 ;;
    ramulator|attacc|neupims|venv) TARGETS+=("$arg") ;;
    *) die "unknown argument '$arg' (try --help)" ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(venv ramulator attacc neupims)

for t in "${TARGETS[@]}"; do
  case "$t" in
    venv)      setup_venv ;;
    ramulator) setup_venv; setup_ramulator ;;
    attacc)    setup_venv; setup_attacc ;;
    neupims)   setup_neupims ;;
  esac
done
report
say "cross-validation runs:"
printf '    make ramulator     # stock Ramulator 2 -> results/phaseE/\n'
printf '    make attacc        # AttAcc all-bank PIM -> results/phaseE2/\n'
