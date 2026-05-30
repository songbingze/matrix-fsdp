#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
RUN_FULL_TESTS="${RUN_FULL_TESTS:-0}"
DIST_DIR="${DIST_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/matrix-fsdp-dist.XXXXXX")}"
CLEAN_BUILD_ARTIFACTS="${CLEAN_BUILD_ARTIFACTS:-1}"

HAD_BUILD_DIR=0
HAD_EGG_INFO=0
[[ -e "$REPO_ROOT/build" ]] && HAD_BUILD_DIR=1
[[ -e "$REPO_ROOT/matrix_fsdp.egg-info" ]] && HAD_EGG_INFO=1

cleanup_generated_artifacts() {
  if [[ "$CLEAN_BUILD_ARTIFACTS" != "1" ]]; then
    return
  fi
  if [[ "$HAD_BUILD_DIR" == "0" ]]; then
    rm -rf "$REPO_ROOT/build"
  fi
  if [[ "$HAD_EGG_INFO" == "0" ]]; then
    rm -rf "$REPO_ROOT/matrix_fsdp.egg-info"
  fi
}
trap cleanup_generated_artifacts EXIT

run_cmd() {
  printf "\n+"
  printf " %q" "$@"
  printf "\n"
  "$@"
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON=/path/to/python and retry." >&2
  exit 2
fi

run_cmd "$PYTHON_BIN" -m pip install -e . --no-deps --no-build-isolation
run_cmd "$PYTHON_BIN" -m pytest test/fsdp/test_package_boundary.py -q

if [[ "$RUN_FULL_TESTS" == "1" ]]; then
  run_cmd "$PYTHON_BIN" -m pytest test/fsdp -q
fi

run_cmd "$PYTHON_BIN" -m pip wheel . --no-deps --no-build-isolation -w "$DIST_DIR"
run_cmd "$PYTHON_BIN" setup.py sdist --dist-dir "$DIST_DIR"

run_cmd "$PYTHON_BIN" - "$DIST_DIR" <<'PY'
from pathlib import Path
import sys
import tarfile
import zipfile

dist_dir = Path(sys.argv[1])
wheels = sorted(dist_dir.glob("matrix_fsdp-*.whl"))
sdists = sorted(dist_dir.glob("matrix_fsdp-*.tar.gz")) + sorted(dist_dir.glob("matrix-fsdp-*.tar.gz"))

if len(wheels) != 1:
    raise SystemExit(f"expected exactly one wheel in {dist_dir}, found {wheels}")
if len(sdists) != 1:
    raise SystemExit(f"expected exactly one sdist in {dist_dir}, found {sdists}")

with zipfile.ZipFile(wheels[0]) as wheel:
    names = set(wheel.namelist())
    if "matrix_fsdp/__init__.py" not in names:
        raise SystemExit(f"{wheels[0].name} is missing matrix_fsdp/__init__.py")
    if not any(name.endswith(".dist-info/METADATA") for name in names):
        raise SystemExit(f"{wheels[0].name} is missing dist-info metadata")

with tarfile.open(sdists[0]) as sdist:
    names = set(sdist.getnames())
    suffixes = ("matrix_fsdp/__init__.py", "pyproject.toml", "README.md")
    missing = [suffix for suffix in suffixes if not any(name.endswith(suffix) for name in names)]
    if missing:
        raise SystemExit(f"{sdists[0].name} is missing {missing}")

print(f"validated wheel: {wheels[0]}")
print(f"validated sdist: {sdists[0]}")
PY

run_cmd git diff --check

echo
echo "Package check passed. Artifacts are in: $DIST_DIR"
