#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python}"
BUILD_GIN="${MATRIX_FSDP_BUILD_GIN_KERNELS:-0}"

export MATRIX_FSDP_BUILD_CUDA_KERNELS=1
export MATRIX_FSDP_BUILD_GIN_KERNELS="${BUILD_GIN}"

"${PYTHON_BIN}" -m pip install -e . --no-build-isolation "$@"

"${PYTHON_BIN}" - <<'PY'
from matrix_fsdp.kernels.native import native_kernel_status

status = native_kernel_status()
print(status)
if not status.available:
    raise SystemExit("MatrixFSDP CUDA extension is unavailable after build.")

import matrix_fsdp._matrix_fsdp_cuda as extension

required = {
    "copy_range",
    "copy_rank_segments_to_full",
    "destroy_nccl_comm",
    "get_nccl_unique_id",
    "group_broadcast_rank_segments",
    "init_nccl_comm",
    "nccl_comm_lanes_supported",
    "reduce_rank_chunks",
    "sendrecv_rank_chunks",
    "sendrecv_rank_segments",
}
missing = sorted(name for name in required if not hasattr(extension, name))
if missing:
    raise SystemExit(f"MatrixFSDP CUDA extension is missing symbols: {missing}")
print("MatrixFSDP CUDA extension build/probe passed.")
PY
