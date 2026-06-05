#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON:-python}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/benchmark_results/gpu_regression_$TIMESTAMP}"
SUMMARY_FILE="$OUT_DIR/summary.txt"

WORLD_SIZE="${WORLD_SIZE:-}"
if [[ -z "$WORLD_SIZE" ]]; then
  WORLD_SIZE="$("$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
fi

MODEL="${MODEL:-transformer_split_qkv}"
UNIT="${UNIT:-block}"
LAYERS="${LAYERS:-8}"
HIDDEN="${HIDDEN:-1024}"
INTERMEDIATE="${INTERMEDIATE:-4096}"
SEQ_LEN="${SEQ_LEN:-512}"
HEADS="${HEADS:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
STEPS="${STEPS:-10}"

CORRECTNESS_LAYERS="${CORRECTNESS_LAYERS:-2}"
CORRECTNESS_HIDDEN="${CORRECTNESS_HIDDEN:-256}"
CORRECTNESS_INTERMEDIATE="${CORRECTNESS_INTERMEDIATE:-1024}"
CORRECTNESS_SEQ_LEN="${CORRECTNESS_SEQ_LEN:-128}"
CORRECTNESS_HEADS="${CORRECTNESS_HEADS:-8}"

USE_ACTIVATION_CHECKPOINT="${USE_ACTIVATION_CHECKPOINT:-0}"
USE_CHECKPOINT_WRAPPER="${USE_CHECKPOINT_WRAPPER:-1}"
RUN_CORRECTNESS="${RUN_CORRECTNESS:-1}"
RUN_FULLY_SHARD_API="${RUN_FULLY_SHARD_API:-1}"
RUN_ADAMW="${RUN_ADAMW:-1}"
RUN_MUON="${RUN_MUON:-1}"

CUSTOM_ALLGATHERV_IMPL="${MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL:-auto}"
CUSTOM_REDUCE_SCATTERV_IMPL="${MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL:-native_reduce}"
MATRIX_WORKSPACE_CACHE_PER_KEY="${MATRIX_WORKSPACE_CACHE_PER_KEY:-1}"

mkdir -p "$OUT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_gpu_regression.sh

Environment overrides:
  PYTHON=python
  WORLD_SIZE=8
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  LAYERS=8 HIDDEN=1024 INTERMEDIATE=4096 SEQ_LEN=512 HEADS=16
  WARMUP_STEPS=3 STEPS=10
  USE_ACTIVATION_CHECKPOINT=1 USE_CHECKPOINT_WRAPPER=1
  RUN_CORRECTNESS=1 RUN_FULLY_SHARD_API=1 RUN_ADAMW=1 RUN_MUON=1
  MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=auto
  MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL=native_reduce
  MATRIX_WORKSPACE_CACHE_PER_KEY=1

Outputs:
  benchmark_results/gpu_regression_<timestamp>/*.log
  benchmark_results/gpu_regression_<timestamp>/summary.txt
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

log_header() {
  {
    echo "MatrixFSDP GPU regression"
    echo "generated_at=$TIMESTAMP"
    echo "repo_root=$REPO_ROOT"
    echo "out_dir=$OUT_DIR"
    echo "world_size=$WORLD_SIZE"
    echo "model=$MODEL"
    echo "unit=$UNIT"
    echo "layers=$LAYERS"
    echo "hidden=$HIDDEN"
    echo "intermediate=$INTERMEDIATE"
    echo "seq_len=$SEQ_LEN"
    echo "heads=$HEADS"
    echo "batch_size=$BATCH_SIZE"
    echo "dtype=$DTYPE"
    echo "warmup_steps=$WARMUP_STEPS"
    echo "steps=$STEPS"
    echo "use_activation_checkpoint=$USE_ACTIVATION_CHECKPOINT"
    echo "use_checkpoint_wrapper=$USE_CHECKPOINT_WRAPPER"
    echo "custom_allgatherv_impl=$CUSTOM_ALLGATHERV_IMPL"
    echo "custom_reduce_scatterv_impl=$CUSTOM_REDUCE_SCATTERV_IMPL"
    echo "matrix_workspace_cache_per_key=$MATRIX_WORKSPACE_CACHE_PER_KEY"
    echo
  } > "$SUMMARY_FILE"
}

run_logged() {
  local name="$1"
  shift
  local log_file="$OUT_DIR/$name.log"

  {
    echo
    echo "== $name =="
    printf '+'
    printf ' %q' "$@"
    echo
  } | tee -a "$SUMMARY_FILE"

  set +e
  "$@" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e

  {
    echo "exit_code=$rc"
    echo "log_file=$log_file"
  } | tee -a "$SUMMARY_FILE"
  return "$rc"
}

append_table_lines() {
  local name="$1"
  local log_file="$OUT_DIR/$name.log"
  [[ -f "$log_file" ]] || return 0
  {
    echo
    echo "-- $name key lines --"
    grep -E '^(mode|----|fsdp2|matrix_|loss max abs diff|.* vs fsdp2)' "$log_file" || true
  } >> "$SUMMARY_FILE"
}

checkpoint_args=()
if [[ "$USE_ACTIVATION_CHECKPOINT" == "1" ]]; then
  checkpoint_args+=(--activation-checkpoint)
  if [[ "$USE_CHECKPOINT_WRAPPER" == "1" ]]; then
    checkpoint_args+=(--activation-checkpoint-wrapper)
  fi
fi

common_args=(
  --device cuda
  --world-size "$WORLD_SIZE"
  --model "$MODEL"
  --unit "$UNIT"
  --layers "$LAYERS"
  --hidden "$HIDDEN"
  --intermediate "$INTERMEDIATE"
  --seq-len "$SEQ_LEN"
  --heads "$HEADS"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --warmup-steps "$WARMUP_STEPS"
  --steps "$STEPS"
  "${checkpoint_args[@]}"
)

correctness_common_args=(
  --device cuda
  --world-size "$WORLD_SIZE"
  --model "$MODEL"
  --unit "$UNIT"
  --layers "$CORRECTNESS_LAYERS"
  --hidden "$CORRECTNESS_HIDDEN"
  --intermediate "$CORRECTNESS_INTERMEDIATE"
  --seq-len "$CORRECTNESS_SEQ_LEN"
  --heads "$CORRECTNESS_HEADS"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --warmup-steps 1
  --steps 1
  "${checkpoint_args[@]}"
)

log_header

run_logged "system_info" bash -lc \
  'hostname; date; nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true'

run_logged "torch_matrix_info" "$PYTHON_BIN" - <<'PY'
import torch
from matrix_fsdp.kernels.custom_collectives import (
    custom_reduce_scatterv_impl,
    resolve_custom_allgatherv_impl,
)
from matrix_fsdp.kernels.native import native_kernel_status
from matrix_fsdp.layout import LayoutSegment

rank_chunks = ((LayoutSegment(0, 2, 0),), (LayoutSegment(2, 5, 0),))
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("has_muon", hasattr(torch.optim, "Muon"))
print("native_kernel_status", native_kernel_status())
print("custom_allgatherv_rank_chunk_impl", resolve_custom_allgatherv_impl(rank_chunks))
print("custom_reduce_scatterv_impl", custom_reduce_scatterv_impl())
PY

if [[ "$RUN_FULLY_SHARD_API" == "1" ]]; then
  run_logged "fully_shard_api_adamw" "$PYTHON_BIN" scripts/compare_fully_shard_api.py \
    --device cuda \
    --world-size "$WORLD_SIZE" \
    --model "$MODEL" \
    --layers "$LAYERS" \
    --hidden "$HIDDEN" \
    --intermediate "$INTERMEDIATE" \
    --seq-len "$SEQ_LEN" \
    --heads "$HEADS" \
    --batch-size "$BATCH_SIZE" \
    --optimizer adamw \
    --dtype "$DTYPE" \
    --warmup-steps "$WARMUP_STEPS" \
    --steps "$STEPS" \
    --phase-timing || true
fi

if [[ "$RUN_ADAMW" == "1" ]]; then
  run_logged "adamw_phase" "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${common_args[@]}" \
    --optimizer adamw \
    --mode fsdp2 \
    --mode matrix_default \
    --phase-timing \
    --runtime-summary || true
fi

if [[ "$RUN_MUON" == "1" ]]; then
  run_logged "muon_phase" env \
    "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=$CUSTOM_ALLGATHERV_IMPL" \
    "MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL=$CUSTOM_REDUCE_SCATTERV_IMPL" \
    "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${common_args[@]}" \
      --optimizer muon \
      --mode fsdp2 \
      --mode matrix_owner_muon_role_greedy_custom_collective \
      --matrix-max-cached-elastic-workspaces-per-key "$MATRIX_WORKSPACE_CACHE_PER_KEY" \
      --phase-timing \
      --runtime-summary || true
fi

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
  run_logged "adamw_correctness" "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${correctness_common_args[@]}" \
    --optimizer adamw \
    --mode fsdp2 \
    --mode matrix_default \
    --correctness \
    --candidate-mode matrix_default || true

  if "$PYTHON_BIN" - <<'PY'
import torch
raise SystemExit(0 if hasattr(torch.optim, "Muon") else 1)
PY
  then
    run_logged "muon_correctness" env \
      "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=$CUSTOM_ALLGATHERV_IMPL" \
      "MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL=$CUSTOM_REDUCE_SCATTERV_IMPL" \
      "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${correctness_common_args[@]}" \
        --optimizer muon \
        --mode fsdp2 \
        --mode matrix_owner_muon_role_greedy_custom_collective \
        --matrix-max-cached-elastic-workspaces-per-key "$MATRIX_WORKSPACE_CACHE_PER_KEY" \
        --correctness \
        --candidate-mode matrix_owner_muon_role_greedy_custom_collective || true
  else
    echo "muon_correctness skipped: torch.optim.Muon unavailable" | tee -a "$SUMMARY_FILE"
  fi
fi

append_table_lines "fully_shard_api_adamw"
append_table_lines "adamw_phase"
append_table_lines "muon_phase"
append_table_lines "adamw_correctness"
append_table_lines "muon_correctness"

echo
echo "GPU regression logs written to $OUT_DIR"
echo "Summary: $SUMMARY_FILE"
