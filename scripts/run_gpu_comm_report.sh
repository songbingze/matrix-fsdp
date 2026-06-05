#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/benchmark_results/gpu_comm_${TIMESTAMP}}"
REPORT="${OUT_DIR}/report.md"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_gpu_comm_report.sh

This is the one-click GPU trace/report entrypoint for MatrixFSDP communication,
phase timing, memory trace, and runtime event diagnosis. It writes logs plus a
single Markdown report under benchmark_results/gpu_comm_<timestamp>/ by default.

Common environment overrides:
  PYTHON=python
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  WORLD_SIZE=8
  MODEL=transformer_split_qkv UNIT=block
  LAYERS=32 HIDDEN=4096 INTERMEDIATE=16384 SEQ_LEN=4096 HEADS=32
  DTYPE=bfloat16
  MODEL_WARMUP_STEPS=2 MODEL_STEPS=3
  MEMORY_WARMUP_STEPS=1 MEMORY_STEPS=1
  RUNTIME_PROFILE_STEPS=1
  MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=auto
  MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL=native_reduce
  OUT_DIR=/path/to/output
  DRY_RUN=1

Optional switches:
  RUN_COMM_BENCH=0
  RUN_MODEL_PHASE=0
  RUN_MEMORY_TRACE=0
  RUN_RUNTIME_PROFILE=0
  RUN_COPYIN_BENCH=0
  RUN_PARAM_MATERIALIZATION_BENCH=0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

WORLD_SIZE="${WORLD_SIZE:-}"
if [[ -z "${WORLD_SIZE}" ]]; then
  WORLD_SIZE="$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
fi

DTYPE="${DTYPE:-bfloat16}"
RUN_COMM_BENCH="${RUN_COMM_BENCH:-1}"
RUN_MODEL_PHASE="${RUN_MODEL_PHASE:-1}"
RUN_MEMORY_TRACE="${RUN_MEMORY_TRACE:-1}"
RUN_RUNTIME_PROFILE="${RUN_RUNTIME_PROFILE:-1}"
RUN_COPYIN_BENCH="${RUN_COPYIN_BENCH:-1}"
RUN_PARAM_MATERIALIZATION_BENCH="${RUN_PARAM_MATERIALIZATION_BENCH:-1}"
COMM_WARMUP_STEPS="${COMM_WARMUP_STEPS:-3}"
COMM_STEPS="${COMM_STEPS:-8}"
MODEL_WARMUP_STEPS="${MODEL_WARMUP_STEPS:-1}"
MODEL_STEPS="${MODEL_STEPS:-3}"
MEMORY_WARMUP_STEPS="${MEMORY_WARMUP_STEPS:-1}"
MEMORY_STEPS="${MEMORY_STEPS:-1}"
RUNTIME_PROFILE_WARMUP_STEPS="${RUNTIME_PROFILE_WARMUP_STEPS:-0}"
RUNTIME_PROFILE_STEPS="${RUNTIME_PROFILE_STEPS:-1}"
COPYIN_WARMUP_STEPS="${COPYIN_WARMUP_STEPS:-3}"
COPYIN_STEPS="${COPYIN_STEPS:-20}"
PARAM_MATERIALIZATION_WARMUP_STEPS="${PARAM_MATERIALIZATION_WARMUP_STEPS:-3}"
PARAM_MATERIALIZATION_STEPS="${PARAM_MATERIALIZATION_STEPS:-20}"
MODEL="${MODEL:-transformer_split_qkv}"
UNIT="${UNIT:-block}"
LAYERS="${LAYERS:-8}"
HIDDEN="${HIDDEN:-1024}"
INTERMEDIATE="${INTERMEDIATE:-4096}"
SEQ_LEN="${SEQ_LEN:-512}"
HEADS="${HEADS:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OPTIMIZER="${OPTIMIZER:-muon}"
SHARD_SIZES="${SHARD_SIZES:-}"

if [[ -z "${SHARD_SIZES}" ]]; then
  SHARD_SIZES="$(WORLD_SIZE="${WORLD_SIZE}" "${PYTHON_BIN}" - <<'PY'
import os
world_size = int(os.environ["WORLD_SIZE"])
base = 1_000_000
print(",".join(str((world_size - rank) * base) for rank in range(world_size)))
PY
  )"
fi

write_section() {
  local title="$1"
  {
    echo
    echo "## ${title}"
    echo
  } >> "${REPORT}"
}

run_logged() {
  local name="$1"
  shift
  local log_file="${OUT_DIR}/${name}.log"
  write_section "${name}"
  {
    echo '```bash'
    printf '%q ' "$@"
    echo
    echo '```'
    echo
    echo '```text'
  } >> "${REPORT}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      echo "[dry-run] command skipped"
      echo '```'
      echo
      echo "exit_code=0"
    } >> "${REPORT}"
    printf "[dry-run] %s\n" "${name}"
    return 0
  fi
  set +e
  "$@" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e
  cat "${log_file}" >> "${REPORT}"
  {
    echo '```'
    echo
    echo "exit_code=${rc}"
  } >> "${REPORT}"
  return "${rc}"
}

run_logged_env() {
  local name="$1"
  local env_spec="$2"
  shift 2
  local log_file="${OUT_DIR}/${name}.log"
  write_section "${name}"
  {
    echo '```bash'
    echo "${env_spec} $(printf '%q ' "$@")"
    echo '```'
    echo
    echo '```text'
  } >> "${REPORT}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      echo "[dry-run] command skipped"
      echo '```'
      echo
      echo "exit_code=0"
    } >> "${REPORT}"
    printf "[dry-run] %s\n" "${name}"
    return 0
  fi
  set +e
  env ${env_spec} "$@" 2>&1 | tee "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e
  cat "${log_file}" >> "${REPORT}"
  {
    echo '```'
    echo
    echo "exit_code=${rc}"
  } >> "${REPORT}"
  return "${rc}"
}

append_quick_summary() {
  write_section "quick_summary"
  {
    echo "### Communication Microbenchmarks"
    echo
    echo '```text'
    grep -h '^impl=' "${OUT_DIR}"/comm_*.log 2>/dev/null || true
    echo '```'
    echo
    echo "### Model Phase Timing"
    echo
    for log_file in "${OUT_DIR}"/model_*.log; do
      [[ -e "${log_file}" ]] || continue
      echo "#### $(basename "${log_file}" .log)"
      echo
      echo '```text'
      grep -E '^(mode|----|matrix_|fsdp2)' "${log_file}" || true
      echo '```'
      echo
    done
    echo "### Model Memory Trace"
    echo
    for log_file in "${OUT_DIR}"/memory_*.log; do
      [[ -e "${log_file}" ]] || continue
      echo "#### $(basename "${log_file}" .log)"
      echo
      echo '```text'
      grep -E '^(mode|rank|----|fsdp2|matrix_|stage|allocated|reserved|peak|after_|before_)' "${log_file}" || true
      echo '```'
      echo
    done
    echo "### Runtime Communication Summary"
    echo
    for log_file in "${OUT_DIR}"/model_*.log; do
      [[ -e "${log_file}" ]] || continue
      echo "#### $(basename "${log_file}" .log)"
      echo
      echo '```text'
      grep -E 'gather_backends|workspace_kind|owner_segment:|matrix_all_gather:[0-9]|single_rank_copy:|equal_all_gather:|fsdp2[[:space:]].*[[:space:]]0[[:space:]]+-[[:space:]]+-' "${log_file}" || true
      echo '```'
      echo
    done
    echo "### Runtime Event Phase Timings"
    echo
    for log_file in "${OUT_DIR}"/runtime_profile_*.log; do
      [[ -e "${log_file}" ]] || continue
      echo "#### $(basename "${log_file}" .log)"
      echo
      echo '```text'
      grep -E 'native_enqueue_ms|wait_ms|workspace=|materialize=' "${log_file}" || true
      echo '```'
      echo
    done
    echo "### Runtime Profile Communication Events"
    echo
    for log_file in "${OUT_DIR}"/runtime_profile_*.log; do
      [[ -e "${log_file}" ]] || continue
      echo "#### $(basename "${log_file}" .log)"
      echo
      echo '```text'
      grep -A 12 'communication_event_stats:' "${log_file}" || true
      echo '```'
      echo
    done
  } >> "${REPORT}"
}

torchrun_cmd() {
  torchrun --standalone --nproc_per_node="${WORLD_SIZE}" "$@"
}

comm_bench_args=(
  -m matrix_fsdp.tools.comm_bench
  --dtype "${DTYPE}"
  --shard-sizes "${SHARD_SIZES}"
  --warmup-steps "${COMM_WARMUP_STEPS}"
  --steps "${COMM_STEPS}"
  --check
)

copyin_bench_common_args=(
  test/fsdp/bench_fsdp2_compare.py
  --copyin-bench
  --world-size "${WORLD_SIZE}"
  --device cuda
  --dtype "${DTYPE}"
  --warmup-steps "${COPYIN_WARMUP_STEPS}"
  --steps "${COPYIN_STEPS}"
)

param_materialization_bench_args=(
  test/fsdp/bench_fsdp2_compare.py
  --param-materialization-bench
  --world-size "${WORLD_SIZE}"
  --device cuda
  --dtype "${DTYPE}"
  --warmup-steps "${PARAM_MATERIALIZATION_WARMUP_STEPS}"
  --steps "${PARAM_MATERIALIZATION_STEPS}"
  --param-materialization-backend view_assign
  --param-materialization-backend copy_out
)

model_bench_args=(
  test/fsdp/bench_fsdp2_compare.py
  --world-size "${WORLD_SIZE}"
  --device cuda
  --model "${MODEL}"
  --unit "${UNIT}"
  --layers "${LAYERS}"
  --hidden "${HIDDEN}"
  --intermediate "${INTERMEDIATE}"
  --seq-len "${SEQ_LEN}"
  --heads "${HEADS}"
  --batch-size "${BATCH_SIZE}"
  --dtype "${DTYPE}"
  --warmup-steps "${MODEL_WARMUP_STEPS}"
  --steps "${MODEL_STEPS}"
  --activation-checkpoint
  --activation-checkpoint-wrapper
  --phase-timing
  --runtime-summary
)

memory_trace_common_args=(
  test/fsdp/bench_fsdp2_compare.py
  --world-size "${WORLD_SIZE}"
  --device cuda
  --model "${MODEL}"
  --unit "${UNIT}"
  --layers "${LAYERS}"
  --hidden "${HIDDEN}"
  --intermediate "${INTERMEDIATE}"
  --seq-len "${SEQ_LEN}"
  --heads "${HEADS}"
  --batch-size "${BATCH_SIZE}"
  --dtype "${DTYPE}"
  --warmup-steps "${MEMORY_WARMUP_STEPS}"
  --steps "${MEMORY_STEPS}"
  --activation-checkpoint
  --activation-checkpoint-wrapper
  --memory-trace
  --memory-by-rank
)

runtime_profile_args=(
  -m matrix_fsdp.tools.runtime_profile
  --world-size "${WORLD_SIZE}"
  --device cuda
  --model "${MODEL}"
  --unit "${UNIT}"
  --layers "${LAYERS}"
  --hidden "${HIDDEN}"
  --intermediate "${INTERMEDIATE}"
  --seq-len "${SEQ_LEN}"
  --heads "${HEADS}"
  --batch-size "${BATCH_SIZE}"
  --optimizer "${OPTIMIZER}"
  --dtype "${DTYPE}"
  --warmup-steps "${RUNTIME_PROFILE_WARMUP_STEPS}"
  --steps "${RUNTIME_PROFILE_STEPS}"
)

cat > "${REPORT}" <<EOF
# MatrixFSDP GPU Communication Report

- generated_at: ${TIMESTAMP}
- root_dir: ${ROOT_DIR}
- world_size: ${WORLD_SIZE}
- dtype: ${DTYPE}
- shard_sizes: ${SHARD_SIZES}
- model: ${MODEL}
- unit: ${UNIT}
- layers: ${LAYERS}
- hidden: ${HIDDEN}
- intermediate: ${INTERMEDIATE}
- seq_len: ${SEQ_LEN}
- heads: ${HEADS}
- batch_size: ${BATCH_SIZE}
- optimizer: ${OPTIMIZER}
- runtime_profile_steps: ${RUNTIME_PROFILE_STEPS}
- dry_run: ${DRY_RUN}
- run_comm_bench: ${RUN_COMM_BENCH}
- run_model_phase: ${RUN_MODEL_PHASE}
- run_memory_trace: ${RUN_MEMORY_TRACE}
- run_runtime_profile: ${RUN_RUNTIME_PROFILE}

EOF

cat > "${OUT_DIR}/torch_and_matrix_fsdp_info.py" <<'PY'
import torch
from matrix_fsdp.kernels.native import native_kernel_status
from matrix_fsdp.kernels.custom_collectives import default_custom_allgatherv_impl, resolve_custom_allgatherv_impl
from matrix_fsdp.layout import LayoutSegment

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("has_muon", hasattr(torch.optim, "Muon"))
print("default_custom_allgatherv_impl", default_custom_allgatherv_impl())
print("native_kernel_status", native_kernel_status())
rank_chunks = ((LayoutSegment(0, 2, 0),), (LayoutSegment(2, 5, 0),))
print("rank_chunk_auto_resolve", resolve_custom_allgatherv_impl(rank_chunks))
PY

run_logged "system_info" bash -lc 'hostname; date; nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits; echo; nvidia-smi topo -m'
run_logged "torch_and_matrix_fsdp_info" "${PYTHON_BIN}" "${OUT_DIR}/torch_and_matrix_fsdp_info.py"

if [[ "${RUN_COMM_BENCH}" == "1" ]]; then
  run_logged "comm_auto" torchrun_cmd "${comm_bench_args[@]}" --impl auto || true
  run_logged "comm_auto_sendrecv_optin" bash -lc "MATRIX_FSDP_AUTO_NATIVE_SENDRECV_CHUNKS=1 torchrun --standalone --nproc_per_node=${WORLD_SIZE} ${comm_bench_args[*]} --impl auto" || true
  run_logged "comm_native_group_broadcast" torchrun_cmd "${comm_bench_args[@]}" --impl native_group_broadcast || true
  run_logged "comm_native_sendrecv" torchrun_cmd "${comm_bench_args[@]}" --impl native_sendrecv || true
  run_logged "comm_padded_all_gather" torchrun_cmd "${comm_bench_args[@]}" --impl padded_all_gather || true
fi

if [[ "${RUN_COPYIN_BENCH}" == "1" ]]; then
  run_logged "copyin_flat" "${PYTHON_BIN}" "${copyin_bench_common_args[@]}" \
    --copyin-layout flat \
    --copyin-backend auto \
    --copyin-backend flat_cat \
    --copyin-backend foreach_copy \
    --copyin-backend segment_copy || true
  run_logged "copyin_matrix" "${PYTHON_BIN}" "${copyin_bench_common_args[@]}" \
    --copyin-layout matrix \
    --copyin-backend auto \
    --copyin-backend segment_copy || true
  run_logged "copyin_chunk_cat" "${PYTHON_BIN}" "${copyin_bench_common_args[@]}" \
    --copyin-layout chunk_cat \
    --copyin-backend auto \
    --copyin-backend chunk_cat \
    --copyin-backend segment_copy || true
fi

if [[ "${RUN_PARAM_MATERIALIZATION_BENCH}" == "1" ]]; then
  run_logged "param_materialization" "${PYTHON_BIN}" "${param_materialization_bench_args[@]}" || true
fi

HAS_MUON=1
if [[ "${DRY_RUN}" != "1" ]]; then
  if "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if hasattr(torch.optim, "Muon") else 1)'; then
    HAS_MUON=1
  else
    HAS_MUON=0
  fi
fi

if [[ "${HAS_MUON}" == "1" ]]; then
  if [[ "${RUN_MODEL_PHASE}" == "1" ]]; then
    run_logged "model_adamw_fsdp2_vs_matrix_default" "${PYTHON_BIN}" "${model_bench_args[@]}" --optimizer adamw --mode fsdp2 --mode matrix_default || true
    run_logged "model_matrix_custom_auto" "${PYTHON_BIN}" "${model_bench_args[@]}" --optimizer "${OPTIMIZER}" --mode matrix_owner_muon_role_greedy_custom_collective || true
    run_logged_env "model_matrix_custom_group_broadcast" "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=native_group_broadcast" "${PYTHON_BIN}" "${model_bench_args[@]}" --optimizer "${OPTIMIZER}" --mode matrix_owner_muon_role_greedy_custom_collective || true
    run_logged_env "model_matrix_custom_sendrecv" "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=native_sendrecv" "${PYTHON_BIN}" "${model_bench_args[@]}" --optimizer "${OPTIMIZER}" --mode matrix_owner_muon_role_greedy_custom_collective || true
    run_logged "model_matrix_all_gather" "${PYTHON_BIN}" "${model_bench_args[@]}" --optimizer "${OPTIMIZER}" --mode matrix_owner_muon_role_greedy_matrix_all_gather || true
    run_logged "model_fsdp2_muon" "${PYTHON_BIN}" "${model_bench_args[@]}" --optimizer "${OPTIMIZER}" --mode fsdp2 || true
  fi
  if [[ "${RUN_MEMORY_TRACE}" == "1" ]]; then
    run_logged "memory_adamw_fsdp2_vs_matrix_default" "${PYTHON_BIN}" "${memory_trace_common_args[@]}" --optimizer adamw --mode fsdp2 --mode matrix_default --memory-accounting || true
    run_logged "memory_muon_custom_auto" env \
      MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=auto \
      MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL=native_reduce \
      "${PYTHON_BIN}" "${memory_trace_common_args[@]}" \
        --optimizer muon \
        --mode fsdp2 \
        --mode matrix_owner_muon_role_greedy_custom_collective \
        --matrix-max-cached-elastic-workspaces-per-key 1 || true
  fi
  if [[ "${RUN_RUNTIME_PROFILE}" == "1" ]]; then
    run_logged "runtime_profile_matrix_custom_auto" "${PYTHON_BIN}" "${runtime_profile_args[@]}" --mode matrix_owner_muon_role_greedy_custom_collective || true
    run_logged_env "runtime_profile_matrix_custom_group_broadcast" "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=native_group_broadcast" "${PYTHON_BIN}" "${runtime_profile_args[@]}" --mode matrix_owner_muon_role_greedy_custom_collective || true
    run_logged_env "runtime_profile_matrix_custom_sendrecv" "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=native_sendrecv" "${PYTHON_BIN}" "${runtime_profile_args[@]}" --mode matrix_owner_muon_role_greedy_custom_collective || true
  fi
else
  write_section "model_benchmarks_skipped"
  echo "torch.optim.Muon is unavailable; Muon model benchmarks were skipped." >> "${REPORT}"
  if [[ "${RUN_MODEL_PHASE}" == "1" ]]; then
    run_logged "model_adamw_fsdp2_vs_matrix_default" "${PYTHON_BIN}" "${model_bench_args[@]}" --optimizer adamw --mode fsdp2 --mode matrix_default || true
  fi
  if [[ "${RUN_MEMORY_TRACE}" == "1" ]]; then
    run_logged "memory_adamw_fsdp2_vs_matrix_default" "${PYTHON_BIN}" "${memory_trace_common_args[@]}" --optimizer adamw --mode fsdp2 --mode matrix_default --memory-accounting || true
  fi
fi

append_quick_summary
"${PYTHON_BIN}" scripts/summarize_gpu_comm_report.py "${OUT_DIR}" --output "${OUT_DIR}/summary.txt"

echo
echo "Report written to ${REPORT}"
echo "Summary written to ${OUT_DIR}/summary.txt"
