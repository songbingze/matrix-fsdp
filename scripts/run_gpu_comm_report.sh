#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/benchmark_results/gpu_comm_${TIMESTAMP}}"
REPORT="${OUT_DIR}/report.md"
mkdir -p "${OUT_DIR}"

WORLD_SIZE="${WORLD_SIZE:-}"
if [[ -z "${WORLD_SIZE}" ]]; then
  WORLD_SIZE="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
fi

DTYPE="${DTYPE:-bfloat16}"
COMM_WARMUP_STEPS="${COMM_WARMUP_STEPS:-3}"
COMM_STEPS="${COMM_STEPS:-8}"
MODEL_WARMUP_STEPS="${MODEL_WARMUP_STEPS:-1}"
MODEL_STEPS="${MODEL_STEPS:-3}"
RUNTIME_PROFILE_WARMUP_STEPS="${RUNTIME_PROFILE_WARMUP_STEPS:-0}"
RUNTIME_PROFILE_STEPS="${RUNTIME_PROFILE_STEPS:-1}"
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
  SHARD_SIZES="$(WORLD_SIZE="${WORLD_SIZE}" python - <<'PY'
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
  --optimizer "${OPTIMIZER}"
  --dtype "${DTYPE}"
  --warmup-steps "${MODEL_WARMUP_STEPS}"
  --steps "${MODEL_STEPS}"
  --activation-checkpoint
  --activation-checkpoint-wrapper
  --phase-timing
  --runtime-summary
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

EOF

run_logged "system_info" bash -lc 'hostname; date; nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits; echo; nvidia-smi topo -m'
run_logged "torch_and_matrix_fsdp_info" python - <<'PY'
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

run_logged "comm_auto" torchrun_cmd "${comm_bench_args[@]}" --impl auto || true
run_logged "comm_auto_sendrecv_optin" bash -lc "MATRIX_FSDP_AUTO_NATIVE_SENDRECV_CHUNKS=1 torchrun --standalone --nproc_per_node=${WORLD_SIZE} ${comm_bench_args[*]} --impl auto" || true
run_logged "comm_native_group_broadcast" torchrun_cmd "${comm_bench_args[@]}" --impl native_group_broadcast || true
run_logged "comm_native_sendrecv" torchrun_cmd "${comm_bench_args[@]}" --impl native_sendrecv || true
run_logged "comm_padded_all_gather" torchrun_cmd "${comm_bench_args[@]}" --impl padded_all_gather || true

if python - <<'PY'
import torch
raise SystemExit(0 if hasattr(torch.optim, "Muon") else 1)
PY
then
  run_logged "model_matrix_custom_auto" python "${model_bench_args[@]}" --mode matrix_owner_muon_role_greedy_custom_collective || true
  run_logged_env "model_matrix_custom_group_broadcast" "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=native_group_broadcast" python "${model_bench_args[@]}" --mode matrix_owner_muon_role_greedy_custom_collective || true
  run_logged_env "model_matrix_custom_sendrecv" "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=native_sendrecv" python "${model_bench_args[@]}" --mode matrix_owner_muon_role_greedy_custom_collective || true
  run_logged "model_matrix_all_gather" python "${model_bench_args[@]}" --mode matrix_owner_muon_role_greedy_matrix_all_gather || true
  run_logged "model_fsdp2" python "${model_bench_args[@]}" --mode fsdp2 || true
  run_logged "runtime_profile_matrix_custom_auto" python "${runtime_profile_args[@]}" --mode matrix_owner_muon_role_greedy_custom_collective || true
  run_logged_env "runtime_profile_matrix_custom_group_broadcast" "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=native_group_broadcast" python "${runtime_profile_args[@]}" --mode matrix_owner_muon_role_greedy_custom_collective || true
  run_logged_env "runtime_profile_matrix_custom_sendrecv" "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=native_sendrecv" python "${runtime_profile_args[@]}" --mode matrix_owner_muon_role_greedy_custom_collective || true
else
  write_section "model_benchmarks_skipped"
  echo "torch.optim.Muon is unavailable; Muon model benchmarks were skipped." >> "${REPORT}"
fi

append_quick_summary

echo
echo "Report written to ${REPORT}"
