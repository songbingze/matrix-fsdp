#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON:-python}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/benchmark_results/gpu_preflight_$TIMESTAMP}"
COMMAND_FILE="$OUT_DIR/commands.sh"
SUMMARY_FILE="$OUT_DIR/preflight_summary.txt"

WORLD_SIZE="${WORLD_SIZE:-8}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
LAYERS="${LAYERS:-32}"
HIDDEN="${HIDDEN:-4096}"
INTERMEDIATE="${INTERMEDIATE:-16384}"
SEQ_LEN="${SEQ_LEN:-4096}"
LONG_SEQ_LEN="${LONG_SEQ_LEN:-8192}"
HEADS="${HEADS:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
STEPS="${STEPS:-3}"
CORRECTNESS_LAYERS="${CORRECTNESS_LAYERS:-4}"
CORRECTNESS_HIDDEN="${CORRECTNESS_HIDDEN:-1024}"
CORRECTNESS_INTERMEDIATE="${CORRECTNESS_INTERMEDIATE:-4096}"
CORRECTNESS_SEQ_LEN="${CORRECTNESS_SEQ_LEN:-512}"
CORRECTNESS_HEADS="${CORRECTNESS_HEADS:-8}"
RUN_LOCAL_CHECKS="${RUN_LOCAL_CHECKS:-1}"

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_gpu_benchmark.sh

This prepares the GPU benchmark runbook without requiring a GPU. It writes:
  benchmark_results/gpu_preflight_<timestamp>/commands.sh
  benchmark_results/gpu_preflight_<timestamp>/preflight_summary.txt

Common overrides:
  WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  LAYERS=32 HIDDEN=4096 INTERMEDIATE=16384 SEQ_LEN=4096 LONG_SEQ_LEN=8192 HEADS=32
  WARMUP_STEPS=2 STEPS=3
  RUN_LOCAL_CHECKS=0

The generated commands run:
  1. AdamW FSDP2 vs MatrixFSDP default
  2. Muon FSDP2 vs MatrixFSDP owner layouts
  3. correctness checks on a smaller shape
  4. memory/rank trace
  5. runtime event profile
  6. communication microbenchmarks
  7. summary generation
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

mkdir -p "$OUT_DIR"

run_local_check() {
  local name="$1"
  shift
  {
    echo
    echo "== $name =="
    printf '+'
    printf ' %q' "$@"
    echo
  } | tee -a "$SUMMARY_FILE"
  "$@" 2>&1 | tee -a "$SUMMARY_FILE"
}

cat > "$SUMMARY_FILE" <<EOF
MatrixFSDP GPU benchmark preflight
generated_at=$TIMESTAMP
repo_root=$REPO_ROOT
out_dir=$OUT_DIR
world_size=$WORLD_SIZE
cuda_visible_devices=$CUDA_VISIBLE_DEVICES_VALUE
layers=$LAYERS
hidden=$HIDDEN
intermediate=$INTERMEDIATE
seq_len=$SEQ_LEN
long_seq_len=$LONG_SEQ_LEN
heads=$HEADS
batch_size=$BATCH_SIZE
dtype=$DTYPE
warmup_steps=$WARMUP_STEPS
steps=$STEPS

Default benchmark policy:
- AdamW: fsdp2 vs matrix_default
- Muon: fsdp2 vs role_greedy, scope_greedy, and cost_aware custom collectives
- activation checkpoint: torch checkpoint_wrapper path enabled
- custom gather: MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=auto
- custom reduce: MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL=native_reduce

EOF

if [[ "$RUN_LOCAL_CHECKS" == "1" ]]; then
  run_local_check "python_compile" "$PYTHON_BIN" -m py_compile \
    scripts/summarize_gpu_comm_report.py \
    test/fsdp/bench_fsdp2_compare.py \
    matrix_fsdp/runtime/fsdp_unit.py \
    matrix_fsdp/runtime/summary.py
  run_local_check "cpu_profile_summary_tests" "$PYTHON_BIN" -m pytest \
    test/fsdp/test_summary.py \
    test/fsdp/test_runtime_profile_tool.py \
    test/fsdp/test_gpu_report_summary.py \
    -q
  run_local_check "diff_check" git diff --check
else
  echo "local checks skipped: RUN_LOCAL_CHECKS=0" | tee -a "$SUMMARY_FILE"
fi

cat > "$COMMAND_FILE" <<EOF
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="\${ROOT_DIR:-\$(pwd)}"
cd "\$ROOT_DIR"
export PYTHONPATH="\$ROOT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES_VALUE}"
export WORLD_SIZE="\${WORLD_SIZE:-$WORLD_SIZE}"
export MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL="\${MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL:-auto}"
export MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL="\${MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL:-native_reduce}"
export MATRIX_WORKSPACE_CACHE_PER_KEY="\${MATRIX_WORKSPACE_CACHE_PER_KEY:-1}"

PYTHON_BIN="\${PYTHON:-python}"
TIMESTAMP="\$(date +%Y%m%d_%H%M%S)"
RUN_DIR="\${RUN_DIR:-\$ROOT_DIR/benchmark_results/gpu_suite_\$TIMESTAMP}"
mkdir -p "\$RUN_DIR"

run_logged() {
  local name="\$1"
  shift
  local log_file="\$RUN_DIR/\$name.log"
  printf '\\n== %s ==\\n' "\$name" | tee -a "\$RUN_DIR/run.log"
  printf '+ ' | tee -a "\$RUN_DIR/run.log"
  printf '%q ' "\$@" | tee -a "\$RUN_DIR/run.log"
  printf '\\n' | tee -a "\$RUN_DIR/run.log"
  "\$@" 2>&1 | tee "\$log_file"
}

COMMON_ARGS=(
  test/fsdp/bench_fsdp2_compare.py
  --device cuda
  --world-size "\$WORLD_SIZE"
  --model transformer_split_qkv
  --unit block
  --layers "$LAYERS"
  --hidden "$HIDDEN"
  --intermediate "$INTERMEDIATE"
  --seq-len "$SEQ_LEN"
  --heads "$HEADS"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --warmup-steps "$WARMUP_STEPS"
  --steps "$STEPS"
  --activation-checkpoint
  --activation-checkpoint-wrapper
)

LONG_ARGS=(
  test/fsdp/bench_fsdp2_compare.py
  --device cuda
  --world-size "\$WORLD_SIZE"
  --model transformer_split_qkv
  --unit block
  --layers "$LAYERS"
  --hidden "$HIDDEN"
  --intermediate "$INTERMEDIATE"
  --seq-len "$LONG_SEQ_LEN"
  --heads "$HEADS"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --warmup-steps "$WARMUP_STEPS"
  --steps "$STEPS"
  --activation-checkpoint
  --activation-checkpoint-wrapper
)

CORRECTNESS_ARGS=(
  test/fsdp/bench_fsdp2_compare.py
  --device cuda
  --world-size "\$WORLD_SIZE"
  --model transformer_split_qkv
  --unit block
  --layers "$CORRECTNESS_LAYERS"
  --hidden "$CORRECTNESS_HIDDEN"
  --intermediate "$CORRECTNESS_INTERMEDIATE"
  --seq-len "$CORRECTNESS_SEQ_LEN"
  --heads "$CORRECTNESS_HEADS"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --warmup-steps 1
  --steps 1
  --activation-checkpoint
  --activation-checkpoint-wrapper
)

run_logged system_info bash -lc 'hostname; date; nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits; echo; nvidia-smi topo -m || true'
run_logged torch_matrix_info "\$PYTHON_BIN" - <<'PY'
import torch
from matrix_fsdp.kernels.custom_collectives import custom_reduce_scatterv_impl, default_custom_allgatherv_impl
from matrix_fsdp.kernels.native import native_kernel_status

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("has_muon", hasattr(torch.optim, "Muon"))
print("default_custom_allgatherv_impl", default_custom_allgatherv_impl())
print("custom_reduce_scatterv_impl", custom_reduce_scatterv_impl())
print("native_kernel_status", native_kernel_status())
PY

run_logged model_adamw_phase "\$PYTHON_BIN" "\${COMMON_ARGS[@]}" \\
  --optimizer adamw \\
  --mode fsdp2 \\
  --mode matrix_default \\
  --phase-timing \\
  --runtime-summary

run_logged memory_adamw "\$PYTHON_BIN" "\${COMMON_ARGS[@]}" \\
  --optimizer adamw \\
  --mode fsdp2 \\
  --mode matrix_default \\
  --memory-trace \\
  --memory-by-rank \\
  --memory-accounting

run_logged adamw_correctness "\$PYTHON_BIN" "\${CORRECTNESS_ARGS[@]}" \\
  --optimizer adamw \\
  --mode fsdp2 \\
  --mode matrix_default \\
  --correctness \\
  --candidate-mode matrix_default

run_logged model_muon_phase "\$PYTHON_BIN" "\${COMMON_ARGS[@]}" \\
  --optimizer muon \\
  --mode fsdp2 \\
  --mode matrix_owner_muon_role_greedy_custom_collective \\
  --mode matrix_owner_muon_scope_greedy_custom_collective \\
  --mode matrix_owner_muon_cost_aware_custom_collective \\
  --matrix-max-cached-elastic-workspaces-per-key "\$MATRIX_WORKSPACE_CACHE_PER_KEY" \\
  --phase-timing \\
  --runtime-summary

run_logged memory_muon "\$PYTHON_BIN" "\${COMMON_ARGS[@]}" \\
  --optimizer muon \\
  --mode fsdp2 \\
  --mode matrix_owner_muon_role_greedy_custom_collective \\
  --mode matrix_owner_muon_scope_greedy_custom_collective \\
  --mode matrix_owner_muon_cost_aware_custom_collective \\
  --matrix-max-cached-elastic-workspaces-per-key "\$MATRIX_WORKSPACE_CACHE_PER_KEY" \\
  --memory-trace \\
  --memory-by-rank \\
  --memory-accounting

run_logged muon_correctness "\$PYTHON_BIN" "\${CORRECTNESS_ARGS[@]}" \\
  --optimizer muon \\
  --mode fsdp2 \\
  --mode matrix_owner_muon_cost_aware_custom_collective \\
  --matrix-max-cached-elastic-workspaces-per-key "\$MATRIX_WORKSPACE_CACHE_PER_KEY" \\
  --correctness \\
  --candidate-mode matrix_owner_muon_cost_aware_custom_collective

run_logged runtime_profile_muon "\$PYTHON_BIN" -m matrix_fsdp.tools.runtime_profile \\
  --device cuda \\
  --world-size "\$WORLD_SIZE" \\
  --model transformer_split_qkv \\
  --unit block \\
  --layers "$LAYERS" \\
  --hidden "$HIDDEN" \\
  --intermediate "$INTERMEDIATE" \\
  --seq-len "$SEQ_LEN" \\
  --heads "$HEADS" \\
  --batch-size "$BATCH_SIZE" \\
  --optimizer muon \\
  --dtype "$DTYPE" \\
  --warmup-steps 0 \\
  --steps 1 \\
  --mode matrix_owner_muon_cost_aware_custom_collective

run_logged model_long_seq_adamw_phase "\$PYTHON_BIN" "\${LONG_ARGS[@]}" \\
  --optimizer adamw \\
  --mode fsdp2 \\
  --mode matrix_default \\
  --phase-timing

run_logged model_long_seq_muon_phase "\$PYTHON_BIN" "\${LONG_ARGS[@]}" \\
  --optimizer muon \\
  --mode fsdp2 \\
  --mode matrix_owner_muon_cost_aware_custom_collective \\
  --matrix-max-cached-elastic-workspaces-per-key "\$MATRIX_WORKSPACE_CACHE_PER_KEY" \\
  --phase-timing

RUN_COMM_BENCH=1 \\
RUN_MODEL_PHASE=0 \\
RUN_MEMORY_TRACE=0 \\
RUN_RUNTIME_PROFILE=0 \\
RUN_COPYIN_BENCH=1 \\
RUN_PARAM_MATERIALIZATION_BENCH=1 \\
WORLD_SIZE="\$WORLD_SIZE" \\
CUDA_VISIBLE_DEVICES="\$CUDA_VISIBLE_DEVICES" \\
LAYERS="$LAYERS" \\
HIDDEN="$HIDDEN" \\
INTERMEDIATE="$INTERMEDIATE" \\
SEQ_LEN="$SEQ_LEN" \\
HEADS="$HEADS" \\
DTYPE="$DTYPE" \\
OUT_DIR="\$RUN_DIR/comm_report" \\
scripts/run_gpu_comm_report.sh

"\$PYTHON_BIN" scripts/summarize_gpu_comm_report.py "\$RUN_DIR" --output "\$RUN_DIR/summary.txt"
"\$PYTHON_BIN" scripts/summarize_gpu_comm_report.py "\$RUN_DIR/comm_report" --output "\$RUN_DIR/comm_report_summary.txt"

echo
echo "GPU benchmark suite written to \$RUN_DIR"
echo "Main summary: \$RUN_DIR/summary.txt"
echo "Comm summary: \$RUN_DIR/comm_report_summary.txt"
EOF

chmod +x "$COMMAND_FILE"

{
  echo
  echo "Generated GPU command file:"
  echo "$COMMAND_FILE"
  echo
  echo "Run it on the GPU machine with:"
  echo "  bash $COMMAND_FILE"
} | tee -a "$SUMMARY_FILE"

echo
echo "Preflight summary: $SUMMARY_FILE"
echo "GPU command file: $COMMAND_FILE"
