#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
WORLD_SIZE="${WORLD_SIZE:-4}"
PLANNER_WORLD_SIZES="${PLANNER_WORLD_SIZES:-4 8}"
LARGE_WORLD_SIZES="${LARGE_WORLD_SIZES:-4 8}"
LAYERS="${LAYERS:-16}"
LARGE_LAYERS="${LARGE_LAYERS:-16 24 32}"
HIDDEN="${HIDDEN:-1024}"
INTERMEDIATE="${INTERMEDIATE:-4096}"
HEADS="${HEADS:-8}"
SEQ_LEN="${SEQ_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
PROFILE_STEPS="${PROFILE_STEPS:-3}"
STEPS="${STEPS:-8}"

BENCH_COMMON_ARGS=(
  --device cuda
  --world-size "$WORLD_SIZE"
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
  --profile-steps "$PROFILE_STEPS"
  --steps "$STEPS"
)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_gpu_test_batch.sh <batch>

Batches:
  cuda-smoke       Run selected 2-rank CUDA/NCCL correctness tests.
  cuda-full        Run the full distributed unittest class, including CUDA tests.
  planner          Print planner reports for shard4 and shard8 by default.
  compare-adamw    Compare FSDP2 and MatrixFSDP with AdamW.
  compare-adamw-vs-muon
                    Compare FSDP2 AdamW against the best MatrixFSDP Muon path.
  fully-shard-api  Compare public fully_shard APIs using plain torch optimizers.
  compare-muon     Compare FSDP2 and MatrixFSDP with matrix-owner Muon.
  runtime-profile  Capture MatrixFSDP event-level runtime memory traces.
  reserved-trace    Trace per-rank allocated/reserved memory on the large checkpointed transformer.
  checkpoint-large  Run large checkpoint-wrapper FSDP2/Matrix AdamW timing and correctness.
  copyin           Run CUDA grad bucket copy-in microbenchmarks.
  large            Run opt-in larger transformer benchmark sweeps.
  all              Run cuda-smoke, planner, compare-adamw, fully-shard-api, compare-muon, runtime-profile, reserved-trace, copyin.

Useful environment overrides:
  PYTHON=python
  WORLD_SIZE=4
  PLANNER_WORLD_SIZES="4 8"
  LARGE_WORLD_SIZES="4 8"
  LAYERS=16
  LARGE_LAYERS="16 24 32"
  LARGE_TRACE_LAYERS=32
  LARGE_SEQ_LEN=16384
  LARGE_TRACE_WARMUP_STEPS=1
  LARGE_TRACE_STEPS=2
  ADAMW_VS_MUON_LAYERS=32
  ADAMW_VS_MUON_SEQ_LEN=4096
  ADAMW_VS_MUON_WARMUP_STEPS=2
  ADAMW_VS_MUON_STEPS=3
  HIDDEN=1024
  INTERMEDIATE=4096
  HEADS=8
  SEQ_LEN=4096
  BATCH_SIZE=1
  DTYPE=bfloat16
  WARMUP_STEPS=3
  PROFILE_STEPS=3
  STEPS=8

Run this from an environment with CUDA/NCCL, for example the remote dreamer
conda environment. Set CUDA_VISIBLE_DEVICES outside this script if needed.
EOF
}

run_cmd() {
  printf "\n+"
  printf " %q" "$@"
  printf "\n"
  "$@"
}

run_cuda_smoke() {
  run_cmd "$PYTHON_BIN" -m unittest \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_step_matches_eager_model \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_reshard_after_forward_step_matches_eager_model \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_finalize_after_backward_step_matches_eager_model \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_mixed_precision_step_matches_eager_model \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_grad_shards_match_averaged_eager_grads \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_parameter_boundary_grad_shards_match_averaged_eager_grads \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_ordered_group_grad_shards_match_averaged_eager_grads \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_multi_unit_forward_prefetch_step_matches_eager_model \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_multi_unit_backward_prefetch_step_matches_eager_model \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_wrap_policy_forward_prefetch_step_matches_eager_model \
    test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest.test_two_rank_cuda_wrap_policy_backward_prefetch_step_matches_eager_model
}

run_cuda_full() {
  run_cmd "$PYTHON_BIN" -m unittest test.fsdp.test_matrix_fsdp_distributed.MatrixFSDPDistributedTest
}

run_planner() {
  local ws
  for ws in $PLANNER_WORLD_SIZES; do
    run_cmd "$PYTHON_BIN" -m matrix_fsdp.tools.planner_report \
      --model transformer_split_qkv \
      --layers "$LAYERS" \
      --hidden "$HIDDEN" \
      --intermediate "$INTERMEDIATE" \
      --heads "$HEADS" \
      --world-size "$ws" \
      --unit block \
      --policy muon_shard_aware \
      --selected-only
  done
}

run_compare_adamw() {
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${BENCH_COMMON_ARGS[@]}" \
    --optimizer adamw \
    --mode fsdp2 \
    --mode matrix_default \
    --mode matrix_prefetch_late_backward_fsdp2_chunk
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${BENCH_COMMON_ARGS[@]}" \
    --optimizer adamw \
    --mode fsdp2 \
    --mode matrix_default \
    --correctness \
    --candidate-mode matrix_default
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${BENCH_COMMON_ARGS[@]}" \
    --optimizer adamw \
    --mode fsdp2 \
    --mode matrix_prefetch_late_backward_fsdp2_chunk \
    --correctness \
    --candidate-mode matrix_prefetch_late_backward_fsdp2_chunk
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${BENCH_COMMON_ARGS[@]}" \
    --optimizer adamw \
    --mode fsdp2 \
    --mode matrix_default \
    --mode matrix_prefetch_late_backward_fsdp2_chunk \
    --memory-trace
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${BENCH_COMMON_ARGS[@]}" \
    --optimizer adamw \
    --mode fsdp2 \
    --mode matrix_default \
    --mode matrix_prefetch_late_backward_fsdp2_chunk \
    --phase-timing
}

run_fully_shard_api() {
  run_cmd "$PYTHON_BIN" scripts/compare_fully_shard_api.py \
    --device cuda \
    --world-size "$WORLD_SIZE" \
    --model transformer_split_qkv \
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
    --phase-timing
  run_cmd "$PYTHON_BIN" scripts/compare_fully_shard_api.py \
    --device cuda \
    --world-size "$WORLD_SIZE" \
    --model transformer_split_qkv \
    --layers "${CORRECTNESS_LAYERS:-4}" \
    --hidden "${CORRECTNESS_HIDDEN:-1024}" \
    --intermediate "${CORRECTNESS_INTERMEDIATE:-4096}" \
    --seq-len "${CORRECTNESS_SEQ_LEN:-512}" \
    --heads "${CORRECTNESS_HEADS:-8}" \
    --batch-size "$BATCH_SIZE" \
    --optimizer adamw \
    --dtype "$DTYPE" \
    --warmup-steps 1 \
    --steps 1 \
    --correctness
}

run_compare_muon() {
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${BENCH_COMMON_ARGS[@]}" \
    --optimizer muon \
    --mode fsdp2 \
    --mode matrix_owner_muon \
    --mode matrix_owner_muon_role_greedy
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${BENCH_COMMON_ARGS[@]}" \
    --optimizer muon \
    --mode fsdp2 \
    --mode matrix_owner_muon \
    --correctness \
    --candidate-mode matrix_owner_muon
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${BENCH_COMMON_ARGS[@]}" \
    --optimizer muon \
    --mode fsdp2 \
    --mode matrix_owner_muon_role_greedy \
    --correctness \
    --candidate-mode matrix_owner_muon_role_greedy
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${BENCH_COMMON_ARGS[@]}" \
    --optimizer muon \
    --mode fsdp2 \
    --mode matrix_owner_muon \
    --mode matrix_owner_muon_role_greedy \
    --memory-trace
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py "${BENCH_COMMON_ARGS[@]}" \
    --optimizer muon \
    --mode fsdp2 \
    --mode matrix_owner_muon \
    --mode matrix_owner_muon_role_greedy \
    --mode matrix_owner_muon_role_greedy_custom_collective \
    --phase-timing
}

run_compare_adamw_vs_muon() {
  local layers="${ADAMW_VS_MUON_LAYERS:-32}"
  local hidden="${ADAMW_VS_MUON_HIDDEN:-4096}"
  local intermediate="${ADAMW_VS_MUON_INTERMEDIATE:-16384}"
  local seq_len="${ADAMW_VS_MUON_SEQ_LEN:-4096}"
  local heads="${ADAMW_VS_MUON_HEADS:-32}"
  local warmup_steps="${ADAMW_VS_MUON_WARMUP_STEPS:-2}"
  local steps="${ADAMW_VS_MUON_STEPS:-3}"
  local custom_gather="${MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL:-native_sendrecv}"
  local custom_reduce="${MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL:-native_reduce}"
  run_cmd env \
    "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=$custom_gather" \
    "MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL=$custom_reduce" \
    "$PYTHON_BIN" scripts/compare_adamw_fsdp2_vs_matrix_muon.py \
      --device cuda \
      --world-size "$WORLD_SIZE" \
      --layers "$layers" \
      --hidden "$hidden" \
      --intermediate "$intermediate" \
      --seq-len "$seq_len" \
      --heads "$heads" \
      --batch-size "$BATCH_SIZE" \
      --dtype "$DTYPE" \
      --warmup-steps "$warmup_steps" \
      --steps "$steps" \
      --custom-allgatherv-impl "$custom_gather" \
      --custom-reduce-scatterv-impl "$custom_reduce" \
      --json-config
}

run_runtime_profile() {
  run_cmd "$PYTHON_BIN" -m matrix_fsdp.tools.runtime_profile \
    --device cuda \
    --world-size "$WORLD_SIZE" \
    --model transformer_split_qkv \
    --unit block \
    --layers "$LAYERS" \
    --hidden "$HIDDEN" \
    --intermediate "$INTERMEDIATE" \
    --seq-len "$SEQ_LEN" \
    --heads "$HEADS" \
    --batch-size "$BATCH_SIZE" \
    --optimizer adamw \
    --dtype "$DTYPE" \
    --warmup-steps "$WARMUP_STEPS" \
    --profile-steps 0 \
    --steps "$STEPS" \
    --mode prefetch_bucket_reduce_scatter
  run_cmd "$PYTHON_BIN" -m matrix_fsdp.tools.runtime_profile \
    --device cuda \
    --world-size "$WORLD_SIZE" \
    --model transformer_split_qkv \
    --unit block \
    --layers "$LAYERS" \
    --hidden "$HIDDEN" \
    --intermediate "$INTERMEDIATE" \
    --seq-len "$SEQ_LEN" \
    --heads "$HEADS" \
    --batch-size "$BATCH_SIZE" \
    --optimizer adamw \
    --dtype "$DTYPE" \
    --warmup-steps "$WARMUP_STEPS" \
    --profile-steps 0 \
    --steps "$STEPS" \
    --mode prefetch_bucket_reduce_scatter \
    --max-active-full-param-buffers 1
  run_cmd "$PYTHON_BIN" -m matrix_fsdp.tools.runtime_profile \
    --device cuda \
    --world-size "$WORLD_SIZE" \
    --model transformer_split_qkv \
    --unit block \
    --layers "$LAYERS" \
    --hidden "$HIDDEN" \
    --intermediate "$INTERMEDIATE" \
    --seq-len "$SEQ_LEN" \
    --heads "$HEADS" \
    --batch-size "$BATCH_SIZE" \
    --optimizer muon \
    --dtype "$DTYPE" \
    --warmup-steps "$WARMUP_STEPS" \
    --profile-steps 0 \
    --steps "$STEPS" \
    --mode matrix_owner_muon_role_greedy
}

run_reserved_trace() {
  local large_seq_len="${LARGE_SEQ_LEN:-16384}"
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py \
    --memory-trace \
    --memory-by-rank \
    --activation-checkpoint \
    --activation-checkpoint-wrapper \
    --device cuda \
    --world-size "$WORLD_SIZE" \
    --model transformer_split_qkv \
    --unit block \
    --layers "${LARGE_TRACE_LAYERS:-32}" \
    --hidden "$HIDDEN" \
    --intermediate "$INTERMEDIATE" \
    --seq-len "$large_seq_len" \
    --heads "$HEADS" \
    --batch-size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --warmup-steps "${LARGE_TRACE_WARMUP_STEPS:-1}" \
    --steps "${LARGE_TRACE_STEPS:-1}" \
    --optimizer adamw \
    --mode matrix_default
}

run_checkpoint_large() {
  local large_seq_len="${LARGE_SEQ_LEN:-16384}"
  local large_steps="${LARGE_TRACE_STEPS:-2}"
  local large_warmup="${LARGE_TRACE_WARMUP_STEPS:-1}"
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py \
    --phase-timing \
    --activation-checkpoint \
    --activation-checkpoint-wrapper \
    --device cuda \
    --world-size "$WORLD_SIZE" \
    --model transformer_split_qkv \
    --unit block \
    --layers "${LARGE_TRACE_LAYERS:-32}" \
    --hidden "$HIDDEN" \
    --intermediate "$INTERMEDIATE" \
    --seq-len "$large_seq_len" \
    --heads "$HEADS" \
    --batch-size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --warmup-steps "$large_warmup" \
    --steps "$large_steps" \
    --optimizer adamw \
    --mode fsdp2 \
    --mode matrix_default
  run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py \
    --correctness \
    --activation-checkpoint \
    --activation-checkpoint-wrapper \
    --device cuda \
    --world-size "$WORLD_SIZE" \
    --model transformer_split_qkv \
    --unit block \
    --layers "${CORRECTNESS_LAYERS:-4}" \
    --hidden "${CORRECTNESS_HIDDEN:-1024}" \
    --intermediate "${CORRECTNESS_INTERMEDIATE:-4096}" \
    --seq-len "${CORRECTNESS_SEQ_LEN:-512}" \
    --heads "${CORRECTNESS_HEADS:-8}" \
    --batch-size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --warmup-steps 1 \
    --steps 1 \
    --optimizer adamw \
    --mode fsdp2 \
    --mode matrix_default \
    --candidate-mode matrix_default
}

run_copyin() {
  local layout
  for layout in flat matrix chunk_cat; do
    run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py \
      --copyin-bench \
      --device cuda \
      --world-size "$WORLD_SIZE" \
      --dtype "$DTYPE" \
      --steps "$STEPS" \
      --warmup-steps "$WARMUP_STEPS" \
      --copyin-layout "$layout" \
      --copyin-backend auto \
      --copyin-backend segment_copy \
      --copyin-backend foreach_copy \
      --copyin-backend chunk_cat
  done
}

run_large() {
  local ws
  local layers
  for ws in $LARGE_WORLD_SIZES; do
    for layers in $LARGE_LAYERS; do
      run_cmd "$PYTHON_BIN" test/fsdp/bench_fsdp2_compare.py \
        --device cuda \
        --world-size "$ws" \
        --model transformer_split_qkv \
        --unit block \
        --layers "$layers" \
        --hidden "$HIDDEN" \
        --intermediate "$INTERMEDIATE" \
        --seq-len "$SEQ_LEN" \
        --heads "$HEADS" \
        --batch-size "$BATCH_SIZE" \
        --dtype "$DTYPE" \
        --optimizer adamw \
        --mode fsdp2 \
        --mode matrix_default \
        --warmup-steps "$WARMUP_STEPS" \
        --profile-steps "$PROFILE_STEPS" \
        --steps "$STEPS"
    done
  done
}

batch="${1:-cuda-smoke}"

case "$batch" in
  -h|--help|help)
    usage
    ;;
  cuda-smoke)
    run_cuda_smoke
    ;;
  cuda-full)
    run_cuda_full
    ;;
  planner)
    run_planner
    ;;
  compare-adamw)
    run_compare_adamw
    ;;
  compare-adamw-vs-muon)
    run_compare_adamw_vs_muon
    ;;
  fully-shard-api)
    run_fully_shard_api
    ;;
  compare-muon)
    run_compare_muon
    ;;
  runtime-profile)
    run_runtime_profile
    ;;
  reserved-trace)
    run_reserved_trace
    ;;
  checkpoint-large)
    run_checkpoint_large
    ;;
  copyin)
    run_copyin
    ;;
  large)
    run_large
    ;;
  all)
    run_cuda_smoke
    run_planner
    run_compare_adamw
    run_fully_shard_api
    run_compare_muon
    run_runtime_profile
    run_reserved_trace
    run_copyin
    ;;
  *)
    usage
    echo
    echo "Unknown batch: $batch" >&2
    exit 2
    ;;
esac
