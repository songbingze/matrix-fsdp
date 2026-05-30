# MoE Validation Commands

This checklist covers the current DeepSeek-style MoE path:

- upstream EP owns routed expert placement, routing, all-to-all, and expert execution;
- routed local experts are passed to MatrixFSDP through `ignored_params`;
- MatrixFSDP shards the remaining dense/router/shared/norm/MTP parameters;
- eFSDP / EP-local expert FSDP is not part of this default path.

Run commands from the repository root. Set `PYTHON=/path/to/python` if the active
shell does not already point at the desired environment.

## CPU Closeout

Focused DeepSeek-style EP smoke:

```bash
${PYTHON:-python} -m pytest test/fsdp/test_deepseek_v3_ep_train.py -q
```

Optimizer wrapper coverage for the Muon/AdamW helper used by the MoE recipe:

```bash
${PYTHON:-python} -m pytest \
  test/fsdp/test_optim_state.py::OptimizerStateTest::test_mixed_muon_adamw_optimizer_groups_params_from_shard_hints \
  test/fsdp/test_bench_fsdp2_compare.py::FSDP2CompareBenchTest::test_muon_optimizer_can_delay_state_allocation_until_step \
  test/fsdp/test_bench_fsdp2_compare.py::FSDP2CompareBenchTest::test_muon_optimizer_param_split_uses_adamw_for_non_2d_params \
  -q
```

Full CPU suite:

```bash
GLOO_SOCKET_IFNAME=lo0 ${PYTHON:-python} -m pytest test/fsdp -q
```

On Linux, use `GLOO_SOCKET_IFNAME=lo`. If Gloo fails before project code runs in
a sandboxed environment, rerun the same command outside that sandbox.

## Planner Sweep

Use this to check shard8, shard16, and shard32 owner balance for the dense
Muon-aware planner:

```bash
for ws in 8 16 32; do
  ${PYTHON:-python} -m matrix_fsdp.tools.planner_report \
    --model transformer_split_qkv \
    --layers 40 \
    --hidden 4096 \
    --intermediate 16384 \
    --heads 32 \
    --world-size "$ws" \
    --unit block \
    --policy muon_shard_aware \
    --selected-only
done
```

For the existing GPU batch wrapper, the equivalent override is:

```bash
PLANNER_WORLD_SIZES="8 16 32" \
LAYERS=40 HIDDEN=4096 INTERMEDIATE=16384 HEADS=32 \
scripts/run_gpu_test_batch.sh planner
```

## CUDA DeepSeek-Style EP Smoke

Large two-rank EP smoke with routed experts ignored by dense MatrixFSDP:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MATRIX_FSDP_RUN_DEEPSEEK_GPU_LARGE=1 \
${PYTHON:-python} -m pytest \
  test/fsdp/test_deepseek_v3_ep_train.py::DeepSeekV3ExpertParallelTrainTest::test_two_rank_cuda_large_v3_like_ep_dispatch_train_step_with_dense_fsdp \
  -q -s
```

## Seq8192 Compare Harness

AdamW, with activation checkpointing enabled:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MATRIX_FSDP_RUN_DEEPSEEK_GPU_COMPARE_4B=1 \
MATRIX_FSDP_COMPARE_SEQ_LEN=8192 \
MATRIX_FSDP_COMPARE_ACTIVATION_CHECKPOINT=1 \
MATRIX_FSDP_COMPARE_OPTIMIZER=adamw \
MATRIX_FSDP_COMPARE_OPTIMIZER_SCOPE=full_local \
MATRIX_FSDP_COMPARE_WARMUP_STEPS=1 \
MATRIX_FSDP_COMPARE_STEPS=3 \
MATRIX_FSDP_COMPARE_MEMORY_TRACE=1 \
${PYTHON:-python} -m pytest \
  test/fsdp/test_deepseek_v3_ep_train.py::DeepSeekV3ExpertParallelTrainTest::test_two_rank_cuda_4b_seq8192_matrix_vs_fsdp2 \
  -q -s
```

Muon, using the native send/recv custom gather path and dense optimizer scope:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MATRIX_FSDP_ENABLE_NATIVE_NCCL=1 \
MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=native_sendrecv \
MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL=uneven_reduce_scatter \
MATRIX_FSDP_RUN_DEEPSEEK_GPU_COMPARE_4B=1 \
MATRIX_FSDP_COMPARE_SEQ_LEN=8192 \
MATRIX_FSDP_COMPARE_ACTIVATION_CHECKPOINT=1 \
MATRIX_FSDP_COMPARE_OPTIMIZER=muon \
MATRIX_FSDP_COMPARE_OPTIMIZER_SCOPE=dense \
MATRIX_FSDP_COMPARE_MATRIX_BACKEND=custom \
MATRIX_FSDP_COMPARE_WARMUP_STEPS=1 \
MATRIX_FSDP_COMPARE_STEPS=3 \
MATRIX_FSDP_COMPARE_MEMORY_TRACE=1 \
${PYTHON:-python} -m pytest \
  test/fsdp/test_deepseek_v3_ep_train.py::DeepSeekV3ExpertParallelTrainTest::test_two_rank_cuda_4b_seq8192_matrix_vs_fsdp2 \
  -q -s
```

## Cross-Optimizer GPU Benchmark

This compares FSDP2 AdamW against the current best MatrixFSDP matrix-owner Muon
path. It is a performance and memory comparison, not an optimizer-correctness
comparison.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MATRIX_FSDP_ENABLE_NATIVE_NCCL=1 \
${PYTHON:-python} scripts/compare_adamw_fsdp2_vs_matrix_muon.py \
  --world-size 4 \
  --layers 32 \
  --hidden 4096 \
  --intermediate 16384 \
  --heads 32 \
  --seq-len 4096 \
  --batch-size 1 \
  --warmup-steps 2 \
  --steps 3 \
  --output-json /tmp/matrix_muon_vs_fsdp2_adamw.json
```

Record the git SHA, torch/CUDA/NCCL versions, visible GPUs, command, phase
timing table, peak memory, and any correctness skips with the result.
