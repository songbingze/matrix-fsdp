# MatrixFSDP

MatrixFSDP is an experimental FSDP2-style runtime for training large PyTorch
models with matrix-aware sharding. It keeps the public API close to PyTorch
FSDP2: choose a DP or HSDP `DeviceMesh`, call `fully_shard(...)`, then train
with the normal PyTorch loop.

The main difference is the layout policy. MatrixFSDP can keep selected matrices
whole on owner ranks for Muon-style optimizers, split dense weights by rows or
blocks, and leave TP/EP-owned parameters to the upper-level parallelism stack.
This makes the same runtime usable for AdamW baselines, matrix-owner Muon runs,
and MoE models where FSDP should manage only the DP/HSDP parameter set.

```python
model = fully_shard(model, mesh=mesh, dp_mesh_dims=dp_mesh_dims)
optim = configure_optimizer(model, "adamw", lr=3e-4)

loss.backward()
optim.step()
optim.zero_grad(set_to_none=True)
```

Create optimizers after sharding so they see the sharded parameter views. For
most users, `fully_shard(...)` is the only sharding entry point.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Main APIs

- `fully_shard(model, mesh=..., dp_mesh_dims=...)`: public FSDP2-like entry
  point.
- `DataParallelMeshDims(shard="dp_shard")`: selects the `DeviceMesh` dimension
  used for parameter sharding.
- `DataParallelMeshDims(shard="dp_shard", replicate="dp_replicate")`: enables
  HSDP-style replicate x shard layouts.
- `optimizer_policy="mixed_muon_adamw"`: enables Muon-aware matrix-owner
  planning for `fully_shard(...)`.
- `MatrixFSDPOptimizer` and `configure_optimizer(...)`: optimizer lifecycle
  helpers for AdamW, Muon, and mixed Muon/AdamW training.
- `save_matrix_dcp(...)` and `load_matrix_dcp(...)`: DCP checkpoint helpers for
  MatrixFSDP shard metadata and optimizer state.

## Documentation

- [Usage Guide](docs/usage.md): dense AdamW, dense Muon/HSDP, block-level
  sharding, and DeepSeek-style MoE with EP-owned routed experts.
- [Training Integrations](docs/training_integrations.md): activation
  checkpointing, DCP save/load, full-state debug checkpoints, and resharded
  load.
- [MoE Validation Commands](docs/moe_validation_commands.md): CPU closeout,
  shard8/16/32 planner sweeps, CUDA EP smoke, seq8192 compare, and
  cross-optimizer GPU benchmark commands.
