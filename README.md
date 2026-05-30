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

## Performance Highlights

Representative 4x A100-PCIE-40GB measurements with PyTorch 2.10, CUDA 12.8,
BF16 parameters/compute, and transformer-block sharding. The large transformer
rows use activation checkpointing:

| Case | FSDP2 | MatrixFSDP | Result |
| --- | ---: | ---: | --- |
| AdamW, 8L hidden2048 seq1024 | 175.095 ms / 2653.3 MB | 171.491 ms / 2328.7 MB | 2.1% faster, 12.2% lower peak memory |
| AdamW, 32L hidden4096 seq16384 | 8844.463 ms / 17267.4 MB | 8549.254 ms / 17267.4 MB | 3.3% lower step time at equal peak memory |
| Muon, 32L hidden4096 seq4096 | 10432.683 ms / 11057.9 MB | 4223.246 ms / 11443.0 MB | 2.47x faster total step |
| Muon, 40L hidden5120 seq4096 | 20073.158 ms / 20640.7 MB | 8404.598 ms / 21240.3 MB | 2.39x faster total step |

The Muon rows use the same model shape, BF16 compute, block boundaries, and
activation checkpointing on both sides. MatrixFSDP's advantage there comes from
matrix-owner sharding, which avoids the expensive full-matrix optimizer gather
path.

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
- `configure_optimizer(...)`: optimizer helper for AdamW, SGD, Muon, and mixed
  Muon/AdamW training.
- `save_matrix_dcp(...)` and `load_matrix_dcp(...)`: DCP checkpoint helpers for
  MatrixFSDP shard metadata and optimizer state.

## Documentation

- [Introduction](docs/introduction.md): basic concepts, public APIs, `fully_shard(...)`
  arguments, DeviceMesh setup, optimizers, and MoE boundaries.
- [Tutorial](docs/tutorial.md): activation checkpointing, DCP save/load,
  full-state debug checkpoints, and resharded load.
