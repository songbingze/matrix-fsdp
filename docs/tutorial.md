# Tutorial

This tutorial covers activation checkpointing and MatrixFSDP checkpoint
save/load. The examples assume the model has already been built on the target
device and that `torch.distributed` has been initialized when running multi-rank
training.

## Activation Checkpointing

The recommended path is the same one used by TorchTitan-style code:

1. build the original model;
2. apply activation checkpoint wrappers to transformer blocks;
3. shard the same block boundaries with MatrixFSDP;
4. construct the optimizer after sharding.

Use `CheckpointImpl.NO_REENTRANT` and shard the checkpointed blocks with
`fully_shard(...)`. This is the path covered by the local and CUDA correctness
tests.

```python
from functools import partial

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from matrix_fsdp import configure_optimizer, fully_shard


class TransformerBlock(nn.Module):
    ...


model = build_model().to(device=device, dtype=torch.bfloat16)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=False,
    ),
    check_fn=lambda module: isinstance(module, TransformerBlock),
)

for block in (module for module in model.modules() if isinstance(module, TransformerBlock)):
    fully_shard(block, mesh=mesh, reshard_after_forward=True)

optim = configure_optimizer(
    model,
    "adamw",
    lr=3e-4,
    weight_decay=0.01,
    foreach=False,
    max_unsharded_prefetch_units=1,
)
```

Training stays ordinary:

```python
loss = model(input_ids, labels)
loss.backward()
optim.step()
optim.zero_grad(set_to_none=True)
```

### Checkpointing Notes

- Apply activation checkpointing before `fully_shard(...)`, so block classes are
  still visible.
- Keep checkpoint boundaries and MatrixFSDP sharding boundaries aligned when
  possible.
- Prefer `NO_REENTRANT` for new runs.
- Use the public `fully_shard(...)` path for examples and training scripts.

## DCP Checkpoint Save/Load

MatrixFSDP provides DCP helpers that save tensor payloads through PyTorch
Distributed Checkpoint and store MatrixFSDP layout metadata in a single global
`matrix_metadata.pt` file. Older checkpoints with `matrix_metadata_rank_*.pt`
sidecars remain loadable as a compatibility fallback.

Main APIs:

- `save_matrix_dcp(model, checkpoint_dir, optimizer=...)`
- `load_matrix_dcp(model, checkpoint_dir, optimizer=...)`

Every rank should call save/load with the same `checkpoint_dir`. Use
`no_dist=True` only for single-process debug or tests.

### Same-Layout Save/Load

Save model shards and optimizer state:

```python
from matrix_fsdp import load_matrix_dcp, save_matrix_dcp


save_matrix_dcp(
    model,
    "/checkpoints/run_000100",
    optimizer=optim,
)
```

Restore by rebuilding the same model, applying the same activation checkpoint
wrappers, sharding with the same mesh and policy, constructing the matching
optimizer, then loading:

```python
restored_model = build_model().to(device=device, dtype=torch.bfloat16)

apply_activation_checkpointing(
    restored_model,
    checkpoint_wrapper_fn=partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=False,
    ),
    check_fn=lambda module: isinstance(module, TransformerBlock),
)

for block in (module for module in restored_model.modules() if isinstance(module, TransformerBlock)):
    fully_shard(block, mesh=mesh, reshard_after_forward=True)

restored_optim = configure_optimizer(
    restored_model,
    "adamw",
    lr=3e-4,
    weight_decay=0.01,
    foreach=False,
)

load_matrix_dcp(
    restored_model,
    "/checkpoints/run_000100",
    optimizer=restored_optim,
)
```

After load, continue with the normal training loop.

### Mixed Muon/AdamW Optimizer Checkpoint

Construct the same mixed optimizer on restore:

```python
from matrix_fsdp import configure_optimizer


optim = configure_optimizer(
    model,
    "mixed_muon_adamw",
    default_matrix_optimizer="muon",
    default_other_optimizer="adamw",
    lr=3e-4,
    weight_decay=0.01,
    muon_momentum=0.5,
    muon_ns_steps=2,
    adamw_foreach=False,
    lazy_muon_init=True,
)

save_matrix_dcp(model, checkpoint_dir, optimizer=optim)

restored_optim = configure_optimizer(
    restored_model,
    "mixed_muon_adamw",
    default_matrix_optimizer="muon",
    default_other_optimizer="adamw",
    lr=3e-4,
    weight_decay=0.01,
    muon_momentum=0.5,
    muon_ns_steps=2,
    adamw_foreach=False,
    lazy_muon_init=True,
)
load_matrix_dcp(restored_model, checkpoint_dir, optimizer=restored_optim)
```

### Resharded Load

The default load path expects the same world size and same layout metadata. For
debug or migration flows, `allow_reshard=True` can load compatible checkpoints
into a different sharding layout:

```python
load_matrix_dcp(
    restored_model,
    checkpoint_dir,
    optimizer=restored_optim,
    allow_reshard=True,
)
```

Use this only when model FQNs and tensor shapes match. Validate the result with a
short forward/backward step before resuming long training.

### Grad Shards

To include local gradient shards:

```python
save_matrix_dcp(model, checkpoint_dir, include_grads=True)
load_matrix_dcp(model, checkpoint_dir)
```

This is useful for debugging and specialized resume flows. Most training
checkpoints only need parameters and optimizer state.

## Integration Order

Use this ordering for the most stable setup:

1. initialize distributed;
2. build model on the target device;
3. apply activation checkpointing, if used;
4. call `fully_shard(...)`;
5. construct the optimizer;
6. train with the sharded `model`;
7. save/load DCP through the sharded `model` and optimizer.
