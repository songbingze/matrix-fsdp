# Training Integrations

This guide covers activation checkpointing, `torch.compile`, and MatrixFSDP
checkpoint save/load. The examples assume the model has already been built on
the target device and that `torch.distributed` has been initialized when running
multi-rank training.

## Activation Checkpointing

The recommended path is the same one used by TorchTitan-style code:

1. build the original model;
2. apply activation checkpoint wrappers to transformer blocks;
3. shard the same block boundaries with MatrixFSDP;
4. construct the optimizer after sharding.

Use `CheckpointImpl.NO_REENTRANT` and set `use_saved_tensor_hooks=False` on the
MatrixFSDP groups. This is the path covered by the local and CUDA correctness
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

from matrix_fsdp import MatrixFSDPOptimizer, module_type_policy, matrix_fully_shard


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

model = matrix_fully_shard(
    model,
    mesh=mesh,
    wrap_policy=module_type_policy(TransformerBlock),
    reshard_after_forward=True,
    finalize_after_backward=True,
    backward_reduce_strategy="bucket_reduce_scatter",
    use_saved_tensor_hooks=False,
    use_zero_copy_grad_bucket=False,
)
optim = MatrixFSDPOptimizer(
    torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False),
    model,
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

- Apply activation checkpointing before `matrix_fully_shard(...)`, so wrap
  policies still see the original module structure.
- Keep checkpoint boundaries and FSDP param-group boundaries aligned when
  possible.
- Prefer `NO_REENTRANT` for new runs.
- Use `use_saved_tensor_hooks=False` when combining MatrixFSDP with activation
  checkpoint wrappers.

## torch.compile

`torch.compile` is usable for small smoke tests, but it is not the primary
validated performance path yet. MatrixFSDP uses Python hooks, runtime state
transitions, and distributed collectives, so expect graph breaks. Start with
`fullgraph=False` and verify numerics before relying on compiled runs.

Recommended top-level pattern:

```python
model = build_model().to(device=device, dtype=torch.bfloat16)

# Optional: apply activation checkpointing first.
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=False,
    ),
    check_fn=lambda module: isinstance(module, TransformerBlock),
)

model = matrix_fully_shard(
    model,
    mesh=mesh,
    wrap_policy=module_type_policy(TransformerBlock),
    reshard_after_forward=True,
    finalize_after_backward=True,
    backward_reduce_strategy="bucket_reduce_scatter",
    use_saved_tensor_hooks=False,
)
optim = MatrixFSDPOptimizer(
    torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False),
    model,
)

compiled_model = torch.compile(model, fullgraph=False)

loss = compiled_model(input_ids, labels)
loss.backward()
optim.step()
optim.zero_grad(set_to_none=True)
```

Keep the original sharded `model` handle for checkpointing and optimizer
construction. Use the compiled wrapper only for forward calls.

### Compile Notes

- Do not compile the optimizer or checkpoint save/load calls.
- If full-model compile is unstable, compile pure compute submodules before
  applying activation checkpointing and sharding.
- If using `wrap_policy`, apply it to the original module classes before any
  compile wrapper hides those classes.
- Use `fullgraph=False`. MatrixFSDP collectives and lifecycle hooks are expected
  graph-break points.
- Re-run a small correctness check after changing PyTorch versions, compile
  backends, activation checkpoint settings, or FSDP boundaries.

## DCP Checkpoint Save/Load

MatrixFSDP provides DCP helpers that save tensor payloads through PyTorch
Distributed Checkpoint and store MatrixFSDP layout metadata in a single global
`matrix_metadata.pt` file. Older checkpoints with `matrix_metadata_rank_*.pt`
sidecars remain loadable as a compatibility fallback.

Main APIs:

- `save_matrix_dcp(model_or_groups, checkpoint_dir, optimizer=...)`
- `load_matrix_dcp(model_or_groups, checkpoint_dir, optimizer=...)`
- `load_matrix_dcp_full_state(checkpoint_dir)`

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

restored_model = matrix_fully_shard(
    restored_model,
    mesh=mesh,
    wrap_policy=module_type_policy(TransformerBlock),
    reshard_after_forward=True,
    finalize_after_backward=True,
    backward_reduce_strategy="bucket_reduce_scatter",
    use_saved_tensor_hooks=False,
)
restored_optim = MatrixFSDPOptimizer(
    torch.optim.AdamW(restored_model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False),
    restored_model,
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
from matrix_fsdp import MatrixFSDPOptimizer, collect_param_groups


param_groups = collect_param_groups(model)
optim = MatrixFSDPOptimizer.from_shard_hints(
    param_groups,
    default_matrix_optimizer="muon",
    default_other_optimizer="adamw",
    muon_lr=0.03,
    muon_momentum=0.5,
    muon_ns_steps=2,
    adamw_lr=3e-4,
    adamw_foreach=False,
    lazy_muon_init=True,
)

save_matrix_dcp(model, checkpoint_dir, optimizer=optim)

restored_param_groups = collect_param_groups(restored_model)
restored_optim = MatrixFSDPOptimizer.from_shard_hints(
    restored_param_groups,
    default_matrix_optimizer="muon",
    default_other_optimizer="adamw",
    muon_lr=0.03,
    muon_momentum=0.5,
    muon_ns_steps=2,
    adamw_lr=3e-4,
    adamw_foreach=False,
    lazy_muon_init=True,
)
load_matrix_dcp(restored_model, checkpoint_dir, optimizer=restored_optim)
restored_optim.validate_local_state_shapes()
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

### Full-State Debug Checkpoint

For inspection or conversion, save full parameters in addition to local shards:

```python
from matrix_fsdp import load_matrix_dcp_full_state, save_matrix_dcp


save_matrix_dcp(model, checkpoint_dir, full_state=True)
full_state = load_matrix_dcp_full_state(checkpoint_dir)
params = full_state["params"]
```

This is a debug/conversion path, not the memory-efficient training checkpoint
path.

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
4. call `fully_shard(...)` or `matrix_fully_shard(...)`;
5. construct the optimizer;
6. optionally create `compiled_model = torch.compile(model, fullgraph=False)`;
7. train using either `model(...)` or `compiled_model(...)`;
8. save/load DCP through the original sharded `model` and optimizer.
