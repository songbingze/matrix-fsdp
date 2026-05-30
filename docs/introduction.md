# Introduction

MatrixFSDP is an experimental FSDP2-style runtime for PyTorch models that need
matrix-aware sharding. The public path is intentionally small:

1. build the model on the target device;
2. call `fully_shard(...)` on the model or on repeated blocks;
3. create the optimizer with `configure_optimizer(...)`;
4. train with the ordinary PyTorch loop.

```python
loss = model(input_ids, labels)
loss.backward()
optim.step()
optim.zero_grad(set_to_none=True)
```

Create optimizers after `fully_shard(...)` so they see the sharded parameter
views.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Public API

- `fully_shard(...)`: the main FSDP2-like sharding entry point.
- `DataParallelMeshDims`: selects which `DeviceMesh` dimensions are used for
  DP shard and optional DP replicate groups.
- `configure_optimizer(...)`: builds the supported optimizer path from the
  sharded model. Use this for AdamW, SGD, Muon, and mixed Muon/AdamW.
- `save_matrix_dcp(...)` and `load_matrix_dcp(...)`: save and load MatrixFSDP
  sharded checkpoints through PyTorch Distributed Checkpoint.

Regular training scripts should only need these public APIs.

## `fully_shard(...)`

```python
fully_shard(
    module,
    *,
    mesh=None,
    reshard_after_forward=None,
    shard_placement_fn=None,
    mp_policy=MixedPrecisionPolicy(),
    offload_policy=OffloadPolicy(),
    ignored_params=None,
    dp_mesh_dims=None,
    optimizer_policy=None,
)
```

Arguments:

- `module`: the `nn.Module` to manage. You can shard the whole model or call
  `fully_shard(...)` on repeated blocks.
- `mesh`: a `torch.distributed.device_mesh.DeviceMesh`. If omitted, MatrixFSDP
  falls back to the current distributed world as a 1D shard group.
- `dp_mesh_dims`: a `DataParallelMeshDims` value that maps `mesh` dimensions to
  MatrixFSDP data parallel roles.
  `DataParallelMeshDims(shard="dp_shard")` uses one sharding dimension.
  `DataParallelMeshDims(shard="dp_shard", replicate="dp_replicate")` enables an
  HSDP-style replicate x shard layout.
- `reshard_after_forward`: controls whether full parameters are released after
  forward. `None` follows the FSDP2 root-module default and keeps full
  parameters for the simple whole-model case. Use `True` when sharding blocks
  and you want each block to release full parameters after its forward. Integer
  subgroup resharding is reserved for a future implementation.
- `shard_placement_fn`: optional advanced callback for assigning per-parameter
  shard hints. Most users should leave this unset and use `optimizer_policy`.
  It cannot be combined with `optimizer_policy`.
- `mp_policy`: a PyTorch FSDP `MixedPrecisionPolicy`. It controls parameter,
  reduction, and output dtypes for managed parameters.
- `offload_policy`: a PyTorch FSDP `OffloadPolicy`. The default is no offload;
  CUDA parameter/gradient/optimizer-state offload is not the default path.
- `ignored_params`: parameters that MatrixFSDP should leave unmanaged. This is
  useful when another parallelism stack owns those tensors, such as routed MoE
  experts handled by EP.
- `optimizer_policy`: optional layout policy for optimizer-aware sharding.
  Leave unset for AdamW/SGD-style dense sharding. Use
  `"mixed_muon_adamw"` for matrix-owner Muon planning with AdamW fallback for
  non-matrix parameters.

The function returns the same module object, with MatrixFSDP runtime state
attached.

## DeviceMesh

1D data parallel sharding:

```python
import torch
from torch.distributed.device_mesh import DeviceMesh

from matrix_fsdp import DataParallelMeshDims, fully_shard


world_size = torch.distributed.get_world_size()
mesh = DeviceMesh("cuda", torch.arange(world_size), mesh_dim_names=("dp_shard",))

model = fully_shard(
    model,
    mesh=mesh,
    dp_mesh_dims=DataParallelMeshDims(shard="dp_shard"),
)
```

HSDP-style replicate x shard layout:

```python
dp_replicate = 2
dp_shard = world_size // dp_replicate
mesh = DeviceMesh(
    "cuda",
    torch.arange(world_size).reshape(dp_replicate, dp_shard),
    mesh_dim_names=("dp_replicate", "dp_shard"),
)

model = fully_shard(
    model,
    mesh=mesh,
    dp_mesh_dims=DataParallelMeshDims(shard="dp_shard", replicate="dp_replicate"),
)
```

## Optimizers

Use `configure_optimizer(...)` after sharding.

AdamW:

```python
from matrix_fsdp import configure_optimizer


optim = configure_optimizer(
    model,
    "adamw",
    lr=3e-4,
    weight_decay=0.01,
    foreach=False,
)
```

Mixed Muon/AdamW:

`optimizer_policy="mixed_muon_adamw"` controls the MatrixFSDP layout planner. It
does not create an optimizer by itself. Create the optimizer with
`configure_optimizer(model, "mixed_muon_adamw", ...)` after sharding.

```python
model = fully_shard(
    model,
    mesh=mesh,
    dp_mesh_dims=DataParallelMeshDims(shard="dp_shard", replicate="dp_replicate"),
    optimizer_policy="mixed_muon_adamw",
)

optim = configure_optimizer(
    model,
    "mixed_muon_adamw",
    lr=3e-4,
    weight_decay=0.01,
    muon_momentum=0.5,
    muon_ns_steps=2,
    adamw_foreach=False,
    lazy_muon_init=True,
)
```

The mixed optimizer accepts AdamW-style shared knobs. By default, `lr` is used
for both Muon matrix parameters and AdamW tail parameters, and `weight_decay` is
used for both paths. Pass `muon_lr` / `adamw_lr` or `muon_weight_decay` /
`adamw_weight_decay` only when you want to tune the two paths separately.

Muon-specific defaults are:

- `muon_momentum=0.5`
- `muon_ns_steps=2`
- `muon_adjust_lr_fn="match_rms_adamw"`

`match_rms_adamw` follows the Moonshot-style RMS matching rule exposed by
PyTorch Muon, so the recommended starting point is to reuse the AdamW learning
rate, weight decay, and schedule.

`configure_optimizer(...)` also accepts `"sgd"` and `"muon"` when the PyTorch
build provides the requested optimizer.

## Minimal AdamW Training

```python
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from matrix_fsdp import DataParallelMeshDims, configure_optimizer, fully_shard


def train() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    mesh = DeviceMesh("cuda", torch.arange(world_size), mesh_dim_names=("dp_shard",))

    model = build_model().to(device=device, dtype=torch.bfloat16)
    model = fully_shard(
        model,
        mesh=mesh,
        dp_mesh_dims=DataParallelMeshDims(shard="dp_shard"),
    )
    optim = configure_optimizer(model, "adamw", lr=3e-4, weight_decay=0.01)

    for _ in range(10):
        input_ids, labels = next_batch(device)
        loss = model(input_ids, labels)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

    dist.destroy_process_group()
```

Launch:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

## MoE Boundary

MatrixFSDP does not implement token routing, token dispatch/combine, EP
all-to-all, expert execution, or attention kernels. Those belong to the upper
training stack. MatrixFSDP manages only the DP/HSDP parameter set selected by
the user.

For DeepSeek-style MoE, let the EP stack own routed experts and pass those
parameters through `ignored_params`:

```python
def routed_expert_params(module):
    return {
        param
        for name, param in module.named_parameters(remove_duplicate=True)
        if ".local_experts." in name or ".experts." in name
    }


for block in model.layers:
    fully_shard(
        block,
        mesh=mesh,
        dp_mesh_dims=DataParallelMeshDims(shard="dp_shard"),
        ignored_params=routed_expert_params(block),
        optimizer_policy="mixed_muon_adamw",
        reshard_after_forward=True,
    )

optim = configure_optimizer(model, "mixed_muon_adamw", lr=3e-4, weight_decay=0.01)
```

The routed expert optimizer, EP-local mesh selection, and expert communication
remain outside MatrixFSDP.
