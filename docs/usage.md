# MatrixFSDP Usage Guide

This guide shows the recommended user-facing paths for dense transformer models
and DeepSeek-style MoE models. Keep `README.md` as the short entry point; put
complete examples here.

## Setup

Install the package in the training environment:

```bash
pip install -r requirements.txt
pip install -e .
```

MatrixFSDP expects optimizers to be constructed after sharding, so they see the
local sharded parameter views.

The normal training loop shape is unchanged:

```python
loss = model(input_ids, labels)
loss.backward()
optim.step()
optim.zero_grad(set_to_none=True)
```

## API Map

- `fully_shard(...)` is the public FSDP2-like API.
- `matrix_fully_shard(...)` is the lower-level API for explicit planners,
  communication paths, wrap policies, and MoE `ignored_params`.
- `DataParallelMeshDims` maps a `DeviceMesh` to `dp_shard` and optional
  `dp_replicate` dimensions.
- `MatrixFSDPOptimizer` wraps a torch optimizer and owns runtime prefetch,
  backward finalization, and optimizer state bookkeeping.
- `configure_optimizer(...)` and `MatrixFSDPOptimizer.from_shard_hints(...)`
  build the mixed Muon/AdamW path from MatrixFSDP shard hints.
- `collect_param_groups(model)` returns the live MatrixFSDP parameter groups
  when you shard multiple modules.

## Shared Dense Model

The examples below use this tiny language model shape:

```python
import torch
import torch.nn.functional as F
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_ratio * hidden_size),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 8192,
        hidden_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten())
```

## Dense AdamW

Use this path for standard dense transformer training. It uses a 1D
`dp_shard` mesh and a regular `torch.optim.AdamW` created after
`fully_shard(...)`.

```python
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from matrix_fsdp import DataParallelMeshDims, fully_shard


def train_dense_adamw() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    mesh = DeviceMesh("cuda", torch.arange(world_size), mesh_dim_names=("dp_shard",))

    torch.manual_seed(0)
    model = TinyTransformerLM(
        vocab_size=8192,
        hidden_size=1024,
        num_layers=4,
        num_heads=8,
    ).to(device=device, dtype=torch.bfloat16)
    model = fully_shard(
        model,
        mesh=mesh,
        dp_mesh_dims=DataParallelMeshDims(shard="dp_shard"),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for step in range(10):
        input_ids = torch.randint(model.vocab_size, (2, 512), device=device)
        labels = torch.randint(model.vocab_size, (2, 512), device=device)

        loss = model(input_ids, labels)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        if dist.get_rank() == 0:
            print(f"step={step} loss={loss.item():.4f}")

    dist.destroy_process_group()
```

Launch:

```bash
torchrun --standalone --nproc_per_node=4 train_dense_adamw.py
```

## Dense Muon + AdamW Tail

This path uses matrix-owner planning for 2D parameters. Muon updates whole local
owner matrices; non-2D parameters and AdamW-tail matrix shards use AdamW.

The example uses a 2D mesh:

- `dp_replicate`: replicated data-parallel groups;
- `dp_shard`: the ranks inside each MatrixFSDP shard group.

```python
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from matrix_fsdp import DataParallelMeshDims, configure_optimizer, fully_shard


def train_dense_muon_hsdp() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    dp_replicate = 2
    if world_size % dp_replicate != 0:
        raise ValueError("world_size must be divisible by dp_replicate.")
    dp_shard = world_size // dp_replicate
    mesh = DeviceMesh(
        "cuda",
        torch.arange(world_size).reshape(dp_replicate, dp_shard),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )

    model = TinyTransformerLM(
        vocab_size=8192,
        hidden_size=1024,
        num_layers=4,
        num_heads=8,
    ).to(device=device, dtype=torch.bfloat16)
    model = fully_shard(
        model,
        mesh=mesh,
        dp_mesh_dims=DataParallelMeshDims(shard="dp_shard", replicate="dp_replicate"),
        optimizer_policy="mixed_muon_adamw",
    )
    optim = configure_optimizer(
        model,
        "mixed_muon_adamw",
        muon_lr=0.03,
        muon_momentum=0.5,
        muon_ns_steps=2,
        muon_weight_decay=0.0,
        adamw_lr=3e-4,
        adamw_weight_decay=0.01,
        adamw_foreach=False,
        lazy_muon_init=True,
    )

    for step in range(10):
        input_ids = torch.randint(model.vocab_size, (2, 512), device=device)
        labels = torch.randint(model.vocab_size, (2, 512), device=device)

        loss = model(input_ids, labels)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        if dist.get_rank() == 0:
            print(f"step={step} loss={loss.item():.4f}")

    dist.destroy_process_group()
```

`muon_ns_steps=2` is the repository benchmark setting. The raw
`torch.optim.Muon` default in current PyTorch builds may differ.

For the custom Muon gather path, enable the native NCCL/sendrecv kernel when it
is available:

```bash
export MATRIX_FSDP_ENABLE_NATIVE_NCCL=1
export MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL=native_sendrecv
export MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL=uneven_reduce_scatter
```

## Block-Level Dense Sharding

For large transformers, shard each block as its own param group so prefetch and
backward reduce scheduling have useful boundaries.

```python
import torch

from matrix_fsdp import (
    MatrixFSDPOptimizer,
    collect_param_groups,
    module_type_policy,
    matrix_fully_shard,
)


model = TinyTransformerLM(...).to(device=device, dtype=torch.bfloat16)
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
param_groups = collect_param_groups(model)
optim = MatrixFSDPOptimizer(
    torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False),
    param_groups,
    max_unsharded_prefetch_units=1,
)
```

## DeepSeek-Style MoE

The recommended MoE contract is deliberately narrow:

- upstream EP owns routed expert placement, routing, token dispatch/combine,
  all-to-all, and expert execution;
- each rank sees only its EP-local routed expert tensors;
- MatrixFSDP shards dense/router/shared/norm/MTP parameters;
- routed local expert parameters are passed through `ignored_params`;
- eFSDP / EP-local expert FSDP is not part of the default path.

### MoE Helpers

```python
from torch import nn

from matrix_fsdp import build_shard_hints


def routed_expert_params(module: nn.Module) -> set[nn.Parameter]:
    return {
        param
        for name, param in module.named_parameters(remove_duplicate=True)
        if ".local_experts." in name or ".experts." in name
    }


def dense_shard_hints(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
) -> dict[str, object]:
    ignored_ids = {id(param) for param in ignored_params}
    params_by_name = dict(module.named_parameters(remove_duplicate=True))
    return {
        name: hint
        for name, hint in build_shard_hints(module).items()
        if id(params_by_name[name]) not in ignored_ids
    }


def non_routed_expert_params(model: nn.Module) -> list[nn.Parameter]:
    return [
        param
        for name, param in model.named_parameters(remove_duplicate=True)
        if ".local_experts." not in name and ".experts." not in name
    ]
```

### Shard Dense Blocks

Use one stateful Muon-aware planner across blocks. This lets owner assignment
balance cumulatively across the model instead of trying to make every block
perfectly balanced on its own.

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from matrix_fsdp import make_muon_shard_aware_group_planner, matrix_fully_shard


world_size = dist.get_world_size()
mesh = DeviceMesh("cuda", torch.arange(world_size), mesh_dim_names=("dp_shard",))
dense_planner = make_muon_shard_aware_group_planner(owner_assignment="role_greedy")
dense_fsdp_groups = []

for block in model.layers:
    ignored = routed_expert_params(block)
    matrix_fully_shard(
        block,
        mesh=mesh,
        ignored_params=ignored,
        shard_hints=dense_shard_hints(block, ignored),
        auto_shard_hints=False,
        group_planner=dense_planner,
        reshard_after_forward=True,
        finalize_after_backward=True,
        backward_reduce_strategy="bucket_reduce_scatter",
        use_saved_tensor_hooks=False,
        use_zero_copy_grad_bucket=False,
    )
    dense_fsdp_groups.append(block._matrix_fsdp_param_group)
```

The dense group may include attention, router/gate, norms, shared experts, MTP,
embeddings, and lm head if those parameters are not owned by another framework
component. Routed experts stay ignored.

### MoE AdamW

If the EP framework owns routed expert optimizer state, construct the optimizer
from dense parameters only:

```python
import torch

from matrix_fsdp import MatrixFSDPOptimizer


dense_params = non_routed_expert_params(model)
optim = MatrixFSDPOptimizer(
    torch.optim.AdamW(dense_params, lr=3e-4, weight_decay=0.01, foreach=False),
    dense_fsdp_groups,
    max_unsharded_prefetch_units=1,
)
```

For a single-framework smoke test, including `model.parameters()` is acceptable:
local experts will remain ordinary local tensors updated by the torch optimizer,
not FSDP-sharded tensors.

### MoE Muon

Use the dense FSDP groups so only dense 2D matrix-owner parameters enter Muon.
Non-2D dense tail parameters use AdamW. Routed local experts remain under the EP
stack.

```python
from matrix_fsdp import MatrixFSDPOptimizer


optim = MatrixFSDPOptimizer.from_shard_hints(
    dense_fsdp_groups,
    default_matrix_optimizer="muon",
    default_other_optimizer="adamw",
    muon_lr=0.03,
    muon_momentum=0.5,
    muon_ns_steps=2,
    muon_weight_decay=0.0,
    adamw_lr=3e-4,
    adamw_weight_decay=0.01,
    adamw_foreach=False,
    lazy_muon_init=True,
    max_unsharded_prefetch_units=1,
)
```

### MoE Validation

The ready-to-run CPU/GPU smoke tests and benchmark commands are in
[MoE Validation Commands](moe_validation_commands.md).
