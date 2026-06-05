from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import nn


class ParamRuntimeKind:
    FSDP_GATHER = "fsdp_gather"
    EP_LOCAL_FSDP = "ep_local_fsdp"
    EXPERT_OWNER = "expert_owner"


@dataclass(frozen=True)
class ParamShardHint:
    optimizer_type: str | None = None
    split_granularity: str | None = None
    block_shape: tuple[int, ...] | None = None
    runtime_kind: str | None = None
    parallel_role: str | None = None
    expert_id: int | None = None
    expert_group_id: str | None = None
    owner_rank: int | None = None
    owner_replica_ranks: tuple[int, ...] = ()


@dataclass
class ManagedParam:
    fqn: str
    param: nn.Parameter
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    numel: int
    offset: int
    end: int
    shard_hint: ParamShardHint = field(default_factory=ParamShardHint)

    @property
    def local_name(self) -> str:
        return self.fqn.rsplit(".", 1)[-1]


@dataclass
class ManagedParamRegistry:
    params: tuple[ManagedParam, ...]
    _by_fqn: dict[str, ManagedParam] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_fqn = {}
        for managed_param in self.params:
            if managed_param.fqn in self._by_fqn:
                raise ValueError(f"Duplicate managed parameter fqn: {managed_param.fqn}.")
            self._by_fqn[managed_param.fqn] = managed_param

    def __iter__(self):
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, index: int) -> ManagedParam:
        return self.params[index]

    @property
    def total_numel(self) -> int:
        return sum(managed_param.numel for managed_param in self.params)

    @property
    def fqns(self) -> tuple[str, ...]:
        return tuple(managed_param.fqn for managed_param in self.params)

    def param(self, fqn: str) -> ManagedParam:
        return self._by_fqn[fqn]

    def as_list(self) -> list[ManagedParam]:
        return list(self.params)

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        shard_hints: Mapping[str, ParamShardHint] | None = None,
        ignored_params: set[nn.Parameter] | None = None,
    ) -> "ManagedParamRegistry":
        shard_hints = shard_hints or {}
        ignored_param_ids = {id(param) for param in ignored_params or set()}
        fqn_by_param_id: dict[int, list[str]] = {}
        for fqn, param in module.named_parameters(recurse=True, remove_duplicate=False):
            if id(param) in ignored_param_ids:
                continue
            fqn_by_param_id.setdefault(id(param), []).append(fqn)
        shared_params = [fqns for fqns in fqn_by_param_id.values() if len(fqns) > 1]
        if shared_params:
            shared = "; ".join(",".join(fqns) for fqns in shared_params)
            raise NotImplementedError(
                "MatrixFSDP does not support shared parameters yet. "
                f"Shared parameter aliases: {shared}."
            )
        managed_params: list[ManagedParam] = []
        offset = 0
        for fqn, param in module.named_parameters(recurse=True, remove_duplicate=True):
            if id(param) in ignored_param_ids:
                continue
            shard_hint = shard_hints.get(fqn, ParamShardHint())
            validate_param_shard_hint(fqn, param.shape, param.numel(), shard_hint)
            managed_params.append(
                ManagedParam(
                    fqn=fqn,
                    param=param,
                    shape=param.shape,
                    dtype=param.dtype,
                    device=param.device,
                    numel=param.numel(),
                    offset=offset,
                    end=offset + param.numel(),
                    shard_hint=shard_hint,
                )
            )
            offset += param.numel()
        unknown_hints = set(shard_hints) - {managed_param.fqn for managed_param in managed_params}
        if unknown_hints:
            unknown = ", ".join(sorted(unknown_hints))
            raise ValueError(f"Shard hints refer to unknown parameters: {unknown}.")
        return cls(tuple(managed_params))


def validate_param_shard_hint(
    fqn: str,
    shape: torch.Size | tuple[int, ...],
    numel: int,
    shard_hint: ParamShardHint,
) -> None:
    valid_granularities = {None, "parameter", "matrix_owner", "row_block", "block"}
    if shard_hint.split_granularity not in valid_granularities:
        valid = ", ".join(repr(granularity) for granularity in sorted(valid_granularities, key=lambda value: str(value)))
        raise ValueError(f"Param {fqn} has unknown split_granularity={shard_hint.split_granularity!r}; expected {valid}.")
    valid_runtime_kinds = {
        None,
        ParamRuntimeKind.FSDP_GATHER,
        ParamRuntimeKind.EP_LOCAL_FSDP,
        ParamRuntimeKind.EXPERT_OWNER,
    }
    if shard_hint.runtime_kind not in valid_runtime_kinds:
        valid = ", ".join(repr(kind) for kind in sorted(valid_runtime_kinds, key=lambda value: str(value)))
        raise ValueError(f"Param {fqn} has unknown runtime_kind={shard_hint.runtime_kind!r}; expected {valid}.")
    valid_parallel_roles = {
        None,
        "dense",
        "embedding",
        "norm",
        "router",
        "routed_expert",
        "shared_expert",
    }
    if shard_hint.parallel_role not in valid_parallel_roles:
        valid = ", ".join(repr(role) for role in sorted(valid_parallel_roles, key=lambda value: str(value)))
        raise ValueError(f"Param {fqn} has unknown parallel_role={shard_hint.parallel_role!r}; expected {valid}.")
    if shard_hint.expert_id is not None and shard_hint.expert_id < 0:
        raise ValueError(f"Param {fqn} expert_id must be non-negative, got {shard_hint.expert_id}.")
    if shard_hint.owner_rank is not None and shard_hint.owner_rank < 0:
        raise ValueError(f"Param {fqn} owner_rank must be non-negative, got {shard_hint.owner_rank}.")
    if any(rank < 0 for rank in shard_hint.owner_replica_ranks):
        raise ValueError(f"Param {fqn} owner_replica_ranks must be non-negative, got {shard_hint.owner_replica_ranks}.")
    if shard_hint.runtime_kind == ParamRuntimeKind.EXPERT_OWNER and not shard_hint.expert_group_id:
        raise ValueError(f"Param {fqn} runtime_kind='expert_owner' requires expert_group_id metadata.")
    if shard_hint.block_shape is not None:
        if not shard_hint.block_shape:
            raise ValueError(f"Param {fqn} block_shape cannot be empty.")
        if any(dim <= 0 for dim in shard_hint.block_shape):
            raise ValueError(f"Param {fqn} block_shape dims must be positive, got {shard_hint.block_shape}.")

    if shard_hint.split_granularity in {None, "parameter", "matrix_owner"}:
        if shard_hint.block_shape is not None:
            raise ValueError(
                f"Param {fqn} split_granularity={shard_hint.split_granularity!r} does not accept block_shape."
            )
        if shard_hint.split_granularity == "matrix_owner" and len(shape) != 2:
            raise ValueError(f"Param {fqn} matrix_owner split_granularity expects a 2D tensor, got shape={tuple(shape)}.")
        return

    if shard_hint.split_granularity == "block":
        if shard_hint.block_shape is None or len(shard_hint.block_shape) != 1:
            raise ValueError(f"Param {fqn} block split_granularity expects a 1D block_shape.")
        block_units = shard_hint.block_shape[0]
        if numel % block_units != 0:
            raise ValueError(f"Param {fqn} numel={numel} is not divisible by block_shape={shard_hint.block_shape}.")
        return

    if shard_hint.split_granularity == "row_block":
        if len(shape) != 2:
            raise ValueError(f"Param {fqn} row_block split_granularity expects a 2D tensor, got shape={tuple(shape)}.")
        if shard_hint.block_shape is None or len(shard_hint.block_shape) not in {1, 2}:
            raise ValueError(f"Param {fqn} row_block split_granularity expects a 1D or 2D block_shape.")
        rows, cols = tuple(shape)
        row_block = shard_hint.block_shape[0]
        if len(shard_hint.block_shape) == 2 and shard_hint.block_shape[1] != cols:
            raise ValueError(
                f"Param {fqn} row_block block_shape second dim={shard_hint.block_shape[1]} must match cols={cols}."
            )
        if rows % row_block != 0:
            raise ValueError(f"Param {fqn} rows={rows} is not divisible by row block={row_block}.")
