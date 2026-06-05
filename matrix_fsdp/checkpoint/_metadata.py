from __future__ import annotations

from typing import Any

import torch

from matrix_fsdp.core.layout import LayoutSegment, ParamLayout, ParamSegment, MatrixGroupLayout
from matrix_fsdp.core.managed_param import ManagedParam, ParamShardHint
from matrix_fsdp.core.placement import MatrixShard
from matrix_fsdp.runtime.flat_buffer import MatrixFlatBuffer
from matrix_fsdp.runtime.param_group import MatrixFSDPParamGroup


def _canonical_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonical_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_metadata(item) for item in value)
    return value


def _param_metadata(
    unit: MatrixFSDPParamGroup,
    flat_buffer: MatrixFlatBuffer,
    managed_param: ManagedParam,
) -> dict[str, Any]:
    return {
        "shape": tuple(managed_param.shape),
        "dtype": str(managed_param.dtype),
        "device": str(managed_param.device),
        "numel": managed_param.numel,
        "offset": managed_param.offset,
        "end": managed_param.end,
        "owner_ranks": tuple(unit.owner_ranks(managed_param.fqn)),
        "local_segments": [
            _param_segment_metadata(segment)
            for segment in unit.rank_segments_for_param(managed_param.fqn, unit.rank)
        ],
        "shard_sizes": tuple(flat_buffer.param_shard_sizes(managed_param)),
        "matrix_shard": _matrix_shard_metadata(flat_buffer.param_matrix_shard(managed_param)),
        "shard_hint": _param_shard_hint_metadata(managed_param.shard_hint),
    }


def _param_shard_hint_metadata(shard_hint: ParamShardHint) -> dict[str, Any]:
    return {
        "optimizer_type": shard_hint.optimizer_type,
        "split_granularity": shard_hint.split_granularity,
        "block_shape": shard_hint.block_shape,
        "runtime_kind": shard_hint.runtime_kind,
        "parallel_role": shard_hint.parallel_role,
        "expert_id": shard_hint.expert_id,
        "expert_group_id": shard_hint.expert_group_id,
        "owner_rank": shard_hint.owner_rank,
        "owner_replica_ranks": tuple(shard_hint.owner_replica_ranks),
    }


def _layout_metadata(layout: MatrixGroupLayout | None) -> dict[str, Any] | None:
    if layout is None:
        return None
    return {
        "total_numel": layout.total_numel,
        "world_size": layout.world_size,
        "shard_sizes": tuple(layout.shard_sizes),
        "ranks": [
            {
                "rank": rank_layout.rank,
                "local_units": rank_layout.local_units,
                "segments": [_layout_segment_metadata(segment) for segment in rank_layout.segments],
            }
            for rank_layout in layout.ranks
        ],
        "params": [_param_layout_metadata(param_layout) for param_layout in layout.params],
    }


def _param_layout_metadata(param_layout: ParamLayout) -> dict[str, Any]:
    return {
        "fqn": param_layout.fqn,
        "global_start": param_layout.global_start,
        "global_end": param_layout.global_end,
        "segments": [_param_segment_metadata(segment) for segment in param_layout.segments],
    }


def _layout_segment_metadata(segment: LayoutSegment) -> dict[str, int]:
    return {
        "global_start": segment.global_start,
        "global_end": segment.global_end,
        "local_start": segment.local_start,
        "local_end": segment.local_end,
        "numel": segment.numel,
    }


def _param_segment_metadata(segment: ParamSegment) -> dict[str, Any]:
    return {
        "fqn": segment.fqn,
        "rank": segment.rank,
        "global_start": segment.global_start,
        "global_end": segment.global_end,
        "local_start": segment.local_start,
        "local_end": segment.local_end,
        "numel": segment.numel,
    }


def _matrix_shard_metadata(placement: MatrixShard | None) -> dict[str, Any] | None:
    if placement is None:
        return None
    return {
        "dims": tuple(placement.dims),
        "local_units": tuple(placement.local_units),
    }


def _checkpoint_tensor(tensor: torch.Tensor, *, clone: bool) -> torch.Tensor:
    tensor = tensor.detach()
    return tensor.clone() if clone else tensor


def _param_key(module_fqn: str, param_fqn: str, unit_index: int) -> str:
    if module_fqn:
        return f"{module_fqn}.{param_fqn}"
    if param_fqn:
        return param_fqn
    return f"unit{unit_index}"


def _dtype_from_string(dtype: str) -> torch.dtype:
    if not dtype.startswith("torch."):
        raise ValueError(f"Unsupported dtype string {dtype!r}.")
    dtype_name = dtype.split(".", 1)[1]
    value = getattr(torch, dtype_name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unsupported dtype string {dtype!r}.")
    return value
