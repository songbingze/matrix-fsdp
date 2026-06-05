from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch.distributed.device_mesh import DeviceMesh

from .mesh import MeshDim, normalize_mesh_dim
from .placement import MatrixShard

if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Placement


def make_matrix_dtensor(
    local_tensor: torch.Tensor,
    mesh: DeviceMesh,
    placement: MatrixShard,
    *,
    global_shape: Sequence[int],
    global_stride: Sequence[int] | None = None,
    requires_grad: bool | None = None,
    shard_mesh_dim: MeshDim = 0,
    spec=None,
) -> "DTensor":
    """
    Wrap a local sharded tensor as a PyTorch DTensor with MatrixShard placement.

    This is a metadata wrapper around the existing local tensor storage. MatrixFSDP
    still owns the communication and parameter lifecycle.
    """

    from torch.distributed.tensor import DTensor
    spec = spec or make_matrix_dtensor_spec(
        mesh,
        placement,
        global_shape=global_shape,
        global_stride=global_stride,
        dtype=local_tensor.dtype,
        shard_mesh_dim=shard_mesh_dim,
    )
    return DTensor(
        local_tensor.view_as(local_tensor),
        spec,
        requires_grad=local_tensor.requires_grad if requires_grad is None else requires_grad,
    )


def make_matrix_dtensor_spec(
    mesh: DeviceMesh,
    placement: MatrixShard,
    *,
    global_shape: Sequence[int],
    global_stride: Sequence[int] | None = None,
    dtype: torch.dtype,
    shard_mesh_dim: MeshDim = 0,
):
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

    shape = torch.Size(tuple(global_shape))
    stride = tuple(global_stride) if global_stride is not None else _contiguous_stride(shape)
    return DTensorSpec(
        mesh,
        make_matrix_placements(mesh, placement, shard_mesh_dim=shard_mesh_dim),
        tensor_meta=TensorMeta(
            shape=shape,
            stride=stride,
            dtype=dtype,
        ),
    )


def make_matrix_placements(
    mesh: DeviceMesh,
    placement: MatrixShard,
    *,
    shard_mesh_dim: MeshDim = 0,
) -> tuple["Placement", ...]:
    """
    Build FSDP-style DTensor placements for a matrix shard.

    The matrix placement lives on the dp_shard dimension. Other mesh dimensions
    are replicated, matching the usual FSDP2/HSDP interpretation of a 2D
    ``(dp_replicate, dp_shard)`` mesh.
    """

    from torch.distributed.tensor.placement_types import Replicate

    shard_dim = normalize_mesh_dim(mesh, shard_mesh_dim, arg_name="shard_mesh_dim")
    placements: list[Placement] = [Replicate() for _ in range(mesh.ndim)]
    placements[shard_dim] = placement
    return tuple(placements)


def _contiguous_stride(shape: torch.Size) -> tuple[int, ...]:
    stride = []
    running = 1
    for dim_size in reversed(shape):
        stride.append(running)
        running *= dim_size
    return tuple(reversed(stride))
