from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from .mesh import MeshDim, mesh_metadata
from .placement import MatrixShard
from .torch_dtensor import make_matrix_dtensor

if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor


class MatrixShardedState:
    """
    Own the logical sharded state for one local tensor.

    When a DeviceMesh and MatrixShard placement are available, the state is
    represented by a DTensor and the local tensor is recovered via ``to_local()``.
    Without a compatible placement, the same object still carries the local
    tensor so non-matrix layouts can share the same runtime interface.
    """

    def __init__(
        self,
        name: str,
        local_tensor: torch.Tensor,
        *,
        mesh: DeviceMesh | None,
        placement: MatrixShard | None,
        global_shape: Sequence[int],
        global_stride: Sequence[int] | None = None,
        shard_mesh_dim: MeshDim = 0,
        requires_grad: bool | None = None,
        dtensor_spec=None,
    ) -> None:
        self.name = name
        self.mesh = mesh
        self.placement = placement
        self.global_shape = torch.Size(tuple(global_shape))
        self.global_stride = tuple(global_stride) if global_stride is not None else None
        self.shard_mesh_dim = shard_mesh_dim
        self.requires_grad = local_tensor.requires_grad if requires_grad is None else requires_grad
        self.dtensor_spec = dtensor_spec
        self.mesh_metadata = (
            None
            if mesh is None
            else mesh_metadata(mesh, shard_mesh_dim=shard_mesh_dim).as_dict()
        )
        self._local_tensor: torch.Tensor | None = local_tensor
        self._dtensor = self._build_dtensor(local_tensor)
        if self._dtensor is not None:
            self._local_tensor = None

    @property
    def dtensor(self) -> "DTensor | None":
        return self._dtensor

    @property
    def local_tensor(self) -> torch.Tensor:
        if self._dtensor is not None:
            return self._dtensor.to_local()
        if self._local_tensor is None:
            raise RuntimeError(f"Sharded state {self.name!r} does not have local tensor storage.")
        return self._local_tensor

    @property
    def uses_dtensor(self) -> bool:
        return self._dtensor is not None

    @property
    def fallback_tensor(self) -> torch.Tensor | None:
        return self._local_tensor

    def refresh(self, local_tensor: torch.Tensor) -> "MatrixShardedState":
        return MatrixShardedState(
            self.name,
            local_tensor,
            mesh=self.mesh,
            placement=self.placement,
            global_shape=self.global_shape,
            global_stride=self.global_stride,
            shard_mesh_dim=self.shard_mesh_dim,
            requires_grad=local_tensor.requires_grad,
            dtensor_spec=self.dtensor_spec,
        )

    def shares_storage_with(self, tensor: torch.Tensor) -> bool:
        return self.local_tensor.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()

    def has_same_data_ptr(self, tensor: torch.Tensor) -> bool:
        return self.local_tensor.data_ptr() == tensor.data_ptr()

    def as_metadata(self) -> dict[str, Any]:
        return matrix_sharded_state_metadata(self)

    def _build_dtensor(self, local_tensor: torch.Tensor) -> "DTensor | None":
        if self.mesh is None or self.placement is None:
            return None
        return make_matrix_dtensor(
            local_tensor,
            self.mesh,
            self.placement,
            global_shape=self.global_shape,
            global_stride=self.global_stride,
            requires_grad=self.requires_grad,
            shard_mesh_dim=self.shard_mesh_dim,
            spec=self.dtensor_spec,
        )


def matrix_sharded_state_metadata(state: MatrixShardedState) -> dict[str, Any]:
    local_tensor = state.local_tensor
    return {
        "name": state.name,
        "uses_dtensor": state.uses_dtensor,
        "global_shape": tuple(state.global_shape),
        "global_stride": state.global_stride,
        "local_shape": tuple(local_tensor.shape),
        "local_numel": local_tensor.numel(),
        "dtype": str(local_tensor.dtype),
        "device": str(local_tensor.device),
        "requires_grad": state.requires_grad,
        "shard_mesh_dim": state.shard_mesh_dim,
        "device_mesh": state.mesh_metadata,
        "matrix_shard": matrix_shard_metadata(state.placement),
        "dtensor_spec": _dtensor_spec_metadata(state.dtensor),
    }


def matrix_shard_metadata(placement: MatrixShard | None) -> dict[str, Any] | None:
    if placement is None:
        return None
    return {
        "type": "MatrixShard",
        "dims": tuple(placement.dims),
        "local_units": tuple(placement.local_units),
    }


def _dtensor_spec_metadata(dtensor: "DTensor | None") -> dict[str, Any] | None:
    if dtensor is None:
        return None
    spec = dtensor._spec
    tensor_meta = getattr(spec, "tensor_meta", None)
    return {
        "mesh_shape": tuple(spec.mesh.shape),
        "mesh_dim_names": _mesh_dim_names(spec.mesh),
        "placements": tuple(_placement_metadata(placement) for placement in spec.placements),
        "shape": tuple(spec.shape),
        "stride": None if tensor_meta is None else tuple(tensor_meta.stride),
        "dtype": None if tensor_meta is None else str(tensor_meta.dtype),
    }


def _mesh_dim_names(mesh: DeviceMesh) -> tuple[str, ...] | None:
    names = getattr(mesh, "mesh_dim_names", None)
    if names is None:
        return None
    return tuple(names)


def _placement_metadata(placement: object) -> dict[str, Any]:
    if isinstance(placement, MatrixShard):
        return matrix_shard_metadata(placement) or {"type": "MatrixShard"}
    return {
        "type": type(placement).__name__,
        "repr": repr(placement),
    }
