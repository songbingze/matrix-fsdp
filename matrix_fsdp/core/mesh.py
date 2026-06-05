from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.distributed.device_mesh import DeviceMesh

MeshDim = int | str


@dataclass(frozen=True)
class DataParallelMeshDims:
    shard: MeshDim | tuple[MeshDim, ...] | None = None
    replicate: MeshDim | tuple[MeshDim, ...] | None = None


@dataclass(frozen=True)
class DeviceMeshMetadata:
    device_type: str | None
    ndim: int
    shape: tuple[int, ...]
    mesh_dim_names: tuple[str, ...] | None
    coordinate: tuple[int, ...] | None
    shard_mesh_dim: int
    shard_mesh_dim_name: str | None
    shard_mesh_size: int
    replicate_mesh_dim: int | None
    replicate_mesh_dim_name: str | None
    replicate_mesh_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "ndim": self.ndim,
            "shape": self.shape,
            "mesh_dim_names": self.mesh_dim_names,
            "coordinate": self.coordinate,
            "shard_mesh_dim": self.shard_mesh_dim,
            "shard_mesh_dim_name": self.shard_mesh_dim_name,
            "shard_mesh_size": self.shard_mesh_size,
            "replicate_mesh_dim": self.replicate_mesh_dim,
            "replicate_mesh_dim_name": self.replicate_mesh_dim_name,
            "replicate_mesh_size": self.replicate_mesh_size,
        }


def infer_shard_mesh_dim(mesh: DeviceMesh | None, shard_mesh_dim: MeshDim | None = None) -> int:
    if mesh is None:
        if isinstance(shard_mesh_dim, str):
            raise ValueError("dp_shard_mesh_dim by name requires a DeviceMesh.")
        return 0 if shard_mesh_dim is None else shard_mesh_dim
    if shard_mesh_dim is not None:
        return normalize_mesh_dim(mesh, shard_mesh_dim, arg_name="dp_shard_mesh_dim")

    mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
    if mesh_dim_names is not None and "dp_shard" in mesh_dim_names:
        return normalize_mesh_dim(mesh, "dp_shard", arg_name="dp_shard_mesh_dim")
    if mesh.ndim == 1:
        return 0
    if mesh.ndim == 2:
        return 1
    raise ValueError(
        "Could not infer dp_shard_mesh_dim. Use a mesh dimension named 'dp_shard' "
        "or pass dp_shard_mesh_dim explicitly."
    )


def normalize_mesh_dim(mesh: DeviceMesh, mesh_dim: MeshDim, *, arg_name: str = "mesh_dim") -> int:
    if isinstance(mesh_dim, str):
        mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
        if mesh_dim_names is None:
            raise ValueError(f"{arg_name}={mesh_dim!r} requires a DeviceMesh with mesh_dim_names.")
        try:
            return mesh_dim_names.index(mesh_dim)
        except ValueError as error:
            raise ValueError(f"{arg_name}={mesh_dim!r} is not in mesh_dim_names={mesh_dim_names}.") from error

    normalized = mesh_dim
    if normalized < 0:
        normalized += mesh.ndim
    if normalized < 0 or normalized >= mesh.ndim:
        raise ValueError(f"{arg_name}={mesh_dim!r} is out of range for a {mesh.ndim}D DeviceMesh.")
    return normalized


def infer_replicate_mesh_dim(
    mesh: DeviceMesh | None,
    shard_mesh_dim: int,
    replicate_mesh_dim: MeshDim | None,
) -> int | None:
    if mesh is None:
        if replicate_mesh_dim is not None:
            raise ValueError("dp_replicate_mesh_dim requires a DeviceMesh.")
        return None
    if replicate_mesh_dim is not None:
        normalized = normalize_mesh_dim(mesh, replicate_mesh_dim, arg_name="dp_replicate_mesh_dim")
        if normalized == shard_mesh_dim:
            raise ValueError("dp_replicate_mesh_dim must be different from dp_shard_mesh_dim.")
        return normalized
    if mesh.ndim == 2:
        mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
        if mesh_dim_names is not None:
            if "dp_replicate" not in mesh_dim_names:
                return None
            normalized = normalize_mesh_dim(mesh, "dp_replicate", arg_name="dp_replicate_mesh_dim")
            if normalized == shard_mesh_dim:
                raise ValueError("Inferred dp_replicate mesh dim must be different from dp_shard_mesh_dim.")
            return normalized
        return 1 - shard_mesh_dim
    return None


def mesh_metadata(
    mesh: DeviceMesh | None,
    *,
    shard_mesh_dim: MeshDim | None = None,
    replicate_mesh_dim: MeshDim | None = None,
) -> DeviceMeshMetadata | None:
    if mesh is None:
        return None
    normalized_shard_dim = infer_shard_mesh_dim(mesh, shard_mesh_dim)
    normalized_replicate_dim = infer_replicate_mesh_dim(mesh, normalized_shard_dim, replicate_mesh_dim)
    return DeviceMeshMetadata(
        device_type=getattr(mesh, "device_type", None),
        ndim=int(mesh.ndim),
        shape=mesh_shape(mesh),
        mesh_dim_names=mesh_dim_names(mesh),
        coordinate=mesh_coordinate(mesh),
        shard_mesh_dim=normalized_shard_dim,
        shard_mesh_dim_name=mesh_dim_name(mesh, normalized_shard_dim),
        shard_mesh_size=int(mesh.size(normalized_shard_dim)),
        replicate_mesh_dim=normalized_replicate_dim,
        replicate_mesh_dim_name=(
            mesh_dim_name(mesh, normalized_replicate_dim)
            if normalized_replicate_dim is not None
            else None
        ),
        replicate_mesh_size=(
            int(mesh.size(normalized_replicate_dim))
            if normalized_replicate_dim is not None
            else 1
        ),
    )


def mesh_shape(mesh: DeviceMesh) -> tuple[int, ...]:
    mesh_tensor = getattr(mesh, "mesh", None)
    if mesh_tensor is not None:
        return tuple(int(dim) for dim in mesh_tensor.shape)
    return tuple(int(mesh.size(dim)) for dim in range(mesh.ndim))


def mesh_dim_names(mesh: DeviceMesh) -> tuple[str, ...] | None:
    names = getattr(mesh, "mesh_dim_names", None)
    if names is None:
        return None
    return tuple(str(name) for name in names)


def mesh_dim_name(mesh: DeviceMesh, mesh_dim: int | None) -> str | None:
    if mesh_dim is None:
        return None
    names = mesh_dim_names(mesh)
    if names is None:
        return None
    if mesh_dim < 0 or mesh_dim >= len(names):
        return None
    return names[mesh_dim]


def mesh_coordinate(mesh: DeviceMesh) -> tuple[int, ...] | None:
    try:
        coordinate = mesh.get_coordinate()
    except RuntimeError:
        return None
    if coordinate is None:
        return None
    return tuple(int(index) for index in coordinate)
