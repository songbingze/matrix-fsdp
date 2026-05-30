from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import torch

from .layout import MatrixGroupLayout, ShardPlan

try:
    from torch.distributed.tensor.placement_types import Placement as TorchPlacement
except Exception:  # pragma: no cover - only for environments without PyTorch DTensor placement types.
    class TorchPlacement:  # type: ignore[no-redef]
        pass


def is_matrix_shard_placement(placement: object) -> bool:
    return isinstance(placement, MatrixShard)


if not hasattr(TorchPlacement, "is_matrix_shard"):
    setattr(TorchPlacement, "is_matrix_shard", is_matrix_shard_placement)


@dataclass(frozen=True, init=False)
class MatrixShard(TorchPlacement):
    """
    Lightweight MatrixShard placement used by MatrixFSDP.

    This intentionally does not depend on an external DTensor runtime or the
    PyTorch DTensor dispatcher. It is a placement object rather than a DTensor
    subclass.
    It reuses the core MatrixShard contract: a logical tensor is flattened over
    ``dims`` and split into rank-ordered contiguous chunks whose sizes are
    proportional to ``local_units``.
    """

    dims: tuple[int, ...]
    local_units: tuple[int, ...]

    def __init__(self, dims: tuple[int, ...], local_units: tuple[int, ...]) -> None:
        # PyTorch 2.10 moved Placement initialization into a C++ base class.
        # Dataclass-generated __init__ would bypass it, so initialize the base
        # explicitly before assigning frozen fields.
        super().__init__()
        object.__setattr__(self, "dims", tuple(dims))
        object.__setattr__(self, "local_units", tuple(local_units))
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.local_units:
            raise ValueError("local_units cannot be empty.")
        if any(unit < 0 for unit in self.local_units):
            raise ValueError(f"local_units must be non-negative, got {self.local_units}.")
        if sum(self.local_units) == 0:
            raise ValueError("local_units cannot all be zero.")
        if any(dim < 0 for dim in self.dims):
            raise ValueError(f"dims must be non-negative, got {self.dims}.")

    @property
    def world_size(self) -> int:
        return len(self.local_units)

    @property
    def total_units(self) -> int:
        return sum(self.local_units)

    def shard_lengths(self, total_numel: int) -> tuple[int, ...]:
        ratio = self._ratio(total_numel)
        return tuple(unit * ratio for unit in self.local_units)

    def shard_offset(self, rank: int, total_numel: int) -> int:
        self._validate_rank(rank)
        ratio = self._ratio(total_numel)
        return sum(self.local_units[:rank]) * ratio

    def shard_range(self, rank: int, total_numel: int) -> tuple[int, int]:
        start = self.shard_offset(rank, total_numel)
        return start, start + self.shard_lengths(total_numel)[rank]

    def split_tensor(self, tensor: torch.Tensor, *, num_chunks: int | None = None) -> tuple[torch.Tensor, ...]:
        """
        Split a contiguous tensor into rank-ordered matrix shards.

        Returned shards are views into ``tensor``. Callers that need owned
        storage should clone or materialize their selected shard.
        """
        if not tensor.is_contiguous():
            raise ValueError("MatrixShard expects a contiguous tensor.")
        if num_chunks is not None and num_chunks != self.world_size:
            raise ValueError(f"num_chunks={num_chunks} must equal len(local_units)={self.world_size}.")

        flat_tensor = tensor.view(-1)
        lengths = self.shard_lengths(flat_tensor.numel())
        start = 0
        shards = []
        for length in lengths:
            shards.append(flat_tensor.narrow(0, start, length))
            start += length
        return tuple(shards)

    def local_shard(self, tensor: torch.Tensor, rank: int) -> torch.Tensor:
        self._validate_rank(rank)
        return self.split_tensor(tensor)[rank]

    def is_matrix_shard(self) -> bool:
        return True

    def reconstruct_tensor_from_flat(
        self,
        flat_tensor: torch.Tensor,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        if flat_tensor.ndim != 1:
            raise ValueError(f"flat_tensor must be 1D, got shape={tuple(flat_tensor.shape)}.")

        ndim = len(self.dims)
        if self.dims != tuple(range(ndim)):
            raise ValueError(f"dims must be a prefix tuple like (0, 1, ...), got {self.dims}.")
        trailing_numel = 1
        for dim_size in shape[ndim:]:
            trailing_numel *= dim_size
        if flat_tensor.numel() % trailing_numel != 0:
            raise ValueError(
                f"flat_tensor.numel()={flat_tensor.numel()} must be divisible by trailing shape product={trailing_numel}."
            )
        return flat_tensor.view(-1, *shape[ndim:])

    def _ratio(self, total_numel: int) -> int:
        if total_numel < 0:
            raise ValueError(f"total_numel must be non-negative, got {total_numel}.")
        if total_numel % self.total_units != 0:
            raise ValueError(
                f"total_numel={total_numel} must be divisible by sum(local_units)={self.total_units}."
            )
        return total_numel // self.total_units

    def _validate_rank(self, rank: int) -> None:
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"rank={rank} is out of range for world_size={self.world_size}.")

    def __repr__(self) -> str:
        return f"MatrixShard(dims={self.dims}, local_units={self.local_units})"

    __str__ = __repr__


MatrixShardPlacement = MatrixShard


@dataclass(frozen=True)
class PlacementCompatibility:
    compatible: bool
    reason: str | None = None
    requires_flat_reorder: bool = False
    requires_multi_segment_runtime: bool = False


def shard_sizes_to_matrix_local_units(shard_sizes: tuple[int, ...]) -> tuple[int, ...]:
    if not shard_sizes:
        raise ValueError("shard_sizes cannot be empty.")
    if any(shard_size < 0 for shard_size in shard_sizes):
        raise ValueError(f"shard_sizes must be non-negative, got {shard_sizes}.")
    total_numel = sum(shard_sizes)
    if total_numel == 0:
        raise ValueError("Matrix local_units cannot represent an all-empty shard plan.")
    divisor = 0
    for shard_size in shard_sizes:
        divisor = gcd(divisor, shard_size)
    divisor = max(divisor, 1)
    return tuple(shard_size // divisor for shard_size in shard_sizes)


def explain_matrix_shard_compatibility(
    layout_or_plan: MatrixGroupLayout | ShardPlan,
) -> PlacementCompatibility:
    plan = _to_shard_plan(layout_or_plan)
    if plan.total_numel <= 0:
        return PlacementCompatibility(False, "empty shard plans cannot be represented as matrix placements")
    if plan.rank_segments is None:
        return PlacementCompatibility(False, "missing rank segment metadata")

    cursor = 0
    for rank, segments in enumerate(plan.rank_segments):
        if plan.shard_sizes[rank] == 0:
            if segments:
                return PlacementCompatibility(False, f"rank {rank} has segments for an empty shard")
            continue
        if len(segments) != 1:
            return PlacementCompatibility(
                False,
                f"rank {rank} has {len(segments)} local segments, expected exactly one",
                requires_flat_reorder=True,
                requires_multi_segment_runtime=True,
            )
        segment = segments[0]
        if segment.local_start != 0:
            return PlacementCompatibility(False, f"rank {rank} segment local_start={segment.local_start}, expected 0")
        if segment.global_start != cursor:
            return PlacementCompatibility(
                False,
                f"rank {rank} segment starts at {segment.global_start}, expected rank-ordered cursor {cursor}",
                requires_flat_reorder=True,
            )
        if segment.numel != plan.shard_sizes[rank]:
            return PlacementCompatibility(
                False,
                f"rank {rank} segment size={segment.numel}, expected shard size={plan.shard_sizes[rank]}",
            )
        cursor = segment.global_end

    if cursor != plan.total_numel:
        return PlacementCompatibility(False, f"rank segments cover {cursor} elements, expected {plan.total_numel}")
    return PlacementCompatibility(True)


def is_matrix_shard_compatible_plan(layout_or_plan: MatrixGroupLayout | ShardPlan) -> bool:
    return explain_matrix_shard_compatibility(layout_or_plan).compatible


def matrix_shard_from_plan(plan: ShardPlan, *, dims: tuple[int, ...] = (0,)) -> MatrixShard:
    compatibility = explain_matrix_shard_compatibility(plan)
    if not compatibility.compatible:
        raise ValueError(
            "MatrixShard can directly represent only contiguous rank-ordered flat shard plans. "
            f"Reason: {compatibility.reason}."
        )
    return MatrixShard(dims=dims, local_units=shard_sizes_to_matrix_local_units(plan.shard_sizes))


def matrix_shard_from_layout(layout: MatrixGroupLayout, *, dims: tuple[int, ...] = (0,)) -> MatrixShard:
    return matrix_shard_from_plan(layout.to_shard_plan(), dims=dims)


def _to_shard_plan(layout_or_plan: MatrixGroupLayout | ShardPlan) -> ShardPlan:
    if isinstance(layout_or_plan, MatrixGroupLayout):
        return layout_or_plan.to_shard_plan()
    return layout_or_plan
