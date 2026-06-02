from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from matrix_fsdp.core.layout import LayoutSegment


CustomAllGatherResolver = Callable[[tuple[tuple[LayoutSegment, ...], ...]], tuple[str, str]]


class ElasticParamBufferWorkspaceLease:
    def __init__(self, workspace: "ElasticParamBufferWorkspace", key: tuple[object, ...], tensor: torch.Tensor) -> None:
        self.workspace = workspace
        self.key = key
        self.tensor = tensor
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.workspace.release(self)
        self.released = True


class ElasticParamBufferWorkspace:
    def __init__(self, *, max_cached_per_key: int = 1) -> None:
        if max_cached_per_key < 0:
            raise ValueError("max_cached_per_key must be non-negative.")
        self._entries: dict[tuple[object, ...], list[dict[str, object]]] = {}
        self.max_cached_per_key = max_cached_per_key
        self.acquire_count = 0
        self.reuse_count = 0
        self.allocate_count = 0

    def acquire(self, reference: torch.Tensor, numel: int) -> ElasticParamBufferWorkspaceLease:
        if numel < 0:
            raise ValueError(f"numel must be non-negative, got {numel}.")
        key = _workspace_key(reference, numel)
        entries = self._entries.setdefault(key, [])
        self.acquire_count += 1
        for entry in entries:
            if not entry["in_use"]:
                entry["in_use"] = True
                self.reuse_count += 1
                return ElasticParamBufferWorkspaceLease(self, key, entry["tensor"])  # type: ignore[arg-type]
        tensor = reference.new_empty(numel)
        entries.append({"tensor": tensor, "in_use": True})
        self.allocate_count += 1
        return ElasticParamBufferWorkspaceLease(self, key, tensor)

    def release(self, lease: ElasticParamBufferWorkspaceLease) -> None:
        entries = self._entries.get(lease.key, ())
        for index, entry in enumerate(entries):
            if entry["tensor"] is lease.tensor:
                if self.max_cached_per_key == 0:
                    entries.pop(index)
                    if not entries:
                        self._entries.pop(lease.key, None)
                    return
                entry["in_use"] = False
                self._trim_idle_entries(lease.key)
                return
        raise RuntimeError("Attempted to release a workspace tensor that is not owned by this workspace.")

    def set_max_cached_per_key(self, max_cached_per_key: int) -> None:
        if max_cached_per_key < 0:
            raise ValueError("max_cached_per_key must be non-negative.")
        self.max_cached_per_key = max_cached_per_key
        for key in tuple(self._entries):
            self._trim_idle_entries(key)

    def stats(self) -> dict[str, object]:
        allocated_numel = 0
        in_use_numel = 0
        allocated_tensors = 0
        in_use_tensors = 0
        for entries in self._entries.values():
            for entry in entries:
                tensor = entry["tensor"]
                allocated_tensors += 1
                allocated_numel += tensor.numel()  # type: ignore[union-attr]
                if entry["in_use"]:
                    in_use_tensors += 1
                    in_use_numel += tensor.numel()  # type: ignore[union-attr]
        return {
            "workspace_acquire_count": self.acquire_count,
            "workspace_reuse_count": self.reuse_count,
            "workspace_allocate_count": self.allocate_count,
            "workspace_max_cached_per_key": self.max_cached_per_key,
            "workspace_allocated_tensors": allocated_tensors,
            "workspace_in_use_tensors": in_use_tensors,
            "workspace_allocated_numel": allocated_numel,
            "workspace_in_use_numel": in_use_numel,
        }

    def clear(self) -> None:
        self._entries.clear()

    def _trim_idle_entries(self, key: tuple[object, ...]) -> None:
        entries = self._entries.get(key)
        if not entries:
            return
        idle_indices = [index for index, entry in enumerate(entries) if not entry["in_use"]]
        overflow = len(idle_indices) - self.max_cached_per_key
        for index in reversed(idle_indices[: max(overflow, 0)]):
            entries.pop(index)
        if not entries:
            self._entries.pop(key, None)


def _workspace_key(reference: torch.Tensor, numel: int) -> tuple[object, ...]:
    device_index = reference.device.index if reference.device.index is not None else -1
    return (reference.device.type, device_index, reference.dtype, numel)


@dataclass(frozen=True)
class ElasticParamBufferLayout:
    total_numel: int
    shard_sizes: tuple[int, ...]
    rank_segments: tuple[tuple[LayoutSegment, ...], ...] | None

    @property
    def world_size(self) -> int:
        return len(self.shard_sizes)

    @property
    def total_shard_numel(self) -> int:
        return sum(self.shard_sizes)

    @property
    def max_shard_size(self) -> int:
        return max(self.shard_sizes, default=0)

    @property
    def min_shard_size(self) -> int:
        return min(self.shard_sizes, default=0)

    @property
    def padded_numel(self) -> int:
        return self.world_size * self.max_shard_size

    @property
    def padding_waste_numel(self) -> int:
        return max(self.padded_numel - self.total_shard_numel, 0)

    @property
    def padding_waste_ratio(self) -> float:
        return self.padding_waste_numel / self.total_shard_numel if self.total_shard_numel else 0.0

    @property
    def owner_imbalance_ratio(self) -> float:
        mean_shard_size = self.total_shard_numel / self.world_size if self.world_size else 0.0
        return self.max_shard_size / mean_shard_size if mean_shard_size else 0.0

    @property
    def segment_count(self) -> int:
        return sum(len(segments) for segments in self.rank_segments or ())

    @property
    def max_segments_per_rank(self) -> int:
        return max((len(segments) for segments in self.rank_segments or ()), default=0)

    @property
    def rank_chunk_fast_path(self) -> bool:
        return rank_segments_are_rank_contiguous_chunks(self.rank_segments)

    @property
    def packed_rank_shards_are_full_tensor_order(self) -> bool:
        if self.total_numel != self.total_shard_numel:
            return False
        if self.rank_segments is None:
            return True
        for rank, segments in enumerate(self.rank_segments):
            rank_base = sum(self.shard_sizes[:rank])
            local_cursor = 0
            for segment in segments:
                if segment.local_start != local_cursor:
                    return False
                packed_start = rank_base + segment.local_start
                packed_end = packed_start + segment.numel
                if segment.global_start != packed_start or segment.global_end != packed_end:
                    return False
                local_cursor = segment.local_end
            if local_cursor != self.shard_sizes[rank]:
                return False
        return True

    def as_summary(self) -> dict[str, object]:
        return {
            "rank_chunk_fast_path": self.rank_chunk_fast_path,
            "packed_rank_shards_are_full_tensor_order": self.packed_rank_shards_are_full_tensor_order,
            "segment_count": self.segment_count,
            "max_segments_per_rank": self.max_segments_per_rank,
            "shard_sizes": self.shard_sizes,
            "max_shard_size": self.max_shard_size,
            "min_shard_size": self.min_shard_size,
            "padding_waste_numel": self.padding_waste_numel,
            "padding_waste_ratio": self.padding_waste_ratio,
            "owner_imbalance_ratio": self.owner_imbalance_ratio,
        }


@dataclass(frozen=True)
class ElasticParamBufferWorkspacePlan:
    full_param_numel: int
    compact_rank_chunks_numel: int
    padded_rank_chunks_numel: int
    padding_waste_numel: int
    rank_chunk_fast_path: bool
    packed_full_order: bool
    owner_segment_collectives: bool
    native_group_broadcast_capable: bool
    native_sendrecv_chunk_capable: bool
    padded_all_gather_capable: bool
    compact_owner_reduce_scatter_capable: bool

    @property
    def preferred_workspace_kind(self) -> str:
        if self.native_group_broadcast_capable:
            return "owner_segment"
        if self.padded_all_gather_capable:
            return "padded_rank_chunks"
        if self.native_sendrecv_chunk_capable:
            return "rank_chunks"
        return "matrix_all_gather"

    @property
    def preferred_workspace_numel(self) -> int:
        if self.preferred_workspace_kind == "owner_segment":
            return self.full_param_numel
        if self.preferred_workspace_kind == "padded_rank_chunks":
            return self.padded_rank_chunks_numel
        if self.preferred_workspace_kind == "rank_chunks":
            return self.compact_rank_chunks_numel
        return self.padded_rank_chunks_numel

    @property
    def padding_waste_ratio(self) -> float:
        return self.padding_waste_numel / self.compact_rank_chunks_numel if self.compact_rank_chunks_numel else 0.0

    def as_summary(self) -> dict[str, object]:
        return {
            "workspace_full_param_numel": self.full_param_numel,
            "workspace_compact_rank_chunks_numel": self.compact_rank_chunks_numel,
            "workspace_padded_rank_chunks_numel": self.padded_rank_chunks_numel,
            "workspace_padding_waste_numel": self.padding_waste_numel,
            "workspace_padding_waste_ratio": self.padding_waste_ratio,
            "workspace_preferred_kind": self.preferred_workspace_kind,
            "workspace_preferred_numel": self.preferred_workspace_numel,
            "workspace_rank_chunk_fast_path": self.rank_chunk_fast_path,
            "workspace_packed_full_order": self.packed_full_order,
            "workspace_owner_segment_collectives": self.owner_segment_collectives,
            "workspace_native_group_broadcast_capable": self.native_group_broadcast_capable,
            "workspace_native_sendrecv_chunk_capable": self.native_sendrecv_chunk_capable,
            "workspace_padded_all_gather_capable": self.padded_all_gather_capable,
            "workspace_compact_owner_reduce_scatter_capable": self.compact_owner_reduce_scatter_capable,
        }


@dataclass(frozen=True)
class ElasticParamBuffer:
    layout: ElasticParamBufferLayout
    workspace: ElasticParamBufferWorkspace = field(default_factory=ElasticParamBufferWorkspace)

    def workspace_plan(
        self,
        *,
        can_direct_all_gather: bool,
        owner_segment_backend: str | None,
        native_kernel_available: bool = False,
        native_sendrecv_chunk_enabled: bool = True,
    ) -> ElasticParamBufferWorkspacePlan:
        owner_segment_collectives = owner_segment_backend is not None
        return ElasticParamBufferWorkspacePlan(
            full_param_numel=self.layout.total_numel,
            compact_rank_chunks_numel=self.layout.total_shard_numel,
            padded_rank_chunks_numel=self.layout.padded_numel,
            padding_waste_numel=self.layout.padding_waste_numel,
            rank_chunk_fast_path=self.layout.rank_chunk_fast_path,
            packed_full_order=self.layout.packed_rank_shards_are_full_tensor_order,
            owner_segment_collectives=owner_segment_collectives,
            native_group_broadcast_capable=owner_segment_backend == "custom" and native_kernel_available,
            native_sendrecv_chunk_capable=(
                owner_segment_backend == "custom"
                and native_kernel_available
                and native_sendrecv_chunk_enabled
                and self.layout.rank_chunk_fast_path
            ),
            padded_all_gather_capable=can_direct_all_gather,
            compact_owner_reduce_scatter_capable=owner_segment_collectives and self.layout.packed_rank_shards_are_full_tensor_order,
        )

    def communication_summary(
        self,
        *,
        param_gather_strategy: str,
        matrix_collective_backend: str,
        can_direct_all_gather: bool,
        owner_segment_backend: str | None,
        custom_allgather_resolver: CustomAllGatherResolver | None = None,
        native_kernel_available: bool = False,
        native_sendrecv_chunk_enabled: bool = True,
        custom_reduce_scatterv_impl: str | None = None,
    ) -> dict[str, object]:
        owner_segment_collectives = owner_segment_backend is not None
        custom_policy = None
        resolved_custom_impl = None
        if owner_segment_backend == "custom" and self.layout.rank_segments is not None:
            if custom_allgather_resolver is not None:
                custom_policy, resolved_custom_impl = custom_allgather_resolver(self.layout.rank_segments)
        effective_reduce_backend = "owner_reduce"
        resolved_custom_reduce_impl = None
        if owner_segment_backend == "custom":
            resolved_custom_reduce_impl = custom_reduce_scatterv_impl
            if custom_reduce_scatterv_impl == "native_reduce":
                effective_reduce_backend = "native_reduce" if native_kernel_available else "uneven_reduce_scatter"
            elif custom_reduce_scatterv_impl is not None:
                effective_reduce_backend = custom_reduce_scatterv_impl

        return {
            "param_gather_strategy": param_gather_strategy,
            "matrix_collective_backend": matrix_collective_backend,
            "effective_param_gather_backend": self.effective_param_gather_backend(
                param_gather_strategy=param_gather_strategy,
                can_direct_all_gather=can_direct_all_gather,
                owner_segment_backend=owner_segment_backend,
            ),
            "owner_segment_collectives": owner_segment_collectives,
            "owner_segment_backend": owner_segment_backend,
            "custom_allgatherv_policy": custom_policy,
            "resolved_custom_allgatherv_impl": resolved_custom_impl,
            "effective_grad_reduce_backend": effective_reduce_backend,
            "resolved_custom_reduce_scatterv_impl": resolved_custom_reduce_impl,
            **self.layout.as_summary(),
            **self.workspace_plan(
                can_direct_all_gather=can_direct_all_gather,
                owner_segment_backend=owner_segment_backend,
                native_kernel_available=native_kernel_available,
                native_sendrecv_chunk_enabled=native_sendrecv_chunk_enabled,
            ).as_summary(),
            **self.workspace.stats(),
        }

    def effective_param_gather_backend(
        self,
        *,
        param_gather_strategy: str,
        can_direct_all_gather: bool,
        owner_segment_backend: str | None,
    ) -> str:
        if self.layout.world_size == 1:
            return "single_rank_copy"
        if param_gather_strategy != "matrix_all_gather" and can_direct_all_gather:
            return "equal_all_gather"
        if owner_segment_backend is not None:
            return f"owner_segment:{owner_segment_backend}"
        return "matrix_all_gather"


def rank_segments_are_rank_contiguous_chunks(
    rank_segments: tuple[tuple[LayoutSegment, ...], ...] | None,
) -> bool:
    if rank_segments is None:
        return False
    cursor = 0
    for segments in rank_segments:
        if not segments:
            continue
        if len(segments) != 1:
            return False
        segment = segments[0]
        if segment.local_start != 0 or segment.global_start != cursor:
            return False
        cursor = segment.global_end
    return True
