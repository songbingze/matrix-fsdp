from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from matrix_fsdp.core.layout import LayoutSegment
from matrix_fsdp.runtime.workspace_cache import CommWorkspaceCache, CommWorkspaceLease, comm_workspace_key


CustomAllGatherResolver = Callable[[tuple[tuple[LayoutSegment, ...], ...]], tuple[str, str]]
ElasticParamBufferWorkspace = CommWorkspaceCache
ElasticParamBufferWorkspaceLease = CommWorkspaceLease


@dataclass(frozen=True)
class ElasticRankChunk:
    rank: int
    global_start: int
    global_end: int
    local_start: int = 0

    @property
    def numel(self) -> int:
        return self.global_end - self.global_start

    def as_segment(self) -> LayoutSegment:
        return LayoutSegment(self.global_start, self.global_end, self.local_start)


@dataclass(frozen=True)
class ElasticCommunicationPlan:
    rank_segments: tuple[tuple[LayoutSegment, ...], ...]
    coalesced_rank_segments: tuple[tuple[LayoutSegment, ...], ...]
    rank_chunks: tuple[ElasticRankChunk, ...]
    rank_chunk_shard_sizes: tuple[int, ...] | None
    rank_chunk_segments: tuple[tuple[LayoutSegment, ...], ...] | None
    rank_chunk_fast_path: bool
    packed_full_order: bool

    @property
    def world_size(self) -> int:
        return len(self.rank_segments)

    @property
    def segment_count(self) -> int:
        return sum(len(segments) for segments in self.rank_segments)

    @property
    def coalesced_segment_count(self) -> int:
        return sum(len(segments) for segments in self.coalesced_rank_segments)

    @property
    def max_segments_per_rank(self) -> int:
        return max((len(segments) for segments in self.rank_segments), default=0)

    @property
    def max_coalesced_segments_per_rank(self) -> int:
        return max((len(segments) for segments in self.coalesced_rank_segments), default=0)

    @property
    def rank_chunk_count(self) -> int:
        return len(self.rank_chunks)

    @property
    def nonempty_rank_chunk_count(self) -> int:
        return sum(1 for chunk in self.rank_chunks if chunk.numel > 0)

    def as_summary(self) -> dict[str, object]:
        return {
            "communication_plan_segment_count": self.segment_count,
            "communication_plan_coalesced_segment_count": self.coalesced_segment_count,
            "communication_plan_max_segments_per_rank": self.max_segments_per_rank,
            "communication_plan_max_coalesced_segments_per_rank": self.max_coalesced_segments_per_rank,
            "communication_plan_rank_chunk_count": self.rank_chunk_count,
            "communication_plan_nonempty_rank_chunk_count": self.nonempty_rank_chunk_count,
            "communication_plan_rank_chunk_fast_path": self.rank_chunk_fast_path,
            "communication_plan_packed_full_order": self.packed_full_order,
            "communication_plan_has_rank_chunk_shard_sizes": self.rank_chunk_shard_sizes is not None,
        }


class ElasticRankChunkWorkspaceLease:
    def __init__(
        self,
        buffer: "ElasticParamBuffer",
        key: tuple[object, ...],
        tensor: torch.Tensor,
        *,
        persistent: bool,
        fallback_lease: CommWorkspaceLease | None = None,
    ) -> None:
        self.buffer = buffer
        self.key = key
        self.tensor = tensor
        self.persistent = persistent
        self.fallback_lease = fallback_lease
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        if self.fallback_lease is not None:
            self.fallback_lease.release()
        else:
            self.buffer.release_persistent_rank_chunk_workspace(self)
        self.released = True


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

    def build_communication_plan(self) -> ElasticCommunicationPlan:
        rank_segments = self.rank_segments or _rank_segments_from_shard_sizes(self.shard_sizes)
        coalesced_rank_segments = _coalesce_rank_segments(rank_segments)
        rank_chunk_shard_sizes = _rank_chunk_shard_sizes(rank_segments)
        rank_chunk_segments = (
            _rank_chunk_segments_from_sizes(rank_chunk_shard_sizes)
            if rank_chunk_shard_sizes is not None
            else None
        )
        return ElasticCommunicationPlan(
            rank_segments=rank_segments,
            coalesced_rank_segments=coalesced_rank_segments,
            rank_chunks=_rank_chunks_from_segments(rank_segments),
            rank_chunk_shard_sizes=rank_chunk_shard_sizes,
            rank_chunk_segments=rank_chunk_segments,
            rank_chunk_fast_path=rank_chunk_shard_sizes is not None,
            packed_full_order=self.packed_rank_shards_are_full_tensor_order,
        )


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
        if self.native_sendrecv_chunk_capable:
            return "rank_chunks"
        if self.native_group_broadcast_capable:
            return "owner_segment"
        if self.padded_all_gather_capable:
            return "padded_rank_chunks"
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
    communication_plan: ElasticCommunicationPlan = field(init=False)
    _persistent_rank_chunk_workspaces: dict[tuple[object, ...], dict[str, object]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    persistent_workspace_acquire_count: int = field(default=0, init=False)
    persistent_workspace_reuse_count: int = field(default=0, init=False)
    persistent_workspace_allocate_count: int = field(default=0, init=False)
    persistent_workspace_fallback_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "communication_plan", self.layout.build_communication_plan())

    def rank_chunk_workspace_numel(self, *, compact: bool) -> int:
        return self.layout.total_shard_numel if compact else self.layout.padded_numel

    def acquire_rank_chunk_workspace(
        self,
        reference: torch.Tensor,
        *,
        compact: bool,
        persistent: bool = False,
    ) -> ElasticParamBufferWorkspaceLease | ElasticRankChunkWorkspaceLease:
        numel = self.rank_chunk_workspace_numel(compact=compact)
        if not persistent:
            return self.workspace.acquire(reference, numel)
        return self.acquire_persistent_rank_chunk_workspace(reference, compact=compact)

    def acquire_persistent_rank_chunk_workspace(
        self,
        reference: torch.Tensor,
        *,
        compact: bool,
    ) -> ElasticRankChunkWorkspaceLease:
        object.__setattr__(self, "persistent_workspace_acquire_count", self.persistent_workspace_acquire_count + 1)
        numel = self.rank_chunk_workspace_numel(compact=compact)
        key = (*comm_workspace_key(reference, numel), "compact" if compact else "padded")
        entry = self._persistent_rank_chunk_workspaces.get(key)
        if entry is None:
            tensor = reference.new_empty(numel)
            self._persistent_rank_chunk_workspaces[key] = {"tensor": tensor, "in_use": True}
            object.__setattr__(
                self,
                "persistent_workspace_allocate_count",
                self.persistent_workspace_allocate_count + 1,
            )
            return ElasticRankChunkWorkspaceLease(self, key, tensor, persistent=True)
        if not entry["in_use"]:
            entry["in_use"] = True
            object.__setattr__(
                self,
                "persistent_workspace_reuse_count",
                self.persistent_workspace_reuse_count + 1,
            )
            return ElasticRankChunkWorkspaceLease(self, key, entry["tensor"], persistent=True)  # type: ignore[arg-type]
        object.__setattr__(
            self,
            "persistent_workspace_fallback_count",
            self.persistent_workspace_fallback_count + 1,
        )
        fallback_lease = self.workspace.acquire(reference, numel)
        return ElasticRankChunkWorkspaceLease(
            self,
            key,
            fallback_lease.tensor,
            persistent=False,
            fallback_lease=fallback_lease,
        )

    def release_persistent_rank_chunk_workspace(self, lease: ElasticRankChunkWorkspaceLease) -> None:
        entry = self._persistent_rank_chunk_workspaces.get(lease.key)
        if entry is None or entry["tensor"] is not lease.tensor:
            raise RuntimeError("Attempted to release a persistent elastic workspace that is not owned by this buffer.")
        if self.workspace.max_cached_per_key == 0:
            self._persistent_rank_chunk_workspaces.pop(lease.key, None)
            return
        entry["in_use"] = False

    def clear_idle_persistent_rank_chunk_workspaces(self) -> None:
        stale_keys = [
            key
            for key, entry in self._persistent_rank_chunk_workspaces.items()
            if not entry["in_use"]
        ]
        for key in stale_keys:
            self._persistent_rank_chunk_workspaces.pop(key, None)

    def persistent_workspace_stats(self) -> dict[str, object]:
        allocated_numel = 0
        in_use_numel = 0
        allocated_tensors = 0
        in_use_tensors = 0
        for entry in self._persistent_rank_chunk_workspaces.values():
            tensor = entry["tensor"]
            allocated_tensors += 1
            allocated_numel += tensor.numel()  # type: ignore[union-attr]
            if entry["in_use"]:
                in_use_tensors += 1
                in_use_numel += tensor.numel()  # type: ignore[union-attr]
        return {
            "persistent_workspace_acquire_count": self.persistent_workspace_acquire_count,
            "persistent_workspace_reuse_count": self.persistent_workspace_reuse_count,
            "persistent_workspace_allocate_count": self.persistent_workspace_allocate_count,
            "persistent_workspace_fallback_count": self.persistent_workspace_fallback_count,
            "persistent_workspace_allocated_tensors": allocated_tensors,
            "persistent_workspace_in_use_tensors": in_use_tensors,
            "persistent_workspace_allocated_numel": allocated_numel,
            "persistent_workspace_in_use_numel": in_use_numel,
        }

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
            "param_buffer_type": "elastic",
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
            **self.communication_plan.as_summary(),
            **self.workspace_plan(
                can_direct_all_gather=can_direct_all_gather,
                owner_segment_backend=owner_segment_backend,
                native_kernel_available=native_kernel_available,
                native_sendrecv_chunk_enabled=native_sendrecv_chunk_enabled,
            ).as_summary(),
            **self.workspace.stats(),
            **self.persistent_workspace_stats(),
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


def _rank_segments_from_shard_sizes(shard_sizes: tuple[int, ...]) -> tuple[tuple[LayoutSegment, ...], ...]:
    cursor = 0
    rank_segments: list[tuple[LayoutSegment, ...]] = []
    for shard_size in shard_sizes:
        if shard_size == 0:
            rank_segments.append(())
            continue
        rank_segments.append((LayoutSegment(cursor, cursor + shard_size, 0),))
        cursor += shard_size
    return tuple(rank_segments)


def _coalesce_rank_segments(
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
) -> tuple[tuple[LayoutSegment, ...], ...]:
    coalesced_ranks: list[tuple[LayoutSegment, ...]] = []
    for segments in rank_segments:
        if not segments:
            coalesced_ranks.append(())
            continue
        coalesced: list[LayoutSegment] = []
        for segment in segments:
            if (
                coalesced
                and coalesced[-1].global_end == segment.global_start
                and coalesced[-1].local_end == segment.local_start
            ):
                previous = coalesced[-1]
                coalesced[-1] = LayoutSegment(previous.global_start, segment.global_end, previous.local_start)
            else:
                coalesced.append(segment)
        coalesced_ranks.append(tuple(coalesced))
    return tuple(coalesced_ranks)


def _rank_chunk_shard_sizes(
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
) -> tuple[int, ...] | None:
    cursor = 0
    shard_sizes: list[int] = []
    for segments in rank_segments:
        if not segments:
            shard_sizes.append(0)
            continue
        if len(segments) != 1:
            return None
        segment = segments[0]
        if segment.local_start != 0 or segment.global_start != cursor:
            return None
        shard_sizes.append(segment.numel)
        cursor = segment.global_end
    return tuple(shard_sizes)


def _rank_chunk_segments_from_sizes(shard_sizes: tuple[int, ...]) -> tuple[tuple[LayoutSegment, ...], ...]:
    return _rank_segments_from_shard_sizes(shard_sizes)


def _rank_chunks_from_segments(
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
) -> tuple[ElasticRankChunk, ...]:
    chunks = []
    for rank, segments in enumerate(rank_segments):
        for segment in segments:
            chunks.append(
                ElasticRankChunk(
                    rank=rank,
                    global_start=segment.global_start,
                    global_end=segment.global_end,
                    local_start=segment.local_start,
                )
            )
    return tuple(chunks)
