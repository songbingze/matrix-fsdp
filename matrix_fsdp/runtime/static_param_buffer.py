from __future__ import annotations

from dataclasses import dataclass, field

from matrix_fsdp.core.layout import LayoutSegment
from matrix_fsdp.runtime.workspace_cache import CommWorkspaceCache


@dataclass(frozen=True)
class StaticParamBufferLayout:
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
            "owner_imbalance_ratio": 1.0,
        }


@dataclass(frozen=True)
class StaticParamBufferWorkspacePlan:
    full_param_numel: int
    padded_rank_chunks_numel: int
    padding_waste_numel: int
    rank_chunk_fast_path: bool
    packed_full_order: bool
    equal_collective_capable: bool
    padded_collective_capable: bool

    @property
    def preferred_workspace_kind(self) -> str:
        if self.equal_collective_capable or self.padded_collective_capable:
            return "padded_rank_chunks"
        return "matrix_all_gather"

    @property
    def preferred_workspace_numel(self) -> int:
        if self.preferred_workspace_kind == "padded_rank_chunks":
            return self.padded_rank_chunks_numel
        return self.full_param_numel

    @property
    def padding_waste_ratio(self) -> float:
        dense_numel = self.full_param_numel
        return self.padding_waste_numel / dense_numel if dense_numel else 0.0

    def as_summary(self) -> dict[str, object]:
        return {
            "workspace_full_param_numel": self.full_param_numel,
            "workspace_compact_rank_chunks_numel": self.full_param_numel,
            "workspace_padded_rank_chunks_numel": self.padded_rank_chunks_numel,
            "workspace_padding_waste_numel": self.padding_waste_numel,
            "workspace_padding_waste_ratio": self.padding_waste_ratio,
            "workspace_preferred_kind": self.preferred_workspace_kind,
            "workspace_preferred_numel": self.preferred_workspace_numel,
            "workspace_rank_chunk_fast_path": self.rank_chunk_fast_path,
            "workspace_packed_full_order": self.packed_full_order,
            "workspace_owner_segment_collectives": False,
            "workspace_native_group_broadcast_capable": False,
            "workspace_native_sendrecv_chunk_capable": False,
            "workspace_padded_all_gather_capable": self.padded_collective_capable,
            "workspace_compact_owner_reduce_scatter_capable": False,
        }


@dataclass(frozen=True)
class StaticParamBuffer:
    layout: StaticParamBufferLayout
    workspace: CommWorkspaceCache = field(default_factory=CommWorkspaceCache)

    def workspace_plan(self, *, can_direct_all_gather: bool) -> StaticParamBufferWorkspacePlan:
        return StaticParamBufferWorkspacePlan(
            full_param_numel=self.layout.total_numel,
            padded_rank_chunks_numel=self.layout.padded_numel,
            padding_waste_numel=self.layout.padding_waste_numel,
            rank_chunk_fast_path=self.layout.rank_chunk_fast_path,
            packed_full_order=self.layout.packed_rank_shards_are_full_tensor_order,
            equal_collective_capable=can_direct_all_gather,
            padded_collective_capable=True,
        )

    def communication_summary(
        self,
        *,
        param_gather_strategy: str,
        matrix_collective_backend: str,
        can_direct_all_gather: bool,
    ) -> dict[str, object]:
        return {
            "param_buffer_type": "static",
            "param_gather_strategy": param_gather_strategy,
            "matrix_collective_backend": matrix_collective_backend,
            "effective_param_gather_backend": self.effective_param_gather_backend(
                param_gather_strategy=param_gather_strategy,
                can_direct_all_gather=can_direct_all_gather,
            ),
            "owner_segment_collectives": False,
            "owner_segment_backend": None,
            "custom_allgatherv_policy": None,
            "resolved_custom_allgatherv_impl": None,
            "effective_grad_reduce_backend": "equal_reduce_scatter" if can_direct_all_gather else "padded_reduce_scatter",
            "resolved_custom_reduce_scatterv_impl": None,
            **self.layout.as_summary(),
            **self.workspace_plan(can_direct_all_gather=can_direct_all_gather).as_summary(),
            **self.workspace.stats(),
        }

    def effective_param_gather_backend(
        self,
        *,
        param_gather_strategy: str,
        can_direct_all_gather: bool,
    ) -> str:
        if self.layout.world_size == 1:
            return "single_rank_copy"
        if param_gather_strategy != "matrix_all_gather" and can_direct_all_gather:
            return "equal_all_gather"
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
