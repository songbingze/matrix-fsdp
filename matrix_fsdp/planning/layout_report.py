from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from matrix_fsdp.core.layout import MatrixGroupLayout
from matrix_fsdp.planning.planner import ParamBlock


@dataclass(frozen=True)
class MatrixGroupLayoutReport:
    total_numel: int
    world_size: int
    rank_units: tuple[int, ...]
    max_rank_units: int
    min_rank_units: int
    avg_rank_units: float
    imbalance_units: int
    imbalance_ratio: float
    non_empty_ranks: int
    num_params: int
    num_whole_params: int
    whole_param_fqns: tuple[str, ...]
    num_split_params: int
    split_param_fqns: tuple[str, ...]
    split_param_units: int
    num_blocks: int
    blocks_by_kind: dict[str, int]
    params_by_rank: tuple[tuple[str, ...], ...]
    rank_segment_counts: tuple[int, ...]
    num_rank_segments: int
    max_rank_segments: int
    fragmented_rank_segments: int
    fragmented_rank_units: int
    param_segment_counts: dict[str, int]
    num_param_segments: int
    split_param_segments: int
    max_param_segments: int
    total_comm_units: int
    max_rank_comm_units: int
    estimated_padding_units: int
    estimated_collectives: int

    def as_metadata(self) -> dict[str, object]:
        return {
            "total_numel": self.total_numel,
            "world_size": self.world_size,
            "rank_units": self.rank_units,
            "max_rank_units": self.max_rank_units,
            "min_rank_units": self.min_rank_units,
            "avg_rank_units": self.avg_rank_units,
            "imbalance_units": self.imbalance_units,
            "imbalance_ratio": self.imbalance_ratio,
            "non_empty_ranks": self.non_empty_ranks,
            "num_params": self.num_params,
            "num_whole_params": self.num_whole_params,
            "whole_param_fqns": self.whole_param_fqns,
            "num_split_params": self.num_split_params,
            "split_param_fqns": self.split_param_fqns,
            "split_param_units": self.split_param_units,
            "num_blocks": self.num_blocks,
            "blocks_by_kind": dict(self.blocks_by_kind),
            "params_by_rank": self.params_by_rank,
            "rank_segment_counts": self.rank_segment_counts,
            "num_rank_segments": self.num_rank_segments,
            "max_rank_segments": self.max_rank_segments,
            "fragmented_rank_segments": self.fragmented_rank_segments,
            "fragmented_rank_units": self.fragmented_rank_units,
            "param_segment_counts": dict(self.param_segment_counts),
            "num_param_segments": self.num_param_segments,
            "split_param_segments": self.split_param_segments,
            "max_param_segments": self.max_param_segments,
            "total_comm_units": self.total_comm_units,
            "max_rank_comm_units": self.max_rank_comm_units,
            "estimated_padding_units": self.estimated_padding_units,
            "estimated_collectives": self.estimated_collectives,
        }


def report_group_layout(
    layout: MatrixGroupLayout,
    blocks: Sequence[ParamBlock] = (),
    *,
    padding_alignment: int | None = None,
    collectives_per_segmented_exchange: int = 2,
) -> MatrixGroupLayoutReport:
    rank_units = layout.shard_sizes
    max_rank_units = max(rank_units, default=0)
    min_rank_units = min(rank_units, default=0)
    avg_rank_units = (sum(rank_units) / len(rank_units)) if rank_units else 0
    imbalance_units = max_rank_units - min_rank_units
    imbalance_ratio = (max_rank_units / avg_rank_units) if avg_rank_units else 0.0
    non_empty_ranks = sum(1 for units in rank_units if units > 0)
    whole_param_fqns = tuple(param.fqn for param in layout.params if len(param.segments) == 1)
    split_param_fqns = tuple(param.fqn for param in layout.params if len(param.segments) > 1)
    split_param_units = sum(param.numel for param in layout.params if len(param.segments) > 1)
    blocks_by_kind = dict(Counter(block.kind for block in blocks))
    rank_segment_counts = tuple(len(rank.segments) for rank in layout.ranks)
    num_rank_segments = sum(rank_segment_counts)
    max_rank_segments = max(rank_segment_counts, default=0)
    fragmented_rank_segments = sum(max(0, count - 1) for count in rank_segment_counts)
    fragmented_rank_units = sum(rank.local_units for rank in layout.ranks if len(rank.segments) > 1)
    param_segment_counts = {param.fqn: len(param.segments) for param in layout.params}
    num_param_segments = sum(param_segment_counts.values())
    split_param_segments = sum(max(0, count - 1) for count in param_segment_counts.values())
    max_param_segments = max(param_segment_counts.values(), default=0)
    estimated_padding_units = _estimate_padding_units(rank_units, padding_alignment)
    estimated_collectives = _estimate_collectives(num_rank_segments, collectives_per_segmented_exchange)
    return MatrixGroupLayoutReport(
        total_numel=layout.total_numel,
        world_size=layout.world_size,
        rank_units=rank_units,
        max_rank_units=max_rank_units,
        min_rank_units=min_rank_units,
        avg_rank_units=avg_rank_units,
        imbalance_units=imbalance_units,
        imbalance_ratio=imbalance_ratio,
        non_empty_ranks=non_empty_ranks,
        num_params=len(layout.params),
        num_whole_params=len(whole_param_fqns),
        whole_param_fqns=whole_param_fqns,
        num_split_params=len(split_param_fqns),
        split_param_fqns=split_param_fqns,
        split_param_units=split_param_units,
        num_blocks=len(blocks),
        blocks_by_kind=blocks_by_kind,
        params_by_rank=tuple(layout.params_for_rank(rank) for rank in range(layout.world_size)),
        rank_segment_counts=rank_segment_counts,
        num_rank_segments=num_rank_segments,
        max_rank_segments=max_rank_segments,
        fragmented_rank_segments=fragmented_rank_segments,
        fragmented_rank_units=fragmented_rank_units,
        param_segment_counts=param_segment_counts,
        num_param_segments=num_param_segments,
        split_param_segments=split_param_segments,
        max_param_segments=max_param_segments,
        total_comm_units=layout.total_numel * layout.world_size,
        max_rank_comm_units=max_rank_units * layout.world_size,
        estimated_padding_units=estimated_padding_units,
        estimated_collectives=estimated_collectives,
    )


def _estimate_padding_units(rank_units: Sequence[int], padding_alignment: int | None) -> int:
    if padding_alignment is None:
        return 0
    if padding_alignment <= 0:
        raise ValueError(f"padding_alignment must be positive, got {padding_alignment}.")
    padding = 0
    for units in rank_units:
        remainder = units % padding_alignment
        if remainder:
            padding += padding_alignment - remainder
    return padding


def _estimate_collectives(num_rank_segments: int, collectives_per_segmented_exchange: int) -> int:
    if collectives_per_segmented_exchange < 0:
        raise ValueError(
            f"collectives_per_segmented_exchange must be non-negative, got {collectives_per_segmented_exchange}."
        )
    if num_rank_segments == 0:
        return 0
    return collectives_per_segmented_exchange
