from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class LayoutParam(Protocol):
    fqn: str
    offset: int
    end: int
    numel: int


@dataclass(frozen=True)
class LayoutSegment:
    global_start: int
    global_end: int
    local_start: int

    @property
    def numel(self) -> int:
        return self.global_end - self.global_start

    @property
    def local_end(self) -> int:
        return self.local_start + self.numel

    def intersects(self, start: int, end: int) -> bool:
        return max(self.global_start, start) < min(self.global_end, end)


@dataclass(frozen=True)
class ParamSegment:
    fqn: str
    rank: int
    global_start: int
    global_end: int
    local_start: int

    @property
    def numel(self) -> int:
        return self.global_end - self.global_start

    @property
    def local_end(self) -> int:
        return self.local_start + self.numel

    def to_layout_segment(self) -> LayoutSegment:
        return LayoutSegment(
            global_start=self.global_start,
            global_end=self.global_end,
            local_start=self.local_start,
        )


@dataclass(frozen=True)
class ShardPlan:
    total_numel: int
    shard_sizes: tuple[int, ...]
    shard_offsets: tuple[int, ...]
    rank_segments: tuple[tuple[LayoutSegment, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.rank_segments is not None:
            return
        segments = []
        for offset, size in zip(self.shard_offsets, self.shard_sizes):
            segments.append((LayoutSegment(offset, offset + size, 0),))
        object.__setattr__(self, "rank_segments", tuple(segments))

    def local_range(self, rank: int) -> tuple[int, int]:
        start = self.shard_offsets[rank]
        return start, start + self.shard_sizes[rank]

    def local_segments(self, rank: int) -> tuple[LayoutSegment, ...]:
        assert self.rank_segments is not None
        return self.rank_segments[rank]


@dataclass(frozen=True)
class RankLayout:
    rank: int
    local_units: int
    segments: tuple[LayoutSegment, ...]


@dataclass(frozen=True)
class ParamLayout:
    fqn: str
    global_start: int
    global_end: int
    segments: tuple[ParamSegment, ...]

    @property
    def numel(self) -> int:
        return self.global_end - self.global_start


@dataclass(frozen=True)
class MatrixGroupLayout:
    """
    Layout for one planner/runtime communication group.

    The "global" offsets here are group-local flat-buffer offsets, not
    whole-model offsets. A higher-level scheduler may build many groups and run
    this layout contract independently for each group.
    """

    total_numel: int
    ranks: tuple[RankLayout, ...]
    params: tuple[ParamLayout, ...] = ()

    @classmethod
    def from_rank_segments(
        cls,
        total_numel: int,
        rank_segments: tuple[tuple[LayoutSegment, ...], ...],
        params: tuple[ParamLayout, ...] = (),
    ) -> "MatrixGroupLayout":
        ranks = tuple(
            RankLayout(
                rank=rank,
                local_units=sum(segment.numel for segment in segments),
                segments=segments,
            )
            for rank, segments in enumerate(rank_segments)
        )
        return cls(total_numel=total_numel, ranks=ranks, params=params)

    @classmethod
    def from_shard_plan(
        cls,
        plan: ShardPlan,
        params: Sequence[LayoutParam] = (),
    ) -> "MatrixGroupLayout":
        param_layouts = []
        for param in params:
            segments = []
            for rank, rank_segments in enumerate(plan.rank_segments or ()):
                for segment in rank_segments:
                    global_start = max(param.offset, segment.global_start)
                    global_end = min(param.end, segment.global_end)
                    if global_start >= global_end:
                        continue
                    segments.append(
                        ParamSegment(
                            fqn=param.fqn,
                            rank=rank,
                            global_start=global_start,
                            global_end=global_end,
                            local_start=segment.local_start + global_start - segment.global_start,
                        )
                    )
            param_layouts.append(
                ParamLayout(
                    fqn=param.fqn,
                    global_start=param.offset,
                    global_end=param.end,
                    segments=tuple(segments),
                )
            )
        return cls.from_rank_segments(
            total_numel=plan.total_numel,
            rank_segments=plan.rank_segments or (),
            params=tuple(param_layouts),
        )

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    @property
    def shard_sizes(self) -> tuple[int, ...]:
        return tuple(rank.local_units for rank in self.ranks)

    @property
    def rank_segments(self) -> tuple[tuple[LayoutSegment, ...], ...]:
        return tuple(rank.segments for rank in self.ranks)

    def param(self, fqn: str) -> ParamLayout:
        for param_layout in self.params:
            if param_layout.fqn == fqn:
                return param_layout
        raise KeyError(fqn)

    def owner_ranks(self, fqn: str) -> tuple[int, ...]:
        return tuple(sorted({segment.rank for segment in self.param(fqn).segments}))

    def param_segments_for_rank(self, rank: int) -> tuple[ParamSegment, ...]:
        return tuple(segment for param in self.params for segment in param.segments if segment.rank == rank)

    def rank_segments_for_param(self, rank: int, fqn: str) -> tuple[ParamSegment, ...]:
        return tuple(segment for segment in self.param(fqn).segments if segment.rank == rank)

    def params_for_rank(self, rank: int) -> tuple[str, ...]:
        return tuple(dict.fromkeys(segment.fqn for segment in self.param_segments_for_rank(rank)))

    def to_shard_plan(self) -> ShardPlan:
        shard_offsets = []
        cursor = 0
        for shard_size in self.shard_sizes:
            shard_offsets.append(cursor)
            cursor += shard_size
        return ShardPlan(
            total_numel=self.total_numel,
            shard_sizes=self.shard_sizes,
            shard_offsets=tuple(shard_offsets),
            rank_segments=self.rank_segments,
        )


def contiguous_shard_plan(
    total_numel: int,
    shard_sizes: tuple[int, ...],
    shard_offsets: tuple[int, ...],
) -> ShardPlan:
    segments = tuple(
        (LayoutSegment(offset, offset + size, 0),) for offset, size in zip(shard_offsets, shard_sizes)
    )
    return MatrixGroupLayout.from_rank_segments(total_numel, segments).to_shard_plan()


GlobalMatrixLayout = MatrixGroupLayout
