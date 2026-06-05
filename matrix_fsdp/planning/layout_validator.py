from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from matrix_fsdp.core.layout import LayoutSegment, ParamSegment, MatrixGroupLayout
from matrix_fsdp.core.managed_param import ManagedParam, ParamRuntimeKind
from matrix_fsdp.core.placement import explain_matrix_shard_compatibility


@dataclass(frozen=True)
class RuntimeLayoutCompatibility:
    compatible: bool
    mode: str
    reason: str | None = None
    requires_flat_reorder: bool = False
    matrix_shard_compatible: bool = False
    matrix_shard_reason: str | None = None


def validate_group_layout(
    layout: MatrixGroupLayout,
    params: Sequence[ManagedParam],
    world_size: int,
) -> None:
    _validate_group_shape(layout, params, world_size)
    _validate_rank_segments(layout)
    _validate_param_segments(layout, params)
    _validate_param_segments_match_rank_segments(layout)
    _validate_shard_hint_constraints(layout, params)


def explain_runtime_layout_compatibility(
    layout: MatrixGroupLayout,
    params: Sequence[ManagedParam],
    world_size: int,
    *,
    allow_flat_reorder: bool = True,
    allow_segment_runtime: bool = True,
) -> RuntimeLayoutCompatibility:
    validate_group_layout(layout, params, world_size)
    placement_compatibility = explain_matrix_shard_compatibility(layout)
    if placement_compatibility.compatible:
        return RuntimeLayoutCompatibility(
            compatible=True,
            mode="matrix_shard",
            matrix_shard_compatible=True,
        )

    if _can_use_segment_runtime_layout(layout, params):
        if not allow_segment_runtime:
            return RuntimeLayoutCompatibility(
                compatible=False,
                mode="unsupported",
                reason="layout requires segment_runtime, but segment runtime layouts are disabled",
                matrix_shard_compatible=False,
                matrix_shard_reason=placement_compatibility.reason,
            )
        return RuntimeLayoutCompatibility(
            compatible=True,
            mode="segment_runtime",
            reason=placement_compatibility.reason,
            matrix_shard_compatible=False,
            matrix_shard_reason=placement_compatibility.reason,
        )

    if (
        allow_flat_reorder
        and placement_compatibility.requires_flat_reorder
        and _can_flat_reorder_whole_param_layout(layout, params)
    ):
        return RuntimeLayoutCompatibility(
            compatible=True,
            mode="flat_reorder",
            reason=placement_compatibility.reason,
            requires_flat_reorder=True,
            matrix_shard_compatible=False,
            matrix_shard_reason=placement_compatibility.reason,
        )

    return RuntimeLayoutCompatibility(
        compatible=False,
        mode="unsupported",
        reason=_unsupported_runtime_layout_reason(layout, params, placement_compatibility.reason),
        requires_flat_reorder=placement_compatibility.requires_flat_reorder,
        matrix_shard_compatible=False,
        matrix_shard_reason=placement_compatibility.reason,
    )


def validate_runtime_layout(
    layout: MatrixGroupLayout,
    params: Sequence[ManagedParam],
    world_size: int,
    *,
    allow_flat_reorder: bool = True,
    allow_segment_runtime: bool = True,
) -> RuntimeLayoutCompatibility:
    compatibility = explain_runtime_layout_compatibility(
        layout,
        params,
        world_size,
        allow_flat_reorder=allow_flat_reorder,
        allow_segment_runtime=allow_segment_runtime,
    )
    if not compatibility.compatible:
        raise ValueError(
            "MatrixFSDP runtime cannot execute planner layout. "
            f"Reason: {compatibility.reason}."
        )
    return compatibility


def _can_use_segment_runtime_layout(
    layout: MatrixGroupLayout,
    params: Sequence[ManagedParam],
) -> bool:
    if not any(len(param_layout.segments) > 1 for param_layout in layout.params):
        return False
    if sum(layout.shard_sizes) != layout.total_numel:
        return False
    for rank, rank_layout in enumerate(layout.ranks):
        local_cursor = 0
        seen_fqns: set[str] = set()
        for segment in rank_layout.segments:
            if segment.local_start != local_cursor:
                return False
            overlapping_fqns = _managed_param_fqns_for_segment(segment, params)
            if len(overlapping_fqns) != 1:
                return False
            fqn = overlapping_fqns[0]
            if fqn in seen_fqns:
                return False
            seen_fqns.add(fqn)
            local_cursor = segment.local_end
        if local_cursor != rank_layout.local_units:
            return False
    return True


def _can_flat_reorder_whole_param_layout(
    layout: MatrixGroupLayout,
    params: Sequence[ManagedParam],
) -> bool:
    params_by_fqn = {param.fqn: param for param in params}
    for param_layout in layout.params:
        managed_param = params_by_fqn[param_layout.fqn]
        if len(param_layout.segments) != 1:
            return False
        segment = param_layout.segments[0]
        if segment.numel != managed_param.numel:
            return False
        if segment.global_start != managed_param.offset or segment.global_end != managed_param.end:
            return False
    return True


def _unsupported_runtime_layout_reason(
    layout: MatrixGroupLayout,
    params: Sequence[ManagedParam],
    placement_reason: str | None,
) -> str:
    for rank, rank_layout in enumerate(layout.ranks):
        seen_fqns: set[str] = set()
        for segment in rank_layout.segments:
            overlapping_fqns = _managed_param_fqns_for_segment(segment, params)
            if len(overlapping_fqns) != 1:
                return f"rank {rank} segment {segment} overlaps {len(overlapping_fqns)} parameters"
            fqn = overlapping_fqns[0]
            if fqn in seen_fqns:
                return f"rank {rank} owns multiple local segments of parameter {fqn!r}"
            seen_fqns.add(fqn)
    for param_layout in layout.params:
        if len(param_layout.segments) > 1:
            return f"parameter {param_layout.fqn!r} is split in a layout that cannot be exposed as local views"
    return placement_reason or "layout is not rank-contiguous and cannot be flat-reordered"


def _managed_param_fqns_for_segment(
    segment: LayoutSegment,
    params: Sequence[ManagedParam],
) -> tuple[str, ...]:
    fqns = []
    for param in params:
        if max(param.offset, segment.global_start) < min(param.end, segment.global_end):
            fqns.append(param.fqn)
    return tuple(fqns)


def _validate_group_shape(
    layout: MatrixGroupLayout,
    params: Sequence[ManagedParam],
    world_size: int,
) -> None:
    total_numel = sum(param.numel for param in params)
    if layout.total_numel != total_numel:
        raise ValueError(f"Group layout total_numel={layout.total_numel}, expected {total_numel}.")
    if layout.world_size != world_size:
        raise ValueError(f"Group layout has {layout.world_size} ranks, expected world_size={world_size}.")
    layout_fqns = tuple(param.fqn for param in layout.params)
    expected_fqns = tuple(param.fqn for param in params)
    if layout_fqns != expected_fqns:
        raise ValueError(f"Group layout params {layout_fqns} do not match managed params {expected_fqns}.")


def _validate_rank_segments(layout: MatrixGroupLayout) -> None:
    coverage: list[LayoutSegment] = []
    for rank, rank_layout in enumerate(layout.ranks):
        if rank_layout.rank != rank:
            raise ValueError(f"Rank layout at index {rank} has rank={rank_layout.rank}.")
        local_units = sum(segment.numel for segment in rank_layout.segments)
        if rank_layout.local_units != local_units:
            raise ValueError(
                f"Rank {rank} local_units={rank_layout.local_units}, expected {local_units} from segments."
            )
        local_cursor = 0
        for segment in rank_layout.segments:
            _validate_segment_range(segment.global_start, segment.global_end, layout.total_numel, f"rank {rank}")
            if segment.local_start != local_cursor:
                raise ValueError(
                    f"Rank {rank} segment {segment} has local_start={segment.local_start}, "
                    f"expected contiguous local_start={local_cursor}."
                )
            local_cursor = segment.local_end
            coverage.append(segment)
    _validate_global_coverage(
        [(segment.global_start, segment.global_end, f"rank segment {segment}") for segment in coverage],
        0,
        layout.total_numel,
        "rank segments",
    )


def _validate_param_segments(
    layout: MatrixGroupLayout,
    params: Sequence[ManagedParam],
) -> None:
    params_by_fqn = {param.fqn: param for param in params}
    for param_layout in layout.params:
        managed_param = params_by_fqn[param_layout.fqn]
        if param_layout.global_start != managed_param.offset or param_layout.global_end != managed_param.end:
            raise ValueError(
                f"Param layout {param_layout.fqn} range "
                f"[{param_layout.global_start}, {param_layout.global_end}) does not match managed range "
                f"[{managed_param.offset}, {managed_param.end})."
            )
        ranges = []
        for segment in param_layout.segments:
            if segment.fqn != param_layout.fqn:
                raise ValueError(f"Param {param_layout.fqn} contains segment for {segment.fqn}.")
            if segment.rank < 0 or segment.rank >= layout.world_size:
                raise ValueError(f"Param {param_layout.fqn} segment has invalid rank={segment.rank}.")
            _validate_segment_range(
                segment.global_start,
                segment.global_end,
                layout.total_numel,
                f"param {param_layout.fqn}",
            )
            if segment.global_start < managed_param.offset or segment.global_end > managed_param.end:
                raise ValueError(f"Param {param_layout.fqn} segment {segment} crosses parameter boundary.")
            rank_layout = layout.ranks[segment.rank]
            if segment.local_start < 0 or segment.local_end > rank_layout.local_units:
                raise ValueError(
                    f"Param {param_layout.fqn} segment {segment} is outside rank {segment.rank} local shard."
                )
            ranges.append((segment.global_start, segment.global_end, f"param segment {segment}"))
        _validate_global_coverage(ranges, managed_param.offset, managed_param.end, f"param {param_layout.fqn}")


def _validate_param_segments_match_rank_segments(layout: MatrixGroupLayout) -> None:
    for param_layout in layout.params:
        for segment in param_layout.segments:
            if not _rank_contains_param_segment(layout.ranks[segment.rank].segments, segment):
                raise ValueError(
                    f"Param {param_layout.fqn} segment {segment} does not project onto rank {segment.rank} segments."
                )


def _validate_shard_hint_constraints(
    layout: MatrixGroupLayout,
    params: Sequence[ManagedParam],
) -> None:
    layout_by_fqn = {param_layout.fqn: param_layout for param_layout in layout.params}
    expert_group_owner_ranks: dict[str, set[int]] = {}
    for managed_param in params:
        hint = managed_param.shard_hint
        param_layout = layout_by_fqn[managed_param.fqn]
        if hint.split_granularity in {"parameter", "matrix_owner"} and len(param_layout.segments) != 1:
            raise ValueError(
                f"Param {managed_param.fqn} split_granularity={hint.split_granularity!r} "
                "requires a single whole-parameter owner segment."
            )
        if hint.split_granularity == "block" and hint.block_shape is not None:
            _validate_segment_alignment(param_layout.segments, managed_param.offset, hint.block_shape[0], managed_param.fqn)
        if hint.split_granularity == "row_block" and hint.block_shape is not None:
            cols = tuple(managed_param.shape)[1]
            block_units = hint.block_shape[0] * cols
            _validate_segment_alignment(param_layout.segments, managed_param.offset, block_units, managed_param.fqn)
        if hint.runtime_kind == ParamRuntimeKind.EXPERT_OWNER:
            if len(param_layout.segments) != 1:
                raise ValueError(
                    f"Param {managed_param.fqn} runtime_kind='expert_owner' requires one whole owner segment."
                )
            expert_group_owner_ranks.setdefault(hint.expert_group_id or managed_param.fqn, set()).add(
                param_layout.segments[0].rank
            )
    for expert_group_id, owner_ranks in expert_group_owner_ranks.items():
        if len(owner_ranks) != 1:
            raise ValueError(
                f"Expert group {expert_group_id!r} must have one owner rank, got {tuple(sorted(owner_ranks))}."
            )


def _validate_segment_alignment(
    segments: tuple[ParamSegment, ...],
    param_offset: int,
    block_units: int,
    fqn: str,
) -> None:
    for segment in segments:
        relative_start = segment.global_start - param_offset
        relative_end = segment.global_end - param_offset
        if relative_start % block_units != 0 or relative_end % block_units != 0:
            raise ValueError(f"Param {fqn} segment {segment} is not aligned to block_units={block_units}.")


def _rank_contains_param_segment(
    rank_segments: tuple[LayoutSegment, ...],
    param_segment: ParamSegment,
) -> bool:
    for rank_segment in rank_segments:
        if param_segment.global_start < rank_segment.global_start or param_segment.global_end > rank_segment.global_end:
            continue
        expected_local_start = rank_segment.local_start + param_segment.global_start - rank_segment.global_start
        if param_segment.local_start == expected_local_start:
            return True
    return False


def _validate_segment_range(start: int, end: int, total_numel: int, context: str) -> None:
    if start < 0 or end < 0 or start >= end or end > total_numel:
        raise ValueError(f"Invalid {context} segment range [{start}, {end}) for total_numel={total_numel}.")


def _validate_global_coverage(
    ranges: list[tuple[int, int, str]],
    expected_start: int,
    expected_end: int,
    context: str,
) -> None:
    cursor = expected_start
    for start, end, label in sorted(ranges, key=lambda item: (item[0], item[1])):
        if start < cursor:
            raise ValueError(f"{context} overlap at {label}; expected next start >= {cursor}.")
        if start > cursor:
            raise ValueError(f"{context} gap before {label}; expected next start {cursor}.")
        cursor = end
    if cursor != expected_end:
        raise ValueError(f"{context} coverage ends at {cursor}, expected {expected_end}.")
