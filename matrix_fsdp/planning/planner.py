from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from matrix_fsdp.core.layout import (
    LayoutSegment,
    ParamLayout,
    ParamSegment,
    MatrixGroupLayout,
    ShardPlan,
    contiguous_shard_plan,
)
from matrix_fsdp.core.managed_param import ParamRuntimeKind, ParamShardHint


class PlannerParam(Protocol):
    fqn: str
    numel: int
    offset: int
    end: int
    shape: Sequence[int]
    shard_hint: ParamShardHint


BlockSizeFn = Callable[[PlannerParam], int]
BlockBuilder = Callable[[PlannerParam], Sequence["ParamBlock"]]
RowBlockFn = Callable[[PlannerParam], int]
ShardGranularity = str


@dataclass(frozen=True)
class ParamBlock:
    fqn: str
    global_start: int
    global_end: int
    block_index: int
    kind: str = "parameter"

    @property
    def numel(self) -> int:
        return self.global_end - self.global_start


def contiguous_even_plan(total_numel: int, world_size: int) -> ShardPlan:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    base, remainder = divmod(total_numel, world_size)
    shard_sizes = tuple(base + (1 if rank < remainder else 0) for rank in range(world_size))
    offsets: list[int] = []
    cursor = 0
    for shard_size in shard_sizes:
        offsets.append(cursor)
        cursor += shard_size
    return contiguous_shard_plan(total_numel=total_numel, shard_sizes=shard_sizes, shard_offsets=tuple(offsets))


def _ordered_params(
    params: Sequence[PlannerParam],
    order_policy: str = "default",
) -> list[PlannerParam]:
    if order_policy == "default":
        return list(params)
    if order_policy == "largest_first":
        return sorted(params, key=lambda param: (-param.numel, param.offset))
    if order_policy == "offset":
        return sorted(params, key=lambda param: param.offset)
    raise ValueError(f"Unknown order_policy={order_policy!r}.")


def parameter_boundary_plan(
    params: Sequence[PlannerParam],
    world_size: int,
    *,
    order_policy: str = "default",
) -> MatrixGroupLayout:
    """
    Build a group-local layout that assigns whole parameters to ranks.

    The planner consumes the ordered parameter list for one communication group.
    It does not try to optimize across the whole model; higher-level code should
    form groups first, then run this planner inside each group.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")

    rank_segments: list[list[LayoutSegment]] = [[] for _ in range(world_size)]
    rank_sizes = [0 for _ in range(world_size)]
    param_segments: dict[str, tuple[ParamSegment, ...]] = {}

    for param in _ordered_params(params, order_policy):
        rank = min(range(world_size), key=lambda candidate: (rank_sizes[candidate], candidate))
        segment = LayoutSegment(
            global_start=param.offset,
            global_end=param.end,
            local_start=rank_sizes[rank],
        )
        rank_segments[rank].append(segment)
        rank_sizes[rank] += param.numel
        param_segments[param.fqn] = (
            ParamSegment(
                fqn=param.fqn,
                rank=rank,
                global_start=param.offset,
                global_end=param.end,
                local_start=segment.local_start,
            ),
        )

    param_layouts = tuple(
        ParamLayout(
            fqn=param.fqn,
            global_start=param.offset,
            global_end=param.end,
            segments=param_segments[param.fqn],
        )
        for param in params
    )
    total_numel = sum(param.numel for param in params)
    return MatrixGroupLayout.from_rank_segments(
        total_numel=total_numel,
        rank_segments=tuple(tuple(segments) for segments in rank_segments),
        params=param_layouts,
    )


def ordered_matrix_owner_tail_plan(
    params: Sequence[PlannerParam],
    world_size: int,
) -> MatrixGroupLayout:
    """
    Assign full 2D matrices to owner ranks in module order and pack small tails.

    This is a Muon-oriented owner layout for groups whose large compute tensors
    are already exposed as separate parameters, e.g. q/k/v/proj/up/down. Every
    2D tensor stays whole on one rank, while non-2D tensors are packed onto the
    rank after the matrix owners. If there are more matrices than ranks, matrix
    owners wrap around and the tail is packed onto the last rank.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")

    matrix_params = [param for param in params if len(param.shape) == 2]
    tail_params = [param for param in params if len(param.shape) != 2]
    tail_rank = min(len(matrix_params), world_size - 1)

    assignments: list[tuple[PlannerParam, int]] = []
    for index, param in enumerate(matrix_params):
        assignments.append((param, index % world_size))
    for param in tail_params:
        assignments.append((param, tail_rank))

    return _whole_param_owner_layout(params, world_size, assignments)


def load_balanced_matrix_owner_tail_plan(
    params: Sequence[PlannerParam],
    world_size: int,
    *,
    initial_rank_units: Sequence[int] | None = None,
    merge_adjacent_rank_segments: bool = False,
) -> MatrixGroupLayout:
    """
    Assign full matrix/tail roles to the currently lightest owner ranks.

    Unlike ``ordered_matrix_owner_tail_plan`` plus rank rotation, this planner
    decides ownership per role. Each 2D matrix is one role, and all non-2D
    parameters are kept together as a tail role. The resulting layout still uses
    only whole-parameter segments, so it remains compatible with one
    rank-contiguous MatrixShard buffer per group.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if initial_rank_units is None:
        rank_units = [0 for _ in range(world_size)]
    else:
        if len(initial_rank_units) != world_size:
            raise ValueError(
                f"initial_rank_units length must match world_size={world_size}, got {len(initial_rank_units)}."
            )
        rank_units = list(initial_rank_units)

    matrix_params = [param for param in params if len(param.shape) == 2]
    tail_params = [param for param in params if len(param.shape) != 2]
    roles: list[tuple[int, int, tuple[PlannerParam, ...]]] = [
        (param.numel, index, (param,)) for index, param in enumerate(matrix_params)
    ]
    if tail_params:
        roles.append((sum(param.numel for param in tail_params), len(matrix_params), tuple(tail_params)))

    assignments: list[tuple[PlannerParam, int]] = []
    for role_units, _, role_params in sorted(roles, key=lambda role: (-role[0], role[1])):
        rank = min(range(world_size), key=lambda candidate: (rank_units[candidate], candidate))
        for param in role_params:
            assignments.append((param, rank))
        rank_units[rank] += role_units

    return _whole_param_owner_layout(
        params,
        world_size,
        assignments,
        merge_adjacent_rank_segments=merge_adjacent_rank_segments,
    )


def load_balanced_matrix_owner_tail_group_plans(
    param_groups: Sequence[Sequence[PlannerParam]],
    world_size: int,
    *,
    initial_rank_units: Sequence[int] | None = None,
    merge_adjacent_rank_segments: bool = True,
) -> tuple[MatrixGroupLayout, ...]:
    """
    Assign whole-matrix owners across multiple runtime groups.

    The returned layouts are still one layout per input group, so runtime
    materialization can stay at the transformer-block granularity. The owner
    rank decision is made over all roles in ``param_groups`` at once, which
    avoids an online planner over-fitting each group independently.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if initial_rank_units is None:
        rank_units = [0 for _ in range(world_size)]
    else:
        if len(initial_rank_units) != world_size:
            raise ValueError(
                f"initial_rank_units length must match world_size={world_size}, got {len(initial_rank_units)}."
            )
        rank_units = list(initial_rank_units)

    roles: list[tuple[int, int, int, tuple[PlannerParam, ...]]] = []
    for group_index, params in enumerate(param_groups):
        matrix_params = [param for param in params if len(param.shape) == 2]
        tail_params = [param for param in params if len(param.shape) != 2]
        for role_index, param in enumerate(matrix_params):
            roles.append((param.numel, group_index, role_index, (param,)))
        if tail_params:
            roles.append(
                (sum(param.numel for param in tail_params), group_index, len(matrix_params), tuple(tail_params))
            )

    assignments_by_group: list[list[tuple[PlannerParam, int]]] = [[] for _ in param_groups]
    for role_units, group_index, role_index, role_params in sorted(
        roles,
        key=lambda role: (-role[0], role[1], role[2]),
    ):
        fixed_rank = _fixed_owner_rank(role_params, world_size)
        rank = (
            fixed_rank
            if fixed_rank is not None
            else min(range(world_size), key=lambda candidate: (rank_units[candidate], candidate))
        )
        for param in role_params:
            assignments_by_group[group_index].append((param, rank))
        rank_units[rank] += role_units

    return tuple(
        _whole_param_owner_layout(
            params,
            world_size,
            assignments_by_group[group_index],
            merge_adjacent_rank_segments=merge_adjacent_rank_segments,
        )
        for group_index, params in enumerate(param_groups)
    )


def expert_owner_tail_plan(
    params: Sequence[PlannerParam],
    world_size: int,
    *,
    initial_rank_units: Sequence[int] | None = None,
    dense_order_policy: str = "largest_first",
    merge_adjacent_rank_segments: bool = True,
) -> MatrixGroupLayout:
    """
    Keep all parameters for each routed expert on one owner rank.

    Expert grouping comes from ``ParamShardHint.runtime_kind='expert_owner'``
    and ``expert_group_id``. Each expert group is assigned as one role, then
    dense/router/norm tail parameters are greedily used to fill the lightest
    ranks. The resulting layout still consists of whole-parameter segments, so
    runtime can flat-reorder it without requiring non-contiguous local storage.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if initial_rank_units is None:
        rank_units = [0 for _ in range(world_size)]
    else:
        if len(initial_rank_units) != world_size:
            raise ValueError(
                f"initial_rank_units length must match world_size={world_size}, got {len(initial_rank_units)}."
            )
        rank_units = list(initial_rank_units)

    expert_groups: dict[str, list[PlannerParam]] = {}
    dense_params: list[PlannerParam] = []
    for param in params:
        hint = _param_shard_hint(param)
        if hint.runtime_kind == ParamRuntimeKind.EXPERT_OWNER or hint.expert_group_id:
            group_id = hint.expert_group_id or param.fqn
            expert_groups.setdefault(group_id, []).append(param)
        else:
            dense_params.append(param)

    assignments: list[tuple[PlannerParam, int]] = []
    expert_roles = sorted(
        (
            (sum(param.numel for param in group_params), min(param.offset for param in group_params), tuple(group_params))
            for group_params in expert_groups.values()
        ),
        key=lambda role: (-role[0], role[1]),
    )
    for role_units, _, role_params in expert_roles:
        fixed_rank = _fixed_owner_rank(role_params, world_size)
        rank = fixed_rank if fixed_rank is not None else min(range(world_size), key=lambda candidate: (rank_units[candidate], candidate))
        for param in role_params:
            assignments.append((param, rank))
        rank_units[rank] += role_units

    for param in _ordered_params(dense_params, dense_order_policy):
        fixed_rank = _fixed_owner_rank((param,), world_size)
        rank = fixed_rank if fixed_rank is not None else min(range(world_size), key=lambda candidate: (rank_units[candidate], candidate))
        assignments.append((param, rank))
        rank_units[rank] += param.numel

    return _whole_param_owner_layout(
        params,
        world_size,
        assignments,
        merge_adjacent_rank_segments=merge_adjacent_rank_segments,
    )


def _whole_param_owner_layout(
    params: Sequence[PlannerParam],
    world_size: int,
    assignments: Sequence[tuple[PlannerParam, int]],
    *,
    merge_adjacent_rank_segments: bool = False,
) -> MatrixGroupLayout:
    rank_segments: list[list[LayoutSegment]] = [[] for _ in range(world_size)]
    rank_sizes = [0 for _ in range(world_size)]
    param_segments: dict[str, list[ParamSegment]] = {param.fqn: [] for param in params}

    for param, rank in assignments:
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Owner rank for parameter {param.fqn!r} must be in [0, {world_size}), got {rank}.")
        local_start = rank_sizes[rank]
        rank_segments[rank].append(LayoutSegment(param.offset, param.end, local_start))
        param_segments[param.fqn].append(
            ParamSegment(
                fqn=param.fqn,
                rank=rank,
                global_start=param.offset,
                global_end=param.end,
                local_start=local_start,
            )
        )
        rank_sizes[rank] += param.numel

    planned_rank_segments: tuple[tuple[LayoutSegment, ...], ...]
    if merge_adjacent_rank_segments:
        planned_rank_segments = tuple(_merge_adjacent_layout_segments(segments) for segments in rank_segments)
    else:
        planned_rank_segments = tuple(tuple(segments) for segments in rank_segments)

    return MatrixGroupLayout.from_rank_segments(
        total_numel=sum(param.numel for param in params),
        rank_segments=planned_rank_segments,
        params=tuple(
            ParamLayout(
                fqn=param.fqn,
                global_start=param.offset,
                global_end=param.end,
                segments=tuple(param_segments[param.fqn]),
            )
            for param in params
        ),
    )


def _merge_adjacent_layout_segments(segments: Sequence[LayoutSegment]) -> tuple[LayoutSegment, ...]:
    if not segments:
        return ()
    merged: list[LayoutSegment] = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        if previous.global_end == segment.global_start and previous.local_end == segment.local_start:
            merged[-1] = LayoutSegment(
                global_start=previous.global_start,
                global_end=segment.global_end,
                local_start=previous.local_start,
            )
            continue
        merged.append(segment)
    return tuple(merged)


def _fixed_owner_rank(params: Sequence[PlannerParam], world_size: int) -> int | None:
    owner_ranks = {
        _param_shard_hint(param).owner_rank
        for param in params
        if _param_shard_hint(param).owner_rank is not None
    }
    if not owner_ranks:
        return None
    if len(owner_ranks) != 1:
        fqns = ", ".join(param.fqn for param in params)
        raise ValueError(f"Conflicting owner_rank hints for params: {fqns}.")
    owner_rank = next(iter(owner_ranks))
    assert owner_rank is not None
    if owner_rank >= world_size:
        raise ValueError(f"owner_rank={owner_rank} must be in [0, {world_size}).")
    return owner_rank


def fsdp2_chunk_plan(
    params: Sequence[PlannerParam],
    world_size: int,
    *,
    order_policy: str = "default",
) -> MatrixGroupLayout:
    """
    Build an FSDP2-style per-parameter chunk layout for one communication group.

    Each parameter is split evenly across ranks, and each rank stores its local
    buffer as ``[p0_rank_chunk, p1_rank_chunk, ...]``. This matches
    ``torch._chunk_cat`` reduce-scatter copy-in. The first runtime version keeps
    this padding-free and requires every parameter numel to be divisible by
    ``world_size``.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")

    ordered = _ordered_params(params, order_policy)
    rank_segments: list[list[LayoutSegment]] = [[] for _ in range(world_size)]
    rank_sizes = [0 for _ in range(world_size)]
    param_segments: dict[str, list[ParamSegment]] = {param.fqn: [] for param in params}

    for param in ordered:
        if param.numel % world_size != 0:
            raise ValueError(
                f"fsdp2_chunk_plan requires parameter {param.fqn!r} numel={param.numel} "
                f"to be divisible by world_size={world_size}."
            )
        chunk_size = param.numel // world_size
        for rank in range(world_size):
            global_start = param.offset + rank * chunk_size
            global_end = global_start + chunk_size
            local_start = rank_sizes[rank]
            rank_segments[rank].append(LayoutSegment(global_start, global_end, local_start))
            param_segments[param.fqn].append(
                ParamSegment(
                    fqn=param.fqn,
                    rank=rank,
                    global_start=global_start,
                    global_end=global_end,
                    local_start=local_start,
                )
            )
            rank_sizes[rank] += chunk_size

    return MatrixGroupLayout.from_rank_segments(
        total_numel=sum(param.numel for param in params),
        rank_segments=tuple(tuple(segments) for segments in rank_segments),
        params=tuple(
            ParamLayout(
                fqn=param.fqn,
                global_start=param.offset,
                global_end=param.end,
                segments=tuple(param_segments[param.fqn]),
            )
            for param in params
        ),
    )


def ordered_group_plan(
    params: Sequence[PlannerParam],
    world_size: int,
    *,
    block_builder: BlockBuilder | None = None,
    block_size_fn: BlockSizeFn | None = None,
    order_policy: str = "default",
) -> MatrixGroupLayout:
    """
    Build a group-local layout by splitting an ordered tensor list at block boundaries.

    This is the first, correctness-oriented version of the paper-style group
    planner. It preserves the chosen tensor order and cuts the group buffer into
    rank-local contiguous intervals, but only at boundaries produced by
    ``block_builder`` or ``block_size_fn``. By default, each parameter is one
    block.
    """

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if block_builder is not None and block_size_fn is not None:
        raise ValueError("Pass only one of block_builder or block_size_fn.")
    ordered = _ordered_params(params, order_policy)
    total_numel = sum(param.numel for param in params)
    if total_numel == 0:
        return MatrixGroupLayout.from_rank_segments(total_numel=0, rank_segments=tuple(() for _ in range(world_size)))

    blocks = _build_ordered_blocks(ordered, block_builder or _block_builder_from_size_fn(block_size_fn))
    rank_block_groups = _split_blocks_across_ranks(blocks, total_numel, world_size)
    rank_segments = _rank_segments_from_block_groups(rank_block_groups)
    param_segments = _param_segments_from_rank_segments(params, rank_segments)
    return MatrixGroupLayout.from_rank_segments(
        total_numel=total_numel,
        rank_segments=rank_segments,
        params=tuple(
            ParamLayout(
                fqn=param.fqn,
                global_start=param.offset,
                global_end=param.end,
                segments=param_segments[param.fqn],
            )
            for param in params
        ),
    )


def hinted_ordered_group_plan(
    params: Sequence[PlannerParam],
    world_size: int,
    *,
    default_granularity: ShardGranularity = "parameter",
    target_block_units: int | None = None,
    row_block_units: int | RowBlockFn | None = None,
    block_kind: str = "block",
    order_policy: str = "default",
) -> MatrixGroupLayout:
    """
    Build a MatrixShard-compatible ordered layout from per-parameter shard hints.

    This is the planner-facing contract we want the runtime to consume long term:
    each parameter chooses its legal split boundaries through ``ParamShardHint``,
    while the final group layout remains rank-contiguous so it can be represented
    by one flat ``MatrixShard(local_units=...)`` placement.

    Supported granularities:
    - ``parameter`` / ``matrix_owner``: keep the whole parameter on one rank.
    - ``block``: split on 1D block boundaries.
    - ``row_block``: split 2D matrices on row-block boundaries.

    Explicit ``ParamShardHint.block_shape`` values are strict. Default block
    sizing is best-effort and allows a final smaller tail block.
    """

    block_builder = hint_aware_block_builder(
        default_granularity=default_granularity,
        target_block_units=target_block_units,
        row_block_units=row_block_units,
        block_kind=block_kind,
    )
    return ordered_group_plan(
        params,
        world_size,
        block_builder=block_builder,
        order_policy=order_policy,
    )


def rotate_layout_ranks(layout: MatrixGroupLayout, rank_offset: int) -> MatrixGroupLayout:
    """
    Rotate rank ownership in an existing group layout.

    This is useful when each unit keeps whole-matrix ownership internally but a
    sequence of similar units should not all choose the same physical owner
    ranks. The local order within each rotated rank is preserved.
    """

    world_size = layout.world_size
    if world_size <= 0:
        return layout
    rank_offset %= world_size
    if rank_offset == 0:
        return layout

    rank_segments: list[list[LayoutSegment]] = [[] for _ in range(world_size)]
    rank_local_cursors = [0 for _ in range(world_size)]
    param_segments_by_fqn: dict[str, list[ParamSegment]] = {param.fqn: [] for param in layout.params}

    for old_rank, segments in enumerate(layout.rank_segments):
        new_rank = (old_rank + rank_offset) % world_size
        for segment in segments:
            new_segment = LayoutSegment(
                global_start=segment.global_start,
                global_end=segment.global_end,
                local_start=rank_local_cursors[new_rank],
            )
            rank_segments[new_rank].append(new_segment)
            for param_segment in _param_segments_overlapping_layout_segment(layout.params, segment):
                local_start = new_segment.local_start + param_segment.global_start - segment.global_start
                param_segments_by_fqn[param_segment.fqn].append(
                    ParamSegment(
                        fqn=param_segment.fqn,
                        rank=new_rank,
                        global_start=param_segment.global_start,
                        global_end=param_segment.global_end,
                        local_start=local_start,
                    )
                )
            rank_local_cursors[new_rank] += segment.numel

    return MatrixGroupLayout.from_rank_segments(
        total_numel=layout.total_numel,
        rank_segments=tuple(tuple(segments) for segments in rank_segments),
        params=tuple(
            ParamLayout(
                fqn=param_layout.fqn,
                global_start=param_layout.global_start,
                global_end=param_layout.global_end,
                segments=tuple(
                    sorted(param_segments_by_fqn[param_layout.fqn], key=lambda segment: (segment.rank, segment.local_start))
                ),
            )
            for param_layout in layout.params
        ),
    )


def _param_segments_overlapping_layout_segment(
    param_layouts: Sequence[ParamLayout],
    layout_segment: LayoutSegment,
) -> tuple[ParamSegment, ...]:
    overlapping_segments: list[ParamSegment] = []
    for param_layout in param_layouts:
        for param_segment in param_layout.segments:
            if param_segment.global_start >= layout_segment.global_end:
                continue
            if param_segment.global_end <= layout_segment.global_start:
                continue
            overlapping_segments.append(param_segment)
    return tuple(overlapping_segments)


def hint_aware_block_builder(
    *,
    default_granularity: ShardGranularity = "parameter",
    target_block_units: int | None = None,
    row_block_units: int | RowBlockFn | None = None,
    block_kind: str = "block",
) -> BlockBuilder:
    valid_granularities = {"parameter", "matrix_owner", "block", "row_block"}
    if default_granularity not in valid_granularities:
        valid = ", ".join(sorted(valid_granularities))
        raise ValueError(f"Unknown default_granularity={default_granularity!r}. Valid granularities: {valid}.")
    if target_block_units is not None and target_block_units <= 0:
        raise ValueError(f"target_block_units must be positive, got {target_block_units}.")
    if isinstance(row_block_units, int) and row_block_units <= 0:
        raise ValueError(f"row_block_units must be positive, got {row_block_units}.")

    def build_blocks(param: PlannerParam) -> tuple[ParamBlock, ...]:
        hint = _param_shard_hint(param)
        granularity = hint.split_granularity or default_granularity
        if granularity in {"parameter", "matrix_owner"}:
            return whole_param_blocks(param)
        if granularity == "block":
            if hint.split_granularity == "block":
                return uniform_block_builder(
                    lambda candidate: _default_block_units(candidate, target_block_units),
                    kind=block_kind,
                )(param)
            return bounded_block_builder(
                lambda candidate: _default_block_units(candidate, target_block_units),
                kind=block_kind,
            )(param)
        if granularity == "row_block":
            if len(param.shape) != 2:
                if hint.split_granularity == "row_block":
                    return matrix_row_block_builder(lambda candidate: _default_row_block_units(candidate, row_block_units))(
                        param
                    )
                return whole_param_blocks(param)
            if hint.split_granularity == "row_block":
                return matrix_row_block_builder(lambda candidate: _default_row_block_units(candidate, row_block_units))(
                    param
                )
            return bounded_matrix_row_block_builder(
                lambda candidate: _default_row_block_units(candidate, row_block_units)
            )(param)
        raise ValueError(f"Unknown split_granularity={granularity!r} for parameter {param.fqn}.")

    return build_blocks


def whole_param_blocks(param: PlannerParam) -> tuple[ParamBlock, ...]:
    return (
        ParamBlock(
            fqn=param.fqn,
            global_start=param.offset,
            global_end=param.end,
            block_index=0,
            kind="parameter",
        ),
    )


def bounded_block_builder(
    block_size_fn: BlockSizeFn,
    *,
    kind: str = "block",
) -> BlockBuilder:
    def build_blocks(param: PlannerParam) -> tuple[ParamBlock, ...]:
        hint = _param_shard_hint(param)
        if _requires_whole_parameter(hint):
            return whole_param_blocks(param)
        if hint.split_granularity == "block":
            return uniform_block_builder(block_size_fn, kind=kind)(param)
        block_size = block_size_fn(param)
        if block_size <= 0:
            raise ValueError(f"block_size_fn returned {block_size} for {param.fqn}.")
        blocks = []
        cursor = param.offset
        block_index = 0
        while cursor < param.end:
            block_end = min(cursor + block_size, param.end)
            blocks.append(ParamBlock(param.fqn, cursor, block_end, block_index, kind))
            cursor = block_end
            block_index += 1
        return tuple(blocks)

    return build_blocks


def uniform_block_builder(
    block_size_fn: BlockSizeFn,
    *,
    kind: str = "uniform",
) -> BlockBuilder:
    def build_blocks(param: PlannerParam) -> tuple[ParamBlock, ...]:
        hint = _param_shard_hint(param)
        if _requires_whole_parameter(hint):
            return whole_param_blocks(param)
        block_size = _block_size_from_hint(hint) or block_size_fn(param)
        if block_size <= 0:
            raise ValueError(f"block_size_fn returned {block_size} for {param.fqn}.")
        if param.numel % block_size != 0:
            raise ValueError(f"Parameter {param.fqn} numel={param.numel} is not divisible by block_size={block_size}.")
        return tuple(
            ParamBlock(
                fqn=param.fqn,
                global_start=block_start,
                global_end=block_start + block_size,
                block_index=block_index,
                kind=kind,
            )
            for block_index, block_start in enumerate(range(param.offset, param.end, block_size))
        )

    return build_blocks


def bounded_matrix_row_block_builder(
    row_block_fn: RowBlockFn,
    *,
    kind: str = "matrix_row_block",
) -> BlockBuilder:
    def build_blocks(param: PlannerParam) -> tuple[ParamBlock, ...]:
        hint = _param_shard_hint(param)
        if _requires_whole_parameter(hint):
            return whole_param_blocks(param)
        if hint.split_granularity == "row_block":
            return matrix_row_block_builder(row_block_fn, kind=kind)(param)
        if len(param.shape) != 2:
            raise ValueError(f"Parameter {param.fqn} must be 2D for matrix row blocks, got shape={tuple(param.shape)}.")
        rows, cols = tuple(param.shape)
        row_block = row_block_fn(param)
        if row_block <= 0:
            raise ValueError(f"row_block_fn returned {row_block} for {param.fqn}.")
        return tuple(
            ParamBlock(
                fqn=param.fqn,
                global_start=param.offset + row_start * cols,
                global_end=param.offset + min(row_start + row_block, rows) * cols,
                block_index=block_index,
                kind=kind,
            )
            for block_index, row_start in enumerate(range(0, rows, row_block))
        )

    return build_blocks


def matrix_row_block_builder(
    row_block_fn: RowBlockFn,
    *,
    kind: str = "matrix_row_block",
) -> BlockBuilder:
    def build_blocks(param: PlannerParam) -> tuple[ParamBlock, ...]:
        hint = _param_shard_hint(param)
        if _requires_whole_parameter(hint):
            return whole_param_blocks(param)
        if len(param.shape) != 2:
            raise ValueError(f"Parameter {param.fqn} must be 2D for matrix row blocks, got shape={tuple(param.shape)}.")
        rows, cols = tuple(param.shape)
        row_block = _row_block_from_hint(hint) or row_block_fn(param)
        if row_block <= 0:
            raise ValueError(f"row_block_fn returned {row_block} for {param.fqn}.")
        if rows % row_block != 0:
            raise ValueError(f"Parameter {param.fqn} rows={rows} is not divisible by row_block={row_block}.")
        block_size = row_block * cols
        return tuple(
            ParamBlock(
                fqn=param.fqn,
                global_start=param.offset + row_start * cols,
                global_end=param.offset + (row_start + row_block) * cols,
                block_index=block_index,
                kind=kind,
            )
            for block_index, row_start in enumerate(range(0, rows, row_block))
        )

    return build_blocks


def _block_builder_from_size_fn(block_size_fn: BlockSizeFn | None) -> BlockBuilder:
    if block_size_fn is None:
        return whole_param_blocks
    return uniform_block_builder(block_size_fn)


def _param_shard_hint(param: PlannerParam) -> ParamShardHint:
    return getattr(param, "shard_hint", ParamShardHint())


def _requires_whole_parameter(hint: ParamShardHint) -> bool:
    return hint.split_granularity in {"parameter", "matrix_owner"}


def _block_size_from_hint(hint: ParamShardHint) -> int | None:
    if hint.block_shape is None or hint.split_granularity != "block":
        return None
    if len(hint.block_shape) != 1:
        raise ValueError(f"block split_granularity expects a 1D block_shape, got {hint.block_shape}.")
    return hint.block_shape[0]


def _row_block_from_hint(hint: ParamShardHint) -> int | None:
    if hint.block_shape is None or hint.split_granularity != "row_block":
        return None
    if len(hint.block_shape) == 1:
        return hint.block_shape[0]
    if len(hint.block_shape) == 2:
        # Full shape validation lives in ManagedParamRegistry; this helper only
        # needs the row count to build blocks.
        return hint.block_shape[0]
    raise ValueError(f"row_block split_granularity expects a 1D or 2D block_shape, got {hint.block_shape}.")


def _default_block_units(param: PlannerParam, explicit_target: int | None) -> int:
    return explicit_target or max(1, param.numel)


def _default_row_block_units(param: PlannerParam, explicit_target: int | RowBlockFn | None) -> int:
    if callable(explicit_target):
        return explicit_target(param)
    if explicit_target is not None:
        return explicit_target
    if len(param.shape) != 2:
        return 1
    return max(1, tuple(param.shape)[0])


def _build_ordered_blocks(
    params: Sequence[PlannerParam],
    block_builder: BlockBuilder,
) -> list[ParamBlock]:
    blocks = []
    for param in params:
        param_blocks = tuple(block_builder(param))
        validate_param_blocks(param, param_blocks)
        blocks.extend(param_blocks)
    return blocks


def validate_param_blocks(param: PlannerParam, blocks: Sequence[ParamBlock]) -> None:
    cursor = param.offset
    seen_indices = set()
    for block in sorted(blocks, key=lambda candidate: (candidate.global_start, candidate.global_end)):
        if block.fqn != param.fqn:
            raise ValueError(f"Block for {block.fqn} cannot be used for parameter {param.fqn}.")
        if block.block_index in seen_indices:
            raise ValueError(f"Parameter {param.fqn} has duplicate block_index={block.block_index}.")
        seen_indices.add(block.block_index)
        if block.global_start < param.offset or block.global_end > param.end:
            raise ValueError(f"Block {block} crosses parameter {param.fqn} boundary.")
        if block.global_start < cursor:
            raise ValueError(f"Parameter {param.fqn} blocks overlap at {block}.")
        if block.global_start > cursor:
            raise ValueError(f"Parameter {param.fqn} blocks have gap before {block}.")
        if block.global_start >= block.global_end:
            raise ValueError(f"Parameter {param.fqn} has empty block {block}.")
        cursor = block.global_end
    if cursor != param.end:
        raise ValueError(f"Parameter {param.fqn} block coverage ends at {cursor}, expected {param.end}.")


def _split_blocks_across_ranks(
    blocks: Sequence[ParamBlock],
    total_numel: int,
    world_size: int,
) -> list[list[ParamBlock]]:
    rank_block_groups: list[list[ParamBlock]] = []
    block_idx = 0
    assigned = 0
    for rank in range(world_size):
        remaining_ranks = world_size - rank
        if remaining_ranks == 1:
            rank_block_groups.append(list(blocks[block_idx:]))
            break
        target_end = round(total_numel * (rank + 1) / world_size)
        rank_blocks = []
        while block_idx < len(blocks):
            block = blocks[block_idx]
            block_len = block.numel
            if rank_blocks and abs((assigned + block_len) - target_end) > abs(assigned - target_end):
                break
            rank_blocks.append(block)
            assigned += block_len
            block_idx += 1
            if assigned >= target_end:
                break
        rank_block_groups.append(rank_blocks)
    while len(rank_block_groups) < world_size:
        rank_block_groups.append([])
    return rank_block_groups


def _rank_segments_from_block_groups(
    rank_block_groups: Sequence[Sequence[ParamBlock]],
) -> tuple[tuple[LayoutSegment, ...], ...]:
    rank_segments = []
    for blocks in rank_block_groups:
        segments = []
        local_start = 0
        for start, end in _coalesce_blocks(blocks):
            segments.append(LayoutSegment(start, end, local_start))
            local_start += end - start
        rank_segments.append(tuple(segments))
    return tuple(rank_segments)


def _coalesce_blocks(blocks: Sequence[ParamBlock]) -> list[tuple[int, int]]:
    block_ranges = [(block.global_start, block.global_end) if isinstance(block, ParamBlock) else block for block in blocks]
    blocks = block_ranges
    if not blocks:
        return []
    segments = []
    start, end = blocks[0]
    for block_start, block_end in blocks[1:]:
        if block_start == end:
            end = block_end
            continue
        segments.append((start, end))
        start, end = block_start, block_end
    segments.append((start, end))
    return segments


def _param_segments_from_rank_segments(
    params: Sequence[PlannerParam],
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
) -> dict[str, tuple[ParamSegment, ...]]:
    param_segments = {}
    for param in params:
        segments = []
        for rank, rank_layout_segments in enumerate(rank_segments):
            for rank_segment in rank_layout_segments:
                global_start = max(param.offset, rank_segment.global_start)
                global_end = min(param.end, rank_segment.global_end)
                if global_start >= global_end:
                    continue
                segments.append(
                    ParamSegment(
                        fqn=param.fqn,
                        rank=rank,
                        global_start=global_start,
                        global_end=global_end,
                        local_start=rank_segment.local_start + global_start - rank_segment.global_start,
                    )
                )
        param_segments[param.fqn] = tuple(segments)
    return param_segments
