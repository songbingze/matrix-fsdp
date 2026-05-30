from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import ceil

from matrix_fsdp.core.layout import MatrixGroupLayout
from matrix_fsdp.planning.layout_report import report_group_layout
from matrix_fsdp.planning.layout_validator import explain_runtime_layout_compatibility
from matrix_fsdp.core.managed_param import ManagedParam
from matrix_fsdp.planning.planner import (
    ParamBlock,
    expert_owner_tail_plan,
    hint_aware_block_builder,
    hinted_ordered_group_plan,
    load_balanced_matrix_owner_tail_plan,
    ordered_group_plan,
    ordered_matrix_owner_tail_plan,
    parameter_boundary_plan,
    rotate_layout_ranks,
    whole_param_blocks,
)
from matrix_fsdp.planning.planner_eval import (
    GroupPlannerCandidate,
    PlannerCandidate,
    PlannerCostWeights,
    PlannerEvaluation,
    PlannerResourceEstimate,
    PlannerResult,
    compare_group_planners,
    estimate_layout_cost_breakdown,
    estimate_layout_resources,
    planner_result_from_output,
    select_best_group_planner,
)


@dataclass(frozen=True)
class AutoGroupPlannerCandidates:
    candidates: tuple[PlannerCandidate, ...]

    @property
    def candidate_by_name(self) -> Mapping[str, PlannerCandidate]:
        return {candidate.name: candidate for candidate in self.candidates}

    @property
    def planners(self) -> Mapping[str, GroupPlannerCandidate]:
        return {candidate.name: candidate.planner for candidate in self.candidates}

    @property
    def blocks_by_name(self) -> Mapping[str, tuple[ParamBlock, ...]]:
        return {candidate.name: candidate.blocks for candidate in self.candidates}

    @property
    def padding_units_by_name(self) -> Mapping[str, int]:
        return {candidate.name: candidate.padding_units for candidate in self.candidates if candidate.padding_units}

    @property
    def num_collectives_by_name(self) -> Mapping[str, int]:
        return {
            candidate.name: candidate.num_collectives
            for candidate in self.candidates
            if candidate.num_collectives is not None
        }


@dataclass(frozen=True)
class AutoPlannerPolicy:
    name: str
    weights: PlannerCostWeights


@dataclass(frozen=True)
class AutoPlannerReportRow:
    policy: str
    candidate: str
    selected: bool
    cost: float
    rank_units: tuple[int, ...]
    params_by_rank: tuple[tuple[str, ...], ...]
    block_kinds: tuple[tuple[str, int], ...]
    cost_terms: tuple[tuple[str, float], ...]
    num_rank_segments: int
    max_rank_segments: int
    fragmented_rank_segments: int
    fragmented_rank_units: int
    num_split_params: int
    split_param_units: int
    split_param_segments: int
    total_comm_units: int
    max_rank_comm_units: int
    rank_memory_bytes: tuple[int, ...]
    rank_comm_bytes: tuple[int, ...]
    rank_muon_param_bytes: tuple[int, ...]
    rank_adamw_param_bytes: tuple[int, ...]
    rank_optimizer_bytes: tuple[int, ...]
    max_rank_memory_bytes: int
    memory_imbalance_bytes: int
    total_comm_bytes: int
    max_rank_comm_bytes: int
    muon_param_imbalance_bytes: int
    estimated_padding_units: int
    estimated_collectives: int
    candidate_padding_units: int
    candidate_collectives: int
    constraint_counts: tuple[tuple[str, int], ...]
    runtime_mode: str
    runtime_compatible: bool
    runtime_requires_flat_reorder: bool
    runtime_reason: str | None
    warnings: tuple[str, ...]
    planner_summary: dict[str, object]
    layout_contract: dict[str, object]
    report: dict[str, object]
    resources: dict[str, object] | None
    expert_owner_groups: tuple[dict[str, object], ...]
    rank_role_units: dict[str, tuple[int, ...]]


BALANCED_POLICY = AutoPlannerPolicy(
    name="balanced",
    weights=PlannerCostWeights(
        runtime_flat_reorder=0.01,
        collective=1.0,
        max_rank_unit=1.0,
        padding_unit=1.0,
        split_param=0.0,
        split_param_unit=0.001,
        rank_segment=0.25,
        max_rank_segment=0.25,
        fragmented_rank_segment=0.5,
        fragmented_rank_unit=0.001,
        split_param_segment=0.25,
        imbalance_unit=0.05,
        memory_imbalance_byte=0.00001,
        total_comm_byte=0.000001,
    ),
)
MIN_COMM_POLICY = AutoPlannerPolicy(
    name="min_comm",
    weights=PlannerCostWeights(
        collective=4.0,
        max_rank_unit=0.1,
        padding_unit=2.0,
        split_param=8.0,
        split_param_unit=0.01,
        rank_segment=4.0,
        max_rank_segment=2.0,
        fragmented_rank_segment=8.0,
        fragmented_rank_unit=0.01,
        param_segment=1.0,
        split_param_segment=2.0,
        max_param_segment=1.0,
        total_comm_unit=0.01,
        max_rank_comm_unit=0.02,
    ),
)
MAX_BALANCE_POLICY = AutoPlannerPolicy(
    name="max_balance",
    weights=PlannerCostWeights(
        collective=0.5,
        max_rank_unit=4.0,
        padding_unit=0.5,
        split_param=0.0,
        split_param_unit=0.0005,
        rank_segment=0.1,
        fragmented_rank_segment=0.25,
        fragmented_rank_unit=0.0005,
        imbalance_unit=1.0,
    ),
)
MUON_FULL_MATRIX_POLICY = AutoPlannerPolicy(
    name="muon_full_matrix",
    weights=PlannerCostWeights(
        collective=1.0,
        max_rank_unit=0.5,
        padding_unit=1.0,
        split_param=100000.0,
        split_param_unit=10.0,
        rank_segment=1.0,
        max_rank_segment=1.0,
        fragmented_rank_segment=2.0,
        param_segment=10.0,
        split_param_segment=20.0,
        max_param_segment=10.0,
    ),
)
ZERO_COPY_FRIENDLY_POLICY = AutoPlannerPolicy(
    name="zero_copy_friendly",
    weights=PlannerCostWeights(
        runtime_flat_reorder=256.0,
        runtime_segment_runtime=0.0,
        collective=1.0,
        max_rank_unit=1.0,
        padding_unit=1.0,
        split_param=0.0,
        split_param_unit=0.001,
        rank_segment=0.25,
        max_rank_segment=0.25,
        fragmented_rank_segment=0.5,
        fragmented_rank_unit=0.001,
        split_param_segment=0.25,
        total_comm_unit=0.001,
        max_rank_comm_unit=0.001,
        imbalance_unit=0.05,
    ),
)
MUON_ZERO_COPY_POLICY = AutoPlannerPolicy(
    name="muon_zero_copy",
    weights=PlannerCostWeights(
        runtime_flat_reorder=256.0,
        runtime_segment_runtime=0.0,
        collective=1.0,
        max_rank_unit=1.0,
        padding_unit=1.0,
        split_param=100000.0,
        split_param_unit=10.0,
        rank_segment=0.5,
        max_rank_segment=0.5,
        fragmented_rank_segment=1.0,
        fragmented_rank_unit=0.001,
        param_segment=10.0,
        split_param_segment=20.0,
        max_param_segment=10.0,
        total_comm_unit=0.001,
        max_rank_comm_unit=0.001,
        imbalance_unit=0.25,
        parameter_block=0.001,
        matrix_owner_matrix=0.0001,
        muon_matrix_owner=0.0001,
        matrix_owner_tail_param=0.0001,
        adamw_tail_param=0.0001,
        max_rank_memory_byte=0.00001,
        memory_imbalance_byte=0.00002,
        max_rank_muon_param_byte=0.00002,
        muon_param_imbalance_byte=0.00005,
        total_comm_byte=0.000001,
        max_rank_comm_byte=0.000002,
    ),
)
MUON_SHARD_AWARE_POLICY = AutoPlannerPolicy(
    name="muon_shard_aware",
    weights=PlannerCostWeights(
        collective=1.0,
        max_rank_unit=1.0,
        padding_unit=1.0,
        split_param=100000.0,
        split_param_unit=10.0,
        rank_segment=1.0,
        max_rank_segment=1.0,
        fragmented_rank_segment=2.0,
        fragmented_rank_unit=0.001,
        param_segment=10.0,
        split_param_segment=20.0,
        max_param_segment=10.0,
        imbalance_unit=0.5,
        parameter_block=0.001,
        matrix_owner_matrix=0.0001,
        matrix_owner_tail_param=0.0001,
        adamw_tail_param=0.0001,
        max_rank_memory_byte=0.00002,
        memory_imbalance_byte=0.00005,
        max_rank_optimizer_byte=0.00002,
        optimizer_imbalance_byte=0.00005,
        max_rank_muon_param_byte=0.00005,
        muon_param_imbalance_byte=0.0001,
        total_comm_byte=0.000001,
        max_rank_comm_byte=0.000002,
        comm_imbalance_byte=0.000001,
    ),
)
DEBUG_POLICY = AutoPlannerPolicy(
    name="debug",
    weights=PlannerCostWeights(
        collective=1.0,
        max_rank_unit=1.0,
        padding_unit=1.0,
        split_param=1.0,
        split_param_unit=0.01,
        rank_segment=1.0,
        max_rank_segment=1.0,
        fragmented_rank_segment=1.0,
        fragmented_rank_unit=0.01,
        param_segment=1.0,
        split_param_segment=1.0,
        max_param_segment=1.0,
        total_comm_unit=0.01,
        max_rank_comm_unit=0.01,
        imbalance_unit=0.1,
        block=0.01,
    ),
)
_POLICIES = {
    policy.name: policy
    for policy in (
        BALANCED_POLICY,
        MIN_COMM_POLICY,
        MAX_BALANCE_POLICY,
        MUON_FULL_MATRIX_POLICY,
        ZERO_COPY_FRIENDLY_POLICY,
        MUON_ZERO_COPY_POLICY,
        MUON_SHARD_AWARE_POLICY,
        DEBUG_POLICY,
    )
}

MuonShardAwareGroupPlanner = Callable[[Sequence[ManagedParam], int], PlannerResult]
PlannerCandidatesInput = Mapping[str, GroupPlannerCandidate | PlannerCandidate] | Sequence[PlannerCandidate]


def auto_group_plan(
    params: Sequence[ManagedParam],
    world_size: int,
    *,
    candidates: PlannerCandidatesInput | None = None,
    policy: str | AutoPlannerPolicy = "balanced",
    weights: PlannerCostWeights | None = None,
    target_block_units: int | None = None,
) -> PlannerEvaluation:
    return select_auto_group_plan(
        params,
        world_size,
        candidates=candidates,
        policy=policy,
        weights=weights,
        target_block_units=target_block_units,
    )


def select_auto_group_plan(
    params: Sequence[ManagedParam],
    world_size: int,
    *,
    candidates: PlannerCandidatesInput | None = None,
    policy: str | AutoPlannerPolicy = "balanced",
    weights: PlannerCostWeights | None = None,
    target_block_units: int | None = None,
) -> PlannerEvaluation:
    resolved_policy = resolve_auto_planner_policy(policy)
    resolved_weights = resolved_policy.weights if weights is None else weights
    if candidates is None:
        candidate_bundle = default_auto_group_planner_candidates(
            params,
            world_size,
            target_block_units=target_block_units,
            include_matrix_owner_tail=resolved_policy.name in {"muon_shard_aware", "muon_zero_copy"},
        )
        evaluation = select_best_group_planner(
            params,
            world_size,
            candidate_bundle.candidates,
            weights=resolved_weights,
        )
        return replace(evaluation, policy=resolved_policy.name)
    evaluation = select_best_group_planner(params, world_size, candidates, weights=resolved_weights)
    return replace(evaluation, policy=resolved_policy.name)


def evaluate_auto_group_planners(
    params: Sequence[ManagedParam],
    world_size: int,
    *,
    candidates: PlannerCandidatesInput | None = None,
    policy: str | AutoPlannerPolicy = "balanced",
    weights: PlannerCostWeights | None = None,
    target_block_units: int | None = None,
) -> tuple[PlannerEvaluation, ...]:
    resolved_policy = resolve_auto_planner_policy(policy)
    resolved_weights = resolved_policy.weights if weights is None else weights
    if candidates is None:
        candidate_bundle = default_auto_group_planner_candidates(
            params,
            world_size,
            target_block_units=target_block_units,
            include_matrix_owner_tail=resolved_policy.name in {"muon_shard_aware", "muon_zero_copy"},
        )
        evaluations = compare_group_planners(
            params,
            world_size,
            candidate_bundle.candidates,
            weights=resolved_weights,
        )
        return tuple(replace(evaluation, policy=resolved_policy.name) for evaluation in evaluations)
    evaluations = compare_group_planners(params, world_size, candidates, weights=resolved_weights)
    return tuple(replace(evaluation, policy=resolved_policy.name) for evaluation in evaluations)


def resolve_auto_planner_policy(policy: str | AutoPlannerPolicy) -> AutoPlannerPolicy:
    if isinstance(policy, AutoPlannerPolicy):
        return policy
    try:
        return _POLICIES[policy]
    except KeyError as exc:
        valid = ", ".join(sorted(_POLICIES))
        raise ValueError(f"Unknown auto planner policy={policy!r}. Valid policies: {valid}.") from exc


def available_auto_planner_policies() -> tuple[str, ...]:
    return tuple(sorted(_POLICIES))


def build_auto_planner_report(
    params: Sequence[ManagedParam],
    world_size: int,
    *,
    policy: str | AutoPlannerPolicy = "balanced",
    target_block_units: int | None = None,
    show_candidates: bool = True,
) -> tuple[AutoPlannerReportRow, ...]:
    evaluations = evaluate_auto_group_planners(
        params,
        world_size,
        policy=policy,
        target_block_units=target_block_units,
    )
    if not evaluations:
        return ()
    selected_name = evaluations[0].name
    selected_policy = evaluations[0].policy or (policy.name if isinstance(policy, AutoPlannerPolicy) else policy)
    rows = tuple(
        _auto_planner_report_row(selected_policy, evaluation, evaluation.name == selected_name)
        for evaluation in evaluations
    )
    return rows if show_candidates else rows[:1]


def format_auto_planner_report(rows: Sequence[AutoPlannerReportRow]) -> str:
    headers = (
        "policy",
        "candidate",
        "sel",
        "cost",
        "rank_units",
        "rank_segs",
        "max_seg",
        "frag_seg",
        "frag_units",
        "splits",
        "split_units",
        "split_seg",
        "blocks",
        "constraints",
        "runtime",
        "rank_mem",
        "rank_comm",
        "rank_muon",
        "rank_adamw",
        "expert_owners",
        "rank_roles",
        "cand_pad",
        "cand_coll",
        "warnings",
        "cost_terms",
        "params_by_rank",
    )
    table_rows = [
        (
            row.policy,
            row.candidate,
            "*" if row.selected else "",
            f"{row.cost:.3f}",
            _format_int_tuple(row.rank_units),
            str(row.num_rank_segments),
            str(row.max_rank_segments),
            str(row.fragmented_rank_segments),
            str(row.fragmented_rank_units),
            str(row.num_split_params),
            str(row.split_param_units),
            str(row.split_param_segments),
            _format_name_counts(row.block_kinds),
            _format_name_counts(row.constraint_counts),
            _format_runtime(row),
            _format_int_tuple(row.rank_memory_bytes),
            _format_int_tuple(row.rank_comm_bytes),
            _format_int_tuple(row.rank_muon_param_bytes),
            _format_int_tuple(row.rank_adamw_param_bytes),
            _format_expert_owner_groups(row.expert_owner_groups),
            _format_rank_role_units(row.rank_role_units),
            str(row.candidate_padding_units),
            str(row.candidate_collectives),
            _format_strings(row.warnings),
            _format_cost_terms(row.cost_terms),
            _format_params_by_rank(row.params_by_rank),
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(table_row, widths) for table_row in table_rows)
    return "\n".join(lines)


def make_muon_shard_aware_group_planner(
    *,
    matrix_owner_tail: bool = False,
    rotate_units: bool = True,
    rotation_strategy: str = "greedy_balance",
    owner_assignment: str = "rotate",
    policy: str | AutoPlannerPolicy = "muon_shard_aware",
) -> MuonShardAwareGroupPlanner:
    """
    Build a stateful group planner for Muon-style full-matrix ownership.

    The base layout keeps each matrix whole on one owner rank. ``rotate_units``
    shifts those owner ranks across successive FSDP param groups so repeated blocks do
    not all place the same tensor roles on the same physical ranks. The default
    rotation strategy greedily chooses the next rank offset from cumulative
    per-rank load. ``owner_assignment="role_greedy"`` instead assigns each
    matrix/tail role directly to the currently lightest rank.
    """

    valid_strategies = {"greedy_balance", "round_robin"}
    if rotation_strategy not in valid_strategies:
        valid = ", ".join(sorted(valid_strategies))
        raise ValueError(f"Unknown rotation_strategy={rotation_strategy!r}. Valid strategies: {valid}.")
    valid_owner_assignments = {"rotate", "role_greedy"}
    if owner_assignment not in valid_owner_assignments:
        valid = ", ".join(sorted(valid_owner_assignments))
        raise ValueError(f"Unknown owner_assignment={owner_assignment!r}. Valid assignments: {valid}.")

    unit_index = 0
    cumulative_rank_units: list[int] | None = None

    def group_planner(params: Sequence[ManagedParam], world_size: int) -> PlannerResult:
        nonlocal cumulative_rank_units, unit_index
        if cumulative_rank_units is None or len(cumulative_rank_units) != world_size:
            cumulative_rank_units = [0 for _ in range(world_size)]
        if owner_assignment == "role_greedy":
            evaluation = _load_balanced_matrix_owner_tail_evaluation(
                params,
                world_size,
                initial_rank_units=cumulative_rank_units,
                policy=policy,
            )
            unit_index += 1
            for rank, units in enumerate(evaluation.layout.shard_sizes):
                cumulative_rank_units[rank] += units
            return evaluation

        layout_or_evaluation: MatrixGroupLayout | PlannerResult
        if matrix_owner_tail:
            layout_or_evaluation = ordered_matrix_owner_tail_plan(params, world_size)
        else:
            layout_or_evaluation = auto_group_plan(params, world_size, policy=policy)

        layout = _layout_from_layout_or_evaluation(layout_or_evaluation)
        rank_offset = _select_rotation_offset(
            layout,
            cumulative_rank_units,
            unit_index=unit_index,
            rotate_units=rotate_units,
            rotation_strategy=rotation_strategy,
        )
        unit_index += 1
        rotated = (
            layout_or_evaluation
            if rank_offset == 0
            else _rotate_layout_or_evaluation(layout_or_evaluation, rank_offset, params, world_size)
        )
        rotated_layout = _layout_from_layout_or_evaluation(rotated)
        for rank, units in enumerate(rotated_layout.shard_sizes):
            cumulative_rank_units[rank] += units
        return rotated

    return group_planner


def _load_balanced_matrix_owner_tail_evaluation(
    params: Sequence[ManagedParam],
    world_size: int,
    *,
    initial_rank_units: Sequence[int],
    policy: str | AutoPlannerPolicy,
) -> PlannerEvaluation:
    resolved_policy = resolve_auto_planner_policy(policy)
    layout = load_balanced_matrix_owner_tail_plan(
        params,
        world_size,
        initial_rank_units=initial_rank_units,
        merge_adjacent_rank_segments=True,
    )
    return planner_result_from_output(
        "matrix_owner_tail_role_greedy",
        layout,
        params,
        world_size,
        blocks=_matrix_owner_tail_blocks(params),
        weights=resolved_policy.weights,
        policy=resolved_policy.name,
    )


def _select_rotation_offset(
    layout: MatrixGroupLayout,
    cumulative_rank_units: Sequence[int],
    *,
    unit_index: int,
    rotate_units: bool,
    rotation_strategy: str,
) -> int:
    world_size = layout.world_size
    if not rotate_units or world_size <= 0:
        return 0
    if rotation_strategy == "round_robin":
        return unit_index % world_size

    best: tuple[tuple[float, ...], int] | None = None
    for rank_offset in range(world_size):
        rotated_shard_sizes = _rotated_shard_sizes(layout.shard_sizes, rank_offset)
        next_rank_units = tuple(
            cumulative_rank_units[rank] + rotated_shard_sizes[rank]
            for rank in range(world_size)
        )
        avg_units = sum(next_rank_units) / world_size
        score = (
            float(max(next_rank_units) - min(next_rank_units)),
            float(max(next_rank_units)),
            sum((units - avg_units) ** 2 for units in next_rank_units),
            float(rank_offset),
        )
        if best is None or score < best[0]:
            best = (score, rank_offset)
    assert best is not None
    return best[1]


def _rotated_shard_sizes(shard_sizes: Sequence[int], rank_offset: int) -> tuple[int, ...]:
    world_size = len(shard_sizes)
    if world_size == 0:
        return ()
    rank_offset %= world_size
    return tuple(shard_sizes[(rank - rank_offset) % world_size] for rank in range(world_size))


def _layout_from_layout_or_evaluation(
    layout_or_evaluation: MatrixGroupLayout | PlannerResult,
) -> MatrixGroupLayout:
    if isinstance(layout_or_evaluation, MatrixGroupLayout):
        return layout_or_evaluation
    return layout_or_evaluation.layout


def _rotate_layout_or_evaluation(
    layout_or_evaluation: MatrixGroupLayout | PlannerResult,
    rank_offset: int,
    params: Sequence[ManagedParam],
    world_size: int,
) -> MatrixGroupLayout | PlannerResult:
    if isinstance(layout_or_evaluation, MatrixGroupLayout):
        return rotate_layout_ranks(layout_or_evaluation, rank_offset)

    rotated_layout = rotate_layout_ranks(layout_or_evaluation.layout, rank_offset)
    report = report_group_layout(rotated_layout)
    report = replace(
        report,
        num_blocks=layout_or_evaluation.report.num_blocks,
        blocks_by_kind=layout_or_evaluation.report.blocks_by_kind,
    )
    runtime_compatibility = explain_runtime_layout_compatibility(rotated_layout, params, world_size)
    resource_estimate = estimate_layout_resources(rotated_layout, params)
    warnings = layout_or_evaluation.warnings
    if not runtime_compatibility.compatible:
        runtime_warning = f"runtime unsupported: {runtime_compatibility.reason}"
        if runtime_warning not in warnings:
            warnings = (*warnings, runtime_warning)
    cost_breakdown = estimate_layout_cost_breakdown(
        report,
        weights=layout_or_evaluation.cost_weights,
        padding_units=layout_or_evaluation.padding_units,
        num_collectives=layout_or_evaluation.num_collectives,
        runtime_compatible=runtime_compatibility.compatible,
        runtime_mode=runtime_compatibility.mode,
        resources=resource_estimate,
    )
    return replace(
        layout_or_evaluation,
        layout=rotated_layout,
        report=report,
        cost=cost_breakdown.total,
        runtime_compatible=runtime_compatibility.compatible,
        runtime_mode=runtime_compatibility.mode,
        runtime_reason=runtime_compatibility.reason,
        runtime_requires_flat_reorder=runtime_compatibility.requires_flat_reorder,
        matrix_shard_compatible=runtime_compatibility.matrix_shard_compatible,
        matrix_shard_reason=runtime_compatibility.matrix_shard_reason,
        warnings=warnings,
        total_comm_units=report.total_comm_units,
        max_rank_comm_units=report.max_rank_comm_units,
        resource_estimate=resource_estimate,
        num_rank_segments=report.num_rank_segments,
        max_rank_segments=report.max_rank_segments,
        cost_breakdown=cost_breakdown,
    )


def default_auto_group_planner_candidates(
    params: Sequence[ManagedParam],
    world_size: int,
    *,
    target_block_units: int | None = None,
    include_matrix_owner_tail: bool = False,
) -> AutoGroupPlannerCandidates:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")

    if _has_expert_owner_params(params):
        return AutoGroupPlannerCandidates(
            candidates=(
                PlannerCandidate(
                    name="expert_owner_tail",
                    planner=expert_owner_tail_plan,
                    blocks=_expert_owner_tail_blocks(params),
                ),
            )
        )

    target_units = _target_block_units(params, world_size, target_block_units)
    def target_row_block_units(param: ManagedParam) -> int:
        if len(param.shape) != 2:
            return 1
        cols = tuple(param.shape)[1]
        return max(1, target_units // cols)

    hinted_mixed_builder = hint_aware_block_builder(
        default_granularity="parameter",
        target_block_units=target_units,
        row_block_units=target_row_block_units,
    )
    ordered_block_builder = hint_aware_block_builder(
        default_granularity="block",
        target_block_units=target_units,
        row_block_units=target_row_block_units,
        block_kind="ordered_block",
    )
    matrix_row_block_builder = hint_aware_block_builder(
        default_granularity="row_block",
        target_block_units=target_units,
        row_block_units=target_row_block_units,
    )

    candidates: list[PlannerCandidate] = [
        PlannerCandidate(
            name="whole_param",
            planner=parameter_boundary_plan,
            blocks=_blocks_for_params(params, whole_param_blocks),
        ),
        PlannerCandidate(
            name="ordered_block",
            planner=lambda planner_params, planner_world_size: ordered_group_plan(
                planner_params,
                planner_world_size,
                block_builder=ordered_block_builder,
            ),
            blocks=_blocks_for_params(params, ordered_block_builder),
        ),
        PlannerCandidate(
            name="matrix_row_block",
            planner=lambda planner_params, planner_world_size: ordered_group_plan(
                planner_params,
                planner_world_size,
                block_builder=matrix_row_block_builder,
            ),
            blocks=_blocks_for_params(params, matrix_row_block_builder),
        ),
    ]
    if include_matrix_owner_tail and _has_matrix_params(params):
        candidates.append(
            PlannerCandidate(
                name="matrix_owner_tail",
                planner=ordered_matrix_owner_tail_plan,
                blocks=_matrix_owner_tail_blocks(params),
            )
        )
        candidates.append(
            PlannerCandidate(
                name="matrix_owner_tail_role_greedy",
                planner=load_balanced_matrix_owner_tail_plan,
                blocks=_matrix_owner_tail_blocks(params),
            )
        )
    if _has_explicit_shard_hints(params):
        candidates.append(
            PlannerCandidate(
                name="hinted_mixed",
                planner=lambda planner_params, planner_world_size: hinted_ordered_group_plan(
                    planner_params,
                    planner_world_size,
                    default_granularity="parameter",
                    target_block_units=target_units,
                    row_block_units=target_row_block_units,
                ),
                blocks=_blocks_for_params(params, hinted_mixed_builder),
            )
        )
    return AutoGroupPlannerCandidates(candidates=tuple(candidates))


def _target_block_units(
    params: Sequence[ManagedParam],
    world_size: int,
    explicit_target: int | None,
) -> int:
    if explicit_target is not None:
        if explicit_target <= 0:
            raise ValueError(f"target_block_units must be positive, got {explicit_target}.")
        return explicit_target
    total_numel = sum(param.numel for param in params)
    if total_numel == 0:
        return 1
    return max(1, ceil(total_numel / max(1, world_size * 2)))


def _blocks_for_params(params: Sequence[ManagedParam], block_builder) -> tuple[ParamBlock, ...]:
    return tuple(block for param in params for block in block_builder(param))


def _auto_planner_report_row(policy: str, evaluation: PlannerEvaluation, selected: bool) -> AutoPlannerReportRow:
    report = evaluation.report
    resources = evaluation.resource_estimate
    if resources is None:
        resources = PlannerResourceEstimate.empty(report.world_size)
    metadata = evaluation.as_metadata()
    return AutoPlannerReportRow(
        policy=policy,
        candidate=evaluation.name,
        selected=selected,
        cost=evaluation.cost,
        rank_units=report.rank_units,
        params_by_rank=report.params_by_rank,
        block_kinds=tuple(sorted(report.blocks_by_kind.items())),
        cost_terms=tuple(sorted(evaluation.cost_breakdown.terms.items())),
        num_rank_segments=report.num_rank_segments,
        max_rank_segments=report.max_rank_segments,
        fragmented_rank_segments=report.fragmented_rank_segments,
        fragmented_rank_units=report.fragmented_rank_units,
        num_split_params=report.num_split_params,
        split_param_units=report.split_param_units,
        split_param_segments=report.split_param_segments,
        total_comm_units=report.total_comm_units,
        max_rank_comm_units=report.max_rank_comm_units,
        rank_memory_bytes=resources.rank_memory_bytes,
        rank_comm_bytes=resources.rank_comm_bytes,
        rank_muon_param_bytes=resources.rank_muon_param_bytes,
        rank_adamw_param_bytes=resources.rank_adamw_param_bytes,
        rank_optimizer_bytes=resources.rank_optimizer_bytes,
        max_rank_memory_bytes=resources.max_rank_memory_bytes,
        memory_imbalance_bytes=resources.memory_imbalance_bytes,
        total_comm_bytes=resources.total_comm_bytes,
        max_rank_comm_bytes=resources.max_rank_comm_bytes,
        muon_param_imbalance_bytes=resources.muon_param_imbalance_bytes,
        estimated_padding_units=report.estimated_padding_units,
        estimated_collectives=report.estimated_collectives,
        candidate_padding_units=evaluation.padding_units,
        candidate_collectives=evaluation.num_collectives,
        constraint_counts=tuple(evaluation.as_metadata()["constraint_counts"]),
        runtime_mode=evaluation.runtime_mode,
        runtime_compatible=evaluation.runtime_compatible,
        runtime_requires_flat_reorder=evaluation.runtime_requires_flat_reorder,
        runtime_reason=evaluation.runtime_reason,
        warnings=evaluation.warnings,
        planner_summary=evaluation.summary(),
        layout_contract=evaluation.layout_contract().as_metadata(),
        report=report.as_metadata(),
        resources=resources.as_metadata(),
        expert_owner_groups=metadata["expert_owner_groups"],  # type: ignore[arg-type]
        rank_role_units=metadata["rank_role_units"],  # type: ignore[arg-type]
    )


def _matrix_owner_tail_blocks(params: Sequence[ManagedParam]) -> tuple[ParamBlock, ...]:
    return tuple(
        ParamBlock(
            fqn=param.fqn,
            global_start=param.offset,
            global_end=param.end,
            block_index=0,
            kind=_matrix_owner_tail_block_kind(param),
        )
        for param in params
    )


def _matrix_owner_tail_block_kind(param: ManagedParam) -> str:
    if len(param.shape) == 2 and param.shard_hint.optimizer_type == "muon":
        return "muon_matrix_owner"
    if len(param.shape) == 2:
        return "matrix_owner_matrix"
    if param.shard_hint.optimizer_type == "adamw":
        return "adamw_tail_param"
    return "matrix_owner_tail_param"


def _expert_owner_tail_blocks(params: Sequence[ManagedParam]) -> tuple[ParamBlock, ...]:
    return tuple(
        ParamBlock(
            fqn=param.fqn,
            global_start=param.offset,
            global_end=param.end,
            block_index=0,
            kind=_expert_owner_tail_block_kind(param),
        )
        for param in params
    )


def _expert_owner_tail_block_kind(param: ManagedParam) -> str:
    if param.shard_hint.runtime_kind == "expert_owner" or param.shard_hint.expert_group_id:
        return "expert_owner_param"
    if param.shard_hint.parallel_role == "router":
        return "router_tail_param"
    if len(param.shape) == 2 and param.shard_hint.optimizer_type == "muon":
        return "dense_muon_tail_param"
    return "dense_tail_param"


def _has_matrix_params(params: Sequence[ManagedParam]) -> bool:
    return any(len(param.shape) == 2 for param in params)


def _has_expert_owner_params(params: Sequence[ManagedParam]) -> bool:
    return any(
        param.shard_hint.runtime_kind == "expert_owner" or param.shard_hint.expert_group_id
        for param in params
    )


def _format_int_tuple(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _format_expert_owner_groups(groups: Sequence[Mapping[str, object]]) -> str:
    if not groups:
        return "-"
    parts = []
    for group in groups:
        owner_rank = group.get("owner_rank")
        owner = f"r{owner_rank}" if owner_rank is not None else f"r{group.get('owner_ranks')}"
        parts.append(f"{group.get('expert_group_id')}->{owner}")
    return ";".join(parts)


def _format_rank_role_units(rank_role_units: Mapping[str, Sequence[int]]) -> str:
    non_zero_roles = {
        role: tuple(units)
        for role, units in rank_role_units.items()
        if any(unit != 0 for unit in units)
    }
    if not non_zero_roles:
        return "-"
    return ";".join(f"{role}={_format_int_tuple(units)}" for role, units in sorted(non_zero_roles.items()))


def _format_name_counts(items: Sequence[tuple[str, int]]) -> str:
    if not items:
        return "-"
    return ",".join(f"{name}={count}" for name, count in items)


def _format_strings(values: Sequence[str]) -> str:
    if not values:
        return "-"
    return "|".join(values)


def _format_runtime(row: AutoPlannerReportRow) -> str:
    status = row.runtime_mode
    if not row.runtime_compatible:
        status = f"{status}:unsupported"
    elif row.runtime_requires_flat_reorder:
        status = f"{status}:reorder"
    if row.runtime_reason and not row.runtime_compatible:
        return f"{status}({row.runtime_reason})"
    return status


def _format_cost_terms(items: Sequence[tuple[str, float]]) -> str:
    if not items:
        return "-"
    return ",".join(f"{name}={value:.3g}" for name, value in items)


def _format_params_by_rank(params_by_rank: Sequence[Sequence[str]]) -> str:
    if not params_by_rank:
        return "-"
    formatted_ranks = []
    for rank, params in enumerate(params_by_rank):
        formatted_ranks.append(f"r{rank}:{'|'.join(params) if params else '-'}")
    return ";".join(formatted_ranks)


def _format_table_row(values: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))


def _has_explicit_shard_hints(params: Sequence[ManagedParam]) -> bool:
    return any(
        param.shard_hint.optimizer_type is not None
        or param.shard_hint.split_granularity is not None
        or param.shard_hint.block_shape is not None
        or param.shard_hint.runtime_kind is not None
        or param.shard_hint.parallel_role is not None
        or param.shard_hint.expert_id is not None
        or param.shard_hint.expert_group_id is not None
        or param.shard_hint.owner_rank is not None
        or bool(param.shard_hint.owner_replica_ranks)
        for param in params
    )
