from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import torch

from matrix_fsdp.planning.constraints import (
    ParamShardConstraints,
    constraint_count_metadata,
    constraint_counts,
    constraints_metadata,
    infer_group_constraints,
)
from matrix_fsdp.core.layout import LayoutSegment, ParamLayout, ParamSegment, MatrixGroupLayout, ShardPlan
from matrix_fsdp.planning.layout_report import MatrixGroupLayoutReport, report_group_layout
from matrix_fsdp.planning.layout_validator import explain_runtime_layout_compatibility
from matrix_fsdp.core.managed_param import ManagedParam
from matrix_fsdp.planning.planner import ParamBlock

GroupPlannerCandidate = Callable[[Sequence[ManagedParam], int], "PlannerOutput"]


@dataclass(frozen=True)
class PlannerCostWeights:
    runtime_unsupported: float = 1_000_000_000_000.0
    runtime_flat_reorder: float = 0.0
    runtime_segment_runtime: float = 0.0
    collective: float = 1.0
    max_rank_unit: float = 1.0
    padding_unit: float = 1.0
    split_param: float = 0.0
    split_param_unit: float = 0.0
    rank_segment: float = 0.0
    max_rank_segment: float = 0.0
    fragmented_rank_segment: float = 0.0
    fragmented_rank_unit: float = 0.0
    param_segment: float = 0.0
    split_param_segment: float = 0.0
    max_param_segment: float = 0.0
    total_comm_unit: float = 0.0
    max_rank_comm_unit: float = 0.0
    imbalance_unit: float = 0.0
    imbalance_ratio: float = 0.0
    block: float = 0.0
    parameter_block: float = 0.0
    ordered_block: float = 0.0
    matrix_row_block: float = 0.0
    quant_block: float = 0.0
    matrix_owner_matrix: float = 0.0
    muon_matrix_owner: float = 0.0
    matrix_owner_tail_param: float = 0.0
    adamw_tail_param: float = 0.0
    max_rank_memory_byte: float = 0.0
    memory_imbalance_byte: float = 0.0
    max_rank_optimizer_byte: float = 0.0
    optimizer_imbalance_byte: float = 0.0
    max_rank_muon_param_byte: float = 0.0
    muon_param_imbalance_byte: float = 0.0
    total_comm_byte: float = 0.0
    max_rank_comm_byte: float = 0.0
    comm_imbalance_byte: float = 0.0
    workspace_preferred_unit: float = 0.0
    workspace_padding_unit: float = 0.0
    workspace_padding_ratio: float = 0.0


@dataclass(frozen=True)
class PlannerResourceEstimate:
    rank_param_bytes: tuple[int, ...]
    rank_grad_bytes: tuple[int, ...]
    rank_optimizer_bytes: tuple[int, ...]
    rank_memory_bytes: tuple[int, ...]
    rank_comm_bytes: tuple[int, ...]
    rank_muon_param_bytes: tuple[int, ...]
    rank_adamw_param_bytes: tuple[int, ...]
    rank_unknown_param_bytes: tuple[int, ...]
    total_param_bytes: int
    total_grad_bytes: int
    total_optimizer_bytes: int
    total_memory_bytes: int
    total_comm_bytes: int
    max_rank_memory_bytes: int
    memory_imbalance_bytes: int
    max_rank_optimizer_bytes: int
    optimizer_imbalance_bytes: int
    max_rank_muon_param_bytes: int
    muon_param_imbalance_bytes: int
    max_rank_comm_bytes: int
    comm_imbalance_bytes: int
    workspace_preferred_numel: int
    workspace_padding_waste_numel: int
    workspace_padding_waste_ratio: float

    @classmethod
    def empty(cls, world_size: int = 0) -> "PlannerResourceEstimate":
        rank_values = (0,) * world_size
        return cls(
            rank_param_bytes=rank_values,
            rank_grad_bytes=rank_values,
            rank_optimizer_bytes=rank_values,
            rank_memory_bytes=rank_values,
            rank_comm_bytes=rank_values,
            rank_muon_param_bytes=rank_values,
            rank_adamw_param_bytes=rank_values,
            rank_unknown_param_bytes=rank_values,
            total_param_bytes=0,
            total_grad_bytes=0,
            total_optimizer_bytes=0,
            total_memory_bytes=0,
            total_comm_bytes=0,
            max_rank_memory_bytes=0,
            memory_imbalance_bytes=0,
            max_rank_optimizer_bytes=0,
            optimizer_imbalance_bytes=0,
            max_rank_muon_param_bytes=0,
            muon_param_imbalance_bytes=0,
            max_rank_comm_bytes=0,
            comm_imbalance_bytes=0,
            workspace_preferred_numel=0,
            workspace_padding_waste_numel=0,
            workspace_padding_waste_ratio=0.0,
        )

    def as_metadata(self) -> dict[str, object]:
        return {
            "rank_param_bytes": self.rank_param_bytes,
            "rank_grad_bytes": self.rank_grad_bytes,
            "rank_optimizer_bytes": self.rank_optimizer_bytes,
            "rank_memory_bytes": self.rank_memory_bytes,
            "rank_comm_bytes": self.rank_comm_bytes,
            "rank_muon_param_bytes": self.rank_muon_param_bytes,
            "rank_adamw_param_bytes": self.rank_adamw_param_bytes,
            "rank_unknown_param_bytes": self.rank_unknown_param_bytes,
            "total_param_bytes": self.total_param_bytes,
            "total_grad_bytes": self.total_grad_bytes,
            "total_optimizer_bytes": self.total_optimizer_bytes,
            "total_memory_bytes": self.total_memory_bytes,
            "total_comm_bytes": self.total_comm_bytes,
            "max_rank_memory_bytes": self.max_rank_memory_bytes,
            "memory_imbalance_bytes": self.memory_imbalance_bytes,
            "max_rank_optimizer_bytes": self.max_rank_optimizer_bytes,
            "optimizer_imbalance_bytes": self.optimizer_imbalance_bytes,
            "max_rank_muon_param_bytes": self.max_rank_muon_param_bytes,
            "muon_param_imbalance_bytes": self.muon_param_imbalance_bytes,
            "max_rank_comm_bytes": self.max_rank_comm_bytes,
            "comm_imbalance_bytes": self.comm_imbalance_bytes,
            "workspace_preferred_numel": self.workspace_preferred_numel,
            "workspace_padding_waste_numel": self.workspace_padding_waste_numel,
            "workspace_padding_waste_ratio": self.workspace_padding_waste_ratio,
        }


@dataclass(frozen=True)
class PlannerCostBreakdown:
    terms: Mapping[str, float]

    @property
    def total(self) -> float:
        return sum(self.terms.values())


@dataclass(frozen=True)
class PlannerLayoutContract:
    """
    Stable planner/runtime view of a group layout.

    The contract is the boundary between planner output and runtime setup. It
    keeps the full layout object available for low-level adapters while exposing
    the small set of rank/parameter ownership queries that the FSDP param group needs.
    """

    layout: MatrixGroupLayout

    @property
    def total_numel(self) -> int:
        return self.layout.total_numel

    @property
    def world_size(self) -> int:
        return self.layout.world_size

    @property
    def shard_sizes(self) -> tuple[int, ...]:
        return self.layout.shard_sizes

    @property
    def rank_segments(self) -> tuple[tuple[LayoutSegment, ...], ...]:
        return self.layout.rank_segments

    @property
    def params(self) -> tuple[ParamLayout, ...]:
        return self.layout.params

    def to_shard_plan(self) -> ShardPlan:
        return self.layout.to_shard_plan()

    def params_for_rank(self, rank: int) -> tuple[str, ...]:
        return self.layout.params_for_rank(rank)

    def owner_ranks(self, fqn: str) -> tuple[int, ...]:
        return self.layout.owner_ranks(fqn)

    def rank_segments_for_param(self, rank: int, fqn: str) -> tuple[ParamSegment, ...]:
        return self.layout.rank_segments_for_param(rank, fqn)

    def as_metadata(self) -> dict[str, object]:
        return {
            "total_numel": self.layout.total_numel,
            "world_size": self.layout.world_size,
            "rank_units": self.layout.shard_sizes,
            "ranks": tuple(_rank_layout_metadata(rank_layout) for rank_layout in self.layout.ranks),
            "params": tuple(_param_layout_contract_metadata(param_layout) for param_layout in self.layout.params),
            "params_by_rank": tuple(self.layout.params_for_rank(rank) for rank in range(self.layout.world_size)),
        }

    def validate(self) -> None:
        _validate_layout_contract(self.layout)


@dataclass(frozen=True)
class PlannerResult:
    name: str
    layout: MatrixGroupLayout
    report: MatrixGroupLayoutReport
    cost: float
    constraints_satisfied: bool = True
    constraints: tuple[ParamShardConstraints, ...] = ()
    warnings: tuple[str, ...] = ()
    runtime_compatible: bool = True
    runtime_mode: str = "unknown"
    runtime_reason: str | None = None
    runtime_requires_flat_reorder: bool = False
    matrix_shard_compatible: bool = False
    matrix_shard_reason: str | None = None
    shard_mesh_dim: int | str | None = None
    padding_units: int = 0
    num_collectives: int = 0
    total_comm_units: int = 0
    max_rank_comm_units: int = 0
    resource_estimate: PlannerResourceEstimate | None = None
    num_rank_segments: int = 0
    max_rank_segments: int = 0
    policy: str | None = None
    cost_weights: PlannerCostWeights = PlannerCostWeights()
    cost_breakdown: PlannerCostBreakdown = field(default_factory=lambda: PlannerCostBreakdown({}))

    @property
    def planner_name(self) -> str:
        return self.name

    @property
    def world_size(self) -> int:
        return self.layout.world_size

    def layout_contract(self) -> PlannerLayoutContract:
        return PlannerLayoutContract(self.layout)

    def validate(self) -> None:
        self.layout_contract().validate()
        if self.report.total_numel != self.layout.total_numel:
            raise ValueError(
                f"Planner report total_numel={self.report.total_numel} does not match layout "
                f"{self.layout.total_numel}."
            )
        if self.report.world_size != self.layout.world_size:
            raise ValueError(
                f"Planner report world_size={self.report.world_size} does not match layout "
                f"{self.layout.world_size}."
            )
        if self.report.rank_units != self.layout.shard_sizes:
            raise ValueError(
                f"Planner report rank_units={self.report.rank_units} do not match layout "
                f"{self.layout.shard_sizes}."
            )
        if self.resource_estimate is not None and len(self.resource_estimate.rank_memory_bytes) != self.world_size:
            raise ValueError(
                f"Planner resource estimate has {len(self.resource_estimate.rank_memory_bytes)} ranks, "
                f"expected {self.world_size}."
            )

    def summary(self) -> dict[str, object]:
        expert_owner_groups = _expert_owner_groups_metadata(self.constraints, self.layout)
        rank_role_units = _rank_role_units_metadata(self.constraints, self.layout)
        return {
            "planner_name": self.planner_name,
            "policy": self.policy,
            "cost": self.cost,
            "world_size": self.world_size,
            "constraints_satisfied": self.constraints_satisfied,
            "runtime": {
                "compatible": self.runtime_compatible,
                "mode": self.runtime_mode,
                "reason": self.runtime_reason,
                "requires_flat_reorder": self.runtime_requires_flat_reorder,
                "matrix_shard_compatible": self.matrix_shard_compatible,
                "matrix_shard_reason": self.matrix_shard_reason,
            },
            "layout": self.layout_contract().as_metadata(),
            "report": self.report.as_metadata(),
            "resources": self.resource_estimate.as_metadata() if self.resource_estimate is not None else None,
            "expert_owner_groups": expert_owner_groups,
            "rank_role_units": rank_role_units,
            "cost_terms": tuple(self.cost_breakdown.terms.items()),
            "warnings": self.warnings,
        }

    def as_metadata(self) -> dict[str, object]:
        expert_owner_groups = _expert_owner_groups_metadata(self.constraints, self.layout)
        rank_role_units = _rank_role_units_metadata(self.constraints, self.layout)
        return {
            "planner_name": self.planner_name,
            "policy": self.policy,
            "cost": self.cost,
            "constraints_satisfied": self.constraints_satisfied,
            "constraints": constraints_metadata(self.constraints),
            "constraint_counts": constraint_count_metadata(constraint_counts(self.constraints)),
            "warnings": self.warnings,
            "runtime_compatible": self.runtime_compatible,
            "runtime_mode": self.runtime_mode,
            "runtime_reason": self.runtime_reason,
            "runtime_requires_flat_reorder": self.runtime_requires_flat_reorder,
            "matrix_shard_compatible": self.matrix_shard_compatible,
            "matrix_shard_reason": self.matrix_shard_reason,
            "world_size": self.world_size,
            "shard_mesh_dim": self.shard_mesh_dim,
            "rank_units": self.layout.shard_sizes,
            "layout_contract": self.layout_contract().as_metadata(),
            "report": self.report.as_metadata(),
            "cost_terms": tuple(self.cost_breakdown.terms.items()),
            "resource_estimate": self.resource_estimate.as_metadata() if self.resource_estimate is not None else None,
            "expert_owner_groups": expert_owner_groups,
            "rank_role_units": rank_role_units,
        }


PlannerEvaluation = PlannerResult
PlannerOutput = MatrixGroupLayout | ShardPlan | PlannerResult
GroupPlanner = Callable[[Sequence[ManagedParam], int], PlannerOutput]


def _rank_layout_metadata(rank_layout) -> dict[str, object]:
    return {
        "rank": rank_layout.rank,
        "local_units": rank_layout.local_units,
        "segments": tuple(_layout_segment_metadata(segment) for segment in rank_layout.segments),
    }


def _param_layout_contract_metadata(param_layout) -> dict[str, object]:
    owner_ranks = tuple(sorted({segment.rank for segment in param_layout.segments}))
    return {
        "fqn": param_layout.fqn,
        "global_start": param_layout.global_start,
        "global_end": param_layout.global_end,
        "numel": param_layout.numel,
        "owner_ranks": owner_ranks,
        "segments": tuple(_param_segment_metadata(segment) for segment in param_layout.segments),
    }


def _expert_owner_groups_metadata(
    constraints: tuple[ParamShardConstraints, ...],
    layout: MatrixGroupLayout,
) -> tuple[dict[str, object], ...]:
    constraints_by_fqn = {constraint.fqn: constraint for constraint in constraints}
    groups: dict[str, dict[str, object]] = {}
    for param_layout in layout.params:
        constraint = constraints_by_fqn.get(param_layout.fqn)
        if constraint is None or not constraint.expert_group_id:
            continue
        group = groups.setdefault(
            constraint.expert_group_id,
            {
                "expert_group_id": constraint.expert_group_id,
                "expert_id": _expert_id_from_group_id(constraint.expert_group_id),
                "param_fqns": [],
                "owner_ranks": set(),
                "numel": 0,
            },
        )
        group["param_fqns"].append(param_layout.fqn)  # type: ignore[union-attr]
        group["owner_ranks"].update(segment.rank for segment in param_layout.segments)  # type: ignore[union-attr]
        group["numel"] = int(group["numel"]) + param_layout.numel

    metadata = []
    for group_id in sorted(groups):
        group = groups[group_id]
        owner_ranks = tuple(sorted(group["owner_ranks"]))  # type: ignore[arg-type]
        metadata.append(
            {
                "expert_group_id": group["expert_group_id"],
                "expert_id": group["expert_id"],
                "owner_rank": owner_ranks[0] if len(owner_ranks) == 1 else None,
                "owner_ranks": owner_ranks,
                "param_fqns": tuple(group["param_fqns"]),  # type: ignore[arg-type]
                "numel": group["numel"],
            }
        )
    return tuple(metadata)


def _rank_role_units_metadata(
    constraints: tuple[ParamShardConstraints, ...],
    layout: MatrixGroupLayout,
) -> dict[str, tuple[int, ...]]:
    constraints_by_fqn = {constraint.fqn: constraint for constraint in constraints}
    role_units = {
        "expert": [0 for _ in range(layout.world_size)],
        "router": [0 for _ in range(layout.world_size)],
        "norm": [0 for _ in range(layout.world_size)],
        "dense": [0 for _ in range(layout.world_size)],
    }
    for param_layout in layout.params:
        constraint = constraints_by_fqn.get(param_layout.fqn)
        role = _metadata_role_for_param(param_layout.fqn, constraint)
        for segment in param_layout.segments:
            role_units[role][segment.rank] += segment.numel
    return {role: tuple(units) for role, units in role_units.items()}


def _metadata_role_for_param(fqn: str, constraint: ParamShardConstraints | None) -> str:
    if constraint is not None:
        if constraint.expert_group_id or constraint.runtime_kind == "expert_owner":
            return "expert"
        if constraint.parallel_role in {"router", "norm"}:
            return constraint.parallel_role
    parts = fqn.split(".")
    if any(part in {"router", "gate"} for part in parts):
        return "router"
    if any("norm" in part.lower() for part in parts):
        return "norm"
    return "dense"


def _expert_id_from_group_id(expert_group_id: str) -> int | None:
    tail = expert_group_id.rsplit(".", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _layout_segment_metadata(segment) -> dict[str, int]:
    return {
        "global_start": segment.global_start,
        "global_end": segment.global_end,
        "local_start": segment.local_start,
        "local_end": segment.local_end,
        "numel": segment.numel,
    }


def _param_segment_metadata(segment) -> dict[str, object]:
    return {
        "fqn": segment.fqn,
        "rank": segment.rank,
        "global_start": segment.global_start,
        "global_end": segment.global_end,
        "local_start": segment.local_start,
        "local_end": segment.local_end,
        "numel": segment.numel,
    }


def _validate_layout_contract(layout: MatrixGroupLayout) -> None:
    if layout.world_size != len(layout.ranks):
        raise ValueError(f"Layout world_size={layout.world_size} does not match rank count {len(layout.ranks)}.")
    for expected_rank, rank_layout in enumerate(layout.ranks):
        if rank_layout.rank != expected_rank:
            raise ValueError(f"Rank layout at index {expected_rank} has rank={rank_layout.rank}.")
        local_units = sum(segment.numel for segment in rank_layout.segments)
        if local_units != rank_layout.local_units:
            raise ValueError(
                f"Rank {rank_layout.rank} local_units={rank_layout.local_units} does not match "
                f"segments total {local_units}."
            )
        for segment in rank_layout.segments:
            if segment.global_start < 0 or segment.global_end < segment.global_start:
                raise ValueError(f"Invalid rank segment range {segment}.")
            if segment.global_end > layout.total_numel:
                raise ValueError(f"Rank segment {segment} exceeds layout total_numel={layout.total_numel}.")
    rank_segments_by_rank = {
        rank_layout.rank: rank_layout.segments
        for rank_layout in layout.ranks
    }
    for param_layout in layout.params:
        if param_layout.global_start < 0 or param_layout.global_end < param_layout.global_start:
            raise ValueError(f"Invalid param layout range for {param_layout.fqn!r}.")
        if param_layout.global_end > layout.total_numel:
            raise ValueError(
                f"Param {param_layout.fqn!r} end={param_layout.global_end} exceeds "
                f"layout total_numel={layout.total_numel}."
            )
        for segment in param_layout.segments:
            if segment.fqn != param_layout.fqn:
                raise ValueError(
                    f"Param {param_layout.fqn!r} contains segment for {segment.fqn!r}."
                )
            if segment.rank < 0 or segment.rank >= layout.world_size:
                raise ValueError(f"Param {param_layout.fqn!r} contains invalid rank {segment.rank}.")
            if segment.global_start < param_layout.global_start or segment.global_end > param_layout.global_end:
                raise ValueError(f"Param segment {segment} is outside param {param_layout.fqn!r}.")
            if not any(
                segment.global_start >= rank_segment.global_start
                and segment.global_end <= rank_segment.global_end
                and segment.local_start >= rank_segment.local_start
                and segment.local_end <= rank_segment.local_end
                for rank_segment in rank_segments_by_rank[segment.rank]
            ):
                raise ValueError(
                    f"Param segment {segment} is not contained by any rank {segment.rank} segment."
                )


@dataclass(frozen=True)
class PlannerCandidate:
    name: str
    planner: GroupPlanner
    blocks: tuple[ParamBlock, ...] = ()
    padding_units: int = 0
    num_collectives: int | None = None
    padding_alignment: int | None = None
    collectives_per_segmented_exchange: int = 2

    def evaluate(
        self,
        params: Sequence[ManagedParam],
        world_size: int,
        *,
        weights: PlannerCostWeights = PlannerCostWeights(),
        policy: str | None = None,
        shard_mesh_dim: int | str | None = None,
    ) -> PlannerResult:
        return call_group_planner(
            self.planner,
            params,
            world_size,
            name=self.name,
            blocks=self.blocks,
            weights=weights,
            padding_units=self.padding_units,
            num_collectives=self.num_collectives,
            padding_alignment=self.padding_alignment,
            collectives_per_segmented_exchange=self.collectives_per_segmented_exchange,
            policy=policy,
            shard_mesh_dim=shard_mesh_dim,
        )


def call_group_planner(
    planner: GroupPlanner,
    params: Sequence[ManagedParam],
    world_size: int,
    *,
    name: str | None = None,
    blocks: Sequence[ParamBlock] = (),
    weights: PlannerCostWeights = PlannerCostWeights(),
    padding_units: int = 0,
    num_collectives: int | None = None,
    padding_alignment: int | None = None,
    collectives_per_segmented_exchange: int = 2,
    policy: str | None = None,
    shard_mesh_dim: int | str | None = None,
) -> PlannerResult:
    planner_name = name or planner_display_name(planner)
    return planner_result_from_output(
        planner_name,
        planner(params, world_size),
        params,
        world_size,
        blocks=blocks,
        weights=weights,
        padding_units=padding_units,
        num_collectives=num_collectives,
        padding_alignment=padding_alignment,
        collectives_per_segmented_exchange=collectives_per_segmented_exchange,
        policy=policy,
        shard_mesh_dim=shard_mesh_dim,
    )


def planner_display_name(planner: object) -> str:
    return getattr(planner, "__name__", planner.__class__.__name__)


def estimate_layout_cost(
    report: MatrixGroupLayoutReport,
    *,
    weights: PlannerCostWeights = PlannerCostWeights(),
    padding_units: int = 0,
    num_collectives: int | None = None,
    runtime_compatible: bool = True,
    runtime_mode: str | None = None,
    resources: PlannerResourceEstimate | None = None,
) -> float:
    return estimate_layout_cost_breakdown(
        report,
        weights=weights,
        padding_units=padding_units,
        num_collectives=num_collectives,
        runtime_compatible=runtime_compatible,
        runtime_mode=runtime_mode,
        resources=resources,
    ).total


def estimate_layout_cost_breakdown(
    report: MatrixGroupLayoutReport,
    *,
    weights: PlannerCostWeights = PlannerCostWeights(),
    padding_units: int = 0,
    num_collectives: int | None = None,
    runtime_compatible: bool = True,
    runtime_mode: str | None = None,
    resources: PlannerResourceEstimate | None = None,
) -> PlannerCostBreakdown:
    effective_padding_units = padding_units + report.estimated_padding_units
    effective_num_collectives = report.estimated_collectives if num_collectives is None else num_collectives
    resources = resources or PlannerResourceEstimate.empty(report.world_size)
    terms = {
        "runtime_unsupported": weights.runtime_unsupported * (0 if runtime_compatible else 1),
        "runtime_flat_reorder": weights.runtime_flat_reorder * (1 if runtime_mode == "flat_reorder" else 0),
        "runtime_segment_runtime": weights.runtime_segment_runtime * (1 if runtime_mode == "segment_runtime" else 0),
        "collective": weights.collective * effective_num_collectives,
        "max_rank_unit": weights.max_rank_unit * report.max_rank_units,
        "padding_unit": weights.padding_unit * effective_padding_units,
        "split_param": weights.split_param * report.num_split_params,
        "split_param_unit": weights.split_param_unit * report.split_param_units,
        "rank_segment": weights.rank_segment * report.num_rank_segments,
        "max_rank_segment": weights.max_rank_segment * report.max_rank_segments,
        "fragmented_rank_segment": weights.fragmented_rank_segment * report.fragmented_rank_segments,
        "fragmented_rank_unit": weights.fragmented_rank_unit * report.fragmented_rank_units,
        "param_segment": weights.param_segment * report.num_param_segments,
        "split_param_segment": weights.split_param_segment * report.split_param_segments,
        "max_param_segment": weights.max_param_segment * report.max_param_segments,
        "total_comm_unit": weights.total_comm_unit * report.total_comm_units,
        "max_rank_comm_unit": weights.max_rank_comm_unit * report.max_rank_comm_units,
        "imbalance_unit": weights.imbalance_unit * report.imbalance_units,
        "imbalance_ratio": weights.imbalance_ratio * report.imbalance_ratio,
        "block": weights.block * report.num_blocks,
        "parameter_block": weights.parameter_block * report.blocks_by_kind.get("parameter", 0),
        "ordered_block": weights.ordered_block * report.blocks_by_kind.get("ordered_block", 0),
        "matrix_row_block": weights.matrix_row_block * report.blocks_by_kind.get("matrix_row_block", 0),
        "quant_block": weights.quant_block * report.blocks_by_kind.get("quant_block", 0),
        "matrix_owner_matrix": weights.matrix_owner_matrix * report.blocks_by_kind.get("matrix_owner_matrix", 0),
        "muon_matrix_owner": weights.muon_matrix_owner * report.blocks_by_kind.get("muon_matrix_owner", 0),
        "matrix_owner_tail_param": weights.matrix_owner_tail_param
        * report.blocks_by_kind.get("matrix_owner_tail_param", 0),
        "adamw_tail_param": weights.adamw_tail_param * report.blocks_by_kind.get("adamw_tail_param", 0),
        "max_rank_memory_byte": weights.max_rank_memory_byte * resources.max_rank_memory_bytes,
        "memory_imbalance_byte": weights.memory_imbalance_byte * resources.memory_imbalance_bytes,
        "max_rank_optimizer_byte": weights.max_rank_optimizer_byte * resources.max_rank_optimizer_bytes,
        "optimizer_imbalance_byte": weights.optimizer_imbalance_byte * resources.optimizer_imbalance_bytes,
        "max_rank_muon_param_byte": weights.max_rank_muon_param_byte * resources.max_rank_muon_param_bytes,
        "muon_param_imbalance_byte": weights.muon_param_imbalance_byte * resources.muon_param_imbalance_bytes,
        "total_comm_byte": weights.total_comm_byte * resources.total_comm_bytes,
        "max_rank_comm_byte": weights.max_rank_comm_byte * resources.max_rank_comm_bytes,
        "comm_imbalance_byte": weights.comm_imbalance_byte * resources.comm_imbalance_bytes,
        "workspace_preferred_unit": weights.workspace_preferred_unit * resources.workspace_preferred_numel,
        "workspace_padding_unit": weights.workspace_padding_unit * resources.workspace_padding_waste_numel,
        "workspace_padding_ratio": weights.workspace_padding_ratio * resources.workspace_padding_waste_ratio,
    }
    return PlannerCostBreakdown({name: value for name, value in terms.items() if value != 0.0})


def estimate_layout_resources(
    layout: MatrixGroupLayout,
    params: Sequence[ManagedParam],
) -> PlannerResourceEstimate:
    """
    Estimate local memory and communication pressure for a planner layout.

    This is intentionally simple and deterministic: parameters and gradients use
    the parameter dtype, Muon gets one local momentum-like state, AdamW gets two
    local tensor states, and communication is estimated as one parameter gather
    plus one gradient reduce-scatter over the sharded group.
    """

    world_size = layout.world_size
    if world_size == 0:
        return PlannerResourceEstimate.empty()

    params_by_fqn = {param.fqn: param for param in params}
    rank_param_bytes = [0 for _ in range(world_size)]
    rank_grad_bytes = [0 for _ in range(world_size)]
    rank_optimizer_bytes = [0 for _ in range(world_size)]
    rank_muon_param_bytes = [0 for _ in range(world_size)]
    rank_adamw_param_bytes = [0 for _ in range(world_size)]
    rank_unknown_param_bytes = [0 for _ in range(world_size)]

    for param_layout in layout.params:
        managed_param = params_by_fqn.get(param_layout.fqn)
        if managed_param is None:
            continue
        element_size = _dtype_element_size(managed_param.dtype)
        optimizer_type = _optimizer_type_for_resource(managed_param)
        optimizer_multiplier = _optimizer_state_multiplier(optimizer_type)
        for segment in param_layout.segments:
            local_bytes = segment.numel * element_size
            rank_param_bytes[segment.rank] += local_bytes
            rank_grad_bytes[segment.rank] += local_bytes
            rank_optimizer_bytes[segment.rank] += optimizer_multiplier * local_bytes
            if optimizer_type == "muon":
                rank_muon_param_bytes[segment.rank] += local_bytes
            elif optimizer_type == "adamw":
                rank_adamw_param_bytes[segment.rank] += local_bytes
            else:
                rank_unknown_param_bytes[segment.rank] += local_bytes

    total_param_bytes = sum(rank_param_bytes)
    total_grad_bytes = sum(rank_grad_bytes)
    total_optimizer_bytes = sum(rank_optimizer_bytes)
    workspace_preferred_numel = _workspace_preferred_numel(layout)
    workspace_padding_waste_numel = max(workspace_preferred_numel - layout.total_numel, 0)
    workspace_padding_waste_ratio = workspace_padding_waste_numel / layout.total_numel if layout.total_numel else 0.0
    rank_memory_bytes = tuple(
        param_bytes + grad_bytes + optimizer_bytes
        for param_bytes, grad_bytes, optimizer_bytes in zip(
            rank_param_bytes,
            rank_grad_bytes,
            rank_optimizer_bytes,
        )
    )
    rank_comm_bytes = tuple(2 * max(0, total_param_bytes - rank_bytes) for rank_bytes in rank_param_bytes)

    return PlannerResourceEstimate(
        rank_param_bytes=tuple(rank_param_bytes),
        rank_grad_bytes=tuple(rank_grad_bytes),
        rank_optimizer_bytes=tuple(rank_optimizer_bytes),
        rank_memory_bytes=rank_memory_bytes,
        rank_comm_bytes=rank_comm_bytes,
        rank_muon_param_bytes=tuple(rank_muon_param_bytes),
        rank_adamw_param_bytes=tuple(rank_adamw_param_bytes),
        rank_unknown_param_bytes=tuple(rank_unknown_param_bytes),
        total_param_bytes=total_param_bytes,
        total_grad_bytes=total_grad_bytes,
        total_optimizer_bytes=total_optimizer_bytes,
        total_memory_bytes=sum(rank_memory_bytes),
        total_comm_bytes=sum(rank_comm_bytes),
        max_rank_memory_bytes=max(rank_memory_bytes, default=0),
        memory_imbalance_bytes=_imbalance(rank_memory_bytes),
        max_rank_optimizer_bytes=max(rank_optimizer_bytes, default=0),
        optimizer_imbalance_bytes=_imbalance(rank_optimizer_bytes),
        max_rank_muon_param_bytes=max(rank_muon_param_bytes, default=0),
        muon_param_imbalance_bytes=_imbalance(rank_muon_param_bytes),
        max_rank_comm_bytes=max(rank_comm_bytes, default=0),
        comm_imbalance_bytes=_imbalance(rank_comm_bytes),
        workspace_preferred_numel=workspace_preferred_numel,
        workspace_padding_waste_numel=workspace_padding_waste_numel,
        workspace_padding_waste_ratio=workspace_padding_waste_ratio,
    )


def _workspace_preferred_numel(layout: MatrixGroupLayout) -> int:
    if layout.world_size <= 1:
        return layout.total_numel
    max_shard_size = max(layout.shard_sizes, default=0)
    return layout.world_size * max_shard_size


def _dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _optimizer_type_for_resource(param: ManagedParam) -> str | None:
    optimizer_type = param.shard_hint.optimizer_type
    if optimizer_type in {"muon", "adamw"}:
        return optimizer_type
    return None


def _optimizer_state_multiplier(optimizer_type: str | None) -> int:
    if optimizer_type == "muon":
        return 1
    if optimizer_type == "adamw":
        return 2
    return 0


def _imbalance(values: Sequence[int]) -> int:
    if not values:
        return 0
    return max(values) - min(values)


def evaluate_group_planner(
    name: str,
    params: Sequence[ManagedParam],
    world_size: int,
    planner: GroupPlannerCandidate,
    *,
    blocks: Sequence[ParamBlock] = (),
    weights: PlannerCostWeights = PlannerCostWeights(),
    padding_units: int = 0,
    num_collectives: int | None = None,
    padding_alignment: int | None = None,
    collectives_per_segmented_exchange: int = 2,
) -> PlannerEvaluation:
    return call_group_planner(
        planner,
        params,
        world_size,
        name=name,
        blocks=blocks,
        weights=weights,
        padding_units=padding_units,
        num_collectives=num_collectives,
        padding_alignment=padding_alignment,
        collectives_per_segmented_exchange=collectives_per_segmented_exchange,
    )


def planner_result_from_output(
    name: str,
    output: PlannerOutput,
    params: Sequence[ManagedParam],
    world_size: int,
    *,
    blocks: Sequence[ParamBlock] = (),
    weights: PlannerCostWeights = PlannerCostWeights(),
    padding_units: int = 0,
    num_collectives: int | None = None,
    padding_alignment: int | None = None,
    collectives_per_segmented_exchange: int = 2,
    policy: str | None = None,
    shard_mesh_dim: int | str | None = None,
) -> PlannerResult:
    if isinstance(output, PlannerResult):
        runtime_compatibility = explain_runtime_layout_compatibility(output.layout, params, world_size)
        resource_estimate = estimate_layout_resources(output.layout, params)
        runtime_cost_breakdown = estimate_layout_cost_breakdown(
            output.report,
            weights=output.cost_weights,
            padding_units=output.padding_units,
            num_collectives=output.num_collectives,
            runtime_compatible=runtime_compatibility.compatible,
            runtime_mode=runtime_compatibility.mode,
            resources=resource_estimate,
        )
        replacements = {}
        if shard_mesh_dim is not None and output.shard_mesh_dim is None:
            replacements["shard_mesh_dim"] = shard_mesh_dim
        if not output.constraints:
            replacements["constraints"] = infer_group_constraints(params)
        replacements.update(
            _runtime_replacements(
                runtime_compatibility,
                output.warnings,
                cost_breakdown=runtime_cost_breakdown,
            )
        )
        replacements["resource_estimate"] = resource_estimate
        if not replacements:
            return output
        return replace(output, **replacements)

    if isinstance(output, ShardPlan):
        layout = MatrixGroupLayout.from_shard_plan(output, params)
    else:
        layout = output
    runtime_compatibility = explain_runtime_layout_compatibility(layout, params, world_size)
    resource_estimate = estimate_layout_resources(layout, params)
    report = report_group_layout(
        layout,
        blocks,
        padding_alignment=padding_alignment,
        collectives_per_segmented_exchange=collectives_per_segmented_exchange,
    )
    cost_breakdown = estimate_layout_cost_breakdown(
        report,
        weights=weights,
        padding_units=padding_units,
        num_collectives=num_collectives,
        runtime_compatible=runtime_compatibility.compatible,
        runtime_mode=runtime_compatibility.mode,
        resources=resource_estimate,
    )
    return PlannerResult(
        name=name,
        layout=layout,
        report=report,
        cost=cost_breakdown.total,
        shard_mesh_dim=shard_mesh_dim,
        constraints=infer_group_constraints(params),
        warnings=_runtime_warnings(runtime_compatibility, ()),
        runtime_compatible=runtime_compatibility.compatible,
        runtime_mode=runtime_compatibility.mode,
        runtime_reason=runtime_compatibility.reason,
        runtime_requires_flat_reorder=runtime_compatibility.requires_flat_reorder,
        matrix_shard_compatible=runtime_compatibility.matrix_shard_compatible,
        matrix_shard_reason=runtime_compatibility.matrix_shard_reason,
        padding_units=padding_units,
        num_collectives=report.estimated_collectives if num_collectives is None else num_collectives,
        total_comm_units=report.total_comm_units,
        max_rank_comm_units=report.max_rank_comm_units,
        resource_estimate=resource_estimate,
        num_rank_segments=report.num_rank_segments,
        max_rank_segments=report.max_rank_segments,
        policy=policy,
        cost_weights=weights,
        cost_breakdown=cost_breakdown,
    )


def compare_group_planners(
    params: Sequence[ManagedParam],
    world_size: int,
    planners: Mapping[str, GroupPlannerCandidate | PlannerCandidate] | Sequence[PlannerCandidate],
    *,
    blocks_by_name: Mapping[str, Sequence[ParamBlock]] | None = None,
    weights: PlannerCostWeights = PlannerCostWeights(),
    padding_units_by_name: Mapping[str, int] | None = None,
    num_collectives_by_name: Mapping[str, int] | None = None,
    padding_alignment_by_name: Mapping[str, int] | None = None,
    collectives_per_segmented_exchange_by_name: Mapping[str, int] | None = None,
) -> tuple[PlannerEvaluation, ...]:
    planner_candidates = normalize_planner_candidates(
        planners,
        blocks_by_name=blocks_by_name,
        padding_units_by_name=padding_units_by_name,
        num_collectives_by_name=num_collectives_by_name,
        padding_alignment_by_name=padding_alignment_by_name,
        collectives_per_segmented_exchange_by_name=collectives_per_segmented_exchange_by_name,
    )
    evaluations = []
    for candidate in planner_candidates:
        evaluations.append(candidate.evaluate(params, world_size, weights=weights))
    return tuple(
        sorted(
            evaluations,
            key=lambda evaluation: (not evaluation.runtime_compatible, evaluation.cost, evaluation.name),
        )
    )


def _runtime_replacements(
    runtime_compatibility,
    warnings: tuple[str, ...],
    *,
    cost_breakdown: PlannerCostBreakdown,
) -> dict[str, object]:
    return {
        "runtime_compatible": runtime_compatibility.compatible,
        "runtime_mode": runtime_compatibility.mode,
        "runtime_reason": runtime_compatibility.reason,
        "runtime_requires_flat_reorder": runtime_compatibility.requires_flat_reorder,
        "matrix_shard_compatible": runtime_compatibility.matrix_shard_compatible,
        "matrix_shard_reason": runtime_compatibility.matrix_shard_reason,
        "warnings": _runtime_warnings(runtime_compatibility, warnings),
        "cost": cost_breakdown.total,
        "cost_breakdown": cost_breakdown,
    }


def _runtime_warnings(runtime_compatibility, warnings: tuple[str, ...]) -> tuple[str, ...]:
    if runtime_compatibility.compatible:
        return warnings
    runtime_warning = f"runtime unsupported: {runtime_compatibility.reason}"
    if runtime_warning in warnings:
        return warnings
    return (*warnings, runtime_warning)


def select_best_group_planner(
    params: Sequence[ManagedParam],
    world_size: int,
    planners: Mapping[str, GroupPlannerCandidate | PlannerCandidate] | Sequence[PlannerCandidate],
    **kwargs,
) -> PlannerEvaluation:
    evaluations = compare_group_planners(params, world_size, planners, **kwargs)
    if not evaluations:
        raise ValueError("Expected at least one planner candidate.")
    return evaluations[0]


def normalize_planner_candidates(
    planners: Mapping[str, GroupPlannerCandidate | PlannerCandidate] | Sequence[PlannerCandidate],
    *,
    blocks_by_name: Mapping[str, Sequence[ParamBlock]] | None = None,
    padding_units_by_name: Mapping[str, int] | None = None,
    num_collectives_by_name: Mapping[str, int] | None = None,
    padding_alignment_by_name: Mapping[str, int] | None = None,
    collectives_per_segmented_exchange_by_name: Mapping[str, int] | None = None,
) -> tuple[PlannerCandidate, ...]:
    if not isinstance(planners, Mapping):
        return tuple(planners)

    blocks_by_name = blocks_by_name or {}
    padding_units_by_name = padding_units_by_name or {}
    num_collectives_by_name = num_collectives_by_name or {}
    padding_alignment_by_name = padding_alignment_by_name or {}
    collectives_per_segmented_exchange_by_name = collectives_per_segmented_exchange_by_name or {}
    candidates = []
    for name, planner_or_candidate in planners.items():
        if isinstance(planner_or_candidate, PlannerCandidate):
            candidates.append(planner_or_candidate)
            continue
        candidates.append(
            PlannerCandidate(
                name=name,
                planner=planner_or_candidate,
                blocks=tuple(blocks_by_name.get(name, ())),
                padding_units=padding_units_by_name.get(name, 0),
                num_collectives=num_collectives_by_name.get(name),
                padding_alignment=padding_alignment_by_name.get(name),
                collectives_per_segmented_exchange=collectives_per_segmented_exchange_by_name.get(name, 2),
            )
        )
    return tuple(candidates)
