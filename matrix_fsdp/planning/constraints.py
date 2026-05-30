from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from matrix_fsdp.core.managed_param import ManagedParam, ParamRuntimeKind


class ShardConstraint(str, Enum):
    NO_SPLIT_PARAM = "no_split_param"
    NO_SPLIT_MATRIX = "no_split_matrix"
    WHOLE_PARAM_OWNER = "whole_param_owner"
    BLOCK_ALIGNED = "block_aligned"
    ROW_BLOCK_ALIGNED = "row_block_aligned"
    MUON_MATRIX_OWNER = "muon_matrix_owner"
    EXPERT_OWNER = "expert_owner"


@dataclass(frozen=True)
class ParamShardConstraints:
    fqn: str
    constraints: tuple[ShardConstraint, ...]
    block_shape: tuple[int, ...] | None = None
    optimizer_type: str | None = None
    runtime_kind: str | None = None
    parallel_role: str | None = None
    expert_group_id: str | None = None


def infer_param_constraints(param: ManagedParam) -> ParamShardConstraints:
    hint = param.shard_hint
    constraints: list[ShardConstraint] = []

    if hint.split_granularity == "parameter":
        constraints.append(ShardConstraint.NO_SPLIT_PARAM)
        constraints.append(ShardConstraint.WHOLE_PARAM_OWNER)
    elif hint.split_granularity == "matrix_owner":
        constraints.append(ShardConstraint.NO_SPLIT_PARAM)
        constraints.append(ShardConstraint.NO_SPLIT_MATRIX)
        constraints.append(ShardConstraint.WHOLE_PARAM_OWNER)
        constraints.append(ShardConstraint.MUON_MATRIX_OWNER)
    elif hint.split_granularity == "block":
        constraints.append(ShardConstraint.BLOCK_ALIGNED)
    elif hint.split_granularity == "row_block":
        constraints.append(ShardConstraint.ROW_BLOCK_ALIGNED)
    if hint.runtime_kind == ParamRuntimeKind.EXPERT_OWNER:
        constraints.append(ShardConstraint.EXPERT_OWNER)

    return ParamShardConstraints(
        fqn=param.fqn,
        constraints=tuple(constraints),
        block_shape=hint.block_shape,
        optimizer_type=hint.optimizer_type,
        runtime_kind=hint.runtime_kind,
        parallel_role=hint.parallel_role,
        expert_group_id=hint.expert_group_id,
    )


def infer_group_constraints(params: Sequence[ManagedParam]) -> tuple[ParamShardConstraints, ...]:
    return tuple(infer_param_constraints(param) for param in params)


def constraint_counts(
    constraints: Sequence[ParamShardConstraints],
) -> dict[ShardConstraint, int]:
    counts: dict[ShardConstraint, int] = {}
    for param_constraints in constraints:
        for constraint in param_constraints.constraints:
            counts[constraint] = counts.get(constraint, 0) + 1
    return counts


def constraints_metadata(
    constraints: Sequence[ParamShardConstraints],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "fqn": param_constraints.fqn,
            "constraints": tuple(constraint.value for constraint in param_constraints.constraints),
            "block_shape": param_constraints.block_shape,
            "optimizer_type": param_constraints.optimizer_type,
            "runtime_kind": param_constraints.runtime_kind,
            "parallel_role": param_constraints.parallel_role,
            "expert_group_id": param_constraints.expert_group_id,
        }
        for param_constraints in constraints
    )


def constraint_count_metadata(
    counts: Mapping[ShardConstraint, int],
) -> tuple[tuple[str, int], ...]:
    return tuple((constraint.value, count) for constraint, count in sorted(counts.items(), key=lambda item: item[0].value))
