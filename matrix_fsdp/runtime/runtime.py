from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import NewType

RuntimeParamGroupId = NewType("RuntimeParamGroupId", str)
RuntimeUnitId = RuntimeParamGroupId
PlannerGroupId = NewType("PlannerGroupId", str)
CommBufferId = NewType("CommBufferId", str)

_PARAM_GROUP_COUNTER = count()


@dataclass(frozen=True, init=False)
class RuntimeUnitMetadata:
    """Identifiers for lifecycle, planner scope, and comm-buffer lifetime."""

    runtime_param_group_id: RuntimeParamGroupId
    planner_group_id: PlannerGroupId
    comm_buffer_id: CommBufferId

    def __init__(
        self,
        *,
        runtime_param_group_id: RuntimeParamGroupId | str | None = None,
        planner_group_id: PlannerGroupId | str,
        comm_buffer_id: CommBufferId | str,
        runtime_unit_id: RuntimeUnitId | str | None = None,
    ) -> None:
        if runtime_param_group_id is None:
            if runtime_unit_id is None:
                raise TypeError("RuntimeUnitMetadata requires runtime_param_group_id.")
            runtime_param_group_id = RuntimeParamGroupId(str(runtime_unit_id))
        elif runtime_unit_id is not None and str(runtime_param_group_id) != str(runtime_unit_id):
            raise ValueError("runtime_param_group_id and runtime_unit_id must match when both are provided.")

        object.__setattr__(self, "runtime_param_group_id", RuntimeParamGroupId(str(runtime_param_group_id)))
        object.__setattr__(self, "planner_group_id", PlannerGroupId(str(planner_group_id)))
        object.__setattr__(self, "comm_buffer_id", CommBufferId(str(comm_buffer_id)))

    @property
    def runtime_unit_id(self) -> RuntimeUnitId:
        return RuntimeUnitId(self.runtime_param_group_id)


def new_runtime_unit_metadata(prefix: str = "param_group") -> RuntimeUnitMetadata:
    param_group_id = RuntimeParamGroupId(f"{prefix}_{next(_PARAM_GROUP_COUNTER)}")
    return RuntimeUnitMetadata(
        runtime_param_group_id=param_group_id,
        planner_group_id=PlannerGroupId(param_group_id),
        comm_buffer_id=CommBufferId(param_group_id),
    )
