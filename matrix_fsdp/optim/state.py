from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.optim import Optimizer

from matrix_fsdp.runtime.flat_buffer import MatrixFlatBuffer
from matrix_fsdp.runtime.param_group import MatrixFSDPParamGroup
from matrix_fsdp.core.managed_param import ManagedParam
from matrix_fsdp.core.state import MatrixShardedState


@dataclass(frozen=True)
class LocalOptimizerParamInfo:
    flat_buffer: MatrixFlatBuffer
    managed_param: ManagedParam


@dataclass(frozen=True)
class LocalOptimizerStateTensor:
    fqn: str
    name: str
    tensor: torch.Tensor
    sharded_state: MatrixShardedState


@dataclass(frozen=True)
class OptimizerStateSummary:
    param_numel: int
    state_entries: int
    tensor_state_numel: int
    tensor_state_numel_by_name: dict[str, int]
    scalar_state_tensor_numel: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "param_numel": self.param_numel,
            "state_entries": self.state_entries,
            "tensor_state_numel": self.tensor_state_numel,
            "tensor_state_numel_by_name": dict(self.tensor_state_numel_by_name),
            "scalar_state_tensor_numel": self.scalar_state_tensor_numel,
        }


class MatrixFSDPOptimizerStateManager:
    """
    Build MatrixShardedState views for local optimizer tensor state.

    Torch optimizers still own ordinary local tensors for mutation. This manager
    attaches MatrixShard metadata to shape-compatible tensor states so validation
    and checkpointing can reason about optimizer state with the same layout
    vocabulary as parameters and gradients. DTensor wrappers are intentionally
    not materialized on the runtime path.
    """

    def __init__(self, optimizer: Optimizer, param_groups: list[MatrixFSDPParamGroup]) -> None:
        self.optimizer = optimizer
        self.fsdp_param_groups = param_groups
        self._state_objects: dict[str, dict[str, MatrixShardedState]] = {}
        self._state_dtensors: dict[str, dict[str, Any]] = {}

    @property
    def runtime_param_groups(self) -> list[MatrixFSDPParamGroup]:
        return self.fsdp_param_groups

    @property
    def state_objects(self) -> dict[str, dict[str, MatrixShardedState]]:
        return self._state_objects

    @property
    def state_dtensors(self) -> dict[str, dict[str, Any]]:
        return self._state_dtensors

    def local_state_summary(self) -> dict[str, Any]:
        return self.summarize().as_dict()

    def summarize(self) -> OptimizerStateSummary:
        params = [param for group in self.optimizer.param_groups for param in group["params"]]
        tensor_state_numel_by_name: dict[str, int] = {}
        tensor_state_numel = 0
        scalar_state_tensor_numel = 0
        for state in self.optimizer.state.values():
            for name, value in state.items():
                if not torch.is_tensor(value):
                    continue
                if value.ndim == 0:
                    scalar_state_tensor_numel += value.numel()
                    continue
                tensor_state_numel += value.numel()
                tensor_state_numel_by_name[name] = tensor_state_numel_by_name.get(name, 0) + value.numel()
        return OptimizerStateSummary(
            param_numel=sum(param.numel() for param in params),
            state_entries=len(self.optimizer.state),
            tensor_state_numel=tensor_state_numel,
            tensor_state_numel_by_name=tensor_state_numel_by_name,
            scalar_state_tensor_numel=scalar_state_tensor_numel,
        )

    def local_state_dtensors(self) -> dict[str, dict[str, Any]]:
        state_dtensors: dict[str, dict[str, Any]] = {}
        for fqn, states in self.local_state_objects().items():
            wrapped_state = {name: state.dtensor for name, state in states.items() if state.dtensor is not None}
            if wrapped_state:
                state_dtensors[fqn] = wrapped_state
        return state_dtensors

    def local_state_objects(self) -> dict[str, dict[str, MatrixShardedState]]:
        state_objects: dict[str, dict[str, MatrixShardedState]] = {}
        for state_tensor in self.iter_local_state_tensors():
            state_objects.setdefault(state_tensor.fqn, {})[state_tensor.name] = state_tensor.sharded_state
        return state_objects

    def local_state_metadata(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {
            fqn: {name: matrix_sharded_state_metadata(state) for name, state in states.items()}
            for fqn, states in self.local_state_objects().items()
        }

    def iter_local_state_tensors(self) -> tuple[LocalOptimizerStateTensor, ...]:
        param_info_by_id = self._param_info_by_id()
        state_tensors: list[LocalOptimizerStateTensor] = []
        for param, state in self.optimizer.state.items():
            if not torch.is_tensor(param) or param.numel() == 0:
                continue
            param_info = param_info_by_id.get(id(param))
            if param_info is None:
                continue
            for name, value in state.items():
                if not self._is_param_shaped_tensor_state(value, param):
                    continue
                sharded_state = param_info.flat_buffer.make_param_state(param_info.managed_param, name, value)
                if sharded_state is None:
                    continue
                state_tensors.append(
                    LocalOptimizerStateTensor(
                        fqn=param_info.managed_param.fqn,
                        name=name,
                        tensor=value,
                        sharded_state=sharded_state,
                    )
                )
        return tuple(state_tensors)

    def refresh_state_dtensors(self) -> dict[str, dict[str, Any]]:
        self._state_objects = self.local_state_objects()
        self._state_dtensors = {}
        for fqn, states in self._state_objects.items():
            wrapped_state = {name: state.dtensor for name, state in states.items() if state.dtensor is not None}
            if wrapped_state:
                self._state_dtensors[fqn] = wrapped_state
        return self._state_dtensors

    def validate_local_state_shapes(self) -> None:
        for param, state in self.optimizer.state.items():
            if not torch.is_tensor(param):
                continue
            for name, value in state.items():
                if not torch.is_tensor(value) or value.ndim == 0:
                    continue
                if value.numel() != param.numel():
                    raise RuntimeError(
                        f"Optimizer state {name!r} has {value.numel()} elements for a "
                        f"local parameter with {param.numel()} elements."
                    )

    def _param_info_by_id(self) -> dict[int, LocalOptimizerParamInfo]:
        param_info_by_id: dict[int, LocalOptimizerParamInfo] = {}
        for unit in self.fsdp_param_groups:
            flat_buffer = unit.flat_buffer
            if flat_buffer is None:
                continue
            for view in flat_buffer.local_param_views:
                param_info_by_id[id(view.managed_param.param)] = LocalOptimizerParamInfo(
                    flat_buffer=flat_buffer,
                    managed_param=view.managed_param,
                )
        return param_info_by_id

    def _is_param_shaped_tensor_state(self, value: Any, param: torch.Tensor) -> bool:
        if not torch.is_tensor(value) or value.ndim == 0:
            return False
        return value.numel() == param.numel()


def matrix_sharded_state_metadata(state: MatrixShardedState) -> dict[str, Any]:
    return state.as_metadata()
