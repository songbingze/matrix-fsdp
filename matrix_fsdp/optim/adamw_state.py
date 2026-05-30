from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

try:
    from torch.optim.adam import _device_dtype_check_for_fused, _get_scalar_dtype
except ImportError:  # pragma: no cover - older PyTorch fallback.
    _device_dtype_check_for_fused = None

    def _get_scalar_dtype(is_fused=None):
        return torch.float64 if torch.get_default_dtype() == torch.float64 and not is_fused else torch.float32

from matrix_fsdp.runtime.flat_buffer import LocalParamView, MatrixFlatBuffer
from matrix_fsdp.runtime.param_group import MatrixFSDPParamGroup


@dataclass(frozen=True)
class FlatAdamWStateBuffers:
    total_numel: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    max_exp_avg_sq: torch.Tensor | None = None


@dataclass(frozen=True)
class _LocalParamStateView:
    flat_buffer: MatrixFlatBuffer
    local_view: LocalParamView
    state_start: int
    state_end: int


@dataclass(frozen=True)
class _FlatAdamWStateLayout:
    param_views_by_id: dict[int, _LocalParamStateView]
    total_numel: int


def initialize_flat_adamw_state(
    optimizer: Optimizer,
    fsdp_param_groups: list[MatrixFSDPParamGroup],
) -> bool:
    """
    Pre-initialize AdamW tensor state from flat-buffer views.

    PyTorch AdamW lazily creates ``zeros_like(param)`` states on the first step.
    For matrix layouts that can produce a different local parameter-size
    histogram per rank even when total local numel is balanced. Initializing the
    state from optimizer-local flat buffers gives us a diagnostic path for
    making allocator allocation shapes identical across ranks while keeping
    ordinary torch AdamW in charge of the update.
    """

    if not isinstance(optimizer, torch.optim.AdamW):
        return False
    state_layout = _local_param_state_layout(fsdp_param_groups)
    if not state_layout.param_views_by_id:
        return False

    flat_state_buffers = _optimizer_flat_state_buffers(optimizer)
    initialized = False
    for group in optimizer.param_groups:
        for param in group.get("params", ()):
            if not isinstance(param, nn.Parameter) or param.grad is None or param.numel() == 0:
                continue
            if param.grad.is_sparse:
                continue
            local_state_view = state_layout.param_views_by_id.get(id(param))
            if local_state_view is None:
                continue
            if len(optimizer.state[param]) != 0:
                continue
            _initialize_param_adamw_state(
                optimizer,
                group,
                param,
                local_state_view,
                state_layout.total_numel,
                flat_state_buffers,
            )
            initialized = True
    return initialized


def _local_param_state_layout(
    fsdp_param_groups: list[MatrixFSDPParamGroup],
) -> _FlatAdamWStateLayout:
    views_by_id: dict[int, _LocalParamStateView] = {}
    ambiguous: set[int] = set()
    state_base = 0
    for param_group in fsdp_param_groups:
        flat_buffer = param_group.flat_buffer
        if flat_buffer is None:
            continue
        flat_buffer_base = state_base
        state_base += flat_buffer.local_numel
        for local_view in flat_buffer.local_param_views:
            param = local_view.managed_param.param
            param_id = id(param)
            if param.numel() == 0 or local_view.numel != param.numel():
                ambiguous.add(param_id)
                views_by_id.pop(param_id, None)
                continue
            if param_id in views_by_id:
                ambiguous.add(param_id)
                views_by_id.pop(param_id, None)
                continue
            state_start = flat_buffer_base + local_view.shard_start
            views_by_id[param_id] = _LocalParamStateView(
                flat_buffer=flat_buffer,
                local_view=local_view,
                state_start=state_start,
                state_end=state_start + local_view.numel,
            )
    for param_id in ambiguous:
        views_by_id.pop(param_id, None)
    return _FlatAdamWStateLayout(views_by_id, state_base)


def _optimizer_flat_state_buffers(
    optimizer: Optimizer,
) -> dict[tuple[str, int | None, torch.dtype], FlatAdamWStateBuffers]:
    buffers = getattr(optimizer, "_matrix_fsdp_flat_adamw_state_buffers", None)
    if buffers is None:
        buffers = {}
        optimizer._matrix_fsdp_flat_adamw_state_buffers = buffers  # type: ignore[attr-defined]
    return buffers


def _initialize_param_adamw_state(
    optimizer: Optimizer,
    group: dict[str, Any],
    param: nn.Parameter,
    local_state_view: _LocalParamStateView,
    total_state_numel: int,
    flat_state_buffers: dict[tuple[str, int | None, torch.dtype], FlatAdamWStateBuffers],
) -> None:
    fused = bool(group.get("fused", False))
    if fused and _device_dtype_check_for_fused is not None:
        _device_dtype_check_for_fused(param)

    state = optimizer.state[param]
    state["step"] = _new_adamw_step_tensor(param, group)

    buffers = _flat_state_buffers_for_param(
        param,
        total_state_numel=total_state_numel,
        need_max_exp_avg_sq=bool(group.get("amsgrad", False)),
        flat_state_buffers=flat_state_buffers,
    )
    start = local_state_view.state_start
    end = local_state_view.state_end
    state["exp_avg"] = buffers.exp_avg[start:end].view_as(param)
    state["exp_avg_sq"] = buffers.exp_avg_sq[start:end].view_as(param)
    if group.get("amsgrad", False):
        if buffers.max_exp_avg_sq is None:
            raise RuntimeError("Internal error: missing max_exp_avg_sq buffer for AdamW amsgrad state.")
        state["max_exp_avg_sq"] = buffers.max_exp_avg_sq[start:end].view_as(param)


def _new_adamw_step_tensor(param: nn.Parameter, group: dict[str, Any]) -> torch.Tensor:
    fused = bool(group.get("fused", False))
    if group.get("capturable", False) or fused:
        return torch.zeros(
            (),
            dtype=_get_scalar_dtype(is_fused=fused),
            device=param.device,
        )
    return torch.tensor(0.0, dtype=_get_scalar_dtype())


def _flat_state_buffers_for_param(
    param: nn.Parameter,
    *,
    total_state_numel: int,
    need_max_exp_avg_sq: bool,
    flat_state_buffers: dict[tuple[str, int | None, torch.dtype], FlatAdamWStateBuffers],
) -> FlatAdamWStateBuffers:
    key = (param.device.type, param.device.index, param.dtype)
    existing = flat_state_buffers.get(key)
    if existing is not None:
        if existing.total_numel != total_state_numel:
            raise RuntimeError(
                f"Existing flat AdamW state has {existing.total_numel} elements, "
                f"but the current MatrixFSDP optimizer layout expects {total_state_numel}."
            )
        if need_max_exp_avg_sq and existing.max_exp_avg_sq is None:
            flat_state_buffers[key] = FlatAdamWStateBuffers(
                total_numel=existing.total_numel,
                exp_avg=existing.exp_avg,
                exp_avg_sq=existing.exp_avg_sq,
                max_exp_avg_sq=torch.zeros(
                    total_state_numel,
                    device=param.device,
                    dtype=param.dtype,
                ),
            )
        return flat_state_buffers[key]

    exp_avg = torch.zeros(total_state_numel, device=param.device, dtype=param.dtype)
    exp_avg_sq = torch.zeros(total_state_numel, device=param.device, dtype=param.dtype)
    max_exp_avg_sq = (
        torch.zeros(total_state_numel, device=param.device, dtype=param.dtype)
        if need_max_exp_avg_sq
        else None
    )
    flat_state_buffers[key] = FlatAdamWStateBuffers(
        total_numel=total_state_numel,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        max_exp_avg_sq=max_exp_avg_sq,
    )
    return flat_state_buffers[key]
