from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy


def mixed_precision_policy_metadata(policy: MixedPrecisionPolicy | None) -> dict[str, object]:
    return {
        "param_dtype": _dtype_name(policy.param_dtype if policy is not None else None),
        "reduce_dtype": _dtype_name(policy.reduce_dtype if policy is not None else None),
        "output_dtype": _dtype_name(policy.output_dtype if policy is not None else None),
        "cast_forward_inputs": True if policy is None else bool(policy.cast_forward_inputs),
    }


def offload_policy_metadata(policy: OffloadPolicy | None) -> dict[str, object]:
    metadata = {
        "type": type(policy).__name__ if policy is not None else type(OffloadPolicy()).__name__,
    }
    if isinstance(policy, CPUOffloadPolicy):
        metadata["pin_memory"] = bool(policy.pin_memory)
        metadata["cpu_offload"] = True
    else:
        metadata["cpu_offload"] = False
    return metadata


def validate_mixed_precision_policy(policy: MixedPrecisionPolicy) -> None:
    for field_name in ("param_dtype", "reduce_dtype", "output_dtype"):
        dtype = getattr(policy, field_name)
        if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError(f"mp_policy.{field_name} must be a floating-point dtype, got {dtype}.")


def validate_offload_policy(policy: OffloadPolicy) -> None:
    if type(policy) in (OffloadPolicy, CPUOffloadPolicy):
        return
    if not isinstance(policy, OffloadPolicy):
        raise TypeError(f"offload_policy must be an OffloadPolicy, got {type(policy).__name__}.")
    raise NotImplementedError(
        "MatrixFSDP currently accepts OffloadPolicy and CPUOffloadPolicy only; "
        f"got {type(policy).__name__}."
    )


def validate_offload_policy_for_device(policy: OffloadPolicy, device: torch.device) -> None:
    if isinstance(policy, CPUOffloadPolicy) and device.type != "cpu":
        raise NotImplementedError(
            "MatrixFSDP CPUOffloadPolicy is currently supported only for CPU-resident "
            "models. CUDA parameter/gradient/optimizer-state offload requires a "
            "separate host-device transfer runtime."
        )


def cast_managed_parameter_dtype(
    module: nn.Module,
    *,
    param_dtype: torch.dtype | None,
    ignored_params: set[nn.Parameter] | None,
) -> None:
    if param_dtype is None:
        return
    ignored_param_ids = {id(param) for param in ignored_params or set()}
    for param in module.parameters(recurse=True):
        if id(param) in ignored_param_ids:
            continue
        if not param.is_floating_point():
            continue
        if param.dtype == param_dtype:
            continue
        param.data = param.data.to(dtype=param_dtype)
        if param.grad is not None:
            param.grad = param.grad.to(dtype=param_dtype)


def cast_floating_tensors(value: Any, dtype: torch.dtype | None) -> Any:
    if dtype is None:
        return value
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value
    if isinstance(value, tuple):
        if hasattr(value, "_fields"):
            return type(value)(*(cast_floating_tensors(item, dtype) for item in value))
        return tuple(cast_floating_tensors(item, dtype) for item in value)
    if isinstance(value, list):
        return [cast_floating_tensors(item, dtype) for item in value]
    if isinstance(value, Mapping):
        return type(value)((key, cast_floating_tensors(item, dtype)) for key, item in value.items())
    return value


def _dtype_name(dtype: torch.dtype | None) -> str | None:
    return None if dtype is None else str(dtype)
