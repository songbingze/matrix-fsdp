from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


def _unwrap_optimizer(optimizer: Any):
    return getattr(optimizer, "optimizer", optimizer)


def _optimizer_components(optimizer: Any) -> tuple[tuple[str, Any], ...]:
    torch_optimizer = _unwrap_optimizer(optimizer)
    iter_named_optimizers = getattr(torch_optimizer, "iter_named_optimizers", None)
    if iter_named_optimizers is not None:
        return tuple(iter_named_optimizers())
    return (("optimizer", torch_optimizer),)


def _optimizer_kind(optimizer: Any) -> str:
    torch_optimizer = _unwrap_optimizer(optimizer)
    if hasattr(torch_optimizer, "iter_named_optimizers"):
        return "mixed_muon_adamw"
    return type(torch_optimizer).__name__


def _optimizer_group_summary_metadata(optimizer: Any) -> dict[str, Any] | None:
    optimizer_group_summary = getattr(optimizer, "optimizer_group_summary", None)
    if optimizer_group_summary is not None:
        return deepcopy(optimizer_group_summary())
    torch_optimizer = _unwrap_optimizer(optimizer)
    optimizer_group_summary = getattr(torch_optimizer, "optimizer_group_summary", None)
    if optimizer_group_summary is not None:
        return deepcopy(optimizer_group_summary())
    return None


def _optimizer_param_device(torch_optimizer: Any, state_id: int) -> torch.device:
    optimizer_state_dict = torch_optimizer.state_dict()
    for param_group, state_param_group in zip(torch_optimizer.param_groups, optimizer_state_dict["param_groups"]):
        for param, param_state_id in zip(param_group["params"], state_param_group["params"]):
            if param_state_id == state_id:
                return param.device
    return torch.device("cpu")


def _validate_optimizer_param_groups(
    current_param_groups: list[dict[str, Any]],
    saved_param_groups: list[dict[str, Any]],
) -> None:
    if len(current_param_groups) != len(saved_param_groups):
        raise ValueError(
            f"Optimizer has {len(current_param_groups)} param groups, "
            f"checkpoint has {len(saved_param_groups)}."
        )
    for current_group, saved_group in zip(current_param_groups, saved_param_groups):
        if len(current_group["params"]) != len(saved_group["params"]):
            raise ValueError(
                f"Optimizer param group has {len(current_group['params'])} params, "
                f"checkpoint group has {len(saved_group['params'])}."
            )
        saved_group["params"] = list(current_group["params"])


def _optimizer_param_groups_for_reshard(
    current_param_groups: list[dict[str, Any]],
    saved_param_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(current_param_groups) != len(saved_param_groups):
        raise ValueError(
            f"Optimizer has {len(current_param_groups)} param groups, "
            f"checkpoint has {len(saved_param_groups)}."
        )
    resharded_param_groups = []
    for current_group, saved_group in zip(current_param_groups, saved_param_groups):
        resharded_group = deepcopy(saved_group)
        resharded_group["params"] = list(current_group["params"])
        resharded_param_groups.append(resharded_group)
    return resharded_param_groups
