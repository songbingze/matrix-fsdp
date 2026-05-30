from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from matrix_fsdp.runtime.param_group import MatrixFSDPParamGroup


def collect_param_groups(
    module_or_param_groups: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
) -> list[MatrixFSDPParamGroup]:
    if isinstance(module_or_param_groups, MatrixFSDPParamGroup):
        return [module_or_param_groups]
    if isinstance(module_or_param_groups, nn.Module):
        return _collect_param_groups_from_module(module_or_param_groups)

    param_groups = list(module_or_param_groups)
    _validate_param_groups(param_groups)
    return param_groups


def _collect_param_groups_from_module(module: nn.Module) -> list[MatrixFSDPParamGroup]:
    param_groups: list[MatrixFSDPParamGroup] = []
    seen: set[int] = set()
    for submodule in module.modules():
        param_group = getattr(submodule, "_matrix_fsdp_param_group", None)
        if param_group is None:
            continue
        if not isinstance(param_group, MatrixFSDPParamGroup):
            raise TypeError(f"Expected MatrixFSDPParamGroup, got {type(param_group)!r}.")
        param_group_id = id(param_group)
        if param_group_id in seen:
            continue
        seen.add(param_group_id)
        param_groups.append(param_group)
    return param_groups


def _validate_param_groups(param_groups: list[MatrixFSDPParamGroup]) -> None:
    if not param_groups:
        raise ValueError("Expected at least one MatrixFSDPParamGroup.")
    for param_group in param_groups:
        if not isinstance(param_group, MatrixFSDPParamGroup):
            raise TypeError(f"Expected MatrixFSDPParamGroup, got {type(param_group)!r}.")
