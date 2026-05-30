from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from matrix_fsdp.api import WrapPolicy


def module_type_policy(module_types: type[nn.Module] | Iterable[type[nn.Module]]) -> WrapPolicy:
    if isinstance(module_types, type):
        module_types = (module_types,)
    else:
        module_types = tuple(module_types)
    if not module_types:
        raise ValueError("module_type_policy() expects at least one module type.")
    if not all(isinstance(module_type, type) and issubclass(module_type, nn.Module) for module_type in module_types):
        raise TypeError("module_type_policy() expects nn.Module types.")

    def policy(module: nn.Module) -> bool:
        return isinstance(module, module_types)

    return policy


def size_based_policy(min_num_params: int, *, recurse: bool = False) -> WrapPolicy:
    if min_num_params < 0:
        raise ValueError("min_num_params must be non-negative.")

    def policy(module: nn.Module) -> bool:
        return sum(param.numel() for param in module.parameters(recurse=recurse)) >= min_num_params

    return policy


def or_policy(*policies: WrapPolicy) -> WrapPolicy:
    _validate_policies(policies)

    def policy(module: nn.Module) -> bool:
        return any(candidate(module) for candidate in policies)

    return policy


def and_policy(*policies: WrapPolicy) -> WrapPolicy:
    _validate_policies(policies)

    def policy(module: nn.Module) -> bool:
        return all(candidate(module) for candidate in policies)

    return policy


def not_policy(policy: WrapPolicy) -> WrapPolicy:
    if not callable(policy):
        raise TypeError("not_policy() expects a callable policy.")

    def negated(module: nn.Module) -> bool:
        return not policy(module)

    return negated


def _validate_policies(policies: tuple[WrapPolicy, ...]) -> None:
    if not policies:
        raise ValueError("Expected at least one wrap policy.")
    if not all(callable(policy) for policy in policies):
        raise TypeError("Wrap policies must be callable.")
