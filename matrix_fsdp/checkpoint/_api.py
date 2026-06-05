from __future__ import annotations

import functools
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import torch
from torch import nn


_PATCHED_STATE_DICT_CALLS: set[Any] = set()


def matrix_get_state_dict(
    module: nn.Module,
    optimizers: Any | Iterable[Any] | None = None,
    *,
    options: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return a PyTorch-DCP-style ``(model_state_dict, optim_state_dict)`` tuple.

    This is a bridge for training code that expects the shape of
    ``torch.distributed.checkpoint.state_dict.get_state_dict`` while keeping
    MatrixFSDP's layout sidecar metadata. We intentionally avoid monkeypatching
    PyTorch's private FSDP2/DCP state-dict internals by default.
    """

    from matrix_fsdp.checkpoint.state_dict import matrix_state_dict

    full_state = bool(getattr(options, "full_state_dict", False))
    cpu_offload = bool(getattr(options, "cpu_offload", False))
    model_state = matrix_state_dict(module, full_state=full_state, clone=True)
    optim_state = _optimizer_state_dict_for_dcp_style(optimizers)
    if cpu_offload:
        model_state = _move_nested_tensors_to_cpu(model_state)
        optim_state = _move_nested_tensors_to_cpu(optim_state)
    return model_state, optim_state


def get_state_dict(
    model: nn.Module,
    optimizers: Any | Iterable[Any] | None = None,
    *,
    submodules: set[nn.Module] | None = None,
    options: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    FSDP2/DCP-shaped state-dict facade for MatrixFSDP.

    The return shape matches ``torch.distributed.checkpoint.state_dict.get_state_dict``:
    ``(model_state_dict, optim_state_dict)``. MatrixFSDP-specific layout metadata
    is preserved in the model state dict.
    """

    _reject_submodules(submodules)
    return matrix_get_state_dict(model, optimizers, options=options)


def get_model_state_dict(
    model: nn.Module,
    *,
    submodules: set[nn.Module] | None = None,
    options: Any | None = None,
) -> dict[str, Any]:
    _reject_submodules(submodules)
    return matrix_get_state_dict(model, None, options=options)[0]


def get_optimizer_state_dict(
    model: nn.Module,
    optimizers: Any | Iterable[Any],
    *,
    submodules: set[nn.Module] | None = None,
    options: Any | None = None,
) -> dict[str, Any]:
    _reject_submodules(submodules)
    return matrix_get_state_dict(model, optimizers, options=options)[1]


def matrix_set_state_dict(
    module: nn.Module,
    optimizers: Any | Iterable[Any] | None = None,
    *,
    model_state_dict: dict[str, Any],
    optim_state_dict: dict[str, Any] | None = None,
    options: Any | None = None,
) -> None:
    """
    Load a state tuple returned by ``matrix_get_state_dict``.

    The ``options`` argument is accepted for API symmetry with PyTorch DCP's
    ``set_state_dict``; strictness is currently enforced by the underlying
    MatrixFSDP same-layout loader and optimizer ``load_state_dict``.
    """

    from matrix_fsdp.checkpoint.state_dict import load_matrix_state_dict

    del options
    load_matrix_state_dict(module, model_state_dict)
    if optimizers is not None and optim_state_dict is not None:
        _load_optimizer_state_dict_for_dcp_style(optimizers, optim_state_dict)


def set_state_dict(
    model: nn.Module,
    optimizers: Any | Iterable[Any] | None = None,
    *,
    model_state_dict: dict[str, Any],
    optim_state_dict: dict[str, Any] | None = None,
    options: Any | None = None,
) -> None:
    matrix_set_state_dict(
        model,
        optimizers,
        model_state_dict=model_state_dict,
        optim_state_dict=optim_state_dict,
        options=options,
    )


def set_model_state_dict(
    model: nn.Module,
    model_state_dict: dict[str, Any],
    *,
    options: Any | None = None,
) -> None:
    matrix_set_state_dict(model, None, model_state_dict=model_state_dict, options=options)


def set_optimizer_state_dict(
    model: nn.Module,
    optimizers: Any | Iterable[Any],
    optim_state_dict: dict[str, Any],
    *,
    options: Any | None = None,
) -> None:
    del model, options
    _load_optimizer_state_dict_for_dcp_style(optimizers, optim_state_dict)


def patch_model_state_dict(model: nn.Module, *, options: Any | None = None) -> nn.Module:
    """
    Patch ``model.state_dict()`` and ``model.load_state_dict()`` to use MatrixFSDP.

    This mirrors PyTorch DCP's convenience patching API at a smaller scope. It is
    intended for integration code that expects to call methods on the model
    object; explicit ``get_state_dict``/``set_state_dict`` remains the clearer
    default path.
    """

    state_dict_call = functools.partial(get_model_state_dict, model, options=options)
    load_state_dict_call = functools.partial(_patched_model_load_state_dict, model, options=options)
    model.state_dict = state_dict_call  # type: ignore[method-assign]
    model.load_state_dict = load_state_dict_call  # type: ignore[method-assign]
    _PATCHED_STATE_DICT_CALLS.add(state_dict_call)
    _PATCHED_STATE_DICT_CALLS.add(load_state_dict_call)
    return model


def patch_optimizer_state_dict(
    model: nn.Module,
    optimizers: Any | Iterable[Any],
    *,
    options: Any | None = None,
) -> Any | Iterable[Any]:
    optimizer_tuple = _as_optimizer_tuple(optimizers)
    for optimizer in optimizer_tuple:
        state_dict_call = functools.partial(get_optimizer_state_dict, model, optimizer, options=options)
        load_state_dict_call = functools.partial(
            _patched_optimizer_load_state_dict,
            model,
            optimizer,
            options=options,
        )
        optimizer.state_dict = state_dict_call  # type: ignore[method-assign]
        optimizer.load_state_dict = load_state_dict_call  # type: ignore[method-assign]
        _PATCHED_STATE_DICT_CALLS.add(state_dict_call)
        _PATCHED_STATE_DICT_CALLS.add(load_state_dict_call)
    return optimizers


def _patched_model_load_state_dict(
    model: nn.Module,
    state_dict: dict[str, Any],
    strict: bool = True,
    assign: bool = False,
    *,
    options: Any | None = None,
):
    del assign
    if not strict:
        options = _replace_options_strict(options, strict=False)
    set_model_state_dict(model, state_dict, options=options)
    return torch.nn.modules.module._IncompatibleKeys([], [])


def _patched_optimizer_load_state_dict(
    model: nn.Module,
    optimizer: Any,
    state_dict: dict[str, Any],
    *,
    options: Any | None = None,
) -> None:
    set_optimizer_state_dict(model, optimizer, state_dict, options=options)


def _replace_options_strict(options: Any | None, *, strict: bool) -> Any:
    if options is None:
        try:
            from torch.distributed.checkpoint.state_dict import StateDictOptions

            return StateDictOptions(strict=strict)
        except Exception:
            return options
    if hasattr(options, "strict"):
        options = deepcopy(options)
        options.strict = strict
    return options


def _reject_submodules(submodules: set[nn.Module] | None) -> None:
    if submodules:
        raise NotImplementedError("MatrixFSDP state_dict submodule filtering is not supported yet.")


def _optimizer_state_dict_for_dcp_style(optimizers: Any | Iterable[Any] | None) -> dict[str, Any]:
    if optimizers is None:
        return {}
    optimizer_tuple = _as_optimizer_tuple(optimizers)
    if len(optimizer_tuple) == 1:
        return _state_dict_fn(optimizer_tuple[0], "state_dict")()
    return {str(index): _state_dict_fn(optimizer, "state_dict")() for index, optimizer in enumerate(optimizer_tuple)}


def _load_optimizer_state_dict_for_dcp_style(optimizers: Any | Iterable[Any], state_dict: dict[str, Any]) -> None:
    optimizer_tuple = _as_optimizer_tuple(optimizers)
    if len(optimizer_tuple) == 1:
        _state_dict_fn(optimizer_tuple[0], "load_state_dict")(state_dict)
        refresh = getattr(optimizer_tuple[0], "refresh_state_dtensors", None)
        if refresh is not None:
            refresh()
        return
    for index, optimizer in enumerate(optimizer_tuple):
        key = str(index)
        if key not in state_dict:
            raise ValueError(f"Missing optimizer state for optimizer index {index}.")
        _state_dict_fn(optimizer, "load_state_dict")(state_dict[key])
        refresh = getattr(optimizer, "refresh_state_dtensors", None)
        if refresh is not None:
            refresh()


def _as_optimizer_tuple(optimizers: Any | Iterable[Any]) -> tuple[Any, ...]:
    if hasattr(optimizers, "state_dict") and hasattr(optimizers, "load_state_dict"):
        return (optimizers,)
    return tuple(optimizers)


def _state_dict_fn(obj: Any, api: str):
    call = getattr(obj, api)
    if call in _PATCHED_STATE_DICT_CALLS:
        return functools.partial(getattr(obj.__class__, api), obj)
    return call


def _move_nested_tensors_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {key: _move_nested_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_nested_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors_to_cpu(item) for item in value)
    return value
