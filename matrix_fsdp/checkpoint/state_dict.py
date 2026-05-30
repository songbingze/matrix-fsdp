from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import nn

from matrix_fsdp.runtime.collectives import all_gather_matrix_shard_1d_async
from matrix_fsdp.runtime.flat_buffer import MatrixFlatBuffer
from matrix_fsdp.runtime.param_group import FSDPLifecycleState, MatrixFSDPParamGroup
from matrix_fsdp.core.managed_param import ManagedParam
from matrix_fsdp.core.mixed_precision import mixed_precision_policy_metadata, offload_policy_metadata
from matrix_fsdp.optim.state import MatrixFSDPOptimizerStateManager
from matrix_fsdp.checkpoint._api import (
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    patch_model_state_dict,
    patch_optimizer_state_dict,
    matrix_get_state_dict,
    matrix_set_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    set_state_dict,
)
from matrix_fsdp.checkpoint._dcp_utils import (
    _dcp_payload_rank,
    _dcp_rank,
    _dcp_tensor_key,
    _infer_no_dist,
    _load_all_dcp_metadata,
    _load_dcp_metadata_for_rank,
    _require_dcp,
    _save_dcp_metadata,
)
from matrix_fsdp.checkpoint._metadata import (
    _canonical_metadata,
    _checkpoint_tensor,
    _dtype_from_string,
    _layout_segment_metadata,
    _layout_metadata,
    _param_key,
    _param_metadata,
    _matrix_shard_metadata,
)
from matrix_fsdp.checkpoint._optimizer_utils import (
    _optimizer_components,
    _optimizer_group_summary_metadata,
    _optimizer_kind,
    _optimizer_param_device,
    _optimizer_param_groups_for_reshard,
    _unwrap_optimizer,
    _validate_optimizer_param_groups,
)


MATRIX_STATE_DICT_VERSION = 1
MATRIX_DCP_FORMAT_VERSION = 1


def save_matrix_dcp(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    checkpoint_id: str | PathLike[str],
    *,
    optimizer: Any | None = None,
    full_state: bool = False,
    include_grads: bool = False,
    dedup_replicates: bool = True,
    process_group=None,
    no_dist: bool | None = None,
) -> Any:
    """
    Save local MatrixFSDP shards through PyTorch Distributed Checkpoint.

    Tensor payloads go through DCP. MatrixFSDP layout metadata is written as one
    global metadata file, with legacy per-rank sidecar loading kept for older
    checkpoints.
    """
    dcp = _require_dcp()
    no_dist = _infer_no_dist(no_dist)
    dcp_rank = _dcp_rank(process_group, no_dist)
    checkpoint_path = Path(checkpoint_id)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    tensor_state, metadata = _dcp_tensor_state_and_metadata(
        module_or_units,
        dcp_rank=dcp_rank,
        full_state=full_state,
        include_grads=include_grads,
        dedup_replicates=dedup_replicates,
    )
    if optimizer is not None:
        optimizer_tensor_state, optimizer_metadata = _dcp_optimizer_tensor_state_and_metadata(
            optimizer,
            module_or_units,
            dcp_rank=dcp_rank,
            dedup_replicates=dedup_replicates,
        )
        tensor_state.update(optimizer_tensor_state)
        metadata["optimizer"] = optimizer_metadata
    _save_dcp_metadata(
        checkpoint_path,
        metadata,
        process_group=process_group,
        no_dist=no_dist,
    )
    return dcp.save(
        tensor_state,
        checkpoint_id=checkpoint_path,
        process_group=process_group,
        no_dist=no_dist,
    )


def load_matrix_dcp(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    checkpoint_id: str | PathLike[str],
    *,
    optimizer: Any | None = None,
    allow_reshard: bool = False,
    process_group=None,
    no_dist: bool | None = None,
) -> None:
    """
    Load a same-world, same-layout MatrixFSDP checkpoint saved by DCP.
    """
    dcp = _require_dcp()
    no_dist = _infer_no_dist(no_dist)
    dcp_rank = _dcp_rank(process_group, no_dist)
    checkpoint_path = Path(checkpoint_id)
    try:
        metadata = _load_dcp_metadata_for_rank(checkpoint_path, dcp_rank)
    except FileNotFoundError:
        if not allow_reshard:
            raise
        _load_matrix_dcp_resharded(
            module_or_units,
            checkpoint_path,
            dcp=dcp,
            process_group=process_group,
            no_dist=no_dist,
            optimizer=optimizer,
        )
        return

    if allow_reshard and _metadata_requires_reshard(module_or_units, metadata):
        _load_matrix_dcp_resharded(
            module_or_units,
            checkpoint_path,
            dcp=dcp,
            process_group=process_group,
            no_dist=no_dist,
            optimizer=optimizer,
        )
        return
    tensor_state, matrix_state = _dcp_load_tensor_state_and_matrix_state(
        module_or_units,
        metadata,
        dcp_rank=dcp_rank,
    )
    optimizer_load_plan = None
    if optimizer is not None:
        optimizer_load_plan = _dcp_optimizer_load_plan(optimizer, module_or_units, metadata)
        tensor_state.update(optimizer_load_plan["tensor_state"])
    dcp.load(
        tensor_state,
        checkpoint_id=checkpoint_path,
        process_group=process_group,
        no_dist=no_dist,
    )
    load_matrix_state_dict(module_or_units, matrix_state)
    if optimizer_load_plan is not None:
        _load_optimizer_from_dcp_plan(optimizer, optimizer_load_plan)


def load_matrix_dcp_full_state(
    checkpoint_id: str | PathLike[str],
    *,
    source_rank: int = 0,
    process_group=None,
    no_dist: bool | None = None,
) -> dict[str, Any]:
    """
    Load debug full parameters saved by ``save_matrix_dcp(full_state=True)``.
    """
    dcp = _require_dcp()
    no_dist = _infer_no_dist(no_dist)
    checkpoint_path = Path(checkpoint_id)
    metadata = _load_dcp_metadata_for_rank(checkpoint_path, source_rank)

    tensor_state: dict[str, torch.Tensor] = {}
    params: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    for unit_metadata in _param_group_state_list(metadata, context="DCP metadata"):
        module_fqn = unit_metadata["module_fqn"]
        params_metadata = unit_metadata.get("params", {})
        for fqn, tensor_key in unit_metadata.get("full_param_keys", {}).items():
            param_metadata = params_metadata[fqn]
            tensor = torch.empty(
                tuple(param_metadata["shape"]),
                dtype=_dtype_from_string(param_metadata["dtype"]),
            )
            tensor_state[tensor_key] = tensor
            params[_param_key(module_fqn, fqn, unit_metadata["unit_index"])] = tensor
        for fqn, tensor_key in unit_metadata.get("full_grad_keys", {}).items():
            param_metadata = params_metadata[fqn]
            tensor = torch.empty(
                tuple(param_metadata["shape"]),
                dtype=_dtype_from_string(param_metadata["dtype"]),
            )
            tensor_state[tensor_key] = tensor
            grads[_param_key(module_fqn, fqn, unit_metadata["unit_index"])] = tensor
    if not params:
        raise ValueError("This MatrixFSDP DCP checkpoint does not contain full_state tensors.")
    dcp.load(
        tensor_state,
        checkpoint_id=checkpoint_path,
        process_group=process_group,
        no_dist=no_dist,
    )
    result = {
        "metadata": metadata.get("metadata", {}),
        "params": params,
    }
    if grads:
        result["grads"] = grads
    return result


def matrix_state_dict(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    *,
    full_state: bool = False,
    include_grads: bool = False,
    clone: bool = True,
) -> dict[str, Any]:
    """
    Return a checkpoint-oriented MatrixFSDP state dict.

    The default form stores each unit's local parameter shard plus plain layout
    metadata. ``full_state=True`` adds debug-friendly full parameter tensors
    assembled from the current local shards, while keeping the local shards so
    same-layout load remains possible.
    """
    param_group_refs = _collect_param_group_refs(module_or_units)
    state: dict[str, Any] = {
        "metadata": {
            "version": MATRIX_STATE_DICT_VERSION,
            "state_dict_type": "matrix_full" if full_state else "matrix_sharded",
            "num_param_groups": len(param_group_refs),
            "num_units": len(param_group_refs),
            "include_grads": include_grads,
        },
        "param_groups": [],
        "units": [],
    }
    if full_state:
        state["params"] = {}
        if include_grads:
            state["grads"] = {}

    for param_group_index, (module_fqn, param_group) in enumerate(param_group_refs):
        param_group_state = _unit_state_dict(
            param_group,
            unit_index=param_group_index,
            module_fqn=module_fqn,
            include_grads=include_grads,
            full_state=full_state,
            clone=clone,
        )
        state["param_groups"].append(param_group_state)
        state["units"].append(param_group_state)
        if full_state:
            for fqn, tensor in param_group_state["full_params"].items():
                state["params"][_param_key(module_fqn, fqn, param_group_index)] = tensor
            if include_grads:
                for fqn, tensor in param_group_state.get("full_grads", {}).items():
                    state["grads"][_param_key(module_fqn, fqn, param_group_index)] = tensor
    return state


def load_matrix_state_dict(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    state_dict: dict[str, Any],
) -> None:
    """
    Load a same-world, same-layout MatrixFSDP state dict.

    This intentionally starts with the conservative case: each rank loads the
    local shard saved for the same rank/layout. World-size-changing reshard
    loads can build on the plain metadata emitted here.
    """
    metadata = state_dict.get("metadata", {})
    if metadata.get("version") != MATRIX_STATE_DICT_VERSION:
        raise ValueError(
            f"Unsupported MatrixFSDP state_dict version {metadata.get('version')!r}; "
            f"expected {MATRIX_STATE_DICT_VERSION}."
        )

    param_group_refs = _collect_param_group_refs(module_or_units)
    param_group_states = _param_group_state_list(state_dict, context="state_dict")
    if len(param_group_refs) != len(param_group_states):
        raise ValueError(
            f"State dict has {len(param_group_states)} param groups, but target has {len(param_group_refs)} param groups."
        )

    for (module_fqn, param_group), param_group_state in zip(param_group_refs, param_group_states):
        _load_unit_state_dict(param_group, param_group_state, module_fqn=module_fqn)


def _collect_param_group_refs(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
) -> list[tuple[str, MatrixFSDPParamGroup]]:
    if isinstance(module_or_units, MatrixFSDPParamGroup):
        return [("", module_or_units)]
    if isinstance(module_or_units, nn.Module):
        unit_refs: list[tuple[str, MatrixFSDPParamGroup]] = []
        seen: set[int] = set()
        for module_fqn, module in module_or_units.named_modules(remove_duplicate=True):
            unit = getattr(module, "_matrix_fsdp_param_group", None)
            if unit is None:
                continue
            if not isinstance(unit, MatrixFSDPParamGroup):
                raise TypeError(f"Expected MatrixFSDPParamGroup, got {type(unit)!r}.")
            unit_id = id(unit)
            if unit_id in seen:
                continue
            seen.add(unit_id)
            unit_refs.append((module_fqn, unit))
        return unit_refs

    unit_refs = []
    for index, unit in enumerate(module_or_units):
        if not isinstance(unit, MatrixFSDPParamGroup):
            raise TypeError(f"Expected MatrixFSDPParamGroup, got {type(unit)!r}.")
        unit_refs.append((f"unit{index}", unit))
    return unit_refs


_collect_unit_refs = _collect_param_group_refs


def _param_group_state_list(state: Mapping[str, Any], *, context: str) -> list[dict[str, Any]]:
    param_group_states = state.get("param_groups")
    if param_group_states is None:
        param_group_states = state.get("units")
    if not isinstance(param_group_states, list):
        raise TypeError(f"MatrixFSDP {context} must contain a list at key 'param_groups' or legacy key 'units'.")
    return param_group_states


def _dcp_tensor_state_and_metadata(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    *,
    dcp_rank: int,
    full_state: bool,
    include_grads: bool,
    dedup_replicates: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    state = matrix_state_dict(module_or_units, full_state=full_state, include_grads=include_grads, clone=False)
    tensor_state: dict[str, torch.Tensor] = {}
    metadata_units = []
    for unit_state in state["param_groups"]:
        unit_index = unit_state["unit_index"]
        key_rank = _dcp_payload_rank(unit_state, dcp_rank=dcp_rank, dedup_replicates=dedup_replicates)
        param_key = _dcp_tensor_key(key_rank, unit_index, "param_shard")
        tensor_state[param_key] = unit_state["param_shard"]

        unit_metadata = dict(unit_state)
        unit_metadata.pop("param_shard")
        unit_metadata["dcp_payload_rank"] = key_rank
        unit_metadata["dcp_dedup_replicates"] = key_rank != f"rank_{dcp_rank}"
        unit_metadata["param_shard_key"] = param_key

        full_params = unit_metadata.pop("full_params", {})
        unit_metadata["full_param_keys"] = {}
        for fqn, tensor in full_params.items():
            full_param_key = _dcp_tensor_key(key_rank, unit_index, f"full_param.{fqn}")
            tensor_state[full_param_key] = tensor
            unit_metadata["full_param_keys"][fqn] = full_param_key

        grad_shard = unit_metadata.pop("grad_shard", None)
        if torch.is_tensor(grad_shard):
            grad_key = _dcp_tensor_key(key_rank, unit_index, "grad_shard")
            tensor_state[grad_key] = grad_shard
            unit_metadata["grad_shard_key"] = grad_key
        else:
            unit_metadata["grad_shard_key"] = None

        full_grads = unit_metadata.pop("full_grads", {})
        unit_metadata["full_grad_keys"] = {}
        for fqn, tensor in full_grads.items():
            full_grad_key = _dcp_tensor_key(key_rank, unit_index, f"full_grad.{fqn}")
            tensor_state[full_grad_key] = tensor
            unit_metadata["full_grad_keys"][fqn] = full_grad_key
        metadata_units.append(unit_metadata)

    metadata = {
        "metadata": {
            "version": MATRIX_STATE_DICT_VERSION,
            "dcp_format_version": MATRIX_DCP_FORMAT_VERSION,
            "state_dict_type": "matrix_dcp_sharded",
            "num_param_groups": len(metadata_units),
            "num_units": len(metadata_units),
            "full_state": full_state,
            "include_grads": include_grads,
            "dedup_replicates": dedup_replicates,
            "dcp_rank": dcp_rank,
        },
        "param_groups": metadata_units,
        "units": metadata_units,
    }
    return tensor_state, metadata


def _dcp_load_tensor_state_and_matrix_state(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    metadata: dict[str, Any],
    *,
    dcp_rank: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_metadata = metadata.get("metadata", {})
    if checkpoint_metadata.get("version") != MATRIX_STATE_DICT_VERSION:
        raise ValueError(
            f"Unsupported MatrixFSDP state_dict version {checkpoint_metadata.get('version')!r}; "
            f"expected {MATRIX_STATE_DICT_VERSION}."
        )
    if checkpoint_metadata.get("dcp_format_version") != MATRIX_DCP_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported MatrixFSDP DCP format version "
            f"{checkpoint_metadata.get('dcp_format_version')!r}; expected {MATRIX_DCP_FORMAT_VERSION}."
        )
    if checkpoint_metadata.get("dcp_rank") != dcp_rank:
        raise ValueError(
            f"Loaded metadata for DCP rank {checkpoint_metadata.get('dcp_rank')}, "
            f"but current DCP rank is {dcp_rank}."
        )

    param_group_refs = _collect_param_group_refs(module_or_units)
    metadata_units = _param_group_state_list(metadata, context="DCP metadata")
    if len(param_group_refs) != len(metadata_units):
        raise ValueError(
            f"DCP metadata has {len(metadata_units)} param groups, but target has {len(param_group_refs)} param groups."
        )

    tensor_state: dict[str, torch.Tensor] = {}
    matrix_units = []
    for (module_fqn, unit), unit_metadata in zip(param_group_refs, metadata_units):
        flat_buffer = _require_flat_buffer(unit)
        _validate_same_layout_unit(unit, flat_buffer, unit_metadata, module_fqn=module_fqn)
        unit_state = dict(unit_metadata)
        param_key = unit_state.pop("param_shard_key")
        tensor_state[param_key] = flat_buffer.local_shard
        unit_state["param_shard"] = flat_buffer.local_shard

        grad_key = unit_state.pop("grad_shard_key", None)
        if grad_key is not None:
            grad_shard = flat_buffer.local_shard.new_empty(flat_buffer.local_numel)
            tensor_state[grad_key] = grad_shard
            unit_state["grad_shard"] = grad_shard
        matrix_units.append(unit_state)

    return tensor_state, {
        "metadata": {
            "version": MATRIX_STATE_DICT_VERSION,
            "state_dict_type": "matrix_sharded",
            "num_param_groups": len(matrix_units),
            "num_units": len(matrix_units),
            "include_grads": checkpoint_metadata.get("include_grads", False),
        },
        "param_groups": matrix_units,
        "units": matrix_units,
    }


def _dcp_optimizer_tensor_state_and_metadata(
    optimizer: Any,
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    *,
    dcp_rank: int,
    dedup_replicates: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    components = _optimizer_components(optimizer)
    state_layout_metadata = _optimizer_state_layout_metadata(optimizer)
    payload_rank_by_fqn = _dcp_payload_rank_by_fqn(
        module_or_units,
        dcp_rank=dcp_rank,
        dedup_replicates=dedup_replicates,
    )
    tensor_state: dict[str, torch.Tensor] = {}
    component_metadata: dict[str, dict[str, Any]] = {}

    for component_name, component_optimizer in components:
        optimizer_state_dict = component_optimizer.state_dict()
        param_fqn_by_state_id = _optimizer_param_fqns_by_state_id(component_optimizer, module_or_units)
        state_metadata: dict[int, dict[str, Any]] = {}

        for state_id, state in optimizer_state_dict["state"].items():
            fqn = param_fqn_by_state_id.get(state_id)
            if fqn is None:
                continue
            entries = {}
            for name, value in state.items():
                if torch.is_tensor(value):
                    key_rank = payload_rank_by_fqn.get(fqn, f"rank_{dcp_rank}")
                    tensor_key = _dcp_tensor_key(key_rank, 0, f"optimizer.{component_name}.{fqn}.{name}")
                    tensor_state[tensor_key] = value.detach()
                    entries[name] = {
                        "kind": "tensor",
                        "key": tensor_key,
                        "shape": tuple(value.shape),
                        "dtype": str(value.dtype),
                        "requires_grad": value.requires_grad,
                    }
                    matrix_state_metadata = state_layout_metadata.get(fqn, {}).get(name)
                    if matrix_state_metadata is not None:
                        entries[name]["matrix_state"] = matrix_state_metadata
                else:
                    entries[name] = {
                        "kind": "value",
                        "value": deepcopy(value),
                    }
            state_metadata[state_id] = {
                "fqn": fqn,
                "entries": entries,
            }

        component_metadata[component_name] = {
            "class_name": type(component_optimizer).__name__,
            "param_groups": deepcopy(optimizer_state_dict["param_groups"]),
            "state": state_metadata,
        }

    metadata = {
        "optimizer_format_version": 2,
        "kind": _optimizer_kind(optimizer),
        "group_summary": _optimizer_group_summary_metadata(optimizer),
        "components": component_metadata,
    }
    if len(components) == 1 and components[0][0] == "optimizer":
        # Keep the original metadata shape for legacy tests and checkpoints.
        legacy_component = component_metadata["optimizer"]
        metadata["param_groups"] = deepcopy(legacy_component["param_groups"])
        metadata["state"] = legacy_component["state"]
    return tensor_state, metadata


def _optimizer_state_layout_metadata(optimizer: Any) -> dict[str, dict[str, dict[str, Any]]]:
    state_manager = getattr(optimizer, "state_manager", None)
    if isinstance(state_manager, MatrixFSDPOptimizerStateManager):
        return state_manager.local_state_metadata()
    return {}


def _dcp_optimizer_load_plan(
    optimizer: Any,
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    optimizer_metadata = metadata.get("optimizer")
    if optimizer_metadata is None:
        raise ValueError("MatrixFSDP DCP checkpoint does not contain optimizer state.")
    if "components" in optimizer_metadata:
        return _dcp_optimizer_components_load_plan(optimizer, module_or_units, optimizer_metadata)

    torch_optimizer = _unwrap_optimizer(optimizer)
    current_state_dict = torch_optimizer.state_dict()
    saved_param_groups = deepcopy(optimizer_metadata["param_groups"])
    _validate_optimizer_param_groups(current_state_dict["param_groups"], saved_param_groups)

    current_param_fqns_by_state_id = _optimizer_param_fqns_by_state_id(torch_optimizer, module_or_units)
    current_state_id_by_fqn = {fqn: state_id for state_id, fqn in current_param_fqns_by_state_id.items()}
    tensor_state: dict[str, torch.Tensor] = {}
    loaded_state: dict[int, dict[str, Any]] = {}

    for saved_state_id, state_metadata in optimizer_metadata["state"].items():
        fqn = state_metadata["fqn"]
        current_state_id = current_state_id_by_fqn.get(fqn)
        if current_state_id is None:
            raise ValueError(f"Optimizer state references unknown local parameter {fqn!r}.")
        loaded_entries: dict[str, Any] = {}
        for name, entry in state_metadata["entries"].items():
            if entry["kind"] == "tensor":
                tensor = torch.empty(
                    tuple(entry["shape"]),
                    dtype=_dtype_from_string(entry["dtype"]),
                    device=_optimizer_param_device(torch_optimizer, current_state_id),
                )
                tensor_state[entry["key"]] = tensor
                loaded_entries[name] = tensor
            else:
                loaded_entries[name] = deepcopy(entry["value"])
        loaded_state[current_state_id] = loaded_entries

    return {
        "tensor_state": tensor_state,
        "optimizer_state_dict": {
            "state": loaded_state,
            "param_groups": saved_param_groups,
        },
    }


def _dcp_optimizer_components_load_plan(
    optimizer: Any,
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    optimizer_metadata: dict[str, Any],
) -> dict[str, Any]:
    components = dict(_optimizer_components(optimizer))
    saved_components = optimizer_metadata.get("components", {})
    missing_components = sorted(set(components) - set(saved_components))
    if missing_components:
        raise ValueError(
            "Optimizer checkpoint is missing component(s) "
            f"{missing_components}; available components are {sorted(saved_components)}."
        )
    extra_components = sorted(set(saved_components) - set(components))
    if extra_components:
        raise ValueError(
            "Optimizer checkpoint contains unknown component(s) "
            f"{extra_components}; target components are {sorted(components)}."
        )
    tensor_state: dict[str, torch.Tensor] = {}
    component_plans: dict[str, dict[str, Any]] = {}

    for component_name, component_metadata in saved_components.items():
        component_optimizer = components[component_name]
        current_state_dict = component_optimizer.state_dict()
        saved_param_groups = deepcopy(component_metadata["param_groups"])
        _validate_optimizer_param_groups(current_state_dict["param_groups"], saved_param_groups)

        current_param_fqns_by_state_id = _optimizer_param_fqns_by_state_id(component_optimizer, module_or_units)
        current_state_id_by_fqn = {fqn: state_id for state_id, fqn in current_param_fqns_by_state_id.items()}
        loaded_state: dict[int, dict[str, Any]] = {}

        for saved_state_id, state_metadata in component_metadata["state"].items():
            fqn = state_metadata["fqn"]
            current_state_id = current_state_id_by_fqn.get(fqn)
            if current_state_id is None:
                raise ValueError(
                    f"Optimizer component {component_name!r} references unknown local parameter {fqn!r}."
                )
            loaded_entries: dict[str, Any] = {}
            for name, entry in state_metadata["entries"].items():
                if entry["kind"] == "tensor":
                    tensor = torch.empty(
                        tuple(entry["shape"]),
                        dtype=_dtype_from_string(entry["dtype"]),
                        device=_optimizer_param_device(component_optimizer, current_state_id),
                    )
                    tensor_state[entry["key"]] = tensor
                    loaded_entries[name] = tensor
                else:
                    loaded_entries[name] = deepcopy(entry["value"])
            loaded_state[current_state_id] = loaded_entries

        component_plans[component_name] = {
            "optimizer_state_dict": {
                "state": loaded_state,
                "param_groups": saved_param_groups,
            },
        }

    return {
        "tensor_state": tensor_state,
        "components": component_plans,
    }


def _load_optimizer_from_dcp_plan(optimizer: Any, optimizer_load_plan: dict[str, Any]) -> None:
    if "components" in optimizer_load_plan:
        components = dict(_optimizer_components(optimizer))
        for component_name, component_plan in optimizer_load_plan["components"].items():
            component_optimizer = components.get(component_name)
            if component_optimizer is None:
                raise ValueError(f"Optimizer load plan contains unknown component {component_name!r}.")
            component_optimizer.load_state_dict(component_plan["optimizer_state_dict"])
        if hasattr(optimizer, "refresh_state_dtensors"):
            optimizer.refresh_state_dtensors()
        return

    torch_optimizer = _unwrap_optimizer(optimizer)
    torch_optimizer.load_state_dict(optimizer_load_plan["optimizer_state_dict"])
    if hasattr(optimizer, "refresh_state_dtensors"):
        optimizer.refresh_state_dtensors()


def _dcp_optimizer_reshard_load_plan(
    optimizer: Any,
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    source_metadata_by_rank: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    first_optimizer_metadata = next(iter(source_metadata_by_rank.values())).get("optimizer")
    if first_optimizer_metadata is None:
        raise ValueError("MatrixFSDP DCP checkpoint does not contain optimizer state.")
    if "components" in first_optimizer_metadata:
        return _dcp_optimizer_components_reshard_load_plan(optimizer, module_or_units, source_metadata_by_rank)

    torch_optimizer = _unwrap_optimizer(optimizer)
    current_state_dict = torch_optimizer.state_dict()
    saved_param_groups = deepcopy(first_optimizer_metadata["param_groups"])
    _validate_optimizer_param_groups(current_state_dict["param_groups"], saved_param_groups)
    current_state_id_by_fqn = {
        fqn: state_id
        for state_id, fqn in _optimizer_param_fqns_by_state_id(torch_optimizer, module_or_units).items()
    }
    target_param_by_fqn = _target_param_by_full_fqn(module_or_units)

    tensor_state: dict[str, torch.Tensor] = {}
    entries_by_fqn: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]] = {}
    values_by_fqn: dict[str, dict[str, Any]] = {}
    for source_rank, metadata in source_metadata_by_rank.items():
        optimizer_metadata = metadata.get("optimizer")
        if optimizer_metadata is None:
            continue
        for state_metadata in optimizer_metadata["state"].values():
            fqn = state_metadata["fqn"]
            entries_by_name = entries_by_fqn.setdefault(fqn, {})
            values_by_name = values_by_fqn.setdefault(fqn, {})
            for name, entry in state_metadata["entries"].items():
                if entry["kind"] == "tensor":
                    tensor = torch.empty(tuple(entry["shape"]), dtype=_dtype_from_string(entry["dtype"]))
                    tensor_state[entry["key"]] = tensor
                    entries_by_name.setdefault(name, []).append((source_rank, entry))
                elif name not in values_by_name:
                    values_by_name[name] = deepcopy(entry["value"])

    return {
        "tensor_state": tensor_state,
        "param_groups": saved_param_groups,
        "current_state_id_by_fqn": current_state_id_by_fqn,
        "target_param_by_fqn": target_param_by_fqn,
        "entries_by_fqn": entries_by_fqn,
        "values_by_fqn": values_by_fqn,
    }


def _dcp_optimizer_components_reshard_load_plan(
    optimizer: Any,
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    source_metadata_by_rank: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    first_optimizer_metadata = next(iter(source_metadata_by_rank.values())).get("optimizer")
    components = dict(_optimizer_components(optimizer))
    tensor_state: dict[str, torch.Tensor] = {}
    component_plans: dict[str, dict[str, Any]] = {}

    for component_name, component_metadata in first_optimizer_metadata["components"].items():
        component_optimizer = components.get(component_name)
        if component_optimizer is None:
            continue
        component_source_metadata: dict[int, dict[str, Any]] = {}
        for source_rank, metadata in source_metadata_by_rank.items():
            optimizer_metadata = metadata.get("optimizer")
            if optimizer_metadata is None:
                continue
            source_component_metadata = optimizer_metadata.get("components", {}).get(component_name)
            if source_component_metadata is not None:
                component_source_metadata[source_rank] = {"optimizer": source_component_metadata}
        component_plan = _dcp_optimizer_component_reshard_load_plan(
            component_optimizer,
            module_or_units,
            component_source_metadata,
        )
        tensor_state.update(component_plan["tensor_state"])
        component_plans[component_name] = component_plan

    return {
        "tensor_state": tensor_state,
        "components": component_plans,
    }


def _dcp_optimizer_component_reshard_load_plan(
    torch_optimizer: Any,
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    source_metadata_by_rank: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    first_optimizer_metadata = next(iter(source_metadata_by_rank.values())).get("optimizer")
    if first_optimizer_metadata is None:
        raise ValueError("MatrixFSDP DCP checkpoint does not contain optimizer state.")

    current_state_dict = torch_optimizer.state_dict()
    saved_param_groups = deepcopy(first_optimizer_metadata["param_groups"])
    saved_param_groups = _optimizer_param_groups_for_reshard(
        current_state_dict["param_groups"],
        saved_param_groups,
    )
    current_state_id_by_fqn = {
        fqn: state_id
        for state_id, fqn in _optimizer_param_fqns_by_state_id(torch_optimizer, module_or_units).items()
    }
    target_param_by_fqn = _target_param_by_full_fqn(module_or_units)

    tensor_state: dict[str, torch.Tensor] = {}
    entries_by_fqn: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]] = {}
    values_by_fqn: dict[str, dict[str, Any]] = {}
    for source_rank, metadata in source_metadata_by_rank.items():
        optimizer_metadata = metadata.get("optimizer")
        if optimizer_metadata is None:
            continue
        for state_metadata in optimizer_metadata["state"].values():
            fqn = state_metadata["fqn"]
            entries_by_name = entries_by_fqn.setdefault(fqn, {})
            values_by_name = values_by_fqn.setdefault(fqn, {})
            for name, entry in state_metadata["entries"].items():
                if entry["kind"] == "tensor":
                    tensor = torch.empty(tuple(entry["shape"]), dtype=_dtype_from_string(entry["dtype"]))
                    tensor_state[entry["key"]] = tensor
                    entries_by_name.setdefault(name, []).append((source_rank, entry))
                elif name not in values_by_name:
                    values_by_name[name] = deepcopy(entry["value"])

    return {
        "tensor_state": tensor_state,
        "param_groups": saved_param_groups,
        "current_state_id_by_fqn": current_state_id_by_fqn,
        "target_param_by_fqn": target_param_by_fqn,
        "entries_by_fqn": entries_by_fqn,
        "values_by_fqn": values_by_fqn,
    }


def _load_optimizer_from_reshard_plan(optimizer: Any, optimizer_reshard_plan: dict[str, Any]) -> None:
    if "components" in optimizer_reshard_plan:
        components = dict(_optimizer_components(optimizer))
        for component_name, component_plan in optimizer_reshard_plan["components"].items():
            component_optimizer = components.get(component_name)
            if component_optimizer is None:
                raise ValueError(f"Optimizer reshard plan contains unknown component {component_name!r}.")
            _load_optimizer_component_from_reshard_plan(component_optimizer, component_plan)
        if hasattr(optimizer, "refresh_state_dtensors"):
            optimizer.refresh_state_dtensors()
        return

    torch_optimizer = _unwrap_optimizer(optimizer)
    _load_optimizer_component_from_reshard_plan(torch_optimizer, optimizer_reshard_plan)
    if hasattr(optimizer, "refresh_state_dtensors"):
        optimizer.refresh_state_dtensors()


def _load_optimizer_component_from_reshard_plan(torch_optimizer: Any, optimizer_reshard_plan: dict[str, Any]) -> None:
    loaded_state: dict[int, dict[str, Any]] = {}
    for fqn, current_state_id in optimizer_reshard_plan["current_state_id_by_fqn"].items():
        target_param_info = optimizer_reshard_plan["target_param_by_fqn"].get(fqn)
        if target_param_info is None:
            continue
        unit, managed_param = target_param_info
        state_entries: dict[str, Any] = {}
        for name, value in optimizer_reshard_plan["values_by_fqn"].get(fqn, {}).items():
            state_entries[name] = deepcopy(value)
        for name, ranked_entries in optimizer_reshard_plan["entries_by_fqn"].get(fqn, {}).items():
            ranked_entries = sorted(ranked_entries, key=lambda item: item[0])
            tensors = [optimizer_reshard_plan["tensor_state"][entry["key"]] for _, entry in ranked_entries]
            if not tensors:
                continue
            if tensors[0].ndim == 0:
                state_entries[name] = tensors[0].to(managed_param.param.device)
                continue
            full_tensor = torch.cat([tensor.reshape(-1) for tensor in tensors])
            if full_tensor.numel() == managed_param.numel:
                state_entries[name] = _target_local_param_tensor_from_full(unit, managed_param, full_tensor)
            else:
                state_entries[name] = tensors[0].to(managed_param.param.device)
        if state_entries:
            loaded_state[current_state_id] = state_entries
    torch_optimizer.load_state_dict(
        {
            "state": loaded_state,
            "param_groups": optimizer_reshard_plan["param_groups"],
        }
    )


def _load_matrix_dcp_resharded(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    checkpoint_path: Path,
    *,
    dcp,
    process_group,
    no_dist: bool,
    optimizer: Any | None,
) -> None:
    source_metadata_by_rank = _load_all_dcp_metadata(checkpoint_path)
    source_world_size = _source_world_size(source_metadata_by_rank)
    param_group_refs = _collect_param_group_refs(module_or_units)
    tensor_state: dict[str, torch.Tensor] = {}
    source_rank_unit_metadata: list[list[dict[str, Any]]] = []

    for source_rank in range(source_world_size):
        metadata = source_metadata_by_rank[source_rank]
        units_metadata = _param_group_state_list(metadata, context=f"source rank {source_rank} DCP metadata")
        if len(units_metadata) != len(param_group_refs):
            raise ValueError(
                f"Source rank {source_rank} has {len(units_metadata)} param groups, "
                f"but target has {len(param_group_refs)} param groups."
            )
        source_rank_unit_metadata.append(units_metadata)
        for (_, target_unit), unit_metadata in zip(param_group_refs, units_metadata):
            flat_buffer = _require_flat_buffer(target_unit)
            shard_size = int(unit_metadata["shard_sizes"][source_rank])
            param_key = unit_metadata["param_shard_key"]
            tensor_state[param_key] = flat_buffer.local_shard.new_empty(shard_size)
            grad_key = unit_metadata.get("grad_shard_key")
            if grad_key is not None:
                tensor_state[grad_key] = flat_buffer.local_shard.new_empty(shard_size)

    optimizer_reshard_plan = None
    if optimizer is not None:
        optimizer_reshard_plan = _dcp_optimizer_reshard_load_plan(
            optimizer,
            module_or_units,
            source_metadata_by_rank,
        )
        tensor_state.update(optimizer_reshard_plan["tensor_state"])

    dcp.load(
        tensor_state,
        checkpoint_id=checkpoint_path,
        process_group=process_group,
        no_dist=no_dist,
    )

    for unit_index, (_, target_unit) in enumerate(param_group_refs):
        flat_buffer = _require_flat_buffer(target_unit)
        source_unit_metadatas = [rank_metadata[unit_index] for rank_metadata in source_rank_unit_metadata]
        _validate_reshard_unit_metadata(target_unit, source_unit_metadatas)
        full_param = torch.cat(
            [tensor_state[unit_metadata["param_shard_key"]].to(flat_buffer.local_shard.device) for unit_metadata in source_unit_metadatas]
        )
        flat_buffer.local_shard.copy_(_target_local_shard_from_full(flat_buffer, full_param))
        flat_buffer.use_local_shards()
        flat_buffer.clear_full_params()

        if all(unit_metadata.get("grad_shard_key") is not None for unit_metadata in source_unit_metadatas):
            full_grad = torch.cat(
                [
                    tensor_state[unit_metadata["grad_shard_key"]].to(flat_buffer.local_shard.device)
                    for unit_metadata in source_unit_metadatas
                ]
            )
            flat_buffer.use_local_grad_shard(_target_local_shard_from_full(flat_buffer, full_grad).clone())
        target_unit.lifecycle_state = FSDPLifecycleState.SHARDED

    if optimizer_reshard_plan is not None:
        _load_optimizer_from_reshard_plan(optimizer, optimizer_reshard_plan)


def _unit_state_dict(
    unit: MatrixFSDPParamGroup,
    *,
    unit_index: int,
    module_fqn: str,
    include_grads: bool,
    full_state: bool,
    clone: bool,
) -> dict[str, Any]:
    flat_buffer = _require_flat_buffer(unit)
    planner_metadata = unit.planner_result.as_metadata() if unit.planner_result is not None else None
    planner_summary = unit.planner_result.summary() if unit.planner_result is not None else None
    if unit.planner_layout_contract is not None:
        planner_layout_contract = unit.planner_layout_contract.as_metadata()
    elif planner_summary is not None:
        planner_layout_contract = planner_summary["layout"]
    else:
        planner_layout_contract = None
    runtime_layout_contract = (
        unit.runtime_layout_contract.as_metadata()
        if unit.runtime_layout_contract is not None
        else None
    )
    unit_state: dict[str, Any] = {
        "param_group_index": unit_index,
        "unit_index": unit_index,
        "module_fqn": module_fqn,
        "runtime_param_group_id": str(unit.runtime_metadata.runtime_param_group_id),
        "runtime_unit_id": str(unit.runtime_metadata.runtime_unit_id),
        "planner_group_id": str(unit.runtime_metadata.planner_group_id),
        "comm_buffer_id": str(unit.runtime_metadata.comm_buffer_id),
        "rank": unit.rank,
        "world_size": unit.world_size,
        "replicate_world_size": unit.replicate_world_size,
        "dp_shard_mesh_dim": unit.dp_shard_mesh_dim,
        "dp_replicate_mesh_dim": unit.dp_replicate_mesh_dim,
        "device_mesh": unit.device_mesh_metadata,
        "mixed_precision": mixed_precision_policy_metadata(unit.mp_policy),
        "offload_policy": offload_policy_metadata(unit.offload_policy),
        "total_numel": flat_buffer.plan.total_numel,
        "local_start": flat_buffer.local_start,
        "local_end": flat_buffer.local_end,
        "local_segments": [_layout_segment_metadata(segment) for segment in flat_buffer.local_segments],
        "shard_sizes": tuple(flat_buffer.shard_sizes),
        "matrix_shard": _matrix_shard_metadata(flat_buffer.placement),
        "param_shard_state": flat_buffer.param_state.as_metadata(),
        "layout": _layout_metadata(unit.group_layout),
        "planner_layout": _layout_metadata(unit.global_layout),
        "runtime_layout": _layout_metadata(unit.group_layout),
        "planner_layout_contract": planner_layout_contract,
        "runtime_layout_contract": runtime_layout_contract,
        "layout_flat_reordered": unit.global_layout != unit.group_layout,
        "runtime_layout_policy": unit.runtime_layout_policy,
        "runtime_layout_mode": (
            unit.runtime_layout_compatibility.mode if unit.runtime_layout_compatibility is not None else None
        ),
        "runtime_layout_reason": (
            unit.runtime_layout_compatibility.reason if unit.runtime_layout_compatibility is not None else None
        ),
        "runtime_layout_requires_flat_reorder": (
            unit.runtime_layout_compatibility.requires_flat_reorder
            if unit.runtime_layout_compatibility is not None
            else False
        ),
        "planner_metadata": planner_metadata,
        "planner_summary": planner_summary,
        "planner_report": planner_summary["report"] if planner_summary is not None else None,
        "planner_resource_estimate": (
            planner_summary["resources"]
            if planner_summary is not None and planner_summary["resources"] is not None
            else None
        ),
        "param_gather_strategy": unit.param_gather_strategy,
        "matrix_collective_backend": unit.matrix_collective_backend,
        "grad_reduce_strategy": unit.backward_reduce_strategy,
        "param_fqns": tuple(managed_param.fqn for managed_param in unit.managed_params),
        "params": {
            managed_param.fqn: _param_metadata(unit, flat_buffer, managed_param)
            for managed_param in unit.managed_params
        },
        "param_shard": _checkpoint_tensor(flat_buffer.local_shard, clone=clone),
    }
    if include_grads:
        local_grad_shard = flat_buffer.local_grad_shard
        unit_state["grad_shard"] = (
            None if local_grad_shard is None else _checkpoint_tensor(local_grad_shard, clone=clone)
        )
        unit_state["grad_shard_state"] = (
            None if flat_buffer.grad_state is None else flat_buffer.grad_state.as_metadata()
        )

    if full_state:
        full_param = _gather_full_flat_param(flat_buffer)
        unit_state["full_params"] = {
            managed_param.fqn: _checkpoint_tensor(
                full_param[managed_param.offset : managed_param.end].view(managed_param.shape),
                clone=clone,
            )
            for managed_param in unit.managed_params
        }
        if include_grads and flat_buffer.local_grad_shard is not None:
            full_grad = _gather_full_flat_grad(flat_buffer)
            unit_state["full_grads"] = {
                managed_param.fqn: _checkpoint_tensor(
                    full_grad[managed_param.offset : managed_param.end].view(managed_param.shape),
                    clone=clone,
                )
                for managed_param in unit.managed_params
            }
    return unit_state


def _load_unit_state_dict(unit: MatrixFSDPParamGroup, unit_state: dict[str, Any], *, module_fqn: str) -> None:
    flat_buffer = _require_flat_buffer(unit)
    _validate_same_layout_unit(unit, flat_buffer, unit_state, module_fqn=module_fqn)

    param_shard = unit_state.get("param_shard")
    if not torch.is_tensor(param_shard):
        raise TypeError("Unit state must contain tensor key 'param_shard'.")
    if param_shard.numel() != flat_buffer.local_numel:
        raise ValueError(
            f"Unit local shard has {param_shard.numel()} elements, expected {flat_buffer.local_numel}."
        )
    flat_buffer.local_shard.copy_(
        param_shard.detach().to(device=flat_buffer.local_shard.device, dtype=flat_buffer.local_shard.dtype)
    )
    flat_buffer.use_local_shards()
    flat_buffer.clear_full_params()

    grad_shard = unit_state.get("grad_shard")
    if torch.is_tensor(grad_shard):
        if grad_shard.numel() != flat_buffer.local_numel:
            raise ValueError(
                f"Unit local grad shard has {grad_shard.numel()} elements, expected {flat_buffer.local_numel}."
            )
        flat_buffer.use_local_grad_shard(
            grad_shard.detach().to(device=flat_buffer.local_shard.device, dtype=flat_buffer.local_shard.dtype).clone()
        )

    unit.lifecycle_state = FSDPLifecycleState.SHARDED
    unit._pending_backward_context = None
    unit._active_backward_context = None
    unit._post_backward_seen_param_ids.clear()
    unit._finalized_after_backward = False
    unit._forward_prefetched = False
    unit._backward_prefetched = False
    unit._unshard_inflight = False
    unit._unshard_handle = None


def _validate_same_layout_unit(
    unit: MatrixFSDPParamGroup,
    flat_buffer: MatrixFlatBuffer,
    unit_state: dict[str, Any],
    *,
    module_fqn: str,
) -> None:
    if unit_state.get("module_fqn") not in {module_fqn, f"unit{unit_state.get('unit_index')}"}:
        raise ValueError(
            f"State unit module_fqn={unit_state.get('module_fqn')!r} does not match target {module_fqn!r}."
        )
    expected_fqns = tuple(managed_param.fqn for managed_param in unit.managed_params)
    if tuple(unit_state.get("param_fqns", ())) != expected_fqns:
        raise ValueError(
            f"State unit params {tuple(unit_state.get('param_fqns', ()))} do not match target params {expected_fqns}."
        )
    if unit_state.get("rank") != unit.rank:
        raise ValueError(f"State rank {unit_state.get('rank')} does not match target rank {unit.rank}.")
    if unit_state.get("world_size") != unit.world_size:
        raise ValueError(
            f"State world_size {unit_state.get('world_size')} does not match target world_size {unit.world_size}."
        )
    _validate_same_layout_field(unit_state, "replicate_world_size", unit.replicate_world_size)
    _validate_same_layout_field(unit_state, "dp_shard_mesh_dim", unit.dp_shard_mesh_dim)
    _validate_same_layout_field(unit_state, "dp_replicate_mesh_dim", unit.dp_replicate_mesh_dim)
    _validate_same_layout_device_mesh(unit_state.get("device_mesh"), unit.device_mesh_metadata)
    if unit_state.get("total_numel") != flat_buffer.plan.total_numel:
        raise ValueError(
            f"State total_numel {unit_state.get('total_numel')} does not match target "
            f"{flat_buffer.plan.total_numel}."
        )
    _validate_same_layout_field(unit_state, "local_start", flat_buffer.local_start)
    _validate_same_layout_field(unit_state, "local_end", flat_buffer.local_end)
    _validate_same_layout_field(
        unit_state,
        "local_segments",
        [_layout_segment_metadata(segment) for segment in flat_buffer.local_segments],
    )
    if tuple(unit_state.get("shard_sizes", ())) != tuple(flat_buffer.shard_sizes):
        raise ValueError(
            f"State shard_sizes {tuple(unit_state.get('shard_sizes', ()))} do not match target "
            f"{tuple(flat_buffer.shard_sizes)}."
        )
    _validate_same_layout_field(unit_state, "matrix_shard", _matrix_shard_metadata(flat_buffer.placement))
    _validate_same_layout_field(unit_state, "layout", _layout_metadata(unit.group_layout))
    _validate_same_layout_field(unit_state, "runtime_layout", _layout_metadata(unit.group_layout))
    _validate_same_layout_field(unit_state, "planner_layout", _layout_metadata(unit.global_layout))
    _validate_same_layout_param_metadata(unit, flat_buffer, unit_state)


def _validate_same_layout_field(unit_state: dict[str, Any], key: str, expected: Any) -> None:
    if key not in unit_state:
        return
    saved = unit_state.get(key)
    if _canonical_metadata(saved) != _canonical_metadata(expected):
        raise ValueError(
            f"State layout field {key!r}={saved!r} does not match target {key!r}={expected!r}. "
            "Same-layout load requires identical MatrixFSDP layout metadata."
        )


def _validate_same_layout_device_mesh(
    saved_mesh: dict[str, Any] | None,
    target_mesh: dict[str, Any] | None,
) -> None:
    if saved_mesh is None and target_mesh is None:
        return
    if saved_mesh is None or target_mesh is None:
        raise ValueError(
            f"State device_mesh {saved_mesh!r} does not match target device_mesh {target_mesh!r}."
        )
    layout_keys = (
        "ndim",
        "shape",
        "mesh_dim_names",
        "coordinate",
        "shard_mesh_dim",
        "shard_mesh_dim_name",
        "shard_mesh_size",
        "replicate_mesh_dim",
        "replicate_mesh_dim_name",
        "replicate_mesh_size",
    )
    saved_layout = {key: saved_mesh.get(key) for key in layout_keys}
    target_layout = {key: target_mesh.get(key) for key in layout_keys}
    if _canonical_metadata(saved_layout) != _canonical_metadata(target_layout):
        raise ValueError(
            f"State device_mesh layout {saved_layout!r} does not match target device_mesh layout "
            f"{target_layout!r}."
        )


def _validate_same_layout_param_metadata(
    unit: MatrixFSDPParamGroup,
    flat_buffer: MatrixFlatBuffer,
    unit_state: dict[str, Any],
) -> None:
    saved_params = unit_state.get("params")
    if saved_params is None:
        return
    expected_fqns = tuple(managed_param.fqn for managed_param in unit.managed_params)
    if tuple(saved_params) != expected_fqns:
        raise ValueError(
            f"State param metadata keys {tuple(saved_params)} do not match target params {expected_fqns}."
        )
    for managed_param in unit.managed_params:
        expected_metadata = _param_metadata(unit, flat_buffer, managed_param)
        saved_metadata = saved_params[managed_param.fqn]
        for key in (
            "shape",
            "dtype",
            "numel",
            "offset",
            "end",
            "owner_ranks",
            "local_segments",
            "shard_sizes",
            "matrix_shard",
            "shard_hint",
        ):
            if key not in saved_metadata:
                continue
            if _canonical_metadata(saved_metadata[key]) != _canonical_metadata(expected_metadata[key]):
                raise ValueError(
                    f"State param {managed_param.fqn!r} metadata field {key!r}="
                    f"{saved_metadata[key]!r} does not match target {expected_metadata[key]!r}."
                )


def _require_flat_buffer(unit: MatrixFSDPParamGroup) -> MatrixFlatBuffer:
    if unit.flat_buffer is None:
        raise ValueError("MatrixFSDP checkpointing requires a unit with managed parameters.")
    return unit.flat_buffer


def _gather_full_flat_param(flat_buffer: MatrixFlatBuffer) -> torch.Tensor:
    old_full_buffer = flat_buffer.full_buffer
    handle = flat_buffer.start_all_gather_full_params()
    try:
        return handle.wait()
    finally:
        flat_buffer.full_buffer = old_full_buffer


def _gather_full_flat_grad(flat_buffer: MatrixFlatBuffer) -> torch.Tensor:
    local_grad_shard = flat_buffer.local_grad_shard
    if local_grad_shard is None:
        raise RuntimeError("Cannot build full gradients before local grad shard exists.")
    return all_gather_matrix_shard_1d_async(
        local_grad_shard,
        flat_buffer.matrix_shard,
        flat_buffer.plan.total_numel,
        group=flat_buffer.group,
        cuda_stream=flat_buffer.cuda_comm_stream,
    ).wait()


def _dcp_payload_rank_by_fqn(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    *,
    dcp_rank: int,
    dedup_replicates: bool,
) -> dict[str, str]:
    payload_rank_by_fqn: dict[str, str] = {}
    for unit_index, (module_fqn, unit) in enumerate(_collect_param_group_refs(module_or_units)):
        if dedup_replicates and unit.replicate_world_size > 1:
            payload_rank = f"shard_{unit.rank}"
        else:
            payload_rank = f"rank_{dcp_rank}"
        for managed_param in unit.managed_params:
            payload_rank_by_fqn[_param_key(module_fqn, managed_param.fqn, unit_index)] = payload_rank
    return payload_rank_by_fqn
def _source_world_size(metadata_by_rank: dict[int, dict[str, Any]]) -> int:
    first_metadata = next(iter(metadata_by_rank.values()))
    param_groups = _param_group_state_list(first_metadata, context="DCP metadata")
    if not param_groups:
        raise ValueError("MatrixFSDP DCP metadata does not contain any param groups.")
    source_world_size = int(param_groups[0]["world_size"])
    expected_ranks = set(range(source_world_size))
    if not expected_ranks.issubset(metadata_by_rank):
        raise ValueError(
            f"Expected DCP metadata sidecars for source ranks {sorted(expected_ranks)}, "
            f"found {sorted(metadata_by_rank)}."
        )
    return source_world_size


def _metadata_requires_reshard(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    metadata: dict[str, Any],
) -> bool:
    param_group_refs = _collect_param_group_refs(module_or_units)
    metadata_units = _param_group_state_list(metadata, context="DCP metadata")
    if len(param_group_refs) != len(metadata_units):
        return True
    for (_, unit), unit_metadata in zip(param_group_refs, metadata_units):
        flat_buffer = _require_flat_buffer(unit)
        if unit_metadata.get("world_size") != unit.world_size:
            return True
        if tuple(unit_metadata.get("shard_sizes", ())) != tuple(flat_buffer.shard_sizes):
            return True
        if tuple(unit_metadata.get("param_fqns", ())) != tuple(managed_param.fqn for managed_param in unit.managed_params):
            return True
    return False


def _validate_reshard_unit_metadata(
    target_unit: MatrixFSDPParamGroup,
    source_unit_metadatas: list[dict[str, Any]],
) -> None:
    if not source_unit_metadatas:
        raise ValueError("Expected source unit metadata for at least one rank.")
    expected_param_fqns = tuple(managed_param.fqn for managed_param in target_unit.managed_params)
    expected_total_numel = _require_flat_buffer(target_unit).plan.total_numel
    for source_unit_metadata in source_unit_metadatas:
        if tuple(source_unit_metadata.get("param_fqns", ())) != expected_param_fqns:
            raise ValueError(
                f"Source params {tuple(source_unit_metadata.get('param_fqns', ()))} do not match "
                f"target params {expected_param_fqns}."
            )
        if source_unit_metadata.get("total_numel") != expected_total_numel:
            raise ValueError(
                f"Source total_numel {source_unit_metadata.get('total_numel')} does not match "
                f"target {expected_total_numel}."
            )


def _target_local_shard_from_full(flat_buffer: MatrixFlatBuffer, full_tensor: torch.Tensor) -> torch.Tensor:
    if full_tensor.numel() != flat_buffer.plan.total_numel:
        raise ValueError(f"Full tensor has {full_tensor.numel()} elements, expected {flat_buffer.plan.total_numel}.")
    if not flat_buffer.local_segments:
        return full_tensor.new_empty(0).to(device=flat_buffer.local_shard.device, dtype=flat_buffer.local_shard.dtype)
    return torch.cat(
        [
            full_tensor[segment.global_start : segment.global_end].to(
                device=flat_buffer.local_shard.device,
                dtype=flat_buffer.local_shard.dtype,
            )
            for segment in flat_buffer.local_segments
        ]
    )


def _target_local_param_tensor_from_full(
    unit: MatrixFSDPParamGroup,
    managed_param: ManagedParam,
    full_tensor: torch.Tensor,
) -> torch.Tensor:
    segments = unit.rank_segments_for_param(managed_param.fqn, unit.rank)
    if not segments:
        return full_tensor.new_empty(0).to(device=managed_param.param.device, dtype=managed_param.dtype)
    pieces = [
        full_tensor[segment.global_start - managed_param.offset : segment.global_end - managed_param.offset]
        for segment in segments
    ]
    local_tensor = torch.cat(pieces).to(device=managed_param.param.device, dtype=managed_param.dtype)
    if local_tensor.numel() == managed_param.param.numel():
        return local_tensor.view_as(managed_param.param)
    return local_tensor


def _optimizer_param_fqns_by_state_id(
    torch_optimizer: Any,
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
) -> dict[int, str]:
    fqn_by_param_id = _managed_param_fqn_by_param_id(module_or_units)
    optimizer_state_dict = torch_optimizer.state_dict()
    fqn_by_state_id: dict[int, str] = {}
    for param_group, state_param_group in zip(torch_optimizer.param_groups, optimizer_state_dict["param_groups"]):
        for param, state_id in zip(param_group["params"], state_param_group["params"]):
            fqn = fqn_by_param_id.get(id(param))
            if fqn is not None:
                fqn_by_state_id[state_id] = fqn
    return fqn_by_state_id


def _managed_param_fqn_by_param_id(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
) -> dict[int, str]:
    fqn_by_param_id: dict[int, str] = {}
    for unit_index, (module_fqn, unit) in enumerate(_collect_param_group_refs(module_or_units)):
        for managed_param in unit.managed_params:
            fqn_by_param_id[id(managed_param.param)] = _param_key(module_fqn, managed_param.fqn, unit_index)
    return fqn_by_param_id


def _target_param_by_full_fqn(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
) -> dict[str, tuple[MatrixFSDPParamGroup, ManagedParam]]:
    param_by_fqn: dict[str, tuple[MatrixFSDPParamGroup, ManagedParam]] = {}
    for unit_index, (module_fqn, unit) in enumerate(_collect_param_group_refs(module_or_units)):
        for managed_param in unit.managed_params:
            param_by_fqn[_param_key(module_fqn, managed_param.fqn, unit_index)] = (unit, managed_param)
    return param_by_fqn
