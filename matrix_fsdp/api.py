from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard

from .planning.auto_planner import (
    AutoPlannerPolicy,
    auto_group_plan,
    make_cost_aware_muon_shard_aware_group_planner,
    make_muon_shard_aware_group_planner,
    make_scoped_muon_shard_aware_group_planner,
)
from .runtime.param_group import MatrixFSDPNoSync, MatrixFSDPParamGroup
from .core.managed_param import ManagedParamRegistry, ParamShardHint
from .core.mesh import DataParallelMeshDims, MeshDim
from .core.mixed_precision import validate_mixed_precision_policy, validate_offload_policy
from .planning.planner import ShardPlan, hinted_ordered_group_plan
from .planning.planner_eval import GroupPlanner
from .runtime.scheduler import DEFAULT_MAX_UNSHARDED_PREFETCH_UNITS, MatrixFSDPScheduler
from .planning.shard_hint import build_shard_hints
from .runtime.unit_collection import collect_param_groups

LayoutPlanner = GroupPlanner
WrapPolicy = Callable[[nn.Module], bool]


_DEFAULT_MP_POLICY = MixedPrecisionPolicy()
_DEFAULT_OFFLOAD_POLICY = OffloadPolicy()
_DEFAULT_ROTATION_STRATEGY = "greedy_balance"
_DEFAULT_OWNER_ASSIGNMENT = "rotate"
_DEFAULT_RESHARD_AFTER_FORWARD = True
_DEFAULT_FORWARD_PREFETCH = True
_DEFAULT_BACKWARD_PREFETCH = True
_DEFAULT_FINALIZE_AFTER_BACKWARD = True
_DEFAULT_BACKWARD_REDUCE_STRATEGY = "bucket_reduce_scatter"
_DEFAULT_USE_SAVED_TENSOR_HOOKS = False
_DEFAULT_USE_ZERO_COPY_GRAD_BUCKET = False
_FULLY_SHARD_MUON_OPTIMIZER_POLICIES = {
    "mixed_muon_adamw",
    "muon_adamw",
    "muon_shard_aware",
    "muon",
}
_FULLY_SHARD_NOOP_OPTIMIZER_POLICIES = {
    "default",
    "none",
    "adamw",
    "sgd",
}


def _single_dp_mesh_dim(value: MeshDim | tuple[MeshDim, ...] | None, *, arg_name: str) -> MeshDim | None:
    if not isinstance(value, tuple):
        return value
    if len(value) != 1:
        raise NotImplementedError(
            f"{arg_name} with flattened mesh dims is not supported yet; pass a single mesh dimension."
        )
    return value[0]


def _resolve_dp_mesh_dims(
    dp_mesh_dims: DataParallelMeshDims | object | None,
    *,
    dp_shard_mesh_dim: MeshDim | None,
    dp_replicate_mesh_dim: MeshDim | None,
) -> tuple[MeshDim | None, MeshDim | None]:
    if dp_mesh_dims is None:
        return dp_shard_mesh_dim, dp_replicate_mesh_dim
    if dp_shard_mesh_dim is not None or dp_replicate_mesh_dim is not None:
        raise ValueError("Pass either dp_mesh_dims or dp_shard_mesh_dim/dp_replicate_mesh_dim, not both.")

    shard = getattr(dp_mesh_dims, "shard", None)
    replicate = getattr(dp_mesh_dims, "replicate", None)
    if shard is None and replicate is None:
        raise ValueError("dp_mesh_dims must specify at least one of shard or replicate.")
    return (
        _single_dp_mesh_dim(shard, arg_name="dp_mesh_dims.shard"),
        _single_dp_mesh_dim(replicate, arg_name="dp_mesh_dims.replicate"),
    )


def _resolve_grad_reduce_strategy(
    grad_reduce_strategy: str | None,
    backward_reduce_strategy: str | None,
) -> str:
    if backward_reduce_strategy is None:
        return grad_reduce_strategy or _DEFAULT_BACKWARD_REDUCE_STRATEGY
    if grad_reduce_strategy is None:
        return backward_reduce_strategy
    if grad_reduce_strategy != backward_reduce_strategy:
        raise ValueError("Pass only one of grad_reduce_strategy or backward_reduce_strategy, or pass the same value.")
    return grad_reduce_strategy


def _resolve_auto_group_planner(
    policy: str | AutoPlannerPolicy,
    *,
    target_block_units: int | None = None,
    rotation_strategy: str = _DEFAULT_ROTATION_STRATEGY,
    owner_assignment: str = _DEFAULT_OWNER_ASSIGNMENT,
) -> GroupPlanner:
    policy_name = policy.name if isinstance(policy, AutoPlannerPolicy) else policy
    if policy_name == "muon_shard_aware":
        return make_muon_shard_aware_group_planner(
            policy=policy,
            rotation_strategy=rotation_strategy,
            owner_assignment=owner_assignment,
        )
    if rotation_strategy != _DEFAULT_ROTATION_STRATEGY or owner_assignment != _DEFAULT_OWNER_ASSIGNMENT:
        raise ValueError(
            "rotation_strategy and owner_assignment are only supported with "
            "auto_planner_policy='muon_shard_aware'."
        )
    return partial(auto_group_plan, policy=policy, target_block_units=target_block_units)


def _resolve_shard_placement_hints(
    module: nn.Module,
    shard_placement_fn: Callable[[nn.Parameter], object | None] | None,
    *,
    ignored_params: set[nn.Parameter] | None = None,
) -> tuple[dict[str, ParamShardHint] | None, GroupPlanner | None]:
    if shard_placement_fn is None:
        return None, None
    ignored_param_ids = {id(param) for param in ignored_params or set()}
    shard_hints: dict[str, ParamShardHint] = {}
    for fqn, param in module.named_parameters():
        if id(param) in ignored_param_ids:
            continue
        placement = shard_placement_fn(param)
        if placement is None:
            continue
        if not isinstance(placement, Shard):
            raise NotImplementedError(
                "MatrixFSDP fully_shard shard_placement_fn currently supports only "
                "None or torch.distributed.tensor.Shard placements."
            )
        if placement.dim != 0:
            raise NotImplementedError(
                "MatrixFSDP fully_shard shard_placement_fn currently supports only "
                "Shard(0). Sharding non-leading tensor dimensions would require "
                "non-contiguous local storage segments."
            )
        shard_hints[fqn] = _leading_dim_shard_hint(fqn, param)
    if not shard_hints:
        return None, None
    return (
        shard_hints,
        partial(hinted_ordered_group_plan, default_granularity="parameter"),
    )


def _leading_dim_shard_hint(fqn: str, param: nn.Parameter) -> ParamShardHint:
    if param.ndim == 0:
        raise NotImplementedError(
            f"MatrixFSDP cannot apply Shard(0) to scalar parameter {fqn}."
        )
    if param.ndim == 2:
        return ParamShardHint(split_granularity="row_block", block_shape=(1, int(param.shape[1])))
    leading_dim_units = int(param.numel() // param.shape[0])
    return ParamShardHint(split_granularity="block", block_shape=(leading_dim_units,))


def _resolve_fully_shard_reshard_after_forward(value: bool | int | None) -> bool | int:
    if value is None:
        # Keep the public API shape close to FSDP2 while defaulting MatrixFSDP
        # to the training-friendly path that frees full params after forward.
        return True
    return value


def _resolve_fully_shard_optimizer_policy(
    optimizer_policy: str | None,
) -> tuple[bool, str | AutoPlannerPolicy | None, str]:
    if optimizer_policy is None:
        return False, None, _DEFAULT_OWNER_ASSIGNMENT
    policy_name = optimizer_policy.lower()
    if policy_name in _FULLY_SHARD_NOOP_OPTIMIZER_POLICIES:
        return False, None, _DEFAULT_OWNER_ASSIGNMENT
    if policy_name in _FULLY_SHARD_MUON_OPTIMIZER_POLICIES:
        return True, "muon_shard_aware", "role_greedy"
    allowed = sorted(_FULLY_SHARD_MUON_OPTIMIZER_POLICIES | _FULLY_SHARD_NOOP_OPTIMIZER_POLICIES)
    raise ValueError(f"Unknown fully_shard optimizer_policy={optimizer_policy!r}; expected one of {allowed}.")


def _filter_ignored_shard_hints(
    module: nn.Module,
    shard_hints: Mapping[str, ParamShardHint] | None,
    ignored_params: set[nn.Parameter] | None,
) -> Mapping[str, ParamShardHint] | None:
    if not shard_hints or not ignored_params:
        return shard_hints
    ignored_param_ids = {id(param) for param in ignored_params}
    managed_fqns = {
        fqn
        for fqn, param in module.named_parameters(recurse=True, remove_duplicate=True)
        if id(param) not in ignored_param_ids
    }
    return {fqn: hint for fqn, hint in shard_hints.items() if fqn in managed_fqns}


def _module_has_unignored_params(module: nn.Module, ignored_params: set[nn.Parameter] | None) -> bool:
    ignored_param_ids = {id(param) for param in ignored_params or set()}
    return any(id(param) not in ignored_param_ids for param in module.parameters(recurse=True))


def fully_shard(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn: Callable[[nn.Parameter], object | None] | None = None,
    mp_policy: MixedPrecisionPolicy = _DEFAULT_MP_POLICY,
    offload_policy: OffloadPolicy = _DEFAULT_OFFLOAD_POLICY,
    ignored_params: set[nn.Parameter] | None = None,
    dp_mesh_dims: DataParallelMeshDims | object | None = None,
    optimizer_policy: str | None = None,
) -> nn.Module:
    validate_mixed_precision_policy(mp_policy)
    validate_offload_policy(offload_policy)
    auto_shard_hints, auto_planner_policy, owner_assignment = _resolve_fully_shard_optimizer_policy(
        optimizer_policy
    )
    if auto_planner_policy is not None and shard_placement_fn is not None:
        raise ValueError("fully_shard optimizer_policy cannot be combined with shard_placement_fn.")
    shard_hints, group_planner = _resolve_shard_placement_hints(
        module,
        shard_placement_fn,
        ignored_params=ignored_params,
    )
    dp_shard_mesh_dim, dp_replicate_mesh_dim = _resolve_dp_mesh_dims(
        dp_mesh_dims,
        dp_shard_mesh_dim=None,
        dp_replicate_mesh_dim=None,
    )
    resolved_reshard_after_forward = _resolve_fully_shard_reshard_after_forward(reshard_after_forward)
    return _matrix_fully_shard_single(
        module,
        mesh=mesh,
        dp_shard_mesh_dim=dp_shard_mesh_dim,
        dp_replicate_mesh_dim=dp_replicate_mesh_dim,
        ignored_params=ignored_params,
        shard_hints=shard_hints,
        group_planner=group_planner,
        auto_shard_hints=auto_shard_hints,
        auto_planner_policy=auto_planner_policy,
        owner_assignment=owner_assignment,
        reshard_after_forward=resolved_reshard_after_forward,
        forward_prefetch=_DEFAULT_FORWARD_PREFETCH,
        backward_prefetch=_DEFAULT_BACKWARD_PREFETCH,
        finalize_after_backward=_DEFAULT_FINALIZE_AFTER_BACKWARD,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        backward_reduce_strategy=_DEFAULT_BACKWARD_REDUCE_STRATEGY,
        use_saved_tensor_hooks=_DEFAULT_USE_SAVED_TENSOR_HOOKS,
        use_zero_copy_grad_bucket=_DEFAULT_USE_ZERO_COPY_GRAD_BUCKET,
    )


def matrix_fully_shard(
    module: nn.Module,
    mesh: DeviceMesh | None = None,
    *,
    dp_mesh_dims: DataParallelMeshDims | object | None = None,
    dp_shard_mesh_dim: MeshDim | None = None,
    dp_replicate_mesh_dim: MeshDim | None = None,
    planner: Callable[[int, int], ShardPlan] | None = None,
    group_planner: GroupPlanner | None = None,
    layout_planner: LayoutPlanner | None = None,
    shard_hints: Mapping[str, ParamShardHint] | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    auto_shard_hints: bool = False,
    auto_planner_policy: str | AutoPlannerPolicy | None = None,
    target_block_units: int | None = None,
    rotation_strategy: str = _DEFAULT_ROTATION_STRATEGY,
    owner_assignment: str = _DEFAULT_OWNER_ASSIGNMENT,
    divide_grads_by_world: bool = True,
    reshard_after_forward: bool | int = _DEFAULT_RESHARD_AFTER_FORWARD,
    forward_prefetch: bool = _DEFAULT_FORWARD_PREFETCH,
    backward_prefetch: bool = _DEFAULT_BACKWARD_PREFETCH,
    finalize_after_backward: bool = _DEFAULT_FINALIZE_AFTER_BACKWARD,
    mp_policy: MixedPrecisionPolicy = _DEFAULT_MP_POLICY,
    offload_policy: OffloadPolicy = _DEFAULT_OFFLOAD_POLICY,
    param_gather_strategy: str = "auto",
    matrix_collective_backend: str = "owner_broadcast",
    grad_reduce_strategy: str | None = None,
    backward_reduce_strategy: str | None = None,
    runtime_layout_policy: str = "auto",
    wrap_policy: WrapPolicy | None = None,
    runtime_trace_enabled: bool = True,
    use_saved_tensor_hooks: bool = _DEFAULT_USE_SAVED_TENSOR_HOOKS,
    use_zero_copy_grad_bucket: bool = _DEFAULT_USE_ZERO_COPY_GRAD_BUCKET,
    shrink_full_param_storage_after_backward: bool = True,
) -> nn.Module:
    backward_reduce_strategy = _resolve_grad_reduce_strategy(grad_reduce_strategy, backward_reduce_strategy)
    validate_mixed_precision_policy(mp_policy)
    validate_offload_policy(offload_policy)
    dp_shard_mesh_dim, dp_replicate_mesh_dim = _resolve_dp_mesh_dims(
        dp_mesh_dims,
        dp_shard_mesh_dim=dp_shard_mesh_dim,
        dp_replicate_mesh_dim=dp_replicate_mesh_dim,
    )
    if wrap_policy is not None:
        if shard_hints is not None:
            raise ValueError("wrap_policy does not support explicit shard_hints yet.")
        if auto_planner_policy is not None:
            if group_planner is not None or layout_planner is not None:
                raise ValueError("Pass either auto_planner_policy or an explicit group/layout planner, not both.")
            if owner_assignment in {"scope_greedy", "cost_aware"}:
                selected_modules = _collect_wrap_policy_modules(module, wrap_policy, ignored_params=ignored_params)
                if not selected_modules:
                    raise ValueError("wrap_policy did not select any modules to shard.")
                scoped_param_groups = tuple(
                    ManagedParamRegistry.from_module(
                        selected_module,
                        shard_hints=build_shard_hints(selected_module) if auto_shard_hints else None,
                        ignored_params=ignored_params,
                    ).params
                    for selected_module in selected_modules
                )
                if owner_assignment == "cost_aware":
                    group_planner = make_cost_aware_muon_shard_aware_group_planner(
                        scoped_param_groups,
                        policy=auto_planner_policy,
                    )
                else:
                    group_planner = make_scoped_muon_shard_aware_group_planner(
                        scoped_param_groups,
                        policy=auto_planner_policy,
                    )
            else:
                group_planner = _resolve_auto_group_planner(
                    auto_planner_policy,
                    target_block_units=target_block_units,
                    rotation_strategy=rotation_strategy,
                    owner_assignment=owner_assignment,
                )
            auto_planner_policy = None
        wrapped_units = _apply_wrap_policy(
            module,
            wrap_policy,
            mesh=mesh,
            dp_shard_mesh_dim=dp_shard_mesh_dim,
            dp_replicate_mesh_dim=dp_replicate_mesh_dim,
            planner=planner,
            group_planner=group_planner,
            layout_planner=layout_planner,
            ignored_params=ignored_params,
            auto_shard_hints=auto_shard_hints,
            auto_planner_policy=auto_planner_policy,
            target_block_units=target_block_units,
            rotation_strategy=rotation_strategy,
            owner_assignment=owner_assignment,
            divide_grads_by_world=divide_grads_by_world,
            reshard_after_forward=reshard_after_forward,
            forward_prefetch=forward_prefetch,
            backward_prefetch=backward_prefetch,
            finalize_after_backward=finalize_after_backward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            param_gather_strategy=param_gather_strategy,
            matrix_collective_backend=matrix_collective_backend,
            backward_reduce_strategy=backward_reduce_strategy,
            runtime_layout_policy=runtime_layout_policy,
            runtime_trace_enabled=runtime_trace_enabled,
            use_saved_tensor_hooks=use_saved_tensor_hooks,
            use_zero_copy_grad_bucket=use_zero_copy_grad_bucket,
            shrink_full_param_storage_after_backward=shrink_full_param_storage_after_backward,
        )
        if wrapped_units == 0:
            raise ValueError("wrap_policy did not select any modules to shard.")
        MatrixFSDPScheduler(
            collect_param_groups(module),
            max_unsharded_prefetch_units=DEFAULT_MAX_UNSHARDED_PREFETCH_UNITS,
        )
        _attach_no_sync(module, collect_param_groups(module))
        return module
    return _matrix_fully_shard_single(
        module,
        mesh=mesh,
        dp_shard_mesh_dim=dp_shard_mesh_dim,
        dp_replicate_mesh_dim=dp_replicate_mesh_dim,
        planner=planner,
        group_planner=group_planner,
        layout_planner=layout_planner,
        shard_hints=shard_hints,
        ignored_params=ignored_params,
        auto_shard_hints=auto_shard_hints,
        auto_planner_policy=auto_planner_policy,
        target_block_units=target_block_units,
        rotation_strategy=rotation_strategy,
        owner_assignment=owner_assignment,
        divide_grads_by_world=divide_grads_by_world,
        reshard_after_forward=reshard_after_forward,
        forward_prefetch=forward_prefetch,
        backward_prefetch=backward_prefetch,
        finalize_after_backward=finalize_after_backward,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        param_gather_strategy=param_gather_strategy,
        matrix_collective_backend=matrix_collective_backend,
        backward_reduce_strategy=backward_reduce_strategy,
        runtime_layout_policy=runtime_layout_policy,
        runtime_trace_enabled=runtime_trace_enabled,
        use_saved_tensor_hooks=use_saved_tensor_hooks,
        use_zero_copy_grad_bucket=use_zero_copy_grad_bucket,
        shrink_full_param_storage_after_backward=shrink_full_param_storage_after_backward,
    )


def _matrix_fully_shard_single(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = None,
    dp_shard_mesh_dim: MeshDim | None = None,
    dp_replicate_mesh_dim: MeshDim | None = None,
    planner: Callable[[int, int], ShardPlan] | None = None,
    group_planner: GroupPlanner | None = None,
    layout_planner: LayoutPlanner | None = None,
    shard_hints: Mapping[str, ParamShardHint] | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    auto_shard_hints: bool = False,
    auto_planner_policy: str | AutoPlannerPolicy | None = None,
    target_block_units: int | None = None,
    rotation_strategy: str = _DEFAULT_ROTATION_STRATEGY,
    owner_assignment: str = _DEFAULT_OWNER_ASSIGNMENT,
    divide_grads_by_world: bool = True,
    reshard_after_forward: bool | int = _DEFAULT_RESHARD_AFTER_FORWARD,
    forward_prefetch: bool = _DEFAULT_FORWARD_PREFETCH,
    backward_prefetch: bool = _DEFAULT_BACKWARD_PREFETCH,
    finalize_after_backward: bool = _DEFAULT_FINALIZE_AFTER_BACKWARD,
    mp_policy: MixedPrecisionPolicy = _DEFAULT_MP_POLICY,
    offload_policy: OffloadPolicy = _DEFAULT_OFFLOAD_POLICY,
    param_gather_strategy: str = "auto",
    matrix_collective_backend: str = "owner_broadcast",
    grad_reduce_strategy: str | None = None,
    backward_reduce_strategy: str | None = None,
    runtime_layout_policy: str = "auto",
    runtime_trace_enabled: bool = True,
    use_saved_tensor_hooks: bool = _DEFAULT_USE_SAVED_TENSOR_HOOKS,
    use_zero_copy_grad_bucket: bool = _DEFAULT_USE_ZERO_COPY_GRAD_BUCKET,
    shrink_full_param_storage_after_backward: bool = True,
) -> nn.Module:
    backward_reduce_strategy = _resolve_grad_reduce_strategy(grad_reduce_strategy, backward_reduce_strategy)
    validate_mixed_precision_policy(mp_policy)
    validate_offload_policy(offload_policy)
    if auto_shard_hints:
        shard_hints = build_shard_hints(module, overrides=shard_hints)
    shard_hints = _filter_ignored_shard_hints(module, shard_hints, ignored_params)
    if auto_planner_policy is not None:
        if group_planner is not None or layout_planner is not None:
            raise ValueError("Pass either auto_planner_policy or an explicit group/layout planner, not both.")
        group_planner = _resolve_auto_group_planner(
            auto_planner_policy,
            target_block_units=target_block_units,
            rotation_strategy=rotation_strategy,
            owner_assignment=owner_assignment,
        )
    unit = MatrixFSDPParamGroup(
        module,
        mesh,
        dp_shard_mesh_dim=dp_shard_mesh_dim,
        dp_replicate_mesh_dim=dp_replicate_mesh_dim,
        planner=planner,
        group_planner=group_planner,
        layout_planner=layout_planner,
        shard_hints=shard_hints,
        ignored_params=ignored_params,
        divide_grads_by_world=divide_grads_by_world,
        reshard_after_forward=reshard_after_forward,
        forward_prefetch=forward_prefetch,
        backward_prefetch=backward_prefetch,
        finalize_after_backward=finalize_after_backward,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        param_gather_strategy=param_gather_strategy,
        matrix_collective_backend=matrix_collective_backend,
        backward_reduce_strategy=backward_reduce_strategy,
        runtime_layout_policy=runtime_layout_policy,
        runtime_trace_enabled=runtime_trace_enabled,
        use_saved_tensor_hooks=use_saved_tensor_hooks,
        use_zero_copy_grad_bucket=use_zero_copy_grad_bucket,
        shrink_full_param_storage_after_backward=shrink_full_param_storage_after_backward,
    )
    unit.init()
    _attach_no_sync(module, (unit,))
    return module


def _attach_no_sync(module: nn.Module, param_groups) -> None:
    param_group_tuple = tuple(param_groups)

    def no_sync() -> MatrixFSDPNoSync:
        return MatrixFSDPNoSync(param_group_tuple)

    module.no_sync = no_sync  # type: ignore[method-assign, attr-defined]


def _apply_wrap_policy(
    module: nn.Module,
    wrap_policy: WrapPolicy,
    *,
    mesh: DeviceMesh | None = None,
    dp_shard_mesh_dim: MeshDim | None = None,
    dp_replicate_mesh_dim: MeshDim | None = None,
    planner: Callable[[int, int], ShardPlan] | None = None,
    group_planner: GroupPlanner | None = None,
    layout_planner: LayoutPlanner | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    auto_shard_hints: bool = False,
    auto_planner_policy: str | AutoPlannerPolicy | None = None,
    target_block_units: int | None = None,
    rotation_strategy: str = _DEFAULT_ROTATION_STRATEGY,
    owner_assignment: str = _DEFAULT_OWNER_ASSIGNMENT,
    divide_grads_by_world: bool = True,
    reshard_after_forward: bool | int = _DEFAULT_RESHARD_AFTER_FORWARD,
    forward_prefetch: bool = _DEFAULT_FORWARD_PREFETCH,
    backward_prefetch: bool = _DEFAULT_BACKWARD_PREFETCH,
    finalize_after_backward: bool = _DEFAULT_FINALIZE_AFTER_BACKWARD,
    mp_policy: MixedPrecisionPolicy = _DEFAULT_MP_POLICY,
    offload_policy: OffloadPolicy = _DEFAULT_OFFLOAD_POLICY,
    param_gather_strategy: str = "auto",
    matrix_collective_backend: str = "owner_broadcast",
    grad_reduce_strategy: str | None = None,
    backward_reduce_strategy: str | None = None,
    runtime_layout_policy: str = "auto",
    runtime_trace_enabled: bool = True,
    use_saved_tensor_hooks: bool = _DEFAULT_USE_SAVED_TENSOR_HOOKS,
    use_zero_copy_grad_bucket: bool = _DEFAULT_USE_ZERO_COPY_GRAD_BUCKET,
    shrink_full_param_storage_after_backward: bool = True,
) -> int:
    backward_reduce_strategy = _resolve_grad_reduce_strategy(grad_reduce_strategy, backward_reduce_strategy)
    validate_mixed_precision_policy(mp_policy)
    validate_offload_policy(offload_policy)
    if wrap_policy(module):
        if not _module_has_unignored_params(module, ignored_params):
            return 0
        _matrix_fully_shard_single(
            module,
            mesh=mesh,
            dp_shard_mesh_dim=dp_shard_mesh_dim,
            dp_replicate_mesh_dim=dp_replicate_mesh_dim,
            planner=planner,
            group_planner=group_planner,
            layout_planner=layout_planner,
            ignored_params=ignored_params,
            auto_shard_hints=auto_shard_hints,
            auto_planner_policy=auto_planner_policy,
            target_block_units=target_block_units,
            rotation_strategy=rotation_strategy,
            owner_assignment=owner_assignment,
            divide_grads_by_world=divide_grads_by_world,
            reshard_after_forward=reshard_after_forward,
            forward_prefetch=forward_prefetch,
            backward_prefetch=backward_prefetch,
            finalize_after_backward=finalize_after_backward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            param_gather_strategy=param_gather_strategy,
            matrix_collective_backend=matrix_collective_backend,
            backward_reduce_strategy=backward_reduce_strategy,
            runtime_layout_policy=runtime_layout_policy,
            runtime_trace_enabled=runtime_trace_enabled,
            use_saved_tensor_hooks=use_saved_tensor_hooks,
            use_zero_copy_grad_bucket=use_zero_copy_grad_bucket,
            shrink_full_param_storage_after_backward=shrink_full_param_storage_after_backward,
        )
        return 1

    wrapped_units = 0
    for child in module.children():
        wrapped_units += _apply_wrap_policy(
            child,
            wrap_policy,
            mesh=mesh,
            dp_shard_mesh_dim=dp_shard_mesh_dim,
            dp_replicate_mesh_dim=dp_replicate_mesh_dim,
            planner=planner,
            group_planner=group_planner,
            layout_planner=layout_planner,
            ignored_params=ignored_params,
            auto_shard_hints=auto_shard_hints,
            auto_planner_policy=auto_planner_policy,
            target_block_units=target_block_units,
            rotation_strategy=rotation_strategy,
            owner_assignment=owner_assignment,
            divide_grads_by_world=divide_grads_by_world,
            reshard_after_forward=reshard_after_forward,
            forward_prefetch=forward_prefetch,
            backward_prefetch=backward_prefetch,
            finalize_after_backward=finalize_after_backward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            param_gather_strategy=param_gather_strategy,
            matrix_collective_backend=matrix_collective_backend,
            backward_reduce_strategy=backward_reduce_strategy,
            runtime_layout_policy=runtime_layout_policy,
            runtime_trace_enabled=runtime_trace_enabled,
            use_saved_tensor_hooks=use_saved_tensor_hooks,
            use_zero_copy_grad_bucket=use_zero_copy_grad_bucket,
            shrink_full_param_storage_after_backward=shrink_full_param_storage_after_backward,
        )
    return wrapped_units


def _collect_wrap_policy_modules(
    module: nn.Module,
    wrap_policy: WrapPolicy,
    *,
    ignored_params: set[nn.Parameter] | None = None,
) -> tuple[nn.Module, ...]:
    if wrap_policy(module):
        return (module,) if _module_has_unignored_params(module, ignored_params) else ()
    selected: list[nn.Module] = []
    for child in module.children():
        selected.extend(_collect_wrap_policy_modules(child, wrap_policy, ignored_params=ignored_params))
    return tuple(selected)
