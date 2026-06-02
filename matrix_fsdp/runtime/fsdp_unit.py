from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from enum import Enum
from time import perf_counter
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy

from matrix_fsdp.runtime.collectives import MatrixCollectiveHandle, MatrixTensorCollectiveHandle
from matrix_fsdp.runtime.flat_buffer import MatrixFlatBuffer
from matrix_fsdp.runtime.hooks import ForwardBackwardContext, register_pre_backward_hooks_with_context
from matrix_fsdp.core.layout import LayoutSegment, ParamLayout, ParamSegment, MatrixGroupLayout
from matrix_fsdp.planning.layout_validator import RuntimeLayoutCompatibility, explain_runtime_layout_compatibility, validate_group_layout
from matrix_fsdp.core.managed_param import ManagedParam, ManagedParamRegistry, ParamShardHint
from matrix_fsdp.core.mesh import MeshDim, infer_replicate_mesh_dim, infer_shard_mesh_dim, mesh_metadata
from matrix_fsdp.core.mixed_precision import (
    cast_floating_tensors,
    mixed_precision_policy_metadata,
    offload_policy_metadata,
    validate_offload_policy_for_device,
)
from matrix_fsdp.planning.planner import ShardPlan, contiguous_even_plan
from matrix_fsdp.planning.planner_eval import (
    GroupPlanner,
    PlannerEvaluation,
    PlannerLayoutContract,
    PlannerResult,
    call_group_planner,
    planner_display_name,
)
from matrix_fsdp.runtime.runtime import RuntimeUnitMetadata, new_runtime_unit_metadata
from matrix_fsdp.runtime.runtime_event import RuntimeEvent, next_runtime_event_sequence

if TYPE_CHECKING:
    from matrix_fsdp.runtime.buffer_pool import FullParamBufferPool
    from matrix_fsdp.runtime.scheduler import MatrixFSDPCommContext, MatrixFSDPScheduler

LayoutPlanner = GroupPlanner
RuntimeLayoutPolicy = str
_RUNTIME_LAYOUT_POLICIES = ("auto", "no_reorder", "matrix_shard_only")


def _is_in_backward_graph_task() -> bool:
    current_graph_task_id = getattr(torch._C, "_current_graph_task_id", None)
    return current_graph_task_id is not None and current_graph_task_id() != -1


@dataclass(frozen=True)
class _SavedFullParamView:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int


class FSDPLifecycleState(str, Enum):
    SHARDED = "sharded"
    UNSHARDED = "unsharded"
    FORWARD_RESHARDED = "forward_resharded"


class FSDPRuntimeState(str, Enum):
    SHARDED = "sharded"
    UNSHARDED_FORWARD = "unsharded_forward"
    FORWARD_RESHARDED = "forward_resharded"
    UNSHARDED_BACKWARD = "unsharded_backward"
    BACKWARD_DEFERRED = "backward_deferred"
    REDUCE_IN_FLIGHT = "reduce_in_flight"


class MatrixFSDPNoSync:
    def __init__(self, units: Iterable["MatrixFSDPParamGroup"]) -> None:
        self.units = tuple(units)
        self._entered_units: list[MatrixFSDPParamGroup] = []

    def __enter__(self) -> "MatrixFSDPNoSync":
        try:
            for unit in self.units:
                unit._enter_no_sync()
                self._entered_units.append(unit)
        except Exception:
            for entered_unit in reversed(self._entered_units):
                entered_unit._exit_no_sync()
            self._entered_units.clear()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for unit in reversed(self._entered_units):
            unit._exit_no_sync()
        self._entered_units.clear()
        return None


class MatrixFSDPParamGroup:
    """
    Correctness-first MatrixShard-style FSDP parameter group.

    This is the runtime object closest to FSDP2's ``FSDPParamGroup``: it owns
    the managed parameters, flat buffers, hooks, and parameter lifecycle for one
    wrapped module.
    """

    def __init__(
        self,
        module: nn.Module,
        mesh: DeviceMesh | None = None,
        *,
        dp_shard_mesh_dim: MeshDim | None = None,
        dp_replicate_mesh_dim: MeshDim | None = None,
        planner: Callable[[int, int], ShardPlan] | None = None,
        group_planner: GroupPlanner | None = None,
        layout_planner: LayoutPlanner | None = None,
        shard_hints: Mapping[str, ParamShardHint] | None = None,
        divide_grads_by_world: bool = True,
        reshard_after_forward: bool | int = False,
        forward_prefetch: bool = False,
        backward_prefetch: bool = False,
        finalize_after_backward: bool = False,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        param_gather_strategy: str = "auto",
        matrix_collective_backend: str = "owner_broadcast",
        backward_reduce_strategy: str = "flat",
        runtime_layout_policy: RuntimeLayoutPolicy = "auto",
        ignored_params: set[nn.Parameter] | None = None,
        runtime_metadata: RuntimeUnitMetadata | None = None,
        runtime_trace_enabled: bool = True,
        use_saved_tensor_hooks: bool = True,
        use_zero_copy_grad_bucket: bool = True,
        shrink_full_param_storage_after_backward: bool = True,
    ) -> None:
        self.module = module
        self.mesh = mesh
        self.dp_shard_mesh_dim = infer_shard_mesh_dim(mesh, dp_shard_mesh_dim)
        self.dp_replicate_mesh_dim = infer_replicate_mesh_dim(
            mesh,
            self.dp_shard_mesh_dim,
            dp_replicate_mesh_dim,
        )
        self.planner = planner or contiguous_even_plan
        if group_planner is not None and layout_planner is not None:
            raise ValueError("Pass only one of group_planner or layout_planner.")
        self.group_planner = group_planner if group_planner is not None else layout_planner
        self.layout_planner = self.group_planner
        self.shard_hints = shard_hints
        self.divide_grads_by_world = divide_grads_by_world
        self.reshard_after_forward_policy = reshard_after_forward
        self.reshard_after_forward_world_size = reshard_after_forward if type(reshard_after_forward) is int else None
        self.reshard_after_forward_enabled = bool(reshard_after_forward)
        self.forward_prefetch_enabled = forward_prefetch
        self.backward_prefetch_enabled = backward_prefetch
        self.finalize_after_backward_enabled = finalize_after_backward
        self.mp_policy = mp_policy or MixedPrecisionPolicy()
        self.offload_policy = offload_policy or OffloadPolicy()
        self.param_dtype = self.mp_policy.param_dtype
        self.reduce_dtype = self.mp_policy.reduce_dtype
        self.output_dtype = self.mp_policy.output_dtype
        self.cast_forward_inputs = self.mp_policy.cast_forward_inputs
        if param_gather_strategy not in ("auto", "equal_all_gather", "matrix_all_gather", "owner_broadcast"):
            raise ValueError(
                "param_gather_strategy must be 'auto', 'equal_all_gather', 'matrix_all_gather', or 'owner_broadcast'."
            )
        self.param_gather_strategy = param_gather_strategy
        if matrix_collective_backend not in ("torch", "owner_broadcast", "custom"):
            raise ValueError(
                "matrix_collective_backend must be 'torch', 'owner_broadcast', or 'custom'."
            )
        self.matrix_collective_backend = matrix_collective_backend
        if backward_reduce_strategy not in ("flat", "bucket_reduce_scatter", "per_param", "per_param_allreduce"):
            raise ValueError(
                "backward_reduce_strategy must be 'flat', 'bucket_reduce_scatter', "
                "'per_param', or 'per_param_allreduce'."
            )
        if backward_reduce_strategy in ("per_param", "per_param_allreduce") and not finalize_after_backward:
            raise ValueError(f"backward_reduce_strategy={backward_reduce_strategy!r} requires finalize_after_backward=True.")
        self.backward_reduce_strategy = backward_reduce_strategy
        if runtime_layout_policy not in _RUNTIME_LAYOUT_POLICIES:
            valid = ", ".join(repr(policy) for policy in _RUNTIME_LAYOUT_POLICIES)
            raise ValueError(f"runtime_layout_policy must be one of {valid}, got {runtime_layout_policy!r}.")
        self.runtime_layout_policy = runtime_layout_policy
        self.ignored_params = ignored_params
        self.runtime_metadata = runtime_metadata or new_runtime_unit_metadata()
        self.runtime_trace_enabled = runtime_trace_enabled
        self.use_saved_tensor_hooks = use_saved_tensor_hooks
        self.use_zero_copy_grad_bucket = use_zero_copy_grad_bucket
        self.shrink_full_param_storage_after_backward = shrink_full_param_storage_after_backward

        self.group = self._get_group(mesh, self.dp_shard_mesh_dim)
        self.replicate_group = self._get_group(mesh, self.dp_replicate_mesh_dim)
        self.rank = self._get_rank(mesh, self.dp_shard_mesh_dim)
        self.world_size = self._get_world_size(mesh, self.dp_shard_mesh_dim)
        self.replicate_world_size = self._get_world_size(mesh, self.dp_replicate_mesh_dim)
        self.device_mesh_metadata = (
            None
            if mesh is None
            else mesh_metadata(
                mesh,
                shard_mesh_dim=self.dp_shard_mesh_dim,
                replicate_mesh_dim=self.dp_replicate_mesh_dim,
            ).as_dict()
        )

        self.param_registry: ManagedParamRegistry | None = None
        self.managed_params: list[ManagedParam] = []
        self.group_layout: MatrixGroupLayout | None = None
        self.global_layout: MatrixGroupLayout | None = None
        self.planner_layout_contract: PlannerLayoutContract | None = None
        self.runtime_layout_contract: PlannerLayoutContract | None = None
        self.runtime_layout_compatibility: RuntimeLayoutCompatibility | None = None
        self.planner_result: PlannerResult | None = None
        self.planner_evaluation: PlannerEvaluation | None = None
        self.shard_plan: ShardPlan | None = None
        self.flat_buffer: MatrixFlatBuffer | None = None
        self.lifecycle_state = FSDPLifecycleState.SHARDED
        self._pending_backward_context: ForwardBackwardContext | None = None
        self._active_backward_context: ForwardBackwardContext | None = None
        self._saved_tensors_hooks_context: AbstractContextManager | None = None
        self._saved_full_param_views = 0
        self._checkpoint_recompute_forward_depth = 0
        self._post_backward_seen_param_ids: set[int] = set()
        self._post_backward_expected_param_ids: set[int] = set()
        self._post_backward_param_by_id: dict[int, ManagedParam] = {}
        self._no_sync_depth = 0
        self._defer_backward_reduce = False
        self._finalized_after_backward = False
        self._forward_prefetched = False
        self._backward_prefetched = False
        self._unshard_inflight = False
        self._unshard_handle: MatrixCollectiveHandle | None = None
        self._post_backward_reduce_handle: MatrixTensorCollectiveHandle | None = None
        self._post_backward_reduce_should_accumulate = False
        self._grad_bucket_prepared_zero_copy = False
        self._cuda_comm_stream: torch.cuda.Stream | None = None
        self._cuda_all_gather_stream: torch.cuda.Stream | None = None
        self._cuda_reduce_scatter_stream: torch.cuda.Stream | None = None
        self._scheduler: MatrixFSDPScheduler | None = None
        self.runtime_events: list[RuntimeEvent] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def init(self) -> None:
        self.param_registry = ManagedParamRegistry.from_module(
            self.module,
            shard_hints=self.shard_hints,
            ignored_params=self.ignored_params,
        )
        self.managed_params = self.param_registry.as_list()
        self.module._matrix_fsdp_param_group = self  # type: ignore[attr-defined]
        if not self.managed_params:
            return

        self._mark_managed_params_for_optimizer_auto_prepare()
        self._validate_managed_params()
        self._validate_reshard_after_forward_policy()
        from matrix_fsdp.optim.wrapper import install_matrix_optimizer_auto_prepare

        install_matrix_optimizer_auto_prepare()
        self._cuda_all_gather_stream, self._cuda_reduce_scatter_stream = self._create_cuda_comm_streams()
        self._cuda_comm_stream = self._cuda_all_gather_stream
        total_numel = self.param_registry.total_numel
        self.planner_layout_contract = self._build_planner_layout_contract(total_numel)
        self.global_layout = self.planner_layout_contract.layout
        self.runtime_layout_contract = self._prepare_runtime_layout(self.planner_layout_contract)
        # Segment layouts may materialize full params through an unpack buffer,
        # so saved tensors need hooks to follow the current full-param storage.
        if (
            self.runtime_layout_compatibility.mode == "segment_runtime"
            and self.reshard_after_forward_enabled
            and not self.use_saved_tensor_hooks
        ):
            self.use_saved_tensor_hooks = True
        self.group_layout = self.runtime_layout_contract.layout
        self._validate_layout_contract(self.runtime_layout_contract, total_numel)
        plan = self.runtime_layout_contract.to_shard_plan()
        self._validate_plan(plan, total_numel)
        self.shard_plan = plan
        self.flat_buffer = MatrixFlatBuffer(
            self.managed_params,
            plan,
            self.rank,
            mesh=self.mesh,
            dp_shard_mesh_dim=self.dp_shard_mesh_dim,
            replicate_group=self.replicate_group,
            replicate_world_size=self.replicate_world_size,
            group=self.group,
            divide_grads_by_world=self.divide_grads_by_world,
            cuda_all_gather_stream=self._cuda_all_gather_stream,
            cuda_reduce_scatter_stream=self._cuda_reduce_scatter_stream,
            param_gather_strategy=self.param_gather_strategy,
            matrix_collective_backend=self.matrix_collective_backend,
            param_dtype=self.param_dtype,
            reduce_dtype=self.reduce_dtype,
            collective_key=str(self.runtime_metadata.runtime_param_group_id),
        )
        self.flat_buffer.use_local_shards()
        self._record_event("init")
        self._handles.append(self.module.register_forward_pre_hook(self._pre_forward, with_kwargs=True))
        if self.reshard_after_forward_enabled or self.output_dtype is not None:
            self._handles.append(self.module.register_forward_hook(self._post_forward, with_kwargs=True))
        if self.finalize_after_backward_enabled:
            self._register_post_backward_hooks()

    def _mark_managed_params_for_optimizer_auto_prepare(self) -> None:
        param_group_ref = weakref.ref(self)
        for managed_param in self.managed_params:
            managed_param.param._matrix_fsdp_param_group_ref = param_group_ref  # type: ignore[attr-defined]

    def unshard(self) -> None:
        self.start_unshard("explicit")
        self.wait_unshard("explicit")

    def start_unshard(
        self,
        reason: str,
        *,
        validate_owner_collective_signature: bool | None = None,
    ) -> bool:
        if self.flat_buffer is None or self.lifecycle_state == FSDPLifecycleState.UNSHARDED:
            return False
        self._record_event(f"start_unshard:{reason}")
        is_prefetch = reason.endswith("_prefetch")
        if validate_owner_collective_signature is None:
            validate_owner_collective_signature = not is_prefetch
        enqueue_start = perf_counter()
        self._unshard_handle = self.flat_buffer.start_all_gather_full_params(
            validate_owner_collective_signature=validate_owner_collective_signature,
        )
        self._record_event(
            "enqueue_all_gather_full_params",
            duration_ms=(perf_counter() - enqueue_start) * 1000.0,
        )
        self.lifecycle_state = FSDPLifecycleState.UNSHARDED
        self._unshard_inflight = True
        self._record_full_param_buffer_snapshot(f"start_unshard:{reason}")
        return True

    def wait_unshard(self, reason: str) -> None:
        if self.flat_buffer is None or self.lifecycle_state != FSDPLifecycleState.UNSHARDED:
            return
        if self._unshard_inflight:
            if self._unshard_handle is None:
                raise RuntimeError("wait_unshard() was called without a pending unshard handle.")
            wait_start = perf_counter()
            self.flat_buffer.finish_all_gather_full_params(self._unshard_handle)
            wait_duration_ms = (perf_counter() - wait_start) * 1000.0
            self._unshard_handle = None
            self._record_event(f"wait_unshard:{reason}", duration_ms=wait_duration_ms)
            self._record_event("unshard")
            self._record_full_param_buffer_snapshot(f"wait_unshard:{reason}")
        self._unshard_inflight = False
        if self._pending_backward_context is not None and not self._pending_backward_context.pending_backward:
            self._pending_backward_context = None

    def reshard_after_forward(self, *, needs_pre_backward_unshard: bool | None = None) -> None:
        if self.flat_buffer is None or self.lifecycle_state != FSDPLifecycleState.UNSHARDED:
            return
        if needs_pre_backward_unshard is None:
            needs_pre_backward_unshard = torch.is_grad_enabled()
        target_state = (
            FSDPLifecycleState.FORWARD_RESHARDED
            if needs_pre_backward_unshard
            else FSDPLifecycleState.SHARDED
        )
        self.flat_buffer.use_local_shards(
            preserve_full_grad_buffer=self._defer_backward_reduce,
            preserve_grad_bucket_input=self._defer_backward_reduce,
            preserve_local_grad_shard=True,
        )
        self.flat_buffer.clear_full_params(
            shrink_storage=self.reshard_after_forward_enabled
            and needs_pre_backward_unshard
            and not self.use_saved_tensor_hooks
        )
        self.lifecycle_state = target_state
        self._record_event("reshard_after_forward")
        self._record_full_param_buffer_snapshot("reshard_after_forward")

    def finalize_backward(self) -> None:
        if self.flat_buffer is None:
            return
        if self._post_backward_reduce_handle is not None:
            self.wait_post_backward_reduce()
            return
        if self._no_sync_depth > 0:
            raise RuntimeError("finalize_backward() cannot run inside MatrixFSDP no_sync().")
        self._accumulate_copy_in_grad_bucket_if_needed()
        if self._defer_backward_reduce and self.lifecycle_state == FSDPLifecycleState.SHARDED:
            self._finalize_deferred_backward_from_sharded()
            return
        if self.backward_reduce_strategy in ("per_param", "per_param_allreduce") and self.flat_buffer.local_grad_accumulator is not None:
            finalize_start = perf_counter()
            self._finish_backward_with_local_grad_shard(
                self.flat_buffer.finish_local_grad_accumulator(),
                finalize_start=finalize_start,
            )
            return
        if self.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED:
            raise RuntimeError(
                "finalize_backward() cannot run while parameters are forward-resharded. "
                "Call unshard() before backward, or disable reshard_after_forward until "
                "pre-backward unshard support is available."
            )
        if self.lifecycle_state != FSDPLifecycleState.UNSHARDED:
            raise RuntimeError("finalize_backward() expects parameters to be unsharded.")
        finalize_start = perf_counter()
        if self.backward_reduce_strategy == "bucket_reduce_scatter":
            self._finish_backward_with_grad_bucket(finalize_start=finalize_start)
            return
        reduce_start = perf_counter()
        local_grad_shard = self.flat_buffer.reduce_full_grads_to_local_shard()
        reduce_duration_ms = (perf_counter() - reduce_start) * 1000.0
        self._record_event("reduce_grad", duration_ms=reduce_duration_ms)
        self._finish_backward_with_local_grad_shard(local_grad_shard, finalize_start=finalize_start)

    def _finish_backward_with_grad_bucket(self, *, finalize_start: float) -> None:
        if self.flat_buffer is None:
            return
        collect_start = perf_counter()
        grad_bucket = self.flat_buffer.collect_grad_bucket()
        self._record_event("collect_grad_bucket", duration_ms=(perf_counter() - collect_start) * 1000.0)
        had_local_grad_shard = self.flat_buffer.local_grad_shard is not None
        self.flat_buffer.use_local_shards(preserve_local_grad_shard=True)
        self.flat_buffer.clear_full_params(
            shrink_storage=self.shrink_full_param_storage_after_backward,
            keep_shrunk_tensor=False,
        )
        self.lifecycle_state = FSDPLifecycleState.SHARDED
        self._record_event("reshard_before_reduce_grad")
        self._record_full_param_buffer_snapshot("reshard_before_reduce_grad")
        if self._scheduler is not None:
            self._scheduler.before_backward_reduce(self)
        reduce_start = perf_counter()
        reduce_start_result = self.flat_buffer.start_reduce_grad_bucket_to_local_shard_with_stats(grad_bucket)
        reduce_handle = reduce_start_result.handle
        reduce_stats = reduce_start_result.stats
        self._post_backward_reduce_handle = reduce_handle
        self._post_backward_reduce_should_accumulate = had_local_grad_shard
        self._record_event(
            f"grad_bucket_layout:{reduce_stats.layout_kind}",
            duration_ms=0.0,
        )
        if reduce_stats.needs_copy_in:
            self._record_event(
                "copy_in_grad_bucket",
                duration_ms=reduce_stats.copy_in_ms,
                reduce_scatter_input_bytes=reduce_stats.packed_bytes,
            )
        else:
            self._record_event(
                "copy_in_grad_bucket_skipped",
                duration_ms=0.0,
                reduce_scatter_input_bytes=reduce_stats.packed_bytes,
            )
        self._record_event(
            "enqueue_reduce_scatter_grad_bucket",
            duration_ms=reduce_stats.reduce_scatter_enqueue_ms,
            reduce_scatter_input_bytes=reduce_stats.packed_bytes,
        )
        self._record_event("start_reduce_grad_bucket", duration_ms=(perf_counter() - reduce_start) * 1000.0)
        local_grad_shard = reduce_handle.tensor
        self._record_event("reduce_grad_bucket", duration_ms=(perf_counter() - reduce_start) * 1000.0)
        if not had_local_grad_shard:
            self.flat_buffer.use_local_grad_shard(local_grad_shard)
        if not self.finalize_after_backward_enabled:
            self.wait_post_backward_reduce()
        elif self._scheduler is not None:
            self._scheduler.after_backward_reduce_started(self)
            self._record_event("pending_backward_reduce")
            self._scheduler.on_post_backward_reshard(self)
        else:
            self.wait_post_backward_reduce()
        self._finish_backward_common(finalize_start=finalize_start)

    @property
    def has_pending_backward_reduce(self) -> bool:
        return self._post_backward_reduce_handle is not None

    def wait_post_backward_reduce(self) -> None:
        if self.flat_buffer is None or self._post_backward_reduce_handle is None:
            return
        wait_start = perf_counter()
        local_grad_shard = self._post_backward_reduce_handle.wait()
        self._post_backward_reduce_handle = None
        should_accumulate = self._post_backward_reduce_should_accumulate
        self._post_backward_reduce_should_accumulate = False
        if should_accumulate:
            accumulated = self.flat_buffer.accumulate_local_grad_shard(local_grad_shard)
            if accumulated:
                self._record_event("accumulate_local_grad_shard")
        self._record_event("wait_reduce_grad_bucket", duration_ms=(perf_counter() - wait_start) * 1000.0)

    def _finish_backward_with_local_grad_shard(self, local_grad_shard: torch.Tensor, *, finalize_start: float) -> None:
        if self.flat_buffer is None:
            return
        self.flat_buffer.use_local_shards(preserve_local_grad_shard=True)
        accumulated = self.flat_buffer.accumulate_local_grad_shard(local_grad_shard)
        if accumulated:
            self._record_event("accumulate_local_grad_shard")
        self.flat_buffer.clear_full_params(
            shrink_storage=self.shrink_full_param_storage_after_backward,
            keep_shrunk_tensor=False,
        )
        self._finish_backward_common(finalize_start=finalize_start)

    def _finish_backward_common(self, *, finalize_start: float) -> None:
        self.lifecycle_state = FSDPLifecycleState.SHARDED
        self._pending_backward_context = None
        self._post_backward_seen_param_ids.clear()
        self._checkpoint_recompute_forward_depth = 0
        self._defer_backward_reduce = False
        self._forward_prefetched = False
        self._backward_prefetched = False
        self._unshard_inflight = False
        self._unshard_handle = None
        if self.finalize_after_backward_enabled:
            self._finalized_after_backward = True
        self._record_event("finalize_backward", duration_ms=(perf_counter() - finalize_start) * 1000.0)

    def _finalize_deferred_backward_from_sharded(self) -> None:
        if self.flat_buffer is None:
            return
        finalize_start = perf_counter()
        if self.backward_reduce_strategy == "bucket_reduce_scatter":
            self._finish_backward_with_grad_bucket(finalize_start=finalize_start)
            return
        reduce_start = perf_counter()
        local_grad_shard = self.flat_buffer.reduce_full_grads_to_local_shard()
        self._record_event("reduce_grad", duration_ms=(perf_counter() - reduce_start) * 1000.0)
        self._finish_backward_with_local_grad_shard(local_grad_shard, finalize_start=finalize_start)

    def _reshard_after_deferred_no_sync_backward(self) -> None:
        if self.flat_buffer is None or self.lifecycle_state != FSDPLifecycleState.UNSHARDED:
            return
        self.flat_buffer.use_local_shards(
            preserve_full_grad_buffer=True,
            preserve_grad_bucket_input=True,
            preserve_local_grad_shard=True,
        )
        self.flat_buffer.clear_full_params(
            shrink_storage=self.shrink_full_param_storage_after_backward,
            keep_shrunk_tensor=False,
        )
        self.lifecycle_state = FSDPLifecycleState.SHARDED
        self._pending_backward_context = None
        self._post_backward_seen_param_ids.clear()
        self._forward_prefetched = False
        self._backward_prefetched = False
        self._unshard_inflight = False
        self._unshard_handle = None
        self._record_event("reshard_after_no_sync_backward")
        self._record_full_param_buffer_snapshot("reshard_after_no_sync_backward")

    def state_dict(self) -> dict[str, object]:
        if self.flat_buffer is None:
            return {}
        matrix_shard_placement = self.flat_buffer.placement
        local_units = (
            matrix_shard_placement.local_units if matrix_shard_placement is not None else self.flat_buffer.shard_sizes
        )
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "dp_shard_mesh_dim": self.dp_shard_mesh_dim,
            "dp_replicate_mesh_dim": self.dp_replicate_mesh_dim,
            "device_mesh": self.device_mesh_metadata,
            "replicate_world_size": self.replicate_world_size,
            "local_start": self.flat_buffer.local_start,
            "local_end": self.flat_buffer.local_end,
            "local_segments": self.flat_buffer.local_segments,
            "local_shard": self.flat_buffer.local_shard,
            "local_shard_dtensor": self.flat_buffer.local_shard_dtensor,
            "sharded_param": self.flat_buffer.sharded_param,
            "local_grad_shard": self.flat_buffer.local_grad_shard,
            "local_grad_shard_dtensor": self.flat_buffer.local_grad_shard_dtensor,
            "sharded_grad": self.flat_buffer.sharded_grad,
            "full_grad_buffer": self.flat_buffer.full_grad_buffer,
            "param_data_alias_full_buffer": self.flat_buffer.param_data_alias_full_buffer(),
            "param_data_alias_local_shard": self.flat_buffer.param_data_alias_local_shard(),
            "param_grads_alias_full_grad_buffer": self.flat_buffer.param_grads_alias_full_grad_buffer(),
            "full_param_buffer_pool": self.flat_buffer.full_param_buffer_pool.stats(),
            "shard_sizes": self.flat_buffer.shard_sizes,
            "local_units": local_units,
            "matrix_shard_placement": matrix_shard_placement,
            "matrix_shard_compatibility": self.flat_buffer.placement_compatibility,
            "layout": self.group_layout,
            "group_layout": self.group_layout,
            "planner_layout": self.global_layout,
            "runtime_layout": self.group_layout,
            "planner_layout_contract": (
                self.planner_layout_contract.as_metadata()
                if self.planner_layout_contract is not None
                else None
            ),
            "runtime_layout_contract": (
                self.runtime_layout_contract.as_metadata()
                if self.runtime_layout_contract is not None
                else None
            ),
            "layout_flat_reordered": self.global_layout != self.group_layout,
            "runtime_layout_policy": self.runtime_layout_policy,
            "runtime_layout_mode": (
                self.runtime_layout_compatibility.mode if self.runtime_layout_compatibility is not None else None
            ),
            "runtime_layout_reason": (
                self.runtime_layout_compatibility.reason if self.runtime_layout_compatibility is not None else None
            ),
            "runtime_layout_requires_flat_reorder": (
                self.runtime_layout_compatibility.requires_flat_reorder
                if self.runtime_layout_compatibility is not None
                else False
            ),
            "runtime_layout_compatibility": self.runtime_layout_compatibility,
            "runtime_state": self.runtime_state,
            "no_sync_depth": self._no_sync_depth,
            "defer_backward_reduce": self._defer_backward_reduce,
            "planner_result": self.planner_result,
            "planner_evaluation": self.planner_evaluation,
            "planner_metadata": self.planner_result.as_metadata() if self.planner_result is not None else None,
            "planner_summary": self.planner_result.summary() if self.planner_result is not None else None,
            "planner_report": (
                self.planner_result.report.as_metadata()
                if self.planner_result is not None
                else None
            ),
            "planner_resource_estimate": (
                self.planner_result.resource_estimate.as_metadata()
                if self.planner_result is not None and self.planner_result.resource_estimate is not None
                else None
            ),
            "runtime_metadata": self.runtime_metadata,
            "runtime_param_group_id": self.runtime_metadata.runtime_param_group_id,
            "runtime_unit_id": self.runtime_metadata.runtime_unit_id,
            "planner_group_id": self.runtime_metadata.planner_group_id,
            "comm_buffer_id": self.runtime_metadata.comm_buffer_id,
            "shard_hints": {mp.fqn: mp.shard_hint for mp in self.managed_params},
            "lifecycle_state": self.lifecycle_state,
            "reshard_after_forward_policy": self.reshard_after_forward_policy,
            "reshard_after_forward_world_size": self.reshard_after_forward_world_size,
            "reshard_after_forward": self.reshard_after_forward_enabled,
            "forward_prefetch": self.forward_prefetch_enabled,
            "backward_prefetch": self.backward_prefetch_enabled,
            "finalize_after_backward": self.finalize_after_backward_enabled,
            "mixed_precision": mixed_precision_policy_metadata(self.mp_policy),
            "offload_policy": offload_policy_metadata(self.offload_policy),
            "param_gather_strategy": self.param_gather_strategy,
            "matrix_collective_backend": self.matrix_collective_backend,
            "backward_reduce_strategy": self.backward_reduce_strategy,
            "grad_reduce_strategy": self.backward_reduce_strategy,
            "has_pending_backward_reduce": self.has_pending_backward_reduce,
            "has_cuda_comm_stream": self._cuda_comm_stream is not None,
            "has_cuda_all_gather_stream": self._cuda_all_gather_stream is not None,
            "has_cuda_reduce_scatter_stream": self._cuda_reduce_scatter_stream is not None,
            "saved_full_param_views": self._saved_full_param_views,
            "runtime_events": tuple(self.runtime_events),
            "runtime_trace_enabled": self.runtime_trace_enabled,
            "shrink_full_param_storage_after_backward": self.shrink_full_param_storage_after_backward,
            "param_fqns": [mp.fqn for mp in self.managed_params],
        }

    @property
    def finalized_after_backward(self) -> bool:
        return self._finalized_after_backward

    @property
    def _is_unsharded(self) -> bool:
        return self.lifecycle_state == FSDPLifecycleState.UNSHARDED

    @property
    def runtime_state(self) -> FSDPRuntimeState:
        if self._post_backward_reduce_handle is not None:
            return FSDPRuntimeState.REDUCE_IN_FLIGHT
        if self._defer_backward_reduce:
            return FSDPRuntimeState.BACKWARD_DEFERRED
        if self.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED:
            return FSDPRuntimeState.FORWARD_RESHARDED
        if self.lifecycle_state == FSDPLifecycleState.UNSHARDED:
            if self._post_backward_seen_param_ids:
                return FSDPRuntimeState.UNSHARDED_BACKWARD
            return FSDPRuntimeState.UNSHARDED_FORWARD
        return FSDPRuntimeState.SHARDED

    @property
    def layout(self) -> MatrixGroupLayout | None:
        return self.group_layout

    def params_for_rank(self, rank: int | None = None) -> tuple[str, ...]:
        if self.runtime_layout_contract is None:
            return ()
        return self.runtime_layout_contract.params_for_rank(self.rank if rank is None else rank)

    def owner_ranks(self, fqn: str) -> tuple[int, ...]:
        if self.runtime_layout_contract is None:
            raise KeyError(fqn)
        return self.runtime_layout_contract.owner_ranks(fqn)

    def rank_segments_for_param(self, fqn: str, rank: int | None = None) -> tuple[ParamSegment, ...]:
        if self.runtime_layout_contract is None:
            raise KeyError(fqn)
        return self.runtime_layout_contract.rank_segments_for_param(self.rank if rank is None else rank, fqn)

    def no_sync(self) -> MatrixFSDPNoSync:
        return MatrixFSDPNoSync((self,))

    @property
    def is_no_sync_active(self) -> bool:
        return self._no_sync_depth > 0

    @property
    def has_deferred_backward_reduce(self) -> bool:
        return self._defer_backward_reduce

    def reset_grad_accumulation(self) -> None:
        self._defer_backward_reduce = False
        self._post_backward_seen_param_ids.clear()

    def set_scheduler(self, scheduler: MatrixFSDPScheduler | None) -> None:
        self._scheduler = scheduler

    def set_comm_context(self, comm_context: MatrixFSDPCommContext | None) -> None:
        if comm_context is None:
            return
        self._cuda_all_gather_stream = comm_context.all_gather_stream
        self._cuda_reduce_scatter_stream = comm_context.reduce_scatter_stream
        self._cuda_comm_stream = self._cuda_all_gather_stream
        if self.flat_buffer is not None:
            self.flat_buffer.set_cuda_streams(
                all_gather_stream=comm_context.all_gather_stream,
                reduce_scatter_stream=comm_context.reduce_scatter_stream,
            )

    def set_full_param_buffer_pool(self, pool: FullParamBufferPool) -> None:
        if self.flat_buffer is not None:
            self.flat_buffer.set_full_param_buffer_pool(pool)

    def prefetch_forward(self, *, validate_owner_collective_signature: bool = False) -> bool:
        if self.flat_buffer is None or self.lifecycle_state != FSDPLifecycleState.SHARDED:
            return False
        skip_reason = self.flat_buffer.owner_segment_prefetch_skip_reason()
        if skip_reason is not None:
            self._record_event(f"forward_prefetch_skipped:{skip_reason}")
            return False
        self._record_event("forward_prefetch")
        self._forward_prefetched = True
        return self.start_unshard(
            "forward_prefetch",
            validate_owner_collective_signature=validate_owner_collective_signature,
        )

    def prefetch_backward(self, *, validate_owner_collective_signature: bool = False) -> bool:
        if self.flat_buffer is None or self.lifecycle_state != FSDPLifecycleState.FORWARD_RESHARDED:
            return False
        skip_reason = self.flat_buffer.owner_segment_prefetch_skip_reason()
        if skip_reason is not None:
            self._record_event(f"backward_prefetch_skipped:{skip_reason}")
            return False
        self._record_event("backward_prefetch")
        self._backward_prefetched = True
        return self.start_unshard(
            "backward_prefetch",
            validate_owner_collective_signature=validate_owner_collective_signature,
        )

    def _pre_forward(
        self,
        module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        in_backward_graph_task = _is_in_backward_graph_task()
        if not in_backward_graph_task and self._checkpoint_recompute_forward_depth > 0:
            self._checkpoint_recompute_forward_depth = 0
            self._record_event("checkpoint_recompute_depth_reset")
        if (
            self.reshard_after_forward_enabled
            and in_backward_graph_task
            and self.lifecycle_state == FSDPLifecycleState.UNSHARDED
        ):
            self._checkpoint_recompute_forward_depth += 1
            self._record_event("checkpoint_recompute_pre_forward")
            return args, kwargs
        if self._pending_backward_context is not None and self._pending_backward_context.pending_backward:
            raise RuntimeError(
                "reshard_after_forward=True does not support running another forward before "
                "the previous forward has entered backward."
            )
        if (
            self.reshard_after_forward_enabled
            and self.lifecycle_state == FSDPLifecycleState.UNSHARDED
            and not self._forward_prefetched
        ):
            raise RuntimeError(
                "reshard_after_forward=True expects each forward/backward to be finalized "
                "before running another forward."
            )
        if self.reshard_after_forward_enabled and self.lifecycle_state not in (
            FSDPLifecycleState.SHARDED,
            FSDPLifecycleState.UNSHARDED,
        ):
            raise RuntimeError(
                "reshard_after_forward=True expects each forward/backward to be finalized "
                "before running another forward."
            )
        self._post_backward_seen_param_ids.clear()
        self._finalized_after_backward = False
        self._forward_prefetched = False
        self._backward_prefetched = False
        self._record_event("pre_forward")
        self.start_unshard("pre_forward")
        self.wait_unshard("pre_forward")
        self._prefetch_next_forward()
        if torch.is_grad_enabled() and not self.reshard_after_forward_enabled:
            self._prepare_backward_grad_storage()
        if self.reshard_after_forward_enabled and torch.is_grad_enabled():
            self._active_backward_context = ForwardBackwardContext(self._pre_backward_unshard)
            if self.use_saved_tensor_hooks:
                self._enter_saved_tensors_hooks()
        if self.cast_forward_inputs and self.param_dtype is not None:
            args = cast_floating_tensors(args, self.param_dtype)
            kwargs = cast_floating_tensors(kwargs, self.param_dtype)
        return args, kwargs

    def _post_forward(
        self,
        module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        output: object,
    ) -> object:
        if self._checkpoint_recompute_forward_depth > 0 and _is_in_backward_graph_task():
            self._checkpoint_recompute_forward_depth -= 1
            self._record_event("checkpoint_recompute_post_forward")
            return self._cast_forward_output(output)
        if self._checkpoint_recompute_forward_depth > 0:
            self._checkpoint_recompute_forward_depth = 0
            self._record_event("checkpoint_recompute_depth_reset")
        if not self.reshard_after_forward_enabled:
            return self._cast_forward_output(output)
        self._exit_saved_tensors_hooks()
        context = self._active_backward_context
        if torch.is_grad_enabled():
            if context is None:
                context = ForwardBackwardContext(self._pre_backward_unshard)
            registered_hooks = register_pre_backward_hooks_with_context(output, context)
            context.mark_registered(registered_hooks)
        else:
            registered_hooks = 0
        self._active_backward_context = None
        self._pending_backward_context = context if context is not None and context.pending_backward else None
        self.reshard_after_forward(needs_pre_backward_unshard=registered_hooks > 0)
        if registered_hooks > 0 and self._scheduler is not None:
            self._scheduler.record_post_forward(self)
        return self._cast_forward_output(output)

    def _cast_forward_output(self, output: object) -> object:
        if self.output_dtype is None:
            return output
        return cast_floating_tensors(output, self.output_dtype)

    def _pre_backward_unshard(self) -> None:
        self._record_event("pre_backward_unshard")
        self.start_unshard("pre_backward")
        self.wait_unshard("pre_backward")
        self._backward_prefetched = False
        if self._scheduler is not None:
            self._scheduler.on_pre_backward(self)
        self._prepare_backward_grad_storage()

    def _prepare_backward_grad_storage(self) -> None:
        if self._no_sync_depth > 0:
            self._defer_backward_reduce = True
        if self.backward_reduce_strategy in ("per_param", "per_param_allreduce"):
            self._prepare_local_grad_accumulator()
            return
        if self.backward_reduce_strategy == "bucket_reduce_scatter":
            self._prepare_grad_bucket()
            return
        self._prepare_full_grad_buffer()

    def _prepare_full_grad_buffer(self) -> None:
        if self.flat_buffer is None:
            return
        reused = self.flat_buffer.prepare_full_grad_buffer(accumulate=self._defer_backward_reduce)
        if reused:
            self._record_event("reuse_full_grad_buffer_for_accumulation")
            return
        self._record_event("prepare_full_grad_buffer")

    def _prepare_local_grad_accumulator(self) -> None:
        if self.flat_buffer is None:
            return
        self.flat_buffer.prepare_local_grad_accumulator()
        self._record_event("prepare_local_grad_accumulator")

    def _prepare_grad_bucket(self) -> None:
        if self.flat_buffer is None:
            return
        had_accumulated_bucket = self._defer_backward_reduce and self.flat_buffer.grad_bucket_input is not None
        zero_copy = self.flat_buffer.prepare_grad_bucket(
            zero_copy=self.use_zero_copy_grad_bucket,
            accumulate=self._defer_backward_reduce,
        )
        self._grad_bucket_prepared_zero_copy = zero_copy
        self._record_event("prepare_grad_bucket")
        if zero_copy and had_accumulated_bucket:
            self._record_event("reuse_grad_bucket_for_accumulation")
        if zero_copy:
            self._record_event("prepare_grad_bucket_zero_copy")
        else:
            self._record_event("prepare_grad_bucket_copy_in")

    def _accumulate_copy_in_grad_bucket_if_needed(self) -> None:
        if (
            self.flat_buffer is None
            or self.backward_reduce_strategy != "bucket_reduce_scatter"
            or self._grad_bucket_prepared_zero_copy
            or not self._defer_backward_reduce
        ):
            return
        accumulated = self.flat_buffer.accumulate_grad_bucket_input_from_param_grads()
        self._record_event("copy_in_grad_bucket_for_accumulation")
        if accumulated:
            self._record_event("reuse_grad_bucket_for_accumulation")

    def _prefetch_next_forward(self) -> None:
        if self._scheduler is None:
            return
        self._scheduler.on_pre_forward(self)

    def _enter_saved_tensors_hooks(self) -> None:
        if self._saved_tensors_hooks_context is not None:
            raise RuntimeError("Saved tensor hooks are already active for this MatrixFSDP param group.")
        self._saved_tensors_hooks_context = torch.autograd.graph.saved_tensors_hooks(
            self._pack_saved_tensor,
            self._unpack_saved_tensor,
        )
        self._saved_tensors_hooks_context.__enter__()

    def _exit_saved_tensors_hooks(self) -> None:
        if self._saved_tensors_hooks_context is None:
            return
        self._saved_tensors_hooks_context.__exit__(None, None, None)
        self._saved_tensors_hooks_context = None

    def _pack_saved_tensor(self, tensor: torch.Tensor) -> torch.Tensor | _SavedFullParamView:
        if self.flat_buffer is None or self.flat_buffer.full_buffer is None:
            return tensor
        full_buffer = self.flat_buffer.full_buffer
        if tensor.untyped_storage().data_ptr() != full_buffer.untyped_storage().data_ptr():
            return tensor
        self._saved_full_param_views += 1
        return _SavedFullParamView(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
        )

    def _unpack_saved_tensor(self, saved: torch.Tensor | _SavedFullParamView) -> torch.Tensor:
        if not isinstance(saved, _SavedFullParamView):
            return saved
        context = self._pending_backward_context or self._active_backward_context
        if context is not None:
            context.pre_backward()
        elif self.lifecycle_state != FSDPLifecycleState.UNSHARDED:
            self.unshard()
        if self.flat_buffer is None or self.flat_buffer.full_buffer is None:
            raise RuntimeError("Saved full parameter view was unpacked before full parameters were available.")
        return self.flat_buffer.full_buffer.as_strided(
            size=saved.shape,
            stride=saved.stride,
            storage_offset=saved.storage_offset,
        )

    def _register_post_backward_hooks(self) -> None:
        expected_param_ids = set()
        param_by_id = {}
        for mp in self.managed_params:
            param = mp.param
            if not hasattr(param, "register_post_accumulate_grad_hook"):
                raise NotImplementedError(
                    "finalize_after_backward=True requires Tensor.register_post_accumulate_grad_hook()."
                )
            expected_param_ids.add(id(param))
            param_by_id[id(param)] = mp
            self._handles.append(param.register_post_accumulate_grad_hook(self._post_accumulate_grad))
        self._post_backward_expected_param_ids = expected_param_ids
        self._post_backward_param_by_id = param_by_id

    def _post_accumulate_grad(self, param: torch.Tensor) -> None:
        if self.backward_reduce_strategy in ("per_param", "per_param_allreduce"):
            self._reduce_post_accumulated_param_grad(param)
        self._post_backward_seen_param_ids.add(id(param))
        if self._post_backward_seen_param_ids != self._post_backward_expected_param_ids:
            return
        if self._no_sync_depth > 0:
            self._defer_backward_reduce = True
            self._record_event("defer_backward_reduce:no_sync")
            self._accumulate_copy_in_grad_bucket_if_needed()
            if self.reshard_after_forward_enabled:
                self._reshard_after_deferred_no_sync_backward()
            return
        self._accumulate_copy_in_grad_bucket_if_needed()
        if self.lifecycle_state != FSDPLifecycleState.UNSHARDED:
            return
        self.finalize_backward()

    def _enter_no_sync(self) -> None:
        if self.backward_reduce_strategy not in ("flat", "bucket_reduce_scatter"):
            raise NotImplementedError(
                "MatrixFSDP no_sync() currently supports only 'flat' and 'bucket_reduce_scatter' grad paths."
            )
        self._no_sync_depth += 1
        self._record_event("no_sync_enter")

    def _exit_no_sync(self) -> None:
        if self._no_sync_depth == 0:
            raise RuntimeError("MatrixFSDP no_sync() exit without a matching enter.")
        self._no_sync_depth -= 1
        self._record_event("no_sync_exit")
        if self._no_sync_depth == 0 and self._defer_backward_reduce:
            self._accumulate_copy_in_grad_bucket_if_needed()
            if self.reshard_after_forward_enabled:
                self._reshard_after_deferred_no_sync_backward()

    def _reduce_post_accumulated_param_grad(self, param: torch.Tensor) -> None:
        if self.flat_buffer is None:
            return
        managed_param = self._post_backward_param_by_id.get(id(param))
        if managed_param is None:
            return
        reduce_start = perf_counter()
        if self.backward_reduce_strategy == "per_param_allreduce":
            self.flat_buffer.all_reduce_param_grad_to_local_accumulator(managed_param)
        else:
            self.flat_buffer.reduce_param_grad_to_local_accumulator(managed_param)
        self._record_event(f"reduce_param_grad:{managed_param.fqn}", duration_ms=(perf_counter() - reduce_start) * 1000.0)

    def _record_event(
        self,
        name: str,
        *,
        duration_ms: float | None = None,
        reduce_scatter_input_bytes: int = 0,
    ) -> None:
        if not self.runtime_trace_enabled:
            return
        snapshot = self._runtime_memory_snapshot(name, reduce_scatter_input_bytes=reduce_scatter_input_bytes)
        self.runtime_events.append(
            RuntimeEvent(
                name=name,
                runtime_param_group_id=self.runtime_metadata.runtime_param_group_id,
                rank=self.rank,
                lifecycle_state=self.lifecycle_state.value,
                sequence=next_runtime_event_sequence(),
                duration_ms=duration_ms,
                active_full_param_buffers=int(snapshot["active_full_param_buffers"]),
                active_full_param_numel=int(snapshot["active_full_param_numel"]),
                active_full_param_bytes=int(snapshot["active_full_param_bytes"]),
                unit_full_param_bytes=int(snapshot["unit_full_param_bytes"]),
                unit_grad_bucket_bytes=int(snapshot["unit_grad_bucket_bytes"]),
                unit_reduce_scatter_input_bytes=int(snapshot["unit_reduce_scatter_input_bytes"]),
                unit_local_grad_shard_bytes=int(snapshot["unit_local_grad_shard_bytes"]),
                pending_backward_reduces=int(snapshot["pending_backward_reduces"]),
                param_data_alias_full_buffer=bool(snapshot["param_data_alias_full_buffer"]),
                param_data_alias_local_shard=bool(snapshot["param_data_alias_local_shard"]),
            )
        )

    def _record_full_param_buffer_snapshot(self, reason: str) -> None:
        if self.runtime_trace_enabled and self._scheduler is not None:
            self._scheduler.record_full_param_buffer_snapshot(reason)

    def _runtime_memory_snapshot(self, reason: str, *, reduce_scatter_input_bytes: int = 0) -> dict[str, int | str | bool]:
        if self._scheduler is not None:
            return self._scheduler.record_runtime_memory_snapshot(
                reason,
                self,
                reduce_scatter_input_bytes=reduce_scatter_input_bytes,
            )
        full_param_bytes, grad_bucket_bytes, local_grad_shard_bytes = self._local_runtime_buffer_bytes()
        active_full_param_buffers = 1 if full_param_bytes else 0
        active_full_param_numel = 0
        if (
            full_param_bytes
            and self.flat_buffer is not None
            and self.flat_buffer.full_buffer is not None
        ):
            active_full_param_numel = int(self.flat_buffer.full_buffer.numel())
        return {
            "reason": reason,
            "active_full_param_buffers": active_full_param_buffers,
            "active_full_param_numel": active_full_param_numel,
            "active_full_param_bytes": full_param_bytes,
            "unit_full_param_bytes": full_param_bytes,
            "unit_grad_bucket_bytes": grad_bucket_bytes,
            "unit_reduce_scatter_input_bytes": reduce_scatter_input_bytes,
            "unit_local_grad_shard_bytes": local_grad_shard_bytes,
            "pending_backward_reduces": 0,
            "param_data_alias_full_buffer": self.flat_buffer.param_data_alias_full_buffer()
            if self.flat_buffer is not None
            else False,
            "param_data_alias_local_shard": self.flat_buffer.param_data_alias_local_shard()
            if self.flat_buffer is not None
            else False,
        }

    def _local_runtime_buffer_bytes(self) -> tuple[int, int, int]:
        if self.flat_buffer is None:
            return 0, 0, 0
        return (
            _tensor_nbytes(self.flat_buffer.full_buffer),
            _tensor_nbytes(self.flat_buffer.grad_bucket_input),
            _tensor_nbytes(self.flat_buffer.local_grad_shard),
        )

    def _build_planner_layout_contract(self, total_numel: int) -> PlannerLayoutContract:
        if self.group_planner is None:
            plan = self.planner(total_numel, self.world_size)
            result = call_group_planner(
                lambda _params, _world_size: plan,
                self.managed_params,
                self.world_size,
                name=planner_display_name(self.planner),
                shard_mesh_dim=self.dp_shard_mesh_dim,
            )
            self.planner_result = result
            self.planner_evaluation = result
            return result.layout_contract()
        layout_or_plan = self.group_planner(self.managed_params, self.world_size)
        result = call_group_planner(
            lambda _params, _world_size: layout_or_plan,
            self.managed_params,
            self.world_size,
            name=planner_display_name(self.group_planner),
            shard_mesh_dim=self.dp_shard_mesh_dim,
        )
        self.planner_result = result
        self.planner_evaluation = result
        return result.layout_contract()

    def _prepare_runtime_layout(self, contract: PlannerLayoutContract) -> PlannerLayoutContract:
        contract.validate()
        layout = contract.layout
        compatibility = explain_runtime_layout_compatibility(
            layout,
            self.managed_params,
            self.world_size,
            allow_flat_reorder=self.runtime_layout_policy == "auto",
            allow_segment_runtime=self.runtime_layout_policy != "matrix_shard_only",
        )
        self.runtime_layout_compatibility = compatibility
        if not compatibility.compatible:
            raise ValueError(
                "MatrixFSDP runtime cannot execute planner layout under "
                f"runtime_layout_policy={self.runtime_layout_policy!r}. "
                f"Runtime mode={compatibility.mode!r}. Reason: {compatibility.reason}."
            )
        if compatibility.mode in ("matrix_shard", "segment_runtime"):
            return contract
        runtime_layout = self._reorder_whole_param_layout_for_matrix_shard(layout, compatibility.matrix_shard_reason)
        return PlannerLayoutContract(runtime_layout)

    def _reorder_whole_param_layout_for_matrix_shard(
        self,
        layout: MatrixGroupLayout,
        reason: str | None,
    ) -> MatrixGroupLayout:
        params_by_fqn = {param.fqn: param for param in self.managed_params}
        rank_entries: list[list[tuple[int, ParamLayout, ManagedParam]]] = [[] for _ in range(layout.world_size)]
        for param_layout in layout.params:
            managed_param = params_by_fqn[param_layout.fqn]
            if len(param_layout.segments) != 1:
                raise ValueError(
                    "MatrixFSDP runtime can flat-reorder only whole-parameter owner layouts. "
                    f"Param {param_layout.fqn!r} has {len(param_layout.segments)} segments. "
                    f"Original incompatibility: {reason}."
                )
            segment = param_layout.segments[0]
            if segment.numel != managed_param.numel:
                raise ValueError(
                    "MatrixFSDP runtime can flat-reorder only whole-parameter owner layouts. "
                    f"Param {param_layout.fqn!r} segment has {segment.numel} elements, "
                    f"expected {managed_param.numel}."
                )
            rank_entries[segment.rank].append((segment.local_start, param_layout, managed_param))

        new_managed_params: list[ManagedParam] = []
        new_param_layouts: list[ParamLayout] = []
        new_rank_segments: list[tuple[LayoutSegment, ...]] = []
        global_cursor = 0
        for rank, entries in enumerate(rank_entries):
            rank_start = global_cursor
            local_cursor = 0
            for _, _, managed_param in sorted(entries, key=lambda item: item[0]):
                new_start = global_cursor
                new_end = new_start + managed_param.numel
                new_managed_param = replace(managed_param, offset=new_start, end=new_end)
                new_managed_params.append(new_managed_param)
                new_param_layouts.append(
                    ParamLayout(
                        fqn=managed_param.fqn,
                        global_start=new_start,
                        global_end=new_end,
                        segments=(
                            ParamSegment(
                                fqn=managed_param.fqn,
                                rank=rank,
                                global_start=new_start,
                                global_end=new_end,
                                local_start=local_cursor,
                            ),
                        ),
                    )
                )
                local_cursor += managed_param.numel
                global_cursor = new_end
            if local_cursor == 0:
                new_rank_segments.append(())
            else:
                new_rank_segments.append((LayoutSegment(rank_start, global_cursor, 0),))

        if len(new_managed_params) != len(self.managed_params):
            raise ValueError("Flat reorder did not preserve all managed parameters.")

        self.managed_params = new_managed_params
        self.param_registry = ManagedParamRegistry(tuple(new_managed_params))
        return MatrixGroupLayout.from_rank_segments(
            total_numel=layout.total_numel,
            rank_segments=tuple(new_rank_segments),
            params=tuple(new_param_layouts),
        )

    def _validate_managed_params(self) -> None:
        first = self.managed_params[0]
        validate_offload_policy_for_device(self.offload_policy, first.device)
        for mp in self.managed_params:
            if mp.dtype != first.dtype:
                raise NotImplementedError("MatrixFSDP V0 only supports one dtype per unit.")
            if mp.device != first.device:
                raise NotImplementedError("MatrixFSDP V0 only supports one device per unit.")
            if not mp.param.requires_grad:
                raise NotImplementedError("MatrixFSDP V0 expects all managed parameters to require grad.")

    def _validate_reshard_after_forward_policy(self) -> None:
        if self.reshard_after_forward_world_size is None:
            return
        target_world_size = self.reshard_after_forward_world_size
        if target_world_size <= 1 or target_world_size >= self.world_size or self.world_size % target_world_size != 0:
            raise ValueError(
                "reshard_after_forward as an int must be a non-trivial divisor of "
                f"the shard world size; got {target_world_size} for world_size={self.world_size}."
            )
        raise NotImplementedError(
            "reshard_after_forward=int requires post-forward subgroup resharding. "
            "MatrixFSDP currently supports bool reshard_after_forward only."
        )

    def _create_cuda_comm_streams(self) -> tuple[torch.cuda.Stream | None, torch.cuda.Stream | None]:
        device = self.managed_params[0].device
        if device.type != "cuda":
            return None, None
        with torch.cuda.device(device):
            all_gather_stream = torch.cuda.Stream(device=device, priority=-1)
            reduce_scatter_stream = torch.cuda.Stream(device=device, priority=-1)
        return all_gather_stream, reduce_scatter_stream

    def _validate_plan(self, plan: ShardPlan, total_numel: int) -> None:
        if plan.total_numel != total_numel:
            raise ValueError(f"Planner returned total_numel={plan.total_numel}, expected {total_numel}.")
        if len(plan.shard_sizes) != self.world_size:
            raise ValueError(f"Planner returned {len(plan.shard_sizes)} shards for world_size={self.world_size}.")
        if len(plan.shard_offsets) != self.world_size:
            raise ValueError(f"Planner returned {len(plan.shard_offsets)} offsets for world_size={self.world_size}.")
        if sum(plan.shard_sizes) != total_numel:
            raise ValueError("Planner shard sizes must sum to total_numel.")
        if len(plan.rank_segments) != self.world_size:
            raise ValueError(f"Planner returned {len(plan.rank_segments)} segment lists for world_size={self.world_size}.")
        for rank, segments in enumerate(plan.rank_segments):
            if sum(segment.numel for segment in segments) != plan.shard_sizes[rank]:
                raise ValueError(f"Planner segments for rank {rank} must sum to its shard size.")

    def _validate_layout_contract(self, contract: PlannerLayoutContract, total_numel: int) -> None:
        if contract.total_numel != total_numel:
            raise ValueError(f"Planner returned total_numel={contract.total_numel}, expected {total_numel}.")
        contract.validate()
        validate_group_layout(contract.layout, self.managed_params, self.world_size)

    def _get_group(self, mesh: DeviceMesh | None, mesh_dim: int | None):
        if mesh is None or mesh_dim is None:
            return None
        return mesh.get_group(mesh_dim)

    def _get_rank(self, mesh: DeviceMesh | None, mesh_dim: int) -> int:
        if mesh is None or not (dist.is_available() and dist.is_initialized()):
            return 0
        coordinate = mesh.get_coordinate()
        if coordinate is None:
            raise RuntimeError("Current rank is not part of the provided device mesh.")
        return coordinate[mesh_dim]

    def _get_world_size(self, mesh: DeviceMesh | None, mesh_dim: int | None) -> int:
        if mesh is None or mesh_dim is None:
            return 1
        return mesh.size(mesh_dim)


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    logical_nbytes = int(tensor.numel() * tensor.element_size())
    storage_nbytes = int(tensor.untyped_storage().nbytes())
    return min(logical_nbytes, storage_nbytes)
