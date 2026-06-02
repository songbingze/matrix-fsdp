from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import Literal

import torch
import torch.distributed as dist

from matrix_fsdp.runtime.buffer_pool import FullParamBufferPool
from matrix_fsdp.runtime.param_group import MatrixFSDPParamGroup

PrefetchPolicy = Literal["static", "adaptive", "profile_guided"]
BackwardPrefetchTiming = Literal["pre_backward", "post_reshard"]
DEFAULT_MAX_UNSHARDED_PREFETCH_UNITS = 1


@dataclass(frozen=True)
class PrefetchProfileResult:
    budget: int | None
    avg_step_ms: float
    peak_memory_mb: float


@dataclass(frozen=True)
class MatrixFSDPSchedulerConfig:
    max_unsharded_prefetch_units: int | None = None
    max_forward_prefetch_units: int | None = None
    max_backward_prefetch_units: int | None = None
    backward_prefetch_timing: BackwardPrefetchTiming = "pre_backward"
    prefetch_policy: PrefetchPolicy = "static"
    max_cached_full_param_buffers_per_key: int = 0
    max_cached_elastic_workspaces_per_key: int = 0
    max_active_full_param_buffers: int | None = None
    max_active_full_param_numel: int | None = None
    max_active_full_param_memory_mb: float | None = None
    max_pending_backward_reduces: int | None = 1
    trim_cuda_cache: bool = False
    cuda_cache_trim_threshold_mb: float = 1024.0

    def as_scheduler_kwargs(self) -> dict[str, object]:
        return {
            "max_unsharded_prefetch_units": self.max_unsharded_prefetch_units,
            "max_forward_prefetch_units": self.max_forward_prefetch_units,
            "max_backward_prefetch_units": self.max_backward_prefetch_units,
            "backward_prefetch_timing": self.backward_prefetch_timing,
            "prefetch_policy": self.prefetch_policy,
            "max_cached_full_param_buffers_per_key": self.max_cached_full_param_buffers_per_key,
            "max_cached_elastic_workspaces_per_key": self.max_cached_elastic_workspaces_per_key,
            "max_active_full_param_buffers": self.max_active_full_param_buffers,
            "max_active_full_param_numel": self.max_active_full_param_numel,
            "max_active_full_param_memory_mb": self.max_active_full_param_memory_mb,
            "max_pending_backward_reduces": self.max_pending_backward_reduces,
            "trim_cuda_cache": self.trim_cuda_cache,
            "cuda_cache_trim_threshold_mb": self.cuda_cache_trim_threshold_mb,
        }


class MatrixFSDPCommContext:
    """
    Shared CUDA streams for FSDP2-like collective scheduling.

    FSDP2 keeps all-gather and reduce-scatter on separate high-priority
    streams. MatrixFSDP does the same while leaving parameter views owned by
    each unit's flat buffer.
    """

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("MatrixFSDPCommContext requires a CUDA device.")
        with torch.cuda.device(device):
            self.all_gather_stream = torch.cuda.Stream(device=device, priority=-1)
            self.reduce_scatter_stream = torch.cuda.Stream(device=device, priority=-1)


class MatrixFSDPScheduler:
    """
    Coordinates unit-level lifecycle actions.

    The scheduler owns prefetch policy while each unit owns its own collective
    handles and parameter views.
    """

    def __init__(
        self,
        units: Sequence[MatrixFSDPParamGroup],
        *,
        max_unsharded_prefetch_units: int | None = None,
        max_forward_prefetch_units: int | None = None,
        max_backward_prefetch_units: int | None = None,
        backward_prefetch_timing: BackwardPrefetchTiming = "pre_backward",
        prefetch_policy: PrefetchPolicy = "static",
        max_cached_full_param_buffers_per_key: int = 0,
        max_cached_elastic_workspaces_per_key: int = 0,
        max_active_full_param_buffers: int | None = None,
        max_active_full_param_numel: int | None = None,
        max_active_full_param_memory_mb: float | None = None,
        max_pending_backward_reduces: int | None = 1,
        trim_cuda_cache: bool = False,
        cuda_cache_trim_threshold_mb: float = 1024.0,
    ) -> None:
        if max_unsharded_prefetch_units is not None and max_unsharded_prefetch_units < 0:
            raise ValueError("max_unsharded_prefetch_units must be non-negative.")
        if max_forward_prefetch_units is not None and max_forward_prefetch_units < 0:
            raise ValueError("max_forward_prefetch_units must be non-negative.")
        if max_backward_prefetch_units is not None and max_backward_prefetch_units < 0:
            raise ValueError("max_backward_prefetch_units must be non-negative.")
        if max_unsharded_prefetch_units is not None and max_forward_prefetch_units is not None:
            raise ValueError("Pass only one of max_unsharded_prefetch_units or max_forward_prefetch_units.")
        if max_cached_full_param_buffers_per_key < 0:
            raise ValueError("max_cached_full_param_buffers_per_key must be non-negative.")
        if max_cached_elastic_workspaces_per_key < 0:
            raise ValueError("max_cached_elastic_workspaces_per_key must be non-negative.")
        if max_active_full_param_buffers is not None and max_active_full_param_buffers <= 0:
            raise ValueError("max_active_full_param_buffers must be positive.")
        if max_active_full_param_numel is not None and max_active_full_param_numel <= 0:
            raise ValueError("max_active_full_param_numel must be positive.")
        if max_active_full_param_memory_mb is not None and max_active_full_param_memory_mb <= 0:
            raise ValueError("max_active_full_param_memory_mb must be positive.")
        if max_pending_backward_reduces is not None and max_pending_backward_reduces < 0:
            raise ValueError("max_pending_backward_reduces must be non-negative.")
        if cuda_cache_trim_threshold_mb < 0:
            raise ValueError("cuda_cache_trim_threshold_mb must be non-negative.")
        if prefetch_policy not in ("static", "adaptive", "profile_guided"):
            raise ValueError("prefetch_policy must be 'static', 'adaptive', or 'profile_guided'.")
        if backward_prefetch_timing not in ("pre_backward", "post_reshard"):
            raise ValueError("backward_prefetch_timing must be 'pre_backward' or 'post_reshard'.")
        self.units = tuple(sorted(units, key=_runtime_param_group_order_key))
        self.prefetch_policy = prefetch_policy
        self.backward_prefetch_timing = backward_prefetch_timing
        self.requested_max_unsharded_prefetch_units = max_unsharded_prefetch_units
        self.requested_max_forward_prefetch_units = (
            max_forward_prefetch_units
            if max_forward_prefetch_units is not None
            else max_unsharded_prefetch_units
        )
        self.requested_max_backward_prefetch_units = max_backward_prefetch_units
        self.max_forward_prefetch_units = self._resolve_forward_prefetch_budget(
            self.requested_max_forward_prefetch_units
        )
        self.max_backward_prefetch_units = self._resolve_backward_prefetch_budget(max_backward_prefetch_units)
        self.max_unsharded_prefetch_units = self.max_forward_prefetch_units
        self.selected_forward_prefetch_budget = self.max_forward_prefetch_units
        self.selected_backward_prefetch_budget = self.max_backward_prefetch_units
        self.selected_prefetch_budget = self.selected_forward_prefetch_budget
        self.profile_results: tuple[PrefetchProfileResult, ...] = ()
        self.full_param_buffer_pool = FullParamBufferPool(max_cached_per_key=max_cached_full_param_buffers_per_key)
        self.max_cached_elastic_workspaces_per_key = max_cached_elastic_workspaces_per_key
        self.max_active_full_param_buffers_limit = max_active_full_param_buffers
        self.max_active_full_param_numel_limit = max_active_full_param_numel
        self.max_active_full_param_bytes_limit = (
            int(max_active_full_param_memory_mb * 1024 * 1024)
            if max_active_full_param_memory_mb is not None
            else None
        )
        self.max_pending_backward_reduces = max_pending_backward_reduces
        self.trim_cuda_cache = trim_cuda_cache
        self.cuda_cache_trim_threshold_bytes = int(cuda_cache_trim_threshold_mb * 1024 * 1024)
        self.cuda_cache_trim_count = 0
        self.comm_context = self._create_comm_context()
        self.forward_prefetch_issued = 0
        self.backward_prefetch_issued = 0
        self.forward_prefetch_budget_blocked = 0
        self.backward_prefetch_budget_blocked = 0
        self.backward_prefetch_memory_deferred = 0
        self.backward_reduce_waits = 0
        self._pending_backward_reduce_units: list[MatrixFSDPParamGroup] = []
        self._deferred_backward_prefetch_unit: MatrixFSDPParamGroup | None = None
        self.full_param_buffer_snapshots: list[dict[str, int | str]] = []
        self.runtime_memory_snapshots: list[dict[str, int | str | bool]] = []
        self.max_active_full_param_buffers = 0
        self.max_active_full_param_numel = 0
        self.max_active_full_param_bytes = 0
        self.max_grad_bucket_bytes = 0
        self.max_reduce_scatter_input_bytes = 0
        self.max_local_grad_shard_bytes = 0
        self.max_pending_backward_reduce_count = 0
        self._post_forward_order: list[MatrixFSDPParamGroup] = []
        self._post_forward_indices_by_unit_id: dict[int, list[int]] = {}
        self._index_by_unit_id = {id(unit): index for index, unit in enumerate(self.units)}
        self._owner_forward_prefetch_next_index = 1
        self._owner_backward_prefetch_next_post_forward_index: int | None = None
        self._owner_ordered_prefetch_issued_since_sync = False
        self._owner_ordered_prefetch_enabled = self._distributed_unit_order_is_consistent()
        self._debug_owner_prefetch_unit_order()
        for unit in self.units:
            unit.set_scheduler(self)
            if hasattr(unit, "set_comm_context"):
                unit.set_comm_context(self.comm_context)
            if hasattr(unit, "set_full_param_buffer_pool"):
                unit.set_full_param_buffer_pool(self.full_param_buffer_pool)
            if hasattr(unit, "set_elastic_workspace_cache_limit"):
                unit.set_elastic_workspace_cache_limit(self.max_cached_elastic_workspaces_per_key)

    def next_forward_unit(self, unit: MatrixFSDPParamGroup) -> MatrixFSDPParamGroup | None:
        index = self._index_by_unit_id[id(unit)]
        if index + 1 >= len(self.units):
            return None
        return self.units[index + 1]

    def previous_backward_prefetch_unit(self, unit: MatrixFSDPParamGroup) -> MatrixFSDPParamGroup | None:
        target = self._previous_backward_prefetch_unit_from_post_forward_order(unit, consume=False)
        if target is not None:
            return target
        index = self._index_by_unit_id[id(unit)]
        if index == 0:
            return None
        return self.units[index - 1]

    def record_post_forward(self, unit: MatrixFSDPParamGroup) -> None:
        post_forward_index = len(self._post_forward_order)
        self._post_forward_order.append(unit)
        self._post_forward_indices_by_unit_id.setdefault(id(unit), []).append(post_forward_index)

    def on_pre_forward(self, unit: MatrixFSDPParamGroup) -> None:
        if not unit.forward_prefetch_enabled:
            return
        if _is_in_backward_graph_task():
            return
        # Reentrant checkpoint recomputes module forwards during backward.
        # Those units are already in the post-forward order for this iteration.
        if id(unit) in self._post_forward_indices_by_unit_id:
            return
        target = self.next_forward_unit(unit)
        if target is None:
            return
        if self._uses_ordered_owner_prefetch(target):
            self._prefetch_next_forward_owner_ordered(target)
            return
        has_budget = self._has_forward_prefetch_budget()
        if not has_budget:
            self.forward_prefetch_budget_blocked += 1
            return
        if target.prefetch_forward():
            self.forward_prefetch_issued += 1

    def on_pre_backward(self, unit: MatrixFSDPParamGroup) -> None:
        if self.backward_prefetch_timing != "pre_backward":
            return
        self._prefetch_previous_backward_unit(unit)

    def on_post_backward_reshard(self, unit: MatrixFSDPParamGroup) -> None:
        if self._deferred_backward_prefetch_unit is not None:
            target = self._deferred_backward_prefetch_unit
            self._deferred_backward_prefetch_unit = None
            self._issue_backward_prefetch(target)
            return
        if self.backward_prefetch_timing != "post_reshard":
            return
        self._prefetch_previous_backward_unit(unit)

    def _prefetch_previous_backward_unit(self, unit: MatrixFSDPParamGroup) -> None:
        if not unit.backward_prefetch_enabled:
            return
        entry = self._previous_backward_prefetch_entry_from_post_forward_order(unit, consume=False)
        target = entry[1] if entry is not None else None
        if target is None:
            target = self.previous_backward_prefetch_unit(unit)
        if target is None:
            return
        if self._uses_ordered_owner_prefetch(target):
            if self._prefetch_previous_backward_owner_ordered(target, target_post_forward_index=entry[0] if entry else None):
                if entry is not None:
                    self._previous_backward_prefetch_entry_from_post_forward_order(unit, consume=True)
            return
        if entry is not None:
            self._previous_backward_prefetch_entry_from_post_forward_order(unit, consume=True)
        has_budget = self._has_backward_prefetch_budget()
        has_memory = not (
            self.backward_prefetch_timing == "pre_backward"
            and not self._has_full_param_buffer_budget_for_prefetch(target)
        )
        if not has_budget:
            self.backward_prefetch_budget_blocked += 1
            return
        if not has_memory:
            self.backward_prefetch_memory_deferred += 1
            self._deferred_backward_prefetch_unit = target
            return
        self._issue_backward_prefetch(target)

    def _uses_ordered_owner_prefetch(self, target: MatrixFSDPParamGroup) -> bool:
        flat_buffer = getattr(target, "flat_buffer", None)
        if flat_buffer is None:
            return False
        return (
            self._owner_ordered_prefetch_enabled
            and
            getattr(flat_buffer, "matrix_collective_backend", None) == "custom"
            and bool(flat_buffer.owner_segment_prefetch_order_gate_required())
        )

    def _prefetch_next_forward_owner_ordered(self, target: MatrixFSDPParamGroup) -> None:
        if self.max_forward_prefetch_units == 0:
            self.forward_prefetch_budget_blocked += 1
            return
        target_index = self._index_by_unit_id[id(target)]
        self._debug_owner_prefetch_queue("forward_attempt", target, target_index, self._owner_forward_prefetch_next_index)
        if target_index != self._owner_forward_prefetch_next_index:
            self._record_owner_prefetch_queue_skip(target, "forward", target_index, self._owner_forward_prefetch_next_index)
            return
        if target.prefetch_forward():
            self.forward_prefetch_issued += 1
            self._owner_ordered_prefetch_issued_since_sync = True
            self._owner_forward_prefetch_next_index += 1
            self._debug_owner_prefetch_queue("forward_issued", target, target_index, self._owner_forward_prefetch_next_index)

    def _prefetch_previous_backward_owner_ordered(
        self,
        target: MatrixFSDPParamGroup,
        *,
        target_post_forward_index: int | None,
    ) -> bool:
        if self.max_backward_prefetch_units == 0:
            self.backward_prefetch_budget_blocked += 1
            return False
        if self._owner_ordered_prefetch_uses_rank_local_memory_caps():
            self.backward_prefetch_memory_deferred += 1
            self._record_owner_prefetch_queue_skip(target, "backward", -1, -1, reason="memory_cap")
            return False
        if target_post_forward_index is None:
            target_order_index = self._index_by_unit_id[id(target)]
            expected_index = target_order_index
        else:
            self._ensure_owner_backward_prefetch_cursor()
            target_order_index = target_post_forward_index
            expected_index = self._owner_backward_prefetch_next_post_forward_index
        self._debug_owner_prefetch_queue("backward_attempt", target, target_order_index, expected_index)
        if expected_index is None or target_order_index != expected_index:
            self._record_owner_prefetch_queue_skip(target, "backward", target_order_index, expected_index)
            return False
        if not self._issue_backward_prefetch(target):
            return False
        self._owner_ordered_prefetch_issued_since_sync = True
        if target_post_forward_index is not None and self._owner_backward_prefetch_next_post_forward_index is not None:
            self._owner_backward_prefetch_next_post_forward_index -= 1
        self._debug_owner_prefetch_queue(
            "backward_issued",
            target,
            target_order_index,
            self._owner_backward_prefetch_next_post_forward_index,
        )
        return True

    def _issue_backward_prefetch(
        self,
        target: MatrixFSDPParamGroup,
        *,
        validate_owner_collective_signature: bool = False,
    ) -> bool:
        if target.prefetch_backward(validate_owner_collective_signature=validate_owner_collective_signature):
            self.backward_prefetch_issued += 1
            return True
        return False

    def before_backward_reduce(self, unit: MatrixFSDPParamGroup) -> None:
        self._prune_completed_backward_reduces()
        if self.max_pending_backward_reduces is None:
            return
        if self.max_pending_backward_reduces == 0:
            self._wait_all_pending_backward_reduces()
            return
        while len(self._pending_backward_reduce_units) >= self.max_pending_backward_reduces:
            self._wait_oldest_pending_backward_reduce()

    def after_backward_reduce_started(self, unit: MatrixFSDPParamGroup) -> None:
        if not unit.has_pending_backward_reduce:
            return
        self._prune_completed_backward_reduces()
        if unit not in self._pending_backward_reduce_units:
            self._pending_backward_reduce_units.append(unit)
        self._update_pending_backward_reduce_high_water()
        if self.max_pending_backward_reduces == 0:
            self._wait_all_pending_backward_reduces()
            return
        if self._index_by_unit_id[id(unit)] == 0:
            self._wait_all_pending_backward_reduces()
            return

    def finish_backward_iteration(self) -> None:
        self._synchronize_ordered_owner_prefetch_epoch()
        self._post_forward_order.clear()
        self._post_forward_indices_by_unit_id.clear()
        self._reset_ordered_prefetch_queues_for_forward()

    def wait_pending_backward_reduce(self) -> None:
        self._wait_all_pending_backward_reduces()

    def maybe_trim_cuda_cache(self) -> bool:
        if not self.trim_cuda_cache:
            return False
        if not torch.cuda.is_available():
            return False
        device = self._cuda_device()
        if device is None:
            return False
        active_count, _, _ = self._active_full_param_buffer_stats()
        if active_count != 0 or self.pending_backward_reduce_count != 0:
            return False
        with torch.cuda.device(device):
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            if reserved - allocated < self.cuda_cache_trim_threshold_bytes:
                return False
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self.cuda_cache_trim_count += 1
        return True

    def _has_forward_prefetch_budget(self) -> bool:
        if self.max_forward_prefetch_units == 0:
            return False
        if self.max_forward_prefetch_units is None:
            return True
        forward_prefetched_units = sum(1 for unit in self.units if getattr(unit, "_forward_prefetched", False))
        return forward_prefetched_units < self.max_forward_prefetch_units

    def _has_backward_prefetch_budget(self) -> bool:
        if self.max_backward_prefetch_units == 0:
            return False
        if self.max_backward_prefetch_units is None:
            return True
        backward_prefetched_units = sum(1 for unit in self.units if getattr(unit, "_backward_prefetched", False))
        return backward_prefetched_units < self.max_backward_prefetch_units

    def _has_full_param_buffer_budget_for_prefetch(self, target: MatrixFSDPParamGroup) -> bool:
        active_count, active_numel, active_bytes = self._active_full_param_buffer_stats()
        target_numel, target_bytes = self._full_param_buffer_size(target)
        if (
            self.max_active_full_param_buffers_limit is not None
            and active_count + 1 > self.max_active_full_param_buffers_limit
        ):
            return False
        if (
            self.max_active_full_param_numel_limit is not None
            and active_numel + target_numel > self.max_active_full_param_numel_limit
        ):
            return False
        if (
            self.max_active_full_param_bytes_limit is not None
            and active_bytes + target_bytes > self.max_active_full_param_bytes_limit
        ):
            return False
        return True

    def _resolve_forward_prefetch_budget(self, requested_budget: int | None) -> int | None:
        if self.prefetch_policy == "static":
            return requested_budget
        if requested_budget is not None:
            return requested_budget
        if not self.units:
            return 0

        shard_size = max(self._unit_shard_size(unit) for unit in self.units)
        unit_count = len(self.units)
        if shard_size >= 4:
            return min(1, unit_count)
        return min(2, unit_count)

    def _resolve_backward_prefetch_budget(self, requested_budget: int | None) -> int | None:
        if requested_budget is not None:
            return requested_budget
        if not self.units:
            return 0
        if self.prefetch_policy == "adaptive":
            shard_size = max(self._unit_shard_size(unit) for unit in self.units)
            return 0 if shard_size >= 4 else 1
        return 1 if self.units else 0

    def set_prefetch_budget(self, budget: int | None) -> None:
        self.set_forward_prefetch_budget(budget)

    def set_forward_prefetch_budget(self, budget: int | None) -> None:
        if budget is not None and budget < 0:
            raise ValueError("prefetch budget must be non-negative.")
        self.max_forward_prefetch_units = budget
        self.max_unsharded_prefetch_units = budget
        self.selected_forward_prefetch_budget = budget
        self.selected_prefetch_budget = budget

    def set_backward_prefetch_budget(self, budget: int | None) -> None:
        if budget is not None and budget < 0:
            raise ValueError("prefetch budget must be non-negative.")
        self.max_backward_prefetch_units = budget
        self.selected_backward_prefetch_budget = budget

    def set_profile_results(self, results: Sequence[PrefetchProfileResult]) -> None:
        self.profile_results = tuple(results)

    def reset_runtime_counters(self) -> None:
        self.forward_prefetch_issued = 0
        self.backward_prefetch_issued = 0
        self.forward_prefetch_budget_blocked = 0
        self.backward_prefetch_budget_blocked = 0
        self.backward_prefetch_memory_deferred = 0
        self.backward_reduce_waits = 0
        self._pending_backward_reduce_units.clear()
        self._deferred_backward_prefetch_unit = None
        self._reset_ordered_prefetch_queues_for_forward()
        self.full_param_buffer_snapshots.clear()
        self.runtime_memory_snapshots.clear()
        self.max_active_full_param_buffers = 0
        self.max_active_full_param_numel = 0
        self.max_active_full_param_bytes = 0
        self.max_grad_bucket_bytes = 0
        self.max_local_grad_shard_bytes = 0
        self.max_pending_backward_reduce_count = 0
        self.finish_backward_iteration()

    def select_profiled_budget(self, *, memory_limit_mb: float | None = None) -> int | None:
        if not self.profile_results:
            return self.max_forward_prefetch_units
        candidates = self.profile_results
        if memory_limit_mb is not None:
            candidates = tuple(result for result in candidates if result.peak_memory_mb <= memory_limit_mb)
        if not candidates:
            candidates = self.profile_results
        selected = min(candidates, key=lambda result: (result.avg_step_ms, result.peak_memory_mb))
        self.set_forward_prefetch_budget(selected.budget)
        return selected.budget

    @property
    def budget_blocked_prefetch_count(self) -> int:
        return self.forward_prefetch_budget_blocked + self.backward_prefetch_budget_blocked

    def record_full_param_buffer_snapshot(self, reason: str) -> None:
        active_count, active_numel, active_bytes = self._active_full_param_buffer_stats()
        self.max_active_full_param_buffers = max(self.max_active_full_param_buffers, active_count)
        self.max_active_full_param_numel = max(self.max_active_full_param_numel, active_numel)
        self.max_active_full_param_bytes = max(self.max_active_full_param_bytes, active_bytes)
        self.full_param_buffer_snapshots.append(
            {
                "reason": reason,
                "active_count": active_count,
                "active_numel": active_numel,
                "active_bytes": active_bytes,
            }
        )

    def record_runtime_memory_snapshot(
        self,
        reason: str,
        unit: MatrixFSDPParamGroup,
        *,
        reduce_scatter_input_bytes: int = 0,
    ) -> dict[str, int | str | bool]:
        active_count, active_numel, active_bytes = self._active_full_param_buffer_stats()
        unit_full_param_bytes, unit_grad_bucket_bytes, unit_local_grad_shard_bytes = self._unit_runtime_buffer_bytes(
            unit
        )
        pending_backward_reduces = self.pending_backward_reduce_count
        flat_buffer = unit.flat_buffer
        snapshot = {
            "reason": reason,
            "active_full_param_buffers": active_count,
            "active_full_param_numel": active_numel,
            "active_full_param_bytes": active_bytes,
            "unit_full_param_bytes": unit_full_param_bytes,
            "unit_grad_bucket_bytes": unit_grad_bucket_bytes,
            "unit_reduce_scatter_input_bytes": reduce_scatter_input_bytes,
            "unit_local_grad_shard_bytes": unit_local_grad_shard_bytes,
            "pending_backward_reduces": pending_backward_reduces,
            "param_data_alias_full_buffer": (
                flat_buffer.param_data_alias_full_buffer() if flat_buffer is not None else False
            ),
            "param_data_alias_local_shard": (
                flat_buffer.param_data_alias_local_shard() if flat_buffer is not None else False
            ),
        }
        self.runtime_memory_snapshots.append(snapshot)
        self._update_runtime_memory_high_water(snapshot)
        return snapshot

    def _active_full_param_buffer_stats(self) -> tuple[int, int, int]:
        active_count = 0
        active_numel = 0
        active_bytes = 0
        for unit in self.units:
            flat_buffer = getattr(unit, "flat_buffer", None)
            full_buffer = getattr(flat_buffer, "full_buffer", None)
            if full_buffer is None:
                continue
            full_buffer_bytes = _tensor_nbytes(full_buffer)
            if full_buffer_bytes == 0:
                continue
            active_count += 1
            active_numel += int(full_buffer.numel())
            active_bytes += full_buffer_bytes
        return active_count, active_numel, active_bytes

    @property
    def pending_backward_reduce_count(self) -> int:
        self._prune_completed_backward_reduces()
        return len(self._pending_backward_reduce_units)

    def _update_runtime_memory_high_water(self, snapshot: dict[str, int | str | bool]) -> None:
        self.max_active_full_param_buffers = max(
            self.max_active_full_param_buffers,
            int(snapshot["active_full_param_buffers"]),
        )
        self.max_active_full_param_numel = max(
            self.max_active_full_param_numel,
            int(snapshot["active_full_param_numel"]),
        )
        self.max_active_full_param_bytes = max(
            self.max_active_full_param_bytes,
            int(snapshot["active_full_param_bytes"]),
        )
        self.max_grad_bucket_bytes = max(self.max_grad_bucket_bytes, int(snapshot["unit_grad_bucket_bytes"]))
        self.max_reduce_scatter_input_bytes = max(
            self.max_reduce_scatter_input_bytes,
            int(snapshot["unit_reduce_scatter_input_bytes"]),
        )
        self.max_local_grad_shard_bytes = max(
            self.max_local_grad_shard_bytes,
            int(snapshot["unit_local_grad_shard_bytes"]),
        )
        self.max_pending_backward_reduce_count = max(
            self.max_pending_backward_reduce_count,
            int(snapshot["pending_backward_reduces"]),
        )

    def _update_pending_backward_reduce_high_water(self) -> None:
        self.max_pending_backward_reduce_count = max(
            self.max_pending_backward_reduce_count,
            len(self._pending_backward_reduce_units),
        )

    def _prune_completed_backward_reduces(self) -> None:
        self._pending_backward_reduce_units = [
            unit for unit in self._pending_backward_reduce_units if unit.has_pending_backward_reduce
        ]

    def _wait_oldest_pending_backward_reduce(self) -> None:
        self._prune_completed_backward_reduces()
        if not self._pending_backward_reduce_units:
            return
        unit = self._pending_backward_reduce_units.pop(0)
        if not unit.has_pending_backward_reduce:
            return
        unit.wait_post_backward_reduce()
        self.backward_reduce_waits += 1

    def _wait_all_pending_backward_reduces(self) -> None:
        self._prune_completed_backward_reduces()
        while self._pending_backward_reduce_units:
            self._wait_oldest_pending_backward_reduce()

    def _full_param_buffer_size(self, unit: MatrixFSDPParamGroup) -> tuple[int, int]:
        flat_buffer = getattr(unit, "flat_buffer", None)
        if flat_buffer is None:
            return 0, 0
        numel = int(flat_buffer.plan.total_numel)
        return numel, numel * int(flat_buffer.local_shard.element_size())

    def _unit_runtime_buffer_bytes(self, unit: MatrixFSDPParamGroup) -> tuple[int, int, int]:
        flat_buffer = getattr(unit, "flat_buffer", None)
        if flat_buffer is None:
            return 0, 0, 0
        return (
            _tensor_nbytes(getattr(flat_buffer, "full_buffer", None)),
            _tensor_nbytes(getattr(flat_buffer, "grad_bucket_input", None)),
            _tensor_nbytes(getattr(flat_buffer, "local_grad_shard", None)),
        )

    def _unit_shard_size(self, unit: MatrixFSDPParamGroup) -> int:
        mesh = getattr(unit, "mesh", None)
        shard_mesh_dim = getattr(unit, "dp_shard_mesh_dim", None)
        if mesh is not None and shard_mesh_dim is not None:
            try:
                return int(mesh.size(shard_mesh_dim))
            except (AttributeError, TypeError, ValueError):
                mesh_tensor = getattr(mesh, "mesh", None)
                if mesh_tensor is not None:
                    return int(mesh_tensor.size(int(shard_mesh_dim)))
        return int(getattr(unit, "world_size", 1))

    def _previous_backward_prefetch_unit_from_post_forward_order(
        self,
        unit: MatrixFSDPParamGroup,
        *,
        consume: bool,
    ) -> MatrixFSDPParamGroup | None:
        entry = self._previous_backward_prefetch_entry_from_post_forward_order(unit, consume=consume)
        return entry[1] if entry is not None else None

    def _previous_backward_prefetch_entry_from_post_forward_order(
        self,
        unit: MatrixFSDPParamGroup,
        *,
        consume: bool,
    ) -> tuple[int, MatrixFSDPParamGroup] | None:
        indices = self._post_forward_indices_by_unit_id.get(id(unit))
        if not indices:
            return None
        current_index = indices[-1]
        if consume:
            indices.pop()
        if current_index <= 0:
            return None
        target_index = current_index - 1
        return target_index, self._post_forward_order[target_index]

    def _reset_ordered_prefetch_queues_for_forward(self) -> None:
        self._owner_forward_prefetch_next_index = 1
        self._owner_backward_prefetch_next_post_forward_index = None

    def _ensure_owner_backward_prefetch_cursor(self) -> None:
        if self._owner_backward_prefetch_next_post_forward_index is None:
            self._owner_backward_prefetch_next_post_forward_index = len(self._post_forward_order) - 2

    def _owner_ordered_prefetch_uses_rank_local_memory_caps(self) -> bool:
        return (
            self.max_active_full_param_buffers_limit is not None
            or self.max_active_full_param_numel_limit is not None
            or self.max_active_full_param_bytes_limit is not None
        )

    def _record_owner_prefetch_queue_skip(
        self,
        target: MatrixFSDPParamGroup,
        phase: str,
        actual_index: int,
        expected_index: int | None,
        *,
        reason: str = "order",
    ) -> None:
        record_event = getattr(target, "_record_event", None)
        if record_event is None:
            return
        record_event(
            f"{phase}_prefetch_skipped:owner_ordered_queue:{reason}:"
            f"actual={actual_index}:expected={expected_index}"
        )

    def _synchronize_ordered_owner_prefetch_epoch(self) -> None:
        if not self._owner_ordered_prefetch_issued_since_sync:
            return
        self._owner_ordered_prefetch_issued_since_sync = False
        self._barrier_ordered_owner_prefetch_group()

    def _barrier_ordered_owner_prefetch_group(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        group = self._owner_ordered_prefetch_group()
        dist.barrier(group=group)

    def _owner_ordered_prefetch_group(self):
        for unit in self.units:
            if self._uses_ordered_owner_prefetch(unit):
                return getattr(unit, "group", None)
        return None

    def _debug_owner_prefetch_queue(
        self,
        event: str,
        target: MatrixFSDPParamGroup,
        actual_index: int,
        expected_index: int | None,
    ) -> None:
        if os.environ.get("MATRIX_FSDP_DEBUG_OWNER_PREFETCH_QUEUE", "").lower() not in {"1", "true", "yes", "on"}:
            return
        rank = int(getattr(target, "rank", -1))
        target_id = str(getattr(target.runtime_metadata, "runtime_param_group_id", "unknown"))
        lifecycle = getattr(getattr(target, "lifecycle_state", None), "value", getattr(target, "lifecycle_state", None))
        path = f"/tmp/matrix_fsdp_owner_prefetch_rank{rank}.log"
        with open(path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"{event} target={target_id} actual={actual_index} expected={expected_index} "
                f"lifecycle={lifecycle} post_forward={len(self._post_forward_order)}\n"
            )

    def _debug_owner_prefetch_unit_order(self) -> None:
        if os.environ.get("MATRIX_FSDP_DEBUG_OWNER_PREFETCH_QUEUE", "").lower() not in {"1", "true", "yes", "on"}:
            return
        rank = _scheduler_rank(self.units)
        path = f"/tmp/matrix_fsdp_owner_prefetch_rank{rank}.log"
        ordered_ids = ",".join(str(unit.runtime_metadata.runtime_param_group_id) for unit in self.units)
        with open(path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"scheduler_init owner_enabled={self._owner_ordered_prefetch_enabled} units={ordered_ids}\n"
            )

    def _distributed_unit_order_is_consistent(self) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return True
        group = None
        for unit in self.units:
            group = getattr(unit, "group", None)
            if group is not None:
                break
        local_order = tuple(str(unit.runtime_metadata.runtime_param_group_id) for unit in self.units)
        gathered: list[object] = [None for _ in range(dist.get_world_size(group=group))]
        dist.all_gather_object(gathered, local_order, group=group)
        return all(order == local_order for order in gathered)

    def _create_comm_context(self) -> MatrixFSDPCommContext | None:
        device = self._cuda_device()
        if device is None:
            return None
        return MatrixFSDPCommContext(device)

    def _cuda_device(self) -> torch.device | None:
        for unit in self.units:
            flat_buffer = getattr(unit, "flat_buffer", None)
            if flat_buffer is None:
                continue
            local_shard = getattr(flat_buffer, "local_shard", None)
            if local_shard is not None and local_shard.device.type == "cuda":
                return local_shard.device
        return None


def _is_in_backward_graph_task() -> bool:
    current_graph_task_id = getattr(torch._C, "_current_graph_task_id", None)
    return current_graph_task_id is not None and current_graph_task_id() != -1


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    logical_nbytes = int(tensor.numel() * tensor.element_size())
    storage_nbytes = int(tensor.untyped_storage().nbytes())
    return min(logical_nbytes, storage_nbytes)


def _runtime_param_group_order_key(unit: MatrixFSDPParamGroup) -> tuple[str, int | str]:
    runtime_metadata = getattr(unit, "runtime_metadata", None)
    if runtime_metadata is None:
        return type(unit).__name__, id(unit)
    runtime_id = str(runtime_metadata.runtime_param_group_id)
    prefix, sep, suffix = runtime_id.rpartition("_")
    if sep and suffix.isdigit():
        return prefix, int(suffix)
    return runtime_id, runtime_id


def _scheduler_rank(units: Sequence[MatrixFSDPParamGroup]) -> int:
    for unit in units:
        try:
            return int(getattr(unit, "rank"))
        except (TypeError, ValueError):
            continue
    return -1


def configure_forward_prefetch(
    units: Sequence[MatrixFSDPParamGroup],
    *,
    scheduler_config: MatrixFSDPSchedulerConfig | None = None,
    max_unsharded_prefetch_units: int | None = None,
    max_forward_prefetch_units: int | None = None,
    max_backward_prefetch_units: int | None = None,
    prefetch_policy: PrefetchPolicy = "static",
    max_cached_full_param_buffers_per_key: int = 0,
    max_cached_elastic_workspaces_per_key: int = 0,
    max_active_full_param_buffers: int | None = None,
    max_active_full_param_numel: int | None = None,
    max_active_full_param_memory_mb: float | None = None,
    max_pending_backward_reduces: int | None = 1,
    trim_cuda_cache: bool = False,
    cuda_cache_trim_threshold_mb: float = 1024.0,
) -> MatrixFSDPScheduler:
    if scheduler_config is not None:
        _raise_if_scheduler_config_conflicts(
            scheduler_config,
            max_unsharded_prefetch_units=max_unsharded_prefetch_units,
            max_forward_prefetch_units=max_forward_prefetch_units,
            max_backward_prefetch_units=max_backward_prefetch_units,
            prefetch_policy=prefetch_policy,
            max_cached_full_param_buffers_per_key=max_cached_full_param_buffers_per_key,
            max_cached_elastic_workspaces_per_key=max_cached_elastic_workspaces_per_key,
            max_active_full_param_buffers=max_active_full_param_buffers,
            max_active_full_param_numel=max_active_full_param_numel,
            max_active_full_param_memory_mb=max_active_full_param_memory_mb,
            max_pending_backward_reduces=max_pending_backward_reduces,
            trim_cuda_cache=trim_cuda_cache,
            cuda_cache_trim_threshold_mb=cuda_cache_trim_threshold_mb,
        )
        return MatrixFSDPScheduler(units, **scheduler_config.as_scheduler_kwargs())
    return MatrixFSDPScheduler(
        units,
        max_unsharded_prefetch_units=max_unsharded_prefetch_units,
        max_forward_prefetch_units=max_forward_prefetch_units,
        max_backward_prefetch_units=max_backward_prefetch_units,
        prefetch_policy=prefetch_policy,
        max_cached_full_param_buffers_per_key=max_cached_full_param_buffers_per_key,
        max_cached_elastic_workspaces_per_key=max_cached_elastic_workspaces_per_key,
        max_active_full_param_buffers=max_active_full_param_buffers,
        max_active_full_param_numel=max_active_full_param_numel,
        max_active_full_param_memory_mb=max_active_full_param_memory_mb,
        max_pending_backward_reduces=max_pending_backward_reduces,
        trim_cuda_cache=trim_cuda_cache,
        cuda_cache_trim_threshold_mb=cuda_cache_trim_threshold_mb,
    )


def _raise_if_scheduler_config_conflicts(
    scheduler_config: MatrixFSDPSchedulerConfig,
    **kwargs: object,
) -> None:
    defaults = MatrixFSDPSchedulerConfig()
    default_kwargs = defaults.as_scheduler_kwargs()
    config_kwargs = scheduler_config.as_scheduler_kwargs()
    conflicts = [
        name
        for name, value in kwargs.items()
        if value != default_kwargs[name] and value != config_kwargs[name]
    ]
    if conflicts:
        joined = ", ".join(sorted(conflicts))
        raise ValueError(f"Pass either scheduler_config or explicit scheduler kwargs, not both: {joined}.")
