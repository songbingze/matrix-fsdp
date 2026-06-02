from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from torch import nn

from matrix_fsdp.runtime.param_group import MatrixFSDPParamGroup
from matrix_fsdp.core.mixed_precision import mixed_precision_policy_metadata, offload_policy_metadata
from matrix_fsdp.runtime.unit_collection import collect_param_groups


def summarize_param_groups(
    module_or_param_groups: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
) -> dict[str, Any]:
    param_groups = collect_param_groups(module_or_param_groups)
    if not param_groups:
        raise ValueError("Expected at least one MatrixFSDPParamGroup.")
    param_group_summaries = [_summarize_param_group(param_group) for param_group in param_groups]
    total_numel = sum(param_group_summary["total_numel"] for param_group_summary in param_group_summaries)
    local_numel = sum(param_group_summary["local_numel"] for param_group_summary in param_group_summaries)
    communication_summary = _aggregate_communication_summaries(param_group_summaries)
    return {
        "num_param_groups": len(param_group_summaries),
        "param_groups": param_group_summaries,
        "num_units": len(param_group_summaries),
        "total_numel": total_numel,
        "local_numel": local_numel,
        "rank_total_memory_bytes": _sum_rank_resource(param_group_summaries, "rank_memory_bytes"),
        "rank_total_comm_bytes": _sum_rank_resource(param_group_summaries, "rank_comm_bytes"),
        "rank_total_muon_param_bytes": _sum_rank_resource(param_group_summaries, "rank_muon_param_bytes"),
        "rank_total_adamw_param_bytes": _sum_rank_resource(param_group_summaries, "rank_adamw_param_bytes"),
        "rank_total_optimizer_bytes": _sum_rank_resource(param_group_summaries, "rank_optimizer_bytes"),
        "communication_summary": communication_summary,
        "units": param_group_summaries,
    }


def summarize_runtime_events(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
) -> dict[str, Any]:
    param_groups = collect_param_groups(module_or_units)
    if not param_groups:
        raise ValueError("Expected at least one MatrixFSDPParamGroup.")
    param_group_index_by_id = {id(param_group): index for index, param_group in enumerate(param_groups)}
    events = []
    for param_group in param_groups:
        for event in param_group.runtime_events:
            events.append(
                {
                    "sequence": event.sequence,
                    "name": event.name,
                    "runtime_param_group_id": event.runtime_param_group_id,
                    "runtime_unit_id": event.runtime_unit_id,
                    "param_group_index": param_group_index_by_id[id(param_group)],
                    "unit_index": param_group_index_by_id[id(param_group)],
                    "rank": event.rank,
                    "lifecycle_state": event.lifecycle_state,
                    "timestamp_ns": event.timestamp_ns,
                    "duration_ms": event.duration_ms,
                    "active_full_param_buffers": event.active_full_param_buffers,
                    "active_full_param_numel": event.active_full_param_numel,
                    "active_full_param_bytes": event.active_full_param_bytes,
                    "unit_full_param_bytes": event.unit_full_param_bytes,
                    "unit_grad_bucket_bytes": event.unit_grad_bucket_bytes,
                    "unit_reduce_scatter_input_bytes": event.unit_reduce_scatter_input_bytes,
                    "unit_local_grad_shard_bytes": event.unit_local_grad_shard_bytes,
                    "pending_backward_reduces": event.pending_backward_reduces,
                    "param_data_alias_full_buffer": event.param_data_alias_full_buffer,
                    "param_data_alias_local_shard": event.param_data_alias_local_shard,
                }
            )
    events.sort(key=lambda event_summary: event_summary["sequence"])
    schedulers = _summarize_schedulers(param_groups, param_group_index_by_id)
    param_group_summaries = [_summarize_param_group(param_group) for param_group in param_groups]
    return {
        "num_param_groups": len(param_groups),
        "num_units": len(param_groups),
        "num_events": len(events),
        "num_schedulers": len(schedulers),
        "events": events,
        "event_stats": _summarize_event_stats(events),
        "communication_event_stats": _summarize_communication_event_stats(events),
        "schedulers": schedulers,
        "communication_summary": _aggregate_communication_summaries(param_group_summaries),
    }


def format_param_group_summary(summary: dict[str, Any]) -> str:
    lines = [
        (
            "MatrixFSDP param groups: "
            f"{summary['num_param_groups']} total_numel={summary['total_numel']} "
            f"local_numel={summary['local_numel']}"
        )
    ]
    for param_group_summary in summary["param_groups"]:
        lines.append(
            "  "
            f"{param_group_summary['runtime_param_group_id']} "
            f"planner={param_group_summary['planner_group_id']} "
            f"comm={param_group_summary['comm_buffer_id']} "
            f"planner_name={param_group_summary['planner_name']} "
            f"policy={param_group_summary['planner_policy']} "
            f"runtime={param_group_summary['planner_runtime_mode']} "
            f"runtime_layout_policy={param_group_summary['runtime_layout_policy']} "
            f"runtime_layout={param_group_summary['runtime_layout_mode']} "
            f"rank={param_group_summary['rank']}/{param_group_summary['world_size']} "
            f"params={param_group_summary['num_params']} "
            f"total={param_group_summary['total_numel']} "
            f"local={param_group_summary['local_numel']} "
            f"rank_mem={_format_rank_values(param_group_summary['rank_memory_bytes'])} "
            f"rank_comm={_format_rank_values(param_group_summary['rank_comm_bytes'])} "
            f"gather={param_group_summary['communication_summary']['effective_param_gather_backend']} "
            f"custom={param_group_summary['communication_summary']['resolved_custom_allgatherv_impl']} "
            f"reduce={param_group_summary['communication_summary']['effective_grad_reduce_backend']} "
            f"custom_reduce={param_group_summary['communication_summary']['resolved_custom_reduce_scatterv_impl']} "
            f"chunk_fast={param_group_summary['communication_summary']['rank_chunk_fast_path']} "
            f"pad_waste={param_group_summary['communication_summary']['padding_waste_ratio']:.3f} "
            f"imbalance={param_group_summary['communication_summary']['owner_imbalance_ratio']:.3f} "
            f"workspace={param_group_summary['communication_summary']['workspace_preferred_kind']} "
            f"ws_numel={param_group_summary['communication_summary']['workspace_preferred_numel']} "
            f"ws_alloc={param_group_summary['communication_summary']['workspace_allocate_count']} "
            f"ws_reuse={param_group_summary['communication_summary']['workspace_reuse_count']} "
            f"state={param_group_summary['lifecycle_state']} "
            f"reshard_after_forward={param_group_summary['reshard_after_forward']} "
            f"forward_prefetch={param_group_summary['forward_prefetch']} "
            f"backward_prefetch={param_group_summary['backward_prefetch']} "
            f"finalize_after_backward={param_group_summary['finalize_after_backward']}"
        )
    return "\n".join(lines)


def format_runtime_events(summary: dict[str, Any]) -> str:
    lines = [
        f"MatrixFSDP runtime events: {summary['num_events']} events across "
        f"{summary['num_param_groups']} param groups"
    ]
    for scheduler_summary in summary.get("schedulers", ()):
        lines.append(
            "  "
            f"scheduler param_groups={scheduler_summary['param_group_indices']} "
            f"policy={scheduler_summary['prefetch_policy']} "
            f"forward_budget={_format_budget(scheduler_summary['selected_forward_prefetch_budget'])} "
            f"backward_budget={_format_budget(scheduler_summary['selected_backward_prefetch_budget'])} "
            f"blocked={scheduler_summary['budget_blocked_prefetch_count']} "
            f"forward_blocked={scheduler_summary['forward_prefetch_budget_blocked']} "
            f"backward_blocked={scheduler_summary['backward_prefetch_budget_blocked']} "
            f"backward_deferred={scheduler_summary['backward_prefetch_memory_deferred']} "
            f"max_full_buffers={scheduler_summary['max_active_full_param_buffers']} "
            f"max_full_numel={scheduler_summary['max_active_full_param_numel']} "
            f"max_full_bytes={scheduler_summary['max_active_full_param_bytes']} "
            f"max_grad_bucket_bytes={scheduler_summary['max_grad_bucket_bytes']} "
            f"max_reduce_scatter_input_bytes={scheduler_summary['max_reduce_scatter_input_bytes']} "
            f"max_local_grad_shard_bytes={scheduler_summary['max_local_grad_shard_bytes']} "
            f"max_pending_rs={scheduler_summary['max_pending_backward_reduce_count']}"
        )
        for profile_result in scheduler_summary["profile_results"]:
            lines.append(
                "    "
                f"profile budget={_format_budget(profile_result['budget'])} "
                f"avg_step_ms={profile_result['avg_step_ms']:.3f} "
                f"peak_mem_mb={profile_result['peak_memory_mb']:.1f}"
            )
    timed_event_stats = [stat for stat in summary.get("event_stats", ()) if stat["duration_count"] > 0]
    communication_event_stats = [stat for stat in summary.get("communication_event_stats", ()) if stat["duration_count"] > 0]
    if communication_event_stats:
        lines.append("  communication_event_stats:")
        for stat in communication_event_stats:
            lines.append(
                "    "
                f"{stat['category']} "
                f"count={stat['count']} "
                f"timed={stat['duration_count']} "
                f"sum_ms={stat['duration_sum_ms']:.3f} "
                f"avg_ms={stat['duration_avg_ms']:.3f} "
                f"max_ms={stat['duration_max_ms']:.3f}"
            )
    if timed_event_stats:
        lines.append("  event_stats top_by_sum_ms:")
        for stat in sorted(timed_event_stats, key=lambda item: (-item["duration_sum_ms"], item["name"]))[:12]:
            lines.append(
                "    "
                f"{stat['name']} "
                f"count={stat['count']} "
                f"timed={stat['duration_count']} "
                f"sum_ms={stat['duration_sum_ms']:.3f} "
                f"avg_ms={stat['duration_avg_ms']:.3f} "
                f"max_ms={stat['duration_max_ms']:.3f}"
            )
    for event_summary in summary["events"]:
        duration = event_summary["duration_ms"]
        duration_text = "" if duration is None else f" duration_ms={duration:.3f}"
        memory_text = _format_event_memory(event_summary)
        lines.append(
            "  "
            f"#{event_summary['sequence']} "
            f"param_group={event_summary['param_group_index']} "
            f"{event_summary['runtime_param_group_id']} "
            f"rank={event_summary['rank']} "
            f"state={event_summary['lifecycle_state']} "
            f"name={event_summary['name']}"
            f"{duration_text}"
            f"{memory_text}"
        )
    return "\n".join(lines)


def _summarize_event_stats(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats_by_name: dict[str, dict[str, Any]] = {}
    for event in events:
        name = event["name"]
        stats = stats_by_name.setdefault(
            name,
            {
                "name": name,
                "count": 0,
                "duration_count": 0,
                "duration_sum_ms": 0.0,
                "duration_avg_ms": 0.0,
                "duration_max_ms": 0.0,
            },
        )
        stats["count"] += 1
        duration = event["duration_ms"]
        if duration is None:
            continue
        stats["duration_count"] += 1
        stats["duration_sum_ms"] += float(duration)
        stats["duration_max_ms"] = max(stats["duration_max_ms"], float(duration))
    for stats in stats_by_name.values():
        if stats["duration_count"]:
            stats["duration_avg_ms"] = stats["duration_sum_ms"] / stats["duration_count"]
    return sorted(stats_by_name.values(), key=lambda stats: stats["name"])


def _summarize_communication_event_stats(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats_by_category: dict[str, dict[str, Any]] = {}
    for event in events:
        category = _communication_event_category(event["name"])
        if category is None:
            continue
        stats = stats_by_category.setdefault(
            category,
            {
                "category": category,
                "count": 0,
                "duration_count": 0,
                "duration_sum_ms": 0.0,
                "duration_avg_ms": 0.0,
                "duration_max_ms": 0.0,
            },
        )
        stats["count"] += 1
        duration = event["duration_ms"]
        if duration is None:
            continue
        stats["duration_count"] += 1
        stats["duration_sum_ms"] += float(duration)
        stats["duration_max_ms"] = max(stats["duration_max_ms"], float(duration))
    for stats in stats_by_category.values():
        if stats["duration_count"]:
            stats["duration_avg_ms"] = stats["duration_sum_ms"] / stats["duration_count"]
    order = {
        "all_gather_enqueue": 0,
        "all_gather_wait": 1,
        "grad_copy_in": 2,
        "reduce_scatter_enqueue": 3,
        "reduce_scatter_wait": 4,
        "grad_bucket_collect": 5,
    }
    return sorted(stats_by_category.values(), key=lambda stats: order.get(stats["category"], 99))


def _communication_event_category(name: str) -> str | None:
    if name == "enqueue_all_gather_full_params":
        return "all_gather_enqueue"
    if name.startswith("wait_unshard:"):
        return "all_gather_wait"
    if name in ("copy_in_grad_bucket", "copy_in_grad_bucket_for_accumulation"):
        return "grad_copy_in"
    if name == "enqueue_reduce_scatter_grad_bucket":
        return "reduce_scatter_enqueue"
    if name == "wait_reduce_grad_bucket":
        return "reduce_scatter_wait"
    if name in ("collect_grad_bucket", "prepare_grad_bucket_zero_copy"):
        return "grad_bucket_collect"
    return None


def _summarize_schedulers(
    param_groups: list[MatrixFSDPParamGroup],
    param_group_index_by_id: dict[int, int],
) -> list[dict[str, Any]]:
    scheduler_entries: dict[int, dict[str, Any]] = {}
    for param_group in param_groups:
        scheduler = getattr(param_group, "_scheduler", None)
        if scheduler is None:
            continue
        entry = scheduler_entries.setdefault(
            id(scheduler),
            {
                "scheduler": scheduler,
                "unit_indices": [],
            },
        )
        entry["unit_indices"].append(param_group_index_by_id[id(param_group)])

    summaries = []
    for entry in scheduler_entries.values():
        scheduler = entry["scheduler"]
        unit_indices = tuple(entry["unit_indices"])
        summaries.append(
            {
                "param_group_indices": unit_indices,
                "unit_indices": unit_indices,
                "prefetch_policy": scheduler.prefetch_policy,
                "requested_max_unsharded_prefetch_units": scheduler.requested_max_unsharded_prefetch_units,
                "requested_max_forward_prefetch_units": scheduler.requested_max_forward_prefetch_units,
                "requested_max_backward_prefetch_units": scheduler.requested_max_backward_prefetch_units,
                "max_unsharded_prefetch_units": scheduler.max_unsharded_prefetch_units,
                "max_forward_prefetch_units": scheduler.max_forward_prefetch_units,
                "max_backward_prefetch_units": scheduler.max_backward_prefetch_units,
                "max_active_full_param_buffers_limit": scheduler.max_active_full_param_buffers_limit,
                "max_active_full_param_numel_limit": scheduler.max_active_full_param_numel_limit,
                "max_active_full_param_bytes_limit": scheduler.max_active_full_param_bytes_limit,
                "max_pending_backward_reduces": scheduler.max_pending_backward_reduces,
                "trim_cuda_cache": scheduler.trim_cuda_cache,
                "cuda_cache_trim_threshold_bytes": scheduler.cuda_cache_trim_threshold_bytes,
                "cuda_cache_trim_count": scheduler.cuda_cache_trim_count,
                "pending_backward_reduce_count": scheduler.pending_backward_reduce_count,
                "selected_prefetch_budget": scheduler.selected_prefetch_budget,
                "selected_forward_prefetch_budget": scheduler.selected_forward_prefetch_budget,
                "selected_backward_prefetch_budget": scheduler.selected_backward_prefetch_budget,
                "forward_prefetch_issued": scheduler.forward_prefetch_issued,
                "backward_prefetch_issued": scheduler.backward_prefetch_issued,
                "forward_prefetch_budget_blocked": scheduler.forward_prefetch_budget_blocked,
                "backward_prefetch_budget_blocked": scheduler.backward_prefetch_budget_blocked,
                "backward_prefetch_memory_deferred": scheduler.backward_prefetch_memory_deferred,
                "budget_blocked_prefetch_count": scheduler.budget_blocked_prefetch_count,
                "backward_reduce_waits": scheduler.backward_reduce_waits,
                "max_active_full_param_buffers": scheduler.max_active_full_param_buffers,
                "max_active_full_param_numel": scheduler.max_active_full_param_numel,
                "max_active_full_param_bytes": scheduler.max_active_full_param_bytes,
                "max_grad_bucket_bytes": scheduler.max_grad_bucket_bytes,
                "max_reduce_scatter_input_bytes": scheduler.max_reduce_scatter_input_bytes,
                "max_local_grad_shard_bytes": scheduler.max_local_grad_shard_bytes,
                "max_pending_backward_reduce_count": scheduler.max_pending_backward_reduce_count,
                "full_param_buffer_snapshots": tuple(dict(snapshot) for snapshot in scheduler.full_param_buffer_snapshots),
                "runtime_memory_snapshots": tuple(dict(snapshot) for snapshot in scheduler.runtime_memory_snapshots),
                "profile_results": tuple(
                    {
                        "budget": result.budget,
                        "avg_step_ms": result.avg_step_ms,
                        "peak_memory_mb": result.peak_memory_mb,
                    }
                    for result in scheduler.profile_results
                ),
            }
        )
    summaries.sort(key=lambda summary: summary["unit_indices"])
    return summaries


def _format_budget(budget: int | None) -> str:
    return "none" if budget is None else str(budget)


def _format_event_memory(event_summary: dict[str, Any]) -> str:
    fields = (
        ("active_full", event_summary["active_full_param_buffers"]),
        ("full_bytes", event_summary["active_full_param_bytes"]),
        ("unit_full_bytes", event_summary["unit_full_param_bytes"]),
        ("grad_bucket_bytes", event_summary["unit_grad_bucket_bytes"]),
        ("reduce_scatter_input_bytes", event_summary["unit_reduce_scatter_input_bytes"]),
        ("local_grad_bytes", event_summary["unit_local_grad_shard_bytes"]),
        ("pending_rs", event_summary["pending_backward_reduces"]),
    )
    nonzero_fields = [f"{name}={value}" for name, value in fields if value]
    if event_summary.get("param_data_alias_full_buffer"):
        nonzero_fields.append("param_alias=full")
    elif event_summary.get("param_data_alias_local_shard"):
        nonzero_fields.append("param_alias=local")
    if not nonzero_fields:
        return ""
    return " " + " ".join(nonzero_fields)


def _format_rank_values(values: tuple[int, ...]) -> str:
    return "-".join(str(value) for value in values) if values else "-"


def _sum_rank_resource(param_group_summaries: list[dict[str, Any]], key: str) -> tuple[int, ...]:
    max_world_size = max((len(param_group_summary[key]) for param_group_summary in param_group_summaries), default=0)
    totals = [0 for _ in range(max_world_size)]
    for param_group_summary in param_group_summaries:
        for rank, value in enumerate(param_group_summary[key]):
            totals[rank] += value
    return tuple(totals)


def _aggregate_communication_summaries(param_group_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    communication_summaries = [
        param_group_summary["communication_summary"]
        for param_group_summary in param_group_summaries
        if param_group_summary.get("communication_summary") is not None
    ]
    gather_backend_counts = Counter(
        str(summary.get("effective_param_gather_backend")) for summary in communication_summaries
    )
    custom_impl_counts = Counter(
        str(summary.get("resolved_custom_allgatherv_impl"))
        for summary in communication_summaries
        if summary.get("resolved_custom_allgatherv_impl") is not None
    )
    grad_reduce_counts = Counter(
        str(summary.get("effective_grad_reduce_backend")) for summary in communication_summaries
    )
    custom_reduce_counts = Counter(
        str(summary.get("resolved_custom_reduce_scatterv_impl"))
        for summary in communication_summaries
        if summary.get("resolved_custom_reduce_scatterv_impl") is not None
    )
    workspace_kind_counts = Counter(
        str(summary.get("workspace_preferred_kind"))
        for summary in communication_summaries
        if summary.get("workspace_preferred_kind") is not None
    )
    return {
        "num_param_groups": len(communication_summaries),
        "gather_backend_counts": dict(sorted(gather_backend_counts.items())),
        "resolved_custom_allgatherv_counts": dict(sorted(custom_impl_counts.items())),
        "grad_reduce_backend_counts": dict(sorted(grad_reduce_counts.items())),
        "resolved_custom_reduce_scatterv_counts": dict(sorted(custom_reduce_counts.items())),
        "workspace_preferred_kind_counts": dict(sorted(workspace_kind_counts.items())),
        "rank_chunk_fast_path_count": sum(
            1 for summary in communication_summaries if summary.get("rank_chunk_fast_path")
        ),
        "packed_full_order_count": sum(
            1 for summary in communication_summaries if summary.get("packed_rank_shards_are_full_tensor_order")
        ),
        "max_segment_count": max((int(summary.get("segment_count", 0)) for summary in communication_summaries), default=0),
        "max_segments_per_rank": max(
            (int(summary.get("max_segments_per_rank", 0)) for summary in communication_summaries),
            default=0,
        ),
        "max_padding_waste_ratio": max(
            (float(summary.get("padding_waste_ratio", 0.0)) for summary in communication_summaries),
            default=0.0,
        ),
        "max_owner_imbalance_ratio": max(
            (float(summary.get("owner_imbalance_ratio", 0.0)) for summary in communication_summaries),
            default=0.0,
        ),
        "max_shard_size": max((int(summary.get("max_shard_size", 0)) for summary in communication_summaries), default=0),
        "min_shard_size": min((int(summary.get("min_shard_size", 0)) for summary in communication_summaries), default=0),
        "max_workspace_preferred_numel": max(
            (int(summary.get("workspace_preferred_numel", 0)) for summary in communication_summaries),
            default=0,
        ),
        "max_workspace_padded_rank_chunks_numel": max(
            (int(summary.get("workspace_padded_rank_chunks_numel", 0)) for summary in communication_summaries),
            default=0,
        ),
        "max_workspace_padding_waste_ratio": max(
            (float(summary.get("workspace_padding_waste_ratio", 0.0)) for summary in communication_summaries),
            default=0.0,
        ),
        "workspace_total_acquire_count": sum(
            int(summary.get("workspace_acquire_count", 0)) for summary in communication_summaries
        ),
        "workspace_total_reuse_count": sum(
            int(summary.get("workspace_reuse_count", 0)) for summary in communication_summaries
        ),
        "workspace_total_allocate_count": sum(
            int(summary.get("workspace_allocate_count", 0)) for summary in communication_summaries
        ),
        "max_workspace_allocated_numel": max(
            (int(summary.get("workspace_allocated_numel", 0)) for summary in communication_summaries),
            default=0,
        ),
        "max_workspace_in_use_numel": max(
            (int(summary.get("workspace_in_use_numel", 0)) for summary in communication_summaries),
            default=0,
        ),
    }


def _summarize_param_group(param_group: MatrixFSDPParamGroup) -> dict[str, Any]:
    param_fqns = [mp.fqn for mp in param_group.managed_params]
    total_numel = sum(mp.numel for mp in param_group.managed_params)
    local_numel = param_group.flat_buffer.local_numel if param_group.flat_buffer is not None else 0
    compatibility = param_group.flat_buffer.placement_compatibility if param_group.flat_buffer is not None else None
    placement = param_group.flat_buffer.placement if param_group.flat_buffer is not None else None
    param_shard_state = (
        param_group.flat_buffer.param_state.as_metadata()
        if param_group.flat_buffer is not None
        else None
    )
    grad_shard_state = (
        param_group.flat_buffer.grad_state.as_metadata()
        if param_group.flat_buffer is not None and param_group.flat_buffer.grad_state is not None
        else None
    )
    planner_result = param_group.planner_result
    resource_metadata = (
        planner_result.resource_estimate.as_metadata()
        if planner_result is not None and planner_result.resource_estimate is not None
        else {}
    )
    planner_metadata = planner_result.as_metadata() if planner_result is not None else None
    planner_summary = planner_result.summary() if planner_result is not None else None
    if param_group.planner_layout_contract is not None:
        planner_layout_contract = param_group.planner_layout_contract.as_metadata()
    elif planner_summary is not None:
        planner_layout_contract = planner_summary["layout"]
    else:
        planner_layout_contract = None
    runtime_layout_contract = (
        param_group.runtime_layout_contract.as_metadata()
        if param_group.runtime_layout_contract is not None
        else None
    )
    planner_report = planner_summary["report"] if planner_summary is not None else None
    communication_summary = (
        param_group.flat_buffer.communication_summary()
        if param_group.flat_buffer is not None
        else {
            "param_gather_strategy": None,
            "matrix_collective_backend": None,
            "effective_param_gather_backend": None,
            "owner_segment_collectives": False,
            "owner_segment_backend": None,
            "custom_allgatherv_policy": None,
            "resolved_custom_allgatherv_impl": None,
            "effective_grad_reduce_backend": None,
            "resolved_custom_reduce_scatterv_impl": None,
            "rank_chunk_fast_path": False,
            "packed_rank_shards_are_full_tensor_order": False,
            "segment_count": 0,
            "max_segments_per_rank": 0,
            "shard_sizes": (),
            "max_shard_size": 0,
            "min_shard_size": 0,
            "padding_waste_numel": 0,
            "padding_waste_ratio": 0.0,
            "owner_imbalance_ratio": 0.0,
            "workspace_full_param_numel": 0,
            "workspace_compact_rank_chunks_numel": 0,
            "workspace_padded_rank_chunks_numel": 0,
            "workspace_padding_waste_numel": 0,
            "workspace_padding_waste_ratio": 0.0,
            "workspace_preferred_kind": None,
            "workspace_preferred_numel": 0,
            "workspace_rank_chunk_fast_path": False,
            "workspace_packed_full_order": False,
            "workspace_owner_segment_collectives": False,
            "workspace_native_group_broadcast_capable": False,
            "workspace_native_sendrecv_chunk_capable": False,
            "workspace_padded_all_gather_capable": False,
            "workspace_compact_owner_reduce_scatter_capable": False,
            "workspace_acquire_count": 0,
            "workspace_reuse_count": 0,
            "workspace_allocate_count": 0,
            "workspace_allocated_tensors": 0,
            "workspace_in_use_tensors": 0,
            "workspace_allocated_numel": 0,
            "workspace_in_use_numel": 0,
        }
    )
    return {
        "runtime_param_group_id": param_group.runtime_metadata.runtime_param_group_id,
        "runtime_unit_id": param_group.runtime_metadata.runtime_unit_id,
        "planner_group_id": param_group.runtime_metadata.planner_group_id,
        "comm_buffer_id": param_group.runtime_metadata.comm_buffer_id,
        "rank": param_group.rank,
        "world_size": param_group.world_size,
        "replicate_world_size": param_group.replicate_world_size,
        "dp_shard_mesh_dim": param_group.dp_shard_mesh_dim,
        "dp_replicate_mesh_dim": param_group.dp_replicate_mesh_dim,
        "device_mesh": param_group.device_mesh_metadata,
        "mixed_precision": mixed_precision_policy_metadata(param_group.mp_policy),
        "offload_policy": offload_policy_metadata(param_group.offload_policy),
        "num_params": len(param_fqns),
        "total_numel": total_numel,
        "local_numel": local_numel,
        "matrix_shard_placement": placement,
        "param_shard_state": param_shard_state,
        "grad_shard_state": grad_shard_state,
        "matrix_shard_compatible": compatibility.compatible if compatibility is not None else False,
        "matrix_shard_compatibility": compatibility,
        "planner_name": planner_result.planner_name if planner_result is not None else None,
        "planner_policy": planner_result.policy if planner_result is not None else None,
        "planner_runtime_mode": planner_result.runtime_mode if planner_result is not None else None,
        "planner_runtime_compatible": planner_result.runtime_compatible if planner_result is not None else False,
        "planner_runtime_requires_flat_reorder": (
            planner_result.runtime_requires_flat_reorder if planner_result is not None else False
        ),
        "runtime_layout_policy": param_group.runtime_layout_policy,
        "runtime_layout_mode": (
            param_group.runtime_layout_compatibility.mode if param_group.runtime_layout_compatibility is not None else None
        ),
        "runtime_layout_reason": (
            param_group.runtime_layout_compatibility.reason if param_group.runtime_layout_compatibility is not None else None
        ),
        "runtime_layout_requires_flat_reorder": (
            param_group.runtime_layout_compatibility.requires_flat_reorder
            if param_group.runtime_layout_compatibility is not None
            else False
        ),
        "planner_cost": planner_result.cost if planner_result is not None else None,
        "planner_cost_terms": tuple(planner_result.cost_breakdown.terms.items()) if planner_result is not None else (),
        "planner_metadata": planner_metadata,
        "planner_summary": planner_summary,
        "planner_layout_contract": planner_layout_contract,
        "runtime_layout_contract": runtime_layout_contract,
        "planner_report": planner_report,
        "planner_resource_estimate": resource_metadata,
        "communication_summary": communication_summary,
        "planner_rank_units": (
            tuple(planner_layout_contract["rank_units"]) if planner_layout_contract is not None else ()
        ),
        "planner_params_by_rank": planner_report["params_by_rank"] if planner_report is not None else (),
        "rank_memory_bytes": tuple(resource_metadata.get("rank_memory_bytes", ())),
        "rank_comm_bytes": tuple(resource_metadata.get("rank_comm_bytes", ())),
        "rank_muon_param_bytes": tuple(resource_metadata.get("rank_muon_param_bytes", ())),
        "rank_adamw_param_bytes": tuple(resource_metadata.get("rank_adamw_param_bytes", ())),
        "rank_optimizer_bytes": tuple(resource_metadata.get("rank_optimizer_bytes", ())),
        "param_fqns": param_fqns,
        "lifecycle_state": param_group.lifecycle_state.value,
        "reshard_after_forward": param_group.reshard_after_forward_enabled,
        "forward_prefetch": param_group.forward_prefetch_enabled,
        "backward_prefetch": param_group.backward_prefetch_enabled,
        "finalize_after_backward": param_group.finalize_after_backward_enabled,
        "shrink_full_param_storage_after_backward": param_group.shrink_full_param_storage_after_backward,
    }
