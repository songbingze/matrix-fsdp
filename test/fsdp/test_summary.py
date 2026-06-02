import unittest

import torch
from torch import nn

from matrix_fsdp import (
    PrefetchProfileResult,
    MatrixFSDPOptimizer,
    MatrixFSDPScheduler,
    format_runtime_events,
    format_param_group_summary,
    module_type_policy,
    matrix_fully_shard,
    summarize_runtime_events,
    summarize_param_groups,
)


class SummaryTest(unittest.TestCase):
    def test_summarizes_single_unit_model(self):
        model = matrix_fully_shard(nn.Linear(4, 2), reshard_after_forward=True, finalize_after_backward=True)
        unit = model._matrix_fsdp_param_group

        summary = summarize_param_groups(model)
        unit_summary = summary["units"][0]

        self.assertEqual(summary["num_param_groups"], 1)
        self.assertEqual(summary["num_units"], 1)
        self.assertIs(summary["param_groups"], summary["units"])
        self.assertEqual(summary["total_numel"], 10)
        self.assertEqual(summary["local_numel"], unit.flat_buffer.local_numel)
        self.assertEqual(summary["rank_total_memory_bytes"], unit_summary["rank_memory_bytes"])
        self.assertEqual(summary["rank_total_comm_bytes"], unit_summary["rank_comm_bytes"])
        self.assertEqual(unit_summary["runtime_param_group_id"], unit.runtime_metadata.runtime_param_group_id)
        self.assertEqual(unit_summary["runtime_unit_id"], unit.runtime_metadata.runtime_unit_id)
        self.assertEqual(unit_summary["planner_group_id"], unit.runtime_metadata.planner_group_id)
        self.assertEqual(unit_summary["comm_buffer_id"], unit.runtime_metadata.comm_buffer_id)
        self.assertEqual(unit_summary["num_params"], 2)
        self.assertEqual(unit_summary["param_fqns"], ["weight", "bias"])
        self.assertTrue(unit_summary["matrix_shard_compatible"])
        self.assertEqual(unit_summary["matrix_shard_placement"].local_units, (1,))
        self.assertEqual(unit_summary["param_shard_state"]["name"], "param")
        self.assertEqual(unit_summary["param_shard_state"]["local_numel"], unit.flat_buffer.local_numel)
        self.assertIsNone(unit_summary["grad_shard_state"])
        self.assertEqual(unit_summary["planner_name"], "contiguous_even_plan")
        self.assertIsNone(unit_summary["planner_policy"])
        self.assertEqual(unit_summary["planner_runtime_mode"], "matrix_shard")
        self.assertTrue(unit_summary["planner_runtime_compatible"])
        self.assertEqual(unit_summary["runtime_layout_policy"], "auto")
        self.assertEqual(unit_summary["runtime_layout_mode"], "matrix_shard")
        self.assertFalse(unit_summary["runtime_layout_requires_flat_reorder"])
        self.assertEqual(unit_summary["planner_rank_units"], unit.group_layout.shard_sizes)
        self.assertEqual(summary["communication_summary"]["gather_backend_counts"], {"single_rank_copy": 1})
        self.assertEqual(summary["communication_summary"]["rank_chunk_fast_path_count"], 1)
        self.assertEqual(summary["communication_summary"]["workspace_preferred_kind_counts"], {"padded_rank_chunks": 1})
        self.assertEqual(summary["communication_summary"]["max_workspace_preferred_numel"], 10)
        self.assertEqual(unit_summary["communication_summary"]["effective_param_gather_backend"], "single_rank_copy")
        self.assertTrue(unit_summary["communication_summary"]["rank_chunk_fast_path"])
        self.assertEqual(unit_summary["communication_summary"]["padding_waste_ratio"], 0.0)
        self.assertEqual(unit_summary["communication_summary"]["workspace_preferred_kind"], "padded_rank_chunks")
        self.assertEqual(unit_summary["planner_metadata"], unit.planner_result.as_metadata())
        self.assertEqual(unit_summary["planner_summary"], unit.planner_result.summary())
        self.assertEqual(
            unit_summary["planner_layout_contract"],
            unit.planner_result.layout_contract().as_metadata(),
        )
        self.assertEqual(unit_summary["runtime_layout_contract"], unit.runtime_layout_contract.as_metadata())
        self.assertEqual(unit_summary["planner_report"], unit.planner_result.report.as_metadata())
        self.assertEqual(
            unit_summary["planner_resource_estimate"],
            unit.planner_result.resource_estimate.as_metadata(),
        )
        self.assertEqual(unit_summary["rank_memory_bytes"], (80,))
        self.assertEqual(unit_summary["rank_comm_bytes"], (0,))
        self.assertEqual(unit_summary["lifecycle_state"], "sharded")
        self.assertTrue(unit_summary["reshard_after_forward"])
        self.assertTrue(unit_summary["forward_prefetch"])
        self.assertTrue(unit_summary["backward_prefetch"])
        self.assertTrue(unit_summary["finalize_after_backward"])

    def test_summarizes_multi_unit_model(self):
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)),
            wrap_policy=module_type_policy(nn.Linear),
            reshard_after_forward=True,
        )

        summary = summarize_param_groups(model)

        self.assertEqual(summary["num_param_groups"], 2)
        self.assertEqual(summary["num_units"], 2)
        self.assertEqual(summary["total_numel"], 58)
        self.assertEqual(summary["local_numel"], 58)
        self.assertEqual(summary["units"][0]["param_fqns"], ["weight", "bias"])
        self.assertEqual(summary["units"][1]["param_fqns"], ["weight", "bias"])
        self.assertNotEqual(
            summary["param_groups"][0]["runtime_param_group_id"],
            summary["param_groups"][1]["runtime_param_group_id"],
        )

    def test_summarizes_unit_iterable(self):
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)),
            wrap_policy=module_type_policy(nn.Linear),
        )
        units = [model[0]._matrix_fsdp_param_group, model[2]._matrix_fsdp_param_group]

        summary = summarize_param_groups(units)

        self.assertEqual(summary["num_param_groups"], 2)
        self.assertEqual(summary["num_units"], 2)
        self.assertEqual([unit_summary["total_numel"] for unit_summary in summary["units"]], [40, 18])

    def test_format_param_group_summary(self):
        model = matrix_fully_shard(nn.Linear(4, 2), reshard_after_forward=True, finalize_after_backward=True)

        text = format_param_group_summary(summarize_param_groups(model))

        self.assertIn("MatrixFSDP param groups: 1", text)
        self.assertIn("total_numel=10", text)
        self.assertIn("params=2", text)
        self.assertIn("planner_name=contiguous_even_plan", text)
        self.assertIn("runtime=matrix_shard", text)
        self.assertIn("runtime_layout_policy=auto", text)
        self.assertIn("runtime_layout=matrix_shard", text)
        self.assertIn("rank_mem=80", text)
        self.assertIn("rank_comm=0", text)
        self.assertIn("gather=single_rank_copy", text)
        self.assertIn("chunk_fast=True", text)
        self.assertIn("pad_waste=0.000", text)
        self.assertIn("state=sharded", text)
        self.assertIn("reshard_after_forward=True", text)
        self.assertIn("forward_prefetch=True", text)
        self.assertIn("backward_prefetch=True", text)
        self.assertIn("finalize_after_backward=True", text)

    def test_summarizes_runtime_events_across_units(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)),
            wrap_policy=module_type_policy(nn.Linear),
            reshard_after_forward=True,
            finalize_after_backward=True,
        )
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        optim.step()

        summary = summarize_runtime_events(model)
        events = summary["events"]
        event_names = [event["name"] for event in events]
        sequences = [event["sequence"] for event in events]
        unit1_pre_backward = next(
            event for event in events if event["param_group_index"] == 1 and event["name"] == "pre_backward_unshard"
        )
        unit0_pre_backward = next(
            event for event in events if event["param_group_index"] == 0 and event["name"] == "pre_backward_unshard"
        )

        self.assertEqual(summary["num_param_groups"], 2)
        self.assertEqual(summary["num_units"], 2)
        self.assertEqual(sequences, sorted(sequences))
        self.assertIn("finalize_backward", event_names)
        self.assertTrue(any(stat["name"] == "finalize_backward" for stat in summary["event_stats"]))
        self.assertTrue(any(stat["duration_count"] > 0 for stat in summary["event_stats"]))
        self.assertTrue(any(stat["category"] == "all_gather_enqueue" for stat in summary["communication_event_stats"]))
        self.assertTrue(any(stat["category"] == "all_gather_wait" for stat in summary["communication_event_stats"]))
        self.assertTrue(all("timestamp_ns" in event for event in events))
        self.assertTrue(all("runtime_param_group_id" in event for event in events))
        self.assertTrue(all("active_full_param_bytes" in event for event in events))
        self.assertTrue(all("unit_reduce_scatter_input_bytes" in event for event in events))
        self.assertTrue(any(event["active_full_param_buffers"] > 0 for event in events))
        self.assertTrue(any(event["duration_ms"] is not None for event in events if event["name"].startswith("wait_unshard")))
        self.assertLess(unit1_pre_backward["sequence"], unit0_pre_backward["sequence"])

    def test_runtime_events_include_event_level_memory_trace(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)),
            wrap_policy=module_type_policy(nn.Linear),
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=True,
        )
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        optim.step()

        summary = summarize_runtime_events(model)
        events = summary["events"]
        scheduler_summary = summary["schedulers"][0]
        grad_bucket_events = [event for event in events if event["name"].startswith("prepare_grad_bucket")]
        pending_events = [event for event in events if event["name"] == "pending_backward_reduce"]

        self.assertTrue(any(event["unit_grad_bucket_bytes"] > 0 for event in grad_bucket_events))
        self.assertTrue(any(event["pending_backward_reduces"] > 0 for event in pending_events))
        self.assertGreater(scheduler_summary["max_grad_bucket_bytes"], 0)
        self.assertGreater(scheduler_summary["max_local_grad_shard_bytes"], 0)
        self.assertGreater(scheduler_summary["max_pending_backward_reduce_count"], 0)
        self.assertTrue(scheduler_summary["runtime_memory_snapshots"])
        self.assertTrue(any(event["param_data_alias_full_buffer"] for event in events))
        self.assertTrue(any(event["param_data_alias_local_shard"] for event in events))
        self.assertTrue(
            any(snapshot["param_data_alias_full_buffer"] for snapshot in scheduler_summary["runtime_memory_snapshots"])
        )
        self.assertTrue(
            any(snapshot["param_data_alias_local_shard"] for snapshot in scheduler_summary["runtime_memory_snapshots"])
        )

        text = format_runtime_events(summary)

        self.assertIn("grad_bucket_bytes=", text)
        self.assertIn("local_grad_bytes=", text)
        self.assertIn("pending_rs=", text)
        self.assertIn("param_alias=full", text)
        self.assertIn("param_alias=local", text)
        self.assertIn("max_grad_bucket_bytes=", text)
        self.assertIn("communication_event_stats:", text)
        self.assertIn("all_gather_enqueue", text)
        self.assertIn("reduce_scatter_enqueue", text)

    def test_runtime_events_track_copy_in_reduce_scatter_input_memory(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)),
            wrap_policy=module_type_policy(nn.Linear),
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
        )
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        optim.step()

        summary = summarize_runtime_events(model)
        copy_in_events = [event for event in summary["events"] if event["name"] == "copy_in_grad_bucket"]
        scheduler_summary = summary["schedulers"][0]

        self.assertTrue(copy_in_events)
        self.assertTrue(any(event["unit_reduce_scatter_input_bytes"] > 0 for event in copy_in_events))
        self.assertTrue(all(event["unit_full_param_bytes"] == 0 for event in copy_in_events))
        self.assertGreater(scheduler_summary["max_reduce_scatter_input_bytes"], 0)

        text = format_runtime_events(summary)

        self.assertIn("reduce_scatter_input_bytes=", text)
        self.assertIn("max_reduce_scatter_input_bytes=", text)

    def test_summarizes_scheduler_profile_and_budget_blocks(self):
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8), reshard_after_forward=True, forward_prefetch=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2), reshard_after_forward=True, forward_prefetch=True),
        )
        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_unsharded_prefetch_units=0,
            prefetch_policy="profile_guided",
        )
        optim.scheduler.set_profile_results(
            (
                PrefetchProfileResult(budget=0, avg_step_ms=3.0, peak_memory_mb=10.0),
                PrefetchProfileResult(budget=1, avg_step_ms=2.0, peak_memory_mb=12.0),
            )
        )

        model(torch.randn(3, 4)).sum()
        summary = summarize_runtime_events(model)
        scheduler_summary = summary["schedulers"][0]

        self.assertEqual(summary["num_schedulers"], 1)
        self.assertEqual(scheduler_summary["prefetch_policy"], "profile_guided")
        self.assertEqual(scheduler_summary["selected_prefetch_budget"], 0)
        self.assertEqual(scheduler_summary["selected_forward_prefetch_budget"], 0)
        self.assertEqual(scheduler_summary["selected_backward_prefetch_budget"], 1)
        self.assertEqual(scheduler_summary["budget_blocked_prefetch_count"], 1)
        self.assertEqual(scheduler_summary["forward_prefetch_budget_blocked"], 1)
        self.assertIn("max_active_full_param_buffers", scheduler_summary)
        self.assertIn("max_active_full_param_bytes", scheduler_summary)
        self.assertIn("max_active_full_param_numel_limit", scheduler_summary)
        self.assertIn("max_active_full_param_bytes_limit", scheduler_summary)
        self.assertIn("full_param_buffer_snapshots", scheduler_summary)
        self.assertEqual(scheduler_summary["profile_results"][1]["budget"], 1)

    def test_scheduler_summary_can_be_built_from_manual_scheduler(self):
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8), reshard_after_forward=True, forward_prefetch=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2), reshard_after_forward=True, forward_prefetch=True),
        )
        units = [model[0]._matrix_fsdp_param_group, model[2]._matrix_fsdp_param_group]
        scheduler = MatrixFSDPScheduler(units, prefetch_policy="adaptive")
        scheduler.set_prefetch_budget(1)

        summary = summarize_runtime_events(units)

        self.assertEqual(summary["schedulers"][0]["param_group_indices"], (0, 1))
        self.assertEqual(summary["schedulers"][0]["unit_indices"], (0, 1))
        self.assertEqual(summary["schedulers"][0]["selected_prefetch_budget"], 1)
        self.assertEqual(summary["schedulers"][0]["selected_forward_prefetch_budget"], 1)
        self.assertEqual(summary["schedulers"][0]["selected_backward_prefetch_budget"], 1)

    def test_format_runtime_events(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2), reshard_after_forward=True)

        model(torch.randn(3, 4)).sum().backward()
        text = format_runtime_events(summarize_runtime_events(model))

        self.assertIn("MatrixFSDP runtime events:", text)
        self.assertIn("param_group=0", text)
        self.assertIn("name=pre_forward", text)
        self.assertIn("state=forward_resharded", text)

    def test_format_runtime_events_includes_scheduler_and_durations(self):
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8), reshard_after_forward=True, forward_prefetch=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2), reshard_after_forward=True, forward_prefetch=True),
        )
        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_unsharded_prefetch_units=0,
        )

        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        optim.step()
        text = format_runtime_events(summarize_runtime_events(model))

        self.assertIn("scheduler param_groups=(0, 1)", text)
        self.assertIn("forward_budget=0", text)
        self.assertIn("backward_budget=1", text)
        self.assertIn("blocked=1", text)
        self.assertIn("max_full_buffers=", text)
        self.assertIn("max_full_bytes=", text)
        self.assertIn("max_grad_bucket_bytes=", text)
        self.assertIn("duration_ms=", text)
        self.assertIn("event_stats top_by_sum_ms:", text)

    def test_rejects_unwrapped_module(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_param_groups(nn.Linear(4, 2))


if __name__ == "__main__":
    unittest.main()
