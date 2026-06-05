import json
import tempfile
import unittest

import torch

from matrix_fsdp.tools.runtime_profile import RuntimeProfileConfig, format_profile_report, main, run_profile


class RuntimeProfileToolTest(unittest.TestCase):
    def test_run_profile_returns_scheduler_and_runtime_summary(self):
        result = run_profile(
            RuntimeProfileConfig(
                model="mlp",
                mode="prefetch_profile_guided",
                layers=1,
                hidden=8,
                intermediate=16,
                batch_size=2,
                warmup_steps=0,
                profile_steps=1,
                steps=1,
            )
        )

        self.assertEqual(result["config"]["mode"], "prefetch_profile_guided")
        self.assertEqual(result["unit_summary"]["num_param_groups"], 1)
        self.assertEqual(result["unit_summary"]["num_units"], 1)
        self.assertEqual(result["param_group_summary"]["num_param_groups"], 1)
        self.assertEqual(result["param_group_summary"], result["unit_summary"])
        self.assertIn("avg_step_ms", result["step_stats"])
        self.assertEqual(result["runtime_summary"]["num_schedulers"], 1)
        self.assertEqual(len(result["runtime_summary"]["schedulers"][0]["profile_results"]), 4)
        self.assertIn("event_stats", result["runtime_summary"])
        self.assertTrue(any(event["duration_ms"] is not None for event in result["runtime_summary"]["events"]))
        self.assertTrue(all("active_full_param_bytes" in event for event in result["runtime_summary"]["events"]))
        self.assertIn("runtime_memory_snapshots", result["runtime_summary"]["schedulers"][0])

    def test_format_profile_report_includes_scheduler_and_events(self):
        result = run_profile(
            RuntimeProfileConfig(
                model="mlp",
                mode="prefetch_cap1",
                layers=1,
                hidden=8,
                intermediate=16,
                batch_size=2,
                warmup_steps=0,
                profile_steps=0,
                steps=1,
            )
        )

        report = format_profile_report(result)

        self.assertIn("MatrixFSDP runtime profile", report)
        self.assertIn("MatrixFSDP param groups:", report)
        self.assertIn("MatrixFSDP runtime events:", report)
        self.assertIn("scheduler param_groups=", report)
        self.assertIn("max_grad_bucket_bytes=", report)
        self.assertIn("event_stats top_by_sum_ms:", report)

    def test_bucket_copy_in_profile_reports_copy_in_and_enqueue_events(self):
        result = run_profile(
            RuntimeProfileConfig(
                model="mlp",
                mode="prefetch_bucket_copy_in",
                layers=1,
                hidden=8,
                intermediate=16,
                batch_size=2,
                warmup_steps=0,
                profile_steps=0,
                steps=1,
            )
        )

        event_names = [event["name"] for event in result["runtime_summary"]["events"]]

        self.assertIn("copy_in_grad_bucket", event_names)
        self.assertIn("enqueue_reduce_scatter_grad_bucket", event_names)
        self.assertTrue(any(name.startswith("grad_bucket_layout:") for name in event_names))

    def test_bucket_reduce_scatter_profile_reports_zero_copy_grad_bucket(self):
        result = run_profile(
            RuntimeProfileConfig(
                model="mlp",
                mode="zero_copy_grad_bucket",
                layers=1,
                hidden=8,
                intermediate=16,
                batch_size=2,
                warmup_steps=0,
                profile_steps=0,
                steps=1,
            )
        )

        event_names = [event["name"] for event in result["runtime_summary"]["events"]]

        self.assertIn("prepare_grad_bucket_zero_copy", event_names)
        self.assertGreater(result["runtime_summary"]["schedulers"][0]["max_grad_bucket_bytes"], 0)

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_matrix_owner_muon_profile_reports_split_qkv_memory_events(self):
        result = run_profile(
            RuntimeProfileConfig(
                model="transformer_split_qkv",
                mode="matrix_owner_muon_role_greedy_custom_collective_zero_copy_grad_bucket",
                optimizer="muon",
                layers=1,
                hidden=16,
                intermediate=32,
                heads=4,
                seq_len=8,
                batch_size=2,
                warmup_steps=0,
                profile_steps=0,
                steps=1,
            )
        )

        scheduler_summary = result["runtime_summary"]["schedulers"][0]
        event_names = [event["name"] for event in result["runtime_summary"]["events"]]

        self.assertEqual(result["config"]["mode"], "matrix_owner_muon_role_greedy_custom_collective_zero_copy_grad_bucket")
        self.assertIn("prepare_grad_bucket_zero_copy", event_names)
        self.assertGreater(scheduler_summary["max_grad_bucket_bytes"], 0)
        self.assertTrue(scheduler_summary["runtime_memory_snapshots"])
        self.assertTrue(
            any(
                snapshot["param_data_alias_full_buffer"]
                for snapshot in scheduler_summary["runtime_memory_snapshots"]
            )
        )
        self.assertTrue(
            any(
                snapshot["param_data_alias_local_shard"]
                for snapshot in scheduler_summary["runtime_memory_snapshots"]
            )
        )

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_matrix_owner_muon_copy_in_profile_reports_copy_in_grad_bucket(self):
        result = run_profile(
            RuntimeProfileConfig(
                model="transformer_split_qkv",
                mode="matrix_owner_muon_role_greedy_custom_collective",
                optimizer="muon",
                layers=1,
                hidden=16,
                intermediate=32,
                heads=4,
                seq_len=8,
                batch_size=2,
                warmup_steps=0,
                profile_steps=0,
                steps=1,
            )
        )

        event_names = [event["name"] for event in result["runtime_summary"]["events"]]

        self.assertEqual(result["config"]["mode"], "matrix_owner_muon_role_greedy_custom_collective")
        self.assertIn("prepare_grad_bucket_copy_in", event_names)
        self.assertIn("copy_in_grad_bucket", event_names)
        self.assertNotIn("prepare_grad_bucket_zero_copy", event_names)

    def test_main_can_write_json_output(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as output_file:
            exit_code = main(
                [
                    "--model",
                    "mlp",
                    "--mode",
                    "prefetch_profile_guided",
                    "--layers",
                    "1",
                    "--hidden",
                    "8",
                    "--intermediate",
                    "16",
                    "--batch-size",
                    "2",
                    "--warmup-steps",
                    "0",
                    "--profile-steps",
                    "1",
                    "--steps",
                    "1",
                    "--format",
                    "json",
                    "--output",
                    output_file.name,
                ]
            )
            output_file.seek(0)
            payload = json.load(output_file)

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["config"]["mode"], "prefetch_profile_guided")
        self.assertIn("runtime_summary", payload)
        self.assertIn("event_stats", payload["runtime_summary"])
        self.assertIn("communication_event_stats", payload["runtime_summary"])


if __name__ == "__main__":
    unittest.main()
