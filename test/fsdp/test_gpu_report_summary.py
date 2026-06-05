import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUMMARY_SCRIPT = _REPO_ROOT / "scripts" / "summarize_gpu_comm_report.py"
_SPEC = importlib.util.spec_from_file_location("summarize_gpu_comm_report", _SUMMARY_SCRIPT)
assert _SPEC is not None
summarize_gpu_comm_report = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = summarize_gpu_comm_report
_SPEC.loader.exec_module(summarize_gpu_comm_report)


class GPUReportSummaryTest(unittest.TestCase):
    def test_summarizes_phase_delta_memory_imbalance_and_collective_timings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            _write_synthetic_report_logs(report_dir)

            summary = summarize_gpu_comm_report.summarize_report(report_dir)
            text = summarize_gpu_comm_report.format_summary(summary)

        self.assertEqual(len(summary["comm"]), 2)
        self.assertEqual(summary["comm"][0]["impl"], "native_group_broadcast")
        self.assertEqual(summary["comm"][1]["impl"], "native_sendrecv")

        phase_by_mode = {row.mode: row for row in summary["phase"]}
        self.assertIn("fsdp2", phase_by_mode)
        self.assertIn("matrix_owner_muon_role_greedy_custom_collective", phase_by_mode)
        self.assertEqual(phase_by_mode["fsdp2"].optimizer, "muon")
        self.assertEqual(phase_by_mode["fsdp2"].world, 8)
        self.assertAlmostEqual(phase_by_mode["fsdp2"].total_ms, 100.0)
        self.assertAlmostEqual(phase_by_mode["matrix_owner_muon_role_greedy_custom_collective"].total_ms, 80.0)

        self.assertEqual(len(summary["phase_vs_fsdp2"]), 1)
        comparison = summary["phase_vs_fsdp2"][0]
        self.assertEqual(comparison["mode"], "matrix_owner_muon_role_greedy_custom_collective")
        self.assertAlmostEqual(comparison["total_delta_ms"], -20.0)
        self.assertAlmostEqual(comparison["fwd_delta_ms"], -10.0)
        self.assertAlmostEqual(comparison["bwd_delta_ms"], -10.0)
        self.assertAlmostEqual(comparison["step_delta_ms"], 0.0)
        self.assertAlmostEqual(comparison["total_delta_pct"], -20.0)
        self.assertAlmostEqual(comparison["peak_mem_delta_mb"], 10.0)

        after_forward = next(row for row in summary["memory"] if row.phase == "after_forward")
        self.assertEqual(after_forward.rank_count, 2)
        self.assertAlmostEqual(after_forward.current_mem_max_mb, 120.0)
        self.assertAlmostEqual(after_forward.current_mem_imbalance_mb, 20.0)
        self.assertAlmostEqual(after_forward.peak_reserved_max_mb, 220.0)

        collective = summary["collective_timing"][0]
        self.assertEqual(collective.event_count, 2)
        self.assertAlmostEqual(collective.native_enqueue_total_ms, 0.5)
        self.assertAlmostEqual(collective.wait_total_ms, 1.5)
        self.assertEqual(collective.workspace_kinds, {"compact_rank_chunks": 1, "static": 1})
        self.assertEqual(
            collective.materialization_kinds,
            {"owner_segment:custom": 1, "single_rank_copy": 1},
        )

        self.assertIn("Matrix vs FSDP2", text)
        self.assertIn("total_delta_ms", text)
        self.assertIn("fwd_delta", text)
        self.assertIn("bwd_delta", text)
        self.assertIn("step_delta", text)
        self.assertIn("current_imb", text)
        self.assertIn("enqueue_total", text)

    def test_summarizes_runtime_communication_event_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            (report_dir / "runtime_profile_stats.log").write_text(
                "\n".join(
                    [
                        "    all_gather_enqueue count=16 timed=16 sum_ms=950.425 avg_ms=59.402 max_ms=238.245",
                        "    all_gather_wait count=16 timed=16 sum_ms=3.410 avg_ms=0.213 max_ms=0.384",
                        "    reduce_scatter_enqueue count=8 timed=8 sum_ms=13.178 avg_ms=1.647 max_ms=2.531",
                        "    reduce_scatter_wait count=8 timed=8 sum_ms=0.848 avg_ms=0.106 max_ms=0.230",
                        "  #10 name=enqueue_all_gather_full_params workspace=matrix_all_gather param_materialization=owner_segment:custom",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_gpu_comm_report.summarize_report(report_dir)

        collective = summary["collective_timing"][0]
        self.assertEqual(collective.event_count, 48)
        self.assertAlmostEqual(collective.native_enqueue_total_ms, 963.603)
        self.assertAlmostEqual(collective.native_enqueue_max_ms, 238.245)
        self.assertAlmostEqual(collective.wait_total_ms, 4.258)
        self.assertAlmostEqual(collective.wait_max_ms, 0.384)
        self.assertEqual(collective.workspace_kinds, {"matrix_all_gather": 1})
        self.assertEqual(collective.materialization_kinds, {"owner_segment:custom": 1})

    def test_empty_report_is_still_formattable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = summarize_gpu_comm_report.summarize_report(Path(tmpdir))
            text = summarize_gpu_comm_report.format_summary(summary)

        self.assertEqual(summary["comm"], [])
        self.assertEqual(summary["phase"], [])
        self.assertIn("no comm_*.log benchmark rows found", text)
        self.assertIn("no model_*.log phase timing rows found", text)


def _write_synthetic_report_logs(report_dir: Path) -> None:
    (report_dir / "comm_native_group_broadcast.log").write_text(
        "impl=native_group_broadcast chunk_fast_path=1 dtype=bfloat16 total_numel=36 "
        "max_ms=10.000 avg_ms=9.000 logical_GBps=7.000 shard_sizes=8,7,6,5,4,3,2,1\n",
        encoding="utf-8",
    )
    (report_dir / "comm_native_sendrecv.log").write_text(
        "impl=native_sendrecv chunk_fast_path=1 dtype=bfloat16 total_numel=36 "
        "max_ms=12.000 avg_ms=11.000 logical_GBps=6.000 shard_sizes=8,7,6,5,4,3,2,1\n",
        encoding="utf-8",
    )
    (report_dir / "model_phase.log").write_text(
        "\n".join(
            [
                "mode                                             model                  unit   device  world  optim  dtype     seq  budget   zero_ms  fwd_ms  bwd_ms  step_ms  total_ms  peak_mem_mb",
                "-----------------------------------------------  ---------------------  -----  ------  -----  -----  --------  ---  -------  -------  ------  ------  -------  --------  -----------",
                "fsdp2                                            transformer_split_qkv  block  cuda    8      muon   bfloat16  512           1.000    20.000  30.000  49.000   100.000   100.0",
                "matrix_owner_muon_role_greedy_custom_collective  transformer_split_qkv  block  cuda    8      muon   bfloat16  512  f=1/b=1  1.000    10.000  20.000  49.000   80.000    110.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "memory_trace.log").write_text(
        "\n".join(
            [
                "mode            rank  model                  unit   device  world  optim  dtype     seq  phase          current_mem_mb  peak_mem_mb  reserved_mb  peak_reserved_mb  budget   full_bufs  full_mb  local_mb  full_grad_mb  bucket_mb  local_grad_mb  opt_state_mb  pending_rs",
                "--------------  ----  ---------------------  -----  ------  -----  -----  --------  ---  -------------  --------------  -----------  -----------  ----------------  -------  ---------  -------  --------  ------------  ---------  -------------  ------------  ----------",
                "matrix_default  0     transformer_split_qkv  block  cuda    8      adamw  bfloat16  512  after_forward  100.0           130.0        180.0        200.0             f=1/b=1  1          20.0     10.0      0.0           0.0        0.0            0.0           0",
                "matrix_default  1     transformer_split_qkv  block  cuda    8      adamw  bfloat16  512  after_forward  120.0           140.0        190.0        220.0             f=1/b=1  1          20.0     10.0      0.0           0.0        0.0            0.0           0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "runtime_profile_matrix_custom_auto.log").write_text(
        "\n".join(
            [
                "enqueue_all_gather native_enqueue_ms=0.200ms wait_ms=0.500ms workspace=compact_rank_chunks materialize=owner_segment:custom",
                "wait_all_gather native_enqueue_ms=0.300ms wait_ms=1.000ms workspace=static materialize=single_rank_copy",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
