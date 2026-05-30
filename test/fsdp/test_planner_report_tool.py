import io
import json
import unittest
from contextlib import redirect_stdout

from matrix_fsdp.tools.planner_report import PlannerReportConfig, format_planner_report, main, run_planner_report


class PlannerReportToolTest(unittest.TestCase):
    def test_run_planner_report_selects_matrix_owner_tail_for_split_qkv(self):
        config = PlannerReportConfig(
            model="transformer_split_qkv",
            layers=1,
            hidden=16,
            intermediate=64,
            heads=4,
            world_size=8,
            policy="muon_shard_aware",
        )

        result = run_planner_report(config)
        row = result.group_rows[0]

        self.assertEqual(row.candidate, "matrix_owner_tail")
        self.assertTrue(row.selected)
        self.assertIn(("muon_matrix_owner", 7), row.block_kinds)
        self.assertIn(("adamw_tail_param", 6), row.block_kinds)
        self.assertEqual(result.rank_total_units, row.rank_units)
        self.assertEqual(result.rank_total_memory_bytes, row.rank_memory_bytes)
        self.assertEqual(result.rank_total_comm_bytes, row.rank_comm_bytes)
        self.assertEqual(row.planner_summary["layout"], row.layout_contract)
        self.assertEqual(row.planner_summary["report"], row.report)
        self.assertEqual(row.planner_summary["resources"], row.resources)
        self.assertEqual(row.layout_contract["rank_units"], row.rank_units)
        self.assertEqual(row.report["params_by_rank"], row.params_by_rank)
        self.assertEqual(row.resources["rank_memory_bytes"], row.rank_memory_bytes)
        self.assertTrue(any(value > 0 for value in result.rank_total_memory_bytes))
        self.assertTrue(any(value > 0 for value in result.rank_total_comm_bytes))

    def test_format_planner_report_includes_header_and_candidate_table(self):
        config = PlannerReportConfig(hidden=16, intermediate=64, heads=4)
        result = run_planner_report(config)

        text = format_planner_report(config, result)

        self.assertIn("MatrixFSDP auto planner report", text)
        self.assertIn("model=transformer_split_qkv", text)
        self.assertIn("matrix_owner_tail", text)
        self.assertIn("cost_terms", text)
        self.assertIn("constraints", text)
        self.assertIn("runtime", text)
        self.assertIn("no_split_matrix=7", text)
        self.assertIn("rank_load_summary", text)
        self.assertIn("rank_mem", text)
        self.assertIn("rank_comm", text)
        self.assertIn("memory_bytes", text)
        self.assertIn("muon_param_bytes", text)

    def test_block_unit_planner_report_summarizes_owner_rotation(self):
        config = PlannerReportConfig(
            model="transformer_split_qkv",
            layers=4,
            hidden=16,
            intermediate=64,
            heads=4,
            unit="block",
            world_size=8,
            policy="muon_shard_aware",
            show_candidates=False,
        )

        result = run_planner_report(config)

        self.assertEqual(len(result.group_rows), 0)
        self.assertEqual(len(result.unit_rows), 4)
        self.assertTrue(all(row.candidate == "matrix_owner_tail" for row in result.unit_rows))
        self.assertEqual(result.rank_total_units, (1536, 1344, 1088, 1280, 1088, 1280, 1536, 1344))
        self.assertTrue(any(value > 0 for value in result.rank_total_memory_bytes))
        self.assertTrue(any(value > 0 for value in result.rank_total_comm_bytes))
        self.assertTrue(any(value > 0 for value in result.rank_total_muon_param_bytes))
        self.assertTrue(any(value > 0 for value in result.rank_total_adamw_param_bytes))
        self.assertIn(("no_split_matrix", 7), result.unit_rows[0].constraint_counts)
        self.assertEqual(result.unit_rows[0].warnings, ())
        self.assertIn(("adamw_tail_param", 4), result.rank_block_kinds[1])
        self.assertIn(("muon_matrix_owner", 4), result.rank_block_kinds[6])
        self.assertEqual(result.unit_rows[0].planner_summary["layout"], result.unit_rows[0].layout_contract)
        self.assertEqual(result.unit_rows[0].report["rank_units"], result.unit_rows[0].rank_units)
        self.assertEqual(result.unit_rows[0].resources["rank_comm_bytes"], result.unit_rows[0].rank_comm_bytes)

    def test_block_unit_planner_report_can_use_round_robin_rotation(self):
        config = PlannerReportConfig(
            model="transformer_split_qkv",
            layers=4,
            hidden=16,
            intermediate=64,
            heads=4,
            unit="block",
            world_size=8,
            policy="muon_shard_aware",
            rotation_strategy="round_robin",
            show_candidates=False,
        )

        result = run_planner_report(config)

        self.assertEqual(result.rank_total_units, (1344, 1088, 832, 1024, 1280, 1536, 1792, 1600))

    def test_block_unit_planner_report_can_assign_owner_roles_greedily(self):
        config = PlannerReportConfig(
            model="transformer_split_qkv",
            layers=4,
            hidden=16,
            intermediate=64,
            heads=4,
            unit="block",
            world_size=8,
            policy="muon_shard_aware",
            owner_assignment="role_greedy",
            show_candidates=False,
        )

        result = run_planner_report(config)

        self.assertEqual(len(result.unit_rows), 4)
        self.assertTrue(all(row.candidate == "matrix_owner_tail_role_greedy" for row in result.unit_rows))
        self.assertEqual(result.rank_total_units, (1536, 1280, 1152, 1280, 1344, 1280, 1280, 1344))
        self.assertIn(("muon_matrix_owner", 4), result.rank_block_kinds[0])

    def test_planner_report_main_can_emit_json(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                (
                    "--model",
                    "transformer_split_qkv",
                    "--hidden",
                    "16",
                    "--intermediate",
                    "64",
                    "--heads",
                    "4",
                    "--world-size",
                    "8",
                    "--format",
                    "json",
                    "--selected-only",
                )
            )

        payload = json.loads(output.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["config"]["model"], "transformer_split_qkv")
        self.assertEqual(payload["config"]["rotation_strategy"], "greedy_balance")
        self.assertEqual(payload["config"]["owner_assignment"], "rotate")
        self.assertEqual(len(payload["group_rows"]), 1)
        self.assertEqual(payload["group_rows"][0]["candidate"], "matrix_owner_tail")
        self.assertIn(["muon_matrix_owner", 7], payload["group_rows"][0]["constraint_counts"])
        self.assertEqual(payload["group_rows"][0]["warnings"], [])
        self.assertIn("planner_summary", payload["group_rows"][0])
        self.assertIn("layout_contract", payload["group_rows"][0])
        self.assertIn("report", payload["group_rows"][0])
        self.assertIn("resources", payload["group_rows"][0])
        self.assertEqual(
            payload["group_rows"][0]["planner_summary"]["layout"],
            payload["group_rows"][0]["layout_contract"],
        )
        self.assertEqual(
            payload["group_rows"][0]["planner_summary"]["report"],
            payload["group_rows"][0]["report"],
        )
        self.assertEqual(
            payload["group_rows"][0]["planner_summary"]["resources"],
            payload["group_rows"][0]["resources"],
        )
        self.assertEqual(payload["group_rows"][0]["layout_contract"]["rank_units"], payload["group_rows"][0]["rank_units"])
        self.assertEqual(payload["group_rows"][0]["report"]["rank_units"], payload["group_rows"][0]["rank_units"])
        self.assertEqual(payload["unit_rows"], [])
        self.assertIn("rank_total_units", payload)
        self.assertIn("rank_total_memory_bytes", payload)
        self.assertIn("rank_total_comm_bytes", payload)
        self.assertTrue(any(value > 0 for value in payload["rank_total_memory_bytes"]))
        self.assertTrue(any(value > 0 for value in payload["rank_total_comm_bytes"]))


if __name__ == "__main__":
    unittest.main()
