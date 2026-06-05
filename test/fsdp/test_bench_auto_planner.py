import unittest

from torch import nn

from test.fsdp.bench_auto_planner import PlannerBenchScenario, format_benchmark_table, run_planner_benchmark


class AutoPlannerBenchTest(unittest.TestCase):
    def test_run_planner_benchmark_returns_selected_policy_rows(self):
        rows = run_planner_benchmark(
            world_size=2,
            policies=("balanced", "min_comm"),
            scenarios=(PlannerBenchScenario("tiny", lambda: nn.Sequential(nn.Linear(4, 6, bias=False))),),
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual({row.scenario for row in rows}, {"tiny"})
        self.assertEqual({row.policy for row in rows}, {"balanced", "min_comm"})
        self.assertTrue(all(row.selected for row in rows))
        self.assertTrue(all(row.total_comm_units > 0 for row in rows))

    def test_run_planner_benchmark_can_show_all_candidates(self):
        rows = run_planner_benchmark(
            world_size=2,
            policies=("balanced",),
            scenarios=(PlannerBenchScenario("tiny", lambda: nn.Sequential(nn.Linear(4, 6, bias=False))),),
            show_candidates=True,
        )

        self.assertEqual(tuple(row.candidate for row in rows), ("matrix_row_block", "ordered_block", "whole_param"))
        self.assertEqual(sum(row.selected for row in rows), 1)

    def test_time_train_steps_can_time_all_candidates(self):
        from test.fsdp.bench_auto_planner import time_train_steps

        rows = time_train_steps(
            world_size=1,
            device="cpu",
            policies=("balanced",),
            scenarios=(PlannerBenchScenario("tiny", lambda: nn.Sequential(nn.Linear(4, 6, bias=False)), input_shape=(2, 4)),),
            warmup_steps=0,
            steps=1,
            time_candidates=True,
        )

        self.assertEqual(tuple(row.candidate for row in rows), ("matrix_row_block", "ordered_block", "whole_param"))
        self.assertEqual(sum(row.selected for row in rows), 1)
        self.assertTrue(all(row.avg_step_ms > 0 for row in rows))

    def test_format_benchmark_table_includes_core_columns(self):
        rows = run_planner_benchmark(
            world_size=2,
            policies=("balanced",),
            scenarios=(PlannerBenchScenario("tiny", lambda: nn.Sequential(nn.Linear(4, 6, bias=False))),),
        )

        table = format_benchmark_table(rows)

        self.assertIn("scenario", table)
        self.assertIn("policy", table)
        self.assertIn("candidate", table)
        self.assertIn("sel", table)
        self.assertIn("rank_units", table)
        self.assertIn("tiny", table)


if __name__ == "__main__":
    unittest.main()
