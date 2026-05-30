from dataclasses import replace
import unittest

from torch import nn

from matrix_fsdp import ParamShardHint
from matrix_fsdp.layout import LayoutSegment, ParamLayout, ParamSegment, MatrixGroupLayout
from matrix_fsdp.layout_report import report_group_layout
from matrix_fsdp.managed_param import ManagedParamRegistry
from matrix_fsdp.planner import (
    ParamBlock,
    matrix_row_block_builder,
    ordered_group_plan,
    parameter_boundary_plan,
    uniform_block_builder,
)
from matrix_fsdp.planner_eval import (
    PlannerCandidate,
    PlannerCostWeights,
    PlannerEvaluation,
    PlannerLayoutContract,
    PlannerResult,
    call_group_planner,
    compare_group_planners,
    estimate_layout_cost_breakdown,
    estimate_layout_cost,
    estimate_layout_resources,
    evaluate_group_planner,
    normalize_planner_candidates,
    select_best_group_planner,
)


class PlannerEvalTest(unittest.TestCase):
    def test_evaluate_group_planner_returns_valid_layout_report_and_cost(self):
        params = self._matrix_params()

        evaluation = evaluate_group_planner(
            "matrix_rows",
            params,
            2,
            lambda planner_params, world_size: ordered_group_plan(
                planner_params,
                world_size,
                block_builder=matrix_row_block_builder(lambda param: 2),
            ),
        )

        self.assertEqual(evaluation.name, "matrix_rows")
        self.assertIsInstance(evaluation, PlannerResult)
        self.assertIs(PlannerEvaluation, PlannerResult)
        self.assertEqual(evaluation.planner_name, "matrix_rows")
        self.assertTrue(evaluation.constraints_satisfied)
        self.assertEqual(evaluation.warnings, ())
        self.assertEqual(evaluation.world_size, 2)
        self.assertEqual(evaluation.report.rank_units, (16, 16))
        self.assertEqual(evaluation.report.num_split_params, 1)
        self.assertEqual(evaluation.report.total_comm_units, 64)
        self.assertEqual(evaluation.num_collectives, 2)
        self.assertEqual(evaluation.cost, 18.0)
        self.assertEqual(evaluation.cost_breakdown.terms, {"collective": 2.0, "max_rank_unit": 16.0})

    def test_planner_result_metadata_exposes_stable_runtime_contract(self):
        params = self._matrix_params()
        result = evaluate_group_planner("whole_param", params, 2, parameter_boundary_plan)

        self.assertEqual(
            result.as_metadata(),
            {
                "planner_name": "whole_param",
                "policy": None,
                "cost": result.cost,
                "constraints_satisfied": True,
                "constraints": (
                    {
                        "fqn": "0.weight",
                        "constraints": (),
                        "block_shape": None,
                        "optimizer_type": None,
                        "runtime_kind": None,
                        "parallel_role": None,
                        "expert_group_id": None,
                    },
                    {
                        "fqn": "1.weight",
                        "constraints": (),
                        "block_shape": None,
                        "optimizer_type": None,
                        "runtime_kind": None,
                        "parallel_role": None,
                        "expert_group_id": None,
                    },
                ),
                "constraint_counts": (),
                "warnings": (),
                "runtime_compatible": result.runtime_compatible,
                "runtime_mode": result.runtime_mode,
                "runtime_reason": result.runtime_reason,
                "runtime_requires_flat_reorder": result.runtime_requires_flat_reorder,
                "matrix_shard_compatible": result.matrix_shard_compatible,
                "matrix_shard_reason": result.matrix_shard_reason,
                "world_size": 2,
                "shard_mesh_dim": None,
                "rank_units": result.layout.shard_sizes,
                "layout_contract": result.layout_contract().as_metadata(),
                "report": result.report.as_metadata(),
                "cost_terms": tuple(result.cost_breakdown.terms.items()),
                "resource_estimate": result.resource_estimate.as_metadata(),
                "expert_owner_groups": (),
                "rank_role_units": {
                    "expert": (0, 0),
                    "router": (0, 0),
                    "norm": (0, 0),
                    "dense": (24, 8),
                },
            },
        )

    def test_planner_result_summary_exposes_layout_contract(self):
        params = self._matrix_params()
        result = evaluate_group_planner("whole_param", params, 2, parameter_boundary_plan)

        contract = result.layout_contract()
        summary = result.summary()

        self.assertIsInstance(contract, PlannerLayoutContract)
        self.assertEqual(contract.total_numel, result.layout.total_numel)
        self.assertEqual(contract.world_size, result.layout.world_size)
        self.assertEqual(contract.shard_sizes, result.layout.shard_sizes)
        self.assertEqual(contract.params_for_rank(0), ("0.weight",))
        self.assertEqual(contract.owner_ranks("1.weight"), (1,))
        self.assertEqual(contract.to_shard_plan(), result.layout.to_shard_plan())
        self.assertEqual(summary["planner_name"], "whole_param")
        self.assertEqual(summary["layout"], contract.as_metadata())
        self.assertEqual(summary["report"], result.report.as_metadata())
        self.assertEqual(summary["resources"], result.resource_estimate.as_metadata())
        self.assertEqual(
            contract.as_metadata()["params"],
            (
                {
                    "fqn": "0.weight",
                    "global_start": 0,
                    "global_end": 24,
                    "numel": 24,
                    "owner_ranks": (0,),
                    "segments": (
                        {
                            "fqn": "0.weight",
                            "rank": 0,
                            "global_start": 0,
                            "global_end": 24,
                            "local_start": 0,
                            "local_end": 24,
                            "numel": 24,
                        },
                    ),
                },
                {
                    "fqn": "1.weight",
                    "global_start": 24,
                    "global_end": 32,
                    "numel": 8,
                    "owner_ranks": (1,),
                    "segments": (
                        {
                            "fqn": "1.weight",
                            "rank": 1,
                            "global_start": 24,
                            "global_end": 32,
                            "local_start": 0,
                            "local_end": 8,
                            "numel": 8,
                        },
                    ),
                },
            ),
        )
        result.validate()

    def test_planner_result_validate_rejects_mismatched_report(self):
        params = self._matrix_params()
        result = evaluate_group_planner("whole_param", params, 2, parameter_boundary_plan)
        invalid_result = replace(result, report=replace(result.report, rank_units=(1, 2)))

        with self.assertRaisesRegex(ValueError, "rank_units"):
            invalid_result.validate()

    def test_layout_contract_validate_rejects_invalid_param_segment(self):
        invalid_layout = MatrixGroupLayout.from_rank_segments(
            total_numel=8,
            rank_segments=((LayoutSegment(0, 4, 0),), (LayoutSegment(4, 8, 0),)),
            params=(
                ParamLayout(
                    fqn="w",
                    global_start=0,
                    global_end=8,
                    segments=(ParamSegment("w", 0, 6, 8, 0),),
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "not contained"):
            PlannerLayoutContract(invalid_layout).validate()

    def test_call_group_planner_normalizes_legacy_outputs(self):
        params = self._matrix_params()

        layout_result = call_group_planner(parameter_boundary_plan, params, 2)
        plan_result = call_group_planner(
            lambda _params, _world_size: layout_result.layout.to_shard_plan(),
            params,
            2,
            name="legacy_plan",
        )
        existing_result = call_group_planner(lambda _params, _world_size: layout_result, params, 2)

        self.assertIsInstance(layout_result, PlannerResult)
        self.assertEqual(layout_result.planner_name, "parameter_boundary_plan")
        self.assertEqual(plan_result.planner_name, "legacy_plan")
        self.assertEqual(plan_result.layout.shard_sizes, layout_result.layout.shard_sizes)
        self.assertEqual(existing_result.layout, layout_result.layout)
        self.assertEqual(existing_result.runtime_mode, layout_result.runtime_mode)

    def test_call_group_planner_validates_existing_planner_result_outputs(self):
        params = self._matrix_params()
        layout_result = call_group_planner(parameter_boundary_plan, params, 2)
        invalid_layout = MatrixGroupLayout(
            total_numel=layout_result.layout.total_numel,
            ranks=layout_result.layout.ranks[:1],
            params=layout_result.layout.params,
        )
        invalid_result = replace(layout_result, layout=invalid_layout)

        with self.assertRaisesRegex(ValueError, "expected world_size=2"):
            call_group_planner(lambda _params, _world_size: invalid_result, params, 2)

    def test_runtime_unsupported_candidate_is_penalized_and_sorted_last(self):
        params = self._single_matrix_params()
        unsupported_layout = self._unsupported_repeated_segment_layout()

        evaluations = compare_group_planners(
            params,
            2,
            {
                "unsupported": lambda _params, _world_size: unsupported_layout,
                "whole_param": parameter_boundary_plan,
            },
            weights=PlannerCostWeights(
                runtime_unsupported=1000.0,
                collective=0.0,
                max_rank_unit=0.0,
            ),
        )

        self.assertEqual(tuple(evaluation.name for evaluation in evaluations), ("whole_param", "unsupported"))
        unsupported = evaluations[1]
        self.assertFalse(unsupported.runtime_compatible)
        self.assertEqual(unsupported.runtime_mode, "unsupported")
        self.assertIn("runtime_unsupported", unsupported.cost_breakdown.terms)
        self.assertIn("runtime unsupported", unsupported.warnings[0])

    def test_compare_group_planners_orders_by_estimated_cost(self):
        params = self._matrix_params()
        evaluations = compare_group_planners(
            params,
            2,
            {
                "whole_param": parameter_boundary_plan,
                "matrix_rows": lambda planner_params, world_size: ordered_group_plan(
                    planner_params,
                    world_size,
                    block_builder=matrix_row_block_builder(lambda param: 2),
                ),
            },
        )

        self.assertEqual(tuple(evaluation.name for evaluation in evaluations), ("matrix_rows", "whole_param"))
        self.assertLess(evaluations[0].report.max_rank_units, evaluations[1].report.max_rank_units)

    def test_compare_group_planners_accepts_candidate_objects(self):
        params = self._matrix_params()
        candidates = (
            PlannerCandidate(
                name="whole_param",
                planner=parameter_boundary_plan,
                padding_units=20,
            ),
            PlannerCandidate(
                name="matrix_rows",
                planner=lambda planner_params, world_size: ordered_group_plan(
                    planner_params,
                    world_size,
                    block_builder=matrix_row_block_builder(lambda param: 2),
                ),
                blocks=(ParamBlock("0.weight", 0, 16, 0, "matrix_row_block"),),
            ),
        )

        evaluations = compare_group_planners(params, 2, candidates)

        self.assertEqual(tuple(evaluation.name for evaluation in evaluations), ("matrix_rows", "whole_param"))
        self.assertEqual(evaluations[0].report.blocks_by_kind, {"matrix_row_block": 1})
        self.assertEqual(evaluations[1].padding_units, 20)

    def test_normalize_planner_candidates_preserves_legacy_mapping_metadata(self):
        candidates = normalize_planner_candidates(
            {"whole_param": parameter_boundary_plan},
            blocks_by_name={"whole_param": (ParamBlock("0.weight", 0, 16, 0),)},
            padding_units_by_name={"whole_param": 3},
            num_collectives_by_name={"whole_param": 4},
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "whole_param")
        self.assertIs(candidates[0].planner, parameter_boundary_plan)
        self.assertEqual(candidates[0].blocks[0].fqn, "0.weight")
        self.assertEqual(candidates[0].padding_units, 3)
        self.assertEqual(candidates[0].num_collectives, 4)

    def test_select_best_group_planner_penalizes_padding(self):
        params = self._matrix_params()
        best = select_best_group_planner(
            params,
            2,
            {
                "whole_param": parameter_boundary_plan,
                "matrix_rows_with_padding": lambda planner_params, world_size: ordered_group_plan(
                    planner_params,
                    world_size,
                    block_builder=matrix_row_block_builder(lambda param: 2),
                ),
            },
            padding_units_by_name={"matrix_rows_with_padding": 20},
        )

        self.assertEqual(best.name, "whole_param")

    def test_split_param_penalty_can_change_winner(self):
        params = self._matrix_params()
        best = select_best_group_planner(
            params,
            2,
            {
                "whole_param": parameter_boundary_plan,
                "matrix_rows": lambda planner_params, world_size: ordered_group_plan(
                    planner_params,
                    world_size,
                    block_builder=matrix_row_block_builder(lambda param: 2),
                ),
            },
            weights=PlannerCostWeights(split_param=20.0),
        )

        self.assertEqual(best.name, "whole_param")

    def test_blocks_feed_report_statistics(self):
        params = self._matrix_params()
        block_builder = uniform_block_builder(lambda param: 4, kind="quant_block")
        blocks = tuple(block for param in params for block in block_builder(param))

        evaluation = evaluate_group_planner(
            "uniform",
            params,
            2,
            lambda planner_params, world_size: ordered_group_plan(
                planner_params,
                world_size,
                block_builder=block_builder,
            ),
            blocks=blocks,
        )

        self.assertEqual(evaluation.report.num_blocks, 8)
        self.assertEqual(evaluation.report.blocks_by_kind, {"quant_block": 8})

    def test_estimate_layout_cost_uses_all_terms(self):
        params = self._matrix_params()
        evaluation = evaluate_group_planner(
            "matrix_rows",
            params,
            2,
            lambda planner_params, world_size: ordered_group_plan(
                planner_params,
                world_size,
                block_builder=matrix_row_block_builder(lambda param: 2),
            ),
        )

        cost = estimate_layout_cost(
            evaluation.report,
            weights=PlannerCostWeights(collective=2.0, max_rank_unit=3.0, padding_unit=5.0, split_param=7.0),
            padding_units=11,
            num_collectives=4,
        )

        self.assertEqual(cost, 2.0 * 4 + 3.0 * 16 + 5.0 * 11 + 7.0 * 1)

    def test_estimate_layout_cost_uses_communication_terms(self):
        compact = MatrixGroupLayout.from_rank_segments(
            total_numel=16,
            rank_segments=((LayoutSegment(0, 8, 0),), (LayoutSegment(8, 16, 0),)),
        )
        fragmented = MatrixGroupLayout.from_rank_segments(
            total_numel=16,
            rank_segments=(
                (LayoutSegment(0, 4, 0), LayoutSegment(8, 12, 4)),
                (LayoutSegment(4, 8, 0), LayoutSegment(12, 16, 4)),
            ),
        )

        compact_cost = estimate_layout_cost(
            report_group_layout(compact),
            weights=PlannerCostWeights(rank_segment=10.0, max_rank_segment=3.0, total_comm_unit=0.5),
        )
        fragmented_cost = estimate_layout_cost(
            report_group_layout(fragmented),
            weights=PlannerCostWeights(rank_segment=10.0, max_rank_segment=3.0, total_comm_unit=0.5),
        )

        self.assertGreater(fragmented_cost, compact_cost)

    def test_estimate_layout_cost_uses_fragmentation_and_block_terms(self):
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=16,
            rank_segments=(
                (LayoutSegment(0, 4, 0), LayoutSegment(8, 12, 4)),
                (LayoutSegment(4, 8, 0), LayoutSegment(12, 16, 4)),
            ),
        )
        blocks = (
            ParamBlock("p0", 0, 4, 0, "ordered_block"),
            ParamBlock("p0", 4, 8, 1, "ordered_block"),
            ParamBlock("p1", 8, 12, 0, "matrix_row_block"),
            ParamBlock("p1", 12, 16, 1, "matrix_row_block"),
        )
        report = report_group_layout(layout, blocks)

        breakdown = estimate_layout_cost_breakdown(
            report,
            weights=PlannerCostWeights(
                collective=0.0,
                max_rank_unit=0.0,
                fragmented_rank_segment=3.0,
                fragmented_rank_unit=0.5,
                imbalance_unit=2.0,
                block=1.0,
                ordered_block=5.0,
                matrix_row_block=7.0,
            ),
        )

        self.assertEqual(
            breakdown.terms,
            {
                "fragmented_rank_segment": 6.0,
                "fragmented_rank_unit": 8.0,
                "block": 4.0,
                "ordered_block": 10.0,
                "matrix_row_block": 14.0,
            },
        )
        self.assertEqual(breakdown.total, 42.0)

    def test_estimate_layout_cost_uses_matrix_owner_block_terms(self):
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=12,
            rank_segments=((LayoutSegment(0, 8, 0),), (LayoutSegment(8, 12, 0),)),
        )
        blocks = (
            ParamBlock("w0", 0, 4, 0, "muon_matrix_owner"),
            ParamBlock("w1", 4, 8, 0, "matrix_owner_matrix"),
            ParamBlock("norm", 8, 10, 0, "adamw_tail_param"),
            ParamBlock("other", 10, 12, 0, "matrix_owner_tail_param"),
        )
        report = report_group_layout(layout, blocks)

        breakdown = estimate_layout_cost_breakdown(
            report,
            weights=PlannerCostWeights(
                collective=0.0,
                max_rank_unit=0.0,
                muon_matrix_owner=2.0,
                matrix_owner_matrix=3.0,
                adamw_tail_param=5.0,
                matrix_owner_tail_param=7.0,
            ),
        )

        self.assertEqual(
            breakdown.terms,
            {
                "matrix_owner_matrix": 3.0,
                "muon_matrix_owner": 2.0,
                "matrix_owner_tail_param": 7.0,
                "adamw_tail_param": 5.0,
            },
        )
        self.assertEqual(breakdown.total, 17.0)

    def test_estimate_layout_cost_uses_runtime_terms(self):
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=8,
            rank_segments=((LayoutSegment(0, 4, 0),), (LayoutSegment(4, 8, 0),)),
        )
        report = report_group_layout(layout)

        breakdown = estimate_layout_cost_breakdown(
            report,
            weights=PlannerCostWeights(
                collective=0.0,
                max_rank_unit=0.0,
                runtime_unsupported=11.0,
                runtime_flat_reorder=13.0,
                runtime_segment_runtime=17.0,
            ),
            runtime_compatible=False,
            runtime_mode="flat_reorder",
        )

        self.assertEqual(
            breakdown.terms,
            {
                "runtime_unsupported": 11.0,
                "runtime_flat_reorder": 13.0,
            },
        )

    def test_estimate_layout_resources_tracks_optimizer_and_comm_bytes(self):
        module = nn.Sequential(nn.Linear(4, 4, bias=False), nn.LayerNorm(4))
        params = ManagedParamRegistry.from_module(
            module,
            shard_hints={
                "0.weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
                "1.weight": ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
                "1.bias": ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
            },
        ).params

        layout = parameter_boundary_plan(params, 2)
        resources = estimate_layout_resources(layout, params)

        self.assertEqual(resources.rank_param_bytes, (64, 32))
        self.assertEqual(resources.rank_grad_bytes, (64, 32))
        self.assertEqual(resources.rank_optimizer_bytes, (64, 64))
        self.assertEqual(resources.rank_memory_bytes, (192, 128))
        self.assertEqual(resources.rank_comm_bytes, (64, 128))
        self.assertEqual(resources.rank_muon_param_bytes, (64, 0))
        self.assertEqual(resources.rank_adamw_param_bytes, (0, 32))
        self.assertEqual(resources.total_param_bytes, 96)
        self.assertEqual(resources.total_memory_bytes, 320)
        self.assertEqual(resources.total_comm_bytes, 192)
        self.assertEqual(resources.memory_imbalance_bytes, 64)
        self.assertEqual(resources.muon_param_imbalance_bytes, 64)

    def test_estimate_layout_cost_uses_resource_terms(self):
        module = nn.Sequential(nn.Linear(4, 4, bias=False), nn.LayerNorm(4))
        params = ManagedParamRegistry.from_module(
            module,
            shard_hints={
                "0.weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
                "1.weight": ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
                "1.bias": ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
            },
        ).params
        evaluation = evaluate_group_planner(
            "whole_param",
            params,
            2,
            parameter_boundary_plan,
            weights=PlannerCostWeights(
                collective=0.0,
                max_rank_unit=0.0,
                max_rank_memory_byte=0.1,
                memory_imbalance_byte=0.2,
                total_comm_byte=0.3,
            ),
        )

        self.assertEqual(
            set(evaluation.cost_breakdown.terms),
            {"max_rank_memory_byte", "memory_imbalance_byte", "total_comm_byte"},
        )
        self.assertAlmostEqual(evaluation.cost_breakdown.terms["max_rank_memory_byte"], 19.2)
        self.assertAlmostEqual(evaluation.cost_breakdown.terms["memory_imbalance_byte"], 12.8)
        self.assertAlmostEqual(evaluation.cost_breakdown.terms["total_comm_byte"], 57.6)
        self.assertAlmostEqual(evaluation.cost, 89.6)
        self.assertEqual(evaluation.resource_estimate.rank_memory_bytes, (192, 128))

    def test_evaluate_group_planner_accounts_for_padding_alignment(self):
        params = self._matrix_params()

        evaluation = evaluate_group_planner(
            "matrix_rows",
            params,
            2,
            lambda planner_params, world_size: ordered_group_plan(
                planner_params,
                world_size,
                block_builder=matrix_row_block_builder(lambda param: 2),
            ),
            weights=PlannerCostWeights(padding_unit=5.0),
            padding_alignment=24,
        )

        self.assertEqual(evaluation.report.estimated_padding_units, 16)
        self.assertEqual(evaluation.cost, 2.0 + 16.0 + 5.0 * 16.0)

    def test_select_best_group_planner_rejects_empty_candidates(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            select_best_group_planner(self._matrix_params(), 2, {})

    def _matrix_params(self):
        module = nn.Sequential(
            nn.Linear(4, 6, bias=False),
            nn.Linear(4, 2, bias=False),
        )
        return ManagedParamRegistry.from_module(module).params

    def _single_matrix_params(self):
        module = nn.Linear(16, 1, bias=False)
        return ManagedParamRegistry.from_module(module).params

    def _unsupported_repeated_segment_layout(self):
        return MatrixGroupLayout.from_rank_segments(
            total_numel=16,
            rank_segments=(
                (LayoutSegment(0, 4, 0), LayoutSegment(8, 12, 4)),
                (LayoutSegment(4, 8, 0), LayoutSegment(12, 16, 4)),
            ),
            params=(
                ParamLayout(
                    fqn="weight",
                    global_start=0,
                    global_end=16,
                    segments=(
                        ParamSegment("weight", 0, 0, 4, 0),
                        ParamSegment("weight", 1, 4, 8, 0),
                        ParamSegment("weight", 0, 8, 12, 4),
                        ParamSegment("weight", 1, 12, 16, 4),
                    ),
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
