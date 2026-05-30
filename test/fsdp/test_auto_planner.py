import copy
import unittest

import torch
from torch import nn

from matrix_fsdp import (
    ParamShardHint,
    MatrixFSDPOptimizer,
    auto_group_plan,
    build_shard_hints,
    load_balanced_matrix_owner_tail_plan,
    make_muon_shard_aware_group_planner,
    ordered_matrix_owner_tail_plan,
    matrix_fully_shard,
)
from matrix_fsdp.auto_planner import (
    AutoPlannerPolicy,
    available_auto_planner_policies,
    build_auto_planner_report,
    default_auto_group_planner_candidates,
    evaluate_auto_group_planners,
    format_auto_planner_report,
    resolve_auto_planner_policy,
    select_auto_group_plan,
)
from matrix_fsdp.layout import LayoutSegment, ParamLayout, ParamSegment, MatrixGroupLayout
from matrix_fsdp.layout_validator import explain_runtime_layout_compatibility
from matrix_fsdp.managed_param import ManagedParamRegistry
from matrix_fsdp.planner import hinted_ordered_group_plan, parameter_boundary_plan
from matrix_fsdp.placement import matrix_shard_from_layout
from matrix_fsdp.planner_eval import PlannerCandidate, PlannerCostWeights


class AutoPlannerTest(unittest.TestCase):
    def test_default_candidates_include_block_metadata(self):
        params = self._matrix_params()

        candidates = default_auto_group_planner_candidates(params, world_size=2, target_block_units=8)

        self.assertEqual(
            tuple(candidate.name for candidate in candidates.candidates),
            ("whole_param", "ordered_block", "matrix_row_block"),
        )
        self.assertEqual(tuple(candidates.planners), ("whole_param", "ordered_block", "matrix_row_block"))
        self.assertIs(candidates.candidate_by_name["whole_param"].planner, candidates.planners["whole_param"])
        self.assertEqual(candidates.blocks_by_name["whole_param"][0].kind, "parameter")
        self.assertEqual(candidates.blocks_by_name["ordered_block"][0].kind, "ordered_block")
        self.assertEqual(candidates.blocks_by_name["matrix_row_block"][0].kind, "matrix_row_block")

    def test_muon_aware_candidates_include_matrix_owner_tail_metadata(self):
        params = self._split_qkv_like_params()

        candidates = default_auto_group_planner_candidates(
            params,
            world_size=8,
            include_matrix_owner_tail=True,
        )

        self.assertIn("matrix_owner_tail", candidates.planners)
        self.assertIn("matrix_owner_tail", candidates.candidate_by_name)
        self.assertEqual(
            candidates.blocks_by_name["matrix_owner_tail"][0].kind,
            "matrix_owner_matrix",
        )
        self.assertEqual(
            candidates.blocks_by_name["matrix_owner_tail"][-1].kind,
            "matrix_owner_tail_param",
        )

    def test_default_candidates_include_expert_owner_tail_for_moe_hints(self):
        model = _TinyMoEForPlanner()
        params = ManagedParamRegistry.from_module(model, shard_hints=build_shard_hints(model)).params

        candidates = default_auto_group_planner_candidates(params, world_size=2)
        layout = candidates.planners["expert_owner_tail"](params, 2)

        self.assertIn("expert_owner_tail", candidates.planners)
        self.assertIn("expert_owner_param", {block.kind for block in candidates.blocks_by_name["expert_owner_tail"]})
        self.assertEqual(layout.owner_ranks("moe.experts.0.w1.weight"), layout.owner_ranks("moe.experts.0.w2.weight"))
        self.assertEqual(layout.owner_ranks("moe.experts.1.w1.weight"), layout.owner_ranks("moe.experts.1.w2.weight"))
        compatibility = explain_runtime_layout_compatibility(layout, params, world_size=2)
        self.assertTrue(compatibility.compatible)

        report_row = next(row for row in build_auto_planner_report(params, world_size=2) if row.candidate == "expert_owner_tail")
        self.assertEqual(
            tuple(group["expert_group_id"] for group in report_row.expert_owner_groups),
            ("moe.experts.0", "moe.experts.1"),
        )
        self.assertEqual(report_row.rank_role_units["expert"], (64, 64))
        self.assertEqual(report_row.rank_role_units["router"], (8, 0))

    def test_evaluate_auto_group_planners_orders_default_candidates(self):
        params = self._matrix_params()

        evaluations = evaluate_auto_group_planners(params, world_size=2, target_block_units=8)

        self.assertEqual(evaluations[0].name, "matrix_row_block")
        self.assertEqual(evaluations[0].policy, "balanced")
        self.assertEqual(evaluations[0].report.rank_units, (16, 16))
        self.assertEqual(evaluations[0].report.blocks_by_kind, {"matrix_row_block": 4})

    def test_select_auto_group_plan_can_penalize_split_params(self):
        params = self._matrix_params()

        evaluation = select_auto_group_plan(
            params,
            world_size=2,
            target_block_units=8,
            weights=PlannerCostWeights(split_param=20.0),
        )

        self.assertEqual(evaluation.name, "whole_param")

    def test_auto_planner_policies_select_expected_tradeoffs(self):
        params = self._matrix_params()

        self.assertEqual(
            select_auto_group_plan(params, world_size=2, policy="balanced", target_block_units=8).name,
            "matrix_row_block",
        )
        self.assertEqual(
            select_auto_group_plan(params, world_size=2, policy="max_balance", target_block_units=8).name,
            "matrix_row_block",
        )
        self.assertEqual(
            select_auto_group_plan(params, world_size=2, policy="min_comm", target_block_units=8).name,
            "whole_param",
        )
        self.assertEqual(
            select_auto_group_plan(params, world_size=2, policy="muon_full_matrix", target_block_units=8).name,
            "whole_param",
        )

    def test_muon_shard_aware_policy_selects_matrix_owner_tail_candidate(self):
        params = self._split_qkv_like_params()

        evaluations = evaluate_auto_group_planners(params, world_size=8, policy="muon_shard_aware")
        evaluation = select_auto_group_plan(params, world_size=8, policy="muon_shard_aware")
        costs_by_name = {candidate.name: candidate.cost for candidate in evaluations}

        self.assertEqual(evaluation.name, "matrix_owner_tail")
        self.assertEqual(evaluation.policy, "muon_shard_aware")
        self.assertEqual(
            evaluation.report.blocks_by_kind,
            {"matrix_owner_matrix": 7, "matrix_owner_tail_param": 4},
        )
        self.assertLess(costs_by_name["matrix_owner_tail"], costs_by_name["whole_param"])

    def test_muon_shard_aware_policy_preserves_optimizer_hint_report(self):
        params = self._split_qkv_like_params(shard_hints=True)

        evaluation = select_auto_group_plan(params, world_size=8, policy="muon_shard_aware")

        self.assertEqual(evaluation.name, "matrix_owner_tail")
        self.assertEqual(
            evaluation.report.blocks_by_kind,
            {"muon_matrix_owner": 7, "adamw_tail_param": 4},
        )

    def test_muon_zero_copy_policy_preserves_muon_hints_and_prefers_matrix_shard(self):
        params = self._split_qkv_like_params(shard_hints=True)

        evaluations = evaluate_auto_group_planners(params, world_size=8, policy="muon_zero_copy")
        evaluation = evaluations[0]
        candidates_by_name = {candidate.name: candidate for candidate in evaluations}

        self.assertEqual(evaluation.name, "hinted_mixed")
        self.assertEqual(evaluation.policy, "muon_zero_copy")
        self.assertEqual(evaluation.runtime_mode, "matrix_shard")
        self.assertEqual(evaluation.report.blocks_by_kind, {"parameter": 11})
        self.assertEqual(evaluation.layout.owner_ranks("q.weight"), (0,))
        self.assertEqual(evaluation.layout.owner_ranks("mlp.down.weight"), (5,))
        self.assertIn("matrix_owner_tail", candidates_by_name)
        self.assertEqual(candidates_by_name["matrix_owner_tail"].runtime_mode, "flat_reorder")
        self.assertIn("runtime_flat_reorder", candidates_by_name["matrix_owner_tail"].cost_breakdown.terms)

    def test_build_auto_planner_report_exposes_candidate_cost_details(self):
        params = self._split_qkv_like_params(shard_hints=True)

        rows = build_auto_planner_report(params, world_size=8, policy="muon_shard_aware")
        selected = rows[0]

        self.assertEqual(selected.candidate, "matrix_owner_tail")
        self.assertTrue(selected.selected)
        self.assertEqual(selected.block_kinds, (("adamw_tail_param", 4), ("muon_matrix_owner", 7)))
        self.assertIn(("max_rank_unit", 512.0), selected.cost_terms)
        self.assertEqual(selected.params_by_rank[0], ("q.weight",))
        self.assertTrue(selected.runtime_compatible)
        self.assertIn(selected.runtime_mode, {"matrix_shard", "flat_reorder", "segment_runtime"})
        self.assertTrue(any(value > 0 for value in selected.rank_memory_bytes))
        self.assertTrue(any(value > 0 for value in selected.rank_comm_bytes))
        self.assertTrue(any(value > 0 for value in selected.rank_muon_param_bytes))
        self.assertTrue(any(value > 0 for value in selected.rank_adamw_param_bytes))
        self.assertGreater(selected.total_comm_bytes, 0)
        self.assertGreater(selected.max_rank_memory_bytes, 0)
        self.assertGreater(len(rows), 1)

    def test_format_auto_planner_report_includes_blocks_cost_terms_and_rank_params(self):
        params = self._split_qkv_like_params(shard_hints=True)
        rows = build_auto_planner_report(params, world_size=8, policy="muon_shard_aware", show_candidates=False)

        text = format_auto_planner_report(rows)

        self.assertIn("candidate", text)
        self.assertIn("matrix_owner_tail", text)
        self.assertIn("muon_matrix_owner=7", text)
        self.assertIn("adamw_tail_param=4", text)
        self.assertIn("runtime", text)
        self.assertIn("rank_mem", text)
        self.assertIn("rank_comm", text)
        self.assertIn("rank_muon", text)
        self.assertIn("rank_adamw", text)
        self.assertIn("max_rank_unit=512", text)
        self.assertIn("r0:q.weight", text)

    def test_auto_planner_candidates_respect_matrix_owner_hints(self):
        params = self._matrix_params(
            shard_hints={"0.weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner")}
        )

        candidates = default_auto_group_planner_candidates(params, world_size=2, target_block_units=8)

        self.assertIn("hinted_mixed", candidates.planners)
        self.assertEqual(candidates.blocks_by_name["matrix_row_block"][0].kind, "parameter")
        self.assertEqual(candidates.blocks_by_name["matrix_row_block"][0].fqn, "0.weight")

    def test_hinted_mixed_candidate_uses_explicit_hint_boundaries(self):
        params = self._matrix_params(
            shard_hints={
                "0.weight": ParamShardHint(split_granularity="row_block", block_shape=(2, 4)),
                "1.weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
            }
        )

        candidates = default_auto_group_planner_candidates(params, world_size=2, target_block_units=8)
        layout = candidates.planners["hinted_mixed"](params, 2)

        self.assertEqual(candidates.blocks_by_name["hinted_mixed"][0].kind, "matrix_row_block")
        self.assertEqual(candidates.blocks_by_name["hinted_mixed"][-1].kind, "parameter")
        self.assertEqual(layout.owner_ranks("0.weight"), (0, 1))
        self.assertEqual(layout.owner_ranks("1.weight"), (1,))
        self.assertEqual(matrix_shard_from_layout(layout).shard_lengths(layout.total_numel), layout.shard_sizes)

    def test_auto_planner_evaluates_hinted_candidates(self):
        params = self._matrix_params(
            shard_hints={"0.weight": ParamShardHint(split_granularity="row_block", block_shape=(2, 4))}
        )

        evaluations = evaluate_auto_group_planners(params, world_size=2, target_block_units=8)

        self.assertIn("hinted_mixed", tuple(evaluation.name for evaluation in evaluations))
        hinted = next(evaluation for evaluation in evaluations if evaluation.name == "hinted_mixed")
        self.assertEqual(hinted.report.blocks_by_kind, {"matrix_row_block": 3, "parameter": 1})

    def test_auto_planner_policy_can_be_customized(self):
        params = self._matrix_params()
        policy = AutoPlannerPolicy(
            name="custom_no_split",
            weights=PlannerCostWeights(split_param=20.0),
        )

        evaluation = select_auto_group_plan(params, world_size=2, policy=policy, target_block_units=8)

        self.assertEqual(evaluation.name, "whole_param")
        self.assertEqual(evaluation.policy, "custom_no_split")
        self.assertEqual(evaluation.cost_weights, policy.weights)

    def test_auto_planner_accepts_candidate_objects(self):
        params = self._matrix_params()
        candidates = (
            PlannerCandidate(
                name="whole_param",
                planner=load_balanced_matrix_owner_tail_plan,
            ),
            PlannerCandidate(
                name="matrix_owner",
                planner=ordered_matrix_owner_tail_plan,
                blocks=(),
                padding_units=100,
            ),
        )

        evaluations = evaluate_auto_group_planners(params, world_size=2, candidates=candidates)

        self.assertEqual(tuple(evaluation.name for evaluation in evaluations), ("whole_param", "matrix_owner"))
        self.assertEqual(evaluations[1].padding_units, 100)

    def test_auto_planner_rejects_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "Unknown auto planner policy"):
            resolve_auto_planner_policy("unknown")

    def test_lists_available_auto_planner_policies(self):
        self.assertEqual(
            available_auto_planner_policies(),
            (
                "balanced",
                "debug",
                "max_balance",
                "min_comm",
                "muon_full_matrix",
                "muon_shard_aware",
                "muon_zero_copy",
                "zero_copy_friendly",
            ),
        )

    def test_zero_copy_friendly_policy_prefers_matrix_shard_over_flat_reorder(self):
        params = self._matrix_params()
        candidates = (
            PlannerCandidate(
                name="flat_reorder_owner",
                planner=lambda planner_params, planner_world_size: self._flat_reorder_owner_layout(planner_params),
            ),
            PlannerCandidate(
                name="rank_ordered_matrix",
                planner=parameter_boundary_plan,
                padding_units=64,
            ),
        )

        balanced = select_auto_group_plan(params, world_size=2, candidates=candidates, policy="balanced")
        zero_copy = select_auto_group_plan(params, world_size=2, candidates=candidates, policy="zero_copy_friendly")

        self.assertEqual(balanced.name, "flat_reorder_owner")
        self.assertEqual(balanced.runtime_mode, "flat_reorder")
        self.assertEqual(zero_copy.name, "rank_ordered_matrix")
        self.assertEqual(zero_copy.runtime_mode, "matrix_shard")

    def test_runtime_contracts_are_explicit_for_supported_planner_granularities(self):
        whole_params = self._matrix_params()
        whole_layout = parameter_boundary_plan(whole_params, world_size=2)
        whole_contract = explain_runtime_layout_compatibility(whole_layout, whole_params, world_size=2)

        self.assertEqual(whole_contract.mode, "matrix_shard")
        self.assertFalse(whole_contract.requires_flat_reorder)
        self.assertTrue(all(len(param_layout.segments) == 1 for param_layout in whole_layout.params))

        row_block_params = self._matrix_params(
            shard_hints={"0.weight": ParamShardHint(split_granularity="row_block", block_shape=(3, 4))}
        )
        row_block_layout = hinted_ordered_group_plan(row_block_params, world_size=2, default_granularity="parameter")
        row_block_contract = explain_runtime_layout_compatibility(
            row_block_layout,
            row_block_params,
            world_size=2,
        )

        self.assertEqual(row_block_contract.mode, "matrix_shard")
        self.assertFalse(row_block_contract.requires_flat_reorder)
        self.assertEqual(row_block_layout.owner_ranks("0.weight"), (0, 1))
        self.assertEqual(row_block_layout.owner_ranks("1.weight"), (1,))

        matrix_params = self._split_qkv_like_params(shard_hints=True)
        matrix_owner_layout = ordered_matrix_owner_tail_plan(matrix_params, world_size=8)
        matrix_owner_contract = explain_runtime_layout_compatibility(
            matrix_owner_layout,
            matrix_params,
            world_size=8,
        )

        self.assertEqual(matrix_owner_contract.mode, "flat_reorder")
        self.assertTrue(matrix_owner_contract.requires_flat_reorder)
        self.assertEqual(matrix_owner_layout.owner_ranks("q.weight"), (0,))
        self.assertEqual(matrix_owner_layout.owner_ranks("mlp.down.weight"), (6,))
        for managed_param, param_layout in zip(matrix_params, matrix_owner_layout.params):
            if len(managed_param.shape) == 2:
                self.assertEqual(len(param_layout.segments), 1)
                self.assertEqual(param_layout.segments[0].numel, managed_param.numel)

    def test_auto_group_plan_returns_selected_evaluation(self):
        params = self._matrix_params()

        evaluation = auto_group_plan(params, world_size=2, target_block_units=8)
        layout = evaluation.layout

        self.assertEqual(evaluation.name, "matrix_row_block")
        self.assertEqual(evaluation.policy, "balanced")
        self.assertEqual(layout.shard_sizes, (16, 16))
        self.assertEqual(layout.params_for_rank(0), ("0.weight",))
        self.assertEqual(layout.params_for_rank(1), ("0.weight", "1.weight"))

    def test_muon_shard_aware_policy_is_accepted_by_fully_shard_api(self):
        model = nn.Sequential(nn.Linear(4, 8, bias=False), nn.ReLU(), nn.Linear(8, 2, bias=False))

        sharded_model = matrix_fully_shard(model, auto_planner_policy="muon_shard_aware")
        unit = sharded_model._matrix_fsdp_param_group

        self.assertIsNotNone(unit.planner_evaluation)
        self.assertEqual(unit.planner_evaluation.policy, "muon_shard_aware")
        self.assertIs(unit.planner_result, unit.planner_evaluation)
        self.assertEqual(unit.planner_result.policy, "muon_shard_aware")
        self.assertIs(unit.planner_evaluation.layout, unit.group_layout)

    def test_muon_zero_copy_policy_is_accepted_by_fully_shard_api(self):
        model = nn.Sequential(nn.Linear(4, 8, bias=False), nn.ReLU(), nn.Linear(8, 2, bias=False))

        sharded_model = matrix_fully_shard(
            model,
            auto_shard_hints=True,
            auto_planner_policy="muon_zero_copy",
        )
        unit = sharded_model._matrix_fsdp_param_group

        self.assertIsNotNone(unit.planner_evaluation)
        self.assertEqual(unit.planner_evaluation.policy, "muon_zero_copy")
        self.assertEqual(unit.planner_evaluation.runtime_mode, "matrix_shard")
        self.assertEqual(unit.param_registry.param("0.weight").shard_hint.split_granularity, "matrix_owner")

    def test_ordered_matrix_owner_tail_plan_maps_matrices_and_small_params(self):
        params = self._split_qkv_like_params()

        layout = ordered_matrix_owner_tail_plan(params, world_size=8)

        self.assertEqual(layout.owner_ranks("q.weight"), (0,))
        self.assertEqual(layout.owner_ranks("k.weight"), (1,))
        self.assertEqual(layout.owner_ranks("v.weight"), (2,))
        self.assertEqual(layout.owner_ranks("proj.weight"), (3,))
        self.assertEqual(layout.owner_ranks("mlp.up0.weight"), (4,))
        self.assertEqual(layout.owner_ranks("mlp.up1.weight"), (5,))
        self.assertEqual(layout.owner_ranks("mlp.down.weight"), (6,))
        self.assertEqual(layout.owner_ranks("norm1.weight"), (7,))
        self.assertEqual(layout.owner_ranks("norm1.bias"), (7,))
        self.assertEqual(layout.owner_ranks("norm2.weight"), (7,))
        self.assertEqual(layout.owner_ranks("norm2.bias"), (7,))

    def test_load_balanced_matrix_owner_tail_plan_assigns_roles_to_lightest_ranks(self):
        params = self._split_qkv_like_params()

        layout = load_balanced_matrix_owner_tail_plan(
            params,
            world_size=8,
            initial_rank_units=(0, 0, 0, 1024, 1024, 0, 0, 0),
        )

        self.assertEqual(layout.owner_ranks("mlp.up0.weight"), (0,))
        self.assertEqual(layout.owner_ranks("mlp.up1.weight"), (1,))
        self.assertEqual(layout.owner_ranks("mlp.down.weight"), (2,))
        self.assertEqual(layout.owner_ranks("q.weight"), (5,))
        self.assertEqual(layout.owner_ranks("norm1.weight"), (6,))
        self.assertEqual(layout.shard_sizes, (512, 512, 512, 0, 0, 512, 320, 256))
        self.assertEqual(tuple(segment.numel for segment in layout.ranks[6].segments), (256, 16, 16, 16, 16))

    def test_matrix_owner_tail_merges_adjacent_rank_segments_without_merging_param_owners(self):
        params = self._split_qkv_like_params()

        layout = load_balanced_matrix_owner_tail_plan(params, world_size=4, merge_adjacent_rank_segments=True)

        self.assertEqual(layout.owner_ranks("q.weight"), (3,))
        self.assertEqual(layout.owner_ranks("k.weight"), (3,))
        self.assertEqual(layout.owner_ranks("norm1.weight"), (2,))
        self.assertEqual(tuple(segment.numel for segment in layout.ranks[3].segments), (512,))
        self.assertEqual(tuple(segment.numel for segment in layout.ranks[2].segments), (576,))
        self.assertEqual(layout.rank_segments_for_param(3, "q.weight")[0].numel, 256)
        self.assertEqual(layout.rank_segments_for_param(3, "k.weight")[0].numel, 256)
        self.assertEqual(layout.rank_segments_for_param(2, "norm1.weight")[0].numel, 16)

    def test_muon_shard_aware_group_planner_greedily_balances_matrix_owners(self):
        params = self._split_qkv_like_params()
        planner = make_muon_shard_aware_group_planner()

        first = planner(params, 8).layout
        second = planner(params, 8).layout

        self.assertEqual(first.owner_ranks("q.weight"), (0,))
        self.assertEqual(second.owner_ranks("q.weight"), (3,))
        self.assertEqual(first.owner_ranks("mlp.down.weight"), (6,))
        self.assertEqual(second.owner_ranks("mlp.down.weight"), (1,))
        self.assertEqual(second.owner_ranks("norm1.weight"), (2,))

    def test_muon_shard_aware_group_planner_can_use_round_robin_rotation(self):
        params = self._split_qkv_like_params()
        planner = make_muon_shard_aware_group_planner(rotation_strategy="round_robin")

        first = planner(params, 8).layout
        second = planner(params, 8).layout

        self.assertEqual(first.owner_ranks("q.weight"), (0,))
        self.assertEqual(second.owner_ranks("q.weight"), (1,))
        self.assertEqual(second.owner_ranks("mlp.down.weight"), (7,))

    def test_muon_shard_aware_group_planner_can_assign_owner_roles_greedily(self):
        params = self._split_qkv_like_params()
        planner = make_muon_shard_aware_group_planner(owner_assignment="role_greedy")

        evaluations = [planner(params, 8) for _ in range(4)]
        layouts = [evaluation.layout for evaluation in evaluations]
        rank_totals = tuple(sum(layout.shard_sizes[rank] for layout in layouts) for rank in range(8))

        self.assertTrue(all(evaluation.name == "matrix_owner_tail_role_greedy" for evaluation in evaluations))
        self.assertEqual(layouts[0].owner_ranks("mlp.up0.weight"), (0,))
        self.assertEqual(layouts[0].owner_ranks("q.weight"), (3,))
        self.assertEqual(layouts[0].owner_ranks("norm1.weight"), (7,))
        self.assertEqual(layouts[1].owner_ranks("mlp.up0.weight"), (7,))
        self.assertEqual(layouts[1].owner_ranks("q.weight"), (5,))
        self.assertEqual(rank_totals, (1536, 1280, 1152, 1280, 1344, 1280, 1280, 1344))

    def test_muon_shard_aware_group_planner_rejects_unknown_rotation_strategy(self):
        with self.assertRaisesRegex(ValueError, "Unknown rotation_strategy"):
            make_muon_shard_aware_group_planner(rotation_strategy="unknown")

    def test_muon_shard_aware_group_planner_rejects_unknown_owner_assignment(self):
        with self.assertRaisesRegex(ValueError, "Unknown owner_assignment"):
            make_muon_shard_aware_group_planner(owner_assignment="unknown")

    def test_single_rank_auto_group_plan_matches_eager_model(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(model)

        sharded_model = matrix_fully_shard(model, group_planner=auto_group_plan)
        unit = sharded_model._matrix_fsdp_param_group
        self.assertIsNotNone(unit.planner_evaluation)
        self.assertEqual(unit.planner_evaluation.name, "matrix_row_block")
        self.assertEqual(unit.planner_evaluation.policy, "balanced")
        self.assertIs(unit.planner_result, unit.planner_evaluation)
        self.assertEqual(unit.planner_result.planner_name, "matrix_row_block")
        self.assertIs(unit.planner_evaluation.layout, unit.group_layout)
        self.assertEqual(unit.planner_evaluation.report.rank_units, unit.group_layout.shard_sizes)
        self.assertIn("planner_evaluation", unit.state_dict())
        self.assertIn("planner_result", unit.state_dict())

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()
        sharded_optim.step()
        sharded_model._matrix_fsdp_param_group.unshard()

        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_single_rank_auto_group_plan_with_hints_matches_eager_model(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8, bias=False), nn.ReLU(), nn.Linear(8, 2, bias=False))
        eager_model = copy.deepcopy(model)
        shard_hints = {
            "0.weight": ParamShardHint(split_granularity="row_block", block_shape=(2, 4)),
            "2.weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
        }

        sharded_model = matrix_fully_shard(model, group_planner=auto_group_plan, shard_hints=shard_hints)
        unit = sharded_model._matrix_fsdp_param_group
        self.assertIsNotNone(unit.planner_evaluation)
        self.assertIn(unit.planner_evaluation.name, {"hinted_mixed", "matrix_row_block"})
        self.assertIn("matrix_row_block", unit.planner_evaluation.report.blocks_by_kind)
        self.assertEqual(unit.param_registry.param("2.weight").shard_hint.split_granularity, "matrix_owner")

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()
        sharded_optim.step()
        sharded_model._matrix_fsdp_param_group.unshard()

        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def _matrix_params(self, shard_hints=None):
        module = nn.Sequential(
            nn.Linear(4, 6, bias=False),
            nn.Linear(4, 2, bias=False),
        )
        return ManagedParamRegistry.from_module(module, shard_hints=shard_hints).params

    def _flat_reorder_owner_layout(self, params):
        first, second = params
        return MatrixGroupLayout.from_rank_segments(
            total_numel=first.numel + second.numel,
            rank_segments=(
                (LayoutSegment(second.offset, second.end, 0),),
                (LayoutSegment(first.offset, first.end, 0),),
            ),
            params=(
                ParamLayout(
                    fqn=first.fqn,
                    global_start=first.offset,
                    global_end=first.end,
                    segments=(
                        ParamSegment(
                            fqn=first.fqn,
                            rank=1,
                            global_start=first.offset,
                            global_end=first.end,
                            local_start=0,
                        ),
                    ),
                ),
                ParamLayout(
                    fqn=second.fqn,
                    global_start=second.offset,
                    global_end=second.end,
                    segments=(
                        ParamSegment(
                            fqn=second.fqn,
                            rank=0,
                            global_start=second.offset,
                            global_end=second.end,
                            local_start=0,
                        ),
                    ),
                ),
            ),
        )

    def _split_qkv_like_params(self, *, shard_hints: bool = False):
        module = _SplitQKVLikeBlock(hidden=16, intermediate=64)
        hints = None
        if shard_hints:
            from matrix_fsdp.shard_hint import build_shard_hints

            hints = build_shard_hints(module)
        return ManagedParamRegistry.from_module(module, shard_hints=hints).params


class _SplitQKVLikeMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        half_intermediate = intermediate // 2
        self.up0 = nn.Linear(hidden, half_intermediate, bias=False)
        self.up1 = nn.Linear(hidden, half_intermediate, bias=False)
        self.down = nn.Linear(half_intermediate, hidden, bias=False)


class _SplitQKVLikeBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = _SplitQKVLikeMLP(hidden, intermediate)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)


class _TinyExpertForPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(4, 8, bias=False)
        self.w2 = nn.Linear(8, 4, bias=False)


class _TinyMoEBlockForPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(4, 2, bias=False)
        self.experts = nn.ModuleList([_TinyExpertForPlanner(), _TinyExpertForPlanner()])


class _TinyMoEForPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe = _TinyMoEBlockForPlanner()


if __name__ == "__main__":
    unittest.main()
