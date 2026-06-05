import unittest

from torch import nn

from matrix_fsdp import (
    ParamRuntimeKind,
    ParamShardHint,
    ShardConstraint,
    constraint_counts,
    infer_group_constraints,
)
from matrix_fsdp.managed_param import ManagedParamRegistry
from matrix_fsdp.planner import parameter_boundary_plan
from matrix_fsdp.planner_eval import evaluate_group_planner


class ShardConstraintTest(unittest.TestCase):
    def test_infers_constraints_from_param_shard_hints(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8))
        params = ManagedParamRegistry.from_module(
            model,
            shard_hints={
                "0.weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
                "0.bias": ParamShardHint(split_granularity="parameter"),
                "1.weight": ParamShardHint(split_granularity="row_block", block_shape=(1, 8)),
                "1.bias": ParamShardHint(
                    split_granularity="block",
                    block_shape=(4,),
                    runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
                    parallel_role="routed_expert",
                    expert_group_id="expert.0",
                ),
            },
        ).params

        constraints = infer_group_constraints(params)
        by_fqn = {param_constraints.fqn: param_constraints for param_constraints in constraints}

        self.assertEqual(
            by_fqn["0.weight"].constraints,
            (
                ShardConstraint.NO_SPLIT_PARAM,
                ShardConstraint.NO_SPLIT_MATRIX,
                ShardConstraint.WHOLE_PARAM_OWNER,
                ShardConstraint.MUON_MATRIX_OWNER,
            ),
        )
        self.assertEqual(by_fqn["0.weight"].optimizer_type, "muon")
        self.assertEqual(
            by_fqn["0.bias"].constraints,
            (ShardConstraint.NO_SPLIT_PARAM, ShardConstraint.WHOLE_PARAM_OWNER),
        )
        self.assertEqual(by_fqn["1.weight"].constraints, (ShardConstraint.ROW_BLOCK_ALIGNED,))
        self.assertEqual(by_fqn["1.weight"].block_shape, (1, 8))
        self.assertEqual(by_fqn["1.bias"].constraints, (ShardConstraint.BLOCK_ALIGNED, ShardConstraint.EXPERT_OWNER))
        self.assertEqual(by_fqn["1.bias"].runtime_kind, ParamRuntimeKind.EXPERT_OWNER)
        self.assertEqual(by_fqn["1.bias"].parallel_role, "routed_expert")
        self.assertEqual(by_fqn["1.bias"].expert_group_id, "expert.0")

        counts = constraint_counts(constraints)
        self.assertEqual(counts[ShardConstraint.NO_SPLIT_PARAM], 2)
        self.assertEqual(counts[ShardConstraint.WHOLE_PARAM_OWNER], 2)

    def test_planner_result_metadata_includes_constraint_summary(self):
        model = nn.Linear(4, 8)
        params = ManagedParamRegistry.from_module(
            model,
            shard_hints={"weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner")},
        ).params

        result = evaluate_group_planner("whole_param", params, 2, parameter_boundary_plan)
        metadata = result.as_metadata()

        self.assertTrue(metadata["constraints_satisfied"])
        self.assertIn(("muon_matrix_owner", 1), metadata["constraint_counts"])
        self.assertEqual(metadata["constraints"][0]["fqn"], "weight")
        self.assertIn("no_split_matrix", metadata["constraints"][0]["constraints"])
        self.assertIn("runtime_kind", metadata["constraints"][0])


if __name__ == "__main__":
    unittest.main()
