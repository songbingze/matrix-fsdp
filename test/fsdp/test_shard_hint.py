import unittest

from torch import nn

from matrix_fsdp import ParamRuntimeKind, ParamShardHint, build_shard_hints, matrix_fully_shard
from matrix_fsdp.shard_hint import ShardHintRule, default_shard_hint_rules


class ShardHintTest(unittest.TestCase):
    def test_build_shard_hints_applies_default_rules(self):
        model = nn.ModuleDict(
            {
                "embed": nn.Embedding(8, 4),
                "linear": nn.Linear(4, 6),
                "norm": nn.LayerNorm(6),
            }
        )

        hints = build_shard_hints(model)

        self.assertEqual(
            hints["linear.weight"],
            ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
        )
        self.assertEqual(
            hints["linear.bias"],
            ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
        )
        self.assertEqual(
            hints["embed.weight"],
            ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
        )
        self.assertEqual(
            hints["norm.weight"],
            ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
        )
        self.assertEqual(
            hints["norm.bias"],
            ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
        )

    def test_build_shard_hints_accepts_overrides(self):
        model = nn.Sequential(nn.Linear(4, 6))
        override = ParamShardHint(optimizer_type="adamw", split_granularity="row_block", block_shape=(2, 4))

        hints = build_shard_hints(model, overrides={"0.weight": override})

        self.assertEqual(hints["0.weight"], override)
        self.assertEqual(hints["0.bias"], ParamShardHint(optimizer_type="adamw", split_granularity="parameter"))

    def test_build_shard_hints_rejects_unknown_override(self):
        model = nn.Linear(4, 6)

        with self.assertRaisesRegex(ValueError, "unknown parameters"):
            build_shard_hints(model, overrides={"missing": ParamShardHint()})

    def test_build_shard_hints_accepts_custom_rules(self):
        model = nn.Sequential(nn.Linear(4, 6, bias=False))
        hint = ParamShardHint(optimizer_type="adamw", split_granularity="parameter")

        def custom_rule(fqn, owning_module, local_name, param):
            return hint if fqn == "0.weight" else None

        hints = build_shard_hints(model, rules=(ShardHintRule("custom", custom_rule),))

        self.assertEqual(hints, {"0.weight": hint})

    def test_default_shard_hint_rules_are_named(self):
        self.assertEqual(
            tuple(rule.name for rule in default_shard_hint_rules()),
            (
                "moe_expert_owner",
                "router",
                "muon_linear_weight",
                "no_split_embedding",
                "no_split_norm",
                "no_split_bias",
                "no_split_1d",
            ),
        )

    def test_build_shard_hints_marks_moe_experts_and_router(self):
        model = _TinyMoE()

        hints = build_shard_hints(model)

        self.assertEqual(
            hints["moe.experts.0.w1.weight"],
            ParamShardHint(
                optimizer_type="muon",
                split_granularity="matrix_owner",
                runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
                parallel_role="routed_expert",
                expert_id=0,
                expert_group_id="moe.experts.0",
            ),
        )
        self.assertEqual(
            hints["moe.experts.1.w2.weight"].expert_group_id,
            "moe.experts.1",
        )
        self.assertEqual(
            hints["moe.router.weight"],
            ParamShardHint(
                optimizer_type="adamw",
                split_granularity="parameter",
                runtime_kind=ParamRuntimeKind.FSDP_GATHER,
                parallel_role="router",
            ),
        )

    def test_matrix_fully_shard_accepts_resolved_hints(self):
        model = nn.Sequential(nn.Linear(4, 6), nn.LayerNorm(6), nn.Linear(6, 2))

        sharded_model = matrix_fully_shard(model, shard_hints=build_shard_hints(model))
        unit = sharded_model._matrix_fsdp_param_group

        self.assertEqual(
            unit.param_registry.param("0.weight").shard_hint,
            ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
        )
        self.assertEqual(
            unit.param_registry.param("1.weight").shard_hint,
            ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
        )


class _TinyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(4, 8, bias=False)
        self.w2 = nn.Linear(8, 4, bias=False)


class _TinyMoEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = nn.Linear(4, 2, bias=False)
        self.experts = nn.ModuleList([_TinyExpert(), _TinyExpert()])


class _TinyMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe = _TinyMoEBlock()


if __name__ == "__main__":
    unittest.main()
