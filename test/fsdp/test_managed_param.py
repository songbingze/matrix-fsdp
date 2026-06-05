import unittest

from torch import nn

from matrix_fsdp.managed_param import ManagedParamRegistry, ParamRuntimeKind, ParamShardHint


class ManagedParamRegistryTest(unittest.TestCase):
    def test_from_module_records_stable_flat_metadata(self):
        module = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2, bias=False))

        registry = ManagedParamRegistry.from_module(module)

        self.assertEqual(registry.fqns, ("0.weight", "0.bias", "1.weight"))
        self.assertEqual(registry.total_numel, 21)
        self.assertEqual([managed_param.offset for managed_param in registry], [0, 12, 15])
        self.assertEqual([managed_param.end for managed_param in registry], [12, 15, 21])
        self.assertIs(registry.param("0.weight").param, module[0].weight)
        self.assertEqual(registry.param("0.weight").shard_hint, ParamShardHint())
        with self.assertRaises(KeyError):
            registry.param("missing")

    def test_from_module_attaches_shard_hints_by_fqn(self):
        module = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2, bias=False))
        hint = ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner")

        registry = ManagedParamRegistry.from_module(module, shard_hints={"0.weight": hint})

        self.assertEqual(registry.param("0.weight").shard_hint, hint)
        self.assertEqual(registry.param("0.bias").shard_hint, ParamShardHint())

    def test_from_module_rejects_unknown_shard_hint_fqn(self):
        module = nn.Linear(4, 3)

        with self.assertRaisesRegex(ValueError, "unknown parameters"):
            ManagedParamRegistry.from_module(module, shard_hints={"missing": ParamShardHint()})

    def test_from_module_validates_shard_hint_shape_constraints(self):
        module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))

        with self.assertRaisesRegex(ValueError, "matrix_owner"):
            ManagedParamRegistry.from_module(
                module,
                shard_hints={"0.bias": ParamShardHint(split_granularity="matrix_owner")},
            )
        with self.assertRaisesRegex(ValueError, "second dim"):
            ManagedParamRegistry.from_module(
                module,
                shard_hints={"0.weight": ParamShardHint(split_granularity="row_block", block_shape=(2, 5))},
            )
        with self.assertRaisesRegex(ValueError, "not divisible"):
            ManagedParamRegistry.from_module(
                module,
                shard_hints={"0.weight": ParamShardHint(split_granularity="row_block", block_shape=(4, 4))},
            )
        with self.assertRaisesRegex(ValueError, "does not accept block_shape"):
            ManagedParamRegistry.from_module(
                module,
                shard_hints={"0.weight": ParamShardHint(split_granularity="matrix_owner", block_shape=(2, 4))},
            )
        with self.assertRaisesRegex(ValueError, "requires expert_group_id"):
            ManagedParamRegistry.from_module(
                module,
                shard_hints={
                    "0.weight": ParamShardHint(
                        split_granularity="matrix_owner",
                        runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
                    )
                },
            )
        with self.assertRaisesRegex(ValueError, "unknown parallel_role"):
            ManagedParamRegistry.from_module(
                module,
                shard_hints={"0.weight": ParamShardHint(parallel_role="unknown")},
            )


if __name__ == "__main__":
    unittest.main()
