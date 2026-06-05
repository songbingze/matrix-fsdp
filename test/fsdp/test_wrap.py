import unittest

from torch import nn

from matrix_fsdp.wrap import and_policy, module_type_policy, not_policy, or_policy, size_based_policy


class WrapPolicyTest(unittest.TestCase):
    def test_module_type_policy_accepts_single_type(self):
        policy = module_type_policy(nn.Linear)

        self.assertTrue(policy(nn.Linear(4, 2)))
        self.assertFalse(policy(nn.ReLU()))

    def test_module_type_policy_accepts_multiple_types(self):
        policy = module_type_policy({nn.Linear, nn.LayerNorm})

        self.assertTrue(policy(nn.Linear(4, 2)))
        self.assertTrue(policy(nn.LayerNorm(4)))
        self.assertFalse(policy(nn.ReLU()))

    def test_module_type_policy_rejects_empty_or_invalid_types(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            module_type_policy(set())
        with self.assertRaisesRegex(TypeError, "nn.Module types"):
            module_type_policy({nn.Linear, object})

    def test_size_based_policy_uses_non_recursive_params_by_default(self):
        module = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        policy = size_based_policy(1)

        self.assertFalse(policy(module))
        self.assertTrue(policy(module[0]))

    def test_size_based_policy_can_count_recursive_params(self):
        module = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        total_numel = sum(param.numel() for param in module.parameters())

        self.assertTrue(size_based_policy(total_numel, recurse=True)(module))
        self.assertFalse(size_based_policy(total_numel + 1, recurse=True)(module))

    def test_size_based_policy_rejects_negative_threshold(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            size_based_policy(-1)

    def test_policy_combinators(self):
        linear_policy = module_type_policy(nn.Linear)
        large_policy = size_based_policy(32)
        small_policy = not_policy(large_policy)

        self.assertTrue(or_policy(linear_policy, module_type_policy(nn.LayerNorm))(nn.Linear(4, 2)))
        self.assertFalse(or_policy(module_type_policy(nn.LayerNorm), module_type_policy(nn.ReLU))(nn.Linear(4, 2)))
        self.assertTrue(and_policy(linear_policy, large_policy)(nn.Linear(4, 8)))
        self.assertFalse(and_policy(linear_policy, large_policy)(nn.Linear(2, 2)))
        self.assertTrue(small_policy(nn.Linear(2, 2)))

    def test_policy_combinators_reject_empty_or_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            or_policy()
        with self.assertRaisesRegex(ValueError, "at least one"):
            and_policy()
        with self.assertRaisesRegex(TypeError, "callable"):
            or_policy(module_type_policy(nn.Linear), object())
        with self.assertRaisesRegex(TypeError, "callable"):
            not_policy(object())


if __name__ == "__main__":
    unittest.main()
