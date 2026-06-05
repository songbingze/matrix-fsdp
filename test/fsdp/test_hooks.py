import unittest

import torch

from matrix_fsdp.hooks import (
    ForwardBackwardContext,
    register_pre_backward_hooks,
    register_pre_backward_hooks_with_context,
)


class HookTest(unittest.TestCase):
    def test_registers_hook_on_tensor_output(self):
        x = torch.randn(3, requires_grad=True)
        calls = 0

        def callback():
            nonlocal calls
            calls += 1

        y = x.square()
        registered = register_pre_backward_hooks(y, callback)
        self.assertEqual(registered, 1)

        y.sum().backward()
        self.assertEqual(calls, 1)

    def test_skips_non_differentiable_tensor_output(self):
        calls = 0

        def callback():
            nonlocal calls
            calls += 1

        registered = register_pre_backward_hooks(torch.randn(3), callback)

        self.assertEqual(registered, 0)
        self.assertEqual(calls, 0)

    def test_registers_hooks_on_nested_outputs(self):
        x = torch.randn(3, requires_grad=True)
        y = x * 2
        z = x.square()
        calls = 0

        def callback():
            nonlocal calls
            calls += 1

        aux = z.mean()
        output = {
            "main": y,
            "aux": [aux, (torch.randn(2), "ignored")],
        }

        registered = register_pre_backward_hooks(output, callback)
        self.assertEqual(registered, 2)

        (y.sum() + aux).backward()
        self.assertEqual(calls, 1)

    def test_callback_can_be_idempotent_for_multiple_outputs(self):
        x = torch.randn(3, requires_grad=True)
        calls = 0

        def callback():
            nonlocal calls
            calls += 1

        y = x * 2
        z = x * 3
        registered = register_pre_backward_hooks((y, z), callback)
        self.assertEqual(registered, 2)

        (y.sum() + z.sum()).backward()
        self.assertEqual(calls, 1)

    def test_forward_backward_context_tracks_pending_state(self):
        x = torch.randn(3, requires_grad=True)
        calls = 0

        def callback():
            nonlocal calls
            calls += 1

        context = ForwardBackwardContext(callback)
        y = x * 2
        z = x * 3
        registered = register_pre_backward_hooks_with_context((y, z), context)
        context.mark_registered(registered)

        self.assertEqual(registered, 2)
        self.assertTrue(context.pending_backward)
        (y.sum() + z.sum()).backward()
        self.assertEqual(calls, 1)
        self.assertTrue(context.pre_backward_called)
        self.assertFalse(context.pending_backward)

    def test_empty_forward_backward_context_is_not_pending(self):
        calls = 0

        def callback():
            nonlocal calls
            calls += 1

        context = ForwardBackwardContext(callback)
        registered = register_pre_backward_hooks_with_context(torch.randn(3), context)
        context.mark_registered(registered)

        self.assertEqual(registered, 0)
        self.assertFalse(context.pending_backward)
        self.assertEqual(calls, 0)

    def test_ignores_non_container_outputs(self):
        calls = 0

        def callback():
            nonlocal calls
            calls += 1

        registered = register_pre_backward_hooks({"loss": 1.0, "tag": "ok"}, callback)

        self.assertEqual(registered, 0)
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
