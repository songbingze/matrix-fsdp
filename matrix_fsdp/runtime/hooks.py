from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

PreBackwardCallback = Callable[[], None]


@dataclass
class ForwardBackwardContext:
    callback: PreBackwardCallback
    registered_hooks: int = 0
    pre_backward_called: bool = False
    _hook: Callable[[torch.Tensor], torch.Tensor] | None = field(default=None, init=False, repr=False)

    @property
    def pending_backward(self) -> bool:
        return self.registered_hooks > 0 and not self.pre_backward_called

    def mark_registered(self, count: int) -> None:
        self.registered_hooks = count

    def pre_backward(self) -> None:
        if self.pre_backward_called:
            return
        self.pre_backward_called = True
        self.callback()

    def hook(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self._hook is None:
            self._hook = _make_pre_backward_hook(self)
        return self._hook


def register_pre_backward_hooks(output: object, callback: PreBackwardCallback) -> int:
    """
    Register ``callback`` on every differentiable tensor in a forward output.

    The callback is intentionally tensor-agnostic so the traversal can be tested
    independently from MatrixFSDPParamGroup state and buffer management.
    """
    context = ForwardBackwardContext(callback)
    count = register_pre_backward_hooks_with_context(output, context)
    context.mark_registered(count)
    return count


def register_pre_backward_hooks_with_context(output: object, context: ForwardBackwardContext) -> int:
    """
    Register a shared forward/backward context on all differentiable outputs.

    Multiple tensor hooks from the same forward all share the context, so the
    runtime transition is triggered only once.
    """
    if isinstance(output, torch.Tensor):
        if not output.requires_grad:
            return 0
        output.register_hook(context.hook())
        return 1
    if isinstance(output, Mapping):
        return sum(register_pre_backward_hooks_with_context(value, context) for value in output.values())
    if isinstance(output, tuple):
        return sum(register_pre_backward_hooks_with_context(value, context) for value in output)
    if isinstance(output, list):
        return sum(register_pre_backward_hooks_with_context(value, context) for value in output)
    return 0


def _make_pre_backward_hook(context: ForwardBackwardContext) -> Callable[[torch.Tensor], torch.Tensor]:
    def hook(grad: torch.Tensor) -> torch.Tensor:
        context.pre_backward()
        return grad

    return hook
