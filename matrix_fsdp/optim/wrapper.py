from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from matrix_fsdp.core.managed_param import ManagedParam
from matrix_fsdp.optim.adamw_state import initialize_flat_adamw_state
from matrix_fsdp.runtime.param_group import FSDPLifecycleState, MatrixFSDPNoSync, MatrixFSDPParamGroup
from matrix_fsdp.optim.state import MatrixFSDPOptimizerStateManager
from matrix_fsdp.runtime.scheduler import (
    DEFAULT_MAX_UNSHARDED_PREFETCH_UNITS,
    BackwardPrefetchTiming,
    PrefetchPolicy,
    MatrixFSDPScheduler,
    MatrixFSDPSchedulerConfig,
)
from matrix_fsdp.core.state import MatrixShardedState
from matrix_fsdp.runtime.unit_collection import collect_param_groups

DEFAULT_MUON_ADJUST_LR_FN = "match_rms_adamw"
_ORIGINAL_OPTIMIZER_INIT: Any | None = None
_MATRIX_OPTIMIZER_AUTO_PREPARE_INSTALLED = False
_SCHEDULER_KWARG_NAMES = frozenset(
    {
        "max_unsharded_prefetch_units",
        "max_forward_prefetch_units",
        "max_backward_prefetch_units",
        "backward_prefetch_timing",
        "prefetch_policy",
        "max_cached_full_param_buffers_per_key",
        "max_active_full_param_buffers",
        "max_active_full_param_numel",
        "max_active_full_param_memory_mb",
        "max_pending_backward_reduces",
        "trim_cuda_cache",
        "cuda_cache_trim_threshold_mb",
        "scheduler_config",
        "flat_adamw_state",
    }
)


@dataclass(frozen=True)
class MatrixOptimizerParamGroupSummary:
    name: str
    optimizer_type: str
    fqns: tuple[str, ...]
    param_numel: int
    local_numel: int

    @property
    def num_params(self) -> int:
        return len(self.fqns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "optimizer_type": self.optimizer_type,
            "fqns": self.fqns,
            "num_params": self.num_params,
            "param_numel": self.param_numel,
            "local_numel": self.local_numel,
        }


@dataclass(frozen=True)
class ClassifiedMatrixOptimizerParams:
    muon_params: tuple[nn.Parameter, ...]
    adamw_params: tuple[nn.Parameter, ...]
    group_summaries: tuple[MatrixOptimizerParamGroupSummary, ...]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "muon_param_count": len(self.muon_params),
            "adamw_param_count": len(self.adamw_params),
            "groups": tuple(group.as_dict() for group in self.group_summaries),
        }


@dataclass(frozen=True)
class MatrixOptimizerConfig:
    optimizer: str = "adamw"
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    default_matrix_optimizer: str = "muon"
    default_other_optimizer: str = "adamw"


class PreparedMatrixOptimizer:
    """
    Attach MatrixFSDP lifecycle hooks to a regular torch optimizer.

    The wrapped optimizer object remains the user-facing optimizer. This helper
    owns the hook handles and exposes the same metadata utilities that the
    heavier ``MatrixFSDPOptimizer`` wrapper exposes.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
        *,
        max_unsharded_prefetch_units: int | None = None,
        max_forward_prefetch_units: int | None = None,
        max_backward_prefetch_units: int | None = None,
        backward_prefetch_timing: BackwardPrefetchTiming = "pre_backward",
        prefetch_policy: PrefetchPolicy = "static",
        max_cached_full_param_buffers_per_key: int = 0,
        max_active_full_param_buffers: int | None = None,
        max_active_full_param_numel: int | None = None,
        max_active_full_param_memory_mb: float | None = None,
        max_pending_backward_reduces: int | None = 1,
        trim_cuda_cache: bool = False,
        cuda_cache_trim_threshold_mb: float = 1024.0,
        flat_adamw_state: bool = False,
        scheduler_config: MatrixFSDPSchedulerConfig | None = None,
    ) -> None:
        self.optimizer = optimizer
        self._fsdp_param_groups = _normalize_param_groups(module_or_units)
        self.flat_adamw_state = flat_adamw_state
        scheduler_kwargs = _resolve_scheduler_kwargs(
            scheduler_config=scheduler_config,
            max_unsharded_prefetch_units=max_unsharded_prefetch_units,
            max_forward_prefetch_units=max_forward_prefetch_units,
            max_backward_prefetch_units=max_backward_prefetch_units,
            backward_prefetch_timing=backward_prefetch_timing,
            prefetch_policy=prefetch_policy,
            max_cached_full_param_buffers_per_key=max_cached_full_param_buffers_per_key,
            max_active_full_param_buffers=max_active_full_param_buffers,
            max_active_full_param_numel=max_active_full_param_numel,
            max_active_full_param_memory_mb=max_active_full_param_memory_mb,
            max_pending_backward_reduces=max_pending_backward_reduces,
            trim_cuda_cache=trim_cuda_cache,
            cuda_cache_trim_threshold_mb=cuda_cache_trim_threshold_mb,
        )
        self.scheduler = MatrixFSDPScheduler(self.fsdp_param_groups, **scheduler_kwargs)
        self.state_manager = MatrixFSDPOptimizerStateManager(self.optimizer, self.fsdp_param_groups)
        self._original_zero_grad = optimizer.zero_grad
        self._removed = False
        self._step_pre_handle = optimizer.register_step_pre_hook(self._step_pre_hook)
        self._step_post_handle = optimizer.register_step_post_hook(self._step_post_hook)
        optimizer.zero_grad = MethodType(_prepared_optimizer_zero_grad, optimizer)  # type: ignore[method-assign]
        optimizer.matrix_fsdp = self  # type: ignore[attr-defined]
        optimizer.state_manager = self.state_manager  # type: ignore[attr-defined]
        optimizer.matrix_fsdp_scheduler = self.scheduler  # type: ignore[attr-defined]
        optimizer.no_sync = self.no_sync  # type: ignore[attr-defined]
        optimizer.refresh_state_dtensors = self.refresh_state_dtensors  # type: ignore[attr-defined]
        optimizer.validate_local_state_shapes = self.validate_local_state_shapes  # type: ignore[attr-defined]

    @property
    def runtime_param_groups(self) -> list[MatrixFSDPParamGroup]:
        return self.fsdp_param_groups

    @property
    def fsdp_param_groups(self) -> list[MatrixFSDPParamGroup]:
        return self._fsdp_param_groups

    def no_sync(self) -> MatrixFSDPNoSync:
        return MatrixFSDPNoSync(self.fsdp_param_groups)

    def local_state_summary(self) -> dict[str, Any]:
        return self.state_manager.local_state_summary()

    def local_state_dtensors(self) -> dict[str, dict[str, Any]]:
        return self.state_manager.local_state_dtensors()

    def local_state_objects(self) -> dict[str, dict[str, MatrixShardedState]]:
        return self.state_manager.local_state_objects()

    @property
    def state_dtensors(self) -> dict[str, dict[str, Any]]:
        return self.refresh_state_dtensors()

    @property
    def state_objects(self) -> dict[str, dict[str, MatrixShardedState]]:
        if not self.state_manager.state_objects:
            self.refresh_state_dtensors()
        return self.state_manager.state_objects

    def refresh_state_dtensors(self) -> dict[str, dict[str, Any]]:
        return self.state_manager.refresh_state_dtensors()

    def validate_local_state_shapes(self) -> None:
        self.state_manager.validate_local_state_shapes()

    def remove(self) -> None:
        if self._removed:
            return
        self._step_pre_handle.remove()
        self._step_post_handle.remove()
        self.optimizer.zero_grad = self._original_zero_grad  # type: ignore[method-assign]
        for attr in (
            "matrix_fsdp",
            "state_manager",
            "matrix_fsdp_scheduler",
            "no_sync",
            "refresh_state_dtensors",
            "validate_local_state_shapes",
        ):
            try:
                delattr(self.optimizer, attr)
            except AttributeError:
                pass
        self._removed = True

    def _step_pre_hook(self, optimizer: Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]):
        if _optimizer_step_has_closure(args, kwargs):
            raise NotImplementedError("MatrixFSDP prepared optimizers do not support optimizer closures yet.")
        _raise_if_no_sync_active(self.fsdp_param_groups, "optimizer.step")
        _prepare_matrix_optimizer_step(self.fsdp_param_groups, self.scheduler)
        _initialize_matrix_optimizer_state(
            self.optimizer,
            self.fsdp_param_groups,
            flat_adamw_state=self.flat_adamw_state,
        )
        return None

    def _step_post_hook(self, optimizer: Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.scheduler.maybe_trim_cuda_cache()

    def _zero_grad(self, *args: Any, **kwargs: Any) -> None:
        _raise_if_no_sync_active(self.fsdp_param_groups, "optimizer.zero_grad")
        _prepare_matrix_optimizer_zero_grad(self.fsdp_param_groups, self.scheduler)
        self._original_zero_grad(*args, **kwargs)
        _finish_matrix_optimizer_zero_grad(self.fsdp_param_groups, self.scheduler)


def prepare_matrix_optimizer(
    optimizer: Optimizer,
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    **kwargs: Any,
) -> PreparedMatrixOptimizer:
    """
    Register MatrixFSDP hooks on a regular torch optimizer.

    This keeps the user-facing object as a regular ``torch.optim`` optimizer
    while adding the runtime lifecycle pieces that ordinary optimizers do not
    know about: no-sync guards, pending backward reduce finalization, zero-grad
    local-shard cleanup, scheduler cleanup, CUDA cache trimming, and sharded
    optimizer-state metadata.
    """

    existing = getattr(optimizer, "matrix_fsdp", None)
    if isinstance(existing, PreparedMatrixOptimizer):
        existing.remove()
    return PreparedMatrixOptimizer(optimizer, module_or_units, **kwargs)


def install_matrix_optimizer_auto_prepare() -> None:
    """
    Install one process-local hook that auto-prepares torch optimizers.

    MatrixFSDP-managed parameters carry a weak reference to their runtime
    parameter group. Since users are expected to construct optimizers after
    ``fully_shard()``, the base ``torch.optim.Optimizer`` constructor can detect
    those parameters and attach the same lifecycle hooks that
    ``prepare_matrix_optimizer()`` would attach explicitly.
    """

    global _ORIGINAL_OPTIMIZER_INIT, _MATRIX_OPTIMIZER_AUTO_PREPARE_INSTALLED
    if _MATRIX_OPTIMIZER_AUTO_PREPARE_INSTALLED:
        return
    _ORIGINAL_OPTIMIZER_INIT = Optimizer.__init__

    def _matrix_optimizer_init(self: Optimizer, *args: Any, **kwargs: Any) -> None:
        assert _ORIGINAL_OPTIMIZER_INIT is not None
        _ORIGINAL_OPTIMIZER_INIT(self, *args, **kwargs)
        _auto_prepare_matrix_optimizer(self)

    Optimizer.__init__ = _matrix_optimizer_init  # type: ignore[method-assign]
    _MATRIX_OPTIMIZER_AUTO_PREPARE_INSTALLED = True


def _auto_prepare_matrix_optimizer(optimizer: Optimizer) -> None:
    if isinstance(getattr(optimizer, "matrix_fsdp", None), PreparedMatrixOptimizer):
        return
    fsdp_param_groups = _collect_optimizer_matrix_param_groups(optimizer)
    if fsdp_param_groups:
        prepare_matrix_optimizer(optimizer, fsdp_param_groups)


def _collect_optimizer_matrix_param_groups(optimizer: Optimizer) -> list[MatrixFSDPParamGroup]:
    fsdp_param_groups: list[MatrixFSDPParamGroup] = []
    seen: set[int] = set()
    for optim_group in optimizer.param_groups:
        for param in optim_group.get("params", ()):
            param_group_ref = getattr(param, "_matrix_fsdp_param_group_ref", None)
            if param_group_ref is None:
                continue
            param_group = param_group_ref()
            if param_group is None:
                continue
            param_group_id = id(param_group)
            if param_group_id in seen:
                continue
            seen.add(param_group_id)
            fsdp_param_groups.append(param_group)
    return fsdp_param_groups


def _remove_prepared_matrix_optimizer(optimizer: Any) -> None:
    prepared = getattr(optimizer, "matrix_fsdp", None)
    if isinstance(prepared, PreparedMatrixOptimizer):
        prepared.remove()


def _prepared_optimizer_zero_grad(optimizer: Optimizer, *args: Any, **kwargs: Any) -> None:
    prepared = getattr(optimizer, "matrix_fsdp", None)
    if not isinstance(prepared, PreparedMatrixOptimizer):
        raise RuntimeError("optimizer.zero_grad is patched for MatrixFSDP but no prepared state is attached.")
    prepared._zero_grad(*args, **kwargs)


class MixedMuonAdamWOptimizer:
    """
    Small mixed optimizer used by MatrixFSDP matrix-owner layouts.

    Muon is applied to local full 2D matrix-owner parameters; AdamW is applied
    to the remaining local parameters such as norm weights, biases, and partial
    shards. The object intentionally exposes the small optimizer surface needed
    by ``MatrixFSDPOptimizer`` while keeping the two inner torch optimizers
    separate.
    """

    def __init__(
        self,
        muon_params: Iterable[nn.Parameter],
        adamw_params: Iterable[nn.Parameter],
        *,
        group_summaries: Iterable[MatrixOptimizerParamGroupSummary] = (),
        lr: float = 0.001,
        weight_decay: float = 0.01,
        muon_lr: float | None = None,
        adamw_lr: float | None = None,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float | None = None,
        adamw_foreach: bool | None = None,
        muon_weight_decay: float | None = None,
        muon_momentum: float = 0.5,
        muon_ns_steps: int = 2,
        muon_adjust_lr_fn: str | None = DEFAULT_MUON_ADJUST_LR_FN,
        lazy_muon_init: bool = False,
        fsdp_param_groups: Iterable[MatrixFSDPParamGroup] | None = None,
        max_unsharded_prefetch_units: int | None = None,
        max_forward_prefetch_units: int | None = None,
        max_backward_prefetch_units: int | None = None,
        backward_prefetch_timing: BackwardPrefetchTiming = "pre_backward",
        prefetch_policy: PrefetchPolicy = "static",
        max_cached_full_param_buffers_per_key: int = 0,
        max_active_full_param_buffers: int | None = None,
        max_active_full_param_numel: int | None = None,
        max_active_full_param_memory_mb: float | None = None,
        max_pending_backward_reduces: int | None = 1,
        trim_cuda_cache: bool = False,
        cuda_cache_trim_threshold_mb: float = 1024.0,
        flat_adamw_state: bool = False,
        scheduler_config: MatrixFSDPSchedulerConfig | None = None,
    ) -> None:
        self.muon_params = tuple(param for param in muon_params if param.numel() > 0)
        self.adamw_params = tuple(param for param in adamw_params if param.numel() > 0)
        self.group_summaries = tuple(group_summaries)
        self.muon = None
        self.adamw = None
        self._fsdp_param_groups = list(fsdp_param_groups or ())
        self._matrix_lifecycle_enabled = bool(self._fsdp_param_groups)
        self.flat_adamw_state = flat_adamw_state
        self.scheduler = None
        self.state_manager = None
        resolved_muon_lr = lr if muon_lr is None else muon_lr
        resolved_adamw_lr = lr if adamw_lr is None else adamw_lr
        resolved_muon_weight_decay = weight_decay if muon_weight_decay is None else muon_weight_decay
        resolved_adamw_weight_decay = weight_decay if adamw_weight_decay is None else adamw_weight_decay
        if self.muon_params:
            if not hasattr(torch.optim, "Muon"):
                raise RuntimeError("torch.optim.Muon is not available in this PyTorch build.")
            muon_kwargs = {
                "lr": resolved_muon_lr,
                "momentum": muon_momentum,
                "ns_steps": muon_ns_steps,
                "weight_decay": resolved_muon_weight_decay,
                "adjust_lr_fn": muon_adjust_lr_fn,
            }
            self.muon = (
                _LazyMuonOptimizer(self.muon_params, **muon_kwargs)
                if lazy_muon_init
                else torch.optim.Muon(self.muon_params, **muon_kwargs)
            )
            _remove_prepared_matrix_optimizer(self.muon)
        if self.adamw_params:
            adamw_kwargs = {
                "lr": resolved_adamw_lr,
                "betas": adamw_betas,
                "eps": adamw_eps,
                "weight_decay": resolved_adamw_weight_decay,
            }
            if adamw_foreach is not None:
                adamw_kwargs["foreach"] = adamw_foreach
            self.adamw = torch.optim.AdamW(self.adamw_params, **adamw_kwargs)
            _remove_prepared_matrix_optimizer(self.adamw)
        if self.muon is None and self.adamw is None:
            raise RuntimeError("MixedMuonAdamWOptimizer requires at least one non-empty parameter.")
        if self._matrix_lifecycle_enabled:
            scheduler_kwargs = _resolve_scheduler_kwargs(
                scheduler_config=scheduler_config,
                max_unsharded_prefetch_units=max_unsharded_prefetch_units,
                max_forward_prefetch_units=max_forward_prefetch_units,
                max_backward_prefetch_units=max_backward_prefetch_units,
                backward_prefetch_timing=backward_prefetch_timing,
                prefetch_policy=prefetch_policy,
                max_cached_full_param_buffers_per_key=max_cached_full_param_buffers_per_key,
                max_active_full_param_buffers=max_active_full_param_buffers,
                max_active_full_param_numel=max_active_full_param_numel,
                max_active_full_param_memory_mb=max_active_full_param_memory_mb,
                max_pending_backward_reduces=max_pending_backward_reduces,
                trim_cuda_cache=trim_cuda_cache,
                cuda_cache_trim_threshold_mb=cuda_cache_trim_threshold_mb,
            )
            self.scheduler = MatrixFSDPScheduler(self.fsdp_param_groups, **scheduler_kwargs)
            self.state_manager = MatrixFSDPOptimizerStateManager(self, self.fsdp_param_groups)
            self.matrix_fsdp = self

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups

    @property
    def state(self) -> dict[Any, Any]:
        state = {}
        if self.muon is not None:
            state.update(self.muon.state)
        if self.adamw is not None:
            state.update(self.adamw.state)
        return state

    @property
    def defaults(self) -> dict[str, Any]:
        return {
            "muon": dict(self.muon.defaults) if self.muon is not None else None,
            "adamw": dict(self.adamw.defaults) if self.adamw is not None else None,
        }

    @property
    def runtime_param_groups(self) -> list[MatrixFSDPParamGroup]:
        return self.fsdp_param_groups

    @property
    def fsdp_param_groups(self) -> list[MatrixFSDPParamGroup]:
        return self._fsdp_param_groups

    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("MixedMuonAdamWOptimizer does not support closures.")
        if self._matrix_lifecycle_enabled:
            assert self.scheduler is not None
            _raise_if_no_sync_active(self.fsdp_param_groups, "MixedMuonAdamWOptimizer.step()")
            _prepare_matrix_optimizer_step(self.fsdp_param_groups, self.scheduler)
            if self.flat_adamw_state and self.adamw is not None:
                initialize_flat_adamw_state(self.adamw, self.fsdp_param_groups)
        result = None
        if self.muon is not None:
            result = self.muon.step()
        if self.adamw is not None:
            adamw_result = self.adamw.step()
            result = result if result is not None else adamw_result
        if self._matrix_lifecycle_enabled:
            assert self.scheduler is not None
            self.scheduler.maybe_trim_cuda_cache()
        return result

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        if self._matrix_lifecycle_enabled:
            assert self.scheduler is not None
            _raise_if_no_sync_active(self.fsdp_param_groups, "MixedMuonAdamWOptimizer.zero_grad()")
            _prepare_matrix_optimizer_zero_grad(self.fsdp_param_groups, self.scheduler)
        if self.muon is not None:
            self.muon.zero_grad(*args, **kwargs)
        if self.adamw is not None:
            self.adamw.zero_grad(*args, **kwargs)
        if self._matrix_lifecycle_enabled:
            assert self.scheduler is not None
            _finish_matrix_optimizer_zero_grad(self.fsdp_param_groups, self.scheduler)

    def state_dict(self) -> dict[str, Any]:
        return {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
            "group_summaries": tuple(group.as_dict() for group in self.group_summaries),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.muon is not None and state_dict.get("muon") is not None:
            self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None and state_dict.get("adamw") is not None:
            self.adamw.load_state_dict(state_dict["adamw"])

    def iter_named_optimizers(self) -> tuple[tuple[str, Any], ...]:
        optimizers = []
        if self.muon is not None:
            optimizers.append(("muon", self.muon))
        if self.adamw is not None:
            optimizers.append(("adamw", self.adamw))
        return tuple(optimizers)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        raise NotImplementedError("MixedMuonAdamWOptimizer does not support adding param groups after construction.")

    def no_sync(self) -> MatrixFSDPNoSync:
        if not self._matrix_lifecycle_enabled:
            raise RuntimeError("MixedMuonAdamWOptimizer was not constructed from a MatrixFSDP module.")
        return MatrixFSDPNoSync(self.fsdp_param_groups)

    def local_state_summary(self) -> dict[str, Any]:
        if self.state_manager is None:
            return {}
        return self.state_manager.local_state_summary()

    def local_state_dtensors(self) -> dict[str, dict[str, Any]]:
        if self.state_manager is None:
            return {}
        return self.state_manager.local_state_dtensors()

    def local_state_objects(self) -> dict[str, dict[str, MatrixShardedState]]:
        if self.state_manager is None:
            return {}
        return self.state_manager.local_state_objects()

    @property
    def state_dtensors(self) -> dict[str, dict[str, Any]]:
        return self.refresh_state_dtensors()

    @property
    def state_objects(self) -> dict[str, dict[str, MatrixShardedState]]:
        if self.state_manager is None:
            return {}
        if not self.state_manager.state_objects:
            self.refresh_state_dtensors()
        return self.state_manager.state_objects

    def refresh_state_dtensors(self) -> dict[str, dict[str, Any]]:
        if self.state_manager is None:
            return {}
        return self.state_manager.refresh_state_dtensors()

    def validate_local_state_shapes(self) -> None:
        if self.state_manager is not None:
            self.state_manager.validate_local_state_shapes()

    def remove_matrix_lifecycle(self) -> None:
        self._matrix_lifecycle_enabled = False
        self._fsdp_param_groups = []
        self.scheduler = None
        self.state_manager = None
        try:
            delattr(self, "matrix_fsdp")
        except AttributeError:
            pass

    def optimizer_group_summary(self) -> dict[str, Any]:
        return {
            "optimizer": "mixed_muon_adamw",
            "has_muon": self.muon is not None,
            "has_adamw": self.adamw is not None,
            "groups": tuple(group.as_dict() for group in self.group_summaries),
        }


class _LazyMuonOptimizer:
    """
    Delay torch.optim.Muon construction until the first optimizer step.

    PyTorch Muon allocates momentum/state eagerly in current builds. For large
    activation-checkpointed models this can make the first forward run out of
    memory even though the same state would fit after backward frees transient
    activations. This small adapter keeps the normal optimizer surface while
    deferring that state allocation to the first step.
    """

    def __init__(self, params: Iterable[nn.Parameter], **kwargs: Any) -> None:
        self.params = tuple(params)
        self.kwargs = dict(kwargs)
        self._optimizer: Optimizer | None = None
        self._param_groups = [{**self.kwargs, "params": list(self.params)}]

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._optimizer.param_groups if self._optimizer is not None else self._param_groups

    @property
    def state(self) -> dict[Any, Any]:
        return self._optimizer.state if self._optimizer is not None else {}

    @property
    def defaults(self) -> dict[str, Any]:
        return self._optimizer.defaults if self._optimizer is not None else dict(self.kwargs)

    def step(self, closure=None):
        optimizer = self._materialize()
        return optimizer.step() if closure is None else optimizer.step(closure=closure)

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        if self._optimizer is not None:
            self._optimizer.zero_grad(*args, **kwargs)
            return
        set_to_none = kwargs.get("set_to_none", True)
        if args:
            # Optimizer.zero_grad accepts set_to_none as its first positional arg.
            set_to_none = args[0]
        for param in self.params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.detach_()
                param.grad.zero_()

    def state_dict(self) -> dict[str, Any]:
        if self._optimizer is not None:
            return self._optimizer.state_dict()
        param_groups = []
        for group in self.param_groups:
            state_group = {key: value for key, value in group.items() if key != "params"}
            state_group["params"] = list(range(len(group["params"])))
            param_groups.append(state_group)
        return {"state": {}, "param_groups": param_groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._materialize().load_state_dict(state_dict)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self._materialize().add_param_group(param_group)

    def _materialize(self) -> Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.Muon(self.params, **self.kwargs)
            _remove_prepared_matrix_optimizer(self._optimizer)
        return self._optimizer


def classify_matrix_optimizer_params(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    *,
    default_matrix_optimizer: str = "muon",
    default_other_optimizer: str = "adamw",
) -> ClassifiedMatrixOptimizerParams:
    param_groups = collect_param_groups(module_or_units)
    if not param_groups:
        raise ValueError("Expected a module wrapped by matrix_fully_shard().")
    if default_matrix_optimizer not in ("muon", "adamw"):
        raise ValueError("default_matrix_optimizer must be 'muon' or 'adamw'.")
    if default_other_optimizer not in ("muon", "adamw"):
        raise ValueError("default_other_optimizer must be 'muon' or 'adamw'.")

    grouped: dict[str, list[tuple[ManagedParam, nn.Parameter]]] = {"muon": [], "adamw": []}
    seen_param_ids: set[int] = set()
    for unit in param_groups:
        flat_buffer = unit.flat_buffer
        if flat_buffer is None:
            continue
        for view in flat_buffer.local_param_views:
            managed_param = view.managed_param
            param = managed_param.param
            if id(param) in seen_param_ids or param.numel() == 0:
                continue
            seen_param_ids.add(id(param))
            optimizer_type = _optimizer_type_for_param(
                managed_param,
                param,
                default_matrix_optimizer=default_matrix_optimizer,
                default_other_optimizer=default_other_optimizer,
            )
            grouped[optimizer_type].append((managed_param, param))

    muon_params = tuple(param for _, param in grouped["muon"])
    adamw_params = tuple(param for _, param in grouped["adamw"])
    group_summaries = tuple(
        _optimizer_group_summary(name, optimizer_type, entries)
        for name, optimizer_type, entries in (
            ("muon", "muon", grouped["muon"]),
            ("adamw", "adamw", grouped["adamw"]),
        )
        if entries
    )
    if not muon_params and not adamw_params:
        raise RuntimeError("No non-empty local MatrixFSDP parameters were found for optimizer construction.")
    return ClassifiedMatrixOptimizerParams(
        muon_params=muon_params,
        adamw_params=adamw_params,
        group_summaries=group_summaries,
    )


def make_mixed_muon_adamw_optimizer(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    *,
    default_matrix_optimizer: str = "muon",
    default_other_optimizer: str = "adamw",
    manage_matrix_lifecycle: bool = True,
    **optimizer_kwargs: Any,
) -> MixedMuonAdamWOptimizer:
    fsdp_param_groups = _normalize_param_groups(module_or_units)
    classified = classify_matrix_optimizer_params(
        fsdp_param_groups,
        default_matrix_optimizer=default_matrix_optimizer,
        default_other_optimizer=default_other_optimizer,
    )
    return MixedMuonAdamWOptimizer(
        classified.muon_params,
        classified.adamw_params,
        group_summaries=classified.group_summaries,
        fsdp_param_groups=fsdp_param_groups if manage_matrix_lifecycle else None,
        **optimizer_kwargs,
    )


def configure_optimizer(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    optimizer: str | MatrixOptimizerConfig = "adamw",
    **kwargs: Any,
) -> Optimizer | MixedMuonAdamWOptimizer:
    """
    Build an optimizer for a MatrixFSDP-managed module.

    Regular torch optimizers returned here are still ordinary torch optimizer
    objects. If the parameters are MatrixFSDP-managed, the auto-prepare hook
    attaches lifecycle handling during optimizer construction. The mixed
    Muon/AdamW route uses ``MixedMuonAdamWOptimizer`` because it needs
    MatrixFSDP shard hints to split parameters by optimizer role.
    """

    default_matrix_optimizer = "muon"
    default_other_optimizer = "adamw"
    optimizer_kwargs: dict[str, Any] = {}
    if isinstance(optimizer, MatrixOptimizerConfig):
        optimizer_kind = optimizer.optimizer
        optimizer_kwargs.update(dict(optimizer.kwargs))
        default_matrix_optimizer = optimizer.default_matrix_optimizer
        default_other_optimizer = optimizer.default_other_optimizer
    else:
        optimizer_kind = optimizer
    optimizer_kwargs.update(kwargs)

    default_matrix_optimizer = optimizer_kwargs.pop("default_matrix_optimizer", default_matrix_optimizer)
    default_other_optimizer = optimizer_kwargs.pop("default_other_optimizer", default_other_optimizer)
    scheduler_kwargs, optimizer_kwargs = _split_scheduler_kwargs(optimizer_kwargs)

    optimizer_kind = optimizer_kind.lower()
    if optimizer_kind in ("mixed_muon_adamw", "muon_adamw", "mixed"):
        return make_mixed_muon_adamw_optimizer(
            module_or_units,
            default_matrix_optimizer=default_matrix_optimizer,
            default_other_optimizer=default_other_optimizer,
            **optimizer_kwargs,
            **scheduler_kwargs,
        )

    params = _optimizer_params_from_module_or_units(module_or_units)
    if optimizer_kind == "adamw":
        configured_optimizer: Optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
    elif optimizer_kind == "sgd":
        configured_optimizer = torch.optim.SGD(params, **optimizer_kwargs)
    elif optimizer_kind == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("torch.optim.Muon is not available in this PyTorch build.")
        configured_optimizer = torch.optim.Muon(params, **optimizer_kwargs)
    else:
        raise ValueError(
            "optimizer must be one of 'adamw', 'sgd', 'muon', or 'mixed_muon_adamw', "
            f"got {optimizer_kind!r}."
        )

    if scheduler_kwargs:
        prepare_matrix_optimizer(configured_optimizer, module_or_units, **scheduler_kwargs)
    return configured_optimizer


configure_matrix_optimizer = configure_optimizer


def _optimizer_type_for_param(
    managed_param: ManagedParam,
    param: nn.Parameter,
    *,
    default_matrix_optimizer: str,
    default_other_optimizer: str,
) -> str:
    optimizer_type = managed_param.shard_hint.optimizer_type
    if optimizer_type == "muon":
        if len(managed_param.shape) != 2 or param.ndim != 2:
            raise ValueError(
                f"Param {managed_param.fqn!r} is marked for Muon, but its local view is not a full 2D matrix. "
                f"Use matrix_owner planning for Muon parameters."
            )
        return "muon"
    if optimizer_type == "adamw":
        return "adamw"
    if len(managed_param.shape) == 2 and param.ndim == 2:
        return default_matrix_optimizer
    return default_other_optimizer


def _optimizer_group_summary(
    name: str,
    optimizer_type: str,
    entries: list[tuple[ManagedParam, nn.Parameter]],
) -> MatrixOptimizerParamGroupSummary:
    return MatrixOptimizerParamGroupSummary(
        name=name,
        optimizer_type=optimizer_type,
        fqns=tuple(managed_param.fqn for managed_param, _ in entries),
        param_numel=sum(managed_param.numel for managed_param, _ in entries),
        local_numel=sum(param.numel() for _, param in entries),
    )


def _resolve_scheduler_kwargs(
    *,
    scheduler_config: MatrixFSDPSchedulerConfig | None,
    max_unsharded_prefetch_units: int | None,
    max_forward_prefetch_units: int | None,
    max_backward_prefetch_units: int | None,
    backward_prefetch_timing: BackwardPrefetchTiming,
    prefetch_policy: PrefetchPolicy,
    max_cached_full_param_buffers_per_key: int,
    max_active_full_param_buffers: int | None,
    max_active_full_param_numel: int | None,
    max_active_full_param_memory_mb: float | None,
    max_pending_backward_reduces: int | None,
    trim_cuda_cache: bool,
    cuda_cache_trim_threshold_mb: float,
) -> dict[str, Any]:
    kwargs = {
        "max_unsharded_prefetch_units": max_unsharded_prefetch_units,
        "max_forward_prefetch_units": max_forward_prefetch_units,
        "max_backward_prefetch_units": max_backward_prefetch_units,
        "backward_prefetch_timing": backward_prefetch_timing,
        "prefetch_policy": prefetch_policy,
        "max_cached_full_param_buffers_per_key": max_cached_full_param_buffers_per_key,
        "max_active_full_param_buffers": max_active_full_param_buffers,
        "max_active_full_param_numel": max_active_full_param_numel,
        "max_active_full_param_memory_mb": max_active_full_param_memory_mb,
        "max_pending_backward_reduces": max_pending_backward_reduces,
        "trim_cuda_cache": trim_cuda_cache,
        "cuda_cache_trim_threshold_mb": cuda_cache_trim_threshold_mb,
    }
    if scheduler_config is not None:
        defaults = MatrixFSDPSchedulerConfig().as_scheduler_kwargs()
        conflicts = [name for name, value in kwargs.items() if value != defaults[name]]
        if conflicts:
            joined = ", ".join(sorted(conflicts))
            raise ValueError(f"Pass either scheduler_config or explicit scheduler kwargs, not both: {joined}.")
        kwargs = scheduler_config.as_scheduler_kwargs()
    if (
        kwargs["max_unsharded_prefetch_units"] is None
        and kwargs["max_forward_prefetch_units"] is None
        and kwargs["prefetch_policy"] == "static"
    ):
        kwargs["max_unsharded_prefetch_units"] = DEFAULT_MAX_UNSHARDED_PREFETCH_UNITS
    return kwargs


def _normalize_param_groups(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
) -> list[MatrixFSDPParamGroup]:
    param_groups = collect_param_groups(module_or_units)
    if not param_groups:
        raise ValueError("Expected a module wrapped by matrix_fully_shard().")
    return param_groups


def _optimizer_params_from_module_or_units(
    module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
) -> tuple[nn.Parameter, ...]:
    if isinstance(module_or_units, nn.Module):
        return tuple(module_or_units.parameters())
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for param_group in _normalize_param_groups(module_or_units):
        for managed_param in param_group.managed_params:
            param_id = id(managed_param.param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(managed_param.param)
    return tuple(params)


def _split_scheduler_kwargs(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scheduler_kwargs = {key: value for key, value in kwargs.items() if key in _SCHEDULER_KWARG_NAMES}
    optimizer_kwargs = {key: value for key, value in kwargs.items() if key not in _SCHEDULER_KWARG_NAMES}
    return scheduler_kwargs, optimizer_kwargs


def _optimizer_step_has_closure(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if kwargs.get("closure") is not None:
        return True
    return len(args) > 1 and args[1] is not None


def _raise_if_no_sync_active(fsdp_param_groups: Iterable[MatrixFSDPParamGroup], operation: str) -> None:
    active_units = [unit for unit in fsdp_param_groups if unit.is_no_sync_active]
    if active_units:
        raise RuntimeError(f"{operation} cannot run inside no_sync().")


def _prepare_matrix_optimizer_step(
    fsdp_param_groups: Iterable[MatrixFSDPParamGroup],
    scheduler: MatrixFSDPScheduler,
) -> None:
    scheduler.wait_pending_backward_reduce()
    for unit in fsdp_param_groups:
        if unit.has_pending_backward_reduce:
            unit.finalize_backward()
            continue
        if unit.finalize_after_backward_enabled and unit.finalized_after_backward:
            continue
        if unit.lifecycle_state == FSDPLifecycleState.SHARDED and not _unit_has_grad(unit):
            continue
        unit.finalize_backward()
    scheduler.finish_backward_iteration()


def _prepare_matrix_optimizer_zero_grad(
    fsdp_param_groups: Iterable[MatrixFSDPParamGroup],
    scheduler: MatrixFSDPScheduler,
) -> None:
    scheduler.wait_pending_backward_reduce()
    for unit in fsdp_param_groups:
        if unit.has_pending_backward_reduce:
            unit.finalize_backward()


def _finish_matrix_optimizer_zero_grad(
    fsdp_param_groups: Iterable[MatrixFSDPParamGroup],
    scheduler: MatrixFSDPScheduler,
) -> None:
    scheduler.finish_backward_iteration()
    for unit in fsdp_param_groups:
        unit.reset_grad_accumulation()
        if unit.flat_buffer is not None:
            unit.flat_buffer.clear_local_grad_shard()
    scheduler.maybe_trim_cuda_cache()


class MatrixFSDPOptimizer:
    """
    Thin optimizer wrapper for MatrixFSDP V0.

    V0 keeps gradient finalization explicit inside the runtime, but this wrapper
    gives users the normal ``loss.backward(); optim.step()`` shape by calling
    ``MatrixFSDPParamGroup.finalize_backward()`` before delegating to the inner
    optimizer.
    """

    def __init__(
        self,
        optimizer: Optimizer | MixedMuonAdamWOptimizer,
        module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
        *,
        max_unsharded_prefetch_units: int | None = None,
        max_forward_prefetch_units: int | None = None,
        max_backward_prefetch_units: int | None = None,
        backward_prefetch_timing: BackwardPrefetchTiming = "pre_backward",
        prefetch_policy: PrefetchPolicy = "static",
        max_cached_full_param_buffers_per_key: int = 0,
        max_active_full_param_buffers: int | None = None,
        max_active_full_param_numel: int | None = None,
        max_active_full_param_memory_mb: float | None = None,
        max_pending_backward_reduces: int | None = 1,
        trim_cuda_cache: bool = False,
        cuda_cache_trim_threshold_mb: float = 1024.0,
        flat_adamw_state: bool = False,
        scheduler_config: MatrixFSDPSchedulerConfig | None = None,
    ):
        if isinstance(optimizer, Optimizer):
            existing_prepared = getattr(optimizer, "matrix_fsdp", None)
            if isinstance(existing_prepared, PreparedMatrixOptimizer):
                existing_prepared.remove()
        if isinstance(optimizer, MixedMuonAdamWOptimizer) and getattr(optimizer, "_matrix_lifecycle_enabled", False):
            optimizer.remove_matrix_lifecycle()
        self.optimizer = optimizer
        self._fsdp_param_groups = self._normalize_units(module_or_units)
        self.flat_adamw_state = flat_adamw_state
        scheduler_kwargs = _resolve_scheduler_kwargs(
            scheduler_config=scheduler_config,
            max_unsharded_prefetch_units=max_unsharded_prefetch_units,
            max_forward_prefetch_units=max_forward_prefetch_units,
            max_backward_prefetch_units=max_backward_prefetch_units,
            backward_prefetch_timing=backward_prefetch_timing,
            prefetch_policy=prefetch_policy,
            max_cached_full_param_buffers_per_key=max_cached_full_param_buffers_per_key,
            max_active_full_param_buffers=max_active_full_param_buffers,
            max_active_full_param_numel=max_active_full_param_numel,
            max_active_full_param_memory_mb=max_active_full_param_memory_mb,
            max_pending_backward_reduces=max_pending_backward_reduces,
            trim_cuda_cache=trim_cuda_cache,
            cuda_cache_trim_threshold_mb=cuda_cache_trim_threshold_mb,
        )
        self.scheduler = MatrixFSDPScheduler(self.fsdp_param_groups, **scheduler_kwargs)
        self.state_manager = MatrixFSDPOptimizerStateManager(self.optimizer, self.fsdp_param_groups)

    @classmethod
    def from_shard_hints(
        cls,
        module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
        *,
        default_matrix_optimizer: str = "muon",
        default_other_optimizer: str = "adamw",
        max_unsharded_prefetch_units: int | None = None,
        max_forward_prefetch_units: int | None = None,
        max_backward_prefetch_units: int | None = None,
        backward_prefetch_timing: BackwardPrefetchTiming = "pre_backward",
        prefetch_policy: PrefetchPolicy = "static",
        max_cached_full_param_buffers_per_key: int = 0,
        max_active_full_param_buffers: int | None = None,
        max_active_full_param_numel: int | None = None,
        max_active_full_param_memory_mb: float | None = None,
        max_pending_backward_reduces: int | None = 1,
        trim_cuda_cache: bool = False,
        cuda_cache_trim_threshold_mb: float = 1024.0,
        flat_adamw_state: bool = False,
        scheduler_config: MatrixFSDPSchedulerConfig | None = None,
        **optimizer_kwargs: Any,
    ) -> "MatrixFSDPOptimizer":
        inner_optimizer = make_mixed_muon_adamw_optimizer(
            module_or_units,
            default_matrix_optimizer=default_matrix_optimizer,
            default_other_optimizer=default_other_optimizer,
            manage_matrix_lifecycle=False,
            **optimizer_kwargs,
        )
        return cls(
            inner_optimizer,
            module_or_units,
            max_unsharded_prefetch_units=max_unsharded_prefetch_units,
            max_forward_prefetch_units=max_forward_prefetch_units,
            max_backward_prefetch_units=max_backward_prefetch_units,
            backward_prefetch_timing=backward_prefetch_timing,
            prefetch_policy=prefetch_policy,
            max_cached_full_param_buffers_per_key=max_cached_full_param_buffers_per_key,
            max_active_full_param_buffers=max_active_full_param_buffers,
            max_active_full_param_numel=max_active_full_param_numel,
            max_active_full_param_memory_mb=max_active_full_param_memory_mb,
            max_pending_backward_reduces=max_pending_backward_reduces,
            trim_cuda_cache=trim_cuda_cache,
            cuda_cache_trim_threshold_mb=cuda_cache_trim_threshold_mb,
            flat_adamw_state=flat_adamw_state,
            scheduler_config=scheduler_config,
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    @property
    def runtime_param_groups(self) -> list[MatrixFSDPParamGroup]:
        return self.fsdp_param_groups

    @property
    def fsdp_param_groups(self) -> list[MatrixFSDPParamGroup]:
        return self._fsdp_param_groups

    @property
    def state(self) -> dict[Any, Any]:
        return self.optimizer.state

    @property
    def defaults(self) -> dict[str, Any]:
        return self.optimizer.defaults

    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError("MatrixFSDPOptimizer V0 does not support optimizer closures.")
        _raise_if_no_sync_active(self.fsdp_param_groups, "MatrixFSDPOptimizer.step()")
        _prepare_matrix_optimizer_step(self.fsdp_param_groups, self.scheduler)
        _initialize_matrix_optimizer_state(
            self.optimizer,
            self.fsdp_param_groups,
            flat_adamw_state=self.flat_adamw_state,
        )
        result = self.optimizer.step()
        self.scheduler.maybe_trim_cuda_cache()
        return result

    def no_sync(self) -> MatrixFSDPNoSync:
        return MatrixFSDPNoSync(self.fsdp_param_groups)

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        _raise_if_no_sync_active(self.fsdp_param_groups, "MatrixFSDPOptimizer.zero_grad()")
        _prepare_matrix_optimizer_zero_grad(self.fsdp_param_groups, self.scheduler)
        self.optimizer.zero_grad(*args, **kwargs)
        _finish_matrix_optimizer_zero_grad(self.fsdp_param_groups, self.scheduler)

    def local_state_summary(self) -> dict[str, Any]:
        """
        Summarize optimizer state currently attached to local sharded params.

        Optimizers must be constructed after ``matrix_fully_shard()``, so their
        tensor states should follow the local parameter views rather than the
        original full parameter sizes. This summary makes that invariant easy to
        test and inspect.
        """
        return self.state_manager.local_state_summary()

    def optimizer_group_summary(self) -> dict[str, Any]:
        if hasattr(self.optimizer, "optimizer_group_summary"):
            return self.optimizer.optimizer_group_summary()
        groups = []
        for index, group in enumerate(self.optimizer.param_groups):
            params = tuple(param for param in group["params"] if torch.is_tensor(param))
            groups.append(
                {
                    "name": f"group_{index}",
                    "optimizer_type": type(self.optimizer).__name__,
                    "fqns": (),
                    "num_params": len(params),
                    "param_numel": sum(param.numel() for param in params),
                    "local_numel": sum(param.numel() for param in params),
                }
            )
        return {
            "optimizer": type(self.optimizer).__name__,
            "has_muon": False,
            "has_adamw": type(self.optimizer).__name__.lower() == "adamw",
            "groups": tuple(groups),
        }

    def local_state_dtensors(self) -> dict[str, dict[str, Any]]:
        """
        Return DTensor metadata wrappers for local sharded optimizer states.

        The inner torch optimizer still owns ordinary local tensors for the
        actual step. These wrappers alias that storage and attach MatrixShard
        placement metadata so checkpointing and validation can reason about
        optimizer state with the same mesh semantics as sharded weights/grads.
        """
        return self.state_manager.local_state_dtensors()

    def local_state_objects(self) -> dict[str, dict[str, MatrixShardedState]]:
        return self.state_manager.local_state_objects()

    @property
    def state_dtensors(self) -> dict[str, dict[str, Any]]:
        return self.refresh_state_dtensors()

    @property
    def state_objects(self) -> dict[str, dict[str, MatrixShardedState]]:
        if not self.state_manager.state_objects:
            self.refresh_state_dtensors()
        return self.state_manager.state_objects

    def refresh_state_dtensors(self) -> dict[str, dict[str, Any]]:
        return self.state_manager.refresh_state_dtensors()

    def validate_local_state_shapes(self) -> None:
        self.state_manager.validate_local_state_shapes()

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self.optimizer.add_param_group(param_group)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)

    def _normalize_units(
        self,
        module_or_units: nn.Module | MatrixFSDPParamGroup | Iterable[MatrixFSDPParamGroup],
    ) -> list[MatrixFSDPParamGroup]:
        return _normalize_param_groups(module_or_units)


def _unit_has_grad(unit: MatrixFSDPParamGroup) -> bool:
    return any(managed_param.param.grad is not None for managed_param in unit.managed_params)


def _initialize_matrix_optimizer_state(
    optimizer: Optimizer | MixedMuonAdamWOptimizer,
    fsdp_param_groups: list[MatrixFSDPParamGroup],
    *,
    flat_adamw_state: bool,
) -> None:
    if not flat_adamw_state:
        return
    if isinstance(optimizer, MixedMuonAdamWOptimizer):
        if optimizer.adamw is not None:
            initialize_flat_adamw_state(optimizer.adamw, fsdp_param_groups)
        return
    initialize_flat_adamw_state(optimizer, fsdp_param_groups)
