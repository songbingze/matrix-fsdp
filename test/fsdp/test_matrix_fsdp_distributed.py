import copy
import os
import socket
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard
from torch.utils.checkpoint import checkpoint

from matrix_fsdp import (
    DataParallelMeshDims,
    FSDPLifecycleState,
    FSDPRuntimeState,
    ParamShardHint,
    MatrixFSDPOptimizer,
    auto_group_plan,
    configure_optimizer,
    fully_shard,
    load_matrix_dcp,
    load_matrix_state_dict,
    matrix_fully_shard,
    matrix_get_state_dict,
    matrix_set_state_dict,
    matrix_state_dict,
    save_matrix_dcp,
)
from matrix_fsdp.grad_bucket import BucketParamGrad, MatrixGradBucket, classify_copy_in_layout
from matrix_fsdp.planner import fsdp2_chunk_plan, ordered_group_plan, parameter_boundary_plan
from matrix_fsdp.state import MatrixShardedState


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _make_muon_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8, bias=False), nn.ReLU(), nn.Linear(8, 2, bias=False))


def _make_mixed_optimizer_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2, bias=False))


def _load_dcp_metadata(checkpoint_dir: str, rank: int) -> dict:
    payload = torch.load(os.path.join(checkpoint_dir, "matrix_metadata.pt"), map_location="cpu")
    return payload["ranks"][rank]


class _CheckpointBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(4, 8)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(8, 4)

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class _CheckpointModel(nn.Module):
    def __init__(self, *, use_reentrant: bool) -> None:
        super().__init__()
        self.use_reentrant = use_reentrant
        self.b1 = _CheckpointBlock()
        self.b2 = _CheckpointBlock()
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        x = checkpoint(self.b1, x, use_reentrant=self.use_reentrant)
        x = checkpoint(self.b2, x, use_reentrant=self.use_reentrant)
        return self.out(x)


def _local_muon_params(module: nn.Module) -> list[nn.Parameter]:
    return [param for param in module.parameters() if param.ndim == 2 and param.numel() > 0]


def _assert_torch_optimizer_state_dict_close(actual: dict, expected: dict) -> None:
    assert actual["param_groups"] == expected["param_groups"]
    assert set(actual["state"]) == set(expected["state"])
    for param_id, expected_state in expected["state"].items():
        actual_state = actual["state"][param_id]
        assert actual_state.keys() == expected_state.keys()
        for name, value in expected_state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(actual_state[name], value)
            else:
                assert actual_state[name] == value


def _assert_mixed_optimizer_state_dict_close(actual: dict, expected: dict) -> None:
    assert actual["group_summaries"] == expected["group_summaries"]
    for component_name in ("muon", "adamw"):
        if expected[component_name] is None:
            assert actual[component_name] is None
            continue
        assert actual[component_name] is not None
        _assert_torch_optimizer_state_dict_close(actual[component_name], expected[component_name])


def _mixed_optimizer_state_by_fqn(optimizer: MatrixFSDPOptimizer, module) -> dict[str, dict[str, torch.Tensor]]:
    states_by_fqn = {}
    params_by_id = {id(param): fqn for fqn, param in module.named_parameters()}
    inner_optimizer = optimizer.optimizer
    for component in (getattr(inner_optimizer, "muon", None), getattr(inner_optimizer, "adamw", None)):
        if component is None:
            continue
        for param, state in component.state.items():
            fqn = params_by_id.get(id(param))
            if fqn is not None:
                states_by_fqn[fqn] = state
    return states_by_fqn


def _matrix_fully_shard_linear_units(
    model: nn.Sequential,
    mesh: DeviceMesh,
    *,
    reshard_after_forward: bool = False,
    forward_prefetch: bool = False,
    backward_prefetch: bool = False,
    finalize_after_backward: bool = False,
) -> nn.Sequential:
    model[0] = matrix_fully_shard(
        model[0],
        mesh,
        reshard_after_forward=reshard_after_forward,
        forward_prefetch=forward_prefetch,
        backward_prefetch=backward_prefetch,
        finalize_after_backward=finalize_after_backward,
    )
    model[2] = matrix_fully_shard(
        model[2],
        mesh,
        reshard_after_forward=reshard_after_forward,
        forward_prefetch=forward_prefetch,
        backward_prefetch=backward_prefetch,
        finalize_after_backward=finalize_after_backward,
    )
    return model


def _flatten_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors])


def _flatten_param_grads(module: nn.Module) -> torch.Tensor:
    grads = []
    for param in module.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param).reshape(-1))
        else:
            grads.append(param.grad.detach().reshape(-1))
    return torch.cat(grads)


def _full_param_buffer_released(flat_buffer) -> bool:
    full_buffer = flat_buffer.full_buffer
    return full_buffer is None or full_buffer.untyped_storage().nbytes() == 0


def _flatten_param_grads_by_fqns(module: nn.Module, fqns: tuple[str, ...]) -> torch.Tensor:
    params_by_fqn = dict(module.named_parameters())
    grads = []
    for fqn in fqns:
        param = params_by_fqn[fqn]
        if param.grad is None:
            grads.append(torch.zeros_like(param).reshape(-1))
        else:
            grads.append(param.grad.detach().reshape(-1))
    return torch.cat(grads)


def _set_param_grads_from_flat(module: nn.Module, flat_grad: torch.Tensor) -> None:
    offset = 0
    for param in module.parameters():
        grad = flat_grad[offset : offset + param.numel()].view_as(param).clone()
        param.grad = grad
        offset += param.numel()


def _pack_full_tensor_by_segments(full_tensor: torch.Tensor, segments) -> torch.Tensor:
    return torch.cat([full_tensor[segment.global_start : segment.global_end] for segment in segments])


def _flatten_local_grads_by_shard_order(flat_buffer) -> torch.Tensor:
    pieces = []
    for view in sorted(flat_buffer.local_param_views, key=lambda local_view: local_view.shard_start):
        grad = view.managed_param.param.grad
        assert grad is not None
        flat_grad = grad.detach().reshape(-1)
        if flat_grad.numel() == view.numel:
            pieces.append(flat_grad)
        else:
            pieces.append(flat_grad[view.param_start : view.param_end])
    if not pieces:
        return flat_buffer.local_shard.new_empty(0)
    return torch.cat(pieces)


def _clone_deferred_grad_payload(unit) -> torch.Tensor | None:
    flat_buffer = unit.flat_buffer
    assert flat_buffer is not None
    if flat_buffer.grad_bucket_input is not None:
        return flat_buffer.grad_bucket_input.detach().clone()
    if flat_buffer.full_grad_buffer is not None:
        return flat_buffer.full_grad_buffer.detach().clone()
    pieces = []
    for managed_param in unit.managed_params:
        grad = managed_param.param.grad
        if grad is not None:
            pieces.append(grad.detach().reshape(-1))
    if not pieces:
        return None
    return torch.cat(pieces).clone()


def _assert_unit_layout_matches_flat_buffer(unit, rank: int) -> None:
    flat_buffer = unit.flat_buffer
    assert flat_buffer is not None
    assert unit.layout is not None
    assert unit.shard_plan == flat_buffer.plan
    assert unit.layout.to_shard_plan() == flat_buffer.plan

    expected_fqns = tuple(dict.fromkeys(view.managed_param.fqn for view in flat_buffer.local_param_views))
    assert unit.params_for_rank(rank) == expected_fqns
    for fqn in expected_fqns:
        assert rank in unit.owner_ranks(fqn)
        assert unit.rank_segments_for_param(fqn, rank)


def _assert_finalize_after_backward_events(unit) -> None:
    event_names = [event.name for event in unit.runtime_events]
    cursor = 0
    for expected_name in (
        "pre_forward",
        "unshard",
        "reshard_after_forward",
        "pre_backward_unshard",
        "unshard",
        "finalize_backward",
    ):
        cursor = event_names.index(expected_name, cursor) + 1


def _assert_default_fast_path_contract(unit, *, expected_reshard_after_forward: bool = True) -> None:
    assert unit.reshard_after_forward_enabled is expected_reshard_after_forward
    assert unit.forward_prefetch_enabled
    assert unit.backward_prefetch_enabled
    assert unit.finalize_after_backward_enabled
    assert unit.backward_reduce_strategy == "bucket_reduce_scatter"
    assert not unit.use_saved_tensor_hooks
    assert not unit.use_zero_copy_grad_bucket


def _event_sequence(unit, name: str) -> int:
    for event in unit.runtime_events:
        if event.name == name:
            return event.sequence
    raise AssertionError(f"Missing runtime event {name!r}.")


def _assert_forward_prefetch_events(units) -> None:
    assert _event_sequence(units[1], "forward_prefetch") < _event_sequence(units[1], "pre_forward")
    assert _event_sequence(units[1], "start_unshard:forward_prefetch") < _event_sequence(
        units[1],
        "pre_forward",
    )
    assert _event_sequence(units[1], "pre_forward") < _event_sequence(units[1], "wait_unshard:pre_forward")


def _assert_backward_prefetch_events(units) -> None:
    assert _event_sequence(units[0], "backward_prefetch") < _event_sequence(units[0], "pre_backward_unshard")
    assert _event_sequence(units[0], "start_unshard:backward_prefetch") < _event_sequence(
        units[0],
        "pre_backward_unshard",
    )
    assert _event_sequence(units[0], "pre_backward_unshard") < _event_sequence(
        units[0],
        "wait_unshard:pre_backward",
    )


def _loopback_interface_name() -> str:
    names = {name for _, name in socket.if_nameindex()}
    for candidate in ("lo0", "lo"):
        if candidate in names:
            return candidate
    return "lo"


def _run_two_rank_step(
    rank: int,
    world_size: int,
    init_file: str,
    backend: str,
    device_type: str,
    use_parameter_boundary_plan: bool = False,
    use_fsdp2_chunk_plan: bool = False,
    reshard_after_forward: bool = False,
    finalize_after_backward: bool = False,
    backward_reduce_strategy: str = "flat",
) -> None:
    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    if device_type == "cuda":
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        eager_model = copy.deepcopy(model)

        mesh = DeviceMesh(device_type, torch.arange(world_size))
        if use_parameter_boundary_plan and use_fsdp2_chunk_plan:
            raise ValueError("Select at most one group planner.")
        group_planner = None
        if use_parameter_boundary_plan:
            group_planner = parameter_boundary_plan
        if use_fsdp2_chunk_plan:
            group_planner = fsdp2_chunk_plan
        sharded_model = matrix_fully_shard(
            model,
            mesh,
            group_planner=group_planner,
            reshard_after_forward=reshard_after_forward,
            finalize_after_backward=finalize_after_backward,
            backward_reduce_strategy=backward_reduce_strategy,
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None

        total_numel = sum(param.numel() for param in eager_model.parameters())
        plan = flat_buffer.plan
        local_start, local_end = plan.local_range(rank)

        assert unit.rank == rank
        assert unit.world_size == world_size
        assert flat_buffer.local_start == local_start
        assert flat_buffer.local_end == local_end
        assert flat_buffer.local_numel == plan.shard_sizes[rank]
        assert flat_buffer.local_shard.numel() == plan.shard_sizes[rank]
        if flat_buffer.placement is not None:
            assert flat_buffer.local_shard_dtensor is None
            assert flat_buffer.sharded_param is flat_buffer.param_state
            assert flat_buffer.param_state.has_same_data_ptr(flat_buffer.local_shard)
            assert flat_buffer.param_state.as_metadata()["matrix_shard"] == {
                "type": "MatrixShard",
                "dims": (0,),
                "local_units": flat_buffer.placement.local_units,
            }
        _assert_unit_layout_matches_flat_buffer(unit, rank)
        if use_fsdp2_chunk_plan:
            assert unit.runtime_layout_compatibility.mode == "segment_runtime"
            assert unit.use_saved_tensor_hooks

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        torch.manual_seed(1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        if reshard_after_forward:
            assert unit.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED
            assert not unit._is_unsharded
            assert _full_param_buffer_released(flat_buffer)
        else:
            assert unit.lifecycle_state == FSDPLifecycleState.UNSHARDED
            assert unit._is_unsharded
            assert flat_buffer.full_buffer is not None
            assert flat_buffer.full_buffer.numel() == total_numel

        sharded_loss.backward()
        if finalize_after_backward:
            assert unit.lifecycle_state == FSDPLifecycleState.SHARDED
            assert unit.finalized_after_backward
            assert _full_param_buffer_released(flat_buffer)
            if backward_reduce_strategy == "bucket_reduce_scatter":
                event_names = [event.name for event in unit.runtime_events]
                assert "prepare_grad_bucket" in event_names
                assert "collect_grad_bucket" in event_names
                assert "reduce_grad_bucket" in event_names
                assert "wait_reduce_grad_bucket" in event_names
                assert not unit.has_pending_backward_reduce
        else:
            assert unit.lifecycle_state == FSDPLifecycleState.UNSHARDED
            assert flat_buffer.full_buffer is not None
            assert flat_buffer.full_buffer.numel() == total_numel
        sharded_optim.step()
        if backward_reduce_strategy == "bucket_reduce_scatter":
            event_names = [event.name for event in unit.runtime_events]
            assert "wait_reduce_grad_bucket" in event_names
            assert not unit.has_pending_backward_reduce
        if use_fsdp2_chunk_plan:
            bucket = MatrixGradBucket(
                param_grads=tuple(
                    BucketParamGrad(mp, torch.zeros(mp.numel, device=device, dtype=mp.dtype))
                    for mp in unit.managed_params
                ),
                total_numel=flat_buffer.plan.total_numel,
                shard_sizes=flat_buffer.shard_sizes,
                rank_segments=flat_buffer.plan.rank_segments,
            )
            assert classify_copy_in_layout(bucket) == "fsdp2_chunk"
        assert unit.lifecycle_state == FSDPLifecycleState.SHARDED
        assert _full_param_buffer_released(flat_buffer)
        assert flat_buffer.local_shard.numel() == plan.shard_sizes[rank]

        sharded_optim.zero_grad()
        assert all(param.grad is None for param in sharded_model.parameters())

        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            assert eager_param.shape == sharded_param.shape
            torch.testing.assert_close(eager_param, sharded_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_default_fast_path_step(
    rank: int,
    world_size: int,
    init_file: str,
    api_name: str,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        eager_model = copy.deepcopy(model)

        mesh = DeviceMesh("cpu", torch.arange(world_size))
        if api_name == "fully_shard":
            sharded_model = fully_shard(model, mesh=mesh)
        elif api_name == "matrix_fully_shard":
            sharded_model = matrix_fully_shard(model, mesh)
        else:
            raise ValueError(f"Unknown API name {api_name!r}.")
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None
        _assert_default_fast_path_contract(
            unit,
            expected_reshard_after_forward=True,
        )
        _assert_unit_layout_matches_flat_buffer(unit, rank)

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        torch.manual_seed(rank + 19)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        averaged_grad = _flatten_param_grads(eager_model)
        dist.all_reduce(averaged_grad, group=unit.group)
        averaged_grad.div_(world_size)
        _set_param_grads_from_flat(eager_model, averaged_grad)
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        assert unit.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED
        assert _full_param_buffer_released(flat_buffer)

        sharded_loss.backward()
        assert unit.lifecycle_state == FSDPLifecycleState.SHARDED
        assert unit.finalized_after_backward
        assert _full_param_buffer_released(flat_buffer)
        assert not unit.has_pending_backward_reduce

        sharded_optim.step()
        event_names = [event.name for event in unit.runtime_events]
        assert "prepare_grad_bucket" in event_names
        assert "copy_in_grad_bucket" in event_names
        assert "reduce_grad_bucket" in event_names
        assert "wait_reduce_grad_bucket" in event_names
        assert "prepare_grad_bucket_zero_copy" not in event_names

        sharded_optim.zero_grad()
        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_multi_unit_step(
    rank: int,
    world_size: int,
    init_file: str,
    backend: str,
    device_type: str,
    reshard_after_forward: bool = False,
    use_wrap_policy: bool = False,
    forward_prefetch: bool = False,
    backward_prefetch: bool = False,
    finalize_after_backward: bool = False,
) -> None:
    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    if device_type == "cuda":
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        eager_model = copy.deepcopy(model)

        mesh = DeviceMesh(device_type, torch.arange(world_size))
        if use_wrap_policy:
            sharded_model = matrix_fully_shard(
                model,
                mesh,
                wrap_policy=lambda module: isinstance(module, nn.Linear),
                reshard_after_forward=reshard_after_forward,
                forward_prefetch=forward_prefetch,
                backward_prefetch=backward_prefetch,
                finalize_after_backward=finalize_after_backward,
            )
        else:
            sharded_model = _matrix_fully_shard_linear_units(
                model,
                mesh,
                reshard_after_forward=reshard_after_forward,
                forward_prefetch=forward_prefetch,
                backward_prefetch=backward_prefetch,
                finalize_after_backward=finalize_after_backward,
            )
        units = [sharded_model[0]._matrix_fsdp_param_group, sharded_model[2]._matrix_fsdp_param_group]
        for unit in units:
            assert unit.rank == rank
            assert unit.world_size == world_size
            assert unit.flat_buffer is not None
            _assert_unit_layout_matches_flat_buffer(unit, rank)

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)
        assert len(sharded_optim.runtime_param_groups) == 2

        torch.manual_seed(1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        if reshard_after_forward:
            assert all(unit.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED for unit in units)
            assert all(_full_param_buffer_released(unit.flat_buffer) for unit in units)
            if forward_prefetch:
                _assert_forward_prefetch_events(units)
        else:
            assert all(unit.lifecycle_state == FSDPLifecycleState.UNSHARDED for unit in units)
            assert all(unit.flat_buffer.full_buffer is not None for unit in units)

        sharded_loss.backward()
        if backward_prefetch:
            _assert_backward_prefetch_events(units)
        if finalize_after_backward:
            assert all(unit.lifecycle_state == FSDPLifecycleState.SHARDED for unit in units)
            assert all(unit.finalized_after_backward for unit in units)
            assert all(_full_param_buffer_released(unit.flat_buffer) for unit in units)
            for unit in units:
                _assert_finalize_after_backward_events(unit)
            assert _event_sequence(units[1], "pre_backward_unshard") < _event_sequence(
                units[0],
                "pre_backward_unshard",
            )
            assert _event_sequence(units[1], "finalize_backward") < _event_sequence(
                units[0],
                "finalize_backward",
            )
        else:
            assert all(unit.lifecycle_state == FSDPLifecycleState.UNSHARDED for unit in units)
            assert all(unit.flat_buffer.full_buffer is not None for unit in units)

        sharded_optim.step()
        assert all(unit.lifecycle_state == FSDPLifecycleState.SHARDED for unit in units)
        assert all(_full_param_buffer_released(unit.flat_buffer) for unit in units)

        sharded_optim.zero_grad()
        assert all(param.grad is None for param in sharded_model.parameters())

        for unit in units:
            unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            assert eager_param.shape == sharded_param.shape
            torch.testing.assert_close(eager_param, sharded_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_gradient_checkpoint_step(
    rank: int,
    world_size: int,
    init_file: str,
    use_reentrant: bool,
    wrap_blocks: bool,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(13)
        base_model = _CheckpointModel(use_reentrant=use_reentrant).to(device)
        eager_model = copy.deepcopy(base_model)
        sharded_model = copy.deepcopy(base_model)

        mesh = DeviceMesh("cpu", torch.arange(world_size))
        if wrap_blocks:
            sharded_model = matrix_fully_shard(
                sharded_model,
                mesh,
                wrap_policy=lambda module: isinstance(module, _CheckpointBlock),
                reshard_after_forward=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
            )
            units = [sharded_model.b1._matrix_fsdp_param_group, sharded_model.b2._matrix_fsdp_param_group]
        else:
            sharded_model = matrix_fully_shard(
                sharded_model,
                mesh,
                reshard_after_forward=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
            )
            units = [sharded_model._matrix_fsdp_param_group]

        for unit in units:
            assert unit.rank == rank
            assert unit.world_size == world_size
            assert unit.flat_buffer is not None
            _assert_unit_layout_matches_flat_buffer(unit, rank)

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.01)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.01), sharded_model)

        torch.manual_seed(17)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)
        eager_x = x.detach().clone().requires_grad_()
        sharded_x = x.detach().clone().requires_grad_()

        eager_loss = (eager_model(eager_x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(sharded_x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        sharded_loss.backward()
        sharded_optim.step()

        torch.testing.assert_close(sharded_x.grad, eager_x.grad)
        for unit in units:
            assert unit.lifecycle_state == FSDPLifecycleState.SHARDED
            assert _full_param_buffer_released(unit.flat_buffer)
            event_names = [event.name for event in unit.runtime_events]
            assert "pre_backward_unshard" in event_names
            assert "finalize_backward" in event_names
            assert "prepare_grad_bucket" in event_names
            assert "reduce_grad_bucket" in event_names
            assert "wait_reduce_grad_bucket" in event_names
            unit.unshard()

        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            assert eager_param.shape == sharded_param.shape
            torch.testing.assert_close(eager_param, sharded_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_step(rank, world_size, init_file, "gloo", "cpu")


def _run_two_rank_cuda_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_step(rank, world_size, init_file, "nccl", "cuda")


def _run_two_rank_cpu_reshard_after_forward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_step(rank, world_size, init_file, "gloo", "cpu", reshard_after_forward=True)


def _run_two_rank_cuda_reshard_after_forward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_step(rank, world_size, init_file, "nccl", "cuda", reshard_after_forward=True)


def _run_two_rank_cpu_finalize_after_backward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        finalize_after_backward=True,
    )


def _run_two_rank_cpu_bucket_reduce_scatter_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        finalize_after_backward=True,
        backward_reduce_strategy="bucket_reduce_scatter",
    )


def _run_two_rank_cpu_fsdp2_chunk_bucket_reduce_scatter_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        use_fsdp2_chunk_plan=True,
        reshard_after_forward=True,
        finalize_after_backward=True,
        backward_reduce_strategy="bucket_reduce_scatter",
    )


def _run_two_rank_mixed_precision_step(
    rank: int,
    world_size: int,
    init_file: str,
    backend: str,
    device_type: str,
) -> None:
    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    if device_type == "cuda":
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        eager_model = copy.deepcopy(model)

        mesh = DeviceMesh(device_type, torch.arange(world_size))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        )
        sharded_model = matrix_fully_shard(
            model,
            mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None
        assert flat_buffer.local_shard.dtype == torch.float32

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        torch.manual_seed(1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            eager_out = eager_model(x)
        eager_loss = (eager_out.float() - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        assert sharded_loss.dtype == torch.float32
        torch.testing.assert_close(sharded_loss, eager_loss)
        sharded_loss.backward()
        assert unit.lifecycle_state == FSDPLifecycleState.SHARDED
        assert flat_buffer.local_grad_shard is not None
        assert flat_buffer.local_grad_shard.dtype == torch.float32

        sharded_optim.step()
        sharded_optim.zero_grad()

        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(sharded_param.float(), eager_param, rtol=2e-2, atol=2e-2)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_mixed_precision_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_mixed_precision_step(rank, world_size, init_file, "gloo", "cpu")


def _run_two_rank_cuda_mixed_precision_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_mixed_precision_step(rank, world_size, init_file, "nccl", "cuda")


def _run_two_rank_no_sync_step(
    rank: int,
    world_size: int,
    init_file: str,
    backend: str,
    device_type: str,
    *,
    reshard_after_forward: bool = False,
    backward_reduce_strategy: str = "flat",
    mixed_precision: bool = False,
    use_zero_copy_grad_bucket: bool = True,
    use_api_defaults: bool = False,
) -> None:
    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    if device_type == "cuda":
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        eager_model = copy.deepcopy(model)
        mp_policy = (
            MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )
            if mixed_precision
            else MixedPrecisionPolicy()
        )

        mesh = DeviceMesh(device_type, torch.arange(world_size))
        if use_api_defaults:
            sharded_model = matrix_fully_shard(model, mesh, mp_policy=mp_policy)
            reshard_after_forward = True
            backward_reduce_strategy = "bucket_reduce_scatter"
            use_zero_copy_grad_bucket = False
        else:
            sharded_model = matrix_fully_shard(
                model,
                mesh,
                reshard_after_forward=reshard_after_forward,
                backward_reduce_strategy=backward_reduce_strategy,
                mp_policy=mp_policy,
                use_zero_copy_grad_bucket=use_zero_copy_grad_bucket,
            )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None
        if use_api_defaults:
            _assert_default_fast_path_contract(unit)
        _assert_unit_layout_matches_flat_buffer(unit, rank)

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        torch.manual_seed(rank + 11)
        x1 = torch.randn(3, 4, device=device)
        y1 = torch.randn(3, 2, device=device)
        x2 = torch.randn(3, 4, device=device)
        y2 = torch.randn(3, 2, device=device)

        if mixed_precision:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                eager_out1 = eager_model(x1)
            eager_loss1 = (eager_out1.float() - y1).pow(2).mean()
        else:
            eager_loss1 = (eager_model(x1) - y1).pow(2).mean()
        eager_loss1.backward()
        if mixed_precision:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                eager_out2 = eager_model(x2)
            eager_loss2 = (eager_out2.float() - y2).pow(2).mean()
        else:
            eager_loss2 = (eager_model(x2) - y2).pow(2).mean()
        eager_loss2.backward()
        averaged_module_order_grad = _flatten_param_grads(eager_model)
        dist.all_reduce(averaged_module_order_grad, group=unit.group)
        averaged_module_order_grad.div_(world_size)
        averaged_runtime_order_grad = _flatten_param_grads_by_fqns(
            eager_model,
            tuple(managed_param.fqn for managed_param in unit.managed_params),
        )
        dist.all_reduce(averaged_runtime_order_grad, group=unit.group)
        averaged_runtime_order_grad.div_(world_size)
        eager_flat_before_step = _flatten_tensors([param.detach() for param in eager_model.parameters()])
        eager_flat_after_step = eager_flat_before_step - 0.1 * averaged_module_order_grad
        _set_param_grads_from_flat(eager_model, averaged_module_order_grad)
        eager_optim.step()

        with sharded_model.no_sync():
            sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
            torch.testing.assert_close(sharded_loss1, eager_loss1)
            sharded_loss1.backward()

        assert unit.runtime_state == FSDPRuntimeState.BACKWARD_DEFERRED
        assert unit.state_dict()["runtime_state"] == FSDPRuntimeState.BACKWARD_DEFERRED
        if reshard_after_forward:
            assert unit.lifecycle_state == FSDPLifecycleState.SHARDED
            assert _full_param_buffer_released(flat_buffer)
        else:
            assert unit.lifecycle_state == FSDPLifecycleState.UNSHARDED
            assert flat_buffer.full_buffer is not None
        accumulated_after_first = _clone_deferred_grad_payload(unit)
        assert accumulated_after_first is not None

        sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss2, eager_loss2)
        if reshard_after_forward:
            assert unit.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED
            assert _clone_deferred_grad_payload(unit) is not None
        sharded_loss2.backward()

        accumulated_after_second = _clone_deferred_grad_payload(unit)
        assert accumulated_after_second is not None
        if accumulated_after_second.shape == accumulated_after_first.shape:
            if not (
                backward_reduce_strategy == "bucket_reduce_scatter"
                and (mixed_precision or not use_zero_copy_grad_bucket)
            ):
                assert (accumulated_after_second - accumulated_after_first).abs().sum().item() > 0
        sharded_optim.step()
        assert unit.lifecycle_state == FSDPLifecycleState.SHARDED
        assert flat_buffer.local_grad_shard is not None

        actual_local_grad = _flatten_local_grads_by_shard_order(flat_buffer)
        expected_local_grad = _pack_full_tensor_by_segments(
            averaged_runtime_order_grad,
            flat_buffer.plan.local_segments(rank),
        )
        torch.testing.assert_close(actual_local_grad, expected_local_grad)

        event_names = [event.name for event in unit.runtime_events]
        assert "no_sync_enter" in event_names
        assert "no_sync_exit" in event_names
        if backward_reduce_strategy == "bucket_reduce_scatter":
            assert (
                "reuse_grad_bucket_for_accumulation" in event_names
                or "prepare_grad_bucket_zero_copy" in event_names
                or "copy_in_grad_bucket_for_accumulation" in event_names
            )
            assert "wait_reduce_grad_bucket" in event_names
        else:
            assert (
                "reuse_full_grad_buffer_for_accumulation" in event_names
                or "prepare_full_grad_buffer" in event_names
            )

        sharded_optim.zero_grad()
        unit.unshard()
        sharded_full_params = _flatten_tensors([param.detach() for param in sharded_model.parameters()])
        eager_full_params = _flatten_tensors([param.detach() for param in eager_model.parameters()])
        torch.testing.assert_close(eager_full_params, eager_flat_after_step)
        if mixed_precision:
            torch.testing.assert_close(sharded_full_params.float(), eager_flat_after_step, rtol=2e-2, atol=2e-2)
        else:
            torch.testing.assert_close(sharded_full_params, eager_flat_after_step)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_no_sync_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_no_sync_step(rank, world_size, init_file, "gloo", "cpu")


def _run_two_rank_cpu_no_sync_reshard_after_forward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_no_sync_step(rank, world_size, init_file, "gloo", "cpu", reshard_after_forward=True)


def _run_two_rank_cpu_no_sync_bucket_reduce_scatter_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_no_sync_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        backward_reduce_strategy="bucket_reduce_scatter",
    )


def _run_two_rank_cpu_mixed_precision_no_sync_bucket_reduce_scatter_step(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    _run_two_rank_no_sync_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        backward_reduce_strategy="bucket_reduce_scatter",
        mixed_precision=True,
    )


def _run_two_rank_cpu_default_fast_path_no_sync_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_no_sync_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        use_api_defaults=True,
    )


def _run_two_rank_cuda_finalize_after_backward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_step(
        rank,
        world_size,
        init_file,
        "nccl",
        "cuda",
        reshard_after_forward=True,
        finalize_after_backward=True,
    )


def _run_two_rank_cpu_multi_unit_reshard_after_forward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(rank, world_size, init_file, "gloo", "cpu", reshard_after_forward=True)


def _run_two_rank_cuda_multi_unit_reshard_after_forward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(rank, world_size, init_file, "nccl", "cuda", reshard_after_forward=True)


def _run_two_rank_cpu_multi_unit_forward_prefetch_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        forward_prefetch=True,
    )


def _run_two_rank_cuda_multi_unit_forward_prefetch_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "nccl",
        "cuda",
        reshard_after_forward=True,
        forward_prefetch=True,
    )


def _run_two_rank_cpu_multi_unit_backward_prefetch_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        backward_prefetch=True,
    )


def _run_two_rank_cuda_multi_unit_backward_prefetch_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "nccl",
        "cuda",
        reshard_after_forward=True,
        backward_prefetch=True,
    )


def _run_two_rank_cpu_multi_unit_finalize_after_backward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        finalize_after_backward=True,
    )


def _run_two_rank_cuda_multi_unit_finalize_after_backward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "nccl",
        "cuda",
        reshard_after_forward=True,
        finalize_after_backward=True,
    )


def _run_two_rank_cpu_wrap_policy_reshard_after_forward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        use_wrap_policy=True,
    )


def _run_two_rank_cuda_wrap_policy_reshard_after_forward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "nccl",
        "cuda",
        reshard_after_forward=True,
        use_wrap_policy=True,
    )


def _run_two_rank_cpu_wrap_policy_forward_prefetch_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        use_wrap_policy=True,
        forward_prefetch=True,
    )


def _run_two_rank_cuda_wrap_policy_forward_prefetch_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "nccl",
        "cuda",
        reshard_after_forward=True,
        use_wrap_policy=True,
        forward_prefetch=True,
    )


def _run_two_rank_cpu_wrap_policy_backward_prefetch_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        use_wrap_policy=True,
        backward_prefetch=True,
    )


def _run_two_rank_cuda_wrap_policy_backward_prefetch_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "nccl",
        "cuda",
        reshard_after_forward=True,
        use_wrap_policy=True,
        backward_prefetch=True,
    )


def _run_two_rank_cpu_wrap_policy_finalize_after_backward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        reshard_after_forward=True,
        use_wrap_policy=True,
        finalize_after_backward=True,
    )


def _run_two_rank_cuda_wrap_policy_finalize_after_backward_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_multi_unit_step(
        rank,
        world_size,
        init_file,
        "nccl",
        "cuda",
        reshard_after_forward=True,
        use_wrap_policy=True,
        finalize_after_backward=True,
    )


def _run_two_rank_adamw_optimizer_state_step(
    rank: int,
    world_size: int,
    init_file: str,
    backend: str,
    device_type: str,
) -> None:
    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    if device_type == "cuda":
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        mesh = DeviceMesh(device_type, torch.arange(world_size))
        sharded_model = matrix_fully_shard(
            model,
            mesh,
            reshard_after_forward=True,
            finalize_after_backward=True,
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None

        optimizer = MatrixFSDPOptimizer(torch.optim.AdamW(sharded_model.parameters(), lr=0.01), sharded_model)

        torch.manual_seed(1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)
        loss = (sharded_model(x) - y).pow(2).mean()
        loss.backward()
        optimizer.step()

        optimizer.validate_local_state_shapes()
        summary = optimizer.local_state_summary()
        assert summary["param_numel"] == flat_buffer.local_numel
        assert summary["tensor_state_numel_by_name"]["exp_avg"] == flat_buffer.local_numel
        assert summary["tensor_state_numel_by_name"]["exp_avg_sq"] == flat_buffer.local_numel
        assert summary["tensor_state_numel"] == 2 * flat_buffer.local_numel
        assert optimizer.state_dtensors == {}
        assert optimizer.local_state_dtensors() == {}
        assert optimizer.state_objects
        state_object_numel_by_name = {}
        managed_params_by_fqn = {mp.fqn: mp for mp in unit.managed_params}
        for states in optimizer.state_objects.values():
            assert "exp_avg" in states
            assert "exp_avg_sq" in states
        for fqn, states in optimizer.state_objects.items():
            expected_placement = flat_buffer.param_matrix_shard(managed_params_by_fqn[fqn])
            for name, state_object in states.items():
                assert isinstance(state_object, MatrixShardedState)
                assert state_object.dtensor is None
                assert not state_object.uses_dtensor
                assert state_object.placement == expected_placement
                state_object_numel_by_name[name] = (
                    state_object_numel_by_name.get(name, 0) + state_object.local_tensor.numel()
                )
        assert state_object_numel_by_name["exp_avg"] == flat_buffer.local_numel
        assert state_object_numel_by_name["exp_avg_sq"] == flat_buffer.local_numel

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_matrix_owner_muon_step(
    rank: int,
    world_size: int,
    init_file: str,
    backend: str,
    device_type: str,
) -> None:
    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    if device_type == "cuda":
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")

        torch.manual_seed(0)
        model = _make_muon_model().to(device)
        eager_model = copy.deepcopy(model)
        shard_hints = {
            "0.weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
            "2.weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
        }

        def group_planner(params, planner_world_size):
            return auto_group_plan(params, planner_world_size, policy="muon_full_matrix")

        mesh = DeviceMesh(device_type, torch.arange(world_size))
        sharded_model = matrix_fully_shard(
            model,
            mesh,
            group_planner=group_planner,
            shard_hints=shard_hints,
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None
        assert unit.planner_evaluation is not None
        assert unit.planner_evaluation.policy == "muon_full_matrix"

        owner_by_fqn = {}
        for managed_param in unit.managed_params:
            owner_ranks = unit.owner_ranks(managed_param.fqn)
            assert len(owner_ranks) == 1
            owner_by_fqn[managed_param.fqn] = owner_ranks[0]
            shard_sizes = flat_buffer.param_shard_sizes(managed_param)
            assert sum(1 for shard_size in shard_sizes if shard_size > 0) == 1
            assert shard_sizes[owner_ranks[0]] == managed_param.numel

        eager_optim = torch.optim.Muon(
            eager_model.parameters(),
            lr=0.03,
            momentum=0.5,
            ns_steps=2,
            weight_decay=0.0,
            adjust_lr_fn="match_rms_adamw",
        )
        sharded_optim = MatrixFSDPOptimizer(
            torch.optim.Muon(
                _local_muon_params(sharded_model),
                lr=0.03,
                momentum=0.5,
                ns_steps=2,
                weight_decay=0.0,
                adjust_lr_fn="match_rms_adamw",
            ),
            sharded_model,
        )

        torch.manual_seed(rank + 1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        averaged_grad = _flatten_param_grads(eager_model)
        dist.all_reduce(averaged_grad, group=unit.group)
        averaged_grad.div_(world_size)
        _set_param_grads_from_flat(eager_model, averaged_grad)

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()
        assert unit.lifecycle_state == FSDPLifecycleState.SHARDED
        assert _full_param_buffer_released(flat_buffer)
        assert flat_buffer.param_data_alias_local_shard()

        eager_optim.step()
        sharded_optim.step()
        sharded_optim.validate_local_state_shapes()
        assert unit.lifecycle_state == FSDPLifecycleState.SHARDED
        assert _full_param_buffer_released(flat_buffer)
        assert flat_buffer.param_data_alias_local_shard()

        if owner_by_fqn.get("0.weight") == rank or owner_by_fqn.get("2.weight") == rank:
            assert sharded_optim.state_objects
            for states in sharded_optim.state_objects.values():
                assert "momentum_buffer" in states
        else:
            assert not sharded_optim.state_objects

        unit.unshard()
        for name, eager_tensor in eager_model.state_dict().items():
            torch.testing.assert_close(sharded_model.state_dict()[name], eager_tensor, rtol=1e-5, atol=1e-6)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_matrix_owner_muon_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_matrix_owner_muon_step(rank, world_size, init_file, "gloo", "cpu")


def _run_two_rank_cuda_matrix_owner_muon_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_matrix_owner_muon_step(rank, world_size, init_file, "nccl", "cuda")


def _run_two_rank_cpu_adamw_optimizer_state_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_adamw_optimizer_state_step(rank, world_size, init_file, "gloo", "cpu")


def _run_two_rank_cuda_adamw_optimizer_state_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_adamw_optimizer_state_step(rank, world_size, init_file, "nccl", "cuda")


def _run_two_rank_grad_shard_step(
    rank: int,
    world_size: int,
    init_file: str,
    backend: str,
    device_type: str,
    use_parameter_boundary_plan: bool = False,
    use_ordered_group_plan: bool = False,
    use_shard_placement_fn: bool = False,
) -> None:
    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    if device_type == "cuda":
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        eager_model = copy.deepcopy(model)

        mesh = DeviceMesh(device_type, torch.arange(world_size))
        if use_shard_placement_fn and (use_parameter_boundary_plan or use_ordered_group_plan):
            raise ValueError("shard_placement_fn path should not be combined with explicit group planners.")
        group_planner = _select_group_planner(use_parameter_boundary_plan, use_ordered_group_plan)
        if use_shard_placement_fn:
            sharded_model = fully_shard(model, mesh=mesh, shard_placement_fn=lambda _param: Shard(0))
        else:
            sharded_model = matrix_fully_shard(model, mesh, group_planner=group_planner)
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None

        plan = flat_buffer.plan
        _assert_unit_layout_matches_flat_buffer(unit, rank)
        assert flat_buffer.placement is not None
        assert flat_buffer.placement_compatibility.compatible
        assert len(flat_buffer.local_segments) <= 1
        if use_parameter_boundary_plan:
            assert unit.global_layout != unit.group_layout
            assert unit.state_dict()["layout_flat_reordered"]
        if use_shard_placement_fn:
            assert unit.param_registry.param("0.weight").shard_hint.split_granularity == "row_block"
            assert unit.param_registry.param("0.bias").shard_hint.split_granularity == "block"
            assert unit.planner_result is not None
            assert unit.planner_result.planner_name is not None

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = torch.optim.SGD(sharded_model.parameters(), lr=0.1)

        torch.manual_seed(rank + 1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        averaged_module_order_grad = _flatten_param_grads(eager_model)
        dist.all_reduce(averaged_module_order_grad, group=unit.group)
        averaged_module_order_grad.div_(world_size)
        averaged_runtime_order_grad = _flatten_param_grads_by_fqns(
            eager_model,
            tuple(managed_param.fqn for managed_param in unit.managed_params),
        )
        dist.all_reduce(averaged_runtime_order_grad, group=unit.group)
        averaged_runtime_order_grad.div_(world_size)

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()
        if unit.lifecycle_state != FSDPLifecycleState.SHARDED:
            unit.finalize_backward()

        if flat_buffer.placement is not None:
            assert flat_buffer.local_grad_shard_dtensor is None
            assert flat_buffer.sharded_grad is flat_buffer.grad_state
            assert flat_buffer.grad_state is not None
            assert flat_buffer.grad_state.has_same_data_ptr(flat_buffer.local_grad_shard)
            assert flat_buffer.grad_state.placement == flat_buffer.placement
        actual_local_grad = _flatten_local_grads_by_shard_order(flat_buffer)
        expected_local_grad = _pack_full_tensor_by_segments(averaged_runtime_order_grad, plan.local_segments(rank))
        torch.testing.assert_close(actual_local_grad, expected_local_grad)

        eager_flat_before_step = _flatten_tensors([param.detach() for param in eager_model.parameters()])
        eager_flat_after_step = eager_flat_before_step - 0.1 * averaged_module_order_grad

        _set_param_grads_from_flat(eager_model, averaged_module_order_grad)
        eager_optim.step()
        sharded_optim.step()

        unit.unshard()
        sharded_full_params = _flatten_tensors([param.detach() for param in sharded_model.parameters()])
        eager_full_params = _flatten_tensors([param.detach() for param in eager_model.parameters()])
        torch.testing.assert_close(eager_full_params, eager_flat_after_step)
        torch.testing.assert_close(sharded_full_params, eager_flat_after_step)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_grad_shard_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_grad_shard_step(rank, world_size, init_file, "gloo", "cpu")


def _run_two_rank_cuda_grad_shard_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_grad_shard_step(rank, world_size, init_file, "nccl", "cuda")


def _run_two_rank_cpu_parameter_boundary_grad_shard_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_grad_shard_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        use_parameter_boundary_plan=True,
    )


def _run_two_rank_cuda_parameter_boundary_grad_shard_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_grad_shard_step(
        rank,
        world_size,
        init_file,
        "nccl",
        "cuda",
        use_parameter_boundary_plan=True,
    )


def _run_two_rank_cpu_ordered_group_grad_shard_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_grad_shard_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        use_ordered_group_plan=True,
    )


def _run_two_rank_cpu_shard_placement_fn_shard0_grad_shard_step(
    rank: int,
    world_size: int,
    init_file: str,
) -> None:
    _run_two_rank_grad_shard_step(
        rank,
        world_size,
        init_file,
        "gloo",
        "cpu",
        use_shard_placement_fn=True,
    )


def _run_two_rank_cuda_ordered_group_grad_shard_step(rank: int, world_size: int, init_file: str) -> None:
    _run_two_rank_grad_shard_step(
        rank,
        world_size,
        init_file,
        "nccl",
        "cuda",
        use_ordered_group_plan=True,
    )


def _run_four_rank_cpu_2d_mesh_grad_shard_step(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        eager_model = copy.deepcopy(model)

        mesh = DeviceMesh(
            "cpu",
            torch.arange(world_size).reshape(2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        sharded_model = fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="dp_shard", replicate="dp_replicate"),
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None
        assert unit.rank == rank % 2
        assert unit.world_size == 2
        assert unit.replicate_world_size == 2
        _assert_unit_layout_matches_flat_buffer(unit, unit.rank)

        assert flat_buffer.local_shard_dtensor is None
        assert flat_buffer.sharded_param is flat_buffer.param_state
        assert flat_buffer._local_shard_fallback is flat_buffer.local_shard
        assert flat_buffer.param_state.mesh_metadata["mesh_dim_names"] == ("dp_replicate", "dp_shard")
        assert flat_buffer.param_state.mesh_metadata["shard_mesh_dim_name"] == "dp_shard"
        assert flat_buffer.param_state.mesh_metadata["replicate_mesh_dim_name"] == "dp_replicate"
        assert flat_buffer.param_state.placement == flat_buffer.placement

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = torch.optim.SGD(sharded_model.parameters(), lr=0.1)

        torch.manual_seed(rank + 1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        averaged_full_grad = _flatten_param_grads(eager_model)
        dist.all_reduce(averaged_full_grad)
        averaged_full_grad.div_(world_size)

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()
        if unit.lifecycle_state != FSDPLifecycleState.SHARDED:
            unit.finalize_backward()

        assert flat_buffer.local_grad_shard_dtensor is None
        assert flat_buffer.sharded_grad is flat_buffer.grad_state
        assert flat_buffer.grad_state is not None
        assert flat_buffer._local_grad_shard_fallback is flat_buffer.local_grad_shard
        assert flat_buffer.grad_state.placement == flat_buffer.placement

        actual_local_grad = _flatten_local_grads_by_shard_order(flat_buffer)
        expected_local_grad = _pack_full_tensor_by_segments(
            averaged_full_grad,
            flat_buffer.plan.local_segments(unit.rank),
        )
        torch.testing.assert_close(actual_local_grad, expected_local_grad)

        eager_flat_before_step = _flatten_tensors([param.detach() for param in eager_model.parameters()])
        eager_flat_after_step = eager_flat_before_step - 0.1 * averaged_full_grad

        _set_param_grads_from_flat(eager_model, averaged_full_grad)
        eager_optim.step()
        sharded_optim.step()

        unit.unshard()
        sharded_full_params = _flatten_tensors([param.detach() for param in sharded_model.parameters()])
        eager_full_params = _flatten_tensors([param.detach() for param in eager_model.parameters()])
        torch.testing.assert_close(eager_full_params, eager_flat_after_step)
        torch.testing.assert_close(sharded_full_params, eager_flat_after_step)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_four_rank_cpu_3d_mesh_extra_tp_grad_shard_step(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        eager_model = copy.deepcopy(model)

        mesh = DeviceMesh(
            "cpu",
            torch.arange(world_size).reshape(2, 2, 1),
            mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
        )
        sharded_model = fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="dp_shard", replicate="dp_replicate"),
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None
        assert unit.rank == rank % 2
        assert unit.world_size == 2
        assert unit.replicate_world_size == 2
        _assert_unit_layout_matches_flat_buffer(unit, unit.rank)

        assert unit.device_mesh_metadata["shape"] == (2, 2, 1)
        assert unit.device_mesh_metadata["mesh_dim_names"] == ("dp_replicate", "dp_shard", "tp")
        assert unit.device_mesh_metadata["shard_mesh_dim"] == 1
        assert unit.device_mesh_metadata["shard_mesh_dim_name"] == "dp_shard"
        assert unit.device_mesh_metadata["replicate_mesh_dim"] == 0
        assert unit.device_mesh_metadata["replicate_mesh_dim_name"] == "dp_replicate"

        assert flat_buffer.local_shard_dtensor is None
        assert flat_buffer.sharded_param is flat_buffer.param_state
        assert flat_buffer._local_shard_fallback is flat_buffer.local_shard
        assert flat_buffer.param_state.mesh_metadata["mesh_dim_names"] == ("dp_replicate", "dp_shard", "tp")
        assert flat_buffer.param_state.mesh_metadata["shard_mesh_dim_name"] == "dp_shard"
        assert flat_buffer.param_state.placement == flat_buffer.placement

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = torch.optim.SGD(sharded_model.parameters(), lr=0.1)

        torch.manual_seed(rank + 1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        averaged_full_grad = _flatten_param_grads(eager_model)
        dist.all_reduce(averaged_full_grad)
        averaged_full_grad.div_(world_size)

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()
        if unit.lifecycle_state != FSDPLifecycleState.SHARDED:
            unit.finalize_backward()

        assert flat_buffer.local_grad_shard_dtensor is None
        assert flat_buffer.sharded_grad is flat_buffer.grad_state
        assert flat_buffer.grad_state is not None
        assert flat_buffer._local_grad_shard_fallback is flat_buffer.local_grad_shard
        assert flat_buffer.grad_state.placement == flat_buffer.placement

        actual_local_grad = _flatten_local_grads_by_shard_order(flat_buffer)
        expected_local_grad = _pack_full_tensor_by_segments(
            averaged_full_grad,
            flat_buffer.plan.local_segments(unit.rank),
        )
        torch.testing.assert_close(actual_local_grad, expected_local_grad)

        eager_flat_before_step = _flatten_tensors([param.detach() for param in eager_model.parameters()])
        eager_flat_after_step = eager_flat_before_step - 0.1 * averaged_full_grad

        _set_param_grads_from_flat(eager_model, averaged_full_grad)
        eager_optim.step()
        sharded_optim.step()

        unit.unshard()
        sharded_full_params = _flatten_tensors([param.detach() for param in sharded_model.parameters()])
        eager_full_params = _flatten_tensors([param.detach() for param in eager_model.parameters()])
        torch.testing.assert_close(eager_full_params, eager_flat_after_step)
        torch.testing.assert_close(sharded_full_params, eager_flat_after_step)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_four_rank_cpu_2d_mesh_bucket_reduce_scatter_step(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        eager_model = copy.deepcopy(model)

        mesh = DeviceMesh(
            "cpu",
            torch.arange(world_size).reshape(2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        sharded_model = matrix_fully_shard(
            model,
            mesh,
            dp_shard_mesh_dim="dp_shard",
            dp_replicate_mesh_dim="dp_replicate",
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None
        assert unit.rank == rank % 2
        assert unit.world_size == 2
        assert unit.replicate_world_size == 2
        assert unit.planner_result is not None
        assert unit.planner_result.world_size == 2
        assert unit.state_dict()["device_mesh"]["replicate_mesh_dim_name"] == "dp_replicate"
        assert unit.state_dict()["device_mesh"]["shard_mesh_dim_name"] == "dp_shard"
        _assert_unit_layout_matches_flat_buffer(unit, unit.rank)

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        torch.manual_seed(rank + 11)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        averaged_full_grad = _flatten_param_grads(eager_model)
        dist.all_reduce(averaged_full_grad)
        averaged_full_grad.div_(world_size)
        eager_flat_before_step = _flatten_tensors([param.detach() for param in eager_model.parameters()])
        eager_flat_after_step = eager_flat_before_step - 0.1 * averaged_full_grad
        _set_param_grads_from_flat(eager_model, averaged_full_grad)
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        assert unit.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED
        sharded_loss.backward()

        assert unit.lifecycle_state == FSDPLifecycleState.SHARDED
        assert unit.finalized_after_backward
        assert not unit.has_pending_backward_reduce
        assert _full_param_buffer_released(flat_buffer)
        assert flat_buffer.local_grad_shard is not None
        event_names = [event.name for event in unit.runtime_events]
        assert "prepare_grad_bucket" in event_names
        assert "reduce_grad_bucket" in event_names
        assert "wait_reduce_grad_bucket" in event_names

        actual_local_grad = _flatten_local_grads_by_shard_order(flat_buffer)
        expected_local_grad = _pack_full_tensor_by_segments(
            averaged_full_grad,
            flat_buffer.plan.local_segments(unit.rank),
        )
        torch.testing.assert_close(actual_local_grad, expected_local_grad)

        sharded_optim.step()
        unit.unshard()
        sharded_full_params = _flatten_tensors([param.detach() for param in sharded_model.parameters()])
        eager_full_params = _flatten_tensors([param.detach() for param in eager_model.parameters()])
        torch.testing.assert_close(eager_full_params, eager_flat_after_step)
        torch.testing.assert_close(sharded_full_params, eager_flat_after_step)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_four_rank_cpu_2d_mesh_dcp_roundtrip(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        mesh = DeviceMesh(
            "cpu",
            torch.arange(world_size).reshape(2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        sharded_model = matrix_fully_shard(
            model,
            mesh,
            dp_shard_mesh_dim="dp_shard",
            dp_replicate_mesh_dim="dp_replicate",
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None

        optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=0.01)
        torch.manual_seed(rank + 17)
        x = torch.randn(3, 4, device=device)
        loss = sharded_model(x).pow(2).mean()
        loss.backward()
        optimizer.step()
        expected_optimizer_state = copy.deepcopy(optimizer.state_dict())

        checkpoint_dir = os.path.join(os.path.dirname(init_file), "matrix_fsdp_2d_mesh_dcp")
        save_matrix_dcp(sharded_model, checkpoint_dir, optimizer=optimizer)
        dist.barrier()

        metadata = _load_dcp_metadata(checkpoint_dir, rank)
        unit_metadata = metadata["units"][0]
        assert metadata["metadata"]["dcp_rank"] == rank
        assert unit_metadata["rank"] == rank % 2
        assert unit_metadata["world_size"] == 2
        assert unit_metadata["replicate_world_size"] == 2
        assert unit_metadata["dp_shard_mesh_dim"] == 1
        assert unit_metadata["dp_replicate_mesh_dim"] == 0
        assert unit_metadata["device_mesh"]["shape"] == (2, 2)
        assert unit_metadata["device_mesh"]["mesh_dim_names"] == ("dp_replicate", "dp_shard")
        assert unit_metadata["device_mesh"]["coordinate"] == (rank // 2, rank % 2)
        assert unit_metadata["device_mesh"]["shard_mesh_dim_name"] == "dp_shard"
        assert unit_metadata["device_mesh"]["replicate_mesh_dim_name"] == "dp_replicate"
        expected_payload_rank = f"shard_{rank % 2}"
        assert unit_metadata["dcp_payload_rank"] == expected_payload_rank
        assert unit_metadata["dcp_dedup_replicates"]
        assert unit_metadata["param_shard_key"] == f"{expected_payload_rank}.unit_0.param_shard"

        if rank == 0:
            import torch.distributed.checkpoint as dcp

            global_metadata = torch.load(os.path.join(checkpoint_dir, "matrix_metadata.pt"), map_location="cpu")
            assert global_metadata["metadata"]["format"] == "matrix_dcp_global_metadata"
            assert global_metadata["metadata"]["ranks"] == (0, 1, 2, 3)
            assert set(global_metadata["ranks"]) == {0, 1, 2, 3}
            assert not [
                name
                for name in os.listdir(checkpoint_dir)
                if name.startswith("matrix_metadata_rank_")
            ]

            dcp_metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata()
            param_shard_keys = {
                key
                for key in dcp_metadata.state_dict_metadata
                if key.endswith(".param_shard")
            }
            optimizer_keys = {
                key
                for key in dcp_metadata.state_dict_metadata
                if ".optimizer." in key
            }
            assert param_shard_keys == {
                "shard_0.unit_0.param_shard",
                "shard_1.unit_0.param_shard",
            }
            assert optimizer_keys
            assert all(key.startswith(("shard_0.", "shard_1.")) for key in optimizer_keys)

        torch.manual_seed(1234)
        restored_model = _make_model().to(device)
        restored_model = matrix_fully_shard(
            restored_model,
            mesh,
            dp_shard_mesh_dim="dp_shard",
            dp_replicate_mesh_dim="dp_replicate",
        )
        restored_unit = restored_model._matrix_fsdp_param_group
        restored_flat_buffer = restored_unit.flat_buffer
        assert restored_flat_buffer is not None
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.2)

        load_matrix_dcp(restored_model, checkpoint_dir, optimizer=restored_optimizer)
        torch.testing.assert_close(restored_flat_buffer.local_shard, flat_buffer.local_shard)
        _assert_torch_optimizer_state_dict_close(restored_optimizer.state_dict(), expected_optimizer_state)

        unit.unshard()
        restored_unit.unshard()
        for sharded_param, restored_param in zip(sharded_model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(sharded_param, restored_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_state_dict_roundtrip(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        eager_model = copy.deepcopy(model)
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        sharded_model = matrix_fully_shard(model, mesh)
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None

        sharded_state = matrix_state_dict(sharded_model)
        full_state = matrix_state_dict(sharded_model, full_state=True)
        assert full_state["metadata"]["state_dict_type"] == "matrix_full"
        assert set(full_state["params"]) == set(eager_model.state_dict())
        for name, eager_tensor in eager_model.state_dict().items():
            torch.testing.assert_close(full_state["params"][name], eager_tensor)

        torch.manual_seed(1234)
        restored_model = _make_model().to(device)
        restored_model = matrix_fully_shard(restored_model, mesh)
        restored_unit = restored_model._matrix_fsdp_param_group
        restored_flat_buffer = restored_unit.flat_buffer
        assert restored_flat_buffer is not None

        load_matrix_state_dict(restored_model, sharded_state)
        torch.testing.assert_close(restored_flat_buffer.local_shard, flat_buffer.local_shard)

        unit.unshard()
        restored_unit.unshard()
        for sharded_param, restored_param in zip(sharded_model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(sharded_param, restored_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_dcp_roundtrip(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        sharded_model = matrix_fully_shard(model, mesh)
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None

        checkpoint_dir = os.path.join(os.path.dirname(init_file), "matrix_fsdp_dcp")
        save_matrix_dcp(sharded_model, checkpoint_dir)
        dist.barrier()

        torch.manual_seed(1234)
        restored_model = _make_model().to(device)
        restored_model = matrix_fully_shard(restored_model, mesh)
        restored_unit = restored_model._matrix_fsdp_param_group
        restored_flat_buffer = restored_unit.flat_buffer
        assert restored_flat_buffer is not None

        load_matrix_dcp(restored_model, checkpoint_dir)
        torch.testing.assert_close(restored_flat_buffer.local_shard, flat_buffer.local_shard)

        unit.unshard()
        restored_unit.unshard()
        for sharded_param, restored_param in zip(sharded_model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(sharded_param, restored_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_dcp_style_state_dict_bridge_roundtrip(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        sharded_model = matrix_fully_shard(model, mesh)
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None
        optimizer = configure_optimizer(sharded_model, "adamw", lr=0.01)

        torch.manual_seed(rank + 13)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)
        loss = (sharded_model(x) - y).pow(2).mean()
        loss.backward()
        optimizer.step()
        expected_optimizer_state = copy.deepcopy(optimizer.state_dict())

        model_state, optim_state = matrix_get_state_dict(sharded_model, optimizer)
        assert model_state["metadata"]["state_dict_type"] == "matrix_sharded"
        assert model_state["metadata"]["num_param_groups"] == 1
        assert model_state["param_groups"][0]["rank"] == rank

        torch.manual_seed(1234)
        restored_model = _make_model().to(device)
        restored_model = matrix_fully_shard(restored_model, mesh)
        restored_unit = restored_model._matrix_fsdp_param_group
        restored_flat_buffer = restored_unit.flat_buffer
        assert restored_flat_buffer is not None
        restored_optimizer = configure_optimizer(restored_model, "adamw", lr=0.2)

        matrix_set_state_dict(
            restored_model,
            restored_optimizer,
            model_state_dict=model_state,
            optim_state_dict=optim_state,
        )
        torch.testing.assert_close(restored_flat_buffer.local_shard, flat_buffer.local_shard)
        _assert_torch_optimizer_state_dict_close(restored_optimizer.state_dict(), expected_optimizer_state)

        unit.unshard()
        restored_unit.unshard()
        for sharded_param, restored_param in zip(sharded_model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(sharded_param, restored_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_dcp_save_for_single_rank_reshard(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        sharded_model = matrix_fully_shard(model, mesh)

        save_matrix_dcp(sharded_model, checkpoint_dir)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_dcp_optimizer_roundtrip(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_model().to(device)
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        sharded_model = matrix_fully_shard(model, mesh)
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None
        _assert_default_fast_path_contract(unit)
        optimizer = MatrixFSDPOptimizer(torch.optim.AdamW(sharded_model.parameters(), lr=0.01), sharded_model)

        torch.manual_seed(rank + 1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)
        loss = (sharded_model(x) - y).pow(2).mean()
        loss.backward()
        optimizer.step()
        expected_optimizer_state = copy.deepcopy(optimizer.state_dict())

        checkpoint_dir = os.path.join(os.path.dirname(init_file), "matrix_fsdp_dcp_optimizer")
        save_matrix_dcp(sharded_model, checkpoint_dir, optimizer=optimizer)
        dist.barrier()

        torch.manual_seed(1234)
        restored_model = _make_model().to(device)
        restored_model = matrix_fully_shard(restored_model, mesh)
        restored_unit = restored_model._matrix_fsdp_param_group
        restored_flat_buffer = restored_unit.flat_buffer
        assert restored_flat_buffer is not None
        _assert_default_fast_path_contract(restored_unit)
        restored_optimizer = MatrixFSDPOptimizer(
            torch.optim.AdamW(restored_model.parameters(), lr=0.2),
            restored_model,
        )

        load_matrix_dcp(restored_model, checkpoint_dir, optimizer=restored_optimizer)
        torch.testing.assert_close(restored_flat_buffer.local_shard, flat_buffer.local_shard)
        assert restored_optimizer.state_dict()["param_groups"] == expected_optimizer_state["param_groups"]
        for param_id, state in expected_optimizer_state["state"].items():
            restored_state = restored_optimizer.state_dict()["state"][param_id]
            assert restored_state.keys() == state.keys()
            for name, value in state.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(restored_state[name], value)
                else:
                    assert restored_state[name] == value

        optimizer.zero_grad()
        restored_optimizer.zero_grad()
        torch.manual_seed(rank + 101)
        next_x = torch.randn(3, 4, device=device)
        next_y = torch.randn(3, 2, device=device)

        next_loss = (sharded_model(next_x) - next_y).pow(2).mean()
        restored_next_loss = (restored_model(next_x) - next_y).pow(2).mean()
        torch.testing.assert_close(restored_next_loss, next_loss)
        next_loss.backward()
        restored_next_loss.backward()
        optimizer.step()
        restored_optimizer.step()

        unit.unshard()
        restored_unit.unshard()
        for sharded_param, restored_param in zip(sharded_model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(restored_param, sharded_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_dcp_mixed_optimizer_roundtrip(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        model = _make_mixed_optimizer_model().to(device)
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        sharded_model = matrix_fully_shard(
            model,
            mesh,
            auto_shard_hints=True,
            auto_planner_policy="muon_shard_aware",
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        assert flat_buffer is not None
        optimizer = MatrixFSDPOptimizer.from_shard_hints(sharded_model)

        torch.manual_seed(rank + 1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)
        loss = (sharded_model(x) - y).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.validate_local_state_shapes()
        expected_optimizer_state = copy.deepcopy(optimizer.state_dict())

        checkpoint_dir = os.path.join(os.path.dirname(init_file), "matrix_fsdp_dcp_mixed_optimizer")
        save_matrix_dcp(sharded_model, checkpoint_dir, optimizer=optimizer)
        dist.barrier()

        metadata = _load_dcp_metadata(checkpoint_dir, rank)
        optimizer_metadata = metadata["optimizer"]
        assert optimizer_metadata["kind"] == "mixed_muon_adamw"
        assert optimizer_metadata["group_summary"]["optimizer"] == "mixed_muon_adamw"
        assert set(optimizer_metadata["components"]).issubset({"muon", "adamw"})
        assert optimizer_metadata["components"]

        torch.manual_seed(1234)
        restored_model = _make_mixed_optimizer_model().to(device)
        restored_model = matrix_fully_shard(
            restored_model,
            mesh,
            auto_shard_hints=True,
            auto_planner_policy="muon_shard_aware",
        )
        restored_unit = restored_model._matrix_fsdp_param_group
        restored_flat_buffer = restored_unit.flat_buffer
        assert restored_flat_buffer is not None
        restored_optimizer = MatrixFSDPOptimizer.from_shard_hints(restored_model)

        load_matrix_dcp(restored_model, checkpoint_dir, optimizer=restored_optimizer)
        torch.testing.assert_close(restored_flat_buffer.local_shard, flat_buffer.local_shard)
        _assert_mixed_optimizer_state_dict_close(restored_optimizer.state_dict(), expected_optimizer_state)
        restored_optimizer.validate_local_state_shapes()

        unit.unshard()
        restored_unit.unshard()
        for sharded_param, restored_param in zip(sharded_model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(sharded_param, restored_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_dcp_reshard_from_single_rank(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        eager_model = _make_model().to(device)

        torch.manual_seed(1234)
        restored_model = _make_model().to(device)
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        restored_model = matrix_fully_shard(restored_model, mesh)
        restored_unit = restored_model._matrix_fsdp_param_group

        load_matrix_dcp(restored_model, checkpoint_dir, allow_reshard=True)
        restored_unit.unshard()
        for eager_param, restored_param in zip(eager_model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(eager_param, restored_param)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_dcp_mixed_optimizer_reshard_from_single_rank(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        source_model = _make_mixed_optimizer_model().to(device)
        source_model = matrix_fully_shard(
            source_model,
            auto_shard_hints=True,
            auto_planner_policy="muon_shard_aware",
        )
        source_optimizer = MatrixFSDPOptimizer.from_shard_hints(source_model)
        torch.manual_seed(1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)
        loss = (source_model(x) - y).pow(2).mean()
        loss.backward()
        source_optimizer.step()
        source_optimizer.validate_local_state_shapes()
        expected_states_by_fqn = _mixed_optimizer_state_by_fqn(source_optimizer, source_model)
        source_model._matrix_fsdp_param_group.unshard()

        torch.manual_seed(1234)
        restored_model = _make_mixed_optimizer_model().to(device)
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        restored_model = matrix_fully_shard(
            restored_model,
            mesh,
            auto_shard_hints=True,
            auto_planner_policy="muon_shard_aware",
        )
        restored_unit = restored_model._matrix_fsdp_param_group
        restored_optimizer = MatrixFSDPOptimizer.from_shard_hints(restored_model)

        load_matrix_dcp(
            restored_model,
            checkpoint_dir,
            optimizer=restored_optimizer,
            allow_reshard=True,
        )
        restored_optimizer.validate_local_state_shapes()
        restored_states_by_fqn = _mixed_optimizer_state_by_fqn(restored_optimizer, restored_model)

        restored_unit.unshard()
        for source_param, restored_param in zip(source_model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(source_param, restored_param)

        for managed_param in restored_unit.managed_params:
            segments = restored_unit.rank_segments_for_param(managed_param.fqn, rank)
            if not segments:
                assert managed_param.fqn not in restored_states_by_fqn
                continue
            assert managed_param.fqn in restored_states_by_fqn, (
                rank,
                managed_param.fqn,
                sum(segment.numel for segment in segments),
                sorted(restored_states_by_fqn),
                sorted(expected_states_by_fqn),
            )
            expected_state = expected_states_by_fqn[managed_param.fqn]
            actual_state = restored_states_by_fqn[managed_param.fqn]
            for name, expected_value in expected_state.items():
                actual_value = actual_state[name]
                if not torch.is_tensor(expected_value):
                    assert actual_value == expected_value
                    continue
                if expected_value.ndim == 0:
                    torch.testing.assert_close(actual_value, expected_value)
                    continue
                expected_pieces = []
                for segment in segments:
                    start = segment.global_start - managed_param.offset
                    end = segment.global_end - managed_param.offset
                    expected_pieces.append(expected_value.reshape(-1)[start:end])
                expected_local_value = torch.cat(expected_pieces).view_as(actual_value)
                torch.testing.assert_close(actual_value, expected_local_value)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_two_rank_cpu_dcp_optimizer_reshard_from_single_rank(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cpu")

        torch.manual_seed(0)
        eager_model = _make_model().to(device)
        eager_optimizer = torch.optim.AdamW(eager_model.parameters(), lr=0.01)
        torch.manual_seed(1)
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 2, device=device)
        loss = (eager_model(x) - y).pow(2).mean()
        loss.backward()
        eager_optimizer.step()
        eager_states_by_fqn = {
            fqn: eager_optimizer.state[param]
            for fqn, param in eager_model.named_parameters()
        }

        torch.manual_seed(1234)
        restored_model = _make_model().to(device)
        mesh = DeviceMesh("cpu", torch.arange(world_size))
        restored_model = matrix_fully_shard(restored_model, mesh)
        restored_unit = restored_model._matrix_fsdp_param_group
        restored_optimizer = MatrixFSDPOptimizer(
            torch.optim.AdamW(restored_model.parameters(), lr=0.2),
            restored_model,
        )

        load_matrix_dcp(
            restored_model,
            checkpoint_dir,
            optimizer=restored_optimizer,
            allow_reshard=True,
        )
        restored_unit.unshard()
        for eager_param, restored_param in zip(eager_model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(eager_param, restored_param)

        restored_state = restored_optimizer.optimizer.state
        for managed_param in restored_unit.managed_params:
            param_state = restored_state.get(managed_param.param)
            if not restored_unit.rank_segments_for_param(managed_param.fqn, rank):
                assert param_state
                for value in param_state.values():
                    if torch.is_tensor(value) and value.ndim > 0:
                        assert value.numel() == 0
                continue
            assert param_state
            eager_state = eager_states_by_fqn[managed_param.fqn]
            for name, eager_value in eager_state.items():
                actual_value = param_state[name]
                if eager_value.ndim == 0:
                    torch.testing.assert_close(actual_value, eager_value)
                    continue
                expected_pieces = []
                for segment in restored_unit.rank_segments_for_param(managed_param.fqn, rank):
                    start = segment.global_start - managed_param.offset
                    end = segment.global_end - managed_param.offset
                    expected_pieces.append(eager_value.reshape(-1)[start:end])
                expected_value = torch.cat(expected_pieces).view_as(actual_value)
                torch.testing.assert_close(actual_value, expected_value)

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _select_group_planner(use_parameter_boundary_plan: bool, use_ordered_group_plan: bool):
    if use_parameter_boundary_plan and use_ordered_group_plan:
        raise ValueError("Select at most one group planner.")
    if use_parameter_boundary_plan:
        return parameter_boundary_plan
    if use_ordered_group_plan:
        return ordered_group_plan
    return None


@unittest.skipUnless(dist.is_available(), "torch.distributed is not available")
class MatrixFSDPDistributedTest(unittest.TestCase):
    def test_two_rank_cpu_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_init")
            mp.spawn(
                _run_two_rank_cpu_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_matrix_fully_shard_default_fast_path_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_default_fast_init")
            mp.spawn(
                _run_two_rank_default_fast_path_step,
                args=(world_size, init_file, "matrix_fully_shard"),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_fully_shard_default_fast_path_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_fully_shard_default_fast_init")
            mp.spawn(
                _run_two_rank_default_fast_path_step,
                args=(world_size, init_file, "fully_shard"),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_reshard_after_forward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_reshard_after_forward_init")
            mp.spawn(
                _run_two_rank_cpu_reshard_after_forward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_finalize_after_backward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_finalize_after_backward_init")
            mp.spawn(
                _run_two_rank_cpu_finalize_after_backward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_bucket_reduce_scatter_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_bucket_reduce_scatter_init")
            mp.spawn(
                _run_two_rank_cpu_bucket_reduce_scatter_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_fsdp2_chunk_bucket_reduce_scatter_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_fsdp2_chunk_bucket_reduce_scatter_init")
            mp.spawn(
                _run_two_rank_cpu_fsdp2_chunk_bucket_reduce_scatter_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_mixed_precision_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_mixed_precision_init")
            mp.spawn(
                _run_two_rank_cpu_mixed_precision_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_no_sync_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_no_sync_init")
            mp.spawn(
                _run_two_rank_cpu_no_sync_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_no_sync_reshard_after_forward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_no_sync_reshard_after_forward_init")
            mp.spawn(
                _run_two_rank_cpu_no_sync_reshard_after_forward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_no_sync_bucket_reduce_scatter_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_no_sync_bucket_reduce_scatter_init")
            mp.spawn(
                _run_two_rank_cpu_no_sync_bucket_reduce_scatter_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_default_fast_path_no_sync_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_default_fast_no_sync_init")
            mp.spawn(
                _run_two_rank_cpu_default_fast_path_no_sync_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_mixed_precision_no_sync_bucket_reduce_scatter_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_mixed_precision_no_sync_bucket_reduce_scatter_init")
            mp.spawn(
                _run_two_rank_cpu_mixed_precision_no_sync_bucket_reduce_scatter_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_multi_unit_reshard_after_forward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_multi_unit_reshard_after_forward_init")
            mp.spawn(
                _run_two_rank_cpu_multi_unit_reshard_after_forward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_multi_unit_forward_prefetch_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_multi_unit_forward_prefetch_init")
            mp.spawn(
                _run_two_rank_cpu_multi_unit_forward_prefetch_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_multi_unit_backward_prefetch_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_multi_unit_backward_prefetch_init")
            mp.spawn(
                _run_two_rank_cpu_multi_unit_backward_prefetch_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_multi_unit_finalize_after_backward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_multi_unit_finalize_after_backward_init")
            mp.spawn(
                _run_two_rank_cpu_multi_unit_finalize_after_backward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_wrap_policy_reshard_after_forward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_wrap_policy_reshard_after_forward_init")
            mp.spawn(
                _run_two_rank_cpu_wrap_policy_reshard_after_forward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_wrap_policy_forward_prefetch_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_wrap_policy_forward_prefetch_init")
            mp.spawn(
                _run_two_rank_cpu_wrap_policy_forward_prefetch_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_wrap_policy_backward_prefetch_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_wrap_policy_backward_prefetch_init")
            mp.spawn(
                _run_two_rank_cpu_wrap_policy_backward_prefetch_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_wrap_policy_finalize_after_backward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_wrap_policy_finalize_after_backward_init")
            mp.spawn(
                _run_two_rank_cpu_wrap_policy_finalize_after_backward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_gradient_checkpoint_root_unit_matches_eager_model(self):
        world_size = 2
        for use_reentrant in (False, True):
            with self.subTest(use_reentrant=use_reentrant):
                with tempfile.TemporaryDirectory() as tmpdir:
                    init_file = os.path.join(
                        tmpdir,
                        f"matrix_fsdp_checkpoint_root_{int(use_reentrant)}_init",
                    )
                    mp.spawn(
                        _run_two_rank_gradient_checkpoint_step,
                        args=(world_size, init_file, use_reentrant, False),
                        nprocs=world_size,
                        join=True,
                    )

    def test_two_rank_cpu_gradient_checkpoint_multi_unit_matches_eager_model(self):
        world_size = 2
        for use_reentrant in (False, True):
            with self.subTest(use_reentrant=use_reentrant):
                with tempfile.TemporaryDirectory() as tmpdir:
                    init_file = os.path.join(
                        tmpdir,
                        f"matrix_fsdp_checkpoint_multi_unit_{int(use_reentrant)}_init",
                    )
                    mp.spawn(
                        _run_two_rank_gradient_checkpoint_step,
                        args=(world_size, init_file, use_reentrant, True),
                        nprocs=world_size,
                        join=True,
                    )

    def test_two_rank_cpu_adamw_optimizer_state_is_local_shard_sized(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_adamw_optimizer_state_init")
            mp.spawn(
                _run_two_rank_cpu_adamw_optimizer_state_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_two_rank_cpu_matrix_owner_muon_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_matrix_owner_muon_init")
            mp.spawn(
                _run_two_rank_cpu_matrix_owner_muon_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_grad_shards_match_averaged_eager_grads(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_grad_init")
            mp.spawn(
                _run_two_rank_cpu_grad_shard_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_parameter_boundary_grad_shards_match_averaged_eager_grads(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_param_boundary_grad_init")
            mp.spawn(
                _run_two_rank_cpu_parameter_boundary_grad_shard_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_ordered_group_grad_shards_match_averaged_eager_grads(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_ordered_group_grad_init")
            mp.spawn(
                _run_two_rank_cpu_ordered_group_grad_shard_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_shard_placement_fn_shard0_grad_shards_match_averaged_eager_grads(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_shard_placement_fn_shard0_grad_init")
            mp.spawn(
                _run_two_rank_cpu_shard_placement_fn_shard0_grad_shard_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_four_rank_cpu_2d_mesh_grad_shards_match_world_averaged_eager_grads(self):
        world_size = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_2d_mesh_grad_init")
            mp.spawn(
                _run_four_rank_cpu_2d_mesh_grad_shard_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_four_rank_cpu_3d_mesh_extra_tp_dim_grad_shards_match_world_averaged_eager_grads(self):
        world_size = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_3d_mesh_extra_tp_grad_init")
            mp.spawn(
                _run_four_rank_cpu_3d_mesh_extra_tp_grad_shard_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_four_rank_cpu_2d_mesh_bucket_reduce_scatter_matches_world_averaged_eager_grads(self):
        world_size = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_2d_mesh_bucket_reduce_scatter_init")
            mp.spawn(
                _run_four_rank_cpu_2d_mesh_bucket_reduce_scatter_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_four_rank_cpu_2d_mesh_dcp_roundtrip_preserves_replicate_metadata(self):
        world_size = 4
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_2d_mesh_dcp_init")
            mp.spawn(
                _run_four_rank_cpu_2d_mesh_dcp_roundtrip,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_state_dict_roundtrip_restores_local_shards(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_state_dict_init")
            mp.spawn(
                _run_two_rank_cpu_state_dict_roundtrip,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_dcp_roundtrip_restores_local_shards(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_dcp_init")
            mp.spawn(
                _run_two_rank_cpu_dcp_roundtrip,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_dcp_style_state_dict_bridge_roundtrip(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_dcp_style_state_dict_bridge_init")
            mp.spawn(
                _run_two_rank_cpu_dcp_style_state_dict_bridge_roundtrip,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_dcp_optimizer_roundtrip_restores_local_state(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_dcp_optimizer_init")
            mp.spawn(
                _run_two_rank_cpu_dcp_optimizer_roundtrip,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_two_rank_cpu_dcp_mixed_optimizer_roundtrip_restores_local_state(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_dcp_mixed_optimizer_init")
            mp.spawn(
                _run_two_rank_cpu_dcp_mixed_optimizer_roundtrip,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    def test_two_rank_cpu_dcp_can_reshard_single_rank_checkpoint(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "matrix_fsdp_single_rank_dcp")
            torch.manual_seed(0)
            source_model = _make_model()
            source_model = matrix_fully_shard(source_model)
            save_matrix_dcp(source_model, checkpoint_dir, no_dist=True)

            init_file = os.path.join(tmpdir, "matrix_fsdp_dcp_reshard_init")
            mp.spawn(
                _run_two_rank_cpu_dcp_reshard_from_single_rank,
                args=(world_size, init_file, checkpoint_dir),
                nprocs=world_size,
                join=True,
            )

    def test_single_rank_cpu_dcp_can_reshard_two_rank_checkpoint(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "matrix_fsdp_two_rank_dcp")
            init_file = os.path.join(tmpdir, "matrix_fsdp_dcp_two_to_one_save_init")
            mp.spawn(
                _run_two_rank_cpu_dcp_save_for_single_rank_reshard,
                args=(world_size, init_file, checkpoint_dir),
                nprocs=world_size,
                join=True,
            )

            torch.manual_seed(0)
            eager_model = _make_model()

            torch.manual_seed(1234)
            restored_model = _make_model()
            restored_model = matrix_fully_shard(restored_model)
            restored_unit = restored_model._matrix_fsdp_param_group

            load_matrix_dcp(restored_model, checkpoint_dir, allow_reshard=True, no_dist=True)
            restored_unit.unshard()
            for eager_param, restored_param in zip(eager_model.parameters(), restored_model.parameters()):
                torch.testing.assert_close(eager_param, restored_param)

    def test_two_rank_cpu_dcp_can_reshard_single_rank_optimizer_checkpoint(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "matrix_fsdp_single_rank_optimizer_dcp")
            torch.manual_seed(0)
            source_model = _make_model()
            source_model = matrix_fully_shard(source_model)
            source_optimizer = MatrixFSDPOptimizer(torch.optim.AdamW(source_model.parameters(), lr=0.01), source_model)
            torch.manual_seed(1)
            x = torch.randn(3, 4)
            y = torch.randn(3, 2)
            loss = (source_model(x) - y).pow(2).mean()
            loss.backward()
            source_optimizer.step()
            save_matrix_dcp(source_model, checkpoint_dir, optimizer=source_optimizer, no_dist=True)

            init_file = os.path.join(tmpdir, "matrix_fsdp_dcp_optimizer_reshard_init")
            mp.spawn(
                _run_two_rank_cpu_dcp_optimizer_reshard_from_single_rank,
                args=(world_size, init_file, checkpoint_dir),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_two_rank_cpu_dcp_can_reshard_single_rank_mixed_optimizer_checkpoint(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "matrix_fsdp_single_rank_mixed_optimizer_dcp")
            torch.manual_seed(0)
            source_model = _make_mixed_optimizer_model()
            source_model = matrix_fully_shard(
                source_model,
                auto_shard_hints=True,
                auto_planner_policy="muon_shard_aware",
            )
            source_optimizer = MatrixFSDPOptimizer.from_shard_hints(source_model)
            torch.manual_seed(1)
            x = torch.randn(3, 4)
            y = torch.randn(3, 2)
            loss = (source_model(x) - y).pow(2).mean()
            loss.backward()
            source_optimizer.step()
            save_matrix_dcp(source_model, checkpoint_dir, optimizer=source_optimizer, no_dist=True)

            init_file = os.path.join(tmpdir, "matrix_fsdp_dcp_mixed_optimizer_reshard_init")
            mp.spawn(
                _run_two_rank_cpu_dcp_mixed_optimizer_reshard_from_single_rank,
                args=(world_size, init_file, checkpoint_dir),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        hasattr(torch.optim, "Muon") and dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires torch.optim.Muon, NCCL, and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_multi_unit_reshard_after_forward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_multi_unit_reshard_after_forward_init")
            mp.spawn(
                _run_two_rank_cuda_multi_unit_reshard_after_forward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_multi_unit_forward_prefetch_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_multi_unit_forward_prefetch_init")
            mp.spawn(
                _run_two_rank_cuda_multi_unit_forward_prefetch_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_multi_unit_backward_prefetch_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_multi_unit_backward_prefetch_init")
            mp.spawn(
                _run_two_rank_cuda_multi_unit_backward_prefetch_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_multi_unit_finalize_after_backward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_multi_unit_finalize_after_backward_init")
            mp.spawn(
                _run_two_rank_cuda_multi_unit_finalize_after_backward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_wrap_policy_reshard_after_forward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_wrap_policy_reshard_after_forward_init")
            mp.spawn(
                _run_two_rank_cuda_wrap_policy_reshard_after_forward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_wrap_policy_forward_prefetch_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_wrap_policy_forward_prefetch_init")
            mp.spawn(
                _run_two_rank_cuda_wrap_policy_forward_prefetch_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_wrap_policy_backward_prefetch_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_wrap_policy_backward_prefetch_init")
            mp.spawn(
                _run_two_rank_cuda_wrap_policy_backward_prefetch_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_wrap_policy_finalize_after_backward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_wrap_policy_finalize_after_backward_init")
            mp.spawn(
                _run_two_rank_cuda_wrap_policy_finalize_after_backward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_reshard_after_forward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_reshard_after_forward_init")
            mp.spawn(
                _run_two_rank_cuda_reshard_after_forward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_init")
            mp.spawn(
                _run_two_rank_cuda_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_mixed_precision_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_mixed_precision_init")
            mp.spawn(
                _run_two_rank_cuda_mixed_precision_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_adamw_optimizer_state_is_local_shard_sized(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_adamw_optimizer_state_init")
            mp.spawn(
                _run_two_rank_cuda_adamw_optimizer_state_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_matrix_owner_muon_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_matrix_owner_muon_init")
            mp.spawn(
                _run_two_rank_cuda_matrix_owner_muon_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_grad_shards_match_averaged_eager_grads(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_grad_init")
            mp.spawn(
                _run_two_rank_cuda_grad_shard_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_parameter_boundary_grad_shards_match_averaged_eager_grads(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_param_boundary_grad_init")
            mp.spawn(
                _run_two_rank_cuda_parameter_boundary_grad_shard_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_ordered_group_grad_shards_match_averaged_eager_grads(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_ordered_group_grad_init")
            mp.spawn(
                _run_two_rank_cuda_ordered_group_grad_shard_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_finalize_after_backward_step_matches_eager_model(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "matrix_fsdp_cuda_finalize_after_backward_init")
            mp.spawn(
                _run_two_rank_cuda_finalize_after_backward_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
