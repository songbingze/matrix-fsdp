import copy
import inspect
import tempfile
import unittest
from functools import partial
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.utils.checkpoint import checkpoint

from matrix_fsdp import (
    DataParallelMeshDims,
    FSDPLifecycleState,
    FSDPRuntimeState,
    MixedMuonAdamWOptimizer,
    ParamShardHint,
    PrefetchProfileResult,
    PreparedMatrixOptimizer,
    MatrixFSDPOptimizer,
    MatrixFSDPParamGroup,
    MatrixFSDPScheduler,
    MatrixFSDPSchedulerConfig,
    MatrixOptimizerConfig,
    RuntimeUnitMetadata,
    collect_param_groups,
    configure_optimizer,
    fully_shard,
    make_mixed_muon_adamw_optimizer,
    module_type_policy,
    matrix_fully_shard,
    summarize_runtime_events,
)
from matrix_fsdp.layout import LayoutSegment, ParamLayout, ParamSegment, MatrixGroupLayout
from matrix_fsdp.planner import contiguous_even_plan, hinted_ordered_group_plan, ordered_group_plan


class MatrixFSDPE2ETest(unittest.TestCase):
    def _assert_full_buffer_released_or_shrunk(self, full_buffer) -> None:
        if full_buffer is None:
            return
        self.assertEqual(full_buffer.untyped_storage().nbytes(), 0)

    def _assert_finalize_after_backward_events(self, unit) -> None:
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

    def _event_sequence(self, unit, name: str) -> int:
        for event in unit.runtime_events:
            if event.name == name:
                return event.sequence
        raise AssertionError(f"Missing runtime event {name!r}.")

    def _summary_event_sequence(self, summary, unit_index: int, name: str) -> int:
        for event in summary["events"]:
            if event["unit_index"] == unit_index and event["name"] == name:
                return event["sequence"]
        raise AssertionError(f"Missing runtime event {name!r} for unit {unit_index}.")

    def _single_rank_flat_reorder_layout(self, params, world_size):
        self.assertEqual(world_size, 1)
        first, second = params[:2]
        return MatrixGroupLayout.from_rank_segments(
            total_numel=sum(param.numel for param in params),
            rank_segments=(
                (
                    LayoutSegment(second.offset, second.end, 0),
                    LayoutSegment(first.offset, first.end, second.numel),
                ),
            ),
            params=(
                ParamLayout(
                    fqn=first.fqn,
                    global_start=first.offset,
                    global_end=first.end,
                    segments=(ParamSegment(first.fqn, 0, first.offset, first.end, second.numel),),
                ),
                ParamLayout(
                    fqn=second.fqn,
                    global_start=second.offset,
                    global_end=second.end,
                    segments=(ParamSegment(second.fqn, 0, second.offset, second.end, 0),),
                ),
            ),
        )

    def _assert_checkpoint_step_matches_eager(
        self,
        *,
        use_reentrant: bool,
        reshard_after_forward: bool,
        wrap_blocks: bool,
        use_saved_tensor_hooks: bool = True,
        use_checkpoint_wrapper: bool = False,
    ) -> None:
        class CheckpointBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = nn.Linear(4, 8)
                self.act = nn.GELU()
                self.lin2 = nn.Linear(8, 4)

            def forward(self, x):
                return self.lin2(self.act(self.lin1(x)))

        class CheckpointModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b1 = CheckpointBlock()
                self.b2 = CheckpointBlock()
                self.out = nn.Linear(4, 2)

            def forward(self, x):
                if use_checkpoint_wrapper:
                    x = self.b1(x)
                    x = self.b2(x)
                else:
                    x = checkpoint(self.b1, x, use_reentrant=use_reentrant)
                    x = checkpoint(self.b2, x, use_reentrant=use_reentrant)
                return self.out(x)

        torch.manual_seed(13)
        base_model = CheckpointModel()
        eager_model = copy.deepcopy(base_model)
        sharded_model = copy.deepcopy(base_model)
        if use_checkpoint_wrapper:
            checkpoint_impl = CheckpointImpl.REENTRANT if use_reentrant else CheckpointImpl.NO_REENTRANT
            wrapper = partial(checkpoint_wrapper, checkpoint_impl=checkpoint_impl)
            for model in (eager_model, sharded_model):
                apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=wrapper,
                    check_fn=lambda module: isinstance(module, CheckpointBlock),
                )

        if wrap_blocks:
            sharded_model = matrix_fully_shard(
                sharded_model,
                wrap_policy=module_type_policy(CheckpointBlock),
                reshard_after_forward=reshard_after_forward,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
                use_saved_tensor_hooks=use_saved_tensor_hooks,
            )
        else:
            if use_saved_tensor_hooks:
                sharded_model = fully_shard(
                    sharded_model,
                    reshard_after_forward=reshard_after_forward,
                )
            else:
                sharded_model = matrix_fully_shard(
                    sharded_model,
                    reshard_after_forward=reshard_after_forward,
                    finalize_after_backward=True,
                    backward_reduce_strategy="bucket_reduce_scatter",
                    use_saved_tensor_hooks=False,
                )

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.01)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.01), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_x = x.detach().clone().requires_grad_()
        sharded_x = x.detach().clone().requires_grad_()

        eager_loss = (eager_model(eager_x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(sharded_x) - y).pow(2).mean()
        sharded_loss.backward()
        sharded_optim.step()

        if wrap_blocks:
            sharded_model.b1._matrix_fsdp_param_group.unshard()
            sharded_model.b2._matrix_fsdp_param_group.unshard()
        else:
            sharded_model._matrix_fsdp_param_group.unshard()

        torch.testing.assert_close(sharded_loss, eager_loss)
        torch.testing.assert_close(sharded_x.grad, eager_x.grad)
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            self.assertEqual(eager_param.shape, sharded_param.shape)
            torch.testing.assert_close(eager_param, sharded_param)

    def test_fully_shard_public_api_matches_fsdp2_shape(self):
        parameters = inspect.signature(fully_shard).parameters

        self.assertEqual(
            tuple(parameters),
            (
                "module",
                "mesh",
                "reshard_after_forward",
                "shard_placement_fn",
                "mp_policy",
                "offload_policy",
                "ignored_params",
                "dp_mesh_dims",
                "optimizer_policy",
            ),
        )
        for name in tuple(parameters)[1:]:
            self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameters["reshard_after_forward"].default)
        self.assertIsNone(parameters["optimizer_policy"].default)

    def test_runtime_metadata_accepts_legacy_unit_id_constructor_alias(self):
        metadata = RuntimeUnitMetadata(
            runtime_unit_id="legacy_0",
            planner_group_id="legacy_0",
            comm_buffer_id="legacy_0",
        )

        self.assertEqual(metadata.runtime_param_group_id, "legacy_0")
        self.assertEqual(metadata.runtime_unit_id, metadata.runtime_param_group_id)

    def test_fully_shard_accepts_noop_shard_placement_fn(self):
        model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))
        seen_param_ids = []

        def shard_placement_fn(param):
            seen_param_ids.append(id(param))
            return None

        sharded_model = fully_shard(model, shard_placement_fn=shard_placement_fn)

        self.assertIs(sharded_model, model)
        self.assertEqual(seen_param_ids, [id(param) for param in model.parameters()])
        self.assertIsInstance(model._matrix_fsdp_param_group, MatrixFSDPParamGroup)

    def test_fully_shard_maps_shard_placement_fn_shard0_to_leading_dim_hints(self):
        model = nn.Sequential(nn.Linear(4, 2), nn.LayerNorm(2))

        sharded_model = fully_shard(model, shard_placement_fn=lambda _param: Shard(0))
        unit = sharded_model._matrix_fsdp_param_group

        self.assertIs(sharded_model, model)
        self.assertEqual(
            unit.param_registry.param("0.weight").shard_hint,
            ParamShardHint(split_granularity="row_block", block_shape=(1, 4)),
        )
        self.assertEqual(
            unit.param_registry.param("0.bias").shard_hint,
            ParamShardHint(split_granularity="block", block_shape=(1,)),
        )
        self.assertIsNotNone(unit.group_layout)
        self.assertEqual(unit.planner_result.runtime_mode, "matrix_shard")

    def test_fully_shard_skips_ignored_params_for_shard_placement_fn(self):
        model = nn.Sequential(nn.Linear(4, 2), nn.LayerNorm(2))
        ignored_param = model[1].weight
        seen_param_ids = []

        def shard_placement_fn(param):
            seen_param_ids.append(id(param))
            return Shard(0)

        sharded_model = fully_shard(
            model,
            shard_placement_fn=shard_placement_fn,
            ignored_params={ignored_param},
        )
        unit = sharded_model._matrix_fsdp_param_group

        self.assertNotIn(id(ignored_param), seen_param_ids)
        self.assertNotIn("1.weight", unit.param_registry.fqns)
        self.assertEqual(
            unit.param_registry.param("1.bias").shard_hint,
            ParamShardHint(split_granularity="block", block_shape=(1,)),
        )

    def test_fully_shard_optimizer_policy_enables_muon_aware_layout(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))

        sharded_model = fully_shard(model, optimizer_policy="mixed_muon_adamw")
        unit = sharded_model._matrix_fsdp_param_group

        self.assertIs(sharded_model, model)
        self.assertIsNotNone(unit.planner_evaluation)
        self.assertEqual(unit.planner_evaluation.policy, "muon_shard_aware")
        self.assertEqual(unit.planner_evaluation.name, "matrix_owner_tail_role_greedy")
        self.assertEqual(
            unit.param_registry.param("0.weight").shard_hint,
            ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
        )
        self.assertEqual(
            unit.param_registry.param("0.bias").shard_hint,
            ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
        )

    def test_fully_shard_optimizer_policy_rejects_shard_placement_fn(self):
        model = nn.Linear(4, 2)

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            fully_shard(
                model,
                optimizer_policy="mixed_muon_adamw",
                shard_placement_fn=lambda _param: Shard(0),
            )

    def test_fully_shard_optimizer_policy_respects_ignored_params(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2))
        ignored_param = model[1].weight

        sharded_model = fully_shard(
            model,
            optimizer_policy="mixed_muon_adamw",
            ignored_params={ignored_param},
        )
        unit = sharded_model._matrix_fsdp_param_group

        self.assertNotIn("1.weight", unit.param_registry.fqns)
        self.assertEqual(
            unit.param_registry.param("1.bias").shard_hint,
            ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
        )

    def test_fully_shard_optimizer_policy_rejects_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "Unknown fully_shard optimizer_policy"):
            fully_shard(nn.Linear(4, 2), optimizer_policy="unknown")

    def test_matrix_fully_shard_wrap_policy_supports_ignored_params(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        ignored_param = model[0].weight

        sharded_model = matrix_fully_shard(
            model,
            wrap_policy=lambda module: isinstance(module, nn.Linear),
            ignored_params={ignored_param},
        )
        first_unit = sharded_model[0]._matrix_fsdp_param_group
        second_unit = sharded_model[2]._matrix_fsdp_param_group

        self.assertEqual(first_unit.param_registry.fqns, ("bias",))
        self.assertNotIn("weight", first_unit.param_registry.fqns)
        self.assertEqual(second_unit.param_registry.fqns, ("weight", "bias"))
        self.assertIsNone(getattr(ignored_param, "_matrix_sharded_state", None))

    def test_frozen_param_can_be_left_unmanaged_with_ignored_params(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        model[0].bias.requires_grad_(False)
        eager_model = copy.deepcopy(model)
        ignored_param = model[0].bias

        sharded_model = fully_shard(model, ignored_params={ignored_param})
        unit = sharded_model._matrix_fsdp_param_group
        optimizer = torch.optim.SGD(sharded_model.parameters(), lr=0.1)
        eager_optimizer = torch.optim.SGD(eager_model.parameters(), lr=0.1)

        self.assertNotIn("0.bias", unit.param_registry.fqns)
        self.assertIsNone(getattr(ignored_param, "_matrix_fsdp_param_group_ref", None))

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optimizer.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()
        optimizer.step()

        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_fully_shard_rejects_non_leading_dim_shard_placement_fn_until_noncontiguous_layout_exists(self):
        with self.assertRaisesRegex(NotImplementedError, "Shard\\(0\\)"):
            fully_shard(nn.Linear(4, 2), shard_placement_fn=lambda _param: Shard(1))

    def test_fully_shard_rejects_flattened_dp_mesh_dims_until_supported(self):
        with self.assertRaisesRegex(NotImplementedError, "flattened mesh dims"):
            fully_shard(nn.Linear(4, 2), dp_mesh_dims=DataParallelMeshDims(shard=("dp", "tp")))

    def test_matrix_fully_shard_rejects_conflicting_dp_mesh_dim_apis(self):
        with self.assertRaisesRegex(ValueError, "either dp_mesh_dims"):
            matrix_fully_shard(
                nn.Linear(4, 2),
                dp_mesh_dims=DataParallelMeshDims(shard="dp_shard"),
                dp_shard_mesh_dim="dp_shard",
            )

    def test_fully_shard_uses_only_selected_dp_dims_from_3d_mesh(self):
        if dist.is_initialized():
            self.skipTest("requires owning the temporary process group")
        with tempfile.TemporaryDirectory() as tmpdir:
            dist.init_process_group("gloo", init_method=f"file://{tmpdir}/init", rank=0, world_size=1)
            try:
                mesh = DeviceMesh(
                    "cpu",
                    torch.arange(1).reshape(1, 1, 1),
                    mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
                )

                model = fully_shard(
                    nn.Linear(4, 2),
                    mesh=mesh,
                    dp_mesh_dims=DataParallelMeshDims(shard="dp_shard", replicate="dp_replicate"),
                )
                unit = model._matrix_fsdp_param_group
                flat_buffer = unit.flat_buffer
                self.assertIsNotNone(flat_buffer)

                state = unit.state_dict()
                self.assertEqual(state["device_mesh"]["mesh_dim_names"], ("dp_replicate", "dp_shard", "tp"))
                self.assertEqual(state["device_mesh"]["shard_mesh_dim"], 1)
                self.assertEqual(state["device_mesh"]["shard_mesh_dim_name"], "dp_shard")
                self.assertEqual(state["device_mesh"]["replicate_mesh_dim"], 0)
                self.assertEqual(state["device_mesh"]["replicate_mesh_dim_name"], "dp_replicate")
                self.assertEqual(unit.world_size, 1)
                self.assertEqual(unit.replicate_world_size, 1)

                placements = flat_buffer.local_shard_dtensor._spec.placements
                self.assertEqual(len(placements), 3)
                self.assertIsInstance(placements[0], Replicate)
                self.assertEqual(placements[1], flat_buffer.placement)
                self.assertIsInstance(placements[2], Replicate)
            finally:
                dist.destroy_process_group()

    def test_fully_shard_accepts_fsdp2_style_call(self):
        torch.manual_seed(0)
        model = fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        optim = torch.optim.SGD(model.parameters(), lr=0.1)

        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        optim.step()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertTrue(unit.reshard_after_forward_enabled)
        self.assertTrue(unit.forward_prefetch_enabled)
        self.assertTrue(unit.backward_prefetch_enabled)
        self.assertTrue(unit.finalize_after_backward_enabled)
        self.assertEqual(unit.backward_reduce_strategy, "bucket_reduce_scatter")

    def test_matrix_fully_shard_defaults_to_fast_fsdp2_like_runtime(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group

        self.assertIs(model._matrix_fsdp_param_group, unit)
        self.assertIsInstance(unit, MatrixFSDPParamGroup)
        self.assertTrue(unit.reshard_after_forward_enabled)
        self.assertTrue(unit.forward_prefetch_enabled)
        self.assertTrue(unit.backward_prefetch_enabled)
        self.assertTrue(unit.finalize_after_backward_enabled)
        self.assertEqual(unit.backward_reduce_strategy, "bucket_reduce_scatter")
        self.assertFalse(unit.use_saved_tensor_hooks)
        self.assertFalse(unit.use_zero_copy_grad_bucket)

    def test_fully_shard_supports_mixed_precision_policy(self):
        torch.manual_seed(0)
        model = fully_shard(
            nn.Linear(4, 2),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            ),
            reshard_after_forward=False,
        )
        unit = model._matrix_fsdp_param_group
        optim = torch.optim.SGD(model.parameters(), lr=0.1)

        self.assertEqual(unit.flat_buffer.local_shard.dtype, torch.float32)
        self.assertTrue(all(param.dtype == torch.float32 for param in model.parameters()))
        x = torch.randn(3, 4, dtype=torch.float32)
        out = model(x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(unit.flat_buffer.full_buffer.dtype, torch.bfloat16)
        self.assertTrue(all(param.dtype == torch.bfloat16 for param in model.parameters()))
        out.sum().backward()
        optim.step()

        state = unit.state_dict()
        self.assertEqual(state["mixed_precision"]["param_dtype"], "torch.bfloat16")
        self.assertEqual(state["mixed_precision"]["reduce_dtype"], "torch.float32")
        self.assertEqual(state["mixed_precision"]["output_dtype"], "torch.float32")
        self.assertEqual(unit.flat_buffer.local_shard.dtype, torch.float32)
        self.assertEqual(unit.flat_buffer.local_grad_shard.dtype, torch.float32)

    def test_mixed_precision_adamw_state_stays_master_param_dtype(self):
        torch.manual_seed(0)
        model = fully_shard(
            nn.Linear(4, 2),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            ),
            reshard_after_forward=False,
        )
        unit = model._matrix_fsdp_param_group
        optim = torch.optim.AdamW(model.parameters(), lr=0.01)

        out = model(torch.randn(3, 4))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(unit.flat_buffer.full_buffer.dtype, torch.bfloat16)
        out.square().mean().backward()
        optim.step()

        self.assertEqual(unit.flat_buffer.local_shard.dtype, torch.float32)
        self.assertEqual(unit.flat_buffer.local_grad_shard.dtype, torch.float32)
        for state in optim.state.values():
            for name in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(state[name].dtype, torch.float32)

    def test_multi_unit_mixed_precision_policy_records_each_param_group(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)),
            wrap_policy=lambda module: isinstance(module, nn.Linear),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            ),
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
        )
        optimizer = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        out = model(torch.randn(3, 4))
        self.assertEqual(out.dtype, torch.float32)
        out.square().mean().backward()
        optimizer.step()

        self.assertEqual(len(optimizer.runtime_param_groups), 2)
        for unit in optimizer.runtime_param_groups:
            flat_buffer = unit.flat_buffer
            self.assertIsNotNone(flat_buffer)
            self.assertEqual(flat_buffer.local_shard.dtype, torch.float32)
            self.assertEqual(flat_buffer.local_grad_shard.dtype, torch.float32)
            self.assertEqual(unit.state_dict()["mixed_precision"]["param_dtype"], "torch.bfloat16")
            self.assertEqual(unit.state_dict()["mixed_precision"]["reduce_dtype"], "torch.float32")
            self.assertEqual(unit.state_dict()["mixed_precision"]["output_dtype"], "torch.float32")

    def test_mixed_precision_no_sync_flat_accumulation_matches_eager(self):
        torch.manual_seed(0)
        base_model = nn.Linear(4, 2)
        eager_model = copy.deepcopy(base_model)
        sharded_model = matrix_fully_shard(
            base_model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            ),
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        self.assertIsNotNone(flat_buffer)

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x1 = torch.randn(3, 4)
        y1 = torch.randn(3, 2)
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 2)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            eager_out1 = eager_model(x1)
        eager_loss1 = (eager_out1.float() - y1).pow(2).mean()
        eager_loss1.backward()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            eager_out2 = eager_model(x2)
        eager_loss2 = (eager_out2.float() - y2).pow(2).mean()
        eager_loss2.backward()
        eager_optim.step()

        with sharded_model.no_sync():
            sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
            self.assertEqual(sharded_loss1.dtype, torch.float32)
            torch.testing.assert_close(sharded_loss1, eager_loss1)
            sharded_loss1.backward()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertTrue(unit.state_dict()["defer_backward_reduce"])

        sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss2, eager_loss2)
        sharded_loss2.backward()

        sharded_optim.step()
        sharded_optim.zero_grad()
        unit.unshard()

        event_names = [event.name for event in unit.runtime_events]
        self.assertIn("no_sync_enter", event_names)
        self.assertIn("no_sync_exit", event_names)
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(sharded_param.float(), eager_param, rtol=2e-2, atol=2e-2)

    def test_mixed_precision_bucket_no_sync_copy_in_accumulation_matches_eager(self):
        torch.manual_seed(0)
        base_model = nn.Linear(4, 2)
        eager_model = copy.deepcopy(base_model)
        sharded_model = matrix_fully_shard(
            base_model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            ),
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="bucket_reduce_scatter",
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        self.assertIsNotNone(flat_buffer)

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x1 = torch.randn(3, 4)
        y1 = torch.randn(3, 2)
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 2)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            eager_out1 = eager_model(x1)
        eager_loss1 = (eager_out1.float() - y1).pow(2).mean()
        eager_loss1.backward()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            eager_out2 = eager_model(x2)
        eager_loss2 = (eager_out2.float() - y2).pow(2).mean()
        eager_loss2.backward()
        eager_optim.step()

        with sharded_optim.no_sync():
            sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
            torch.testing.assert_close(sharded_loss1, eager_loss1)
            sharded_loss1.backward()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertTrue(unit.state_dict()["defer_backward_reduce"])
        self.assertIsNotNone(flat_buffer.grad_bucket_input)
        self.assertEqual(flat_buffer.grad_bucket_input.dtype, torch.float32)
        accumulated_after_first = flat_buffer.grad_bucket_input.clone()

        sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss2, eager_loss2)
        sharded_loss2.backward()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.UNSHARDED)
        self.assertIsNotNone(flat_buffer.grad_bucket_input)
        torch.testing.assert_close(flat_buffer.grad_bucket_input, accumulated_after_first)
        sharded_optim.step()
        self.assertFalse(unit.state_dict()["defer_backward_reduce"])
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertIsNone(flat_buffer.grad_bucket_input)
        self.assertIsNotNone(flat_buffer.local_grad_shard)
        self.assertEqual(flat_buffer.local_grad_shard.dtype, torch.float32)
        sharded_optim.zero_grad()
        unit.unshard()

        event_names = [event.name for event in unit.runtime_events]
        self.assertIn("prepare_grad_bucket_copy_in", event_names)
        self.assertIn("copy_in_grad_bucket_for_accumulation", event_names)
        self.assertIn("reuse_grad_bucket_for_accumulation", event_names)
        self.assertIn("wait_reduce_grad_bucket", event_names)
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(sharded_param.float(), eager_param, rtol=2e-2, atol=2e-2)

    def test_fully_shard_accepts_cpu_offload_policy_for_cpu_model(self):
        model = fully_shard(nn.Linear(4, 2), offload_policy=CPUOffloadPolicy(pin_memory=False))
        unit_state = model._matrix_fsdp_param_group.state_dict()

        self.assertEqual(unit_state["offload_policy"]["type"], "CPUOffloadPolicy")
        self.assertTrue(unit_state["offload_policy"]["cpu_offload"])
        self.assertFalse(unit_state["offload_policy"]["pin_memory"])

    def test_fully_shard_reshard_after_forward_none_defaults_to_training_fast_path(self):
        model = fully_shard(nn.Linear(4, 2), reshard_after_forward=None)
        self.assertTrue(model._matrix_fsdp_param_group.state_dict()["reshard_after_forward"])

    def test_reshard_after_forward_int_rejects_without_subgroup_runtime(self):
        with self.assertRaisesRegex(ValueError, "non-trivial divisor"):
            fully_shard(nn.Linear(4, 2), reshard_after_forward=2)

    def test_frozen_managed_parameter_rejects_clearly(self):
        model = nn.Linear(4, 2)
        model.bias.requires_grad_(False)
        with self.assertRaisesRegex(NotImplementedError, "require grad"):
            matrix_fully_shard(model)

    def test_shared_managed_parameter_rejects_clearly(self):
        class SharedLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.Linear(4, 4)
                self.right = nn.Linear(4, 4)
                self.right.weight = self.left.weight

            def forward(self, x):
                return self.right(self.left(x))

        with self.assertRaisesRegex(NotImplementedError, "shared parameters"):
            matrix_fully_shard(SharedLinear())

    def test_gradient_checkpoint_root_unit_matches_eager_model(self):
        for use_reentrant in (False, True):
            for reshard_after_forward in (False, True):
                with self.subTest(use_reentrant=use_reentrant, reshard_after_forward=reshard_after_forward):
                    self._assert_checkpoint_step_matches_eager(
                        use_reentrant=use_reentrant,
                        reshard_after_forward=reshard_after_forward,
                        wrap_blocks=False,
                    )

    def test_gradient_checkpoint_multi_unit_matches_eager_model(self):
        for use_reentrant in (False, True):
            for reshard_after_forward in (False, True):
                with self.subTest(use_reentrant=use_reentrant, reshard_after_forward=reshard_after_forward):
                    self._assert_checkpoint_step_matches_eager(
                        use_reentrant=use_reentrant,
                        reshard_after_forward=reshard_after_forward,
                        wrap_blocks=True,
                    )

    def test_gradient_checkpoint_multi_unit_without_saved_tensor_hooks_matches_eager_model(self):
        self._assert_checkpoint_step_matches_eager(
            use_reentrant=False,
            reshard_after_forward=True,
            wrap_blocks=True,
            use_saved_tensor_hooks=False,
        )

    def test_apply_activation_checkpointing_multi_unit_matches_eager_model(self):
        self._assert_checkpoint_step_matches_eager(
            use_reentrant=False,
            reshard_after_forward=True,
            wrap_blocks=True,
            use_saved_tensor_hooks=False,
            use_checkpoint_wrapper=True,
        )

    def test_apply_activation_checkpointing_repeated_forward_reshards_after_step(self):
        class CheckpointBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = nn.Linear(4, 8)
                self.act = nn.GELU()
                self.lin2 = nn.Linear(8, 4)

            def forward(self, x):
                return self.lin2(self.act(self.lin1(x)))

        class CheckpointModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b1 = CheckpointBlock()
                self.b2 = CheckpointBlock()

            def forward(self, x):
                return self.b2(self.b1(x))

        torch.manual_seed(17)
        model = CheckpointModel()
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, CheckpointBlock),
        )
        model = matrix_fully_shard(
            model,
            wrap_policy=module_type_policy(CheckpointBlock),
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_saved_tensor_hooks=False,
        )
        optimizer = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.01), model)
        x = torch.randn(3, 4, requires_grad=True)

        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        model(x.detach().clone().requires_grad_()).sum()

        for param_group in collect_param_groups(model):
            self.assertEqual(param_group.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)
            self.assertTrue(param_group.flat_buffer.param_data_alias_local_shard())
            full_buffer = param_group.flat_buffer.full_buffer
            self.assertTrue(full_buffer is None or full_buffer.untyped_storage().nbytes() == 0)
        self.assertEqual(optimizer.scheduler.full_param_buffer_pool.stats()["cached_buffers"], 0)

    def test_matrix_fully_shard_accepts_explicit_comm_strategies(self):
        model = matrix_fully_shard(
            nn.Linear(4, 2),
            param_gather_strategy="matrix_all_gather",
            grad_reduce_strategy="bucket_reduce_scatter",
            finalize_after_backward=True,
        )
        unit = model._matrix_fsdp_param_group

        self.assertEqual(unit.param_gather_strategy, "matrix_all_gather")
        self.assertEqual(unit.backward_reduce_strategy, "bucket_reduce_scatter")
        self.assertEqual(unit.state_dict()["param_gather_strategy"], "matrix_all_gather")
        self.assertEqual(unit.state_dict()["grad_reduce_strategy"], "bucket_reduce_scatter")

    def test_default_planner_populates_planner_result_contract(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group

        self.assertIsNotNone(unit.planner_result)
        self.assertIs(unit.planner_result, unit.planner_evaluation)
        self.assertEqual(unit.planner_result.planner_name, "contiguous_even_plan")
        self.assertEqual(unit.planner_result.world_size, unit.world_size)
        self.assertTrue(unit.planner_result.constraints_satisfied)
        self.assertIs(unit.planner_result.layout, unit.global_layout)
        self.assertIs(unit.planner_layout_contract.layout, unit.global_layout)
        self.assertIs(unit.runtime_layout_contract.layout, unit.group_layout)
        self.assertIs(unit.state_dict()["planner_result"], unit.planner_result)
        self.assertEqual(unit.state_dict()["planner_metadata"], unit.planner_result.as_metadata())
        self.assertEqual(unit.state_dict()["planner_summary"], unit.planner_result.summary())
        self.assertEqual(unit.state_dict()["planner_report"], unit.planner_result.report.as_metadata())
        self.assertEqual(
            unit.state_dict()["planner_layout_contract"],
            unit.planner_layout_contract.as_metadata(),
        )
        self.assertEqual(
            unit.state_dict()["runtime_layout_contract"],
            unit.runtime_layout_contract.as_metadata(),
        )
        self.assertEqual(unit.state_dict()["planner_layout_contract"], unit.state_dict()["runtime_layout_contract"])
        self.assertEqual(
            unit.state_dict()["planner_resource_estimate"],
            unit.planner_result.resource_estimate.as_metadata(),
        )
        self.assertEqual(unit.state_dict()["runtime_layout_policy"], "auto")
        self.assertEqual(unit.state_dict()["runtime_layout_mode"], "matrix_shard")
        self.assertFalse(unit.state_dict()["runtime_layout_requires_flat_reorder"])

    def test_matrix_fully_shard_rejects_conflicting_grad_reduce_aliases(self):
        with self.assertRaisesRegex(ValueError, "Pass only one of grad_reduce_strategy"):
            matrix_fully_shard(
                nn.Linear(4, 2),
                grad_reduce_strategy="bucket_reduce_scatter",
                backward_reduce_strategy="per_param",
                finalize_after_backward=True,
            )

    def test_matrix_fully_shard_rejects_unknown_param_gather_strategy(self):
        with self.assertRaisesRegex(ValueError, "param_gather_strategy"):
            matrix_fully_shard(nn.Linear(4, 2), param_gather_strategy="unknown")

    def test_unit_exposes_layout_queries(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))

        sharded_model = matrix_fully_shard(model)
        unit = sharded_model._matrix_fsdp_param_group

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertIsNotNone(unit.param_registry)
        self.assertIsNotNone(unit.layout)
        self.assertEqual(unit.param_registry.fqns, ("0.weight", "0.bias", "1.weight", "1.bias"))
        self.assertEqual(unit.params_for_rank(), unit.param_registry.fqns)
        self.assertEqual(unit.owner_ranks("0.weight"), (0,))
        self.assertEqual(
            unit.rank_segments_for_param("0.weight"),
            (ParamSegment("0.weight", 0, 0, model[0].weight.numel(), 0),),
        )

    def test_unit_exposes_runtime_planner_and_comm_metadata(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        state = unit.state_dict()

        self.assertEqual(state["runtime_metadata"], unit.runtime_metadata)
        self.assertEqual(state["runtime_param_group_id"], unit.runtime_metadata.runtime_param_group_id)
        self.assertEqual(state["runtime_unit_id"], unit.runtime_metadata.runtime_unit_id)
        self.assertEqual(state["planner_group_id"], unit.runtime_metadata.planner_group_id)
        self.assertEqual(state["comm_buffer_id"], unit.runtime_metadata.comm_buffer_id)
        self.assertEqual(state["runtime_unit_id"], state["planner_group_id"])
        self.assertEqual(state["runtime_unit_id"], state["comm_buffer_id"])
        self.assertTrue(state["matrix_shard_compatibility"].compatible)
        self.assertEqual(state["matrix_shard_placement"].dims, (0,))
        self.assertEqual(state["matrix_shard_placement"].local_units, (1,))

    def test_lifecycle_state_tracks_unshard_and_finalize(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Linear(4, 2),
            reshard_after_forward=False,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_cached_full_param_buffers_per_key=1,
        )

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertFalse(unit._is_unsharded)
        self.assertTrue(unit.state_dict()["param_data_alias_local_shard"])
        self.assertFalse(unit.state_dict()["param_data_alias_full_buffer"])
        x = torch.randn(3, 4)
        loss = model(x).sum()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.UNSHARDED)
        self.assertTrue(unit._is_unsharded)
        self.assertFalse(unit.state_dict()["param_data_alias_local_shard"])
        self.assertTrue(unit.state_dict()["param_data_alias_full_buffer"])
        loss.backward()
        optim.step()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertFalse(unit._is_unsharded)
        self.assertTrue(unit.state_dict()["param_data_alias_local_shard"])
        self.assertFalse(unit.state_dict()["param_data_alias_full_buffer"])
        self.assertEqual(unit.state_dict()["lifecycle_state"], FSDPLifecycleState.SHARDED)

    def test_forward_params_are_views_into_full_param_buffer(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)),
            reshard_after_forward=False,
            finalize_after_backward=False,
        )
        unit = model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        self.assertIsNotNone(flat_buffer)

        self.assertTrue(flat_buffer.param_data_alias_local_shard())
        self.assertFalse(flat_buffer.param_data_alias_full_buffer())

        out = model(torch.randn(3, 4))

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.UNSHARDED)
        self.assertIsNotNone(flat_buffer.full_buffer)
        self.assertTrue(flat_buffer.param_data_alias_full_buffer())
        self.assertFalse(flat_buffer.param_data_alias_local_shard())
        full_storage_ptr = flat_buffer.full_buffer.untyped_storage().data_ptr()
        local_storage_ptr = flat_buffer.local_shard.untyped_storage().data_ptr()
        for managed_param in unit.managed_params:
            expected = flat_buffer.full_buffer[managed_param.offset : managed_param.end].view(managed_param.shape)
            self.assertEqual(managed_param.param.data.untyped_storage().data_ptr(), full_storage_ptr)
            self.assertNotEqual(managed_param.param.data.untyped_storage().data_ptr(), local_storage_ptr)
            self.assertEqual(managed_param.param.data.data_ptr(), expected.data_ptr())
            self.assertEqual(tuple(managed_param.param.data.shape), managed_param.shape)
            torch.testing.assert_close(managed_param.param.detach(), expected)

        out.sum().backward()
        unit.finalize_backward()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertTrue(flat_buffer.param_data_alias_local_shard())
        self.assertFalse(flat_buffer.param_data_alias_full_buffer())
        self._assert_full_buffer_released_or_shrunk(flat_buffer.full_buffer)

    def test_backward_grads_accumulate_directly_into_full_grad_buffer(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Linear(4, 2),
            reshard_after_forward=False,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        unit = model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        self.assertIsNotNone(flat_buffer)

        x = torch.randn(3, 4)
        loss = model(x).sum()

        self.assertIsNotNone(flat_buffer.full_grad_buffer)
        self.assertTrue(flat_buffer.param_grads_alias_full_grad_buffer())
        loss.backward()
        self.assertTrue(flat_buffer.param_grads_alias_full_grad_buffer())
        self.assertGreater(flat_buffer.full_grad_buffer.abs().sum().item(), 0.0)

        unit.finalize_backward()

        self.assertIsNone(flat_buffer.full_grad_buffer)
        self.assertIsNotNone(flat_buffer.local_grad_shard)
        for param in model.parameters():
            if param.grad is not None:
                self.assertEqual(
                    param.grad.untyped_storage().data_ptr(),
                    flat_buffer.local_grad_shard.untyped_storage().data_ptr(),
                )

    def test_no_sync_accumulates_full_grads_until_optimizer_step(self):
        torch.manual_seed(0)
        base_model = nn.Linear(4, 2)
        eager_model = copy.deepcopy(base_model)
        sharded_model = matrix_fully_shard(
            base_model,
            reshard_after_forward=False,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x1 = torch.randn(3, 4)
        y1 = torch.randn(3, 2)
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 2)

        eager_loss1 = (eager_model(x1) - y1).pow(2).mean()
        eager_loss1.backward()
        eager_loss2 = (eager_model(x2) - y2).pow(2).mean()
        eager_loss2.backward()
        eager_optim.step()

        with sharded_model.no_sync():
            sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
            torch.testing.assert_close(sharded_loss1, eager_loss1)
            sharded_loss1.backward()
        self.assertTrue(unit.state_dict()["defer_backward_reduce"])
        self.assertIsNotNone(flat_buffer.full_grad_buffer)
        accumulated_after_first = flat_buffer.full_grad_buffer.clone()

        sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss2, eager_loss2)
        sharded_loss2.backward()

        self.assertTrue(flat_buffer.param_grads_alias_full_grad_buffer())
        self.assertGreater(
            (flat_buffer.full_grad_buffer - accumulated_after_first).abs().sum().item(),
            0.0,
        )
        sharded_optim.step()
        self.assertFalse(unit.state_dict()["defer_backward_reduce"])
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        sharded_optim.zero_grad()
        unit.unshard()

        event_names = [event.name for event in unit.runtime_events]
        self.assertIn("no_sync_enter", event_names)
        self.assertIn("no_sync_exit", event_names)
        self.assertIn("reuse_full_grad_buffer_for_accumulation", event_names)
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_optimizer_no_sync_reuses_unit_context(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        with optim.no_sync():
            model(torch.randn(3, 4)).sum().backward()

        self.assertTrue(unit.state_dict()["defer_backward_reduce"])
        optim.zero_grad()
        self.assertFalse(unit.state_dict()["defer_backward_reduce"])

    def test_optimizer_step_and_zero_grad_reject_active_no_sync(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        with optim.no_sync():
            with self.assertRaisesRegex(RuntimeError, "cannot run inside no_sync"):
                optim.step()
            with self.assertRaisesRegex(RuntimeError, "cannot run inside no_sync"):
                optim.zero_grad()

        self.assertFalse(unit.is_no_sync_active)

    def test_no_sync_supports_reshard_after_forward_accumulation(self):
        torch.manual_seed(0)
        base_model = nn.Linear(4, 2)
        eager_model = copy.deepcopy(base_model)
        sharded_model = matrix_fully_shard(
            base_model,
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x1 = torch.randn(3, 4)
        y1 = torch.randn(3, 2)
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 2)

        eager_loss1 = (eager_model(x1) - y1).pow(2).mean()
        eager_loss1.backward()
        eager_loss2 = (eager_model(x2) - y2).pow(2).mean()
        eager_loss2.backward()
        eager_optim.step()

        with sharded_model.no_sync():
            sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
            torch.testing.assert_close(sharded_loss1, eager_loss1)
            sharded_loss1.backward()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertTrue(unit.state_dict()["defer_backward_reduce"])
        self._assert_full_buffer_released_or_shrunk(flat_buffer.full_buffer)
        self.assertIsNotNone(flat_buffer.full_grad_buffer)
        accumulated_after_first = flat_buffer.full_grad_buffer.clone()

        sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss2, eager_loss2)
        sharded_loss2.backward()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.UNSHARDED)
        self.assertTrue(flat_buffer.param_grads_alias_full_grad_buffer())
        self.assertGreater(
            (flat_buffer.full_grad_buffer - accumulated_after_first).abs().sum().item(),
            0.0,
        )
        sharded_optim.step()
        self.assertFalse(unit.state_dict()["defer_backward_reduce"])
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        sharded_optim.zero_grad()
        unit.unshard()

        event_names = [event.name for event in unit.runtime_events]
        self.assertIn("reshard_after_no_sync_backward", event_names)
        self.assertIn("reuse_full_grad_buffer_for_accumulation", event_names)
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_no_sync_supports_bucket_reduce_scatter_accumulation(self):
        torch.manual_seed(0)
        base_model = nn.Linear(4, 2)
        eager_model = copy.deepcopy(base_model)
        sharded_model = matrix_fully_shard(
            base_model,
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=True,
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x1 = torch.randn(3, 4)
        y1 = torch.randn(3, 2)
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 2)

        eager_loss1 = (eager_model(x1) - y1).pow(2).mean()
        eager_loss1.backward()
        eager_loss2 = (eager_model(x2) - y2).pow(2).mean()
        eager_loss2.backward()
        eager_optim.step()

        with sharded_model.no_sync():
            sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
            torch.testing.assert_close(sharded_loss1, eager_loss1)
            sharded_loss1.backward()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertEqual(unit.runtime_state, FSDPRuntimeState.BACKWARD_DEFERRED)
        self.assertTrue(unit.state_dict()["defer_backward_reduce"])
        self.assertEqual(unit.state_dict()["runtime_state"], FSDPRuntimeState.BACKWARD_DEFERRED)
        self._assert_full_buffer_released_or_shrunk(flat_buffer.full_buffer)
        self.assertIsNotNone(flat_buffer.grad_bucket_input)
        accumulated_after_first = flat_buffer.grad_bucket_input.clone()

        sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss2, eager_loss2)
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)
        self.assertIsNotNone(flat_buffer.grad_bucket_input)
        sharded_loss2.backward()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.UNSHARDED)
        self.assertIsNotNone(flat_buffer.grad_bucket_input)
        self.assertGreater(
            (flat_buffer.grad_bucket_input - accumulated_after_first).abs().sum().item(),
            0.0,
        )
        sharded_optim.step()
        self.assertFalse(unit.state_dict()["defer_backward_reduce"])
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        sharded_optim.zero_grad()
        unit.unshard()

        event_names = [event.name for event in unit.runtime_events]
        self.assertIn("reshard_after_no_sync_backward", event_names)
        self.assertIn("reuse_grad_bucket_for_accumulation", event_names)
        self.assertIn("collect_grad_bucket", event_names)
        self.assertIn("wait_reduce_grad_bucket", event_names)
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_checkpoint_no_sync_bucket_reduce_scatter_accumulates_to_eager(self):
        class CheckpointBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = nn.Linear(4, 8)
                self.act = nn.GELU()
                self.lin2 = nn.Linear(8, 2)

            def forward(self, x):
                return self.lin2(self.act(self.lin1(x)))

        class CheckpointModel(nn.Module):
            def __init__(self, *, use_reentrant: bool) -> None:
                super().__init__()
                self.use_reentrant = use_reentrant
                self.block = CheckpointBlock()

            def forward(self, x):
                return checkpoint(self.block, x, use_reentrant=self.use_reentrant)

        for use_reentrant in (False, True):
            for use_zero_copy_grad_bucket in (True, False):
                with self.subTest(
                    use_reentrant=use_reentrant,
                    use_zero_copy_grad_bucket=use_zero_copy_grad_bucket,
                ):
                    torch.manual_seed(23)
                    base_model = CheckpointModel(use_reentrant=use_reentrant)
                    eager_model = copy.deepcopy(base_model)
                    sharded_model = matrix_fully_shard(
                        copy.deepcopy(base_model),
                        reshard_after_forward=True,
                        finalize_after_backward=True,
                        backward_reduce_strategy="bucket_reduce_scatter",
                        use_zero_copy_grad_bucket=use_zero_copy_grad_bucket,
                    )
                    unit = sharded_model._matrix_fsdp_param_group

                    eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.05)
                    sharded_optim = MatrixFSDPOptimizer(
                        torch.optim.SGD(sharded_model.parameters(), lr=0.05),
                        sharded_model,
                    )

                    torch.manual_seed(29)
                    x1 = torch.randn(3, 4)
                    y1 = torch.randn(3, 2)
                    x2 = torch.randn(3, 4)
                    y2 = torch.randn(3, 2)
                    eager_x1 = x1.detach().clone().requires_grad_()
                    eager_x2 = x2.detach().clone().requires_grad_()
                    sharded_x1 = x1.detach().clone().requires_grad_()
                    sharded_x2 = x2.detach().clone().requires_grad_()

                    eager_loss1 = (eager_model(eager_x1) - y1).pow(2).mean()
                    eager_loss1.backward()
                    eager_loss2 = (eager_model(eager_x2) - y2).pow(2).mean()
                    eager_loss2.backward()
                    eager_optim.step()

                    with sharded_model.no_sync():
                        sharded_loss1 = (sharded_model(sharded_x1) - y1).pow(2).mean()
                        torch.testing.assert_close(sharded_loss1, eager_loss1)
                        sharded_loss1.backward()

                    self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
                    self.assertIsNotNone(unit.flat_buffer.grad_bucket_input)
                    accumulated_after_first = unit.flat_buffer.grad_bucket_input.clone()

                    sharded_loss2 = (sharded_model(sharded_x2) - y2).pow(2).mean()
                    torch.testing.assert_close(sharded_loss2, eager_loss2)
                    sharded_loss2.backward()

                    self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
                    self.assertIsNotNone(unit.flat_buffer.local_grad_shard)
                    self.assertIsNone(unit.flat_buffer.grad_bucket_input)
                    sharded_optim.step()
                    unit.unshard()

                    event_names = [event.name for event in unit.runtime_events]
                    self.assertIn("reuse_grad_bucket_for_accumulation", event_names)
                    self.assertIn("prepare_grad_bucket", event_names)
                    self.assertIn("wait_reduce_grad_bucket", event_names)
                    if use_zero_copy_grad_bucket:
                        self.assertIn("prepare_grad_bucket_zero_copy", event_names)
                    else:
                        self.assertIn("prepare_grad_bucket_copy_in", event_names)
                        self.assertIn("copy_in_grad_bucket_for_accumulation", event_names)
                    self.assertTrue(accumulated_after_first.abs().sum().item() > 0.0)
                    torch.testing.assert_close(sharded_x1.grad, eager_x1.grad)
                    torch.testing.assert_close(sharded_x2.grad, eager_x2.grad)
                    for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
                        torch.testing.assert_close(eager_param, sharded_param)

    def test_multi_unit_no_sync_bucket_copy_in_accumulation_matches_eager(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        base_model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(base_model)
        sharded_model = matrix_fully_shard(
            copy.deepcopy(base_model),
            wrap_policy=module_type_policy(nn.Linear),
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
        )
        units = collect_param_groups(sharded_model)
        self.assertEqual(len(units), 2)

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x1 = torch.randn(3, 4)
        y1 = torch.randn(3, 2)
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 2)

        eager_loss1 = (eager_model(x1) - y1).pow(2).mean()
        eager_loss1.backward()
        eager_loss2 = (eager_model(x2) - y2).pow(2).mean()
        eager_loss2.backward()
        eager_optim.step()

        with sharded_model.no_sync():
            sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
            torch.testing.assert_close(sharded_loss1, eager_loss1)
            sharded_loss1.backward()

        for unit in units:
            self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
            self.assertTrue(unit.state_dict()["defer_backward_reduce"])
            self.assertIsNotNone(unit.flat_buffer.grad_bucket_input)

        sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss2, eager_loss2)
        sharded_loss2.backward()
        sharded_optim.step()
        sharded_optim.zero_grad(set_to_none=True)
        for unit in units:
            unit.unshard()
            event_names = [event.name for event in unit.runtime_events]
            self.assertIn("prepare_grad_bucket_copy_in", event_names)
            self.assertIn("copy_in_grad_bucket_for_accumulation", event_names)
            self.assertIn("reuse_grad_bucket_for_accumulation", event_names)
            self.assertIn("wait_reduce_grad_bucket", event_names)
            self.assertIsNone(unit.flat_buffer.grad_bucket_input)
            self.assertIsNone(unit.flat_buffer.local_grad_shard)
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_bucket_copy_in_no_sync_multiple_accumulation_cycles_and_zero_grad_modes(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        base_model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(base_model)
        sharded_model = matrix_fully_shard(
            copy.deepcopy(base_model),
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
        )
        unit = sharded_model._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.05)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.05), sharded_model)

        for step_idx, set_to_none in enumerate((True, False)):
            torch.manual_seed(100 + step_idx)
            x1 = torch.randn(3, 4)
            y1 = torch.randn(3, 2)
            x2 = torch.randn(3, 4)
            y2 = torch.randn(3, 2)

            eager_loss1 = (eager_model(x1) - y1).pow(2).mean()
            eager_loss1.backward()
            eager_loss2 = (eager_model(x2) - y2).pow(2).mean()
            eager_loss2.backward()
            eager_optim.step()
            eager_optim.zero_grad(set_to_none=set_to_none)

            with sharded_model.no_sync():
                sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
                torch.testing.assert_close(sharded_loss1, eager_loss1)
                sharded_loss1.backward()

            self.assertIsNotNone(unit.flat_buffer.grad_bucket_input)
            first_bucket = unit.flat_buffer.grad_bucket_input.clone()
            sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
            torch.testing.assert_close(sharded_loss2, eager_loss2)
            sharded_loss2.backward()
            torch.testing.assert_close(unit.flat_buffer.grad_bucket_input, first_bucket)
            sharded_optim.step()
            sharded_optim.zero_grad(set_to_none=set_to_none)
            self.assertFalse(unit.state_dict()["defer_backward_reduce"])
            self.assertIsNone(unit.flat_buffer.grad_bucket_input)
            self.assertIsNone(unit.flat_buffer.local_grad_shard)

        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)
        event_names = [event.name for event in unit.runtime_events]
        self.assertGreaterEqual(event_names.count("copy_in_grad_bucket_for_accumulation"), 2)
        self.assertGreaterEqual(event_names.count("reuse_grad_bucket_for_accumulation"), 2)

    def test_unused_unit_no_sync_bucket_copy_in_is_skipped(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")

        class BranchModel(nn.Module):
            def __init__(self, *, shard: bool) -> None:
                super().__init__()
                used = nn.Linear(4, 2)
                unused = nn.Linear(4, 2)
                if shard:
                    used = matrix_fully_shard(
                        used,
                        reshard_after_forward=True,
                        finalize_after_backward=False,
                        backward_reduce_strategy="bucket_reduce_scatter",
                        use_zero_copy_grad_bucket=False,
                    )
                    unused = matrix_fully_shard(
                        unused,
                        reshard_after_forward=True,
                        finalize_after_backward=False,
                        backward_reduce_strategy="bucket_reduce_scatter",
                        use_zero_copy_grad_bucket=False,
                    )
                self.used = used
                self.unused = unused

            def forward(self, x):
                return self.used(x)

        torch.manual_seed(0)
        eager_model = BranchModel(shard=False)
        sharded_model = BranchModel(shard=True)
        sharded_model.load_state_dict(eager_model.state_dict())
        used_unit = sharded_model.used._matrix_fsdp_param_group
        unused_unit = sharded_model.unused._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x1 = torch.randn(3, 4)
        y1 = torch.randn(3, 2)
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 2)

        eager_loss1 = (eager_model(x1) - y1).pow(2).mean()
        eager_loss1.backward()
        eager_loss2 = (eager_model(x2) - y2).pow(2).mean()
        eager_loss2.backward()
        eager_optim.step()

        with sharded_optim.no_sync():
            sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
            torch.testing.assert_close(sharded_loss1, eager_loss1)
            sharded_loss1.backward()

        self.assertTrue(used_unit.state_dict()["defer_backward_reduce"])
        self.assertFalse(unused_unit.state_dict()["defer_backward_reduce"])
        self.assertIsNotNone(used_unit.flat_buffer.grad_bucket_input)
        self.assertIsNone(unused_unit.flat_buffer.grad_bucket_input)

        sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss2, eager_loss2)
        sharded_loss2.backward()
        sharded_optim.step()
        used_unit.unshard()
        unused_unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

        used_events = [event.name for event in used_unit.runtime_events]
        unused_events = [event.name for event in unused_unit.runtime_events]
        self.assertIn("copy_in_grad_bucket_for_accumulation", used_events)
        self.assertNotIn("pre_forward", unused_events)
        self.assertNotIn("copy_in_grad_bucket_for_accumulation", unused_events)

    def test_optimizer_scheduler_shares_full_param_buffer_pool_across_units(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 4), reshard_after_forward=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(4, 4), reshard_after_forward=True),
        )
        units = [model[0]._matrix_fsdp_param_group, model[2]._matrix_fsdp_param_group]
        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_cached_full_param_buffers_per_key=1,
        )

        self.assertIs(units[0].flat_buffer.full_param_buffer_pool, units[1].flat_buffer.full_param_buffer_pool)

        units[0].unshard()
        first_ptr = units[0].flat_buffer.full_buffer.data_ptr()
        units[0].reshard_after_forward(needs_pre_backward_unshard=False)
        self.assertIsNone(units[0].flat_buffer.full_buffer)

        units[1].unshard()
        second_ptr = units[1].flat_buffer.full_buffer.data_ptr()

        self.assertEqual(second_ptr, first_ptr)
        self.assertGreaterEqual(optim.scheduler.full_param_buffer_pool.stats()["reuses"], 1)

    def test_optimizer_scheduler_defaults_to_fsdp2_like_full_param_resize(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 4), reshard_after_forward=True)
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        unit.unshard()
        full_buffer = unit.flat_buffer.full_buffer
        self.assertIsNotNone(full_buffer)
        unit.reshard_after_forward(needs_pre_backward_unshard=False)

        stats = optim.scheduler.full_param_buffer_pool.stats()
        self.assertEqual(stats["max_cached_per_key"], 0)
        self.assertEqual(stats["cached_buffers"], 0)
        self._assert_full_buffer_released_or_shrunk(unit.flat_buffer.full_buffer)

    def test_scheduler_counts_pending_full_param_release_in_active_budget(self):
        class FakeEvent:
            def __init__(self) -> None:
                self.complete = False

            def query(self) -> bool:
                return self.complete

        model = matrix_fully_shard(nn.Linear(4, 4), reshard_after_forward=True)
        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_cached_full_param_buffers_per_key=1,
        )
        buffer = torch.empty(16)
        event = FakeEvent()

        optim.scheduler.full_param_buffer_pool.release(buffer, cuda_event=event)
        optim.scheduler.record_full_param_buffer_snapshot("pending_release")

        self.assertEqual(optim.scheduler.full_param_buffer_snapshots[-1]["active_count"], 1)
        self.assertEqual(optim.scheduler.full_param_buffer_snapshots[-1]["active_numel"], 16)

        event.complete = True
        optim.scheduler.record_full_param_buffer_snapshot("pending_release_drained")

        self.assertEqual(optim.scheduler.full_param_buffer_snapshots[-1]["active_count"], 0)

    def test_optimizer_step_records_post_optimizer_event(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)
        recorder = mock.Mock(return_value=False)
        optim.scheduler.record_post_optimizer_event = recorder

        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        optim.step()

        recorder.assert_called_once_with()

    def test_start_unshard_waits_for_post_optimizer_event_before_all_gather(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)
        unit = model._matrix_fsdp_param_group
        waiter = mock.Mock(return_value=True)
        optim.scheduler.wait_for_post_optimizer_event_before_all_gather = waiter

        unit.start_unshard("explicit")

        waiter.assert_called_once_with()
        self.assertIn(
            "wait_post_optimizer_event_before_all_gather",
            [event.name for event in unit.runtime_events],
        )

    def test_lifecycle_transition_events_cover_forward_backward_step(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2), reshard_after_forward=True)
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)
        unit = model._matrix_fsdp_param_group

        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        optim.step()

        event_names = [event.name for event in unit.runtime_events]
        self.assertIn(
            "lifecycle_transition:sharded->unsharded:start_unshard:pre_forward",
            event_names,
        )
        self.assertIn(
            "lifecycle_transition:unsharded->forward_resharded:reshard_after_forward",
            event_names,
        )
        self.assertIn(
            "lifecycle_transition:forward_resharded->unsharded:start_unshard:pre_backward",
            event_names,
        )
        self.assertTrue(
            "lifecycle_transition:unsharded->sharded:reshard_before_reduce_grad" in event_names
            or "lifecycle_transition:unsharded->sharded:finish_backward" in event_names
        )

    def test_lifecycle_transition_rejects_invalid_state_change(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group

        with self.assertRaisesRegex(RuntimeError, "Invalid MatrixFSDP lifecycle transition"):
            unit._transition_lifecycle_state(
                FSDPLifecycleState.FORWARD_RESHARDED,
                reason="test_invalid_transition",
                allowed_from=(FSDPLifecycleState.UNSHARDED,),
            )

    def test_lifecycle_invariants_reject_inflight_unshard_without_handle(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        unit._unshard_inflight = True

        with self.assertRaisesRegex(RuntimeError, "in-flight unshard"):
            unit._validate_lifecycle_invariants("test_invalid_inflight")

    def test_optimizer_ready_rejects_unsharded_params(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group

        unit.unshard()

        with self.assertRaisesRegex(RuntimeError, "expects MatrixFSDP parameters to be sharded"):
            unit.assert_optimizer_ready("test_optimizer_ready")

    def test_optimizer_step_checks_param_groups_are_optimizer_ready(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)
        unit = model._matrix_fsdp_param_group

        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        with mock.patch.object(unit, "assert_optimizer_ready", wraps=unit.assert_optimizer_ready) as checker:
            optim.step()

        checker.assert_called_once_with("optimizer.step")

    def test_finalize_backward_requires_unsharded_state(self):
        model = matrix_fully_shard(nn.Linear(4, 2))

        with self.assertRaisesRegex(RuntimeError, "expects parameters to be unsharded"):
            model._matrix_fsdp_param_group.finalize_backward()

    def test_reshard_after_forward_no_grad_returns_to_sharded_state(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2), reshard_after_forward=True)
        unit = model._matrix_fsdp_param_group

        with torch.no_grad():
            out = model(torch.randn(3, 4))

        self.assertEqual(out.shape, (3, 2))
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertFalse(unit._is_unsharded)
        self._assert_full_buffer_released_or_shrunk(unit.flat_buffer.full_buffer)
        self.assertTrue(unit.state_dict()["reshard_after_forward"])

    def test_reshard_after_forward_rejects_step_before_backward_unshard(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2), reshard_after_forward=True)
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        model(torch.randn(3, 4)).sum()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)
        self._assert_full_buffer_released_or_shrunk(unit.flat_buffer.full_buffer)

        with self.assertRaisesRegex(RuntimeError, "forward-resharded"):
            optim.step()

    def test_reshard_after_forward_training_auto_unshards_before_backward(self):
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(
            model,
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        unit = sharded_model._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        eager_loss = eager_model(x).sum()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = sharded_model(x).sum()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)
        self.assertIsNone(unit.flat_buffer.full_grad_buffer)
        sharded_loss.backward()
        self.assertIn("prepare_full_grad_buffer", [event.name for event in unit.runtime_events])
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.UNSHARDED)
        self.assertTrue(unit.flat_buffer.param_grads_alias_full_grad_buffer())
        sharded_optim.step()
        unit.unshard()

        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_finalize_after_backward_auto_finalizes_before_optimizer_step(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(
            model,
            reshard_after_forward=True,
            finalize_after_backward=True,
        )
        unit = sharded_model._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)
        sharded_loss.backward()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertTrue(unit.finalized_after_backward)
        self._assert_full_buffer_released_or_shrunk(unit.flat_buffer.full_buffer)
        self.assertTrue(unit.state_dict()["finalize_after_backward"])
        self._assert_finalize_after_backward_events(unit)

        sharded_optim.step()
        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_plain_torch_optimizer_single_step_matches_eager(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")

        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(
            model,
            reshard_after_forward=True,
            finalize_after_backward=True,
        )
        unit = sharded_model._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = torch.optim.SGD(sharded_model.parameters(), lr=0.1)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()
        self.assertTrue(unit.finalized_after_backward)
        self.assertIsNotNone(unit.flat_buffer.local_grad_shard)
        sharded_optim.step()

        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_torch_optimizer_auto_prepares_matrix_hooks(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")

        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2), reshard_after_forward=True, finalize_after_backward=True)
        unit = model._matrix_fsdp_param_group
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        prepared = optimizer.matrix_fsdp

        self.assertIs(optimizer.matrix_fsdp, prepared)
        self.assertIs(optimizer.state_manager, prepared.state_manager)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()
        self.assertTrue(prepared.state_objects)
        self.assertTrue(optimizer.matrix_fsdp.state_objects)
        prepared.validate_local_state_shapes()
        optimizer.zero_grad()
        self.assertIsNone(unit.flat_buffer.local_grad_shard)
        prepared.remove()
        self.assertFalse(hasattr(optimizer, "matrix_fsdp"))

    def test_auto_prepared_torch_optimizer_multistep_matches_eager(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")

        for set_to_none in (True, False):
            with self.subTest(set_to_none=set_to_none):
                torch.manual_seed(0)
                model = nn.Linear(4, 2)
                eager_model = copy.deepcopy(model)
                sharded_model = matrix_fully_shard(
                    model,
                    reshard_after_forward=True,
                    finalize_after_backward=True,
                )
                unit = sharded_model._matrix_fsdp_param_group

                eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
                sharded_optim = torch.optim.SGD(sharded_model.parameters(), lr=0.1)
                self.assertIsInstance(sharded_optim.matrix_fsdp, PreparedMatrixOptimizer)

                for _ in range(2):
                    x = torch.randn(3, 4)
                    y = torch.randn(3, 2)

                    eager_loss = (eager_model(x) - y).pow(2).mean()
                    eager_loss.backward()
                    eager_optim.step()
                    eager_optim.zero_grad(set_to_none=set_to_none)

                    sharded_loss = (sharded_model(x) - y).pow(2).mean()
                    sharded_loss.backward()
                    sharded_optim.step()
                    sharded_optim.zero_grad(set_to_none=set_to_none)
                    self.assertIsNone(unit.flat_buffer.local_grad_shard)

                unit.unshard()
                for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
                    torch.testing.assert_close(eager_param, sharded_param)

    def test_auto_prepared_optimizer_step_and_zero_grad_reject_active_no_sync(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with optimizer.no_sync():
            with self.assertRaisesRegex(RuntimeError, "cannot run inside no_sync"):
                optimizer.step()
            with self.assertRaisesRegex(RuntimeError, "cannot run inside no_sync"):
                optimizer.zero_grad()

        self.assertFalse(unit.is_no_sync_active)

    def test_configure_optimizer_adamw_auto_manages_matrix_lifecycle(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(
            model,
            reshard_after_forward=True,
            finalize_after_backward=True,
        )
        unit = sharded_model._matrix_fsdp_param_group

        eager_optim = torch.optim.AdamW(eager_model.parameters(), lr=0.01)
        sharded_optim = configure_optimizer(sharded_model, "adamw", lr=0.01)

        self.assertIsInstance(sharded_optim, torch.optim.AdamW)
        self.assertIsInstance(sharded_optim.matrix_fsdp, PreparedMatrixOptimizer)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()
        sharded_optim.step()
        sharded_optim.zero_grad()
        self.assertIsNone(unit.flat_buffer.local_grad_shard)

        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_mixed_muon_adamw_optimizer_auto_manages_matrix_lifecycle(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")

        for set_to_none in (True, False):
            with self.subTest(set_to_none=set_to_none):
                torch.manual_seed(0)
                model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
                eager_model = copy.deepcopy(model)
                sharded_model = matrix_fully_shard(
                    model,
                    reshard_after_forward=True,
                    finalize_after_backward=True,
                )
                unit = sharded_model._matrix_fsdp_param_group

                eager_optim = torch.optim.AdamW(eager_model.parameters(), lr=0.01)
                sharded_optim = make_mixed_muon_adamw_optimizer(
                    sharded_model,
                    default_matrix_optimizer="adamw",
                    default_other_optimizer="adamw",
                    adamw_lr=0.01,
                )
                self.assertIs(sharded_optim.matrix_fsdp, sharded_optim)
                self.assertIsNotNone(sharded_optim.state_manager)

                for _ in range(2):
                    x = torch.randn(3, 4)
                    y = torch.randn(3, 2)

                    eager_loss = (eager_model(x) - y).pow(2).mean()
                    eager_loss.backward()
                    eager_optim.step()
                    eager_optim.zero_grad(set_to_none=set_to_none)

                    sharded_loss = (sharded_model(x) - y).pow(2).mean()
                    sharded_loss.backward()
                    sharded_optim.step()
                    sharded_optim.zero_grad(set_to_none=set_to_none)
                    self.assertIsNone(unit.flat_buffer.local_grad_shard)

                unit.unshard()
                for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
                    torch.testing.assert_close(eager_param, sharded_param)

    def test_configure_optimizer_mixed_route_auto_manages_matrix_lifecycle(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")

        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)),
            reshard_after_forward=True,
            finalize_after_backward=True,
        )
        unit = model._matrix_fsdp_param_group
        optimizer = configure_optimizer(
            model,
            MatrixOptimizerConfig(
                optimizer="mixed_muon_adamw",
                default_matrix_optimizer="adamw",
                default_other_optimizer="adamw",
                kwargs={"adamw_lr": 0.01},
            ),
        )

        self.assertIsInstance(optimizer, MixedMuonAdamWOptimizer)
        self.assertIs(optimizer.matrix_fsdp, optimizer)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        self.assertIsNone(unit.flat_buffer.local_grad_shard)

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_torch_muon_auto_prepares_matrix_hooks(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Linear(4, 2, bias=False),
            reshard_after_forward=True,
            finalize_after_backward=True,
        )
        unit = model._matrix_fsdp_param_group
        optimizer = torch.optim.Muon(model.parameters(), lr=0.01)

        self.assertIsInstance(optimizer.matrix_fsdp, PreparedMatrixOptimizer)
        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        self.assertIsNone(unit.flat_buffer.local_grad_shard)

    def test_per_param_backward_reduce_auto_finalizes_without_full_grad_buffer(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        for strategy in ("per_param", "per_param_allreduce"):
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
            eager_model = copy.deepcopy(model)
            sharded_model = matrix_fully_shard(
                model,
                reshard_after_forward=True,
                finalize_after_backward=True,
                backward_reduce_strategy=strategy,
            )
            unit = sharded_model._matrix_fsdp_param_group

            eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
            sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

            x = torch.randn(3, 4)
            y = torch.randn(3, 2)
            eager_loss = (eager_model(x) - y).pow(2).mean()
            eager_loss.backward()
            eager_optim.step()

            sharded_loss = (sharded_model(x) - y).pow(2).mean()
            sharded_loss.backward()

            self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
            self.assertTrue(unit.finalized_after_backward)
            self.assertIsNone(unit.flat_buffer.full_grad_buffer)
            self.assertIsNone(unit.flat_buffer.local_grad_accumulator)
            self.assertIsNotNone(unit.flat_buffer.local_grad_shard)
            self.assertIn("prepare_local_grad_accumulator", [event.name for event in unit.runtime_events])
            sharded_optim.step()
            unit.unshard()
            for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
                torch.testing.assert_close(eager_param, sharded_param)

    def test_bucket_reduce_scatter_auto_finalizes_without_full_grad_buffer(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(
            model,
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
        )
        unit = sharded_model._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()

        event_names = [event.name for event in unit.runtime_events]
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertTrue(unit.finalized_after_backward)
        self.assertIsNone(unit.flat_buffer.full_grad_buffer)
        self.assertIsNone(unit.flat_buffer.local_grad_accumulator)
        self.assertIsNotNone(unit.flat_buffer.local_grad_shard)
        self.assertIn("prepare_grad_bucket", event_names)
        self.assertIn("prepare_grad_bucket_copy_in", event_names)
        self.assertIn("collect_grad_bucket", event_names)
        self.assertIn("reshard_before_reduce_grad", event_names)
        self.assertTrue(any(name.startswith("full_param_buffer_clear:pre_reduce_grad:") for name in event_names))
        self.assertIn("before_backward_reduce_scheduler_runtime", event_names)
        self.assertIn("reduce_grad_bucket", event_names)
        self.assertIn("after_backward_reduce_started_scheduler_runtime", event_names)
        self.assertIn("post_backward_reshard_scheduler_runtime", event_names)
        self.assertIn("wait_reduce_grad_bucket_handle", event_names)
        self.assertIn("materialize_reduced_local_grad_shard", event_names)
        self.assertIn("wait_reduce_grad_bucket", event_names)
        self.assertFalse(unit.has_pending_backward_reduce)
        sharded_optim.step()
        self.assertFalse(unit.has_pending_backward_reduce)
        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_bucket_reduce_scatter_copy_in_matches_eager_model(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(
            model,
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
        )
        unit = sharded_model._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        sharded_loss.backward()

        event_names = [event.name for event in unit.runtime_events]
        self.assertIn("prepare_grad_bucket", event_names)
        self.assertIn("prepare_grad_bucket_copy_in", event_names)
        self.assertNotIn("prepare_grad_bucket_zero_copy", event_names)
        sharded_optim.step()
        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_bucket_copy_in_accumulates_local_grad_shards_across_backwards(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(
            model,
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
        )
        unit = sharded_model._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x1 = torch.randn(3, 4)
        y1 = torch.randn(3, 2)
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 2)

        eager_loss1 = (eager_model(x1) - y1).pow(2).mean()
        eager_loss1.backward()
        eager_loss2 = (eager_model(x2) - y2).pow(2).mean()
        eager_loss2.backward()
        eager_optim.step()

        sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
        torch.testing.assert_close(sharded_loss1, eager_loss1)
        sharded_loss1.backward()
        self.assertIsNotNone(unit.flat_buffer.local_grad_shard)
        accumulated_after_first = unit.flat_buffer.local_grad_shard.clone()

        sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss2, eager_loss2)
        sharded_loss2.backward()
        self.assertIsNotNone(unit.flat_buffer.local_grad_shard)
        self.assertGreater(
            (unit.flat_buffer.local_grad_shard - accumulated_after_first).abs().sum().item(),
            0.0,
        )

        sharded_optim.step()
        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)
        self.assertIn("accumulate_local_grad_shard", [event.name for event in unit.runtime_events])

    def test_no_sync_supports_bucket_copy_in_accumulation(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(
            model,
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
        )
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x1 = torch.randn(3, 4)
        y1 = torch.randn(3, 2)
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 2)

        eager_loss1 = (eager_model(x1) - y1).pow(2).mean()
        eager_loss1.backward()
        eager_loss2 = (eager_model(x2) - y2).pow(2).mean()
        eager_loss2.backward()
        eager_optim.step()

        with sharded_model.no_sync():
            sharded_loss1 = (sharded_model(x1) - y1).pow(2).mean()
            torch.testing.assert_close(sharded_loss1, eager_loss1)
            sharded_loss1.backward()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertIsNotNone(flat_buffer.grad_bucket_input)
        accumulated_after_first = flat_buffer.grad_bucket_input.clone()

        sharded_loss2 = (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss2, eager_loss2)
        sharded_loss2.backward()

        self.assertIsNotNone(flat_buffer.grad_bucket_input)
        torch.testing.assert_close(flat_buffer.grad_bucket_input, accumulated_after_first)
        sharded_optim.step()
        self.assertIsNone(flat_buffer.grad_bucket_input)
        sharded_optim.zero_grad()
        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

        event_names = [event.name for event in unit.runtime_events]
        self.assertIn("prepare_grad_bucket_copy_in", event_names)
        self.assertIn("copy_in_grad_bucket_for_accumulation", event_names)
        self.assertIn("reuse_grad_bucket_for_accumulation", event_names)
        self.assertIn("wait_reduce_grad_bucket", event_names)

    def test_scheduler_allows_configured_pending_backward_reduce_queue(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        base_model = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        eager_model = copy.deepcopy(base_model)
        sharded_model = nn.Sequential(
            matrix_fully_shard(
                base_model[0],
                reshard_after_forward=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
            ),
            base_model[1],
            matrix_fully_shard(
                base_model[2],
                reshard_after_forward=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
            ),
            base_model[3],
            matrix_fully_shard(
                base_model[4],
                reshard_after_forward=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
            ),
        )
        units = [
            sharded_model[0]._matrix_fsdp_param_group,
            sharded_model[2]._matrix_fsdp_param_group,
            sharded_model[4]._matrix_fsdp_param_group,
        ]

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(
            torch.optim.SGD(sharded_model.parameters(), lr=0.1),
            sharded_model,
            max_pending_backward_reduces=2,
        )

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        sharded_loss.backward()

        self.assertLess(
            self._event_sequence(units[2], "start_reduce_grad_bucket"),
            self._event_sequence(units[1], "start_reduce_grad_bucket"),
        )
        self.assertLess(
            self._event_sequence(units[1], "start_reduce_grad_bucket"),
            self._event_sequence(units[2], "wait_reduce_grad_bucket"),
        )
        self.assertTrue(all(unit.lifecycle_state == FSDPLifecycleState.SHARDED for unit in units))
        self.assertTrue(all(unit.finalized_after_backward for unit in units))
        self.assertFalse(any(unit.has_pending_backward_reduce for unit in units))
        self.assertEqual(sharded_optim.scheduler.backward_reduce_waits, 3)
        summary = summarize_runtime_events(sharded_model)
        self.assertEqual(summary["schedulers"][0]["max_pending_backward_reduces"], 2)
        self.assertEqual(summary["schedulers"][0]["pending_backward_reduce_count"], 0)

        sharded_optim.step()
        for unit in units:
            unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_no_saved_tensor_hooks_shrinks_full_buffer_storage_after_forward(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Linear(4, 2),
            reshard_after_forward=True,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_saved_tensor_hooks=False,
        )
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        loss = model(torch.randn(3, 4)).sum()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)
        self.assertIsNotNone(unit.flat_buffer.full_buffer)
        self.assertEqual(unit.flat_buffer.full_buffer.untyped_storage().nbytes(), 0)
        self.assertEqual(optim.scheduler.full_param_buffer_pool.stats()["cached_buffers"], 0)

        loss.backward()
        optim.step()
        unit.unshard()

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.UNSHARDED)
        self.assertGreater(unit.flat_buffer.full_buffer.untyped_storage().nbytes(), 0)

    def test_saved_full_param_views_do_not_keep_full_buffer_after_forward(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(
            model,
            reshard_after_forward=True,
            finalize_after_backward=True,
            use_saved_tensor_hooks=True,
        )
        unit = sharded_model._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        eager_x = x.detach().clone().requires_grad_()
        sharded_x = x.detach().clone().requires_grad_()
        y = torch.randn(3, 2)
        eager_loss = (eager_model(eager_x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(sharded_x) - y).pow(2).mean()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)
        self._assert_full_buffer_released_or_shrunk(unit.flat_buffer.full_buffer)
        self.assertGreater(unit.state_dict()["saved_full_param_views"], 0)
        sharded_loss.backward()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)

        sharded_optim.step()
        unit.unshard()
        torch.testing.assert_close(sharded_x.grad, eager_x.grad)
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_adamw_optimizer_state_summary_tracks_local_params(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)))
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

        x = torch.randn(3, 4)
        loss = model(x).sum()
        loss.backward()
        optim.step()

        optim.validate_local_state_shapes()
        summary = optim.local_state_summary()
        self.assertEqual(summary["param_numel"], unit.flat_buffer.local_numel)
        self.assertEqual(summary["tensor_state_numel_by_name"]["exp_avg"], unit.flat_buffer.local_numel)
        self.assertEqual(summary["tensor_state_numel_by_name"]["exp_avg_sq"], unit.flat_buffer.local_numel)
        self.assertEqual(summary["tensor_state_numel"], 2 * unit.flat_buffer.local_numel)

    def test_multi_unit_finalize_after_backward_auto_finalizes_each_unit(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")
        torch.manual_seed(0)
        base_model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(base_model)
        sharded_model = nn.Sequential(
            matrix_fully_shard(
                base_model[0],
                reshard_after_forward=True,
                finalize_after_backward=True,
            ),
            base_model[1],
            matrix_fully_shard(
                base_model[2],
                reshard_after_forward=True,
                finalize_after_backward=True,
            ),
        )
        units = [sharded_model[0]._matrix_fsdp_param_group, sharded_model[2]._matrix_fsdp_param_group]

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        self.assertTrue(all(unit.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED for unit in units))

        sharded_loss.backward()
        for unit in units:
            self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
            self.assertTrue(unit.finalized_after_backward)
            self._assert_full_buffer_released_or_shrunk(unit.flat_buffer.full_buffer)
            self._assert_finalize_after_backward_events(unit)
        self.assertLess(
            self._event_sequence(units[1], "pre_backward_unshard"),
            self._event_sequence(units[0], "pre_backward_unshard"),
        )
        self.assertLess(
            self._event_sequence(units[1], "finalize_backward"),
            self._event_sequence(units[0], "finalize_backward"),
        )

        sharded_optim.step()
        for unit in units:
            unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_unused_unit_is_skipped_by_optimizer_step(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")

        class BranchModel(nn.Module):
            def __init__(self, *, shard: bool) -> None:
                super().__init__()
                used = nn.Linear(4, 2)
                unused = nn.Linear(4, 2)
                if shard:
                    used = matrix_fully_shard(
                        used,
                        reshard_after_forward=True,
                        finalize_after_backward=True,
                        backward_reduce_strategy="bucket_reduce_scatter",
                    )
                    unused = matrix_fully_shard(
                        unused,
                        reshard_after_forward=True,
                        finalize_after_backward=True,
                        backward_reduce_strategy="bucket_reduce_scatter",
                    )
                self.used = used
                self.unused = unused

            def forward(self, x):
                return self.used(x)

        torch.manual_seed(0)
        eager_model = BranchModel(shard=False)
        sharded_model = BranchModel(shard=True)
        sharded_model.load_state_dict(eager_model.state_dict())
        used_unit = sharded_model.used._matrix_fsdp_param_group
        unused_unit = sharded_model.unused._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        sharded_loss.backward()
        sharded_optim.step()

        self.assertEqual(used_unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertEqual(unused_unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertNotIn("pre_forward", [event.name for event in unused_unit.runtime_events])
        used_unit.unshard()
        unused_unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_multiple_forward_single_backward_matches_eager_model(self):
        torch.manual_seed(0)
        base_model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(base_model)
        sharded_model = matrix_fully_shard(copy.deepcopy(base_model), reshard_after_forward=False)
        unit = sharded_model._matrix_fsdp_param_group

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x1 = torch.randn(3, 4)
        y1 = torch.randn(3, 2)
        x2 = torch.randn(3, 4)
        y2 = torch.randn(3, 2)

        eager_loss = (eager_model(x1) - y1).pow(2).mean() + 0.25 * (eager_model(x2) - y2).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x1) - y1).pow(2).mean() + 0.25 * (sharded_model(x2) - y2).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        sharded_loss.backward()
        sharded_optim.step()
        unit.unshard()

        event_names = [event.name for event in unit.runtime_events]
        self.assertGreaterEqual(event_names.count("pre_forward"), 2)
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_parallel_branch_multi_unit_backward_matches_eager_model(self):
        if not hasattr(torch.nn.Parameter(torch.empty(1)), "register_post_accumulate_grad_hook"):
            self.skipTest("requires Tensor.register_post_accumulate_grad_hook")

        class ParallelModel(nn.Module):
            def __init__(self, *, shard: bool) -> None:
                super().__init__()
                left = nn.Linear(4, 4)
                right = nn.Linear(4, 4)
                out = nn.Linear(4, 2)
                if shard:
                    left = matrix_fully_shard(
                        left,
                        reshard_after_forward=True,
                        finalize_after_backward=True,
                        backward_reduce_strategy="bucket_reduce_scatter",
                    )
                    right = matrix_fully_shard(
                        right,
                        reshard_after_forward=True,
                        finalize_after_backward=True,
                        backward_reduce_strategy="bucket_reduce_scatter",
                    )
                    out = matrix_fully_shard(
                        out,
                        reshard_after_forward=True,
                        finalize_after_backward=True,
                        backward_reduce_strategy="bucket_reduce_scatter",
                    )
                self.left = left
                self.right = right
                self.out = out

            def forward(self, x):
                return self.out(torch.tanh(self.left(x)) + torch.sigmoid(self.right(x)))

        torch.manual_seed(0)
        eager_model = ParallelModel(shard=False)
        sharded_model = ParallelModel(shard=True)
        sharded_model.load_state_dict(eager_model.state_dict())
        units = [
            sharded_model.left._matrix_fsdp_param_group,
            sharded_model.right._matrix_fsdp_param_group,
            sharded_model.out._matrix_fsdp_param_group,
        ]

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        sharded_loss.backward()

        for unit in units:
            self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
            self.assertIn("prepare_grad_bucket", [event.name for event in unit.runtime_events])

        sharded_optim.step()
        for unit in units:
            unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_reshard_after_forward_rejects_multi_forward_before_backward(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Linear(4, 2),
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        unit = model._matrix_fsdp_param_group

        model(torch.randn(3, 4)).sum()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)

        with self.assertRaisesRegex(RuntimeError, "another forward before"):
            model(torch.randn(3, 4)).sum()

    def test_reshard_after_forward_rejects_forward_before_finalize_backward(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Linear(4, 2),
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        unit = model._matrix_fsdp_param_group

        first_loss = model(torch.randn(3, 4)).sum()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)
        first_loss.backward()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.UNSHARDED)

        with self.assertRaisesRegex(RuntimeError, "finalized"):
            model(torch.randn(3, 4)).sum()

    def test_reshard_after_forward_allows_forward_after_optimizer_step(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Linear(4, 2),
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        first_loss = model(torch.randn(3, 4)).sum()
        first_loss.backward()
        optim.step()
        optim.zero_grad()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)

        second_loss = model(torch.randn(3, 4)).sum()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)
        second_loss.backward()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.UNSHARDED)

    def test_reshard_after_forward_registers_hooks_on_nested_outputs(self):
        class NestedOutputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 2)

            def forward(self, x):
                y = self.proj(x)
                return {"main": y, "aux": (y.square().mean(),)}

        torch.manual_seed(0)
        model = matrix_fully_shard(
            NestedOutputModule(),
            reshard_after_forward=True,
            finalize_after_backward=False,
            backward_reduce_strategy="flat",
        )
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        out = model(torch.randn(3, 4))
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.FORWARD_RESHARDED)
        loss = out["main"].sum() + out["aux"][0]
        loss.backward()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.UNSHARDED)
        optim.step()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)

    def test_optimizer_collects_multiple_units_from_module_tree(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8), reshard_after_forward=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2), reshard_after_forward=True),
        )

        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        self.assertEqual(len(optim.runtime_param_groups), 2)
        self.assertIs(optim.runtime_param_groups[0], model[0]._matrix_fsdp_param_group)
        self.assertIs(optim.runtime_param_groups[1], model[2]._matrix_fsdp_param_group)
        self.assertIs(optim.runtime_param_groups, optim.fsdp_param_groups)

    def test_multi_unit_metadata_ids_are_unique_and_unit_local(self):
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)),
            wrap_policy=lambda module: isinstance(module, nn.Linear),
        )
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)
        states = [param_group.state_dict() for param_group in optim.runtime_param_groups]

        runtime_param_group_ids = [state["runtime_param_group_id"] for state in states]
        runtime_unit_ids = [state["runtime_unit_id"] for state in states]
        self.assertEqual(runtime_param_group_ids, runtime_unit_ids)
        self.assertEqual(len(runtime_param_group_ids), len(set(runtime_param_group_ids)))
        self.assertEqual(len(runtime_unit_ids), len(set(runtime_unit_ids)))
        for state in states:
            self.assertEqual(state["runtime_unit_id"], state["planner_group_id"])
            self.assertEqual(state["runtime_unit_id"], state["comm_buffer_id"])

    def test_scheduler_tracks_unit_order(self):
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8), reshard_after_forward=True, forward_prefetch=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2), reshard_after_forward=True, forward_prefetch=True),
        )
        units = [model[0]._matrix_fsdp_param_group, model[2]._matrix_fsdp_param_group]

        scheduler = MatrixFSDPScheduler(units)

        self.assertEqual(scheduler.units, tuple(units))
        self.assertIs(scheduler.next_forward_unit(units[0]), units[1])
        self.assertIsNone(scheduler.next_forward_unit(units[1]))
        self.assertIsNone(scheduler.previous_backward_prefetch_unit(units[0]))
        self.assertIs(scheduler.previous_backward_prefetch_unit(units[1]), units[0])
        self.assertIs(units[0]._scheduler, scheduler)
        self.assertIs(units[1]._scheduler, scheduler)

    def test_owner_prefetch_queue_enforces_forward_param_group_order(self):
        class FakeOwnerFlatBuffer:
            local_shard = None
            matrix_collective_backend = "custom"

            def __init__(self, *, independent_comm_lanes: bool = False):
                self.independent_comm_lanes = independent_comm_lanes

            def owner_segment_prefetch_order_gate_required(self):
                return True

            def owner_segment_collective_has_independent_comm_lanes(self):
                return self.independent_comm_lanes

        class FakeRuntimeMetadata:
            def __init__(self, index):
                self.runtime_param_group_id = f"param_group_{index}"

        class FakeUnit:
            def __init__(self, index):
                self.forward_prefetch_enabled = True
                self.backward_prefetch_enabled = True
                self.flat_buffer = FakeOwnerFlatBuffer()
                self.runtime_metadata = FakeRuntimeMetadata(index)
                self.world_size = 8
                self._forward_prefetched = False
                self._backward_prefetched = False
                self.events = []
                self.forward_prefetch_kwargs = []
                self.backward_prefetch_kwargs = []
                self.has_pending_backward_reduce = False

            def set_scheduler(self, scheduler):
                self.scheduler = scheduler

            def set_comm_context(self, comm_context):
                self.comm_context = comm_context

            def set_full_param_buffer_pool(self, pool):
                self.pool = pool

            def prefetch_forward(self, **kwargs):
                self.forward_prefetch_kwargs.append(kwargs)
                self._forward_prefetched = True
                self.events.append("forward_prefetch")
                return True

            def prefetch_backward(self, **kwargs):
                self.backward_prefetch_kwargs.append(kwargs)
                self._backward_prefetched = True
                self.events.append("backward_prefetch")
                return True

            def wait_post_backward_reduce(self):
                self.has_pending_backward_reduce = False
                self.events.append("wait_post_backward_reduce")

            def wait_unshard(self, reason):
                self._unshard_inflight = False
                self.events.append(f"wait_unshard:{reason}")

            def _record_event(self, name):
                self.events.append(name)

        units = [FakeUnit(index) for index in range(3)]
        scheduler = MatrixFSDPScheduler(units, max_unsharded_prefetch_units=1)

        scheduler.on_pre_forward(units[1])
        self.assertNotIn("forward_prefetch", units[2].events)
        self.assertIn("forward_prefetch_skipped:owner_ordered_queue", " ".join(units[2].events))

        scheduler.on_pre_forward(units[0])
        scheduler.on_pre_forward(units[1])

        self.assertEqual(units[1].events.count("forward_prefetch"), 1)
        self.assertEqual(units[2].events.count("forward_prefetch"), 1)
        self.assertEqual(scheduler.forward_prefetch_issued, 2)
        for unit in units[1:]:
            self.assertEqual(
                unit.forward_prefetch_kwargs,
                [{"validate_owner_collective_signature": True, "ordered_owner_collective": True}],
            )

    def test_owner_prefetch_queue_enforces_backward_post_forward_order(self):
        class FakeOwnerFlatBuffer:
            local_shard = None
            matrix_collective_backend = "custom"

            def __init__(self, *, independent_comm_lanes: bool = False):
                self.independent_comm_lanes = independent_comm_lanes

            def owner_segment_prefetch_order_gate_required(self):
                return True

            def owner_segment_collective_has_independent_comm_lanes(self):
                return self.independent_comm_lanes

        class FakeRuntimeMetadata:
            def __init__(self, index):
                self.runtime_param_group_id = f"param_group_{index}"

        class FakeUnit:
            def __init__(self, index):
                self.forward_prefetch_enabled = True
                self.backward_prefetch_enabled = True
                self.flat_buffer = FakeOwnerFlatBuffer()
                self.runtime_metadata = FakeRuntimeMetadata(index)
                self.world_size = 8
                self._forward_prefetched = False
                self._backward_prefetched = False
                self.events = []
                self.forward_prefetch_kwargs = []
                self.backward_prefetch_kwargs = []
                self.has_pending_backward_reduce = False

            def set_scheduler(self, scheduler):
                self.scheduler = scheduler

            def set_comm_context(self, comm_context):
                self.comm_context = comm_context

            def set_full_param_buffer_pool(self, pool):
                self.pool = pool

            def prefetch_forward(self, **kwargs):
                self.forward_prefetch_kwargs.append(kwargs)
                self._forward_prefetched = True
                self.events.append("forward_prefetch")
                return True

            def prefetch_backward(self, **kwargs):
                self.backward_prefetch_kwargs.append(kwargs)
                self._backward_prefetched = True
                self.events.append("backward_prefetch")
                return True

            def wait_post_backward_reduce(self):
                self.has_pending_backward_reduce = False
                self.events.append("wait_post_backward_reduce")

            def wait_unshard(self, reason):
                self._unshard_inflight = False
                self.events.append(f"wait_unshard:{reason}")

            def _record_event(self, name):
                self.events.append(name)

        units = [FakeUnit(index) for index in range(3)]
        scheduler = MatrixFSDPScheduler(units, max_unsharded_prefetch_units=1)
        for unit in units:
            scheduler.record_post_forward(unit)

        scheduler.on_pre_backward(units[1])
        self.assertNotIn("backward_prefetch", units[0].events)
        self.assertIn("backward_prefetch_skipped:owner_ordered_queue", " ".join(units[0].events))

        scheduler.on_pre_backward(units[2])
        scheduler.on_pre_backward(units[1])

        self.assertEqual(units[1].events.count("backward_prefetch"), 1)
        self.assertEqual(units[0].events.count("backward_prefetch"), 1)
        self.assertEqual(scheduler.backward_prefetch_issued, 2)
        for unit in units[:2]:
            self.assertEqual(
                unit.backward_prefetch_kwargs,
                [{"validate_owner_collective_signature": True, "ordered_owner_collective": True}],
            )

        units = [FakeUnit(index) for index in range(3)]
        scheduler = MatrixFSDPScheduler(units, max_unsharded_prefetch_units=1)
        for unit in units:
            scheduler.record_post_forward(unit)
        units[2].has_pending_backward_reduce = True
        scheduler._pending_backward_reduce_units.append(units[2])

        scheduler.on_pre_backward(units[2])

        self.assertNotIn("wait_post_backward_reduce", units[2].events)
        self.assertNotIn("backward_prefetch", units[1].events)
        self.assertIn("backward_prefetch_skipped:owner_ordered_queue:pending_reduce", " ".join(units[1].events))
        self.assertEqual(scheduler.backward_prefetch_memory_deferred, 1)

        units = [FakeUnit(index) for index in range(3)]
        units[1].flat_buffer = FakeOwnerFlatBuffer(independent_comm_lanes=True)
        scheduler = MatrixFSDPScheduler(units, max_unsharded_prefetch_units=1)
        for unit in units:
            scheduler.record_post_forward(unit)
        units[2].has_pending_backward_reduce = True
        scheduler._pending_backward_reduce_units.append(units[2])

        scheduler.on_pre_backward(units[2])

        self.assertIn("backward_prefetch", units[1].events)
        self.assertEqual(scheduler.backward_prefetch_issued, 1)
        self.assertEqual(scheduler.backward_prefetch_memory_deferred, 0)

        units = [FakeUnit(index) for index in range(3)]
        scheduler = MatrixFSDPScheduler(units, max_unsharded_prefetch_units=1)
        for unit in units:
            scheduler.record_post_forward(unit)
        units[2].has_pending_backward_reduce = True
        scheduler._pending_backward_reduce_units.append(units[2])

        with unittest.mock.patch.dict(
            "os.environ",
            {"MATRIX_FSDP_OWNER_BACKWARD_PREFETCH_WAIT_PENDING_REDUCE": "1"},
        ):
            scheduler.on_pre_backward(units[2])

        self.assertIn("wait_post_backward_reduce", units[2].events)
        self.assertIn("backward_prefetch", units[1].events)
        self.assertIn(
            "backward_prefetch_skipped:owner_ordered_queue:waited_pending_reduce",
            " ".join(units[1].events),
        )
        self.assertEqual(scheduler.backward_prefetch_pending_reduce_waits, 1)

        units = [FakeUnit(index) for index in range(3)]
        scheduler = MatrixFSDPScheduler(units, max_unsharded_prefetch_units=1)
        units[1]._backward_prefetched = True
        units[1]._unshard_inflight = True

        scheduler.before_backward_reduce(units[2])

        self.assertIn("wait_unshard:owner_prefetch_before_reduce", units[1].events)
        self.assertEqual(scheduler.owner_prefetch_waits_before_reduce, 1)

    def test_owner_prefetch_queue_fills_forward_budget_window(self):
        class FakeOwnerFlatBuffer:
            local_shard = None
            matrix_collective_backend = "custom"

            def owner_segment_prefetch_order_gate_required(self):
                return True

            def owner_segment_collective_has_independent_comm_lanes(self):
                return True

        class FakeRuntimeMetadata:
            def __init__(self, index):
                self.runtime_param_group_id = f"param_group_{index}"

        class FakeUnit:
            def __init__(self, index):
                self.forward_prefetch_enabled = True
                self.backward_prefetch_enabled = True
                self.flat_buffer = FakeOwnerFlatBuffer()
                self.runtime_metadata = FakeRuntimeMetadata(index)
                self.world_size = 8
                self._forward_prefetched = False
                self._backward_prefetched = False
                self.has_pending_backward_reduce = False
                self.events = []

            def set_scheduler(self, scheduler):
                self.scheduler = scheduler

            def set_comm_context(self, comm_context):
                self.comm_context = comm_context

            def set_full_param_buffer_pool(self, pool):
                self.pool = pool

            def prefetch_forward(self, **kwargs):
                self._forward_prefetched = True
                self.events.append("forward_prefetch")
                return True

            def prefetch_backward(self, **kwargs):
                self._backward_prefetched = True
                self.events.append("backward_prefetch")
                return True

            def wait_post_backward_reduce(self):
                self.has_pending_backward_reduce = False

            def wait_unshard(self, reason):
                self._unshard_inflight = False

            def _record_event(self, name):
                self.events.append(name)

        units = [FakeUnit(index) for index in range(4)]
        scheduler = MatrixFSDPScheduler(units, max_unsharded_prefetch_units=2)

        scheduler.on_pre_forward(units[0])

        self.assertEqual(units[1].events.count("forward_prefetch"), 1)
        self.assertEqual(units[2].events.count("forward_prefetch"), 1)
        self.assertEqual(units[3].events.count("forward_prefetch"), 0)
        self.assertEqual(scheduler.forward_prefetch_issued, 2)

        scheduler.on_pre_forward(units[1])

        self.assertEqual(units[3].events.count("forward_prefetch"), 1)
        self.assertEqual(scheduler.forward_prefetch_issued, 3)

    def test_owner_prefetch_queue_refills_forward_window_after_post_forward(self):
        class FakeOwnerFlatBuffer:
            matrix_collective_backend = "custom"

            def __init__(self):
                class FakePlan:
                    total_numel = 1

                self.plan = FakePlan()
                self.local_shard = torch.empty(1)
                self.full_buffer = None

            def owner_segment_prefetch_order_gate_required(self):
                return True

            def owner_segment_collective_has_independent_comm_lanes(self):
                return True

        class FakeRuntimeMetadata:
            def __init__(self, index):
                self.runtime_param_group_id = f"param_group_{index}"

        class FakeUnit:
            def __init__(self, index):
                self.forward_prefetch_enabled = True
                self.backward_prefetch_enabled = True
                self.flat_buffer = FakeOwnerFlatBuffer()
                self.runtime_metadata = FakeRuntimeMetadata(index)
                self.world_size = 8
                self._forward_prefetched = False
                self._backward_prefetched = False
                self.has_pending_backward_reduce = False
                self.events = []

            def set_scheduler(self, scheduler):
                self.scheduler = scheduler

            def set_comm_context(self, comm_context):
                self.comm_context = comm_context

            def set_full_param_buffer_pool(self, pool):
                self.pool = pool

            def prefetch_forward(self, **kwargs):
                self._forward_prefetched = True
                self.flat_buffer.full_buffer = torch.empty(1)
                self.events.append("forward_prefetch")
                return True

            def prefetch_backward(self, **kwargs):
                self._backward_prefetched = True
                self.events.append("backward_prefetch")
                return True

            def wait_post_backward_reduce(self):
                self.has_pending_backward_reduce = False

            def wait_unshard(self, reason):
                self._unshard_inflight = False

            def _record_event(self, name):
                self.events.append(name)

        units = [FakeUnit(index) for index in range(4)]
        units[0].flat_buffer.full_buffer = torch.empty(1)
        scheduler = MatrixFSDPScheduler(units, max_unsharded_prefetch_units=2, max_active_full_param_buffers=2)

        scheduler.on_pre_forward(units[0])

        self.assertEqual(units[1].events.count("forward_prefetch"), 1)
        self.assertEqual(units[2].events.count("forward_prefetch"), 0)
        self.assertEqual(units[3].events.count("forward_prefetch"), 0)

        units[0]._forward_prefetched = False
        units[0].flat_buffer.full_buffer = None
        scheduler.record_post_forward(units[0])

        self.assertEqual(units[2].events.count("forward_prefetch"), 1)
        self.assertEqual(units[3].events.count("forward_prefetch"), 0)
        self.assertEqual(scheduler.forward_prefetch_issued, 2)

    def test_owner_prefetch_queue_fills_backward_budget_window(self):
        class FakeOwnerFlatBuffer:
            local_shard = None
            matrix_collective_backend = "custom"

            def owner_segment_prefetch_order_gate_required(self):
                return True

            def owner_segment_collective_has_independent_comm_lanes(self):
                return True

        class FakeRuntimeMetadata:
            def __init__(self, index):
                self.runtime_param_group_id = f"param_group_{index}"

        class FakeUnit:
            def __init__(self, index):
                self.forward_prefetch_enabled = True
                self.backward_prefetch_enabled = True
                self.flat_buffer = FakeOwnerFlatBuffer()
                self.runtime_metadata = FakeRuntimeMetadata(index)
                self.world_size = 8
                self._forward_prefetched = False
                self._backward_prefetched = False
                self.has_pending_backward_reduce = False
                self.events = []

            def set_scheduler(self, scheduler):
                self.scheduler = scheduler

            def set_comm_context(self, comm_context):
                self.comm_context = comm_context

            def set_full_param_buffer_pool(self, pool):
                self.pool = pool

            def prefetch_forward(self, **kwargs):
                self._forward_prefetched = True
                self.events.append("forward_prefetch")
                return True

            def prefetch_backward(self, **kwargs):
                self._backward_prefetched = True
                self.events.append("backward_prefetch")
                return True

            def wait_post_backward_reduce(self):
                self.has_pending_backward_reduce = False

            def wait_unshard(self, reason):
                self._unshard_inflight = False

            def _record_event(self, name):
                self.events.append(name)

        units = [FakeUnit(index) for index in range(4)]
        scheduler = MatrixFSDPScheduler(units, max_backward_prefetch_units=2)
        for unit in units:
            scheduler.record_post_forward(unit)

        scheduler.on_pre_backward(units[3])

        self.assertEqual(units[2].events.count("backward_prefetch"), 1)
        self.assertEqual(units[1].events.count("backward_prefetch"), 1)
        self.assertEqual(units[0].events.count("backward_prefetch"), 0)
        self.assertEqual(scheduler.backward_prefetch_issued, 2)

        scheduler.on_pre_backward(units[2])

        self.assertEqual(units[0].events.count("backward_prefetch"), 1)
        self.assertEqual(scheduler.backward_prefetch_issued, 3)

    def test_mixed_optimizer_inner_adamw_does_not_replace_unit_scheduler(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        scheduler = MatrixFSDPScheduler([unit])

        optim = MixedMuonAdamWOptimizer((), model.parameters())

        self.assertIs(unit._scheduler, scheduler)
        self.assertIsNone(getattr(optim.adamw, "matrix_fsdp", None))

    def test_backward_prefetch_uses_actual_post_forward_order(self):
        class BranchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = matrix_fully_shard(
                    nn.Linear(4, 8),
                    reshard_after_forward=True,
                    backward_prefetch=True,
                    finalize_after_backward=True,
                )
                self.unused = matrix_fully_shard(
                    nn.Linear(8, 8),
                    reshard_after_forward=True,
                    backward_prefetch=True,
                    finalize_after_backward=True,
                )
                self.last = matrix_fully_shard(
                    nn.Linear(8, 2),
                    reshard_after_forward=True,
                    backward_prefetch=True,
                    finalize_after_backward=True,
                )

            def forward(self, x):
                return self.last(torch.relu(self.first(x)))

        torch.manual_seed(0)
        model = BranchModel()
        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_forward_prefetch_units=0,
            max_backward_prefetch_units=1,
        )

        loss = model(torch.randn(2, 4)).sum()
        loss.backward()

        first_events = [event.name for event in model.first._matrix_fsdp_param_group.runtime_events]
        unused_events = [event.name for event in model.unused._matrix_fsdp_param_group.runtime_events]
        self.assertIn("backward_prefetch", first_events)
        self.assertNotIn("backward_prefetch", unused_events)
        self.assertEqual(optim.scheduler.backward_prefetch_issued, 1)

    def test_late_backward_prefetch_runs_after_current_unit_reshard(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            matrix_fully_shard(
                nn.Linear(4, 8),
                reshard_after_forward=True,
                backward_prefetch=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
            ),
            nn.ReLU(),
            matrix_fully_shard(
                nn.Linear(8, 2),
                reshard_after_forward=True,
                backward_prefetch=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
            ),
        )
        first_unit = model[0]._matrix_fsdp_param_group
        last_unit = model[2]._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_forward_prefetch_units=0,
            max_backward_prefetch_units=1,
            backward_prefetch_timing="post_reshard",
        )

        model(torch.randn(3, 4)).sum().backward()

        self.assertIn("backward_prefetch", [event.name for event in first_unit.runtime_events])
        self.assertLess(
            self._event_sequence(last_unit, "reshard_before_reduce_grad"),
            self._event_sequence(first_unit, "backward_prefetch"),
        )
        self.assertLess(
            self._event_sequence(last_unit, "start_reduce_grad_bucket"),
            self._event_sequence(first_unit, "backward_prefetch"),
        )
        self.assertEqual(optim.scheduler.backward_prefetch_issued, 1)

    def test_fsdp2_backward_prefetch_runs_before_current_unit_reduce(self):
        torch.manual_seed(0)
        model = nn.Sequential(
            matrix_fully_shard(
                nn.Linear(4, 8),
                reshard_after_forward=True,
                backward_prefetch=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
            ),
            nn.ReLU(),
            matrix_fully_shard(
                nn.Linear(8, 2),
                reshard_after_forward=True,
                backward_prefetch=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
            ),
        )
        first_unit = model[0]._matrix_fsdp_param_group
        last_unit = model[2]._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_forward_prefetch_units=0,
            max_backward_prefetch_units=1,
            backward_prefetch_timing="pre_backward",
        )

        model(torch.randn(3, 4)).sum().backward()

        self.assertIn("backward_prefetch", [event.name for event in first_unit.runtime_events])
        self.assertLess(
            self._event_sequence(last_unit, "pre_backward_unshard"),
            self._event_sequence(first_unit, "backward_prefetch"),
        )
        self.assertLess(
            self._event_sequence(first_unit, "backward_prefetch"),
            self._event_sequence(last_unit, "reshard_before_reduce_grad"),
        )
        self.assertLess(
            self._event_sequence(first_unit, "backward_prefetch"),
            self._event_sequence(last_unit, "start_reduce_grad_bucket"),
        )
        self.assertEqual(optim.scheduler.backward_prefetch_issued, 1)

    def test_backward_prefetch_timing_tracks_active_full_buffer_peak(self):
        def run_step(
            backward_prefetch_timing: str,
            *,
            max_active_full_param_buffers: int | None = None,
            max_active_full_param_numel: int | None = None,
            max_active_full_param_memory_mb: float | None = None,
        ) -> MatrixFSDPOptimizer:
            torch.manual_seed(0)
            model = nn.Sequential(
                matrix_fully_shard(
                    nn.Linear(4, 8),
                    reshard_after_forward=True,
                    backward_prefetch=True,
                    finalize_after_backward=True,
                    backward_reduce_strategy="bucket_reduce_scatter",
                ),
                nn.ReLU(),
                matrix_fully_shard(
                    nn.Linear(8, 2),
                    reshard_after_forward=True,
                    backward_prefetch=True,
                    finalize_after_backward=True,
                    backward_reduce_strategy="bucket_reduce_scatter",
                ),
            )
            optim = MatrixFSDPOptimizer(
                torch.optim.SGD(model.parameters(), lr=0.1),
                model,
                max_forward_prefetch_units=0,
                max_backward_prefetch_units=1,
                backward_prefetch_timing=backward_prefetch_timing,
                max_active_full_param_buffers=max_active_full_param_buffers,
                max_active_full_param_numel=max_active_full_param_numel,
                max_active_full_param_memory_mb=max_active_full_param_memory_mb,
            )
            model(torch.randn(3, 4)).sum().backward()
            return optim

        fsdp2_like_optim = run_step("pre_backward")
        memory_capped_optim = run_step("pre_backward", max_active_full_param_buffers=1)
        numel_capped_optim = run_step("pre_backward", max_active_full_param_numel=57)
        memory_mb_capped_optim = run_step("pre_backward", max_active_full_param_memory_mb=57 * 4 / (1024 * 1024))
        numel_uncapped_optim = run_step("pre_backward", max_active_full_param_numel=58)
        late_optim = run_step("post_reshard")

        self.assertGreaterEqual(fsdp2_like_optim.scheduler.max_active_full_param_buffers, 2)
        self.assertEqual(memory_capped_optim.scheduler.max_active_full_param_buffers, 1)
        self.assertEqual(memory_capped_optim.scheduler.backward_prefetch_memory_deferred, 1)
        self.assertEqual(numel_capped_optim.scheduler.max_active_full_param_buffers, 1)
        self.assertEqual(numel_capped_optim.scheduler.backward_prefetch_memory_deferred, 1)
        self.assertEqual(memory_mb_capped_optim.scheduler.max_active_full_param_buffers, 1)
        self.assertEqual(memory_mb_capped_optim.scheduler.backward_prefetch_memory_deferred, 1)
        self.assertEqual(numel_uncapped_optim.scheduler.max_active_full_param_numel, 58)
        self.assertEqual(numel_uncapped_optim.scheduler.backward_prefetch_memory_deferred, 0)
        self.assertEqual(late_optim.scheduler.max_active_full_param_buffers, 1)
        self.assertTrue(
            any(
                snapshot["reason"] == "start_unshard:backward_prefetch"
                and snapshot["active_count"] >= 2
                and snapshot["active_bytes"] > 0
                for snapshot in fsdp2_like_optim.scheduler.full_param_buffer_snapshots
            )
        )

    def test_optimizer_creates_scheduler_for_collected_units(self):
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8), reshard_after_forward=True, forward_prefetch=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2), reshard_after_forward=True, forward_prefetch=True),
        )

        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        self.assertEqual(optim.scheduler.units, tuple(optim.runtime_param_groups))
        self.assertIs(optim.runtime_param_groups[0]._scheduler, optim.scheduler)
        self.assertIs(optim.runtime_param_groups[1]._scheduler, optim.scheduler)

    def test_optimizer_static_prefetch_budget_defaults_to_fast_cap1(self):
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8)),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2)),
        )

        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        self.assertEqual(optim.scheduler.max_unsharded_prefetch_units, 1)
        self.assertEqual(optim.scheduler.max_forward_prefetch_units, 1)
        self.assertEqual(optim.scheduler.max_backward_prefetch_units, 1)
        self.assertEqual(optim.scheduler.full_param_buffer_pool.stats()["max_cached_per_key"], 0)
        self.assertFalse(optim.scheduler.trim_cuda_cache)
        self.assertFalse(optim.scheduler.maybe_trim_cuda_cache())
        for unit in optim.runtime_param_groups:
            for param_buffer in (unit.flat_buffer.static_param_buffer, unit.flat_buffer.elastic_param_buffer):
                stats = param_buffer.workspace.stats()
                self.assertEqual(stats["workspace_max_cached_per_key"], 0)

    def test_api_defaults_to_fast_copy_in_no_saved_hooks_path(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group

        self.assertEqual(unit.runtime_layout_compatibility.mode, "matrix_shard")
        self.assertEqual(unit.backward_reduce_strategy, "bucket_reduce_scatter")
        self.assertFalse(unit.use_saved_tensor_hooks)
        self.assertFalse(unit.use_zero_copy_grad_bucket)

    def test_optimizer_accepts_scheduler_config_object(self):
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8), reshard_after_forward=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2), reshard_after_forward=True),
        )
        config = MatrixFSDPSchedulerConfig(
            max_forward_prefetch_units=0,
            max_backward_prefetch_units=0,
            max_cached_full_param_buffers_per_key=2,
            max_cached_elastic_workspaces_per_key=1,
            max_pending_backward_reduces=0,
            trim_cuda_cache=True,
        )

        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            scheduler_config=config,
        )

        self.assertEqual(optim.scheduler.max_forward_prefetch_units, 0)
        self.assertEqual(optim.scheduler.max_backward_prefetch_units, 0)
        self.assertEqual(optim.scheduler.max_pending_backward_reduces, 0)
        self.assertTrue(optim.scheduler.trim_cuda_cache)
        self.assertEqual(optim.scheduler.full_param_buffer_pool.stats()["max_cached_per_key"], 2)
        for unit in optim.runtime_param_groups:
            for param_buffer in (unit.flat_buffer.static_param_buffer, unit.flat_buffer.elastic_param_buffer):
                stats = param_buffer.workspace.stats()
                self.assertEqual(stats["workspace_max_cached_per_key"], 1)

    def test_optimizer_rejects_scheduler_config_with_explicit_scheduler_kwargs(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        with self.assertRaisesRegex(ValueError, "scheduler_config"):
            MatrixFSDPOptimizer(
                torch.optim.SGD(model.parameters(), lr=0.1),
                model,
                scheduler_config=MatrixFSDPSchedulerConfig(max_forward_prefetch_units=0),
                max_forward_prefetch_units=1,
            )

    def test_optimizer_does_not_cache_full_param_buffer_by_default(self):
        model = nn.Sequential(
            matrix_fully_shard(
                nn.Linear(4, 8, bias=False),
                reshard_after_forward=True,
                use_saved_tensor_hooks=False,
            ),
            nn.ReLU(),
            matrix_fully_shard(
                nn.Linear(8, 4, bias=False),
                reshard_after_forward=True,
                use_saved_tensor_hooks=False,
            ),
        )
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optim.step()

        stats = optim.scheduler.full_param_buffer_pool.stats()
        self.assertEqual(stats["max_cached_per_key"], 0)
        self.assertEqual(stats["cached_buffers"], 0)
        for param_group in optim.runtime_param_groups:
            self._assert_full_buffer_released_or_shrunk(param_group.flat_buffer.full_buffer)

    def test_optimizer_can_disable_backward_prefetch_independently(self):
        model = nn.Sequential(
            matrix_fully_shard(
                nn.Linear(4, 8),
                reshard_after_forward=True,
                backward_prefetch=True,
                finalize_after_backward=False,
                backward_reduce_strategy="flat",
            ),
            nn.ReLU(),
            matrix_fully_shard(
                nn.Linear(8, 2),
                reshard_after_forward=True,
                backward_prefetch=True,
                finalize_after_backward=False,
                backward_reduce_strategy="flat",
            ),
        )

        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_unsharded_prefetch_units=1,
            max_backward_prefetch_units=0,
        )

        self.assertEqual(optim.scheduler.max_unsharded_prefetch_units, 1)
        self.assertEqual(optim.scheduler.max_forward_prefetch_units, 1)
        self.assertEqual(optim.scheduler.max_backward_prefetch_units, 0)

        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        event_names = [event.name for unit in optim.runtime_param_groups for event in unit.runtime_events]

        self.assertNotIn("backward_prefetch", event_names)
        self.assertTrue(all(unit.lifecycle_state == FSDPLifecycleState.UNSHARDED for unit in optim.runtime_param_groups))

    def test_forward_prefetch_budget_does_not_block_backward_prefetch(self):
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8), reshard_after_forward=True, backward_prefetch=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2), reshard_after_forward=True, backward_prefetch=True),
        )

        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            max_unsharded_prefetch_units=1,
        )

        loss = model(torch.randn(3, 4)).sum()
        loss.backward()
        event_names = [event.name for unit in optim.runtime_param_groups for event in unit.runtime_events]

        self.assertEqual(optim.scheduler.max_forward_prefetch_units, 1)
        self.assertEqual(optim.scheduler.max_backward_prefetch_units, 1)
        self.assertIn("backward_prefetch", event_names)

    def test_optimizer_adaptive_prefetch_policy_resolves_budget_from_shard_mesh_size(self):
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8), reshard_after_forward=True, forward_prefetch=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2), reshard_after_forward=True, forward_prefetch=True),
        )
        units = [model[0]._matrix_fsdp_param_group, model[2]._matrix_fsdp_param_group]
        for unit in units:
            unit.world_size = 8

        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            prefetch_policy="adaptive",
        )

        self.assertEqual(optim.scheduler.prefetch_policy, "adaptive")
        self.assertEqual(optim.scheduler.max_unsharded_prefetch_units, 1)
        self.assertEqual(optim.scheduler.max_forward_prefetch_units, 1)
        self.assertEqual(optim.scheduler.max_backward_prefetch_units, 0)

        model(torch.randn(3, 4)).sum()
        event_names = [event.name for unit in optim.runtime_param_groups for event in unit.runtime_events]
        self.assertIn("forward_prefetch", event_names)

    def test_optimizer_adaptive_prefetch_policy_allows_small_shard_mesh_prefetch(self):
        model = nn.Sequential(
            matrix_fully_shard(nn.Linear(4, 8), reshard_after_forward=True, forward_prefetch=True),
            nn.ReLU(),
            matrix_fully_shard(nn.Linear(8, 2), reshard_after_forward=True, forward_prefetch=True),
        )

        optim = MatrixFSDPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            model,
            prefetch_policy="adaptive",
        )

        self.assertEqual(optim.scheduler.max_unsharded_prefetch_units, 2)
        self.assertEqual(optim.scheduler.max_forward_prefetch_units, 2)
        self.assertEqual(optim.scheduler.max_backward_prefetch_units, 1)

    def test_scheduler_adaptive_policy_uses_mesh_shard_dim_over_global_world_size(self):
        class FakeMesh:
            def __init__(self, shape):
                self.shape = shape

            def size(self, mesh_dim):
                return self.shape[mesh_dim]

        class FakeUnit:
            mesh = FakeMesh((2, 4))
            dp_shard_mesh_dim = 0
            world_size = 8
            _is_unsharded = False

            def set_scheduler(self, scheduler):
                self.scheduler = scheduler

        scheduler = MatrixFSDPScheduler([FakeUnit(), FakeUnit()], prefetch_policy="adaptive")

        self.assertEqual(scheduler.max_unsharded_prefetch_units, 2)
        self.assertEqual(scheduler.max_forward_prefetch_units, 2)
        self.assertEqual(scheduler.max_backward_prefetch_units, 1)

    def test_scheduler_adaptive_policy_uses_named_shard_dim_size(self):
        class FakeMesh:
            def size(self, mesh_dim):
                self.mesh_dim = mesh_dim
                return 4

        class FakeUnit:
            mesh = FakeMesh()
            dp_shard_mesh_dim = 1
            world_size = 8
            _is_unsharded = False

            def set_scheduler(self, scheduler):
                self.scheduler = scheduler

        scheduler = MatrixFSDPScheduler([FakeUnit(), FakeUnit()], prefetch_policy="adaptive")

        self.assertEqual(scheduler.max_unsharded_prefetch_units, 1)
        self.assertEqual(scheduler.max_forward_prefetch_units, 1)
        self.assertEqual(scheduler.max_backward_prefetch_units, 0)
        self.assertEqual(FakeUnit.mesh.mesh_dim, 1)

    def test_scheduler_profile_guided_policy_selects_fastest_budget(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        scheduler = MatrixFSDPScheduler([model._matrix_fsdp_param_group], prefetch_policy="profile_guided")
        results = (
            PrefetchProfileResult(budget=None, avg_step_ms=3.0, peak_memory_mb=20.0),
            PrefetchProfileResult(budget=0, avg_step_ms=5.0, peak_memory_mb=10.0),
            PrefetchProfileResult(budget=1, avg_step_ms=2.0, peak_memory_mb=12.0),
        )

        scheduler.set_profile_results(results)
        selected = scheduler.select_profiled_budget()

        self.assertEqual(selected, 1)
        self.assertEqual(scheduler.max_unsharded_prefetch_units, 1)
        self.assertEqual(scheduler.selected_prefetch_budget, 1)
        self.assertEqual(scheduler.selected_forward_prefetch_budget, 1)
        self.assertEqual(scheduler.selected_backward_prefetch_budget, 1)
        self.assertEqual(scheduler.profile_results, results)

    def test_scheduler_profile_guided_policy_respects_memory_limit(self):
        model = matrix_fully_shard(nn.Linear(4, 2))
        scheduler = MatrixFSDPScheduler([model._matrix_fsdp_param_group], prefetch_policy="profile_guided")
        scheduler.set_profile_results(
            (
                PrefetchProfileResult(budget=None, avg_step_ms=1.0, peak_memory_mb=30.0),
                PrefetchProfileResult(budget=1, avg_step_ms=2.0, peak_memory_mb=12.0),
                PrefetchProfileResult(budget=0, avg_step_ms=4.0, peak_memory_mb=8.0),
            )
        )

        selected = scheduler.select_profiled_budget(memory_limit_mb=15.0)

        self.assertEqual(selected, 1)
        self.assertEqual(scheduler.max_unsharded_prefetch_units, 1)
        self.assertEqual(scheduler.max_forward_prefetch_units, 1)
        self.assertEqual(scheduler.max_backward_prefetch_units, 1)

    def test_scheduler_rejects_negative_prefetch_budget(self):
        model = matrix_fully_shard(nn.Linear(4, 2))

        with self.assertRaisesRegex(ValueError, "non-negative"):
            MatrixFSDPScheduler([model._matrix_fsdp_param_group], max_unsharded_prefetch_units=-1)

        with self.assertRaisesRegex(ValueError, "max_pending_backward_reduces"):
            MatrixFSDPScheduler([model._matrix_fsdp_param_group], max_pending_backward_reduces=-1)

    def test_scheduler_rejects_unknown_prefetch_policy(self):
        model = matrix_fully_shard(nn.Linear(4, 2))

        with self.assertRaisesRegex(ValueError, "prefetch_policy"):
            MatrixFSDPScheduler([model._matrix_fsdp_param_group], prefetch_policy="unknown")  # type: ignore[arg-type]

    def test_multi_unit_reshard_after_forward_step_matches_eager_model(self):
        torch.manual_seed(0)
        base_model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(base_model)
        sharded_model = nn.Sequential(
            matrix_fully_shard(
                base_model[0],
                reshard_after_forward=True,
                finalize_after_backward=False,
                backward_reduce_strategy="flat",
            ),
            base_model[1],
            matrix_fully_shard(
                base_model[2],
                reshard_after_forward=True,
                finalize_after_backward=False,
                backward_reduce_strategy="flat",
            ),
        )
        units = [sharded_model[0]._matrix_fsdp_param_group, sharded_model[2]._matrix_fsdp_param_group]

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)

        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        self.assertTrue(all(unit.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED for unit in units))
        for unit in units:
            self._assert_full_buffer_released_or_shrunk(unit.flat_buffer.full_buffer)

        sharded_loss.backward()
        self.assertTrue(all(unit.lifecycle_state == FSDPLifecycleState.UNSHARDED for unit in units))
        self.assertTrue(all(unit.flat_buffer.full_buffer is not None for unit in units))

        sharded_optim.step()
        self.assertTrue(all(unit.lifecycle_state == FSDPLifecycleState.SHARDED for unit in units))
        sharded_optim.zero_grad()
        for unit in units:
            unit.unshard()

        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            self.assertEqual(eager_param.shape, sharded_param.shape)
            torch.testing.assert_close(eager_param, sharded_param)

    def test_multi_unit_forward_prefetch_unshards_next_unit_before_its_forward(self):
        torch.manual_seed(0)
        base_model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(base_model)
        sharded_model = nn.Sequential(
            matrix_fully_shard(
                base_model[0],
                reshard_after_forward=True,
                forward_prefetch=True,
                finalize_after_backward=False,
                backward_reduce_strategy="flat",
            ),
            base_model[1],
            matrix_fully_shard(
                base_model[2],
                reshard_after_forward=True,
                forward_prefetch=True,
                finalize_after_backward=False,
                backward_reduce_strategy="flat",
            ),
        )
        units = [sharded_model[0]._matrix_fsdp_param_group, sharded_model[2]._matrix_fsdp_param_group]

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        summary = summarize_runtime_events(sharded_model)
        self.assertLess(
            self._summary_event_sequence(summary, 1, "forward_prefetch"),
            self._summary_event_sequence(summary, 1, "pre_forward"),
        )
        self.assertLess(
            self._summary_event_sequence(summary, 1, "start_unshard:forward_prefetch"),
            self._summary_event_sequence(summary, 1, "pre_forward"),
        )
        self.assertLess(
            self._summary_event_sequence(summary, 1, "pre_forward"),
            self._summary_event_sequence(summary, 1, "wait_unshard:pre_forward"),
        )
        self.assertTrue(all(unit.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED for unit in units))

        sharded_loss.backward()
        self.assertTrue(all(unit.lifecycle_state == FSDPLifecycleState.UNSHARDED for unit in units))
        sharded_optim.step()
        for unit in units:
            unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_multi_unit_backward_prefetch_unshards_previous_unit_before_its_backward(self):
        torch.manual_seed(0)
        base_model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(base_model)
        sharded_model = nn.Sequential(
            matrix_fully_shard(
                base_model[0],
                reshard_after_forward=True,
                backward_prefetch=True,
                finalize_after_backward=False,
                backward_reduce_strategy="flat",
            ),
            base_model[1],
            matrix_fully_shard(
                base_model[2],
                reshard_after_forward=True,
                backward_prefetch=True,
                finalize_after_backward=False,
                backward_reduce_strategy="flat",
            ),
        )
        units = [sharded_model[0]._matrix_fsdp_param_group, sharded_model[2]._matrix_fsdp_param_group]

        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        self.assertTrue(all(unit.lifecycle_state == FSDPLifecycleState.FORWARD_RESHARDED for unit in units))

        sharded_loss.backward()
        summary = summarize_runtime_events(sharded_model)
        self.assertLess(
            self._summary_event_sequence(summary, 0, "backward_prefetch"),
            self._summary_event_sequence(summary, 0, "pre_backward_unshard"),
        )
        self.assertLess(
            self._summary_event_sequence(summary, 0, "start_unshard:backward_prefetch"),
            self._summary_event_sequence(summary, 0, "pre_backward_unshard"),
        )
        self.assertLess(
            self._summary_event_sequence(summary, 0, "pre_backward_unshard"),
            self._summary_event_sequence(summary, 0, "wait_unshard:pre_backward"),
        )
        self.assertTrue(all(unit.lifecycle_state == FSDPLifecycleState.UNSHARDED for unit in units))

        sharded_optim.step()
        for unit in units:
            unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_wrap_policy_shards_matching_modules(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(model)

        sharded_model = matrix_fully_shard(
            model,
            wrap_policy=module_type_policy(nn.Linear),
            reshard_after_forward=True,
        )
        sharded_optim = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.1), sharded_model)
        eager_optim = torch.optim.SGD(eager_model.parameters(), lr=0.1)

        self.assertFalse(hasattr(sharded_model, "_matrix_fsdp_param_group"))
        self.assertEqual(len(sharded_optim.runtime_param_groups), 2)
        self.assertIs(sharded_optim.runtime_param_groups[0], sharded_model[0]._matrix_fsdp_param_group)
        self.assertIs(sharded_optim.runtime_param_groups[1], sharded_model[2]._matrix_fsdp_param_group)

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        eager_loss = (eager_model(x) - y).pow(2).mean()
        eager_loss.backward()
        eager_optim.step()

        sharded_loss = (sharded_model(x) - y).pow(2).mean()
        torch.testing.assert_close(sharded_loss, eager_loss)
        sharded_loss.backward()
        sharded_optim.step()
        for unit in sharded_optim.runtime_param_groups:
            self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
            unit.unshard()

        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_wrap_policy_stops_at_selected_parent(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)

            def forward(self, x):
                return self.proj(x).relu()

        model = nn.Sequential(Block(), Block())

        matrix_fully_shard(model, wrap_policy=lambda module: isinstance(module, Block))
        optim = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), model)

        self.assertEqual(len(optim.runtime_param_groups), 2)
        self.assertIs(optim.runtime_param_groups[0], model[0]._matrix_fsdp_param_group)
        self.assertIs(optim.runtime_param_groups[1], model[1]._matrix_fsdp_param_group)
        self.assertFalse(hasattr(model[0].proj, "_matrix_fsdp_param_group"))
        self.assertFalse(hasattr(model[1].proj, "_matrix_fsdp_param_group"))

    def test_wrap_policy_rejects_no_selected_modules(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

        with self.assertRaisesRegex(ValueError, "did not select"):
            matrix_fully_shard(model, wrap_policy=lambda module: False)

    def test_wrap_policy_rejects_explicit_shard_hints(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

        with self.assertRaisesRegex(ValueError, "explicit shard_hints"):
            matrix_fully_shard(
                model,
                wrap_policy=lambda module: isinstance(module, nn.Linear),
                shard_hints={"0.weight": ParamShardHint(split_granularity="parameter")},
            )

    def test_empty_module_lifecycle_state_is_stable(self):
        model = matrix_fully_shard(nn.ReLU())
        unit = model._matrix_fsdp_param_group

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        unit.unshard()
        unit.finalize_backward()
        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        self.assertEqual(unit.state_dict(), {})

    def test_runtime_accepts_external_group_layout(self):
        model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))
        captured_fqns = None

        def group_planner(params, world_size):
            nonlocal captured_fqns
            captured_fqns = tuple(param.fqn for param in params)
            plan = contiguous_even_plan(sum(param.numel for param in params), world_size)
            rank_segments = plan.rank_segments
            param_layouts = tuple(
                ParamLayout(
                    fqn=param.fqn,
                    global_start=param.offset,
                    global_end=param.end,
                    segments=(
                        ParamSegment(
                            fqn=param.fqn,
                            rank=0,
                            global_start=param.offset,
                            global_end=param.end,
                            local_start=param.offset,
                        ),
                    ),
                )
                for param in params
            )
            return MatrixGroupLayout.from_rank_segments(
                total_numel=plan.total_numel,
                rank_segments=rank_segments,
                params=param_layouts,
            )

        sharded_model = matrix_fully_shard(model, group_planner=group_planner)
        unit = sharded_model._matrix_fsdp_param_group

        self.assertEqual(captured_fqns, ("0.weight", "0.bias", "1.weight", "1.bias"))
        self.assertIsNotNone(unit.layout)
        self.assertEqual(unit.layout.params, unit.group_layout.params)
        self.assertEqual(unit.global_layout, unit.group_layout)
        self.assertEqual(unit.params_for_rank(), captured_fqns)

    def test_runtime_attaches_shard_hints_to_managed_params(self):
        model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))
        hint = ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner")

        sharded_model = matrix_fully_shard(model, shard_hints={"0.weight": hint})
        unit = sharded_model._matrix_fsdp_param_group

        self.assertEqual(unit.param_registry.param("0.weight").shard_hint, hint)
        self.assertEqual(unit.state_dict()["shard_hints"]["0.weight"], hint)
        self.assertEqual(unit.param_registry.param("0.bias").shard_hint, ParamShardHint())

    def test_runtime_can_resolve_auto_shard_hints(self):
        model = nn.Sequential(nn.Linear(4, 2), nn.LayerNorm(2))

        sharded_model = matrix_fully_shard(model, auto_shard_hints=True)
        unit = sharded_model._matrix_fsdp_param_group

        self.assertEqual(
            unit.param_registry.param("0.weight").shard_hint,
            ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
        )
        self.assertEqual(
            unit.param_registry.param("1.weight").shard_hint,
            ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
        )

    def test_runtime_auto_shard_hints_accept_explicit_overrides(self):
        model = nn.Sequential(nn.Linear(4, 2))
        override = ParamShardHint(optimizer_type="adamw", split_granularity="row_block", block_shape=(1, 4))

        sharded_model = matrix_fully_shard(model, auto_shard_hints=True, shard_hints={"0.weight": override})
        unit = sharded_model._matrix_fsdp_param_group

        self.assertEqual(unit.param_registry.param("0.weight").shard_hint, override)

    def test_runtime_auto_planner_policy_sets_planner_evaluation(self):
        model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))

        sharded_model = matrix_fully_shard(model, auto_shard_hints=True, auto_planner_policy="muon_full_matrix")
        unit = sharded_model._matrix_fsdp_param_group

        self.assertIsNotNone(unit.planner_evaluation)
        self.assertEqual(unit.planner_evaluation.policy, "muon_full_matrix")

    def test_runtime_auto_planner_policy_accepts_target_block_units(self):
        model = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 2, bias=False))

        sharded_model = matrix_fully_shard(
            model,
            auto_planner_policy="balanced",
            target_block_units=4,
        )
        unit = sharded_model._matrix_fsdp_param_group

        self.assertEqual(unit.planner_evaluation.policy, "balanced")
        self.assertEqual(unit.planner_evaluation.name, "matrix_row_block")
        self.assertEqual(unit.planner_evaluation.report.blocks_by_kind, {"matrix_row_block": 10})
        self.assertEqual(unit.state_dict()["planner_metadata"]["resource_estimate"], unit.state_dict()["planner_resource_estimate"])

    def test_runtime_auto_planner_policy_accepts_muon_owner_assignment(self):
        model = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 2, bias=False), nn.LayerNorm(2))

        sharded_model = matrix_fully_shard(
            model,
            auto_shard_hints=True,
            auto_planner_policy="muon_shard_aware",
            owner_assignment="role_greedy",
        )
        unit = sharded_model._matrix_fsdp_param_group

        self.assertEqual(unit.planner_evaluation.policy, "muon_shard_aware")
        self.assertEqual(unit.planner_evaluation.name, "matrix_owner_tail_role_greedy")
        self.assertIn("muon_matrix_owner", unit.planner_evaluation.report.blocks_by_kind)
        self.assertIn("adamw_tail_param", unit.planner_evaluation.report.blocks_by_kind)

    def test_runtime_auto_planner_policy_rejects_irrelevant_owner_options(self):
        model = nn.Linear(4, 2)

        with self.assertRaisesRegex(ValueError, "only supported"):
            matrix_fully_shard(
                model,
                auto_planner_policy="balanced",
                owner_assignment="role_greedy",
            )

    def test_runtime_layout_policy_allows_matrix_shard_without_reorder(self):
        model = nn.Linear(4, 2)

        sharded_model = matrix_fully_shard(
            model,
            runtime_layout_policy="no_reorder",
        )
        unit = sharded_model._matrix_fsdp_param_group

        self.assertEqual(unit.state_dict()["runtime_layout_policy"], "no_reorder")
        self.assertEqual(unit.state_dict()["runtime_layout_mode"], "matrix_shard")
        self.assertFalse(unit.state_dict()["layout_flat_reordered"])
        self.assertIs(unit.global_layout, unit.group_layout)
        self.assertEqual(unit.state_dict()["planner_layout_contract"], unit.state_dict()["runtime_layout_contract"])

    def test_runtime_auto_policy_builds_runtime_contract_after_flat_reorder(self):
        model = nn.Sequential(nn.Linear(4, 2, bias=False), nn.Linear(2, 1, bias=False))

        sharded_model = matrix_fully_shard(
            model,
            group_planner=self._single_rank_flat_reorder_layout,
            runtime_layout_policy="auto",
        )
        unit = sharded_model._matrix_fsdp_param_group
        state = unit.state_dict()

        self.assertEqual(state["runtime_layout_policy"], "auto")
        self.assertEqual(state["runtime_layout_mode"], "flat_reorder")
        self.assertTrue(state["runtime_layout_requires_flat_reorder"])
        self.assertTrue(state["layout_flat_reordered"])
        self.assertNotEqual(state["planner_layout_contract"], state["runtime_layout_contract"])
        self.assertEqual(state["runtime_layout_contract"]["rank_units"], state["shard_sizes"])
        self.assertIs(unit.planner_layout_contract.layout, unit.global_layout)
        self.assertIs(unit.runtime_layout_contract.layout, unit.group_layout)

    def test_runtime_layout_policy_rejects_flat_reorder_when_no_reorder(self):
        model = nn.Sequential(nn.Linear(4, 2, bias=False), nn.Linear(2, 1, bias=False))

        with self.assertRaisesRegex(ValueError, "runtime_layout_policy='no_reorder'"):
            matrix_fully_shard(
                model,
                group_planner=self._single_rank_flat_reorder_layout,
                runtime_layout_policy="no_reorder",
            )

    def test_runtime_layout_policy_rejects_flat_reorder_when_matrix_only(self):
        model = nn.Sequential(nn.Linear(4, 2, bias=False), nn.Linear(2, 1, bias=False))

        with self.assertRaisesRegex(ValueError, "runtime_layout_policy='matrix_shard_only'"):
            matrix_fully_shard(
                model,
                group_planner=self._single_rank_flat_reorder_layout,
                runtime_layout_policy="matrix_shard_only",
            )

    def test_runtime_layout_policy_rejects_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "runtime_layout_policy"):
            matrix_fully_shard(nn.Linear(4, 2), runtime_layout_policy="unknown")

    def test_runtime_rejects_auto_policy_with_explicit_planner(self):
        model = nn.Linear(4, 2)

        with self.assertRaisesRegex(ValueError, "auto_planner_policy"):
            matrix_fully_shard(model, auto_planner_policy="balanced", group_planner=ordered_group_plan)

    def test_single_rank_ordered_group_plan_matches_eager_model(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(model)

        sharded_model = matrix_fully_shard(model, group_planner=ordered_group_plan)
        unit = sharded_model._matrix_fsdp_param_group
        self.assertIsNotNone(unit.group_layout)
        self.assertEqual(unit.group_layout.params_for_rank(0), unit.param_registry.fqns)

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

    def test_single_rank_hinted_ordered_group_plan_matches_eager_model(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8, bias=False), nn.ReLU(), nn.Linear(8, 2, bias=False))
        eager_model = copy.deepcopy(model)
        shard_hints = {
            "0.weight": ParamShardHint(split_granularity="row_block", block_shape=(2, 4)),
            "2.weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
        }

        def group_planner(params, world_size):
            return hinted_ordered_group_plan(params, world_size, default_granularity="block", target_block_units=8)

        sharded_model = matrix_fully_shard(model, group_planner=group_planner, shard_hints=shard_hints)
        unit = sharded_model._matrix_fsdp_param_group
        self.assertIsNotNone(unit.group_layout)
        self.assertIsNotNone(unit.flat_buffer.placement)
        self.assertEqual(unit.flat_buffer.placement.local_units, (1,))

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

    def test_runtime_rejects_both_group_and_legacy_layout_planners(self):
        model = nn.Linear(2, 2)

        def group_planner(params, world_size):
            return contiguous_even_plan(sum(param.numel for param in params), world_size)

        with self.assertRaisesRegex(ValueError, "Pass only one"):
            matrix_fully_shard(model, group_planner=group_planner, layout_planner=group_planner)

    def test_runtime_rejects_invalid_group_layout(self):
        model = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))

        def group_planner(params, world_size):
            plan = contiguous_even_plan(sum(param.numel for param in params), world_size)
            layout = MatrixGroupLayout.from_shard_plan(plan, params)
            return MatrixGroupLayout(
                total_numel=layout.total_numel,
                ranks=layout.ranks,
                params=(layout.params[1], layout.params[0], *layout.params[2:]),
            )

        with self.assertRaisesRegex(ValueError, "do not match managed params"):
            matrix_fully_shard(model, group_planner=group_planner)

    def test_single_rank_step_matches_eager_model(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        eager_model = copy.deepcopy(model)

        sharded_model = matrix_fully_shard(model)

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
        sharded_optim.zero_grad()
        self.assertTrue(all(param.grad is None for param in sharded_model.parameters()))
        sharded_model._matrix_fsdp_param_group.unshard()

        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            self.assertEqual(eager_param.shape, sharded_param.shape)
            torch.testing.assert_close(eager_param, sharded_param)


if __name__ == "__main__":
    unittest.main()
