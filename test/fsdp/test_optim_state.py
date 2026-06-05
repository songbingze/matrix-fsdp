import unittest
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch import nn

from matrix_fsdp import (
    DEFAULT_MUON_ADJUST_LR_FN,
    MixedMuonAdamWOptimizer,
    MatrixFSDPOptimizer,
    classify_matrix_optimizer_params,
    configure_optimizer,
    prepare_matrix_optimizer,
    matrix_fully_shard,
)
from matrix_fsdp.optim_state import MatrixFSDPOptimizerStateManager


class OptimizerStateTest(unittest.TestCase):
    def test_state_manager_wraps_adamw_tensor_state_as_matrix_state(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)))
        optim = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model, flat_adamw_state=True)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optim.step()

        self.assertIsInstance(optim.state_manager, MatrixFSDPOptimizerStateManager)
        state_tensors = optim.state_manager.iter_local_state_tensors()
        self.assertTrue(state_tensors)
        self.assertEqual({state.name for state in state_tensors}, {"exp_avg", "exp_avg_sq"})

        for state_tensor in state_tensors:
            self.assertEqual(tuple(state_tensor.sharded_state.global_shape), (state_tensor.tensor.numel(),))
            self.assertFalse(state_tensor.sharded_state.uses_dtensor)
            self.assertTrue(state_tensor.sharded_state.shares_storage_with(state_tensor.tensor))
            self.assertEqual(state_tensor.sharded_state.local_tensor.data_ptr(), state_tensor.tensor.view(-1).data_ptr())

        state_objects = optim.local_state_objects()
        self.assertIn("0.weight", state_objects)
        self.assertEqual(set(state_objects["0.weight"]), {"exp_avg", "exp_avg_sq"})

    def test_adamw_state_uses_flat_buffer_views(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2)))
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model, flat_adamw_state=True)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optim.step()

        flat_state_buffers = getattr(optim.optimizer, "_matrix_fsdp_flat_adamw_state_buffers")
        self.assertEqual(len(flat_state_buffers), 1)
        flat_state = next(iter(flat_state_buffers.values()))
        self.assertEqual(flat_state.exp_avg.numel(), unit.flat_buffer.local_numel)
        self.assertEqual(flat_state.exp_avg_sq.numel(), unit.flat_buffer.local_numel)

        exp_avg_storages = set()
        exp_avg_sq_storages = set()
        for param, state in optim.optimizer.state.items():
            if param.numel() == 0:
                continue
            self.assertIn("exp_avg", state)
            self.assertIn("exp_avg_sq", state)
            exp_avg_storages.add(state["exp_avg"].untyped_storage().data_ptr())
            exp_avg_sq_storages.add(state["exp_avg_sq"].untyped_storage().data_ptr())
        self.assertEqual(exp_avg_storages, {flat_state.exp_avg.untyped_storage().data_ptr()})
        self.assertEqual(exp_avg_sq_storages, {flat_state.exp_avg_sq.untyped_storage().data_ptr()})

    def test_adamw_flat_state_is_opt_in(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)))
        optim = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optim.step()

        self.assertFalse(hasattr(optim.optimizer, "_matrix_fsdp_flat_adamw_state_buffers"))

    def test_prepared_adamw_state_can_use_flat_buffer_views(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)))
        optim = torch.optim.AdamW(model.parameters(), lr=0.01)
        optim.matrix_fsdp.remove()
        prepare_matrix_optimizer(optim, model, flat_adamw_state=True)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optim.step()

        self.assertTrue(hasattr(optim, "matrix_fsdp"))
        self.assertTrue(hasattr(optim, "_matrix_fsdp_flat_adamw_state_buffers"))
        flat_state_buffers = getattr(optim, "_matrix_fsdp_flat_adamw_state_buffers")
        self.assertEqual(len(flat_state_buffers), 1)

    def test_configure_optimizer_adamw_returns_plain_prepared_torch_optimizer(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)))

        optim = configure_optimizer(model, "adamw", lr=0.01)

        self.assertIsInstance(optim, torch.optim.AdamW)
        self.assertTrue(hasattr(optim, "matrix_fsdp"))

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optim.step()
        optim.zero_grad()

        self.assertTrue(optim.state)
        self.assertIsNotNone(model._matrix_fsdp_param_group.flat_buffer.local_shard)

    def test_flat_adamw_state_supports_amsgrad(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)))
        optim = MatrixFSDPOptimizer(
            torch.optim.AdamW(model.parameters(), lr=0.01, amsgrad=True),
            model,
            flat_adamw_state=True,
        )

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optim.step()

        flat_state = next(iter(getattr(optim.optimizer, "_matrix_fsdp_flat_adamw_state_buffers").values()))
        self.assertIsNotNone(flat_state.max_exp_avg_sq)
        max_exp_avg_sq_storages = {
            state["max_exp_avg_sq"].untyped_storage().data_ptr()
            for param, state in optim.optimizer.state.items()
            if param.numel() > 0
        }
        self.assertEqual(max_exp_avg_sq_storages, {flat_state.max_exp_avg_sq.untyped_storage().data_ptr()})

    def test_optimizer_wrapper_refreshes_state_object_cache_after_step(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        optim = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optim.step()

        self.assertTrue(optim.state_objects)
        self.assertEqual(optim.state_dtensors, {})
        self.assertEqual(optim.refresh_state_dtensors(), {})

    def test_state_manager_reports_checkpoint_metadata_for_local_state(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        optim = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optim.step()

        metadata = optim.state_manager.local_state_metadata()
        self.assertEqual(set(metadata), {"weight", "bias"})
        weight_exp_avg = metadata["weight"]["exp_avg"]
        self.assertEqual(weight_exp_avg["name"], "exp_avg")
        self.assertEqual(weight_exp_avg["global_shape"], (8,))
        self.assertEqual(weight_exp_avg["global_stride"], (1,))
        self.assertEqual(weight_exp_avg["local_shape"], (8,))
        self.assertEqual(weight_exp_avg["local_numel"], 8)
        self.assertEqual(weight_exp_avg["dtype"], "torch.float32")
        self.assertEqual(weight_exp_avg["matrix_shard"], {"type": "MatrixShard", "dims": (0,), "local_units": (1,)})
        self.assertIsNone(weight_exp_avg["device_mesh"])
        self.assertIsNone(weight_exp_avg["dtensor_spec"])
        self.assertFalse(weight_exp_avg["uses_dtensor"])

    def test_state_manager_metadata_preserves_device_mesh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dist.init_process_group("gloo", init_method=f"file://{tmpdir}/init", rank=0, world_size=1)
            try:
                mesh = DeviceMesh(
                    "cpu",
                    torch.arange(1).reshape(1, 1),
                    mesh_dim_names=("dp_replicate", "dp_shard"),
                )
                torch.manual_seed(0)
                model = matrix_fully_shard(nn.Linear(4, 2), mesh, dp_shard_mesh_dim="dp_shard")
                optim = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

                x = torch.randn(3, 4)
                model(x).sum().backward()
                optim.step()

                metadata = optim.state_manager.local_state_metadata()
                weight_exp_avg = metadata["weight"]["exp_avg"]
                self.assertFalse(weight_exp_avg["uses_dtensor"])
                self.assertIsNone(weight_exp_avg["dtensor_spec"])
                self.assertEqual(weight_exp_avg["device_mesh"]["mesh_dim_names"], ("dp_replicate", "dp_shard"))
                self.assertEqual(weight_exp_avg["device_mesh"]["shard_mesh_dim_name"], "dp_shard")
                self.assertEqual(weight_exp_avg["device_mesh"]["replicate_mesh_dim_name"], "dp_replicate")

                self.assertEqual(optim.state_dtensors, {})
                state_object = optim.state_objects["weight"]["exp_avg"]
                self.assertIsNone(state_object.dtensor)
                self.assertTrue(state_object.has_same_data_ptr(state_object.local_tensor))
            finally:
                dist.destroy_process_group()

    def test_state_manager_ignores_non_param_shaped_tensors_but_validation_rejects_them(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        optim = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optim.step()

        param, state = next(iter(optim.optimizer.state.items()))
        state["mismatched"] = torch.zeros(param.numel() + 1)
        state["scalar_ok"] = torch.tensor(1.0)

        state_objects = optim.local_state_objects()
        self.assertNotIn("mismatched", next(iter(state_objects.values())))
        self.assertNotIn("scalar_ok", next(iter(state_objects.values())))
        with self.assertRaisesRegex(RuntimeError, "mismatched"):
            optim.validate_local_state_shapes()

    def test_state_summary_tracks_only_tensor_state_numel_by_name(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        optim = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optim.step()

        summary = optim.state_manager.summarize()
        self.assertEqual(summary.param_numel, unit.flat_buffer.local_numel)
        self.assertEqual(summary.tensor_state_numel_by_name["exp_avg"], unit.flat_buffer.local_numel)
        self.assertEqual(summary.tensor_state_numel_by_name["exp_avg_sq"], unit.flat_buffer.local_numel)
        self.assertEqual(summary.as_dict(), optim.local_state_summary())

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_mixed_muon_adamw_optimizer_groups_params_from_shard_hints(self):
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2, bias=False)),
            auto_shard_hints=True,
            auto_planner_policy="muon_shard_aware",
        )

        classified = classify_matrix_optimizer_params(model)
        optimizer = MatrixFSDPOptimizer.from_shard_hints(
            model,
            adamw_foreach=False,
            max_cached_elastic_workspaces_per_key=1,
        )
        summary = optimizer.optimizer_group_summary()
        groups_by_type = {group["optimizer_type"]: group for group in summary["groups"]}

        self.assertIsInstance(optimizer.optimizer, MixedMuonAdamWOptimizer)
        self.assertEqual(optimizer.scheduler.max_cached_elastic_workspaces_per_key, 1)
        for param_buffer in (
            model._matrix_fsdp_param_group.flat_buffer.static_param_buffer,
            model._matrix_fsdp_param_group.flat_buffer.elastic_param_buffer,
        ):
            self.assertEqual(param_buffer.workspace.stats()["workspace_max_cached_per_key"], 1)
        self.assertFalse(optimizer.optimizer.adamw.defaults["foreach"])
        self.assertEqual(DEFAULT_MUON_ADJUST_LR_FN, "match_rms_adamw")
        self.assertEqual(optimizer.optimizer.muon.param_groups[0]["adjust_lr_fn"], DEFAULT_MUON_ADJUST_LR_FN)
        self.assertTrue(summary["has_muon"])
        self.assertTrue(summary["has_adamw"])
        self.assertEqual(set(groups_by_type["muon"]["fqns"]), {"0.weight", "2.weight"})
        self.assertEqual(set(groups_by_type["adamw"]["fqns"]), {"0.bias", "1.weight", "1.bias"})
        self.assertEqual(len(classified.muon_params), groups_by_type["muon"]["num_params"])
        self.assertEqual(len(classified.adamw_params), groups_by_type["adamw"]["num_params"])

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()

        state_summary = optimizer.local_state_summary()
        self.assertGreater(state_summary["state_entries"], 0)
        self.assertGreater(state_summary["tensor_state_numel"], 0)
        optimizer.validate_local_state_shapes()

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_configure_optimizer_mixed_muon_adamw_uses_same_lifecycle_and_state_api(self):
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2, bias=False)),
            auto_shard_hints=True,
            auto_planner_policy="muon_shard_aware",
        )

        optimizer = configure_optimizer(
            model,
            "mixed_muon_adamw",
            lr=0.004,
            weight_decay=0.07,
            max_cached_elastic_workspaces_per_key=1,
        )

        self.assertIsInstance(optimizer, MixedMuonAdamWOptimizer)
        self.assertIs(optimizer.matrix_fsdp, optimizer)
        self.assertEqual(optimizer.scheduler.max_cached_elastic_workspaces_per_key, 1)
        for param_buffer in (
            model._matrix_fsdp_param_group.flat_buffer.static_param_buffer,
            model._matrix_fsdp_param_group.flat_buffer.elastic_param_buffer,
        ):
            self.assertEqual(param_buffer.workspace.stats()["workspace_max_cached_per_key"], 1)
        self.assertEqual(optimizer.runtime_param_groups, [model._matrix_fsdp_param_group])
        self.assertEqual(optimizer.muon.defaults["lr"], 0.004)
        self.assertEqual(optimizer.adamw.defaults["lr"], 0.004)
        self.assertEqual(optimizer.muon.defaults["weight_decay"], 0.07)
        self.assertEqual(optimizer.adamw.defaults["weight_decay"], 0.07)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        state = optimizer.state_dict()
        self.assertIn("muon", state)
        self.assertIn("adamw", state)
        self.assertTrue(state["group_summaries"])
        self.assertTrue(optimizer.local_state_summary())
        optimizer.validate_local_state_shapes()

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_configure_optimizer_mixed_muon_adamw_allows_per_path_overrides(self):
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2, bias=False)),
            auto_shard_hints=True,
            auto_planner_policy="muon_shard_aware",
        )

        optimizer = configure_optimizer(
            model,
            "mixed_muon_adamw",
            lr=0.004,
            weight_decay=0.07,
            muon_lr=0.006,
            adamw_lr=0.008,
            muon_weight_decay=0.0,
            adamw_weight_decay=0.01,
        )

        self.assertIsInstance(optimizer, MixedMuonAdamWOptimizer)
        self.assertEqual(optimizer.muon.defaults["lr"], 0.006)
        self.assertEqual(optimizer.adamw.defaults["lr"], 0.008)
        self.assertEqual(optimizer.muon.defaults["weight_decay"], 0.0)
        self.assertEqual(optimizer.adamw.defaults["weight_decay"], 0.01)


if __name__ == "__main__":
    unittest.main()
