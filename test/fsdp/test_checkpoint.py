import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions

from matrix_fsdp import (
    FSDPLifecycleState,
    build_shard_hints,
    configure_optimizer,
    expert_owner_tail_plan,
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    load_matrix_dcp,
    load_matrix_dcp_full_state,
    load_matrix_state_dict,
    patch_model_state_dict,
    patch_optimizer_state_dict,
    MatrixFSDPOptimizer,
    fully_shard,
    matrix_fully_shard,
    matrix_get_state_dict,
    matrix_set_state_dict,
    matrix_state_dict,
    save_matrix_dcp,
    set_model_state_dict,
    set_optimizer_state_dict,
    set_state_dict,
)


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _make_moe_model() -> nn.Module:
    class Expert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = nn.Linear(4, 8, bias=False)
            self.w2 = nn.Linear(8, 4, bias=False)

    class MoEBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = nn.Linear(4, 2, bias=False)
            self.experts = nn.ModuleList([Expert(), Expert()])

    class TinyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.moe = MoEBlock()

    return TinyMoE()


def _load_global_dcp_payload(checkpoint_dir: str) -> dict:
    return torch.load(f"{checkpoint_dir}/matrix_metadata.pt", map_location="cpu")


def _load_dcp_metadata(checkpoint_dir: str, rank: int = 0) -> dict:
    return _load_global_dcp_payload(checkpoint_dir)["ranks"][rank]


def _save_dcp_metadata(checkpoint_dir: str, metadata: dict, rank: int = 0) -> None:
    payload = _load_global_dcp_payload(checkpoint_dir)
    payload["ranks"][rank] = metadata
    torch.save(payload, f"{checkpoint_dir}/matrix_metadata.pt")


def _assert_torch_optimizer_state_dict_close(
    test_case: unittest.TestCase,
    actual: dict,
    expected: dict,
) -> None:
    test_case.assertEqual(actual["param_groups"], expected["param_groups"])
    test_case.assertEqual(set(actual["state"]), set(expected["state"]))
    for param_id, expected_state in expected["state"].items():
        actual_state = actual["state"][param_id]
        test_case.assertEqual(set(actual_state), set(expected_state))
        for name, expected_value in expected_state.items():
            actual_value = actual_state[name]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                test_case.assertEqual(actual_value, expected_value)


def _assert_mixed_optimizer_state_dict_close(
    test_case: unittest.TestCase,
    actual: dict,
    expected: dict,
) -> None:
    test_case.assertEqual(actual["group_summaries"], expected["group_summaries"])
    for component_name in ("muon", "adamw"):
        if expected[component_name] is None:
            test_case.assertIsNone(actual[component_name])
            continue
        test_case.assertIsNotNone(actual[component_name])
        _assert_torch_optimizer_state_dict_close(
            test_case,
            actual[component_name],
            expected[component_name],
        )


def _all_tensor_devices(value) -> list[torch.device]:
    if torch.is_tensor(value):
        return [value.device]
    if isinstance(value, dict):
        devices = []
        for item in value.values():
            devices.extend(_all_tensor_devices(item))
        return devices
    if isinstance(value, (list, tuple)):
        devices = []
        for item in value:
            devices.extend(_all_tensor_devices(item))
        return devices
    return []


class MatrixFSDPCheckpointTest(unittest.TestCase):
    def test_sharded_state_dict_load_restores_local_param_shard(self):
        torch.manual_seed(0)
        model = _make_model()
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(model)
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        self.assertIsNotNone(flat_buffer)

        state = matrix_state_dict(sharded_model)

        self.assertEqual(state["metadata"]["version"], 1)
        self.assertEqual(state["metadata"]["state_dict_type"], "matrix_sharded")
        self.assertEqual(state["metadata"]["num_param_groups"], 1)
        self.assertEqual(state["metadata"]["num_units"], 1)
        self.assertEqual(len(state["param_groups"]), 1)
        self.assertEqual(len(state["units"]), 1)
        self.assertIs(state["param_groups"][0], state["units"][0])
        unit_state = state["units"][0]
        self.assertEqual(unit_state["param_group_index"], 0)
        self.assertEqual(unit_state["module_fqn"], "")
        self.assertEqual(unit_state["param_fqns"], ("0.weight", "0.bias", "2.weight", "2.bias"))
        self.assertEqual(unit_state["matrix_shard"], {"dims": (0,), "local_units": (1,)})
        self.assertEqual(unit_state["param_shard_state"]["name"], "param")
        self.assertEqual(unit_state["param_shard_state"]["local_numel"], flat_buffer.local_numel)
        self.assertEqual(
            unit_state["param_shard_state"]["matrix_shard"],
            {"type": "MatrixShard", "dims": (0,), "local_units": (1,)},
        )
        self.assertIsInstance(unit_state["layout"], dict)

        flat_buffer.local_shard.add_(10.0)
        flat_buffer.use_local_shards()
        load_matrix_state_dict(sharded_model, state)

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        torch.testing.assert_close(flat_buffer.local_shard, unit_state["param_shard"])
        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_dcp_style_state_dict_bridge_roundtrips_model_and_optimizer(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(_make_model())
        optimizer = configure_optimizer(model, "adamw", lr=0.01)
        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()

        model_state, optim_state = matrix_get_state_dict(
            model,
            optimizer,
            options=StateDictOptions(cpu_offload=True),
        )
        self.assertEqual(model_state["metadata"]["state_dict_type"], "matrix_sharded")
        self.assertTrue(_all_tensor_devices(model_state))
        self.assertTrue(all(device.type == "cpu" for device in _all_tensor_devices(model_state)))
        self.assertTrue(all(device.type == "cpu" for device in _all_tensor_devices(optim_state)))

        restored_model = matrix_fully_shard(_make_model())
        restored_optimizer = configure_optimizer(restored_model, "adamw", lr=0.01)
        matrix_set_state_dict(
            restored_model,
            restored_optimizer,
            model_state_dict=model_state,
            optim_state_dict=optim_state,
        )

        torch.testing.assert_close(
            restored_model._matrix_fsdp_param_group.flat_buffer.local_shard,
            model._matrix_fsdp_param_group.flat_buffer.local_shard,
        )
        self.assertEqual(restored_optimizer.state_dict()["param_groups"], optimizer.state_dict()["param_groups"])

    def test_fsdp2_style_state_dict_api_roundtrips_model_and_optimizer(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(_make_model())
        optimizer = configure_optimizer(model, "adamw", lr=0.01)
        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()

        model_state, optim_state = get_state_dict(
            model,
            optimizer,
            options=StateDictOptions(cpu_offload=True),
        )
        model_state_only = get_model_state_dict(model, options=StateDictOptions(cpu_offload=True))
        torch.testing.assert_close(
            model_state["param_groups"][0]["param_shard"],
            model_state_only["param_groups"][0]["param_shard"],
        )
        self.assertEqual(optim_state["param_groups"], get_optimizer_state_dict(model, optimizer)["param_groups"])

        torch.manual_seed(1234)
        restored_model = matrix_fully_shard(_make_model())
        restored_optimizer = configure_optimizer(restored_model, "adamw", lr=0.2)
        set_state_dict(
            restored_model,
            restored_optimizer,
            model_state_dict=model_state,
            optim_state_dict=optim_state,
        )

        torch.testing.assert_close(
            restored_model._matrix_fsdp_param_group.flat_buffer.local_shard,
            model._matrix_fsdp_param_group.flat_buffer.local_shard,
        )
        self.assertEqual(restored_optimizer.state_dict()["param_groups"], optimizer.state_dict()["param_groups"])

        restored_model._matrix_fsdp_param_group.flat_buffer.local_shard.add_(1.0)
        set_model_state_dict(restored_model, model_state)
        torch.testing.assert_close(
            restored_model._matrix_fsdp_param_group.flat_buffer.local_shard,
            model._matrix_fsdp_param_group.flat_buffer.local_shard,
        )

        restored_optimizer.param_groups[0]["lr"] = 0.2
        set_optimizer_state_dict(restored_model, restored_optimizer, optim_state)
        self.assertEqual(restored_optimizer.state_dict()["param_groups"], optimizer.state_dict()["param_groups"])

    def test_plain_torch_optimizer_state_dict_roundtrip_ignores_experimental_collective_env(self):
        torch.manual_seed(0)
        with mock.patch.dict(
            "os.environ",
            {
                "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL": "gin_device",
                "MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL": "reduce",
            },
        ):
            model = fully_shard(_make_model())
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            self.assertTrue(hasattr(optimizer, "matrix_fsdp"))

            x = torch.randn(3, 4)
            model(x).sum().backward()
            optimizer.step()
            model_state, optim_state = get_state_dict(
                model,
                optimizer,
                options=StateDictOptions(cpu_offload=True),
            )

        torch.manual_seed(1234)
        restored_model = fully_shard(_make_model())
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.2)
        set_state_dict(
            restored_model,
            restored_optimizer,
            model_state_dict=model_state,
            optim_state_dict=optim_state,
        )

        torch.testing.assert_close(
            restored_model._matrix_fsdp_param_group.flat_buffer.local_shard,
            model._matrix_fsdp_param_group.flat_buffer.local_shard,
        )
        _assert_torch_optimizer_state_dict_close(
            self,
            restored_optimizer.state_dict(),
            optimizer.state_dict(),
        )

    def test_patch_state_dict_methods_use_matrix_state_dict_api(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(_make_model())
        optimizer = configure_optimizer(model, "adamw", lr=0.01)
        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()

        patch_model_state_dict(model)
        patch_optimizer_state_dict(model, optimizer)
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()
        self.assertEqual(model_state["metadata"]["state_dict_type"], "matrix_sharded")
        self.assertIn("param_groups", optim_state)

        torch.manual_seed(1234)
        restored_model = matrix_fully_shard(_make_model())
        restored_optimizer = configure_optimizer(restored_model, "adamw", lr=0.2)
        patch_model_state_dict(restored_model)
        patch_optimizer_state_dict(restored_model, restored_optimizer)

        incompatible = restored_model.load_state_dict(model_state)
        restored_optimizer.load_state_dict(optim_state)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        torch.testing.assert_close(
            restored_model._matrix_fsdp_param_group.flat_buffer.local_shard,
            model._matrix_fsdp_param_group.flat_buffer.local_shard,
        )
        self.assertEqual(restored_optimizer.state_dict()["param_groups"], optimizer.state_dict()["param_groups"])

    def test_state_dict_load_accepts_param_groups_without_legacy_units(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        self.assertIsNotNone(flat_buffer)
        expected_local_shard = flat_buffer.local_shard.clone()

        state = matrix_state_dict(model)
        canonical_state = copy.deepcopy(state)
        canonical_state.pop("units")

        flat_buffer.local_shard.add_(1.0)
        flat_buffer.use_local_shards()
        load_matrix_state_dict(model, canonical_state)

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        torch.testing.assert_close(flat_buffer.local_shard, state["param_groups"][0]["param_shard"])

    def test_state_dict_metadata_includes_planner_and_runtime_contract(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)),
            auto_planner_policy="balanced",
            target_block_units=4,
            runtime_layout_policy="auto",
        )
        unit = model._matrix_fsdp_param_group

        state = matrix_state_dict(model)
        unit_state = state["units"][0]

        self.assertEqual(unit_state["runtime_layout_policy"], unit.state_dict()["runtime_layout_policy"])
        self.assertEqual(unit_state["runtime_layout_mode"], unit.state_dict()["runtime_layout_mode"])
        self.assertEqual(
            unit_state["runtime_layout_requires_flat_reorder"],
            unit.state_dict()["runtime_layout_requires_flat_reorder"],
        )
        self.assertEqual(unit_state["planner_metadata"], unit.state_dict()["planner_metadata"])
        self.assertEqual(unit_state["planner_summary"], unit.planner_result.summary())
        self.assertEqual(unit_state["planner_layout_contract"], unit.planner_layout_contract.as_metadata())
        self.assertEqual(unit_state["runtime_layout_contract"], unit.runtime_layout_contract.as_metadata())
        self.assertEqual(unit_state["planner_report"], unit.planner_result.report.as_metadata())
        self.assertEqual(unit_state["planner_resource_estimate"], unit.state_dict()["planner_resource_estimate"])
        self.assertEqual(
            unit_state["planner_metadata"]["resource_estimate"],
            unit_state["planner_resource_estimate"],
        )
        self.assertEqual(unit_state["param_gather_strategy"], unit.param_gather_strategy)
        self.assertEqual(unit_state["grad_reduce_strategy"], unit.backward_reduce_strategy)

    def test_state_dict_metadata_preserves_moe_shard_hints_and_expert_owner_report(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            _make_moe_model(),
            shard_hints=build_shard_hints(_make_moe_model()),
            group_planner=expert_owner_tail_plan,
        )

        state = matrix_state_dict(model)
        unit_state = state["param_groups"][0]
        w1_hint = unit_state["params"]["moe.experts.0.w1.weight"]["shard_hint"]

        self.assertEqual(w1_hint["runtime_kind"], "expert_owner")
        self.assertEqual(w1_hint["parallel_role"], "routed_expert")
        self.assertEqual(w1_hint["expert_id"], 0)
        self.assertEqual(w1_hint["expert_group_id"], "moe.experts.0")
        self.assertEqual(
            unit_state["planner_metadata"]["expert_owner_groups"],
            (
                {
                    "expert_group_id": "moe.experts.0",
                    "expert_id": 0,
                    "owner_rank": 0,
                    "owner_ranks": (0,),
                    "param_fqns": ("moe.experts.0.w1.weight", "moe.experts.0.w2.weight"),
                    "numel": 64,
                },
                {
                    "expert_group_id": "moe.experts.1",
                    "expert_id": 1,
                    "owner_rank": 0,
                    "owner_ranks": (0,),
                    "param_fqns": ("moe.experts.1.w1.weight", "moe.experts.1.w2.weight"),
                    "numel": 64,
                },
            ),
        )
        self.assertEqual(unit_state["planner_metadata"]["rank_role_units"]["expert"], (128,))
        self.assertEqual(unit_state["planner_metadata"]["rank_role_units"]["router"], (8,))

    def test_state_dict_load_rejects_layout_metadata_mismatch(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        state = matrix_state_dict(model)

        mesh_mismatch_state = copy.deepcopy(state)
        mesh_mismatch_state["param_groups"][0]["dp_shard_mesh_dim"] = 1
        with self.assertRaisesRegex(ValueError, "dp_shard_mesh_dim"):
            load_matrix_state_dict(model, mesh_mismatch_state)

        layout_mismatch_state = copy.deepcopy(state)
        layout_mismatch_state["param_groups"][0]["layout"]["total_numel"] += 1
        with self.assertRaisesRegex(ValueError, "layout"):
            load_matrix_state_dict(model, layout_mismatch_state)

    def test_full_state_dict_exposes_debug_full_params(self):
        torch.manual_seed(0)
        model = _make_model()
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(model)

        state = matrix_state_dict(sharded_model, full_state=True)

        self.assertEqual(state["metadata"]["state_dict_type"], "matrix_full")
        self.assertEqual(set(state["params"]), set(eager_model.state_dict()))
        for name, eager_tensor in eager_model.state_dict().items():
            torch.testing.assert_close(state["params"][name], eager_tensor)
        self.assertIn("param_shard", state["units"][0])

    def test_full_state_dict_uses_module_prefixes_for_multiple_units(self):
        torch.manual_seed(0)
        model = _make_model()
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(
            model,
            wrap_policy=lambda module: isinstance(module, nn.Linear),
        )

        state = matrix_state_dict(sharded_model, full_state=True)

        self.assertEqual(set(state["params"]), set(eager_model.state_dict()))
        self.assertEqual([unit_state["module_fqn"] for unit_state in state["units"]], ["0", "2"])
        for name, eager_tensor in eager_model.state_dict().items():
            torch.testing.assert_close(state["params"][name], eager_tensor)

    def test_state_dict_can_include_and_restore_local_grad_shard(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        self.assertIsNotNone(flat_buffer)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        self.assertIsNotNone(flat_buffer.local_grad_shard)

        state = matrix_state_dict(model, include_grads=True)
        grad_shard = state["units"][0]["grad_shard"]
        self.assertIsNotNone(grad_shard)
        grad_shard_state = state["units"][0]["grad_shard_state"]
        self.assertEqual(grad_shard_state["name"], "grad")
        self.assertEqual(grad_shard_state["local_numel"], flat_buffer.local_numel)
        self.assertEqual(
            grad_shard_state["matrix_shard"],
            {"type": "MatrixShard", "dims": (0,), "local_units": (1,)},
        )

        flat_buffer.clear_local_grad_shard()
        for param in model.parameters():
            param.grad = None
        load_matrix_state_dict(model, state)

        self.assertIsNotNone(flat_buffer.local_grad_shard)
        torch.testing.assert_close(flat_buffer.local_grad_shard, grad_shard)

    def test_dcp_save_load_restores_local_param_shard(self):
        torch.manual_seed(0)
        model = _make_model()
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(model)
        unit = sharded_model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        self.assertIsNotNone(flat_buffer)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(sharded_model, tmpdir, no_dist=True)
            flat_buffer.local_shard.add_(10.0)
            flat_buffer.use_local_shards()
            load_matrix_dcp(sharded_model, tmpdir, no_dist=True)

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        unit.unshard()
        for eager_param, sharded_param in zip(eager_model.parameters(), sharded_model.parameters()):
            torch.testing.assert_close(eager_param, sharded_param)

    def test_dcp_save_writes_single_global_metadata_file(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(model, tmpdir, no_dist=True)
            checkpoint_dir = Path(tmpdir)
            payload = torch.load(checkpoint_dir / "matrix_metadata.pt", map_location="cpu")

            self.assertEqual(payload["metadata"]["format"], "matrix_dcp_global_metadata")
            self.assertEqual(payload["metadata"]["ranks"], (0,))
            self.assertEqual(set(payload["ranks"]), {0})
            self.assertFalse(list(checkpoint_dir.glob("matrix_metadata_rank_*.pt")))

    def test_dcp_load_accepts_legacy_rank_metadata_sidecar(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        self.assertIsNotNone(flat_buffer)
        expected_local_shard = flat_buffer.local_shard.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(model, tmpdir, no_dist=True)
            checkpoint_dir = Path(tmpdir)
            metadata = _load_dcp_metadata(tmpdir)
            (checkpoint_dir / "matrix_metadata.pt").unlink()
            torch.save(metadata, checkpoint_dir / "matrix_metadata_rank_0.pt")

            flat_buffer.local_shard.add_(10.0)
            flat_buffer.use_local_shards()
            load_matrix_dcp(model, tmpdir, no_dist=True)

        self.assertEqual(unit.lifecycle_state, FSDPLifecycleState.SHARDED)
        torch.testing.assert_close(flat_buffer.local_shard, expected_local_shard)

    def test_dcp_load_rejects_same_layout_metadata_mismatch_before_tensor_load(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(model, tmpdir, no_dist=True)
            metadata = _load_dcp_metadata(tmpdir)
            metadata["units"][0]["local_end"] += 1
            _save_dcp_metadata(tmpdir, metadata)

            with self.assertRaisesRegex(ValueError, "local_end"):
                load_matrix_dcp(model, tmpdir, no_dist=True)

    def test_dcp_save_load_can_include_grad_shard(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        unit = model._matrix_fsdp_param_group
        flat_buffer = unit.flat_buffer
        self.assertIsNotNone(flat_buffer)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        expected_grad = flat_buffer.local_grad_shard.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(model, tmpdir, include_grads=True, no_dist=True)
            flat_buffer.clear_local_grad_shard()
            load_matrix_dcp(model, tmpdir, no_dist=True)

        self.assertIsNotNone(flat_buffer.local_grad_shard)
        torch.testing.assert_close(flat_buffer.local_grad_shard, expected_grad)

    def test_dcp_full_state_can_be_loaded_for_debug(self):
        torch.manual_seed(0)
        model = _make_model()
        eager_model = copy.deepcopy(model)
        sharded_model = matrix_fully_shard(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(sharded_model, tmpdir, full_state=True, no_dist=True)
            full_state = load_matrix_dcp_full_state(tmpdir, no_dist=True)

        self.assertEqual(set(full_state["params"]), set(eager_model.state_dict()))
        for name, eager_tensor in eager_model.state_dict().items():
            torch.testing.assert_close(full_state["params"][name], eager_tensor)

    def test_dcp_save_load_restores_optimizer_state(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        optimizer = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()
        expected_optimizer_state = copy.deepcopy(optimizer.state_dict())

        torch.manual_seed(1234)
        restored_model = matrix_fully_shard(nn.Linear(4, 2))
        restored_optimizer = MatrixFSDPOptimizer(torch.optim.AdamW(restored_model.parameters(), lr=0.2), restored_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(model, tmpdir, optimizer=optimizer, no_dist=True)
            load_matrix_dcp(restored_model, tmpdir, optimizer=restored_optimizer, no_dist=True)

        self.assertEqual(restored_optimizer.state_dict()["param_groups"], expected_optimizer_state["param_groups"])
        for param_id, state in expected_optimizer_state["state"].items():
            restored_state = restored_optimizer.state_dict()["state"][param_id]
            self.assertEqual(set(restored_state), set(state))
            for name, value in state.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(restored_state[name], value)
                else:
                    self.assertEqual(restored_state[name], value)

    def test_dcp_optimizer_load_rejects_missing_or_extra_component_metadata(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        optimizer = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()

        torch.manual_seed(1234)
        restored_model = matrix_fully_shard(nn.Linear(4, 2))
        restored_optimizer = MatrixFSDPOptimizer(torch.optim.AdamW(restored_model.parameters(), lr=0.01), restored_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(model, tmpdir, optimizer=optimizer, no_dist=True)
            metadata = _load_dcp_metadata(tmpdir)
            metadata["optimizer"]["components"] = {}
            _save_dcp_metadata(tmpdir, metadata)

            with self.assertRaisesRegex(ValueError, "missing component"):
                load_matrix_dcp(restored_model, tmpdir, optimizer=restored_optimizer, no_dist=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(model, tmpdir, optimizer=optimizer, no_dist=True)
            metadata = _load_dcp_metadata(tmpdir)
            metadata["optimizer"]["components"]["extra"] = copy.deepcopy(
                metadata["optimizer"]["components"]["optimizer"]
            )
            _save_dcp_metadata(tmpdir, metadata)

            with self.assertRaisesRegex(ValueError, "unknown component"):
                load_matrix_dcp(restored_model, tmpdir, optimizer=restored_optimizer, no_dist=True)

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires torch.optim.Muon")
    def test_dcp_save_load_restores_mixed_muon_adamw_optimizer_state(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2, bias=False)),
            auto_shard_hints=True,
            auto_planner_policy="muon_shard_aware",
        )
        optimizer = MatrixFSDPOptimizer.from_shard_hints(model)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()
        expected_optimizer_state = copy.deepcopy(optimizer.state_dict())

        torch.manual_seed(1234)
        restored_model = matrix_fully_shard(
            nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2, bias=False)),
            auto_shard_hints=True,
            auto_planner_policy="muon_shard_aware",
        )
        restored_optimizer = MatrixFSDPOptimizer.from_shard_hints(restored_model)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(model, tmpdir, optimizer=optimizer, no_dist=True)
            metadata = _load_dcp_metadata(tmpdir)
            load_matrix_dcp(restored_model, tmpdir, optimizer=restored_optimizer, no_dist=True)

        optimizer_metadata = metadata["optimizer"]
        self.assertEqual(optimizer_metadata["kind"], "mixed_muon_adamw")
        self.assertEqual(optimizer_metadata["group_summary"]["optimizer"], "mixed_muon_adamw")
        self.assertEqual(set(optimizer_metadata["components"]), {"muon", "adamw"})
        self.assertIn("0.weight", optimizer_metadata["components"]["muon"]["state"][0]["fqn"])
        _assert_mixed_optimizer_state_dict_close(
            self,
            restored_optimizer.state_dict(),
            expected_optimizer_state,
        )
        restored_optimizer.validate_local_state_shapes()

    def test_dcp_optimizer_metadata_includes_matrix_state_layout(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        optimizer = MatrixFSDPOptimizer(torch.optim.AdamW(model.parameters(), lr=0.01), model)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(model, tmpdir, optimizer=optimizer, no_dist=True)
            metadata = _load_dcp_metadata(tmpdir)

        optimizer_state = metadata["optimizer"]["state"]
        tensor_entries = [
            entry
            for state_metadata in optimizer_state.values()
            for entry in state_metadata["entries"].values()
            if entry["kind"] == "tensor" and tuple(entry["shape"]) != ()
        ]
        self.assertTrue(tensor_entries)
        for entry in tensor_entries:
            self.assertIn("matrix_state", entry)
            matrix_state = entry["matrix_state"]
            entry_numel = 1
            for dim in entry["shape"]:
                entry_numel *= dim
            self.assertEqual(matrix_state["matrix_shard"], {"type": "MatrixShard", "dims": (0,), "local_units": (1,)})
            self.assertEqual(matrix_state["local_numel"], entry_numel)
            self.assertEqual(matrix_state["global_shape"], (entry_numel,))

        scalar_entries = [
            entry
            for state_metadata in optimizer_state.values()
            for entry in state_metadata["entries"].values()
            if entry["kind"] == "tensor" and tuple(entry["shape"]) == ()
        ]
        self.assertTrue(scalar_entries)
        self.assertTrue(all("matrix_state" not in entry for entry in scalar_entries))

    def test_dcp_auto_prepared_optimizer_metadata_includes_matrix_state_layout(self):
        torch.manual_seed(0)
        model = matrix_fully_shard(nn.Linear(4, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        x = torch.randn(3, 4)
        model(x).sum().backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(model, tmpdir, optimizer=optimizer, no_dist=True)
            metadata = _load_dcp_metadata(tmpdir)

        optimizer_state = metadata["optimizer"]["state"]
        tensor_entries = [
            entry
            for state_metadata in optimizer_state.values()
            for entry in state_metadata["entries"].values()
            if entry["kind"] == "tensor" and tuple(entry["shape"]) != ()
        ]
        self.assertTrue(tensor_entries)
        self.assertTrue(all("matrix_state" in entry for entry in tensor_entries))

    def test_dcp_metadata_preserves_moe_shard_hints(self):
        torch.manual_seed(0)
        model = _make_moe_model()
        hints = build_shard_hints(model)
        sharded_model = matrix_fully_shard(model, shard_hints=hints, group_planner=expert_owner_tail_plan)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_matrix_dcp(sharded_model, tmpdir, no_dist=True)
            metadata = _load_dcp_metadata(tmpdir)

            restored_model = _make_moe_model()
            restored_hints = build_shard_hints(restored_model)
            restored_model = matrix_fully_shard(
                restored_model,
                shard_hints=restored_hints,
                group_planner=expert_owner_tail_plan,
            )
            load_matrix_dcp(restored_model, tmpdir, no_dist=True)

        restored_unit = restored_model._matrix_fsdp_param_group
        w2_hint = metadata["param_groups"][0]["params"]["moe.experts.1.w2.weight"]["shard_hint"]

        self.assertEqual(w2_hint["runtime_kind"], "expert_owner")
        self.assertEqual(w2_hint["expert_group_id"], "moe.experts.1")
        self.assertEqual(
            metadata["param_groups"][0]["planner_metadata"]["expert_owner_groups"][1]["expert_group_id"],
            "moe.experts.1",
        )
        self.assertEqual(restored_unit.lifecycle_state, FSDPLifecycleState.SHARDED)


if __name__ == "__main__":
    unittest.main()
