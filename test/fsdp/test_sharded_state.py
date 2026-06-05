import tempfile
import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate

from matrix_fsdp import MatrixShard, MatrixShardedState


class MatrixShardedStateTest(unittest.TestCase):
    def test_mesh_state_uses_dtensor_and_aliases_local_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dist.init_process_group("gloo", init_method=f"file://{tmpdir}/init", rank=0, world_size=1)
            try:
                mesh = DeviceMesh("cpu", torch.arange(1))
                placement = MatrixShard(dims=(0,), local_units=(1,))
                local_tensor = torch.arange(6, dtype=torch.float32)

                state = MatrixShardedState(
                    "param",
                    local_tensor,
                    mesh=mesh,
                    placement=placement,
                    global_shape=(6,),
                    global_stride=(1,),
                )

                self.assertTrue(state.uses_dtensor)
                self.assertIsNotNone(state.dtensor)
                self.assertIsNone(state.fallback_tensor)
                self.assertTrue(state.has_same_data_ptr(local_tensor))
                self.assertTrue(state.shares_storage_with(local_tensor))
                self.assertEqual(state.dtensor._spec.placements, (placement,))
                self.assertEqual(state.mesh_metadata["shape"], (1,))
                self.assertEqual(state.mesh_metadata["shard_mesh_dim"], 0)
                self.assertIsNone(state.mesh_metadata["replicate_mesh_dim"])
                metadata = state.as_metadata()
                self.assertEqual(metadata["name"], "param")
                self.assertEqual(metadata["matrix_shard"], {"type": "MatrixShard", "dims": (0,), "local_units": (1,)})
                self.assertEqual(metadata["dtensor_spec"]["shape"], (6,))
                self.assertEqual(metadata["dtensor_spec"]["dtype"], "torch.float32")
                self.assertEqual(
                    metadata["dtensor_spec"]["placements"],
                    ({"type": "MatrixShard", "dims": (0,), "local_units": (1,)},),
                )
            finally:
                dist.destroy_process_group()

    def test_2d_mesh_state_uses_replicate_and_matrix_placements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dist.init_process_group("gloo", init_method=f"file://{tmpdir}/init", rank=0, world_size=1)
            try:
                mesh = DeviceMesh(
                    "cpu",
                    torch.arange(1).reshape(1, 1),
                    mesh_dim_names=("dp_replicate", "dp_shard"),
                )
                placement = MatrixShard(dims=(0,), local_units=(1,))
                local_tensor = torch.arange(6, dtype=torch.float32)

                state = MatrixShardedState(
                    "param",
                    local_tensor,
                    mesh=mesh,
                    placement=placement,
                    global_shape=(6,),
                    shard_mesh_dim="dp_shard",
                )

                self.assertTrue(state.uses_dtensor)
                self.assertIsInstance(state.dtensor._spec.placements[0], Replicate)
                self.assertEqual(state.dtensor._spec.placements[1], placement)
                self.assertEqual(state.mesh_metadata["mesh_dim_names"], ("dp_replicate", "dp_shard"))
                self.assertEqual(state.mesh_metadata["shard_mesh_dim"], 1)
                self.assertEqual(state.mesh_metadata["shard_mesh_dim_name"], "dp_shard")
                self.assertEqual(state.mesh_metadata["replicate_mesh_dim"], 0)
                self.assertEqual(state.mesh_metadata["replicate_mesh_dim_name"], "dp_replicate")
                metadata = state.as_metadata()
                self.assertEqual(metadata["dtensor_spec"]["mesh_shape"], (1, 1))
                self.assertEqual(metadata["dtensor_spec"]["mesh_dim_names"], ("dp_replicate", "dp_shard"))
                self.assertEqual(metadata["dtensor_spec"]["placements"][0]["type"], "Replicate")
                self.assertEqual(
                    metadata["dtensor_spec"]["placements"][1],
                    {"type": "MatrixShard", "dims": (0,), "local_units": (1,)},
                )
            finally:
                dist.destroy_process_group()

    def test_state_without_mesh_keeps_local_tensor_interface(self):
        local_tensor = torch.arange(4, dtype=torch.float32)
        placement = MatrixShard(dims=(0,), local_units=(1,))

        state = MatrixShardedState(
            "param",
            local_tensor,
            mesh=None,
            placement=placement,
            global_shape=(4,),
        )

        self.assertFalse(state.uses_dtensor)
        self.assertIsNone(state.dtensor)
        self.assertIsNone(state.mesh_metadata)
        self.assertIs(state.fallback_tensor, local_tensor)
        self.assertTrue(state.has_same_data_ptr(local_tensor))
        self.assertIsNone(state.as_metadata()["dtensor_spec"])


if __name__ == "__main__":
    unittest.main()
