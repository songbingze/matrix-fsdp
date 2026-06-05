import unittest

import torch

from matrix_fsdp.layout import LayoutSegment, MatrixGroupLayout, ShardPlan
from matrix_fsdp.placement import (
    MatrixShard,
    explain_matrix_shard_compatibility,
    is_matrix_shard_compatible_plan,
    is_matrix_shard_placement,
    matrix_shard_from_layout,
    matrix_shard_from_plan,
    shard_sizes_to_matrix_local_units,
)


class PlacementTest(unittest.TestCase):
    def test_matrix_shard_is_a_torch_placement_when_available(self):
        from torch.distributed.tensor.placement_types import Placement

        placement = MatrixShard(dims=(0,), local_units=(1, 1))

        self.assertIsInstance(placement, Placement)
        self.assertTrue(placement.is_matrix_shard())
        self.assertTrue(is_matrix_shard_placement(placement))

    def test_matrix_shard_splits_contiguous_tensor_by_local_units(self):
        placement = MatrixShard(dims=(0,), local_units=(1, 2, 1))
        tensor = torch.arange(16)

        shards = placement.split_tensor(tensor)

        self.assertEqual([shard.tolist() for shard in shards], [[0, 1, 2, 3], list(range(4, 12)), [12, 13, 14, 15]])
        self.assertEqual(placement.shard_range(1, tensor.numel()), (4, 12))
        self.assertTrue(torch.equal(placement.local_shard(tensor, 1), tensor[4:12]))

    def test_matrix_shard_reconstructs_prefix_sharded_flat_tensor(self):
        placement = MatrixShard(dims=(0,), local_units=(1, 1))
        flat = torch.arange(12)

        reconstructed = placement.reconstruct_tensor_from_flat(flat, (6, 2))

        self.assertEqual(tuple(reconstructed.shape), (6, 2))
        self.assertTrue(torch.equal(reconstructed, flat.view(6, 2)))

    def test_matrix_shard_rejects_invalid_split(self):
        placement = MatrixShard(dims=(0,), local_units=(1, 2))

        with self.assertRaisesRegex(ValueError, "divisible"):
            placement.split_tensor(torch.arange(5))

    def test_shard_sizes_to_reduced_matrix_local_units(self):
        self.assertEqual(shard_sizes_to_matrix_local_units((4, 8, 4)), (1, 2, 1))
        self.assertEqual(shard_sizes_to_matrix_local_units((0, 6, 3)), (0, 2, 1))
        self.assertEqual(shard_sizes_to_matrix_local_units((3, 5)), (3, 5))

    def test_matrix_shard_from_contiguous_flat_plan(self):
        plan = ShardPlan(
            total_numel=16,
            shard_sizes=(4, 8, 4),
            shard_offsets=(0, 4, 12),
        )

        placement = matrix_shard_from_plan(plan)

        self.assertEqual(placement.dims, (0,))
        self.assertEqual(placement.local_units, (1, 2, 1))

    def test_matrix_shard_from_layout(self):
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=12,
            rank_segments=(
                (LayoutSegment(0, 3, 0),),
                (LayoutSegment(3, 12, 0),),
            ),
        )

        placement = matrix_shard_from_layout(layout)

        self.assertEqual(placement.local_units, (1, 3))

    def test_rejects_non_contiguous_rank_segments(self):
        plan = ShardPlan(
            total_numel=8,
            shard_sizes=(4, 4),
            shard_offsets=(0, 4),
            rank_segments=(
                (LayoutSegment(0, 2, 0), LayoutSegment(6, 8, 2)),
                (LayoutSegment(2, 6, 0),),
            ),
        )

        compatibility = explain_matrix_shard_compatibility(plan)

        self.assertFalse(is_matrix_shard_compatible_plan(plan))
        self.assertFalse(compatibility.compatible)
        self.assertTrue(compatibility.requires_flat_reorder)
        self.assertTrue(compatibility.requires_multi_segment_runtime)
        with self.assertRaisesRegex(ValueError, "contiguous rank-ordered"):
            matrix_shard_from_plan(plan)


if __name__ == "__main__":
    unittest.main()
