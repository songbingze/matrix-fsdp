import unittest

import torch

from matrix_fsdp.layout import LayoutSegment
from matrix_fsdp.runtime.elastic_param_buffer import (
    ElasticParamBuffer,
    ElasticParamBufferLayout,
    ElasticParamBufferWorkspace,
    ElasticParamBufferWorkspacePlan,
    rank_segments_are_rank_contiguous_chunks,
)


class ElasticParamBufferTest(unittest.TestCase):
    def test_layout_reports_rank_chunk_padding_and_imbalance(self):
        rank_segments = (
            (LayoutSegment(0, 6, 0),),
            (LayoutSegment(6, 10, 0),),
        )
        layout = ElasticParamBufferLayout(total_numel=10, shard_sizes=(6, 4), rank_segments=rank_segments)

        self.assertTrue(layout.rank_chunk_fast_path)
        self.assertTrue(layout.packed_rank_shards_are_full_tensor_order)
        self.assertEqual(layout.segment_count, 2)
        self.assertEqual(layout.max_segments_per_rank, 1)
        self.assertEqual(layout.padding_waste_numel, 2)
        self.assertAlmostEqual(layout.padding_waste_ratio, 0.2)
        self.assertAlmostEqual(layout.owner_imbalance_ratio, 1.2)

    def test_layout_rejects_multi_segment_chunk_fast_path(self):
        rank_segments = (
            (
                LayoutSegment(0, 2, 0),
                LayoutSegment(4, 6, 2),
            ),
            (LayoutSegment(2, 4, 0),),
        )

        self.assertFalse(rank_segments_are_rank_contiguous_chunks(rank_segments))
        layout = ElasticParamBufferLayout(total_numel=6, shard_sizes=(4, 2), rank_segments=rank_segments)

        self.assertFalse(layout.rank_chunk_fast_path)
        self.assertFalse(layout.packed_rank_shards_are_full_tensor_order)
        self.assertEqual(layout.segment_count, 3)
        self.assertEqual(layout.max_segments_per_rank, 2)

    def test_communication_summary_resolves_custom_backend(self):
        rank_segments = (
            (LayoutSegment(0, 6, 0),),
            (LayoutSegment(6, 10, 0),),
        )
        buffer = ElasticParamBuffer(
            ElasticParamBufferLayout(total_numel=10, shard_sizes=(6, 4), rank_segments=rank_segments)
        )

        summary = buffer.communication_summary(
            param_gather_strategy="auto",
            matrix_collective_backend="custom",
            can_direct_all_gather=False,
            owner_segment_backend="custom",
            custom_allgather_resolver=lambda segments: ("auto", "native_group_broadcast"),
        )

        self.assertEqual(summary["effective_param_gather_backend"], "owner_segment:custom")
        self.assertEqual(summary["custom_allgatherv_policy"], "auto")
        self.assertEqual(summary["resolved_custom_allgatherv_impl"], "native_group_broadcast")
        self.assertTrue(summary["rank_chunk_fast_path"])
        self.assertEqual(summary["workspace_preferred_kind"], "matrix_all_gather")
        self.assertEqual(summary["workspace_padded_rank_chunks_numel"], 12)
        self.assertEqual(summary["workspace_compact_rank_chunks_numel"], 10)
        self.assertAlmostEqual(summary["workspace_padding_waste_ratio"], 0.2)

    def test_workspace_plan_prefers_native_group_broadcast_when_available(self):
        buffer = ElasticParamBuffer(
            ElasticParamBufferLayout(
                total_numel=10,
                shard_sizes=(6, 4),
                rank_segments=(
                    (LayoutSegment(0, 6, 0),),
                    (LayoutSegment(6, 10, 0),),
                ),
            )
        )

        plan = buffer.workspace_plan(
            can_direct_all_gather=False,
            owner_segment_backend="custom",
            native_kernel_available=True,
        )

        self.assertIsInstance(plan, ElasticParamBufferWorkspacePlan)
        self.assertEqual(plan.preferred_workspace_kind, "owner_segment")
        self.assertEqual(plan.preferred_workspace_numel, 10)
        self.assertTrue(plan.native_group_broadcast_capable)
        self.assertTrue(plan.native_sendrecv_chunk_capable)
        self.assertTrue(plan.compact_owner_reduce_scatter_capable)

    def test_effective_backend_prefers_equal_all_gather_when_available(self):
        buffer = ElasticParamBuffer(
            ElasticParamBufferLayout(
                total_numel=8,
                shard_sizes=(4, 4),
                rank_segments=(
                    (LayoutSegment(0, 4, 0),),
                    (LayoutSegment(4, 8, 0),),
                ),
            )
        )

        summary = buffer.communication_summary(
            param_gather_strategy="auto",
            matrix_collective_backend="owner_broadcast",
            can_direct_all_gather=True,
            owner_segment_backend="owner_broadcast",
        )

        self.assertEqual(summary["effective_param_gather_backend"], "equal_all_gather")

    def test_workspace_reuses_released_tensor_by_shape_dtype_and_device(self):
        workspace = ElasticParamBufferWorkspace()
        reference = torch.empty((), dtype=torch.float32)

        first = workspace.acquire(reference, 8)
        first_ptr = first.tensor.data_ptr()
        self.assertEqual(workspace.stats()["workspace_allocate_count"], 1)
        self.assertEqual(workspace.stats()["workspace_in_use_tensors"], 1)

        first.release()
        second = workspace.acquire(reference, 8)

        self.assertEqual(second.tensor.data_ptr(), first_ptr)
        self.assertEqual(workspace.stats()["workspace_acquire_count"], 2)
        self.assertEqual(workspace.stats()["workspace_reuse_count"], 1)
        self.assertEqual(workspace.stats()["workspace_allocate_count"], 1)
        second.release()

    def test_workspace_can_disable_released_tensor_cache(self):
        workspace = ElasticParamBufferWorkspace(max_cached_per_key=0)
        reference = torch.empty((), dtype=torch.float32)

        first = workspace.acquire(reference, 8)
        first.release()
        second = workspace.acquire(reference, 8)

        self.assertEqual(workspace.stats()["workspace_max_cached_per_key"], 0)
        self.assertEqual(workspace.stats()["workspace_reuse_count"], 0)
        self.assertEqual(workspace.stats()["workspace_allocate_count"], 2)
        second.release()
        self.assertEqual(workspace.stats()["workspace_allocated_tensors"], 0)

    def test_workspace_does_not_reuse_in_use_tensor(self):
        workspace = ElasticParamBufferWorkspace()
        reference = torch.empty((), dtype=torch.float32)

        first = workspace.acquire(reference, 8)
        second = workspace.acquire(reference, 8)

        self.assertNotEqual(second.tensor.data_ptr(), first.tensor.data_ptr())
        self.assertEqual(workspace.stats()["workspace_allocate_count"], 2)

        first.release()
        second.release()


if __name__ == "__main__":
    unittest.main()
