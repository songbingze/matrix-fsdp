import unittest

from matrix_fsdp.layout import LayoutSegment
from matrix_fsdp.runtime.static_param_buffer import (
    StaticParamBuffer,
    StaticParamBufferLayout,
    StaticParamBufferWorkspacePlan,
    rank_segments_are_rank_contiguous_chunks,
)


class StaticParamBufferTest(unittest.TestCase):
    def test_layout_reports_padded_rank_chunk_cost(self):
        rank_segments = (
            (LayoutSegment(0, 6, 0),),
            (LayoutSegment(6, 10, 0),),
        )
        layout = StaticParamBufferLayout(total_numel=10, shard_sizes=(6, 4), rank_segments=rank_segments)

        self.assertTrue(layout.rank_chunk_fast_path)
        self.assertTrue(layout.packed_rank_shards_are_full_tensor_order)
        self.assertEqual(layout.padded_numel, 12)
        self.assertEqual(layout.padding_waste_numel, 2)
        self.assertAlmostEqual(layout.padding_waste_ratio, 0.2)
        self.assertEqual(layout.segment_count, 2)
        self.assertEqual(layout.max_segments_per_rank, 1)

    def test_layout_rejects_non_chunk_fast_path(self):
        rank_segments = (
            (
                LayoutSegment(0, 2, 0),
                LayoutSegment(4, 6, 2),
            ),
            (LayoutSegment(2, 4, 0),),
        )
        layout = StaticParamBufferLayout(total_numel=6, shard_sizes=(4, 2), rank_segments=rank_segments)

        self.assertFalse(rank_segments_are_rank_contiguous_chunks(rank_segments))
        self.assertFalse(layout.rank_chunk_fast_path)
        self.assertFalse(layout.packed_rank_shards_are_full_tensor_order)
        self.assertEqual(layout.max_segments_per_rank, 2)

    def test_workspace_plan_prefers_padded_rank_chunks(self):
        buffer = StaticParamBuffer(
            StaticParamBufferLayout(
                total_numel=10,
                shard_sizes=(6, 4),
                rank_segments=(
                    (LayoutSegment(0, 6, 0),),
                    (LayoutSegment(6, 10, 0),),
                ),
            )
        )

        plan = buffer.workspace_plan(can_direct_all_gather=False)

        self.assertIsInstance(plan, StaticParamBufferWorkspacePlan)
        self.assertEqual(plan.preferred_workspace_kind, "padded_rank_chunks")
        self.assertEqual(plan.preferred_workspace_numel, 12)
        self.assertTrue(plan.padded_collective_capable)
        self.assertFalse(plan.equal_collective_capable)

    def test_communication_summary_reports_static_path(self):
        buffer = StaticParamBuffer(
            StaticParamBufferLayout(
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
            matrix_collective_backend="equal",
            can_direct_all_gather=True,
        )

        self.assertEqual(summary["param_buffer_type"], "static")
        self.assertEqual(summary["effective_param_gather_backend"], "equal_all_gather")
        self.assertEqual(summary["effective_grad_reduce_backend"], "equal_reduce_scatter")
        self.assertEqual(summary["workspace_preferred_kind"], "padded_rank_chunks")
        self.assertEqual(summary["workspace_max_cached_per_key"], 1)
        self.assertFalse(summary["owner_segment_collectives"])


if __name__ == "__main__":
    unittest.main()
