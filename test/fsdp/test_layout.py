import unittest

from matrix_fsdp.layout import (
    GlobalMatrixLayout,
    LayoutSegment,
    ParamLayout,
    ParamSegment,
    MatrixGroupLayout,
    ShardPlan,
)


class _ParamStub:
    def __init__(self, fqn: str, offset: int, end: int) -> None:
        self.fqn = fqn
        self.offset = offset
        self.end = end
        self.numel = end - offset


class LayoutTest(unittest.TestCase):
    def test_layout_segment_properties(self):
        segment = LayoutSegment(global_start=3, global_end=8, local_start=11)

        self.assertEqual(segment.numel, 5)
        self.assertEqual(segment.local_end, 16)
        self.assertTrue(segment.intersects(0, 4))
        self.assertTrue(segment.intersects(7, 9))
        self.assertFalse(segment.intersects(8, 10))

    def test_group_layout_projects_to_shard_plan(self):
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=8,
            rank_segments=(
                (LayoutSegment(0, 2, 0), LayoutSegment(6, 8, 2)),
                (LayoutSegment(2, 6, 0),),
            ),
            params=(
                ParamLayout(
                    fqn="weight",
                    global_start=0,
                    global_end=8,
                    segments=(
                        ParamSegment("weight", 0, 0, 2, 0),
                        ParamSegment("weight", 0, 6, 8, 2),
                        ParamSegment("weight", 1, 2, 6, 0),
                    ),
                ),
            ),
        )

        self.assertEqual(layout.world_size, 2)
        self.assertEqual(layout.shard_sizes, (4, 4))
        self.assertEqual(layout.params[0].numel, 8)
        self.assertEqual(layout.param("weight"), layout.params[0])
        self.assertEqual(layout.owner_ranks("weight"), (0, 1))
        self.assertEqual(layout.params_for_rank(0), ("weight",))
        self.assertEqual(
            layout.rank_segments_for_param(0, "weight"),
            (ParamSegment("weight", 0, 0, 2, 0), ParamSegment("weight", 0, 6, 8, 2)),
        )
        with self.assertRaises(KeyError):
            layout.param("missing")
        plan = layout.to_shard_plan()
        self.assertEqual(plan.shard_sizes, (4, 4))
        self.assertEqual(plan.shard_offsets, (0, 4))
        self.assertEqual(plan.local_segments(0), (LayoutSegment(0, 2, 0), LayoutSegment(6, 8, 2)))

    def test_group_layout_projects_param_segments_from_shard_plan(self):
        plan = ShardPlan(
            total_numel=8,
            shard_sizes=(3, 5),
            shard_offsets=(0, 3),
        )
        layout = MatrixGroupLayout.from_shard_plan(
            plan,
            params=(_ParamStub("p0", 0, 4), _ParamStub("p1", 4, 8)),
        )

        self.assertEqual(layout.owner_ranks("p0"), (0, 1))
        self.assertEqual(layout.owner_ranks("p1"), (1,))
        self.assertEqual(
            layout.rank_segments_for_param(0, "p0"),
            (ParamSegment("p0", 0, 0, 3, 0),),
        )
        self.assertEqual(
            layout.rank_segments_for_param(1, "p0"),
            (ParamSegment("p0", 1, 3, 4, 0),),
        )
        self.assertEqual(
            layout.rank_segments_for_param(1, "p1"),
            (ParamSegment("p1", 1, 4, 8, 1),),
        )
        self.assertEqual(layout.params_for_rank(1), ("p0", "p1"))

    def test_global_layout_name_is_backward_compatible_alias(self):
        self.assertIs(GlobalMatrixLayout, MatrixGroupLayout)


if __name__ == "__main__":
    unittest.main()
