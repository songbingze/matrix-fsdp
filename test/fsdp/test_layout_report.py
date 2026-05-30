import unittest

from matrix_fsdp.layout import LayoutSegment, ParamLayout, ParamSegment, MatrixGroupLayout
from matrix_fsdp.layout_report import report_group_layout
from matrix_fsdp.planner import ParamBlock


class LayoutReportTest(unittest.TestCase):
    def test_reports_rank_balance_and_split_params(self):
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=12,
            rank_segments=(
                (LayoutSegment(0, 3, 0),),
                (LayoutSegment(3, 9, 0),),
                (LayoutSegment(9, 12, 0),),
            ),
            params=(
                ParamLayout(
                    fqn="p0",
                    global_start=0,
                    global_end=6,
                    segments=(ParamSegment("p0", 0, 0, 3, 0), ParamSegment("p0", 1, 3, 6, 0)),
                ),
                ParamLayout(
                    fqn="p1",
                    global_start=6,
                    global_end=12,
                    segments=(ParamSegment("p1", 1, 6, 9, 3), ParamSegment("p1", 2, 9, 12, 0)),
                ),
            ),
        )

        report = report_group_layout(layout)

        self.assertEqual(report.total_numel, 12)
        self.assertEqual(report.world_size, 3)
        self.assertEqual(report.rank_units, (3, 6, 3))
        self.assertEqual(report.max_rank_units, 6)
        self.assertEqual(report.min_rank_units, 3)
        self.assertEqual(report.avg_rank_units, 4.0)
        self.assertEqual(report.imbalance_units, 3)
        self.assertEqual(report.imbalance_ratio, 1.5)
        self.assertEqual(report.non_empty_ranks, 3)
        self.assertEqual(report.num_params, 2)
        self.assertEqual(report.num_whole_params, 0)
        self.assertEqual(report.whole_param_fqns, ())
        self.assertEqual(report.num_split_params, 2)
        self.assertEqual(report.split_param_fqns, ("p0", "p1"))
        self.assertEqual(report.split_param_units, 12)
        self.assertEqual(report.params_by_rank, (("p0",), ("p0", "p1"), ("p1",)))
        self.assertEqual(report.rank_segment_counts, (1, 1, 1))
        self.assertEqual(report.num_rank_segments, 3)
        self.assertEqual(report.max_rank_segments, 1)
        self.assertEqual(report.fragmented_rank_segments, 0)
        self.assertEqual(report.fragmented_rank_units, 0)
        self.assertEqual(report.param_segment_counts, {"p0": 2, "p1": 2})
        self.assertEqual(report.num_param_segments, 4)
        self.assertEqual(report.split_param_segments, 2)
        self.assertEqual(report.max_param_segments, 2)
        self.assertEqual(report.total_comm_units, 36)
        self.assertEqual(report.max_rank_comm_units, 18)
        self.assertEqual(report.estimated_padding_units, 0)
        self.assertEqual(report.estimated_collectives, 2)
        self.assertEqual(report.as_metadata()["params_by_rank"], report.params_by_rank)
        self.assertEqual(report.as_metadata()["param_segment_counts"], {"p0": 2, "p1": 2})

    def test_estimates_padding_for_rank_units(self):
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=12,
            rank_segments=(
                (LayoutSegment(0, 3, 0),),
                (LayoutSegment(3, 9, 0),),
                (LayoutSegment(9, 12, 0),),
            ),
        )

        report = report_group_layout(layout, padding_alignment=4)

        self.assertEqual(report.estimated_padding_units, 4)

    def test_reports_block_kinds(self):
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=8,
            rank_segments=((LayoutSegment(0, 4, 0),), (LayoutSegment(4, 8, 0),)),
            params=(
                ParamLayout(
                    fqn="w",
                    global_start=0,
                    global_end=8,
                    segments=(ParamSegment("w", 0, 0, 4, 0), ParamSegment("w", 1, 4, 8, 0)),
                ),
            ),
        )
        blocks = (
            ParamBlock("w", 0, 2, 0, "matrix_row_block"),
            ParamBlock("w", 2, 4, 1, "matrix_row_block"),
            ParamBlock("w", 4, 6, 2, "quant_block"),
            ParamBlock("w", 6, 8, 3, "quant_block"),
        )

        report = report_group_layout(layout, blocks)

        self.assertEqual(report.num_blocks, 4)
        self.assertEqual(report.blocks_by_kind, {"matrix_row_block": 2, "quant_block": 2})

    def test_reports_whole_params_and_rank_fragmentation(self):
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=16,
            rank_segments=(
                (LayoutSegment(0, 4, 0), LayoutSegment(8, 12, 4)),
                (LayoutSegment(4, 8, 0), LayoutSegment(12, 16, 4)),
            ),
            params=(
                ParamLayout(
                    fqn="p0",
                    global_start=0,
                    global_end=8,
                    segments=(ParamSegment("p0", 0, 0, 4, 0), ParamSegment("p0", 1, 4, 8, 0)),
                ),
                ParamLayout(
                    fqn="p1",
                    global_start=8,
                    global_end=16,
                    segments=(ParamSegment("p1", 0, 8, 12, 4), ParamSegment("p1", 1, 12, 16, 4)),
                ),
            ),
        )

        report = report_group_layout(layout)

        self.assertEqual(report.rank_segment_counts, (2, 2))
        self.assertEqual(report.fragmented_rank_segments, 2)
        self.assertEqual(report.fragmented_rank_units, 16)
        self.assertEqual(report.num_param_segments, 4)
        self.assertEqual(report.split_param_segments, 2)

    def test_reports_empty_layout(self):
        layout = MatrixGroupLayout.from_rank_segments(total_numel=0, rank_segments=((), ()))

        report = report_group_layout(layout)

        self.assertEqual(report.rank_units, (0, 0))
        self.assertEqual(report.max_rank_units, 0)
        self.assertEqual(report.min_rank_units, 0)
        self.assertEqual(report.imbalance_ratio, 0.0)
        self.assertEqual(report.non_empty_ranks, 0)
        self.assertEqual(report.params_by_rank, ((), ()))


if __name__ == "__main__":
    unittest.main()
