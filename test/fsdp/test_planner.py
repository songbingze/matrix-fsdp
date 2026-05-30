import unittest
from dataclasses import dataclass

from matrix_fsdp.layout import LayoutSegment, ParamSegment
from matrix_fsdp.layout_validator import validate_group_layout
from matrix_fsdp.managed_param import ParamRuntimeKind, ParamShardHint
from matrix_fsdp.placement import matrix_shard_from_layout
from matrix_fsdp.planner import (
    ParamBlock,
    bounded_block_builder,
    bounded_matrix_row_block_builder,
    contiguous_even_plan,
    expert_owner_tail_plan,
    fsdp2_chunk_plan,
    hinted_ordered_group_plan,
    matrix_row_block_builder,
    ordered_group_plan,
    parameter_boundary_plan,
    rotate_layout_ranks,
    uniform_block_builder,
    whole_param_blocks,
)


@dataclass(frozen=True)
class PlannerParamStub:
    fqn: str
    numel: int
    offset: int
    shape: tuple[int, ...] | None = None
    shard_hint: ParamShardHint = ParamShardHint()

    def __post_init__(self):
        if self.shape is None:
            object.__setattr__(self, "shape", (self.numel,))

    @property
    def end(self) -> int:
        return self.offset + self.numel


class PlannerTest(unittest.TestCase):
    def test_contiguous_even_plan(self):
        plan = contiguous_even_plan(total_numel=10, world_size=4)

        self.assertEqual(plan.shard_sizes, (3, 3, 2, 2))
        self.assertEqual(plan.shard_offsets, (0, 3, 6, 8))
        self.assertEqual(plan.local_range(2), (6, 8))
        self.assertEqual(plan.local_segments(2), (LayoutSegment(6, 8, 0),))

    def test_contiguous_even_plan_rejects_invalid_world_size(self):
        with self.assertRaisesRegex(ValueError, "world_size must be positive"):
            contiguous_even_plan(total_numel=10, world_size=0)

    def test_parameter_boundary_plan_assigns_whole_params(self):
        params = (
            PlannerParamStub("p0", numel=5, offset=0),
            PlannerParamStub("p1", numel=4, offset=5),
            PlannerParamStub("p2", numel=3, offset=9),
            PlannerParamStub("p3", numel=2, offset=12),
        )

        layout = parameter_boundary_plan(params, world_size=2)

        self.assertEqual(layout.total_numel, 14)
        self.assertEqual(layout.shard_sizes, (7, 7))
        self.assertEqual(
            layout.rank_segments,
            (
                (LayoutSegment(0, 5, 0), LayoutSegment(12, 14, 5)),
                (LayoutSegment(5, 9, 0), LayoutSegment(9, 12, 4)),
            ),
        )
        self.assertEqual(tuple(param.fqn for param in layout.params), ("p0", "p1", "p2", "p3"))
        self.assertEqual(layout.owner_ranks("p0"), (0,))
        self.assertEqual(layout.owner_ranks("p1"), (1,))
        self.assertEqual(layout.params_for_rank(0), ("p0", "p3"))
        self.assertEqual(
            layout.rank_segments_for_param(0, "p3"),
            (ParamSegment("p3", 0, 12, 14, 5),),
        )
        self.assertEqual(layout.to_shard_plan().local_segments(0), layout.rank_segments[0])

    def test_parameter_boundary_plan_respects_group_order_by_default(self):
        params = (
            PlannerParamStub("small0", numel=1, offset=0),
            PlannerParamStub("large", numel=8, offset=1),
            PlannerParamStub("small1", numel=1, offset=9),
        )

        default_layout = parameter_boundary_plan(params, world_size=2)
        largest_first_layout = parameter_boundary_plan(params, world_size=2, order_policy="largest_first")

        self.assertEqual(default_layout.params_for_rank(0), ("small0", "small1"))
        self.assertEqual(default_layout.params_for_rank(1), ("large",))
        self.assertEqual(largest_first_layout.params_for_rank(0), ("large",))
        self.assertEqual(largest_first_layout.params_for_rank(1), ("small0", "small1"))

    def test_rotate_layout_ranks_preserves_segments_with_shifted_owners(self):
        params = (
            PlannerParamStub("p0", numel=5, offset=0),
            PlannerParamStub("p1", numel=4, offset=5),
            PlannerParamStub("p2", numel=3, offset=9),
        )
        layout = parameter_boundary_plan(params, world_size=3)

        rotated = rotate_layout_ranks(layout, rank_offset=1)

        self.assertEqual(rotated.total_numel, layout.total_numel)
        self.assertEqual(rotated.shard_sizes, (3, 5, 4))
        self.assertEqual(rotated.owner_ranks("p0"), (1,))
        self.assertEqual(rotated.owner_ranks("p1"), (2,))
        self.assertEqual(rotated.owner_ranks("p2"), (0,))
        self.assertEqual(rotated.rank_segments[0], (LayoutSegment(9, 12, 0),))
        self.assertEqual(rotated.rank_segments[1], (LayoutSegment(0, 5, 0),))
        self.assertEqual(rotated.rank_segments[2], (LayoutSegment(5, 9, 0),))
        validate_group_layout(rotated, params, world_size=3)

    def test_parameter_boundary_plan_rejects_unknown_order_policy(self):
        with self.assertRaisesRegex(ValueError, "Unknown order_policy"):
            parameter_boundary_plan((), world_size=1, order_policy="unknown")

    def test_parameter_boundary_plan_rejects_invalid_world_size(self):
        with self.assertRaisesRegex(ValueError, "world_size must be positive"):
            parameter_boundary_plan((), world_size=0)

    def test_expert_owner_tail_plan_keeps_expert_params_together_and_uses_dense_tail(self):
        params = (
            PlannerParamStub("dense.weight", numel=10, offset=0, shape=(2, 5)),
            PlannerParamStub(
                "moe.experts.0.w1.weight",
                numel=8,
                offset=10,
                shape=(2, 4),
                shard_hint=ParamShardHint(
                    optimizer_type="muon",
                    split_granularity="matrix_owner",
                    runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
                    expert_id=0,
                    expert_group_id="moe.experts.0",
                ),
            ),
            PlannerParamStub(
                "moe.experts.0.w2.weight",
                numel=8,
                offset=18,
                shape=(4, 2),
                shard_hint=ParamShardHint(
                    optimizer_type="muon",
                    split_granularity="matrix_owner",
                    runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
                    expert_id=0,
                    expert_group_id="moe.experts.0",
                ),
            ),
            PlannerParamStub(
                "moe.experts.1.w1.weight",
                numel=8,
                offset=26,
                shape=(2, 4),
                shard_hint=ParamShardHint(
                    optimizer_type="muon",
                    split_granularity="matrix_owner",
                    runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
                    expert_id=1,
                    expert_group_id="moe.experts.1",
                ),
            ),
            PlannerParamStub(
                "moe.experts.1.w2.weight",
                numel=8,
                offset=34,
                shape=(4, 2),
                shard_hint=ParamShardHint(
                    optimizer_type="muon",
                    split_granularity="matrix_owner",
                    runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
                    expert_id=1,
                    expert_group_id="moe.experts.1",
                ),
            ),
            PlannerParamStub(
                "moe.router.weight",
                numel=2,
                offset=42,
                shape=(1, 2),
                shard_hint=ParamShardHint(optimizer_type="adamw", split_granularity="parameter"),
            ),
        )

        layout = expert_owner_tail_plan(params, world_size=2)

        self.assertEqual(layout.owner_ranks("moe.experts.0.w1.weight"), (0,))
        self.assertEqual(layout.owner_ranks("moe.experts.0.w2.weight"), (0,))
        self.assertEqual(layout.owner_ranks("moe.experts.1.w1.weight"), (1,))
        self.assertEqual(layout.owner_ranks("moe.experts.1.w2.weight"), (1,))
        self.assertEqual(layout.owner_ranks("dense.weight"), (0,))
        self.assertEqual(layout.owner_ranks("moe.router.weight"), (1,))
        self.assertEqual(layout.shard_sizes, (26, 18))
        validate_group_layout(layout, params, world_size=2)

    def test_fsdp2_chunk_plan_builds_rank_major_param_chunks(self):
        params = (
            PlannerParamStub("p0", numel=6, offset=0),
            PlannerParamStub("p1", numel=4, offset=6),
        )

        layout = fsdp2_chunk_plan(params, world_size=2)

        self.assertEqual(layout.total_numel, 10)
        self.assertEqual(layout.shard_sizes, (5, 5))
        self.assertEqual(
            layout.rank_segments,
            (
                (LayoutSegment(0, 3, 0), LayoutSegment(6, 8, 3)),
                (LayoutSegment(3, 6, 0), LayoutSegment(8, 10, 3)),
            ),
        )
        self.assertEqual(layout.rank_segments_for_param(0, "p0"), (ParamSegment("p0", 0, 0, 3, 0),))
        self.assertEqual(layout.rank_segments_for_param(1, "p1"), (ParamSegment("p1", 1, 8, 10, 3),))

    def test_fsdp2_chunk_plan_rejects_padding_for_now(self):
        params = (PlannerParamStub("p0", numel=5, offset=0),)

        with self.assertRaisesRegex(ValueError, "divisible"):
            fsdp2_chunk_plan(params, world_size=2)

    def test_ordered_group_plan_defaults_to_param_boundary_blocks(self):
        params = (
            PlannerParamStub("p0", numel=5, offset=0),
            PlannerParamStub("p1", numel=4, offset=5),
            PlannerParamStub("p2", numel=3, offset=9),
            PlannerParamStub("p3", numel=2, offset=12),
        )

        layout = ordered_group_plan(params, world_size=2)

        self.assertEqual(layout.shard_sizes, (9, 5))
        self.assertEqual(layout.rank_segments, ((LayoutSegment(0, 9, 0),), (LayoutSegment(9, 14, 0),)))
        self.assertEqual(layout.params_for_rank(0), ("p0", "p1"))
        self.assertEqual(layout.params_for_rank(1), ("p2", "p3"))
        validate_group_layout(layout, params, world_size=2)

    def test_ordered_group_plan_splits_params_only_on_block_boundaries(self):
        params = (
            PlannerParamStub("p0", numel=6, offset=0),
            PlannerParamStub("p1", numel=6, offset=6),
        )

        layout = ordered_group_plan(params, world_size=3, block_size_fn=lambda param: 3)

        self.assertEqual(layout.shard_sizes, (3, 6, 3))
        self.assertEqual(
            layout.rank_segments,
            ((LayoutSegment(0, 3, 0),), (LayoutSegment(3, 9, 0),), (LayoutSegment(9, 12, 0),)),
        )
        self.assertEqual(layout.owner_ranks("p0"), (0, 1))
        self.assertEqual(layout.owner_ranks("p1"), (1, 2))
        self.assertEqual(
            layout.rank_segments_for_param(1, "p0"),
            (ParamSegment("p0", 1, 3, 6, 0),),
        )
        self.assertEqual(
            layout.rank_segments_for_param(1, "p1"),
            (ParamSegment("p1", 1, 6, 9, 3),),
        )
        validate_group_layout(layout, params, world_size=3)

    def test_ordered_group_plan_accepts_explicit_block_builder(self):
        params = (
            PlannerParamStub("p0", numel=8, offset=0),
            PlannerParamStub("p1", numel=4, offset=8),
        )

        def block_builder(param):
            block_size = 4
            return tuple(
                ParamBlock(
                    fqn=param.fqn,
                    global_start=start,
                    global_end=start + block_size,
                    block_index=index,
                    kind="matrix_row_block",
                )
                for index, start in enumerate(range(param.offset, param.end, block_size))
            )

        layout = ordered_group_plan(params, world_size=2, block_builder=block_builder)

        self.assertEqual(layout.shard_sizes, (8, 4))
        self.assertEqual(layout.rank_segments, ((LayoutSegment(0, 8, 0),), (LayoutSegment(8, 12, 0),)))
        self.assertEqual(layout.owner_ranks("p0"), (0,))
        self.assertEqual(layout.owner_ranks("p1"), (1,))
        validate_group_layout(layout, params, world_size=2)

    def test_whole_param_blocks(self):
        param = PlannerParamStub("p0", numel=8, offset=3)

        self.assertEqual(whole_param_blocks(param), (ParamBlock("p0", 3, 11, 0, "parameter"),))

    def test_uniform_block_builder(self):
        param = PlannerParamStub("p0", numel=8, offset=3)
        builder = uniform_block_builder(lambda candidate: 2, kind="quant_block")

        self.assertEqual(
            builder(param),
            (
                ParamBlock("p0", 3, 5, 0, "quant_block"),
                ParamBlock("p0", 5, 7, 1, "quant_block"),
                ParamBlock("p0", 7, 9, 2, "quant_block"),
                ParamBlock("p0", 9, 11, 3, "quant_block"),
            ),
        )

    def test_uniform_block_builder_uses_block_shape_hint(self):
        param = PlannerParamStub(
            "p0",
            numel=8,
            offset=3,
            shard_hint=ParamShardHint(split_granularity="block", block_shape=(4,)),
        )
        builder = uniform_block_builder(lambda candidate: 2, kind="quant_block")

        self.assertEqual(
            builder(param),
            (
                ParamBlock("p0", 3, 7, 0, "quant_block"),
                ParamBlock("p0", 7, 11, 1, "quant_block"),
            ),
        )

    def test_bounded_block_builder_allows_tail_block_without_hint(self):
        param = PlannerParamStub("p0", numel=10, offset=3)
        builder = bounded_block_builder(lambda candidate: 4, kind="ordered_block")

        self.assertEqual(
            builder(param),
            (
                ParamBlock("p0", 3, 7, 0, "ordered_block"),
                ParamBlock("p0", 7, 11, 1, "ordered_block"),
                ParamBlock("p0", 11, 13, 2, "ordered_block"),
            ),
        )

    def test_bounded_block_builder_keeps_explicit_block_hint_strict(self):
        param = PlannerParamStub(
            "p0",
            numel=10,
            offset=3,
            shard_hint=ParamShardHint(split_granularity="block", block_shape=(4,)),
        )

        with self.assertRaisesRegex(ValueError, "not divisible"):
            bounded_block_builder(lambda candidate: 4)(param)

    def test_matrix_row_block_builder(self):
        param = PlannerParamStub("w", numel=24, offset=10, shape=(6, 4))
        builder = matrix_row_block_builder(lambda candidate: 2)

        self.assertEqual(
            builder(param),
            (
                ParamBlock("w", 10, 18, 0, "matrix_row_block"),
                ParamBlock("w", 18, 26, 1, "matrix_row_block"),
                ParamBlock("w", 26, 34, 2, "matrix_row_block"),
            ),
        )

    def test_matrix_row_block_builder_respects_matrix_owner_hint(self):
        param = PlannerParamStub(
            "w",
            numel=24,
            offset=10,
            shape=(6, 4),
            shard_hint=ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
        )

        self.assertEqual(matrix_row_block_builder(lambda candidate: 2)(param), whole_param_blocks(param))

    def test_matrix_row_block_builder_uses_row_block_hint(self):
        param = PlannerParamStub(
            "w",
            numel=24,
            offset=10,
            shape=(6, 4),
            shard_hint=ParamShardHint(split_granularity="row_block", block_shape=(3, 4)),
        )

        self.assertEqual(
            matrix_row_block_builder(lambda candidate: 2)(param),
            (
                ParamBlock("w", 10, 22, 0, "matrix_row_block"),
                ParamBlock("w", 22, 34, 1, "matrix_row_block"),
            ),
        )

    def test_matrix_row_block_builder_rejects_non_matrix_params(self):
        with self.assertRaisesRegex(ValueError, "must be 2D"):
            matrix_row_block_builder(lambda param: 2)(PlannerParamStub("bias", numel=8, offset=0, shape=(8,)))

    def test_matrix_row_block_builder_rejects_invalid_row_block(self):
        param = PlannerParamStub("w", numel=24, offset=0, shape=(6, 4))
        with self.assertRaisesRegex(ValueError, "not divisible"):
            matrix_row_block_builder(lambda candidate: 4)(param)
        with self.assertRaisesRegex(ValueError, "row_block_fn returned"):
            matrix_row_block_builder(lambda candidate: 0)(param)

    def test_bounded_matrix_row_block_builder_allows_tail_rows_without_hint(self):
        param = PlannerParamStub("w", numel=20, offset=10, shape=(5, 4))
        builder = bounded_matrix_row_block_builder(lambda candidate: 2)

        self.assertEqual(
            builder(param),
            (
                ParamBlock("w", 10, 18, 0, "matrix_row_block"),
                ParamBlock("w", 18, 26, 1, "matrix_row_block"),
                ParamBlock("w", 26, 30, 2, "matrix_row_block"),
            ),
        )

    def test_ordered_group_plan_with_matrix_row_blocks(self):
        params = (
            PlannerParamStub("w0", numel=24, offset=0, shape=(6, 4)),
            PlannerParamStub("w1", numel=8, offset=24, shape=(2, 4)),
        )

        layout = ordered_group_plan(params, world_size=2, block_builder=matrix_row_block_builder(lambda param: 2))

        self.assertEqual(layout.shard_sizes, (16, 16))
        self.assertEqual(layout.rank_segments, ((LayoutSegment(0, 16, 0),), (LayoutSegment(16, 32, 0),)))
        self.assertEqual(layout.owner_ranks("w0"), (0, 1))
        self.assertEqual(layout.owner_ranks("w1"), (1,))
        validate_group_layout(layout, params, world_size=2)

    def test_ordered_group_plan_rejects_both_block_apis(self):
        params = (PlannerParamStub("p0", numel=4, offset=0),)

        with self.assertRaisesRegex(ValueError, "Pass only one"):
            ordered_group_plan(
                params,
                world_size=2,
                block_builder=lambda param: (ParamBlock(param.fqn, param.offset, param.end, 0),),
                block_size_fn=lambda param: param.numel,
            )

    def test_ordered_group_plan_rejects_invalid_block_size(self):
        params = (PlannerParamStub("p0", numel=5, offset=0),)

        with self.assertRaisesRegex(ValueError, "not divisible"):
            ordered_group_plan(params, world_size=2, block_size_fn=lambda param: 2)
        with self.assertRaisesRegex(ValueError, "block_size_fn returned"):
            ordered_group_plan(params, world_size=2, block_size_fn=lambda param: 0)

    def test_ordered_group_plan_rejects_bad_block_builder_output(self):
        params = (PlannerParamStub("p0", numel=8, offset=0),)

        with self.assertRaisesRegex(ValueError, "gap"):
            ordered_group_plan(
                params,
                world_size=2,
                block_builder=lambda param: (
                    ParamBlock(param.fqn, 0, 4, 0),
                    ParamBlock(param.fqn, 5, 8, 1),
                ),
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            ordered_group_plan(
                params,
                world_size=2,
                block_builder=lambda param: (
                    ParamBlock(param.fqn, 0, 5, 0),
                    ParamBlock(param.fqn, 4, 8, 1),
                ),
            )
        with self.assertRaisesRegex(ValueError, "crosses"):
            ordered_group_plan(
                params,
                world_size=2,
                block_builder=lambda param: (ParamBlock(param.fqn, 0, 9, 0),),
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ordered_group_plan(
                params,
                world_size=2,
                block_builder=lambda param: (
                    ParamBlock(param.fqn, 0, 4, 0),
                    ParamBlock(param.fqn, 4, 8, 0),
                ),
            )

    def test_ordered_group_plan_rejects_invalid_world_size(self):
        with self.assertRaisesRegex(ValueError, "world_size must be positive"):
            ordered_group_plan((), world_size=0)

    def test_hinted_ordered_group_plan_mixes_expected_granularities(self):
        params = (
            PlannerParamStub(
                "muon",
                numel=24,
                offset=0,
                shape=(6, 4),
                shard_hint=ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner"),
            ),
            PlannerParamStub(
                "rows",
                numel=24,
                offset=24,
                shape=(6, 4),
                shard_hint=ParamShardHint(split_granularity="row_block", block_shape=(2, 4)),
            ),
            PlannerParamStub(
                "blocks",
                numel=8,
                offset=48,
                shard_hint=ParamShardHint(split_granularity="block", block_shape=(4,)),
            ),
            PlannerParamStub("bias", numel=3, offset=56),
        )

        layout = hinted_ordered_group_plan(params, world_size=3, default_granularity="parameter")

        self.assertEqual(layout.shard_sizes, (24, 16, 19))
        self.assertEqual(
            layout.rank_segments,
            (
                (LayoutSegment(0, 24, 0),),
                (LayoutSegment(24, 40, 0),),
                (LayoutSegment(40, 59, 0),),
            ),
        )
        self.assertEqual(layout.owner_ranks("muon"), (0,))
        self.assertEqual(layout.owner_ranks("rows"), (1, 2))
        self.assertEqual(layout.owner_ranks("blocks"), (2,))
        self.assertEqual(layout.owner_ranks("bias"), (2,))
        self.assertEqual(matrix_shard_from_layout(layout).local_units, (24, 16, 19))
        validate_group_layout(layout, params, world_size=3)

    def test_hinted_ordered_group_plan_can_default_to_row_blocks(self):
        params = (
            PlannerParamStub("w", numel=20, offset=0, shape=(5, 4)),
            PlannerParamStub("bias", numel=5, offset=20, shape=(5,)),
        )

        layout = hinted_ordered_group_plan(
            params,
            world_size=2,
            default_granularity="row_block",
            row_block_units=2,
        )

        self.assertEqual(layout.shard_sizes, (16, 9))
        self.assertEqual(layout.owner_ranks("w"), (0, 1))
        self.assertEqual(layout.owner_ranks("bias"), (1,))
        self.assertEqual(matrix_shard_from_layout(layout).local_units, (16, 9))
        validate_group_layout(layout, params, world_size=2)

    def test_hinted_ordered_group_plan_rejects_unknown_default_granularity(self):
        params = (PlannerParamStub("p0", numel=4, offset=0),)

        with self.assertRaisesRegex(ValueError, "Unknown default_granularity"):
            hinted_ordered_group_plan(params, world_size=2, default_granularity="unknown")


if __name__ == "__main__":
    unittest.main()
