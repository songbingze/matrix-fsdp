import unittest

import torch
from torch import nn

from matrix_fsdp.layout import LayoutSegment, ParamLayout, ParamSegment, MatrixGroupLayout, RankLayout
from matrix_fsdp.layout_validator import (
    explain_runtime_layout_compatibility,
    validate_group_layout,
    validate_runtime_layout,
)
from matrix_fsdp.managed_param import ManagedParamRegistry, ParamRuntimeKind, ParamShardHint


class LayoutValidatorTest(unittest.TestCase):
    def setUp(self):
        self.module = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 1))
        self.registry = ManagedParamRegistry.from_module(self.module)
        self.params = self.registry.params

    def test_accepts_valid_group_layout(self):
        validate_group_layout(self._valid_layout(), self.params, world_size=2)

    def test_runtime_layout_reports_matrix_shard_mode_for_rank_contiguous_layout(self):
        compatibility = explain_runtime_layout_compatibility(self._rank_contiguous_layout(), self.params, world_size=2)

        self.assertTrue(compatibility.compatible)
        self.assertEqual(compatibility.mode, "matrix_shard")
        self.assertFalse(compatibility.requires_flat_reorder)

    def test_runtime_layout_reports_flat_reorder_for_whole_param_owner_layout(self):
        compatibility = explain_runtime_layout_compatibility(self._valid_layout(), self.params, world_size=2)

        self.assertTrue(compatibility.compatible)
        self.assertEqual(compatibility.mode, "flat_reorder")
        self.assertTrue(compatibility.requires_flat_reorder)

    def test_runtime_layout_can_disable_flat_reorder(self):
        compatibility = explain_runtime_layout_compatibility(
            self._valid_layout(),
            self.params,
            world_size=2,
            allow_flat_reorder=False,
        )

        self.assertFalse(compatibility.compatible)
        self.assertEqual(compatibility.mode, "unsupported")
        self.assertTrue(compatibility.requires_flat_reorder)
        with self.assertRaisesRegex(ValueError, "cannot execute planner layout"):
            validate_runtime_layout(self._valid_layout(), self.params, world_size=2, allow_flat_reorder=False)

    def test_runtime_layout_reports_segment_runtime_for_supported_split_layout(self):
        module = self._two_vector_module()
        registry = ManagedParamRegistry.from_module(module)
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=16,
            rank_segments=(
                (LayoutSegment(0, 4, 0), LayoutSegment(8, 12, 4)),
                (LayoutSegment(4, 8, 0), LayoutSegment(12, 16, 4)),
            ),
            params=(
                ParamLayout(
                    fqn="a",
                    global_start=0,
                    global_end=8,
                    segments=(ParamSegment("a", 0, 0, 4, 0), ParamSegment("a", 1, 4, 8, 0)),
                ),
                ParamLayout(
                    fqn="b",
                    global_start=8,
                    global_end=16,
                    segments=(ParamSegment("b", 0, 8, 12, 4), ParamSegment("b", 1, 12, 16, 4)),
                ),
            ),
        )

        compatibility = explain_runtime_layout_compatibility(layout, registry.as_list(), world_size=2)

        self.assertTrue(compatibility.compatible)
        self.assertEqual(compatibility.mode, "segment_runtime")
        self.assertFalse(compatibility.requires_flat_reorder)

        disabled = explain_runtime_layout_compatibility(
            layout,
            registry.as_list(),
            world_size=2,
            allow_segment_runtime=False,
        )
        self.assertFalse(disabled.compatible)
        self.assertEqual(disabled.mode, "unsupported")
        self.assertIn("segment_runtime", disabled.reason)

    def test_runtime_layout_rejects_repeated_local_segments_for_one_param(self):
        module = nn.Linear(16, 1, bias=False)
        registry = ManagedParamRegistry.from_module(module)
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=16,
            rank_segments=(
                (LayoutSegment(0, 4, 0), LayoutSegment(8, 12, 4)),
                (LayoutSegment(4, 8, 0), LayoutSegment(12, 16, 4)),
            ),
            params=(
                ParamLayout(
                    fqn="weight",
                    global_start=0,
                    global_end=16,
                    segments=(
                        ParamSegment("weight", 0, 0, 4, 0),
                        ParamSegment("weight", 1, 4, 8, 0),
                        ParamSegment("weight", 0, 8, 12, 4),
                        ParamSegment("weight", 1, 12, 16, 4),
                    ),
                ),
            ),
        )

        compatibility = explain_runtime_layout_compatibility(layout, registry.as_list(), world_size=2)

        self.assertFalse(compatibility.compatible)
        self.assertEqual(compatibility.mode, "unsupported")
        self.assertIn("multiple local segments", compatibility.reason)
        with self.assertRaisesRegex(ValueError, "cannot execute planner layout"):
            validate_runtime_layout(layout, registry.as_list(), world_size=2)

    def test_rejects_wrong_total_numel(self):
        layout = MatrixGroupLayout(
            total_numel=12,
            ranks=self._valid_layout().ranks,
            params=self._valid_layout().params,
        )

        with self.assertRaisesRegex(ValueError, "total_numel"):
            validate_group_layout(layout, self.params, world_size=2)

    def test_rejects_rank_segment_gap(self):
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=13,
            rank_segments=((LayoutSegment(0, 8, 0),), (LayoutSegment(9, 13, 0),)),
            params=self._valid_layout().params,
        )

        with self.assertRaisesRegex(ValueError, "rank segments gap"):
            validate_group_layout(layout, self.params, world_size=2)

    def test_rejects_param_order_mismatch(self):
        valid = self._valid_layout()
        layout = MatrixGroupLayout(
            total_numel=valid.total_numel,
            ranks=valid.ranks,
            params=(valid.params[1], valid.params[0], *valid.params[2:]),
        )

        with self.assertRaisesRegex(ValueError, "do not match managed params"):
            validate_group_layout(layout, self.params, world_size=2)

    def test_rejects_param_gap(self):
        valid = self._valid_layout()
        bad_param = ParamLayout(
            fqn="0.weight",
            global_start=0,
            global_end=8,
            segments=(ParamSegment("0.weight", 0, 0, 7, 0),),
        )
        layout = MatrixGroupLayout(total_numel=valid.total_numel, ranks=valid.ranks, params=(bad_param, *valid.params[1:]))

        with self.assertRaisesRegex(ValueError, "param 0.weight coverage"):
            validate_group_layout(layout, self.params, world_size=2)

    def test_rejects_param_overlap(self):
        valid = self._valid_layout()
        bad_param = ParamLayout(
            fqn="0.weight",
            global_start=0,
            global_end=8,
            segments=(
                ParamSegment("0.weight", 0, 0, 5, 0),
                ParamSegment("0.weight", 0, 4, 8, 4),
            ),
        )
        layout = MatrixGroupLayout(total_numel=valid.total_numel, ranks=valid.ranks, params=(bad_param, *valid.params[1:]))

        with self.assertRaisesRegex(ValueError, "param 0.weight overlap"):
            validate_group_layout(layout, self.params, world_size=2)

    def test_rejects_invalid_param_rank(self):
        valid = self._valid_layout()
        bad_param = ParamLayout(
            fqn="0.weight",
            global_start=0,
            global_end=8,
            segments=(ParamSegment("0.weight", 2, 0, 8, 0),),
        )
        layout = MatrixGroupLayout(total_numel=valid.total_numel, ranks=valid.ranks, params=(bad_param, *valid.params[1:]))

        with self.assertRaisesRegex(ValueError, "invalid rank"):
            validate_group_layout(layout, self.params, world_size=2)

    def test_rejects_param_segment_that_crosses_boundary(self):
        valid = self._valid_layout()
        bad_param = ParamLayout(
            fqn="0.bias",
            global_start=8,
            global_end=10,
            segments=(ParamSegment("0.bias", 1, 7, 10, 0),),
        )
        layout = MatrixGroupLayout(total_numel=valid.total_numel, ranks=valid.ranks, params=(valid.params[0], bad_param, *valid.params[2:]))

        with self.assertRaisesRegex(ValueError, "crosses parameter boundary"):
            validate_group_layout(layout, self.params, world_size=2)

    def test_rejects_param_segment_with_wrong_local_start(self):
        valid = self._valid_layout()
        bad_param = ParamLayout(
            fqn="0.bias",
            global_start=8,
            global_end=10,
            segments=(ParamSegment("0.bias", 1, 8, 10, 1),),
        )
        layout = MatrixGroupLayout(total_numel=valid.total_numel, ranks=valid.ranks, params=(valid.params[0], bad_param, *valid.params[2:]))

        with self.assertRaisesRegex(ValueError, "does not project onto rank"):
            validate_group_layout(layout, self.params, world_size=2)

    def test_rejects_rank_local_start_gap(self):
        valid = self._valid_layout()
        ranks = (
            valid.ranks[0],
            RankLayout(rank=1, local_units=5, segments=(LayoutSegment(8, 10, 1), LayoutSegment(10, 13, 3))),
        )
        layout = MatrixGroupLayout(total_numel=valid.total_numel, ranks=ranks, params=valid.params)

        with self.assertRaisesRegex(ValueError, "expected contiguous local_start=0"):
            validate_group_layout(layout, self.params, world_size=2)

    def test_rejects_split_matrix_owner_hint(self):
        module = nn.Linear(4, 4, bias=False)
        registry = ManagedParamRegistry.from_module(
            module,
            shard_hints={"weight": ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner")},
        )
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=16,
            rank_segments=((LayoutSegment(0, 8, 0),), (LayoutSegment(8, 16, 0),)),
            params=(
                ParamLayout(
                    fqn="weight",
                    global_start=0,
                    global_end=16,
                    segments=(ParamSegment("weight", 0, 0, 8, 0), ParamSegment("weight", 1, 8, 16, 0)),
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "matrix_owner"):
            validate_group_layout(layout, registry.as_list(), world_size=2)

    def test_rejects_misaligned_row_block_hint(self):
        module = nn.Linear(4, 4, bias=False)
        registry = ManagedParamRegistry.from_module(
            module,
            shard_hints={"weight": ParamShardHint(split_granularity="row_block", block_shape=(2, 4))},
        )
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=16,
            rank_segments=((LayoutSegment(0, 4, 0),), (LayoutSegment(4, 16, 0),)),
            params=(
                ParamLayout(
                    fqn="weight",
                    global_start=0,
                    global_end=16,
                    segments=(ParamSegment("weight", 0, 0, 4, 0), ParamSegment("weight", 1, 4, 16, 0)),
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "not aligned"):
            validate_group_layout(layout, registry.as_list(), world_size=2)

    def test_rejects_expert_owner_group_spread_across_ranks(self):
        module = nn.Sequential(nn.Linear(4, 2, bias=False), nn.Linear(2, 4, bias=False))
        registry = ManagedParamRegistry.from_module(
            module,
            shard_hints={
                "0.weight": ParamShardHint(
                    split_granularity="matrix_owner",
                    runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
                    expert_group_id="experts.0",
                ),
                "1.weight": ParamShardHint(
                    split_granularity="matrix_owner",
                    runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
                    expert_group_id="experts.0",
                ),
            },
        )
        layout = MatrixGroupLayout.from_rank_segments(
            total_numel=16,
            rank_segments=((LayoutSegment(0, 8, 0),), (LayoutSegment(8, 16, 0),)),
            params=(
                ParamLayout(
                    fqn="0.weight",
                    global_start=0,
                    global_end=8,
                    segments=(ParamSegment("0.weight", 0, 0, 8, 0),),
                ),
                ParamLayout(
                    fqn="1.weight",
                    global_start=8,
                    global_end=16,
                    segments=(ParamSegment("1.weight", 1, 8, 16, 0),),
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "Expert group"):
            validate_group_layout(layout, registry.as_list(), world_size=2)

    def _valid_layout(self):
        return MatrixGroupLayout.from_rank_segments(
            total_numel=13,
            rank_segments=(
                (LayoutSegment(0, 8, 0),),
                (LayoutSegment(8, 10, 0), LayoutSegment(10, 13, 2)),
            ),
            params=(
                ParamLayout(
                    fqn="0.weight",
                    global_start=0,
                    global_end=8,
                    segments=(ParamSegment("0.weight", 0, 0, 8, 0),),
                ),
                ParamLayout(
                    fqn="0.bias",
                    global_start=8,
                    global_end=10,
                    segments=(ParamSegment("0.bias", 1, 8, 10, 0),),
                ),
                ParamLayout(
                    fqn="1.weight",
                    global_start=10,
                    global_end=12,
                    segments=(ParamSegment("1.weight", 1, 10, 12, 2),),
                ),
                ParamLayout(
                    fqn="1.bias",
                    global_start=12,
                    global_end=13,
                    segments=(ParamSegment("1.bias", 1, 12, 13, 4),),
                ),
            ),
        )

    def _rank_contiguous_layout(self):
        return MatrixGroupLayout.from_rank_segments(
            total_numel=13,
            rank_segments=(
                (LayoutSegment(0, 8, 0),),
                (LayoutSegment(8, 13, 0),),
            ),
            params=(
                ParamLayout(
                    fqn="0.weight",
                    global_start=0,
                    global_end=8,
                    segments=(ParamSegment("0.weight", 0, 0, 8, 0),),
                ),
                ParamLayout(
                    fqn="0.bias",
                    global_start=8,
                    global_end=10,
                    segments=(ParamSegment("0.bias", 1, 8, 10, 0),),
                ),
                ParamLayout(
                    fqn="1.weight",
                    global_start=10,
                    global_end=12,
                    segments=(ParamSegment("1.weight", 1, 10, 12, 2),),
                ),
                ParamLayout(
                    fqn="1.bias",
                    global_start=12,
                    global_end=13,
                    segments=(ParamSegment("1.bias", 1, 12, 13, 4),),
                ),
            ),
        )

    def _two_vector_module(self):
        module = nn.Module()
        module.register_parameter("a", nn.Parameter(torch.empty(8)))
        module.register_parameter("b", nn.Parameter(torch.empty(8)))
        return module


if __name__ == "__main__":
    unittest.main()
