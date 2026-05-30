import unittest
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate
from torch import nn

from matrix_fsdp.buffer_pool import FullParamBufferPool
from matrix_fsdp.flat_buffer import MatrixFlatBuffer
from matrix_fsdp.grad_bucket import (
    BucketParamGrad,
    MatrixGradBucket,
    _can_use_chunk_cat_fast_path,
    _can_use_flat_cat_fast_path,
    build_reduce_scatter_input,
    classify_copy_in_layout,
    fill_reduce_scatter_input,
)
from matrix_fsdp.layout import LayoutSegment, ShardPlan
from matrix_fsdp.managed_param import ManagedParam
from matrix_fsdp.planner import fsdp2_chunk_plan
from matrix_fsdp.state import MatrixShardedState


def _managed_param(param: nn.Parameter) -> ManagedParam:
    return ManagedParam(
        fqn="weight",
        param=param,
        shape=param.shape,
        dtype=param.dtype,
        device=param.device,
        numel=param.numel(),
        offset=0,
        end=param.numel(),
    )


def _managed_params(*params: tuple[str, nn.Parameter]) -> list[ManagedParam]:
    managed_params = []
    offset = 0
    for fqn, param in params:
        managed_params.append(
            ManagedParam(
                fqn=fqn,
                param=param,
                shape=param.shape,
                dtype=param.dtype,
                device=param.device,
                numel=param.numel(),
                offset=offset,
                end=offset + param.numel(),
            )
        )
        offset += param.numel()
    return managed_params


class FlatBufferTest(unittest.TestCase):
    def test_mesh_sharded_state_exposes_dtensor_wrappers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dist.init_process_group("gloo", init_method=f"file://{tmpdir}/init", rank=0, world_size=1)
            try:
                mesh = DeviceMesh("cpu", torch.arange(1))
                param = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
                plan = ShardPlan(
                    total_numel=6,
                    shard_sizes=(6,),
                    shard_offsets=(0,),
                )

                flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0, mesh=mesh)

                self.assertIsInstance(flat_buffer.param_state, MatrixShardedState)
                self.assertTrue(flat_buffer.param_state.uses_dtensor)
                self.assertTrue(flat_buffer.param_state.has_same_data_ptr(flat_buffer.local_shard))
                self.assertEqual(flat_buffer.param_state.mesh_metadata["shape"], (1,))
                self.assertEqual(flat_buffer.param_state.mesh_metadata["shard_mesh_dim"], 0)
                self.assertEqual(flat_buffer.param_state.mesh_metadata["shard_mesh_size"], 1)
                self.assertIsNotNone(flat_buffer.local_shard_dtensor)
                self.assertIs(flat_buffer.sharded_param, flat_buffer.local_shard_dtensor)
                self.assertIsNone(flat_buffer._local_shard_fallback)
                self.assertEqual(flat_buffer.local_shard_dtensor._spec.placements, (flat_buffer.placement,))
                self.assertEqual(flat_buffer.local_shard_dtensor._spec.shape, torch.Size((6,)))
                self.assertEqual(flat_buffer.local_shard_dtensor.to_local().data_ptr(), flat_buffer.local_shard.data_ptr())

                local_grad_shard = torch.arange(6, dtype=torch.float32)
                flat_buffer.use_local_grad_shard(local_grad_shard)

                self.assertIsInstance(flat_buffer.grad_state, MatrixShardedState)
                self.assertTrue(flat_buffer.grad_state.uses_dtensor)
                self.assertTrue(flat_buffer.grad_state.has_same_data_ptr(local_grad_shard))
                self.assertEqual(flat_buffer.grad_state.mesh_metadata, flat_buffer.param_state.mesh_metadata)
                self.assertIsNotNone(flat_buffer.local_grad_shard_dtensor)
                self.assertIs(flat_buffer.sharded_grad, flat_buffer.local_grad_shard_dtensor)
                self.assertIsNone(flat_buffer._local_grad_shard_fallback)
                self.assertEqual(
                    flat_buffer.local_grad_shard_dtensor.to_local().data_ptr(),
                    local_grad_shard.data_ptr(),
                )
            finally:
                dist.destroy_process_group()

    def test_2d_mesh_sharded_state_uses_replicate_and_matrix_placements(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dist.init_process_group("gloo", init_method=f"file://{tmpdir}/init", rank=0, world_size=1)
            try:
                mesh = DeviceMesh(
                    "cpu",
                    torch.arange(1).reshape(1, 1),
                    mesh_dim_names=("dp_replicate", "dp_shard"),
                )
                param = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
                plan = ShardPlan(
                    total_numel=6,
                    shard_sizes=(6,),
                    shard_offsets=(0,),
                )

                flat_buffer = MatrixFlatBuffer(
                    [_managed_param(param)],
                    plan,
                    rank=0,
                    mesh=mesh,
                    dp_shard_mesh_dim="dp_shard",
                )

                self.assertIsNotNone(flat_buffer.local_shard_dtensor)
                placements = flat_buffer.local_shard_dtensor._spec.placements
                self.assertIsInstance(placements[0], Replicate)
                self.assertEqual(placements[1], flat_buffer.placement)
                self.assertEqual(flat_buffer.param_state.mesh_metadata["mesh_dim_names"], ("dp_replicate", "dp_shard"))
                self.assertEqual(flat_buffer.param_state.mesh_metadata["shard_mesh_dim_name"], "dp_shard")
                self.assertEqual(flat_buffer.param_state.mesh_metadata["replicate_mesh_dim_name"], "dp_replicate")
                self.assertEqual(
                    flat_buffer.local_shard_dtensor.to_local().data_ptr(),
                    flat_buffer.local_shard.data_ptr(),
                )
            finally:
                dist.destroy_process_group()

    def test_rank_ordered_shards_are_backed_by_matrix_shard_placement(self):
        param = nn.Parameter(torch.arange(16, dtype=torch.float32))
        plan = ShardPlan(
            total_numel=16,
            shard_sizes=(4, 8, 4),
            shard_offsets=(0, 4, 12),
        )

        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=1)

        self.assertIsNotNone(flat_buffer.placement)
        self.assertTrue(flat_buffer.placement_compatibility.compatible)
        self.assertEqual(flat_buffer.placement.local_units, (1, 2, 1))
        self.assertEqual(flat_buffer.shard_sizes, (4, 8, 4))
        self.assertEqual((flat_buffer.local_start, flat_buffer.local_end), (4, 12))
        torch.testing.assert_close(flat_buffer.local_shard, torch.arange(4, 12, dtype=torch.float32))

    def test_runtime_rejects_non_matrix_shard_compatible_plan(self):
        param = nn.Parameter(torch.arange(8, dtype=torch.float32))
        plan = ShardPlan(
            total_numel=8,
            shard_sizes=(4, 4),
            shard_offsets=(0, 4),
            rank_segments=(
                (LayoutSegment(0, 2, 0), LayoutSegment(6, 8, 2)),
                (LayoutSegment(2, 6, 0),),
            ),
        )

        with self.assertRaisesRegex(ValueError, "MatrixShard-compatible"):
            MatrixFlatBuffer([_managed_param(param)], plan, rank=0)

    def test_local_shard_views_full_local_param_with_original_shape(self):
        param = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        plan = ShardPlan(
            total_numel=6,
            shard_sizes=(6,),
            shard_offsets=(0,),
        )
        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0)

        flat_buffer.use_local_shards()

        self.assertEqual(param.shape, torch.Size([2, 3]))
        self.assertEqual(param.data.data_ptr(), flat_buffer.local_shard.data_ptr())

    def test_async_all_gather_finish_maps_full_param_views(self):
        param = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        plan = ShardPlan(
            total_numel=6,
            shard_sizes=(6,),
            shard_offsets=(0,),
        )
        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0)
        flat_buffer.use_local_shards()

        handle = flat_buffer.start_all_gather_full_params()

        self.assertEqual(param.data.data_ptr(), flat_buffer.local_shard.data_ptr())
        self.assertTrue(flat_buffer.param_data_alias_local_shard())
        self.assertFalse(flat_buffer.param_data_alias_full_buffer())

        flat_buffer.finish_all_gather_full_params(handle)

        self.assertIsNotNone(flat_buffer.full_buffer)
        self.assertEqual(param.shape, torch.Size([2, 3]))
        self.assertEqual(param.data.data_ptr(), flat_buffer.full_buffer.data_ptr())
        self.assertFalse(flat_buffer.param_data_alias_local_shard())
        self.assertTrue(flat_buffer.param_data_alias_full_buffer())
        torch.testing.assert_close(param.detach(), torch.arange(6, dtype=torch.float32).view(2, 3))

    def test_param_data_views_alias_flat_storage_across_multiple_params(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        bias = nn.Parameter(torch.arange(3, dtype=torch.float32))
        plan = ShardPlan(
            total_numel=9,
            shard_sizes=(9,),
            shard_offsets=(0,),
        )
        flat_buffer = MatrixFlatBuffer(_managed_params(("weight", weight), ("bias", bias)), plan, rank=0)

        flat_buffer.use_local_shards()

        self.assertTrue(flat_buffer.param_data_alias_local_shard())
        self.assertEqual(weight.data_ptr(), flat_buffer.local_shard[:6].view(2, 3).data_ptr())
        self.assertEqual(bias.data_ptr(), flat_buffer.local_shard[6:9].data_ptr())

        flat_buffer.all_gather_full_params()

        self.assertTrue(flat_buffer.param_data_alias_full_buffer())
        self.assertEqual(weight.data_ptr(), flat_buffer.full_buffer[:6].view(2, 3).data_ptr())
        self.assertEqual(bias.data_ptr(), flat_buffer.full_buffer[6:9].data_ptr())

        flat_buffer.use_local_shards()

        self.assertTrue(flat_buffer.param_data_alias_local_shard())
        self.assertEqual(weight.data_ptr(), flat_buffer.local_shard[:6].view(2, 3).data_ptr())
        self.assertEqual(bias.data_ptr(), flat_buffer.local_shard[6:9].data_ptr())

    def test_full_param_buffer_pool_reuses_inactive_full_storage(self):
        param = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        plan = ShardPlan(
            total_numel=6,
            shard_sizes=(6,),
            shard_offsets=(0,),
        )
        pool = FullParamBufferPool(max_cached_per_key=1)
        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0, full_param_buffer_pool=pool)

        flat_buffer.all_gather_full_params()
        first_ptr = flat_buffer.full_buffer.data_ptr()
        self.assertEqual(pool.stats()["allocations"], 1)

        flat_buffer.use_local_shards()
        flat_buffer.clear_full_params()

        self.assertIsNone(flat_buffer.full_buffer)
        self.assertEqual(pool.stats()["cached_buffers"], 1)

        flat_buffer.all_gather_full_params()
        second_ptr = flat_buffer.full_buffer.data_ptr()

        self.assertEqual(second_ptr, first_ptr)
        self.assertTrue(flat_buffer.param_data_alias_full_buffer())
        self.assertEqual(pool.stats()["reuses"], 1)

    def test_shrink_full_params_uses_pool_when_cache_capacity_exists(self):
        param = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        plan = ShardPlan(
            total_numel=6,
            shard_sizes=(6,),
            shard_offsets=(0,),
        )
        pool = FullParamBufferPool(max_cached_per_key=1)
        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0, full_param_buffer_pool=pool)

        flat_buffer.all_gather_full_params()
        first_ptr = flat_buffer.full_buffer.data_ptr()
        flat_buffer.use_local_shards()

        flat_buffer.clear_full_params(shrink_storage=True)

        self.assertIsNone(flat_buffer.full_buffer)
        self.assertEqual(pool.stats()["cached_buffers"], 1)

        flat_buffer.all_gather_full_params()

        self.assertEqual(flat_buffer.full_buffer.data_ptr(), first_ptr)
        self.assertEqual(pool.stats()["reuses"], 1)

    def test_full_grad_buffer_maps_param_grad_views(self):
        param = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        plan = ShardPlan(
            total_numel=6,
            shard_sizes=(6,),
            shard_offsets=(0,),
        )
        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0)
        flat_buffer.all_gather_full_params()

        flat_buffer.prepare_full_grad_buffer()

        self.assertIsNotNone(flat_buffer.full_grad_buffer)
        self.assertTrue(flat_buffer.param_grads_alias_full_grad_buffer())
        self.assertEqual(param.grad.data_ptr(), flat_buffer.full_grad_buffer.data_ptr())
        param.grad.add_(1.0)
        torch.testing.assert_close(flat_buffer.full_grad_buffer, torch.ones(6))

    def test_reduce_dtype_casts_communication_result_back_to_local_shard_dtype(self):
        param = nn.Parameter(torch.arange(6, dtype=torch.bfloat16).view(2, 3))
        plan = ShardPlan(
            total_numel=6,
            shard_sizes=(6,),
            shard_offsets=(0,),
        )
        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0, reduce_dtype=torch.float32)
        flat_buffer.all_gather_full_params()
        flat_buffer.prepare_full_grad_buffer()
        param.grad.copy_(torch.arange(6, dtype=torch.bfloat16).view(2, 3))

        local_grad_shard = flat_buffer.reduce_full_grads_to_local_shard()

        self.assertEqual(local_grad_shard.dtype, torch.bfloat16)
        torch.testing.assert_close(local_grad_shard.float(), torch.arange(6, dtype=torch.float32))

    def test_param_dtype_casts_full_params_without_casting_local_shard(self):
        param = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        plan = ShardPlan(
            total_numel=6,
            shard_sizes=(6,),
            shard_offsets=(0,),
        )
        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0, param_dtype=torch.bfloat16)

        flat_buffer.all_gather_full_params()

        self.assertEqual(flat_buffer.local_shard.dtype, torch.float32)
        self.assertEqual(flat_buffer.full_buffer.dtype, torch.bfloat16)
        self.assertEqual(param.dtype, torch.bfloat16)
        torch.testing.assert_close(param.float(), torch.arange(6, dtype=torch.float32).view(2, 3))

    def test_full_grad_buffer_accumulation_reuses_existing_storage(self):
        param = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        plan = ShardPlan(
            total_numel=6,
            shard_sizes=(6,),
            shard_offsets=(0,),
        )
        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0)
        flat_buffer.all_gather_full_params()
        flat_buffer.prepare_full_grad_buffer()
        flat_buffer.full_grad_buffer.copy_(torch.arange(6, dtype=torch.float32))
        first_ptr = flat_buffer.full_grad_buffer.data_ptr()
        param.grad = None

        reused = flat_buffer.prepare_full_grad_buffer(accumulate=True)

        self.assertTrue(reused)
        self.assertEqual(flat_buffer.full_grad_buffer.data_ptr(), first_ptr)
        self.assertTrue(flat_buffer.param_grads_alias_full_grad_buffer())
        torch.testing.assert_close(flat_buffer.full_grad_buffer, torch.arange(6, dtype=torch.float32))

    def test_grad_bucket_packs_rank_major_reduce_scatter_input(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        bias = nn.Parameter(torch.arange(4, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        plan = ShardPlan(
            total_numel=10,
            shard_sizes=(4, 6),
            shard_offsets=(0, 4),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)
        weight.grad = torch.arange(10, 16, dtype=torch.float32).view(2, 3)
        bias.grad = torch.arange(20, 24, dtype=torch.float32)

        bucket = flat_buffer.collect_grad_bucket()
        packed = build_reduce_scatter_input(bucket, flat_buffer.local_shard)

        self.assertIsNone(weight.grad)
        self.assertIsNone(bias.grad)
        torch.testing.assert_close(
            packed,
            torch.tensor(
                [
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    0.0,
                    0.0,
                    14.0,
                    15.0,
                    20.0,
                    21.0,
                    22.0,
                    23.0,
                ]
            ),
        )

    def test_grad_bucket_uses_flat_cat_fast_path_for_equal_contiguous_layout(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        bias = nn.Parameter(torch.arange(4, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        plan = ShardPlan(
            total_numel=10,
            shard_sizes=(5, 5),
            shard_offsets=(0, 5),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)
        weight.grad = torch.arange(10, 16, dtype=torch.float32).view(2, 3)
        bias.grad = torch.arange(20, 24, dtype=torch.float32)

        bucket = flat_buffer.collect_grad_bucket()
        packed = flat_buffer.local_shard.new_empty(bucket.world_size * bucket.max_shard_size)
        filled = fill_reduce_scatter_input(bucket, packed)

        self.assertTrue(_can_use_flat_cat_fast_path(bucket))
        self.assertEqual(classify_copy_in_layout(bucket), "flat_contiguous")
        self.assertEqual(filled.data_ptr(), packed.data_ptr())
        torch.testing.assert_close(
            filled,
            torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0]),
        )

    def test_grad_bucket_foreach_copy_matches_flat_cat_for_equal_contiguous_layout(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        bias = nn.Parameter(torch.arange(4, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        plan = ShardPlan(
            total_numel=10,
            shard_sizes=(5, 5),
            shard_offsets=(0, 5),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)
        weight.grad = torch.arange(10, 16, dtype=torch.float32).view(2, 3)
        bias.grad = torch.arange(20, 24, dtype=torch.float32)

        bucket = flat_buffer.collect_grad_bucket()
        expected = flat_buffer.local_shard.new_empty(bucket.world_size * bucket.max_shard_size)
        actual = flat_buffer.local_shard.new_empty(bucket.world_size * bucket.max_shard_size)

        fill_reduce_scatter_input(bucket, expected, backend="flat_cat")
        fill_reduce_scatter_input(bucket, actual, backend="foreach_copy")

        torch.testing.assert_close(actual, expected)

    def test_grad_bucket_chunk_cat_matches_segment_copy_for_fsdp2_style_layout(self):
        weight = nn.Parameter(torch.empty(5, dtype=torch.float32))
        bias = nn.Parameter(torch.empty(6, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        weight_grad = torch.arange(5, dtype=torch.float32)
        bias_grad = torch.arange(10, 16, dtype=torch.float32)
        rank_segments = (
            (
                LayoutSegment(0, 3, 0),
                LayoutSegment(5, 8, 3),
            ),
            (
                LayoutSegment(3, 5, 0),
                LayoutSegment(8, 11, 3),
            ),
        )
        bucket = MatrixGradBucket(
            param_grads=(
                BucketParamGrad(managed_params[0], weight_grad),
                BucketParamGrad(managed_params[1], bias_grad),
            ),
            total_numel=11,
            shard_sizes=(6, 6),
            rank_segments=rank_segments,
        )
        expected = torch.empty(12, dtype=torch.float32)
        actual = torch.empty(12, dtype=torch.float32)

        fill_reduce_scatter_input(bucket, expected, backend="segment_copy")
        fill_reduce_scatter_input(bucket, actual, backend="chunk_cat")

        self.assertTrue(_can_use_chunk_cat_fast_path(bucket))
        self.assertEqual(classify_copy_in_layout(bucket), "fsdp2_chunk")
        torch.testing.assert_close(
            actual,
            torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 3.0, 4.0, 0.0, 13.0, 14.0, 15.0]),
        )
        torch.testing.assert_close(actual, expected)

    def test_grad_bucket_classifies_padded_matrix_layout_as_generic_segment(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        bias = nn.Parameter(torch.arange(4, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        plan = ShardPlan(
            total_numel=10,
            shard_sizes=(4, 6),
            shard_offsets=(0, 4),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)
        weight.grad = torch.arange(10, 16, dtype=torch.float32).view(2, 3)
        bias.grad = torch.arange(20, 24, dtype=torch.float32)

        bucket = flat_buffer.collect_grad_bucket()

        self.assertEqual(classify_copy_in_layout(bucket), "generic_segment")

    def test_fsdp2_chunk_plan_local_shard_uses_rank_major_param_chunks(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32))
        bias = nn.Parameter(torch.arange(10, 14, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        layout = fsdp2_chunk_plan(managed_params, world_size=2)
        plan = layout.to_shard_plan()

        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=1)

        self.assertFalse(flat_buffer.placement_compatibility.compatible)
        torch.testing.assert_close(flat_buffer.local_shard, torch.tensor([3.0, 4.0, 5.0, 12.0, 13.0]))
        self.assertEqual(
            flat_buffer.local_segments,
            (LayoutSegment(3, 6, 0), LayoutSegment(8, 10, 3)),
        )

        packed = flat_buffer._pack_full_tensor_by_rank_segments(torch.cat((weight.detach(), bias.detach())))
        torch.testing.assert_close(
            packed,
            torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 3.0, 4.0, 5.0, 12.0, 13.0]),
        )
        unpacked = flat_buffer._unpack_rank_shards_to_full_tensor(packed)
        torch.testing.assert_close(unpacked, torch.cat((weight.detach(), bias.detach())))

    def test_rank_ordered_unpack_reuses_gather_buffer(self):
        param = nn.Parameter(torch.arange(16, dtype=torch.float32))
        plan = ShardPlan(
            total_numel=16,
            shard_sizes=(4, 8, 4),
            shard_offsets=(0, 4, 12),
        )
        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=1)
        gathered = torch.arange(16, dtype=torch.float32)

        unpacked = flat_buffer._unpack_rank_shards_to_full_tensor(gathered)

        self.assertEqual(unpacked.data_ptr(), gathered.data_ptr())
        torch.testing.assert_close(unpacked, gathered)

    def test_segment_runtime_unpack_copies_when_rank_order_differs_from_full_order(self):
        first = nn.Parameter(torch.arange(2, dtype=torch.float32))
        second = nn.Parameter(torch.arange(10, 12, dtype=torch.float32))
        managed_params = _managed_params(("first", first), ("second", second))
        plan = ShardPlan(
            total_numel=4,
            shard_sizes=(2, 2),
            shard_offsets=(0, 2),
            rank_segments=(
                (LayoutSegment(2, 4, 0),),
                (LayoutSegment(0, 2, 0),),
            ),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)
        packed = torch.tensor([20.0, 21.0, 10.0, 11.0])

        unpacked = flat_buffer._unpack_rank_shards_to_full_tensor(packed)

        self.assertNotEqual(unpacked.data_ptr(), packed.data_ptr())
        torch.testing.assert_close(unpacked, torch.tensor([10.0, 11.0, 20.0, 21.0]))

    def test_owner_broadcast_fast_path_accepts_whole_parameter_owner_layout(self):
        first = nn.Parameter(torch.arange(2, dtype=torch.float32))
        second = nn.Parameter(torch.arange(10, 12, dtype=torch.float32))
        managed_params = _managed_params(("first", first), ("second", second))
        plan = ShardPlan(
            total_numel=4,
            shard_sizes=(2, 2),
            shard_offsets=(0, 2),
            rank_segments=(
                (LayoutSegment(0, 2, 0),),
                (LayoutSegment(2, 4, 0),),
            ),
        )

        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0, param_gather_strategy="owner_broadcast")

        self.assertTrue(flat_buffer._can_owner_broadcast_full_params())

    def test_owner_broadcast_fast_path_rejects_split_parameter_layout(self):
        param = nn.Parameter(torch.arange(4, dtype=torch.float32))
        plan = ShardPlan(
            total_numel=4,
            shard_sizes=(2, 2),
            shard_offsets=(0, 2),
            rank_segments=(
                (LayoutSegment(0, 2, 0),),
                (LayoutSegment(2, 4, 0),),
            ),
        )

        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0)

        self.assertFalse(flat_buffer._can_owner_broadcast_full_params())

    def test_owner_layout_grad_bucket_uses_compact_rank_chunks(self):
        first = nn.Parameter(torch.arange(3, dtype=torch.float32))
        second = nn.Parameter(torch.arange(10, 11, dtype=torch.float32))
        managed_params = _managed_params(("first", first), ("second", second))
        plan = ShardPlan(
            total_numel=4,
            shard_sizes=(3, 1),
            shard_offsets=(0, 3),
            rank_segments=(
                (LayoutSegment(0, 3, 0),),
                (LayoutSegment(3, 4, 0),),
            ),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)

        self.assertTrue(flat_buffer.prepare_grad_bucket())

        self.assertTrue(flat_buffer.grad_bucket_input_is_compact)
        self.assertEqual(flat_buffer.grad_bucket_input.numel(), 4)
        self.assertEqual(first.grad.untyped_storage().data_ptr(), flat_buffer.grad_bucket_input.untyped_storage().data_ptr())
        self.assertEqual(second.grad.untyped_storage().data_ptr(), flat_buffer.grad_bucket_input.untyped_storage().data_ptr())

        first.grad.copy_(torch.tensor([1.0, 2.0, 3.0]))
        second.grad.copy_(torch.tensor([4.0]))
        bucket = flat_buffer.collect_grad_bucket()

        self.assertTrue(bucket.packed_input_is_compact)
        torch.testing.assert_close(bucket.packed_input, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_torch_collective_backend_keeps_padded_grad_bucket_for_owner_layout(self):
        first = nn.Parameter(torch.arange(3, dtype=torch.float32))
        second = nn.Parameter(torch.arange(10, 11, dtype=torch.float32))
        managed_params = _managed_params(("first", first), ("second", second))
        plan = ShardPlan(
            total_numel=4,
            shard_sizes=(3, 1),
            shard_offsets=(0, 3),
            rank_segments=(
                (LayoutSegment(0, 3, 0),),
                (LayoutSegment(3, 4, 0),),
            ),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0, matrix_collective_backend="torch")

        self.assertTrue(flat_buffer.prepare_grad_bucket())

        self.assertFalse(flat_buffer.grad_bucket_input_is_compact)
        self.assertEqual(flat_buffer.grad_bucket_input.numel(), 6)

    def test_custom_collective_backend_uses_compact_owner_grad_bucket_fallback(self):
        first = nn.Parameter(torch.arange(3, dtype=torch.float32))
        second = nn.Parameter(torch.arange(10, 11, dtype=torch.float32))
        managed_params = _managed_params(("first", first), ("second", second))
        plan = ShardPlan(
            total_numel=4,
            shard_sizes=(3, 1),
            shard_offsets=(0, 3),
            rank_segments=(
                (LayoutSegment(0, 3, 0),),
                (LayoutSegment(3, 4, 0),),
            ),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0, matrix_collective_backend="custom")

        self.assertTrue(flat_buffer.prepare_grad_bucket())

        self.assertTrue(flat_buffer.grad_bucket_input_is_compact)
        self.assertEqual(flat_buffer.grad_bucket_input.numel(), 4)

    def test_flat_buffer_rejects_unknown_matrix_collective_backend(self):
        param = nn.Parameter(torch.arange(4, dtype=torch.float32))
        plan = ShardPlan(
            total_numel=4,
            shard_sizes=(4,),
            shard_offsets=(0,),
        )

        with self.assertRaisesRegex(ValueError, "matrix_collective_backend"):
            MatrixFlatBuffer([_managed_param(param)], plan, rank=0, matrix_collective_backend="unknown")

    def test_grad_bucket_can_prepare_zero_copy_param_grad_views(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        bias = nn.Parameter(torch.arange(4, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        plan = ShardPlan(
            total_numel=10,
            shard_sizes=(5, 5),
            shard_offsets=(0, 5),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)

        self.assertTrue(flat_buffer.prepare_grad_bucket())
        self.assertIsNotNone(flat_buffer.grad_bucket_input)
        self.assertEqual(weight.grad.untyped_storage().data_ptr(), flat_buffer.grad_bucket_input.untyped_storage().data_ptr())
        self.assertEqual(bias.grad.untyped_storage().data_ptr(), flat_buffer.grad_bucket_input.untyped_storage().data_ptr())

        weight.grad.copy_(torch.arange(10, 16, dtype=torch.float32).view(2, 3))
        bias.grad.copy_(torch.arange(20, 24, dtype=torch.float32))
        bucket = flat_buffer.collect_grad_bucket()

        self.assertIsNone(weight.grad)
        self.assertIsNone(bias.grad)
        self.assertIsNotNone(bucket.packed_input)
        torch.testing.assert_close(
            bucket.packed_input,
            torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0]),
        )

    def test_grad_bucket_accumulation_reuses_packed_views(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        bias = nn.Parameter(torch.arange(4, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        plan = ShardPlan(
            total_numel=10,
            shard_sizes=(5, 5),
            shard_offsets=(0, 5),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)

        self.assertTrue(flat_buffer.prepare_grad_bucket())
        self.assertIsNotNone(flat_buffer.grad_bucket_input)
        packed = flat_buffer.grad_bucket_input
        weight.grad.copy_(torch.arange(10, 16, dtype=torch.float32).view(2, 3))
        bias.grad.copy_(torch.arange(20, 24, dtype=torch.float32))
        expected_after_first = packed.clone()

        weight.grad = None
        bias.grad = None
        self.assertTrue(flat_buffer.prepare_grad_bucket(accumulate=True))
        self.assertEqual(flat_buffer.grad_bucket_input.data_ptr(), packed.data_ptr())
        torch.testing.assert_close(flat_buffer.grad_bucket_input, expected_after_first)
        self.assertEqual(weight.grad.untyped_storage().data_ptr(), packed.untyped_storage().data_ptr())
        self.assertEqual(bias.grad.untyped_storage().data_ptr(), packed.untyped_storage().data_ptr())

        weight.grad.add_(1.0)
        bias.grad.add_(2.0)
        bucket = flat_buffer.collect_grad_bucket()

        self.assertIsNotNone(bucket.packed_input)
        torch.testing.assert_close(
            bucket.packed_input,
            torch.tensor([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 22.0, 23.0, 24.0, 25.0]),
        )

    def test_grad_bucket_zero_copy_declines_noncontiguous_padded_param_layout(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        bias = nn.Parameter(torch.arange(4, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        plan = ShardPlan(
            total_numel=10,
            shard_sizes=(4, 6),
            shard_offsets=(0, 4),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)

        self.assertFalse(flat_buffer.prepare_grad_bucket())
        self.assertIsNone(flat_buffer.grad_bucket_input)
        self.assertIsNone(weight.grad)
        self.assertIsNone(bias.grad)

    def test_grad_bucket_single_rank_reduce_maps_local_grad_shard(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        bias = nn.Parameter(torch.arange(4, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        plan = ShardPlan(
            total_numel=10,
            shard_sizes=(10,),
            shard_offsets=(0,),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)
        weight.grad = torch.arange(10, 16, dtype=torch.float32).view(2, 3)
        bias.grad = torch.arange(20, 24, dtype=torch.float32)

        bucket = flat_buffer.collect_grad_bucket()
        flat_buffer.use_local_shards()
        local_grad_shard = flat_buffer.reduce_grad_bucket_to_local_shard(bucket)
        flat_buffer.use_local_grad_shard(local_grad_shard)

        torch.testing.assert_close(
            local_grad_shard,
            torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0]),
        )
        self.assertIsNotNone(flat_buffer.local_grad_shard)
        self.assertEqual(weight.grad.data_ptr(), flat_buffer.local_grad_shard.data_ptr())

    def test_grad_bucket_reduce_start_stats_reports_copy_in_and_layout(self):
        weight = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        bias = nn.Parameter(torch.arange(4, dtype=torch.float32))
        managed_params = _managed_params(("weight", weight), ("bias", bias))
        plan = ShardPlan(
            total_numel=10,
            shard_sizes=(10,),
            shard_offsets=(0,),
        )
        flat_buffer = MatrixFlatBuffer(managed_params, plan, rank=0)
        weight.grad = torch.arange(10, 16, dtype=torch.float32).view(2, 3)
        bias.grad = torch.arange(20, 24, dtype=torch.float32)

        bucket = flat_buffer.collect_grad_bucket()
        result = flat_buffer.start_reduce_grad_bucket_to_local_shard_with_stats(bucket)

        self.assertEqual(result.stats.layout_kind, "flat_contiguous")
        self.assertTrue(result.stats.needs_copy_in)
        self.assertGreaterEqual(result.stats.copy_in_ms, 0.0)
        self.assertGreaterEqual(result.stats.reduce_scatter_enqueue_ms, 0.0)
        self.assertEqual(result.stats.packed_numel, 10)
        self.assertEqual(result.stats.packed_bytes, 40)
        torch.testing.assert_close(
            result.handle.wait(),
            torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0]),
        )

    def test_local_grad_shard_is_tracked_and_cleared(self):
        param = nn.Parameter(torch.arange(6, dtype=torch.float32).view(2, 3))
        plan = ShardPlan(
            total_numel=6,
            shard_sizes=(6,),
            shard_offsets=(0,),
        )
        flat_buffer = MatrixFlatBuffer([_managed_param(param)], plan, rank=0)
        local_grad_shard = torch.arange(6, dtype=torch.float32)

        flat_buffer.use_local_grad_shard(local_grad_shard)

        self.assertIs(flat_buffer.local_grad_shard, local_grad_shard)
        self.assertEqual(param.grad.data_ptr(), local_grad_shard.data_ptr())

        flat_buffer.clear_local_grad_shard()

        self.assertIsNone(flat_buffer.local_grad_shard)

    def test_direct_all_gather_fast_path_requires_equal_contiguous_rank_segments(self):
        param = nn.Parameter(torch.arange(8, dtype=torch.float32))
        direct_plan = ShardPlan(
            total_numel=8,
            shard_sizes=(4, 4),
            shard_offsets=(0, 4),
        )

        self.assertTrue(MatrixFlatBuffer([_managed_param(param)], direct_plan, rank=0)._can_direct_all_gather_full_params())


if __name__ == "__main__":
    unittest.main()
