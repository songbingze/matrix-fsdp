import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from matrix_fsdp.collectives import (
    all_gatherv_rank_segments_1d_into_async,
    reduce_scatterv_owner_rank_chunks_1d_async,
)
from matrix_fsdp.runtime.collectives import (
    _make_uneven_all_gather_output_list,
    _make_uneven_reduce_scatter_input_list,
)
from matrix_fsdp.layout import LayoutSegment


def _run_two_rank_allgatherv_backend(rank: int, world_size: int, init_file: str, backend: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        rank_segments = (
            (LayoutSegment(0, 3, 0),),
            (LayoutSegment(3, 5, 0),),
        )
        local = torch.arange(0, 3, dtype=torch.float32) if rank == 0 else torch.arange(3, 5, dtype=torch.float32)
        output = torch.empty(5, dtype=torch.float32)

        handle = all_gatherv_rank_segments_1d_into_async(
            local,
            output,
            rank_segments,
            rank,
            backend=backend,
        )

        torch.testing.assert_close(handle.wait(), torch.arange(5, dtype=torch.float32))
    finally:
        dist.destroy_process_group()


def _run_two_rank_fused_allgatherv_backend(
    rank: int,
    world_size: int,
    init_file: str,
    backend: str,
    custom_impl: str | None = None,
) -> None:
    if custom_impl is not None:
        os.environ["MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL"] = custom_impl
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        rank_segments = (
            (
                LayoutSegment(0, 2, 0),
                LayoutSegment(2, 3, 2),
            ),
            (
                LayoutSegment(3, 4, 0),
                LayoutSegment(4, 6, 1),
            ),
        )
        local = torch.arange(0, 3, dtype=torch.float32) if rank == 0 else torch.arange(3, 6, dtype=torch.float32)
        output = torch.empty(6, dtype=torch.float32)

        handle = all_gatherv_rank_segments_1d_into_async(
            local,
            output,
            rank_segments,
            rank,
            backend=backend,
        )

        torch.testing.assert_close(handle.wait(), torch.arange(6, dtype=torch.float32))
    finally:
        if custom_impl is not None:
            os.environ.pop("MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL", None)
        dist.destroy_process_group()


def _run_two_rank_reduce_scatterv_backend(rank: int, world_size: int, init_file: str, backend: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        local_full_grad = torch.arange(5, dtype=torch.float32) + rank * 10.0

        handle = reduce_scatterv_owner_rank_chunks_1d_async(
            local_full_grad,
            (3, 2),
            rank,
            backend=backend,
            divide_by_world=True,
            compact=True,
        )

        expected = torch.tensor([5.0, 6.0, 7.0]) if rank == 0 else torch.tensor([8.0, 9.0])
        torch.testing.assert_close(handle.wait(), expected)
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available(), "torch.distributed is not available")
class MatrixCollectiveBackendTest(unittest.TestCase):
    def test_uneven_all_gather_uses_full_buffer_views_for_contiguous_rank_segments(self):
        local = torch.empty(3, dtype=torch.float32)
        output = torch.empty(5, dtype=torch.float32)
        output_list, staged_outputs = _make_uneven_all_gather_output_list(
            local,
            output,
            (
                (LayoutSegment(0, 3, 0),),
                (LayoutSegment(3, 5, 0),),
            ),
            (3, 2),
        )

        self.assertEqual(staged_outputs, [])
        self.assertEqual(output_list[0].data_ptr(), output[:3].data_ptr())
        self.assertEqual(output_list[1].data_ptr(), output[3:5].data_ptr())

    def test_uneven_all_gather_stages_fragmented_rank_segments(self):
        local = torch.empty(3, dtype=torch.float32)
        output = torch.empty(6, dtype=torch.float32)
        output_list, staged_outputs = _make_uneven_all_gather_output_list(
            local,
            output,
            (
                (LayoutSegment(0, 2, 0), LayoutSegment(4, 5, 2)),
                (LayoutSegment(2, 4, 0), LayoutSegment(5, 6, 2)),
            ),
            (3, 3),
        )

        self.assertEqual(len(staged_outputs), 2)
        self.assertNotEqual(output_list[0].data_ptr(), output.data_ptr())

    def test_uneven_reduce_scatter_uses_compact_rank_chunk_views(self):
        packed = torch.arange(5, dtype=torch.float32)
        input_list = _make_uneven_reduce_scatter_input_list(packed, (3, 2))

        self.assertEqual(input_list[0].data_ptr(), packed.data_ptr())
        self.assertEqual(input_list[1].data_ptr(), packed[3:].data_ptr())
        torch.testing.assert_close(input_list[0], torch.tensor([0.0, 1.0, 2.0]))
        torch.testing.assert_close(input_list[1], torch.tensor([3.0, 4.0]))

    def test_two_rank_cpu_owner_broadcast_allgatherv_semantics(self):
        self._run_two_rank_allgatherv_backend("owner_broadcast")

    def test_two_rank_cpu_custom_allgatherv_fallback_semantics(self):
        self._run_two_rank_allgatherv_backend("custom")

    def test_two_rank_cpu_owner_broadcast_fused_allgatherv_semantics(self):
        self._run_two_rank_fused_allgatherv_backend("owner_broadcast")

    def test_two_rank_cpu_custom_fused_allgatherv_semantics(self):
        self._run_two_rank_fused_allgatherv_backend("custom")

    def test_two_rank_cpu_custom_all_reduce_allgatherv_semantics(self):
        self._run_two_rank_fused_allgatherv_backend("custom", custom_impl="all_reduce")

    def test_two_rank_cpu_custom_uneven_all_gather_allgatherv_semantics(self):
        self._run_two_rank_fused_allgatherv_backend("custom", custom_impl="uneven_all_gather")

    def test_two_rank_cpu_owner_reduce_scatterv_semantics(self):
        self._run_two_rank_reduce_scatterv_backend("owner_broadcast")

    def test_two_rank_cpu_custom_reduce_scatterv_fallback_semantics(self):
        self._run_two_rank_reduce_scatterv_backend("custom")

    def _run_two_rank_allgatherv_backend(self, backend: str) -> None:
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, f"allgatherv_{backend}_init")
            mp.spawn(
                _run_two_rank_allgatherv_backend,
                args=(world_size, init_file, backend),
                nprocs=world_size,
                join=True,
            )

    def _run_two_rank_fused_allgatherv_backend(self, backend: str, custom_impl: str | None = None) -> None:
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, f"fused_allgatherv_{backend}_init")
            mp.spawn(
                _run_two_rank_fused_allgatherv_backend,
                args=(world_size, init_file, backend, custom_impl),
                nprocs=world_size,
                join=True,
            )

    def _run_two_rank_reduce_scatterv_backend(self, backend: str) -> None:
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, f"reduce_scatterv_{backend}_init")
            mp.spawn(
                _run_two_rank_reduce_scatterv_backend,
                args=(world_size, init_file, backend),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
