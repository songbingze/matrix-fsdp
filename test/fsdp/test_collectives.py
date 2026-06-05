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
    MatrixCollectiveHandle,
    MatrixTensorCollectiveHandle,
    _make_uneven_all_gather_output_list,
    _make_uneven_reduce_scatter_input_list,
    owner_collective_signature_validation_enabled,
    validate_owner_collective_signature,
    zero_size_uneven_reduce_scatter_enabled,
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


def _run_two_rank_owner_signature_mismatch(rank: int, world_size: int, init_file: str) -> None:
    os.environ["MATRIX_FSDP_VALIDATE_OWNER_COLLECTIVE_SIGNATURE"] = "1"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        if rank == 0:
            rank_segments = (
                (LayoutSegment(0, 2, 0),),
                (LayoutSegment(2, 5, 0),),
            )
        else:
            rank_segments = (
                (LayoutSegment(0, 3, 0),),
                (LayoutSegment(3, 5, 0),),
            )
        try:
            validate_owner_collective_signature(
                collective_key="mismatch_unit",
                backend="native_sendrecv_rank_chunks",
                rank_segments=rank_segments,
                output_numel=5,
            )
        except RuntimeError as exc:
            assert "owner collective signature mismatch" in str(exc)
        else:
            raise AssertionError("Expected owner collective signature mismatch.")
    finally:
        os.environ.pop("MATRIX_FSDP_VALIDATE_OWNER_COLLECTIVE_SIGNATURE", None)
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available(), "torch.distributed is not available")
class MatrixCollectiveBackendTest(unittest.TestCase):
    def test_collective_handles_release_wait_closure_after_wait(self):
        result = torch.ones(2)
        handle = MatrixCollectiveHandle(lambda: result)

        self.assertIs(handle.wait(), result)
        self.assertIsNone(handle._wait_fn)
        self.assertIs(handle.wait(), result)

    def test_tensor_collective_handles_release_wait_closure_after_wait(self):
        initial = torch.zeros(2)
        result = torch.ones(2)
        handle = MatrixTensorCollectiveHandle(initial, lambda: result)

        self.assertIs(handle.wait(), result)
        self.assertIsNone(handle._wait_fn)
        self.assertTrue(handle._waited)
        self.assertIs(handle.wait(), result)

    def test_owner_collective_signature_validation_is_debug_opt_in(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(owner_collective_signature_validation_enabled())
        with unittest.mock.patch.dict(os.environ, {"MATRIX_FSDP_VALIDATE_OWNER_COLLECTIVE_SIGNATURE": "1"}):
            self.assertTrue(owner_collective_signature_validation_enabled())

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

    def test_uneven_reduce_scatter_keeps_zero_size_rank_chunks(self):
        packed = torch.arange(3, dtype=torch.float32)
        input_list = _make_uneven_reduce_scatter_input_list(packed, (3, 0))

        self.assertEqual(input_list[0].data_ptr(), packed.data_ptr())
        self.assertEqual(input_list[1].numel(), 0)
        torch.testing.assert_close(input_list[0], torch.tensor([0.0, 1.0, 2.0]))

    def test_zero_size_uneven_reduce_scatter_env_toggle(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(zero_size_uneven_reduce_scatter_enabled())
        with unittest.mock.patch.dict(os.environ, {"MATRIX_FSDP_UNEVEN_REDUCE_SCATTER_ZERO_SIZE": "owner_reduce"}):
            self.assertFalse(zero_size_uneven_reduce_scatter_enabled())
        with unittest.mock.patch.dict(os.environ, {"MATRIX_FSDP_UNEVEN_REDUCE_SCATTER_ZERO_SIZE": "allow"}):
            self.assertTrue(zero_size_uneven_reduce_scatter_enabled())

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

    def test_two_rank_owner_collective_signature_mismatch_raises(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "owner_signature_mismatch_init")
            mp.spawn(
                _run_two_rank_owner_signature_mismatch,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

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
