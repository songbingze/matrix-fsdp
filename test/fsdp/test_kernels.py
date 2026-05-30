from unittest import mock
import unittest

import torch

from matrix_fsdp.kernels import (
    NativeKernelStatus,
    coalesce_contiguous_segments,
    native_copy_kernels_enabled,
    native_nccl_collectives_enabled,
    native_kernel_available,
    native_kernel_status,
)
from matrix_fsdp.kernels.custom_collectives import (
    custom_allgatherv_impl,
    custom_all_gatherv_rank_segments_1d_into_async,
    custom_reduce_scatterv_owner_rank_chunks_1d_async,
    default_custom_allgatherv_impl,
    experimental_custom_allgatherv_impls,
    is_experimental_custom_allgatherv_impl,
    native_sendrecv_chunk_fast_path_enabled,
    _rank_chunk_shard_sizes,
)
from matrix_fsdp.kernels.gin import (
    gin_backend_info,
    gin_kernels_enabled,
    gin_native_available,
    gin_native_status,
    rma_putsignal_rank_chunks,
)
from matrix_fsdp.kernels.native import (
    _rank_segment_metadata_tensors,
    native_copy_rank_chunk_from_packed,
    native_copy_rank_segments_to_full,
    native_sendrecv_rank_chunks,
)
from matrix_fsdp.layout import LayoutSegment


class MatrixKernelTest(unittest.TestCase):
    def test_native_kernel_status_is_reportable_without_cuda_extension(self):
        status = native_kernel_status()

        self.assertIsInstance(status, NativeKernelStatus)
        self.assertIsInstance(status.available, bool)
        self.assertEqual(native_kernel_available(), status.available)

    def test_native_copy_helpers_fallback_on_cpu(self):
        source = torch.arange(3, dtype=torch.float32)
        output = torch.empty(3, dtype=torch.float32)

        self.assertFalse(native_copy_rank_segments_to_full(source, output, (LayoutSegment(0, 3, 0),)))
        self.assertFalse(native_copy_rank_chunk_from_packed(source, output, (3,), 0, compact=True))
        self.assertFalse(native_sendrecv_rank_chunks(source, output, (3,), 0))
        self.assertFalse(rma_putsignal_rank_chunks(source, output, (3,), 0))

    def test_coalesce_contiguous_segments_preserves_gaps(self):
        fused = coalesce_contiguous_segments(
            (
                LayoutSegment(0, 2, 0),
                LayoutSegment(2, 5, 2),
                LayoutSegment(7, 9, 5),
                LayoutSegment(9, 10, 7),
            )
        )

        self.assertEqual(
            fused,
            (
                LayoutSegment(0, 5, 0),
                LayoutSegment(7, 10, 5),
            ),
        )

    def test_native_copy_kernels_are_opt_in(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(native_copy_kernels_enabled())
        with mock.patch.dict("os.environ", {"MATRIX_FSDP_ENABLE_NATIVE_COPY_KERNELS": "1"}):
            self.assertTrue(native_copy_kernels_enabled())

    def test_native_nccl_collectives_are_opt_in(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(native_nccl_collectives_enabled())
        with mock.patch.dict("os.environ", {"MATRIX_FSDP_ENABLE_NATIVE_NCCL": "1"}):
            self.assertTrue(native_nccl_collectives_enabled())
        with mock.patch.dict("os.environ", {"MATRIX_FSDP_ENABLE_NATIVE_NCCL": "group_broadcast"}):
            self.assertTrue(native_nccl_collectives_enabled())
        with mock.patch.dict("os.environ", {"MATRIX_FSDP_ENABLE_NATIVE_NCCL": "unsafe_group_broadcast"}):
            self.assertTrue(native_nccl_collectives_enabled())

    def test_custom_allgatherv_defaults_to_native_sendrecv(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(custom_allgatherv_impl(), "native_sendrecv")
        self.assertEqual(default_custom_allgatherv_impl(), "native_sendrecv")

    def test_custom_allgatherv_accepts_native_sendrecv_impl(self):
        with mock.patch.dict("os.environ", {"MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL": "native_sendrecv"}):
            self.assertEqual(custom_allgatherv_impl(), "native_sendrecv")
        with mock.patch.dict("os.environ", {"MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL": "rma_put_signal"}):
            self.assertEqual(custom_allgatherv_impl(), "rma_put_signal")
        with mock.patch.dict("os.environ", {"MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL": "gin_device"}):
            self.assertEqual(custom_allgatherv_impl(), "gin_device")

    def test_experimental_allgatherv_impls_are_explicit(self):
        self.assertEqual(experimental_custom_allgatherv_impls(), ("gin_device", "rma_put_signal"))
        self.assertFalse(is_experimental_custom_allgatherv_impl(default_custom_allgatherv_impl()))
        self.assertTrue(is_experimental_custom_allgatherv_impl("rma_put_signal"))
        self.assertTrue(is_experimental_custom_allgatherv_impl("gin_device"))

    def test_gin_native_status_is_reportable_without_extension(self):
        status = gin_native_status()

        self.assertIsInstance(status.available, bool)
        self.assertEqual(gin_native_available(), status.available)
        self.assertIn("available", gin_backend_info())

    def test_gin_kernels_are_opt_in(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(gin_kernels_enabled())
        with mock.patch.dict("os.environ", {"MATRIX_FSDP_ENABLE_GIN_KERNELS": "1"}):
            self.assertTrue(gin_kernels_enabled())

    def test_native_sendrecv_chunk_fast_path_defaults_enabled(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertTrue(native_sendrecv_chunk_fast_path_enabled())
        with mock.patch.dict("os.environ", {"MATRIX_FSDP_NATIVE_SENDRECV_CHUNK_FAST_PATH": "0"}):
            self.assertFalse(native_sendrecv_chunk_fast_path_enabled())

    def test_native_nccl_rank_segment_metadata_is_cpu_readable(self):
        src_ranks, global_starts, local_starts, numels = _rank_segment_metadata_tensors(
            (
                (LayoutSegment(0, 3, 0),),
                (LayoutSegment(3, 5, 0),),
            ),
            device=torch.device("cuda", 0),
        )

        self.assertFalse(src_ranks.is_cuda)
        torch.testing.assert_close(global_starts, torch.tensor([0, 3]))
        torch.testing.assert_close(local_starts, torch.tensor([0, 0]))
        torch.testing.assert_close(numels, torch.tensor([3, 2]))

    def test_custom_allgatherv_uses_native_segment_copy_when_available(self):
        def fake_native_copy(local_tensor, output_tensor, segments):
            for segment in segments:
                output_tensor[segment.global_start : segment.global_end].copy_(
                    local_tensor[segment.local_start : segment.local_end]
                )
            return True

        local = torch.arange(4, dtype=torch.float32)
        output = torch.empty(4, dtype=torch.float32)

        with mock.patch.dict("os.environ", {"MATRIX_FSDP_ENABLE_NATIVE_COPY_KERNELS": "1"}), mock.patch(
            "matrix_fsdp.kernels.custom_collectives.native_copy_rank_segments_to_full",
            side_effect=fake_native_copy,
        ) as patched:
            handle = custom_all_gatherv_rank_segments_1d_into_async(
                local,
                output,
                ((LayoutSegment(0, 4, 0),),),
                0,
            )

        patched.assert_called_once()
        torch.testing.assert_close(handle.wait(), local)

    def test_custom_reduce_scatterv_uses_native_chunk_copy_when_available(self):
        def fake_native_copy(packed_rank_chunks, output_tensor, shard_sizes, rank, *, compact):
            offset = sum(shard_sizes[:rank]) if compact else rank * max(shard_sizes)
            output_tensor.copy_(packed_rank_chunks[offset : offset + shard_sizes[rank]])
            return True

        packed = torch.arange(4, dtype=torch.float32)

        with mock.patch.dict("os.environ", {"MATRIX_FSDP_ENABLE_NATIVE_COPY_KERNELS": "1"}), mock.patch(
            "matrix_fsdp.kernels.custom_collectives.native_copy_rank_chunk_from_packed",
            side_effect=fake_native_copy,
        ) as patched:
            handle = custom_reduce_scatterv_owner_rank_chunks_1d_async(
                packed,
                (4,),
                0,
                compact=True,
            )

        patched.assert_called_once()
        torch.testing.assert_close(handle.wait(), packed)

    def test_rank_chunk_shard_sizes_detects_contiguous_rank_chunks(self):
        self.assertEqual(
            _rank_chunk_shard_sizes(
                (
                    (LayoutSegment(0, 3, 0),),
                    (LayoutSegment(3, 5, 0),),
                    (),
                    (LayoutSegment(5, 9, 0),),
                )
            ),
            (3, 2, 0, 4),
        )
        self.assertIsNone(
            _rank_chunk_shard_sizes(
                (
                    (LayoutSegment(0, 3, 0), LayoutSegment(7, 9, 3)),
                    (LayoutSegment(3, 7, 0),),
                )
            )
        )
        self.assertIsNone(
            _rank_chunk_shard_sizes(
                (
                    (LayoutSegment(0, 3, 1),),
                    (LayoutSegment(3, 7, 0),),
                )
            )
        )

    def test_custom_allgatherv_uses_native_sendrecv_chunk_path_for_rank_chunks(self):
        local = torch.arange(2, dtype=torch.float32)
        output = torch.empty(5, dtype=torch.float32)
        rank_segments = (
            (LayoutSegment(0, 2, 0),),
            (LayoutSegment(2, 5, 0),),
        )

        with mock.patch.dict("os.environ", {"MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL": "native_sendrecv"}), mock.patch(
            "matrix_fsdp.runtime.collectives.dist_is_ready",
            return_value=True,
        ), mock.patch(
            "matrix_fsdp.kernels.custom_collectives.native_sendrecv_rank_chunks",
            return_value=True,
        ) as chunk_sendrecv, mock.patch(
            "matrix_fsdp.kernels.custom_collectives.native_sendrecv_rank_segments",
            return_value=True,
        ) as segment_sendrecv:
            handle = custom_all_gatherv_rank_segments_1d_into_async(
                local,
                output,
                rank_segments,
                0,
            )

        self.assertIs(handle.wait(), output)
        chunk_sendrecv.assert_called_once()
        segment_sendrecv.assert_not_called()

    def test_custom_allgatherv_can_try_rma_backend_for_rank_chunks(self):
        local = torch.arange(2, dtype=torch.float32)
        output = torch.empty(5, dtype=torch.float32)
        rank_segments = (
            (LayoutSegment(0, 2, 0),),
            (LayoutSegment(2, 5, 0),),
        )

        with mock.patch.dict("os.environ", {"MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL": "rma_put_signal"}), mock.patch(
            "matrix_fsdp.runtime.collectives.dist_is_ready",
            return_value=True,
        ), mock.patch(
            "matrix_fsdp.kernels.custom_collectives.rma_putsignal_rank_chunks",
            return_value=True,
        ) as rma_backend, mock.patch(
            "matrix_fsdp.kernels.custom_collectives.native_sendrecv_rank_segments",
            return_value=True,
        ) as segment_sendrecv:
            handle = custom_all_gatherv_rank_segments_1d_into_async(
                local,
                output,
                rank_segments,
                0,
            )

        self.assertIs(handle.wait(), output)
        rma_backend.assert_called_once()
        segment_sendrecv.assert_not_called()

    def test_custom_allgatherv_falls_back_when_gin_backend_unavailable(self):
        local = torch.arange(2, dtype=torch.float32)
        output = torch.empty(5, dtype=torch.float32)
        rank_segments = (
            (LayoutSegment(0, 2, 0),),
            (LayoutSegment(2, 5, 0),),
        )

        with mock.patch.dict("os.environ", {"MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL": "gin_device"}), mock.patch(
            "matrix_fsdp.runtime.collectives.dist_is_ready",
            return_value=True,
        ), mock.patch(
            "matrix_fsdp.kernels.custom_collectives.gin_device_rank_chunks",
            return_value=False,
        ) as gin_backend, mock.patch(
            "matrix_fsdp.kernels.custom_collectives.native_sendrecv_rank_segments",
            return_value=True,
        ) as segment_sendrecv:
            handle = custom_all_gatherv_rank_segments_1d_into_async(
                local,
                output,
                rank_segments,
                0,
            )

        self.assertIs(handle.wait(), output)
        gin_backend.assert_called_once()
        segment_sendrecv.assert_called_once()


if __name__ == "__main__":
    unittest.main()
