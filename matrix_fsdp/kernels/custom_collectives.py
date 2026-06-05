from __future__ import annotations

import os
from time import perf_counter
from typing import Literal

import torch
import torch.distributed as dist

from matrix_fsdp.core.layout import LayoutSegment
from matrix_fsdp.kernels.gin.native import gin_device_rank_chunks, rma_putsignal_rank_chunks
from matrix_fsdp.kernels.native import (
    native_copy_rank_chunk_from_packed,
    native_copy_rank_segments_to_full,
    native_group_broadcast_rank_segments,
    native_kernel_available,
    native_nccl_comm_lanes_available,
    native_reduce_rank_chunks,
    native_sendrecv_rank_chunks,
    native_sendrecv_rank_segments,
)
from matrix_fsdp.kernels.segment_fusion import coalesce_rank_segments

MatrixCollectiveBackend = Literal["torch", "owner_broadcast", "custom"]
_MATRIX_COLLECTIVE_BACKENDS = {"torch", "owner_broadcast", "custom"}
_CUSTOM_ALLGATHERV_IMPL_ENV = "MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL"
_NATIVE_SENDRECV_CHUNK_FAST_PATH_ENV = "MATRIX_FSDP_NATIVE_SENDRECV_CHUNK_FAST_PATH"
_TORCH_BATCH_SENDRECV_CHUNK_FAST_PATH_ENV = "MATRIX_FSDP_TORCH_BATCH_SENDRECV_CHUNK_FAST_PATH"
_AUTO_NATIVE_SENDRECV_CHUNKS_ENV = "MATRIX_FSDP_AUTO_NATIVE_SENDRECV_CHUNKS"
_ALLOW_NATIVE_SEGMENT_P2P_ENV = "MATRIX_FSDP_ALLOW_NATIVE_SEGMENT_P2P"
_OWNER_SEGMENT_PREFETCH_ENV = "MATRIX_FSDP_OWNER_SEGMENT_PREFETCH"
_AUTO_CUSTOM_ALLGATHERV_IMPL = "auto"
_DEFAULT_CUSTOM_ALLGATHERV_IMPL = _AUTO_CUSTOM_ALLGATHERV_IMPL
_STABLE_CUSTOM_ALLGATHERV_IMPLS = {
    _AUTO_CUSTOM_ALLGATHERV_IMPL,
    "native_group_broadcast",
    "native_sendrecv",
    "uneven_all_gather",
}
_DIAGNOSTIC_CUSTOM_ALLGATHERV_IMPLS = {
    "broadcast",
    "all_reduce",
}
_EXPERIMENTAL_CUSTOM_ALLGATHERV_IMPLS = {
    "rma_put_signal",
    "gin_device",
}
_CUSTOM_ALLGATHERV_IMPLS = (
    _STABLE_CUSTOM_ALLGATHERV_IMPLS | _DIAGNOSTIC_CUSTOM_ALLGATHERV_IMPLS | _EXPERIMENTAL_CUSTOM_ALLGATHERV_IMPLS
)
_CUSTOM_REDUCE_SCATTERV_IMPL_ENV = "MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL"
_CUSTOM_REDUCE_SCATTERV_IMPLS = {"native_reduce", "reduce", "uneven_reduce_scatter"}


def normalize_matrix_collective_backend(backend: str) -> MatrixCollectiveBackend:
    if backend not in _MATRIX_COLLECTIVE_BACKENDS:
        valid = ", ".join(repr(name) for name in sorted(_MATRIX_COLLECTIVE_BACKENDS))
        raise ValueError(f"matrix_collective_backend must be one of {valid}, got {backend!r}.")
    return backend  # type: ignore[return-value]


def custom_all_gatherv_rank_segments_1d_into_async(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    collective_key: str | None = None,
    validate_signature: bool = True,
    coalesced_rank_segments: tuple[tuple[LayoutSegment, ...], ...] | None = None,
    rank_chunk_shard_sizes: tuple[int, ...] | None = None,
    rank_chunk_segments: tuple[tuple[LayoutSegment, ...], ...] | None = None,
):
    from matrix_fsdp.runtime.collectives import (
        MatrixCollectiveHandle,
        dist_broadcast_group_rank,
        dist_is_ready,
        set_collective_metadata,
        validate_owner_collective_signature,
    )

    _validate_owner_allgatherv_inputs(local_tensor, output_tensor, rank_segments, rank, cuda_stream)
    fused_rank_segments = coalesced_rank_segments
    if len(rank_segments) == 1 or not dist_is_ready():
        if len(rank_segments) != 1:
            raise RuntimeError("Custom owner allgatherv without a process group only supports one local shard.")
        _copy_local_rank_segments(local_tensor, output_tensor, rank_segments[rank])
        return MatrixCollectiveHandle(lambda: output_tensor)

    impl = resolve_custom_allgatherv_impl(rank_segments)
    if impl == "native_group_broadcast":
        phase_timings: dict[str, float] = {}
        if validate_signature:
            phase_start = perf_counter()
            validate_owner_collective_signature(
                collective_key=collective_key,
                backend="native_group_broadcast",
                rank_segments=fused_rank_segments or coalesce_rank_segments(rank_segments),
                output_numel=output_tensor.numel(),
                group=group,
            )
            phase_timings["signature_validate_ms"] = (perf_counter() - phase_start) * 1000.0
        phase_start = perf_counter()
        native_handle = _try_native_group_broadcast_rank_segments(
            local_tensor,
            output_tensor,
            rank_segments,
            rank,
            group=group,
            cuda_stream=cuda_stream,
            force=True,
        )
        phase_timings["native_enqueue_ms"] = (perf_counter() - phase_start) * 1000.0
        if native_handle is not None:
            return _with_phase_timings(
                set_collective_metadata(
                    native_handle,
                    kind="param_all_gather",
                    backend="custom",
                    impl="native_group_broadcast",
                    numel=output_tensor.numel(),
                    element_size=output_tensor.element_size(),
                    count=sum(
                        1
                        for segments in (fused_rank_segments or coalesce_rank_segments(rank_segments))
                        for segment in segments
                        if segment.numel > 0
                    ),
                ),
                phase_timings,
            )
        return _all_gather_uneven_rank_segments_1d_into_async(
            local_tensor,
            output_tensor,
            rank_segments,
            rank,
            group=group,
            cuda_stream=cuda_stream,
        )

    if impl == "native_sendrecv":
        shard_sizes = (
            rank_chunk_shard_sizes
            if native_sendrecv_chunk_fast_path_enabled()
            else None
        )
        if shard_sizes is None and native_sendrecv_chunk_fast_path_enabled():
            shard_sizes = _rank_chunk_shard_sizes(rank_segments)
        if shard_sizes is not None:
            phase_timings: dict[str, float] = {}
            if validate_signature:
                phase_start = perf_counter()
                validate_owner_collective_signature(
                    collective_key=collective_key,
                    backend="native_sendrecv_rank_chunks",
                    rank_segments=rank_chunk_segments or _rank_chunk_segments_from_sizes(shard_sizes),
                    output_numel=output_tensor.numel(),
                    group=group,
                )
                phase_timings["signature_validate_ms"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
            native_chunk_handle = _try_native_sendrecv_rank_chunks(
                local_tensor,
                output_tensor,
                shard_sizes,
                rank,
                group=group,
                cuda_stream=cuda_stream,
                force=True,
            )
            phase_timings["native_enqueue_ms"] = (perf_counter() - phase_start) * 1000.0
            if native_chunk_handle is not None:
                return _with_phase_timings(
                    set_collective_metadata(
                        native_chunk_handle,
                        kind="param_all_gather",
                        backend="custom",
                        impl="native_sendrecv",
                        numel=output_tensor.numel(),
                        element_size=output_tensor.element_size(),
                        count=sum(1 for size in shard_sizes if size > 0),
                    ),
                    phase_timings,
                )
            phase_start = perf_counter()
            torch_chunk_handle = _try_torch_batch_sendrecv_rank_chunks(
                local_tensor,
                output_tensor,
                shard_sizes,
                rank,
                group=group,
                cuda_stream=cuda_stream,
            )
            phase_timings["torch_batch_enqueue_ms"] = (perf_counter() - phase_start) * 1000.0
            if torch_chunk_handle is not None:
                return _with_phase_timings(
                    set_collective_metadata(
                        torch_chunk_handle,
                        kind="param_all_gather",
                        backend="custom",
                        impl="torch_batch_sendrecv",
                        numel=output_tensor.numel(),
                        element_size=output_tensor.element_size(),
                        count=sum(1 for size in shard_sizes if size > 0),
                    ),
                    phase_timings,
                )
        if native_segment_p2p_enabled():
            phase_timings = {}
            if validate_signature:
                phase_start = perf_counter()
                validate_owner_collective_signature(
                    collective_key=collective_key,
                    backend="native_sendrecv_rank_segments",
                    rank_segments=fused_rank_segments or coalesce_rank_segments(rank_segments),
                    output_numel=output_tensor.numel(),
                    group=group,
                )
                phase_timings["signature_validate_ms"] = (perf_counter() - phase_start) * 1000.0
            phase_start = perf_counter()
            native_handle = _try_native_sendrecv_rank_segments(
                local_tensor,
                output_tensor,
                rank_segments,
                rank,
                group=group,
                cuda_stream=cuda_stream,
                force=True,
            )
            phase_timings["native_enqueue_ms"] = (perf_counter() - phase_start) * 1000.0
            if native_handle is not None:
                return _with_phase_timings(
                    set_collective_metadata(
                        native_handle,
                        kind="param_all_gather",
                        backend="custom",
                        impl="native_sendrecv_segments",
                        numel=output_tensor.numel(),
                        element_size=output_tensor.element_size(),
                        count=sum(1 for segments in rank_segments for segment in segments if segment.numel > 0),
                    ),
                    phase_timings,
                )
        return _all_gather_uneven_rank_segments_1d_into_async(
            local_tensor,
            output_tensor,
            rank_segments,
            rank,
            group=group,
            cuda_stream=cuda_stream,
        )

    if impl in {"rma_put_signal", "gin_device"}:
        shard_sizes = rank_chunk_shard_sizes
        if shard_sizes is None:
            shard_sizes = _rank_chunk_shard_sizes(rank_segments)
        if shard_sizes is not None:
            if validate_signature:
                validate_owner_collective_signature(
                    collective_key=collective_key,
                    backend=impl,
                    rank_segments=rank_chunk_segments or _rank_chunk_segments_from_sizes(shard_sizes),
                    output_numel=output_tensor.numel(),
                    group=group,
                )
            backend = rma_putsignal_rank_chunks if impl == "rma_put_signal" else gin_device_rank_chunks
            native_handle = _try_experimental_rank_chunk_backend(
                backend,
                local_tensor,
                output_tensor,
                shard_sizes,
                rank,
                group=group,
                cuda_stream=cuda_stream,
                force=True,
            )
            if native_handle is not None:
                return native_handle
        return _native_sendrecv_or_uneven_fallback(
            local_tensor,
            output_tensor,
            rank_segments,
            rank,
            group=group,
            cuda_stream=cuda_stream,
            collective_key=collective_key,
            validate_signature=validate_signature,
        )

    if impl == "uneven_all_gather":
        return _all_gather_uneven_rank_segments_1d_into_async(
            local_tensor,
            output_tensor,
            rank_segments,
            rank,
            group=group,
            cuda_stream=cuda_stream,
        )

    if impl == "all_reduce":
        return _all_reduce_rank_segments_1d_into_async(
            local_tensor,
            output_tensor,
            rank_segments,
            rank,
            group=group,
            cuda_stream=cuda_stream,
        )

    fused_rank_segments = fused_rank_segments or coalesce_rank_segments(rank_segments)
    if impl != "broadcast":
        raise AssertionError(f"Unhandled custom allgatherv impl: {impl}")

    if validate_signature:
        validate_owner_collective_signature(
            collective_key=collective_key,
            backend="custom_broadcast",
            rank_segments=fused_rank_segments,
            output_numel=output_tensor.numel(),
            group=group,
        )
    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(local_tensor.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            _copy_local_rank_segments(local_tensor, output_tensor, fused_rank_segments[rank])
            for src_rank, segments in enumerate(fused_rank_segments):
                for segment in segments:
                    dist_broadcast_group_rank(
                        output_tensor[segment.global_start : segment.global_end],
                        src_rank,
                        group=group,
                        async_op=False,
                    )
            event = torch.cuda.Event()
            event.record(cuda_stream)

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(local_tensor.device).wait_event(event)
            return output_tensor

        return MatrixCollectiveHandle(wait)

    _copy_local_rank_segments(local_tensor, output_tensor, fused_rank_segments[rank])
    works = []
    for src_rank, segments in enumerate(fused_rank_segments):
        for segment in segments:
            works.append(
                dist_broadcast_group_rank(
                    output_tensor[segment.global_start : segment.global_end],
                    src_rank,
                    group=group,
                    async_op=True,
                )
            )

    def wait() -> torch.Tensor:
        for work in works:
            work.wait()
        return output_tensor

    return MatrixCollectiveHandle(wait)


def _all_gather_uneven_rank_segments_1d_into_async(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
):
    from matrix_fsdp.runtime.collectives import all_gather_uneven_rank_segments_1d_into_async

    return all_gather_uneven_rank_segments_1d_into_async(
        local_tensor,
        output_tensor,
        rank_segments,
        rank,
        group=group,
        cuda_stream=cuda_stream,
    )


def _with_phase_timings(handle, phase_timings: dict[str, float]):
    handle.collective_phase_timings.update(phase_timings)
    original_wait = handle.wait

    def wait():
        phase_start = perf_counter()
        result = original_wait()
        handle.collective_phase_timings["wait_ms"] = (perf_counter() - phase_start) * 1000.0
        return result

    handle.wait = wait
    return handle


def _try_native_group_broadcast_rank_segments(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    force: bool = False,
):
    from matrix_fsdp.runtime.collectives import MatrixCollectiveHandle

    native_result = native_group_broadcast_rank_segments(
        local_tensor,
        output_tensor,
        rank_segments,
        rank,
        group=group,
        cuda_stream=cuda_stream,
        force=force,
    )
    if native_result is False:
        return None
    if isinstance(native_result, torch.cuda.Event):
        event = native_result

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(local_tensor.device).wait_event(event)
            return output_tensor

        return MatrixCollectiveHandle(wait)
    return MatrixCollectiveHandle(lambda: output_tensor)


def _try_native_sendrecv_rank_chunks(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    force: bool = False,
):
    from matrix_fsdp.runtime.collectives import MatrixCollectiveHandle

    native_result = native_sendrecv_rank_chunks(
        local_tensor,
        output_tensor,
        shard_sizes,
        rank,
        group=group,
        cuda_stream=cuda_stream,
        force=force,
    )
    if native_result is False:
        return None
    if isinstance(native_result, torch.cuda.Event):
        event = native_result

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(local_tensor.device).wait_event(event)
            return output_tensor

        return MatrixCollectiveHandle(wait)
    return MatrixCollectiveHandle(lambda: output_tensor)


def _try_torch_batch_sendrecv_rank_chunks(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
):
    from matrix_fsdp.runtime.collectives import MatrixCollectiveHandle

    if not torch_batch_sendrecv_chunk_fast_path_enabled():
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    if not hasattr(dist, "batch_isend_irecv") or not hasattr(dist, "P2POp"):
        return None
    if local_tensor.numel() != shard_sizes[rank]:
        return None
    if not local_tensor.is_contiguous() or not output_tensor.is_contiguous():
        return None

    rank_offsets = _rank_chunk_offsets(shard_sizes)
    if len(shard_sizes) == 1:
        _copy_rank_chunk_to_full(local_tensor, output_tensor, shard_sizes, rank, rank_offsets)
        return MatrixCollectiveHandle(lambda: output_tensor)

    def enqueue_batch() -> list:
        _copy_rank_chunk_to_full(local_tensor, output_tensor, shard_sizes, rank, rank_offsets)
        ops = []
        for peer_rank, shard_size in enumerate(shard_sizes):
            if peer_rank == rank or shard_size == 0:
                continue
            start = rank_offsets[peer_rank]
            recv_tensor = output_tensor[start : start + shard_size]
            ops.append(_p2p_op(dist.irecv, recv_tensor, peer_rank, group=group))
        if shard_sizes[rank] > 0:
            for peer_rank in range(len(shard_sizes)):
                if peer_rank == rank:
                    continue
                ops.append(_p2p_op(dist.isend, local_tensor, peer_rank, group=group))
        return dist.batch_isend_irecv(ops) if ops else []

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(local_tensor.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            works = enqueue_batch()
            event = torch.cuda.Event()
            event.record(cuda_stream)

        def wait() -> torch.Tensor:
            for work in works:
                work.wait()
            torch.cuda.current_stream(local_tensor.device).wait_event(event)
            return output_tensor

        return MatrixCollectiveHandle(wait)

    works = enqueue_batch()

    def wait() -> torch.Tensor:
        for work in works:
            work.wait()
        return output_tensor

    return MatrixCollectiveHandle(wait)


def _try_experimental_rank_chunk_backend(
    backend,
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    force: bool = False,
):
    from matrix_fsdp.runtime.collectives import MatrixCollectiveHandle

    native_result = backend(
        local_tensor,
        output_tensor,
        shard_sizes,
        rank,
        group=group,
        cuda_stream=cuda_stream,
        force=force,
    )
    if native_result is False:
        return None
    if isinstance(native_result, torch.cuda.Event):
        event = native_result

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(local_tensor.device).wait_event(event)
            return output_tensor

        return MatrixCollectiveHandle(wait)
    return MatrixCollectiveHandle(lambda: output_tensor)


def _try_native_sendrecv_rank_segments(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    force: bool = False,
):
    from matrix_fsdp.runtime.collectives import MatrixCollectiveHandle

    native_result = native_sendrecv_rank_segments(
        local_tensor,
        output_tensor,
        rank_segments,
        rank,
        group=group,
        cuda_stream=cuda_stream,
        force=force,
    )
    if native_result is False:
        return None
    if isinstance(native_result, torch.cuda.Event):
        event = native_result

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(local_tensor.device).wait_event(event)
            return output_tensor

        return MatrixCollectiveHandle(wait)
    return MatrixCollectiveHandle(lambda: output_tensor)


def _native_sendrecv_or_uneven_fallback(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    collective_key: str | None = None,
    validate_signature: bool = True,
):
    from matrix_fsdp.runtime.collectives import validate_owner_collective_signature

    shard_sizes = _rank_chunk_shard_sizes(rank_segments)
    if shard_sizes is not None and native_sendrecv_chunk_fast_path_enabled():
        if validate_signature:
            validate_owner_collective_signature(
                collective_key=collective_key,
                backend="native_sendrecv_rank_chunks",
                rank_segments=_rank_chunk_segments_from_sizes(shard_sizes),
                output_numel=output_tensor.numel(),
                group=group,
            )
        native_chunk_handle = _try_native_sendrecv_rank_chunks(
            local_tensor,
            output_tensor,
            shard_sizes,
            rank,
            group=group,
            cuda_stream=cuda_stream,
            force=True,
        )
        if native_chunk_handle is not None:
            return native_chunk_handle
    if native_segment_p2p_enabled():
        if validate_signature:
            validate_owner_collective_signature(
                collective_key=collective_key,
                backend="native_sendrecv_rank_segments",
                rank_segments=coalesce_rank_segments(rank_segments),
                output_numel=output_tensor.numel(),
                group=group,
            )
        native_handle = _try_native_sendrecv_rank_segments(
            local_tensor,
            output_tensor,
            rank_segments,
            rank,
            group=group,
            cuda_stream=cuda_stream,
            force=True,
        )
        if native_handle is not None:
            return native_handle
    return _all_gather_uneven_rank_segments_1d_into_async(
        local_tensor,
        output_tensor,
        rank_segments,
        rank,
        group=group,
        cuda_stream=cuda_stream,
    )


def custom_allgatherv_impl() -> str:
    value = os.environ.get(_CUSTOM_ALLGATHERV_IMPL_ENV, _DEFAULT_CUSTOM_ALLGATHERV_IMPL).lower()
    if value not in _CUSTOM_ALLGATHERV_IMPLS:
        valid = ", ".join(repr(name) for name in sorted(_CUSTOM_ALLGATHERV_IMPLS))
        raise ValueError(f"{_CUSTOM_ALLGATHERV_IMPL_ENV} must be one of {valid}, got {value!r}.")
    return value


def default_custom_allgatherv_impl() -> str:
    return _DEFAULT_CUSTOM_ALLGATHERV_IMPL


def resolve_custom_allgatherv_impl(rank_segments: tuple[tuple[LayoutSegment, ...], ...]) -> str:
    impl = custom_allgatherv_impl()
    if impl != _AUTO_CUSTOM_ALLGATHERV_IMPL:
        return impl
    if (
        auto_native_sendrecv_chunks_enabled()
        and native_sendrecv_chunk_fast_path_enabled()
        and _rank_chunk_shard_sizes(rank_segments) is not None
    ):
        return "native_sendrecv"
    return "uneven_all_gather"


def experimental_custom_allgatherv_impls() -> tuple[str, ...]:
    return tuple(sorted(_EXPERIMENTAL_CUSTOM_ALLGATHERV_IMPLS))


def is_experimental_custom_allgatherv_impl(value: str) -> bool:
    return value in _EXPERIMENTAL_CUSTOM_ALLGATHERV_IMPLS


def custom_allgatherv_allows_owner_prefetch(
    value: str | None = None,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...] | None = None,
    *,
    ordered: bool = False,
) -> bool:
    policy = os.environ.get(_OWNER_SEGMENT_PREFETCH_ENV, "auto").lower()
    impl = custom_allgatherv_impl() if value is None else value.lower()
    if policy in {"0", "false", "no", "off"}:
        return False
    if policy in {"1", "true", "yes", "on"}:
        return True
    if policy != "auto":
        raise ValueError(
            f"{_OWNER_SEGMENT_PREFETCH_ENV} must be 'auto', 'on', or 'off', got {policy!r}."
        )
    resolved_impl = _resolve_prefetch_impl(impl, rank_segments)
    if ordered and resolved_impl in {"native_group_broadcast", "native_sendrecv", "uneven_all_gather"}:
        return rank_segments is not None
    if resolved_impl == "native_group_broadcast":
        return native_kernel_available()
    if resolved_impl == "native_sendrecv":
        return (
            native_kernel_available()
            and native_sendrecv_chunk_fast_path_enabled()
            and rank_segments is not None
            and _rank_chunk_shard_sizes(rank_segments) is not None
        )
    return False


def custom_allgatherv_owner_prefetch_skip_reason(
    value: str | None = None,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...] | None = None,
    *,
    ordered: bool = False,
) -> str | None:
    impl = custom_allgatherv_impl() if value is None else value.lower()
    if custom_allgatherv_allows_owner_prefetch(impl, rank_segments, ordered=ordered):
        return None
    return f"custom_allgatherv:{_resolve_prefetch_impl(impl, rank_segments)}"


def custom_allgatherv_has_independent_native_comm_lanes(
    value: str | None = None,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...] | None = None,
) -> bool:
    impl = custom_allgatherv_impl() if value is None else value.lower()
    resolved_impl = _resolve_prefetch_impl(impl, rank_segments)
    if resolved_impl not in {"native_group_broadcast", "native_sendrecv"}:
        return False
    return native_nccl_comm_lanes_available()


def _resolve_prefetch_impl(
    impl: str,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...] | None,
) -> str:
    if impl != _AUTO_CUSTOM_ALLGATHERV_IMPL:
        return impl
    if (
        auto_native_sendrecv_chunks_enabled()
        and native_sendrecv_chunk_fast_path_enabled()
        and rank_segments is not None
        and _rank_chunk_shard_sizes(rank_segments) is not None
    ):
        return "native_sendrecv"
    return "uneven_all_gather"


def custom_reduce_scatterv_impl() -> str:
    value = os.environ.get(_CUSTOM_REDUCE_SCATTERV_IMPL_ENV, "uneven_reduce_scatter").lower()
    if value not in _CUSTOM_REDUCE_SCATTERV_IMPLS:
        valid = ", ".join(repr(name) for name in sorted(_CUSTOM_REDUCE_SCATTERV_IMPLS))
        raise ValueError(f"{_CUSTOM_REDUCE_SCATTERV_IMPL_ENV} must be one of {valid}, got {value!r}.")
    return value


def native_sendrecv_chunk_fast_path_enabled() -> bool:
    value = os.environ.get(_NATIVE_SENDRECV_CHUNK_FAST_PATH_ENV, "1").lower()
    return value not in {"0", "false", "no", "off"}


def torch_batch_sendrecv_chunk_fast_path_enabled() -> bool:
    value = os.environ.get(_TORCH_BATCH_SENDRECV_CHUNK_FAST_PATH_ENV, "").lower()
    return value in {"1", "true", "yes", "on"}


def auto_native_sendrecv_chunks_enabled() -> bool:
    value = os.environ.get(_AUTO_NATIVE_SENDRECV_CHUNKS_ENV, "0").lower()
    return value not in {"0", "false", "no", "off"}


def native_segment_p2p_enabled() -> bool:
    value = os.environ.get(_ALLOW_NATIVE_SEGMENT_P2P_ENV, "").lower()
    return value in {"1", "true", "yes", "on"}


def _all_reduce_rank_segments_1d_into_async(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
):
    from matrix_fsdp.runtime.collectives import MatrixCollectiveHandle

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(local_tensor.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            output_tensor.zero_()
            _copy_local_rank_segments(local_tensor, output_tensor, rank_segments[rank])
            torch.distributed.all_reduce(output_tensor, group=group, async_op=False)
            event = torch.cuda.Event()
            event.record(cuda_stream)

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(local_tensor.device).wait_event(event)
            return output_tensor

        return MatrixCollectiveHandle(wait)

    output_tensor.zero_()
    _copy_local_rank_segments(local_tensor, output_tensor, rank_segments[rank])
    work = torch.distributed.all_reduce(output_tensor, group=group, async_op=True)

    def wait() -> torch.Tensor:
        work.wait()
        return output_tensor

    return MatrixCollectiveHandle(wait)


def custom_reduce_scatterv_owner_rank_chunks_1d_async(
    packed_rank_chunks: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    divide_by_world: bool = True,
    cuda_stream: torch.cuda.Stream | None = None,
    compact: bool = False,
):
    from matrix_fsdp.runtime.collectives import (
        MatrixTensorCollectiveHandle,
        dist_is_ready,
        dist_reduce_group_rank,
    )

    _validate_owner_reduce_scatterv_inputs(packed_rank_chunks, shard_sizes, cuda_stream, compact)
    if len(shard_sizes) == 1 or not dist_is_ready():
        if len(shard_sizes) != 1:
            raise RuntimeError("Custom owner reduce without a process group only supports one local shard.")
        local_grad_shard = packed_rank_chunks.new_empty(shard_sizes[rank])
        _copy_rank_chunk_from_packed(packed_rank_chunks, local_grad_shard, shard_sizes, rank, compact=compact)
        return MatrixTensorCollectiveHandle(local_grad_shard, lambda: local_grad_shard, _waited=True)

    reduce_impl = custom_reduce_scatterv_impl()
    if reduce_impl == "native_reduce":
        phase_start = perf_counter()
        native_handle = _try_native_reduce_rank_chunks(
            packed_rank_chunks,
            shard_sizes,
            rank,
            group=group,
            divide_by_world=divide_by_world,
            cuda_stream=cuda_stream,
            compact=compact,
            force=True,
        )
        phase_timings = {"native_enqueue_ms": (perf_counter() - phase_start) * 1000.0}
        if native_handle is not None:
            return _with_phase_timings(native_handle, phase_timings)
        reduce_impl = "uneven_reduce_scatter"

    if reduce_impl == "uneven_reduce_scatter":
        from matrix_fsdp.runtime.collectives import reduce_scatter_uneven_rank_chunks_1d_async, set_collective_metadata

        phase_start = perf_counter()
        handle = reduce_scatter_uneven_rank_chunks_1d_async(
            packed_rank_chunks,
            shard_sizes,
            rank,
            group=group,
            divide_by_world=divide_by_world,
            cuda_stream=cuda_stream,
            compact=compact,
        )
        phase_timings = {"torch_enqueue_ms": (perf_counter() - phase_start) * 1000.0}
        return _with_phase_timings(
            set_collective_metadata(
                handle,
                kind="grad_reduce_scatter",
                backend="custom",
                impl="uneven_reduce_scatter",
                numel=sum(shard_sizes),
                element_size=packed_rank_chunks.element_size(),
                count=sum(1 for size in shard_sizes if size > 0),
            ),
            phase_timings,
        )

    local_grad_shard = packed_rank_chunks.new_empty(shard_sizes[rank])

    def rank_chunk(owner_rank: int) -> torch.Tensor:
        start = _rank_chunk_offset(shard_sizes, owner_rank, compact=compact)
        return packed_rank_chunks[start : start + shard_sizes[owner_rank]]

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(packed_rank_chunks.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            for owner_rank, size in enumerate(shard_sizes):
                if size == 0:
                    continue
                dist_reduce_group_rank(rank_chunk(owner_rank), owner_rank, group=group, async_op=False)
            if divide_by_world:
                rank_chunk(rank).div_(len(shard_sizes))
            _copy_rank_chunk_from_packed(packed_rank_chunks, local_grad_shard, shard_sizes, rank, compact=compact)
            event = torch.cuda.Event()
            event.record(cuda_stream)

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(packed_rank_chunks.device).wait_event(event)
            return local_grad_shard

        return MatrixTensorCollectiveHandle(local_grad_shard, wait)

    works = []
    for owner_rank, size in enumerate(shard_sizes):
        if size == 0:
            continue
        works.append(dist_reduce_group_rank(rank_chunk(owner_rank), owner_rank, group=group, async_op=True))

    def wait() -> torch.Tensor:
        for work in works:
            work.wait()
        if divide_by_world:
            rank_chunk(rank).div_(len(shard_sizes))
        _copy_rank_chunk_from_packed(packed_rank_chunks, local_grad_shard, shard_sizes, rank, compact=compact)
        return local_grad_shard

    return MatrixTensorCollectiveHandle(local_grad_shard, wait)


def _try_native_reduce_rank_chunks(
    packed_rank_chunks: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    divide_by_world: bool = True,
    cuda_stream: torch.cuda.Stream | None = None,
    compact: bool = False,
    force: bool = False,
):
    from matrix_fsdp.runtime.collectives import MatrixTensorCollectiveHandle, set_collective_metadata

    local_grad_shard = packed_rank_chunks.new_empty(shard_sizes[rank])
    native_result = native_reduce_rank_chunks(
        packed_rank_chunks,
        local_grad_shard,
        shard_sizes,
        rank,
        group=group,
        divide_by_world=divide_by_world,
        compact=compact,
        cuda_stream=cuda_stream,
        force=force,
    )
    if native_result is False:
        return None
    if isinstance(native_result, torch.cuda.Event):
        event = native_result

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(packed_rank_chunks.device).wait_event(event)
            return local_grad_shard

        handle = MatrixTensorCollectiveHandle(local_grad_shard, wait)
    else:
        handle = MatrixTensorCollectiveHandle(local_grad_shard, lambda: local_grad_shard)
    return set_collective_metadata(
        handle,
        kind="grad_reduce_scatter",
        backend="custom",
        impl="native_reduce",
        numel=sum(shard_sizes),
        element_size=packed_rank_chunks.element_size(),
        count=sum(1 for size in shard_sizes if size > 0),
    )


def _validate_owner_allgatherv_inputs(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    cuda_stream: torch.cuda.Stream | None,
) -> None:
    if local_tensor.ndim != 1:
        raise ValueError(f"local_tensor must be 1D, got shape {tuple(local_tensor.shape)}.")
    if output_tensor.ndim != 1:
        raise ValueError(f"output_tensor must be 1D, got shape {tuple(output_tensor.shape)}.")
    expected_numel = sum(segment.numel for segments in rank_segments for segment in segments)
    if output_tensor.numel() != expected_numel:
        raise ValueError(f"output_tensor has {output_tensor.numel()} elements, expected {expected_numel}.")
    local_numel = sum(segment.numel for segment in rank_segments[rank])
    if local_tensor.numel() != local_numel:
        raise ValueError(f"local_tensor has {local_tensor.numel()} elements, expected {local_numel}.")
    if cuda_stream is not None and not local_tensor.is_cuda:
        raise ValueError("cuda_stream can only be used with CUDA tensors.")


def _copy_local_rank_segments(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    segments: tuple[LayoutSegment, ...],
) -> None:
    if native_copy_rank_segments_to_full(local_tensor, output_tensor, segments):
        return
    for segment in segments:
        output_tensor[segment.global_start : segment.global_end].copy_(
            local_tensor[segment.local_start : segment.local_end]
        )


def _rank_chunk_shard_sizes(
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
) -> tuple[int, ...] | None:
    cursor = 0
    shard_sizes: list[int] = []
    for segments in rank_segments:
        if not segments:
            shard_sizes.append(0)
            continue
        if len(segments) != 1:
            return None
        segment = segments[0]
        if segment.global_start != cursor or segment.local_start != 0:
            return None
        shard_sizes.append(segment.numel)
        cursor = segment.global_end
    return tuple(shard_sizes)


def _rank_chunk_segments_from_sizes(shard_sizes: tuple[int, ...]) -> tuple[tuple[LayoutSegment, ...], ...]:
    cursor = 0
    rank_segments: list[tuple[LayoutSegment, ...]] = []
    for shard_size in shard_sizes:
        if shard_size == 0:
            rank_segments.append(())
            continue
        rank_segments.append((LayoutSegment(cursor, cursor + shard_size, 0),))
        cursor += shard_size
    return tuple(rank_segments)


def _rank_chunk_offsets(shard_sizes: tuple[int, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    for shard_size in shard_sizes:
        offsets.append(cursor)
        cursor += shard_size
    return tuple(offsets)


def _copy_rank_chunk_to_full(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    rank_offsets: tuple[int, ...],
) -> None:
    shard_size = shard_sizes[rank]
    if shard_size == 0:
        return
    start = rank_offsets[rank]
    output_tensor[start : start + shard_size].copy_(local_tensor)


def _p2p_op(op, tensor: torch.Tensor, peer_rank: int, *, group=None):
    if group is not None:
        try:
            return dist.P2POp(op, tensor, group=group, group_peer=peer_rank)
        except TypeError:
            pass
        peer_rank = dist.get_global_rank(group, peer_rank) if hasattr(dist, "get_global_rank") else peer_rank
    return dist.P2POp(op, tensor, peer_rank, group=group)


def _validate_owner_reduce_scatterv_inputs(
    packed_rank_chunks: torch.Tensor,
    shard_sizes: tuple[int, ...],
    cuda_stream: torch.cuda.Stream | None,
    compact: bool,
) -> None:
    if packed_rank_chunks.ndim != 1:
        raise ValueError(f"packed_rank_chunks must be 1D, got shape {tuple(packed_rank_chunks.shape)}.")
    if not shard_sizes:
        raise ValueError("shard_sizes must be non-empty.")
    max_shard_size = max(shard_sizes)
    expected_numel = sum(shard_sizes) if compact else len(shard_sizes) * max_shard_size
    if packed_rank_chunks.numel() != expected_numel:
        raise ValueError(f"packed_rank_chunks has {packed_rank_chunks.numel()} elements, expected {expected_numel}.")
    if cuda_stream is not None and not packed_rank_chunks.is_cuda:
        raise ValueError("cuda_stream can only be used with CUDA tensors.")


def _copy_rank_chunk_from_packed(
    packed_rank_chunks: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    compact: bool,
) -> None:
    if native_copy_rank_chunk_from_packed(packed_rank_chunks, output_tensor, shard_sizes, rank, compact=compact):
        return
    offset = _rank_chunk_offset(shard_sizes, rank, compact=compact)
    output_tensor.copy_(packed_rank_chunks[offset : offset + shard_sizes[rank]])


def _rank_chunk_offset(shard_sizes: tuple[int, ...], rank: int, *, compact: bool) -> int:
    return sum(shard_sizes[:rank]) if compact else rank * max(shard_sizes)
