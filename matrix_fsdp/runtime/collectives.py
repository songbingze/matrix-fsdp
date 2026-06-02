from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
import os

import torch
import torch.distributed as dist

from matrix_fsdp.core.layout import LayoutSegment
from matrix_fsdp.core.placement import MatrixShard
from matrix_fsdp.kernels import (
    MatrixCollectiveBackend,
    coalesce_rank_segments,
    custom_all_gatherv_rank_segments_1d_into_async,
    custom_reduce_scatterv_owner_rank_chunks_1d_async,
    normalize_matrix_collective_backend,
)

_VALIDATE_OWNER_COLLECTIVE_SIGNATURE_ENV = "MATRIX_FSDP_VALIDATE_OWNER_COLLECTIVE_SIGNATURE"
_OWNER_COLLECTIVE_SEQUENCE = count()


@dataclass
class MatrixCollectiveHandle:
    _wait_fn: Callable[[], torch.Tensor]
    _result: torch.Tensor | None = None
    collective_kind: str | None = None
    collective_backend: str | None = None
    collective_impl: str | None = None
    collective_numel: int = 0
    collective_bytes: int = 0
    collective_count: int = 0

    def wait(self) -> torch.Tensor:
        if self._result is None:
            self._result = self._wait_fn()
        return self._result


@dataclass
class MatrixTensorCollectiveHandle:
    tensor: torch.Tensor
    _wait_fn: Callable[[], torch.Tensor]
    _waited: bool = False
    collective_kind: str | None = None
    collective_backend: str | None = None
    collective_impl: str | None = None
    collective_numel: int = 0
    collective_bytes: int = 0
    collective_count: int = 0

    def wait(self) -> torch.Tensor:
        if not self._waited:
            self.tensor = self._wait_fn()
            self._waited = True
        return self.tensor


def set_collective_metadata(
    handle: MatrixCollectiveHandle | MatrixTensorCollectiveHandle,
    *,
    kind: str,
    backend: str | None,
    impl: str | None,
    numel: int,
    element_size: int,
    count: int,
) -> MatrixCollectiveHandle | MatrixTensorCollectiveHandle:
    handle.collective_kind = kind
    handle.collective_backend = backend
    handle.collective_impl = impl
    handle.collective_numel = int(numel)
    handle.collective_bytes = int(numel) * int(element_size)
    handle.collective_count = int(count)
    return handle


def dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def validate_owner_collective_signature(
    *,
    collective_key: str | None,
    backend: str,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    output_numel: int,
    group=None,
) -> None:
    if not owner_collective_signature_validation_enabled():
        return
    if not (dist.is_available() and dist.is_initialized()):
        return

    segment_signature = _rank_segments_signature(rank_segments)
    shard_sizes = tuple(sum(segment.numel for segment in segments) for segments in rank_segments)
    sequence = next(_OWNER_COLLECTIVE_SEQUENCE)
    local_signature = (
        "matrix_fsdp_owner_collective_v1",
        sequence,
        collective_key or "",
        backend,
        len(rank_segments),
        output_numel,
        shard_sizes,
        segment_signature,
    )
    gathered: list[object] = [None for _ in range(dist.get_world_size(group=group))]
    dist.all_gather_object(gathered, local_signature, group=group)
    first = gathered[0]
    if any(signature != first for signature in gathered):
        detail = "\n".join(f"rank{idx}: {signature!r}" for idx, signature in enumerate(gathered))
        raise RuntimeError(
            "MatrixFSDP owner collective signature mismatch before entering the P2P/NCCL path. "
            "All ranks must issue the same owner collective sequence with identical rank layout metadata.\n"
            f"{detail}"
        )


def owner_collective_signature_validation_enabled() -> bool:
    value = os.environ.get(_VALIDATE_OWNER_COLLECTIVE_SIGNATURE_ENV, "1").lower()
    return value not in {"0", "false", "no", "off"}


def _rank_segments_signature(
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    return tuple(
        tuple((segment.global_start, segment.global_end, segment.local_start, segment.numel) for segment in segments)
        for segments in rank_segments
    )


def all_gather_matrix_1d(
    local_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    output_tensor = local_tensor.new_empty(sum(shard_sizes))
    handle = all_gather_matrix_1d_async(
        local_tensor,
        shard_sizes,
        output_tensor=output_tensor,
        group=group,
        cuda_stream=cuda_stream,
    )
    return handle.wait()


def all_gather_matrix_1d_async(
    local_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    *,
    output_tensor: torch.Tensor | None = None,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
) -> MatrixCollectiveHandle:
    output_tensor = output_tensor if output_tensor is not None else local_tensor.new_empty(sum(shard_sizes))
    return all_gather_matrix_1d_into_async(
        local_tensor,
        output_tensor,
        shard_sizes,
        group=group,
        cuda_stream=cuda_stream,
    )


def all_gather_matrix_1d_into_async(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
) -> MatrixCollectiveHandle:
    if len(shard_sizes) == 1 or not dist_is_ready():
        if output_tensor.numel() != local_tensor.numel():
            raise RuntimeError("Matrix all-gather without a process group only supports one local shard.")
        return MatrixCollectiveHandle(lambda: output_tensor.copy_(local_tensor))

    max_size = max(shard_sizes)
    if local_tensor.ndim != 1:
        raise ValueError(f"local_tensor must be 1D, got shape {tuple(local_tensor.shape)}.")
    if output_tensor.ndim != 1:
        raise ValueError(f"output_tensor must be 1D, got shape {tuple(output_tensor.shape)}.")
    if output_tensor.numel() != sum(shard_sizes):
        raise ValueError(f"output_tensor has {output_tensor.numel()} elements, expected {sum(shard_sizes)}.")

    if cuda_stream is not None and not local_tensor.is_cuda:
        raise ValueError("cuda_stream can only be used with CUDA tensors.")

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(local_tensor.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            padded = local_tensor.new_zeros(max_size)
            padded[: local_tensor.numel()].copy_(local_tensor)
            gathered_padded = _new_matrix_all_gather_output(padded, len(shard_sizes))
            if gathered_padded is not None:
                dist.all_gather_into_tensor(gathered_padded, padded, group=group)
                gathered = None
            else:
                gathered = [local_tensor.new_empty(max_size) for _ in shard_sizes]
                dist.all_gather(gathered, padded, group=group)
            event = torch.cuda.Event()
            event.record(cuda_stream)
    else:
        padded = local_tensor.new_zeros(max_size)
        padded[: local_tensor.numel()].copy_(local_tensor)
        gathered_padded = _new_matrix_all_gather_output(padded, len(shard_sizes))
        if gathered_padded is not None:
            gathered = None
            work = dist.all_gather_into_tensor(gathered_padded, padded, group=group, async_op=True)
        else:
            gathered = [local_tensor.new_empty(max_size) for _ in shard_sizes]
            work = dist.all_gather(gathered, padded, group=group, async_op=True)
        event = None

    def wait() -> torch.Tensor:
        if event is not None:
            torch.cuda.current_stream(local_tensor.device).wait_event(event)
        else:
            work.wait()
        offset = 0
        for rank, size in enumerate(shard_sizes):
            tensor = (
                gathered_padded[rank * max_size : (rank + 1) * max_size]
                if gathered_padded is not None
                else gathered[rank]
            )
            output_tensor[offset : offset + size].copy_(tensor[:size])
            offset += size
        return output_tensor

    return MatrixCollectiveHandle(wait)


def _new_matrix_all_gather_output(local_tensor: torch.Tensor, world_size: int) -> torch.Tensor | None:
    if not hasattr(dist, "all_gather_into_tensor"):
        return None
    if not local_tensor.is_cuda:
        return None
    return local_tensor.new_empty(local_tensor.numel() * world_size)


def all_gatherv_rank_segments_1d_into_async(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    backend: MatrixCollectiveBackend = "owner_broadcast",
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    collective_key: str | None = None,
    validate_owner_collective_signature: bool = True,
) -> MatrixCollectiveHandle:
    backend = normalize_matrix_collective_backend(backend)
    if backend == "custom":
        return custom_all_gatherv_rank_segments_1d_into_async(
            local_tensor,
            output_tensor,
            rank_segments,
            rank,
            group=group,
            cuda_stream=cuda_stream,
            collective_key=collective_key,
            validate_signature=validate_owner_collective_signature,
        )
    if backend != "owner_broadcast":
        raise ValueError("all_gatherv_rank_segments_1d_into_async() supports 'owner_broadcast' or 'custom'.")
    return broadcast_rank_segments_1d_into_async(
        local_tensor,
        output_tensor,
        rank_segments,
        rank,
        group=group,
        cuda_stream=cuda_stream,
        collective_key=collective_key,
        validate_signature=validate_owner_collective_signature,
    )


def all_gather_uneven_rank_segments_1d_into_async(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
) -> MatrixCollectiveHandle:
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

    if len(rank_segments) == 1 or not dist_is_ready():
        if len(rank_segments) != 1:
            raise RuntimeError("Uneven all-gather without a process group only supports one local shard.")
        _copy_local_rank_segments(local_tensor, output_tensor, rank_segments[rank])
        return MatrixCollectiveHandle(lambda: output_tensor)

    if not _supports_uneven_all_gather(local_tensor, group):
        return broadcast_rank_segments_1d_into_async(
            local_tensor,
            output_tensor,
            rank_segments,
            rank,
            group=group,
            cuda_stream=cuda_stream,
        )

    fused_rank_segments = coalesce_rank_segments(rank_segments)
    shard_sizes = tuple(sum(segment.numel for segment in segments) for segments in fused_rank_segments)
    if any(size == 0 for size in shard_sizes):
        return broadcast_rank_segments_1d_into_async(
            local_tensor,
            output_tensor,
            rank_segments,
            rank,
            group=group,
            cuda_stream=cuda_stream,
        )

    collective_input = local_tensor if local_tensor.is_contiguous() else local_tensor.contiguous()
    output_list, staged_outputs = _make_uneven_all_gather_output_list(
        collective_input,
        output_tensor,
        fused_rank_segments,
        shard_sizes,
    )

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(local_tensor.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            dist.all_gather(output_list, collective_input, group=group, async_op=False)
            _copy_staged_uneven_all_gather_outputs(output_tensor, fused_rank_segments, staged_outputs)
            event = torch.cuda.Event()
            event.record(cuda_stream)

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(local_tensor.device).wait_event(event)
            return output_tensor

        return MatrixCollectiveHandle(wait)

    work = dist.all_gather(output_list, collective_input, group=group, async_op=True)

    def wait() -> torch.Tensor:
        work.wait()
        _copy_staged_uneven_all_gather_outputs(output_tensor, fused_rank_segments, staged_outputs)
        return output_tensor

    return MatrixCollectiveHandle(wait)


def broadcast_rank_segments_1d_into_async(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    collective_key: str | None = None,
    validate_signature: bool = True,
) -> MatrixCollectiveHandle:
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

    if len(rank_segments) == 1 or not dist_is_ready():
        if len(rank_segments) != 1:
            raise RuntimeError("Owner broadcast without a process group only supports one local shard.")
        _copy_local_rank_segments(local_tensor, output_tensor, rank_segments[rank])
        return MatrixCollectiveHandle(lambda: output_tensor)

    fused_rank_segments = coalesce_rank_segments(rank_segments)
    if validate_signature:
        validate_owner_collective_signature(
            collective_key=collective_key,
            backend="owner_broadcast",
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


def _copy_local_rank_segments(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    segments: tuple[LayoutSegment, ...],
) -> None:
    for segment in segments:
        output_tensor[segment.global_start : segment.global_end].copy_(
            local_tensor[segment.local_start : segment.local_end]
        )


def _supports_uneven_all_gather(local_tensor: torch.Tensor, group) -> bool:
    if not local_tensor.is_cuda:
        return False
    try:
        backend = dist.get_backend(group)
    except Exception:
        return False
    return str(backend).lower() == "nccl"


def _make_uneven_all_gather_output_list(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    shard_sizes: tuple[int, ...],
) -> tuple[list[torch.Tensor], list[tuple[int, torch.Tensor]]]:
    output_list: list[torch.Tensor] = []
    staged_outputs: list[tuple[int, torch.Tensor]] = []
    for src_rank, segments in enumerate(rank_segments):
        direct_output = _direct_uneven_all_gather_output_view(output_tensor, segments, shard_sizes[src_rank])
        if direct_output is not None:
            output_list.append(direct_output)
            continue
        staged = local_tensor.new_empty(shard_sizes[src_rank])
        output_list.append(staged)
        staged_outputs.append((src_rank, staged))
    return output_list, staged_outputs


def _direct_uneven_all_gather_output_view(
    output_tensor: torch.Tensor,
    segments: tuple[LayoutSegment, ...],
    shard_size: int,
) -> torch.Tensor | None:
    if len(segments) != 1:
        return None
    segment = segments[0]
    if segment.local_start != 0 or segment.numel != shard_size:
        return None
    output_view = output_tensor[segment.global_start : segment.global_end]
    return output_view if output_view.is_contiguous() else None


def _copy_staged_uneven_all_gather_outputs(
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    staged_outputs: list[tuple[int, torch.Tensor]],
) -> None:
    for src_rank, staged_output in staged_outputs:
        _copy_local_rank_segments(staged_output, output_tensor, rank_segments[src_rank])


def dist_broadcast_group_rank(
    tensor: torch.Tensor,
    src_rank: int,
    *,
    group=None,
    async_op: bool = False,
):
    if group is not None:
        try:
            return dist.broadcast(tensor, group=group, group_src=src_rank, async_op=async_op)
        except TypeError:
            pass
    src = dist.get_global_rank(group, src_rank) if group is not None and hasattr(dist, "get_global_rank") else src_rank
    return dist.broadcast(tensor, src=src, group=group, async_op=async_op)


def reduce_scatterv_owner_rank_chunks_1d_async(
    packed_rank_chunks: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    backend: MatrixCollectiveBackend = "owner_broadcast",
    group=None,
    divide_by_world: bool = True,
    cuda_stream: torch.cuda.Stream | None = None,
    compact: bool = False,
) -> MatrixTensorCollectiveHandle:
    backend = normalize_matrix_collective_backend(backend)
    if backend == "custom":
        return custom_reduce_scatterv_owner_rank_chunks_1d_async(
            packed_rank_chunks,
            shard_sizes,
            rank,
            group=group,
            divide_by_world=divide_by_world,
            cuda_stream=cuda_stream,
            compact=compact,
        )
    if backend != "owner_broadcast":
        raise ValueError("reduce_scatterv_owner_rank_chunks_1d_async() supports 'owner_broadcast' or 'custom'.")
    return reduce_owner_rank_chunks_1d_async(
        packed_rank_chunks,
        shard_sizes,
        rank,
        group=group,
        divide_by_world=divide_by_world,
        cuda_stream=cuda_stream,
        compact=compact,
    )


def reduce_scatter_uneven_rank_chunks_1d_async(
    packed_rank_chunks: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    divide_by_world: bool = True,
    cuda_stream: torch.cuda.Stream | None = None,
    compact: bool = False,
) -> MatrixTensorCollectiveHandle:
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

    if (
        len(shard_sizes) == 1
        or not dist_is_ready()
        or not compact
        or any(size == 0 for size in shard_sizes)
        or not _supports_uneven_reduce_scatter(packed_rank_chunks, group)
    ):
        return reduce_owner_rank_chunks_1d_async(
            packed_rank_chunks,
            shard_sizes,
            rank,
            group=group,
            divide_by_world=divide_by_world,
            cuda_stream=cuda_stream,
            compact=compact,
        )

    local_grad_shard = packed_rank_chunks.new_empty(shard_sizes[rank])
    input_list = _make_uneven_reduce_scatter_input_list(packed_rank_chunks, shard_sizes)

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(packed_rank_chunks.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            dist.reduce_scatter(local_grad_shard, input_list, op=dist.ReduceOp.SUM, group=group, async_op=False)
            if divide_by_world:
                local_grad_shard.div_(len(shard_sizes))
            event = torch.cuda.Event()
            event.record(cuda_stream)

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(packed_rank_chunks.device).wait_event(event)
            return local_grad_shard

        return MatrixTensorCollectiveHandle(local_grad_shard, wait)

    work = dist.reduce_scatter(local_grad_shard, input_list, op=dist.ReduceOp.SUM, group=group, async_op=True)

    def wait() -> torch.Tensor:
        work.wait()
        if divide_by_world:
            local_grad_shard.div_(len(shard_sizes))
        return local_grad_shard

    return MatrixTensorCollectiveHandle(local_grad_shard, wait)


def reduce_owner_rank_chunks_1d_async(
    packed_rank_chunks: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    divide_by_world: bool = True,
    cuda_stream: torch.cuda.Stream | None = None,
    compact: bool = False,
) -> MatrixTensorCollectiveHandle:
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

    local_grad_shard = packed_rank_chunks.new_empty(shard_sizes[rank])
    if len(shard_sizes) == 1 or not dist_is_ready():
        if len(shard_sizes) != 1:
            raise RuntimeError("Owner reduce without a process group only supports one local shard.")
        local_grad_shard.copy_(packed_rank_chunks[: shard_sizes[rank]])
        return MatrixTensorCollectiveHandle(local_grad_shard, lambda: local_grad_shard, _waited=True)

    def rank_chunk(owner_rank: int) -> torch.Tensor:
        start = sum(shard_sizes[:owner_rank]) if compact else owner_rank * max_shard_size
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
            local_grad_shard.copy_(rank_chunk(rank))
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
        local_grad_shard.copy_(rank_chunk(rank))
        return local_grad_shard

    return MatrixTensorCollectiveHandle(local_grad_shard, wait)


def dist_reduce_group_rank(
    tensor: torch.Tensor,
    dst_rank: int,
    *,
    group=None,
    async_op: bool = False,
):
    if group is not None:
        try:
            return dist.reduce(tensor, dst=None, group=group, group_dst=dst_rank, async_op=async_op)
        except TypeError:
            pass
    dst = dist.get_global_rank(group, dst_rank) if group is not None and hasattr(dist, "get_global_rank") else dst_rank
    return dist.reduce(tensor, dst=dst, group=group, async_op=async_op)


def _supports_uneven_reduce_scatter(tensor: torch.Tensor, group) -> bool:
    if not tensor.is_cuda:
        return False
    try:
        backend = dist.get_backend(group)
    except Exception:
        return False
    return str(backend).lower() == "nccl"


def _make_uneven_reduce_scatter_input_list(
    packed_rank_chunks: torch.Tensor,
    shard_sizes: tuple[int, ...],
) -> list[torch.Tensor]:
    input_list = []
    offset = 0
    for size in shard_sizes:
        input_list.append(packed_rank_chunks[offset : offset + size])
        offset += size
    return input_list


def all_gather_matrix_shard_1d_async(
    local_tensor: torch.Tensor,
    placement: MatrixShard,
    total_numel: int,
    *,
    output_tensor: torch.Tensor | None = None,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
) -> MatrixCollectiveHandle:
    return all_gather_matrix_1d_async(
        local_tensor,
        placement.shard_lengths(total_numel),
        output_tensor=output_tensor,
        group=group,
        cuda_stream=cuda_stream,
    )


def all_gather_matrix_shard_1d_into_async(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    placement: MatrixShard,
    total_numel: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
) -> MatrixCollectiveHandle:
    return all_gather_matrix_1d_into_async(
        local_tensor,
        output_tensor,
        placement.shard_lengths(total_numel),
        group=group,
        cuda_stream=cuda_stream,
    )


def all_gather_equal_1d_into_async(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
) -> MatrixCollectiveHandle:
    if local_tensor.ndim != 1:
        raise ValueError(f"local_tensor must be 1D, got shape {tuple(local_tensor.shape)}.")
    if output_tensor.ndim != 1:
        raise ValueError(f"output_tensor must be 1D, got shape {tuple(output_tensor.shape)}.")
    if output_tensor.numel() == local_tensor.numel():
        return MatrixCollectiveHandle(lambda: output_tensor.copy_(local_tensor))
    if not dist_is_ready():
        raise RuntimeError("all_gather_equal_1d_into_async() requires an initialized process group.")
    world_size = dist.get_world_size(group)
    if output_tensor.numel() != local_tensor.numel() * world_size:
        raise ValueError(
            f"output_tensor has {output_tensor.numel()} elements, expected {local_tensor.numel() * world_size}."
        )
    if cuda_stream is not None and not local_tensor.is_cuda:
        raise ValueError("cuda_stream can only be used with CUDA tensors.")

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(local_tensor.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            dist.all_gather_into_tensor(output_tensor, local_tensor, group=group)
            event = torch.cuda.Event()
            event.record(cuda_stream)
    else:
        work = dist.all_gather_into_tensor(output_tensor, local_tensor, group=group, async_op=True)
        event = None

    def wait() -> torch.Tensor:
        if event is not None:
            torch.cuda.current_stream(local_tensor.device).wait_event(event)
        else:
            work.wait()
        return output_tensor

    return MatrixCollectiveHandle(wait)


def all_reduce_full_grad(
    grad_buffer: torch.Tensor,
    *,
    group=None,
    divide_by_world: bool = True,
) -> torch.Tensor:
    if not dist_is_ready():
        return grad_buffer

    dist.all_reduce(grad_buffer, group=group)
    if divide_by_world:
        world_size = dist.get_world_size(group)
        grad_buffer.div_(world_size)
    return grad_buffer


def reduce_scatter_equal_1d(
    full_tensor: torch.Tensor,
    local_size: int,
    *,
    group=None,
    divide_by_world: bool = True,
) -> torch.Tensor:
    if full_tensor.ndim != 1:
        raise ValueError(f"full_tensor must be 1D, got shape {tuple(full_tensor.shape)}.")
    if local_size < 0:
        raise ValueError(f"local_size must be non-negative, got {local_size}.")
    if full_tensor.numel() == local_size or not dist_is_ready():
        return full_tensor[:local_size].contiguous()

    world_size = dist.get_world_size(group)
    if full_tensor.numel() != local_size * world_size:
        raise ValueError(f"full_tensor has {full_tensor.numel()} elements, expected {local_size * world_size}.")

    output = full_tensor.new_empty(local_size)
    dist.reduce_scatter_tensor(output, full_tensor, group=group)
    if divide_by_world:
        output.div_(world_size)
    return output


def reduce_scatter_padded_rank_chunks_1d(
    packed_rank_chunks: torch.Tensor,
    local_size: int,
    *,
    group=None,
    divide_by_world: bool = True,
) -> torch.Tensor:
    return reduce_scatter_padded_rank_chunks_1d_async(
        packed_rank_chunks,
        local_size,
        group=group,
        divide_by_world=divide_by_world,
    ).wait()


def reduce_scatter_padded_rank_chunks_1d_async(
    packed_rank_chunks: torch.Tensor,
    local_size: int,
    *,
    group=None,
    divide_by_world: bool = True,
    cuda_stream: torch.cuda.Stream | None = None,
) -> MatrixTensorCollectiveHandle:
    if packed_rank_chunks.ndim != 1:
        raise ValueError(f"packed_rank_chunks must be 1D, got shape {tuple(packed_rank_chunks.shape)}.")
    if local_size < 0:
        raise ValueError(f"local_size must be non-negative, got {local_size}.")
    if not dist_is_ready():
        local_tensor = packed_rank_chunks[:local_size].contiguous()
        return MatrixTensorCollectiveHandle(local_tensor, lambda: local_tensor, _waited=True)
    if cuda_stream is not None and not packed_rank_chunks.is_cuda:
        raise ValueError("cuda_stream can only be used with CUDA tensors.")

    world_size = dist.get_world_size(group)
    if packed_rank_chunks.numel() % world_size != 0:
        raise ValueError(
            f"packed_rank_chunks has {packed_rank_chunks.numel()} elements, which is not divisible by {world_size}."
        )
    padded_local_size = packed_rank_chunks.numel() // world_size
    if local_size > padded_local_size:
        raise ValueError(f"local_size {local_size} exceeds padded local size {padded_local_size}.")

    output = packed_rank_chunks.new_empty(padded_local_size)
    local_tensor = output if local_size == padded_local_size else output[:local_size]

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(packed_rank_chunks.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            # Match FSDP2's scheduling: enqueue the collective on the dedicated
            # comm stream and use stream events for async behavior relative to
            # compute, instead of holding an async Work handle.
            dist.reduce_scatter_tensor(output, packed_rank_chunks, group=group, async_op=False)
            if divide_by_world:
                output.div_(world_size)
            event = torch.cuda.Event()
            event.record(cuda_stream)

        def wait() -> torch.Tensor:
            torch.cuda.current_stream(packed_rank_chunks.device).wait_event(event)
            return local_tensor

        return MatrixTensorCollectiveHandle(local_tensor, wait)

    work = dist.reduce_scatter_tensor(output, packed_rank_chunks, group=group, async_op=True)

    def wait() -> torch.Tensor:
        work.wait()
        if divide_by_world:
            output.div_(world_size)
        return local_tensor

    return MatrixTensorCollectiveHandle(local_tensor, wait)


def reduce_scatter_matrix_1d(
    full_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    divide_by_world: bool = True,
) -> torch.Tensor:
    if full_tensor.ndim != 1:
        raise ValueError(f"full_tensor must be 1D, got shape {tuple(full_tensor.shape)}.")
    if full_tensor.numel() != sum(shard_sizes):
        raise ValueError(f"full_tensor has {full_tensor.numel()} elements, expected {sum(shard_sizes)}.")

    local_size = shard_sizes[rank]
    if len(shard_sizes) == 1 or not dist_is_ready():
        return full_tensor[:local_size].contiguous()

    max_size = max(shard_sizes)
    padded_chunks = []
    offset = 0
    for shard_size in shard_sizes:
        chunk = full_tensor[offset : offset + shard_size]
        padded = full_tensor.new_zeros(max_size)
        padded[:shard_size].copy_(chunk)
        padded_chunks.append(padded)
        offset += shard_size

    padded_output = full_tensor.new_empty(max_size)
    if hasattr(dist, "reduce_scatter_tensor"):
        dist.reduce_scatter_tensor(padded_output, torch.cat(padded_chunks), group=group)
    else:
        dist.reduce_scatter(padded_output, padded_chunks, group=group)

    if divide_by_world:
        world_size = dist.get_world_size(group)
        padded_output.div_(world_size)
    return padded_output[:local_size].contiguous()


def reduce_scatter_matrix_shard_1d(
    full_tensor: torch.Tensor,
    placement: MatrixShard,
    rank: int,
    *,
    group=None,
    divide_by_world: bool = True,
) -> torch.Tensor:
    return reduce_scatter_matrix_1d(
        full_tensor,
        placement.shard_lengths(full_tensor.numel()),
        rank,
        group=group,
        divide_by_world=divide_by_world,
    )
