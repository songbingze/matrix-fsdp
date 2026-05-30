from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from matrix_fsdp.core.layout import LayoutSegment
from matrix_fsdp.core.managed_param import ManagedParam

CopyInBackend = Literal["auto", "flat_cat", "foreach_copy", "chunk_cat", "segment_copy"]
CopyInLayoutKind = Literal["flat_contiguous", "fsdp2_chunk", "generic_segment"]


@dataclass(frozen=True)
class BucketParamGrad:
    managed_param: ManagedParam
    grad: torch.Tensor


@dataclass(frozen=True)
class MatrixGradBucket:
    param_grads: tuple[BucketParamGrad, ...]
    total_numel: int
    shard_sizes: tuple[int, ...]
    rank_segments: tuple[tuple[LayoutSegment, ...], ...]
    packed_input: torch.Tensor | None = None
    packed_input_is_compact: bool = False

    @property
    def world_size(self) -> int:
        return len(self.shard_sizes)

    @property
    def max_shard_size(self) -> int:
        return max(self.shard_sizes, default=0)

    @property
    def has_grads(self) -> bool:
        return bool(self.param_grads) or self.packed_input is not None


def build_reduce_scatter_input(
    bucket: MatrixGradBucket,
    reference: torch.Tensor,
    *,
    backend: CopyInBackend = "auto",
) -> torch.Tensor:
    """
    Packs full parameter gradients into rank-major reduce-scatter chunks.

    The output layout is ``[rank0_padded_chunk, rank1_padded_chunk, ...]``.
    Each rank chunk follows the runtime matrix layout's local offsets and is
    padded to the maximum local shard size so that ``reduce_scatter_tensor`` can
    consume equal-sized chunks.
    """
    if bucket.world_size == 0:
        raise ValueError("MatrixGradBucket requires at least one rank.")
    if bucket.packed_input is not None:
        return bucket.packed_input
    packed = reference.new_empty(bucket.world_size * bucket.max_shard_size)
    return fill_reduce_scatter_input(bucket, packed, backend=backend)


def new_reduce_scatter_input(bucket: MatrixGradBucket, reference: torch.Tensor) -> torch.Tensor:
    if bucket.world_size == 0:
        raise ValueError("MatrixGradBucket requires at least one rank.")
    return reference.new_empty(bucket.world_size * bucket.max_shard_size)


def fill_reduce_scatter_input(
    bucket: MatrixGradBucket,
    packed: torch.Tensor,
    *,
    backend: CopyInBackend = "auto",
) -> torch.Tensor:
    if bucket.world_size == 0:
        raise ValueError("MatrixGradBucket requires at least one rank.")
    expected_numel = bucket.world_size * bucket.max_shard_size
    if packed.ndim != 1:
        raise ValueError(f"packed must be 1D, got shape {tuple(packed.shape)}.")
    if packed.numel() != expected_numel:
        raise ValueError(f"packed has {packed.numel()} elements, expected {expected_numel}.")
    if backend not in ("auto", "flat_cat", "foreach_copy", "chunk_cat", "segment_copy"):
        raise ValueError("backend must be 'auto', 'flat_cat', 'foreach_copy', 'chunk_cat', or 'segment_copy'.")
    layout_kind = classify_copy_in_layout(bucket)
    can_use_flat_fast_path = layout_kind == "flat_contiguous"
    can_use_chunk_cat_fast_path = layout_kind == "fsdp2_chunk"
    if backend in ("flat_cat", "foreach_copy") and not can_use_flat_fast_path:
        raise ValueError(f"{backend} backend requires an equal, rank-contiguous flat bucket layout.")
    if backend == "chunk_cat" and not can_use_chunk_cat_fast_path:
        raise ValueError("chunk_cat backend requires an FSDP2-style per-parameter chunk bucket layout.")
    if backend == "foreach_copy" and not hasattr(torch, "_foreach_copy_"):
        raise RuntimeError("foreach_copy backend requires torch._foreach_copy_.")
    if backend == "chunk_cat" and not hasattr(torch, "_chunk_cat"):
        raise RuntimeError("chunk_cat backend requires torch._chunk_cat.")
    if backend == "foreach_copy":
        _foreach_copy_flat_grads(bucket, packed)
        return packed
    if backend == "flat_cat" or (backend == "auto" and can_use_flat_fast_path):
        torch.cat([param_grad.grad.reshape(-1) for param_grad in bucket.param_grads], out=packed)
        return packed
    if backend == "chunk_cat" or (backend == "auto" and can_use_chunk_cat_fast_path):
        _chunk_cat_param_grads(bucket, packed)
        return packed
    packed.zero_()
    packed_2d = packed.view(bucket.world_size, bucket.max_shard_size)
    for param_grad in bucket.param_grads:
        mp = param_grad.managed_param
        grad = param_grad.grad.reshape(-1)
        if grad.numel() != mp.numel:
            raise RuntimeError(
                f"Gradient for {mp.fqn} has {grad.numel()} elements, expected {mp.numel}."
            )
        _copy_param_grad_to_rank_chunks(packed_2d, mp, grad, bucket.rank_segments)
    return packed


def classify_copy_in_layout(bucket: MatrixGradBucket) -> CopyInLayoutKind:
    if _can_use_flat_cat_fast_path(bucket):
        return "flat_contiguous"
    if _can_use_chunk_cat_fast_path(bucket):
        return "fsdp2_chunk"
    return "generic_segment"


def _can_use_flat_cat_fast_path(bucket: MatrixGradBucket) -> bool:
    if not bucket.param_grads:
        return False
    if len(set(bucket.shard_sizes)) != 1:
        return False

    cursor = 0
    for shard_size, segments in zip(bucket.shard_sizes, bucket.rank_segments):
        if len(segments) != 1:
            return False
        segment = segments[0]
        if segment.local_start != 0:
            return False
        if segment.global_start != cursor or segment.global_end != cursor + shard_size:
            return False
        cursor = segment.global_end
    if cursor != bucket.total_numel:
        return False

    cursor = 0
    for param_grad in bucket.param_grads:
        mp = param_grad.managed_param
        grad = param_grad.grad.reshape(-1)
        if mp.offset != cursor or mp.end != cursor + mp.numel:
            return False
        if grad.numel() != mp.numel:
            return False
        cursor = mp.end
    return cursor == bucket.total_numel


def _can_use_chunk_cat_fast_path(bucket: MatrixGradBucket) -> bool:
    if not bucket.param_grads or bucket.world_size == 0:
        return False

    chunk_offsets = []
    local_cursor = 0
    for param_grad in bucket.param_grads:
        mp = param_grad.managed_param
        grad = param_grad.grad.reshape(-1)
        if grad.numel() != mp.numel or mp.end != mp.offset + mp.numel:
            return False
        padded_chunk_size = _ceil_div(mp.numel, bucket.world_size)
        chunk_offsets.append((mp, local_cursor, padded_chunk_size))
        local_cursor += padded_chunk_size

    if bucket.max_shard_size != local_cursor:
        return False
    if any(shard_size != local_cursor for shard_size in bucket.shard_sizes):
        return False

    expected_rank_segments = []
    for rank in range(bucket.world_size):
        rank_segments = []
        for mp, local_start, padded_chunk_size in chunk_offsets:
            global_start = mp.offset + rank * padded_chunk_size
            global_end = min(mp.end, global_start + padded_chunk_size)
            if global_start < global_end:
                rank_segments.append(LayoutSegment(global_start, global_end, local_start))
        expected_rank_segments.append(tuple(rank_segments))
    return bucket.rank_segments == tuple(expected_rank_segments)


def _foreach_copy_flat_grads(bucket: MatrixGradBucket, packed: torch.Tensor) -> None:
    dst_views = []
    src_views = []
    for param_grad in bucket.param_grads:
        mp = param_grad.managed_param
        dst_views.append(packed[mp.offset : mp.end])
        src_views.append(param_grad.grad.reshape(-1))
    torch._foreach_copy_(dst_views, src_views)


def _chunk_cat_param_grads(bucket: MatrixGradBucket, packed: torch.Tensor) -> None:
    out = packed.view(bucket.world_size, bucket.max_shard_size)
    torch._chunk_cat([param_grad.grad.reshape(-1) for param_grad in bucket.param_grads], 0, bucket.world_size, out=out)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _copy_param_grad_to_rank_chunks(
    packed: torch.Tensor,
    managed_param: ManagedParam,
    grad: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
) -> None:
    for rank, segments in enumerate(rank_segments):
        for segment in segments:
            global_start = max(managed_param.offset, segment.global_start)
            global_end = min(managed_param.end, segment.global_end)
            if global_start >= global_end:
                continue
            src_start = global_start - managed_param.offset
            src_end = global_end - managed_param.offset
            dst_start = segment.local_start + global_start - segment.global_start
            dst_end = dst_start + (global_end - global_start)
            packed[rank, dst_start:dst_end].copy_(grad[src_start:src_end])
