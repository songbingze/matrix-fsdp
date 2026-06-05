from __future__ import annotations

import atexit
from dataclasses import dataclass
from functools import lru_cache
import importlib
import os
from types import ModuleType

import torch
import torch.distributed as dist

from matrix_fsdp.core.layout import LayoutSegment
from matrix_fsdp.kernels.segment_fusion import coalesce_rank_segments

_NATIVE_MODULE_NAME = "matrix_fsdp._matrix_fsdp_cuda"
_DISABLE_NATIVE_ENV = "MATRIX_FSDP_DISABLE_NATIVE_KERNELS"
_ENABLE_NATIVE_COPY_ENV = "MATRIX_FSDP_ENABLE_NATIVE_COPY_KERNELS"
_ENABLE_NATIVE_NCCL_ENV = "MATRIX_FSDP_ENABLE_NATIVE_NCCL"
_SEGMENT_METADATA_CACHE: dict[
    tuple[str, int | None, tuple[tuple[int, int, int], ...]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
] = {}
_RANK_SEGMENT_METADATA_CACHE: dict[
    tuple[str, int | None, tuple[tuple[int, int, int, int], ...]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}
_SHARD_SIZE_METADATA_CACHE: dict[tuple[int, ...], torch.Tensor] = {}
_DEFAULT_NCCL_COMM_LANE = "default"
PARAM_ALL_GATHER_COMM_LANE = "param_all_gather"
GRAD_REDUCE_COMM_LANE = "grad_reduce"
_NCCL_COMM_LANE_IDS = {
    _DEFAULT_NCCL_COMM_LANE: 0,
    PARAM_ALL_GATHER_COMM_LANE: 1,
    GRAD_REDUCE_COMM_LANE: 2,
}
_NCCL_COMM_KEYS: dict[int, tuple[int | None, int, int, int | None]] = {}


@dataclass(frozen=True)
class NativeKernelStatus:
    available: bool
    module_name: str = _NATIVE_MODULE_NAME
    reason: str | None = None


@lru_cache(maxsize=1)
def native_kernel_status() -> NativeKernelStatus:
    if os.environ.get(_DISABLE_NATIVE_ENV, "").lower() in {"1", "true", "yes", "on"}:
        return NativeKernelStatus(False, reason=f"{_DISABLE_NATIVE_ENV} is set")
    try:
        importlib.import_module(_NATIVE_MODULE_NAME)
    except Exception as exc:  # pragma: no cover - depends on optional CUDA build.
        return NativeKernelStatus(False, reason=f"{type(exc).__name__}: {exc}")
    return NativeKernelStatus(True)


def native_kernel_available() -> bool:
    return native_kernel_status().available


def native_copy_rank_segments_to_full(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    segments: tuple[LayoutSegment, ...],
) -> bool:
    if not native_copy_kernels_enabled():
        return False
    module = _load_native_kernel()
    if module is None or not _can_use_native_copy(local_tensor, output_tensor):
        return False
    if not segments:
        return True
    global_starts, local_starts, numels, max_numel = _segment_metadata_tensors(segments, device=local_tensor.device)
    module.copy_rank_segments_to_full(local_tensor, output_tensor, global_starts, local_starts, numels, max_numel)
    return True


def native_copy_rank_chunk_from_packed(
    packed_rank_chunks: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    compact: bool,
) -> bool:
    if not native_copy_kernels_enabled():
        return False
    module = _load_native_kernel()
    if module is None or not _can_use_native_copy(packed_rank_chunks, output_tensor):
        return False
    offset = sum(shard_sizes[:rank]) if compact else rank * max(shard_sizes)
    module.copy_range(packed_rank_chunks, output_tensor, offset)
    return True


def _load_native_kernel() -> ModuleType | None:
    if not native_kernel_available():
        return None
    return importlib.import_module(_NATIVE_MODULE_NAME)


def native_copy_kernels_enabled() -> bool:
    return os.environ.get(_ENABLE_NATIVE_COPY_ENV, "").lower() in {"1", "true", "yes", "on"}


def native_nccl_collectives_enabled() -> bool:
    return os.environ.get(_ENABLE_NATIVE_NCCL_ENV, "").lower() in {
        "1",
        "true",
        "yes",
        "on",
        "group_broadcast",
        "unsafe_group_broadcast",
    }


def native_nccl_comm_lanes_available() -> bool:
    module = _load_native_kernel()
    return module is not None and _native_nccl_comm_lanes_supported(module)


def native_group_broadcast_rank_segments(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    force: bool = False,
    comm_lane: str = PARAM_ALL_GATHER_COMM_LANE,
) -> torch.cuda.Event | bool:
    if not force and not native_nccl_collectives_enabled():
        return False
    module = _load_native_kernel()
    if module is None or not _can_use_native_copy(local_tensor, output_tensor):
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    lane_id = _ensure_nccl_comm(module, rank, len(rank_segments), group, local_tensor.device, comm_lane=comm_lane)
    fused_rank_segments = coalesce_rank_segments(rank_segments)
    src_ranks, global_starts, local_starts, numels = _rank_segment_metadata_tensors(
        fused_rank_segments,
        device=local_tensor.device,
    )

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(local_tensor.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            _call_native_nccl_collective(
                module,
                "group_broadcast_rank_segments",
                local_tensor,
                output_tensor,
                src_ranks,
                global_starts,
                local_starts,
                numels,
                rank,
                lane_id=lane_id,
            )
            event = torch.cuda.Event()
            event.record(cuda_stream)
        return event

    _call_native_nccl_collective(
        module,
        "group_broadcast_rank_segments",
        local_tensor,
        output_tensor,
        src_ranks,
        global_starts,
        local_starts,
        numels,
        rank,
        lane_id=lane_id,
    )
    return True


def native_sendrecv_rank_segments(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    force: bool = False,
    comm_lane: str = PARAM_ALL_GATHER_COMM_LANE,
) -> torch.cuda.Event | bool:
    if not force and not native_nccl_collectives_enabled():
        return False
    module = _load_native_kernel()
    if module is None or not _can_use_native_copy(local_tensor, output_tensor):
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    lane_id = _ensure_nccl_comm(module, rank, len(rank_segments), group, local_tensor.device, comm_lane=comm_lane)
    fused_rank_segments = coalesce_rank_segments(rank_segments)
    src_ranks, global_starts, local_starts, numels = _rank_segment_metadata_tensors(
        fused_rank_segments,
        device=local_tensor.device,
    )

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(local_tensor.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            _call_native_nccl_collective(
                module,
                "sendrecv_rank_segments",
                local_tensor,
                output_tensor,
                src_ranks,
                global_starts,
                local_starts,
                numels,
                rank,
                lane_id=lane_id,
            )
            event = torch.cuda.Event()
            event.record(cuda_stream)
        return event

    _call_native_nccl_collective(
        module,
        "sendrecv_rank_segments",
        local_tensor,
        output_tensor,
        src_ranks,
        global_starts,
        local_starts,
        numels,
        rank,
        lane_id=lane_id,
    )
    return True


def native_sendrecv_rank_chunks(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    force: bool = False,
    comm_lane: str = PARAM_ALL_GATHER_COMM_LANE,
) -> torch.cuda.Event | bool:
    if not force and not native_nccl_collectives_enabled():
        return False
    module = _load_native_kernel()
    if module is None or not _can_use_native_copy(local_tensor, output_tensor):
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    lane_id = _ensure_nccl_comm(module, rank, len(shard_sizes), group, local_tensor.device, comm_lane=comm_lane)
    shard_size_tensor = _shard_size_metadata_tensor(shard_sizes)

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(local_tensor.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            _call_native_nccl_collective(
                module,
                "sendrecv_rank_chunks",
                local_tensor,
                output_tensor,
                shard_size_tensor,
                rank,
                lane_id=lane_id,
            )
            event = torch.cuda.Event()
            event.record(cuda_stream)
        return event

    _call_native_nccl_collective(
        module,
        "sendrecv_rank_chunks",
        local_tensor,
        output_tensor,
        shard_size_tensor,
        rank,
        lane_id=lane_id,
    )
    return True


def native_reduce_rank_chunks(
    packed_rank_chunks: torch.Tensor,
    local_output: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    divide_by_world: bool = True,
    compact: bool,
    cuda_stream: torch.cuda.Stream | None = None,
    force: bool = False,
    comm_lane: str = GRAD_REDUCE_COMM_LANE,
) -> torch.cuda.Event | bool:
    if not force and not native_nccl_collectives_enabled():
        return False
    module = _load_native_kernel()
    if module is None or not _can_use_native_copy(packed_rank_chunks, local_output):
        return False
    if not hasattr(module, "reduce_rank_chunks"):
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    lane_id = _ensure_nccl_comm(module, rank, len(shard_sizes), group, packed_rank_chunks.device, comm_lane=comm_lane)
    shard_size_tensor = _shard_size_metadata_tensor(shard_sizes)

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(packed_rank_chunks.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            _call_native_nccl_collective(
                module,
                "reduce_rank_chunks",
                packed_rank_chunks,
                local_output,
                shard_size_tensor,
                rank,
                compact,
                lane_id=lane_id,
            )
            if divide_by_world:
                local_output.div_(len(shard_sizes))
            event = torch.cuda.Event()
            event.record(cuda_stream)
        return event

    _call_native_nccl_collective(
        module,
        "reduce_rank_chunks",
        packed_rank_chunks,
        local_output,
        shard_size_tensor,
        rank,
        compact,
        lane_id=lane_id,
    )
    if divide_by_world:
        local_output.div_(len(shard_sizes))
    return True


def _can_use_native_copy(source: torch.Tensor, destination: torch.Tensor) -> bool:
    return (
        source.is_cuda
        and destination.is_cuda
        and source.is_contiguous()
        and destination.is_contiguous()
        and source.dtype == destination.dtype
    )


def _ensure_nccl_comm(
    module: ModuleType,
    rank: int,
    world_size: int,
    group,
    device: torch.device,
    *,
    comm_lane: str = _DEFAULT_NCCL_COMM_LANE,
) -> int:
    lane_id = _native_nccl_comm_lane_id(module, comm_lane)
    key = (id(group) if group is not None else None, rank, world_size, device.index)
    if key == _NCCL_COMM_KEYS.get(lane_id):
        return lane_id
    if rank == 0:
        unique_id = module.get_nccl_unique_id()
    else:
        unique_id = None
    object_list = [unique_id]
    src = dist.get_global_rank(group, 0) if group is not None and hasattr(dist, "get_global_rank") else 0
    dist.broadcast_object_list(object_list, src=src, group=group)
    with torch.cuda.device(device):
        if _native_nccl_comm_lanes_supported(module):
            module.init_nccl_comm(object_list[0], rank, world_size, lane_id)
        else:
            module.init_nccl_comm(object_list[0], rank, world_size)
            lane_id = 0
    _NCCL_COMM_KEYS[lane_id] = key
    return lane_id


def destroy_native_nccl_comms() -> None:
    if not _NCCL_COMM_KEYS:
        return
    module = _load_native_kernel()
    if module is not None:
        module.destroy_nccl_comm()
    _NCCL_COMM_KEYS.clear()


def _native_nccl_comm_lanes_supported(module: ModuleType) -> bool:
    probe = getattr(module, "nccl_comm_lanes_supported", None)
    if probe is None:
        return False
    return bool(probe())


def _native_nccl_comm_lane_id(module: ModuleType, comm_lane: str) -> int:
    if not _native_nccl_comm_lanes_supported(module):
        return 0
    try:
        return _NCCL_COMM_LANE_IDS[comm_lane]
    except KeyError as exc:
        valid = ", ".join(sorted(_NCCL_COMM_LANE_IDS))
        raise ValueError(f"Unknown native NCCL communicator lane {comm_lane!r}; expected one of {valid}.") from exc


def _call_native_nccl_collective(module: ModuleType, name: str, *args, lane_id: int) -> None:
    func = getattr(module, name)
    if _native_nccl_comm_lanes_supported(module):
        func(*args, lane_id)
    else:
        func(*args)


def _copy_local_segments_to_output(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    segments: tuple[LayoutSegment, ...],
) -> None:
    for segment in segments:
        output_tensor[segment.global_start : segment.global_end].copy_(
            local_tensor[segment.local_start : segment.local_end]
        )


def _rank_segment_metadata_tensors(
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flattened = tuple(
        (rank, segment.global_start, segment.local_start, segment.numel)
        for rank, segments in enumerate(rank_segments)
        for segment in segments
    )
    key = (device.type, device.index, flattened)
    cached = _RANK_SEGMENT_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    src_ranks = torch.tensor([item[0] for item in flattened], dtype=torch.int64)
    global_starts = torch.tensor([item[1] for item in flattened], dtype=torch.int64)
    local_starts = torch.tensor([item[2] for item in flattened], dtype=torch.int64)
    numels = torch.tensor([item[3] for item in flattened], dtype=torch.int64)
    metadata = (src_ranks, global_starts, local_starts, numels)
    _RANK_SEGMENT_METADATA_CACHE[key] = metadata
    return metadata


def _shard_size_metadata_tensor(shard_sizes: tuple[int, ...]) -> torch.Tensor:
    cached = _SHARD_SIZE_METADATA_CACHE.get(shard_sizes)
    if cached is not None:
        return cached
    tensor = torch.tensor(shard_sizes, dtype=torch.int64)
    _SHARD_SIZE_METADATA_CACHE[shard_sizes] = tensor
    return tensor


def _segment_metadata_tensors(
    segments: tuple[LayoutSegment, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    key = (
        device.type,
        device.index,
        tuple((segment.global_start, segment.local_start, segment.numel) for segment in segments),
    )
    cached = _SEGMENT_METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    global_starts = torch.tensor([segment.global_start for segment in segments], dtype=torch.int64, device=device)
    local_starts = torch.tensor([segment.local_start for segment in segments], dtype=torch.int64, device=device)
    numels = torch.tensor([segment.numel for segment in segments], dtype=torch.int64, device=device)
    max_numel = max((segment.numel for segment in segments), default=0)
    metadata = (global_starts, local_starts, numels, max_numel)
    _SEGMENT_METADATA_CACHE[key] = metadata
    return metadata


atexit.register(destroy_native_nccl_comms)
