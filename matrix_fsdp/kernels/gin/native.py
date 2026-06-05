from __future__ import annotations

import atexit
from dataclasses import dataclass
from functools import lru_cache
import importlib
import os
from types import ModuleType

import torch
import torch.distributed as dist

_GIN_MODULE_NAME = "matrix_fsdp._matrix_fsdp_gin_cuda"
_DISABLE_GIN_ENV = "MATRIX_FSDP_DISABLE_GIN_KERNELS"
_ENABLE_GIN_ENV = "MATRIX_FSDP_ENABLE_GIN_KERNELS"
_COMM_KEY: tuple[int | None, int, int, int | None] | None = None
_SHARD_SIZE_METADATA_CACHE: dict[tuple[int, ...], torch.Tensor] = {}


@dataclass(frozen=True)
class GinNativeStatus:
    available: bool
    module_name: str = _GIN_MODULE_NAME
    reason: str | None = None


@lru_cache(maxsize=1)
def gin_native_status() -> GinNativeStatus:
    if os.environ.get(_DISABLE_GIN_ENV, "").lower() in {"1", "true", "yes", "on"}:
        return GinNativeStatus(False, reason=f"{_DISABLE_GIN_ENV} is set")
    try:
        importlib.import_module(_GIN_MODULE_NAME)
    except Exception as exc:  # pragma: no cover - depends on optional CUDA/NCCL build.
        return GinNativeStatus(False, reason=f"{type(exc).__name__}: {exc}")
    return GinNativeStatus(True)


def gin_native_available() -> bool:
    return gin_native_status().available


def gin_kernels_enabled() -> bool:
    return os.environ.get(_ENABLE_GIN_ENV, "").lower() in {"1", "true", "yes", "on", "rma", "gin"}


def gin_backend_info() -> dict[str, object]:
    module = _load_gin_native_module()
    if module is None:
        status = gin_native_status()
        return {
            "available": False,
            "reason": status.reason,
        }
    info = dict(module.backend_info())
    info["available"] = True
    return info


def rma_putsignal_rank_chunks(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    force: bool = False,
) -> torch.cuda.Event | bool:
    return _call_rank_chunk_backend(
        "rma_putsignal_rank_chunks",
        local_tensor,
        output_tensor,
        shard_sizes,
        rank,
        group=group,
        cuda_stream=cuda_stream,
        force=force,
    )


def gin_device_rank_chunks(
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None = None,
    force: bool = False,
) -> torch.cuda.Event | bool:
    return _call_rank_chunk_backend(
        "gin_device_rank_chunks",
        local_tensor,
        output_tensor,
        shard_sizes,
        rank,
        group=group,
        cuda_stream=cuda_stream,
        force=force,
    )


def _call_rank_chunk_backend(
    function_name: str,
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    rank: int,
    *,
    group=None,
    cuda_stream: torch.cuda.Stream | None,
    force: bool,
) -> torch.cuda.Event | bool:
    if not force and not gin_kernels_enabled():
        return False
    module = _load_gin_native_module()
    if module is None or not _can_use_gin_native(local_tensor, output_tensor):
        return False
    if not dist.is_available() or not dist.is_initialized():
        return False
    _ensure_comm(module, rank, len(shard_sizes), group, local_tensor.device)
    shard_size_tensor = _shard_size_metadata_tensor(shard_sizes)
    function = getattr(module, function_name)

    if cuda_stream is not None:
        current_stream = torch.cuda.current_stream(local_tensor.device)
        cuda_stream.wait_stream(current_stream)
        with torch.cuda.stream(cuda_stream):
            used_backend = function(local_tensor, output_tensor, shard_size_tensor, rank)
            if not used_backend:
                return False
            event = torch.cuda.Event()
            event.record(cuda_stream)
        return event

    return bool(function(local_tensor, output_tensor, shard_size_tensor, rank))


def _load_gin_native_module() -> ModuleType | None:
    if not gin_native_available():
        return None
    return importlib.import_module(_GIN_MODULE_NAME)


def _can_use_gin_native(source: torch.Tensor, destination: torch.Tensor) -> bool:
    return (
        source.is_cuda
        and destination.is_cuda
        and source.is_contiguous()
        and destination.is_contiguous()
        and source.dtype == destination.dtype
    )


def _ensure_comm(module: ModuleType, rank: int, world_size: int, group, device: torch.device) -> None:
    global _COMM_KEY
    key = (id(group) if group is not None else None, rank, world_size, device.index)
    if key == _COMM_KEY:
        return
    unique_id = module.get_nccl_unique_id() if rank == 0 else None
    object_list = [unique_id]
    src = dist.get_global_rank(group, 0) if group is not None and hasattr(dist, "get_global_rank") else 0
    dist.broadcast_object_list(object_list, src=src, group=group)
    with torch.cuda.device(device):
        module.init_nccl_comm(object_list[0], rank, world_size)
    _COMM_KEY = key


def destroy_gin_native_comms() -> None:
    global _COMM_KEY
    if _COMM_KEY is None:
        return
    module = _load_gin_native_module()
    if module is not None:
        module.destroy_nccl_comm()
    _COMM_KEY = None


def _shard_size_metadata_tensor(shard_sizes: tuple[int, ...]) -> torch.Tensor:
    cached = _SHARD_SIZE_METADATA_CACHE.get(shard_sizes)
    if cached is not None:
        return cached
    tensor = torch.tensor(shard_sizes, dtype=torch.int64)
    _SHARD_SIZE_METADATA_CACHE[shard_sizes] = tensor
    return tensor


atexit.register(destroy_gin_native_comms)
