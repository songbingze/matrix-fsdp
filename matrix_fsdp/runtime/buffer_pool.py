from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch


@dataclass(frozen=True)
class FullParamBufferKey:
    device_type: str
    device_index: int | None
    dtype: torch.dtype
    numel: int


class FullParamBufferPool:
    """
    Small reusable pool for full-parameter communication buffers.

    Buffers are returned to the pool when a unit reshard/free step makes them
    inactive. The next unit with the same flattened size, dtype, and device can
    all-gather directly into the same storage, while param.data remains a view.
    """

    def __init__(self, *, max_cached_per_key: int = 0) -> None:
        if max_cached_per_key < 0:
            raise ValueError("max_cached_per_key must be non-negative.")
        self.max_cached_per_key = max_cached_per_key
        self._cached: dict[FullParamBufferKey, list[torch.Tensor]] = {}
        self._lock = Lock()
        self.allocations = 0
        self.reuses = 0
        self.releases = 0
        self.drops = 0

    def acquire(self, reference: torch.Tensor, numel: int) -> torch.Tensor:
        key = self._key(reference, numel)
        with self._lock:
            bucket = self._cached.get(key)
            if bucket:
                self.reuses += 1
                return bucket.pop()
            self.allocations += 1
        return reference.new_empty(numel)

    def can_cache(self, buffer: torch.Tensor | None) -> bool:
        if buffer is None or self.max_cached_per_key == 0:
            return False
        if buffer.ndim != 1:
            buffer = buffer.reshape(-1)
        key = self._key(buffer, buffer.numel())
        with self._lock:
            return len(self._cached.get(key, ())) < self.max_cached_per_key

    def release(self, buffer: torch.Tensor | None) -> None:
        if buffer is None:
            return
        if buffer.ndim != 1:
            buffer = buffer.reshape(-1)
        key = self._key(buffer, buffer.numel())
        detached = buffer.detach()
        with self._lock:
            bucket = self._cached.setdefault(key, [])
            if len(bucket) >= self.max_cached_per_key:
                self.drops += 1
                return
            bucket.append(detached)
            self.releases += 1

    def clear(self) -> None:
        with self._lock:
            self._cached.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            cached_buffers = sum(len(bucket) for bucket in self._cached.values())
            cached_numel = sum(key.numel * len(bucket) for key, bucket in self._cached.items())
            keys = len(self._cached)
        return {
            "allocations": self.allocations,
            "reuses": self.reuses,
            "releases": self.releases,
            "drops": self.drops,
            "cached_buffers": cached_buffers,
            "cached_numel": cached_numel,
            "keys": keys,
            "max_cached_per_key": self.max_cached_per_key,
        }

    def _key(self, reference: torch.Tensor, numel: int) -> FullParamBufferKey:
        return FullParamBufferKey(
            device_type=reference.device.type,
            device_index=reference.device.index,
            dtype=reference.dtype,
            numel=numel,
        )


def clear_global_full_param_buffer_pool() -> None:
    """Backward-compatible no-op.

    Full-param buffer pools are now owned by each scheduler/runtime instance
    instead of a module-level singleton. This function remains so older tests or
    scripts that clear global runtime state do not fail.
    """
