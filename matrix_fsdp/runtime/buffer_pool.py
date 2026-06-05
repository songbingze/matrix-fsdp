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


@dataclass(frozen=True)
class FullParamBufferAcquire:
    tensor: torch.Tensor
    reused: bool


@dataclass(frozen=True)
class FullParamBufferRelease:
    kind: str
    numel: int = 0
    bytes: int = 0


@dataclass(frozen=True)
class _PendingFullParamBufferRelease:
    key: FullParamBufferKey
    tensor: torch.Tensor
    event: Any


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
        self._pending: list[_PendingFullParamBufferRelease] = []
        self._lock = Lock()
        self.allocations = 0
        self.reuses = 0
        self.releases = 0
        self.pending_releases = 0
        self.drops = 0

    def acquire(self, reference: torch.Tensor, numel: int) -> torch.Tensor:
        return self.acquire_with_stats(reference, numel).tensor

    def acquire_with_stats(self, reference: torch.Tensor, numel: int) -> FullParamBufferAcquire:
        key = self._key(reference, numel)
        with self._lock:
            self._drain_pending_locked()
            bucket = self._cached.get(key)
            if bucket:
                self.reuses += 1
                return FullParamBufferAcquire(bucket.pop(), reused=True)
            self.allocations += 1
        return FullParamBufferAcquire(reference.new_empty(numel), reused=False)

    def can_cache(self, buffer: torch.Tensor | None) -> bool:
        if buffer is None or self.max_cached_per_key == 0:
            return False
        if buffer.ndim != 1:
            buffer = buffer.reshape(-1)
        key = self._key(buffer, buffer.numel())
        with self._lock:
            self._drain_pending_locked()
            return self._cached_count_for_key_locked(key) + self._pending_count_for_key_locked(key) < self.max_cached_per_key

    def release(
        self,
        buffer: torch.Tensor | None,
        *,
        cuda_stream: torch.cuda.Stream | None = None,
        cuda_event: Any | None = None,
    ) -> FullParamBufferRelease:
        if buffer is None:
            return FullParamBufferRelease(kind="none")
        if buffer.ndim != 1:
            buffer = buffer.reshape(-1)
        key = self._key(buffer, buffer.numel())
        detached = buffer.detach()
        release_numel = int(detached.numel())
        release_bytes = _buffer_key_bytes(key)
        if cuda_event is None and detached.is_cuda and self.max_cached_per_key > 0:
            if cuda_stream is not None:
                cuda_event = torch.cuda.Event()
                cuda_event.record(cuda_stream)
            else:
                cuda_event = torch.cuda.Event()
                cuda_event.record(torch.cuda.current_stream(detached.device))
        with self._lock:
            self._drain_pending_locked()
            if self._cached_count_for_key_locked(key) + self._pending_count_for_key_locked(key) >= self.max_cached_per_key:
                self.drops += 1
                return FullParamBufferRelease(kind="dropped", numel=release_numel, bytes=release_bytes)
            if cuda_event is not None and not self._event_complete(cuda_event):
                self._pending.append(_PendingFullParamBufferRelease(key=key, tensor=detached, event=cuda_event))
                self.pending_releases += 1
                return FullParamBufferRelease(kind="pending_event", numel=release_numel, bytes=release_bytes)
            if self._cache_locked(key, detached):
                return FullParamBufferRelease(kind="cached", numel=release_numel, bytes=release_bytes)
            return FullParamBufferRelease(kind="dropped", numel=release_numel, bytes=release_bytes)

    def clear(self) -> None:
        with self._lock:
            self._cached.clear()
            self._pending.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._drain_pending_locked()
            cached_buffers = sum(len(bucket) for bucket in self._cached.values())
            cached_numel = sum(key.numel * len(bucket) for key, bucket in self._cached.items())
            cached_bytes = sum(_buffer_key_bytes(key) * len(bucket) for key, bucket in self._cached.items())
            pending_buffers = len(self._pending)
            pending_numel = sum(item.key.numel for item in self._pending)
            pending_bytes = sum(_buffer_key_bytes(item.key) for item in self._pending)
            keys = len(self._cached)
        return {
            "allocations": self.allocations,
            "reuses": self.reuses,
            "releases": self.releases,
            "pending_releases": self.pending_releases,
            "drops": self.drops,
            "cached_buffers": cached_buffers,
            "cached_numel": cached_numel,
            "cached_bytes": cached_bytes,
            "pending_buffers": pending_buffers,
            "pending_numel": pending_numel,
            "pending_bytes": pending_bytes,
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

    def _cache_locked(self, key: FullParamBufferKey, buffer: torch.Tensor) -> bool:
        bucket = self._cached.setdefault(key, [])
        if len(bucket) >= self.max_cached_per_key:
            self.drops += 1
            return False
        bucket.append(buffer)
        self.releases += 1
        return True

    def _drain_pending_locked(self) -> None:
        if not self._pending:
            return
        still_pending: list[_PendingFullParamBufferRelease] = []
        for pending in self._pending:
            if self._event_complete(pending.event):
                self._cache_locked(pending.key, pending.tensor)
            else:
                still_pending.append(pending)
        self._pending = still_pending

    def _cached_count_for_key_locked(self, key: FullParamBufferKey) -> int:
        return len(self._cached.get(key, ()))

    def _pending_count_for_key_locked(self, key: FullParamBufferKey) -> int:
        return sum(1 for pending in self._pending if pending.key == key)

    @staticmethod
    def _event_complete(event: Any) -> bool:
        query = getattr(event, "query", None)
        if query is None:
            return True
        return bool(query())


def _buffer_key_bytes(key: FullParamBufferKey) -> int:
    return int(key.numel) * int(torch.empty((), dtype=key.dtype).element_size())


def clear_global_full_param_buffer_pool() -> None:
    """Backward-compatible no-op.

    Full-param buffer pools are now owned by each scheduler/runtime instance
    instead of a module-level singleton. This function remains so older tests or
    scripts that clear global runtime state do not fail.
    """
