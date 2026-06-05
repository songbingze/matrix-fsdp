from __future__ import annotations

import torch


class CommWorkspaceLease:
    def __init__(self, workspace: "CommWorkspaceCache", key: tuple[object, ...], tensor: torch.Tensor) -> None:
        self.workspace = workspace
        self.key = key
        self.tensor = tensor
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.workspace.release(self)
        self.released = True


class CommWorkspaceCache:
    def __init__(self, *, max_cached_per_key: int = 1) -> None:
        if max_cached_per_key < 0:
            raise ValueError("max_cached_per_key must be non-negative.")
        self._entries: dict[tuple[object, ...], list[dict[str, object]]] = {}
        self.max_cached_per_key = max_cached_per_key
        self.acquire_count = 0
        self.reuse_count = 0
        self.allocate_count = 0

    def acquire(self, reference: torch.Tensor, numel: int) -> CommWorkspaceLease:
        if numel < 0:
            raise ValueError(f"numel must be non-negative, got {numel}.")
        key = _workspace_key(reference, numel)
        entries = self._entries.setdefault(key, [])
        self.acquire_count += 1
        for entry in entries:
            if not entry["in_use"]:
                entry["in_use"] = True
                self.reuse_count += 1
                return CommWorkspaceLease(self, key, entry["tensor"])  # type: ignore[arg-type]
        tensor = reference.new_empty(numel)
        entries.append({"tensor": tensor, "in_use": True})
        self.allocate_count += 1
        return CommWorkspaceLease(self, key, tensor)

    def release(self, lease: CommWorkspaceLease) -> None:
        entries = self._entries.get(lease.key, ())
        for index, entry in enumerate(entries):
            if entry["tensor"] is lease.tensor:
                if self.max_cached_per_key == 0:
                    entries.pop(index)
                    if not entries:
                        self._entries.pop(lease.key, None)
                    return
                entry["in_use"] = False
                self._trim_idle_entries(lease.key)
                return
        raise RuntimeError("Attempted to release a workspace tensor that is not owned by this workspace.")

    def set_max_cached_per_key(self, max_cached_per_key: int) -> None:
        if max_cached_per_key < 0:
            raise ValueError("max_cached_per_key must be non-negative.")
        self.max_cached_per_key = max_cached_per_key
        for key in tuple(self._entries):
            self._trim_idle_entries(key)

    def stats(self) -> dict[str, object]:
        allocated_numel = 0
        in_use_numel = 0
        allocated_tensors = 0
        in_use_tensors = 0
        for entries in self._entries.values():
            for entry in entries:
                tensor = entry["tensor"]
                allocated_tensors += 1
                allocated_numel += tensor.numel()  # type: ignore[union-attr]
                if entry["in_use"]:
                    in_use_tensors += 1
                    in_use_numel += tensor.numel()  # type: ignore[union-attr]
        return {
            "workspace_acquire_count": self.acquire_count,
            "workspace_reuse_count": self.reuse_count,
            "workspace_allocate_count": self.allocate_count,
            "workspace_max_cached_per_key": self.max_cached_per_key,
            "workspace_allocated_tensors": allocated_tensors,
            "workspace_in_use_tensors": in_use_tensors,
            "workspace_allocated_numel": allocated_numel,
            "workspace_in_use_numel": in_use_numel,
        }

    def clear(self) -> None:
        self._entries.clear()

    def _trim_idle_entries(self, key: tuple[object, ...]) -> None:
        entries = self._entries.get(key)
        if not entries:
            return
        idle_indices = [index for index, entry in enumerate(entries) if not entry["in_use"]]
        overflow = len(idle_indices) - self.max_cached_per_key
        for index in reversed(idle_indices[: max(overflow, 0)]):
            entries.pop(index)
        if not entries:
            self._entries.pop(key, None)


def _workspace_key(reference: torch.Tensor, numel: int) -> tuple[object, ...]:
    return comm_workspace_key(reference, numel)


def comm_workspace_key(reference: torch.Tensor, numel: int) -> tuple[object, ...]:
    device_index = reference.device.index if reference.device.index is not None else -1
    return (reference.device.type, device_index, reference.dtype, numel)


# Backward-compatible aliases. The cache is shared by static and elastic param
# buffers, so new code should prefer the CommWorkspace* names.
ElasticParamBufferWorkspaceLease = CommWorkspaceLease
ElasticParamBufferWorkspace = CommWorkspaceCache
