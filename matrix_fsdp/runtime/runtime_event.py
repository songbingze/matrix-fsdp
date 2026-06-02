from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from threading import Lock
from time import perf_counter_ns

from matrix_fsdp.runtime.runtime import RuntimeParamGroupId, RuntimeUnitId

_event_sequence = count()
_event_sequence_lock = Lock()


@dataclass(frozen=True)
class RuntimeEvent:
    name: str
    runtime_param_group_id: RuntimeParamGroupId
    rank: int
    lifecycle_state: str
    sequence: int
    timestamp_ns: int = field(default_factory=perf_counter_ns)
    duration_ms: float | None = None
    active_full_param_buffers: int = 0
    active_full_param_numel: int = 0
    active_full_param_bytes: int = 0
    unit_full_param_bytes: int = 0
    unit_grad_bucket_bytes: int = 0
    unit_reduce_scatter_input_bytes: int = 0
    unit_local_grad_shard_bytes: int = 0
    pending_backward_reduces: int = 0
    param_data_alias_full_buffer: bool = False
    param_data_alias_local_shard: bool = False
    collective_kind: str | None = None
    collective_backend: str | None = None
    collective_impl: str | None = None
    collective_numel: int = 0
    collective_bytes: int = 0
    collective_count: int = 0

    @property
    def runtime_unit_id(self) -> RuntimeUnitId:
        return RuntimeUnitId(self.runtime_param_group_id)


def next_runtime_event_sequence() -> int:
    with _event_sequence_lock:
        return next(_event_sequence)
