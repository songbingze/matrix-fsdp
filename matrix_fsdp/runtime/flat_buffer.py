from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch.distributed.device_mesh import DeviceMesh

from matrix_fsdp.runtime.collectives import (
    MatrixCollectiveHandle,
    all_reduce_full_grad,
    all_gather_equal_1d_into_async,
    all_gatherv_rank_segments_1d_into_async,
    normalize_matrix_collective_backend,
    reduce_scatter_equal_1d,
    all_gather_matrix_shard_1d_into_async,
    reduce_scatter_padded_rank_chunks_1d,
    reduce_scatter_padded_rank_chunks_1d_async,
    reduce_scatter_matrix_1d,
    reduce_scatter_matrix_shard_1d,
    reduce_scatterv_owner_rank_chunks_1d_async,
    MatrixTensorCollectiveHandle,
)
from matrix_fsdp.runtime.buffer_pool import FullParamBufferPool, FullParamBufferRelease
from matrix_fsdp.runtime.elastic_param_buffer import (
    ElasticParamBuffer,
    ElasticParamBufferLayout,
    ElasticRankChunkWorkspaceLease,
)
from matrix_fsdp.runtime.static_param_buffer import StaticParamBuffer, StaticParamBufferLayout
from matrix_fsdp.runtime.workspace_cache import CommWorkspaceCache, CommWorkspaceLease
from matrix_fsdp.runtime.grad_bucket import (
    BucketParamGrad,
    CopyInLayoutKind,
    MatrixGradBucket,
    classify_copy_in_layout,
    fill_reduce_scatter_input,
    new_reduce_scatter_input,
)
from matrix_fsdp.core.layout import LayoutSegment
from matrix_fsdp.core.managed_param import ManagedParam
from matrix_fsdp.core.mesh import MeshDim
from matrix_fsdp.core.placement import (
    PlacementCompatibility,
    MatrixShard,
    explain_matrix_shard_compatibility,
    matrix_shard_from_plan,
    shard_sizes_to_matrix_local_units,
)
from matrix_fsdp.planning.planner import ShardPlan
from matrix_fsdp.core.state import MatrixShardedState


@dataclass
class LocalParamView:
    managed_param: ManagedParam
    param_start: int
    param_end: int
    shard_start: int
    shard_end: int

    @property
    def numel(self) -> int:
        return self.param_end - self.param_start


@dataclass(frozen=True)
class GradBucketReduceStartStats:
    layout_kind: CopyInLayoutKind | str
    needs_copy_in: bool
    copy_in_ms: float
    reduce_scatter_enqueue_ms: float
    packed_numel: int
    packed_bytes: int
    workspace_kind: str | None = None
    workspace_numel: int = 0
    workspace_padding_waste_numel: int = 0
    workspace_persistent: bool = False


@dataclass(frozen=True)
class GradBucketReduceStart:
    handle: MatrixTensorCollectiveHandle
    stats: GradBucketReduceStartStats


@dataclass(frozen=True)
class FullParamClearStats:
    kind: str
    release: FullParamBufferRelease | None = None
    numel: int = 0
    bytes: int = 0


class MatrixFlatBuffer:
    def __init__(
        self,
        managed_params: list[ManagedParam],
        plan: ShardPlan,
        rank: int,
        *,
        mesh: DeviceMesh | None = None,
        dp_shard_mesh_dim: MeshDim = 0,
        replicate_group=None,
        replicate_world_size: int = 1,
        group=None,
        divide_grads_by_world: bool = True,
        cuda_comm_stream: torch.cuda.Stream | None = None,
        cuda_all_gather_stream: torch.cuda.Stream | None = None,
        cuda_reduce_scatter_stream: torch.cuda.Stream | None = None,
        full_param_buffer_pool: FullParamBufferPool | None = None,
        param_gather_strategy: str = "auto",
        matrix_collective_backend: str = "owner_broadcast",
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        collective_key: str | None = None,
    ) -> None:
        self.managed_params = managed_params
        self.plan = plan
        self.rank = rank
        self.mesh = mesh
        self.dp_shard_mesh_dim = dp_shard_mesh_dim
        self.replicate_group = replicate_group
        self.replicate_world_size = replicate_world_size
        self.group = group
        self.divide_grads_by_world = divide_grads_by_world
        self.cuda_all_gather_stream = cuda_all_gather_stream if cuda_all_gather_stream is not None else cuda_comm_stream
        self.cuda_reduce_scatter_stream = (
            cuda_reduce_scatter_stream if cuda_reduce_scatter_stream is not None else cuda_comm_stream
        )
        self.cuda_comm_stream = self.cuda_all_gather_stream
        self.full_param_buffer_pool = full_param_buffer_pool or FullParamBufferPool()
        self.reduce_dtype = reduce_dtype
        if param_gather_strategy not in ("auto", "equal_all_gather", "matrix_all_gather", "owner_broadcast"):
            raise ValueError(
                "param_gather_strategy must be 'auto', 'equal_all_gather', 'matrix_all_gather', or 'owner_broadcast'."
            )
        self.param_gather_strategy = param_gather_strategy
        self.matrix_collective_backend = normalize_matrix_collective_backend(matrix_collective_backend)
        self.param_dtype = param_dtype
        self.collective_key = collective_key
        self.max_cached_elastic_workspaces_per_key = 0
        self.matrix_shard_compatibility = explain_matrix_shard_compatibility(plan)
        if not self.matrix_shard_compatibility.compatible and not self._can_use_segment_runtime_plan(plan):
            raise ValueError(
                "MatrixFlatBuffer runtime requires a MatrixShard-compatible rank-contiguous plan. "
                "Use a planner flat reorder before constructing the runtime buffer. "
                f"Reason: {self.matrix_shard_compatibility.reason}."
            )
        self.matrix_shard = (
            matrix_shard_from_plan(plan)
            if self.matrix_shard_compatibility.compatible
            else MatrixShard(dims=(0,), local_units=shard_sizes_to_matrix_local_units(plan.shard_sizes))
        )

        self.local_start, self.local_end = self._local_range(rank)
        self.local_segments = self._local_segments(rank)
        local_shard = self._init_local_shard()
        self.param_state = self._make_sharded_state(
            "param",
            local_shard,
            self.matrix_shard,
            global_shape=(self.plan.total_numel,),
            global_stride=(1,),
        )
        self.grad_state: MatrixShardedState | None = None
        self.local_param_views = self._build_local_param_views()
        self._local_views_by_param_cache = self._build_local_views_by_param()
        self.elastic_param_buffer = ElasticParamBuffer(
            ElasticParamBufferLayout(
                total_numel=self.plan.total_numel,
                shard_sizes=self.shard_sizes,
                rank_segments=self.plan.rank_segments,
            )
        )
        self.static_param_buffer = StaticParamBuffer(
            StaticParamBufferLayout(
                total_numel=self.plan.total_numel,
                shard_sizes=self.shard_sizes,
                rank_segments=self.plan.rank_segments,
            )
        )
        self.full_buffer: torch.Tensor | None = None
        self.full_grad_buffer: torch.Tensor | None = None
        self.local_grad_accumulator: torch.Tensor | None = None
        self.grad_bucket_input: torch.Tensor | None = None
        self.grad_bucket_input_is_compact: bool = False
        self._packed_grad_offset_cache: dict[bool, dict[str, int | None]] = {}

    @property
    def sharded_param(self):
        return self.param_state

    @property
    def sharded_grad(self):
        return self.grad_state

    @property
    def local_shard_dtensor(self):
        return self.param_state.dtensor

    @property
    def local_grad_shard_dtensor(self):
        return self.grad_state.dtensor if self.grad_state is not None else None

    @property
    def local_shard(self) -> torch.Tensor:
        return self.param_state.local_tensor

    @property
    def local_grad_shard(self) -> torch.Tensor | None:
        return self.grad_state.local_tensor if self.grad_state is not None else None

    @property
    def _local_shard_fallback(self) -> torch.Tensor | None:
        return self.param_state.fallback_tensor

    @property
    def _local_grad_shard_fallback(self) -> torch.Tensor | None:
        return self.grad_state.fallback_tensor if self.grad_state is not None else None

    @property
    def local_numel(self) -> int:
        return self.shard_sizes[self.rank]

    @property
    def shard_sizes(self) -> tuple[int, ...]:
        if self.matrix_shard is None:
            return self.plan.shard_sizes
        return self.matrix_shard.shard_lengths(self.plan.total_numel)

    @property
    def placement(self) -> MatrixShard | None:
        return self.matrix_shard

    @property
    def placement_compatibility(self) -> PlacementCompatibility:
        return self.matrix_shard_compatibility

    def _init_local_shard(self) -> torch.Tensor:
        full_buffer = torch.cat([mp.param.detach().reshape(-1) for mp in self.managed_params])
        packed_shards = self._pack_full_tensor_by_rank_segments(full_buffer)
        return packed_shards[self.local_start : self.local_end].clone()

    def _build_local_param_views(self) -> list[LocalParamView]:
        views: list[LocalParamView] = []
        for mp in self.managed_params:
            for segment in self.local_segments:
                view = self._build_local_param_view(mp, segment)
                if view is not None:
                    views.append(view)
        return views

    def _build_local_param_view(self, mp: ManagedParam, segment: LayoutSegment) -> LocalParamView | None:
        global_start = max(mp.offset, segment.global_start)
        global_end = min(mp.end, segment.global_end)
        if global_start >= global_end:
            return None
        shard_start = segment.local_start + global_start - segment.global_start
        shard_end = segment.local_start + global_end - segment.global_start
        return LocalParamView(
            managed_param=mp,
            param_start=global_start - mp.offset,
            param_end=global_end - mp.offset,
            shard_start=shard_start,
            shard_end=shard_end,
        )

    def use_local_shards(
        self,
        *,
        preserve_full_grad_buffer: bool = False,
        preserve_grad_bucket_input: bool = False,
        preserve_local_grad_shard: bool = False,
    ) -> None:
        if not preserve_full_grad_buffer:
            self.clear_full_grad_buffer()
        if preserve_local_grad_shard:
            if not preserve_grad_bucket_input:
                self.grad_bucket_input = None
        else:
            self.clear_local_grad_shard(preserve_grad_bucket_input=preserve_grad_bucket_input)
        local_views_by_param = self._local_views_by_param()
        for view in self.local_param_views:
            mp = view.managed_param
            if not preserve_local_grad_shard:
                mp.param.grad = None
            views = local_views_by_param[mp.fqn]
            if len(views) != 1:
                raise NotImplementedError("MatrixFSDP V0 cannot expose multi-segment parameters yet.")
            if view.numel != mp.numel:
                mp.param.data = self.local_shard[view.shard_start : view.shard_end]
                continue
            mp.param.data = self.local_shard[view.shard_start : view.shard_end].view(mp.shape)
        for mp in self.managed_params:
            if mp.fqn not in local_views_by_param:
                mp.param.data = self.local_shard.new_empty(0)
                if not preserve_local_grad_shard:
                    mp.param.grad = None

    def all_gather_full_params(self) -> None:
        self.finish_all_gather_full_params(self.start_all_gather_full_params())

    def start_all_gather_full_params(
        self,
        *,
        validate_owner_collective_signature: bool = True,
    ) -> MatrixCollectiveHandle:
        all_gather_input = self._maybe_to_param_dtype(self.local_shard)
        full_buffer_reused = self.full_buffer is not None
        if self.full_buffer is None:
            acquired = self.full_param_buffer_pool.acquire_with_stats(
                self._param_reference_tensor(),
                self.plan.total_numel,
            )
            self.full_buffer = acquired.tensor
            full_buffer_reused = acquired.reused
        else:
            self._resize_full_buffer_storage(self.plan.total_numel)
        if self._is_single_rank_shard_group():
            self.full_buffer.copy_(all_gather_input)
            handle = MatrixCollectiveHandle(lambda: self.full_buffer)
            return self._annotate_param_materialization_handle(
                handle,
                kind="single_rank_copy",
                full_buffer_reused=full_buffer_reused,
            )
        if self.param_gather_strategy == "equal_all_gather" and not self._can_direct_all_gather_full_params():
            raise RuntimeError("param_gather_strategy='equal_all_gather' requires equal rank-ordered shards.")
        if self.param_gather_strategy != "matrix_all_gather" and self._can_direct_all_gather_full_params():
            handle = all_gather_equal_1d_into_async(
                all_gather_input,
                self.full_buffer,
                group=self.group,
                cuda_stream=self.cuda_all_gather_stream,
            )
            return self._annotate_param_materialization_handle(
                handle,
                kind="equal_all_gather",
                full_buffer_reused=full_buffer_reused,
            )
        if self.param_gather_strategy == "owner_broadcast" and not self._can_owner_broadcast_full_params():
            raise RuntimeError("param_gather_strategy='owner_broadcast' requires whole-parameter owner shards.")
        if self._should_use_owner_segment_collectives():
            communication_plan = self.elastic_param_buffer.communication_plan
            handle = all_gatherv_rank_segments_1d_into_async(
                all_gather_input,
                self.full_buffer,
                communication_plan.rank_segments,
                self.rank,
                backend=self._owner_segment_collective_backend(),
                group=self.group,
                cuda_stream=self.cuda_all_gather_stream,
                collective_key=self.collective_key,
                validate_owner_collective_signature=validate_owner_collective_signature,
                coalesced_rank_segments=communication_plan.coalesced_rank_segments,
                rank_chunk_shard_sizes=communication_plan.rank_chunk_shard_sizes,
                rank_chunk_segments=communication_plan.rank_chunk_segments,
            )
            return self._annotate_param_materialization_handle(
                handle,
                kind=f"owner_segment:{self._owner_segment_collective_backend()}",
                full_buffer_reused=full_buffer_reused,
            )

        packed_handle = all_gather_matrix_shard_1d_into_async(
            all_gather_input,
            self.full_buffer,
            self.matrix_shard,
            self.plan.total_numel,
            group=self.group,
            cuda_stream=self.cuda_all_gather_stream,
        )

        def wait() -> torch.Tensor:
            return self._unpack_rank_shards_to_full_tensor(packed_handle.wait())

        return self._annotate_param_materialization_handle(
            MatrixCollectiveHandle(wait),
            kind="matrix_all_gather",
            full_buffer_reused=full_buffer_reused,
        )

    def _annotate_param_materialization_handle(
        self,
        handle: MatrixCollectiveHandle,
        *,
        kind: str,
        full_buffer_reused: bool,
    ) -> MatrixCollectiveHandle:
        handle.param_materialization_kind = kind
        handle.param_materialization_numel = int(self.plan.total_numel)
        handle.param_materialization_bytes = int(self.plan.total_numel) * int(self._param_reference_tensor().element_size())
        handle.param_materialization_reused = bool(full_buffer_reused)
        handle.param_materialization_rank_chunk_fast_path = bool(
            self.elastic_param_buffer.layout.rank_chunk_fast_path
            if kind.startswith("owner_segment")
            else self.static_param_buffer.layout.rank_chunk_fast_path
        )
        handle.param_materialization_packed_full_order = bool(
            self.elastic_param_buffer.layout.packed_rank_shards_are_full_tensor_order
            if kind.startswith("owner_segment")
            else self.static_param_buffer.layout.packed_rank_shards_are_full_tensor_order
        )
        return handle

    def finish_all_gather_full_params(self, handle: MatrixCollectiveHandle) -> None:
        self.use_full_param_buffer(handle.wait())

    def use_full_param_shards(self, gathered_shards: torch.Tensor) -> None:
        self.use_full_param_buffer(self._unpack_rank_shards_to_full_tensor(gathered_shards))

    def use_full_param_buffer(self, full_buffer: torch.Tensor) -> None:
        if (
            self.full_buffer is not None
            and self.full_buffer.untyped_storage().data_ptr() != full_buffer.untyped_storage().data_ptr()
        ):
            self._release_full_buffer_if_inactive()
        self.full_buffer = full_buffer
        if self.full_buffer.numel() != self.plan.total_numel:
            raise RuntimeError(
                f"Gathered {self.full_buffer.numel()} elements, expected {self.plan.total_numel}."
            )
        for mp in self.managed_params:
            mp.param.data = self.full_buffer[mp.offset : mp.end].view(mp.shape)

    def prepare_full_grad_buffer(self, *, accumulate: bool = False) -> bool:
        if self.full_buffer is None:
            raise RuntimeError("prepare_full_grad_buffer() requires full parameters to be available.")
        reused = accumulate and self.full_grad_buffer is not None
        if reused and self.full_grad_buffer.numel() != self.plan.total_numel:
            raise RuntimeError(
                f"Accumulated full grad buffer has {self.full_grad_buffer.numel()} elements, "
                f"expected {self.plan.total_numel}."
            )
        if not reused:
            self.full_grad_buffer = self._grad_accumulation_reference_tensor().new_zeros(self.plan.total_numel)
        for mp in self.managed_params:
            mp.param.grad_dtype = self.full_grad_buffer.dtype
            mp.param.grad = self.full_grad_buffer[mp.offset : mp.end].view(mp.shape)
        return reused

    def prepare_local_grad_accumulator(self) -> None:
        self.clear_full_grad_buffer()
        self.local_grad_accumulator = self.local_shard.new_zeros(self.local_numel)

    def prepare_grad_bucket(self, *, zero_copy: bool = True, accumulate: bool = False) -> bool:
        self.clear_full_grad_buffer()
        self.local_grad_accumulator = None
        if self._param_compute_dtype() != self.local_shard.dtype:
            zero_copy = False
        if self.reduce_dtype is not None and self.reduce_dtype != self.local_shard.dtype:
            zero_copy = False
        if not accumulate:
            self.grad_bucket_input = None
            self.grad_bucket_input_is_compact = False
        if not zero_copy:
            for mp in self.managed_params:
                mp.param.grad = None
            return False
        if not self._can_prepare_zero_copy_grad_bucket():
            return False
        compact_owner_bucket = self._can_use_compact_owner_grad_bucket()
        packed_numel = self.plan.total_numel if compact_owner_bucket else len(self.shard_sizes) * max(self.shard_sizes, default=0)
        if accumulate and self.grad_bucket_input is not None:
            if self.grad_bucket_input.numel() != packed_numel:
                raise RuntimeError(
                    f"Accumulated grad bucket has {self.grad_bucket_input.numel()} elements, expected {packed_numel}."
                )
            if self.grad_bucket_input_is_compact != compact_owner_bucket:
                raise RuntimeError("Accumulated grad bucket compact layout changed between backward passes.")
            packed = self.grad_bucket_input
        else:
            packed = self.local_shard.new_zeros(packed_numel)
        for mp in self.managed_params:
            view = self._packed_grad_view_for_param(mp, packed, compact=compact_owner_bucket)
            if view is None:
                self.grad_bucket_input = None
                self.grad_bucket_input_is_compact = False
                for reset_mp in self.managed_params:
                    reset_mp.param.grad = None
                return False
            mp.param.grad = view.view(mp.shape)
        self.grad_bucket_input = packed
        self.grad_bucket_input_is_compact = compact_owner_bucket
        return True

    def accumulate_grad_bucket_input_from_param_grads(self) -> bool:
        if self.plan.rank_segments is None:
            raise RuntimeError("MatrixGradBucket requires rank segment metadata.")
        param_grads = []
        for mp in self.managed_params:
            grad = mp.param.grad
            if grad is None:
                continue
            param_grads.append(BucketParamGrad(mp, self._maybe_to_reduce_dtype(grad.detach().reshape(-1))))
            mp.param.grad = None
        if not param_grads:
            return False
        bucket = MatrixGradBucket(
            param_grads=tuple(param_grads),
            total_numel=self.plan.total_numel,
            shard_sizes=self.shard_sizes,
            rank_segments=self.plan.rank_segments,
        )
        packed = new_reduce_scatter_input(bucket, self._reduce_reference_tensor())
        fill_reduce_scatter_input(bucket, packed)
        if self.grad_bucket_input is None:
            self.grad_bucket_input = packed
            self.grad_bucket_input_is_compact = False
            return False
        if self.grad_bucket_input.numel() != packed.numel():
            raise RuntimeError(
                f"Accumulated grad bucket has {self.grad_bucket_input.numel()} elements, "
                f"new grad bucket has {packed.numel()}."
            )
        if self.grad_bucket_input.dtype != packed.dtype:
            packed = packed.to(dtype=self.grad_bucket_input.dtype)
        self.grad_bucket_input.add_(packed)
        return True

    def reduce_param_grad_to_local_accumulator(self, mp: ManagedParam) -> None:
        if self.local_grad_accumulator is None:
            raise RuntimeError("reduce_param_grad_to_local_accumulator() requires an active local grad accumulator.")
        if mp.param.grad is None:
            return
        shard_sizes = self.param_shard_sizes(mp)
        param_grad = self._maybe_to_reduce_dtype(mp.param.grad.detach().reshape(-1))
        if self._is_single_rank_shard_group():
            local_param_grad = param_grad
        else:
            local_param_grad = reduce_scatter_matrix_1d(
                param_grad,
                shard_sizes,
                self.rank,
                group=self.group,
                divide_by_world=self.divide_grads_by_world,
            )
        local_param_grad = self._sync_replicated_grad_shard(local_param_grad)
        local_param_grad = self._maybe_to_local_dtype(local_param_grad)
        views = self._local_views_by_param().get(mp.fqn, ())
        if views:
            if len(views) != 1:
                raise NotImplementedError("MatrixFSDP V0 cannot expose multi-segment parameter gradients yet.")
            view = views[0]
            expected_numel = view.shard_end - view.shard_start
            if local_param_grad.numel() != expected_numel:
                raise RuntimeError(
                    f"Reduced grad for {mp.fqn} has {local_param_grad.numel()} elements, expected {expected_numel}."
                )
            self.local_grad_accumulator[view.shard_start : view.shard_end].copy_(local_param_grad)
        mp.param.grad = None

    def all_reduce_param_grad_to_local_accumulator(self, mp: ManagedParam) -> None:
        if self.local_grad_accumulator is None:
            raise RuntimeError("all_reduce_param_grad_to_local_accumulator() requires an active local grad accumulator.")
        if mp.param.grad is None:
            return
        full_param_grad = self._maybe_to_reduce_dtype(mp.param.grad.detach().reshape(-1))
        if not self._is_single_rank_shard_group():
            full_param_grad = all_reduce_full_grad(
                full_param_grad,
                group=self.group,
                divide_by_world=self.divide_grads_by_world,
            )
        views = self._local_views_by_param().get(mp.fqn, ())
        if views:
            if len(views) != 1:
                raise NotImplementedError("MatrixFSDP V0 cannot expose multi-segment parameter gradients yet.")
            view = views[0]
            local_param_grad = full_param_grad[view.param_start : view.param_end]
            local_param_grad = self._sync_replicated_grad_shard(local_param_grad)
            local_param_grad = self._maybe_to_local_dtype(local_param_grad)
            expected_numel = view.shard_end - view.shard_start
            if local_param_grad.numel() != expected_numel:
                raise RuntimeError(
                    f"Reduced grad for {mp.fqn} has {local_param_grad.numel()} elements, expected {expected_numel}."
                )
            self.local_grad_accumulator[view.shard_start : view.shard_end].copy_(local_param_grad)
        mp.param.grad = None

    def finish_local_grad_accumulator(self) -> torch.Tensor:
        if self.local_grad_accumulator is None:
            raise RuntimeError("finish_local_grad_accumulator() requires an active local grad accumulator.")
        local_grad_shard = self.local_grad_accumulator
        self.local_grad_accumulator = None
        return local_grad_shard

    def collect_grad_bucket(self) -> MatrixGradBucket:
        if self.grad_bucket_input is not None:
            packed_input = self.grad_bucket_input
            packed_input_is_compact = self.grad_bucket_input_is_compact
            self.grad_bucket_input = None
            self.grad_bucket_input_is_compact = False
            for mp in self.managed_params:
                mp.param.grad = None
            if self.plan.rank_segments is None:
                raise RuntimeError("MatrixGradBucket requires rank segment metadata.")
            return MatrixGradBucket(
                param_grads=(),
                total_numel=self.plan.total_numel,
                shard_sizes=self.shard_sizes,
                rank_segments=self.plan.rank_segments,
                packed_input=packed_input,
                packed_input_is_compact=packed_input_is_compact,
            )
        param_grads = []
        for mp in self.managed_params:
            grad = mp.param.grad
            if grad is None:
                continue
            param_grads.append(BucketParamGrad(mp, self._maybe_to_reduce_dtype(grad.detach().reshape(-1))))
            mp.param.grad = None
        if self.plan.rank_segments is None:
            raise RuntimeError("MatrixGradBucket requires rank segment metadata.")
        return MatrixGradBucket(
            param_grads=tuple(param_grads),
            total_numel=self.plan.total_numel,
            shard_sizes=self.shard_sizes,
            rank_segments=self.plan.rank_segments,
        )

    def reduce_grad_bucket_to_local_shard(self, bucket: MatrixGradBucket) -> torch.Tensor:
        return self.start_reduce_grad_bucket_to_local_shard(bucket).wait()

    def start_reduce_grad_bucket_to_local_shard(self, bucket: MatrixGradBucket) -> MatrixTensorCollectiveHandle:
        return self.start_reduce_grad_bucket_to_local_shard_with_stats(bucket).handle

    def start_reduce_grad_bucket_to_local_shard_with_stats(self, bucket: MatrixGradBucket) -> GradBucketReduceStart:
        if bucket.total_numel != self.plan.total_numel:
            raise RuntimeError(
                f"Grad bucket has total_numel={bucket.total_numel}, expected {self.plan.total_numel}."
            )
        if bucket.shard_sizes != self.shard_sizes:
            raise RuntimeError(f"Grad bucket shard sizes {bucket.shard_sizes} do not match {self.shard_sizes}.")
        if not bucket.has_grads:
            local_grad_shard = self.local_shard.new_zeros(self.local_numel)
            handle = MatrixTensorCollectiveHandle(local_grad_shard, lambda: local_grad_shard, _waited=True)
            return GradBucketReduceStart(
                handle=handle,
                stats=GradBucketReduceStartStats(
                    layout_kind="empty",
                    needs_copy_in=False,
                    copy_in_ms=0.0,
                    reduce_scatter_enqueue_ms=0.0,
                    packed_numel=0,
                    packed_bytes=0,
                ),
            )
        if self._is_single_rank_shard_group():
            packed_rank_chunks = bucket.packed_input
            needs_copy_in = packed_rank_chunks is None
            layout_kind = classify_copy_in_layout(bucket)
            copy_in_ms = 0.0
            workspace_lease = None
            if packed_rank_chunks is None:
                workspace_lease = self._comm_workspace().acquire(
                    self._reduce_reference_tensor(),
                    bucket.world_size * bucket.max_shard_size,
                )
                packed_rank_chunks = workspace_lease.tensor
                copy_start = perf_counter()
                fill_reduce_scatter_input(bucket, packed_rank_chunks)
                copy_in_ms = (perf_counter() - copy_start) * 1000.0
            local_grad_shard = self._maybe_to_local_dtype(packed_rank_chunks[: self.local_numel].contiguous())
            if workspace_lease is not None:
                local_grad_shard = local_grad_shard.clone()
            if workspace_lease is not None:
                workspace_lease.release()
            handle = MatrixTensorCollectiveHandle(local_grad_shard, lambda: local_grad_shard, _waited=True)
            return GradBucketReduceStart(
                handle=handle,
                stats=GradBucketReduceStartStats(
                    layout_kind=layout_kind,
                    needs_copy_in=needs_copy_in,
                    copy_in_ms=copy_in_ms,
                    reduce_scatter_enqueue_ms=0.0,
                    packed_numel=packed_rank_chunks.numel(),
                    packed_bytes=packed_rank_chunks.numel() * packed_rank_chunks.element_size(),
                ),
            )
        if self.replicate_group is not None and self.replicate_world_size > 1:
            workspace_lease = None
            copy_start = perf_counter()
            if bucket.packed_input is not None:
                packed_rank_chunks = self._maybe_to_reduce_dtype(bucket.packed_input)
            else:
                workspace_lease = self._comm_workspace().acquire(
                    self._reduce_reference_tensor(),
                    bucket.world_size * bucket.max_shard_size,
                )
                packed_rank_chunks = workspace_lease.tensor
                fill_reduce_scatter_input(bucket, packed_rank_chunks)
            copy_in_ms = (perf_counter() - copy_start) * 1000.0
            enqueue_start = perf_counter()
            local_grad_shard = reduce_scatter_padded_rank_chunks_1d(
                packed_rank_chunks,
                self.local_numel,
                group=self.group,
                divide_by_world=self.divide_grads_by_world,
            )
            local_grad_shard = self._sync_replicated_grad_shard(local_grad_shard)
            local_grad_shard = self._maybe_to_local_dtype(local_grad_shard)
            if workspace_lease is not None:
                workspace_lease.release()
            handle = MatrixTensorCollectiveHandle(local_grad_shard, lambda: local_grad_shard, _waited=True)
            return GradBucketReduceStart(
                handle=handle,
                stats=GradBucketReduceStartStats(
                    layout_kind=classify_copy_in_layout(bucket),
                    needs_copy_in=bucket.packed_input is None,
                    copy_in_ms=copy_in_ms,
                    reduce_scatter_enqueue_ms=(perf_counter() - enqueue_start) * 1000.0,
                    packed_numel=packed_rank_chunks.numel(),
                    packed_bytes=packed_rank_chunks.numel() * packed_rank_chunks.element_size(),
                ),
            )
        packed_rank_chunks = bucket.packed_input
        needs_copy_in = packed_rank_chunks is None
        layout_kind = classify_copy_in_layout(bucket)
        packed_input_is_compact = bucket.packed_input_is_compact
        workspace_kind = "compact_rank_chunks" if packed_input_is_compact else "padded_rank_chunks"
        workspace_numel = packed_rank_chunks.numel() if packed_rank_chunks is not None else 0
        workspace_padding_waste_numel = 0 if packed_input_is_compact else self.elastic_param_buffer.layout.padding_waste_numel
        workspace_persistent = False
        if packed_rank_chunks is not None:
            packed_rank_chunks = self._maybe_to_reduce_dtype(packed_rank_chunks)
        if packed_rank_chunks is not None and packed_input_is_compact and not self._can_owner_broadcast_full_params():
            raise RuntimeError("Compact owner grad buckets require whole-parameter owner shards.")
        workspace_lease = None
        if packed_rank_chunks is None:
            packed_input_is_compact = self._can_copy_bucket_to_compact_owner_rank_chunks(bucket)
            if packed_input_is_compact:
                workspace_lease = self.elastic_param_buffer.acquire_rank_chunk_workspace(
                    self._reduce_reference_tensor(),
                    compact=True,
                    persistent=self.max_cached_elastic_workspaces_per_key > 0,
                )
                workspace_kind = "compact_rank_chunks"
                workspace_padding_waste_numel = 0
            else:
                workspace_lease = self._comm_workspace().acquire(
                    self._reduce_reference_tensor(),
                    bucket.world_size * bucket.max_shard_size,
                )
                workspace_kind = "padded_rank_chunks"
                workspace_padding_waste_numel = bucket.world_size * bucket.max_shard_size - sum(bucket.shard_sizes)
            packed_rank_chunks = workspace_lease.tensor
            workspace_numel = packed_rank_chunks.numel()
            workspace_persistent = bool(getattr(workspace_lease, "persistent", False))
        copy_in_ms = 0.0
        if needs_copy_in:
            copy_start = perf_counter()
            if packed_input_is_compact:
                self._fill_compact_owner_reduce_scatter_input(bucket, packed_rank_chunks)
                layout_kind = "compact_owner"
            else:
                fill_reduce_scatter_input(bucket, packed_rank_chunks)
            copy_in_ms = (perf_counter() - copy_start) * 1000.0
            del bucket
        enqueue_start = perf_counter()
        if self._should_use_owner_segment_collectives():
            handle = reduce_scatterv_owner_rank_chunks_1d_async(
                packed_rank_chunks,
                self.shard_sizes,
                self.rank,
                backend=self._owner_segment_collective_backend(),
                group=self.group,
                divide_by_world=self.divide_grads_by_world,
                cuda_stream=self.cuda_reduce_scatter_stream,
                compact=packed_input_is_compact,
            )
        else:
            handle = reduce_scatter_padded_rank_chunks_1d_async(
                packed_rank_chunks,
                self.local_numel,
                group=self.group,
                divide_by_world=self.divide_grads_by_world,
                cuda_stream=self.cuda_reduce_scatter_stream,
            )
        if workspace_lease is not None:
            handle = _release_workspace_after_wait(handle, workspace_lease)
        if self.reduce_dtype is not None and self.reduce_dtype != self.local_shard.dtype:
            local_grad_shard = self._maybe_to_local_dtype(handle.wait())
            handle = MatrixTensorCollectiveHandle(
                local_grad_shard,
                lambda: local_grad_shard,
                _waited=True,
                collective_kind=handle.collective_kind,
                collective_backend=handle.collective_backend,
                collective_impl=handle.collective_impl,
                collective_numel=handle.collective_numel,
                collective_bytes=handle.collective_bytes,
                collective_count=handle.collective_count,
                collective_sync_mode=handle.collective_sync_mode,
            )
        return GradBucketReduceStart(
            handle=handle,
            stats=GradBucketReduceStartStats(
                layout_kind=layout_kind,
                needs_copy_in=needs_copy_in,
                copy_in_ms=copy_in_ms,
                reduce_scatter_enqueue_ms=(perf_counter() - enqueue_start) * 1000.0,
                packed_numel=packed_rank_chunks.numel(),
                packed_bytes=packed_rank_chunks.numel() * packed_rank_chunks.element_size(),
                workspace_kind=workspace_kind,
                workspace_numel=workspace_numel,
                workspace_padding_waste_numel=workspace_padding_waste_numel,
                workspace_persistent=workspace_persistent,
            ),
        )

    def reduce_full_grads_to_local_shard(self) -> torch.Tensor:
        if self.full_grad_buffer is not None:
            full_grad = self.full_grad_buffer
            for mp in self.managed_params:
                mp.param.grad = None
        else:
            reference = self.local_shard
            full_grad = reference.new_zeros(self.plan.total_numel)
            for mp in self.managed_params:
                if mp.param.grad is None:
                    continue
                full_grad[mp.offset : mp.end].copy_(mp.param.grad.detach().reshape(-1))
                mp.param.grad = None

        if self._is_single_rank_shard_group():
            local_grad_shard = self._maybe_to_local_dtype(full_grad[: self.local_numel].contiguous())
            self.clear_full_grad_buffer()
            return self._sync_replicated_grad_shard(local_grad_shard)

        if hasattr(torch.distributed, "reduce_scatter_tensor") and self._can_direct_rank_order_collectives():
            local_grad_shard = reduce_scatter_equal_1d(
                self._maybe_to_reduce_dtype(full_grad),
                self.shard_sizes[self.rank],
                group=self.group,
                divide_by_world=self.divide_grads_by_world,
            )
            self.clear_full_grad_buffer()
            return self._maybe_to_local_dtype(self._sync_replicated_grad_shard(local_grad_shard))

        packed_full_grad = self._pack_full_tensor_by_rank_segments(self._maybe_to_reduce_dtype(full_grad))
        local_grad_shard = reduce_scatter_matrix_shard_1d(
            packed_full_grad,
            self.matrix_shard,
            self.rank,
            group=self.group,
            divide_by_world=self.divide_grads_by_world,
        )
        self.clear_full_grad_buffer()
        return self._maybe_to_local_dtype(self._sync_replicated_grad_shard(local_grad_shard))

    def use_local_grad_shard(self, local_grad_shard: torch.Tensor) -> None:
        if local_grad_shard.numel() != self.local_numel:
            raise RuntimeError(
                f"Local grad shard has {local_grad_shard.numel()} elements, expected {self.local_numel}."
            )
        self.grad_state = self._make_sharded_state(
            "grad",
            local_grad_shard,
            self.matrix_shard,
            global_shape=(self.plan.total_numel,),
            global_stride=(1,),
        )
        local_views_by_param = self._local_views_by_param()
        for mp in self.managed_params:
            views = local_views_by_param.get(mp.fqn)
            if not views:
                mp.param.grad = None
                continue
            if len(views) != 1:
                raise NotImplementedError("MatrixFSDP V0 cannot expose multi-segment parameter gradients yet.")
            view = views[0]
            if view.numel != mp.numel:
                mp.param.grad = local_grad_shard[view.shard_start : view.shard_end]
                continue
            mp.param.grad = local_grad_shard[view.shard_start : view.shard_end].view(mp.shape)

    def accumulate_local_grad_shard(self, local_grad_shard: torch.Tensor) -> bool:
        existing = self.local_grad_shard
        if existing is None:
            self.use_local_grad_shard(local_grad_shard)
            return False
        if existing.numel() != local_grad_shard.numel():
            raise RuntimeError(
                f"Accumulated local grad shard has {existing.numel()} elements, "
                f"new local grad shard has {local_grad_shard.numel()}."
            )
        if existing.dtype != local_grad_shard.dtype:
            local_grad_shard = local_grad_shard.to(dtype=existing.dtype)
        existing.add_(local_grad_shard)
        self.use_local_grad_shard(existing)
        return True

    def clear_full_params(
        self,
        *,
        shrink_storage: bool = False,
        keep_shrunk_tensor: bool = True,
    ) -> FullParamClearStats:
        if self.full_buffer is None:
            return FullParamClearStats(kind="already_clear")
        clear_numel = int(self.full_buffer.numel())
        clear_bytes = int(self.full_buffer.untyped_storage().nbytes())
        if shrink_storage and self.full_buffer is not None:
            if self.full_param_buffer_pool.can_cache(self.full_buffer):
                release = self._release_full_buffer_if_inactive()
                self.full_buffer = None
                return FullParamClearStats(
                    kind=f"released:{release.kind if release is not None else 'skipped'}",
                    release=release,
                    numel=clear_numel,
                    bytes=clear_bytes,
                )
            self._resize_full_buffer_storage(0)
            if not keep_shrunk_tensor:
                self.full_buffer = None
            return FullParamClearStats(kind="shrunk", numel=clear_numel, bytes=clear_bytes)
        release = self._release_full_buffer_if_inactive()
        self.full_buffer = None
        return FullParamClearStats(
            kind=f"released:{release.kind if release is not None else 'skipped'}",
            release=release,
            numel=clear_numel,
            bytes=clear_bytes,
        )

    def _resize_full_buffer_storage(self, numel: int) -> None:
        if self.full_buffer is None:
            return
        self.full_buffer.untyped_storage().resize_(numel * self.full_buffer.element_size())

    def set_full_param_buffer_pool(self, pool: FullParamBufferPool) -> None:
        self._release_full_buffer_if_inactive()
        self.full_buffer = None
        self.full_param_buffer_pool = pool

    def set_elastic_workspace_cache_limit(self, max_cached_per_key: int) -> None:
        self.max_cached_elastic_workspaces_per_key = max_cached_per_key
        self.elastic_param_buffer.workspace.set_max_cached_per_key(max_cached_per_key)
        self.static_param_buffer.workspace.set_max_cached_per_key(max_cached_per_key)
        if max_cached_per_key == 0:
            self.elastic_param_buffer.clear_idle_persistent_rank_chunk_workspaces()

    def _comm_workspace(self) -> CommWorkspaceCache:
        if self._should_use_owner_segment_collectives():
            return self.elastic_param_buffer.workspace
        return self.static_param_buffer.workspace

    def set_cuda_streams(
        self,
        *,
        all_gather_stream: torch.cuda.Stream | None = None,
        reduce_scatter_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.cuda_all_gather_stream = all_gather_stream
        self.cuda_reduce_scatter_stream = reduce_scatter_stream
        self.cuda_comm_stream = all_gather_stream

    def clear_full_grad_buffer(self) -> None:
        self.full_grad_buffer = None

    def clear_local_grad_shard(self, *, preserve_grad_bucket_input: bool = False) -> None:
        self.grad_state = None
        self.local_grad_accumulator = None
        if not preserve_grad_bucket_input:
            self.grad_bucket_input = None

    def param_grads_alias_full_grad_buffer(self) -> bool:
        if self.full_grad_buffer is None:
            return False
        if (
            self.full_grad_buffer.untyped_storage().nbytes()
            < self.full_grad_buffer.numel() * self.full_grad_buffer.element_size()
        ):
            return False
        full_grad_storage_ptr = self.full_grad_buffer.untyped_storage().data_ptr()
        for mp in self.managed_params:
            grad = mp.param.grad
            if grad is None:
                return False
            if grad.untyped_storage().data_ptr() != full_grad_storage_ptr:
                return False
            expected = self.full_grad_buffer[mp.offset : mp.end].view(mp.shape)
            if grad.data_ptr() != expected.data_ptr():
                return False
        return True

    def param_data_alias_full_buffer(self) -> bool:
        if self.full_buffer is None:
            return False
        if (
            self.full_buffer.untyped_storage().nbytes()
            < self.full_buffer.numel() * self.full_buffer.element_size()
        ):
            return False
        full_storage_ptr = self.full_buffer.untyped_storage().data_ptr()
        for mp in self.managed_params:
            data = mp.param.data
            if data.untyped_storage().data_ptr() != full_storage_ptr:
                return False
            expected = self.full_buffer[mp.offset : mp.end].view(mp.shape)
            if data.data_ptr() != expected.data_ptr():
                return False
        return True

    def param_data_alias_local_shard(self) -> bool:
        local_storage_ptr = self.local_shard.untyped_storage().data_ptr()
        local_views_by_param = self._local_views_by_param()
        for mp in self.managed_params:
            views = local_views_by_param.get(mp.fqn)
            data = mp.param.data
            if not views:
                if data.numel() != 0:
                    return False
                continue
            if len(views) != 1:
                return False
            view = views[0]
            if data.untyped_storage().data_ptr() != local_storage_ptr:
                return False
            expected = self.local_shard[view.shard_start : view.shard_end]
            if view.numel == mp.numel:
                expected = expected.view(mp.shape)
            if data.data_ptr() != expected.data_ptr():
                return False
        return True

    def _release_full_buffer_if_inactive(self) -> FullParamBufferRelease | None:
        if self.full_buffer is None:
            return None
        if self.param_data_alias_full_buffer():
            return FullParamBufferRelease(
                kind="active_param_view",
                numel=int(self.full_buffer.numel()),
                bytes=int(self.full_buffer.untyped_storage().nbytes()),
            )
        return self.full_param_buffer_pool.release(self.full_buffer)

    def _can_direct_all_gather_full_params(self) -> bool:
        if not hasattr(torch.distributed, "all_gather_into_tensor"):
            return False
        return self._can_direct_rank_order_collectives()

    def _can_direct_rank_order_collectives(self) -> bool:
        if not self.matrix_shard_compatibility.compatible:
            return False
        if len(set(self.shard_sizes)) != 1:
            return False
        return True

    def _can_owner_broadcast_full_params(self) -> bool:
        if self.plan.rank_segments is None:
            return False
        if sum(self.shard_sizes) != self.plan.total_numel:
            return False
        for mp in self.managed_params:
            matching_segments = []
            for segments in self.plan.rank_segments:
                for segment in segments:
                    global_start = max(mp.offset, segment.global_start)
                    global_end = min(mp.end, segment.global_end)
                    if global_start < global_end:
                        matching_segments.append((global_start, global_end))
            if len(matching_segments) != 1:
                return False
            global_start, global_end = matching_segments[0]
            if global_start != mp.offset or global_end != mp.end:
                return False
        return True

    def _should_use_owner_segment_collectives(self) -> bool:
        if not self._can_owner_broadcast_full_params():
            return False
        if self.param_gather_strategy == "owner_broadcast":
            return True
        if self.param_gather_strategy == "matrix_all_gather":
            return False
        return self.matrix_collective_backend in ("owner_broadcast", "custom")

    def uses_owner_segment_collectives(self) -> bool:
        return self._should_use_owner_segment_collectives()

    def communication_summary(self) -> dict[str, object]:
        owner_backend = self._owner_segment_collective_backend() if self._should_use_owner_segment_collectives() else None

        def resolve_custom(rank_segments):
            from matrix_fsdp.kernels.custom_collectives import custom_allgatherv_impl, resolve_custom_allgatherv_impl

            return custom_allgatherv_impl(), resolve_custom_allgatherv_impl(rank_segments)

        native_available = False
        native_sendrecv_chunk_enabled = True
        custom_reduce_impl = None
        if owner_backend == "custom":
            from matrix_fsdp.kernels.custom_collectives import (
                custom_reduce_scatterv_impl,
                native_sendrecv_chunk_fast_path_enabled,
            )
            from matrix_fsdp.kernels.native import native_kernel_available

            native_available = native_kernel_available()
            native_sendrecv_chunk_enabled = native_sendrecv_chunk_fast_path_enabled()
            custom_reduce_impl = custom_reduce_scatterv_impl()

        if owner_backend is None:
            return self.static_param_buffer.communication_summary(
                param_gather_strategy=self.param_gather_strategy,
                matrix_collective_backend=self.matrix_collective_backend,
                can_direct_all_gather=self._can_direct_all_gather_full_params(),
            )
        return self.elastic_param_buffer.communication_summary(
            param_gather_strategy=self.param_gather_strategy,
            matrix_collective_backend=self.matrix_collective_backend,
            can_direct_all_gather=self._can_direct_all_gather_full_params(),
            owner_segment_backend=owner_backend,
            custom_allgather_resolver=resolve_custom,
            native_kernel_available=native_available,
            native_sendrecv_chunk_enabled=native_sendrecv_chunk_enabled,
            custom_reduce_scatterv_impl=custom_reduce_impl,
        )

    def owner_segment_prefetch_skip_reason(self, *, ordered: bool = False) -> str | None:
        if not self._should_use_owner_segment_collectives():
            return None
        if self._owner_segment_collective_backend() != "custom":
            return None
        from matrix_fsdp.kernels.custom_collectives import custom_allgatherv_owner_prefetch_skip_reason

        return custom_allgatherv_owner_prefetch_skip_reason(rank_segments=self.plan.rank_segments, ordered=ordered)

    def owner_segment_collective_has_independent_comm_lanes(self) -> bool:
        if not self._should_use_owner_segment_collectives():
            return False
        if self._owner_segment_collective_backend() != "custom":
            return False
        from matrix_fsdp.kernels.custom_collectives import custom_allgatherv_has_independent_native_comm_lanes

        return custom_allgatherv_has_independent_native_comm_lanes(rank_segments=self.plan.rank_segments)

    def owner_segment_prefetch_order_gate_required(self) -> bool:
        return self._should_use_owner_segment_collectives() and self._owner_segment_collective_backend() == "custom"

    def _owner_segment_collective_backend(self) -> str:
        if self.matrix_collective_backend == "custom":
            return "custom"
        return "owner_broadcast"

    def _build_local_views_by_param(self) -> dict[str, list[LocalParamView]]:
        views_by_param: dict[str, list[LocalParamView]] = {}
        for view in self.local_param_views:
            views_by_param.setdefault(view.managed_param.fqn, []).append(view)
        return views_by_param

    def _local_views_by_param(self) -> dict[str, list[LocalParamView]]:
        return self._local_views_by_param_cache

    def _reduce_reference_tensor(self) -> torch.Tensor:
        if self.reduce_dtype is None or self.reduce_dtype == self.local_shard.dtype:
            return self.local_shard
        return torch.empty((), device=self.local_shard.device, dtype=self.reduce_dtype)

    def _grad_accumulation_reference_tensor(self) -> torch.Tensor:
        if self.reduce_dtype is not None:
            return self._reduce_reference_tensor()
        if self.full_buffer is None:
            return self.local_shard
        return self.full_buffer

    def _param_compute_dtype(self) -> torch.dtype:
        return self.param_dtype or self.local_shard.dtype

    def _param_reference_tensor(self) -> torch.Tensor:
        compute_dtype = self._param_compute_dtype()
        if compute_dtype == self.local_shard.dtype:
            return self.local_shard
        return torch.empty((), device=self.local_shard.device, dtype=compute_dtype)

    def _maybe_to_param_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._param_compute_dtype()
        if tensor.dtype == compute_dtype:
            return tensor
        if not tensor.is_floating_point():
            return tensor
        return tensor.to(dtype=compute_dtype)

    def _maybe_to_reduce_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.reduce_dtype is None or tensor.dtype == self.reduce_dtype:
            return tensor
        if not tensor.is_floating_point():
            return tensor
        return tensor.to(dtype=self.reduce_dtype)

    def _maybe_to_local_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == self.local_shard.dtype:
            return tensor
        if not tensor.is_floating_point():
            return tensor
        return tensor.to(dtype=self.local_shard.dtype)

    def _can_prepare_zero_copy_grad_bucket(self) -> bool:
        if self._param_compute_dtype() != self.local_shard.dtype:
            return False
        if self.plan.rank_segments is None:
            return False
        if max(self.shard_sizes, default=0) == 0:
            return False
        offsets = self._packed_grad_offsets(compact=False)
        return all(offsets.get(mp.fqn) is not None for mp in self.managed_params)

    def _can_use_compact_owner_grad_bucket(self) -> bool:
        if not self._should_use_owner_segment_collectives():
            return False
        offsets = self._packed_grad_offsets(compact=True)
        return all(offsets.get(mp.fqn) is not None for mp in self.managed_params)

    def _can_copy_bucket_to_compact_owner_rank_chunks(self, bucket: MatrixGradBucket) -> bool:
        if not self._should_use_owner_segment_collectives():
            return False
        if bucket.packed_input is not None:
            return bool(bucket.packed_input_is_compact)
        offsets = self._packed_grad_offsets(compact=True)
        return all(offsets.get(param_grad.managed_param.fqn) is not None for param_grad in bucket.param_grads)

    def _fill_compact_owner_reduce_scatter_input(
        self,
        bucket: MatrixGradBucket,
        packed_rank_chunks: torch.Tensor,
    ) -> torch.Tensor:
        expected_numel = self.elastic_param_buffer.rank_chunk_workspace_numel(compact=True)
        if packed_rank_chunks.ndim != 1:
            raise ValueError(f"packed_rank_chunks must be 1D, got shape {tuple(packed_rank_chunks.shape)}.")
        if packed_rank_chunks.numel() != expected_numel:
            raise ValueError(f"packed_rank_chunks has {packed_rank_chunks.numel()} elements, expected {expected_numel}.")
        packed_rank_chunks.zero_()
        for param_grad in bucket.param_grads:
            mp = param_grad.managed_param
            grad = param_grad.grad.reshape(-1)
            if grad.numel() != mp.numel:
                raise RuntimeError(f"Gradient for {mp.fqn} has {grad.numel()} elements, expected {mp.numel}.")
            view = self._packed_grad_view_for_param(mp, packed_rank_chunks, compact=True)
            if view is None:
                raise RuntimeError(f"Gradient for {mp.fqn} cannot be packed into compact owner rank chunks.")
            view.copy_(grad)
        return packed_rank_chunks

    def _is_single_rank_shard_group(self) -> bool:
        return len(self.shard_sizes) == 1

    def _packed_grad_view_for_param(self, mp: ManagedParam, packed: torch.Tensor, *, compact: bool = False) -> torch.Tensor | None:
        offset = self._packed_grad_offset_for_param(mp, compact=compact)
        if offset is None:
            return None
        return packed[offset : offset + mp.numel]

    def _packed_grad_offset_for_param(self, mp: ManagedParam, *, compact: bool = False) -> int | None:
        return self._packed_grad_offsets(compact=compact).get(mp.fqn)

    def _packed_grad_offsets(self, *, compact: bool) -> dict[str, int | None]:
        cached = self._packed_grad_offset_cache.get(compact)
        if cached is not None:
            return cached
        offsets = {mp.fqn: self._compute_packed_grad_offset_for_param(mp, compact=compact) for mp in self.managed_params}
        self._packed_grad_offset_cache[compact] = offsets
        return offsets

    def _compute_packed_grad_offset_for_param(self, mp: ManagedParam, *, compact: bool = False) -> int | None:
        if self.plan.rank_segments is None:
            return None
        max_shard_size = max(self.shard_sizes, default=0)
        cursor = 0
        packed_start: int | None = None
        packed_end: int | None = None
        for rank, segments in enumerate(self.plan.rank_segments):
            for segment in segments:
                global_start = max(mp.offset, segment.global_start)
                global_end = min(mp.end, segment.global_end)
                if global_start >= global_end:
                    continue
                param_start = global_start - mp.offset
                param_end = global_end - mp.offset
                if param_start != cursor:
                    return None
                rank_base = self._rank_packed_offset(rank) if compact else rank * max_shard_size
                current_start = rank_base + segment.local_start + global_start - segment.global_start
                current_end = current_start + (param_end - param_start)
                if packed_start is None:
                    packed_start = current_start
                elif packed_end != current_start:
                    return None
                packed_end = current_end
                cursor = param_end
        if packed_start is None or cursor != mp.numel:
            return None
        return packed_start

    def _pack_full_tensor_by_rank_segments(self, full_tensor: torch.Tensor) -> torch.Tensor:
        if full_tensor.ndim != 1:
            full_tensor = full_tensor.reshape(-1)
        if full_tensor.numel() != self.plan.total_numel:
            raise RuntimeError(f"Full tensor has {full_tensor.numel()} elements, expected {self.plan.total_numel}.")
        packed = full_tensor.new_empty(sum(self.shard_sizes))
        for rank, segments in enumerate(self.plan.rank_segments or ()):
            rank_base = self._rank_packed_offset(rank)
            for segment in segments:
                dst_start = rank_base + segment.local_start
                dst_end = dst_start + segment.numel
                packed[dst_start:dst_end].copy_(full_tensor[segment.global_start : segment.global_end])
        return packed

    def _unpack_rank_shards_to_full_tensor(self, rank_shards: torch.Tensor) -> torch.Tensor:
        if rank_shards.ndim != 1:
            rank_shards = rank_shards.reshape(-1)
        expected_numel = sum(self.shard_sizes)
        if rank_shards.numel() != expected_numel:
            raise RuntimeError(f"Packed rank shards have {rank_shards.numel()} elements, expected {expected_numel}.")
        if self._packed_rank_shards_are_full_tensor_order():
            return rank_shards
        full_tensor = rank_shards.new_empty(self.plan.total_numel)
        for rank, segments in enumerate(self.plan.rank_segments or ()):
            rank_base = self._rank_packed_offset(rank)
            for segment in segments:
                src_start = rank_base + segment.local_start
                src_end = src_start + segment.numel
                full_tensor[segment.global_start : segment.global_end].copy_(rank_shards[src_start:src_end])
        return full_tensor

    def _packed_rank_shards_are_full_tensor_order(self) -> bool:
        if self.plan.total_numel != sum(self.shard_sizes):
            return False
        if self.plan.rank_segments is None:
            return True
        for rank, segments in enumerate(self.plan.rank_segments):
            rank_base = self._rank_packed_offset(rank)
            local_cursor = 0
            for segment in segments:
                if segment.local_start != local_cursor:
                    return False
                packed_start = rank_base + segment.local_start
                packed_end = packed_start + segment.numel
                if segment.global_start != packed_start or segment.global_end != packed_end:
                    return False
                local_cursor = segment.local_end
            if local_cursor != self.plan.shard_sizes[rank]:
                return False
        return True

    def _local_range(self, rank: int) -> tuple[int, int]:
        start = self._rank_packed_offset(rank)
        return start, start + self.plan.shard_sizes[rank]

    def _local_segments(self, rank: int) -> tuple[LayoutSegment, ...]:
        return self.plan.local_segments(rank)

    def _rank_packed_offset(self, rank: int) -> int:
        return sum(self.plan.shard_sizes[:rank])

    def _can_use_segment_runtime_plan(self, plan: ShardPlan) -> bool:
        if plan.rank_segments is None:
            return False
        if plan.total_numel <= 0 or sum(plan.shard_sizes) != plan.total_numel:
            return False
        if len(plan.rank_segments) != len(plan.shard_sizes):
            return False
        for rank, segments in enumerate(plan.rank_segments):
            local_cursor = 0
            seen_fqns: set[str] = set()
            for segment in segments:
                if segment.local_start != local_cursor:
                    return False
                overlapping_fqns = self._managed_param_fqns_for_segment(segment)
                if len(overlapping_fqns) != 1:
                    return False
                fqn = overlapping_fqns[0]
                if fqn in seen_fqns:
                    return False
                seen_fqns.add(fqn)
                local_cursor = segment.local_end
            if local_cursor != plan.shard_sizes[rank]:
                return False
        return True

    def _managed_param_fqns_for_segment(self, segment: LayoutSegment) -> tuple[str, ...]:
        fqns = []
        for mp in self.managed_params:
            if max(mp.offset, segment.global_start) < min(mp.end, segment.global_end):
                fqns.append(mp.fqn)
        return tuple(fqns)

    def _make_sharded_state(
        self,
        name: str,
        local_tensor: torch.Tensor,
        placement: MatrixShard | None,
        *,
        global_shape: tuple[int, ...],
        global_stride: tuple[int, ...],
    ) -> MatrixShardedState:
        return MatrixShardedState(
            name,
            local_tensor,
            mesh=self.mesh,
            placement=placement,
            global_shape=global_shape,
            global_stride=global_stride,
            requires_grad=local_tensor.requires_grad,
            shard_mesh_dim=self.dp_shard_mesh_dim,
        )

    def make_param_state(self, mp: ManagedParam, name: str, local_tensor: torch.Tensor) -> MatrixShardedState | None:
        placement = self.param_matrix_shard(mp)
        if placement is None:
            return None
        return self._make_sharded_state(
            name,
            local_tensor.view(-1),
            placement,
            global_shape=(mp.numel,),
            global_stride=(1,),
        )

    def make_param_state_dtensor(self, mp: ManagedParam, local_tensor: torch.Tensor):
        state = self.make_param_state(mp, "optimizer", local_tensor)
        return state.dtensor if state is not None else None

    def param_matrix_shard(self, mp: ManagedParam) -> MatrixShard | None:
        shard_sizes = self.param_shard_sizes(mp)
        if sum(shard_sizes) != mp.numel:
            return None
        if not self._param_segments_are_rank_ordered(mp):
            return None
        return MatrixShard(dims=(0,), local_units=shard_sizes_to_matrix_local_units(shard_sizes))

    def param_shard_sizes(self, mp: ManagedParam) -> tuple[int, ...]:
        sizes = []
        for rank in range(len(self.plan.shard_sizes)):
            local_numel = 0
            for segment in self.plan.local_segments(rank):
                global_start = max(mp.offset, segment.global_start)
                global_end = min(mp.end, segment.global_end)
                if global_start < global_end:
                    local_numel += global_end - global_start
            sizes.append(local_numel)
        return tuple(sizes)

    def _param_segments_are_rank_ordered(self, mp: ManagedParam) -> bool:
        cursor = 0
        for rank in range(len(self.plan.shard_sizes)):
            intervals = []
            for segment in self.plan.local_segments(rank):
                global_start = max(mp.offset, segment.global_start)
                global_end = min(mp.end, segment.global_end)
                if global_start < global_end:
                    intervals.append((global_start - mp.offset, global_end - mp.offset))
            if not intervals:
                continue
            if len(intervals) != 1:
                return False
            start, end = intervals[0]
            if start != cursor:
                return False
            cursor = end
        return cursor == mp.numel

    def _sync_replicated_grad_shard(self, local_grad_shard: torch.Tensor) -> torch.Tensor:
        if self.replicate_group is None or self.replicate_world_size <= 1:
            return local_grad_shard
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return local_grad_shard
        torch.distributed.all_reduce(local_grad_shard, group=self.replicate_group)
        if self.divide_grads_by_world:
            local_grad_shard.div_(self.replicate_world_size)
        return local_grad_shard


def _release_workspace_after_wait(
    handle: MatrixTensorCollectiveHandle,
    lease: CommWorkspaceLease | ElasticRankChunkWorkspaceLease,
) -> MatrixTensorCollectiveHandle:
    def wait() -> torch.Tensor:
        try:
            return handle.wait()
        finally:
            lease.release()

    return MatrixTensorCollectiveHandle(
        handle.tensor,
        wait,
        collective_kind=handle.collective_kind,
        collective_backend=handle.collective_backend,
        collective_impl=handle.collective_impl,
        collective_numel=handle.collective_numel,
        collective_bytes=handle.collective_bytes,
        collective_count=handle.collective_count,
    )
