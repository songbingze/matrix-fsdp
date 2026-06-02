from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import tempfile
import time
from functools import partial
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.checkpoint import checkpoint

from matrix_fsdp.grad_bucket import BucketParamGrad, MatrixGradBucket, classify_copy_in_layout, fill_reduce_scatter_input
from matrix_fsdp.layout import LayoutSegment
from matrix_fsdp.managed_param import ManagedParam
from matrix_fsdp import (
    DEFAULT_MUON_ADJUST_LR_FN,
    MixedMuonAdamWOptimizer,
    PrefetchProfileResult,
    MatrixFSDPOptimizer,
    summarize_param_groups,
    clear_global_full_param_buffer_pool,
    fully_shard as matrix_fully_shard_api,
    fsdp2_chunk_plan,
    make_muon_shard_aware_group_planner,
    matrix_fully_shard,
)

try:
    from torch.distributed.fsdp import fully_shard as torch_fully_shard
except ImportError:  # pragma: no cover - depends on local PyTorch version.
    torch_fully_shard = None


DEFAULT_FAST_MATRIX_MODE = "matrix_default"
DEFAULT_FAST_MATRIX_MODE_ALIASES = (
    DEFAULT_FAST_MATRIX_MODE,
    "matrix_fsdp2_copy_in",
    "matrix_fast_default",
    "matrix_prefetch_fsdp2_schedule_bucket_copy_in_no_saved_hooks",
)
DEFAULT_MEMORY_ACCOUNTING_MODES = (
    "fsdp2",
    DEFAULT_FAST_MATRIX_MODE,
    "matrix_memory_capped",
    "matrix_zero_copy_grad_bucket",
    "matrix_no_prefetch",
)
DEFAULT_FULLY_SHARD_API_COMPARE_MODES = (
    "fsdp2_api",
    "matrix_api",
)

DEFAULT_MODES = (
    "fsdp2",
    DEFAULT_FAST_MATRIX_MODE,
)

EXPERIMENTAL_MODES = (
    "eager",
    "matrix_zero_copy_grad_bucket",
    "matrix_auto_finalize",
    "matrix_prefetch",
    "matrix_prefetch_cap1",
    "matrix_prefetch_cap2",
    "matrix_prefetch_fsdp2_schedule_bucket_copy_in",
    "matrix_prefetch_fsdp2_schedule_bucket_reduce_scatter_wait_rs",
    "matrix_prefetch_fsdp2_schedule_bucket_copy_in_wait_rs",
    "matrix_prefetch_fsdp2_schedule_bucket_copy_in_mem_cap1",
    "matrix_prefetch_late_backward_bucket_reduce_scatter",
    "matrix_prefetch_late_backward_bucket_copy_in",
    "matrix_prefetch_late_backward_fsdp2_chunk",
    "matrix_prefetch_adaptive",
    "matrix_prefetch_adaptive_bucket_reduce_scatter",
    "matrix_prefetch_adaptive_no_saved_hooks",
    "matrix_prefetch_adaptive_per_param",
    "matrix_prefetch_adaptive_per_param_allreduce",
    "matrix_prefetch_profile_guided",
    "matrix_owner_muon",
    "matrix_owner_muon_pre_backward",
    "matrix_owner_muon_role_greedy",
    "matrix_owner_muon_role_greedy_pre_backward",
    "matrix_owner_muon_role_greedy_matrix_all_gather",
    "matrix_owner_muon_role_greedy_matrix_all_gather_pre_backward",
    "matrix_owner_muon_role_greedy_custom_collective",
    "matrix_owner_muon_role_greedy_custom_collective_pre_backward",
)

@dataclass(frozen=True)
class FSDP2CompareConfig:
    world_size: int = 2
    device: str = "cpu"
    modes: tuple[str, ...] = DEFAULT_MODES
    unit: str = "linear"
    layers: int = 4
    hidden: int = 512
    intermediate: int = 2048
    batch_size: int = 8
    optimizer: str = "sgd"
    dtype: str = "float32"
    model: str = "mlp"
    seq_len: int = 128
    heads: int = 8
    warmup_steps: int = 5
    profile_steps: int = 3
    profile_memory_limit_mb: float = 0.0
    matrix_max_active_full_param_buffers: int | None = None
    matrix_max_active_full_param_numel: int | None = None
    matrix_max_active_full_param_memory_mb: float = 0.0
    activation_checkpoint: bool = False
    checkpoint_use_reentrant: bool = False
    activation_checkpoint_wrapper: bool = False
    steps: int = 20
    init_file: str = ""
    output_file: str = ""
    memory_by_rank: bool = False
    empty_cache_after_warmup: bool = False


@dataclass(frozen=True)
class FSDP2CompareRow:
    mode: str
    unit: str
    device: str
    world_size: int
    layers: int
    hidden: int
    intermediate: int
    batch_size: int
    optimizer: str
    dtype: str
    param_count: int
    avg_step_ms: float
    peak_memory_mb: float
    model: str = "mlp"
    seq_len: int = 128
    heads: int = 8
    prefetch_budget: str = ""


@dataclass(frozen=True)
class FSDP2MemoryTraceRow:
    mode: str
    unit: str
    device: str
    world_size: int
    optimizer: str
    dtype: str
    phase: str
    current_memory_mb: float
    peak_memory_mb: float
    rank: int = -1
    current_reserved_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    model: str = "mlp"
    seq_len: int = 128
    prefetch_budget: str = ""
    active_full_param_buffers: int = 0
    full_param_buffer_mb: float = 0.0
    local_shard_mb: float = 0.0
    full_grad_buffer_mb: float = 0.0
    grad_bucket_mb: float = 0.0
    local_grad_shard_mb: float = 0.0
    optimizer_state_mb: float = 0.0
    pending_backward_reduces: int = 0


@dataclass(frozen=True)
class FSDP2PhaseTimingRow:
    mode: str
    unit: str
    device: str
    world_size: int
    optimizer: str
    dtype: str
    avg_zero_grad_ms: float
    avg_forward_ms: float
    avg_backward_ms: float
    avg_step_ms: float
    avg_total_ms: float
    peak_memory_mb: float
    model: str = "mlp"
    seq_len: int = 128
    prefetch_budget: str = ""


@dataclass(frozen=True)
class MatrixRuntimeCommunicationRow:
    mode: str
    model: str
    unit: str
    device: str
    world_size: int
    optimizer: str
    dtype: str
    param_groups: int
    gather_backend_counts: str
    resolved_custom_allgatherv_counts: str
    rank_chunk_fast_paths: int
    packed_full_order: int
    max_segment_count: int
    max_segments_per_rank: int
    max_padding_waste_ratio: float
    max_owner_imbalance_ratio: float
    workspace_preferred_kind_counts: str
    max_workspace_preferred_numel: int
    max_workspace_padded_numel: int
    max_workspace_padding_waste_ratio: float
    workspace_acquires: int
    workspace_reuses: int
    workspace_allocates: int
    max_workspace_allocated_numel: int
    min_shard_size: int
    max_shard_size: int
    seq_len: int = 128


@dataclass(frozen=True)
class FSDP2CorrectnessResult:
    reference_mode: str
    candidate_mode: str
    unit: str
    device: str
    world_size: int
    layers: int
    hidden: int
    intermediate: int
    batch_size: int
    optimizer: str
    dtype: str
    steps: int
    max_loss_abs_diff: float
    max_output_abs_diff: float
    max_grad_abs_diff: float
    max_grad_rel_diff: float
    grad_checked_param_count: int
    grad_mismatched_param_count: int
    grad_mismatched_param_names: tuple[str, ...] = ()
    model: str = "mlp"
    seq_len: int = 128
    heads: int = 8


@dataclass(frozen=True)
class CopyInBenchmarkRow:
    backend: str
    resolved_layout: str
    layout: str
    device: str
    dtype: str
    world_size: int
    param_count: int
    param_numel: int
    total_numel: int
    avg_ms: float
    bandwidth_gb_s: float


@dataclass(frozen=True)
class ParamMaterializationBenchmarkRow:
    backend: str
    device: str
    dtype: str
    param_count: int
    param_numel: int
    total_numel: int
    avg_ms: float
    bandwidth_gb_s: float


class MLPBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.act = nn.GELU()
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class SplitGELUMLPBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        if intermediate % 2 != 0:
            raise ValueError(f"intermediate={intermediate} must be divisible by 2 for split GELU MLP.")
        half_intermediate = intermediate // 2
        self.up0 = nn.Linear(hidden, half_intermediate, bias=False)
        self.up1 = nn.Linear(hidden, half_intermediate, bias=False)
        self.act = nn.GELU()
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = torch.cat((self.up0(x), self.up1(x)), dim=-1)
        return self.down(self.act(up))


class MLPStack(nn.Module):
    def __init__(
        self,
        layers: int,
        hidden: int,
        intermediate: int,
        *,
        activation_checkpoint: bool = False,
        checkpoint_use_reentrant: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MLPBlock(hidden, intermediate) for _ in range(layers)])
        self.activation_checkpoint = activation_checkpoint
        self.checkpoint_use_reentrant = checkpoint_use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            layer_out = _checkpoint_layer(layer, x, self.checkpoint_use_reentrant) if self.activation_checkpoint else layer(x)
            x = x + layer_out
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, heads: int) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}.")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = MLPBlock(hidden, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        qkv = self.qkv(self.norm1(x))
        qkv = qkv.view(batch, seq_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        x = x + self.proj(attn)
        return x + self.mlp(self.norm2(x))


class TransformerStack(nn.Module):
    def __init__(
        self,
        layers: int,
        hidden: int,
        intermediate: int,
        heads: int,
        *,
        activation_checkpoint: bool = False,
        checkpoint_use_reentrant: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(hidden, intermediate, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.activation_checkpoint = activation_checkpoint
        self.checkpoint_use_reentrant = checkpoint_use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = _checkpoint_layer(layer, x, self.checkpoint_use_reentrant) if self.activation_checkpoint else layer(x)
        return self.norm(x)


class SplitQKVTransformerBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, heads: int) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}.")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        # Register large 2D params first so matrix-owner planning sees the
        # intended q/k/v/proj/up0/up1/down order before tiny norm params.
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = SplitGELUMLPBlock(hidden, intermediate)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        normed = self.norm1(x)
        query = self.q(normed).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        key = self.k(normed).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        value = self.v(normed).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        x = x + self.proj(attn)
        return x + self.mlp(self.norm2(x))


class SplitQKVTransformerStack(nn.Module):
    def __init__(
        self,
        layers: int,
        hidden: int,
        intermediate: int,
        heads: int,
        *,
        activation_checkpoint: bool = False,
        checkpoint_use_reentrant: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([SplitQKVTransformerBlock(hidden, intermediate, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.activation_checkpoint = activation_checkpoint
        self.checkpoint_use_reentrant = checkpoint_use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = _checkpoint_layer(layer, x, self.checkpoint_use_reentrant) if self.activation_checkpoint else layer(x)
        return self.norm(x)


def _checkpoint_layer(layer: nn.Module, x: torch.Tensor, use_reentrant: bool) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return layer(x)
    return checkpoint(layer, x, use_reentrant=use_reentrant)


def run_compare_benchmark(config: FSDP2CompareConfig) -> tuple[FSDP2CompareRow, ...]:
    if config.world_size <= 0:
        raise ValueError(f"world_size must be positive, got {config.world_size}.")
    if config.steps <= 0:
        raise ValueError(f"steps must be positive, got {config.steps}.")
    if config.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {config.warmup_steps}.")
    if config.profile_steps < 0:
        raise ValueError(f"profile_steps must be non-negative, got {config.profile_steps}.")
    if config.profile_memory_limit_mb < 0:
        raise ValueError(f"profile_memory_limit_mb must be non-negative, got {config.profile_memory_limit_mb}.")
    _validate_model_config(config)
    if config.device == "cuda" and torch.cuda.device_count() < config.world_size:
        raise RuntimeError(
            f"CUDA benchmark requires at least {config.world_size} devices, "
            f"found {torch.cuda.device_count()}."
        )
    if _uses_fsdp2(config.modes) and torch_fully_shard is None:
        raise RuntimeError("torch.distributed.fsdp.fully_shard is not available in this PyTorch build.")

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for mode_index, mode in enumerate(config.modes):
            init_file = config.init_file or os.path.join(tmpdir, f"fsdp2_compare_init_{mode_index}")
            output_file = (
                config.output_file
                if config.output_file and len(config.modes) == 1
                else os.path.join(tmpdir, f"fsdp2_compare_rows_{mode_index}.json")
            )
            worker_config = FSDP2CompareConfig(
                **{
                    **asdict(config),
                    "modes": (mode,),
                    "init_file": init_file,
                    "output_file": output_file,
                }
            )
            mp.spawn(_benchmark_worker, args=(worker_config,), nprocs=config.world_size, join=True)
            with open(output_file, encoding="utf-8") as result_file:
                rows.extend(json.load(result_file))
    return tuple(FSDP2CompareRow(**row) for row in rows)


def run_correctness_check(
    config: FSDP2CompareConfig,
    *,
    reference_mode: str = "fsdp2",
    candidate_mode: str = "matrix_prefetch",
) -> FSDP2CorrectnessResult:
    if torch_fully_shard is None and reference_mode.startswith("fsdp2"):
        raise RuntimeError("torch.distributed.fsdp.fully_shard is not available in this PyTorch build.")
    _validate_model_config(config)
    if config.device == "cuda" and torch.cuda.device_count() < config.world_size:
        raise RuntimeError(
            f"CUDA correctness check requires at least {config.world_size} devices, "
            f"found {torch.cuda.device_count()}."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = config.init_file or os.path.join(tmpdir, "fsdp2_correctness_init")
        output_file = config.output_file or os.path.join(tmpdir, "fsdp2_correctness_result.json")
        worker_config = FSDP2CompareConfig(**{**asdict(config), "init_file": init_file, "output_file": output_file})
        mp.spawn(
            _correctness_worker,
            args=(worker_config, reference_mode, candidate_mode),
            nprocs=config.world_size,
            join=True,
        )
        with open(output_file, encoding="utf-8") as result_file:
            result = json.load(result_file)
    return FSDP2CorrectnessResult(**result)


def run_memory_trace(config: FSDP2CompareConfig) -> tuple[FSDP2MemoryTraceRow, ...]:
    if config.device != "cuda":
        raise RuntimeError("Memory tracing currently requires --device cuda.")
    _validate_model_config(config)
    if torch.cuda.device_count() < config.world_size:
        raise RuntimeError(
            f"CUDA memory trace requires at least {config.world_size} devices, "
            f"found {torch.cuda.device_count()}."
        )
    if _uses_fsdp2(config.modes) and torch_fully_shard is None:
        raise RuntimeError("torch.distributed.fsdp.fully_shard is not available in this PyTorch build.")

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for mode_index, mode in enumerate(config.modes):
            init_file = config.init_file or os.path.join(tmpdir, f"fsdp2_memory_trace_init_{mode_index}")
            output_file = (
                config.output_file
                if config.output_file and len(config.modes) == 1
                else os.path.join(tmpdir, f"fsdp2_memory_trace_rows_{mode_index}.json")
            )
            worker_config = FSDP2CompareConfig(
                **{
                    **asdict(config),
                    "modes": (mode,),
                    "init_file": init_file,
                    "output_file": output_file,
                }
            )
            mp.spawn(_memory_trace_worker, args=(worker_config,), nprocs=config.world_size, join=True)
            with open(output_file, encoding="utf-8") as result_file:
                rows.extend(json.load(result_file))
    return tuple(FSDP2MemoryTraceRow(**row) for row in rows)


def run_phase_timing(config: FSDP2CompareConfig) -> tuple[FSDP2PhaseTimingRow, ...]:
    if config.world_size <= 0:
        raise ValueError(f"world_size must be positive, got {config.world_size}.")
    if config.steps <= 0:
        raise ValueError(f"steps must be positive, got {config.steps}.")
    if config.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {config.warmup_steps}.")
    _validate_model_config(config)
    if config.device == "cuda" and torch.cuda.device_count() < config.world_size:
        raise RuntimeError(
            f"CUDA phase timing requires at least {config.world_size} devices, "
            f"found {torch.cuda.device_count()}."
        )
    if _uses_fsdp2(config.modes) and torch_fully_shard is None:
        raise RuntimeError("torch.distributed.fsdp.fully_shard is not available in this PyTorch build.")

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for mode_index, mode in enumerate(config.modes):
            init_file = config.init_file or os.path.join(tmpdir, f"fsdp2_phase_timing_init_{mode_index}")
            output_file = (
                config.output_file
                if config.output_file and len(config.modes) == 1
                else os.path.join(tmpdir, f"fsdp2_phase_timing_rows_{mode_index}.json")
            )
            worker_config = FSDP2CompareConfig(
                **{
                    **asdict(config),
                    "modes": (mode,),
                    "init_file": init_file,
                    "output_file": output_file,
                }
            )
            mp.spawn(_phase_timing_worker, args=(worker_config,), nprocs=config.world_size, join=True)
            with open(output_file, encoding="utf-8") as result_file:
                rows.extend(json.load(result_file))
    return tuple(FSDP2PhaseTimingRow(**row) for row in rows)


def run_runtime_communication_summary(config: FSDP2CompareConfig) -> tuple[MatrixRuntimeCommunicationRow, ...]:
    if config.world_size <= 0:
        raise ValueError(f"world_size must be positive, got {config.world_size}.")
    _validate_model_config(config)
    if config.device == "cuda" and torch.cuda.device_count() < config.world_size:
        raise RuntimeError(
            f"CUDA runtime summary requires at least {config.world_size} devices, "
            f"found {torch.cuda.device_count()}."
        )
    if _uses_fsdp2(config.modes) and torch_fully_shard is None:
        raise RuntimeError("torch.distributed.fsdp.fully_shard is not available in this PyTorch build.")

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for mode_index, mode in enumerate(config.modes):
            init_file = config.init_file or os.path.join(tmpdir, f"fsdp2_runtime_summary_init_{mode_index}")
            output_file = (
                config.output_file
                if config.output_file and len(config.modes) == 1
                else os.path.join(tmpdir, f"fsdp2_runtime_summary_rows_{mode_index}.json")
            )
            worker_config = FSDP2CompareConfig(
                **{
                    **asdict(config),
                    "modes": (mode,),
                    "init_file": init_file,
                    "output_file": output_file,
                }
            )
            mp.spawn(_runtime_communication_summary_worker, args=(worker_config,), nprocs=config.world_size, join=True)
            with open(output_file, encoding="utf-8") as result_file:
                rows.extend(json.load(result_file))
    return tuple(MatrixRuntimeCommunicationRow(**row) for row in rows)


def format_compare_table(rows: Sequence[FSDP2CompareRow]) -> str:
    headers = (
        "mode",
        "model",
        "unit",
        "device",
        "world",
        "layers",
        "hidden",
        "inter",
        "seq",
        "batch",
        "optim",
        "dtype",
        "params",
        "budget",
        "avg_step_ms",
        "peak_mem_mb",
    )
    table_rows = [
        (
            row.mode,
            row.model,
            row.unit,
            row.device,
            str(row.world_size),
            str(row.layers),
            str(row.hidden),
            str(row.intermediate),
            str(row.seq_len if _is_transformer_model(row.model) else "-"),
            str(row.batch_size),
            row.optimizer,
            row.dtype,
            str(row.param_count),
            row.prefetch_budget,
            f"{row.avg_step_ms:.3f}",
            f"{row.peak_memory_mb:.1f}",
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(table_row, widths) for table_row in table_rows)
    return "\n".join(lines)


def format_memory_trace_table(rows: Sequence[FSDP2MemoryTraceRow]) -> str:
    headers = (
        "mode",
        "rank",
        "model",
        "unit",
        "device",
        "world",
        "optim",
        "dtype",
        "seq",
        "phase",
        "current_mem_mb",
        "peak_mem_mb",
        "reserved_mb",
        "peak_reserved_mb",
        "budget",
        "full_bufs",
        "full_mb",
        "local_mb",
        "full_grad_mb",
        "bucket_mb",
        "local_grad_mb",
        "opt_state_mb",
        "pending_rs",
    )
    table_rows = [
        (
            row.mode,
            "max" if row.rank < 0 else str(row.rank),
            row.model,
            row.unit,
            row.device,
            str(row.world_size),
            row.optimizer,
            row.dtype,
            str(row.seq_len if _is_transformer_model(row.model) else "-"),
            row.phase,
            f"{row.current_memory_mb:.1f}",
            f"{row.peak_memory_mb:.1f}",
            f"{row.current_reserved_mb:.1f}",
            f"{row.peak_reserved_mb:.1f}",
            row.prefetch_budget,
            str(row.active_full_param_buffers),
            f"{row.full_param_buffer_mb:.1f}",
            f"{row.local_shard_mb:.1f}",
            f"{row.full_grad_buffer_mb:.1f}",
            f"{row.grad_bucket_mb:.1f}",
            f"{row.local_grad_shard_mb:.1f}",
            f"{row.optimizer_state_mb:.1f}",
            str(row.pending_backward_reduces),
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(table_row, widths) for table_row in table_rows)
    return "\n".join(lines)


def format_phase_timing_table(rows: Sequence[FSDP2PhaseTimingRow]) -> str:
    headers = (
        "mode",
        "model",
        "unit",
        "device",
        "world",
        "optim",
        "dtype",
        "seq",
        "budget",
        "zero_ms",
        "fwd_ms",
        "bwd_ms",
        "step_ms",
        "total_ms",
        "peak_mem_mb",
    )
    table_rows = [
        (
            row.mode,
            row.model,
            row.unit,
            row.device,
            str(row.world_size),
            row.optimizer,
            row.dtype,
            str(row.seq_len if _is_transformer_model(row.model) else "-"),
            row.prefetch_budget,
            f"{row.avg_zero_grad_ms:.3f}",
            f"{row.avg_forward_ms:.3f}",
            f"{row.avg_backward_ms:.3f}",
            f"{row.avg_step_ms:.3f}",
            f"{row.avg_total_ms:.3f}",
            f"{row.peak_memory_mb:.1f}",
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(table_row, widths) for table_row in table_rows)
    return "\n".join(lines)


def format_runtime_communication_table(rows: Sequence[MatrixRuntimeCommunicationRow]) -> str:
    headers = (
        "mode",
        "model",
        "unit",
        "device",
        "world",
        "optim",
        "dtype",
        "seq",
        "groups",
        "gather_backends",
        "custom_impls",
        "chunk_fast",
        "full_order",
        "max_segments",
        "max_segments_rank",
        "pad_waste",
        "imbalance",
        "workspace_kind",
        "workspace_numel",
        "workspace_padded",
        "workspace_waste",
        "workspace_acq",
        "workspace_reuse",
        "workspace_alloc",
        "workspace_alloc_numel",
        "min_shard",
        "max_shard",
    )
    table_rows = [
        (
            row.mode,
            row.model,
            row.unit,
            row.device,
            str(row.world_size),
            row.optimizer,
            row.dtype,
            str(row.seq_len if _is_transformer_model(row.model) else "-"),
            str(row.param_groups),
            row.gather_backend_counts,
            row.resolved_custom_allgatherv_counts,
            str(row.rank_chunk_fast_paths),
            str(row.packed_full_order),
            str(row.max_segment_count),
            str(row.max_segments_per_rank),
            f"{row.max_padding_waste_ratio:.3f}",
            f"{row.max_owner_imbalance_ratio:.3f}",
            row.workspace_preferred_kind_counts,
            str(row.max_workspace_preferred_numel),
            str(row.max_workspace_padded_numel),
            f"{row.max_workspace_padding_waste_ratio:.3f}",
            str(row.workspace_acquires),
            str(row.workspace_reuses),
            str(row.workspace_allocates),
            str(row.max_workspace_allocated_numel),
            str(row.min_shard_size),
            str(row.max_shard_size),
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(table_row, widths) for table_row in table_rows)
    return "\n".join(lines)


def format_correctness_result(result: FSDP2CorrectnessResult) -> str:
    text = (
        f"{result.candidate_mode} vs {result.reference_mode} "
        f"({result.model}, unit={result.unit}): "
        f"loss max abs diff={result.max_loss_abs_diff:.6g}, "
        f"output max abs diff={result.max_output_abs_diff:.6g}, "
        f"grad max abs diff={result.max_grad_abs_diff:.6g}, "
        f"grad max rel diff={result.max_grad_rel_diff:.6g}, "
        f"grad checked params={result.grad_checked_param_count}, "
        f"grad mismatched params={result.grad_mismatched_param_count}"
    )
    if result.grad_mismatched_param_names:
        text += ", grad mismatched names=" + ",".join(result.grad_mismatched_param_names)
    return text


def run_copyin_benchmark(
    *,
    device: str,
    dtype: str,
    world_size: int,
    param_count: int,
    param_numel: int,
    steps: int,
    warmup_steps: int,
    layout: str,
    backends: Sequence[str],
) -> tuple[CopyInBenchmarkRow, ...]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA copy-in benchmark requires CUDA.")
    torch_device = torch.device("cuda", 0) if device == "cuda" else torch.device("cpu")
    torch_dtype = getattr(torch, dtype)
    rows = []
    for backend in backends:
        bucket, packed = _make_copyin_bucket(
            device=torch_device,
            dtype=torch_dtype,
            world_size=world_size,
            param_count=param_count,
            param_numel=param_numel,
            layout=layout,
        )
        fill_reduce_scatter_input(bucket, packed, backend=backend)
        _synchronize(device)
        for _ in range(warmup_steps):
            fill_reduce_scatter_input(bucket, packed, backend=backend)
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(steps):
            fill_reduce_scatter_input(bucket, packed, backend=backend)
        _synchronize(device)
        avg_ms = (time.perf_counter() - start) * 1000.0 / max(steps, 1)
        bytes_copied = bucket.total_numel * torch.empty((), dtype=torch_dtype).element_size()
        bandwidth = bytes_copied / (avg_ms / 1000.0) / 1e9 if avg_ms > 0 else float("inf")
        rows.append(
            CopyInBenchmarkRow(
                backend=backend,
                resolved_layout=classify_copy_in_layout(bucket),
                layout=layout,
                device=device,
                dtype=dtype,
                world_size=world_size,
                param_count=param_count,
                param_numel=param_numel,
                total_numel=bucket.total_numel,
                avg_ms=avg_ms,
                bandwidth_gb_s=bandwidth,
            )
        )
    return tuple(rows)


def format_copyin_table(rows: Sequence[CopyInBenchmarkRow]) -> str:
    headers = (
        "backend",
        "resolved",
        "layout",
        "device",
        "dtype",
        "world",
        "params",
        "param_numel",
        "total_numel",
        "avg_ms",
        "GB/s",
    )
    table_rows = [
        (
            row.backend,
            row.resolved_layout,
            row.layout,
            row.device,
            row.dtype,
            str(row.world_size),
            str(row.param_count),
            str(row.param_numel),
            str(row.total_numel),
            f"{row.avg_ms:.3f}",
            f"{row.bandwidth_gb_s:.2f}",
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(table_row, widths) for table_row in table_rows)
    return "\n".join(lines)


def run_param_materialization_benchmark(
    *,
    device: str,
    dtype: str,
    param_count: int,
    param_numel: int,
    steps: int,
    warmup_steps: int,
    backends: Sequence[str],
) -> tuple[ParamMaterializationBenchmarkRow, ...]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA parameter materialization benchmark requires CUDA.")
    if param_count <= 0:
        raise ValueError("param_count must be positive.")
    if param_numel <= 0:
        raise ValueError("param_numel must be positive.")
    torch_device = torch.device("cuda", 0) if device == "cuda" else torch.device("cpu")
    torch_dtype = getattr(torch, dtype)
    total_numel = param_count * param_numel
    full_buffer = torch.randn(total_numel, device=torch_device, dtype=torch_dtype)
    view_params = [nn.Parameter(torch.empty(0, device=torch_device, dtype=torch_dtype)) for _ in range(param_count)]
    copy_params = [
        nn.Parameter(torch.empty(param_numel, device=torch_device, dtype=torch_dtype))
        for _ in range(param_count)
    ]

    rows = []
    for backend in backends:
        if backend == "view_assign":
            op = lambda: _materialize_params_as_views(view_params, full_buffer, param_numel)
            bytes_touched = total_numel * torch.empty((), dtype=torch_dtype).element_size()
        elif backend == "copy_out":
            op = lambda: _materialize_params_by_copy(copy_params, full_buffer, param_numel)
            bytes_touched = total_numel * torch.empty((), dtype=torch_dtype).element_size()
        else:
            raise ValueError(f"Unknown param materialization backend: {backend}.")
        op()
        _synchronize(device)
        for _ in range(warmup_steps):
            op()
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(steps):
            op()
        _synchronize(device)
        avg_ms = (time.perf_counter() - start) * 1000.0 / max(steps, 1)
        bandwidth = bytes_touched / (avg_ms / 1000.0) / 1e9 if avg_ms > 0 else float("inf")
        rows.append(
            ParamMaterializationBenchmarkRow(
                backend=backend,
                device=device,
                dtype=dtype,
                param_count=param_count,
                param_numel=param_numel,
                total_numel=total_numel,
                avg_ms=avg_ms,
                bandwidth_gb_s=bandwidth,
            )
        )
    return tuple(rows)


def _materialize_params_as_views(
    params: Sequence[nn.Parameter],
    full_buffer: torch.Tensor,
    param_numel: int,
) -> None:
    for index, param in enumerate(params):
        start = index * param_numel
        param.data = full_buffer[start : start + param_numel]


def _materialize_params_by_copy(
    params: Sequence[nn.Parameter],
    full_buffer: torch.Tensor,
    param_numel: int,
) -> None:
    for index, param in enumerate(params):
        start = index * param_numel
        param.data.copy_(full_buffer[start : start + param_numel])


def format_param_materialization_table(rows: Sequence[ParamMaterializationBenchmarkRow]) -> str:
    headers = (
        "backend",
        "device",
        "dtype",
        "params",
        "param_numel",
        "total_numel",
        "avg_ms",
        "GB/s",
    )
    table_rows = [
        (
            row.backend,
            row.device,
            row.dtype,
            str(row.param_count),
            str(row.param_numel),
            str(row.total_numel),
            f"{row.avg_ms:.3f}",
            f"{row.bandwidth_gb_s:.2f}",
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(table_row, widths) for table_row in table_rows)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare MatrixFSDP step time against PyTorch FSDP2.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--mode",
        action="append",
        choices=(
            "eager",
            "fsdp2",
            "fsdp2_api",
            "fsdp2_no_root",
            "matrix_default",
            "matrix_api",
            "matrix_fsdp2_copy_in",
            "matrix_zero_copy_grad_bucket",
            "matrix_fast_default",
            "matrix_memory_capped",
            "matrix_no_prefetch",
            "matrix_optimizer_finalize",
            "matrix_auto_finalize",
            "matrix_prefetch",
            "matrix_prefetch_cap1",
            "matrix_prefetch_cap2",
            "matrix_prefetch_fsdp2_schedule_bucket_reduce_scatter",
            "matrix_prefetch_fsdp2_schedule_bucket_copy_in",
            "matrix_prefetch_fsdp2_schedule_bucket_copy_in_no_saved_hooks",
            "matrix_prefetch_fsdp2_schedule_bucket_reduce_scatter_wait_rs",
            "matrix_prefetch_fsdp2_schedule_bucket_copy_in_wait_rs",
            "matrix_prefetch_fsdp2_schedule_bucket_copy_in_mem_cap1",
            "matrix_prefetch_late_backward_bucket_reduce_scatter",
            "matrix_prefetch_late_backward_bucket_copy_in",
            "matrix_prefetch_late_backward_fsdp2_chunk",
            "matrix_prefetch_adaptive",
            "matrix_prefetch_adaptive_bucket_reduce_scatter",
            "matrix_prefetch_adaptive_no_saved_hooks",
            "matrix_prefetch_adaptive_per_param",
            "matrix_prefetch_adaptive_per_param_allreduce",
            "matrix_prefetch_profile_guided",
            "matrix_owner_muon",
            "matrix_owner_muon_pre_backward",
            "matrix_owner_muon_role_greedy",
            "matrix_owner_muon_role_greedy_pre_backward",
            "matrix_owner_muon_role_greedy_matrix_all_gather",
            "matrix_owner_muon_role_greedy_matrix_all_gather_pre_backward",
            "matrix_owner_muon_role_greedy_custom_collective",
            "matrix_owner_muon_role_greedy_custom_collective_pre_backward",
        ),
        help="Mode to benchmark. Can be passed multiple times.",
    )
    parser.add_argument("--unit", choices=("linear", "block"), default="linear")
    parser.add_argument("--model", choices=("mlp", "transformer", "transformer_split_qkv"), default="mlp")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimizer", choices=("sgd", "adamw", "muon"), default="sgd")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--profile-memory-limit-mb", type=float, default=0.0)
    parser.add_argument("--matrix-max-active-full-param-buffers", type=int, default=None)
    parser.add_argument("--matrix-max-active-full-param-numel", type=int, default=None)
    parser.add_argument("--matrix-max-active-full-param-memory-mb", type=float, default=0.0)
    parser.add_argument("--activation-checkpoint", action="store_true", help="Checkpoint each benchmark block/layer.")
    parser.add_argument(
        "--activation-checkpoint-wrapper",
        action="store_true",
        help=(
            "Apply activation checkpointing through "
            "torch.distributed.algorithms._checkpoint.checkpoint_wrapper."
        ),
    )
    parser.add_argument(
        "--checkpoint-use-reentrant",
        action="store_true",
        help="Use reentrant activation checkpointing. The default is PyTorch's recommended non-reentrant path.",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--correctness", action="store_true", help="Compare candidate losses/outputs against FSDP2.")
    parser.add_argument("--candidate-mode", default=DEFAULT_FAST_MATRIX_MODE)
    parser.add_argument("--memory-trace", action="store_true", help="Print per-phase CUDA memory for one measured step.")
    parser.add_argument(
        "--memory-by-rank",
        action="store_true",
        help="With --memory-trace, print local memory/accounting rows for every rank instead of cross-rank maxima.",
    )
    parser.add_argument(
        "--empty-cache-after-warmup",
        action="store_true",
        help="For benchmark diagnosis, call torch.cuda.empty_cache() after warmup and before measuring/tracing.",
    )
    parser.add_argument(
        "--memory-accounting",
        action="store_true",
        help=(
            "Print memory trace with MatrixFSDP runtime buffer accounting for "
            "fsdp2/copy-in/capped/zero-copy-grad/no-prefetch modes."
        ),
    )
    parser.add_argument("--phase-timing", action="store_true", help="Print synchronized per-phase timing for each mode.")
    parser.add_argument(
        "--runtime-summary",
        action="store_true",
        help="Print MatrixFSDP communication/layout summary for each benchmark mode.",
    )
    parser.add_argument("--copyin-bench", action="store_true", help="Benchmark MatrixFSDP grad bucket copy-in only.")
    parser.add_argument("--copyin-layout", choices=("flat", "matrix", "chunk_cat"), default="flat")
    parser.add_argument(
        "--copyin-backend",
        action="append",
        choices=("auto", "flat_cat", "foreach_copy", "chunk_cat", "segment_copy"),
    )
    parser.add_argument("--copyin-param-count", type=int, default=64)
    parser.add_argument("--copyin-param-numel", type=int, default=1_048_576)
    parser.add_argument(
        "--param-materialization-bench",
        action="store_true",
        help="Benchmark full-param view assignment against per-param copy-out materialization.",
    )
    parser.add_argument(
        "--param-materialization-backend",
        action="append",
        choices=("view_assign", "copy_out"),
    )
    parser.add_argument("--param-materialization-param-count", type=int, default=64)
    parser.add_argument("--param-materialization-param-numel", type=int, default=1_048_576)
    args = parser.parse_args(argv)
    if args.activation_checkpoint_wrapper and not args.activation_checkpoint:
        parser.error("--activation-checkpoint-wrapper requires --activation-checkpoint.")

    config = FSDP2CompareConfig(
        world_size=args.world_size,
        device=args.device,
        modes=tuple(args.mode or DEFAULT_MODES),
        unit=args.unit,
        layers=args.layers,
        hidden=args.hidden,
        intermediate=args.intermediate,
        model=args.model,
        seq_len=args.seq_len,
        heads=args.heads,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        dtype=args.dtype,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        profile_memory_limit_mb=args.profile_memory_limit_mb,
        matrix_max_active_full_param_buffers=args.matrix_max_active_full_param_buffers,
        matrix_max_active_full_param_numel=args.matrix_max_active_full_param_numel,
        matrix_max_active_full_param_memory_mb=args.matrix_max_active_full_param_memory_mb,
        activation_checkpoint=args.activation_checkpoint,
        checkpoint_use_reentrant=args.checkpoint_use_reentrant,
        activation_checkpoint_wrapper=args.activation_checkpoint_wrapper,
        steps=args.steps,
        memory_by_rank=args.memory_by_rank,
        empty_cache_after_warmup=args.empty_cache_after_warmup,
    )
    if args.copyin_bench:
        rows = run_copyin_benchmark(
            device=args.device,
            dtype=args.dtype,
            world_size=args.world_size,
            param_count=args.copyin_param_count,
            param_numel=args.copyin_param_numel,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            layout=args.copyin_layout,
            backends=tuple(args.copyin_backend or ("auto", "segment_copy")),
        )
        print(format_copyin_table(rows))
        return 0
    if args.param_materialization_bench:
        rows = run_param_materialization_benchmark(
            device=args.device,
            dtype=args.dtype,
            param_count=args.param_materialization_param_count,
            param_numel=args.param_materialization_param_numel,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            backends=tuple(args.param_materialization_backend or ("view_assign", "copy_out")),
        )
        print(format_param_materialization_table(rows))
        return 0
    if args.correctness:
        result = run_correctness_check(config, candidate_mode=args.candidate_mode)
        print(format_correctness_result(result))
        return 0
    if args.memory_trace or args.memory_accounting:
        trace_config = config
        if args.memory_accounting and not args.mode:
            trace_config = FSDP2CompareConfig(**{**asdict(config), "modes": DEFAULT_MEMORY_ACCOUNTING_MODES})
        rows = run_memory_trace(trace_config)
        print(format_memory_trace_table(rows))
        return 0
    if args.phase_timing:
        rows = run_phase_timing(config)
        print(format_phase_timing_table(rows))
        if args.runtime_summary:
            runtime_rows = run_runtime_communication_summary(config)
            print()
            print(format_runtime_communication_table(runtime_rows))
        return 0
    if args.runtime_summary:
        rows = run_runtime_communication_summary(config)
        print(format_runtime_communication_table(rows))
        return 0
    rows = run_compare_benchmark(config)
    print(format_compare_table(rows))
    return 0


def _benchmark_worker(rank: int, config: FSDP2CompareConfig) -> None:
    _init_benchmark_process_group(rank, config)
    try:
        device = torch.device("cuda", rank) if config.device == "cuda" else torch.device("cpu")
        mesh = _make_mesh(config)
        rows = []
        for mode in config.modes:
            rows.append(_benchmark_mode(mode, config, mesh, device))
            _cleanup_after_mode(config.device)
        if rank == 0:
            with open(config.output_file, "w", encoding="utf-8") as result_file:
                json.dump([asdict(row) for row in rows], result_file, indent=2)
    finally:
        dist.destroy_process_group()


def _make_copyin_bucket(
    *,
    device: torch.device,
    dtype: torch.dtype,
    world_size: int,
    param_count: int,
    param_numel: int,
    layout: str,
) -> tuple[MatrixGradBucket, torch.Tensor]:
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if param_count <= 0:
        raise ValueError("param_count must be positive.")
    if param_numel <= 0:
        raise ValueError("param_numel must be positive.")
    total_numel = param_count * param_numel
    if layout == "flat" and total_numel % world_size != 0:
        raise ValueError("flat copy-in layout requires total_numel to be divisible by world_size.")

    param_grads = []
    offset = 0
    for index in range(param_count):
        grad = torch.randn(param_numel, device=device, dtype=dtype)
        managed_param = ManagedParam(
            fqn=f"p{index}",
            param=nn.Parameter(torch.empty(0, device=device, dtype=dtype), requires_grad=False),
            shape=(param_numel,),
            dtype=dtype,
            device=device,
            numel=param_numel,
            offset=offset,
            end=offset + param_numel,
        )
        param_grads.append(BucketParamGrad(managed_param, grad))
        offset += param_numel

    if layout == "chunk_cat":
        rank_segments = _copyin_chunk_cat_rank_segments(world_size, param_count, param_numel)
        padded_shard_size = param_count * ((param_numel + world_size - 1) // world_size)
        shard_sizes = tuple(padded_shard_size for _ in range(world_size))
    else:
        rank_segments = _copyin_rank_segments(total_numel, world_size, layout)
        shard_sizes = tuple(sum(segment.numel for segment in segments) for segments in rank_segments)
    bucket = MatrixGradBucket(
        param_grads=tuple(param_grads),
        total_numel=total_numel,
        shard_sizes=shard_sizes,
        rank_segments=rank_segments,
    )
    packed = torch.empty(bucket.world_size * bucket.max_shard_size, device=device, dtype=dtype)
    return bucket, packed


def _copyin_rank_segments(total_numel: int, world_size: int, layout: str) -> tuple[tuple[LayoutSegment, ...], ...]:
    if layout not in ("flat", "matrix"):
        raise ValueError("layout must be 'flat' or 'matrix'.")
    if layout == "flat":
        shard_size = total_numel // world_size
        return tuple(
            (LayoutSegment(rank * shard_size, (rank + 1) * shard_size, 0),)
            for rank in range(world_size)
        )

    base = total_numel // world_size
    remainder = total_numel % world_size
    segments = []
    cursor = 0
    for rank in range(world_size):
        size = base + (1 if rank < remainder else 0)
        segments.append((LayoutSegment(cursor, cursor + size, 0),))
        cursor += size
    return tuple(segments)


def _copyin_chunk_cat_rank_segments(
    world_size: int,
    param_count: int,
    param_numel: int,
) -> tuple[tuple[LayoutSegment, ...], ...]:
    padded_chunk_size = (param_numel + world_size - 1) // world_size
    rank_segments = []
    for rank in range(world_size):
        segments = []
        for param_index in range(param_count):
            param_offset = param_index * param_numel
            global_start = param_offset + rank * padded_chunk_size
            global_end = min(param_offset + param_numel, global_start + padded_chunk_size)
            if global_start < global_end:
                segments.append(
                    LayoutSegment(
                        global_start=global_start,
                        global_end=global_end,
                        local_start=param_index * padded_chunk_size,
                    )
                )
        rank_segments.append(tuple(segments))
    return tuple(rank_segments)


def _correctness_worker(
    rank: int,
    config: FSDP2CompareConfig,
    reference_mode: str,
    candidate_mode: str,
) -> None:
    _init_benchmark_process_group(rank, config)
    try:
        device = torch.device("cuda", rank) if config.device == "cuda" else torch.device("cpu")
        mesh = _make_mesh(config)
        reference_losses, reference_output, reference_grads = _run_mode_trace(reference_mode, config, mesh, device)
        candidate_losses, candidate_output, candidate_grads = _run_mode_trace(candidate_mode, config, mesh, device)
        loss_diff = (candidate_losses - reference_losses).abs().max()
        output_diff = (candidate_output - reference_output).abs().max()
        grad_abs_diff, grad_rel_diff, grad_checked_count, local_grad_mismatch_names = _max_named_grad_diffs(
            reference_grads,
            candidate_grads,
            device,
        )
        dist.all_reduce(loss_diff, op=dist.ReduceOp.MAX)
        dist.all_reduce(output_diff, op=dist.ReduceOp.MAX)
        dist.all_reduce(grad_abs_diff, op=dist.ReduceOp.MAX)
        dist.all_reduce(grad_rel_diff, op=dist.ReduceOp.MAX)
        dist.all_reduce(grad_checked_count, op=dist.ReduceOp.MAX)
        gathered_grad_mismatch_names = [None for _ in range(config.world_size)]
        dist.all_gather_object(gathered_grad_mismatch_names, tuple(local_grad_mismatch_names))
        if rank == 0:
            grad_mismatch_names = tuple(
                sorted(
                    {
                        name
                        for names in gathered_grad_mismatch_names
                        if names is not None
                        for name in names
                    }
                )
            )
            result = FSDP2CorrectnessResult(
                reference_mode=reference_mode,
                candidate_mode=candidate_mode,
                unit=config.unit,
                device=config.device,
                world_size=config.world_size,
                layers=config.layers,
                hidden=config.hidden,
                intermediate=config.intermediate,
                batch_size=config.batch_size,
                optimizer=config.optimizer,
                dtype=config.dtype,
                steps=config.steps,
                max_loss_abs_diff=loss_diff.item(),
                max_output_abs_diff=output_diff.item(),
                max_grad_abs_diff=grad_abs_diff.item(),
                max_grad_rel_diff=grad_rel_diff.item(),
                grad_checked_param_count=int(grad_checked_count.item()),
                grad_mismatched_param_count=len(grad_mismatch_names),
                grad_mismatched_param_names=grad_mismatch_names,
                model=config.model,
                seq_len=config.seq_len,
                heads=config.heads,
            )
            with open(config.output_file, "w", encoding="utf-8") as result_file:
                json.dump(asdict(result), result_file, indent=2)
    finally:
        dist.destroy_process_group()


def _memory_trace_worker(rank: int, config: FSDP2CompareConfig) -> None:
    _init_benchmark_process_group(rank, config)
    try:
        device = torch.device("cuda", rank)
        mesh = _make_mesh(config)
        rows = []
        for mode in config.modes:
            rows.extend(_memory_trace_mode(mode, config, mesh, device))
            _cleanup_after_mode(config.device)
        if config.memory_by_rank:
            gathered_rows = [None for _ in range(config.world_size)]
            dist.all_gather_object(gathered_rows, rows)
            if rank == 0:
                rows = [
                    row
                    for rank_rows in gathered_rows
                    if rank_rows is not None
                    for row in rank_rows
                ]
        if rank == 0:
            with open(config.output_file, "w", encoding="utf-8") as result_file:
                json.dump([asdict(row) for row in rows], result_file, indent=2)
    finally:
        dist.destroy_process_group()


def _phase_timing_worker(rank: int, config: FSDP2CompareConfig) -> None:
    _init_benchmark_process_group(rank, config)
    try:
        device = torch.device("cuda", rank) if config.device == "cuda" else torch.device("cpu")
        mesh = _make_mesh(config)
        rows = []
        for mode in config.modes:
            rows.append(_phase_timing_mode(mode, config, mesh, device))
            _cleanup_after_mode(config.device)
        if rank == 0:
            with open(config.output_file, "w", encoding="utf-8") as result_file:
                json.dump([asdict(row) for row in rows], result_file, indent=2)
    finally:
        dist.destroy_process_group()


def _runtime_communication_summary_worker(rank: int, config: FSDP2CompareConfig) -> None:
    _init_benchmark_process_group(rank, config)
    try:
        device = torch.device("cuda", rank) if config.device == "cuda" else torch.device("cpu")
        mesh = _make_mesh(config)
        rows = []
        for mode in config.modes:
            rows.append(_runtime_communication_summary_mode(mode, config, mesh, device))
            _cleanup_after_mode(config.device)
        if rank == 0:
            with open(config.output_file, "w", encoding="utf-8") as result_file:
                json.dump([asdict(row) for row in rows], result_file, indent=2)
    finally:
        dist.destroy_process_group()


def _benchmark_mode(
    mode: str,
    config: FSDP2CompareConfig,
    mesh: DeviceMesh,
    device: torch.device,
) -> FSDP2CompareRow:
    torch.manual_seed(0)
    model = _make_model(config).to(device=device, dtype=_torch_dtype(config))
    param_count = sum(param.numel() for param in model.parameters())
    model, optimizer = _prepare_mode(mode, model, mesh, config)
    x, target = _make_inputs(config, device)

    _synchronize(config.device)
    _profile_prefetch_budget_if_needed(mode, model, optimizer, x, target, config, device)
    prefetch_budget = _prefetch_budget(optimizer)
    _run_steps(model, optimizer, x, target, config.warmup_steps, config.device)
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    _synchronize(config.device)
    start = time.perf_counter()
    _run_steps(model, optimizer, x, target, config.steps, config.device)
    _synchronize(config.device)
    elapsed = time.perf_counter() - start
    elapsed_tensor = torch.tensor(elapsed, device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    peak_memory_mb = _peak_memory_mb(config.device, device)
    peak_memory_tensor = torch.tensor(peak_memory_mb, device=device)
    dist.all_reduce(peak_memory_tensor, op=dist.ReduceOp.MAX)
    dist.barrier()

    return FSDP2CompareRow(
        mode=mode,
        unit=config.unit,
        device=config.device,
        world_size=config.world_size,
        layers=config.layers,
        hidden=config.hidden,
        intermediate=config.intermediate,
        batch_size=config.batch_size,
        optimizer=config.optimizer,
        dtype=config.dtype,
        param_count=param_count,
        avg_step_ms=elapsed_tensor.item() * 1000.0 / config.steps,
        peak_memory_mb=peak_memory_tensor.item(),
        model=config.model,
        seq_len=config.seq_len,
        heads=config.heads,
        prefetch_budget=prefetch_budget,
    )


def _phase_timing_mode(
    mode: str,
    config: FSDP2CompareConfig,
    mesh: DeviceMesh,
    device: torch.device,
) -> FSDP2PhaseTimingRow:
    torch.manual_seed(0)
    model = _make_model(config).to(device=device, dtype=_torch_dtype(config))
    model, optimizer = _prepare_mode(mode, model, mesh, config)
    x, target = _make_inputs(config, device)

    _synchronize(config.device)
    _profile_prefetch_budget_if_needed(mode, model, optimizer, x, target, config, device)
    prefetch_budget = _prefetch_budget(optimizer)
    _run_steps(model, optimizer, x, target, config.warmup_steps, config.device)
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    _synchronize(config.device)

    zero_grad_elapsed = 0.0
    forward_elapsed = 0.0
    backward_elapsed = 0.0
    step_elapsed = 0.0
    for _ in range(config.steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        _synchronize(config.device)
        zero_grad_elapsed += time.perf_counter() - start

        start = time.perf_counter()
        loss = _loss(model(x), target)
        _synchronize(config.device)
        forward_elapsed += time.perf_counter() - start

        start = time.perf_counter()
        loss.backward()
        _synchronize(config.device)
        backward_elapsed += time.perf_counter() - start

        start = time.perf_counter()
        optimizer.step()
        _synchronize(config.device)
        step_elapsed += time.perf_counter() - start

    elapsed_values = torch.tensor(
        [zero_grad_elapsed, forward_elapsed, backward_elapsed, step_elapsed],
        device=device,
    )
    dist.all_reduce(elapsed_values, op=dist.ReduceOp.MAX)
    peak_memory_mb = _peak_memory_mb(config.device, device)
    peak_memory_tensor = torch.tensor(peak_memory_mb, device=device)
    dist.all_reduce(peak_memory_tensor, op=dist.ReduceOp.MAX)
    dist.barrier()

    per_step_ms = elapsed_values * (1000.0 / config.steps)
    return FSDP2PhaseTimingRow(
        mode=mode,
        unit=config.unit,
        device=config.device,
        world_size=config.world_size,
        optimizer=config.optimizer,
        dtype=config.dtype,
        avg_zero_grad_ms=per_step_ms[0].item(),
        avg_forward_ms=per_step_ms[1].item(),
        avg_backward_ms=per_step_ms[2].item(),
        avg_step_ms=per_step_ms[3].item(),
        avg_total_ms=per_step_ms.sum().item(),
        peak_memory_mb=peak_memory_tensor.item(),
        model=config.model,
        seq_len=config.seq_len,
        prefetch_budget=prefetch_budget,
    )


def _runtime_communication_summary_mode(
    mode: str,
    config: FSDP2CompareConfig,
    mesh: DeviceMesh,
    device: torch.device,
) -> MatrixRuntimeCommunicationRow:
    torch.manual_seed(0)
    model = _make_model(config).to(device=device, dtype=_torch_dtype(config))
    model, _optimizer = _prepare_mode(mode, model, mesh, config)
    try:
        summary = summarize_param_groups(model)
    except ValueError:
        communication_summary = {
            "num_param_groups": 0,
            "gather_backend_counts": {},
            "resolved_custom_allgatherv_counts": {},
            "rank_chunk_fast_path_count": 0,
            "packed_full_order_count": 0,
            "max_segment_count": 0,
            "max_segments_per_rank": 0,
            "max_padding_waste_ratio": 0.0,
            "max_owner_imbalance_ratio": 0.0,
            "workspace_preferred_kind_counts": {},
            "max_workspace_preferred_numel": 0,
            "max_workspace_padded_rank_chunks_numel": 0,
            "max_workspace_padding_waste_ratio": 0.0,
            "workspace_total_acquire_count": 0,
            "workspace_total_reuse_count": 0,
            "workspace_total_allocate_count": 0,
            "max_workspace_allocated_numel": 0,
            "max_shard_size": 0,
            "min_shard_size": 0,
        }
    else:
        communication_summary = summary["communication_summary"]

    return MatrixRuntimeCommunicationRow(
        mode=mode,
        model=config.model,
        unit=config.unit,
        device=config.device,
        world_size=config.world_size,
        optimizer=config.optimizer,
        dtype=config.dtype,
        param_groups=int(communication_summary["num_param_groups"]),
        gather_backend_counts=_format_count_mapping(communication_summary["gather_backend_counts"]),
        resolved_custom_allgatherv_counts=_format_count_mapping(
            communication_summary["resolved_custom_allgatherv_counts"]
        ),
        rank_chunk_fast_paths=int(communication_summary["rank_chunk_fast_path_count"]),
        packed_full_order=int(communication_summary["packed_full_order_count"]),
        max_segment_count=int(communication_summary["max_segment_count"]),
        max_segments_per_rank=int(communication_summary["max_segments_per_rank"]),
        max_padding_waste_ratio=float(communication_summary["max_padding_waste_ratio"]),
        max_owner_imbalance_ratio=float(communication_summary["max_owner_imbalance_ratio"]),
        workspace_preferred_kind_counts=_format_count_mapping(communication_summary["workspace_preferred_kind_counts"]),
        max_workspace_preferred_numel=int(communication_summary["max_workspace_preferred_numel"]),
        max_workspace_padded_numel=int(communication_summary["max_workspace_padded_rank_chunks_numel"]),
        max_workspace_padding_waste_ratio=float(communication_summary["max_workspace_padding_waste_ratio"]),
        workspace_acquires=int(communication_summary["workspace_total_acquire_count"]),
        workspace_reuses=int(communication_summary["workspace_total_reuse_count"]),
        workspace_allocates=int(communication_summary["workspace_total_allocate_count"]),
        max_workspace_allocated_numel=int(communication_summary["max_workspace_allocated_numel"]),
        min_shard_size=int(communication_summary["min_shard_size"]),
        max_shard_size=int(communication_summary["max_shard_size"]),
        seq_len=config.seq_len,
    )


def _memory_trace_mode(
    mode: str,
    config: FSDP2CompareConfig,
    mesh: DeviceMesh,
    device: torch.device,
) -> tuple[FSDP2MemoryTraceRow, ...]:
    torch.manual_seed(0)
    model = _make_model(config).to(device=device, dtype=_torch_dtype(config))
    model, optimizer = _prepare_mode(mode, model, mesh, config)
    x, target = _make_inputs(config, device)

    _synchronize(config.device)
    _run_steps(model, optimizer, x, target, config.warmup_steps, config.device)
    optimizer.zero_grad(set_to_none=True)
    if config.empty_cache_after_warmup and config.device == "cuda":
        torch.cuda.empty_cache()
    _synchronize(config.device)
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()

    rows = [_memory_trace_row(mode, config, device, "start", model, optimizer)]
    optimizer.zero_grad(set_to_none=True)
    _synchronize(config.device)
    rows.append(_memory_trace_row(mode, config, device, "after_zero_grad_before_forward", model, optimizer))
    loss = _loss(model(x), target)
    _synchronize(config.device)
    rows.append(_memory_trace_row(mode, config, device, "after_forward", model, optimizer))
    loss.backward()
    _synchronize(config.device)
    rows.append(_memory_trace_row(mode, config, device, "after_backward", model, optimizer))
    optimizer.step()
    _synchronize(config.device)
    rows.append(_memory_trace_row(mode, config, device, "after_step", model, optimizer))
    optimizer.zero_grad(set_to_none=True)
    _synchronize(config.device)
    rows.append(_memory_trace_row(mode, config, device, "after_zero_grad_after_step", model, optimizer))
    dist.barrier()
    return tuple(rows)


def _run_mode_trace(
    mode: str,
    config: FSDP2CompareConfig,
    mesh: DeviceMesh,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    model = _make_model(config).to(device=device, dtype=_torch_dtype(config))
    model, optimizer = _prepare_mode(mode, model, mesh, config)
    torch.manual_seed(1)
    x, target = _make_inputs(config, device)

    losses = []
    last_grad: dict[str, torch.Tensor] = {}
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model(x), target)
        losses.append(loss.detach())
        loss.backward()
        last_grad = _full_named_param_grads(model, optimizer, device)
        optimizer.step()
    _synchronize(config.device)
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        output = model(x).detach()
    _synchronize(config.device)
    return torch.stack(losses), output, last_grad


def _full_named_param_grads(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | MatrixFSDPOptimizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    matrix_runtime = _matrix_runtime_from_optimizer(optimizer)
    if matrix_runtime is not None:
        return _full_named_param_grads_from_matrix(model, matrix_runtime, device)
    return _full_named_param_grads_from_model(model, device)


def _full_named_param_grads_from_model(model: nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    seen_param_ids: set[int] = set()
    for name, param in model.named_parameters():
        if id(param) in seen_param_ids:
            continue
        seen_param_ids.add(id(param))
        normalized_name = _normalize_param_fqn(name)
        grad = param.grad
        if grad is None:
            grads[normalized_name] = torch.zeros(param.numel(), device=device, dtype=torch.float32)
            continue
        if hasattr(grad, "full_tensor"):
            grad = grad.full_tensor()
        if hasattr(grad, "to_local") and not isinstance(grad, torch.Tensor):
            grad = grad.to_local()
        grads[normalized_name] = grad.detach().to(device=device, dtype=torch.float32).reshape(-1)
    return grads


def _full_named_param_grads_from_matrix(
    model: nn.Module,
    optimizer: object,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    full_fqn_by_param_id = {
        id(param): _normalize_param_fqn(name)
        for name, param in model.named_parameters(remove_duplicate=True)
    }
    _finalize_matrix_backward_for_grad_collection(optimizer)
    grads: dict[str, torch.Tensor] = {}
    managed_fqns: set[str] = set()
    for unit in optimizer.runtime_param_groups:
        flat_buffer = unit.flat_buffer
        if flat_buffer is None:
            continue
        local_grad_shard = flat_buffer.local_grad_shard
        views_by_fqn: dict[str, list[object]] = {}
        for view in flat_buffer.local_param_views:
            views_by_fqn.setdefault(view.managed_param.fqn, []).append(view)
        for managed_param in unit.managed_params:
            fqn = full_fqn_by_param_id.get(id(managed_param.param), _normalize_param_fqn(managed_param.fqn))
            managed_fqns.add(fqn)
            full_grad = torch.zeros(managed_param.numel, device=device, dtype=torch.float32)
            if local_grad_shard is not None:
                for view in views_by_fqn.get(managed_param.fqn, ()):
                    full_grad[view.param_start : view.param_end] = local_grad_shard[
                        view.shard_start : view.shard_end
                    ].detach().to(device=device, dtype=torch.float32)
            if dist.is_initialized() and unit.group is not None:
                dist.all_reduce(full_grad, group=unit.group, op=dist.ReduceOp.SUM)
            grads[fqn] = full_grad

    for name, param in model.named_parameters(remove_duplicate=True):
        normalized_name = _normalize_param_fqn(name)
        if normalized_name in managed_fqns:
            continue
        grad = param.grad
        if grad is None:
            grads[normalized_name] = torch.zeros(param.numel(), device=device, dtype=torch.float32)
            continue
        if hasattr(grad, "full_tensor"):
            grad = grad.full_tensor()
        if hasattr(grad, "to_local") and not isinstance(grad, torch.Tensor):
            grad = grad.to_local()
        grads[normalized_name] = grad.detach().to(device=device, dtype=torch.float32).reshape(-1)
    return grads


def _finalize_matrix_backward_for_grad_collection(optimizer: object) -> None:
    optimizer.scheduler.wait_pending_backward_reduce()
    for unit in optimizer.runtime_param_groups:
        if unit.has_pending_backward_reduce:
            unit.finalize_backward()
            continue
        if unit.finalize_after_backward_enabled and unit.finalized_after_backward:
            continue
        if unit.lifecycle_state.name == "SHARDED" and not any(mp.param.grad is not None for mp in unit.managed_params):
            continue
        unit.finalize_backward()


def _normalize_param_fqn(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _max_named_grad_diffs(
    reference_grads: Mapping[str, torch.Tensor],
    candidate_grads: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...]]:
    max_abs_diff = torch.tensor(0.0, device=device)
    max_rel_diff = torch.tensor(0.0, device=device)
    mismatch_names: list[str] = []
    names = sorted(set(reference_grads) | set(candidate_grads))
    for name in names:
        reference = reference_grads.get(name)
        candidate = candidate_grads.get(name)
        if reference is None or candidate is None or reference.numel() != candidate.numel():
            mismatch_names.append(name)
            max_abs_diff = torch.maximum(max_abs_diff, torch.tensor(float("inf"), device=device))
            max_rel_diff = torch.maximum(max_rel_diff, torch.tensor(float("inf"), device=device))
            continue
        reference = reference.to(device=device, dtype=torch.float32).reshape(-1)
        candidate = candidate.to(device=device, dtype=torch.float32).reshape(-1)
        diff = (candidate - reference).abs()
        if diff.numel() == 0:
            continue
        abs_diff = diff.max()
        denom = reference.abs().clamp_min(1e-12)
        rel_diff = (diff / denom).max()
        max_abs_diff = torch.maximum(max_abs_diff, abs_diff)
        max_rel_diff = torch.maximum(max_rel_diff, rel_diff)
        if not torch.isfinite(abs_diff) or not torch.isfinite(rel_diff):
            mismatch_names.append(name)
    checked_count = torch.tensor(len(names), device=device, dtype=torch.long)
    return max_abs_diff, max_rel_diff, checked_count, tuple(mismatch_names)


def _memory_trace_row(
    mode: str,
    config: FSDP2CompareConfig,
    device: torch.device,
    phase: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | MatrixFSDPOptimizer,
) -> FSDP2MemoryTraceRow:
    current_memory_mb, peak_memory_mb, current_reserved_mb, peak_reserved_mb = _memory_snapshot_mb(
        config.device,
        device,
    )
    accounting = _memory_accounting(model, optimizer)
    values = torch.tensor(
        [
            current_memory_mb,
            peak_memory_mb,
            current_reserved_mb,
            peak_reserved_mb,
            float(accounting["active_full_param_buffers"]),
            accounting["full_param_buffer_mb"],
            accounting["local_shard_mb"],
            accounting["full_grad_buffer_mb"],
            accounting["grad_bucket_mb"],
            accounting["local_grad_shard_mb"],
            accounting["optimizer_state_mb"],
            float(accounting["pending_backward_reduces"]),
        ],
        dtype=torch.float64,
        device=device,
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    if config.memory_by_rank:
        reduced_values = values
        row_rank = rank
    else:
        reduced_values = values.clone()
        dist.all_reduce(reduced_values, op=dist.ReduceOp.MAX)
        row_rank = -1
    return FSDP2MemoryTraceRow(
        mode=mode,
        unit=config.unit,
        device=config.device,
        world_size=config.world_size,
        optimizer=config.optimizer,
        dtype=config.dtype,
        phase=phase,
        current_memory_mb=reduced_values[0].item(),
        peak_memory_mb=reduced_values[1].item(),
        rank=row_rank,
        current_reserved_mb=reduced_values[2].item(),
        peak_reserved_mb=reduced_values[3].item(),
        model=config.model,
        seq_len=config.seq_len,
        prefetch_budget=_prefetch_budget(optimizer),
        active_full_param_buffers=int(reduced_values[4].item()),
        full_param_buffer_mb=reduced_values[5].item(),
        local_shard_mb=reduced_values[6].item(),
        full_grad_buffer_mb=reduced_values[7].item(),
        grad_bucket_mb=reduced_values[8].item(),
        local_grad_shard_mb=reduced_values[9].item(),
        optimizer_state_mb=reduced_values[10].item(),
        pending_backward_reduces=int(reduced_values[11].item()),
    )


def _prepare_mode(
    mode: str,
    model: nn.Module,
    mesh: DeviceMesh,
    config: FSDP2CompareConfig,
) -> tuple[nn.Module, torch.optim.Optimizer | MatrixFSDPOptimizer]:
    if mode == "eager":
        return model, _make_optimizer(model.parameters(), config)
    if mode == "fsdp2":
        _apply_fsdp2(model, mesh, config, shard_root=True)
        return model, _make_optimizer(model.parameters(), config)
    if mode == "fsdp2_api":
        assert torch_fully_shard is not None
        torch_fully_shard(model, mesh=mesh)
        return model, _make_optimizer(model.parameters(), config)
    if mode == "fsdp2_no_root":
        _apply_fsdp2(model, mesh, config, shard_root=False)
        return model, _make_optimizer(model.parameters(), config)
    if mode == "matrix_api":
        sharded_model = _apply_matrix_api(model, mesh, config)
        return sharded_model, _make_optimizer(sharded_model.parameters(), config)
    if mode in DEFAULT_FAST_MATRIX_MODE_ALIASES:
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
            use_saved_tensor_hooks=False,
        )
    if mode == "matrix_zero_copy_grad_bucket":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            backward_prefetch_timing="post_reshard",
            use_zero_copy_grad_bucket=True,
        )
    if mode == "matrix_memory_capped":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            backward_prefetch_timing="post_reshard",
            use_zero_copy_grad_bucket=False,
            max_active_full_param_buffers=1,
        )
    if mode == "matrix_no_prefetch":
        sharded_model = matrix_fully_shard(
            model,
            mesh,
            wrap_policy=_wrap_policy(config),
            reshard_after_forward=True,
            forward_prefetch=False,
            backward_prefetch=False,
            finalize_after_backward=True,
            backward_reduce_strategy="bucket_reduce_scatter",
            runtime_trace_enabled=False,
        )
        optimizer = MatrixFSDPOptimizer(_make_optimizer(sharded_model.parameters(), config), sharded_model)
        return sharded_model, optimizer
    if mode == "matrix_optimizer_finalize":
        sharded_model = matrix_fully_shard(
            model,
            mesh,
            wrap_policy=_wrap_policy(config),
            reshard_after_forward=True,
            finalize_after_backward=False,
            runtime_trace_enabled=False,
        )
        optimizer = MatrixFSDPOptimizer(_make_optimizer(sharded_model.parameters(), config), sharded_model)
        return sharded_model, optimizer
    if mode == "matrix_auto_finalize":
        sharded_model = matrix_fully_shard(
            model,
            mesh,
            wrap_policy=_wrap_policy(config),
            reshard_after_forward=True,
            finalize_after_backward=True,
            runtime_trace_enabled=False,
        )
        optimizer = MatrixFSDPOptimizer(_make_optimizer(sharded_model.parameters(), config), sharded_model)
        return sharded_model, optimizer
    if mode == "matrix_prefetch":
        return _prepare_matrix_prefetch(model, mesh, config)
    if mode == "matrix_prefetch_cap1":
        return _prepare_matrix_prefetch(model, mesh, config, max_unsharded_prefetch_units=1)
    if mode == "matrix_prefetch_cap2":
        return _prepare_matrix_prefetch(model, mesh, config, max_unsharded_prefetch_units=2)
    if mode == "matrix_prefetch_fsdp2_schedule_bucket_reduce_scatter":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
        )
    if mode == "matrix_prefetch_fsdp2_schedule_bucket_copy_in":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
        )
    if mode == "matrix_prefetch_fsdp2_schedule_bucket_copy_in_no_saved_hooks":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
            use_saved_tensor_hooks=False,
        )
    if mode == "matrix_prefetch_fsdp2_schedule_bucket_reduce_scatter_wait_rs":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            max_pending_backward_reduces=0,
        )
    if mode == "matrix_prefetch_fsdp2_schedule_bucket_copy_in_wait_rs":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
            max_pending_backward_reduces=0,
        )
    if mode == "matrix_prefetch_fsdp2_schedule_bucket_copy_in_mem_cap1":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            use_zero_copy_grad_bucket=False,
            max_active_full_param_buffers=1,
        )
    if mode == "matrix_prefetch_late_backward_bucket_reduce_scatter":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            backward_prefetch_timing="post_reshard",
        )
    if mode == "matrix_prefetch_late_backward_bucket_copy_in":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            backward_prefetch_timing="post_reshard",
            use_zero_copy_grad_bucket=False,
        )
    if mode == "matrix_prefetch_late_backward_fsdp2_chunk":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            max_unsharded_prefetch_units=1,
            backward_reduce_strategy="bucket_reduce_scatter",
            backward_prefetch_timing="post_reshard",
            use_zero_copy_grad_bucket=False,
            group_planner=fsdp2_chunk_plan,
        )
    if mode == "matrix_prefetch_adaptive":
        return _prepare_matrix_prefetch(model, mesh, config, prefetch_policy="adaptive")
    if mode == "matrix_prefetch_adaptive_bucket_reduce_scatter":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            prefetch_policy="adaptive",
            backward_reduce_strategy="bucket_reduce_scatter",
        )
    if mode == "matrix_prefetch_adaptive_no_saved_hooks":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            prefetch_policy="adaptive",
            backward_reduce_strategy="bucket_reduce_scatter",
            use_saved_tensor_hooks=False,
        )
    if mode == "matrix_prefetch_adaptive_per_param":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            prefetch_policy="adaptive",
            backward_reduce_strategy="per_param",
        )
    if mode == "matrix_prefetch_adaptive_per_param_allreduce":
        return _prepare_matrix_prefetch(
            model,
            mesh,
            config,
            prefetch_policy="adaptive",
            backward_reduce_strategy="per_param_allreduce",
        )
    if mode == "matrix_prefetch_profile_guided":
        return _prepare_matrix_prefetch(model, mesh, config, prefetch_policy="profile_guided")
    if mode == "matrix_owner_muon":
        return _prepare_matrix_owner_muon(model, mesh, config)
    if mode == "matrix_owner_muon_pre_backward":
        return _prepare_matrix_owner_muon(model, mesh, config, backward_prefetch_timing="pre_backward")
    if mode == "matrix_owner_muon_role_greedy":
        return _prepare_matrix_owner_muon(model, mesh, config, owner_assignment="role_greedy")
    if mode == "matrix_owner_muon_role_greedy_pre_backward":
        return _prepare_matrix_owner_muon(
            model,
            mesh,
            config,
            owner_assignment="role_greedy",
            backward_prefetch_timing="pre_backward",
        )
    if mode == "matrix_owner_muon_role_greedy_matrix_all_gather":
        return _prepare_matrix_owner_muon(
            model,
            mesh,
            config,
            owner_assignment="role_greedy",
            param_gather_strategy="matrix_all_gather",
        )
    if mode == "matrix_owner_muon_role_greedy_matrix_all_gather_pre_backward":
        return _prepare_matrix_owner_muon(
            model,
            mesh,
            config,
            owner_assignment="role_greedy",
            backward_prefetch_timing="pre_backward",
            param_gather_strategy="matrix_all_gather",
        )
    if mode == "matrix_owner_muon_role_greedy_custom_collective":
        return _prepare_matrix_owner_muon(
            model,
            mesh,
            config,
            owner_assignment="role_greedy",
            matrix_collective_backend="custom",
        )
    if mode == "matrix_owner_muon_role_greedy_custom_collective_pre_backward":
        return _prepare_matrix_owner_muon(
            model,
            mesh,
            config,
            owner_assignment="role_greedy",
            backward_prefetch_timing="pre_backward",
            matrix_collective_backend="custom",
        )
    raise ValueError(f"Unknown benchmark mode: {mode}.")


def _prepare_matrix_prefetch(
    model: nn.Module,
    mesh: DeviceMesh,
    config: FSDP2CompareConfig,
    *,
    max_unsharded_prefetch_units: int | None = None,
    prefetch_policy: str = "static",
    backward_reduce_strategy: str = "flat",
    use_saved_tensor_hooks: bool = True,
    backward_prefetch_timing: str = "pre_backward",
    use_zero_copy_grad_bucket: bool = True,
    group_planner=None,
    max_active_full_param_buffers: int | None = None,
    max_active_full_param_numel: int | None = None,
    max_active_full_param_memory_mb: float | None = None,
    max_pending_backward_reduces: int | None = 1,
) -> tuple[nn.Module, MatrixFSDPOptimizer]:
    max_active_full_param_buffers = (
        config.matrix_max_active_full_param_buffers
        if max_active_full_param_buffers is None
        else max_active_full_param_buffers
    )
    max_active_full_param_numel = (
        config.matrix_max_active_full_param_numel
        if max_active_full_param_numel is None
        else max_active_full_param_numel
    )
    if max_active_full_param_memory_mb is None:
        max_active_full_param_memory_mb = config.matrix_max_active_full_param_memory_mb or None
    sharded_model = matrix_fully_shard(
        model,
        mesh,
        wrap_policy=_wrap_policy(config),
        reshard_after_forward=True,
        forward_prefetch=True,
        backward_prefetch=True,
        finalize_after_backward=True,
        group_planner=group_planner,
        backward_reduce_strategy=backward_reduce_strategy,
        runtime_trace_enabled=False,
        use_saved_tensor_hooks=use_saved_tensor_hooks,
        use_zero_copy_grad_bucket=use_zero_copy_grad_bucket,
    )
    optimizer = MatrixFSDPOptimizer(
        _make_optimizer(sharded_model.parameters(), config),
        sharded_model,
        max_unsharded_prefetch_units=max_unsharded_prefetch_units,
        backward_prefetch_timing=backward_prefetch_timing,  # type: ignore[arg-type]
        prefetch_policy=prefetch_policy,
        max_active_full_param_buffers=max_active_full_param_buffers,
        max_active_full_param_numel=max_active_full_param_numel,
        max_active_full_param_memory_mb=max_active_full_param_memory_mb,
        max_pending_backward_reduces=max_pending_backward_reduces,
    )
    return sharded_model, optimizer


def _prepare_matrix_owner_muon(
    model: nn.Module,
    mesh: DeviceMesh,
    config: FSDP2CompareConfig,
    *,
    owner_assignment: str = "rotate",
    backward_prefetch_timing: str = "post_reshard",
    param_gather_strategy: str = "auto",
    matrix_collective_backend: str = "owner_broadcast",
) -> tuple[nn.Module, MatrixFSDPOptimizer]:
    if config.optimizer != "muon":
        raise ValueError("matrix_owner_muon modes require --optimizer muon.")

    group_planner = make_muon_shard_aware_group_planner(owner_assignment=owner_assignment)

    sharded_model = matrix_fully_shard(
        model,
        mesh,
        wrap_policy=_wrap_policy(config),
        auto_shard_hints=True,
        group_planner=group_planner,
        reshard_after_forward=True,
        forward_prefetch=True,
        backward_prefetch=True,
        finalize_after_backward=True,
        backward_reduce_strategy="bucket_reduce_scatter",
        runtime_trace_enabled=False,
        use_saved_tensor_hooks=False,
        use_zero_copy_grad_bucket=True,
        param_gather_strategy=param_gather_strategy,
        matrix_collective_backend=matrix_collective_backend,
    )
    optimizer = MatrixFSDPOptimizer(
        _make_optimizer(sharded_model.parameters(), config),
        sharded_model,
        max_unsharded_prefetch_units=1,
        backward_prefetch_timing=backward_prefetch_timing,  # type: ignore[arg-type]
    )
    return sharded_model, optimizer


def _apply_fsdp2(model: nn.Module, mesh: DeviceMesh, config: FSDP2CompareConfig, *, shard_root: bool) -> None:
    assert torch_fully_shard is not None
    for module in model.modules():
        if _is_unit(module, config):
            torch_fully_shard(module, mesh=mesh, reshard_after_forward=True)
    if shard_root:
        torch_fully_shard(model, mesh=mesh, reshard_after_forward=True)


def _apply_matrix_api(model: nn.Module, mesh: DeviceMesh, config: FSDP2CompareConfig) -> nn.Module:
    wrapped_units = 0
    for module in model.modules():
        if _is_unit(module, config):
            matrix_fully_shard_api(module, mesh=mesh)
            wrapped_units += 1
    if wrapped_units == 0:
        return matrix_fully_shard_api(model, mesh=mesh)
    return model


def _make_optimizer(params, config: FSDP2CompareConfig) -> torch.optim.Optimizer:
    params = list(params)
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=0.01)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=0.001)
    if config.optimizer == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("torch.optim.Muon is not available in this PyTorch build.")
        muon_params = _muon_params(params)
        if not muon_params:
            raise RuntimeError("torch.optim.Muon benchmark found no non-empty 2D parameters.")
        return MixedMuonAdamWOptimizer(muon_params, _adamw_params(params), lazy_muon_init=True)
    raise ValueError(f"Unknown optimizer: {config.optimizer}.")


def _muon_params(params) -> list[nn.Parameter]:
    return [param for param in params if param.ndim == 2 and param.numel() > 0]


def _adamw_params(params) -> list[nn.Parameter]:
    return [param for param in params if param.ndim != 2 and param.numel() > 0]


def _prefetch_budget(optimizer: torch.optim.Optimizer | MatrixFSDPOptimizer) -> str:
    scheduler = _scheduler_from_optimizer(optimizer)
    if scheduler is None:
        return ""
    forward_budget = getattr(scheduler, "max_forward_prefetch_units", None)
    backward_budget = getattr(scheduler, "max_backward_prefetch_units", None)
    parts = [f"f={_format_prefetch_budget(forward_budget)}/b={_format_prefetch_budget(backward_budget)}"]
    full_buffer_limit = getattr(scheduler, "max_active_full_param_buffers_limit", None)
    full_numel_limit = getattr(scheduler, "max_active_full_param_numel_limit", None)
    full_bytes_limit = getattr(scheduler, "max_active_full_param_bytes_limit", None)
    if full_buffer_limit is not None:
        parts.append(f"full_buf={full_buffer_limit}")
    if full_numel_limit is not None:
        parts.append(f"full_numel={full_numel_limit}")
    if full_bytes_limit is not None:
        parts.append(f"full_mb={full_bytes_limit / (1024 * 1024):.1f}")
    return ",".join(parts)


def _format_prefetch_budget(budget: int | None) -> str:
    return "none" if budget is None else str(budget)


def _memory_accounting(
    _model: nn.Module,
    optimizer: torch.optim.Optimizer | MatrixFSDPOptimizer,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "active_full_param_buffers": 0,
        "full_param_buffer_mb": 0.0,
        "local_shard_mb": 0.0,
        "full_grad_buffer_mb": 0.0,
        "grad_bucket_mb": 0.0,
        "local_grad_shard_mb": 0.0,
        "optimizer_state_mb": _bytes_to_mb(_optimizer_state_bytes(optimizer)),
        "pending_backward_reduces": 0,
    }
    matrix_runtime = _matrix_runtime_from_optimizer(optimizer)
    if matrix_runtime is None:
        return result

    active_full_param_buffers = 0
    full_param_buffer_bytes = 0
    local_shard_bytes = 0
    full_grad_buffer_bytes = 0
    grad_bucket_bytes = 0
    local_grad_shard_bytes = 0
    for unit in matrix_runtime.runtime_param_groups:
        flat_buffer = getattr(unit, "flat_buffer", None)
        if flat_buffer is None:
            continue
        full_buffer = getattr(flat_buffer, "full_buffer", None)
        if full_buffer is not None:
            active_full_param_buffers += 1
            full_param_buffer_bytes += _tensor_nbytes(full_buffer)
        local_shard = getattr(flat_buffer, "local_shard", None)
        if local_shard is not None:
            local_shard_bytes += _tensor_nbytes(local_shard)
        full_grad_buffer = getattr(flat_buffer, "full_grad_buffer", None)
        if full_grad_buffer is not None:
            full_grad_buffer_bytes += _tensor_nbytes(full_grad_buffer)
        grad_bucket_input = getattr(flat_buffer, "grad_bucket_input", None)
        if grad_bucket_input is not None:
            grad_bucket_bytes += _tensor_nbytes(grad_bucket_input)
        local_grad_shard = getattr(flat_buffer, "local_grad_shard", None)
        if local_grad_shard is not None:
            local_grad_shard_bytes += _tensor_nbytes(local_grad_shard)

    scheduler = _scheduler_from_optimizer(optimizer)
    pending_backward_reduces = getattr(scheduler, "pending_backward_reduce_count", 0) if scheduler is not None else 0
    result.update(
        {
            "active_full_param_buffers": active_full_param_buffers,
            "full_param_buffer_mb": _bytes_to_mb(full_param_buffer_bytes),
            "local_shard_mb": _bytes_to_mb(local_shard_bytes),
            "full_grad_buffer_mb": _bytes_to_mb(full_grad_buffer_bytes),
            "grad_bucket_mb": _bytes_to_mb(grad_bucket_bytes),
            "local_grad_shard_mb": _bytes_to_mb(local_grad_shard_bytes),
            "pending_backward_reduces": int(pending_backward_reduces),
        }
    )
    return result


def _matrix_runtime_from_optimizer(optimizer: torch.optim.Optimizer | MatrixFSDPOptimizer) -> object | None:
    if isinstance(optimizer, MatrixFSDPOptimizer):
        return optimizer
    return getattr(optimizer, "matrix_fsdp", None)


def _scheduler_from_optimizer(optimizer: torch.optim.Optimizer | MatrixFSDPOptimizer) -> object | None:
    scheduler = getattr(optimizer, "scheduler", None)
    if scheduler is not None:
        return scheduler
    scheduler = getattr(optimizer, "matrix_fsdp_scheduler", None)
    if scheduler is not None:
        return scheduler
    matrix_runtime = getattr(optimizer, "matrix_fsdp", None)
    return getattr(matrix_runtime, "scheduler", None)


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer | MatrixFSDPOptimizer) -> int:
    state = getattr(optimizer, "state", {})
    if not isinstance(state, Mapping):
        return 0
    total = 0
    seen_storage_ptrs: set[int] = set()
    for state_value in state.values():
        if not isinstance(state_value, Mapping):
            continue
        for value in state_value.values():
            if not torch.is_tensor(value):
                continue
            local_value = _local_tensor_for_accounting(value)
            storage_key = _tensor_storage_key(local_value)
            if storage_key in seen_storage_ptrs:
                continue
            seen_storage_ptrs.add(storage_key)
            total += _tensor_nbytes(local_value)
    return total


def _local_tensor_for_accounting(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        local_tensor = to_local()
        if torch.is_tensor(local_tensor):
            return local_tensor
    return tensor


def _tensor_storage_key(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except RuntimeError:
        return id(tensor)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def _profile_prefetch_budget_if_needed(
    mode: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | MatrixFSDPOptimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    config: FSDP2CompareConfig,
    device: torch.device,
) -> None:
    if mode != "matrix_prefetch_profile_guided":
        return
    scheduler = getattr(optimizer, "scheduler", None)
    if scheduler is None:
        return
    profile_steps = config.profile_steps
    if profile_steps <= 0:
        return

    results: list[PrefetchProfileResult] = []
    for budget in (None, 0, 1, 2):
        scheduler.set_prefetch_budget(budget)
        optimizer.zero_grad(set_to_none=True)
        _synchronize(config.device)
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        start = time.perf_counter()
        _run_steps(model, optimizer, x, target, profile_steps, config.device)
        _synchronize(config.device)
        elapsed = time.perf_counter() - start
        elapsed_tensor = torch.tensor(elapsed, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        peak_memory_mb = _peak_memory_mb(config.device, device)
        peak_memory_tensor = torch.tensor(peak_memory_mb, device=device)
        dist.all_reduce(peak_memory_tensor, op=dist.ReduceOp.MAX)
        results.append(
            PrefetchProfileResult(
                budget=budget,
                avg_step_ms=elapsed_tensor.item() * 1000.0 / profile_steps,
                peak_memory_mb=peak_memory_tensor.item(),
            )
        )
        dist.barrier()

    scheduler.set_profile_results(results)
    memory_limit = config.profile_memory_limit_mb or None
    scheduler.select_profiled_budget(memory_limit_mb=memory_limit)


def _make_model(config: FSDP2CompareConfig) -> nn.Module:
    use_inline_checkpoint = config.activation_checkpoint and not config.activation_checkpoint_wrapper
    if config.model == "mlp":
        model = MLPStack(
            config.layers,
            config.hidden,
            config.intermediate,
            activation_checkpoint=use_inline_checkpoint,
            checkpoint_use_reentrant=config.checkpoint_use_reentrant,
        )
    elif config.model == "transformer":
        model = TransformerStack(
            config.layers,
            config.hidden,
            config.intermediate,
            config.heads,
            activation_checkpoint=use_inline_checkpoint,
            checkpoint_use_reentrant=config.checkpoint_use_reentrant,
        )
    elif config.model == "transformer_split_qkv":
        model = SplitQKVTransformerStack(
            config.layers,
            config.hidden,
            config.intermediate,
            config.heads,
            activation_checkpoint=use_inline_checkpoint,
            checkpoint_use_reentrant=config.checkpoint_use_reentrant,
        )
    else:
        raise ValueError(f"Unknown model: {config.model}.")
    if config.activation_checkpoint and config.activation_checkpoint_wrapper:
        _apply_checkpoint_wrapper(model, config)
    return model


def _apply_checkpoint_wrapper(model: nn.Module, config: FSDP2CompareConfig) -> None:
    checkpoint_impl = CheckpointImpl.REENTRANT if config.checkpoint_use_reentrant else CheckpointImpl.NO_REENTRANT
    wrapper = partial(checkpoint_wrapper, checkpoint_impl=checkpoint_impl)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=_activation_checkpoint_check_fn(config),
    )


def _activation_checkpoint_check_fn(config: FSDP2CompareConfig):
    if config.model == "mlp":
        target_types = (MLPBlock,)
    elif config.model == "transformer":
        target_types = (TransformerBlock,)
    elif config.model == "transformer_split_qkv":
        target_types = (SplitQKVTransformerBlock,)
    else:
        raise ValueError(f"Unknown model: {config.model}.")
    return lambda module: isinstance(module, target_types)


def _make_mesh(config: FSDP2CompareConfig) -> DeviceMesh:
    return DeviceMesh(config.device, torch.arange(config.world_size), mesh_dim_names=("dp_shard",))


def _make_inputs(config: FSDP2CompareConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_transformer_model(config.model):
        shape = (config.batch_size, config.seq_len, config.hidden)
    else:
        shape = (config.batch_size, config.hidden)
    dtype = _torch_dtype(config)
    return torch.randn(*shape, device=device, dtype=dtype), torch.randn(*shape, device=device, dtype=dtype)


def _validate_model_config(config: FSDP2CompareConfig) -> None:
    if config.model not in ("mlp", "transformer", "transformer_split_qkv"):
        raise ValueError(f"Unknown model: {config.model}.")
    if config.seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {config.seq_len}.")
    if config.heads <= 0:
        raise ValueError(f"heads must be positive, got {config.heads}.")
    if config.matrix_max_active_full_param_buffers is not None and config.matrix_max_active_full_param_buffers <= 0:
        raise ValueError("matrix_max_active_full_param_buffers must be positive.")
    if config.matrix_max_active_full_param_numel is not None and config.matrix_max_active_full_param_numel <= 0:
        raise ValueError("matrix_max_active_full_param_numel must be positive.")
    if config.matrix_max_active_full_param_memory_mb < 0:
        raise ValueError("matrix_max_active_full_param_memory_mb must be non-negative.")
    if _is_transformer_model(config.model) and config.hidden % config.heads != 0:
        raise ValueError(f"hidden={config.hidden} must be divisible by heads={config.heads}.")
    if config.model == "transformer_split_qkv" and config.intermediate % 2 != 0:
        raise ValueError(f"intermediate={config.intermediate} must be divisible by 2 for transformer_split_qkv.")
    _torch_dtype(config)


def _wrap_policy(config: FSDP2CompareConfig):
    return lambda module: _is_unit(module, config)


def _is_unit(module: nn.Module, config: FSDP2CompareConfig) -> bool:
    if config.unit == "linear":
        return isinstance(module, nn.Linear)
    if config.unit == "block" and config.model == "mlp":
        return isinstance(module, MLPBlock)
    if config.unit == "block" and config.model == "transformer":
        return isinstance(module, TransformerBlock)
    if config.unit == "block" and config.model == "transformer_split_qkv":
        return isinstance(module, SplitQKVTransformerBlock)
    raise ValueError(f"Unknown unit: {config.unit}.")


def _is_transformer_model(model: str) -> bool:
    return model in {"transformer", "transformer_split_qkv"}


def _run_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | MatrixFSDPOptimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    device: str,
) -> None:
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model(x), target)
        loss.backward()
        optimizer.step()
    _synchronize(device)


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (output.float() - target.float()).pow(2).mean()


def _torch_dtype(config: FSDP2CompareConfig) -> torch.dtype:
    if config.dtype == "float32":
        return torch.float32
    if config.dtype == "bfloat16":
        return torch.bfloat16
    if config.dtype == "float16":
        return torch.float16
    raise ValueError(f"Unknown dtype: {config.dtype}.")


def _peak_memory_mb(device: str, torch_device: torch.device) -> float:
    if device != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(torch_device) / (1024.0 * 1024.0)


def _memory_snapshot_mb(device: str, torch_device: torch.device) -> tuple[float, float, float, float]:
    if device != "cuda":
        return 0.0, 0.0, 0.0, 0.0
    scale = 1024.0 * 1024.0
    return (
        torch.cuda.memory_allocated(torch_device) / scale,
        torch.cuda.max_memory_allocated(torch_device) / scale,
        torch.cuda.memory_reserved(torch_device) / scale,
        torch.cuda.max_memory_reserved(torch_device) / scale,
    )


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _cleanup_after_mode(device: str) -> None:
    gc.collect()
    clear_global_full_param_buffer_pool()
    if device == "cuda":
        torch.cuda.empty_cache()
        dist.barrier(device_ids=[torch.cuda.current_device()])
        return
    dist.barrier()


def _init_benchmark_process_group(rank: int, config: FSDP2CompareConfig) -> None:
    if config.device == "cuda":
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{config.init_file}",
            rank=rank,
            world_size=config.world_size,
            device_id=torch.device("cuda", rank),
        )
        return

    os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{config.init_file}",
        rank=rank,
        world_size=config.world_size,
    )


def _loopback_interface_name() -> str:
    names = {name for _, name in socket.if_nameindex()}
    for candidate in ("lo0", "lo"):
        if candidate in names:
            return candidate
    return "lo"


def _uses_fsdp2(modes: Sequence[str]) -> bool:
    return any(mode.startswith("fsdp2") for mode in modes)


def _format_count_mapping(counts: Mapping[str, int]) -> str:
    normalized = Counter({str(key): int(value) for key, value in counts.items() if int(value) != 0})
    if not normalized:
        return "-"
    return ",".join(f"{key}:{normalized[key]}" for key in sorted(normalized))


def _format_table_row(values: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))


if __name__ == "__main__":
    raise SystemExit(main())
