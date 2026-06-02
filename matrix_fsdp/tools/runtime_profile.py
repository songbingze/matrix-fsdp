from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import socket
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F

from matrix_fsdp import (
    MixedMuonAdamWOptimizer,
    PrefetchProfileResult,
    MatrixFSDPOptimizer,
    format_runtime_events,
    format_param_group_summary,
    make_muon_shard_aware_group_planner,
    matrix_fully_shard,
    summarize_runtime_events,
    summarize_param_groups,
)


@dataclass(frozen=True)
class RuntimeProfileConfig:
    world_size: int = 1
    device: str = "cpu"
    mode: str = "prefetch_bucket_copy_in"
    unit: str = "block"
    model: str = "mlp"
    layers: int = 2
    hidden: int = 128
    intermediate: int = 512
    seq_len: int = 128
    heads: int = 8
    batch_size: int = 4
    optimizer: str = "sgd"
    dtype: str = "float32"
    warmup_steps: int = 1
    profile_steps: int = 2
    profile_memory_limit_mb: float = 0.0
    max_active_full_param_buffers: int | None = None
    max_active_full_param_numel: int | None = None
    max_active_full_param_memory_mb: float = 0.0
    steps: int = 1
    output_format: str = "text"
    output_file: str = ""
    init_file: str = ""


@dataclass(frozen=True)
class RuntimeProfileStepStats:
    avg_step_ms: float
    peak_memory_mb: float


class MLPBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.act = nn.GELU()
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class MLPStack(nn.Module):
    def __init__(self, layers: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MLPBlock(hidden, intermediate) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
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


class SplitQKVTransformerBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, heads: int) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}.")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = MLPBlock(hidden, intermediate)

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


class TransformerStack(nn.Module):
    def __init__(
        self,
        block_cls: type[TransformerBlock] | type[SplitQKVTransformerBlock],
        layers: int,
        hidden: int,
        intermediate: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([block_cls(hidden, intermediate, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def run_profile(config: RuntimeProfileConfig) -> dict[str, Any]:
    _validate_config(config)
    if config.world_size == 1:
        return _profile_worker_impl(rank=0, config=config)

    if config.device == "cuda" and torch.cuda.device_count() < config.world_size:
        raise RuntimeError(
            f"CUDA runtime profile requires at least {config.world_size} devices, "
            f"found {torch.cuda.device_count()}."
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        init_file = config.init_file or os.path.join(tmpdir, "matrix_fsdp_runtime_profile_init")
        output_file = config.output_file or os.path.join(tmpdir, "matrix_fsdp_runtime_profile.json")
        worker_config = dataclasses.replace(config, init_file=init_file, output_file=output_file)
        mp.spawn(_profile_worker, args=(worker_config,), nprocs=config.world_size, join=True)
        with open(output_file, encoding="utf-8") as result_file:
            return json.load(result_file)


def format_profile_report(result: dict[str, Any]) -> str:
    step_stats = result["step_stats"]
    lines = [
        "MatrixFSDP runtime profile",
        (
            f"mode={result['config']['mode']} model={result['config']['model']} unit={result['config']['unit']} "
            f"device={result['config']['device']} world_size={result['config']['world_size']} "
            f"dtype={result['config']['dtype']}"
        ),
        f"avg_step_ms={step_stats['avg_step_ms']:.3f} peak_mem_mb={step_stats['peak_memory_mb']:.1f}",
        "",
        format_param_group_summary(result.get("param_group_summary", result["unit_summary"])),
        "",
        format_runtime_events(result["runtime_summary"]),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile MatrixFSDP runtime events and scheduler decisions.")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--mode",
        choices=(
            "auto_finalize",
            "prefetch",
            "prefetch_cap1",
            "prefetch_bucket_reduce_scatter",
            "prefetch_bucket_copy_in",
            "zero_copy_grad_bucket",
            "prefetch_adaptive",
            "prefetch_profile_guided",
            "matrix_owner_muon",
            "matrix_owner_muon_role_greedy",
            "matrix_owner_muon_role_greedy_custom_collective",
        ),
        default="prefetch_bucket_copy_in",
    )
    parser.add_argument("--unit", choices=("linear", "block"), default="block")
    parser.add_argument("--model", choices=("mlp", "transformer", "transformer_split_qkv"), default="mlp")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--optimizer", choices=("sgd", "adamw", "muon"), default="sgd")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--profile-memory-limit-mb", type=float, default=0.0)
    parser.add_argument("--max-active-full-param-buffers", type=int, default=None)
    parser.add_argument("--max-active-full-param-numel", type=int, default=None)
    parser.add_argument("--max-active-full-param-memory-mb", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--format", choices=("text", "json"), default="text", dest="output_format")
    parser.add_argument("--output", default="", dest="output_file", help="Optional file path for text or JSON output.")
    args = parser.parse_args(argv)

    config = RuntimeProfileConfig(
        world_size=args.world_size,
        device=args.device,
        mode=args.mode,
        unit=args.unit,
        model=args.model,
        layers=args.layers,
        hidden=args.hidden,
        intermediate=args.intermediate,
        seq_len=args.seq_len,
        heads=args.heads,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        dtype=args.dtype,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        profile_memory_limit_mb=args.profile_memory_limit_mb,
        max_active_full_param_buffers=args.max_active_full_param_buffers,
        max_active_full_param_numel=args.max_active_full_param_numel,
        max_active_full_param_memory_mb=args.max_active_full_param_memory_mb,
        steps=args.steps,
        output_format=args.output_format,
        output_file=args.output_file,
    )
    result = run_profile(config)
    output = json.dumps(_json_safe(result), indent=2) if config.output_format == "json" else format_profile_report(result)
    if config.output_file:
        with open(config.output_file, "w", encoding="utf-8") as profile_file:
            profile_file.write(output)
            profile_file.write("\n")
    else:
        print(output)
    return 0


def _profile_worker(rank: int, config: RuntimeProfileConfig) -> None:
    result = _profile_worker_impl(rank=rank, config=config)
    if rank == 0:
        with open(config.output_file, "w", encoding="utf-8") as result_file:
            json.dump(_json_safe(result), result_file, indent=2)


def _profile_worker_impl(rank: int, config: RuntimeProfileConfig) -> dict[str, Any]:
    initialized_process_group = False
    if config.world_size > 1:
        if config.device == "cuda":
            torch.cuda.set_device(rank)
        if config.device == "cpu":
            os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
        backend = "nccl" if config.device == "cuda" else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method=f"file://{config.init_file}",
            rank=rank,
            world_size=config.world_size,
        )
        initialized_process_group = True
    try:
        device = torch.device("cuda", rank) if config.device == "cuda" else torch.device("cpu")
        torch.manual_seed(0)
        model = _make_model(config).to(device=device, dtype=_torch_dtype(config.dtype))
        mesh = _make_mesh(config)
        model, optimizer = _prepare_mode(model, mesh, config)
        x, target = _make_inputs(config, device)

        _run_steps(model, optimizer, x, target, config.warmup_steps, config.device)
        _profile_prefetch_budget_if_needed(model, optimizer, x, target, config, device)
        _clear_runtime_trace(optimizer)
        stats = _measure_steps(model, optimizer, x, target, config.steps, config.device, device)

        param_group_summary = summarize_param_groups(model)
        result = {
            "rank": rank,
            "config": asdict(config),
            "step_stats": asdict(stats),
            "param_group_summary": param_group_summary,
            "unit_summary": param_group_summary,
            "runtime_summary": summarize_runtime_events(model),
        }
        if config.world_size > 1:
            dist.barrier()
        return _json_safe(result)
    finally:
        if initialized_process_group:
            dist.destroy_process_group()


def _prepare_mode(
    model: nn.Module,
    mesh: DeviceMesh | None,
    config: RuntimeProfileConfig,
) -> tuple[nn.Module, MatrixFSDPOptimizer]:
    if config.mode in (
        "matrix_owner_muon",
        "matrix_owner_muon_role_greedy",
        "matrix_owner_muon_role_greedy_custom_collective",
    ):
        return _prepare_matrix_owner_muon(model, mesh, config)

    forward_prefetch = config.mode.startswith("prefetch") or config.mode == "zero_copy_grad_bucket"
    backward_prefetch = config.mode.startswith("prefetch") or config.mode == "zero_copy_grad_bucket"
    max_unsharded_prefetch_units: int | None = None
    prefetch_policy = "static"
    if config.mode == "prefetch_cap1":
        max_unsharded_prefetch_units = 1
    elif config.mode in ("prefetch_bucket_reduce_scatter", "prefetch_bucket_copy_in", "zero_copy_grad_bucket"):
        max_unsharded_prefetch_units = 1
    elif config.mode == "prefetch_adaptive":
        prefetch_policy = "adaptive"
    elif config.mode == "prefetch_profile_guided":
        prefetch_policy = "profile_guided"
    backward_reduce_strategy = (
        "bucket_reduce_scatter"
        if config.mode in ("prefetch_bucket_reduce_scatter", "prefetch_bucket_copy_in", "zero_copy_grad_bucket")
        else "flat"
    )
    use_zero_copy_grad_bucket = config.mode in ("prefetch_bucket_reduce_scatter", "zero_copy_grad_bucket")

    sharded_model = matrix_fully_shard(
        model,
        mesh,
        wrap_policy=_wrap_policy(config),
        reshard_after_forward=True,
        forward_prefetch=forward_prefetch,
        backward_prefetch=backward_prefetch,
        finalize_after_backward=True,
        backward_reduce_strategy=backward_reduce_strategy,
        use_zero_copy_grad_bucket=use_zero_copy_grad_bucket,
    )
    optimizer = MatrixFSDPOptimizer(
        _make_optimizer(sharded_model.parameters(), config),
        sharded_model,
        max_unsharded_prefetch_units=max_unsharded_prefetch_units,
        backward_prefetch_timing="pre_backward",
        prefetch_policy=prefetch_policy,
        max_active_full_param_buffers=config.max_active_full_param_buffers,
        max_active_full_param_numel=config.max_active_full_param_numel,
        max_active_full_param_memory_mb=config.max_active_full_param_memory_mb or None,
    )
    return sharded_model, optimizer


def _prepare_matrix_owner_muon(
    model: nn.Module,
    mesh: DeviceMesh | None,
    config: RuntimeProfileConfig,
) -> tuple[nn.Module, MatrixFSDPOptimizer]:
    if config.optimizer != "muon":
        raise ValueError("matrix_owner_muon modes require --optimizer muon.")
    owner_assignment = "role_greedy" if "role_greedy" in config.mode else "rotate"
    matrix_collective_backend = "custom" if config.mode.endswith("_custom_collective") else "owner_broadcast"
    sharded_model = matrix_fully_shard(
        model,
        mesh,
        wrap_policy=_wrap_policy(config),
        auto_shard_hints=True,
        group_planner=make_muon_shard_aware_group_planner(owner_assignment=owner_assignment),
        reshard_after_forward=True,
        forward_prefetch=True,
        backward_prefetch=True,
        finalize_after_backward=True,
        backward_reduce_strategy="bucket_reduce_scatter",
        use_zero_copy_grad_bucket=True,
        matrix_collective_backend=matrix_collective_backend,
    )
    optimizer = MatrixFSDPOptimizer(
        _make_optimizer(sharded_model.parameters(), config),
        sharded_model,
        max_unsharded_prefetch_units=1,
        backward_prefetch_timing="post_reshard",
        max_active_full_param_buffers=config.max_active_full_param_buffers,
        max_active_full_param_numel=config.max_active_full_param_numel,
        max_active_full_param_memory_mb=config.max_active_full_param_memory_mb or None,
    )
    return sharded_model, optimizer


def _profile_prefetch_budget_if_needed(
    model: nn.Module,
    optimizer: MatrixFSDPOptimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    config: RuntimeProfileConfig,
    device: torch.device,
) -> None:
    if config.mode != "prefetch_profile_guided" or config.profile_steps <= 0:
        return
    results = []
    for budget in (None, 0, 1, 2):
        optimizer.scheduler.set_prefetch_budget(budget)
        _clear_runtime_trace(optimizer)
        stats = _measure_steps(model, optimizer, x, target, config.profile_steps, config.device, device)
        results.append(
            PrefetchProfileResult(
                budget=budget,
                avg_step_ms=stats.avg_step_ms,
                peak_memory_mb=stats.peak_memory_mb,
            )
        )
        _cleanup_after_profile_step(config.device)
    optimizer.scheduler.set_profile_results(results)
    optimizer.scheduler.select_profiled_budget(memory_limit_mb=config.profile_memory_limit_mb or None)


def _measure_steps(
    model: nn.Module,
    optimizer: MatrixFSDPOptimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    device_type: str,
    device: torch.device,
) -> RuntimeProfileStepStats:
    if steps <= 0:
        return RuntimeProfileStepStats(avg_step_ms=0.0, peak_memory_mb=_peak_memory_mb(device_type, device))
    _synchronize(device_type)
    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    start = time.perf_counter()
    _run_steps(model, optimizer, x, target, steps, device_type)
    _synchronize(device_type)
    elapsed = time.perf_counter() - start
    avg_step_ms = elapsed * 1000.0 / steps
    peak_memory_mb = _peak_memory_mb(device_type, device)
    if dist.is_available() and dist.is_initialized():
        avg_tensor = torch.tensor(avg_step_ms, device=device)
        peak_tensor = torch.tensor(peak_memory_mb, device=device)
        dist.all_reduce(avg_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        avg_step_ms = avg_tensor.item()
        peak_memory_mb = peak_tensor.item()
    return RuntimeProfileStepStats(avg_step_ms=avg_step_ms, peak_memory_mb=peak_memory_mb)


def _run_steps(
    model: nn.Module,
    optimizer: MatrixFSDPOptimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    device_type: str,
) -> None:
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target).pow(2).mean()
        loss.backward()
        optimizer.step()
    _synchronize(device_type)


def _clear_runtime_trace(optimizer: MatrixFSDPOptimizer) -> None:
    for unit in optimizer.runtime_param_groups:
        unit.runtime_events.clear()
    optimizer.scheduler.reset_runtime_counters()


def _make_optimizer(params, config: RuntimeProfileConfig) -> torch.optim.Optimizer:
    params = list(params)
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=0.01)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=0.001)
    if config.optimizer == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("torch.optim.Muon is not available in this PyTorch build.")
        muon_params = [param for param in params if param.ndim == 2 and param.numel() > 0]
        adamw_params = [param for param in params if param.ndim != 2 and param.numel() > 0]
        return MixedMuonAdamWOptimizer(muon_params, adamw_params)
    raise ValueError(f"Unknown optimizer: {config.optimizer}.")


def _make_model(config: RuntimeProfileConfig) -> nn.Module:
    if config.model == "mlp":
        return MLPStack(config.layers, config.hidden, config.intermediate)
    if config.model == "transformer":
        return TransformerStack(TransformerBlock, config.layers, config.hidden, config.intermediate, config.heads)
    if config.model == "transformer_split_qkv":
        return TransformerStack(
            SplitQKVTransformerBlock,
            config.layers,
            config.hidden,
            config.intermediate,
            config.heads,
        )
    raise ValueError(f"Unknown model: {config.model}.")


def _make_mesh(config: RuntimeProfileConfig) -> DeviceMesh | None:
    if config.world_size == 1:
        return None
    return DeviceMesh(config.device, torch.arange(config.world_size), mesh_dim_names=("dp_shard",))


def _make_inputs(config: RuntimeProfileConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if config.model in ("transformer", "transformer_split_qkv"):
        shape = (config.batch_size, config.seq_len, config.hidden)
    else:
        shape = (config.batch_size, config.hidden)
    dtype = _torch_dtype(config.dtype)
    return torch.randn(*shape, device=device, dtype=dtype), torch.randn(*shape, device=device, dtype=dtype)


def _wrap_policy(config: RuntimeProfileConfig):
    return lambda module: _is_unit(module, config)


def _is_unit(module: nn.Module, config: RuntimeProfileConfig) -> bool:
    if config.unit == "linear":
        return isinstance(module, nn.Linear)
    if config.unit == "block" and config.model == "mlp":
        return isinstance(module, MLPBlock)
    if config.unit == "block" and config.model == "transformer":
        return isinstance(module, TransformerBlock)
    if config.unit == "block" and config.model == "transformer_split_qkv":
        return isinstance(module, SplitQKVTransformerBlock)
    raise ValueError(f"Unknown unit: {config.unit}.")


def _validate_config(config: RuntimeProfileConfig) -> None:
    if config.world_size <= 0:
        raise ValueError(f"world_size must be positive, got {config.world_size}.")
    if config.steps < 0:
        raise ValueError(f"steps must be non-negative, got {config.steps}.")
    if config.warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {config.warmup_steps}.")
    if config.profile_steps < 0:
        raise ValueError(f"profile_steps must be non-negative, got {config.profile_steps}.")
    if config.profile_memory_limit_mb < 0:
        raise ValueError(f"profile_memory_limit_mb must be non-negative, got {config.profile_memory_limit_mb}.")
    if config.max_active_full_param_buffers is not None and config.max_active_full_param_buffers <= 0:
        raise ValueError("max_active_full_param_buffers must be positive.")
    if config.max_active_full_param_numel is not None and config.max_active_full_param_numel <= 0:
        raise ValueError("max_active_full_param_numel must be positive.")
    if config.max_active_full_param_memory_mb < 0:
        raise ValueError("max_active_full_param_memory_mb must be non-negative.")
    if config.dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError(f"Unknown dtype: {config.dtype}.")
    if config.model in ("transformer", "transformer_split_qkv") and config.hidden % config.heads != 0:
        raise ValueError(f"hidden={config.hidden} must be divisible by heads={config.heads}.")
    if config.mode.startswith("matrix_owner_muon") and config.optimizer != "muon":
        raise ValueError("matrix_owner_muon modes require --optimizer muon.")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime profile requested but CUDA is not available.")


def _peak_memory_mb(device_type: str, device: torch.device) -> float:
    if device_type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    raise ValueError(f"Unknown dtype: {dtype}.")


def _synchronize(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()


def _cleanup_after_profile_step(device_type: str) -> None:
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _loopback_interface_name() -> str:
    names = {name for _, name in socket.if_nameindex()}
    for candidate in ("lo0", "lo"):
        if candidate in names:
            return candidate
    return "lo"


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
