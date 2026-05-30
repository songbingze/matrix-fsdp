import gc
import os
import socket
import tempfile
import time
import unittest
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard as torch_fully_shard

from matrix_fsdp import (
    MixedMuonAdamWOptimizer,
    make_muon_shard_aware_group_planner,
    matrix_fully_shard,
    MatrixFSDPOptimizer,
)

try:
    from matrix_fsdp.shard_hint import build_shard_hints
except ImportError:
    from matrix_fsdp.planning.shard_hint import build_shard_hints


@dataclass(frozen=True)
class DeepSeekV3EPTrainConfig:
    vocab_size: int = 64
    hidden_size: int = 32
    num_hidden_layers: int = 4
    first_k_dense_replace: int = 1
    q_lora_rank: int = 8
    kv_lora_rank: int = 8
    qk_rope_head_dim: int = 4
    intermediate_size: int = 64
    moe_intermediate_size: int = 16
    n_routed_experts: int = 4
    n_shared_experts: int = 1


class DeepSeekV3ExpertParallelTrainTest(unittest.TestCase):
    def test_two_rank_cpu_v3_like_ep_dispatch_train_step_with_dense_fsdp(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "deepseek_v3_ep_train_init")
            mp.spawn(
                _run_two_rank_v3_like_ep_train_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        os.environ.get("MATRIX_FSDP_RUN_DEEPSEEK_GPU_LARGE") == "1",
        "set MATRIX_FSDP_RUN_DEEPSEEK_GPU_LARGE=1 to run the large GPU EP train test",
    )
    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_large_v3_like_ep_dispatch_train_step_with_dense_fsdp(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "deepseek_v3_ep_large_gpu_train_init")
            mp.spawn(
                _run_two_rank_large_cuda_v3_like_ep_train_step,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )

    @unittest.skipUnless(
        os.environ.get("MATRIX_FSDP_RUN_DEEPSEEK_GPU_COMPARE_4B") == "1",
        "set MATRIX_FSDP_RUN_DEEPSEEK_GPU_COMPARE_4B=1 to compare MatrixFSDP with PyTorch FSDP2",
    )
    @unittest.skipUnless(
        dist.is_nccl_available() and torch.cuda.device_count() >= 2,
        "requires NCCL and at least 2 CUDA devices",
    )
    def test_two_rank_cuda_4b_seq8192_matrix_vs_fsdp2(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "deepseek_v3_ep_4b_seq8192_compare_init")
            mp.spawn(
                _run_two_rank_cuda_4b_seq8192_matrix_vs_fsdp2,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
            )


def _run_two_rank_v3_like_ep_train_step(rank: int, world_size: int, init_file: str) -> None:
    _run_v3_like_ep_train_step(
        rank,
        world_size,
        init_file,
        backend="gloo",
        device_type="cpu",
        config=DeepSeekV3EPTrainConfig(),
        batch_size=2,
        seq_len=4,
        steps=2,
        lr=0.03,
        emit_report=False,
    )


def _run_two_rank_large_cuda_v3_like_ep_train_step(rank: int, world_size: int, init_file: str) -> None:
    _run_v3_like_ep_train_step(
        rank,
        world_size,
        init_file,
        backend="nccl",
        device_type="cuda",
        config=DeepSeekV3EPTrainConfig(
            vocab_size=4096,
            hidden_size=1024,
            num_hidden_layers=12,
            first_k_dense_replace=2,
            q_lora_rank=256,
            kv_lora_rank=256,
            qk_rope_head_dim=64,
            intermediate_size=4096,
            moe_intermediate_size=1024,
            n_routed_experts=16,
            n_shared_experts=1,
        ),
        batch_size=2,
        seq_len=16,
        steps=1,
        lr=0.01,
        emit_report=True,
    )


def _run_two_rank_cuda_4b_seq8192_matrix_vs_fsdp2(rank: int, world_size: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cuda", rank)
        config = DeepSeekV3EPTrainConfig(
            vocab_size=4096,
            hidden_size=2048,
            num_hidden_layers=12,
            first_k_dense_replace=2,
            q_lora_rank=512,
            kv_lora_rank=512,
            qk_rope_head_dim=128,
            intermediate_size=8192,
            moe_intermediate_size=2048,
            n_routed_experts=32,
            n_shared_experts=1,
        )
        batch_size = int(os.environ.get("MATRIX_FSDP_COMPARE_BATCH_SIZE", "1"))
        seq_len = int(os.environ.get("MATRIX_FSDP_COMPARE_SEQ_LEN", "8192"))
        optimizer_name = os.environ.get("MATRIX_FSDP_COMPARE_OPTIMIZER", "adamw").lower()
        if optimizer_name == "adamw":
            default_lr = "0.0001"
        elif optimizer_name == "muon":
            default_lr = "0.03"
        else:
            default_lr = "0.01"
        lr = float(os.environ.get("MATRIX_FSDP_COMPARE_LR", default_lr))
        adamw_lr = float(os.environ.get("MATRIX_FSDP_COMPARE_ADAMW_LR", "0.001"))
        optimizer_scope = os.environ.get(
            "MATRIX_FSDP_COMPARE_OPTIMIZER_SCOPE",
            "dense" if optimizer_name == "muon" else "full_local",
        ).lower()
        use_activation_checkpointing = os.environ.get("MATRIX_FSDP_COMPARE_ACTIVATION_CHECKPOINT", "1") == "1"
        warmup_steps = int(os.environ.get("MATRIX_FSDP_COMPARE_WARMUP_STEPS", "1"))
        measured_steps = int(os.environ.get("MATRIX_FSDP_COMPARE_STEPS", "1"))
        checkpoint_before_shard = os.environ.get("MATRIX_FSDP_COMPARE_CHECKPOINT_BEFORE_SHARD", "0") == "1"
        trace_memory = os.environ.get("MATRIX_FSDP_COMPARE_MEMORY_TRACE", "0") == "1"
        default_matrix_backend = "custom" if optimizer_name == "muon" else "owner_broadcast"
        requested_matrix_backend = os.environ.get("MATRIX_FSDP_COMPARE_MATRIX_BACKEND", default_matrix_backend)
        matrix_collective_backend = (
            "owner_broadcast" if optimizer_name == "adamw" else requested_matrix_backend
        )

        matrix = _run_compare_impl(
            "matrix",
            rank,
            world_size,
            device,
            config,
            batch_size=batch_size,
            seq_len=seq_len,
            optimizer_name=optimizer_name,
            use_activation_checkpointing=use_activation_checkpointing,
            matrix_collective_backend=matrix_collective_backend,
            optimizer_scope=optimizer_scope,
            lr=lr,
            adamw_lr=adamw_lr,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            checkpoint_before_shard=checkpoint_before_shard,
            trace_memory=trace_memory,
        )
        fsdp2 = _run_compare_impl(
            "fsdp2",
            rank,
            world_size,
            device,
            config,
            batch_size=batch_size,
            seq_len=seq_len,
            optimizer_name=optimizer_name,
            use_activation_checkpointing=use_activation_checkpointing,
            matrix_collective_backend=matrix_collective_backend,
            optimizer_scope=optimizer_scope,
            lr=lr,
            adamw_lr=adamw_lr,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            checkpoint_before_shard=checkpoint_before_shard,
            trace_memory=trace_memory,
        )

        _assert_compare_stats_close(matrix, fsdp2, optimizer_name=optimizer_name)
        _emit_compare_report(
            rank,
            world_size,
            config,
            batch_size,
            seq_len,
            optimizer_name,
            optimizer_scope,
            use_activation_checkpointing,
            matrix_collective_backend,
            warmup_steps,
            measured_steps,
            checkpoint_before_shard,
            trace_memory,
            matrix,
            fsdp2,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_v3_like_ep_train_step(
    rank: int,
    world_size: int,
    init_file: str,
    *,
    backend: str,
    device_type: str,
    config: DeepSeekV3EPTrainConfig,
    batch_size: int,
    seq_len: int,
    steps: int,
    lr: float,
    emit_report: bool,
) -> None:
    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", _loopback_interface_name())
    if device_type == "cuda":
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
        torch.manual_seed(1234)
        with torch.device(device):
            model = DeepSeekV3EPTrainModel(config, ep_rank=rank, ep_world_size=world_size)
        param_stats = _local_param_stats(model)
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        mesh = DeviceMesh(device_type, torch.arange(world_size))

        dense_planner = make_muon_shard_aware_group_planner(owner_assignment="role_greedy")
        for block in model.layers:
            ignored_params = _routed_expert_params(block)
            matrix_fully_shard(
                block,
                mesh,
                ignored_params=ignored_params,
                shard_hints=_dense_shard_hints(block, ignored_params),
                auto_shard_hints=False,
                group_planner=dense_planner,
                reshard_after_forward=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
                use_saved_tensor_hooks=False,
                use_zero_copy_grad_bucket=False,
            )

        units = _collect_matrix_fsdp_groups(model)
        assert len(units) == config.num_hidden_layers
        for unit in units:
            assert all("local_experts" not in fqn for fqn in _param_group_fqns(unit))

        first_moe = model.layers[config.first_k_dense_replace].mlp
        assert isinstance(first_moe, DeepSeekV3EPMoE)
        first_expert_weight = first_moe.local_experts[0].gate_proj.weight
        assert not hasattr(first_expert_weight, "_matrix_fsdp_param_group_ref")
        initial_expert_weight = first_expert_weight.detach().clone()

        optimizer = MatrixFSDPOptimizer(torch.optim.SGD(model.parameters(), lr=lr), units)
        assert _optimizer_matrix_groups(optimizer) == units

        for step in range(steps):
            torch.manual_seed(2000 + rank * 17 + step)
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert not torch.allclose(first_expert_weight, initial_expert_weight)
        remote_dispatches = [
            int(block.mlp.dispatcher.last_send_counts[1 - rank].item())
            for block in model.layers[config.first_k_dense_replace :]
            if isinstance(block.mlp, DeepSeekV3EPMoE)
        ]
        assert remote_dispatches and all(count > 0 for count in remote_dispatches)

        for unit in units:
            assert _lifecycle_state_name(unit) == "SHARDED"

        if emit_report:
            _emit_train_report(rank, device, config, param_stats, batch_size=batch_size, seq_len=seq_len)

        dist.barrier()
    finally:
        dist.destroy_process_group()


class DeepSeekV3EPTrainModel(nn.Module):
    def __init__(self, config: DeepSeekV3EPTrainConfig, *, ep_rank: int, ep_world_size: int) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DeepSeekV3EPDecoderLayer(config, layer_idx, ep_rank=ep_rank, ep_world_size=ep_world_size)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.lm_head(self.norm(hidden_states))


class DeepSeekV3EPDecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV3EPTrainConfig,
        layer_idx: int,
        *,
        ep_rank: int,
        ep_world_size: int,
    ) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.self_attn = DeepSeekV3MLA(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp = (
            DeepSeekV3DenseMLP(config.hidden_size, config.intermediate_size)
            if layer_idx < config.first_k_dense_replace
            else DeepSeekV3EPMoE(config, ep_rank=ep_rank, ep_world_size=ep_world_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class DeepSeekV3MLA(nn.Module):
    def __init__(self, config: DeepSeekV3EPTrainConfig) -> None:
        super().__init__()
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.LayerNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, config.hidden_size, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = nn.LayerNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        kv_and_rope = self.kv_a_proj_with_mqa(hidden_states)
        kv_width = self.kv_a_layernorm.normalized_shape[0]
        kv, rope = torch.split(kv_and_rope, (kv_width, kv_and_rope.shape[-1] - kv_width), dim=-1)
        value = self.kv_b_proj(self.kv_a_layernorm(kv))
        rope_bias = rope.mean(dim=-1, keepdim=True)
        return self.o_proj(F.silu(query + value + rope_bias))


class DeepSeekV3DenseMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DeepSeekV3EPMoE(nn.Module):
    def __init__(self, config: DeepSeekV3EPTrainConfig, *, ep_rank: int, ep_world_size: int) -> None:
        super().__init__()
        if config.n_routed_experts % ep_world_size != 0:
            raise ValueError("n_routed_experts must be divisible by ep_world_size.")
        self.n_routed_experts = config.n_routed_experts
        self.experts_per_rank = config.n_routed_experts // ep_world_size
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.local_experts = nn.ModuleList(
            DeepSeekV3DenseMLP(config.hidden_size, config.moe_intermediate_size)
            for _ in range(self.experts_per_rank)
        )
        self.shared_experts = DeepSeekV3DenseMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
        )
        self.dispatcher = TorchTitanStyleEPTokenDispatcher(
            n_routed_experts=config.n_routed_experts,
            experts_per_rank=self.experts_per_rank,
            ep_rank=ep_rank,
            ep_world_size=ep_world_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        gate_logits = self.gate(hidden_states).reshape(flat_hidden.shape[0], self.n_routed_experts)
        route_expert_ids = self.dispatcher.deterministic_route_ids(flat_hidden.shape[0], flat_hidden.device)
        route_weights = torch.sigmoid(gate_logits.gather(-1, route_expert_ids.unsqueeze(-1)).squeeze(-1))
        routed = self.dispatcher.dispatch_and_combine(
            flat_hidden,
            route_expert_ids,
            route_weights,
            tuple(self.local_experts),
        )
        routed = routed.view_as(hidden_states)
        return routed + self.shared_experts(hidden_states)


class TorchTitanStyleEPTokenDispatcher(nn.Module):
    """
    Minimal TorchTitan-style EP dispatcher for a training smoke test.

    Each rank owns a contiguous range of global experts. Tokens are grouped by
    expert owner rank, exchanged with all-to-all, processed by local experts,
    and sent back to their source rank for combine.
    """

    def __init__(
        self,
        *,
        n_routed_experts: int,
        experts_per_rank: int,
        ep_rank: int,
        ep_world_size: int,
    ) -> None:
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.experts_per_rank = experts_per_rank
        self.ep_rank = ep_rank
        self.ep_world_size = ep_world_size
        self.register_buffer("last_send_counts", torch.zeros(ep_world_size, dtype=torch.long), persistent=False)
        self.register_buffer("last_recv_counts", torch.zeros(ep_world_size, dtype=torch.long), persistent=False)

    def deterministic_route_ids(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        token_ids = torch.arange(num_tokens, device=device, dtype=torch.long)
        return (token_ids + self.ep_rank * num_tokens) % self.n_routed_experts

    def dispatch_and_combine(
        self,
        flat_hidden: torch.Tensor,
        route_expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
        local_experts: tuple[nn.Module, ...],
    ) -> torch.Tensor:
        owner_ranks = torch.div(route_expert_ids, self.experts_per_rank, rounding_mode="floor")
        local_expert_ids = route_expert_ids.remainder(self.experts_per_rank)
        send_counts = torch.bincount(owner_ranks, minlength=self.ep_world_size).to(torch.long)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        self.last_send_counts.copy_(send_counts.detach().cpu())
        self.last_recv_counts.copy_(recv_counts.detach().cpu())

        send_order = torch.argsort(owner_ranks, stable=True)
        send_hidden = flat_hidden.index_select(0, send_order).contiguous()
        send_local_expert_ids = local_expert_ids.index_select(0, send_order).contiguous()

        send_splits = [int(value) for value in send_counts.tolist()]
        recv_splits = [int(value) for value in recv_counts.tolist()]
        recv_total = int(recv_counts.sum().item())
        recv_hidden = dist_nn.all_to_all_single(
            flat_hidden.new_empty((recv_total, flat_hidden.shape[-1])),
            send_hidden,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        recv_local_expert_ids = torch.empty(recv_total, dtype=torch.long, device=flat_hidden.device)
        dist.all_to_all_single(
            recv_local_expert_ids,
            send_local_expert_ids,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )

        recv_output = self._run_local_experts(recv_hidden, recv_local_expert_ids, local_experts)
        returned_output = dist_nn.all_to_all_single(
            flat_hidden.new_empty(send_hidden.shape),
            recv_output.contiguous(),
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
        )

        weighted_output = returned_output * route_weights.index_select(0, send_order).unsqueeze(-1)
        combined = flat_hidden.new_zeros(flat_hidden.shape)
        combined.scatter_add_(0, send_order.unsqueeze(-1).expand_as(weighted_output), weighted_output)
        return combined

    def _run_local_experts(
        self,
        recv_hidden: torch.Tensor,
        recv_local_expert_ids: torch.Tensor,
        local_experts: tuple[nn.Module, ...],
    ) -> torch.Tensor:
        output = recv_hidden.new_zeros(recv_hidden.shape)
        for expert_id, expert in enumerate(local_experts):
            indices = torch.nonzero(recv_local_expert_ids == expert_id, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            expert_hidden = recv_hidden.index_select(0, indices)
            output.index_copy_(0, indices, expert(expert_hidden))
        return output


def _run_compare_impl(
    impl: str,
    rank: int,
    world_size: int,
    device: torch.device,
    config: DeepSeekV3EPTrainConfig,
    *,
    batch_size: int,
    seq_len: int,
    optimizer_name: str,
    use_activation_checkpointing: bool,
    matrix_collective_backend: str,
    optimizer_scope: str,
    lr: float,
    adamw_lr: float,
    warmup_steps: int,
    measured_steps: int,
    checkpoint_before_shard: bool,
    trace_memory: bool,
) -> dict[str, float | int | tuple[float, ...]]:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if measured_steps <= 0:
        raise ValueError("measured_steps must be positive.")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(1234)
    with torch.device(device):
        model = DeepSeekV3EPTrainModel(config, ep_rank=rank, ep_world_size=world_size)
    param_stats = _local_param_stats(model)
    mesh = DeviceMesh("cuda", torch.arange(world_size))
    if use_activation_checkpointing and checkpoint_before_shard:
        _apply_torchtitan_activation_checkpointing(model)
    units = _shard_model_for_compare(model, mesh, impl, matrix_collective_backend=matrix_collective_backend)
    if use_activation_checkpointing and not checkpoint_before_shard:
        _apply_torchtitan_activation_checkpointing(model)
    optimizer = _make_compare_optimizer(
        model,
        units,
        impl=impl,
        optimizer_name=optimizer_name,
        optimizer_scope=optimizer_scope,
        lr=lr,
        adamw_lr=adamw_lr,
    )
    optimizer_stats = _optimizer_param_stats(optimizer)

    grad_stats: dict[str, float | int] = {}
    dist.barrier()

    for step in range(warmup_steps):
        input_ids, labels = _make_compare_batch(
            config,
            rank,
            device,
            batch_size=batch_size,
            seq_len=seq_len,
            step=step,
        )
        logits = model(input_ids)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del logits, loss, input_ids, labels

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    last_loss = None
    trace = _empty_memory_trace()
    if trace_memory:
        _record_memory_trace(trace, "start", device)
    for step in range(measured_steps):
        input_ids, labels = _make_compare_batch(
            config,
            rank,
            device,
            batch_size=batch_size,
            seq_len=seq_len,
            step=warmup_steps + step,
        )
        logits = model(input_ids)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        assert torch.isfinite(loss)
        if trace_memory and step == measured_steps - 1:
            _record_memory_trace(trace, "after_forward", device)
        loss.backward()
        if trace_memory and step == measured_steps - 1:
            _record_memory_trace(trace, "after_backward", device)
        if step == measured_steps - 1:
            grad_stats = _grad_stats(model)
            last_loss = float(loss.detach().cpu())
        optimizer.step()
        if trace_memory and step == measured_steps - 1:
            _record_memory_trace(trace, "after_step", device)
        optimizer.zero_grad()
        if trace_memory and step == measured_steps - 1:
            _record_memory_trace(trace, "after_zero_grad", device)
        del logits, loss, input_ids, labels
    end.record()
    torch.cuda.synchronize(device)
    wall_time_s = (time.perf_counter() - wall_start) / measured_steps
    cuda_time_s = (start.elapsed_time(end) / 1000.0) / measured_steps
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    _unshard_compare_model(model, units, impl)
    param_digest = _param_digest(model)
    result = {
        "loss": float(last_loss),
        "optimizer_name": optimizer_name,
        "optimizer_scope": optimizer_scope,
        **optimizer_stats,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "checkpoint_before_shard": int(checkpoint_before_shard),
        "cuda_time_s": cuda_time_s,
        "wall_time_s": wall_time_s,
        "max_allocated_mib": max_allocated / 1024 / 1024,
        "max_reserved_mib": max_reserved / 1024 / 1024,
        **trace,
        "global_param_numel": _global_param_numel(param_stats, world_size),
        "local_total_numel": param_stats["local_total_numel"],
        "fsdp_managed_dense_numel": param_stats["fsdp_managed_dense_numel"],
        "ignored_ep_expert_numel": param_stats["ignored_ep_expert_numel"],
        **grad_stats,
        **param_digest,
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def _empty_memory_trace() -> dict[str, float]:
    return {
        "trace_start_alloc_mib": 0.0,
        "trace_start_peak_mib": 0.0,
        "trace_after_forward_alloc_mib": 0.0,
        "trace_after_forward_peak_mib": 0.0,
        "trace_after_backward_alloc_mib": 0.0,
        "trace_after_backward_peak_mib": 0.0,
        "trace_after_step_alloc_mib": 0.0,
        "trace_after_step_peak_mib": 0.0,
        "trace_after_zero_grad_alloc_mib": 0.0,
        "trace_after_zero_grad_peak_mib": 0.0,
    }


def _record_memory_trace(trace: dict[str, float], phase: str, device: torch.device) -> None:
    torch.cuda.synchronize(device)
    trace[f"trace_{phase}_alloc_mib"] = torch.cuda.memory_allocated(device) / 1024 / 1024
    trace[f"trace_{phase}_peak_mib"] = torch.cuda.max_memory_allocated(device) / 1024 / 1024


def _apply_torchtitan_activation_checkpointing(model: nn.Module) -> None:
    checkpoint_wrapper_fn = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=False,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper_fn,
        check_fn=lambda module: isinstance(module, DeepSeekV3EPDecoderLayer),
    )


def _make_compare_optimizer(
    model: nn.Module,
    units: list[object],
    *,
    impl: str,
    optimizer_name: str,
    optimizer_scope: str,
    lr: float,
    adamw_lr: float,
) -> torch.optim.Optimizer | MatrixFSDPOptimizer | MixedMuonAdamWOptimizer:
    if optimizer_name == "muon":
        params = _compare_optimizer_params(model, optimizer_scope)
        scheduler_kwargs = (
            {"fsdp_param_groups": units, "max_unsharded_prefetch_units": 1}
            if impl == "matrix"
            else {}
        )
        return MixedMuonAdamWOptimizer(
            _muon_params(params),
            _adamw_params(params),
            muon_lr=lr,
            adamw_lr=adamw_lr,
            adamw_foreach=False,
            lazy_muon_init=True,
            **scheduler_kwargs,
        )

    params = _compare_optimizer_params(model, optimizer_scope)
    if optimizer_name == "sgd":
        torch_optimizer = torch.optim.SGD(params, lr=lr, foreach=False)
    elif optimizer_name == "adamw":
        torch_optimizer = torch.optim.AdamW(params, lr=lr, foreach=False)
    else:
        raise ValueError(f"Unknown compare optimizer: {optimizer_name}")
    if impl == "matrix":
        return MatrixFSDPOptimizer(
            torch_optimizer,
            units,
            max_unsharded_prefetch_units=1,
        )
    return torch_optimizer


def _muon_params(params) -> list[nn.Parameter]:
    return [param for param in params if param.ndim == 2 and param.numel() > 0]


def _adamw_params(params) -> list[nn.Parameter]:
    return [param for param in params if param.ndim != 2 and param.numel() > 0]


def _compare_optimizer_params(model: nn.Module, optimizer_scope: str) -> list[nn.Parameter]:
    if optimizer_scope not in {"dense", "managed_dense", "full_local"}:
        raise ValueError(
            "MATRIX_FSDP_COMPARE_OPTIMIZER_SCOPE must be one of "
            "'dense', 'managed_dense', or 'full_local'."
        )
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for fqn, param in model.named_parameters(remove_duplicate=True):
        normalized_fqn = fqn.replace("_checkpoint_wrapped_module.", "")
        if optimizer_scope != "full_local" and ".local_experts." in normalized_fqn:
            continue
        if optimizer_scope == "managed_dense" and not normalized_fqn.startswith("layers."):
            continue
        if id(param) in seen or param.numel() == 0:
            continue
        seen.add(id(param))
        params.append(param)
    return params


def _optimizer_param_stats(optimizer: object) -> dict[str, int]:
    target = optimizer.optimizer if isinstance(optimizer, MatrixFSDPOptimizer) else optimizer
    if isinstance(target, MixedMuonAdamWOptimizer):
        return {
            "optimizer_muon_numel": sum(param.numel() for param in target.muon_params),
            "optimizer_adamw_numel": sum(param.numel() for param in target.adamw_params),
            "optimizer_muon_local_numel": sum(_local_param_numel(param) for param in target.muon_params),
            "optimizer_adamw_local_numel": sum(_local_param_numel(param) for param in target.adamw_params),
        }
    params = []
    for group in getattr(target, "param_groups", ()):
        params.extend(group.get("params", ()))
    return {
        "optimizer_muon_numel": 0,
        "optimizer_adamw_numel": sum(param.numel() for param in params if param.numel() > 0),
        "optimizer_muon_local_numel": 0,
        "optimizer_adamw_local_numel": sum(_local_param_numel(param) for param in params if param.numel() > 0),
    }


def _local_param_numel(param: nn.Parameter) -> int:
    if hasattr(param, "to_local"):
        return int(param.to_local().numel())
    return int(param.numel())


def _shard_model_for_compare(
    model: DeepSeekV3EPTrainModel,
    mesh: DeviceMesh,
    impl: str,
    *,
    matrix_collective_backend: str,
) -> list[object]:
    if impl not in {"matrix", "fsdp2"}:
        raise ValueError(f"Unknown FSDP implementation: {impl}")
    dense_planner = make_muon_shard_aware_group_planner(owner_assignment="role_greedy")
    for block in model.layers:
        ignored_params = _routed_expert_params(block)
        if impl == "matrix":
            matrix_fully_shard(
                block,
                mesh,
                ignored_params=ignored_params,
                shard_hints=_dense_shard_hints(block, ignored_params),
                auto_shard_hints=False,
                group_planner=dense_planner,
                reshard_after_forward=True,
                finalize_after_backward=True,
                backward_reduce_strategy="bucket_reduce_scatter",
                use_saved_tensor_hooks=False,
                use_zero_copy_grad_bucket=False,
                matrix_collective_backend=matrix_collective_backend,
            )
        else:
            torch_fully_shard(
                block,
                mesh=mesh,
                ignored_params=ignored_params,
                reshard_after_forward=True,
            )
    units = _collect_matrix_fsdp_groups(model)
    if impl == "matrix":
        assert len(units) == model.config.num_hidden_layers
    return units


def _unshard_compare_model(model: DeepSeekV3EPTrainModel, units: list[object], impl: str) -> None:
    if impl == "matrix":
        for unit in units:
            unit.unshard()
        return
    for block in model.layers:
        unshard_target = block if hasattr(block, "unshard") else getattr(block, "_checkpoint_wrapped_module", block)
        if hasattr(unshard_target, "unshard"):
            unshard_target.unshard()


def _make_compare_batch(
    config: DeepSeekV3EPTrainConfig,
    rank: int,
    device: torch.device,
    *,
    batch_size: int,
    seq_len: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(9000 + step * 131 + rank)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        generator=generator,
        device=device,
    )
    labels = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        generator=generator,
        device=device,
    )
    return input_ids, labels


def _grad_stats(model: nn.Module) -> dict[str, float | int]:
    dense_sum = dense_sq = expert_sum = expert_sq = 0.0
    dense_numel = expert_numel = 0
    for fqn, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        grad_data = grad.detach().float()
        grad_sum = float(grad_data.sum().cpu())
        grad_sq = float(grad_data.square().sum().cpu())
        if ".local_experts." in fqn:
            expert_sum += grad_sum
            expert_sq += grad_sq
            expert_numel += grad_data.numel()
        else:
            dense_sum += grad_sum
            dense_sq += grad_sq
            dense_numel += grad_data.numel()
    return {
        "dense_grad_sum": dense_sum,
        "dense_grad_l2": dense_sq ** 0.5,
        "dense_grad_numel": dense_numel,
        "expert_grad_sum": expert_sum,
        "expert_grad_l2": expert_sq ** 0.5,
        "expert_grad_numel": expert_numel,
    }


def _param_digest(model: nn.Module) -> dict[str, float | int]:
    dense_sum = dense_sq = expert_sum = expert_sq = 0.0
    dense_numel = expert_numel = 0
    for fqn, param in model.named_parameters():
        data = param.detach().float()
        param_sum = float(data.sum().cpu())
        param_sq = float(data.square().sum().cpu())
        if ".local_experts." in fqn:
            expert_sum += param_sum
            expert_sq += param_sq
            expert_numel += data.numel()
        else:
            dense_sum += param_sum
            dense_sq += param_sq
            dense_numel += data.numel()
    return {
        "dense_param_sum": dense_sum,
        "dense_param_l2": dense_sq ** 0.5,
        "dense_param_numel": dense_numel,
        "expert_param_sum": expert_sum,
        "expert_param_l2": expert_sq ** 0.5,
        "expert_param_numel": expert_numel,
    }


def _assert_compare_stats_close(
    matrix: dict[str, float | int],
    fsdp2: dict[str, float | int],
    *,
    optimizer_name: str,
) -> None:
    keys: tuple[str, ...] = ()
    if optimizer_name != "muon":
        keys = (
            "loss",
            "expert_grad_sum",
            "expert_grad_l2",
            "dense_param_sum",
            "dense_param_l2",
            "expert_param_sum",
            "expert_param_l2",
        )
    for key in keys:
        actual = float(matrix[key])
        expected = float(fsdp2[key])
        tolerance = 1e-3 * max(1.0, abs(expected))
        assert abs(actual - expected) <= tolerance, (key, actual, expected, tolerance)
    for key in ("global_param_numel", "local_total_numel", "fsdp_managed_dense_numel", "ignored_ep_expert_numel"):
        assert int(matrix[key]) == int(fsdp2[key]), key


def _emit_compare_report(
    rank: int,
    world_size: int,
    config: DeepSeekV3EPTrainConfig,
    batch_size: int,
    seq_len: int,
    optimizer_name: str,
    optimizer_scope: str,
    use_activation_checkpointing: bool,
    matrix_collective_backend: str,
    warmup_steps: int,
    measured_steps: int,
    checkpoint_before_shard: bool,
    trace_memory: bool,
    matrix: dict[str, float | int],
    fsdp2: dict[str, float | int],
) -> None:
    local = torch.tensor(
        [
            matrix["cuda_time_s"],
            fsdp2["cuda_time_s"],
            matrix["wall_time_s"],
            fsdp2["wall_time_s"],
            matrix["max_allocated_mib"],
            fsdp2["max_allocated_mib"],
            matrix["max_reserved_mib"],
            fsdp2["max_reserved_mib"],
            matrix["loss"],
            fsdp2["loss"],
            matrix["expert_grad_l2"],
            fsdp2["expert_grad_l2"],
            matrix["dense_param_l2"],
            fsdp2["dense_param_l2"],
            matrix["optimizer_muon_numel"],
            fsdp2["optimizer_muon_numel"],
            matrix["optimizer_adamw_numel"],
            fsdp2["optimizer_adamw_numel"],
            matrix["optimizer_muon_local_numel"],
            fsdp2["optimizer_muon_local_numel"],
            matrix["optimizer_adamw_local_numel"],
            fsdp2["optimizer_adamw_local_numel"],
            matrix["trace_start_alloc_mib"],
            fsdp2["trace_start_alloc_mib"],
            matrix["trace_after_forward_alloc_mib"],
            fsdp2["trace_after_forward_alloc_mib"],
            matrix["trace_after_backward_alloc_mib"],
            fsdp2["trace_after_backward_alloc_mib"],
            matrix["trace_after_step_alloc_mib"],
            fsdp2["trace_after_step_alloc_mib"],
            matrix["trace_after_zero_grad_alloc_mib"],
            fsdp2["trace_after_zero_grad_alloc_mib"],
            matrix["trace_after_forward_peak_mib"],
            fsdp2["trace_after_forward_peak_mib"],
            matrix["trace_after_backward_peak_mib"],
            fsdp2["trace_after_backward_peak_mib"],
            matrix["trace_after_step_peak_mib"],
            fsdp2["trace_after_step_peak_mib"],
        ],
        dtype=torch.float64,
        device=torch.device("cuda", rank),
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    if rank != 0:
        return
    rows = torch.stack(gathered).cpu()
    report = (
        "DEEPSEEK_V3_4B_SEQ8192_MATRIX_VS_FSDP2 "
        f"optimizer={optimizer_name} "
        f"optimizer_scope={optimizer_scope} "
        f"activation_checkpoint={use_activation_checkpointing} "
        f"checkpoint_order={'before_shard' if checkpoint_before_shard else 'after_shard'} "
        f"matrix_grad_path=bucket_copy_in_reduce_scatter "
        f"matrix_param_gather_backend={matrix_collective_backend} "
        f"warmup_steps={warmup_steps} "
        f"measured_steps={measured_steps} "
        f"global_params={int(matrix['global_param_numel'])} "
        f"local_total_params={int(matrix['local_total_numel'])} "
        f"fsdp_dense_params={int(matrix['fsdp_managed_dense_numel'])} "
        f"ignored_ep_expert_params={int(matrix['ignored_ep_expert_numel'])} "
        f"layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} "
        f"routed_experts={config.n_routed_experts} "
        f"experts_per_rank={config.n_routed_experts // world_size} "
        f"batch={batch_size} seq={seq_len} "
        f"matrix_cuda_s={_rounded_tuple(rows[:, 0])} "
        f"fsdp2_cuda_s={_rounded_tuple(rows[:, 1])} "
        f"matrix_wall_s={_rounded_tuple(rows[:, 2])} "
        f"fsdp2_wall_s={_rounded_tuple(rows[:, 3])} "
        f"matrix_alloc_mib={_rounded_tuple(rows[:, 4])} "
        f"fsdp2_alloc_mib={_rounded_tuple(rows[:, 5])} "
        f"matrix_reserved_mib={_rounded_tuple(rows[:, 6])} "
        f"fsdp2_reserved_mib={_rounded_tuple(rows[:, 7])} "
        f"loss={_rounded_pair(rows[0, 8], rows[0, 9])} "
        f"expert_grad_l2={_rounded_pair(rows[0, 10], rows[0, 11])} "
        f"dense_param_l2_after_step={_rounded_pair(rows[0, 12], rows[0, 13])} "
        f"matrix_muon_params={_rounded_int_tuple(rows[:, 14])} "
        f"fsdp2_muon_params={_rounded_int_tuple(rows[:, 15])} "
        f"matrix_adamw_params={_rounded_int_tuple(rows[:, 16])} "
        f"fsdp2_adamw_params={_rounded_int_tuple(rows[:, 17])} "
        f"matrix_muon_local_params={_rounded_int_tuple(rows[:, 18])} "
        f"fsdp2_muon_local_params={_rounded_int_tuple(rows[:, 19])} "
        f"matrix_adamw_local_params={_rounded_int_tuple(rows[:, 20])} "
        f"fsdp2_adamw_local_params={_rounded_int_tuple(rows[:, 21])}"
    )
    if trace_memory:
        report += (
            f" trace_start_alloc_mib={_rounded_pair(rows[0, 22], rows[0, 23])} "
            f"trace_after_forward_alloc_mib={_rounded_pair(rows[0, 24], rows[0, 25])} "
            f"trace_after_backward_alloc_mib={_rounded_pair(rows[0, 26], rows[0, 27])} "
            f"trace_after_step_alloc_mib={_rounded_pair(rows[0, 28], rows[0, 29])} "
            f"trace_after_zero_grad_alloc_mib={_rounded_pair(rows[0, 30], rows[0, 31])} "
            f"trace_after_forward_peak_mib={_rounded_pair(rows[0, 32], rows[0, 33])} "
            f"trace_after_backward_peak_mib={_rounded_pair(rows[0, 34], rows[0, 35])} "
            f"trace_after_step_peak_mib={_rounded_pair(rows[0, 36], rows[0, 37])}"
        )
    print(report)


def _rounded_tuple(values: torch.Tensor) -> tuple[float, ...]:
    return tuple(round(float(value), 4) for value in values)


def _rounded_int_tuple(values: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(round(float(value))) for value in values)


def _rounded_pair(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    return (round(float(left), 6), round(float(right), 6))


def _routed_expert_params(module: nn.Module) -> set[nn.Parameter]:
    return {
        param
        for fqn, param in module.named_parameters()
        if ".local_experts." in fqn
    }


def _local_param_stats(model: DeepSeekV3EPTrainModel) -> dict[str, int]:
    ignored_expert_param_ids: set[int] = set()
    fsdp_managed_param_ids: set[int] = set()
    for block in model.layers:
        ignored_params = _routed_expert_params(block)
        ignored_expert_param_ids.update(id(param) for param in ignored_params)
        for param in block.parameters():
            if id(param) not in ignored_expert_param_ids:
                fsdp_managed_param_ids.add(id(param))

    return {
        "local_total_numel": sum(param.numel() for param in model.parameters()),
        "fsdp_managed_dense_numel": sum(
            param.numel()
            for param in model.parameters()
            if id(param) in fsdp_managed_param_ids
        ),
        "ignored_ep_expert_numel": sum(
            param.numel()
            for param in model.parameters()
            if id(param) in ignored_expert_param_ids
        ),
    }


def _global_param_numel(param_stats: dict[str, int], world_size: int) -> int:
    return (
        param_stats["local_total_numel"]
        - param_stats["ignored_ep_expert_numel"]
        + param_stats["ignored_ep_expert_numel"] * world_size
    )


def _emit_train_report(
    rank: int,
    device: torch.device,
    config: DeepSeekV3EPTrainConfig,
    param_stats: dict[str, int],
    *,
    batch_size: int,
    seq_len: int,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
    else:
        max_allocated = 0
        max_reserved = 0

    local_stats = torch.tensor(
        [
            param_stats["local_total_numel"],
            param_stats["fsdp_managed_dense_numel"],
            param_stats["ignored_ep_expert_numel"],
            max_allocated,
            max_reserved,
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered = [torch.empty_like(local_stats) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_stats)
    if rank != 0:
        return

    rows = torch.stack(gathered).cpu()
    allocated_mib = [round(float(value) / 1024 / 1024, 1) for value in rows[:, 3]]
    reserved_mib = [round(float(value) / 1024 / 1024, 1) for value in rows[:, 4]]
    print(
        "DEEPSEEK_V3_GPU_LARGE_REPORT "
        f"layers={config.num_hidden_layers} "
        f"hidden={config.hidden_size} "
        f"routed_experts={config.n_routed_experts} "
        f"experts_per_rank={config.n_routed_experts // dist.get_world_size()} "
        f"batch={batch_size} "
        f"seq={seq_len} "
        f"local_total_params_by_rank={tuple(int(value) for value in rows[:, 0])} "
        f"fsdp_dense_params_by_rank={tuple(int(value) for value in rows[:, 1])} "
        f"ignored_ep_expert_params_by_rank={tuple(int(value) for value in rows[:, 2])} "
        f"max_allocated_mib_by_rank={tuple(allocated_mib)} "
        f"max_reserved_mib_by_rank={tuple(reserved_mib)}"
    )


def _dense_shard_hints(module: nn.Module, ignored_params: set[nn.Parameter]) -> dict[str, object]:
    ignored_param_ids = {id(param) for param in ignored_params}
    param_by_fqn = dict(module.named_parameters(recurse=True, remove_duplicate=True))
    hint_source = getattr(module, "_checkpoint_wrapped_module", module)
    hints: dict[str, object] = {}
    for fqn, hint in build_shard_hints(hint_source).items():
        module_fqn = fqn
        if module_fqn not in param_by_fqn:
            wrapped_fqn = f"_checkpoint_wrapped_module.{fqn}"
            module_fqn = wrapped_fqn if wrapped_fqn in param_by_fqn else module_fqn
        if id(param_by_fqn[module_fqn]) not in ignored_param_ids:
            hints[module_fqn] = hint
    return hints


def _collect_matrix_fsdp_groups(module: nn.Module) -> list[object]:
    groups: list[object] = []
    seen: set[int] = set()
    for child in module.modules():
        group = getattr(child, "_matrix_fsdp_param_group", None)
        if group is None:
            group = getattr(child, "_matrix_fsdp_unit", None)
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        groups.append(group)
    return groups


def _param_group_fqns(param_group: object) -> tuple[str, ...]:
    registry = getattr(param_group, "param_registry", None)
    if registry is not None and hasattr(registry, "fqns"):
        return tuple(registry.fqns)
    managed_params = getattr(param_group, "managed_params", ())
    return tuple(getattr(managed_param, "fqn", "") for managed_param in managed_params)


def _lifecycle_state_name(param_group: object) -> str:
    state = getattr(param_group, "lifecycle_state", None)
    return getattr(state, "name", str(state))


def _optimizer_matrix_groups(optimizer: object) -> list[object] | None:
    groups = getattr(optimizer, "fsdp_param_groups", None)
    if groups is None:
        groups = getattr(optimizer, "units", None)
    return groups


def _loopback_interface_name() -> str:
    names = {name for _, name in socket.if_nameindex()}
    for candidate in ("lo0", "lo"):
        if candidate in names:
            return candidate
    return "lo"


if __name__ == "__main__":
    unittest.main()
