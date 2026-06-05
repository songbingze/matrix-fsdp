"""Full benchmark comparison: peak memory and step time with activation checkpoint."""
import gc
import os
import sys
import time
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from bench_fsdp2_compare import (
    _make_model, _prepare_mode, _make_inputs,
    FSDP2CompareConfig, _torch_dtype, _run_steps, _synchronize,
)


def run_benchmark(mode_name, mode_key, config, mesh, device, rank, warmup=2, steps=3):
    torch.manual_seed(0)
    model = _make_model(config).to(device=device, dtype=_torch_dtype(config))
    model, optimizer = _prepare_mode(mode_key, model, mesh, config)
    x, target = _make_inputs(config, device)

    _run_steps(model, optimizer, x, target, warmup, "cuda")
    gc.collect()
    torch.cuda.empty_cache()
    _synchronize("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()

    start = time.perf_counter()
    _run_steps(model, optimizer, x, target, steps, "cuda")
    _synchronize("cuda")
    elapsed = time.perf_counter() - start

    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    elapsed_tensor = torch.tensor(elapsed, device=device)
    peak_tensor = torch.tensor(peak, device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)

    if rank == 0:
        avg_step = elapsed_tensor.item() * 1000.0 / steps
        print(f"{mode_name:50s}  peak={peak_tensor.item():.1f} MB  step={avg_step:.1f} ms")

    del model, optimizer, x, target
    gc.collect()
    torch.cuda.empty_cache()


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world_size,))

    config = FSDP2CompareConfig(
        model="transformer_split_qkv", unit="block",
        layers=32, hidden=4096, intermediate=16384,
        heads=32, seq_len=4096, batch_size=1,
        optimizer="muon", dtype="bfloat16",
        device="cuda", world_size=world_size,
        warmup_steps=2, steps=3,
        activation_checkpoint=True,
        activation_checkpoint_wrapper=True,
    )

    if rank == 0:
        print(f"world_size={world_size}, layers={config.layers}, hidden={config.hidden}")
        print(f"{'Mode':50s}  {'Peak':>15s}  {'Step':>10s}")
        print("-" * 80)

    run_benchmark("FSDP2", "fsdp2", config, mesh, device, rank)
    dist.barrier()
    run_benchmark("MatrixFSDP", "matrix_owner_muon_role_greedy_custom_collective", config, mesh, device, rank)
    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
