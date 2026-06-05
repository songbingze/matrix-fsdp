"""
Measure memory delta between FSDP2 and MatrixFSDP at each step phase.
Distinguishes gc-freeable (reference cycles) from strongly-referenced memory.
"""
import gc
import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from bench_fsdp2_compare import (
    _make_model, _prepare_mode, _make_inputs,
    FSDP2CompareConfig, _torch_dtype, _run_steps, _loss, _synchronize,
)


def _mem(device):
    return torch.cuda.memory_allocated(device) / 1024**2


def _gpu_tensor_summary(device, top_n=10):
    """Walk all Python objects and find GPU tensors."""
    seen = set()
    tensors = []
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj) or obj.device.type != "cuda" or obj.device.index != device.index:
                continue
            storage = obj.untyped_storage()
            size_bytes = storage.nbytes()
            if size_bytes == 0:
                continue
            ptr = storage.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            tensors.append((size_bytes, obj.shape, obj.dtype, ptr))
        except (RuntimeError, AttributeError):
            continue
    tensors.sort(reverse=True)
    total = sum(t[0] for t in tensors)
    print(f"    Live GPU tensors: {total / 1024**2:.1f} MB ({len(tensors)} unique storages)")
    for i, (nbytes, shape, dtype, ptr) in enumerate(tensors[:top_n]):
        print(f"      [{i:3d}] {nbytes / 1024**2:8.1f} MB  shape={str(shape):30s}  dtype={dtype}")
    if len(tensors) > top_n:
        rest = sum(t[0] for t in tensors[top_n:])
        print(f"      ... {len(tensors) - top_n} more totaling {rest / 1024**2:.1f} MB")


def run_mode(mode_name, mode_key, config, mesh, device, rank):
    torch.manual_seed(0)
    model = _make_model(config).to(device=device, dtype=_torch_dtype(config))
    model, optimizer = _prepare_mode(mode_key, model, mesh, config)
    x, target = _make_inputs(config, device)

    # Warmup
    _run_steps(model, optimizer, x, target, 1, "cuda")
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    _synchronize("cuda")

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  {mode_name} ({mode_key})")
        print(f"{'='*60}")

    # Measure phases
    torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        print(f"\n  [start] alloc={_mem(device):.1f} MB")

    # Forward
    optimizer.zero_grad(set_to_none=True)
    _synchronize("cuda")
    if rank == 0:
        print(f"  [after zero_grad] alloc={_mem(device):.1f} MB")

    loss = _loss(model(x), target)
    _synchronize("cuda")
    if rank == 0:
        print(f"  [after forward] alloc={_mem(device):.1f} MB, peak={torch.cuda.max_memory_allocated(device)/1024**2:.1f} MB")

    loss.backward()
    _synchronize("cuda")
    if rank == 0:
        alloc_before_gc = _mem(device)
        print(f"  [after backward] alloc={alloc_before_gc:.1f} MB")

    # Try gc.collect to see how much is cyclic garbage
    gc.collect()
    _synchronize("cuda")
    if rank == 0:
        alloc_after_gc = _mem(device)
        freed_by_gc = alloc_before_gc - alloc_after_gc
        print(f"  [after backward + gc.collect] alloc={alloc_after_gc:.1f} MB (gc freed {freed_by_gc:.1f} MB)")

    optimizer.step()
    _synchronize("cuda")
    if rank == 0:
        alloc_after_step = _mem(device)
        print(f"  [after step] alloc={alloc_after_step:.1f} MB")

    gc.collect()
    if rank == 0:
        alloc_after_step_gc = _mem(device)
        freed = alloc_after_step - alloc_after_step_gc
        print(f"  [after step + gc.collect] alloc={alloc_after_step_gc:.1f} MB (gc freed {freed:.1f} MB)")
        _gpu_tensor_summary(device)

    optimizer.zero_grad(set_to_none=True)
    _synchronize("cuda")
    if rank == 0:
        alloc_after_zg = _mem(device)
        freed = alloc_after_step_gc - alloc_after_zg
        print(f"  [after zero_grad] alloc={alloc_after_zg:.1f} MB (zero_grad freed {freed:.1f} MB)")

    gc.collect()
    if rank == 0:
        alloc_final = _mem(device)
        freed = alloc_after_zg - alloc_final
        print(f"  [after zero_grad + gc.collect] alloc={alloc_final:.1f} MB (gc freed {freed:.1f} MB)")

    del model, optimizer, loss, x, target
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
        warmup_steps=1, steps=1,
        activation_checkpoint=True,
        activation_checkpoint_wrapper=True,
    )

    if rank == 0:
        print(f"world_size={world_size}")

    # Run FSDP2 first
    run_mode("FSDP2", "fsdp2", config, mesh, device, rank)
    dist.barrier()

    # Run MatrixFSDP
    run_mode("MatrixFSDP", "matrix_owner_muon_role_greedy_custom_collective", config, mesh, device, rank)
    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
