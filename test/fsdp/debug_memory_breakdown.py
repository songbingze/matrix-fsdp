"""Quick diagnostic: find where 14 GB of unaccounted memory lives in MatrixFSDP."""
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
    FSDP2CompareConfig, _torch_dtype, _run_steps, _loss,
)


def _gpu_tensors_by_size(device, top_n=20):
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
    print(f"\n  Total GPU memory in live Python tensors: {total / 1024**2:.1f} MB ({len(tensors)} unique storages)")
    for i, (nbytes, shape, dtype, ptr) in enumerate(tensors[:top_n]):
        print(f"    [{i:3d}] {nbytes / 1024**2:10.2f} MB  shape={str(shape):30s}  dtype={dtype}  ptr=0x{ptr:x}")
    if len(tensors) > top_n:
        rest = sum(t[0] for t in tensors[top_n:])
        print(f"    ... {len(tensors) - top_n} more tensors totaling {rest / 1024**2:.1f} MB")


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
        print(f"\n=== Before model creation ===")
        print(f"  CUDA allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")

    torch.manual_seed(0)
    model = _make_model(config).to(device=device, dtype=_torch_dtype(config))

    if rank == 0:
        print(f"\n=== After model.to(cuda) ===")
        print(f"  CUDA allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
        print(f"  Model param count: {sum(p.numel() for p in model.parameters())}")
        print(f"  Model param bytes: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.1f} MB")

    model, optimizer = _prepare_mode(
        "matrix_owner_muon_role_greedy_custom_collective", model, mesh, config
    )
    gc.collect()
    torch.cuda.empty_cache()

    if rank == 0:
        print(f"\n=== After matrix_fully_shard + optimizer + gc.collect ===")
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(f"  CUDA allocated: {alloc:.1f} MB")
        _gpu_tensors_by_size(device)

    x, target = _make_inputs(config, device)
    _run_steps(model, optimizer, x, target, 1, "cuda")
    gc.collect()
    torch.cuda.empty_cache()

    if rank == 0:
        print(f"\n=== After 1 warmup step + gc.collect ===")
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(f"  CUDA allocated: {alloc:.1f} MB")

    # Manual step to measure after backward WITHOUT gc.collect
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(f"\n=== Before forward (after zero_grad which calls gc.collect) ===")
        print(f"  CUDA allocated: {alloc:.1f} MB")

    loss = _loss(model(x), target)
    torch.cuda.synchronize(device)

    if rank == 0:
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\n=== After forward ===")
        print(f"  CUDA allocated: {alloc:.1f} MB, peak: {peak:.1f} MB")

    loss.backward()
    torch.cuda.synchronize(device)
    optimizer.step()
    torch.cuda.synchronize(device)

    if rank == 0:
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\n=== After backward (NO gc.collect yet) ===")
        print(f"  CUDA allocated: {alloc:.1f} MB, peak: {peak:.1f} MB")
        _gpu_tensors_by_size(device, top_n=10)

        # Trace what holds the big tensors
        big_tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device.type == "cuda" and obj.device.index == device.index:
                    if obj.untyped_storage().nbytes() > 300 * 1024 * 1024:
                        big_tensors.append(obj)
            except (RuntimeError, AttributeError):
                continue

        print(f"\n  Found {len(big_tensors)} tensors > 300 MB")
        for i, t in enumerate(big_tensors[:3]):
            print(f"\n  --- Tensor {i}: shape={t.shape}, nbytes={t.untyped_storage().nbytes() / 1024**2:.0f} MB ---")
            referrers = gc.get_referrers(t)
            for j, ref in enumerate(referrers[:5]):
                ref_type = type(ref).__name__
                if isinstance(ref, dict):
                    keys = [k for k, v in ref.items() if v is t]
                    print(f"    referrer[{j}]: dict with matching keys={keys[:3]}, total keys={len(ref)}")
                elif isinstance(ref, list):
                    idx = [k for k, v in enumerate(ref) if v is t]
                    print(f"    referrer[{j}]: list[{len(ref)}], indices={idx[:3]}")
                elif isinstance(ref, tuple):
                    idx = [k for k, v in enumerate(ref) if v is t]
                    print(f"    referrer[{j}]: tuple[{len(ref)}], indices={idx[:3]}")
                else:
                    print(f"    referrer[{j}]: {ref_type}")
                    if hasattr(ref, '__dict__'):
                        matching = [k for k, v in ref.__dict__.items() if v is t]
                        if matching:
                            print(f"      attrs: {matching[:3]}")

    del loss
    if rank == 0:
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(f"\n=== After del loss ===")
        print(f"  CUDA allocated: {alloc:.1f} MB")

    gc.collect()
    if rank == 0:
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(f"\n=== After gc.collect ===")
        print(f"  CUDA allocated: {alloc:.1f} MB")
        _gpu_tensors_by_size(device, top_n=10)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
