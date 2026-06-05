"""
Instrument MatrixFSDP forward pass block-by-block to find where peak memory occurs.
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
    FSDP2CompareConfig, _torch_dtype, _loss, _synchronize, _run_steps,
)


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

    x, target = _make_inputs(config, device)

    torch.manual_seed(0)
    model = _make_model(config).to(device=device, dtype=_torch_dtype(config))
    model, optimizer = _prepare_mode(
        "matrix_owner_muon_role_greedy_custom_collective", model, mesh, config
    )

    # Warmup
    _run_steps(model, optimizer, x, target, 1, "cuda")
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    _synchronize("cuda")

    if rank == 0:
        baseline = torch.cuda.memory_allocated(device) / 1024**2
        print(f"\nBaseline after warmup+gc: {baseline:.1f} MB", flush=True)

    # Add hooks to track memory at each block
    memory_log = []

    from matrix_fsdp.runtime.fsdp_unit import MatrixFSDPParamGroup

    original_unshard = MatrixFSDPParamGroup.unshard
    original_reshard = MatrixFSDPParamGroup.reshard_after_forward
    original_finalize = MatrixFSDPParamGroup.finalize_backward

    def patched_unshard(self, *args, **kwargs):
        if rank == 0:
            alloc = torch.cuda.memory_allocated(device) / 1024**2
            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            uid = getattr(self.runtime_metadata, 'runtime_param_group_id', '?')
            memory_log.append(('unshard_start', uid, alloc, peak))
        result = original_unshard(self, *args, **kwargs)
        if rank == 0:
            alloc = torch.cuda.memory_allocated(device) / 1024**2
            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            uid = getattr(self.runtime_metadata, 'runtime_param_group_id', '?')
            memory_log.append(('unshard_done', uid, alloc, peak))
        return result

    def patched_reshard(self, *args, **kwargs):
        result = original_reshard(self, *args, **kwargs)
        if rank == 0:
            alloc = torch.cuda.memory_allocated(device) / 1024**2
            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            uid = getattr(self.runtime_metadata, 'runtime_param_group_id', '?')
            memory_log.append(('reshard', uid, alloc, peak))
        return result

    def patched_finalize(self, *args, **kwargs):
        if rank == 0:
            alloc_before = torch.cuda.memory_allocated(device) / 1024**2
        result = original_finalize(self, *args, **kwargs)
        if rank == 0:
            alloc = torch.cuda.memory_allocated(device) / 1024**2
            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            uid = getattr(self.runtime_metadata, 'runtime_param_group_id', '?')
            memory_log.append(('finalize', uid, alloc, peak))
        return result

    MatrixFSDPParamGroup.unshard = patched_unshard
    MatrixFSDPParamGroup.reshard_after_forward = patched_reshard
    MatrixFSDPParamGroup.finalize_backward = patched_finalize

    # Run one step with instrumentation
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    if rank == 0:
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(f"After zero_grad: {alloc:.1f} MB\n", flush=True)

    output = model(x)
    _synchronize("cuda")

    if rank == 0:
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\nAfter forward: alloc={alloc:.1f} MB, peak={peak:.1f} MB", flush=True)
        print(f"\n=== Forward/Reshard Log ===", flush=True)
        for event, uid, alloc, peak in memory_log:
            if 'unshard' in event or 'reshard' in event:
                print(f"  {event:20s}  unit={uid}  alloc={alloc:.1f} MB  peak={peak:.1f} MB", flush=True)

    memory_log.clear()
    loss = _loss(output, target)
    loss.backward()
    _synchronize("cuda")

    if rank == 0:
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\nAfter backward: alloc={alloc:.1f} MB, peak={peak:.1f} MB", flush=True)
        print(f"\n=== Backward Log (first 10 + last 5) ===", flush=True)
        for i, (event, uid, alloc, peak) in enumerate(memory_log):
            if i < 10 or i >= len(memory_log) - 5:
                print(f"  {event:20s}  unit={uid}  alloc={alloc:.1f} MB  peak={peak:.1f} MB", flush=True)
            elif i == 10:
                print(f"  ... ({len(memory_log) - 15} entries omitted) ...", flush=True)

    # Restore original methods
    MatrixFSDPParamGroup.unshard = original_unshard
    MatrixFSDPParamGroup.reshard_after_forward = original_reshard
    MatrixFSDPParamGroup.finalize_backward = original_finalize

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
