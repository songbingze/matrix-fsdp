"""Trace what holds the 12 GB of gc-freeable memory after backward."""
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


def trace_cycle(device, rank):
    """Find the large GPU tensors and trace what keeps them alive."""
    if rank != 0:
        return

    # Find all large GPU tensors
    big_tensors = []
    seen = set()
    for obj in gc.get_objects():
        try:
            if not torch.is_tensor(obj) or obj.device.type != "cuda" or obj.device.index != device.index:
                continue
            storage = obj.untyped_storage()
            nbytes = storage.nbytes()
            if nbytes < 100 * 1024 * 1024:  # > 100 MB
                continue
            ptr = storage.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            big_tensors.append((nbytes, obj))
        except (RuntimeError, AttributeError):
            continue

    big_tensors.sort(key=lambda x: -x[0])
    print(f"\n  Found {len(big_tensors)} tensors > 100 MB")

    # Trace the first 3
    for i, (nbytes, t) in enumerate(big_tensors[:3]):
        print(f"\n  === Tensor {i}: {nbytes/1024**2:.0f} MB, shape={t.shape}, dtype={t.dtype} ===")
        _trace_referrers(t, depth=0, max_depth=5, visited=set())


def _trace_referrers(obj, depth, max_depth, visited):
    if depth >= max_depth:
        return
    obj_id = id(obj)
    if obj_id in visited:
        print(f"{'  '*(depth+2)}[already visited id={obj_id}]")
        return
    visited.add(obj_id)

    referrers = gc.get_referrers(obj)
    # Filter out frames and gc internals
    referrers = [r for r in referrers if not isinstance(r, type) and type(r).__name__ != 'frame']

    for j, ref in enumerate(referrers[:5]):
        prefix = '  ' * (depth + 2)
        ref_type = type(ref).__name__

        if isinstance(ref, dict):
            keys = [k for k, v in ref.items() if v is obj]
            # Check if this dict is an object's __dict__
            dict_owners = gc.get_referrers(ref)
            owner_types = [type(o).__name__ for o in dict_owners
                          if not isinstance(o, (type, dict, list, tuple)) and type(o).__name__ != 'frame']
            owner_info = f" (owned by: {owner_types[:3]})" if owner_types else ""
            print(f"{prefix}[{j}] dict keys={keys[:5]}, total_keys={len(ref)}{owner_info}")
            if owner_types and depth < 3:
                for o in dict_owners:
                    if not isinstance(o, (type, dict, list, tuple)) and type(o).__name__ != 'frame':
                        oname = type(o).__name__
                        print(f"{prefix}  -> {oname}")
                        if hasattr(o, '__class__'):
                            _trace_referrers(o, depth+1, max_depth, visited)
                        break
        elif isinstance(ref, list):
            idx = [k for k, v in enumerate(ref) if v is obj]
            print(f"{prefix}[{j}] list[{len(ref)}] at indices={idx[:5]}")
        elif isinstance(ref, tuple):
            idx = [k for k, v in enumerate(ref) if v is obj]
            print(f"{prefix}[{j}] tuple[{len(ref)}] at indices={idx[:5]}")
            # Check who holds this tuple
            tuple_owners = gc.get_referrers(ref)
            for to in tuple_owners[:3]:
                tot = type(to).__name__
                if tot not in ('frame', 'type'):
                    print(f"{prefix}  held by {tot}")
                    if isinstance(to, dict):
                        tk = [k for k, v in to.items() if v is ref]
                        print(f"{prefix}    dict keys: {tk[:3]}")
        elif hasattr(ref, '__class__'):
            attrs = []
            if hasattr(ref, '__dict__'):
                attrs = [k for k, v in ref.__dict__.items() if v is obj]
            print(f"{prefix}[{j}] {ref_type} attrs={attrs[:5]}")
            if depth < 2:
                _trace_referrers(ref, depth+1, max_depth, visited)
        else:
            print(f"{prefix}[{j}] {ref_type}")


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

    torch.manual_seed(0)
    model = _make_model(config).to(device=device, dtype=_torch_dtype(config))
    model, optimizer = _prepare_mode(
        "matrix_owner_muon_role_greedy_custom_collective", model, mesh, config
    )
    x, target = _make_inputs(config, device)

    _run_steps(model, optimizer, x, target, 1, "cuda")
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    _synchronize("cuda")

    # One measured step
    loss = _loss(model(x), target)
    loss.backward()
    _synchronize("cuda")

    if rank == 0:
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(f"After backward: {alloc:.1f} MB")

    # Trace BEFORE gc.collect
    trace_cycle(device, rank)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
