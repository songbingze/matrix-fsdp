"""Trace reference cycle: find which closures hold the leaked 384 MB tensors."""
import gc
import os
import sys
import types
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from bench_fsdp2_compare import (
    _make_model, _prepare_mode, _make_inputs,
    FSDP2CompareConfig, _torch_dtype, _run_steps, _loss, _synchronize,
)


def find_owning_functions(obj, max_depth=6):
    """Walk up from obj through referrer chain to find function objects."""
    visited = set()
    queue = [(obj, 0, "root")]
    functions = []

    while queue:
        current, depth, path = queue.pop(0)
        cid = id(current)
        if cid in visited or depth > max_depth:
            continue
        visited.add(cid)

        if isinstance(current, types.FunctionType):
            functions.append((current, depth, path))
            continue

        referrers = gc.get_referrers(current)
        for ref in referrers:
            if isinstance(ref, type) or type(ref).__name__ == 'frame':
                continue
            rtype = type(ref).__name__
            if rtype == 'cell':
                queue.append((ref, depth + 1, f"{path} -> cell"))
            elif isinstance(ref, types.FunctionType):
                functions.append((ref, depth + 1, f"{path} -> func"))
            elif isinstance(ref, tuple):
                queue.append((ref, depth + 1, f"{path} -> tuple"))
            elif isinstance(ref, list):
                queue.append((ref, depth + 1, f"{path} -> list"))
            elif isinstance(ref, dict):
                keys = [k for k, v in ref.items() if v is current]
                # Find dict owner
                dict_owners = [o for o in gc.get_referrers(ref)
                             if not isinstance(o, type) and type(o).__name__ != 'frame']
                for owner in dict_owners[:2]:
                    otype = type(owner).__name__
                    queue.append((owner, depth + 1, f"{path} -> dict[{keys[:2]}] -> {otype}"))

    return functions


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

    loss = _loss(model(x), target)
    loss.backward()
    _synchronize("cuda")

    if rank == 0:
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        print(f"After backward: {alloc:.1f} MB")

        # Find first large tensor
        big = None
        for obj in gc.get_objects():
            try:
                if (torch.is_tensor(obj) and obj.device.type == "cuda"
                    and obj.device.index == device.index
                    and obj.untyped_storage().nbytes() > 300 * 1024 * 1024):
                    big = obj
                    break
            except:
                continue

        if big is not None:
            nbytes = big.untyped_storage().nbytes()
            print(f"\nTracing tensor: {nbytes/1024**2:.0f} MB, shape={big.shape}")

            funcs = find_owning_functions(big)
            print(f"\nFound {len(funcs)} functions in referrer chain:")
            seen_qualnames = set()
            for func, depth, path in funcs[:20]:
                qualname = getattr(func, '__qualname__', getattr(func, '__name__', '?'))
                module = getattr(func, '__module__', '?')
                key = f"{module}.{qualname}"
                if key in seen_qualnames:
                    continue
                seen_qualnames.add(key)
                print(f"  depth={depth}: {module}.{qualname}")
                # Show closure vars
                if func.__closure__:
                    for i, cell in enumerate(func.__closure__):
                        try:
                            val = cell.cell_contents
                            vtype = type(val).__name__
                            if torch.is_tensor(val):
                                vtype = f"Tensor(shape={val.shape}, device={val.device})"
                            elif isinstance(val, (list, tuple)):
                                vtype = f"{type(val).__name__}[{len(val)}]"
                            print(f"    closure[{i}]: {vtype}")
                        except ValueError:
                            print(f"    closure[{i}]: <empty>")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
