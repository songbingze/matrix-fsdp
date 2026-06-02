from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import torch
import torch.distributed as dist

from matrix_fsdp.core.layout import LayoutSegment
from matrix_fsdp.kernels.custom_collectives import (
    default_custom_allgatherv_impl,
    experimental_custom_allgatherv_impls,
    custom_all_gatherv_rank_segments_1d_into_async,
    is_experimental_custom_allgatherv_impl,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark MatrixFSDP custom allgatherv kernels.",
        epilog=(
            f"Default runtime gather is {default_custom_allgatherv_impl()!r}. "
            f"Experimental implementations {experimental_custom_allgatherv_impls()} are opt-in only and may fall "
            "back to the default path when the optional NCCL feature is unavailable."
        ),
    )
    parser.add_argument("--backend", choices=("nccl",), default="nccl")
    parser.add_argument(
        "--impl",
        choices=(
            "native_sendrecv",
            "auto",
            "rma_put_signal",
            "gin_device",
            "native_group_broadcast",
            "uneven_all_gather",
            "broadcast",
            "all_reduce",
            "padded_all_gather",
        ),
        default=default_custom_allgatherv_impl(),
        help="Allgatherv implementation to benchmark. rma_put_signal/gin_device are experimental.",
    )
    parser.add_argument("--chunk-fast-path", choices=("0", "1"), default="1")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--shard-sizes", required=True, help="Comma-separated per-rank shard sizes in elements.")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--check", action="store_true", help="Validate gathered segment values after warmup.")
    args = parser.parse_args(argv)

    os.environ["MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL"] = args.impl
    os.environ["MATRIX_FSDP_NATIVE_SENDRECV_CHUNK_FAST_PATH"] = args.chunk_fast_path
    if is_experimental_custom_allgatherv_impl(args.impl):
        os.environ.setdefault("MATRIX_FSDP_ENABLE_GIN_KERNELS", "1")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(args.backend, device_id=torch.device("cuda", local_rank))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)
        dtype = _dtype(args.dtype)
        shard_sizes = _parse_shard_sizes(args.shard_sizes, world_size)
        rank_segments = _rank_segments_from_shard_sizes(shard_sizes)
        total_numel = sum(shard_sizes)
        local_tensor = torch.full((shard_sizes[rank],), float(rank + 1), device=device, dtype=dtype)
        output_tensor = torch.empty((total_numel,), device=device, dtype=dtype)
        padded_workspace = _make_padded_workspace(args.impl, local_tensor, shard_sizes, device)

        for _ in range(args.warmup_steps):
            _run_allgatherv_once(args.impl, local_tensor, output_tensor, rank_segments, shard_sizes, rank, padded_workspace)
        torch.cuda.synchronize(device)
        if args.check:
            _validate_output(output_tensor, rank_segments)
            dist.barrier(device_ids=[local_rank])

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.steps):
            _run_allgatherv_once(args.impl, local_tensor, output_tensor, rank_segments, shard_sizes, rank, padded_workspace)
        end.record()
        end.synchronize()

        local_ms = torch.tensor([start.elapsed_time(end) / args.steps], device=device)
        gathered = [torch.empty_like(local_ms) for _ in range(world_size)]
        dist.all_gather(gathered, local_ms)
        if rank == 0:
            per_rank_ms = [item.item() for item in gathered]
            _print_result(args, shard_sizes, per_rank_ms, dtype)
    finally:
        dist.destroy_process_group()
    return 0


def _parse_shard_sizes(value: str, world_size: int) -> tuple[int, ...]:
    shard_sizes = tuple(int(part) for part in value.split(",") if part)
    if len(shard_sizes) != world_size:
        raise ValueError(f"Expected {world_size} shard sizes, got {len(shard_sizes)}.")
    if any(size < 0 for size in shard_sizes):
        raise ValueError(f"Shard sizes must be non-negative, got {shard_sizes}.")
    return shard_sizes


def _rank_segments_from_shard_sizes(shard_sizes: tuple[int, ...]) -> tuple[tuple[LayoutSegment, ...], ...]:
    cursor = 0
    rank_segments = []
    for shard_size in shard_sizes:
        if shard_size == 0:
            rank_segments.append(())
            continue
        rank_segments.append((LayoutSegment(cursor, cursor + shard_size, 0),))
        cursor += shard_size
    return tuple(rank_segments)


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise AssertionError(f"Unhandled dtype {name!r}.")


def _validate_output(
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
) -> None:
    for src_rank, segments in enumerate(rank_segments):
        expected = float(src_rank + 1)
        for segment in segments:
            actual = output_tensor[segment.global_start : segment.global_end]
            if not torch.all(actual == expected):
                raise RuntimeError(f"Gather validation failed for rank {src_rank} segment {segment}.")


def _make_padded_workspace(
    impl: str,
    local_tensor: torch.Tensor,
    shard_sizes: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if impl != "padded_all_gather":
        return None
    max_shard_size = max(shard_sizes, default=0)
    padded_local = torch.empty((max_shard_size,), device=device, dtype=local_tensor.dtype)
    padded_output = torch.empty((max_shard_size * len(shard_sizes),), device=device, dtype=local_tensor.dtype)
    return padded_local, padded_output


def _run_allgatherv_once(
    impl: str,
    local_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    rank_segments: tuple[tuple[LayoutSegment, ...], ...],
    shard_sizes: tuple[int, ...],
    rank: int,
    padded_workspace: tuple[torch.Tensor, torch.Tensor] | None,
) -> None:
    if impl == "padded_all_gather":
        if padded_workspace is None:
            raise RuntimeError("padded_all_gather requires padded workspace.")
        padded_local, padded_output = padded_workspace
        local_numel = shard_sizes[rank]
        if local_numel:
            padded_local[:local_numel].copy_(local_tensor)
        dist.all_gather_into_tensor(padded_output, padded_local)
        max_shard_size = padded_local.numel()
        output_cursor = 0
        for src_rank, shard_size in enumerate(shard_sizes):
            if shard_size:
                output_tensor[output_cursor : output_cursor + shard_size].copy_(
                    padded_output[src_rank * max_shard_size : src_rank * max_shard_size + shard_size]
                )
            output_cursor += shard_size
        return

    handle = custom_all_gatherv_rank_segments_1d_into_async(
        local_tensor,
        output_tensor,
        rank_segments,
        rank,
        group=dist.group.WORLD,
    )
    handle.wait()


def _print_result(
    args: argparse.Namespace,
    shard_sizes: tuple[int, ...],
    per_rank_ms: list[float],
    dtype: torch.dtype,
) -> None:
    element_size = torch.empty((), dtype=dtype).element_size()
    total_bytes = sum(shard_sizes) * element_size
    max_ms = max(per_rank_ms)
    gbps = total_bytes / max_ms / 1e6 if max_ms > 0 else 0.0
    print(
        "impl={impl} chunk_fast_path={chunk} dtype={dtype} total_numel={total} "
        "total_bytes={bytes} max_ms={max_ms:.3f} min_ms={min_ms:.3f} avg_ms={avg_ms:.3f} "
        "logical_GBps={gbps:.2f} shard_sizes={shards} per_rank_ms={per_rank}".format(
            impl=args.impl,
            chunk=args.chunk_fast_path,
            dtype=args.dtype,
            total=sum(shard_sizes),
            bytes=total_bytes,
            max_ms=max_ms,
            min_ms=min(per_rank_ms),
            avg_ms=sum(per_rank_ms) / len(per_rank_ms),
            gbps=gbps,
            shards=",".join(str(size) for size in shard_sizes),
            per_rank=",".join(f"{value:.3f}" for value in per_rank_ms),
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
