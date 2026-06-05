#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from test.fsdp.bench_fsdp2_compare import (  # noqa: E402
    DEFAULT_FULLY_SHARD_API_COMPARE_MODES,
    FSDP2CompareConfig,
    format_compare_table,
    format_correctness_result,
    format_phase_timing_table,
    run_compare_benchmark,
    run_correctness_check,
    run_phase_timing,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PyTorch FSDP2 fully_shard() against MatrixFSDP fully_shard() "
            "using only default public APIs and plain torch optimizers."
        )
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model", choices=("mlp", "transformer", "transformer_split_qkv"), default="mlp")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="adamw")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--phase-timing", action="store_true", help="Print zero/fwd/bwd/step timing breakdown.")
    parser.add_argument("--correctness", action="store_true", help="Also compare losses, outputs, and full grads.")
    parser.add_argument("--json-config", action="store_true", help="Print the resolved benchmark config first.")
    args = parser.parse_args(argv)

    config = FSDP2CompareConfig(
        world_size=args.world_size,
        device=args.device,
        modes=DEFAULT_FULLY_SHARD_API_COMPARE_MODES,
        model=args.model,
        unit="linear",
        layers=args.layers,
        hidden=args.hidden,
        intermediate=args.intermediate,
        seq_len=args.seq_len,
        heads=args.heads,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        dtype=args.dtype,
        warmup_steps=args.warmup_steps,
        profile_steps=0,
        steps=args.steps,
    )
    if args.json_config:
        print(asdict(config))

    print("== fully_shard API benchmark ==")
    print(format_compare_table(run_compare_benchmark(config)))

    if args.phase_timing:
        print()
        print("== fully_shard API phase timing ==")
        print(format_phase_timing_table(run_phase_timing(config)))

    if args.correctness:
        print()
        print("== fully_shard API correctness ==")
        print(
            format_correctness_result(
                run_correctness_check(
                    config,
                    reference_mode="fsdp2_api",
                    candidate_mode="matrix_api",
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
