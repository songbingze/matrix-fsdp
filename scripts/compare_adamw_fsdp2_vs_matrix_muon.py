#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test.fsdp.bench_fsdp2_compare import (  # noqa: E402
    FSDP2CompareConfig,
    format_phase_timing_table,
    run_phase_timing,
)


DEFAULT_MATRIX_MUON_MODE = "matrix_owner_muon_role_greedy_custom_collective"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current best MatrixFSDP matrix-owner Muon path against "
            "PyTorch FSDP2 AdamW on the same transformer shape."
        )
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=16384)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--activation-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply activation checkpointing to each transformer block.",
    )
    parser.add_argument(
        "--activation-checkpoint-wrapper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.distributed checkpoint_wrapper for activation checkpointing.",
    )
    parser.add_argument(
        "--matrix-mode",
        default=DEFAULT_MATRIX_MUON_MODE,
        choices=(
            "matrix_owner_muon_role_greedy",
            "matrix_owner_muon_role_greedy_pre_backward",
            "matrix_owner_muon_role_greedy_custom_collective",
            "matrix_owner_muon_role_greedy_custom_collective_pre_backward",
        ),
    )
    parser.add_argument(
        "--custom-allgatherv-impl",
        default="native_sendrecv",
        choices=("native_sendrecv", "native_group_broadcast", "uneven_all_gather", "broadcast", "all_reduce"),
        help="Custom MatrixFSDP allgatherv implementation for the Matrix Muon mode.",
    )
    parser.add_argument(
        "--custom-reduce-scatterv-impl",
        default="uneven_reduce_scatter",
        choices=("uneven_reduce_scatter", "reduce"),
        help="Custom MatrixFSDP reduce-scatterv implementation for the Matrix Muon mode.",
    )
    parser.add_argument("--output-json", default="", help="Optional path to write combined phase timing rows.")
    parser.add_argument("--json-config", action="store_true", help="Print the two resolved benchmark configs.")
    args = parser.parse_args(argv)

    if args.activation_checkpoint_wrapper and not args.activation_checkpoint:
        parser.error("--activation-checkpoint-wrapper requires --activation-checkpoint.")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("This comparison requires a PyTorch build with torch.optim.Muon.")

    os.environ["MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL"] = args.custom_allgatherv_impl
    os.environ["MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL"] = args.custom_reduce_scatterv_impl

    common = {
        "world_size": args.world_size,
        "device": args.device,
        "unit": "block",
        "model": "transformer_split_qkv",
        "layers": args.layers,
        "hidden": args.hidden,
        "intermediate": args.intermediate,
        "seq_len": args.seq_len,
        "heads": args.heads,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "warmup_steps": args.warmup_steps,
        "profile_steps": 0,
        "steps": args.steps,
        "activation_checkpoint": args.activation_checkpoint,
        "activation_checkpoint_wrapper": args.activation_checkpoint_wrapper,
    }
    fsdp2_adamw_config = FSDP2CompareConfig(
        **common,
        modes=("fsdp2",),
        optimizer="adamw",
    )
    matrix_muon_config = FSDP2CompareConfig(
        **common,
        modes=(args.matrix_mode,),
        optimizer="muon",
    )

    if args.json_config:
        print("== FSDP2 AdamW config ==")
        print(json.dumps(asdict(fsdp2_adamw_config), indent=2, sort_keys=True))
        print()
        print("== MatrixFSDP Muon config ==")
        print(json.dumps(asdict(matrix_muon_config), indent=2, sort_keys=True))
        print()

    rows = []
    print("== FSDP2 AdamW phase timing ==")
    fsdp2_rows = run_phase_timing(fsdp2_adamw_config)
    rows.extend(fsdp2_rows)
    print(format_phase_timing_table(fsdp2_rows))

    print()
    print("== MatrixFSDP Muon phase timing ==")
    matrix_rows = run_phase_timing(matrix_muon_config)
    rows.extend(matrix_rows)
    print(format_phase_timing_table(matrix_rows))

    print()
    print("== Combined comparison ==")
    print(format_phase_timing_table(rows))
    _print_speedup_summary(rows)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            json.dump([asdict(row) for row in rows], output_file, indent=2)
    return 0


def _print_speedup_summary(rows) -> None:
    if len(rows) != 2:
        return
    reference, candidate = rows
    if candidate.avg_total_ms <= 0:
        return
    speedup = reference.avg_total_ms / candidate.avg_total_ms
    memory_delta_mb = candidate.peak_memory_mb - reference.peak_memory_mb
    print()
    print(
        "MatrixFSDP Muon vs FSDP2 AdamW: "
        f"{speedup:.3f}x total-step speed ratio, "
        f"{memory_delta_mb:+.1f} MB peak-memory delta."
    )


if __name__ == "__main__":
    raise SystemExit(main())
