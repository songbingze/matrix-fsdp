#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "test" / "fsdp" / "bench_fsdp2_compare.py"


@dataclass(frozen=True)
class IsolatedBenchmarkRun:
    mode: str
    trial: int
    returncode: int
    avg_step_ms: float | None = None
    total_ms: float | None = None
    peak_mem_mb: float | None = None
    output: str = ""
    command: tuple[str, ...] = ()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run MatrixFSDP/FSDP2 benchmark modes in isolated subprocesses. "
            "This avoids cross-mode CUDA allocator, kernel-cache, and warmup order effects."
        )
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--mode", action="append", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model", choices=("mlp", "transformer", "transformer_split_qkv"), default="transformer_split_qkv")
    parser.add_argument("--unit", choices=("linear", "block"), default="block")
    parser.add_argument("--block-group-size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--optimizer", choices=("sgd", "adamw", "muon"), default="muon")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--activation-checkpoint-wrapper", action="store_true")
    parser.add_argument("--phase-timing", action="store_true")
    parser.add_argument("--runtime-summary", action="store_true")
    parser.add_argument("--matrix-max-cached-elastic-workspaces-per-key", type=int, default=0)
    parser.add_argument("--custom-allgatherv-impl", default="")
    parser.add_argument("--custom-reduce-scatterv-impl", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.trials <= 0:
        raise SystemExit("--trials must be positive.")
    if args.activation_checkpoint_wrapper and not args.activation_checkpoint:
        raise SystemExit("--activation-checkpoint-wrapper requires --activation-checkpoint.")

    runs = []
    for trial in range(args.trials):
        for mode in args.mode:
            command = _benchmark_command(args, mode)
            if args.dry_run:
                print(" ".join(command))
                continue
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            if args.custom_allgatherv_impl:
                env["MATRIX_FSDP_CUSTOM_ALLGATHERV_IMPL"] = args.custom_allgatherv_impl
            if args.custom_reduce_scatterv_impl:
                env["MATRIX_FSDP_CUSTOM_REDUCE_SCATTERV_IMPL"] = args.custom_reduce_scatterv_impl
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            run = _parse_benchmark_output(mode, trial, completed.returncode, completed.stdout, tuple(command))
            runs.append(run)
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
            print(_format_run(run))

    if args.dry_run:
        return 0
    print()
    print(format_isolated_summary(runs))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            json.dump([asdict(run) for run in runs], output_file, indent=2)
            output_file.write("\n")
    return 1 if any(run.returncode != 0 for run in runs) else 0


def _benchmark_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        args.python_bin,
        str(BENCH_SCRIPT),
        "--world-size",
        str(args.world_size),
        "--device",
        args.device,
        "--model",
        args.model,
        "--unit",
        args.unit,
        "--block-group-size",
        str(args.block_group_size),
        "--layers",
        str(args.layers),
        "--hidden",
        str(args.hidden),
        "--intermediate",
        str(args.intermediate),
        "--seq-len",
        str(args.seq_len),
        "--heads",
        str(args.heads),
        "--batch-size",
        str(args.batch_size),
        "--optimizer",
        args.optimizer,
        "--dtype",
        args.dtype,
        "--warmup-steps",
        str(args.warmup_steps),
        "--steps",
        str(args.steps),
        "--matrix-max-cached-elastic-workspaces-per-key",
        str(args.matrix_max_cached_elastic_workspaces_per_key),
        "--mode",
        mode,
    ]
    if args.activation_checkpoint:
        command.append("--activation-checkpoint")
    if args.activation_checkpoint_wrapper:
        command.append("--activation-checkpoint-wrapper")
    if args.phase_timing:
        command.append("--phase-timing")
    if args.runtime_summary:
        command.append("--runtime-summary")
    return command


def _parse_benchmark_output(
    mode: str,
    trial: int,
    returncode: int,
    output: str,
    command: tuple[str, ...],
) -> IsolatedBenchmarkRun:
    row = _last_table_row_for_mode(output, mode)
    avg_step_ms = _float_or_none(row.get("avg_step_ms"))
    total_ms = _float_or_none(row.get("total_ms"))
    peak_mem_mb = _float_or_none(row.get("peak_mem_mb"))
    return IsolatedBenchmarkRun(
        mode=mode,
        trial=trial,
        returncode=returncode,
        avg_step_ms=avg_step_ms,
        total_ms=total_ms,
        peak_mem_mb=peak_mem_mb,
        output=output,
        command=command,
    )


def _last_table_row_for_mode(output: str, mode: str) -> dict[str, str]:
    headers: list[str] | None = None
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = _split_table_line(stripped)
        if not fields:
            continue
        if fields[0] == "mode":
            headers = fields
            continue
        if headers is None or fields[0] != mode or len(fields) != len(headers):
            continue
        parsed = dict(zip(headers, fields))
    return parsed


def _split_table_line(line: str) -> list[str]:
    if set(line) <= {"-", " "}:
        return []
    return [field for field in re.split(r"\s{2,}", line.strip()) if field]


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_run(run: IsolatedBenchmarkRun) -> str:
    metric = run.avg_step_ms if run.avg_step_ms is not None else run.total_ms
    metric_name = "avg_step_ms" if run.avg_step_ms is not None else "total_ms"
    metric_text = "n/a" if metric is None else f"{metric:.3f}"
    mem_text = "n/a" if run.peak_mem_mb is None else f"{run.peak_mem_mb:.1f}"
    return f"[isolated] mode={run.mode} trial={run.trial} rc={run.returncode} {metric_name}={metric_text} peak_mem_mb={mem_text}"


def format_isolated_summary(runs: Sequence[IsolatedBenchmarkRun]) -> str:
    headers = ("mode", "runs", "metric", "p50_ms", "mean_ms", "min_ms", "max_ms", "p50_mem_mb", "failures")
    rows = []
    by_mode: dict[str, list[IsolatedBenchmarkRun]] = defaultdict(list)
    for run in runs:
        by_mode[run.mode].append(run)
    for mode, mode_runs in by_mode.items():
        values = [
            run.avg_step_ms if run.avg_step_ms is not None else run.total_ms
            for run in mode_runs
            if run.returncode == 0 and (run.avg_step_ms is not None or run.total_ms is not None)
        ]
        mem_values = [run.peak_mem_mb for run in mode_runs if run.returncode == 0 and run.peak_mem_mb is not None]
        metric_name = "avg_step_ms" if any(run.avg_step_ms is not None for run in mode_runs) else "total_ms"
        failures = sum(1 for run in mode_runs if run.returncode != 0)
        rows.append(
            (
                mode,
                str(len(mode_runs)),
                metric_name,
                _format_stat(statistics.median(values) if values else None),
                _format_stat(statistics.fmean(values) if values else None),
                _format_stat(min(values) if values else None),
                _format_stat(max(values) if values else None),
                _format_stat(statistics.median(mem_values) if mem_values else None),
                str(failures),
            )
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(row, widths) for row in rows)
    return "\n".join(lines)


def _format_stat(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_table_row(values: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths))


if __name__ == "__main__":
    raise SystemExit(main())
