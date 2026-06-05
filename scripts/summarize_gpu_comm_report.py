#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([^ \n]+)")
_PHASE_TIMING_RE = re.compile(r"([A-Za-z0-9_]+_ms)=([0-9.]+)ms")
_COMM_EVENT_STATS_RE = re.compile(
    r"\b(all_gather_enqueue|reduce_scatter_enqueue|all_gather_wait|reduce_scatter_wait) "
    r"count=([0-9]+).*?sum_ms=([0-9.]+).*?max_ms=([0-9.]+)"
)


@dataclass(frozen=True)
class PhaseRow:
    log: str
    mode: str
    optimizer: str
    world: int
    seq: int | None
    zero_ms: float
    fwd_ms: float
    bwd_ms: float
    step_ms: float
    total_ms: float
    peak_mem_mb: float


@dataclass(frozen=True)
class MemorySummaryRow:
    log: str
    mode: str
    optimizer: str
    phase: str
    rank_count: int
    current_mem_max_mb: float
    current_mem_imbalance_mb: float
    peak_mem_max_mb: float
    reserved_max_mb: float
    peak_reserved_max_mb: float


@dataclass(frozen=True)
class CollectiveTimingRow:
    log: str
    event_count: int
    native_enqueue_total_ms: float
    native_enqueue_max_ms: float
    wait_total_ms: float
    wait_max_ms: float
    workspace_kinds: dict[str, int]
    materialization_kinds: dict[str, int]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MatrixFSDP GPU report logs.")
    parser.add_argument("report_dir", help="Directory produced by scripts/run_gpu_comm_report.sh.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", default="", help="Optional output file.")
    args = parser.parse_args()

    report_dir = Path(args.report_dir).expanduser().resolve()
    if not report_dir.is_dir():
        raise SystemExit(f"report_dir does not exist or is not a directory: {report_dir}")

    summary = summarize_report(report_dir)
    if args.format == "json":
        output = json.dumps(_json_safe(summary), indent=2, sort_keys=True)
    else:
        output = format_summary(summary)

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


def summarize_report(report_dir: Path) -> dict[str, Any]:
    phase_rows = _parse_phase_rows(report_dir)
    memory_rows = _parse_memory_rows(report_dir)
    comm_rows = _parse_comm_rows(report_dir)
    collective_rows = _parse_collective_timing_rows(report_dir)
    return {
        "report_dir": str(report_dir),
        "comm": comm_rows,
        "phase": phase_rows,
        "phase_vs_fsdp2": _phase_vs_fsdp2(phase_rows),
        "memory": memory_rows,
        "collective_timing": collective_rows,
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"MatrixFSDP GPU report summary: {summary['report_dir']}"]
    lines.append("")
    lines.extend(_format_comm(summary["comm"]))
    lines.append("")
    lines.extend(_format_phase(summary["phase"], summary["phase_vs_fsdp2"]))
    lines.append("")
    lines.extend(_format_memory(summary["memory"]))
    lines.append("")
    lines.extend(_format_collective_timing(summary["collective_timing"]))
    return "\n".join(lines).rstrip()


def _parse_comm_rows(report_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for log_path in sorted(report_dir.glob("comm_*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if not line.startswith("impl="):
                continue
            kv = _parse_key_values(line)
            rows.append(
                {
                    "log": log_path.name,
                    "impl": kv.get("impl", ""),
                    "chunk_fast_path": _to_int(kv.get("chunk_fast_path")),
                    "dtype": kv.get("dtype", ""),
                    "total_numel": _to_int(kv.get("total_numel")),
                    "max_ms": _to_float(kv.get("max_ms")),
                    "avg_ms": _to_float(kv.get("avg_ms")),
                    "logical_GBps": _to_float(kv.get("logical_GBps")),
                    "shard_sizes": kv.get("shard_sizes", ""),
                }
            )
    rows.sort(key=lambda row: (row["max_ms"] is None, row["max_ms"] or 0.0))
    return rows


def _parse_phase_rows(report_dir: Path) -> list[PhaseRow]:
    rows = []
    for log_path in sorted(report_dir.glob("model_*.log")):
        rows.extend(_parse_table(log_path, required_headers=("mode", "zero_ms", "fwd_ms", "bwd_ms", "step_ms")))
    return [
        PhaseRow(
            log=row["_log"],
            mode=row["mode"],
            optimizer=row.get("optim", ""),
            world=_to_int(row.get("world")) or 0,
            seq=_to_int(row.get("seq")),
            zero_ms=_to_float(row.get("zero_ms")) or 0.0,
            fwd_ms=_to_float(row.get("fwd_ms")) or 0.0,
            bwd_ms=_to_float(row.get("bwd_ms")) or 0.0,
            step_ms=_to_float(row.get("step_ms")) or 0.0,
            total_ms=_to_float(row.get("total_ms")) or 0.0,
            peak_mem_mb=_to_float(row.get("peak_mem_mb")) or 0.0,
        )
        for row in rows
    ]


def _parse_memory_rows(report_dir: Path) -> list[MemorySummaryRow]:
    raw_rows = []
    for log_path in sorted(report_dir.glob("memory_*.log")):
        raw_rows.extend(_parse_table(log_path, required_headers=("mode", "rank", "phase", "current_mem_mb")))
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["_log"], row["mode"], row.get("optim", ""), row["phase"])].append(row)

    summaries = []
    for (log, mode, optim, phase), rows in sorted(grouped.items()):
        current = [_to_float(row.get("current_mem_mb")) or 0.0 for row in rows]
        peak = [_to_float(row.get("peak_mem_mb")) or 0.0 for row in rows]
        reserved = [_to_float(row.get("reserved_mb")) or 0.0 for row in rows]
        peak_reserved = [_to_float(row.get("peak_reserved_mb")) or 0.0 for row in rows]
        summaries.append(
            MemorySummaryRow(
                log=log,
                mode=mode,
                optimizer=optim,
                phase=phase,
                rank_count=len(rows),
                current_mem_max_mb=max(current, default=0.0),
                current_mem_imbalance_mb=(max(current) - min(current)) if current else 0.0,
                peak_mem_max_mb=max(peak, default=0.0),
                reserved_max_mb=max(reserved, default=0.0),
                peak_reserved_max_mb=max(peak_reserved, default=0.0),
            )
        )
    return summaries


def _parse_collective_timing_rows(report_dir: Path) -> list[CollectiveTimingRow]:
    rows = []
    for log_path in sorted(report_dir.glob("runtime_profile_*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        enqueue_values = []
        wait_values = []
        enqueue_total_ms = 0.0
        enqueue_max_ms = 0.0
        wait_total_ms = 0.0
        wait_max_ms = 0.0
        workspace_kinds: Counter[str] = Counter()
        materialization_kinds: Counter[str] = Counter()
        event_count = 0
        for line in text.splitlines():
            event_stats_match = _COMM_EVENT_STATS_RE.search(line)
            if event_stats_match:
                event_name, count, sum_ms, max_ms = event_stats_match.groups()
                event_count += int(count)
                if event_name.endswith("_enqueue"):
                    enqueue_total_ms += float(sum_ms)
                    enqueue_max_ms = max(enqueue_max_ms, float(max_ms))
                else:
                    wait_total_ms += float(sum_ms)
                    wait_max_ms = max(wait_max_ms, float(max_ms))
            for key, value in _PHASE_TIMING_RE.findall(line):
                if key == "native_enqueue_ms":
                    enqueue_values.append(float(value))
                elif key == "wait_ms":
                    wait_values.append(float(value))
            for value in re.findall(r"workspace=([^ ,)]+)", line):
                workspace_kinds[value] += 1
            for value in re.findall(r"(?:materialize|param_materialization)=([^ ,)]+)", line):
                materialization_kinds[value] += 1
        if enqueue_values or wait_values or workspace_kinds or materialization_kinds:
            rows.append(
                CollectiveTimingRow(
                    log=log_path.name,
                    event_count=event_count or max(len(enqueue_values), len(wait_values)),
                    native_enqueue_total_ms=enqueue_total_ms or sum(enqueue_values),
                    native_enqueue_max_ms=enqueue_max_ms or max(enqueue_values, default=0.0),
                    wait_total_ms=wait_total_ms or sum(wait_values),
                    wait_max_ms=wait_max_ms or max(wait_values, default=0.0),
                    workspace_kinds=dict(workspace_kinds),
                    materialization_kinds=dict(materialization_kinds),
                )
            )
    return rows


def _phase_vs_fsdp2(rows: list[PhaseRow]) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, int | None], PhaseRow] = {}
    for row in rows:
        if row.mode == "fsdp2":
            baselines[(row.optimizer, row.seq)] = row
    comparisons = []
    for row in rows:
        if row.mode == "fsdp2":
            continue
        baseline = baselines.get((row.optimizer, row.seq))
        if baseline is None:
            continue
        comparisons.append(
            {
                "mode": row.mode,
                "optimizer": row.optimizer,
                "seq": row.seq,
                "total_delta_ms": row.total_ms - baseline.total_ms,
                "fwd_delta_ms": row.fwd_ms - baseline.fwd_ms,
                "bwd_delta_ms": row.bwd_ms - baseline.bwd_ms,
                "step_delta_ms": row.step_ms - baseline.step_ms,
                "total_delta_pct": _pct_delta(row.total_ms, baseline.total_ms),
                "fwd_delta_pct": _pct_delta(row.fwd_ms, baseline.fwd_ms),
                "bwd_delta_pct": _pct_delta(row.bwd_ms, baseline.bwd_ms),
                "step_delta_pct": _pct_delta(row.step_ms, baseline.step_ms),
                "peak_mem_delta_mb": row.peak_mem_mb - baseline.peak_mem_mb,
            }
        )
    return comparisons


def _parse_table(log_path: Path, *, required_headers: tuple[str, ...]) -> list[dict[str, str]]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for index, line in enumerate(lines):
        headers = _split_table_line(line)
        if not headers or not all(header in headers for header in required_headers):
            continue
        if index + 1 >= len(lines) or not _looks_like_separator(lines[index + 1]):
            continue

        for row_line in lines[index + 2 :]:
            parts = _split_table_line(row_line)
            if parts and len(parts) == len(headers) - 1 and "budget" in headers:
                parts.insert(headers.index("budget"), "")
            if not parts or len(parts) < len(headers):
                break
            if parts[: len(headers)] == headers:
                continue
            if _looks_like_separator(row_line):
                continue
            row = dict(zip(headers, parts[: len(headers)]))
            row["_log"] = log_path.name
            rows.append(row)
    return rows


def _split_table_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    return re.split(r"\s{2,}", stripped)


def _looks_like_separator(line: str) -> bool:
    parts = _split_table_line(line)
    return bool(parts) and all(set(part) <= {"-"} for part in parts)


def _format_comm(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Communication microbench"]
    if not rows:
        return lines + ["no comm_*.log benchmark rows found"]
    table_rows = [
        (
            row["log"],
            row["impl"],
            str(row["chunk_fast_path"] if row["chunk_fast_path"] is not None else "-"),
            _fmt(row["max_ms"]),
            _fmt(row["avg_ms"]),
            _fmt(row["logical_GBps"]),
        )
        for row in rows
    ]
    return lines + _format_table(("log", "impl", "chunk", "max_ms", "avg_ms", "GBps"), table_rows)


def _format_phase(rows: list[PhaseRow], comparisons: list[dict[str, Any]]) -> list[str]:
    lines = ["## Model phase timing"]
    if not rows:
        return lines + ["no model_*.log phase timing rows found"]
    table_rows = [
        (
            row.log,
            row.mode,
            row.optimizer,
            str(row.world),
            str(row.seq or "-"),
            f"{row.fwd_ms:.3f}",
            f"{row.bwd_ms:.3f}",
            f"{row.step_ms:.3f}",
            f"{row.total_ms:.3f}",
            f"{row.peak_mem_mb:.1f}",
        )
        for row in rows
    ]
    lines.extend(
        _format_table(
            ("log", "mode", "optim", "world", "seq", "fwd_ms", "bwd_ms", "step_ms", "total_ms", "peak_mb"),
            table_rows,
        )
    )
    if comparisons:
        lines.append("")
        lines.append("### Matrix vs FSDP2")
        compare_rows = [
            (
                row["mode"],
                row["optimizer"],
                str(row["seq"] or "-"),
                f"{row['total_delta_ms']:.3f}",
                f"{row['fwd_delta_ms']:.3f}",
                f"{row['bwd_delta_ms']:.3f}",
                f"{row['step_delta_ms']:.3f}",
                f"{row['total_delta_pct']:.1f}",
                f"{row['fwd_delta_pct']:.1f}",
                f"{row['bwd_delta_pct']:.1f}",
                f"{row['step_delta_pct']:.1f}",
                f"{row['peak_mem_delta_mb']:.1f}",
            )
            for row in comparisons
        ]
        lines.extend(
            _format_table(
                (
                    "mode",
                    "optim",
                    "seq",
                    "total_delta_ms",
                    "fwd_delta",
                    "bwd_delta",
                    "step_delta",
                    "total_pct",
                    "fwd_pct",
                    "bwd_pct",
                    "step_pct",
                    "mem_delta",
                ),
                compare_rows,
            )
        )
    return lines


def _format_memory(rows: list[MemorySummaryRow]) -> list[str]:
    lines = ["## Memory by rank"]
    if not rows:
        return lines + ["no memory_*.log trace rows found"]
    interesting_phases = {"after_forward", "after_backward", "after_step", "after_zero_grad_after_step"}
    filtered = [row for row in rows if row.phase in interesting_phases] or rows
    table_rows = [
        (
            row.log,
            row.mode,
            row.optimizer,
            row.phase,
            str(row.rank_count),
            f"{row.current_mem_max_mb:.1f}",
            f"{row.current_mem_imbalance_mb:.1f}",
            f"{row.peak_mem_max_mb:.1f}",
            f"{row.reserved_max_mb:.1f}",
            f"{row.peak_reserved_max_mb:.1f}",
        )
        for row in filtered
    ]
    return lines + _format_table(
        (
            "log",
            "mode",
            "optim",
            "phase",
            "ranks",
            "current_max",
            "current_imb",
            "peak_max",
            "reserved_max",
            "peak_reserved",
        ),
        table_rows,
    )


def _format_collective_timing(rows: list[CollectiveTimingRow]) -> list[str]:
    lines = ["## Runtime collective timing"]
    if not rows:
        return lines + ["no runtime_profile_*.log collective timing rows found"]
    table_rows = [
        (
            row.log,
            str(row.event_count),
            f"{row.native_enqueue_total_ms:.3f}",
            f"{row.native_enqueue_max_ms:.3f}",
            f"{row.wait_total_ms:.3f}",
            f"{row.wait_max_ms:.3f}",
            _format_counts(row.workspace_kinds),
            _format_counts(row.materialization_kinds),
        )
        for row in rows
    ]
    return lines + _format_table(
        ("log", "events", "enqueue_total", "enqueue_max", "wait_total", "wait_max", "workspace", "materialize"),
        table_rows,
    )


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_row(headers, widths), _format_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_row(row, widths) for row in rows)
    return lines


def _format_row(row: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ",".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _parse_key_values(line: str) -> dict[str, str]:
    return {key: value for key, value in _KEY_VALUE_RE.findall(line)}


def _to_float(value: str | None) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    if value in (None, "", "-"):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def _pct_delta(value: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0
    return (value - baseline) * 100.0 / baseline


def _json_safe(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    return value


if __name__ == "__main__":
    raise SystemExit(main())
