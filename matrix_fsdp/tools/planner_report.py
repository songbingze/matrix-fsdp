from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from matrix_fsdp import make_muon_shard_aware_group_planner
from matrix_fsdp.auto_planner import (
    available_auto_planner_policies,
    build_auto_planner_report,
    format_auto_planner_report,
)
from matrix_fsdp.managed_param import ManagedParamRegistry
from matrix_fsdp.planner_eval import call_group_planner
from matrix_fsdp.planner_eval import PlannerResourceEstimate
from matrix_fsdp.shard_hint import build_shard_hints


@dataclass(frozen=True)
class PlannerReportConfig:
    model: str = "transformer_split_qkv"
    layers: int = 1
    hidden: int = 1024
    intermediate: int = 4096
    heads: int = 8
    unit: str = "model"
    world_size: int = 8
    policy: str = "muon_shard_aware"
    rotation_strategy: str = "greedy_balance"
    owner_assignment: str = "rotate"
    target_block_units: int | None = None
    auto_shard_hints: bool = True
    show_candidates: bool = True
    output_format: str = "text"


@dataclass(frozen=True)
class UnitPlannerReport:
    unit_name: str
    unit_type: str
    candidate: str
    cost: float
    rank_units: tuple[int, ...]
    block_kinds: tuple[tuple[str, int], ...]
    constraint_counts: tuple[tuple[str, int], ...]
    runtime_mode: str
    runtime_compatible: bool
    runtime_requires_flat_reorder: bool
    runtime_reason: str | None
    warnings: tuple[str, ...]
    params_by_rank: tuple[tuple[str, ...], ...]
    rank_memory_bytes: tuple[int, ...]
    rank_comm_bytes: tuple[int, ...]
    rank_muon_param_bytes: tuple[int, ...]
    rank_adamw_param_bytes: tuple[int, ...]
    rank_optimizer_bytes: tuple[int, ...]
    planner_summary: dict[str, object]
    layout_contract: dict[str, object]
    report: dict[str, object]
    resources: dict[str, object] | None


@dataclass(frozen=True)
class PlannerReportResult:
    group_rows: tuple
    unit_rows: tuple[UnitPlannerReport, ...]
    rank_total_units: tuple[int, ...]
    rank_total_memory_bytes: tuple[int, ...]
    rank_total_comm_bytes: tuple[int, ...]
    rank_total_muon_param_bytes: tuple[int, ...]
    rank_total_adamw_param_bytes: tuple[int, ...]
    rank_total_optimizer_bytes: tuple[int, ...]
    rank_block_kinds: tuple[tuple[tuple[str, int], ...], ...]


class MLPBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.act = nn.GELU()
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class MLPStack(nn.Module):
    def __init__(self, layers: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MLPBlock(hidden, intermediate) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, heads: int) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}.")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = MLPBlock(hidden, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        qkv = self.qkv(self.norm1(x))
        qkv = qkv.view(batch, seq_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        x = x + self.proj(attn)
        return x + self.mlp(self.norm2(x))


class SplitGELUMLPBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        if intermediate % 2 != 0:
            raise ValueError(f"intermediate={intermediate} must be even for split GELU MLP.")
        half = intermediate // 2
        self.up0 = nn.Linear(hidden, half, bias=False)
        self.up1 = nn.Linear(hidden, half, bias=False)
        self.down = nn.Linear(half, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up0(x)) * self.up1(x))


class SplitQKVTransformerBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int, heads: int) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}.")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = SplitGELUMLPBlock(hidden, intermediate)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        normed = self.norm1(x)
        query = self.q(normed).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        key = self.k(normed).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        value = self.v(normed).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        x = x + self.proj(attn)
        return x + self.mlp(self.norm2(x))


class TransformerStack(nn.Module):
    def __init__(self, block_cls: type[nn.Module], layers: int, hidden: int, intermediate: int, heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([block_cls(hidden, intermediate, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def run_planner_report(config: PlannerReportConfig) -> PlannerReportResult:
    module = _make_model(config)
    if config.unit != "model":
        return _run_unit_planner_report(module, config)
    shard_hints = build_shard_hints(module) if config.auto_shard_hints else None
    params = ManagedParamRegistry.from_module(module, shard_hints=shard_hints).params
    rows = build_auto_planner_report(
        params,
        config.world_size,
        policy=config.policy,
        target_block_units=config.target_block_units,
        show_candidates=config.show_candidates,
    )
    return PlannerReportResult(
        group_rows=rows,
        unit_rows=(),
        rank_total_units=rows[0].rank_units if rows else (),
        rank_total_memory_bytes=rows[0].rank_memory_bytes if rows else (),
        rank_total_comm_bytes=rows[0].rank_comm_bytes if rows else (),
        rank_total_muon_param_bytes=rows[0].rank_muon_param_bytes if rows else (),
        rank_total_adamw_param_bytes=rows[0].rank_adamw_param_bytes if rows else (),
        rank_total_optimizer_bytes=rows[0].rank_optimizer_bytes if rows else (),
        rank_block_kinds=_rank_block_kinds_from_group_rows(rows[:1], config.world_size),
    )


def format_planner_report(config: PlannerReportConfig, result: PlannerReportResult) -> str:
    header = (
        "MatrixFSDP auto planner report\n"
        f"model={config.model} layers={config.layers} hidden={config.hidden} "
        f"intermediate={config.intermediate} heads={config.heads} "
        f"unit={config.unit} world_size={config.world_size} "
        f"policy={config.policy} rotation={config.rotation_strategy} "
        f"owner_assignment={config.owner_assignment}"
    )
    sections = [header]
    if result.group_rows:
        sections.append(format_auto_planner_report(result.group_rows))
    if result.unit_rows:
        sections.append(format_unit_planner_report(result.unit_rows))
    sections.append(format_rank_load_summary(result))
    return "\n".join(sections)


def format_unit_planner_report(rows: Sequence[UnitPlannerReport]) -> str:
    headers = (
        "unit",
        "type",
        "candidate",
        "cost",
        "rank_units",
        "rank_mem",
        "rank_comm",
        "rank_muon",
        "rank_adamw",
        "blocks",
        "constraints",
        "runtime",
        "warnings",
        "params_by_rank",
    )
    table_rows = [
        (
            row.unit_name,
            row.unit_type,
            row.candidate,
            f"{row.cost:.3f}",
            _format_int_tuple(row.rank_units),
            _format_int_tuple(row.rank_memory_bytes),
            _format_int_tuple(row.rank_comm_bytes),
            _format_int_tuple(row.rank_muon_param_bytes),
            _format_int_tuple(row.rank_adamw_param_bytes),
            _format_name_counts(row.block_kinds),
            _format_name_counts(row.constraint_counts),
            _format_runtime(row),
            _format_strings(row.warnings),
            _format_params_by_rank(row.params_by_rank),
        )
        for row in rows
    ]
    return _format_table(headers, table_rows)


def format_rank_load_summary(result: PlannerReportResult) -> str:
    headers = (
        "rank",
        "total_units",
        "memory_bytes",
        "comm_bytes",
        "muon_param_bytes",
        "adamw_param_bytes",
        "block_kinds",
    )
    table_rows = [
        (
            str(rank),
            str(total_units),
            str(result.rank_total_memory_bytes[rank]),
            str(result.rank_total_comm_bytes[rank]),
            str(result.rank_total_muon_param_bytes[rank]),
            str(result.rank_total_adamw_param_bytes[rank]),
            _format_name_counts(result.rank_block_kinds[rank]),
        )
        for rank, total_units in enumerate(result.rank_total_units)
    ]
    return "rank_load_summary\n" + _format_table(headers, table_rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect MatrixFSDP auto planner candidate layouts.")
    parser.add_argument(
        "--model",
        choices=("mlp", "transformer", "transformer_split_qkv"),
        default="transformer_split_qkv",
    )
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--unit", choices=("model", "block", "linear"), default="model")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--policy", choices=available_auto_planner_policies(), default="muon_shard_aware")
    parser.add_argument("--rotation-strategy", choices=("greedy_balance", "round_robin"), default="greedy_balance")
    parser.add_argument("--owner-assignment", choices=("rotate", "role_greedy"), default="rotate")
    parser.add_argument("--target-block-units", type=int, default=None)
    parser.add_argument("--no-auto-shard-hints", action="store_true")
    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text", dest="output_format")
    args = parser.parse_args(argv)

    config = PlannerReportConfig(
        model=args.model,
        layers=args.layers,
        hidden=args.hidden,
        intermediate=args.intermediate,
        heads=args.heads,
        unit=args.unit,
        world_size=args.world_size,
        policy=args.policy,
        rotation_strategy=args.rotation_strategy,
        owner_assignment=args.owner_assignment,
        target_block_units=args.target_block_units,
        auto_shard_hints=not args.no_auto_shard_hints,
        show_candidates=not args.selected_only,
        output_format=args.output_format,
    )
    result = run_planner_report(config)
    if config.output_format == "json":
        print(
            json.dumps(
                {
                    "config": asdict(config),
                    "group_rows": [dataclasses.asdict(row) for row in result.group_rows],
                    "unit_rows": [dataclasses.asdict(row) for row in result.unit_rows],
                    "rank_total_units": result.rank_total_units,
                    "rank_total_memory_bytes": result.rank_total_memory_bytes,
                    "rank_total_comm_bytes": result.rank_total_comm_bytes,
                    "rank_total_muon_param_bytes": result.rank_total_muon_param_bytes,
                    "rank_total_adamw_param_bytes": result.rank_total_adamw_param_bytes,
                    "rank_total_optimizer_bytes": result.rank_total_optimizer_bytes,
                    "rank_block_kinds": result.rank_block_kinds,
                },
                indent=2,
            )
        )
    else:
        print(format_planner_report(config, result))
    return 0


def _make_model(config: PlannerReportConfig) -> nn.Module:
    if config.model == "mlp":
        return MLPStack(config.layers, config.hidden, config.intermediate)
    if config.model == "transformer":
        return TransformerStack(TransformerBlock, config.layers, config.hidden, config.intermediate, config.heads)
    if config.model == "transformer_split_qkv":
        return TransformerStack(
            SplitQKVTransformerBlock,
            config.layers,
            config.hidden,
            config.intermediate,
            config.heads,
        )
    raise ValueError(f"Unknown model={config.model!r}.")


def _run_unit_planner_report(module: nn.Module, config: PlannerReportConfig) -> PlannerReportResult:
    unit_rows: list[UnitPlannerReport] = []
    rank_total_units = [0 for _ in range(config.world_size)]
    rank_total_memory_bytes = [0 for _ in range(config.world_size)]
    rank_total_comm_bytes = [0 for _ in range(config.world_size)]
    rank_total_muon_param_bytes = [0 for _ in range(config.world_size)]
    rank_total_adamw_param_bytes = [0 for _ in range(config.world_size)]
    rank_total_optimizer_bytes = [0 for _ in range(config.world_size)]
    rank_block_kind_counts = [Counter() for _ in range(config.world_size)]
    group_planner = _make_group_planner(config)

    for unit_name, unit_module in _iter_units(module, config):
        shard_hints = build_shard_hints(unit_module) if config.auto_shard_hints else None
        params = ManagedParamRegistry.from_module(unit_module, shard_hints=shard_hints).params
        evaluation = call_group_planner(group_planner, params, config.world_size)
        layout = evaluation.layout
        candidate = evaluation.name
        block_kinds = tuple(sorted(evaluation.report.blocks_by_kind.items()))
        metadata = evaluation.as_metadata()
        resources = evaluation.resource_estimate or PlannerResourceEstimate.empty(config.world_size)
        planner_summary = evaluation.summary()
        for rank, units in enumerate(layout.shard_sizes):
            rank_total_units[rank] += units
            rank_total_memory_bytes[rank] += resources.rank_memory_bytes[rank]
            rank_total_comm_bytes[rank] += resources.rank_comm_bytes[rank]
            rank_total_muon_param_bytes[rank] += resources.rank_muon_param_bytes[rank]
            rank_total_adamw_param_bytes[rank] += resources.rank_adamw_param_bytes[rank]
            rank_total_optimizer_bytes[rank] += resources.rank_optimizer_bytes[rank]
        params_by_rank = tuple(layout.params_for_rank(rank) for rank in range(layout.world_size))
        _accumulate_rank_block_kinds(rank_block_kind_counts, params_by_rank, block_kinds)
        unit_rows.append(
            UnitPlannerReport(
                unit_name=unit_name,
                unit_type=type(unit_module).__name__,
                candidate=candidate,
                cost=evaluation.cost,
                rank_units=layout.shard_sizes,
                block_kinds=block_kinds,
                constraint_counts=tuple(metadata["constraint_counts"]),
                runtime_mode=evaluation.runtime_mode,
                runtime_compatible=evaluation.runtime_compatible,
                runtime_requires_flat_reorder=evaluation.runtime_requires_flat_reorder,
                runtime_reason=evaluation.runtime_reason,
                warnings=evaluation.warnings,
                params_by_rank=params_by_rank,
                rank_memory_bytes=resources.rank_memory_bytes,
                rank_comm_bytes=resources.rank_comm_bytes,
                rank_muon_param_bytes=resources.rank_muon_param_bytes,
                rank_adamw_param_bytes=resources.rank_adamw_param_bytes,
                rank_optimizer_bytes=resources.rank_optimizer_bytes,
                planner_summary=planner_summary,
                layout_contract=planner_summary["layout"],
                report=planner_summary["report"],
                resources=planner_summary["resources"],
            )
        )

    return PlannerReportResult(
        group_rows=(),
        unit_rows=tuple(unit_rows),
        rank_total_units=tuple(rank_total_units),
        rank_total_memory_bytes=tuple(rank_total_memory_bytes),
        rank_total_comm_bytes=tuple(rank_total_comm_bytes),
        rank_total_muon_param_bytes=tuple(rank_total_muon_param_bytes),
        rank_total_adamw_param_bytes=tuple(rank_total_adamw_param_bytes),
        rank_total_optimizer_bytes=tuple(rank_total_optimizer_bytes),
        rank_block_kinds=tuple(tuple(sorted(counter.items())) for counter in rank_block_kind_counts),
    )


def _make_group_planner(config: PlannerReportConfig):
    if config.policy == "muon_shard_aware":
        return make_muon_shard_aware_group_planner(
            rotation_strategy=config.rotation_strategy,
            owner_assignment=config.owner_assignment,
        )

    def group_planner(params, world_size):
        from matrix_fsdp.auto_planner import auto_group_plan

        return auto_group_plan(
            params,
            world_size,
            policy=config.policy,
            target_block_units=config.target_block_units,
        )

    return group_planner


def _iter_units(module: nn.Module, config: PlannerReportConfig):
    for name, submodule in module.named_modules():
        if not name:
            continue
        if _is_unit(submodule, config):
            yield name, submodule


def _is_unit(module: nn.Module, config: PlannerReportConfig) -> bool:
    if config.unit == "linear":
        return isinstance(module, nn.Linear)
    if config.unit == "block" and config.model == "mlp":
        return isinstance(module, MLPBlock)
    if config.unit == "block" and config.model == "transformer":
        return isinstance(module, TransformerBlock)
    if config.unit == "block" and config.model == "transformer_split_qkv":
        return isinstance(module, SplitQKVTransformerBlock)
    return False


def _rank_block_kinds_from_group_rows(rows, world_size: int):
    if not rows:
        return tuple(() for _ in range(world_size))
    counts = [Counter() for _ in range(world_size)]
    row = rows[0]
    _accumulate_rank_block_kinds(counts, row.params_by_rank, row.block_kinds)
    return tuple(tuple(sorted(counter.items())) for counter in counts)


def _accumulate_rank_block_kinds(
    rank_block_kind_counts: list[Counter],
    params_by_rank: Sequence[Sequence[str]],
    block_kinds: Sequence[tuple[str, int]],
) -> None:
    if not block_kinds:
        return
    dominant_kind = block_kinds[0][0] if len(block_kinds) == 1 else None
    if dominant_kind is not None:
        for rank, params in enumerate(params_by_rank):
            rank_block_kind_counts[rank][dominant_kind] += len(params)
        return
    for rank, params in enumerate(params_by_rank):
        for param in params:
            if param.endswith(".weight") and "norm" not in param:
                rank_block_kind_counts[rank]["muon_matrix_owner"] += 1
            else:
                rank_block_kind_counts[rank]["adamw_tail_param"] += 1


def _format_table(headers: Sequence[str], table_rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(table_row, widths) for table_row in table_rows)
    return "\n".join(lines)


def _format_int_tuple(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _format_name_counts(items: Sequence[tuple[str, int]]) -> str:
    if not items:
        return "-"
    return ",".join(f"{name}={count}" for name, count in items)


def _format_strings(values: Sequence[str]) -> str:
    if not values:
        return "-"
    return "|".join(values)


def _format_runtime(row: UnitPlannerReport) -> str:
    status = row.runtime_mode
    if not row.runtime_compatible:
        status = f"{status}:unsupported"
    elif row.runtime_requires_flat_reorder:
        status = f"{status}:reorder"
    if row.runtime_reason and not row.runtime_compatible:
        return f"{status}({row.runtime_reason})"
    return status


def _format_params_by_rank(params_by_rank: Sequence[Sequence[str]]) -> str:
    return ";".join(f"r{rank}:{'|'.join(params) if params else '-'}" for rank, params in enumerate(params_by_rank))


def _format_table_row(values: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))


if __name__ == "__main__":
    raise SystemExit(main())
