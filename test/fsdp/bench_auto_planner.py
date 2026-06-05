from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from matrix_fsdp import MatrixFSDPOptimizer, matrix_fully_shard
from matrix_fsdp.auto_planner import (
    available_auto_planner_policies,
    default_auto_group_planner_candidates,
    evaluate_auto_group_planners,
)
from matrix_fsdp.managed_param import ManagedParamRegistry
from matrix_fsdp.planner_eval import PlannerEvaluation


ModelFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class PlannerBenchScenario:
    name: str
    factory: ModelFactory
    target_block_units: int | None = None
    input_shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlannerBenchRow:
    scenario: str
    policy: str
    candidate: str
    selected: bool
    cost: float
    rank_units: tuple[int, ...]
    imbalance_units: int
    num_rank_segments: int
    max_rank_segments: int
    fragmented_rank_segments: int
    fragmented_rank_units: int
    num_split_params: int
    split_param_units: int
    split_param_segments: int
    total_comm_units: int
    max_rank_comm_units: int
    estimated_padding_units: int
    estimated_collectives: int
    cost_terms: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PlannerTimingRow:
    scenario: str
    policy: str
    candidate: str
    selected: bool
    device: str
    world_size: int
    avg_step_ms: float


def default_scenarios() -> tuple[PlannerBenchScenario, ...]:
    return (
        PlannerBenchScenario("many_small_tensors", _many_small_tensors, target_block_units=128, input_shape=(8, 16)),
        PlannerBenchScenario("few_large_matrices", _few_large_matrices, target_block_units=32768, input_shape=(8, 512)),
        PlannerBenchScenario("transformer_mlp", _transformer_mlp, target_block_units=16384, input_shape=(8, 256)),
    )


def run_planner_benchmark(
    *,
    world_size: int,
    device: str = "cpu",
    policies: Sequence[str] | None = None,
    scenarios: Sequence[PlannerBenchScenario] | None = None,
    show_candidates: bool = False,
) -> tuple[PlannerBenchRow, ...]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    selected_policies = tuple(policies or available_auto_planner_policies())
    selected_scenarios = tuple(scenarios or default_scenarios())
    rows = []
    for scenario in selected_scenarios:
        module = scenario.factory().to(torch.device(device))
        params = ManagedParamRegistry.from_module(module).params
        for policy in selected_policies:
            evaluations = evaluate_auto_group_planners(
                params,
                world_size,
                policy=policy,
                target_block_units=scenario.target_block_units,
            )
            if show_candidates:
                selected_names = {evaluations[0].name} if evaluations else set()
                chosen_evaluations = evaluations
            else:
                selected_names = {evaluations[0].name} if evaluations else set()
                chosen_evaluations = evaluations[:1]
            for evaluation in chosen_evaluations:
                rows.append(_row_from_evaluation(scenario.name, policy, evaluation, evaluation.name in selected_names))
    return tuple(rows)


def format_benchmark_table(rows: Sequence[PlannerBenchRow]) -> str:
    headers = (
        "scenario",
        "policy",
        "candidate",
        "sel",
        "cost",
        "rank_units",
        "imb",
        "segments",
        "max_seg",
        "frag_seg",
        "frag_units",
        "splits",
        "split_units",
        "split_seg",
        "comm",
        "max_comm",
        "padding",
        "collectives",
    )
    table_rows = [
        (
            row.scenario,
            row.policy,
            row.candidate,
            "*" if row.selected else "",
            f"{row.cost:.2f}",
            ",".join(str(unit) for unit in row.rank_units),
            str(row.imbalance_units),
            str(row.num_rank_segments),
            str(row.max_rank_segments),
            str(row.fragmented_rank_segments),
            str(row.fragmented_rank_units),
            str(row.num_split_params),
            str(row.split_param_units),
            str(row.split_param_segments),
            str(row.total_comm_units),
            str(row.max_rank_comm_units),
            str(row.estimated_padding_units),
            str(row.estimated_collectives),
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(table_row, widths) for table_row in table_rows)
    return "\n".join(lines)


def time_train_steps(
    *,
    world_size: int,
    device: str,
    policies: Sequence[str],
    scenarios: Sequence[PlannerBenchScenario],
    warmup_steps: int,
    steps: int,
    time_candidates: bool = False,
) -> tuple[PlannerTimingRow, ...]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")
    if world_size > 1 and not dist.is_initialized():
        raise RuntimeError("Multi-rank timing requires torchrun or an initialized process group.")

    rank = dist.get_rank() if dist.is_initialized() else 0
    if device == "cuda":
        torch.cuda.set_device(rank)
        torch_device = torch.device("cuda", rank)
    else:
        torch_device = torch.device("cpu")
    mesh = DeviceMesh(device, torch.arange(world_size)) if world_size > 1 else None

    rows = []
    for scenario in scenarios:
        if not scenario.input_shape:
            continue
        for policy in policies:
            probe_model = scenario.factory().to(torch_device)
            probe_params = ManagedParamRegistry.from_module(probe_model).params
            evaluations = evaluate_auto_group_planners(
                probe_params,
                world_size,
                policy=policy,
                target_block_units=scenario.target_block_units,
            )
            candidate_bundle = default_auto_group_planner_candidates(
                probe_params,
                world_size,
                target_block_units=scenario.target_block_units,
            )
            timed_evaluations = evaluations if time_candidates else evaluations[:1]
            selected_name = evaluations[0].name if evaluations else ""
            for evaluation in timed_evaluations:
                rows.append(
                    _time_candidate(
                        scenario=scenario,
                        policy=policy,
                        candidate=evaluation.name,
                        selected=evaluation.name == selected_name,
                        planner=candidate_bundle.planners[evaluation.name],
                        mesh=mesh,
                        torch_device=torch_device,
                        device=device,
                        world_size=world_size,
                        warmup_steps=warmup_steps,
                        steps=steps,
                    )
                )
    return tuple(rows)


def format_timing_table(rows: Sequence[PlannerTimingRow]) -> str:
    headers = ("scenario", "policy", "candidate", "sel", "device", "world_size", "avg_step_ms")
    table_rows = [
        (
            row.scenario,
            row.policy,
            row.candidate,
            "*" if row.selected else "",
            row.device,
            str(row.world_size),
            f"{row.avg_step_ms:.3f}",
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for table_row in table_rows:
        for index, value in enumerate(table_row):
            widths[index] = max(widths[index], len(value))
    lines = [_format_table_row(headers, widths), _format_table_row(tuple("-" * width for width in widths), widths)]
    lines.extend(_format_table_row(table_row, widths) for table_row in table_rows)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect MatrixFSDP auto planner choices.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--policy", action="append", choices=available_auto_planner_policies())
    parser.add_argument("--show-candidates", action="store_true")
    parser.add_argument("--time-step", action="store_true")
    parser.add_argument("--time-candidates", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    initialized_here = _init_process_group_from_env(args.device)
    runtime_world_size = dist.get_world_size() if dist.is_initialized() else args.world_size
    policies = tuple(args.policy or available_auto_planner_policies())
    scenarios = default_scenarios()
    rows = run_planner_benchmark(
        world_size=runtime_world_size,
        device=args.device,
        policies=policies,
        scenarios=scenarios,
        show_candidates=args.show_candidates,
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    try:
        if rank == 0:
            print(format_benchmark_table(rows))
        if args.time_step:
            timing_rows = time_train_steps(
                world_size=runtime_world_size,
                device=args.device,
                policies=policies,
                scenarios=scenarios,
                warmup_steps=args.warmup_steps,
                steps=args.steps,
                time_candidates=args.time_candidates,
            )
            if rank == 0:
                print()
                print(format_timing_table(timing_rows))
        return 0
    finally:
        if initialized_here:
            dist.destroy_process_group()


def _row_from_evaluation(
    scenario: str,
    policy: str,
    evaluation: PlannerEvaluation,
    selected: bool,
) -> PlannerBenchRow:
    report = evaluation.report
    return PlannerBenchRow(
        scenario=scenario,
        policy=policy,
        candidate=evaluation.name,
        selected=selected,
        cost=evaluation.cost,
        rank_units=report.rank_units,
        imbalance_units=report.imbalance_units,
        num_rank_segments=report.num_rank_segments,
        max_rank_segments=report.max_rank_segments,
        fragmented_rank_segments=report.fragmented_rank_segments,
        fragmented_rank_units=report.fragmented_rank_units,
        num_split_params=report.num_split_params,
        split_param_units=report.split_param_units,
        split_param_segments=report.split_param_segments,
        total_comm_units=report.total_comm_units,
        max_rank_comm_units=report.max_rank_comm_units,
        estimated_padding_units=report.estimated_padding_units,
        estimated_collectives=report.estimated_collectives,
        cost_terms=tuple(sorted(evaluation.cost_breakdown.terms.items())),
    )


def _format_table_row(values: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))


def _time_candidate(
    *,
    scenario: PlannerBenchScenario,
    policy: str,
    candidate: str,
    selected: bool,
    planner,
    mesh: DeviceMesh | None,
    torch_device: torch.device,
    device: str,
    world_size: int,
    warmup_steps: int,
    steps: int,
) -> PlannerTimingRow:
    torch.manual_seed(0)
    model = scenario.factory().to(torch_device)
    sharded_model = matrix_fully_shard(model, mesh, group_planner=planner)
    optimizer = MatrixFSDPOptimizer(torch.optim.SGD(sharded_model.parameters(), lr=0.01), sharded_model)
    x = torch.randn(*scenario.input_shape, device=torch_device)
    with torch.no_grad():
        target = torch.randn_like(sharded_model(x))
    _synchronize(device)
    _run_timed_steps(sharded_model, optimizer, x, target, warmup_steps, device)
    if dist.is_initialized():
        dist.barrier()
    start = time.perf_counter()
    _run_timed_steps(sharded_model, optimizer, x, target, steps, device)
    _synchronize(device)
    if dist.is_initialized():
        dist.barrier()
    elapsed = time.perf_counter() - start
    return PlannerTimingRow(
        scenario=scenario.name,
        policy=policy,
        candidate=candidate,
        selected=selected,
        device=device,
        world_size=world_size,
        avg_step_ms=elapsed * 1000.0 / steps,
    )


def _init_process_group_from_env(device: str) -> bool:
    if dist.is_initialized():
        return False
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    backend = "nccl" if device == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    return True


def _run_timed_steps(
    model: nn.Module,
    optimizer: MatrixFSDPOptimizer,
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    device: str,
) -> None:
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (model(x) - target).pow(2).mean()
        loss.backward()
        optimizer.step()
    _synchronize(device)


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _many_small_tensors() -> nn.Module:
    layers = []
    for _ in range(8):
        layers.append(nn.Linear(16, 16))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _few_large_matrices() -> nn.Module:
    return nn.Sequential(
        nn.Linear(512, 512, bias=False),
        nn.ReLU(),
        nn.Linear(512, 256, bias=False),
    )


def _transformer_mlp() -> nn.Module:
    hidden = 256
    intermediate = 1024
    return nn.Sequential(
        nn.Linear(hidden, intermediate, bias=False),
        nn.GELU(),
        nn.Linear(intermediate, hidden, bias=False),
    )


if __name__ == "__main__":
    raise SystemExit(main())
