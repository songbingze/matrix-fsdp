from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from torch import nn

from matrix_fsdp.core.managed_param import ParamRuntimeKind, ParamShardHint


ShardHintRuleFn = Callable[[str, nn.Module, str, nn.Parameter], ParamShardHint | None]


@dataclass(frozen=True)
class ShardHintRule:
    name: str
    apply: ShardHintRuleFn


def build_shard_hints(
    module: nn.Module,
    *,
    rules: Sequence[ShardHintRule] | None = None,
    overrides: Mapping[str, ParamShardHint] | None = None,
) -> dict[str, ParamShardHint]:
    selected_rules = tuple(rules or default_shard_hint_rules())
    hints: dict[str, ParamShardHint] = {}
    module_by_fqn = dict(module.named_modules())
    for fqn, param in module.named_parameters(recurse=True, remove_duplicate=True):
        module_fqn, local_name = _split_param_fqn(fqn)
        owning_module = module_by_fqn[module_fqn]
        for rule in selected_rules:
            hint = rule.apply(fqn, owning_module, local_name, param)
            if hint is not None:
                hints[fqn] = hint
                break
    if overrides:
        param_fqns = {fqn for fqn, _ in module.named_parameters(recurse=True, remove_duplicate=True)}
        unknown = set(overrides) - param_fqns
        if unknown:
            unknown_names = ", ".join(sorted(unknown))
            raise ValueError(f"Shard hint overrides refer to unknown parameters: {unknown_names}.")
        hints.update(overrides)
    return hints


def default_shard_hint_rules() -> tuple[ShardHintRule, ...]:
    return (
        moe_expert_owner_rule(),
        router_rule(),
        muon_linear_weight_rule(),
        no_split_embedding_rule(),
        no_split_norm_rule(),
        no_split_bias_rule(),
        no_split_1d_rule(),
    )


def moe_expert_owner_rule(
    *,
    expert_module_names: Sequence[str] = ("experts",),
) -> ShardHintRule:
    expert_names = tuple(expert_module_names)

    def apply(fqn: str, owning_module: nn.Module, local_name: str, param: nn.Parameter) -> ParamShardHint | None:
        expert_metadata = _parse_expert_fqn(fqn, expert_module_names=expert_names)
        if expert_metadata is None:
            return None
        expert_id, expert_group_id = expert_metadata
        if param.ndim == 2:
            return ParamShardHint(
                optimizer_type="muon",
                split_granularity="matrix_owner",
                runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
                parallel_role="routed_expert",
                expert_id=expert_id,
                expert_group_id=expert_group_id,
            )
        return ParamShardHint(
            optimizer_type="adamw",
            split_granularity="parameter",
            runtime_kind=ParamRuntimeKind.EXPERT_OWNER,
            parallel_role="routed_expert",
            expert_id=expert_id,
            expert_group_id=expert_group_id,
        )

    return ShardHintRule("moe_expert_owner", apply)


def router_rule(*, module_names: Sequence[str] = ("gate", "router")) -> ShardHintRule:
    router_names = tuple(module_names)

    def apply(fqn: str, owning_module: nn.Module, local_name: str, param: nn.Parameter) -> ParamShardHint | None:
        module_parts = _split_param_fqn(fqn)[0].split(".")
        if any(part in router_names for part in module_parts):
            return ParamShardHint(
                optimizer_type="adamw",
                split_granularity="parameter",
                runtime_kind=ParamRuntimeKind.FSDP_GATHER,
                parallel_role="router",
            )
        return None

    return ShardHintRule("router", apply)


def muon_linear_weight_rule() -> ShardHintRule:
    def apply(fqn: str, owning_module: nn.Module, local_name: str, param: nn.Parameter) -> ParamShardHint | None:
        if isinstance(owning_module, nn.Linear) and local_name == "weight" and param.ndim == 2:
            return ParamShardHint(optimizer_type="muon", split_granularity="matrix_owner")
        return None

    return ShardHintRule("muon_linear_weight", apply)


def no_split_embedding_rule() -> ShardHintRule:
    def apply(fqn: str, owning_module: nn.Module, local_name: str, param: nn.Parameter) -> ParamShardHint | None:
        if isinstance(owning_module, nn.Embedding) and local_name == "weight":
            return ParamShardHint(optimizer_type="adamw", split_granularity="parameter")
        return None

    return ShardHintRule("no_split_embedding", apply)


def no_split_norm_rule() -> ShardHintRule:
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
    )

    def apply(fqn: str, owning_module: nn.Module, local_name: str, param: nn.Parameter) -> ParamShardHint | None:
        if isinstance(owning_module, norm_types):
            return ParamShardHint(optimizer_type="adamw", split_granularity="parameter")
        return None

    return ShardHintRule("no_split_norm", apply)


def no_split_bias_rule() -> ShardHintRule:
    def apply(fqn: str, owning_module: nn.Module, local_name: str, param: nn.Parameter) -> ParamShardHint | None:
        if local_name == "bias":
            return ParamShardHint(optimizer_type="adamw", split_granularity="parameter")
        return None

    return ShardHintRule("no_split_bias", apply)


def no_split_1d_rule() -> ShardHintRule:
    def apply(fqn: str, owning_module: nn.Module, local_name: str, param: nn.Parameter) -> ParamShardHint | None:
        if param.ndim <= 1:
            return ParamShardHint(optimizer_type="adamw", split_granularity="parameter")
        return None

    return ShardHintRule("no_split_1d", apply)


def _split_param_fqn(fqn: str) -> tuple[str, str]:
    if "." not in fqn:
        return "", fqn
    module_fqn, local_name = fqn.rsplit(".", 1)
    return module_fqn, local_name


def _parse_expert_fqn(
    fqn: str,
    *,
    expert_module_names: Sequence[str],
) -> tuple[int, str] | None:
    parts = fqn.split(".")
    for index, part in enumerate(parts[:-2]):
        if part not in expert_module_names:
            continue
        expert_id_part = parts[index + 1]
        if not expert_id_part.isdigit():
            continue
        expert_group_id = ".".join(parts[: index + 2])
        return int(expert_id_part), expert_group_id
    return None
