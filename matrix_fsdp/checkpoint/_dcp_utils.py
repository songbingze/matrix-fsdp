from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

MATRIX_DCP_GLOBAL_METADATA_VERSION = 1
MATRIX_DCP_GLOBAL_METADATA_FILENAME = "matrix_metadata.pt"


def _require_dcp():
    try:
        import torch.distributed.checkpoint as dcp
    except ImportError as error:
        raise RuntimeError("PyTorch Distributed Checkpoint is not available.") from error
    return dcp


def _infer_no_dist(no_dist: bool | None) -> bool:
    if no_dist is not None:
        return no_dist
    return not (dist.is_available() and dist.is_initialized())


def _dcp_rank(process_group, no_dist: bool) -> int:
    if no_dist or not (dist.is_available() and dist.is_initialized()):
        return 0
    return dist.get_rank(process_group)


def _dcp_world_size(process_group, no_dist: bool) -> int:
    if no_dist or not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size(process_group)


def _dcp_payload_rank(
    unit_state: Mapping[str, Any],
    *,
    dcp_rank: int,
    dedup_replicates: bool,
) -> str:
    if not dedup_replicates or int(unit_state.get("replicate_world_size", 1)) <= 1:
        return f"rank_{dcp_rank}"
    return f"shard_{int(unit_state['rank'])}"


def _dcp_tensor_key(payload_rank: str | int, unit_index: int, name: str) -> str:
    if isinstance(payload_rank, int):
        payload_rank = f"rank_{payload_rank}"
    return f"{payload_rank}.unit_{unit_index}.{name}"


def _dcp_metadata_path(checkpoint_id: Path, dcp_rank: int) -> Path:
    return checkpoint_id / f"matrix_metadata_rank_{dcp_rank}.pt"


def _dcp_global_metadata_path(checkpoint_id: Path) -> Path:
    return checkpoint_id / MATRIX_DCP_GLOBAL_METADATA_FILENAME


def _save_dcp_metadata(
    checkpoint_id: Path,
    metadata: dict[str, Any],
    *,
    process_group,
    no_dist: bool,
) -> None:
    dcp_rank = _dcp_rank(process_group, no_dist)
    metadata_by_rank = _gather_dcp_metadata(metadata, process_group=process_group, no_dist=no_dist)
    if dcp_rank != 0:
        return

    for legacy_path in checkpoint_id.glob("matrix_metadata_rank_*.pt"):
        legacy_path.unlink()
    torch.save(
        {
            "metadata": {
                "format": "matrix_dcp_global_metadata",
                "version": MATRIX_DCP_GLOBAL_METADATA_VERSION,
                "num_ranks": len(metadata_by_rank),
                "ranks": tuple(sorted(metadata_by_rank)),
            },
            "ranks": metadata_by_rank,
        },
        _dcp_global_metadata_path(checkpoint_id),
    )


def _gather_dcp_metadata(
    metadata: dict[str, Any],
    *,
    process_group,
    no_dist: bool,
) -> dict[int, dict[str, Any]]:
    if no_dist or not (dist.is_available() and dist.is_initialized()):
        return {0: metadata}
    gathered: list[object] = [None for _ in range(_dcp_world_size(process_group, no_dist))]
    dist.all_gather_object(gathered, metadata, group=process_group)
    metadata_by_rank: dict[int, dict[str, Any]] = {}
    for item in gathered:
        if not isinstance(item, dict):
            raise TypeError(f"Expected gathered MatrixFSDP DCP metadata dict, got {type(item)!r}.")
        rank = int(item.get("metadata", {}).get("dcp_rank", len(metadata_by_rank)))
        metadata_by_rank[rank] = item
    return metadata_by_rank


def _load_dcp_metadata_for_rank(checkpoint_id: Path, dcp_rank: int) -> dict[str, Any]:
    metadata_by_rank = _load_global_dcp_metadata(checkpoint_id)
    if metadata_by_rank is not None:
        metadata = metadata_by_rank.get(dcp_rank)
        if metadata is None:
            raise FileNotFoundError(
                f"MatrixFSDP global DCP metadata under {checkpoint_id} does not contain rank {dcp_rank}."
            )
        return metadata

    metadata_path = _dcp_metadata_path(checkpoint_id, dcp_rank)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing MatrixFSDP DCP metadata: expected {_dcp_global_metadata_path(checkpoint_id)} "
            f"or legacy sidecar {metadata_path}."
        )
    return torch.load(metadata_path, map_location="cpu")


def _load_all_dcp_metadata(checkpoint_id: Path) -> dict[int, dict[str, Any]]:
    metadata_by_rank = _load_global_dcp_metadata(checkpoint_id)
    if metadata_by_rank is not None:
        return metadata_by_rank

    metadata_by_rank: dict[int, dict[str, Any]] = {}
    for path in sorted(checkpoint_id.glob("matrix_metadata_rank_*.pt")):
        rank_text = path.stem.rsplit("_", 1)[-1]
        rank = int(rank_text)
        metadata_by_rank[rank] = torch.load(path, map_location="cpu")
    if not metadata_by_rank:
        raise FileNotFoundError(f"No MatrixFSDP DCP metadata found under {checkpoint_id}.")
    return metadata_by_rank


def _load_global_dcp_metadata(checkpoint_id: Path) -> dict[int, dict[str, Any]] | None:
    metadata_path = _dcp_global_metadata_path(checkpoint_id)
    if not metadata_path.exists():
        return None
    payload = torch.load(metadata_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    if metadata.get("format") != "matrix_dcp_global_metadata":
        raise ValueError(f"Unsupported MatrixFSDP global DCP metadata format {metadata.get('format')!r}.")
    if metadata.get("version") != MATRIX_DCP_GLOBAL_METADATA_VERSION:
        raise ValueError(
            f"Unsupported MatrixFSDP global DCP metadata version {metadata.get('version')!r}; "
            f"expected {MATRIX_DCP_GLOBAL_METADATA_VERSION}."
        )
    ranks = payload.get("ranks")
    if not isinstance(ranks, dict):
        raise TypeError("MatrixFSDP global DCP metadata must contain a dict at key 'ranks'.")
    return {int(rank): rank_metadata for rank, rank_metadata in ranks.items()}
