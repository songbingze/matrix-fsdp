from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sysconfig

import torch

_DEVICE_API_MIN_VERSION = (2, 28, 0)
_GIN_MIN_VERSION = (2, 28, 7)
_HEADER_GLOBS = ("nccl*.h",)
_DEVICE_API_TOKENS = (
    "ncclDevComm",
    "ncclDevCommCreate",
    "ncclCommWindowRegister",
)
_GIN_TOKENS = (
    "ncclGin",
    "NCCL_GIN",
    "ncclLsa",
)


@dataclass(frozen=True)
class GinDeviceApiProbe:
    torch_cuda_available: bool
    torch_nccl_version: tuple[int, ...] | None
    include_dirs: tuple[str, ...]
    headers: tuple[str, ...]
    header_has_device_api: bool
    header_has_gin_tokens: bool
    device_api_version_ready: bool
    gin_version_ready: bool
    device_api_ready: bool
    gin_ready: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_gin_device_api(*, extra_include_dirs: tuple[str, ...] = ()) -> GinDeviceApiProbe:
    torch_nccl_version = _torch_nccl_version()
    include_dirs = _candidate_include_dirs(extra_include_dirs)
    headers = _find_nccl_headers(include_dirs)
    header_text = _read_header_text(headers)
    header_has_device_api = any(token in header_text for token in _DEVICE_API_TOKENS)
    header_has_gin_tokens = any(token in header_text for token in _GIN_TOKENS)
    device_api_version_ready = _version_at_least(torch_nccl_version, _DEVICE_API_MIN_VERSION)
    gin_version_ready = _version_at_least(torch_nccl_version, _GIN_MIN_VERSION)
    torch_cuda_available = torch.cuda.is_available()
    device_api_ready = bool(torch_cuda_available and device_api_version_ready and header_has_device_api)
    gin_ready = bool(device_api_ready and gin_version_ready and header_has_gin_tokens)
    return GinDeviceApiProbe(
        torch_cuda_available=torch_cuda_available,
        torch_nccl_version=torch_nccl_version,
        include_dirs=tuple(str(path) for path in include_dirs),
        headers=tuple(str(path) for path in headers),
        header_has_device_api=header_has_device_api,
        header_has_gin_tokens=header_has_gin_tokens,
        device_api_version_ready=device_api_version_ready,
        gin_version_ready=gin_version_ready,
        device_api_ready=device_api_ready,
        gin_ready=gin_ready,
        reason=_reason(
            torch_cuda_available=torch_cuda_available,
            torch_nccl_version=torch_nccl_version,
            headers=headers,
            header_has_device_api=header_has_device_api,
            header_has_gin_tokens=header_has_gin_tokens,
            device_api_version_ready=device_api_version_ready,
            gin_version_ready=gin_version_ready,
        ),
    )


def _torch_nccl_version() -> tuple[int, ...] | None:
    if not hasattr(torch.cuda, "nccl"):
        return None
    try:
        version = torch.cuda.nccl.version()
    except Exception:
        return None
    if isinstance(version, int):
        major = version // 1000
        minor = (version % 1000) // 100
        patch = version % 100
        return (major, minor, patch)
    if isinstance(version, str):
        parts = tuple(int(part) for part in version.split(".") if part.isdigit())
        return parts or None
    try:
        return tuple(int(part) for part in version)
    except TypeError:
        return None


def _candidate_include_dirs(extra_include_dirs: tuple[str, ...]) -> tuple[Path, ...]:
    candidates: list[Path] = []
    _append_existing(candidates, (Path(path) for path in extra_include_dirs))
    env_paths = []
    if os.environ.get("NCCL_INCLUDE_DIR"):
        env_paths.append(Path(os.environ["NCCL_INCLUDE_DIR"]))
    if os.environ.get("NCCL_ROOT"):
        env_paths.append(Path(os.environ["NCCL_ROOT"]) / "include")
    if os.environ.get("CUDA_HOME"):
        env_paths.append(Path(os.environ["CUDA_HOME"]) / "include")
    _append_existing(candidates, env_paths)

    site_packages = Path(sysconfig.get_paths()["purelib"])
    _append_existing(
        candidates,
        (
            site_packages / "nvidia" / "nccl" / "include",
            Path("/usr/local/cuda/include"),
            Path("/usr/include"),
            Path("/usr/local/include"),
        ),
    )
    try:
        from torch.utils.cpp_extension import include_paths

        _append_existing(candidates, (Path(path) for path in include_paths(cuda=True)))
    except Exception:
        pass
    return tuple(candidates)


def _append_existing(candidates: list[Path], paths) -> None:
    seen = {path.resolve() for path in candidates}
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            continue
        candidates.append(resolved)
        seen.add(resolved)


def _find_nccl_headers(include_dirs: tuple[Path, ...]) -> tuple[Path, ...]:
    headers: list[Path] = []
    for include_dir in include_dirs:
        for pattern in _HEADER_GLOBS:
            headers.extend(sorted(include_dir.glob(pattern)))
    return tuple(headers)


def _read_header_text(headers: tuple[Path, ...]) -> str:
    pieces = []
    for header in headers:
        try:
            pieces.append(header.read_text(errors="ignore"))
        except OSError:
            continue
    return "\n".join(pieces)


def _version_at_least(version: tuple[int, ...] | None, minimum: tuple[int, int, int]) -> bool:
    if version is None:
        return False
    padded = tuple(version) + (0,) * max(0, len(minimum) - len(version))
    return padded[: len(minimum)] >= minimum


def _reason(
    *,
    torch_cuda_available: bool,
    torch_nccl_version: tuple[int, ...] | None,
    headers: tuple[Path, ...],
    header_has_device_api: bool,
    header_has_gin_tokens: bool,
    device_api_version_ready: bool,
    gin_version_ready: bool,
) -> str:
    if not torch_cuda_available:
        return "CUDA is not available in this Python environment."
    if torch_nccl_version is None:
        return "PyTorch did not report an NCCL runtime version."
    if not device_api_version_ready:
        return f"NCCL runtime {torch_nccl_version} is older than {_DEVICE_API_MIN_VERSION}."
    if not headers:
        return "No NCCL headers were found."
    if not header_has_device_api:
        return "NCCL headers do not expose Device API tokens."
    if not gin_version_ready:
        return f"NCCL runtime {torch_nccl_version} is older than GIN target {_GIN_MIN_VERSION}."
    if not header_has_gin_tokens:
        return "NCCL headers expose Device API but no GIN/LSA tokens were found."
    return "NCCL Device API and GIN-like tokens are visible."
