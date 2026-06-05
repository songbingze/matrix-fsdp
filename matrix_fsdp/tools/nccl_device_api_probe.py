from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from matrix_fsdp.kernels.gin import probe_gin_device_api


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe NCCL Device API / GIN readiness for MatrixFSDP kernels.")
    parser.add_argument(
        "--include-dir",
        action="append",
        default=(),
        help="Extra NCCL include directory to scan. Can be passed multiple times.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    probe = probe_gin_device_api(extra_include_dirs=tuple(args.include_dir))
    if args.json:
        print(json.dumps(probe.as_dict(), indent=2, sort_keys=True))
    else:
        _print_text(probe.as_dict())
    return 0


def _print_text(data: dict[str, object]) -> None:
    rows = [
        ("CUDA available", data["torch_cuda_available"]),
        ("torch NCCL version", _format_version(data["torch_nccl_version"])),
        ("headers found", len(data["headers"])),  # type: ignore[arg-type]
        ("header has Device API", data["header_has_device_api"]),
        ("header has GIN tokens", data["header_has_gin_tokens"]),
        ("Device API version ready", data["device_api_version_ready"]),
        ("GIN version ready", data["gin_version_ready"]),
        ("Device API ready", data["device_api_ready"]),
        ("GIN ready", data["gin_ready"]),
        ("reason", data["reason"]),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}} : {value}")
    headers = data["headers"]
    if headers:
        print("NCCL headers:")
        for header in headers:  # type: ignore[union-attr]
            print(f"  {header}")


def _format_version(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        return ".".join(str(part) for part in value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
