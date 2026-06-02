from __future__ import annotations

import os
from pathlib import Path
import sysconfig

from setuptools import setup


def _cuda_kernel_build_enabled() -> bool:
    return os.environ.get("MATRIX_FSDP_BUILD_CUDA_KERNELS", "").lower() in {"1", "true", "yes", "on"}


def _gin_kernel_build_enabled() -> bool:
    return os.environ.get("MATRIX_FSDP_BUILD_GIN_KERNELS", "").lower() in {"1", "true", "yes", "on"}


def _optional_cuda_extensions():
    if not (_cuda_kernel_build_enabled() or _gin_kernel_build_enabled()):
        return [], {}
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as exc:  # pragma: no cover - depends on optional CUDA build env.
        raise RuntimeError(
            "MatrixFSDP CUDA extension builds require torch with CUDA extension build support. "
            "Use --no-build-isolation so setup.py can import the installed torch package."
        ) from exc

    extensions = []
    nvidia_include_dirs, nvidia_library_dirs, nvidia_runtime_dirs = _nvidia_package_paths()
    nccl_libraries, nccl_extra_link_args = _nccl_link_args(nvidia_library_dirs)
    common_kwargs = {
        "include_dirs": nvidia_include_dirs,
        "library_dirs": nvidia_library_dirs,
        "libraries": nccl_libraries,
        "extra_compile_args": {
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math"],
        },
        "extra_link_args": nccl_extra_link_args + [f"-Wl,-rpath,{path}" for path in nvidia_runtime_dirs],
    }
    if _cuda_kernel_build_enabled():
        sources = [
            Path("matrix_fsdp") / "kernels" / "native" / "matrix_copy.cpp",
            Path("matrix_fsdp") / "kernels" / "native" / "matrix_copy_cuda.cu",
            Path("matrix_fsdp") / "kernels" / "native" / "matrix_nccl.cpp",
        ]
        extensions.append(
            CUDAExtension(
                "matrix_fsdp._matrix_fsdp_cuda",
                sources=[str(path) for path in sources],
                **common_kwargs,
            )
        )
    if _gin_kernel_build_enabled():
        sources = [
            Path("matrix_fsdp") / "kernels" / "gin" / "native" / "rma_nccl.cpp",
        ]
        extensions.append(
            CUDAExtension(
                "matrix_fsdp._matrix_fsdp_gin_cuda",
                sources=[str(path) for path in sources],
                **common_kwargs,
            )
        )
    return extensions, {"build_ext": BuildExtension}


def _nvidia_package_paths() -> tuple[list[str], list[str], list[str]]:
    site_package_roots = {
        Path(path)
        for key in ("purelib", "platlib")
        if (path := sysconfig.get_paths().get(key))
    }
    nvidia_roots = [site_packages / "nvidia" for site_packages in site_package_roots]
    include_dirs: list[str] = []
    library_dirs: list[str] = []
    for nvidia_root in nvidia_roots:
        if not nvidia_root.exists():
            continue
        include_dirs.extend(str(path) for path in sorted(nvidia_root.glob("*/include")) if path.is_dir())
        library_dirs.extend(str(path) for path in sorted(nvidia_root.glob("*/lib")) if path.is_dir())
    return _unique_paths(include_dirs), _unique_paths(library_dirs), _unique_paths(library_dirs)


def _nccl_link_args(library_dirs: list[str]) -> tuple[list[str], list[str]]:
    nccl_dirs = [Path(path) for path in library_dirs if Path(path).name == "lib" and Path(path).parent.name == "nccl"]
    if not nccl_dirs:
        return [], []
    if any((path / "libnccl.so").exists() for path in nccl_dirs):
        return ["nccl"], []
    for nccl_dir in nccl_dirs:
        candidates = sorted(nccl_dir.glob("libnccl.so*"))
        if candidates:
            return [], [str(candidates[0])]
    return [], []


def _unique_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


ext_modules, cmdclass = _optional_cuda_extensions()

setup(ext_modules=ext_modules, cmdclass=cmdclass)
