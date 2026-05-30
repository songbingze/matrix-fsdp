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
    nccl_include_dirs, nccl_library_dirs, nccl_runtime_dirs = _nccl_paths()
    common_kwargs = {
        "include_dirs": nccl_include_dirs,
        "library_dirs": nccl_library_dirs,
        "libraries": ["nccl"] if nccl_library_dirs else [],
        "extra_compile_args": {
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math"],
        },
        "extra_link_args": [f"-Wl,-rpath,{path}" for path in nccl_runtime_dirs],
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


def _nccl_paths() -> tuple[list[str], list[str], list[str]]:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    nccl_root = site_packages / "nvidia" / "nccl"
    include_dir = nccl_root / "include"
    lib_dir = nccl_root / "lib"
    include_dirs = [str(include_dir)] if (include_dir / "nccl.h").exists() else []
    library_dirs = [str(lib_dir)] if any(lib_dir.glob("libnccl.so*")) else []
    return include_dirs, library_dirs, library_dirs


ext_modules, cmdclass = _optional_cuda_extensions()

setup(ext_modules=ext_modules, cmdclass=cmdclass)
