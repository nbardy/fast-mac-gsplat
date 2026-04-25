from pathlib import Path
import shutil

from setuptools import setup


this_dir = Path(__file__).resolve().parent


def _cuda_build_error() -> RuntimeError:
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception as exc:
        return RuntimeError(
            "v9_cuda_compute_first requires a PyTorch install with CUDA extension "
            f"support. Importing torch/extension helpers failed: {exc!r}"
        )

    reasons = []
    if not torch.backends.cuda.is_built():
        reasons.append("PyTorch was not built with CUDA")
    if CUDA_HOME is None:
        reasons.append("torch.utils.cpp_extension.CUDA_HOME is unset")
    if shutil.which("nvcc") is None:
        reasons.append("nvcc is not on PATH")

    if reasons:
        return RuntimeError(
            "v9_cuda_compute_first is a CUDA-only scaffold and cannot be built on "
            "this host: " + "; ".join(reasons)
        )
    return None


build_error = _cuda_build_error()
if build_error is not None:
    raise build_error


from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402


sources = [
    str(this_dir / "csrc" / "bindings.cpp"),
    str(this_dir / "csrc" / "cuda" / "project_count_fused.cu"),
    str(this_dir / "csrc" / "cuda" / "emit_pairs.cu"),
    str(this_dir / "csrc" / "cuda" / "tile_forward_train.cu"),
    str(this_dir / "csrc" / "cuda" / "tile_backward_replay.cu"),
]


ext_modules = [
    CUDAExtension(
        name="torch_gsplat_bridge_v9_cuda_compute_first._C",
        sources=sources,
        include_dirs=[str(this_dir / "csrc" / "include")],
        extra_compile_args={
            "cxx": ["-std=c++17"],
            "nvcc": ["-std=c++17", "--use_fast_math"],
        },
    )
]


setup(
    name="torch-gsplat-bridge-v9-cuda-compute-first",
    version="0.1.0",
    description="Source-level CUDA compute-first scaffold for exact 3DGS rasterization",
    packages=["torch_gsplat_bridge_v9_cuda_compute_first"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
