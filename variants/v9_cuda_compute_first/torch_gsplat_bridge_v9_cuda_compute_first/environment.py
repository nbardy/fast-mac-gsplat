from __future__ import annotations

import platform
import shutil
from typing import Any


def cuda_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nvcc": shutil.which("nvcc"),
        "nvidia_smi": shutil.which("nvidia-smi"),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_built"] = bool(torch.backends.cuda.is_built())
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_device_count"] = int(torch.cuda.device_count())
        info["torch_mps_available"] = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
    except Exception as exc:  # pragma: no cover - environment reporting path.
        info["torch_import_error"] = repr(exc)
    return info


def require_cuda_extension() -> None:
    info = cuda_environment()
    reasons = []
    if not info.get("torch_cuda_built"):
        reasons.append("PyTorch was not built with CUDA")
    if not info.get("torch_cuda_available"):
        reasons.append("torch.cuda.is_available() is false")
    if info.get("nvcc") is None:
        reasons.append("nvcc is not on PATH")
    if info.get("nvidia_smi") is None:
        reasons.append("nvidia-smi is not on PATH")
    if reasons:
        raise RuntimeError(
            "v9_cuda_compute_first requires a CUDA host before native kernels can "
            "run: " + "; ".join(reasons)
        )

