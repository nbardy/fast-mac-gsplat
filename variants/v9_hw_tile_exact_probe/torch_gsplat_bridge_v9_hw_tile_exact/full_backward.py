from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class V9FullBackwardStatus:
    available: bool
    backend: str
    exact_forward: bool
    exact_backward: bool
    hardware_forward_state: bool
    uses_v8_compute_replay: bool
    error: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _variant_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / name


def _import_variant_package(package: str, variant_dir: str):
    try:
        return __import__(package)
    except ModuleNotFoundError:
        path = _variant_path(variant_dir)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        return __import__(package)


def _ops_ready(namespace: str) -> bool:
    ns = getattr(torch.ops, namespace, None)
    return ns is not None and hasattr(ns, "bin")


def _load_full_backward_backend():
    candidates = (
        ("torch_gsplat_bridge_v8", "v8", "gsplat_metal_v8", "v8_compute_replay"),
        ("torch_gsplat_bridge_v8_hw_eval", "v8_hw_eval", "gsplat_metal_v8_hw_eval", "v8_hw_eval_compute_replay"),
    )
    errors: list[str] = []
    for package, variant_dir, namespace, label in candidates:
        try:
            module = _import_variant_package(package, variant_dir)
            if _ops_ready(namespace):
                return module, label
            errors.append(f"{package} imported but torch.ops.{namespace}.bin is not registered")
        except Exception as exc:
            errors.append(f"{package}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def probe_full_backward() -> V9FullBackwardStatus:
    try:
        backend, label = _load_full_backward_backend()
        if not torch.backends.mps.is_available():
            return V9FullBackwardStatus(
                available=False,
                backend=label,
                exact_forward=True,
                exact_backward=True,
                hardware_forward_state=False,
                uses_v8_compute_replay=True,
                error="MPS is not available",
            )
        _ = backend.get_runtime_shader_config()
        return V9FullBackwardStatus(
            available=True,
            backend=label,
            exact_forward=True,
            exact_backward=True,
            hardware_forward_state=False,
            uses_v8_compute_replay=True,
            error="",
        )
    except Exception as exc:
        return V9FullBackwardStatus(
            available=False,
            backend="v8_compute_replay",
            exact_forward=True,
            exact_backward=True,
            hardware_forward_state=False,
            uses_v8_compute_replay=True,
            error=f"{type(exc).__name__}: {exc}",
        )


def make_full_backward_config(*args: Any, **kwargs: Any):
    """Create the V8 RasterConfig used by the current V9 full-backward base."""
    backend, _ = _load_full_backward_backend()
    return backend.RasterConfig(*args, **kwargs)


def rasterize_projected_gaussians_full_backward(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: Any,
) -> Tensor:
    """Run the complete exact forward/backward training path.

    This is intentionally the V8 compute replay backend. The V9 hardware-raster
    path is not allowed to own training gradients until it can emit V8-equivalent
    sorted bins and candidate-prefix `tile_stop_counts` for real Gaussians.
    """
    status = probe_full_backward()
    if not status.available:
        raise RuntimeError(f"V9 full-backward backend unavailable: {status.error}")
    backend, _ = _load_full_backward_backend()
    return backend.rasterize_projected_gaussians(
        means2d,
        conics,
        colors,
        opacities,
        depths,
        config,
    )


class ProjectedGaussianRasterizerFullBackward(torch.nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

    def forward(self, means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor) -> Tensor:
        return rasterize_projected_gaussians_full_backward(means2d, conics, colors, opacities, depths, self.config)
