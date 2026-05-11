from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor


Activation = Literal["sigmoid"]


@dataclass(frozen=True)
class RgbGradHandoffMemory:
    batch: int
    height: int
    width: int
    feature_dim: int
    dtype_bytes: int
    current_grad_features_bytes: int
    current_grad_alpha_bytes: int
    current_dense_backward_input_bytes: int
    handoff_grad_rgb_bytes: int
    handoff_dense_backward_input_bytes: int
    avoided_bytes: int
    avoided_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def estimate_rgb_grad_handoff_memory(
    *,
    batch: int,
    height: int,
    width: int,
    feature_dim: int,
    dtype_bytes: int = 4,
) -> RgbGradHandoffMemory:
    """Estimate dense backward-input traffic removed by RGB-gradient handoff.

    Current v11-compatible backward consumes `grad_features[B,H,W,F]` and
    `grad_alpha[B,H,W]`. The v13b target consumes an image-space
    `grad_composed_rgb[B,H,W,3]` and computes feature/alpha VJPs inside the
    raster backward kernel.
    """
    for name, value in {
        "batch": batch,
        "height": height,
        "width": width,
        "feature_dim": feature_dim,
        "dtype_bytes": dtype_bytes,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    pixels = int(batch) * int(height) * int(width)
    grad_features = pixels * int(feature_dim) * int(dtype_bytes)
    grad_alpha = pixels * int(dtype_bytes)
    current = grad_features + grad_alpha
    handoff_rgb = pixels * 3 * int(dtype_bytes)
    avoided = current - handoff_rgb
    return RgbGradHandoffMemory(
        batch=int(batch),
        height=int(height),
        width=int(width),
        feature_dim=int(feature_dim),
        dtype_bytes=int(dtype_bytes),
        current_grad_features_bytes=grad_features,
        current_grad_alpha_bytes=grad_alpha,
        current_dense_backward_input_bytes=current,
        handoff_grad_rgb_bytes=handoff_rgb,
        handoff_dense_backward_input_bytes=handoff_rgb,
        avoided_bytes=avoided,
        avoided_fraction=(float(avoided) / float(current)) if current else 0.0,
    )


def _as_weight_3f(weight: Tensor, feature_dim: int) -> Tensor:
    if weight.ndim == 4:
        if tuple(weight.shape[0:1] + weight.shape[2:]) != (3, 1, 1):
            raise ValueError(f"4D colorizer weight must have shape [3,F,1,1], got {tuple(weight.shape)}")
        weight = weight.reshape(3, weight.shape[1])
    if tuple(weight.shape) != (3, feature_dim):
        raise ValueError(f"color_weight must have shape {(3, feature_dim)}, got {tuple(weight.shape)}")
    return weight.contiguous()


def rgb_grad_handoff_backward(
    out_features: Tensor,
    out_alpha: Tensor,
    grad_composed_rgb: Tensor,
    background_rgb: Tensor,
    color_weight: Tensor,
    color_bias: Tensor,
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    meta_i32: Tensor,
    meta_f32: Tensor,
    meta_host_i32: Tensor,
    meta_host_f32: Tensor,
    tile_counts: Tensor,
    tile_offsets: Tensor,
    binned_ids: Tensor,
    tile_stop_counts: Tensor,
    *,
    activation: Activation = "sigmoid",
    compute_color_param_grads: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Target v13b fused backward boundary.

    This wrapper defines the low-level handoff contract only. The registered
    C++ op currently raises because the Metal kernel is still missing.

    Shapes:
        out_features: [B,H,W,F] float32 MPS from forward-state rasterization
        out_alpha: [B,H,W] float32 MPS
        grad_composed_rgb/background_rgb: [B,H,W,3] float32 MPS
        color_weight: [3,F] or [3,F,1,1]
        color_bias: [3]

    The future kernel should compute `rgb = sigmoid(W @ out_features + b)`,
    apply `grad_composed_rgb` to get local feature/alpha/colorizer gradients,
    and stream the local feature/alpha gradients directly into the reverse
    raster contributor loop without allocating `grad_features[B,H,W,F]`.
    """
    if activation != "sigmoid":
        raise ValueError("v13b scaffold currently defines only sigmoid colorizer handoff")
    if out_features.ndim != 4:
        raise ValueError(f"out_features must have shape [B,H,W,F], got {tuple(out_features.shape)}")
    if out_alpha.shape != out_features.shape[:3]:
        raise ValueError(f"out_alpha must match out_features[:3], got {tuple(out_alpha.shape)}")
    if grad_composed_rgb.shape != (*out_features.shape[:3], 3):
        raise ValueError(
            "grad_composed_rgb must have shape "
            f"{(*out_features.shape[:3], 3)}, got {tuple(grad_composed_rgb.shape)}"
        )
    if background_rgb.shape != grad_composed_rgb.shape:
        raise ValueError(
            f"background_rgb must match grad_composed_rgb, got {tuple(background_rgb.shape)}"
        )
    feature_dim = int(out_features.shape[-1])
    weight = _as_weight_3f(color_weight, feature_dim)
    if tuple(color_bias.shape) != (3,):
        raise ValueError(f"color_bias must have shape (3,), got {tuple(color_bias.shape)}")
    if not hasattr(torch.ops, "gsplat_metal_v13b_rgb_grad_handoff"):
        raise RuntimeError("gsplat_metal_v13b_rgb_grad_handoff custom ops not found. Build the extension first.")

    params = torch.tensor(
        [1.0 if compute_color_param_grads else 0.0],
        device=out_features.device,
        dtype=torch.float32,
    )
    return torch.ops.gsplat_metal_v13b_rgb_grad_handoff.render_fast_backward_rgb_grad_handoff(
        out_features.contiguous(),
        out_alpha.contiguous(),
        grad_composed_rgb.contiguous(),
        background_rgb.contiguous(),
        weight,
        color_bias.contiguous(),
        params,
        means2d,
        conics,
        colors,
        opacities,
        meta_i32,
        meta_f32,
        meta_host_i32,
        meta_host_f32,
        tile_counts,
        tile_offsets,
        binned_ids,
        tile_stop_counts,
    )
