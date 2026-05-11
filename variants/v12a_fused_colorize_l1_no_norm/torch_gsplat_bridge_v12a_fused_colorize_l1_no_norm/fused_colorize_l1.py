from __future__ import annotations

import torch
from torch import Tensor

try:
    from . import _C  # noqa: F401
except Exception:
    _C = None


def _as_weight_3f(weight: Tensor) -> Tensor:
    if weight.ndim == 4:
        if weight.shape[0] != 3 or weight.shape[2:] != (1, 1):
            raise ValueError(f"4D colorizer weight must have shape [3,F,1,1], got {tuple(weight.shape)}")
        return weight.reshape(3, weight.shape[1]).contiguous()
    if weight.ndim == 2:
        if weight.shape[0] != 3:
            raise ValueError(f"2D colorizer weight must have shape [3,F], got {tuple(weight.shape)}")
        return weight.contiguous()
    raise ValueError(f"colorizer weight must have shape [3,F] or [3,F,1,1], got {tuple(weight.shape)}")


def fused_no_norm_l1_grad(
    features: Tensor,
    alpha: Tensor,
    target_rgb: Tensor,
    background_rgb: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return no-norm sigmoid colorize + alpha-compose + L1 loss gradients.

    The op is a prototype gradient producer, not a torch autograd Function.

    Shapes:
        features: [N,H,W,F] contiguous float32 MPS
        alpha: [N,H,W] contiguous float32 MPS
        target_rgb/background_rgb: [N,3,H,W] contiguous float32 MPS
        weight: [3,F] or Conv2d-style [3,F,1,1]
        bias: [3]

    Returned gradients match `mean_n(mean_cyx(abs(pred-target)))`.
    """
    if not hasattr(torch.ops, "gsplat_metal_v12a_fused_colorize_l1_no_norm"):
        raise RuntimeError("gsplat_metal_v12a_fused_colorize_l1_no_norm custom ops not found. Build the extension first.")
    return torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.fused_no_norm_l1_grad(
        features.contiguous(),
        alpha.contiguous(),
        target_rgb.contiguous(),
        background_rgb.contiguous(),
        _as_weight_3f(weight),
        bias.contiguous(),
    )


def dssim_forward_grad(
    prediction: Tensor,
    target: Tensor,
    *,
    window_size: int = 11,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> tuple[Tensor, Tensor]:
    """Return per-image DSSIM and gradient for `dssim_per_image(...).mean()`.

    This is a cost-probe op for the current PyTorch DSSIM implementation:
    reflect-pad local means with an odd square window, then
    `0.5 * (1 - mean(ssim_map))`. It intentionally returns the dense RGB
    gradient rather than integrating with the rasterizer.

    Shapes:
        prediction/target: [N,C,H,W] contiguous float32 MPS

    Returns:
        loss_per_image: [N]
        grad_prediction: [N,C,H,W], scaled for `loss_per_image.mean()`
    """
    if not hasattr(torch.ops, "gsplat_metal_v12a_fused_colorize_l1_no_norm"):
        raise RuntimeError("gsplat_metal_v12a_fused_colorize_l1_no_norm custom ops not found. Build the extension first.")
    return torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.dssim_forward_grad(
        prediction.contiguous(),
        target.contiguous(),
        int(window_size),
        float(c1),
        float(c2),
    )
