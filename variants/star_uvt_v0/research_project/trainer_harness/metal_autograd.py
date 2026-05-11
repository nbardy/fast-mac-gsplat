from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig, render_uvt_tubes  # noqa: E402

try:
    from .model import dense_differentiable_render_uvt_tubes
except ImportError:  # pragma: no cover - script execution fallback.
    from model import dense_differentiable_render_uvt_tubes


class _MetalForwardDenseBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        depth_beta: Tensor,
        opacity: Tensor,
        color: Tensor,
        config: UVTRenderConfig,
    ) -> Tensor:
        ctx.config = config
        ctx.save_for_backward(ma, q_uvt, depth0, depth_beta, opacity, color)
        return render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        ma, q_uvt, depth0, depth_beta, opacity, color = ctx.saved_tensors
        with torch.enable_grad():
            ma_ref = ma.detach().requires_grad_(True)
            q_ref = q_uvt.detach().requires_grad_(True)
            depth0_ref = depth0.detach().requires_grad_(True)
            depth_beta_ref = depth_beta.detach().requires_grad_(True)
            opacity_ref = opacity.detach().requires_grad_(True)
            color_ref = color.detach().requires_grad_(True)
            image_ref = dense_differentiable_render_uvt_tubes(
                ma_ref,
                q_ref,
                depth0_ref,
                depth_beta_ref,
                opacity_ref,
                color_ref,
                ctx.config,
            )
            grads = torch.autograd.grad(
                image_ref,
                (ma_ref, q_ref, depth0_ref, depth_beta_ref, opacity_ref, color_ref),
                grad_output,
                allow_unused=True,
            )
        return (*grads, None)


def render_uvt_tubes_metal_forward_dense_backward(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: UVTRenderConfig,
) -> Tensor:
    """Use Metal forward while keeping dense PyTorch as the backward reference."""

    return _MetalForwardDenseBackward.apply(ma, q_uvt, depth0, depth_beta, opacity, color, config)

