from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    projective_rational_direct_serial_backward,
    projective_rational_tile_pair_atomic_backward,
    projective_rational_tile_pixel_atomic_backward,
    render_projective_rational_tubes_direct,
    render_projective_rational_tubes_tiled,
)


ForwardMode = Literal["direct", "tiled"]
BackwardMode = Literal["direct_serial", "tile_pair_atomic", "tile_pixel_atomic"]


def _render_forward(
    h_coeff: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: UVTRenderConfig,
    forward_mode: ForwardMode,
) -> Tensor:
    if forward_mode == "direct":
        return render_projective_rational_tubes_direct(
            h_coeff,
            lambda_uv,
            lambda_t,
            center_t,
            opacity,
            color,
            config,
        )
    if forward_mode == "tiled":
        return render_projective_rational_tubes_tiled(
            h_coeff,
            lambda_uv,
            lambda_t,
            center_t,
            opacity,
            color,
            config,
        )
    raise ValueError("forward_mode must be 'direct' or 'tiled'")


class _ProjectiveRationalDirectSerialBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        h_coeff: Tensor,
        lambda_uv: Tensor,
        lambda_t: Tensor,
        center_t: Tensor,
        opacity: Tensor,
        color: Tensor,
        config: UVTRenderConfig,
        forward_mode: ForwardMode,
        backward_mode: BackwardMode,
    ) -> Tensor:
        ctx.config = config
        ctx.forward_mode = forward_mode
        ctx.backward_mode = backward_mode
        ctx.save_for_backward(h_coeff, lambda_uv, lambda_t, center_t, opacity, color)
        return _render_forward(
            h_coeff,
            lambda_uv,
            lambda_t,
            center_t,
            opacity,
            color,
            config,
            forward_mode,
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        h_coeff, lambda_uv, lambda_t, center_t, opacity, color = ctx.saved_tensors
        backward_fn = {
            "direct_serial": projective_rational_direct_serial_backward,
            "tile_pair_atomic": projective_rational_tile_pair_atomic_backward,
            "tile_pixel_atomic": projective_rational_tile_pixel_atomic_backward,
        }[ctx.backward_mode]
        result = backward_fn(
            h_coeff.detach(),
            lambda_uv.detach(),
            lambda_t.detach(),
            center_t.detach(),
            opacity.detach(),
            color.detach(),
            grad_output.contiguous(),
            ctx.config,
        )
        grads = result[:6]
        return (*grads, None, None, None)


def render_projective_rational_tubes_metal_direct_serial_backward(
    h_coeff: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: UVTRenderConfig,
    *,
    forward_mode: ForwardMode = "tiled",
    backward_mode: BackwardMode = "direct_serial",
) -> Tensor:
    """Use Metal PRT forward with an explicit Metal backward mode."""

    return _ProjectiveRationalDirectSerialBackward.apply(
        h_coeff,
        lambda_uv,
        lambda_t,
        center_t,
        opacity,
        color,
        config,
        forward_mode,
        backward_mode,
    )
