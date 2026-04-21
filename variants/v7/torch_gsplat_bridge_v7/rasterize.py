from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

try:
    from . import _C  # noqa: F401
except Exception:
    _C = None


@dataclass(frozen=True)
class RasterConfig:
    height: int
    width: int
    front_k: int = 8
    alpha_threshold: float = 1.0 / 255.0
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def _normalize_inputs(means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor):
    was_batched = means2d.ndim == 3
    if means2d.ndim == 2:
        means2d = means2d.unsqueeze(0)
        conics = conics.unsqueeze(0)
        colors = colors.unsqueeze(0)
        opacities = opacities.unsqueeze(0)
        depths = depths.unsqueeze(0)
    if means2d.ndim != 3 or means2d.shape[-1] != 2:
        raise ValueError("means2d must be [G,2] or [B,G,2]")
    if conics.ndim != 3 or conics.shape[-1] != 3:
        raise ValueError("conics must be [G,3] or [B,G,3]")
    if colors.ndim != 3 or colors.shape[-1] != 3:
        raise ValueError("colors must be [G,3] or [B,G,3]")
    if opacities.ndim != 2 or depths.ndim != 2:
        raise ValueError("opacities/depths must be [G] or [B,G]")
    B, G, _ = means2d.shape
    if conics.shape[:2] != (B, G) or colors.shape[:2] != (B, G) or opacities.shape != (B, G) or depths.shape != (B, G):
        raise ValueError("inputs must agree on [B,G]")
    if not (means2d.device == conics.device == colors.device == opacities.device == depths.device):
        raise ValueError("all tensors must be on same device")
    if means2d.dtype != torch.float32 or conics.dtype != torch.float32 or colors.dtype != torch.float32 or opacities.dtype != torch.float32 or depths.dtype != torch.float32:
        raise ValueError("all inputs must be float32")
    return means2d, conics, colors, opacities, depths, was_batched


def _make_meta(config: RasterConfig, device: torch.device, B: int, G: int):
    meta_i32 = torch.tensor([config.height, config.width, B, G, config.front_k], device=device, dtype=torch.int32)
    meta_f32 = torch.tensor([config.alpha_threshold, config.background[0], config.background[1], config.background[2], 1e-8], device=device, dtype=torch.float32)
    return meta_i32, meta_f32


class _RasterizeProjectedGaussiansV7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor, meta_i32: Tensor, meta_f32: Tensor) -> Tensor:
        if not hasattr(torch.ops, "gsplat_metal_v7"):
            raise RuntimeError("gsplat_metal_v7 custom ops not found. Build the extension first.")
        B, G, _ = means2d.shape
        perm = torch.argsort(depths.detach(), dim=1, stable=True)
        means2d_s = torch.gather(means2d, 1, perm.unsqueeze(-1).expand(-1, -1, 2)).contiguous()
        conics_s = torch.gather(conics, 1, perm.unsqueeze(-1).expand(-1, -1, 3)).contiguous()
        colors_s = torch.gather(colors, 1, perm.unsqueeze(-1).expand(-1, -1, 3)).contiguous()
        opacities_s = torch.gather(opacities, 1, perm).contiguous()
        depths_s = torch.gather(depths, 1, perm).contiguous()
        out, aux = torch.ops.gsplat_metal_v7.forward(means2d_s, conics_s, colors_s, opacities_s, depths_s, meta_i32, meta_f32)
        ctx.save_for_backward(means2d_s, conics_s, colors_s, opacities_s, depths_s, perm, meta_i32, meta_f32, aux)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        means2d_s, conics_s, colors_s, opacities_s, depths_s, perm, meta_i32, meta_f32, aux = ctx.saved_tensors
        g_means_s, g_conics_s, g_colors_s, g_opacities_s, g_depths_s = torch.ops.gsplat_metal_v7.backward(
            grad_out.contiguous(), means2d_s, conics_s, colors_s, opacities_s, depths_s, meta_i32, meta_f32, aux
        )
        def unsort3(x: Tensor, chans: int) -> Tensor:
            out = torch.empty_like(x)
            out.scatter_(1, perm.unsqueeze(-1).expand(-1, -1, chans), x)
            return out
        def unsort2(x: Tensor) -> Tensor:
            out = torch.empty_like(x)
            out.scatter_(1, perm, x)
            return out
        return unsort3(g_means_s, 2), unsort3(g_conics_s, 3), unsort3(g_colors_s, 3), unsort2(g_opacities_s), unsort2(g_depths_s), None, None


def rasterize_projected_gaussians(means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor, config: RasterConfig) -> Tensor:
    means2d, conics, colors, opacities, depths, was_batched = _normalize_inputs(means2d, conics, colors, opacities, depths)
    meta_i32, meta_f32 = _make_meta(config, means2d.device, means2d.shape[0], means2d.shape[1])
    out = _RasterizeProjectedGaussiansV7.apply(means2d, conics, colors, opacities, depths, meta_i32, meta_f32)
    return out if was_batched else out.squeeze(0)


class ProjectedGaussianRasterizer(torch.nn.Module):
    def __init__(self, config: RasterConfig):
        super().__init__()
        self.config = config

    def forward(self, means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor) -> Tensor:
        return rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, self.config)
