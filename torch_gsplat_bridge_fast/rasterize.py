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
    tile_size: int = 16
    max_tile_pairs: int = 4096
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1e-4
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class _RasterizeProjectedGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        conics: Tensor,
        colors: Tensor,
        opacities: Tensor,
        depths: Tensor,
        meta_i32: Tensor,
        meta_f32: Tensor,
    ) -> Tensor:
        if not hasattr(torch.ops, "gsplat_metal_fast"):
            raise RuntimeError("gsplat_metal_fast custom ops not found. Build the extension first.")

        # Sort outside the custom op so the Metal kernels only ever see a monotone depth order.
        # We detach the permutation source because this renderer treats ordering as piecewise constant.
        perm = torch.argsort(depths.detach(), dim=0, stable=True)
        means2d_s = means2d.index_select(0, perm).contiguous()
        conics_s = conics.index_select(0, perm).contiguous()
        colors_s = colors.index_select(0, perm).contiguous()
        opacities_s = opacities.index_select(0, perm).contiguous()

        out, tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_fast.forward(
            means2d_s, conics_s, colors_s, opacities_s, meta_i32, meta_f32
        )
        ctx.save_for_backward(
            perm,
            means2d_s,
            conics_s,
            colors_s,
            opacities_s,
            depths,
            meta_i32,
            meta_f32,
            tile_counts,
            tile_offsets,
            binned_ids,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            perm,
            means2d_s,
            conics_s,
            colors_s,
            opacities_s,
            depths,
            meta_i32,
            meta_f32,
            tile_counts,
            tile_offsets,
            binned_ids,
        ) = ctx.saved_tensors

        g_means2d_s, g_conics_s, g_colors_s, g_opacities_s = torch.ops.gsplat_metal_fast.backward(
            grad_out.contiguous(),
            means2d_s,
            conics_s,
            colors_s,
            opacities_s,
            meta_i32,
            meta_f32,
            tile_counts,
            tile_offsets,
            binned_ids,
        )

        def unsort(grad: Tensor) -> Tensor:
            out = torch.empty_like(grad)
            out[perm] = grad
            return out

        g_means2d = unsort(g_means2d_s)
        g_conics = unsort(g_conics_s)
        g_colors = unsort(g_colors_s)
        g_opacities = unsort(g_opacities_s)
        g_depths = torch.zeros_like(depths)
        return g_means2d, g_conics, g_colors, g_opacities, g_depths, None, None


def _make_meta(config: RasterConfig, device: torch.device):
    if config.tile_size != 16:
        raise ValueError("The fast Metal path is currently specialized for 16x16 tiles.")
    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    meta_i32 = torch.tensor(
        [
            config.height,
            config.width,
            tiles_y,
            tiles_x,
            config.tile_size,
            0,  # patched per-call with G
            tiles_y * tiles_x,
            config.max_tile_pairs,
        ],
        device=device,
        dtype=torch.int32,
    )
    meta_f32 = torch.tensor(
        [
            config.alpha_threshold,
            config.transmittance_threshold,
            config.background[0],
            config.background[1],
            config.background[2],
            1e-8,
            0.99,
        ],
        device=device,
        dtype=torch.float32,
    )
    return meta_i32, meta_f32


def rasterize_projected_gaussians(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: RasterConfig,
) -> Tensor:
    if means2d.ndim != 2 or means2d.shape[-1] != 2:
        raise ValueError("means2d must have shape [G,2]")
    if conics.ndim != 2 or conics.shape[-1] != 3:
        raise ValueError("conics must have shape [G,3]")
    if colors.ndim != 2 or colors.shape[-1] != 3:
        raise ValueError("colors must have shape [G,3]")
    if opacities.ndim != 1:
        raise ValueError("opacities must have shape [G]")
    if depths.ndim != 1:
        raise ValueError("depths must have shape [G]")
    G = means2d.shape[0]
    device = means2d.device
    meta_i32, meta_f32 = _make_meta(config, device)
    meta_i32 = meta_i32.clone()
    meta_i32[5] = G
    return _RasterizeProjectedGaussians.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        depths.contiguous(),
        meta_i32,
        meta_f32,
    )


class ProjectedGaussianRasterizer(torch.nn.Module):
    def __init__(self, config: RasterConfig):
        super().__init__()
        self.config = config

    def forward(self, means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor) -> Tensor:
        return rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, self.config)
