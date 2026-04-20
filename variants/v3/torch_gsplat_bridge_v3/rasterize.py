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
    max_fast_pairs: int = 2048
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1e-4
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    enable_overflow_fallback: bool = True


def _make_meta(config: RasterConfig, device: torch.device) -> tuple[Tensor, Tensor]:
    if config.tile_size != 16:
        raise ValueError("The v3 Metal path is specialized for 16x16 tiles.")
    if config.max_fast_pairs > 2048:
        raise ValueError("max_fast_pairs cannot exceed the shader compile-time cap of 2048.")
    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    meta_i32 = torch.tensor(
        [
            config.height,
            config.width,
            tiles_y,
            tiles_x,
            config.tile_size,
            0,  # patched per call with G
            tiles_y * tiles_x,
            config.max_fast_pairs,
        ],
        device=device,
        dtype=torch.int32,
    )
    meta_f32 = torch.tensor(
        [
            float(config.alpha_threshold),
            float(config.transmittance_threshold),
            float(config.background[0]),
            float(config.background[1]),
            float(config.background[2]),
            1e-8,
            0.99,
        ],
        device=device,
        dtype=torch.float32,
    )
    return meta_i32, meta_f32


def _check_inputs(means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor) -> None:
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
    if conics.shape[0] != G or colors.shape[0] != G or opacities.shape[0] != G or depths.shape[0] != G:
        raise ValueError("all inputs must agree on G")


def _gather_overflow_segments(
    tile_counts: Tensor,
    tile_offsets: Tensor,
    binned_ids: Tensor,
    max_fast_pairs: int,
) -> tuple[Tensor, Tensor, Tensor]:
    # Rare slow path: use Python loops over only the overflow tiles.
    overflow_tile_ids = torch.nonzero(tile_counts > int(max_fast_pairs), as_tuple=False).flatten()
    if overflow_tile_ids.numel() == 0:
        device = tile_counts.device
        empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
        return empty_i32, torch.zeros((1,), device=device, dtype=torch.int32), empty_i32

    segments: list[Tensor] = []
    counts: list[int] = []
    for tile_id in overflow_tile_ids.tolist():
        start = int(tile_offsets[tile_id].item())
        end = int(tile_offsets[tile_id + 1].item())
        ids_t = binned_ids[start:end]
        if ids_t.numel() == 0:
            counts.append(0)
            continue
        perm = torch.argsort(ids_t, dim=0, stable=True)
        segments.append(ids_t.index_select(0, perm))
        counts.append(end - start)

    if segments:
        overflow_sorted_ids = torch.cat(segments, dim=0).contiguous()
    else:
        overflow_sorted_ids = torch.empty((0,), device=binned_ids.device, dtype=torch.int32)

    ov_counts = torch.tensor(counts, device=tile_counts.device, dtype=torch.int32)
    ov_offsets = torch.cat(
        [torch.zeros((1,), device=tile_counts.device, dtype=torch.int32), torch.cumsum(ov_counts, dim=0, dtype=torch.int32)],
        dim=0,
    ).contiguous()
    return overflow_tile_ids.to(torch.int32).contiguous(), ov_offsets, overflow_sorted_ids.to(torch.int32).contiguous()


def _tile_origin(tile_id: int, tiles_x: int, tile_size: int) -> tuple[int, int]:
    tx = tile_id % tiles_x
    ty = tile_id // tiles_x
    return tx * tile_size, ty * tile_size


def _scatter_tile_images_(base: Tensor, tile_ids: Tensor, tile_imgs: Tensor, tiles_x: int, tile_size: int) -> None:
    if tile_ids.numel() == 0:
        return
    H, W = base.shape[:2]
    for i, tile_id in enumerate(tile_ids.tolist()):
        x0, y0 = _tile_origin(int(tile_id), tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        base[y0:y1, x0:x1, :] = tile_imgs[i, : y1 - y0, : x1 - x0, :]


def _gather_tile_images(img: Tensor, tile_ids: Tensor, tiles_x: int, tile_size: int) -> Tensor:
    if tile_ids.numel() == 0:
        return torch.empty((0, tile_size, tile_size, img.shape[-1]), device=img.device, dtype=img.dtype)
    out = torch.zeros((tile_ids.numel(), tile_size, tile_size, img.shape[-1]), device=img.device, dtype=img.dtype)
    H, W = img.shape[:2]
    for i, tile_id in enumerate(tile_ids.tolist()):
        x0, y0 = _tile_origin(int(tile_id), tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        out[i, : y1 - y0, : x1 - x0, :] = img[y0:y1, x0:x1, :]
    return out


def _zero_tile_images_(img: Tensor, tile_ids: Tensor, tiles_x: int, tile_size: int) -> None:
    if tile_ids.numel() == 0:
        return
    H, W = img.shape[:2]
    for tile_id in tile_ids.tolist():
        x0, y0 = _tile_origin(int(tile_id), tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        img[y0:y1, x0:x1, :] = 0


class _RasterizeProjectedGaussiansV3(torch.autograd.Function):
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
        enable_overflow_fallback: bool,
    ) -> Tensor:
        if not hasattr(torch.ops, "gsplat_metal_v3"):
            raise RuntimeError("gsplat_metal_v3 custom ops not found. Build the extension first.")

        perm = torch.argsort(depths.detach(), dim=0, stable=True)
        means2d_s = means2d.index_select(0, perm).contiguous()
        conics_s = conics.index_select(0, perm).contiguous()
        colors_s = colors.index_select(0, perm).contiguous()
        opacities_s = opacities.index_select(0, perm).contiguous()

        tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v3.bin(
            means2d_s, conics_s, colors_s, opacities_s, meta_i32, meta_f32
        )
        out_fast = torch.ops.gsplat_metal_v3.render_fast_forward(
            means2d_s, conics_s, colors_s, opacities_s, meta_i32, meta_f32, tile_counts, tile_offsets, binned_ids
        )

        tiles_x = int(meta_i32[3].item())
        tile_size = int(meta_i32[4].item())
        max_fast_pairs = int(meta_i32[7].item())
        overflow_tile_ids = torch.empty((0,), device=means2d.device, dtype=torch.int32)
        overflow_tile_offsets = torch.zeros((1,), device=means2d.device, dtype=torch.int32)
        overflow_sorted_ids = torch.empty((0,), device=means2d.device, dtype=torch.int32)
        overflow_tile_imgs = torch.empty((0, tile_size, tile_size, 3), device=means2d.device, dtype=torch.float32)

        if enable_overflow_fallback:
            overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids = _gather_overflow_segments(
                tile_counts, tile_offsets, binned_ids, max_fast_pairs
            )
            if overflow_tile_ids.numel() > 0:
                overflow_tile_imgs = torch.ops.gsplat_metal_v3.render_overflow_forward(
                    means2d_s,
                    conics_s,
                    colors_s,
                    opacities_s,
                    meta_i32,
                    meta_f32,
                    overflow_tile_ids,
                    overflow_tile_offsets,
                    overflow_sorted_ids,
                )
                out = out_fast.clone()
                _scatter_tile_images_(out, overflow_tile_ids, overflow_tile_imgs, tiles_x, tile_size)
            else:
                out = out_fast
        else:
            if bool((tile_counts > max_fast_pairs).any().item()):
                raise RuntimeError(
                    f"Tile overflow detected with max_fast_pairs={max_fast_pairs}. "
                    "Enable overflow fallback or increase the runtime cap."
                )
            out = out_fast

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
            overflow_tile_ids,
            overflow_tile_offsets,
            overflow_sorted_ids,
        )
        ctx.tiles_x = tiles_x
        ctx.tile_size = tile_size
        ctx.enable_overflow_fallback = enable_overflow_fallback
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
            overflow_tile_ids,
            overflow_tile_offsets,
            overflow_sorted_ids,
        ) = ctx.saved_tensors

        grad_fast = grad_out.contiguous().clone()
        if ctx.enable_overflow_fallback and overflow_tile_ids.numel() > 0:
            _zero_tile_images_(grad_fast, overflow_tile_ids, ctx.tiles_x, ctx.tile_size)

        g_means2d_s, g_conics_s, g_colors_s, g_opacities_s = torch.ops.gsplat_metal_v3.render_fast_backward(
            grad_fast,
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

        if ctx.enable_overflow_fallback and overflow_tile_ids.numel() > 0:
            grad_tiles = _gather_tile_images(grad_out.contiguous(), overflow_tile_ids, ctx.tiles_x, ctx.tile_size)
            go_means, go_conics, go_colors, go_opacities = torch.ops.gsplat_metal_v3.render_overflow_backward(
                grad_tiles,
                means2d_s,
                conics_s,
                colors_s,
                opacities_s,
                meta_i32,
                meta_f32,
                overflow_tile_ids,
                overflow_tile_offsets,
                overflow_sorted_ids,
            )
            g_means2d_s = g_means2d_s + go_means
            g_conics_s = g_conics_s + go_conics
            g_colors_s = g_colors_s + go_colors
            g_opacities_s = g_opacities_s + go_opacities

        def unsort(grad: Tensor) -> Tensor:
            out = torch.empty_like(grad)
            out[perm] = grad
            return out

        g_means2d = unsort(g_means2d_s)
        g_conics = unsort(g_conics_s)
        g_colors = unsort(g_colors_s)
        g_opacities = unsort(g_opacities_s)
        g_depths = torch.zeros_like(depths)
        return g_means2d, g_conics, g_colors, g_opacities, g_depths, None, None, None


def rasterize_projected_gaussians(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: RasterConfig,
) -> Tensor:
    _check_inputs(means2d, conics, colors, opacities, depths)
    G = means2d.shape[0]
    meta_i32, meta_f32 = _make_meta(config, means2d.device)
    meta_i32 = meta_i32.clone()
    meta_i32[5] = G
    return _RasterizeProjectedGaussiansV3.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        depths.contiguous(),
        meta_i32,
        meta_f32,
        bool(config.enable_overflow_fallback),
    )


class ProjectedGaussianRasterizer(torch.nn.Module):
    def __init__(self, config: RasterConfig):
        super().__init__()
        self.config = config

    def forward(self, means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor) -> Tensor:
        return rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, self.config)
