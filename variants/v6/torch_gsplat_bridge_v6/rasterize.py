
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

try:
    from . import _C  # noqa: F401
except Exception:
    _C = None


@dataclass(frozen=True)
class RuntimeShaderConfig:
    tile_size: int
    threads: int
    chunk_size: int
    fast_cap: int
    simdgroups: int


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


def get_runtime_shader_config() -> RuntimeShaderConfig:
    tile_size = _env_int("GSP_TILE_SIZE", 16)
    chunk_size = _env_int("GSP_CHUNK", 64)
    fast_cap = _env_int("GSP_FAST_CAP", 2048)
    if tile_size not in (8, 16, 32):
        raise ValueError(f"GSP_TILE_SIZE must be one of 8, 16, 32; got {tile_size}")
    threads = tile_size * tile_size
    if threads > 1024:
        raise ValueError(f"tile_size={tile_size} implies {threads} threads, which exceeds 1024")
    if chunk_size <= 0:
        raise ValueError("GSP_CHUNK must be positive")
    if fast_cap <= 0:
        raise ValueError("GSP_FAST_CAP must be positive")
    simdgroups = (threads + 31) // 32
    return RuntimeShaderConfig(
        tile_size=tile_size,
        threads=threads,
        chunk_size=chunk_size,
        fast_cap=fast_cap,
        simdgroups=simdgroups,
    )


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
    batch_strategy: str = "auto"  # auto | flatten | serial
    batch_launch_limit_tiles: int = 262144
    batch_launch_limit_gaussians: int = 262144
    use_active_tiles: bool = False
    active_policy: str = "off"  # off | on | auto
    sort_active_tiles_by_count: bool = True
    stop_count_mode: str = "adaptive"  # always | never | adaptive
    stop_count_dense_threshold: int = 64
    max_pairs_per_launch: int = 0  # 0 disables pair-budget training chunks


def _runtime_validate(config: RasterConfig) -> RuntimeShaderConfig:
    rt = get_runtime_shader_config()
    if config.tile_size != rt.tile_size:
        raise ValueError(
            f"RasterConfig.tile_size={config.tile_size} does not match runtime shader tile size {rt.tile_size}. "
            "Set GSP_TILE_SIZE before importing/running the extension, or adjust RasterConfig."
        )
    if config.max_fast_pairs > rt.fast_cap:
        raise ValueError(
            f"RasterConfig.max_fast_pairs={config.max_fast_pairs} exceeds compiled fast cap {rt.fast_cap}. "
            "Lower the runtime cap or set GSP_FAST_CAP before import."
        )
    if config.batch_strategy not in ("auto", "flatten", "serial"):
        raise ValueError("batch_strategy must be one of: auto, flatten, serial")
    if config.active_policy not in ("off", "on", "auto"):
        raise ValueError("active_policy must be one of: off, on, auto")
    if config.stop_count_mode not in ("always", "never", "adaptive"):
        raise ValueError("stop_count_mode must be one of: always, never, adaptive")
    if config.stop_count_dense_threshold <= 0:
        raise ValueError("stop_count_dense_threshold must be positive")
    if config.max_pairs_per_launch < 0:
        raise ValueError("max_pairs_per_launch must be non-negative")
    return rt


def _stop_mode_to_int(mode: str) -> int:
    return {"always": 0, "never": 1, "adaptive": 2}[mode]


def _active_policy_to_int(policy: str) -> int:
    return {"off": 0, "on": 1, "auto": 2}[policy]


def _active_policy_from_int(policy: int) -> str:
    return {0: "off", 1: "on", 2: "auto"}[int(policy)]


def _effective_active_policy(config: RasterConfig) -> str:
    # Preserve the pre-v6.1 API: RasterConfig(use_active_tiles=True) still means "on".
    if config.active_policy == "off" and config.use_active_tiles:
        return "on"
    return config.active_policy


def _make_meta(config: RasterConfig, device: torch.device, batch_size: int, gaussians_per_batch: int) -> tuple[Tensor, Tensor]:
    _runtime_validate(config)
    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    tiles_per_image = tiles_y * tiles_x
    total_tiles = batch_size * tiles_per_image
    total_gaussians = batch_size * gaussians_per_batch
    meta_i32 = torch.tensor(
        [
            config.height,
            config.width,
            tiles_y,
            tiles_x,
            config.tile_size,
            total_gaussians,
            total_tiles,
            config.max_fast_pairs,
            batch_size,
            gaussians_per_batch,
            tiles_per_image,
            _stop_mode_to_int(config.stop_count_mode),
            int(config.stop_count_dense_threshold),
            0,
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


def _normalize_inputs(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, bool]:
    if means2d.ndim == 2:
        return (
            means2d.unsqueeze(0),
            conics.unsqueeze(0),
            colors.unsqueeze(0),
            opacities.unsqueeze(0),
            depths.unsqueeze(0),
            False,
        )
    if means2d.ndim != 3:
        raise ValueError("means2d must have shape [G,2] or [B,G,2]")
    return means2d, conics, colors, opacities, depths, True


def _check_inputs(means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor) -> None:
    if means2d.ndim not in (2, 3):
        raise ValueError("means2d must have shape [G,2] or [B,G,2]")
    if conics.ndim != means2d.ndim or colors.ndim != means2d.ndim:
        raise ValueError("conics/colors rank must match means2d rank")
    if opacities.ndim != means2d.ndim - 1 or depths.ndim != means2d.ndim - 1:
        raise ValueError("opacities/depths rank must be one less than means2d rank")
    if means2d.shape[-1] != 2:
        raise ValueError("means2d must have last dim = 2")
    if conics.shape[-1] != 3:
        raise ValueError("conics must have last dim = 3")
    if colors.shape[-1] != 3:
        raise ValueError("colors must have last dim = 3")
    if means2d.shape[:-1] != conics.shape[:-1] or means2d.shape[:-1] != colors.shape[:-1]:
        raise ValueError("means2d/conics/colors batch/G dimensions must match")
    if means2d.shape[:-1] != opacities.shape or means2d.shape[:-1] != depths.shape:
        raise ValueError("means2d/opacities/depths batch/G dimensions must match")
    for name, t in {
        "means2d": means2d,
        "conics": conics,
        "colors": colors,
        "opacities": opacities,
        "depths": depths,
    }.items():
        if t.device != means2d.device:
            raise ValueError(f"{name} must be on the same device as means2d")
        if t.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
    if means2d.device.type != "mps":
        raise ValueError("v6 fast path requires MPS tensors")


def _batched_gather_2d(x: Tensor, perm: Tensor) -> Tensor:
    return x.gather(1, perm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _batched_gather_1d(x: Tensor, perm: Tensor) -> Tensor:
    return x.gather(1, perm)


def _unsort_batched(grad: Tensor, perm: Tensor) -> Tensor:
    out = torch.empty_like(grad)
    if grad.ndim == 3:
        out.scatter_(1, perm.unsqueeze(-1).expand_as(grad), grad)
    elif grad.ndim == 2:
        out.scatter_(1, perm, grad)
    else:
        raise ValueError(f"unexpected grad rank: {grad.ndim}")
    return out


def _gather_overflow_segments(
    tile_counts: Tensor,
    tile_offsets: Tensor,
    binned_ids: Tensor,
    max_fast_pairs: int,
) -> tuple[Tensor, Tensor, Tensor]:
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

    overflow_sorted_ids = (
        torch.cat(segments, dim=0).contiguous()
        if segments
        else torch.empty((0,), device=binned_ids.device, dtype=torch.int32)
    )
    ov_counts = torch.tensor(counts, device=tile_counts.device, dtype=torch.int32)
    ov_offsets = torch.cat(
        [torch.zeros((1,), device=tile_counts.device, dtype=torch.int32), torch.cumsum(ov_counts, dim=0, dtype=torch.int32)],
        dim=0,
    ).contiguous()
    return overflow_tile_ids.to(torch.int32).contiguous(), ov_offsets, overflow_sorted_ids.to(torch.int32).contiguous()


def _tile_origin_global(tile_id: int, tiles_per_image: int, tiles_x: int, tile_size: int) -> tuple[int, int, int]:
    batch = tile_id // tiles_per_image
    local_tile = tile_id % tiles_per_image
    tx = local_tile % tiles_x
    ty = local_tile // tiles_x
    return batch, tx * tile_size, ty * tile_size


def _scatter_tile_images_(base: Tensor, tile_ids: Tensor, tile_imgs: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> None:
    if tile_ids.numel() == 0:
        return
    _, H, W = base.shape[:3]
    for i, tile_id in enumerate(tile_ids.tolist()):
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        base[b, y0:y1, x0:x1, :] = tile_imgs[i, : y1 - y0, : x1 - x0, :]


def _gather_tile_images(img: Tensor, tile_ids: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> Tensor:
    if tile_ids.numel() == 0:
        return torch.empty((0, tile_size, tile_size, img.shape[-1]), device=img.device, dtype=img.dtype)
    out = torch.zeros((tile_ids.numel(), tile_size, tile_size, img.shape[-1]), device=img.device, dtype=img.dtype)
    _, H, W = img.shape[:3]
    for i, tile_id in enumerate(tile_ids.tolist()):
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        out[i, : y1 - y0, : x1 - x0, :] = img[b, y0:y1, x0:x1, :]
    return out


def _should_use_training_path(means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor) -> bool:
    if not torch.is_grad_enabled():
        return False
    return bool(means2d.requires_grad or conics.requires_grad or colors.requires_grad or opacities.requires_grad)


def _choose_batch_chunk_size(config: RasterConfig, batch_size: int, gaussians_per_batch: int, tiles_per_image: int) -> int:
    if config.batch_strategy == "flatten":
        return batch_size
    if config.batch_strategy == "serial":
        return 1
    by_tiles = max(1, config.batch_launch_limit_tiles // max(tiles_per_image, 1))
    by_gaussians = max(1, config.batch_launch_limit_gaussians // max(gaussians_per_batch, 1))
    return max(1, min(batch_size, by_tiles, by_gaussians))


def _contiguous_batch_slices(batch_size: int, chunk_b: int) -> list[tuple[int, int]]:
    return [(b0, min(batch_size, b0 + chunk_b)) for b0 in range(0, batch_size, chunk_b)]


@torch.no_grad()
def _choose_batch_slices(
    means2d_b: Tensor,
    conics_b: Tensor,
    colors_b: Tensor,
    opacities_b: Tensor,
    config: RasterConfig,
    *,
    train_mode: bool,
    tiles_per_image: int,
) -> list[tuple[int, int]]:
    B, G = means2d_b.shape[:2]
    chunk_b = _choose_batch_chunk_size(config, B, G, tiles_per_image)
    base_slices = _contiguous_batch_slices(B, chunk_b)
    if (
        not train_mode
        or config.batch_strategy != "auto"
        or config.max_pairs_per_launch <= 0
        or B <= 1
    ):
        return base_slices

    budget = int(config.max_pairs_per_launch)
    split_slices: list[tuple[int, int]] = []
    for b0, b1 in base_slices:
        m = means2d_b[b0:b1].detach().contiguous()
        q = conics_b[b0:b1].detach().contiguous()
        c = colors_b[b0:b1].detach().contiguous()
        o = opacities_b[b0:b1].detach().contiguous()
        meta_i32, meta_f32 = _make_meta(config, m.device, b1 - b0, G)
        tile_counts, _, _ = torch.ops.gsplat_metal_v6.bin(
            m.reshape(-1, 2).contiguous(),
            q.reshape(-1, 3).contiguous(),
            c.reshape(-1, 3).contiguous(),
            o.reshape(-1).contiguous(),
            meta_i32,
            meta_f32,
        )
        per_image_pairs = tile_counts.view(b1 - b0, tiles_per_image).sum(dim=1).detach().cpu().tolist()
        cur_start = b0
        cur_pairs = 0
        for offset, pairs_raw in enumerate(per_image_pairs):
            pairs = int(pairs_raw)
            image_index = b0 + offset
            if image_index > cur_start and cur_pairs + pairs > budget:
                split_slices.append((cur_start, image_index))
                cur_start = image_index
                cur_pairs = 0
            cur_pairs += pairs
        if cur_start < b1:
            split_slices.append((cur_start, b1))
    return split_slices


def _make_active_tile_ids(tile_counts: Tensor, max_fast_pairs: int, *, sort_by_count: bool) -> Tensor:
    fast_mask = (tile_counts > 0) & (tile_counts <= int(max_fast_pairs))
    active_tile_ids = torch.nonzero(fast_mask, as_tuple=False).flatten().to(torch.int32)
    if active_tile_ids.numel() == 0:
        return active_tile_ids.contiguous()
    if sort_by_count:
        counts = tile_counts.index_select(0, active_tile_ids.to(torch.long))
        perm = torch.argsort(counts, stable=True)
        active_tile_ids = active_tile_ids.index_select(0, perm)
    return active_tile_ids.contiguous()


def _resolve_active_tile_use(
    *,
    policy: str,
    tile_counts: Tensor,
    max_fast_pairs: int,
    total_tiles: int,
) -> bool:
    if policy == "off":
        return False
    if policy == "on":
        return True

    active_tile_count = int((tile_counts > 0).sum().item())
    active_tile_fraction = active_tile_count / max(int(total_tiles), 1)
    overflow_tile_count = int((tile_counts > int(max_fast_pairs)).sum().item())
    max_pairs_per_tile = int(tile_counts.max().item()) if tile_counts.numel() else 0

    if active_tile_fraction > 0.75 and overflow_tile_count == 0:
        return False
    return (
        active_tile_fraction < 0.10
        or overflow_tile_count > 0
        or max_pairs_per_tile > 2 * int(max_fast_pairs)
    )


def _band_counts(tile_counts: Tensor, active_tile_ids: Tensor, dense_threshold: int) -> dict[str, int]:
    if active_tile_ids.numel() == 0:
        return {"active_tile_count": 0, "dense_active_tile_count": 0, "light_active_tile_count": 0}
    counts = tile_counts.index_select(0, active_tile_ids.to(torch.long)).to(torch.int32)
    dense = counts >= int(dense_threshold)
    return {
        "active_tile_count": int(active_tile_ids.numel()),
        "dense_active_tile_count": int(dense.sum().item()),
        "light_active_tile_count": int((~dense).sum().item()),
    }


class _RasterizeProjectedGaussiansV6(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d_b: Tensor,
        conics_b: Tensor,
        colors_b: Tensor,
        opacities_b: Tensor,
        depths_b: Tensor,
        meta_i32: Tensor,
        meta_f32: Tensor,
        enable_overflow_fallback: bool,
        use_active_tiles: bool,
        active_policy_code: int,
        sort_active_tiles_by_count: bool,
    ) -> Tensor:
        if not hasattr(torch.ops, "gsplat_metal_v6"):
            raise RuntimeError("gsplat_metal_v6 custom ops not found. Build the extension first.")

        B, G = means2d_b.shape[:2]
        perm = torch.argsort(depths_b.detach(), dim=1, stable=True)
        means2d_s = _batched_gather_2d(means2d_b, perm).contiguous()
        conics_s = _batched_gather_2d(conics_b, perm).contiguous()
        colors_s = _batched_gather_2d(colors_b, perm).contiguous()
        opacities_s = _batched_gather_1d(opacities_b, perm).contiguous()

        means_flat = means2d_s.reshape(B * G, 2).contiguous()
        conics_flat = conics_s.reshape(B * G, 3).contiguous()
        colors_flat = colors_s.reshape(B * G, 3).contiguous()
        opacities_flat = opacities_s.reshape(B * G).contiguous()

        tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v6.bin(
            means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32
        )

        active_policy = _active_policy_from_int(active_policy_code)
        if active_policy == "off" and use_active_tiles:
            active_policy = "on"
        resolved_use_active_tiles = _resolve_active_tile_use(
            policy=active_policy,
            tile_counts=tile_counts,
            max_fast_pairs=int(meta_i32[7].item()),
            total_tiles=int(meta_i32[6].item()),
        )

        if resolved_use_active_tiles:
            active_tile_ids = _make_active_tile_ids(
                tile_counts,
                int(meta_i32[7].item()),
                sort_by_count=bool(sort_active_tiles_by_count),
            )
            out_fast, tile_stop_counts = torch.ops.gsplat_metal_v6.render_active_forward_state(
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                meta_i32,
                meta_f32,
                binned_ids,
                active_tile_ids,
                tile_counts,
                tile_offsets,
            )
        else:
            active_tile_ids = torch.empty((0,), device=means2d_b.device, dtype=torch.int32)
            out_fast, tile_stop_counts = torch.ops.gsplat_metal_v6.render_fast_forward_state(
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                meta_i32,
                meta_f32,
                binned_ids,
                tile_counts,
                tile_offsets,
            )

        tile_size = int(meta_i32[4].item())
        tiles_x = int(meta_i32[3].item())
        tiles_per_image = int(meta_i32[10].item())
        max_fast_pairs = int(meta_i32[7].item())

        overflow_tile_ids = torch.empty((0,), device=means2d_b.device, dtype=torch.int32)
        overflow_tile_offsets = torch.zeros((1,), device=means2d_b.device, dtype=torch.int32)
        overflow_sorted_ids = torch.empty((0,), device=means2d_b.device, dtype=torch.int32)

        if enable_overflow_fallback:
            overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids = _gather_overflow_segments(
                tile_counts, tile_offsets, binned_ids, max_fast_pairs
            )
            if overflow_tile_ids.numel() > 0:
                overflow_tile_imgs = torch.ops.gsplat_metal_v6.render_overflow_forward(
                    means_flat,
                    conics_flat,
                    colors_flat,
                    opacities_flat,
                    meta_i32,
                    meta_f32,
                    overflow_tile_ids,
                    overflow_tile_offsets,
                    overflow_sorted_ids,
                )
                out = out_fast.clone()
                _scatter_tile_images_(out, overflow_tile_ids, overflow_tile_imgs, tiles_per_image, tiles_x, tile_size)
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
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            depths_b,
            meta_i32,
            meta_f32,
            active_tile_ids,
            tile_counts,
            tile_offsets,
            binned_ids,
            tile_stop_counts,
            overflow_tile_ids,
            overflow_tile_offsets,
            overflow_sorted_ids,
        )
        ctx.batch_size = B
        ctx.gaussians_per_batch = G
        ctx.tiles_per_image = tiles_per_image
        ctx.tiles_x = tiles_x
        ctx.tile_size = tile_size
        ctx.enable_overflow_fallback = enable_overflow_fallback
        ctx.use_active_tiles = bool(resolved_use_active_tiles)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            perm,
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            depths_b,
            meta_i32,
            meta_f32,
            active_tile_ids,
            tile_counts,
            tile_offsets,
            binned_ids,
            tile_stop_counts,
            overflow_tile_ids,
            overflow_tile_offsets,
            overflow_sorted_ids,
        ) = ctx.saved_tensors

        grad_fast = grad_out.contiguous()
        if ctx.use_active_tiles:
            g_means_flat, g_conics_flat, g_colors_flat, g_opacities_flat = torch.ops.gsplat_metal_v6.render_active_backward_saved(
                grad_fast,
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                meta_i32,
                meta_f32,
                active_tile_ids,
                tile_counts,
                tile_offsets,
                binned_ids,
                tile_stop_counts,
            )
        else:
            g_means_flat, g_conics_flat, g_colors_flat, g_opacities_flat = torch.ops.gsplat_metal_v6.render_fast_backward_saved(
                grad_fast,
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                meta_i32,
                meta_f32,
                tile_counts,
                tile_offsets,
                binned_ids,
                tile_stop_counts,
            )

        if ctx.enable_overflow_fallback and overflow_tile_ids.numel() > 0:
            grad_tiles = _gather_tile_images(grad_out.contiguous(), overflow_tile_ids, ctx.tiles_per_image, ctx.tiles_x, ctx.tile_size)
            go_means, go_conics, go_colors, go_opacities = torch.ops.gsplat_metal_v6.render_overflow_backward(
                grad_tiles,
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                meta_i32,
                meta_f32,
                overflow_tile_ids,
                overflow_tile_offsets,
                overflow_sorted_ids,
            )
            g_means_flat = g_means_flat + go_means
            g_conics_flat = g_conics_flat + go_conics
            g_colors_flat = g_colors_flat + go_colors
            g_opacities_flat = g_opacities_flat + go_opacities

        B = ctx.batch_size
        G = ctx.gaussians_per_batch
        g_means_b = _unsort_batched(g_means_flat.view(B, G, 2), perm)
        g_conics_b = _unsort_batched(g_conics_flat.view(B, G, 3), perm)
        g_colors_b = _unsort_batched(g_colors_flat.view(B, G, 3), perm)
        g_opacities_b = _unsort_batched(g_opacities_flat.view(B, G), perm)
        g_depths_b = torch.zeros_like(depths_b)
        return g_means_b, g_conics_b, g_colors_b, g_opacities_b, g_depths_b, None, None, None, None, None, None


def _rasterize_chunk_eval(
    means2d_b: Tensor,
    conics_b: Tensor,
    colors_b: Tensor,
    opacities_b: Tensor,
    depths_b: Tensor,
    config: RasterConfig,
) -> Tensor:
    B, G = means2d_b.shape[:2]
    meta_i32, meta_f32 = _make_meta(config, means2d_b.device, B, G)
    perm = torch.argsort(depths_b.detach(), dim=1, stable=True)
    means2d_s = _batched_gather_2d(means2d_b, perm).contiguous()
    conics_s = _batched_gather_2d(conics_b, perm).contiguous()
    colors_s = _batched_gather_2d(colors_b, perm).contiguous()
    opacities_s = _batched_gather_1d(opacities_b, perm).contiguous()

    means_flat = means2d_s.reshape(B * G, 2).contiguous()
    conics_flat = conics_s.reshape(B * G, 3).contiguous()
    colors_flat = colors_s.reshape(B * G, 3).contiguous()
    opacities_flat = opacities_s.reshape(B * G).contiguous()

    tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v6.bin(
        means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32
    )

    resolved_use_active_tiles = _resolve_active_tile_use(
        policy=_effective_active_policy(config),
        tile_counts=tile_counts,
        max_fast_pairs=int(meta_i32[7].item()),
        total_tiles=int(meta_i32[6].item()),
    )

    if resolved_use_active_tiles:
        active_tile_ids = _make_active_tile_ids(
            tile_counts,
            int(meta_i32[7].item()),
            sort_by_count=bool(config.sort_active_tiles_by_count),
        )
        out_fast = torch.ops.gsplat_metal_v6.render_active_forward_eval(
            means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32, active_tile_ids, tile_counts, tile_offsets, binned_ids
        )
    else:
        out_fast = torch.ops.gsplat_metal_v6.render_fast_forward_eval(
            means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32, tile_counts, tile_offsets, binned_ids
        )

    if config.enable_overflow_fallback:
        overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids = _gather_overflow_segments(
            tile_counts, tile_offsets, binned_ids, int(meta_i32[7].item())
        )
        if overflow_tile_ids.numel() > 0:
            overflow_tile_imgs = torch.ops.gsplat_metal_v6.render_overflow_forward(
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                meta_i32,
                meta_f32,
                overflow_tile_ids,
                overflow_tile_offsets,
                overflow_sorted_ids,
            )
            out = out_fast.clone()
            _scatter_tile_images_(out, overflow_tile_ids, overflow_tile_imgs, int(meta_i32[10].item()), int(meta_i32[3].item()), int(meta_i32[4].item()))
            return out
    else:
        if bool((tile_counts > int(meta_i32[7].item())).any().item()):
            raise RuntimeError(
                f"Tile overflow detected with max_fast_pairs={int(meta_i32[7].item())}. "
                "Enable overflow fallback or increase the runtime cap."
            )
    return out_fast


def _rasterize_batched(
    means2d_b: Tensor,
    conics_b: Tensor,
    colors_b: Tensor,
    opacities_b: Tensor,
    depths_b: Tensor,
    config: RasterConfig,
) -> Tensor:
    B, G = means2d_b.shape[:2]
    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size

    outs = []
    train_mode = _should_use_training_path(means2d_b, conics_b, colors_b, opacities_b)
    batch_slices = _choose_batch_slices(
        means2d_b,
        conics_b,
        colors_b,
        opacities_b,
        config,
        train_mode=train_mode,
        tiles_per_image=tiles_y * tiles_x,
    )
    for b0, b1 in batch_slices:
        m = means2d_b[b0:b1].contiguous()
        q = conics_b[b0:b1].contiguous()
        c = colors_b[b0:b1].contiguous()
        o = opacities_b[b0:b1].contiguous()
        d = depths_b[b0:b1].contiguous()

        if train_mode:
            meta_i32, meta_f32 = _make_meta(config, m.device, b1 - b0, G)
            outs.append(
                _RasterizeProjectedGaussiansV6.apply(
                    m,
                    q,
                    c,
                    o,
                    d,
                    meta_i32,
                    meta_f32,
                    bool(config.enable_overflow_fallback),
                    bool(config.use_active_tiles),
                    _active_policy_to_int(config.active_policy),
                    bool(config.sort_active_tiles_by_count),
                )
            )
        else:
            outs.append(_rasterize_chunk_eval(m, q, c, o, d, config))
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def rasterize_projected_gaussians(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: RasterConfig,
) -> Tensor:
    _check_inputs(means2d, conics, colors, opacities, depths)
    means2d_b, conics_b, colors_b, opacities_b, depths_b, was_batched = _normalize_inputs(
        means2d, conics, colors, opacities, depths
    )
    _runtime_validate(config)
    out = _rasterize_batched(means2d_b, conics_b, colors_b, opacities_b, depths_b, config)
    return out if was_batched else out[0]


@torch.no_grad()
def profile_projected_gaussians(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: RasterConfig,
    *,
    run_forward: bool = False,
    return_image: bool = False,
) -> Dict[str, Any]:
    _check_inputs(means2d, conics, colors, opacities, depths)
    means2d_b, conics_b, colors_b, opacities_b, depths_b, was_batched = _normalize_inputs(
        means2d, conics, colors, opacities, depths
    )
    B, G = means2d_b.shape[:2]
    _runtime_validate(config)

    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    tiles_per_image = tiles_y * tiles_x
    chunk_b = _choose_batch_chunk_size(config, B, G, tiles_per_image)
    batch_slices = _choose_batch_slices(
        means2d_b,
        conics_b,
        colors_b,
        opacities_b,
        config,
        train_mode=True,
        tiles_per_image=tiles_per_image,
    )

    all_tile_counts = []
    all_stop_counts = []
    all_active_counts = []
    all_dense_active = []
    resolved_active_chunks = []
    images = []

    for b0, b1 in batch_slices:
        m = means2d_b[b0:b1].contiguous()
        q = conics_b[b0:b1].contiguous()
        c = colors_b[b0:b1].contiguous()
        o = opacities_b[b0:b1].contiguous()
        d = depths_b[b0:b1].contiguous()

        perm = torch.argsort(d.detach(), dim=1, stable=True)
        m_s = _batched_gather_2d(m, perm).contiguous().reshape(-1, 2)
        q_s = _batched_gather_2d(q, perm).contiguous().reshape(-1, 3)
        c_s = _batched_gather_2d(c, perm).contiguous().reshape(-1, 3)
        o_s = _batched_gather_1d(o, perm).contiguous().reshape(-1)

        meta_i32, meta_f32 = _make_meta(config, means2d_b.device, b1 - b0, G)
        tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v6.bin(
            m_s, q_s, c_s, o_s, meta_i32, meta_f32
        )
        all_tile_counts.append(tile_counts.detach().cpu().to(torch.float32))

        active_policy = _effective_active_policy(config)
        resolved_use_active_tiles = _resolve_active_tile_use(
            policy=active_policy,
            tile_counts=tile_counts,
            max_fast_pairs=int(meta_i32[7].item()),
            total_tiles=int(meta_i32[6].item()),
        )
        resolved_active_chunks.append(bool(resolved_use_active_tiles))
        active_tile_ids = _make_active_tile_ids(
            tile_counts,
            int(meta_i32[7].item()),
            sort_by_count=bool(config.sort_active_tiles_by_count),
        )
        band_stats = _band_counts(tile_counts, active_tile_ids, config.stop_count_dense_threshold)
        all_active_counts.append(band_stats["active_tile_count"])
        all_dense_active.append(band_stats["dense_active_tile_count"])

        if run_forward or return_image:
            if return_image:
                chunk_img = _rasterize_chunk_eval(m, q, c, o, d, config)
                images.append(chunk_img)
            if resolved_use_active_tiles:
                _, stop_counts = torch.ops.gsplat_metal_v6.render_active_forward_state(
                    m_s, q_s, c_s, o_s, meta_i32, meta_f32, binned_ids, active_tile_ids, tile_counts, tile_offsets
                )
            else:
                _, stop_counts = torch.ops.gsplat_metal_v6.render_fast_forward_state(
                    m_s, q_s, c_s, o_s, meta_i32, meta_f32, binned_ids, tile_counts, tile_offsets
                )
            all_stop_counts.append(stop_counts.detach().cpu().to(torch.float32))

    counts_cpu = torch.cat(all_tile_counts, dim=0) if all_tile_counts else torch.zeros(0, dtype=torch.float32)
    stats: Dict[str, Any] = {
        "batch_size": int(B),
        "gaussians_per_batch": int(G),
        "height": int(config.height),
        "width": int(config.width),
        "tile_size": int(config.tile_size),
        "tiles": int(counts_cpu.numel()),
        "total_pairs": int(counts_cpu.sum().item()) if counts_cpu.numel() else 0,
        "mean_pairs_per_tile": float(counts_cpu.mean().item()) if counts_cpu.numel() else 0.0,
        "p95_pairs_per_tile": float(torch.quantile(counts_cpu, 0.95).item()) if counts_cpu.numel() else 0.0,
        "max_pairs_per_tile": int(counts_cpu.max().item()) if counts_cpu.numel() else 0,
        "overflow_tile_count": int((counts_cpu > int(config.max_fast_pairs)).sum().item()) if counts_cpu.numel() else 0,
        "active_tile_count": int(sum(all_active_counts)),
        "active_tile_fraction": float((counts_cpu > 0).sum().item() / max(counts_cpu.numel(), 1)) if counts_cpu.numel() else 0.0,
        "dense_active_tile_count": int(sum(all_dense_active)),
        "chosen_batch_chunk": int(chunk_b),
        "chosen_batch_chunks": [int(b1 - b0) for b0, b1 in batch_slices],
        "max_pairs_per_launch": int(config.max_pairs_per_launch),
        "active_policy": _effective_active_policy(config),
        "resolved_use_active_tiles": bool(any(resolved_active_chunks)),
        "resolved_active_chunks": [bool(x) for x in resolved_active_chunks],
        "stop_count_mode": config.stop_count_mode,
    }

    if all_stop_counts:
        stop_cpu = torch.cat(all_stop_counts, dim=0)
        denom = torch.clamp(counts_cpu, min=1.0)
        stop_ratio = torch.where(counts_cpu > 0, stop_cpu / denom, torch.zeros_like(stop_cpu))
        stats.update(
            {
                "mean_stop_count": float(stop_cpu.mean().item()),
                "p95_stop_count": float(torch.quantile(stop_cpu, 0.95).item()),
                "max_stop_count": int(stop_cpu.max().item()),
                "mean_stop_ratio": float(stop_ratio.mean().item()),
                "p95_stop_ratio": float(torch.quantile(stop_ratio, 0.95).item()),
            }
        )

    if return_image:
        out = torch.cat(images, dim=0) if len(images) > 1 else images[0]
        return {"image": out if was_batched else out[0], "stats": stats}
    return stats


class ProjectedGaussianRasterizer(torch.nn.Module):
    def __init__(self, config: RasterConfig):
        super().__init__()
        self.config = config

    def forward(self, means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor) -> Tensor:
        return rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, self.config)
