from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
    feature_cap: int


@dataclass(frozen=True)
class MetaBundle:
    gpu_i32: Tensor
    gpu_f32: Tensor
    host_i32: Tensor
    host_f32: Tensor


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


def get_runtime_shader_config() -> RuntimeShaderConfig:
    tile_size = _env_int("GSP_TILE_SIZE", 16)
    chunk_size = _env_int("GSP_CHUNK", 64)
    fast_cap = _env_int("GSP_FAST_CAP", 2048)
    feature_cap = _env_int("GSP_FEATURE_CAP", 64)
    if tile_size not in (8, 16, 32):
        raise ValueError(f"GSP_TILE_SIZE must be one of 8, 16, 32; got {tile_size}")
    threads = tile_size * tile_size
    if threads > 1024:
        raise ValueError(f"tile_size={tile_size} implies {threads} threads, which exceeds 1024")
    if chunk_size <= 0:
        raise ValueError("GSP_CHUNK must be positive")
    if fast_cap <= 0:
        raise ValueError("GSP_FAST_CAP must be positive")
    if feature_cap <= 0:
        raise ValueError("GSP_FEATURE_CAP must be positive")
    simdgroups = (threads + 31) // 32
    return RuntimeShaderConfig(
        tile_size=tile_size,
        threads=threads,
        chunk_size=chunk_size,
        fast_cap=fast_cap,
        simdgroups=simdgroups,
        feature_cap=feature_cap,
    )


@dataclass(frozen=True)
class RasterConfig:
    height: int
    width: int
    tile_size: int = 16
    max_fast_pairs: int = 2048
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1e-4
    # Either one value broadcast to all feature channels or exactly F values.
    background: Tuple[float, ...] = (0.0,)
    enable_overflow_fallback: bool = True
    batch_strategy: str = "auto"  # auto | flatten | serial
    batch_launch_limit_tiles: int = 262144
    batch_launch_limit_gaussians: int = 262144
    # Caller guarantees all per-splat inputs are already stably sorted by
    # nondecreasing depth per batch. This skips argsort/gather and backward unsort.
    inputs_sorted_by_depth: bool = False
    # Direct-tile kernels remain the default. Active scheduling mirrors
    # v6_refined: opt in explicitly or let auto choose sparse/overflow tails.
    use_active_tiles: Optional[bool] = None
    active_policy: str = "off"  # off | on | auto
    sort_active_tiles_by_count: bool = True
    active_sparse_fraction_threshold: float = 0.45
    active_dense_multiplier: float = 2.0
    stop_count_mode: str = "adaptive"  # always | never | adaptive
    stop_count_dense_threshold: int = 64


def _runtime_validate(config: RasterConfig, feature_dim: int | None = None) -> RuntimeShaderConfig:
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
    if not (0.0 <= float(config.active_sparse_fraction_threshold) <= 1.0):
        raise ValueError("active_sparse_fraction_threshold must be in [0,1]")
    if float(config.active_dense_multiplier) <= 0.0:
        raise ValueError("active_dense_multiplier must be positive")
    if config.stop_count_mode not in ("always", "never", "adaptive"):
        raise ValueError("stop_count_mode must be one of: always, never, adaptive")
    if config.stop_count_dense_threshold <= 0:
        raise ValueError("stop_count_dense_threshold must be positive")
    if feature_dim is not None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if feature_dim > rt.feature_cap:
            raise ValueError(
                f"feature_dim={feature_dim} exceeds runtime feature cap {rt.feature_cap}. "
                "Set GSP_FEATURE_CAP before importing/running the extension, or reduce colors.shape[-1]."
            )
    return rt


def _stop_mode_to_int(mode: str) -> int:
    return {"always": 0, "never": 1, "adaptive": 2}[mode]


def _background_for_feature_dim(config: RasterConfig, feature_dim: int, feature_cap: int) -> list[float]:
    background = tuple(float(v) for v in config.background)
    if len(background) == 1:
        values = [background[0]] * feature_dim
    elif len(background) == feature_dim:
        values = list(background)
    else:
        raise ValueError(
            f"RasterConfig.background must have length 1 or feature_dim={feature_dim}; got {len(background)}"
        )
    return values + [0.0] * (feature_cap - feature_dim)


def _background_zero_flag(background: list[float], feature_dim: int) -> int:
    return 2 if all(float(value) == 0.0 for value in background[:feature_dim]) else 0


def _make_meta(
    config: RasterConfig,
    device: torch.device,
    batch_size: int,
    gaussians_per_batch: int,
    feature_dim: int,
) -> MetaBundle:
    rt = _runtime_validate(config, feature_dim)
    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    tiles_per_image = tiles_y * tiles_x
    total_tiles = batch_size * tiles_per_image
    total_gaussians = batch_size * gaussians_per_batch
    bg = _background_for_feature_dim(config, feature_dim, rt.feature_cap)
    i32_values = [
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
        feature_dim,
        _stop_mode_to_int(config.stop_count_mode),
        int(config.stop_count_dense_threshold),
        _background_zero_flag(bg, feature_dim),
    ]
    f32_values = [
        float(config.alpha_threshold),
        float(config.transmittance_threshold),
        1e-8,
        0.99,
        *bg,
    ]
    return MetaBundle(
        gpu_i32=torch.tensor(i32_values, device=device, dtype=torch.int32),
        gpu_f32=torch.tensor(f32_values, device=device, dtype=torch.float32),
        host_i32=torch.tensor(i32_values, device="cpu", dtype=torch.int32),
        host_f32=torch.tensor(f32_values, device="cpu", dtype=torch.float32),
    )


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
    tensors = {
        "means2d": means2d,
        "conics": conics,
        "colors": colors,
        "opacities": opacities,
        "depths": depths,
    }
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError("means2d/conics/colors/opacities/depths must be on the same device")
    if means2d.device.type != "mps":
        raise ValueError("v12a_fused_colorize_l1_no_norm Metal rasterizer inputs must be on MPS")
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
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
    if colors.shape[-1] <= 0:
        raise ValueError("colors/features must have a positive last dim")
    if means2d.shape[:-1] != conics.shape[:-1] or means2d.shape[:-1] != colors.shape[:-1]:
        raise ValueError("means2d/conics/colors batch/G dimensions must match")
    if means2d.shape[:-1] != opacities.shape or means2d.shape[:-1] != depths.shape:
        raise ValueError("means2d/opacities/depths batch/G dimensions must match")


def _batched_gather_2d(x: Tensor, perm: Tensor) -> Tensor:
    return x.gather(1, perm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _batched_gather_1d(x: Tensor, perm: Tensor) -> Tensor:
    return x.gather(1, perm)


def _maybe_sort_inputs_by_depth(
    means2d_b: Tensor,
    conics_b: Tensor,
    colors_b: Tensor,
    opacities_b: Tensor,
    depths_b: Tensor,
    *,
    inputs_sorted_by_depth: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if inputs_sorted_by_depth:
        empty_perm = torch.empty((0,), device=depths_b.device, dtype=torch.int64)
        return (
            empty_perm,
            means2d_b.contiguous(),
            conics_b.contiguous(),
            colors_b.contiguous(),
            opacities_b.contiguous(),
        )
    perm = torch.argsort(depths_b.detach(), dim=1, stable=True)
    return (
        perm,
        _batched_gather_2d(means2d_b, perm).contiguous(),
        _batched_gather_2d(conics_b, perm).contiguous(),
        _batched_gather_2d(colors_b, perm).contiguous(),
        _batched_gather_1d(opacities_b, perm).contiguous(),
    )


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


def _scatter_tile_scalars_(base: Tensor, tile_ids: Tensor, tile_values: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> None:
    if tile_ids.numel() == 0:
        return
    _, H, W = base.shape
    for i, tile_id in enumerate(tile_ids.tolist()):
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        base[b, y0:y1, x0:x1] = tile_values[i, : y1 - y0, : x1 - x0]


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


def _gather_tile_scalars(img: Tensor, tile_ids: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> Tensor:
    if tile_ids.numel() == 0:
        return torch.empty((0, tile_size, tile_size), device=img.device, dtype=img.dtype)
    out = torch.zeros((tile_ids.numel(), tile_size, tile_size), device=img.device, dtype=img.dtype)
    _, H, W = img.shape
    for i, tile_id in enumerate(tile_ids.tolist()):
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        out[i, : y1 - y0, : x1 - x0] = img[b, y0:y1, x0:x1]
    return out


def _zero_tile_images_(img: Tensor, tile_ids: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> None:
    if tile_ids.numel() == 0:
        return
    _, H, W = img.shape[:3]
    for tile_id in tile_ids.tolist():
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        img[b, y0:y1, x0:x1, :] = 0


def _zero_tile_scalars_(img: Tensor, tile_ids: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> None:
    if tile_ids.numel() == 0:
        return
    _, H, W = img.shape
    for tile_id in tile_ids.tolist():
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        img[b, y0:y1, x0:x1] = 0


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


def _resolve_active_tile_mode(
    tile_counts: Tensor,
    max_fast_pairs: int,
    *,
    use_active_tiles_override: Optional[bool],
    active_policy: str,
    sparse_fraction_threshold: float,
    dense_multiplier: float,
) -> tuple[bool, dict[str, Any]]:
    total_tiles = int(tile_counts.numel())
    if total_tiles == 0:
        return False, {
            "active_tile_count": 0,
            "active_tile_fraction": 0.0,
            "overflow_tile_count": 0,
            "max_pairs_per_tile": 0,
            "selected_active_policy": "off",
            "selected_use_active_tiles": False,
            "selected_active_reason": "no_tiles",
        }

    active_tile_count = int((tile_counts > 0).sum().item())
    active_tile_fraction = float(active_tile_count) / float(max(total_tiles, 1))
    overflow_tile_count = int((tile_counts > int(max_fast_pairs)).sum().item())
    max_pairs_per_tile = int(tile_counts.max().item()) if tile_counts.numel() else 0

    if use_active_tiles_override is not None:
        use_active_tiles = bool(use_active_tiles_override)
        selected_active_policy = "legacy"
        selected_active_reason = "override_true" if use_active_tiles else "override_false"
    elif active_policy == "on":
        use_active_tiles = True
        selected_active_policy = "on"
        selected_active_reason = "forced_on"
    elif active_policy == "off":
        use_active_tiles = False
        selected_active_policy = "off"
        selected_active_reason = "forced_off"
    else:
        sparse = active_tile_fraction < float(sparse_fraction_threshold)
        overflow = overflow_tile_count > 0
        dense_tail = max_pairs_per_tile > int(float(dense_multiplier) * int(max_fast_pairs))
        use_active_tiles = bool(sparse or overflow or dense_tail)
        selected_active_policy = "auto"
        reasons = []
        if sparse:
            reasons.append("sparse")
        if overflow:
            reasons.append("overflow")
        if dense_tail:
            reasons.append("dense_tail")
        selected_active_reason = "+".join(reasons) if reasons else "uniform_dense"

    return bool(use_active_tiles), {
        "active_tile_count": active_tile_count,
        "active_tile_fraction": active_tile_fraction,
        "overflow_tile_count": overflow_tile_count,
        "max_pairs_per_tile": max_pairs_per_tile,
        "selected_active_policy": selected_active_policy,
        "selected_use_active_tiles": bool(use_active_tiles),
        "selected_active_reason": selected_active_reason,
    }


class _RasterizeProjectedGaussiansV6RefinedFeatures(torch.autograd.Function):
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
        meta_host_i32: Tensor,
        meta_host_f32: Tensor,
        enable_overflow_fallback: bool,
        inputs_sorted_by_depth: bool,
        use_active_tiles_override: Optional[bool],
        active_policy: str,
        active_sparse_fraction_threshold: float,
        active_dense_multiplier: float,
        sort_active_tiles_by_count: bool,
    ) -> tuple[Tensor, Tensor]:
        if not hasattr(torch.ops, "gsplat_metal_v12a_fused_colorize_l1_no_norm"):
            raise RuntimeError("gsplat_metal_v12a_fused_colorize_l1_no_norm custom ops not found. Build the extension first.")

        B, G = means2d_b.shape[:2]
        F = colors_b.shape[-1]
        perm, means2d_s, conics_s, colors_s, opacities_s = _maybe_sort_inputs_by_depth(
            means2d_b,
            conics_b,
            colors_b,
            opacities_b,
            depths_b,
            inputs_sorted_by_depth=inputs_sorted_by_depth,
        )

        means_flat = means2d_s.reshape(B * G, 2).contiguous()
        conics_flat = conics_s.reshape(B * G, 3).contiguous()
        colors_flat = colors_s.reshape(B * G, F).contiguous()
        opacities_flat = opacities_s.reshape(B * G).contiguous()

        tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.bin(
            means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32, meta_host_i32, meta_host_f32
        )

        use_active_tiles, _ = _resolve_active_tile_mode(
            tile_counts,
            int(meta_host_i32[7].item()),
            use_active_tiles_override=use_active_tiles_override,
            active_policy=active_policy,
            sparse_fraction_threshold=float(active_sparse_fraction_threshold),
            dense_multiplier=float(active_dense_multiplier),
        )
        if use_active_tiles:
            active_tile_ids = _make_active_tile_ids(
                tile_counts, int(meta_host_i32[7].item()), sort_by_count=bool(sort_active_tiles_by_count)
            )
            out_fast, alpha_fast, tile_stop_counts = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.render_active_forward_state(
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                meta_i32,
                meta_f32,
                meta_host_i32,
                meta_host_f32,
                binned_ids,
                active_tile_ids,
                tile_counts,
                tile_offsets,
            )
        else:
            active_tile_ids = torch.empty((0,), device=means2d_b.device, dtype=torch.int32)
            out_fast, alpha_fast, tile_stop_counts = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.render_fast_forward_state(
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                meta_i32,
                meta_f32,
                meta_host_i32,
                meta_host_f32,
                binned_ids,
                tile_counts,
                tile_offsets,
            )

        tile_size = int(meta_host_i32[4].item())
        tiles_x = int(meta_host_i32[3].item())
        tiles_per_image = int(meta_host_i32[10].item())
        max_fast_pairs = int(meta_host_i32[7].item())

        overflow_tile_ids = torch.empty((0,), device=means2d_b.device, dtype=torch.int32)
        overflow_tile_offsets = torch.zeros((1,), device=means2d_b.device, dtype=torch.int32)
        overflow_sorted_ids = torch.empty((0,), device=means2d_b.device, dtype=torch.int32)

        if bool((tile_counts > max_fast_pairs).any().item()):
            raise RuntimeError(
                f"Tile overflow detected with max_fast_pairs={max_fast_pairs}. "
                "The fixedbin fork only supports no-overflow fast-path rows; "
                "use a non-fixedbin fork or increase the runtime cap."
            )
        out = out_fast
        alpha = alpha_fast

        ctx.save_for_backward(
            perm,
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            depths_b,
            meta_i32,
            meta_f32,
            meta_host_i32,
            meta_host_f32,
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
        ctx.feature_dim = F
        ctx.tiles_per_image = tiles_per_image
        ctx.tiles_x = tiles_x
        ctx.tile_size = tile_size
        ctx.enable_overflow_fallback = enable_overflow_fallback
        ctx.inputs_sorted_by_depth = inputs_sorted_by_depth
        ctx.use_active_tiles = bool(use_active_tiles)
        return out, alpha

    @staticmethod
    def backward(ctx, grad_features: Tensor | None, grad_alpha: Tensor | None):
        (
            perm,
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            depths_b,
            meta_i32,
            meta_f32,
            meta_host_i32,
            meta_host_f32,
            active_tile_ids,
            tile_counts,
            tile_offsets,
            binned_ids,
            tile_stop_counts,
            overflow_tile_ids,
            overflow_tile_offsets,
            overflow_sorted_ids,
        ) = ctx.saved_tensors

        if grad_features is None:
            grad_features = torch.zeros(
                (ctx.batch_size, int(meta_host_i32[0].item()), int(meta_host_i32[1].item()), ctx.feature_dim),
                device=means_flat.device,
                dtype=means_flat.dtype,
            )
        if grad_alpha is None:
            grad_alpha = torch.zeros(
                (ctx.batch_size, int(meta_host_i32[0].item()), int(meta_host_i32[1].item())),
                device=means_flat.device,
                dtype=means_flat.dtype,
            )

        has_overflow = ctx.enable_overflow_fallback and overflow_tile_ids.numel() > 0
        needs_color_grad = bool(ctx.needs_input_grad[2])
        backward_meta_i32 = meta_i32
        backward_meta_host_i32 = meta_host_i32
        if not needs_color_grad:
            backward_meta_i32 = meta_i32.clone()
            backward_meta_i32[14] = int(backward_meta_i32[14].item()) | 1
            backward_meta_host_i32 = meta_host_i32.clone()
            backward_meta_host_i32[14] = int(backward_meta_host_i32[14].item()) | 1
        grad_fast = grad_features.contiguous()
        grad_alpha_fast = grad_alpha.contiguous()
        if has_overflow:
            grad_fast = grad_fast.clone()
            grad_alpha_fast = grad_alpha_fast.clone()
            _zero_tile_images_(grad_fast, overflow_tile_ids, ctx.tiles_per_image, ctx.tiles_x, ctx.tile_size)
            _zero_tile_scalars_(grad_alpha_fast, overflow_tile_ids, ctx.tiles_per_image, ctx.tiles_x, ctx.tile_size)

        if ctx.use_active_tiles:
            g_means_flat, g_conics_flat, g_colors_flat, g_opacities_flat = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.render_active_backward_saved(
                grad_fast,
                grad_alpha_fast,
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                backward_meta_i32,
                meta_f32,
                backward_meta_host_i32,
                meta_host_f32,
                active_tile_ids,
                tile_counts,
                tile_offsets,
                binned_ids,
                tile_stop_counts,
            )
        else:
            g_means_flat, g_conics_flat, g_colors_flat, g_opacities_flat = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.render_fast_backward_saved(
                grad_fast,
                grad_alpha_fast,
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                backward_meta_i32,
                meta_f32,
                backward_meta_host_i32,
                meta_host_f32,
                tile_counts,
                tile_offsets,
                binned_ids,
                tile_stop_counts,
            )

        if has_overflow:
            grad_tiles = _gather_tile_images(grad_features.contiguous(), overflow_tile_ids, ctx.tiles_per_image, ctx.tiles_x, ctx.tile_size)
            grad_alpha_tiles = _gather_tile_scalars(grad_alpha.contiguous(), overflow_tile_ids, ctx.tiles_per_image, ctx.tiles_x, ctx.tile_size)
            go_means, go_conics, go_colors, go_opacities = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.render_overflow_backward(
                grad_tiles,
                grad_alpha_tiles,
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                backward_meta_i32,
                meta_f32,
                backward_meta_host_i32,
                meta_host_f32,
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
        F = ctx.feature_dim
        if ctx.inputs_sorted_by_depth:
            g_means_b = g_means_flat.view(B, G, 2)
            g_conics_b = g_conics_flat.view(B, G, 3)
            g_colors_b = g_colors_flat.view(B, G, F) if needs_color_grad else None
            g_opacities_b = g_opacities_flat.view(B, G)
        else:
            g_means_b = _unsort_batched(g_means_flat.view(B, G, 2), perm)
            g_conics_b = _unsort_batched(g_conics_flat.view(B, G, 3), perm)
            g_colors_b = _unsort_batched(g_colors_flat.view(B, G, F), perm) if needs_color_grad else None
            g_opacities_b = _unsort_batched(g_opacities_flat.view(B, G), perm)
        g_depths_b = torch.zeros_like(depths_b)
        return (
            g_means_b,
            g_conics_b,
            g_colors_b,
            g_opacities_b,
            g_depths_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _rasterize_chunk_eval(
    means2d_b: Tensor,
    conics_b: Tensor,
    colors_b: Tensor,
    opacities_b: Tensor,
    depths_b: Tensor,
    config: RasterConfig,
) -> tuple[Tensor, Tensor]:
    B, G = means2d_b.shape[:2]
    F = colors_b.shape[-1]
    meta = _make_meta(config, means2d_b.device, B, G, F)
    _, means2d_s, conics_s, colors_s, opacities_s = _maybe_sort_inputs_by_depth(
        means2d_b,
        conics_b,
        colors_b,
        opacities_b,
        depths_b,
        inputs_sorted_by_depth=bool(config.inputs_sorted_by_depth),
    )

    means_flat = means2d_s.reshape(B * G, 2).contiguous()
    conics_flat = conics_s.reshape(B * G, 3).contiguous()
    colors_flat = colors_s.reshape(B * G, F).contiguous()
    opacities_flat = opacities_s.reshape(B * G).contiguous()

    tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.bin(
        means_flat, conics_flat, colors_flat, opacities_flat, meta.gpu_i32, meta.gpu_f32, meta.host_i32, meta.host_f32
    )

    use_active_tiles, _ = _resolve_active_tile_mode(
        tile_counts,
        int(meta.host_i32[7].item()),
        use_active_tiles_override=config.use_active_tiles,
        active_policy=config.active_policy,
        sparse_fraction_threshold=float(config.active_sparse_fraction_threshold),
        dense_multiplier=float(config.active_dense_multiplier),
    )
    if use_active_tiles:
        active_tile_ids = _make_active_tile_ids(
            tile_counts, int(meta.host_i32[7].item()), sort_by_count=bool(config.sort_active_tiles_by_count)
        )
        out_fast, alpha_fast = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.render_active_forward_eval(
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            meta.gpu_i32,
            meta.gpu_f32,
            meta.host_i32,
            meta.host_f32,
            active_tile_ids,
            tile_counts,
            tile_offsets,
            binned_ids,
        )
    else:
        out_fast, alpha_fast = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.render_fast_forward_eval(
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            meta.gpu_i32,
            meta.gpu_f32,
            meta.host_i32,
            meta.host_f32,
            tile_counts,
            tile_offsets,
            binned_ids,
        )

    if bool((tile_counts > int(meta.host_i32[7].item())).any().item()):
        raise RuntimeError(
            f"Tile overflow detected with max_fast_pairs={int(meta.host_i32[7].item())}. "
            "The fixedbin fork only supports no-overflow fast-path rows; "
            "use a non-fixedbin fork or increase the runtime cap."
        )
    return out_fast, alpha_fast


def _rasterize_batched(
    means2d_b: Tensor,
    conics_b: Tensor,
    colors_b: Tensor,
    opacities_b: Tensor,
    depths_b: Tensor,
    config: RasterConfig,
) -> tuple[Tensor, Tensor]:
    B, G = means2d_b.shape[:2]
    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    chunk_b = _choose_batch_chunk_size(config, B, G, tiles_y * tiles_x)

    outs = []
    alphas = []
    train_mode = _should_use_training_path(means2d_b, conics_b, colors_b, opacities_b)
    for b0 in range(0, B, chunk_b):
        b1 = min(B, b0 + chunk_b)
        m = means2d_b[b0:b1].contiguous()
        q = conics_b[b0:b1].contiguous()
        c = colors_b[b0:b1].contiguous()
        o = opacities_b[b0:b1].contiguous()
        d = depths_b[b0:b1].contiguous()

        if train_mode:
            meta = _make_meta(config, m.device, b1 - b0, G, c.shape[-1])
            chunk_out, chunk_alpha = (
                _RasterizeProjectedGaussiansV6RefinedFeatures.apply(
                    m,
                    q,
                    c,
                    o,
                    d,
                    meta.gpu_i32,
                    meta.gpu_f32,
                    meta.host_i32,
                    meta.host_f32,
                    bool(config.enable_overflow_fallback),
                    bool(config.inputs_sorted_by_depth),
                    config.use_active_tiles,
                    config.active_policy,
                    float(config.active_sparse_fraction_threshold),
                    float(config.active_dense_multiplier),
                    bool(config.sort_active_tiles_by_count),
                )
            )
            outs.append(chunk_out)
            alphas.append(chunk_alpha)
        else:
            chunk_out, chunk_alpha = _rasterize_chunk_eval(m, q, c, o, d, config)
            outs.append(chunk_out)
            alphas.append(chunk_alpha)
    out = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
    alpha = torch.cat(alphas, dim=0) if len(alphas) > 1 else alphas[0]
    return out, alpha


def rasterize_projected_gaussians(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: RasterConfig,
) -> tuple[Tensor, Tensor]:
    _check_inputs(means2d, conics, colors, opacities, depths)
    means2d_b, conics_b, colors_b, opacities_b, depths_b, was_batched = _normalize_inputs(
        means2d, conics, colors, opacities, depths
    )
    _runtime_validate(config, colors_b.shape[-1])
    out, alpha = _rasterize_batched(means2d_b, conics_b, colors_b, opacities_b, depths_b, config)
    return (out, alpha) if was_batched else (out[0], alpha[0])


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
    F = colors_b.shape[-1]
    _runtime_validate(config, F)

    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    tiles_per_image = tiles_y * tiles_x
    chunk_b = _choose_batch_chunk_size(config, B, G, tiles_per_image)

    all_tile_counts = []
    all_stop_counts = []
    all_active_counts = []
    all_dense_active = []
    all_selected_use_active = []
    all_active_fractions = []
    all_selected_reasons = []
    images = []

    for b0 in range(0, B, chunk_b):
        b1 = min(B, b0 + chunk_b)
        m = means2d_b[b0:b1].contiguous()
        q = conics_b[b0:b1].contiguous()
        c = colors_b[b0:b1].contiguous()
        o = opacities_b[b0:b1].contiguous()
        d = depths_b[b0:b1].contiguous()

        _, m_s_b, q_s_b, c_s_b, o_s_b = _maybe_sort_inputs_by_depth(
            m,
            q,
            c,
            o,
            d,
            inputs_sorted_by_depth=bool(config.inputs_sorted_by_depth),
        )
        m_s = m_s_b.reshape(-1, 2)
        q_s = q_s_b.reshape(-1, 3)
        c_s = c_s_b.reshape(-1, F)
        o_s = o_s_b.reshape(-1)

        meta = _make_meta(config, means2d_b.device, b1 - b0, G, F)
        tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.bin(
            m_s, q_s, c_s, o_s, meta.gpu_i32, meta.gpu_f32, meta.host_i32, meta.host_f32
        )
        all_tile_counts.append(tile_counts.detach().cpu().to(torch.float32))

        use_active_tiles, mode_stats = _resolve_active_tile_mode(
            tile_counts,
            int(meta.host_i32[7].item()),
            use_active_tiles_override=config.use_active_tiles,
            active_policy=config.active_policy,
            sparse_fraction_threshold=float(config.active_sparse_fraction_threshold),
            dense_multiplier=float(config.active_dense_multiplier),
        )
        if use_active_tiles:
            active_tile_ids = _make_active_tile_ids(
                tile_counts, int(meta.host_i32[7].item()), sort_by_count=bool(config.sort_active_tiles_by_count)
            )
        else:
            active_tile_ids = torch.empty((0,), device=tile_counts.device, dtype=torch.int32)
        band_stats = (
            _band_counts(tile_counts, active_tile_ids, config.stop_count_dense_threshold)
            if use_active_tiles
            else {
                "active_tile_count": mode_stats["active_tile_count"],
                "dense_active_tile_count": 0,
                "light_active_tile_count": mode_stats["active_tile_count"],
            }
        )
        all_active_counts.append(band_stats["active_tile_count"])
        all_dense_active.append(band_stats["dense_active_tile_count"])
        all_selected_use_active.append(bool(mode_stats["selected_use_active_tiles"]))
        all_active_fractions.append(float(mode_stats["active_tile_fraction"]))
        all_selected_reasons.append(str(mode_stats["selected_active_reason"]))

        if run_forward or return_image:
            if return_image:
                chunk_img, _chunk_alpha = _rasterize_chunk_eval(m, q, c, o, d, config)
                images.append(chunk_img)
            if use_active_tiles:
                _, _alpha, stop_counts = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.render_active_forward_state(
                    m_s,
                    q_s,
                    c_s,
                    o_s,
                    meta.gpu_i32,
                    meta.gpu_f32,
                    meta.host_i32,
                    meta.host_f32,
                    binned_ids,
                    active_tile_ids,
                    tile_counts,
                    tile_offsets,
                )
            else:
                _, _alpha, stop_counts = torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm.render_fast_forward_state(
                    m_s,
                    q_s,
                    c_s,
                    o_s,
                    meta.gpu_i32,
                    meta.gpu_f32,
                    meta.host_i32,
                    meta.host_f32,
                    binned_ids,
                    tile_counts,
                    tile_offsets,
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
        "chosen_batch_chunk": int(chunk_b),
        "active_tile_count": int(sum(all_active_counts)),
        "dense_active_tile_count": int(sum(all_dense_active)),
        "stop_count_mode": config.stop_count_mode,
        "active_policy": config.active_policy,
        "use_active_tiles_override": None if config.use_active_tiles is None else bool(config.use_active_tiles),
    }

    stats.update(
        {
            "mean_active_tile_fraction": float(sum(all_active_fractions) / len(all_active_fractions)) if all_active_fractions else 0.0,
            "selected_use_active_tiles": bool(any(all_selected_use_active)) if all_selected_use_active else False,
            "selected_active_reason": ",".join(sorted(set(all_selected_reasons))) if all_selected_reasons else "",
        }
    )

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
