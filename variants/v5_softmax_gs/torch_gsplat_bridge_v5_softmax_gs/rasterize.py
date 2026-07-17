from __future__ import annotations

import os
from dataclasses import dataclass, replace
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
    # Caller guarantees all per-splat inputs are already stably sorted by
    # nondecreasing depth per batch. This skips argsort/gather and backward unsort.
    inputs_sorted_by_depth: bool = False
    softmax_gs_enabled: bool = False
    softmax_gs_beta: float = 0.0
    softmax_gs_gamma: float = 0.0
    softmax_gs_tape_k: int = 0


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
    return rt


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
            1 if config.softmax_gs_enabled else 0,
            int(config.softmax_gs_tape_k),
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
            float(config.softmax_gs_beta),
            float(config.softmax_gs_gamma),
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
        raise ValueError("v5 Metal rasterizer inputs must be on MPS")
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
    if colors.shape[-1] != 3:
        raise ValueError("colors must have last dim = 3")
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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if inputs_sorted_by_depth:
        empty_perm = torch.empty((0,), device=depths_b.device, dtype=torch.int64)
        return (
            empty_perm,
            means2d_b.contiguous(),
            conics_b.contiguous(),
            colors_b.contiguous(),
            opacities_b.contiguous(),
            depths_b.contiguous(),
        )
    perm = torch.argsort(depths_b.detach(), dim=1, stable=True)
    return (
        perm,
        _batched_gather_2d(means2d_b, perm).contiguous(),
        _batched_gather_2d(conics_b, perm).contiguous(),
        _batched_gather_2d(colors_b, perm).contiguous(),
        _batched_gather_1d(opacities_b, perm).contiguous(),
        _batched_gather_1d(depths_b, perm).contiguous(),
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


def _scatter_tile_tensor_(base: Tensor, tile_ids: Tensor, tile_values: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> None:
    if tile_ids.numel() == 0:
        return
    _, H, W = base.shape[:3]
    for i, tile_id in enumerate(tile_ids.tolist()):
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        base[b, y0:y1, x0:x1, ...] = tile_values[i, : y1 - y0, : x1 - x0, ...]


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


def _zero_tile_images_(img: Tensor, tile_ids: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> None:
    if tile_ids.numel() == 0:
        return
    _, H, W = img.shape[:3]
    for tile_id in tile_ids.tolist():
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        img[b, y0:y1, x0:x1, :] = 0


def _should_use_training_path(means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor) -> bool:
    if not torch.is_grad_enabled():
        return False
    return bool(means2d.requires_grad or conics.requires_grad or colors.requires_grad or opacities.requires_grad)


def _rescale_softmax_pair_for_transmittance(
    past_absorbance: Tensor,
    current_absorbance: Tensor,
    target_transmittance: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    pair_product = past_absorbance * current_absorbance
    pair_sum = past_absorbance + current_absorbance
    discriminant = torch.clamp(
        pair_sum.square() - 4.0 * (1.0 - target_transmittance) * pair_product,
        min=eps,
    )
    scale = 2.0 * (1.0 - target_transmittance) / torch.clamp(pair_sum + torch.sqrt(discriminant), min=eps)
    scale = torch.where(pair_product > eps, scale, torch.ones_like(scale))
    return past_absorbance * scale, current_absorbance * scale


def _rasterize_softmax_gs_torch_train(
    means2d_b: Tensor,
    conics_b: Tensor,
    colors_b: Tensor,
    opacities_b: Tensor,
    depths_b: Tensor,
    config: RasterConfig,
) -> Tensor:
    """Differentiable training fallback for Softmax-GS.

    The Metal fork owns the fast forward/eval path. This Torch route exists to
    unblock tiny trainable dynamic-GS smokes while the native backward/tape is
    still under construction.
    """

    B, G = means2d_b.shape[:2]
    _perm, means2d_s, conics_s, colors_s, opacities_s, depths_s = _maybe_sort_inputs_by_depth(
        means2d_b,
        conics_b,
        colors_b,
        opacities_b,
        depths_b,
        inputs_sorted_by_depth=bool(config.inputs_sorted_by_depth),
    )
    device = means2d_b.device
    dtype = means2d_b.dtype
    ys = torch.arange(config.height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(config.width, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    px = grid_x.unsqueeze(0)
    py = grid_y.unsqueeze(0)

    bg = torch.tensor(config.background, device=device, dtype=dtype).view(1, 1, 1, 3)
    eps = 1.0e-8
    beta = torch.as_tensor(float(config.softmax_gs_beta), device=device, dtype=dtype)
    gamma = torch.as_tensor(max(float(config.softmax_gs_gamma), 0.0), device=device, dtype=dtype)
    alpha_threshold = float(config.alpha_threshold)
    trans_threshold = float(config.transmittance_threshold)
    max_alpha = 0.99

    accum = torch.zeros((B, config.height, config.width, 3), device=device, dtype=dtype)
    transmittance = torch.ones((B, config.height, config.width), device=device, dtype=dtype)
    past_depth = torch.zeros_like(transmittance)
    past_power = torch.zeros_like(transmittance)

    for j in range(G):
        mean = means2d_s[:, j, :]
        conic = conics_s[:, j, :]
        dx = px - mean[:, 0].view(B, 1, 1)
        dy = py - mean[:, 1].view(B, 1, 1)
        power = -0.5 * (
            conic[:, 0].view(B, 1, 1) * dx.square()
            + 2.0 * conic[:, 1].view(B, 1, 1) * dx * dy
            + conic[:, 2].view(B, 1, 1) * dy.square()
        )
        raw_alpha = opacities_s[:, j].view(B, 1, 1) * torch.exp(power)
        alpha = torch.minimum(raw_alpha, torch.full_like(raw_alpha, max_alpha))
        active = (power <= 0.0) & (alpha >= alpha_threshold) & (transmittance > trans_threshold)
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))

        has_past = active & (transmittance < 1.0 - eps)
        if bool(config.softmax_gs_enabled):
            original_transmittance = transmittance * (1.0 - alpha)
            past_absorbance = 1.0 - transmittance
            current_weight = torch.sigmoid(beta * (power - past_power))
            soft_current = current_weight * alpha
            soft_past = (1.0 - current_weight) * past_absorbance
            soft_denom = torch.clamp(soft_past + soft_current, min=eps)
            tilde_past = soft_past * (1.0 - original_transmittance) / soft_denom
            tilde_current = soft_current * (1.0 - original_transmittance) / torch.clamp(
                soft_current + soft_past * original_transmittance,
                min=eps,
            )
            depth = depths_s[:, j].view(B, 1, 1)
            decay = torch.exp(-gamma * torch.abs(depth - past_depth))
            effective_past = decay * tilde_past + (1.0 - decay) * past_absorbance
            alpha_soft = decay * tilde_current + (1.0 - decay) * alpha
            effective_past, alpha_soft = _rescale_softmax_pair_for_transmittance(
                effective_past,
                alpha_soft,
                original_transmittance,
                eps,
            )
            trans_soft = 1.0 - effective_past
            accum_soft = accum * (effective_past / torch.clamp(past_absorbance, min=eps)).unsqueeze(-1)
            transmittance = torch.where(has_past, trans_soft, transmittance)
            accum = torch.where(has_past.unsqueeze(-1), accum_soft, accum)
            alpha = torch.where(has_past, alpha_soft, alpha)

        weight = transmittance * alpha
        accum = accum + weight.unsqueeze(-1) * colors_s[:, j, :].view(B, 1, 1, 3)
        denom = torch.clamp(1.0 - transmittance + weight, min=eps)
        depth_j = depths_s[:, j].view(B, 1, 1)
        past_depth_next = (past_depth * (1.0 - transmittance) + depth_j * weight) / denom
        past_power_next = (past_power * (1.0 - transmittance) + power * weight) / denom
        past_depth = torch.where(active, past_depth_next, past_depth)
        past_power = torch.where(active, past_power_next, past_power)
        transmittance = transmittance * (1.0 - alpha)

    return accum + transmittance.unsqueeze(-1) * bg


class _RasterizeProjectedGaussiansSoftmaxGSTorchBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d_b: Tensor,
        conics_b: Tensor,
        colors_b: Tensor,
        opacities_b: Tensor,
        depths_b: Tensor,
        config: RasterConfig,
    ) -> Tensor:
        ctx.config = config
        ctx.save_for_backward(means2d_b, conics_b, colors_b, opacities_b, depths_b)
        return _rasterize_chunk_eval(means2d_b, conics_b, colors_b, opacities_b, depths_b, config)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        means2d_b, conics_b, colors_b, opacities_b, depths_b = ctx.saved_tensors
        needs = ctx.needs_input_grad
        m = means2d_b.detach().requires_grad_(needs[0])
        q = conics_b.detach().requires_grad_(needs[1])
        c = colors_b.detach().requires_grad_(needs[2])
        o = opacities_b.detach().requires_grad_(needs[3])
        d = depths_b.detach().requires_grad_(needs[4])
        grad_inputs = (m, q, c, o, d)
        with torch.enable_grad():
            out = _rasterize_softmax_gs_torch_train(m, q, c, o, d, ctx.config)
        grads = torch.autograd.grad(
            out,
            grad_inputs,
            grad_out.contiguous(),
            allow_unused=True,
        )
        filled_grads = tuple(torch.zeros_like(tensor) if grad is None else grad for tensor, grad in zip(grad_inputs, grads))
        return (*filled_grads, None)


def _choose_batch_chunk_size(config: RasterConfig, batch_size: int, gaussians_per_batch: int, tiles_per_image: int) -> int:
    if config.batch_strategy == "flatten":
        return batch_size
    if config.batch_strategy == "serial":
        return 1
    by_tiles = max(1, config.batch_launch_limit_tiles // max(tiles_per_image, 1))
    by_gaussians = max(1, config.batch_launch_limit_gaussians // max(gaussians_per_batch, 1))
    return max(1, min(batch_size, by_tiles, by_gaussians))


class _RasterizeProjectedGaussiansV5(torch.autograd.Function):
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
        inputs_sorted_by_depth: bool,
    ) -> Tensor:
        if not hasattr(torch.ops, "gsplat_metal_v5_softmax_gs"):
            raise RuntimeError("gsplat_metal_v5_softmax_gs custom ops not found. Build the extension first.")

        B, G = means2d_b.shape[:2]
        perm, means2d_s, conics_s, colors_s, opacities_s, depths_s = _maybe_sort_inputs_by_depth(
            means2d_b,
            conics_b,
            colors_b,
            opacities_b,
            depths_b,
            inputs_sorted_by_depth=inputs_sorted_by_depth,
        )

        means_flat = means2d_s.reshape(B * G, 2).contiguous()
        conics_flat = conics_s.reshape(B * G, 3).contiguous()
        colors_flat = colors_s.reshape(B * G, 3).contiguous()
        opacities_flat = opacities_s.reshape(B * G).contiguous()
        depths_flat = depths_s.reshape(B * G).contiguous()

        tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v5_softmax_gs.bin(
            means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32
        )
        out_fast, tile_stop_counts = torch.ops.gsplat_metal_v5_softmax_gs.render_fast_forward_state(
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            depths_flat,
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
                overflow_tile_imgs = torch.ops.gsplat_metal_v5_softmax_gs.render_overflow_forward(
                    means_flat,
                    conics_flat,
                    colors_flat,
                    opacities_flat,
                    depths_flat,
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

        selected_ids = torch.empty((0,), device=means2d_b.device, dtype=torch.int32)
        selected_weights = torch.empty((0,), device=means2d_b.device, dtype=means2d_b.dtype)
        softmax_enabled = bool(int(meta_i32[11].item()))
        tape_k = int(meta_i32[12].item()) if meta_i32.numel() > 12 else 0
        if softmax_enabled and tape_k > 0:
            selected_ids, selected_weights, _residual_weight, _final_alpha = (
                torch.ops.gsplat_metal_v5_softmax_gs.render_fast_softmax_bounded_tape(
                    means_flat,
                    conics_flat,
                    opacities_flat,
                    depths_flat,
                    meta_i32,
                    meta_f32,
                    tile_counts,
                    tile_offsets,
                    binned_ids,
                )
            )
            if enable_overflow_fallback and overflow_tile_ids.numel() > 0:
                overflow_ids, overflow_weights, _overflow_residual, _overflow_alpha = (
                    torch.ops.gsplat_metal_v5_softmax_gs.render_overflow_softmax_bounded_tape(
                        means_flat,
                        conics_flat,
                        opacities_flat,
                        depths_flat,
                        meta_i32,
                        meta_f32,
                        overflow_tile_ids,
                        overflow_tile_offsets,
                        overflow_sorted_ids,
                    )
                )
                selected_ids = selected_ids.clone()
                selected_weights = selected_weights.clone()
                _scatter_tile_tensor_(selected_ids, overflow_tile_ids, overflow_ids, tiles_per_image, tiles_x, tile_size)
                _scatter_tile_tensor_(selected_weights, overflow_tile_ids, overflow_weights, tiles_per_image, tiles_x, tile_size)

        ctx.save_for_backward(
            perm,
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            depths_flat,
            meta_i32,
            meta_f32,
            tile_counts,
            tile_offsets,
            binned_ids,
            tile_stop_counts,
            overflow_tile_ids,
            overflow_tile_offsets,
            overflow_sorted_ids,
            selected_ids,
            selected_weights,
        )
        ctx.batch_size = B
        ctx.gaussians_per_batch = G
        ctx.tiles_per_image = tiles_per_image
        ctx.tiles_x = tiles_x
        ctx.tile_size = tile_size
        ctx.enable_overflow_fallback = enable_overflow_fallback
        ctx.inputs_sorted_by_depth = inputs_sorted_by_depth
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            perm,
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            depths_flat,
            meta_i32,
            meta_f32,
            tile_counts,
            tile_offsets,
            binned_ids,
            tile_stop_counts,
            overflow_tile_ids,
            overflow_tile_offsets,
            overflow_sorted_ids,
            selected_ids,
            selected_weights,
        ) = ctx.saved_tensors

        softmax_enabled = bool(int(meta_i32[11].item()))
        use_tape_backward = softmax_enabled and selected_ids.numel() > 0
        if use_tape_backward:
            g_means_flat, g_conics_flat, g_opacities_flat, g_depths_flat = (
                torch.ops.gsplat_metal_v5_softmax_gs.render_softmax_tape_scalar_backward(
                    grad_out.contiguous(),
                    means_flat,
                    conics_flat,
                    colors_flat,
                    opacities_flat,
                    depths_flat,
                    selected_ids,
                    meta_i32,
                    meta_f32,
                )
            )
            g_colors_flat = torch.ops.gsplat_metal_v5_softmax_gs.render_softmax_tape_color_backward(
                grad_out.contiguous(),
                selected_ids,
                selected_weights,
                meta_i32,
                meta_f32,
            )
        else:
            grad_fast = grad_out.contiguous().clone()
            if ctx.enable_overflow_fallback and overflow_tile_ids.numel() > 0:
                _zero_tile_images_(grad_fast, overflow_tile_ids, ctx.tiles_per_image, ctx.tiles_x, ctx.tile_size)

            if softmax_enabled:
                g_means_flat, g_conics_flat, g_colors_flat, g_opacities_flat, g_depths_flat = (
                    torch.ops.gsplat_metal_v5_softmax_gs.render_fast_backward_softmax_recompute(
                        grad_fast,
                        means_flat,
                        conics_flat,
                        colors_flat,
                        opacities_flat,
                        depths_flat,
                        meta_i32,
                        meta_f32,
                        tile_counts,
                        tile_offsets,
                        binned_ids,
                        tile_stop_counts,
                    )
                )
            else:
                g_means_flat, g_conics_flat, g_colors_flat, g_opacities_flat = torch.ops.gsplat_metal_v5_softmax_gs.render_fast_backward_saved(
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
                g_depths_flat = torch.zeros_like(depths_flat)

            if ctx.enable_overflow_fallback and overflow_tile_ids.numel() > 0:
                grad_tiles = _gather_tile_images(grad_out.contiguous(), overflow_tile_ids, ctx.tiles_per_image, ctx.tiles_x, ctx.tile_size)
                if softmax_enabled:
                    go_means, go_conics, go_colors, go_opacities, go_depths = (
                        torch.ops.gsplat_metal_v5_softmax_gs.render_overflow_backward_softmax_recompute(
                            grad_tiles,
                            means_flat,
                            conics_flat,
                            colors_flat,
                            opacities_flat,
                            depths_flat,
                            meta_i32,
                            meta_f32,
                            overflow_tile_ids,
                            overflow_tile_offsets,
                            overflow_sorted_ids,
                        )
                    )
                    g_depths_flat = g_depths_flat + go_depths
                else:
                    go_means, go_conics, go_colors, go_opacities = torch.ops.gsplat_metal_v5_softmax_gs.render_overflow_backward(
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
        if ctx.inputs_sorted_by_depth:
            g_means_b = g_means_flat.view(B, G, 2)
            g_conics_b = g_conics_flat.view(B, G, 3)
            g_colors_b = g_colors_flat.view(B, G, 3)
            g_opacities_b = g_opacities_flat.view(B, G)
            g_depths_b = g_depths_flat.view(B, G)
        else:
            g_means_b = _unsort_batched(g_means_flat.view(B, G, 2), perm)
            g_conics_b = _unsort_batched(g_conics_flat.view(B, G, 3), perm)
            g_colors_b = _unsort_batched(g_colors_flat.view(B, G, 3), perm)
            g_opacities_b = _unsort_batched(g_opacities_flat.view(B, G), perm)
            g_depths_b = _unsort_batched(g_depths_flat.view(B, G), perm)
        return g_means_b, g_conics_b, g_colors_b, g_opacities_b, g_depths_b, None, None, None, None


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
    _, means2d_s, conics_s, colors_s, opacities_s, depths_s = _maybe_sort_inputs_by_depth(
        means2d_b,
        conics_b,
        colors_b,
        opacities_b,
        depths_b,
        inputs_sorted_by_depth=bool(config.inputs_sorted_by_depth),
    )

    means_flat = means2d_s.reshape(B * G, 2).contiguous()
    conics_flat = conics_s.reshape(B * G, 3).contiguous()
    colors_flat = colors_s.reshape(B * G, 3).contiguous()
    opacities_flat = opacities_s.reshape(B * G).contiguous()
    depths_flat = depths_s.reshape(B * G).contiguous()

    tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v5_softmax_gs.bin(
        means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32
    )
    out_fast = torch.ops.gsplat_metal_v5_softmax_gs.render_fast_forward_eval(
        means_flat,
        conics_flat,
        colors_flat,
        opacities_flat,
        depths_flat,
        meta_i32,
        meta_f32,
        tile_counts,
        tile_offsets,
        binned_ids,
    )

    if config.enable_overflow_fallback:
        overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids = _gather_overflow_segments(
            tile_counts, tile_offsets, binned_ids, int(meta_i32[7].item())
        )
        if overflow_tile_ids.numel() > 0:
            overflow_tile_imgs = torch.ops.gsplat_metal_v5_softmax_gs.render_overflow_forward(
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                depths_flat,
                meta_i32,
                meta_f32,
                overflow_tile_ids,
                overflow_tile_offsets,
                overflow_sorted_ids,
            )
            out = out_fast.clone()
            _scatter_tile_images_(out, overflow_tile_ids, overflow_tile_imgs, int(meta_i32[10].item()), int(meta_i32[3].item()), int(meta_i32[4].item()))
            return out
    elif bool((tile_counts > int(meta_i32[7].item())).any().item()):
        raise RuntimeError(
            f"Tile overflow detected with max_fast_pairs={int(meta_i32[7].item())}. "
            "Enable overflow fallback or increase the runtime cap."
        )
    return out_fast


def _bounded_tape_chunk(
    means2d_b: Tensor,
    conics_b: Tensor,
    colors_b: Tensor,
    opacities_b: Tensor,
    depths_b: Tensor,
    config: RasterConfig,
    *,
    k_limit: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if int(k_limit) < 1:
        raise ValueError(f"k_limit must be >= 1, got {k_limit}.")
    config = replace(config, softmax_gs_tape_k=int(k_limit))
    B, G = means2d_b.shape[:2]
    meta_i32, meta_f32 = _make_meta(config, means2d_b.device, B, G)
    _, means2d_s, conics_s, colors_s, opacities_s, depths_s = _maybe_sort_inputs_by_depth(
        means2d_b,
        conics_b,
        colors_b,
        opacities_b,
        depths_b,
        inputs_sorted_by_depth=bool(config.inputs_sorted_by_depth),
    )

    means_flat = means2d_s.reshape(B * G, 2).contiguous()
    conics_flat = conics_s.reshape(B * G, 3).contiguous()
    colors_flat = colors_s.reshape(B * G, 3).contiguous()
    opacities_flat = opacities_s.reshape(B * G).contiguous()
    depths_flat = depths_s.reshape(B * G).contiguous()

    tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v5_softmax_gs.bin(
        means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32
    )
    selected_ids, selected_weights, residual_weight, final_alpha = (
        torch.ops.gsplat_metal_v5_softmax_gs.render_fast_softmax_bounded_tape(
            means_flat,
            conics_flat,
            opacities_flat,
            depths_flat,
            meta_i32,
            meta_f32,
            tile_counts,
            tile_offsets,
            binned_ids,
        )
    )

    max_fast_pairs = int(meta_i32[7].item())
    if config.enable_overflow_fallback:
        overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids = _gather_overflow_segments(
            tile_counts, tile_offsets, binned_ids, max_fast_pairs
        )
        if overflow_tile_ids.numel() > 0:
            overflow_ids, overflow_weights, overflow_residual, overflow_alpha = (
                torch.ops.gsplat_metal_v5_softmax_gs.render_overflow_softmax_bounded_tape(
                    means_flat,
                    conics_flat,
                    opacities_flat,
                    depths_flat,
                    meta_i32,
                    meta_f32,
                    overflow_tile_ids,
                    overflow_tile_offsets,
                    overflow_sorted_ids,
                )
            )
            tile_size = int(meta_i32[4].item())
            tiles_x = int(meta_i32[3].item())
            tiles_per_image = int(meta_i32[10].item())
            selected_ids = selected_ids.clone()
            selected_weights = selected_weights.clone()
            residual_weight = residual_weight.clone()
            final_alpha = final_alpha.clone()
            _scatter_tile_tensor_(selected_ids, overflow_tile_ids, overflow_ids, tiles_per_image, tiles_x, tile_size)
            _scatter_tile_tensor_(selected_weights, overflow_tile_ids, overflow_weights, tiles_per_image, tiles_x, tile_size)
            _scatter_tile_tensor_(residual_weight, overflow_tile_ids, overflow_residual, tiles_per_image, tiles_x, tile_size)
            _scatter_tile_tensor_(final_alpha, overflow_tile_ids, overflow_alpha, tiles_per_image, tiles_x, tile_size)
    elif bool((tile_counts > max_fast_pairs).any().item()):
        raise RuntimeError(
            f"Tile overflow detected with max_fast_pairs={max_fast_pairs}. "
            "Enable overflow fallback or increase the runtime cap."
        )
    return selected_ids, selected_weights, residual_weight, final_alpha


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
    chunk_b = _choose_batch_chunk_size(config, B, G, tiles_y * tiles_x)

    outs = []
    train_mode = _should_use_training_path(means2d_b, conics_b, colors_b, opacities_b)
    for b0 in range(0, B, chunk_b):
        b1 = min(B, b0 + chunk_b)
        m = means2d_b[b0:b1].contiguous()
        q = conics_b[b0:b1].contiguous()
        c = colors_b[b0:b1].contiguous()
        o = opacities_b[b0:b1].contiguous()
        d = depths_b[b0:b1].contiguous()

        if train_mode:
            meta_i32, meta_f32 = _make_meta(config, m.device, b1 - b0, G)
            outs.append(
                _RasterizeProjectedGaussiansV5.apply(
                    m,
                    q,
                    c,
                    o,
                    d,
                    meta_i32,
                    meta_f32,
                    bool(config.enable_overflow_fallback),
                    bool(config.inputs_sorted_by_depth),
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


def rasterize_softmax_gs_bounded_tape(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: RasterConfig,
    *,
    k_limit: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return Metal-computed per-pixel top-K Softmax-GS contribution tape.

    The selected IDs are flattened IDs in the sorted input order consumed by
    the shader. They are returned in front-to-back/ray order; empty slots are
    `-1` with zero weight.
    """

    _check_inputs(means2d, conics, colors, opacities, depths)
    means2d_b, conics_b, colors_b, opacities_b, depths_b, was_batched = _normalize_inputs(
        means2d, conics, colors, opacities, depths
    )
    _runtime_validate(config)
    if int(k_limit) < 1:
        raise ValueError(f"k_limit must be >= 1, got {k_limit}.")

    B, G = means2d_b.shape[:2]
    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    chunk_b = _choose_batch_chunk_size(config, B, G, tiles_y * tiles_x)

    id_chunks = []
    weight_chunks = []
    residual_chunks = []
    alpha_chunks = []
    for b0 in range(0, B, chunk_b):
        b1 = min(B, b0 + chunk_b)
        ids, weights, residual, alpha = _bounded_tape_chunk(
            means2d_b[b0:b1].contiguous(),
            conics_b[b0:b1].contiguous(),
            colors_b[b0:b1].contiguous(),
            opacities_b[b0:b1].contiguous(),
            depths_b[b0:b1].contiguous(),
            config,
            k_limit=int(k_limit),
        )
        id_chunks.append(ids)
        weight_chunks.append(weights)
        residual_chunks.append(residual)
        alpha_chunks.append(alpha)

    selected_ids = torch.cat(id_chunks, dim=0) if len(id_chunks) > 1 else id_chunks[0]
    selected_weights = torch.cat(weight_chunks, dim=0) if len(weight_chunks) > 1 else weight_chunks[0]
    residual_weight = torch.cat(residual_chunks, dim=0) if len(residual_chunks) > 1 else residual_chunks[0]
    final_alpha = torch.cat(alpha_chunks, dim=0) if len(alpha_chunks) > 1 else alpha_chunks[0]
    if was_batched:
        return selected_ids, selected_weights, residual_weight, final_alpha
    return selected_ids[0], selected_weights[0], residual_weight[0], final_alpha[0]


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

    all_tile_counts = []
    all_stop_counts = []
    images = []

    for b0 in range(0, B, chunk_b):
        b1 = min(B, b0 + chunk_b)
        m = means2d_b[b0:b1].contiguous()
        q = conics_b[b0:b1].contiguous()
        c = colors_b[b0:b1].contiguous()
        o = opacities_b[b0:b1].contiguous()
        d = depths_b[b0:b1].contiguous()

        _, m_s_b, q_s_b, c_s_b, o_s_b, d_s_b = _maybe_sort_inputs_by_depth(
            m,
            q,
            c,
            o,
            d,
            inputs_sorted_by_depth=bool(config.inputs_sorted_by_depth),
        )
        m_s = m_s_b.reshape(-1, 2)
        q_s = q_s_b.reshape(-1, 3)
        c_s = c_s_b.reshape(-1, 3)
        o_s = o_s_b.reshape(-1)
        d_s = d_s_b.reshape(-1)

        meta_i32, meta_f32 = _make_meta(config, means2d_b.device, b1 - b0, G)
        tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v5_softmax_gs.bin(
            m_s, q_s, c_s, o_s, meta_i32, meta_f32
        )
        all_tile_counts.append(tile_counts.detach().cpu().to(torch.float32))

        if run_forward or return_image:
            if return_image:
                chunk_img = _rasterize_chunk_eval(m, q, c, o, d, config)
                images.append(chunk_img)
            _, stop_counts = torch.ops.gsplat_metal_v5_softmax_gs.render_fast_forward_state(
                m_s,
                q_s,
                c_s,
                o_s,
                d_s,
                meta_i32,
                meta_f32,
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
