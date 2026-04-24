from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class ReferenceState:
    image: Tensor
    packed_gaussians: Tensor
    out_cpu: Tensor
    front_ids: Tensor
    front_raw_alpha: Tensor
    front_meta: Tensor
    tile_offsets: Tensor
    tile_ids: Tensor
    perm: Tensor


def sort_projected_inputs(means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor):
    perm = torch.argsort(depths.detach(), dim=1, stable=True)
    means2d_s = torch.gather(means2d, 1, perm.unsqueeze(-1).expand(-1, -1, 2)).contiguous()
    conics_s = torch.gather(conics, 1, perm.unsqueeze(-1).expand(-1, -1, 3)).contiguous()
    colors_s = torch.gather(colors, 1, perm.unsqueeze(-1).expand(-1, -1, 3)).contiguous()
    opacities_s = torch.gather(opacities, 1, perm).contiguous()
    depths_s = torch.gather(depths, 1, perm).contiguous()
    return means2d_s, conics_s, colors_s, opacities_s, depths_s, perm


def pack_gaussians_cpu(means2d_s: Tensor, conics_s: Tensor, colors_s: Tensor, opacities_s: Tensor) -> Tensor:
    packed = torch.zeros(
        (means2d_s.shape[0], means2d_s.shape[1], 12),
        dtype=torch.float32,
        device="cpu",
    )
    packed[..., 0:2] = means2d_s.detach().cpu()
    packed[..., 2] = opacities_s.detach().cpu()
    packed[..., 4:7] = conics_s.detach().cpu()
    packed[..., 8:11] = colors_s.detach().cpu()
    return packed.contiguous()


def dense_reference_forward(
    means2d_s: Tensor,
    conics_s: Tensor,
    colors_s: Tensor,
    opacities_s: Tensor,
    height: int,
    width: int,
    alpha_threshold: float,
    background: Tuple[float, float, float],
) -> Tensor:
    ys = torch.arange(height, dtype=means2d_s.dtype, device=means2d_s.device) + 0.5
    xs = torch.arange(width, dtype=means2d_s.dtype, device=means2d_s.device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    bg_t = torch.tensor(background, dtype=means2d_s.dtype, device=means2d_s.device)
    outs = []
    for b in range(means2d_s.shape[0]):
        img = torch.zeros(height, width, 3, dtype=means2d_s.dtype, device=means2d_s.device)
        T = torch.ones(height, width, dtype=means2d_s.dtype, device=means2d_s.device)
        for i in range(means2d_s.shape[1]):
            dx = xx - means2d_s[b, i, 0]
            dy = yy - means2d_s[b, i, 1]
            power = -0.5 * (
                conics_s[b, i, 0] * dx * dx
                + 2.0 * conics_s[b, i, 1] * dx * dy
                + conics_s[b, i, 2] * dy * dy
            )
            raw = opacities_s[b, i] * torch.exp(power)
            alpha = torch.clamp(raw, max=0.99)
            alpha = torch.where((power <= 0.0) & (alpha >= alpha_threshold), alpha, torch.zeros_like(alpha))
            w = T * alpha
            img = img + w[..., None] * colors_s[b, i]
            T = T * (1.0 - alpha)
        outs.append(img + T[..., None] * bg_t)
    return torch.stack(outs, dim=0)


def pack_front_meta(count: int, overflow: bool) -> int:
    return (count & 0x0F) | (0x80 if overflow else 0)


def unpack_front_count(meta: int) -> int:
    return meta & 0x0F


def unpack_front_overflow(meta: int) -> bool:
    return (meta & 0x80) != 0


def support_bbox(
    mean_x: float,
    mean_y: float,
    opacity: float,
    a: float,
    b: float,
    c: float,
    *,
    height: int,
    width: int,
    alpha_threshold: float,
    eps: float,
) -> tuple[int, int, int, int] | None:
    if opacity <= alpha_threshold:
        return None
    ratio = max(alpha_threshold / max(opacity, eps), eps)
    tau = -2.0 * math.log(ratio)
    if not (math.isfinite(tau) and tau > 0.0):
        return None
    det = max(a * c - b * b, eps)
    hx = math.sqrt(max(tau * c / det, 0.0))
    hy = math.sqrt(max(tau * a / det, 0.0))
    x0 = max(0.0, min(float(width), mean_x - hx))
    x1 = max(0.0, min(float(width), mean_x + hx))
    y0 = max(0.0, min(float(height), mean_y - hy))
    y1 = max(0.0, min(float(height), mean_y + hy))
    tx0 = int(math.floor(x0))
    tx1 = int(math.floor(x1))
    ty0 = int(math.floor(y0))
    ty1 = int(math.floor(y1))
    return tx0, tx1, ty0, ty1


def build_tile_bins(
    means2d_s: Tensor,
    conics_s: Tensor,
    opacities_s: Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
    alpha_threshold: float,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    B, G, _ = means2d_s.shape
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    tiles_per_batch = tiles_x * tiles_y

    offsets = torch.empty((B, tiles_per_batch + 1), dtype=torch.int32, device="cpu")
    flat_ids: list[int] = []
    for b in range(B):
        bins: list[list[int]] = [[] for _ in range(tiles_per_batch)]
        for i in range(G):
            bbox = support_bbox(
                float(means2d_s[b, i, 0]),
                float(means2d_s[b, i, 1]),
                float(opacities_s[b, i]),
                float(conics_s[b, i, 0]),
                float(conics_s[b, i, 1]),
                float(conics_s[b, i, 2]),
                height=height,
                width=width,
                alpha_threshold=alpha_threshold,
                eps=eps,
            )
            if bbox is None:
                continue
            x0, x1, y0, y1 = bbox
            tx0 = max(0, min(tiles_x - 1, x0 // tile_size))
            tx1 = max(0, min(tiles_x - 1, x1 // tile_size))
            ty0 = max(0, min(tiles_y - 1, y0 // tile_size))
            ty1 = max(0, min(tiles_y - 1, y1 // tile_size))
            for ty in range(ty0, ty1 + 1):
                base = ty * tiles_x
                for tx in range(tx0, tx1 + 1):
                    bins[base + tx].append(i)
        offsets[b, 0] = len(flat_ids)
        for t, lst in enumerate(bins):
            flat_ids.extend(lst)
            offsets[b, t + 1] = len(flat_ids)
    tile_ids = torch.tensor(flat_ids, dtype=torch.uint16, device="cpu")
    return offsets, tile_ids


def capture_frontk_state_dense(
    means2d_s: Tensor,
    conics_s: Tensor,
    opacities_s: Tensor,
    *,
    height: int,
    width: int,
    front_k: int,
    alpha_threshold: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    B, G, _ = means2d_s.shape
    front_ids = torch.zeros((B, height, width, front_k), dtype=torch.uint16, device="cpu")
    front_raw_alpha = torch.zeros((B, height, width, front_k), dtype=torch.float32, device="cpu")
    front_count = torch.zeros((B, height, width), dtype=torch.int32, device="cpu")
    overflow_mask = torch.zeros((B, height, width), dtype=torch.uint8, device="cpu")

    for b in range(B):
        for y in range(height):
            py = float(y) + 0.5
            for x in range(width):
                px = float(x) + 0.5
                captured = 0
                overflow = False
                for i in range(G):
                    dx = px - float(means2d_s[b, i, 0])
                    dy = py - float(means2d_s[b, i, 1])
                    power = -0.5 * (
                        float(conics_s[b, i, 0]) * dx * dx
                        + 2.0 * float(conics_s[b, i, 1]) * dx * dy
                        + float(conics_s[b, i, 2]) * dy * dy
                    )
                    if power > 0.0:
                        continue
                    raw = float(opacities_s[b, i]) * math.exp(power)
                    alpha = min(0.99, raw)
                    if alpha < alpha_threshold:
                        continue
                    if captured < front_k:
                        front_ids[b, y, x, captured] = i
                        front_raw_alpha[b, y, x, captured] = raw
                        captured += 1
                    else:
                        overflow = True
                        break
                front_count[b, y, x] = captured
                overflow_mask[b, y, x] = 1 if overflow else 0
    return front_ids, front_raw_alpha, front_count, overflow_mask


def capture_frontk_state_tiled(
    means2d_s: Tensor,
    conics_s: Tensor,
    opacities_s: Tensor,
    tile_offsets: Tensor,
    tile_ids: Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
    front_k: int,
    alpha_threshold: float,
) -> tuple[Tensor, Tensor, Tensor]:
    B, G, _ = means2d_s.shape
    del G
    tiles_x = (width + tile_size - 1) // tile_size

    front_ids = torch.zeros((B, height, width, front_k), dtype=torch.uint16, device="cpu")
    front_raw_alpha = torch.zeros((B, height, width, front_k), dtype=torch.float32, device="cpu")
    front_meta = torch.zeros((B, height, width), dtype=torch.uint8, device="cpu")

    for b in range(B):
        for y in range(height):
            py = float(y) + 0.5
            ty = y // tile_size
            for x in range(width):
                px = float(x) + 0.5
                tx = x // tile_size
                tile = ty * tiles_x + tx
                start = int(tile_offsets[b, tile])
                end = int(tile_offsets[b, tile + 1])
                captured = 0
                overflow = False
                for ptr in range(start, end):
                    idx = int(tile_ids[ptr])
                    dx = px - float(means2d_s[b, idx, 0])
                    dy = py - float(means2d_s[b, idx, 1])
                    power = -0.5 * (
                        float(conics_s[b, idx, 0]) * dx * dx
                        + 2.0 * float(conics_s[b, idx, 1]) * dx * dy
                        + float(conics_s[b, idx, 2]) * dy * dy
                    )
                    if power > 0.0:
                        continue
                    raw = float(opacities_s[b, idx]) * math.exp(power)
                    alpha = min(0.99, raw)
                    if alpha < alpha_threshold:
                        continue
                    if captured < front_k:
                        front_ids[b, y, x, captured] = idx
                        front_raw_alpha[b, y, x, captured] = raw
                        captured += 1
                    else:
                        overflow = True
                        break
                front_meta[b, y, x] = pack_front_meta(captured, overflow)
    return front_ids, front_raw_alpha, front_meta


def tiled_scan_work(tile_offsets: Tensor, *, height: int, width: int, tile_size: int) -> int:
    tile_offsets = tile_offsets.cpu()
    B = tile_offsets.shape[0]
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    work = 0
    for b in range(B):
        for ty in range(tiles_y):
            tile_h = tile_size if ty + 1 < tiles_y else height - ty * tile_size
            for tx in range(tiles_x):
                tile_w = tile_size if tx + 1 < tiles_x else width - tx * tile_size
                tile = ty * tiles_x + tx
                count = int(tile_offsets[b, tile + 1] - tile_offsets[b, tile])
                work += tile_w * tile_h * count
    return work


def reference_forward_state(
    means2d_s: Tensor,
    conics_s: Tensor,
    colors_s: Tensor,
    opacities_s: Tensor,
    depths_s: Tensor,
    *,
    height: int,
    width: int,
    front_k: int,
    tile_size: int,
    alpha_threshold: float,
    background: Tuple[float, float, float],
    eps: float = 1e-8,
) -> ReferenceState:
    del depths_s
    image = dense_reference_forward(means2d_s, conics_s, colors_s, opacities_s, height, width, alpha_threshold, background)
    packed_gaussians = pack_gaussians_cpu(means2d_s, conics_s, colors_s, opacities_s)
    tile_offsets, tile_ids = build_tile_bins(
        means2d_s,
        conics_s,
        opacities_s,
        height=height,
        width=width,
        tile_size=tile_size,
        alpha_threshold=alpha_threshold,
        eps=eps,
    )
    front_ids, front_raw_alpha, front_meta = capture_frontk_state_tiled(
        means2d_s,
        conics_s,
        opacities_s,
        tile_offsets,
        tile_ids,
        height=height,
        width=width,
        tile_size=tile_size,
        front_k=front_k,
        alpha_threshold=alpha_threshold,
    )
    dummy_perm = torch.empty(0, dtype=torch.int64, device=means2d_s.device)
    return ReferenceState(
        image=image,
        packed_gaussians=packed_gaussians,
        out_cpu=image.detach().cpu(),
        front_ids=front_ids,
        front_raw_alpha=front_raw_alpha,
        front_meta=front_meta,
        tile_offsets=tile_offsets,
        tile_ids=tile_ids,
        perm=dummy_perm,
    )


def manual_backward_from_state(
    grad_out: Tensor,
    means2d_s: Tensor,
    conics_s: Tensor,
    colors_s: Tensor,
    opacities_s: Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
    alpha_threshold: float,
    background: Tuple[float, float, float],
    eps: float,
    state: ReferenceState,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    B, G, _ = means2d_s.shape
    g_means = torch.zeros_like(means2d_s)
    g_conics = torch.zeros_like(conics_s)
    g_colors = torch.zeros_like(colors_s)
    g_opacities = torch.zeros_like(opacities_s)
    g_depths = torch.zeros_like(opacities_s)
    bg_t = torch.tensor(background, dtype=means2d_s.dtype, device=means2d_s.device)

    tiles_x = (width + tile_size - 1) // tile_size

    for b in range(B):
        for y in range(height):
            py = float(y) + 0.5
            ty = y // tile_size
            for x in range(width):
                px = float(x) + 0.5
                tx = x // tile_size
                tile = ty * tiles_x + tx
                go = grad_out[b, y, x]
                meta = int(state.front_meta[b, y, x])
                if not unpack_front_overflow(meta):
                    n = unpack_front_count(meta)
                    if n == 0:
                        continue
                    alphas = []
                    colors = []
                    ids = []
                    for j in range(n):
                        idx = int(state.front_ids[b, y, x, j])
                        ids.append(idx)
                        raw = float(state.front_raw_alpha[b, y, x, j])
                        alphas.append(min(0.99, raw))
                        colors.append(colors_s[b, idx])
                    suffix = [bg_t for _ in range(n + 1)]
                    for j in range(n - 1, -1, -1):
                        suffix[j] = alphas[j] * colors[j] + (1.0 - alphas[j]) * suffix[j + 1]
                    T = 1.0
                    for j in range(n):
                        idx = ids[j]
                        alpha = alphas[j]
                        raw = float(state.front_raw_alpha[b, y, x, j])
                        color = colors_s[b, idx]
                        dL_dalpha = torch.dot(go, T * (color - suffix[j + 1]))
                        dL_draw = dL_dalpha if raw < 0.99 else torch.zeros((), dtype=go.dtype, device=go.device)
                        w = T * alpha
                        g_colors[b, idx] += go * w
                        dx = px - float(means2d_s[b, idx, 0])
                        dy = py - float(means2d_s[b, idx, 1])
                        a = float(conics_s[b, idx, 0])
                        bxy = float(conics_s[b, idx, 1])
                        c = float(conics_s[b, idx, 2])
                        exp_term = raw / max(float(opacities_s[b, idx]), eps)
                        g_opacities[b, idx] += dL_draw * exp_term
                        g_power = dL_draw * raw
                        g_means[b, idx, 0] += g_power * (a * dx + bxy * dy)
                        g_means[b, idx, 1] += g_power * (bxy * dx + c * dy)
                        g_conics[b, idx, 0] += g_power * (-0.5 * dx * dx)
                        g_conics[b, idx, 1] += g_power * (-dx * dy)
                        g_conics[b, idx, 2] += g_power * (-0.5 * dy * dy)
                        T *= (1.0 - alpha)
                else:
                    A = torch.zeros(3, dtype=means2d_s.dtype, device=means2d_s.device)
                    T = 1.0
                    outc = state.image[b, y, x]
                    start = int(state.tile_offsets[b, tile])
                    end = int(state.tile_offsets[b, tile + 1])
                    for ptr in range(start, end):
                        idx = int(state.tile_ids[ptr])
                        dx = px - float(means2d_s[b, idx, 0])
                        dy = py - float(means2d_s[b, idx, 1])
                        power = -0.5 * (
                            float(conics_s[b, idx, 0]) * dx * dx
                            + 2.0 * float(conics_s[b, idx, 1]) * dx * dy
                            + float(conics_s[b, idx, 2]) * dy * dy
                        )
                        if power > 0.0:
                            continue
                        raw = float(opacities_s[b, idx]) * math.exp(power)
                        alpha = min(0.99, raw)
                        if alpha < alpha_threshold:
                            continue
                        w = T * alpha
                        color = colors_s[b, idx]
                        denom = max(T * (1.0 - alpha), eps)
                        behind = (outc - A - w * color) / denom
                        dL_dalpha = torch.dot(go, T * (color - behind))
                        dL_draw = dL_dalpha if raw < 0.99 else torch.zeros((), dtype=go.dtype, device=go.device)
                        g_colors[b, idx] += go * w
                        exp_term = raw / max(float(opacities_s[b, idx]), eps)
                        g_opacities[b, idx] += dL_draw * exp_term
                        g_power = dL_draw * raw
                        a = float(conics_s[b, idx, 0])
                        bxy = float(conics_s[b, idx, 1])
                        c = float(conics_s[b, idx, 2])
                        g_means[b, idx, 0] += g_power * (a * dx + bxy * dy)
                        g_means[b, idx, 1] += g_power * (bxy * dx + c * dy)
                        g_conics[b, idx, 0] += g_power * (-0.5 * dx * dx)
                        g_conics[b, idx, 1] += g_power * (-dx * dy)
                        g_conics[b, idx, 2] += g_power * (-0.5 * dy * dy)
                        A = A + w * color
                        T *= (1.0 - alpha)
    return g_means, g_conics, g_colors, g_opacities, g_depths
