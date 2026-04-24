from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class ReferenceState:
    image: Tensor
    front_ids: Tensor
    front_raw_alpha: Tensor
    front_count: Tensor
    overflow_mask: Tensor
    perm: Tensor


def sort_projected_inputs(means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor):
    perm = torch.argsort(depths.detach(), dim=1, stable=True)
    means2d_s = torch.gather(means2d, 1, perm.unsqueeze(-1).expand(-1, -1, 2)).contiguous()
    conics_s = torch.gather(conics, 1, perm.unsqueeze(-1).expand(-1, -1, 3)).contiguous()
    colors_s = torch.gather(colors, 1, perm.unsqueeze(-1).expand(-1, -1, 3)).contiguous()
    opacities_s = torch.gather(opacities, 1, perm).contiguous()
    depths_s = torch.gather(depths, 1, perm).contiguous()
    return means2d_s, conics_s, colors_s, opacities_s, depths_s, perm


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


def capture_frontk_state(
    means2d_s: Tensor,
    conics_s: Tensor,
    opacities_s: Tensor,
    height: int,
    width: int,
    front_k: int,
    alpha_threshold: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    B, G, _ = means2d_s.shape
    front_ids = torch.full((B, height, width, front_k), -1, dtype=torch.int32, device=means2d_s.device)
    front_raw_alpha = torch.zeros((B, height, width, front_k), dtype=means2d_s.dtype, device=means2d_s.device)
    front_count = torch.zeros((B, height, width), dtype=torch.int32, device=means2d_s.device)
    overflow_mask = torch.zeros((B, height, width), dtype=torch.uint8, device=means2d_s.device)

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
    alpha_threshold: float,
    background: Tuple[float, float, float],
) -> ReferenceState:
    del depths_s
    image = dense_reference_forward(means2d_s, conics_s, colors_s, opacities_s, height, width, alpha_threshold, background)
    front_ids, front_raw_alpha, front_count, overflow_mask = capture_frontk_state(
        means2d_s,
        conics_s,
        opacities_s,
        height,
        width,
        front_k,
        alpha_threshold,
    )
    dummy_perm = torch.empty(0, dtype=torch.int64, device=means2d_s.device)
    return ReferenceState(image=image, front_ids=front_ids, front_raw_alpha=front_raw_alpha, front_count=front_count, overflow_mask=overflow_mask, perm=dummy_perm)


def manual_backward_from_state(
    grad_out: Tensor,
    means2d_s: Tensor,
    conics_s: Tensor,
    colors_s: Tensor,
    opacities_s: Tensor,
    *,
    height: int,
    width: int,
    front_k: int,
    alpha_threshold: float,
    background: Tuple[float, float, float],
    eps: float,
    state: ReferenceState,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    del front_k
    B, G, _ = means2d_s.shape
    g_means = torch.zeros_like(means2d_s)
    g_conics = torch.zeros_like(conics_s)
    g_colors = torch.zeros_like(colors_s)
    g_opacities = torch.zeros_like(opacities_s)
    g_depths = torch.zeros_like(opacities_s)
    bg_t = torch.tensor(background, dtype=means2d_s.dtype, device=means2d_s.device)

    for b in range(B):
        for y in range(height):
            py = float(y) + 0.5
            for x in range(width):
                px = float(x) + 0.5
                go = grad_out[b, y, x]
                if int(state.overflow_mask[b, y, x]) == 0:
                    n = int(state.front_count[b, y, x])
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
                    for idx in range(G):
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
