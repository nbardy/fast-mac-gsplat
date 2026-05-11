from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

try:
    from . import _C  # noqa: F401
except Exception:
    _C = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default) if raw is None or raw == "" else int(raw)


@dataclass(frozen=True)
class UVTRenderConfig:
    height: int
    width: int
    frames: int
    tile_x: int = 8
    tile_y: int = 8
    tile_t: int = 2
    tile_capacity: int = 128
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1.0e-4
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_alpha: float = 0.99


@dataclass(frozen=True)
class Gate0Stats:
    mean_rgb_error: float | None
    max_rgb_error: float | None
    forward_wall_clock_ms: float | None
    uvt_tile_tube_pairs: int
    summed_per_frame_tile_splat_pairs: int
    pair_ratio: float
    effective_pair_ratio_after_unstable_fallback: float
    stable_tile_fraction: float
    unstable_tile_fraction: float
    overflow_tile_count: int
    max_tile_count: int
    mean_tile_count: float
    metal_buffer_memory: int


@dataclass(frozen=True)
class UVTRenderResult:
    image: Tensor
    tile_counts: Tensor
    tile_overflow: Tensor
    tile_unstable: Tensor
    stats: Gate0Stats | None = None


def _runtime_validate(config: UVTRenderConfig) -> None:
    if config.height <= 0 or config.width <= 0 or config.frames <= 0:
        raise ValueError("height, width, and frames must be positive")
    if config.tile_x not in (8, 16):
        raise ValueError("tile_x must be 8 or 16")
    if config.tile_y not in (8, 16):
        raise ValueError("tile_y must be 8 or 16")
    if config.tile_t not in (1, 2, 4):
        raise ValueError("tile_t must be 1, 2, or 4")
    if config.tile_capacity not in (32, 64, 128, 256):
        raise ValueError("tile_capacity must be 32, 64, 128, or 256")
    if config.tile_x != _env_int("STAR_UVT_TILE_X", 8):
        raise ValueError("config.tile_x must match STAR_UVT_TILE_X")
    if config.tile_y != _env_int("STAR_UVT_TILE_Y", 8):
        raise ValueError("config.tile_y must match STAR_UVT_TILE_Y")
    if config.tile_t != _env_int("STAR_UVT_TILE_T", 2):
        raise ValueError("config.tile_t must match STAR_UVT_TILE_T")
    if config.tile_capacity != _env_int("STAR_UVT_TILE_CAPACITY", 128):
        raise ValueError("config.tile_capacity must match STAR_UVT_TILE_CAPACITY")


def _check_inputs(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    *,
    require_mps: bool,
) -> None:
    if ma.ndim != 2 or ma.shape[-1] != 3:
        raise ValueError("ma must have shape [N,3]")
    if q_uvt.shape != (ma.shape[0], 6):
        raise ValueError("q_uvt must have shape [N,6]")
    if depth0.shape != (ma.shape[0],):
        raise ValueError("depth0 must have shape [N]")
    if depth_beta.shape != (ma.shape[0], 3):
        raise ValueError("depth_beta must have shape [N,3]")
    if opacity.shape != (ma.shape[0],):
        raise ValueError("opacity must have shape [N]")
    if color.shape != (ma.shape[0], 3):
        raise ValueError("color must have shape [N,3]")
    for name, tensor in {
        "ma": ma,
        "q_uvt": q_uvt,
        "depth0": depth0,
        "depth_beta": depth_beta,
        "opacity": opacity,
        "color": color,
    }.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != ma.device:
            raise ValueError(f"{name} must be on the same device as ma")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if require_mps and ma.device.type != "mps":
        raise ValueError("Metal STAR-UVT render requires MPS tensors")


def _make_meta(config: UVTRenderConfig, device: torch.device, tube_count: int) -> tuple[Tensor, Tensor]:
    _runtime_validate(config)
    tiles_x = (config.width + config.tile_x - 1) // config.tile_x
    tiles_y = (config.height + config.tile_y - 1) // config.tile_y
    tiles_t = (config.frames + config.tile_t - 1) // config.tile_t
    tile_count = tiles_x * tiles_y * tiles_t
    meta_i32 = torch.tensor(
        [
            config.height,
            config.width,
            config.frames,
            config.tile_x,
            config.tile_y,
            config.tile_t,
            tiles_x,
            tiles_y,
            tiles_t,
            tile_count,
            tube_count,
            config.tile_capacity,
            0,
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
            1.0e-8,
            float(config.max_alpha),
        ],
        device=device,
        dtype=torch.float32,
    )
    return meta_i32, meta_f32


def _frame_time(frame: int, frames: int) -> float:
    return float(frame) - 0.5 * float(frames - 1)


def _quadratic(q: Tensor, d: Tensor) -> Tensor:
    return (
        q[..., 0] * d[..., 0] * d[..., 0]
        + 2.0 * q[..., 1] * d[..., 0] * d[..., 1]
        + 2.0 * q[..., 2] * d[..., 0] * d[..., 2]
        + q[..., 3] * d[..., 1] * d[..., 1]
        + 2.0 * q[..., 4] * d[..., 1] * d[..., 2]
        + q[..., 5] * d[..., 2] * d[..., 2]
    )


def _depth_at(ma: Tensor, depth0: Tensor, depth_beta: Tensor, a: Tensor) -> Tensor:
    return depth0 + ((a.unsqueeze(0) - ma) * depth_beta).sum(dim=-1)


def brute_force_render_uvt_tubes(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: UVTRenderConfig,
) -> Tensor:
    _runtime_validate(config)
    _check_inputs(ma, q_uvt, depth0, depth_beta, opacity, color, require_mps=False)
    device = ma.device
    bg = torch.tensor(config.background, dtype=torch.float32, device=device)
    out = torch.empty((config.frames, config.height, config.width, 3), dtype=torch.float32, device=device)
    for f in range(config.frames):
        t = _frame_time(f, config.frames)
        for y in range(config.height):
            for x in range(config.width):
                a = torch.tensor([x + 0.5, y + 0.5, t], dtype=torch.float32, device=device)
                d = a.unsqueeze(0) - ma
                qv = _quadratic(q_uvt, d)
                alpha = torch.clamp(opacity * torch.exp(-0.5 * qv), max=config.max_alpha)
                active = torch.nonzero(alpha >= config.alpha_threshold, as_tuple=False).flatten()
                if active.numel() == 0:
                    out[f, y, x] = bg
                    continue
                depths = _depth_at(ma.index_select(0, active), depth0.index_select(0, active), depth_beta.index_select(0, active), a)
                order = torch.argsort(depths, stable=True)
                accum = torch.zeros((3,), dtype=torch.float32, device=device)
                transmittance = torch.tensor(1.0, dtype=torch.float32, device=device)
                for local_idx in order.tolist():
                    tube_id = int(active[local_idx])
                    ai = alpha[tube_id]
                    accum = accum + transmittance * ai * color[tube_id]
                    transmittance = transmittance * (1.0 - ai)
                    if float(transmittance) <= config.transmittance_threshold:
                        break
                out[f, y, x] = accum + transmittance * bg
    return out


def _support_tau(opacity_value: float, config: UVTRenderConfig) -> float | None:
    if opacity_value <= config.alpha_threshold:
        return None
    return -2.0 * math.log(max(config.alpha_threshold / max(opacity_value, 1.0e-8), 1.0e-8))


def sliced_per_frame_pair_count(ma: Tensor, q_uvt: Tensor, opacity: Tensor, config: UVTRenderConfig) -> int:
    ma_cpu = ma.detach().cpu()
    q_cpu = q_uvt.detach().cpu()
    op_cpu = opacity.detach().cpu()
    tiles_x = (config.width + config.tile_x - 1) // config.tile_x
    total = 0
    for tube_id in range(ma_cpu.shape[0]):
        tau = _support_tau(float(op_cpu[tube_id]), config)
        if tau is None:
            continue
        m = ma_cpu[tube_id]
        q = q_cpu[tube_id]
        q2 = torch.tensor([[q[0], q[1]], [q[1], q[3]]], dtype=torch.float64)
        det = float(torch.linalg.det(q2))
        if not math.isfinite(det) or abs(det) < 1.0e-10:
            continue
        inv2 = torch.linalg.inv(q2)
        cross = torch.tensor([q[2], q[4]], dtype=torch.float64)
        for f in range(config.frames):
            dt = _frame_time(f, config.frames) - float(m[2])
            center_shift = -(inv2 @ cross) * dt
            center = m[:2].to(torch.float64) + center_shift
            constant = float(q[5]) * dt * dt - float(cross @ inv2 @ cross) * dt * dt
            tau2 = tau - constant
            if tau2 <= 0.0:
                continue
            half_x = math.sqrt(max(tau2 * float(inv2[0, 0]), 0.0))
            half_y = math.sqrt(max(tau2 * float(inv2[1, 1]), 0.0))
            x0 = max(0, int(math.floor(float(center[0]) - half_x - 0.5)))
            x1 = min(config.width - 1, int(math.ceil(float(center[0]) + half_x - 0.5)))
            y0 = max(0, int(math.floor(float(center[1]) - half_y - 0.5)))
            y1 = min(config.height - 1, int(math.ceil(float(center[1]) + half_y - 0.5)))
            if x0 > x1 or y0 > y1:
                continue
            total += ((x1 // config.tile_x) - (x0 // config.tile_x) + 1) * ((y1 // config.tile_y) - (y0 // config.tile_y) + 1)
    return int(total)


def _stats_from_aux(
    tile_counts: Tensor,
    tile_overflow: Tensor,
    tile_unstable: Tensor,
    ma: Tensor,
    q_uvt: Tensor,
    opacity: Tensor,
    config: UVTRenderConfig,
    *,
    image: Tensor | None,
    reference: Tensor | None,
    forward_wall_clock_ms: float | None,
) -> Gate0Stats:
    counts = tile_counts.detach().cpu().to(torch.int64)
    overflow = tile_overflow.detach().cpu().to(torch.int64)
    unstable = tile_unstable.detach().cpu().to(torch.int64)
    active = counts > 0
    active_count = int(active.sum().item())
    unstable_count = int(((unstable > 0) & active).sum().item())
    clipped_pairs = torch.clamp(counts, max=config.tile_capacity)
    uvt_pairs = int(clipped_pairs.sum().item())
    per_frame_pairs = sliced_per_frame_pair_count(ma, q_uvt, opacity, config)
    pair_ratio = float(uvt_pairs / max(per_frame_pairs, 1))
    # Gate 0 uses exact per-sample fallback for unstable tiles, so the effective
    # ratio is tracked separately but initially equals the clipped UVT pair load.
    effective_ratio = pair_ratio
    max_count = int(counts.max().item()) if counts.numel() else 0
    mean_count = float(counts.to(torch.float32).mean().item()) if counts.numel() else 0.0
    max_err = mean_err = None
    if image is not None and reference is not None:
        err = (image.detach().cpu() - reference.detach().cpu()).abs()
        max_err = float(err.max().item())
        mean_err = float(err.mean().item())
    return Gate0Stats(
        mean_rgb_error=mean_err,
        max_rgb_error=max_err,
        forward_wall_clock_ms=forward_wall_clock_ms,
        uvt_tile_tube_pairs=uvt_pairs,
        summed_per_frame_tile_splat_pairs=per_frame_pairs,
        pair_ratio=pair_ratio,
        effective_pair_ratio_after_unstable_fallback=effective_ratio,
        stable_tile_fraction=float((active_count - unstable_count) / max(active_count, 1)),
        unstable_tile_fraction=float(unstable_count / max(active_count, 1)),
        overflow_tile_count=int((overflow > 0).sum().item()),
        max_tile_count=max_count,
        mean_tile_count=mean_count,
        metal_buffer_memory=int(counts.numel() * config.tile_capacity * (4 + 4) + counts.numel() * 3 * 4),
    )


def render_uvt_tubes(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: UVTRenderConfig,
    *,
    return_aux: bool = False,
    reference: Tensor | Literal["cpu"] | None = None,
) -> Tensor | UVTRenderResult:
    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    color = color.contiguous()
    _check_inputs(ma, q_uvt, depth0, depth_beta, opacity, color, require_mps=True)
    if not hasattr(torch.ops, "star_uvt_v0"):
        raise RuntimeError("star_uvt_v0 custom ops not found. Build the extension first.")
    meta_i32, meta_f32 = _make_meta(config, ma.device, ma.shape[0])
    started_at = time.perf_counter() if return_aux else None
    image, tile_counts, tile_overflow, tile_unstable = torch.ops.star_uvt_v0.render(
        ma, q_uvt, depth0, depth_beta, opacity, color, meta_i32, meta_f32
    )
    if not return_aux:
        return image
    if ma.device.type == "mps":
        torch.mps.synchronize()
    forward_wall_clock_ms = None if started_at is None else (time.perf_counter() - started_at) * 1000.0
    ref = None
    if isinstance(reference, str):
        if reference != "cpu":
            raise ValueError("reference string must be 'cpu'")
        ref = brute_force_render_uvt_tubes(
            ma.detach().cpu(),
            q_uvt.detach().cpu(),
            depth0.detach().cpu(),
            depth_beta.detach().cpu(),
            opacity.detach().cpu(),
            color.detach().cpu(),
            config,
        )
    elif reference is not None:
        ref = reference
    stats = _stats_from_aux(
        tile_counts,
        tile_overflow,
        tile_unstable,
        ma,
        q_uvt,
        opacity,
        config,
        image=image,
        reference=ref,
        forward_wall_clock_ms=forward_wall_clock_ms,
    )
    return UVTRenderResult(image=image, tile_counts=tile_counts, tile_overflow=tile_overflow, tile_unstable=tile_unstable, stats=stats)


def simple_backward_samples(
    ma: Tensor,
    q_uvt: Tensor,
    opacity: Tensor,
    color: Tensor,
    grad_image: Tensor,
    config: UVTRenderConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    opacity = opacity.contiguous()
    color = color.contiguous()
    grad_image = grad_image.contiguous()
    dummy_depth0 = torch.zeros((ma.shape[0],), dtype=torch.float32, device=ma.device)
    dummy_depth_beta = torch.zeros((ma.shape[0], 3), dtype=torch.float32, device=ma.device)
    _check_inputs(ma, q_uvt, dummy_depth0, dummy_depth_beta, opacity, color, require_mps=True)
    if grad_image.shape != (config.frames, config.height, config.width, 3):
        raise ValueError("grad_image must have shape [frames,height,width,3]")
    if grad_image.dtype != torch.float32 or grad_image.device != ma.device:
        raise ValueError("grad_image must be float32 and on the same device as ma")
    meta_i32, meta_f32 = _make_meta(config, ma.device, ma.shape[0])
    return torch.ops.star_uvt_v0.simple_backward_samples(ma, q_uvt, opacity, color, grad_image, meta_i32, meta_f32)


def stable_backward_samples(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    grad_image: Tensor,
    config: UVTRenderConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    color = color.contiguous()
    grad_image = grad_image.contiguous()
    _check_inputs(ma, q_uvt, depth0, depth_beta, opacity, color, require_mps=True)
    if grad_image.shape != (config.frames, config.height, config.width, 3):
        raise ValueError("grad_image must have shape [frames,height,width,3]")
    if grad_image.dtype != torch.float32 or grad_image.device != ma.device:
        raise ValueError("grad_image must be float32 and on the same device as ma")
    meta_i32, meta_f32 = _make_meta(config, ma.device, ma.shape[0])
    ids, grad_ma, grad_q, grad_opacity, grad_color, tile_unstable, grad_count = torch.ops.star_uvt_v0.stable_backward_samples(
        ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    count = int(grad_count.detach().cpu().item())
    return (
        ids[:count],
        grad_ma[:count],
        grad_q[:count],
        grad_opacity[:count],
        grad_color[:count],
        tile_unstable,
    )


def _q_from_axis_velocity(lambda_u: float, lambda_v: float, lambda_t: float, velocity_u: float, velocity_v: float) -> list[float]:
    return [
        lambda_u,
        0.0,
        -lambda_u * velocity_u,
        lambda_v,
        -lambda_v * velocity_v,
        lambda_t + lambda_u * velocity_u * velocity_u + lambda_v * velocity_v * velocity_v,
    ]


def make_gate0_scene(
    scene: str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, UVTRenderConfig]:
    config = UVTRenderConfig(height=16, width=16, frames=4)
    dev = torch.device(device)
    if scene == "single_static":
        ma = torch.tensor([[8.0, 8.0, 0.0]], dtype=torch.float32, device=dev)
        q = torch.tensor([[0.22, 0.0, 0.0, 0.22, 0.0, 0.45]], dtype=torch.float32, device=dev)
        depth0 = torch.tensor([1.0], dtype=torch.float32, device=dev)
        depth_beta = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=dev)
        opacity = torch.tensor([0.75], dtype=torch.float32, device=dev)
        color = torch.tensor([[0.9, 0.2, 0.1]], dtype=torch.float32, device=dev)
    elif scene == "moving_diagonal":
        ma = torch.tensor([[8.0, 8.0, 0.0]], dtype=torch.float32, device=dev)
        q = torch.tensor([[0.28, 0.0, -0.18, 0.28, -0.18, 0.38]], dtype=torch.float32, device=dev)
        depth0 = torch.tensor([1.0], dtype=torch.float32, device=dev)
        depth_beta = torch.tensor([[0.01, 0.01, 0.02]], dtype=torch.float32, device=dev)
        opacity = torch.tensor([0.72], dtype=torch.float32, device=dev)
        color = torch.tensor([[0.1, 0.8, 0.25]], dtype=torch.float32, device=dev)
    elif scene == "crossing_depth":
        ma = torch.tensor([[7.0, 8.0, 0.0], [9.0, 8.0, 0.0]], dtype=torch.float32, device=dev)
        q = torch.tensor(
            [
                [0.20, 0.0, 0.0, 0.24, 0.0, 0.35],
                [0.20, 0.0, 0.0, 0.24, 0.0, 0.35],
            ],
            dtype=torch.float32,
            device=dev,
        )
        depth0 = torch.tensor([0.9, 1.1], dtype=torch.float32, device=dev)
        depth_beta = torch.tensor([[0.0, 0.0, 0.16], [0.0, 0.0, -0.16]], dtype=torch.float32, device=dev)
        opacity = torch.tensor([0.55, 0.55], dtype=torch.float32, device=dev)
        color = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.2, 0.9]], dtype=torch.float32, device=dev)
    elif scene == "two_non_crossing":
        ma = torch.tensor([[5.5, 5.5, 0.0], [10.5, 10.5, 0.0]], dtype=torch.float32, device=dev)
        q = torch.tensor(
            [
                _q_from_axis_velocity(0.25, 0.25, 0.40, 0.15, 0.05),
                _q_from_axis_velocity(0.23, 0.27, 0.38, -0.10, -0.20),
            ],
            dtype=torch.float32,
            device=dev,
        )
        depth0 = torch.tensor([0.8, 1.2], dtype=torch.float32, device=dev)
        depth_beta = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device=dev)
        opacity = torch.tensor([0.62, 0.58], dtype=torch.float32, device=dev)
        color = torch.tensor([[0.9, 0.7, 0.1], [0.1, 0.7, 0.9]], dtype=torch.float32, device=dev)
    elif scene == "fast_screen_motion":
        ma = torch.tensor([[8.0, 8.0, 0.0]], dtype=torch.float32, device=dev)
        q = torch.tensor([_q_from_axis_velocity(0.28, 0.28, 0.40, 1.20, -0.70)], dtype=torch.float32, device=dev)
        depth0 = torch.tensor([1.0], dtype=torch.float32, device=dev)
        depth_beta = torch.tensor([[0.02, -0.01, 0.03]], dtype=torch.float32, device=dev)
        opacity = torch.tensor([0.70], dtype=torch.float32, device=dev)
        color = torch.tensor([[0.2, 0.9, 0.9]], dtype=torch.float32, device=dev)
    elif scene == "wide_temporal_support":
        ma = torch.tensor([[8.0, 8.0, 0.0]], dtype=torch.float32, device=dev)
        q = torch.tensor([_q_from_axis_velocity(0.18, 0.18, 0.05, 0.35, 0.15)], dtype=torch.float32, device=dev)
        depth0 = torch.tensor([1.0], dtype=torch.float32, device=dev)
        depth_beta = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=dev)
        opacity = torch.tensor([0.50], dtype=torch.float32, device=dev)
        color = torch.tensor([[0.8, 0.3, 0.9]], dtype=torch.float32, device=dev)
    else:
        raise ValueError(f"unknown Gate 0 scene {scene!r}")
    return ma, q, depth0, depth_beta, opacity, color, config
