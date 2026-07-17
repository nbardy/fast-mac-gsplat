from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

try:
    from . import _C  # noqa: F401
except Exception:
    _C = None

Backend = Literal["auto", "dense", "metal"]


@dataclass(frozen=True)
class PRTRenderConfig:
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
    depth_epsilon: float = 1.0e-8


def _runtime_validate(config: PRTRenderConfig) -> None:
    if config.height <= 0 or config.width <= 0 or config.frames <= 0:
        raise ValueError("height, width, and frames must be positive")
    if config.tile_x <= 0 or config.tile_y <= 0 or config.tile_t <= 0:
        raise ValueError("tile_x, tile_y, and tile_t must be positive")
    if config.tile_capacity <= 0:
        raise ValueError("tile_capacity must be positive")
    if config.alpha_threshold <= 0.0:
        raise ValueError("alpha_threshold must be positive")
    if config.transmittance_threshold <= 0.0:
        raise ValueError("transmittance_threshold must be positive")
    if config.max_alpha <= 0.0 or config.max_alpha > 1.0:
        raise ValueError("max_alpha must be in (0, 1]")
    if config.depth_epsilon <= 0.0:
        raise ValueError("depth_epsilon must be positive")
    if len(config.background) != 3:
        raise ValueError("background must contain three RGB values")


def _frame_times(config: PRTRenderConfig, device: torch.device) -> Tensor:
    return torch.arange(config.frames, dtype=torch.float32, device=device) - 0.5 * float(config.frames - 1)


def _check_float_tensor(name: str, tensor: Tensor, *, device: torch.device) -> None:
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same device as the curve tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_projective_inputs(
    h_coeff: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
) -> None:
    if h_coeff.ndim != 3 or h_coeff.shape[-1] != 3:
        raise ValueError("h_coeff must have shape [N,H,3]")
    if h_coeff.shape[1] <= 0:
        raise ValueError("h_coeff must contain at least one polynomial term")
    tube_count = h_coeff.shape[0]
    expected_shapes = {
        "lambda_uv": (tube_count, 3),
        "lambda_t": (tube_count,),
        "center_t": (tube_count,),
        "opacity": (tube_count,),
        "color": (tube_count, 3),
    }
    tensors = {
        "lambda_uv": lambda_uv,
        "lambda_t": lambda_t,
        "center_t": center_t,
        "opacity": opacity,
        "color": color,
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensors[name].shape)}")
    for name, tensor in {"h_coeff": h_coeff, **tensors}.items():
        _check_float_tensor(name, tensor, device=h_coeff.device)


def _check_compiled_inputs(
    curve_uv_depth: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: PRTRenderConfig,
) -> None:
    if curve_uv_depth.ndim != 3 or curve_uv_depth.shape[-1] != 3:
        raise ValueError("curve_uv_depth must have shape [F,N,3]")
    if curve_uv_depth.shape[0] != config.frames:
        raise ValueError(f"curve_uv_depth frame count must match config.frames={config.frames}")
    tube_count = curve_uv_depth.shape[1]
    expected_shapes = {
        "lambda_uv": (tube_count, 3),
        "lambda_t": (tube_count,),
        "center_t": (tube_count,),
        "opacity": (tube_count,),
        "color": (tube_count, 3),
    }
    tensors = {
        "lambda_uv": lambda_uv,
        "lambda_t": lambda_t,
        "center_t": center_t,
        "opacity": opacity,
        "color": color,
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensors[name].shape)}")
    for name, tensor in {"curve_uv_depth": curve_uv_depth, **tensors}.items():
        _check_float_tensor(name, tensor, device=curve_uv_depth.device)


def _check_grad_image(grad_image: Tensor, config: PRTRenderConfig, device: torch.device) -> None:
    expected = (config.frames, config.height, config.width, 3)
    if tuple(grad_image.shape) != expected:
        raise ValueError(f"grad_image must have shape {expected}, got {tuple(grad_image.shape)}")
    _check_float_tensor("grad_image", grad_image, device=device)


def _make_meta(config: PRTRenderConfig, device: torch.device, tube_count: int, h_terms: int) -> tuple[Tensor, Tensor]:
    _runtime_validate(config)
    tiles_x = (config.width + config.tile_x - 1) // config.tile_x
    tiles_y = (config.height + config.tile_y - 1) // config.tile_y
    tiles_t = (config.frames + config.tile_t - 1) // config.tile_t
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
            tiles_x * tiles_y * tiles_t,
            tube_count,
            config.tile_capacity,
            h_terms,
            0,
        ],
        dtype=torch.int32,
        device=device,
    )
    meta_f32 = torch.tensor(
        [
            config.alpha_threshold,
            config.transmittance_threshold,
            config.background[0],
            config.background[1],
            config.background[2],
            config.depth_epsilon,
            config.max_alpha,
            0.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    return meta_i32, meta_f32


def compile_projective_rational_curve(h_coeff: Tensor, center_t: Tensor, config: PRTRenderConfig) -> Tensor:
    """Compile homogeneous PRT coefficients to per-frame (u, v, depth)."""
    _runtime_validate(config)
    h_coeff = h_coeff.contiguous()
    center_t = center_t.contiguous()
    if h_coeff.ndim != 3 or h_coeff.shape[-1] != 3:
        raise ValueError("h_coeff must have shape [N,H,3]")
    if center_t.shape != (h_coeff.shape[0],):
        raise ValueError(f"center_t must have shape {(h_coeff.shape[0],)}, got {tuple(center_t.shape)}")
    _check_float_tensor("h_coeff", h_coeff, device=h_coeff.device)
    _check_float_tensor("center_t", center_t, device=h_coeff.device)
    frame_times = _frame_times(config, h_coeff.device)
    tau = frame_times.view(-1, 1) - center_t.view(1, -1)
    powers = torch.stack([tau.pow(k) for k in range(int(h_coeff.shape[1]))], dim=-1)
    h = torch.einsum("fnh,nhc->fnc", powers, h_coeff)
    depth = h[..., 2].clamp_min(config.depth_epsilon)
    uv = h[..., :2] / depth.unsqueeze(-1)
    return torch.cat((uv, depth.unsqueeze(-1)), dim=-1).contiguous()


def dense_render_compiled_curve_tubes(
    curve_uv_depth: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: PRTRenderConfig,
) -> Tensor:
    """Slow dense reference path for compiled PRT curves."""
    _runtime_validate(config)
    curve_uv_depth = curve_uv_depth.contiguous()
    lambda_uv = lambda_uv.contiguous()
    lambda_t = lambda_t.contiguous()
    center_t = center_t.contiguous()
    opacity = opacity.contiguous()
    color = color.contiguous()
    _check_compiled_inputs(curve_uv_depth, lambda_uv, lambda_t, center_t, opacity, color, config)

    device = curve_uv_depth.device
    frame_times = _frame_times(config, device)
    bg = torch.tensor(config.background, dtype=torch.float32, device=device)
    out = torch.empty((config.frames, config.height, config.width, 3), dtype=torch.float32, device=device)
    tube_count = int(curve_uv_depth.shape[1])
    if tube_count == 0:
        return bg.view(1, 1, 1, 3).expand_as(out).clone()

    for f in range(config.frames):
        centers = curve_uv_depth[f, :, :2]
        depth = curve_uv_depth[f, :, 2]
        temporal = lambda_t * (frame_times[f] - center_t).square()
        for y in range(config.height):
            py = torch.tensor(float(y) + 0.5, dtype=torch.float32, device=device)
            for x in range(config.width):
                px = torch.tensor(float(x) + 0.5, dtype=torch.float32, device=device)
                du = px - centers[:, 0]
                dv = py - centers[:, 1]
                spatial = lambda_uv[:, 0] * du.square() + 2.0 * lambda_uv[:, 1] * du * dv + lambda_uv[:, 2] * dv.square()
                alpha = torch.clamp(opacity * torch.exp(-0.5 * (spatial + temporal)), max=config.max_alpha)
                active = torch.nonzero(alpha >= config.alpha_threshold, as_tuple=False).flatten()
                if active.numel() == 0:
                    out[f, y, x] = bg
                    continue

                order = active[torch.argsort(depth.index_select(0, active), stable=True)]
                accum = torch.zeros((3,), dtype=torch.float32, device=device)
                transmittance = torch.tensor(1.0, dtype=torch.float32, device=device)
                for tube_id in order.detach().cpu().tolist():
                    ai = alpha[tube_id]
                    accum = accum + transmittance * ai * color[tube_id]
                    transmittance = transmittance * (1.0 - ai)
                    if float(transmittance.detach().cpu()) <= config.transmittance_threshold:
                        break
                out[f, y, x] = accum + transmittance * bg
    return out


def dense_render_projective_rational_tubes(
    h_coeff: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: PRTRenderConfig,
) -> Tensor:
    """Slow dense PRT reference renderer."""
    _runtime_validate(config)
    h_coeff = h_coeff.contiguous()
    lambda_uv = lambda_uv.contiguous()
    lambda_t = lambda_t.contiguous()
    center_t = center_t.contiguous()
    opacity = opacity.contiguous()
    color = color.contiguous()
    _check_projective_inputs(h_coeff, lambda_uv, lambda_t, center_t, opacity, color)
    curve_uv_depth = compile_projective_rational_curve(h_coeff, center_t, config)
    return dense_render_compiled_curve_tubes(curve_uv_depth, lambda_uv, lambda_t, center_t, opacity, color, config)


def metal_render_projective_rational_tubes(
    h_coeff: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: PRTRenderConfig,
) -> Tensor:
    _runtime_validate(config)
    h_coeff = h_coeff.contiguous()
    lambda_uv = lambda_uv.contiguous()
    lambda_t = lambda_t.contiguous()
    center_t = center_t.contiguous()
    opacity = opacity.contiguous()
    color = color.contiguous()
    _check_projective_inputs(h_coeff, lambda_uv, lambda_t, center_t, opacity, color)
    if _C is None or not hasattr(torch.ops, "star_prt_v0"):
        raise RuntimeError("star_prt_v0 custom ops not found. Build the extension first.")
    meta_i32, meta_f32 = _make_meta(config, h_coeff.device, h_coeff.shape[0], h_coeff.shape[1])
    return torch.ops.star_prt_v0.render_projective_rational_tubes(
        h_coeff,
        lambda_uv,
        lambda_t,
        center_t,
        opacity,
        color,
        meta_i32,
        meta_f32,
    )


def metal_render_compiled_curve_tubes(
    curve_uv_depth: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: PRTRenderConfig,
) -> Tensor:
    _runtime_validate(config)
    curve_uv_depth = curve_uv_depth.contiguous()
    lambda_uv = lambda_uv.contiguous()
    lambda_t = lambda_t.contiguous()
    center_t = center_t.contiguous()
    opacity = opacity.contiguous()
    color = color.contiguous()
    _check_compiled_inputs(curve_uv_depth, lambda_uv, lambda_t, center_t, opacity, color, config)
    if _C is None or not hasattr(torch.ops, "star_prt_v0"):
        raise RuntimeError("star_prt_v0 custom ops not found. Build the extension first.")
    meta_i32, meta_f32 = _make_meta(config, curve_uv_depth.device, curve_uv_depth.shape[1], 0)
    return torch.ops.star_prt_v0.render_compiled_curve_tubes(
        curve_uv_depth,
        lambda_uv,
        lambda_t,
        center_t,
        opacity,
        color,
        meta_i32,
        meta_f32,
    )


def metal_compact_backward_projective_rational_tubes(
    h_coeff: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    grad_image: Tensor,
    config: PRTRenderConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    _runtime_validate(config)
    h_coeff = h_coeff.contiguous()
    lambda_uv = lambda_uv.contiguous()
    lambda_t = lambda_t.contiguous()
    center_t = center_t.contiguous()
    opacity = opacity.contiguous()
    color = color.contiguous()
    grad_image = grad_image.contiguous()
    _check_projective_inputs(h_coeff, lambda_uv, lambda_t, center_t, opacity, color)
    _check_grad_image(grad_image, config, h_coeff.device)
    if _C is None or not hasattr(torch.ops, "star_prt_v0"):
        raise RuntimeError("star_prt_v0 custom ops not found. Build the extension first.")
    meta_i32, meta_f32 = _make_meta(config, h_coeff.device, h_coeff.shape[0], h_coeff.shape[1])
    return torch.ops.star_prt_v0.compact_backward_projective_rational_tubes(
        h_coeff,
        lambda_uv,
        lambda_t,
        center_t,
        opacity,
        color,
        grad_image,
        meta_i32,
        meta_f32,
    )


def render_projective_rational_tubes(
    h_coeff: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: PRTRenderConfig,
    *,
    backend: Backend = "auto",
) -> Tensor:
    if backend in ("auto", "dense"):
        return dense_render_projective_rational_tubes(h_coeff, lambda_uv, lambda_t, center_t, opacity, color, config)
    if backend == "metal":
        return metal_render_projective_rational_tubes(h_coeff, lambda_uv, lambda_t, center_t, opacity, color, config)
    raise ValueError(f"unknown backend {backend!r}")


def render_compiled_curve_tubes(
    curve_uv_depth: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: PRTRenderConfig,
    *,
    backend: Backend = "auto",
) -> Tensor:
    if backend in ("auto", "dense"):
        return dense_render_compiled_curve_tubes(curve_uv_depth, lambda_uv, lambda_t, center_t, opacity, color, config)
    if backend == "metal":
        return metal_render_compiled_curve_tubes(curve_uv_depth, lambda_uv, lambda_t, center_t, opacity, color, config)
    raise ValueError(f"unknown backend {backend!r}")
