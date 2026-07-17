#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DYNAWORLD = ROOT.parents[3]
WORLD_FOAM_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORLD_FOAM_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_FOAM_DIR))

from gate0_beam_toy import ToyConfig, default_sites, linspace, make_boundaries, slab_events  # noqa: E402
from gate0_shared_forward_backward import (  # noqa: E402
    build_shared_slab_cache,
    gradient_seed,
    make_boundary_lookup,
    render_ray_from_candidates,
)
from torch_world_foam_lane2_fused_slab import PowerBoundaryConfig, shared_rgb_replay  # noqa: E402


DEFAULT_SITE_RGB = (
    (0.90, 0.18, 0.10),
    (0.10, 0.75, 0.24),
    (0.18, 0.36, 0.96),
    (0.95, 0.80, 0.15),
    (0.70, 0.22, 0.88),
)


def candidate_masks(
    *,
    u_values: list[float],
    boundaries: tuple[Any, ...],
    config: ToyConfig,
) -> list[int]:
    boundary_index = {
        (boundary.left, boundary.right): idx
        for idx, boundary in enumerate(boundaries)
    }
    if len(boundary_index) > 31:
        raise ValueError("Gate 0.7 mask smoke supports at most 31 boundaries")
    masks: list[int] = []
    for u in u_values:
        events, invalid = slab_events(
            boundaries,
            u=u,
            t0=0.0,
            t1=1.0,
            near=config.near,
            far=config.far,
            camera_velocity_x=config.camera_velocity_x,
            invalid_epsilon=config.invalid_epsilon,
        )
        if invalid:
            raise ValueError(f"unexpected invalid denominators for u={u}: {invalid}")
        mask = 0
        for event in events:
            mask |= 1 << boundary_index[event]
        masks.append(mask)
    return masks


def grad_seed(u_index: int, t_index: int, channel: int) -> float:
    return gradient_seed(u_index, t_index) * (1.0 + 0.125 * float(channel))


def cpu_rgb_reference(
    *,
    config: ToyConfig,
    u_values: list[float],
    frame_times: list[float],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    sites = default_sites()
    boundaries = make_boundaries(sites)
    boundary_lookup = make_boundary_lookup(boundaries)
    slab_cache, _invalid = build_shared_slab_cache(
        u_values=u_values,
        boundaries=boundaries,
        config=config,
    )
    image = torch.empty((len(frame_times), len(u_values), 3), dtype=torch.float32)
    color_grad = torch.zeros((len(sites), 3), dtype=torch.float32)
    loss = 0.0
    for channel in range(3):
        signals = tuple(rgb[channel] for rgb in DEFAULT_SITE_RGB)
        for u_index, u in enumerate(u_values):
            for t_index, t in enumerate(frame_times):
                slab_index = min(int(math.floor(t * config.time_slabs)), config.time_slabs - 1)
                grad_output = grad_seed(u_index, t_index, channel)
                tape = render_ray_from_candidates(
                    sites=sites,
                    boundary_lookup=boundary_lookup,
                    candidate_events=slab_cache[(u, slab_index)],
                    site_signals=signals,
                    u=u,
                    t=t,
                    near=config.near,
                    far=config.far,
                    camera_velocity_x=config.camera_velocity_x,
                    slab_index=slab_index,
                    grad_output=grad_output,
                )
                image[t_index, u_index, channel] = tape.output
                loss += tape.output * grad_output
                for segment in tape.segments:
                    color_grad[segment.site_id, channel] += grad_output * segment.length
    return image, color_grad, loss


def write_ppm(path: Path, image: torch.Tensor) -> None:
    normalized = image - image.min()
    scale = float(normalized.max().item())
    if scale > 0.0:
        normalized = normalized / scale
    pixels = (normalized.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    height, width, _channels = pixels.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(pixels.numpy().tobytes())


def run_smoke(*, timing_iters: int, ppm_out: Path | None) -> dict[str, Any]:
    if timing_iters <= 0:
        raise ValueError("timing_iters must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    config = ToyConfig(
        frame_counts=(16,),
        u_samples=17,
        time_slabs=1,
        near=0.25,
        far=3.0,
        camera_velocity_x=0.35,
        invalid_epsilon=1.0e-7,
    )
    sites = default_sites()
    boundaries = make_boundaries(sites)
    u_values = linspace(-1.0, 1.0, config.u_samples)
    frame_times = linspace(0.0, 1.0, 16)
    masks = candidate_masks(u_values=u_values, boundaries=boundaries, config=config)

    device = torch.device("mps")
    boundary_f32 = torch.tensor(
        [[boundary.nx, boundary.nz, boundary.nt, boundary.b] for boundary in boundaries],
        dtype=torch.float32,
        device=device,
    )
    candidate_mask_u32 = torch.tensor(masks, dtype=torch.int32, device=device)
    sites_f32 = torch.tensor(
        [[site.x, site.z, site.t, site.weight] for site in sites],
        dtype=torch.float32,
        device=device,
    )
    beam_f32 = torch.tensor(
        [[u, 0.0, 1.0, config.near, config.far] for u in u_values],
        dtype=torch.float32,
        device=device,
    )
    frame_t_f32 = torch.tensor(frame_times, dtype=torch.float32, device=device)

    site_rgb_f32 = torch.tensor(DEFAULT_SITE_RGB, dtype=torch.float32, device=device)
    grad_output_rgb = torch.tensor(
        [
            [
                [grad_seed(u_index, t_index, channel) for channel in range(3)]
                for t_index in range(len(frame_times))
            ]
            for u_index in range(len(u_values))
        ],
        dtype=torch.float32,
        device=device,
    )
    output_rgb, grad_samples_rgb = shared_rgb_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgb_f32,
        beam_f32,
        frame_t_f32,
        grad_output_rgb,
        PowerBoundaryConfig(camera_velocity_x=config.camera_velocity_x, invalid_epsilon=config.invalid_epsilon),
    )
    torch.mps.synchronize()
    mps_image = output_rgb.transpose(0, 1).cpu()
    mps_color_grad = grad_samples_rgb.sum(dim=(0, 1)).cpu()
    mps_loss = float((output_rgb * grad_output_rgb).sum().cpu().item())

    started_at = time.perf_counter()
    timed_output = output_rgb
    timed_grad_samples = grad_samples_rgb
    for _ in range(timing_iters):
        timed_output, timed_grad_samples = shared_rgb_replay(
            boundary_f32,
            candidate_mask_u32,
            sites_f32,
            site_rgb_f32,
            beam_f32,
            frame_t_f32,
            grad_output_rgb,
            PowerBoundaryConfig(camera_velocity_x=config.camera_velocity_x, invalid_epsilon=config.invalid_epsilon),
        )
    torch.mps.synchronize()
    _timed_shapes = (timed_output.shape, timed_grad_samples.shape)
    mps_rgb_strip_wall_clock_ms = (time.perf_counter() - started_at) * 1000.0 / float(timing_iters)

    cpu_image, cpu_color_grad, cpu_loss = cpu_rgb_reference(
        config=config,
        u_values=u_values,
        frame_times=frame_times,
    )
    max_rgb_abs_error = float((mps_image - cpu_image).abs().max().item())
    mean_rgb_abs_error = float((mps_image - cpu_image).abs().mean().item())
    color_gradient_max_abs_error = float((mps_color_grad - cpu_color_grad).abs().max().item())
    loss_abs_error = abs(mps_loss - cpu_loss)
    if ppm_out is not None:
        write_ppm(ppm_out, mps_image)

    tolerance = 1.0e-4
    finite_mask = torch.isfinite(mps_image)
    nan_pixel_count = int(torch.isnan(mps_image).sum().item())
    inf_pixel_count = int(torch.isinf(mps_image).sum().item())
    finite_pixel_count = int(finite_mask.sum().item())
    acceptance = {
        "output_shape_is_rgb_strip": list(mps_image.shape) == [len(frame_times), len(u_values), 3],
        "output_is_float32": mps_image.dtype == torch.float32,
        "all_pixels_finite": bool(finite_mask.all().item()),
        "rgb_nonconstant": float(mps_image.std().item()) > 0.0,
        "matches_cpu_reference": max_rgb_abs_error <= tolerance,
        "color_gradient_matches_cpu_reference": color_gradient_max_abs_error <= tolerance,
        "loss_matches_cpu_reference": loss_abs_error <= tolerance,
        "uses_mps_shared_replay": True,
        "uses_mps_shared_rgb_replay": True,
    }
    return {
        "benchmark": "world_foam_lane2_gate0_7_mps_rgb_strip_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "device": "mps",
        "gate": "0.7",
        "renderer_scope": "rgb_strip_single_mps_shared_replay_no_alpha_depth_geometry_gradients_or_trainer",
        "gradient_scope": "mps_site_rgb_signal_only_fixed_segments_geometry_gradients_not_implemented",
        "strip_image_shape": list(mps_image.shape),
        "frames": len(frame_times),
        "height": 1,
        "width": len(u_values),
        "pixel_ray_count": len(frame_times) * len(u_values),
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "dtype": "float32",
        "ppm_out": str(ppm_out) if ppm_out is not None else None,
        "acceptance": acceptance,
        "max_rgb_abs_error": max_rgb_abs_error,
        "mean_rgb_abs_error": mean_rgb_abs_error,
        "color_gradient_max_abs_error": color_gradient_max_abs_error,
        "loss_abs_error": loss_abs_error,
        "rgb_min": float(mps_image.min().item()),
        "rgb_max": float(mps_image.max().item()),
        "rgb_std": float(mps_image.std().item()),
        "finite_pixel_count": finite_pixel_count,
        "nan_pixel_count": nan_pixel_count,
        "inf_pixel_count": inf_pixel_count,
        "mps_rgb_strip_wall_clock_ms": mps_rgb_strip_wall_clock_ms,
        "timing_iters": timing_iters,
        "tolerance": tolerance,
        "shared_forward_backward_boundary_scan_ratio": 0.03125,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke the World Foam Lane 2 MPS RGB strip renderer.")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--ppm-out", type=Path)
    parser.add_argument("--timing-iters", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke(timing_iters=args.timing_iters, ppm_out=args.ppm_out)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
