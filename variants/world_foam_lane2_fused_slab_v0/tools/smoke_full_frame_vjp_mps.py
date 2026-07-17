#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from gate0_beam_toy import ToyConfig, default_sites, linspace, make_boundaries  # noqa: E402
from smoke_composite_strip_mps import write_ppm  # noqa: E402
from smoke_composite_vjp_mps import (  # noqa: E402
    DEFAULT_SITE_RGBA,
    build_segment_tape,
    cpu_loss_outputs_and_grad,
    finite_difference_grad,
    grad_alpha_seed,
    grad_depth_seed,
    grad_rgb_seed,
)
from smoke_composite_vjp_slab_mask_mps import candidate_masks_by_slab  # noqa: E402
from torch_world_foam_lane2_fused_slab import PowerBoundaryConfig, shared_rgba_depth_vjp  # noqa: E402


def image_u_values(*, height: int, width: int, row_u_offset: float) -> list[float]:
    xs = linspace(-1.0, 1.0, width)
    ys = linspace(-1.0, 1.0, height)
    return [x + row_u_offset * y for y in ys for x in xs]


def grad_rgb_full_seed(pixel_index: int, t_index: int, channel: int) -> float:
    return grad_rgb_seed(pixel_index % 97, t_index, channel)


def grad_alpha_full_seed(pixel_index: int, t_index: int) -> float:
    return grad_alpha_seed(pixel_index % 97, t_index)


def grad_depth_full_seed(pixel_index: int, t_index: int) -> float:
    return grad_depth_seed(pixel_index % 97, t_index)


def run_smoke(
    *,
    height: int,
    width: int,
    frames: int,
    time_slabs: int,
    row_u_offset: float,
    timing_iters: int,
    finite_difference_epsilon: float,
    ppm_out: Path | None,
) -> dict[str, Any]:
    if height <= 1 or width <= 1:
        raise ValueError("height and width must be greater than one")
    if frames <= 1:
        raise ValueError("frames must be greater than one")
    if time_slabs <= 0:
        raise ValueError("time_slabs must be positive")
    if timing_iters <= 0:
        raise ValueError("timing_iters must be positive")
    if finite_difference_epsilon <= 0.0:
        raise ValueError("finite_difference_epsilon must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    config = ToyConfig(
        frame_counts=(frames,),
        u_samples=height * width,
        time_slabs=time_slabs,
        near=0.25,
        far=3.0,
        camera_velocity_x=0.35,
        invalid_epsilon=1.0e-7,
    )
    sites = default_sites()
    boundaries = make_boundaries(sites)
    u_values = image_u_values(height=height, width=width, row_u_offset=row_u_offset)
    frame_times = linspace(0.0, 1.0, frames)
    masks, total_candidates, max_candidates_per_slab = candidate_masks_by_slab(
        u_values=u_values,
        boundaries=boundaries,
        config=config,
    )
    rays = build_segment_tape(config=config, u_values=u_values, frame_times=frame_times)
    max_segment_count = max(len(ray) for ray in rays)
    segment_overflow_count = sum(1 for ray in rays if len(ray) > 32)

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
    site_rgba_cpu = torch.tensor(DEFAULT_SITE_RGBA, dtype=torch.float32)
    site_rgba_f32 = site_rgba_cpu.to(device)
    beam_f32 = torch.tensor(
        [[u, 0.0, 1.0, config.near, config.far] for u in u_values],
        dtype=torch.float32,
        device=device,
    )
    frame_t_f32 = torch.tensor(frame_times, dtype=torch.float32, device=device)
    grad_rgb_cpu = torch.tensor(
        [
            [
                [grad_rgb_full_seed(pixel_index, t_index, channel) for channel in range(3)]
                for t_index in range(frames)
            ]
            for pixel_index in range(height * width)
        ],
        dtype=torch.float32,
    )
    grad_alpha_cpu = torch.tensor(
        [[grad_alpha_full_seed(pixel_index, t_index) for t_index in range(frames)] for pixel_index in range(height * width)],
        dtype=torch.float32,
    )
    grad_depth_cpu = torch.tensor(
        [[grad_depth_full_seed(pixel_index, t_index) for t_index in range(frames)] for pixel_index in range(height * width)],
        dtype=torch.float32,
    )
    grad_rgb_f32 = grad_rgb_cpu.to(device)
    grad_alpha_f32 = grad_alpha_cpu.to(device)
    grad_depth_f32 = grad_depth_cpu.to(device)
    op_config = PowerBoundaryConfig(camera_velocity_x=config.camera_velocity_x, invalid_epsilon=config.invalid_epsilon)

    output_rgb, output_alpha, output_depth, grad_samples_rgba = shared_rgba_depth_vjp(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgba_f32,
        beam_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        op_config,
    )
    torch.mps.synchronize()
    mps_rgb = output_rgb.cpu()
    mps_alpha = output_alpha.cpu()
    mps_depth = output_depth.cpu()
    mps_grad_rgba = grad_samples_rgba.sum(dim=(0, 1)).cpu()
    mps_loss = float(
        (
            (output_rgb * grad_rgb_f32).sum()
            + (output_alpha * grad_alpha_f32).sum()
            + (output_depth * grad_depth_f32).sum()
        )
        .cpu()
        .item()
    )

    started_at = time.perf_counter()
    timed = (output_rgb, output_alpha, output_depth, grad_samples_rgba)
    for _ in range(timing_iters):
        timed = shared_rgba_depth_vjp(
            boundary_f32,
            candidate_mask_u32,
            sites_f32,
            site_rgba_f32,
            beam_f32,
            frame_t_f32,
            grad_rgb_f32,
            grad_alpha_f32,
            grad_depth_f32,
            op_config,
        )
    torch.mps.synchronize()
    _timed_shapes = tuple(tensor.shape for tensor in timed)
    wall_clock_ms = (time.perf_counter() - started_at) * 1000.0 / float(timing_iters)

    cpu_loss, cpu_rgb, cpu_alpha, cpu_depth, cpu_grad_rgba = cpu_loss_outputs_and_grad(
        site_rgba_values=site_rgba_cpu,
        rays=rays,
        grad_rgb=grad_rgb_cpu,
        grad_alpha=grad_alpha_cpu,
        grad_depth=grad_depth_cpu,
        far_depth=config.far,
    )
    finite_difference = finite_difference_grad(
        site_rgba_values=site_rgba_cpu,
        rays=rays,
        grad_rgb=grad_rgb_cpu,
        grad_alpha=grad_alpha_cpu,
        grad_depth=grad_depth_cpu,
        far_depth=config.far,
        epsilon=finite_difference_epsilon,
    )

    max_rgb_abs_error = float((mps_rgb - cpu_rgb).abs().max().item())
    max_alpha_abs_error = float((mps_alpha - cpu_alpha).abs().max().item())
    max_depth_abs_error = float((mps_depth - cpu_depth).abs().max().item())
    max_grad_abs_error = float((mps_grad_rgba - cpu_grad_rgba).abs().max().item())
    finite_difference_max_abs_error = float((finite_difference - cpu_grad_rgba).abs().max().item())
    loss_abs_error = abs(mps_loss - cpu_loss)

    image_rgb = mps_rgb.transpose(0, 1).reshape(frames, height, width, 3)
    image_alpha = mps_alpha.transpose(0, 1).reshape(frames, height, width)
    image_depth = mps_depth.transpose(0, 1).reshape(frames, height, width)
    if ppm_out is not None:
        write_ppm(ppm_out, image_rgb[0])

    tolerance = 4.0e-4
    fd_tolerance = 2.0e-3
    direct_forward_boundary_scans = len(boundaries) * height * width * frames
    shared_forward_boundary_scans = len(boundaries) * height * width * time_slabs
    acceptance = {
        "rgb_shape_is_full_frame_sequence": list(image_rgb.shape) == [frames, height, width, 3],
        "alpha_shape_is_full_frame_sequence": list(image_alpha.shape) == [frames, height, width],
        "depth_shape_is_full_frame_sequence": list(image_depth.shape) == [frames, height, width],
        "all_outputs_finite": bool(
            torch.isfinite(image_rgb).all().item()
            and torch.isfinite(image_alpha).all().item()
            and torch.isfinite(image_depth).all().item()
        ),
        "image_has_horizontal_variation": float(image_rgb[:, :, 1:, :].sub(image_rgb[:, :, :-1, :]).abs().max().item()) > 0.0,
        "image_has_vertical_variation": float(image_rgb[:, 1:, :, :].sub(image_rgb[:, :-1, :, :]).abs().max().item()) > 0.0,
        "outputs_match_cpu_reference": max(max_rgb_abs_error, max_alpha_abs_error, max_depth_abs_error) <= tolerance,
        "loss_matches_cpu_reference": loss_abs_error <= tolerance,
        "rgba_gradients_match_cpu_reference": max_grad_abs_error <= tolerance,
        "finite_difference_matches_cpu_gradient": finite_difference_max_abs_error <= fd_tolerance,
        "candidate_mask_is_slab_indexed": len(masks) == height * width * time_slabs,
        "segment_count_within_metal_limit": segment_overflow_count == 0,
        "uses_mps_shared_rgba_depth_vjp": True,
    }
    return {
        "benchmark": "world_foam_lane2_gate1_mps_full_frame_vjp_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "device": "mps",
        "gate": "1_image_shape",
        "renderer_scope": "orthographic_2d_time_full_frame_image_smoke_no_real_camera_ray_projection",
        "projection_scope": "v_axis_extruded_by_row_dependent_u_offset_current_cells_remain_x_z_t_only",
        "gradient_scope": "fixed_segment_site_rgba_only_no_geometry_or_topology_gradients",
        "height": height,
        "width": width,
        "frames": frames,
        "time_slabs": time_slabs,
        "row_u_offset": row_u_offset,
        "rgb_shape": list(image_rgb.shape),
        "alpha_shape": list(image_alpha.shape),
        "depth_shape": list(image_depth.shape),
        "ppm_out": str(ppm_out) if ppm_out is not None else None,
        "acceptance": acceptance,
        "max_rgb_abs_error": max_rgb_abs_error,
        "max_alpha_abs_error": max_alpha_abs_error,
        "max_depth_abs_error": max_depth_abs_error,
        "max_rgba_gradient_abs_error": max_grad_abs_error,
        "finite_difference_max_abs_error": finite_difference_max_abs_error,
        "loss_abs_error": loss_abs_error,
        "mps_full_frame_vjp_wall_clock_ms": wall_clock_ms,
        "timing_iters": timing_iters,
        "tolerance": tolerance,
        "finite_difference_tolerance": fd_tolerance,
        "finite_difference_epsilon": finite_difference_epsilon,
        "pixel_count": height * width,
        "pixel_ray_count": height * width * frames,
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "candidate_mask_shape": [height * width, time_slabs],
        "total_candidates": total_candidates,
        "max_candidates_per_slab": max_candidates_per_slab,
        "max_segment_count": max_segment_count,
        "segment_overflow_count": segment_overflow_count,
        "direct_forward_boundary_scans": direct_forward_boundary_scans,
        "shared_forward_boundary_scans": shared_forward_boundary_scans,
        "shared_forward_boundary_scan_ratio": shared_forward_boundary_scans
        / float(max(direct_forward_boundary_scans, 1)),
        "rgb_min": float(image_rgb.min().item()),
        "rgb_max": float(image_rgb.max().item()),
        "rgb_std": float(image_rgb.std().item()),
        "alpha_min": float(image_alpha.min().item()),
        "alpha_max": float(image_alpha.max().item()),
        "alpha_std": float(image_alpha.std().item()),
        "depth_min": float(image_depth.min().item()),
        "depth_max": float(image_depth.max().item()),
        "depth_std": float(image_depth.std().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke World Foam image-shaped MPS compositor VJP.")
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=17)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--time-slabs", type=int, default=2)
    parser.add_argument("--row-u-offset", type=float, default=0.25)
    parser.add_argument("--timing-iters", type=int, default=20)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--ppm-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke(
        height=args.height,
        width=args.width,
        frames=args.frames,
        time_slabs=args.time_slabs,
        row_u_offset=args.row_u_offset,
        timing_iters=args.timing_iters,
        finite_difference_epsilon=args.finite_difference_epsilon,
        ppm_out=args.ppm_out,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
