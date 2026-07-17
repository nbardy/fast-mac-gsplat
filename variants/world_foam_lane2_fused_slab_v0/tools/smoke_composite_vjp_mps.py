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

from gate0_beam_toy import ToyConfig, default_sites, linspace, make_boundaries  # noqa: E402
from gate0_shared_forward_backward import (  # noqa: E402
    build_shared_slab_cache,
    make_boundary_lookup,
    render_ray_from_candidates,
)
from smoke_composite_strip_mps import DEFAULT_SITE_RGBA, candidate_masks  # noqa: E402
from torch_world_foam_lane2_fused_slab import PowerBoundaryConfig, shared_rgba_depth_vjp  # noqa: E402


def grad_rgb_seed(u_index: int, t_index: int, channel: int) -> float:
    return 0.05 + 0.0075 * float(u_index + 1) + 0.011 * float(t_index + 1) + 0.013 * float(channel + 1)


def grad_alpha_seed(u_index: int, t_index: int) -> float:
    return 0.03 + 0.004 * float(u_index + 1) - 0.0025 * float(t_index + 1)


def grad_depth_seed(u_index: int, t_index: int) -> float:
    return -0.015 + 0.002 * float(u_index + 1) + 0.00125 * float(t_index + 1)


def build_segment_tape(
    *,
    config: ToyConfig,
    u_values: list[float],
    frame_times: list[float],
) -> list[list[tuple[int, float, float]]]:
    sites = default_sites()
    boundaries = make_boundaries(sites)
    boundary_lookup = make_boundary_lookup(boundaries)
    slab_cache, _invalid = build_shared_slab_cache(
        u_values=u_values,
        boundaries=boundaries,
        config=config,
    )
    site_signals = tuple(0.0 for _ in sites)
    rays: list[list[tuple[int, float, float]]] = []
    for u in u_values:
        for t in frame_times:
            slab_index = min(int(math.floor(t * config.time_slabs)), config.time_slabs - 1)
            tape = render_ray_from_candidates(
                sites=sites,
                boundary_lookup=boundary_lookup,
                candidate_events=slab_cache[(u, slab_index)],
                site_signals=site_signals,
                u=u,
                t=t,
                near=config.near,
                far=config.far,
                camera_velocity_x=config.camera_velocity_x,
                slab_index=slab_index,
                grad_output=0.0,
            )
            rays.append(
                [
                    (segment.site_id, segment.length, 0.5 * (segment.depth0 + segment.depth1))
                    for segment in tape.segments
                ]
            )
    return rays


def cpu_loss_outputs_and_grad(
    *,
    site_rgba_values: torch.Tensor,
    rays: list[list[tuple[int, float, float]]],
    grad_rgb: torch.Tensor,
    grad_alpha: torch.Tensor,
    grad_depth: torch.Tensor,
    far_depth: float,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    site_rgba = site_rgba_values.clone().detach().to(torch.float64).requires_grad_(True)
    rgb_rows = []
    alpha_rows = []
    depth_rows = []
    loss = torch.zeros((), dtype=torch.float64)
    ray_id = 0
    for beam_id in range(grad_alpha.shape[0]):
        rgb_row = []
        alpha_row = []
        depth_row = []
        for frame_id in range(grad_alpha.shape[1]):
            rgb_accum = torch.zeros(3, dtype=torch.float64)
            alpha_accum = torch.zeros((), dtype=torch.float64)
            depth_weighted = torch.zeros((), dtype=torch.float64)
            transmittance = torch.ones((), dtype=torch.float64)
            for site_id, length, mid_depth in rays[ray_id]:
                if float(transmittance.detach().item()) <= 1.0e-5:
                    break
                if length <= 1.0e-8:
                    continue
                rgba = site_rgba[site_id]
                density = torch.clamp(rgba[3], min=0.0)
                segment_transmittance = torch.exp(-density * float(length))
                segment_alpha = 1.0 - segment_transmittance
                weight = transmittance * segment_alpha
                rgb_accum = rgb_accum + weight * rgba[:3]
                alpha_accum = alpha_accum + weight
                depth_weighted = depth_weighted + weight * float(mid_depth)
                transmittance = transmittance * segment_transmittance
            depth_value = depth_weighted / alpha_accum if float(alpha_accum.detach().item()) > 1.0e-8 else torch.tensor(far_depth, dtype=torch.float64)
            rgb_row.append(rgb_accum)
            alpha_row.append(alpha_accum)
            depth_row.append(depth_value)
            loss = loss + (rgb_accum * grad_rgb[beam_id, frame_id].to(torch.float64)).sum()
            loss = loss + alpha_accum * grad_alpha[beam_id, frame_id].to(torch.float64)
            loss = loss + depth_value * grad_depth[beam_id, frame_id].to(torch.float64)
            ray_id += 1
        rgb_rows.append(torch.stack(rgb_row))
        alpha_rows.append(torch.stack(alpha_row))
        depth_rows.append(torch.stack(depth_row))
    loss.backward()
    return (
        float(loss.detach().item()),
        torch.stack(rgb_rows).detach().to(torch.float32),
        torch.stack(alpha_rows).detach().to(torch.float32),
        torch.stack(depth_rows).detach().to(torch.float32),
        site_rgba.grad.detach().to(torch.float32),
    )


def cpu_loss_only(
    *,
    site_rgba_values: torch.Tensor,
    rays: list[list[tuple[int, float, float]]],
    grad_rgb: torch.Tensor,
    grad_alpha: torch.Tensor,
    grad_depth: torch.Tensor,
    far_depth: float,
) -> float:
    loss, _rgb, _alpha, _depth, _grad = cpu_loss_outputs_and_grad(
        site_rgba_values=site_rgba_values,
        rays=rays,
        grad_rgb=grad_rgb,
        grad_alpha=grad_alpha,
        grad_depth=grad_depth,
        far_depth=far_depth,
    )
    return loss


def finite_difference_grad(
    *,
    site_rgba_values: torch.Tensor,
    rays: list[list[tuple[int, float, float]]],
    grad_rgb: torch.Tensor,
    grad_alpha: torch.Tensor,
    grad_depth: torch.Tensor,
    far_depth: float,
    epsilon: float,
) -> torch.Tensor:
    out = torch.empty_like(site_rgba_values, dtype=torch.float32)
    for site_id in range(site_rgba_values.shape[0]):
        for channel in range(site_rgba_values.shape[1]):
            plus = site_rgba_values.clone()
            minus = site_rgba_values.clone()
            plus[site_id, channel] += epsilon
            minus[site_id, channel] -= epsilon
            loss_plus = cpu_loss_only(
                site_rgba_values=plus,
                rays=rays,
                grad_rgb=grad_rgb,
                grad_alpha=grad_alpha,
                grad_depth=grad_depth,
                far_depth=far_depth,
            )
            loss_minus = cpu_loss_only(
                site_rgba_values=minus,
                rays=rays,
                grad_rgb=grad_rgb,
                grad_alpha=grad_alpha,
                grad_depth=grad_depth,
                far_depth=far_depth,
            )
            out[site_id, channel] = (loss_plus - loss_minus) / (2.0 * epsilon)
    return out


def run_smoke(*, timing_iters: int, finite_difference_epsilon: float) -> dict[str, Any]:
    if timing_iters <= 0:
        raise ValueError("timing_iters must be positive")
    if finite_difference_epsilon <= 0.0:
        raise ValueError("finite_difference_epsilon must be positive")
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
    rays = build_segment_tape(config=config, u_values=u_values, frame_times=frame_times)

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
                [grad_rgb_seed(u_index, t_index, channel) for channel in range(3)]
                for t_index in range(len(frame_times))
            ]
            for u_index in range(len(u_values))
        ],
        dtype=torch.float32,
    )
    grad_alpha_cpu = torch.tensor(
        [[grad_alpha_seed(u_index, t_index) for t_index in range(len(frame_times))] for u_index in range(len(u_values))],
        dtype=torch.float32,
    )
    grad_depth_cpu = torch.tensor(
        [[grad_depth_seed(u_index, t_index) for t_index in range(len(frame_times))] for u_index in range(len(u_values))],
        dtype=torch.float32,
    )
    grad_rgb_f32 = grad_rgb_cpu.to(device)
    grad_alpha_f32 = grad_alpha_cpu.to(device)
    grad_depth_f32 = grad_depth_cpu.to(device)

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
        PowerBoundaryConfig(camera_velocity_x=config.camera_velocity_x, invalid_epsilon=config.invalid_epsilon),
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
            PowerBoundaryConfig(camera_velocity_x=config.camera_velocity_x, invalid_epsilon=config.invalid_epsilon),
        )
    torch.mps.synchronize()
    _timed_shapes = tuple(tensor.shape for tensor in timed)
    mps_composite_vjp_wall_clock_ms = (time.perf_counter() - started_at) * 1000.0 / float(timing_iters)

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
    tolerance = 2.0e-4
    fd_tolerance = 2.0e-3
    acceptance = {
        "outputs_match_cpu_reference": max(max_rgb_abs_error, max_alpha_abs_error, max_depth_abs_error) <= tolerance,
        "loss_matches_cpu_reference": loss_abs_error <= tolerance,
        "rgba_gradients_match_cpu_reference": max_grad_abs_error <= tolerance,
        "finite_difference_matches_cpu_gradient": finite_difference_max_abs_error <= fd_tolerance,
        "uses_mps_shared_rgba_depth_vjp": True,
        "gradient_scope_is_fixed_segment_rgba_only": True,
    }
    direct_forward_boundary_scans = len(boundaries) * len(u_values) * len(frame_times)
    shared_forward_boundary_scans = len(boundaries) * len(u_values) * config.time_slabs
    return {
        "benchmark": "world_foam_lane2_gate0_9_mps_composite_vjp_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "device": "mps",
        "gate": "0.9",
        "renderer_scope": "rgb_alpha_depth_composite_strip",
        "gradient_scope": "fixed_segment_site_rgba_only_no_geometry_or_topology_gradients",
        "acceptance": acceptance,
        "max_rgb_abs_error": max_rgb_abs_error,
        "max_alpha_abs_error": max_alpha_abs_error,
        "max_depth_abs_error": max_depth_abs_error,
        "max_rgba_gradient_abs_error": max_grad_abs_error,
        "finite_difference_max_abs_error": finite_difference_max_abs_error,
        "loss_abs_error": loss_abs_error,
        "mps_composite_vjp_wall_clock_ms": mps_composite_vjp_wall_clock_ms,
        "timing_iters": timing_iters,
        "tolerance": tolerance,
        "finite_difference_tolerance": fd_tolerance,
        "finite_difference_epsilon": finite_difference_epsilon,
        "frames": len(frame_times),
        "width": len(u_values),
        "pixel_ray_count": len(frame_times) * len(u_values),
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "direct_forward_boundary_scans": direct_forward_boundary_scans,
        "shared_forward_boundary_scans": shared_forward_boundary_scans,
        "shared_forward_boundary_scan_ratio": shared_forward_boundary_scans
        / float(max(direct_forward_boundary_scans, 1)),
        "gradient_shape": list(mps_grad_rgba.shape),
        "cpu_rgba_gradient": cpu_grad_rgba.tolist(),
        "mps_rgba_gradient": mps_grad_rgba.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke the World Foam Lane 2 fixed-segment compositor VJP.")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--timing-iters", type=int, default=20)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke(
        timing_iters=args.timing_iters,
        finite_difference_epsilon=args.finite_difference_epsilon,
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
