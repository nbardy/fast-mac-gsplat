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

from gate0_beam_toy import ToyConfig, default_sites, linspace, make_boundaries, slab_events, slab_ranges  # noqa: E402
from smoke_composite_vjp_mps import (  # noqa: E402
    DEFAULT_SITE_RGBA,
    build_segment_tape,
    cpu_loss_outputs_and_grad,
    finite_difference_grad,
    grad_alpha_seed,
    grad_depth_seed,
    grad_rgb_seed,
)
from torch_world_foam_lane2_fused_slab import PowerBoundaryConfig, shared_rgba_depth_vjp  # noqa: E402


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("all integers must be positive")
    return values


def candidate_masks_by_slab(
    *,
    u_values: list[float],
    boundaries: tuple[Any, ...],
    config: ToyConfig,
) -> tuple[list[int], int, int]:
    boundary_index = {
        (boundary.left, boundary.right): idx
        for idx, boundary in enumerate(boundaries)
    }
    if len(boundary_index) > 31:
        raise ValueError("slab-mask smoke supports at most 31 boundaries")

    masks: list[int] = []
    total_candidates = 0
    max_candidates_per_slab = 0
    for u in u_values:
        for t0, t1 in slab_ranges(config.time_slabs):
            events, invalid = slab_events(
                boundaries,
                u=u,
                t0=t0,
                t1=t1,
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
            candidate_count = len(events)
            total_candidates += candidate_count
            max_candidates_per_slab = max(max_candidates_per_slab, candidate_count)
    return masks, total_candidates, max_candidates_per_slab


def run_one(
    *,
    time_slabs: int,
    timing_iters: int,
    finite_difference_epsilon: float,
) -> dict[str, Any]:
    config = ToyConfig(
        frame_counts=(16,),
        u_samples=17,
        time_slabs=time_slabs,
        near=0.25,
        far=3.0,
        camera_velocity_x=0.35,
        invalid_epsilon=1.0e-7,
    )
    sites = default_sites()
    boundaries = make_boundaries(sites)
    u_values = linspace(-1.0, 1.0, config.u_samples)
    frame_times = linspace(0.0, 1.0, 16)
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
    tolerance = 2.0e-4
    fd_tolerance = 2.0e-3
    direct_forward_boundary_scans = len(boundaries) * len(u_values) * len(frame_times)
    shared_forward_boundary_scans = len(boundaries) * len(u_values) * config.time_slabs
    acceptance = {
        "outputs_match_cpu_reference": max(max_rgb_abs_error, max_alpha_abs_error, max_depth_abs_error) <= tolerance,
        "loss_matches_cpu_reference": loss_abs_error <= tolerance,
        "rgba_gradients_match_cpu_reference": max_grad_abs_error <= tolerance,
        "finite_difference_matches_cpu_gradient": finite_difference_max_abs_error <= fd_tolerance,
        "candidate_mask_is_slab_indexed": len(masks) == len(u_values) * time_slabs,
        "segment_count_within_metal_limit": segment_overflow_count == 0,
        "uses_mps_shared_rgba_depth_vjp": True,
        "gradient_scope_is_fixed_segment_rgba_only": True,
    }
    return {
        "time_slabs": time_slabs,
        "status": "ok" if all(acceptance.values()) else "failed",
        "acceptance": acceptance,
        "candidate_mask_shape": [len(u_values), time_slabs],
        "total_candidates": total_candidates,
        "max_candidates_per_slab": max_candidates_per_slab,
        "max_segment_count": max_segment_count,
        "segment_overflow_count": segment_overflow_count,
        "max_rgb_abs_error": max_rgb_abs_error,
        "max_alpha_abs_error": max_alpha_abs_error,
        "max_depth_abs_error": max_depth_abs_error,
        "max_rgba_gradient_abs_error": max_grad_abs_error,
        "finite_difference_max_abs_error": finite_difference_max_abs_error,
        "loss_abs_error": loss_abs_error,
        "mps_composite_vjp_wall_clock_ms": wall_clock_ms,
        "direct_forward_boundary_scans": direct_forward_boundary_scans,
        "shared_forward_boundary_scans": shared_forward_boundary_scans,
        "shared_forward_boundary_scan_ratio": shared_forward_boundary_scans
        / float(max(direct_forward_boundary_scans, 1)),
    }


def run_smoke(
    *,
    time_slabs: tuple[int, ...],
    timing_iters: int,
    finite_difference_epsilon: float,
) -> dict[str, Any]:
    if timing_iters <= 0:
        raise ValueError("timing_iters must be positive")
    if finite_difference_epsilon <= 0.0:
        raise ValueError("finite_difference_epsilon must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    rows = [
        run_one(
            time_slabs=value,
            timing_iters=timing_iters,
            finite_difference_epsilon=finite_difference_epsilon,
        )
        for value in time_slabs
    ]
    return {
        "benchmark": "world_foam_lane2_gate0_95_mps_composite_vjp_slab_mask_smoke",
        "status": "ok" if all(row["status"] == "ok" for row in rows) else "failed",
        "device": "mps",
        "gate": "0.95",
        "renderer_scope": "rgb_alpha_depth_composite_strip_with_slab_indexed_bitmask_candidates",
        "gradient_scope": "fixed_segment_site_rgba_only_no_geometry_or_topology_gradients",
        "frames": 16,
        "width": 17,
        "site_count": 5,
        "boundary_count": 10,
        "time_slabs": list(time_slabs),
        "timing_iters": timing_iters,
        "finite_difference_epsilon": finite_difference_epsilon,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke World Foam slab-indexed mask compositor VJP.")
    parser.add_argument("--time-slabs", default="1,2,4")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--timing-iters", type=int, default=20)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke(
        time_slabs=parse_int_list(args.time_slabs),
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
