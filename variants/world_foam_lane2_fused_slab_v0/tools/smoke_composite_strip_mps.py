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
    make_boundary_lookup,
    render_ray_from_candidates,
)
from torch_world_foam_lane2_fused_slab import PowerBoundaryConfig, shared_rgba_depth_replay  # noqa: E402


DEFAULT_SITE_RGBA = (
    (0.90, 0.18, 0.10, 0.55),
    (0.10, 0.75, 0.24, 0.80),
    (0.18, 0.36, 0.96, 0.42),
    (0.95, 0.80, 0.15, 0.65),
    (0.70, 0.22, 0.88, 0.36),
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
        raise ValueError("Gate 0.8 mask smoke supports at most 31 boundaries")
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


def cpu_composite_reference(
    *,
    config: ToyConfig,
    u_values: list[float],
    frame_times: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sites = default_sites()
    boundaries = make_boundaries(sites)
    boundary_lookup = make_boundary_lookup(boundaries)
    slab_cache, _invalid = build_shared_slab_cache(
        u_values=u_values,
        boundaries=boundaries,
        config=config,
    )
    rgb = torch.empty((len(frame_times), len(u_values), 3), dtype=torch.float32)
    alpha = torch.empty((len(frame_times), len(u_values)), dtype=torch.float32)
    depth = torch.empty((len(frame_times), len(u_values)), dtype=torch.float32)
    site_signals = tuple(0.0 for _ in sites)
    for u_index, u in enumerate(u_values):
        for t_index, t in enumerate(frame_times):
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
            rgb_accum = torch.zeros(3, dtype=torch.float32)
            alpha_accum = 0.0
            depth_weighted = 0.0
            transmittance = 1.0
            for segment in tape.segments:
                if transmittance <= 1.0e-5:
                    break
                length = segment.length
                if length <= 1.0e-8:
                    continue
                rgba = DEFAULT_SITE_RGBA[segment.site_id]
                density = max(float(rgba[3]), 0.0)
                segment_transmittance = math.exp(-density * length)
                segment_alpha = 1.0 - segment_transmittance
                weight = transmittance * segment_alpha
                mid_depth = 0.5 * (segment.depth0 + segment.depth1)
                rgb_accum += torch.tensor(rgba[:3], dtype=torch.float32) * weight
                alpha_accum += weight
                depth_weighted += weight * mid_depth
                transmittance *= segment_transmittance
            rgb[t_index, u_index] = rgb_accum
            alpha[t_index, u_index] = alpha_accum
            depth[t_index, u_index] = depth_weighted / alpha_accum if alpha_accum > 1.0e-8 else config.far
    return rgb, alpha, depth


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
    site_rgba_f32 = torch.tensor(DEFAULT_SITE_RGBA, dtype=torch.float32, device=device)
    beam_f32 = torch.tensor(
        [[u, 0.0, 1.0, config.near, config.far] for u in u_values],
        dtype=torch.float32,
        device=device,
    )
    frame_t_f32 = torch.tensor(frame_times, dtype=torch.float32, device=device)

    output_rgb, output_alpha, output_depth = shared_rgba_depth_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgba_f32,
        beam_f32,
        frame_t_f32,
        PowerBoundaryConfig(camera_velocity_x=config.camera_velocity_x, invalid_epsilon=config.invalid_epsilon),
    )
    torch.mps.synchronize()
    mps_rgb = output_rgb.transpose(0, 1).cpu()
    mps_alpha = output_alpha.transpose(0, 1).cpu()
    mps_depth = output_depth.transpose(0, 1).cpu()

    started_at = time.perf_counter()
    timed_rgb = output_rgb
    timed_alpha = output_alpha
    timed_depth = output_depth
    for _ in range(timing_iters):
        timed_rgb, timed_alpha, timed_depth = shared_rgba_depth_replay(
            boundary_f32,
            candidate_mask_u32,
            sites_f32,
            site_rgba_f32,
            beam_f32,
            frame_t_f32,
            PowerBoundaryConfig(camera_velocity_x=config.camera_velocity_x, invalid_epsilon=config.invalid_epsilon),
        )
    torch.mps.synchronize()
    _timed_shapes = (timed_rgb.shape, timed_alpha.shape, timed_depth.shape)
    mps_composite_wall_clock_ms = (time.perf_counter() - started_at) * 1000.0 / float(timing_iters)

    cpu_rgb, cpu_alpha, cpu_depth = cpu_composite_reference(
        config=config,
        u_values=u_values,
        frame_times=frame_times,
    )
    max_rgb_abs_error = float((mps_rgb - cpu_rgb).abs().max().item())
    max_alpha_abs_error = float((mps_alpha - cpu_alpha).abs().max().item())
    max_depth_abs_error = float((mps_depth - cpu_depth).abs().max().item())
    if ppm_out is not None:
        write_ppm(ppm_out, mps_rgb)

    tolerance = 1.0e-4
    finite_rgb = torch.isfinite(mps_rgb)
    finite_alpha = torch.isfinite(mps_alpha)
    finite_depth = torch.isfinite(mps_depth)
    acceptance = {
        "rgb_shape_is_strip": list(mps_rgb.shape) == [len(frame_times), len(u_values), 3],
        "alpha_shape_is_strip": list(mps_alpha.shape) == [len(frame_times), len(u_values)],
        "depth_shape_is_strip": list(mps_depth.shape) == [len(frame_times), len(u_values)],
        "all_outputs_finite": bool(finite_rgb.all().item() and finite_alpha.all().item() and finite_depth.all().item()),
        "alpha_in_unit_interval": bool(((mps_alpha >= -tolerance) & (mps_alpha <= 1.0 + tolerance)).all().item()),
        "depth_in_range": bool(((mps_depth >= config.near - tolerance) & (mps_depth <= config.far + tolerance)).all().item()),
        "rgb_nonconstant": float(mps_rgb.std().item()) > 0.0,
        "alpha_nonconstant": float(mps_alpha.std().item()) > 0.0,
        "matches_cpu_reference": max(max_rgb_abs_error, max_alpha_abs_error, max_depth_abs_error) <= tolerance,
        "uses_mps_shared_rgba_depth_replay": True,
    }
    direct_forward_boundary_scans = len(boundaries) * len(u_values) * len(frame_times)
    shared_forward_boundary_scans = len(boundaries) * len(u_values) * config.time_slabs
    return {
        "benchmark": "world_foam_lane2_gate0_8_mps_composite_strip_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "device": "mps",
        "gate": "0.8",
        "renderer_scope": "rgb_alpha_depth_composite_strip_forward_only_no_geometry_gradients_or_trainer",
        "gradient_scope": "none_forward_only_compositor",
        "rgb_shape": list(mps_rgb.shape),
        "alpha_shape": list(mps_alpha.shape),
        "depth_shape": list(mps_depth.shape),
        "frames": len(frame_times),
        "height": 1,
        "width": len(u_values),
        "pixel_ray_count": len(frame_times) * len(u_values),
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "ppm_out": str(ppm_out) if ppm_out is not None else None,
        "acceptance": acceptance,
        "max_rgb_abs_error": max_rgb_abs_error,
        "max_alpha_abs_error": max_alpha_abs_error,
        "max_depth_abs_error": max_depth_abs_error,
        "rgb_min": float(mps_rgb.min().item()),
        "rgb_max": float(mps_rgb.max().item()),
        "rgb_std": float(mps_rgb.std().item()),
        "alpha_min": float(mps_alpha.min().item()),
        "alpha_max": float(mps_alpha.max().item()),
        "alpha_std": float(mps_alpha.std().item()),
        "depth_min": float(mps_depth.min().item()),
        "depth_max": float(mps_depth.max().item()),
        "depth_std": float(mps_depth.std().item()),
        "mps_composite_wall_clock_ms": mps_composite_wall_clock_ms,
        "timing_iters": timing_iters,
        "tolerance": tolerance,
        "direct_forward_boundary_scans": direct_forward_boundary_scans,
        "shared_forward_boundary_scans": shared_forward_boundary_scans,
        "shared_forward_boundary_scan_ratio": shared_forward_boundary_scans
        / float(max(direct_forward_boundary_scans, 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke the World Foam Lane 2 MPS composite strip renderer.")
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
