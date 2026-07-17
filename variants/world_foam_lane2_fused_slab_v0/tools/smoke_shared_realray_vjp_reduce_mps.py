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

from gate1_realray_per_sample_reference import (  # noqa: E402
    DEFAULT_CONFIG,
    _load_config,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
    render_samples,
)
from smoke_shared_realray_vjp_mps import (  # noqa: E402
    _build_candidate_bundle,
    _cpu_vjp_split,
    _make_gradients,
    _mps_vjp_split,
    _reshape_rgb,
)
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    shared_realray_rgba_depth_autograd,
    shared_realray_rgba_depth_vjp_reduce,
)


REDUCTION_CHUNK_SIZE = 4


def _mps_reduced_split(
    *,
    sites_f32: torch.Tensor,
    boundary_f32: torch.Tensor,
    site_rgba_f32: torch.Tensor,
    bundle: dict[str, Any],
    frame_count: int,
    config: RealRayReplayConfig,
    timing_iters: int,
) -> dict[str, Any]:
    device = torch.device("mps")
    frame_t_f32 = torch.tensor(
        [float(frame) / float(max(frame_count - 1, 1)) for frame in range(frame_count)],
        dtype=torch.float32,
        device=device,
    )
    track_rays_f32 = bundle["track_rays"].to(device)
    candidate_mask_i32 = bundle["candidate_mask"].to(device)
    grad_rgb_cpu, grad_alpha_cpu, grad_depth_cpu = _make_gradients(
        track_count=track_rays_f32.shape[0],
        frame_count=frame_count,
    )
    grad_rgb_f32 = grad_rgb_cpu.to(device)
    grad_alpha_f32 = grad_alpha_cpu.to(device)
    grad_depth_f32 = grad_depth_cpu.to(device)
    output_rgb, output_alpha, output_depth, grad_site_rgba = shared_realray_rgba_depth_vjp_reduce(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config,
    )
    site_rgba_autograd = site_rgba_f32.detach().clone().requires_grad_(True)
    auto_started_at = time.perf_counter()
    auto_rgb, auto_alpha, auto_depth = shared_realray_rgba_depth_autograd(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_autograd,
        track_rays_f32,
        frame_t_f32,
        config,
    )
    auto_loss = (
        (auto_rgb * grad_rgb_f32).sum()
        + (auto_alpha * grad_alpha_f32).sum()
        + (auto_depth * grad_depth_f32).sum()
    )
    auto_loss.backward()
    torch.mps.synchronize()
    autograd_elapsed_ms = (time.perf_counter() - auto_started_at) * 1000.0
    started_at = time.perf_counter()
    timed = (output_rgb, output_alpha, output_depth, grad_site_rgba)
    for _ in range(timing_iters):
        timed = shared_realray_rgba_depth_vjp_reduce(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            grad_rgb_f32,
            grad_alpha_f32,
            grad_depth_f32,
            config,
        )
    torch.mps.synchronize()
    _timed_shapes = tuple(tensor.shape for tensor in timed)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0 / float(timing_iters)
    return {
        "rgb": output_rgb.cpu(),
        "alpha": output_alpha.cpu(),
        "depth": output_depth.cpu(),
        "grad_site_rgba": grad_site_rgba.cpu(),
        "autograd_grad_site_rgba": site_rgba_autograd.grad.detach().cpu(),
        "autograd_loss": auto_loss.detach().cpu(),
        "autograd_rgb": auto_rgb.detach().cpu(),
        "autograd_alpha": auto_alpha.detach().cpu(),
        "autograd_depth": auto_depth.detach().cpu(),
        "grad_rgb": grad_rgb_cpu,
        "grad_alpha": grad_alpha_cpu,
        "grad_depth": grad_depth_cpu,
        "mps_shared_realray_autograd_backward_wall_clock_ms": float(autograd_elapsed_ms),
        "mps_shared_realray_reduced_vjp_wall_clock_ms": float(elapsed_ms),
    }


def _split_summary(
    *,
    split: str,
    reduced: dict[str, Any],
    full_mps: dict[str, Any],
    cpu: dict[str, Any],
    direct_cpu: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    view_count = int(bundle["view_count"])
    frame_count = int(reduced["alpha"].shape[1])
    height = int(bundle["height"])
    width = int(bundle["width"])
    reduced_rgb_image = _reshape_rgb(
        reduced["rgb"],
        view_count=view_count,
        frame_count=frame_count,
        height=height,
        width=width,
    )
    cpu_reduced_grad = cpu["grad_samples_rgba"].sum(dim=(0, 1))
    full_mps_reduced_grad = full_mps["grad_samples_rgba"].sum(dim=(0, 1))
    loss_mps = (
        (reduced["rgb"] * reduced["grad_rgb"]).sum()
        + (reduced["alpha"] * reduced["grad_alpha"]).sum()
        + (reduced["depth"] * reduced["grad_depth"]).sum()
    )
    partial_chunk_count = (int(bundle["pixel_rays"]) + REDUCTION_CHUNK_SIZE - 1) // REDUCTION_CHUNK_SIZE
    partial_gradient_shape = [partial_chunk_count, int(reduced["grad_site_rgba"].shape[0]), 4]
    partial_gradient_float_count = partial_chunk_count * int(reduced["grad_site_rgba"].shape[0]) * 4
    sample_gradient_oracle_float_count = int(full_mps["grad_samples_rgba"].numel())
    return {
        "split": split,
        "rgb_shape": list(reduced_rgb_image.shape),
        "alpha_shape": [view_count * frame_count, height, width],
        "depth_shape": [view_count * frame_count, height, width],
        "gradient_shape": list(reduced["grad_site_rgba"].shape),
        "sample_gradient_oracle_shape": list(full_mps["grad_samples_rgba"].shape),
        "partial_gradient_shape": partial_gradient_shape,
        "partial_gradient_float_count": partial_gradient_float_count,
        "sample_gradient_oracle_float_count": sample_gradient_oracle_float_count,
        "partial_vs_oracle_gradient_float_ratio": float(partial_gradient_float_count)
        / float(max(sample_gradient_oracle_float_count, 1)),
        "does_not_materialize_sample_gradients": "grad_samples_rgba" not in reduced,
        "pixel_tracks": int(bundle["pixel_tracks"]),
        "pixel_rays": int(bundle["pixel_rays"]),
        "candidate_mask_shape": bundle["candidate_mask_shape"],
        "mask_word_count": int(bundle["word_count"]),
        "per_frame_event_sum": int(bundle["per_frame_event_sum"]),
        "shared_slab_event_sum": int(bundle["shared_slab_event_sum"]),
        "event_sharing_ratio": float(bundle["event_sharing_ratio"]),
        "missing_sample_events": int(bundle["missing_sample_events"]),
        "extra_candidate_events": int(bundle["extra_candidate_events"]),
        "max_candidates_per_slab": int(bundle["max_candidates_per_slab"]),
        "direct_forward_boundary_scans": int(bundle["direct_forward_boundary_scans"]),
        "shared_forward_boundary_scans": int(bundle["shared_forward_boundary_scans"]),
        "shared_forward_boundary_scan_ratio": float(bundle["shared_forward_boundary_scan_ratio"]),
        "max_rgb_abs_error": float((reduced["rgb"] - cpu["rgb"]).abs().max().item()),
        "max_alpha_abs_error": float((reduced["alpha"] - cpu["alpha"]).abs().max().item()),
        "max_depth_abs_error": float((reduced["depth"] - cpu["depth"]).abs().max().item()),
        "max_direct_cpu_rgb_abs_error": float((reduced_rgb_image - direct_cpu["rgb"]).abs().max().item()),
        "max_unreduced_mps_rgb_abs_error": float((reduced["rgb"] - full_mps["rgb"]).abs().max().item()),
        "max_unreduced_mps_alpha_abs_error": float((reduced["alpha"] - full_mps["alpha"]).abs().max().item()),
        "max_unreduced_mps_depth_abs_error": float((reduced["depth"] - full_mps["depth"]).abs().max().item()),
        "max_autograd_rgb_abs_error": float((reduced["autograd_rgb"] - reduced["rgb"]).abs().max().item()),
        "max_autograd_alpha_abs_error": float((reduced["autograd_alpha"] - reduced["alpha"]).abs().max().item()),
        "max_autograd_depth_abs_error": float((reduced["autograd_depth"] - reduced["depth"]).abs().max().item()),
        "max_autograd_rgba_gradient_abs_error": float(
            (reduced["autograd_grad_site_rgba"] - reduced["grad_site_rgba"]).abs().max().item()
        ),
        "max_reduced_rgba_gradient_abs_error": float((reduced["grad_site_rgba"] - cpu_reduced_grad).abs().max().item()),
        "max_reduced_vs_unreduced_mps_sum_abs_error": float(
            (reduced["grad_site_rgba"] - full_mps_reduced_grad).abs().max().item()
        ),
        "autograd_loss_abs_error": float((reduced["autograd_loss"] - cpu["loss"]).abs().item()),
        "loss_abs_error": float((loss_mps - cpu["loss"]).abs().item()),
        "cpu_vjp_elapsed_s": float(cpu["elapsed_s"]),
        "cpu_render_elapsed_s": float(direct_cpu["elapsed_s"]),
        "mps_shared_realray_reduced_vjp_wall_clock_ms": float(
            reduced["mps_shared_realray_reduced_vjp_wall_clock_ms"]
        ),
        "mps_shared_realray_autograd_backward_wall_clock_ms": float(
            reduced["mps_shared_realray_autograd_backward_wall_clock_ms"]
        ),
        "mps_unreduced_oracle_vjp_wall_clock_ms": float(full_mps["mps_shared_realray_vjp_wall_clock_ms"]),
        "cpu_total_segments": int(cpu["total_segments"]),
        "cpu_max_segments_per_ray": int(cpu["max_segments_per_ray"]),
        "reduced_grad_site_rgba_abs_max": float(reduced["grad_site_rgba"].abs().max().item()),
        "reduced_grad_site_rgba_abs_sum": float(reduced["grad_site_rgba"].abs().sum().item()),
        "mps_rgb_std": float(reduced["rgb"].std().item()),
        "mps_alpha_min": float(reduced["alpha"].min().item()),
        "mps_alpha_max": float(reduced["alpha"].max().item()),
        "ray_time_delta": bundle["ray_time_delta"],
    }


def run_smoke(
    *,
    config_path: Path,
    max_frames: int | None,
    render_size: int | None,
    site_count: int,
    time_slabs: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    timing_iters: int,
) -> dict[str, Any]:
    if timing_iters <= 0:
        raise ValueError("timing_iters must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    cfg = _load_config(config_path, max_frames=max_frames, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    sample_rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    sample_frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    heldout_targets = data["heldout_targets"]
    heldout_rays = data["heldout_rays"]
    heldout_frame_indices = data["heldout_frame_indices"]
    if heldout_targets is None or heldout_rays is None or heldout_frame_indices is None:
        raise ValueError("shared real-ray reduced VJP smoke requires heldout targets, rays, and frame indices")

    frame_count = int(data["frame_count"])
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=sample_rays,
        frame_indices=sample_frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
    )
    boundaries = make_boundaries_4d(sites)
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    device = torch.device("mps")
    sites_f32 = torch.tensor([[site.x, site.y, site.z, site.t, site.weight] for site in sites], dtype=torch.float32, device=device)
    boundary_f32 = torch.tensor(
        [[boundary.nx, boundary.ny, boundary.nz, boundary.nt, boundary.b] for boundary in boundaries],
        dtype=torch.float32,
        device=device,
    )
    site_rgba_f32 = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)

    train_bundle = _build_candidate_bundle(
        boundaries=boundaries,
        rays=sample_rays,
        frame_indices=sample_frame_indices,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        split="train",
    )
    heldout_bundle = _build_candidate_bundle(
        boundaries=boundaries,
        rays=heldout_rays.detach().cpu().to(dtype=torch.float32),
        frame_indices=heldout_frame_indices.detach().cpu().to(dtype=torch.long),
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        split="heldout",
    )
    cpu_train_render = render_samples(
        sites=sites,
        boundaries=boundaries,
        rays=sample_rays,
        frame_indices=sample_frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    cpu_heldout_render = render_samples(
        sites=sites,
        boundaries=boundaries,
        rays=heldout_rays.detach().cpu().to(dtype=torch.float32),
        frame_indices=heldout_frame_indices.detach().cpu().to(dtype=torch.long),
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )

    full_mps_train = _mps_vjp_split(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=train_bundle,
        frame_count=frame_count,
        config=op_config,
        timing_iters=timing_iters,
    )
    cpu_train = _cpu_vjp_split(
        sites=sites,
        boundaries=boundaries,
        bundle=train_bundle,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
        grad_rgb=full_mps_train["grad_rgb"],
        grad_alpha=full_mps_train["grad_alpha"],
        grad_depth=full_mps_train["grad_depth"],
    )
    reduced_mps_train = _mps_reduced_split(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=train_bundle,
        frame_count=frame_count,
        config=op_config,
        timing_iters=timing_iters,
    )
    full_mps_heldout = _mps_vjp_split(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=heldout_bundle,
        frame_count=frame_count,
        config=op_config,
        timing_iters=timing_iters,
    )
    cpu_heldout = _cpu_vjp_split(
        sites=sites,
        boundaries=boundaries,
        bundle=heldout_bundle,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
        grad_rgb=full_mps_heldout["grad_rgb"],
        grad_alpha=full_mps_heldout["grad_alpha"],
        grad_depth=full_mps_heldout["grad_depth"],
    )
    reduced_mps_heldout = _mps_reduced_split(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=heldout_bundle,
        frame_count=frame_count,
        config=op_config,
        timing_iters=timing_iters,
    )

    train = _split_summary(
        split="train",
        reduced=reduced_mps_train,
        full_mps=full_mps_train,
        cpu=cpu_train,
        direct_cpu=cpu_train_render,
        bundle=train_bundle,
    )
    heldout = _split_summary(
        split="heldout",
        reduced=reduced_mps_heldout,
        full_mps=full_mps_heldout,
        cpu=cpu_heldout,
        direct_cpu=cpu_heldout_render,
        bundle=heldout_bundle,
    )
    tolerance = 5.0e-4
    max_forward_error = max(
        train["max_rgb_abs_error"],
        train["max_alpha_abs_error"],
        train["max_depth_abs_error"],
        heldout["max_rgb_abs_error"],
        heldout["max_alpha_abs_error"],
        heldout["max_depth_abs_error"],
    )
    max_reduced_gradient_error = max(
        train["max_reduced_rgba_gradient_abs_error"],
        heldout["max_reduced_rgba_gradient_abs_error"],
    )
    max_unreduced_sum_error = max(
        train["max_reduced_vs_unreduced_mps_sum_abs_error"],
        heldout["max_reduced_vs_unreduced_mps_sum_abs_error"],
    )
    expected_gradient_shape = [len(sites), 4]
    acceptance = {
        "loaded_real_multicam_bundle": str(cfg["data"]["frame_source"]) == "multicam_val",
        "consumed_train_camera_rays": list(sample_rays.shape) == [targets.shape[0], targets.shape[2], targets.shape[3], 6],
        "consumed_heldout_camera_rays": list(heldout_rays.shape)
        == [heldout_targets.shape[0], heldout_targets.shape[2], heldout_targets.shape[3], 6],
        "zero_missing_sample_events": train["missing_sample_events"] == 0 and heldout["missing_sample_events"] == 0,
        "real_rays_static_within_views_over_time": bool(
            train["ray_time_delta"]["max_origin_delta_within_view_over_time"] == 0.0
            and train["ray_time_delta"]["max_direction_delta_within_view_over_time"] == 0.0
            and heldout["ray_time_delta"]["max_origin_delta_within_view_over_time"] == 0.0
            and heldout["ray_time_delta"]["max_direction_delta_within_view_over_time"] == 0.0
        ),
        "shared_outputs_match_cpu_vjp_reference": max_forward_error <= tolerance,
        "reduced_rgba_gradient_matches_cpu_vjp_reference": max_reduced_gradient_error <= tolerance,
        "reduced_rgba_gradient_matches_sample_sum_reference": max_unreduced_sum_error <= tolerance,
        "reduced_gradient_shape_is_site_rgba": train["gradient_shape"] == expected_gradient_shape
        and heldout["gradient_shape"] == expected_gradient_shape,
        "does_not_materialize_sample_gradients": bool(
            train["does_not_materialize_sample_gradients"] and heldout["does_not_materialize_sample_gradients"]
        ),
        "loss_matches_cpu_vjp_reference": train["loss_abs_error"] <= tolerance and heldout["loss_abs_error"] <= tolerance,
        "shared_scan_ratio_sublinear": train["shared_forward_boundary_scan_ratio"] <= 1.0
        and heldout["shared_forward_boundary_scan_ratio"] <= 1.0,
        "all_reduced_gradients_finite": bool(
            torch.isfinite(reduced_mps_train["grad_site_rgba"]).all().item()
            and torch.isfinite(reduced_mps_heldout["grad_site_rgba"]).all().item()
        ),
        "autograd_gradients_match_reduced_vjp": train["max_autograd_rgba_gradient_abs_error"] <= tolerance
        and heldout["max_autograd_rgba_gradient_abs_error"] <= tolerance,
        "autograd_outputs_match_reduced_forward": max(
            train["max_autograd_rgb_abs_error"],
            train["max_autograd_alpha_abs_error"],
            train["max_autograd_depth_abs_error"],
            heldout["max_autograd_rgb_abs_error"],
            heldout["max_autograd_alpha_abs_error"],
            heldout["max_autograd_depth_abs_error"],
        )
        <= tolerance,
        "autograd_loss_matches_cpu_vjp_reference": train["autograd_loss_abs_error"] <= tolerance
        and heldout["autograd_loss_abs_error"] <= tolerance,
        "reduced_gradients_nonzero": train["reduced_grad_site_rgba_abs_sum"] > 0.0
        and heldout["reduced_grad_site_rgba_abs_sum"] > 0.0,
        "alpha_in_unit_interval": bool(
            reduced_mps_train["alpha"].min().item() >= -1.0e-6
            and reduced_mps_train["alpha"].max().item() <= 1.0 + 1.0e-6
            and reduced_mps_heldout["alpha"].min().item() >= -1.0e-6
            and reduced_mps_heldout["alpha"].max().item() <= 1.0 + 1.0e-6
        ),
    }
    return {
        "benchmark": "world_foam_lane2_gate2d_mps_shared_realray_reduced_vjp_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "2D_realray_mps_shared_reduced_fixed_segment_vjp",
        "device": "mps",
        "config_path": str(config_path),
        "sample_id": data["source_label"],
        "train_views": list(data["train_views"]),
        "heldout_views": list(data["heldout_views"]),
        "pose_source": data["pose_source"],
        "frame_counts": [frame_count],
        "frame_count": frame_count,
        "render_size": int(cfg["render"]["render_size"]),
        "time_slabs": time_slabs,
        "site_count": site_count,
        "boundary_count": len(boundaries),
        "near": float(near),
        "far": float(far),
        "density": float(density),
        "renderer_scope": "mps_real_camera_ray_4d_power_cell_time_slab_shared_forward_and_reduced_fixed_segment_vjp",
        "gradient_scope": "frozen_geometry_autograd_reduced_site_rgba_only_no_sample_gradient_materialization_no_geometry_or_topology_gradients",
        "sharing_scope": "mps_real_camera_ray_time_slab_candidate_forward_and_reduced_fixed_segment_vjp",
        "quality_claim": False,
        "training_claim": False,
        "world_foam_renderer_status": "mps_shared_real_camera_ray_reduced_fixed_segment_vjp_with_frozen_geometry_site_rgba_autograd_no_geometry_topology_gradients_no_training",
        "reduction_impl": "chunked_partial_reduction_same_python_op",
        "reduction_chunk_size": REDUCTION_CHUNK_SIZE,
        "partial_reduction_materializes_chunk_site_gradients": True,
        "autograd_wrapper": "shared_realray_rgba_depth_autograd_frozen_geometry_site_rgba_only",
        "reduced_op_materializes_sample_gradients": False,
        "oracle_materializes_sample_gradients_for_validation": True,
        "tolerance": tolerance,
        "timing_iters": timing_iters,
        "train": train,
        "heldout": heldout,
        "acceptance": acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke World Foam shared true real-camera-ray MPS reduced VJP.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--render-size", type=int)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--timing-iters", type=int, default=10)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke(
        config_path=args.config,
        max_frames=args.max_frames,
        render_size=args.render_size,
        site_count=args.site_count,
        time_slabs=args.time_slabs,
        near=args.near,
        far=args.far,
        density=args.density,
        invalid_epsilon=args.invalid_epsilon,
        transmittance_threshold=args.transmittance_threshold,
        timing_iters=args.timing_iters,
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
