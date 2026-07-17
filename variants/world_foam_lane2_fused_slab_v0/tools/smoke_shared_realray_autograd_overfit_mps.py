#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


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
)
from smoke_shared_realray_vjp_mps import _build_candidate_bundle, _reshape_rgb  # noqa: E402
from torch_world_foam_lane2_fused_slab import RealRayReplayConfig, shared_realray_rgba_depth_autograd  # noqa: E402


def _frame_times(frame_count: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [float(frame) / float(max(frame_count - 1, 1)) for frame in range(frame_count)],
        dtype=torch.float32,
        device=device,
    )


def _starting_rgba_from_teacher(teacher: torch.Tensor) -> torch.Tensor:
    start = teacher.detach().clone()
    start[:, :3] = torch.clamp(start[:, :3] * 0.65 + 0.17, min=0.0, max=1.0)
    start[:, 3] = torch.clamp(start[:, 3] * 0.55, min=0.01)
    return start


def _loss(
    *,
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    target_rgb: torch.Tensor,
    target_alpha: torch.Tensor,
    target_depth: torch.Tensor,
) -> torch.Tensor:
    return (
        F.mse_loss(rgb, target_rgb)
        + 0.10 * F.mse_loss(alpha, target_alpha)
        + 0.01 * F.mse_loss(depth, target_depth)
    )


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
    steps: int,
    lr: float,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    cfg = _load_config(config_path, max_frames=max_frames, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    sample_rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    sample_frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
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
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )

    device = torch.device("mps")
    sites_f32 = torch.tensor(
        [[site.x, site.y, site.z, site.t, site.weight] for site in sites],
        dtype=torch.float32,
        device=device,
    )
    boundary_f32 = torch.tensor(
        [[boundary.nx, boundary.ny, boundary.nz, boundary.nt, boundary.b] for boundary in boundaries],
        dtype=torch.float32,
        device=device,
    )
    teacher_site_rgba = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)
    site_rgba = _starting_rgba_from_teacher(teacher_site_rgba).detach().requires_grad_(True)
    track_rays_f32 = train_bundle["track_rays"].to(device)
    candidate_mask_i32 = train_bundle["candidate_mask"].to(device)
    frame_t_f32 = _frame_times(frame_count, device)

    with torch.no_grad():
        target_rgb, target_alpha, target_depth = shared_realray_rgba_depth_autograd(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            teacher_site_rgba,
            track_rays_f32,
            frame_t_f32,
            op_config,
        )
        initial_rgba_error = float((site_rgba.detach() - teacher_site_rgba).abs().mean().cpu().item())

    optimizer = torch.optim.Adam([site_rgba], lr=lr)
    loss_history: list[float] = []
    first_grad_abs_sum = 0.0
    first_grad_abs_max = 0.0
    started_at = time.perf_counter()

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        rgb, alpha, depth = shared_realray_rgba_depth_autograd(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba,
            track_rays_f32,
            frame_t_f32,
            op_config,
        )
        loss = _loss(
            rgb=rgb,
            alpha=alpha,
            depth=depth,
            target_rgb=target_rgb,
            target_alpha=target_alpha,
            target_depth=target_depth,
        )
        loss_history.append(float(loss.detach().cpu().item()))
        loss.backward()
        if step == 0:
            first_grad_abs_sum = float(site_rgba.grad.detach().abs().sum().cpu().item())
            first_grad_abs_max = float(site_rgba.grad.detach().abs().max().cpu().item())
        optimizer.step()
        with torch.no_grad():
            site_rgba[:, :3].clamp_(0.0, 1.0)
            site_rgba[:, 3].clamp_(min=0.01)

    torch.mps.synchronize()
    elapsed_s = time.perf_counter() - started_at
    with torch.no_grad():
        final_rgb, final_alpha, final_depth = shared_realray_rgba_depth_autograd(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba,
            track_rays_f32,
            frame_t_f32,
            op_config,
        )
        final_loss_tensor = _loss(
            rgb=final_rgb,
            alpha=final_alpha,
            depth=final_depth,
            target_rgb=target_rgb,
            target_alpha=target_alpha,
            target_depth=target_depth,
        )
        final_rgba_error = float((site_rgba.detach() - teacher_site_rgba).abs().mean().cpu().item())

    initial_loss = loss_history[0]
    final_loss = float(final_loss_tensor.cpu().item())
    start_site_rgba = _starting_rgba_from_teacher(teacher_site_rgba)
    parameter_update_abs_max = float((site_rgba.detach() - start_site_rgba).abs().max().cpu().item())
    view_count = int(train_bundle["view_count"])
    height = int(train_bundle["height"])
    width = int(train_bundle["width"])
    final_rgb_image = _reshape_rgb(
        final_rgb.detach().cpu(),
        view_count=view_count,
        frame_count=frame_count,
        height=height,
        width=width,
    )
    target_rgb_image = _reshape_rgb(
        target_rgb.detach().cpu(),
        view_count=view_count,
        frame_count=frame_count,
        height=height,
        width=width,
    )
    acceptance = {
        "loaded_real_multicam_bundle": str(cfg["data"]["frame_source"]) == "multicam_val",
        "consumed_train_camera_rays": list(sample_rays.shape) == [targets.shape[0], targets.shape[2], targets.shape[3], 6],
        "loss_decreased": final_loss < initial_loss,
        "loss_ratio_under_0p5": final_loss / float(max(initial_loss, 1.0e-12)) < 0.5,
        "parameters_updated": parameter_update_abs_max > 1.0e-6,
        "moved_toward_teacher_rgba": final_rgba_error < initial_rgba_error,
        "gradients_nonzero": first_grad_abs_sum > 0.0 and first_grad_abs_max > 0.0,
        "all_losses_finite": all(torch.isfinite(torch.tensor(value)).item() for value in [*loss_history, final_loss]),
        "shared_scan_ratio_sublinear": train_bundle["shared_forward_boundary_scan_ratio"] <= 1.0,
        "zero_missing_sample_events": train_bundle["missing_sample_events"] == 0,
    }
    return {
        "benchmark": "world_foam_lane2_gate2e_mps_shared_realray_autograd_overfit_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "2E_realray_mps_frozen_geometry_site_rgba_teacher_overfit",
        "device": "mps",
        "config_path": str(config_path),
        "sample_id": data["source_label"],
        "train_views": list(data["train_views"]),
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
        "renderer_scope": "mps_real_camera_ray_4d_power_cell_time_slab_shared_forward",
        "gradient_scope": "frozen_geometry_autograd_site_rgba_only_no_geometry_or_topology_gradients",
        "sharing_scope": "mps_real_camera_ray_time_slab_candidate_forward_and_reduced_vjp_backward",
        "autograd_wrapper": "shared_realray_rgba_depth_autograd_frozen_geometry_site_rgba_only",
        "quality_claim": False,
        "trainer_claim": False,
        "parameter_update_claim": True,
        "teacher_target_claim": True,
        "real_target_training_claim": False,
        "world_foam_renderer_status": "mps_shared_real_camera_ray_frozen_geometry_site_rgba_teacher_overfit_no_geometry_topology_gradients_no_real_target_trainer",
        "steps": steps,
        "lr": lr,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / float(max(initial_loss, 1.0e-12)),
        "loss_history": loss_history,
        "initial_mean_abs_rgba_error_to_teacher": initial_rgba_error,
        "final_mean_abs_rgba_error_to_teacher": final_rgba_error,
        "parameter_update_abs_max": parameter_update_abs_max,
        "first_grad_abs_sum": first_grad_abs_sum,
        "first_grad_abs_max": first_grad_abs_max,
        "elapsed_s": float(elapsed_s),
        "train": {
            "rgb_shape": list(final_rgb_image.shape),
            "target_rgb_shape": list(target_rgb_image.shape),
            "alpha_shape": [view_count * frame_count, height, width],
            "depth_shape": [view_count * frame_count, height, width],
            "pixel_tracks": int(train_bundle["pixel_tracks"]),
            "pixel_rays": int(train_bundle["pixel_rays"]),
            "candidate_mask_shape": train_bundle["candidate_mask_shape"],
            "per_frame_event_sum": int(train_bundle["per_frame_event_sum"]),
            "shared_slab_event_sum": int(train_bundle["shared_slab_event_sum"]),
            "event_sharing_ratio": float(train_bundle["event_sharing_ratio"]),
            "missing_sample_events": int(train_bundle["missing_sample_events"]),
            "direct_forward_boundary_scans": int(train_bundle["direct_forward_boundary_scans"]),
            "shared_forward_boundary_scans": int(train_bundle["shared_forward_boundary_scans"]),
            "shared_forward_boundary_scan_ratio": float(train_bundle["shared_forward_boundary_scan_ratio"]),
            "max_final_rgb_abs_error_to_teacher": float((final_rgb_image - target_rgb_image).abs().max().item()),
            "max_final_alpha_abs_error_to_teacher": float((final_alpha - target_alpha).abs().max().cpu().item()),
            "max_final_depth_abs_error_to_teacher": float((final_depth - target_depth).abs().max().cpu().item()),
        },
        "acceptance": acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke World Foam shared true real-camera-ray MPS frozen-geometry autograd overfit."
    )
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
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.05)
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
        steps=args.steps,
        lr=args.lr,
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
