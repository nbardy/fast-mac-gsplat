#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DYNAWORLD = ROOT.parents[3]
WORLD_FOAM_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2"
RESULTS_DIR = WORLD_FOAM_DIR / "results"
COMPARATOR_CONFIG = (
    DYNAWORLD
    / "src"
    / "train_configs"
    / "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alphaab_alpha1_128.jsonc"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORLD_FOAM_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_FOAM_DIR))

from gate1_realray_per_sample_reference import (  # noqa: E402
    _load_config,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
    write_ppm,
)
from smoke_shared_realray_csr_candidate_storage_mps import (  # noqa: E402
    _build_tiled_csr,
    _csr_stats,
    _csr_valid,
)
from smoke_shared_realray_vjp_mps import _build_candidate_bundle, _reshape_rgb  # noqa: E402
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    shared_realray_rgba_depth_csr_autograd,
)


def _frame_times(frame_count: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [float(frame) / float(max(frame_count - 1, 1)) for frame in range(frame_count)],
        dtype=torch.float32,
        device=device,
    )


def _psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1.0e-12))


def _ssim_torch(rendered: torch.Tensor, target: torch.Tensor) -> float:
    rendered = rendered.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    target = target.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    if rendered.ndim != 4 or target.ndim != 4:
        raise ValueError("SSIM inputs must have shape [N,3,H,W]")
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(rendered, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(rendered * rendered, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(rendered * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / denominator.clamp_min(1.0e-12)).mean().cpu().item())


def _metrics(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    rendered = rendered.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0)
    target = target.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0)
    l1 = float(torch.mean(torch.abs(rendered - target)).item())
    mse = float(torch.mean((rendered - target).square()).item())
    return {
        "l1": l1,
        "mse": mse,
        "psnr": _psnr_from_mse(mse),
        "ssim": _ssim_torch(rendered, target),
    }


def _summarize_timing_rows(rows: list[dict[str, float]], *, skip_keys: set[str] | None = None) -> dict[str, dict[str, float | int]]:
    skipped = set() if skip_keys is None else skip_keys
    keys = sorted({key for row in rows for key in row if key not in skipped})
    summary: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        summary[key] = {
            "count": len(values),
            "mean_s": statistics.fmean(values),
            "min_s": min(values),
            "max_s": max(values),
            "total_s": sum(values),
        }
    mean_total = float(summary.get("total", {}).get("mean_s", 0.0))
    if mean_total > 0.0:
        for value in summary.values():
            value["mean_pct_of_total"] = float(value["mean_s"]) / mean_total
    return summary


def _split_layout_summary(
    *,
    bundle: dict[str, Any],
    layout: dict[str, Any],
    boundary_count: int,
) -> dict[str, Any]:
    bitset_bytes = int(bundle["candidate_mask"].numel() * bundle["candidate_mask"].element_size())
    tiled = _csr_stats(layout, bundle=bundle, boundary_count=boundary_count, bitset_bytes=bitset_bytes)
    direct_scans = int(bundle["direct_forward_boundary_scans"])
    return {
        "pixel_tracks": int(bundle["pixel_tracks"]),
        "pixel_rays": int(bundle["pixel_rays"]),
        "candidate_mask_shape": bundle["candidate_mask_shape"],
        "bitset_storage_bytes": bitset_bytes,
        "tiled_csr": tiled,
        "tiled_csr_valid": _csr_valid(layout, boundary_count=boundary_count),
        "tile_shape": layout["tile_shape"],
        "tile_grid_shape": layout["tile_grid_shape"],
        "per_frame_event_sum": int(bundle["per_frame_event_sum"]),
        "shared_slab_event_sum": int(bundle["shared_slab_event_sum"]),
        "event_sharing_ratio": float(bundle["event_sharing_ratio"]),
        "missing_sample_events": int(bundle["missing_sample_events"]),
        "direct_forward_boundary_scans": direct_scans,
        "shared_forward_boundary_scans": int(bundle["shared_forward_boundary_scans"]),
        "shared_forward_boundary_scan_ratio": float(bundle["shared_forward_boundary_scan_ratio"]),
        "tiled_candidate_iteration_vs_direct_scan_ratio": float(tiled["candidate_iterations"])
        / float(max(direct_scans, 1)),
    }


def _to_device_layout(layout: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor | int]:
    return {
        "row_index": layout["row_index"].to(device),
        "row_offsets": layout["row_offsets"].to(device),
        "candidate_ids": layout["candidate_ids"].to(device),
        "row_count": int(layout["row_count"]),
    }


def _render_split(
    *,
    boundary_f32: torch.Tensor,
    sites_f32: torch.Tensor,
    site_rgba: torch.Tensor,
    track_rays_f32: torch.Tensor,
    layout_device: dict[str, torch.Tensor | int],
    frame_t_f32: torch.Tensor,
    config: RealRayReplayConfig,
    time_slabs: int,
    view_count: int,
    frame_count: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    started_at = time.perf_counter()
    rgb, alpha, depth = shared_realray_rgba_depth_csr_autograd(
        boundary_f32,
        layout_device["row_index"],  # type: ignore[arg-type]
        layout_device["row_offsets"],  # type: ignore[arg-type]
        layout_device["candidate_ids"],  # type: ignore[arg-type]
        sites_f32,
        site_rgba,
        track_rays_f32,
        frame_t_f32,
        config,
        time_slab_count=time_slabs,
        row_count=int(layout_device["row_count"]),
    )
    torch.mps.synchronize()
    elapsed_s = time.perf_counter() - started_at
    rgb_image = _reshape_rgb(rgb, view_count=view_count, frame_count=frame_count, height=height, width=width)
    return rgb_image, alpha, depth, float(elapsed_s)


def run_train_eval(
    *,
    config_path: Path,
    max_frames: int,
    render_size: int,
    site_count: int,
    time_slabs: int,
    tile_h: int,
    tile_w: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    steps: int,
    warmup_steps: int = 0,
    lr: float,
    train_ppm_out: Path | None,
    heldout_ppm_out: Path | None,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    started_total = time.perf_counter()
    cfg = _load_config(config_path, max_frames=max_frames, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    sample_rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    sample_frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    heldout_targets = data["heldout_targets"]
    heldout_rays = data["heldout_rays"]
    heldout_frame_indices = data["heldout_frame_indices"]
    if heldout_targets is None or heldout_rays is None or heldout_frame_indices is None:
        raise ValueError("train/eval requires heldout targets, rays, and frame indices")
    heldout_targets_cpu = heldout_targets.detach().cpu().to(dtype=torch.float32)
    heldout_rays_cpu = heldout_rays.detach().cpu().to(dtype=torch.float32)
    heldout_frame_indices_cpu = heldout_frame_indices.detach().cpu().to(dtype=torch.long)
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
    heldout_bundle = _build_candidate_bundle(
        boundaries=boundaries,
        rays=heldout_rays_cpu,
        frame_indices=heldout_frame_indices_cpu,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        split="heldout",
    )
    train_layout = _build_tiled_csr(train_bundle, time_slabs=time_slabs, tile_h=tile_h, tile_w=tile_w)
    heldout_layout = _build_tiled_csr(heldout_bundle, time_slabs=time_slabs, tile_h=tile_h, tile_w=tile_w)

    device = torch.device("mps")
    target_rgb_image = targets.to(device)
    sites_f32 = torch.tensor([[site.x, site.y, site.z, site.t, site.weight] for site in sites], dtype=torch.float32, device=device)
    boundary_f32 = torch.tensor(
        [[boundary.nx, boundary.ny, boundary.nz, boundary.nt, boundary.b] for boundary in boundaries],
        dtype=torch.float32,
        device=device,
    )
    initial_site_rgba = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)
    site_rgba = initial_site_rgba.detach().clone().requires_grad_(True)
    frame_t_f32 = _frame_times(frame_count, device)
    train_track_rays = train_bundle["track_rays"].to(device)
    heldout_track_rays = heldout_bundle["track_rays"].to(device)
    train_layout_device = _to_device_layout(train_layout, device)
    heldout_layout_device = _to_device_layout(heldout_layout, device)
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )

    train_view_count = int(train_bundle["view_count"])
    heldout_view_count = int(heldout_bundle["view_count"])
    height = int(train_bundle["height"])
    width = int(train_bundle["width"])
    optimizer = torch.optim.Adam([site_rgba], lr=lr)
    loss_history: list[float] = []
    train_psnr_history: list[float] = []
    all_loss_history: list[float] = []
    all_train_psnr_history: list[float] = []
    step_rows: list[dict[str, float]] = []
    first_grad_abs_sum = 0.0
    first_grad_abs_max = 0.0
    started_train = time.perf_counter()
    total_optimizer_steps = warmup_steps + steps
    for step in range(total_optimizer_steps):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        rgb_image, _alpha, _depth, _elapsed_s = _render_split(
            boundary_f32=boundary_f32,
            sites_f32=sites_f32,
            site_rgba=site_rgba,
            track_rays_f32=train_track_rays,
            layout_device=train_layout_device,
            frame_t_f32=frame_t_f32,
            config=op_config,
            time_slabs=time_slabs,
            view_count=train_view_count,
            frame_count=frame_count,
            height=height,
            width=width,
        )
        loss_started = time.perf_counter()
        rgb_mse = F.mse_loss(rgb_image, target_rgb_image)
        torch.mps.synchronize()
        loss_elapsed_s = time.perf_counter() - loss_started
        loss_value = float(rgb_mse.detach().cpu().item())
        all_loss_history.append(loss_value)
        all_train_psnr_history.append(_psnr_from_mse(loss_value))
        backward_started = time.perf_counter()
        rgb_mse.backward()
        torch.mps.synchronize()
        backward_elapsed_s = time.perf_counter() - backward_started
        if step == 0:
            first_grad_abs_sum = float(site_rgba.grad.detach().abs().sum().cpu().item())
            first_grad_abs_max = float(site_rgba.grad.detach().abs().max().cpu().item())
        optimizer_started = time.perf_counter()
        optimizer.step()
        with torch.no_grad():
            site_rgba[:, :3].clamp_(0.0, 1.0)
            site_rgba[:, 3].clamp_(min=0.01)
        torch.mps.synchronize()
        optimizer_elapsed_s = time.perf_counter() - optimizer_started
        step_row = {
            "zero_grad": 0.0,
            "render": float(_elapsed_s),
            "loss": float(loss_elapsed_s),
            "backward": float(backward_elapsed_s),
            "optimizer": float(optimizer_elapsed_s),
            "total": float(time.perf_counter() - step_started),
            "loss_value": loss_value,
        }
        if step >= warmup_steps:
            step_rows.append(step_row)
            loss_history.append(loss_value)
            train_psnr_history.append(_psnr_from_mse(loss_value))
    torch.mps.synchronize()
    train_loop_elapsed_s = time.perf_counter() - started_train
    measured_train_loop_elapsed_s = float(sum(row["total"] for row in step_rows))

    with torch.no_grad():
        final_train_rgb, final_train_alpha, final_train_depth, train_render_elapsed_s = _render_split(
            boundary_f32=boundary_f32,
            sites_f32=sites_f32,
            site_rgba=site_rgba,
            track_rays_f32=train_track_rays,
            layout_device=train_layout_device,
            frame_t_f32=frame_t_f32,
            config=op_config,
            time_slabs=time_slabs,
            view_count=train_view_count,
            frame_count=frame_count,
            height=height,
            width=width,
        )
        final_heldout_rgb, final_heldout_alpha, final_heldout_depth, heldout_render_elapsed_s = _render_split(
            boundary_f32=boundary_f32,
            sites_f32=sites_f32,
            site_rgba=site_rgba,
            track_rays_f32=heldout_track_rays,
            layout_device=heldout_layout_device,
            frame_t_f32=frame_t_f32,
            config=op_config,
            time_slabs=time_slabs,
            view_count=heldout_view_count,
            frame_count=frame_count,
            height=height,
            width=width,
        )

    train_metrics = _metrics(final_train_rgb, targets)
    heldout_metrics = _metrics(final_heldout_rgb, heldout_targets_cpu)
    if train_ppm_out is not None:
        write_ppm(train_ppm_out, final_train_rgb[0])
    if heldout_ppm_out is not None:
        write_ppm(heldout_ppm_out, final_heldout_rgb[0])

    parameter_update_abs_max = float((site_rgba.detach() - initial_site_rgba).abs().max().cpu().item())
    train_summary = {
        **_split_layout_summary(bundle=train_bundle, layout=train_layout, boundary_count=len(boundaries)),
        "rgb_shape": list(final_train_rgb.shape),
        "alpha_shape": [train_view_count * frame_count, height, width],
        "depth_shape": [train_view_count * frame_count, height, width],
        "target_rgb_shape": list(targets.shape),
        "target_l1": train_metrics["l1"],
        "target_mse": train_metrics["mse"],
        "target_psnr": train_metrics["psnr"],
        "target_ssim": train_metrics["ssim"],
        "render_elapsed_s": train_render_elapsed_s,
        "alpha_min": float(final_train_alpha.detach().cpu().min().item()),
        "alpha_max": float(final_train_alpha.detach().cpu().max().item()),
        "depth_min": float(final_train_depth.detach().cpu().min().item()),
        "depth_max": float(final_train_depth.detach().cpu().max().item()),
    }
    heldout_summary = {
        **_split_layout_summary(bundle=heldout_bundle, layout=heldout_layout, boundary_count=len(boundaries)),
        "rgb_shape": list(final_heldout_rgb.shape),
        "alpha_shape": [heldout_view_count * frame_count, height, width],
        "depth_shape": [heldout_view_count * frame_count, height, width],
        "target_rgb_shape": list(heldout_targets_cpu.shape),
        "target_l1": heldout_metrics["l1"],
        "target_mse": heldout_metrics["mse"],
        "target_psnr": heldout_metrics["psnr"],
        "target_ssim": heldout_metrics["ssim"],
        "render_elapsed_s": heldout_render_elapsed_s,
        "alpha_min": float(final_heldout_alpha.detach().cpu().min().item()),
        "alpha_max": float(final_heldout_alpha.detach().cpu().max().item()),
        "depth_min": float(final_heldout_depth.detach().cpu().min().item()),
        "depth_max": float(final_heldout_depth.detach().cpu().max().item()),
    }
    metrics = {
        "eval_l1": train_metrics["l1"],
        "eval_mse": train_metrics["mse"],
        "eval_psnr": train_metrics["psnr"],
        "eval_ssim": train_metrics["ssim"],
        "heldout_eval_l1": heldout_metrics["l1"],
        "heldout_eval_mse": heldout_metrics["mse"],
        "heldout_eval_psnr": heldout_metrics["psnr"],
        "heldout_eval_ssim": heldout_metrics["ssim"],
        "eval_render_only_elapsed_s": train_render_elapsed_s + heldout_render_elapsed_s,
        "eval_train_render_only_elapsed_s": train_render_elapsed_s,
        "eval_heldout_render_only_elapsed_s": heldout_render_elapsed_s,
    }
    target_shape = {
        "render_size_is_256": int(cfg["render"]["render_size"]) == 256,
        "frame_count_is_16": frame_count == 16,
    }
    acceptance = {
        "loaded_real_multicam_bundle": str(cfg["data"]["frame_source"]) == "multicam_val",
        "same_split_as_star_dynamic": list(data["train_views"]) == ["camera_0006", "camera_0014"]
        and list(data["heldout_views"]) == ["camera_0005"],
        "consumed_train_camera_rays": list(sample_rays.shape) == [targets.shape[0], targets.shape[2], targets.shape[3], 6],
        "consumed_heldout_camera_rays": list(heldout_rays_cpu.shape)
        == [heldout_targets_cpu.shape[0], heldout_targets_cpu.shape[2], heldout_targets_cpu.shape[3], 6],
        "train_output_shape_matches_targets": train_summary["rgb_shape"] == list(targets.shape),
        "heldout_output_shape_matches_targets": heldout_summary["rgb_shape"] == list(heldout_targets_cpu.shape),
        "loss_decreased": train_metrics["mse"] < loss_history[0],
        "parameters_updated": parameter_update_abs_max > 1.0e-6,
        "gradients_nonzero": first_grad_abs_sum > 0.0 and first_grad_abs_max > 0.0,
        "zero_missing_sample_events": int(train_bundle["missing_sample_events"]) == 0
        and int(heldout_bundle["missing_sample_events"]) == 0,
        "tiled_csr_rows_valid": all(train_summary["tiled_csr_valid"].values())
        and all(heldout_summary["tiled_csr_valid"].values()),
        "outputs_are_finite": bool(
            torch.isfinite(final_train_rgb).all().item()
            and torch.isfinite(final_train_alpha).all().item()
            and torch.isfinite(final_train_depth).all().item()
            and torch.isfinite(final_heldout_rgb).all().item()
            and torch.isfinite(final_heldout_alpha).all().item()
            and torch.isfinite(final_heldout_depth).all().item()
            and all(math.isfinite(float(value)) for value in metrics.values())
        ),
        "heldout_metrics_present": all(
            math.isfinite(float(metrics[key]))
            for key in ("heldout_eval_l1", "heldout_eval_mse", "heldout_eval_psnr", "heldout_eval_ssim")
        ),
    }
    return {
        "benchmark": "world_foam_lane2_gate3_mps_shared_realray_csr_train_eval",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "3_quality_realray_mps_frozen_geometry_site_rgba_csr_train_eval",
        "device": "mps",
        "config_path": str(config_path),
        "sample_id": data["source_label"],
        "pose_source": data["pose_source"],
        "train_views": list(data["train_views"]),
        "heldout_views": list(data["heldout_views"]),
        "frame_count": frame_count,
        "frame_counts": [frame_count],
        "render_size": int(cfg["render"]["render_size"]),
        "time_slabs": time_slabs,
        "tile_shape": [tile_h, tile_w],
        "site_count": site_count,
        "boundary_count": len(boundaries),
        "near": float(near),
        "far": float(far),
        "density": float(density),
        "steps": steps,
        "warmup_steps": warmup_steps,
        "total_optimizer_steps": total_optimizer_steps,
        "lr": lr,
        "train_loop_elapsed_s": float(train_loop_elapsed_s),
        "measured_train_loop_elapsed_s": measured_train_loop_elapsed_s,
        "step_rows": step_rows,
        "step_summary": _summarize_timing_rows(step_rows, skip_keys={"loss_value"}),
        "total_elapsed_s": float(time.perf_counter() - started_total),
        "renderer_scope": "mps_real_camera_ray_4d_power_cell_time_slab_shared_csr_forward",
        "gradient_scope": "frozen_geometry_csr_autograd_site_rgba_only_no_geometry_or_topology_gradients",
        "sharing_scope": "mps_real_camera_ray_time_slab_tiled_csr_candidate_forward_and_reduced_vjp_backward",
        "autograd_wrapper": "shared_realray_rgba_depth_csr_autograd_frozen_geometry_site_rgba_only",
        "comparison_unit": "world_foam_mps_shared_realray_csr_quality",
        "quality_claim": target_shape["render_size_is_256"] and target_shape["frame_count_is_16"],
        "heldout_quality_metric_claim": True,
        "full_trainer_claim": False,
        "full_geometry_gradient_claim": False,
        "world_foam_renderer_status": "mps_shared_real_camera_ray_csr_frozen_geometry_site_rgba_train_eval_no_geometry_topology_gradients",
        "selection_metric": "final_train_mse",
        "selected_step": total_optimizer_steps,
        "selected_uses_heldout_for_selection": False,
        "initial_train_mse": loss_history[0],
        "initial_all_train_mse": all_loss_history[0],
        "final_train_mse": train_metrics["mse"],
        "initial_train_psnr": train_psnr_history[0],
        "initial_all_train_psnr": all_train_psnr_history[0],
        "final_train_psnr": train_metrics["psnr"],
        "final_heldout_psnr": heldout_metrics["psnr"],
        "train_psnr_delta": train_metrics["psnr"] - train_psnr_history[0],
        "loss_ratio": train_metrics["mse"] / float(max(loss_history[0], 1.0e-12)),
        "loss_history": loss_history,
        "train_psnr_history": train_psnr_history,
        "first_grad_abs_sum": first_grad_abs_sum,
        "first_grad_abs_max": first_grad_abs_max,
        "parameter_update_abs_max": parameter_update_abs_max,
        "metrics": metrics,
        "train": train_summary,
        "heldout": heldout_summary,
        "target_shape": target_shape,
        "acceptance": acceptance,
        "ssim_method": "torch_3x3_mean_pool_local_ssim",
        "proof_images": {
            "train_ppm": str(train_ppm_out) if train_ppm_out is not None else None,
            "heldout_ppm": str(heldout_ppm_out) if heldout_ppm_out is not None else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/eval World Foam shared real-ray tiled-CSR autograd on MPS.")
    parser.add_argument("--config", type=Path, default=COMPARATOR_CONFIG)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--train-ppm-out", type=Path, default=RESULTS_DIR / "gate3_mps_shared_realray_csr_quality_256px_16f_train.ppm")
    parser.add_argument(
        "--heldout-ppm-out",
        type=Path,
        default=RESULTS_DIR / "gate3_mps_shared_realray_csr_quality_256px_16f_heldout.ppm",
    )
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "gate3_mps_shared_realray_csr_quality_256px_16f.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_train_eval(
        config_path=args.config,
        max_frames=args.max_frames,
        render_size=args.render_size,
        site_count=args.site_count,
        time_slabs=args.time_slabs,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        near=args.near,
        far=args.far,
        density=args.density,
        invalid_epsilon=args.invalid_epsilon,
        transmittance_threshold=args.transmittance_threshold,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        train_ppm_out=args.train_ppm_out,
        heldout_ppm_out=args.heldout_ppm_out,
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
