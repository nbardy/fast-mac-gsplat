#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DYNAWORLD = ROOT.parents[3]
WORLD_FOAM_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2"
RESULTS_DIR = WORLD_FOAM_DIR / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORLD_FOAM_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_FOAM_DIR))

from gate4_moving_ray_slab_compiler import (  # noqa: E402
    DEFAULT_CONFIG,
    SyntheticRayMotion,
    _load_config,
    apply_synthetic_ray_motion,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)
from gate4_affine_slab_tape import Gate4AffineSlabTape, build_gate4_affine_slab_tape  # noqa: E402
from smoke_fused_slab_affine_realray_mps import _parse_int_list  # noqa: E402
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    MAX_REALRAY_BOUNDARIES,
    MAX_REALRAY_FUSED_MSE_BOUNDARIES,
    RealRayReplayConfig,
    fused_slab_affine_num32_den16_autograd,
    fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only,
    fused_slab_affine_num32_den16_ownerupdate_autograd,
)


@dataclass(frozen=True)
class RenderOutputs:
    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor


def _psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1.0e-12))


def _metrics(rendered: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    rendered = rendered.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    target = target.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    mse = float(torch.mean((rendered - target).square()).cpu().item())
    return {
        "mse": mse,
        "psnr": _psnr_from_mse(mse),
        "l1": float(torch.mean(torch.abs(rendered - target)).cpu().item()),
    }


def _reshape_rgb(
    rgb: torch.Tensor,
    *,
    view_count: int,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    return (
        rgb.reshape(view_count, height, width, frame_count, 3)
        .permute(0, 3, 4, 1, 2)
        .reshape(view_count * frame_count, 3, height, width)
    )


def _target_rgb_track_major(
    target_rgb: torch.Tensor,
    *,
    view_count: int,
    frame_count: int,
    height: int,
    width: int,
) -> torch.Tensor:
    expected = (view_count * frame_count, 3, height, width)
    if tuple(target_rgb.shape) != expected:
        raise ValueError(f"target_rgb must have shape {expected}, got {tuple(target_rgb.shape)}")
    return (
        target_rgb.reshape(view_count, frame_count, 3, height, width)
        .permute(0, 3, 4, 1, 2)
        .reshape(view_count * height * width, frame_count, 3)
        .contiguous()
    )


def _to_device_bundle(
    *,
    bundle: dict[str, Any],
    sites: list[Any],
    device: torch.device,
) -> dict[str, Any]:
    coeffs = bundle["candidate_depth_coeffs"]
    return {
        "row_index": bundle["row_index"].to(device),
        "row_offsets": bundle["row_offsets"].to(device),
        "candidate_ids": bundle["candidate_ids"].to(device),
        "candidate_depth_num": coeffs[:, :2].contiguous().to(device),
        "candidate_depth_den": coeffs[:, 2:].contiguous().to(device=device, dtype=torch.float16),
        "ray_coeff": bundle["ray_coeff"].to(device),
        "frame_t": bundle["frame_t"].to(device),
        "row_count": int(bundle["row_count"]),
        "view_count": int(bundle["view_count"]),
        "height": int(bundle["height"]),
        "width": int(bundle["width"]),
        "candidate_count": int(bundle["candidate_count"]),
        "candidate_replay_iterations": int(bundle["candidate_replay_iterations"]),
        "candidate_depth_order": bundle["candidate_depth_order"],
        "missing_sample_events": int(bundle["missing_sample_events"]),
        "extra_candidate_events": int(bundle["extra_candidate_events"]),
        "avg_candidates_per_row": float(bundle["avg_candidates_per_row"]),
        "max_candidates_per_row": int(bundle["max_candidates_per_row"]),
        "sites_f32": torch.tensor(
            [[site.x, site.y, site.z, site.t, site.weight] for site in sites],
            dtype=torch.float32,
            device=device,
        ),
    }


def _storage_bytes(*tensors: torch.Tensor) -> int:
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))


def _mixed_tape_storage_bytes(tape: Gate4AffineSlabTape) -> int:
    return _storage_bytes(
        tape.row_index,
        tape.row_offsets,
        tape.candidate_depth_num,
        tape.candidate_depth_den(),
        tape.ray_coeff,
    )


def _build_train_eval_tape(
    *,
    boundaries: tuple[Any, ...],
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    time_slabs: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    residual_depth_padding: float,
) -> Gate4AffineSlabTape:
    return build_gate4_affine_slab_tape(
        boundaries=boundaries,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        residual_depth_padding=residual_depth_padding,
        layout="per-track",
        tile_h=1,
        tile_w=1,
        candidate_order="slab-mid-depth",
    )


def _render(
    *,
    bundle_device: dict[str, torch.Tensor | int],
    site_rgba: torch.Tensor,
    config: RealRayReplayConfig,
    time_slabs: int,
    frame_count: int,
    reduce_chunk_size: int,
    vjp_mode: str,
) -> RenderOutputs:
    if vjp_mode == "direct_atomic_grad_only_ownerupdate":
        rgb, alpha, depth = fused_slab_affine_num32_den16_ownerupdate_autograd(
            bundle_device["row_index"],  # type: ignore[arg-type]
            bundle_device["row_offsets"],  # type: ignore[arg-type]
            bundle_device["candidate_ids"],  # type: ignore[arg-type]
            bundle_device["candidate_depth_num"],  # type: ignore[arg-type]
            bundle_device["candidate_depth_den"],  # type: ignore[arg-type]
            bundle_device["boundary_site_pairs"],  # type: ignore[arg-type]
            bundle_device["sites_f32"],  # type: ignore[arg-type]
            site_rgba,
            bundle_device["ray_coeff"],  # type: ignore[arg-type]
            bundle_device["frame_t"],  # type: ignore[arg-type]
            config,
            time_slab_count=time_slabs,
            row_count=int(bundle_device["row_count"]),
        )
    else:
        rgb, alpha, depth = fused_slab_affine_num32_den16_autograd(
            bundle_device["row_index"],  # type: ignore[arg-type]
            bundle_device["row_offsets"],  # type: ignore[arg-type]
            bundle_device["candidate_depth_num"],  # type: ignore[arg-type]
            bundle_device["candidate_depth_den"],  # type: ignore[arg-type]
            bundle_device["sites_f32"],  # type: ignore[arg-type]
            site_rgba,
            bundle_device["ray_coeff"],  # type: ignore[arg-type]
            bundle_device["frame_t"],  # type: ignore[arg-type]
            config,
            time_slab_count=time_slabs,
            row_count=int(bundle_device["row_count"]),
            reduce_chunk_size=reduce_chunk_size,
            vjp_mode=vjp_mode,
        )
    return RenderOutputs(
        rgb=_reshape_rgb(
            rgb,
            view_count=int(bundle_device["view_count"]),
            frame_count=frame_count,
            height=int(bundle_device["height"]),
            width=int(bundle_device["width"]),
        ),
        alpha=alpha,
        depth=depth,
    )


def _alpha_depth_aux_loss(
    rendered: RenderOutputs,
    *,
    alpha_aux_weight: float,
    depth_aux_weight: float,
    far: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if alpha_aux_weight <= 0.0 and depth_aux_weight <= 0.0:
        return rendered.rgb.new_zeros(()), {}
    alpha_aux = rendered.alpha.square().mean()
    depth_scale = max(float(far), 1.0e-6)
    depth_aux = (rendered.depth / depth_scale).square().mean()
    total = alpha_aux * float(alpha_aux_weight) + depth_aux * float(depth_aux_weight)
    return total, {
        "alpha_aux_loss": float(alpha_aux.detach().cpu().item()),
        "depth_aux_loss": float(depth_aux.detach().cpu().item()),
        "alpha_aux_weight": float(alpha_aux_weight),
        "depth_aux_weight": float(depth_aux_weight),
    }


def _slice_loaded_training_data(data: dict[str, Any], *, frame_count: int) -> dict[str, Any]:
    train_frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    train_mask = train_frame_indices < int(frame_count)
    if not bool(train_mask.any().item()):
        raise ValueError(f"cached train data has no frames below requested frame_count={frame_count}")

    out = dict(data)
    out["targets"] = data["targets"].detach().cpu().to(dtype=torch.float32)[train_mask].contiguous()
    out["sample_frame_indices"] = train_frame_indices[train_mask].contiguous()
    out["sample_rays"] = data["sample_rays"].detach().cpu().to(dtype=torch.float32)[train_mask].contiguous()

    heldout_frame_indices = data.get("heldout_frame_indices")
    if heldout_frame_indices is not None:
        heldout_frame_indices = heldout_frame_indices.detach().cpu().to(dtype=torch.long)
        heldout_mask = heldout_frame_indices < int(frame_count)
        if not bool(heldout_mask.any().item()):
            raise ValueError(f"cached heldout data has no frames below requested frame_count={frame_count}")
        out["heldout_targets"] = data["heldout_targets"].detach().cpu().to(dtype=torch.float32)[heldout_mask].contiguous()
        out["heldout_frame_indices"] = heldout_frame_indices[heldout_mask].contiguous()
        out["heldout_rays"] = data["heldout_rays"].detach().cpu().to(dtype=torch.float32)[heldout_mask].contiguous()

    init_frames = data.get("init_frames")
    if isinstance(init_frames, torch.Tensor) and init_frames.shape[0] >= int(frame_count):
        out["init_frames"] = init_frames[: int(frame_count)].detach().cpu().contiguous()
    out["frame_count"] = int(frame_count)
    return out


def _summarize_steps(rows: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    keys = ("render", "loss_eval", "backward", "optimizer", "total")
    out: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        out[key] = {
            "count": len(values),
            "mean_s": statistics.fmean(values),
            "median_s": statistics.median(values),
            "min_s": min(values),
            "max_s": max(values),
        }
    return out


def _summarize_values(rows: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        out[key] = {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return out


def _run_one(
    *,
    config_path: Path,
    frame_count: int,
    render_size: int,
    site_count: int,
    time_slabs: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    residual_depth_padding: float,
    synthetic_motion: SyntheticRayMotion,
    steps: int,
    warmup_steps: int,
    lr: float,
    vjp_reduce_chunk_size: int,
    vjp_mode: str,
    alpha_aux_weight: float,
    depth_aux_weight: float,
    cached_training_data: dict[str, Any] | None,
) -> dict[str, Any]:
    run_start = time.perf_counter()
    setup_timings: dict[str, float] = {}
    phase_start = time.perf_counter()
    if cached_training_data is None:
        cfg = _load_config(config_path, max_frames=frame_count, render_size=render_size)
        data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    else:
        data = _slice_loaded_training_data(cached_training_data, frame_count=frame_count)
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    train_rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    train_frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    heldout_targets = data["heldout_targets"]
    heldout_rays = data["heldout_rays"]
    heldout_frame_indices = data["heldout_frame_indices"]
    if heldout_targets is None or heldout_rays is None or heldout_frame_indices is None:
        raise ValueError("train/eval requires heldout targets, rays, and frame indices")
    heldout_targets = heldout_targets.detach().cpu().to(dtype=torch.float32)
    heldout_rays = heldout_rays.detach().cpu().to(dtype=torch.float32)
    heldout_frame_indices = heldout_frame_indices.detach().cpu().to(dtype=torch.long)
    setup_timings["load_or_slice_data_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    train_rays = apply_synthetic_ray_motion(
        train_rays,
        frame_indices=train_frame_indices,
        frame_count=frame_count,
        motion=synthetic_motion,
    )
    heldout_rays = apply_synthetic_ray_motion(
        heldout_rays,
        frame_indices=heldout_frame_indices,
        frame_count=frame_count,
        motion=synthetic_motion,
    )
    setup_timings["apply_motion_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=train_rays,
        frame_indices=train_frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
    )
    boundaries = make_boundaries_4d(sites)
    setup_timings["initialize_sites_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    train_tape = _build_train_eval_tape(
        boundaries=boundaries,
        rays=train_rays,
        frame_indices=train_frame_indices,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        residual_depth_padding=residual_depth_padding,
    )
    setup_timings["build_train_tape_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    heldout_tape = _build_train_eval_tape(
        boundaries=boundaries,
        rays=heldout_rays,
        frame_indices=heldout_frame_indices,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        residual_depth_padding=residual_depth_padding,
    )
    setup_timings["build_heldout_tape_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    train_bundle = train_tape.to_legacy_bundle()
    heldout_bundle = heldout_tape.to_legacy_bundle()
    setup_timings["legacy_bundle_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    device = torch.device("mps")
    train_targets = targets.to(device)
    heldout_targets_device = heldout_targets.to(device)
    train_device = _to_device_bundle(bundle=train_bundle, sites=sites, device=device)
    heldout_device = _to_device_bundle(bundle=heldout_bundle, sites=sites, device=device)
    boundary_site_pairs = torch.tensor([[boundary.left, boundary.right] for boundary in boundaries], dtype=torch.int32, device=device)
    train_device["boundary_site_pairs"] = boundary_site_pairs
    heldout_device["boundary_site_pairs"] = boundary_site_pairs
    site_rgba_initial = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)
    site_rgba = site_rgba_initial.detach().clone().requires_grad_(True)
    train_targets_track = _target_rgb_track_major(
        train_targets,
        view_count=int(train_device["view_count"]),
        frame_count=frame_count,
        height=int(train_device["height"]),
        width=int(train_device["width"]),
    )
    torch.mps.synchronize()
    setup_timings["device_transfer_s"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    optimizer = torch.optim.Adam([site_rgba], lr=lr)
    setup_timings["optimizer_init_s"] = time.perf_counter() - phase_start

    train_loop_start = time.perf_counter()
    step_rows: list[dict[str, float]] = []
    loss_history: list[float] = []
    rgb_loss_history: list[float] = []
    first_grad_abs_sum = 0.0
    first_alpha_output_grad_abs_sum = 0.0
    first_depth_output_grad_abs_sum = 0.0
    aux_loss_active = alpha_aux_weight > 0.0 or depth_aux_weight > 0.0
    fused_mse_active = vjp_mode == "fused_mse_rgb_only"
    total_steps = warmup_steps + steps
    for step in range(total_steps):
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        render_start = time.perf_counter()
        if fused_mse_active:
            torch.mps.synchronize()
            render_s = time.perf_counter() - render_start
            loss_start = time.perf_counter()
            torch.mps.synchronize()
            loss_s = time.perf_counter() - loss_start
            backward_start = time.perf_counter()
            fused_loss, fused_grad = fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only(
                train_device["row_index"],  # type: ignore[arg-type]
                train_device["row_offsets"],  # type: ignore[arg-type]
                train_device["candidate_depth_num"],  # type: ignore[arg-type]
                train_device["candidate_depth_den"],  # type: ignore[arg-type]
                train_device["sites_f32"],  # type: ignore[arg-type]
                site_rgba,
                train_device["ray_coeff"],  # type: ignore[arg-type]
                train_device["frame_t"],  # type: ignore[arg-type]
                train_targets_track,
                op_config,
                time_slab_count=time_slabs,
                row_count=int(train_device["row_count"]),
            )
            loss = fused_loss.reshape(())
            rgb_loss = loss
            site_rgba.grad = fused_grad
            torch.mps.synchronize()
            backward_s = time.perf_counter() - backward_start
            aux_terms = {}
        else:
            rendered = _render(
                bundle_device=train_device,
                site_rgba=site_rgba,
                config=op_config,
                time_slabs=time_slabs,
                frame_count=frame_count,
                reduce_chunk_size=vjp_reduce_chunk_size,
                vjp_mode=vjp_mode,
            )
            if step == 0 and aux_loss_active:
                rendered.alpha.retain_grad()
                rendered.depth.retain_grad()
            torch.mps.synchronize()
            render_s = time.perf_counter() - render_start
            loss_start = time.perf_counter()
            rgb_loss = F.mse_loss(rendered.rgb, train_targets)
            if aux_loss_active:
                aux_loss, aux_terms = _alpha_depth_aux_loss(
                    rendered,
                    alpha_aux_weight=alpha_aux_weight,
                    depth_aux_weight=depth_aux_weight,
                    far=far,
                )
                loss = rgb_loss + aux_loss
            else:
                aux_terms = {}
                loss = rgb_loss
            torch.mps.synchronize()
            loss_s = time.perf_counter() - loss_start
            backward_start = time.perf_counter()
            loss.backward()
            torch.mps.synchronize()
            backward_s = time.perf_counter() - backward_start
        if step == 0:
            first_grad_abs_sum = float(site_rgba.grad.detach().abs().sum().cpu().item())
            if not fused_mse_active and aux_loss_active and rendered.alpha.grad is not None:
                first_alpha_output_grad_abs_sum = float(rendered.alpha.grad.detach().abs().sum().cpu().item())
            if not fused_mse_active and aux_loss_active and rendered.depth.grad is not None:
                first_depth_output_grad_abs_sum = float(rendered.depth.grad.detach().abs().sum().cpu().item())
        optimizer_start = time.perf_counter()
        optimizer.step()
        with torch.no_grad():
            site_rgba[:, :3].clamp_(0.0, 1.0)
            site_rgba[:, 3].clamp_(min=0.01)
        torch.mps.synchronize()
        optimizer_s = time.perf_counter() - optimizer_start
        row = {
            "render": float(render_s),
            "loss_eval": float(loss_s),
            "backward": float(backward_s),
            "optimizer": float(optimizer_s),
            "total": float(time.perf_counter() - step_start),
            "loss": float(loss.detach().cpu().item()),
            "rgb_loss": float(rgb_loss.detach().cpu().item()),
            **aux_terms,
        }
        if step >= warmup_steps:
            step_rows.append(row)
            loss_history.append(row["loss"])
            rgb_loss_history.append(row["rgb_loss"])
    train_loop_s = time.perf_counter() - train_loop_start

    final_eval_start = time.perf_counter()
    with torch.no_grad():
        eval_vjp_mode = "direct_atomic_rgb_only" if fused_mse_active else vjp_mode
        final_train = _render(
            bundle_device=train_device,
            site_rgba=site_rgba,
            config=op_config,
            time_slabs=time_slabs,
            frame_count=frame_count,
            reduce_chunk_size=vjp_reduce_chunk_size,
            vjp_mode=eval_vjp_mode,
        )
        final_heldout = _render(
            bundle_device=heldout_device,
            site_rgba=site_rgba,
            config=op_config,
            time_slabs=time_slabs,
            frame_count=frame_count,
            reduce_chunk_size=vjp_reduce_chunk_size,
            vjp_mode=eval_vjp_mode,
        )
        torch.mps.synchronize()
    final_eval_s = time.perf_counter() - final_eval_start

    train_metrics = _metrics(final_train.rgb, train_targets)
    heldout_metrics = _metrics(final_heldout.rgb, heldout_targets_device)
    param_update = float((site_rgba.detach() - site_rgba_initial).abs().max().cpu().item())
    max_realray_boundaries = MAX_REALRAY_FUSED_MSE_BOUNDARIES if fused_mse_active else MAX_REALRAY_BOUNDARIES
    acceptance = {
        "loss_decreased": bool(rgb_loss_history and train_metrics["mse"] < rgb_loss_history[0]),
        "gradients_nonzero": first_grad_abs_sum > 0.0,
        "parameters_updated": param_update > 1.0e-6,
        "candidate_rows_under_metal_cap": int(train_device["max_candidates_per_row"]) <= max_realray_boundaries
        and int(heldout_device["max_candidates_per_row"]) <= max_realray_boundaries,
        "zero_missing_sample_events": int(train_bundle["missing_sample_events"]) == 0
        and int(heldout_bundle["missing_sample_events"]) == 0,
        "outputs_are_finite": bool(
            torch.isfinite(final_train.rgb).all().item()
            and torch.isfinite(final_train.alpha).all().item()
            and torch.isfinite(final_train.depth).all().item()
            and torch.isfinite(final_heldout.rgb).all().item()
            and torch.isfinite(final_heldout.alpha).all().item()
            and torch.isfinite(final_heldout.depth).all().item()
        ),
        "alpha_depth_aux_vjp_seed_nonzero": not aux_loss_active
        or (first_alpha_output_grad_abs_sum > 0.0 and first_depth_output_grad_abs_sum > 0.0),
    }
    return {
        "frame_count": frame_count,
        "render_size": render_size,
        "site_count": site_count,
        "boundary_count": len(boundaries),
        "steps": steps,
        "warmup_steps": warmup_steps,
        "lr": lr,
        "used_shared_loaded_data": cached_training_data is not None,
        "vjp_mode": vjp_mode,
        "vjp_reduce_chunk_size": vjp_reduce_chunk_size,
        "initial_measured_loss": loss_history[0] if loss_history else None,
        "initial_measured_rgb_mse": rgb_loss_history[0] if rgb_loss_history else None,
        "final_train_mse": train_metrics["mse"],
        "final_train_psnr": train_metrics["psnr"],
        "final_train_l1": train_metrics["l1"],
        "final_heldout_mse": heldout_metrics["mse"],
        "final_heldout_psnr": heldout_metrics["psnr"],
        "final_heldout_l1": heldout_metrics["l1"],
        "step_summary": _summarize_steps(step_rows),
        "wall_timing": {
            **setup_timings,
            "train_loop_s": float(train_loop_s),
            "final_eval_s": float(final_eval_s),
            "total_run_s": float(time.perf_counter() - run_start),
        },
        "loss_summary": _summarize_values(
            step_rows,
            ("loss", "rgb_loss", "alpha_aux_loss", "depth_aux_loss"),
        ),
        "first_grad_abs_sum": first_grad_abs_sum,
        "first_alpha_output_grad_abs_sum": first_alpha_output_grad_abs_sum,
        "first_depth_output_grad_abs_sum": first_depth_output_grad_abs_sum,
        "loss_terms": {
            "alpha_aux_weight": float(alpha_aux_weight),
            "depth_aux_weight": float(depth_aux_weight),
            "alpha_depth_aux_active": bool(aux_loss_active),
        },
        "parameter_update_abs_max": param_update,
        "train_candidate_count": int(train_device["candidate_count"]),
        "heldout_candidate_count": int(heldout_device["candidate_count"]),
        "train_candidate_replay_iterations": int(train_device["candidate_replay_iterations"]),
        "heldout_candidate_replay_iterations": int(heldout_device["candidate_replay_iterations"]),
        "train_candidate_depth_order": train_device["candidate_depth_order"],
        "heldout_candidate_depth_order": heldout_device["candidate_depth_order"],
        "train_max_candidates_per_row": int(train_device["max_candidates_per_row"]),
        "heldout_max_candidates_per_row": int(heldout_device["max_candidates_per_row"]),
        "max_realray_boundaries": max_realray_boundaries,
        "train_avg_candidates_per_row": float(train_device["avg_candidates_per_row"]),
        "heldout_avg_candidates_per_row": float(heldout_device["avg_candidates_per_row"]),
        "train_mixed_tape_storage_bytes": _mixed_tape_storage_bytes(train_tape),
        "heldout_mixed_tape_storage_bytes": _mixed_tape_storage_bytes(heldout_tape),
        "train_explicit_ray_storage_bytes": _storage_bytes(train_tape.explicit_rays),
        "heldout_explicit_ray_storage_bytes": _storage_bytes(heldout_tape.explicit_rays),
        "train_compiled_boundary_test_ratio": float(train_tape.compiled_boundary_tests)
        / float(max(train_tape.direct_boundary_iterations, 1)),
        "heldout_compiled_boundary_test_ratio": float(heldout_tape.compiled_boundary_tests)
        / float(max(heldout_tape.direct_boundary_iterations, 1)),
        "acceptance": acceptance,
        "status": "ok" if all(acceptance.values()) else "failed",
    }


def run_train_eval(
    *,
    config_path: Path,
    frame_counts: tuple[int, ...],
    render_size: int,
    site_count: int,
    time_slabs: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    residual_depth_padding: float,
    synthetic_motion: SyntheticRayMotion,
    steps: int,
    warmup_steps: int,
    lr: float,
    vjp_reduce_chunk_size: int,
    vjp_mode: str,
    alpha_aux_weight: float,
    depth_aux_weight: float,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if vjp_reduce_chunk_size <= 0:
        raise ValueError("vjp_reduce_chunk_size must be positive")
    if alpha_aux_weight < 0.0 or depth_aux_weight < 0.0:
        raise ValueError("alpha/depth aux weights must be nonnegative")
    if vjp_mode not in {
        "reduce",
        "direct_atomic",
        "direct_atomic_grad_only",
        "direct_atomic_grad_only_ownerupdate",
        "direct_atomic_rgb_only",
        "direct_atomic_track",
        "fused_mse_rgb_only",
    }:
        raise ValueError(
            "vjp_mode must be 'reduce', 'direct_atomic', 'direct_atomic_grad_only', "
            "'direct_atomic_grad_only_ownerupdate', 'direct_atomic_rgb_only', "
            "'direct_atomic_track', or 'fused_mse_rgb_only'"
        )
    if vjp_mode == "fused_mse_rgb_only" and (alpha_aux_weight > 0.0 or depth_aux_weight > 0.0):
        raise ValueError("fused_mse_rgb_only supports RGB MSE only; alpha/depth aux weights must be zero")
    shared_load_start = time.perf_counter()
    shared_data_cfg = _load_config(config_path, max_frames=max(frame_counts), render_size=render_size)
    shared_training_data = load_powerfoam_training_data(shared_data_cfg, torch.device("cpu"))
    shared_load_s = time.perf_counter() - shared_load_start
    rows = [
        _run_one(
            config_path=config_path,
            frame_count=frame_count,
            render_size=render_size,
            site_count=site_count,
            time_slabs=time_slabs,
            near=near,
            far=far,
            density=density,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
            residual_depth_padding=residual_depth_padding,
            synthetic_motion=synthetic_motion,
            steps=steps,
            warmup_steps=warmup_steps,
            lr=lr,
            vjp_reduce_chunk_size=vjp_reduce_chunk_size,
            vjp_mode=vjp_mode,
            alpha_aux_weight=alpha_aux_weight,
            depth_aux_weight=depth_aux_weight,
            cached_training_data=shared_training_data,
        )
        for frame_count in frame_counts
    ]
    return {
        "benchmark": "world_foam_lane2_fused_slab_mixed_train_eval_mps",
        "status": "ok" if all(row["status"] == "ok" for row in rows) else "failed",
        "gate": "mixed_num32_den16_affine_moving_ray_site_rgba_train_eval",
        "device": "mps",
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "time_slabs": time_slabs,
        "layout": "per-track",
        "candidate_order": "slab-mid-depth",
        "gradient_scope": f"frozen_geometry_site_rgba_only_mixed_num32_den16_vjp_{vjp_mode}",
        "loss_scope": "rgb_mse_plus_optional_alpha_depth_aux",
        "full_trainer_claim": False,
        "full_geometry_gradient_claim": False,
        "quality_claim": False,
        "synthetic_motion": synthetic_motion.to_dict(),
        "vjp_mode": vjp_mode,
        "vjp_reduce_chunk_size": vjp_reduce_chunk_size,
        "alpha_aux_weight": float(alpha_aux_weight),
        "depth_aux_weight": float(depth_aux_weight),
        "shared_loaded_data": {
            "enabled": True,
            "max_frame_count": int(max(frame_counts)),
            "load_s": float(shared_load_s),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/eval mixed affine fused slab World Foam on MPS.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--density", type=float, default=10.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--residual-depth-padding", type=float, default=0.001)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument(
        "--vjp-mode",
        choices=(
            "reduce",
            "direct_atomic",
            "direct_atomic_grad_only",
            "direct_atomic_grad_only_ownerupdate",
            "direct_atomic_rgb_only",
            "direct_atomic_track",
            "fused_mse_rgb_only",
        ),
        default="direct_atomic_rgb_only",
    )
    parser.add_argument("--vjp-reduce-chunk-size", type=int, default=16)
    parser.add_argument("--alpha-aux-weight", type=float, default=0.0)
    parser.add_argument("--depth-aux-weight", type=float, default=0.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "fused_slab_mixed_train_eval_mps.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_train_eval(
        config_path=args.config,
        frame_counts=_parse_int_list(args.frame_counts),
        render_size=args.render_size,
        site_count=args.site_count,
        time_slabs=args.time_slabs,
        near=args.near,
        far=args.far,
        density=args.density,
        invalid_epsilon=args.invalid_epsilon,
        transmittance_threshold=args.transmittance_threshold,
        residual_depth_padding=args.residual_depth_padding,
        synthetic_motion=SyntheticRayMotion(
            origin_velocity=(args.origin_velocity_x, args.origin_velocity_y, args.origin_velocity_z),
            direction_velocity=(args.direction_velocity_x, args.direction_velocity_y, args.direction_velocity_z),
        ),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        vjp_reduce_chunk_size=args.vjp_reduce_chunk_size,
        vjp_mode=args.vjp_mode,
        alpha_aux_weight=args.alpha_aux_weight,
        depth_aux_weight=args.depth_aux_weight,
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
