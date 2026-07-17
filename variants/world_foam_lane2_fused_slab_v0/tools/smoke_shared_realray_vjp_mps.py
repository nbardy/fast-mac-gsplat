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

from gate1_realray_per_sample_reference import (  # noqa: E402
    DEFAULT_CONFIG,
    EPS,
    Boundary4D,
    Site4D,
    _frame_time,
    _load_config,
    crossing_depth_4d,
    dedupe_sorted_depths,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
    owner_at_4d,
    render_samples,
)
from gate2_realray_event_sharing import event_set_for_ray, ray_time_delta, slab_event_set_for_ray  # noqa: E402
from torch_world_foam_lane2_fused_slab import RealRayReplayConfig, shared_realray_rgba_depth_vjp  # noqa: E402


def _signed_i32_word(word: int) -> int:
    return word - (1 << 32) if word >= (1 << 31) else word


def _ray_tuple(ray: torch.Tensor) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(ray[0].item()), float(ray[1].item()), float(ray[2].item())),
        (float(ray[3].item()), float(ray[4].item()), float(ray[5].item())),
    )


def _validate_view_major_frames(*, frame_indices: torch.Tensor, frame_count: int, split: str) -> int:
    if frame_indices.ndim != 1:
        raise ValueError(f"{split} frame_indices must be rank-1")
    if frame_indices.shape[0] % frame_count != 0:
        raise ValueError(f"{split} sample count must be view_count * frame_count")
    view_count = int(frame_indices.shape[0] // frame_count)
    for view in range(view_count):
        for frame in range(frame_count):
            actual = int(frame_indices[view * frame_count + frame].item())
            if actual != frame:
                raise ValueError(f"{split} frame index order mismatch at view {view}, frame {frame}: {actual}")
    return view_count


def _build_candidate_bundle(
    *,
    boundaries: tuple[Boundary4D, ...],
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    time_slabs: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    split: str,
) -> dict[str, Any]:
    rays_cpu = rays.detach().cpu().to(dtype=torch.float32)
    frame_indices_cpu = frame_indices.detach().cpu().to(dtype=torch.long)
    sample_count, height, width, payload = rays_cpu.shape
    if payload != 6:
        raise ValueError(f"{split} rays must have payload dimension 6")
    view_count = _validate_view_major_frames(frame_indices=frame_indices_cpu, frame_count=frame_count, split=split)
    if sample_count != view_count * frame_count:
        raise ValueError(f"{split} rays sample count mismatch")
    if len(boundaries) > 128:
        raise ValueError("shared real-ray VJP smoke currently supports at most 128 boundaries")
    if time_slabs <= 0:
        raise ValueError("time_slabs must be positive")

    word_count = max(1, (len(boundaries) + 31) // 32)
    mask_rows: list[list[int]] = []
    track_rays: list[torch.Tensor] = []
    candidate_sets: dict[tuple[int, int], tuple[int, ...]] = {}
    per_frame_events = 0
    shared_candidate_events = 0
    missing_events = 0
    extra_candidate_events = 0
    invalid_per_frame = 0
    invalid_shared = 0
    max_candidates_per_slab = 0

    for view in range(view_count):
        base_rays = rays_cpu[view * frame_count]
        for y in range(height):
            for x in range(width):
                track_id = len(track_rays)
                ray = base_rays[y, x].contiguous()
                track_rays.append(ray)
                origin, direction = _ray_tuple(ray)
                for slab_id in range(time_slabs):
                    t0 = float(slab_id) / float(time_slabs)
                    t1 = float(slab_id + 1) / float(time_slabs)
                    events, invalid = slab_event_set_for_ray(
                        boundaries=boundaries,
                        origin=origin,
                        direction=direction,
                        t0=t0,
                        t1=t1,
                        near=near,
                        far=far,
                        invalid_epsilon=invalid_epsilon,
                    )
                    words = [0] * word_count
                    for boundary_id in events:
                        words[boundary_id // 32] |= 1 << (boundary_id % 32)
                    mask_rows.append([_signed_i32_word(word) for word in words])
                    candidate_sets[(track_id, slab_id)] = tuple(sorted(events))
                    shared_candidate_events += len(events)
                    max_candidates_per_slab = max(max_candidates_per_slab, len(events))
                    invalid_shared += invalid

    for view in range(view_count):
        for frame in range(frame_count):
            sample_index = view * frame_count + frame
            t = _frame_time(int(frame_indices_cpu[sample_index].item()), frame_count)
            slab_id = min(int(math.floor(t * time_slabs)), time_slabs - 1)
            for y in range(height):
                for x in range(width):
                    track_id = view * height * width + y * width + x
                    origin, direction = _ray_tuple(rays_cpu[sample_index, y, x])
                    sample_events, invalid = event_set_for_ray(
                        boundaries=boundaries,
                        origin=origin,
                        direction=direction,
                        t=t,
                        near=near,
                        far=far,
                        invalid_epsilon=invalid_epsilon,
                    )
                    candidates = set(candidate_sets[(track_id, slab_id)])
                    per_frame_events += len(sample_events)
                    missing_events += len(sample_events - candidates)
                    extra_candidate_events += len(candidates - sample_events)
                    invalid_per_frame += invalid

    track_rays_tensor = torch.stack(track_rays, dim=0).contiguous()
    mask_tensor = torch.tensor(mask_rows, dtype=torch.int32).contiguous()
    pixel_tracks = int(view_count * height * width)
    pixel_rays = int(view_count * frame_count * height * width)
    return {
        "track_rays": track_rays_tensor,
        "candidate_mask": mask_tensor,
        "candidate_sets": candidate_sets,
        "view_count": view_count,
        "height": int(height),
        "width": int(width),
        "word_count": int(word_count),
        "pixel_tracks": pixel_tracks,
        "pixel_rays": pixel_rays,
        "per_frame_event_sum": int(per_frame_events),
        "shared_slab_event_sum": int(shared_candidate_events),
        "event_sharing_ratio": float(shared_candidate_events) / float(max(per_frame_events, 1)),
        "missing_sample_events": int(missing_events),
        "extra_candidate_events": int(extra_candidate_events),
        "invalid_per_frame_denominator_count": int(invalid_per_frame),
        "invalid_shared_denominator_count": int(invalid_shared),
        "max_candidates_per_slab": int(max_candidates_per_slab),
        "candidate_mask_shape": list(mask_tensor.shape),
        "direct_forward_boundary_scans": int(pixel_rays * len(boundaries)),
        "shared_forward_boundary_scans": int(pixel_tracks * time_slabs * len(boundaries)),
        "shared_forward_boundary_scan_ratio": float(pixel_tracks * time_slabs * len(boundaries))
        / float(max(pixel_rays * len(boundaries), 1)),
        "ray_time_delta": ray_time_delta(rays_cpu, view_count=view_count, frame_count=frame_count),
    }


def _candidate_depths_from_ids(
    *,
    boundaries: tuple[Boundary4D, ...],
    candidate_ids: tuple[int, ...],
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    t: float,
    near: float,
    far: float,
    invalid_epsilon: float,
) -> list[float]:
    depths: list[float] = []
    for boundary_id in candidate_ids:
        depth = crossing_depth_4d(
            boundaries[boundary_id],
            origin=origin,
            direction=direction,
            t=t,
            invalid_epsilon=invalid_epsilon,
        )
        if depth is not None and near <= depth <= far:
            depths.append(depth)
    return dedupe_sorted_depths(depths)


def _render_vjp_one(
    *,
    sites: tuple[Site4D, ...],
    boundaries: tuple[Boundary4D, ...],
    candidate_ids: tuple[int, ...],
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    t: float,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    grad_rgb: tuple[float, float, float],
    grad_alpha: float,
    grad_depth: float,
) -> tuple[tuple[float, float, float], float, float, torch.Tensor, int]:
    depths = _candidate_depths_from_ids(
        boundaries=boundaries,
        candidate_ids=candidate_ids,
        origin=origin,
        direction=direction,
        t=t,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
    )
    cuts = [near, *depths, far]
    owners: list[int] = []
    lengths: list[float] = []
    mids: list[float] = []
    trans_before: list[float] = []
    segment_trans: list[float] = []
    segment_alpha: list[float] = []
    weights: list[float] = []
    segment_rgb: list[tuple[float, float, float]] = []
    rgb = [0.0, 0.0, 0.0]
    alpha_accum = 0.0
    depth_weighted = 0.0
    transmittance = 1.0
    ox, oy, oz = origin
    dx, dy, dz = direction
    for depth0, depth1 in zip(cuts, cuts[1:]):
        length = depth1 - depth0
        if length <= EPS or transmittance <= transmittance_threshold:
            continue
        mid = 0.5 * (depth0 + depth1)
        x = ox + dx * mid
        y = oy + dy * mid
        z = oz + dz * mid
        owner = owner_at_4d(sites, x=x, y=y, z=z, t=t)
        site = sites[owner]
        density = max(float(site.rgba[3]), 0.0)
        seg_trans = math.exp(-density * length)
        seg_alpha = 1.0 - seg_trans
        weight = transmittance * seg_alpha
        color = (float(site.rgba[0]), float(site.rgba[1]), float(site.rgba[2]))
        owners.append(owner)
        lengths.append(length)
        mids.append(mid)
        trans_before.append(transmittance)
        segment_trans.append(seg_trans)
        segment_alpha.append(seg_alpha)
        weights.append(weight)
        segment_rgb.append(color)
        rgb[0] += weight * color[0]
        rgb[1] += weight * color[1]
        rgb[2] += weight * color[2]
        alpha_accum += weight
        depth_weighted += weight * mid
        transmittance *= seg_trans
    expected_depth = depth_weighted / alpha_accum if alpha_accum > EPS else far

    grad_samples = torch.zeros((len(sites), 4), dtype=torch.float32)
    adj_next_transmittance = 0.0
    for segment_id in range(len(owners) - 1, -1, -1):
        owner = owners[segment_id]
        d_loss_d_weight = (
            grad_rgb[0] * segment_rgb[segment_id][0]
            + grad_rgb[1] * segment_rgb[segment_id][1]
            + grad_rgb[2] * segment_rgb[segment_id][2]
            + grad_alpha
        )
        if alpha_accum > EPS:
            d_loss_d_weight += grad_depth * (
                (mids[segment_id] * alpha_accum - depth_weighted) / (alpha_accum * alpha_accum)
            )
        grad_samples[owner, 0] += weights[segment_id] * grad_rgb[0]
        grad_samples[owner, 1] += weights[segment_id] * grad_rgb[1]
        grad_samples[owner, 2] += weights[segment_id] * grad_rgb[2]
        adj_trans_before = d_loss_d_weight * segment_alpha[segment_id] + adj_next_transmittance * segment_trans[segment_id]
        adj_segment_alpha = d_loss_d_weight * trans_before[segment_id]
        adj_segment_trans = adj_next_transmittance * trans_before[segment_id] - adj_segment_alpha
        raw_density = float(sites[owner].rgba[3])
        if raw_density > 0.0:
            grad_samples[owner, 3] += adj_segment_trans * (-lengths[segment_id] * segment_trans[segment_id])
        adj_next_transmittance = adj_trans_before
    return (rgb[0], rgb[1], rgb[2]), alpha_accum, expected_depth, grad_samples, len(owners)


def _make_gradients(track_count: int, frame_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.arange(track_count * frame_count, dtype=torch.float32).reshape(track_count, frame_count)
    grad_rgb = torch.stack(
        (
            0.17 + 0.03 * torch.sin(ids * 0.013),
            -0.11 + 0.02 * torch.cos(ids * 0.017),
            0.07 + 0.025 * torch.sin(ids * 0.019 + 0.3),
        ),
        dim=2,
    ).contiguous()
    grad_alpha = (0.05 + 0.01 * torch.cos(ids * 0.011)).contiguous()
    grad_depth = (-0.025 + 0.006 * torch.sin(ids * 0.007)).contiguous()
    return grad_rgb, grad_alpha, grad_depth


def _cpu_vjp_split(
    *,
    sites: tuple[Site4D, ...],
    boundaries: tuple[Boundary4D, ...],
    bundle: dict[str, Any],
    frame_count: int,
    time_slabs: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    grad_rgb: torch.Tensor,
    grad_alpha: torch.Tensor,
    grad_depth: torch.Tensor,
) -> dict[str, Any]:
    track_rays = bundle["track_rays"]
    candidate_sets = bundle["candidate_sets"]
    track_count = int(track_rays.shape[0])
    output_rgb = torch.empty((track_count, frame_count, 3), dtype=torch.float32)
    output_alpha = torch.empty((track_count, frame_count), dtype=torch.float32)
    output_depth = torch.empty((track_count, frame_count), dtype=torch.float32)
    grad_samples = torch.empty((track_count, frame_count, len(sites), 4), dtype=torch.float32)
    total_segments = 0
    max_segments = 0
    started_at = time.perf_counter()
    for track_id in range(track_count):
        origin, direction = _ray_tuple(track_rays[track_id])
        for frame in range(frame_count):
            t = _frame_time(frame, frame_count)
            slab_id = min(int(math.floor(t * time_slabs)), time_slabs - 1)
            rgb, alpha, depth, sample_grad, segment_count = _render_vjp_one(
                sites=sites,
                boundaries=boundaries,
                candidate_ids=candidate_sets[(track_id, slab_id)],
                origin=origin,
                direction=direction,
                t=t,
                near=near,
                far=far,
                invalid_epsilon=invalid_epsilon,
                transmittance_threshold=transmittance_threshold,
                grad_rgb=(
                    float(grad_rgb[track_id, frame, 0].item()),
                    float(grad_rgb[track_id, frame, 1].item()),
                    float(grad_rgb[track_id, frame, 2].item()),
                ),
                grad_alpha=float(grad_alpha[track_id, frame].item()),
                grad_depth=float(grad_depth[track_id, frame].item()),
            )
            output_rgb[track_id, frame, 0] = rgb[0]
            output_rgb[track_id, frame, 1] = rgb[1]
            output_rgb[track_id, frame, 2] = rgb[2]
            output_alpha[track_id, frame] = alpha
            output_depth[track_id, frame] = depth
            grad_samples[track_id, frame] = sample_grad
            total_segments += segment_count
            max_segments = max(max_segments, segment_count)
    elapsed_s = time.perf_counter() - started_at
    loss = (
        (output_rgb * grad_rgb).sum()
        + (output_alpha * grad_alpha).sum()
        + (output_depth * grad_depth).sum()
    )
    return {
        "rgb": output_rgb,
        "alpha": output_alpha,
        "depth": output_depth,
        "grad_samples_rgba": grad_samples,
        "loss": loss,
        "elapsed_s": elapsed_s,
        "total_segments": total_segments,
        "max_segments_per_ray": max_segments,
    }


def _mps_vjp_split(
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
    frame_t_f32 = torch.tensor([_frame_time(frame, frame_count) for frame in range(frame_count)], dtype=torch.float32, device=device)
    track_rays_f32 = bundle["track_rays"].to(device)
    candidate_mask_i32 = bundle["candidate_mask"].to(device)
    grad_rgb_cpu, grad_alpha_cpu, grad_depth_cpu = _make_gradients(track_count=track_rays_f32.shape[0], frame_count=frame_count)
    grad_rgb_f32 = grad_rgb_cpu.to(device)
    grad_alpha_f32 = grad_alpha_cpu.to(device)
    grad_depth_f32 = grad_depth_cpu.to(device)
    output_rgb, output_alpha, output_depth, grad_samples = shared_realray_rgba_depth_vjp(
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
    started_at = time.perf_counter()
    timed = (output_rgb, output_alpha, output_depth, grad_samples)
    for _ in range(timing_iters):
        timed = shared_realray_rgba_depth_vjp(
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
        "grad_samples_rgba": grad_samples.cpu(),
        "grad_rgb": grad_rgb_cpu,
        "grad_alpha": grad_alpha_cpu,
        "grad_depth": grad_depth_cpu,
        "mps_shared_realray_vjp_wall_clock_ms": float(elapsed_ms),
    }


def _reshape_rgb(rgb_tracks: torch.Tensor, *, view_count: int, frame_count: int, height: int, width: int) -> torch.Tensor:
    return (
        rgb_tracks.reshape(view_count, height, width, frame_count, 3)
        .permute(0, 3, 4, 1, 2)
        .reshape(view_count * frame_count, 3, height, width)
        .contiguous()
    )


def _split_summary(
    *,
    split: str,
    mps: dict[str, Any],
    cpu: dict[str, Any],
    direct_cpu: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    view_count = int(bundle["view_count"])
    frame_count = int(mps["alpha"].shape[1])
    height = int(bundle["height"])
    width = int(bundle["width"])
    mps_rgb_image = _reshape_rgb(mps["rgb"], view_count=view_count, frame_count=frame_count, height=height, width=width)
    direct_rgb = direct_cpu["rgb"]
    loss_mps = (
        (mps["rgb"] * mps["grad_rgb"]).sum()
        + (mps["alpha"] * mps["grad_alpha"]).sum()
        + (mps["depth"] * mps["grad_depth"]).sum()
    )
    return {
        "split": split,
        "rgb_shape": list(mps_rgb_image.shape),
        "alpha_shape": [view_count * frame_count, height, width],
        "depth_shape": [view_count * frame_count, height, width],
        "gradient_shape": list(mps["grad_samples_rgba"].shape),
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
        "max_rgb_abs_error": float((mps["rgb"] - cpu["rgb"]).abs().max().item()),
        "max_alpha_abs_error": float((mps["alpha"] - cpu["alpha"]).abs().max().item()),
        "max_depth_abs_error": float((mps["depth"] - cpu["depth"]).abs().max().item()),
        "max_direct_cpu_rgb_abs_error": float((mps_rgb_image - direct_rgb).abs().max().item()),
        "max_rgba_gradient_abs_error": float((mps["grad_samples_rgba"] - cpu["grad_samples_rgba"]).abs().max().item()),
        "loss_abs_error": float((loss_mps - cpu["loss"]).abs().item()),
        "cpu_vjp_elapsed_s": float(cpu["elapsed_s"]),
        "cpu_render_elapsed_s": float(direct_cpu["elapsed_s"]),
        "mps_shared_realray_vjp_wall_clock_ms": float(mps["mps_shared_realray_vjp_wall_clock_ms"]),
        "cpu_total_segments": int(cpu["total_segments"]),
        "cpu_max_segments_per_ray": int(cpu["max_segments_per_ray"]),
        "grad_samples_abs_max": float(mps["grad_samples_rgba"].abs().max().item()),
        "grad_samples_abs_sum": float(mps["grad_samples_rgba"].abs().sum().item()),
        "mps_rgb_std": float(mps["rgb"].std().item()),
        "mps_alpha_min": float(mps["alpha"].min().item()),
        "mps_alpha_max": float(mps["alpha"].max().item()),
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
        raise ValueError("shared real-ray VJP smoke requires heldout targets, rays, and frame indices")

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
    mps_train = _mps_vjp_split(
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
        grad_rgb=mps_train["grad_rgb"],
        grad_alpha=mps_train["grad_alpha"],
        grad_depth=mps_train["grad_depth"],
    )
    mps_heldout = _mps_vjp_split(
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
        grad_rgb=mps_heldout["grad_rgb"],
        grad_alpha=mps_heldout["grad_alpha"],
        grad_depth=mps_heldout["grad_depth"],
    )

    train = _split_summary(split="train", mps=mps_train, cpu=cpu_train, direct_cpu=cpu_train_render, bundle=train_bundle)
    heldout = _split_summary(split="heldout", mps=mps_heldout, cpu=cpu_heldout, direct_cpu=cpu_heldout_render, bundle=heldout_bundle)
    tolerance = 5.0e-4
    max_forward_error = max(
        train["max_rgb_abs_error"],
        train["max_alpha_abs_error"],
        train["max_depth_abs_error"],
        heldout["max_rgb_abs_error"],
        heldout["max_alpha_abs_error"],
        heldout["max_depth_abs_error"],
    )
    max_gradient_error = max(train["max_rgba_gradient_abs_error"], heldout["max_rgba_gradient_abs_error"])
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
        "shared_rgba_gradients_match_cpu_vjp_reference": max_gradient_error <= tolerance,
        "loss_matches_cpu_vjp_reference": train["loss_abs_error"] <= tolerance and heldout["loss_abs_error"] <= tolerance,
        "shared_scan_ratio_sublinear": train["shared_forward_boundary_scan_ratio"] <= 1.0
        and heldout["shared_forward_boundary_scan_ratio"] <= 1.0,
        "all_gradients_finite": bool(
            torch.isfinite(mps_train["grad_samples_rgba"]).all().item()
            and torch.isfinite(mps_heldout["grad_samples_rgba"]).all().item()
        ),
        "gradients_nonzero": train["grad_samples_abs_sum"] > 0.0 and heldout["grad_samples_abs_sum"] > 0.0,
        "alpha_in_unit_interval": bool(
            mps_train["alpha"].min().item() >= -1.0e-6
            and mps_train["alpha"].max().item() <= 1.0 + 1.0e-6
            and mps_heldout["alpha"].min().item() >= -1.0e-6
            and mps_heldout["alpha"].max().item() <= 1.0 + 1.0e-6
        ),
    }
    return {
        "benchmark": "world_foam_lane2_gate2c_mps_shared_realray_vjp_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "2C_realray_mps_shared_fixed_segment_vjp",
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
        "renderer_scope": "mps_real_camera_ray_4d_power_cell_time_slab_shared_forward_and_fixed_segment_vjp",
        "gradient_scope": "fixed_segment_site_rgba_only_no_geometry_or_topology_gradients",
        "sharing_scope": "mps_real_camera_ray_time_slab_candidate_forward_and_fixed_segment_vjp",
        "quality_claim": False,
        "training_claim": False,
        "world_foam_renderer_status": "mps_shared_real_camera_ray_fixed_segment_vjp_no_geometry_topology_gradients_no_training",
        "tolerance": tolerance,
        "timing_iters": timing_iters,
        "train": train,
        "heldout": heldout,
        "acceptance": acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke World Foam shared true real-camera-ray MPS fixed-segment VJP.")
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
