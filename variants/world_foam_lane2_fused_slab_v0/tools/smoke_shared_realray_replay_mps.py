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
    _frame_time,
    _load_config,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
    render_samples,
    write_ppm,
)
from gate2_realray_event_sharing import (  # noqa: E402
    event_set_for_ray,
    ray_time_delta,
    slab_event_set_for_ray,
)
from torch_world_foam_lane2_fused_slab import RealRayReplayConfig, shared_realray_rgba_depth_replay  # noqa: E402


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


def _build_candidate_masks(
    *,
    boundaries: tuple[Any, ...],
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
        raise ValueError("shared real-ray smoke currently supports at most 128 boundaries")
    if time_slabs <= 0:
        raise ValueError("time_slabs must be positive")

    word_count = max(1, (len(boundaries) + 31) // 32)
    mask_rows: list[list[int]] = []
    track_rays: list[torch.Tensor] = []
    per_frame_events = 0
    shared_candidate_events = 0
    missing_events = 0
    extra_candidate_events = 0
    invalid_per_frame = 0
    invalid_shared = 0
    max_candidates_per_slab = 0
    candidate_sets: dict[tuple[int, int], set[int]] = {}

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
                    candidate_sets[(track_id, slab_id)] = events
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
                    candidates = candidate_sets[(track_id, slab_id)]
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


def _mps_render_split(
    *,
    split: str,
    boundaries: tuple[Any, ...],
    sites_f32: torch.Tensor,
    boundary_f32: torch.Tensor,
    site_rgba_f32: torch.Tensor,
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    time_slabs: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    config: RealRayReplayConfig,
    timing_iters: int,
) -> dict[str, Any]:
    candidates = _build_candidate_masks(
        boundaries=boundaries,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        split=split,
    )
    device = torch.device("mps")
    frame_t_f32 = torch.tensor([_frame_time(frame, frame_count) for frame in range(frame_count)], dtype=torch.float32, device=device)
    track_rays_f32 = candidates["track_rays"].to(device)
    candidate_mask_i32 = candidates["candidate_mask"].to(device)
    output_rgb, output_alpha, output_depth = shared_realray_rgba_depth_replay(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        config,
    )
    torch.mps.synchronize()
    view_count = int(candidates["view_count"])
    height = int(candidates["height"])
    width = int(candidates["width"])
    rgb_cpu = (
        output_rgb.cpu()
        .reshape(view_count, height, width, frame_count, 3)
        .permute(0, 3, 4, 1, 2)
        .reshape(view_count * frame_count, 3, height, width)
        .contiguous()
    )
    alpha_cpu = (
        output_alpha.cpu()
        .reshape(view_count, height, width, frame_count)
        .permute(0, 3, 1, 2)
        .reshape(view_count * frame_count, height, width)
        .contiguous()
    )
    depth_cpu = (
        output_depth.cpu()
        .reshape(view_count, height, width, frame_count)
        .permute(0, 3, 1, 2)
        .reshape(view_count * frame_count, height, width)
        .contiguous()
    )

    started_at = time.perf_counter()
    timed = (output_rgb, output_alpha, output_depth)
    for _ in range(timing_iters):
        timed = shared_realray_rgba_depth_replay(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            config,
        )
    torch.mps.synchronize()
    _timed_shapes = tuple(tensor.shape for tensor in timed)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0 / float(timing_iters)
    return {
        "rgb": rgb_cpu,
        "alpha": alpha_cpu,
        "depth": depth_cpu,
        "mps_shared_realray_forward_wall_clock_ms": float(elapsed_ms),
        **{key: value for key, value in candidates.items() if key not in {"track_rays", "candidate_mask"}},
    }


def _split_summary(*, split: str, mps: dict[str, Any], cpu: dict[str, Any]) -> dict[str, Any]:
    mps_rgb = mps["rgb"]
    mps_alpha = mps["alpha"]
    mps_depth = mps["depth"]
    cpu_rgb = cpu["rgb"]
    cpu_alpha = cpu["alpha"]
    cpu_depth = cpu["depth"]
    return {
        "split": split,
        "rgb_shape": list(mps_rgb.shape),
        "alpha_shape": list(mps_alpha.shape),
        "depth_shape": list(mps_depth.shape),
        "pixel_tracks": int(mps["pixel_tracks"]),
        "pixel_rays": int(mps["pixel_rays"]),
        "candidate_mask_shape": mps["candidate_mask_shape"],
        "mask_word_count": int(mps["word_count"]),
        "per_frame_event_sum": int(mps["per_frame_event_sum"]),
        "shared_slab_event_sum": int(mps["shared_slab_event_sum"]),
        "event_sharing_ratio": float(mps["event_sharing_ratio"]),
        "missing_sample_events": int(mps["missing_sample_events"]),
        "extra_candidate_events": int(mps["extra_candidate_events"]),
        "max_candidates_per_slab": int(mps["max_candidates_per_slab"]),
        "direct_forward_boundary_scans": int(mps["direct_forward_boundary_scans"]),
        "shared_forward_boundary_scans": int(mps["shared_forward_boundary_scans"]),
        "shared_forward_boundary_scan_ratio": float(mps["shared_forward_boundary_scan_ratio"]),
        "invalid_per_frame_denominator_count": int(mps["invalid_per_frame_denominator_count"]),
        "invalid_shared_denominator_count": int(mps["invalid_shared_denominator_count"]),
        "max_rgb_abs_error": float((mps_rgb - cpu_rgb).abs().max().item()),
        "max_alpha_abs_error": float((mps_alpha - cpu_alpha).abs().max().item()),
        "max_depth_abs_error": float((mps_depth - cpu_depth).abs().max().item()),
        "mps_shared_realray_forward_wall_clock_ms": float(mps["mps_shared_realray_forward_wall_clock_ms"]),
        "cpu_render_elapsed_s": float(cpu["elapsed_s"]),
        "mps_rgb_std": float(mps_rgb.std().item()),
        "mps_alpha_min": float(mps_alpha.min().item()),
        "mps_alpha_max": float(mps_alpha.max().item()),
        "mps_depth_min": float(mps_depth.min().item()),
        "mps_depth_max": float(mps_depth.max().item()),
        "ray_time_delta": mps["ray_time_delta"],
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
    train_ppm_out: Path | None,
    heldout_ppm_out: Path | None,
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
        raise ValueError("shared real-ray MPS smoke requires heldout targets, rays, and frame indices")

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
    site_rgba_f32 = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    cpu_train = render_samples(
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
    cpu_heldout = render_samples(
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
    mps_train = _mps_render_split(
        split="train",
        boundaries=boundaries,
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        rays=sample_rays,
        frame_indices=sample_frame_indices,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        config=op_config,
        timing_iters=timing_iters,
    )
    mps_heldout = _mps_render_split(
        split="heldout",
        boundaries=boundaries,
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        rays=heldout_rays.detach().cpu().to(dtype=torch.float32),
        frame_indices=heldout_frame_indices.detach().cpu().to(dtype=torch.long),
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        config=op_config,
        timing_iters=timing_iters,
    )
    if train_ppm_out is not None:
        write_ppm(train_ppm_out, mps_train["rgb"][0])
    if heldout_ppm_out is not None:
        write_ppm(heldout_ppm_out, mps_heldout["rgb"][0])

    train = _split_summary(split="train", mps=mps_train, cpu=cpu_train)
    heldout = _split_summary(split="heldout", mps=mps_heldout, cpu=cpu_heldout)
    tolerance = 5.0e-4
    max_error = max(
        train["max_rgb_abs_error"],
        train["max_alpha_abs_error"],
        train["max_depth_abs_error"],
        heldout["max_rgb_abs_error"],
        heldout["max_alpha_abs_error"],
        heldout["max_depth_abs_error"],
    )
    acceptance = {
        "loaded_real_multicam_bundle": str(cfg["data"]["frame_source"]) == "multicam_val",
        "consumed_train_camera_rays": list(sample_rays.shape) == [targets.shape[0], targets.shape[2], targets.shape[3], 6],
        "consumed_heldout_camera_rays": list(heldout_rays.shape)
        == [heldout_targets.shape[0], heldout_targets.shape[2], heldout_targets.shape[3], 6],
        "real_rays_static_within_train_views_over_time": bool(
            train["ray_time_delta"]["max_origin_delta_within_view_over_time"] == 0.0
            and train["ray_time_delta"]["max_direction_delta_within_view_over_time"] == 0.0
        ),
        "real_rays_static_within_heldout_views_over_time": bool(
            heldout["ray_time_delta"]["max_origin_delta_within_view_over_time"] == 0.0
            and heldout["ray_time_delta"]["max_direction_delta_within_view_over_time"] == 0.0
        ),
        "zero_missing_sample_events": train["missing_sample_events"] == 0 and heldout["missing_sample_events"] == 0,
        "shared_outputs_match_direct_cpu": max_error <= tolerance,
        "shared_scan_ratio_sublinear": train["shared_forward_boundary_scan_ratio"] <= 1.0
        and heldout["shared_forward_boundary_scan_ratio"] <= 1.0,
        "all_outputs_finite": bool(
            torch.isfinite(mps_train["rgb"]).all().item()
            and torch.isfinite(mps_train["alpha"]).all().item()
            and torch.isfinite(mps_train["depth"]).all().item()
            and torch.isfinite(mps_heldout["rgb"]).all().item()
            and torch.isfinite(mps_heldout["alpha"]).all().item()
            and torch.isfinite(mps_heldout["depth"]).all().item()
        ),
        "alpha_in_unit_interval": bool(
            mps_train["alpha"].min().item() >= -1.0e-6
            and mps_train["alpha"].max().item() <= 1.0 + 1.0e-6
            and mps_heldout["alpha"].min().item() >= -1.0e-6
            and mps_heldout["alpha"].max().item() <= 1.0 + 1.0e-6
        ),
        "uses_4d_power_boundaries": len(boundaries) == site_count * (site_count - 1) // 2,
    }
    return {
        "benchmark": "world_foam_lane2_gate2b_mps_shared_realray_forward_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "2B_realray_mps_shared_forward",
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
        "renderer_scope": "mps_real_camera_ray_4d_power_cell_time_slab_shared_forward",
        "gradient_scope": "none_forward_only_no_backward",
        "sharing_scope": "mps_real_camera_ray_time_slab_candidate_forward",
        "quality_claim": False,
        "world_foam_renderer_status": "mps_shared_real_camera_ray_forward_no_backward_no_training",
        "tolerance": tolerance,
        "timing_iters": timing_iters,
        "train": train,
        "heldout": heldout,
        "acceptance": acceptance,
        "proof_images": {
            "train_ppm": str(train_ppm_out) if train_ppm_out is not None else None,
            "heldout_ppm": str(heldout_ppm_out) if heldout_ppm_out is not None else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke World Foam shared true real-camera-ray MPS forward replay.")
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
    parser.add_argument("--timing-iters", type=int, default=20)
    parser.add_argument("--train-ppm-out", type=Path)
    parser.add_argument("--heldout-ppm-out", type=Path)
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
