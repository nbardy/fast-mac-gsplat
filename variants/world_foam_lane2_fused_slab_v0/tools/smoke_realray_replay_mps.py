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
    _frame_time,
    _load_config,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
    metric_block,
    render_samples,
    write_ppm,
)
from torch_world_foam_lane2_fused_slab import RealRayReplayConfig, realray_rgba_depth_replay  # noqa: E402


def flatten_rays_and_times(*, rays: torch.Tensor, frame_indices: torch.Tensor, frame_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if rays.ndim != 4 or rays.shape[-1] != 6:
        raise ValueError(f"Expected rays [B,H,W,6], got {tuple(rays.shape)}.")
    sample_count, height, width, _payload = rays.shape
    rays_flat = rays.reshape(sample_count * height * width, 6).contiguous()
    times = torch.tensor(
        [_frame_time(int(frame_indices[sample_index].item()), frame_count) for sample_index in range(sample_count)],
        dtype=torch.float32,
    )
    return rays_flat, times[:, None, None].expand(sample_count, height, width).reshape(-1).contiguous()


def mps_render_split(
    *,
    sites_f32: torch.Tensor,
    boundary_f32: torch.Tensor,
    site_rgba_f32: torch.Tensor,
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    config: RealRayReplayConfig,
    timing_iters: int,
) -> dict[str, Any]:
    sample_count, height, width, _payload = rays.shape
    rays_flat_cpu, frame_t_cpu = flatten_rays_and_times(
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
    )
    device = torch.device("mps")
    rays_f32 = rays_flat_cpu.to(device)
    frame_t_f32 = frame_t_cpu.to(device)
    output_rgb, output_alpha, output_depth = realray_rgba_depth_replay(
        boundary_f32,
        sites_f32,
        site_rgba_f32,
        rays_f32,
        frame_t_f32,
        config,
    )
    torch.mps.synchronize()
    rgb_cpu = output_rgb.cpu().reshape(sample_count, height, width, 3).permute(0, 3, 1, 2).contiguous()
    alpha_cpu = output_alpha.cpu().reshape(sample_count, height, width).contiguous()
    depth_cpu = output_depth.cpu().reshape(sample_count, height, width).contiguous()

    started_at = time.perf_counter()
    timed = (output_rgb, output_alpha, output_depth)
    for _ in range(timing_iters):
        timed = realray_rgba_depth_replay(
            boundary_f32,
            sites_f32,
            site_rgba_f32,
            rays_f32,
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
        "mps_realray_replay_wall_clock_ms": elapsed_ms,
    }


def split_summary(
    *,
    split: str,
    mps: dict[str, Any],
    cpu: dict[str, Any],
    target: torch.Tensor,
    boundary_count: int,
) -> dict[str, Any]:
    mps_rgb = mps["rgb"]
    mps_alpha = mps["alpha"]
    mps_depth = mps["depth"]
    cpu_rgb = cpu["rgb"]
    cpu_alpha = cpu["alpha"]
    cpu_depth = cpu["depth"]
    metrics = metric_block(mps_rgb, target)
    pixel_ray_count = int(mps_rgb.shape[0] * mps_rgb.shape[2] * mps_rgb.shape[3])
    return {
        "split": split,
        "rgb_shape": list(mps_rgb.shape),
        "alpha_shape": list(mps_alpha.shape),
        "depth_shape": list(mps_depth.shape),
        "pixel_ray_count": pixel_ray_count,
        "linear_boundary_scans": int(pixel_ray_count * boundary_count),
        "max_rgb_abs_error": float((mps_rgb - cpu_rgb).abs().max().item()),
        "max_alpha_abs_error": float((mps_alpha - cpu_alpha).abs().max().item()),
        "max_depth_abs_error": float((mps_depth - cpu_depth).abs().max().item()),
        "target_l1": metrics["l1"],
        "target_mse": metrics["mse"],
        "target_psnr": metrics["psnr"],
        "mps_realray_replay_wall_clock_ms": float(mps["mps_realray_replay_wall_clock_ms"]),
        "cpu_render_elapsed_s": float(cpu["elapsed_s"]),
        "rgb_min": float(mps_rgb.min().item()),
        "rgb_max": float(mps_rgb.max().item()),
        "rgb_std": float(mps_rgb.std().item()),
        "alpha_min": float(mps_alpha.min().item()),
        "alpha_max": float(mps_alpha.max().item()),
        "alpha_std": float(mps_alpha.std().item()),
        "depth_min": float(mps_depth.min().item()),
        "depth_max": float(mps_depth.max().item()),
        "depth_std": float(mps_depth.std().item()),
    }


def run_smoke(
    *,
    config_path: Path,
    max_frames: int | None,
    render_size: int | None,
    site_count: int,
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
        raise ValueError("real-ray MPS smoke requires heldout targets, rays, and frame indices")

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
    mps_train = mps_render_split(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        rays=sample_rays,
        frame_indices=sample_frame_indices,
        frame_count=frame_count,
        config=op_config,
        timing_iters=timing_iters,
    )
    mps_heldout = mps_render_split(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        rays=heldout_rays.detach().cpu().to(dtype=torch.float32),
        frame_indices=heldout_frame_indices.detach().cpu().to(dtype=torch.long),
        frame_count=frame_count,
        config=op_config,
        timing_iters=timing_iters,
    )
    if train_ppm_out is not None:
        write_ppm(train_ppm_out, mps_train["rgb"][0])
    if heldout_ppm_out is not None:
        write_ppm(heldout_ppm_out, mps_heldout["rgb"][0])

    train = split_summary(split="train", mps=mps_train, cpu=cpu_train, target=targets, boundary_count=len(boundaries))
    heldout = split_summary(
        split="heldout",
        mps=mps_heldout,
        cpu=cpu_heldout,
        target=heldout_targets.detach().cpu().to(dtype=torch.float32),
        boundary_count=len(boundaries),
    )
    tolerance = 5.0e-4
    acceptance = {
        "loaded_real_multicam_bundle": str(cfg["data"]["frame_source"]) == "multicam_val",
        "consumed_train_camera_rays": True,
        "consumed_heldout_camera_rays": True,
        "outputs_match_cpu_reference": max(
            train["max_rgb_abs_error"],
            train["max_alpha_abs_error"],
            train["max_depth_abs_error"],
            heldout["max_rgb_abs_error"],
            heldout["max_alpha_abs_error"],
            heldout["max_depth_abs_error"],
        )
        <= tolerance,
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
        "rgb_nonconstant": bool(mps_train["rgb"].std().item() > 0.0 and mps_heldout["rgb"].std().item() > 0.0),
        "uses_4d_power_boundaries": len(boundaries) == site_count * (site_count - 1) // 2,
    }
    return {
        "benchmark": "world_foam_lane2_gate1_mps_realray_replay_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "1C_realray_mps_forward",
        "device": "mps",
        "config_path": str(config_path),
        "sample_id": data["source_label"],
        "train_views": list(data["train_views"]),
        "heldout_views": list(data["heldout_views"]),
        "pose_source": data["pose_source"],
        "frame_count": frame_count,
        "render_size": int(cfg["render"]["render_size"]),
        "site_count": site_count,
        "boundary_count": len(boundaries),
        "near": float(near),
        "far": float(far),
        "density": float(density),
        "renderer_scope": "mps_real_camera_ray_4d_power_cell_per_sample_forward",
        "gradient_scope": "none_forward_only_no_backward",
        "sharing_scope": "none_linear_per_sample_baseline",
        "quality_claim": False,
        "world_foam_renderer_status": "mps_per_sample_real_camera_ray_forward_no_sharing_no_backward_no_training",
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
    parser = argparse.ArgumentParser(description="Smoke World Foam true real-camera-ray MPS forward replay.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--render-size", type=int)
    parser.add_argument("--site-count", type=int, default=12)
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
