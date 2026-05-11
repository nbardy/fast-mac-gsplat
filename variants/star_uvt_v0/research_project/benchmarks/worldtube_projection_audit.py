from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from multicam_heldout_compare import (
    DEFAULT_BASELINE_CONFIG,
    config_data_for_run,
    initialize_world_tubes_from_view,
    load_config_file,
    load_multicam_video_bundle,
    make_pinhole_camera,
    project_world_tubes_pinhole,
    resolve_device,
    resolve_dynaworld_path,
    resolve_variant_path,
    select_view_K,
    select_view_w2c,
    serialize_config_value,
    write_json,
)
from research_project.trainer_harness.world_tube import WorldTubeBatch
from torch_gsplat_bridge_star_uvt import UVTRenderConfig


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    cpu = values.detach().float().flatten().cpu()
    if cpu.numel() == 0:
        return {}
    qs = torch.quantile(cpu, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))
    return {
        "min": float(qs[0]),
        "p10": float(qs[1]),
        "median": float(qs[2]),
        "p90": float(qs[3]),
        "max": float(qs[4]),
    }


def _support_tau(opacity: float, alpha_threshold: float) -> float:
    return -2.0 * math.log(max(float(alpha_threshold) / max(float(opacity), 1.0e-8), 1.0e-8))


def summarize_projection(
    *,
    batch: WorldTubeBatch,
    K: torch.Tensor,
    w2c: torch.Tensor,
    config: UVTRenderConfig,
    opacity: float,
) -> dict[str, object]:
    ma, q_uvt, depth0, _depth_beta, _opacity, _color = project_world_tubes_pinhole(batch, make_pinhole_camera(K, w2c), config)
    lambda_u = q_uvt[:, 0].clamp_min(1.0e-8)
    lambda_v = q_uvt[:, 3].clamp_min(1.0e-8)
    tau = _support_tau(opacity, config.alpha_threshold)
    radius_u = torch.sqrt(torch.tensor(tau, dtype=torch.float32, device=lambda_u.device) / lambda_u)
    radius_v = torch.sqrt(torch.tensor(tau, dtype=torch.float32, device=lambda_v.device) / lambda_v)
    inside = (ma[:, 0] >= 0.0) & (ma[:, 0] < float(config.width)) & (ma[:, 1] >= 0.0) & (ma[:, 1] < float(config.height))
    return {
        "center_u": _quantiles(ma[:, 0]),
        "center_v": _quantiles(ma[:, 1]),
        "depth": _quantiles(depth0),
        "inside_frame_fraction": float(inside.float().mean().detach().cpu()),
        "lambda_u": _quantiles(lambda_u),
        "lambda_v": _quantiles(lambda_v),
        "sigma_u_px": _quantiles(torch.rsqrt(lambda_u)),
        "sigma_v_px": _quantiles(torch.rsqrt(lambda_v)),
        "support_radius_u_px": _quantiles(radius_u),
        "support_radius_v_px": _quantiles(radius_v),
    }


def run_audit(
    *,
    baseline_config: Path,
    target_size: int,
    max_frames: int,
    tube_count: int,
    init_depth: float,
    init_precision_xy: float,
    init_lambda_t: float,
    init_opacity: float,
    seed: int,
    device: str,
) -> dict[str, object]:
    dev = resolve_device(device)
    config = load_config_file(resolve_dynaworld_path(baseline_config))
    data_cfg = config_data_for_run(config, target_size=target_size, max_frames=max_frames)
    camera_cfg = dict(config["camera"])
    bundle = load_multicam_video_bundle(data_cfg=data_cfg, camera_cfg=camera_cfg, target_size=target_size, device=dev)
    init_x0, init_color = initialize_world_tubes_from_view(
        bundle.train_frames[0],
        select_view_K(bundle.train_K, 0),
        select_view_w2c(bundle.train_w2c, 0),
        tube_count=tube_count,
        init_depth=init_depth,
        seed=seed,
    )
    batch = WorldTubeBatch(
        x0=init_x0,
        velocity=torch.zeros_like(init_x0),
        t0=torch.zeros((tube_count,), dtype=torch.float32, device=dev),
        precision_xy=torch.full((tube_count, 2), float(init_precision_xy), dtype=torch.float32, device=dev),
        lambda_t=torch.full((tube_count,), float(init_lambda_t), dtype=torch.float32, device=dev),
        opacity=torch.full((tube_count,), float(init_opacity), dtype=torch.float32, device=dev),
        color=init_color,
    )
    render_config = UVTRenderConfig(height=target_size, width=target_size, frames=int(bundle.frame_count))
    train_views = {
        name: summarize_projection(
            batch=batch,
            K=select_view_K(bundle.train_K, view),
            w2c=select_view_w2c(bundle.train_w2c, view),
            config=render_config,
            opacity=init_opacity,
        )
        for view, name in enumerate(bundle.train_camera_names)
    }
    heldout_views = {
        name: summarize_projection(
            batch=batch,
            K=select_view_K(bundle.heldout_K, view),
            w2c=select_view_w2c(bundle.heldout_w2c, view),
            config=render_config,
            opacity=init_opacity,
        )
        for view, name in enumerate(bundle.heldout_camera_names)
    }
    return {
        "baseline_config": str(resolve_dynaworld_path(baseline_config)),
        "target_size": target_size,
        "max_frames": max_frames,
        "tube_count": tube_count,
        "init_depth": init_depth,
        "init_precision_xy": init_precision_xy,
        "init_lambda_t": init_lambda_t,
        "init_opacity": init_opacity,
        "seed": seed,
        "device": str(dev),
        "pose_source": bundle.pose_source,
        "train_cameras": bundle.train_camera_names,
        "heldout_cameras": bundle.heldout_camera_names,
        "config_data": serialize_config_value(data_cfg),
        "train_projection": train_views,
        "heldout_projection": heldout_views,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--tube-count", type=int, default=256)
    parser.add_argument("--init-depth", type=float, default=2.0)
    parser.add_argument("--init-precision-xy", type=float, default=30.0)
    parser.add_argument("--init-lambda-t", type=float, default=0.35)
    parser.add_argument("--init-opacity", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    report = run_audit(
        baseline_config=args.baseline_config,
        target_size=args.target_size,
        max_frames=args.max_frames,
        tube_count=args.tube_count,
        init_depth=args.init_depth,
        init_precision_xy=args.init_precision_xy,
        init_lambda_t=args.init_lambda_t,
        init_opacity=args.init_opacity,
        seed=args.seed,
        device=args.device,
    )
    if args.out_json is not None:
        out_path = resolve_variant_path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
