from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from multicam_heldout_compare import (
    DEFAULT_BASELINE_CONFIG,
    FreeDynamic3DGS,
    SplatRenderConfig,
    UVTRenderConfig,
    WorldTubeModel,
    apply_uvt_tile_env,
    camera_from_K_w2c,
    config_data_for_run,
    initialize_material_points_from_first_frame,
    initialize_world_tubes_from_train_views,
    load_config_file,
    load_multicam_video_bundle,
    prefix_metrics,
    project_world_tube_sequence,
    projected_regularization,
    render_gaussian_frame,
    render_projected_sequence,
    resolve_device,
    resolve_dynaworld_path,
    resolve_variant_path,
    robust_l1,
    select_K_for_view_time,
    select_view_K,
    select_view_w2c,
    select_w2c_for_view_time,
    serialize_config_value,
    synchronize_device,
    world_tube_metal_stats,
    write_json,
)
from research_project.trainer_harness.tile_metal_autograd import _reduce_sample_bundle
from torch_gsplat_bridge_star_uvt import stable_backward_samples


def timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    synchronize_device(device)
    started = time.perf_counter()
    value = fn()
    synchronize_device(device)
    return value, time.perf_counter() - started


def summarize(rows: list[dict[str, float]], *, skip_keys: set[str] | None = None) -> dict[str, dict[str, float | int]]:
    skipped = set() if skip_keys is None else skip_keys
    keys = sorted({key for row in rows for key in row if key not in skipped})
    out: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        out[key] = {
            "count": len(values),
            "mean_s": statistics.fmean(values),
            "min_s": min(values),
            "max_s": max(values),
            "total_s": sum(values),
        }
    mean_total = float(out.get("total", {}).get("mean_s", 0.0))
    if mean_total > 0.0:
        for value in out.values():
            value["mean_pct_of_total"] = float(value["mean_s"]) / mean_total
    return out


def summarize_scalars(rows: list[dict[str, float]], key: str) -> dict[str, float | int]:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def build_world_tube_model(bundle, args: argparse.Namespace, device: torch.device) -> WorldTubeModel:
    init_x0, init_color = initialize_world_tubes_from_train_views(
        bundle,
        tube_count=args.uvt_tubes,
        init_depth=args.init_depth,
        seed=args.seed,
        init_views=args.uvt_init_views,
    )
    return WorldTubeModel(
        init_x0=init_x0,
        init_color=init_color,
        frames=bundle.frame_count,
        init_precision_xy=args.uvt_init_precision_xy,
        init_lambda_t=args.uvt_init_lambda_t,
        init_opacity=args.uvt_init_opacity,
        min_precision_xy=args.uvt_min_precision_xy,
        min_lambda_t=args.uvt_min_lambda_t,
        velocity_reg_weight=args.uvt_velocity_reg,
        depth_velocity_reg_weight=args.uvt_depth_velocity_reg,
        position_reg_weight=args.uvt_position_reg,
    ).to(device)


def build_splat_model(bundle, args: argparse.Namespace, device: torch.device) -> tuple[FreeDynamic3DGS, SplatRenderConfig]:
    init_xyz, init_rgb = initialize_material_points_from_first_frame(
        video=bundle.train_frames[0].permute(0, 2, 3, 1).contiguous(),
        K=bundle.train_K[0],
        num_elements=args.splat_count,
        init_depth=args.init_depth,
    )
    model = FreeDynamic3DGS(
        init_xyz=init_xyz,
        init_rgb=init_rgb,
        num_frames=bundle.frame_count,
        splat_mode="per_frame",
        init_scale=args.splat_init_scale,
        scale_init_log_jitter=0.0,
        init_alpha_logit=0.0,
        init_xyz_noise=0.001,
        init_quat_noise=0.0,
        log_scale_min=-12.0,
        log_scale_max=4.0,
    ).to(device)
    _, _, _, height, width = bundle.train_frames.shape
    return model, SplatRenderConfig(
        height=height,
        width=width,
        renderer=args.splat_renderer,
        tile_size=16 if args.splat_renderer == "fast_mac" else 8,
        bound_scale=3.0,
        alpha_threshold=1.0 / 255.0,
        near_plane=1.0e-3,
        camera_projection="legacy_pinhole",
    )


def star_step(
    *,
    model: WorldTubeModel,
    optimizer: torch.optim.Optimizer,
    bundle,
    args: argparse.Namespace,
    render_config: UVTRenderConfig,
    window_config: UVTRenderConfig,
    step: int,
    device: torch.device,
) -> dict[str, float]:
    view_count, frames, _, _, _ = bundle.train_frames.shape
    view = step % view_count
    frame_start = 0
    config = render_config
    full_frames = None
    if args.uvt_loss_scope == "temporal_window":
        max_start = frames - args.uvt_window_frames
        frame_start = (step * args.frame_stride_for_probe) % (max_start + 1)
        config = window_config
        full_frames = frames

    phases: dict[str, float] = {}
    step_started = time.perf_counter()
    _, phases["zero_grad"] = timed(device, lambda: optimizer.zero_grad(set_to_none=True))
    projected, phases["project"] = timed(
        device,
        lambda: project_world_tube_sequence(
            model,
            select_view_K(bundle.train_K, view),
            select_view_w2c(bundle.train_w2c, view),
            config,
            full_frames=full_frames,
            frame_start=frame_start,
        ),
    )
    rendered, phases["render"] = timed(device, lambda: render_projected_sequence(projected, config, backend=args.uvt_render_backend))

    def compute_loss() -> torch.Tensor:
        if args.uvt_loss_scope == "sampled_frame":
            frame = step % frames
            target = bundle.train_frames[view, frame].permute(1, 2, 0)
            recon_loss = robust_l1(rendered.rgb[frame] - target)
        elif args.uvt_loss_scope == "view_sequence":
            target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
            recon_loss = robust_l1(rendered.rgb - target)
        else:
            target = bundle.train_frames[view, frame_start : frame_start + args.uvt_window_frames].permute(0, 2, 3, 1).contiguous()
            recon_loss = robust_l1(rendered.rgb - target)
        projected_reg, _ = projected_regularization(
            projected,
            config,
            tile_load_weight=args.uvt_tile_load_reg,
            tile_load_target=args.uvt_tile_load_target,
            depth_slope_weight=args.uvt_depth_slope_reg,
            depth_margin_weight=args.uvt_depth_margin_reg,
            depth_margin=args.uvt_depth_margin,
        )
        return recon_loss + model.regularization() + projected_reg

    loss, phases["loss"] = timed(device, compute_loss)
    _, phases["backward"] = timed(device, loss.backward)
    _, phases["optimizer"] = timed(
        device,
        lambda: (torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0), optimizer.step()),
    )
    synchronize_device(device)
    phases["total"] = time.perf_counter() - step_started
    phases["loss_value"] = float(loss.detach().cpu())
    return phases


def splat_step(
    *,
    model: FreeDynamic3DGS,
    render_cfg: SplatRenderConfig,
    optimizer: torch.optim.Optimizer,
    bundle,
    step: int,
    device: torch.device,
) -> dict[str, float]:
    view_count, frames, _, height, width = bundle.train_frames.shape
    view = step % view_count
    frame = (step * 7) % frames
    phases: dict[str, float] = {}
    step_started = time.perf_counter()
    _, phases["zero_grad"] = timed(device, lambda: optimizer.zero_grad(set_to_none=True))
    camera = camera_from_K_w2c(
        select_K_for_view_time(bundle.train_K, view=view, t=frame, view_count=view_count),
        select_w2c_for_view_time(bundle.train_w2c, view=view, t=frame),
    )
    splats, phases["frame_model"] = timed(device, lambda: model.frame(frame))
    image, phases["render"] = timed(
        device,
        lambda: render_gaussian_frame(
            splats,
            camera,
            height=height,
            width=width,
            mode=render_cfg.renderer,
            tile_size=render_cfg.tile_size,
            bound_scale=render_cfg.bound_scale,
            alpha_threshold=render_cfg.alpha_threshold,
            near_plane=render_cfg.near_plane,
            camera_projection=render_cfg.camera_projection,
        ).permute(1, 2, 0),
    )

    def compute_loss() -> torch.Tensor:
        target = bundle.train_frames[view, frame].permute(1, 2, 0)
        loss = robust_l1(image - target)
        return loss + 1.0e-4 * model.scale_loss() + 1.0e-3 * model.temporal_smoothness_loss()

    loss, phases["loss"] = timed(device, compute_loss)
    _, phases["backward"] = timed(device, loss.backward)
    _, phases["optimizer"] = timed(
        device,
        lambda: (torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0), optimizer.step()),
    )
    synchronize_device(device)
    phases["total"] = time.perf_counter() - step_started
    phases["loss_value"] = float(loss.detach().cpu())
    return phases


def star_backward_microbreakdown(
    *,
    model: WorldTubeModel,
    bundle,
    args: argparse.Namespace,
    render_config: UVTRenderConfig,
    window_config: UVTRenderConfig,
    device: torch.device,
) -> dict[str, Any]:
    if args.uvt_render_backend != "metal_tile":
        return {"skipped": "STAR backward microbreakdown currently targets the metal_tile backend."}

    frames = int(bundle.train_frames.shape[1])
    config = window_config if args.uvt_loss_scope == "temporal_window" else render_config
    full_frames = frames if args.uvt_loss_scope == "temporal_window" else None
    frame_start = 0
    projected, project_s = timed(
        device,
        lambda: project_world_tube_sequence(
            model,
            select_view_K(bundle.train_K, 0),
            select_view_w2c(bundle.train_w2c, 0),
            config,
            full_frames=full_frames,
            frame_start=frame_start,
        ),
    )
    grad_image = torch.ones((config.frames, config.height, config.width, 3), dtype=torch.float32, device=device)

    def sample_backward() -> tuple[torch.Tensor, ...]:
        return stable_backward_samples(
            projected.ma.detach(),
            projected.q_uvt.detach(),
            projected.depth0.detach(),
            projected.depth_beta.detach(),
            projected.opacity.detach(),
            projected.color.detach(),
            grad_image,
            config,
        )

    samples, sample_s = timed(device, sample_backward)
    ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, tile_unstable = samples
    tube_count = int(projected.ma.shape[0])
    (grad_ma, grad_q, grad_opacity, grad_color), reduce_bundle_s = timed(
        device,
        lambda: _reduce_sample_bundle(
            ids,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        ),
    )
    zero_depth0 = torch.zeros_like(projected.depth0)
    zero_depth_beta = torch.zeros_like(projected.depth_beta)
    model.zero_grad(set_to_none=True)
    _, projection_vjp_s = timed(
        device,
        lambda: torch.autograd.backward(
            (
                projected.ma,
                projected.q_uvt,
                projected.depth0,
                projected.depth_beta,
                projected.opacity,
                projected.color,
            ),
            (
                grad_ma,
                grad_q,
                zero_depth0,
                zero_depth_beta,
                grad_opacity,
                grad_color,
            ),
        ),
    )
    tile_count = ((config.width + config.tile_x - 1) // config.tile_x) * (
        (config.height + config.tile_y - 1) // config.tile_y
    ) * ((config.frames + config.tile_t - 1) // config.tile_t)
    allocated_sample_slots = tile_count * config.tile_x * config.tile_y * config.tile_t * config.tile_capacity
    sample_count = int(ids.numel())
    return {
        "project_s": project_s,
        "sample_backward_s": sample_s,
        "reduce_total_s": reduce_bundle_s,
        "reduce_bundle_s": reduce_bundle_s,
        "projection_vjp_s": projection_vjp_s,
        "sample_plus_reduce_s": sample_s + reduce_bundle_s,
        "allocated_sample_slot_count": int(allocated_sample_slots),
        "sample_count": sample_count,
        "compact_sample_fraction": float(sample_count) / float(max(allocated_sample_slots, 1)),
        "valid_sample_count": int(((ids >= 0) & (ids < tube_count)).sum().detach().cpu()),
        "unstable_tile_fraction": float(tile_unstable.float().mean().detach().cpu()),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    if args.uvt_render_backend == "metal_tile" and device.type != "mps":
        raise ValueError("--uvt-render-backend=metal_tile requires device=mps")
    torch.manual_seed(args.seed)
    config = load_config_file(resolve_dynaworld_path(args.baseline_config))
    data_cfg = config_data_for_run(config, target_size=args.target_size, max_frames=args.max_frames)
    bundle = load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=dict(config["camera"]),
        target_size=args.target_size,
        device=device,
    )
    _, frames, _, height, width = bundle.train_frames.shape
    if args.uvt_loss_scope == "temporal_window" and args.uvt_window_frames > frames:
        raise ValueError(f"--uvt-window-frames={args.uvt_window_frames} exceeds loaded frame count {frames}")

    render_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        tile_x=args.uvt_tile_x,
        tile_y=args.uvt_tile_y,
        tile_t=args.uvt_tile_t,
        tile_capacity=args.uvt_tile_capacity,
    )
    apply_uvt_tile_env(render_config)
    window_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=args.uvt_window_frames if args.uvt_loss_scope == "temporal_window" else frames,
        tile_x=render_config.tile_x,
        tile_y=render_config.tile_y,
        tile_t=render_config.tile_t,
        tile_capacity=render_config.tile_capacity,
        alpha_threshold=render_config.alpha_threshold,
        transmittance_threshold=render_config.transmittance_threshold,
        background=render_config.background,
        max_alpha=render_config.max_alpha,
    )

    star_model = build_world_tube_model(bundle, args, device)
    star_optimizer = torch.optim.Adam(star_model.parameters(), lr=args.uvt_lr)
    splat_model, splat_render_cfg = build_splat_model(bundle, args, device)
    splat_optimizer = torch.optim.Adam(splat_model.parameters(), lr=args.splat_lr)

    star_rows = []
    splat_rows = []
    for step in range(args.warmup_steps + args.steps):
        star_row = star_step(
            model=star_model,
            optimizer=star_optimizer,
            bundle=bundle,
            args=args,
            render_config=render_config,
            window_config=window_config,
            step=step,
            device=device,
        )
        splat_row = splat_step(
            model=splat_model,
            render_cfg=splat_render_cfg,
            optimizer=splat_optimizer,
            bundle=bundle,
            step=step,
            device=device,
        )
        if step >= args.warmup_steps:
            star_rows.append(star_row)
            splat_rows.append(splat_row)

    star_stats = world_tube_metal_stats(star_model, bundle, render_config=render_config) if args.uvt_render_backend == "metal_tile" else None
    star_backward_breakdown = star_backward_microbreakdown(
        model=star_model,
        bundle=bundle,
        args=args,
        render_config=render_config,
        window_config=window_config,
        device=device,
    )
    return {
        "meta": {
            "baseline_config": str(resolve_dynaworld_path(args.baseline_config)),
            "target_size": args.target_size,
            "max_frames": args.max_frames,
            "device": str(device),
            "train_cameras": bundle.train_camera_names,
            "heldout_cameras": bundle.heldout_camera_names,
            "pose_source": bundle.pose_source,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "config_data": serialize_config_value(data_cfg),
            "note": "Per-step timing only; not quality evidence.",
        },
        "star_uvt": {
            "tube_count": args.uvt_tubes,
            "render_backend": args.uvt_render_backend,
            "loss_scope": args.uvt_loss_scope,
            "window_frames": args.uvt_window_frames if args.uvt_loss_scope == "temporal_window" else None,
            "lr": args.uvt_lr,
            "tile_x": render_config.tile_x,
            "tile_y": render_config.tile_y,
            "tile_t": render_config.tile_t,
            "tile_capacity": render_config.tile_capacity,
            "tile_load_reg": args.uvt_tile_load_reg,
            "tile_load_target": args.uvt_tile_load_target,
            "depth_slope_reg": args.uvt_depth_slope_reg,
            "rows": star_rows,
            "summary": summarize(star_rows, skip_keys={"loss_value"}),
            "loss_value_summary": summarize_scalars(star_rows, "loss_value"),
            "metal_stats_after_probe": star_stats,
            "backward_microbreakdown": star_backward_breakdown,
        },
        "free_dynamic_splats": {
            "splat_count": args.splat_count,
            "renderer": args.splat_renderer,
            "lr": args.splat_lr,
            "rows": splat_rows,
            "summary": summarize(splat_rows, skip_keys={"loss_value"}),
            "loss_value_summary": summarize_scalars(splat_rows, "loss_value"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--frame-stride-for-probe", type=int, default=3)
    parser.add_argument("--uvt-tubes", type=int, default=256)
    parser.add_argument("--uvt-lr", type=float, default=0.03)
    parser.add_argument("--uvt-render-backend", choices=("dense", "metal_tile"), default="metal_tile")
    parser.add_argument("--uvt-loss-scope", choices=("sampled_frame", "view_sequence", "temporal_window"), default="temporal_window")
    parser.add_argument("--uvt-window-frames", type=int, default=4)
    parser.add_argument("--uvt-init-precision-xy", type=float, default=30.0)
    parser.add_argument("--uvt-init-lambda-t", type=float, default=0.35)
    parser.add_argument("--uvt-init-opacity", type=float, default=0.35)
    parser.add_argument("--uvt-min-precision-xy", type=float, default=1.0e-5)
    parser.add_argument("--uvt-min-lambda-t", type=float, default=1.0e-5)
    parser.add_argument("--uvt-velocity-reg", type=float, default=1.0e-4)
    parser.add_argument("--uvt-depth-velocity-reg", type=float, default=0.0)
    parser.add_argument("--uvt-position-reg", type=float, default=1.0e-6)
    parser.add_argument("--uvt-tile-load-reg", type=float, default=0.001)
    parser.add_argument("--uvt-tile-load-target", type=float, default=7000.0)
    parser.add_argument("--uvt-depth-slope-reg", type=float, default=0.05)
    parser.add_argument("--uvt-depth-margin-reg", type=float, default=0.0)
    parser.add_argument("--uvt-depth-margin", type=float, default=0.05)
    parser.add_argument("--uvt-tile-x", type=int, default=8)
    parser.add_argument("--uvt-tile-y", type=int, default=8)
    parser.add_argument("--uvt-tile-t", type=int, default=1)
    parser.add_argument("--uvt-tile-capacity", type=int, default=256)
    parser.add_argument("--uvt-init-views", choices=("first", "all_train"), default="first")
    parser.add_argument("--splat-count", type=int, default=2048)
    parser.add_argument("--splat-lr", type=float, default=0.002)
    parser.add_argument("--splat-renderer", choices=("dense", "fast_mac"), default="fast_mac")
    parser.add_argument("--splat-init-scale", type=float, default=0.035)
    parser.add_argument("--init-depth", type=float, default=2.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research_project/benchmarks/results/multicam_train_step_timing_probe_mps_256_16f_current_default.json"),
    )
    args = parser.parse_args()
    report = run_probe(args)
    out_path = resolve_variant_path(args.out_json)
    write_json(out_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote multicam train-step timing probe to {out_path}")


if __name__ == "__main__":
    main()
