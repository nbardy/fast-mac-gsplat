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
    DYNAWORLD_ROOT,
    FreeDynamic3DGS,
    SplatRenderConfig,
    UVTRenderConfig,
    WorldTubeModel,
    camera_from_K_w2c,
    config_data_for_run,
    dense_differentiable_render_uvt_tubes,
    initialize_material_points_from_first_frame,
    initialize_world_tubes_from_train_views,
    load_config_file,
    load_multicam_video_bundle,
    make_pinhole_camera,
    project_world_tubes_pinhole,
    render_splat_sequence,
    render_uvt_tubes,
    render_uvt_tubes_metal_tile_backward,
    render_world_tube_sequence,
    resolve_device,
    resolve_dynaworld_path,
    resolve_variant_path,
    select_K_for_view_time,
    select_view_K,
    select_view_w2c,
    select_w2c_for_view_time,
    serialize_config_value,
    synchronize_device,
    write_json,
)


def timed_iterations(
    device: torch.device,
    fn: Callable[[], Any],
    *,
    warmup_iterations: int,
    iterations: int,
) -> dict[str, float | int]:
    with torch.no_grad():
        for _ in range(warmup_iterations):
            fn()
        synchronize_device(device)
        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            fn()
            synchronize_device(device)
            samples.append(time.perf_counter() - started)
    return {
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "mean_s": statistics.fmean(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "total_s": sum(samples),
    }


def summarize_rows(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    values = [float(row[key]["mean_s"]) for row in rows]
    if not values:
        return {"sequence_count": 0, "mean_sequence_s": 0.0, "max_sequence_s": 0.0, "total_sequence_mean_s": 0.0}
    return {
        "sequence_count": len(values),
        "mean_sequence_s": statistics.fmean(values),
        "max_sequence_s": max(values),
        "total_sequence_mean_s": sum(values),
    }


def make_world_tube_model(bundle, args: argparse.Namespace, device: torch.device) -> WorldTubeModel:
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


def make_splat_model(bundle, args: argparse.Namespace, device: torch.device) -> tuple[FreeDynamic3DGS, SplatRenderConfig]:
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
        init_scale=0.035,
        scale_init_log_jitter=0.0,
        init_alpha_logit=0.0,
        init_xyz_noise=0.001,
        init_quat_noise=0.0,
        log_scale_min=-12.0,
        log_scale_max=4.0,
    ).to(device)
    _, frames, _, height, width = bundle.train_frames.shape
    del frames
    render_cfg = SplatRenderConfig(
        height=height,
        width=width,
        renderer=args.splat_renderer,
        tile_size=16 if args.splat_renderer == "fast_mac" else 8,
        bound_scale=3.0,
        alpha_threshold=1.0 / 255.0,
        near_plane=1.0e-3,
        camera_projection="legacy_pinhole",
    )
    return model, render_cfg


def star_render_only(
    *,
    backend: str,
    ma: torch.Tensor,
    q_uvt: torch.Tensor,
    depth0: torch.Tensor,
    depth_beta: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    config: UVTRenderConfig,
) -> torch.Tensor:
    if backend == "dense":
        return dense_differentiable_render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config)
    if backend == "metal_tile":
        return render_uvt_tubes_metal_tile_backward(ma, q_uvt, depth0, depth_beta, opacity, color, config)
    raise ValueError("backend must be one of: dense, metal_tile")


def star_sequence_row(
    *,
    model: WorldTubeModel,
    K: torch.Tensor,
    w2c: torch.Tensor,
    config: UVTRenderConfig,
    backend: str,
    device: torch.device,
    warmup_iterations: int,
    iterations: int,
    camera_name: str,
    split: str,
) -> dict[str, Any]:
    camera = make_pinhole_camera(K, w2c)
    ma, q_uvt, depth0, depth_beta, opacity, color = project_world_tubes_pinhole(model.batch(), camera, config)
    metal_stats = None
    if backend == "metal_tile":
        result = render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config, return_aux=True)
        if result.stats is None:
            raise AssertionError("Metal render did not return stats")
        metal_stats = result.stats.__dict__
    return {
        "split": split,
        "camera": camera_name,
        "metal_stats": metal_stats,
        "full_project_render": timed_iterations(
            device,
            lambda: render_world_tube_sequence(model, K, w2c, config, backend=backend),
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        ),
        "project_only": timed_iterations(
            device,
            lambda: project_world_tubes_pinhole(model.batch(), camera, config),
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        ),
        "render_only": timed_iterations(
            device,
            lambda: star_render_only(
                backend=backend,
                ma=ma,
                q_uvt=q_uvt,
                depth0=depth0,
                depth_beta=depth_beta,
                opacity=opacity,
                color=color,
                config=config,
            ),
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        ),
    }


def splat_sequence_row(
    *,
    model: FreeDynamic3DGS,
    render_cfg: SplatRenderConfig,
    K: torch.Tensor,
    w2c: torch.Tensor,
    view_count: int,
    view: int,
    frames: int,
    device: torch.device,
    warmup_iterations: int,
    iterations: int,
    camera_name: str,
    split: str,
) -> dict[str, Any]:
    cameras = [
        camera_from_K_w2c(
            select_K_for_view_time(K, view=view, t=frame, view_count=view_count),
            select_w2c_for_view_time(w2c, view=view, t=frame),
        )
        for frame in range(frames)
    ]
    return {
        "split": split,
        "camera": camera_name,
        "full_sequence_render": timed_iterations(
            device,
            lambda: render_splat_sequence(model, cameras, render_cfg),
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        ),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    if args.uvt_render_backend == "metal_tile" and device.type != "mps":
        raise ValueError("--uvt-render-backend=metal_tile requires device=mps")
    config = load_config_file(resolve_dynaworld_path(args.baseline_config))
    data_cfg = config_data_for_run(config, target_size=args.target_size, max_frames=args.max_frames)
    bundle = load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=dict(config["camera"]),
        target_size=args.target_size,
        device=device,
    )
    render_config = UVTRenderConfig(height=args.target_size, width=args.target_size, frames=int(bundle.frame_count))
    star_model = make_world_tube_model(bundle, args, device)
    splat_model, splat_render_cfg = make_splat_model(bundle, args, device)

    star_rows = [
        star_sequence_row(
            model=star_model,
            K=select_view_K(bundle.train_K, view),
            w2c=select_view_w2c(bundle.train_w2c, view),
            config=render_config,
            backend=args.uvt_render_backend,
            device=device,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            camera_name=bundle.train_camera_names[view],
            split="train",
        )
        for view in range(bundle.train_view_count)
    ]
    splat_rows = [
        splat_sequence_row(
            model=splat_model,
            render_cfg=splat_render_cfg,
            K=bundle.train_K,
            w2c=bundle.train_w2c,
            view_count=bundle.train_view_count,
            view=view,
            frames=bundle.frame_count,
            device=device,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            camera_name=bundle.train_camera_names[view],
            split="train",
        )
        for view in range(bundle.train_view_count)
    ]
    if not args.train_only and bundle.heldout_frames is not None:
        star_rows.extend(
            star_sequence_row(
                model=star_model,
                K=select_view_K(bundle.heldout_K, view),
                w2c=select_view_w2c(bundle.heldout_w2c, view),
                config=render_config,
                backend=args.uvt_render_backend,
                device=device,
                warmup_iterations=args.warmup_iterations,
                iterations=args.iterations,
                camera_name=bundle.heldout_camera_names[view],
                split="heldout",
            )
            for view in range(bundle.heldout_view_count)
        )
        splat_rows.extend(
            splat_sequence_row(
                model=splat_model,
                render_cfg=splat_render_cfg,
                K=bundle.heldout_K,
                w2c=bundle.heldout_w2c,
                view_count=bundle.heldout_view_count,
                view=view,
                frames=bundle.frame_count,
                device=device,
                warmup_iterations=args.warmup_iterations,
                iterations=args.iterations,
                camera_name=bundle.heldout_camera_names[view],
                split="heldout",
            )
            for view in range(bundle.heldout_view_count)
        )

    return {
        "meta": {
            "baseline_config": str(resolve_dynaworld_path(args.baseline_config)),
            "target_size": args.target_size,
            "max_frames": args.max_frames,
            "device": str(device),
            "train_cameras": bundle.train_camera_names,
            "heldout_cameras": [] if args.train_only else bundle.heldout_camera_names,
            "pose_source": bundle.pose_source,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "config_data": serialize_config_value(data_cfg),
            "note": "Initialized-model render timing probe; not trained-model quality evidence.",
        },
        "star_uvt": {
            "tube_count": args.uvt_tubes,
            "render_backend": args.uvt_render_backend,
            "init_precision_xy": args.uvt_init_precision_xy,
            "init_lambda_t": args.uvt_init_lambda_t,
            "init_opacity": args.uvt_init_opacity,
            "min_precision_xy": args.uvt_min_precision_xy,
            "min_lambda_t": args.uvt_min_lambda_t,
            "velocity_reg": args.uvt_velocity_reg,
            "depth_velocity_reg": args.uvt_depth_velocity_reg,
            "position_reg": args.uvt_position_reg,
            "init_views": args.uvt_init_views,
            "rows": star_rows,
            "full_project_render_summary": summarize_rows(star_rows, "full_project_render"),
            "project_only_summary": summarize_rows(star_rows, "project_only"),
            "render_only_summary": summarize_rows(star_rows, "render_only"),
        },
        "free_dynamic_splats": {
            "splat_count": args.splat_count,
            "renderer": args.splat_renderer,
            "rows": splat_rows,
            "full_sequence_render_summary": summarize_rows(splat_rows, "full_sequence_render"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--uvt-tubes", type=int, default=256)
    parser.add_argument("--uvt-render-backend", choices=("dense", "metal_tile"), default="dense")
    parser.add_argument("--uvt-init-precision-xy", type=float, default=30.0)
    parser.add_argument("--uvt-init-lambda-t", type=float, default=0.35)
    parser.add_argument("--uvt-init-opacity", type=float, default=0.35)
    parser.add_argument("--uvt-min-precision-xy", type=float, default=1.0e-5)
    parser.add_argument("--uvt-min-lambda-t", type=float, default=1.0e-5)
    parser.add_argument("--uvt-velocity-reg", type=float, default=1.0e-4)
    parser.add_argument("--uvt-depth-velocity-reg", type=float, default=0.0)
    parser.add_argument("--uvt-position-reg", type=float, default=1.0e-6)
    parser.add_argument("--uvt-init-views", choices=("first", "all_train"), default="first")
    parser.add_argument("--splat-count", type=int, default=2048)
    parser.add_argument("--splat-renderer", choices=("dense", "fast_mac"), default="dense")
    parser.add_argument("--init-depth", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research_project/benchmarks/results/multicam_render_timing_probe.json"),
    )
    args = parser.parse_args()
    report = run_probe(args)
    out_path = resolve_variant_path(args.out_json)
    write_json(out_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote multicam render timing probe to {out_path}")


if __name__ == "__main__":
    main()
