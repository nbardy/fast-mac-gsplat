from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import multicam_heldout_compare as mhc


def state_digest(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_delta(left: dict[str, Tensor], right: dict[str, Tensor]) -> dict[str, float]:
    max_abs = 0.0
    mean_abs_num = 0.0
    mean_abs_den = 0
    for key in sorted(left):
        delta = (left[key].detach().cpu() - right[key].detach().cpu()).abs()
        if delta.numel() == 0:
            continue
        max_abs = max(max_abs, float(delta.max().item()))
        mean_abs_num += float(delta.sum().item())
        mean_abs_den += int(delta.numel())
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs_num / mean_abs_den if mean_abs_den else 0.0,
    }


def metric_span(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value is not None:
            values.append(float(value))
    return max(values) - min(values) if values else None


def selected_metal_summary(metal_stats: dict[str, Any] | None) -> dict[str, float | int] | None:
    if not metal_stats or "rows" not in metal_stats:
        return None
    rows = metal_stats["rows"]
    return {
        "max_pair_ratio": max(float(row["stats"]["pair_ratio"]) for row in rows),
        "max_tile_count": max(int(row["stats"]["max_tile_count"]) for row in rows),
        "overflow_tile_count": sum(int(row["stats"]["overflow_tile_count"]) for row in rows),
        "max_unstable_tile_fraction": max(float(row["stats"]["unstable_tile_fraction"]) for row in rows),
    }


def run_star_once(
    *,
    args: argparse.Namespace,
    bundle: Any,
    render_config: mhc.UVTRenderConfig,
    frame_metric_splits: dict[str, list[int]] | None,
) -> dict[str, Any]:
    model, train_report, checkpoints = mhc.train_world_tubes(
        bundle=bundle,
        tube_count=args.uvt_tubes,
        train_seconds=args.train_seconds,
        max_steps=args.max_steps,
        lr=args.uvt_lr,
        lr_decay_step=args.uvt_lr_decay_step,
        lr_decay_factor=args.uvt_lr_decay_factor,
        init_depth=args.init_depth,
        init_views=args.uvt_init_views,
        init_sampling=args.uvt_init_sampling,
        init_frames=args.uvt_init_frames,
        init_precision_xy=args.uvt_init_precision_xy,
        init_lambda_t=args.uvt_init_lambda_t,
        init_opacity=args.uvt_init_opacity,
        min_precision_xy=args.uvt_min_precision_xy,
        min_lambda_t=args.uvt_min_lambda_t,
        velocity_reg_weight=args.uvt_velocity_reg,
        depth_velocity_reg_weight=args.uvt_depth_velocity_reg,
        position_reg_weight=args.uvt_position_reg,
        tile_load_reg_weight=args.uvt_tile_load_reg,
        tile_load_target=args.uvt_tile_load_target,
        depth_slope_reg_weight=args.uvt_depth_slope_reg,
        depth_margin_reg_weight=args.uvt_depth_margin_reg,
        depth_margin=args.uvt_depth_margin,
        seed=args.seed,
        backend=args.uvt_render_backend,
        camera_projection=args.uvt_camera_projection,
        loss_scope=args.uvt_loss_scope,
        window_frames=args.uvt_window_frames,
        train_schedule=args.uvt_train_schedule,
        optimizer_train_views=args.uvt_optimizer_train_views,
        validation_frame_stride=args.uvt_validation_frame_stride,
        validation_frame_offset=args.uvt_validation_frame_offset,
        sequence_consistency_every_steps=args.uvt_sequence_consistency_every_steps,
        sequence_consistency_frames=args.uvt_sequence_consistency_frames,
        sequence_consistency_weight=args.uvt_sequence_consistency_weight,
        multiscale_loss_weight=args.uvt_multiscale_loss_weight,
        multiscale_loss_factor=args.uvt_multiscale_loss_factor,
        crop_loss_weight=args.uvt_crop_loss_weight,
        crop_loss_size=args.uvt_crop_loss_size,
        checkpoint_every_steps=args.uvt_checkpoint_every_steps,
        render_config=render_config,
        reduction_mode=args.uvt_reduction_mode,
        sample_emission_mode=args.uvt_sample_emission_mode,
    )
    final_state = mhc.snapshot_world_tube_state(model)
    final_eval = mhc.eval_world_tubes(
        model,
        bundle,
        backend=args.uvt_render_backend,
        camera_projection=args.uvt_camera_projection,
        render_config=render_config,
        frame_metric_splits=frame_metric_splits,
    )
    checkpoint_curve = mhc.eval_world_tube_checkpoints(
        model,
        checkpoints,
        bundle,
        backend=args.uvt_render_backend,
        camera_projection=args.uvt_camera_projection,
        render_config=render_config,
        frame_metric_splits=frame_metric_splits,
    )
    selected_report = None
    selected_state = None
    if args.uvt_select_checkpoint != "none":
        if checkpoint_curve is None:
            raise ValueError("--uvt-select-checkpoint requires --uvt-checkpoint-every-steps > 0")
        selected_row, selection_metric, uses_heldout_for_selection, selection_detail = mhc.select_world_tube_checkpoint_row(
            checkpoint_curve,
            selector=args.uvt_select_checkpoint,
            train_psnr_plateau_delta=args.uvt_select_train_psnr_plateau_delta,
            train_psnr_plateau_patience=args.uvt_select_train_psnr_plateau_patience,
            train_psnr_gain_drop=args.uvt_select_train_psnr_gain_drop,
            train_view_gap_collapse=args.uvt_select_train_view_gap_collapse,
            train_view_gap_max=args.uvt_select_train_view_gap_max,
            train_view_index=args.uvt_select_train_view_index,
        )
        model.load_state_dict(mhc.find_checkpoint_state(checkpoints, selected_row))
        selected_state = mhc.snapshot_world_tube_state(model)
        selected_eval = mhc.eval_world_tubes(
            model,
            bundle,
            backend=args.uvt_render_backend,
            camera_projection=args.uvt_camera_projection,
            render_config=render_config,
            frame_metric_splits=frame_metric_splits,
        )
        metal_stats = (
            mhc.world_tube_metal_stats(
                model,
                bundle,
                camera_projection=args.uvt_camera_projection,
                render_config=render_config,
            )
            if args.uvt_render_backend == "metal_tile"
            else None
        )
        selected_report = {
            "selector": args.uvt_select_checkpoint,
            "selection_metric": selection_metric,
            "uses_heldout_for_selection": uses_heldout_for_selection,
            "selection_detail": selection_detail,
            "selected_step": selected_row["step"],
            "selected_elapsed_s": selected_row["elapsed_s"],
            "metrics": selected_eval["metrics"],
            "metal_stats_summary": selected_metal_summary(metal_stats),
            "state_digest": state_digest(selected_state),
        }
        model.load_state_dict(final_state)
    return {
        "train": train_report,
        "final": {
            "metrics": final_eval["metrics"],
            "state_digest": state_digest(final_state),
        },
        "selected": selected_report,
        "checkpoint_curve": checkpoint_curve,
        "_final_state": final_state,
        "_selected_state": selected_state,
    }


def public_run(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=mhc.DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--train-seconds", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--torch-deterministic", choices=("off", "warn", "error"), default="off")
    parser.add_argument("--uvt-tubes", type=int, default=128)
    parser.add_argument("--uvt-lr", type=float, default=0.03)
    parser.add_argument("--uvt-lr-decay-step", type=int, default=0)
    parser.add_argument("--uvt-lr-decay-factor", type=float, default=1.0)
    parser.add_argument("--uvt-init-precision-xy", type=float, default=30.0)
    parser.add_argument("--uvt-init-lambda-t", type=float, default=0.35)
    parser.add_argument("--uvt-init-opacity", type=float, default=0.35)
    parser.add_argument("--uvt-min-precision-xy", type=float, default=1.0e-5)
    parser.add_argument("--uvt-min-lambda-t", type=float, default=1.0e-5)
    parser.add_argument("--uvt-velocity-reg", type=float, default=1.0e-4)
    parser.add_argument("--uvt-depth-velocity-reg", type=float, default=0.0)
    parser.add_argument("--uvt-position-reg", type=float, default=1.0e-6)
    parser.add_argument("--uvt-tile-load-reg", type=float, default=0.0)
    parser.add_argument("--uvt-tile-load-target", type=float, default=0.0)
    parser.add_argument("--uvt-depth-slope-reg", type=float, default=0.0)
    parser.add_argument("--uvt-depth-margin-reg", type=float, default=0.0)
    parser.add_argument("--uvt-depth-margin", type=float, default=0.05)
    parser.add_argument("--uvt-tile-x", type=int, default=mhc.env_int("STAR_UVT_TILE_X", 8))
    parser.add_argument("--uvt-tile-y", type=int, default=mhc.env_int("STAR_UVT_TILE_Y", 8))
    parser.add_argument("--uvt-tile-t", type=int, default=mhc.env_int("STAR_UVT_TILE_T", 2))
    parser.add_argument("--uvt-tile-capacity", type=int, default=mhc.env_int("STAR_UVT_TILE_CAPACITY", 128))
    parser.add_argument("--uvt-render-backend", choices=("dense", "metal_tile"), default="dense")
    parser.add_argument(
        "--uvt-reduction-mode",
        choices=(
            "index_add",
            "sorted_cpu",
            "scan_metal",
            "compensated_scan_metal",
            "sort_scan_metal",
            "sort_compensated_scan_metal",
            "key_sort_scan_metal",
            "key_sort_compensated_scan_metal",
            "key_sort_segmented_metal",
        ),
        default="index_add",
    )
    parser.add_argument(
        "--uvt-sample-emission-mode",
        choices=(
            "atomic_append",
            "with_keys",
            "tile_pair",
            "tile_pair_compensated",
            "tile_pair_grouped",
            "tile_pair_parallel",
            "tile_pair_scanline",
            "tile_pair_sharedsort",
            "tile_pair_target_bounds",
            "tile_pair_suffix",
            "direct_atomic",
            "direct_fixedpoint",
            "direct_split_fixedpoint",
            "direct_serial",
            "tile_pair_atomic",
            "tile_pair_fixedpoint",
            "tile_pair_reduced",
            "tile_pair_reduced_parallel",
            "tile_pair_suffix_reduced",
        ),
        default="atomic_append",
    )
    parser.add_argument("--uvt-camera-projection", choices=("legacy_pinhole", "dataset_lens"), default="legacy_pinhole")
    parser.add_argument("--uvt-loss-scope", choices=("sampled_frame", "view_sequence", "temporal_window"), default="sampled_frame")
    parser.add_argument("--uvt-window-frames", type=int, default=4)
    parser.add_argument("--uvt-sequence-consistency-every-steps", type=int, default=0)
    parser.add_argument("--uvt-sequence-consistency-frames", type=int, default=0)
    parser.add_argument("--uvt-sequence-consistency-weight", type=float, default=0.0)
    parser.add_argument("--uvt-multiscale-loss-weight", type=float, default=0.0)
    parser.add_argument("--uvt-multiscale-loss-factor", type=int, default=4)
    parser.add_argument("--uvt-crop-loss-weight", type=float, default=0.0)
    parser.add_argument("--uvt-crop-loss-size", type=int, default=128)
    parser.add_argument("--uvt-train-schedule", choices=mhc.TRAIN_SCHEDULE_CHOICES, default="random")
    parser.add_argument("--uvt-optimizer-train-views", choices=("all", "first_only"), default="all")
    parser.add_argument("--uvt-checkpoint-every-steps", type=int, default=0)
    parser.add_argument(
        "--uvt-select-checkpoint",
        choices=(
            "none",
            "best_heldout",
            "best_train_psnr",
            "best_min_train_view_psnr",
            "best_train_view_psnr",
            "best_train_dev_frame_psnr",
            "first_train_psnr_plateau",
            "first_train_psnr_gain_drop",
            "first_train_view_gap_collapse",
            "first_balanced_train_psnr_plateau",
        ),
        default="none",
    )
    parser.add_argument("--uvt-select-train-psnr-plateau-delta", type=float, default=0.5)
    parser.add_argument("--uvt-select-train-psnr-plateau-patience", type=int, default=1)
    parser.add_argument("--uvt-select-train-psnr-gain-drop", type=float, default=0.02)
    parser.add_argument("--uvt-select-train-view-gap-collapse", type=float, default=0.7)
    parser.add_argument("--uvt-select-train-view-gap-max", type=float, default=1.2)
    parser.add_argument("--uvt-select-train-view-index", type=int, default=1)
    parser.add_argument("--uvt-validation-frame-stride", type=int, default=0)
    parser.add_argument("--uvt-validation-frame-offset", type=int, default=1)
    parser.add_argument("--init-depth", type=float, default=2.0)
    parser.add_argument("--uvt-init-views", choices=("first", "all_train"), default="first")
    parser.add_argument("--uvt-init-sampling", choices=("random", "grid"), default="random")
    parser.add_argument("--uvt-init-frames", choices=("first", "all", "fit"), default="first")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=mhc.ROOT / "research_project" / "benchmarks" / "results" / "multicam_star_repeatability_probe.json",
    )
    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if args.torch_deterministic != "off":
        torch.use_deterministic_algorithms(True, warn_only=args.torch_deterministic == "warn")

    device = mhc.resolve_device(args.device)
    if args.uvt_render_backend == "metal_tile" and device.type != "mps":
        raise ValueError("--uvt-render-backend=metal_tile requires device=mps")
    if args.uvt_render_backend != "metal_tile" and (
        args.uvt_reduction_mode != "index_add" or args.uvt_sample_emission_mode != "atomic_append"
    ):
        raise ValueError("custom UVT reduction/sample emission modes require --uvt-render-backend metal_tile")
    if args.uvt_reduction_mode in (
        "key_sort_scan_metal",
        "key_sort_compensated_scan_metal",
        "key_sort_segmented_metal",
    ) and args.uvt_sample_emission_mode not in (
        "with_keys",
        "tile_pair",
        "tile_pair_compensated",
        "tile_pair_grouped",
        "tile_pair_parallel",
        "tile_pair_scanline",
        "tile_pair_sharedsort",
        "tile_pair_target_bounds",
        "tile_pair_suffix",
    ):
        raise ValueError(
            "keyed sort reduction requires --uvt-sample-emission-mode with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, or tile_pair_suffix"
        )
    if args.uvt_sample_emission_mode in (
        "direct_atomic",
        "direct_fixedpoint",
        "direct_split_fixedpoint",
        "direct_serial",
        "tile_pair_atomic",
        "tile_pair_fixedpoint",
        "tile_pair_reduced",
        "tile_pair_reduced_parallel",
        "tile_pair_suffix_reduced",
    ) and args.uvt_reduction_mode != "index_add":
        raise ValueError(f"{args.uvt_sample_emission_mode} bypasses the reducer and requires --uvt-reduction-mode index_add")
    config = mhc.load_config_file(mhc.resolve_dynaworld_path(args.baseline_config))
    data_cfg = mhc.config_data_for_run(config, target_size=args.target_size, max_frames=args.max_frames)
    camera_cfg = dict(config["camera"])
    bundle = mhc.load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=camera_cfg,
        target_size=args.target_size,
        device=device,
    )
    render_config = mhc.UVTRenderConfig(
        height=int(bundle.train_frames.shape[-2]),
        width=int(bundle.train_frames.shape[-1]),
        frames=int(bundle.frame_count),
        tile_x=args.uvt_tile_x,
        tile_y=args.uvt_tile_y,
        tile_t=args.uvt_tile_t,
        tile_capacity=args.uvt_tile_capacity,
    )
    mhc.apply_uvt_tile_env(render_config)
    validation_frames = mhc.validation_frame_indices(
        int(bundle.frame_count),
        args.uvt_validation_frame_stride,
        args.uvt_validation_frame_offset,
    )
    optimizer_frames = mhc.optimizer_frame_indices(int(bundle.frame_count), validation_frames)
    frame_metric_splits = {"fit": optimizer_frames, "dev": validation_frames} if validation_frames else None

    private_runs = [
        run_star_once(
            args=args,
            bundle=bundle,
            render_config=render_config,
            frame_metric_splits=frame_metric_splits,
        )
        for _ in range(args.repeats)
    ]
    report = {
        "meta": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "baseline_config": str(mhc.resolve_dynaworld_path(args.baseline_config)),
            "target_size": args.target_size,
            "max_frames": args.max_frames,
            "train_seconds": args.train_seconds,
            "max_steps": args.max_steps,
            "device": str(device),
            "seed": args.seed,
            "repeats": args.repeats,
            "torch": {
                "version": torch.__version__,
                "deterministic_mode": args.torch_deterministic,
                "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                "mps_available": torch.backends.mps.is_available(),
                "cuda_available": torch.cuda.is_available(),
            },
            "env": mhc.selected_env(
                (
                    "PYTHONHASHSEED",
                    "PYTORCH_ENABLE_MPS_FALLBACK",
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                    "PYTORCH_MPS_ALLOCATOR_POLICY",
                    "STAR_UVT_TILE_X",
                    "STAR_UVT_TILE_Y",
                    "STAR_UVT_TILE_T",
                    "STAR_UVT_TILE_CAPACITY",
                    "STAR_UVT_FIXEDPOINT_SCALE",
                    "STAR_UVT_SPLIT_FIXEDPOINT_COARSE_SCALE",
                    "STAR_UVT_SPLIT_FIXEDPOINT_FINE_SCALE",
                )
            ),
            "train_cameras": bundle.train_camera_names,
            "heldout_cameras": bundle.heldout_camera_names,
            "pose_source": bundle.pose_source,
            "uvt_camera_projection": args.uvt_camera_projection,
            "uvt_reduction_mode": args.uvt_reduction_mode,
            "uvt_sample_emission_mode": args.uvt_sample_emission_mode,
            "train_lens_models": bundle.train_lens_models,
            "heldout_lens_models": bundle.heldout_lens_models,
        },
        "runs": [public_run(row) for row in private_runs],
        "deltas": {
            "selected_heldout_psnr_span": metric_span(private_runs, ("selected", "metrics", "heldout_eval_psnr")),
            "selected_train_psnr_span": metric_span(private_runs, ("selected", "metrics", "eval_psnr")),
            "final_heldout_psnr_span": metric_span(private_runs, ("final", "metrics", "heldout_eval_psnr")),
            "final_train_psnr_span": metric_span(private_runs, ("final", "metrics", "eval_psnr")),
            "final_state_0_1": state_delta(private_runs[0]["_final_state"], private_runs[1]["_final_state"]),
            "selected_state_0_1": state_delta(private_runs[0]["_selected_state"], private_runs[1]["_selected_state"])
            if private_runs[0]["_selected_state"] is not None and private_runs[1]["_selected_state"] is not None
            else None,
        },
    }
    out_json = mhc.resolve_variant_path(args.out_json)
    mhc.write_json(out_json, report)
    print(f"Wrote STAR repeatability probe to {out_json}")
    print(
        {
            "selected_heldout_psnr_span": report["deltas"]["selected_heldout_psnr_span"],
            "final_state_max_abs": report["deltas"]["final_state_0_1"]["max_abs"],
        }
    )


if __name__ == "__main__":
    main()
