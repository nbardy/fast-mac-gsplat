from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

import multicam_heldout_compare as mhc
from multicam_star_repeatability_probe import state_delta, state_digest


def train_mode(
    *,
    args: argparse.Namespace,
    bundle: Any,
    render_config: mhc.UVTRenderConfig,
    reduction_mode: str,
    sample_emission_mode: str,
    label: str,
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
        checkpoint_every_steps=args.checkpoint_every_steps,
        render_config=render_config,
        reduction_mode=reduction_mode,
        sample_emission_mode=sample_emission_mode,
    )
    final_state = mhc.snapshot_world_tube_state(model)
    final_eval = mhc.eval_world_tubes(
        model,
        bundle,
        backend=args.uvt_render_backend,
        camera_projection=args.uvt_camera_projection,
        render_config=render_config,
        frame_metric_splits=None,
    )
    public_checkpoints = [
        {
            "step": int(row["step"]),
            "elapsed_s": float(row["elapsed_s"]),
            "state_digest": state_digest(row["state"]),
        }
        for row in checkpoints
    ]
    return {
        "label": label,
        "reduction_mode": reduction_mode,
        "sample_emission_mode": sample_emission_mode,
        "train": train_report,
        "final": {
            "metrics": final_eval["metrics"],
            "state_digest": state_digest(final_state),
        },
        "checkpoints": public_checkpoints,
        "_final_state": final_state,
        "_checkpoints": checkpoints,
    }


def checkpoint_map(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["step"]): item for item in row["_checkpoints"]}


def matched_checkpoint_deltas(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_by_step = checkpoint_map(left)
    right_by_step = checkpoint_map(right)
    rows = []
    for step in sorted(set(left_by_step) & set(right_by_step)):
        left_row = left_by_step[step]
        right_row = right_by_step[step]
        rows.append(
            {
                "step": step,
                "elapsed_s": {
                    left["label"]: float(left_row["elapsed_s"]),
                    right["label"]: float(right_row["elapsed_s"]),
                },
                "state_delta": state_delta(left_row["state"], right_row["state"]),
                "state_digest": {
                    left["label"]: state_digest(left_row["state"]),
                    right["label"]: state_digest(right_row["state"]),
                },
            }
        )
    return rows


def public_mode(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=mhc.DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--train-seconds", type=float, default=999.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every-steps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
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
    parser.add_argument("--uvt-render-backend", choices=("dense", "metal_tile"), default="metal_tile")
    parser.add_argument(
        "--reduction-mode",
        choices=("key_sort_scan_metal", "key_sort_compensated_scan_metal", "key_sort_segmented_metal"),
        default="key_sort_scan_metal",
    )
    parser.add_argument(
        "--compare-sample-emission-mode",
        choices=(
            "tile_pair",
            "tile_pair_compensated",
            "tile_pair_grouped",
            "tile_pair_parallel",
            "tile_pair_scanline",
            "tile_pair_sharedsort",
            "tile_pair_target_bounds",
            "tile_pair_suffix",
        ),
        default="tile_pair",
    )
    parser.add_argument("--uvt-camera-projection", choices=("legacy_pinhole", "dataset_lens"), default="legacy_pinhole")
    parser.add_argument("--uvt-loss-scope", choices=("sampled_frame", "view_sequence", "temporal_window"), default="temporal_window")
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
    parser.add_argument("--uvt-validation-frame-stride", type=int, default=0)
    parser.add_argument("--uvt-validation-frame-offset", type=int, default=1)
    parser.add_argument("--init-depth", type=float, default=2.0)
    parser.add_argument("--uvt-init-views", choices=("first", "all_train"), default="first")
    parser.add_argument("--uvt-init-sampling", choices=("random", "grid"), default="random")
    parser.add_argument("--uvt-init-frames", choices=("first", "all", "fit"), default="first")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=mhc.ROOT / "research_project" / "benchmarks" / "results" / "multicam_star_mode_compare_probe.json",
    )
    args = parser.parse_args()

    if args.checkpoint_every_steps <= 0:
        raise ValueError("--checkpoint-every-steps must be positive")
    if args.torch_deterministic != "off":
        torch.use_deterministic_algorithms(True, warn_only=args.torch_deterministic == "warn")
    device = mhc.resolve_device(args.device)
    if args.uvt_render_backend == "metal_tile" and device.type != "mps":
        raise ValueError("--uvt-render-backend=metal_tile requires device=mps")

    config = mhc.load_config_file(mhc.resolve_dynaworld_path(args.baseline_config))
    data_cfg = mhc.config_data_for_run(config, target_size=args.target_size, max_frames=args.max_frames)
    bundle = mhc.load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=dict(config["camera"]),
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

    keyed = train_mode(
        args=args,
        bundle=bundle,
        render_config=render_config,
        reduction_mode=args.reduction_mode,
        sample_emission_mode="with_keys",
        label="keyed_per_pixel",
    )
    compare_label = args.compare_sample_emission_mode
    compare = train_mode(
        args=args,
        bundle=bundle,
        render_config=render_config,
        reduction_mode=args.reduction_mode,
        sample_emission_mode=args.compare_sample_emission_mode,
        label=compare_label,
    )
    checkpoint_deltas = matched_checkpoint_deltas(keyed, compare)
    report = {
        "meta": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "baseline_config": str(mhc.resolve_dynaworld_path(args.baseline_config)),
            "target_size": args.target_size,
            "max_frames": args.max_frames,
            "train_seconds": args.train_seconds,
            "max_steps": args.max_steps,
            "checkpoint_every_steps": args.checkpoint_every_steps,
            "device": str(device),
            "seed": args.seed,
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
                )
            ),
            "train_cameras": bundle.train_camera_names,
            "heldout_cameras": bundle.heldout_camera_names,
            "pose_source": bundle.pose_source,
            "uvt_camera_projection": args.uvt_camera_projection,
            "reduction_mode": args.reduction_mode,
            "compare_sample_emission_mode": args.compare_sample_emission_mode,
            "train_lens_models": bundle.train_lens_models,
            "heldout_lens_models": bundle.heldout_lens_models,
        },
        "modes": {
            "keyed_per_pixel": public_mode(keyed),
            compare_label: public_mode(compare),
        },
        "deltas": {
            "final_state": state_delta(keyed["_final_state"], compare["_final_state"]),
            f"final_train_psnr_delta_{compare_label}_minus_keyed": float(
                compare["final"]["metrics"]["eval_psnr"] - keyed["final"]["metrics"]["eval_psnr"]
            ),
            f"final_heldout_psnr_delta_{compare_label}_minus_keyed": float(
                compare["final"]["metrics"]["heldout_eval_psnr"] - keyed["final"]["metrics"]["heldout_eval_psnr"]
            ),
            "checkpoints": checkpoint_deltas,
        },
    }
    out_json = mhc.resolve_variant_path(args.out_json)
    mhc.write_json(out_json, report)
    print(f"Wrote STAR mode compare probe to {out_json}")
    print(
        {
            "final_state_max_abs": report["deltas"]["final_state"]["max_abs"],
            f"final_train_psnr_delta_{compare_label}_minus_keyed": report["deltas"][
                f"final_train_psnr_delta_{compare_label}_minus_keyed"
            ],
            f"final_heldout_psnr_delta_{compare_label}_minus_keyed": report["deltas"][
                f"final_heldout_psnr_delta_{compare_label}_minus_keyed"
            ],
        }
    )


if __name__ == "__main__":
    main()
