from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import multicam_heldout_compare as mhc
from torch_gsplat_bridge_star_uvt import (
    stable_backward_samples,
    stable_backward_samples_with_keys,
    tile_pair_backward_samples,
    tile_pair_backward_samples_compensated,
    tile_pair_grouped_backward_samples,
    tile_pair_parallel_backward_samples,
    tile_pair_scanline_backward_samples,
    tile_pair_sharedsort_backward_samples,
    tile_pair_suffix_backward_samples,
    tile_pair_target_bounds_backward_samples,
)

try:
    from research_project.trainer_harness.tile_metal_autograd import _reduce_sample_bundle
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = Path(__file__).resolve().parents[1] / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from tile_metal_autograd import _reduce_sample_bundle


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_digest(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tensor_delta(left: Tensor, right: Tensor) -> dict[str, float | int]:
    left_cpu = left.detach().cpu()
    right_cpu = right.detach().cpu()
    if left_cpu.shape != right_cpu.shape:
        raise ValueError(f"shape mismatch: {left_cpu.shape} != {right_cpu.shape}")
    if left_cpu.numel() == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "different_count": 0}
    delta = (left_cpu.to(torch.float32) - right_cpu.to(torch.float32)).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "different_count": int((left_cpu != right_cpu).sum().item()),
    }


def bundle_digest(values: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        digest.update(tensor_digest(values[key]).encode("utf-8"))
    return digest.hexdigest()


def bundle_delta(left: dict[str, Tensor], right: dict[str, Tensor]) -> dict[str, Any]:
    keys = sorted(left)
    return {
        "max_abs": max(float(tensor_delta(left[key], right[key])["max_abs"]) for key in keys),
        "mean_abs_by_tensor": {key: tensor_delta(left[key], right[key])["mean_abs"] for key in keys},
        "max_abs_by_tensor": {key: tensor_delta(left[key], right[key])["max_abs"] for key in keys},
        "different_count_by_tensor": {key: tensor_delta(left[key], right[key])["different_count"] for key in keys},
    }


def make_initial_model(args: argparse.Namespace, bundle: Any, render_config: mhc.UVTRenderConfig) -> tuple[Any, dict[str, Any]]:
    model, train_report, _checkpoints = mhc.train_world_tubes(
        bundle=bundle,
        tube_count=args.uvt_tubes,
        train_seconds=999.0 if args.pretrain_steps > 0 else 0.0,
        max_steps=args.pretrain_steps,
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
        checkpoint_every_steps=0,
        render_config=render_config,
        reduction_mode=args.pretrain_reduction_mode,
        sample_emission_mode=args.pretrain_sample_emission_mode,
    )
    return model, train_report


def fixed_window_for_step(args: argparse.Namespace, bundle: Any) -> tuple[int, int]:
    if args.uvt_loss_scope != "temporal_window":
        raise ValueError("uvt_gradient_repeatability_probe currently expects temporal_window loss")
    view_count, frames = int(bundle.train_frames.shape[0]), int(bundle.train_frames.shape[1])
    active_views = mhc.optimizer_train_view_indices(view_count, args.uvt_optimizer_train_views)
    validation_frames = mhc.validation_frame_indices(
        frames,
        args.uvt_validation_frame_stride,
        args.uvt_validation_frame_offset,
    )
    active_frames = mhc.optimizer_frame_indices(frames, validation_frames)
    window_starts = mhc.optimizer_window_starts(frames, args.uvt_window_frames, active_frames)
    if args.uvt_train_schedule == "view_shuffled_cycle":
        return mhc.view_shuffled_cycle_pair(active_views, window_starts, step=args.step_index, seed=args.seed + 3001)
    if args.uvt_train_schedule == "shuffled_cycle":
        return mhc.shuffled_cycle_pairs(active_views, window_starts, seed=args.seed + 3001)[
            int(args.step_index % (len(active_views) * len(window_starts)))
        ]
    if args.uvt_train_schedule == "cycle":
        view = mhc.select_train_view(args.step_index, active_views, bundle.train_frames.device, "cycle")
        start = mhc.select_train_window_start(args.step_index, len(active_views), window_starts, bundle.train_frames.device, "cycle")
        return view, start
    raise ValueError(f"Unsupported train schedule for this probe: {args.uvt_train_schedule}")


def project_fixed_window(
    model: Any,
    args: argparse.Namespace,
    bundle: Any,
    window_config: mhc.UVTRenderConfig,
    *,
    view: int,
    frame_start: int,
) -> Any:
    lens_model, distortion = mhc.select_lens(
        bundle.train_lens_models,
        bundle.train_distortions,
        view,
        camera_projection=args.uvt_camera_projection,
    )
    return mhc.project_world_tube_sequence(
        model,
        mhc.select_view_K(bundle.train_K, view),
        mhc.select_view_w2c(bundle.train_w2c, view),
        window_config,
        camera_projection=args.uvt_camera_projection,
        lens_model=lens_model,
        distortion=distortion,
        full_frames=int(bundle.frame_count),
        frame_start=frame_start,
    )


def compute_loss(
    model: Any,
    args: argparse.Namespace,
    bundle: Any,
    window_config: mhc.UVTRenderConfig,
    *,
    view: int,
    frame_start: int,
) -> tuple[Tensor, Any, Tensor]:
    projected = project_fixed_window(model, args, bundle, window_config, view=view, frame_start=frame_start)
    rendered = mhc.render_projected_sequence(
        projected,
        window_config,
        backend=args.uvt_render_backend,
        reduction_mode=args.autograd_reduction_mode,
        sample_emission_mode=args.sample_emission_mode,
    )
    target = bundle.train_frames[view, frame_start : frame_start + args.uvt_window_frames].permute(0, 2, 3, 1).contiguous()
    recon_loss = mhc.robust_l1(rendered.rgb - target)
    model_reg = model.regularization()
    projected_reg, _metrics = mhc.projected_regularization(
        projected,
        window_config,
        tile_load_weight=args.uvt_tile_load_reg,
        tile_load_target=args.uvt_tile_load_target,
        depth_slope_weight=args.uvt_depth_slope_reg,
        depth_margin_weight=args.uvt_depth_margin_reg,
        depth_margin=args.uvt_depth_margin,
    )
    return recon_loss + model_reg + projected_reg, projected, rendered.rgb


def run_sample_phase(
    projected: Any,
    config: mhc.UVTRenderConfig,
    repeats: int,
    *,
    sample_emission_mode: str,
) -> list[dict[str, Tensor]]:
    grad_output = torch.ones((config.frames, config.height, config.width, 3), dtype=torch.float32, device=projected.ma.device)
    rows = []
    for _ in range(repeats):
        keys = None
        if sample_emission_mode == "with_keys":
            ids, grad_ma, grad_q, grad_opacity, grad_color, keys, tile_unstable = stable_backward_samples_with_keys(
                projected.ma.detach(),
                projected.q_uvt.detach(),
                projected.depth0.detach(),
                projected.depth_beta.detach(),
                projected.opacity.detach(),
                projected.color.detach(),
                grad_output.contiguous(),
                config,
            )
        elif sample_emission_mode == "atomic_append":
            ids, grad_ma, grad_q, grad_opacity, grad_color, tile_unstable = stable_backward_samples(
                projected.ma.detach(),
                projected.q_uvt.detach(),
                projected.depth0.detach(),
                projected.depth_beta.detach(),
                projected.opacity.detach(),
                projected.color.detach(),
                grad_output.contiguous(),
                config,
            )
        elif sample_emission_mode in (
            "tile_pair",
            "tile_pair_compensated",
            "tile_pair_grouped",
            "tile_pair_parallel",
            "tile_pair_scanline",
            "tile_pair_sharedsort",
            "tile_pair_target_bounds",
            "tile_pair_suffix",
        ):
            tile_pair_fn = {
                "tile_pair": tile_pair_backward_samples,
                "tile_pair_compensated": tile_pair_backward_samples_compensated,
                "tile_pair_grouped": tile_pair_grouped_backward_samples,
                "tile_pair_parallel": tile_pair_parallel_backward_samples,
                "tile_pair_scanline": tile_pair_scanline_backward_samples,
                "tile_pair_sharedsort": tile_pair_sharedsort_backward_samples,
                "tile_pair_target_bounds": tile_pair_target_bounds_backward_samples,
                "tile_pair_suffix": tile_pair_suffix_backward_samples,
            }[sample_emission_mode]
            ids, grad_ma, grad_q, grad_opacity, grad_color, keys, tile_unstable = tile_pair_fn(
                projected.ma.detach(),
                projected.q_uvt.detach(),
                projected.depth0.detach(),
                projected.depth_beta.detach(),
                projected.opacity.detach(),
                projected.color.detach(),
                grad_output.contiguous(),
                config,
            )
        else:
            raise ValueError(
                "sample emission mode must be one of: atomic_append, with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, tile_pair_suffix"
            )
        synchronize(projected.ma.device)
        row = {
            "ids": ids,
            "grad_ma_samples": grad_ma,
            "grad_q_samples": grad_q,
            "grad_opacity_samples": grad_opacity,
            "grad_color_samples": grad_color,
            "tile_unstable": tile_unstable,
        }
        if keys is not None:
            row["keys"] = keys
        rows.append(row)
    return rows


def run_reduce_phase(
    sample_row: dict[str, Tensor],
    tube_count: int,
    repeats: int,
    *,
    reduction_mode: str,
) -> list[dict[str, Tensor]]:
    rows = []
    for _ in range(repeats):
        grad_ma, grad_q, grad_opacity, grad_color = _reduce_sample_bundle(
            sample_row["ids"],
            sample_row["grad_ma_samples"],
            sample_row["grad_q_samples"],
            sample_row["grad_opacity_samples"],
            sample_row["grad_color_samples"],
            tube_count,
            mode=reduction_mode,
            keys=sample_row.get("keys"),
        )
        synchronize(grad_ma.device)
        rows.append(
            {
                "grad_ma": grad_ma,
                "grad_q": grad_q,
                "grad_opacity": grad_opacity,
                "grad_color": grad_color,
            }
        )
    return rows


def reduce_generated_sample_rows(
    sample_rows: list[dict[str, Tensor]],
    tube_count: int,
    *,
    reduction_mode: str,
) -> list[dict[str, Tensor]]:
    rows = []
    for sample_row in sample_rows:
        rows.extend(run_reduce_phase(sample_row, tube_count, 1, reduction_mode=reduction_mode))
    return rows


def grad_bundle(model: Any) -> dict[str, Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def run_autograd_phase(
    model: Any,
    initial_state: dict[str, Tensor],
    args: argparse.Namespace,
    bundle: Any,
    window_config: mhc.UVTRenderConfig,
    *,
    view: int,
    frame_start: int,
    repeats: int,
) -> list[dict[str, Any]]:
    rows = []
    for _ in range(repeats):
        model.load_state_dict(initial_state)
        model.zero_grad(set_to_none=True)
        loss, _projected, _rgb = compute_loss(model, args, bundle, window_config, view=view, frame_start=frame_start)
        loss.backward()
        synchronize(bundle.train_frames.device)
        grads = grad_bundle(model)
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                "grad_digest": bundle_digest(grads),
                "grads": grads,
            }
        )
    return rows


def summarize_tensor_rows(rows: list[dict[str, Tensor]]) -> dict[str, Any]:
    first = rows[0]
    return {
        "digests": [bundle_digest(row) for row in rows],
        "delta_vs_first": [bundle_delta(first, row) for row in rows],
    }


def summarize_autograd_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]["grads"]
    losses = [float(row["loss"]) for row in rows]
    return {
        "losses": losses,
        "loss_span": max(losses) - min(losses),
        "grad_digests": [row["grad_digest"] for row in rows],
        "grad_delta_vs_first": [bundle_delta(first, row["grads"]) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=mhc.DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument(
        "--pretrain-reduction-mode",
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
        "--pretrain-sample-emission-mode",
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
        ),
        default="atomic_append",
    )
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
        "--sample-emission-mode",
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
        ),
        default="atomic_append",
    )
    parser.add_argument(
        "--autograd-reduction-mode",
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
        "--diagnostic-reduction-mode",
        choices=(
            "none",
            "sorted_cpu",
            "scan_metal",
            "compensated_scan_metal",
            "sort_scan_metal",
            "sort_compensated_scan_metal",
            "key_sort_scan_metal",
            "key_sort_compensated_scan_metal",
            "key_sort_segmented_metal",
        ),
        default="sorted_cpu",
    )
    parser.add_argument(
        "--compare-sample-emission-mode",
        choices=(
            "none",
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
        ),
        default="none",
    )
    parser.add_argument(
        "--compare-autograd-reduction-mode",
        choices=(
            "none",
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
        default="none",
    )
    parser.add_argument("--uvt-camera-projection", choices=("legacy_pinhole", "dataset_lens"), default="legacy_pinhole")
    parser.add_argument("--uvt-loss-scope", choices=("temporal_window",), default="temporal_window")
    parser.add_argument("--uvt-window-frames", type=int, default=4)
    parser.add_argument("--uvt-sequence-consistency-every-steps", type=int, default=0)
    parser.add_argument("--uvt-sequence-consistency-frames", type=int, default=0)
    parser.add_argument("--uvt-sequence-consistency-weight", type=float, default=0.0)
    parser.add_argument("--uvt-multiscale-loss-weight", type=float, default=0.0)
    parser.add_argument("--uvt-multiscale-loss-factor", type=int, default=4)
    parser.add_argument("--uvt-crop-loss-weight", type=float, default=0.0)
    parser.add_argument("--uvt-crop-loss-size", type=int, default=128)
    parser.add_argument("--uvt-train-schedule", choices=("cycle", "shuffled_cycle", "view_shuffled_cycle"), default="cycle")
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
        default=mhc.ROOT / "research_project" / "benchmarks" / "results" / "uvt_gradient_repeatability_probe.json",
    )
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    keyed_modes = (
        "with_keys",
        "tile_pair",
        "tile_pair_compensated",
        "tile_pair_grouped",
        "tile_pair_parallel",
        "tile_pair_scanline",
        "tile_pair_sharedsort",
        "tile_pair_target_bounds",
        "tile_pair_suffix",
    )
    keyed_reduction_modes = ("key_sort_scan_metal", "key_sort_compensated_scan_metal", "key_sort_segmented_metal")
    if args.sample_emission_mode not in keyed_modes and (
        args.autograd_reduction_mode in keyed_reduction_modes or args.diagnostic_reduction_mode in keyed_reduction_modes
    ):
        raise ValueError(
            "keyed sort reduction requires --sample-emission-mode with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, or tile_pair_suffix"
        )
    if args.compare_sample_emission_mode == "none" and args.compare_autograd_reduction_mode != "none":
        raise ValueError("--compare-autograd-reduction-mode requires --compare-sample-emission-mode")
    if args.compare_sample_emission_mode != "none" and args.compare_autograd_reduction_mode == "none":
        raise ValueError("--compare-sample-emission-mode requires --compare-autograd-reduction-mode")
    if (
        args.compare_autograd_reduction_mode in keyed_reduction_modes
        and args.compare_sample_emission_mode not in keyed_modes
    ):
        raise ValueError(
            "keyed sort reduction requires --compare-sample-emission-mode with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, or tile_pair_suffix"
        )
    if args.pretrain_reduction_mode in keyed_reduction_modes and args.pretrain_sample_emission_mode not in keyed_modes:
        raise ValueError(
            "keyed sort reduction requires --pretrain-sample-emission-mode with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, or tile_pair_suffix"
        )
    if args.pretrain_sample_emission_mode == "direct_atomic" and args.pretrain_reduction_mode != "index_add":
        raise ValueError("direct_atomic bypasses the reducer and requires --pretrain-reduction-mode index_add")

    device = mhc.resolve_device(args.device)
    if args.uvt_render_backend != "metal_tile" or device.type != "mps":
        raise ValueError("uvt_gradient_repeatability_probe currently requires --device mps --uvt-render-backend metal_tile")
    config = mhc.load_config_file(mhc.resolve_dynaworld_path(args.baseline_config))
    data_cfg = mhc.config_data_for_run(config, target_size=args.target_size, max_frames=args.max_frames)
    bundle = mhc.load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=dict(config["camera"]),
        target_size=args.target_size,
        device=device,
    )
    full_config = mhc.UVTRenderConfig(
        height=int(bundle.train_frames.shape[-2]),
        width=int(bundle.train_frames.shape[-1]),
        frames=int(bundle.frame_count),
        tile_x=args.uvt_tile_x,
        tile_y=args.uvt_tile_y,
        tile_t=args.uvt_tile_t,
        tile_capacity=args.uvt_tile_capacity,
    )
    mhc.apply_uvt_tile_env(full_config)
    window_config = mhc.UVTRenderConfig(
        height=full_config.height,
        width=full_config.width,
        frames=args.uvt_window_frames,
        tile_x=full_config.tile_x,
        tile_y=full_config.tile_y,
        tile_t=full_config.tile_t,
        tile_capacity=full_config.tile_capacity,
        alpha_threshold=full_config.alpha_threshold,
        transmittance_threshold=full_config.transmittance_threshold,
        background=full_config.background,
        max_alpha=full_config.max_alpha,
    )
    model, pretrain_report = make_initial_model(args, bundle, full_config)
    initial_state = mhc.snapshot_world_tube_state(model)
    view, frame_start = fixed_window_for_step(args, bundle)
    model.load_state_dict(initial_state)
    loss, projected, _rgb = compute_loss(model, args, bundle, window_config, view=view, frame_start=frame_start)
    sample_rows = run_sample_phase(
        projected,
        window_config,
        args.repeats,
        sample_emission_mode=args.sample_emission_mode,
    )
    reduce_rows = run_reduce_phase(
        sample_rows[0],
        int(projected.ma.shape[0]),
        args.repeats,
        reduction_mode="index_add",
    )
    diagnostic_fixed_reduce_rows = None
    diagnostic_generated_reduce_rows = None
    if args.diagnostic_reduction_mode != "none":
        diagnostic_fixed_reduce_rows = run_reduce_phase(
            sample_rows[0],
            int(projected.ma.shape[0]),
            args.repeats,
            reduction_mode=args.diagnostic_reduction_mode,
        )
        diagnostic_generated_reduce_rows = reduce_generated_sample_rows(
            sample_rows,
            int(projected.ma.shape[0]),
            reduction_mode=args.diagnostic_reduction_mode,
        )
    autograd_rows = run_autograd_phase(
        model,
        initial_state,
        args,
        bundle,
        window_config,
        view=view,
        frame_start=frame_start,
        repeats=args.repeats,
    )
    comparison_autograd_rows = None
    if args.compare_sample_emission_mode != "none":
        compare_args = copy.copy(args)
        compare_args.sample_emission_mode = args.compare_sample_emission_mode
        compare_args.autograd_reduction_mode = args.compare_autograd_reduction_mode
        comparison_autograd_rows = run_autograd_phase(
            model,
            initial_state,
            compare_args,
            bundle,
            window_config,
            view=view,
            frame_start=frame_start,
            repeats=args.repeats,
        )
    report = {
        "meta": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "target_size": args.target_size,
            "max_frames": args.max_frames,
            "device": str(device),
            "seed": args.seed,
            "step_index": args.step_index,
            "repeats": args.repeats,
            "pretrain_steps": args.pretrain_steps,
            "pretrain_reduction_mode": args.pretrain_reduction_mode,
            "pretrain_sample_emission_mode": args.pretrain_sample_emission_mode,
            "autograd_reduction_mode": args.autograd_reduction_mode,
            "diagnostic_reduction_mode": args.diagnostic_reduction_mode,
            "sample_emission_mode": args.sample_emission_mode,
            "compare_autograd_reduction_mode": args.compare_autograd_reduction_mode,
            "compare_sample_emission_mode": args.compare_sample_emission_mode,
            "view": view,
            "frame_start": frame_start,
            "initial_loss": float(loss.detach().cpu()),
            "torch": {
                "version": torch.__version__,
                "mps_available": torch.backends.mps.is_available(),
                "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
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
        },
        "sample_generation": summarize_tensor_rows(sample_rows),
        "sample_generation_shapes": {key: list(value.shape) for key, value in sample_rows[0].items()},
        "reduction": summarize_tensor_rows(reduce_rows),
        "autograd": summarize_autograd_rows(autograd_rows),
        "pretrain": {
            "steps": pretrain_report.get("steps"),
            "train_loop_elapsed_s": pretrain_report.get("train_loop_elapsed_s"),
            "stopped_reason": pretrain_report.get("stopped_reason"),
            "last_log": pretrain_report.get("logs", [None])[-1] if pretrain_report.get("logs") else None,
        },
    }
    if diagnostic_fixed_reduce_rows is not None and diagnostic_generated_reduce_rows is not None:
        report["diagnostic_fixed_reduction"] = summarize_tensor_rows(diagnostic_fixed_reduce_rows)
        report["diagnostic_generated_sample_reduction"] = summarize_tensor_rows(diagnostic_generated_reduce_rows)
    if comparison_autograd_rows is not None:
        primary_first = autograd_rows[0]
        comparison_first = comparison_autograd_rows[0]
        report["comparison_autograd"] = summarize_autograd_rows(comparison_autograd_rows)
        report["comparison_autograd_delta_vs_primary_first"] = {
            "loss_delta": float(comparison_first["loss"] - primary_first["loss"]),
            "grad_delta": bundle_delta(primary_first["grads"], comparison_first["grads"]),
        }
    out_json = mhc.resolve_variant_path(args.out_json)
    mhc.write_json(out_json, report)
    print(f"Wrote UVT gradient repeatability probe to {out_json}")
    print(
        {
            "sample_digest_unique": len(set(report["sample_generation"]["digests"])),
            "reduction_digest_unique": len(set(report["reduction"]["digests"])),
            "diagnostic_fixed_reduction_digest_unique": (
                len(set(report["diagnostic_fixed_reduction"]["digests"]))
                if "diagnostic_fixed_reduction" in report
                else None
            ),
            "diagnostic_generated_reduction_digest_unique": (
                len(set(report["diagnostic_generated_sample_reduction"]["digests"]))
                if "diagnostic_generated_sample_reduction" in report
                else None
            ),
            "autograd_grad_digest_unique": len(set(report["autograd"]["grad_digests"])),
            "autograd_loss_span": report["autograd"]["loss_span"],
            "comparison_autograd_grad_digest_unique": (
                len(set(report["comparison_autograd"]["grad_digests"])) if "comparison_autograd" in report else None
            ),
            "comparison_max_abs_vs_primary": (
                report["comparison_autograd_delta_vs_primary_first"]["grad_delta"]["max_abs"]
                if "comparison_autograd_delta_vs_primary_first" in report
                else None
            ),
        }
    )


if __name__ == "__main__":
    main()
