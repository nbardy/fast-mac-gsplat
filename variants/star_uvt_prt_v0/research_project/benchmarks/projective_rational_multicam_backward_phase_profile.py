from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any

import torch

from projective_rational_multicam_train_breakdown import (  # noqa: E402
    DEFAULT_BASELINE_CONFIG,
    MulticamPRTWorldTubeModel,
    _camera_path_to_device,
    _camera_sequence,
    _compile_detached_footprint,
    _fit_prt_with_breakdown,
    _render_projected_train,
    _resolve_dynaworld_path,
    _sync,
    centered_frame_times,
    fit_camera_path_polynomial,
    load_config_file,
    load_multicam_video_bundle,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    apply_projective_rational_tile_env,
    parse_projective_rational_tile_config,
    profile_projective_rational_tile_pixel_atomic_backward,
    recommend_projective_rational_train_speed_tile_policy,
    recommend_projective_rational_tile_config,
)


TIMING_KEYS = (
    "alloc_tiles_ms",
    "clear_tiles_ms",
    "bin_tubes_ms",
    "alloc_grads_ms",
    "clear_grads_ms",
    "backward_kernel_ms",
    "compute_only_kernel_ms",
    "total_ms",
)


def _summarize_ms(rows: list[dict[str, float]]) -> dict[str, Any]:
    summary = {}
    for key in TIMING_KEYS:
        samples = [float(row[key]) for row in rows]
        summary[key] = {
            "samples_ms": samples,
            "min_ms": min(samples),
            "median_ms": statistics.median(samples),
            "max_ms": max(samples),
        }
    total = float(summary["total_ms"]["median_ms"])
    summary["median_share_of_total"] = {
        key: 0.0 if total <= 0.0 else float(value["median_ms"]) / total
        for key, value in summary.items()
        if isinstance(value, dict) and key.endswith("_ms")
    }
    return summary


def _make_grad_image(image: torch.Tensor, target: torch.Tensor, frame: int, loss_mode: str) -> torch.Tensor:
    if loss_mode == "sampled_frame":
        grad = torch.zeros_like(image)
        grad[frame] = 2.0 * (image[frame] - target[frame]) / float(image[frame].numel())
        return grad.contiguous()
    if loss_mode == "sequence":
        return (2.0 * (image - target) / float(image.numel())).contiguous()
    raise ValueError("loss_mode must be one of: sampled_frame, sequence")


def _profile_backward(
    *,
    projected,
    config: UVTRenderConfig,
    grad_image: torch.Tensor,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        profile_projective_rational_tile_pixel_atomic_backward(
            projected.h_coeff,
            projected.lambda_uv,
            projected.lambda_t,
            projected.center_t,
            projected.opacity,
            projected.color,
            grad_image,
            config,
        )
    _sync(device)

    timing_rows = []
    last_result = None
    for _ in range(repeats):
        last_result = profile_projective_rational_tile_pixel_atomic_backward(
            projected.h_coeff,
            projected.lambda_uv,
            projected.lambda_t,
            projected.center_t,
            projected.opacity,
            projected.color,
            grad_image,
            config,
        )
        timing_rows.append(dict(last_result.timings_ms))
    if last_result is None:
        raise ValueError("repeats must be positive")

    tile_counts = last_result.tile_counts.detach().cpu()
    tile_overflow = last_result.tile_overflow.detach().cpu()
    tile_unstable = last_result.tile_unstable.detach().cpu()
    grad_tensors = (
        last_result.grad_h_coeff,
        last_result.grad_lambda_uv,
        last_result.grad_lambda_t,
        last_result.grad_center_t,
        last_result.grad_opacity,
        last_result.grad_color,
    )
    grad_max_abs = max(float(t.detach().abs().max().cpu()) for t in grad_tensors)
    grad_all_finite = all(bool(torch.isfinite(t.detach()).all().cpu()) for t in grad_tensors)
    return {
        "warmups": warmups,
        "repeats": repeats,
        "timing_rows_ms": timing_rows,
        "timing_summary_ms": _summarize_ms(timing_rows),
        "max_tile_count": int(tile_counts.max().item()),
        "mean_tile_count": float(tile_counts.float().mean().item()),
        "overflow_tile_count": int((tile_overflow > 0).sum().item()),
        "unstable_tile_count": int((tile_unstable > 0).sum().item()),
        "grad_max_abs": grad_max_abs,
        "grad_all_finite": grad_all_finite,
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "mps":
        raise ValueError("projective rational multicam backward phase profile currently requires --device=mps")
    torch.manual_seed(args.seed)

    config = load_config_file(_resolve_dynaworld_path(args.baseline_config))
    data_cfg = dict(config["data"])
    camera_cfg = dict(config["camera"])
    if data_cfg.get("multicam_manifest") is not None:
        data_cfg["multicam_manifest"] = str(_resolve_dynaworld_path(data_cfg["multicam_manifest"]))
    data_cfg["max_frames"] = int(args.max_frames)
    bundle = load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=camera_cfg,
        target_size=int(args.target_size),
        device=device,
    )
    _views, frames, _channels, height, width = bundle.train_frames.shape
    frame_times_cpu = centered_frame_times(frames, device="cpu")

    train_camera_paths = []
    train_fit_errors = []
    for view in range(bundle.train_view_count):
        k_seq, w2c_seq = _camera_sequence(
            bundle.train_K.detach().cpu(),
            bundle.train_w2c.detach().cpu(),
            view=view,
            frames=frames,
            view_count=bundle.train_view_count,
        )
        path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=args.camera_poly_degree, frame_times=frame_times_cpu)
        train_fit_errors.append(path.fit_error)
        train_camera_paths.append(_camera_path_to_device(path, device))

    prt_support_alpha_threshold = args.prt_support_alpha_threshold
    if args.prt_tile_policy == "train_speed":
        if args.tile_config != "auto":
            raise ValueError("--prt-tile-policy train_speed requires --tile-config auto")
        tile_policy = recommend_projective_rational_train_speed_tile_policy(tube_count=args.prt_tubes)
        tile_config = tile_policy.tile_config
        if prt_support_alpha_threshold is None:
            prt_support_alpha_threshold = tile_policy.support_alpha_threshold
        prt_tile_policy = tile_policy.name
    else:
        tile_config = (
            recommend_projective_rational_tile_config(tube_count=args.prt_tubes)
            if args.tile_config == "auto"
            else parse_projective_rational_tile_config(args.tile_config)
        )
        prt_tile_policy = "generic_auto" if args.tile_config == "auto" else "explicit_tile_config"
    apply_projective_rational_tile_env(tile_config)
    prt_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        alpha_threshold=args.prt_alpha_threshold,
        support_alpha_threshold=prt_support_alpha_threshold,
        background=(1.0, 1.0, 1.0),
        **tile_config.as_render_kwargs(),
    )
    model = MulticamPRTWorldTubeModel(
        bundle=bundle,
        tube_count=args.prt_tubes,
        init_depth=args.init_depth,
        init_precision_xy=args.prt_init_precision_xy,
        init_lambda_t=args.prt_init_lambda_t,
        init_opacity=args.prt_init_opacity,
        seed=args.seed,
        device=device,
    ).to(device)
    train = _fit_prt_with_breakdown(
        model=model,
        bundle=bundle,
        train_camera_paths=train_camera_paths,
        config=prt_config,
        steps=args.steps,
        lr=args.prt_lr,
        loss_mode=args.prt_loss_mode,
        device=device,
        seed=args.seed + 17,
    )

    view = int(args.profile_view) % bundle.train_view_count
    frame = int(args.profile_frame) % frames
    projected = _compile_detached_footprint(model, train_camera_paths[view])
    image = _render_projected_train(projected, prt_config)
    target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
    grad_image = _make_grad_image(image.detach(), target, frame, args.prt_loss_mode)
    _sync(device)

    backward_profile = _profile_backward(
        projected=projected,
        config=prt_config,
        grad_image=grad_image,
        device=device,
        warmups=args.profile_warmups,
        repeats=args.profile_repeats,
    )

    finite_losses = math.isfinite(float(train["initial_loss"])) and math.isfinite(float(train["final_loss"]))
    return {
        "name": "projective_rational_multicam_backward_phase_profile",
        "note": (
            "Diagnostic internal phase timing for the PRT tile-pixel atomic backward on the real D2 multicam bundle. "
            "Each profiled phase synchronizes, so totals are diagnostic rather than normal train wall time."
        ),
        "pass": finite_losses and backward_profile["overflow_tile_count"] == 0 and backward_profile["grad_all_finite"],
        "meta": {
            "baseline_config": str(_resolve_dynaworld_path(args.baseline_config)),
            "target_size": args.target_size,
            "max_frames": args.max_frames,
            "frames": frames,
            "height": height,
            "width": width,
            "device": str(device),
            "seed": args.seed,
            "steps": args.steps,
            "profile_view": view,
            "profile_frame": frame,
            "train_cameras": bundle.train_camera_names,
            "heldout_cameras": bundle.heldout_camera_names,
            "pose_source": bundle.pose_source,
            "sample_id": None if bundle.metadata is None else bundle.metadata.get("sample_id"),
            "camera_poly_degree": args.camera_poly_degree,
            "prt_alpha_threshold": args.prt_alpha_threshold,
            "prt_support_alpha_threshold": prt_support_alpha_threshold,
            "prt_tile_policy": prt_tile_policy,
            "train_camera_fit_errors": train_fit_errors,
        },
        "projective_rational": {
            "tube_count": args.prt_tubes,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "tile_config_key": tile_config.key,
            "tile_config": tile_config.as_dict(),
            "lr": args.prt_lr,
            "loss_mode": args.prt_loss_mode,
            "init_depth": args.init_depth,
            "init_precision_xy": args.prt_init_precision_xy,
            "init_lambda_t": args.prt_init_lambda_t,
            "init_opacity": args.prt_init_opacity,
            "train": train,
            "backward_profile": backward_profile,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--camera-poly-degree", type=int, default=1)
    parser.add_argument("--tile-config", default="auto", help="'auto' or an explicit config like 8x8x2:128")
    parser.add_argument(
        "--prt-tile-policy",
        choices=("generic", "train_speed"),
        default="generic",
        help="generic preserves the fail-closed auto selector; train_speed opts into measured support-pruned 1024 policy.",
    )
    parser.add_argument("--prt-tubes", type=int, default=128)
    parser.add_argument("--prt-lr", type=float, default=0.02)
    parser.add_argument("--prt-loss-mode", choices=("sampled_frame", "sequence"), default="sampled_frame")
    parser.add_argument("--prt-init-precision-xy", type=float, default=36.0)
    parser.add_argument("--prt-init-lambda-t", type=float, default=0.25)
    parser.add_argument("--prt-init-opacity", type=float, default=0.35)
    parser.add_argument("--prt-alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--prt-support-alpha-threshold", type=float)
    parser.add_argument("--init-depth", type=float, default=0.5)
    parser.add_argument("--profile-view", type=int, default=0)
    parser.add_argument("--profile-frame", type=int, default=0)
    parser.add_argument("--profile-warmups", type=int, default=1)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = run_profile(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
