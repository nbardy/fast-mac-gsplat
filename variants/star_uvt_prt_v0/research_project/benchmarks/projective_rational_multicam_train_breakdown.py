from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch

from projective_rational_multicam_splat_compare import (  # noqa: E402
    DEFAULT_BASELINE_CONFIG,
    MulticamPRTWorldTubeModel,
    _camera_path_to_device,
    _camera_sequence,
    _compile_detached_footprint,
    _eval_prt,
    _resolve_dynaworld_path,
    _sync,
    centered_frame_times,
    fit_camera_path_polynomial,
    load_config_file,
    load_multicam_video_bundle,
)
from research_project.trainer_harness.projective_rational_metal_autograd import (  # noqa: E402
    render_projective_rational_tubes_metal_direct_serial_backward,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    apply_projective_rational_tile_env,
    parse_projective_rational_tile_config,
    projective_rational_tile_pixel_fused_mse_backward,
    projective_rational_tile_pixel_fused_mse_train_used_backward,
    recommend_projective_rational_tile_config,
)


TIMING_KEYS = (
    "sample_s",
    "zero_grad_s",
    "compile_s",
    "forward_s",
    "loss_s",
    "fused_mse_s",
    "backward_s",
    "clip_grad_s",
    "optimizer_s",
    "step_total_s",
)


def _time_call(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    _sync(device)
    started = time.perf_counter()
    result = fn()
    _sync(device)
    return result, time.perf_counter() - started


def _summarize_seconds(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_s": samples,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "max_s": max(samples),
    }


def _summarize_timing(rows: list[dict[str, float]]) -> dict[str, Any]:
    summary = {key: _summarize_seconds([float(row[key]) for row in rows]) for key in TIMING_KEYS}
    total = float(summary["step_total_s"]["median_s"])
    summary["median_share_of_step"] = {
        key: 0.0 if total <= 0.0 else float(value["median_s"]) / total
        for key, value in summary.items()
        if isinstance(value, dict) and key.endswith("_s")
    }
    return summary


def _render_projected_train(projected, config: UVTRenderConfig) -> torch.Tensor:
    return render_projective_rational_tubes_metal_direct_serial_backward(
        projected.h_coeff,
        projected.lambda_uv,
        projected.lambda_t,
        projected.center_t,
        projected.opacity,
        projected.color,
        config,
        forward_mode="tiled",
        backward_mode="tile_pixel_atomic",
    )


def _backward_projected_fused_mse(projected, result) -> None:
    backward_pairs = (
        (projected.h_coeff, result.grad_h_coeff),
        (projected.lambda_t, result.grad_lambda_t),
        (projected.center_t, result.grad_center_t),
        (projected.opacity, result.grad_opacity),
        (projected.color, result.grad_color),
    )
    tensors = [tensor for tensor, _grad in backward_pairs if tensor.requires_grad]
    grads = [grad for tensor, grad in backward_pairs if tensor.requires_grad]
    torch.autograd.backward(tensors, grads)


def _run_fused_mse_projected(projected, target: torch.Tensor, config: UVTRenderConfig, train_mode: str):
    op = (
        projective_rational_tile_pixel_fused_mse_train_used_backward
        if train_mode == "fused_mse_train_used"
        else projective_rational_tile_pixel_fused_mse_backward
    )
    result = op(
        projected.h_coeff,
        projected.lambda_uv,
        projected.lambda_t,
        projected.center_t,
        projected.opacity,
        projected.color,
        target,
        config,
    )
    if int((result.tile_overflow.detach().cpu() > 0).sum().item()) > 0:
        raise RuntimeError("fused MSE PRT train step overflowed tile capacity")
    loss_value = float((result.loss_sum.detach().cpu()[0] / float(target.numel())).item())
    return result, loss_value


def _fit_prt_with_breakdown(
    *,
    model: MulticamPRTWorldTubeModel,
    bundle,
    train_camera_paths,
    config: UVTRenderConfig,
    steps: int,
    lr: float,
    loss_mode: str,
    train_mode: str,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    if loss_mode not in {"sampled_frame", "sequence"}:
        raise ValueError("loss_mode must be one of: sampled_frame, sequence")
    if train_mode not in {"separate", "fused_mse", "fused_mse_train_used"}:
        raise ValueError("train_mode must be one of: separate, fused_mse, fused_mse_train_used")
    if train_mode in {"fused_mse", "fused_mse_train_used"} and loss_mode != "sequence":
        raise ValueError("fused MSE train modes currently require --prt-loss-mode sequence")
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    timing_rows = []
    started = time.perf_counter()
    for step in range(steps + 1):
        _sync(device)
        step_started = time.perf_counter()

        (view, frame), sample_s = _time_call(
            device,
            lambda: (
                int(torch.randint(0, bundle.train_view_count, (1,), generator=generator, device=device).item()),
                int(torch.randint(0, bundle.frame_count, (1,), generator=generator, device=device).item()),
            ),
        )
        _unused, zero_grad_s = _time_call(device, lambda: optimizer.zero_grad(set_to_none=True))
        projected, compile_s = _time_call(
            device,
            lambda view=view: _compile_detached_footprint(model, train_camera_paths[view]),
        )
        target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
        forward_s = 0.0
        loss_s = 0.0
        fused_mse_s = 0.0
        result = None
        if train_mode in {"fused_mse", "fused_mse_train_used"}:
            (result, loss_value), fused_mse_s = _time_call(
                device,
                lambda projected=projected, target=target: _run_fused_mse_projected(
                    projected,
                    target,
                    config,
                    train_mode,
                ),
            )
        else:
            image, forward_s = _time_call(device, lambda: _render_projected_train(projected, config))

            def make_loss() -> torch.Tensor:
                if loss_mode == "sampled_frame":
                    return (image[frame] - target[frame]).square().mean()
                return (image - target).square().mean()

            loss, loss_s = _time_call(device, make_loss)
            loss_value = float(loss.detach().cpu())

        backward_s = 0.0
        clip_grad_s = 0.0
        optimizer_s = 0.0
        if step < steps:
            if train_mode in {"fused_mse", "fused_mse_train_used"}:
                _unused, backward_s = _time_call(device, lambda: _backward_projected_fused_mse(projected, result))
            else:
                _unused, backward_s = _time_call(device, lambda: loss.backward())
            _unused, clip_grad_s = _time_call(device, lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            _unused, optimizer_s = _time_call(device, optimizer.step)

        _sync(device)
        step_total_s = time.perf_counter() - step_started
        losses.append({"step": step, "view": view, "frame": frame, "loss": loss_value})
        timing_rows.append(
            {
                "step": step,
                "view": view,
                "frame": frame,
                "sample_s": sample_s,
                "zero_grad_s": zero_grad_s,
                "compile_s": compile_s,
                "forward_s": forward_s,
                "loss_s": loss_s,
                "fused_mse_s": fused_mse_s,
                "backward_s": backward_s,
                "clip_grad_s": clip_grad_s,
                "optimizer_s": optimizer_s,
                "step_total_s": step_total_s,
            }
        )

    return {
        "steps": steps,
        "loss_mode": loss_mode,
        "train_mode": train_mode,
        "train_loop_elapsed_s": time.perf_counter() - started,
        "diagnostic_sync_boundaries": True,
        "losses": losses,
        "initial_loss": losses[0]["loss"],
        "final_loss": losses[-1]["loss"],
        "sampled_loss_decreased": losses[-1]["loss"] < losses[0]["loss"],
        "timing_summary_s": _summarize_timing(timing_rows),
        "timing_rows": timing_rows,
    }


def _finite_loss_pair(train_report: dict[str, Any]) -> bool:
    return math.isfinite(float(train_report["initial_loss"])) and math.isfinite(float(train_report["final_loss"]))


def run_breakdown(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "mps":
        raise ValueError("projective rational multicam train breakdown currently requires --device=mps")
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

    heldout_camera_paths = []
    heldout_fit_errors = []
    if bundle.heldout_K is not None and bundle.heldout_w2c is not None:
        for view in range(bundle.heldout_view_count):
            k_seq, w2c_seq = _camera_sequence(
                bundle.heldout_K.detach().cpu(),
                bundle.heldout_w2c.detach().cpu(),
                view=view,
                frames=frames,
                view_count=bundle.heldout_view_count,
            )
            path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=args.camera_poly_degree, frame_times=frame_times_cpu)
            heldout_fit_errors.append(path.fit_error)
            heldout_camera_paths.append(_camera_path_to_device(path, device))

    tile_config = (
        recommend_projective_rational_tile_config(tube_count=args.prt_tubes)
        if args.tile_config == "auto"
        else parse_projective_rational_tile_config(args.tile_config)
    )
    apply_projective_rational_tile_env(tile_config)
    prt_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        background=(1.0, 1.0, 1.0),
        **tile_config.as_render_kwargs(),
        support_alpha_threshold=args.prt_support_alpha_threshold,
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
        train_mode=args.prt_train_mode,
        device=device,
        seed=args.seed + 17,
    )
    eval_report = _eval_prt(
        model=model,
        bundle=bundle,
        train_camera_paths=train_camera_paths,
        heldout_camera_paths=heldout_camera_paths,
        config=prt_config,
        device=device,
        render_warmups=args.render_warmups,
        render_repeats=args.render_repeats,
        cache_compiled=True,
    )

    return {
        "name": "projective_rational_multicam_train_breakdown",
        "note": (
            "Diagnostic sync-boundary PRT train-step breakdown on the real D2 multicam bundle. "
            "Timing rows are slower than normal training because each segment synchronizes."
        ),
        "pass": _finite_loss_pair(train) and int(eval_report["overflow_tile_count"]) == 0,
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
            "render_warmups": args.render_warmups,
            "render_repeats": args.render_repeats,
            "prt_support_alpha_threshold": args.prt_support_alpha_threshold,
            "train_cameras": bundle.train_camera_names,
            "heldout_cameras": bundle.heldout_camera_names,
            "pose_source": bundle.pose_source,
            "sample_id": None if bundle.metadata is None else bundle.metadata.get("sample_id"),
            "camera_poly_degree": args.camera_poly_degree,
            "train_camera_fit_errors": train_fit_errors,
            "heldout_camera_fit_errors": heldout_fit_errors,
        },
        "projective_rational": {
            "tube_count": args.prt_tubes,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "tile_config_key": tile_config.key,
            "tile_config": tile_config.as_dict(),
            "lr": args.prt_lr,
            "loss_mode": args.prt_loss_mode,
            "train_mode": args.prt_train_mode,
            "support_alpha_threshold": args.prt_support_alpha_threshold,
            "init_depth": args.init_depth,
            "init_precision_xy": args.prt_init_precision_xy,
            "init_lambda_t": args.prt_init_lambda_t,
            "init_opacity": args.prt_init_opacity,
            "train": train,
            "eval": eval_report,
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
    parser.add_argument("--prt-tubes", type=int, default=128)
    parser.add_argument("--prt-lr", type=float, default=0.02)
    parser.add_argument("--prt-loss-mode", choices=("sampled_frame", "sequence"), default="sampled_frame")
    parser.add_argument("--prt-train-mode", choices=("separate", "fused_mse", "fused_mse_train_used"), default="separate")
    parser.add_argument("--prt-init-precision-xy", type=float, default=36.0)
    parser.add_argument("--prt-init-lambda-t", type=float, default=0.25)
    parser.add_argument("--prt-init-opacity", type=float, default=0.35)
    parser.add_argument("--prt-support-alpha-threshold", type=float)
    parser.add_argument("--init-depth", type=float, default=0.5)
    parser.add_argument("--render-warmups", type=int, default=1)
    parser.add_argument("--render-repeats", type=int, default=3)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = run_breakdown(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
