from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_dynaworld_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "train" / "multicam_video_data.py").exists():
            return parent
    raise FileNotFoundError("could not find dynaworld root from PRT variant")


DYNAWORLD_ROOT = _find_dynaworld_root()
TRAIN_SRC = DYNAWORLD_ROOT / "src" / "train"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from config_utils import load_config_file, serialize_config_value  # noqa: E402
from multicam_video_data import load_multicam_video_bundle  # noqa: E402
from research_project.benchmarks.projective_rational_learnable_camera_gate import (  # noqa: E402
    LearnableSE3CameraPath,
)
from research_project.benchmarks.projective_rational_multicam_splat_compare import (  # noqa: E402
    DEFAULT_BASELINE_CONFIG,
    MulticamPRTWorldTubeModel,
    _aggregate_metric_rows,
    _camera_path_to_device,
    _camera_sequence,
    _compile_detached_footprint,
    _metrics,
    _resolve_dynaworld_path,
    _render_prt_train,
    _summarize_seconds,
    _sync,
    _time_repeated,
)
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    CameraPathPolynomial,
    centered_frame_times,
    fit_camera_path_polynomial,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    apply_projective_rational_tile_env,
    parse_projective_rational_tile_config,
    recommend_projective_rational_train_speed_tile_policy,
    recommend_projective_rational_tile_config,
    render_projective_rational_tubes_tiled,
)
from video_io import save_mp4  # noqa: E402


def _init_wandb(args: argparse.Namespace, report_config: dict[str, Any]):
    if not args.wandb_enabled:
        return None
    import wandb

    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "tags": [tag for tag in args.wandb_tags.split(",") if tag],
        "config": serialize_config_value(report_config),
    }
    if args.wandb_mode is not None:
        init_kwargs["mode"] = args.wandb_mode
    return wandb.init(**init_kwargs)


def _make_wandb_video(sequence_nchw: torch.Tensor, fps: float):
    import wandb

    video = (sequence_nchw.detach().cpu().clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
    return wandb.Video(video, fps=max(1, int(round(float(fps)))), format="mp4")


def _nhwc_to_nchw(sequence: torch.Tensor) -> torch.Tensor:
    return sequence.detach().permute(0, 3, 1, 2).contiguous()


def _side_by_side_nchw(target_nhwc: torch.Tensor, render_nhwc: torch.Tensor) -> torch.Tensor:
    return _nhwc_to_nchw(torch.cat((target_nhwc.detach(), render_nhwc.detach()), dim=2))


def _camera_metrics(camera_paths: list[LearnableSE3CameraPath]) -> dict[str, float]:
    rows = [camera.metrics() for camera in camera_paths]
    if not rows:
        return {}
    return {
        key: sum(float(row[key]) for row in rows) / float(len(rows))
        for key in rows[0]
    }


def _gradient_checks(logs: list[dict[str, Any]]) -> dict[str, float]:
    if not logs:
        return {"raw_rotation_grad_norm_max": 0.0, "raw_translation_grad_norm_max": 0.0}
    return {
        "raw_rotation_grad_norm_max": max(float(row["camera"]["raw_rotation_grad_norm"]) for row in logs),
        "raw_translation_grad_norm_max": max(float(row["camera"]["raw_translation_grad_norm"]) for row in logs),
    }


def _fit_prt_learnable_camera(
    *,
    model: MulticamPRTWorldTubeModel,
    bundle,
    train_cameras: list[LearnableSE3CameraPath],
    config: UVTRenderConfig,
    steps: int,
    world_lr: float,
    camera_lr: float,
    camera_reg_weight: float,
    camera_temporal_weight: float,
    loss_mode: str,
    device: torch.device,
    seed: int,
    log_every: int,
) -> dict[str, Any]:
    if loss_mode not in {"sampled_frame", "sequence"}:
        raise ValueError("loss_mode must be one of: sampled_frame, sequence")
    params = [
        {"params": model.parameters(), "lr": world_lr},
        {"params": [param for camera in train_cameras for param in camera.parameters()], "lr": camera_lr},
    ]
    optimizer = torch.optim.Adam(params)
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    logs = []
    started = time.perf_counter()
    for step in range(steps + 1):
        view = int(torch.randint(0, bundle.train_view_count, (1,), generator=generator, device=device).item())
        frame = int(torch.randint(0, bundle.frame_count, (1,), generator=generator, device=device).item())
        optimizer.zero_grad(set_to_none=True)
        image = _render_prt_train(model, train_cameras[view].camera_path(), config)
        target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
        if loss_mode == "sampled_frame":
            recon = (image[frame] - target[frame]).square().mean()
        else:
            recon = (image - target).square().mean()
        camera_reg = train_cameras[view].regularization_loss()
        camera_temporal = train_cameras[view].temporal_smoothness_loss()
        loss = recon + float(camera_reg_weight) * camera_reg + float(camera_temporal_weight) * camera_temporal
        losses.append({"step": int(step), "view": view, "frame": frame, "loss": float(recon.detach().cpu())})
        if step == steps:
            break
        loss.backward()
        camera_row = train_cameras[view].metrics()
        if step % max(1, log_every) == 0 or step == steps - 1:
            logs.append(
                {
                    "step": int(step),
                    "view": view,
                    "frame": frame,
                    "loss": float(loss.detach().cpu()),
                    "recon_loss": float(recon.detach().cpu()),
                    "camera_regularization": float(camera_reg.detach().cpu()),
                    "camera_temporal": float(camera_temporal.detach().cpu()),
                    "camera": camera_row,
                }
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for camera in train_cameras:
            torch.nn.utils.clip_grad_norm_(camera.parameters(), 1.0)
        optimizer.step()
        _sync(device)
    return {
        "steps": steps,
        "loss_mode": loss_mode,
        "train_loop_elapsed_s": time.perf_counter() - started,
        "losses": losses,
        "logs": logs,
        "initial_loss": losses[0]["loss"],
        "final_loss": losses[-1]["loss"],
        "sampled_loss_decreased": losses[-1]["loss"] < losses[0]["loss"],
        "gradient_checks": _gradient_checks(logs),
        "camera_metrics": _camera_metrics(train_cameras),
    }


def _render_eval_sequence(model: MulticamPRTWorldTubeModel, camera_path: CameraPathPolynomial, config: UVTRenderConfig):
    projected = _compile_detached_footprint(model, camera_path)
    return render_projective_rational_tubes_tiled(
        projected.h_coeff,
        projected.lambda_uv,
        projected.lambda_t,
        projected.center_t,
        projected.opacity,
        projected.color,
        config,
        return_aux=True,
    )


@torch.no_grad()
def _eval_prt_learnable_camera(
    *,
    model: MulticamPRTWorldTubeModel,
    bundle,
    train_cameras: list[LearnableSE3CameraPath],
    heldout_camera_paths: list[CameraPathPolynomial],
    config: UVTRenderConfig,
    device: torch.device,
    render_warmups: int,
    render_repeats: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    train_rows = []
    train_times = []
    max_tile_count = 0
    overflow_tile_count = 0
    media: dict[str, torch.Tensor] = {}
    for view, camera in enumerate(train_cameras):
        aux, elapsed = _time_repeated(
            device,
            lambda camera=camera: _render_eval_sequence(model, camera.camera_path(), config),
            warmups=render_warmups,
            repeats=render_repeats,
        )
        train_times.extend(elapsed)
        max_tile_count = max(max_tile_count, int(aux.tile_counts.max().detach().cpu()))
        overflow_tile_count += int((aux.tile_overflow > 0).sum().detach().cpu())
        target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
        train_rows.append(_metrics(aux.image, target))
        if view == 0:
            media["train_render_nhwc"] = aux.image.detach()
            media["train_target_nhwc"] = target.detach()

    heldout_rows = []
    heldout_times = []
    if bundle.heldout_frames is not None:
        for view, camera_path in enumerate(heldout_camera_paths):
            aux, elapsed = _time_repeated(
                device,
                lambda camera_path=camera_path: _render_eval_sequence(model, camera_path, config),
                warmups=render_warmups,
                repeats=render_repeats,
            )
            heldout_times.extend(elapsed)
            max_tile_count = max(max_tile_count, int(aux.tile_counts.max().detach().cpu()))
            overflow_tile_count += int((aux.tile_overflow > 0).sum().detach().cpu())
            target = bundle.heldout_frames[view].permute(0, 2, 3, 1).contiguous()
            heldout_rows.append(_metrics(aux.image, target))
            if view == 0:
                media["heldout_render_nhwc"] = aux.image.detach()
                media["heldout_target_nhwc"] = target.detach()

    metrics = _aggregate_metric_rows(train_rows)
    if heldout_rows:
        metrics.update({f"heldout_{key}": value for key, value in _aggregate_metric_rows(heldout_rows).items()})
    return (
        {
            "metrics": metrics,
            "train_render_seconds": _summarize_seconds(train_times),
            "heldout_render_seconds": None if not heldout_times else _summarize_seconds(heldout_times),
            "max_tile_count": max_tile_count,
            "overflow_tile_count": overflow_tile_count,
        },
        media,
    )


def _build_train_cameras(
    *,
    bundle,
    frame_times: torch.Tensor,
    degree: int,
    max_rotation_degrees: float,
    max_translation: float,
    base_x_offset: float,
    base_y_offset: float,
    base_z_offset: float,
    device: torch.device,
) -> tuple[list[LearnableSE3CameraPath], list[float]]:
    cameras = []
    fit_errors = []
    for view in range(bundle.train_view_count):
        k_seq_cpu, w2c_seq_cpu = _camera_sequence(
            bundle.train_K.detach().cpu(),
            bundle.train_w2c.detach().cpu(),
            view=view,
            frames=bundle.frame_count,
            view_count=bundle.train_view_count,
        )
        w2c_seq_cpu = w2c_seq_cpu.clone()
        w2c_seq_cpu[:, 0, 3] += float(base_x_offset)
        w2c_seq_cpu[:, 1, 3] += float(base_y_offset)
        w2c_seq_cpu[:, 2, 3] += float(base_z_offset)
        fit_errors.append(
            fit_camera_path_polynomial(k_seq_cpu, w2c_seq_cpu, degree=degree, frame_times=frame_times).fit_error
        )
        cameras.append(
            LearnableSE3CameraPath(
                k_seq_cpu.to(device),
                w2c_seq_cpu.to(device),
                frame_times.to(device),
                degree=degree,
                max_rotation_degrees=max_rotation_degrees,
                max_translation=max_translation,
            ).to(device)
        )
    return cameras, fit_errors


def _build_heldout_camera_paths(
    *,
    bundle,
    frame_times: torch.Tensor,
    degree: int,
    device: torch.device,
) -> tuple[list[CameraPathPolynomial], list[float]]:
    paths = []
    fit_errors = []
    if bundle.heldout_K is None or bundle.heldout_w2c is None:
        return paths, fit_errors
    for view in range(bundle.heldout_view_count):
        k_seq_cpu, w2c_seq_cpu = _camera_sequence(
            bundle.heldout_K.detach().cpu(),
            bundle.heldout_w2c.detach().cpu(),
            view=view,
            frames=bundle.frame_count,
            view_count=bundle.heldout_view_count,
        )
        path = fit_camera_path_polynomial(k_seq_cpu, w2c_seq_cpu, degree=degree, frame_times=frame_times)
        fit_errors.append(path.fit_error)
        paths.append(_camera_path_to_device(path, device))
    return paths, fit_errors


def _write_media(out_dir: Path, media: dict[str, torch.Tensor], fps: float) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for prefix in ("train", "heldout"):
        render = media.get(f"{prefix}_render_nhwc")
        target = media.get(f"{prefix}_target_nhwc")
        if render is None or target is None:
            continue
        render_path = out_dir / f"{prefix}_render.mp4"
        side_path = out_dir / f"{prefix}_gt_render.mp4"
        save_mp4(render_path, _nhwc_to_nchw(render), fps=fps)
        save_mp4(side_path, _side_by_side_nchw(target, render), fps=fps)
        written[f"{prefix}_render"] = str(render_path)
        written[f"{prefix}_gt_render"] = str(side_path)
    return written


def _log_wandb(run, report: dict[str, Any], media: dict[str, torch.Tensor], fps: float) -> None:
    if run is None:
        return
    payload = {
        "Train/PSNR": report["eval"]["metrics"].get("psnr"),
        "Train/MSE": report["eval"]["metrics"].get("mse"),
        "Train/L1": report["eval"]["metrics"].get("l1"),
        "Train/FinalSampledLoss": report["train"]["final_loss"],
        "Camera/RawRotationGradMax": report["train"]["gradient_checks"]["raw_rotation_grad_norm_max"],
        "Camera/RawTranslationGradMax": report["train"]["gradient_checks"]["raw_translation_grad_norm_max"],
        "Tiles/MaxTileCount": report["eval"]["max_tile_count"],
        "Tiles/OverflowTileCount": report["eval"]["overflow_tile_count"],
    }
    if "heldout_psnr" in report["eval"]["metrics"]:
        payload["Heldout/PSNR"] = report["eval"]["metrics"]["heldout_psnr"]
        payload["Heldout/MSE"] = report["eval"]["metrics"]["heldout_mse"]
        payload["Heldout/L1"] = report["eval"]["metrics"]["heldout_l1"]
    for prefix in ("train", "heldout"):
        render = media.get(f"{prefix}_render_nhwc")
        target = media.get(f"{prefix}_target_nhwc")
        if render is None or target is None:
            continue
        payload[f"{prefix}/Render_Video"] = _make_wandb_video(_nhwc_to_nchw(render), fps)
        payload[f"{prefix}/GT_Render_Video"] = _make_wandb_video(_side_by_side_nchw(target, render), fps)
    run.log(payload, step=int(report["train"]["steps"]))
    run.finish()


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "mps":
        raise ValueError("real-clip learnable-camera PRT trainer currently requires --device=mps")
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

    tile_policy_name = "generic_auto"
    prt_support_alpha_threshold = args.prt_support_alpha_threshold
    if args.prt_tile_policy == "train_speed":
        if args.tile_config != "auto":
            raise ValueError("--prt-tile-policy train_speed requires --tile-config auto")
        tile_policy = recommend_projective_rational_train_speed_tile_policy(tube_count=args.prt_tubes)
        tile_config = tile_policy.tile_config
        tile_policy_name = tile_policy.name
        if prt_support_alpha_threshold is None:
            prt_support_alpha_threshold = tile_policy.support_alpha_threshold
    else:
        tile_config = (
            recommend_projective_rational_tile_config(tube_count=args.prt_tubes)
            if args.tile_config == "auto"
            else parse_projective_rational_tile_config(args.tile_config)
        )
    apply_projective_rational_tile_env(tile_config)
    render_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        alpha_threshold=args.prt_alpha_threshold,
        support_alpha_threshold=prt_support_alpha_threshold,
        background=(1.0, 1.0, 1.0),
        **tile_config.as_render_kwargs(),
    )

    train_cameras, train_fit_errors = _build_train_cameras(
        bundle=bundle,
        frame_times=frame_times_cpu,
        degree=args.camera_poly_degree,
        max_rotation_degrees=args.max_rotation_degrees,
        max_translation=args.max_translation,
        base_x_offset=args.train_camera_base_x_offset,
        base_y_offset=args.train_camera_base_y_offset,
        base_z_offset=args.train_camera_base_z_offset,
        device=device,
    )
    heldout_camera_paths, heldout_fit_errors = _build_heldout_camera_paths(
        bundle=bundle,
        frame_times=frame_times_cpu,
        degree=args.camera_poly_degree,
        device=device,
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
    report_config = {
        "baseline_config": str(_resolve_dynaworld_path(args.baseline_config)),
        "target_size": args.target_size,
        "max_frames": args.max_frames,
        "frames": frames,
        "height": height,
        "width": width,
        "seed": args.seed,
        "train_cameras": bundle.train_camera_names,
        "heldout_cameras": bundle.heldout_camera_names,
        "sample_id": None if bundle.metadata is None else bundle.metadata.get("sample_id"),
        "prt_tubes": args.prt_tubes,
        "steps": args.steps,
        "world_lr": args.world_lr,
        "camera_lr": args.camera_lr,
        "camera_poly_degree": args.camera_poly_degree,
        "tile_config": tile_config.as_dict(),
        "tile_policy": tile_policy_name,
    }
    wandb_run = _init_wandb(args, report_config)
    train = _fit_prt_learnable_camera(
        model=model,
        bundle=bundle,
        train_cameras=train_cameras,
        config=render_config,
        steps=args.steps,
        world_lr=args.world_lr,
        camera_lr=args.camera_lr,
        camera_reg_weight=args.camera_reg_weight,
        camera_temporal_weight=args.camera_temporal_weight,
        loss_mode=args.loss_mode,
        device=device,
        seed=args.seed + 17,
        log_every=args.log_every,
    )
    eval_report, media = _eval_prt_learnable_camera(
        model=model,
        bundle=bundle,
        train_cameras=train_cameras,
        heldout_camera_paths=heldout_camera_paths,
        config=render_config,
        device=device,
        render_warmups=args.render_warmups,
        render_repeats=args.render_repeats,
    )
    report = {
        "name": "projective_rational_multicam_learnable_camera_train",
        "note": (
            "Real multicam clip PRT trainer. This forks the real-clip PRT-vs-splat harness and trains "
            "world tubes plus bounded train-camera SE3 residuals; heldout cameras stay fixed for evaluation."
        ),
        "pass": (
            math.isfinite(float(train["initial_loss"]))
            and math.isfinite(float(train["final_loss"]))
            and eval_report["overflow_tile_count"] == 0
            and train["gradient_checks"]["raw_translation_grad_norm_max"] > 0.0
        ),
        "meta": {
            **report_config,
            "device": str(device),
            "pose_source": bundle.pose_source,
            "camera_regularization_weight": args.camera_reg_weight,
            "camera_temporal_weight": args.camera_temporal_weight,
            "train_camera_base_offsets": {
                "x": args.train_camera_base_x_offset,
                "y": args.train_camera_base_y_offset,
                "z": args.train_camera_base_z_offset,
            },
            "train_camera_fit_errors": train_fit_errors,
            "heldout_camera_fit_errors": heldout_fit_errors,
            "prt_support_alpha_threshold": prt_support_alpha_threshold,
            "prt_alpha_threshold": args.prt_alpha_threshold,
        },
        "model": {
            "parameter_count": sum(parameter.numel() for parameter in model.parameters())
            + sum(parameter.numel() for camera in train_cameras for parameter in camera.parameters()),
            "world_tube_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "camera_parameter_count": sum(parameter.numel() for camera in train_cameras for parameter in camera.parameters()),
            "tile_config_key": tile_config.key,
            "tile_config": tile_config.as_dict(),
        },
        "train": train,
        "eval": eval_report,
    }
    if args.out_dir is not None:
        report["media_paths"] = _write_media(args.out_dir, media, fps=args.video_fps)
    _log_wandb(wandb_run, report, media, fps=args.video_fps)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--camera-poly-degree", type=int, default=1)
    parser.add_argument("--max-rotation-degrees", type=float, default=3.0)
    parser.add_argument("--max-translation", type=float, default=0.08)
    parser.add_argument("--train-camera-base-x-offset", type=float, default=0.0)
    parser.add_argument("--train-camera-base-y-offset", type=float, default=0.0)
    parser.add_argument("--train-camera-base-z-offset", type=float, default=0.0)
    parser.add_argument("--tile-config", default="auto", help="'auto' or an explicit config like 8x8x2:128")
    parser.add_argument("--prt-tile-policy", choices=("generic", "train_speed"), default="generic")
    parser.add_argument("--prt-tubes", type=int, default=128)
    parser.add_argument("--world-lr", type=float, default=0.02)
    parser.add_argument("--camera-lr", type=float, default=0.03)
    parser.add_argument("--camera-reg-weight", type=float, default=0.001)
    parser.add_argument("--camera-temporal-weight", type=float, default=0.001)
    parser.add_argument("--loss-mode", choices=("sampled_frame", "sequence"), default="sampled_frame")
    parser.add_argument("--prt-init-precision-xy", type=float, default=36.0)
    parser.add_argument("--prt-init-lambda-t", type=float, default=0.25)
    parser.add_argument("--prt-init-opacity", type=float, default=0.35)
    parser.add_argument("--prt-alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--prt-support-alpha-threshold", type=float)
    parser.add_argument("--init-depth", type=float, default=0.5)
    parser.add_argument("--render-warmups", type=int, default=0)
    parser.add_argument("--render-repeats", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", default="dynaworld")
    parser.add_argument("--wandb-run-name", default="star-uvt-prt-real-clip-learnable-camera")
    parser.add_argument("--wandb-tags", default="star-uvt-prt,real-clip,learnable-camera")
    parser.add_argument("--wandb-mode")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    report = run_train(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
