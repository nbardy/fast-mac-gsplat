from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from projective_rational_multicam_splat_compare import (  # noqa: E402
    DEFAULT_BASELINE_CONFIG,
    MulticamPRTWorldTubeModel,
    _camera_path_to_device,
    _camera_sequence,
    _compile_detached_footprint,
    _fit_prt,
    _metrics,
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
    profile_projective_rational_tubes_tiled,
    recommend_projective_rational_tile_config,
)


def _finite_loss_pair(train_report: dict[str, Any]) -> bool:
    return math.isfinite(float(train_report["initial_loss"])) and math.isfinite(float(train_report["final_loss"]))


def _summarize_ms(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": samples,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "max_ms": max(samples),
    }


def _summarize_phase_rows(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = tuple(rows[0])
    phase = {key: _summarize_ms([float(row[key]) for row in rows]) for key in keys}
    total = phase["total_ms"]["median_ms"]
    phase["median_share_of_total"] = {
        key: (0.0 if total <= 0.0 else float(value["median_ms"]) / float(total))
        for key, value in phase.items()
        if isinstance(value, dict) and key.endswith("_ms")
    }
    return phase


@torch.no_grad()
def _profile_camera(
    *,
    model: MulticamPRTWorldTubeModel,
    camera_path,
    target: torch.Tensor,
    config: UVTRenderConfig,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    def compile_once():
        return _compile_detached_footprint(model, camera_path)

    for _ in range(warmups):
        _sync(device)
        compile_once()
        _sync(device)
    compile_ms = []
    projected = None
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        projected = compile_once()
        _sync(device)
        compile_ms.append((time.perf_counter() - started) * 1000.0)
    if projected is None:
        raise ValueError("repeats must be positive")

    def one_call():
        return profile_projective_rational_tubes_tiled(
            projected.h_coeff,
            projected.lambda_uv,
            projected.lambda_t,
            projected.center_t,
            projected.opacity,
            projected.color,
            config,
        )

    for _ in range(warmups):
        one_call()
    rows = []
    walls = []
    result = None
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        result = one_call()
        _sync(device)
        walls.append((time.perf_counter() - started) * 1000.0)
        rows.append(result.timings_ms)
    if result is None:
        raise ValueError("repeats must be positive")
    return {
        "compile_ms": _summarize_ms(compile_ms),
        "metrics": _metrics(result.image, target),
        "wall_ms": _summarize_ms(walls),
        "phase_ms": _summarize_phase_rows(rows),
        "max_tile_count": int(result.tile_counts.max().detach().cpu()),
        "overflow_tile_count": int((result.tile_overflow > 0).sum().detach().cpu()),
        "unstable_tile_count": int((result.tile_unstable > 0).sum().detach().cpu()),
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.render_warmups < 0:
        raise ValueError("render_warmups must be non-negative")
    if args.render_repeats <= 0:
        raise ValueError("render_repeats must be positive")
    device = torch.device(args.device)
    if device.type != "mps":
        raise ValueError("projective rational multicam phase profile currently requires --device=mps")
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
    train = _fit_prt(
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

    train_target = bundle.train_frames[args.train_profile_view].permute(0, 2, 3, 1).contiguous()
    train_profile = _profile_camera(
        model=model,
        camera_path=train_camera_paths[args.train_profile_view],
        target=train_target,
        config=prt_config,
        device=device,
        warmups=args.render_warmups,
        repeats=args.render_repeats,
    )

    heldout_profile = None
    if heldout_camera_paths:
        heldout_target = bundle.heldout_frames[args.heldout_profile_view].permute(0, 2, 3, 1).contiguous()
        heldout_profile = _profile_camera(
            model=model,
            camera_path=heldout_camera_paths[args.heldout_profile_view],
            target=heldout_target,
            config=prt_config,
            device=device,
            warmups=args.render_warmups,
            repeats=args.render_repeats,
        )

    overflow_count = int(train_profile["overflow_tile_count"])
    if heldout_profile is not None:
        overflow_count += int(heldout_profile["overflow_tile_count"])
    return {
        "name": "projective_rational_multicam_phase_profile",
        "note": (
            "Profiles the PRT tiled Metal forward op phases after training a small real-multicam PRT model. "
            "The phase op synchronizes after clear, bin, and render, so these numbers are diagnostic wall timings."
        ),
        "pass": _finite_loss_pair(train) and overflow_count == 0,
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
            "train_cameras": bundle.train_camera_names,
            "heldout_cameras": bundle.heldout_camera_names,
            "train_profile_view": args.train_profile_view,
            "heldout_profile_view": None if not heldout_camera_paths else args.heldout_profile_view,
            "pose_source": bundle.pose_source,
            "sample_id": None if bundle.metadata is None else bundle.metadata.get("sample_id"),
            "camera_poly_degree": args.camera_poly_degree,
            "train_camera_fit_errors": train_fit_errors,
            "heldout_camera_fit_errors": heldout_fit_errors,
            "profile_phase_compile_excluded": True,
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
            "train_profile": train_profile,
            "heldout_profile": heldout_profile,
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
    parser.add_argument("--prt-init-precision-xy", type=float, default=36.0)
    parser.add_argument("--prt-init-lambda-t", type=float, default=0.25)
    parser.add_argument("--prt-init-opacity", type=float, default=0.35)
    parser.add_argument("--init-depth", type=float, default=0.5)
    parser.add_argument("--render-warmups", type=int, default=1)
    parser.add_argument("--render-repeats", type=int, default=5)
    parser.add_argument("--train-profile-view", type=int, default=0)
    parser.add_argument("--heldout-profile-view", type=int, default=0)
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
