from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_metal_forward_timing_probe import (  # noqa: E402
    _batch,
    _camera,
)
from research_project.benchmarks.projective_rational_world_camera_forward_probe import (  # noqa: E402
    dense_render_direct_world_tubes,
)
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    CameraPathPolynomial,
    WorldTubeBatch,
    centered_frame_times,
    compile_projective_rational_tubes,
    fit_camera_path_polynomial,
)
from research_project.trainer_harness.projective_rational_metal_autograd import (  # noqa: E402
    render_projective_rational_tubes_metal_direct_serial_backward,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    apply_projective_rational_tile_env,
    parse_projective_rational_tile_config,
    recommend_projective_rational_tile_config,
    render_projective_rational_tubes_tiled,
)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _logit(value: Tensor) -> Tensor:
    return torch.logit(value.clamp(1.0e-4, 1.0 - 1.0e-4))


def _softplus_inverse(value: Tensor) -> Tensor:
    return torch.log(torch.expm1(value.clamp_min(1.0e-6)))


def _psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1.0e-12))


def _image_metrics(image: Tensor, target: Tensor) -> dict[str, float]:
    mse = float((image - target).square().mean().detach().cpu())
    return {
        "mse": mse,
        "psnr": _psnr(mse),
        "l1": float((image - target).abs().mean().detach().cpu()),
        "max_abs": float((image - target).abs().max().detach().cpu()),
    }


def _camera_path_to_device(path: CameraPathPolynomial, device: torch.device) -> CameraPathPolynomial:
    return CameraPathPolynomial(
        p_coeff=path.p_coeff.to(device),
        frame_times=path.frame_times.to(device),
        fit_error=path.fit_error,
    )


def _holdout_camera(
    frames: int,
    width: int,
    height: int,
    times: Tensor,
    *,
    motion_scale: float,
    x_offset: float,
) -> tuple[Tensor, Tensor]:
    k_seq, w2c_seq = _camera(frames, width, height, times, motion_scale=motion_scale)
    w2c_seq = w2c_seq.clone()
    w2c_seq[:, 0, 3] += float(x_offset)
    w2c_seq[:, 1, 3] -= 0.4 * float(x_offset)
    return k_seq, w2c_seq


class TrainableWorldTubeModel(torch.nn.Module):
    def __init__(
        self,
        gt_batch: WorldTubeBatch,
        *,
        device: torch.device,
        seed: int,
        geometry_noise: float,
        velocity_noise: float,
        color_noise: float,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        x0 = gt_batch.x0 + geometry_noise * torch.randn(gt_batch.x0.shape, generator=generator)
        velocity = gt_batch.velocity + velocity_noise * torch.randn(gt_batch.velocity.shape, generator=generator)
        color = (gt_batch.color + color_noise * torch.randn(gt_batch.color.shape, generator=generator)).clamp(0.02, 0.98)
        opacity = gt_batch.opacity.clamp(0.02, 0.98)

        self.x0 = torch.nn.Parameter(x0.to(device))
        self.velocity = torch.nn.Parameter(velocity.to(device))
        self.raw_lambda_t = torch.nn.Parameter(_softplus_inverse(gt_batch.lambda_t.to(device)))
        self.raw_opacity = torch.nn.Parameter(_logit(opacity.to(device) / 0.99))
        self.raw_color = torch.nn.Parameter(_logit(color.to(device)))
        self.register_buffer("t0", gt_batch.t0.to(device))
        self.register_buffer("precision_xy", gt_batch.precision_xy.to(device))

    def batch(self) -> WorldTubeBatch:
        return WorldTubeBatch(
            x0=self.x0,
            velocity=self.velocity,
            t0=self.t0,
            precision_xy=self.precision_xy,
            lambda_t=torch.nn.functional.softplus(self.raw_lambda_t).clamp_min(1.0e-5),
            opacity=0.99 * torch.sigmoid(self.raw_opacity),
            color=torch.sigmoid(self.raw_color),
        )


def _compile_detached_footprint(model: TrainableWorldTubeModel, camera_path: CameraPathPolynomial):
    projected = compile_projective_rational_tubes(model.batch(), camera_path)
    return type(projected)(
        h_coeff=projected.h_coeff,
        lambda_uv=projected.lambda_uv.detach().contiguous(),
        lambda_t=projected.lambda_t,
        center_t=projected.center_t,
        depth_coeff=projected.depth_coeff,
        opacity=projected.opacity,
        color=projected.color,
        camera_fit_error=projected.camera_fit_error,
    )


def _render_prt_train(model: TrainableWorldTubeModel, camera_path: CameraPathPolynomial, config: UVTRenderConfig) -> Tensor:
    projected = _compile_detached_footprint(model, camera_path)
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


def _render_prt_eval(model: TrainableWorldTubeModel, camera_path: CameraPathPolynomial, config: UVTRenderConfig):
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


def _render_direct_world(
    model: TrainableWorldTubeModel,
    camera_path: CameraPathPolynomial,
    k_seq: Tensor,
    w2c_seq: Tensor,
    times: Tensor,
    *,
    height: int,
    width: int,
) -> Tensor:
    projected = _compile_detached_footprint(model, camera_path)
    return dense_render_direct_world_tubes(
        model.batch(),
        k_seq,
        w2c_seq,
        times,
        projected.lambda_uv,
        height=height,
        width=width,
    )


def _fit_model(
    model: TrainableWorldTubeModel,
    target: Tensor,
    render_fn: Callable[[], Tensor],
    *,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[list[float], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    started = time.perf_counter()
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image = render_fn()
        loss = (image - target).square().mean()
        losses.append(float(loss.detach().cpu()))
        if step == steps:
            break
        loss.backward()
        optimizer.step()
        _sync(device)
    _sync(device)
    return losses, (time.perf_counter() - started) * 1000.0


def _timed_render(render_fn: Callable[[], Tensor], *, device: torch.device, warmups: int, repeats: int) -> tuple[Tensor, dict[str, Any]]:
    image = None
    samples = []
    with torch.no_grad():
        for _ in range(warmups):
            image = render_fn()
            _sync(device)
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            image = render_fn()
            _sync(device)
            samples.append((time.perf_counter() - started) * 1000.0)
    if image is None:
        raise AssertionError("render did not run")
    return image, {
        "samples": samples,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "max_ms": max(samples),
        "warmups": warmups,
        "repeats": repeats,
    }


def _summarize_model(
    *,
    name: str,
    model: TrainableWorldTubeModel,
    train_image: Tensor,
    holdout_image: Tensor,
    train_target: Tensor,
    holdout_target: Tensor,
    train_render_timing: dict[str, Any],
    holdout_render_timing: dict[str, Any],
    train_wall_ms: float,
    losses: list[float],
    aux,
) -> dict[str, Any]:
    return {
        "name": name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / max(losses[0], 1.0e-12),
        "losses": losses,
        "train_wall_ms": train_wall_ms,
        "train_metrics": _image_metrics(train_image, train_target),
        "holdout_metrics": _image_metrics(holdout_image, holdout_target),
        "train_render_benchmark_ms": train_render_timing,
        "holdout_render_benchmark_ms": holdout_render_timing,
        "max_tile_count": None if aux is None else int(aux.tile_counts.max().detach().cpu()),
        "overflow_tile_count": None if aux is None else int((aux.tile_overflow > 0).sum().detach().cpu()),
        "active_tile_count": None if aux is None else int((aux.tile_counts > 0).sum().detach().cpu()),
    }


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "mps":
        raise ValueError("world-camera train compare currently requires --device=mps")
    torch.manual_seed(args.seed)

    times_cpu = centered_frame_times(args.frames, device="cpu")
    train_k_cpu, train_w2c_cpu = _camera(
        args.frames,
        args.width,
        args.height,
        times_cpu,
        motion_scale=args.camera_motion_scale,
    )
    holdout_k_cpu, holdout_w2c_cpu = _holdout_camera(
        args.frames,
        args.width,
        args.height,
        times_cpu,
        motion_scale=args.holdout_camera_motion_scale,
        x_offset=args.holdout_x_offset,
    )
    train_camera_path = _camera_path_to_device(
        fit_camera_path_polynomial(train_k_cpu, train_w2c_cpu, degree=args.camera_poly_degree, frame_times=times_cpu),
        device,
    )
    holdout_camera_path = _camera_path_to_device(
        fit_camera_path_polynomial(holdout_k_cpu, holdout_w2c_cpu, degree=args.camera_poly_degree, frame_times=times_cpu),
        device,
    )
    tile_config = (
        recommend_projective_rational_tile_config(
            tube_count=args.tube_count,
            camera_motion_scale=max(args.camera_motion_scale, args.holdout_camera_motion_scale),
        )
        if args.tile_config == "auto"
        else parse_projective_rational_tile_config(args.tile_config)
    )
    apply_projective_rational_tile_env(tile_config)
    config = UVTRenderConfig(height=args.height, width=args.width, frames=args.frames, **tile_config.as_render_kwargs())

    gt_cpu = _batch(args.tube_count, seed=args.seed)
    gt = WorldTubeBatch(
        x0=gt_cpu.x0.to(device),
        velocity=gt_cpu.velocity.to(device),
        t0=gt_cpu.t0.to(device),
        precision_xy=gt_cpu.precision_xy.to(device),
        lambda_t=gt_cpu.lambda_t.to(device),
        opacity=gt_cpu.opacity.to(device),
        color=gt_cpu.color.to(device),
    )
    train_k = train_k_cpu.to(device)
    train_w2c = train_w2c_cpu.to(device)
    holdout_k = holdout_k_cpu.to(device)
    holdout_w2c = holdout_w2c_cpu.to(device)
    times = times_cpu.to(device)
    train_gt_projected = compile_projective_rational_tubes(gt, train_camera_path)
    holdout_gt_projected = compile_projective_rational_tubes(gt, holdout_camera_path)
    with torch.no_grad():
        train_target = dense_render_direct_world_tubes(
            gt,
            train_k,
            train_w2c,
            times,
            train_gt_projected.lambda_uv.detach(),
            height=args.height,
            width=args.width,
        ).detach()
        holdout_target = dense_render_direct_world_tubes(
            gt,
            holdout_k,
            holdout_w2c,
            times,
            holdout_gt_projected.lambda_uv.detach(),
            height=args.height,
            width=args.width,
        ).detach()

    prt_model = TrainableWorldTubeModel(
        gt_cpu,
        device=device,
        seed=args.seed + 100,
        geometry_noise=args.init_geometry_noise,
        velocity_noise=args.init_velocity_noise,
        color_noise=args.init_color_noise,
    )
    direct_model = TrainableWorldTubeModel(
        gt_cpu,
        device=device,
        seed=args.seed + 100,
        geometry_noise=args.init_geometry_noise,
        velocity_noise=args.init_velocity_noise,
        color_noise=args.init_color_noise,
    )

    prt_losses, prt_train_wall_ms = _fit_model(
        prt_model,
        train_target,
        lambda: _render_prt_train(prt_model, train_camera_path, config),
        steps=args.steps,
        lr=args.lr,
        device=device,
    )
    direct_losses, direct_train_wall_ms = _fit_model(
        direct_model,
        train_target,
        lambda: _render_direct_world(
            direct_model,
            train_camera_path,
            train_k,
            train_w2c,
            times,
            height=args.height,
            width=args.width,
        ),
        steps=args.steps,
        lr=args.direct_lr,
        device=device,
    )

    prt_train_aux, prt_train_timing = _timed_render(
        lambda: _render_prt_eval(prt_model, train_camera_path, config),
        device=device,
        warmups=args.render_warmups,
        repeats=args.render_repeats,
    )
    prt_holdout_aux, prt_holdout_timing = _timed_render(
        lambda: _render_prt_eval(prt_model, holdout_camera_path, config),
        device=device,
        warmups=args.render_warmups,
        repeats=args.render_repeats,
    )
    direct_train_image, direct_train_timing = _timed_render(
        lambda: _render_direct_world(
            direct_model,
            train_camera_path,
            train_k,
            train_w2c,
            times,
            height=args.height,
            width=args.width,
        ),
        device=device,
        warmups=args.render_warmups,
        repeats=args.render_repeats,
    )
    direct_holdout_image, direct_holdout_timing = _timed_render(
        lambda: _render_direct_world(
            direct_model,
            holdout_camera_path,
            holdout_k,
            holdout_w2c,
            times,
            height=args.height,
            width=args.width,
        ),
        device=device,
        warmups=args.render_warmups,
        repeats=args.render_repeats,
    )

    prt = _summarize_model(
        name="projective_rational_world_tubes",
        model=prt_model,
        train_image=prt_train_aux.image,
        holdout_image=prt_holdout_aux.image,
        train_target=train_target,
        holdout_target=holdout_target,
        train_render_timing=prt_train_timing,
        holdout_render_timing=prt_holdout_timing,
        train_wall_ms=prt_train_wall_ms,
        losses=prt_losses,
        aux=prt_train_aux,
    )
    direct = _summarize_model(
        name="dense_direct_per_frame_projection_world_tubes",
        model=direct_model,
        train_image=direct_train_image,
        holdout_image=direct_holdout_image,
        train_target=train_target,
        holdout_target=holdout_target,
        train_render_timing=direct_train_timing,
        holdout_render_timing=direct_holdout_timing,
        train_wall_ms=direct_train_wall_ms,
        losses=direct_losses,
        aux=None,
    )
    return {
        "name": "projective_rational_world_camera_train_compare",
        "note": (
            "Synthetic world-camera train/holdout comparison. PRT uses tiled Metal projective-rational "
            "forward/backward. The baseline uses exact dense per-frame projection of the same trainable "
            "world tubes, not full 3DGS."
        ),
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "tube_count": args.tube_count,
        "steps": args.steps,
        "seed": args.seed,
        "device": str(device),
        "camera_motion_scale": args.camera_motion_scale,
        "holdout_camera_motion_scale": args.holdout_camera_motion_scale,
        "holdout_x_offset": args.holdout_x_offset,
        "camera_poly_degree": args.camera_poly_degree,
        "tile_config_key": tile_config.key,
        "train_camera_fit_error": train_camera_path.fit_error,
        "holdout_camera_fit_error": holdout_camera_path.fit_error,
        "tile_config": tile_config.as_dict(),
        "projective_rational": prt,
        "dense_direct_per_frame_projection": direct,
        "pass": prt["final_loss"] < prt["initial_loss"]
        and direct["final_loss"] < direct["initial_loss"]
        and int(prt["overflow_tile_count"]) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--tube-count", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--camera-motion-scale", type=float, default=3.0)
    parser.add_argument("--holdout-camera-motion-scale", type=float, default=2.0)
    parser.add_argument("--holdout-x-offset", type=float, default=0.08)
    parser.add_argument("--camera-poly-degree", type=int, default=2)
    parser.add_argument("--tile-config", default="auto", help="'auto' or an explicit config like 8x8x2:128")
    parser.add_argument("--init-geometry-noise", type=float, default=0.02)
    parser.add_argument("--init-velocity-noise", type=float, default=0.001)
    parser.add_argument("--init-color-noise", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--direct-lr", type=float, default=0.02)
    parser.add_argument("--render-warmups", type=int, default=1)
    parser.add_argument("--render-repeats", type=int, default=3)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = run_compare(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
