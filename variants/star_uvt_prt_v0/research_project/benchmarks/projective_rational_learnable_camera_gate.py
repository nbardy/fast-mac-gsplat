from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

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
from research_project.benchmarks.projective_rational_world_camera_train_compare import (  # noqa: E402
    TrainableWorldTubeModel,
    _image_metrics,
    _psnr,
    _sync,
)
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    CameraPathPolynomial,
    WorldTubeBatch,
    centered_frame_times,
    compile_projective_rational_tubes,
    evaluate_camera_polynomial,
    fit_camera_path_polynomial,
    projection_matrices,
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


def _skew_symmetric(vectors: Tensor) -> Tensor:
    skew = vectors.new_zeros((int(vectors.shape[0]), 3, 3))
    skew[:, 0, 1] = -vectors[:, 2]
    skew[:, 0, 2] = vectors[:, 1]
    skew[:, 1, 0] = vectors[:, 2]
    skew[:, 1, 2] = -vectors[:, 0]
    skew[:, 2, 0] = -vectors[:, 1]
    skew[:, 2, 1] = vectors[:, 0]
    return skew


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    angles = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axes = axis_angle / angles.clamp_min(1.0e-8)
    skew = _skew_symmetric(axes)
    eye = (
        torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
        .unsqueeze(0)
        .expand(axis_angle.shape[0], -1, -1)
    )
    rotation = eye + torch.sin(angles).unsqueeze(-1) * skew
    rotation = rotation + (1.0 - torch.cos(angles)).unsqueeze(-1) * (skew @ skew)
    small_angle = angles.squeeze(-1) < 1.0e-6
    if torch.any(small_angle):
        rotation[small_angle] = eye[small_angle] + _skew_symmetric(axis_angle[small_angle])
    return rotation


def _polynomial_fit_matrix(frame_times: Tensor, degree: int) -> Tensor:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if int(frame_times.numel()) < degree + 1:
        raise ValueError("need at least degree + 1 frames to fit a camera polynomial")
    times64 = frame_times.detach().cpu().to(dtype=torch.float64)
    vandermonde = torch.stack([times64.pow(k) for k in range(degree + 1)], dim=-1)
    normal = vandermonde.T @ vandermonde
    return torch.linalg.solve(normal, vandermonde.T).to(dtype=torch.float32)


class LearnableSE3CameraPath(torch.nn.Module):
    """Bounded per-frame SE3 residuals around a fixed world-to-camera path."""

    def __init__(
        self,
        base_k: Tensor,
        base_w2c: Tensor,
        frame_times: Tensor,
        *,
        degree: int,
        max_rotation_degrees: float,
        max_translation: float,
    ) -> None:
        super().__init__()
        if base_k.ndim != 3 or base_k.shape[1:] != (3, 3):
            raise ValueError("base_k must have shape [F,3,3]")
        if base_w2c.ndim != 3 or base_w2c.shape[1:] != (4, 4):
            raise ValueError("base_w2c must have shape [F,4,4]")
        if base_k.shape[0] != base_w2c.shape[0] or base_k.shape[0] != frame_times.numel():
            raise ValueError("base camera tensors and frame_times must have the same frame count")

        self.degree = int(degree)
        self.max_rotation_radians = math.radians(float(max_rotation_degrees))
        self.max_translation = float(max_translation)
        self.register_buffer("base_k", base_k.detach().clone(), persistent=False)
        self.register_buffer("base_w2c", base_w2c.detach().clone(), persistent=False)
        self.register_buffer("frame_times", frame_times.detach().clone(), persistent=False)
        self.register_buffer(
            "fit_matrix",
            _polynomial_fit_matrix(frame_times, degree).to(device=base_k.device, dtype=base_k.dtype),
            persistent=False,
        )
        self.raw_rotation = torch.nn.Parameter(torch.zeros((int(base_k.shape[0]), 3), dtype=base_k.dtype, device=base_k.device))
        self.raw_translation = torch.nn.Parameter(
            torch.zeros((int(base_k.shape[0]), 3), dtype=base_k.dtype, device=base_k.device)
        )

    def bounded_residuals(self) -> tuple[Tensor, Tensor]:
        rotation = torch.tanh(self.raw_rotation) * self.max_rotation_radians
        translation = torch.tanh(self.raw_translation) * self.max_translation
        return rotation, translation

    def camera_sequence(self) -> tuple[Tensor, Tensor]:
        rotation, translation = self.bounded_residuals()
        delta = torch.eye(4, dtype=rotation.dtype, device=rotation.device).unsqueeze(0).repeat(rotation.shape[0], 1, 1)
        delta[:, :3, :3] = axis_angle_to_matrix(rotation)
        delta[:, :3, 3] = translation
        return self.base_k, delta @ self.base_w2c

    def camera_path(self) -> CameraPathPolynomial:
        k_seq, w2c_seq = self.camera_sequence()
        p_seq = projection_matrices(k_seq, w2c_seq)
        coeff = (self.fit_matrix @ p_seq.reshape(int(p_seq.shape[0]), 12)).reshape(self.degree + 1, 3, 4)
        recon = evaluate_camera_polynomial(coeff, self.frame_times)
        denom = p_seq.reshape(int(p_seq.shape[0]), -1).norm(dim=1).clamp_min(1.0e-8)
        rel = (recon - p_seq).reshape(int(p_seq.shape[0]), -1).norm(dim=1) / denom
        return CameraPathPolynomial(
            p_coeff=coeff,
            frame_times=self.frame_times,
            fit_error=float(rel.max().detach().cpu()),
        )

    def regularization_loss(self) -> Tensor:
        rotation, translation = self.bounded_residuals()
        return rotation.square().mean() + translation.square().mean()

    def temporal_smoothness_loss(self) -> Tensor:
        rotation, translation = self.bounded_residuals()
        if int(rotation.shape[0]) < 2:
            return rotation.new_tensor(0.0)
        return (rotation[1:] - rotation[:-1]).square().mean() + (translation[1:] - translation[:-1]).square().mean()

    @torch.no_grad()
    def metrics(self) -> dict[str, float]:
        rotation, translation = self.bounded_residuals()
        return {
            "rotation_degrees_mean": float(torch.rad2deg(torch.linalg.norm(rotation, dim=-1)).mean().detach().cpu()),
            "rotation_degrees_max": float(torch.rad2deg(torch.linalg.norm(rotation, dim=-1)).max().detach().cpu()),
            "translation_mean": float(torch.linalg.norm(translation, dim=-1).mean().detach().cpu()),
            "translation_max": float(torch.linalg.norm(translation, dim=-1).max().detach().cpu()),
            "raw_rotation_grad_norm": 0.0
            if self.raw_rotation.grad is None
            else float(self.raw_rotation.grad.norm().detach().cpu()),
            "raw_translation_grad_norm": 0.0
            if self.raw_translation.grad is None
            else float(self.raw_translation.grad.norm().detach().cpu()),
        }


def _world_tubes_to_device(batch: WorldTubeBatch, device: torch.device) -> WorldTubeBatch:
    return WorldTubeBatch(
        x0=batch.x0.to(device),
        velocity=batch.velocity.to(device),
        t0=batch.t0.to(device),
        precision_xy=batch.precision_xy.to(device),
        lambda_t=batch.lambda_t.to(device),
        opacity=batch.opacity.to(device),
        color=batch.color.to(device),
    )


def _camera_path_to_device(path: CameraPathPolynomial, device: torch.device) -> CameraPathPolynomial:
    return CameraPathPolynomial(
        p_coeff=path.p_coeff.to(device),
        frame_times=path.frame_times.to(device),
        fit_error=path.fit_error,
    )


def _render_prt(
    batch: WorldTubeBatch,
    camera_path: CameraPathPolynomial,
    config: UVTRenderConfig,
    *,
    detach_footprint: bool,
) -> Tensor:
    projected = compile_projective_rational_tubes(batch, camera_path)
    lambda_uv = projected.lambda_uv.detach().contiguous() if detach_footprint else projected.lambda_uv
    return render_projective_rational_tubes_metal_direct_serial_backward(
        projected.h_coeff,
        lambda_uv,
        projected.lambda_t,
        projected.center_t,
        projected.opacity,
        projected.color,
        config,
        forward_mode="tiled",
        backward_mode="tile_pixel_atomic",
    )


def _render_prt_eval(
    batch: WorldTubeBatch,
    camera_path: CameraPathPolynomial,
    config: UVTRenderConfig,
    *,
    detach_footprint: bool,
):
    projected = compile_projective_rational_tubes(batch, camera_path)
    lambda_uv = projected.lambda_uv.detach().contiguous() if detach_footprint else projected.lambda_uv
    return render_projective_rational_tubes_tiled(
        projected.h_coeff,
        lambda_uv,
        projected.lambda_t,
        projected.center_t,
        projected.opacity,
        projected.color,
        config,
        return_aux=True,
    )


def _timed_render(call, *, device: torch.device, warmups: int, repeats: int) -> tuple[Any, dict[str, Any]]:
    result = None
    samples = []
    with torch.no_grad():
        for _ in range(warmups):
            result = call()
            _sync(device)
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            result = call()
            _sync(device)
            samples.append((time.perf_counter() - started) * 1000.0)
    if result is None:
        raise AssertionError("render did not run")
    return result, {
        "samples": samples,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "max_ms": max(samples),
        "warmups": warmups,
        "repeats": repeats,
    }


def _train_camera_only(
    *,
    batch: WorldTubeBatch,
    camera: LearnableSE3CameraPath,
    target: Tensor,
    config: UVTRenderConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(camera.parameters(), lr=args.camera_lr)
    losses = []
    logs = []
    initial_image = None
    final_image = None
    started = time.perf_counter()
    for step in range(args.camera_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image = _render_prt(
            batch,
            camera.camera_path(),
            config,
            detach_footprint=args.detach_footprint,
        )
        recon = (image - target).square().mean()
        loss = recon
        loss = loss + float(args.camera_reg_weight) * camera.regularization_loss()
        loss = loss + float(args.camera_temporal_weight) * camera.temporal_smoothness_loss()
        if step == 0:
            initial_image = image.detach()
        final_image = image.detach()
        if step < args.camera_steps:
            loss.backward()
            logs.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "recon_loss": float(recon.detach().cpu()),
                    "camera": camera.metrics(),
                }
            )
            optimizer.step()
            _sync(device)
        losses.append(float(recon.detach().cpu()))
    if initial_image is None or final_image is None:
        raise AssertionError("camera-only training did not render")
    gradient_checks = {
        "raw_rotation_grad_norm_max": max(item["camera"]["raw_rotation_grad_norm"] for item in logs),
        "raw_translation_grad_norm_max": max(item["camera"]["raw_translation_grad_norm"] for item in logs),
    }
    return {
        "name": "camera_only_prt",
        "steps": args.camera_steps,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / max(losses[0], 1.0e-12),
        "initial_psnr": _psnr(losses[0]),
        "final_psnr": _psnr(losses[-1]),
        "losses": losses,
        "logs": logs,
        "gradient_checks": gradient_checks,
        "camera": camera.metrics(),
        "initial_image": initial_image,
        "final_image": final_image,
    }


def _train_joint(
    *,
    gt_cpu: WorldTubeBatch,
    camera: LearnableSE3CameraPath,
    target: Tensor,
    holdout_target: Tensor,
    holdout_camera_path: CameraPathPolynomial,
    config: UVTRenderConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model = TrainableWorldTubeModel(
        gt_cpu,
        device=device,
        seed=args.seed + 100,
        geometry_noise=args.init_geometry_noise,
        velocity_noise=args.init_velocity_noise,
        color_noise=args.init_color_noise,
    )
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": args.world_lr},
            {"params": camera.parameters(), "lr": args.camera_lr},
        ]
    )
    losses = []
    logs = []
    started = time.perf_counter()
    for step in range(args.joint_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image = _render_prt(
            model.batch(),
            camera.camera_path(),
            config,
            detach_footprint=args.detach_footprint,
        )
        recon = (image - target).square().mean()
        loss = recon
        loss = loss + float(args.camera_reg_weight) * camera.regularization_loss()
        loss = loss + float(args.camera_temporal_weight) * camera.temporal_smoothness_loss()
        if step < args.joint_steps:
            loss.backward()
            logs.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "recon_loss": float(recon.detach().cpu()),
                    "camera": camera.metrics(),
                }
            )
            optimizer.step()
            _sync(device)
        losses.append(float(recon.detach().cpu()))

    gradient_checks = {
        "raw_rotation_grad_norm_max": max(item["camera"]["raw_rotation_grad_norm"] for item in logs),
        "raw_translation_grad_norm_max": max(item["camera"]["raw_translation_grad_norm"] for item in logs),
    }
    train_aux, train_timing = _timed_render(
        lambda: _render_prt_eval(
            model.batch(),
            camera.camera_path(),
            config,
            detach_footprint=args.detach_footprint,
        ),
        device=device,
        warmups=args.render_warmups,
        repeats=args.render_repeats,
    )
    holdout_aux, holdout_timing = _timed_render(
        lambda: _render_prt_eval(
            model.batch(),
            holdout_camera_path,
            config,
            detach_footprint=args.detach_footprint,
        ),
        device=device,
        warmups=args.render_warmups,
        repeats=args.render_repeats,
    )
    return {
        "name": "joint_world_and_camera_prt",
        "steps": args.joint_steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()) + sum(
            parameter.numel() for parameter in camera.parameters()
        ),
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / max(losses[0], 1.0e-12),
        "losses": losses,
        "logs": logs,
        "gradient_checks": gradient_checks,
        "camera": camera.metrics(),
        "train_metrics": _image_metrics(train_aux.image, target),
        "holdout_metrics": _image_metrics(holdout_aux.image, holdout_target),
        "train_render_benchmark_ms": train_timing,
        "holdout_render_benchmark_ms": holdout_timing,
        "max_tile_count": int(train_aux.tile_counts.max().detach().cpu()),
        "overflow_tile_count": int((train_aux.tile_overflow > 0).sum().detach().cpu()),
        "train_image": train_aux.image.detach(),
    }


def _save_sheet(path: Path, images: dict[str, Tensor]) -> None:
    from PIL import Image, ImageDraw

    tiles = []
    for label, tensor in images.items():
        frame = tensor.detach().cpu()[0].clamp(0.0, 1.0)
        rgb = (frame * 255.0).to(torch.uint8).numpy()
        tile = Image.fromarray(rgb)
        canvas = Image.new("RGB", (tile.width, tile.height + 16), (16, 16, 16))
        canvas.paste(tile, (0, 16))
        ImageDraw.Draw(canvas).text((2, 2), label, fill=(240, 240, 240))
        tiles.append(canvas)
    sheet = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)))
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "mps":
        raise ValueError("learnable camera PRT gate currently requires --device=mps")
    torch.manual_seed(args.seed)

    times_cpu = centered_frame_times(args.frames, device="cpu")
    true_k_cpu, true_w2c_cpu = _camera(
        args.frames,
        args.width,
        args.height,
        times_cpu,
        motion_scale=args.camera_motion_scale,
    )
    base_w2c_cpu = true_w2c_cpu.clone()
    base_w2c_cpu[:, 0, 3] += float(args.base_x_offset)
    base_w2c_cpu[:, 1, 3] += float(args.base_y_offset)
    base_w2c_cpu[:, 2, 3] += float(args.base_z_offset)
    holdout_k_cpu, holdout_w2c_cpu = _camera(
        args.frames,
        args.width,
        args.height,
        times_cpu,
        motion_scale=args.holdout_camera_motion_scale,
    )
    holdout_w2c_cpu = holdout_w2c_cpu.clone()
    holdout_w2c_cpu[:, 0, 3] += float(args.holdout_x_offset)

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
    gt = _world_tubes_to_device(gt_cpu, device)
    times = times_cpu.to(device)
    true_k = true_k_cpu.to(device)
    true_w2c = true_w2c_cpu.to(device)
    base_k = true_k_cpu.to(device)
    base_w2c = base_w2c_cpu.to(device)
    holdout_k = holdout_k_cpu.to(device)
    holdout_w2c = holdout_w2c_cpu.to(device)
    true_camera_path = _camera_path_to_device(
        fit_camera_path_polynomial(
            true_k_cpu,
            true_w2c_cpu,
            degree=args.camera_poly_degree,
            frame_times=times_cpu,
        ),
        device,
    )
    base_camera_path = _camera_path_to_device(
        fit_camera_path_polynomial(
            base_k.detach().cpu(),
            base_w2c.detach().cpu(),
            degree=args.camera_poly_degree,
            frame_times=times_cpu,
        ),
        device,
    )
    holdout_camera_path = _camera_path_to_device(
        fit_camera_path_polynomial(
            holdout_k_cpu,
            holdout_w2c_cpu,
            degree=args.camera_poly_degree,
            frame_times=times_cpu,
        ),
        device,
    )

    true_projected = compile_projective_rational_tubes(gt, true_camera_path)
    holdout_projected = compile_projective_rational_tubes(gt, holdout_camera_path)
    with torch.no_grad():
        target = dense_render_direct_world_tubes(
            gt,
            true_k,
            true_w2c,
            times,
            true_projected.lambda_uv.detach(),
            height=args.height,
            width=args.width,
        ).detach()
        holdout_target = dense_render_direct_world_tubes(
            gt,
            holdout_k,
            holdout_w2c,
            times,
            holdout_projected.lambda_uv.detach(),
            height=args.height,
            width=args.width,
        ).detach()
        wrong_camera_aux = _render_prt_eval(gt, base_camera_path, config, detach_footprint=args.detach_footprint)
        oracle_camera_aux = _render_prt_eval(gt, true_camera_path, config, detach_footprint=args.detach_footprint)

    camera_only = LearnableSE3CameraPath(
        base_k,
        base_w2c,
        times,
        degree=args.camera_poly_degree,
        max_rotation_degrees=args.max_rotation_degrees,
        max_translation=args.max_translation,
    ).to(device)
    camera_only_report = _train_camera_only(
        batch=gt,
        camera=camera_only,
        target=target,
        config=config,
        args=args,
        device=device,
    )

    joint_camera = LearnableSE3CameraPath(
        base_k,
        base_w2c,
        times,
        degree=args.camera_poly_degree,
        max_rotation_degrees=args.max_rotation_degrees,
        max_translation=args.max_translation,
    ).to(device)
    joint_report = _train_joint(
        gt_cpu=gt_cpu,
        camera=joint_camera,
        target=target,
        holdout_target=holdout_target,
        holdout_camera_path=holdout_camera_path,
        config=config,
        args=args,
        device=device,
    )

    wrong_mse = float((wrong_camera_aux.image - target).square().mean().detach().cpu())
    oracle_mse = float((oracle_camera_aux.image - target).square().mean().detach().cpu())
    report = {
        "name": "projective_rational_learnable_camera_gate",
        "note": (
            "Synthetic PRT gate proving bounded per-frame SE3 camera residuals can feed the "
            "projective-rational compiler, receive gradients through Metal PRT autograd, and improve "
            "a deliberately perturbed train camera. The heldout path remains fixed/oracle."
        ),
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "tube_count": args.tube_count,
        "seed": args.seed,
        "device": str(device),
        "camera_poly_degree": args.camera_poly_degree,
        "camera_motion_scale": args.camera_motion_scale,
        "holdout_camera_motion_scale": args.holdout_camera_motion_scale,
        "base_camera_offsets": {
            "x": args.base_x_offset,
            "y": args.base_y_offset,
            "z": args.base_z_offset,
        },
        "max_rotation_degrees": args.max_rotation_degrees,
        "max_translation": args.max_translation,
        "detach_footprint": bool(args.detach_footprint),
        "tile_config_key": tile_config.key,
        "tile_config": tile_config.as_dict(),
        "wrong_camera": {
            "mse": wrong_mse,
            "psnr": _psnr(wrong_mse),
            "camera_fit_error": base_camera_path.fit_error,
        },
        "oracle_camera": {
            "mse": oracle_mse,
            "psnr": _psnr(oracle_mse),
            "camera_fit_error": true_camera_path.fit_error,
        },
        "camera_only": {
            key: value
            for key, value in camera_only_report.items()
            if key not in {"initial_image", "final_image"}
        },
        "joint_world_and_camera": {
            key: value
            for key, value in joint_report.items()
            if key != "train_image"
        },
    }
    report["pass"] = (
        camera_only_report["final_loss"] < 0.75 * camera_only_report["initial_loss"]
        and camera_only_report["gradient_checks"]["raw_translation_grad_norm_max"] > 0.0
        and joint_report["final_loss"] < joint_report["initial_loss"]
        and joint_report["gradient_checks"]["raw_translation_grad_norm_max"] > 0.0
        and int(joint_report["overflow_tile_count"]) == 0
    )

    if args.out_png is not None:
        _save_sheet(
            args.out_png,
            {
                "target": target,
                "wrong": wrong_camera_aux.image,
                "camera_fit": camera_only_report["final_image"],
                "joint_fit": joint_report["train_image"],
            },
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--tube-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--camera-motion-scale", type=float, default=3.0)
    parser.add_argument("--holdout-camera-motion-scale", type=float, default=2.0)
    parser.add_argument("--holdout-x-offset", type=float, default=0.08)
    parser.add_argument("--base-x-offset", type=float, default=0.04)
    parser.add_argument("--base-y-offset", type=float, default=-0.016)
    parser.add_argument("--base-z-offset", type=float, default=0.02)
    parser.add_argument("--camera-poly-degree", type=int, default=2)
    parser.add_argument("--max-rotation-degrees", type=float, default=3.0)
    parser.add_argument("--max-translation", type=float, default=0.08)
    parser.add_argument("--camera-steps", type=int, default=40)
    parser.add_argument("--joint-steps", type=int, default=40)
    parser.add_argument("--camera-lr", type=float, default=0.08)
    parser.add_argument("--world-lr", type=float, default=0.02)
    parser.add_argument("--camera-reg-weight", type=float, default=0.001)
    parser.add_argument("--camera-temporal-weight", type=float, default=0.001)
    parser.add_argument("--init-geometry-noise", type=float, default=0.02)
    parser.add_argument("--init-velocity-noise", type=float, default=0.001)
    parser.add_argument("--init-color-noise", type=float, default=0.20)
    parser.add_argument("--tile-config", default="auto", help="'auto' or an explicit config like 8x8x2:128")
    parser.add_argument("--detach-footprint", action="store_true")
    parser.add_argument("--render-warmups", type=int, default=1)
    parser.add_argument("--render-repeats", type=int, default=3)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-png", type=Path)
    args = parser.parse_args()

    report = run_gate(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
