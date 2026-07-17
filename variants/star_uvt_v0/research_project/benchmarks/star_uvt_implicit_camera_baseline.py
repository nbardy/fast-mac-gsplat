from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_dynaworld_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "train" / "sequence_data.py").exists():
            return parent
    raise FileNotFoundError("Could not find dynaworld root from STAR-UVT variant")


DYNAWORLD_ROOT = find_dynaworld_root()
TRAIN_SRC = DYNAWORLD_ROOT / "src" / "train"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from config_utils import load_config_file, serialize_config_value  # noqa: E402
from gs_models.implicit_camera import (  # noqa: E402
    PathCameraHead,
    build_global_camera_head,
    compose_camera_with_se3_delta,
)
from gs_models.time_conditioning import build_time_projector  # noqa: E402
from objective.loss import reconstruction_loss_per_image, reconstruction_loss_spec_from_mapping  # noqa: E402
from pipeline.losses import build_camera_loss  # noqa: E402
from runtime_types import CameraState  # noqa: E402
from sequence_data import load_uncalibrated_sequence  # noqa: E402
from torch_gsplat_bridge_star_uvt import UVTRenderConfig  # noqa: E402

try:
    from research_project.trainer_harness.model import dense_differentiable_render_uvt_tubes
    from research_project.trainer_harness.tile_metal_autograd import render_uvt_tubes_metal_tile_backward
    from research_project.trainer_harness.world_tube import (
        PinholeCameraMotion,
        WorldTubeBatch,
        project_world_tubes_pinhole_projective_motion,
    )
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = ROOT / "research_project" / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from model import dense_differentiable_render_uvt_tubes
    from tile_metal_autograd import render_uvt_tubes_metal_tile_backward
    from world_tube import PinholeCameraMotion, WorldTubeBatch, project_world_tubes_pinhole_projective_motion


DEFAULT_BASELINE_CONFIG = (
    DYNAWORLD_ROOT
    / "src"
    / "train_configs"
    / "local_mac_compare_free_linear_time_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
)


def resolve_dynaworld_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return DYNAWORLD_ROOT / path


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default) if raw is None or raw == "" else int(raw)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def inverse_sigmoid(value: float) -> float:
    clamped = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(clamped) - math.log1p(-clamped)


def inverse_tanh_values(values: Tensor) -> Tensor:
    clamped = values.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
    return 0.5 * (torch.log1p(clamped) - torch.log1p(-clamped))


def inv_softplus(values: Tensor) -> Tensor:
    clamped = values.clamp_min(1.0e-8)
    return clamped + torch.log(-torch.expm1(-clamped))


def mse_to_psnr(mse: float) -> float:
    return -10.0 * math.log10(max(float(mse), 1.0e-12))


def scalar_tensor(value: object, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype).reshape(())
    return torch.tensor(float(value), device=device, dtype=dtype)


def camera_to_K(camera: object, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    fx = scalar_tensor(getattr(camera, "fx"), device=device, dtype=dtype)
    fy = scalar_tensor(getattr(camera, "fy"), device=device, dtype=dtype)
    cx = scalar_tensor(getattr(camera, "cx"), device=device, dtype=dtype)
    cy = scalar_tensor(getattr(camera, "cy"), device=device, dtype=dtype)
    zero = torch.zeros((), dtype=dtype, device=device)
    one = torch.ones((), dtype=dtype, device=device)
    return torch.stack(
        (
            torch.stack((fx, zero, cx)),
            torch.stack((zero, fy, cy)),
            torch.stack((zero, zero, one)),
        )
    )


@dataclass(frozen=True)
class CameraSequence:
    cameras: tuple[object, ...]
    state: CameraState
    K: Tensor
    w2c: Tensor


class StarUVTFreeLinearImplicitCamera(nn.Module):
    """STAR-UVT fork of the direct free-linear implicit-camera GS baseline.

    Mirrors `free_linear_time_splats` where possible: one direct learnable bank,
    linear xyz motion in normalized video time, fixed RGB, fixed spatial size,
    and the same global/path implicit camera heads.
    """

    def __init__(
        self,
        *,
        tube_count: int,
        image_size: int,
        full_frame_count: int,
        feat_dim: int,
        xy_extent: float,
        z_min: float,
        z_max: float,
        scale_init: float,
        scale_init_log_jitter: float,
        opacity_init: float,
        position_init_extent_coverage: float,
        rgb_init_min: float,
        rgb_init_max: float,
        query_token_init_std: float,
        free_velocity_extent: float,
        free_velocity_init_std: float,
        free_time_center: float,
        init_lambda_t: float,
        min_precision_xy: float,
        min_lambda_t: float,
        camera_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        if tube_count < 1:
            raise ValueError("tube_count must be positive")
        if full_frame_count < 1:
            raise ValueError("full_frame_count must be positive")
        if scale_init <= 0.0:
            raise ValueError("scale_init must be positive")
        if init_lambda_t <= min_lambda_t:
            raise ValueError("init_lambda_t must be greater than min_lambda_t")
        self.tube_count = int(tube_count)
        self.image_size = int(image_size)
        self.full_frame_count = int(full_frame_count)
        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_extent = float(z_max) - float(z_min)
        self.free_velocity_extent = float(free_velocity_extent)
        self.free_time_center = float(free_time_center)
        self.min_precision_xy = float(min_precision_xy)
        self.min_lambda_t = float(min_lambda_t)
        if self.z_extent <= 0.0:
            raise ValueError("z_max must be greater than z_min")

        coverage = float(position_init_extent_coverage)
        if coverage > 0.0:
            z_margin = 0.5 * (1.0 - coverage)
            raw_xy = inverse_tanh_values(torch.empty(tube_count, 2).uniform_(-coverage, coverage))
            raw_z = torch.logit(torch.empty(tube_count, 1).uniform_(z_margin, 1.0 - z_margin), eps=1.0e-6)
            raw_xyz = torch.cat((raw_xy, raw_z), dim=-1)
        else:
            raw_xyz = torch.zeros(tube_count, 3)
        raw_velocity = torch.zeros(tube_count, 3)
        if free_velocity_init_std > 0.0:
            raw_velocity.normal_(mean=0.0, std=float(free_velocity_init_std))

        raw_color = torch.empty(tube_count, 3).uniform_(float(rgb_init_min), float(rgb_init_max))
        precision_init = torch.full((tube_count, 2), 1.0 / (float(scale_init) * float(scale_init)))
        if scale_init_log_jitter > 0.0:
            precision_init = precision_init * torch.exp(
                torch.empty(tube_count, 2).uniform_(
                    -2.0 * float(scale_init_log_jitter),
                    2.0 * float(scale_init_log_jitter),
                )
            )
        lambda_init = torch.full((tube_count,), float(init_lambda_t))

        self.raw_xyz = nn.Parameter(raw_xyz)
        self.raw_velocity = nn.Parameter(raw_velocity)
        self.raw_precision_xy = nn.Parameter(inv_softplus(precision_init - self.min_precision_xy))
        self.raw_lambda_t = nn.Parameter(inv_softplus(lambda_init - self.min_lambda_t))
        self.raw_opacity = nn.Parameter(torch.full((tube_count,), inverse_sigmoid(float(opacity_init) / 0.99)))
        self.raw_color = nn.Parameter(torch.logit(raw_color, eps=1.0e-6))

        self.global_camera_token = nn.Parameter(torch.randn(feat_dim) * float(query_token_init_std))
        self.path_camera_token = nn.Parameter(torch.randn(feat_dim) * float(query_token_init_std))
        self.path_time_proj = build_time_projector(1, feat_dim)
        self.global_camera_head = build_global_camera_head(
            str(camera_cfg["global_head"]).lower(),
            feat_dim=feat_dim,
            lens_model=str(camera_cfg["lens_model"]).lower(),
            base_fov_degrees=float(camera_cfg["base_fov_degrees"]),
            base_radius=float(camera_cfg["base_radius"]),
            max_fov_delta_degrees=float(camera_cfg["max_fov_delta_degrees"]),
            max_radius_scale=float(camera_cfg["max_radius_scale"]),
            max_aspect_log_delta=float(camera_cfg.get("max_aspect_log_delta", 0.0)),
            max_principal_point_delta=float(camera_cfg.get("max_principal_point_delta", 0.0)),
            distortion_max_abs=float(camera_cfg.get("distortion_max_abs", 0.0)),
            base_distortion=camera_cfg.get("base_distortion"),
        )
        self.path_camera_head = PathCameraHead(
            feat_dim=feat_dim,
            max_rotation_degrees=float(camera_cfg["max_rotation_degrees"]),
            max_translation_ratio=float(camera_cfg["max_translation_ratio"]),
        )

    def camera_for_time(self, decode_time: Tensor) -> tuple[object, CameraState]:
        decode_time = decode_time.reshape(1, 1).to(device=self.raw_xyz.device, dtype=self.raw_xyz.dtype)
        path_token = self.path_camera_token + self.path_time_proj(decode_time).squeeze(0)
        base_camera, base_state = self.global_camera_head(self.global_camera_token, image_size=self.image_size)
        rotation_delta, translation_delta, path_residuals = self.path_camera_head(
            path_token.unsqueeze(0),
            base_radius=base_state["radius"],
        )
        camera = compose_camera_with_se3_delta(base_camera, rotation_delta, translation_delta)[0]
        return camera, CameraState(
            fov_degrees=base_state["fov_degrees"],
            radius=base_state["radius"],
            global_residuals=base_state["global_residuals"],
            rotation_delta=rotation_delta,
            translation_delta=translation_delta,
            path_residuals=path_residuals,
        )

    def camera_sequence(self, decode_times: Tensor) -> CameraSequence:
        decoded = [self.camera_for_time(decode_times[index]) for index in range(int(decode_times.shape[0]))]
        cameras = tuple(item[0] for item in decoded)
        states = [item[1] for item in decoded]
        camera_state = CameraState(
            fov_degrees=torch.stack([state.fov_degrees for state in states]).mean(),
            radius=torch.stack([state.radius for state in states]).mean(),
            global_residuals=torch.stack([state.global_residuals for state in states]).mean(dim=0),
            rotation_delta=torch.cat([state.rotation_delta for state in states], dim=0),
            translation_delta=torch.cat([state.translation_delta for state in states], dim=0),
            path_residuals=torch.cat([state.path_residuals for state in states if state.path_residuals is not None], dim=0),
        )
        K_seq = torch.stack(
            [
                camera_to_K(camera, device=self.raw_xyz.device, dtype=self.raw_xyz.dtype)
                for camera in cameras
            ],
            dim=0,
        )
        w2c_seq = torch.stack(
            [torch.linalg.inv(camera.camera_to_world.to(device=self.raw_xyz.device, dtype=self.raw_xyz.dtype)) for camera in cameras],
            dim=0,
        )
        return CameraSequence(cameras=cameras, state=camera_state, K=K_seq, w2c=w2c_seq)

    def world_batch(self, *, frame_start: int, window_frames: int) -> WorldTubeBatch:
        x0 = torch.cat(
            (
                torch.tanh(self.raw_xyz[:, :2]) * self.xy_extent,
                torch.sigmoid(self.raw_xyz[:, 2:3]) * self.z_extent + self.z_min,
            ),
            dim=-1,
        )
        velocity_normalized = torch.tanh(self.raw_velocity) * self.free_velocity_extent
        velocity_per_frame = velocity_normalized / float(max(self.full_frame_count - 1, 1))
        local_t0 = self.free_time_center * float(max(self.full_frame_count - 1, 1))
        local_t0 = local_t0 - float(frame_start) - 0.5 * float(window_frames - 1)
        return WorldTubeBatch(
            x0=x0,
            velocity=velocity_per_frame,
            t0=torch.full((self.tube_count,), local_t0, dtype=x0.dtype, device=x0.device),
            precision_xy=F.softplus(self.raw_precision_xy) + self.min_precision_xy,
            lambda_t=F.softplus(self.raw_lambda_t) + self.min_lambda_t,
            opacity=torch.sigmoid(self.raw_opacity) * 0.99,
            color=torch.sigmoid(self.raw_color),
        )

    def camera_motion_chart(
        self,
        *,
        camera_sequence: CameraSequence,
        chart_index: int,
        chart_time: float,
    ) -> PinholeCameraMotion:
        frames = int(camera_sequence.K.shape[0])
        prev_index = max(0, int(chart_index) - 1)
        next_index = min(frames - 1, int(chart_index) + 1)
        if next_index == prev_index:
            K_dot = torch.zeros_like(camera_sequence.K[chart_index])
            w2c_dot = torch.zeros_like(camera_sequence.w2c[chart_index])
        else:
            delta = float(next_index - prev_index)
            K_dot = (camera_sequence.K[next_index] - camera_sequence.K[prev_index]) / delta
            w2c_dot = (camera_sequence.w2c[next_index] - camera_sequence.w2c[prev_index]) / delta
        K = camera_sequence.K[chart_index]
        return PinholeCameraMotion(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            fx_dot=K_dot[0, 0],
            fy_dot=K_dot[1, 1],
            cx_dot=K_dot[0, 2],
            cy_dot=K_dot[1, 2],
            world_to_camera=camera_sequence.w2c[chart_index],
            world_to_camera_dot=w2c_dot,
            chart_time=chart_time,
        )

    def project(self, *, decode_times: Tensor, frame_start: int, config: UVTRenderConfig) -> tuple[Any, CameraState]:
        camera_sequence = self.camera_sequence(decode_times)
        chart_index = int(round(0.5 * float(config.frames - 1)))
        chart_time = 0.0
        motion = self.camera_motion_chart(
            camera_sequence=camera_sequence,
            chart_index=chart_index,
            chart_time=chart_time,
        )
        projected = project_world_tubes_pinhole_projective_motion(
            self.world_batch(frame_start=frame_start, window_frames=int(config.frames)),
            motion,
            config,
        )
        return projected, camera_sequence.state


def render_projected(projected: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], config: UVTRenderConfig, backend: str) -> Tensor:
    ma, q_uvt, depth0, depth_beta, opacity, color = projected
    if backend == "dense":
        return dense_differentiable_render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config)
    if backend == "metal_tile":
        return render_uvt_tubes_metal_tile_backward(ma, q_uvt, depth0, depth_beta, opacity, color, config)
    raise ValueError("backend must be one of: dense, metal_tile")


def defaulted_losses(loss_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(loss_cfg)
    cfg.setdefault("type", "standard_gs")
    cfg.setdefault("l1_weight", 0.8)
    cfg.setdefault("dssim_weight", 0.2)
    cfg.setdefault("mse_weight", 0.2)
    cfg.setdefault("ssim_window_size", 11)
    cfg.setdefault("ssim_c1", 0.0001)
    cfg.setdefault("ssim_c2", 0.0009)
    cfg.setdefault("dssim_backend", "torch")
    cfg.setdefault("camera_motion_weight", 0.0)
    cfg.setdefault("camera_temporal_weight", 0.0)
    cfg.setdefault("camera_global_weight", 0.0)
    cfg.setdefault("background", {"train_mode": "fixed", "eval_mode": "fixed", "fixed_rgb": (1.0, 1.0, 1.0)})
    return cfg


def load_clip(cfg: dict[str, Any], *, render_size: int, max_source_frames: int, device: torch.device) -> tuple[Tensor, int, float, Path | None]:
    data_cfg = cfg["data"]
    sequence = load_uncalibrated_sequence(
        sequence_dir=resolve_dynaworld_path(data_cfg["sequence_dir"]) or DYNAWORLD_ROOT,
        frames_dir=resolve_dynaworld_path(data_cfg.get("frames_dir")),
        video_path=resolve_dynaworld_path(data_cfg.get("video_path")),
        target_size=render_size,
        max_frames=max_source_frames,
        frame_source=data_cfg.get("frame_source", "explicit_video"),
        device=device,
    )
    return sequence.frames, sequence.frame_count, float(sequence.video_fps), sequence.source_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    cfg = load_config_file(args.config)
    model_cfg = cfg["model"]
    camera_cfg = cfg["camera"]
    render_cfg = cfg["render"]
    train_cfg = cfg["train"]
    loss_cfg = defaulted_losses(cfg["losses"])

    render_size = int(args.render_size or render_cfg["render_size"])
    train_frames = int(args.train_frames or model_cfg["train_frame_count"])
    steps = int(args.steps if args.steps is not None else train_cfg["steps"])
    tube_count = int(args.tube_count or int(model_cfg["tokens"]) * int(model_cfg["gaussians_per_token"]))
    max_source_frames = int(args.max_source_frames)
    source_frames, full_frame_count, video_fps, source_path = load_clip(
        cfg,
        render_size=render_size,
        max_source_frames=max_source_frames,
        device=device,
    )
    frame_start = int(args.frame_start)
    if frame_start < 0 or frame_start + train_frames > full_frame_count:
        raise ValueError(
            f"frame window [{frame_start}, {frame_start + train_frames}) exceeds loaded frame count {full_frame_count}"
        )
    target = source_frames[frame_start : frame_start + train_frames].contiguous()
    frame_indices = torch.arange(frame_start, frame_start + train_frames, device=device, dtype=torch.float32)
    decode_times = frame_indices / float(max(full_frame_count - 1, 1))

    model = StarUVTFreeLinearImplicitCamera(
        tube_count=tube_count,
        image_size=render_size,
        full_frame_count=full_frame_count,
        feat_dim=int(model_cfg["model_dim"]),
        xy_extent=float(model_cfg.get("xy_extent") or model_cfg["scene_extent"]),
        z_min=float(model_cfg.get("z_min", -float(model_cfg["scene_extent"]))),
        z_max=float(model_cfg.get("z_max", float(model_cfg["scene_extent"]))),
        scale_init=float(model_cfg["scale_init"]),
        scale_init_log_jitter=float(model_cfg.get("scale_init_log_jitter", 0.0)),
        opacity_init=float(model_cfg.get("opacity_init", 0.1)),
        position_init_extent_coverage=float(model_cfg.get("position_init_extent_coverage", 0.0)),
        rgb_init_min=float(model_cfg.get("rgb_init_min", 0.0)),
        rgb_init_max=float(model_cfg.get("rgb_init_max", 1.0)),
        query_token_init_std=float(model_cfg.get("query_token_init_std", 0.02)),
        free_velocity_extent=float(model_cfg.get("free_velocity_extent", 1.0)),
        free_velocity_init_std=float(model_cfg.get("free_velocity_init_std", 0.0)),
        free_time_center=float(model_cfg.get("free_time_center", 0.5)),
        init_lambda_t=float(args.init_lambda_t),
        min_precision_xy=float(args.min_precision_xy),
        min_lambda_t=float(args.min_lambda_t),
        camera_cfg=camera_cfg,
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr if args.lr is not None else train_cfg["lr"]))
    loss_spec = reconstruction_loss_spec_from_mapping(loss_cfg)
    background = tuple(float(value) for value in render_cfg.get("fast_mac", {}).get("background", (1.0, 1.0, 1.0)))
    tile_size = (
        int(args.tile_size)
        if args.tile_size is not None
        else env_int("STAR_UVT_TILE_X", 8)
        if args.backend == "metal_tile"
        else int(render_cfg.get("tile_size", 16))
    )
    tile_t = int(args.tile_t) if args.tile_t is not None else env_int("STAR_UVT_TILE_T", 2)
    tile_capacity = (
        int(args.tile_capacity) if args.tile_capacity is not None else env_int("STAR_UVT_TILE_CAPACITY", 128)
    )
    uvt_config = UVTRenderConfig(
        height=render_size,
        width=render_size,
        frames=train_frames,
        tile_x=tile_size,
        tile_y=tile_size,
        tile_t=tile_t,
        tile_capacity=tile_capacity,
        alpha_threshold=float(render_cfg.get("alpha_threshold", 1.0 / 255.0)),
        transmittance_threshold=float(render_cfg.get("fast_mac", {}).get("transmittance_threshold", 1.0e-4)),
        background=background,
    )

    logs: list[dict[str, float | int]] = []
    started = time.perf_counter()
    final_render = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        synchronize_device(device)
        step_started = time.perf_counter()
        projected, camera_state = model.project(decode_times=decode_times, frame_start=frame_start, config=uvt_config)
        rendered = render_projected(projected, uvt_config, args.backend)
        rendered_nchw = rendered.permute(0, 3, 1, 2).contiguous()
        recon_per_image = reconstruction_loss_per_image(rendered_nchw, target, loss_spec)
        recon_loss = recon_per_image.mean()
        camera_loss, camera_motion_loss, camera_temporal_loss, camera_global_loss = build_camera_loss(
            decode_times.reshape(1, -1),
            camera_state,
            loss_cfg,
        )
        loss = recon_loss + camera_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        synchronize_device(device)
        elapsed = time.perf_counter() - step_started
        final_render = rendered_nchw.detach()
        if step == 0 or (step + 1) % int(args.log_every) == 0 or step + 1 == steps:
            mse = float((final_render - target).square().mean().detach().cpu())
            logs.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "recon_loss": float(recon_loss.detach().cpu()),
                    "camera_loss": float(camera_loss.detach().cpu()),
                    "camera_motion_loss": float(camera_motion_loss.detach().cpu()),
                    "camera_temporal_loss": float(camera_temporal_loss.detach().cpu()),
                    "camera_global_loss": float(camera_global_loss.detach().cpu()),
                    "psnr": mse_to_psnr(mse),
                    "step_elapsed_s": elapsed,
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
            )

    if final_render is None:
        raise RuntimeError("training loop produced no render")
    final_mse = float((final_render - target).square().mean().detach().cpu())
    global_grad = None if model.global_camera_token.grad is None else float(model.global_camera_token.grad.norm().detach().cpu())
    path_grad = None if model.path_camera_token.grad is None else float(model.path_camera_token.grad.norm().detach().cpu())
    summary: dict[str, Any] = {
        "source_baseline_config": str(Path(args.config).resolve()),
        "source_path": None if source_path is None else str(source_path),
        "status": "ok",
        "method": "star_uvt_free_linear_time_splats_implicit_camera",
        "backend": args.backend,
        "device": str(device),
        "render_size": render_size,
        "train_frames": train_frames,
        "full_frame_count": full_frame_count,
        "frame_start": frame_start,
        "tube_count": tube_count,
        "steps": steps,
        "final_mse": final_mse,
        "final_psnr": mse_to_psnr(final_mse),
        "elapsed_s": time.perf_counter() - started,
        "camera_global_token_grad_norm": global_grad,
        "camera_path_token_grad_norm": path_grad,
        "logs": logs,
        "config": serialize_config_value(cfg),
        "notes": {
            "forked_from": "free_linear_time_splats implicit-camera GS baseline",
            "matched": [
                "direct learnable primitive bank",
                "linear xyz motion in normalized video time",
                "fixed RGB over time",
                "same global/path implicit camera heads",
                "same camera regularization weights",
                "same reconstruction loss parser",
            ],
            "star_specific": [
                "STAR uses one UVT world tube per primitive instead of per-frame 3D Gaussian raster calls",
                "opacity is fixed per tube; temporal support is controlled by lambda_t",
                "spatial footprint is fronto-parallel precision_xy, not full 3DGS anisotropic scale/rotation",
            ],
        },
    }
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="STAR-UVT implicit-camera fork of the free-linear 3DGS baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=("dense", "metal_tile"), default="dense")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--render-size", type=int, default=None)
    parser.add_argument("--train-frames", type=int, default=None)
    parser.add_argument("--tube-count", type=int, default=None)
    parser.add_argument("--max-source-frames", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--init-lambda-t", type=float, default=0.08)
    parser.add_argument("--min-precision-xy", type=float, default=1.0e-4)
    parser.add_argument("--min-lambda-t", type=float, default=1.0e-4)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-t", type=int, default=None)
    parser.add_argument("--tile-capacity", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    summary = run(args)
    printable = {key: value for key, value in summary.items() if key not in {"config", "logs"}}
    printable["last_log"] = summary["logs"][-1] if summary["logs"] else None
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
