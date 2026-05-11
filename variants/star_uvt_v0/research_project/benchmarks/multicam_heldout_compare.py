from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_dynaworld_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "train" / "multicam_video_data.py").exists():
            return parent
    raise FileNotFoundError("Could not find dynaworld root from STAR-UVT variant")


DYNAWORLD_ROOT = find_dynaworld_root()
TRAIN_SRC = DYNAWORLD_ROOT / "src" / "train"
GAUGE_EXPERIMENTS = DYNAWORLD_ROOT / "research_experiments" / "gauge_fields"
for path in (TRAIN_SRC, GAUGE_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig, render_uvt_tubes  # noqa: E402
from camera import CameraSpec  # noqa: E402
from config_utils import load_config_file, serialize_config_value  # noqa: E402
from common import prefix_metrics, robust_l1, save_preview_strip, save_side_by_side_mp4, video_metrics, write_json  # noqa: E402
from multicam_video_data import load_multicam_video_bundle  # noqa: E402
from renderers.projection import project_points_camera  # noqa: E402
from train_splat_baseline import (  # noqa: E402
    FreeDynamic3DGS,
    SplatRenderConfig,
    camera_from_K_w2c,
    initialize_material_points_from_first_frame,
    render_gaussian_frame,
    render_splat_sequence,
    select_K_for_view_time,
    select_w2c_for_view_time,
)

try:
    from research_project.trainer_harness.model import dense_differentiable_render_uvt_tubes
    from research_project.trainer_harness.tile_metal_autograd import render_uvt_tubes_metal_tile_backward
    from research_project.trainer_harness.world_tube import (
        PinholeCamera,
        WorldTubeBatch,
        project_world_tubes_from_pixel_jacobian,
        project_world_tubes_pinhole,
    )
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = ROOT / "research_project" / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from model import dense_differentiable_render_uvt_tubes
    from tile_metal_autograd import render_uvt_tubes_metal_tile_backward
    from world_tube import PinholeCamera, WorldTubeBatch, project_world_tubes_from_pixel_jacobian, project_world_tubes_pinhole


DEFAULT_BASELINE_CONFIG = (
    DYNAWORLD_ROOT
    / "src"
    / "train_configs"
    / "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alphaab_alpha1_128.jsonc"
)
TRAIN_SCHEDULE_CHOICES = (
    "random",
    "cycle",
    "shuffled_cycle",
    "reshuffled_cycle",
    "phase_rotated_cycle",
    "view_shuffled_cycle",
)


def resolve_dynaworld_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return DYNAWORLD_ROOT / value


def resolve_variant_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return ROOT / value


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def scalar_from_tensor(value: Tensor) -> float:
    return float(value.detach().cpu())


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def render_time_metrics(train_times: list[float], heldout_times: list[float]) -> dict[str, float | int]:
    all_times = train_times + heldout_times
    if not all_times:
        return {
            "eval_render_sequence_count": 0,
            "eval_render_only_elapsed_s": 0.0,
            "eval_render_mean_sequence_s": 0.0,
            "eval_render_max_sequence_s": 0.0,
            "eval_train_render_only_elapsed_s": 0.0,
            "eval_heldout_render_only_elapsed_s": 0.0,
        }
    return {
        "eval_render_sequence_count": len(all_times),
        "eval_render_only_elapsed_s": sum(all_times),
        "eval_render_mean_sequence_s": sum(all_times) / float(len(all_times)),
        "eval_render_max_sequence_s": max(all_times),
        "eval_train_render_only_elapsed_s": sum(train_times),
        "eval_heldout_render_only_elapsed_s": sum(heldout_times),
    }


def subset_video_metrics(rendered: Tensor, target: Tensor, frame_indices: list[int]) -> dict[str, float]:
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    indices = torch.tensor(frame_indices, dtype=torch.long, device=rendered.device)
    return video_metrics(rendered.index_select(0, indices), target.index_select(0, indices))


def downsampled_robust_l1(rendered: Tensor, target: Tensor, factor: int) -> Tensor:
    if factor < 1:
        raise ValueError("multiscale loss factor must be positive")
    if factor == 1:
        return robust_l1(rendered - target)
    if rendered.shape != target.shape:
        raise ValueError(f"multiscale loss shape mismatch: {tuple(rendered.shape)} vs {tuple(target.shape)}")
    if rendered.ndim == 3:
        rendered_nchw = rendered.permute(2, 0, 1).unsqueeze(0)
        target_nchw = target.permute(2, 0, 1).unsqueeze(0)
    elif rendered.ndim == 4:
        rendered_nchw = rendered.permute(0, 3, 1, 2)
        target_nchw = target.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"multiscale loss expects HWC or THWC tensors, got ndim={rendered.ndim}")
    height = int(rendered_nchw.shape[-2])
    width = int(rendered_nchw.shape[-1])
    effective_factor = min(factor, height, width)
    pooled_h = max(1, height // effective_factor)
    pooled_w = max(1, width // effective_factor)
    crop_h = pooled_h * effective_factor
    crop_w = pooled_w * effective_factor
    rendered_nchw = rendered_nchw[..., :crop_h, :crop_w]
    target_nchw = target_nchw[..., :crop_h, :crop_w]
    return robust_l1(
        F.avg_pool2d(rendered_nchw, kernel_size=effective_factor, stride=effective_factor)
        - F.avg_pool2d(target_nchw, kernel_size=effective_factor, stride=effective_factor)
    )


def crop_robust_l1(rendered: Tensor, target: Tensor, crop_size: int, crop_index: int) -> Tensor:
    if crop_size < 1:
        raise ValueError("crop loss size must be positive")
    if rendered.shape != target.shape:
        raise ValueError(f"crop loss shape mismatch: {tuple(rendered.shape)} vs {tuple(target.shape)}")
    if rendered.ndim not in {3, 4}:
        raise ValueError(f"crop loss expects HWC or THWC tensors, got ndim={rendered.ndim}")
    height = int(rendered.shape[-3])
    width = int(rendered.shape[-2])
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    y_candidates = sorted({0, max(0, (height - crop_h) // 2), max(0, height - crop_h)})
    x_candidates = sorted({0, max(0, (width - crop_w) // 2), max(0, width - crop_w)})
    y = y_candidates[(crop_index // len(x_candidates)) % len(y_candidates)]
    x = x_candidates[crop_index % len(x_candidates)]
    if rendered.ndim == 3:
        return robust_l1(rendered[y : y + crop_h, x : x + crop_w] - target[y : y + crop_h, x : x + crop_w])
    return robust_l1(rendered[:, y : y + crop_h, x : x + crop_w] - target[:, y : y + crop_h, x : x + crop_w])


def snapshot_world_tube_state(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def tensor_scalar_or_none(value: Tensor) -> float | None:
    scalar = float(value.detach().cpu())
    return scalar if math.isfinite(scalar) else None


def time_render_sequence(device: torch.device, render_fn: Callable[[], Any]) -> tuple[Any, float]:
    synchronize_device(device)
    started = time.perf_counter()
    rendered = render_fn()
    synchronize_device(device)
    return rendered, time.perf_counter() - started


def make_pinhole_camera(K: Tensor, w2c: Tensor) -> PinholeCamera:
    return PinholeCamera(
        fx=scalar_from_tensor(K[0, 0]),
        fy=scalar_from_tensor(K[1, 1]),
        cx=scalar_from_tensor(K[0, 2]),
        cy=scalar_from_tensor(K[1, 2]),
        world_to_camera=w2c.to(dtype=torch.float32),
    )


def select_lens(
    lens_models: list[str] | None,
    distortions: Tensor | None,
    view: int,
    *,
    camera_projection: str,
) -> tuple[str, Tensor | None]:
    if camera_projection == "legacy_pinhole":
        return "pinhole", None
    if camera_projection != "dataset_lens":
        raise ValueError("camera_projection must be one of: legacy_pinhole, dataset_lens")
    lens_model = "pinhole" if lens_models is None else str(lens_models[view])
    distortion = None if distortions is None else distortions[view]
    return lens_model, distortion


def camera_from_K_w2c_lens(
    K: Tensor,
    w2c: Tensor,
    *,
    lens_model: str = "pinhole",
    distortion: Tensor | None = None,
) -> CameraSpec:
    if lens_model == "pinhole" and distortion is None:
        return camera_from_K_w2c(K, w2c)
    return CameraSpec(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        camera_to_world=torch.linalg.inv(w2c),
        lens_model=lens_model,  # type: ignore[arg-type]
        distortion=distortion,
    )


def splat_camera_for_view_time(
    bundle,
    *,
    split: str,
    view: int,
    frame: int,
    camera_projection: str,
) -> CameraSpec:
    if split == "train":
        K = select_K_for_view_time(bundle.train_K, view=view, t=frame, view_count=bundle.train_view_count)
        w2c = select_w2c_for_view_time(bundle.train_w2c, view=view, t=frame)
        lens_model, distortion = select_lens(
            bundle.train_lens_models,
            bundle.train_distortions,
            view,
            camera_projection=camera_projection,
        )
    elif split == "heldout":
        K = select_K_for_view_time(bundle.heldout_K, view=view, t=frame, view_count=bundle.heldout_view_count)
        w2c = select_w2c_for_view_time(bundle.heldout_w2c, view=view, t=frame)
        lens_model, distortion = select_lens(
            bundle.heldout_lens_models,
            bundle.heldout_distortions,
            view,
            camera_projection=camera_projection,
        )
    else:
        raise ValueError("split must be one of: train, heldout")
    return camera_from_K_w2c_lens(K, w2c, lens_model=lens_model, distortion=distortion)


def project_world_tubes_dataset_lens(
    batch: WorldTubeBatch,
    K: Tensor,
    w2c: Tensor,
    config: UVTRenderConfig,
    *,
    lens_model: str,
    distortion: Tensor | None,
) -> ProjectedTubeSequence:
    world_to_camera = w2c.to(dtype=torch.float32)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    center_cam = batch.x0 @ rotation.T + translation
    camera = CameraSpec(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        camera_to_world=torch.linalg.inv(world_to_camera),
        lens_model=lens_model,  # type: ignore[arg-type]
        distortion=distortion,
    )
    pixels, _depths, pixel_jacobian, _front_mask = project_points_camera(center_cam, camera)
    ma, q_uvt, depth0, depth_beta, opacity, color = project_world_tubes_from_pixel_jacobian(
        batch,
        world_to_camera,
        pixels,
        pixel_jacobian,
        config,
    )
    return ProjectedTubeSequence(ma=ma, q_uvt=q_uvt, depth0=depth0, depth_beta=depth_beta, opacity=opacity, color=color)


def select_view_K(K: Tensor, view: int) -> Tensor:
    if K.ndim == 3:
        return K[view]
    if K.ndim == 4:
        return K[view, 0]
    raise ValueError(f"Expected K with shape [V,3,3] or [V,T,3,3], got {tuple(K.shape)}")


def select_view_w2c(w2c: Tensor, view: int) -> Tensor:
    if w2c.ndim != 4:
        raise ValueError(f"Expected w2c with shape [V,T,4,4], got {tuple(w2c.shape)}")
    return w2c[view, 0]


def _inv_softplus(value: Tensor) -> Tensor:
    clamped = value.clamp_min(1.0e-8)
    return clamped + torch.log(-torch.expm1(-clamped))


def _logit(value: Tensor) -> Tensor:
    clamped = value.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(clamped) - torch.log1p(-clamped)


def sample_init_pixels(
    *,
    tube_count: int,
    height: int,
    width: int,
    seed: int,
    sampling: str,
) -> tuple[Tensor, Tensor]:
    if sampling == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        ys = torch.randint(0, height, (tube_count,), generator=generator, device="cpu")
        xs = torch.randint(0, width, (tube_count,), generator=generator, device="cpu")
        return ys, xs
    if sampling != "grid":
        raise ValueError("sampling must be one of: random, grid")
    cols = max(1, int(round((tube_count * float(width) / float(height)) ** 0.5)))
    rows = max(1, (tube_count + cols - 1) // cols)
    xs_grid = torch.linspace(0.5, float(width) - 0.5, cols, device="cpu").round().long().clamp(0, width - 1)
    ys_grid = torch.linspace(0.5, float(height) - 0.5, rows, device="cpu").round().long().clamp(0, height - 1)
    yy, xx = torch.meshgrid(ys_grid, xs_grid, indexing="ij")
    return yy.reshape(-1)[:tube_count], xx.reshape(-1)[:tube_count]


def centered_frame_time(frame: int, frames: int) -> float:
    return float(frame) - 0.5 * float(frames - 1)


def initialize_world_tubes_from_view(
    frames: Tensor,
    K: Tensor,
    w2c: Tensor,
    *,
    tube_count: int,
    init_depth: float,
    seed: int,
    sampling: str,
    frame: int = 0,
    centered_t0: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    _, _, height, width = frames.shape
    ys_cpu, xs_cpu = sample_init_pixels(
        tube_count=tube_count,
        height=height,
        width=width,
        seed=seed,
        sampling=sampling,
    )
    ys = ys_cpu.to(frames.device)
    xs = xs_cpu.to(frames.device)
    colors = frames[frame, :, ys, xs].permute(1, 0).contiguous()
    z = torch.full((tube_count,), float(init_depth), dtype=torch.float32, device=frames.device)
    x_cam = (xs.float() + 0.5 - K[0, 2]) * z / K[0, 0]
    y_cam = (ys.float() + 0.5 - K[1, 2]) * z / K[1, 1]
    cam_points = torch.stack((x_cam, y_cam, z, torch.ones_like(z)), dim=-1)
    c2w = torch.linalg.inv(w2c)
    world_points = (cam_points @ c2w.T)[:, :3]
    t0 = torch.full((tube_count,), float(centered_t0), dtype=torch.float32, device=frames.device)
    return world_points.contiguous(), colors.clamp(1.0e-5, 1.0 - 1.0e-5), t0


def initialize_world_tubes_from_train_views(
    bundle,
    *,
    tube_count: int,
    init_depth: float,
    seed: int,
    init_views: str,
    init_sampling: str,
    init_frames: str,
    init_frame_indices: list[int] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    if init_frames not in {"first", "all", "fit"}:
        raise ValueError("init_frames must be one of: first, all, fit")
    if init_views not in {"first", "all_train"}:
        raise ValueError("init_views must be one of: first, all_train")
    if init_views == "first" and init_frames == "first":
        return initialize_world_tubes_from_view(
            bundle.train_frames[0],
            select_view_K(bundle.train_K, 0),
            select_view_w2c(bundle.train_w2c, 0),
            tube_count=tube_count,
            init_depth=init_depth,
            seed=seed,
            sampling=init_sampling,
        )
    train_view_count = int(bundle.train_frames.shape[0])
    total_frames = int(bundle.train_frames.shape[1])
    view_count = 1 if init_views == "first" else train_view_count
    if init_frames == "all":
        frame_indices = list(range(total_frames))
    elif init_frames == "fit":
        if not init_frame_indices:
            raise ValueError("init_frames=fit requires nonempty init_frame_indices")
        frame_indices = list(init_frame_indices)
    else:
        frame_indices = [0]
    frame_count = len(frame_indices)
    group_count = view_count * frame_count
    base_count = tube_count // group_count
    remainder = tube_count % group_count
    points = []
    colors = []
    t0_values = []
    for view in range(view_count):
        for frame_offset, source_frame in enumerate(frame_indices):
            group = view * frame_count + frame_offset
            count = base_count + (1 if group < remainder else 0)
            if count == 0:
                continue
            x0, rgb, t0 = initialize_world_tubes_from_view(
                bundle.train_frames[view],
                select_K_for_view_time(bundle.train_K, view=view, t=source_frame, view_count=train_view_count),
                select_w2c_for_view_time(bundle.train_w2c, view=view, t=source_frame),
                tube_count=count,
                init_depth=init_depth,
                seed=seed + view * 9973 + source_frame * 433,
                sampling=init_sampling,
                frame=source_frame,
                centered_t0=centered_frame_time(source_frame, total_frames) if init_frames in {"all", "fit"} else 0.0,
            )
            points.append(x0)
            colors.append(rgb)
            t0_values.append(t0)
    return torch.cat(points, dim=0).contiguous(), torch.cat(colors, dim=0).contiguous(), torch.cat(t0_values, dim=0).contiguous()


class WorldTubeModel(nn.Module):
    def __init__(
        self,
        *,
        init_x0: Tensor,
        init_color: Tensor,
        init_t0: Tensor,
        frames: int,
        init_precision_xy: float,
        init_lambda_t: float,
        init_opacity: float,
        min_precision_xy: float,
        min_lambda_t: float,
        velocity_reg_weight: float,
        depth_velocity_reg_weight: float,
        position_reg_weight: float,
    ) -> None:
        super().__init__()
        tube_count = int(init_x0.shape[0])
        self.tube_count = tube_count
        self.frames = int(frames)
        self.min_precision_xy = float(min_precision_xy)
        self.min_lambda_t = float(min_lambda_t)
        self.velocity_reg_weight = float(velocity_reg_weight)
        self.depth_velocity_reg_weight = float(depth_velocity_reg_weight)
        self.position_reg_weight = float(position_reg_weight)
        self.x0 = nn.Parameter(init_x0)
        self.velocity = nn.Parameter(torch.zeros_like(init_x0))
        precision = torch.full((tube_count, 2), float(init_precision_xy), dtype=torch.float32, device=init_x0.device)
        lambda_t = torch.full((tube_count,), float(init_lambda_t), dtype=torch.float32, device=init_x0.device)
        opacity = torch.full((tube_count,), float(init_opacity), dtype=torch.float32, device=init_x0.device)
        self.raw_precision_xy = nn.Parameter(_inv_softplus(precision - self.min_precision_xy))
        self.raw_lambda_t = nn.Parameter(_inv_softplus(lambda_t - self.min_lambda_t))
        self.raw_opacity = nn.Parameter(_logit(opacity / 0.99))
        self.raw_color = nn.Parameter(_logit(init_color))
        self.t0 = nn.Parameter(init_t0)

    def batch(self) -> WorldTubeBatch:
        return WorldTubeBatch(
            x0=self.x0,
            velocity=self.velocity,
            t0=self.t0,
            precision_xy=F.softplus(self.raw_precision_xy) + self.min_precision_xy,
            lambda_t=F.softplus(self.raw_lambda_t) + self.min_lambda_t,
            opacity=torch.sigmoid(self.raw_opacity) * 0.99,
            color=torch.sigmoid(self.raw_color),
        )

    def regularization(self) -> Tensor:
        reg = self.x0.new_tensor(0.0)
        if self.velocity_reg_weight:
            reg = reg + self.velocity_reg_weight * self.velocity.square().mean()
        if self.depth_velocity_reg_weight:
            reg = reg + self.depth_velocity_reg_weight * self.velocity[:, 2].square().mean()
        if self.position_reg_weight:
            reg = reg + self.position_reg_weight * self.x0.square().mean()
        return reg


@dataclass(frozen=True)
class RenderedSequence:
    rgb: Tensor
    alpha: Tensor


@dataclass(frozen=True)
class ProjectedTubeSequence:
    ma: Tensor
    q_uvt: Tensor
    depth0: Tensor
    depth_beta: Tensor
    opacity: Tensor
    color: Tensor


def project_world_tube_sequence(
    model: WorldTubeModel,
    K: Tensor,
    w2c: Tensor,
    config: UVTRenderConfig,
    *,
    camera_projection: str = "legacy_pinhole",
    lens_model: str = "pinhole",
    distortion: Tensor | None = None,
    full_frames: int | None = None,
    frame_start: int = 0,
) -> ProjectedTubeSequence:
    if camera_projection == "legacy_pinhole":
        camera = make_pinhole_camera(K, w2c)
        ma, q_uvt, depth0, depth_beta, opacity, color = project_world_tubes_pinhole(model.batch(), camera, config)
        projected = ProjectedTubeSequence(ma=ma, q_uvt=q_uvt, depth0=depth0, depth_beta=depth_beta, opacity=opacity, color=color)
    elif camera_projection == "dataset_lens":
        projected = project_world_tubes_dataset_lens(
            model.batch(),
            K,
            w2c,
            config,
            lens_model=lens_model,
            distortion=distortion,
        )
    else:
        raise ValueError("camera_projection must be one of: legacy_pinhole, dataset_lens")
    if full_frames is not None and int(full_frames) != int(config.frames):
        if frame_start < 0 or frame_start + int(config.frames) > int(full_frames):
            raise ValueError(
                f"frame window [{frame_start}, {frame_start + int(config.frames)}) exceeds full frame count {full_frames}."
            )
        global_minus_local_t = float(frame_start) - 0.5 * float(int(full_frames) - 1) + 0.5 * float(config.frames - 1)
        ma = torch.cat((projected.ma[:, :2], (projected.ma[:, 2:3] - global_minus_local_t)), dim=-1).contiguous()
        projected = ProjectedTubeSequence(
            ma=ma,
            q_uvt=projected.q_uvt,
            depth0=projected.depth0,
            depth_beta=projected.depth_beta,
            opacity=projected.opacity,
            color=projected.color,
        )
    return projected


def render_projected_sequence(projected: ProjectedTubeSequence, config: UVTRenderConfig, *, backend: str) -> RenderedSequence:
    if backend == "dense":
        rgb = dense_differentiable_render_uvt_tubes(
            projected.ma,
            projected.q_uvt,
            projected.depth0,
            projected.depth_beta,
            projected.opacity,
            projected.color,
            config,
        )
    elif backend == "metal_tile":
        rgb = render_uvt_tubes_metal_tile_backward(
            projected.ma,
            projected.q_uvt,
            projected.depth0,
            projected.depth_beta,
            projected.opacity,
            projected.color,
            config,
        )
    else:
        raise ValueError("backend must be one of: dense, metal_tile")
    alpha = torch.ones((config.frames, config.height, config.width), dtype=rgb.dtype, device=rgb.device)
    return RenderedSequence(rgb=rgb, alpha=alpha)


def _projected_uvt_inv_diag(q_uvt: Tensor) -> Tensor:
    a = q_uvt[:, 0]
    b = q_uvt[:, 1]
    c = q_uvt[:, 2]
    d = q_uvt[:, 3]
    e = q_uvt[:, 4]
    f = q_uvt[:, 5]
    co00 = d * f - e * e
    co11 = a * f - c * c
    co22 = a * d - b * b
    det = a * co00 - b * (b * f - c * e) + c * (b * e - c * d)
    eps = det.new_tensor(1.0e-8)
    safe_det = torch.where(det.abs() < eps, torch.where(det >= 0.0, eps, -eps), det)
    return torch.stack((co00, co11, co22), dim=-1).div(safe_det.unsqueeze(-1)).abs().clamp_min(1.0e-8)


def projected_tile_load_proxy(ma: Tensor, q_uvt: Tensor, opacity: Tensor, config: UVTRenderConfig) -> Tensor:
    del ma
    opacity_safe = opacity.clamp_min(float(config.alpha_threshold) * 1.0001)
    tau = -2.0 * torch.log((float(config.alpha_threshold) / opacity_safe).clamp_min(1.0e-8))
    half_extent = torch.sqrt((tau.unsqueeze(-1) * _projected_uvt_inv_diag(q_uvt)).clamp_min(0.0))
    span_x = 1.0 + 2.0 * half_extent[:, 0] / float(config.tile_x)
    span_y = 1.0 + 2.0 * half_extent[:, 1] / float(config.tile_y)
    span_t = 1.0 + 2.0 * half_extent[:, 2] / float(config.tile_t)
    return (span_x * span_y * span_t).mean()


def projected_depth_slope_proxy(depth_beta: Tensor, config: UVTRenderConfig) -> Tensor:
    half_extent = depth_beta.new_tensor(
        [
            0.5 * float(config.tile_x),
            0.5 * float(config.tile_y),
            max(0.0, 0.5 * float(config.tile_t - 1)),
        ]
    )
    return (depth_beta.abs() * half_extent).sum(dim=-1).mean()


def projected_depth_margin_proxy(ma: Tensor, depth0: Tensor, opacity: Tensor, config: UVTRenderConfig, *, margin: float) -> Tensor:
    tube_count = int(ma.shape[0])
    if tube_count < 2 or margin <= 0.0:
        return ma.new_tensor(0.0)
    ids_i, ids_j = torch.triu_indices(tube_count, tube_count, offset=1, device=ma.device)
    delta = ma.index_select(0, ids_i) - ma.index_select(0, ids_j)
    normalized_delta = torch.stack(
        (
            delta[:, 0] / float(config.tile_x),
            delta[:, 1] / float(config.tile_y),
            delta[:, 2] / float(config.tile_t),
        ),
        dim=-1,
    )
    proximity = torch.exp(-0.5 * normalized_delta.square().sum(dim=-1))
    opacity_weight = opacity.index_select(0, ids_i) * opacity.index_select(0, ids_j)
    depth_gap = (depth0.index_select(0, ids_i) - depth0.index_select(0, ids_j)).abs()
    margin_tensor = depth_gap.new_tensor(float(margin))
    violation = F.relu(margin_tensor - depth_gap).div(margin_tensor)
    weights = proximity * opacity_weight
    return (weights * violation).sum() / weights.sum().clamp_min(1.0e-8)


def projected_regularization(
    projected: ProjectedTubeSequence,
    config: UVTRenderConfig,
    *,
    tile_load_weight: float,
    tile_load_target: float,
    depth_slope_weight: float,
    depth_margin_weight: float,
    depth_margin: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    tile_proxy = projected_tile_load_proxy(projected.ma, projected.q_uvt, projected.opacity, config)
    slope_proxy = projected_depth_slope_proxy(projected.depth_beta, config)
    margin_proxy = projected_depth_margin_proxy(
        projected.ma,
        projected.depth0,
        projected.opacity,
        config,
        margin=depth_margin,
    )
    loss = projected.ma.new_tensor(0.0)
    if tile_load_weight:
        if tile_load_target > 0.0:
            target = projected.ma.new_tensor(float(tile_load_target))
            tile_loss = F.relu(tile_proxy - target).div(target).square()
        else:
            tile_loss = tile_proxy
        loss = loss + float(tile_load_weight) * tile_loss
    if depth_slope_weight:
        loss = loss + float(depth_slope_weight) * slope_proxy
    if depth_margin_weight:
        loss = loss + float(depth_margin_weight) * margin_proxy
    return loss, {
        "tile_load_proxy": tile_proxy,
        "depth_slope_proxy": slope_proxy,
        "depth_margin_proxy": margin_proxy,
    }


def render_world_tube_sequence(
    model: WorldTubeModel,
    K: Tensor,
    w2c: Tensor,
    config: UVTRenderConfig,
    *,
    backend: str,
    camera_projection: str = "legacy_pinhole",
    lens_model: str = "pinhole",
    distortion: Tensor | None = None,
    full_frames: int | None = None,
    frame_start: int = 0,
) -> RenderedSequence:
    projected = project_world_tube_sequence(
        model,
        K,
        w2c,
        config,
        camera_projection=camera_projection,
        lens_model=lens_model,
        distortion=distortion,
        full_frames=full_frames,
        frame_start=frame_start,
    )
    return render_projected_sequence(projected, config, backend=backend)


def select_train_view(step: int, train_views: list[int], device: torch.device, schedule: str) -> int:
    if not train_views:
        raise ValueError("train_views must not be empty")
    if schedule == "random":
        index = int(torch.randint(0, len(train_views), (1,), device=device).item())
        return train_views[index]
    if schedule == "cycle":
        return train_views[int(step % len(train_views))]
    raise ValueError(f"train_schedule must be one of: {', '.join(TRAIN_SCHEDULE_CHOICES)}")


def shuffled_cycle_values(values: list[int], *, seed: int) -> list[int]:
    if not values:
        raise ValueError("values must not be empty")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(values), generator=generator).tolist()
    return [values[int(index)] for index in order]


def cycle_pairs(left_values: list[int], right_values: list[int]) -> list[tuple[int, int]]:
    if not left_values or not right_values:
        raise ValueError("cycle pair values must not be empty")
    return [(left, right) for right in right_values for left in left_values]


def shuffled_cycle_pairs(left_values: list[int], right_values: list[int], *, seed: int) -> list[tuple[int, int]]:
    pairs = cycle_pairs(left_values, right_values)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator).tolist()
    return [pairs[int(index)] for index in order]


def reshuffled_cycle_item(items: list[Any], *, step: int, seed: int) -> Any:
    if not items:
        raise ValueError("reshuffled cycle items must not be empty")
    epoch, offset = divmod(step, len(items))
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    order = torch.randperm(len(items), generator=generator).tolist()
    return items[int(order[offset])]


def phase_rotated_cycle_item(items: list[Any], *, step: int, seed: int) -> Any:
    if not items:
        raise ValueError("phase-rotated cycle items must not be empty")
    epoch, offset = divmod(step, len(items))
    if len(items) == 1:
        return items[0]
    stride = 1 + seed % (len(items) - 1)
    while math.gcd(stride, len(items)) != 1:
        stride += 1
        if stride >= len(items):
            stride = 1
            break
    return items[int((offset + epoch * stride) % len(items))]


def view_shuffled_cycle_pair(left_values: list[int], right_values: list[int], *, step: int, seed: int) -> tuple[int, int]:
    if not left_values or not right_values:
        raise ValueError("view-shuffled cycle values must not be empty")
    right_step, left_offset = divmod(step, len(left_values))
    right_index = right_step % len(right_values)
    epoch = right_step // len(right_values)
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch * len(right_values) + right_index)
    order = torch.randperm(len(left_values), generator=generator).tolist()
    return left_values[int(order[left_offset])], right_values[right_index]


def optimizer_train_view_indices(view_count: int, mode: str) -> list[int]:
    if mode == "all":
        return list(range(view_count))
    if mode == "first_only":
        return [0]
    raise ValueError("optimizer_train_views must be one of: all, first_only")


def validation_frame_indices(frames: int, stride: int, offset: int) -> list[int]:
    if stride < 0:
        raise ValueError("validation_frame_stride must be nonnegative")
    if stride == 0:
        return []
    if offset < 0 or offset >= stride:
        raise ValueError("validation_frame_offset must satisfy 0 <= offset < validation_frame_stride")
    indices = list(range(offset, frames, stride))
    if not indices:
        raise ValueError("validation frame split produced no frames")
    if len(indices) >= frames:
        raise ValueError("validation frame split leaves no optimizer frames")
    return indices


def optimizer_frame_indices(frames: int, validation_indices: list[int]) -> list[int]:
    validation_set = set(validation_indices)
    indices = [frame for frame in range(frames) if frame not in validation_set]
    if not indices:
        raise ValueError("optimizer frame split is empty")
    return indices


def optimizer_window_starts(frames: int, window_frames: int, frame_indices: list[int]) -> list[int]:
    frame_set = set(frame_indices)
    starts = [
        start
        for start in range(frames - window_frames + 1)
        if all((start + offset) in frame_set for offset in range(window_frames))
    ]
    if not starts:
        raise ValueError("optimizer window split is empty; reduce window_frames or change the validation frame split")
    return starts


def select_train_frame(step: int, view_count: int, frame_indices: list[int], device: torch.device, schedule: str) -> int:
    if schedule == "random":
        index = int(torch.randint(0, len(frame_indices), (1,), device=device).item())
        return frame_indices[index]
    if schedule == "cycle":
        return frame_indices[int((step // view_count) % len(frame_indices))]
    raise ValueError(f"train_schedule must be one of: {', '.join(TRAIN_SCHEDULE_CHOICES)}")


def select_train_window_start(
    step: int,
    view_count: int,
    window_starts: list[int],
    device: torch.device,
    schedule: str,
) -> int:
    if schedule == "random":
        index = int(torch.randint(0, len(window_starts), (1,), device=device).item())
        return window_starts[index]
    if schedule == "cycle":
        return window_starts[int((step // view_count) % len(window_starts))]
    raise ValueError(f"train_schedule must be one of: {', '.join(TRAIN_SCHEDULE_CHOICES)}")


def train_world_tubes(
    *,
    bundle,
    tube_count: int,
    train_seconds: float,
    max_steps: int,
    lr: float,
    lr_decay_step: int,
    lr_decay_factor: float,
    init_depth: float,
    init_views: str,
    init_sampling: str,
    init_frames: str,
    init_precision_xy: float,
    init_lambda_t: float,
    init_opacity: float,
    min_precision_xy: float,
    min_lambda_t: float,
    velocity_reg_weight: float,
    depth_velocity_reg_weight: float,
    position_reg_weight: float,
    tile_load_reg_weight: float,
    tile_load_target: float,
    depth_slope_reg_weight: float,
    depth_margin_reg_weight: float,
    depth_margin: float,
    seed: int,
    backend: str,
    camera_projection: str,
    loss_scope: str,
    window_frames: int,
    train_schedule: str,
    optimizer_train_views: str,
    validation_frame_stride: int,
    validation_frame_offset: int,
    sequence_consistency_every_steps: int,
    sequence_consistency_frames: int,
    sequence_consistency_weight: float,
    multiscale_loss_weight: float,
    multiscale_loss_factor: int,
    crop_loss_weight: float,
    crop_loss_size: int,
    checkpoint_every_steps: int,
    render_config: UVTRenderConfig,
) -> tuple[WorldTubeModel, dict[str, Any], list[dict[str, Any]]]:
    if loss_scope not in {"sampled_frame", "view_sequence", "temporal_window"}:
        raise ValueError("loss_scope must be one of: sampled_frame, view_sequence, temporal_window")
    if train_schedule not in set(TRAIN_SCHEDULE_CHOICES):
        raise ValueError(f"train_schedule must be one of: {', '.join(TRAIN_SCHEDULE_CHOICES)}")
    if optimizer_train_views not in {"all", "first_only"}:
        raise ValueError("optimizer_train_views must be one of: all, first_only")
    if checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be nonnegative")
    if lr_decay_step < 0:
        raise ValueError("lr_decay_step must be nonnegative")
    if lr_decay_factor <= 0.0:
        raise ValueError("lr_decay_factor must be positive")
    if sequence_consistency_every_steps < 0:
        raise ValueError("sequence_consistency_every_steps must be nonnegative")
    if sequence_consistency_frames < 0:
        raise ValueError("sequence_consistency_frames must be nonnegative")
    if sequence_consistency_weight < 0.0:
        raise ValueError("sequence_consistency_weight must be nonnegative")
    if multiscale_loss_weight < 0.0:
        raise ValueError("multiscale_loss_weight must be nonnegative")
    if multiscale_loss_factor < 1:
        raise ValueError("multiscale_loss_factor must be positive")
    if crop_loss_weight < 0.0:
        raise ValueError("crop_loss_weight must be nonnegative")
    if crop_loss_size < 1:
        raise ValueError("crop_loss_size must be positive")
    torch.manual_seed(seed)
    train_frames = bundle.train_frames
    device = train_frames.device
    view_count, frames, _, height, width = train_frames.shape
    if sequence_consistency_frames > frames:
        raise ValueError(f"sequence_consistency_frames={sequence_consistency_frames} exceeds frame count {frames}")
    active_train_views = optimizer_train_view_indices(view_count, optimizer_train_views)
    validation_frames = validation_frame_indices(frames, validation_frame_stride, validation_frame_offset)
    active_train_frames = optimizer_frame_indices(frames, validation_frames)
    if window_frames < 1:
        raise ValueError("window_frames must be positive")
    if loss_scope == "temporal_window" and window_frames > frames:
        raise ValueError(f"window_frames={window_frames} exceeds frame count {frames}")
    active_window_starts = (
        optimizer_window_starts(frames, window_frames, active_train_frames)
        if loss_scope == "temporal_window"
        else []
    )
    consistency_window_frames = int(sequence_consistency_frames)
    consistency_window_starts = (
        optimizer_window_starts(frames, consistency_window_frames, active_train_frames)
        if consistency_window_frames > 0 and consistency_window_frames < frames
        else []
    )
    shuffled_view_cycle = shuffled_cycle_values(active_train_views, seed=seed + 1009)
    frame_pairs = cycle_pairs(active_train_views, active_train_frames)
    shuffled_frame_pairs = shuffled_cycle_pairs(active_train_views, active_train_frames, seed=seed + 2003)
    window_pairs = cycle_pairs(active_train_views, active_window_starts) if active_window_starts else []
    shuffled_window_pairs = (
        shuffled_cycle_pairs(active_train_views, active_window_starts, seed=seed + 3001)
        if active_window_starts
        else []
    )
    shuffled_consistency_window_starts = (
        shuffled_cycle_values(consistency_window_starts, seed=seed + 4001)
        if consistency_window_starts
        else []
    )
    active_train_frame_tensor = torch.tensor(active_train_frames, dtype=torch.long, device=device)
    init_x0, init_color, init_t0 = initialize_world_tubes_from_train_views(
        bundle,
        tube_count=tube_count,
        init_depth=init_depth,
        seed=seed,
        init_views=init_views,
        init_sampling=init_sampling,
        init_frames=init_frames,
        init_frame_indices=active_train_frames,
    )
    model = WorldTubeModel(
        init_x0=init_x0,
        init_color=init_color,
        init_t0=init_t0,
        frames=frames,
        init_precision_xy=init_precision_xy,
        init_lambda_t=init_lambda_t,
        init_opacity=init_opacity,
        min_precision_xy=min_precision_xy,
        min_lambda_t=min_lambda_t,
        velocity_reg_weight=velocity_reg_weight,
        depth_velocity_reg_weight=depth_velocity_reg_weight,
        position_reg_weight=position_reg_weight,
    ).to(device)
    full_config = render_config
    if full_config.height != height or full_config.width != width or full_config.frames != frames:
        raise ValueError("render_config dimensions must match bundle train frames")
    window_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=window_frames if loss_scope == "temporal_window" else frames,
        tile_x=full_config.tile_x,
        tile_y=full_config.tile_y,
        tile_t=full_config.tile_t,
        tile_capacity=full_config.tile_capacity,
        alpha_threshold=full_config.alpha_threshold,
        transmittance_threshold=full_config.transmittance_threshold,
        background=full_config.background,
        max_alpha=full_config.max_alpha,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    started_at = time.perf_counter()
    logs = []
    checkpoints: list[dict[str, Any]] = []
    last_finite_state = snapshot_world_tube_state(model)
    last_finite_step = 0
    stopped_reason: str | None = None
    stopped_step: int | None = None

    def append_train_log(
        *,
        completed_step: int,
        elapsed_after_step: float,
        current_lr: float,
        loss: Tensor,
        recon_loss: Tensor,
        crop_loss: Tensor,
        crop_term: Tensor,
        multiscale_loss: Tensor,
        multiscale_term: Tensor,
        sequence_consistency_loss: Tensor,
        consistency_term: Tensor,
        model_reg: Tensor,
        projected_reg: Tensor,
        projected_reg_metrics: dict[str, Tensor],
        grad_norm: Tensor | None = None,
        stop_reason: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "step": completed_step,
            "loss": tensor_scalar_or_none(loss),
            "recon_loss": tensor_scalar_or_none(recon_loss),
            "crop_loss": tensor_scalar_or_none(crop_loss),
            "crop_term": tensor_scalar_or_none(crop_term),
            "multiscale_loss": tensor_scalar_or_none(multiscale_loss),
            "multiscale_term": tensor_scalar_or_none(multiscale_term),
            "sequence_consistency_loss": tensor_scalar_or_none(sequence_consistency_loss),
            "sequence_consistency_term": tensor_scalar_or_none(consistency_term),
            "model_reg": tensor_scalar_or_none(model_reg),
            "projected_reg": tensor_scalar_or_none(projected_reg),
            "tile_load_proxy": tensor_scalar_or_none(projected_reg_metrics["tile_load_proxy"]),
            "depth_slope_proxy": tensor_scalar_or_none(projected_reg_metrics["depth_slope_proxy"]),
            "depth_margin_proxy": tensor_scalar_or_none(projected_reg_metrics["depth_margin_proxy"]),
            "lr": current_lr,
            "elapsed_s": elapsed_after_step,
        }
        if grad_norm is not None:
            entry["grad_norm"] = tensor_scalar_or_none(grad_norm)
        if stop_reason is not None:
            entry["stop_reason"] = stop_reason
        logs.append(entry)

    step = 0
    while step < max_steps:
        elapsed = time.perf_counter() - started_at
        if step > 0 and elapsed >= train_seconds:
            break
        current_lr = lr * lr_decay_factor if lr_decay_step > 0 and step >= lr_decay_step else lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        frame_override: int | None = None
        window_start_override: int | None = None
        if train_schedule in {"shuffled_cycle", "reshuffled_cycle", "phase_rotated_cycle", "view_shuffled_cycle"}:
            if loss_scope == "sampled_frame":
                if train_schedule == "shuffled_cycle":
                    view, frame_override = shuffled_frame_pairs[int(step % len(shuffled_frame_pairs))]
                elif train_schedule == "reshuffled_cycle":
                    view, frame_override = reshuffled_cycle_item(frame_pairs, step=step, seed=seed + 2003)
                elif train_schedule == "phase_rotated_cycle":
                    view, frame_override = phase_rotated_cycle_item(frame_pairs, step=step, seed=seed + 2003)
                else:
                    view, frame_override = view_shuffled_cycle_pair(
                        active_train_views,
                        active_train_frames,
                        step=step,
                        seed=seed + 2003,
                    )
            elif loss_scope == "temporal_window":
                if train_schedule == "shuffled_cycle":
                    view, window_start_override = shuffled_window_pairs[int(step % len(shuffled_window_pairs))]
                elif train_schedule == "reshuffled_cycle":
                    view, window_start_override = reshuffled_cycle_item(window_pairs, step=step, seed=seed + 3001)
                elif train_schedule == "phase_rotated_cycle":
                    view, window_start_override = phase_rotated_cycle_item(window_pairs, step=step, seed=seed + 3001)
                else:
                    view, window_start_override = view_shuffled_cycle_pair(
                        active_train_views,
                        active_window_starts,
                        step=step,
                        seed=seed + 3001,
                    )
            else:
                if train_schedule == "shuffled_cycle":
                    view = shuffled_view_cycle[int(step % len(shuffled_view_cycle))]
                elif train_schedule == "reshuffled_cycle":
                    view = reshuffled_cycle_item(active_train_views, step=step, seed=seed + 1009)
                elif train_schedule == "phase_rotated_cycle":
                    view = phase_rotated_cycle_item(active_train_views, step=step, seed=seed + 1009)
                else:
                    view = reshuffled_cycle_item(active_train_views, step=step, seed=seed + 1009)
        else:
            view = select_train_view(step, active_train_views, device, train_schedule)
        lens_model, distortion = select_lens(
            bundle.train_lens_models,
            bundle.train_distortions,
            view,
            camera_projection=camera_projection,
        )
        optimizer.zero_grad(set_to_none=True)
        if loss_scope == "sampled_frame":
            frame = (
                frame_override
                if frame_override is not None
                else select_train_frame(step, len(active_train_views), active_train_frames, device, train_schedule)
            )
            projected = project_world_tube_sequence(
                model,
                select_view_K(bundle.train_K, view),
                select_view_w2c(bundle.train_w2c, view),
                full_config,
                camera_projection=camera_projection,
                lens_model=lens_model,
                distortion=distortion,
            )
            rendered = render_projected_sequence(projected, full_config, backend=backend)
            target = train_frames[view, frame].permute(1, 2, 0)
            recon_loss = robust_l1(rendered.rgb[frame] - target)
            multiscale_loss = (
                downsampled_robust_l1(rendered.rgb[frame], target, multiscale_loss_factor)
                if multiscale_loss_weight > 0.0
                else train_frames.new_tensor(0.0)
            )
            crop_loss = (
                crop_robust_l1(rendered.rgb[frame], target, crop_loss_size, step)
                if crop_loss_weight > 0.0
                else train_frames.new_tensor(0.0)
            )
        elif loss_scope == "view_sequence":
            projected = project_world_tube_sequence(
                model,
                select_view_K(bundle.train_K, view),
                select_view_w2c(bundle.train_w2c, view),
                full_config,
                camera_projection=camera_projection,
                lens_model=lens_model,
                distortion=distortion,
            )
            rendered = render_projected_sequence(projected, full_config, backend=backend)
            target = train_frames[view].permute(0, 2, 3, 1).contiguous()
            rendered_active = rendered.rgb.index_select(0, active_train_frame_tensor)
            target_active = target.index_select(0, active_train_frame_tensor)
            recon_loss = robust_l1(
                rendered_active
                - target_active
            )
            multiscale_loss = (
                downsampled_robust_l1(rendered_active, target_active, multiscale_loss_factor)
                if multiscale_loss_weight > 0.0
                else train_frames.new_tensor(0.0)
            )
            crop_loss = (
                crop_robust_l1(rendered_active, target_active, crop_loss_size, step)
                if crop_loss_weight > 0.0
                else train_frames.new_tensor(0.0)
            )
        else:
            frame_start = (
                window_start_override
                if window_start_override is not None
                else select_train_window_start(
                    step,
                    len(active_train_views),
                    active_window_starts,
                    device,
                    train_schedule,
                )
            )
            projected = project_world_tube_sequence(
                model,
                select_view_K(bundle.train_K, view),
                select_view_w2c(bundle.train_w2c, view),
                window_config,
                camera_projection=camera_projection,
                lens_model=lens_model,
                distortion=distortion,
                full_frames=frames,
                frame_start=frame_start,
            )
            rendered = render_projected_sequence(projected, window_config, backend=backend)
            target = train_frames[view, frame_start : frame_start + window_frames].permute(0, 2, 3, 1).contiguous()
            recon_loss = robust_l1(rendered.rgb - target)
            multiscale_loss = (
                downsampled_robust_l1(rendered.rgb, target, multiscale_loss_factor)
                if multiscale_loss_weight > 0.0
                else train_frames.new_tensor(0.0)
            )
            crop_loss = (
                crop_robust_l1(rendered.rgb, target, crop_loss_size, step)
                if crop_loss_weight > 0.0
                else train_frames.new_tensor(0.0)
            )
        sequence_consistency_loss = train_frames.new_tensor(0.0)
        consistency_due = (
            sequence_consistency_weight > 0.0
            and sequence_consistency_every_steps > 0
            and (step + 1) % sequence_consistency_every_steps == 0
        )
        if consistency_due:
            if consistency_window_frames > 0 and consistency_window_frames < frames:
                consistency_config = UVTRenderConfig(
                    height=height,
                    width=width,
                    frames=consistency_window_frames,
                    tile_x=full_config.tile_x,
                    tile_y=full_config.tile_y,
                    tile_t=full_config.tile_t,
                    tile_capacity=full_config.tile_capacity,
                    alpha_threshold=full_config.alpha_threshold,
                    transmittance_threshold=full_config.transmittance_threshold,
                    background=full_config.background,
                    max_alpha=full_config.max_alpha,
                )
                if train_schedule in {"random", "cycle"}:
                    consistency_start = select_train_window_start(
                        step,
                        len(active_train_views),
                        consistency_window_starts,
                        device,
                        train_schedule,
                    )
                elif train_schedule == "shuffled_cycle":
                    consistency_start = shuffled_consistency_window_starts[int(step % len(shuffled_consistency_window_starts))]
                elif train_schedule == "reshuffled_cycle":
                    consistency_start = reshuffled_cycle_item(
                        consistency_window_starts,
                        step=step,
                        seed=seed + 4001,
                    )
                elif train_schedule == "phase_rotated_cycle":
                    consistency_start = phase_rotated_cycle_item(
                        consistency_window_starts,
                        step=step,
                        seed=seed + 4001,
                    )
                else:
                    consistency_start = consistency_window_starts[
                        int((step // len(active_train_views)) % len(consistency_window_starts))
                    ]
            else:
                consistency_config = full_config
                consistency_start = 0
            sequence_projected = project_world_tube_sequence(
                model,
                select_view_K(bundle.train_K, view),
                select_view_w2c(bundle.train_w2c, view),
                consistency_config,
                camera_projection=camera_projection,
                lens_model=lens_model,
                distortion=distortion,
                full_frames=frames if consistency_config.frames != frames else None,
                frame_start=consistency_start,
            )
            sequence_rendered = render_projected_sequence(sequence_projected, consistency_config, backend=backend)
            if consistency_config.frames == frames:
                sequence_target = train_frames[view].permute(0, 2, 3, 1).contiguous()
                sequence_consistency_loss = robust_l1(
                    sequence_rendered.rgb.index_select(0, active_train_frame_tensor)
                    - sequence_target.index_select(0, active_train_frame_tensor)
                )
            else:
                sequence_target = train_frames[
                    view, consistency_start : consistency_start + consistency_window_frames
                ].permute(0, 2, 3, 1).contiguous()
                sequence_consistency_loss = robust_l1(sequence_rendered.rgb - sequence_target)
        train_render_config = window_config if loss_scope == "temporal_window" else full_config
        model_reg = model.regularization()
        projected_reg, projected_reg_metrics = projected_regularization(
            projected,
            train_render_config,
            tile_load_weight=tile_load_reg_weight,
            tile_load_target=tile_load_target,
            depth_slope_weight=depth_slope_reg_weight,
            depth_margin_weight=depth_margin_reg_weight,
            depth_margin=depth_margin,
        )
        consistency_term = float(sequence_consistency_weight) * sequence_consistency_loss
        multiscale_term = float(multiscale_loss_weight) * multiscale_loss
        crop_term = float(crop_loss_weight) * crop_loss
        loss = recon_loss + crop_term + multiscale_term + consistency_term + model_reg + projected_reg
        completed_step = step + 1
        should_log = step == 0 or completed_step % 10 == 0
        if should_log and not bool(torch.isfinite(loss.detach()).all().item()):
            stopped_reason = "nonfinite_loss"
            stopped_step = completed_step
            elapsed_after_step = time.perf_counter() - started_at
            append_train_log(
                completed_step=completed_step,
                elapsed_after_step=elapsed_after_step,
                current_lr=current_lr,
                loss=loss,
                recon_loss=recon_loss,
                crop_loss=crop_loss,
                crop_term=crop_term,
                multiscale_loss=multiscale_loss,
                multiscale_term=multiscale_term,
                sequence_consistency_loss=sequence_consistency_loss,
                consistency_term=consistency_term,
                model_reg=model_reg,
                projected_reg=projected_reg,
                projected_reg_metrics=projected_reg_metrics,
                stop_reason=stopped_reason,
            )
            model.load_state_dict(last_finite_state)
            step = last_finite_step
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        elapsed_after_step = time.perf_counter() - started_at
        if should_log:
            append_train_log(
                completed_step=completed_step,
                elapsed_after_step=elapsed_after_step,
                current_lr=current_lr,
                loss=loss,
                recon_loss=recon_loss,
                crop_loss=crop_loss,
                crop_term=crop_term,
                multiscale_loss=multiscale_loss,
                multiscale_term=multiscale_term,
                sequence_consistency_loss=sequence_consistency_loss,
                consistency_term=consistency_term,
                model_reg=model_reg,
                projected_reg=projected_reg,
                projected_reg_metrics=projected_reg_metrics,
                grad_norm=grad_norm,
            )
        if checkpoint_every_steps > 0 and completed_step % checkpoint_every_steps == 0:
            checkpoint_state = snapshot_world_tube_state(model)
            checkpoints.append(
                {
                    "step": completed_step,
                    "elapsed_s": elapsed_after_step,
                    "state": checkpoint_state,
                }
            )
            last_finite_state = checkpoint_state
            last_finite_step = completed_step
        step += 1
    train_elapsed = time.perf_counter() - started_at
    if checkpoint_every_steps > 0 and (not checkpoints or checkpoints[-1]["step"] != step):
        checkpoints.append({"step": step, "elapsed_s": train_elapsed, "state": snapshot_world_tube_state(model)})
    return (
        model,
        {
            "steps": step,
            "train_loop_elapsed_s": train_elapsed,
            "optimizer_train_views": optimizer_train_views,
            "optimizer_train_view_indices": active_train_views,
            "optimizer_frame_indices": active_train_frames,
            "validation_frame_indices": validation_frames,
            "validation_frame_stride": validation_frame_stride,
            "validation_frame_offset": validation_frame_offset,
            "sequence_consistency_every_steps": sequence_consistency_every_steps,
            "sequence_consistency_frames": sequence_consistency_frames,
            "sequence_consistency_weight": sequence_consistency_weight,
            "multiscale_loss_weight": multiscale_loss_weight,
            "multiscale_loss_factor": multiscale_loss_factor,
            "crop_loss_weight": crop_loss_weight,
            "crop_loss_size": crop_loss_size,
            "stopped_reason": stopped_reason,
            "stopped_step": stopped_step,
            "logs": logs,
        },
        checkpoints,
    )


def train_free_splats(
    *,
    bundle,
    splat_count: int,
    train_seconds: float,
    max_steps: int,
    lr: float,
    init_depth: float,
    init_scale: float,
    seed: int,
    renderer: str,
    camera_projection: str,
) -> tuple[FreeDynamic3DGS, SplatRenderConfig, dict[str, Any]]:
    torch.manual_seed(seed)
    train_video = bundle.train_frames
    view_count, frames, _, height, width = train_video.shape
    init_xyz, init_rgb = initialize_material_points_from_first_frame(
        video=train_video[0].permute(0, 2, 3, 1).contiguous(),
        K=bundle.train_K[0],
        num_elements=splat_count,
        init_depth=init_depth,
    )
    model = FreeDynamic3DGS(
        init_xyz=init_xyz,
        init_rgb=init_rgb,
        num_frames=frames,
        splat_mode="per_frame",
        init_scale=init_scale,
        scale_init_log_jitter=0.0,
        init_alpha_logit=0.0,
        init_xyz_noise=0.001,
        init_quat_noise=0.0,
        log_scale_min=-12.0,
        log_scale_max=4.0,
    ).to(train_video.device)
    render_cfg = SplatRenderConfig(
        height=height,
        width=width,
        renderer=renderer,
        tile_size=16 if renderer == "fast_mac" else 8,
        bound_scale=3.0,
        alpha_threshold=1.0 / 255.0,
        near_plane=1.0e-3,
        camera_projection="camera_model" if camera_projection == "dataset_lens" else "legacy_pinhole",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    started_at = time.perf_counter()
    logs = []
    step = 0
    while step < max_steps:
        elapsed = time.perf_counter() - started_at
        if step > 0 and elapsed >= train_seconds:
            break
        view = int(torch.randint(0, view_count, (1,), device=train_video.device).item())
        frame = int(torch.randint(0, frames, (1,), device=train_video.device).item())
        optimizer.zero_grad(set_to_none=True)
        camera = splat_camera_for_view_time(
            bundle,
            split="train",
            view=view,
            frame=frame,
            camera_projection=camera_projection,
        )
        image = render_gaussian_frame(
            model.frame(frame),
            camera,
            height=height,
            width=width,
            mode=render_cfg.renderer,
            tile_size=render_cfg.tile_size,
            bound_scale=render_cfg.bound_scale,
            alpha_threshold=render_cfg.alpha_threshold,
            near_plane=render_cfg.near_plane,
            camera_projection=render_cfg.camera_projection,
        ).permute(1, 2, 0)
        loss = robust_l1(image - train_video[view, frame].permute(1, 2, 0))
        loss = loss + 1.0e-4 * model.scale_loss() + 1.0e-3 * model.temporal_smoothness_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            logs.append({"step": step + 1, "loss": float(loss.detach().cpu()), "elapsed_s": time.perf_counter() - started_at})
        step += 1
    return model, render_cfg, {"steps": step, "train_loop_elapsed_s": time.perf_counter() - started_at, "logs": logs}


@torch.no_grad()
def eval_world_tubes(
    model: WorldTubeModel,
    bundle,
    *,
    backend: str,
    camera_projection: str,
    render_config: UVTRenderConfig,
    frame_metric_splits: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    _, frames, _, height, width = bundle.train_frames.shape
    config = render_config
    if config.height != height or config.width != width or config.frames != frames:
        raise ValueError("render_config dimensions must match bundle train frames")
    train_rows = []
    train_metrics = []
    train_frame_split_metrics: dict[str, list[dict[str, float]]] = {
        name: [] for name, indices in (frame_metric_splits or {}).items() if indices
    }
    train_render_times = []
    device = bundle.train_frames.device
    render_started = time.perf_counter()
    for view in range(bundle.train_view_count):
        lens_model, distortion = select_lens(
            bundle.train_lens_models,
            bundle.train_distortions,
            view,
            camera_projection=camera_projection,
        )
        rendered, render_elapsed = time_render_sequence(
            device,
            lambda view=view, lens_model=lens_model, distortion=distortion: render_world_tube_sequence(
                model,
                select_view_K(bundle.train_K, view),
                select_view_w2c(bundle.train_w2c, view),
                config,
                backend=backend,
                camera_projection=camera_projection,
                lens_model=lens_model,
                distortion=distortion,
            ),
        )
        train_render_times.append(render_elapsed)
        target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
        train_rows.append((target, rendered))
        train_metrics.append(video_metrics(rendered.rgb, target))
        for name, metrics_rows in train_frame_split_metrics.items():
            metrics_rows.append(subset_video_metrics(rendered.rgb, target, frame_metric_splits[name]))
    heldout_rows = []
    heldout_metrics = []
    heldout_render_times = []
    if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
        for view in range(bundle.heldout_view_count):
            lens_model, distortion = select_lens(
                bundle.heldout_lens_models,
                bundle.heldout_distortions,
                view,
                camera_projection=camera_projection,
            )
            rendered, render_elapsed = time_render_sequence(
                device,
                lambda view=view, lens_model=lens_model, distortion=distortion: render_world_tube_sequence(
                    model,
                    select_view_K(bundle.heldout_K, view),
                    select_view_w2c(bundle.heldout_w2c, view),
                    config,
                    backend=backend,
                    camera_projection=camera_projection,
                    lens_model=lens_model,
                    distortion=distortion,
                ),
            )
            heldout_render_times.append(render_elapsed)
            target = bundle.heldout_frames[view].permute(0, 2, 3, 1).contiguous()
            heldout_rows.append((target, rendered))
            heldout_metrics.append(video_metrics(rendered.rgb, target))
    metrics = aggregate_view_metrics(train_metrics)
    for name, metrics_rows in train_frame_split_metrics.items():
        metrics.update(prefix_metrics(f"train_{name}_frame", aggregate_view_metrics(metrics_rows)))
    if heldout_metrics:
        metrics.update(prefix_metrics("heldout", aggregate_view_metrics(heldout_metrics)))
    metrics["eval_render_elapsed_s"] = time.perf_counter() - render_started
    metrics.update(render_time_metrics(train_render_times, heldout_render_times))
    return {
        "metrics": metrics,
        "train_rows": train_rows,
        "heldout_rows": heldout_rows,
        "train_view_metrics": train_metrics,
        "heldout_view_metrics": heldout_metrics,
    }


@torch.no_grad()
def eval_world_tube_checkpoints(
    model: WorldTubeModel,
    checkpoints: list[dict[str, Any]],
    bundle,
    *,
    backend: str,
    camera_projection: str,
    render_config: UVTRenderConfig,
    frame_metric_splits: dict[str, list[int]] | None = None,
) -> dict[str, Any] | None:
    if not checkpoints:
        return None
    final_state = snapshot_world_tube_state(model)
    rows = []
    for checkpoint in checkpoints:
        model.load_state_dict(checkpoint["state"])
        eval_result = eval_world_tubes(
            model,
            bundle,
            backend=backend,
            camera_projection=camera_projection,
            render_config=render_config,
            frame_metric_splits=frame_metric_splits,
        )
        metrics = eval_result["metrics"]
        train_view_eval_psnr = [view_metrics.get("eval_psnr") for view_metrics in eval_result["train_view_metrics"]]
        train_view_eval_l1 = [view_metrics.get("eval_l1") for view_metrics in eval_result["train_view_metrics"]]
        train_view_psnr_values = [float(value) for value in train_view_eval_psnr if value is not None]
        rows.append(
            {
                "step": checkpoint["step"],
                "elapsed_s": checkpoint["elapsed_s"],
                "eval_psnr": metrics.get("eval_psnr"),
                "heldout_eval_psnr": metrics.get("heldout_eval_psnr"),
                "eval_l1": metrics.get("eval_l1"),
                "heldout_eval_l1": metrics.get("heldout_eval_l1"),
                "train_view_eval_psnr": train_view_eval_psnr,
                "train_view_eval_l1": train_view_eval_l1,
                "train_min_view_eval_psnr": min(train_view_psnr_values) if train_view_psnr_values else None,
                "train_view_eval_psnr_gap": max(train_view_psnr_values) - min(train_view_psnr_values)
                if train_view_psnr_values
                else None,
                "train_fit_frame_eval_psnr": metrics.get("train_fit_frame_eval_psnr"),
                "train_dev_frame_eval_psnr": metrics.get("train_dev_frame_eval_psnr"),
                "train_fit_frame_eval_l1": metrics.get("train_fit_frame_eval_l1"),
                "train_dev_frame_eval_l1": metrics.get("train_dev_frame_eval_l1"),
                "eval_render_only_elapsed_s": metrics.get("eval_render_only_elapsed_s"),
                "eval_heldout_render_only_elapsed_s": metrics.get("eval_heldout_render_only_elapsed_s"),
            }
        )
    model.load_state_dict(final_state)
    best = max(rows, key=lambda row: row["heldout_eval_psnr"] if row["heldout_eval_psnr"] is not None else row["eval_psnr"])
    best_train = max(rows, key=lambda row: row["eval_psnr"])
    best_min_train_view = max(
        rows,
        key=lambda row: row["train_min_view_eval_psnr"]
        if row["train_min_view_eval_psnr"] is not None
        else float("-inf"),
    )
    rows_with_dev_frame = [row for row in rows if row["train_dev_frame_eval_psnr"] is not None]
    best_train_dev_frame = (
        max(rows_with_dev_frame, key=lambda row: row["train_dev_frame_eval_psnr"])
        if rows_with_dev_frame
        else None
    )
    return {
        "rows": rows,
        "best_by_heldout_psnr": best,
        "best_by_train_psnr": best_train,
        "best_by_min_train_view_psnr": best_min_train_view,
        "best_by_train_dev_frame_psnr": best_train_dev_frame,
    }


def select_world_tube_checkpoint_row(
    checkpoint_curve: dict[str, Any],
    *,
    selector: str,
    train_psnr_plateau_delta: float,
    train_psnr_plateau_patience: int,
    train_psnr_gain_drop: float,
    train_view_gap_collapse: float,
    train_view_gap_max: float,
    train_view_index: int,
) -> tuple[dict[str, Any], str, bool, dict[str, Any]]:
    if selector == "best_heldout":
        return checkpoint_curve["best_by_heldout_psnr"], "heldout_eval_psnr", True, {}
    if selector == "best_train_psnr":
        return checkpoint_curve["best_by_train_psnr"], "eval_psnr", False, {}
    if selector == "best_min_train_view_psnr":
        return checkpoint_curve["best_by_min_train_view_psnr"], "train_min_view_eval_psnr", False, {}
    if selector == "best_train_dev_frame_psnr":
        selected = checkpoint_curve["best_by_train_dev_frame_psnr"]
        if selected is None or selected.get("train_dev_frame_eval_psnr") is None:
            raise ValueError("best_train_dev_frame_psnr requires --uvt-validation-frame-stride > 0")
        return selected, "train_dev_frame_eval_psnr", False, {}
    if selector == "best_train_view_psnr":
        if train_view_index < 0:
            raise ValueError("train_view_index must be nonnegative")

        def train_view_psnr(row: dict[str, Any]) -> float:
            values = row.get("train_view_eval_psnr")
            if values is None or train_view_index >= len(values) or values[train_view_index] is None:
                return float("-inf")
            return float(values[train_view_index])

        selected = max(checkpoint_curve["rows"], key=train_view_psnr)
        return (
            selected,
            f"train_view_{train_view_index}_eval_psnr",
            False,
            {"train_view_index": train_view_index, "selected_train_view_eval_psnr": train_view_psnr(selected)},
        )
    if selector == "first_balanced_train_psnr_plateau":
        if train_psnr_plateau_delta < 0.0:
            raise ValueError("train_psnr_plateau_delta must be nonnegative")
        if train_view_gap_max < 0.0:
            raise ValueError("train_view_gap_max must be nonnegative")
        rows = checkpoint_curve["rows"]
        if not rows:
            raise ValueError("checkpoint_curve has no rows")
        previous = rows[0]
        for row in rows[1:]:
            previous_psnr = previous.get("eval_psnr")
            eval_psnr = row.get("eval_psnr")
            train_view_gap = previous.get("train_view_eval_psnr_gap")
            if eval_psnr is None or previous_psnr is None or train_view_gap is None:
                previous = row
                continue
            gain = float(eval_psnr) - float(previous_psnr)
            if gain <= train_psnr_plateau_delta and float(train_view_gap) <= train_view_gap_max:
                return (
                    previous,
                    "eval_psnr_plateau_with_train_view_balance",
                    False,
                    {
                        "train_psnr_plateau_delta": train_psnr_plateau_delta,
                        "train_view_gap_max": train_view_gap_max,
                        "selected_train_view_eval_psnr_gap": train_view_gap,
                        "next_step": row["step"],
                        "next_eval_psnr": eval_psnr,
                        "next_gain": gain,
                    },
                )
            previous = row
        return (
            rows[-1],
            "eval_psnr_plateau_with_train_view_balance",
            False,
            {
                "train_psnr_plateau_delta": train_psnr_plateau_delta,
                "train_view_gap_max": train_view_gap_max,
                "fallback": "no_balanced_plateau_before_final_checkpoint",
            },
        )
    if selector == "first_train_view_gap_collapse":
        if train_view_gap_collapse < 0.0:
            raise ValueError("train_view_gap_collapse must be nonnegative")
        rows = checkpoint_curve["rows"]
        if not rows:
            raise ValueError("checkpoint_curve has no rows")
        previous = rows[0]
        for row in rows[1:]:
            train_view_gap = row.get("train_view_eval_psnr_gap")
            if train_view_gap is None:
                previous = row
                continue
            if float(train_view_gap) <= train_view_gap_collapse:
                return (
                    previous,
                    "train_view_gap_collapse_previous_checkpoint",
                    False,
                    {
                        "train_view_gap_collapse": train_view_gap_collapse,
                        "selected_train_view_eval_psnr_gap": previous.get("train_view_eval_psnr_gap"),
                        "next_step": row["step"],
                        "next_train_view_eval_psnr_gap": train_view_gap,
                    },
                )
            previous = row
        return (
            rows[-1],
            "train_view_gap_collapse_previous_checkpoint",
            False,
            {
                "train_view_gap_collapse": train_view_gap_collapse,
                "fallback": "no_train_view_gap_collapse_before_final_checkpoint",
            },
        )
    if selector == "first_train_psnr_gain_drop":
        if train_psnr_plateau_delta < 0.0:
            raise ValueError("train_psnr_plateau_delta must be nonnegative")
        if train_psnr_gain_drop < 0.0:
            raise ValueError("train_psnr_gain_drop must be nonnegative")
        rows = checkpoint_curve["rows"]
        if not rows:
            raise ValueError("checkpoint_curve has no rows")
        previous = rows[0]
        previous_gain: float | None = None
        saw_low_gain = False
        for row in rows[1:]:
            previous_psnr = previous.get("eval_psnr")
            eval_psnr = row.get("eval_psnr")
            if eval_psnr is None or previous_psnr is None:
                previous = row
                previous_gain = None
                saw_low_gain = False
                continue
            gain = float(eval_psnr) - float(previous_psnr)
            if previous_gain is not None and saw_low_gain:
                gain_drop = previous_gain - gain
                if gain_drop >= train_psnr_gain_drop:
                    return (
                        previous,
                        "eval_psnr_gain_drop_after_low_gain",
                        False,
                        {
                            "train_psnr_low_gain_delta": train_psnr_plateau_delta,
                            "train_psnr_gain_drop": train_psnr_gain_drop,
                            "selected_gain": previous_gain,
                            "next_step": row["step"],
                            "next_eval_psnr": eval_psnr,
                            "next_gain": gain,
                            "observed_gain_drop": gain_drop,
                        },
                    )
            if gain <= train_psnr_plateau_delta:
                saw_low_gain = True
            previous = row
            previous_gain = gain
        return (
            rows[-1],
            "eval_psnr_gain_drop_after_low_gain",
            False,
            {
                "train_psnr_low_gain_delta": train_psnr_plateau_delta,
                "train_psnr_gain_drop": train_psnr_gain_drop,
                "fallback": "no_gain_drop_after_low_gain_before_final_checkpoint",
            },
        )
    if selector != "first_train_psnr_plateau":
        raise ValueError(
            "selector must be one of: best_heldout, best_train_psnr, "
            "best_min_train_view_psnr, best_train_view_psnr, "
            "best_train_dev_frame_psnr, "
            "first_train_psnr_plateau, first_train_psnr_gain_drop, "
            "first_train_view_gap_collapse, "
            "first_balanced_train_psnr_plateau"
        )
    if train_psnr_plateau_delta < 0.0:
        raise ValueError("train_psnr_plateau_delta must be nonnegative")
    if train_psnr_plateau_patience < 1:
        raise ValueError("train_psnr_plateau_patience must be at least 1")
    rows = checkpoint_curve["rows"]
    if not rows:
        raise ValueError("checkpoint_curve has no rows")
    previous = rows[0]
    previous_psnr = previous.get("eval_psnr")
    plateau_run_length = 0
    for row in rows[1:]:
        eval_psnr = row.get("eval_psnr")
        if eval_psnr is None or previous_psnr is None:
            previous = row
            previous_psnr = eval_psnr
            plateau_run_length = 0
            continue
        gain = float(eval_psnr) - float(previous_psnr)
        if gain <= train_psnr_plateau_delta:
            plateau_run_length += 1
        else:
            plateau_run_length = 0
        if plateau_run_length >= train_psnr_plateau_patience:
            return (
                row,
                "eval_psnr_gain_from_previous_checkpoint",
                False,
                {
                    "train_psnr_plateau_delta": train_psnr_plateau_delta,
                    "train_psnr_plateau_patience": train_psnr_plateau_patience,
                    "previous_step": previous["step"],
                    "previous_eval_psnr": previous_psnr,
                    "selected_gain": gain,
                    "selected_plateau_run_length": plateau_run_length,
                },
            )
        previous = row
        previous_psnr = eval_psnr
    return (
        rows[-1],
        "eval_psnr_gain_from_previous_checkpoint",
        False,
        {
            "train_psnr_plateau_delta": train_psnr_plateau_delta,
            "train_psnr_plateau_patience": train_psnr_plateau_patience,
            "fallback": "no_plateau_before_final_checkpoint",
        },
    )


def find_checkpoint_state(checkpoints: list[dict[str, Any]], selected_row: dict[str, Any]) -> dict[str, Tensor]:
    for checkpoint in checkpoints:
        if checkpoint["step"] == selected_row["step"] and checkpoint["elapsed_s"] == selected_row["elapsed_s"]:
            return checkpoint["state"]
    raise ValueError(f"Selected checkpoint state not found for row: {selected_row}")


@torch.no_grad()
def world_tube_metal_stats(
    model: WorldTubeModel,
    bundle,
    *,
    camera_projection: str,
    render_config: UVTRenderConfig,
) -> dict[str, Any]:
    if bundle.train_frames.device.type != "mps":
        return {"skipped": "Metal stats require MPS tensors."}
    _, frames, _, height, width = bundle.train_frames.shape
    config = render_config
    if config.height != height or config.width != width or config.frames != frames:
        raise ValueError("render_config dimensions must match bundle train frames")

    def row(
        split: str,
        camera_name: str,
        K: Tensor,
        w2c: Tensor,
        lens_model: str,
        distortion: Tensor | None,
    ) -> dict[str, Any]:
        projected = project_world_tube_sequence(
            model,
            K,
            w2c,
            config,
            camera_projection=camera_projection,
            lens_model=lens_model,
            distortion=distortion,
        )
        result = render_uvt_tubes(
            projected.ma,
            projected.q_uvt,
            projected.depth0,
            projected.depth_beta,
            projected.opacity,
            projected.color,
            config,
            return_aux=True,
        )
        if result.stats is None:
            raise AssertionError("Metal render did not return stats")
        return {"split": split, "camera": camera_name, "stats": result.stats.__dict__}

    rows = [
        row(
            "train",
            name,
            select_view_K(bundle.train_K, view),
            select_view_w2c(bundle.train_w2c, view),
            *select_lens(
                bundle.train_lens_models,
                bundle.train_distortions,
                view,
                camera_projection=camera_projection,
            ),
        )
        for view, name in enumerate(bundle.train_camera_names)
    ]
    if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
        rows.extend(
            row(
                "heldout",
                name,
                select_view_K(bundle.heldout_K, view),
                select_view_w2c(bundle.heldout_w2c, view),
                *select_lens(
                    bundle.heldout_lens_models,
                    bundle.heldout_distortions,
                    view,
                    camera_projection=camera_projection,
                ),
            )
            for view, name in enumerate(bundle.heldout_camera_names)
        )
    return {"rows": rows}


@torch.no_grad()
def eval_free_splats(
    model: FreeDynamic3DGS,
    render_cfg: SplatRenderConfig,
    bundle,
    *,
    camera_projection: str,
) -> dict[str, Any]:
    train_cameras = tuple(
        tuple(
            splat_camera_for_view_time(
                bundle,
                split="train",
                view=view,
                frame=frame,
                camera_projection=camera_projection,
            )
            for frame in range(bundle.frame_count)
        )
        for view in range(bundle.train_view_count)
    )
    render_started = time.perf_counter()
    train_rows = []
    train_metrics = []
    train_render_times = []
    device = bundle.train_frames.device
    for view, cameras in enumerate(train_cameras):
        rendered, render_elapsed = time_render_sequence(
            device,
            lambda cameras=cameras: render_splat_sequence(model, list(cameras), render_cfg),
        )
        train_render_times.append(render_elapsed)
        target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
        train_rows.append((target, RenderedSequence(rgb=rendered["rgb"], alpha=rendered["alpha"])))
        train_metrics.append(video_metrics(rendered["rgb"], target))
    heldout_rows = []
    heldout_metrics = []
    heldout_render_times = []
    if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
        for view in range(bundle.heldout_view_count):
            cameras = [
                splat_camera_for_view_time(
                    bundle,
                    split="heldout",
                    view=view,
                    frame=frame,
                    camera_projection=camera_projection,
                )
                for frame in range(bundle.frame_count)
            ]
            rendered, render_elapsed = time_render_sequence(
                device,
                lambda cameras=cameras: render_splat_sequence(model, cameras, render_cfg),
            )
            heldout_render_times.append(render_elapsed)
            target = bundle.heldout_frames[view].permute(0, 2, 3, 1).contiguous()
            heldout_rows.append((target, RenderedSequence(rgb=rendered["rgb"], alpha=rendered["alpha"])))
            heldout_metrics.append(video_metrics(rendered["rgb"], target))
    metrics = aggregate_view_metrics(train_metrics)
    if heldout_metrics:
        metrics.update(prefix_metrics("heldout", aggregate_view_metrics(heldout_metrics)))
    metrics["eval_render_elapsed_s"] = time.perf_counter() - render_started
    metrics.update(render_time_metrics(train_render_times, heldout_render_times))
    return {"metrics": metrics, "train_rows": train_rows, "heldout_rows": heldout_rows}


def aggregate_view_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {key: sum(float(row[key]) for row in rows) / float(len(rows)) for key in keys}


def save_first_row_media(output_dir: Path, prefix: str, rows: list[tuple[Tensor, RenderedSequence]], fps: float) -> None:
    if not rows:
        return
    target, rendered = rows[0]
    save_preview_strip(output_dir / f"{prefix}_preview.png", target=target, rendered=rendered.rgb, alpha=rendered.alpha)
    save_side_by_side_mp4(output_dir / f"{prefix}_side_by_side.mp4", target=target, rendered=rendered.rgb, fps=fps)


def config_data_for_run(config: dict[str, Any], *, target_size: int, max_frames: int) -> dict[str, Any]:
    del target_size
    data_cfg = dict(config["data"])
    data_cfg["max_frames"] = max_frames
    if data_cfg.get("multicam_manifest") is not None:
        data_cfg["multicam_manifest"] = str(resolve_dynaworld_path(data_cfg["multicam_manifest"]))
    return data_cfg


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def apply_uvt_tile_env(config: UVTRenderConfig) -> None:
    os.environ["STAR_UVT_TILE_X"] = str(config.tile_x)
    os.environ["STAR_UVT_TILE_Y"] = str(config.tile_y)
    os.environ["STAR_UVT_TILE_T"] = str(config.tile_t)
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(config.tile_capacity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--train-seconds", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
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
    parser.add_argument("--uvt-tile-x", type=int, default=env_int("STAR_UVT_TILE_X", 8))
    parser.add_argument("--uvt-tile-y", type=int, default=env_int("STAR_UVT_TILE_Y", 8))
    parser.add_argument("--uvt-tile-t", type=int, default=env_int("STAR_UVT_TILE_T", 2))
    parser.add_argument("--uvt-tile-capacity", type=int, default=env_int("STAR_UVT_TILE_CAPACITY", 128))
    parser.add_argument("--uvt-render-backend", choices=("dense", "metal_tile"), default="dense")
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
    parser.add_argument("--uvt-train-schedule", choices=TRAIN_SCHEDULE_CHOICES, default="random")
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
    parser.add_argument("--splat-count", type=int, default=512)
    parser.add_argument("--splat-lr", type=float, default=0.002)
    parser.add_argument("--splat-renderer", choices=("dense", "fast_mac"), default="dense")
    parser.add_argument("--splat-camera-projection", choices=("legacy_pinhole", "dataset_lens"), default="legacy_pinhole")
    parser.add_argument("--skip-splats", action="store_true")
    parser.add_argument("--init-depth", type=float, default=2.0)
    parser.add_argument("--uvt-init-views", choices=("first", "all_train"), default="first")
    parser.add_argument("--uvt-init-sampling", choices=("random", "grid"), default="random")
    parser.add_argument("--uvt-init-frames", choices=("first", "all", "fit"), default="first")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "research_project" / "benchmarks" / "results" / "multicam_heldout_compare")
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = load_config_file(resolve_dynaworld_path(args.baseline_config))
    data_cfg = config_data_for_run(config, target_size=args.target_size, max_frames=args.max_frames)
    camera_cfg = dict(config["camera"])
    bundle = load_multicam_video_bundle(data_cfg=data_cfg, camera_cfg=camera_cfg, target_size=args.target_size, device=device)
    if args.uvt_render_backend == "metal_tile" and device.type != "mps":
        raise ValueError("--uvt-render-backend=metal_tile requires device=mps")
    render_config = UVTRenderConfig(
        height=int(bundle.train_frames.shape[-2]),
        width=int(bundle.train_frames.shape[-1]),
        frames=int(bundle.frame_count),
        tile_x=args.uvt_tile_x,
        tile_y=args.uvt_tile_y,
        tile_t=args.uvt_tile_t,
        tile_capacity=args.uvt_tile_capacity,
    )
    apply_uvt_tile_env(render_config)
    uvt_validation_frames = validation_frame_indices(
        int(bundle.frame_count),
        args.uvt_validation_frame_stride,
        args.uvt_validation_frame_offset,
    )
    uvt_optimizer_frames = optimizer_frame_indices(int(bundle.frame_count), uvt_validation_frames)
    uvt_frame_metric_splits = (
        {"fit": uvt_optimizer_frames, "dev": uvt_validation_frames}
        if uvt_validation_frames
        else None
    )

    out_dir = resolve_variant_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "baseline_config": str(resolve_dynaworld_path(args.baseline_config)),
        "target_size": args.target_size,
        "max_frames": args.max_frames,
        "train_seconds": args.train_seconds,
        "device": str(device),
        "seed": args.seed,
        "train_cameras": bundle.train_camera_names,
        "heldout_cameras": bundle.heldout_camera_names,
        "pose_source": bundle.pose_source,
        "uvt_camera_projection": args.uvt_camera_projection,
        "splat_camera_projection": args.splat_camera_projection,
        "skip_splats": args.skip_splats,
        "train_lens_models": bundle.train_lens_models,
        "heldout_lens_models": bundle.heldout_lens_models,
        "reference_vjepa_f32_256_16f_alpha1_128": {
            "heldout_eval_psnr": 13.6248,
            "train_psnr": 19.4875,
            "wall_clock": "18m00s train loop; 18m22s W&B runtime",
            "source": "dynaworld/BASELINES.md",
        },
    }
    write_json(out_dir / "run_meta.json", {**run_meta, "config_data": serialize_config_value(data_cfg)})

    uvt_model, uvt_train, uvt_checkpoints = train_world_tubes(
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
    )
    uvt_eval = eval_world_tubes(
        uvt_model,
        bundle,
        backend=args.uvt_render_backend,
        camera_projection=args.uvt_camera_projection,
        render_config=render_config,
        frame_metric_splits=uvt_frame_metric_splits,
    )
    uvt_checkpoint_curve = eval_world_tube_checkpoints(
        uvt_model,
        uvt_checkpoints,
        bundle,
        backend=args.uvt_render_backend,
        camera_projection=args.uvt_camera_projection,
        render_config=render_config,
        frame_metric_splits=uvt_frame_metric_splits,
    )
    selected_report: dict[str, Any] | None = None
    if args.uvt_select_checkpoint != "none":
        if uvt_checkpoint_curve is None:
            raise ValueError("--uvt-select-checkpoint requires --uvt-checkpoint-every-steps > 0")
        final_state = snapshot_world_tube_state(uvt_model)
        selected_row, selection_metric, uses_heldout_for_selection, selection_detail = select_world_tube_checkpoint_row(
            uvt_checkpoint_curve,
            selector=args.uvt_select_checkpoint,
            train_psnr_plateau_delta=args.uvt_select_train_psnr_plateau_delta,
            train_psnr_plateau_patience=args.uvt_select_train_psnr_plateau_patience,
            train_psnr_gain_drop=args.uvt_select_train_psnr_gain_drop,
            train_view_gap_collapse=args.uvt_select_train_view_gap_collapse,
            train_view_gap_max=args.uvt_select_train_view_gap_max,
            train_view_index=args.uvt_select_train_view_index,
        )
        uvt_model.load_state_dict(find_checkpoint_state(uvt_checkpoints, selected_row))
        selected_eval = eval_world_tubes(
            uvt_model,
            bundle,
            backend=args.uvt_render_backend,
            camera_projection=args.uvt_camera_projection,
            render_config=render_config,
            frame_metric_splits=uvt_frame_metric_splits,
        )
        save_first_row_media(
            out_dir,
            "star_uvt_selected_train_view0",
            selected_eval["train_rows"],
            fps=float(bundle.metadata.get("fps", 4.0)),
        )
        save_first_row_media(
            out_dir,
            "star_uvt_selected_heldout_view0",
            selected_eval["heldout_rows"],
            fps=float(bundle.metadata.get("fps", 4.0)),
        )
        selected_report = {
            "selector": args.uvt_select_checkpoint,
            "selection_metric": selection_metric,
            "uses_heldout_for_selection": uses_heldout_for_selection,
            "selection_detail": selection_detail,
            "selected_step": selected_row["step"],
            "selected_elapsed_s": selected_row["elapsed_s"],
            "metrics": selected_eval["metrics"],
            "metal_stats": world_tube_metal_stats(
                uvt_model,
                bundle,
                camera_projection=args.uvt_camera_projection,
                render_config=render_config,
            )
            if args.uvt_render_backend == "metal_tile"
            else None,
        }
        uvt_model.load_state_dict(final_state)
    save_first_row_media(out_dir, "star_uvt_train_view0", uvt_eval["train_rows"], fps=float(bundle.metadata.get("fps", 4.0)))
    save_first_row_media(out_dir, "star_uvt_heldout_view0", uvt_eval["heldout_rows"], fps=float(bundle.metadata.get("fps", 4.0)))

    splat_report: dict[str, Any] | None = None
    if not args.skip_splats:
        splat_model, splat_render_cfg, splat_train = train_free_splats(
            bundle=bundle,
            splat_count=args.splat_count,
            train_seconds=args.train_seconds,
            max_steps=args.max_steps,
            lr=args.splat_lr,
            init_depth=args.init_depth,
            init_scale=0.035,
            seed=args.seed,
            renderer=args.splat_renderer,
            camera_projection=args.splat_camera_projection,
        )
        splat_eval = eval_free_splats(
            splat_model,
            splat_render_cfg,
            bundle,
            camera_projection=args.splat_camera_projection,
        )
        save_first_row_media(out_dir, "free_dynamic_splats_train_view0", splat_eval["train_rows"], fps=float(bundle.metadata.get("fps", 4.0)))
        save_first_row_media(out_dir, "free_dynamic_splats_heldout_view0", splat_eval["heldout_rows"], fps=float(bundle.metadata.get("fps", 4.0)))
        splat_report = {
            "splat_count": args.splat_count,
            "renderer": args.splat_renderer,
            "camera_projection": args.splat_camera_projection,
            "render_camera_projection": splat_render_cfg.camera_projection,
            **splat_train,
            "metrics": splat_eval["metrics"],
        }

    report = {
        "meta": run_meta,
        "star_uvt": {
            "tube_count": args.uvt_tubes,
            "render_backend": args.uvt_render_backend,
            "camera_projection": args.uvt_camera_projection,
            "lr": args.uvt_lr,
            "lr_decay_step": args.uvt_lr_decay_step,
            "lr_decay_factor": args.uvt_lr_decay_factor,
            "init_precision_xy": args.uvt_init_precision_xy,
            "init_lambda_t": args.uvt_init_lambda_t,
            "init_opacity": args.uvt_init_opacity,
            "min_precision_xy": args.uvt_min_precision_xy,
            "min_lambda_t": args.uvt_min_lambda_t,
            "velocity_reg": args.uvt_velocity_reg,
            "depth_velocity_reg": args.uvt_depth_velocity_reg,
            "position_reg": args.uvt_position_reg,
            "tile_load_reg": args.uvt_tile_load_reg,
            "tile_load_target": args.uvt_tile_load_target,
            "depth_slope_reg": args.uvt_depth_slope_reg,
            "depth_margin_reg": args.uvt_depth_margin_reg,
            "depth_margin": args.uvt_depth_margin,
            "tile_x": render_config.tile_x,
            "tile_y": render_config.tile_y,
            "tile_t": render_config.tile_t,
            "tile_capacity": render_config.tile_capacity,
            "init_views": args.uvt_init_views,
            "init_sampling": args.uvt_init_sampling,
            "init_frames": args.uvt_init_frames,
            "loss_scope": args.uvt_loss_scope,
            "window_frames": args.uvt_window_frames if args.uvt_loss_scope == "temporal_window" else None,
            "sequence_consistency_every_steps": args.uvt_sequence_consistency_every_steps,
            "sequence_consistency_frames": args.uvt_sequence_consistency_frames,
            "sequence_consistency_weight": args.uvt_sequence_consistency_weight,
            "multiscale_loss_weight": args.uvt_multiscale_loss_weight,
            "multiscale_loss_factor": args.uvt_multiscale_loss_factor,
            "crop_loss_weight": args.uvt_crop_loss_weight,
            "crop_loss_size": args.uvt_crop_loss_size,
            "train_schedule": args.uvt_train_schedule,
            "optimizer_train_views_arg": args.uvt_optimizer_train_views,
            "checkpoint_every_steps": args.uvt_checkpoint_every_steps,
            "select_checkpoint": args.uvt_select_checkpoint,
            "select_train_psnr_plateau_delta": args.uvt_select_train_psnr_plateau_delta,
            "select_train_psnr_plateau_patience": args.uvt_select_train_psnr_plateau_patience,
            "select_train_psnr_gain_drop": args.uvt_select_train_psnr_gain_drop,
            "select_train_view_gap_collapse": args.uvt_select_train_view_gap_collapse,
            "select_train_view_gap_max": args.uvt_select_train_view_gap_max,
            "select_train_view_index": args.uvt_select_train_view_index,
            "validation_frame_stride": args.uvt_validation_frame_stride,
            "validation_frame_offset": args.uvt_validation_frame_offset,
            **uvt_train,
            "metrics": uvt_eval["metrics"],
            "checkpoint_curve": uvt_checkpoint_curve,
            "metal_stats": world_tube_metal_stats(
                uvt_model,
                bundle,
                camera_projection=args.uvt_camera_projection,
                render_config=render_config,
            )
            if args.uvt_render_backend == "metal_tile"
            else None,
        },
        "star_uvt_selected": selected_report,
        "free_dynamic_splats": splat_report,
    }
    write_json(out_dir / "comparison_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote STAR-UVT multicam heldout comparison to {out_dir}")


if __name__ == "__main__":
    main()
