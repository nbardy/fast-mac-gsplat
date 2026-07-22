from __future__ import annotations

import argparse
import gc
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
from device_memory import DeviceMemorySampler  # noqa: E402
from multicam_video_data import load_multicam_video_bundle  # noqa: E402
from paper_training_protocol import (  # noqa: E402
    PaperCostTracker,
    PaperPhaseTimer,
    SpacetimeEpochSampler,
    apply_paper_dataset_contract,
    normalize_image_size,
    normalize_paper_stages,
    paper_stage_for_step,
    resize_video_frames,
    scale_intrinsics,
)
from paper_training_types import MetalKernelSpec  # noqa: E402
from perceptual_metrics import video_lpips  # noqa: E402
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
    from research_project.trainer_harness.tile_metal_autograd import (
        BACKWARD_POLICY_NAMES,
        render_uvt_tubes_metal_tile_backward,
        resolve_backward_policy,
    )
    from research_project.trainer_harness.world_tube import (
        PinholeCameraMotion,
        PinholeCamera,
        WorldTubeBatch,
        project_world_tubes_from_pixel_jacobian,
        project_world_tubes_pinhole,
        project_world_tubes_pinhole_motion,
        project_world_tubes_pinhole_projective_motion,
    )
    from research_project.trainer_harness.variable_camera_segments import project_piecewise_camera_time_segments
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = ROOT / "research_project" / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from model import dense_differentiable_render_uvt_tubes
    from tile_metal_autograd import BACKWARD_POLICY_NAMES, render_uvt_tubes_metal_tile_backward, resolve_backward_policy
    from variable_camera_segments import project_piecewise_camera_time_segments
    from world_tube import (
        PinholeCamera,
        PinholeCameraMotion,
        WorldTubeBatch,
        project_world_tubes_from_pixel_jacobian,
        project_world_tubes_pinhole,
        project_world_tubes_pinhole_motion,
        project_world_tubes_pinhole_projective_motion,
    )


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
    "epoch_view_shuffled_cycle",
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


@dataclass
class VideoMetricAccumulator:
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    element_count: int = 0
    ssim_sum: float = 0.0
    frame_count: int = 0

    def update(self, rendered: Tensor, target: Tensor) -> None:
        if rendered.shape != target.shape or rendered.ndim != 4:
            raise ValueError(
                f"streamed video metrics require matching [T,H,W,C], got {tuple(rendered.shape)} and {tuple(target.shape)}"
            )
        diff = rendered.float() - target.float()
        self.absolute_error_sum += float(diff.abs().sum().detach().cpu())
        self.squared_error_sum += float(diff.square().sum().detach().cpu())
        self.element_count += int(diff.numel())
        chunk_frames = int(rendered.shape[0])
        self.ssim_sum += video_metrics(rendered, target)["eval_ssim"] * chunk_frames
        self.frame_count += chunk_frames

    def metrics(self) -> dict[str, float]:
        if self.element_count < 1 or self.frame_count < 1:
            raise ValueError("streamed video metrics require at least one frame")
        l1 = self.absolute_error_sum / float(self.element_count)
        mse = self.squared_error_sum / float(self.element_count)
        return {
            "eval_l1": l1,
            "eval_mse": mse,
            "eval_psnr": -10.0 * math.log10(max(mse, 1.0e-12)),
            "eval_ssim": self.ssim_sum / float(self.frame_count),
        }


def media_frame_positions(frame_count: int, max_frames: int) -> set[int]:
    count = min(int(frame_count), int(max_frames))
    if count < 1:
        return set()
    return set(torch.linspace(0, frame_count - 1, steps=count).round().to(torch.long).tolist())


def append_chunk_media(
    *,
    start: int,
    stop: int,
    selected: set[int],
    target: Tensor,
    rendered: Tensor,
    alpha: Tensor,
    targets_out: list[Tensor],
    rendered_out: list[Tensor],
    alpha_out: list[Tensor],
) -> None:
    local_positions = [position - start for position in sorted(selected) if start <= position < stop]
    if not local_positions:
        return
    local = torch.tensor(local_positions, dtype=torch.long, device=rendered.device)
    targets_out.append(target.index_select(0, local).detach().cpu())
    rendered_out.append(rendered.index_select(0, local).detach().cpu())
    alpha_out.append(alpha.index_select(0, local).detach().cpu())


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


def select_view_K_sequence(K: Tensor, *, view: int, frames: int, view_count: int) -> Tensor:
    if K.ndim == 3:
        return torch.stack([select_K_for_view_time(K, view=view, t=frame, view_count=view_count) for frame in range(frames)])
    if K.ndim == 4:
        return K[view, :frames].contiguous()
    raise ValueError(f"Expected K with shape [V,3,3] or [V,T,3,3], got {tuple(K.shape)}")


def select_view_w2c_sequence(w2c: Tensor, *, view: int, frames: int) -> Tensor:
    if w2c.ndim != 4:
        raise ValueError(f"Expected w2c with shape [V,T,4,4], got {tuple(w2c.shape)}")
    return w2c[view, :frames].contiguous()


def apply_synthetic_camera_motion(
    K_seq: Tensor,
    w2c_seq: Tensor,
    *,
    pan_x: float,
    pan_y: float,
    dolly_z: float,
    zoom: float,
    principal_x: float,
    principal_y: float,
) -> tuple[Tensor, Tensor]:
    if not any((pan_x, pan_y, dolly_z, zoom, principal_x, principal_y)):
        return K_seq, w2c_seq
    frames = int(K_seq.shape[0])
    phase = (
        torch.zeros((1,), dtype=K_seq.dtype, device=K_seq.device)
        if frames == 1
        else torch.linspace(-0.5, 0.5, frames, dtype=K_seq.dtype, device=K_seq.device)
    )
    moved_K = K_seq.clone()
    moved_w2c = w2c_seq.clone()
    moved_w2c[:, 0, 3] = moved_w2c[:, 0, 3] + float(pan_x) * phase
    moved_w2c[:, 1, 3] = moved_w2c[:, 1, 3] + float(pan_y) * phase
    moved_w2c[:, 2, 3] = moved_w2c[:, 2, 3] + float(dolly_z) * phase
    moved_K[:, 0, 0] = moved_K[:, 0, 0] * (1.0 + float(zoom) * phase)
    moved_K[:, 1, 1] = moved_K[:, 1, 1] * (1.0 + float(zoom) * phase)
    moved_K[:, 0, 2] = moved_K[:, 0, 2] + float(principal_x) * phase
    moved_K[:, 1, 2] = moved_K[:, 1, 2] + float(principal_y) * phase
    return moved_K, moved_w2c


def camera_sequences_for_view(
    K: Tensor,
    w2c: Tensor,
    *,
    view: int,
    frames: int,
    view_count: int,
    synthetic_pan_x: float,
    synthetic_pan_y: float,
    synthetic_dolly_z: float,
    synthetic_zoom: float,
    synthetic_principal_x: float,
    synthetic_principal_y: float,
) -> tuple[Tensor, Tensor]:
    K_seq = select_view_K_sequence(K, view=view, frames=frames, view_count=view_count)
    w2c_seq = select_view_w2c_sequence(w2c, view=view, frames=frames)
    return apply_synthetic_camera_motion(
        K_seq,
        w2c_seq,
        pan_x=synthetic_pan_x,
        pan_y=synthetic_pan_y,
        dolly_z=synthetic_dolly_z,
        zoom=synthetic_zoom,
        principal_x=synthetic_principal_x,
        principal_y=synthetic_principal_y,
    )


def local_frame_time(frame: float, frames: int) -> float:
    return float(frame) - 0.5 * float(frames - 1)


def global_to_local_time(global_t: float, *, full_frames: int, config: UVTRenderConfig, frame_start: int) -> float:
    global_minus_local_t = float(frame_start) - 0.5 * float(full_frames - 1) + 0.5 * float(config.frames - 1)
    return float(global_t) - global_minus_local_t


def project_world_tube_sequence_dynamic_first_order(
    *,
    model: WorldTubeModel,
    K_seq: Tensor,
    w2c_seq: Tensor,
    config: UVTRenderConfig,
    full_frames: int,
    frame_start: int,
    projective_gauge: bool = False,
) -> ProjectedTubeSequence:
    window_mid_frame = float(frame_start) + 0.5 * float(config.frames - 1)
    mid_index = int(round(window_mid_frame))
    mid_index = max(0, min(int(full_frames) - 1, mid_index))
    prev_index = max(0, mid_index - 1)
    next_index = min(int(full_frames) - 1, mid_index + 1)
    if next_index == prev_index:
        K_dot = torch.zeros_like(K_seq[mid_index])
        w2c_dot = torch.zeros_like(w2c_seq[mid_index])
    else:
        dt = float(next_index - prev_index)
        K_dot = (K_seq[next_index] - K_seq[prev_index]) / dt
        w2c_dot = (w2c_seq[next_index] - w2c_seq[prev_index]) / dt
    K_mid = K_seq[mid_index]
    chart_global_t = local_frame_time(window_mid_frame, int(full_frames))
    camera = PinholeCameraMotion(
        fx=float(K_mid[0, 0].detach().cpu()),
        fy=float(K_mid[1, 1].detach().cpu()),
        cx=float(K_mid[0, 2].detach().cpu()),
        cy=float(K_mid[1, 2].detach().cpu()),
        fx_dot=float(K_dot[0, 0].detach().cpu()),
        fy_dot=float(K_dot[1, 1].detach().cpu()),
        cx_dot=float(K_dot[0, 2].detach().cpu()),
        cy_dot=float(K_dot[1, 2].detach().cpu()),
        world_to_camera=w2c_seq[mid_index].to(dtype=torch.float32),
        world_to_camera_dot=w2c_dot.to(dtype=torch.float32),
        chart_time=chart_global_t,
    )
    projector = project_world_tubes_pinhole_projective_motion if projective_gauge else project_world_tubes_pinhole_motion
    ma, q_uvt, depth0, depth_beta, opacity, color = projector(model.batch(), camera, config)
    local_t = global_to_local_time(chart_global_t, full_frames=int(full_frames), config=config, frame_start=frame_start)
    ma = torch.cat((ma[:, :2], torch.full_like(ma[:, 2:3], local_t)), dim=-1).contiguous()
    return ProjectedTubeSequence(ma=ma, q_uvt=q_uvt, depth0=depth0, depth_beta=depth_beta, opacity=opacity, color=color)


def project_world_tube_sequence_segmented_camera(
    *,
    model: WorldTubeModel,
    K_seq: Tensor,
    w2c_seq: Tensor,
    config: UVTRenderConfig,
    full_frames: int,
    frame_start: int,
    frames_per_segment: int,
) -> ProjectedTubeSequence:
    segments = project_piecewise_camera_time_segments(
        model.batch(),
        K_seq,
        w2c_seq,
        config,
        full_frames=full_frames,
        frame_start=frame_start,
        frames_per_segment=frames_per_segment,
    )
    return ProjectedTubeSequence(
        ma=segments.ma,
        q_uvt=segments.q_uvt,
        depth0=segments.depth0,
        depth_beta=segments.depth_beta,
        opacity=segments.opacity,
        color=segments.color,
    )


def project_world_tube_sequence_camera_mode(
    *,
    model: WorldTubeModel,
    K_seq: Tensor,
    w2c_seq: Tensor,
    config: UVTRenderConfig,
    full_frames: int,
    frame_start: int,
    camera_sequence_mode: str,
    segment_frames: int,
) -> ProjectedTubeSequence:
    if camera_sequence_mode == "static_view":
        return project_world_tube_sequence(
            model,
            K_seq[frame_start],
            w2c_seq[frame_start],
            config,
            full_frames=full_frames,
            frame_start=frame_start,
        )
    if camera_sequence_mode == "dynamic_first_order":
        return project_world_tube_sequence_dynamic_first_order(
            model=model,
            K_seq=K_seq,
            w2c_seq=w2c_seq,
            config=config,
            full_frames=full_frames,
            frame_start=frame_start,
        )
    if camera_sequence_mode == "projective_first_order":
        return project_world_tube_sequence_dynamic_first_order(
            model=model,
            K_seq=K_seq,
            w2c_seq=w2c_seq,
            config=config,
            full_frames=full_frames,
            frame_start=frame_start,
            projective_gauge=True,
        )
    if camera_sequence_mode == "segmented":
        return project_world_tube_sequence_segmented_camera(
            model=model,
            K_seq=K_seq,
            w2c_seq=w2c_seq,
            config=config,
            full_frames=full_frames,
            frame_start=frame_start,
            frames_per_segment=segment_frames,
        )
    raise ValueError(
        "camera_sequence_mode must be one of: static_view, dynamic_first_order, projective_first_order, segmented"
    )


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
    device = K.device
    ys_cpu, xs_cpu = sample_init_pixels(
        tube_count=tube_count,
        height=height,
        width=width,
        seed=seed,
        sampling=sampling,
    )
    frame_ys = ys_cpu.to(frames.device)
    frame_xs = xs_cpu.to(frames.device)
    colors = frames[frame, :, frame_ys, frame_xs].permute(1, 0).contiguous().to(device)
    ys = ys_cpu.to(device)
    xs = xs_cpu.to(device)
    z = torch.full((tube_count,), float(init_depth), dtype=torch.float32, device=device)
    x_cam = (xs.float() + 0.5 - K[0, 2]) * z / K[0, 0]
    y_cam = (ys.float() + 0.5 - K[1, 2]) * z / K[1, 1]
    cam_points = torch.stack((x_cam, y_cam, z, torch.ones_like(z)), dim=-1)
    c2w = torch.linalg.inv(w2c)
    world_points = (cam_points @ c2w.T)[:, :3]
    t0 = torch.full((tube_count,), float(centered_t0), dtype=torch.float32, device=device)
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


def initialize_world_tubes_with_static_fraction(
    bundle,
    *,
    tube_count: int,
    init_depth: float,
    seed: int,
    init_views: str,
    init_sampling: str,
    init_frames: str,
    init_frame_indices: list[int] | None,
    init_lambda_t: float,
    static_tube_fraction: float,
    static_init_lambda_t: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    if tube_count < 1:
        raise ValueError("tube_count must be positive")
    if not math.isfinite(static_tube_fraction) or static_tube_fraction < 0.0 or static_tube_fraction >= 1.0:
        raise ValueError("static_tube_fraction must be finite and in [0, 1)")
    if static_tube_fraction == 0.0 or tube_count == 1:
        init_x0, init_color, init_t0 = initialize_world_tubes_from_train_views(
            bundle,
            tube_count=tube_count,
            init_depth=init_depth,
            seed=seed,
            init_views=init_views,
            init_sampling=init_sampling,
            init_frames=init_frames,
            init_frame_indices=init_frame_indices,
        )
        init_lambda_t_values = torch.full(
            (tube_count,),
            float(init_lambda_t),
            dtype=torch.float32,
            device=init_x0.device,
        )
        return (
            init_x0,
            init_color,
            init_t0,
            init_lambda_t_values,
            {
                "static_tube_fraction": 0.0,
                "static_tube_count": 0,
                "dynamic_tube_count": tube_count,
                "static_init_lambda_t": float(static_init_lambda_t),
                "dynamic_init_lambda_t": float(init_lambda_t),
            },
        )

    static_tube_count = min(tube_count - 1, max(1, int(round(static_tube_fraction * tube_count))))
    dynamic_tube_count = tube_count - static_tube_count
    dynamic_x0, dynamic_color, dynamic_t0 = initialize_world_tubes_from_train_views(
        bundle,
        tube_count=dynamic_tube_count,
        init_depth=init_depth,
        seed=seed,
        init_views=init_views,
        init_sampling=init_sampling,
        init_frames=init_frames,
        init_frame_indices=init_frame_indices,
    )
    static_x0, static_color, static_t0 = initialize_world_tubes_from_train_views(
        bundle,
        tube_count=static_tube_count,
        init_depth=init_depth,
        seed=seed + 7919,
        init_views=init_views,
        init_sampling=init_sampling,
        init_frames="first",
        init_frame_indices=None,
    )
    init_x0 = torch.cat((dynamic_x0, static_x0), dim=0).contiguous()
    init_color = torch.cat((dynamic_color, static_color), dim=0).contiguous()
    init_t0 = torch.cat((dynamic_t0, static_t0), dim=0).contiguous()
    init_lambda_t_values = torch.cat(
        (
            torch.full(
                (dynamic_tube_count,),
                float(init_lambda_t),
                dtype=torch.float32,
                device=init_x0.device,
            ),
            torch.full(
                (static_tube_count,),
                float(static_init_lambda_t),
                dtype=torch.float32,
                device=init_x0.device,
            ),
        ),
        dim=0,
    ).contiguous()
    return (
        init_x0,
        init_color,
        init_t0,
        init_lambda_t_values,
        {
            "static_tube_fraction": float(static_tube_fraction),
            "static_tube_count": static_tube_count,
            "dynamic_tube_count": dynamic_tube_count,
            "static_init_lambda_t": float(static_init_lambda_t),
            "dynamic_init_lambda_t": float(init_lambda_t),
        },
    )


class WorldTubeModel(nn.Module):
    def __init__(
        self,
        *,
        init_x0: Tensor,
        init_color: Tensor,
        init_t0: Tensor,
        frames: int,
        init_precision_xy: float,
        init_lambda_t: float | Tensor,
        init_opacity: float,
        min_precision_xy: float,
        min_lambda_t: float,
        velocity_reg_weight: float,
        depth_velocity_reg_weight: float,
        position_reg_weight: float,
        static_tube_count: int = 0,
        static_velocity_reg_weight: float = 0.0,
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
        self.static_tube_count = int(static_tube_count)
        self.static_velocity_reg_weight = float(static_velocity_reg_weight)
        self.active_tube_count = tube_count
        if self.static_tube_count < 0 or self.static_tube_count > tube_count:
            raise ValueError("static_tube_count must be between 0 and tube_count")
        if self.static_velocity_reg_weight < 0.0:
            raise ValueError("static_velocity_reg_weight must be nonnegative")
        self.x0 = nn.Parameter(init_x0)
        self.velocity = nn.Parameter(torch.zeros_like(init_x0))
        precision = torch.full((tube_count, 2), float(init_precision_xy), dtype=torch.float32, device=init_x0.device)
        if isinstance(init_lambda_t, Tensor):
            if tuple(init_lambda_t.shape) != (tube_count,):
                raise ValueError(f"init_lambda_t tensor must have shape ({tube_count},)")
            lambda_t = init_lambda_t.to(dtype=torch.float32, device=init_x0.device)
        else:
            lambda_t = torch.full((tube_count,), float(init_lambda_t), dtype=torch.float32, device=init_x0.device)
        if bool((lambda_t <= self.min_lambda_t).any().item()):
            raise ValueError("init_lambda_t values must be greater than min_lambda_t")
        opacity = torch.full((tube_count,), float(init_opacity), dtype=torch.float32, device=init_x0.device)
        self.raw_precision_xy = nn.Parameter(_inv_softplus(precision - self.min_precision_xy))
        self.raw_lambda_t = nn.Parameter(_inv_softplus(lambda_t - self.min_lambda_t))
        self.raw_opacity = nn.Parameter(_logit(opacity / 0.99))
        self.raw_color = nn.Parameter(_logit(init_color))
        self.t0 = nn.Parameter(init_t0)

    def set_active_tube_count(self, count: int) -> None:
        if not 1 <= int(count) <= self.tube_count:
            raise ValueError(f"active tube count must be in [1, {self.tube_count}], got {count}")
        self.active_tube_count = int(count)

    def batch(self) -> WorldTubeBatch:
        active = slice(0, self.active_tube_count)
        return WorldTubeBatch(
            x0=self.x0[active],
            velocity=self.velocity[active],
            t0=self.t0[active],
            precision_xy=F.softplus(self.raw_precision_xy[active]) + self.min_precision_xy,
            lambda_t=F.softplus(self.raw_lambda_t[active]) + self.min_lambda_t,
            opacity=torch.sigmoid(self.raw_opacity[active]) * 0.99,
            color=torch.sigmoid(self.raw_color[active]),
        )

    def regularization(self) -> Tensor:
        reg = self.x0.new_tensor(0.0)
        if self.velocity_reg_weight:
            reg = reg + self.velocity_reg_weight * self.velocity[: self.active_tube_count].square().mean()
        if self.depth_velocity_reg_weight:
            reg = reg + self.depth_velocity_reg_weight * self.velocity[: self.active_tube_count, 2].square().mean()
        if self.position_reg_weight:
            reg = reg + self.position_reg_weight * self.x0[: self.active_tube_count].square().mean()
        if self.static_velocity_reg_weight and self.static_tube_count:
            static_start = self.tube_count - self.static_tube_count
            if self.active_tube_count > static_start:
                reg = reg + self.static_velocity_reg_weight * self.velocity[
                    static_start : self.active_tube_count
                ].square().mean()
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


def render_projected_sequence(
    projected: ProjectedTubeSequence,
    config: UVTRenderConfig,
    *,
    backend: str,
    reduction_mode: str = "index_add",
    sample_emission_mode: str = "atomic_append",
) -> RenderedSequence:
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
            reduction_mode=reduction_mode,
            sample_emission_mode=sample_emission_mode,
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


def epoch_view_shuffled_cycle_pair(
    left_values: list[int],
    right_values: list[int],
    *,
    step: int,
    seed: int,
) -> tuple[int, int]:
    if not left_values or not right_values:
        raise ValueError("epoch-view-shuffled cycle values must not be empty")
    right_step, left_offset = divmod(step, len(left_values))
    right_index = right_step % len(right_values)
    epoch = right_step // len(right_values)
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
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
    camera_sequence_mode: str,
    segment_frames: int,
    synthetic_pan_x: float,
    synthetic_pan_y: float,
    synthetic_dolly_z: float,
    synthetic_zoom: float,
    synthetic_principal_x: float,
    synthetic_principal_y: float,
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
    reduction_mode: str = "index_add",
    sample_emission_mode: str = "atomic_append",
    static_tube_fraction: float = 0.0,
    static_init_lambda_t: float = 0.02,
    static_velocity_reg_weight: float = 0.0,
    paper_protocol: dict[str, Any] | None = None,
) -> tuple[WorldTubeModel, dict[str, Any], list[dict[str, Any]]]:
    if loss_scope not in {"sampled_frame", "view_sequence", "temporal_window", "paper_batch"}:
        raise ValueError("loss_scope must be one of: sampled_frame, view_sequence, temporal_window, paper_batch")
    if backend != "metal_tile" and (reduction_mode != "index_add" or sample_emission_mode != "atomic_append"):
        raise ValueError("custom reduction/sample emission modes require backend=metal_tile")
    if reduction_mode in (
        "key_sort_scan_metal",
        "key_sort_compensated_scan_metal",
        "key_sort_segmented_metal",
    ) and sample_emission_mode not in (
        "with_keys",
        "tile_pair",
        "tile_pair_compensated",
        "tile_pair_grouped",
        "tile_pair_parallel",
        "tile_pair_scanline",
        "tile_pair_sharedsort",
        "tile_pair_target_bounds",
        "tile_pair_suffix",
    ):
        raise ValueError(
            "keyed sort reduction requires sample_emission_mode=with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, or tile_pair_suffix"
        )
    if sample_emission_mode in (
        "direct_atomic",
        "direct_fixedpoint",
        "direct_split_fixedpoint",
        "direct_serial",
        "tile_pair_atomic",
        "tile_pair_fixedpoint",
        "tile_pair_reduced",
        "tile_pair_reduced_parallel",
        "tile_pair_suffix_reduced",
    ) and reduction_mode != "index_add":
        raise ValueError(f"{sample_emission_mode} bypasses the reducer and requires reduction_mode=index_add")
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
    if static_velocity_reg_weight < 0.0:
        raise ValueError("static_velocity_reg_weight must be nonnegative")
    if camera_sequence_mode not in {"static_view", "dynamic_first_order", "projective_first_order", "segmented"}:
        raise ValueError(
            "camera_sequence_mode must be one of: static_view, dynamic_first_order, projective_first_order, segmented"
        )
    if segment_frames < 1:
        raise ValueError("segment_frames must be positive")
    synthetic_camera_active = any(
        (
            synthetic_pan_x,
            synthetic_pan_y,
            synthetic_dolly_z,
            synthetic_zoom,
            synthetic_principal_x,
            synthetic_principal_y,
        )
    )
    if (camera_sequence_mode != "static_view" or synthetic_camera_active) and camera_projection != "legacy_pinhole":
        raise ValueError("variable/synthetic camera STAR quality runs currently require camera_projection=legacy_pinhole")
    torch.manual_seed(seed)
    train_frames = bundle.train_frames
    device = bundle.train_K.device
    view_count, frames, _, height, width = train_frames.shape
    source_image_size = normalize_image_size((height, width))
    paper_values = paper_protocol or {}
    paper_enabled = bool(paper_values.get("enabled", False))
    if paper_enabled != (loss_scope == "paper_batch"):
        raise ValueError("paper_protocol.enabled and loss_scope='paper_batch' must be selected together")
    paper_stages = normalize_paper_stages(
        paper_values.get("stages") if paper_enabled else None,
        total_steps=max_steps,
        default_image_size=source_image_size,
        default_primitive_count=tube_count,
        default_frames_per_step=int(paper_values.get("frames_per_step", 1)),
    )
    if paper_stages[-1].image_size != source_image_size:
        raise ValueError("the final paper stage image size must match the loaded multicam image size")
    if paper_stages[-1].primitive_count != tube_count:
        raise ValueError("the final paper stage primitive_count must match tube_count")
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
    init_x0, init_color, init_t0, init_lambda_t_values, init_metadata = initialize_world_tubes_with_static_fraction(
        bundle,
        tube_count=tube_count,
        init_depth=init_depth,
        seed=seed,
        init_views=init_views,
        init_sampling=init_sampling,
        init_frames=init_frames,
        init_frame_indices=active_train_frames,
        init_lambda_t=init_lambda_t,
        static_tube_fraction=static_tube_fraction,
        static_init_lambda_t=static_init_lambda_t,
    )
    model = WorldTubeModel(
        init_x0=init_x0,
        init_color=init_color,
        init_t0=init_t0,
        frames=frames,
        init_precision_xy=init_precision_xy,
        init_lambda_t=init_lambda_t_values,
        init_opacity=init_opacity,
        min_precision_xy=min_precision_xy,
        min_lambda_t=min_lambda_t,
        velocity_reg_weight=velocity_reg_weight,
        depth_velocity_reg_weight=depth_velocity_reg_weight,
        position_reg_weight=position_reg_weight,
        static_tube_count=int(init_metadata["static_tube_count"]),
        static_velocity_reg_weight=static_velocity_reg_weight,
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
    paper_sampler = (
        SpacetimeEpochSampler(
            view_count=len(active_train_views),
            frame_indices=active_train_frames,
            batch_size=max(stage.frames_per_step for stage in paper_stages),
            same_time_count=int(paper_values.get("same_time_count", 1)),
            local_time_count=int(paper_values.get("local_time_count", 0)),
            local_time_radius=int(paper_values.get("local_time_radius", 0)),
            seed=seed + int(paper_values.get("sampler_seed_offset", 7001)),
        )
        if paper_enabled
        else None
    )
    paper_costs = PaperCostTracker()
    paper_phase_timer = PaperPhaseTimer(device)
    paper_memory_sampler = DeviceMemorySampler(device)
    paper_memory_sampler.start()
    paper_stage_cache: dict[str, tuple[Tensor, UVTRenderConfig]] = {}

    def paper_stage_payload(stage) -> tuple[Tensor, UVTRenderConfig]:
        cached = paper_stage_cache.get(stage.label)
        if cached is not None:
            return cached
        stage_K = scale_intrinsics(bundle.train_K, source=source_image_size, target=stage.image_size)
        stage_config = UVTRenderConfig(
            height=stage.image_size.height,
            width=stage.image_size.width,
            frames=frames,
            tile_x=full_config.tile_x,
            tile_y=full_config.tile_y,
            tile_t=full_config.tile_t,
            tile_capacity=full_config.tile_capacity,
            alpha_threshold=full_config.alpha_threshold,
            transmittance_threshold=full_config.transmittance_threshold,
            background=full_config.background,
            max_alpha=full_config.max_alpha,
        )
        cached = (stage_K, stage_config)
        paper_stage_cache[stage.label] = cached
        return cached
    started_at = time.perf_counter()
    logs = []
    checkpoints: list[dict[str, Any]] = []
    last_finite_state = snapshot_world_tube_state(model)
    last_finite_step = 0
    stopped_reason: str | None = None
    stopped_step: int | None = None

    def project_for_view(
        *,
        view: int,
        render_cfg: UVTRenderConfig,
        frame_start_value: int,
        lens_model_value: str,
        distortion_value: Tensor | None,
        K_value: Tensor | None = None,
    ) -> ProjectedTubeSequence:
        full_frame_count = int(frames)
        selected_K = bundle.train_K if K_value is None else K_value
        if camera_sequence_mode == "static_view" and not synthetic_camera_active:
            return project_world_tube_sequence(
                model,
                select_view_K(selected_K, view),
                select_view_w2c(bundle.train_w2c, view),
                render_cfg,
                camera_projection=camera_projection,
                lens_model=lens_model_value,
                distortion=distortion_value,
                full_frames=full_frame_count if int(render_cfg.frames) != full_frame_count else None,
                frame_start=frame_start_value,
            )
        K_seq, w2c_seq = camera_sequences_for_view(
            selected_K,
            bundle.train_w2c,
            view=view,
            frames=full_frame_count,
            view_count=view_count,
            synthetic_pan_x=synthetic_pan_x,
            synthetic_pan_y=synthetic_pan_y,
            synthetic_dolly_z=synthetic_dolly_z,
            synthetic_zoom=synthetic_zoom,
            synthetic_principal_x=synthetic_principal_x,
            synthetic_principal_y=synthetic_principal_y,
        )
        return project_world_tube_sequence_camera_mode(
            model=model,
            K_seq=K_seq,
            w2c_seq=w2c_seq,
            config=render_cfg,
            full_frames=full_frame_count,
            frame_start=frame_start_value,
            camera_sequence_mode=camera_sequence_mode,
            segment_frames=segment_frames,
        )

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
        paper_stage = paper_stage_for_step(paper_stages, step)
        model.set_active_tube_count(paper_stage.primitive_count)
        current_lr = (lr * lr_decay_factor if lr_decay_step > 0 and step >= lr_decay_step else lr) * paper_stage.lr_multiplier
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        frame_override: int | None = None
        window_start_override: int | None = None
        paper_batch = paper_sampler.next_batch(paper_stage.frames_per_step) if paper_sampler is not None else None
        if paper_batch is not None:
            view = paper_batch.samples[0].view_index
        elif train_schedule in {
            "shuffled_cycle",
            "reshuffled_cycle",
            "phase_rotated_cycle",
            "view_shuffled_cycle",
            "epoch_view_shuffled_cycle",
        }:
            if loss_scope == "sampled_frame":
                if train_schedule == "shuffled_cycle":
                    view, frame_override = shuffled_frame_pairs[int(step % len(shuffled_frame_pairs))]
                elif train_schedule == "reshuffled_cycle":
                    view, frame_override = reshuffled_cycle_item(frame_pairs, step=step, seed=seed + 2003)
                elif train_schedule == "phase_rotated_cycle":
                    view, frame_override = phase_rotated_cycle_item(frame_pairs, step=step, seed=seed + 2003)
                elif train_schedule == "view_shuffled_cycle":
                    view, frame_override = view_shuffled_cycle_pair(
                        active_train_views,
                        active_train_frames,
                        step=step,
                        seed=seed + 2003,
                    )
                else:
                    view, frame_override = epoch_view_shuffled_cycle_pair(
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
                elif train_schedule == "view_shuffled_cycle":
                    view, window_start_override = view_shuffled_cycle_pair(
                        active_train_views,
                        active_window_starts,
                        step=step,
                        seed=seed + 3001,
                    )
                else:
                    view, window_start_override = epoch_view_shuffled_cycle_pair(
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
        step_K = bundle.train_K
        step_full_config = full_config
        step_train_render_config = full_config
        if paper_batch is not None:
            step_K, step_full_config = paper_stage_payload(paper_stage)
        optimizer.zero_grad(set_to_none=True)
        paper_forward_started_at = paper_phase_timer.start("forward")
        if loss_scope == "paper_batch":
            if paper_batch is None:
                raise RuntimeError("paper_batch loss requires an active paper sampler")
            predictions = []
            targets = []
            projected_sequences = []
            selected_frame_config = UVTRenderConfig(
                height=step_full_config.height,
                width=step_full_config.width,
                frames=1,
                tile_x=step_full_config.tile_x,
                tile_y=step_full_config.tile_y,
                tile_t=step_full_config.tile_t,
                tile_capacity=step_full_config.tile_capacity,
                alpha_threshold=step_full_config.alpha_threshold,
                transmittance_threshold=step_full_config.transmittance_threshold,
                background=step_full_config.background,
                max_alpha=step_full_config.max_alpha,
            )
            for sample in paper_batch.samples:
                batch_lens, batch_distortion = select_lens(
                    bundle.train_lens_models,
                    bundle.train_distortions,
                    sample.view_index,
                    camera_projection=camera_projection,
                )
                projected = project_for_view(
                    view=sample.view_index,
                    render_cfg=selected_frame_config,
                    frame_start_value=sample.frame_index,
                    lens_model_value=batch_lens,
                    distortion_value=batch_distortion,
                    K_value=step_K,
                )
                projected_sequences.append(projected)
                rendered = render_projected_sequence(
                    projected,
                    selected_frame_config,
                    backend=backend,
                    reduction_mode=reduction_mode,
                    sample_emission_mode=sample_emission_mode,
                )
                predictions.append(rendered.rgb[0])
                targets.append(train_frames[sample.view_index, sample.frame_index])
            rendered_active = torch.stack(predictions)
            target_active = (
                resize_video_frames(torch.stack(targets), paper_stage.image_size)
                .to(device=device, dtype=torch.float32)
                .permute(0, 2, 3, 1)
            )
            recon_loss = robust_l1(rendered_active - target_active)
            multiscale_loss = (
                downsampled_robust_l1(rendered_active, target_active, multiscale_loss_factor)
                if multiscale_loss_weight > 0.0
                else model.x0.new_tensor(0.0)
            )
            crop_loss = (
                crop_robust_l1(rendered_active, target_active, crop_loss_size, step)
                if crop_loss_weight > 0.0
                else model.x0.new_tensor(0.0)
            )
            step_target_frames = len(paper_batch.samples)
            step_rasterized_frames = len(paper_batch.samples)
            step_train_render_config = selected_frame_config
        elif loss_scope == "sampled_frame":
            frame = (
                frame_override
                if frame_override is not None
                else select_train_frame(step, len(active_train_views), active_train_frames, device, train_schedule)
            )
            projected = project_for_view(
                view=view,
                render_cfg=full_config,
                frame_start_value=0,
                lens_model_value=lens_model,
                distortion_value=distortion,
            )
            rendered = render_projected_sequence(
                projected,
                full_config,
                backend=backend,
                reduction_mode=reduction_mode,
                sample_emission_mode=sample_emission_mode,
            )
            target = train_frames[view, frame].permute(1, 2, 0)
            recon_loss = robust_l1(rendered.rgb[frame] - target)
            multiscale_loss = (
                downsampled_robust_l1(rendered.rgb[frame], target, multiscale_loss_factor)
                if multiscale_loss_weight > 0.0
                else model.x0.new_tensor(0.0)
            )
            crop_loss = (
                crop_robust_l1(rendered.rgb[frame], target, crop_loss_size, step)
                if crop_loss_weight > 0.0
                else model.x0.new_tensor(0.0)
            )
            projected_sequences = [projected]
            step_target_frames = 1
            step_rasterized_frames = frames
        elif loss_scope == "view_sequence":
            projected = project_for_view(
                view=view,
                render_cfg=full_config,
                frame_start_value=0,
                lens_model_value=lens_model,
                distortion_value=distortion,
            )
            rendered = render_projected_sequence(
                projected,
                full_config,
                backend=backend,
                reduction_mode=reduction_mode,
                sample_emission_mode=sample_emission_mode,
            )
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
                else model.x0.new_tensor(0.0)
            )
            crop_loss = (
                crop_robust_l1(rendered_active, target_active, crop_loss_size, step)
                if crop_loss_weight > 0.0
                else model.x0.new_tensor(0.0)
            )
            projected_sequences = [projected]
            step_target_frames = len(active_train_frames)
            step_rasterized_frames = frames
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
            projected = project_for_view(
                view=view,
                render_cfg=window_config,
                frame_start_value=frame_start,
                lens_model_value=lens_model,
                distortion_value=distortion,
            )
            rendered = render_projected_sequence(
                projected,
                window_config,
                backend=backend,
                reduction_mode=reduction_mode,
                sample_emission_mode=sample_emission_mode,
            )
            target = train_frames[view, frame_start : frame_start + window_frames].permute(0, 2, 3, 1).contiguous()
            recon_loss = robust_l1(rendered.rgb - target)
            multiscale_loss = (
                downsampled_robust_l1(rendered.rgb, target, multiscale_loss_factor)
                if multiscale_loss_weight > 0.0
                else model.x0.new_tensor(0.0)
            )
            crop_loss = (
                crop_robust_l1(rendered.rgb, target, crop_loss_size, step)
                if crop_loss_weight > 0.0
                else model.x0.new_tensor(0.0)
            )
            projected_sequences = [projected]
            step_target_frames = window_frames
            step_rasterized_frames = window_frames
        sequence_consistency_loss = model.x0.new_tensor(0.0)
        consistency_due = (
            sequence_consistency_weight > 0.0
            and sequence_consistency_every_steps > 0
            and (step + 1) % sequence_consistency_every_steps == 0
        )
        if consistency_due:
            if consistency_window_frames > 0 and consistency_window_frames < frames:
                consistency_config = UVTRenderConfig(
                    height=step_full_config.height,
                    width=step_full_config.width,
                    frames=consistency_window_frames,
                    tile_x=step_full_config.tile_x,
                    tile_y=step_full_config.tile_y,
                    tile_t=step_full_config.tile_t,
                    tile_capacity=step_full_config.tile_capacity,
                    alpha_threshold=step_full_config.alpha_threshold,
                    transmittance_threshold=step_full_config.transmittance_threshold,
                    background=step_full_config.background,
                    max_alpha=step_full_config.max_alpha,
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
                elif train_schedule in {"view_shuffled_cycle", "epoch_view_shuffled_cycle"}:
                    consistency_start = consistency_window_starts[
                        int((step // len(active_train_views)) % len(consistency_window_starts))
                    ]
                else:
                    raise ValueError(f"Unsupported train_schedule for consistency: {train_schedule}")
            else:
                consistency_config = step_full_config
                consistency_start = 0
            sequence_projected = project_for_view(
                view=view,
                render_cfg=consistency_config,
                frame_start_value=consistency_start,
                lens_model_value=lens_model,
                distortion_value=distortion,
                K_value=step_K,
            )
            sequence_rendered = render_projected_sequence(
                sequence_projected,
                consistency_config,
                backend=backend,
                reduction_mode=reduction_mode,
                sample_emission_mode=sample_emission_mode,
            )
            if consistency_config.frames == frames:
                sequence_target = (
                    resize_video_frames(
                        train_frames[view].index_select(0, active_train_frame_tensor.to(train_frames.device)),
                        paper_stage.image_size,
                    )
                    .to(device=device, dtype=torch.float32)
                    .permute(0, 2, 3, 1)
                )
                sequence_consistency_loss = robust_l1(
                    sequence_rendered.rgb.index_select(0, active_train_frame_tensor)
                    - sequence_target
                )
            else:
                sequence_target = (
                    resize_video_frames(
                        train_frames[
                            view,
                            consistency_start : consistency_start + consistency_window_frames,
                        ],
                        paper_stage.image_size,
                    )
                    .to(device=device, dtype=torch.float32)
                    .permute(0, 2, 3, 1)
                )
                sequence_consistency_loss = robust_l1(sequence_rendered.rgb - sequence_target)
        train_render_config = (
            window_config
            if loss_scope == "temporal_window"
            else (step_train_render_config if loss_scope == "paper_batch" else full_config)
        )
        model_reg = model.regularization()
        projected_reg_rows = [
            projected_regularization(
                projected_item,
                train_render_config,
                tile_load_weight=tile_load_reg_weight,
                tile_load_target=tile_load_target,
                depth_slope_weight=depth_slope_reg_weight,
                depth_margin_weight=depth_margin_reg_weight,
                depth_margin=depth_margin,
            )
            for projected_item in projected_sequences
        ]
        projected_reg = torch.stack([row[0] for row in projected_reg_rows]).mean()
        projected_reg_metrics = {
            key: torch.stack([row[1][key] for row in projected_reg_rows]).mean()
            for key in projected_reg_rows[0][1]
        }
        consistency_term = float(sequence_consistency_weight) * sequence_consistency_loss
        multiscale_term = float(multiscale_loss_weight) * multiscale_loss
        crop_term = float(crop_loss_weight) * crop_loss
        loss = recon_loss + crop_term + multiscale_term + consistency_term + model_reg + projected_reg
        paper_phase_timer.stop("forward", paper_forward_started_at)
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
        paper_backward_started_at = paper_phase_timer.start("backward")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        paper_phase_timer.stop("backward", paper_backward_started_at)
        paper_optimizer_started_at = paper_phase_timer.start("optimizer")
        optimizer.step()
        paper_phase_timer.stop("optimizer", paper_optimizer_started_at)
        paper_costs.record(
            stage=paper_stage,
            target_frames=step_target_frames,
            rasterized_frames=step_rasterized_frames,
        )
        elapsed_after_step = time.perf_counter() - started_at
        if checkpoint_every_steps <= 0:
            last_finite_state = snapshot_world_tube_state(model)
            last_finite_step = completed_step
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
            logs[-1].update(
                {
                    "paper_stage": paper_stage.label,
                    "paper_height": paper_stage.image_size.height,
                    "paper_width": paper_stage.image_size.width,
                    "paper_active_tubes": model.active_tube_count,
                    "paper_epoch": None if paper_batch is None else paper_batch.epoch,
                    "paper_batch_index": None if paper_batch is None else paper_batch.batch_index,
                    "paper_epoch_complete": None if paper_batch is None else paper_batch.completes_epoch,
                }
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
    paper_memory_sampler.stop()
    model.set_active_tube_count(tube_count)
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
            **init_metadata,
            "static_velocity_reg": static_velocity_reg_weight,
            "sequence_consistency_every_steps": sequence_consistency_every_steps,
            "sequence_consistency_frames": sequence_consistency_frames,
            "sequence_consistency_weight": sequence_consistency_weight,
            "multiscale_loss_weight": multiscale_loss_weight,
            "multiscale_loss_factor": multiscale_loss_factor,
            "crop_loss_weight": crop_loss_weight,
            "crop_loss_size": crop_loss_size,
            "stopped_reason": stopped_reason,
            "stopped_step": stopped_step,
            "paper_protocol": {
                "enabled": paper_enabled,
                "kernel": MetalKernelSpec(
                    representation="world_tubes",
                    family="star_uvt",
                    forward=f"{backend}_selected_time" if paper_enabled else backend,
                    backward=f"{sample_emission_mode}+{reduction_mode}",
                    deterministic=sample_emission_mode != "direct_atomic",
                    implementation="third_party/fast-mac-gsplat/variants/star_uvt_v0",
                ).as_dict(),
                "sampling": {
                    "mode": "spacetime_epoch" if paper_enabled else train_schedule,
                    "same_time_count": int(paper_values.get("same_time_count", 1)),
                    "local_time_count": int(paper_values.get("local_time_count", 0)),
                    "local_time_radius": int(paper_values.get("local_time_radius", 0)),
                },
                "stages": [stage.as_dict() for stage in paper_stages],
                "cost": paper_costs.snapshot(
                    model=model,
                    optimizer=optimizer,
                    elapsed_s=train_elapsed,
                    memory=paper_memory_sampler.stats(),
                ).as_dict(),
                "timing": paper_phase_timer.snapshot(train_wall_s=train_elapsed),
            },
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
    paper_protocol: dict[str, Any] | None = None,
) -> tuple[FreeDynamic3DGS, SplatRenderConfig, dict[str, Any]]:
    torch.manual_seed(seed)
    train_video = bundle.train_frames
    device = bundle.train_K.device
    view_count, frames, _, height, width = train_video.shape
    source_image_size = normalize_image_size((height, width))
    paper_values = paper_protocol or {}
    paper_enabled = bool(paper_values.get("enabled", False))
    paper_stages = normalize_paper_stages(
        paper_values.get("stages") if paper_enabled else None,
        total_steps=max_steps,
        default_image_size=source_image_size,
        default_primitive_count=splat_count,
        default_frames_per_step=int(paper_values.get("frames_per_step", 1)),
    )
    if paper_stages[-1].image_size != source_image_size:
        raise ValueError("the final paper stage image size must match the loaded multicam image size")
    if paper_stages[-1].primitive_count != splat_count:
        raise ValueError("the final paper stage primitive_count must match splat_count")
    init_xyz, init_rgb = initialize_material_points_from_first_frame(
        video=train_video[0, :1].permute(0, 2, 3, 1).contiguous().to(device),
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
    ).to(device)
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
    paper_sampler = (
        SpacetimeEpochSampler(
            view_count=view_count,
            frame_indices=range(frames),
            batch_size=max(stage.frames_per_step for stage in paper_stages),
            same_time_count=int(paper_values.get("same_time_count", 1)),
            local_time_count=int(paper_values.get("local_time_count", 0)),
            local_time_radius=int(paper_values.get("local_time_radius", 0)),
            seed=seed + int(paper_values.get("sampler_seed_offset", 7001)),
        )
        if paper_enabled
        else None
    )
    paper_costs = PaperCostTracker()
    paper_phase_timer = PaperPhaseTimer(device)
    paper_memory_sampler = DeviceMemorySampler(device)
    paper_memory_sampler.start()
    paper_stage_cache: dict[str, tuple[Tensor, SplatRenderConfig]] = {}

    def splat_stage_payload(stage) -> tuple[Tensor, SplatRenderConfig]:
        cached = paper_stage_cache.get(stage.label)
        if cached is not None:
            return cached
        stage_K = scale_intrinsics(bundle.train_K, source=source_image_size, target=stage.image_size)
        stage_render_cfg = SplatRenderConfig(
            height=stage.image_size.height,
            width=stage.image_size.width,
            renderer=renderer,
            tile_size=16 if renderer == "fast_mac" else 8,
            bound_scale=3.0,
            alpha_threshold=1.0 / 255.0,
            near_plane=1.0e-3,
            camera_projection="camera_model" if camera_projection == "dataset_lens" else "legacy_pinhole",
        )
        cached = (stage_K, stage_render_cfg)
        paper_stage_cache[stage.label] = cached
        return cached

    started_at = time.perf_counter()
    logs = []
    step = 0
    while step < max_steps:
        elapsed = time.perf_counter() - started_at
        if step > 0 and elapsed >= train_seconds:
            break
        paper_stage = paper_stage_for_step(paper_stages, step)
        model.set_active_splat_count(paper_stage.primitive_count)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * paper_stage.lr_multiplier
        paper_batch = paper_sampler.next_batch(paper_stage.frames_per_step) if paper_sampler is not None else None
        if paper_batch is None:
            sample_pairs = [
                (
                    int(torch.randint(0, view_count, (1,), device=device).item()),
                    int(torch.randint(0, frames, (1,), device=device).item()),
                )
            ]
            stage_video, stage_K, stage_render_cfg = train_video, bundle.train_K, render_cfg
        else:
            sample_pairs = [(sample.view_index, sample.frame_index) for sample in paper_batch.samples]
            stage_video = train_video
            stage_K, stage_render_cfg = splat_stage_payload(paper_stage)
        optimizer.zero_grad(set_to_none=True)
        paper_forward_started_at = paper_phase_timer.start("forward")
        images = []
        target_rows = []
        for view, frame in sample_pairs:
            lens_model, distortion = select_lens(
                bundle.train_lens_models,
                bundle.train_distortions,
                view,
                camera_projection=camera_projection,
            )
            camera = camera_from_K_w2c_lens(
                select_K_for_view_time(stage_K, view=view, t=frame, view_count=view_count),
                select_w2c_for_view_time(bundle.train_w2c, view=view, t=frame),
                lens_model=lens_model,
                distortion=distortion,
            )
            images.append(
                render_gaussian_frame(
                    model.frame(frame),
                    camera,
                    height=stage_render_cfg.height,
                    width=stage_render_cfg.width,
                    mode=stage_render_cfg.renderer,
                    tile_size=stage_render_cfg.tile_size,
                    bound_scale=stage_render_cfg.bound_scale,
                    alpha_threshold=stage_render_cfg.alpha_threshold,
                    near_plane=stage_render_cfg.near_plane,
                    camera_projection=stage_render_cfg.camera_projection,
                ).permute(1, 2, 0)
            )
            target_rows.append(stage_video[view, frame])
        target_batch = (
            resize_video_frames(torch.stack(target_rows), paper_stage.image_size)
            .to(device=device, dtype=torch.float32)
            .permute(0, 2, 3, 1)
        )
        loss = robust_l1(torch.stack(images) - target_batch)
        loss = loss + 1.0e-4 * model.scale_loss() + 1.0e-3 * model.temporal_smoothness_loss()
        paper_phase_timer.stop("forward", paper_forward_started_at)
        paper_backward_started_at = paper_phase_timer.start("backward")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        paper_phase_timer.stop("backward", paper_backward_started_at)
        paper_optimizer_started_at = paper_phase_timer.start("optimizer")
        optimizer.step()
        paper_phase_timer.stop("optimizer", paper_optimizer_started_at)
        paper_costs.record(
            stage=paper_stage,
            target_frames=len(sample_pairs),
            rasterized_frames=len(sample_pairs),
        )
        if step == 0 or (step + 1) % 10 == 0:
            logs.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "elapsed_s": time.perf_counter() - started_at,
                    "paper_stage": paper_stage.label,
                    "paper_height": paper_stage.image_size.height,
                    "paper_width": paper_stage.image_size.width,
                    "paper_active_splats": model.active_splat_count,
                    "paper_epoch": None if paper_batch is None else paper_batch.epoch,
                    "paper_batch_index": None if paper_batch is None else paper_batch.batch_index,
                }
            )
        step += 1
    train_elapsed = time.perf_counter() - started_at
    paper_memory_sampler.stop()
    model.set_active_splat_count(splat_count)
    return model, render_cfg, {
        "steps": step,
        "train_loop_elapsed_s": train_elapsed,
        "paper_protocol": {
            "enabled": paper_enabled,
            "kernel": MetalKernelSpec(
                representation="dynamic_3dgs",
                family="fast_mac",
                forward=renderer,
                backward="fast_mac_autograd",
                deterministic=False,
                implementation="third_party/fast-mac-gsplat",
            ).as_dict(),
            "sampling": {
                "mode": "spacetime_epoch" if paper_enabled else "iid_with_replacement",
                "same_time_count": int(paper_values.get("same_time_count", 1)),
                "local_time_count": int(paper_values.get("local_time_count", 0)),
                "local_time_radius": int(paper_values.get("local_time_radius", 0)),
            },
            "stages": [stage.as_dict() for stage in paper_stages],
            "cost": paper_costs.snapshot(
                model=model,
                optimizer=optimizer,
                elapsed_s=train_elapsed,
                memory=paper_memory_sampler.stats(),
            ).as_dict(),
            "timing": paper_phase_timer.snapshot(train_wall_s=train_elapsed),
        },
        "logs": logs,
    }


@torch.no_grad()
def eval_world_tubes(
    model: WorldTubeModel,
    bundle,
    *,
    backend: str,
    camera_projection: str,
    camera_sequence_mode: str,
    segment_frames: int,
    synthetic_pan_x: float,
    synthetic_pan_y: float,
    synthetic_dolly_z: float,
    synthetic_zoom: float,
    synthetic_principal_x: float,
    synthetic_principal_y: float,
    render_config: UVTRenderConfig,
    frame_metric_splits: dict[str, list[int]] | None = None,
    chunk_frames: int = 4,
    media_max_frames: int = 32,
) -> dict[str, Any]:
    _, frames, _, height, width = bundle.train_frames.shape
    config = render_config
    if config.height != height or config.width != width or config.frames != frames:
        raise ValueError("render_config dimensions must match bundle train frames")
    synthetic_camera_active = any(
        (
            synthetic_pan_x,
            synthetic_pan_y,
            synthetic_dolly_z,
            synthetic_zoom,
            synthetic_principal_x,
            synthetic_principal_y,
        )
    )
    if camera_sequence_mode not in {"static_view", "dynamic_first_order", "projective_first_order", "segmented"}:
        raise ValueError(
            "camera_sequence_mode must be one of: static_view, dynamic_first_order, projective_first_order, segmented"
        )
    if (camera_sequence_mode != "static_view" or synthetic_camera_active) and camera_projection != "legacy_pinhole":
        raise ValueError("variable/synthetic camera STAR quality eval currently requires camera_projection=legacy_pinhole")
    if chunk_frames < 1:
        raise ValueError("eval chunk_frames must be positive")
    if media_max_frames < 1:
        raise ValueError("eval media_max_frames must be positive")

    def render_eval_chunk(
        *,
        K_all: Tensor,
        w2c_all: Tensor,
        view: int,
        view_count: int,
        lens_model_value: str,
        distortion_value: Tensor | None,
        frame_start: int,
        frame_stop: int,
    ) -> RenderedSequence:
        chunk_config = UVTRenderConfig(
            height=config.height,
            width=config.width,
            frames=frame_stop - frame_start,
            tile_x=config.tile_x,
            tile_y=config.tile_y,
            tile_t=config.tile_t,
            tile_capacity=config.tile_capacity,
            alpha_threshold=config.alpha_threshold,
            transmittance_threshold=config.transmittance_threshold,
            background=config.background,
            max_alpha=config.max_alpha,
        )
        if camera_sequence_mode == "static_view" and not synthetic_camera_active:
            return render_world_tube_sequence(
                model,
                select_view_K(K_all, view),
                select_view_w2c(w2c_all, view),
                chunk_config,
                backend=backend,
                camera_projection=camera_projection,
                lens_model=lens_model_value,
                distortion=distortion_value,
                full_frames=frames,
                frame_start=frame_start,
            )
        K_seq, w2c_seq = camera_sequences_for_view(
            K_all,
            w2c_all,
            view=view,
            frames=frames,
            view_count=view_count,
            synthetic_pan_x=synthetic_pan_x,
            synthetic_pan_y=synthetic_pan_y,
            synthetic_dolly_z=synthetic_dolly_z,
            synthetic_zoom=synthetic_zoom,
            synthetic_principal_x=synthetic_principal_x,
            synthetic_principal_y=synthetic_principal_y,
        )
        projected = project_world_tube_sequence_camera_mode(
            model=model,
            K_seq=K_seq,
            w2c_seq=w2c_seq,
            config=chunk_config,
            full_frames=frames,
            frame_start=frame_start,
            camera_sequence_mode=camera_sequence_mode,
            segment_frames=segment_frames,
        )
        return render_projected_sequence(projected, chunk_config, backend=backend)

    device = next(model.parameters()).device
    render_started = time.perf_counter()

    def eval_split(
        *,
        split: str,
        frames_tensor: Tensor,
        K_all: Tensor,
        w2c_all: Tensor,
        lens_models: list[str] | None,
        distortions: Tensor | None,
        split_metrics: dict[str, list[int]] | None = None,
    ) -> tuple[list, list, list, dict[str, list[dict[str, float]]]]:
        rows = []
        metrics_rows = []
        render_times = []
        selected = media_frame_positions(frames, media_max_frames)
        split_rows: dict[str, list[dict[str, float]]] = {
            name: [] for name, indices in (split_metrics or {}).items() if indices
        }
        for view in range(int(frames_tensor.shape[0])):
            lens_model, distortion = select_lens(
                lens_models,
                distortions,
                view,
                camera_projection=camera_projection,
            )
            accumulator = VideoMetricAccumulator()
            frame_accumulators = {name: VideoMetricAccumulator() for name in split_rows}
            lpips_sum = 0.0
            lpips_count = 0
            view_render_elapsed = 0.0
            media_targets: list[Tensor] = []
            media_renders: list[Tensor] = []
            media_alphas: list[Tensor] = []
            for start in range(0, frames, chunk_frames):
                stop = min(start + chunk_frames, frames)
                rendered, render_elapsed = time_render_sequence(
                    device,
                    lambda start=start, stop=stop: render_eval_chunk(
                        K_all=K_all,
                        w2c_all=w2c_all,
                        view=view,
                        view_count=int(frames_tensor.shape[0]),
                        lens_model_value=lens_model,
                        distortion_value=distortion,
                        frame_start=start,
                        frame_stop=stop,
                    ),
                )
                view_render_elapsed += render_elapsed
                target = frames_tensor[view, start:stop].permute(0, 2, 3, 1).contiguous().cpu()
                rendered = RenderedSequence(
                    rgb=rendered.rgb.detach().cpu(),
                    alpha=rendered.alpha.detach().cpu(),
                )
                accumulator.update(rendered.rgb, target)
                for name in frame_accumulators:
                    indices = split_metrics[name]
                    local_positions = [index - start for index in indices if start <= index < stop]
                    if local_positions:
                        local = torch.tensor(local_positions, dtype=torch.long)
                        frame_accumulators[name].update(
                            rendered.rgb.index_select(0, local),
                            target.index_select(0, local),
                        )
                if split == "heldout":
                    count = stop - start
                    lpips_sum += video_lpips(rendered.rgb, target) * count
                    lpips_count += count
                append_chunk_media(
                    start=start,
                    stop=stop,
                    selected=selected,
                    target=target,
                    rendered=rendered.rgb,
                    alpha=rendered.alpha,
                    targets_out=media_targets,
                    rendered_out=media_renders,
                    alpha_out=media_alphas,
                )
                del rendered, target
            row_metrics = accumulator.metrics()
            if split == "heldout":
                row_metrics["eval_lpips"] = lpips_sum / float(lpips_count)
            metrics_rows.append(row_metrics)
            render_times.append(view_render_elapsed)
            for name, frame_accumulator in frame_accumulators.items():
                split_rows[name].append(frame_accumulator.metrics())
            rows.append(
                (
                    torch.cat(media_targets, dim=0),
                    RenderedSequence(
                        rgb=torch.cat(media_renders, dim=0),
                        alpha=torch.cat(media_alphas, dim=0),
                    ),
                )
            )
        return rows, metrics_rows, render_times, split_rows

    train_rows, train_metrics, train_render_times, train_frame_split_metrics = eval_split(
        split="train",
        frames_tensor=bundle.train_frames,
        K_all=bundle.train_K,
        w2c_all=bundle.train_w2c,
        lens_models=bundle.train_lens_models,
        distortions=bundle.train_distortions,
        split_metrics=frame_metric_splits,
    )
    heldout_rows: list = []
    heldout_metrics: list = []
    heldout_render_times: list = []
    if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
        heldout_rows, heldout_metrics, heldout_render_times, _ = eval_split(
            split="heldout",
            frames_tensor=bundle.heldout_frames,
            K_all=bundle.heldout_K,
            w2c_all=bundle.heldout_w2c,
            lens_models=bundle.heldout_lens_models,
            distortions=bundle.heldout_distortions,
        )
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
    camera_sequence_mode: str,
    segment_frames: int,
    synthetic_pan_x: float,
    synthetic_pan_y: float,
    synthetic_dolly_z: float,
    synthetic_zoom: float,
    synthetic_principal_x: float,
    synthetic_principal_y: float,
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
            camera_sequence_mode=camera_sequence_mode,
            segment_frames=segment_frames,
            synthetic_pan_x=synthetic_pan_x,
            synthetic_pan_y=synthetic_pan_y,
            synthetic_dolly_z=synthetic_dolly_z,
            synthetic_zoom=synthetic_zoom,
            synthetic_principal_x=synthetic_principal_x,
            synthetic_principal_y=synthetic_principal_y,
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
    camera_sequence_mode: str,
    segment_frames: int,
    render_config: UVTRenderConfig,
) -> dict[str, Any]:
    if next(model.parameters()).device.type != "mps":
        return {"skipped": "Metal stats require MPS tensors."}
    _, frames, _, height, width = bundle.train_frames.shape
    config = render_config
    if config.height != height or config.width != width or config.frames != frames:
        raise ValueError("render_config dimensions must match bundle train frames")

    def row(
        split: str,
        camera_name: str,
        K_all: Tensor,
        w2c_all: Tensor,
        view: int,
        view_count: int,
        lens_model: str,
        distortion: Tensor | None,
    ) -> dict[str, Any]:
        if camera_sequence_mode == "static_view":
            projected = project_world_tube_sequence(
                model,
                select_view_K(K_all, view),
                select_view_w2c(w2c_all, view),
                config,
                camera_projection=camera_projection,
                lens_model=lens_model,
                distortion=distortion,
            )
        else:
            K_seq, w2c_seq = camera_sequences_for_view(
                K_all,
                w2c_all,
                view=view,
                frames=frames,
                view_count=view_count,
                synthetic_pan_x=0.0,
                synthetic_pan_y=0.0,
                synthetic_dolly_z=0.0,
                synthetic_zoom=0.0,
                synthetic_principal_x=0.0,
                synthetic_principal_y=0.0,
            )
            projected = project_world_tube_sequence_camera_mode(
                model=model,
                K_seq=K_seq,
                w2c_seq=w2c_seq,
                config=config,
                full_frames=frames,
                frame_start=0,
                camera_sequence_mode=camera_sequence_mode,
                segment_frames=segment_frames,
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
        return {
            "split": split,
            "camera": camera_name,
            "stats": {
                **result.stats.__dict__,
                "projected_trace_count": int(projected.ma.shape[0]),
            },
        }

    rows = [
        row(
            "train",
            name,
            bundle.train_K,
            bundle.train_w2c,
            view,
            bundle.train_view_count,
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
                bundle.heldout_K,
                bundle.heldout_w2c,
                view,
                bundle.heldout_view_count,
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
    chunk_frames: int = 4,
    media_max_frames: int = 32,
) -> dict[str, Any]:
    if chunk_frames < 1:
        raise ValueError("eval chunk_frames must be positive")
    if media_max_frames < 1:
        raise ValueError("eval media_max_frames must be positive")
    render_started = time.perf_counter()
    device = next(model.parameters()).device

    def eval_split(split: str, frames_tensor: Tensor) -> tuple[list, list, list]:
        rows = []
        metrics_rows = []
        render_times = []
        selected = media_frame_positions(bundle.frame_count, media_max_frames)
        for view in range(int(frames_tensor.shape[0])):
            accumulator = VideoMetricAccumulator()
            lpips_sum = 0.0
            lpips_count = 0
            view_render_elapsed = 0.0
            media_targets: list[Tensor] = []
            media_renders: list[Tensor] = []
            media_alphas: list[Tensor] = []
            for start in range(0, bundle.frame_count, chunk_frames):
                stop = min(start + chunk_frames, bundle.frame_count)
                cameras = [
                    splat_camera_for_view_time(
                        bundle,
                        split=split,
                        view=view,
                        frame=frame,
                        camera_projection=camera_projection,
                    )
                    for frame in range(start, stop)
                ]
                rendered, render_elapsed = time_render_sequence(
                    device,
                    lambda cameras=cameras: render_splat_sequence(model, cameras, render_cfg),
                )
                view_render_elapsed += render_elapsed
                target = frames_tensor[view, start:stop].permute(0, 2, 3, 1).contiguous().cpu()
                rendered = {
                    "rgb": rendered["rgb"].detach().cpu(),
                    "alpha": rendered["alpha"].detach().cpu(),
                }
                accumulator.update(rendered["rgb"], target)
                if split == "heldout":
                    count = stop - start
                    lpips_sum += video_lpips(rendered["rgb"], target) * count
                    lpips_count += count
                append_chunk_media(
                    start=start,
                    stop=stop,
                    selected=selected,
                    target=target,
                    rendered=rendered["rgb"],
                    alpha=rendered["alpha"],
                    targets_out=media_targets,
                    rendered_out=media_renders,
                    alpha_out=media_alphas,
                )
                del rendered, target
            row_metrics = accumulator.metrics()
            if split == "heldout":
                row_metrics["eval_lpips"] = lpips_sum / float(lpips_count)
            metrics_rows.append(row_metrics)
            render_times.append(view_render_elapsed)
            rows.append(
                (
                    torch.cat(media_targets, dim=0),
                    RenderedSequence(
                        rgb=torch.cat(media_renders, dim=0),
                        alpha=torch.cat(media_alphas, dim=0),
                    ),
                )
            )
        return rows, metrics_rows, render_times

    train_rows, train_metrics, train_render_times = eval_split("train", bundle.train_frames)
    heldout_rows: list = []
    heldout_metrics: list = []
    heldout_render_times: list = []
    if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
        heldout_rows, heldout_metrics, heldout_render_times = eval_split("heldout", bundle.heldout_frames)
    metrics = aggregate_view_metrics(train_metrics)
    if heldout_metrics:
        metrics.update(prefix_metrics("heldout", aggregate_view_metrics(heldout_metrics)))
    metrics["eval_render_elapsed_s"] = time.perf_counter() - render_started
    metrics.update(render_time_metrics(train_render_times, heldout_render_times))
    return {"metrics": metrics, "train_rows": train_rows, "heldout_rows": heldout_rows}


def run_dynamic_splats_lane(
    *,
    args: argparse.Namespace,
    bundle,
    paper_protocol: dict[str, Any] | None,
    out_dir: Path,
    eval_chunk_frames: int,
    eval_media_max_frames: int,
) -> dict[str, Any]:
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
        paper_protocol=paper_protocol,
    )
    splat_eval = eval_free_splats(
        splat_model,
        splat_render_cfg,
        bundle,
        camera_projection=args.splat_camera_projection,
        chunk_frames=eval_chunk_frames,
        media_max_frames=eval_media_max_frames,
    )
    save_first_row_media(
        out_dir,
        "free_dynamic_splats_train_view0",
        splat_eval["train_rows"],
        fps=float(bundle.metadata.get("fps", 4.0)),
    )
    save_first_row_media(
        out_dir,
        "free_dynamic_splats_heldout_view0",
        splat_eval["heldout_rows"],
        fps=float(bundle.metadata.get("fps", 4.0)),
    )
    return {
        "splat_count": args.splat_count,
        "renderer": args.splat_renderer,
        "camera_projection": args.splat_camera_projection,
        "render_camera_projection": splat_render_cfg.camera_projection,
        **splat_train,
        "metrics": splat_eval["metrics"],
    }


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


def selected_env(keys: tuple[str, ...]) -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in keys}


def apply_uvt_tile_env(config: UVTRenderConfig) -> None:
    os.environ["STAR_UVT_TILE_X"] = str(config.tile_x)
    os.environ["STAR_UVT_TILE_Y"] = str(config.tile_y)
    os.environ["STAR_UVT_TILE_T"] = str(config.tile_t)
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(config.tile_capacity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument(
        "--camera-rig-init",
        choices=("dnerf", "deepview", "aist", "neural_3d_video", "vivo", "camxtime", "orthogonal_origin"),
        default=None,
        help="Override the baseline config camera rig for a paper dataset adapter.",
    )
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--train-seconds", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--torch-deterministic", choices=("off", "warn", "error"), default="off")
    parser.add_argument("--uvt-tubes", type=int, default=128)
    parser.add_argument("--uvt-lr", type=float, default=0.03)
    parser.add_argument("--uvt-lr-decay-step", type=int, default=0)
    parser.add_argument("--uvt-lr-decay-factor", type=float, default=1.0)
    parser.add_argument("--uvt-init-precision-xy", type=float, default=30.0)
    parser.add_argument("--uvt-init-lambda-t", type=float, default=0.35)
    parser.add_argument("--uvt-static-tube-fraction", type=float, default=0.0)
    parser.add_argument("--uvt-static-init-lambda-t", type=float, default=0.02)
    parser.add_argument("--uvt-static-velocity-reg", type=float, default=0.0)
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
    parser.add_argument("--uvt-backward-policy", choices=("manual", *BACKWARD_POLICY_NAMES), default="manual")
    parser.add_argument(
        "--uvt-reduction-mode",
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
        "--uvt-sample-emission-mode",
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
            "direct_fixedpoint",
            "direct_split_fixedpoint",
            "direct_serial",
            "tile_pair_atomic",
            "tile_pair_fixedpoint",
            "tile_pair_reduced",
            "tile_pair_reduced_parallel",
            "tile_pair_suffix_reduced",
        ),
        default="atomic_append",
    )
    parser.add_argument("--uvt-camera-projection", choices=("legacy_pinhole", "dataset_lens"), default="legacy_pinhole")
    parser.add_argument(
        "--uvt-camera-sequence-mode",
        choices=("static_view", "dynamic_first_order", "projective_first_order", "segmented"),
        default="static_view",
    )
    parser.add_argument("--uvt-segment-frames", type=int, default=4)
    parser.add_argument("--uvt-synthetic-pan-x", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-pan-y", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-dolly-z", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-zoom", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-principal-x", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-principal-y", type=float, default=0.0)
    parser.add_argument(
        "--uvt-loss-scope",
        choices=("sampled_frame", "view_sequence", "temporal_window", "paper_batch"),
        default="sampled_frame",
    )
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
    parser.add_argument("--paper-protocol", type=Path, default=None)
    parser.add_argument("--eval-chunk-frames", type=int, default=4)
    parser.add_argument("--eval-media-max-frames", type=int, default=32)
    parser.add_argument(
        "--allow-paper-local-mps-execution",
        action="store_true",
        help="Required for paper-protocol MPS execution; the unified runner owns the safety preflight.",
    )
    parser.add_argument(
        "--only-lane",
        choices=("combined", "world_tubes", "dynamic_3dgs"),
        default="combined",
        help="Run one representation in this process so allocator state is released at process exit.",
    )
    parser.add_argument("--skip-splats", action="store_true")
    parser.add_argument("--init-depth", type=float, default=2.0)
    parser.add_argument("--uvt-init-views", choices=("first", "all_train"), default="first")
    parser.add_argument("--uvt-init-sampling", choices=("random", "grid"), default="random")
    parser.add_argument("--uvt-init-frames", choices=("first", "all", "fit"), default="first")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "research_project" / "benchmarks" / "results" / "multicam_heldout_compare")
    args = parser.parse_args()

    if args.only_lane == "world_tubes":
        args.skip_splats = True

    if args.torch_deterministic != "off":
        torch.use_deterministic_algorithms(True, warn_only=args.torch_deterministic == "warn")

    device = resolve_device(args.device)
    config = load_config_file(resolve_dynaworld_path(args.baseline_config))
    paper_protocol = None if args.paper_protocol is None else load_config_file(resolve_dynaworld_path(args.paper_protocol))
    if paper_protocol is not None and device.type == "mps" and not args.allow_paper_local_mps_execution:
        raise RuntimeError(
            "Paper-protocol MPS execution is fail-closed after the 2026-07-22 memory-pressure incident. "
            "Launch through the unified runner after explicit user approval."
        )
    if paper_protocol is not None and not bool(paper_protocol.get("enabled", False)):
        raise ValueError("--paper-protocol requires enabled=true")
    if paper_protocol is not None and args.uvt_loss_scope != "paper_batch":
        raise ValueError("--paper-protocol requires --uvt-loss-scope=paper_batch")
    load_image_size = normalize_image_size(args.target_size)
    if paper_protocol is not None:
        protocol_stages = normalize_paper_stages(
            paper_protocol.get("stages"),
            total_steps=args.max_steps,
            default_image_size=load_image_size,
            default_primitive_count=args.uvt_tubes,
            default_frames_per_step=int(paper_protocol.get("frames_per_step", 1)),
        )
        load_image_size = protocol_stages[-1].image_size
    data_cfg = apply_paper_dataset_contract(
        config_data_for_run(config, target_size=args.target_size, max_frames=args.max_frames),
        paper_protocol,
    )
    camera_cfg = dict(config["camera"])
    if args.camera_rig_init is not None:
        camera_cfg["rig_init"] = args.camera_rig_init
    bundle = load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=camera_cfg,
        target_size=(load_image_size.height, load_image_size.width),
        device=device,
        frame_device=torch.device("cpu") if paper_protocol is not None else device,
    )
    backward_policy = None
    if args.uvt_backward_policy != "manual":
        backward_policy = resolve_backward_policy(args.uvt_backward_policy)
        args.uvt_reduction_mode = backward_policy.reduction_mode
        args.uvt_sample_emission_mode = backward_policy.sample_emission_mode
        if args.uvt_render_backend != "metal_tile":
            raise ValueError("--uvt-backward-policy requires --uvt-render-backend metal_tile")
    if args.uvt_render_backend == "metal_tile" and device.type != "mps":
        raise ValueError("--uvt-render-backend=metal_tile requires device=mps")
    if args.uvt_render_backend != "metal_tile" and (
        args.uvt_reduction_mode != "index_add" or args.uvt_sample_emission_mode != "atomic_append"
    ):
        raise ValueError("custom UVT reduction/sample emission modes require --uvt-render-backend metal_tile")
    if args.uvt_reduction_mode in (
        "key_sort_scan_metal",
        "key_sort_compensated_scan_metal",
        "key_sort_segmented_metal",
    ) and args.uvt_sample_emission_mode not in (
        "with_keys",
        "tile_pair",
        "tile_pair_compensated",
        "tile_pair_grouped",
        "tile_pair_parallel",
        "tile_pair_scanline",
        "tile_pair_sharedsort",
        "tile_pair_target_bounds",
        "tile_pair_suffix",
    ):
        raise ValueError(
            "keyed sort reduction requires --uvt-sample-emission-mode with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, or tile_pair_suffix"
        )
    if args.uvt_sample_emission_mode in (
        "direct_atomic",
        "direct_fixedpoint",
        "direct_split_fixedpoint",
        "direct_serial",
        "tile_pair_atomic",
        "tile_pair_fixedpoint",
        "tile_pair_reduced",
        "tile_pair_reduced_parallel",
        "tile_pair_suffix_reduced",
    ) and args.uvt_reduction_mode != "index_add":
        raise ValueError(f"{args.uvt_sample_emission_mode} bypasses the reducer and requires --uvt-reduction-mode index_add")
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
        "target_size": args.target_size if paper_protocol is None else load_image_size.as_list(),
        "image_size": load_image_size.as_list(),
        "max_frames": args.max_frames,
        "frame_count": int(bundle.frame_count),
        "train_seconds": args.train_seconds,
        "device": str(device),
        "seed": args.seed,
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "torch": {
            "version": torch.__version__,
            "deterministic_mode": args.torch_deterministic,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "env": selected_env(
            (
                "PYTHONHASHSEED",
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_ALLOCATOR_POLICY",
                "STAR_UVT_TILE_X",
                "STAR_UVT_TILE_Y",
                "STAR_UVT_TILE_T",
                "STAR_UVT_TILE_CAPACITY",
                "STAR_UVT_FIXEDPOINT_SCALE",
                "STAR_UVT_SPLIT_FIXEDPOINT_COARSE_SCALE",
                "STAR_UVT_SPLIT_FIXEDPOINT_FINE_SCALE",
            )
        ),
        "train_cameras": bundle.train_camera_names,
        "heldout_cameras": bundle.heldout_camera_names,
        "pose_source": bundle.pose_source,
        "uvt_camera_projection": args.uvt_camera_projection,
        "uvt_camera_sequence_mode": args.uvt_camera_sequence_mode,
        "uvt_segment_frames": args.uvt_segment_frames,
        "uvt_synthetic_camera_motion": {
            "pan_x": args.uvt_synthetic_pan_x,
            "pan_y": args.uvt_synthetic_pan_y,
            "dolly_z": args.uvt_synthetic_dolly_z,
            "zoom": args.uvt_synthetic_zoom,
            "principal_x": args.uvt_synthetic_principal_x,
            "principal_y": args.uvt_synthetic_principal_y,
        },
        "uvt_reduction_mode": args.uvt_reduction_mode,
        "uvt_sample_emission_mode": args.uvt_sample_emission_mode,
        "uvt_backward_policy": None if backward_policy is None else backward_policy.as_dict(),
        "splat_camera_projection": args.splat_camera_projection,
        "skip_splats": args.skip_splats,
        "only_lane": args.only_lane,
        "eval_chunk_frames": args.eval_chunk_frames,
        "eval_media_max_frames": args.eval_media_max_frames,
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

    if args.only_lane == "dynamic_3dgs":
        report = {
            "meta": run_meta,
            "star_uvt": None,
            "star_uvt_selected": None,
            "free_dynamic_splats": run_dynamic_splats_lane(
                args=args,
                bundle=bundle,
                paper_protocol=paper_protocol,
                out_dir=out_dir,
                eval_chunk_frames=args.eval_chunk_frames,
                eval_media_max_frames=args.eval_media_max_frames,
            ),
        }
        write_json(out_dir / "comparison_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote dynamic-3DGS-only multicam comparison to {out_dir}")
        return
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
        static_tube_fraction=args.uvt_static_tube_fraction,
        static_init_lambda_t=args.uvt_static_init_lambda_t,
        static_velocity_reg_weight=args.uvt_static_velocity_reg,
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
        camera_sequence_mode=args.uvt_camera_sequence_mode,
        segment_frames=args.uvt_segment_frames,
        synthetic_pan_x=args.uvt_synthetic_pan_x,
        synthetic_pan_y=args.uvt_synthetic_pan_y,
        synthetic_dolly_z=args.uvt_synthetic_dolly_z,
        synthetic_zoom=args.uvt_synthetic_zoom,
        synthetic_principal_x=args.uvt_synthetic_principal_x,
        synthetic_principal_y=args.uvt_synthetic_principal_y,
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
        reduction_mode=args.uvt_reduction_mode,
        sample_emission_mode=args.uvt_sample_emission_mode,
        paper_protocol=paper_protocol,
    )
    uvt_eval = eval_world_tubes(
        uvt_model,
        bundle,
        backend=args.uvt_render_backend,
        camera_projection=args.uvt_camera_projection,
        camera_sequence_mode=args.uvt_camera_sequence_mode,
        segment_frames=args.uvt_segment_frames,
        synthetic_pan_x=args.uvt_synthetic_pan_x,
        synthetic_pan_y=args.uvt_synthetic_pan_y,
        synthetic_dolly_z=args.uvt_synthetic_dolly_z,
        synthetic_zoom=args.uvt_synthetic_zoom,
        synthetic_principal_x=args.uvt_synthetic_principal_x,
        synthetic_principal_y=args.uvt_synthetic_principal_y,
        render_config=render_config,
        frame_metric_splits=uvt_frame_metric_splits,
        chunk_frames=args.eval_chunk_frames,
        media_max_frames=args.eval_media_max_frames,
    )
    uvt_checkpoint_curve = eval_world_tube_checkpoints(
        uvt_model,
        uvt_checkpoints,
        bundle,
        backend=args.uvt_render_backend,
        camera_projection=args.uvt_camera_projection,
        camera_sequence_mode=args.uvt_camera_sequence_mode,
        segment_frames=args.uvt_segment_frames,
        synthetic_pan_x=args.uvt_synthetic_pan_x,
        synthetic_pan_y=args.uvt_synthetic_pan_y,
        synthetic_dolly_z=args.uvt_synthetic_dolly_z,
        synthetic_zoom=args.uvt_synthetic_zoom,
        synthetic_principal_x=args.uvt_synthetic_principal_x,
        synthetic_principal_y=args.uvt_synthetic_principal_y,
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
            camera_sequence_mode=args.uvt_camera_sequence_mode,
            segment_frames=args.uvt_segment_frames,
            synthetic_pan_x=args.uvt_synthetic_pan_x,
            synthetic_pan_y=args.uvt_synthetic_pan_y,
            synthetic_dolly_z=args.uvt_synthetic_dolly_z,
            synthetic_zoom=args.uvt_synthetic_zoom,
            synthetic_principal_x=args.uvt_synthetic_principal_x,
            synthetic_principal_y=args.uvt_synthetic_principal_y,
            render_config=render_config,
            frame_metric_splits=uvt_frame_metric_splits,
            chunk_frames=args.eval_chunk_frames,
            media_max_frames=args.eval_media_max_frames,
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
                camera_sequence_mode=args.uvt_camera_sequence_mode,
                segment_frames=args.uvt_segment_frames,
                render_config=render_config,
            )
            if (
                args.uvt_render_backend == "metal_tile"
                and not any(run_meta["uvt_synthetic_camera_motion"].values())
            )
            else None,
        }
        uvt_model.load_state_dict(final_state)
    save_first_row_media(out_dir, "star_uvt_train_view0", uvt_eval["train_rows"], fps=float(bundle.metadata.get("fps", 4.0)))
    save_first_row_media(out_dir, "star_uvt_heldout_view0", uvt_eval["heldout_rows"], fps=float(bundle.metadata.get("fps", 4.0)))
    uvt_metrics = uvt_eval["metrics"]
    uvt_metal_stats = (
        world_tube_metal_stats(
            uvt_model,
            bundle,
            camera_projection=args.uvt_camera_projection,
            camera_sequence_mode=args.uvt_camera_sequence_mode,
            segment_frames=args.uvt_segment_frames,
            render_config=render_config,
        )
        if (
            args.uvt_render_backend == "metal_tile"
            and not any(run_meta["uvt_synthetic_camera_motion"].values())
        )
        else None
    )
    # Full 300-frame evaluation rows retain hundreds of MB of rendered RGB and
    # autograd-adjacent Metal allocations. Only the scalar metrics and saved
    # media are needed after this point; release them before dynamic 3DGS eval.
    del uvt_eval
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    splat_report: dict[str, Any] | None = None
    if not args.skip_splats:
        splat_report = run_dynamic_splats_lane(
            args=args,
            bundle=bundle,
            paper_protocol=paper_protocol,
            out_dir=out_dir,
            eval_chunk_frames=args.eval_chunk_frames,
            eval_media_max_frames=args.eval_media_max_frames,
        )

    report = {
        "meta": run_meta,
        "star_uvt": {
            "tube_count": args.uvt_tubes,
            "render_backend": args.uvt_render_backend,
            "reduction_mode": args.uvt_reduction_mode,
            "sample_emission_mode": args.uvt_sample_emission_mode,
            "camera_projection": args.uvt_camera_projection,
            "camera_sequence_mode": args.uvt_camera_sequence_mode,
            "segment_frames": args.uvt_segment_frames,
            "synthetic_camera_motion": run_meta["uvt_synthetic_camera_motion"],
            "lr": args.uvt_lr,
            "lr_decay_step": args.uvt_lr_decay_step,
            "lr_decay_factor": args.uvt_lr_decay_factor,
            "init_precision_xy": args.uvt_init_precision_xy,
            "init_lambda_t": args.uvt_init_lambda_t,
            "static_tube_fraction": args.uvt_static_tube_fraction,
            "static_init_lambda_t": args.uvt_static_init_lambda_t,
            "static_velocity_reg": args.uvt_static_velocity_reg,
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
            "metrics": uvt_metrics,
            "checkpoint_curve": uvt_checkpoint_curve,
            "metal_stats": uvt_metal_stats,
        },
        "star_uvt_selected": selected_report,
        "free_dynamic_splats": splat_report,
    }
    write_json(out_dir / "comparison_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote STAR-UVT multicam heldout comparison to {out_dir}")


if __name__ == "__main__":
    main()
