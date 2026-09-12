from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, replace
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
for path in (DYNAWORLD_ROOT, TRAIN_SRC, GAUGE_EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    UVTRenderConfig,
    render_uvt_tubes,
    slice_projective_trace_cell_atlas_frames,
    uvt_tubes_to_projective_trace_cell_atlas,
)
from camera import CameraSpec  # noqa: E402
from config_utils import load_config_file, serialize_config_value  # noqa: E402
from common import prefix_metrics, robust_l1, save_preview_strip, save_side_by_side_mp4, video_metrics, write_json  # noqa: E402
from device_memory import DeviceMemorySampler, device_memory_stats  # noqa: E402
from multicam_video_data import load_multicam_video_bundle  # noqa: E402
from paper_training_protocol import (  # noqa: E402
    PaperCostTracker,
    PaperPhaseTimer,
    PaperRGBMetricAccumulator,
    PaperSampleScheduleDigest,
    SpacetimeEpochSampler,
    apply_paper_dataset_contract,
    normalize_image_size,
    normalize_paper_stages,
    paper_dataset_bundle_identity,
    paper_evaluator_contract,
    paper_native_module_identity,
    paper_runtime_identity,
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
from research_experiments.spd4_world_tubes.hybrid_transfer import (  # noqa: E402
    render_variance_certified_hybrid_metal,
)
from research_experiments.spd4_world_tubes.retained_fiber_metal import (  # noqa: E402
    render_retained_fiber_metal,
)
from research_experiments.paper_runner_suite.frozen_atlas_storage import (  # noqa: E402
    LOGICAL_PAYLOAD_DEFINITION,
    REPLAY_STORAGE_REASON,
    RETAINED_STORAGE_DEFINITION,
    ROUTE_MEMORY_DEFINITION,
    ROUTE_MEMORY_MEASUREMENT_SOURCE,
    TENSOR_NAMES as FROZEN_ATLAS_TENSOR_NAMES,
    write_retained_storage_artifact,
)

try:
    from research_project.trainer_harness.model import dense_differentiable_render_uvt_tubes
    from research_project.trainer_harness.spd4_world_atom import (
        SPD4WorldAtomBatch,
        SPD4WorldAtomModel,
        project_spd4_world_atoms_from_pixel_jacobian,
        project_spd4_world_atoms_pinhole,
        project_spd4_world_atoms_pinhole_motion,
    )
    from research_project.trainer_harness.tile_metal_autograd import (
        BACKWARD_POLICY_NAMES,
        ProjectiveCellIntervalTrainerState,
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
    from spd4_world_atom import (
        SPD4WorldAtomBatch,
        SPD4WorldAtomModel,
        project_spd4_world_atoms_from_pixel_jacobian,
        project_spd4_world_atoms_pinhole,
        project_spd4_world_atoms_pinhole_motion,
    )
    from tile_metal_autograd import (
        BACKWARD_POLICY_NAMES,
        ProjectiveCellIntervalTrainerState,
        render_uvt_tubes_metal_tile_backward,
        resolve_backward_policy,
    )
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
UVT_RENDER_BACKENDS = (
    "dense",
    "metal_tile",
    "retained_fiber_metal",
    "hybrid_retained_fiber",
)
UVT_FAST_METAL_BACKENDS = {"metal_tile", "hybrid_retained_fiber"}
UVT_NATIVE_METAL_BACKENDS = {
    "metal_tile",
    "retained_fiber_metal",
    "hybrid_retained_fiber",
}
FROZEN_WORLD_ACCEPTANCE = {
    "image_max_abs_error": 1.0e-5,
    "loss_absolute_delta": 1.0e-5,
    "gradient_global_normalized_l2_error": 1.0e-5,
    "gradient_max_parameter_normalized_l2_error": 1.0e-5,
    "min_world_vjp_l2_norm": 1.0e-12,
    "fallback_fraction": 0.20,
}
FROZEN_WORLD_CANONICAL_FRAME_COUNTS = (4, 8, 16, 32, 64, 128)
FROZEN_WORLD_MIN_TIMING_WARMUPS = 1
FROZEN_WORLD_MIN_TIMING_REPEATS = 3
FROZEN_WORLD_MAX_FRAME_COUNT_REQUESTS = 16
FROZEN_WORLD_MAX_TIMING_WARMUPS = 10
FROZEN_WORLD_MAX_TIMING_REPEATS = 20
PAPER_DYNAMIC_FAST_MAC_OPTIONS = {
    "rgb_variant": "v5",
    "background": [0.0, 0.0, 0.0],
}


def parse_frozen_world_frame_counts(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    tokens = value.split(",")
    if not tokens or any(not token.strip() for token in tokens):
        raise ValueError(
            "--frozen-world-frame-counts must be a comma-separated list of "
            "nonnegative integers"
        )
    try:
        counts = tuple(int(token.strip()) for token in tokens)
    except ValueError as error:
        raise ValueError(
            "--frozen-world-frame-counts must contain only base-10 integers"
        ) from error
    if any(count < 0 for count in counts):
        raise ValueError("--frozen-world-frame-counts must be nonnegative")
    if len(counts) > FROZEN_WORLD_MAX_FRAME_COUNT_REQUESTS:
        raise ValueError(
            "--frozen-world-frame-counts has too many entries; maximum is "
            f"{FROZEN_WORLD_MAX_FRAME_COUNT_REQUESTS}"
        )
    return counts


def validate_frozen_world_timing_controls(
    *,
    warmups: int,
    repeats: int,
) -> None:
    if warmups < 0 or warmups > FROZEN_WORLD_MAX_TIMING_WARMUPS:
        raise ValueError(
            "frozen-world timing warmups must be in "
            f"[0, {FROZEN_WORLD_MAX_TIMING_WARMUPS}]"
        )
    if repeats < 1 or repeats > FROZEN_WORLD_MAX_TIMING_REPEATS:
        raise ValueError(
            "frozen-world timing repeats must be in "
            f"[1, {FROZEN_WORLD_MAX_TIMING_REPEATS}]"
        )


def resolve_frozen_world_frame_counts(
    *,
    full_frames: int,
    primary_max_frames: int,
    requested_frame_counts: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if full_frames < 1:
        raise ValueError("frozen-world full frame count must be positive")
    if len(requested_frame_counts or ()) > FROZEN_WORLD_MAX_FRAME_COUNT_REQUESTS:
        raise ValueError("frozen-world frame-count request is too large")
    candidates = (int(primary_max_frames), *(requested_frame_counts or ()))
    resolved: list[int] = []
    for requested in candidates:
        if requested < 0:
            raise ValueError("frozen-world frame counts must be nonnegative")
        frame_count = full_frames if requested == 0 else min(requested, full_frames)
        if frame_count not in resolved:
            resolved.append(frame_count)
    if full_frames > 1 and 1 in resolved:
        raise ValueError(
            "frozen-world full-interval sampling requires at least two frames"
        )
    return tuple(sorted(resolved))


def frozen_world_full_interval_frame_indices(
    full_frames: int,
    frame_count: int,
) -> tuple[int, ...]:
    if full_frames < 1 or frame_count < 1 or frame_count > full_frames:
        raise ValueError(
            "frozen-world sampled frame count must be in [1, full_frames]"
        )
    if frame_count == 1:
        if full_frames > 1:
            raise ValueError(
                "frozen-world full-interval sampling requires at least two frames"
            )
        return (full_frames // 2,)
    denominator = frame_count - 1
    indices = tuple(
        (
            sample * (full_frames - 1) + denominator // 2
        )
        // denominator
        for sample in range(frame_count)
    )
    if len(set(indices)) != frame_count:
        raise RuntimeError("frozen-world full-interval time grid is not unique")
    return indices


def frozen_world_sequence_sha256(values: tuple[int | float, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(values),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def frozen_world_sweep_publication_eligible(
    *,
    requested_frame_counts: tuple[int, ...] | None,
    full_frames: int,
    timing_warmups: int,
    timing_repeats: int,
    selected_time_slice_parity_accepted: bool,
    all_rows_storage_publication_ready: bool,
    all_rows_route_memory_publication_ready: bool,
) -> bool:
    requested = set(requested_frame_counts or ())
    return (
        full_frames >= max(FROZEN_WORLD_CANONICAL_FRAME_COUNTS)
        and (0 in requested or full_frames in requested)
        and set(FROZEN_WORLD_CANONICAL_FRAME_COUNTS).issubset(requested)
        and timing_warmups >= FROZEN_WORLD_MIN_TIMING_WARMUPS
        and timing_repeats >= FROZEN_WORLD_MIN_TIMING_REPEATS
        and selected_time_slice_parity_accepted
        and all_rows_storage_publication_ready
        and all_rows_route_memory_publication_ready
    )


def _timing_quantile(sorted_samples: list[float], probability: float) -> float:
    if not sorted_samples:
        raise ValueError("timing samples must be nonempty")
    position = float(len(sorted_samples) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_samples[lower]
    fraction = position - float(lower)
    return (
        sorted_samples[lower] * (1.0 - fraction)
        + sorted_samples[upper] * fraction
    )


def frozen_world_timing_summary(samples: list[float]) -> dict[str, float | int]:
    if not samples or any(not math.isfinite(value) or value < 0.0 for value in samples):
        raise ValueError("timing samples must be finite and nonnegative")
    ordered = sorted(float(value) for value in samples)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _timing_quantile(ordered, 0.25),
        "median": statistics.median(ordered),
        "p75": _timing_quantile(ordered, 0.75),
        "max": ordered[-1],
        "mean": math.fsum(ordered) / float(len(ordered)),
    }


def write_json_atomic(path: Path, payload: Any) -> Path:
    """Durably replace one JSON artifact without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                serialize_config_value(payload),
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def star_uvt_native_extension_identity() -> dict[str, Any]:
    native = paper_native_module_identity(
        "torch_gsplat_bridge_star_uvt._C",
        runtime_source_root=ROOT / "csrc" / "metal",
    )
    source_files = sorted(
        (
            *(
                candidate
                for candidate in (ROOT / "csrc").rglob("*")
                if candidate.is_file()
            ),
            ROOT / "setup.py",
        )
    )
    source_digest = hashlib.sha256()
    for source_path in source_files:
        relative = source_path.relative_to(ROOT)
        source_digest.update(str(relative).encode("utf-8"))
        source_digest.update(file_sha256(source_path).encode("ascii"))
    return {
        **native,
        "source_tree_sha256": source_digest.hexdigest(),
        "source_file_count": len(source_files),
    }


def fast_mac_v5_native_extension_identity() -> dict[str, Any]:
    variant = (
        DYNAWORLD_ROOT
        / "third_party"
        / "fast-mac-gsplat"
        / "variants"
        / "v5"
    )
    if str(variant) not in sys.path:
        sys.path.insert(0, str(variant))
    return paper_native_module_identity(
        "torch_gsplat_bridge_v5._C",
        runtime_source_root=variant / "csrc" / "metal",
    )


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


class VideoMetricAccumulator(PaperRGBMetricAccumulator):
    """Backward-compatible name for the canonical paper evaluator."""


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
    return {
        key: value.detach().cpu().contiguous().clone()
        for key, value in model.state_dict().items()
    }


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
    batch: WorldTubeBatch | SPD4WorldAtomBatch,
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
    if isinstance(batch, SPD4WorldAtomBatch):
        projected = project_spd4_world_atoms_from_pixel_jacobian(
            batch,
            world_to_camera,
            pixels,
            pixel_jacobian,
        )
        return ProjectedTubeSequence(
            ma=projected.ma,
            q_uvt=projected.q_uvt,
            depth0=projected.depth0,
            depth_beta=projected.depth_beta,
            opacity=projected.opacity,
            color=projected.color,
            depth_variance=projected.depth_variance,
            peak_to_fiber_scale=projected.peak_to_fiber_scale,
        )
    ma, q_uvt, depth0, depth_beta, opacity, color = (
        project_world_tubes_from_pixel_jacobian(
            batch,
            world_to_camera,
            pixels,
            pixel_jacobian,
            config,
        )
    )
    return ProjectedTubeSequence(
        ma=ma,
        q_uvt=q_uvt,
        depth0=depth0,
        depth_beta=depth_beta,
        opacity=opacity,
        color=color,
    )


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
    model: WorldTubeModel | SPD4WorldAtomModel,
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
        fx=K_mid[0, 0],
        fy=K_mid[1, 1],
        cx=K_mid[0, 2],
        cy=K_mid[1, 2],
        fx_dot=K_dot[0, 0],
        fy_dot=K_dot[1, 1],
        cx_dot=K_dot[0, 2],
        cy_dot=K_dot[1, 2],
        world_to_camera=w2c_seq[mid_index].to(dtype=torch.float32),
        world_to_camera_dot=w2c_dot.to(dtype=torch.float32),
        chart_time=chart_global_t,
    )
    batch = model.batch()
    if isinstance(batch, SPD4WorldAtomBatch):
        spd4_projected = project_spd4_world_atoms_pinhole_motion(batch, camera)
        projected = ProjectedTubeSequence(
            ma=spd4_projected.ma,
            q_uvt=spd4_projected.q_uvt,
            depth0=spd4_projected.depth0,
            depth_beta=spd4_projected.depth_beta,
            opacity=spd4_projected.opacity,
            color=spd4_projected.color,
            depth_variance=spd4_projected.depth_variance,
            peak_to_fiber_scale=spd4_projected.peak_to_fiber_scale,
        )
    else:
        projector = (
            project_world_tubes_pinhole_projective_motion
            if projective_gauge
            else project_world_tubes_pinhole_motion
        )
        ma, q_uvt, depth0, depth_beta, opacity, color = projector(
            batch,
            camera,
            config,
        )
        projected = ProjectedTubeSequence(
            ma=ma,
            q_uvt=q_uvt,
            depth0=depth0,
            depth_beta=depth_beta,
            opacity=opacity,
            color=color,
        )
    local_t = global_to_local_time(chart_global_t, full_frames=int(full_frames), config=config, frame_start=frame_start)
    local_ma = torch.cat(
        (
            projected.ma[:, :2],
            projected.ma[:, 2:3]
            - projected.ma.new_tensor(chart_global_t - local_t),
        ),
        dim=-1,
    ).contiguous()
    return ProjectedTubeSequence(
        ma=local_ma,
        q_uvt=projected.q_uvt,
        depth0=projected.depth0,
        depth_beta=projected.depth_beta,
        opacity=projected.opacity,
        color=projected.color,
        depth_variance=projected.depth_variance,
        peak_to_fiber_scale=projected.peak_to_fiber_scale,
    )


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
    model: WorldTubeModel | SPD4WorldAtomModel,
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
        if isinstance(model, SPD4WorldAtomModel):
            raise ValueError(
                "full_spd4 segmented compilation is not implemented; use one "
                "of static_view, dynamic_first_order, or projective_first_order"
            )
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


def opacity_semantics(
    alpha_mode: str,
    amplitude_convention: str = "fiber_integrated",
) -> str:
    if alpha_mode == "peak_splat":
        return "peak_alpha_amplitude"
    if alpha_mode == "beer_lambert":
        if amplitude_convention == "fiber_integrated":
            return "nonnegative_fiber_integrated_peak_optical_thickness"
        if amplitude_convention == "peak_density":
            return "nonnegative_world_peak_extinction_density"
    raise ValueError("alpha_mode must be one of: peak_splat, beer_lambert")


def max_alpha_for_mode(alpha_mode: str) -> float:
    """Return the compositing cap appropriate to the opacity parameterization."""

    if alpha_mode == "peak_splat":
        return 0.99
    if alpha_mode == "beer_lambert":
        # Beer-Lambert opacity is 1-exp(-tau), so 1.0 removes the historical
        # peak-splat cap without limiting the trainable optical thickness tau.
        return 1.0
    raise ValueError("alpha_mode must be one of: peak_splat, beer_lambert")


def initial_raw_opacity(
    init_source_amplitude: float,
    *,
    alpha_mode: str,
    amplitude_convention: str = "fiber_integrated",
    count: int,
    reference: Tensor,
) -> Tensor:
    if amplitude_convention == "peak_density":
        if alpha_mode != "beer_lambert":
            raise ValueError("peak_density initialization requires beer_lambert")
        if not float(init_source_amplitude) > 0.0:
            raise ValueError("peak-density init_opacity must be positive")
        peak_density = torch.full(
            (count,),
            float(init_source_amplitude),
            dtype=reference.dtype,
            device=reference.device,
        )
        return _inv_softplus(peak_density)
    if not 0.0 < float(init_source_amplitude) < 0.99:
        raise ValueError("init_opacity must be an initial center alpha in (0, 0.99)")
    center_alpha = torch.full(
        (count,),
        float(init_source_amplitude),
        dtype=reference.dtype,
        device=reference.device,
    )
    if alpha_mode == "peak_splat":
        return _logit(center_alpha / 0.99)
    if alpha_mode == "beer_lambert":
        peak_optical_thickness = -torch.log1p(-center_alpha)
        return _inv_softplus(peak_optical_thickness)
    raise ValueError("alpha_mode must be one of: peak_splat, beer_lambert")


def opacity_from_raw(raw_opacity: Tensor, *, alpha_mode: str) -> Tensor:
    if alpha_mode == "peak_splat":
        return torch.sigmoid(raw_opacity) * 0.99
    if alpha_mode == "beer_lambert":
        return F.softplus(raw_opacity)
    raise ValueError("alpha_mode must be one of: peak_splat, beer_lambert")


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
    # Progressive budgets select a prefix. Interleave source groups so that a
    # coarse stage covers cameras and times instead of exhausting camera zero.
    order = torch.argsort(
        torch.cat([torch.arange(group.shape[0]) for group in points]), stable=True,
    ).to(points[0].device)
    return tuple(
        torch.cat(groups, dim=0).index_select(0, order).contiguous()
        for groups in (points, colors, t0_values)
    )


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
    representation_name = "legacy_tube"
    geometry_dof_per_atom = 10
    total_dof_per_atom = 14

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
        alpha_mode: str = "peak_splat",
        amplitude_convention: str = "fiber_integrated",
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
        self.alpha_mode = str(alpha_mode)
        self.amplitude_convention = str(amplitude_convention)
        if self.amplitude_convention != "fiber_integrated":
            raise ValueError(
                "legacy_tube supports only amplitude_convention=fiber_integrated"
            )
        self.opacity_semantics = opacity_semantics(
            self.alpha_mode,
            self.amplitude_convention,
        )
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
        self.raw_precision_xy = nn.Parameter(_inv_softplus(precision - self.min_precision_xy))
        self.raw_lambda_t = nn.Parameter(_inv_softplus(lambda_t - self.min_lambda_t))
        self.raw_opacity = nn.Parameter(
            initial_raw_opacity(
                init_opacity,
                alpha_mode=self.alpha_mode,
                amplitude_convention=self.amplitude_convention,
                count=tube_count,
                reference=init_x0,
            )
        )
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
            opacity=opacity_from_raw(
                self.raw_opacity[active],
                alpha_mode=self.alpha_mode,
            ),
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

    def representation_metadata(self) -> dict[str, int | str]:
        return {
            "world_representation": self.representation_name,
            "geometry_dof_per_atom": self.geometry_dof_per_atom,
            "total_dof_per_atom": self.total_dof_per_atom,
            "motion_parameterization": "explicit_velocity",
            "spatial_covariance_parameterization": "axis_aligned_xy_precision",
            "alpha_mode": self.alpha_mode,
            "amplitude_convention": self.amplitude_convention,
            "opacity_semantics": self.opacity_semantics,
        }


@dataclass(frozen=True)
class RenderedSequence:
    rgb: Tensor
    alpha: Tensor
    fallback_tiles: Tensor | None = None
    fallback_reason_bits: Tensor | None = None
    fallback_active_counts: Tensor | None = None
    minimum_pair_separation: Tensor | None = None


@dataclass(frozen=True)
class ProjectedTubeSequence:
    ma: Tensor
    q_uvt: Tensor
    depth0: Tensor
    depth_beta: Tensor
    opacity: Tensor
    color: Tensor
    depth_variance: Tensor | None = None
    peak_to_fiber_scale: Tensor | None = None


def compiled_projected_opacity(
    projected: ProjectedTubeSequence,
    config: UVTRenderConfig,
) -> Tensor:
    if config.amplitude_convention == "fiber_integrated":
        return projected.opacity
    if config.amplitude_convention != "peak_density":
        raise ValueError(
            "amplitude_convention must be one of: fiber_integrated, peak_density"
        )
    if config.alpha_mode != "beer_lambert":
        raise ValueError("peak_density amplitude requires alpha_mode=beer_lambert")
    if projected.peak_to_fiber_scale is None:
        raise ValueError(
            "peak_density amplitude requires full_spd4 gauge measure and "
            "conditional depth variance"
        )
    return (projected.opacity * projected.peak_to_fiber_scale).contiguous()


def project_world_tube_sequence(
    model: WorldTubeModel | SPD4WorldAtomModel,
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
        batch = model.batch()
        if isinstance(batch, SPD4WorldAtomBatch):
            spd4_projected = project_spd4_world_atoms_pinhole(batch, camera)
            projected = ProjectedTubeSequence(
                ma=spd4_projected.ma,
                q_uvt=spd4_projected.q_uvt,
                depth0=spd4_projected.depth0,
                depth_beta=spd4_projected.depth_beta,
                opacity=spd4_projected.opacity,
                color=spd4_projected.color,
                depth_variance=spd4_projected.depth_variance,
                peak_to_fiber_scale=spd4_projected.peak_to_fiber_scale,
            )
        else:
            ma, q_uvt, depth0, depth_beta, opacity, color = project_world_tubes_pinhole(
                batch, camera, config
            )
            projected = ProjectedTubeSequence(
                ma=ma,
                q_uvt=q_uvt,
                depth0=depth0,
                depth_beta=depth_beta,
                opacity=opacity,
                color=color,
            )
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
            depth_variance=projected.depth_variance,
            peak_to_fiber_scale=projected.peak_to_fiber_scale,
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
    render_opacity = compiled_projected_opacity(projected, config)
    if (
        backend in {"metal_tile", "hybrid_retained_fiber"}
        and config.alpha_mode == "beer_lambert"
        and torch.is_grad_enabled()
        and (reduction_mode != "index_add" or sample_emission_mode != "direct_atomic")
    ):
        raise ValueError(
            "Beer-Lambert Metal training is validated only for the q-UVT "
            "direct_atomic+index_add backward path"
        )
    fallback_tiles = None
    fallback_reason_bits = None
    fallback_active_counts = None
    minimum_pair_separation = None
    if backend == "dense":
        rgb = dense_differentiable_render_uvt_tubes(
            projected.ma,
            projected.q_uvt,
            projected.depth0,
            projected.depth_beta,
            render_opacity,
            projected.color,
            config,
        )
    elif backend == "metal_tile":
        rgb = render_uvt_tubes_metal_tile_backward(
            projected.ma,
            projected.q_uvt,
            projected.depth0,
            projected.depth_beta,
            render_opacity,
            projected.color,
            config,
            reduction_mode=reduction_mode,
            sample_emission_mode=sample_emission_mode,
        )
    elif backend in {"retained_fiber_metal", "hybrid_retained_fiber"}:
        if projected.depth_variance is None:
            raise ValueError(
                f"{backend} requires native full_spd4 conditional depth variance"
            )
        if config.alpha_mode != "beer_lambert":
            raise ValueError(f"{backend} requires alpha_mode=beer_lambert")
        if config.max_alpha != 1.0:
            raise ValueError(f"{backend} requires max_alpha=1.0")
        if projected.ma.device.type != "mps":
            raise ValueError(f"{backend} requires MPS tensors")
        times = (
            torch.arange(
                config.frames,
                dtype=torch.float32,
                device=projected.ma.device,
            )
            - 0.5 * float(config.frames - 1)
        ).contiguous()
        if backend == "retained_fiber_metal":
            rgb = render_retained_fiber_metal(
                projected.ma,
                projected.q_uvt,
                projected.depth0,
                projected.depth_beta,
                projected.depth_variance,
                render_opacity,
                projected.color,
                times,
                height=config.height,
                width=config.width,
                depth_samples=config.retained_depth_samples,
                sigma_extent=config.retained_sigma_extent,
                background=config.background,
                alpha_threshold=config.alpha_threshold,
            )
            tile_shape = (
                (config.frames + config.tile_t - 1) // config.tile_t,
                (config.height + config.tile_y - 1) // config.tile_y,
                (config.width + config.tile_x - 1) // config.tile_x,
            )
            fallback_tiles = torch.ones(
                tile_shape,
                dtype=torch.int32,
                device=projected.ma.device,
            )
        else:
            fast_rgb = render_uvt_tubes_metal_tile_backward(
                projected.ma,
                projected.q_uvt,
                projected.depth0,
                projected.depth_beta,
                render_opacity,
                projected.color,
                config,
                reduction_mode=reduction_mode,
                sample_emission_mode=sample_emission_mode,
            ).contiguous()
            hybrid = render_variance_certified_hybrid_metal(
                fast_rgb=fast_rgb,
                ma=projected.ma,
                q_uvt=projected.q_uvt,
                depth0=projected.depth0,
                depth_beta=projected.depth_beta,
                depth_variance=projected.depth_variance,
                optical_thickness=render_opacity,
                color=projected.color,
                times=times,
                height=config.height,
                width=config.width,
                tile_x=config.tile_x,
                tile_y=config.tile_y,
                tile_t=config.tile_t,
                alpha_threshold=config.alpha_threshold,
                max_alpha=config.max_alpha,
                depth_samples=config.retained_depth_samples,
                sigma_extent=config.retained_sigma_extent,
                certificate_sigma=config.order_certificate_sigma,
                required_gap=config.order_certificate_min_gap,
                background=config.background,
            )
            rgb = hybrid.rgb
            fallback_tiles = hybrid.certificate.fallback_tiles
            fallback_reason_bits = hybrid.certificate.reason_bits
            fallback_active_counts = hybrid.certificate.active_counts
            minimum_pair_separation = hybrid.certificate.minimum_pair_separation
    else:
        raise ValueError(
            "backend must be one of: dense, metal_tile, "
            "retained_fiber_metal, hybrid_retained_fiber"
        )
    alpha = torch.ones((config.frames, config.height, config.width), dtype=rgb.dtype, device=rgb.device)
    return RenderedSequence(
        rgb=rgb,
        alpha=alpha,
        fallback_tiles=fallback_tiles,
        fallback_reason_bits=fallback_reason_bits,
        fallback_active_counts=fallback_active_counts,
        minimum_pair_separation=minimum_pair_separation,
    )


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
    if config.alpha_mode == "peak_splat":
        support_numerator = float(config.alpha_threshold)
    elif config.alpha_mode == "beer_lambert":
        support_numerator = -math.log1p(-float(config.alpha_threshold))
    else:
        raise ValueError("alpha_mode must be one of: peak_splat, beer_lambert")
    support_numerator = max(support_numerator, 1.0e-12)
    opacity_safe = opacity.clamp_min(support_numerator * 1.0001)
    support_qv = -2.0 * torch.log(
        (support_numerator / opacity_safe).clamp_min(1.0e-8)
    )
    half_extent = torch.sqrt(
        (support_qv.unsqueeze(-1) * _projected_uvt_inv_diag(q_uvt)).clamp_min(0.0)
    )
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
    compiled_opacity = compiled_projected_opacity(projected, config)
    tile_proxy = projected_tile_load_proxy(
        projected.ma,
        projected.q_uvt,
        compiled_opacity,
        config,
    )
    slope_proxy = projected_depth_slope_proxy(projected.depth_beta, config)
    margin_proxy = projected_depth_margin_proxy(
        projected.ma,
        projected.depth0,
        compiled_opacity,
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
    model: WorldTubeModel | SPD4WorldAtomModel,
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
    world_representation: str = "legacy_tube",
    spd4_min_spatial_scale: float = 1.0e-4,
    spd4_init_precision_z: float | None = None,
    progress_dir: Path | None = None,
) -> tuple[WorldTubeModel | SPD4WorldAtomModel, dict[str, Any], list[dict[str, Any]]]:
    if loss_scope not in {"sampled_frame", "view_sequence", "temporal_window", "paper_batch"}:
        raise ValueError("loss_scope must be one of: sampled_frame, view_sequence, temporal_window, paper_batch")
    if backend not in UVT_RENDER_BACKENDS:
        raise ValueError(f"backend must be one of: {', '.join(UVT_RENDER_BACKENDS)}")
    if backend not in UVT_FAST_METAL_BACKENDS and (
        reduction_mode != "index_add" or sample_emission_mode != "atomic_append"
    ):
        raise ValueError(
            "custom reduction/sample emission modes require backend=metal_tile "
            "or backend=hybrid_retained_fiber"
        )
    if backend in {"retained_fiber_metal", "hybrid_retained_fiber"}:
        if world_representation != "full_spd4":
            raise ValueError(f"{backend} requires world_representation=full_spd4")
        if render_config.alpha_mode != "beer_lambert":
            raise ValueError(f"{backend} requires alpha_mode=beer_lambert")
        if bundle.train_K.device.type != "mps":
            raise ValueError(f"{backend} requires MPS")
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
    if world_representation not in {"legacy_tube", "full_spd4"}:
        raise ValueError("world_representation must be one of: legacy_tube, full_spd4")
    if (
        world_representation == "full_spd4"
        and camera_sequence_mode
        not in {"static_view", "dynamic_first_order", "projective_first_order"}
    ):
        raise ValueError(
            "full_spd4 supports static_view, dynamic_first_order, and "
            "projective_first_order; segmented compilation is not implemented"
        )
    if spd4_min_spatial_scale <= 0.0:
        raise ValueError("spd4_min_spatial_scale must be positive")
    if spd4_init_precision_z is not None and spd4_init_precision_z <= 0.0:
        raise ValueError("spd4_init_precision_z must be positive when provided")
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
    if world_representation == "legacy_tube":
        model: WorldTubeModel | SPD4WorldAtomModel = WorldTubeModel(
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
            alpha_mode=render_config.alpha_mode,
            amplitude_convention=render_config.amplitude_convention,
            static_tube_count=int(init_metadata["static_tube_count"]),
            static_velocity_reg_weight=static_velocity_reg_weight,
        ).to(device)
    else:
        model = SPD4WorldAtomModel(
            init_x0=init_x0,
            init_color=init_color,
            init_t0=init_t0,
            frames=frames,
            init_precision_xy=init_precision_xy,
            init_precision_z=spd4_init_precision_z,
            init_lambda_t=init_lambda_t_values,
            init_opacity=init_opacity,
            min_spatial_scale=spd4_min_spatial_scale,
            min_lambda_t=min_lambda_t,
            tilt_reg_weight=velocity_reg_weight,
            depth_tilt_reg_weight=depth_velocity_reg_weight,
            position_reg_weight=position_reg_weight,
            alpha_mode=render_config.alpha_mode,
            amplitude_convention=render_config.amplitude_convention,
            static_tube_count=int(init_metadata["static_tube_count"]),
            static_tilt_reg_weight=static_velocity_reg_weight,
        ).to(device)
    full_config = render_config
    if full_config.height != height or full_config.width != width or full_config.frames != frames:
        raise ValueError("render_config dimensions must match bundle train frames")
    window_config = replace(
        full_config,
        height=height,
        width=width,
        frames=window_frames if loss_scope == "temporal_window" else frames,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    paper_sampler_seed = seed + int(paper_values.get("sampler_seed_offset", 7001))
    paper_sampler = (
        SpacetimeEpochSampler(
            view_count=len(active_train_views),
            frame_indices=active_train_frames,
            batch_size=max(stage.frames_per_step for stage in paper_stages),
            same_time_count=int(paper_values.get("same_time_count", 1)),
            local_time_count=int(paper_values.get("local_time_count", 0)),
            local_time_radius=int(paper_values.get("local_time_radius", 0)),
            seed=paper_sampler_seed,
        )
        if paper_enabled
        else None
    )
    paper_sample_schedule = (
        PaperSampleScheduleDigest(sampler_seed=paper_sampler_seed)
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
        stage_config = replace(
            full_config,
            height=stage.image_size.height,
            width=stage.image_size.width,
            frames=frames,
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
    fallback_tile_sums: list[Tensor] = []
    ambiguous_tile_sums: list[Tensor] = []
    invalid_tile_sums: list[Tensor] = []
    overflow_tile_sums: list[Tensor] = []
    active_atom_sums: list[Tensor] = []
    physical_tile_count = 0
    physical_render_calls = 0

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

    def render_for_training(
        projected: ProjectedTubeSequence,
        render_cfg: UVTRenderConfig,
    ) -> RenderedSequence:
        nonlocal physical_tile_count, physical_render_calls
        rendered = render_projected_sequence(
            projected,
            render_cfg,
            backend=backend,
            reduction_mode=reduction_mode,
            sample_emission_mode=sample_emission_mode,
        )
        if rendered.fallback_tiles is not None:
            detached_fallback = rendered.fallback_tiles.detach()
            fallback_tile_sums.append(detached_fallback.sum())
            physical_tile_count += int(detached_fallback.numel())
            physical_render_calls += 1
            if rendered.fallback_reason_bits is not None:
                reasons = rendered.fallback_reason_bits.detach()
                overflow_tile_sums.append(((reasons & 1) != 0).sum())
                invalid_tile_sums.append(((reasons & 2) != 0).sum())
                ambiguous_tile_sums.append(((reasons & 4) != 0).sum())
            if rendered.fallback_active_counts is not None:
                active_atom_sums.append(
                    rendered.fallback_active_counts.detach().sum()
                )
        return rendered

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
            paper_sample_schedule.record(
                step=step,
                stage=paper_stage,
                batch=paper_batch,
            )
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
            selected_frame_config = replace(
                step_full_config,
                frames=1,
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
                rendered = render_for_training(projected, selected_frame_config)
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
            rendered = render_for_training(projected, full_config)
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
            rendered = render_for_training(projected, full_config)
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
            rendered = render_for_training(projected, window_config)
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
                consistency_config = replace(
                    step_full_config,
                    frames=consistency_window_frames,
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
            sequence_rendered = render_for_training(
                sequence_projected,
                consistency_config,
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
            if progress_dir is not None and world_representation == "legacy_tube":
                write_json_atomic(
                    progress_dir / "latest.json",
                    {
                        "step": completed_step,
                        "elapsed_s": elapsed_after_step,
                        "logs": logs,
                        "checkpoint": _save_frozen_world_checkpoint(
                            model,
                            progress_dir / f"step_{completed_step:06d}.pt",
                            frame_count=frames,
                            representation=model.representation_name,
                        ),
                        "optimizer_state_saved": False,
                    },
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
    fallback_tile_count = (
        int(torch.stack(fallback_tile_sums).sum().detach().cpu())
        if fallback_tile_sums
        else 0
    )
    ambiguous_tile_count = (
        int(torch.stack(ambiguous_tile_sums).sum().detach().cpu())
        if ambiguous_tile_sums
        else 0
    )
    invalid_tile_count = (
        int(torch.stack(invalid_tile_sums).sum().detach().cpu())
        if invalid_tile_sums
        else 0
    )
    active_atom_total = (
        int(torch.stack(active_atom_sums).sum().detach().cpu())
        if active_atom_sums
        else 0
    )
    certificate_overflow_tile_count = (
        int(torch.stack(overflow_tile_sums).sum().detach().cpu())
        if overflow_tile_sums
        else 0
    )
    physical_visibility_stats = {
        "backend": backend,
        "render_calls": physical_render_calls,
        "tile_count": physical_tile_count,
        "fallback_tile_count": fallback_tile_count,
        "fallback_fraction": (
            float(fallback_tile_count / physical_tile_count)
            if physical_tile_count
            else None
        ),
        "ambiguous_tile_count": ambiguous_tile_count,
        "invalid_tile_count": invalid_tile_count,
        "certificate_overflow_tile_count": certificate_overflow_tile_count,
        "mean_active_atoms_per_tile": (
            float(active_atom_total / physical_tile_count)
            if physical_tile_count and active_atom_sums
            else None
        ),
        "retained_depth_samples": full_config.retained_depth_samples,
        "retained_sigma_extent": full_config.retained_sigma_extent,
        "order_certificate_sigma": full_config.order_certificate_sigma,
        "order_certificate_min_gap": full_config.order_certificate_min_gap,
        "bound_derivatives": "detached_compiler_decision",
    }
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
            **model.representation_metadata(),
            "init_source_amplitude": init_opacity,
            "init_center_alpha": (
                init_opacity
                if render_config.amplitude_convention == "fiber_integrated"
                else None
            ),
            "init_peak_density": (
                init_opacity
                if render_config.amplitude_convention == "peak_density"
                else None
            ),
            "spd4_min_spatial_scale": (
                spd4_min_spatial_scale if world_representation == "full_spd4" else None
            ),
            "spd4_init_precision_z": (
                (
                    init_precision_xy
                    if spd4_init_precision_z is None
                    else spd4_init_precision_z
                )
                if world_representation == "full_spd4"
                else None
            ),
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
                "alpha_mode": full_config.alpha_mode,
                "amplitude_convention": full_config.amplitude_convention,
                "opacity_semantics": full_config.opacity_semantics,
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
                "sample_schedule": (
                    paper_sample_schedule.snapshot()
                    if paper_sample_schedule is not None
                    else None
                ),
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
            "physical_visibility": physical_visibility_stats,
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
        fast_mac_options=(
            dict(PAPER_DYNAMIC_FAST_MAC_OPTIONS) if paper_enabled else None
        ),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    paper_sampler_seed = seed + int(paper_values.get("sampler_seed_offset", 7001))
    paper_sampler = (
        SpacetimeEpochSampler(
            view_count=view_count,
            frame_indices=range(frames),
            batch_size=max(stage.frames_per_step for stage in paper_stages),
            same_time_count=int(paper_values.get("same_time_count", 1)),
            local_time_count=int(paper_values.get("local_time_count", 0)),
            local_time_radius=int(paper_values.get("local_time_radius", 0)),
            seed=paper_sampler_seed,
        )
        if paper_enabled
        else None
    )
    paper_sample_schedule = (
        PaperSampleScheduleDigest(sampler_seed=paper_sampler_seed)
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
            fast_mac_options=(
                dict(PAPER_DYNAMIC_FAST_MAC_OPTIONS) if paper_enabled else None
            ),
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
        if paper_batch is not None:
            paper_sample_schedule.record(
                step=step,
                stage=paper_stage,
                batch=paper_batch,
            )
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
                    fast_mac_options=stage_render_cfg.fast_mac_options,
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
                forward=(
                    "fast_mac_v5_black"
                    if paper_enabled and renderer == "fast_mac"
                    else renderer
                ),
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
            "sample_schedule": (
                paper_sample_schedule.snapshot()
                if paper_sample_schedule is not None
                else None
            ),
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
        chunk_config = replace(
            config,
            frames=frame_stop - frame_start,
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
    ) -> tuple[
        list,
        list,
        list,
        dict[str, list[dict[str, float]]],
        dict[str, float],
    ]:
        rows = []
        metrics_rows = []
        render_times = []
        global_accumulator = VideoMetricAccumulator()
        global_lpips_sum = 0.0
        global_lpips_count = 0
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
                global_accumulator.update(rendered.rgb, target)
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
                    chunk_lpips = video_lpips(rendered.rgb, target)
                    lpips_sum += chunk_lpips * count
                    lpips_count += count
                    global_lpips_sum += chunk_lpips * count
                    global_lpips_count += count
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
        global_metrics = global_accumulator.metrics()
        if split == "heldout":
            global_metrics["eval_lpips"] = (
                global_lpips_sum / float(global_lpips_count)
            )
        return rows, metrics_rows, render_times, split_rows, global_metrics

    (
        train_rows,
        train_metrics,
        train_render_times,
        train_frame_split_metrics,
        train_global_metrics,
    ) = eval_split(
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
        (
            heldout_rows,
            heldout_metrics,
            heldout_render_times,
            _,
            heldout_global_metrics,
        ) = eval_split(
            split="heldout",
            frames_tensor=bundle.heldout_frames,
            K_all=bundle.heldout_K,
            w2c_all=bundle.heldout_w2c,
            lens_models=bundle.heldout_lens_models,
            distortions=bundle.heldout_distortions,
        )
    metrics = train_global_metrics
    for name, metrics_rows in train_frame_split_metrics.items():
        metrics.update(prefix_metrics(f"train_{name}_frame", aggregate_view_metrics(metrics_rows)))
    if heldout_metrics:
        metrics.update(prefix_metrics("heldout", heldout_global_metrics))
    metrics["eval_render_elapsed_s"] = time.perf_counter() - render_started
    metrics.update(render_time_metrics(train_render_times, heldout_render_times))
    return {
        "metrics": metrics,
        "train_rows": train_rows,
        "heldout_rows": heldout_rows,
        "train_view_metrics": train_metrics,
        "heldout_view_metrics": heldout_metrics,
    }


def _tensor_payload_bytes(values: tuple[Tensor | None, ...]) -> int:
    return sum(
        int(value.numel()) * int(value.element_size())
        for value in values
        if value is not None
    )


def _tensor_storage_descriptor(
    value: Tensor | None,
) -> tuple[str, tuple[int, ...], Callable[[], bytes]] | None:
    if value is None:
        return None
    dtype = str(value.dtype).removeprefix("torch.")
    shape = tuple(int(dimension) for dimension in value.shape)

    def materialize_bytes(tensor: Tensor = value) -> bytes:
        return (
            tensor.detach()
            .to(device="cpu")
            .contiguous()
            .numpy()
            .tobytes(order="C")
        )

    return dtype, shape, materialize_bytes


def _atlas_topology_payload(atlas) -> dict[str, Any]:
    return {
        **({"opacity_time_centered": True} if atlas.opacity_time_centered else {}),
        "source_window_indices": list(atlas.source_window_indices),
        "source_primitive_ids": list(atlas.source_primitive_ids),
        "active_start": list(atlas.active_start),
        "active_stop": list(atlas.active_stop),
        "cells": [
            {
                "tile_u": int(cell.tile_u),
                "tile_v": int(cell.tile_v),
                "start": int(cell.start),
                "stop": int(cell.stop),
                "primitive_ids": list(cell.primitive_ids),
                "ordered_primitive_ids": list(cell.ordered_primitive_ids),
                "depth_intervals": [
                    [float(lower), float(upper)]
                    for lower, upper in cell.depth_intervals
                ],
                "fallback": bool(cell.fallback),
                "fallback_reasons": list(cell.fallback_reasons),
            }
            for cell in atlas.cells
        ],
    }


def _write_frozen_atlas_storage(
    atlas,
    *,
    out_dir: Path,
    frame_count: int,
) -> dict[str, Any]:
    tensors = {
        name: _tensor_storage_descriptor(getattr(atlas, name))
        for name in FROZEN_ATLAS_TENSOR_NAMES
    }
    return write_retained_storage_artifact(
        out_dir
        / "frozen_world_retained_storage"
        / f"frame_{frame_count:04d}.world_tubes_atlas",
        frame_count=frame_count,
        trace_count=int(atlas.coeffs.shape[0]),
        cell_count=len(atlas.cells),
        tensors=tensors,
        topology=_atlas_topology_payload(atlas),
    )


def _clean_route_memory_baseline(device: torch.device) -> dict[str, int]:
    synchronize_device(device)
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    synchronize_device(device)
    return device_memory_stats(device)


def _memory_phase_stats(
    name: str,
    sampler: DeviceMemorySampler,
) -> dict[str, Any]:
    return {
        "name": name,
        **sampler.stats(),
    }


def _route_memory_report(
    route: str,
    *,
    device: torch.device,
    baseline: dict[str, int],
    phases: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_current = int(baseline.get("current_allocated_bytes", 0))
    baseline_driver = int(baseline.get("driver_allocated_bytes", 0))
    peak_current = max(
        [baseline_current]
        + [
            int(phase["sampled_peak_current_allocated_bytes"])
            for phase in phases
        ]
    )
    peak_driver = max(
        [baseline_driver]
        + [
            int(phase["sampled_peak_driver_allocated_bytes"])
            for phase in phases
        ]
    )
    sample_count = sum(int(phase["memory_sample_count"]) for phase in phases)
    eligible = (
        device.type in {"mps", "cuda"}
        and set(baseline)
        == {"current_allocated_bytes", "driver_allocated_bytes"}
        and bool(phases)
        and all(int(phase["memory_sample_count"]) > 0 for phase in phases)
        and peak_current >= baseline_current
        and peak_driver >= baseline_driver
    )
    return {
        "schema_version": 1,
        "route": route,
        "device_type": device.type,
        "route_scoped": True,
        "baseline_current_allocated_bytes": baseline_current,
        "baseline_driver_allocated_bytes": baseline_driver,
        "sampled_peak_current_allocated_bytes": peak_current,
        "sampled_peak_driver_allocated_bytes": peak_driver,
        "peak_increment_current_allocated_bytes": max(
            peak_current - baseline_current,
            0,
        ),
        "peak_increment_driver_allocated_bytes": max(
            peak_driver - baseline_driver,
            0,
        ),
        "memory_sample_count": sample_count,
        "phase_count": len(phases),
        "phases": phases,
        "measurement_claim_eligible": eligible,
    }


def _world_parameter_gradients(
    model: nn.Module,
) -> tuple[dict[str, Tensor], tuple[str, ...]]:
    gradients: dict[str, Tensor] = {}
    covered = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            covered.append(name)
        gradients[name] = (
            torch.zeros_like(parameter, device="cpu")
            if parameter.grad is None
            else parameter.grad.detach().cpu().clone()
        )
    return gradients, tuple(covered)


def _gradient_comparison(
    replay: dict[str, Tensor],
    compiled: dict[str, Tensor],
    *,
    replay_covered: tuple[str, ...],
    compiled_covered: tuple[str, ...],
) -> dict[str, Any]:
    if replay.keys() != compiled.keys():
        raise ValueError("frozen-world routes produced different parameter gradient keys")
    per_parameter: dict[str, float] = {}
    difference_sq = 0.0
    reference_sq = 0.0
    dot = 0.0
    replay_sq = 0.0
    compiled_sq = 0.0
    for name in replay:
        replay_grad = replay[name].to(dtype=torch.float64)
        compiled_grad = compiled[name].to(dtype=torch.float64)
        difference = replay_grad - compiled_grad
        replay_norm = float(torch.linalg.vector_norm(replay_grad))
        compiled_norm = float(torch.linalg.vector_norm(compiled_grad))
        difference_norm = float(torch.linalg.vector_norm(difference))
        per_parameter[name] = difference_norm / max(
            replay_norm + compiled_norm,
            1.0e-12,
        )
        difference_sq += float(torch.sum(difference.square()))
        reference_sq += float(torch.sum(replay_grad.square() + compiled_grad.square()))
        dot += float(torch.sum(replay_grad * compiled_grad))
        replay_sq += float(torch.sum(replay_grad.square()))
        compiled_sq += float(torch.sum(compiled_grad.square()))
    return {
        "global_normalized_l2_error": math.sqrt(difference_sq)
        / max(math.sqrt(reference_sq), 1.0e-12),
        "cosine_similarity": dot
        / max(math.sqrt(replay_sq) * math.sqrt(compiled_sq), 1.0e-12),
        "replay_l2_norm": math.sqrt(replay_sq),
        "compiled_l2_norm": math.sqrt(compiled_sq),
        "parameter_tensor_count": len(replay),
        "replay_gradient_tensor_count": len(replay_covered),
        "compiled_gradient_tensor_count": len(compiled_covered),
        "gradient_coverage_matches": replay_covered == compiled_covered,
        "replay_gradient_parameters": list(replay_covered),
        "compiled_gradient_parameters": list(compiled_covered),
        "max_parameter_normalized_l2_error": max(per_parameter.values(), default=0.0),
        "per_parameter_normalized_l2_error": per_parameter,
    }


def _world_state_digest(
    state: dict[str, Tensor],
    *,
    metadata: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _world_state_metadata(
    model: nn.Module,
    *,
    frame_count: int,
    representation: str,
) -> dict[str, Any]:
    return {
        "representation": representation,
        "frame_count": int(frame_count),
        "active_tube_count": int(model.active_tube_count),
        "tube_count": int(model.tube_count),
        "alpha_mode": str(model.alpha_mode),
        "amplitude_convention": str(model.amplitude_convention),
        "min_precision_xy": float(model.min_precision_xy),
        "min_lambda_t": float(model.min_lambda_t),
        "parameter_names": [name for name, _ in model.named_parameters()],
    }


def _tensor_sha256(value: Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape)).encode("utf-8"))
    detached = value.detach()
    if detached.ndim == 0:
        chunks = (detached.reshape(1),)
    else:
        chunks = (detached[index : index + 1] for index in range(detached.shape[0]))
    for chunk in chunks:
        digest.update(
            chunk.to(device="cpu").contiguous().numpy().tobytes(order="C")
        )
    return digest.hexdigest()


def _frozen_evaluation_contract_hashes(
    *,
    target_frames: Tensor,
    heldout_K: Tensor,
    heldout_w2c: Tensor,
    heldout_distortion: Tensor | None,
    heldout_lens_model: str,
    heldout_camera: str,
    camera_projection: str,
    full_frames: int,
    frame_count: int,
    frame_indices: tuple[int, ...],
    centered_frame_times: tuple[float, ...],
    config: UVTRenderConfig,
) -> dict[str, str]:
    target_sha = _tensor_sha256(target_frames)
    frame_indices_sha = frozen_world_sequence_sha256(frame_indices)
    centered_frame_times_sha = frozen_world_sequence_sha256(
        centered_frame_times
    )
    camera_digest = hashlib.sha256()
    camera_digest.update(_tensor_sha256(heldout_K).encode("ascii"))
    camera_digest.update(_tensor_sha256(heldout_w2c).encode("ascii"))
    camera_digest.update(
        (
            "none"
            if heldout_distortion is None
            else _tensor_sha256(heldout_distortion)
        ).encode("ascii")
    )
    camera_digest.update(
        json.dumps(
            {
                "heldout_camera": heldout_camera,
                "heldout_lens_model": heldout_lens_model,
                "camera_projection": camera_projection,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    camera_sha = camera_digest.hexdigest()
    evaluation_digest = hashlib.sha256()
    evaluation_digest.update(target_sha.encode("ascii"))
    evaluation_digest.update(camera_sha.encode("ascii"))
    evaluation_digest.update(frame_indices_sha.encode("ascii"))
    evaluation_digest.update(centered_frame_times_sha.encode("ascii"))
    evaluation_digest.update(
        json.dumps(
            {
                "full_frames": int(full_frames),
                "frame_count": int(frame_count),
                "temporal_sampling": (
                    "ordered_full_interval_integer_lattice_v1"
                ),
                "image_size": [int(config.height), int(config.width)],
                "alpha_mode": config.alpha_mode,
                "amplitude_convention": config.amplitude_convention,
                "alpha_threshold": float(config.alpha_threshold),
                "tile_x": int(config.tile_x),
                "tile_y": int(config.tile_y),
                "tile_t": int(config.tile_t),
                "loss": "sqrt(error^2 + 1e-6) / global_element_count",
                "replay_backend": "metal_tile:index_add:direct_atomic",
                "compiled_backend": "projective_cell_interval:mixed",
                "dtype": "float32",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "target_frames_sha256": target_sha,
        "camera_program_sha256": camera_sha,
        "frame_indices_sha256": frame_indices_sha,
        "centered_frame_times_sha256": centered_frame_times_sha,
        "evaluation_contract_sha256": evaluation_digest.hexdigest(),
    }


def _save_frozen_world_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    frame_count: int,
    representation: str,
) -> dict[str, Any]:
    state = snapshot_world_tube_state(model)
    metadata = _world_state_metadata(
        model,
        frame_count=frame_count,
        representation=representation,
    )
    world_state_sha256 = _world_state_digest(state, metadata=metadata)
    if set(metadata["parameter_names"]) != set(state):
        raise RuntimeError(
            "frozen checkpoint state tensors do not match named parameters"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            **metadata,
            "world_state_sha256": world_state_sha256,
            "state_dict": state,
        },
        path,
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": int(path.stat().st_size),
        "parameter_tensor_count": len(metadata["parameter_names"]),
        "world_state_sha256": world_state_sha256,
        **metadata,
    }


def _frozen_compiled_full_vs_sliced_parity(
    model: WorldTubeModel,
    *,
    heldout_K: Tensor,
    heldout_w2c: Tensor,
    heldout_lens_model: str,
    heldout_distortion: Tensor | None,
    camera_projection: str,
    render_config: UVTRenderConfig,
    config: UVTRenderConfig,
    full_frames: int,
    frame_indices: tuple[int, ...],
    centered_frame_times: tuple[float, ...],
    target_host: Tensor,
    contract_hashes: dict[str, str],
) -> dict[str, Any]:
    """Certify one non-unit atlas against one-frame slices of that same atlas.

    This is intentionally bounded to the one sweep row selected by the caller.
    It is correctness evidence, not a performance measurement.
    """

    frame_count = len(centered_frame_times)
    if len(frame_indices) != frame_count:
        raise ValueError(
            "selected-time atlas-slice parity frame/time count drifted"
        )
    time_steps = tuple(
        centered_frame_times[index + 1] - centered_frame_times[index]
        for index in range(frame_count - 1)
    )
    non_unit_selected_times = any(
        not math.isclose(abs(step), 1.0, rel_tol=0.0, abs_tol=1.0e-7)
        for step in time_steps
    )
    if frame_count < 2 or not non_unit_selected_times:
        raise ValueError(
            "selected-time atlas-slice parity requires at least two non-unit-spaced times"
        )

    device = next(model.parameters()).device
    world_state_metadata = _world_state_metadata(
        model,
        frame_count=full_frames,
        representation=model.representation_name,
    )
    world_state_before = _world_state_digest(
        snapshot_world_tube_state(model),
        metadata=world_state_metadata,
    )
    model.zero_grad(set_to_none=True)
    projection_config = replace(render_config, frames=full_frames)
    projected = project_world_tube_sequence(
        model,
        heldout_K,
        heldout_w2c,
        projection_config,
        camera_projection=camera_projection,
        lens_model=heldout_lens_model,
        distortion=heldout_distortion,
        full_frames=full_frames,
        frame_start=0,
    )
    times = torch.tensor(
        centered_frame_times,
        dtype=torch.float32,
        device=device,
    ).contiguous()
    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        projected.ma,
        projected.q_uvt,
        projected.depth0,
        projected.depth_beta,
        compiled_projected_opacity(projected, projection_config),
        projected.color,
        times,
        sigma_px=1.0,
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        alpha_threshold=float(config.alpha_threshold),
        require_isotropic_spatial=False,
        temporal_mode="centered",
        auto_support_padding_from_alpha=True,
        allow_depth_affine_uv=True,
        stratify_visibility=True,
        mark_visibility_fallback=True,
    )
    full_state = ProjectiveCellIntervalTrainerState(
        atlas=atlas,
        times=times,
        config=config,
        sigma_px=1.0,
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        fallback_render_mode="mixed",
    )
    target_device = target_host.to(device=device)
    target_element_count = int(target_host.numel())
    full_image = full_state.render()
    full_loss = torch.sqrt(
        (full_image - target_device).square() + 1.0e-6
    ).sum() / float(target_element_count)
    full_loss.backward(retain_graph=True)
    synchronize_device(device)
    full_gradients, full_gradient_parameters = _world_parameter_gradients(model)
    full_loss_value = float(full_loss.detach().cpu())
    world_state_after_full = _world_state_digest(
        snapshot_world_tube_state(model),
        metadata=world_state_metadata,
    )

    model.zero_grad(set_to_none=True)
    sliced_images: list[Tensor] = []
    sliced_loss_value = 0.0
    cumulative_sliced_trace_count = 0
    cumulative_sliced_cell_count = 0
    for sample_index in range(frame_count):
        chunk_atlas = slice_projective_trace_cell_atlas_frames(
            atlas,
            start=sample_index,
            stop=sample_index + 1,
        )
        cumulative_sliced_trace_count += int(chunk_atlas.coeffs.shape[0])
        cumulative_sliced_cell_count += len(chunk_atlas.cells)
        chunk_state = ProjectiveCellIntervalTrainerState(
            atlas=chunk_atlas,
            times=times[sample_index : sample_index + 1].contiguous(),
            config=replace(config, frames=1),
            sigma_px=1.0,
            image_width=int(config.width),
            image_height=int(config.height),
            tile_size=int(config.tile_x),
            fallback_render_mode="mixed",
        )
        sliced_image = chunk_state.render()
        sliced_loss = torch.sqrt(
            (
                sliced_image
                - target_device[sample_index : sample_index + 1]
            ).square()
            + 1.0e-6
        ).sum() / float(target_element_count)
        sliced_loss.backward(retain_graph=sample_index + 1 < frame_count)
        sliced_loss_value += float(sliced_loss.detach().cpu())
        sliced_images.append(sliced_image.detach())
        del chunk_atlas, chunk_state, sliced_image, sliced_loss
    synchronize_device(device)
    sliced_gradients, sliced_gradient_parameters = _world_parameter_gradients(
        model
    )
    world_state_after_sliced = _world_state_digest(
        snapshot_world_tube_state(model),
        metadata=world_state_metadata,
    )
    model.zero_grad(set_to_none=True)

    sliced_image_all = torch.cat(sliced_images, dim=0)
    image_difference = (full_image.detach() - sliced_image_all).abs()
    image_max_abs_error = float(image_difference.max().cpu())
    image_mean_abs_error = float(image_difference.mean().cpu())
    loss_absolute_delta = abs(full_loss_value - sliced_loss_value)
    gradient = _gradient_comparison(
        full_gradients,
        sliced_gradients,
        replay_covered=full_gradient_parameters,
        compiled_covered=sliced_gradient_parameters,
    )
    same_world_state = (
        world_state_before
        == world_state_after_full
        == world_state_after_sliced
    )
    acceptance = {
        "image_max_abs_error": FROZEN_WORLD_ACCEPTANCE[
            "image_max_abs_error"
        ],
        "loss_absolute_delta": FROZEN_WORLD_ACCEPTANCE[
            "loss_absolute_delta"
        ],
        "gradient_global_normalized_l2_error": FROZEN_WORLD_ACCEPTANCE[
            "gradient_global_normalized_l2_error"
        ],
        "gradient_max_parameter_normalized_l2_error": FROZEN_WORLD_ACCEPTANCE[
            "gradient_max_parameter_normalized_l2_error"
        ],
        "min_world_vjp_l2_norm": FROZEN_WORLD_ACCEPTANCE[
            "min_world_vjp_l2_norm"
        ],
    }
    checks = {
        "non_unit_selected_times": non_unit_selected_times,
        "same_parent_atlas": True,
        "world_state_unchanged": same_world_state,
        "image_matches": image_max_abs_error
        <= acceptance["image_max_abs_error"],
        "loss_matches": loss_absolute_delta
        <= acceptance["loss_absolute_delta"],
        "world_vjp_matches": gradient["global_normalized_l2_error"]
        <= acceptance["gradient_global_normalized_l2_error"],
        "world_vjp_per_parameter_matches": gradient[
            "max_parameter_normalized_l2_error"
        ]
        <= acceptance["gradient_max_parameter_normalized_l2_error"],
        "world_vjp_nonzero": min(
            gradient["replay_l2_norm"],
            gradient["compiled_l2_norm"],
        )
        > acceptance["min_world_vjp_l2_norm"],
        "world_vjp_coverage_matches": (
            bool(gradient["gradient_coverage_matches"])
            and int(gradient["replay_gradient_tensor_count"])
            == int(gradient["parameter_tensor_count"])
            and int(gradient["compiled_gradient_tensor_count"])
            == int(gradient["parameter_tensor_count"])
        ),
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "accepted": all(checks.values()),
        "scope": (
            "one bounded non-unit selected-time atlas rendered whole versus "
            "one-frame compact slices of the same parent atlas"
        ),
        "timing_claim_eligible": False,
        "frame_count": frame_count,
        "full_dataset_frame_count": full_frames,
        "frame_indices": list(frame_indices),
        "centered_frame_times": list(centered_frame_times),
        "time_steps": list(time_steps),
        "slice_chunk_frames": 1,
        "slice_count": frame_count,
        "parent_atlas_trace_count": int(atlas.coeffs.shape[0]),
        "parent_atlas_cell_count": len(atlas.cells),
        "cumulative_sliced_trace_count": cumulative_sliced_trace_count,
        "cumulative_sliced_cell_count": cumulative_sliced_cell_count,
        "contract_hashes": dict(contract_hashes),
        "world_state": {
            "before_sha256": world_state_before,
            "after_full_atlas_sha256": world_state_after_full,
            "after_sliced_atlas_sha256": world_state_after_sliced,
            "unchanged": same_world_state,
        },
        "loss": {
            "full_atlas": full_loss_value,
            "chunk_sliced": sliced_loss_value,
            "absolute_delta": loss_absolute_delta,
        },
        "image": {
            "max_abs_error": image_max_abs_error,
            "mean_abs_error": image_mean_abs_error,
        },
        "gradient": {
            "global_normalized_l2_error": gradient[
                "global_normalized_l2_error"
            ],
            "cosine_similarity": gradient["cosine_similarity"],
            "full_atlas_l2_norm": gradient["replay_l2_norm"],
            "chunk_sliced_l2_norm": gradient["compiled_l2_norm"],
            "parameter_tensor_count": gradient["parameter_tensor_count"],
            "full_atlas_gradient_tensor_count": gradient[
                "replay_gradient_tensor_count"
            ],
            "chunk_sliced_gradient_tensor_count": gradient[
                "compiled_gradient_tensor_count"
            ],
            "gradient_coverage_matches": gradient[
                "gradient_coverage_matches"
            ],
            "max_parameter_normalized_l2_error": gradient[
                "max_parameter_normalized_l2_error"
            ],
            "per_parameter_normalized_l2_error": gradient[
                "per_parameter_normalized_l2_error"
            ],
            "full_atlas_gradient_parameters": gradient[
                "replay_gradient_parameters"
            ],
            "chunk_sliced_gradient_parameters": gradient[
                "compiled_gradient_parameters"
            ],
        },
        "acceptance": acceptance,
        "checks": checks,
    }
    del (
        projected,
        times,
        atlas,
        full_state,
        target_device,
        full_image,
        full_loss,
        sliced_images,
        sliced_image_all,
        image_difference,
    )
    gc.collect()
    torch.mps.empty_cache()
    return result


def frozen_world_replay_compiled_report(
    model: WorldTubeModel,
    bundle,
    *,
    render_config: UVTRenderConfig,
    camera_projection: str,
    out_dir: Path,
    max_frames: int = 0,
    checkpoint: dict[str, Any] | None = None,
    verify_selected_time_slice_parity: bool = False,
    timing_warmups: int = 0,
    timing_repeats: int = 1,
) -> dict[str, Any]:
    """Compare per-frame replay and one interval atlas from one frozen world.

    Both routes consume the same trained model, held-out camera, target frames,
    loss, alpha law, and float32 precision. The replay route reprojects and
    bins a one-frame STAR sequence for every target time. The compiled route
    projects once, lowers one event-stratified interval atlas, and evaluates
    the same target times through the native compiled forward/VJP.
    """

    validate_frozen_world_timing_controls(
        warmups=timing_warmups,
        repeats=timing_repeats,
    )
    device = next(model.parameters()).device
    if device.type != "mps":
        raise ValueError("frozen replay/compiled comparison requires MPS")
    if model.representation_name != "legacy_tube":
        raise ValueError("frozen replay/compiled comparison currently requires legacy_tube")
    if render_config.alpha_mode != "peak_splat":
        raise ValueError("frozen replay/compiled comparison currently requires peak_splat")
    if render_config.tile_x != render_config.tile_y:
        raise ValueError("projective interval atlas requires equal spatial tile dimensions")
    if (
        bundle.heldout_frames is None
        or bundle.heldout_K is None
        or bundle.heldout_w2c is None
        or int(bundle.heldout_frames.shape[0]) < 1
    ):
        raise ValueError("frozen replay/compiled comparison requires a held-out camera")

    full_frames = int(bundle.frame_count)
    frame_count = full_frames if max_frames <= 0 else min(int(max_frames), full_frames)
    if frame_count < 1:
        raise ValueError("frozen replay/compiled frame count must be positive")
    frame_indices = frozen_world_full_interval_frame_indices(
        full_frames,
        frame_count,
    )
    centered_frame_times = tuple(
        float(frame) - 0.5 * float(full_frames - 1)
        for frame in frame_indices
    )
    config = replace(render_config, frames=frame_count)
    target_indices = torch.tensor(
        frame_indices,
        dtype=torch.long,
        device=bundle.heldout_frames.device,
    )
    target_host = (
        bundle.heldout_frames[0].index_select(0, target_indices)
        .permute(0, 2, 3, 1)
        .to(device="cpu", dtype=torch.float32)
    )
    resident_chunk_frames = max(
        1,
        min(int(render_config.tile_t), frame_count),
    )
    target_element_count = int(target_host.numel())
    heldout_K = select_view_K(bundle.heldout_K, 0)
    heldout_w2c = select_view_w2c(bundle.heldout_w2c, 0)
    heldout_lens_model, heldout_distortion = select_lens(
        bundle.heldout_lens_models,
        bundle.heldout_distortions,
        0,
        camera_projection=camera_projection,
    )
    contract_hashes = _frozen_evaluation_contract_hashes(
        target_frames=target_host,
        heldout_K=heldout_K,
        heldout_w2c=heldout_w2c,
        heldout_distortion=heldout_distortion,
        heldout_lens_model=heldout_lens_model,
        heldout_camera=bundle.heldout_camera_names[0],
        camera_projection=camera_projection,
        full_frames=full_frames,
        frame_count=frame_count,
        frame_indices=frame_indices,
        centered_frame_times=centered_frame_times,
        config=config,
    )
    if checkpoint is None:
        checkpoint = _save_frozen_world_checkpoint(
            model,
            out_dir / "world_tubes_frozen_final_state.pt",
            frame_count=full_frames,
            representation=model.representation_name,
        )
    world_state_metadata = _world_state_metadata(
        model,
        frame_count=full_frames,
        representation=model.representation_name,
    )
    world_state_before_routes = _world_state_digest(
        snapshot_world_tube_state(model),
        metadata=world_state_metadata,
    )
    if world_state_before_routes != checkpoint["world_state_sha256"]:
        raise RuntimeError("saved frozen checkpoint does not match the live world")

    if verify_selected_time_slice_parity:
        selected_time_slice_parity = _frozen_compiled_full_vs_sliced_parity(
            model,
            heldout_K=heldout_K,
            heldout_w2c=heldout_w2c,
            heldout_lens_model=heldout_lens_model,
            heldout_distortion=heldout_distortion,
            camera_projection=camera_projection,
            render_config=render_config,
            config=config,
            full_frames=full_frames,
            frame_indices=frame_indices,
            centered_frame_times=centered_frame_times,
            target_host=target_host,
            contract_hashes=contract_hashes,
        )
        if (
            selected_time_slice_parity["world_state"]["before_sha256"]
            != checkpoint["world_state_sha256"]
            or selected_time_slice_parity["world_state"][
                "after_sliced_atlas_sha256"
            ]
            != checkpoint["world_state_sha256"]
        ):
            raise RuntimeError(
                "selected-time atlas-slice parity changed the frozen world"
            )
    else:
        selected_time_slice_parity = {
            "schema_version": 1,
            "status": "not_run",
            "accepted": False,
            "reason": (
                "sweep runs this bounded proof only on its smallest "
                "non-unit selected-time row"
            ),
            "timing_claim_eligible": False,
        }

    model.zero_grad(set_to_none=True)
    replay_memory_baseline = _clean_route_memory_baseline(device)
    replay_memory_sampler = DeviceMemorySampler(device)
    replay_memory_sampler.start()
    replay_forward_s = 0.0
    replay_backward_s = 0.0
    replay_loss_value = 0.0
    replay_payload_bytes = 0
    for chunk_start in range(0, frame_count, resident_chunk_frames):
        chunk_stop = min(frame_count, chunk_start + resident_chunk_frames)
        synchronize_device(device)
        replay_forward_started = time.perf_counter()
        replay_frames: list[Tensor] = []
        for sample_index in range(chunk_start, chunk_stop):
            frame = frame_indices[sample_index]
            frame_config = replace(config, frames=1)
            projected_frame = project_world_tube_sequence(
                model,
                heldout_K,
                heldout_w2c,
                frame_config,
                camera_projection=camera_projection,
                lens_model=heldout_lens_model,
                distortion=heldout_distortion,
                full_frames=full_frames,
                frame_start=frame,
            )
            replay_payload_bytes += _tensor_payload_bytes(
                (
                    projected_frame.ma,
                    projected_frame.q_uvt,
                    projected_frame.depth0,
                    projected_frame.depth_beta,
                    projected_frame.opacity,
                    projected_frame.color,
                )
            )
            replay_frames.append(
                render_projected_sequence(
                    projected_frame,
                    frame_config,
                    backend="metal_tile",
                    reduction_mode="index_add",
                    sample_emission_mode="direct_atomic",
                ).rgb
            )
        replay_image = torch.cat(replay_frames, dim=0)
        target_chunk = target_host[chunk_start:chunk_stop].to(device=device)
        replay_loss = torch.sqrt(
            (replay_image - target_chunk).square() + 1.0e-6
        ).sum() / float(target_element_count)
        synchronize_device(device)
        replay_forward_s += time.perf_counter() - replay_forward_started
        replay_backward_started = time.perf_counter()
        replay_loss.backward()
        synchronize_device(device)
        replay_backward_s += time.perf_counter() - replay_backward_started
        replay_loss_value += float(replay_loss.detach().cpu())
        del (
            projected_frame,
            replay_frames,
            replay_image,
            replay_loss,
            target_chunk,
        )
    replay_memory_sampler.stop()
    replay_memory_phases = [
        _memory_phase_stats(
            "correctness_forward_backward",
            replay_memory_sampler,
        )
    ]
    replay_route_memory = _route_memory_report(
        "replay",
        device=device,
        baseline=replay_memory_baseline,
        phases=replay_memory_phases,
    )
    replay_gradients, replay_gradient_parameters = _world_parameter_gradients(model)
    world_state_after_replay = _world_state_digest(
        snapshot_world_tube_state(model),
        metadata=_world_state_metadata(
            model,
            frame_count=full_frames,
            representation=model.representation_name,
        ),
    )

    model.zero_grad(set_to_none=True)
    compiled_memory_baseline = _clean_route_memory_baseline(device)
    compiled_memory_phases: list[dict[str, Any]] = []
    compiled_compile_memory_sampler = DeviceMemorySampler(device)
    compiled_compile_memory_sampler.start()
    synchronize_device(device)
    compiled_compile_started = time.perf_counter()
    projection_config = replace(render_config, frames=full_frames)
    projected = project_world_tube_sequence(
        model,
        heldout_K,
        heldout_w2c,
        projection_config,
        camera_projection=camera_projection,
        lens_model=heldout_lens_model,
        distortion=heldout_distortion,
        full_frames=full_frames,
        frame_start=0,
    )
    times = torch.tensor(
        centered_frame_times,
        dtype=torch.float32,
        device=device,
    ).contiguous()
    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        projected.ma,
        projected.q_uvt,
        projected.depth0,
        projected.depth_beta,
        compiled_projected_opacity(projected, projection_config),
        projected.color,
        times,
        sigma_px=1.0,
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        alpha_threshold=float(config.alpha_threshold),
        require_isotropic_spatial=False,
        temporal_mode="centered",
        auto_support_padding_from_alpha=True,
        allow_depth_affine_uv=True,
        stratify_visibility=True,
        mark_visibility_fallback=True,
    )
    compiled_state = ProjectiveCellIntervalTrainerState(
        atlas=atlas,
        times=times,
        config=config,
        sigma_px=1.0,
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        fallback_render_mode="mixed",
    )
    synchronize_device(device)
    compiled_compile_s = time.perf_counter() - compiled_compile_started
    compiled_compile_memory_sampler.stop()
    compiled_memory_phases.append(
        _memory_phase_stats(
            "atlas_compile",
            compiled_compile_memory_sampler,
        )
    )
    fallback = compiled_state.fallback_stats()
    complexity = compiled_state.complexity_stats()
    compiled_payload_bytes = _tensor_payload_bytes(
        (
            atlas.coeffs,
            atlas.opacity,
            atlas.opacity_time_coeffs,
            atlas.spatial_precision_uv,
            atlas.depth_affine_uv,
            atlas.depth_reference_uvt,
            atlas.alpha_cutoff_reference_uvt,
            atlas.color,
        )
    )
    compiled_trace_count = int(atlas.coeffs.shape[0])
    compiled_cell_count = len(atlas.cells)

    compiled_forward_s = 0.0
    compiled_backward_s = 0.0
    parity_replay_forward_s = 0.0
    compiled_loss_value = 0.0
    image_max_abs_error = 0.0
    image_absolute_error_sum = 0.0
    for chunk_start in range(0, frame_count, resident_chunk_frames):
        chunk_stop = min(frame_count, chunk_start + resident_chunk_frames)
        synchronize_device(device)
        compiled_forward_memory_sampler = DeviceMemorySampler(device)
        compiled_forward_memory_sampler.start()
        compiled_forward_started = time.perf_counter()
        chunk_config = replace(config, frames=chunk_stop - chunk_start)
        chunk_times = times[chunk_start:chunk_stop].contiguous()
        chunk_atlas = slice_projective_trace_cell_atlas_frames(
            atlas,
            start=chunk_start,
            stop=chunk_stop,
        )
        chunk_state = ProjectiveCellIntervalTrainerState(
            atlas=chunk_atlas,
            times=chunk_times,
            config=chunk_config,
            sigma_px=1.0,
            image_width=int(config.width),
            image_height=int(config.height),
            tile_size=int(config.tile_x),
            fallback_render_mode="mixed",
        )
        target_chunk = target_host[chunk_start:chunk_stop].to(device=device)
        compiled_image = chunk_state.render()
        compiled_loss = torch.sqrt(
            (compiled_image - target_chunk).square() + 1.0e-6
        ).sum() / float(target_element_count)
        synchronize_device(device)
        compiled_forward_s += time.perf_counter() - compiled_forward_started
        compiled_forward_memory_sampler.stop()
        compiled_memory_phases.append(
            _memory_phase_stats(
                f"chunk_{chunk_start:04d}_{chunk_stop:04d}_forward",
                compiled_forward_memory_sampler,
            )
        )

        synchronize_device(device)
        parity_replay_started = time.perf_counter()
        with torch.no_grad():
            parity_frames: list[Tensor] = []
            for sample_index in range(chunk_start, chunk_stop):
                frame = frame_indices[sample_index]
                frame_config = replace(config, frames=1)
                parity_projected = project_world_tube_sequence(
                    model,
                    heldout_K,
                    heldout_w2c,
                    frame_config,
                    camera_projection=camera_projection,
                    lens_model=heldout_lens_model,
                    distortion=heldout_distortion,
                    full_frames=full_frames,
                    frame_start=frame,
                )
                parity_frames.append(
                    render_projected_sequence(
                        parity_projected,
                        frame_config,
                        backend="metal_tile",
                        reduction_mode="index_add",
                        sample_emission_mode="direct_atomic",
                    ).rgb
                )
            parity_image = torch.cat(parity_frames, dim=0)
            image_difference = (compiled_image.detach() - parity_image).abs()
            image_max_abs_error = max(
                image_max_abs_error,
                float(image_difference.max().cpu()),
            )
            image_absolute_error_sum += float(image_difference.sum().cpu())
        synchronize_device(device)
        parity_replay_forward_s += time.perf_counter() - parity_replay_started
        del (
            parity_projected,
            parity_frames,
            parity_image,
            image_difference,
        )
        gc.collect()
        torch.mps.empty_cache()
        synchronize_device(device)

        compiled_backward_memory_sampler = DeviceMemorySampler(device)
        compiled_backward_memory_sampler.start()
        compiled_backward_started = time.perf_counter()
        compiled_loss.backward(retain_graph=chunk_stop < frame_count)
        synchronize_device(device)
        compiled_backward_s += time.perf_counter() - compiled_backward_started
        compiled_backward_memory_sampler.stop()
        compiled_memory_phases.append(
            _memory_phase_stats(
                f"chunk_{chunk_start:04d}_{chunk_stop:04d}_backward",
                compiled_backward_memory_sampler,
            )
        )
        compiled_loss_value += float(compiled_loss.detach().cpu())
        del (
            chunk_atlas,
            chunk_state,
            chunk_times,
            compiled_image,
            compiled_loss,
            target_chunk,
        )

    compiled_route_memory = _route_memory_report(
        "compiled",
        device=device,
        baseline=compiled_memory_baseline,
        phases=compiled_memory_phases,
    )
    compiled_gradients, compiled_gradient_parameters = _world_parameter_gradients(
        model
    )
    world_state_after_compiled = _world_state_digest(
        snapshot_world_tube_state(model),
        metadata=_world_state_metadata(
            model,
            frame_count=full_frames,
            representation=model.representation_name,
        ),
    )
    model.zero_grad(set_to_none=True)

    gradient = _gradient_comparison(
        replay_gradients,
        compiled_gradients,
        replay_covered=replay_gradient_parameters,
        compiled_covered=compiled_gradient_parameters,
    )
    image_mean_abs_error = image_absolute_error_sum / float(target_element_count)
    loss_absolute_delta = abs(compiled_loss_value - replay_loss_value)
    acceptance = dict(FROZEN_WORLD_ACCEPTANCE)
    same_checkpoint = (
        checkpoint["world_state_sha256"]
        == world_state_before_routes
        == world_state_after_replay
        == world_state_after_compiled
    )
    checks = {
        "checkpoint_matches": same_checkpoint,
        "image_matches": image_max_abs_error
        <= acceptance["image_max_abs_error"],
        "loss_matches": loss_absolute_delta
        <= acceptance["loss_absolute_delta"],
        "world_vjp_matches": gradient["global_normalized_l2_error"]
        <= acceptance["gradient_global_normalized_l2_error"],
        "world_vjp_per_parameter_matches": gradient[
            "max_parameter_normalized_l2_error"
        ]
        <= acceptance["gradient_max_parameter_normalized_l2_error"],
        "world_vjp_nonzero": min(
            gradient["replay_l2_norm"],
            gradient["compiled_l2_norm"],
        )
        > acceptance["min_world_vjp_l2_norm"],
        "world_vjp_coverage_matches": (
            bool(gradient["gradient_coverage_matches"])
            and int(gradient["replay_gradient_tensor_count"])
            == int(gradient["parameter_tensor_count"])
            and int(gradient["compiled_gradient_tensor_count"])
            == int(gradient["parameter_tensor_count"])
        ),
        "fallback_within_budget": fallback.fallback_fraction
        <= acceptance["fallback_fraction"],
    }
    retained_storage_artifact = _write_frozen_atlas_storage(
        atlas,
        out_dir=out_dir,
        frame_count=frame_count,
    )
    retained_storage = {
        "schema_version": 1,
        "definition": RETAINED_STORAGE_DEFINITION,
        "shared_checkpoint_bytes": int(checkpoint["bytes"]),
        "shared_checkpoint_excluded_from_route_totals": True,
        "replay": {
            "route": "replay",
            "serialized_retained_evaluator_bytes": 0,
            "topology_applicable": False,
            "storage_claim_eligible": True,
            "reason": REPLAY_STORAGE_REASON,
        },
        "compiled": {
            "route": "compiled",
            "serialized_retained_evaluator_bytes": int(
                retained_storage_artifact["bytes"]
            ),
            "tensor_payload_bytes": int(
                retained_storage_artifact["tensor_payload_bytes"]
            ),
            "topology_and_container_bytes": int(
                retained_storage_artifact["topology_and_container_bytes"]
            ),
            "topology_bytes_included": True,
            "artifact": retained_storage_artifact,
            "storage_claim_eligible": True,
        },
        "topology_bytes_included": True,
        "storage_claim_eligible": True,
        "publication_claim_eligible": True,
    }
    route_memory = {
        "schema_version": 1,
        "definition": ROUTE_MEMORY_DEFINITION,
        "measurement_source": ROUTE_MEMORY_MEASUREMENT_SOURCE,
        "sampler_interval_ms": 5.0,
        "compiled_parity_replay_excluded": True,
        "replay": replay_route_memory,
        "compiled": compiled_route_memory,
        "publication_claim_eligible": (
            replay_route_memory["measurement_claim_eligible"] is True
            and compiled_route_memory["measurement_claim_eligible"] is True
        ),
    }
    del projected, times, atlas, compiled_state
    gc.collect()
    torch.mps.empty_cache()

    def replay_timing_trial() -> tuple[float, float]:
        model.zero_grad(set_to_none=True)
        total_forward_s = 0.0
        total_backward_s = 0.0
        for chunk_start in range(0, frame_count, resident_chunk_frames):
            chunk_stop = min(frame_count, chunk_start + resident_chunk_frames)
            synchronize_device(device)
            forward_started = time.perf_counter()
            trial_frames: list[Tensor] = []
            for sample_index in range(chunk_start, chunk_stop):
                frame = frame_indices[sample_index]
                frame_config = replace(config, frames=1)
                trial_projected = project_world_tube_sequence(
                    model,
                    heldout_K,
                    heldout_w2c,
                    frame_config,
                    camera_projection=camera_projection,
                    lens_model=heldout_lens_model,
                    distortion=heldout_distortion,
                    full_frames=full_frames,
                    frame_start=frame,
                )
                trial_frames.append(
                    render_projected_sequence(
                        trial_projected,
                        frame_config,
                        backend="metal_tile",
                        reduction_mode="index_add",
                        sample_emission_mode="direct_atomic",
                    ).rgb
                )
            trial_image = torch.cat(trial_frames, dim=0)
            trial_target = target_host[chunk_start:chunk_stop].to(device=device)
            trial_loss = torch.sqrt(
                (trial_image - trial_target).square() + 1.0e-6
            ).sum() / float(target_element_count)
            synchronize_device(device)
            total_forward_s += time.perf_counter() - forward_started
            backward_started = time.perf_counter()
            trial_loss.backward()
            synchronize_device(device)
            total_backward_s += time.perf_counter() - backward_started
            del (
                trial_projected,
                trial_frames,
                trial_image,
                trial_target,
                trial_loss,
            )
        model.zero_grad(set_to_none=True)
        return total_forward_s, total_backward_s

    def compiled_timing_trial() -> tuple[float, float, float]:
        model.zero_grad(set_to_none=True)
        synchronize_device(device)
        compile_started = time.perf_counter()
        trial_projection_config = replace(render_config, frames=full_frames)
        trial_projected = project_world_tube_sequence(
            model,
            heldout_K,
            heldout_w2c,
            trial_projection_config,
            camera_projection=camera_projection,
            lens_model=heldout_lens_model,
            distortion=heldout_distortion,
            full_frames=full_frames,
            frame_start=0,
        )
        trial_times = torch.tensor(
            centered_frame_times,
            dtype=torch.float32,
            device=device,
        ).contiguous()
        trial_atlas = uvt_tubes_to_projective_trace_cell_atlas(
            trial_projected.ma,
            trial_projected.q_uvt,
            trial_projected.depth0,
            trial_projected.depth_beta,
            compiled_projected_opacity(
                trial_projected,
                trial_projection_config,
            ),
            trial_projected.color,
            trial_times,
            sigma_px=1.0,
            image_width=int(config.width),
            image_height=int(config.height),
            tile_size=int(config.tile_x),
            alpha_threshold=float(config.alpha_threshold),
            require_isotropic_spatial=False,
            temporal_mode="centered",
            auto_support_padding_from_alpha=True,
            allow_depth_affine_uv=True,
            stratify_visibility=True,
            mark_visibility_fallback=True,
        )
        synchronize_device(device)
        compile_s = time.perf_counter() - compile_started

        total_forward_s = 0.0
        total_backward_s = 0.0
        for chunk_start in range(0, frame_count, resident_chunk_frames):
            chunk_stop = min(frame_count, chunk_start + resident_chunk_frames)
            synchronize_device(device)
            forward_started = time.perf_counter()
            trial_chunk_atlas = slice_projective_trace_cell_atlas_frames(
                trial_atlas,
                start=chunk_start,
                stop=chunk_stop,
            )
            trial_chunk_state = ProjectiveCellIntervalTrainerState(
                atlas=trial_chunk_atlas,
                times=trial_times[chunk_start:chunk_stop].contiguous(),
                config=replace(config, frames=chunk_stop - chunk_start),
                sigma_px=1.0,
                image_width=int(config.width),
                image_height=int(config.height),
                tile_size=int(config.tile_x),
                fallback_render_mode="mixed",
            )
            trial_target = target_host[chunk_start:chunk_stop].to(device=device)
            trial_image = trial_chunk_state.render()
            trial_loss = torch.sqrt(
                (trial_image - trial_target).square() + 1.0e-6
            ).sum() / float(target_element_count)
            synchronize_device(device)
            total_forward_s += time.perf_counter() - forward_started
            backward_started = time.perf_counter()
            trial_loss.backward(retain_graph=chunk_stop < frame_count)
            synchronize_device(device)
            total_backward_s += time.perf_counter() - backward_started
            del (
                trial_chunk_atlas,
                trial_chunk_state,
                trial_target,
                trial_image,
                trial_loss,
            )
        model.zero_grad(set_to_none=True)
        del trial_projected, trial_times, trial_atlas
        return compile_s, total_forward_s, total_backward_s

    def complete_timing_sample(
        *,
        replay_forward: float,
        replay_backward: float,
        compiled_compile: float,
        compiled_forward: float,
        compiled_backward: float,
    ) -> dict[str, float]:
        return {
            "replay_total_forward": replay_forward,
            "replay_total_backward": replay_backward,
            "replay_total_forward_backward": (
                replay_forward + replay_backward
            ),
            "replay_per_frame_forward": replay_forward / float(frame_count),
            "replay_per_frame_backward": replay_backward / float(frame_count),
            "compiled_atlas_compile": compiled_compile,
            "compiled_total_forward": compiled_forward,
            "compiled_total_backward": compiled_backward,
            "compiled_total_forward_backward": (
                compiled_forward + compiled_backward
            ),
            "compiled_compile_plus_forward_backward": (
                compiled_compile + compiled_forward + compiled_backward
            ),
            "compiled_per_frame_forward": (
                compiled_forward / float(frame_count)
            ),
            "compiled_per_frame_backward": (
                compiled_backward / float(frame_count)
            ),
        }

    timing_publication_ready = (
        timing_warmups >= FROZEN_WORLD_MIN_TIMING_WARMUPS
        and timing_repeats >= FROZEN_WORLD_MIN_TIMING_REPEATS
    )
    if timing_warmups == 0 and timing_repeats == 1:
        timing_samples = [
            complete_timing_sample(
                replay_forward=replay_forward_s,
                replay_backward=replay_backward_s,
                compiled_compile=compiled_compile_s,
                compiled_forward=compiled_forward_s,
                compiled_backward=compiled_backward_s,
            )
        ]
        timing_label = "single_shot_correctness_timing"
        timing_measurement_source = "backward_compatible_correctness_pass"
    else:
        timing_samples = []
        for timing_trial_index in range(timing_warmups + timing_repeats):
            if timing_trial_index % 2 == 0:
                trial_replay_forward, trial_replay_backward = (
                    replay_timing_trial()
                )
                (
                    trial_compiled_compile,
                    trial_compiled_forward,
                    trial_compiled_backward,
                ) = compiled_timing_trial()
            else:
                (
                    trial_compiled_compile,
                    trial_compiled_forward,
                    trial_compiled_backward,
                ) = compiled_timing_trial()
                trial_replay_forward, trial_replay_backward = (
                    replay_timing_trial()
                )
            if timing_trial_index >= timing_warmups:
                timing_samples.append(
                    complete_timing_sample(
                        replay_forward=trial_replay_forward,
                        replay_backward=trial_replay_backward,
                        compiled_compile=trial_compiled_compile,
                        compiled_forward=trial_compiled_forward,
                        compiled_backward=trial_compiled_backward,
                    )
                )
            gc.collect()
        timing_label = (
            "warmed_repeated_wall_timing_v1"
            if timing_publication_ready
            else "diagnostic_repeated_wall_timing_v1"
        )
        timing_measurement_source = (
            "independent_alternating_paired_trials"
        )
    timing_sample_columns = tuple(timing_samples[0])
    timing_samples_by_metric = {
        key: [sample[key] for sample in timing_samples]
        for key in timing_sample_columns
    }
    timing_benchmark = {
        "schema_version": 1,
        "status": "complete",
        "label": timing_label,
        "publication_ready": timing_publication_ready,
        "warmups": timing_warmups,
        "repeats": timing_repeats,
        "measurement_source": timing_measurement_source,
        "timing_definition": (
            "device-synchronized perf_counter segments; forward includes "
            "target transfer; compile includes world projection; summed totals "
            "exclude inter-segment cleanup and optimizer work"
        ),
        "route_order": (
            "alternating_paired_replay_compiled_v1"
            if timing_measurement_source
            == "independent_alternating_paired_trials"
            else "correctness_pass_replay_then_compiled"
        ),
        "device_synchronized_at_boundaries": True,
        "compiled_evaluator_uses_chunk_slices": True,
        "forward_includes_cpu_target_to_device_transfer": True,
        "compiled_atlas_compile_includes_world_projection": True,
        "backward_excludes_optimizer": True,
        "resident_chunk_frames": resident_chunk_frames,
        "correctness_and_slice_parity_time_excluded": (
            timing_measurement_source
            == "independent_alternating_paired_trials"
        ),
        "samples_s": timing_samples_by_metric,
        "summary_s": {
            key: frozen_world_timing_summary(values)
            for key, values in timing_samples_by_metric.items()
        },
    }
    world_state_after_timing = _world_state_digest(
        snapshot_world_tube_state(model),
        metadata=world_state_metadata,
    )
    if world_state_after_timing != checkpoint["world_state_sha256"]:
        raise RuntimeError("frozen-world timing changed the learned world")

    return {
        "schema_version": 2,
        "status": "complete",
        "accepted": all(checks.values()),
        "scope": (
            "one frozen learned world; heldout view 0; ordered samples are "
            "selected from one fixed full-duration camera/world program"
        ),
        "checkpoint": checkpoint,
        "world_state": {
            "checkpoint_sha256": checkpoint["world_state_sha256"],
            "before_routes_sha256": world_state_before_routes,
            "after_replay_sha256": world_state_after_replay,
            "after_compiled_sha256": world_state_after_compiled,
            "matches_checkpoint": same_checkpoint,
        },
        "heldout_camera": bundle.heldout_camera_names[0],
        "frame_count": frame_count,
        "full_dataset_frame_count": full_frames,
        "frame_indices": list(frame_indices),
        "centered_frame_times": list(centered_frame_times),
        "temporal_sampling": "ordered_full_interval_integer_lattice_v1",
        "image_size": [int(config.height), int(config.width)],
        "loss": {
            "name": "robust_l1",
            "replay": replay_loss_value,
            "compiled": compiled_loss_value,
            "absolute_delta": loss_absolute_delta,
        },
        "image": {
            "max_abs_error": image_max_abs_error,
            "mean_abs_error": image_mean_abs_error,
        },
        "gradient": gradient,
        "selected_time_slice_parity": selected_time_slice_parity,
        "timing_s": {
            "replay_total_forward": replay_forward_s,
            "replay_total_backward": replay_backward_s,
            "replay_per_frame_forward": replay_forward_s / float(frame_count),
            "replay_per_frame_backward": replay_backward_s / float(frame_count),
            "compiled_atlas_compile": compiled_compile_s,
            "compiled_total_forward": compiled_forward_s,
            "compiled_total_backward": compiled_backward_s,
            "compiled_per_frame_forward": compiled_forward_s / float(frame_count),
            "compiled_per_frame_backward": compiled_backward_s / float(frame_count),
            "parity_replay_total_forward": parity_replay_forward_s,
        },
        "timing_benchmark": timing_benchmark,
        "payload_bytes": {
            "schema_version": 1,
            "metric_kind": "logical_work_volume_proxy",
            "definition": LOGICAL_PAYLOAD_DEFINITION,
            "topology_bytes_included": False,
            "storage_claim_eligible": False,
            "publication_claim_eligible": False,
            "replay_cumulative_logical_tensor_bytes": replay_payload_bytes,
            "compiled_trace_table_logical_tensor_bytes": compiled_payload_bytes,
            "compiled_to_replay_logical_volume_ratio": compiled_payload_bytes
            / max(replay_payload_bytes, 1),
        },
        "retained_storage_bytes": retained_storage,
        "route_memory": route_memory,
        "atlas": {
            "trace_count": compiled_trace_count,
            "cell_count": compiled_cell_count,
            "interval_trace_entries": complexity.interval_trace_entries,
            "dense_trace_samples": complexity.dense_trace_samples,
            "interval_to_dense_trace_sample_ratio": complexity.interval_to_dense_trace_sample_ratio,
            "fallback_cells": fallback.fallback_cells,
            "total_tile_samples": fallback.total_tile_samples,
            "fallback_tile_samples": fallback.fallback_tile_samples,
            "fallback_fraction": fallback.fallback_fraction,
            "fallback_reasons": list(fallback.fallback_reasons),
        },
        "contract": {
            "same_checkpoint": same_checkpoint,
            "same_heldout_camera": True,
            "same_target_frames": True,
            "same_loss": True,
            "same_precision": True,
            "same_alpha_mode": True,
            "bounded_device_frame_residency": True,
            "host_target_storage": "eager_cpu_selected_frames",
            "resident_chunk_frames": resident_chunk_frames,
            "timing_excludes_parity_replay": True,
            "camera_projection": camera_projection,
            "temporal_sampling": (
                "ordered integer frames spanning the full dataset interval"
            ),
            "replay_route": (
                "one-frame STAR projection/bin/render per selected global time"
            ),
            "compiled_route": (
                "one event-stratified interval atlas evaluated by bounded "
                "frame chunks through native forward/VJP"
            ),
        },
        "contract_hashes": contract_hashes,
        "acceptance": acceptance,
        "checks": checks,
    }


def frozen_world_replay_compiled_sweep_report(
    model: WorldTubeModel,
    bundle,
    *,
    render_config: UVTRenderConfig,
    camera_projection: str,
    out_dir: Path,
    primary_max_frames: int = 0,
    requested_frame_counts: tuple[int, ...] | None = None,
    timing_warmups: int = 0,
    timing_repeats: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate full-interval temporal sample densities from one frozen world."""

    validate_frozen_world_timing_controls(
        warmups=timing_warmups,
        repeats=timing_repeats,
    )
    full_frames = int(bundle.frame_count)
    resolved_frame_counts = resolve_frozen_world_frame_counts(
        full_frames=full_frames,
        primary_max_frames=primary_max_frames,
        requested_frame_counts=requested_frame_counts,
    )
    progress_path = out_dir / "frozen_world_sweep_progress.json"
    progress: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "full_dataset_frame_count": full_frames,
        "requested_frame_counts": list(requested_frame_counts or ()),
        "primary_requested_frame_count": int(primary_max_frames),
        "resolved_frame_counts": list(resolved_frame_counts),
        "timing_warmups": timing_warmups,
        "timing_repeats": timing_repeats,
        "completed_frame_counts": [],
        "row_artifacts": [],
        "cross_process_resume_supported": False,
    }
    write_json(progress_path, progress)
    slice_parity_frame_count = next(
        (
            frame_count
            for frame_count in resolved_frame_counts
            if frame_count < full_frames
            and any(
                right - left != 1
                for left, right in zip(
                    frozen_world_full_interval_frame_indices(
                        full_frames,
                        frame_count,
                    ),
                    frozen_world_full_interval_frame_indices(
                        full_frames,
                        frame_count,
                    )[1:],
                )
            )
        ),
        None,
    )
    rows: list[dict[str, Any]] = []
    try:
        checkpoint = _save_frozen_world_checkpoint(
            model,
            out_dir / "world_tubes_frozen_final_state.pt",
            frame_count=full_frames,
            representation=model.representation_name,
        )
        progress["status"] = "checkpoint_saved"
        progress["checkpoint"] = checkpoint
        write_json(progress_path, progress)
        for frame_count in resolved_frame_counts:
            progress["status"] = "running"
            progress["current_frame_count"] = frame_count
            write_json(progress_path, progress)
            row = frozen_world_replay_compiled_report(
                model,
                bundle,
                render_config=render_config,
                camera_projection=camera_projection,
                out_dir=out_dir,
                max_frames=frame_count,
                checkpoint=checkpoint,
                verify_selected_time_slice_parity=(
                    frame_count == slice_parity_frame_count
                ),
                timing_warmups=timing_warmups,
                timing_repeats=timing_repeats,
            )
            if row["checkpoint"] != checkpoint:
                raise RuntimeError(
                    "frozen-world sweep checkpoint identity drifted"
                )
            if (
                row["world_state"]["checkpoint_sha256"]
                != checkpoint["world_state_sha256"]
            ):
                raise RuntimeError(
                    "frozen-world sweep world-state identity drifted"
                )
            checkpoint_path = Path(checkpoint["path"])
            if (
                int(checkpoint_path.stat().st_size) != int(checkpoint["bytes"])
                or file_sha256(checkpoint_path) != checkpoint["sha256"]
            ):
                raise RuntimeError(
                    "frozen-world sweep checkpoint file drifted"
                )
            row_path = (
                out_dir
                / "frozen_world_sweep_rows"
                / f"frame_{frame_count:04d}.json"
            )
            write_json(row_path, row)
            progress["completed_frame_counts"].append(frame_count)
            progress["row_artifacts"].append(
                {
                    "frame_count": frame_count,
                    "path": str(row_path.resolve()),
                    "sha256": file_sha256(row_path),
                }
            )
            progress["current_frame_count"] = None
            write_json(progress_path, progress)
            rows.append(row)
            gc.collect()
            torch.mps.empty_cache()
    except BaseException as error:
        progress["status"] = "failed"
        progress["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        progress["failed_frame_count"] = progress.get(
            "current_frame_count"
        )
        write_json(progress_path, progress)
        raise

    primary_frame_count = (
        full_frames
        if primary_max_frames <= 0
        else min(int(primary_max_frames), full_frames)
    )
    primary = next(
        row for row in rows if int(row["frame_count"]) == primary_frame_count
    )
    all_rows_accepted = all(row["accepted"] is True for row in rows)
    parity_row = next(
        (
            row
            for row in rows
            if int(row["frame_count"]) == slice_parity_frame_count
        ),
        None,
    )
    selected_time_slice_parity_accepted = (
        parity_row is not None
        and parity_row["selected_time_slice_parity"]["accepted"] is True
    )
    all_rows_timing_publication_ready = all(
        row["timing_benchmark"]["publication_ready"] is True
        for row in rows
    )
    all_rows_storage_publication_ready = all(
        row["retained_storage_bytes"]["publication_claim_eligible"] is True
        for row in rows
    )
    all_rows_route_memory_publication_ready = all(
        row["route_memory"]["publication_claim_eligible"] is True
        for row in rows
    )
    sweep = {
        "schema_version": 1,
        "status": "complete",
        "timing_label": "single_shot_correctness_timing",
        "timing_repeats": 1,
        "timing_warmups": 0,
        "timing_benchmark_label": (
            "warmed_repeated_wall_timing_v1"
            if all_rows_timing_publication_ready
            else rows[0]["timing_benchmark"]["label"]
        ),
        "timing_benchmark_repeats": timing_repeats,
        "timing_benchmark_warmups": timing_warmups,
        "all_rows_timing_publication_ready": (
            all_rows_timing_publication_ready
        ),
        "all_rows_storage_publication_ready": (
            all_rows_storage_publication_ready
        ),
        "all_rows_route_memory_publication_ready": (
            all_rows_route_memory_publication_ready
        ),
        "temporal_sampling": "ordered_full_interval_integer_lattice_v1",
        "requested_frame_counts": list(requested_frame_counts or ()),
        "primary_requested_frame_count": int(primary_max_frames),
        "primary_resolved_frame_count": primary_frame_count,
        "resolved_frame_counts": list(resolved_frame_counts),
        "full_dataset_frame_count": full_frames,
        "shared_checkpoint": checkpoint,
        "shared_checkpoint_file_sha256": checkpoint["sha256"],
        "shared_world_state_sha256": checkpoint["world_state_sha256"],
        "checkpoint_shared_across_rows": True,
        "world_state_shared_across_rows": True,
        "progress_artifact": str(progress_path.resolve()),
        "row_artifacts": list(progress["row_artifacts"]),
        "cross_process_resume_supported": False,
        "selected_time_slice_parity_frame_count": (
            slice_parity_frame_count
        ),
        "selected_time_slice_parity_accepted": (
            selected_time_slice_parity_accepted
        ),
        "all_rows_accepted": all_rows_accepted,
        "publication_eligible": (
            all_rows_accepted
            and frozen_world_sweep_publication_eligible(
                requested_frame_counts=requested_frame_counts,
                full_frames=full_frames,
                timing_warmups=timing_warmups,
                timing_repeats=timing_repeats,
                selected_time_slice_parity_accepted=(
                    selected_time_slice_parity_accepted
                ),
                all_rows_storage_publication_ready=(
                    all_rows_storage_publication_ready
                ),
                all_rows_route_memory_publication_ready=(
                    all_rows_route_memory_publication_ready
                ),
            )
        ),
        "rows": rows,
    }
    progress["status"] = "complete"
    progress["current_frame_count"] = None
    progress["all_rows_accepted"] = all_rows_accepted
    progress["publication_eligible"] = sweep["publication_eligible"]
    write_json(progress_path, progress)
    return primary, sweep


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

    def eval_split(
        split: str,
        frames_tensor: Tensor,
    ) -> tuple[list, list, list, dict[str, float]]:
        rows = []
        metrics_rows = []
        render_times = []
        global_accumulator = VideoMetricAccumulator()
        global_lpips_sum = 0.0
        global_lpips_count = 0
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
                global_accumulator.update(rendered["rgb"], target)
                if split == "heldout":
                    count = stop - start
                    chunk_lpips = video_lpips(rendered["rgb"], target)
                    lpips_sum += chunk_lpips * count
                    lpips_count += count
                    global_lpips_sum += chunk_lpips * count
                    global_lpips_count += count
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
        global_metrics = global_accumulator.metrics()
        if split == "heldout":
            global_metrics["eval_lpips"] = (
                global_lpips_sum / float(global_lpips_count)
            )
        return rows, metrics_rows, render_times, global_metrics

    (
        train_rows,
        train_metrics,
        train_render_times,
        train_global_metrics,
    ) = eval_split("train", bundle.train_frames)
    heldout_rows: list = []
    heldout_metrics: list = []
    heldout_render_times: list = []
    if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
        (
            heldout_rows,
            heldout_metrics,
            heldout_render_times,
            heldout_global_metrics,
        ) = eval_split("heldout", bundle.heldout_frames)
    metrics = train_global_metrics
    if heldout_metrics:
        metrics.update(prefix_metrics("heldout", heldout_global_metrics))
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
        "fast_mac_options": splat_render_cfg.fast_mac_options,
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
    parser.add_argument(
        "--uvt-world-representation",
        choices=("legacy_tube", "full_spd4"),
        default="legacy_tube",
        help="Keep the historical restricted tube as default; opt into native mean+SPD(4) atoms explicitly.",
    )
    parser.add_argument(
        "--uvt-alpha-mode",
        choices=("peak_splat", "beer_lambert"),
        default="peak_splat",
        help=(
            "Interpret the trainable opacity field as bounded peak alpha or "
            "nonnegative unbounded peak optical thickness."
        ),
    )
    parser.add_argument(
        "--uvt-amplitude-convention",
        choices=("fiber_integrated", "peak_density"),
        default="fiber_integrated",
        help=(
            "Use camera-compiled peak optical thickness, or train a native "
            "world peak extinction density and multiply by the affine fiber "
            "Jacobian and sqrt(2*pi*conditional_depth_variance)."
        ),
    )
    parser.add_argument("--uvt-spd4-min-spatial-scale", type=float, default=1.0e-4)
    parser.add_argument(
        "--uvt-spd4-init-precision-z",
        type=float,
        default=None,
        help=(
            "Initial conditional depth precision for full_spd4. The default "
            "matches --uvt-init-precision-xy; use a large value for a "
            "near-planar legacy-lift initialization."
        ),
    )
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
    parser.add_argument(
        "--uvt-render-backend",
        choices=UVT_RENDER_BACKENDS,
        default="dense",
    )
    parser.add_argument(
        "--uvt-retained-depth-samples",
        type=int,
        default=48,
        help="Midpoint depth samples for retained-fiber fallback pixels.",
    )
    parser.add_argument(
        "--uvt-retained-sigma-extent",
        type=float,
        default=6.0,
        help="Conditional-depth standard deviations retained by fiber quadrature.",
    )
    parser.add_argument(
        "--uvt-order-certificate-sigma",
        type=float,
        default=6.0,
        help=(
            "Depth-band radius used by hybrid tile certificates; must be at "
            "least --uvt-retained-sigma-extent."
        ),
    )
    parser.add_argument(
        "--uvt-order-certificate-min-gap",
        type=float,
        default=0.0,
        help="Additional required separation between certified depth bands.",
    )
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
        "--frozen-world-replay-compiled",
        action="store_true",
        help=(
            "After training, compare per-frame STAR replay against one compiled "
            "projective interval atlas from the identical frozen heldout world."
        ),
    )
    parser.add_argument(
        "--frozen-world-max-frames",
        type=int,
        default=0,
        help=(
            "Primary fixed-program sample count; zero uses every dataset "
            "frame, and positive counts span the full temporal interval."
        ),
    )
    parser.add_argument(
        "--frozen-world-frame-counts",
        default=None,
        help=(
            "Optional comma-separated same-checkpoint sweep; zero means the "
            "full dataset. Each resolved unique full-interval density is "
            "evaluated from the same frozen checkpoint."
        ),
    )
    parser.add_argument(
        "--frozen-world-timing-warmups",
        type=int,
        default=0,
        help=(
            "Unreported synchronized replay/compiled timing pairs per frozen "
            "frame-count row."
        ),
    )
    parser.add_argument(
        "--frozen-world-timing-repeats",
        type=int,
        default=1,
        help=(
            "Reported synchronized replay/compiled timing pairs per frozen "
            "frame-count row; publication timing requires at least three."
        ),
    )
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
    frozen_world_frame_counts = parse_frozen_world_frame_counts(
        args.frozen_world_frame_counts
    )

    if args.only_lane == "world_tubes":
        args.skip_splats = True

    if args.torch_deterministic != "off":
        torch.use_deterministic_algorithms(True, warn_only=args.torch_deterministic == "warn")

    device = resolve_device(args.device)
    if args.frozen_world_max_frames < 0:
        raise ValueError("--frozen-world-max-frames must be nonnegative")
    validate_frozen_world_timing_controls(
        warmups=args.frozen_world_timing_warmups,
        repeats=args.frozen_world_timing_repeats,
    )
    if args.frozen_world_max_frames and not args.frozen_world_replay_compiled:
        raise ValueError(
            "--frozen-world-max-frames requires --frozen-world-replay-compiled"
        )
    if (
        frozen_world_frame_counts is not None
        and not args.frozen_world_replay_compiled
    ):
        raise ValueError(
            "--frozen-world-frame-counts requires "
            "--frozen-world-replay-compiled"
        )
    if (
        args.frozen_world_timing_warmups != 0
        or args.frozen_world_timing_repeats != 1
    ) and not args.frozen_world_replay_compiled:
        raise ValueError(
            "frozen-world timing controls require "
            "--frozen-world-replay-compiled"
        )
    if args.frozen_world_replay_compiled:
        if args.only_lane == "dynamic_3dgs":
            raise ValueError("frozen replay/compiled comparison requires the World Tubes lane")
        if device.type != "mps":
            raise ValueError("frozen replay/compiled comparison requires device=mps")
        if args.uvt_world_representation != "legacy_tube":
            raise ValueError("frozen replay/compiled comparison requires legacy_tube")
        if args.uvt_alpha_mode != "peak_splat":
            raise ValueError("frozen replay/compiled comparison requires peak_splat")
        if args.uvt_render_backend != "metal_tile":
            raise ValueError("frozen replay/compiled comparison requires metal_tile")
        if args.uvt_camera_sequence_mode != "static_view":
            raise ValueError("frozen replay/compiled comparison requires static_view")
        if any(
            (
                args.uvt_synthetic_pan_x,
                args.uvt_synthetic_pan_y,
                args.uvt_synthetic_dolly_z,
                args.uvt_synthetic_zoom,
                args.uvt_synthetic_principal_x,
                args.uvt_synthetic_principal_y,
            )
        ):
            raise ValueError(
                "frozen replay/compiled comparison requires the recorded heldout camera"
            )
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
        if args.uvt_render_backend not in UVT_FAST_METAL_BACKENDS:
            raise ValueError(
                "--uvt-backward-policy requires --uvt-render-backend "
                "metal_tile or hybrid_retained_fiber"
            )
    if args.uvt_render_backend in UVT_NATIVE_METAL_BACKENDS and device.type != "mps":
        raise ValueError(
            f"--uvt-render-backend={args.uvt_render_backend} requires device=mps"
        )
    if args.uvt_render_backend not in UVT_FAST_METAL_BACKENDS and (
        args.uvt_reduction_mode != "index_add" or args.uvt_sample_emission_mode != "atomic_append"
    ):
        raise ValueError(
            "custom UVT reduction/sample emission modes require "
            "--uvt-render-backend metal_tile or hybrid_retained_fiber"
        )
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
    if (
        args.uvt_alpha_mode == "beer_lambert"
        and args.uvt_render_backend in UVT_NATIVE_METAL_BACKENDS
    ):
        if args.uvt_world_representation != "full_spd4":
            raise ValueError(
                "Beer-Lambert Metal paper runs are scoped to "
                "--uvt-world-representation=full_spd4"
            )
        if args.uvt_camera_sequence_mode != "static_view":
            raise ValueError(
                "Beer-Lambert Metal paper runs are scoped to "
                "--uvt-camera-sequence-mode=static_view"
            )
        if args.uvt_render_backend in UVT_FAST_METAL_BACKENDS and (
            args.uvt_reduction_mode != "index_add"
            or args.uvt_sample_emission_mode != "direct_atomic"
        ):
            raise ValueError(
                "Beer-Lambert Metal training requires the validated "
                "direct_atomic+index_add q-UVT backward path"
            )
    if (
        args.uvt_render_backend
        in {"retained_fiber_metal", "hybrid_retained_fiber"}
        and args.uvt_alpha_mode != "beer_lambert"
    ):
        raise ValueError(
            f"{args.uvt_render_backend} requires --uvt-alpha-mode=beer_lambert"
        )
    if args.uvt_amplitude_convention == "peak_density":
        if args.uvt_alpha_mode != "beer_lambert":
            raise ValueError(
                "--uvt-amplitude-convention=peak_density requires "
                "--uvt-alpha-mode=beer_lambert"
            )
        if args.uvt_world_representation != "full_spd4":
            raise ValueError(
                "--uvt-amplitude-convention=peak_density requires "
                "--uvt-world-representation=full_spd4"
            )
    render_config = UVTRenderConfig(
        height=int(bundle.train_frames.shape[-2]),
        width=int(bundle.train_frames.shape[-1]),
        frames=int(bundle.frame_count),
        tile_x=args.uvt_tile_x,
        tile_y=args.uvt_tile_y,
        tile_t=args.uvt_tile_t,
        tile_capacity=args.uvt_tile_capacity,
        max_alpha=max_alpha_for_mode(args.uvt_alpha_mode),
        alpha_mode=args.uvt_alpha_mode,
        amplitude_convention=args.uvt_amplitude_convention,
        retained_depth_samples=args.uvt_retained_depth_samples,
        retained_sigma_extent=args.uvt_retained_sigma_extent,
        order_certificate_sigma=args.uvt_order_certificate_sigma,
        order_certificate_min_gap=args.uvt_order_certificate_min_gap,
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
        "paper_dataset_bundle": paper_dataset_bundle_identity(
            bundle,
            image_size=load_image_size,
        ),
        "paper_evaluator": paper_evaluator_contract(),
        "paper_runtime": paper_runtime_identity(),
        "route_native_extension": (
            fast_mac_v5_native_extension_identity()
            if args.only_lane == "dynamic_3dgs"
            else star_uvt_native_extension_identity()
        ),
        "uvt_world_representation": args.uvt_world_representation,
        "uvt_alpha_mode": render_config.alpha_mode,
        "uvt_amplitude_convention": render_config.amplitude_convention,
        "uvt_opacity_semantics": render_config.opacity_semantics,
        "uvt_init_source_amplitude": args.uvt_init_opacity,
        "uvt_init_center_alpha": (
            args.uvt_init_opacity
            if render_config.amplitude_convention == "fiber_integrated"
            else None
        ),
        "uvt_init_peak_density": (
            args.uvt_init_opacity
            if render_config.amplitude_convention == "peak_density"
            else None
        ),
        "uvt_render_backend": args.uvt_render_backend,
        "uvt_retained_depth_samples": render_config.retained_depth_samples,
        "uvt_retained_sigma_extent": render_config.retained_sigma_extent,
        "uvt_order_certificate_sigma": render_config.order_certificate_sigma,
        "uvt_order_certificate_min_gap": render_config.order_certificate_min_gap,
        "uvt_spd4_init_precision_z": (
            (
                args.uvt_init_precision_xy
                if args.uvt_spd4_init_precision_z is None
                else args.uvt_spd4_init_precision_z
            )
            if args.uvt_world_representation == "full_spd4"
            else None
        ),
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
        "frozen_world_replay_compiled": args.frozen_world_replay_compiled,
        "frozen_world_max_frames": args.frozen_world_max_frames,
        "frozen_world_temporal_sampling": (
            "ordered_full_interval_integer_lattice_v1"
            if args.frozen_world_replay_compiled
            else None
        ),
        "frozen_world_frame_counts": (
            None
            if frozen_world_frame_counts is None
            else list(frozen_world_frame_counts)
        ),
        "frozen_world_timing_warmups": (
            args.frozen_world_timing_warmups
        ),
        "frozen_world_timing_repeats": (
            args.frozen_world_timing_repeats
        ),
        "star_uvt_native_extension": star_uvt_native_extension_identity(),
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
        world_representation=args.uvt_world_representation,
        spd4_min_spatial_scale=args.uvt_spd4_min_spatial_scale,
        spd4_init_precision_z=args.uvt_spd4_init_precision_z,
        progress_dir=out_dir / "training_progress",
    )
    # Save the learned state and optimizer-run evidence before evaluation can
    # fail (for example, on overflow in a previously unsampled view/time).
    final_world_checkpoint = (
        _save_frozen_world_checkpoint(
            uvt_model,
            out_dir / (
                "world_tubes_training_final_state.pt"
                if args.frozen_world_replay_compiled
                else "world_tubes_frozen_final_state.pt"
            ),
            frame_count=int(bundle.frame_count),
            representation=uvt_model.representation_name,
        )
        if args.uvt_world_representation == "legacy_tube"
        else None
    )
    write_json_atomic(
        out_dir / "world_tubes_training_result.json",
        {
            "meta": run_meta,
            "training": uvt_train,
            "final_world_checkpoint": final_world_checkpoint,
            "evaluation_status": "not_started_at_checkpoint_save",
        },
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
        del selected_eval
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

    if args.frozen_world_replay_compiled:
        frozen_world_report, frozen_world_sweep = (
            frozen_world_replay_compiled_sweep_report(
                uvt_model,
                bundle,
                render_config=render_config,
                camera_projection=args.uvt_camera_projection,
                out_dir=out_dir,
                primary_max_frames=args.frozen_world_max_frames,
                requested_frame_counts=frozen_world_frame_counts,
                timing_warmups=args.frozen_world_timing_warmups,
                timing_repeats=args.frozen_world_timing_repeats,
            )
        )
    else:
        frozen_world_report = None
        frozen_world_sweep = None

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
            "final_world_checkpoint": final_world_checkpoint,
            "world_representation": args.uvt_world_representation,
            "alpha_mode": render_config.alpha_mode,
            "opacity_semantics": render_config.opacity_semantics,
            "spd4_min_spatial_scale": (
                args.uvt_spd4_min_spatial_scale
                if args.uvt_world_representation == "full_spd4"
                else None
            ),
            "spd4_init_precision_z": (
                (
                    args.uvt_init_precision_xy
                    if args.uvt_spd4_init_precision_z is None
                    else args.uvt_spd4_init_precision_z
                )
                if args.uvt_world_representation == "full_spd4"
                else None
            ),
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
            "init_depth": args.init_depth,
            "init_precision_xy": args.uvt_init_precision_xy,
            "init_lambda_t": args.uvt_init_lambda_t,
            "static_tube_fraction": args.uvt_static_tube_fraction,
            "static_init_lambda_t": args.uvt_static_init_lambda_t,
            "static_velocity_reg": args.uvt_static_velocity_reg,
            "init_opacity": args.uvt_init_opacity,
            "init_opacity_semantics": "initial_center_alpha",
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
            "frozen_world_replay_compiled": frozen_world_report,
            "frozen_world_replay_compiled_sweep": frozen_world_sweep,
        },
        "star_uvt_selected": selected_report,
        "free_dynamic_splats": splat_report,
    }
    write_json(out_dir / "comparison_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote STAR-UVT multicam heldout comparison to {out_dir}")


if __name__ == "__main__":
    main()
