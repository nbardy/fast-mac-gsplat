from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig, render_uvt_tubes, render_uvt_tubes_gated  # noqa: E402

try:
    from research_project.trainer_harness.data import load_video_target
    from research_project.trainer_harness.model import ScreenTimeTubeModel, _inv_softplus, render_model
    from research_project.trainer_harness.per_frame_baseline import (
        PerFrameGaussianModel,
        render_per_frame_gaussians,
        render_per_frame_gaussians_fast_mac,
    )
    from research_project.trainer_harness.tile_metal_autograd import (
        full_active_intervals,
        render_uvt_tubes_metal_interval_gated_backward,
        render_uvt_tubes_metal_tile_backward,
    )
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = Path(__file__).resolve().parents[1] / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from data import load_video_target
    from model import ScreenTimeTubeModel, _inv_softplus, render_model
    from per_frame_baseline import PerFrameGaussianModel, render_per_frame_gaussians, render_per_frame_gaussians_fast_mac
    from tile_metal_autograd import (
        full_active_intervals,
        render_uvt_tubes_metal_interval_gated_backward,
        render_uvt_tubes_metal_tile_backward,
    )


UVT_RENDER_BACKENDS = ("dense", "metal_tile", "metal_tile_interval_gated")
METAL_UVT_RENDER_BACKENDS = ("metal_tile", "metal_tile_interval_gated")


def render_per_frame_model(model: PerFrameGaussianModel, *, backend: str, fast_max_pairs: int) -> torch.Tensor:
    if backend == "dense":
        return render_per_frame_gaussians(model)
    if backend == "fast_mac":
        return render_per_frame_gaussians_fast_mac(model, max_fast_pairs=fast_max_pairs)
    raise ValueError("backend must be one of: dense, fast_mac")


def fit_per_frame(
    model: PerFrameGaussianModel,
    target: torch.Tensor,
    *,
    steps: int,
    lr: float,
    backend: str,
    fast_max_pairs: int,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image = render_per_frame_model(model, backend=backend, fast_max_pairs=fast_max_pairs)
        loss = torch.mean((image - target).square())
        losses.append(float(loss.detach().cpu()))
        if step == steps:
            break
        loss.backward()
        optimizer.step()
    return losses


def render_uvt_model(
    model: ScreenTimeTubeModel,
    *,
    backend: str,
    reduction_mode: str = "index_add",
    sample_emission_mode: str = "atomic_append",
) -> torch.Tensor:
    if backend == "dense":
        return render_model(model)
    if backend == "metal_tile":
        ma, q_uvt, depth0, depth_beta, opacity, color = model.tensors()
        if not torch.is_grad_enabled():
            return render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, model.config)
        return render_uvt_tubes_metal_tile_backward(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            color,
            model.config,
            reduction_mode=reduction_mode,
            sample_emission_mode=sample_emission_mode,
        )
    if backend == "metal_tile_interval_gated":
        ma, q_uvt, depth0, depth_beta, opacity, color = model.tensors()
        active_start, active_stop = full_active_intervals(int(ma.shape[0]), int(model.config.frames), ma.device)
        if not torch.is_grad_enabled():
            return render_uvt_tubes_gated(
                ma,
                q_uvt,
                depth0,
                depth_beta,
                opacity,
                color,
                active_start,
                active_stop,
                model.config,
            )
        return render_uvt_tubes_metal_interval_gated_backward(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            color,
            active_start,
            active_stop,
            model.config,
        )
    raise ValueError(f"backend must be one of: {', '.join(UVT_RENDER_BACKENDS)}")


def _uvt_inv_diag(q_uvt: torch.Tensor) -> torch.Tensor:
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


def uvt_tile_load_proxy(model: ScreenTimeTubeModel) -> torch.Tensor:
    _ma, q_uvt, _depth0, _depth_beta, opacity, _color = model.tensors()
    config = model.config
    opacity_safe = opacity.clamp_min(float(config.alpha_threshold) * 1.0001)
    tau = -2.0 * torch.log((float(config.alpha_threshold) / opacity_safe).clamp_min(1.0e-8))
    half_extent = torch.sqrt((tau.unsqueeze(-1) * _uvt_inv_diag(q_uvt)).clamp_min(0.0))
    span_x = 1.0 + 2.0 * half_extent[:, 0] / float(config.tile_x)
    span_y = 1.0 + 2.0 * half_extent[:, 1] / float(config.tile_y)
    span_t = 1.0 + 2.0 * half_extent[:, 2] / float(config.tile_t)
    return (span_x * span_y * span_t).mean()


def uvt_tile_load_regularization(
    model: ScreenTimeTubeModel,
    *,
    weight: float,
    target: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    proxy = uvt_tile_load_proxy(model)
    if target > 0.0:
        target_tensor = proxy.new_tensor(float(target))
        penalty = torch.relu(proxy - target_tensor).div(target_tensor).square()
    else:
        penalty = proxy
    return proxy, proxy.new_tensor(float(weight)) * penalty


def fit_uvt(
    model: ScreenTimeTubeModel,
    target: torch.Tensor,
    *,
    steps: int,
    lr: float,
    final_lr: float | None,
    final_lr_start_step: int | None,
    backend: str,
    reduction_mode: str,
    sample_emission_mode: str,
    tile_load_reg_weight: float,
    tile_load_target: float,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps + 1):
        if final_lr is not None and final_lr_start_step is not None and step == final_lr_start_step:
            for group in optimizer.param_groups:
                group["lr"] = final_lr
        optimizer.zero_grad(set_to_none=True)
        image = render_uvt_model(
            model,
            backend=backend,
            reduction_mode=reduction_mode,
            sample_emission_mode=sample_emission_mode,
        )
        loss = torch.mean((image - target).square())
        if tile_load_reg_weight > 0.0:
            _tile_proxy, tile_reg_loss = uvt_tile_load_regularization(
                model,
                weight=tile_load_reg_weight,
                target=tile_load_target,
            )
            loss = loss + tile_reg_loss
        losses.append(float(loss.detach().cpu()))
        if step == steps:
            break
        loss.backward()
        optimizer.step()
    return losses


def fit_uvt_appearance(
    model: ScreenTimeTubeModel,
    target: torch.Tensor,
    *,
    steps: int,
    lr: float,
    backend: str,
    reduction_mode: str,
    sample_emission_mode: str,
) -> list[float]:
    optimizer = torch.optim.Adam([model.raw_color, model.raw_opacity], lr=lr)
    losses: list[float] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image = render_uvt_model(
            model,
            backend=backend,
            reduction_mode=reduction_mode,
            sample_emission_mode=sample_emission_mode,
        )
        loss = torch.mean((image - target).square())
        losses.append(float(loss.detach().cpu()))
        if step == steps:
            break
        loss.backward()
        optimizer.step()
    return losses


def mse_to_psnr(mse: float) -> float:
    return -10.0 * torch.log10(torch.tensor(max(mse, 1.0e-12), dtype=torch.float32)).item()


def _gaussian_kernel2d(window_size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - float(window_size - 1) * 0.5
    kernel1d = torch.exp(-(coords.square()) / (2.0 * float(sigma) * float(sigma)))
    kernel1d = kernel1d / kernel1d.sum().clamp_min(1.0e-12)
    kernel2d = kernel1d[:, None] * kernel1d[None, :]
    return kernel2d


def _global_ssim_per_frame(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    reduce_dims = (2, 3)
    mu_x = prediction.mean(dim=reduce_dims, keepdim=True)
    mu_y = target.mean(dim=reduce_dims, keepdim=True)
    sigma_x = (prediction - mu_x).square().mean(dim=reduce_dims, keepdim=True)
    sigma_y = (target - mu_y).square().mean(dim=reduce_dims, keepdim=True)
    sigma_xy = ((prediction - mu_x) * (target - mu_y)).mean(dim=reduce_dims, keepdim=True)
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1.0e-12)).mean(dim=(1, 2, 3))


def ssim_per_frame(prediction: torch.Tensor, target: torch.Tensor, *, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"SSIM tensors must have matching shape, got {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.ndim != 4 or prediction.shape[-1] != 3:
        raise ValueError(f"SSIM expects [F,H,W,3], got {tuple(prediction.shape)}")

    pred_nchw = prediction.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()
    target_nchw = target.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()
    frame_count, channels, height, width = pred_nchw.shape
    if frame_count < 1:
        raise ValueError("SSIM needs at least one frame")

    size = min(int(window_size), int(height), int(width))
    if size % 2 == 0:
        size -= 1
    if size < 3:
        return _global_ssim_per_frame(pred_nchw, target_nchw)

    kernel2d = _gaussian_kernel2d(size, sigma, device=pred_nchw.device, dtype=pred_nchw.dtype)
    kernel = kernel2d.expand(channels, 1, size, size).contiguous()
    padding = size // 2
    c1 = 0.01**2
    c2 = 0.03**2

    pred_padded = F.pad(pred_nchw, (padding, padding, padding, padding), mode="replicate")
    target_padded = F.pad(target_nchw, (padding, padding, padding, padding), mode="replicate")
    mu_x = F.conv2d(pred_padded, kernel, groups=channels)
    mu_y = F.conv2d(target_padded, kernel, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(pred_padded * pred_padded, kernel, groups=channels) - mu_x_sq
    sigma_y = F.conv2d(target_padded * target_padded, kernel, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(pred_padded * target_padded, kernel, groups=channels) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator.clamp_min(1.0e-12)
    return ssim_map.mean(dim=(1, 2, 3))


def summarize_ssim(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    values = ssim_per_frame(prediction, target)
    per_frame = [float(value) for value in values.detach().cpu()]
    mean = float(values.mean().detach().cpu())
    min_value = float(values.min().detach().cpu())
    max_value = float(values.max().detach().cpu())
    return {
        "mean": mean,
        "min": min_value,
        "max": max_value,
        "dssim_mean": (1.0 - mean) * 0.5,
        "per_frame": per_frame,
    }


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def apply_uvt_tile_env(config: UVTRenderConfig) -> None:
    os.environ["STAR_UVT_TILE_X"] = str(config.tile_x)
    os.environ["STAR_UVT_TILE_Y"] = str(config.tile_y)
    os.environ["STAR_UVT_TILE_T"] = str(config.tile_t)
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(config.tile_capacity)


def make_uvt_model(
    target: torch.Tensor,
    config: UVTRenderConfig,
    *,
    tube_count: int,
    seed: int,
    device: torch.device,
    init_mode: str,
    spatial_precision: float,
    temporal_precision: float,
    opacity: float,
    sample_mode: str,
    velocity_init: str,
    velocity_search_radius: int,
    velocity_patch_radius: int,
    velocity_min_improvement_ratio: float,
) -> ScreenTimeTubeModel:
    if init_mode == "random":
        return ScreenTimeTubeModel(tube_count, config, seed=seed, device=device)
    if init_mode == "video_samples":
        return ScreenTimeTubeModel.from_video_samples(
            target,
            config,
            tube_count=tube_count,
            seed=seed,
            spatial_precision=spatial_precision,
            temporal_precision=temporal_precision,
            opacity=opacity,
            sample_mode=sample_mode,
            velocity_init=velocity_init,
            velocity_search_radius=velocity_search_radius,
            velocity_patch_radius=velocity_patch_radius,
            velocity_min_improvement_ratio=velocity_min_improvement_ratio,
        )
    raise ValueError("init_mode must be one of: random, video_samples")


def resize_uvt_model(model: ScreenTimeTubeModel, config: UVTRenderConfig) -> ScreenTimeTubeModel:
    """Promote a learned screen-space UVT model to a new spatial resolution."""

    if int(config.frames) != int(model.config.frames):
        raise ValueError("multi-resolution promotion requires the same frame count")
    scale_x = float(config.width) / float(model.config.width)
    scale_y = float(config.height) / float(model.config.height)
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("invalid spatial scale for UVT promotion")

    child = ScreenTimeTubeModel(
        int(model.tube_count),
        config,
        seed=0,
        device=model.center_uv.device,
        min_precision=model.min_precision,
    )
    scale_uv = torch.tensor([scale_x, scale_y], dtype=model.center_uv.dtype, device=model.center_uv.device)
    with torch.no_grad():
        precision = F.softplus(model.raw_precision) + model.min_precision
        promoted_precision = precision.clone()
        promoted_precision[:, 0].div_(scale_x * scale_x)
        promoted_precision[:, 1].div_(scale_y * scale_y)
        promoted_precision.clamp_(min=model.min_precision * 2.0)

        child.center_uv.copy_(model.center_uv * scale_uv)
        child.center_t.copy_(model.center_t)
        child.velocity_uv.copy_(model.velocity_uv * scale_uv)
        child.raw_precision.copy_(_inv_softplus(promoted_precision - child.min_precision))
        child.raw_opacity.copy_(model.raw_opacity)
        child.raw_color.copy_(model.raw_color)
        child.depth0.copy_(model.depth0)
    return child


def _as_uint8(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().cpu().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def write_contact_sheet(
    path: Path,
    target: torch.Tensor,
    uvt: torch.Tensor | None,
    per_frame: torch.Tensor | None,
    *,
    max_frames: int,
    mode: str,
) -> None:
    source_frame_count = int(target.shape[0])
    frame_count = min(source_frame_count, max(1, int(max_frames)))
    if mode == "first":
        frame_indices = torch.arange(frame_count, dtype=torch.long)
    elif mode == "linspace":
        frame_indices = torch.linspace(0, source_frame_count - 1, frame_count).round().to(dtype=torch.long)
    else:
        raise ValueError("contact sheet mode must be one of: first, linspace")
    rows = [target.index_select(0, frame_indices.to(target.device))]
    if uvt is not None:
        rows.append(uvt.index_select(0, frame_indices.to(uvt.device)))
    if per_frame is not None:
        rows.append(per_frame.index_select(0, frame_indices.to(per_frame.device)))
    height = int(target.shape[1])
    width = int(target.shape[2])
    gutter = 2
    sheet_width = frame_count * width + (frame_count - 1) * gutter
    sheet_height = len(rows) * height + (len(rows) - 1) * gutter
    sheet = Image.new("RGB", (sheet_width, sheet_height), (0, 0, 0))
    for row_idx, frames in enumerate(rows):
        y = row_idx * (height + gutter)
        for frame_idx in range(frame_count):
            x = frame_idx * (width + gutter)
            sheet.paste(_as_uint8(frames[frame_idx]), (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def write_side_by_side_video(
    path: Path,
    target: torch.Tensor,
    uvt: torch.Tensor | None,
    per_frame: torch.Tensor | None,
    *,
    fps: float,
) -> None:
    rows = [target]
    if uvt is not None:
        rows.append(uvt)
    if per_frame is not None:
        rows.append(per_frame)
    if len(rows) < 2:
        raise ValueError("side-by-side video needs at least one prediction row")
    path.parent.mkdir(parents=True, exist_ok=True)
    video = torch.cat(rows, dim=2).detach().cpu().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).numpy()
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=max(1.0, float(fps)), codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in video:
            writer.append_data(frame)


def timed_render_uvt(model: ScreenTimeTubeModel, *, backend: str) -> tuple[torch.Tensor, float]:
    sync_device(model.center_uv.device)
    started = time.perf_counter()
    image = render_uvt_model(model, backend=backend)
    sync_device(model.center_uv.device)
    return image, (time.perf_counter() - started) * 1000.0


def timed_render_per_frame(
    model: PerFrameGaussianModel,
    *,
    backend: str,
    fast_max_pairs: int,
) -> tuple[torch.Tensor, float]:
    sync_device(model.center_uv.device)
    started = time.perf_counter()
    image = render_per_frame_model(model, backend=backend, fast_max_pairs=fast_max_pairs)
    sync_device(model.center_uv.device)
    return image, (time.perf_counter() - started) * 1000.0


def summarize_ms(samples: list[float]) -> dict[str, object]:
    if not samples:
        return {"samples": [], "min": None, "median": None, "max": None}
    return {
        "samples": samples,
        "min": min(samples),
        "median": statistics.median(samples),
        "max": max(samples),
    }


def validate_uvt_backend_modes(
    *,
    uvt_render_backend: str,
    uvt_reduction_mode: str,
    uvt_sample_emission_mode: str,
    device: torch.device,
) -> None:
    if uvt_render_backend not in UVT_RENDER_BACKENDS:
        raise ValueError(f"uvt_render_backend must be one of: {', '.join(UVT_RENDER_BACKENDS)}")
    if uvt_render_backend in METAL_UVT_RENDER_BACKENDS and device.type != "mps":
        raise ValueError(f"--uvt-render-backend={uvt_render_backend} requires --device=mps")
    if uvt_render_backend == "metal_tile_interval_gated":
        if uvt_reduction_mode != "index_add" or uvt_sample_emission_mode != "direct_atomic":
            raise ValueError(
                "--uvt-render-backend=metal_tile_interval_gated uses native direct_atomic_gated "
                "and requires --uvt-reduction-mode=index_add plus --uvt-sample-emission-mode=direct_atomic"
            )
        return
    if uvt_render_backend != "metal_tile" and (
        uvt_reduction_mode != "index_add" or uvt_sample_emission_mode != "atomic_append"
    ):
        raise ValueError("custom UVT reduction/sample emission modes require --uvt-render-backend=metal_tile")
    if uvt_reduction_mode in (
        "key_sort_scan_metal",
        "key_sort_compensated_scan_metal",
        "key_sort_segmented_metal",
    ) and uvt_sample_emission_mode not in (
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
    if uvt_sample_emission_mode in (
        "direct_atomic",
        "direct_fixedpoint",
        "direct_split_fixedpoint",
        "direct_serial",
        "tile_pair_atomic",
        "tile_pair_fixedpoint",
        "tile_pair_reduced",
        "tile_pair_reduced_parallel",
        "tile_pair_suffix_reduced",
    ) and uvt_reduction_mode != "index_add":
        raise ValueError(f"{uvt_sample_emission_mode} bypasses the reducer and requires --uvt-reduction-mode index_add")


def run_video_fit_comparison(
    *,
    video_path: Path,
    start_seconds: float | None,
    fps: float | None,
    duration_seconds: float | None,
    image_crop_mode: str,
    tube_count: int,
    per_frame_splats: int,
    target_size: int,
    max_frames: int,
    steps: int,
    lr: float,
    per_frame_lr: float,
    per_frame_init_mode: str,
    per_frame_render_backend: str,
    per_frame_fast_max_pairs: int,
    per_frame_spatial_precision: float,
    per_frame_opacity: float,
    per_frame_sample_mode: str,
    device: str,
    seed: int,
    uvt_init_mode: str,
    uvt_spatial_precision: float,
    uvt_temporal_precision: float,
    uvt_opacity: float,
    uvt_sample_mode: str,
    uvt_velocity_init: str,
    uvt_velocity_search_radius: int,
    uvt_velocity_patch_radius: int,
    uvt_velocity_min_improvement_ratio: float,
    uvt_final_lr: float | None,
    uvt_final_lr_start_step: int | None,
    uvt_coarse_target_size: int | None,
    uvt_coarse_steps: int,
    uvt_coarse_lr: float | None,
    uvt_appearance_refine_steps: int,
    uvt_appearance_lr: float,
    uvt_temporal_split_step: int | None,
    uvt_temporal_split_offset: float,
    uvt_temporal_split_precision_scale: float,
    uvt_temporal_split_opacity_scale: float,
    uvt_temporal_split_depth_offset: float,
    uvt_temporal_split_lr: float | None,
    uvt_render_backend: str,
    uvt_reduction_mode: str,
    uvt_sample_emission_mode: str,
    uvt_tile_t: int,
    uvt_tile_capacity: int,
    uvt_tile_load_reg_weight: float,
    uvt_tile_load_target: float,
    render_benchmark_repeats: int,
    skip_uvt: bool,
    skip_per_frame: bool,
    contact_sheet: Path | None,
    contact_sheet_frames: int,
    contact_sheet_mode: str,
    side_by_side_video: Path | None,
    side_by_side_fps: float | None,
) -> dict[str, object]:
    dev = torch.device(device)
    validate_uvt_backend_modes(
        uvt_render_backend=uvt_render_backend,
        uvt_reduction_mode=uvt_reduction_mode,
        uvt_sample_emission_mode=uvt_sample_emission_mode,
        device=dev,
    )
    if per_frame_render_backend == "fast_mac" and dev.type != "mps":
        raise ValueError("--per-frame-render-backend=fast_mac requires --device=mps")
    if uvt_temporal_split_step is not None:
        if uvt_temporal_split_step <= 0 or uvt_temporal_split_step >= steps:
            raise ValueError("--uvt-temporal-split-step must be greater than 0 and less than --steps")
        if uvt_final_lr is not None or uvt_final_lr_start_step is not None:
            raise ValueError("temporal split is intentionally not mixed with staged LR in this benchmark")
        if uvt_coarse_steps > 0 or uvt_coarse_target_size is not None:
            raise ValueError("temporal split is intentionally not mixed with multi-resolution promotion")
    if uvt_coarse_steps < 0:
        raise ValueError("--uvt-coarse-steps must be non-negative")
    if uvt_coarse_steps > 0 and uvt_coarse_target_size is None:
        raise ValueError("--uvt-coarse-steps requires --uvt-coarse-target-size")
    if uvt_coarse_target_size is not None:
        if uvt_coarse_target_size <= 0:
            raise ValueError("--uvt-coarse-target-size must be positive")
        if uvt_coarse_target_size == target_size:
            raise ValueError("--uvt-coarse-target-size should differ from --target-size")
    target = load_video_target(
        video_path,
        target_size=target_size,
        max_frames=max_frames,
        device=dev,
        start_seconds=start_seconds,
        fps=fps,
        duration_seconds=duration_seconds,
        image_crop_mode=image_crop_mode,
    )
    config = UVTRenderConfig(
        height=int(target.shape[1]),
        width=int(target.shape[2]),
        frames=int(target.shape[0]),
        tile_t=uvt_tile_t,
        tile_capacity=uvt_tile_capacity,
    )
    apply_uvt_tile_env(config)
    uvt_model = None
    coarse_target = None
    coarse_config = None
    if not skip_uvt:
        model_target = target
        model_config = config
        if uvt_coarse_steps > 0:
            if uvt_coarse_target_size is None:
                raise AssertionError("coarse target size was validated above")
            coarse_target = load_video_target(
                video_path,
                target_size=uvt_coarse_target_size,
                max_frames=max_frames,
                device=dev,
                start_seconds=start_seconds,
                fps=fps,
                duration_seconds=duration_seconds,
                image_crop_mode=image_crop_mode,
            )
            coarse_config = UVTRenderConfig(
                height=int(coarse_target.shape[1]),
                width=int(coarse_target.shape[2]),
                frames=int(coarse_target.shape[0]),
                tile_t=uvt_tile_t,
                tile_capacity=uvt_tile_capacity,
            )
            apply_uvt_tile_env(coarse_config)
            model_target = coarse_target
            model_config = coarse_config
        uvt_model = make_uvt_model(
            model_target,
            model_config,
            tube_count=tube_count,
            seed=seed,
            device=dev,
            init_mode=uvt_init_mode,
            spatial_precision=uvt_spatial_precision,
            temporal_precision=uvt_temporal_precision,
            opacity=uvt_opacity,
            sample_mode=uvt_sample_mode,
            velocity_init=uvt_velocity_init,
            velocity_search_radius=uvt_velocity_search_radius,
            velocity_patch_radius=uvt_velocity_patch_radius,
            velocity_min_improvement_ratio=uvt_velocity_min_improvement_ratio,
        )
    per_frame_model = None
    if not skip_per_frame:
        if per_frame_init_mode == "random":
            per_frame_model = PerFrameGaussianModel(config.frames, per_frame_splats, config, seed=seed, device=dev)
        elif per_frame_init_mode == "video_samples":
            per_frame_model = PerFrameGaussianModel.from_video_samples(
                target,
                config,
                splats_per_frame=per_frame_splats,
                seed=seed,
                spatial_precision=per_frame_spatial_precision,
                opacity=per_frame_opacity,
                sample_mode=per_frame_sample_mode,
            )
        else:
            raise ValueError("per_frame_init_mode must be one of: random, video_samples")

    uvt_started = time.perf_counter()
    uvt_coarse_info = None
    uvt_split_info = None
    uvt_losses = None
    uvt_main_final_loss = None
    uvt_appearance_losses = None
    if uvt_model is not None:
        if uvt_coarse_steps > 0:
            if coarse_target is None or coarse_config is None:
                raise AssertionError("coarse target/config missing")
            coarse_started = time.perf_counter()
            coarse_losses = fit_uvt(
                uvt_model,
                coarse_target,
                steps=uvt_coarse_steps,
                lr=lr if uvt_coarse_lr is None else uvt_coarse_lr,
                final_lr=None,
                final_lr_start_step=None,
                backend=uvt_render_backend,
                reduction_mode=uvt_reduction_mode,
                sample_emission_mode=uvt_sample_emission_mode,
                tile_load_reg_weight=uvt_tile_load_reg_weight,
                tile_load_target=uvt_tile_load_target,
            )
            sync_device(dev)
            coarse_ms = (time.perf_counter() - coarse_started) * 1000.0
            uvt_model = resize_uvt_model(uvt_model, config)
            apply_uvt_tile_env(config)
            fine_losses = fit_uvt(
                uvt_model,
                target,
                steps=steps,
                lr=lr,
                final_lr=uvt_final_lr,
                final_lr_start_step=uvt_final_lr_start_step,
                backend=uvt_render_backend,
                reduction_mode=uvt_reduction_mode,
                sample_emission_mode=uvt_sample_emission_mode,
                tile_load_reg_weight=uvt_tile_load_reg_weight,
                tile_load_target=uvt_tile_load_target,
            )
            uvt_losses = coarse_losses + fine_losses
            uvt_coarse_info = {
                "target_size": int(uvt_coarse_target_size) if uvt_coarse_target_size is not None else None,
                "height": int(coarse_config.height),
                "width": int(coarse_config.width),
                "frames": int(coarse_config.frames),
                "steps": int(uvt_coarse_steps),
                "lr": lr if uvt_coarse_lr is None else uvt_coarse_lr,
                "wall_clock_ms": coarse_ms,
                "initial_loss": coarse_losses[0],
                "final_loss": coarse_losses[-1],
                "fine_initial_loss": fine_losses[0],
            }
        elif uvt_temporal_split_step is None:
            uvt_losses = fit_uvt(
                uvt_model,
                target,
                steps=steps,
                lr=lr,
                final_lr=uvt_final_lr,
                final_lr_start_step=uvt_final_lr_start_step,
                backend=uvt_render_backend,
                reduction_mode=uvt_reduction_mode,
                sample_emission_mode=uvt_sample_emission_mode,
                tile_load_reg_weight=uvt_tile_load_reg_weight,
                tile_load_target=uvt_tile_load_target,
            )
        else:
            pre_split_losses = fit_uvt(
                uvt_model,
                target,
                steps=uvt_temporal_split_step,
                lr=lr,
                final_lr=None,
                final_lr_start_step=None,
                backend=uvt_render_backend,
                reduction_mode=uvt_reduction_mode,
                sample_emission_mode=uvt_sample_emission_mode,
                tile_load_reg_weight=uvt_tile_load_reg_weight,
                tile_load_target=uvt_tile_load_target,
            )
            pre_split_tube_count = int(uvt_model.tube_count)
            uvt_model = uvt_model.temporal_split(
                offset_frames=uvt_temporal_split_offset,
                temporal_precision_scale=uvt_temporal_split_precision_scale,
                opacity_scale=uvt_temporal_split_opacity_scale,
                depth_offset=uvt_temporal_split_depth_offset,
            )
            post_split_losses = fit_uvt(
                uvt_model,
                target,
                steps=steps - uvt_temporal_split_step,
                lr=lr if uvt_temporal_split_lr is None else uvt_temporal_split_lr,
                final_lr=None,
                final_lr_start_step=None,
                backend=uvt_render_backend,
                reduction_mode=uvt_reduction_mode,
                sample_emission_mode=uvt_sample_emission_mode,
                tile_load_reg_weight=uvt_tile_load_reg_weight,
                tile_load_target=uvt_tile_load_target,
            )
            uvt_losses = pre_split_losses + post_split_losses
            uvt_split_info = {
                "step": uvt_temporal_split_step,
                "offset_frames": uvt_temporal_split_offset,
                "temporal_precision_scale": uvt_temporal_split_precision_scale,
                "opacity_scale": uvt_temporal_split_opacity_scale,
                "depth_offset": uvt_temporal_split_depth_offset,
                "lr": lr if uvt_temporal_split_lr is None else uvt_temporal_split_lr,
                "pre_split_tube_count": pre_split_tube_count,
                "post_split_tube_count": int(uvt_model.tube_count),
                "pre_split_loss": pre_split_losses[-1],
                "post_split_initial_loss": post_split_losses[0],
            }
        uvt_main_final_loss = uvt_losses[-1]
        if uvt_appearance_refine_steps > 0:
            uvt_appearance_losses = fit_uvt_appearance(
                uvt_model,
                target,
                steps=uvt_appearance_refine_steps,
                lr=uvt_appearance_lr,
                backend=uvt_render_backend,
                reduction_mode=uvt_reduction_mode,
                sample_emission_mode=uvt_sample_emission_mode,
            )
            uvt_losses.extend(uvt_appearance_losses[1:])
    uvt_ms = (time.perf_counter() - uvt_started) * 1000.0

    per_frame_losses = None
    per_frame_ms = None
    if per_frame_model is not None:
        per_frame_started = time.perf_counter()
        per_frame_losses = fit_per_frame(
            per_frame_model,
            target,
            steps=steps,
            lr=per_frame_lr,
            backend=per_frame_render_backend,
            fast_max_pairs=per_frame_fast_max_pairs,
        )
        per_frame_ms = (time.perf_counter() - per_frame_started) * 1000.0

    uvt_image = None
    uvt_render_ms = None
    uvt_render_samples = None
    if uvt_model is not None:
        with torch.no_grad():
            uvt_image, uvt_render_ms = timed_render_uvt(uvt_model, backend=uvt_render_backend)
            uvt_render_samples = [uvt_render_ms]
            for _ in range(max(0, render_benchmark_repeats - 1)):
                _, render_ms = timed_render_uvt(uvt_model, backend=uvt_render_backend)
                uvt_render_samples.append(render_ms)

    per_frame_image = None
    per_frame_render_ms = None
    per_frame_render_samples = None
    if per_frame_model is not None:
        with torch.no_grad():
            per_frame_image, per_frame_render_ms = timed_render_per_frame(
                per_frame_model,
                backend=per_frame_render_backend,
                fast_max_pairs=per_frame_fast_max_pairs,
            )
            per_frame_render_samples = [per_frame_render_ms]
            for _ in range(max(0, render_benchmark_repeats - 1)):
                _, render_ms = timed_render_per_frame(
                    per_frame_model,
                    backend=per_frame_render_backend,
                    fast_max_pairs=per_frame_fast_max_pairs,
                )
                per_frame_render_samples.append(render_ms)

    with torch.no_grad():
        uvt_l1 = None if uvt_image is None else torch.mean((uvt_image - target).abs()).item()
        uvt_mse = None if uvt_image is None else torch.mean((uvt_image - target).square()).item()
        uvt_ssim = None if uvt_image is None else summarize_ssim(uvt_image, target)
        uvt_tile_load_final_proxy = None if uvt_model is None else float(uvt_tile_load_proxy(uvt_model).detach().cpu())
        per_frame_l1 = None if per_frame_image is None else torch.mean((per_frame_image - target).abs()).item()
        per_frame_mse = None if per_frame_image is None else torch.mean((per_frame_image - target).square()).item()
        per_frame_ssim = None if per_frame_image is None else summarize_ssim(per_frame_image, target)

    if contact_sheet is not None:
        write_contact_sheet(
            contact_sheet,
            target,
            uvt_image,
            per_frame_image,
            max_frames=contact_sheet_frames,
            mode=contact_sheet_mode,
        )
    if side_by_side_video is not None:
        write_side_by_side_video(
            side_by_side_video,
            target,
            uvt_image,
            per_frame_image,
            fps=side_by_side_fps if side_by_side_fps is not None else (fps if fps is not None else 30.0),
        )

    row = {
        "video_path": str(video_path),
        "start_seconds": start_seconds,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "image_crop_mode": image_crop_mode,
        "frames": config.frames,
        "height": config.height,
        "width": config.width,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "device": str(dev),
        "contact_sheet": None if contact_sheet is None else str(contact_sheet),
        "contact_sheet_frames": contact_sheet_frames,
        "contact_sheet_mode": contact_sheet_mode,
        "side_by_side_video": None if side_by_side_video is None else str(side_by_side_video),
        "side_by_side_fps": side_by_side_fps if side_by_side_fps is not None else (fps if fps is not None else 30.0),
        "uvt": None if uvt_model is None or uvt_losses is None or uvt_l1 is None or uvt_mse is None else {
            "initial_tube_count": tube_count,
            "tube_count": int(uvt_model.tube_count),
            "init_mode": uvt_init_mode,
            "init_spatial_precision": uvt_spatial_precision,
            "init_temporal_precision": uvt_temporal_precision,
            "init_opacity": uvt_opacity,
            "sample_mode": uvt_sample_mode,
            "velocity_init": uvt_velocity_init,
            "velocity_search_radius": uvt_velocity_search_radius,
            "velocity_patch_radius": uvt_velocity_patch_radius,
            "velocity_min_improvement_ratio": uvt_velocity_min_improvement_ratio,
            "render_backend": uvt_render_backend,
            "reduction_mode": uvt_reduction_mode,
            "sample_emission_mode": uvt_sample_emission_mode,
            "tile_t": uvt_tile_t,
            "tile_capacity": uvt_tile_capacity,
            "tile_load_reg_weight": uvt_tile_load_reg_weight,
            "tile_load_target": uvt_tile_load_target,
            "tile_load_final_proxy": uvt_tile_load_final_proxy,
            "render_benchmark_repeats": render_benchmark_repeats,
            "render_benchmark_ms": summarize_ms([] if uvt_render_samples is None else uvt_render_samples),
            "final_lr": uvt_final_lr,
            "final_lr_start_step": uvt_final_lr_start_step,
            "coarse_stage": uvt_coarse_info,
            "appearance_refine_steps": uvt_appearance_refine_steps,
            "appearance_lr": uvt_appearance_lr,
            "temporal_split": uvt_split_info,
            "main_final_loss": uvt_main_final_loss,
            "appearance_initial_loss": None if uvt_appearance_losses is None else uvt_appearance_losses[0],
            "appearance_final_loss": None if uvt_appearance_losses is None else uvt_appearance_losses[-1],
            "parameter_count": parameter_count(uvt_model),
            "initial_loss": uvt_losses[0],
            "final_loss": uvt_losses[-1],
            "loss_ratio": uvt_losses[-1] / max(uvt_losses[0], 1.0e-12),
            "final_l1": uvt_l1,
            "final_mse": uvt_mse,
            "final_psnr": mse_to_psnr(uvt_mse),
            "final_ssim_mean": None if uvt_ssim is None else uvt_ssim["mean"],
            "final_ssim_min": None if uvt_ssim is None else uvt_ssim["min"],
            "final_ssim_max": None if uvt_ssim is None else uvt_ssim["max"],
            "final_dssim_mean": None if uvt_ssim is None else uvt_ssim["dssim_mean"],
            "final_ssim_per_frame": None if uvt_ssim is None else uvt_ssim["per_frame"],
            "render_ms": uvt_render_ms,
            "wall_clock_ms": uvt_ms,
        },
        "per_frame": None,
    }
    if per_frame_model is not None and per_frame_losses is not None and per_frame_ms is not None:
        if per_frame_l1 is None or per_frame_mse is None or per_frame_render_ms is None:
            raise AssertionError("per-frame metrics missing despite per-frame model being trained")
        row["per_frame"] = {
            "splats_per_frame": per_frame_splats,
            "total_splats": per_frame_splats * config.frames,
            "init_mode": per_frame_init_mode,
            "render_backend": per_frame_render_backend,
            "fast_max_pairs": per_frame_fast_max_pairs if per_frame_render_backend == "fast_mac" else None,
            "init_spatial_precision": per_frame_spatial_precision,
            "init_opacity": per_frame_opacity,
            "sample_mode": per_frame_sample_mode,
            "lr": per_frame_lr,
            "parameter_count": parameter_count(per_frame_model),
            "render_benchmark_repeats": render_benchmark_repeats,
            "render_benchmark_ms": summarize_ms([] if per_frame_render_samples is None else per_frame_render_samples),
            "initial_loss": per_frame_losses[0],
            "final_loss": per_frame_losses[-1],
            "loss_ratio": per_frame_losses[-1] / max(per_frame_losses[0], 1.0e-12),
            "final_l1": per_frame_l1,
            "final_mse": per_frame_mse,
            "final_psnr": mse_to_psnr(per_frame_mse),
            "final_ssim_mean": None if per_frame_ssim is None else per_frame_ssim["mean"],
            "final_ssim_min": None if per_frame_ssim is None else per_frame_ssim["min"],
            "final_ssim_max": None if per_frame_ssim is None else per_frame_ssim["max"],
            "final_dssim_mean": None if per_frame_ssim is None else per_frame_ssim["dssim_mean"],
            "final_ssim_per_frame": None if per_frame_ssim is None else per_frame_ssim["per_frame"],
            "render_ms": per_frame_render_ms,
            "wall_clock_ms": per_frame_ms,
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--start-seconds", type=float)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--image-crop-mode", choices=("resize", "none", "center_square", "center_crop", "center"), default="resize")
    parser.add_argument("--tube-count", type=int, default=8)
    parser.add_argument("--per-frame-splats", type=int, default=8)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--per-frame-lr", type=float)
    parser.add_argument("--per-frame-init-mode", choices=("random", "video_samples"), default="random")
    parser.add_argument("--per-frame-render-backend", choices=("dense", "fast_mac"), default="dense")
    parser.add_argument("--per-frame-fast-max-pairs", type=int, default=2048)
    parser.add_argument("--per-frame-spatial-precision", type=float, default=0.25)
    parser.add_argument("--per-frame-opacity", type=float, default=0.35)
    parser.add_argument("--per-frame-sample-mode", choices=("random", "stratified"), default="random")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--uvt-init-mode", choices=("random", "video_samples"), default="random")
    parser.add_argument("--uvt-spatial-precision", type=float, default=0.25)
    parser.add_argument("--uvt-temporal-precision", type=float, default=0.25)
    parser.add_argument("--uvt-opacity", type=float, default=0.35)
    parser.add_argument("--uvt-sample-mode", choices=("random", "stratified", "temporal_quarters"), default="random")
    parser.add_argument("--uvt-velocity-init", choices=("zero", "block_match", "block_match_gated"), default="zero")
    parser.add_argument("--uvt-velocity-search-radius", type=int, default=4)
    parser.add_argument("--uvt-velocity-patch-radius", type=int, default=1)
    parser.add_argument("--uvt-velocity-min-improvement-ratio", type=float, default=0.9)
    parser.add_argument("--uvt-final-lr", type=float)
    parser.add_argument("--uvt-final-lr-start-step", type=int)
    parser.add_argument("--uvt-coarse-target-size", type=int)
    parser.add_argument("--uvt-coarse-steps", type=int, default=0)
    parser.add_argument("--uvt-coarse-lr", type=float)
    parser.add_argument("--uvt-appearance-refine-steps", type=int, default=0)
    parser.add_argument("--uvt-appearance-lr", type=float, default=0.04)
    parser.add_argument("--uvt-temporal-split-step", type=int)
    parser.add_argument("--uvt-temporal-split-offset", type=float, default=0.5)
    parser.add_argument("--uvt-temporal-split-precision-scale", type=float, default=2.0)
    parser.add_argument("--uvt-temporal-split-opacity-scale", type=float, default=1.0)
    parser.add_argument("--uvt-temporal-split-depth-offset", type=float, default=1.0e-4)
    parser.add_argument("--uvt-temporal-split-lr", type=float)
    parser.add_argument("--uvt-render-backend", choices=UVT_RENDER_BACKENDS, default="dense")
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
    parser.add_argument("--uvt-tile-t", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--uvt-tile-capacity", type=int, choices=(32, 64, 128, 256), default=128)
    parser.add_argument("--uvt-tile-load-reg", type=float, default=0.0)
    parser.add_argument("--uvt-tile-load-target", type=float, default=0.0)
    parser.add_argument("--render-benchmark-repeats", type=int, default=1)
    parser.add_argument("--skip-uvt", action="store_true")
    parser.add_argument("--skip-per-frame", action="store_true")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--contact-sheet-frames", type=int, default=4)
    parser.add_argument("--contact-sheet-mode", choices=("first", "linspace"), default="first")
    parser.add_argument("--side-by-side-video", type=Path)
    parser.add_argument("--side-by-side-fps", type=float)
    args = parser.parse_args()

    row = run_video_fit_comparison(
        video_path=args.video_path,
        start_seconds=args.start_seconds,
        fps=args.fps,
        duration_seconds=args.duration_seconds,
        image_crop_mode=args.image_crop_mode,
        tube_count=args.tube_count,
        per_frame_splats=args.per_frame_splats,
        target_size=args.target_size,
        max_frames=args.max_frames,
        steps=args.steps,
        lr=args.lr,
        per_frame_lr=args.lr if args.per_frame_lr is None else args.per_frame_lr,
        per_frame_init_mode=args.per_frame_init_mode,
        per_frame_render_backend=args.per_frame_render_backend,
        per_frame_fast_max_pairs=args.per_frame_fast_max_pairs,
        per_frame_spatial_precision=args.per_frame_spatial_precision,
        per_frame_opacity=args.per_frame_opacity,
        per_frame_sample_mode=args.per_frame_sample_mode,
        device=args.device,
        seed=args.seed,
        uvt_init_mode=args.uvt_init_mode,
        uvt_spatial_precision=args.uvt_spatial_precision,
        uvt_temporal_precision=args.uvt_temporal_precision,
        uvt_opacity=args.uvt_opacity,
        uvt_sample_mode=args.uvt_sample_mode,
        uvt_velocity_init=args.uvt_velocity_init,
        uvt_velocity_search_radius=args.uvt_velocity_search_radius,
        uvt_velocity_patch_radius=args.uvt_velocity_patch_radius,
        uvt_velocity_min_improvement_ratio=args.uvt_velocity_min_improvement_ratio,
        uvt_final_lr=args.uvt_final_lr,
        uvt_final_lr_start_step=args.uvt_final_lr_start_step,
        uvt_coarse_target_size=args.uvt_coarse_target_size,
        uvt_coarse_steps=args.uvt_coarse_steps,
        uvt_coarse_lr=args.uvt_coarse_lr,
        uvt_appearance_refine_steps=args.uvt_appearance_refine_steps,
        uvt_appearance_lr=args.uvt_appearance_lr,
        uvt_temporal_split_step=args.uvt_temporal_split_step,
        uvt_temporal_split_offset=args.uvt_temporal_split_offset,
        uvt_temporal_split_precision_scale=args.uvt_temporal_split_precision_scale,
        uvt_temporal_split_opacity_scale=args.uvt_temporal_split_opacity_scale,
        uvt_temporal_split_depth_offset=args.uvt_temporal_split_depth_offset,
        uvt_temporal_split_lr=args.uvt_temporal_split_lr,
        uvt_render_backend=args.uvt_render_backend,
        uvt_reduction_mode=args.uvt_reduction_mode,
        uvt_sample_emission_mode=args.uvt_sample_emission_mode,
        uvt_tile_t=args.uvt_tile_t,
        uvt_tile_capacity=args.uvt_tile_capacity,
        uvt_tile_load_reg_weight=args.uvt_tile_load_reg,
        uvt_tile_load_target=args.uvt_tile_load_target,
        render_benchmark_repeats=args.render_benchmark_repeats,
        skip_uvt=args.skip_uvt,
        skip_per_frame=args.skip_per_frame,
        contact_sheet=args.contact_sheet,
        contact_sheet_frames=args.contact_sheet_frames,
        contact_sheet_mode=args.contact_sheet_mode,
        side_by_side_video=args.side_by_side_video,
        side_by_side_fps=args.side_by_side_fps,
    )
    if row["uvt"] is not None and float(row["uvt"]["final_loss"]) >= float(row["uvt"]["initial_loss"]):
        raise AssertionError(f"UVT loss did not decrease: {row['uvt']}")
    if row["per_frame"] is not None and float(row["per_frame"]["final_loss"]) >= float(row["per_frame"]["initial_loss"]):
        raise AssertionError(f"per-frame loss did not decrease: {row['per_frame']}")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
