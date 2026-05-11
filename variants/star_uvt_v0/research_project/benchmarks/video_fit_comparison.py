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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig  # noqa: E402

try:
    from research_project.trainer_harness.data import load_video_target
    from research_project.trainer_harness.model import ScreenTimeTubeModel, render_model
    from research_project.trainer_harness.per_frame_baseline import (
        PerFrameGaussianModel,
        render_per_frame_gaussians,
        render_per_frame_gaussians_fast_mac,
    )
    from research_project.trainer_harness.tile_metal_autograd import render_uvt_tubes_metal_tile_backward
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = Path(__file__).resolve().parents[1] / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from data import load_video_target
    from model import ScreenTimeTubeModel, render_model
    from per_frame_baseline import PerFrameGaussianModel, render_per_frame_gaussians, render_per_frame_gaussians_fast_mac
    from tile_metal_autograd import render_uvt_tubes_metal_tile_backward


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


def render_uvt_model(model: ScreenTimeTubeModel, *, backend: str) -> torch.Tensor:
    if backend == "dense":
        return render_model(model)
    if backend == "metal_tile":
        ma, q_uvt, depth0, depth_beta, opacity, color = model.tensors()
        return render_uvt_tubes_metal_tile_backward(ma, q_uvt, depth0, depth_beta, opacity, color, model.config)
    raise ValueError("backend must be one of: dense, metal_tile")


def fit_uvt(
    model: ScreenTimeTubeModel,
    target: torch.Tensor,
    *,
    steps: int,
    lr: float,
    final_lr: float | None,
    final_lr_start_step: int | None,
    backend: str,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps + 1):
        if final_lr is not None and final_lr_start_step is not None and step == final_lr_start_step:
            for group in optimizer.param_groups:
                group["lr"] = final_lr
        optimizer.zero_grad(set_to_none=True)
        image = render_uvt_model(model, backend=backend)
        loss = torch.mean((image - target).square())
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
) -> list[float]:
    optimizer = torch.optim.Adam([model.raw_color, model.raw_opacity], lr=lr)
    losses: list[float] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image = render_uvt_model(model, backend=backend)
        loss = torch.mean((image - target).square())
        losses.append(float(loss.detach().cpu()))
        if step == steps:
            break
        loss.backward()
        optimizer.step()
    return losses


def mse_to_psnr(mse: float) -> float:
    return -10.0 * torch.log10(torch.tensor(max(mse, 1.0e-12), dtype=torch.float32)).item()


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


def _as_uint8(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().cpu().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def write_contact_sheet(
    path: Path,
    target: torch.Tensor,
    uvt: torch.Tensor | None,
    per_frame: torch.Tensor | None,
) -> None:
    frame_count = min(int(target.shape[0]), 4)
    rows = [target[:frame_count]]
    if uvt is not None:
        rows.append(uvt[:frame_count])
    if per_frame is not None:
        rows.append(per_frame[:frame_count])
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


def run_video_fit_comparison(
    *,
    video_path: Path,
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
    uvt_appearance_refine_steps: int,
    uvt_appearance_lr: float,
    uvt_temporal_split_step: int | None,
    uvt_temporal_split_offset: float,
    uvt_temporal_split_precision_scale: float,
    uvt_temporal_split_opacity_scale: float,
    uvt_temporal_split_depth_offset: float,
    uvt_temporal_split_lr: float | None,
    uvt_render_backend: str,
    uvt_tile_t: int,
    uvt_tile_capacity: int,
    render_benchmark_repeats: int,
    skip_uvt: bool,
    skip_per_frame: bool,
    contact_sheet: Path | None,
) -> dict[str, object]:
    dev = torch.device(device)
    if uvt_render_backend == "metal_tile" and dev.type != "mps":
        raise ValueError("--uvt-render-backend=metal_tile requires --device=mps")
    if per_frame_render_backend == "fast_mac" and dev.type != "mps":
        raise ValueError("--per-frame-render-backend=fast_mac requires --device=mps")
    if uvt_temporal_split_step is not None:
        if uvt_temporal_split_step <= 0 or uvt_temporal_split_step >= steps:
            raise ValueError("--uvt-temporal-split-step must be greater than 0 and less than --steps")
        if uvt_final_lr is not None or uvt_final_lr_start_step is not None:
            raise ValueError("temporal split is intentionally not mixed with staged LR in this benchmark")
    target = load_video_target(video_path, target_size=target_size, max_frames=max_frames, device=dev)
    config = UVTRenderConfig(
        height=int(target.shape[1]),
        width=int(target.shape[2]),
        frames=int(target.shape[0]),
        tile_t=uvt_tile_t,
        tile_capacity=uvt_tile_capacity,
    )
    apply_uvt_tile_env(config)
    uvt_model = None
    if not skip_uvt:
        uvt_model = make_uvt_model(
            target,
            config,
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
    uvt_split_info = None
    uvt_losses = None
    uvt_main_final_loss = None
    uvt_appearance_losses = None
    if uvt_model is not None:
        if uvt_temporal_split_step is None:
            uvt_losses = fit_uvt(
                uvt_model,
                target,
                steps=steps,
                lr=lr,
                final_lr=uvt_final_lr,
                final_lr_start_step=uvt_final_lr_start_step,
                backend=uvt_render_backend,
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
        per_frame_l1 = None if per_frame_image is None else torch.mean((per_frame_image - target).abs()).item()
        per_frame_mse = None if per_frame_image is None else torch.mean((per_frame_image - target).square()).item()

    if contact_sheet is not None:
        write_contact_sheet(contact_sheet, target, uvt_image, per_frame_image)

    row = {
        "video_path": str(video_path),
        "frames": config.frames,
        "height": config.height,
        "width": config.width,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "device": str(dev),
        "contact_sheet": None if contact_sheet is None else str(contact_sheet),
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
            "tile_t": uvt_tile_t,
            "tile_capacity": uvt_tile_capacity,
            "render_benchmark_repeats": render_benchmark_repeats,
            "render_benchmark_ms": summarize_ms([] if uvt_render_samples is None else uvt_render_samples),
            "final_lr": uvt_final_lr,
            "final_lr_start_step": uvt_final_lr_start_step,
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
            "render_ms": per_frame_render_ms,
            "wall_clock_ms": per_frame_ms,
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
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
    parser.add_argument("--uvt-appearance-refine-steps", type=int, default=0)
    parser.add_argument("--uvt-appearance-lr", type=float, default=0.04)
    parser.add_argument("--uvt-temporal-split-step", type=int)
    parser.add_argument("--uvt-temporal-split-offset", type=float, default=0.5)
    parser.add_argument("--uvt-temporal-split-precision-scale", type=float, default=2.0)
    parser.add_argument("--uvt-temporal-split-opacity-scale", type=float, default=1.0)
    parser.add_argument("--uvt-temporal-split-depth-offset", type=float, default=1.0e-4)
    parser.add_argument("--uvt-temporal-split-lr", type=float)
    parser.add_argument("--uvt-render-backend", choices=("dense", "metal_tile"), default="dense")
    parser.add_argument("--uvt-tile-t", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--uvt-tile-capacity", type=int, choices=(32, 64, 128, 256), default=128)
    parser.add_argument("--render-benchmark-repeats", type=int, default=1)
    parser.add_argument("--skip-uvt", action="store_true")
    parser.add_argument("--skip-per-frame", action="store_true")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    args = parser.parse_args()

    row = run_video_fit_comparison(
        video_path=args.video_path,
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
        uvt_appearance_refine_steps=args.uvt_appearance_refine_steps,
        uvt_appearance_lr=args.uvt_appearance_lr,
        uvt_temporal_split_step=args.uvt_temporal_split_step,
        uvt_temporal_split_offset=args.uvt_temporal_split_offset,
        uvt_temporal_split_precision_scale=args.uvt_temporal_split_precision_scale,
        uvt_temporal_split_opacity_scale=args.uvt_temporal_split_opacity_scale,
        uvt_temporal_split_depth_offset=args.uvt_temporal_split_depth_offset,
        uvt_temporal_split_lr=args.uvt_temporal_split_lr,
        uvt_render_backend=args.uvt_render_backend,
        uvt_tile_t=args.uvt_tile_t,
        uvt_tile_capacity=args.uvt_tile_capacity,
        render_benchmark_repeats=args.render_benchmark_repeats,
        skip_uvt=args.skip_uvt,
        skip_per_frame=args.skip_per_frame,
        contact_sheet=args.contact_sheet,
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
