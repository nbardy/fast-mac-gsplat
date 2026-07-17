from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    UVTRenderConfig,
    direct_atomic_backward,
    direct_fixedpoint_backward,
    direct_split_fixedpoint_backward,
    direct_serial_backward,
    render_uvt_tubes,
    stable_backward_samples,
    stable_backward_samples_with_keys,
    tile_pair_backward_samples,
    tile_pair_backward_samples_compensated,
    tile_pair_atomic_backward,
    tile_pair_fixedpoint_backward,
    tile_pair_grouped_backward_samples,
    tile_pair_parallel_backward_samples,
    tile_pair_reduced_backward,
    tile_pair_reduced_parallel_backward,
    tile_pair_scanline_backward_samples,
    tile_pair_sharedsort_backward_samples,
    tile_pair_suffix_backward_samples,
    tile_pair_suffix_reduced_backward,
    tile_pair_target_bounds_backward_samples,
)

try:
    from research_project.trainer_harness.data import load_video_target
    from research_project.trainer_harness.model import ScreenTimeTubeModel
    from research_project.trainer_harness.tile_metal_autograd import render_uvt_tubes_metal_tile_backward
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = Path(__file__).resolve().parents[1] / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from data import load_video_target
    from model import ScreenTimeTubeModel
    from tile_metal_autograd import render_uvt_tubes_metal_tile_backward


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def apply_uvt_tile_env(config: UVTRenderConfig) -> None:
    os.environ["STAR_UVT_TILE_X"] = str(config.tile_x)
    os.environ["STAR_UVT_TILE_Y"] = str(config.tile_y)
    os.environ["STAR_UVT_TILE_T"] = str(config.tile_t)
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(config.tile_capacity)


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float | list[float]]]:
    if not rows:
        return {}
    out = {}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        out[key] = {
            "samples": values,
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
    return out


def timed(device: torch.device, fn) -> tuple[Any, float]:
    synchronize(device)
    started_at = time.perf_counter()
    value = fn()
    synchronize(device)
    return value, (time.perf_counter() - started_at) * 1000.0


def _fit_loaded_video_target_frames(
    target: torch.Tensor,
    *,
    requested_frame_count: int,
    allow_repeat_loaded_frames: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    requested_frame_count = int(requested_frame_count)
    loaded_frame_count = int(target.shape[0])
    if requested_frame_count < 1:
        raise ValueError(f"requested_frame_count must be positive, got {requested_frame_count}")
    if loaded_frame_count < 1:
        raise ValueError("video loader returned zero frames")
    meta = {
        "requested_frame_count": requested_frame_count,
        "loaded_frame_count": loaded_frame_count,
        "repeat_loaded_frames": bool(allow_repeat_loaded_frames),
        "repeat_loaded_frames_used": False,
        "repeat_loaded_frames_scope": None,
    }
    if loaded_frame_count == requested_frame_count:
        return target, meta
    if loaded_frame_count > requested_frame_count:
        raise ValueError(
            f"video loader returned {loaded_frame_count} frames for requested "
            f"{requested_frame_count}; expected the data loader to crop to the requested count"
        )
    if not allow_repeat_loaded_frames:
        raise ValueError(
            f"video loader returned only {loaded_frame_count} frames for requested "
            f"{requested_frame_count}; pass --repeat-loaded-frames for a synthetic repeated-fixture "
            "speed-scaling smoke, or use a longer real fixture"
        )
    source_frame_indices = torch.arange(requested_frame_count, dtype=torch.long, device=target.device) % loaded_frame_count
    meta["repeat_loaded_frames_used"] = True
    meta["repeat_loaded_frames_scope"] = "video_target"
    return target.index_select(0, source_frame_indices).contiguous(), meta


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


def tile_load_proxy(q_uvt: torch.Tensor, opacity: torch.Tensor, config: UVTRenderConfig) -> torch.Tensor:
    opacity_safe = opacity.clamp_min(float(config.alpha_threshold) * 1.0001)
    tau = -2.0 * torch.log((float(config.alpha_threshold) / opacity_safe).clamp_min(1.0e-8))
    half_extent = torch.sqrt((tau.unsqueeze(-1) * _uvt_inv_diag(q_uvt)).clamp_min(0.0))
    span_x = 1.0 + 2.0 * half_extent[:, 0] / float(config.tile_x)
    span_y = 1.0 + 2.0 * half_extent[:, 1] / float(config.tile_y)
    span_t = 1.0 + 2.0 * half_extent[:, 2] / float(config.tile_t)
    return (span_x * span_y * span_t).mean()


def tile_load_regularization(
    model: ScreenTimeTubeModel,
    config: UVTRenderConfig,
    *,
    weight: float,
    target: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ma, q_uvt, _depth0, _depth_beta, opacity, _color = model.tensors()
    del ma
    proxy = tile_load_proxy(q_uvt, opacity, config)
    if target > 0.0:
        target_tensor = proxy.new_tensor(float(target))
        penalty = torch.relu(proxy - target_tensor).div(target_tensor).square()
    else:
        penalty = proxy
    return proxy, proxy.new_tensor(float(weight)) * penalty


def run_case(
    *,
    video_path: Path,
    target_size: int,
    max_frames: int,
    tube_count: int,
    seed: int,
    spatial_precision: float,
    temporal_precision: float,
    opacity: float,
    tile_t: int,
    tile_capacity: int,
    lr: float,
    steps: int,
    warmup_steps: int,
    sample_count_every: int,
    pair_count_every: int,
    reduction_mode: str,
    sample_emission_mode: str,
    tile_load_reg_weight: float,
    tile_load_target: float,
    repeat_loaded_frames: bool = False,
) -> dict[str, Any]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise RuntimeError("uvt_train_step_timing_probe requires MPS")
    target = load_video_target(video_path, target_size=target_size, max_frames=max_frames, device=device)
    target, frame_count_meta = _fit_loaded_video_target_frames(
        target,
        requested_frame_count=max_frames,
        allow_repeat_loaded_frames=repeat_loaded_frames,
    )
    config = UVTRenderConfig(
        height=int(target.shape[1]),
        width=int(target.shape[2]),
        frames=int(target.shape[0]),
        tile_t=tile_t,
        tile_capacity=tile_capacity,
    )
    apply_uvt_tile_env(config)
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
            "keyed sort reduction requires --uvt-sample-emission-mode with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, or tile_pair_suffix"
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
        raise ValueError(f"{sample_emission_mode} bypasses the reducer and requires --uvt-reduction-mode index_add")
    model = ScreenTimeTubeModel.from_video_samples(
        target,
        config,
        tube_count=tube_count,
        seed=seed,
        spatial_precision=spatial_precision,
        temporal_precision=temporal_precision,
        opacity=opacity,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    measured_rows: list[dict[str, float]] = []
    losses: list[float] = []

    for step in range(warmup_steps + steps):
        row: dict[str, float] = {}
        step_started = time.perf_counter()
        _, zero_ms = timed(device, lambda: optimizer.zero_grad(set_to_none=True))
        row["zero_grad_ms"] = zero_ms

        def render() -> torch.Tensor:
            ma, q_uvt, depth0, depth_beta, alpha, color = model.tensors()
            return render_uvt_tubes_metal_tile_backward(
                ma,
                q_uvt,
                depth0,
                depth_beta,
                alpha,
                color,
                config,
                reduction_mode=reduction_mode,
                sample_emission_mode=sample_emission_mode,
            )

        image, forward_ms = timed(device, render)
        row["forward_ms"] = forward_ms
        def compute_loss() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            recon_loss = torch.mean((image - target).square())
            if tile_load_reg_weight <= 0.0:
                return recon_loss, recon_loss, None, None
            tile_proxy, reg_loss = tile_load_regularization(
                model,
                config,
                weight=tile_load_reg_weight,
                target=tile_load_target,
            )
            return recon_loss + reg_loss, recon_loss, tile_proxy, reg_loss

        loss_bundle, loss_ms = timed(device, compute_loss)
        loss, recon_loss, tile_proxy, reg_loss = loss_bundle
        row["loss_ms"] = loss_ms
        _, backward_ms = timed(device, loss.backward)
        row["backward_ms"] = backward_ms
        _, optimizer_ms = timed(device, optimizer.step)
        row["optimizer_ms"] = optimizer_ms
        synchronize(device)
        row["total_ms"] = (time.perf_counter() - step_started) * 1000.0
        loss_value = float(loss.detach().cpu())
        recon_loss_value = float(recon_loss.detach().cpu())
        losses.append(loss_value)
        row["loss_value"] = loss_value
        row["recon_loss_value"] = recon_loss_value
        if tile_proxy is not None and reg_loss is not None:
            row["tile_load_proxy"] = float(tile_proxy.detach().cpu())
            row["tile_load_reg_loss"] = float(reg_loss.detach().cpu())
        if step >= warmup_steps:
            measured_step = step - warmup_steps
            if sample_count_every > 0 and measured_step % sample_count_every == 0:
                grad_image = torch.ones((config.frames, config.height, config.width, 3), dtype=torch.float32, device=device)

                def count_samples() -> tuple[torch.Tensor, ...]:
                    ma, q_uvt, depth0, depth_beta, alpha, color = model.tensors()
                    if sample_emission_mode == "with_keys":
                        return stable_backward_samples_with_keys(
                            ma.detach(),
                            q_uvt.detach(),
                            depth0.detach(),
                            depth_beta.detach(),
                            alpha.detach(),
                            color.detach(),
                            grad_image,
                            config,
                        )
                    if sample_emission_mode == "atomic_append":
                        return stable_backward_samples(
                            ma.detach(),
                            q_uvt.detach(),
                            depth0.detach(),
                            depth_beta.detach(),
                            alpha.detach(),
                            color.detach(),
                            grad_image,
                            config,
                        )
                    if sample_emission_mode in (
                        "tile_pair",
                        "tile_pair_compensated",
                        "tile_pair_grouped",
                        "tile_pair_parallel",
                        "tile_pair_scanline",
                        "tile_pair_sharedsort",
                        "tile_pair_target_bounds",
                        "tile_pair_suffix",
                    ):
                        tile_pair_fn = {
                            "tile_pair": tile_pair_backward_samples,
                            "tile_pair_compensated": tile_pair_backward_samples_compensated,
                            "tile_pair_grouped": tile_pair_grouped_backward_samples,
                            "tile_pair_parallel": tile_pair_parallel_backward_samples,
                            "tile_pair_scanline": tile_pair_scanline_backward_samples,
                            "tile_pair_sharedsort": tile_pair_sharedsort_backward_samples,
                            "tile_pair_target_bounds": tile_pair_target_bounds_backward_samples,
                            "tile_pair_suffix": tile_pair_suffix_backward_samples,
                        }[sample_emission_mode]
                        return tile_pair_fn(
                            ma.detach(),
                            q_uvt.detach(),
                            depth0.detach(),
                            depth_beta.detach(),
                            alpha.detach(),
                            color.detach(),
                            grad_image,
                            config,
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
                    ):
                        direct_backward = {
                            "direct_atomic": direct_atomic_backward,
                            "direct_fixedpoint": direct_fixedpoint_backward,
                            "direct_split_fixedpoint": direct_split_fixedpoint_backward,
                            "direct_serial": direct_serial_backward,
                            "tile_pair_atomic": tile_pair_atomic_backward,
                            "tile_pair_fixedpoint": tile_pair_fixedpoint_backward,
                            "tile_pair_reduced": tile_pair_reduced_backward,
                            "tile_pair_reduced_parallel": tile_pair_reduced_parallel_backward,
                            "tile_pair_suffix_reduced": tile_pair_suffix_reduced_backward,
                        }[sample_emission_mode]
                        return direct_backward(
                            ma.detach(),
                            q_uvt.detach(),
                            depth0.detach(),
                            depth_beta.detach(),
                            alpha.detach(),
                            color.detach(),
                            grad_image,
                            config,
                        )
                    raise ValueError(
                        "sample emission mode must be one of: atomic_append, with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, tile_pair_suffix, direct_atomic, direct_fixedpoint, direct_split_fixedpoint, direct_serial, tile_pair_atomic, tile_pair_fixedpoint, tile_pair_reduced, tile_pair_reduced_parallel, tile_pair_suffix_reduced"
                    )

                samples, sample_count_ms = timed(device, count_samples)
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
                ):
                    row["direct_grad_tube_count"] = float(samples[0].shape[0])
                else:
                    row["sample_count"] = float(samples[0].numel())
                row["sample_count_ms"] = sample_count_ms
            if pair_count_every > 0 and measured_step % pair_count_every == 0:

                def count_pairs() -> object:
                    ma, q_uvt, depth0, depth_beta, alpha, color = model.tensors()
                    with torch.no_grad():
                        return render_uvt_tubes(
                            ma.detach(),
                            q_uvt.detach(),
                            depth0.detach(),
                            depth_beta.detach(),
                            alpha.detach(),
                            color.detach(),
                            config,
                            return_aux=True,
                        )

                pair_result, pair_count_ms = timed(device, count_pairs)
                if not hasattr(pair_result, "stats") or pair_result.stats is None:
                    raise AssertionError("pair-count diagnostic expected UVTRenderResult stats")
                stats = pair_result.stats
                row["pair_count_ms"] = pair_count_ms
                row["uvt_tile_tube_pairs"] = float(stats.uvt_tile_tube_pairs)
                row["summed_per_frame_tile_splat_pairs"] = float(stats.summed_per_frame_tile_splat_pairs)
                row["pair_ratio"] = float(stats.pair_ratio)
                row["overflow_tile_count"] = float(stats.overflow_tile_count)
                row["unstable_tile_fraction"] = float(stats.unstable_tile_fraction)
                row["max_tile_count"] = float(stats.max_tile_count)
                row["mean_tile_count"] = float(stats.mean_tile_count)
                if "sample_count" in row:
                    row["sample_to_uvt_pair_ratio"] = row["sample_count"] / max(row["uvt_tile_tube_pairs"], 1.0)
                    row["sample_to_splat_pair_ratio"] = row["sample_count"] / max(
                        row["summed_per_frame_tile_splat_pairs"], 1.0
                    )
            measured_rows.append(row)

    return {
        "target_size": target_size,
        "frames": int(config.frames),
        "requested_frames": int(max_frames),
        "loaded_frame_count": frame_count_meta["loaded_frame_count"],
        "repeat_loaded_frames": frame_count_meta["repeat_loaded_frames"],
        "repeat_loaded_frames_used": frame_count_meta["repeat_loaded_frames_used"],
        "repeat_loaded_frames_scope": frame_count_meta["repeat_loaded_frames_scope"],
        "tube_count": tube_count,
        "seed": seed,
        "spatial_precision": spatial_precision,
        "temporal_precision": temporal_precision,
        "opacity": opacity,
        "tile_t": config.tile_t,
        "tile_capacity": config.tile_capacity,
        "lr": lr,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "sample_count_every": sample_count_every,
        "pair_count_every": pair_count_every,
        "reduction_mode": reduction_mode,
        "sample_emission_mode": sample_emission_mode,
        "tile_load_reg_weight": tile_load_reg_weight,
        "tile_load_target": tile_load_target,
        "device": str(device),
        "rows": measured_rows,
        "summary": summarize(measured_rows),
        "losses": losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--tube-count", type=int, default=7168)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--spatial-precision", type=float, default=0.125)
    parser.add_argument("--temporal-precision", type=float, default=2.0)
    parser.add_argument("--opacity", type=float, default=0.7)
    parser.add_argument("--uvt-tile-t", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--uvt-tile-capacity", type=int, choices=(32, 64, 128, 256), default=128)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--sample-count-every", type=int, default=0)
    parser.add_argument("--pair-count-every", type=int, default=0)
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
    parser.add_argument("--tile-load-reg", type=float, default=0.0)
    parser.add_argument("--tile-load-target", type=float, default=0.0)
    parser.add_argument(
        "--repeat-loaded-frames",
        action="store_true",
        help=(
            "Repeat a shorter loaded video fixture when --max-frames exceeds the real fixture. "
            "This is a synthetic speed-scaling smoke, not a real longer-video quality run."
        ),
    )
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    row = run_case(
        video_path=args.video_path,
        target_size=args.target_size,
        max_frames=args.max_frames,
        tube_count=args.tube_count,
        seed=args.seed,
        spatial_precision=args.spatial_precision,
        temporal_precision=args.temporal_precision,
        opacity=args.opacity,
        tile_t=args.uvt_tile_t,
        tile_capacity=args.uvt_tile_capacity,
        lr=args.lr,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        sample_count_every=args.sample_count_every,
        pair_count_every=args.pair_count_every,
        reduction_mode=args.uvt_reduction_mode,
        sample_emission_mode=args.uvt_sample_emission_mode,
        tile_load_reg_weight=args.tile_load_reg,
        tile_load_target=args.tile_load_target,
        repeat_loaded_frames=bool(args.repeat_loaded_frames),
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
