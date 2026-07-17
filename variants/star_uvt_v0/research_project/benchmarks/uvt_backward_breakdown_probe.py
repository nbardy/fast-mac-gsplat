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
    from research_project.trainer_harness.tile_metal_autograd import _reduce_sample_bundle
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = Path(__file__).resolve().parents[1] / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from data import load_video_target
    from model import ScreenTimeTubeModel
    from tile_metal_autograd import _reduce_sample_bundle


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


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "samples": samples,
        "min": min(samples),
        "median": statistics.median(samples),
        "max": max(samples),
    }


def time_call(device: torch.device, fn) -> tuple[Any, float]:
    synchronize(device)
    started_at = time.perf_counter()
    out = fn()
    synchronize(device)
    return out, (time.perf_counter() - started_at) * 1000.0


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
    reduction_mode: str,
    sample_emission_mode: str,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "mps":
        raise RuntimeError("uvt_backward_breakdown_probe requires MPS")
    target = load_video_target(video_path, target_size=target_size, max_frames=max_frames, device=device)
    config = UVTRenderConfig(
        height=int(target.shape[1]),
        width=int(target.shape[2]),
        frames=int(target.shape[0]),
        tile_t=tile_t,
        tile_capacity=tile_capacity,
    )
    apply_uvt_tile_env(config)
    model = ScreenTimeTubeModel.from_video_samples(
        target,
        config,
        tube_count=tube_count,
        seed=seed,
        spatial_precision=spatial_precision,
        temporal_precision=temporal_precision,
        opacity=opacity,
    )
    ma, q_uvt, depth0, depth_beta, alpha, color = model.tensors()
    grad_image = torch.ones((config.frames, config.height, config.width, 3), dtype=torch.float32, device=device)

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

    def sample_backward() -> dict[str, torch.Tensor | None]:
        if sample_emission_mode == "with_keys":
            ids, grad_ma, grad_q, grad_opacity, grad_color, keys, tile_unstable = stable_backward_samples_with_keys(
                ma.detach(),
                q_uvt.detach(),
                depth0.detach(),
                depth_beta.detach(),
                alpha.detach(),
                color.detach(),
                grad_image,
                config,
            )
            return {
                "ids": ids,
                "grad_ma": grad_ma,
                "grad_q": grad_q,
                "grad_opacity": grad_opacity,
                "grad_color": grad_color,
                "keys": keys,
                "tile_unstable": tile_unstable,
            }
        if sample_emission_mode == "atomic_append":
            ids, grad_ma, grad_q, grad_opacity, grad_color, tile_unstable = stable_backward_samples(
                ma.detach(),
                q_uvt.detach(),
                depth0.detach(),
                depth_beta.detach(),
                alpha.detach(),
                color.detach(),
                grad_image,
                config,
            )
            return {
                "ids": ids,
                "grad_ma": grad_ma,
                "grad_q": grad_q,
                "grad_opacity": grad_opacity,
                "grad_color": grad_color,
                "keys": None,
                "tile_unstable": tile_unstable,
            }
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
            ids, grad_ma, grad_q, grad_opacity, grad_color, keys, tile_unstable = tile_pair_fn(
                ma.detach(),
                q_uvt.detach(),
                depth0.detach(),
                depth_beta.detach(),
                alpha.detach(),
                color.detach(),
                grad_image,
                config,
            )
            return {
                "ids": ids,
                "grad_ma": grad_ma,
                "grad_q": grad_q,
                "grad_opacity": grad_opacity,
                "grad_color": grad_color,
                "keys": keys,
                "tile_unstable": tile_unstable,
            }
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
            grad_ma, grad_q, grad_opacity, grad_color, tile_unstable = direct_backward(
                ma.detach(),
                q_uvt.detach(),
                depth0.detach(),
                depth_beta.detach(),
                alpha.detach(),
                color.detach(),
                grad_image,
                config,
            )
            return {
                "ids": None,
                "grad_ma": grad_ma,
                "grad_q": grad_q,
                "grad_opacity": grad_opacity,
                "grad_color": grad_color,
                "keys": None,
                "tile_unstable": tile_unstable,
            }
        raise ValueError(
            "sample emission mode must be one of: atomic_append, with_keys, tile_pair, tile_pair_compensated, tile_pair_grouped, tile_pair_parallel, tile_pair_scanline, tile_pair_sharedsort, tile_pair_target_bounds, tile_pair_suffix, direct_atomic, direct_fixedpoint, direct_split_fixedpoint, direct_serial, tile_pair_atomic, tile_pair_fixedpoint, tile_pair_reduced, tile_pair_reduced_parallel, tile_pair_suffix_reduced"
        )

    def reduce_samples(samples: dict[str, torch.Tensor | None]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = samples["ids"]
        grad_ma = samples["grad_ma"]
        grad_q = samples["grad_q"]
        grad_opacity = samples["grad_opacity"]
        grad_color = samples["grad_color"]
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
            if not all(isinstance(t, torch.Tensor) for t in (grad_ma, grad_q, grad_opacity, grad_color)):
                raise TypeError("direct gradient tensors are missing")
            return grad_ma, grad_q, grad_opacity, grad_color
        ids = samples["ids"]
        if not all(isinstance(t, torch.Tensor) for t in (ids, grad_ma, grad_q, grad_opacity, grad_color)):
            raise TypeError("backward sample tensors are missing")
        keys = samples["keys"]
        if keys is not None and not isinstance(keys, torch.Tensor):
            raise TypeError("sample keys must be a tensor when present")
        return _reduce_sample_bundle(
            ids,
            grad_ma,
            grad_q,
            grad_opacity,
            grad_color,
            tube_count,
            mode=reduction_mode,
            keys=keys,
        )

    for _ in range(warmup_iterations):
        samples = sample_backward()
        reduce_samples(samples)
    synchronize(device)

    sample_ms: list[float] = []
    reduce_ms: list[float] = []
    samples = None
    for _ in range(iterations):
        samples, sample_time = time_call(device, sample_backward)
        sample_ms.append(sample_time)
        _, reduce_time = time_call(device, lambda: reduce_samples(samples))
        reduce_ms.append(reduce_time)
    if samples is None:
        raise AssertionError("no backward samples collected")

    ids = samples["ids"]
    grad_ma_samples = samples["grad_ma"]
    grad_q_samples = samples["grad_q"]
    grad_opacity_samples = samples["grad_opacity"]
    grad_color_samples = samples["grad_color"]
    tile_unstable = samples["tile_unstable"]
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
        if not all(
            isinstance(t, torch.Tensor)
            for t in (grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, tile_unstable)
        ):
            raise TypeError("direct gradient tensors are missing")
    elif not all(
        isinstance(t, torch.Tensor)
        for t in (ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, tile_unstable)
    ):
        raise TypeError("backward sample tensors are missing")
    tile_count = ((config.width + config.tile_x - 1) // config.tile_x) * (
        (config.height + config.tile_y - 1) // config.tile_y
    ) * ((config.frames + config.tile_t - 1) // config.tile_t)
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
        allocated_sample_slots = tube_count
    elif sample_emission_mode in ("tile_pair", "tile_pair_compensated", "tile_pair_grouped", "tile_pair_parallel", "tile_pair_sharedsort", "tile_pair_target_bounds", "tile_pair_suffix"):
        allocated_sample_slots = tile_count * config.tile_capacity
    elif sample_emission_mode == "tile_pair_scanline":
        allocated_sample_slots = tile_count * config.tile_capacity * config.tile_t * config.tile_y
    else:
        allocated_sample_slots = tile_count * config.tile_x * config.tile_y * config.tile_t * config.tile_capacity
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
        sample_count = int(grad_ma_samples.shape[0])
        valid_sample_count = sample_count
    else:
        sample_count = int(ids.numel())
        valid_sample_count = int(((ids >= 0) & (ids < tube_count)).sum().detach().cpu())
    return {
        "target_size": target_size,
        "frames": max_frames,
        "tube_count": tube_count,
        "seed": seed,
        "spatial_precision": spatial_precision,
        "temporal_precision": temporal_precision,
        "opacity": opacity,
        "tile_t": config.tile_t,
        "tile_capacity": config.tile_capacity,
        "fixedpoint_scale": os.environ.get("STAR_UVT_FIXEDPOINT_SCALE", "1000000"),
        "split_fixedpoint_coarse_scale": os.environ.get("STAR_UVT_SPLIT_FIXEDPOINT_COARSE_SCALE", "100"),
        "split_fixedpoint_fine_scale": os.environ.get("STAR_UVT_SPLIT_FIXEDPOINT_FINE_SCALE", "1000000"),
        "reduction_mode": reduction_mode,
        "sample_emission_mode": sample_emission_mode,
        "sample_unit": (
            "tile_pair"
            if sample_emission_mode in ("tile_pair", "tile_pair_compensated", "tile_pair_grouped", "tile_pair_parallel", "tile_pair_sharedsort", "tile_pair_target_bounds", "tile_pair_suffix")
            else "tile_pair_scanline"
            if sample_emission_mode == "tile_pair_scanline"
            else "direct_tube_grad"
            if sample_emission_mode
            in (
                "direct_atomic",
                "direct_fixedpoint",
                "direct_split_fixedpoint",
                "direct_serial",
                "tile_pair_atomic",
                "tile_pair_fixedpoint",
                "tile_pair_reduced",
                "tile_pair_reduced_parallel",
                "tile_pair_suffix_reduced",
            )
            else "pixel_sample"
        ),
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "device": str(device),
        "sample_backward_ms": summarize(sample_ms),
        "reduce_bundle_ms": summarize(reduce_ms),
        "sample_plus_reduce_median_ms": statistics.median(sample_ms) + statistics.median(reduce_ms),
        "sample_count": sample_count,
        "valid_sample_count": valid_sample_count,
        "allocated_sample_slot_count": int(allocated_sample_slots),
        "compact_sample_fraction": float(sample_count) / float(max(allocated_sample_slots, 1)),
        "unstable_tile_fraction": float(tile_unstable.float().mean().detach().cpu()),
        "grad_shapes": {
            "ids": None if ids is None else list(ids.shape),
            "grad_ma": list(grad_ma_samples.shape),
            "grad_q": list(grad_q_samples.shape),
            "grad_opacity": list(grad_opacity_samples.shape),
            "grad_color": list(grad_color_samples.shape),
        },
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
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=1)
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
        reduction_mode=args.uvt_reduction_mode,
        sample_emission_mode=args.uvt_sample_emission_mode,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
