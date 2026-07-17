from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    UVTRenderConfig,
    count_projective_trace_dense_per_frame_tile_pairs,
    direct_backward_projective_trace_cell_interval_atlas_metal,
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_metal,
    has_projective_trace_cell_interval_metal,
    pack_projective_trace_tile_time_bins,
    projective_trace_windows_to_cell_trace_atlas,
    render_projective_trace_cell_atlas_metal,
    render_projective_trace_cell_interval_atlas_metal,
    split_projective_trace_windows,
)


def _pixel_orbit_coeffs(
    *,
    point_x: float,
    base_depth: float,
    vertical: float,
    center_u: float,
    center_v: float,
    scale: float,
) -> list[float]:
    raw_u = torch.tensor([point_x, 2.0, -point_x], dtype=torch.float32)
    raw_v = torch.tensor([vertical, 0.0, vertical], dtype=torch.float32)
    depth = torch.tensor([base_depth + 0.25, 2.0 * point_x, base_depth - 0.25], dtype=torch.float32)
    pixel_u = float(center_u) * depth + float(scale) * raw_u
    pixel_v = float(center_v) * depth + float(scale) * raw_v
    return [*pixel_u.tolist(), *pixel_v.tolist(), *depth.tolist()]


def make_orbit_coeffs() -> torch.Tensor:
    return torch.tensor(
        [
            _pixel_orbit_coeffs(point_x=0.25, base_depth=2.5, vertical=0.1, center_u=48.0, center_v=40.0, scale=18.0),
            _pixel_orbit_coeffs(point_x=-0.20, base_depth=2.8, vertical=-0.1, center_u=72.0, center_v=58.0, scale=16.0),
            _pixel_orbit_coeffs(point_x=0.10, base_depth=3.2, vertical=0.0, center_u=38.0, center_v=82.0, scale=14.0),
            _pixel_orbit_coeffs(point_x=-0.30, base_depth=2.7, vertical=0.2, center_u=92.0, center_v=36.0, scale=12.0),
        ],
        dtype=torch.float32,
    ).contiguous()


def orbit_times(frame_count: int, *, degrees: float) -> torch.Tensor:
    theta = torch.linspace(-math.radians(degrees), math.radians(degrees), frame_count, dtype=torch.float32)
    return torch.tan(0.5 * theta).contiguous()


def _apply_metal_tile_env(config: UVTRenderConfig) -> None:
    os.environ["STAR_UVT_TILE_X"] = str(config.tile_x)
    os.environ["STAR_UVT_TILE_Y"] = str(config.tile_y)
    os.environ["STAR_UVT_TILE_T"] = str(config.tile_t)
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(config.tile_capacity)


def _time_metal_cell_render(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: torch.Tensor,
    config: UVTRenderConfig,
    *,
    sigma_px: float,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"metal_skipped": "MPS is not available"}
    if not has_projective_trace_cell_metal():
        return {"metal_skipped": "projective cell Metal op is unavailable"}

    _apply_metal_tile_env(config)
    device = torch.device("mps")
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to(device),
        opacity=atlas.opacity.to(device),
        color=atlas.color.to(device),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    times_mps = times.to(device)
    with torch.no_grad():
        for _ in range(warmup_iterations):
            render_projective_trace_cell_atlas_metal(atlas_mps, times_mps, config, sigma_px=sigma_px)
        torch.mps.synchronize()
        started_at = time.perf_counter()
        image = None
        for _ in range(iterations):
            image = render_projective_trace_cell_atlas_metal(atlas_mps, times_mps, config, sigma_px=sigma_px)
        torch.mps.synchronize()
    if image is None:
        raise AssertionError("Metal render did not run")
    return {
        "metal_tile_t": int(config.tile_t),
        "metal_render_ms": (time.perf_counter() - started_at) * 1000.0 / float(iterations),
        "metal_image_sum": float(image.sum().detach().cpu().item()),
    }


def _time_metal_interval_cell_render(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: torch.Tensor,
    config: UVTRenderConfig,
    *,
    sigma_px: float,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"metal_interval_skipped": "MPS is not available"}
    if not has_projective_trace_cell_interval_metal():
        return {"metal_interval_skipped": "projective interval cell Metal op is unavailable"}

    _apply_metal_tile_env(config)
    device = torch.device("mps")
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to(device),
        opacity=atlas.opacity.to(device),
        color=atlas.color.to(device),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    times_mps = times.to(device)
    with torch.no_grad():
        for _ in range(warmup_iterations):
            render_projective_trace_cell_interval_atlas_metal(atlas_mps, times_mps, config, sigma_px=sigma_px)
        torch.mps.synchronize()
        started_at = time.perf_counter()
        image = None
        for _ in range(iterations):
            image = render_projective_trace_cell_interval_atlas_metal(atlas_mps, times_mps, config, sigma_px=sigma_px)
        torch.mps.synchronize()
    if image is None:
        raise AssertionError("Metal interval render did not run")
    return {
        "metal_interval_render_ms": (time.perf_counter() - started_at) * 1000.0 / float(iterations),
        "metal_interval_image_sum": float(image.sum().detach().cpu().item()),
    }


def _time_metal_interval_cell_backward(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: torch.Tensor,
    config: UVTRenderConfig,
    *,
    sigma_px: float,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {"metal_interval_backward_skipped": "MPS is not available"}
    if not has_projective_trace_cell_interval_backward_metal():
        return {"metal_interval_backward_skipped": "projective interval cell Metal backward op is unavailable"}

    _apply_metal_tile_env(config)
    device = torch.device("mps")
    atlas_mps = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to(device),
        opacity=atlas.opacity.to(device),
        color=atlas.color.to(device),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
    )
    times_mps = times.to(device)
    sample_count = int(config.frames) * int(config.height) * int(config.width) * 3
    grad_image = torch.linspace(-0.25, 0.35, steps=sample_count, dtype=torch.float32, device=device)
    grad_image = grad_image.reshape(int(config.frames), int(config.height), int(config.width), 3).contiguous()
    with torch.no_grad():
        for _ in range(warmup_iterations):
            direct_backward_projective_trace_cell_interval_atlas_metal(
                atlas_mps,
                times_mps,
                grad_image,
                config,
                sigma_px=sigma_px,
            )
        torch.mps.synchronize()
        started_at = time.perf_counter()
        grads = None
        for _ in range(iterations):
            grads = direct_backward_projective_trace_cell_interval_atlas_metal(
                atlas_mps,
                times_mps,
                grad_image,
                config,
                sigma_px=sigma_px,
            )
        torch.mps.synchronize()
    if grads is None:
        raise AssertionError("Metal interval backward did not run")
    return {
        "metal_interval_backward_ms": (time.perf_counter() - started_at) * 1000.0 / float(iterations),
        "metal_interval_grad_coeff_abs_sum": float(grads.grad_coeffs.abs().sum().detach().cpu().item()),
        "metal_interval_grad_opacity_abs_sum": float(grads.grad_opacity.abs().sum().detach().cpu().item()),
        "metal_interval_grad_color_abs_sum": float(grads.grad_color.abs().sum().detach().cpu().item()),
    }


def run_row(
    *,
    coeffs: torch.Tensor,
    frame_count: int,
    orbit_degrees: float,
    image_size: int,
    tile_size: int,
    uv_padding: float,
    tile_capacity: int,
    metal_tile_t: int,
    run_metal: bool,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    times = orbit_times(frame_count, degrees=orbit_degrees)
    colors = torch.ones((coeffs.shape[0], 3), dtype=torch.float32)
    opacities = torch.full((coeffs.shape[0],), 0.5, dtype=torch.float32)
    windows = split_projective_trace_windows(
        coeffs,
        times,
        degree=2,
        max_residual_uv=0.75,
        min_denominator_abs=1.0e-3,
        min_samples=3,
    )
    if not all(window.accepted for window in windows):
        reasons = sorted({window.reason for window in windows if not window.accepted})
        raise RuntimeError(f"unaccepted projective windows at {frame_count} frames: {reasons}")

    atlas = projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacities,
        color=colors,
        image_width=image_size,
        image_height=image_size,
        tile_size=tile_size,
        uv_padding=uv_padding,
    )
    dense_pairs = count_projective_trace_dense_per_frame_tile_pairs(
        coeffs,
        times,
        image_width=image_size,
        image_height=image_size,
        tile_size=tile_size,
        uv_padding=uv_padding,
    )
    interval_bins = pack_projective_trace_tile_time_bins(
        atlas.cells,
        image_width=image_size,
        image_height=image_size,
        frames=frame_count,
        tile_x=tile_size,
        tile_y=tile_size,
        tile_t=frame_count,
        tile_capacity=tile_capacity,
    )
    slab_bins = pack_projective_trace_tile_time_bins(
        atlas.cells,
        image_width=image_size,
        image_height=image_size,
        frames=frame_count,
        tile_x=tile_size,
        tile_y=tile_size,
        tile_t=metal_tile_t,
        tile_capacity=tile_capacity,
    )
    interval_entries = int(interval_bins.tile_counts.sum().item())
    slab_entries = int(slab_bins.tile_counts.sum().item())
    row: dict[str, Any] = {
        "frames": frame_count,
        "primitive_count": int(coeffs.shape[0]),
        "window_count": len(windows),
        "cell_trace_rows": int(atlas.coeffs.shape[0]),
        "cell_count": len(atlas.cells),
        "dense_per_frame_tile_pairs": dense_pairs,
        "interval_packed_tile_entries": interval_entries,
        "interval_pair_ratio": interval_entries / float(max(dense_pairs, 1)),
        "metal_slab_tile_t": metal_tile_t,
        "metal_slab_packed_tile_entries": slab_entries,
        "metal_slab_pair_ratio": slab_entries / float(max(dense_pairs, 1)),
    }
    if run_metal:
        config = UVTRenderConfig(
            height=image_size,
            width=image_size,
            frames=frame_count,
            tile_x=tile_size,
            tile_y=tile_size,
            tile_t=metal_tile_t,
            tile_capacity=tile_capacity,
            alpha_threshold=1.0e-6,
            transmittance_threshold=0.0,
            max_alpha=1.0,
        )
        row.update(
            _time_metal_cell_render(
                atlas,
                times,
                config,
                sigma_px=1.6,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
            )
        )
        row.update(
            _time_metal_interval_cell_render(
                atlas,
                times,
                config,
                sigma_px=1.6,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
            )
        )
        row.update(
            _time_metal_interval_cell_backward(
                atlas,
                times,
                config,
                sigma_px=1.6,
                iterations=iterations,
                warmup_iterations=warmup_iterations,
            )
        )
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    summary: dict[str, Any] = {
        "frame_growth": float(last["frames"]) / float(first["frames"]),
        "dense_pair_growth": float(last["dense_per_frame_tile_pairs"]) / float(first["dense_per_frame_tile_pairs"]),
        "interval_entry_growth": float(last["interval_packed_tile_entries"]) / float(first["interval_packed_tile_entries"]),
        "metal_slab_entry_growth": float(last["metal_slab_packed_tile_entries"]) / float(first["metal_slab_packed_tile_entries"]),
        "interval_ratio_start": first["interval_pair_ratio"],
        "interval_ratio_end": last["interval_pair_ratio"],
        "metal_slab_ratio_start": first["metal_slab_pair_ratio"],
        "metal_slab_ratio_end": last["metal_slab_pair_ratio"],
    }
    if "metal_render_ms" in first and "metal_render_ms" in last:
        summary["metal_slab_render_ms_start"] = first["metal_render_ms"]
        summary["metal_slab_render_ms_end"] = last["metal_render_ms"]
        summary["metal_slab_render_ms_growth"] = float(last["metal_render_ms"]) / float(first["metal_render_ms"])
    if "metal_interval_render_ms" in first and "metal_interval_render_ms" in last:
        summary["metal_interval_render_ms_start"] = first["metal_interval_render_ms"]
        summary["metal_interval_render_ms_end"] = last["metal_interval_render_ms"]
        summary["metal_interval_render_ms_growth"] = float(last["metal_interval_render_ms"]) / float(first["metal_interval_render_ms"])
    if "metal_interval_backward_ms" in first and "metal_interval_backward_ms" in last:
        summary["metal_interval_backward_ms_start"] = first["metal_interval_backward_ms"]
        summary["metal_interval_backward_ms_end"] = last["metal_interval_backward_ms"]
        summary["metal_interval_backward_ms_growth"] = float(last["metal_interval_backward_ms"]) / float(first["metal_interval_backward_ms"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-counts", default="4,8,16,32,64")
    parser.add_argument("--orbit-degrees", type=float, default=45.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--tile-size", type=int, choices=(8, 16), default=16)
    parser.add_argument("--uv-padding", type=float, default=5.0)
    parser.add_argument("--tile-capacity", type=int, choices=(32, 64, 128, 256, 1024), default=256)
    parser.add_argument("--metal-tile-t", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--run-metal", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    coeffs = make_orbit_coeffs()
    rows = [
        run_row(
            coeffs=coeffs,
            frame_count=int(frame_count.strip()),
            orbit_degrees=args.orbit_degrees,
            image_size=args.image_size,
            tile_size=args.tile_size,
            uv_padding=args.uv_padding,
            tile_capacity=args.tile_capacity,
            metal_tile_t=args.metal_tile_t,
            run_metal=args.run_metal,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
        )
        for frame_count in args.frame_counts.split(",")
        if frame_count.strip()
    ]
    report = {"summary": summarize(rows), "rows": rows}
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
