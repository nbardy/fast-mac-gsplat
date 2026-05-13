from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.depth_banded_homography_flow_residual_probe import (  # noqa: E402
    _camera,
    _center_metrics,
    _compile_prt_centers,
    _depth_bands,
    _estimate_tile_pairs,
    _fit_poly,
    _homography_flow_centers,
    _per_tube_max_error,
    _segmented_centers,
    _world_tubes,
)
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    centered_frame_times,
    direct_project_world_tubes,
)


def _parse_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _tile_estimate(
    centers: torch.Tensor,
    batch: Any,
    lambda_uv: torch.Tensor,
    times: torch.Tensor,
    args: argparse.Namespace,
    *,
    tile_size: int,
    clamp_to_image: bool,
    tube_mask: torch.Tensor | None = None,
) -> dict[str, int]:
    return _estimate_tile_pairs(
        centers,
        batch,
        lambda_uv,
        times,
        width=args.target_size,
        height=args.target_size,
        tile_x=tile_size,
        tile_y=tile_size,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        clamp_to_image=clamp_to_image,
        tube_mask=tube_mask,
    )


def _row(args: argparse.Namespace, *, seed: int, tile_size: int) -> dict[str, Any]:
    times = centered_frame_times(args.frames)
    k_seq, w2c_seq = _camera(
        args.frames,
        args.target_size,
        args.target_size,
        times,
        pan_x=args.pan_x,
        zoom=args.zoom,
        dolly_z=args.dolly_z,
    )
    batch = _world_tubes(args.tubes, seed=seed, velocity_scale=args.velocity_scale)

    direct_centers, direct_depth = direct_project_world_tubes(batch, k_seq, w2c_seq, times)
    prt_centers, _, lambda_uv, _ = _compile_prt_centers(batch, k_seq, w2c_seq, times, degree=args.prt_degree)
    segmented_centers, _, _ = _segmented_centers(
        batch,
        k_seq,
        w2c_seq,
        times,
        segments=args.segments,
        degree=1,
    )

    ref_frame = args.frames // 2
    band_depths, assignments = _depth_bands(direct_depth[ref_frame], bands=args.depth_bands)
    flow_centers = _homography_flow_centers(
        direct_centers[ref_frame],
        band_depths,
        assignments,
        k_seq,
        w2c_seq,
        ref_frame=ref_frame,
    )
    residual = direct_centers - flow_centers
    _, residual_recon = _fit_poly(residual, times, degree=args.residual_degree)
    gauge_centers = flow_centers + residual_recon
    reference_atlas_centers = direct_centers[ref_frame].view(1, -1, 2) + residual_recon
    fallback_mask = _per_tube_max_error(gauge_centers, direct_centers) > args.fallback_max_px
    nonfallback_mask = ~fallback_mask
    hybrid_centers = torch.where(fallback_mask.view(1, -1, 1), prt_centers, gauge_centers)

    residual_nonfallback = _tile_estimate(
        residual_recon,
        batch,
        lambda_uv,
        times,
        args,
        tile_size=tile_size,
        clamp_to_image=False,
        tube_mask=nonfallback_mask,
    )
    image_nonfallback = _tile_estimate(
        gauge_centers,
        batch,
        lambda_uv,
        times,
        args,
        tile_size=tile_size,
        clamp_to_image=True,
        tube_mask=nonfallback_mask,
    )
    reference_atlas_nonfallback = _tile_estimate(
        reference_atlas_centers,
        batch,
        lambda_uv,
        times,
        args,
        tile_size=tile_size,
        clamp_to_image=True,
        tube_mask=nonfallback_mask,
    )
    prt_fallback = _tile_estimate(
        prt_centers,
        batch,
        lambda_uv,
        times,
        args,
        tile_size=tile_size,
        clamp_to_image=True,
        tube_mask=fallback_mask,
    )
    direct_all = _tile_estimate(
        direct_centers,
        batch,
        lambda_uv,
        times,
        args,
        tile_size=tile_size,
        clamp_to_image=True,
    )
    segmented = _tile_estimate(
        segmented_centers,
        batch,
        lambda_uv,
        times,
        args,
        tile_size=tile_size,
        clamp_to_image=True,
    )

    flow_total = residual_nonfallback["total_tile_pairs"] + prt_fallback["total_tile_pairs"]
    reference_atlas_total = reference_atlas_nonfallback["total_tile_pairs"] + prt_fallback["total_tile_pairs"]
    image_total = image_nonfallback["total_tile_pairs"] + prt_fallback["total_tile_pairs"]
    segmented_total = max(segmented["total_tile_pairs"], 1)
    image_total_safe = max(image_total, 1)
    metrics = _center_metrics(hybrid_centers, direct_centers)
    return {
        "seed": seed,
        "tile_size": tile_size,
        "fallback_tubes": int(fallback_mask.sum().item()),
        "hybrid_center_p95_px": metrics["p95_px"],
        "hybrid_center_max_px": metrics["max_px"],
        "flow_sheared_hybrid_tile_pairs": flow_total,
        "reference_atlas_hybrid_tile_pairs": reference_atlas_total,
        "image_space_hybrid_tile_pairs": image_total,
        "direct_image_tile_pairs": direct_all["total_tile_pairs"],
        "segmented_f4_tile_pairs": segmented["total_tile_pairs"],
        "flow_sheared_ratio_vs_segmented_f4": flow_total / segmented_total,
        "reference_atlas_ratio_vs_segmented_f4": reference_atlas_total / segmented_total,
        "image_space_ratio_vs_segmented_f4": image_total / segmented_total,
        "direct_image_ratio_vs_segmented_f4": direct_all["total_tile_pairs"] / segmented_total,
        "flow_sheared_ratio_vs_image_space": flow_total / image_total_safe,
        "reference_atlas_ratio_vs_image_space": reference_atlas_total / image_total_safe,
        "flow_sheared_saves_vs_image_space_tile_pairs": image_total - flow_total,
        "reference_atlas_saves_vs_image_space_tile_pairs": image_total - reference_atlas_total,
        "residual_nonfallback_tile_pairs": residual_nonfallback["total_tile_pairs"],
        "reference_atlas_nonfallback_tile_pairs": reference_atlas_nonfallback["total_tile_pairs"],
        "image_nonfallback_tile_pairs": image_nonfallback["total_tile_pairs"],
        "prt_fallback_tile_pairs": prt_fallback["total_tile_pairs"],
        "pass": metrics["max_px"] <= args.fallback_max_px
        and flow_total < segmented["total_tile_pairs"]
        and reference_atlas_total < segmented["total_tile_pairs"]
        and flow_total < image_total
        and reference_atlas_total < image_total,
    }


def run_control(args: argparse.Namespace) -> dict[str, Any]:
    seeds = _parse_ints(args.seeds)
    tile_sizes = _parse_ints(args.tile_sizes)
    rows = [_row(args, seed=seed, tile_size=tile_size) for tile_size in tile_sizes for seed in seeds]
    summary: dict[str, Any] = {}
    for tile_size in tile_sizes:
        tile_rows = [row for row in rows if row["tile_size"] == tile_size]
        summary[str(tile_size)] = {
            "pass_count": sum(1 for row in tile_rows if row["pass"]),
            "flow_sheared_ratio_vs_segmented_f4": _stats(
                [float(row["flow_sheared_ratio_vs_segmented_f4"]) for row in tile_rows]
            ),
            "reference_atlas_ratio_vs_segmented_f4": _stats(
                [float(row["reference_atlas_ratio_vs_segmented_f4"]) for row in tile_rows]
            ),
            "image_space_ratio_vs_segmented_f4": _stats(
                [float(row["image_space_ratio_vs_segmented_f4"]) for row in tile_rows]
            ),
            "flow_sheared_ratio_vs_image_space": _stats(
                [float(row["flow_sheared_ratio_vs_image_space"]) for row in tile_rows]
            ),
            "reference_atlas_ratio_vs_image_space": _stats(
                [float(row["reference_atlas_ratio_vs_image_space"]) for row in tile_rows]
            ),
            "flow_sheared_saves_vs_image_space_tile_pairs": _stats(
                [float(row["flow_sheared_saves_vs_image_space_tile_pairs"]) for row in tile_rows]
            ),
            "reference_atlas_saves_vs_image_space_tile_pairs": _stats(
                [float(row["reference_atlas_saves_vs_image_space_tile_pairs"]) for row in tile_rows]
            ),
        }
    return {
        "name": "depth_banded_homography_flow_culling_control",
        "note": (
            "Tile-pair control for F0e. It separates the residual-coordinate culling claim from "
            "ordinary image-space culling, and includes a stricter reference-atlas-plus-residual "
            "variant for the likely renderer coordinate system. This still estimates culling work "
            "only; it is not a flow-sheared Metal render-time measurement."
        ),
        "config": {
            "seeds": seeds,
            "tile_sizes": tile_sizes,
            "target_size": args.target_size,
            "frames": args.frames,
            "tubes": args.tubes,
            "pan_x": args.pan_x,
            "zoom": args.zoom,
            "dolly_z": args.dolly_z,
            "velocity_scale": args.velocity_scale,
            "depth_bands": args.depth_bands,
            "residual_degree": args.residual_degree,
            "fallback_max_px": args.fallback_max_px,
            "tile_t": args.tile_t,
        },
        "rows": rows,
        "summary_by_tile_size": summary,
        "pass": all(bool(row["pass"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="17,23,31,47")
    parser.add_argument("--tile-sizes", default="4,8,16")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--tubes", type=int, default=256)
    parser.add_argument("--pan-x", type=float, default=0.09)
    parser.add_argument("--zoom", type=float, default=0.025)
    parser.add_argument("--dolly-z", type=float, default=0.12)
    parser.add_argument("--depth-bands", type=int, default=4)
    parser.add_argument("--residual-degree", type=int, default=3)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--prt-degree", type=int, default=2)
    parser.add_argument("--velocity-scale", type=float, default=0.01)
    parser.add_argument("--tile-t", type=int, default=4)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--fallback-max-px", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = run_control(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
