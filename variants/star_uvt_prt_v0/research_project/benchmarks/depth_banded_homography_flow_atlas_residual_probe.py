from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch
from torch import Tensor


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
    _image_metrics,
    _per_tube_max_error,
    _render_from_centers,
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


def _homography_matrices(
    band_depths: Tensor,
    k_seq: Tensor,
    w2c_seq: Tensor,
    *,
    ref_frame: int,
) -> Tensor:
    frames = int(k_seq.shape[0])
    bands = int(band_depths.numel())
    k_ref_inv = torch.linalg.inv(k_seq[ref_frame])
    c2w_ref = torch.linalg.inv(w2c_seq[ref_frame])
    selector = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).view(1, 3)
    matrices = torch.empty((frames, bands, 3, 3), dtype=torch.float32)
    for frame in range(frames):
        ref_to_frame = w2c_seq[frame] @ c2w_ref
        rotation = ref_to_frame[:3, :3]
        translation = ref_to_frame[:3, 3].view(3, 1)
        for band, depth in enumerate(band_depths.tolist()):
            plane_map = float(depth) * rotation @ k_ref_inv + translation @ selector
            matrices[frame, band] = k_seq[frame] @ plane_map
    return matrices


def _apply_homography(matrix: Tensor, uv: Tensor) -> Tensor:
    uv1 = torch.cat((uv, torch.ones((int(uv.shape[0]), 1), dtype=torch.float32)), dim=-1)
    h = uv1 @ matrix.T
    z = h[:, 2].clamp_min(1.0e-6)
    return torch.stack((h[:, 0] / z, h[:, 1] / z), dim=-1)


def _inverse_homography_atlas_targets(
    direct_centers: Tensor,
    homographies: Tensor,
    assignments: Tensor,
) -> Tensor:
    frames = int(direct_centers.shape[0])
    tube_count = int(direct_centers.shape[1])
    bands = int(homographies.shape[1])
    atlas = torch.empty((frames, tube_count, 2), dtype=torch.float32)
    for frame in range(frames):
        for band in range(bands):
            mask = assignments == band
            if not bool(mask.any()):
                continue
            inv_h = torch.linalg.inv(homographies[frame, band])
            atlas[frame, mask] = _apply_homography(inv_h, direct_centers[frame, mask])
    return atlas


def _forward_homography_centers(
    atlas_centers: Tensor,
    homographies: Tensor,
    assignments: Tensor,
) -> Tensor:
    frames = int(atlas_centers.shape[0])
    tube_count = int(atlas_centers.shape[1])
    bands = int(homographies.shape[1])
    centers = torch.empty((frames, tube_count, 2), dtype=torch.float32)
    for frame in range(frames):
        for band in range(bands):
            mask = assignments == band
            if not bool(mask.any()):
                continue
            centers[frame, mask] = _apply_homography(homographies[frame, band], atlas_centers[frame, mask])
    return centers


def _tile_estimate(
    centers: Tensor,
    batch: Any,
    lambda_uv: Tensor,
    times: Tensor,
    args: argparse.Namespace,
    *,
    tile_size: int,
    clamp_to_image: bool,
    tube_mask: Tensor | None = None,
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
    segmented_centers, segmented_depth, _ = _segmented_centers(
        batch,
        k_seq,
        w2c_seq,
        times,
        segments=args.segments,
        degree=1,
    )

    ref_frame = args.frames // 2
    ref_uv = direct_centers[ref_frame]
    band_depths, assignments = _depth_bands(direct_depth[ref_frame], bands=args.depth_bands)
    homographies = _homography_matrices(band_depths, k_seq, w2c_seq, ref_frame=ref_frame)
    atlas_targets = _inverse_homography_atlas_targets(direct_centers, homographies, assignments)
    atlas_residual_targets = atlas_targets - ref_uv.view(1, -1, 2)
    _, atlas_residual_recon = _fit_poly(atlas_residual_targets, times, degree=args.residual_degree)
    atlas_centers = ref_uv.view(1, -1, 2) + atlas_residual_recon
    warped_centers = _forward_homography_centers(atlas_centers, homographies, assignments)

    fallback_mask = _per_tube_max_error(warped_centers, direct_centers) > args.fallback_max_px
    nonfallback_mask = ~fallback_mask
    hybrid_centers = torch.where(fallback_mask.view(1, -1, 1), prt_centers, warped_centers)

    atlas_nonfallback = _tile_estimate(
        atlas_centers,
        batch,
        lambda_uv,
        times,
        args,
        tile_size=tile_size,
        clamp_to_image=True,
        tube_mask=nonfallback_mask,
    )
    image_nonfallback = _tile_estimate(
        warped_centers,
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
    segmented = _tile_estimate(
        segmented_centers,
        batch,
        lambda_uv,
        times,
        args,
        tile_size=tile_size,
        clamp_to_image=True,
    )

    reference_image = _render_from_centers(
        batch,
        direct_centers,
        direct_depth,
        lambda_uv,
        times,
        height=args.target_size,
        width=args.target_size,
        alpha_threshold=args.alpha_threshold,
    )
    warped_image = _render_from_centers(
        batch,
        warped_centers,
        direct_depth,
        lambda_uv,
        times,
        height=args.target_size,
        width=args.target_size,
        alpha_threshold=args.alpha_threshold,
    )
    hybrid_image = _render_from_centers(
        batch,
        hybrid_centers,
        direct_depth,
        lambda_uv,
        times,
        height=args.target_size,
        width=args.target_size,
        alpha_threshold=args.alpha_threshold,
    )
    segmented_image = _render_from_centers(
        batch,
        segmented_centers,
        segmented_depth,
        lambda_uv,
        times,
        height=args.target_size,
        width=args.target_size,
        alpha_threshold=args.alpha_threshold,
    )

    pure_metrics = _center_metrics(warped_centers, direct_centers)
    hybrid_metrics = _center_metrics(hybrid_centers, direct_centers)
    atlas_total = atlas_nonfallback["total_tile_pairs"] + prt_fallback["total_tile_pairs"]
    image_total = image_nonfallback["total_tile_pairs"] + prt_fallback["total_tile_pairs"]
    segmented_total = max(segmented["total_tile_pairs"], 1)
    return {
        "seed": seed,
        "tile_size": tile_size,
        "fallback_tubes": int(fallback_mask.sum().item()),
        "pure_center_p95_px": pure_metrics["p95_px"],
        "pure_center_max_px": pure_metrics["max_px"],
        "hybrid_center_p95_px": hybrid_metrics["p95_px"],
        "hybrid_center_max_px": hybrid_metrics["max_px"],
        "warped_image_psnr": _image_metrics(warped_image, reference_image)["psnr"],
        "hybrid_image_psnr": _image_metrics(hybrid_image, reference_image)["psnr"],
        "segmented_image_psnr": _image_metrics(segmented_image, reference_image)["psnr"],
        "atlas_hybrid_tile_pairs": atlas_total,
        "image_space_hybrid_tile_pairs": image_total,
        "segmented_f4_tile_pairs": segmented["total_tile_pairs"],
        "atlas_ratio_vs_segmented_f4": atlas_total / segmented_total,
        "image_space_ratio_vs_segmented_f4": image_total / segmented_total,
        "atlas_ratio_vs_image_space": atlas_total / max(image_total, 1),
        "atlas_saves_vs_image_space_tile_pairs": image_total - atlas_total,
        "atlas_nonfallback_tile_pairs": atlas_nonfallback["total_tile_pairs"],
        "image_nonfallback_tile_pairs": image_nonfallback["total_tile_pairs"],
        "prt_fallback_tile_pairs": prt_fallback["total_tile_pairs"],
        "pass": hybrid_metrics["max_px"] <= args.fallback_max_px
        and atlas_total < segmented["total_tile_pairs"]
        and atlas_total < image_total
        and _image_metrics(hybrid_image, reference_image)["psnr"] >= args.psnr_gate,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    seeds = _parse_ints(args.seeds)
    tile_sizes = _parse_ints(args.tile_sizes)
    rows = [_row(args, seed=seed, tile_size=tile_size) for tile_size in tile_sizes for seed in seeds]
    summary: dict[str, Any] = {}
    for tile_size in tile_sizes:
        tile_rows = [row for row in rows if row["tile_size"] == tile_size]
        summary[str(tile_size)] = {
            "pass_count": sum(1 for row in tile_rows if row["pass"]),
            "fallback_tubes": _stats([float(row["fallback_tubes"]) for row in tile_rows]),
            "hybrid_center_max_px": _stats([float(row["hybrid_center_max_px"]) for row in tile_rows]),
            "hybrid_image_psnr": _stats([float(row["hybrid_image_psnr"]) for row in tile_rows]),
            "atlas_ratio_vs_segmented_f4": _stats(
                [float(row["atlas_ratio_vs_segmented_f4"]) for row in tile_rows]
            ),
            "image_space_ratio_vs_segmented_f4": _stats(
                [float(row["image_space_ratio_vs_segmented_f4"]) for row in tile_rows]
            ),
            "atlas_ratio_vs_image_space": _stats([float(row["atlas_ratio_vs_image_space"]) for row in tile_rows]),
            "atlas_saves_vs_image_space_tile_pairs": _stats(
                [float(row["atlas_saves_vs_image_space_tile_pairs"]) for row in tile_rows]
            ),
        }
    return {
        "name": "depth_banded_homography_flow_atlas_residual_probe",
        "note": (
            "Tests the renderer-facing representation: invert each depth-band homography to fit "
            "a residual in reference-atlas coordinates, then warp reference-atlas-plus-residual "
            "centers forward. This is still dense rendering plus tile-pair estimates, not Metal time."
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
            "psnr_gate": args.psnr_gate,
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
    parser.add_argument("--psnr-gate", type=float, default=50.0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = run_probe(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
