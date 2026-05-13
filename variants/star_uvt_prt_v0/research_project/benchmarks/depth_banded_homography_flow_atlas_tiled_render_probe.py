from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.depth_banded_homography_flow_atlas_residual_probe import (  # noqa: E402
    _forward_homography_centers,
    _homography_matrices,
    _inverse_homography_atlas_targets,
)
from research_project.benchmarks.depth_banded_homography_flow_residual_probe import (  # noqa: E402
    _camera,
    _compile_prt_centers,
    _depth_bands,
    _fit_poly,
    _image_metrics,
    _min_eigenvalue,
    _per_tube_max_error,
    _render_from_centers,
    _world_tubes,
)
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    centered_frame_times,
    direct_project_world_tubes,
)


def _tile_id(
    *,
    band: int,
    tx: int,
    ty: int,
    tz: int,
    tiles_x: int,
    tiles_y: int,
    tiles_t: int,
) -> int:
    return (((band * tiles_t + tz) * tiles_y + ty) * tiles_x) + tx


def _apply_homography_single(matrix: Tensor, u: float, v: float) -> tuple[float, float]:
    h0 = float(matrix[0, 0]) * u + float(matrix[0, 1]) * v + float(matrix[0, 2])
    h1 = float(matrix[1, 0]) * u + float(matrix[1, 1]) * v + float(matrix[1, 2])
    h2 = float(matrix[2, 0]) * u + float(matrix[2, 1]) * v + float(matrix[2, 2])
    z = max(h2, 1.0e-6)
    return h0 / z, h1 / z


def _build_atlas_tile_sets(
    atlas_centers: Tensor,
    assignments: Tensor,
    batch: Any,
    lambda_uv: Tensor,
    times: Tensor,
    *,
    width: int,
    height: int,
    tile_size: int,
    tile_t: int,
    alpha_threshold: float,
    support_scale: float,
) -> tuple[list[set[int]], dict[str, int]]:
    frames = int(times.numel())
    tube_count = int(atlas_centers.shape[1])
    bands = int(assignments.max().item()) + 1
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    tiles_t = math.ceil(frames / tile_t)
    tile_sets: list[set[int]] = [set() for _ in range(bands * tiles_t * tiles_y * tiles_x)]
    min_eig = _min_eigenvalue(lambda_uv)
    support_tau = 2.0 * torch.log((batch.opacity / float(alpha_threshold)).clamp_min(1.0 + 1.0e-6))
    total_pairs = 0
    max_pairs = 0
    for tube in range(tube_count):
        band = int(assignments[tube])
        for tz, start in enumerate(range(0, frames, tile_t)):
            end = min(frames, start + tile_t)
            frame_slice = slice(start, end)
            dt = times[frame_slice] - batch.t0[tube]
            budget = support_tau[tube] - batch.lambda_t[tube] * dt.square()
            mask = budget > 0.0
            if not bool(mask.any()):
                continue
            frame_centers = atlas_centers[frame_slice, tube][mask]
            radius = support_scale * (budget[mask] / min_eig[tube]).sqrt()
            min_u = float((frame_centers[:, 0] - radius).min())
            max_u = float((frame_centers[:, 0] + radius).max())
            min_v = float((frame_centers[:, 1] - radius).min())
            max_v = float((frame_centers[:, 1] + radius).max())
            tile_min_x = max(0, min(math.floor(min_u / tile_size), tiles_x - 1))
            tile_max_x = max(0, min(math.floor(max_u / tile_size), tiles_x - 1))
            tile_min_y = max(0, min(math.floor(min_v / tile_size), tiles_y - 1))
            tile_max_y = max(0, min(math.floor(max_v / tile_size), tiles_y - 1))
            pairs = max(0, tile_max_x - tile_min_x + 1) * max(0, tile_max_y - tile_min_y + 1)
            total_pairs += pairs
            max_pairs = max(max_pairs, pairs)
            for ty in range(tile_min_y, tile_max_y + 1):
                for tx in range(tile_min_x, tile_max_x + 1):
                    tile_sets[_tile_id(band=band, tx=tx, ty=ty, tz=tz, tiles_x=tiles_x, tiles_y=tiles_y, tiles_t=tiles_t)].add(tube)
    return tile_sets, {
        "tile_pairs": int(total_pairs),
        "max_pairs_per_tube_time_tile": int(max_pairs),
        "tiles_x": int(tiles_x),
        "tiles_y": int(tiles_y),
        "tiles_t": int(tiles_t),
        "bands": int(bands),
    }


def _render_atlas_tiled_cpu(
    atlas_centers: Tensor,
    warped_centers: Tensor,
    depth: Tensor,
    homographies: Tensor,
    assignments: Tensor,
    tile_sets: list[set[int]],
    batch: Any,
    lambda_uv: Tensor,
    times: Tensor,
    *,
    width: int,
    height: int,
    tile_size: int,
    tile_t: int,
    alpha_threshold: float,
    max_alpha: float,
) -> tuple[Tensor, dict[str, int]]:
    frames = int(times.numel())
    tube_count = int(atlas_centers.shape[1])
    bands = int(homographies.shape[1])
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    tiles_t = math.ceil(frames / tile_t)
    inv_h = torch.linalg.inv(homographies)
    pixel_y = torch.arange(height, dtype=torch.float32).repeat_interleave(width) + 0.5
    pixel_x = torch.arange(width, dtype=torch.float32).repeat(height) + 0.5
    dense_active_by_frame: list[list[set[int]]] = []
    for frame in range(frames):
        du = pixel_x.view(-1, 1) - warped_centers[frame, :, 0].view(1, -1)
        dv = pixel_y.view(-1, 1) - warped_centers[frame, :, 1].view(1, -1)
        spatial = (
            lambda_uv[:, 0].view(1, -1) * du.square()
            + 2.0 * lambda_uv[:, 1].view(1, -1) * du * dv
            + lambda_uv[:, 2].view(1, -1) * dv.square()
        )
        temporal = batch.lambda_t.view(1, -1) * (times[frame] - batch.t0).view(1, -1).square()
        alpha = (batch.opacity.view(1, -1) * torch.exp(-0.5 * (spatial + temporal))).clamp(max=max_alpha)
        active = alpha >= alpha_threshold
        dense_active_by_frame.append(
            [set(torch.nonzero(active[pixel], as_tuple=False).flatten().tolist()) for pixel in range(width * height)]
        )
    image = torch.zeros((frames, height, width, 3), dtype=torch.float32)
    total_candidate_evals = 0
    max_candidates_per_pixel = 0
    missing_active_candidates = 0
    for frame in range(frames):
        tz = frame // tile_t
        t = times[frame]
        for y in range(height):
            py = float(y) + 0.5
            for x in range(width):
                px = float(x) + 0.5
                candidates: set[int] = set()
                for band in range(bands):
                    au, av = _apply_homography_single(inv_h[frame, band], px, py)
                    tx = max(0, min(math.floor(au / tile_size), tiles_x - 1))
                    ty = max(0, min(math.floor(av / tile_size), tiles_y - 1))
                    candidates.update(
                        tile_sets[_tile_id(band=band, tx=tx, ty=ty, tz=tz, tiles_x=tiles_x, tiles_y=tiles_y, tiles_t=tiles_t)]
                    )
                total_candidate_evals += len(candidates)
                max_candidates_per_pixel = max(max_candidates_per_pixel, len(candidates))
                if not candidates:
                    continue
                ordered = sorted(candidates, key=lambda tube: float(depth[frame, tube]))
                accum = torch.zeros((3,), dtype=torch.float32)
                trans = torch.tensor(1.0, dtype=torch.float32)
                for tube in ordered:
                    lambda_uu, lambda_uv_cross, lambda_vv = lambda_uv[tube]
                    du = torch.tensor(px, dtype=torch.float32) - warped_centers[frame, tube, 0]
                    dv = torch.tensor(py, dtype=torch.float32) - warped_centers[frame, tube, 1]
                    spatial = lambda_uu * du.square() + 2.0 * lambda_uv_cross * du * dv + lambda_vv * dv.square()
                    temporal = batch.lambda_t[tube] * (t - batch.t0[tube]).square()
                    alpha = (batch.opacity[tube] * torch.exp(-0.5 * (spatial + temporal))).clamp(max=max_alpha)
                    if float(alpha) < alpha_threshold:
                        continue
                    accum = accum + trans * alpha * batch.color[tube]
                    trans = trans * (1.0 - alpha)
                image[frame, y, x] = accum

                active_dense = dense_active_by_frame[frame][y * width + x]
                missing_active_candidates += len(active_dense.difference(candidates))
    dense_evals = frames * height * width * tube_count
    return image, {
        "candidate_evals": int(total_candidate_evals),
        "dense_candidate_evals": int(dense_evals),
        "candidate_eval_ratio_vs_dense": float(total_candidate_evals) / float(max(dense_evals, 1)),
        "max_candidates_per_pixel": int(max_candidates_per_pixel),
        "missing_active_candidates": int(missing_active_candidates),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
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
    batch = _world_tubes(args.tubes, seed=args.seed, velocity_scale=args.velocity_scale)
    direct_centers, direct_depth = direct_project_world_tubes(batch, k_seq, w2c_seq, times)
    prt_centers, _, lambda_uv, _ = _compile_prt_centers(batch, k_seq, w2c_seq, times, degree=args.prt_degree)
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
    if bool(fallback_mask.any()):
        warped_centers = torch.where(fallback_mask.view(1, -1, 1), prt_centers, warped_centers)

    tile_sets, tile_stats = _build_atlas_tile_sets(
        atlas_centers,
        assignments,
        batch,
        lambda_uv,
        times,
        width=args.target_size,
        height=args.target_size,
        tile_size=args.tile_size,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        support_scale=args.support_scale,
    )
    tiled_image, render_stats = _render_atlas_tiled_cpu(
        atlas_centers,
        warped_centers,
        direct_depth,
        homographies,
        assignments,
        tile_sets,
        batch,
        lambda_uv,
        times,
        width=args.target_size,
        height=args.target_size,
        tile_size=args.tile_size,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        max_alpha=args.max_alpha,
    )
    dense_image = _render_from_centers(
        batch,
        warped_centers,
        direct_depth,
        lambda_uv,
        times,
        height=args.target_size,
        width=args.target_size,
        alpha_threshold=args.alpha_threshold,
    )
    metrics = _image_metrics(tiled_image, dense_image)
    return {
        "name": "depth_banded_homography_flow_atlas_tiled_render_probe",
        "note": (
            "CPU tiled reference for inverse-homography atlas-residual tubes. It builds atlas-coordinate "
            "tile lists, inverse-warps each screen pixel per depth band to find candidates, then shades "
            "with the same screen-space Gaussian approximation used by the dense F0h render."
        ),
        "config": {
            "seed": args.seed,
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
            "tile_size": args.tile_size,
            "tile_t": args.tile_t,
            "support_scale": args.support_scale,
        },
        "fallback_tubes": int(fallback_mask.sum().item()),
        "tile_stats": tile_stats,
        "render_stats": render_stats,
        "image_metrics_vs_dense_atlas_residual": metrics,
        "pass": metrics["psnr"] >= args.psnr_gate and render_stats["missing_active_candidates"] == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--tubes", type=int, default=128)
    parser.add_argument("--pan-x", type=float, default=0.09)
    parser.add_argument("--zoom", type=float, default=0.025)
    parser.add_argument("--dolly-z", type=float, default=0.12)
    parser.add_argument("--depth-bands", type=int, default=4)
    parser.add_argument("--residual-degree", type=int, default=3)
    parser.add_argument("--prt-degree", type=int, default=2)
    parser.add_argument("--velocity-scale", type=float, default=0.01)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--tile-t", type=int, default=4)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--fallback-max-px", type=float, default=1.0)
    parser.add_argument("--support-scale", type=float, default=1.5)
    parser.add_argument("--max-alpha", type=float, default=0.99)
    parser.add_argument("--psnr-gate", type=float, default=80.0)
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
