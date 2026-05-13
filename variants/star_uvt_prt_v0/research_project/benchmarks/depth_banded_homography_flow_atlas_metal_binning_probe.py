from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
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
from research_project.benchmarks.depth_banded_homography_flow_atlas_tiled_render_probe import (  # noqa: E402
    _build_atlas_tile_sets,
)
from research_project.benchmarks.depth_banded_homography_flow_residual_probe import (  # noqa: E402
    _camera,
    _compile_prt_centers,
    _depth_bands,
    _fit_poly,
    _per_tube_max_error,
    _world_tubes,
)
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    centered_frame_times,
    direct_project_world_tubes,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    bin_inverse_homography_atlas_residual_tiles,
)


@dataclass(frozen=True)
class AtlasBinningScene:
    times: Tensor
    batch: Any
    lambda_uv: Tensor
    assignments: Tensor
    homographies: Tensor
    atlas_ref_uv: Tensor
    atlas_residual_coeff: Tensor
    atlas_centers: Tensor
    warped_centers: Tensor
    direct_depth: Tensor
    fallback_mask: Tensor


def _build_scene(args: argparse.Namespace) -> AtlasBinningScene:
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
    atlas_ref_uv = direct_centers[ref_frame]
    band_depths, assignments = _depth_bands(direct_depth[ref_frame], bands=args.depth_bands)
    homographies = _homography_matrices(band_depths, k_seq, w2c_seq, ref_frame=ref_frame)
    atlas_targets = _inverse_homography_atlas_targets(direct_centers, homographies, assignments)
    atlas_residual_targets = atlas_targets - atlas_ref_uv.view(1, -1, 2)
    residual_coeff, atlas_residual_recon = _fit_poly(atlas_residual_targets, times, degree=args.residual_degree)
    atlas_centers = atlas_ref_uv.view(1, -1, 2) + atlas_residual_recon
    warped_centers = _forward_homography_centers(atlas_centers, homographies, assignments)
    fallback_mask = _per_tube_max_error(warped_centers, direct_centers) > args.fallback_max_px
    if bool(fallback_mask.any()):
        warped_centers = torch.where(fallback_mask.view(1, -1, 1), prt_centers, warped_centers)
    return AtlasBinningScene(
        times=times,
        batch=batch,
        lambda_uv=lambda_uv,
        assignments=assignments,
        homographies=homographies,
        atlas_ref_uv=atlas_ref_uv,
        atlas_residual_coeff=residual_coeff.permute(1, 0, 2).contiguous(),
        atlas_centers=atlas_centers,
        warped_centers=warped_centers,
        direct_depth=direct_depth,
        fallback_mask=fallback_mask,
    )


def _metal_sets(
    tile_counts: Tensor,
    tile_tube_ids: Tensor,
    *,
    tile_capacity: int,
    tube_count: int,
) -> tuple[list[set[int]], dict[str, int]]:
    counts = tile_counts.detach().cpu().to(torch.int64)
    ids = tile_tube_ids.detach().cpu().to(torch.int64).view(int(counts.numel()), tile_capacity)
    tile_sets: list[set[int]] = []
    duplicate_ids = 0
    out_of_range_ids = 0
    for tile_id, raw_count in enumerate(counts.tolist()):
        limit = min(int(raw_count), tile_capacity)
        values = [int(v) for v in ids[tile_id, :limit].tolist()]
        valid = [v for v in values if 0 <= v < tube_count]
        out_of_range_ids += len(values) - len(valid)
        unique = set(valid)
        duplicate_ids += len(valid) - len(unique)
        tile_sets.append(unique)
    return tile_sets, {
        "duplicate_metal_ids": int(duplicate_ids),
        "out_of_range_metal_ids": int(out_of_range_ids),
    }


def _compare_tile_sets(
    cpu_sets: list[set[int]],
    metal_sets: list[set[int]],
    tile_counts: Tensor,
    tile_overflow: Tensor,
    metal_id_stats: dict[str, int],
) -> dict[str, int | float]:
    counts = tile_counts.detach().cpu().to(torch.int64)
    overflow = tile_overflow.detach().cpu().to(torch.int64)
    cpu_counts = torch.tensor([len(tile_set) for tile_set in cpu_sets], dtype=torch.int64)
    missing_pairs = 0
    extra_pairs = 0
    set_mismatches = 0
    for cpu, metal in zip(cpu_sets, metal_sets, strict=True):
        missing = cpu.difference(metal)
        extra = metal.difference(cpu)
        missing_pairs += len(missing)
        extra_pairs += len(extra)
        set_mismatches += int(bool(missing or extra))
    return {
        "tile_count": int(counts.numel()),
        "tile_pairs_cpu": int(cpu_counts.sum().item()),
        "tile_pairs_metal": int(counts.sum().item()),
        "active_tile_count_cpu": int((cpu_counts > 0).sum().item()),
        "active_tile_count_metal": int((counts > 0).sum().item()),
        "max_tile_count_cpu": int(cpu_counts.max().item()) if cpu_counts.numel() else 0,
        "max_tile_count_metal": int(counts.max().item()) if counts.numel() else 0,
        "per_tile_count_mismatches": int((cpu_counts != counts).sum().item()),
        "per_tile_set_mismatches": int(set_mismatches),
        "missing_pairs": int(missing_pairs),
        "extra_pairs": int(extra_pairs),
        "overflow_tile_count": int((overflow > 0).sum().item()),
        "overflow_sum": int(overflow.sum().item()),
        **metal_id_stats,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    scene = _build_scene(args)
    cpu_sets, tile_stats = _build_atlas_tile_sets(
        scene.atlas_centers,
        scene.assignments,
        scene.batch,
        scene.lambda_uv,
        scene.times,
        width=args.target_size,
        height=args.target_size,
        tile_size=args.tile_size,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        support_scale=args.support_scale,
    )
    config = UVTRenderConfig(
        height=args.target_size,
        width=args.target_size,
        frames=args.frames,
        tile_x=args.tile_size,
        tile_y=args.tile_size,
        tile_t=args.tile_t,
        tile_capacity=args.tile_capacity,
        alpha_threshold=args.alpha_threshold,
    )
    report: dict[str, Any] = {
        "name": "depth_banded_homography_flow_atlas_metal_binning_probe",
        "note": (
            "F1a bin-only Metal parity check for inverse-homography atlas-residual tubes. "
            "The gate compares per-tile tube-id sets because atomic insertion order is not stable."
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
            "tile_capacity": args.tile_capacity,
            "support_scale": args.support_scale,
        },
        "fallback_tubes": int(scene.fallback_mask.sum().item()),
        "cpu_tile_stats": tile_stats,
        "metal_checked": False,
        "pass": False,
    }
    if not torch.backends.mps.is_available():
        report["metal_error"] = "MPS is not available"
        return report
    result = bin_inverse_homography_atlas_residual_tiles(
        scene.atlas_ref_uv.to("mps"),
        scene.atlas_residual_coeff.to("mps"),
        scene.lambda_uv.to("mps"),
        scene.batch.lambda_t.to("mps"),
        scene.batch.t0.to("mps"),
        scene.batch.opacity.to("mps"),
        scene.assignments.to(torch.int32).to("mps"),
        config,
        band_count=args.depth_bands,
        support_scale=args.support_scale,
    )
    metal_sets, metal_id_stats = _metal_sets(
        result.tile_counts,
        result.tile_tube_ids,
        tile_capacity=args.tile_capacity,
        tube_count=args.tubes,
    )
    comparison = _compare_tile_sets(
        cpu_sets,
        metal_sets,
        result.tile_counts,
        result.tile_overflow,
        metal_id_stats,
    )
    report["metal_checked"] = True
    report["comparison"] = comparison
    report["pass"] = bool(
        comparison["overflow_tile_count"] == 0
        and comparison["max_tile_count_metal"] <= args.tile_capacity
        and comparison["tile_pairs_cpu"] == comparison["tile_pairs_metal"]
        and comparison["active_tile_count_cpu"] == comparison["active_tile_count_metal"]
        and comparison["max_tile_count_cpu"] == comparison["max_tile_count_metal"]
        and comparison["per_tile_count_mismatches"] == 0
        and comparison["missing_pairs"] == 0
        and comparison["extra_pairs"] == 0
        and comparison["duplicate_metal_ids"] == 0
        and comparison["out_of_range_metal_ids"] == 0
    )
    return report


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
    parser.add_argument("--tile-capacity", type=int, default=512)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--fallback-max-px", type=float, default=1.0)
    parser.add_argument("--support-scale", type=float, default=1.4)
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
