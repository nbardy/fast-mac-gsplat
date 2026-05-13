from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.depth_banded_homography_flow_atlas_metal_binning_probe import (  # noqa: E402
    _build_scene,
)
from research_project.benchmarks.depth_banded_homography_flow_atlas_tiled_render_probe import (  # noqa: E402
    _build_atlas_tile_sets,
    _render_atlas_tiled_cpu,
)
from research_project.benchmarks.depth_banded_homography_flow_residual_probe import (  # noqa: E402
    _image_metrics,
    _render_from_centers,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    render_inverse_homography_atlas_residual_tiles,
)


def _tile_count_summary(tile_counts: torch.Tensor, tile_overflow: torch.Tensor) -> dict[str, int]:
    counts = tile_counts.detach().cpu().to(torch.int64)
    overflow = tile_overflow.detach().cpu().to(torch.int64)
    return {
        "tile_pairs_metal": int(counts.sum().item()),
        "active_tile_count_metal": int((counts > 0).sum().item()),
        "max_tile_count_metal": int(counts.max().item()) if counts.numel() else 0,
        "overflow_tile_count": int((overflow > 0).sum().item()),
        "overflow_sum": int(overflow.sum().item()),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    scene = _build_scene(args)
    tile_sets, tile_stats = _build_atlas_tile_sets(
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
    cpu_tiled, cpu_render_stats = _render_atlas_tiled_cpu(
        scene.atlas_centers,
        scene.warped_centers,
        scene.direct_depth,
        scene.homographies,
        scene.assignments,
        tile_sets,
        scene.batch,
        scene.lambda_uv,
        scene.times,
        width=args.target_size,
        height=args.target_size,
        tile_size=args.tile_size,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        max_alpha=args.max_alpha,
    )
    dense = _render_from_centers(
        scene.batch,
        scene.warped_centers,
        scene.direct_depth,
        scene.lambda_uv,
        scene.times,
        height=args.target_size,
        width=args.target_size,
        alpha_threshold=args.alpha_threshold,
    )
    report: dict[str, Any] = {
        "name": "depth_banded_homography_flow_atlas_metal_render_probe",
        "note": (
            "F1b correctness-first Metal render parity for inverse-homography atlas-residual tubes. "
            "The render kernel scans candidate tiles per pixel in depth order; this is an image parity gate, "
            "not a final speed path."
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
        "cpu_render_stats": cpu_render_stats,
        "cpu_tiled_vs_dense": _image_metrics(cpu_tiled, dense),
        "metal_checked": False,
        "pass": False,
    }
    if not torch.backends.mps.is_available():
        report["metal_error"] = "MPS is not available"
        return report

    config = UVTRenderConfig(
        height=args.target_size,
        width=args.target_size,
        frames=args.frames,
        tile_x=args.tile_size,
        tile_y=args.tile_size,
        tile_t=args.tile_t,
        tile_capacity=args.tile_capacity,
        alpha_threshold=args.alpha_threshold,
        max_alpha=args.max_alpha,
    )
    result = render_inverse_homography_atlas_residual_tiles(
        scene.atlas_ref_uv.to("mps"),
        scene.atlas_residual_coeff.to("mps"),
        scene.homographies.to("mps"),
        torch.linalg.inv(scene.homographies).contiguous().to("mps"),
        scene.direct_depth.to("mps"),
        scene.lambda_uv.to("mps"),
        scene.batch.lambda_t.to("mps"),
        scene.batch.t0.to("mps"),
        scene.batch.opacity.to("mps"),
        scene.batch.color.to("mps"),
        scene.assignments.to(torch.int32).to("mps"),
        config,
        band_count=args.depth_bands,
        support_scale=args.support_scale,
        return_aux=True,
    )
    image = result.image.detach().cpu()
    metal_vs_cpu = _image_metrics(image, cpu_tiled)
    metal_vs_dense = _image_metrics(image, dense)
    tile_summary = _tile_count_summary(result.tile_counts, result.tile_overflow)
    report["metal_checked"] = True
    report["metal_tile_summary"] = tile_summary
    report["metal_vs_cpu_tiled"] = metal_vs_cpu
    report["metal_vs_dense"] = metal_vs_dense
    report["pass"] = bool(
        tile_summary["overflow_tile_count"] == 0
        and metal_vs_cpu["max_abs"] <= args.max_abs_gate
        and metal_vs_cpu["psnr"] >= args.psnr_gate
        and metal_vs_dense["psnr"] >= args.psnr_gate
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
    parser.add_argument("--tile-capacity", type=int, default=32)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--fallback-max-px", type=float, default=1.0)
    parser.add_argument("--support-scale", type=float, default=1.4)
    parser.add_argument("--max-alpha", type=float, default=0.99)
    parser.add_argument("--max-abs-gate", type=float, default=5.0e-5)
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
