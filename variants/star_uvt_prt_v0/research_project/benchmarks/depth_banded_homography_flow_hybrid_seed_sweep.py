from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.depth_banded_homography_flow_residual_probe import run_probe  # noqa: E402


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("seeds must not be empty")
    return seeds


def _probe_args(args: argparse.Namespace, *, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        target_size=args.target_size,
        frames=args.frames,
        tubes=args.tubes,
        seed=seed,
        pan_x=args.pan_x,
        zoom=args.zoom,
        dolly_z=args.dolly_z,
        depth_bands=args.depth_bands,
        residual_degree=args.residual_degree,
        segments=args.segments,
        prt_degree=args.prt_degree,
        velocity_scale=args.velocity_scale,
        tile_x=args.tile_x,
        tile_y=args.tile_y,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        p95_ratio_gate=args.p95_ratio_gate,
        fallback_max_px=args.fallback_max_px,
        out_json=None,
    )


def _row(report: dict[str, Any]) -> dict[str, Any]:
    pure = report["homography_flow_residual"]
    hybrid = report["hybrid_gauge_with_prt_fallback"]
    segmented = report["segmented_f4"]
    return {
        "seed": report["config"]["seed"],
        "pure_pass": report["pass"],
        "hybrid_pass": hybrid["pass"],
        "pure_p95_px": pure["center_metrics_vs_direct"]["p95_px"],
        "pure_max_px": pure["center_metrics_vs_direct"]["max_px"],
        "pure_psnr": pure["image_metrics_vs_direct"]["psnr"],
        "fallback_tubes": hybrid["fallback_tubes"],
        "fallback_fraction": pure["fallback_outliers"]["fraction"],
        "hybrid_p95_px": hybrid["center_metrics_vs_direct"]["p95_px"],
        "hybrid_max_px": hybrid["center_metrics_vs_direct"]["max_px"],
        "hybrid_psnr": hybrid["image_metrics_vs_direct"]["psnr"],
        "hybrid_tile_pairs": hybrid["tile_pair_estimate"]["total_tile_pairs"],
        "segmented_p95_px": segmented["center_metrics_vs_direct"]["p95_px"],
        "segmented_psnr": segmented["image_metrics_vs_direct"]["psnr"],
        "segmented_tile_pairs": segmented["tile_pair_estimate"]["total_tile_pairs"],
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    seeds = _parse_seeds(args.seeds)
    rows = [_row(run_probe(_probe_args(args, seed=seed))) for seed in seeds]
    fallback_counts = [float(row["fallback_tubes"]) for row in rows]
    fallback_fractions = [float(row["fallback_fraction"]) for row in rows]
    hybrid_psnrs = [float(row["hybrid_psnr"]) for row in rows]
    hybrid_p95 = [float(row["hybrid_p95_px"]) for row in rows]
    hybrid_max = [float(row["hybrid_max_px"]) for row in rows]
    hybrid_tile_ratios = [
        float(row["hybrid_tile_pairs"]) / max(float(row["segmented_tile_pairs"]), 1.0)
        for row in rows
    ]
    return {
        "name": "depth_banded_homography_flow_hybrid_seed_sweep",
        "note": (
            "Multi-seed robustness sweep for the F0c hybrid gauge-residual plus PRT fallback upper bound. "
            "This is still projection/render-only and reports tile-pair estimates, not actual flow-sheared render time."
        ),
        "config": {
            "seeds": seeds,
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
        },
        "rows": rows,
        "summary": {
            "pure_pass_count": sum(1 for row in rows if row["pure_pass"]),
            "hybrid_pass_count": sum(1 for row in rows if row["hybrid_pass"]),
            "fallback_tubes": _stats(fallback_counts),
            "fallback_fraction": _stats(fallback_fractions),
            "hybrid_p95_px": _stats(hybrid_p95),
            "hybrid_max_px": _stats(hybrid_max),
            "hybrid_psnr": _stats(hybrid_psnrs),
            "hybrid_tile_pair_ratio_vs_segmented_f4": _stats(hybrid_tile_ratios),
        },
        "pass": all(bool(row["hybrid_pass"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="17,23,31,47")
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
    parser.add_argument("--tile-x", type=int, default=8)
    parser.add_argument("--tile-y", type=int, default=8)
    parser.add_argument("--tile-t", type=int, default=4)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--p95-ratio-gate", type=float, default=0.75)
    parser.add_argument("--fallback-max-px", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = run_sweep(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
