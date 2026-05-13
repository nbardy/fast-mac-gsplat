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

from research_project.benchmarks.depth_banded_homography_flow_atlas_tiled_render_probe import run_probe  # noqa: E402


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("seeds must not be empty")
    return seeds


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _probe_args(args: argparse.Namespace, *, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        target_size=args.target_size,
        frames=args.frames,
        tubes=args.tubes,
        pan_x=args.pan_x,
        zoom=args.zoom,
        dolly_z=args.dolly_z,
        depth_bands=args.depth_bands,
        residual_degree=args.residual_degree,
        prt_degree=args.prt_degree,
        velocity_scale=args.velocity_scale,
        tile_size=args.tile_size,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        fallback_max_px=args.fallback_max_px,
        support_scale=args.support_scale,
        max_alpha=args.max_alpha,
        psnr_gate=args.psnr_gate,
        out_json=None,
    )


def _row(report: dict[str, Any]) -> dict[str, Any]:
    render_stats = report["render_stats"]
    image_metrics = report["image_metrics_vs_dense_atlas_residual"]
    return {
        "seed": report["config"]["seed"],
        "pass": report["pass"],
        "fallback_tubes": report["fallback_tubes"],
        "psnr": image_metrics["psnr"],
        "max_abs": image_metrics["max_abs"],
        "candidate_eval_ratio_vs_dense": render_stats["candidate_eval_ratio_vs_dense"],
        "candidate_evals": render_stats["candidate_evals"],
        "missing_active_candidates": render_stats["missing_active_candidates"],
        "max_candidates_per_pixel": render_stats["max_candidates_per_pixel"],
        "tile_pairs": report["tile_stats"]["tile_pairs"],
    }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    rows = [_row(run_probe(_probe_args(args, seed=seed))) for seed in _parse_seeds(args.seeds)]
    return {
        "name": "depth_banded_homography_flow_atlas_tiled_seed_sweep",
        "note": (
            "Multi-seed CPU tiled render coverage check for inverse-homography atlas-residual tubes. "
            "This is a correctness reference for the planned Metal path, not a runtime benchmark."
        ),
        "config": {
            "seeds": [row["seed"] for row in rows],
            "target_size": args.target_size,
            "frames": args.frames,
            "tubes": args.tubes,
            "pan_x": args.pan_x,
            "zoom": args.zoom,
            "dolly_z": args.dolly_z,
            "velocity_scale": args.velocity_scale,
            "depth_bands": args.depth_bands,
            "residual_degree": args.residual_degree,
            "tile_size": args.tile_size,
            "tile_t": args.tile_t,
            "support_scale": args.support_scale,
        },
        "rows": rows,
        "summary": {
            "pass_count": sum(1 for row in rows if row["pass"]),
            "fallback_tubes": _stats([float(row["fallback_tubes"]) for row in rows]),
            "psnr": _stats([float(row["psnr"]) for row in rows]),
            "candidate_eval_ratio_vs_dense": _stats(
                [float(row["candidate_eval_ratio_vs_dense"]) for row in rows]
            ),
            "max_candidates_per_pixel": _stats([float(row["max_candidates_per_pixel"]) for row in rows]),
            "tile_pairs": _stats([float(row["tile_pairs"]) for row in rows]),
            "missing_active_candidates": _stats([float(row["missing_active_candidates"]) for row in rows]),
        },
        "pass": all(bool(row["pass"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="17,23,31,47")
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

    report = run_sweep(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
