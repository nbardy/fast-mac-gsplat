from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.depth_banded_homography_flow_atlas_metal_binning_probe import (  # noqa: E402
    _build_scene,
)
from research_project.benchmarks.depth_banded_homography_flow_residual_probe import _image_metrics  # noqa: E402
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    render_inverse_homography_atlas_residual_tiles,
    render_inverse_homography_atlas_residual_tiles_cached,
    render_inverse_homography_atlas_residual_tiles_cached_select,
)


def _time_mps(fn: Callable[[], Tensor], *, warmup: int, iters: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _speedup(base: dict[str, Any], candidate: dict[str, Any]) -> float:
    return float(base["median_ms"]) / max(float(candidate["median_ms"]), 1.0e-9)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the atlas cached-select timing probe")

    scene = _build_scene(args)
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
    inputs = {
        "atlas_ref_uv": scene.atlas_ref_uv.to("mps"),
        "atlas_residual_coeff": scene.atlas_residual_coeff.to("mps"),
        "homographies": scene.homographies.to("mps"),
        "inv_homographies": torch.linalg.inv(scene.homographies).contiguous().to("mps"),
        "depth": scene.direct_depth.to("mps"),
        "lambda_uv": scene.lambda_uv.to("mps"),
        "lambda_t": scene.batch.lambda_t.to("mps"),
        "center_t": scene.batch.t0.to("mps"),
        "opacity": scene.batch.opacity.to("mps"),
        "color": scene.batch.color.to("mps"),
        "band_ids": scene.assignments.to(torch.int32).to("mps"),
    }

    def render_scan() -> Tensor:
        return render_inverse_homography_atlas_residual_tiles(
            inputs["atlas_ref_uv"],
            inputs["atlas_residual_coeff"],
            inputs["homographies"],
            inputs["inv_homographies"],
            inputs["depth"],
            inputs["lambda_uv"],
            inputs["lambda_t"],
            inputs["center_t"],
            inputs["opacity"],
            inputs["color"],
            inputs["band_ids"],
            config,
            band_count=args.depth_bands,
            support_scale=args.support_scale,
        )

    def render_cached() -> Tensor:
        return render_inverse_homography_atlas_residual_tiles_cached(
            inputs["atlas_ref_uv"],
            inputs["atlas_residual_coeff"],
            inputs["homographies"],
            inputs["inv_homographies"],
            inputs["depth"],
            inputs["lambda_uv"],
            inputs["lambda_t"],
            inputs["center_t"],
            inputs["opacity"],
            inputs["color"],
            inputs["band_ids"],
            config,
            band_count=args.depth_bands,
            support_scale=args.support_scale,
        )

    def render_select() -> Tensor:
        return render_inverse_homography_atlas_residual_tiles_cached_select(
            inputs["atlas_ref_uv"],
            inputs["atlas_residual_coeff"],
            inputs["homographies"],
            inputs["inv_homographies"],
            inputs["depth"],
            inputs["lambda_uv"],
            inputs["lambda_t"],
            inputs["center_t"],
            inputs["opacity"],
            inputs["color"],
            inputs["band_ids"],
            config,
            band_count=args.depth_bands,
            support_scale=args.support_scale,
        )

    scan_image = render_scan()
    cached_image = render_cached()
    select_image = render_select()
    torch.mps.synchronize()
    cached_vs_scan = _image_metrics(cached_image.detach().cpu(), scan_image.detach().cpu())
    select_vs_scan = _image_metrics(select_image.detach().cpu(), scan_image.detach().cpu())
    select_vs_cached = _image_metrics(select_image.detach().cpu(), cached_image.detach().cpu())
    scan = _time_mps(render_scan, warmup=args.warmup, iters=args.iters)
    cached = _time_mps(render_cached, warmup=args.warmup, iters=args.iters)
    select = _time_mps(render_select, warmup=args.warmup, iters=args.iters)
    speedups = {
        "cached_vs_scan": _speedup(scan, cached),
        "select_vs_scan": _speedup(scan, select),
        "select_vs_cached": _speedup(cached, select),
    }
    return {
        "name": "depth_banded_homography_flow_atlas_cached_select_timing_probe",
        "note": (
            "F1i timing probe for cached atlas candidate ordering. The select variant caches candidate "
            "ids/depths unsorted and repeatedly selects the next depth during compositing, avoiding "
            "insertion-sort writes while preserving exact depth order."
        ),
        "config": {
            "seed": args.seed,
            "target_size": args.target_size,
            "frames": args.frames,
            "tubes": args.tubes,
            "depth_bands": args.depth_bands,
            "tile_size": args.tile_size,
            "tile_t": args.tile_t,
            "tile_capacity": args.tile_capacity,
            "support_scale": args.support_scale,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "fallback_tubes": int(scene.fallback_mask.sum().item()),
        "cached_vs_scan": cached_vs_scan,
        "select_vs_scan": select_vs_scan,
        "select_vs_cached": select_vs_cached,
        "scan_render": scan,
        "cached_render": cached,
        "select_render": select,
        "speedups": speedups,
        "speed_read": {
            "select_beats_scan": bool(speedups["select_vs_scan"] > 1.0),
            "select_beats_cached": bool(speedups["select_vs_cached"] > 1.0),
        },
        "pass": bool(
            cached_vs_scan["max_abs"] <= args.max_abs_gate
            and select_vs_scan["max_abs"] <= args.max_abs_gate
            and select_vs_cached["max_abs"] <= args.max_abs_gate
            and speedups["select_vs_cached"] >= args.min_select_speedup_gate
        ),
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
    parser.add_argument("--tile-capacity", type=int, default=32)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--fallback-max-px", type=float, default=1.0)
    parser.add_argument("--support-scale", type=float, default=1.4)
    parser.add_argument("--max-alpha", type=float, default=0.99)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-abs-gate", type=float, default=5.0e-5)
    parser.add_argument("--min-select-speedup-gate", type=float, default=1.0)
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
