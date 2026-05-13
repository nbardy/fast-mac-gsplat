from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.depth_banded_homography_flow_atlas_metal_binning_probe import (  # noqa: E402
    _build_scene,
)
from research_project.benchmarks.depth_banded_homography_flow_residual_probe import (  # noqa: E402
    _camera,
    _image_metrics,
    _render_from_centers,
)
from research_project.benchmarks.projective_rational_world_camera_forward_probe import (  # noqa: E402
    dense_render_direct_world_tubes,
)
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    WorldTubeBatch,
    compile_projective_rational_tubes,
    fit_camera_path_polynomial,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    render_inverse_homography_atlas_residual_tiles,
    render_inverse_homography_atlas_residual_tiles_cached,
    render_projective_rational_tubes_direct,
    render_projective_rational_tubes_tiled,
)


def _time_mps(fn: Callable[[], Any], *, warmup: int, iters: int) -> dict[str, Any]:
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


def _tile_summary(result: Any) -> dict[str, int]:
    counts = result.tile_counts.detach().cpu().to(torch.int64)
    overflow = result.tile_overflow.detach().cpu().to(torch.int64)
    return {
        "tile_pairs": int(counts.sum().item()),
        "active_tile_count": int((counts > 0).sum().item()),
        "max_tile_count": int(counts.max().item()) if counts.numel() else 0,
        "overflow_tile_count": int((overflow > 0).sum().item()),
        "overflow_sum": int(overflow.sum().item()),
    }


def _speedup(base: dict[str, Any], candidate: dict[str, Any]) -> float:
    return float(base["median_ms"]) / max(float(candidate["median_ms"]), 1.0e-9)


def _batch_to_mps(batch: WorldTubeBatch) -> WorldTubeBatch:
    return WorldTubeBatch(
        x0=batch.x0.to("mps"),
        velocity=batch.velocity.to("mps"),
        t0=batch.t0.to("mps"),
        precision_xy=batch.precision_xy.to("mps"),
        lambda_t=batch.lambda_t.to("mps"),
        opacity=batch.opacity.to("mps"),
        color=batch.color.to("mps"),
    )


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the atlas render compare probe")

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
    atlas_inputs = {
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

    k_seq, w2c_seq = _camera(
        args.frames,
        args.target_size,
        args.target_size,
        scene.times,
        pan_x=args.pan_x,
        zoom=args.zoom,
        dolly_z=args.dolly_z,
    )
    batch_mps = _batch_to_mps(scene.batch)
    k_seq_mps = k_seq.to("mps")
    w2c_seq_mps = w2c_seq.to("mps")
    times_mps = scene.times.to("mps")
    camera_path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=args.prt_degree, frame_times=scene.times)
    prt = compile_projective_rational_tubes(scene.batch, camera_path)
    prt_inputs = {
        "h_coeff": prt.h_coeff.to("mps"),
        "lambda_uv": prt.lambda_uv.to("mps"),
        "lambda_t": prt.lambda_t.to("mps"),
        "center_t": prt.center_t.to("mps"),
        "opacity": prt.opacity.to("mps"),
        "color": prt.color.to("mps"),
    }

    def render_scan(*, return_aux: bool = False) -> Any:
        return render_inverse_homography_atlas_residual_tiles(
            atlas_inputs["atlas_ref_uv"],
            atlas_inputs["atlas_residual_coeff"],
            atlas_inputs["homographies"],
            atlas_inputs["inv_homographies"],
            atlas_inputs["depth"],
            atlas_inputs["lambda_uv"],
            atlas_inputs["lambda_t"],
            atlas_inputs["center_t"],
            atlas_inputs["opacity"],
            atlas_inputs["color"],
            atlas_inputs["band_ids"],
            config,
            band_count=args.depth_bands,
            support_scale=args.support_scale,
            return_aux=return_aux,
        )

    def render_cached(*, return_aux: bool = False) -> Any:
        return render_inverse_homography_atlas_residual_tiles_cached(
            atlas_inputs["atlas_ref_uv"],
            atlas_inputs["atlas_residual_coeff"],
            atlas_inputs["homographies"],
            atlas_inputs["inv_homographies"],
            atlas_inputs["depth"],
            atlas_inputs["lambda_uv"],
            atlas_inputs["lambda_t"],
            atlas_inputs["center_t"],
            atlas_inputs["opacity"],
            atlas_inputs["color"],
            atlas_inputs["band_ids"],
            config,
            band_count=args.depth_bands,
            support_scale=args.support_scale,
            return_aux=return_aux,
        )

    def render_prt_direct() -> torch.Tensor:
        return render_projective_rational_tubes_direct(
            prt_inputs["h_coeff"],
            prt_inputs["lambda_uv"],
            prt_inputs["lambda_t"],
            prt_inputs["center_t"],
            prt_inputs["opacity"],
            prt_inputs["color"],
            config,
        )

    def render_prt_tiled(*, return_aux: bool = False) -> Any:
        return render_projective_rational_tubes_tiled(
            prt_inputs["h_coeff"],
            prt_inputs["lambda_uv"],
            prt_inputs["lambda_t"],
            prt_inputs["center_t"],
            prt_inputs["opacity"],
            prt_inputs["color"],
            config,
            return_aux=return_aux,
        )

    def render_direct_dense() -> torch.Tensor:
        return dense_render_direct_world_tubes(
            batch_mps,
            k_seq_mps,
            w2c_seq_mps,
            times_mps,
            atlas_inputs["lambda_uv"],
            height=args.target_size,
            width=args.target_size,
            alpha_threshold=args.alpha_threshold,
            max_alpha=args.max_alpha,
        )

    dense_atlas_reference = _render_from_centers(
        scene.batch,
        scene.warped_centers,
        scene.direct_depth,
        scene.lambda_uv,
        scene.times,
        height=args.target_size,
        width=args.target_size,
        alpha_threshold=args.alpha_threshold,
    )
    scan_aux = render_scan(return_aux=True)
    cached_aux = render_cached(return_aux=True)
    direct_dense = render_direct_dense()
    prt_direct = render_prt_direct()
    prt_tiled_aux = render_prt_tiled(return_aux=True)
    torch.mps.synchronize()

    scan_image = scan_aux.image.detach().cpu()
    cached_image = cached_aux.image.detach().cpu()
    direct_dense_image = direct_dense.detach().cpu()
    prt_direct_image = prt_direct.detach().cpu()
    prt_tiled_image = prt_tiled_aux.image.detach().cpu()

    scan_timing = _time_mps(lambda: render_scan(return_aux=False), warmup=args.warmup, iters=args.iters)
    cached_timing = _time_mps(lambda: render_cached(return_aux=False), warmup=args.warmup, iters=args.iters)
    direct_dense_timing = _time_mps(render_direct_dense, warmup=args.warmup, iters=args.iters)
    prt_direct_timing = _time_mps(render_prt_direct, warmup=args.warmup, iters=args.iters)
    prt_tiled_timing = _time_mps(lambda: render_prt_tiled(return_aux=True), warmup=args.warmup, iters=args.iters)

    atlas_tile_summary = _tile_summary(cached_aux)
    prt_tile_summary = _tile_summary(prt_tiled_aux)
    cached_vs_scan = _image_metrics(cached_image, scan_image)
    cached_vs_dense_atlas = _image_metrics(cached_image, dense_atlas_reference)
    cached_vs_direct_dense = _image_metrics(cached_image, direct_dense_image)
    prt_direct_vs_direct_dense = _image_metrics(prt_direct_image, direct_dense_image)
    prt_tiled_vs_direct = _image_metrics(prt_tiled_image, prt_direct_image)
    cached_path_valid = bool(
        atlas_tile_summary["overflow_tile_count"] == 0
        and cached_vs_scan["max_abs"] <= args.max_abs_gate
        and cached_vs_direct_dense["psnr"] >= args.psnr_gate
    )
    prt_tiled_valid = bool(
        prt_tile_summary["overflow_tile_count"] == 0
        and prt_tiled_vs_direct["max_abs"] <= args.max_abs_gate
    )
    speedups = {
        "cached_vs_scan": _speedup(scan_timing, cached_timing),
        "cached_vs_direct_dense_reference": _speedup(direct_dense_timing, cached_timing),
        "cached_vs_prt_direct": _speedup(prt_direct_timing, cached_timing),
        "cached_vs_prt_tiled": _speedup(prt_tiled_timing, cached_timing),
        "prt_tiled_vs_prt_direct": _speedup(prt_direct_timing, prt_tiled_timing),
    }
    return {
        "name": "depth_banded_homography_flow_atlas_render_compare_probe",
        "note": (
            "F1d same-scene render comparison for inverse-homography atlas-residual cached render. "
            "This is a forward-render speed/validity row, not a training-quality or production direct-splat gate."
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
            "prt_camera_fit_error": camera_path.fit_error,
        },
        "fallback_tubes": int(scene.fallback_mask.sum().item()),
        "metrics": {
            "cached_atlas_vs_scan_atlas": cached_vs_scan,
            "cached_atlas_vs_dense_atlas_reference": cached_vs_dense_atlas,
            "cached_atlas_vs_direct_dense_reference": cached_vs_direct_dense,
            "scan_atlas_vs_dense_atlas_reference": _image_metrics(scan_image, dense_atlas_reference),
            "scan_atlas_vs_direct_dense_reference": _image_metrics(scan_image, direct_dense_image),
            "prt_direct_vs_direct_dense_reference": prt_direct_vs_direct_dense,
            "prt_tiled_vs_prt_direct": prt_tiled_vs_direct,
            "prt_tiled_vs_direct_dense_reference": _image_metrics(prt_tiled_image, direct_dense_image),
        },
        "tile_summaries": {
            "atlas_cached": atlas_tile_summary,
            "prt_tiled": prt_tile_summary,
        },
        "validity": {
            "cached_path_valid": cached_path_valid,
            "prt_tiled_valid_under_cached_cap": prt_tiled_valid,
            "cached_candidate_capacity": args.depth_bands * args.tile_capacity,
        },
        "timing_ms": {
            "atlas_scan": scan_timing,
            "atlas_cached": cached_timing,
            "direct_dense_reference": direct_dense_timing,
            "prt_direct": prt_direct_timing,
            "prt_tiled": prt_tiled_timing,
        },
        "speedups": speedups,
        "speed_read": {
            "cached_beats_scan": bool(speedups["cached_vs_scan"] > 1.0),
            "cached_beats_direct_dense_reference": bool(speedups["cached_vs_direct_dense_reference"] > 1.0),
            "cached_beats_prt_direct": bool(speedups["cached_vs_prt_direct"] > 1.0),
            "cached_beats_prt_tiled_under_cached_cap": bool(prt_tiled_valid and speedups["cached_vs_prt_tiled"] > 1.0),
        },
        "pass": cached_path_valid,
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
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
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
