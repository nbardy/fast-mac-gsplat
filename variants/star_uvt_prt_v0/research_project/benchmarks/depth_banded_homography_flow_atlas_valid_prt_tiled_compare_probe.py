from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
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
    ProjectiveRationalTileConfig,
    UVTRenderConfig,
    parse_projective_rational_tile_config,
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


def _speedup(base_ms: float, candidate_ms: float) -> float:
    return float(base_ms) / max(float(candidate_ms), 1.0e-9)


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


def _scene_camera(args: argparse.Namespace, scene: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return _camera(
        args.frames,
        args.target_size,
        args.target_size,
        scene.times,
        pan_x=args.pan_x,
        zoom=args.zoom,
        dolly_z=args.dolly_z,
    )


def _direct_dense_fn(args: argparse.Namespace, scene: Any, lambda_uv_mps: torch.Tensor) -> Callable[[], torch.Tensor]:
    batch_mps = _batch_to_mps(scene.batch)
    k_seq, w2c_seq = _scene_camera(args, scene)
    k_seq_mps = k_seq.to("mps")
    w2c_seq_mps = w2c_seq.to("mps")
    times_mps = scene.times.to("mps")

    def render() -> torch.Tensor:
        return dense_render_direct_world_tubes(
            batch_mps,
            k_seq_mps,
            w2c_seq_mps,
            times_mps,
            lambda_uv_mps,
            height=args.target_size,
            width=args.target_size,
            alpha_threshold=args.alpha_threshold,
            max_alpha=args.max_alpha,
        )

    return render


def _render_atlas_child(args: argparse.Namespace, tile_config: ProjectiveRationalTileConfig) -> dict[str, Any]:
    if tile_config.tile_capacity * args.depth_bands > 128:
        raise ValueError("cached atlas child requires depth_bands * tile_capacity <= 128")
    scene = _build_scene(args)
    config = UVTRenderConfig(
        height=args.target_size,
        width=args.target_size,
        frames=args.frames,
        **tile_config.as_render_kwargs(),
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

    def render_scan(*, return_aux: bool = False) -> Any:
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
            return_aux=return_aux,
        )

    def render_cached(*, return_aux: bool = False) -> Any:
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
            return_aux=return_aux,
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
    direct_dense_fn = _direct_dense_fn(args, scene, inputs["lambda_uv"])
    scan_aux = render_scan(return_aux=True)
    cached_aux = render_cached(return_aux=True)
    direct_dense = direct_dense_fn()
    torch.mps.synchronize()
    scan_image = scan_aux.image.detach().cpu()
    cached_image = cached_aux.image.detach().cpu()
    direct_dense_image = direct_dense.detach().cpu()
    cached_vs_scan = _image_metrics(cached_image, scan_image)
    cached_vs_direct = _image_metrics(cached_image, direct_dense_image)
    tile_summary = _tile_summary(cached_aux)
    scan_timing = _time_mps(lambda: render_scan(return_aux=False), warmup=args.warmup, iters=args.iters)
    cached_timing = _time_mps(lambda: render_cached(return_aux=False), warmup=args.warmup, iters=args.iters)
    direct_dense_timing = _time_mps(direct_dense_fn, warmup=args.warmup, iters=args.iters)
    return {
        "mode": "atlas_cached",
        "tile_config": tile_config.as_dict(),
        "fallback_tubes": int(scene.fallback_mask.sum().item()),
        "tile_summary": tile_summary,
        "metrics": {
            "cached_atlas_vs_scan_atlas": cached_vs_scan,
            "cached_atlas_vs_dense_atlas_reference": _image_metrics(cached_image, dense_atlas_reference),
            "cached_atlas_vs_direct_dense_reference": cached_vs_direct,
            "scan_atlas_vs_direct_dense_reference": _image_metrics(scan_image, direct_dense_image),
        },
        "timing_ms": {
            "atlas_scan": scan_timing,
            "atlas_cached": cached_timing,
            "direct_dense_reference": direct_dense_timing,
        },
        "pass": bool(
            tile_summary["overflow_tile_count"] == 0
            and cached_vs_scan["max_abs"] <= args.max_abs_gate
            and cached_vs_direct["psnr"] >= args.psnr_gate
        ),
    }


def _render_prt_child(args: argparse.Namespace, tile_config: ProjectiveRationalTileConfig) -> dict[str, Any]:
    scene = _build_scene(args)
    config = UVTRenderConfig(
        height=args.target_size,
        width=args.target_size,
        frames=args.frames,
        **tile_config.as_render_kwargs(),
        alpha_threshold=args.alpha_threshold,
        max_alpha=args.max_alpha,
    )
    k_seq, w2c_seq = _scene_camera(args, scene)
    camera_path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=args.prt_degree, frame_times=scene.times)
    prt = compile_projective_rational_tubes(scene.batch, camera_path)
    inputs = {
        "h_coeff": prt.h_coeff.to("mps"),
        "lambda_uv": prt.lambda_uv.to("mps"),
        "lambda_t": prt.lambda_t.to("mps"),
        "center_t": prt.center_t.to("mps"),
        "opacity": prt.opacity.to("mps"),
        "color": prt.color.to("mps"),
    }
    direct_dense_fn = _direct_dense_fn(args, scene, inputs["lambda_uv"])

    def render_direct() -> torch.Tensor:
        return render_projective_rational_tubes_direct(
            inputs["h_coeff"],
            inputs["lambda_uv"],
            inputs["lambda_t"],
            inputs["center_t"],
            inputs["opacity"],
            inputs["color"],
            config,
        )

    def render_tiled(*, return_aux: bool = False) -> Any:
        return render_projective_rational_tubes_tiled(
            inputs["h_coeff"],
            inputs["lambda_uv"],
            inputs["lambda_t"],
            inputs["center_t"],
            inputs["opacity"],
            inputs["color"],
            config,
            return_aux=return_aux,
        )

    direct = render_direct()
    tiled_aux = render_tiled(return_aux=True)
    direct_dense = direct_dense_fn()
    torch.mps.synchronize()
    direct_image = direct.detach().cpu()
    tiled_image = tiled_aux.image.detach().cpu()
    direct_dense_image = direct_dense.detach().cpu()
    tiled_vs_direct = _image_metrics(tiled_image, direct_image)
    tiled_vs_dense = _image_metrics(tiled_image, direct_dense_image)
    tile_summary = _tile_summary(tiled_aux)
    direct_timing = _time_mps(render_direct, warmup=args.warmup, iters=args.iters)
    tiled_timing = _time_mps(lambda: render_tiled(return_aux=True), warmup=args.warmup, iters=args.iters)
    direct_dense_timing = _time_mps(direct_dense_fn, warmup=args.warmup, iters=args.iters)
    matches_direct = tiled_vs_direct["max_abs"] <= args.max_abs_gate
    matches_dense = tiled_vs_dense["max_abs"] <= args.max_abs_gate
    return {
        "mode": "prt_tiled",
        "tile_config": tile_config.as_dict(),
        "fallback_tubes": int(scene.fallback_mask.sum().item()),
        "camera_fit_error": camera_path.fit_error,
        "tile_summary": tile_summary,
        "metrics": {
            "prt_tiled_vs_prt_direct": tiled_vs_direct,
            "prt_direct_vs_direct_dense_reference": _image_metrics(direct_image, direct_dense_image),
            "prt_tiled_vs_direct_dense_reference": tiled_vs_dense,
        },
        "timing_ms": {
            "prt_direct": direct_timing,
            "prt_tiled": tiled_timing,
            "direct_dense_reference": direct_dense_timing,
        },
        "validity": {
            "matches_prt_direct": bool(matches_direct),
            "matches_direct_dense_reference": bool(matches_dense),
        },
        "pass": bool(tile_summary["overflow_tile_count"] == 0 and (matches_direct or matches_dense)),
    }


def _child_args(args: argparse.Namespace, *, mode: str, tile_config: ProjectiveRationalTileConfig) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-mode",
        mode,
        "--seed",
        str(args.seed),
        "--target-size",
        str(args.target_size),
        "--frames",
        str(args.frames),
        "--tubes",
        str(args.tubes),
        "--pan-x",
        str(args.pan_x),
        "--zoom",
        str(args.zoom),
        "--dolly-z",
        str(args.dolly_z),
        "--depth-bands",
        str(args.depth_bands),
        "--residual-degree",
        str(args.residual_degree),
        "--prt-degree",
        str(args.prt_degree),
        "--velocity-scale",
        str(args.velocity_scale),
        "--tile-config",
        tile_config.key,
        "--alpha-threshold",
        str(args.alpha_threshold),
        "--fallback-max-px",
        str(args.fallback_max_px),
        "--support-scale",
        str(args.support_scale),
        "--max-alpha",
        str(args.max_alpha),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--max-abs-gate",
        str(args.max_abs_gate),
        "--psnr-gate",
        str(args.psnr_gate),
    ]


def _run_child(args: argparse.Namespace, *, mode: str, tile_config: ProjectiveRationalTileConfig) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(tile_config.as_env())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}{os.pathsep}{existing}"
    result = subprocess.run(
        _child_args(args, mode=mode, tile_config=tile_config),
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{mode} child failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        ) from exc
    report["child_returncode"] = int(result.returncode)
    report["child_stderr"] = result.stderr.strip()
    return report


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the valid PRT tiled comparison probe")
    atlas_tile_config = parse_projective_rational_tile_config(args.atlas_tile_config)
    prt_tile_config = parse_projective_rational_tile_config(args.prt_tile_config)
    atlas = _run_child(args, mode="atlas_cached", tile_config=atlas_tile_config)
    prt = _run_child(args, mode="prt_tiled", tile_config=prt_tile_config)
    cached_ms = float(atlas["timing_ms"]["atlas_cached"]["median_ms"])
    scan_ms = float(atlas["timing_ms"]["atlas_scan"]["median_ms"])
    dense_ms = float(atlas["timing_ms"]["direct_dense_reference"]["median_ms"])
    prt_direct_ms = float(prt["timing_ms"]["prt_direct"]["median_ms"])
    prt_tiled_ms = float(prt["timing_ms"]["prt_tiled"]["median_ms"])
    speedups = {
        "cached_vs_scan": _speedup(scan_ms, cached_ms),
        "cached_vs_direct_dense_reference": _speedup(dense_ms, cached_ms),
        "cached_vs_valid_prt_direct": _speedup(prt_direct_ms, cached_ms),
        "cached_vs_valid_prt_tiled": _speedup(prt_tiled_ms, cached_ms),
        "valid_prt_tiled_vs_valid_prt_direct": _speedup(prt_direct_ms, prt_tiled_ms),
    }
    return {
        "name": "depth_banded_homography_flow_atlas_valid_prt_tiled_compare_probe",
        "note": (
            "F1e cross-process comparison. Cached atlas runs at the cap32 candidate-cache limit; "
            "PRT tiled runs in a separate process at a valid higher-capacity tile config."
        ),
        "config": {
            "seed": args.seed,
            "target_size": args.target_size,
            "frames": args.frames,
            "tubes": args.tubes,
            "depth_bands": args.depth_bands,
            "atlas_tile_config": atlas_tile_config.as_dict(),
            "prt_tile_config": prt_tile_config.as_dict(),
            "support_scale": args.support_scale,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "atlas_cached": atlas,
        "valid_prt_tiled": prt,
        "speedups": speedups,
        "speed_read": {
            "cached_beats_scan": bool(speedups["cached_vs_scan"] > 1.0),
            "cached_beats_direct_dense_reference": bool(speedups["cached_vs_direct_dense_reference"] > 1.0),
            "cached_beats_valid_prt_direct": bool(speedups["cached_vs_valid_prt_direct"] > 1.0),
            "cached_beats_valid_prt_tiled": bool(speedups["cached_vs_valid_prt_tiled"] > 1.0),
        },
        "pass": bool(atlas["pass"] and prt["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-mode", choices=("atlas_cached", "prt_tiled"))
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
    parser.add_argument("--tile-config", default="4x4x4:32")
    parser.add_argument("--atlas-tile-config", default="4x4x4:32")
    parser.add_argument("--prt-tile-config", default="4x4x4:128")
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

    tile_config = parse_projective_rational_tile_config(args.tile_config)
    if args.child_mode == "atlas_cached":
        report = _render_atlas_child(args, tile_config)
    elif args.child_mode == "prt_tiled":
        report = _render_prt_child(args, tile_config)
    else:
        report = run_probe(args)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
