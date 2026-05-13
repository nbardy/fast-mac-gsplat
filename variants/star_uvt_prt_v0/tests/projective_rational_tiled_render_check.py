from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_projection_audit import _batch, _constant_k, _w2c_translation_z  # noqa: E402
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    centered_frame_times,
    compile_projective_rational_tubes,
    dense_render_projective_rational_tubes,
    fit_camera_path_polynomial,
)
from torch_gsplat_bridge_star_uvt_prt import render_projective_rational_tubes_tiled  # noqa: E402
from torch_gsplat_bridge_star_uvt_prt import UVTRenderConfig  # noqa: E402


def _scene(*, tile_t: int) -> tuple:
    frames = 4
    times = centered_frame_times(frames)
    batch = _batch()
    camera_path = fit_camera_path_polynomial(
        _constant_k(frames, fx=55.0, fy=55.0, cx=16.0, cy=12.0),
        _w2c_translation_z(times, slope=0.08),
        degree=1,
        frame_times=times,
    )
    projected = compile_projective_rational_tubes(batch, camera_path)
    config = UVTRenderConfig(height=24, width=32, frames=frames, tile_t=tile_t, tile_capacity=128)
    return projected, times, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-t", type=int, default=2)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    projected, times, config = _scene(tile_t=args.tile_t)
    dense = dense_render_projective_rational_tubes(
        projected,
        height=config.height,
        width=config.width,
        frame_times=times,
        alpha_threshold=config.alpha_threshold,
        background=config.background,
    )
    summary = {
        "metal_checked": False,
        "tiled_max_abs_error": None,
        "active_tile_count": None,
        "overflow_tile_count": None,
        "max_tile_count": None,
        "pass": False,
    }
    if torch.backends.mps.is_available():
        try:
            result = render_projective_rational_tubes_tiled(
                projected.h_coeff.to("mps"),
                projected.lambda_uv.to("mps"),
                projected.lambda_t.to("mps"),
                projected.center_t.to("mps"),
                projected.opacity.to("mps"),
                projected.color.to("mps"),
                config,
                return_aux=True,
            )
            image = result.image.cpu()
            tile_counts = result.tile_counts.cpu()
            tile_overflow = result.tile_overflow.cpu()
            summary.update(
                {
                    "metal_checked": True,
                    "tiled_max_abs_error": float((image - dense).abs().max().detach().cpu()),
                    "active_tile_count": int((tile_counts > 0).sum().item()),
                    "overflow_tile_count": int((tile_overflow > 0).sum().item()),
                    "max_tile_count": int(tile_counts.max().item()),
                }
            )
        except (RuntimeError, AttributeError) as exc:
            summary["metal_error"] = str(exc)
    summary["pass"] = bool(
        summary["metal_checked"]
        and float(summary["tiled_max_abs_error"]) <= 5.0e-5
        and int(summary["active_tile_count"]) > 0
        and int(summary["overflow_tile_count"]) == 0
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise AssertionError("projective rational tiled render check failed")


if __name__ == "__main__":
    main()
