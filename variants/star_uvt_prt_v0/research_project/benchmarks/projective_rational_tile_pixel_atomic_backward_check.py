from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_tile_pair_atomic_backward_check import (  # noqa: E402
    PARAM_NAMES,
    _metal_gradients,
)
from research_project.benchmarks.projective_rational_direct_serial_backward_check import _grad_image  # noqa: E402
from research_project.benchmarks.projective_rational_gradient_reference_check import _params  # noqa: E402
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    profile_projective_rational_tile_pixel_atomic_backward,
)


def _profile_gradients(config: UVTRenderConfig) -> tuple[dict[str, torch.Tensor], dict[str, float], int]:
    params = {name: value.detach().to("mps") for name, value in _params(requires_grad=False).items()}
    result = profile_projective_rational_tile_pixel_atomic_backward(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        _grad_image(config, "mps"),
        config,
    )
    grads = (
        result.grad_h_coeff,
        result.grad_lambda_uv,
        result.grad_lambda_t,
        result.grad_center_t,
        result.grad_opacity,
        result.grad_color,
    )
    return (
        {name: grad.detach().cpu() for name, grad in zip(PARAM_NAMES, grads, strict=True)},
        result.timings_ms,
        int((result.tile_overflow.detach().cpu() > 0).sum().item()),
    )


def run_check(*, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    config = UVTRenderConfig(
        height=5,
        width=6,
        frames=3,
        tile_x=int(os.environ.get("STAR_UVT_TILE_X", "8")),
        tile_y=int(os.environ.get("STAR_UVT_TILE_Y", "8")),
        tile_t=int(os.environ.get("STAR_UVT_TILE_T", "2")),
        tile_capacity=int(os.environ.get("STAR_UVT_TILE_CAPACITY", "128")),
    )
    if not torch.backends.mps.is_available():
        return {
            "name": "projective_rational_tile_pixel_atomic_backward_check",
            "metal_checked": False,
            "pass": False,
            "error": "MPS is required for tile-pixel atomic PRT backward check",
        }

    reference, _ = _metal_gradients(config, "direct_serial")
    candidate, tile_unstable = _metal_gradients(config, "tile_pixel_atomic")
    profiled, profile_timings_ms, profile_overflow_tile_count = _profile_gradients(config)
    rows = []
    profile_rows = []
    for name in PARAM_NAMES:
        diff = (candidate[name] - reference[name]).abs()
        ref_abs = reference[name].abs()
        rel = diff / torch.clamp(ref_abs, min=1.0e-8)
        max_abs = float(diff.max().item())
        max_rel = float(rel.max().item())
        rows.append(
            {
                "param": name,
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "pass": max_abs <= abs_tol or max_rel <= rel_tol,
            }
        )
        profile_diff = (profiled[name] - candidate[name]).abs()
        profile_ref_abs = candidate[name].abs()
        profile_rel = profile_diff / torch.clamp(profile_ref_abs, min=1.0e-8)
        profile_max_abs = float(profile_diff.max().item())
        profile_max_rel = float(profile_rel.max().item())
        profile_rows.append(
            {
                "param": name,
                "max_abs_error": profile_max_abs,
                "max_rel_error": profile_max_rel,
                "pass": profile_max_abs <= abs_tol or profile_max_rel <= rel_tol,
            }
        )
    tile_unstable_count = 0 if tile_unstable is None else int(tile_unstable.sum().item())
    return {
        "name": "projective_rational_tile_pixel_atomic_backward_check",
        "note": "Tile-pixel atomic Metal PRT backward against direct-serial Metal PRT backward.",
        "metal_checked": True,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "config": {
            "height": config.height,
            "width": config.width,
            "frames": config.frames,
            "tile_x": config.tile_x,
            "tile_y": config.tile_y,
            "tile_t": config.tile_t,
            "tile_capacity": config.tile_capacity,
        },
        "tile_unstable_count": tile_unstable_count,
        "profile_overflow_tile_count": profile_overflow_tile_count,
        "profile_timings_ms": profile_timings_ms,
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "max_rel_error": max(row["max_rel_error"] for row in rows),
        "profile_max_abs_error": max(row["max_abs_error"] for row in profile_rows),
        "profile_max_rel_error": max(row["max_rel_error"] for row in profile_rows),
        "pass": all(bool(row["pass"]) for row in rows)
        and all(bool(row["pass"]) for row in profile_rows)
        and profile_overflow_tile_count == 0,
        "rows": rows,
        "profile_rows": profile_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abs-tol", type=float, default=5.0e-4)
    parser.add_argument("--rel-tol", type=float, default=5.0e-2)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    summary = run_check(abs_tol=args.abs_tol, rel_tol=args.rel_tol)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
