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

from research_project.benchmarks.projective_rational_direct_serial_backward_check import PARAM_NAMES  # noqa: E402
from research_project.benchmarks.projective_rational_gradient_reference_check import _params  # noqa: E402
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    projective_rational_tile_pixel_atomic_backward,
    projective_rational_tile_pixel_fused_mse_backward,
    projective_rational_tile_pixel_fused_mse_train_used_backward,
    render_projective_rational_tubes_tiled,
)


def _target(config: UVTRenderConfig, device: torch.device | str) -> torch.Tensor:
    count = config.frames * config.height * config.width * 3
    return torch.linspace(0.05, 0.95, count, dtype=torch.float32, device=device).reshape(
        config.frames,
        config.height,
        config.width,
        3,
    )


def _reference(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    config: UVTRenderConfig,
) -> tuple[float, dict[str, torch.Tensor]]:
    image = render_projective_rational_tubes_tiled(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        config,
    )
    loss = (image - target).square().mean()
    grad_image = 2.0 * (image - target) / float(image.numel())
    grads = projective_rational_tile_pixel_atomic_backward(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        grad_image,
        config,
    )[:6]
    return float(loss.detach().cpu()), {name: grad.detach().cpu() for name, grad in zip(PARAM_NAMES, grads, strict=True)}


def _fused(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    config: UVTRenderConfig,
    fused_mode: str,
) -> tuple[float, dict[str, torch.Tensor], int]:
    op = (
        projective_rational_tile_pixel_fused_mse_train_used_backward
        if fused_mode == "train_used"
        else projective_rational_tile_pixel_fused_mse_backward
    )
    result = op(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        target,
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
    loss = result.loss_sum.detach().cpu()[0] / float(config.frames * config.height * config.width * 3)
    overflow_tile_count = int((result.tile_overflow.detach().cpu() > 0).sum().item())
    return float(loss), {name: grad.detach().cpu() for name, grad in zip(PARAM_NAMES, grads, strict=True)}, overflow_tile_count


def run_check(*, abs_tol: float, rel_tol: float, loss_tol: float, fused_mode: str) -> dict[str, Any]:
    if fused_mode not in {"full", "train_used"}:
        raise ValueError("fused_mode must be one of: full, train_used")
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
            "name": "projective_rational_tile_pixel_fused_mse_backward_check",
            "metal_checked": False,
            "pass": False,
            "error": "MPS is required for fused MSE PRT backward check",
        }

    params = {name: value.detach().to("mps") for name, value in _params(requires_grad=False).items()}
    target = _target(config, "mps")
    reference_loss, reference_grads = _reference(params, target, config)
    fused_loss, fused_grads, overflow_tile_count = _fused(params, target, config, fused_mode)
    checked_names = PARAM_NAMES if fused_mode == "full" else ("h_coeff", "lambda_t", "opacity", "color")
    rows = []
    for name in checked_names:
        diff = (fused_grads[name] - reference_grads[name]).abs()
        ref_abs = reference_grads[name].abs()
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
    loss_abs_error = abs(fused_loss - reference_loss)
    return {
        "name": "projective_rational_tile_pixel_fused_mse_backward_check",
        "note": "Research-only fused MSE PRT backward against tiled forward plus tile-pixel backward.",
        "fused_mode": fused_mode,
        "metal_checked": True,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "loss_tol": loss_tol,
        "config": {
            "height": config.height,
            "width": config.width,
            "frames": config.frames,
            "tile_x": config.tile_x,
            "tile_y": config.tile_y,
            "tile_t": config.tile_t,
            "tile_capacity": config.tile_capacity,
        },
        "reference_loss": reference_loss,
        "fused_loss": fused_loss,
        "loss_abs_error": loss_abs_error,
        "overflow_tile_count": overflow_tile_count,
        "checked_params": list(checked_names),
        "skipped_params": [name for name in PARAM_NAMES if name not in checked_names],
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "max_rel_error": max(row["max_rel_error"] for row in rows),
        "pass": loss_abs_error <= loss_tol
        and overflow_tile_count == 0
        and all(bool(row["pass"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abs-tol", type=float, default=5.0e-4)
    parser.add_argument("--rel-tol", type=float, default=5.0e-2)
    parser.add_argument("--loss-tol", type=float, default=1.0e-5)
    parser.add_argument("--fused-mode", choices=("full", "train_used"), default="full")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    summary = run_check(abs_tol=args.abs_tol, rel_tol=args.rel_tol, loss_tol=args.loss_tol, fused_mode=args.fused_mode)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
