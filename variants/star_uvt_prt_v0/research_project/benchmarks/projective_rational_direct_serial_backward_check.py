from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_gradient_reference_check import _loss, _params  # noqa: E402
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    projective_rational_direct_serial_backward,
)


PARAM_NAMES = ("h_coeff", "lambda_uv", "lambda_t", "center_t", "opacity", "color")


def _grad_image(config: UVTRenderConfig, device: torch.device | str) -> torch.Tensor:
    count = config.frames * config.height * config.width * 3
    return torch.linspace(-0.2, 0.3, count, dtype=torch.float32, device=device).reshape(
        config.frames,
        config.height,
        config.width,
        3,
    )


def _reference_gradients(config: UVTRenderConfig) -> dict[str, torch.Tensor]:
    params = _params(requires_grad=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True")
        loss = _loss(params, config)
    loss.backward()
    return {name: params[name].grad.detach().cpu() for name in PARAM_NAMES}


def _metal_gradients(config: UVTRenderConfig) -> dict[str, torch.Tensor]:
    params = {name: value.detach().to("mps") for name, value in _params(requires_grad=False).items()}
    grads = projective_rational_direct_serial_backward(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        _grad_image(config, "mps"),
        config,
    )
    return {name: grad.detach().cpu() for name, grad in zip(PARAM_NAMES, grads, strict=True)}


def run_check(*, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    config = UVTRenderConfig(height=5, width=6, frames=3)
    if not torch.backends.mps.is_available():
        return {
            "name": "projective_rational_direct_serial_backward_check",
            "metal_checked": False,
            "pass": False,
            "error": "MPS is required for direct serial PRT backward check",
        }

    reference = _reference_gradients(config)
    metal = _metal_gradients(config)
    rows = []
    for name in PARAM_NAMES:
        diff = (metal[name] - reference[name]).abs()
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
    return {
        "name": "projective_rational_direct_serial_backward_check",
        "note": "Slow direct serial Metal PRT backward against the CPU C0 gradient target.",
        "metal_checked": True,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "config": {
            "height": config.height,
            "width": config.width,
            "frames": config.frames,
        },
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "max_rel_error": max(row["max_rel_error"] for row in rows),
        "pass": all(bool(row["pass"]) for row in rows),
        "rows": rows,
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
