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

from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    brute_force_render_projective_rational_tubes,
)


PARAM_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("h_coeff", (0, 0, 0)),
    ("h_coeff", (0, 1, 2)),
    ("lambda_uv", (0, 0)),
    ("lambda_t", (1,)),
    ("center_t", (0,)),
    ("opacity", (1,)),
    ("color", (0, 2)),
)


def _params(*, requires_grad: bool) -> dict[str, torch.Tensor]:
    return {
        "h_coeff": torch.tensor(
            [
                [[2.2, 2.0, 3.0], [0.10, 0.04, 0.0]],
                [[3.4, 2.6, 4.0], [-0.05, 0.02, 0.0]],
            ],
            dtype=torch.float32,
            requires_grad=requires_grad,
        ),
        "lambda_uv": torch.tensor(
            [[1.2, 0.0, 1.1], [1.0, 0.0, 1.3]],
            dtype=torch.float32,
            requires_grad=requires_grad,
        ),
        "lambda_t": torch.tensor([0.02, 0.03], dtype=torch.float32, requires_grad=requires_grad),
        "center_t": torch.tensor([0.0, 0.0], dtype=torch.float32, requires_grad=requires_grad),
        "opacity": torch.tensor([0.55, 0.45], dtype=torch.float32, requires_grad=requires_grad),
        "color": torch.tensor(
            [[0.8, 0.2, 0.1], [0.1, 0.5, 0.9]],
            dtype=torch.float32,
            requires_grad=requires_grad,
        ),
    }


def _loss(params: dict[str, torch.Tensor], config: UVTRenderConfig) -> torch.Tensor:
    image = brute_force_render_projective_rational_tubes(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        config,
    )
    grad_image = torch.linspace(-0.2, 0.3, image.numel(), dtype=torch.float32).reshape_as(image)
    return (image * grad_image).sum()


def _finite_difference(param_name: str, index: tuple[int, ...], config: UVTRenderConfig, eps: float) -> float:
    plus = _params(requires_grad=False)
    minus = _params(requires_grad=False)
    plus[param_name][index] += eps
    minus[param_name][index] -= eps
    return float((_loss(plus, config) - _loss(minus, config)) / (2.0 * eps))


def run_check(*, eps: float, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    config = UVTRenderConfig(height=5, width=6, frames=3)
    params = _params(requires_grad=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True")
        loss = _loss(params, config)
    loss.backward()

    rows = []
    for param_name, index in PARAM_SPECS:
        analytic = float(params[param_name].grad[index])
        finite_difference = _finite_difference(param_name, index, config, eps)
        abs_error = abs(analytic - finite_difference)
        rel_error = abs_error / max(abs(finite_difference), 1.0e-8)
        rows.append(
            {
                "param": param_name,
                "index": list(index),
                "analytic": analytic,
                "finite_difference": finite_difference,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "pass": abs_error <= abs_tol or rel_error <= rel_tol,
            }
        )

    return {
        "name": "projective_rational_gradient_reference_check",
        "note": "CPU autograd-vs-finite-difference target for future Metal PRT backward kernels.",
        "eps": eps,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "loss": float(loss.detach()),
        "config": {
            "height": config.height,
            "width": config.width,
            "frames": config.frames,
        },
        "max_abs_error": max(row["abs_error"] for row in rows),
        "max_rel_error": max(row["rel_error"] for row in rows),
        "pass": all(bool(row["pass"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=1.0e-3)
    parser.add_argument("--abs-tol", type=float, default=1.0e-4)
    parser.add_argument("--rel-tol", type=float, default=5.0e-2)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    summary = run_check(eps=args.eps, abs_tol=args.abs_tol, rel_tol=args.rel_tol)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
