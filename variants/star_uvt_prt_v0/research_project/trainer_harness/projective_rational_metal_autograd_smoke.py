from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    render_projective_rational_tubes_tiled,
)
from torch_gsplat_bridge_star_uvt_prt.tile_config import (  # noqa: E402
    apply_projective_rational_tile_env,
    recommend_projective_rational_tile_config,
)

try:
    from .projective_rational_metal_autograd import render_projective_rational_tubes_metal_direct_serial_backward
except ImportError:  # pragma: no cover - script execution fallback.
    from projective_rational_metal_autograd import render_projective_rational_tubes_metal_direct_serial_backward


PARAM_NAMES = ("h_coeff", "lambda_uv", "lambda_t", "center_t", "opacity", "color")


def _base_params(device: torch.device | str, *, requires_grad: bool) -> dict[str, torch.Tensor]:
    return {
        "h_coeff": torch.tensor(
            [
                [[2.2, 2.0, 3.0], [0.10, 0.04, 0.0]],
                [[3.4, 2.6, 4.0], [-0.05, 0.02, 0.0]],
            ],
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        ),
        "lambda_uv": torch.tensor(
            [[1.2, 0.0, 1.1], [1.0, 0.0, 1.3]],
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        ),
        "lambda_t": torch.tensor([0.02, 0.03], dtype=torch.float32, device=device, requires_grad=requires_grad),
        "center_t": torch.tensor([0.0, 0.0], dtype=torch.float32, device=device, requires_grad=requires_grad),
        "opacity": torch.tensor([0.55, 0.45], dtype=torch.float32, device=device, requires_grad=requires_grad),
        "color": torch.tensor(
            [[0.8, 0.2, 0.1], [0.1, 0.5, 0.9]],
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        ),
    }


def _target_params(device: torch.device | str) -> dict[str, torch.Tensor]:
    params = _base_params(device, requires_grad=False)
    params["h_coeff"] = params["h_coeff"].clone()
    params["h_coeff"][0, 0, :2] += torch.tensor([0.15, -0.08], dtype=torch.float32, device=device)
    params["opacity"] = torch.tensor([0.50, 0.62], dtype=torch.float32, device=device)
    params["color"] = torch.tensor(
        [[0.55, 0.36, 0.18], [0.16, 0.42, 0.78]],
        dtype=torch.float32,
        device=device,
    )
    return params


def _loss(params: dict[str, torch.Tensor], target: torch.Tensor, config: UVTRenderConfig, forward_mode: str) -> torch.Tensor:
    image = render_projective_rational_tubes_metal_direct_serial_backward(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        config,
        forward_mode=forward_mode,
    )
    return (image - target).square().mean()


def _norm(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return float(torch.linalg.vector_norm(tensor.detach()).cpu())


def run_smoke(*, steps: int, lr: float, forward_mode: str) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        return {
            "name": "projective_rational_metal_autograd_smoke",
            "metal_checked": False,
            "pass": False,
            "error": "MPS is required for PRT Metal autograd smoke",
        }

    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")

    tile_config = recommend_projective_rational_tile_config(tube_count=2)
    apply_projective_rational_tile_env(tile_config)
    config = UVTRenderConfig(
        height=5,
        width=6,
        frames=3,
        **tile_config.as_render_kwargs(),
    )

    with torch.no_grad():
        target_params = _target_params("mps")
        target = render_projective_rational_tubes_tiled(
            target_params["h_coeff"],
            target_params["lambda_uv"],
            target_params["lambda_t"],
            target_params["center_t"],
            target_params["opacity"],
            target_params["color"],
            config,
        ).detach()

    params = _base_params("mps", requires_grad=True)
    optimizer = torch.optim.SGD([params[name] for name in PARAM_NAMES], lr=lr)
    losses: list[float] = []
    first_grad_norms: dict[str, float] | None = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(params, target, config, forward_mode)
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        if step == 0:
            first_grad_norms = {name: _norm(params[name].grad) for name in PARAM_NAMES}
        optimizer.step()

    with torch.no_grad():
        final_loss = float(_loss(params, target, config, forward_mode).detach().cpu())
    losses.append(final_loss)
    grad_norms = {} if first_grad_norms is None else first_grad_norms
    finite_grads = all(math.isfinite(value) and value > 0.0 for value in grad_norms.values())
    loss_decreased = final_loss < losses[0]
    return {
        "name": "projective_rational_metal_autograd_smoke",
        "note": "Tiled Metal PRT train-step smoke using direct-serial Metal PRT backward.",
        "metal_checked": True,
        "pass": bool(loss_decreased and finite_grads),
        "forward_mode": forward_mode,
        "steps": steps,
        "lr": lr,
        "tile_config_key": tile_config.key,
        "tile_config": tile_config.as_dict(),
        "config": {
            "height": config.height,
            "width": config.width,
            "frames": config.frames,
        },
        "initial_loss": losses[0],
        "final_loss": final_loss,
        "loss_decreased": loss_decreased,
        "first_step_grad_norms": grad_norms,
        "losses": losses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--forward-mode", choices=("direct", "tiled"), default="tiled")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    summary = run_smoke(steps=args.steps, lr=args.lr, forward_mode=args.forward_mode)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
