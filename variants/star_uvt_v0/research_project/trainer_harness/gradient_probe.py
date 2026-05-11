from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .model import render_model
    from .train import make_scene_fit_target
except ImportError:  # pragma: no cover - script execution fallback.
    from model import render_model
    from train import make_scene_fit_target


def _grad_norm(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(parameter.grad.detach()).cpu())


def main() -> None:
    target, model, _config = make_scene_fit_target(
        "moving_diagonal",
        device=torch.device("cpu"),
        seed=5,
        jitter_pixels=0.65,
    )
    image = render_model(model)
    loss = torch.mean((image - target).square())
    loss.backward()
    norms = {
        "center_uv": _grad_norm(model.center_uv),
        "center_t": _grad_norm(model.center_t),
        "velocity_uv": _grad_norm(model.velocity_uv),
        "raw_precision": _grad_norm(model.raw_precision),
        "raw_opacity": _grad_norm(model.raw_opacity),
        "raw_color": _grad_norm(model.raw_color),
    }
    required = ("center_uv", "velocity_uv", "raw_precision", "raw_opacity", "raw_color")
    for name in required:
        value = norms[name]
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise AssertionError(f"expected finite non-zero gradient for {name}, got {value}")
    row = {
        "scene": "moving_diagonal",
        "device": "cpu",
        "loss": float(loss.detach().cpu()),
        "grad_norms": norms,
    }
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
