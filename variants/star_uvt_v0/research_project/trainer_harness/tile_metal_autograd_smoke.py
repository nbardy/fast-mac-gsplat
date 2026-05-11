from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import make_gate0_scene  # noqa: E402

try:
    from .tile_metal_autograd import render_uvt_tubes_metal_tile_backward
except ImportError:  # pragma: no cover - script execution fallback.
    from tile_metal_autograd import render_uvt_tubes_metal_tile_backward


def _norm(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return float(torch.linalg.vector_norm(tensor.detach()).cpu())


def main() -> None:
    if not torch.backends.mps.is_available():
        print(json.dumps({"metal_skipped": "MPS is not available"}, indent=2, sort_keys=True))
        return
    ma, q_uvt, depth0, depth_beta, opacity, color, config = make_gate0_scene("crossing_depth", device="mps")
    ma = ma.detach().requires_grad_(True)
    q_uvt = q_uvt.detach().requires_grad_(True)
    depth0 = depth0.detach().requires_grad_(True)
    depth_beta = depth_beta.detach().requires_grad_(True)
    opacity = opacity.detach().requires_grad_(True)
    color = color.detach().requires_grad_(True)
    image = render_uvt_tubes_metal_tile_backward(ma, q_uvt, depth0, depth_beta, opacity, color, config)
    loss = image.square().mean()
    loss.backward()
    grad_norms = {
        "ma": _norm(ma.grad),
        "q_uvt": _norm(q_uvt.grad),
        "depth0": _norm(depth0.grad),
        "depth_beta": _norm(depth_beta.grad),
        "opacity": _norm(opacity.grad),
        "color": _norm(color.grad),
    }
    for name in ("ma", "q_uvt", "opacity", "color"):
        value = grad_norms[name]
        if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
            raise AssertionError(f"expected finite non-zero gradient for {name}, got {value}")
    print(
        json.dumps(
            {
                "scene": "crossing_depth",
                "device": "mps",
                "loss": float(loss.detach().cpu()),
                "grad_norms": grad_norms,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
