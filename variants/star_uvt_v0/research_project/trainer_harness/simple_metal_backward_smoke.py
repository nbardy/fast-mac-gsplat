from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import make_gate0_scene, simple_backward_samples  # noqa: E402

try:
    from .model import dense_differentiable_render_uvt_tubes
except ImportError:  # pragma: no cover - script execution fallback.
    from model import dense_differentiable_render_uvt_tubes


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().cpu() - right.detach().cpu()).abs().max().item())


def main() -> None:
    if not torch.backends.mps.is_available():
        print(json.dumps({"metal_skipped": "MPS is not available"}, indent=2, sort_keys=True))
        return
    ma, q_uvt, depth0, depth_beta, opacity, color, config = make_gate0_scene("moving_diagonal", device="mps")
    ma_ref = ma.detach().requires_grad_(True)
    q_ref = q_uvt.detach().requires_grad_(True)
    opacity_ref = opacity.detach().requires_grad_(True)
    color_ref = color.detach().requires_grad_(True)
    image_ref = dense_differentiable_render_uvt_tubes(ma_ref, q_ref, depth0.detach(), depth_beta.detach(), opacity_ref, color_ref, config)
    grad_image = torch.linspace(0.1, 0.9, image_ref.numel(), dtype=torch.float32, device="mps").view_as(image_ref).contiguous()
    torch.sum(image_ref * grad_image).backward()

    grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples = simple_backward_samples(
        ma.detach(),
        q_uvt.detach(),
        opacity.detach(),
        color.detach(),
        grad_image,
        config,
    )
    grad_ma = grad_ma_samples.sum(dim=0).view_as(ma)
    grad_q = grad_q_samples.sum(dim=0).view_as(q_uvt)
    grad_opacity = grad_opacity_samples.sum(dim=0).view_as(opacity)
    grad_color = grad_color_samples.sum(dim=0).view_as(color)

    errors = {
        "ma": _max_abs(grad_ma, ma_ref.grad),
        "q_uvt": _max_abs(grad_q, q_ref.grad),
        "opacity": _max_abs(grad_opacity, opacity_ref.grad),
        "color": _max_abs(grad_color, color_ref.grad),
    }
    for name, value in errors.items():
        if value > 5.0e-4:
            raise AssertionError(f"{name} simple Metal backward mismatch: {value}")
    print(
        json.dumps(
            {
                "scene": "moving_diagonal",
                "device": "mps",
                "max_abs_errors": errors,
                "sample_count": int(grad_opacity_samples.shape[0]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
