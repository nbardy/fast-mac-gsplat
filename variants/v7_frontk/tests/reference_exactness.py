from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v71 import RasterConfig, rasterize_projected_gaussians
from torch_gsplat_bridge_v71.reference import dense_reference_forward, sort_projected_inputs


def dense_autograd_reference(means2d, conics, colors, opacities, depths, height, width, bg=(0.0, 0.0, 0.0), alpha_threshold=1.0 / 255.0):
    means2d_s, conics_s, colors_s, opacities_s, _, _ = sort_projected_inputs(means2d, conics, colors, opacities, depths)
    return dense_reference_forward(means2d_s, conics_s, colors_s, opacities_s, height, width, alpha_threshold, bg)


def run_case(case_name: str, front_k: int, means2d: torch.Tensor, conics: torch.Tensor, colors: torch.Tensor, opacities: torch.Tensor, depths: torch.Tensor, H: int, W: int) -> None:
    cfg = RasterConfig(height=H, width=W, front_k=front_k, use_reference_when_unavailable=True, return_debug_state=True)
    means = means2d.clone().requires_grad_(True)
    conic = conics.clone().requires_grad_(True)
    color = colors.clone().requires_grad_(True)
    opacity = opacities.clone().requires_grad_(True)
    out, debug = rasterize_projected_gaussians(means, conic, color, opacity, depths, cfg)
    loss = out.square().mean()
    loss.backward()

    means_r = means2d.clone().requires_grad_(True)
    conic_r = conics.clone().requires_grad_(True)
    color_r = colors.clone().requires_grad_(True)
    opacity_r = opacities.clone().requires_grad_(True)
    ref = dense_autograd_reference(means_r, conic_r, color_r, opacity_r, depths, H, W)
    ref.square().mean().backward()

    print(f"[{case_name}] front_k={front_k}")
    print("  image max error:     ", float((out.detach() - ref.detach()).abs().max()))
    print("  means grad max error:", float((means.grad - means_r.grad).abs().max()))
    print("  conics grad max error:", float((conic.grad - conic_r.grad).abs().max()))
    print("  colors grad max error:", float((color.grad - color_r.grad).abs().max()))
    print("  opacity grad max error:", float((opacity.grad - opacity_r.grad).abs().max()))
    print("  overflow ratio:      ", float(debug.overflow_mask.float().mean()))
    assert torch.allclose(out.detach(), ref.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(means.grad, means_r.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(conic.grad, conic_r.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(color.grad, color_r.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(opacity.grad, opacity_r.grad, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Mixed random case.
    B, G, H, W = 1, 4, 6, 5
    means2d = torch.rand(B, G, 2, dtype=torch.float32) * torch.tensor([W, H], dtype=torch.float32)
    conics = torch.rand(B, G, 3, dtype=torch.float32)
    conics[..., 0] += 0.2
    conics[..., 2] += 0.2
    conics[..., 1] *= 0.05
    colors = torch.rand(B, G, 3, dtype=torch.float32)
    opacities = torch.rand(B, G, dtype=torch.float32) * 0.9 + 0.05
    depths = torch.rand(B, G, dtype=torch.float32)

    run_case("random", 1, means2d, conics, colors, opacities, depths, H, W)
    run_case("random", 2, means2d, conics, colors, opacities, depths, H, W)
    run_case("random", 4, means2d, conics, colors, opacities, depths, H, W)

    # Clamp-heavy overflow case.
    means2d = torch.tensor([[[1.2, 1.5], [2.1, 2.3], [1.8, 2.0], [0.9, 0.8], [3.2, 3.0]]], dtype=torch.float32)
    conics = torch.tensor([[[0.6, 0.0, 0.6], [0.4, 0.1, 0.5], [0.7, -0.05, 0.8], [1.2, 0.0, 1.0], [0.3, 0.0, 0.3]]], dtype=torch.float32)
    colors = torch.rand(1, 5, 3, dtype=torch.float32)
    opacities = torch.tensor([[1.5, 0.02, 0.7, 0.999, 0.3]], dtype=torch.float32)
    depths = torch.rand(1, 5, dtype=torch.float32)

    run_case("clamp_overflow", 1, means2d, conics, colors, opacities, depths, 4, 4)
    run_case("clamp_overflow", 3, means2d, conics, colors, opacities, depths, 4, 4)
