from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v7 import RasterConfig, rasterize_projected_gaussians


def dense_reference(means2d, conics, colors, opacities, depths, H, W, bg=(0.0, 0.0, 0.0)):
    perm = torch.argsort(depths.detach(), dim=-1, stable=True)
    outs = []
    for b in range(means2d.shape[0]):
        m = means2d[b][perm[b]]
        q = conics[b][perm[b]]
        c = colors[b][perm[b]]
        o = opacities[b][perm[b]]
        ys = torch.arange(H, dtype=means2d.dtype, device=means2d.device) + 0.5
        xs = torch.arange(W, dtype=means2d.dtype, device=means2d.device) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        img = torch.zeros(H, W, 3, dtype=means2d.dtype, device=means2d.device)
        T = torch.ones(H, W, dtype=means2d.dtype, device=means2d.device)
        bg_t = torch.tensor(bg, dtype=means2d.dtype, device=means2d.device)
        for i in range(m.shape[0]):
            dx = xx - m[i, 0]
            dy = yy - m[i, 1]
            power = -0.5 * (q[i, 0] * dx * dx + 2 * q[i, 1] * dx * dy + q[i, 2] * dy * dy)
            alpha = torch.clamp(o[i] * torch.exp(power), max=0.99)
            alpha = torch.where((power <= 0) & (alpha >= 1.0 / 255.0), alpha, torch.zeros_like(alpha))
            w = T * alpha
            img = img + w[..., None] * c[i]
            T = T * (1.0 - alpha)
        outs.append(img + T[..., None] * bg_t)
    return torch.stack(outs, dim=0)


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit('MPS is not available.')
    device = torch.device('mps')
    torch.manual_seed(0)
    B, G, H, W = 1, 4, 16, 16
    means2d = torch.tensor([[[4.2, 5.1], [8.0, 7.5], [11.2, 8.1], [6.4, 12.0]]], device=device, dtype=torch.float32, requires_grad=True)
    conics = torch.tensor([[[0.35, 0.02, 0.42], [0.28, -0.01, 0.33], [0.25, 0.04, 0.31], [0.40, 0.00, 0.27]]], device=device, dtype=torch.float32, requires_grad=True)
    colors = torch.rand(B, G, 3, device=device, dtype=torch.float32, requires_grad=True)
    opacities = torch.tensor([[0.8, 0.6, 0.5, 0.7]], device=device, dtype=torch.float32, requires_grad=True)
    depths = torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=device, dtype=torch.float32)
    cfg = RasterConfig(height=H, width=W)
    out = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
    loss = out.square().mean(); loss.backward()
    means2d_r = means2d.detach().cpu().requires_grad_(True)
    conics_r = conics.detach().cpu().requires_grad_(True)
    colors_r = colors.detach().cpu().requires_grad_(True)
    opacities_r = opacities.detach().cpu().requires_grad_(True)
    depths_r = depths.detach().cpu()
    ref = dense_reference(means2d_r, conics_r, colors_r, opacities_r, depths_r, H, W)
    ref.square().mean().backward()
    print('image max error:', float((out.detach().cpu() - ref.detach()).abs().max()))
    print('means grad max error:', float((means2d.grad.detach().cpu() - means2d_r.grad).abs().max()))
    print('conics grad max error:', float((conics.grad.detach().cpu() - conics_r.grad).abs().max()))
    print('colors grad max error:', float((colors.grad.detach().cpu() - colors_r.grad).abs().max()))
    print('opacities grad max error:', float((opacities.grad.detach().cpu() - opacities_r.grad).abs().max()))


if __name__ == "__main__":
    main()
