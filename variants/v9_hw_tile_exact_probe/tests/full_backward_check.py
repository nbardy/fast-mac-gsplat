from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v9_hw_tile_exact import (  # noqa: E402
    make_full_backward_config,
    probe_full_backward,
    rasterize_projected_gaussians_full_backward,
)


def dense_reference(means2d, conics, colors, opacities, depths, height, width, bg=(0.0, 0.0, 0.0)):
    if means2d.ndim == 2:
        means2d = means2d.unsqueeze(0)
        conics = conics.unsqueeze(0)
        colors = colors.unsqueeze(0)
        opacities = opacities.unsqueeze(0)
        depths = depths.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    outs = []
    bg_t = torch.tensor(bg, dtype=means2d.dtype, device=means2d.device)
    ys = torch.arange(height, dtype=means2d.dtype, device=means2d.device) + 0.5
    xs = torch.arange(width, dtype=means2d.dtype, device=means2d.device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    for b in range(means2d.shape[0]):
        perm = torch.argsort(depths[b].detach(), stable=True)
        m = means2d[b][perm]
        q = conics[b][perm]
        c = colors[b][perm]
        o = opacities[b][perm]
        out = torch.zeros(height, width, 3, dtype=means2d.dtype, device=means2d.device)
        trans = torch.ones(height, width, dtype=means2d.dtype, device=means2d.device)
        for i in range(m.shape[0]):
            dx = xx - m[i, 0]
            dy = yy - m[i, 1]
            power = -0.5 * (q[i, 0] * dx * dx + 2 * q[i, 1] * dx * dy + q[i, 2] * dy * dy)
            alpha = torch.clamp(o[i] * torch.exp(power), max=0.99)
            alpha = torch.where((power <= 0) & (alpha >= 1.0 / 255.0), alpha, torch.zeros_like(alpha))
            out = out + (trans * alpha)[..., None] * c[i]
            trans = trans * (1.0 - alpha)
        outs.append(out + trans[..., None] * bg_t)
    out = torch.stack(outs, dim=0)
    return out[0] if squeeze else out


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, threshold: float) -> None:
    err = float((got - ref).detach().abs().max().item())
    print(f"{name} max error: {err}")
    if err > threshold:
        raise AssertionError(f"{name} max error {err} exceeded {threshold}")


def main() -> None:
    status = probe_full_backward()
    print(json.dumps(status.as_dict(), indent=2, sort_keys=True))
    assert status.available, status.error
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    device = torch.device("mps")
    height, width = 16, 16
    means2d = torch.tensor(
        [[4.2, 5.1], [8.0, 7.5], [11.2, 8.1], [6.4, 12.0]],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    conics = torch.tensor(
        [[0.35, 0.02, 0.42], [0.28, -0.01, 0.33], [0.25, 0.04, 0.31], [0.40, 0.00, 0.27]],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    colors = torch.rand(4, 3, device=device, dtype=torch.float32, requires_grad=True)
    opacities = torch.tensor([0.8, 0.6, 0.5, 0.7], device=device, dtype=torch.float32, requires_grad=True)
    depths = torch.tensor([0.1, 0.2, 0.3, 0.4], device=device, dtype=torch.float32)

    cfg = make_full_backward_config(
        height=height,
        width=width,
        tile_size=16,
        max_fast_pairs=128,
        stop_count_mode="adaptive",
    )
    out = rasterize_projected_gaussians_full_backward(means2d, conics, colors, opacities, depths, cfg)
    out.square().mean().backward()

    means_r = means2d.detach().cpu().requires_grad_(True)
    conics_r = conics.detach().cpu().requires_grad_(True)
    colors_r = colors.detach().cpu().requires_grad_(True)
    opacities_r = opacities.detach().cpu().requires_grad_(True)
    depths_r = depths.detach().cpu()
    ref = dense_reference(means_r, conics_r, colors_r, opacities_r, depths_r, height, width)
    ref.square().mean().backward()

    assert_close("image", out.detach().cpu(), ref.detach(), 1.0e-5)
    assert_close("means grad", means2d.grad.detach().cpu(), means_r.grad.detach(), 1.0e-5)
    assert_close("conics grad", conics.grad.detach().cpu(), conics_r.grad.detach(), 1.0e-5)
    assert_close("colors grad", colors.grad.detach().cpu(), colors_r.grad.detach(), 1.0e-5)
    assert_close("opacities grad", opacities.grad.detach().cpu(), opacities_r.grad.detach(), 1.0e-5)


if __name__ == "__main__":
    main()
