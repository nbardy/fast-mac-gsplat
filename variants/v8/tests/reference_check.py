from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v8 import RasterConfig, rasterize_projected_gaussians


def dense_reference(means2d, conics, colors, opacities, depths, H, W, bg=(0.0, 0.0, 0.0)):
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
    for b in range(means2d.shape[0]):
        perm = torch.argsort(depths[b].detach(), stable=True)
        m = means2d[b][perm]
        q = conics[b][perm]
        c = colors[b][perm]
        o = opacities[b][perm]
        ys = torch.arange(H, dtype=means2d.dtype, device=means2d.device) + 0.5
        xs = torch.arange(W, dtype=means2d.dtype, device=means2d.device) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        out = torch.zeros(H, W, 3, dtype=means2d.dtype, device=means2d.device)
        T = torch.ones(H, W, dtype=means2d.dtype, device=means2d.device)
        for i in range(m.shape[0]):
            dx = xx - m[i, 0]
            dy = yy - m[i, 1]
            power = -0.5 * (q[i, 0] * dx * dx + 2 * q[i, 1] * dx * dy + q[i, 2] * dy * dy)
            alpha = torch.clamp(o[i] * torch.exp(power), max=0.99)
            alpha = torch.where((power <= 0) & (alpha >= 1.0 / 255.0), alpha, torch.zeros_like(alpha))
            w = T * alpha
            out = out + w[..., None] * c[i]
            T = T * (1.0 - alpha)
        outs.append(out + T[..., None] * bg_t)
    out = torch.stack(outs, dim=0)
    return out[0] if squeeze else out


def saturating_reference(
    means2d,
    conics,
    colors,
    opacities,
    depths,
    H,
    W,
    bg=(1.0, 1.0, 1.0),
    alpha_threshold=1.0 / 255.0,
    transmittance_threshold=1.0e-4,
):
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
    ys = torch.arange(H, dtype=means2d.dtype, device=means2d.device) + 0.5
    xs = torch.arange(W, dtype=means2d.dtype, device=means2d.device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    for b in range(means2d.shape[0]):
        perm = torch.argsort(depths[b].detach(), stable=True)
        m = means2d[b][perm]
        q = conics[b][perm]
        c = colors[b][perm]
        o = opacities[b][perm]
        out = torch.zeros(H, W, 3, dtype=means2d.dtype, device=means2d.device)
        T = torch.ones(H, W, dtype=means2d.dtype, device=means2d.device)
        for i in range(m.shape[0]):
            active = T > transmittance_threshold
            dx = xx - m[i, 0]
            dy = yy - m[i, 1]
            power = -0.5 * (q[i, 0] * dx * dx + 2 * q[i, 1] * dx * dy + q[i, 2] * dy * dy)
            alpha = torch.clamp(o[i] * torch.exp(power), max=0.99)
            alpha = torch.where(
                (power <= 0) & (alpha >= alpha_threshold) & active,
                alpha,
                torch.zeros_like(alpha),
            )
            w = T * alpha
            out = out + w[..., None] * c[i]
            T = T * (1.0 - alpha)
        outs.append(out + T[..., None] * bg_t)
    out = torch.stack(outs, dim=0)
    return out[0] if squeeze else out


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, threshold: float) -> None:
    err = float((got - ref).detach().abs().max())
    print(f"{name} max error:", err)
    if err > threshold:
        raise AssertionError(f"{name} max error {err} exceeded {threshold}")


def check_case(B: int, *, use_active_tiles: bool):
    device = torch.device("mps")
    torch.manual_seed(0)
    G, H, W = 4, 16, 16
    means2d = torch.tensor(
        [
            [[4.2, 5.1], [8.0, 7.5], [11.2, 8.1], [6.4, 12.0]],
            [[3.7, 4.4], [7.1, 10.2], [12.2, 3.4], [10.4, 12.5]],
        ],
        device=device,
        dtype=torch.float32,
    )[:B].clone().requires_grad_(True)
    conics = torch.tensor(
        [
            [[0.35, 0.02, 0.42], [0.28, -0.01, 0.33], [0.25, 0.04, 0.31], [0.40, 0.00, 0.27]],
            [[0.31, 0.03, 0.38], [0.45, 0.01, 0.29], [0.22, -0.02, 0.35], [0.27, 0.05, 0.41]],
        ],
        device=device,
        dtype=torch.float32,
    )[:B].clone().requires_grad_(True)
    colors = torch.rand(B, G, 3, device=device, dtype=torch.float32, requires_grad=True)
    opacities = torch.tensor(
        [[0.8, 0.6, 0.5, 0.7], [0.65, 0.55, 0.75, 0.45]],
        device=device,
        dtype=torch.float32,
    )[:B].clone().requires_grad_(True)
    depths = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.3, 0.1, 0.4, 0.2]],
        device=device,
        dtype=torch.float32,
    )[:B].clone()

    cfg = RasterConfig(
        height=H,
        width=W,
        tile_size=16,
        max_fast_pairs=128,
        stop_count_mode="adaptive",
        use_active_tiles=use_active_tiles,
    )
    out = rasterize_projected_gaussians(means2d if B > 1 else means2d[0], conics if B > 1 else conics[0], colors if B > 1 else colors[0], opacities if B > 1 else opacities[0], depths if B > 1 else depths[0], cfg)
    loss = out.square().mean()
    loss.backward()

    means_r = (means2d.detach().cpu() if B > 1 else means2d[0].detach().cpu()).requires_grad_(True)
    conics_r = (conics.detach().cpu() if B > 1 else conics[0].detach().cpu()).requires_grad_(True)
    colors_r = (colors.detach().cpu() if B > 1 else colors[0].detach().cpu()).requires_grad_(True)
    opacities_r = (opacities.detach().cpu() if B > 1 else opacities[0].detach().cpu()).requires_grad_(True)
    depths_r = depths.detach().cpu() if B > 1 else depths[0].detach().cpu()
    out_r = dense_reference(means_r, conics_r, colors_r, opacities_r, depths_r, H, W)
    loss_r = out_r.square().mean()
    loss_r.backward()

    means_grad = means2d.grad.detach().cpu() if B > 1 else means2d.grad[0].detach().cpu()
    conics_grad = conics.grad.detach().cpu() if B > 1 else conics.grad[0].detach().cpu()
    colors_grad = colors.grad.detach().cpu() if B > 1 else colors.grad[0].detach().cpu()
    opacities_grad = opacities.grad.detach().cpu() if B > 1 else opacities.grad[0].detach().cpu()

    mode = "active" if use_active_tiles else "direct"
    assert_close(f"B={B} {mode} image", out.detach().cpu(), out_r.detach(), 1.0e-5)
    assert_close(f"B={B} {mode} means grad", means_grad, means_r.grad.detach().cpu(), 1.0e-5)
    assert_close(f"B={B} {mode} conics grad", conics_grad, conics_r.grad.detach().cpu(), 1.0e-5)
    assert_close(f"B={B} {mode} colors grad", colors_grad, colors_r.grad.detach().cpu(), 1.0e-5)
    assert_close(f"B={B} {mode} opacities grad", opacities_grad, opacities_r.grad.detach().cpu(), 1.0e-5)


def check_saturated_many_splats(*, use_active_tiles: bool):
    device = torch.device("mps")
    torch.manual_seed(123)
    B, G, H, W = 1, 64, 32, 32
    means2d = torch.randn(B, G, 2, device=device, dtype=torch.float32)
    means2d = means2d * torch.tensor([W * 0.2, H * 0.2], device=device) + torch.tensor([W * 0.5, H * 0.5], device=device)
    means2d[..., 0].clamp_(0, W - 1)
    means2d[..., 1].clamp_(0, H - 1)
    sigmas = torch.rand(B, G, 2, device=device, dtype=torch.float32) * 8.0 + 3.0
    conics = torch.stack(
        [
            1.0 / sigmas[..., 0].square(),
            torch.zeros(B, G, device=device, dtype=torch.float32),
            1.0 / sigmas[..., 1].square(),
        ],
        dim=-1,
    ).contiguous()
    colors = torch.rand(B, G, 3, device=device, dtype=torch.float32)
    opacities = torch.rand(B, G, device=device, dtype=torch.float32) * 0.45 + 0.45
    depths = (torch.arange(G, device=device, dtype=torch.float32) / float(G - 1)).view(1, G)

    means2d.requires_grad_(True)
    conics.requires_grad_(True)
    colors.requires_grad_(True)
    opacities.requires_grad_(True)

    cfg = RasterConfig(
        height=H,
        width=W,
        tile_size=16,
        max_fast_pairs=2048,
        background=(1.0, 1.0, 1.0),
        stop_count_mode="always",
        use_active_tiles=use_active_tiles,
    )
    out = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
    out.square().mean().backward()

    means_r = means2d.detach().cpu().requires_grad_(True)
    conics_r = conics.detach().cpu().requires_grad_(True)
    colors_r = colors.detach().cpu().requires_grad_(True)
    opacities_r = opacities.detach().cpu().requires_grad_(True)
    depths_r = depths.detach().cpu()
    ref = saturating_reference(means_r, conics_r, colors_r, opacities_r, depths_r, H, W)
    ref.square().mean().backward()

    mode = "active" if use_active_tiles else "direct"
    assert_close(f"saturated {mode} image", out.detach().cpu(), ref.detach(), 1.0e-5)
    assert_close(f"saturated {mode} means grad", means2d.grad.detach().cpu(), means_r.grad, 1.0e-5)
    assert_close(f"saturated {mode} conics grad", conics.grad.detach().cpu(), conics_r.grad, 1.0e-5)
    assert_close(f"saturated {mode} colors grad", colors.grad.detach().cpu(), colors_r.grad, 1.0e-5)
    assert_close(f"saturated {mode} opacities grad", opacities.grad.detach().cpu(), opacities_r.grad, 1.0e-5)


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")
    for use_active_tiles in (False, True):
        check_case(1, use_active_tiles=use_active_tiles)
        check_case(2, use_active_tiles=use_active_tiles)
        check_saturated_many_splats(use_active_tiles=use_active_tiles)


if __name__ == "__main__":
    main()
