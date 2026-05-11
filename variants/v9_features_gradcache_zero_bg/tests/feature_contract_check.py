from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
V5_ROOT = ROOT.parent / "v5"
for path in (ROOT, V5_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch_gsplat_bridge_v5 as v5
import torch_gsplat_bridge_v9_features_gradcache_zero_bg as v9_features_gradcache_zero_bg


def dense_reference(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    height: int,
    width: int,
    background: tuple[float, ...] | None = None,
) -> torch.Tensor:
    if means2d.ndim == 2:
        means2d = means2d.unsqueeze(0)
        conics = conics.unsqueeze(0)
        colors = colors.unsqueeze(0)
        opacities = opacities.unsqueeze(0)
        depths = depths.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    feature_dim = colors.shape[-1]
    if background is None:
        background = (0.0,) * feature_dim
    bg_t = torch.tensor(background, dtype=means2d.dtype, device=means2d.device)
    if bg_t.numel() == 1:
        bg_t = bg_t.expand(feature_dim)

    ys = torch.arange(height, dtype=means2d.dtype, device=means2d.device) + 0.5
    xs = torch.arange(width, dtype=means2d.dtype, device=means2d.device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    outs = []
    for b in range(means2d.shape[0]):
        perm = torch.argsort(depths[b].detach(), stable=True)
        m = means2d[b][perm]
        q = conics[b][perm]
        c = colors[b][perm]
        o = opacities[b][perm]
        out = torch.zeros(height, width, feature_dim, dtype=means2d.dtype, device=means2d.device)
        trans = torch.ones(height, width, dtype=means2d.dtype, device=means2d.device)
        for i in range(m.shape[0]):
            dx = xx - m[i, 0]
            dy = yy - m[i, 1]
            power = -0.5 * (q[i, 0] * dx * dx + 2.0 * q[i, 1] * dx * dy + q[i, 2] * dy * dy)
            alpha = torch.clamp(o[i] * torch.exp(power), max=0.99)
            alpha = torch.where((power <= 0.0) & (alpha >= 1.0 / 255.0), alpha, torch.zeros_like(alpha))
            w = trans * alpha
            out = out + w[..., None] * c[i]
            trans = trans * (1.0 - alpha)
        outs.append(out + trans[..., None] * bg_t)
    out = torch.stack(outs, dim=0)
    return out[0] if squeeze else out


def make_case(batch: int, gaussians: int, feature_dim: int, height: int, width: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    means2d = torch.rand(batch, gaussians, 2, device=device, dtype=torch.float32)
    means2d[..., 0] = means2d[..., 0] * (width - 2) + 1.0
    means2d[..., 1] = means2d[..., 1] * (height - 2) + 1.0
    sigmas = torch.rand(batch, gaussians, 2, device=device, dtype=torch.float32) * 2.0 + 2.0
    conics = torch.stack(
        [
            1.0 / sigmas[..., 0].square(),
            torch.zeros(batch, gaussians, device=device, dtype=torch.float32),
            1.0 / sigmas[..., 1].square(),
        ],
        dim=-1,
    ).contiguous()
    colors = torch.randn(batch, gaussians, feature_dim, device=device, dtype=torch.float32) * 0.3
    opacities = torch.rand(batch, gaussians, device=device, dtype=torch.float32) * 0.4 + 0.2
    depths = torch.rand(batch, gaussians, device=device, dtype=torch.float32)
    return means2d, conics, colors, opacities, depths


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, threshold: float) -> None:
    err = float((got - ref).detach().abs().max().cpu())
    print(f"{name} max_abs={err:.8g}")
    if err > threshold:
        raise AssertionError(f"{name} max_abs {err} exceeded {threshold}")


def check_shapes(*, active_policy: str = "off") -> None:
    device = torch.device("mps")
    height, width, gaussians = 13, 15, 9
    for feature_dim in (1, 3, 4, 8, 16, 32, 64):
        cfg = v9_features_gradcache_zero_bg.RasterConfig(
            height=height,
            width=width,
            tile_size=16,
            max_fast_pairs=128,
            active_policy=active_policy,
        )
        means2d, conics, colors, opacities, depths = make_case(2, gaussians, feature_dim, height, width, device, 10 + feature_dim)
        with torch.no_grad():
            out_b, alpha_b = v9_features_gradcache_zero_bg.rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
            out_s, alpha_s = v9_features_gradcache_zero_bg.rasterize_projected_gaussians(means2d[0], conics[0], colors[0], opacities[0], depths[0], cfg)
        assert tuple(out_b.shape) == (2, height, width, feature_dim)
        assert tuple(alpha_b.shape) == (2, height, width)
        assert tuple(out_s.shape) == (height, width, feature_dim)
        assert tuple(alpha_s.shape) == (height, width)
    print(f"shape contract active_policy={active_policy}: ok")


def check_v5_parity(*, active_policy: str = "off") -> None:
    device = torch.device("mps")
    height, width, gaussians = 24, 24, 24
    means2d, conics, colors, opacities, depths = make_case(2, gaussians, 3, height, width, device, 123)
    cfg_v5 = v5.RasterConfig(height=height, width=width, tile_size=16, max_fast_pairs=256, background=(0.1, 0.2, 0.3))
    cfg_features = v9_features_gradcache_zero_bg.RasterConfig(
        height=height,
        width=width,
        tile_size=16,
        max_fast_pairs=256,
        background=(0.1, 0.2, 0.3),
        active_policy=active_policy,
    )
    with torch.no_grad():
        out_v5 = v5.rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg_v5)
        out_features, _alpha = v9_features_gradcache_zero_bg.rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg_features)
    assert_close(f"F=3 v5 parity active_policy={active_policy}", out_features.cpu(), out_v5.cpu(), 1.0e-6)


def check_feature_grad(feature_dim: int, *, active_policy: str = "off") -> None:
    device = torch.device("mps")
    height, width, gaussians = 10, 11, 7
    means2d, conics, colors, opacities, depths = make_case(1, gaussians, feature_dim, height, width, device, 200 + feature_dim)
    colors_mps = colors.detach().clone().requires_grad_(True)
    cfg = v9_features_gradcache_zero_bg.RasterConfig(
        height=height,
        width=width,
        tile_size=16,
        max_fast_pairs=128,
        active_policy=active_policy,
    )
    out, _alpha = v9_features_gradcache_zero_bg.rasterize_projected_gaussians(means2d, conics, colors_mps, opacities, depths, cfg)
    loss = out.square().mean()
    loss.backward()

    means_cpu = means2d.detach().cpu().double()
    conics_cpu = conics.detach().cpu().double()
    colors_cpu = colors.detach().cpu().double().requires_grad_(True)
    opacities_cpu = opacities.detach().cpu().double()
    depths_cpu = depths.detach().cpu().double()
    ref = dense_reference(means_cpu, conics_cpu, colors_cpu, opacities_cpu, depths_cpu, height, width)
    ref.square().mean().backward()
    assert_close(
        f"F={feature_dim} feature grad active_policy={active_policy}",
        colors_mps.grad.detach().cpu(),
        colors_cpu.grad.detach().float(),
        1.0e-4,
    )


def check_no_nan_smoke(*, active_policy: str = "off") -> None:
    device = torch.device("mps")
    cfg = v9_features_gradcache_zero_bg.RasterConfig(
        height=16,
        width=16,
        tile_size=16,
        max_fast_pairs=256,
        active_policy=active_policy,
    )
    for i in range(100):
        means2d, conics, colors, opacities, depths = make_case(1, 24, 32, 16, 16, device, 1000 + i)
        colors.requires_grad_(True)
        out, _alpha = v9_features_gradcache_zero_bg.rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
        loss = out.square().mean()
        loss.backward()
        tensors = (out, colors.grad)
        if any(not torch.isfinite(t).all().item() for t in tensors):
            raise AssertionError(f"NaN/Inf detected in F=32 smoke iteration {i}")
    print(f"F=32 no-NaN smoke active_policy={active_policy}: ok")


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    for active_policy in ("off", "on"):
        check_shapes(active_policy=active_policy)
        check_v5_parity(active_policy=active_policy)
    for feature_dim in (3, 8, 32, 64):
        check_feature_grad(feature_dim)
    check_feature_grad(32, active_policy="on")
    check_no_nan_smoke()
    check_no_nan_smoke(active_policy="on")


if __name__ == "__main__":
    main()
