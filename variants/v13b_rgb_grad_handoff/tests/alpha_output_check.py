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
from torch_gsplat_bridge_v13b_rgb_grad_handoff import RasterConfig, get_runtime_shader_config, rasterize_projected_gaussians
from tests.reference_alpha import reference_features_and_alpha


def _device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    return torch.device("mps")


def _config(height: int, width: int, feature_dim: int) -> RasterConfig:
    rt = get_runtime_shader_config()
    return RasterConfig(
        height=height,
        width=width,
        tile_size=rt.tile_size,
        max_fast_pairs=2048,
        alpha_threshold=1.0 / 255.0,
        transmittance_threshold=1.0e-4,
        background=(0.0,) * feature_dim,
        enable_overflow_fallback=True,
        inputs_sorted_by_depth=True,
        batch_strategy="serial",
        batch_launch_limit_tiles=262144,
        batch_launch_limit_gaussians=262144,
    )


def _random_inputs(device: torch.device, *, gaussians: int = 4, feature_dim: int = 3, height: int = 8):
    means2d = (torch.rand(gaussians, 2, device=device) * height).requires_grad_(True)
    conics = torch.zeros(gaussians, 3, device=device)
    conics[:, 0] = 1.0
    conics[:, 2] = 1.0
    conics = conics.requires_grad_(True)
    colors = torch.randn(gaussians, feature_dim, device=device).requires_grad_(True)
    opacities = torch.sigmoid(torch.randn(gaussians, device=device)).requires_grad_(True)
    depths = torch.linspace(0.0, 1.0, gaussians, device=device)
    return means2d, conics, colors, opacities, depths


def _max_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().cpu())


def test_forward_alpha_shape_and_values() -> None:
    device = _device()
    height, width, feature_dim = 8, 8, 3
    means2d = torch.tensor(
        [[1.5, 1.5], [4.5, 4.5], [4.5, 4.5], [-10.0, -10.0]],
        device=device,
        dtype=torch.float32,
    )
    conics = torch.tensor([[100.0, 0.0, 100.0]], device=device, dtype=torch.float32).expand(4, 3).contiguous()
    colors = torch.randn(4, feature_dim, device=device, dtype=torch.float32)
    opacities = torch.tensor([0.5, 0.5, 0.5, 0.5], device=device, dtype=torch.float32)
    depths = torch.arange(4, device=device, dtype=torch.float32)

    features, alpha = rasterize_projected_gaussians(
        means2d,
        conics,
        colors,
        opacities,
        depths,
        _config(height, width, feature_dim),
    )
    assert tuple(features.shape) == (height, width, feature_dim)
    assert tuple(alpha.shape) == (height, width)
    assert abs(float(alpha[7, 7].cpu())) < 1.0e-7
    assert abs(float(alpha[1, 1].cpu()) - 0.5) < 1.0e-6
    assert abs(float(alpha[4, 4].cpu()) - 0.75) < 1.0e-6
    print("Test A passed.")


def test_alpha_only_loss_propagates_to_geometry() -> None:
    torch.manual_seed(0)
    device = _device()
    gaussians, feature_dim, height, width = 4, 3, 8, 8

    m1, c1, col1, o1, d1 = _random_inputs(device, gaussians=gaussians, feature_dim=feature_dim, height=height)
    _features_k, alpha_k = rasterize_projected_gaussians(m1, c1, col1, o1, d1, _config(height, width, feature_dim))
    alpha_k.sum().backward()

    torch.manual_seed(0)
    m2, c2, col2, o2, d2 = _random_inputs(device, gaussians=gaussians, feature_dim=feature_dim, height=height)
    feat_bg = torch.zeros(feature_dim, device=device)
    _features_r, alpha_r = reference_features_and_alpha(m2, c2, col2, o2, d2, height, width, feat_bg)
    alpha_r.sum().backward()

    assert _max_abs(m1.grad - m2.grad) < 1.0e-4, "means2d alpha-only grad mismatch"
    assert _max_abs(c1.grad - c2.grad) < 1.0e-4, "conics alpha-only grad mismatch"
    assert _max_abs(o1.grad - o2.grad) < 1.0e-4, "opacities alpha-only grad mismatch"
    assert _max_abs(col1.grad) < 1.0e-6, "alpha-only loss should not produce color grad"
    print("Test B passed.")


def test_alpha_matches_synthetic_feature_channel() -> None:
    torch.manual_seed(0)
    device = _device()
    gaussians, feature_dim, height, width = 4, 3, 8, 8
    means2d, conics, colors, opacities, depths = _random_inputs(
        device,
        gaussians=gaussians,
        feature_dim=feature_dim,
        height=height,
    )
    means2d = means2d.detach()
    conics = conics.detach()
    colors = colors.detach()
    opacities = opacities.detach()

    _features, alpha = rasterize_projected_gaussians(
        means2d,
        conics,
        colors,
        opacities,
        depths,
        _config(height, width, feature_dim),
    )

    colors_marker = torch.cat([colors, torch.ones(gaussians, 1, device=device)], dim=-1)
    features_marker, _alpha_unused = rasterize_projected_gaussians(
        means2d,
        conics,
        colors_marker,
        opacities,
        depths,
        _config(height, width, feature_dim + 1),
    )
    diff = _max_abs(features_marker[..., -1] - alpha)
    assert diff < 1.0e-6, f"alpha-vs-marker-channel parity max abs diff = {diff:g}"
    print("Test C passed.")


def test_combined_backward_linear() -> None:
    torch.manual_seed(0)
    device = _device()
    gaussians, feature_dim, height, width = 4, 3, 8, 8
    config = _config(height, width, feature_dim)

    m, c, col, o, d = _random_inputs(device, gaussians=gaussians, feature_dim=feature_dim, height=height)
    feat, alpha = rasterize_projected_gaussians(m, c, col, o, d, config)
    (feat.sum() + alpha.sum()).backward()
    g_combined = (m.grad.clone(), c.grad.clone(), col.grad.clone(), o.grad.clone())

    torch.manual_seed(0)
    m_f, c_f, col_f, o_f, d_f = _random_inputs(device, gaussians=gaussians, feature_dim=feature_dim, height=height)
    feat_f, _alpha_f = rasterize_projected_gaussians(m_f, c_f, col_f, o_f, d_f, config)
    feat_f.sum().backward()
    g_feat = (m_f.grad, c_f.grad, col_f.grad, o_f.grad)

    torch.manual_seed(0)
    m_a, c_a, col_a, o_a, d_a = _random_inputs(device, gaussians=gaussians, feature_dim=feature_dim, height=height)
    _feat_a, alpha_a = rasterize_projected_gaussians(m_a, c_a, col_a, o_a, d_a, config)
    alpha_a.sum().backward()
    g_alpha = (m_a.grad, c_a.grad, col_a.grad, o_a.grad)

    for combined, feat_grad, alpha_grad, name in zip(g_combined, g_feat, g_alpha, ["means2d", "conics", "colors", "opacities"]):
        diff = _max_abs(combined - (feat_grad + alpha_grad))
        assert diff < 1.0e-5, f"{name}: combined != separate sum, diff={diff:g}"
    print("Test D passed.")


def test_f3_v5_alpha_parity() -> None:
    torch.manual_seed(11)
    device = _device()
    gaussians, feature_dim, height, width = 12, 3, 16, 16
    means2d, conics, colors, opacities, depths = _random_inputs(
        device,
        gaussians=gaussians,
        feature_dim=feature_dim,
        height=height,
    )
    means2d = means2d.detach()
    conics = conics.detach()
    colors = colors.detach()
    opacities = opacities.detach()

    _features, alpha = rasterize_projected_gaussians(
        means2d,
        conics,
        colors,
        opacities,
        depths,
        _config(height, width, feature_dim),
    )

    rt = get_runtime_shader_config()
    v5_cfg = v5.RasterConfig(
        height=height,
        width=width,
        tile_size=rt.tile_size,
        max_fast_pairs=2048,
        alpha_threshold=1.0 / 255.0,
        transmittance_threshold=1.0e-4,
        background=(1.0, 1.0, 1.0),
        enable_overflow_fallback=True,
        inputs_sorted_by_depth=True,
        batch_strategy="serial",
        batch_launch_limit_tiles=262144,
        batch_launch_limit_gaussians=262144,
    )
    zeros = torch.zeros_like(colors)
    transmittance_rgb = v5.rasterize_projected_gaussians(means2d, conics, zeros, opacities, depths, v5_cfg)
    alpha_v5 = 1.0 - transmittance_rgb[..., 0]
    diff = _max_abs(alpha - alpha_v5)
    assert diff < 1.0e-6, f"F=3 alpha parity vs v5 max abs diff = {diff:g}"
    print("Test E passed.")


def test_active_f32_matches_direct_for_feature_and_alpha_grad() -> None:
    torch.manual_seed(23)
    device = _device()
    gaussians, feature_dim, height, width = 16, 32, 16, 16
    means2d, conics, colors, opacities, depths = _random_inputs(
        device,
        gaussians=gaussians,
        feature_dim=feature_dim,
        height=height,
    )

    def run(active_policy: str):
        m = means2d.detach().clone().requires_grad_(True)
        c = conics.detach().clone().requires_grad_(True)
        col = colors.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        cfg = _config(height, width, feature_dim)
        cfg = RasterConfig(
            height=cfg.height,
            width=cfg.width,
            tile_size=cfg.tile_size,
            max_fast_pairs=cfg.max_fast_pairs,
            alpha_threshold=cfg.alpha_threshold,
            transmittance_threshold=cfg.transmittance_threshold,
            background=cfg.background,
            enable_overflow_fallback=cfg.enable_overflow_fallback,
            inputs_sorted_by_depth=cfg.inputs_sorted_by_depth,
            batch_strategy=cfg.batch_strategy,
            batch_launch_limit_tiles=cfg.batch_launch_limit_tiles,
            batch_launch_limit_gaussians=cfg.batch_launch_limit_gaussians,
            active_policy=active_policy,
            stop_count_mode="adaptive",
            stop_count_dense_threshold=8,
        )
        feat, alpha = rasterize_projected_gaussians(m, c, col, o, depths, cfg)
        (feat.square().mean() + alpha.square().mean()).backward()
        return feat, alpha, (m.grad, c.grad, col.grad, o.grad)

    direct_feat, direct_alpha, direct_grads = run("off")
    active_feat, active_alpha, active_grads = run("on")
    assert _max_abs(direct_feat - active_feat) < 1.0e-6, "active F32 features differ from direct"
    assert _max_abs(direct_alpha - active_alpha) < 1.0e-6, "active F32 alpha differs from direct"
    for name, direct_grad, active_grad in zip(
        ("means2d", "conics", "features", "opacities"),
        direct_grads,
        active_grads,
    ):
        diff = _max_abs(direct_grad - active_grad)
        assert diff < 1.0e-5, f"active F32 {name} grad differs from direct, diff={diff:g}"
    print("Test F passed.")


def main() -> None:
    test_forward_alpha_shape_and_values()
    test_alpha_only_loss_propagates_to_geometry()
    test_alpha_matches_synthetic_feature_channel()
    test_combined_backward_linear()
    test_f3_v5_alpha_parity()
    test_active_f32_matches_direct_for_feature_and_alpha_grad()


if __name__ == "__main__":
    main()
