from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import cast

import pytest
import torch


STAR_ROOT = Path(__file__).resolve().parents[1]
if str(STAR_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_ROOT))

from research_project.trainer_harness.model import (  # noqa: E402
    dense_differentiable_render_uvt_tubes,
)
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    UVTRenderConfig,
    brute_force_render_uvt_feature_tubes,
    brute_force_render_uvt_tubes,
    direct_atomic_backward,
    primitive_alpha,
    primitive_alpha_and_vjp_terms,
    render_uvt_tubes,
)
from torch_gsplat_bridge_star_uvt.rasterize import (  # noqa: E402
    AlphaMode,
    _make_meta,
    _runtime_validate,
    _support_tau,
)
from torch_gsplat_bridge_star_uvt.projective_trace import (  # noqa: E402
    _make_projective_interval_meta,
)


@pytest.fixture(autouse=True)
def _canonical_shader_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "STAR_UVT_TILE_X",
        "STAR_UVT_TILE_Y",
        "STAR_UVT_TILE_T",
        "STAR_UVT_TILE_CAPACITY",
    ):
        monkeypatch.delenv(name, raising=False)


def _config(
    alpha_mode: AlphaMode = "peak_splat",
    *,
    height: int = 1,
    width: int = 1,
    frames: int = 1,
    max_alpha: float = 0.99,
    alpha_threshold: float = 1.0 / 255.0,
) -> UVTRenderConfig:
    return UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        alpha_mode=alpha_mode,
        max_alpha=max_alpha,
        alpha_threshold=alpha_threshold,
    )


def test_peak_splat_remains_default_and_bitwise_formula_equivalent() -> None:
    config = _config()
    opacity = torch.tensor([0.2, 0.7], dtype=torch.float32)
    qv = torch.tensor([0.0, 1.25], dtype=torch.float32)

    actual = primitive_alpha(opacity, qv, config)
    historical = torch.clamp(opacity * torch.exp(-0.5 * qv), max=config.max_alpha)

    assert config.alpha_mode == "peak_splat"
    assert config.opacity_semantics == "peak_alpha_amplitude"
    assert torch.equal(actual, historical)


def test_legacy_zero_cutoff_and_unit_cap_remain_valid() -> None:
    config = _config(alpha_threshold=0.0, max_alpha=1.0)

    _runtime_validate(config)
    _meta_i32, meta_f32 = _make_meta(config, torch.device("cpu"), tube_count=1)

    assert float(meta_f32[-1]) == 0.0


def test_beer_lambert_uses_optical_thickness_and_explicit_metadata() -> None:
    config = _config("beer_lambert")
    opacity = torch.tensor([0.0, 0.2, 1.3], dtype=torch.float64)
    qv = torch.tensor([0.0, 0.6, 1.1], dtype=torch.float64)
    density = torch.exp(-0.5 * qv)

    actual = primitive_alpha(opacity, qv, config)
    expected = -torch.expm1(-(opacity * density))
    _meta_i32, meta_f32 = _make_meta(config, torch.device("cpu"), tube_count=3)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert (
        config.opacity_semantics
        == "nonnegative_fiber_integrated_peak_optical_thickness"
    )
    assert meta_f32.numel() == 8
    assert float(meta_f32[-1]) == 1.0


def test_alpha_mode_and_physical_domain_fail_loudly() -> None:
    invalid = _config(cast(AlphaMode, "mystery_transfer"))
    with pytest.raises(ValueError, match="alpha_mode"):
        _runtime_validate(invalid)

    beer = _config("beer_lambert")
    with pytest.raises(ValueError, match="nonnegative optical thickness"):
        primitive_alpha(torch.tensor([-0.01]), torch.tensor([0.0]), beer)
    with pytest.raises(ValueError, match="non-finite optical thickness"):
        primitive_alpha(torch.tensor([math.inf]), torch.tensor([0.0]), beer)


def test_projective_atlas_beer_lambert_remains_fail_loud() -> None:
    with pytest.raises(ValueError, match="supports only alpha_mode='peak_splat'"):
        _make_projective_interval_meta(
            _config("beer_lambert"),
            torch.device("cpu"),
            trace_count=1,
        )


@pytest.mark.parametrize("alpha_mode", ["peak_splat", "beer_lambert"])
def test_analytic_alpha_vjp_matches_autograd_and_finite_difference(alpha_mode: AlphaMode) -> None:
    config = _config(alpha_mode)
    opacity = torch.tensor([0.18, 0.83], dtype=torch.float64, requires_grad=True)
    qv = torch.tensor([0.37, 1.21], dtype=torch.float64, requires_grad=True)
    upstream = torch.tensor([0.7, -1.4], dtype=torch.float64)

    alpha, d_opacity, d_qv = primitive_alpha_and_vjp_terms(opacity, qv, config)
    grad_opacity, grad_qv = torch.autograd.grad((alpha * upstream).sum(), (opacity, qv))

    torch.testing.assert_close(grad_opacity, upstream * d_opacity, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(grad_qv, upstream * d_qv, rtol=1.0e-12, atol=1.0e-12)

    eps = 1.0e-6
    opacity_plus = opacity.detach().clone()
    opacity_minus = opacity.detach().clone()
    opacity_plus[0] += eps
    opacity_minus[0] -= eps
    fd_opacity = (
        primitive_alpha(opacity_plus, qv.detach(), config)[0]
        - primitive_alpha(opacity_minus, qv.detach(), config)[0]
    ) / (2.0 * eps)
    qv_plus = qv.detach().clone()
    qv_minus = qv.detach().clone()
    qv_plus[0] += eps
    qv_minus[0] -= eps
    fd_qv = (
        primitive_alpha(opacity.detach(), qv_plus, config)[0]
        - primitive_alpha(opacity.detach(), qv_minus, config)[0]
    ) / (2.0 * eps)

    torch.testing.assert_close(fd_opacity, d_opacity[0], rtol=2.0e-8, atol=2.0e-10)
    torch.testing.assert_close(fd_qv, d_qv[0], rtol=2.0e-8, atol=2.0e-10)


@pytest.mark.parametrize("alpha_mode", ["peak_splat", "beer_lambert"])
def test_hard_alpha_cap_has_zero_vjp(alpha_mode: AlphaMode) -> None:
    config = _config(alpha_mode, max_alpha=0.5)
    opacity = torch.tensor([10.0], dtype=torch.float64)
    qv = torch.tensor([0.0], dtype=torch.float64)

    alpha, d_opacity, d_qv = primitive_alpha_and_vjp_terms(opacity, qv, config)

    torch.testing.assert_close(alpha, torch.tensor([0.5], dtype=torch.float64))
    assert float(d_opacity[0]) == 0.0
    assert float(d_qv[0]) == 0.0


def test_beer_lambert_support_radius_solves_the_alpha_cutoff() -> None:
    config = _config("beer_lambert", alpha_threshold=0.1)
    opacity_value = 0.7
    support_qv = _support_tau(opacity_value, config)

    assert support_qv is not None
    at_boundary = primitive_alpha(
        torch.tensor([opacity_value], dtype=torch.float64),
        torch.tensor([support_qv], dtype=torch.float64),
        config,
    )
    just_inside = primitive_alpha(
        torch.tensor([opacity_value], dtype=torch.float64),
        torch.tensor([support_qv - 1.0e-5], dtype=torch.float64),
        config,
    )
    just_outside = primitive_alpha(
        torch.tensor([opacity_value], dtype=torch.float64),
        torch.tensor([support_qv + 1.0e-5], dtype=torch.float64),
        config,
    )

    torch.testing.assert_close(at_boundary, torch.tensor([0.1], dtype=torch.float64), atol=1.0e-14, rtol=0.0)
    assert float(just_inside[0]) > config.alpha_threshold
    assert float(just_outside[0]) < config.alpha_threshold


def _tiny_scene() -> tuple[torch.Tensor, ...]:
    ma = torch.tensor([[0.5, 0.5, 0.0], [1.0, 1.0, 0.1]], dtype=torch.float32)
    q_uvt = torch.tensor(
        [
            [0.30, 0.02, 0.01, 0.25, -0.01, 0.40],
            [0.22, -0.01, 0.00, 0.28, 0.02, 0.35],
        ],
        dtype=torch.float32,
    )
    depth0 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    depth_beta = torch.zeros((2, 3), dtype=torch.float32)
    opacity = torch.tensor([0.35, 0.8], dtype=torch.float32)
    color = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.4, 0.8]], dtype=torch.float32)
    return ma, q_uvt, depth0, depth_beta, opacity, color


@pytest.mark.parametrize("alpha_mode", ["peak_splat", "beer_lambert"])
def test_dense_cpu_reference_matches_brute_force_and_has_finite_vjp(alpha_mode: AlphaMode) -> None:
    config = _config(alpha_mode, height=2, width=2)
    scene = _tiny_scene()
    brute = brute_force_render_uvt_tubes(*scene, config)

    ma = scene[0].clone().requires_grad_(True)
    q_uvt = scene[1].clone().requires_grad_(True)
    opacity = scene[4].clone().requires_grad_(True)
    color = scene[5].clone().requires_grad_(True)
    dense = dense_differentiable_render_uvt_tubes(
        ma,
        q_uvt,
        scene[2],
        scene[3],
        opacity,
        color,
        config,
    )

    torch.testing.assert_close(dense, brute, rtol=2.0e-6, atol=2.0e-7)
    dense.square().mean().backward()
    for tensor in (ma, q_uvt, opacity, color):
        assert tensor.grad is not None
        assert bool(torch.isfinite(tensor.grad).all())
        assert float(tensor.grad.abs().sum()) > 0.0


def test_single_primitive_beer_lambert_render_has_expected_transmittance() -> None:
    config = _config("beer_lambert")
    optical_thickness = 0.7
    color = torch.tensor([[0.2, 0.5, 0.9]], dtype=torch.float32)
    image = brute_force_render_uvt_tubes(
        torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.tensor([optical_thickness], dtype=torch.float32),
        color,
        config,
    )
    expected_alpha = 1.0 - math.exp(-optical_thickness)

    torch.testing.assert_close(
        image[0, 0, 0],
        color[0] * expected_alpha,
        rtol=2.0e-7,
        atol=2.0e-7,
    )


def test_feature_cpu_reference_uses_the_same_beer_lambert_transfer() -> None:
    config = _config("beer_lambert")
    optical_thickness = 0.45
    feature = torch.tensor([[0.1, -0.3, 0.8, 1.2]], dtype=torch.float32)
    feature_image, alpha_image = brute_force_render_uvt_feature_tubes(
        torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32),
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.tensor([optical_thickness], dtype=torch.float32),
        feature,
        config,
    )
    expected_alpha = 1.0 - math.exp(-optical_thickness)

    torch.testing.assert_close(
        feature_image[0, :, 0, 0],
        feature[0] * expected_alpha,
        rtol=2.0e-7,
        atol=2.0e-7,
    )
    torch.testing.assert_close(
        alpha_image[0, 0, 0],
        torch.tensor(expected_alpha, dtype=torch.float32),
        rtol=2.0e-7,
        atol=2.0e-7,
    )


def test_metal_beer_lambert_forward_and_direct_atomic_vjp_match_cpu_when_opted_in() -> None:
    if os.environ.get("STAR_UVT_RUN_BEER_LAMBERT_MPS") != "1":
        pytest.skip("set STAR_UVT_RUN_BEER_LAMBERT_MPS=1 to run the Metal parity gate")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")

    config = _config("beer_lambert", height=2, width=2)
    scene = _tiny_scene()
    upstream = torch.linspace(
        -0.4,
        0.7,
        config.frames * config.height * config.width * 3,
        dtype=torch.float32,
    ).reshape(config.frames, config.height, config.width, 3)

    ma = scene[0].clone().requires_grad_(True)
    q_uvt = scene[1].clone().requires_grad_(True)
    opacity = scene[4].clone().requires_grad_(True)
    color = scene[5].clone().requires_grad_(True)
    reference = dense_differentiable_render_uvt_tubes(
        ma,
        q_uvt,
        scene[2],
        scene[3],
        opacity,
        color,
        config,
    )
    reference_grads = torch.autograd.grad(
        (reference * upstream).sum(),
        (ma, q_uvt, opacity, color),
    )

    mps_scene = tuple(value.to("mps") for value in scene)
    metal = render_uvt_tubes(*mps_scene, config)
    metal_grads = direct_atomic_backward(
        *mps_scene,
        upstream.to("mps"),
        config,
    )

    torch.testing.assert_close(
        metal.cpu(),
        reference.detach(),
        rtol=3.0e-5,
        atol=3.0e-6,
    )
    for actual, expected in zip(metal_grads[:4], reference_grads, strict=True):
        torch.testing.assert_close(
            actual.cpu(),
            expected,
            rtol=8.0e-4,
            atol=8.0e-5,
        )
    assert not bool(metal_grads[4].any().cpu())
