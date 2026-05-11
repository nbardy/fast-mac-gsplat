from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v12b_fused_colorize_rmsnorm_l1 import (  # noqa: E402
    fused_rmsnorm_colorize_alpha_l1_loss,
    manual_fused_rmsnorm_colorize_alpha_l1_grads,
)


def _make_case(feature_dim: int, *, dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cpu").manual_seed(1000 + feature_dim)
    features = torch.randn((2, 5, 7, feature_dim), dtype=dtype, generator=gen) * 0.4
    alpha = torch.rand((2, 5, 7), dtype=dtype, generator=gen)
    target = torch.rand((2, 5, 7, 3), dtype=dtype, generator=gen)
    background = torch.rand((2, 3, 1, 1), dtype=dtype, generator=gen)
    weight = torch.randn((3, feature_dim), dtype=dtype, generator=gen) * 0.2
    bias = torch.randn((3,), dtype=dtype, generator=gen) * 0.1
    gamma = torch.rand((feature_dim,), dtype=dtype, generator=gen) + 0.5
    return features, alpha, target, background, weight, bias, gamma


def _assert_close(name: str, actual: torch.Tensor | None, expected: torch.Tensor | None, *, atol: float = 1.0e-10) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{name}: one value is None and the other is not")
        return
    if not torch.allclose(actual, expected, rtol=1.0e-8, atol=atol):
        max_diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name}: max diff {max_diff:.3e} exceeds tolerance")


def _autograd_grads(
    features: torch.Tensor,
    alpha: torch.Tensor,
    target: torch.Tensor,
    background: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    gamma: torch.Tensor | None,
    *,
    norm: str = "rms",
    activation: str = "sigmoid",
) -> tuple[torch.Tensor, ...]:
    leaves = []
    for value in (features, alpha, weight, bias):
        leaf = value.detach().clone().requires_grad_(True)
        leaves.append(leaf)
    if gamma is None:
        gamma_leaf = None
    else:
        gamma_leaf = gamma.detach().clone().requires_grad_(True)

    out = fused_rmsnorm_colorize_alpha_l1_loss(
        leaves[0],
        leaves[1],
        target,
        background,
        leaves[2],
        leaves[3],
        gamma_leaf,
        norm=norm,  # type: ignore[arg-type]
        activation=activation,  # type: ignore[arg-type]
    )
    out.loss.backward()
    grad_gamma = None if gamma_leaf is None else gamma_leaf.grad
    return out.loss.detach(), leaves[0].grad, leaves[1].grad, leaves[2].grad, leaves[3].grad, grad_gamma


def test_rms_manual_gradients() -> None:
    for feature_dim in (3, 8, 32):
        features, alpha, target, background, weight, bias, gamma = _make_case(feature_dim)
        actual = _autograd_grads(features, alpha, target, background, weight, bias, gamma)
        expected = manual_fused_rmsnorm_colorize_alpha_l1_grads(
            features,
            alpha,
            target,
            background,
            weight,
            bias,
            gamma,
        )
        _assert_close(f"F{feature_dim} loss", actual[0], expected.loss)
        _assert_close(f"F{feature_dim} grad_features", actual[1], expected.grad_features)
        _assert_close(f"F{feature_dim} grad_alpha", actual[2], expected.grad_alpha)
        _assert_close(f"F{feature_dim} grad_weight", actual[3], expected.grad_color_weight)
        _assert_close(f"F{feature_dim} grad_bias", actual[4], expected.grad_color_bias)
        _assert_close(f"F{feature_dim} grad_gamma", actual[5], expected.grad_rms_gamma)


def test_identity_activation_and_no_norm() -> None:
    features, alpha, target, background, weight, bias, _gamma = _make_case(8)
    actual = _autograd_grads(features, alpha, target, background, weight, bias, None, norm="none", activation="identity")
    expected = manual_fused_rmsnorm_colorize_alpha_l1_grads(
        features,
        alpha,
        target,
        background,
        weight,
        bias,
        None,
        norm="none",
        activation="identity",
    )
    _assert_close("no_norm loss", actual[0], expected.loss)
    _assert_close("no_norm grad_features", actual[1], expected.grad_features)
    _assert_close("no_norm grad_alpha", actual[2], expected.grad_alpha)
    _assert_close("no_norm grad_weight", actual[3], expected.grad_color_weight)
    _assert_close("no_norm grad_bias", actual[4], expected.grad_color_bias)
    _assert_close("no_norm grad_gamma", actual[5], expected.grad_rms_gamma)


def test_l1_kink_zero_gradient() -> None:
    features, alpha, _target, background, weight, bias, gamma = _make_case(8)
    with torch.no_grad():
        target = fused_rmsnorm_colorize_alpha_l1_loss(
            features,
            alpha,
            torch.zeros((*features.shape[:3], 3), dtype=features.dtype),
            background,
            weight,
            bias,
            gamma,
        ).composed_rgb
    grads = manual_fused_rmsnorm_colorize_alpha_l1_grads(
        features,
        alpha,
        target,
        background,
        weight,
        bias,
        gamma,
    )
    _assert_close("kink grad_features", grads.grad_features, torch.zeros_like(grads.grad_features))
    _assert_close("kink grad_alpha", grads.grad_alpha, torch.zeros_like(grads.grad_alpha))
    _assert_close("kink grad_weight", grads.grad_color_weight, torch.zeros_like(grads.grad_color_weight))
    _assert_close("kink grad_bias", grads.grad_color_bias, torch.zeros_like(grads.grad_color_bias))
    _assert_close("kink grad_gamma", grads.grad_rms_gamma, torch.zeros_like(grads.grad_rms_gamma))


def test_alpha_zero_blocks_colorizer_gradient() -> None:
    features, _alpha, target, background, weight, bias, gamma = _make_case(8)
    alpha = torch.zeros(features.shape[:3], dtype=features.dtype)
    grads = manual_fused_rmsnorm_colorize_alpha_l1_grads(
        features,
        alpha,
        target,
        background,
        weight,
        bias,
        gamma,
    )
    _assert_close("alpha0 grad_features", grads.grad_features, torch.zeros_like(grads.grad_features))
    _assert_close("alpha0 grad_weight", grads.grad_color_weight, torch.zeros_like(grads.grad_color_weight))
    _assert_close("alpha0 grad_bias", grads.grad_color_bias, torch.zeros_like(grads.grad_color_bias))
    _assert_close("alpha0 grad_gamma", grads.grad_rms_gamma, torch.zeros_like(grads.grad_rms_gamma))


def main() -> None:
    test_rms_manual_gradients()
    test_identity_activation_and_no_norm()
    test_l1_kink_zero_gradient()
    test_alpha_zero_blocks_colorizer_gradient()
    print("fused_colorize_l1_reference_check passed")


if __name__ == "__main__":
    main()
