from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v6_feature_lookup_experiment import (
    RasterConfig,
    feature_ids_to_coefficients,
    rasterize_projected_gaussians,
    rasterize_projected_gaussians_feature_ids,
    rasterize_projected_gaussians_feature_lookup,
)


def _device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    return torch.device("mps")


def _clone_leaf(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone().requires_grad_(True)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())


def _make_case(device: torch.device):
    torch.manual_seed(7)
    gaussians, compact_dim, feature_dim = 10, 4, 7
    means = torch.rand(gaussians, 2, device=device) * 20.0 + 6.0
    base = torch.rand(gaussians, 2, device=device) * 0.03 + 0.03
    conics = torch.stack(
        [base[:, 0], torch.zeros(gaussians, device=device), base[:, 1]],
        dim=-1,
    )
    weights = torch.randn(gaussians, compact_dim, device=device) * 0.25
    lookup = torch.randn(compact_dim, feature_dim, device=device) * 0.5
    opacities = torch.rand(gaussians, device=device) * 0.35 + 0.35
    depths = torch.linspace(0.1, 1.0, gaussians, device=device)
    background = tuple(float(x) for x in torch.linspace(-0.2, 0.2, feature_dim))
    config = RasterConfig(
        height=32,
        width=32,
        background=background,
        inputs_sorted_by_depth=True,
        active_policy="off",
        enable_overflow_fallback=False,
    )
    return means, conics, weights, lookup, opacities, depths, config


def test_feature_lookup_matches_direct_features_and_gradients() -> None:
    device = _device()
    means, conics, weights, lookup, opacities, depths, config = _make_case(device)

    means_direct = _clone_leaf(means)
    conics_direct = _clone_leaf(conics)
    weights_direct = _clone_leaf(weights)
    lookup_direct = _clone_leaf(lookup)
    opacities_direct = _clone_leaf(opacities)
    direct_features = weights_direct @ lookup_direct
    out_direct, alpha_direct = rasterize_projected_gaussians(
        means_direct,
        conics_direct,
        direct_features,
        opacities_direct,
        depths,
        config,
    )
    loss_direct = out_direct.square().mean() + 0.17 * alpha_direct.square().mean()
    loss_direct.backward()

    means_lookup = _clone_leaf(means)
    conics_lookup = _clone_leaf(conics)
    weights_lookup = _clone_leaf(weights)
    lookup_lookup = _clone_leaf(lookup)
    opacities_lookup = _clone_leaf(opacities)
    result = rasterize_projected_gaussians_feature_lookup(
        means_lookup,
        conics_lookup,
        weights_lookup,
        lookup_lookup,
        opacities_lookup,
        depths,
        config,
    )
    loss_lookup = result.features.square().mean() + 0.17 * result.alpha.square().mean()
    loss_lookup.backward()

    checks = {
        "features": _max_abs(out_direct, result.features),
        "alpha": _max_abs(alpha_direct, result.alpha),
        "loss": abs(float(loss_direct.detach().cpu()) - float(loss_lookup.detach().cpu())),
        "grad_means": _max_abs(means_direct.grad, means_lookup.grad),
        "grad_conics": _max_abs(conics_direct.grad, conics_lookup.grad),
        "grad_weights": _max_abs(weights_direct.grad, weights_lookup.grad),
        "grad_lookup": _max_abs(lookup_direct.grad, lookup_lookup.grad),
        "grad_opacities": _max_abs(opacities_direct.grad, opacities_lookup.grad),
    }
    for name, value in checks.items():
        print(f"{name} max_abs={value:.8g}")
    bad = {name: value for name, value in checks.items() if value > 2.0e-5}
    if bad:
        raise AssertionError(f"feature lookup parity failed: {bad}")
    print("feature lookup direct parity: ok")


def test_feature_id_skeleton_matches_dense_coefficients() -> None:
    device = _device()
    means, conics, _weights, lookup, opacities, depths, config = _make_case(device)
    feature_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [0, 3], [2, 1], [1, 0], [3, 2]],
        device=device,
        dtype=torch.int64,
    )
    feature_id_weights = (torch.randn(feature_ids.shape, device=device) * 0.2).requires_grad_(True)
    lookup_dense = _clone_leaf(lookup)
    coeffs = feature_ids_to_coefficients(feature_ids, feature_id_weights, int(lookup.shape[0]))

    dense_result = rasterize_projected_gaussians_feature_lookup(
        means,
        conics,
        coeffs,
        lookup_dense,
        opacities,
        depths,
        config,
    )
    id_result = rasterize_projected_gaussians_feature_ids(
        means,
        conics,
        feature_ids,
        feature_id_weights,
        lookup_dense,
        opacities,
        depths,
        config,
    )
    feature_diff = _max_abs(dense_result.features, id_result.features)
    alpha_diff = _max_abs(dense_result.alpha, id_result.alpha)
    print(f"id_skeleton feature max_abs={feature_diff:.8g}")
    print(f"id_skeleton alpha max_abs={alpha_diff:.8g}")
    if feature_diff > 1.0e-6 or alpha_diff > 1.0e-6:
        raise AssertionError("feature ID skeleton does not match dense coefficients")
    print("feature id skeleton parity: ok")


if __name__ == "__main__":
    test_feature_lookup_matches_direct_features_and_gradients()
    test_feature_id_skeleton_matches_dense_coefficients()
