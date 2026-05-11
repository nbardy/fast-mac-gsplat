from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_gsplat_bridge_v12c_fused_raster_color_loss_backward as v12c


def make_case(batch: int, gaussians: int, feature_dim: int, height: int, width: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    means_x = torch.rand((batch, gaussians), device=device, dtype=torch.float32) * (width - 2) + 1.0
    means_y = torch.rand((batch, gaussians), device=device, dtype=torch.float32) * (height - 2) + 1.0
    means2d = torch.stack([means_x, means_y], dim=-1).detach().requires_grad_(True)

    sigmas = torch.rand((batch, gaussians, 2), device=device, dtype=torch.float32) * 1.5 + 1.5
    conics = torch.stack(
        [
            1.0 / sigmas[..., 0].square(),
            torch.zeros((batch, gaussians), device=device, dtype=torch.float32),
            1.0 / sigmas[..., 1].square(),
        ],
        dim=-1,
    ).detach().requires_grad_(True)
    colors = (torch.randn((batch, gaussians, feature_dim), device=device, dtype=torch.float32) * 0.25).requires_grad_(True)
    opacities = (torch.rand((batch, gaussians), device=device, dtype=torch.float32) * 0.4 + 0.2).requires_grad_(True)
    depths = torch.rand((batch, gaussians), device=device, dtype=torch.float32)
    target = torch.rand((batch, height, width, 3), device=device, dtype=torch.float32)
    background = torch.rand((batch, height, width, 3), device=device, dtype=torch.float32) * 0.2
    color_weight = (torch.randn((3, feature_dim), device=device, dtype=torch.float32) * 0.4).requires_grad_(True)
    color_bias = (torch.randn((3,), device=device, dtype=torch.float32) * 0.1).requires_grad_(True)
    return means2d, conics, colors, opacities, depths, target, background, color_weight, color_bias


def clone_for_reference(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    out = []
    for tensor in tensors:
        clone = tensor.detach().clone()
        if tensor.requires_grad:
            clone.requires_grad_(True)
        out.append(clone)
    return tuple(out)


def reference_backward(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    target: torch.Tensor,
    background: torch.Tensor,
    color_weight: torch.Tensor,
    color_bias: torch.Tensor,
    cfg: v12c.RasterConfig,
) -> tuple[torch.Tensor, ...]:
    features, alpha = v12c.rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
    logits = torch.einsum("bhwf,cf->bhwc", features, color_weight) + color_bias.view(1, 1, 1, 3)
    splat_rgb = torch.sigmoid(logits)
    pred = alpha.unsqueeze(-1) * splat_rgb + (1.0 - alpha.unsqueeze(-1)) * background
    loss = (pred - target).square().mean()
    loss.backward()
    return (
        means2d.grad.detach(),
        conics.grad.detach(),
        colors.grad.detach(),
        opacities.grad.detach(),
        color_weight.grad.detach(),
        color_bias.grad.detach(),
    )


def assert_close(name: str, got: torch.Tensor, expected: torch.Tensor, threshold: float) -> None:
    err = float((got - expected).detach().abs().max().cpu())
    print(f"{name} max_abs={err:.8g}")
    if err > threshold:
        raise AssertionError(f"{name} max_abs {err} exceeded {threshold}")


def run_case(batch: int, feature_dim: int, *, active_policy: str = "off") -> None:
    device = torch.device("mps")
    height, width, gaussians = 18, 19, 32
    cfg = v12c.RasterConfig(
        height=height,
        width=width,
        tile_size=16,
        max_fast_pairs=512,
        enable_overflow_fallback=False,
        active_policy=active_policy,
    )
    case = make_case(batch, gaussians, feature_dim, height, width, device, seed=7000 + batch * 100 + feature_dim)
    ref_case = clone_for_reference(case)
    ref = reference_backward(*ref_case, cfg)
    fused = v12c.fused_linear_sigmoid_mse_backward(
        case[0],
        case[1],
        case[2],
        case[3],
        case[4],
        case[5],
        case[7],
        case[8],
        cfg,
        background_rgb=case[6],
    )
    torch.mps.synchronize()
    assert_close(f"B{batch} F{feature_dim} grad_means2d", fused.grad_means2d, ref[0], 3.0e-4)
    assert_close(f"B{batch} F{feature_dim} grad_conics", fused.grad_conics, ref[1], 3.0e-4)
    assert_close(f"B{batch} F{feature_dim} grad_colors", fused.grad_colors, ref[2], 3.0e-4)
    assert_close(f"B{batch} F{feature_dim} grad_opacities", fused.grad_opacities, ref[3], 3.0e-4)
    assert_close(f"B{batch} F{feature_dim} grad_color_weight", fused.grad_color_weight, ref[4], 3.0e-4)
    assert_close(f"B{batch} F{feature_dim} grad_color_bias", fused.grad_color_bias, ref[5], 3.0e-4)


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    run_case(batch=1, feature_dim=3)
    run_case(batch=2, feature_dim=8)
    run_case(batch=2, feature_dim=32)
    print("v12c fused linear sigmoid MSE parity: ok")


if __name__ == "__main__":
    main()
