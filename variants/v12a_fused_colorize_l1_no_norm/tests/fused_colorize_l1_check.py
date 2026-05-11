from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm import fused_no_norm_l1_grad


def _reference(
    features: torch.Tensor,
    alpha: torch.Tensor,
    target_rgb: torch.Tensor,
    background_rgb: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
):
    f = features.detach().clone().requires_grad_(True)
    a = alpha.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True)
    logits = F.conv2d(f.permute(0, 3, 1, 2), w.view(3, w.shape[1], 1, 1), b)
    splat_rgb = torch.sigmoid(logits)
    pred = a.unsqueeze(1) * splat_rgb + (1.0 - a.unsqueeze(1)) * background_rgb
    per_image = (pred - target_rgb).abs().flatten(1).mean(dim=1)
    per_image.mean().backward()
    torch.mps.synchronize()
    return {
        "loss_per_image": per_image.detach(),
        "grad_features": f.grad.detach(),
        "grad_alpha": a.grad.detach(),
        "grad_weight": w.grad.detach(),
        "grad_bias": b.grad.detach(),
    }


def _make_case(n: int, h: int, w: int, *, zero_loss: bool = False):
    device = torch.device("mps")
    feature_dim = 32
    features = torch.randn(n, h, w, feature_dim, device=device, dtype=torch.float32) * 0.25
    alpha = torch.rand(n, h, w, device=device, dtype=torch.float32).mul_(0.9).add_(0.05)
    if zero_loss:
        weight = torch.zeros(3, feature_dim, device=device, dtype=torch.float32)
        bias = torch.zeros(3, device=device, dtype=torch.float32)
        background_rgb = torch.full((n, 3, h, w), 0.5, device=device, dtype=torch.float32)
        target_rgb = torch.full((n, 3, h, w), 0.5, device=device, dtype=torch.float32)
    else:
        weight = torch.randn(3, feature_dim, device=device, dtype=torch.float32) * 0.12
        bias = torch.randn(3, device=device, dtype=torch.float32) * 0.05
        background_rgb = torch.rand(n, 3, h, w, device=device, dtype=torch.float32)
        target_rgb = torch.rand(n, 3, h, w, device=device, dtype=torch.float32)
    return features.contiguous(), alpha.contiguous(), target_rgb.contiguous(), background_rgb.contiguous(), weight.contiguous(), bias.contiguous()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().detach().cpu().item())


def _run_case(name: str, n: int, h: int, w: int, *, zero_loss: bool = False) -> dict[str, float | str]:
    tensors = _make_case(n, h, w, zero_loss=zero_loss)
    ref = _reference(*tensors)
    got_values = fused_no_norm_l1_grad(*tensors)
    torch.mps.synchronize()
    got = {
        "loss_per_image": got_values[0],
        "grad_features": got_values[1],
        "grad_alpha": got_values[2],
        "grad_weight": got_values[3],
        "grad_bias": got_values[4],
    }
    diffs = {key: _max_abs(got[key], ref[key]) for key in ref}
    result = {"case": name, **diffs}
    tol = 2.0e-4
    if zero_loss:
        tol = 1.0e-6
    failed = {key: value for key, value in diffs.items() if value > tol}
    if failed:
        raise AssertionError(f"{name} failed tolerance {tol}: {failed}")
    return result


def main() -> None:
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")
    torch.manual_seed(1234)
    results = [
        _run_case("tiny", 1, 4, 5),
        _run_case("small_batch", 2, 16, 16),
        _run_case("zero_l1_subgradient", 1, 8, 8, zero_loss=True),
    ]
    print(json.dumps({"status": "ok", "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
