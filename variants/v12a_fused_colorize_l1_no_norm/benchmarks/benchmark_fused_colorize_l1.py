from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm import fused_no_norm_l1_grad  # noqa: E402


def _sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _make_inputs(
    *,
    n_images: int,
    height: int,
    width: int,
    feature_dim: int,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the v12a fused Metal benchmark")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    features = torch.randn((n_images, height, width, feature_dim), generator=gen, dtype=torch.float32).to("mps")
    alpha = torch.rand((n_images, height, width), generator=gen, dtype=torch.float32).to("mps")
    target = torch.rand((n_images, 3, height, width), generator=gen, dtype=torch.float32).to("mps")
    background = torch.rand((n_images, 3, height, width), generator=gen, dtype=torch.float32).to("mps")
    weight = (torch.randn((3, feature_dim), generator=gen, dtype=torch.float32) * 0.2).to("mps")
    bias = (torch.randn((3,), generator=gen, dtype=torch.float32) * 0.1).to("mps")
    return features, alpha, target, background, weight, bias


def _torch_einsum_loss_and_backward(inputs: tuple[torch.Tensor, ...]) -> tuple[float, tuple[torch.Tensor, ...]]:
    features, alpha, target, background, weight, bias = (x.detach().clone().requires_grad_(i in {0, 1, 4, 5}) for i, x in enumerate(inputs))
    logits = torch.einsum("nhwf,cf->nhwc", features, weight) + bias.view(1, 1, 1, 3)
    rgb = torch.sigmoid(logits)
    background_nhwc = background.permute(0, 2, 3, 1).contiguous()
    target_nhwc = target.permute(0, 2, 3, 1).contiguous()
    pred = alpha.unsqueeze(-1) * rgb + (1.0 - alpha.unsqueeze(-1)) * background_nhwc
    loss = (pred - target_nhwc).abs().mean()
    loss.backward()
    return float(loss.detach().cpu()), (features.grad, alpha.grad, weight.grad, bias.grad)


def _torch_conv_loss_and_backward(inputs: tuple[torch.Tensor, ...]) -> tuple[float, tuple[torch.Tensor, ...]]:
    features_nhwf, alpha, target, background, weight, bias = (
        x.detach().clone().requires_grad_(i in {0, 1, 4, 5}) for i, x in enumerate(inputs)
    )
    # This mirrors the current trainer boundary: fast-mac raster output is NHWF,
    # then Python permutes to NCHW before FeatureToColor's 1x1 Conv2d.
    features_nfhw = features_nhwf.permute(0, 3, 1, 2).contiguous()
    logits = F.conv2d(features_nfhw, weight[:, :, None, None], bias)
    rgb = torch.sigmoid(logits)
    pred = alpha.unsqueeze(1) * rgb + (1.0 - alpha.unsqueeze(1)) * background
    loss = (pred - target).abs().mean()
    loss.backward()
    return float(loss.detach().cpu()), (features_nhwf.grad, alpha.grad, weight.grad, bias.grad)


def _fused_loss_and_grads(inputs: tuple[torch.Tensor, ...]) -> tuple[float, tuple[torch.Tensor, ...]]:
    loss_per_image, grad_features, grad_alpha, grad_weight, grad_bias = fused_no_norm_l1_grad(*inputs)
    loss = float(loss_per_image.mean().detach().cpu())
    return loss, (grad_features, grad_alpha, grad_weight, grad_bias)


def _time_call(fn, inputs: tuple[torch.Tensor, ...], *, iters: int, warmup: int) -> tuple[list[float], float, tuple[torch.Tensor, ...]]:
    last_loss = 0.0
    last_grads: tuple[torch.Tensor, ...] = ()
    for _ in range(warmup):
        last_loss, last_grads = fn(inputs)
    _sync()

    timings: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        last_loss, last_grads = fn(inputs)
        _sync()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings, last_loss, last_grads


def _parity(inputs: tuple[torch.Tensor, ...]) -> dict[str, float]:
    torch_loss, torch_grads = _torch_conv_loss_and_backward(inputs)
    _sync()
    fused_loss, fused_grads = _fused_loss_and_grads(inputs)
    _sync()
    names = ("features", "alpha", "weight", "bias")
    out = {"loss_abs": abs(torch_loss - fused_loss)}
    for name, ref, actual in zip(names, torch_grads, fused_grads, strict=True):
        out[f"{name}_max_abs"] = float((ref.detach() - actual.detach()).abs().max().cpu())
    return out


def _summarize(timings: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark the v12a fused no-norm colorize+compose+L1 gradient producer.")
    p.add_argument("--n-images", "--images", dest="n_images", type=int, default=16)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true", help="Accepted for README compatibility; parity runs unless --skip-parity is set.")
    p.add_argument("--skip-torch", action="store_true", help="Benchmark only the fused producer.")
    p.add_argument("--skip-parity", action="store_true", help="Skip PyTorch parity, useful for large fused-only rows.")
    p.add_argument("--include-einsum", action="store_true", help="Also time the NHWF torch.einsum reference.")
    p.add_argument("--json-output", type=Path)
    args = p.parse_args()

    inputs = _make_inputs(
        n_images=args.n_images,
        height=args.height,
        width=args.width,
        feature_dim=args.feature_dim,
        seed=args.seed,
    )
    fused_timings, fused_loss, _ = _time_call(
        _fused_loss_and_grads,
        tuple(x.detach() for x in inputs),
        iters=args.iters,
        warmup=args.warmup,
    )
    result: dict[str, object] = {
        "case": {
            "n_images": args.n_images,
            "height": args.height,
            "width": args.width,
            "feature_dim": args.feature_dim,
            "iters": args.iters,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "fused": _summarize(fused_timings),
        "fused_loss": fused_loss,
    }
    if not args.skip_parity:
        result["parity"] = _parity(tuple(x.detach() for x in inputs))

    if not args.skip_torch:
        torch_timings, torch_loss, _ = _time_call(
            _torch_conv_loss_and_backward,
            tuple(x.detach() for x in inputs),
            iters=args.iters,
            warmup=args.warmup,
        )
        result["torch_conv_autograd"] = _summarize(torch_timings)
        result["torch_loss"] = torch_loss
        result["speedup_vs_torch_conv_median"] = result["torch_conv_autograd"]["median_ms"] / result["fused"]["median_ms"]  # type: ignore[index]
        if args.include_einsum:
            einsum_timings, einsum_loss, _ = _time_call(
                _torch_einsum_loss_and_backward,
                tuple(x.detach() for x in inputs),
                iters=args.iters,
                warmup=args.warmup,
            )
            result["torch_einsum_autograd"] = _summarize(einsum_timings)
            result["torch_einsum_loss"] = einsum_loss
            result["speedup_vs_torch_einsum_median"] = result["torch_einsum_autograd"]["median_ms"] / result["fused"]["median_ms"]  # type: ignore[index]

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")


if __name__ == "__main__":
    main()
