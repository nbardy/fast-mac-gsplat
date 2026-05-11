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

from torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm import dssim_forward_grad  # noqa: E402


def _sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _local_mean(images: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size <= 1:
        return images
    pad = window_size // 2
    return F.avg_pool2d(F.pad(images, (pad, pad, pad, pad), mode="reflect"), kernel_size=window_size, stride=1)


def _torch_dssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int,
    c1: float,
    c2: float,
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    mu_x = _local_mean(prediction, window_size)
    mu_y = _local_mean(target, window_size)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x_sq = _local_mean(prediction.square(), window_size) - mu_x_sq
    sigma_y_sq = _local_mean(target.square(), window_size) - mu_y_sq
    sigma_xy = _local_mean(prediction * target, window_size) - mu_xy
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1.0e-12)
    ssim = ssim_map.flatten(1).mean(dim=1).clamp(-1.0, 1.0)
    return (1.0 - ssim) * 0.5


def _make_inputs(*, n_images: int, channels: int, height: int, width: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the Metal DSSIM benchmark")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    prediction = torch.rand((n_images, channels, height, width), generator=gen, dtype=torch.float32).to("mps")
    target = torch.rand((n_images, channels, height, width), generator=gen, dtype=torch.float32).to("mps")
    return prediction, target


def _torch_loss_and_grad(
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    window_size: int,
    c1: float,
    c2: float,
) -> tuple[float, torch.Tensor]:
    prediction, target = (x.detach().clone() for x in inputs)
    prediction.requires_grad_(True)
    loss_per_image = _torch_dssim_per_image(prediction, target, window_size=window_size, c1=c1, c2=c2)
    loss = loss_per_image.mean()
    loss.backward()
    return float(loss.detach().cpu()), prediction.grad.detach()


def _metal_loss_and_grad(
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    window_size: int,
    c1: float,
    c2: float,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    prediction, target = (x.detach() for x in inputs)
    loss_per_image, grad_prediction = dssim_forward_grad(
        prediction,
        target,
        window_size=window_size,
        c1=c1,
        c2=c2,
    )
    return float(loss_per_image.mean().detach().cpu()), loss_per_image, grad_prediction


def _time_call(fn, *, iters: int, warmup: int) -> tuple[list[float], object]:
    last: object = None
    for _ in range(warmup):
        last = fn()
    _sync()
    timings: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        last = fn()
        _sync()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings, last


def _summarize(timings: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def _parity(
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    window_size: int,
    c1: float,
    c2: float,
) -> dict[str, float]:
    torch_loss, torch_grad = _torch_loss_and_grad(inputs, window_size=window_size, c1=c1, c2=c2)
    _sync()
    metal_loss, _loss_per_image, metal_grad = _metal_loss_and_grad(inputs, window_size=window_size, c1=c1, c2=c2)
    _sync()
    diff = (torch_grad - metal_grad).abs()
    return {
        "loss_abs": abs(torch_loss - metal_loss),
        "grad_max_abs": float(diff.max().cpu()),
        "grad_mean_abs": float(diff.mean().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Metal DSSIM forward+gradient against PyTorch autograd.")
    parser.add_argument("--n-images", "--images", dest="n_images", type=int, default=16)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=11)
    parser.add_argument("--c1", type=float, default=0.01**2)
    parser.add_argument("--c2", type=float, default=0.03**2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    inputs = _make_inputs(
        n_images=args.n_images,
        channels=args.channels,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    result: dict[str, object] = {
        "case": {
            "n_images": args.n_images,
            "channels": args.channels,
            "height": args.height,
            "width": args.width,
            "window_size": args.window_size,
            "c1": args.c1,
            "c2": args.c2,
            "iters": args.iters,
            "warmup": args.warmup,
            "seed": args.seed,
        },
    }

    metal_timings, metal_last = _time_call(
        lambda: _metal_loss_and_grad(inputs, window_size=args.window_size, c1=args.c1, c2=args.c2),
        iters=args.iters,
        warmup=args.warmup,
    )
    result["metal_dssim_forward_grad"] = _summarize(metal_timings)
    result["metal_loss"] = metal_last[0]  # type: ignore[index]

    if not args.skip_parity:
        result["parity"] = _parity(inputs, window_size=args.window_size, c1=args.c1, c2=args.c2)

    if not args.skip_torch:
        torch_timings, torch_last = _time_call(
            lambda: _torch_loss_and_grad(inputs, window_size=args.window_size, c1=args.c1, c2=args.c2),
            iters=args.iters,
            warmup=args.warmup,
        )
        result["torch_dssim_autograd"] = _summarize(torch_timings)
        result["torch_loss"] = torch_last[0]  # type: ignore[index]
        result["speedup_vs_torch_median"] = (
            result["torch_dssim_autograd"]["median_ms"] / result["metal_dssim_forward_grad"]["median_ms"]  # type: ignore[index]
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
