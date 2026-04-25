from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v9_hw_tile_exact import (  # noqa: E402
    make_full_backward_config,
    probe_full_backward,
    rasterize_projected_gaussians_full_backward,
)


def make_case(batch_size: int, gaussians: int, height: int, width: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    means2d = torch.rand(batch_size, gaussians, 2, device=device, dtype=torch.float32)
    means2d[..., 0] *= width
    means2d[..., 1] *= height
    sigmas = torch.rand(batch_size, gaussians, 2, device=device, dtype=torch.float32) * 5.0 + 3.0
    conics = torch.stack(
        [
            1.0 / torch.clamp(sigmas[..., 0].square(), min=1.0e-4),
            torch.zeros(batch_size, gaussians, device=device, dtype=torch.float32),
            1.0 / torch.clamp(sigmas[..., 1].square(), min=1.0e-4),
        ],
        dim=-1,
    ).contiguous()
    colors = torch.rand(batch_size, gaussians, 3, device=device, dtype=torch.float32)
    opacities = torch.rand(batch_size, gaussians, device=device, dtype=torch.float32).mul_(0.7).add_(0.1)
    depths = torch.rand(batch_size, gaussians, device=device, dtype=torch.float32)
    return means2d, conics, colors, opacities, depths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--gaussians", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    status = probe_full_backward()
    if not status.available:
        raise SystemExit(f"full-backward backend unavailable: {status.as_dict()}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    device = torch.device("mps")
    means2d, conics, colors, opacities, depths = make_case(
        args.batch_size,
        args.gaussians,
        args.height,
        args.width,
        device,
        args.seed,
    )
    means2d.requires_grad_(True)
    conics.requires_grad_(True)
    colors.requires_grad_(True)
    opacities.requires_grad_(True)

    cfg = make_full_backward_config(
        height=args.height,
        width=args.width,
        tile_size=16,
        max_fast_pairs=2048,
        stop_count_mode="adaptive",
    )

    def step() -> tuple[float, float]:
        t0 = time.perf_counter()
        out = rasterize_projected_gaussians_full_backward(means2d, conics, colors, opacities, depths, cfg)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        out.square().mean().backward()
        torch.mps.synchronize()
        t2 = time.perf_counter()
        for tensor in (means2d, conics, colors, opacities):
            if tensor.grad is not None:
                tensor.grad.zero_()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    for _ in range(args.warmup):
        step()

    forward_ms: list[float] = []
    backward_ms: list[float] = []
    for _ in range(args.iters):
        f_ms, b_ms = step()
        forward_ms.append(f_ms)
        backward_ms.append(b_ms)

    total_ms = [f + b for f, b in zip(forward_ms, backward_ms)]
    row = {
        "backend": status.backend,
        "height": args.height,
        "width": args.width,
        "gaussians": args.gaussians,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "forward_median_ms": statistics.median(forward_ms),
        "backward_median_ms": statistics.median(backward_ms),
        "total_median_ms": statistics.median(total_ms),
        "forward_mean_ms": statistics.fmean(forward_ms),
        "backward_mean_ms": statistics.fmean(backward_ms),
        "total_mean_ms": statistics.fmean(total_ms),
    }
    print(json.dumps(row, sort_keys=True))
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
