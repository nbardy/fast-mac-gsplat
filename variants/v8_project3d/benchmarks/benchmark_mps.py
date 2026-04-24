from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v8_project3d import (
    RasterConfig,
    get_runtime_shader_config,
    profile_projected_gaussians,
    rasterize_projected_gaussians,
)


def make_case(case: str, B: int, G: int, H: int, W: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    if case == "sparse_sigma_1_5":
        means2d = torch.rand(B, G, 2, device=device, dtype=torch.float32)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sig = torch.rand(B, G, 2, device=device) * 4.0 + 1.0
    elif case == "medium_sigma_3_8":
        means2d = torch.rand(B, G, 2, device=device, dtype=torch.float32)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sig = torch.rand(B, G, 2, device=device) * 5.0 + 3.0
    elif case == "heavy_sigma_8_24":
        means2d = torch.rand(B, G, 2, device=device, dtype=torch.float32)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sig = torch.rand(B, G, 2, device=device) * 16.0 + 8.0
    elif case == "center_hotspot":
        means2d = torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([W * 0.06, H * 0.06], device=device)
        means2d = means2d + torch.tensor([W * 0.5, H * 0.5], device=device)
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 8.0 + 4.0
    elif case == "quadrant_cluster":
        means2d = torch.rand(B, G, 2, device=device, dtype=torch.float32)
        means2d[..., 0] *= W * 0.25
        means2d[..., 1] *= H * 0.25
        sig = torch.rand(B, G, 2, device=device) * 10.0 + 6.0
    elif case == "overflow_stress":
        means2d = torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([W * 0.01, H * 0.01], device=device)
        means2d = means2d + torch.tensor([W * 0.5, H * 0.5], device=device)
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 48.0 + 24.0
    else:
        raise ValueError(f"unknown case: {case}")
    conics = torch.stack(
        [
            1.0 / torch.clamp(sig[..., 0] * sig[..., 0], min=1e-8),
            torch.zeros(B, G, device=device),
            1.0 / torch.clamp(sig[..., 1] * sig[..., 1], min=1e-8),
        ],
        dim=-1,
    ).to(torch.float32)
    colors = torch.rand(B, G, 3, device=device, dtype=torch.float32)
    opacities = torch.rand(B, G, device=device, dtype=torch.float32).mul_(0.7).add_(0.1)
    depths = torch.rand(B, G, device=device, dtype=torch.float32)
    return means2d, conics, colors, opacities, depths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--gaussians", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--case", type=str, default="medium_sigma_3_8")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--batch-strategy", type=str, default="auto")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")
    device = torch.device("mps")
    means2d, conics, colors, opacities, depths = make_case(args.case, args.batch_size, args.gaussians, args.height, args.width, device, args.seed)
    if args.backward:
        means2d.requires_grad_(True)
        conics.requires_grad_(True)
        colors.requires_grad_(True)
        opacities.requires_grad_(True)

    rt = get_runtime_shader_config()
    cfg = RasterConfig(
        height=args.height,
        width=args.width,
        tile_size=rt.tile_size,
        max_fast_pairs=rt.fast_cap,
        batch_strategy=args.batch_strategy,
    )

    profile_stats = None
    if args.profile:
        profile_stats = profile_projected_gaussians(
            means2d.detach(), conics.detach(), colors.detach(), opacities.detach(), depths.detach(), cfg, run_forward=True, return_image=False
        )

    def step():
        if args.backward:
            t0 = time.perf_counter()
            out = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
            torch.mps.synchronize()
            t1 = time.perf_counter()
            loss = out.square().mean()
            loss.backward()
            torch.mps.synchronize()
            t2 = time.perf_counter()
            for t in (means2d, conics, colors, opacities):
                if t.grad is not None:
                    t.grad.zero_()
            return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
        else:
            t0 = time.perf_counter()
            _ = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
            torch.mps.synchronize()
            t1 = time.perf_counter()
            return (t1 - t0) * 1000.0, 0.0

    for _ in range(args.warmup):
        step()

    total_times, forward_times, backward_times = [], [], []
    for _ in range(args.iters):
        f_ms, b_ms = step()
        forward_times.append(f_ms)
        backward_times.append(b_ms)
        total_times.append(f_ms + b_ms)

    result = {
        "case": args.case,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "gaussians": args.gaussians,
        "batch_size": args.batch_size,
        "batch_strategy": args.batch_strategy,
        "backward": bool(args.backward),
        "mean_ms": sum(total_times) / len(total_times),
        "median_ms": float(median(total_times)),
        "min_ms": min(total_times),
        "max_ms": max(total_times),
        "forward_ms": sum(forward_times) / len(forward_times),
        "backward_ms": sum(backward_times) / len(backward_times),
        "runtime_tile_size": rt.tile_size,
        "runtime_chunk": rt.chunk_size,
        "runtime_fast_cap": rt.fast_cap,
    }
    if profile_stats is not None:
        result.update({f"profile_{k}": v for k, v in profile_stats.items()})

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        kind = "forward_backward" if args.backward else "forward"
        print(
            f"case={args.case} B={args.batch_size} strat={args.batch_strategy} {kind} "
            f"mean_ms={result['mean_ms']:.3f} median_ms={result['median_ms']:.3f} "
            f"fwd_ms={result['forward_ms']:.3f} bwd_ms={result['backward_ms']:.3f} "
            f"tile={rt.tile_size} chunk={rt.chunk_size} cap={rt.fast_cap}"
        )
        if profile_stats is not None:
            print(json.dumps(profile_stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
