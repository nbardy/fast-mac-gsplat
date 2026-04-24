from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v7 import RasterConfig, rasterize_projected_gaussians


def sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def make_case(B, G, H, W, device, seed):
    torch.manual_seed(seed)
    means2d = torch.rand(B, G, 2, device=device) * torch.tensor([W - 1, H - 1], device=device)
    sigma = 1.0 + 4.0 * torch.rand(B, G, device=device)
    theta = 2.0 * torch.pi * torch.rand(B, G, device=device)
    ct = torch.cos(theta); st = torch.sin(theta)
    a = ct * ct / (sigma * sigma) + st * st / (sigma * sigma)
    b = torch.zeros_like(a)
    c = a.clone()
    conics = torch.stack([a, b, c], dim=-1)
    colors = torch.rand(B, G, 3, device=device)
    opacities = 0.2 + 0.7 * torch.rand(B, G, device=device)
    depths = torch.rand(B, G, device=device)
    return means2d, conics, colors, opacities, depths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--gaussians', type=int, default=8192)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--iters', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--backward', action='store_true')
    p.add_argument('--json', action='store_true')
    args = p.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit('MPS is not available.')
    device = torch.device('mps')
    cfg = RasterConfig(height=args.height, width=args.width)
    means2d, conics, colors, opacities, depths = make_case(args.batch_size, args.gaussians, args.height, args.width, device, args.seed)
    if args.backward:
        means2d.requires_grad_(True)
        conics.requires_grad_(True)
        colors.requires_grad_(True)
        opacities.requires_grad_(True)

    def step():
        sync(); t0 = time.perf_counter()
        out = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
        sync(); t1 = time.perf_counter()
        if args.backward:
            out.square().mean().backward()
            for t in (means2d, conics, colors, opacities):
                if t.grad is not None:
                    t.grad.zero_()
        sync(); t2 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0

    for _ in range(args.warmup):
        step()

    forward_times, backward_times, total_times = [], [], []
    for _ in range(args.iters):
        f_ms, b_ms = step()
        forward_times.append(f_ms)
        backward_times.append(b_ms)
        total_times.append(f_ms + b_ms)
    result = {
        'height': args.height,
        'width': args.width,
        'gaussians': args.gaussians,
        'batch_size': args.batch_size,
        'backward': bool(args.backward),
        'mean_ms': sum(total_times) / len(total_times),
        'median_ms': float(median(total_times)),
        'min_ms': min(total_times),
        'max_ms': max(total_times),
        'forward_ms': sum(forward_times) / len(forward_times),
        'backward_ms': sum(backward_times) / len(backward_times),
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result)


if __name__ == '__main__':
    main()
