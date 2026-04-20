from __future__ import annotations

import argparse
import time
import torch

from torch_gsplat_bridge_v3 import RasterConfig, rasterize_projected_gaussians


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--gaussians", type=int, default=65536)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--backward", action="store_true")
    args = p.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    device = torch.device("mps")
    torch.manual_seed(0)
    G = args.gaussians
    means2d = torch.rand(G, 2, device=device, dtype=torch.float32)
    means2d[:, 0] *= args.width
    means2d[:, 1] *= args.height
    means2d.requires_grad_(args.backward)

    # Small projected radii by default; conic coefficients represent inverse covariance-ish precision.
    radii = torch.rand(G, 2, device=device, dtype=torch.float32) * 0.03 + 0.005
    conics = torch.stack([1.0 / radii[:, 0], torch.zeros(G, device=device), 1.0 / radii[:, 1]], dim=-1)
    conics.requires_grad_(args.backward)
    colors = torch.rand(G, 3, device=device, dtype=torch.float32, requires_grad=args.backward)
    opacities = torch.rand(G, device=device, dtype=torch.float32).mul_(0.7).add_(0.1)
    opacities.requires_grad_(args.backward)
    depths = torch.rand(G, device=device, dtype=torch.float32)

    cfg = RasterConfig(height=args.height, width=args.width)

    for _ in range(args.warmup):
        out = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
        if args.backward:
            loss = out.square().mean()
            loss.backward()
            for t in (means2d, conics, colors, opacities):
                if t.grad is not None:
                    t.grad.zero_()
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
        if args.backward:
            loss = out.square().mean()
            loss.backward()
            for t in (means2d, conics, colors, opacities):
                if t.grad is not None:
                    t.grad.zero_()
    torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / args.iters
    print(f"mean step time: {dt * 1000:.3f} ms")


if __name__ == "__main__":
    main()
