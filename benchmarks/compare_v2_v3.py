from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = ROOT / "variants" / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from torch_gsplat_bridge_fast import RasterConfig as RasterConfigV2
from torch_gsplat_bridge_fast import rasterize_projected_gaussians as rasterize_v2
from torch_gsplat_bridge_v3 import RasterConfig as RasterConfigV3
from torch_gsplat_bridge_v3 import rasterize_projected_gaussians as rasterize_v3


@dataclass(frozen=True)
class Case:
    name: str
    sigma_min: float
    sigma_max: float


def sync() -> None:
    torch.mps.synchronize()


def make_inputs(height: int, width: int, gaussians: int, sigma_min: float, sigma_max: float, seed: int):
    device = torch.device("mps")
    torch.manual_seed(seed)
    means = torch.empty((gaussians, 2), device=device, dtype=torch.float32)
    means[:, 0].uniform_(0, width - 1)
    means[:, 1].uniform_(0, height - 1)
    sigma = torch.empty((gaussians,), device=device, dtype=torch.float32).uniform_(sigma_min, sigma_max)
    inv_var = 1.0 / (sigma * sigma)
    conics = torch.stack([inv_var, torch.zeros_like(inv_var), inv_var], dim=1).contiguous()
    colors = torch.rand((gaussians, 3), device=device, dtype=torch.float32).contiguous()
    opacities = torch.empty((gaussians,), device=device, dtype=torch.float32).uniform_(0.08, 0.85).contiguous()
    depths = torch.rand((gaussians,), device=device, dtype=torch.float32).contiguous()
    return means.contiguous(), conics, colors, opacities, depths


def clone_inputs(inputs, *, backward: bool):
    out = []
    for tensor in inputs:
        cloned = tensor.detach().clone().contiguous()
        cloned.requires_grad_(backward and tensor.ndim > 0 and tensor.shape[-1] != 0)
        out.append(cloned)
    if backward:
        out[-1].requires_grad_(False)
    return tuple(out)


def clear_grads(tensors) -> None:
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad.zero_()


def time_renderer(name: str, fn, cfg, inputs, *, warmup: int, iters: int, backward: bool) -> dict[str, float | str]:
    run_inputs = clone_inputs(inputs, backward=backward)
    for _ in range(warmup):
        out = fn(*run_inputs, cfg)
        if backward:
            loss = out.square().mean()
            loss.backward()
            clear_grads(run_inputs)
    sync()

    elapsed = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*run_inputs, cfg)
        if backward:
            loss = out.square().mean()
            loss.backward()
            clear_grads(run_inputs)
        sync()
        elapsed.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = sum(elapsed) / len(elapsed)
    return {
        "renderer": name,
        "mode": "forward_backward" if backward else "forward",
        "mean_ms": mean_ms,
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare validated v2 fastpath against v3 candidate.")
    parser.add_argument("--height", type=int, default=4096)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--gaussians", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    cases = [
        Case("sparse_sigma_1_5", 1.0, 5.0),
        Case("medium_sigma_3_8", 3.0, 8.0),
    ]
    cfg_v2 = RasterConfigV2(height=args.height, width=args.width, background=(1.0, 1.0, 1.0))
    cfg_v3 = RasterConfigV3(height=args.height, width=args.width, background=(1.0, 1.0, 1.0))

    print(
        f"height={args.height} width={args.width} gaussians={args.gaussians} "
        f"warmup={args.warmup} iters={args.iters} backward={args.backward}"
    )
    for case in cases:
        inputs = make_inputs(args.height, args.width, args.gaussians, case.sigma_min, case.sigma_max, args.seed)
        rows = [
            time_renderer("v2_fastpath", rasterize_v2, cfg_v2, inputs, warmup=args.warmup, iters=args.iters, backward=args.backward),
            time_renderer("v3_candidate", rasterize_v3, cfg_v3, inputs, warmup=args.warmup, iters=args.iters, backward=args.backward),
        ]
        print(f"\ncase={case.name} sigma=[{case.sigma_min}, {case.sigma_max}]")
        base = float(rows[0]["mean_ms"])
        for row in rows:
            ratio = float(row["mean_ms"]) / base if base > 0 else float("nan")
            print(
                f"{row['renderer']:<14} {row['mode']:<16} "
                f"mean_ms={float(row['mean_ms']):>9.3f} "
                f"min_ms={float(row['min_ms']):>9.3f} "
                f"max_ms={float(row['max_ms']):>9.3f} "
                f"vs_v2={ratio:>6.3f}x"
            )


if __name__ == "__main__":
    main()
