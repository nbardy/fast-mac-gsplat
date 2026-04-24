from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v6 import (  # noqa: E402
    RasterConfig,
    get_runtime_shader_config,
    profile_projected_gaussians,
    rasterize_projected_gaussians,
)


def _cpu_gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _rand(shape, gen: torch.Generator, *, low=0.0, high=1.0) -> torch.Tensor:
    t = torch.rand(shape, generator=gen, dtype=torch.float32)
    return t.mul_(high - low).add_(low)


def _sig_to_conics(sig_x: torch.Tensor, sig_y: torch.Tensor, corr: torch.Tensor | None = None) -> torch.Tensor:
    inv_x = 1.0 / torch.clamp(sig_x.square(), min=1e-4)
    inv_y = 1.0 / torch.clamp(sig_y.square(), min=1e-4)
    if corr is None:
        corr = torch.zeros_like(inv_x)
    return torch.stack([inv_x, corr, inv_y], dim=-1).to(torch.float32)


def _move_to_device(device: torch.device, *xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(x.to(device).contiguous() for x in xs)


def _load_trace(trace_file: str, batch_size: int, gaussians: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    data = torch.load(trace_file, map_location="cpu")
    keys = ["means2d", "conics", "colors", "opacities", "depths"]
    if not all(k in data for k in keys):
        missing = [k for k in keys if k not in data]
        raise KeyError(f"trace file missing keys: {missing}")
    means2d, conics, colors, opacities, depths = (data[k].to(torch.float32) for k in keys)
    if means2d.ndim == 2:
        means2d = means2d.unsqueeze(0)
        conics = conics.unsqueeze(0)
        colors = colors.unsqueeze(0)
        opacities = opacities.unsqueeze(0)
        depths = depths.unsqueeze(0)
    B0, G0 = means2d.shape[:2]
    if batch_size <= B0:
        means2d = means2d[:batch_size]
        conics = conics[:batch_size]
        colors = colors[:batch_size]
        opacities = opacities[:batch_size]
        depths = depths[:batch_size]
    else:
        reps = (batch_size + B0 - 1) // B0
        means2d = means2d.repeat((reps, 1, 1))[:batch_size]
        conics = conics.repeat((reps, 1, 1))[:batch_size]
        colors = colors.repeat((reps, 1, 1))[:batch_size]
        opacities = opacities.repeat((reps, 1))[:batch_size]
        depths = depths.repeat((reps, 1))[:batch_size]
    if gaussians <= G0:
        means2d = means2d[:, :gaussians]
        conics = conics[:, :gaussians]
        colors = colors[:, :gaussians]
        opacities = opacities[:, :gaussians]
        depths = depths[:, :gaussians]
    return _move_to_device(device, means2d, conics, colors, opacities, depths)


def make_case(case: str, B: int, G: int, H: int, W: int, device: torch.device, seed: int, trace_file: str = ""):
    if case == "real_trace":
        if not trace_file:
            raise ValueError("case=real_trace requires --trace-file")
        return _load_trace(trace_file, B, G, device)

    gen = _cpu_gen(seed)

    if case in ("uniform_random", "medium_sigma_3_8"):
        means2d = _rand((B, G, 2), gen)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sx = _rand((B, G), gen, low=3.0, high=8.0)
        sy = _rand((B, G), gen, low=3.0, high=8.0)
        corr = torch.zeros((B, G), dtype=torch.float32)
    elif case == "sparse_sigma_1_5":
        means2d = _rand((B, G, 2), gen)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sx = _rand((B, G), gen, low=1.0, high=5.0)
        sy = _rand((B, G), gen, low=1.0, high=5.0)
        corr = torch.zeros((B, G), dtype=torch.float32)
    elif case == "heavy_sigma_8_24":
        means2d = _rand((B, G, 2), gen)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sx = _rand((B, G), gen, low=8.0, high=24.0)
        sy = _rand((B, G), gen, low=8.0, high=24.0)
        corr = torch.zeros((B, G), dtype=torch.float32)
    elif case == "sparse_screen":
        anchors = torch.tensor(
            [[0.15, 0.15], [0.15, 0.75], [0.5, 0.25], [0.78, 0.72]], dtype=torch.float32
        )
        assign = torch.randint(0, anchors.shape[0], (B, G), generator=gen)
        means2d = anchors[assign] + torch.randn((B, G, 2), generator=gen, dtype=torch.float32) * torch.tensor([0.03, 0.03])
        means2d[..., 0].mul_(W).clamp_(0, W - 1)
        means2d[..., 1].mul_(H).clamp_(0, H - 1)
        sx = _rand((B, G), gen, low=1.0, high=4.0)
        sy = _rand((B, G), gen, low=1.0, high=4.0)
        corr = _rand((B, G), gen, low=-0.05, high=0.05)
    elif case == "clustered_hot_tiles":
        anchors = torch.tensor([[0.45, 0.45], [0.55, 0.55]], dtype=torch.float32)
        assign = torch.randint(0, anchors.shape[0], (B, G), generator=gen)
        means2d = anchors[assign] + torch.randn((B, G, 2), generator=gen, dtype=torch.float32) * torch.tensor([0.015, 0.015])
        means2d[..., 0].mul_(W).clamp_(0, W - 1)
        means2d[..., 1].mul_(H).clamp_(0, H - 1)
        sx = _rand((B, G), gen, low=3.0, high=8.0)
        sy = _rand((B, G), gen, low=3.0, high=8.0)
        corr = _rand((B, G), gen, low=-0.1, high=0.1)
    elif case == "layered_depth":
        anchors = torch.tensor([[0.5, 0.5], [0.52, 0.48], [0.48, 0.52]], dtype=torch.float32)
        assign = torch.randint(0, anchors.shape[0], (B, G), generator=gen)
        means2d = anchors[assign] + torch.randn((B, G, 2), generator=gen, dtype=torch.float32) * torch.tensor([0.02, 0.02])
        means2d[..., 0].mul_(W).clamp_(0, W - 1)
        means2d[..., 1].mul_(H).clamp_(0, H - 1)
        sx = _rand((B, G), gen, low=4.0, high=9.0)
        sy = _rand((B, G), gen, low=4.0, high=9.0)
        corr = _rand((B, G), gen, low=-0.08, high=0.08)
    elif case in ("overflow_adversarial", "overflow_stress"):
        centers = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        assign = torch.zeros((B, G), dtype=torch.long)
        means2d = centers[assign] + torch.randn((B, G, 2), generator=gen, dtype=torch.float32) * torch.tensor([0.008, 0.008])
        means2d[..., 0].mul_(W).clamp_(0, W - 1)
        means2d[..., 1].mul_(H).clamp_(0, H - 1)
        sx = _rand((B, G), gen, low=24.0, high=72.0)
        sy = _rand((B, G), gen, low=24.0, high=72.0)
        corr = _rand((B, G), gen, low=-0.12, high=0.12)
    elif case == "temporal_adjacent":
        base = _rand((G, 2), gen)
        base[:, 0] *= W
        base[:, 1] *= H
        vel = torch.randn((G, 2), generator=gen, dtype=torch.float32)
        vel[:, 0] *= W * 0.003
        vel[:, 1] *= H * 0.003
        means_list = []
        for b in range(B):
            t = float(b)
            jitter = torch.randn((G, 2), generator=gen, dtype=torch.float32)
            jitter[:, 0] *= W * 0.001
            jitter[:, 1] *= H * 0.001
            m = base + t * vel + jitter
            m[:, 0].clamp_(0, W - 1)
            m[:, 1].clamp_(0, H - 1)
            means_list.append(m)
        means2d = torch.stack(means_list, dim=0)
        sx = _rand((B, G), gen, low=2.0, high=6.0)
        sy = _rand((B, G), gen, low=2.0, high=6.0)
        corr = _rand((B, G), gen, low=-0.05, high=0.05)
    else:
        raise ValueError(f"unknown case: {case}")

    conics = _sig_to_conics(sx, sy, corr)
    colors = _rand((B, G, 3), gen)
    opacities = _rand((B, G), gen, low=0.1, high=0.8)
    if case == "layered_depth":
        layers = torch.randint(0, 8, (B, G), generator=gen, dtype=torch.int64).to(torch.float32)
        depths = (layers / 8.0) + _rand((B, G), gen, low=0.0, high=0.02)
    elif case == "temporal_adjacent":
        base_depth = _rand((G,), gen)
        depths = torch.stack([torch.clamp(base_depth + 0.01 * b, 0.0, 1.0) for b in range(B)], dim=0)
    else:
        depths = _rand((B, G), gen)
    return _move_to_device(device, means2d, conics, colors, opacities, depths)


def make_config(args, rt) -> RasterConfig:
    use_active_override = None
    if args.active_tiles is not None:
        use_active_override = bool(args.active_tiles)
    return RasterConfig(
        height=args.height,
        width=args.width,
        tile_size=rt.tile_size,
        max_fast_pairs=args.max_fast_pairs if args.max_fast_pairs > 0 else rt.fast_cap,
        batch_strategy=args.batch_strategy,
        batch_launch_limit_tiles=args.batch_launch_limit_tiles,
        batch_launch_limit_gaussians=args.batch_launch_limit_gaussians,
        use_active_tiles=use_active_override,
        active_policy=args.active_policy,
        sort_active_tiles_by_count=bool(args.sort_active_tiles),
        active_sparse_fraction_threshold=args.active_sparse_fraction_threshold,
        active_dense_multiplier=args.active_dense_multiplier,
        stop_count_mode=args.stop_count_mode,
        stop_count_dense_threshold=args.dense_threshold,
    )


def benchmark_once(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    cfg: RasterConfig,
    *,
    warmup: int,
    iters: int,
    backward: bool,
) -> Dict[str, Any]:
    if backward:
        means2d.requires_grad_(True)
        conics.requires_grad_(True)
        colors.requires_grad_(True)
        opacities.requires_grad_(True)

    def step() -> Tuple[float, float]:
        if backward:
            t0 = time.perf_counter()
            out = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
            torch.mps.synchronize()
            t1 = time.perf_counter()
            out.square().mean().backward()
            torch.mps.synchronize()
            t2 = time.perf_counter()
            for t in (means2d, conics, colors, opacities):
                if t.grad is not None:
                    t.grad.zero_()
            return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
        t0 = time.perf_counter()
        _ = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0, 0.0

    for _ in range(warmup):
        step()

    totals, fwds, bwds = [], [], []
    for _ in range(iters):
        f_ms, b_ms = step()
        fwds.append(f_ms)
        bwds.append(b_ms)
        totals.append(f_ms + b_ms)

    return {
        "mean_ms": sum(totals) / len(totals),
        "median_ms": float(median(totals)),
        "min_ms": min(totals),
        "max_ms": max(totals),
        "forward_ms": sum(fwds) / len(fwds),
        "backward_ms": sum(bwds) / len(bwds),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--gaussians", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--case", type=str, default="uniform_random")
    p.add_argument("--trace-file", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--batch-strategy", type=str, default="auto")
    p.add_argument("--batch-launch-limit-tiles", type=int, default=262144)
    p.add_argument("--batch-launch-limit-gaussians", type=int, default=262144)
    p.add_argument("--stop-count-mode", type=str, default="adaptive")
    p.add_argument("--dense-threshold", type=int, default=64)
    p.add_argument("--active-policy", type=str, default="off")
    p.add_argument("--active-tiles", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--sort-active-tiles", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--active-sparse-fraction-threshold", type=float, default=0.45)
    p.add_argument("--active-dense-multiplier", type=float, default=2.0)
    p.add_argument("--max-fast-pairs", type=int, default=-1)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")
    device = torch.device("mps")
    rt = get_runtime_shader_config()
    means2d, conics, colors, opacities, depths = make_case(
        args.case, args.batch_size, args.gaussians, args.height, args.width, device, args.seed, args.trace_file
    )
    cfg = make_config(args, rt)

    profile_stats = None
    if args.profile:
        profile_stats = profile_projected_gaussians(
            means2d.detach(), conics.detach(), colors.detach(), opacities.detach(), depths.detach(), cfg, run_forward=True, return_image=False
        )

    timing = benchmark_once(
        means2d,
        conics,
        colors,
        opacities,
        depths,
        cfg,
        warmup=args.warmup,
        iters=args.iters,
        backward=bool(args.backward),
    )

    result: Dict[str, Any] = {
        "case": args.case,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "gaussians": args.gaussians,
        "batch_size": args.batch_size,
        "trace_file": args.trace_file,
        "batch_strategy": args.batch_strategy,
        "stop_count_mode": args.stop_count_mode,
        "dense_threshold": args.dense_threshold,
        "active_policy": args.active_policy,
        "active_tiles_override": args.active_tiles,
        "sort_active_tiles": bool(args.sort_active_tiles),
        "backward": bool(args.backward),
        "runtime_tile_size": rt.tile_size,
        "runtime_chunk": rt.chunk_size,
        "runtime_fast_cap": rt.fast_cap,
        **timing,
    }
    if profile_stats is not None:
        result.update({f"profile_{k}": v for k, v in profile_stats.items()})

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        kind = "forward_backward" if args.backward else "forward"
        print(
            f"case={args.case} B={args.batch_size} strat={args.batch_strategy} stop={args.stop_count_mode} {kind} "
            f"active_policy={args.active_policy} active_override={args.active_tiles} "
            f"mean_ms={result['mean_ms']:.3f} median_ms={result['median_ms']:.3f} "
            f"fwd_ms={result['forward_ms']:.3f} bwd_ms={result['backward_ms']:.3f} "
            f"tile={rt.tile_size} chunk={rt.chunk_size} cap={rt.fast_cap}"
        )
        if profile_stats is not None:
            print(json.dumps(profile_stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
