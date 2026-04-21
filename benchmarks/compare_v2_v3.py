from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = ROOT / "variants" / "v3"
V5_ROOT = ROOT / "variants" / "v5"
V6_ROOT = ROOT / "variants" / "v6"
V7_ROOT = ROOT / "variants" / "v7"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))
if str(V6_ROOT) not in sys.path:
    sys.path.insert(0, str(V6_ROOT))
if str(V7_ROOT) not in sys.path:
    sys.path.insert(0, str(V7_ROOT))

from torch_gsplat_bridge_fast import RasterConfig as RasterConfigV2
from torch_gsplat_bridge_fast import rasterize_projected_gaussians as rasterize_v2
from torch_gsplat_bridge_v3 import RasterConfig as RasterConfigV3
from torch_gsplat_bridge_v3 import rasterize_projected_gaussians as rasterize_v3
from torch_gsplat_bridge_v3.rasterize import _make_meta as make_meta_v3
from torch_gsplat_bridge_v5 import RasterConfig as RasterConfigV5
from torch_gsplat_bridge_v5 import profile_projected_gaussians as profile_v5
from torch_gsplat_bridge_v5 import rasterize_projected_gaussians as rasterize_v5
from torch_gsplat_bridge_v6 import RasterConfig as RasterConfigV6
from torch_gsplat_bridge_v6 import profile_projected_gaussians as profile_v6
from torch_gsplat_bridge_v6 import rasterize_projected_gaussians as rasterize_v6
from torch_gsplat_bridge_v7 import RasterConfig as RasterConfigV7
from torch_gsplat_bridge_v7 import rasterize_projected_gaussians as rasterize_v7


DEFAULT_BG = (1.0, 1.0, 1.0)


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


def dense_torch_reference(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    cfg: RasterConfigV3,
) -> torch.Tensor:
    perm = torch.argsort(depths.detach(), dim=0, stable=True)
    means = means2d.index_select(0, perm)
    conics_s = conics.index_select(0, perm)
    colors_s = colors.index_select(0, perm)
    opacities_s = opacities.index_select(0, perm)
    ys = torch.arange(cfg.height, dtype=means2d.dtype, device=means2d.device) + 0.5
    xs = torch.arange(cfg.width, dtype=means2d.dtype, device=means2d.device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    out = torch.zeros((cfg.height, cfg.width, 3), dtype=means2d.dtype, device=means2d.device)
    transmittance = torch.ones((cfg.height, cfg.width), dtype=means2d.dtype, device=means2d.device)
    bg = torch.tensor(cfg.background, dtype=means2d.dtype, device=means2d.device)

    for i in range(means.shape[0]):
        dx = xx - means[i, 0]
        dy = yy - means[i, 1]
        q = conics_s[i]
        power = -0.5 * (q[0] * dx * dx + 2.0 * q[1] * dx * dy + q[2] * dy * dy)
        raw_alpha = opacities_s[i] * torch.exp(power)
        alpha = torch.clamp(raw_alpha, max=0.99)
        alpha = torch.where(
            (power <= 0.0) & (alpha >= cfg.alpha_threshold),
            alpha,
            torch.zeros_like(alpha),
        )
        weight = transmittance * alpha
        out = out + weight[..., None] * colors_s[i]
        transmittance = transmittance * (1.0 - alpha)
    return out + transmittance[..., None] * bg


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
        "median_ms": float(median(elapsed)),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
    }


def time_renderer_batched(name: str, fn, cfg, inputs, *, warmup: int, iters: int, backward: bool) -> dict[str, float | str]:
    run_inputs = tuple(x.unsqueeze(0) for x in clone_inputs(inputs, backward=backward))
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
        "median_ms": float(median(elapsed)),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
    }


def print_v3_tile_stats(inputs, cfg: RasterConfigV3) -> None:
    means, conics, colors, opacities, depths = inputs
    meta_i32, meta_f32 = make_meta_v3(cfg, means.device)
    meta_i32 = meta_i32.clone()
    meta_i32[5] = means.shape[0]
    perm = torch.argsort(depths.detach(), dim=0, stable=True)
    means_s = means.index_select(0, perm).contiguous()
    conics_s = conics.index_select(0, perm).contiguous()
    colors_s = colors.index_select(0, perm).contiguous()
    opacities_s = opacities.index_select(0, perm).contiguous()
    tile_counts, _tile_offsets, binned_ids = torch.ops.gsplat_metal_v3.bin(
        means_s, conics_s, colors_s, opacities_s, meta_i32, meta_f32
    )
    sync()
    counts_f = tile_counts.float()
    overflow_tiles = (tile_counts > cfg.max_fast_pairs).sum()
    print(
        "v3_tile_stats "
        f"total_pairs={int(binned_ids.numel())} "
        f"max_tile_count={int(tile_counts.max().item())} "
        f"mean_tile_count={float(counts_f.mean().item()):.3f} "
        f"nonzero_tiles={int((tile_counts > 0).sum().item())} "
        f"overflow_tiles={int(overflow_tiles.item())}"
    )


def print_v5_tile_stats(inputs, cfg: RasterConfigV5) -> None:
    means, conics, colors, opacities, depths = inputs
    stats = profile_v5(
        means,
        conics,
        colors,
        opacities,
        depths,
        cfg,
        run_forward=True,
        return_image=False,
    )
    print(
        "v5_tile_stats "
        f"total_pairs={stats['total_pairs']} "
        f"max_tile_count={stats['max_pairs_per_tile']} "
        f"mean_tile_count={stats['mean_pairs_per_tile']:.3f} "
        f"p95_tile_count={stats['p95_pairs_per_tile']:.3f} "
        f"overflow_tiles={stats['overflow_tile_count']} "
        f"mean_stop_count={stats.get('mean_stop_count', 0.0):.3f} "
        f"p95_stop_ratio={stats.get('p95_stop_ratio', 0.0):.3f} "
        f"batch_chunk={stats['chosen_batch_chunk']}"
    )


def print_v6_tile_stats(inputs, cfg: RasterConfigV6) -> None:
    means, conics, colors, opacities, depths = inputs
    stats = profile_v6(
        means,
        conics,
        colors,
        opacities,
        depths,
        cfg,
        run_forward=True,
        return_image=False,
    )
    print(
        "v6_tile_stats "
        f"total_pairs={stats['total_pairs']} "
        f"max_tile_count={stats['max_pairs_per_tile']} "
        f"mean_tile_count={stats['mean_pairs_per_tile']:.3f} "
        f"p95_tile_count={stats['p95_pairs_per_tile']:.3f} "
        f"overflow_tiles={stats['overflow_tile_count']} "
        f"active_tiles={stats['active_tile_count']} "
        f"dense_active_tiles={stats['dense_active_tile_count']} "
        f"mean_stop_count={stats.get('mean_stop_count', 0.0):.3f} "
        f"p95_stop_ratio={stats.get('p95_stop_ratio', 0.0):.3f} "
        f"batch_chunk={stats['chosen_batch_chunk']} "
        f"stop_mode={stats['stop_count_mode']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare validated v2/v3 fastpaths against v5/v6.")
    parser.add_argument("--height", type=int, default=4096)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--gaussians", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--tile-stats", action="store_true", help="Print v3/v5/v6 tile occupancy stats for each case.")
    parser.add_argument("--include-v7", action="store_true", help="Also time the experimental v7 hardware renderer.")
    parser.add_argument("--include-torch-reference", action="store_true", help="Also time a direct Torch reference renderer.")
    parser.add_argument(
        "--torch-max-work-items",
        type=int,
        default=64_000_000,
        help="Skip direct Torch reference when height * width * gaussians exceeds this value.",
    )
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")

    cases = [
        Case("sparse_sigma_1_5", 1.0, 5.0),
        Case("medium_sigma_3_8", 3.0, 8.0),
    ]
    cfg_v2 = RasterConfigV2(height=args.height, width=args.width, background=DEFAULT_BG)
    cfg_v3 = RasterConfigV3(height=args.height, width=args.width, background=DEFAULT_BG)
    cfg_v5 = RasterConfigV5(height=args.height, width=args.width, background=DEFAULT_BG)
    cfg_v6 = RasterConfigV6(height=args.height, width=args.width, background=DEFAULT_BG)
    cfg_v7 = RasterConfigV7(height=args.height, width=args.width, background=DEFAULT_BG)

    print(
        f"height={args.height} width={args.width} gaussians={args.gaussians} "
        f"warmup={args.warmup} iters={args.iters} backward={args.backward}"
    )
    for case in cases:
        print(f"\ncase={case.name} sigma=[{case.sigma_min}, {case.sigma_max}]")
        inputs = make_inputs(args.height, args.width, args.gaussians, case.sigma_min, case.sigma_max, args.seed)
        if args.tile_stats:
            print_v3_tile_stats(inputs, cfg_v3)
            print_v5_tile_stats(inputs, cfg_v5)
            print_v6_tile_stats(inputs, cfg_v6)
        rows = [
            time_renderer("v2_fastpath", rasterize_v2, cfg_v2, inputs, warmup=args.warmup, iters=args.iters, backward=args.backward),
            time_renderer("v3_candidate", rasterize_v3, cfg_v3, inputs, warmup=args.warmup, iters=args.iters, backward=args.backward),
            time_renderer("v5_batched", rasterize_v5, cfg_v5, inputs, warmup=args.warmup, iters=args.iters, backward=args.backward),
            time_renderer("v6_direct", rasterize_v6, cfg_v6, inputs, warmup=args.warmup, iters=args.iters, backward=args.backward),
        ]
        if args.include_v7:
            rows.append(
                time_renderer_batched("v7_hardware", rasterize_v7, cfg_v7, inputs, warmup=args.warmup, iters=args.iters, backward=args.backward)
            )
        if args.include_torch_reference:
            work_items = args.height * args.width * args.gaussians
            if work_items <= args.torch_max_work_items:
                rows.append(
                    time_renderer(
                        "torch_direct",
                        dense_torch_reference,
                        cfg_v3,
                        inputs,
                        warmup=args.warmup,
                        iters=args.iters,
                        backward=args.backward,
                    )
                )
            else:
                print(
                    "torch_direct skipped "
                    f"work_items={work_items} exceeds torch_max_work_items={args.torch_max_work_items}"
                )
        base = float(rows[0]["mean_ms"])
        for row in rows:
            ratio = float(row["mean_ms"]) / base if base > 0 else float("nan")
            print(
                f"{row['renderer']:<14} {row['mode']:<16} "
                f"mean_ms={float(row['mean_ms']):>9.3f} "
                f"median_ms={float(row['median_ms']):>9.3f} "
                f"min_ms={float(row['min_ms']):>9.3f} "
                f"max_ms={float(row['max_ms']):>9.3f} "
                f"vs_v2={ratio:>6.3f}x"
            )


if __name__ == "__main__":
    main()
