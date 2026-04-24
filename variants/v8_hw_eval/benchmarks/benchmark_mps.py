
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

from torch_gsplat_bridge_v8_hw_eval import (
    RasterConfig,
    get_runtime_shader_config,
    probe_hardware_eval,
    profile_projected_gaussians,
    rasterize_projected_gaussians,
)


def _load_trace(trace_file: str, device: torch.device, requested_b: int, requested_g: int):
    if not trace_file:
        raise ValueError("--trace-file is required when --case real_trace")
    data = torch.load(trace_file, map_location="cpu")
    tensors = []
    for key in ("means2d", "conics", "colors", "opacities", "depths"):
        if key not in data:
            raise ValueError(f"trace file is missing {key!r}")
        tensors.append(data[key].to(device=device, dtype=torch.float32))

    means2d, conics, colors, opacities, depths = tensors
    if means2d.ndim == 2:
        means2d = means2d.unsqueeze(0)
        conics = conics.unsqueeze(0)
        colors = colors.unsqueeze(0)
        opacities = opacities.unsqueeze(0)
        depths = depths.unsqueeze(0)
    B0, G0 = means2d.shape[:2]
    B = min(B0, requested_b) if requested_b > 0 else B0
    G = min(G0, requested_g) if requested_g > 0 else G0
    return means2d[:B, :G].contiguous(), conics[:B, :G].contiguous(), colors[:B, :G].contiguous(), opacities[:B, :G].contiguous(), depths[:B, :G].contiguous()


def make_case(case: str, B: int, G: int, H: int, W: int, device: torch.device, seed: int, trace_file: str = ""):
    torch.manual_seed(seed)
    random.seed(seed)
    if case == "real_trace":
        return _load_trace(trace_file, device, B, G)

    if case == "sparse_sigma_1_5":
        means2d = torch.rand(B, G, 2, device=device, dtype=torch.float32)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sig = torch.rand(B, G, 2, device=device) * 4.0 + 1.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case in ("medium_sigma_3_8", "uniform_random", "microbench_uniform_random"):
        means2d = torch.rand(B, G, 2, device=device, dtype=torch.float32)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sig = torch.rand(B, G, 2, device=device) * 5.0 + 3.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case == "heavy_sigma_8_24":
        means2d = torch.rand(B, G, 2, device=device, dtype=torch.float32)
        means2d[..., 0] *= W
        means2d[..., 1] *= H
        sig = torch.rand(B, G, 2, device=device) * 16.0 + 8.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case == "sparse_screen":
        centers = torch.tensor(
            [[0.22 * W, 0.24 * H], [0.76 * W, 0.30 * H], [0.55 * W, 0.76 * H], [0.24 * W, 0.68 * H]],
            device=device,
            dtype=torch.float32,
        )
        choices = torch.randint(0, centers.shape[0], (B, G), device=device)
        means2d = centers.index_select(0, choices.reshape(-1)).view(B, G, 2)
        jitter = torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([0.025 * W, 0.025 * H], device=device)
        means2d = means2d + jitter
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 4.0 + 2.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case == "center_hotspot":
        means2d = torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([W * 0.06, H * 0.06], device=device)
        means2d = means2d + torch.tensor([W * 0.5, H * 0.5], device=device)
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 8.0 + 4.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case == "clustered_hot_tiles":
        centers = torch.tensor([[0.50 * W, 0.50 * H], [0.53 * W, 0.48 * H]], device=device, dtype=torch.float32)
        choices = torch.randint(0, centers.shape[0], (B, G), device=device)
        means2d = centers.index_select(0, choices.reshape(-1)).view(B, G, 2)
        means2d = means2d + torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([0.018 * W, 0.018 * H], device=device)
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 8.0 + 4.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    elif case == "layered_depth":
        centers = torch.tensor([[0.38 * W, 0.40 * H], [0.58 * W, 0.55 * H], [0.48 * W, 0.70 * H]], device=device, dtype=torch.float32)
        choices = torch.randint(0, centers.shape[0], (B, G), device=device)
        means2d = centers.index_select(0, choices.reshape(-1)).view(B, G, 2)
        means2d = means2d + torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([0.045 * W, 0.045 * H], device=device)
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 10.0 + 4.0
        bands = torch.linspace(0.05, 0.95, 6, device=device, dtype=torch.float32)
        band_ids = torch.arange(G, device=device).remainder(bands.numel()).view(1, G).expand(B, G)
        depths = bands.index_select(0, band_ids.reshape(-1)).view(B, G)
        depths = (depths + torch.randn(B, G, device=device, dtype=torch.float32) * 0.01).clamp_(0.0, 1.0)
    elif case in ("overflow_stress", "overflow_adversarial"):
        means2d = torch.randn(B, G, 2, device=device, dtype=torch.float32) * torch.tensor([W * 0.01, H * 0.01], device=device)
        means2d = means2d + torch.tensor([W * 0.5, H * 0.5], device=device)
        means2d[..., 0].clamp_(0, W - 1)
        means2d[..., 1].clamp_(0, H - 1)
        sig = torch.rand(B, G, 2, device=device) * 48.0 + 24.0
        depths = torch.rand(B, G, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"unknown case: {case}")
    conics = torch.stack(
        [
            1.0 / torch.clamp(sig[..., 0].square(), min=1e-4),
            torch.zeros(B, G, device=device),
            1.0 / torch.clamp(sig[..., 1].square(), min=1e-4),
        ],
        dim=-1,
    ).to(torch.float32)
    colors = torch.rand(B, G, 3, device=device, dtype=torch.float32)
    opacities = torch.rand(B, G, device=device, dtype=torch.float32).mul_(0.7).add_(0.1)
    return means2d, conics, colors, opacities, depths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--gaussians", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--case", type=str, default="medium_sigma_3_8")
    p.add_argument("--trace-file", type=str, default="")
    p.add_argument("--temporal-batch-mode", type=str, default="random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--batch-strategy", type=str, default="auto")
    p.add_argument("--max-pairs-per-launch", type=int, default=0)
    p.add_argument("--stop-count-mode", type=str, default="adaptive")
    p.add_argument("--dense-threshold", type=int, default=64)
    p.add_argument("--active-tiles", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--active-policy", type=str, default="off", choices=("off", "on", "auto"))
    p.add_argument("--sort-active-tiles", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hardware-eval", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--hardware-eval-policy", type=str, default="fallback", choices=("off", "fallback", "require"))
    p.add_argument("--emit-final-T", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--emit-stop-count", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available.")
    device = torch.device("mps")
    means2d, conics, colors, opacities, depths = make_case(
        args.case,
        args.batch_size,
        args.gaussians,
        args.height,
        args.width,
        device,
        args.seed,
        args.trace_file,
    )
    actual_b, actual_g = means2d.shape[:2]
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
        max_pairs_per_launch=args.max_pairs_per_launch,
        stop_count_mode=args.stop_count_mode,
        stop_count_dense_threshold=args.dense_threshold,
        use_active_tiles=args.active_tiles,
        active_policy=args.active_policy,
        sort_active_tiles_by_count=args.sort_active_tiles,
        use_hardware_eval=args.hardware_eval,
        hardware_eval_policy=args.hardware_eval_policy,
        emit_final_T=args.emit_final_T,
        emit_stop_count=args.emit_stop_count,
    )
    hardware_status = probe_hardware_eval(cfg, compile_render_pipeline=bool(args.profile or args.hardware_eval))

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
        "gaussians": int(actual_g),
        "batch_size": int(actual_b),
        "batch_strategy": args.batch_strategy,
        "max_pairs_per_launch": int(args.max_pairs_per_launch),
        "stop_count_mode": args.stop_count_mode,
        "dense_threshold": args.dense_threshold,
        "active_tiles": bool(args.active_tiles),
        "active_policy": args.active_policy,
        "sort_active_tiles": bool(args.sort_active_tiles),
        "use_hardware_eval": bool(args.hardware_eval),
        "hardware_eval_policy": args.hardware_eval_policy,
        "emit_final_T": bool(args.emit_final_T),
        "emit_stop_count": bool(args.emit_stop_count),
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
        "hardware_eval_selection": hardware_status.selection,
        "hardware_eval_supported": bool(hardware_status.supported),
        "hardware_eval_reason": hardware_status.reason,
        "hardware_eval_missing_prerequisites": list(hardware_status.missing_prerequisites),
        "hardware_eval_unknown_prerequisites": list(hardware_status.unknown_prerequisites),
    }
    if profile_stats is not None:
        result.update({f"profile_{k}": v for k, v in profile_stats.items()})

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        kind = "forward_backward" if args.backward else "forward"
        print(
            f"case={args.case} B={actual_b} strat={args.batch_strategy} stop={args.stop_count_mode} {kind} "
            f"active={args.active_tiles} policy={args.active_policy} sort_active={args.sort_active_tiles} "
            f"hw_eval={args.hardware_eval} hw_policy={args.hardware_eval_policy} "
            f"mean_ms={result['mean_ms']:.3f} median_ms={result['median_ms']:.3f} "
            f"fwd_ms={result['forward_ms']:.3f} bwd_ms={result['backward_ms']:.3f} "
            f"tile={rt.tile_size} chunk={rt.chunk_size} cap={rt.fast_cap}"
        )
        if profile_stats is not None:
            print(json.dumps(profile_stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
