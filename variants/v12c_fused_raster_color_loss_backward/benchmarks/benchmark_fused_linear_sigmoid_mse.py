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

import torch_gsplat_bridge_v12c_fused_raster_color_loss_backward as v12c  # noqa: E402
from benchmarks.benchmark_mps import make_case  # noqa: E402


def _sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _clone_trainable(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    out = []
    for tensor in tensors:
        clone = tensor.detach().clone()
        if tensor.requires_grad:
            clone.requires_grad_(True)
        out.append(clone)
    return tuple(out)


def _make_inputs(args: argparse.Namespace) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, v12c.RasterConfig, bool]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the v12c fused Metal benchmark")
    device = torch.device("mps")
    means2d, conics, colors, opacities, depths = make_case(
        args.case,
        args.batch_size,
        args.gaussians,
        args.height,
        args.width,
        args.feature_dim,
        device,
        args.seed,
    )
    means2d.requires_grad_(True)
    conics.requires_grad_(True)
    colors.requires_grad_(True)
    opacities.requires_grad_(True)

    gen = torch.Generator(device="cpu").manual_seed(args.seed + 50000)
    target = torch.rand((args.batch_size, 3, args.height, args.width), generator=gen, dtype=torch.float32).to(device)
    background = (torch.rand((args.batch_size, 3, args.height, args.width), generator=gen, dtype=torch.float32) * 0.2).to(device)
    color_weight = (torch.randn((3, args.feature_dim), generator=gen, dtype=torch.float32) * 0.4).to(device).requires_grad_(True)
    color_bias = (torch.randn((3,), generator=gen, dtype=torch.float32) * 0.1).to(device).requires_grad_(True)

    rt = v12c.get_runtime_shader_config()
    cfg = v12c.RasterConfig(
        height=args.height,
        width=args.width,
        tile_size=rt.tile_size,
        max_fast_pairs=args.max_fast_pairs if args.max_fast_pairs > 0 else rt.fast_cap,
        enable_overflow_fallback=False,
        active_policy="off",
        use_active_tiles=False,
        stop_count_mode=args.stop_count_mode,
        stop_count_dense_threshold=args.dense_threshold,
    )
    return (means2d, conics, colors, opacities, depths), target, background, color_weight, color_bias, cfg, bool(args.freeze_colorizer)


def _unfused_step(inputs: tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, v12c.RasterConfig, bool]) -> tuple[torch.Tensor, ...]:
    raster_inputs, target, background, color_weight, color_bias, cfg, freeze_colorizer = inputs
    means2d, conics, colors, opacities, depths = _clone_trainable(raster_inputs)
    weight = color_weight.detach().clone().requires_grad_(not freeze_colorizer)
    bias = color_bias.detach().clone().requires_grad_(not freeze_colorizer)
    features_bhwf, alpha_bhw = v12c.rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, cfg)
    # Current trainer-style colorize boundary: NHWF raster output is converted
    # to NCHW before a 1x1 Conv2d colorizer.
    logits = F.conv2d(features_bhwf.permute(0, 3, 1, 2).contiguous(), weight[:, :, None, None], bias)
    rgb = torch.sigmoid(logits)
    pred = alpha_bhw.unsqueeze(1) * rgb + (1.0 - alpha_bhw.unsqueeze(1)) * background
    loss = (pred - target).square().mean()
    loss.backward()
    grad_weight = torch.zeros_like(weight) if weight.grad is None else weight.grad
    grad_bias = torch.zeros_like(bias) if bias.grad is None else bias.grad
    return means2d.grad, conics.grad, colors.grad, opacities.grad, grad_weight, grad_bias


def _fused_step(inputs: tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, v12c.RasterConfig, bool]) -> tuple[torch.Tensor, ...]:
    raster_inputs, target, background, color_weight, color_bias, cfg, freeze_colorizer = inputs
    means2d, conics, colors, opacities, depths = _clone_trainable(raster_inputs)
    weight = color_weight.detach().clone().requires_grad_(True)
    bias = color_bias.detach().clone().requires_grad_(True)
    fused = v12c.fused_linear_sigmoid_mse_backward(
        means2d,
        conics,
        colors,
        opacities,
        depths,
        target,
        weight,
        bias,
        cfg,
        background_rgb=background,
        compute_color_param_grads=not freeze_colorizer,
    )
    return (
        fused.grad_means2d,
        fused.grad_conics,
        fused.grad_colors,
        fused.grad_opacities,
        fused.grad_color_weight,
        fused.grad_color_bias,
    )


def _time_call(fn, inputs, *, iters: int, warmup: int) -> tuple[list[float], tuple[torch.Tensor, ...]]:
    last: tuple[torch.Tensor, ...] = ()
    for _ in range(warmup):
        last = fn(inputs)
    _sync()
    timings: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        last = fn(inputs)
        _sync()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings, last


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _parity(ref: tuple[torch.Tensor, ...], fused: tuple[torch.Tensor, ...]) -> dict[str, float]:
    names = ("means2d", "conics", "colors", "opacities", "color_weight", "color_bias")
    return {
        f"{name}_max_abs": float((a.detach() - b.detach()).abs().max().cpu())
        for name, a, b in zip(names, ref, fused, strict=True)
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark v12c fused raster+linear-sigmoid+MSE backward.")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--gaussians", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--feature-dim", type=int, default=32)
    p.add_argument("--case", type=str, default="medium_sigma_3_8")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--stop-count-mode", type=str, default="adaptive")
    p.add_argument("--dense-threshold", type=int, default=64)
    p.add_argument("--max-fast-pairs", type=int, default=-1)
    p.add_argument("--freeze-colorizer", action="store_true")
    p.add_argument("--json-output", type=Path)
    args = p.parse_args()

    inputs = _make_inputs(args)
    ref = _unfused_step(inputs)
    fused = _fused_step(inputs)
    _sync()
    parity = _parity(ref, fused)

    fused_timings, _ = _time_call(_fused_step, inputs, iters=args.iters, warmup=args.warmup)
    unfused_timings, _ = _time_call(_unfused_step, inputs, iters=args.iters, warmup=args.warmup)
    result = {
        "case": {
            "height": args.height,
            "width": args.width,
            "gaussians": args.gaussians,
            "batch_size": args.batch_size,
            "feature_dim": args.feature_dim,
            "case": args.case,
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "stop_count_mode": args.stop_count_mode,
            "dense_threshold": args.dense_threshold,
            "freeze_colorizer": bool(args.freeze_colorizer),
        },
        "parity": parity,
        "fused": _summarize(fused_timings),
        "unfused_trainer_style": _summarize(unfused_timings),
        "speedup_vs_unfused_median": statistics.median(unfused_timings) / statistics.median(fused_timings),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n")


if __name__ == "__main__":
    main()
