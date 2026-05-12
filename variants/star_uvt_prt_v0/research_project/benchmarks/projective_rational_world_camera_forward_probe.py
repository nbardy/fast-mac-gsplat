from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_metal_forward_timing_probe import (  # noqa: E402
    _batch,
    _camera,
)
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    WorldTubeBatch,
    centered_frame_times,
    compile_projective_rational_tubes,
    direct_project_world_tubes,
    fit_camera_path_polynomial,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    apply_projective_rational_tile_env,
    recommend_projective_rational_tile_config,
    render_projective_rational_tubes_tiled,
)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(call: Callable[[], Tensor], *, device: torch.device, warmups: int, repeats: int) -> tuple[Tensor, dict[str, Any]]:
    image = None
    with torch.no_grad():
        for _ in range(warmups):
            image = call()
            _sync(device)
        samples = []
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            image = call()
            _sync(device)
            samples.append((time.perf_counter() - started) * 1000.0)
    if image is None:
        raise AssertionError("timed call did not run")
    return image, {
        "samples": samples,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "max_ms": max(samples),
        "warmups": warmups,
        "repeats": repeats,
    }


def _psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1.0e-12))


def _image_metrics(image: Tensor, target: Tensor) -> dict[str, float]:
    mse = float((image - target).square().mean().detach().cpu())
    return {
        "mse": mse,
        "psnr": _psnr(mse),
        "l1": float((image - target).abs().mean().detach().cpu()),
        "max_abs": float((image - target).abs().max().detach().cpu()),
    }


def dense_render_direct_world_tubes(
    batch: WorldTubeBatch,
    k_seq: Tensor,
    w2c_seq: Tensor,
    frame_times: Tensor,
    lambda_uv: Tensor,
    *,
    height: int,
    width: int,
    alpha_threshold: float = 1.0 / 255.0,
    max_alpha: float = 0.99,
) -> Tensor:
    centers, depth = direct_project_world_tubes(batch, k_seq, w2c_seq, frame_times)
    y = torch.arange(height, dtype=torch.float32, device=batch.x0.device) + 0.5
    x = torch.arange(width, dtype=torch.float32, device=batch.x0.device) + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rows = []
    for frame in range(int(frame_times.numel())):
        order = torch.argsort(depth[frame].detach(), stable=True)
        accum = torch.zeros((height, width, 3), dtype=torch.float32, device=batch.x0.device)
        trans = torch.ones((height, width, 1), dtype=torch.float32, device=batch.x0.device)
        dt = frame_times[frame].to(batch.x0.device) - batch.t0
        for tube in order.tolist():
            lambda_uu, lambda_uv_cross, lambda_vv = lambda_uv[tube]
            du = xx - centers[frame, tube, 0]
            dv = yy - centers[frame, tube, 1]
            spatial = lambda_uu * du.square() + 2.0 * lambda_uv_cross * du * dv + lambda_vv * dv.square()
            temporal = batch.lambda_t[tube] * dt[tube].square()
            alpha = (batch.opacity[tube] * torch.exp(-0.5 * (spatial + temporal))).clamp(max=max_alpha)
            alpha = torch.where(alpha >= alpha_threshold, alpha, torch.zeros_like(alpha))
            alpha3 = alpha.unsqueeze(-1)
            accum = accum + trans * alpha3 * batch.color[tube].view(1, 1, 3)
            trans = trans * (1.0 - alpha3)
        rows.append(accum)
    return torch.stack(rows, dim=0).contiguous()


def _render_prt(projected, config: UVTRenderConfig):
    return render_projective_rational_tubes_tiled(
        projected.h_coeff,
        projected.lambda_uv,
        projected.lambda_t,
        projected.center_t,
        projected.opacity,
        projected.color,
        config,
        return_aux=True,
    )


def _static_camera_sequence(k_seq: Tensor, w2c_seq: Tensor) -> tuple[Tensor, Tensor]:
    frames = int(k_seq.shape[0])
    mid = frames // 2
    return k_seq[mid : mid + 1].repeat(frames, 1, 1), w2c_seq[mid : mid + 1].repeat(frames, 1, 1)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "mps":
        raise ValueError("world-camera PRT probe currently requires --device=mps")
    frame_times = centered_frame_times(args.frames, device="cpu")
    k_seq, w2c_seq = _camera(args.frames, args.width, args.height, frame_times, motion_scale=args.camera_motion_scale)
    world_batch_cpu = _batch(args.tube_count, seed=args.seed)

    prt_camera_path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=args.camera_poly_degree, frame_times=frame_times)
    prt_projected_cpu = compile_projective_rational_tubes(world_batch_cpu, prt_camera_path)
    static_k, static_w2c = _static_camera_sequence(k_seq, w2c_seq)
    static_camera_path = fit_camera_path_polynomial(static_k, static_w2c, degree=0, frame_times=frame_times)
    static_projected_cpu = compile_projective_rational_tubes(world_batch_cpu, static_camera_path)

    tile_config = recommend_projective_rational_tile_config(tube_count=args.tube_count)
    apply_projective_rational_tile_env(tile_config)
    config = UVTRenderConfig(height=args.height, width=args.width, frames=args.frames, **tile_config.as_render_kwargs())

    world_batch = WorldTubeBatch(
        x0=world_batch_cpu.x0.to(device),
        velocity=world_batch_cpu.velocity.to(device),
        t0=world_batch_cpu.t0.to(device),
        precision_xy=world_batch_cpu.precision_xy.to(device),
        lambda_t=world_batch_cpu.lambda_t.to(device),
        opacity=world_batch_cpu.opacity.to(device),
        color=world_batch_cpu.color.to(device),
    )
    k_dev = k_seq.to(device)
    w2c_dev = w2c_seq.to(device)
    times_dev = frame_times.to(device)
    prt_projected = type(prt_projected_cpu)(
        h_coeff=prt_projected_cpu.h_coeff.to(device),
        lambda_uv=prt_projected_cpu.lambda_uv.to(device),
        lambda_t=prt_projected_cpu.lambda_t.to(device),
        center_t=prt_projected_cpu.center_t.to(device),
        depth_coeff=prt_projected_cpu.depth_coeff.to(device),
        opacity=prt_projected_cpu.opacity.to(device),
        color=prt_projected_cpu.color.to(device),
        camera_fit_error=prt_projected_cpu.camera_fit_error,
    )
    static_projected = type(static_projected_cpu)(
        h_coeff=static_projected_cpu.h_coeff.to(device),
        lambda_uv=static_projected_cpu.lambda_uv.to(device),
        lambda_t=static_projected_cpu.lambda_t.to(device),
        center_t=static_projected_cpu.center_t.to(device),
        depth_coeff=static_projected_cpu.depth_coeff.to(device),
        opacity=static_projected_cpu.opacity.to(device),
        color=static_projected_cpu.color.to(device),
        camera_fit_error=static_projected_cpu.camera_fit_error,
    )

    reference_image, reference_timing = _timed(
        lambda: dense_render_direct_world_tubes(
            world_batch,
            k_dev,
            w2c_dev,
            times_dev,
            prt_projected.lambda_uv,
            height=args.height,
            width=args.width,
        ),
        device=device,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    prt_aux, prt_timing = _timed(
        lambda: _render_prt(prt_projected, config),
        device=device,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    static_aux, static_timing = _timed(
        lambda: _render_prt(static_projected, config),
        device=device,
        warmups=args.warmups,
        repeats=args.repeats,
    )

    return {
        "name": "projective_rational_world_camera_forward_probe",
        "note": (
            "Synthetic world-camera forward probe. The reference is exact per-frame projection with dense "
            "screen compositing; PRT and static-camera paths use the tiled Metal projective-rational renderer."
        ),
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "tube_count": args.tube_count,
        "seed": args.seed,
        "device": str(device),
        "camera_motion_scale": args.camera_motion_scale,
        "camera_poly_degree": args.camera_poly_degree,
        "tile_config": tile_config.as_dict(),
        "direct_per_frame_reference": {
            "render_benchmark_ms": reference_timing,
        },
        "projective_rational": {
            "camera_fit_error": prt_camera_path.fit_error,
            "render_benchmark_ms": prt_timing,
            "metrics_vs_direct_reference": _image_metrics(prt_aux.image, reference_image),
            "max_tile_count": int(prt_aux.tile_counts.max().detach().cpu()),
            "overflow_tile_count": int((prt_aux.tile_overflow > 0).sum().detach().cpu()),
            "active_tile_count": int((prt_aux.tile_counts > 0).sum().detach().cpu()),
        },
        "static_camera": {
            "camera_fit_error": static_camera_path.fit_error,
            "render_benchmark_ms": static_timing,
            "metrics_vs_direct_reference": _image_metrics(static_aux.image, reference_image),
            "max_tile_count": int(static_aux.tile_counts.max().detach().cpu()),
            "overflow_tile_count": int((static_aux.tile_overflow > 0).sum().detach().cpu()),
            "active_tile_count": int((static_aux.tile_counts > 0).sum().detach().cpu()),
        },
        "pass": int((prt_aux.tile_overflow > 0).sum().detach().cpu()) == 0
        and _image_metrics(prt_aux.image, reference_image)["psnr"]
        > _image_metrics(static_aux.image, reference_image)["psnr"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--tube-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--camera-motion-scale", type=float, default=3.0)
    parser.add_argument("--camera-poly-degree", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = run_probe(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
