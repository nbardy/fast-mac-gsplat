from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.trainer_harness.projective_rational import (  # noqa: E402
    WorldTubeBatch,
    centered_frame_times,
    compile_projective_rational_tubes,
    fit_camera_path_polynomial,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    render_projective_rational_tubes_direct,
    render_projective_rational_tubes_tiled,
)


def _camera(frames: int, width: int, height: int, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(frames, 1, 1)
    k[:, 0, 0] = 0.9 * float(width)
    k[:, 1, 1] = 0.9 * float(width)
    k[:, 0, 2] = 0.5 * float(width)
    k[:, 1, 2] = 0.5 * float(height)
    zoom = 1.0 + 0.006 * times
    k[:, 0, 0] *= zoom
    k[:, 1, 1] *= zoom

    w2c = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(frames, 1, 1)
    w2c[:, 0, 3] = -0.010 * times
    w2c[:, 1, 3] = 0.004 * times
    w2c[:, 2, 3] = -0.040 * times
    return k, w2c


def _batch(tube_count: int, *, seed: int) -> WorldTubeBatch:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x0 = torch.empty((tube_count, 3), dtype=torch.float32)
    x0[:, 0] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-1.2, 1.2, generator=generator)
    x0[:, 1] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-0.75, 0.75, generator=generator)
    x0[:, 2] = torch.empty((tube_count,), dtype=torch.float32).uniform_(4.2, 6.2, generator=generator)
    velocity = torch.empty((tube_count, 3), dtype=torch.float32)
    velocity[:, 0] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-0.004, 0.004, generator=generator)
    velocity[:, 1] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-0.003, 0.003, generator=generator)
    velocity[:, 2] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-0.002, 0.002, generator=generator)
    return WorldTubeBatch(
        x0=x0,
        velocity=velocity,
        t0=torch.zeros((tube_count,), dtype=torch.float32),
        precision_xy=torch.empty((tube_count, 2), dtype=torch.float32).uniform_(36.0, 72.0, generator=generator),
        lambda_t=torch.empty((tube_count,), dtype=torch.float32).uniform_(0.04, 0.10, generator=generator),
        opacity=torch.empty((tube_count,), dtype=torch.float32).uniform_(0.35, 0.75, generator=generator),
        color=torch.empty((tube_count, 3), dtype=torch.float32).uniform_(0.05, 0.95, generator=generator),
    )


def _sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _timed(call: Callable[[], object], *, warmups: int, repeats: int) -> dict[str, float | int]:
    for _ in range(warmups):
        call()
        _sync()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        _sync()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "warmups": warmups,
        "repeats": repeats,
    }


def _case(
    *,
    tube_count: int,
    frames: int,
    width: int,
    height: int,
    tile_x: int,
    tile_y: int,
    tile_t: int,
    tile_capacity: int,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    times = centered_frame_times(frames)
    k_seq, w2c_seq = _camera(frames, width, height, times)
    camera_path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=2, frame_times=times)
    projected = compile_projective_rational_tubes(_batch(tube_count, seed=seed), camera_path)
    config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        tile_x=tile_x,
        tile_y=tile_y,
        tile_t=tile_t,
        tile_capacity=tile_capacity,
    )

    h_coeff = projected.h_coeff.to("mps")
    lambda_uv = projected.lambda_uv.to("mps")
    lambda_t = projected.lambda_t.to("mps")
    center_t = projected.center_t.to("mps")
    opacity = projected.opacity.to("mps")
    color = projected.color.to("mps")

    direct = render_projective_rational_tubes_direct(h_coeff, lambda_uv, lambda_t, center_t, opacity, color, config)
    tiled = render_projective_rational_tubes_tiled(
        h_coeff,
        lambda_uv,
        lambda_t,
        center_t,
        opacity,
        color,
        config,
        return_aux=True,
    )
    _sync()
    direct_cpu = direct.cpu()
    tiled_cpu = tiled.image.cpu()
    tile_counts = tiled.tile_counts.cpu()
    tile_overflow = tiled.tile_overflow.cpu()
    max_error = float((tiled_cpu - direct_cpu).abs().max().detach().cpu())

    direct_timing = _timed(
        lambda: render_projective_rational_tubes_direct(h_coeff, lambda_uv, lambda_t, center_t, opacity, color, config),
        warmups=warmups,
        repeats=repeats,
    )
    tiled_timing = _timed(
        lambda: render_projective_rational_tubes_tiled(
            h_coeff,
            lambda_uv,
            lambda_t,
            center_t,
            opacity,
            color,
            config,
            return_aux=True,
        ),
        warmups=warmups,
        repeats=repeats,
    )
    active_tile_count = int((tile_counts > 0).sum().item())
    total_tile_pairs = int(torch.clamp(tile_counts, max=config.tile_capacity).sum().item())
    return {
        "tube_count": tube_count,
        "frames": frames,
        "width": width,
        "height": height,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "tile_t": tile_t,
        "tile_capacity": tile_capacity,
        "camera_fit_error": camera_path.fit_error,
        "max_abs_error_vs_direct": max_error,
        "active_tile_count": active_tile_count,
        "total_tile_pairs": total_tile_pairs,
        "max_tile_count": int(tile_counts.max().item()),
        "overflow_tile_count": int((tile_overflow > 0).sum().item()),
        "direct": direct_timing,
        "tiled": tiled_timing,
        "tiled_to_direct_median_ratio": tiled_timing["median_ms"] / max(direct_timing["median_ms"], 1.0e-9),
        "pass": max_error <= 5.0e-5 and int((tile_overflow > 0).sum().item()) == 0 and active_tile_count > 0,
    }


def run_probe(
    *,
    tube_counts: list[int],
    frames: int,
    width: int,
    height: int,
    tile_x: int,
    tile_y: int,
    tile_t: int,
    tile_capacity: int,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the Metal PRT timing probe")
    rows = [
        _case(
            tube_count=tube_count,
            frames=frames,
            width=width,
            height=height,
            tile_x=tile_x,
            tile_y=tile_y,
            tile_t=tile_t,
            tile_capacity=tile_capacity,
            warmups=warmups,
            repeats=repeats,
            seed=seed + index,
        )
        for index, tube_count in enumerate(tube_counts)
    ]
    return {
        "name": "projective_rational_metal_forward_timing_probe",
        "note": "Metal timing is diagnostic. B2 does not by itself prove training speed.",
        "pass": all(bool(row["pass"]) for row in rows),
        "rows": rows,
    }


def _parse_int_list(value: str) -> list[int]:
    out = [int(item) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("values must be positive")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube-counts", type=_parse_int_list, default=[16, 64, 128])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--tile-x", type=int, default=8)
    parser.add_argument("--tile-y", type=int, default=8)
    parser.add_argument("--tile-t", type=int, default=2)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    summary = run_probe(
        tube_counts=args.tube_counts,
        frames=args.frames,
        width=args.width,
        height=args.height,
        tile_x=args.tile_x,
        tile_y=args.tile_y,
        tile_t=args.tile_t,
        tile_capacity=args.tile_capacity,
        warmups=args.warmups,
        repeats=args.repeats,
        seed=args.seed,
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
