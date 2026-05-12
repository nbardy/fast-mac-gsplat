from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.trainer_harness.projective_rational import (  # noqa: E402
    WorldTubeBatch,
    affine_taylor_center_residual,
    centered_frame_times,
    compile_projective_rational_tubes,
    direct_project_world_tubes,
    evaluate_projective_centers,
    fit_camera_path_polynomial,
    projection_matrices,
)


def _constant_k(frames: int, *, width: int = 128, height: int = 128, fx: float = 120.0) -> torch.Tensor:
    k = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(frames, 1, 1)
    k[:, 0, 0] = fx
    k[:, 1, 1] = fx
    k[:, 0, 2] = 0.5 * float(width)
    k[:, 1, 2] = 0.5 * float(height)
    return k


def _moving_k(times: torch.Tensor) -> torch.Tensor:
    frames = int(times.numel())
    k = _constant_k(frames)
    zoom = 1.0 + 0.008 * times
    k[:, 0, 0] *= zoom
    k[:, 1, 1] *= zoom
    return k


def _moving_w2c(times: torch.Tensor) -> torch.Tensor:
    frames = int(times.numel())
    w2c = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(frames, 1, 1)
    w2c[:, 0, 3] = -0.018 * times
    w2c[:, 1, 3] = 0.006 * times
    w2c[:, 2, 3] = -0.055 * times
    return w2c


def _random_batch(tube_count: int, *, seed: int) -> WorldTubeBatch:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x0 = torch.empty((tube_count, 3), dtype=torch.float32)
    x0[:, 0] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-1.4, 1.4, generator=generator)
    x0[:, 1] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-0.9, 0.9, generator=generator)
    x0[:, 2] = torch.empty((tube_count,), dtype=torch.float32).uniform_(3.8, 6.5, generator=generator)
    velocity = torch.empty((tube_count, 3), dtype=torch.float32)
    velocity[:, 0] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-0.010, 0.010, generator=generator)
    velocity[:, 1] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-0.008, 0.008, generator=generator)
    velocity[:, 2] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-0.003, 0.003, generator=generator)
    return WorldTubeBatch(
        x0=x0,
        velocity=velocity,
        t0=torch.zeros((tube_count,), dtype=torch.float32),
        precision_xy=torch.empty((tube_count, 2), dtype=torch.float32).uniform_(12.0, 28.0, generator=generator),
        lambda_t=torch.empty((tube_count,), dtype=torch.float32).uniform_(0.025, 0.075, generator=generator),
        opacity=torch.empty((tube_count,), dtype=torch.float32).uniform_(0.35, 0.85, generator=generator),
        color=torch.empty((tube_count, 3), dtype=torch.float32).uniform_(0.05, 0.95, generator=generator),
    )


def _per_frame_direct_loop(
    batch: WorldTubeBatch,
    k_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    p_seq = projection_matrices(k_seq, w2c_seq)
    centers = []
    depths = []
    ones = torch.ones((int(batch.x0.shape[0]), 1), dtype=torch.float32, device=batch.x0.device)
    for frame in range(int(times.numel())):
        tau = times[frame].to(batch.x0.device) - batch.t0
        points = batch.x0 + batch.velocity * tau.view(-1, 1)
        hom = torch.cat((points, ones), dim=-1)
        h = hom @ p_seq[frame].T
        z = h[:, 2].clamp_min(1.0e-6)
        centers.append(torch.stack((h[:, 0] / z, h[:, 1] / z), dim=-1))
        depths.append(z)
    return torch.stack(centers, dim=0), torch.stack(depths, dim=0)


def _affine_centers(direct_centers: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    frames = int(times.numel())
    mid = frames // 2
    left = max(0, mid - 1)
    right = min(frames - 1, mid + 1)
    slope = (direct_centers[right] - direct_centers[left]) / (times[right] - times[left]).clamp_min(1.0e-6)
    return direct_centers[mid].unsqueeze(0) + (times - times[mid]).view(-1, 1, 1) * slope.unsqueeze(0)


def _touch(value: object) -> float:
    if isinstance(value, tuple):
        return sum(_touch(item) for item in value)
    if torch.is_tensor(value):
        return float(value.reshape(-1)[:16].sum().detach().cpu())
    return 0.0


def _timed(call: Callable[[], object], *, warmups: int, repeats: int) -> dict[str, object]:
    for _ in range(warmups):
        _touch(call())
    elapsed_ms = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        value = call()
        checksum += _touch(value)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
        "repeats": repeats,
        "warmups": warmups,
        "checksum": checksum,
    }


def _case(
    *,
    frames: int,
    tube_count: int,
    repeats: int,
    warmups: int,
    seed: int,
) -> dict[str, object]:
    times = centered_frame_times(frames)
    batch = _random_batch(tube_count, seed=seed)
    k_seq = _moving_k(times)
    w2c_seq = _moving_w2c(times)

    direct_centers, _ = direct_project_world_tubes(batch, k_seq, w2c_seq, times)
    affine_centers = _affine_centers(direct_centers, times)
    camera_path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=2, frame_times=times)
    projected = compile_projective_rational_tubes(batch, camera_path)
    prt_centers, _ = evaluate_projective_centers(projected, times)

    affine_error = float((affine_centers - direct_centers).abs().max().detach().cpu())
    affine_curvature_error = affine_taylor_center_residual(direct_centers, times)
    prt_error = float((prt_centers - direct_centers).abs().max().detach().cpu())
    loop_centers, _ = _per_frame_direct_loop(batch, k_seq, w2c_seq, times)
    loop_error = float((loop_centers - direct_centers).abs().max().detach().cpu())

    static_k = _constant_k(frames)
    static_w2c = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(frames, 1, 1)
    static_mid = frames // 2

    timing = {
        "static_one_frame_project": _timed(
            lambda: direct_project_world_tubes(
                batch,
                static_k[static_mid : static_mid + 1],
                static_w2c[static_mid : static_mid + 1],
                times[static_mid : static_mid + 1],
            ),
            warmups=warmups,
            repeats=repeats,
        ),
        "dynamic_first_order": _timed(
            lambda: _affine_centers(direct_centers, times),
            warmups=warmups,
            repeats=repeats,
        ),
        "projective_rational_compile_eval": _timed(
            lambda: evaluate_projective_centers(
                compile_projective_rational_tubes(
                    batch,
                    fit_camera_path_polynomial(k_seq, w2c_seq, degree=2, frame_times=times),
                ),
                times,
            ),
            warmups=warmups,
            repeats=repeats,
        ),
        "projective_rational_eval_only": _timed(
            lambda: evaluate_projective_centers(projected, times),
            warmups=warmups,
            repeats=repeats,
        ),
        "per_frame_direct_loop": _timed(
            lambda: _per_frame_direct_loop(batch, k_seq, w2c_seq, times),
            warmups=warmups,
            repeats=repeats,
        ),
        "per_frame_direct_vectorized": _timed(
            lambda: direct_project_world_tubes(batch, k_seq, w2c_seq, times),
            warmups=warmups,
            repeats=repeats,
        ),
    }

    prt_loop_ratio = (
        timing["projective_rational_compile_eval"]["median_ms"] / timing["per_frame_direct_loop"]["median_ms"]
        if timing["per_frame_direct_loop"]["median_ms"] > 0.0
        else float("inf")
    )
    eval_loop_ratio = (
        timing["projective_rational_eval_only"]["median_ms"] / timing["per_frame_direct_loop"]["median_ms"]
        if timing["per_frame_direct_loop"]["median_ms"] > 0.0
        else float("inf")
    )

    return {
        "frames": frames,
        "tube_count": tube_count,
        "camera_degree": 2,
        "camera_fit_error": camera_path.fit_error,
        "dynamic_first_order": {
            "center_max_abs_px": affine_error,
            "curvature_residual_px": affine_curvature_error,
        },
        "projective_rational": {
            "center_max_abs_px": prt_error,
            "compile_eval_to_loop_median_ratio": prt_loop_ratio,
            "eval_only_to_loop_median_ratio": eval_loop_ratio,
        },
        "per_frame_direct_loop": {
            "center_max_abs_px": loop_error,
        },
        "timing": timing,
        "pass": (
            prt_error <= 2.5e-4
            and loop_error <= 1.0e-4
            and affine_error > 0.25
            and camera_path.fit_error <= 1.0e-5
        ),
    }


def run_probe(
    *,
    frame_counts: list[int],
    tube_count: int,
    repeats: int,
    warmups: int,
    seed: int,
) -> dict[str, object]:
    rows = [
        _case(frames=frames, tube_count=tube_count, repeats=repeats, warmups=warmups, seed=seed + index)
        for index, frames in enumerate(frame_counts)
    ]
    return {
        "name": "projective_rational_projection_scaling_probe",
        "note": "CPU projection-only probe. Timing is diagnostic and is not a Metal rasterizer speed claim.",
        "frame_counts": frame_counts,
        "tube_count": tube_count,
        "pass": all(bool(row["pass"]) for row in rows),
        "rows": rows,
    }


def _parse_frame_counts(value: str) -> list[int]:
    counts = [int(item) for item in value.split(",") if item.strip()]
    if not counts:
        raise argparse.ArgumentTypeError("must provide at least one frame count")
    if any(count <= 2 for count in counts):
        raise argparse.ArgumentTypeError("frame counts must be greater than 2")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=_parse_frame_counts, default=[8, 16, 32])
    parser.add_argument("--tube-count", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if args.tube_count <= 0:
        raise ValueError("tube count must be positive")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")

    summary = run_probe(
        frame_counts=args.frames,
        tube_count=args.tube_count,
        repeats=args.repeats,
        warmups=args.warmups,
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
