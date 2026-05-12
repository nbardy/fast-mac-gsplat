from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.trainer_harness.projective_rational import (  # noqa: E402
    WorldTubeBatch,
    affine_taylor_center_residual,
    centered_frame_times,
    compile_projective_rational_tubes,
    curvature_selective_mask,
    dense_render_projective_rational_tubes,
    direct_project_world_tubes,
    fit_camera_path_polynomial,
)


def _constant_k(frames: int, *, fx: float = 90.0, fy: float = 90.0, cx: float = 32.0, cy: float = 32.0) -> torch.Tensor:
    k = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(frames, 1, 1)
    k[:, 0, 0] = float(fx)
    k[:, 1, 1] = float(fy)
    k[:, 0, 2] = float(cx)
    k[:, 1, 2] = float(cy)
    return k


def _zoom_k(frames: int, times: torch.Tensor) -> torch.Tensor:
    k = _constant_k(frames)
    zoom = 1.0 + 0.025 * times
    k[:, 0, 0] *= zoom
    k[:, 1, 1] *= zoom
    return k


def _w2c_translation_z(times: torch.Tensor, *, slope: float) -> torch.Tensor:
    frames = int(times.numel())
    w2c = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(frames, 1, 1)
    w2c[:, 2, 3] = -float(slope) * times
    return w2c


def _w2c_cut(times: torch.Tensor) -> torch.Tensor:
    frames = int(times.numel())
    w2c = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(frames, 1, 1)
    w2c[:, 0, 3] = torch.where(times < 0.0, torch.full_like(times, -0.6), torch.full_like(times, 0.6))
    return w2c


def _batch() -> WorldTubeBatch:
    return WorldTubeBatch(
        x0=torch.tensor([[0.8, -0.15, 4.0], [-0.55, 0.25, 5.2]], dtype=torch.float32),
        velocity=torch.tensor([[0.03, 0.015, 0.0], [0.00, -0.02, 0.015]], dtype=torch.float32),
        t0=torch.tensor([0.0, 0.0], dtype=torch.float32),
        precision_xy=torch.tensor([[18.0, 18.0], [14.0, 16.0]], dtype=torch.float32),
        lambda_t=torch.tensor([0.06, 0.04], dtype=torch.float32),
        opacity=torch.tensor([0.75, 0.68], dtype=torch.float32),
        color=torch.tensor([[0.95, 0.25, 0.15], [0.20, 0.50, 0.95]], dtype=torch.float32),
    )


def _projective_residual(batch: WorldTubeBatch, k_seq: torch.Tensor, w2c_seq: torch.Tensor, times: torch.Tensor, degree: int) -> dict[str, float]:
    camera_path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=degree, frame_times=times)
    projected = compile_projective_rational_tubes(batch, camera_path)
    prt_centers, min_hz = projected_centers(projected, times)
    direct_centers, direct_z = direct_project_world_tubes(batch, k_seq, w2c_seq, times)
    return {
        "center_max_abs_px": float((prt_centers - direct_centers).abs().max().detach().cpu()),
        "camera_fit_error": camera_path.fit_error,
        "min_hz": float(min_hz.min().detach().cpu()),
        "min_direct_z": float(direct_z.min().detach().cpu()),
    }


def projected_centers(projected, times):
    from research_project.trainer_harness.projective_rational import evaluate_projective_centers

    return evaluate_projective_centers(projected, times)


def run_audit() -> dict[str, object]:
    torch.manual_seed(0)
    frames = 16
    times = centered_frame_times(frames)
    batch = _batch()

    static_k = _constant_k(frames)
    static_w2c = _w2c_translation_z(times, slope=0.0)
    static = _projective_residual(batch, static_k, static_w2c, times, degree=0)

    moving_k = _constant_k(frames)
    moving_w2c = _w2c_translation_z(times, slope=0.11)
    moving = _projective_residual(batch, moving_k, moving_w2c, times, degree=1)
    moving_direct, _ = direct_project_world_tubes(batch, moving_k, moving_w2c, times)
    affine_error = affine_taylor_center_residual(moving_direct, times)

    zoom_k = _zoom_k(frames, times)
    zoom_w2c = _w2c_translation_z(times, slope=0.04)
    zoom = _projective_residual(batch, zoom_k, zoom_w2c, times, degree=2)

    cut_path = fit_camera_path_polynomial(_constant_k(frames), _w2c_cut(times), degree=2, frame_times=times)

    projected = compile_projective_rational_tubes(batch, fit_camera_path_polynomial(moving_k, moving_w2c, degree=1, frame_times=times))
    image = dense_render_projective_rational_tubes(projected, height=32, width=64, frame_times=times[:4])

    mixed_batch = WorldTubeBatch(
        x0=torch.tensor([[0.0, 0.0, 5.0], [1.2, 0.0, 3.2]], dtype=torch.float32),
        velocity=torch.zeros((2, 3), dtype=torch.float32),
        t0=torch.zeros((2,), dtype=torch.float32),
        precision_xy=torch.full((2, 2), 16.0, dtype=torch.float32),
        lambda_t=torch.full((2,), 0.05, dtype=torch.float32),
        opacity=torch.full((2,), 0.7, dtype=torch.float32),
        color=torch.tensor([[0.8, 0.8, 0.8], [0.9, 0.2, 0.1]], dtype=torch.float32),
    )
    mixed_direct, _ = direct_project_world_tubes(mixed_batch, moving_k, moving_w2c, times)
    prt_mask = curvature_selective_mask(mixed_direct, times, threshold_px=0.25)

    summary = {
        "static_affine_parity": {
            **static,
            "pass": static["center_max_abs_px"] <= 1.0e-4 and static["camera_fit_error"] <= 1.0e-6,
        },
        "moving_camera_projective": {
            **moving,
            "affine_center_max_abs_px": affine_error,
            "pass": moving["center_max_abs_px"] <= 1.0e-4 and affine_error > 0.25,
        },
        "zoom_projective": {
            **zoom,
            "pass": zoom["center_max_abs_px"] <= 2.5e-4 and zoom["camera_fit_error"] <= 1.0e-5,
        },
        "camera_cut_detection": {
            "camera_fit_error": cut_path.fit_error,
            "pass": cut_path.fit_error > 0.05,
        },
        "dense_render_smoke": {
            "shape": list(image.shape),
            "finite": bool(torch.isfinite(image).all().item()),
            "max": float(image.max().detach().cpu()),
            "pass": list(image.shape) == [4, 32, 64, 3] and bool(torch.isfinite(image).all().item()),
        },
        "curvature_selective": {
            "threshold_px": 0.25,
            "prt_mask": [bool(v) for v in prt_mask.detach().cpu().tolist()],
            "pass": [bool(v) for v in prt_mask.detach().cpu().tolist()] == [False, True],
        },
    }
    summary["pass"] = all(bool(value.get("pass")) for key, value in summary.items() if isinstance(value, dict))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    summary = run_audit()
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
