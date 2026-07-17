from __future__ import annotations

import argparse
import json

import torch

try:
    from .curve_tube import (
        CurveTubeRenderConfig,
        centered_frame_times,
        compile_curve_tubes_from_samples,
        compile_projective_rational_tubes,
        dense_render_compiled_curve_tubes,
        dense_render_projective_rational_tubes,
        eval_curve_centers,
        eval_projective_centers,
        psnr,
        sample_world_tube_projection,
    )
except ImportError:  # pragma: no cover
    from curve_tube import (
        CurveTubeRenderConfig,
        centered_frame_times,
        compile_curve_tubes_from_samples,
        compile_projective_rational_tubes,
        dense_render_compiled_curve_tubes,
        dense_render_projective_rational_tubes,
        eval_curve_centers,
        eval_projective_centers,
        psnr,
        sample_world_tube_projection,
    )


def _scene_tensors(device: torch.device) -> tuple[torch.Tensor, ...]:
    x0 = torch.tensor([[-0.12, -0.08, 2.0], [0.18, 0.12, 2.4]], dtype=torch.float32, device=device)
    velocity = torch.tensor([[0.025, 0.015, 0.02], [-0.015, 0.012, -0.01]], dtype=torch.float32, device=device)
    t0 = torch.zeros((2,), dtype=torch.float32, device=device)
    precision_xy = torch.tensor([[80.0, 90.0], [70.0, 85.0]], dtype=torch.float32, device=device)
    lambda_t = torch.full((2,), 0.25, dtype=torch.float32, device=device)
    opacity = torch.full((2,), 0.65, dtype=torch.float32, device=device)
    color = torch.tensor([[0.8, 0.25, 0.1], [0.1, 0.5, 0.9]], dtype=torch.float32, device=device)
    return x0, velocity, t0, precision_xy, lambda_t, opacity, color


def _base_k(frames: int, times: torch.Tensor, *, focal_scale: float) -> torch.Tensor:
    K_seq = torch.eye(3, dtype=torch.float32, device=times.device).repeat(frames, 1, 1)
    K_seq[:, 0, 0] = 32.0
    K_seq[:, 1, 1] = 32.0
    K_seq[:, 0, 2] = 16.0
    K_seq[:, 1, 2] = 16.0
    K_seq[:, 0, 0] *= 1.0 + float(focal_scale) * times
    K_seq[:, 1, 1] *= 1.0 + float(focal_scale) * times
    return K_seq


def _static_w2c(frames: int, device: torch.device) -> torch.Tensor:
    return torch.eye(4, dtype=torch.float32, device=device).repeat(frames, 1, 1)


def _moving_w2c(times: torch.Tensor) -> torch.Tensor:
    w2c_seq = torch.eye(4, dtype=torch.float32, device=times.device).repeat(int(times.numel()), 1, 1)
    w2c_seq[:, 0, 3] = 0.02 * times
    w2c_seq[:, 2, 3] = 0.01 * times
    return w2c_seq


def _compile_prt(
    x0: torch.Tensor,
    velocity: torch.Tensor,
    t0: torch.Tensor,
    precision_xy: torch.Tensor,
    lambda_t: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    K_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
    times: torch.Tensor,
    *,
    degree: int,
):
    return compile_projective_rational_tubes(
        x0=x0,
        velocity=velocity,
        t0=t0,
        precision_xy=precision_xy,
        lambda_t=lambda_t,
        opacity=opacity,
        color=color,
        K_seq=K_seq,
        w2c_seq=w2c_seq,
        times=times,
        camera_degree=degree,
    )


def run_smoke(device: torch.device) -> dict[str, float | bool | list[int] | str]:
    frames = 5
    times = centered_frame_times(frames, device=device)
    x0, velocity, t0, precision_xy, lambda_t, opacity, color = _scene_tensors(device)
    config = CurveTubeRenderConfig(height=32, width=32, frames=frames, alpha_threshold=0.0)

    static_K = _base_k(frames, times, focal_scale=0.0)
    static_w2c = _static_w2c(frames, device)
    static_prt = _compile_prt(
        x0,
        velocity,
        t0,
        precision_xy,
        lambda_t,
        opacity,
        color,
        static_K,
        static_w2c,
        times,
        degree=0,
    )
    static_centers, static_depths = sample_world_tube_projection(x0, velocity, t0, static_K, static_w2c, times)
    static_prt_centers = eval_projective_centers(static_prt, times)
    static_center_error = float((static_prt_centers - static_centers).abs().amax().detach().cpu())

    static_curve = compile_curve_tubes_from_samples(
        center_samples=static_centers,
        depth_samples=static_depths,
        times=times,
        lambda_uv=static_prt.lambda_uv,
        lambda_t=lambda_t,
        center_t=t0,
        opacity=opacity,
        color=color,
        degree=frames - 1,
    )
    static_curve_centers = eval_curve_centers(static_curve, times)
    static_curve_error = float((static_curve_centers - static_centers).abs().amax().detach().cpu())
    static_image_prt = dense_render_projective_rational_tubes(static_prt, config)
    static_image_curve = dense_render_compiled_curve_tubes(static_curve, config)
    static_render_error = float((static_image_prt - static_image_curve).abs().amax().detach().cpu())

    moving_K = _base_k(frames, times, focal_scale=0.03)
    moving_w2c = _moving_w2c(times)
    moving_prt = _compile_prt(
        x0,
        velocity,
        t0,
        precision_xy,
        lambda_t,
        opacity,
        color,
        moving_K,
        moving_w2c,
        times,
        degree=2,
    )
    moving_centers, moving_depths = sample_world_tube_projection(x0, velocity, t0, moving_K, moving_w2c, times)
    moving_prt_centers = eval_projective_centers(moving_prt, times)
    moving_center_error = float((moving_prt_centers - moving_centers).abs().amax().detach().cpu())

    moving_curve = compile_curve_tubes_from_samples(
        center_samples=moving_centers,
        depth_samples=moving_depths,
        times=times,
        lambda_uv=moving_prt.lambda_uv,
        lambda_t=lambda_t,
        center_t=t0,
        opacity=opacity,
        color=color,
        degree=frames - 1,
    )
    moving_curve_centers = eval_curve_centers(moving_curve, times)
    moving_curve_error = float((moving_curve_centers - moving_centers).abs().amax().detach().cpu())
    image_prt = dense_render_projective_rational_tubes(moving_prt, config)
    image_curve = dense_render_compiled_curve_tubes(moving_curve, config)

    pass_thresholds = {
        "static_center_residual_px": 2.0e-5,
        "static_curve_residual_px": 8.0e-5,
        "static_render_max_abs_error": 8.0e-5,
        "moving_center_residual_px": 8.0e-5,
        "moving_curve_residual_px": 1.0e-4,
    }
    report = {
        "device": str(device),
        "frames": frames,
        "tube_count": int(x0.shape[0]),
        "static_center_residual_px": static_center_error,
        "static_curve_residual_px": static_curve_error,
        "static_render_max_abs_error": static_render_error,
        "moving_center_residual_px": moving_center_error,
        "moving_projected_probe_error_px": float(moving_prt.projected_probe_error_px.detach().cpu()),
        "moving_curve_residual_px": moving_curve_error,
        "moving_curve_fit_error": float(moving_curve.fit_error_px.detach().cpu()),
        "prt_curve_psnr": psnr(image_prt, image_curve),
        "image_shape": list(image_prt.shape),
    }
    report["pass"] = all(float(report[key]) <= threshold for key, threshold in pass_thresholds.items())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    report = run_smoke(device)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not bool(report["pass"]):
        raise AssertionError("STAR-PRT curve tube smoke failed")


if __name__ == "__main__":
    main()
