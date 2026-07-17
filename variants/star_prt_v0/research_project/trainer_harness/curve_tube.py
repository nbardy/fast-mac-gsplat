from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class WorldTubeBatch:
    x0: Tensor
    velocity: Tensor
    t0: Tensor
    precision_xy: Tensor
    lambda_t: Tensor
    opacity: Tensor
    color: Tensor


@dataclass(frozen=True)
class CurveTubeRenderConfig:
    height: int
    width: int
    frames: int
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1.0e-4
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_alpha: float = 0.99
    min_hz: float = 1.0e-6
    min_depth: float = 1.0e-4


@dataclass(frozen=True)
class CameraPathPolynomial:
    p_coeff: Tensor
    w_coeff: Tensor
    t_center: float
    t_scale: float
    degree: int
    fit_error_fro: Tensor


@dataclass(frozen=True)
class ProjectiveRationalTubeSequence:
    h_coeff: Tensor
    lambda_uv: Tensor
    lambda_t: Tensor
    center_t: Tensor
    depth_coeff: Tensor
    opacity: Tensor
    color: Tensor
    t_center: float
    t_scale: float
    camera_fit_error: Tensor | None = None
    projected_probe_error_px: Tensor | None = None


@dataclass(frozen=True)
class CompiledCurveTubeSequence:
    center_coeff: Tensor
    lambda_uv: Tensor
    lambda_t: Tensor
    center_t: Tensor
    depth_coeff: Tensor
    opacity: Tensor
    color: Tensor
    t_center: float
    t_scale: float
    fit_error_px: Tensor | None = None


def centered_frame_times(
    frames: int,
    *,
    full_frames: int | None = None,
    frame_start: int = 0,
    device: torch.device | str | None = None,
) -> Tensor:
    dev = None if device is None else torch.device(device)
    local = torch.arange(int(frames), dtype=torch.float32, device=dev)
    if full_frames is None:
        return local - 0.5 * float(frames - 1)
    return local + float(frame_start) - 0.5 * float(full_frames - 1)


def _time_center_scale(times: Tensor) -> tuple[float, float]:
    t_min = float(times.min().detach().cpu())
    t_max = float(times.max().detach().cpu())
    center = 0.5 * (t_min + t_max)
    scale = max(1.0, 0.5 * (t_max - t_min))
    return center, scale


def vandermonde_tau(times: Tensor, degree: int, *, t_center: float, t_scale: float) -> Tensor:
    tau = (times - float(t_center)) / float(t_scale)
    cols = [torch.ones_like(tau)]
    for _ in range(int(degree)):
        cols.append(cols[-1] * tau)
    return torch.stack(cols, dim=-1)


def fit_polynomial_values(
    values: Tensor,
    times: Tensor,
    degree: int,
    *,
    t_center: float | None = None,
    t_scale: float | None = None,
) -> tuple[Tensor, Tensor, float, float]:
    if values.shape[0] != times.shape[0]:
        raise ValueError("values and times must agree on the first dimension")
    if t_center is None or t_scale is None:
        t_center, t_scale = _time_center_scale(times)
    flat = values.reshape(values.shape[0], -1)
    vander = vandermonde_tau(times, int(degree), t_center=t_center, t_scale=t_scale)
    coeff_flat = torch.linalg.lstsq(
        vander.detach().cpu().to(torch.float64),
        flat.detach().cpu().to(torch.float64),
    ).solution.to(device=values.device, dtype=values.dtype)
    recon = vander @ coeff_flat
    denom = torch.linalg.norm(flat).clamp_min(1.0e-8)
    rel_error = torch.linalg.norm(recon - flat) / denom
    coeff = coeff_flat.reshape((int(degree) + 1, *values.shape[1:])).contiguous()
    return coeff, rel_error, float(t_center), float(t_scale)


def fit_camera_path_polynomial(K_seq: Tensor, w2c_seq: Tensor, times: Tensor, degree: int = 2) -> CameraPathPolynomial:
    if K_seq.shape[:1] != w2c_seq.shape[:1] or K_seq.shape[0] != times.shape[0]:
        raise ValueError("K_seq, w2c_seq, and times must share frame count")
    if K_seq.shape[-2:] != (3, 3):
        raise ValueError("K_seq must have shape [F,3,3]")
    if w2c_seq.shape[-2:] != (4, 4):
        raise ValueError("w2c_seq must have shape [F,4,4]")
    p_seq = torch.matmul(K_seq, w2c_seq[:, :3, :])
    w_seq = w2c_seq[:, :3, :]
    t_center, t_scale = _time_center_scale(times)
    p_coeff, p_err, _, _ = fit_polynomial_values(
        p_seq,
        times,
        degree,
        t_center=t_center,
        t_scale=t_scale,
    )
    w_coeff, w_err, _, _ = fit_polynomial_values(
        w_seq,
        times,
        degree,
        t_center=t_center,
        t_scale=t_scale,
    )
    return CameraPathPolynomial(
        p_coeff=p_coeff,
        w_coeff=w_coeff,
        t_center=t_center,
        t_scale=t_scale,
        degree=int(degree),
        fit_error_fro=torch.maximum(p_err, w_err),
    )


def projection_matrices(K_seq: Tensor, w2c_seq: Tensor) -> Tensor:
    if K_seq.shape[:1] != w2c_seq.shape[:1]:
        raise ValueError("K_seq and w2c_seq must share frame count")
    if K_seq.shape[-2:] != (3, 3):
        raise ValueError("K_seq must have shape [F,3,3]")
    if w2c_seq.shape[-2:] != (4, 4):
        raise ValueError("w2c_seq must have shape [F,4,4]")
    return torch.matmul(K_seq, w2c_seq[:, :3, :])


def _homogeneous_center_terms(x0: Tensor, velocity: Tensor, t0: Tensor, t_center: float, t_scale: float) -> tuple[Tensor, Tensor]:
    x_center = x0 + velocity * (float(t_center) - t0).unsqueeze(-1)
    x_slope = velocity * float(t_scale)
    ones = torch.ones((x0.shape[0], 1), dtype=x0.dtype, device=x0.device)
    zeros = torch.zeros_like(ones)
    return torch.cat((x_center, ones), dim=-1), torch.cat((x_slope, zeros), dim=-1)


def compile_h_coeff(x0: Tensor, velocity: Tensor, t0: Tensor, camera_poly: CameraPathPolynomial) -> Tensor:
    x_center_h, x_slope_h = _homogeneous_center_terms(
        x0,
        velocity,
        t0,
        camera_poly.t_center,
        camera_poly.t_scale,
    )
    d_p = int(camera_poly.p_coeff.shape[0] - 1)
    coeff = x0.new_zeros((x0.shape[0], d_p + 2, 3))
    for k in range(d_p + 2):
        if k <= d_p:
            coeff[:, k, :] += x_center_h @ camera_poly.p_coeff[k].T
        if 0 <= k - 1 <= d_p:
            coeff[:, k, :] += x_slope_h @ camera_poly.p_coeff[k - 1].T
    return coeff.contiguous()


def compile_depth_coeff(x0: Tensor, velocity: Tensor, t0: Tensor, camera_poly: CameraPathPolynomial) -> Tensor:
    x_center_h, x_slope_h = _homogeneous_center_terms(
        x0,
        velocity,
        t0,
        camera_poly.t_center,
        camera_poly.t_scale,
    )
    d_w = int(camera_poly.w_coeff.shape[0] - 1)
    coeff = x0.new_zeros((x0.shape[0], d_w + 2))
    for k in range(d_w + 2):
        if k <= d_w:
            coeff[:, k] += x_center_h @ camera_poly.w_coeff[k, 2, :]
        if 0 <= k - 1 <= d_w:
            coeff[:, k] += x_slope_h @ camera_poly.w_coeff[k - 1, 2, :]
    return coeff.contiguous()


def eval_poly(coeff: Tensor, times: Tensor, *, t_center: float, t_scale: float) -> Tensor:
    tau = (times - float(t_center)) / float(t_scale)
    powers = [torch.ones_like(tau)]
    for _ in range(int(coeff.shape[1]) - 1):
        powers.append(powers[-1] * tau)
    basis = torch.stack(powers, dim=0)
    return torch.einsum("nkd,kf->nfd", coeff, basis)


def eval_scalar_poly(coeff: Tensor, times: Tensor, *, t_center: float, t_scale: float) -> Tensor:
    tau = (times - float(t_center)) / float(t_scale)
    powers = [torch.ones_like(tau)]
    for _ in range(int(coeff.shape[1]) - 1):
        powers.append(powers[-1] * tau)
    basis = torch.stack(powers, dim=0)
    return torch.einsum("nk,kf->nf", coeff, basis)


def eval_projective_centers(prt: ProjectiveRationalTubeSequence, times: Tensor, *, min_hz: float = 1.0e-6) -> Tensor:
    h = eval_poly(prt.h_coeff, times, t_center=prt.t_center, t_scale=prt.t_scale)
    hz = h[..., 2].clamp_min(float(min_hz))
    return torch.stack((h[..., 0] / hz, h[..., 1] / hz), dim=-1)


def eval_curve_centers(curve: CompiledCurveTubeSequence, times: Tensor) -> Tensor:
    return eval_poly(curve.center_coeff, times, t_center=curve.t_center, t_scale=curve.t_scale)


def direct_project_points(points: Tensor, K: Tensor, w2c: Tensor, *, min_depth: float = 1.0e-4) -> tuple[Tensor, Tensor]:
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2].clamp_min(float(min_depth))
    pixels = torch.stack((K[0, 0] * cam[:, 0] / z + K[0, 2], K[1, 1] * cam[:, 1] / z + K[1, 2]), dim=-1)
    return pixels, z


def sample_world_tube_projection(
    x0: Tensor,
    velocity: Tensor,
    t0: Tensor,
    K_seq: Tensor,
    w2c_seq: Tensor,
    times: Tensor,
    *,
    min_depth: float = 1.0e-4,
) -> tuple[Tensor, Tensor]:
    pixels = []
    depths = []
    for frame, t in enumerate(times):
        points = x0 + velocity * (t - t0).unsqueeze(-1)
        pixel_f, depth_f = direct_project_points(points, K_seq[frame], w2c_seq[frame], min_depth=min_depth)
        pixels.append(pixel_f)
        depths.append(depth_f)
    return torch.stack(pixels, dim=1).contiguous(), torch.stack(depths, dim=1).contiguous()


def compute_reference_lambda_uv(
    x0: Tensor,
    velocity: Tensor,
    t0: Tensor,
    precision_xy: Tensor,
    K_ref: Tensor,
    w2c_ref: Tensor,
    t_ref: Tensor,
    *,
    min_depth: float = 1.0e-4,
) -> Tensor:
    if t_ref.ndim == 0:
        t_ref = t_ref.expand(x0.shape[0])
    points = x0 + velocity * (t_ref - t0).unsqueeze(-1)
    rotation = w2c_ref[:3, :3]
    translation = w2c_ref[:3, 3]
    cam = points @ rotation.T + translation
    z = cam[:, 2].clamp_min(float(min_depth))
    inv_z = 1.0 / z
    inv_z2 = inv_z.square()
    du_dy = torch.stack((K_ref[0, 0] * inv_z, torch.zeros_like(z), -K_ref[0, 0] * cam[:, 0] * inv_z2), dim=-1)
    dv_dy = torch.stack((torch.zeros_like(z), K_ref[1, 1] * inv_z, -K_ref[1, 1] * cam[:, 1] * inv_z2), dim=-1)
    basis_cam = rotation[:, :2]
    ju = du_dy @ basis_cam
    jv = dv_dy @ basis_cam
    var_x = 1.0 / precision_xy[:, 0].clamp_min(1.0e-6)
    var_y = 1.0 / precision_xy[:, 1].clamp_min(1.0e-6)
    cov_uu = ju[:, 0].square() * var_x + ju[:, 1].square() * var_y + 1.0e-6
    cov_uv = ju[:, 0] * jv[:, 0] * var_x + ju[:, 1] * jv[:, 1] * var_y
    cov_vv = jv[:, 0].square() * var_x + jv[:, 1].square() * var_y + 1.0e-6
    det = (cov_uu * cov_vv - cov_uv.square()).clamp_min(1.0e-12)
    return torch.stack((cov_vv / det, -cov_uv / det, cov_uu / det), dim=-1).contiguous()


def compile_projective_rational_tubes(
    *,
    x0: Tensor,
    velocity: Tensor,
    t0: Tensor,
    precision_xy: Tensor,
    lambda_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    K_seq: Tensor,
    w2c_seq: Tensor,
    times: Tensor,
    camera_degree: int = 2,
    reference_time_mode: str = "tube_t0",
) -> ProjectiveRationalTubeSequence:
    camera_poly = fit_camera_path_polynomial(K_seq, w2c_seq, times, degree=int(camera_degree))
    h_coeff = compile_h_coeff(x0, velocity, t0, camera_poly)
    depth_coeff = compile_depth_coeff(x0, velocity, t0, camera_poly)
    if reference_time_mode == "window_center":
        t_ref = x0.new_full((x0.shape[0],), float(camera_poly.t_center))
    elif reference_time_mode == "tube_t0":
        t_ref = t0
    else:
        raise ValueError("reference_time_mode must be tube_t0 or window_center")
    ref_frame = int(times.numel() // 2)
    lambda_uv = compute_reference_lambda_uv(
        x0,
        velocity,
        t0,
        precision_xy,
        K_seq[ref_frame],
        w2c_seq[ref_frame],
        t_ref,
    )
    prt = ProjectiveRationalTubeSequence(
        h_coeff=h_coeff,
        lambda_uv=lambda_uv,
        lambda_t=lambda_t.contiguous(),
        center_t=t0.contiguous(),
        depth_coeff=depth_coeff,
        opacity=opacity.contiguous(),
        color=color.contiguous(),
        t_center=camera_poly.t_center,
        t_scale=camera_poly.t_scale,
        camera_fit_error=camera_poly.fit_error_fro,
    )
    direct_centers, _ = sample_world_tube_projection(x0, velocity, t0, K_seq, w2c_seq, times)
    approx_centers = eval_projective_centers(prt, times)
    residual = (approx_centers - direct_centers).abs().amax()
    return ProjectiveRationalTubeSequence(**{**prt.__dict__, "projected_probe_error_px": residual})


def compile_world_tube_batch_projective_rational(
    batch: WorldTubeBatch,
    *,
    K_seq: Tensor,
    w2c_seq: Tensor,
    times: Tensor,
    camera_degree: int = 2,
    reference_time_mode: str = "tube_t0",
) -> ProjectiveRationalTubeSequence:
    return compile_projective_rational_tubes(
        x0=batch.x0,
        velocity=batch.velocity,
        t0=batch.t0,
        precision_xy=batch.precision_xy,
        lambda_t=batch.lambda_t,
        opacity=batch.opacity,
        color=batch.color,
        K_seq=K_seq,
        w2c_seq=w2c_seq,
        times=times,
        camera_degree=camera_degree,
        reference_time_mode=reference_time_mode,
    )


def compile_curve_tubes_from_samples(
    *,
    center_samples: Tensor,
    depth_samples: Tensor,
    times: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    degree: int = 3,
) -> CompiledCurveTubeSequence:
    t_center, t_scale = _time_center_scale(times)
    center_by_frame = center_samples.transpose(0, 1).contiguous()
    depth_by_frame = depth_samples.transpose(0, 1).contiguous()
    center_coeff_f, center_err, _, _ = fit_polynomial_values(
        center_by_frame,
        times,
        degree,
        t_center=t_center,
        t_scale=t_scale,
    )
    depth_coeff_f, depth_err, _, _ = fit_polynomial_values(
        depth_by_frame,
        times,
        degree,
        t_center=t_center,
        t_scale=t_scale,
    )
    return CompiledCurveTubeSequence(
        center_coeff=center_coeff_f.permute(1, 0, 2).contiguous(),
        lambda_uv=lambda_uv.contiguous(),
        lambda_t=lambda_t.contiguous(),
        center_t=center_t.contiguous(),
        depth_coeff=depth_coeff_f.permute(1, 0).contiguous(),
        opacity=opacity.contiguous(),
        color=color.contiguous(),
        t_center=t_center,
        t_scale=t_scale,
        fit_error_px=torch.maximum(center_err, depth_err),
    )


def compile_world_tube_batch_curve_from_samples(
    batch: WorldTubeBatch,
    *,
    center_samples: Tensor,
    depth_samples: Tensor,
    times: Tensor,
    lambda_uv: Tensor,
    degree: int = 3,
) -> CompiledCurveTubeSequence:
    return compile_curve_tubes_from_samples(
        center_samples=center_samples,
        depth_samples=depth_samples,
        times=times,
        lambda_uv=lambda_uv,
        lambda_t=batch.lambda_t,
        center_t=batch.t0,
        opacity=batch.opacity,
        color=batch.color,
        degree=degree,
    )


def _render_from_centers(
    centers: Tensor,
    depths: Tensor,
    lambda_uv: Tensor,
    lambda_t: Tensor,
    center_t: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: CurveTubeRenderConfig,
) -> Tensor:
    device = centers.device
    dtype = centers.dtype
    bg = torch.tensor(config.background, dtype=dtype, device=device)
    out = torch.empty((config.frames, config.height, config.width, 3), dtype=dtype, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(config.height, dtype=dtype, device=device) + 0.5,
        torch.arange(config.width, dtype=dtype, device=device) + 0.5,
        indexing="ij",
    )
    frame_times = centered_frame_times(config.frames, device=device).to(dtype=dtype)
    for frame in range(config.frames):
        order = torch.argsort(depths[:, frame].detach(), stable=True).detach().cpu()
        accum = torch.zeros((config.height, config.width, 3), dtype=dtype, device=device)
        trans = torch.ones((config.height, config.width), dtype=dtype, device=device)
        t = frame_times[frame]
        temporal = 0.5 * lambda_t * (t - center_t).square()
        for idx in order.tolist():
            du = xx - centers[idx, frame, 0]
            dv = yy - centers[idx, frame, 1]
            e_uv = 0.5 * (
                lambda_uv[idx, 0] * du.square()
                + 2.0 * lambda_uv[idx, 1] * du * dv
                + lambda_uv[idx, 2] * dv.square()
            )
            alpha = opacity[idx] * torch.exp(-(e_uv + temporal[idx]))
            alpha = torch.where(alpha >= config.alpha_threshold, alpha.clamp(max=config.max_alpha), torch.zeros_like(alpha))
            accum = accum + trans.unsqueeze(-1) * alpha.unsqueeze(-1) * color[idx]
            trans = trans * (1.0 - alpha)
            if bool((trans < config.transmittance_threshold).all()):
                break
        out[frame] = accum + trans.unsqueeze(-1) * bg
    return out


def dense_render_projective_rational_tubes(prt: ProjectiveRationalTubeSequence, config: CurveTubeRenderConfig) -> Tensor:
    times = centered_frame_times(config.frames, device=prt.h_coeff.device).to(dtype=prt.h_coeff.dtype)
    centers = eval_projective_centers(prt, times, min_hz=config.min_hz)
    depths = eval_scalar_poly(prt.depth_coeff, times, t_center=prt.t_center, t_scale=prt.t_scale).clamp_min(config.min_depth)
    return _render_from_centers(centers, depths, prt.lambda_uv, prt.lambda_t, prt.center_t, prt.opacity, prt.color, config)


def dense_render_compiled_curve_tubes(curve: CompiledCurveTubeSequence, config: CurveTubeRenderConfig) -> Tensor:
    times = centered_frame_times(config.frames, device=curve.center_coeff.device).to(dtype=curve.center_coeff.dtype)
    centers = eval_curve_centers(curve, times)
    depths = eval_scalar_poly(curve.depth_coeff, times, t_center=curve.t_center, t_scale=curve.t_scale).clamp_min(config.min_depth)
    return _render_from_centers(centers, depths, curve.lambda_uv, curve.lambda_t, curve.center_t, curve.opacity, curve.color, config)


def psnr(a: Tensor, b: Tensor) -> float:
    mse = float((a - b).square().mean().detach().cpu())
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)
