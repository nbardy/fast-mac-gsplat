from __future__ import annotations

from dataclasses import dataclass
import math

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
class CameraPathPolynomial:
    p_coeff: Tensor
    frame_times: Tensor
    fit_error: float


@dataclass(frozen=True)
class ProjectiveRationalTubeSequence:
    h_coeff: Tensor
    lambda_uv: Tensor
    lambda_t: Tensor
    center_t: Tensor
    depth_coeff: Tensor
    opacity: Tensor
    color: Tensor
    camera_fit_error: float


def centered_frame_times(frames: int, *, device: torch.device | str = "cpu") -> Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    values = torch.arange(frames, dtype=torch.float32, device=device)
    return values - 0.5 * float(frames - 1)


def _check_world_tubes(batch: WorldTubeBatch) -> None:
    tube_count = int(batch.x0.shape[0])
    expected = {
        "x0": (tube_count, 3),
        "velocity": (tube_count, 3),
        "t0": (tube_count,),
        "precision_xy": (tube_count, 2),
        "lambda_t": (tube_count,),
        "opacity": (tube_count,),
        "color": (tube_count, 3),
    }
    for name, shape in expected.items():
        value = getattr(batch, name)
        if not torch.is_tensor(value):
            raise ValueError(f"{name} must be a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if value.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if value.device != batch.x0.device:
            raise ValueError(f"{name} must be on the same device as x0")


def projection_matrices(K_seq: Tensor, w2c_seq: Tensor) -> Tensor:
    if K_seq.ndim != 3 or K_seq.shape[1:] != (3, 3):
        raise ValueError("K_seq must have shape [F,3,3]")
    if w2c_seq.ndim != 3 or w2c_seq.shape[1:] != (4, 4):
        raise ValueError("w2c_seq must have shape [F,4,4]")
    if K_seq.shape[0] != w2c_seq.shape[0]:
        raise ValueError("K_seq and w2c_seq must have the same frame count")
    if K_seq.dtype != torch.float32 or w2c_seq.dtype != torch.float32:
        raise ValueError("K_seq and w2c_seq must be float32")
    return torch.bmm(K_seq, w2c_seq[:, :3, :])


def fit_camera_path_polynomial(
    K_seq: Tensor,
    w2c_seq: Tensor,
    *,
    degree: int,
    frame_times: Tensor | None = None,
) -> CameraPathPolynomial:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    p_seq = projection_matrices(K_seq, w2c_seq)
    frames = int(p_seq.shape[0])
    times = centered_frame_times(frames, device=p_seq.device) if frame_times is None else frame_times.to(p_seq.device)
    if times.shape != (frames,):
        raise ValueError(f"frame_times must have shape ({frames},)")
    if frames < degree + 1:
        raise ValueError("need at least degree + 1 frames to fit a camera polynomial")

    vandermonde = torch.stack([times.pow(k) for k in range(degree + 1)], dim=-1)
    solution = torch.linalg.lstsq(vandermonde.to(torch.float64), p_seq.reshape(frames, 12).to(torch.float64)).solution
    coeff = solution.to(torch.float32).reshape(degree + 1, 3, 4)
    recon = evaluate_camera_polynomial(coeff, times)
    denom = p_seq.reshape(frames, -1).norm(dim=1).clamp_min(1.0e-8)
    rel = (recon - p_seq).reshape(frames, -1).norm(dim=1) / denom
    return CameraPathPolynomial(p_coeff=coeff, frame_times=times, fit_error=float(rel.max().detach().cpu()))


def evaluate_camera_polynomial(p_coeff: Tensor, times: Tensor) -> Tensor:
    if p_coeff.ndim != 3 or p_coeff.shape[1:] != (3, 4):
        raise ValueError("p_coeff must have shape [D+1,3,4]")
    powers = torch.stack([times.pow(k) for k in range(int(p_coeff.shape[0]))], dim=-1)
    return torch.einsum("fd,drc->frc", powers, p_coeff)


def _shift_polynomial_to_tube_centers(p_coeff: Tensor, center_t: Tensor) -> Tensor:
    degree = int(p_coeff.shape[0]) - 1
    shifted = p_coeff.new_zeros((int(center_t.numel()), degree + 1, 3, 4))
    for out_power in range(degree + 1):
        accum = p_coeff.new_zeros((int(center_t.numel()), 3, 4))
        for in_power in range(out_power, degree + 1):
            scale = math.comb(in_power, out_power) * center_t.pow(in_power - out_power)
            accum = accum + scale.view(-1, 1, 1) * p_coeff[in_power].view(1, 3, 4)
        shifted[:, out_power] = accum
    return shifted


def _project_points_with_matrices(p_matrix: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
    if points.shape[-1] != 3:
        raise ValueError("points must end with xyz")
    ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
    hom = torch.cat((points, ones), dim=-1)
    if p_matrix.ndim == 2:
        h = hom @ p_matrix.T
    elif p_matrix.ndim == 3 and hom.ndim == 2:
        h = torch.einsum("nrc,nc->nr", p_matrix, hom)
    elif p_matrix.ndim == 3 and hom.ndim == 3:
        h = torch.einsum("frc,fnc->fnr", p_matrix, hom)
    else:
        raise ValueError("unsupported p_matrix/points rank combination")
    z = h[..., 2].clamp_min(1.0e-6)
    return torch.stack((h[..., 0] / z, h[..., 1] / z), dim=-1), z


def _lambda_uv_from_reference_jacobian(batch: WorldTubeBatch, p_ref: Tensor, *, eps: float = 1.0e-3) -> Tensor:
    base, _ = _project_points_with_matrices(p_ref, batch.x0)
    offset_x = torch.tensor([eps, 0.0, 0.0], dtype=torch.float32, device=batch.x0.device).view(1, 3)
    offset_y = torch.tensor([0.0, eps, 0.0], dtype=torch.float32, device=batch.x0.device).view(1, 3)
    px, _ = _project_points_with_matrices(p_ref, batch.x0 + offset_x)
    py, _ = _project_points_with_matrices(p_ref, batch.x0 + offset_y)
    jac = torch.stack(((px - base) / eps, (py - base) / eps), dim=-1)
    var = 1.0 / batch.precision_xy.clamp_min(1.0e-6)
    cov = (
        jac[:, :, :1] * jac[:, :, :1].transpose(1, 2) * var[:, 0].view(-1, 1, 1)
        + jac[:, :, 1:] * jac[:, :, 1:].transpose(1, 2) * var[:, 1].view(-1, 1, 1)
    )
    cov = cov + torch.eye(2, dtype=torch.float32, device=batch.x0.device).view(1, 2, 2) * 1.0e-6
    det = (cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]).clamp_min(1.0e-20)
    inv_det = 1.0 / det
    return torch.stack(
        (
            cov[:, 1, 1] * inv_det,
            -cov[:, 0, 1] * inv_det,
            cov[:, 0, 0] * inv_det,
        ),
        dim=-1,
    )


def compile_projective_rational_tubes(
    batch: WorldTubeBatch,
    camera_path: CameraPathPolynomial,
) -> ProjectiveRationalTubeSequence:
    _check_world_tubes(batch)
    if camera_path.p_coeff.device != batch.x0.device:
        raise ValueError("camera_path p_coeff must be on the same device as the batch")
    shifted_p = _shift_polynomial_to_tube_centers(camera_path.p_coeff, batch.t0)
    camera_degree = int(camera_path.p_coeff.shape[0]) - 1
    h_degree = camera_degree + 1
    h_coeff = batch.x0.new_zeros((int(batch.x0.shape[0]), h_degree + 1, 3))
    x0_h = torch.cat((batch.x0, torch.ones((int(batch.x0.shape[0]), 1), dtype=torch.float32, device=batch.x0.device)), dim=-1)
    v_h = torch.cat((batch.velocity, torch.zeros((int(batch.x0.shape[0]), 1), dtype=torch.float32, device=batch.x0.device)), dim=-1)
    for power in range(h_degree + 1):
        if power <= camera_degree:
            h_coeff[:, power] = h_coeff[:, power] + torch.einsum("nrc,nc->nr", shifted_p[:, power], x0_h)
        if 0 <= power - 1 <= camera_degree:
            h_coeff[:, power] = h_coeff[:, power] + torch.einsum("nrc,nc->nr", shifted_p[:, power - 1], v_h)

    p_ref = shifted_p[:, 0]
    lambda_uv = _lambda_uv_from_reference_jacobian(batch, p_ref)
    depth_coeff = h_coeff[:, :, 2].contiguous()
    return ProjectiveRationalTubeSequence(
        h_coeff=h_coeff,
        lambda_uv=lambda_uv,
        lambda_t=batch.lambda_t,
        center_t=batch.t0,
        depth_coeff=depth_coeff,
        opacity=batch.opacity,
        color=batch.color,
        camera_fit_error=camera_path.fit_error,
    )


def evaluate_projective_centers(projected: ProjectiveRationalTubeSequence, times: Tensor) -> tuple[Tensor, Tensor]:
    tau = times.to(projected.h_coeff.device).view(-1, 1) - projected.center_t.view(1, -1)
    powers = torch.stack([tau.pow(k) for k in range(int(projected.h_coeff.shape[1]))], dim=-1)
    h = torch.einsum("fnd,ndc->fnc", powers, projected.h_coeff)
    z = h[..., 2].clamp_min(1.0e-6)
    return torch.stack((h[..., 0] / z, h[..., 1] / z), dim=-1), z


def direct_project_world_tubes(
    batch: WorldTubeBatch,
    K_seq: Tensor,
    w2c_seq: Tensor,
    frame_times: Tensor,
) -> tuple[Tensor, Tensor]:
    p_seq = projection_matrices(K_seq, w2c_seq)
    tau = frame_times.to(batch.x0.device).view(-1, 1, 1) - batch.t0.view(1, -1, 1)
    points = batch.x0.view(1, -1, 3) + batch.velocity.view(1, -1, 3) * tau
    return _project_points_with_matrices(p_seq, points)


def dense_render_projective_rational_tubes(
    projected: ProjectiveRationalTubeSequence,
    *,
    height: int,
    width: int,
    frame_times: Tensor,
    alpha_threshold: float = 1.0 / 255.0,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    device = projected.h_coeff.device
    centers, depth = evaluate_projective_centers(projected, frame_times.to(device))
    y = torch.arange(height, dtype=torch.float32, device=device) + 0.5
    x = torch.arange(width, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    image = torch.empty((int(frame_times.numel()), height, width, 3), dtype=torch.float32, device=device)
    bg = torch.tensor(background, dtype=torch.float32, device=device).view(1, 1, 3)
    for frame in range(int(frame_times.numel())):
        order = torch.argsort(depth[frame].detach(), stable=True)
        accum = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
        transmittance = torch.ones((height, width, 1), dtype=torch.float32, device=device)
        time_delta = frame_times[frame].to(device) - projected.center_t
        for tube_id in order.tolist():
            lambda_uu, lambda_uv, lambda_vv = projected.lambda_uv[tube_id]
            du = xx - centers[frame, tube_id, 0]
            dv = yy - centers[frame, tube_id, 1]
            spatial = lambda_uu * du.square() + 2.0 * lambda_uv * du * dv + lambda_vv * dv.square()
            temporal = projected.lambda_t[tube_id] * time_delta[tube_id].square()
            exponent = 0.5 * (spatial + temporal)
            alpha = (projected.opacity[tube_id] * torch.exp(-exponent)).clamp(max=0.99)
            alpha = torch.where(alpha >= alpha_threshold, alpha, torch.zeros_like(alpha))
            alpha_3 = alpha.unsqueeze(-1)
            accum = accum + transmittance * alpha_3 * projected.color[tube_id].view(1, 1, 3)
            transmittance = transmittance * (1.0 - alpha_3)
        image[frame] = accum + transmittance * bg
    return image


def affine_taylor_center_residual(
    direct_centers: Tensor,
    frame_times: Tensor,
    *,
    center_frame: int | None = None,
) -> float:
    frames = int(frame_times.numel())
    if frames < 3:
        raise ValueError("need at least 3 frames for an affine residual diagnostic")
    mid = frames // 2 if center_frame is None else int(center_frame)
    left = max(0, mid - 1)
    right = min(frames - 1, mid + 1)
    slope = (direct_centers[right] - direct_centers[left]) / (frame_times[right] - frame_times[left]).clamp_min(1.0e-6)
    affine = direct_centers[mid].unsqueeze(0) + (frame_times - frame_times[mid]).view(-1, 1, 1) * slope.unsqueeze(0)
    return float((affine - direct_centers).abs().max().detach().cpu())


def curvature_selective_mask(
    direct_centers: Tensor,
    frame_times: Tensor,
    *,
    threshold_px: float,
) -> Tensor:
    residuals = []
    for tube in range(int(direct_centers.shape[1])):
        residuals.append(
            affine_taylor_center_residual(direct_centers[:, tube : tube + 1, :], frame_times)
        )
    return torch.tensor(residuals, dtype=torch.float32, device=direct_centers.device) > float(threshold_px)
