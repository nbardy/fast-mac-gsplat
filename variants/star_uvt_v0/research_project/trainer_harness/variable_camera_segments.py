from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from torch_gsplat_bridge_star_uvt import UVTRenderConfig

try:
    from .world_tube import WorldTubeBatch
except ImportError:  # pragma: no cover - direct script execution fallback.
    from world_tube import WorldTubeBatch


@dataclass(frozen=True)
class SegmentDiagnostics:
    segment_count: int
    mean_segments_per_tube: float
    frames_per_segment: int
    temporal_chunk_count: int


@dataclass(frozen=True)
class ProjectedVariableCameraSegments:
    ma: Tensor
    q_uvt: Tensor
    depth0: Tensor
    depth_beta: Tensor
    opacity: Tensor
    color: Tensor
    parent_id: Tensor
    t_minmax: Tensor
    diagnostics: SegmentDiagnostics


def _check_world_tube_batch(batch: WorldTubeBatch) -> None:
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
        tensor = getattr(batch, name)
        if not torch.is_tensor(tensor):
            raise ValueError(f"{name} must be a tensor")
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != batch.x0.device:
            raise ValueError(f"{name} must be on the same device as x0")


def _check_camera_sequence(K_seq: Tensor, w2c_seq: Tensor, full_frames: int, device: torch.device) -> None:
    if K_seq.shape != (full_frames, 3, 3):
        raise ValueError(f"K_seq must have shape [{full_frames},3,3]")
    if w2c_seq.shape != (full_frames, 4, 4):
        raise ValueError(f"w2c_seq must have shape [{full_frames},4,4]")
    for name, tensor in {"K_seq": K_seq, "w2c_seq": w2c_seq}.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same device as the batch")


def _global_centered_time(frame: Tensor | float, full_frames: int) -> Tensor | float:
    return frame - 0.5 * float(full_frames - 1)


def _global_to_local_time(global_t: Tensor, full_frames: int, config: UVTRenderConfig, frame_start: int) -> Tensor:
    offset = float(frame_start) - 0.5 * float(full_frames - 1) + 0.5 * float(int(config.frames) - 1)
    return global_t - offset


def _project_pinhole(points: Tensor, K: Tensor, w2c: Tensor, *, min_depth: float) -> tuple[Tensor, Tensor]:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    camera_points = points @ rotation.T + translation
    z = camera_points[:, 2].clamp_min(min_depth)
    pixels = torch.stack(
        (
            K[0, 0] * camera_points[:, 0] / z + K[0, 2],
            K[1, 1] * camera_points[:, 1] / z + K[1, 2],
        ),
        dim=-1,
    )
    return pixels, z


def _screen_precision(batch: WorldTubeBatch, points: Tensor, K: Tensor, w2c: Tensor, *, min_depth: float) -> Tensor:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    camera_points = points @ rotation.T + translation
    z = camera_points[:, 2].clamp_min(min_depth)
    inv_z = 1.0 / z
    inv_z2 = inv_z.square()
    du_dx = K[0, 0] * inv_z
    du_dz = -K[0, 0] * camera_points[:, 0] * inv_z2
    dv_dy = K[1, 1] * inv_z
    dv_dz = -K[1, 1] * camera_points[:, 1] * inv_z2

    proj_u_x = du_dx * rotation[0, 0] + du_dz * rotation[2, 0]
    proj_u_y = du_dx * rotation[0, 1] + du_dz * rotation[2, 1]
    proj_v_x = dv_dy * rotation[1, 0] + dv_dz * rotation[2, 0]
    proj_v_y = dv_dy * rotation[1, 1] + dv_dz * rotation[2, 1]

    world_var_x = 1.0 / batch.precision_xy[:, 0].clamp_min(1.0e-6)
    world_var_y = 1.0 / batch.precision_xy[:, 1].clamp_min(1.0e-6)
    cov_uu = proj_u_x.square() * world_var_x + proj_u_y.square() * world_var_y + 1.0e-6
    cov_uv = proj_u_x * proj_v_x * world_var_x + proj_u_y * proj_v_y * world_var_y
    cov_vv = proj_v_x.square() * world_var_x + proj_v_y.square() * world_var_y + 1.0e-6
    inv_det = 1.0 / (cov_uu * cov_vv - cov_uv.square()).clamp_min(1.0e-12)
    return torch.stack((cov_vv * inv_det, -cov_uv * inv_det, cov_uu * inv_det), dim=-1)


def _local_lambda_t(batch_lambda_t: Tensor, frame_count: int, config: UVTRenderConfig) -> Tensor:
    half_width = max(0.5, 0.5 * float(frame_count))
    if config.alpha_mode == "peak_splat":
        support_numerator = float(config.alpha_threshold)
    elif config.alpha_mode == "beer_lambert":
        support_numerator = -math.log1p(-float(config.alpha_threshold))
    else:
        raise ValueError("alpha_mode must be one of: peak_splat, beer_lambert")
    support_numerator = max(support_numerator, 1.0e-12)
    segment_lambda = (
        2.0 * math.log(1.0 / support_numerator) / (half_width * half_width)
    )
    return batch_lambda_t.clamp_min(float(segment_lambda))


def project_piecewise_camera_time_segments(
    batch: WorldTubeBatch,
    K_seq: Tensor,
    w2c_seq: Tensor,
    config: UVTRenderConfig,
    *,
    full_frames: int,
    frame_start: int = 0,
    frames_per_segment: int,
    min_depth: float = 1.0e-4,
) -> ProjectedVariableCameraSegments:
    """Flatten world tubes into projected UVT segments for a moving camera.

    Each chunk uses the midpoint camera and a finite-difference screen velocity
    over neighboring discrete camera frames. The current renderer has no hard
    `t_minmax` support, so temporal locality is only approximated by centering
    every segment at its midpoint and increasing `lambda_t` enough that the
    segment tail reaches `config.alpha_threshold` near the chunk half-width.
    """

    if full_frames <= 0:
        raise ValueError("full_frames must be positive")
    if frames_per_segment <= 0:
        raise ValueError("frames_per_segment must be positive")
    if frame_start < 0 or frame_start + int(config.frames) > full_frames:
        raise ValueError(
            f"frame window [{frame_start}, {frame_start + int(config.frames)}) exceeds {full_frames}"
        )

    _check_world_tube_batch(batch)
    _check_camera_sequence(K_seq, w2c_seq, int(full_frames), batch.x0.device)

    window_end = frame_start + int(config.frames)
    segments = [
        (start, min(start + int(frames_per_segment), window_end))
        for start in range(frame_start, window_end, int(frames_per_segment))
    ]
    tube_count = int(batch.x0.shape[0])

    ma_parts: list[Tensor] = []
    q_parts: list[Tensor] = []
    depth0_parts: list[Tensor] = []
    depth_beta_parts: list[Tensor] = []
    parent_parts: list[Tensor] = []
    t_minmax_parts: list[Tensor] = []

    for start, end in segments:
        mid_frame = 0.5 * float(start + end - 1)
        mid_index = int(round(mid_frame))
        global_mid_t = torch.full(
            (tube_count,),
            _global_centered_time(mid_frame, full_frames),
            dtype=torch.float32,
            device=batch.x0.device,
        )
        world_mid = batch.x0 + batch.velocity * (global_mid_t - batch.t0).unsqueeze(-1)
        K_mid = K_seq[mid_index]
        w2c_mid = w2c_seq[mid_index]
        pixels, depth = _project_pinhole(world_mid, K_mid, w2c_mid, min_depth=min_depth)
        precision = _screen_precision(batch, world_mid, K_mid, w2c_mid, min_depth=min_depth)

        prev_frame = max(0, mid_index - 1)
        next_frame = min(full_frames - 1, mid_index + 1)
        if next_frame == prev_frame:
            velocity_uv = torch.zeros((tube_count, 2), dtype=torch.float32, device=batch.x0.device)
            depth_velocity = torch.zeros((tube_count,), dtype=torch.float32, device=batch.x0.device)
        else:
            prev_t = torch.full(
                (tube_count,),
                _global_centered_time(float(prev_frame), full_frames),
                dtype=torch.float32,
                device=batch.x0.device,
            )
            next_t = torch.full(
                (tube_count,),
                _global_centered_time(float(next_frame), full_frames),
                dtype=torch.float32,
                device=batch.x0.device,
            )
            world_prev = batch.x0 + batch.velocity * (prev_t - batch.t0).unsqueeze(-1)
            world_next = batch.x0 + batch.velocity * (next_t - batch.t0).unsqueeze(-1)
            pixels_prev, depth_prev = _project_pinhole(
                world_prev,
                K_seq[prev_frame],
                w2c_seq[prev_frame],
                min_depth=min_depth,
            )
            pixels_next, depth_next = _project_pinhole(
                world_next,
                K_seq[next_frame],
                w2c_seq[next_frame],
                min_depth=min_depth,
            )
            dt = float(next_frame - prev_frame)
            velocity_uv = (pixels_next - pixels_prev) / dt
            depth_velocity = (depth_next - depth_prev) / dt

        lambda_u = precision[:, 0]
        lambda_uv = precision[:, 1]
        lambda_v = precision[:, 2]
        velocity_u = velocity_uv[:, 0]
        velocity_v = velocity_uv[:, 1]
        lambda_t = _local_lambda_t(batch.lambda_t, end - start, config)
        q_uvt = torch.stack(
            (
                lambda_u,
                lambda_uv,
                -(lambda_u * velocity_u + lambda_uv * velocity_v),
                lambda_v,
                -(lambda_uv * velocity_u + lambda_v * velocity_v),
                lambda_t
                + lambda_u * velocity_u.square()
                + 2.0 * lambda_uv * velocity_u * velocity_v
                + lambda_v * velocity_v.square(),
            ),
            dim=-1,
        )
        local_mid_t = _global_to_local_time(global_mid_t, full_frames, config, frame_start)
        ma_parts.append(torch.cat((pixels, local_mid_t.unsqueeze(-1)), dim=-1))
        q_parts.append(q_uvt)
        depth0_parts.append(depth)
        depth_beta = torch.zeros((tube_count, 3), dtype=torch.float32, device=batch.x0.device)
        depth_beta[:, 2] = depth_velocity
        depth_beta_parts.append(depth_beta)
        parent_parts.append(torch.arange(tube_count, dtype=torch.long, device=batch.x0.device))
        minmax_global = torch.tensor(
            [_global_centered_time(float(start), full_frames), _global_centered_time(float(end - 1), full_frames)],
            dtype=torch.float32,
            device=batch.x0.device,
        )
        t_minmax_parts.append(
            _global_to_local_time(minmax_global, full_frames, config, frame_start).repeat(tube_count, 1)
        )

    temporal_chunk_count = len(segments)
    segment_count = tube_count * temporal_chunk_count
    return ProjectedVariableCameraSegments(
        ma=torch.cat(ma_parts, dim=0).contiguous(),
        q_uvt=torch.cat(q_parts, dim=0).contiguous(),
        depth0=torch.cat(depth0_parts, dim=0).contiguous(),
        depth_beta=torch.cat(depth_beta_parts, dim=0).contiguous(),
        opacity=batch.opacity.repeat(temporal_chunk_count).contiguous(),
        color=batch.color.repeat(temporal_chunk_count, 1).contiguous(),
        parent_id=torch.cat(parent_parts, dim=0).contiguous(),
        t_minmax=torch.cat(t_minmax_parts, dim=0).contiguous(),
        diagnostics=SegmentDiagnostics(
            segment_count=segment_count,
            mean_segments_per_tube=float(temporal_chunk_count),
            frames_per_segment=int(frames_per_segment),
            temporal_chunk_count=temporal_chunk_count,
        ),
    )
