from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from torch_gsplat_bridge_star_uvt import UVTRenderConfig

Scalar = float | Tensor


@dataclass(frozen=True)
class OrthoCamera2D:
    scale_u: float
    scale_v: float
    center_u: float
    center_v: float
    depth_offset: float = 0.0


@dataclass(frozen=True)
class PinholeCamera:
    fx: Scalar
    fy: Scalar
    cx: Scalar
    cy: Scalar
    world_to_camera: Tensor


@dataclass(frozen=True)
class PinholeCameraMotion:
    fx: Scalar
    fy: Scalar
    cx: Scalar
    cy: Scalar
    fx_dot: Scalar
    fy_dot: Scalar
    cx_dot: Scalar
    cy_dot: Scalar
    world_to_camera: Tensor
    world_to_camera_dot: Tensor
    chart_time: float


@dataclass(frozen=True)
class WorldTubeBatch:
    x0: Tensor
    velocity: Tensor
    t0: Tensor
    precision_xy: Tensor
    lambda_t: Tensor
    opacity: Tensor
    color: Tensor


def _check_batch(batch: WorldTubeBatch) -> None:
    tube_count = batch.x0.shape[0]
    if batch.x0.shape != (tube_count, 3):
        raise ValueError("x0 must have shape [N,3]")
    if batch.velocity.shape != (tube_count, 3):
        raise ValueError("velocity must have shape [N,3]")
    if batch.t0.shape != (tube_count,):
        raise ValueError("t0 must have shape [N]")
    if batch.precision_xy.shape != (tube_count, 2):
        raise ValueError("precision_xy must have shape [N,2]")
    if batch.lambda_t.shape != (tube_count,):
        raise ValueError("lambda_t must have shape [N]")
    if batch.opacity.shape != (tube_count,):
        raise ValueError("opacity must have shape [N]")
    if batch.color.shape != (tube_count, 3):
        raise ValueError("color must have shape [N,3]")
    for name, tensor in batch.__dict__.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"{name} must be a tensor")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != batch.x0.device:
            raise ValueError(f"{name} must be on the same device as x0")


def _scalar_tensor(value: Scalar, reference: Tensor, name: str) -> Tensor:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape {tuple(value.shape)}")
        return value.to(device=reference.device, dtype=reference.dtype).reshape(())
    return torch.tensor(float(value), dtype=reference.dtype, device=reference.device)


def _scalar_is_zero(value: Scalar) -> bool:
    if torch.is_tensor(value):
        return bool((value.detach() == 0.0).cpu().item())
    return float(value) == 0.0


def project_world_tubes_ortho(
    batch: WorldTubeBatch,
    camera: OrthoCamera2D,
    _config: UVTRenderConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project fronto-parallel world tubes to the Gate 0 UVT tensor contract."""

    del _config
    _check_batch(batch)
    if camera.scale_u == 0.0 or camera.scale_v == 0.0:
        raise ValueError("orthographic camera scales must be non-zero")

    scale_u = float(camera.scale_u)
    scale_v = float(camera.scale_v)
    center_u = batch.x0[:, 0] * scale_u + float(camera.center_u)
    center_v = batch.x0[:, 1] * scale_v + float(camera.center_v)
    velocity_u = batch.velocity[:, 0] * scale_u
    velocity_v = batch.velocity[:, 1] * scale_v
    lambda_u = batch.precision_xy[:, 0] / (scale_u * scale_u)
    lambda_v = batch.precision_xy[:, 1] / (scale_v * scale_v)

    ma = torch.stack((center_u, center_v, batch.t0), dim=-1)
    zeros = torch.zeros_like(lambda_u)
    q_uvt = torch.stack(
        (
            lambda_u,
            zeros,
            -lambda_u * velocity_u,
            lambda_v,
            -lambda_v * velocity_v,
            batch.lambda_t + lambda_u * velocity_u.square() + lambda_v * velocity_v.square(),
        ),
        dim=-1,
    )
    depth0 = batch.x0[:, 2] + float(camera.depth_offset)
    depth_beta = torch.zeros((batch.x0.shape[0], 3), dtype=torch.float32, device=batch.x0.device)
    depth_beta[:, 2] = batch.velocity[:, 2]
    return ma, q_uvt, depth0, depth_beta, batch.opacity, batch.color


def _check_pinhole_camera(camera: PinholeCamera, device: torch.device) -> None:
    if _scalar_is_zero(camera.fx) or _scalar_is_zero(camera.fy):
        raise ValueError("pinhole focal lengths must be non-zero")
    if camera.world_to_camera.shape != (4, 4):
        raise ValueError("world_to_camera must have shape [4,4]")
    if camera.world_to_camera.dtype != torch.float32:
        raise ValueError("world_to_camera must be float32")
    if camera.world_to_camera.device != device:
        raise ValueError("world_to_camera must be on the same device as the batch")


def _check_pinhole_camera_motion(camera: PinholeCameraMotion, device: torch.device) -> None:
    if _scalar_is_zero(camera.fx) or _scalar_is_zero(camera.fy):
        raise ValueError("pinhole focal lengths must be non-zero")
    if camera.world_to_camera.shape != (4, 4):
        raise ValueError("world_to_camera must have shape [4,4]")
    if camera.world_to_camera_dot.shape != (4, 4):
        raise ValueError("world_to_camera_dot must have shape [4,4]")
    if camera.world_to_camera.dtype != torch.float32:
        raise ValueError("world_to_camera must be float32")
    if camera.world_to_camera_dot.dtype != torch.float32:
        raise ValueError("world_to_camera_dot must be float32")
    if camera.world_to_camera.device != device:
        raise ValueError("world_to_camera must be on the same device as the batch")
    if camera.world_to_camera_dot.device != device:
        raise ValueError("world_to_camera_dot must be on the same device as the batch")


def project_world_tubes_pinhole(
    batch: WorldTubeBatch,
    camera: PinholeCamera,
    _config: UVTRenderConfig,
    *,
    min_depth: float = 1.0e-4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project fronto-parallel world tubes with a pinhole camera linearization."""

    del _config
    _check_batch(batch)
    _check_pinhole_camera(camera, batch.x0.device)

    rotation = camera.world_to_camera[:3, :3]
    translation = camera.world_to_camera[:3, 3]
    fx = _scalar_tensor(camera.fx, batch.x0, "fx")
    fy = _scalar_tensor(camera.fy, batch.x0, "fy")
    cx = _scalar_tensor(camera.cx, batch.x0, "cx")
    cy = _scalar_tensor(camera.cy, batch.x0, "cy")
    center_cam = batch.x0 @ rotation.T + translation
    velocity_cam = batch.velocity @ rotation.T

    z = center_cam[:, 2].clamp_min(min_depth)
    inv_z = 1.0 / z
    x_over_z = center_cam[:, 0] * inv_z
    y_over_z = center_cam[:, 1] * inv_z
    center_u = fx * x_over_z + cx
    center_v = fy * y_over_z + cy

    velocity_u = fx * (velocity_cam[:, 0] * z - center_cam[:, 0] * velocity_cam[:, 2]) * inv_z.square()
    velocity_v = fy * (velocity_cam[:, 1] * z - center_cam[:, 1] * velocity_cam[:, 2]) * inv_z.square()

    inv_z2 = inv_z.square()
    du_dx = fx * inv_z
    du_dz = -fx * center_cam[:, 0] * inv_z2
    dv_dy = fy * inv_z
    dv_dz = -fy * center_cam[:, 1] * inv_z2

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

    lambda_u = cov_vv * inv_det
    lambda_uv = -cov_uv * inv_det
    lambda_v = cov_uu * inv_det
    q_uvt = torch.stack(
        (
            lambda_u,
            lambda_uv,
            -(lambda_u * velocity_u + lambda_uv * velocity_v),
            lambda_v,
            -(lambda_uv * velocity_u + lambda_v * velocity_v),
            batch.lambda_t
            + lambda_u * velocity_u.square()
            + 2.0 * lambda_uv * velocity_u * velocity_v
            + lambda_v * velocity_v.square(),
        ),
        dim=-1,
    )
    ma = torch.stack((center_u, center_v, batch.t0), dim=-1)
    depth0 = z
    depth_beta = torch.zeros((batch.x0.shape[0], 3), dtype=torch.float32, device=batch.x0.device)
    depth_beta[:, 2] = velocity_cam[:, 2]
    return ma, q_uvt, depth0, depth_beta, batch.opacity, batch.color


def project_world_tubes_pinhole_motion(
    batch: WorldTubeBatch,
    camera: PinholeCameraMotion,
    _config: UVTRenderConfig,
    *,
    min_depth: float = 1.0e-4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project world tubes with a first-order moving pinhole camera chart."""

    del _config
    _check_batch(batch)
    _check_pinhole_camera_motion(camera, batch.x0.device)

    rotation = camera.world_to_camera[:3, :3]
    translation = camera.world_to_camera[:3, 3]
    rotation_dot = camera.world_to_camera_dot[:3, :3]
    translation_dot = camera.world_to_camera_dot[:3, 3]
    fx = _scalar_tensor(camera.fx, batch.x0, "fx")
    fy = _scalar_tensor(camera.fy, batch.x0, "fy")
    cx = _scalar_tensor(camera.cx, batch.x0, "cx")
    cy = _scalar_tensor(camera.cy, batch.x0, "cy")
    fx_dot = _scalar_tensor(camera.fx_dot, batch.x0, "fx_dot")
    fy_dot = _scalar_tensor(camera.fy_dot, batch.x0, "fy_dot")
    cx_dot = _scalar_tensor(camera.cx_dot, batch.x0, "cx_dot")
    cy_dot = _scalar_tensor(camera.cy_dot, batch.x0, "cy_dot")

    chart_time = float(camera.chart_time)
    x_s = batch.x0 + batch.velocity * (chart_time - batch.t0).unsqueeze(-1)
    center_cam = x_s @ rotation.T + translation
    velocity_cam = batch.velocity @ rotation.T + x_s @ rotation_dot.T + translation_dot

    z = center_cam[:, 2].clamp_min(min_depth)
    inv_z = 1.0 / z
    x_over_z = center_cam[:, 0] * inv_z
    y_over_z = center_cam[:, 1] * inv_z
    center_u = fx * x_over_z + cx
    center_v = fy * y_over_z + cy

    quotient_u_dot = (velocity_cam[:, 0] * z - center_cam[:, 0] * velocity_cam[:, 2]) * inv_z.square()
    quotient_v_dot = (velocity_cam[:, 1] * z - center_cam[:, 1] * velocity_cam[:, 2]) * inv_z.square()
    velocity_u = fx_dot * x_over_z + fx * quotient_u_dot + cx_dot
    velocity_v = fy_dot * y_over_z + fy * quotient_v_dot + cy_dot

    inv_z2 = inv_z.square()
    du_dx = fx * inv_z
    du_dz = -fx * center_cam[:, 0] * inv_z2
    dv_dy = fy * inv_z
    dv_dz = -fy * center_cam[:, 1] * inv_z2

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

    lambda_u = cov_vv * inv_det
    lambda_uv = -cov_uv * inv_det
    lambda_v = cov_uu * inv_det
    q_uvt = torch.stack(
        (
            lambda_u,
            lambda_uv,
            -(lambda_u * velocity_u + lambda_uv * velocity_v),
            lambda_v,
            -(lambda_uv * velocity_u + lambda_v * velocity_v),
            batch.lambda_t
            + lambda_u * velocity_u.square()
            + 2.0 * lambda_uv * velocity_u * velocity_v
            + lambda_v * velocity_v.square(),
        ),
        dim=-1,
    )
    time_center = torch.full_like(batch.t0, chart_time)
    ma = torch.stack((center_u, center_v, time_center), dim=-1)
    depth0 = z
    depth_beta = torch.zeros((batch.x0.shape[0], 3), dtype=torch.float32, device=batch.x0.device)
    depth_beta[:, 2] = velocity_cam[:, 2]
    return ma, q_uvt, depth0, depth_beta, batch.opacity, batch.color


def project_world_tubes_pinhole_projective_motion(
    batch: WorldTubeBatch,
    camera: PinholeCameraMotion,
    _config: UVTRenderConfig,
    *,
    min_depth: float = 1.0e-4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project moving pinhole cameras through the homogeneous camera-time gauge.

    This emits the same UVT contract as `project_world_tubes_pinhole_motion`,
    but computes the screen velocity from `h(t) = K(t) [R(t)|T(t)] X(t)`.
    """

    del _config
    _check_batch(batch)
    _check_pinhole_camera_motion(camera, batch.x0.device)

    world_to_camera = camera.world_to_camera
    world_to_camera_dot = camera.world_to_camera_dot
    fx = _scalar_tensor(camera.fx, batch.x0, "fx")
    fy = _scalar_tensor(camera.fy, batch.x0, "fy")
    cx = _scalar_tensor(camera.cx, batch.x0, "cx")
    cy = _scalar_tensor(camera.cy, batch.x0, "cy")
    fx_dot = _scalar_tensor(camera.fx_dot, batch.x0, "fx_dot")
    fy_dot = _scalar_tensor(camera.fy_dot, batch.x0, "fy_dot")
    cx_dot = _scalar_tensor(camera.cx_dot, batch.x0, "cx_dot")
    cy_dot = _scalar_tensor(camera.cy_dot, batch.x0, "cy_dot")
    zero = torch.zeros((), dtype=batch.x0.dtype, device=batch.x0.device)
    one = torch.ones((), dtype=batch.x0.dtype, device=batch.x0.device)
    K = torch.stack(
        (
            torch.stack((fx, zero, cx)),
            torch.stack((zero, fy, cy)),
            torch.stack((zero, zero, one)),
        )
    )
    K_dot = torch.stack(
        (
            torch.stack((fx_dot, zero, cx_dot)),
            torch.stack((zero, fy_dot, cy_dot)),
            torch.stack((zero, zero, zero)),
        )
    )
    P = K @ world_to_camera[:3, :]
    P_dot = K_dot @ world_to_camera[:3, :] + K @ world_to_camera_dot[:3, :]

    chart_time = float(camera.chart_time)
    x_s = batch.x0 + batch.velocity * (chart_time - batch.t0).unsqueeze(-1)
    ones = torch.ones((batch.x0.shape[0], 1), dtype=torch.float32, device=batch.x0.device)
    zeros = torch.zeros_like(ones)
    X = torch.cat((x_s, ones), dim=-1)
    X_dot = torch.cat((batch.velocity, zeros), dim=-1)

    h0 = X @ P.T
    h1 = X @ P_dot.T + X_dot @ P.T
    z = h0[:, 2].clamp_min(min_depth)
    inv_z = 1.0 / z
    center_u = h0[:, 0] * inv_z
    center_v = h0[:, 1] * inv_z
    velocity_u = (h1[:, 0] * z - h0[:, 0] * h1[:, 2]) * inv_z.square()
    velocity_v = (h1[:, 1] * z - h0[:, 1] * h1[:, 2]) * inv_z.square()

    inv_z2 = inv_z.square()
    dehom = torch.zeros((batch.x0.shape[0], 2, 3), dtype=torch.float32, device=batch.x0.device)
    dehom[:, 0, 0] = inv_z
    dehom[:, 0, 2] = -h0[:, 0] * inv_z2
    dehom[:, 1, 1] = inv_z
    dehom[:, 1, 2] = -h0[:, 1] * inv_z2
    pixel_jacobian_world = dehom @ P[:, :3]
    proj_u_x = pixel_jacobian_world[:, 0, 0]
    proj_u_y = pixel_jacobian_world[:, 0, 1]
    proj_v_x = pixel_jacobian_world[:, 1, 0]
    proj_v_y = pixel_jacobian_world[:, 1, 1]

    world_var_x = 1.0 / batch.precision_xy[:, 0].clamp_min(1.0e-6)
    world_var_y = 1.0 / batch.precision_xy[:, 1].clamp_min(1.0e-6)
    cov_uu = proj_u_x.square() * world_var_x + proj_u_y.square() * world_var_y + 1.0e-6
    cov_uv = proj_u_x * proj_v_x * world_var_x + proj_u_y * proj_v_y * world_var_y
    cov_vv = proj_v_x.square() * world_var_x + proj_v_y.square() * world_var_y + 1.0e-6
    inv_det = 1.0 / (cov_uu * cov_vv - cov_uv.square()).clamp_min(1.0e-12)

    lambda_u = cov_vv * inv_det
    lambda_uv = -cov_uv * inv_det
    lambda_v = cov_uu * inv_det
    q_uvt = torch.stack(
        (
            lambda_u,
            lambda_uv,
            -(lambda_u * velocity_u + lambda_uv * velocity_v),
            lambda_v,
            -(lambda_uv * velocity_u + lambda_v * velocity_v),
            batch.lambda_t
            + lambda_u * velocity_u.square()
            + 2.0 * lambda_uv * velocity_u * velocity_v
            + lambda_v * velocity_v.square(),
        ),
        dim=-1,
    )
    time_center = torch.full_like(batch.t0, chart_time)
    ma = torch.stack((center_u, center_v, time_center), dim=-1)
    depth0 = z
    depth_beta = torch.zeros((batch.x0.shape[0], 3), dtype=torch.float32, device=batch.x0.device)
    depth_beta[:, 2] = h1[:, 2]
    return ma, q_uvt, depth0, depth_beta, batch.opacity, batch.color


def project_world_tubes_from_pixel_jacobian(
    batch: WorldTubeBatch,
    world_to_camera: Tensor,
    pixels: Tensor,
    pixel_jacobian: Tensor,
    _config: UVTRenderConfig,
    *,
    min_depth: float = 1.0e-4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project world tubes from an arbitrary central-camera pixel Jacobian."""

    del _config
    _check_batch(batch)
    if world_to_camera.shape != (4, 4):
        raise ValueError("world_to_camera must have shape [4,4]")
    if world_to_camera.dtype != torch.float32:
        raise ValueError("world_to_camera must be float32")
    if world_to_camera.device != batch.x0.device:
        raise ValueError("world_to_camera must be on the same device as the batch")
    tube_count = int(batch.x0.shape[0])
    if pixels.shape != (tube_count, 2):
        raise ValueError("pixels must have shape [N,2]")
    if pixel_jacobian.shape != (tube_count, 2, 3):
        raise ValueError("pixel_jacobian must have shape [N,2,3]")

    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    center_cam = batch.x0 @ rotation.T + translation
    velocity_cam = batch.velocity @ rotation.T

    local_plane_to_camera = rotation[:, :2]
    screen_jacobian = pixel_jacobian @ local_plane_to_camera
    velocity_screen = (pixel_jacobian @ velocity_cam.unsqueeze(-1)).squeeze(-1)

    world_var_x = 1.0 / batch.precision_xy[:, 0].clamp_min(1.0e-6)
    world_var_y = 1.0 / batch.precision_xy[:, 1].clamp_min(1.0e-6)
    ju_x = screen_jacobian[:, 0, 0]
    ju_y = screen_jacobian[:, 0, 1]
    jv_x = screen_jacobian[:, 1, 0]
    jv_y = screen_jacobian[:, 1, 1]
    cov_uu = ju_x.square() * world_var_x + ju_y.square() * world_var_y + 1.0e-6
    cov_uv = ju_x * jv_x * world_var_x + ju_y * jv_y * world_var_y
    cov_vv = jv_x.square() * world_var_x + jv_y.square() * world_var_y + 1.0e-6
    inv_det = 1.0 / (cov_uu * cov_vv - cov_uv.square()).clamp_min(1.0e-12)

    lambda_u = cov_vv * inv_det
    lambda_uv = -cov_uv * inv_det
    lambda_v = cov_uu * inv_det
    velocity_u = velocity_screen[:, 0]
    velocity_v = velocity_screen[:, 1]
    q_uvt = torch.stack(
        (
            lambda_u,
            lambda_uv,
            -(lambda_u * velocity_u + lambda_uv * velocity_v),
            lambda_v,
            -(lambda_uv * velocity_u + lambda_v * velocity_v),
            batch.lambda_t
            + lambda_u * velocity_u.square()
            + 2.0 * lambda_uv * velocity_u * velocity_v
            + lambda_v * velocity_v.square(),
        ),
        dim=-1,
    )
    ma = torch.cat((pixels, batch.t0.unsqueeze(-1)), dim=-1)
    depth0 = center_cam[:, 2].clamp_min(min_depth)
    depth_beta = torch.zeros((tube_count, 3), dtype=torch.float32, device=batch.x0.device)
    depth_beta[:, 2] = velocity_cam[:, 2]
    return ma, q_uvt, depth0, depth_beta, batch.opacity, batch.color


def pinhole_from_camera_spec(camera_spec: object, device: torch.device | str | None = None) -> PinholeCamera:
    """Adapt Dynaworld's pinhole CameraSpec-like object to this harness."""

    lens_model = getattr(camera_spec, "lens_model", "pinhole")
    distortion = getattr(camera_spec, "distortion", None)
    if lens_model != "pinhole":
        raise ValueError(f"Only pinhole CameraSpec is supported here, got {lens_model!r}")
    if distortion not in (None, (), []):
        raise ValueError("Distorted CameraSpec is not supported by the STAR-UVT pinhole scaffold")

    c2w = getattr(camera_spec, "camera_to_world")
    if not torch.is_tensor(c2w):
        raise ValueError("camera_spec.camera_to_world must be a tensor")
    dev = c2w.device if device is None else torch.device(device)
    c2w = c2w.to(device=dev, dtype=torch.float32)
    world_to_camera = torch.linalg.inv(c2w)

    return PinholeCamera(
        fx=getattr(camera_spec, "fx"),
        fy=getattr(camera_spec, "fy"),
        cx=getattr(camera_spec, "cx"),
        cy=getattr(camera_spec, "cy"),
        world_to_camera=world_to_camera,
    )


def make_world_tube_demo(device: torch.device | str = "cpu") -> tuple[WorldTubeBatch, OrthoCamera2D, UVTRenderConfig]:
    dev = torch.device(device)
    batch = WorldTubeBatch(
        x0=torch.tensor([[0.0, 0.0, 1.0], [-1.2, 1.0, 1.4]], dtype=torch.float32, device=dev),
        velocity=torch.tensor([[0.45, -0.25, 0.05], [0.15, -0.35, -0.03]], dtype=torch.float32, device=dev),
        t0=torch.tensor([0.0, 0.0], dtype=torch.float32, device=dev),
        precision_xy=torch.tensor([[1.10, 1.00], [0.95, 1.25]], dtype=torch.float32, device=dev),
        lambda_t=torch.tensor([0.35, 0.45], dtype=torch.float32, device=dev),
        opacity=torch.tensor([0.65, 0.55], dtype=torch.float32, device=dev),
        color=torch.tensor([[0.9, 0.25, 0.15], [0.15, 0.45, 0.95]], dtype=torch.float32, device=dev),
    )
    camera = OrthoCamera2D(scale_u=2.0, scale_v=2.0, center_u=8.0, center_v=8.0)
    config = UVTRenderConfig(height=16, width=16, frames=4)
    return batch, camera, config


def make_pinhole_world_tube_demo(device: torch.device | str = "cpu") -> tuple[WorldTubeBatch, PinholeCamera, UVTRenderConfig]:
    dev = torch.device(device)
    batch = WorldTubeBatch(
        x0=torch.tensor([[-0.12, -0.08, 2.0], [0.16, 0.10, 2.3]], dtype=torch.float32, device=dev),
        velocity=torch.tensor([[0.03, 0.02, 0.04], [-0.02, 0.01, -0.03]], dtype=torch.float32, device=dev),
        t0=torch.tensor([0.0, 0.0], dtype=torch.float32, device=dev),
        precision_xy=torch.tensor([[80.0, 90.0], [70.0, 85.0]], dtype=torch.float32, device=dev),
        lambda_t=torch.tensor([0.30, 0.40], dtype=torch.float32, device=dev),
        opacity=torch.tensor([0.62, 0.50], dtype=torch.float32, device=dev),
        color=torch.tensor([[0.8, 0.3, 0.2], [0.2, 0.7, 0.9]], dtype=torch.float32, device=dev),
    )
    camera = PinholeCamera(
        fx=42.0,
        fy=42.0,
        cx=8.0,
        cy=8.0,
        world_to_camera=torch.eye(4, dtype=torch.float32, device=dev),
    )
    config = UVTRenderConfig(height=16, width=16, frames=4)
    return batch, camera, config
