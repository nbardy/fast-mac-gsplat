from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .world_tube import PinholeCamera, PinholeCameraMotion
except ImportError:  # pragma: no cover - direct script execution fallback.
    from world_tube import PinholeCamera, PinholeCameraMotion


def _inv_softplus(value: Tensor) -> Tensor:
    if bool((value <= 0.0).any().item()):
        raise ValueError("inverse-softplus input must be positive")
    return value + torch.log(-torch.expm1(-value))


def _logit(value: Tensor) -> Tensor:
    clamped = value.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(clamped) - torch.log1p(-clamped)


def _opacity_semantics(
    alpha_mode: str,
    amplitude_convention: str = "fiber_integrated",
) -> str:
    if alpha_mode == "peak_splat":
        return "peak_alpha_amplitude"
    if alpha_mode == "beer_lambert":
        if amplitude_convention == "fiber_integrated":
            return "nonnegative_fiber_integrated_peak_optical_thickness"
        if amplitude_convention == "peak_density":
            return "nonnegative_world_peak_extinction_density"
    raise ValueError("alpha_mode must be one of: peak_splat, beer_lambert")


def _initial_raw_opacity(
    init_source_amplitude: float,
    *,
    alpha_mode: str,
    amplitude_convention: str,
    count: int,
    reference: Tensor,
) -> Tensor:
    if amplitude_convention == "peak_density":
        if alpha_mode != "beer_lambert":
            raise ValueError("peak_density initialization requires beer_lambert")
        if not float(init_source_amplitude) > 0.0:
            raise ValueError("peak-density init_opacity must be positive")
        peak_density = torch.full(
            (count,),
            float(init_source_amplitude),
            dtype=reference.dtype,
            device=reference.device,
        )
        return _inv_softplus(peak_density)
    if not 0.0 < float(init_source_amplitude) < 0.99:
        raise ValueError("init_opacity must be an initial center alpha in (0, 0.99)")
    center_alpha = torch.full(
        (count,),
        float(init_source_amplitude),
        dtype=reference.dtype,
        device=reference.device,
    )
    if alpha_mode == "peak_splat":
        return _logit(center_alpha / 0.99)
    if alpha_mode == "beer_lambert":
        peak_optical_thickness = -torch.log1p(-center_alpha)
        return _inv_softplus(peak_optical_thickness)
    raise ValueError("alpha_mode must be one of: peak_splat, beer_lambert")


def _opacity_from_raw(raw_opacity: Tensor, *, alpha_mode: str) -> Tensor:
    if alpha_mode == "peak_splat":
        return torch.sigmoid(raw_opacity) * 0.99
    if alpha_mode == "beer_lambert":
        return F.softplus(raw_opacity)
    raise ValueError("alpha_mode must be one of: peak_splat, beer_lambert")


def _scalar_tensor(value: float | Tensor, reference: Tensor, name: str) -> Tensor:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape {tuple(value.shape)}")
        return value.to(device=reference.device, dtype=reference.dtype).reshape(())
    return reference.new_tensor(float(value))


def _eager_numeric_validation(reference: Tensor) -> bool:
    """Whether scalar value checks are cheap enough for a hot projection path.

    MPS does not currently implement an asynchronous assertion primitive.
    Calling ``.item()`` for every finiteness/SPD guard forces a full device
    synchronization, several times per projection. The trainer already
    performs synchronized non-finite-loss checks and restores its last finite
    state, so production MPS projections retain structural validation here
    while deferring value validation to that outer guard. CPU/reference calls
    keep the eager fail-loud checks.
    """

    return reference.device.type != "mps"


@dataclass(frozen=True)
class SPD4WorldAtomBatch:
    """Trainable native ``mean_xyzt + SPD(4)`` atoms.

    ``conditional_spatial_cholesky`` and ``space_time_tilt`` are retained in
    addition to the assembled covariance so callers can audit the chart.  The
    renderer only consumes the compiled UVT fields.
    """

    mean_xyzt: Tensor
    covariance_xyzt: Tensor
    conditional_spatial_cholesky: Tensor
    space_time_tilt: Tensor
    temporal_precision: Tensor
    opacity: Tensor
    color: Tensor
    amplitude_convention: str

    @property
    def x0(self) -> Tensor:
        return self.mean_xyzt[:, :3]

    @property
    def t0(self) -> Tensor:
        return self.mean_xyzt[:, 3]


@dataclass(frozen=True)
class SPD4ProjectedTubes:
    ma: Tensor
    q_uvt: Tensor
    depth0: Tensor
    depth_beta: Tensor
    depth_variance: Tensor
    peak_to_fiber_scale: Tensor | None
    opacity: Tensor
    color: Tensor


@dataclass(frozen=True)
class SPD4AffineGaugeBatch:
    """Atom-local affine maps from world ``(x,y,z,t)`` to ``(u,v,d,t)``."""

    gauge_from_world: Tensor
    gauge_offset: Tensor
    chart_time: float | None


class SPD4WorldAtomModel(nn.Module):
    """Full-SPD(4) alternative to the restricted legacy WorldTube model.

    The lossless block chart is

    ``Sigma = [[C + c vv.T, cv], [cv.T, c]]``,

    where ``C = L L.T`` is the spatial covariance conditioned on time,
    ``c`` is temporal variance, and ``v`` is the spacetime tilt.  Consequently
    ``E[x | t] = x0 + v (t - t0)`` without storing a separate velocity law.
    """

    representation_name = "full_spd4"
    geometry_dof_per_atom = 14
    total_dof_per_atom = 18

    def __init__(
        self,
        *,
        init_x0: Tensor,
        init_color: Tensor,
        init_t0: Tensor,
        frames: int,
        init_precision_xy: float,
        init_precision_z: float | None,
        init_lambda_t: float | Tensor,
        init_opacity: float,
        min_spatial_scale: float,
        min_lambda_t: float,
        tilt_reg_weight: float,
        depth_tilt_reg_weight: float,
        position_reg_weight: float,
        alpha_mode: str = "peak_splat",
        amplitude_convention: str = "fiber_integrated",
        static_tube_count: int = 0,
        static_tilt_reg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if init_x0.ndim != 2 or init_x0.shape[-1] != 3:
            raise ValueError("init_x0 must have shape [N,3]")
        tube_count = int(init_x0.shape[0])
        if init_color.shape != (tube_count, 3):
            raise ValueError("init_color must have shape [N,3]")
        if init_t0.shape != (tube_count,):
            raise ValueError("init_t0 must have shape [N]")
        if init_precision_xy <= 0.0:
            raise ValueError("init_precision_xy must be positive")
        resolved_precision_z = (
            float(init_precision_xy)
            if init_precision_z is None
            else float(init_precision_z)
        )
        if resolved_precision_z <= 0.0:
            raise ValueError("init_precision_z must be positive")
        if min_spatial_scale <= 0.0:
            raise ValueError("min_spatial_scale must be positive")
        if min_lambda_t <= 0.0:
            raise ValueError("min_lambda_t must be positive")

        self.tube_count = tube_count
        self.frames = int(frames)
        self.min_spatial_scale = float(min_spatial_scale)
        self.min_lambda_t = float(min_lambda_t)
        self.tilt_reg_weight = float(tilt_reg_weight)
        self.depth_tilt_reg_weight = float(depth_tilt_reg_weight)
        self.position_reg_weight = float(position_reg_weight)
        self.alpha_mode = str(alpha_mode)
        self.amplitude_convention = str(amplitude_convention)
        if self.amplitude_convention not in {"fiber_integrated", "peak_density"}:
            raise ValueError(
                "amplitude_convention must be one of: fiber_integrated, peak_density"
            )
        if self.alpha_mode == "peak_splat" and self.amplitude_convention != "fiber_integrated":
            raise ValueError(
                "peak_splat requires amplitude_convention=fiber_integrated"
            )
        self.opacity_semantics = _opacity_semantics(
            self.alpha_mode,
            self.amplitude_convention,
        )
        self.static_tube_count = int(static_tube_count)
        self.static_tilt_reg_weight = float(static_tilt_reg_weight)
        self.active_tube_count = tube_count
        if not 0 <= self.static_tube_count <= tube_count:
            raise ValueError("static_tube_count must be between 0 and tube_count")
        if min(
            self.tilt_reg_weight,
            self.depth_tilt_reg_weight,
            self.position_reg_weight,
            self.static_tilt_reg_weight,
        ) < 0.0:
            raise ValueError("regularization weights must be nonnegative")

        init_scale_xy = float(init_precision_xy) ** -0.5
        init_scale_z = resolved_precision_z**-0.5
        if min(init_scale_xy, init_scale_z) <= self.min_spatial_scale:
            raise ValueError("initial spatial scales must exceed min_spatial_scale")
        spatial_scale = torch.tensor(
            (init_scale_xy, init_scale_xy, init_scale_z),
            dtype=init_x0.dtype,
            device=init_x0.device,
        ).expand(tube_count, -1).clone()
        if isinstance(init_lambda_t, Tensor):
            if init_lambda_t.shape != (tube_count,):
                raise ValueError(f"init_lambda_t tensor must have shape ({tube_count},)")
            lambda_t = init_lambda_t.to(dtype=init_x0.dtype, device=init_x0.device)
        else:
            lambda_t = torch.full(
                (tube_count,),
                float(init_lambda_t),
                dtype=init_x0.dtype,
                device=init_x0.device,
            )
        if bool((lambda_t <= self.min_lambda_t).any().item()):
            raise ValueError("init_lambda_t values must exceed min_lambda_t")
        self.x0 = nn.Parameter(init_x0)
        self.t0 = nn.Parameter(init_t0)
        self.raw_spatial_scale = nn.Parameter(
            _inv_softplus(spatial_scale - self.min_spatial_scale)
        )
        self.spatial_cholesky_offdiag = nn.Parameter(
            torch.zeros((tube_count, 3), dtype=init_x0.dtype, device=init_x0.device)
        )
        self.space_time_tilt = nn.Parameter(torch.zeros_like(init_x0))
        self.raw_lambda_t = nn.Parameter(_inv_softplus(lambda_t - self.min_lambda_t))
        self.raw_opacity = nn.Parameter(
            _initial_raw_opacity(
                init_opacity,
                alpha_mode=self.alpha_mode,
                amplitude_convention=self.amplitude_convention,
                count=tube_count,
                reference=init_x0,
            )
        )
        self.raw_color = nn.Parameter(_logit(init_color))

    def set_active_tube_count(self, count: int) -> None:
        if not 1 <= int(count) <= self.tube_count:
            raise ValueError(
                f"active tube count must be in [1, {self.tube_count}], got {count}"
            )
        self.active_tube_count = int(count)

    def spatial_cholesky(self) -> Tensor:
        active = slice(0, self.active_tube_count)
        diagonal = (
            F.softplus(self.raw_spatial_scale[active]) + self.min_spatial_scale
        )
        offdiag = self.spatial_cholesky_offdiag[active]
        zero = torch.zeros_like(diagonal[:, 0])
        return torch.stack(
            (
                torch.stack((diagonal[:, 0], zero, zero), dim=-1),
                torch.stack((offdiag[:, 0], diagonal[:, 1], zero), dim=-1),
                torch.stack(
                    (offdiag[:, 1], offdiag[:, 2], diagonal[:, 2]), dim=-1
                ),
            ),
            dim=-2,
        )

    def batch(self) -> SPD4WorldAtomBatch:
        active = slice(0, self.active_tube_count)
        spatial_cholesky = self.spatial_cholesky()
        conditional_spatial = (
            spatial_cholesky @ spatial_cholesky.transpose(-1, -2)
        )
        temporal_precision = (
            F.softplus(self.raw_lambda_t[active]) + self.min_lambda_t
        )
        temporal_variance = temporal_precision.reciprocal()
        tilt = self.space_time_tilt[active]
        cross = temporal_variance[:, None] * tilt
        spatial_joint = conditional_spatial + (
            temporal_variance[:, None, None]
            * tilt[:, :, None]
            * tilt[:, None, :]
        )
        covariance = torch.cat(
            (
                torch.cat((spatial_joint, cross[:, :, None]), dim=-1),
                torch.cat((cross, temporal_variance[:, None]), dim=-1)[:, None, :],
            ),
            dim=-2,
        )
        return SPD4WorldAtomBatch(
            mean_xyzt=torch.cat(
                (self.x0[active], self.t0[active, None]), dim=-1
            ),
            covariance_xyzt=covariance,
            conditional_spatial_cholesky=spatial_cholesky,
            space_time_tilt=tilt,
            temporal_precision=temporal_precision,
            opacity=_opacity_from_raw(
                self.raw_opacity[active],
                alpha_mode=self.alpha_mode,
            ),
            color=torch.sigmoid(self.raw_color[active]),
            amplitude_convention=self.amplitude_convention,
        )

    def regularization(self) -> Tensor:
        reg = self.x0.new_tensor(0.0)
        active_tilt = self.space_time_tilt[: self.active_tube_count]
        if self.tilt_reg_weight:
            reg = reg + self.tilt_reg_weight * active_tilt.square().mean()
        if self.depth_tilt_reg_weight:
            reg = (
                reg
                + self.depth_tilt_reg_weight
                * active_tilt[:, 2].square().mean()
            )
        if self.position_reg_weight:
            reg = (
                reg
                + self.position_reg_weight
                * self.x0[: self.active_tube_count].square().mean()
            )
        if self.static_tilt_reg_weight and self.static_tube_count:
            static_start = self.tube_count - self.static_tube_count
            if self.active_tube_count > static_start:
                reg = (
                    reg
                    + self.static_tilt_reg_weight
                    * self.space_time_tilt[
                        static_start : self.active_tube_count
                    ].square().mean()
                )
        return reg

    def representation_metadata(self) -> dict[str, int | str]:
        return {
            "world_representation": self.representation_name,
            "geometry_dof_per_atom": self.geometry_dof_per_atom,
            "total_dof_per_atom": self.total_dof_per_atom,
            "motion_parameterization": "space_time_covariance_tilt",
            "spatial_covariance_parameterization": "conditional_cholesky_3x3",
            "alpha_mode": self.alpha_mode,
            "amplitude_convention": self.amplitude_convention,
            "opacity_semantics": self.opacity_semantics,
        }


def _check_batch(batch: SPD4WorldAtomBatch) -> None:
    count = int(batch.mean_xyzt.shape[0])
    expected = {
        "mean_xyzt": (count, 4),
        "covariance_xyzt": (count, 4, 4),
        "conditional_spatial_cholesky": (count, 3, 3),
        "space_time_tilt": (count, 3),
        "temporal_precision": (count,),
        "opacity": (count,),
        "color": (count, 3),
    }
    for name, shape in expected.items():
        value = getattr(batch, name)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if value.device != batch.mean_xyzt.device:
            raise ValueError(f"{name} must be on the same device as mean_xyzt")
    if batch.amplitude_convention not in {"fiber_integrated", "peak_density"}:
        raise ValueError(
            "amplitude_convention must be one of: fiber_integrated, peak_density"
        )


def _inverse_symmetric_3x3(matrix: Tensor, *, eps: float = 1.0e-12) -> Tensor:
    a = matrix[:, 0, 0]
    b = matrix[:, 0, 1]
    c = matrix[:, 0, 2]
    d = matrix[:, 1, 1]
    e = matrix[:, 1, 2]
    f = matrix[:, 2, 2]
    co00 = d * f - e.square()
    co01 = c * e - b * f
    co02 = b * e - c * d
    co11 = a * f - c.square()
    co12 = b * c - a * e
    co22 = a * d - b.square()
    determinant = a * co00 + b * co01 + c * co02
    determinant_scale = (a * d * f).abs().clamp_min(
        torch.finfo(matrix.dtype).tiny
    )
    determinant = torch.maximum(determinant, eps * determinant_scale)
    inverse = torch.stack(
        (
            torch.stack((co00, co01, co02), dim=-1),
            torch.stack((co01, co11, co12), dim=-1),
            torch.stack((co02, co12, co22), dim=-1),
        ),
        dim=-2,
    )
    return inverse / determinant[:, None, None]


def _pack_symmetric_3x3(matrix: Tensor) -> Tensor:
    return torch.stack(
        (
            matrix[:, 0, 0],
            matrix[:, 0, 1],
            matrix[:, 0, 2],
            matrix[:, 1, 1],
            matrix[:, 1, 2],
            matrix[:, 2, 2],
        ),
        dim=-1,
    )


def _check_affine_gauges(
    batch: SPD4WorldAtomBatch,
    gauges: SPD4AffineGaugeBatch,
) -> None:
    count = int(batch.mean_xyzt.shape[0])
    if gauges.gauge_from_world.shape != (count, 4, 4):
        raise ValueError("gauge_from_world must have shape [N,4,4]")
    if gauges.gauge_offset.shape != (count, 4):
        raise ValueError("gauge_offset must have shape [N,4]")
    for name, value in (
        ("gauge_from_world", gauges.gauge_from_world),
        ("gauge_offset", gauges.gauge_offset),
    ):
        if value.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if value.device != batch.mean_xyzt.device:
            raise ValueError(f"{name} must share the batch device")
        if _eager_numeric_validation(value) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError(f"{name} must be finite")
    if _eager_numeric_validation(gauges.gauge_from_world):
        determinant = torch.linalg.det(gauges.gauge_from_world)
        if bool((determinant == 0.0).any().item()):
            raise ValueError("each affine camera gauge must be invertible")


def project_spd4_world_atoms_affine_gauges(
    batch: SPD4WorldAtomBatch,
    gauges: SPD4AffineGaugeBatch,
    *,
    screen_variance_floor: float = 1.0e-6,
) -> SPD4ProjectedTubes:
    """Exactly push SPD(4) atoms through atom-local affine camera gauges."""

    _check_batch(batch)
    _check_affine_gauges(batch, gauges)
    if screen_variance_floor < 0.0:
        raise ValueError("screen_variance_floor must be nonnegative")
    count = int(batch.mean_xyzt.shape[0])
    pushed_mean = torch.einsum(
        "nij,nj->ni",
        gauges.gauge_from_world,
        batch.mean_xyzt,
    ) + gauges.gauge_offset
    pushed_covariance = (
        gauges.gauge_from_world
        @ batch.covariance_xyzt
        @ gauges.gauge_from_world.transpose(-1, -2)
    )
    pushed_covariance = 0.5 * (
        pushed_covariance + pushed_covariance.transpose(-1, -2)
    )
    a_index = torch.tensor((0, 1, 3), dtype=torch.long, device=batch.x0.device)
    marginal_covariance = (
        pushed_covariance.index_select(-2, a_index).index_select(-1, a_index)
    )
    floor = marginal_covariance.new_tensor(
        (screen_variance_floor, screen_variance_floor, 0.0)
    )
    marginal_covariance = marginal_covariance + torch.diag_embed(
        floor.expand(count, -1)
    )
    if _eager_numeric_validation(marginal_covariance):
        marginal_determinant = torch.linalg.det(marginal_covariance)
        marginal_determinant_scale = torch.diagonal(
            marginal_covariance,
            dim1=-2,
            dim2=-1,
        ).prod(dim=-1)
        relative_determinant = (
            marginal_determinant
            / marginal_determinant_scale.clamp_min(
                torch.finfo(torch.float32).tiny
            )
        )
        if bool((relative_determinant <= 1.0e-8).any().item()):
            raise ValueError(
                "affine gauge produced a singular or ill-conditioned UVT marginal"
            )
    marginal_precision = _inverse_symmetric_3x3(marginal_covariance)
    covariance_a_depth = pushed_covariance.index_select(-2, a_index)[:, :, 2]
    depth_beta = torch.einsum(
        "ni,nij->nj",
        covariance_a_depth,
        marginal_precision,
    )
    depth_variance = pushed_covariance[:, 2, 2] - (
        covariance_a_depth * depth_beta
    ).sum(dim=-1)
    if _eager_numeric_validation(depth_variance) and bool(
        (depth_variance <= 0.0).any().item()
    ):
        raise ValueError("affine gauge produced nonpositive conditional depth variance")
    peak_to_fiber_scale = None
    if batch.amplitude_convention == "peak_density":
        # Camera gauges preserve the physical-time row, so the world-space
        # direction of increasing depth at fixed (u,v,t) is the reciprocal
        # frame vector of the spatial 3x3 block. For rows r_u,r_v,r_d,
        #
        #   d x_world / d depth = (r_u x r_v) / <r_d, r_u x r_v>.
        #
        # This is exactly the spatial part of A^{-1} e_depth, without a
        # batched 4x4 inverse (a severe small-matrix MPS synchronization cost).
        spatial_rows = gauges.gauge_from_world[:, :3, :3]
        fiber_normal = torch.linalg.cross(
            spatial_rows[:, 0],
            spatial_rows[:, 1],
            dim=-1,
        )
        fiber_denominator = (
            spatial_rows[:, 2] * fiber_normal
        ).sum(dim=-1)
        fiber_measure_scale = torch.linalg.vector_norm(
            fiber_normal,
            dim=-1,
        ) / fiber_denominator.abs()
        if _eager_numeric_validation(fiber_measure_scale) and bool(
            (
                (~torch.isfinite(fiber_measure_scale))
                | (fiber_measure_scale <= 0.0)
            ).any().item()
        ):
            raise ValueError(
                "affine gauge produced a nonpositive fiber measure scale"
            )
        peak_to_fiber_scale = (
            fiber_measure_scale
            * torch.sqrt(
                depth_variance * depth_variance.new_tensor(2.0 * torch.pi)
            )
        ).contiguous()
    return SPD4ProjectedTubes(
        ma=pushed_mean.index_select(-1, a_index).contiguous(),
        q_uvt=_pack_symmetric_3x3(marginal_precision).contiguous(),
        depth0=pushed_mean[:, 2].contiguous(),
        depth_beta=depth_beta.contiguous(),
        depth_variance=depth_variance.contiguous(),
        peak_to_fiber_scale=peak_to_fiber_scale,
        opacity=batch.opacity.contiguous(),
        color=batch.color.contiguous(),
    )


def project_spd4_world_atoms_from_pixel_jacobian(
    batch: SPD4WorldAtomBatch,
    world_to_camera: Tensor,
    pixels: Tensor,
    pixel_jacobian: Tensor,
    *,
    min_depth: float = 1.0e-4,
    screen_variance_floor: float = 1.0e-6,
) -> SPD4ProjectedTubes:
    """Compile an atom-local affine camera gauge to the legacy STAR ABI."""

    _check_batch(batch)
    count = int(batch.mean_xyzt.shape[0])
    if world_to_camera.shape != (4, 4):
        raise ValueError("world_to_camera must have shape [4,4]")
    if world_to_camera.dtype != torch.float32:
        raise ValueError("world_to_camera must be float32")
    if world_to_camera.device != batch.mean_xyzt.device:
        raise ValueError("world_to_camera must share the batch device")
    if pixels.shape != (count, 2):
        raise ValueError("pixels must have shape [N,2]")
    if pixel_jacobian.shape != (count, 2, 3):
        raise ValueError("pixel_jacobian must have shape [N,2,3]")

    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    center_cam = batch.x0 @ rotation.T + translation
    depth = center_cam[:, 2].clamp_min(min_depth)
    screen_from_world = pixel_jacobian @ rotation.expand(count, -1, -1)
    zero = torch.zeros((count, 1), dtype=torch.float32, device=batch.x0.device)
    depth_row = torch.cat((rotation[2].expand(count, -1), zero), dim=-1)
    time_row = torch.cat(
        (
            torch.zeros((count, 3), dtype=torch.float32, device=batch.x0.device),
            torch.ones((count, 1), dtype=torch.float32, device=batch.x0.device),
        ),
        dim=-1,
    )
    gauge_linear = torch.stack(
        (
            torch.cat((screen_from_world[:, 0], zero), dim=-1),
            torch.cat((screen_from_world[:, 1], zero), dim=-1),
            depth_row,
            time_row,
        ),
        dim=-2,
    )
    target_uvdt = torch.cat(
        (
            pixels,
            depth[:, None],
            batch.t0[:, None],
        ),
        dim=-1,
    )
    gauge_offset = target_uvdt - torch.einsum(
        "nij,nj->ni",
        gauge_linear,
        batch.mean_xyzt,
    )
    return project_spd4_world_atoms_affine_gauges(
        batch,
        SPD4AffineGaugeBatch(
            gauge_from_world=gauge_linear,
            gauge_offset=gauge_offset,
            chart_time=None,
        ),
        screen_variance_floor=screen_variance_floor,
    )


def compile_spd4_pinhole_motion_affine_gauges(
    batch: SPD4WorldAtomBatch,
    camera: PinholeCameraMotion,
    *,
    min_depth: float = 1.0e-4,
) -> SPD4AffineGaugeBatch:
    """Linearize one smooth moving-camera program without time segmentation.

    The chart is the total first derivative of

    ``pi(K(t) W(t) [x,1])``

    at each atom's conditional spatial center at ``camera.chart_time``.
    Because the atom remains a joint SPD(4) volume, its space/time covariance
    supplies object motion; the camera derivative enters only through this
    affine gauge.
    """

    _check_batch(batch)
    if min_depth <= 0.0:
        raise ValueError("min_depth must be positive")
    if camera.world_to_camera.shape != (4, 4):
        raise ValueError("world_to_camera must have shape [4,4]")
    if camera.world_to_camera_dot.shape != (4, 4):
        raise ValueError("world_to_camera_dot must have shape [4,4]")
    for name, value in (
        ("world_to_camera", camera.world_to_camera),
        ("world_to_camera_dot", camera.world_to_camera_dot),
    ):
        if value.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if value.device != batch.mean_xyzt.device:
            raise ValueError(f"{name} must share the batch device")

    fx = _scalar_tensor(camera.fx, batch.x0, "fx")
    fy = _scalar_tensor(camera.fy, batch.x0, "fy")
    cx = _scalar_tensor(camera.cx, batch.x0, "cx")
    cy = _scalar_tensor(camera.cy, batch.x0, "cy")
    fx_dot = _scalar_tensor(camera.fx_dot, batch.x0, "fx_dot")
    fy_dot = _scalar_tensor(camera.fy_dot, batch.x0, "fy_dot")
    cx_dot = _scalar_tensor(camera.cx_dot, batch.x0, "cx_dot")
    cy_dot = _scalar_tensor(camera.cy_dot, batch.x0, "cy_dot")
    zero = batch.x0.new_zeros(())
    one = batch.x0.new_ones(())
    intrinsic = torch.stack(
        (
            torch.stack((fx, zero, cx)),
            torch.stack((zero, fy, cy)),
            torch.stack((zero, zero, one)),
        )
    )
    intrinsic_dot = torch.stack(
        (
            torch.stack((fx_dot, zero, cx_dot)),
            torch.stack((zero, fy_dot, cy_dot)),
            torch.stack((zero, zero, zero)),
        )
    )
    projection = intrinsic @ camera.world_to_camera[:3, :]
    projection_dot = (
        intrinsic_dot @ camera.world_to_camera[:3, :]
        + intrinsic @ camera.world_to_camera_dot[:3, :]
    )

    chart_time = batch.t0.new_tensor(float(camera.chart_time))
    conditional_center = batch.x0 + batch.space_time_tilt * (
        chart_time - batch.t0
    )[:, None]
    homogeneous_center = torch.cat(
        (
            conditional_center,
            torch.ones(
                (conditional_center.shape[0], 1),
                dtype=torch.float32,
                device=conditional_center.device,
            ),
        ),
        dim=-1,
    )
    homogeneous_image = homogeneous_center @ projection.T
    depth = homogeneous_image[:, 2]
    if _eager_numeric_validation(depth) and bool(
        (depth <= min_depth).any().item()
    ):
        raise ValueError(
            "moving-camera affine gauge requires every chart center in front of the camera"
        )
    inverse_depth = depth.reciprocal()
    pixels = homogeneous_image[:, :2] * inverse_depth[:, None]
    dehomogenization = torch.zeros(
        (conditional_center.shape[0], 2, 3),
        dtype=torch.float32,
        device=conditional_center.device,
    )
    dehomogenization[:, 0, 0] = inverse_depth
    dehomogenization[:, 0, 2] = (
        -homogeneous_image[:, 0] * inverse_depth.square()
    )
    dehomogenization[:, 1, 1] = inverse_depth
    dehomogenization[:, 1, 2] = (
        -homogeneous_image[:, 1] * inverse_depth.square()
    )
    screen_from_world = dehomogenization @ projection[:, :3]
    fixed_world_homogeneous_dot = homogeneous_center @ projection_dot.T
    screen_from_time = torch.einsum(
        "nij,nj->ni",
        dehomogenization,
        fixed_world_homogeneous_dot,
    )
    depth_from_time = (
        homogeneous_center @ camera.world_to_camera_dot[2, :]
    )
    zero_column = torch.zeros(
        (conditional_center.shape[0], 1),
        dtype=torch.float32,
        device=conditional_center.device,
    )
    time_row = torch.cat(
        (
            torch.zeros(
                (conditional_center.shape[0], 3),
                dtype=torch.float32,
                device=conditional_center.device,
            ),
            torch.ones_like(zero_column),
        ),
        dim=-1,
    )
    gauge_linear = torch.cat(
        (
            torch.cat((screen_from_world, screen_from_time[:, :, None]), dim=-1),
            torch.cat(
                (
                    camera.world_to_camera[2, :3].expand(
                        conditional_center.shape[0], -1
                    ),
                    depth_from_time[:, None],
                ),
                dim=-1,
            )[:, None, :],
            time_row[:, None, :],
        ),
        dim=-2,
    )
    chart_point = torch.cat(
        (
            conditional_center,
            chart_time.expand(conditional_center.shape[0], 1),
        ),
        dim=-1,
    )
    target_uvdt = torch.cat(
        (
            pixels,
            depth[:, None],
            chart_time.expand(conditional_center.shape[0], 1),
        ),
        dim=-1,
    )
    gauge_offset = target_uvdt - torch.einsum(
        "nij,nj->ni",
        gauge_linear,
        chart_point,
    )
    gauges = SPD4AffineGaugeBatch(
        gauge_from_world=gauge_linear,
        gauge_offset=gauge_offset,
        chart_time=float(camera.chart_time),
    )
    _check_affine_gauges(batch, gauges)
    return gauges


def project_spd4_world_atoms_pinhole_motion(
    batch: SPD4WorldAtomBatch,
    camera: PinholeCameraMotion,
    *,
    min_depth: float = 1.0e-4,
    screen_variance_floor: float = 1.0e-6,
) -> SPD4ProjectedTubes:
    """Compile a smooth first-order moving pinhole camera to one SPD(4) trace."""

    return project_spd4_world_atoms_affine_gauges(
        batch,
        compile_spd4_pinhole_motion_affine_gauges(
            batch,
            camera,
            min_depth=min_depth,
        ),
        screen_variance_floor=screen_variance_floor,
    )


def project_spd4_world_atoms_pinhole(
    batch: SPD4WorldAtomBatch,
    camera: PinholeCamera,
    *,
    min_depth: float = 1.0e-4,
) -> SPD4ProjectedTubes:
    _check_batch(batch)
    if camera.world_to_camera.shape != (4, 4):
        raise ValueError("world_to_camera must have shape [4,4]")
    if camera.world_to_camera.dtype != torch.float32:
        raise ValueError("world_to_camera must be float32")
    if camera.world_to_camera.device != batch.mean_xyzt.device:
        raise ValueError("world_to_camera must share the batch device")
    rotation = camera.world_to_camera[:3, :3]
    translation = camera.world_to_camera[:3, 3]
    center_cam = batch.x0 @ rotation.T + translation
    z = center_cam[:, 2].clamp_min(min_depth)
    fx = _scalar_tensor(camera.fx, batch.x0, "fx")
    fy = _scalar_tensor(camera.fy, batch.x0, "fy")
    cx = _scalar_tensor(camera.cx, batch.x0, "cx")
    cy = _scalar_tensor(camera.cy, batch.x0, "cy")
    pixels = torch.stack(
        (
            fx * center_cam[:, 0] / z + cx,
            fy * center_cam[:, 1] / z + cy,
        ),
        dim=-1,
    )
    inv_z = z.reciprocal()
    pixel_jacobian = torch.zeros(
        (int(batch.x0.shape[0]), 2, 3),
        dtype=torch.float32,
        device=batch.x0.device,
    )
    pixel_jacobian[:, 0, 0] = fx * inv_z
    pixel_jacobian[:, 0, 2] = -fx * center_cam[:, 0] * inv_z.square()
    pixel_jacobian[:, 1, 1] = fy * inv_z
    pixel_jacobian[:, 1, 2] = -fy * center_cam[:, 1] * inv_z.square()
    return project_spd4_world_atoms_from_pixel_jacobian(
        batch,
        camera.world_to_camera,
        pixels,
        pixel_jacobian,
        min_depth=min_depth,
    )


__all__ = [
    "SPD4AffineGaugeBatch",
    "SPD4ProjectedTubes",
    "SPD4WorldAtomBatch",
    "SPD4WorldAtomModel",
    "compile_spd4_pinhole_motion_affine_gauges",
    "project_spd4_world_atoms_affine_gauges",
    "project_spd4_world_atoms_from_pixel_jacobian",
    "project_spd4_world_atoms_pinhole",
    "project_spd4_world_atoms_pinhole_motion",
]
