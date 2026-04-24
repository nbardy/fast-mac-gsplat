from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

try:
    from . import _C  # noqa: F401
except Exception:
    _C = None


@dataclass(frozen=True)
class RuntimeShaderConfig:
    tile_size: int
    threads: int
    chunk_size: int
    fast_cap: int
    simdgroups: int


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


def get_runtime_shader_config() -> RuntimeShaderConfig:
    tile_size = _env_int("GSP_TILE_SIZE", 16)
    chunk_size = _env_int("GSP_CHUNK", 64)
    fast_cap = _env_int("GSP_FAST_CAP", 2048)
    if tile_size not in (8, 16, 32):
        raise ValueError(f"GSP_TILE_SIZE must be one of 8, 16, 32; got {tile_size}")
    threads = tile_size * tile_size
    if threads > 1024:
        raise ValueError(f"tile_size={tile_size} implies {threads} threads, which exceeds 1024")
    if chunk_size <= 0:
        raise ValueError("GSP_CHUNK must be positive")
    if fast_cap <= 0:
        raise ValueError("GSP_FAST_CAP must be positive")
    simdgroups = (threads + 31) // 32
    return RuntimeShaderConfig(
        tile_size=tile_size,
        threads=threads,
        chunk_size=chunk_size,
        fast_cap=fast_cap,
        simdgroups=simdgroups,
    )


@dataclass(frozen=True)
class RasterConfig:
    height: int
    width: int
    tile_size: int = 16
    max_fast_pairs: int = 2048
    alpha_threshold: float = 1.0 / 255.0
    transmittance_threshold: float = 1e-4
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    enable_overflow_fallback: bool = True
    batch_strategy: str = "auto"  # auto | flatten | serial
    batch_launch_limit_tiles: int = 262144
    batch_launch_limit_gaussians: int = 262144


def _runtime_validate(config: RasterConfig) -> RuntimeShaderConfig:
    rt = get_runtime_shader_config()
    if config.tile_size != rt.tile_size:
        raise ValueError(
            f"RasterConfig.tile_size={config.tile_size} does not match runtime shader tile size {rt.tile_size}. "
            "Set GSP_TILE_SIZE before importing/running the extension, or adjust RasterConfig."
        )
    if config.max_fast_pairs > rt.fast_cap:
        raise ValueError(
            f"RasterConfig.max_fast_pairs={config.max_fast_pairs} exceeds compiled fast cap {rt.fast_cap}. "
            "Lower the runtime cap or set GSP_FAST_CAP before import."
        )
    if config.batch_strategy not in ("auto", "flatten", "serial"):
        raise ValueError("batch_strategy must be one of: auto, flatten, serial")
    return rt


def _make_meta(config: RasterConfig, device: torch.device, batch_size: int, gaussians_per_batch: int) -> tuple[Tensor, Tensor]:
    _runtime_validate(config)
    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    tiles_per_image = tiles_y * tiles_x
    total_tiles = batch_size * tiles_per_image
    total_gaussians = batch_size * gaussians_per_batch
    meta_i32 = torch.tensor(
        [
            config.height,
            config.width,
            tiles_y,
            tiles_x,
            config.tile_size,
            total_gaussians,
            total_tiles,
            config.max_fast_pairs,
            batch_size,
            gaussians_per_batch,
            tiles_per_image,
            0,
        ],
        device=device,
        dtype=torch.int32,
    )
    meta_f32 = torch.tensor(
        [
            float(config.alpha_threshold),
            float(config.transmittance_threshold),
            float(config.background[0]),
            float(config.background[1]),
            float(config.background[2]),
            1e-8,
            0.99,
        ],
        device=device,
        dtype=torch.float32,
    )
    return meta_i32, meta_f32


def _normalize_inputs(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, bool]:
    if means2d.ndim == 2:
        return (
            means2d.unsqueeze(0),
            conics.unsqueeze(0),
            colors.unsqueeze(0),
            opacities.unsqueeze(0),
            depths.unsqueeze(0),
            False,
        )
    if means2d.ndim != 3:
        raise ValueError("means2d must have shape [G,2] or [B,G,2]")
    return means2d, conics, colors, opacities, depths, True


def _check_inputs(means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor) -> None:
    tensors = {
        "means2d": means2d,
        "conics": conics,
        "colors": colors,
        "opacities": opacities,
        "depths": depths,
    }
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError("means2d/conics/colors/opacities/depths must be on the same device")
    if means2d.device.type != "mps":
        raise ValueError("v8_project3d Metal rasterizer inputs must be on MPS")
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
    if means2d.ndim not in (2, 3):
        raise ValueError("means2d must have shape [G,2] or [B,G,2]")
    if conics.ndim != means2d.ndim or colors.ndim != means2d.ndim:
        raise ValueError("conics/colors rank must match means2d rank")
    if opacities.ndim != means2d.ndim - 1 or depths.ndim != means2d.ndim - 1:
        raise ValueError("opacities/depths rank must be one less than means2d rank")
    if means2d.shape[-1] != 2:
        raise ValueError("means2d must have last dim = 2")
    if conics.shape[-1] != 3:
        raise ValueError("conics must have last dim = 3")
    if colors.shape[-1] != 3:
        raise ValueError("colors must have last dim = 3")
    if means2d.shape[:-1] != conics.shape[:-1] or means2d.shape[:-1] != colors.shape[:-1]:
        raise ValueError("means2d/conics/colors batch/G dimensions must match")
    if means2d.shape[:-1] != opacities.shape or means2d.shape[:-1] != depths.shape:
        raise ValueError("means2d/opacities/depths batch/G dimensions must match")


def _normalize_3d_inputs(
    means3d: Tensor,
    scales: Tensor,
    quats: Tensor,
    opacities: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
    if opacities.ndim >= 1 and opacities.shape[-1:] == (1,):
        opacities = opacities.squeeze(-1)
    if means3d.ndim == 2:
        return (
            means3d.unsqueeze(0),
            scales.unsqueeze(0),
            quats.unsqueeze(0),
            opacities.unsqueeze(0),
            False,
        )
    if means3d.ndim != 3:
        raise ValueError("means3d must have shape [G,3] or [B,G,3]")
    return means3d, scales, quats, opacities, True


def _normalize_colors(colors: Tensor, batch_size: int, gaussians: int, was_batched: bool) -> Tensor:
    if colors.ndim == 2:
        colors_b = colors.unsqueeze(0)
    elif colors.ndim == 3:
        colors_b = colors
    else:
        raise ValueError("colors must have shape [G,3] or [B,G,3]")
    if colors_b.shape != (batch_size, gaussians, 3):
        expected = "[B,G,3]" if was_batched else "[G,3]"
        raise ValueError(f"colors must match Gaussian shape {expected}; got {tuple(colors.shape)}")
    return colors_b


def _check_3d_inputs(means3d: Tensor, scales: Tensor, quats: Tensor, opacities: Tensor) -> None:
    tensors = {
        "means3d": means3d,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
    }
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError("means3d/scales/quats/opacities must be on the same device")
    if means3d.device.type != "mps":
        raise ValueError("v8_project3d Metal projection inputs must be on MPS")
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
    if means3d.ndim not in (2, 3) or means3d.shape[-1] != 3:
        raise ValueError("means3d must have shape [G,3] or [B,G,3]")
    if scales.shape != means3d.shape:
        raise ValueError("scales must match means3d shape")
    if quats.shape[:-1] != means3d.shape[:-1] or quats.shape[-1] != 4:
        raise ValueError("quats must have shape [G,4] or [B,G,4]")
    op_shape = opacities.shape[:-1] if opacities.ndim >= 1 and opacities.shape[-1:] == (1,) else opacities.shape
    if op_shape != means3d.shape[:-1]:
        raise ValueError("opacities must have shape [G], [G,1], [B,G], or [B,G,1]")


def _camera_scalar_batch(value, batch_size: int, device: torch.device, dtype: torch.dtype, name: str) -> Tensor:
    if torch.is_tensor(value):
        value = value.to(device=device, dtype=dtype)
        if value.ndim == 0:
            return value.expand(batch_size)
        if value.numel() == batch_size:
            return value.reshape(batch_size)
        raise ValueError(f"{name} must be scalar or have {batch_size} values")
    return torch.full((batch_size,), float(value), device=device, dtype=dtype)


def _camera_to_world_batch(camera_to_world, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if camera_to_world is None:
        return torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    camera_to_world = camera_to_world.to(device=device, dtype=dtype)
    if camera_to_world.shape == (4, 4):
        return camera_to_world.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    if camera_to_world.shape == (batch_size, 4, 4):
        return camera_to_world.contiguous()
    raise ValueError(f"camera_to_world must be [4,4] or [{batch_size},4,4]")


def _make_project_meta(device: torch.device, batch_size: int, gaussians_per_batch: int, near_plane: float) -> tuple[Tensor, Tensor]:
    meta_i32 = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            int(batch_size * gaussians_per_batch),
            0,
            0,
            int(batch_size),
            int(gaussians_per_batch),
            0,
            0,
        ],
        device=device,
        dtype=torch.int32,
    )
    project_f32 = torch.tensor([float(near_plane)], device=device, dtype=torch.float32)
    return meta_i32, project_f32


def _torch_project_pinhole_gaussians_batched(
    means3d_b: Tensor,
    scales_b: Tensor,
    quats_b: Tensor,
    opacities_b: Tensor,
    camera_to_world_b: Tensor,
    camera_params: Tensor,
    near_plane: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch_size, gaussian_count = means3d_b.shape[:2]
    qr, qi, qj, qk = quats_b[..., 0], quats_b[..., 1], quats_b[..., 2], quats_b[..., 3]
    rotation = means3d_b.new_zeros((batch_size, gaussian_count, 3, 3))
    rotation[..., 0, 0] = 1.0 - 2.0 * (qj.square() + qk.square())
    rotation[..., 0, 1] = 2.0 * (qi * qj - qr * qk)
    rotation[..., 0, 2] = 2.0 * (qi * qk + qr * qj)
    rotation[..., 1, 0] = 2.0 * (qi * qj + qr * qk)
    rotation[..., 1, 1] = 1.0 - 2.0 * (qi.square() + qk.square())
    rotation[..., 1, 2] = 2.0 * (qj * qk - qr * qi)
    rotation[..., 2, 0] = 2.0 * (qi * qk - qr * qj)
    rotation[..., 2, 1] = 2.0 * (qj * qk + qr * qi)
    rotation[..., 2, 2] = 1.0 - 2.0 * (qi.square() + qj.square())

    scale_matrix = means3d_b.new_zeros((batch_size, gaussian_count, 3, 3))
    scale_matrix[..., 0, 0] = scales_b[..., 0]
    scale_matrix[..., 1, 1] = scales_b[..., 1]
    scale_matrix[..., 2, 2] = scales_b[..., 2]
    gaussian_basis = rotation @ scale_matrix
    cov3d = gaussian_basis @ gaussian_basis.transpose(-1, -2)

    rotation_cw = camera_to_world_b[:, :3, :3]
    translation = camera_to_world_b[:, :3, 3]
    means_camera = (means3d_b - translation[:, None, :]) @ rotation_cw
    cov_camera = rotation_cw.transpose(1, 2).unsqueeze(1) @ cov3d @ rotation_cw.unsqueeze(1)

    x = means_camera[..., 0]
    y = means_camera[..., 1]
    z = means_camera[..., 2]
    front = z > float(near_plane)
    z_safe = torch.where(front, torch.clamp(z, min=float(near_plane)), torch.ones_like(z))
    x_project = torch.where(front, x, torch.zeros_like(x))
    y_project = torch.where(front, y, torch.zeros_like(y))

    fx = camera_params[:, 0:1]
    fy = camera_params[:, 1:2]
    cx = camera_params[:, 2:3]
    cy = camera_params[:, 3:4]

    jacobian = means3d_b.new_zeros((batch_size, gaussian_count, 2, 3))
    jacobian[..., 0, 0] = fx / z_safe
    jacobian[..., 0, 2] = -(fx * x_project) / z_safe.square()
    jacobian[..., 1, 1] = fy / z_safe
    jacobian[..., 1, 2] = -(fy * y_project) / z_safe.square()

    cov2d = jacobian @ cov_camera @ jacobian.transpose(-1, -2)
    cov2d[..., 0, 0] += 0.3
    cov2d[..., 1, 1] += 0.3
    cov00 = cov2d[..., 0, 0]
    cov01 = cov2d[..., 0, 1]
    cov11 = cov2d[..., 1, 1]
    det = torch.clamp(cov00 * cov11 - cov01 * cov01, min=1.0e-6)
    conics = torch.stack([cov11 / det, -cov01 / det, cov00 / det], dim=-1)

    means2d = torch.stack(
        [
            (fx * x_project) / z_safe + cx,
            (fy * y_project) / z_safe + cy,
        ],
        dim=-1,
    )
    projected_opacities = torch.where(front, opacities_b, torch.zeros_like(opacities_b))
    return means2d, conics, projected_opacities, z


def _projection_backward_requested(*values) -> bool:
    if not torch.is_grad_enabled():
        return False
    return any(torch.is_tensor(value) and value.requires_grad for value in values)


def _zero_like_if_none(grad: Tensor | None, value: Tensor) -> Tensor:
    return torch.zeros_like(value) if grad is None else grad


def _grad_or_zero(grad: Tensor | None, value: Tensor) -> Tensor | None:
    return torch.zeros_like(value) if grad is None else grad


class _ProjectPinholeGaussiansV8Project3D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3d_b: Tensor,
        scales_b: Tensor,
        quats_b: Tensor,
        opacities_b: Tensor,
        camera_to_world_b: Tensor,
        camera_params: Tensor,
        meta_i32: Tensor,
        project_f32: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        means2d_f, conics_f, opacities_f, depths_f = torch.ops.gsplat_metal_v8_project3d.project_pinhole_forward(
            means3d_b.reshape(-1, 3).contiguous(),
            scales_b.reshape(-1, 3).contiguous(),
            quats_b.reshape(-1, 4).contiguous(),
            opacities_b.reshape(-1).contiguous(),
            camera_to_world_b,
            camera_params,
            meta_i32,
            project_f32,
        )
        ctx.save_for_backward(
            means3d_b,
            scales_b,
            quats_b,
            opacities_b,
            camera_to_world_b,
            camera_params,
            project_f32,
        )
        batch_size, gaussians_per_batch = means3d_b.shape[:2]
        return (
            means2d_f.view(batch_size, gaussians_per_batch, 2),
            conics_f.view(batch_size, gaussians_per_batch, 3),
            opacities_f.view(batch_size, gaussians_per_batch),
            depths_f.view(batch_size, gaussians_per_batch),
        )

    @staticmethod
    def backward(
        ctx,
        grad_means2d: Tensor | None,
        grad_conics: Tensor | None,
        grad_opacities: Tensor | None,
        grad_depths: Tensor | None,
    ):
        (
            means3d_b,
            scales_b,
            quats_b,
            opacities_b,
            camera_to_world_b,
            camera_params,
            project_f32,
        ) = ctx.saved_tensors
        near_plane = float(project_f32[0].detach().item())
        needs = ctx.needs_input_grad

        detached_inputs = [
            means3d_b.detach().requires_grad_(needs[0]),
            scales_b.detach().requires_grad_(needs[1]),
            quats_b.detach().requires_grad_(needs[2]),
            opacities_b.detach().requires_grad_(needs[3]),
            camera_to_world_b.detach().requires_grad_(needs[4]),
            camera_params.detach().requires_grad_(needs[5]),
        ]
        active_inputs = [value for value, need in zip(detached_inputs, needs[:6]) if need]
        if not active_inputs:
            return None, None, None, None, None, None, None, None

        with torch.enable_grad():
            outputs = _torch_project_pinhole_gaussians_batched(
                detached_inputs[0],
                detached_inputs[1],
                detached_inputs[2],
                detached_inputs[3],
                detached_inputs[4],
                detached_inputs[5],
                near_plane=near_plane,
            )
            grad_outputs = (
                _zero_like_if_none(grad_means2d, outputs[0]),
                _zero_like_if_none(grad_conics, outputs[1]),
                _zero_like_if_none(grad_opacities, outputs[2]),
                _zero_like_if_none(grad_depths, outputs[3]),
            )
            active_grads = torch.autograd.grad(
                outputs,
                active_inputs,
                grad_outputs=grad_outputs,
                allow_unused=True,
            )

        full_grads: list[Tensor | None] = [None] * 6
        grad_iter = iter(active_grads)
        for index, need in enumerate(needs[:6]):
            if need:
                full_grads[index] = _grad_or_zero(next(grad_iter), detached_inputs[index])
        return (*full_grads, None, None)


def _project_pinhole_gaussians_batched(
    means3d_b: Tensor,
    scales_b: Tensor,
    quats_b: Tensor,
    opacities_b: Tensor,
    fx,
    fy,
    cx,
    cy,
    camera_to_world,
    near_plane: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not hasattr(torch.ops.gsplat_metal_v8_project3d, "project_pinhole_forward"):
        raise RuntimeError("gsplat_metal_v8_project3d custom ops not found. Build the extension first.")

    batch_size, gaussians_per_batch = means3d_b.shape[:2]
    device = means3d_b.device
    dtype = means3d_b.dtype
    camera_params = torch.stack(
        [
            _camera_scalar_batch(fx, batch_size, device, dtype, "fx"),
            _camera_scalar_batch(fy, batch_size, device, dtype, "fy"),
            _camera_scalar_batch(cx, batch_size, device, dtype, "cx"),
            _camera_scalar_batch(cy, batch_size, device, dtype, "cy"),
        ],
        dim=1,
    ).contiguous()
    camera_to_world_b = _camera_to_world_batch(camera_to_world, batch_size, device, dtype)
    meta_i32, project_f32 = _make_project_meta(device, batch_size, gaussians_per_batch, near_plane)

    if _projection_backward_requested(means3d_b, scales_b, quats_b, opacities_b, camera_to_world_b, camera_params):
        return _ProjectPinholeGaussiansV8Project3D.apply(
            means3d_b,
            scales_b,
            quats_b,
            opacities_b,
            camera_to_world_b,
            camera_params,
            meta_i32,
            project_f32,
        )

    means2d_f, conics_f, opacities_f, depths_f = torch.ops.gsplat_metal_v8_project3d.project_pinhole_forward(
        means3d_b.reshape(-1, 3).contiguous(),
        scales_b.reshape(-1, 3).contiguous(),
        quats_b.reshape(-1, 4).contiguous(),
        opacities_b.reshape(-1).contiguous(),
        camera_to_world_b,
        camera_params,
        meta_i32,
        project_f32,
    )
    return (
        means2d_f.view(batch_size, gaussians_per_batch, 2),
        conics_f.view(batch_size, gaussians_per_batch, 3),
        opacities_f.view(batch_size, gaussians_per_batch),
        depths_f.view(batch_size, gaussians_per_batch),
    )


def project_pinhole_gaussians(
    means3d: Tensor,
    scales: Tensor,
    quats: Tensor,
    opacities: Tensor,
    fx,
    fy,
    cx,
    cy,
    camera_to_world=None,
    near_plane: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _check_3d_inputs(means3d, scales, quats, opacities)
    means3d_b, scales_b, quats_b, opacities_b, was_batched = _normalize_3d_inputs(means3d, scales, quats, opacities)
    means2d, conics, projected_opacities, depths = _project_pinhole_gaussians_batched(
        means3d_b.contiguous(),
        scales_b.contiguous(),
        quats_b.contiguous(),
        opacities_b.contiguous(),
        fx,
        fy,
        cx,
        cy,
        camera_to_world,
        near_plane,
    )
    if was_batched:
        return means2d, conics, projected_opacities, depths
    return means2d[0], conics[0], projected_opacities[0], depths[0]


def _batched_gather_2d(x: Tensor, perm: Tensor) -> Tensor:
    return x.gather(1, perm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def _batched_gather_1d(x: Tensor, perm: Tensor) -> Tensor:
    return x.gather(1, perm)


def _unsort_batched(grad: Tensor, perm: Tensor) -> Tensor:
    out = torch.empty_like(grad)
    if grad.ndim == 3:
        out.scatter_(1, perm.unsqueeze(-1).expand_as(grad), grad)
    elif grad.ndim == 2:
        out.scatter_(1, perm, grad)
    else:
        raise ValueError(f"unexpected grad rank: {grad.ndim}")
    return out


def _gather_overflow_segments(
    tile_counts: Tensor,
    tile_offsets: Tensor,
    binned_ids: Tensor,
    max_fast_pairs: int,
) -> tuple[Tensor, Tensor, Tensor]:
    overflow_tile_ids = torch.nonzero(tile_counts > int(max_fast_pairs), as_tuple=False).flatten()
    if overflow_tile_ids.numel() == 0:
        device = tile_counts.device
        empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
        return empty_i32, torch.zeros((1,), device=device, dtype=torch.int32), empty_i32

    segments: list[Tensor] = []
    counts: list[int] = []
    for tile_id in overflow_tile_ids.tolist():
        start = int(tile_offsets[tile_id].item())
        end = int(tile_offsets[tile_id + 1].item())
        ids_t = binned_ids[start:end]
        if ids_t.numel() == 0:
            counts.append(0)
            continue
        perm = torch.argsort(ids_t, dim=0, stable=True)
        segments.append(ids_t.index_select(0, perm))
        counts.append(end - start)

    overflow_sorted_ids = (
        torch.cat(segments, dim=0).contiguous()
        if segments
        else torch.empty((0,), device=binned_ids.device, dtype=torch.int32)
    )
    ov_counts = torch.tensor(counts, device=tile_counts.device, dtype=torch.int32)
    ov_offsets = torch.cat(
        [torch.zeros((1,), device=tile_counts.device, dtype=torch.int32), torch.cumsum(ov_counts, dim=0, dtype=torch.int32)],
        dim=0,
    ).contiguous()
    return overflow_tile_ids.to(torch.int32).contiguous(), ov_offsets, overflow_sorted_ids.to(torch.int32).contiguous()


def _tile_origin_global(tile_id: int, tiles_per_image: int, tiles_x: int, tile_size: int) -> tuple[int, int, int]:
    batch = tile_id // tiles_per_image
    local_tile = tile_id % tiles_per_image
    tx = local_tile % tiles_x
    ty = local_tile // tiles_x
    return batch, tx * tile_size, ty * tile_size


def _scatter_tile_images_(base: Tensor, tile_ids: Tensor, tile_imgs: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> None:
    if tile_ids.numel() == 0:
        return
    _, H, W = base.shape[:3]
    for i, tile_id in enumerate(tile_ids.tolist()):
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        base[b, y0:y1, x0:x1, :] = tile_imgs[i, : y1 - y0, : x1 - x0, :]


def _gather_tile_images(img: Tensor, tile_ids: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> Tensor:
    if tile_ids.numel() == 0:
        return torch.empty((0, tile_size, tile_size, img.shape[-1]), device=img.device, dtype=img.dtype)
    out = torch.zeros((tile_ids.numel(), tile_size, tile_size, img.shape[-1]), device=img.device, dtype=img.dtype)
    _, H, W = img.shape[:3]
    for i, tile_id in enumerate(tile_ids.tolist()):
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        out[i, : y1 - y0, : x1 - x0, :] = img[b, y0:y1, x0:x1, :]
    return out


def _zero_tile_images_(img: Tensor, tile_ids: Tensor, tiles_per_image: int, tiles_x: int, tile_size: int) -> None:
    if tile_ids.numel() == 0:
        return
    _, H, W = img.shape[:3]
    for tile_id in tile_ids.tolist():
        b, x0, y0 = _tile_origin_global(int(tile_id), tiles_per_image, tiles_x, tile_size)
        x1 = min(x0 + tile_size, W)
        y1 = min(y0 + tile_size, H)
        img[b, y0:y1, x0:x1, :] = 0


def _should_use_training_path(means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor) -> bool:
    if not torch.is_grad_enabled():
        return False
    return bool(means2d.requires_grad or conics.requires_grad or colors.requires_grad or opacities.requires_grad)


def _choose_batch_chunk_size(config: RasterConfig, batch_size: int, gaussians_per_batch: int, tiles_per_image: int) -> int:
    if config.batch_strategy == "flatten":
        return batch_size
    if config.batch_strategy == "serial":
        return 1
    by_tiles = max(1, config.batch_launch_limit_tiles // max(tiles_per_image, 1))
    by_gaussians = max(1, config.batch_launch_limit_gaussians // max(gaussians_per_batch, 1))
    return max(1, min(batch_size, by_tiles, by_gaussians))


class _RasterizeProjectedGaussiansV8Project3D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means2d_b: Tensor,
        conics_b: Tensor,
        colors_b: Tensor,
        opacities_b: Tensor,
        depths_b: Tensor,
        meta_i32: Tensor,
        meta_f32: Tensor,
        enable_overflow_fallback: bool,
    ) -> Tensor:
        if not hasattr(torch.ops, "gsplat_metal_v8_project3d"):
            raise RuntimeError("gsplat_metal_v8_project3d custom ops not found. Build the extension first.")

        B, G = means2d_b.shape[:2]
        perm = torch.argsort(depths_b.detach(), dim=1, stable=True)
        means2d_s = _batched_gather_2d(means2d_b, perm).contiguous()
        conics_s = _batched_gather_2d(conics_b, perm).contiguous()
        colors_s = _batched_gather_2d(colors_b, perm).contiguous()
        opacities_s = _batched_gather_1d(opacities_b, perm).contiguous()

        means_flat = means2d_s.reshape(B * G, 2).contiguous()
        conics_flat = conics_s.reshape(B * G, 3).contiguous()
        colors_flat = colors_s.reshape(B * G, 3).contiguous()
        opacities_flat = opacities_s.reshape(B * G).contiguous()

        tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v8_project3d.bin(
            means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32
        )
        out_fast, tile_stop_counts = torch.ops.gsplat_metal_v8_project3d.render_fast_forward_state(
            means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32, binned_ids, tile_counts, tile_offsets
        )

        tile_size = int(meta_i32[4].item())
        tiles_x = int(meta_i32[3].item())
        tiles_per_image = int(meta_i32[10].item())
        max_fast_pairs = int(meta_i32[7].item())

        overflow_tile_ids = torch.empty((0,), device=means2d_b.device, dtype=torch.int32)
        overflow_tile_offsets = torch.zeros((1,), device=means2d_b.device, dtype=torch.int32)
        overflow_sorted_ids = torch.empty((0,), device=means2d_b.device, dtype=torch.int32)

        if enable_overflow_fallback:
            overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids = _gather_overflow_segments(
                tile_counts, tile_offsets, binned_ids, max_fast_pairs
            )
            if overflow_tile_ids.numel() > 0:
                overflow_tile_imgs = torch.ops.gsplat_metal_v8_project3d.render_overflow_forward(
                    means_flat,
                    conics_flat,
                    colors_flat,
                    opacities_flat,
                    meta_i32,
                    meta_f32,
                    overflow_tile_ids,
                    overflow_tile_offsets,
                    overflow_sorted_ids,
                )
                out = out_fast.clone()
                _scatter_tile_images_(out, overflow_tile_ids, overflow_tile_imgs, tiles_per_image, tiles_x, tile_size)
            else:
                out = out_fast
        else:
            if bool((tile_counts > max_fast_pairs).any().item()):
                raise RuntimeError(
                    f"Tile overflow detected with max_fast_pairs={max_fast_pairs}. "
                    "Enable overflow fallback or increase the runtime cap."
                )
            out = out_fast

        ctx.save_for_backward(
            perm,
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            depths_b,
            meta_i32,
            meta_f32,
            tile_counts,
            tile_offsets,
            binned_ids,
            tile_stop_counts,
            overflow_tile_ids,
            overflow_tile_offsets,
            overflow_sorted_ids,
        )
        ctx.batch_size = B
        ctx.gaussians_per_batch = G
        ctx.tiles_per_image = tiles_per_image
        ctx.tiles_x = tiles_x
        ctx.tile_size = tile_size
        ctx.enable_overflow_fallback = enable_overflow_fallback
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (
            perm,
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            depths_b,
            meta_i32,
            meta_f32,
            tile_counts,
            tile_offsets,
            binned_ids,
            tile_stop_counts,
            overflow_tile_ids,
            overflow_tile_offsets,
            overflow_sorted_ids,
        ) = ctx.saved_tensors

        grad_fast = grad_out.contiguous().clone()
        if ctx.enable_overflow_fallback and overflow_tile_ids.numel() > 0:
            _zero_tile_images_(grad_fast, overflow_tile_ids, ctx.tiles_per_image, ctx.tiles_x, ctx.tile_size)

        g_means_flat, g_conics_flat, g_colors_flat, g_opacities_flat = torch.ops.gsplat_metal_v8_project3d.render_fast_backward_saved(
            grad_fast,
            means_flat,
            conics_flat,
            colors_flat,
            opacities_flat,
            meta_i32,
            meta_f32,
            tile_counts,
            tile_offsets,
            binned_ids,
            tile_stop_counts,
        )

        if ctx.enable_overflow_fallback and overflow_tile_ids.numel() > 0:
            grad_tiles = _gather_tile_images(grad_out.contiguous(), overflow_tile_ids, ctx.tiles_per_image, ctx.tiles_x, ctx.tile_size)
            go_means, go_conics, go_colors, go_opacities = torch.ops.gsplat_metal_v8_project3d.render_overflow_backward(
                grad_tiles,
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                meta_i32,
                meta_f32,
                overflow_tile_ids,
                overflow_tile_offsets,
                overflow_sorted_ids,
            )
            g_means_flat = g_means_flat + go_means
            g_conics_flat = g_conics_flat + go_conics
            g_colors_flat = g_colors_flat + go_colors
            g_opacities_flat = g_opacities_flat + go_opacities

        B = ctx.batch_size
        G = ctx.gaussians_per_batch
        g_means_b = _unsort_batched(g_means_flat.view(B, G, 2), perm)
        g_conics_b = _unsort_batched(g_conics_flat.view(B, G, 3), perm)
        g_colors_b = _unsort_batched(g_colors_flat.view(B, G, 3), perm)
        g_opacities_b = _unsort_batched(g_opacities_flat.view(B, G), perm)
        g_depths_b = torch.zeros_like(depths_b)
        return g_means_b, g_conics_b, g_colors_b, g_opacities_b, g_depths_b, None, None, None


def _rasterize_chunk_eval(
    means2d_b: Tensor,
    conics_b: Tensor,
    colors_b: Tensor,
    opacities_b: Tensor,
    depths_b: Tensor,
    config: RasterConfig,
) -> Tensor:
    B, G = means2d_b.shape[:2]
    meta_i32, meta_f32 = _make_meta(config, means2d_b.device, B, G)
    perm = torch.argsort(depths_b.detach(), dim=1, stable=True)
    means2d_s = _batched_gather_2d(means2d_b, perm).contiguous()
    conics_s = _batched_gather_2d(conics_b, perm).contiguous()
    colors_s = _batched_gather_2d(colors_b, perm).contiguous()
    opacities_s = _batched_gather_1d(opacities_b, perm).contiguous()

    means_flat = means2d_s.reshape(B * G, 2).contiguous()
    conics_flat = conics_s.reshape(B * G, 3).contiguous()
    colors_flat = colors_s.reshape(B * G, 3).contiguous()
    opacities_flat = opacities_s.reshape(B * G).contiguous()

    tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v8_project3d.bin(
        means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32
    )
    out_fast = torch.ops.gsplat_metal_v8_project3d.render_fast_forward_eval(
        means_flat, conics_flat, colors_flat, opacities_flat, meta_i32, meta_f32, tile_counts, tile_offsets, binned_ids
    )

    if config.enable_overflow_fallback:
        overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids = _gather_overflow_segments(
            tile_counts, tile_offsets, binned_ids, int(meta_i32[7].item())
        )
        if overflow_tile_ids.numel() > 0:
            overflow_tile_imgs = torch.ops.gsplat_metal_v8_project3d.render_overflow_forward(
                means_flat,
                conics_flat,
                colors_flat,
                opacities_flat,
                meta_i32,
                meta_f32,
                overflow_tile_ids,
                overflow_tile_offsets,
                overflow_sorted_ids,
            )
            out = out_fast.clone()
            _scatter_tile_images_(out, overflow_tile_ids, overflow_tile_imgs, int(meta_i32[10].item()), int(meta_i32[3].item()), int(meta_i32[4].item()))
            return out
    elif bool((tile_counts > int(meta_i32[7].item())).any().item()):
        raise RuntimeError(
            f"Tile overflow detected with max_fast_pairs={int(meta_i32[7].item())}. "
            "Enable overflow fallback or increase the runtime cap."
        )
    return out_fast


def _rasterize_batched(
    means2d_b: Tensor,
    conics_b: Tensor,
    colors_b: Tensor,
    opacities_b: Tensor,
    depths_b: Tensor,
    config: RasterConfig,
) -> Tensor:
    B, G = means2d_b.shape[:2]
    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    chunk_b = _choose_batch_chunk_size(config, B, G, tiles_y * tiles_x)

    outs = []
    train_mode = _should_use_training_path(means2d_b, conics_b, colors_b, opacities_b)
    for b0 in range(0, B, chunk_b):
        b1 = min(B, b0 + chunk_b)
        m = means2d_b[b0:b1].contiguous()
        q = conics_b[b0:b1].contiguous()
        c = colors_b[b0:b1].contiguous()
        o = opacities_b[b0:b1].contiguous()
        d = depths_b[b0:b1].contiguous()

        if train_mode:
            meta_i32, meta_f32 = _make_meta(config, m.device, b1 - b0, G)
            outs.append(_RasterizeProjectedGaussiansV8Project3D.apply(m, q, c, o, d, meta_i32, meta_f32, bool(config.enable_overflow_fallback)))
        else:
            outs.append(_rasterize_chunk_eval(m, q, c, o, d, config))
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def rasterize_projected_gaussians(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: RasterConfig,
) -> Tensor:
    _check_inputs(means2d, conics, colors, opacities, depths)
    means2d_b, conics_b, colors_b, opacities_b, depths_b, was_batched = _normalize_inputs(
        means2d, conics, colors, opacities, depths
    )
    _runtime_validate(config)
    out = _rasterize_batched(means2d_b, conics_b, colors_b, opacities_b, depths_b, config)
    return out if was_batched else out[0]


def rasterize_pinhole_gaussians(
    means3d: Tensor,
    scales: Tensor,
    quats: Tensor,
    opacities: Tensor,
    colors: Tensor,
    fx,
    fy,
    cx,
    cy,
    camera_to_world=None,
    near_plane: float = 1e-4,
    config: RasterConfig | None = None,
) -> Tensor:
    if config is None:
        raise ValueError("config is required")
    _check_3d_inputs(means3d, scales, quats, opacities)
    means3d_b, scales_b, quats_b, opacities_b, was_batched = _normalize_3d_inputs(means3d, scales, quats, opacities)
    batch_size, gaussians_per_batch = means3d_b.shape[:2]
    colors_b = _normalize_colors(colors, batch_size, gaussians_per_batch, was_batched)
    if colors_b.device != means3d_b.device:
        raise ValueError("colors must be on the same device as means3d")
    if colors_b.dtype != torch.float32:
        raise ValueError("colors must be float32")

    means2d, conics, projected_opacities, depths = _project_pinhole_gaussians_batched(
        means3d_b.contiguous(),
        scales_b.contiguous(),
        quats_b.contiguous(),
        opacities_b.contiguous(),
        fx,
        fy,
        cx,
        cy,
        camera_to_world,
        near_plane,
    )
    out = _rasterize_batched(
        means2d.contiguous(),
        conics.contiguous(),
        colors_b.contiguous(),
        projected_opacities.contiguous(),
        depths.contiguous(),
        config,
    )
    return out if was_batched else out[0]


@torch.no_grad()
def profile_projected_gaussians(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    depths: Tensor,
    config: RasterConfig,
    *,
    run_forward: bool = False,
    return_image: bool = False,
) -> Dict[str, Any]:
    _check_inputs(means2d, conics, colors, opacities, depths)
    means2d_b, conics_b, colors_b, opacities_b, depths_b, was_batched = _normalize_inputs(
        means2d, conics, colors, opacities, depths
    )
    B, G = means2d_b.shape[:2]
    _runtime_validate(config)

    tiles_y = (config.height + config.tile_size - 1) // config.tile_size
    tiles_x = (config.width + config.tile_size - 1) // config.tile_size
    tiles_per_image = tiles_y * tiles_x
    chunk_b = _choose_batch_chunk_size(config, B, G, tiles_per_image)

    all_tile_counts = []
    all_stop_counts = []
    images = []

    for b0 in range(0, B, chunk_b):
        b1 = min(B, b0 + chunk_b)
        m = means2d_b[b0:b1].contiguous()
        q = conics_b[b0:b1].contiguous()
        c = colors_b[b0:b1].contiguous()
        o = opacities_b[b0:b1].contiguous()
        d = depths_b[b0:b1].contiguous()

        perm = torch.argsort(d.detach(), dim=1, stable=True)
        m_s = _batched_gather_2d(m, perm).contiguous().reshape(-1, 2)
        q_s = _batched_gather_2d(q, perm).contiguous().reshape(-1, 3)
        c_s = _batched_gather_2d(c, perm).contiguous().reshape(-1, 3)
        o_s = _batched_gather_1d(o, perm).contiguous().reshape(-1)

        meta_i32, meta_f32 = _make_meta(config, means2d_b.device, b1 - b0, G)
        tile_counts, tile_offsets, binned_ids = torch.ops.gsplat_metal_v8_project3d.bin(
            m_s, q_s, c_s, o_s, meta_i32, meta_f32
        )
        all_tile_counts.append(tile_counts.detach().cpu().to(torch.float32))

        if run_forward or return_image:
            if return_image:
                chunk_img = _rasterize_chunk_eval(m, q, c, o, d, config)
                images.append(chunk_img)
            _, stop_counts = torch.ops.gsplat_metal_v8_project3d.render_fast_forward_state(
                m_s, q_s, c_s, o_s, meta_i32, meta_f32, binned_ids, tile_counts, tile_offsets
            )
            all_stop_counts.append(stop_counts.detach().cpu().to(torch.float32))

    counts_cpu = torch.cat(all_tile_counts, dim=0) if all_tile_counts else torch.zeros(0, dtype=torch.float32)
    stats: Dict[str, Any] = {
        "batch_size": int(B),
        "gaussians_per_batch": int(G),
        "height": int(config.height),
        "width": int(config.width),
        "tile_size": int(config.tile_size),
        "tiles": int(counts_cpu.numel()),
        "total_pairs": int(counts_cpu.sum().item()) if counts_cpu.numel() else 0,
        "mean_pairs_per_tile": float(counts_cpu.mean().item()) if counts_cpu.numel() else 0.0,
        "p95_pairs_per_tile": float(torch.quantile(counts_cpu, 0.95).item()) if counts_cpu.numel() else 0.0,
        "max_pairs_per_tile": int(counts_cpu.max().item()) if counts_cpu.numel() else 0,
        "overflow_tile_count": int((counts_cpu > int(config.max_fast_pairs)).sum().item()) if counts_cpu.numel() else 0,
        "chosen_batch_chunk": int(chunk_b),
    }

    if all_stop_counts:
        stop_cpu = torch.cat(all_stop_counts, dim=0)
        denom = torch.clamp(counts_cpu, min=1.0)
        stop_ratio = torch.where(counts_cpu > 0, stop_cpu / denom, torch.zeros_like(stop_cpu))
        stats.update(
            {
                "mean_stop_count": float(stop_cpu.mean().item()),
                "p95_stop_count": float(torch.quantile(stop_cpu, 0.95).item()),
                "max_stop_count": int(stop_cpu.max().item()),
                "mean_stop_ratio": float(stop_ratio.mean().item()),
                "p95_stop_ratio": float(torch.quantile(stop_ratio, 0.95).item()),
            }
        )

    if return_image:
        out = torch.cat(images, dim=0) if len(images) > 1 else images[0]
        return {"image": out if was_batched else out[0], "stats": stats}
    return stats


class ProjectedGaussianRasterizer(torch.nn.Module):
    def __init__(self, config: RasterConfig):
        super().__init__()
        self.config = config

    def forward(self, means2d: Tensor, conics: Tensor, colors: Tensor, opacities: Tensor, depths: Tensor) -> Tensor:
        return rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, self.config)
