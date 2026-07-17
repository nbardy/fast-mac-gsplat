from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

MAX_REALRAY_BOUNDARIES = 128
MAX_REALRAY_FUSED_MSE_BOUNDARIES = 256
MAX_REALRAY_FUSED_MSE_CAP224_BOUNDARIES = 224

_EXTENSION_LOAD_ERROR: Exception | None = None


def _load_extension_library() -> None:
    candidates = sorted(Path(__file__).resolve().parent.glob("_C*.so"))
    if not candidates:
        return
    torch.ops.load_library(str(candidates[0]))


try:
    _load_extension_library()
except Exception as exc:
    _EXTENSION_LOAD_ERROR = exc


@dataclass(frozen=True)
class PowerBoundaryConfig:
    camera_velocity_x: float
    invalid_epsilon: float = 1.0e-7


@dataclass(frozen=True)
class RealRayReplayConfig:
    near: float
    far: float
    invalid_epsilon: float = 1.0e-7
    transmittance_threshold: float = 1.0e-4


def _require_mps_tensor(tensor: Tensor, *, name: str, dtype: torch.dtype, cols: int | None) -> None:
    if tensor.device.type != "mps":
        raise ValueError(f"{name} must be on MPS")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if cols is None:
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be rank-1")
    elif tensor.ndim != 2 or tensor.shape[1] != cols:
        raise ValueError(f"{name} must have shape [N,{cols}]")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _candidate_mask_time_slab_count(candidate_mask_u32: Tensor, *, beam_count: int, name: str) -> int:
    if beam_count <= 0:
        raise ValueError("beam_count must be positive")
    if candidate_mask_u32.shape[0] % beam_count != 0:
        raise ValueError(f"{name} length must be beam_count * time_slab_count")
    time_slab_count = candidate_mask_u32.shape[0] // beam_count
    if time_slab_count <= 0:
        raise ValueError(f"{name} must contain at least one time slab")
    return int(time_slab_count)


def _validate_csr_offsets_cpu(
    row_offsets_cpu: Tensor,
    *,
    candidate_count: int,
    max_boundaries: int = MAX_REALRAY_BOUNDARIES,
    name: str = "candidate_row_offsets_i32",
) -> None:
    if row_offsets_cpu.numel() == 0:
        raise ValueError(f"{name} must contain at least one offset")
    if int(row_offsets_cpu[0].item()) != 0:
        raise ValueError(f"{name}[0] must be 0")
    row_lengths = row_offsets_cpu[1:] - row_offsets_cpu[:-1]
    if bool((row_lengths < 0).any().item()):
        raise ValueError(f"{name} must be monotonic nondecreasing")
    if int(row_offsets_cpu[-1].item()) != int(candidate_count):
        raise ValueError(f"{name}[-1] must match candidate count")
    if row_lengths.numel() and int(row_lengths.max().item()) > max_boundaries:
        raise ValueError(
            f"{name} row contains {int(row_lengths.max().item())} candidates, "
            f"exceeding Metal local boundary cap {max_boundaries}"
        )


def _validate_segment_tape_offsets_cpu(
    segment_offsets_cpu: Tensor,
    *,
    sample_count: int,
    segment_count: int,
    max_segments_per_sample: int | None,
) -> None:
    if segment_offsets_cpu.numel() != sample_count + 1:
        raise ValueError("segment_offsets_i32 length must be track_count * frame_count + 1")
    if int(segment_offsets_cpu[0].item()) != 0:
        raise ValueError("segment_offsets_i32[0] must be 0")
    row_lengths = segment_offsets_cpu[1:] - segment_offsets_cpu[:-1]
    if bool((row_lengths < 0).any().item()):
        raise ValueError("segment_offsets_i32 must be monotonic nondecreasing")
    if int(segment_offsets_cpu[-1].item()) != int(segment_count):
        raise ValueError("segment_offsets_i32[-1] must match segment count")
    if max_segments_per_sample is not None and row_lengths.numel():
        max_row = int(row_lengths.max().item())
        if max_row > max_segments_per_sample:
            raise ValueError(
                f"segment tape row contains {max_row} segments, exceeding Metal replay cap {max_segments_per_sample}"
            )


def _validate_packed_endpoint_records_cpu(
    record_i32: Tensor,
    *,
    name: str,
    site_count: int,
    boundary_count: int,
) -> None:
    records = record_i32.detach().cpu().to(dtype=torch.int64)
    if records.numel() == 0:
        return
    if int(records.min().item()) < 0:
        raise ValueError(f"{name} must contain nonnegative packed endpoint records")
    owner_i64 = records & 255
    if int(owner_i64.max().item()) >= int(site_count):
        raise ValueError(f"{name} owner code must be < site_count")
    for field_name, shift in (("left", 8), ("right", 20)):
        cut_code_i64 = (records >> shift) & 4095
        cut_i64 = cut_code_i64 - 2
        if bool(((cut_code_i64 >= 2) & (cut_i64 >= int(boundary_count))).any().item()):
            raise ValueError(f"{name} {field_name} cut id must be < boundary_count")


def _validate_packed_endpoint_delta_records_cpu(
    *,
    base_record_i32: Tensor,
    change_record_i32: Tensor,
    site_count: int,
    boundary_count: int,
) -> None:
    _validate_packed_endpoint_records_cpu(
        base_record_i32,
        name="base_record_i32",
        site_count=site_count,
        boundary_count=boundary_count,
    )
    _validate_packed_endpoint_records_cpu(
        change_record_i32,
        name="change_record_i32",
        site_count=site_count,
        boundary_count=boundary_count,
    )


def _validate_csr_values(
    *,
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    row_count: int,
    candidate_count: int,
    boundary_count: int,
) -> None:
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    candidate_ids_cpu = candidate_boundary_ids_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_count)
    if candidate_count and (
        int(candidate_ids_cpu.min().item()) < 0 or int(candidate_ids_cpu.max().item()) >= boundary_count
    ):
        raise ValueError("candidate_boundary_ids_i32 values must be in [0, boundary_count)")


def count_power_boundary_events(
    boundary_f32: Tensor,
    beam_f32: Tensor,
    config: PowerBoundaryConfig,
    *,
    boundary_u32: Tensor | None = None,
    beam_u32: Tensor | None = None,
) -> Tensor:
    """Count Gate 0 power-boundary events with the local MPS Metal kernel.

    Layouts:
    - `boundary_f32`: `[B,4]` rows `(nx, nz, nt, b)`.
    - `beam_f32`: `[M,5]` rows `(u_center, t0, t1, near_depth, far_depth)`.
    - `boundary_u32`: optional `[B,4]` rows `(left_site, right_site, 0, 0)`.
    - `beam_u32`: optional `[M,4]` rows `(payload_id, flags, 0, 0)`.

    Returns an int32 `[M,8]` tensor matching `WF2PowerBoundaryCount` field
    order: `(beam_id, payload_id, boundary_event_count,
    invalid_denominator_count, flags, 0, 0, 0)`.
    """
    boundary_f32 = boundary_f32.contiguous()
    beam_f32 = beam_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(beam_f32, name="beam_f32", dtype=torch.float32, cols=5)

    if boundary_u32 is None:
        boundary_u32 = torch.zeros((boundary_f32.shape[0], 4), device=boundary_f32.device, dtype=torch.int32)
    if beam_u32 is None:
        beam_u32 = torch.zeros((beam_f32.shape[0], 4), device=beam_f32.device, dtype=torch.int32)
        beam_u32[:, 0] = torch.arange(beam_f32.shape[0], device=beam_f32.device, dtype=torch.int32)

    boundary_u32 = boundary_u32.contiguous()
    beam_u32 = beam_u32.contiguous()
    _require_mps_tensor(boundary_u32, name="boundary_u32", dtype=torch.int32, cols=4)
    _require_mps_tensor(beam_u32, name="beam_u32", dtype=torch.int32, cols=4)
    if boundary_u32.shape[0] != boundary_f32.shape[0]:
        raise ValueError("boundary_u32 row count must match boundary_f32")
    if beam_u32.shape[0] != beam_f32.shape[0]:
        raise ValueError("beam_u32 row count must match beam_f32")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "count_power_boundary_events"):
        raise RuntimeError("world_foam_lane2_fused_slab_v0 custom ops not found. Build this variant first.")
    config_i32 = torch.tensor(
        [boundary_f32.shape[0], beam_f32.shape[0]],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.camera_velocity_x, config.invalid_epsilon],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.count_power_boundary_events(
        boundary_f32,
        boundary_u32,
        beam_f32,
        beam_u32,
        config_i32,
        config_f32,
    )


def shared_signal_replay(
    boundary_f32: Tensor,
    candidate_mask_u32: Tensor,
    sites_f32: Tensor,
    site_signal_f32: Tensor,
    beam_f32: Tensor,
    frame_t_f32: Tensor,
    grad_output_f32: Tensor,
    config: PowerBoundaryConfig,
) -> tuple[Tensor, Tensor]:
    """Replay shared Gate 0 slab candidates and emit signal-gradient samples.

    `candidate_mask_u32` is an int32 bit mask per beam/time slab in row-major
    `[beam, slab]` order. A legacy length-`M` mask is treated as one slab.
    Bits `0..30` mark boundary rows that are shared candidates for that slab.
    The kernel emits:

    - output: `[M,T]` scalar segment-integral signal values.
    - grad_samples: `[M,T,N]` per-ray per-site signal-gradient samples.

    This is a fixed-segment site-signal VJP reference. It does not implement
    site-position, weight, topology, or sorting gradients.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_u32 = candidate_mask_u32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_signal_f32 = site_signal_f32.contiguous()
    beam_f32 = beam_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_output_f32 = grad_output_f32.contiguous()

    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(candidate_mask_u32, name="candidate_mask_u32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(site_signal_f32, name="site_signal_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(beam_f32, name="beam_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_output_f32.device.type != "mps":
        raise ValueError("grad_output_f32 must be on MPS")
    if grad_output_f32.dtype != torch.float32:
        raise ValueError("grad_output_f32 must be float32")
    if grad_output_f32.ndim != 2:
        raise ValueError("grad_output_f32 must have shape [M,T]")
    if not grad_output_f32.is_contiguous():
        raise ValueError("grad_output_f32 must be contiguous")
    if boundary_f32.shape[0] > 31:
        raise ValueError("shared_signal_replay currently supports at most 31 boundaries")
    if sites_f32.shape[0] > 32:
        raise ValueError("shared_signal_replay currently supports at most 32 sites")
    time_slab_count = _candidate_mask_time_slab_count(
        candidate_mask_u32,
        beam_count=beam_f32.shape[0],
        name="candidate_mask_u32",
    )
    if site_signal_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_signal_f32 length must match sites_f32 rows")
    if grad_output_f32.shape != (beam_f32.shape[0], frame_t_f32.shape[0]):
        raise ValueError("grad_output_f32 shape must be [beam_count, frame_count]")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_signal_replay"):
        raise RuntimeError("world_foam_lane2_fused_slab_v0 shared_signal_replay op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            beam_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor([config.camera_velocity_x], device=boundary_f32.device, dtype=torch.float32)
    return ops.shared_signal_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_signal_f32,
        beam_f32,
        frame_t_f32,
        grad_output_f32,
        config_i32,
        config_f32,
    )


def shared_rgb_replay(
    boundary_f32: Tensor,
    candidate_mask_u32: Tensor,
    sites_f32: Tensor,
    site_rgb_f32: Tensor,
    beam_f32: Tensor,
    frame_t_f32: Tensor,
    grad_output_rgb_f32: Tensor,
    config: PowerBoundaryConfig,
) -> tuple[Tensor, Tensor]:
    """Replay shared Gate 0 slab candidates and emit RGB-gradient samples.

    Returns:
    - output_rgb: `[M,T,3]` RGB segment-integral values.
    - grad_samples_rgb: `[M,T,N,3]` per-ray per-site RGB signal-gradient
      samples.

    `candidate_mask_u32` is row-major `[beam, slab]`; a length-`M` mask is one
    slab. This is still a fixed-segment signal VJP reference. It does not implement
    alpha/depth compositing, site-position, weight, topology, or sorting
    gradients.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_u32 = candidate_mask_u32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgb_f32 = site_rgb_f32.contiguous()
    beam_f32 = beam_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_output_rgb_f32 = grad_output_rgb_f32.contiguous()

    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(candidate_mask_u32, name="candidate_mask_u32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(site_rgb_f32, name="site_rgb_f32", dtype=torch.float32, cols=3)
    _require_mps_tensor(beam_f32, name="beam_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_output_rgb_f32.device.type != "mps":
        raise ValueError("grad_output_rgb_f32 must be on MPS")
    if grad_output_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_output_rgb_f32 must be float32")
    if grad_output_rgb_f32.ndim != 3 or grad_output_rgb_f32.shape[2] != 3:
        raise ValueError("grad_output_rgb_f32 must have shape [M,T,3]")
    if not grad_output_rgb_f32.is_contiguous():
        raise ValueError("grad_output_rgb_f32 must be contiguous")
    if boundary_f32.shape[0] > 31:
        raise ValueError("shared_rgb_replay currently supports at most 31 boundaries")
    if sites_f32.shape[0] > 32:
        raise ValueError("shared_rgb_replay currently supports at most 32 sites")
    time_slab_count = _candidate_mask_time_slab_count(
        candidate_mask_u32,
        beam_count=beam_f32.shape[0],
        name="candidate_mask_u32",
    )
    if site_rgb_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgb_f32 row count must match sites_f32 rows")
    if grad_output_rgb_f32.shape[:2] != (beam_f32.shape[0], frame_t_f32.shape[0]):
        raise ValueError("grad_output_rgb_f32 shape must be [beam_count, frame_count, 3]")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_rgb_replay"):
        raise RuntimeError("world_foam_lane2_fused_slab_v0 shared_rgb_replay op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            beam_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor([config.camera_velocity_x], device=boundary_f32.device, dtype=torch.float32)
    return ops.shared_rgb_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgb_f32,
        beam_f32,
        frame_t_f32,
        grad_output_rgb_f32,
        config_i32,
        config_f32,
    )


def shared_rgba_depth_replay(
    boundary_f32: Tensor,
    candidate_mask_u32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    beam_f32: Tensor,
    frame_t_f32: Tensor,
    config: PowerBoundaryConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay shared candidates and composite RGB, alpha, and expected depth.

    Returns:
    - output_rgb: `[M,T,3]` front-to-back RGB.
    - output_alpha: `[M,T]` accumulated alpha.
    - output_depth: `[M,T]` alpha-weighted expected depth, or far depth when
      alpha is zero.

    `candidate_mask_u32` is row-major `[beam, slab]`; a length-`M` mask is one
    slab. This is a forward-only compositor proof. It does not implement
    gradients, site-position updates, topology changes, or a trainer path.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_u32 = candidate_mask_u32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    beam_f32 = beam_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()

    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(candidate_mask_u32, name="candidate_mask_u32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(beam_f32, name="beam_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if boundary_f32.shape[0] > 31:
        raise ValueError("shared_rgba_depth_replay currently supports at most 31 boundaries")
    if sites_f32.shape[0] > 32:
        raise ValueError("shared_rgba_depth_replay currently supports at most 32 sites")
    time_slab_count = _candidate_mask_time_slab_count(
        candidate_mask_u32,
        beam_count=beam_f32.shape[0],
        name="candidate_mask_u32",
    )
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_rgba_depth_replay"):
        raise RuntimeError("world_foam_lane2_fused_slab_v0 shared_rgba_depth_replay op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            beam_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor([config.camera_velocity_x], device=boundary_f32.device, dtype=torch.float32)
    return ops.shared_rgba_depth_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgba_f32,
        beam_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def shared_rgba_depth_vjp(
    boundary_f32: Tensor,
    candidate_mask_u32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    beam_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: PowerBoundaryConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Replay shared candidates and emit fixed-segment RGBA VJP samples.

    Returns `output_rgb [M,T,3]`, `output_alpha [M,T]`, `output_depth [M,T]`,
    and `grad_samples_rgba [M,T,N,4]`. `candidate_mask_u32` is row-major
    `[beam, slab]`; a length-`M` mask is one slab. The final channel is the
    density gradient. Boundary cuts, owners, site positions, and topology are
    fixed.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_u32 = candidate_mask_u32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    beam_f32 = beam_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()

    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(candidate_mask_u32, name="candidate_mask_u32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(beam_f32, name="beam_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps":
        raise ValueError("grad_rgb_f32 must be on MPS")
    if grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [M,T,3]")
    if not grad_rgb_f32.is_contiguous():
        raise ValueError("grad_rgb_f32 must be contiguous")
    if grad_alpha_f32.device.type != "mps":
        raise ValueError("grad_alpha_f32 must be on MPS")
    if grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [M,T]")
    if not grad_alpha_f32.is_contiguous():
        raise ValueError("grad_alpha_f32 must be contiguous")
    if grad_depth_f32.device.type != "mps":
        raise ValueError("grad_depth_f32 must be on MPS")
    if grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [M,T]")
    if not grad_depth_f32.is_contiguous():
        raise ValueError("grad_depth_f32 must be contiguous")
    if boundary_f32.shape[0] > 31:
        raise ValueError("shared_rgba_depth_vjp currently supports at most 31 boundaries")
    if sites_f32.shape[0] > 32:
        raise ValueError("shared_rgba_depth_vjp currently supports at most 32 sites")
    time_slab_count = _candidate_mask_time_slab_count(
        candidate_mask_u32,
        beam_count=beam_f32.shape[0],
        name="candidate_mask_u32",
    )
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    expected_shape = (beam_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [beam_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 shape must be [beam_count, frame_count]")
    if grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_depth_f32 shape must be [beam_count, frame_count]")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_rgba_depth_vjp"):
        raise RuntimeError("world_foam_lane2_fused_slab_v0 shared_rgba_depth_vjp op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            beam_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor([config.camera_velocity_x], device=boundary_f32.device, dtype=torch.float32)
    return ops.shared_rgba_depth_vjp(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_rgba_f32,
        beam_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def realray_rgba_depth_replay(
    boundary_f32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render true camera rays through 4D power-cell boundaries.

    Layouts:
    - `boundary_f32`: `[B,5]` rows `(nx, ny, nz, nt, b)`.
    - `sites_f32`: `[S,5]` rows `(x, y, z, t, weight)`.
    - `site_rgba_f32`: `[S,4]` rows `(r, g, b, density)`.
    - `rays_f32`: `[R,6]` rows `(ox, oy, oz, dx, dy, dz)`.
    - `frame_t_f32`: `[R]` normalized time for each ray.

    Returns `output_rgb [R,3]`, `output_alpha [R]`, and `output_depth [R]`.
    This is a forward-only linear per-ray baseline. It does not share work over
    time and does not implement backward gradients.
    """
    boundary_f32 = boundary_f32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(rays_f32, name="rays_f32", dtype=torch.float32, cols=6)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if boundary_f32.shape[0] > 128:
        raise ValueError("realray_rgba_depth_replay currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("realray_rgba_depth_replay currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if frame_t_f32.shape[0] != rays_f32.shape[0]:
        raise ValueError("frame_t_f32 length must match rays_f32 rows")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "realray_rgba_depth_replay"):
        raise RuntimeError("world_foam_lane2_fused_slab_v0 realray_rgba_depth_replay op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [boundary_f32.shape[0], rays_f32.shape[0], sites_f32.shape[0]],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.realray_rgba_depth_replay(
        boundary_f32,
        sites_f32,
        site_rgba_f32,
        rays_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def shared_realray_rgba_depth_replay(
    boundary_f32: Tensor,
    candidate_mask_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render true camera-ray tracks using shared time-slab candidate masks.

    Layouts:
    - `boundary_f32`: `[B,5]` rows `(nx, ny, nz, nt, b)`.
    - `candidate_mask_i32`: `[R_tracks * time_slabs, W]` bitset words.
      Word `boundary_id // 32`, bit `boundary_id % 32` marks a shared
      boundary candidate for that pixel track and time slab.
    - `sites_f32`: `[S,5]` rows `(x, y, z, t, weight)`.
    - `site_rgba_f32`: `[S,4]` rows `(r, g, b, density)`.
    - `track_rays_f32`: `[R_tracks,6]` rows `(ox, oy, oz, dx, dy, dz)`.
    - `frame_t_f32`: `[T]` normalized frame times.

    Returns `output_rgb [R_tracks,T,3]`, `output_alpha [R_tracks,T]`, and
    `output_depth [R_tracks,T]`. This is forward-only fixed-candidate replay.
    It does not implement backward gradients, topology changes, or training.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_i32 = candidate_mask_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    track_rays_f32 = track_rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    if candidate_mask_i32.device.type != "mps":
        raise ValueError("candidate_mask_i32 must be on MPS")
    if candidate_mask_i32.dtype != torch.int32:
        raise ValueError("candidate_mask_i32 must have dtype torch.int32")
    if candidate_mask_i32.ndim != 2:
        raise ValueError("candidate_mask_i32 must have shape [track_count * time_slab_count, mask_word_count]")
    if not candidate_mask_i32.is_contiguous():
        raise ValueError("candidate_mask_i32 must be contiguous")
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(track_rays_f32, name="track_rays_f32", dtype=torch.float32, cols=6)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if boundary_f32.shape[0] > 128:
        raise ValueError("shared_realray_rgba_depth_replay currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("shared_realray_rgba_depth_replay currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if candidate_mask_i32.shape[0] % track_rays_f32.shape[0] != 0:
        raise ValueError("candidate_mask_i32 rows must be track_count * time_slab_count")
    time_slab_count = candidate_mask_i32.shape[0] // track_rays_f32.shape[0]
    if time_slab_count <= 0:
        raise ValueError("candidate_mask_i32 must contain at least one time slab")
    expected_words = (boundary_f32.shape[0] + 31) // 32
    if candidate_mask_i32.shape[1] != expected_words:
        raise ValueError(
            f"candidate_mask_i32 word count must be ceil(boundary_count / 32) = {expected_words}"
        )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_realray_rgba_depth_replay op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_rays_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            candidate_mask_i32.shape[1],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.shared_realray_rgba_depth_replay(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def shared_realray_rgba_depth_vjp(
    boundary_f32: Tensor,
    candidate_mask_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Render shared true camera-ray tracks and emit fixed-segment RGBA VJP samples.

    Returns `output_rgb [K,T,3]`, `output_alpha [K,T]`, `output_depth [K,T]`,
    and `grad_samples_rgba [K,T,S,4]`. The final gradient channel is density.
    Candidate cuts, segment owners, site positions, weights, and topology are
    fixed; this is not an autograd trainer boundary.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_i32 = candidate_mask_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    track_rays_f32 = track_rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    if candidate_mask_i32.device.type != "mps":
        raise ValueError("candidate_mask_i32 must be on MPS")
    if candidate_mask_i32.dtype != torch.int32:
        raise ValueError("candidate_mask_i32 must have dtype torch.int32")
    if candidate_mask_i32.ndim != 2:
        raise ValueError("candidate_mask_i32 must have shape [track_count * time_slab_count, mask_word_count]")
    if not candidate_mask_i32.is_contiguous():
        raise ValueError("candidate_mask_i32 must be contiguous")
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(track_rays_f32, name="track_rays_f32", dtype=torch.float32, cols=6)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps":
        raise ValueError("grad_rgb_f32 must be on MPS")
    if grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not grad_rgb_f32.is_contiguous():
        raise ValueError("grad_rgb_f32 must be contiguous")
    if grad_alpha_f32.device.type != "mps":
        raise ValueError("grad_alpha_f32 must be on MPS")
    if grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if not grad_alpha_f32.is_contiguous():
        raise ValueError("grad_alpha_f32 must be contiguous")
    if grad_depth_f32.device.type != "mps":
        raise ValueError("grad_depth_f32 must be on MPS")
    if grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if not grad_depth_f32.is_contiguous():
        raise ValueError("grad_depth_f32 must be contiguous")
    if boundary_f32.shape[0] > 128:
        raise ValueError("shared_realray_rgba_depth_vjp currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("shared_realray_rgba_depth_vjp currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if candidate_mask_i32.shape[0] % track_rays_f32.shape[0] != 0:
        raise ValueError("candidate_mask_i32 rows must be track_count * time_slab_count")
    time_slab_count = candidate_mask_i32.shape[0] // track_rays_f32.shape[0]
    expected_words = (boundary_f32.shape[0] + 31) // 32
    if candidate_mask_i32.shape[1] != expected_words:
        raise ValueError(
            f"candidate_mask_i32 word count must be ceil(boundary_count / 32) = {expected_words}"
        )
    expected_shape = (track_rays_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 shape must be [track_count, frame_count]")
    if grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_depth_f32 shape must be [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_realray_rgba_depth_vjp"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_realray_rgba_depth_vjp op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_rays_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            candidate_mask_i32.shape[1],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.shared_realray_rgba_depth_vjp(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def shared_realray_rgba_depth_vjp_reduce(
    boundary_f32: Tensor,
    candidate_mask_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Render shared true camera-ray tracks and reduce fixed-segment RGBA grads.

    Returns `output_rgb [K,T,3]`, `output_alpha [K,T]`, `output_depth [K,T]`,
    and `grad_site_rgba [S,4]`. This avoids materializing the smoke-only
    `[K,T,S,4]` gradient-sample tensor by using an internal chunked partial
    reduction, but it still fixes segment geometry, owner selection, sorting,
    masks, site positions, weights, rays, and topology.
    """
    boundary_f32 = boundary_f32.contiguous()
    candidate_mask_i32 = candidate_mask_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    track_rays_f32 = track_rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    if candidate_mask_i32.device.type != "mps":
        raise ValueError("candidate_mask_i32 must be on MPS")
    if candidate_mask_i32.dtype != torch.int32:
        raise ValueError("candidate_mask_i32 must have dtype torch.int32")
    if candidate_mask_i32.ndim != 2:
        raise ValueError("candidate_mask_i32 must have shape [track_count * time_slab_count, mask_word_count]")
    if not candidate_mask_i32.is_contiguous():
        raise ValueError("candidate_mask_i32 must be contiguous")
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(track_rays_f32, name="track_rays_f32", dtype=torch.float32, cols=6)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps":
        raise ValueError("grad_rgb_f32 must be on MPS")
    if grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not grad_rgb_f32.is_contiguous():
        raise ValueError("grad_rgb_f32 must be contiguous")
    if grad_alpha_f32.device.type != "mps":
        raise ValueError("grad_alpha_f32 must be on MPS")
    if grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if not grad_alpha_f32.is_contiguous():
        raise ValueError("grad_alpha_f32 must be contiguous")
    if grad_depth_f32.device.type != "mps":
        raise ValueError("grad_depth_f32 must be on MPS")
    if grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if not grad_depth_f32.is_contiguous():
        raise ValueError("grad_depth_f32 must be contiguous")
    if track_rays_f32.shape[0] <= 0:
        raise ValueError("track_rays_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if boundary_f32.shape[0] > 128:
        raise ValueError("shared_realray_rgba_depth_vjp_reduce currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("shared_realray_rgba_depth_vjp_reduce currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if candidate_mask_i32.shape[0] % track_rays_f32.shape[0] != 0:
        raise ValueError("candidate_mask_i32 rows must be track_count * time_slab_count")
    time_slab_count = candidate_mask_i32.shape[0] // track_rays_f32.shape[0]
    expected_words = (boundary_f32.shape[0] + 31) // 32
    if candidate_mask_i32.shape[1] != expected_words:
        raise ValueError(
            f"candidate_mask_i32 word count must be ceil(boundary_count / 32) = {expected_words}"
        )
    expected_shape = (track_rays_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 shape must be [track_count, frame_count]")
    if grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_depth_f32 shape must be [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_realray_rgba_depth_vjp_reduce"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_realray_rgba_depth_vjp_reduce op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_rays_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            candidate_mask_i32.shape[1],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.shared_realray_rgba_depth_vjp_reduce(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def shared_realray_rgba_depth_vjp_reduce_csr(
    boundary_f32: Tensor,
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    reduce_chunk_size: int = 4,
    use_direct_atomic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Render shared real-ray tracks and reduce site RGBA grads from CSR candidates.

    `row_index_i32 [K]` maps each track to a CSR row group. Per-track CSR uses
    `row_index_i32[track] = track`; tiled CSR maps many tracks to the same row
    group. The per-slab row is `row_index * time_slab_count + slab_id`.
    """
    boundary_f32 = boundary_f32.contiguous()
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_boundary_ids_i32 = candidate_boundary_ids_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    track_rays_f32 = track_rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_boundary_ids_i32, name="candidate_boundary_ids_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(track_rays_f32, name="track_rays_f32", dtype=torch.float32, cols=6)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps":
        raise ValueError("grad_rgb_f32 must be on MPS")
    if grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not grad_rgb_f32.is_contiguous():
        raise ValueError("grad_rgb_f32 must be contiguous")
    if grad_alpha_f32.device.type != "mps":
        raise ValueError("grad_alpha_f32 must be on MPS")
    if grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if not grad_alpha_f32.is_contiguous():
        raise ValueError("grad_alpha_f32 must be contiguous")
    if grad_depth_f32.device.type != "mps":
        raise ValueError("grad_depth_f32 must be on MPS")
    if grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if not grad_depth_f32.is_contiguous():
        raise ValueError("grad_depth_f32 must be contiguous")
    if track_rays_f32.shape[0] <= 0:
        raise ValueError("track_rays_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if boundary_f32.shape[0] > 128:
        raise ValueError("shared_realray_rgba_depth_vjp_reduce_csr currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("shared_realray_rgba_depth_vjp_reduce_csr currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != track_rays_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if reduce_chunk_size <= 0:
        raise ValueError("reduce_chunk_size must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    _validate_csr_values(
        row_index_i32=row_index_i32,
        candidate_row_offsets_i32=candidate_row_offsets_i32,
        candidate_boundary_ids_i32=candidate_boundary_ids_i32,
        row_count=row_count,
        candidate_count=candidate_boundary_ids_i32.shape[0],
        boundary_count=boundary_f32.shape[0],
    )
    expected_shape = (track_rays_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 shape must be [track_count, frame_count]")
    if grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_depth_f32 shape must be [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "shared_realray_rgba_depth_vjp_reduce_csr"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 shared_realray_rgba_depth_vjp_reduce_csr op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_rays_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_boundary_ids_i32.shape[0],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.shared_realray_rgba_depth_vjp_reduce_csr(
        boundary_f32,
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_vjp_reduce(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    reduce_chunk_size: int = 4,
    use_direct_atomic: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Render mixed affine CSR candidates and reduce site RGBA gradients.

    This matches `fused_slab_affine_num32_den16_realray_rgba_depth_replay`
    for forward values, while adding the same frozen-geometry site-RGBA VJP
    contract used by the existing CSR autograd path.
    """
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps":
        raise ValueError("grad_rgb_f32 must be on MPS")
    if grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not grad_rgb_f32.is_contiguous():
        raise ValueError("grad_rgb_f32 must be contiguous")
    if grad_alpha_f32.device.type != "mps":
        raise ValueError("grad_alpha_f32 must be on MPS")
    if grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if not grad_alpha_f32.is_contiguous():
        raise ValueError("grad_alpha_f32 must be contiguous")
    if grad_depth_f32.device.type != "mps":
        raise ValueError("grad_depth_f32 must be on MPS")
    if grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if not grad_depth_f32.is_contiguous():
        raise ValueError("grad_depth_f32 must be contiguous")
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if sites_f32.shape[0] > 64:
        raise ValueError("fused_slab_affine_num32_den16_vjp_reduce currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate_depth_den_f16 row count must match candidate_depth_num_f32 rows")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if reduce_chunk_size <= 0:
        raise ValueError("reduce_chunk_size must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 shape must be [track_count, frame_count]")
    if grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_depth_f32 shape must be [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = (
        "fused_slab_affine_num32_den16_vjp_direct_atomic"
        if use_direct_atomic
        else "fused_slab_affine_num32_den16_vjp_reduce"
    )
    if not hasattr(ops, op_name):
        raise RuntimeError(
            f"world_foam_lane2_fused_slab_v0 {op_name} op not found. "
            "Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            reduce_chunk_size,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_vjp_direct_atomic(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Render mixed affine CSR candidates and accumulate site RGBA gradients with atomics."""
    return fused_slab_affine_num32_den16_vjp_reduce(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        reduce_chunk_size=1,
        use_direct_atomic=True,
    )


def fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> Tensor:
    """Accumulate mixed affine site RGBA gradients without writing replay outputs."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate numerator and denominator row counts must match")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape or grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 and grad_depth_f32 must have shape [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(
            f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            1,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def _segment_tape_config_tensors(
    *,
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_segments_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    segment_mid_f32 = segment_mid_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    _require_mps_tensor(segment_offsets_i32, name="segment_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(segment_owner_i32, name="segment_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(segment_length_f32, name="segment_length_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(segment_mid_f32, name="segment_mid_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0:
        raise ValueError("track_count and frame_count must be positive")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 64:
        raise ValueError("site_rgba_f32 row count must be in [1, 64]")
    if segment_owner_i32.shape[0] != segment_length_f32.shape[0] or segment_owner_i32.shape[0] != segment_mid_f32.shape[0]:
        raise ValueError("segment owner, length, and mid arrays must have matching lengths")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        segment_offsets_i32.detach().cpu(),
        sample_count=track_count * frame_count,
        segment_count=segment_owner_i32.shape[0],
        max_segments_per_sample=max_segments_per_sample,
    )
    config_i32 = torch.tensor(
        [track_count, frame_count, site_rgba_f32.shape[0], segment_owner_i32.shape[0]],
        device=segment_offsets_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.far, config.transmittance_threshold],
        device=segment_offsets_i32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def _segment_tape_rgb_mse_config_tensors(
    *,
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_segments_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    _require_mps_tensor(segment_offsets_i32, name="segment_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(segment_owner_i32, name="segment_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(segment_length_f32, name="segment_length_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0:
        raise ValueError("track_count and frame_count must be positive")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 64:
        raise ValueError("site_rgba_f32 row count must be in [1, 64]")
    if segment_owner_i32.shape[0] != segment_length_f32.shape[0]:
        raise ValueError("segment owner and length arrays must have matching lengths")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        segment_offsets_i32.detach().cpu(),
        sample_count=track_count * frame_count,
        segment_count=segment_owner_i32.shape[0],
        max_segments_per_sample=max_segments_per_sample,
    )
    config_i32 = torch.tensor(
        [track_count, frame_count, site_rgba_f32.shape[0], segment_owner_i32.shape[0]],
        device=segment_offsets_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.far, config.transmittance_threshold],
        device=segment_offsets_i32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def segment_tape_rgba_depth_replay(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay a fixed-geometry compact segment tape on MPS.

    Tape layout:
    - `segment_offsets_i32`: `[track_count * frame_count + 1]`
    - `segment_owner_i32`: `[segment_count]`
    - `segment_length_f32`: `[segment_count]`
    - `segment_mid_f32`: `[segment_count]`

    The tape fixes geometry and ownership only; RGBA/density remains live.
    """
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    segment_mid_f32 = segment_mid_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _segment_tape_config_tensors(
        segment_offsets_i32=segment_offsets_i32,
        segment_owner_i32=segment_owner_i32,
        segment_length_f32=segment_length_f32,
        segment_mid_f32=segment_mid_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_segments_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "segment_tape_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def segment_tape_mse_vjp_direct_atomic_rgb_only(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site RGBA gradients for a compact segment/owner-run tape."""
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    segment_mid_f32 = segment_mid_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    config_i32, config_f32 = _segment_tape_config_tensors(
        segment_offsets_i32=segment_offsets_i32,
        segment_owner_i32=segment_owner_i32,
        segment_length_f32=segment_length_f32,
        segment_mid_f32=segment_mid_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_segments_per_sample=129,
    )
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "segment_tape_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def segment_tape_nomids_mse_vjp_direct_atomic_rgb_only(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for a compact segment tape without a resident mids tensor."""
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    config_i32, config_f32 = _segment_tape_rgb_mse_config_tensors(
        segment_offsets_i32=segment_offsets_i32,
        segment_owner_i32=segment_owner_i32,
        segment_length_f32=segment_length_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_segments_per_sample=129,
    )
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "segment_tape_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def _endpoint_run_config_tensors(
    *,
    run_offsets_i32: Tensor,
    run_owner_i32: Tensor,
    run_start_f32: Tensor,
    run_end_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_runs_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    _require_mps_tensor(run_offsets_i32, name="run_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(run_owner_i32, name="run_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(run_start_f32, name="run_start_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(run_end_f32, name="run_end_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if run_start_f32.shape != run_owner_i32.shape:
        raise ValueError("run_start_f32 length must match run_owner_i32")
    if run_end_f32.shape != run_owner_i32.shape:
        raise ValueError("run_end_f32 length must match run_owner_i32")
    if track_count <= 0:
        raise ValueError("track_count must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if site_rgba_f32.shape[0] <= 0:
        raise ValueError("site_rgba_f32 must contain at least one site")
    if site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint run replay supports at most 64 sites")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        run_offsets_i32.detach().cpu(),
        sample_count=track_count * frame_count,
        segment_count=run_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    config_i32 = torch.tensor(
        [track_count, frame_count, site_rgba_f32.shape[0], run_owner_i32.shape[0]],
        device=run_offsets_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.far, config.transmittance_threshold],
        device=run_offsets_i32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def endpoint_run_rgba_depth_replay(
    run_offsets_i32: Tensor,
    run_owner_i32: Tensor,
    run_start_f32: Tensor,
    run_end_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay same-owner endpoint runs with continuous absorption depth."""
    run_offsets_i32 = run_offsets_i32.contiguous()
    run_owner_i32 = run_owner_i32.contiguous()
    run_start_f32 = run_start_f32.contiguous()
    run_end_f32 = run_end_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_run_config_tensors(
        run_offsets_i32=run_offsets_i32,
        run_owner_i32=run_owner_i32,
        run_start_f32=run_start_f32,
        run_end_f32=run_end_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_run_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_run_vjp_direct_atomic_grad_only(
    run_offsets_i32: Tensor,
    run_owner_i32: Tensor,
    run_start_f32: Tensor,
    run_end_f32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate site RGBA gradients for continuous-depth endpoint runs."""
    run_offsets_i32 = run_offsets_i32.contiguous()
    run_owner_i32 = run_owner_i32.contiguous()
    run_start_f32 = run_start_f32.contiguous()
    run_end_f32 = run_end_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _endpoint_run_config_tensors(
        run_offsets_i32=run_offsets_i32,
        run_owner_i32=run_owner_i32,
        run_start_f32=run_start_f32,
        run_end_f32=run_end_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_run_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def endpoint_run_mse_vjp_direct_atomic_rgb_only(
    run_offsets_i32: Tensor,
    run_owner_i32: Tensor,
    run_start_f32: Tensor,
    run_end_f32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site RGBA gradients for continuous-depth endpoint runs."""
    run_offsets_i32 = run_offsets_i32.contiguous()
    run_owner_i32 = run_owner_i32.contiguous()
    run_start_f32 = run_start_f32.contiguous()
    run_end_f32 = run_end_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    config_i32, config_f32 = _endpoint_run_config_tensors(
        run_offsets_i32=run_offsets_i32,
        run_owner_i32=run_owner_i32,
        run_start_f32=run_start_f32,
        run_end_f32=run_end_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_run_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def _endpoint_delta_replace_config_tensors(
    *,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_start_f32: Tensor,
    base_end_f32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_start_f32: Tensor,
    change_end_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_runs_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_owner_i32, name="base_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_start_f32, name="base_start_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_end_f32, name="base_end_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_owner_i32, name="change_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_start_f32, name="change_start_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(change_end_f32, name="change_end_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if base_start_f32.shape != base_owner_i32.shape or base_end_f32.shape != base_owner_i32.shape:
        raise ValueError("base owner, start, and end arrays must have matching lengths")
    if change_start_f32.shape != change_owner_i32.shape or change_end_f32.shape != change_owner_i32.shape:
        raise ValueError("change owner, start, and end arrays must have matching lengths")
    if track_count <= 0:
        raise ValueError("track_count must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if site_rgba_f32.shape[0] <= 0:
        raise ValueError("site_rgba_f32 must contain at least one site")
    if site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint delta replay supports at most 64 sites")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    config_i32 = torch.tensor(
        [
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_owner_i32.shape[0],
            change_frame_i32.shape[0],
            change_owner_i32.shape[0],
        ],
        device=base_offsets_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.far, config.transmittance_threshold],
        device=base_offsets_i32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def endpoint_delta_replace_rgba_depth_replay(
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_start_f32: Tensor,
    base_end_f32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_start_f32: Tensor,
    change_end_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint rows from first-frame rows plus changed replacement rows."""
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_start_f32 = base_start_f32.contiguous()
    base_end_f32 = base_end_f32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_owner_i32 = change_owner_i32.contiguous()
    change_start_f32 = change_start_f32.contiguous()
    change_end_f32 = change_end_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_delta_replace_config_tensors(
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_start_f32=base_start_f32,
        base_end_f32=base_end_f32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        change_offsets_i32=change_offsets_i32,
        change_owner_i32=change_owner_i32,
        change_start_f32=change_start_f32,
        change_end_f32=change_end_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_delta_replace_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        base_offsets_i32,
        base_owner_i32,
        base_start_f32,
        base_end_f32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_start_f32,
        change_end_f32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_delta_replace_vjp_direct_atomic_grad_only(
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_start_f32: Tensor,
    base_end_f32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_start_f32: Tensor,
    change_end_f32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate site RGBA gradients for endpoint replacement-delta replay."""
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_start_f32 = base_start_f32.contiguous()
    base_end_f32 = base_end_f32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_owner_i32 = change_owner_i32.contiguous()
    change_start_f32 = change_start_f32.contiguous()
    change_end_f32 = change_end_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _endpoint_delta_replace_config_tensors(
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_start_f32=base_start_f32,
        base_end_f32=base_end_f32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        change_offsets_i32=change_offsets_i32,
        change_owner_i32=change_owner_i32,
        change_start_f32=change_start_f32,
        change_end_f32=change_end_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_delta_replace_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        base_offsets_i32,
        base_owner_i32,
        base_start_f32,
        base_end_f32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_start_f32,
        change_end_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def _endpoint_record_delta_config_tensors(
    *,
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_left_i32: Tensor,
    change_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_runs_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_owner_i32, name="base_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_left_i32, name="base_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_right_i32, name="base_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_owner_i32, name="change_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_left_i32, name="change_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_right_i32, name="change_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if rays_f32.device.type != "mps" or rays_f32.dtype != torch.float32:
        raise ValueError("rays_f32 must be float32 on MPS")
    if rays_f32.shape != (track_count, frame_count, 6):
        raise ValueError("rays_f32 must have shape [track_count, frame_count, 6]")
    if not rays_f32.is_contiguous():
        raise ValueError("rays_f32 must be contiguous")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_left_i32.shape != base_owner_i32.shape or base_right_i32.shape != base_owner_i32.shape:
        raise ValueError("base owner, left, and right arrays must have matching lengths")
    if change_left_i32.shape != change_owner_i32.shape or change_right_i32.shape != change_owner_i32.shape:
        raise ValueError("change owner, left, and right arrays must have matching lengths")
    if track_count <= 0:
        raise ValueError("track_count must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if boundary_f32.shape[0] <= 0:
        raise ValueError("boundary_f32 must contain at least one boundary")
    if boundary_f32.shape[0] > 32767:
        raise ValueError("endpoint record delta replay supports at most 32767 boundaries")
    if site_rgba_f32.shape[0] <= 0:
        raise ValueError("site_rgba_f32 must contain at least one site")
    if site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint record delta replay supports at most 64 sites")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_owner_i32.shape[0],
            change_frame_i32.shape[0],
            change_owner_i32.shape[0],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def endpoint_record_delta_replace_rgba_depth_replay(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_left_i32: Tensor,
    change_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay owner+cut-id endpoint row deltas and recover depths from rays/boundaries."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_owner_i32 = change_owner_i32.contiguous()
    change_left_i32 = change_left_i32.contiguous()
    change_right_i32 = change_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_delta_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        change_offsets_i32=change_offsets_i32,
        change_owner_i32=change_owner_i32,
        change_left_i32=change_left_i32,
        change_right_i32=change_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_left_i32,
        change_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_vjp_direct_atomic_grad_only(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_left_i32: Tensor,
    change_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate site RGBA gradients for owner+cut-id endpoint row deltas."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_owner_i32 = change_owner_i32.contiguous()
    change_left_i32 = change_left_i32.contiguous()
    change_right_i32 = change_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_delta_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        change_offsets_i32=change_offsets_i32,
        change_owner_i32=change_owner_i32,
        change_left_i32=change_left_i32,
        change_right_i32=change_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_left_i32,
        change_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_owner_i32: Tensor,
    change_left_i32: Tensor,
    change_right_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace endpoint records with coeff16 cut depths."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_owner_i32 = change_owner_i32.contiguous()
    change_left_i32 = change_left_i32.contiguous()
    change_right_i32 = change_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_owner_i32, name="base_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_left_i32, name="base_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_right_i32, name="base_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_owner_i32, name="change_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_left_i32, name="change_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_right_i32, name="change_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32767:
        raise ValueError("endpoint record delta replace coeff16 replay supports boundary count <= 32767")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_left_i32.shape != base_owner_i32.shape or base_right_i32.shape != base_owner_i32.shape:
        raise ValueError("base owner, left, and right arrays must have matching lengths")
    if change_left_i32.shape != change_owner_i32.shape or change_right_i32.shape != change_owner_i32.shape:
        raise ValueError("change owner, left, and right arrays must have matching lengths")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint record delta replace coeff16 replay supports site count in [1, 64]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_owner_i32.shape[0],
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_owner_i32.shape[0],
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_owner_i32.shape[0],
            change_frame_i32.shape[0],
            change_owner_i32.shape[0],
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_owner_i32,
        change_left_i32,
        change_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace endpoint records with coeff16 and i16x3 records."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError("endpoint record delta replace coeff16 i16x3 replay supports boundary count <= 32765")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 3 != 0 or change_record_i16.numel() % 3 != 0:
        raise ValueError("i16x3 record tensors must have length divisible by 3")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError("endpoint record delta replace coeff16 i16x3 replay supports site count in [1, 32767]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 3
    change_record_count = change_record_i16.numel() // 3
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace records with 16-frame threadgroup chunks."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError("endpoint record delta replace coeff16 i16x3 framegroup16 replay supports boundary count <= 32765")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 3 != 0 or change_record_i16.numel() % 3 != 0:
        raise ValueError("i16x3 record tensors must have length divisible by 3")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError("endpoint record delta replace coeff16 i16x3 framegroup16 replay supports site count in [1, 32767]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 3
    change_record_count = change_record_i16.numel() // 3
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    track_chunk_owner_offsets_i32: Tensor,
    track_chunk_owner_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP with a sidecar owner-list framegroup reduction."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    track_chunk_owner_offsets_i32 = track_chunk_owner_offsets_i32.contiguous()
    track_chunk_owner_i16 = track_chunk_owner_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(
        track_chunk_owner_offsets_i32,
        name="track_chunk_owner_offsets_i32",
        dtype=torch.int32,
        cols=None,
    )
    _require_mps_tensor(track_chunk_owner_i16, name="track_chunk_owner_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError(
            "endpoint record delta replace coeff16 i16x3 ownerreduce framegroup16 replay supports boundary count <= 32765"
        )
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 3 != 0 or change_record_i16.numel() % 3 != 0:
        raise ValueError("i16x3 record tensors must have length divisible by 3")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError("ownerreduce framegroup16 replay supports site count in [1, 32767]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 3
    change_record_count = change_record_i16.numel() // 3
    owner_list_count = int(track_chunk_owner_i16.shape[0])
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_owner_offsets_i32.detach().cpu(),
        sample_count=track_count * chunk_count,
        segment_count=owner_list_count,
        max_segments_per_sample=None,
    )
    owner_ids_cpu = track_chunk_owner_i16.detach().cpu()
    if owner_ids_cpu.numel() > 0:
        if int(owner_ids_cpu.min().item()) < 0 or int(owner_ids_cpu.max().item()) >= int(site_rgba_f32.shape[0]):
            raise ValueError("track_chunk_owner_i16 ids must be in [0, site_count)")
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
            owner_list_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_ownerreduce_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        track_chunk_owner_offsets_i32,
        track_chunk_owner_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace records with packed int32 rows and 32-frame chunks."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed framegroup16 delta replace coeff16 replay supports boundary count <= 4093")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed framegroup16 delta replace coeff16 replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_packed_framegroup16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace records with packed int32 rows and scalar sample launches."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed scalar delta replace coeff16 replay supports boundary count <= 4093")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed scalar delta replace coeff16 replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute packed framegroup16 RGB MSE/VJP while recomputing reverse replay terms."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed framegroup16 recompute delta replace coeff16 replay supports boundary count <= 4093")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed framegroup16 recompute delta replace coeff16 replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i16: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i16: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i16: Tensor,
    change_offsets_i16: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute packed framegroup16 RGB MSE/VJP from factorized boundary/ray coefficients."""
    boundary_f32 = boundary_f32.contiguous()
    track_ray_coeff_f32 = track_ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i16 = base_offsets_i16.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i16 = track_change_offsets_i16.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i16 = change_frame_i16.contiguous()
    change_offsets_i16 = change_offsets_i16.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(track_ray_coeff_f32, name="track_ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i16, name="base_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i16, name="track_change_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i16, name="change_frame_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_offsets_i16, name="change_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed framegroup16 factorized delta replace replay supports boundary count <= 4093")
    if boundary_f32.shape[0] != boundary_count:
        raise ValueError("boundary_f32 row count must match boundary_count")
    if track_ray_coeff_f32.shape[0] != track_count:
        raise ValueError("track_ray_coeff_f32 row count must match track_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed framegroup16 factorized delta replace replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 31) // 32
    if change_frame_i16.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i16.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i16.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i16.detach().cpu(),
        sample_count=change_frame_i16.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i16.shape[0],
            change_record_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        track_ray_coeff_f32,
        frame_t_f32,
        base_offsets_i16,
        base_record_i32,
        track_change_offsets_i16,
        track_chunk_change_offsets_i16,
        change_frame_i16,
        change_offsets_i16,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i16: Tensor,
    base_record_i32: Tensor,
    frame_change_index_i16: Tensor,
    change_offsets_i16: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute factorized RGB MSE/VJP using one selected sparse-change index per track/frame."""
    boundary_f32 = boundary_f32.contiguous()
    track_ray_coeff_f32 = track_ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i16 = base_offsets_i16.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    frame_change_index_i16 = frame_change_index_i16.contiguous()
    change_offsets_i16 = change_offsets_i16.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(track_ray_coeff_f32, name="track_ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i16, name="base_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(frame_change_index_i16, name="frame_change_index_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_offsets_i16, name="change_offsets_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("factorized frame-select replay supports boundary count <= 4093")
    if boundary_f32.shape[0] != boundary_count:
        raise ValueError("boundary_f32 row count must match boundary_count")
    if track_ray_coeff_f32.shape[0] != track_count:
        raise ValueError("track_ray_coeff_f32 row count must match track_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("factorized frame-select replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_count = change_offsets_i16.numel() - 1
    change_record_count = change_record_i32.numel()
    if change_count > 32767:
        raise ValueError("frame-select map stores sparse change indices as int16 and requires change count <= 32767")
    if frame_change_index_i16.shape[0] != track_count * max(frame_count - 1, 0):
        raise ValueError("frame_change_index_i16 must have shape [track_count * (frame_count - 1)]")
    frame_select_cpu = frame_change_index_i16.detach().cpu()
    if frame_select_cpu.numel():
        min_index = int(frame_select_cpu.min().item())
        max_index = int(frame_select_cpu.max().item())
        if min_index < -1 or max_index >= change_count:
            raise ValueError(
                f"frame_change_index_i16 values must be in [-1, change_count), got min={min_index} max={max_index}"
            )
    _validate_segment_tape_offsets_cpu(
        base_offsets_i16.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i16.detach().cpu(),
        sample_count=change_count,
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_count,
            change_record_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        track_ray_coeff_f32,
        frame_t_f32,
        base_offsets_i16,
        base_record_i32,
        frame_change_index_i16,
        change_offsets_i16,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    track_ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    track_frame_mask_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute factorized RGB MSE/VJP using per-track frame bitmasks and rank selection."""
    boundary_f32 = boundary_f32.contiguous()
    track_ray_coeff_f32 = track_ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_frame_mask_i32 = track_frame_mask_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(track_ray_coeff_f32, name="track_ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_frame_mask_i32, name="track_frame_mask_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if frame_count > 32:
        raise ValueError("factorized frame-bitmask replay requires frame_count <= 32")
    if boundary_count > 4093:
        raise ValueError("factorized frame-bitmask replay supports boundary count <= 4093")
    if boundary_f32.shape[0] != boundary_count:
        raise ValueError("boundary_f32 row count must match boundary_count")
    if track_ray_coeff_f32.shape[0] != track_count:
        raise ValueError("track_ray_coeff_f32 row count must match track_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if track_change_offsets_i32.shape[0] != track_count + 1:
        raise ValueError("track_change_offsets_i32 must have shape [track_count + 1]")
    if track_frame_mask_i32.shape[0] != track_count:
        raise ValueError("track_frame_mask_i32 must have shape [track_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("factorized frame-bitmask replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    if change_offsets_i32.numel() < 1:
        raise ValueError("change_offsets_i32 must contain at least one offset")
    change_count = change_offsets_i32.numel() - 1
    change_record_count = change_record_i32.numel()
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_count,
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_count,
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    masks_cpu = track_frame_mask_i32.detach().cpu()
    if masks_cpu.numel():
        allowed_mask = 0xFFFFFFFE if frame_count == 32 else ((1 << frame_count) - 1) & ~1
        track_offsets = track_change_offsets_i32.detach().cpu().reshape(-1).tolist()
        for track_id, raw_mask in enumerate(masks_cpu.reshape(-1).tolist()):
            mask = int(raw_mask) & 0xFFFFFFFF
            if mask & ~allowed_mask:
                raise ValueError("track_frame_mask_i32 contains bits outside [1, frame_count)")
            expected_changes = int(track_offsets[track_id + 1]) - int(track_offsets[track_id])
            if mask.bit_count() != expected_changes:
                raise ValueError("track_frame_mask_i32 popcount must match per-track change count")
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_count,
            change_record_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_factorized_framebitmask_recompute_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        track_ray_coeff_f32,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_frame_mask_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for packed 32-frame chunks capped to 16 replay segments."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed framegroup16 smallrun16 delta replace coeff16 replay supports boundary count <= 4093")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed framegroup16 smallrun16 delta replace coeff16 replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=16,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=16,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i32: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for materialized packed int32 delta-replace rows with 16-frame chunks."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i32 = base_record_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i32 = change_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i32, name="base_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i32, name="change_record_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 4093:
        raise ValueError("packed materialized framegroup16 delta replace coeff16 replay supports boundary count <= 4093")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 256:
        raise ValueError("packed materialized framegroup16 delta replace coeff16 replay supports site count in [1, 256]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i32.numel()
    change_record_count = change_record_i32.numel()
    chunk_count = (frame_count + 15) // 16
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    _validate_packed_endpoint_delta_records_cpu(
        base_record_i32=base_record_i32,
        change_record_i32=change_record_i32,
        site_count=site_rgba_f32.shape[0],
        boundary_count=boundary_count,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i32,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for materialized i16x3 delta-replace rows with 16-frame chunks."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError(
            "endpoint record delta replace coeff16 i16x3 materialized framegroup16 replay supports boundary count <= 32765"
        )
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 3 != 0 or change_record_i16.numel() % 3 != 0:
        raise ValueError("i16x3 record tensors must have length divisible by 3")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError(
            "endpoint record delta replace coeff16 i16x3 materialized framegroup16 replay supports site count in [1, 32767]"
        )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 3
    change_record_count = change_record_i16.numel() // 3
    chunk_count = (frame_count + 15) // 16
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace records with 32-frame chunks and padded i16x4 rows."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    track_chunk_change_offsets_i16 = track_chunk_change_offsets_i16.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(
        track_chunk_change_offsets_i16,
        name="track_chunk_change_offsets_i16",
        dtype=torch.int16,
        cols=None,
    )
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError("endpoint record delta replace coeff16 i16x4 framegroup16 replay supports boundary count <= 32765")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 4 != 0 or change_record_i16.numel() % 4 != 0:
        raise ValueError("i16x4 record tensors must have length divisible by 4")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError("endpoint record delta replace coeff16 i16x4 framegroup16 replay supports site count in [1, 32767]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 4
    change_record_count = change_record_i16.numel() // 4
    chunk_count = (frame_count + 31) // 32
    if change_frame_i32.shape[0] > 32767:
        raise ValueError("framegroup16 chunk-start offsets use int16 and require change count <= 32767")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        track_chunk_change_offsets_i16.detach().cpu(),
        sample_count=track_count * (chunk_count + 1) - 1,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x4_framegroup16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


class _EndpointRecordDeltaReplaceCoeff16I16x3Framegroup16MSEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        coeff_f16: Tensor,
        frame_t_f32: Tensor,
        base_offsets_i32: Tensor,
        base_record_i16: Tensor,
        track_change_offsets_i32: Tensor,
        track_chunk_change_offsets_i16: Tensor,
        change_frame_i32: Tensor,
        change_offsets_i32: Tensor,
        change_record_i16: Tensor,
        site_rgba_f32: Tensor,
        target_rgb_f32: Tensor,
        config: RealRayReplayConfig,
        track_count: int,
        frame_count: int,
        boundary_count: int,
    ) -> Tensor:
        loss, grad_site_rgba = endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_vjp_direct_atomic_rgb_only(
            coeff_f16,
            frame_t_f32,
            base_offsets_i32,
            base_record_i16,
            track_change_offsets_i32,
            track_chunk_change_offsets_i16,
            change_frame_i32,
            change_offsets_i32,
            change_record_i16,
            site_rgba_f32,
            target_rgb_f32,
            config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=boundary_count,
        )
        ctx.save_for_backward(grad_site_rgba)
        return loss.reshape(())

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_loss: Tensor) -> tuple[Any, ...]:
        (grad_site_rgba,) = ctx.saved_tensors
        grad_scale = grad_loss.to(device=grad_site_rgba.device, dtype=grad_site_rgba.dtype).reshape(())
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_site_rgba * grad_scale,
            None,
            None,
            None,
            None,
            None,
        )


def endpoint_record_delta_replace_coeff16_i16x3_framegroup16_mse_autograd(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    track_chunk_change_offsets_i16: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> Tensor:
    """Return fused RGB MSE with autograd for frozen-geometry site-RGBA training."""
    return _EndpointRecordDeltaReplaceCoeff16I16x3Framegroup16MSEFunction.apply(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        track_chunk_change_offsets_i16,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config,
        int(track_count),
        int(frame_count),
        int(boundary_count),
    )


def endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_record_i16: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    change_offsets_i32: Tensor,
    change_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for delta-replace endpoint records with coeff16 and padded i16x4 records."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_record_i16 = base_record_i16.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    change_offsets_i32 = change_offsets_i32.contiguous()
    change_record_i16 = change_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_record_i16, name="base_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_offsets_i32, name="change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_record_i16, name="change_record_i16", dtype=torch.int16, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32765:
        raise ValueError("endpoint record delta replace coeff16 i16x4 replay supports boundary count <= 32765")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_record_i16.numel() % 4 != 0 or change_record_i16.numel() % 4 != 0:
        raise ValueError("i16x4 record tensors must have length divisible by 4")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 32767:
        raise ValueError("endpoint record delta replace coeff16 i16x4 replay supports site count in [1, 32767]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    base_record_count = base_record_i16.numel() // 4
    change_record_count = change_record_i16.numel() // 4
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_record_count,
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        change_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=change_record_count,
        max_segments_per_sample=129,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_record_count,
            change_frame_i32.shape[0],
            change_record_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_record_i16,
        track_change_offsets_i32,
        change_frame_i32,
        change_offsets_i32,
        change_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def _endpoint_record_edit_config_tensors(
    *,
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    track_count: int,
    frame_count: int,
    max_runs_per_sample: int | None,
) -> tuple[Tensor, Tensor]:
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_owner_i32, name="base_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_left_i32, name="base_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_right_i32, name="base_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_offsets_i32, name="op_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_type_i32, name="op_type_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_pos_i32, name="op_pos_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_owner_i32, name="op_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_left_i32, name="op_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_right_i32, name="op_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if rays_f32.device.type != "mps" or rays_f32.dtype != torch.float32:
        raise ValueError("rays_f32 must be float32 on MPS")
    if rays_f32.shape != (track_count, frame_count, 6):
        raise ValueError("rays_f32 must have shape [track_count, frame_count, 6]")
    if not rays_f32.is_contiguous():
        raise ValueError("rays_f32 must be contiguous")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_left_i32.shape != base_owner_i32.shape or base_right_i32.shape != base_owner_i32.shape:
        raise ValueError("base owner, left, and right arrays must have matching lengths")
    if op_pos_i32.shape != op_type_i32.shape:
        raise ValueError("op_pos_i32 length must match op_type_i32")
    if op_owner_i32.shape != op_type_i32.shape or op_left_i32.shape != op_type_i32.shape:
        raise ValueError("op owner/left length must match op_type_i32")
    if op_right_i32.shape != op_type_i32.shape:
        raise ValueError("op_right_i32 length must match op_type_i32")
    if track_count <= 0:
        raise ValueError("track_count must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if boundary_f32.shape[0] <= 0 or boundary_f32.shape[0] > 32767:
        raise ValueError("endpoint record edit replay supports boundary count in [1, 32767]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint record edit replay supports site count in [1, 64]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_owner_i32.shape[0],
        max_segments_per_sample=max_runs_per_sample,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        op_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=op_type_i32.shape[0],
        max_segments_per_sample=None,
    )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return config_i32, config_f32


def endpoint_record_edit_rgba_depth_replay(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint owner+cut-id edit streams and recover depths from rays/boundaries."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block4_rgba_depth_replay(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint edit streams from block anchor rows and recover depths from rays/boundaries."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block4_rgba_depth_replay requires block_size > 0")
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    if rays_f32.ndim != 3 or rays_f32.shape[2] != 6:
        raise ValueError("rays_f32 must have shape [track_count, frame_count, 6]")
    if rays_f32.shape[0] != track_count or rays_f32.shape[1] != frame_count:
        raise ValueError("rays_f32 shape must match track_count/frame_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block4_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff_rgba_depth_replay(
    coeff_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint edit streams using precomputed per-track boundary depth coefficients."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff_rgba_depth_replay requires block_size > 0")
    coeff_f32 = coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    if coeff_f32.ndim != 2 or coeff_f32.shape[1] != 4:
        raise ValueError("coeff_f32 must have shape [track_count * boundary_count, 4]")
    if coeff_f32.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f32 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff_rgb_replay(
    coeff_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> Tensor:
    """Replay coefficient-cached block edits and return RGB only for train-path timing."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff_rgb_replay requires block_size > 0")
    coeff_f32 = coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    if coeff_f32.ndim != 2 or coeff_f32.shape[1] != 4:
        raise ValueError("coeff_f32 must have shape [track_count * boundary_count, 4]")
    if coeff_f32.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f32 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff_rgb_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_rgba_depth_replay(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint edit streams using float16 per-track boundary depth coefficients."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff16_rgba_depth_replay requires block_size > 0")
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_rgba_depth_replay"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_rgba_depth_replay_trackloop(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint edit streams by walking each track through frames in one Metal thread."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_rgba_depth_replay_trackloop"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_rgba_depth_replay_framegroup16(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint edit streams with one threadgroup per track and up to 16 frame lanes."""
    if frame_count > 16:
        raise ValueError("endpoint_record_edit_rgba_depth_replay_framegroup16 supports at most 16 frames")
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=None,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_rgba_depth_replay_framegroup16"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_vjp_direct_atomic_grad_only(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate site RGBA gradients for endpoint owner+cut-id edit streams."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate endpoint edit gradients for RGB-only losses."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_mse_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site gradients for compact endpoint edit rows."""
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    config_i32, config_f32 = _endpoint_record_edit_config_tensors(
        boundary_f32=boundary_f32,
        rays_f32=rays_f32,
        frame_t_f32=frame_t_f32,
        base_offsets_i32=base_offsets_i32,
        base_owner_i32=base_owner_i32,
        base_left_i32=base_left_i32,
        base_right_i32=base_right_i32,
        track_change_offsets_i32=track_change_offsets_i32,
        change_frame_i32=change_frame_i32,
        op_offsets_i32=op_offsets_i32,
        op_type_i32=op_type_i32,
        op_pos_i32=op_pos_i32,
        op_owner_i32=op_owner_i32,
        op_left_i32=op_left_i32,
        op_right_i32=op_right_i32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_runs_per_sample=129,
    )
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP for raw endpoint edits with coeff16 cut depths."""
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    base_offsets_i32 = base_offsets_i32.contiguous()
    base_owner_i32 = base_owner_i32.contiguous()
    base_left_i32 = base_left_i32.contiguous()
    base_right_i32 = base_right_i32.contiguous()
    track_change_offsets_i32 = track_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(coeff_f16, name="coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    _require_mps_tensor(base_offsets_i32, name="base_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_owner_i32, name="base_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_left_i32, name="base_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(base_right_i32, name="base_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(track_change_offsets_i32, name="track_change_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(change_frame_i32, name="change_frame_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_offsets_i32, name="op_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_type_i32, name="op_type_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_pos_i32, name="op_pos_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_owner_i32, name="op_owner_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_left_i32, name="op_left_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(op_right_i32, name="op_right_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    if track_count <= 0 or frame_count <= 0 or boundary_count <= 0:
        raise ValueError("track_count, frame_count, and boundary_count must be positive")
    if boundary_count > 32767:
        raise ValueError("endpoint record edit coeff16 replay supports boundary count <= 32767")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must be track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 must have shape [frame_count]")
    if base_left_i32.shape != base_owner_i32.shape or base_right_i32.shape != base_owner_i32.shape:
        raise ValueError("base owner, left, and right arrays must have matching lengths")
    if op_pos_i32.shape != op_type_i32.shape:
        raise ValueError("op_pos_i32 length must match op_type_i32")
    if op_owner_i32.shape != op_type_i32.shape or op_left_i32.shape != op_type_i32.shape:
        raise ValueError("op owner/left length must match op_type_i32")
    if op_right_i32.shape != op_type_i32.shape:
        raise ValueError("op_right_i32 length must match op_type_i32")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if site_rgba_f32.shape[0] <= 0 or site_rgba_f32.shape[0] > 64:
        raise ValueError("endpoint record edit coeff16 replay supports site count in [1, 64]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    _validate_segment_tape_offsets_cpu(
        base_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=base_owner_i32.shape[0],
        max_segments_per_sample=129,
    )
    _validate_segment_tape_offsets_cpu(
        track_change_offsets_i32.detach().cpu(),
        sample_count=track_count,
        segment_count=change_frame_i32.shape[0],
        max_segments_per_sample=None,
    )
    _validate_segment_tape_offsets_cpu(
        op_offsets_i32.detach().cpu(),
        sample_count=change_frame_i32.shape[0],
        segment_count=op_type_i32.shape[0],
        max_segments_per_sample=None,
    )
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            base_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block4_vjp_direct_atomic_rgb_only(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    block_size: int = 4,
) -> Tensor:
    """Accumulate RGB-only site gradients from block-anchored endpoint edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block4_vjp_direct_atomic_rgb_only requires block_size > 0")
    boundary_f32 = boundary_f32.contiguous()
    rays_f32 = rays_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    if rays_f32.ndim != 3 or rays_f32.shape[2] != 6:
        raise ValueError("rays_f32 must have shape [track_count, frame_count, 6]")
    if rays_f32.shape[0] != track_count or rays_f32.shape[1] != frame_count:
        raise ValueError("rays_f32 shape must match track_count/frame_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block4_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
    coeff_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> Tensor:
    """Accumulate RGB-only site gradients from coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f32 = coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    if coeff_f32.ndim != 2 or coeff_f32.shape[1] != 4:
        raise ValueError("coeff_f32 must have shape [track_count * boundary_count, 4]")
    if coeff_f32.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f32 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only(
    coeff_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site gradients for coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f32 = coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    if coeff_f32.ndim != 2 or coeff_f32.shape[1] != 4:
        raise ValueError("coeff_f32 must have shape [track_count * boundary_count, 4]")
    if coeff_f32.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f32 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f32.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site gradients for float16 coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_record_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_record_i32: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP from packed float16 coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_record_i32 = anchor_record_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_record_i32 = op_record_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if anchor_record_i32.ndim != 1 or anchor_record_i32.dtype != torch.int32:
        raise ValueError("anchor_record_i32 must be int32 with shape [anchor_record_count]")
    if op_record_i32.ndim != 1 or op_record_i32.dtype != torch.int32:
        raise ValueError("op_record_i32 must be int32 with shape [op_count]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_record_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_packed_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_record_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_record_i32,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i16: Tensor,
    anchor_left_i16: Tensor,
    anchor_right_i16: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i16: Tensor,
    op_left_i16: Tensor,
    op_right_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP from int16-record float16 coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i16 = anchor_owner_i16.contiguous()
    anchor_left_i16 = anchor_left_i16.contiguous()
    anchor_right_i16 = anchor_right_i16.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i16 = op_owner_i16.contiguous()
    op_left_i16 = op_left_i16.contiguous()
    op_right_i16 = op_right_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    for name, tensor in (
        ("anchor_owner_i16", anchor_owner_i16),
        ("anchor_left_i16", anchor_left_i16),
        ("anchor_right_i16", anchor_right_i16),
        ("op_owner_i16", op_owner_i16),
        ("op_left_i16", op_left_i16),
        ("op_right_i16", op_right_i16),
    ):
        if tensor.ndim != 1 or tensor.dtype != torch.int16:
            raise ValueError(f"{name} must be int16 with shape [N]")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i16.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_i16_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i16,
        anchor_left_i16,
        anchor_right_i16,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i16,
        op_left_i16,
        op_right_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_record_i16: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_record_i16: Tensor,
    site_rgba_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE/VJP from interleaved int16 coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_record_i16 = anchor_record_i16.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_record_i16 = op_record_i16.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if anchor_record_i16.ndim != 1 or anchor_record_i16.dtype != torch.int16:
        raise ValueError("anchor_record_i16 must be int16 with shape [anchor_record_count * 3]")
    if op_record_i16.ndim != 1 or op_record_i16.dtype != torch.int16:
        raise ValueError("op_record_i16 must be int16 with shape [op_count * 3]")
    if anchor_record_i16.numel() % 3 != 0:
        raise ValueError("anchor_record_i16 length must be divisible by 3")
    if op_record_i16.numel() % 3 != 0:
        raise ValueError("op_record_i16 length must be divisible by 3")
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_record_i16.numel() // 3,
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_i16x3_mse_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_record_i16,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_record_i16,
        site_rgba_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only(
    coeff_f16: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> Tensor:
    """Accumulate RGB-only site gradients from float16 coefficient-cached block edit rows."""
    if block_size <= 0:
        raise ValueError("endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only requires block_size > 0")
    coeff_f16 = coeff_f16.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    anchor_offsets_i32 = anchor_offsets_i32.contiguous()
    anchor_owner_i32 = anchor_owner_i32.contiguous()
    anchor_left_i32 = anchor_left_i32.contiguous()
    anchor_right_i32 = anchor_right_i32.contiguous()
    track_block_change_offsets_i32 = track_block_change_offsets_i32.contiguous()
    change_frame_i32 = change_frame_i32.contiguous()
    op_offsets_i32 = op_offsets_i32.contiguous()
    op_type_i32 = op_type_i32.contiguous()
    op_pos_i32 = op_pos_i32.contiguous()
    op_owner_i32 = op_owner_i32.contiguous()
    op_left_i32 = op_left_i32.contiguous()
    op_right_i32 = op_right_i32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    if coeff_f16.dtype != torch.float16:
        raise ValueError("coeff_f16 must be float16")
    if coeff_f16.ndim != 2 or coeff_f16.shape[1] != 4:
        raise ValueError("coeff_f16 must have shape [track_count * boundary_count, 4]")
    if coeff_f16.shape[0] != track_count * boundary_count:
        raise ValueError("coeff_f16 row count must match track_count * boundary_count")
    if frame_t_f32.shape[0] != frame_count:
        raise ValueError("frame_t_f32 length must match frame_count")
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    block_count = (frame_count + block_size - 1) // block_size
    config_i32 = torch.tensor(
        [
            boundary_count,
            track_count,
            frame_count,
            site_rgba_f32.shape[0],
            anchor_owner_i32.shape[0],
            change_frame_i32.shape[0],
            op_type_i32.shape[0],
            block_size,
            block_count,
        ],
        device=coeff_f16.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=coeff_f16.device,
        dtype=torch.float32,
    )
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "endpoint_record_edit_block_coeff16_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        coeff_f16,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        grad_rgb_f32,
        config_i32,
        config_f32,
    )


class _EndpointRecordEditRGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        boundary_f32: Tensor,
        rays_f32: Tensor,
        frame_t_f32: Tensor,
        base_offsets_i32: Tensor,
        base_owner_i32: Tensor,
        base_left_i32: Tensor,
        base_right_i32: Tensor,
        track_change_offsets_i32: Tensor,
        change_frame_i32: Tensor,
        op_offsets_i32: Tensor,
        op_type_i32: Tensor,
        op_pos_i32: Tensor,
        op_owner_i32: Tensor,
        op_left_i32: Tensor,
        op_right_i32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = endpoint_record_edit_rgba_depth_replay(
            boundary_f32,
            rays_f32,
            frame_t_f32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
        )
        ctx.save_for_backward(
            boundary_f32,
            rays_f32,
            frame_t_f32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            boundary_f32,
            rays_f32,
            frame_t_f32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        grad_alpha_is_none = grad_alpha is None
        grad_depth_is_none = grad_depth is None
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha_is_none and grad_depth_is_none:
            grad_site_rgba = endpoint_record_edit_vjp_direct_atomic_rgb_only(
                boundary_f32,
                rays_f32,
                frame_t_f32,
                base_offsets_i32,
                base_owner_i32,
                base_left_i32,
                base_right_i32,
                track_change_offsets_i32,
                change_frame_i32,
                op_offsets_i32,
                op_type_i32,
                op_pos_i32,
                op_owner_i32,
                op_left_i32,
                op_right_i32,
                site_rgba_f32,
                grad_rgb,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        else:
            grad_site_rgba = endpoint_record_edit_vjp_direct_atomic_grad_only(
                boundary_f32,
                rays_f32,
                frame_t_f32,
                base_offsets_i32,
                base_owner_i32,
                base_left_i32,
                base_right_i32,
                track_change_offsets_i32,
                change_frame_i32,
                op_offsets_i32,
                op_type_i32,
                op_pos_i32,
                op_owner_i32,
                op_left_i32,
                op_right_i32,
                site_rgba_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_site_rgba,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def endpoint_record_edit_rgba_depth_autograd(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint owner+cut-id edit streams with frozen-geometry site-RGBA autograd."""
    return _EndpointRecordEditRGBADepthReplayFunction.apply(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


class _EndpointRecordEditBlock4RGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        boundary_f32: Tensor,
        rays_f32: Tensor,
        frame_t_f32: Tensor,
        anchor_offsets_i32: Tensor,
        anchor_owner_i32: Tensor,
        anchor_left_i32: Tensor,
        anchor_right_i32: Tensor,
        track_block_change_offsets_i32: Tensor,
        block_change_frame_i32: Tensor,
        block_op_offsets_i32: Tensor,
        block_op_type_i32: Tensor,
        block_op_pos_i32: Tensor,
        block_op_owner_i32: Tensor,
        block_op_left_i32: Tensor,
        block_op_right_i32: Tensor,
        base_offsets_i32: Tensor,
        base_owner_i32: Tensor,
        base_left_i32: Tensor,
        base_right_i32: Tensor,
        track_change_offsets_i32: Tensor,
        change_frame_i32: Tensor,
        op_offsets_i32: Tensor,
        op_type_i32: Tensor,
        op_pos_i32: Tensor,
        op_owner_i32: Tensor,
        op_left_i32: Tensor,
        op_right_i32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        block_size: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = endpoint_record_edit_block4_rgba_depth_replay(
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
            block_size=int(block_size),
        )
        ctx.save_for_backward(
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        ctx.block_size = int(block_size)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        (
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        block_size = int(ctx.block_size)
        grad_alpha_is_none = grad_alpha is None
        grad_depth_is_none = grad_depth is None
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha_is_none and grad_depth_is_none:
            grad_site_rgba = endpoint_record_edit_block4_vjp_direct_atomic_rgb_only(
                boundary_f32,
                rays_f32,
                frame_t_f32,
                anchor_offsets_i32,
                anchor_owner_i32,
                anchor_left_i32,
                anchor_right_i32,
                track_block_change_offsets_i32,
                block_change_frame_i32,
                block_op_offsets_i32,
                block_op_type_i32,
                block_op_pos_i32,
                block_op_owner_i32,
                block_op_left_i32,
                block_op_right_i32,
                site_rgba_f32,
                grad_rgb,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
                block_size=block_size,
            )
        else:
            grad_site_rgba = endpoint_record_edit_vjp_direct_atomic_grad_only(
                boundary_f32,
                rays_f32,
                frame_t_f32,
                base_offsets_i32,
                base_owner_i32,
                base_left_i32,
                base_right_i32,
                track_change_offsets_i32,
                change_frame_i32,
                op_offsets_i32,
                op_type_i32,
                op_pos_i32,
                op_owner_i32,
                op_left_i32,
                op_right_i32,
                site_rgba_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        return (*([None] * 27), grad_site_rgba, *([None] * 7))


def endpoint_record_edit_block4_rgba_depth_autograd(
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    block_change_frame_i32: Tensor,
    block_op_offsets_i32: Tensor,
    block_op_type_i32: Tensor,
    block_op_pos_i32: Tensor,
    block_op_owner_i32: Tensor,
    block_op_left_i32: Tensor,
    block_op_right_i32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay block4 endpoint edits with frozen-geometry site-RGBA autograd.

    Forward and RGB-only backward use block-anchored replay shaders. Full
    alpha/depth backward still falls back to the exact endpoint-record edit VJP.
    """
    return _EndpointRecordEditBlock4RGBADepthReplayFunction.apply(
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        block_change_frame_i32,
        block_op_offsets_i32,
        block_op_type_i32,
        block_op_pos_i32,
        block_op_owner_i32,
        block_op_left_i32,
        block_op_right_i32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        int(block_size),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


class _EndpointRecordEditBlockCoeffRGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        coeff_f32: Tensor,
        boundary_f32: Tensor,
        rays_f32: Tensor,
        frame_t_f32: Tensor,
        anchor_offsets_i32: Tensor,
        anchor_owner_i32: Tensor,
        anchor_left_i32: Tensor,
        anchor_right_i32: Tensor,
        track_block_change_offsets_i32: Tensor,
        block_change_frame_i32: Tensor,
        block_op_offsets_i32: Tensor,
        block_op_type_i32: Tensor,
        block_op_pos_i32: Tensor,
        block_op_owner_i32: Tensor,
        block_op_left_i32: Tensor,
        block_op_right_i32: Tensor,
        base_offsets_i32: Tensor,
        base_owner_i32: Tensor,
        base_left_i32: Tensor,
        base_right_i32: Tensor,
        track_change_offsets_i32: Tensor,
        change_frame_i32: Tensor,
        op_offsets_i32: Tensor,
        op_type_i32: Tensor,
        op_pos_i32: Tensor,
        op_owner_i32: Tensor,
        op_left_i32: Tensor,
        op_right_i32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        boundary_count: int,
        block_size: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = endpoint_record_edit_block_coeff_rgba_depth_replay(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
            boundary_count=int(boundary_count),
            block_size=int(block_size),
        )
        ctx.save_for_backward(
            coeff_f32,
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        ctx.boundary_count = int(boundary_count)
        ctx.block_size = int(block_size)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        (
            coeff_f32,
            boundary_f32,
            rays_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            base_offsets_i32,
            base_owner_i32,
            base_left_i32,
            base_right_i32,
            track_change_offsets_i32,
            change_frame_i32,
            op_offsets_i32,
            op_type_i32,
            op_pos_i32,
            op_owner_i32,
            op_left_i32,
            op_right_i32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        boundary_count = int(ctx.boundary_count)
        block_size = int(ctx.block_size)
        grad_alpha_is_none = grad_alpha is None
        grad_depth_is_none = grad_depth is None
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha_is_none and grad_depth_is_none:
            grad_site_rgba = endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
                coeff_f32,
                frame_t_f32,
                anchor_offsets_i32,
                anchor_owner_i32,
                anchor_left_i32,
                anchor_right_i32,
                track_block_change_offsets_i32,
                block_change_frame_i32,
                block_op_offsets_i32,
                block_op_type_i32,
                block_op_pos_i32,
                block_op_owner_i32,
                block_op_left_i32,
                block_op_right_i32,
                site_rgba_f32,
                grad_rgb,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
                boundary_count=boundary_count,
                block_size=block_size,
            )
        else:
            grad_site_rgba = endpoint_record_edit_vjp_direct_atomic_grad_only(
                boundary_f32,
                rays_f32,
                frame_t_f32,
                base_offsets_i32,
                base_owner_i32,
                base_left_i32,
                base_right_i32,
                track_change_offsets_i32,
                change_frame_i32,
                op_offsets_i32,
                op_type_i32,
                op_pos_i32,
                op_owner_i32,
                op_left_i32,
                op_right_i32,
                site_rgba_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        return (*([None] * 28), grad_site_rgba, *([None] * 8))


def endpoint_record_edit_block_coeff_rgba_depth_autograd(
    coeff_f32: Tensor,
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    block_change_frame_i32: Tensor,
    block_op_offsets_i32: Tensor,
    block_op_type_i32: Tensor,
    block_op_pos_i32: Tensor,
    block_op_owner_i32: Tensor,
    block_op_left_i32: Tensor,
    block_op_right_i32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay coefficient-cached block edits with frozen-geometry site-RGBA autograd.

    Forward and RGB-only backward use coefficient-cached block replay. Full
    alpha/depth backward falls back to the exact endpoint-record edit VJP.
    """
    return _EndpointRecordEditBlockCoeffRGBADepthReplayFunction.apply(
        coeff_f32,
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        block_change_frame_i32,
        block_op_offsets_i32,
        block_op_type_i32,
        block_op_pos_i32,
        block_op_owner_i32,
        block_op_left_i32,
        block_op_right_i32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        int(boundary_count),
        int(block_size),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


class _EndpointRecordEditBlockCoeffRGBReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        coeff_f32: Tensor,
        boundary_f32: Tensor,
        rays_f32: Tensor,
        frame_t_f32: Tensor,
        anchor_offsets_i32: Tensor,
        anchor_owner_i32: Tensor,
        anchor_left_i32: Tensor,
        anchor_right_i32: Tensor,
        track_block_change_offsets_i32: Tensor,
        block_change_frame_i32: Tensor,
        block_op_offsets_i32: Tensor,
        block_op_type_i32: Tensor,
        block_op_pos_i32: Tensor,
        block_op_owner_i32: Tensor,
        block_op_left_i32: Tensor,
        block_op_right_i32: Tensor,
        base_offsets_i32: Tensor,
        base_owner_i32: Tensor,
        base_left_i32: Tensor,
        base_right_i32: Tensor,
        track_change_offsets_i32: Tensor,
        change_frame_i32: Tensor,
        op_offsets_i32: Tensor,
        op_type_i32: Tensor,
        op_pos_i32: Tensor,
        op_owner_i32: Tensor,
        op_left_i32: Tensor,
        op_right_i32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        boundary_count: int,
        block_size: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> Tensor:
        del boundary_f32, rays_f32, base_offsets_i32, base_owner_i32, base_left_i32, base_right_i32
        del track_change_offsets_i32, change_frame_i32, op_offsets_i32, op_type_i32, op_pos_i32, op_owner_i32
        del op_left_i32, op_right_i32
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb = endpoint_record_edit_block_coeff_rgb_replay(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
            boundary_count=int(boundary_count),
            block_size=int(block_size),
        )
        ctx.save_for_backward(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        ctx.boundary_count = int(boundary_count)
        ctx.block_size = int(block_size)
        return output_rgb

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
    ) -> tuple[Tensor | None, ...]:
        (
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        grad_site_rgba = endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only(
            coeff_f32,
            frame_t_f32,
            anchor_offsets_i32,
            anchor_owner_i32,
            anchor_left_i32,
            anchor_right_i32,
            track_block_change_offsets_i32,
            block_change_frame_i32,
            block_op_offsets_i32,
            block_op_type_i32,
            block_op_pos_i32,
            block_op_owner_i32,
            block_op_left_i32,
            block_op_right_i32,
            site_rgba_f32,
            grad_rgb,
            ctx.config,
            track_count=track_count,
            frame_count=frame_count,
            boundary_count=int(ctx.boundary_count),
            block_size=int(ctx.block_size),
        )
        return (*([None] * 28), grad_site_rgba, *([None] * 8))


def endpoint_record_edit_block_coeff_rgb_autograd(
    coeff_f32: Tensor,
    boundary_f32: Tensor,
    rays_f32: Tensor,
    frame_t_f32: Tensor,
    anchor_offsets_i32: Tensor,
    anchor_owner_i32: Tensor,
    anchor_left_i32: Tensor,
    anchor_right_i32: Tensor,
    track_block_change_offsets_i32: Tensor,
    block_change_frame_i32: Tensor,
    block_op_offsets_i32: Tensor,
    block_op_type_i32: Tensor,
    block_op_pos_i32: Tensor,
    block_op_owner_i32: Tensor,
    block_op_left_i32: Tensor,
    block_op_right_i32: Tensor,
    base_offsets_i32: Tensor,
    base_owner_i32: Tensor,
    base_left_i32: Tensor,
    base_right_i32: Tensor,
    track_change_offsets_i32: Tensor,
    change_frame_i32: Tensor,
    op_offsets_i32: Tensor,
    op_type_i32: Tensor,
    op_pos_i32: Tensor,
    op_owner_i32: Tensor,
    op_left_i32: Tensor,
    op_right_i32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    boundary_count: int,
    block_size: int = 4,
) -> Tensor:
    """RGB-only coefficient-cached replay with the existing RGB VJP."""
    return _EndpointRecordEditBlockCoeffRGBReplayFunction.apply(
        coeff_f32,
        boundary_f32,
        rays_f32,
        frame_t_f32,
        anchor_offsets_i32,
        anchor_owner_i32,
        anchor_left_i32,
        anchor_right_i32,
        track_block_change_offsets_i32,
        block_change_frame_i32,
        block_op_offsets_i32,
        block_op_type_i32,
        block_op_pos_i32,
        block_op_owner_i32,
        block_op_left_i32,
        block_op_right_i32,
        base_offsets_i32,
        base_owner_i32,
        base_left_i32,
        base_right_i32,
        track_change_offsets_i32,
        change_frame_i32,
        op_offsets_i32,
        op_type_i32,
        op_pos_i32,
        op_owner_i32,
        op_left_i32,
        op_right_i32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        int(boundary_count),
        int(block_size),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


class _EndpointRunRGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        run_offsets_i32: Tensor,
        run_owner_i32: Tensor,
        run_start_f32: Tensor,
        run_end_f32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = endpoint_run_rgba_depth_replay(
            run_offsets_i32,
            run_owner_i32,
            run_start_f32,
            run_end_f32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
        )
        ctx.save_for_backward(
            run_offsets_i32,
            run_owner_i32,
            run_start_f32,
            run_end_f32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, None, Tensor, None, None, None, None, None, None]:
        (
            run_offsets_i32,
            run_owner_i32,
            run_start_f32,
            run_end_f32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        grad_site_rgba = endpoint_run_vjp_direct_atomic_grad_only(
            run_offsets_i32,
            run_owner_i32,
            run_start_f32,
            run_end_f32,
            site_rgba_f32,
            grad_rgb,
            grad_alpha,
            grad_depth,
            ctx.config,
            track_count=track_count,
            frame_count=frame_count,
        )
        return None, None, None, None, grad_site_rgba, None, None, None, None, None, None


def endpoint_run_rgba_depth_autograd(
    run_offsets_i32: Tensor,
    run_owner_i32: Tensor,
    run_start_f32: Tensor,
    run_end_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay endpoint runs with frozen-geometry site-RGBA autograd."""
    return _EndpointRunRGBADepthReplayFunction.apply(
        run_offsets_i32,
        run_owner_i32,
        run_start_f32,
        run_end_f32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


def segment_tape_vjp_direct_atomic_grad_only(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate fixed-geometry compact segment-tape site RGBA gradients."""
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    segment_mid_f32 = segment_mid_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _segment_tape_config_tensors(
        segment_offsets_i32=segment_offsets_i32,
        segment_owner_i32=segment_owner_i32,
        segment_length_f32=segment_length_f32,
        segment_mid_f32=segment_mid_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_segments_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "segment_tape_vjp_direct_atomic_grad_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def segment_tape_vjp_direct_atomic_track(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
) -> Tensor:
    """Accumulate compact segment-tape gradients once per track before atomics."""
    segment_offsets_i32 = segment_offsets_i32.contiguous()
    segment_owner_i32 = segment_owner_i32.contiguous()
    segment_length_f32 = segment_length_f32.contiguous()
    segment_mid_f32 = segment_mid_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    config_i32, config_f32 = _segment_tape_config_tensors(
        segment_offsets_i32=segment_offsets_i32,
        segment_owner_i32=segment_owner_i32,
        segment_length_f32=segment_length_f32,
        segment_mid_f32=segment_mid_f32,
        site_rgba_f32=site_rgba_f32,
        config=config,
        track_count=track_count,
        frame_count=frame_count,
        max_segments_per_sample=129,
    )
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.shape != (track_count, frame_count, 3):
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.shape != (track_count, frame_count):
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.shape != (track_count, frame_count):
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "segment_tape_vjp_direct_atomic_track"
    if not hasattr(ops, op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first.")
    return getattr(ops, op_name)(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


class _SegmentTapeRGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        segment_offsets_i32: Tensor,
        segment_owner_i32: Tensor,
        segment_length_f32: Tensor,
        segment_mid_f32: Tensor,
        site_rgba_f32: Tensor,
        track_count: int,
        frame_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
        use_track_vjp: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = segment_tape_rgba_depth_replay(
            segment_offsets_i32,
            segment_owner_i32,
            segment_length_f32,
            segment_mid_f32,
            site_rgba_f32,
            config,
            track_count=int(track_count),
            frame_count=int(frame_count),
        )
        ctx.save_for_backward(
            segment_offsets_i32,
            segment_owner_i32,
            segment_length_f32,
            segment_mid_f32,
            site_rgba_f32,
        )
        ctx.config = config
        ctx.track_count = int(track_count)
        ctx.frame_count = int(frame_count)
        ctx.use_track_vjp = bool(use_track_vjp)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, None, Tensor, None, None, None, None, None, None, None]:
        (
            segment_offsets_i32,
            segment_owner_i32,
            segment_length_f32,
            segment_mid_f32,
            site_rgba_f32,
        ) = ctx.saved_tensors
        track_count = int(ctx.track_count)
        frame_count = int(ctx.frame_count)
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if bool(ctx.use_track_vjp):
            grad_site_rgba = segment_tape_vjp_direct_atomic_track(
                segment_offsets_i32,
                segment_owner_i32,
                segment_length_f32,
                segment_mid_f32,
                site_rgba_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        else:
            grad_site_rgba = segment_tape_vjp_direct_atomic_grad_only(
                segment_offsets_i32,
                segment_owner_i32,
                segment_length_f32,
                segment_mid_f32,
                site_rgba_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                track_count=track_count,
                frame_count=frame_count,
            )
        return None, None, None, None, grad_site_rgba, None, None, None, None, None, None, None


def segment_tape_rgba_depth_autograd(
    segment_offsets_i32: Tensor,
    segment_owner_i32: Tensor,
    segment_length_f32: Tensor,
    segment_mid_f32: Tensor,
    site_rgba_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    track_count: int,
    frame_count: int,
    vjp_mode: str = "direct_atomic_grad_only",
) -> tuple[Tensor, Tensor, Tensor]:
    """Replay a compact segment tape with frozen-geometry site-RGBA autograd."""
    if vjp_mode not in {"direct_atomic_grad_only", "direct_atomic_track"}:
        raise ValueError("vjp_mode must be 'direct_atomic_grad_only' or 'direct_atomic_track'")
    return _SegmentTapeRGBADepthReplayFunction.apply(
        segment_offsets_i32,
        segment_owner_i32,
        segment_length_f32,
        segment_mid_f32,
        site_rgba_f32,
        int(track_count),
        int(frame_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
        vjp_mode == "direct_atomic_track",
    )


def fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> Tensor:
    """Diagnostic VJP using midpoint owner selection for every replay segment."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_boundary_ids_i32 = candidate_boundary_ids_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    boundary_site_pairs_i32 = boundary_site_pairs_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_boundary_ids_i32, name="candidate_boundary_ids_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(boundary_site_pairs_i32, name="boundary_site_pairs_i32", dtype=torch.int32, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate numerator and denominator row counts must match")
    if candidate_boundary_ids_i32.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate_boundary_ids_i32 length must match candidate rows")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    candidate_ids_cpu = candidate_boundary_ids_i32.detach().cpu()
    boundary_pairs_cpu = boundary_site_pairs_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    if candidate_ids_cpu.numel() and (
        int(candidate_ids_cpu.min().item()) < 0 or int(candidate_ids_cpu.max().item()) >= boundary_site_pairs_i32.shape[0]
    ):
        raise ValueError("candidate_boundary_ids_i32 values must be in [0, boundary_count)")
    if boundary_pairs_cpu.numel() and (
        int(boundary_pairs_cpu.min().item()) < 0 or int(boundary_pairs_cpu.max().item()) >= sites_f32.shape[0]
    ):
        raise ValueError("boundary_site_pairs_i32 values must be in [0, site_count)")
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape or grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 and grad_depth_f32 must have shape [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate"
    if not hasattr(ops, op_name):
        raise RuntimeError(
            f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            boundary_site_pairs_i32.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> Tensor:
    """Accumulate mixed affine site RGBA gradients for RGB-only losses."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate numerator and denominator row counts must match")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only"
    if not hasattr(ops, op_name):
        raise RuntimeError(
            f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            1,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    _op_name: str = "fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only",
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and frozen-geometry site-RGBA gradients in one affine replay kernel."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.ndim != 3 or target_rgb_f32.shape[2] != 3:
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not target_rgb_f32.is_contiguous():
        raise ValueError("target_rgb_f32 must be contiguous")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate numerator and denominator row counts must match")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(
        row_offsets_cpu,
        candidate_count=candidate_depth_num_f32.shape[0],
        max_boundaries=MAX_REALRAY_FUSED_MSE_BOUNDARIES,
    )
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if target_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("target_rgb_f32 shape must be [track_count, frame_count, 3]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, _op_name):
        raise RuntimeError(
            f"world_foam_lane2_fused_slab_v0 {_op_name} op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            1,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, _op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Track-thread RGB MSE and frozen-geometry site-RGBA gradients for affine replay."""
    return fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only",
    )


def fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    _op_name: str = "fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only",
    _max_boundaries: int = MAX_REALRAY_FUSED_MSE_BOUNDARIES,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE and site-RGBA gradients using fully half affine depth coefficients."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_coeff_f16 = candidate_depth_coeff_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_coeff_f16, name="candidate_depth_coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.ndim != 3 or target_rgb_f32.shape[2] != 3:
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not target_rgb_f32.is_contiguous():
        raise ValueError("target_rgb_f32 must be contiguous")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(
        row_offsets_cpu,
        candidate_count=candidate_depth_coeff_f16.shape[0],
        max_boundaries=_max_boundaries,
    )
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if target_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("target_rgb_f32 shape must be [track_count, frame_count, 3]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, _op_name):
        raise RuntimeError(
            f"world_foam_lane2_fused_slab_v0 {_op_name} op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_coeff_f16.shape[0],
            1,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, _op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Sample-parallel coeff16 RGB MSE VJP specialized for rows with at most 224 candidates."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_cap224_mse_vjp_direct_atomic_rgb_only",
        _max_boundaries=MAX_REALRAY_FUSED_MSE_CAP224_BOUNDARIES,
    )


def fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Sample-parallel coeff16 RGB MSE VJP caching the forward density activity bit."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_densitymask_mse_vjp_direct_atomic_rgb_only",
    )


def fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    _op_name: str = "fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only",
) -> tuple[Tensor, Tensor]:
    """Framegroup-16 coeff16 RGB MSE VJP that caches row coefficients in threadgroup memory."""
    if time_slab_count != 1:
        raise ValueError("framegroup16 cached coeff16 fused MSE currently requires time_slab_count == 1")
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_mse_vjp_direct_atomic_framegroup16_cached_rgb_only",
    )


def fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Sample-parallel coeff16 RGB MSE VJP with a per-threadgroup cached site table."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_mse_vjp_direct_atomic_sitecache_rgb_only",
    )


def fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    _op_name: str = "fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only",
    _boundary_id_dtype: torch.dtype = torch.int32,
    _boundary_pair_dtype: torch.dtype = torch.int32,
) -> tuple[Tensor, Tensor]:
    """Compute RGB MSE gradients using coeff16 depths and boundary-id owner updates."""
    boundary_id_name = "candidate_boundary_ids_i16" if _boundary_id_dtype == torch.int16 else "candidate_boundary_ids_i32"
    boundary_pair_name = "boundary_site_pairs_i16" if _boundary_pair_dtype == torch.int16 else "boundary_site_pairs_i32"
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_boundary_ids_i32 = candidate_boundary_ids_i32.contiguous()
    candidate_depth_coeff_f16 = candidate_depth_coeff_f16.contiguous()
    boundary_site_pairs_i32 = boundary_site_pairs_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    target_rgb_f32 = target_rgb_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_boundary_ids_i32, name=boundary_id_name, dtype=_boundary_id_dtype, cols=None)
    _require_mps_tensor(candidate_depth_coeff_f16, name="candidate_depth_coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(boundary_site_pairs_i32, name=boundary_pair_name, dtype=_boundary_pair_dtype, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if target_rgb_f32.device.type != "mps" or target_rgb_f32.dtype != torch.float32:
        raise ValueError("target_rgb_f32 must be float32 on MPS")
    if target_rgb_f32.ndim != 3 or target_rgb_f32.shape[2] != 3:
        raise ValueError("target_rgb_f32 must have shape [track_count, frame_count, 3]")
    if not target_rgb_f32.is_contiguous():
        raise ValueError("target_rgb_f32 must be contiguous")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if candidate_boundary_ids_i32.shape[0] != candidate_depth_coeff_f16.shape[0]:
        raise ValueError(f"{boundary_id_name} length must match candidate_depth_coeff_f16 rows")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    candidate_ids_cpu = candidate_boundary_ids_i32.detach().cpu()
    boundary_pairs_cpu = boundary_site_pairs_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(
        row_offsets_cpu,
        candidate_count=candidate_depth_coeff_f16.shape[0],
        max_boundaries=MAX_REALRAY_FUSED_MSE_BOUNDARIES,
    )
    if candidate_ids_cpu.numel() and (
        int(candidate_ids_cpu.min().item()) < 0 or int(candidate_ids_cpu.max().item()) >= boundary_site_pairs_i32.shape[0]
    ):
        raise ValueError(f"{boundary_id_name} values must be in [0, boundary_count)")
    if boundary_pairs_cpu.numel() and (
        int(boundary_pairs_cpu.min().item()) < 0 or int(boundary_pairs_cpu.max().item()) >= sites_f32.shape[0]
    ):
        raise ValueError(f"{boundary_pair_name} values must be in [0, site_count)")
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if target_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("target_rgb_f32 shape must be [track_count, frame_count, 3]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, _op_name):
        raise RuntimeError(f"world_foam_lane2_fused_slab_v0 {_op_name} op not found. Build this variant first.")
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_coeff_f16.shape[0],
            boundary_site_pairs_i32.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, _op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Owner-update coeff16 MSE/VJP that keeps the current owner across unrelated candidate boundaries."""
    return fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_ownerkeep_mse_vjp_direct_atomic_rgb_only",
    )


def fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i16: Tensor,
    candidate_depth_coeff_f16: Tensor,
    boundary_site_pairs_i16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Owner-update coeff16 MSE/VJP using packed int16 boundary ids and site pairs."""
    return fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i16,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_ownerupdate_i16_mse_vjp_direct_atomic_rgb_only",
        _boundary_id_dtype=torch.int16,
        _boundary_pair_dtype=torch.int16,
    )


def fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i16: Tensor,
    candidate_depth_coeff_f16: Tensor,
    boundary_site_pairs_i16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Owner-keep coeff16 MSE/VJP using packed int16 boundary ids and site pairs."""
    return fused_slab_affine_coeff16_ownerupdate_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i16,
        candidate_depth_coeff_f16,
        boundary_site_pairs_i16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_ownerkeep_i16_mse_vjp_direct_atomic_rgb_only",
        _boundary_id_dtype=torch.int16,
        _boundary_pair_dtype=torch.int16,
    )


def fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Sample-parallel coeff16 MSE/VJP with local per-sample site-gradient reduction."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_mse_vjp_direct_atomic_sample_reduce_rgb_only",
    )


def fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Sample-parallel coeff16 MSE/VJP that collect-sorts depths with an in-thread bitonic network."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_sortnet_mse_vjp_direct_atomic_rgb_only",
    )


def fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    target_rgb_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor]:
    """Track-thread RGB MSE and site-RGBA gradients using fully half affine depth coefficients."""
    return fused_slab_affine_coeff16_mse_vjp_direct_atomic_rgb_only(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        target_rgb_f32,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        _op_name="fused_slab_affine_coeff16_mse_vjp_direct_atomic_track_rgb_only",
    )


def fused_slab_affine_num32_den16_vjp_direct_atomic_track(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    grad_rgb_f32: Tensor,
    grad_alpha_f32: Tensor,
    grad_depth_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> Tensor:
    """Accumulate mixed affine site RGBA gradients locally per track, then atomically write sites."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    grad_rgb_f32 = grad_rgb_f32.contiguous()
    grad_alpha_f32 = grad_alpha_f32.contiguous()
    grad_depth_f32 = grad_depth_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if grad_rgb_f32.device.type != "mps" or grad_rgb_f32.dtype != torch.float32:
        raise ValueError("grad_rgb_f32 must be float32 on MPS")
    if grad_rgb_f32.ndim != 3 or grad_rgb_f32.shape[2] != 3:
        raise ValueError("grad_rgb_f32 must have shape [track_count, frame_count, 3]")
    if grad_alpha_f32.device.type != "mps" or grad_alpha_f32.dtype != torch.float32:
        raise ValueError("grad_alpha_f32 must be float32 on MPS")
    if grad_alpha_f32.ndim != 2:
        raise ValueError("grad_alpha_f32 must have shape [track_count, frame_count]")
    if grad_depth_f32.device.type != "mps" or grad_depth_f32.dtype != torch.float32:
        raise ValueError("grad_depth_f32 must be float32 on MPS")
    if grad_depth_f32.ndim != 2:
        raise ValueError("grad_depth_f32 must have shape [track_count, frame_count]")
    if ray_coeff_f32.shape[0] <= 0 or frame_t_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 and frame_t_f32 must be nonempty")
    if sites_f32.shape[0] <= 0 or sites_f32.shape[0] > 64:
        raise ValueError("sites_f32 row count must be in [1, 64]")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate numerator and denominator row counts must match")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if candidate_row_offsets_i32.shape[0] != row_count * time_slab_count + 1:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    expected_shape = (ray_coeff_f32.shape[0], frame_t_f32.shape[0])
    if grad_rgb_f32.shape[:2] != expected_shape:
        raise ValueError("grad_rgb_f32 shape must be [track_count, frame_count, 3]")
    if grad_alpha_f32.shape != expected_shape or grad_depth_f32.shape != expected_shape:
        raise ValueError("grad_alpha_f32 and grad_depth_f32 must have shape [track_count, frame_count]")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    op_name = "fused_slab_affine_num32_den16_vjp_direct_atomic_track"
    if not hasattr(ops, op_name):
        raise RuntimeError(
            f"world_foam_lane2_fused_slab_v0 {op_name} op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            0,
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            1,
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return getattr(ops, op_name)(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config_i32,
        config_f32,
    )


class _FusedSlabAffineNum32Den16ReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        row_index_i32: Tensor,
        candidate_row_offsets_i32: Tensor,
        candidate_depth_num_f32: Tensor,
        candidate_depth_den_f16: Tensor,
        sites_f32: Tensor,
        site_rgba_f32: Tensor,
        ray_coeff_f32: Tensor,
        frame_t_f32: Tensor,
        time_slab_count: int,
        row_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
        reduce_chunk_size: int,
        use_direct_atomic: bool,
        direct_atomic_grad_only: bool,
        direct_atomic_rgb_only: bool,
        direct_atomic_track: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = fused_slab_affine_num32_den16_realray_rgba_depth_replay(
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
            config,
            time_slab_count=int(time_slab_count),
            row_count=int(row_count),
        )
        ctx.save_for_backward(
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
        )
        ctx.config = config
        ctx.time_slab_count = int(time_slab_count)
        ctx.row_count = int(row_count)
        ctx.reduce_chunk_size = int(reduce_chunk_size)
        ctx.use_direct_atomic = bool(use_direct_atomic)
        ctx.direct_atomic_grad_only = bool(direct_atomic_grad_only)
        ctx.direct_atomic_rgb_only = bool(direct_atomic_rgb_only)
        ctx.direct_atomic_track = bool(direct_atomic_track)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, None, None, Tensor, None, None, None, None, None, None, None, None, None, None, None, None, None]:
        (
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
        ) = ctx.saved_tensors
        track_count = ray_coeff_f32.shape[0]
        frame_count = frame_t_f32.shape[0]
        grad_alpha_is_none = grad_alpha is None
        grad_depth_is_none = grad_depth is None
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if bool(ctx.direct_atomic_rgb_only) and grad_alpha_is_none and grad_depth_is_none:
            grad_site_rgba = fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only(
                row_index_i32,
                candidate_row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb,
                ctx.config,
                time_slab_count=ctx.time_slab_count,
                row_count=ctx.row_count,
            )
            return None, None, None, None, None, grad_site_rgba, None, None, None, None, None, None, None, None, None, None, None, None, None
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if bool(ctx.direct_atomic_track):
            grad_site_rgba = fused_slab_affine_num32_den16_vjp_direct_atomic_track(
                row_index_i32,
                candidate_row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                time_slab_count=ctx.time_slab_count,
                row_count=ctx.row_count,
            )
        elif bool(ctx.direct_atomic_grad_only):
            grad_site_rgba = fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only(
                row_index_i32,
                candidate_row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                time_slab_count=ctx.time_slab_count,
                row_count=ctx.row_count,
            )
        elif bool(ctx.use_direct_atomic):
            _, _, _, grad_site_rgba = fused_slab_affine_num32_den16_vjp_direct_atomic(
                row_index_i32,
                candidate_row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                time_slab_count=ctx.time_slab_count,
                row_count=ctx.row_count,
            )
        else:
            _, _, _, grad_site_rgba = fused_slab_affine_num32_den16_vjp_reduce(
                row_index_i32,
                candidate_row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                grad_rgb,
                grad_alpha,
                grad_depth,
                ctx.config,
                time_slab_count=ctx.time_slab_count,
                row_count=ctx.row_count,
                reduce_chunk_size=ctx.reduce_chunk_size,
            )
        return None, None, None, None, None, grad_site_rgba, None, None, None, None, None, None, None, None, None, None, None, None, None


class _FusedSlabAffineNum32Den16OwnerUpdateReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        row_index_i32: Tensor,
        candidate_row_offsets_i32: Tensor,
        candidate_boundary_ids_i32: Tensor,
        candidate_depth_num_f32: Tensor,
        candidate_depth_den_f16: Tensor,
        boundary_site_pairs_i32: Tensor,
        sites_f32: Tensor,
        site_rgba_f32: Tensor,
        ray_coeff_f32: Tensor,
        frame_t_f32: Tensor,
        time_slab_count: int,
        row_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay(
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            boundary_site_pairs_i32,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
            config,
            time_slab_count=int(time_slab_count),
            row_count=int(row_count),
        )
        ctx.save_for_backward(
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            boundary_site_pairs_i32,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
        )
        ctx.config = config
        ctx.time_slab_count = int(time_slab_count)
        ctx.row_count = int(row_count)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, None, None, None, None, Tensor, None, None, None, None, None, None, None, None]:
        (
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            boundary_site_pairs_i32,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
        ) = ctx.saved_tensors
        track_count = ray_coeff_f32.shape[0]
        frame_count = frame_t_f32.shape[0]
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        grad_site_rgba = fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate(
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            candidate_depth_num_f32,
            candidate_depth_den_f16,
            boundary_site_pairs_i32,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
            grad_rgb,
            grad_alpha,
            grad_depth,
            ctx.config,
            time_slab_count=ctx.time_slab_count,
            row_count=ctx.row_count,
        )
        return None, None, None, None, None, None, None, grad_site_rgba, None, None, None, None, None, None, None, None


def fused_slab_affine_num32_den16_autograd(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
    reduce_chunk_size: int = 4,
    vjp_mode: str = "reduce",
) -> tuple[Tensor, Tensor, Tensor]:
    """Render mixed affine CSR candidates with frozen-geometry site-RGBA autograd."""
    if vjp_mode not in {
        "reduce",
        "direct_atomic",
        "direct_atomic_grad_only",
        "direct_atomic_rgb_only",
        "direct_atomic_track",
    }:
        raise ValueError(
            "vjp_mode must be 'reduce', 'direct_atomic', 'direct_atomic_grad_only', "
            "'direct_atomic_rgb_only', or 'direct_atomic_track'"
        )
    return _FusedSlabAffineNum32Den16ReplayFunction.apply(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        int(time_slab_count),
        int(row_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
        int(reduce_chunk_size),
        vjp_mode in {"direct_atomic", "direct_atomic_grad_only", "direct_atomic_rgb_only", "direct_atomic_track"},
        vjp_mode in {"direct_atomic_grad_only", "direct_atomic_rgb_only", "direct_atomic_track"},
        vjp_mode == "direct_atomic_rgb_only",
        vjp_mode == "direct_atomic_track",
    )


def fused_slab_affine_num32_den16_ownerupdate_autograd(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render affine CSR candidates with owner-update forward and frozen-geometry site-RGBA autograd."""
    return _FusedSlabAffineNum32Den16OwnerUpdateReplayFunction.apply(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        int(time_slab_count),
        int(row_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


def fused_slab_affine_realray_rgba_depth_replay(
    boundary_f32: Tensor,
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render affine moving-ray tracks from tiled/slab CSR candidate rows.

    `ray_coeff_f32 [K,12]` stores `(origin_base xyz, origin_slope xyz,
    direction_base xyz, direction_slope xyz)`. The Metal shader evaluates the
    ray at each `frame_t_f32[T]`, maps the track through `row_index_i32[K]`, and
    uses CSR row `row_index * time_slab_count + slab_id` for replay.
    """
    boundary_f32 = boundary_f32.contiguous()
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_boundary_ids_i32 = candidate_boundary_ids_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(boundary_f32, name="boundary_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_boundary_ids_i32, name="candidate_boundary_ids_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if boundary_f32.shape[0] > 128:
        raise ValueError("fused_slab_affine_realray_rgba_depth_replay currently supports at most 128 boundaries")
    if sites_f32.shape[0] > 64:
        raise ValueError("fused_slab_affine_realray_rgba_depth_replay currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    _validate_csr_values(
        row_index_i32=row_index_i32,
        candidate_row_offsets_i32=candidate_row_offsets_i32,
        candidate_boundary_ids_i32=candidate_boundary_ids_i32,
        row_count=row_count,
        candidate_count=candidate_boundary_ids_i32.shape[0],
        boundary_count=boundary_f32.shape[0],
    )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "fused_slab_affine_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 fused_slab_affine_realray_rgba_depth_replay op not found. "
            "Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            boundary_f32.shape[0],
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_boundary_ids_i32.shape[0],
        ],
        device=boundary_f32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=boundary_f32.device,
        dtype=torch.float32,
    )
    return ops.fused_slab_affine_realray_rgba_depth_replay(
        boundary_f32,
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_coeff_realray_rgba_depth_replay(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render per-track affine CSR candidates using precomputed depth coefficients.

    `candidate_depth_coeff_f32 [C,4]` stores `(numer_base, numer_slope,
    denom_base, denom_slope)` for the same CSR cursor order as
    `candidate_row_offsets_i32`. These coefficients are track-specific, so this
    path is intended for per-track rows where `row_index_i32[track] == track`.
    """
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_coeff_f32 = candidate_depth_coeff_f32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_coeff_f32, name="candidate_depth_coeff_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if sites_f32.shape[0] > 64:
        raise ValueError("fused_slab_affine_coeff_realray_rgba_depth_replay currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_coeff_f32.shape[0])
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "fused_slab_affine_coeff_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 fused_slab_affine_coeff_realray_rgba_depth_replay op not found. "
            "Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_coeff_f32.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return ops.fused_slab_affine_coeff_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_coeff16_realray_rgba_depth_replay(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_coeff_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render per-track affine CSR candidates using half-precision depth coefficients."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_coeff_f16 = candidate_depth_coeff_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_coeff_f16, name="candidate_depth_coeff_f16", dtype=torch.float16, cols=4)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if sites_f32.shape[0] > 64:
        raise ValueError("fused_slab_affine_coeff16_realray_rgba_depth_replay currently supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(
        row_offsets_cpu,
        candidate_count=candidate_depth_coeff_f16.shape[0],
        max_boundaries=MAX_REALRAY_FUSED_MSE_BOUNDARIES,
    )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "fused_slab_affine_coeff16_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 fused_slab_affine_coeff16_realray_rgba_depth_replay op not found. "
            "Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_coeff_f16.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return ops.fused_slab_affine_coeff16_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_coeff_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_realray_rgba_depth_replay(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render per-track affine CSR candidates with fp32 numerator and fp16 denominator coefficients."""
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if sites_f32.shape[0] > 64:
        raise ValueError("fused_slab_affine_num32_den16_realray_rgba_depth_replay supports at most 64 sites")
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate_depth_den_f16 row count must match candidate_depth_num_f32 rows")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(
        row_offsets_cpu,
        candidate_count=candidate_depth_num_f32.shape[0],
        max_boundaries=MAX_REALRAY_FUSED_MSE_BOUNDARIES,
    )
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "fused_slab_affine_num32_den16_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 fused_slab_affine_num32_den16_realray_rgba_depth_replay op not found. "
            "Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return ops.fused_slab_affine_num32_den16_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


def fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay(
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    candidate_depth_num_f32: Tensor,
    candidate_depth_den_f16: Tensor,
    boundary_site_pairs_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    ray_coeff_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render mixed-precision affine CSR candidates with midpoint owner selection.

    This is a diagnostic topology path for candidate-id CSR tapes. It keeps
    parity with the mixed replay path by recomputing the winning owner at each
    segment midpoint instead of assuming every candidate is a true owner
    transition.
    """
    row_index_i32 = row_index_i32.contiguous()
    candidate_row_offsets_i32 = candidate_row_offsets_i32.contiguous()
    candidate_boundary_ids_i32 = candidate_boundary_ids_i32.contiguous()
    candidate_depth_num_f32 = candidate_depth_num_f32.contiguous()
    candidate_depth_den_f16 = candidate_depth_den_f16.contiguous()
    boundary_site_pairs_i32 = boundary_site_pairs_i32.contiguous()
    sites_f32 = sites_f32.contiguous()
    site_rgba_f32 = site_rgba_f32.contiguous()
    ray_coeff_f32 = ray_coeff_f32.contiguous()
    frame_t_f32 = frame_t_f32.contiguous()
    _require_mps_tensor(row_index_i32, name="row_index_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_row_offsets_i32, name="candidate_row_offsets_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_boundary_ids_i32, name="candidate_boundary_ids_i32", dtype=torch.int32, cols=None)
    _require_mps_tensor(candidate_depth_num_f32, name="candidate_depth_num_f32", dtype=torch.float32, cols=2)
    _require_mps_tensor(candidate_depth_den_f16, name="candidate_depth_den_f16", dtype=torch.float16, cols=2)
    _require_mps_tensor(boundary_site_pairs_i32, name="boundary_site_pairs_i32", dtype=torch.int32, cols=2)
    _require_mps_tensor(sites_f32, name="sites_f32", dtype=torch.float32, cols=5)
    _require_mps_tensor(site_rgba_f32, name="site_rgba_f32", dtype=torch.float32, cols=4)
    _require_mps_tensor(ray_coeff_f32, name="ray_coeff_f32", dtype=torch.float32, cols=12)
    _require_mps_tensor(frame_t_f32, name="frame_t_f32", dtype=torch.float32, cols=None)
    if ray_coeff_f32.shape[0] <= 0:
        raise ValueError("ray_coeff_f32 must contain at least one track")
    if frame_t_f32.shape[0] <= 0:
        raise ValueError("frame_t_f32 must contain at least one frame")
    if sites_f32.shape[0] <= 0:
        raise ValueError("sites_f32 must contain at least one site")
    if sites_f32.shape[0] > 64:
        raise ValueError(
            "fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay supports at most 64 sites"
        )
    if site_rgba_f32.shape[0] != sites_f32.shape[0]:
        raise ValueError("site_rgba_f32 row count must match sites_f32 rows")
    if row_index_i32.shape[0] != ray_coeff_f32.shape[0]:
        raise ValueError("row_index_i32 length must match track count")
    if candidate_depth_den_f16.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate_depth_den_f16 row count must match candidate_depth_num_f32 rows")
    if candidate_boundary_ids_i32.shape[0] != candidate_depth_num_f32.shape[0]:
        raise ValueError("candidate_boundary_ids_i32 length must match candidate_depth_num_f32 rows")
    if time_slab_count <= 0:
        raise ValueError("time_slab_count must be positive")
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    expected_offsets = row_count * time_slab_count + 1
    if candidate_row_offsets_i32.shape[0] != expected_offsets:
        raise ValueError("candidate_row_offsets_i32 length must be row_count * time_slab_count + 1")
    row_index_cpu = row_index_i32.detach().cpu()
    row_offsets_cpu = candidate_row_offsets_i32.detach().cpu()
    candidate_ids_cpu = candidate_boundary_ids_i32.detach().cpu()
    boundary_pairs_cpu = boundary_site_pairs_i32.detach().cpu()
    if int(row_index_cpu.min().item()) < 0 or int(row_index_cpu.max().item()) >= row_count:
        raise ValueError("row_index_i32 values must be in [0, row_count)")
    _validate_csr_offsets_cpu(row_offsets_cpu, candidate_count=candidate_depth_num_f32.shape[0])
    if candidate_ids_cpu.numel() and (
        int(candidate_ids_cpu.min().item()) < 0 or int(candidate_ids_cpu.max().item()) >= boundary_site_pairs_i32.shape[0]
    ):
        raise ValueError("candidate_boundary_ids_i32 values must be in [0, boundary_count)")
    if boundary_pairs_cpu.numel() and (
        int(boundary_pairs_cpu.min().item()) < 0 or int(boundary_pairs_cpu.max().item()) >= sites_f32.shape[0]
    ):
        raise ValueError("boundary_site_pairs_i32 values must be in [0, site_count)")
    if config.far <= config.near:
        raise ValueError("config.far must be greater than config.near")
    if config.invalid_epsilon <= 0.0:
        raise ValueError("config.invalid_epsilon must be positive")
    if config.transmittance_threshold < 0.0:
        raise ValueError("config.transmittance_threshold must be nonnegative")

    ops = torch.ops.world_foam_lane2_fused_slab_v0
    if not hasattr(ops, "fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay"):
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay "
            "op not found. Build this variant first."
        )
    config_i32 = torch.tensor(
        [
            ray_coeff_f32.shape[0],
            sites_f32.shape[0],
            frame_t_f32.shape[0],
            time_slab_count,
            row_count,
            candidate_depth_num_f32.shape[0],
            boundary_site_pairs_i32.shape[0],
        ],
        device=row_index_i32.device,
        dtype=torch.int32,
    )
    config_f32 = torch.tensor(
        [config.near, config.far, config.invalid_epsilon, config.transmittance_threshold],
        device=row_index_i32.device,
        dtype=torch.float32,
    )
    return ops.fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay(
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        boundary_site_pairs_i32,
        sites_f32,
        site_rgba_f32,
        ray_coeff_f32,
        frame_t_f32,
        config_i32,
        config_f32,
    )


class _SharedRealRayRGBADepthReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        boundary_f32: Tensor,
        candidate_mask_i32: Tensor,
        sites_f32: Tensor,
        site_rgba_f32: Tensor,
        track_rays_f32: Tensor,
        frame_t_f32: Tensor,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        output_rgb, output_alpha, output_depth = shared_realray_rgba_depth_replay(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            config,
        )
        ctx.save_for_backward(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
        )
        ctx.config = config
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, Tensor, None, None, None, None, None, None]:
        (
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
        ) = ctx.saved_tensors
        track_count = track_rays_f32.shape[0]
        frame_count = frame_t_f32.shape[0]
        if grad_rgb is None:
            grad_rgb = torch.zeros(
                (track_count, frame_count, 3),
                device=site_rgba_f32.device,
                dtype=torch.float32,
            )
        if grad_alpha is None:
            grad_alpha = torch.zeros(
                (track_count, frame_count),
                device=site_rgba_f32.device,
                dtype=torch.float32,
            )
        if grad_depth is None:
            grad_depth = torch.zeros(
                (track_count, frame_count),
                device=site_rgba_f32.device,
                dtype=torch.float32,
            )
        _, _, _, grad_site_rgba = shared_realray_rgba_depth_vjp_reduce(
            boundary_f32,
            candidate_mask_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            grad_rgb,
            grad_alpha,
            grad_depth,
            ctx.config,
        )
        return None, None, None, grad_site_rgba, None, None, None, None, None, None


def shared_realray_rgba_depth_autograd(
    boundary_f32: Tensor,
    candidate_mask_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render shared real-ray RGB/alpha/depth with frozen-geometry autograd.

    Backward returns gradients only for `site_rgba_f32 [S,4]`. Boundary cuts,
    candidate masks, site positions/weights, rays, camera geometry, sorting,
    ownership, and topology are fixed and receive no gradients.
    """
    return _SharedRealRayRGBADepthReplayFunction.apply(
        boundary_f32,
        candidate_mask_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )


class _SharedRealRayRGBADepthCSRReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        boundary_f32: Tensor,
        row_index_i32: Tensor,
        candidate_row_offsets_i32: Tensor,
        candidate_boundary_ids_i32: Tensor,
        sites_f32: Tensor,
        site_rgba_f32: Tensor,
        track_rays_f32: Tensor,
        frame_t_f32: Tensor,
        time_slab_count: int,
        row_count: int,
        near: float,
        far: float,
        invalid_epsilon: float,
        transmittance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        config = RealRayReplayConfig(
            near=float(near),
            far=float(far),
            invalid_epsilon=float(invalid_epsilon),
            transmittance_threshold=float(transmittance_threshold),
        )
        track_count = track_rays_f32.shape[0]
        frame_count = frame_t_f32.shape[0]
        grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        output_rgb, output_alpha, output_depth, _ = shared_realray_rgba_depth_vjp_reduce_csr(
            boundary_f32,
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            grad_rgb,
            grad_alpha,
            grad_depth,
            config,
            time_slab_count=int(time_slab_count),
            row_count=int(row_count),
        )
        ctx.save_for_backward(
            boundary_f32,
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
        )
        ctx.config = config
        ctx.time_slab_count = int(time_slab_count)
        ctx.row_count = int(row_count)
        return output_rgb, output_alpha, output_depth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_rgb: Tensor | None,
        grad_alpha: Tensor | None,
        grad_depth: Tensor | None,
    ) -> tuple[None, None, None, None, None, Tensor, None, None, None, None, None, None, None, None]:
        (
            boundary_f32,
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
        ) = ctx.saved_tensors
        track_count = track_rays_f32.shape[0]
        frame_count = frame_t_f32.shape[0]
        if grad_rgb is None:
            grad_rgb = torch.zeros((track_count, frame_count, 3), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_alpha is None:
            grad_alpha = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        if grad_depth is None:
            grad_depth = torch.zeros((track_count, frame_count), device=site_rgba_f32.device, dtype=torch.float32)
        _, _, _, grad_site_rgba = shared_realray_rgba_depth_vjp_reduce_csr(
            boundary_f32,
            row_index_i32,
            candidate_row_offsets_i32,
            candidate_boundary_ids_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            grad_rgb,
            grad_alpha,
            grad_depth,
            ctx.config,
            time_slab_count=ctx.time_slab_count,
            row_count=ctx.row_count,
        )
        return None, None, None, None, None, grad_site_rgba, None, None, None, None, None, None, None, None


def shared_realray_rgba_depth_csr_autograd(
    boundary_f32: Tensor,
    row_index_i32: Tensor,
    candidate_row_offsets_i32: Tensor,
    candidate_boundary_ids_i32: Tensor,
    sites_f32: Tensor,
    site_rgba_f32: Tensor,
    track_rays_f32: Tensor,
    frame_t_f32: Tensor,
    config: RealRayReplayConfig,
    *,
    time_slab_count: int,
    row_count: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Render shared real-ray RGB/alpha/depth with CSR frozen-geometry autograd.

    Backward returns gradients only for `site_rgba_f32 [S,4]`. CSR candidate
    storage, boundary cuts, site positions/weights, rays, camera geometry,
    sorting, ownership, and topology are fixed and receive no gradients.
    """
    return _SharedRealRayRGBADepthCSRReplayFunction.apply(
        boundary_f32,
        row_index_i32,
        candidate_row_offsets_i32,
        candidate_boundary_ids_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        int(time_slab_count),
        int(row_count),
        config.near,
        config.far,
        config.invalid_epsilon,
        config.transmittance_threshold,
    )
