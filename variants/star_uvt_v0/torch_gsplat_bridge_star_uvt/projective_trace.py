from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProjectiveTraceFit:
    poly_coeffs: Tensor
    time_center: Tensor
    time_scale: Tensor
    degree: int
    residual_max_uv: Tensor
    residual_rms_uv: Tensor
    residual_max_depth: Tensor
    denominator_min_abs: Tensor
    denominator_has_root: Tensor
    valid_fraction: Tensor
    valid_count: Tensor


@dataclass(frozen=True)
class ProjectiveTraceWindow:
    start: int
    stop: int
    accepted: bool
    reason: str
    fit: ProjectiveTraceFit


@dataclass(frozen=True)
class ProjectiveTraceSupportBounds:
    start: int
    stop: int
    time_min: Tensor
    time_max: Tensor
    uv_min: Tensor
    uv_max: Tensor
    depth_min: Tensor
    depth_max: Tensor
    residual_max_uv: Tensor
    residual_max_depth: Tensor
    denominator_min_abs: Tensor


@dataclass(frozen=True)
class ProjectiveTraceVisibilitySidecar:
    start: int
    stop: int
    chart_gauge_id: int
    time_min: Tensor
    time_max: Tensor
    depth_coeffs: Tensor
    depth_min: Tensor
    depth_max: Tensor
    depth_slope_min: Tensor
    depth_slope_max: Tensor
    depth_monotonic_sign: Tensor
    depth_uncertainty: Tensor
    denominator_min_abs: Tensor
    denominator_has_root: Tensor


@dataclass(frozen=True)
class ProjectiveTraceDepthOrder:
    a_before_b: Tensor
    b_before_a: Tensor
    crosses: Tensor
    ambiguous: Tensor


@dataclass(frozen=True)
class ProjectiveTraceAppearanceSidecar:
    alpha_max: Tensor
    color_min: Tensor
    color_max: Tensor


@dataclass(frozen=True)
class ProjectiveTraceSwapCost:
    swap_bound: Tensor
    safely_commutable: Tensor
    needs_fallback: Tensor


@dataclass(frozen=True)
class ProjectiveTraceTileTimeRecord:
    primitive_id: int
    window_index: int
    start: int
    stop: int
    tile_u_min: int
    tile_u_max: int
    tile_v_min: int
    tile_v_max: int
    depth_min: float
    depth_max: float
    fallback: bool
    fallback_reason: str


@dataclass(frozen=True)
class ProjectiveTraceTileTimeCell:
    tile_u: int
    tile_v: int
    start: int
    stop: int
    primitive_ids: tuple[int, ...]
    ordered_primitive_ids: tuple[int, ...]
    depth_intervals: tuple[tuple[float, float], ...]
    fallback: bool
    fallback_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProjectiveTraceTileBins:
    tile_counts: Tensor
    tile_primitive_ids: Tensor
    tile_active_start: Tensor
    tile_active_stop: Tensor
    tile_overflow: Tensor


@dataclass(frozen=True)
class ProjectiveTraceCellTraceAtlas:
    coeffs: Tensor
    opacity: Tensor
    color: Tensor
    cells: list[ProjectiveTraceTileTimeCell]
    source_window_indices: tuple[int, ...]
    source_primitive_ids: tuple[int, ...]
    active_start: tuple[int, ...]
    active_stop: tuple[int, ...]
    opacity_time_coeffs: Tensor | None = None
    spatial_precision_uv: Tensor | None = None
    depth_affine_uv: Tensor | None = None
    # Original [ma_u,ma_v,ma_t,depth0,beta_u,beta_v,beta_t] for UVT fallback ordering.
    # It must be regenerated with the UVT projection when the world changes.
    depth_reference_uvt: Tensor | None = None


def slice_projective_trace_cell_atlas_frames(
    atlas: ProjectiveTraceCellTraceAtlas,
    *,
    start: int,
    stop: int,
) -> ProjectiveTraceCellTraceAtlas:
    """Return one frame window with compact, strictly active trace tables."""

    if start < 0 or stop <= start:
        raise ValueError("atlas frame slice must satisfy 0 <= start < stop")
    clipped_cells = []
    for cell in atlas.cells:
        overlap_start = max(int(start), int(cell.start))
        overlap_stop = min(int(stop), int(cell.stop))
        if overlap_stop <= overlap_start:
            continue
        clipped_cells.append(
            replace(
                cell,
                start=overlap_start - int(start),
                stop=overlap_stop - int(start),
            )
        )

    source_trace_indices = sorted(
        {
            int(trace_id)
            for cell in clipped_cells
            for trace_id in (*cell.primitive_ids, *cell.ordered_primitive_ids)
        }
    )
    trace_id_map = {
        source_trace_id: local_trace_id
        for local_trace_id, source_trace_id in enumerate(source_trace_indices)
    }
    cells = [
        replace(
            cell,
            primitive_ids=tuple(
                trace_id_map[int(value)] for value in cell.primitive_ids
            ),
            ordered_primitive_ids=tuple(
                trace_id_map[int(value)] for value in cell.ordered_primitive_ids
            ),
        )
        for cell in clipped_cells
    ]

    active_start = []
    active_stop = []
    for trace_id in source_trace_indices:
        overlap_start = max(int(start), int(atlas.active_start[trace_id]))
        overlap_stop = min(int(stop), int(atlas.active_stop[trace_id]))
        if overlap_stop <= overlap_start:
            raise ValueError(
                "compiled atlas cell references a trace inactive in the frame slice"
            )
        active_start.append(overlap_start - int(start))
        active_stop.append(overlap_stop - int(start))

    index = torch.tensor(
        source_trace_indices,
        dtype=torch.long,
        device=atlas.coeffs.device,
    )

    def select_optional(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.index_select(0, index)

    return replace(
        atlas,
        coeffs=atlas.coeffs.index_select(0, index),
        opacity=atlas.opacity.index_select(0, index),
        color=atlas.color.index_select(0, index),
        cells=cells,
        source_window_indices=tuple(
            int(atlas.source_window_indices[trace_id])
            for trace_id in source_trace_indices
        ),
        source_primitive_ids=tuple(
            int(atlas.source_primitive_ids[trace_id])
            for trace_id in source_trace_indices
        ),
        active_start=tuple(active_start),
        active_stop=tuple(active_stop),
        opacity_time_coeffs=select_optional(atlas.opacity_time_coeffs),
        spatial_precision_uv=select_optional(atlas.spatial_precision_uv),
        depth_affine_uv=select_optional(atlas.depth_affine_uv),
        depth_reference_uvt=select_optional(atlas.depth_reference_uvt),
    )


@dataclass(frozen=True)
class ProjectiveTraceCellAtlasCoverageReport:
    stale: bool
    checked_tile_pairs: int
    missing_tile_pairs: int
    invalid_active_samples: int
    missing_examples: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class ProjectiveTraceCellAtlasSupportMarginReport:
    stale: bool
    checked_tile_pairs: int
    missing_tile_pairs: int
    invalid_active_samples: int
    missing_without_covered_sample: int
    min_boundary_slack_px: float
    mean_boundary_slack_px: float
    p05_boundary_slack_px: float
    max_boundary_overshoot_px: float
    mean_boundary_overshoot_px: float
    p95_boundary_overshoot_px: float
    missing_examples: tuple[tuple[int, int, int, int, float], ...]


@dataclass(frozen=True)
class ProjectiveTraceCellAtlasVisibilityReport:
    stale: bool
    checked_tile_samples: int
    order_mismatch_samples: int
    ambiguous_depth_samples: int
    invalid_depth_samples: int
    mismatch_examples: tuple[tuple[int, int, int, int, int], ...]
    ambiguous_examples: tuple[tuple[int, int, int, int, int], ...]


@dataclass(frozen=True)
class ProjectiveTraceCellAtlasFallbackStats:
    total_cells: int
    fallback_cells: int
    total_tile_samples: int
    fallback_tile_samples: int
    total_trace_samples: int
    fallback_trace_samples: int
    fallback_fraction: float
    fallback_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProjectiveTraceCellAtlasComplexityStats:
    total_cells: int
    tile_active_set_groups: int
    visibility_stratum_split_cells: int
    max_cells_per_active_set_group: int
    interval_trace_entries: int
    dense_trace_samples: int
    interval_to_dense_trace_sample_ratio: float
    fallback_cells: int
    fallback_fraction: float


@dataclass(frozen=True)
class ProjectiveTraceCellAtlasBudgetReport:
    within_budget: bool
    stats: ProjectiveTraceCellAtlasComplexityStats
    failures: tuple[str, ...]
    max_interval_to_dense_trace_sample_ratio: float
    max_fallback_fraction: float
    max_cells_per_active_set_group: int


@dataclass(frozen=True)
class ProjectiveTraceCellVisibilityEvent:
    cell_index: int
    tile_u: int
    tile_v: int
    trace_a: int
    trace_b: int
    time: float


@dataclass(frozen=True)
class ProjectiveTraceCellVisibilityEventReport:
    events: tuple[ProjectiveTraceCellVisibilityEvent, ...]
    split_times: tuple[float, ...]


@dataclass(frozen=True)
class ProjectiveTraceCellUVVisibilityEvent:
    cell_index: int
    tile_u: int
    tile_v: int
    sample_index: int
    trace_a: int
    trace_b: int
    time: float
    line_u: float
    line_v: float
    line_0: float
    min_delta: float
    max_delta: float


@dataclass(frozen=True)
class ProjectiveTraceCellUVVisibilityEventReport:
    events: tuple[ProjectiveTraceCellUVVisibilityEvent, ...]
    event_tile_samples: int


@dataclass(frozen=True)
class ProjectiveTraceCellUVVisibilitySpatialSplitReport:
    atlas: ProjectiveTraceCellTraceAtlas
    accepted: bool
    split_attempted: bool
    input_tile_size: int
    output_tile_size: int
    candidate_tile_sizes: tuple[int, ...]
    parent_cells: int
    parent_uv_events: int
    parent_uv_event_tile_samples: int
    parent_fallback_cells: int
    parent_fallback_fraction: float
    residual_uv_events: int
    residual_uv_event_tile_samples: int
    output_cells: int
    fallback_cells: int
    fallback_fraction: float


@dataclass(frozen=True)
class ProjectiveTraceCellSupportEvent:
    trace_id: int
    axis: str
    side: str
    boundary: float
    time: float


@dataclass(frozen=True)
class ProjectiveTraceCellSupportEventReport:
    events: tuple[ProjectiveTraceCellSupportEvent, ...]
    split_times: tuple[float, ...]


@dataclass(frozen=True)
class ProjectiveTraceCellSensorTimeInterval:
    start_time: float
    stop_time: float


@dataclass(frozen=True)
class ProjectiveTraceCellSensorTimePartition:
    intervals: tuple[ProjectiveTraceCellSensorTimeInterval, ...]
    split_times: tuple[float, ...]
    support_events: tuple[ProjectiveTraceCellSupportEvent, ...]
    visibility_events: tuple[ProjectiveTraceCellVisibilityEvent, ...]


@dataclass(frozen=True)
class ProjectiveTraceCellSensorTimeQuadratureSample:
    interval_index: int
    row_index: int
    start_time: float
    stop_time: float
    time: float
    weight: float


@dataclass(frozen=True)
class ProjectiveTraceCellSensorTimeQuadrature:
    samples: tuple[ProjectiveTraceCellSensorTimeQuadratureSample, ...]
    total_weight: float


@dataclass(frozen=True)
class ProjectiveTraceCellQuadratureLowering:
    atlas: ProjectiveTraceCellTraceAtlas
    times: Tensor
    weights: Tensor
    source_trace_indices: tuple[int, ...]


@dataclass(frozen=True)
class ProjectiveTraceCellRollingQuadratureLowering:
    atlas: ProjectiveTraceCellTraceAtlas
    times: Tensor
    row_weights: Tensor
    source_trace_indices: tuple[int, ...]


@dataclass(frozen=True)
class ProjectiveTraceAtlasGrad:
    grad_coeffs: Tensor
    grad_opacity: Tensor
    grad_color: Tensor
    grad_opacity_time_coeffs: Tensor | None = None
    grad_spatial_precision_uv: Tensor | None = None


@dataclass(frozen=True)
class ProjectiveTraceFamilyAtlasGrad:
    grad_family_coeffs: Tensor
    grad_q_basis: Tensor
    grad_opacity: Tensor
    grad_color: Tensor
    grad_opacity_time_coeffs: Tensor | None = None
    grad_spatial_precision_uv: Tensor | None = None


@dataclass(frozen=True)
class ProjectiveTraceUVTBridge:
    ma: Tensor
    q_uvt: Tensor
    depth0: Tensor
    depth_beta: Tensor
    opacity: Tensor
    color: Tensor
    source_window_indices: tuple[int, ...]
    source_primitive_ids: tuple[int, ...]
    active_start: tuple[int, ...]
    active_stop: tuple[int, ...]


@dataclass(frozen=True)
class ProjectiveTraceUVTBridgeGrad:
    grad_ma: Tensor
    grad_q_uvt: Tensor
    grad_depth0: Tensor
    grad_depth_beta: Tensor
    grad_opacity: Tensor
    grad_color: Tensor
    tile_unstable: Tensor


def _check_projective_trace_render_inputs(
    coeffs: Tensor,
    times: Tensor,
    colors: Tensor,
    opacities: Tensor,
) -> None:
    _check_projective_trace_inputs(coeffs, times)
    if colors.ndim != 2:
        raise ValueError("colors must have shape [N,C]")
    if opacities.ndim != 1:
        raise ValueError("opacities must have shape [N]")
    if colors.shape[0] != coeffs.shape[0] or opacities.shape[0] != coeffs.shape[0]:
        raise ValueError("colors, opacities, and coeffs must share N")
    if colors.dtype != torch.float32 or opacities.dtype != torch.float32:
        raise ValueError("colors and opacities must be float32")
    if colors.device != coeffs.device or opacities.device != coeffs.device:
        raise ValueError("colors, opacities, and coeffs must be on the same device")


def _check_projective_trace_inputs(coeffs: Tensor, times: Tensor) -> None:
    if coeffs.ndim != 2 or coeffs.shape[-1] != 9:
        raise ValueError("coeffs must have shape [N,9]")
    if times.ndim != 1:
        raise ValueError("times must have shape [S]")
    if coeffs.dtype != torch.float32:
        raise ValueError("coeffs must be float32")
    if times.dtype != torch.float32:
        raise ValueError("times must be float32")
    if coeffs.device != times.device:
        raise ValueError("coeffs and times must be on the same device")
    if not coeffs.is_contiguous():
        raise ValueError("coeffs must be contiguous")
    if not times.is_contiguous():
        raise ValueError("times must be contiguous")


def _check_projective_trace_family_inputs(family_coeffs: Tensor, q_basis: Tensor, times: Tensor) -> None:
    if family_coeffs.ndim != 3 or family_coeffs.shape[1] != 9:
        raise ValueError("family_coeffs must have shape [N,9,B]")
    if q_basis.ndim != 2 or q_basis.shape[1] != family_coeffs.shape[2]:
        raise ValueError("q_basis must have shape [Q,B] matching family_coeffs")
    if times.ndim != 1:
        raise ValueError("times must have shape [S]")
    if family_coeffs.dtype != torch.float32 or q_basis.dtype != torch.float32 or times.dtype != torch.float32:
        raise ValueError("family_coeffs, q_basis, and times must be float32")
    if q_basis.device != family_coeffs.device or times.device != family_coeffs.device:
        raise ValueError("family_coeffs, q_basis, and times must be on the same device")
    if not family_coeffs.is_contiguous():
        raise ValueError("family_coeffs must be contiguous")
    if not q_basis.is_contiguous():
        raise ValueError("q_basis must be contiguous")
    if not times.is_contiguous():
        raise ValueError("times must be contiguous")


def _check_projective_trace_cell_inputs(coeffs: Tensor, times: Tensor) -> None:
    if coeffs.ndim != 2 or coeffs.shape[-1] != 9:
        raise ValueError("cell coeffs must have shape [M,9]")
    if times.ndim != 1:
        raise ValueError("times must have shape [S]")
    if coeffs.dtype != torch.float32:
        raise ValueError("cell coeffs must be float32")
    if times.dtype != torch.float32:
        raise ValueError("times must be float32")
    if coeffs.device != times.device:
        raise ValueError("cell coeffs and times must be on the same device")
    if not coeffs.is_contiguous():
        raise ValueError("cell coeffs must be contiguous")
    if not times.is_contiguous():
        raise ValueError("times must be contiguous")


def has_projective_trace_metal() -> bool:
    return hasattr(torch.ops.star_uvt_v0, "projective_trace_eval")


def has_projective_trace_family_metal() -> bool:
    return hasattr(torch.ops.star_uvt_v0, "projective_trace_family_eval")


def has_projective_trace_family_backward_metal() -> bool:
    return hasattr(torch.ops.star_uvt_v0, "projective_trace_family_backward")


def eval_projective_trace_torch(coeffs: Tensor, times: Tensor, *, eps: float = 1.0e-6) -> Tensor:
    _check_projective_trace_inputs(coeffs, times)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    t = times.reshape(1, -1)
    t2 = t.square()
    hu = coeffs[:, 0:1] + coeffs[:, 1:2] * t + coeffs[:, 2:3] * t2
    hv = coeffs[:, 3:4] + coeffs[:, 4:5] * t + coeffs[:, 5:6] * t2
    hz = coeffs[:, 6:7] + coeffs[:, 7:8] * t + coeffs[:, 8:9] * t2
    finite = torch.isfinite(hu) & torch.isfinite(hv) & torch.isfinite(hz)
    valid = finite & (hz.abs() > float(eps))
    safe_hz = torch.where(valid, hz, torch.ones_like(hz))
    u = torch.where(valid, hu / safe_hz, torch.zeros_like(hu))
    v = torch.where(valid, hv / safe_hz, torch.zeros_like(hv))
    valid_sign = torch.where(
        valid,
        torch.where(hz > 0.0, torch.ones_like(hz), -torch.ones_like(hz)),
        torch.zeros_like(hz),
    )
    return torch.stack((u, v, hz, valid_sign), dim=-1)


def eval_projective_trace_family_torch(
    family_coeffs: Tensor,
    q_basis: Tensor,
    times: Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    _check_projective_trace_family_inputs(family_coeffs, q_basis, times)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    coeffs = torch.einsum("nkb,qb->qnk", family_coeffs, q_basis)
    t = times.reshape(1, 1, -1)
    t2 = t.square()
    hu = coeffs[:, :, 0:1] + coeffs[:, :, 1:2] * t + coeffs[:, :, 2:3] * t2
    hv = coeffs[:, :, 3:4] + coeffs[:, :, 4:5] * t + coeffs[:, :, 5:6] * t2
    hz = coeffs[:, :, 6:7] + coeffs[:, :, 7:8] * t + coeffs[:, :, 8:9] * t2
    finite = torch.isfinite(hu) & torch.isfinite(hv) & torch.isfinite(hz)
    valid = finite & (hz.abs() > float(eps))
    safe_hz = torch.where(valid, hz, torch.ones_like(hz))
    u = torch.where(valid, hu / safe_hz, torch.zeros_like(hu))
    v = torch.where(valid, hv / safe_hz, torch.zeros_like(hv))
    valid_sign = torch.where(
        valid,
        torch.where(hz > 0.0, torch.ones_like(hz), -torch.ones_like(hz)),
        torch.zeros_like(hz),
    )
    return torch.stack((u, v, hz, valid_sign), dim=-1)


def eval_projective_trace_cell_torch(coeffs: Tensor, times: Tensor) -> Tensor:
    """Evaluate direct cell-local trace polynomials.

    Unlike homogeneous projective coeffs, cell coeffs store raw-time direct
    polynomials for screen position and conditional depth:
    ``[u0,u1,u2, v0,v1,v2, z0,z1,z2]``.
    """

    _check_projective_trace_cell_inputs(coeffs, times)
    t = times.reshape(1, -1)
    t2 = t.square()
    u = coeffs[:, 0:1] + coeffs[:, 1:2] * t + coeffs[:, 2:3] * t2
    v = coeffs[:, 3:4] + coeffs[:, 4:5] * t + coeffs[:, 5:6] * t2
    depth = coeffs[:, 6:7] + coeffs[:, 7:8] * t + coeffs[:, 8:9] * t2
    valid = torch.isfinite(u) & torch.isfinite(v) & torch.isfinite(depth)
    valid_sign = valid.to(dtype=coeffs.dtype)
    return torch.stack(
        (
            torch.where(valid, u, torch.zeros_like(u)),
            torch.where(valid, v, torch.zeros_like(v)),
            torch.where(valid, depth, torch.zeros_like(depth)),
            valid_sign,
        ),
        dim=-1,
    )


def eval_projective_trace(coeffs: Tensor, times: Tensor, *, eps: float = 1.0e-6) -> Tensor:
    _check_projective_trace_inputs(coeffs, times)
    if coeffs.device.type != "mps":
        return eval_projective_trace_torch(coeffs, times, eps=eps)
    if not has_projective_trace_metal():
        raise RuntimeError("star_uvt_v0.projective_trace_eval Metal op is not available")
    return torch.ops.star_uvt_v0.projective_trace_eval(coeffs, times, float(eps))


def eval_projective_trace_family(
    family_coeffs: Tensor,
    q_basis: Tensor,
    times: Tensor,
    *,
    eps: float = 1.0e-6,
) -> Tensor:
    _check_projective_trace_family_inputs(family_coeffs, q_basis, times)
    if family_coeffs.device.type != "mps":
        return eval_projective_trace_family_torch(family_coeffs, q_basis, times, eps=eps)
    if not has_projective_trace_family_metal():
        raise RuntimeError("star_uvt_v0.projective_trace_family_eval Metal op is not available")
    return torch.ops.star_uvt_v0.projective_trace_family_eval(family_coeffs, q_basis, times, float(eps))


def direct_backward_projective_trace_family_metal(
    family_coeffs: Tensor,
    q_basis: Tensor,
    times: Tensor,
    grad_out: Tensor,
    *,
    eps: float = 1.0e-6,
) -> tuple[Tensor, Tensor]:
    _check_projective_trace_family_inputs(family_coeffs, q_basis, times)
    if grad_out.shape != (q_basis.shape[0], family_coeffs.shape[0], times.shape[0], 4):
        raise ValueError("grad_out must have shape [Q,N,S,4]")
    if grad_out.dtype != torch.float32 or grad_out.device != family_coeffs.device:
        raise ValueError("grad_out must be float32 and on the same device as family_coeffs")
    if not grad_out.is_contiguous():
        raise ValueError("grad_out must be contiguous")
    if family_coeffs.device.type != "mps":
        raise ValueError("projective trace family Metal backward requires MPS tensors")
    if not has_projective_trace_family_backward_metal():
        raise RuntimeError("star_uvt_v0.projective_trace_family_backward Metal op is not available")
    return torch.ops.star_uvt_v0.projective_trace_family_backward(
        family_coeffs,
        q_basis,
        times,
        grad_out,
        float(eps),
    )


def _normalized_time_basis(times: Tensor, degree: int) -> tuple[Tensor, Tensor, Tensor]:
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    center = 0.5 * (times.amin() + times.amax())
    raw_scale = 0.5 * (times.amax() - times.amin())
    scale = torch.where(raw_scale > 0.0, raw_scale, torch.ones_like(raw_scale))
    t = (times - center) / scale
    columns = [torch.ones_like(t), t]
    if degree == 2:
        columns.append(t.square())
    return torch.stack(columns, dim=-1), center, scale


def _denominator_interval_certificate(coeffs: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
    """Return continuous zero-event and minimum-absolute-depth certificates.

    The denominator is quadratic in raw time.  Recenter it on the queried
    interval, then inspect its endpoints and stationary point.  Those are the
    only candidates for the range and for the minimum absolute denominator.
    This avoids classifying a valid quadratic as constant merely because its
    raw-time coefficient is numerically small.
    """

    t_min = times.amin()
    t_max = times.amax()
    time_center = 0.5 * (t_min + t_max)
    raw_scale = 0.5 * (t_max - t_min)
    time_scale = torch.where(raw_scale > 0.0, raw_scale, torch.ones_like(raw_scale))
    s_min = (t_min - time_center) / time_scale
    s_max = (t_max - time_center) / time_scale

    c_raw = coeffs[:, 6]
    b_raw = coeffs[:, 7]
    a_raw = coeffs[:, 8]
    c = c_raw + b_raw * time_center + a_raw * time_center.square()
    b = (b_raw + 2.0 * a_raw * time_center) * time_scale
    a = a_raw * time_scale.square()

    def evaluate(s: Tensor) -> Tensor:
        return c + b * s + a * s.square()

    endpoint_min = evaluate(s_min)
    endpoint_max = evaluate(s_max)
    safe_a = torch.where(a != 0.0, a, torch.ones_like(a))
    stationary = -0.5 * b / safe_a
    stationary_in_interval = (a != 0.0) & (stationary >= s_min) & (stationary <= s_max)
    stationary_value = evaluate(stationary)

    values = torch.stack((endpoint_min, endpoint_max, stationary_value), dim=1)
    candidates = torch.stack(
        (
            torch.ones_like(stationary_in_interval),
            torch.ones_like(stationary_in_interval),
            stationary_in_interval,
        ),
        dim=1,
    )
    finite = torch.isfinite(c) & torch.isfinite(b) & torch.isfinite(a)
    finite = finite & torch.all(torch.isfinite(values) | ~candidates, dim=1)
    inf = torch.full_like(values, float("inf"))
    min_value = torch.where(candidates, values, inf).amin(dim=1)
    max_value = torch.where(candidates, values, -inf).amax(dim=1)
    min_abs = torch.where(candidates, values.abs(), inf).amin(dim=1)

    coefficient_scale = torch.stack((c.abs(), b.abs(), a.abs()), dim=1).amax(dim=1).clamp_min(1.0)
    root_tolerance = 16.0 * torch.finfo(coeffs.dtype).eps * coefficient_scale
    has_root = finite & (min_value <= root_tolerance) & (max_value >= -root_tolerance)
    min_abs = torch.where(finite, torch.where(has_root, torch.zeros_like(min_abs), min_abs), torch.zeros_like(min_abs))
    return has_root, min_abs


def fit_projective_trace_polynomial(
    coeffs: Tensor,
    times: Tensor,
    *,
    degree: int = 1,
    eps: float = 1.0e-6,
    ridge: float = 1.0e-6,
) -> ProjectiveTraceFit:
    _check_projective_trace_inputs(coeffs, times)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")

    samples = eval_projective_trace_torch(coeffs, times, eps=eps)
    values = samples[:, :, :3]
    valid = samples[:, :, 3] != 0.0
    weights = valid.to(dtype=values.dtype)
    basis, center, scale = _normalized_time_basis(times, degree)

    at_w = basis.t().unsqueeze(0) * weights.unsqueeze(1)
    normal = at_w @ basis.unsqueeze(0)
    eye = torch.eye(normal.shape[-1], dtype=normal.dtype, device=normal.device).unsqueeze(0)
    rhs = at_w @ values
    solved = torch.linalg.solve(normal + float(ridge) * eye, rhs)
    poly_coeffs = solved.permute(0, 2, 1).contiguous()

    pred = basis.unsqueeze(0) @ solved
    uv_delta = pred[:, :, :2] - values[:, :, :2]
    uv_norm = uv_delta.square().sum(dim=-1).sqrt()
    depth_abs = (pred[:, :, 2] - values[:, :, 2]).abs()
    masked_uv = torch.where(valid, uv_norm, torch.zeros_like(uv_norm))
    masked_depth = torch.where(valid, depth_abs, torch.zeros_like(depth_abs))
    valid_count = valid.sum(dim=1)
    enough = valid_count >= (degree + 1)
    denominator_has_root, denominator_min_abs = _denominator_interval_certificate(coeffs, times)

    residual_max_uv = masked_uv.amax(dim=1)
    residual_rms_uv = (
        masked_uv.square().sum(dim=1)
        / valid_count.clamp_min(1).to(dtype=masked_uv.dtype)
    ).sqrt()
    residual_max_depth = masked_depth.amax(dim=1)
    inf = torch.full_like(residual_max_uv, float("inf"))
    residual_max_uv = torch.where(enough, residual_max_uv, inf)
    residual_rms_uv = torch.where(enough, residual_rms_uv, inf)
    residual_max_depth = torch.where(enough, residual_max_depth, inf)
    denominator_min_abs = torch.where(enough, denominator_min_abs, inf)
    return ProjectiveTraceFit(
        poly_coeffs=poly_coeffs,
        time_center=center,
        time_scale=scale,
        degree=degree,
        residual_max_uv=residual_max_uv,
        residual_rms_uv=residual_rms_uv,
        residual_max_depth=residual_max_depth,
        denominator_min_abs=denominator_min_abs,
        denominator_has_root=denominator_has_root,
        valid_fraction=valid_count.to(dtype=values.dtype) / float(times.numel()),
        valid_count=valid_count,
    )


def eval_projective_trace_polynomial_fit(fit: ProjectiveTraceFit, times: Tensor) -> Tensor:
    if times.ndim != 1:
        raise ValueError("times must have shape [S]")
    if times.dtype != torch.float32:
        raise ValueError("times must be float32")
    if times.device != fit.poly_coeffs.device:
        raise ValueError("times and fit.poly_coeffs must be on the same device")
    t = (times - fit.time_center) / fit.time_scale
    columns = [torch.ones_like(t), t]
    if fit.degree == 2:
        columns.append(t.square())
    elif fit.degree != 1:
        raise ValueError("fit.degree must be 1 or 2")
    basis = torch.stack(columns, dim=-1)
    return basis.unsqueeze(0) @ fit.poly_coeffs.permute(0, 2, 1)


def _eval_normalized_polynomial(poly_coeffs: Tensor, t: Tensor) -> Tensor:
    value = poly_coeffs[:, :, 0] + poly_coeffs[:, :, 1] * t
    if poly_coeffs.shape[-1] == 3:
        value = value + poly_coeffs[:, :, 2] * t.square()
    return value


def _normalized_polynomial_range(poly_coeffs: Tensor) -> tuple[Tensor, Tensor]:
    if poly_coeffs.ndim != 3 or poly_coeffs.shape[-1] not in (2, 3):
        raise ValueError("poly_coeffs must have shape [N,C,2] or [N,C,3]")
    endpoint_t = torch.tensor([-1.0, 1.0], dtype=poly_coeffs.dtype, device=poly_coeffs.device)
    candidates = [_eval_normalized_polynomial(poly_coeffs, endpoint_t[0]), _eval_normalized_polynomial(poly_coeffs, endpoint_t[1])]

    if poly_coeffs.shape[-1] == 3:
        linear = poly_coeffs[:, :, 1]
        quadratic = poly_coeffs[:, :, 2]
        safe_quadratic = torch.where(quadratic.abs() > 1.0e-12, quadratic, torch.ones_like(quadratic))
        vertex_t = -linear / (2.0 * safe_quadratic)
        vertex_value = _eval_normalized_polynomial(poly_coeffs, vertex_t)
        endpoint_value = candidates[0]
        vertex_valid = (quadratic.abs() > 1.0e-12) & (vertex_t >= -1.0) & (vertex_t <= 1.0)
        candidates.append(torch.where(vertex_valid, vertex_value, endpoint_value))

    stacked = torch.stack(candidates, dim=-1)
    return stacked.amin(dim=-1), stacked.amax(dim=-1)


def _normalized_polynomial_derivative_range(poly_coeffs: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    if poly_coeffs.ndim != 2 or poly_coeffs.shape[-1] not in (2, 3):
        raise ValueError("poly_coeffs must have shape [N,2] or [N,3]")
    inv_scale = 1.0 / scale.clamp_min(1.0e-12)
    if poly_coeffs.shape[-1] == 2:
        derivative = poly_coeffs[:, 1] * inv_scale
        return derivative, derivative
    endpoint_min = (poly_coeffs[:, 1] - 2.0 * poly_coeffs[:, 2]).minimum(poly_coeffs[:, 1] + 2.0 * poly_coeffs[:, 2])
    endpoint_max = (poly_coeffs[:, 1] - 2.0 * poly_coeffs[:, 2]).maximum(poly_coeffs[:, 1] + 2.0 * poly_coeffs[:, 2])
    return endpoint_min * inv_scale, endpoint_max * inv_scale


def _polynomial_has_root_on_unit_interval(poly_coeffs: Tensor, *, eps: float) -> Tensor:
    if poly_coeffs.ndim != 2 or poly_coeffs.shape[-1] not in (2, 3):
        raise ValueError("poly_coeffs must have shape [N,2] or [N,3]")
    c = poly_coeffs[:, 0]
    b = poly_coeffs[:, 1]
    if poly_coeffs.shape[-1] == 2:
        abs_b = b.abs()
        constant = abs_b <= float(eps)
        constant_root = constant & (c.abs() <= float(eps))
        root = -c / torch.where(abs_b > float(eps), b, torch.ones_like(b))
        linear_root = ~constant & (root >= -1.0 - float(eps)) & (root <= 1.0 + float(eps))
        return constant_root | linear_root

    a = poly_coeffs[:, 2]
    abs_a = a.abs()
    abs_b = b.abs()
    constant = (abs_a <= float(eps)) & (abs_b <= float(eps))
    constant_root = constant & (c.abs() <= float(eps))
    linear = (abs_a <= float(eps)) & ~constant
    linear_root = -c / torch.where(abs_b > float(eps), b, torch.ones_like(b))
    linear_in_interval = linear & (linear_root >= -1.0 - float(eps)) & (linear_root <= 1.0 + float(eps))
    quadratic = abs_a > float(eps)
    discriminant = b.square() - 4.0 * a * c
    has_real_roots = quadratic & (discriminant >= -float(eps))
    sqrt_disc = discriminant.clamp_min(0.0).sqrt()
    denom = 2.0 * torch.where(abs_a > float(eps), a, torch.ones_like(a))
    root0 = (-b - sqrt_disc) / denom
    root1 = (-b + sqrt_disc) / denom
    root0_in_interval = (root0 >= -1.0 - float(eps)) & (root0 <= 1.0 + float(eps))
    root1_in_interval = (root1 >= -1.0 - float(eps)) & (root1 <= 1.0 + float(eps))
    return constant_root | linear_in_interval | (has_real_roots & (root0_in_interval | root1_in_interval))


def bound_projective_trace_window(
    window: ProjectiveTraceWindow,
    *,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    require_accepted: bool = True,
) -> ProjectiveTraceSupportBounds:
    if uv_padding < 0.0:
        raise ValueError("uv_padding must be non-negative")
    if depth_padding < 0.0:
        raise ValueError("depth_padding must be non-negative")
    if require_accepted and not window.accepted:
        raise ValueError(f"cannot bound unresolved projective trace window: {window.reason}")

    fit = window.fit
    value_min, value_max = _normalized_polynomial_range(fit.poly_coeffs)
    uv_slack = fit.residual_max_uv.unsqueeze(-1) + float(uv_padding)
    depth_slack = fit.residual_max_depth + float(depth_padding)

    return ProjectiveTraceSupportBounds(
        start=window.start,
        stop=window.stop,
        time_min=fit.time_center - fit.time_scale,
        time_max=fit.time_center + fit.time_scale,
        uv_min=value_min[:, :2] - uv_slack,
        uv_max=value_max[:, :2] + uv_slack,
        depth_min=value_min[:, 2] - depth_slack,
        depth_max=value_max[:, 2] + depth_slack,
        residual_max_uv=fit.residual_max_uv,
        residual_max_depth=fit.residual_max_depth,
        denominator_min_abs=fit.denominator_min_abs,
    )


def bound_projective_trace_windows(
    windows: list[ProjectiveTraceWindow],
    *,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    require_accepted: bool = True,
) -> list[ProjectiveTraceSupportBounds]:
    return [
        bound_projective_trace_window(
            window,
            uv_padding=uv_padding,
            depth_padding=depth_padding,
            require_accepted=require_accepted,
        )
        for window in windows
    ]


def make_projective_trace_visibility_sidecar(
    window: ProjectiveTraceWindow,
    *,
    chart_gauge_id: int = 0,
    depth_uncertainty_padding: float = 0.0,
    monotonic_eps: float = 1.0e-6,
    require_accepted: bool = True,
) -> ProjectiveTraceVisibilitySidecar:
    if depth_uncertainty_padding < 0.0:
        raise ValueError("depth_uncertainty_padding must be non-negative")
    if monotonic_eps < 0.0:
        raise ValueError("monotonic_eps must be non-negative")
    if require_accepted and not window.accepted:
        raise ValueError(f"cannot make visibility sidecar for unresolved window: {window.reason}")

    fit = window.fit
    depth_coeffs = fit.poly_coeffs[:, 2, :]
    depth_minmax = _normalized_polynomial_range(depth_coeffs.unsqueeze(1))
    depth_slope_min, depth_slope_max = _normalized_polynomial_derivative_range(depth_coeffs, fit.time_scale)
    depth_monotonic_sign = torch.where(
        depth_slope_min > float(monotonic_eps),
        torch.ones_like(depth_slope_min, dtype=torch.int64),
        torch.where(
            depth_slope_max < -float(monotonic_eps),
            -torch.ones_like(depth_slope_min, dtype=torch.int64),
            torch.zeros_like(depth_slope_min, dtype=torch.int64),
        ),
    )

    return ProjectiveTraceVisibilitySidecar(
        start=window.start,
        stop=window.stop,
        chart_gauge_id=chart_gauge_id,
        time_min=fit.time_center - fit.time_scale,
        time_max=fit.time_center + fit.time_scale,
        depth_coeffs=depth_coeffs,
        depth_min=depth_minmax[0].squeeze(1),
        depth_max=depth_minmax[1].squeeze(1),
        depth_slope_min=depth_slope_min,
        depth_slope_max=depth_slope_max,
        depth_monotonic_sign=depth_monotonic_sign,
        depth_uncertainty=fit.residual_max_depth + float(depth_uncertainty_padding),
        denominator_min_abs=fit.denominator_min_abs,
        denominator_has_root=fit.denominator_has_root,
    )


def make_projective_trace_visibility_sidecars(
    windows: list[ProjectiveTraceWindow],
    *,
    chart_gauge_id: int = 0,
    depth_uncertainty_padding: float = 0.0,
    monotonic_eps: float = 1.0e-6,
    require_accepted: bool = True,
) -> list[ProjectiveTraceVisibilitySidecar]:
    return [
        make_projective_trace_visibility_sidecar(
            window,
            chart_gauge_id=chart_gauge_id,
            depth_uncertainty_padding=depth_uncertainty_padding,
            monotonic_eps=monotonic_eps,
            require_accepted=require_accepted,
        )
        for window in windows
    ]


def compare_projective_trace_depth_order(
    a: ProjectiveTraceVisibilitySidecar,
    b: ProjectiveTraceVisibilitySidecar,
    *,
    eps: float = 1.0e-6,
) -> ProjectiveTraceDepthOrder:
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    if a.depth_coeffs.shape != b.depth_coeffs.shape:
        raise ValueError("depth sidecars must have matching primitive dimensions")
    same_center = torch.allclose(a.time_min, b.time_min, atol=float(eps), rtol=0.0) and torch.allclose(a.time_max, b.time_max, atol=float(eps), rtol=0.0)
    if not same_center:
        raise ValueError("depth sidecars must share the same time interval")

    uncertainty = a.depth_uncertainty + b.depth_uncertainty
    a_before_b = (a.depth_max + uncertainty) < (b.depth_min - float(eps))
    b_before_a = (b.depth_max + uncertainty) < (a.depth_min - float(eps))
    crosses = _polynomial_has_root_on_unit_interval(a.depth_coeffs - b.depth_coeffs, eps=eps)
    ambiguous = crosses | ~(a_before_b | b_before_a)
    return ProjectiveTraceDepthOrder(
        a_before_b=a_before_b,
        b_before_a=b_before_a,
        crosses=crosses,
        ambiguous=ambiguous,
    )


def make_projective_trace_appearance_sidecar(
    alpha_max: Tensor,
    color: Tensor,
    *,
    color_radius: float = 0.0,
) -> ProjectiveTraceAppearanceSidecar:
    if alpha_max.ndim != 1:
        raise ValueError("alpha_max must have shape [N]")
    if color.ndim != 2:
        raise ValueError("color must have shape [N,C]")
    if alpha_max.shape[0] != color.shape[0]:
        raise ValueError("alpha_max and color must have the same N")
    if alpha_max.dtype != torch.float32 or color.dtype != torch.float32:
        raise ValueError("alpha_max and color must be float32")
    if alpha_max.device != color.device:
        raise ValueError("alpha_max and color must be on the same device")
    if color_radius < 0.0:
        raise ValueError("color_radius must be non-negative")
    if torch.any(alpha_max < 0.0):
        raise ValueError("alpha_max must be non-negative")

    return ProjectiveTraceAppearanceSidecar(
        alpha_max=alpha_max.contiguous(),
        color_min=(color - float(color_radius)).contiguous(),
        color_max=(color + float(color_radius)).contiguous(),
    )


def bound_projective_trace_visible_swap_cost(
    order: ProjectiveTraceDepthOrder,
    a: ProjectiveTraceAppearanceSidecar,
    b: ProjectiveTraceAppearanceSidecar,
    *,
    threshold: float,
) -> ProjectiveTraceSwapCost:
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    if a.alpha_max.shape != b.alpha_max.shape:
        raise ValueError("appearance sidecars must have matching alpha shapes")
    if a.color_min.shape != b.color_min.shape or a.color_max.shape != b.color_max.shape:
        raise ValueError("appearance sidecars must have matching color shapes")
    if a.color_min.shape[0] != a.alpha_max.shape[0]:
        raise ValueError("appearance sidecars must have matching N")
    if order.ambiguous.shape != a.alpha_max.shape:
        raise ValueError("depth order and appearance sidecars must have matching N")

    color_delta = torch.maximum((a.color_max - b.color_min).abs(), (b.color_max - a.color_min).abs()).amax(dim=1)
    swap_bound = a.alpha_max * b.alpha_max * color_delta
    safely_commutable = order.ambiguous & (swap_bound <= float(threshold))
    needs_fallback = order.ambiguous & ~safely_commutable
    return ProjectiveTraceSwapCost(
        swap_bound=swap_bound,
        safely_commutable=safely_commutable,
        needs_fallback=needs_fallback,
    )


def _to_1d_int_list(values: Tensor | list[int] | tuple[int, ...], *, expected: int, name: str) -> list[int]:
    if isinstance(values, Tensor):
        if values.ndim != 1 or values.numel() != expected:
            raise ValueError(f"{name} must have shape [{expected}]")
        return [int(v) for v in values.detach().cpu().tolist()]
    if len(values) != expected:
        raise ValueError(f"{name} must have length {expected}")
    return [int(v) for v in values]


def _to_1d_bool_list(values: Tensor | list[bool] | tuple[bool, ...], *, expected: int, name: str) -> list[bool]:
    if isinstance(values, Tensor):
        if values.ndim != 1 or values.numel() != expected:
            raise ValueError(f"{name} must have shape [{expected}]")
        return [bool(v) for v in values.detach().cpu().tolist()]
    if len(values) != expected:
        raise ValueError(f"{name} must have length {expected}")
    return [bool(v) for v in values]


def bin_projective_trace_support_bounds(
    bounds: list[ProjectiveTraceSupportBounds],
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    primitive_ids: Tensor | list[int] | tuple[int, ...] | None = None,
    fallback_mask: Tensor | list[bool] | tuple[bool, ...] | None = None,
    fallback_reason: str = "",
) -> list[ProjectiveTraceTileTimeRecord]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not bounds:
        return []

    primitive_count = int(bounds[0].uv_min.shape[0])
    for bound in bounds:
        if bound.uv_min.ndim != 2 or bound.uv_min.shape != bound.uv_max.shape or bound.uv_min.shape[-1] != 2:
            raise ValueError("bound uv tensors must have shape [N,2]")
        if int(bound.uv_min.shape[0]) != primitive_count:
            raise ValueError("all bounds must have the same primitive count")

    ids = list(range(primitive_count)) if primitive_ids is None else _to_1d_int_list(primitive_ids, expected=primitive_count, name="primitive_ids")
    fallback = [False] * primitive_count if fallback_mask is None else _to_1d_bool_list(fallback_mask, expected=primitive_count, name="fallback_mask")
    tile_cols = (image_width + tile_size - 1) // tile_size
    tile_rows = (image_height + tile_size - 1) // tile_size
    records: list[ProjectiveTraceTileTimeRecord] = []

    for window_index, bound in enumerate(bounds):
        uv_min = bound.uv_min.detach().cpu()
        uv_max = bound.uv_max.detach().cpu()
        depth_min = bound.depth_min.detach().cpu()
        depth_max = bound.depth_max.detach().cpu()

        for local_id, primitive_id in enumerate(ids):
            u0 = int(torch.floor(uv_min[local_id, 0] / float(tile_size)).item())
            v0 = int(torch.floor(uv_min[local_id, 1] / float(tile_size)).item())
            u1 = int(torch.floor(uv_max[local_id, 0] / float(tile_size)).item()) + 1
            v1 = int(torch.floor(uv_max[local_id, 1] / float(tile_size)).item()) + 1
            if u1 <= 0 or v1 <= 0 or u0 >= tile_cols or v0 >= tile_rows:
                continue
            records.append(
                ProjectiveTraceTileTimeRecord(
                    primitive_id=primitive_id,
                    window_index=window_index,
                    start=bound.start,
                    stop=bound.stop,
                    tile_u_min=max(0, u0),
                    tile_u_max=min(tile_cols, u1),
                    tile_v_min=max(0, v0),
                    tile_v_max=min(tile_rows, v1),
                    depth_min=float(depth_min[local_id].item()),
                    depth_max=float(depth_max[local_id].item()),
                    fallback=fallback[local_id],
                    fallback_reason=fallback_reason if fallback[local_id] else "",
                )
            )

    return records


def assemble_projective_trace_tile_time_atlas(
    records: list[ProjectiveTraceTileTimeRecord],
) -> list[ProjectiveTraceTileTimeCell]:
    groups: dict[tuple[int, int, int, int], list[ProjectiveTraceTileTimeRecord]] = {}
    for record in records:
        if record.tile_u_min >= record.tile_u_max or record.tile_v_min >= record.tile_v_max:
            raise ValueError("tile ranges must be non-empty")
        if record.start >= record.stop:
            raise ValueError("time ranges must be non-empty")
        for tile_u in range(record.tile_u_min, record.tile_u_max):
            for tile_v in range(record.tile_v_min, record.tile_v_max):
                groups.setdefault((tile_u, tile_v, record.start, record.stop), []).append(record)

    cells: list[ProjectiveTraceTileTimeCell] = []
    for (tile_u, tile_v, start, stop), group in groups.items():
        ordered = sorted(
            group,
            key=lambda record: (
                0.5 * (record.depth_min + record.depth_max),
                record.depth_min,
                record.primitive_id,
            ),
        )
        fallback_reasons = tuple(
            sorted({record.fallback_reason for record in group if record.fallback and record.fallback_reason})
        )
        cells.append(
            ProjectiveTraceTileTimeCell(
                tile_u=tile_u,
                tile_v=tile_v,
                start=start,
                stop=stop,
                primitive_ids=tuple(sorted(record.primitive_id for record in group)),
                ordered_primitive_ids=tuple(record.primitive_id for record in ordered),
                depth_intervals=tuple((record.depth_min, record.depth_max) for record in ordered),
                fallback=any(record.fallback for record in group),
                fallback_reasons=fallback_reasons,
            )
        )

    return sorted(cells, key=lambda cell: (cell.start, cell.stop, cell.tile_v, cell.tile_u))


def render_projective_trace_tile_time_atlas_reference(
    cells: list[ProjectiveTraceTileTimeCell],
    coeffs: Tensor,
    times: Tensor,
    colors: Tensor,
    opacities: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    sigma_px: float,
    primitive_ids: Tensor | list[int] | tuple[int, ...] | None = None,
    alpha_cutoff: float = 0.0,
    transmittance_cutoff: float = 0.0,
    allow_fallback_cells: bool = False,
    fallback_sort_live_depth: bool = True,
    eps: float = 1.0e-6,
) -> Tensor:
    """Small CPU/Torch oracle for atlas-driven compositing.

    This is a correctness helper for the compiler path. It intentionally stays
    simple: tile-time cells provide candidate primitive ids and depth order,
    while opacity is evaluated as an isotropic screen-space Gaussian around the
    dense projective center for each sample.
    """

    _check_projective_trace_render_inputs(coeffs, times, colors, opacities)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if alpha_cutoff < 0.0:
        raise ValueError("alpha_cutoff must be non-negative")
    if transmittance_cutoff < 0.0:
        raise ValueError("transmittance_cutoff must be non-negative")

    primitive_count = int(coeffs.shape[0])
    ids = list(range(primitive_count)) if primitive_ids is None else _to_1d_int_list(primitive_ids, expected=primitive_count, name="primitive_ids")
    id_to_index = {primitive_id: index for index, primitive_id in enumerate(ids)}

    dense = eval_projective_trace_torch(coeffs, times, eps=eps)
    entries_by_key: dict[tuple[int, int, int], list[tuple[int, float, float]]] = {}
    fallback_keys: set[tuple[int, int, int]] = set()
    for cell in cells:
        if cell.tile_u < 0 or cell.tile_v < 0:
            raise ValueError("cell tile coordinates must be non-negative")
        if cell.start < 0 or cell.stop > int(times.numel()) or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for times")
        if len(cell.ordered_primitive_ids) != len(cell.depth_intervals):
            raise ValueError("cell ordered ids and depth intervals must match")
        if cell.fallback and not allow_fallback_cells:
            reason = ",".join(cell.fallback_reasons) if cell.fallback_reasons else "fallback"
            raise ValueError(f"cannot reference-render fallback cell: {reason}")
        for primitive_id in cell.ordered_primitive_ids:
            if primitive_id not in id_to_index:
                raise ValueError("cell primitive id is missing from primitive_ids")
        for sample_index in range(cell.start, cell.stop):
            key = (sample_index, cell.tile_u, cell.tile_v)
            if cell.fallback:
                fallback_keys.add(key)
            entries = entries_by_key.setdefault(key, [])
            for primitive_id, depth_interval in zip(cell.ordered_primitive_ids, cell.depth_intervals):
                depth_min, depth_max = depth_interval
                entries.append((primitive_id, 0.5 * (depth_min + depth_max), depth_min))

    ordered_by_key: dict[tuple[int, int, int], tuple[int, ...]] = {}
    for key, entries in entries_by_key.items():
        if key in fallback_keys and fallback_sort_live_depth:
            sample_index, _tile_u, _tile_v = key

            def _live_depth_key(item: tuple[int, float, float]) -> tuple[float, int]:
                primitive_index = id_to_index[item[0]]
                valid_sign = dense[primitive_index, sample_index, 3]
                if float(valid_sign.item()) == 0.0:
                    return (math.inf, int(item[0]))
                return (float(dense[primitive_index, sample_index, 2].item()), int(item[0]))

            entries.sort(key=_live_depth_key)
        else:
            entries.sort(key=lambda item: (item[1], item[2], item[0]))
        seen: set[int] = set()
        ordered: list[int] = []
        for primitive_id, _depth_mid, _depth_min in entries:
            if primitive_id not in seen:
                seen.add(primitive_id)
                ordered.append(primitive_id)
        ordered_by_key[key] = tuple(ordered)

    tile_cols = (image_width + tile_size - 1) // tile_size
    tile_rows = (image_height + tile_size - 1) // tile_size
    out = torch.zeros(
        (int(times.numel()), image_height, image_width, int(colors.shape[1])),
        dtype=colors.dtype,
        device=colors.device,
    )

    for sample_index in range(int(times.numel())):
        for tile_v in range(tile_rows):
            v0 = tile_v * tile_size
            v1 = min(image_height, v0 + tile_size)
            pixel_v = torch.arange(v0, v1, dtype=colors.dtype, device=colors.device) + 0.5
            for tile_u in range(tile_cols):
                ordered_ids = ordered_by_key.get((sample_index, tile_u, tile_v), ())
                if not ordered_ids:
                    continue
                u0 = tile_u * tile_size
                u1 = min(image_width, u0 + tile_size)
                pixel_u = torch.arange(u0, u1, dtype=colors.dtype, device=colors.device) + 0.5
                du = pixel_u.reshape(1, -1)
                dv = pixel_v.reshape(-1, 1)
                tile_rgb = out[sample_index, v0:v1, u0:u1, :]
                transmittance = torch.ones((v1 - v0, u1 - u0), dtype=colors.dtype, device=colors.device)

                for primitive_id in ordered_ids:
                    primitive_index = id_to_index[primitive_id]
                    center_u = dense[primitive_index, sample_index, 0]
                    center_v = dense[primitive_index, sample_index, 1]
                    valid_sign = dense[primitive_index, sample_index, 3]
                    if float(valid_sign.item()) == 0.0:
                        continue
                    radius2 = (du - center_u).square() + (dv - center_v).square()
                    alpha = opacities[primitive_index] * torch.exp(-0.5 * radius2 / float(sigma_px * sigma_px))
                    alpha = alpha.clamp(0.0, 1.0)
                    if alpha_cutoff > 0.0:
                        alpha = torch.where(alpha >= float(alpha_cutoff), alpha, torch.zeros_like(alpha))
                    tile_rgb += transmittance.unsqueeze(-1) * alpha.unsqueeze(-1) * colors[primitive_index]
                    transmittance = transmittance * (1.0 - alpha)
                    if transmittance_cutoff > 0.0 and bool(torch.all(transmittance <= float(transmittance_cutoff)).item()):
                        break

    return out


def pack_projective_trace_tile_time_bins(
    cells: list[ProjectiveTraceTileTimeCell],
    *,
    image_width: int,
    image_height: int,
    frames: int,
    tile_x: int,
    tile_y: int,
    tile_t: int,
    tile_capacity: int,
    device: torch.device | str | None = None,
    allow_fallback_cells: bool = False,
) -> ProjectiveTraceTileBins:
    """Pack compiler-side projective atlas cells into dense tile-time buffers.

    Each slot stores a trace-table id and an active sample interval. Overlapping
    or touching intervals for that same id are unioned; gaps and distinct chart
    ids remain separate. This preserves the active trace set at every sample
    without spending native capacity on repeated visibility-cell entries.
    """

    if image_width <= 0 or image_height <= 0 or frames <= 0:
        raise ValueError("image dimensions and frames must be positive")
    if tile_x <= 0 or tile_y <= 0 or tile_t <= 0:
        raise ValueError("tile sizes must be positive")
    if tile_capacity <= 0:
        raise ValueError("tile_capacity must be positive")

    tiles_x = (image_width + tile_x - 1) // tile_x
    tiles_y = (image_height + tile_y - 1) // tile_y
    tiles_t = (frames + tile_t - 1) // tile_t
    tile_count = tiles_x * tiles_y * tiles_t
    counts = torch.zeros((tile_count,), dtype=torch.int32)
    primitive_ids = torch.full((tile_count * tile_capacity,), -1, dtype=torch.int32)
    active_start = torch.zeros((tile_count * tile_capacity,), dtype=torch.int32)
    active_stop = torch.zeros((tile_count * tile_capacity,), dtype=torch.int32)
    overflow = torch.zeros((tile_count,), dtype=torch.int32)
    ranges: list[dict[int, list[tuple[int, int]]]] = [{} for _ in range(tile_count)]

    for cell in cells:
        if cell.fallback and not allow_fallback_cells:
            reason = ",".join(cell.fallback_reasons) if cell.fallback_reasons else "fallback"
            raise ValueError(f"cannot pack fallback projective atlas cell: {reason}")
        if cell.tile_u < 0 or cell.tile_u >= tiles_x or cell.tile_v < 0 or cell.tile_v >= tiles_y:
            raise ValueError("cell tile coordinates are outside the image tile grid")
        if cell.start < 0 or cell.stop > frames or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for frames")
        if len(cell.ordered_primitive_ids) != len(cell.depth_intervals):
            raise ValueError("cell ordered ids and depth intervals must match")
        tz0 = cell.start // tile_t
        tz1 = (cell.stop - 1) // tile_t
        for tz in range(tz0, tz1 + 1):
            tile_id = (tz * tiles_y + cell.tile_v) * tiles_x + cell.tile_u
            for primitive_id in cell.ordered_primitive_ids:
                ranges[tile_id].setdefault(int(primitive_id), []).append((int(cell.start), int(cell.stop)))

    for tile_id, trace_ranges in enumerate(ranges):
        entries: list[tuple[int, int, int]] = []
        for primitive_id, intervals in trace_ranges.items():
            for start, stop in sorted(intervals):
                if entries and entries[-1][0] == primitive_id and start <= entries[-1][2]:
                    entries[-1] = (primitive_id, entries[-1][1], max(stop, entries[-1][2]))
                else:
                    entries.append((primitive_id, start, stop))
        counts[tile_id] = len(entries)
        overflow[tile_id] = int(len(entries) > tile_capacity)
        for slot, (primitive_id, start, stop) in enumerate(entries[:tile_capacity]):
            offset = tile_id * tile_capacity + slot
            primitive_ids[offset] = primitive_id
            active_start[offset] = start
            active_stop[offset] = stop

    if device is not None:
        counts = counts.to(device=device)
        primitive_ids = primitive_ids.to(device=device)
        active_start = active_start.to(device=device)
        active_stop = active_stop.to(device=device)
        overflow = overflow.to(device=device)

    return ProjectiveTraceTileBins(
        tile_counts=counts.contiguous(),
        tile_primitive_ids=primitive_ids.contiguous(),
        tile_active_start=active_start.contiguous(),
        tile_active_stop=active_stop.contiguous(),
        tile_overflow=overflow.contiguous(),
    )


def count_projective_trace_dense_per_frame_tile_pairs(
    coeffs: Tensor,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    uv_padding: float,
    eps: float = 1.0e-6,
) -> int:
    """Count ordinary per-frame project-and-bin trace/tile entries.

    This is the denominator for compiler-side frame scaling: it counts the
    support tiles that a time-sliced renderer would touch after evaluating each
    projective trace at every sensor-time sample.
    """

    _check_projective_trace_inputs(coeffs, times)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if uv_padding < 0.0:
        raise ValueError("uv_padding must be non-negative")

    tile_cols = (image_width + tile_size - 1) // tile_size
    tile_rows = (image_height + tile_size - 1) // tile_size
    dense = eval_projective_trace_torch(coeffs, times, eps=eps).detach().cpu()
    total = 0

    for primitive_index in range(int(coeffs.shape[0])):
        for sample_index in range(int(times.numel())):
            valid_sign = float(dense[primitive_index, sample_index, 3].item())
            if valid_sign == 0.0:
                continue
            u = float(dense[primitive_index, sample_index, 0].item())
            v = float(dense[primitive_index, sample_index, 1].item())
            if not math.isfinite(u) or not math.isfinite(v):
                continue
            u0 = math.floor((u - float(uv_padding)) / float(tile_size))
            v0 = math.floor((v - float(uv_padding)) / float(tile_size))
            u1 = math.floor((u + float(uv_padding)) / float(tile_size)) + 1
            v1 = math.floor((v + float(uv_padding)) / float(tile_size)) + 1
            if u1 <= 0 or v1 <= 0 or u0 >= tile_cols or v0 >= tile_rows:
                continue
            total += (min(tile_cols, u1) - max(0, u0)) * (min(tile_rows, v1) - max(0, v0))

    return int(total)


def _normalized_trace_poly_to_raw_time(poly_coeffs: Tensor, center: Tensor, scale: Tensor) -> Tensor:
    if poly_coeffs.ndim != 3 or poly_coeffs.shape[1] != 3 or poly_coeffs.shape[-1] not in (2, 3):
        raise ValueError("poly_coeffs must have shape [N,3,2] or [N,3,3]")
    inv_scale = 1.0 / scale.clamp_min(1.0e-12)
    inv_scale2 = inv_scale.square()
    p0 = poly_coeffs[:, :, 0]
    p1 = poly_coeffs[:, :, 1]
    if poly_coeffs.shape[-1] == 3:
        p2 = poly_coeffs[:, :, 2]
    else:
        p2 = torch.zeros_like(p0)

    raw0 = p0 - p1 * center * inv_scale + p2 * center.square() * inv_scale2
    raw1 = p1 * inv_scale - 2.0 * p2 * center * inv_scale2
    raw2 = p2 * inv_scale2
    return torch.stack((raw0, raw1, raw2), dim=-1).contiguous()


def projective_trace_windows_to_cell_trace_atlas(
    windows: list[ProjectiveTraceWindow],
    *,
    opacity: Tensor,
    color: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    primitive_ids: Tensor | list[int] | tuple[int, ...] | None = None,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    require_accepted: bool = True,
) -> ProjectiveTraceCellTraceAtlas:
    """Lower accepted gauge domains into direct polynomial trace rows.

    The existing rational tile renderer proves that Metal can evaluate a
    projective camera orbit. This helper makes the atlas cell itself the
    evaluable object: each accepted window contributes rows containing local
    raw-time polynomials for ``u(t)``, ``v(t)``, and depth, plus tile-time cells
    whose primitive ids index those rows.
    """

    if not windows:
        raise ValueError("windows must not be empty")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    first_fit = windows[0].fit
    if first_fit.poly_coeffs.ndim != 3 or first_fit.poly_coeffs.shape[1] != 3:
        raise ValueError("window fit poly_coeffs must have shape [N,3,D]")
    primitive_count = int(first_fit.poly_coeffs.shape[0])
    if opacity.shape != (primitive_count,):
        raise ValueError("opacity must have shape [N]")
    if color.ndim != 2 or color.shape[0] != primitive_count:
        raise ValueError("color must have shape [N,C]")
    if opacity.dtype != torch.float32 or color.dtype != torch.float32:
        raise ValueError("opacity and color must be float32")
    if opacity.device != first_fit.poly_coeffs.device or color.device != first_fit.poly_coeffs.device:
        raise ValueError("opacity, color, and windows must be on the same device")

    ids = list(range(primitive_count)) if primitive_ids is None else _to_1d_int_list(primitive_ids, expected=primitive_count, name="primitive_ids")
    coeff_parts: list[Tensor] = []
    opacity_parts: list[Tensor] = []
    color_parts: list[Tensor] = []
    records: list[ProjectiveTraceTileTimeRecord] = []
    source_window_indices: list[int] = []
    source_primitive_ids: list[int] = []
    active_start: list[int] = []
    active_stop: list[int] = []
    trace_offset = 0

    for window_index, window in enumerate(windows):
        if require_accepted and not window.accepted:
            raise ValueError(f"cannot lower unresolved projective trace window to cell traces: {window.reason}")
        fit = window.fit
        if fit.poly_coeffs.shape[0] != primitive_count or fit.poly_coeffs.shape[1] != 3:
            raise ValueError("all windows must have matching primitive dimensions")
        if fit.poly_coeffs.dtype != torch.float32 or fit.poly_coeffs.device != first_fit.poly_coeffs.device:
            raise ValueError("all windows must be float32 on the same device")

        raw_coeffs = _normalized_trace_poly_to_raw_time(fit.poly_coeffs, fit.time_center, fit.time_scale)
        coeff_parts.append(raw_coeffs.reshape(primitive_count, 9).contiguous())
        opacity_parts.append(opacity)
        color_parts.append(color)
        source_window_indices.extend([window_index] * primitive_count)
        source_primitive_ids.extend(ids)
        active_start.extend([window.start] * primitive_count)
        active_stop.extend([window.stop] * primitive_count)

        trace_ids = list(range(trace_offset, trace_offset + primitive_count))
        bound = bound_projective_trace_window(
            window,
            uv_padding=uv_padding,
            depth_padding=depth_padding,
            require_accepted=require_accepted,
        )
        records.extend(
            bin_projective_trace_support_bounds(
                [bound],
                image_width=image_width,
                image_height=image_height,
                tile_size=tile_size,
                primitive_ids=trace_ids,
            )
        )
        trace_offset += primitive_count

    return ProjectiveTraceCellTraceAtlas(
        coeffs=torch.cat(coeff_parts, dim=0).contiguous(),
        opacity=torch.cat(opacity_parts, dim=0).contiguous(),
        color=torch.cat(color_parts, dim=0).contiguous(),
        cells=assemble_projective_trace_tile_time_atlas(records),
        source_window_indices=tuple(source_window_indices),
        source_primitive_ids=tuple(source_primitive_ids),
        active_start=tuple(active_start),
        active_stop=tuple(active_stop),
    )


def uvt_tubes_to_projective_trace_cell_atlas(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    times: Tensor,
    *,
    sigma_px: float,
    image_width: int,
    image_height: int,
    tile_size: int,
    primitive_ids: Tensor | list[int] | tuple[int, ...] | None = None,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    alpha_threshold: float = 0.0,
    require_isotropic_spatial: bool = True,
    auto_support_padding_from_alpha: bool = False,
    spatial_precision_rtol: float = 1.0e-4,
    spatial_precision_atol: float = 1.0e-5,
    depth_spatial_atol: float = 1.0e-6,
    temporal_mode: str = "trace",
    temporal_precision_atol: float = 1.0e-6,
    allow_depth_affine_uv: bool = False,
    root_epsilon: float = 1.0e-6,
    stratify_visibility: bool = True,
    mark_visibility_fallback: bool = False,
    depth_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellTraceAtlas:
    """Lower compatible STAR UVT tubes into a projective interval cell atlas.

    STAR UVT stores each tube as a quadratic form over ``(u,v,t)``. Completing
    the square in the spatial variables gives a moving screen-space center:

    ``x_c(t) = ma_uv - A^{-1} b (t - ma_t)``

    where ``A`` is the UV precision block and ``b`` is the UV-time cross block.
    The interval cell renderer now carries the UV precision block as trace
    metadata. The default producer contract still requires the legacy
    isotropic ``sigma_px`` shape, but callers may opt into anisotropic spatial
    lowering by setting ``require_isotropic_spatial=False``. When
    ``auto_support_padding_from_alpha`` is enabled with a positive
    ``alpha_threshold``, each trace is padded by its own conservative axis
    bounds of the anisotropic alpha ellipse over the supplied sample times.
    Temporal opacity is stored as a
    per-trace quadratic envelope so spacetime tubes do not have to collapse
    their time support into a hard interval gate.
    """

    if ma.ndim != 2 or ma.shape[1] != 3:
        raise ValueError("ma must have shape [N,3]")
    tube_count = int(ma.shape[0])
    if q_uvt.shape != (tube_count, 6):
        raise ValueError("q_uvt must have shape [N,6]")
    if depth0.shape != (tube_count,) or depth_beta.shape != (tube_count, 3):
        raise ValueError("depth0/depth_beta must have shapes [N] and [N,3]")
    if opacity.shape != (tube_count,):
        raise ValueError("opacity must have shape [N]")
    if color.ndim != 2 or color.shape[0] != tube_count:
        raise ValueError("color must have shape [N,C]")
    if times.ndim != 1:
        raise ValueError("times must have shape [S]")
    if times.numel() < 1:
        raise ValueError("times must not be empty")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if uv_padding < 0.0:
        raise ValueError("uv_padding must be non-negative")
    if depth_padding < 0.0:
        raise ValueError("depth_padding must be non-negative")
    if alpha_threshold < 0.0:
        raise ValueError("alpha_threshold must be non-negative")
    if not isinstance(auto_support_padding_from_alpha, bool):
        raise ValueError("auto_support_padding_from_alpha must be boolean")
    if spatial_precision_rtol < 0.0 or spatial_precision_atol < 0.0:
        raise ValueError("spatial precision tolerances must be non-negative")
    if depth_spatial_atol < 0.0:
        raise ValueError("depth_spatial_atol must be non-negative")
    if temporal_precision_atol < 0.0:
        raise ValueError("temporal_precision_atol must be non-negative")
    if not isinstance(allow_depth_affine_uv, bool):
        raise ValueError("allow_depth_affine_uv must be boolean")
    if temporal_mode not in {"trace", "gate", "require_zero"}:
        raise ValueError("temporal_mode must be one of: trace, gate, require_zero")

    tensors = (q_uvt, depth0, depth_beta, opacity, color, times)
    for name, tensor in zip(("q_uvt", "depth0", "depth_beta", "opacity", "color", "times"), tensors):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != ma.device:
            raise ValueError(f"{name} must be on the same device as ma")
    if ma.dtype != torch.float32:
        raise ValueError("ma must be float32")
    if times.numel() > 1 and bool(torch.any(times[1:] < times[:-1]).detach().cpu().item()):
        raise ValueError("times must be sorted in nondecreasing order")

    ids = list(range(tube_count)) if primitive_ids is None else _to_1d_int_list(primitive_ids, expected=tube_count, name="primitive_ids")
    q_uu = q_uvt[:, 0]
    q_uv = q_uvt[:, 1]
    q_ut = q_uvt[:, 2]
    q_vv = q_uvt[:, 3]
    q_vt = q_uvt[:, 4]
    q_tt = q_uvt[:, 5]

    det = q_uu * q_vv - q_uv.square()
    if bool(torch.any(det <= 0.0).detach().cpu().item()):
        raise ValueError("q_uvt spatial UV precision block must be positive definite")
    inv00 = q_vv / det
    inv01 = -q_uv / det
    inv11 = q_uu / det
    a_inv_b_u = inv00 * q_ut + inv01 * q_vt
    a_inv_b_v = inv01 * q_ut + inv11 * q_vt
    velocity_u = -a_inv_b_u
    velocity_v = -a_inv_b_v
    temporal_precision = q_tt - (q_ut * a_inv_b_u + q_vt * a_inv_b_v)
    if bool(torch.any(temporal_precision < -float(temporal_precision_atol)).detach().cpu().item()):
        raise ValueError("q_uvt temporal Schur precision must be non-negative")
    temporal_precision = temporal_precision.clamp_min(0.0)

    if require_isotropic_spatial:
        target_precision = torch.full_like(q_uu, 1.0 / float(sigma_px * sigma_px))
        spatial_ok = (
            torch.isclose(q_uu, target_precision, rtol=float(spatial_precision_rtol), atol=float(spatial_precision_atol))
            & torch.isclose(q_vv, target_precision, rtol=float(spatial_precision_rtol), atol=float(spatial_precision_atol))
            & (q_uv.abs() <= float(spatial_precision_atol))
        )
        if not bool(torch.all(spatial_ok).detach().cpu().item()):
            raise ValueError("q_uvt spatial precision must match the atlas isotropic sigma_px")

    has_depth_spatial = bool(torch.any(depth_beta[:, :2].abs() > float(depth_spatial_atol)).detach().cpu().item())
    if has_depth_spatial and not allow_depth_affine_uv:
        raise ValueError("projective cell atlas lowering requires depth_beta[:,0:2] near zero")
    if temporal_mode == "require_zero" and bool(
        torch.any(temporal_precision > float(temporal_precision_atol)).detach().cpu().item()
    ):
        raise ValueError("temporal_mode='require_zero' requires zero residual temporal precision")

    coeff_parts: list[Tensor] = []
    opacity_parts: list[Tensor] = []
    color_parts: list[Tensor] = []
    opacity_time_coeff_parts: list[Tensor] = []
    spatial_precision_parts: list[Tensor] = []
    depth_affine_parts: list[Tensor] = []
    source_ids: list[int] = []
    source_rows: list[int] = []
    active_start: list[int] = []
    active_stop: list[int] = []

    time_grid = times.reshape(1, -1)
    temporal_envelope = torch.exp(
        -0.5 * temporal_precision.reshape(-1, 1) * (time_grid - ma[:, 2:3]).square()
    )
    effective_opacity = opacity.reshape(-1, 1) * temporal_envelope
    trace_uv_padding = None
    if auto_support_padding_from_alpha and alpha_threshold > 0.0 and tube_count > 0:
        max_effective_opacity = effective_opacity.detach().amax(dim=1)
        radius2 = (
            2.0
            * torch.log(
                (max_effective_opacity / float(alpha_threshold)).clamp_min(1.0)
            )
        ).clamp_min(0.0)
        support_u = torch.sqrt((radius2 * inv00.detach().clamp_min(0.0)).clamp_min(0.0))
        support_v = torch.sqrt((radius2 * inv11.detach().clamp_min(0.0)).clamp_min(0.0))
        trace_uv_padding = torch.stack((support_u, support_v), dim=1)

    for tube_id in range(tube_count):
        if alpha_threshold > 0.0:
            active = effective_opacity[tube_id] >= float(alpha_threshold)
        else:
            active = opacity[tube_id] > 0.0
            active = active.expand_as(times).to(dtype=torch.bool)
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if int(active_indices.numel()) == 0:
            continue
        start = int(active_indices[0].item())
        stop = int(active_indices[-1].item()) + 1
        if stop <= start:
            continue

        u_slope = velocity_u[tube_id]
        v_slope = velocity_v[tube_id]
        t_center = ma[tube_id, 2]
        depth_slope = depth_beta[tube_id, 2] + depth_beta[tube_id, 0] * u_slope + depth_beta[tube_id, 1] * v_slope
        coeff_parts.append(
            torch.stack(
                (
                    ma[tube_id, 0] - u_slope * t_center,
                    u_slope,
                    torch.zeros((), dtype=torch.float32, device=ma.device),
                    ma[tube_id, 1] - v_slope * t_center,
                    v_slope,
                    torch.zeros((), dtype=torch.float32, device=ma.device),
                    depth0[tube_id] - depth_slope * t_center,
                    depth_slope,
                    torch.zeros((), dtype=torch.float32, device=ma.device),
                )
            ).reshape(1, 9)
        )
        opacity_parts.append(opacity[tube_id : tube_id + 1])
        color_parts.append(color[tube_id : tube_id + 1])
        temporal_lambda = temporal_precision[tube_id]
        opacity_time_coeff_parts.append(
            torch.stack(
                (
                    temporal_lambda * t_center.square(),
                    -2.0 * temporal_lambda * t_center,
                    temporal_lambda,
                )
            ).reshape(1, 3)
        )
        spatial_precision_parts.append(torch.stack((q_uu[tube_id], q_uv[tube_id], q_vv[tube_id])).reshape(1, 3))
        if allow_depth_affine_uv:
            depth_affine_parts.append(
                torch.stack(
                    (
                        depth_beta[tube_id, 0],
                        torch.zeros((), dtype=torch.float32, device=ma.device),
                        torch.zeros((), dtype=torch.float32, device=ma.device),
                        depth_beta[tube_id, 1],
                        torch.zeros((), dtype=torch.float32, device=ma.device),
                        torch.zeros((), dtype=torch.float32, device=ma.device),
                    )
                ).reshape(1, 6)
            )
        source_ids.append(ids[tube_id])
        source_rows.append(tube_id)
        active_start.append(start)
        active_stop.append(stop)

    channels = int(color.shape[1])
    if coeff_parts:
        coeffs = torch.cat(coeff_parts, dim=0).contiguous()
        atlas_opacity = torch.cat(opacity_parts, dim=0).contiguous()
        atlas_color = torch.cat(color_parts, dim=0).contiguous()
        opacity_time_coeffs = torch.cat(opacity_time_coeff_parts, dim=0).contiguous()
        spatial_precision_uv = torch.cat(spatial_precision_parts, dim=0).contiguous()
        depth_affine_uv = torch.cat(depth_affine_parts, dim=0).contiguous() if allow_depth_affine_uv else None
    else:
        coeffs = torch.empty((0, 9), dtype=torch.float32, device=ma.device)
        atlas_opacity = torch.empty((0,), dtype=torch.float32, device=ma.device)
        atlas_color = torch.empty((0, channels), dtype=torch.float32, device=ma.device)
        opacity_time_coeffs = torch.empty((0, 3), dtype=torch.float32, device=ma.device)
        spatial_precision_uv = torch.empty((0, 3), dtype=torch.float32, device=ma.device)
        depth_affine_uv = torch.empty((0, 6), dtype=torch.float32, device=ma.device) if allow_depth_affine_uv else None

    atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=atlas_opacity,
        color=atlas_color,
        cells=[],
        source_window_indices=tuple(0 for _ in range(int(coeffs.shape[0]))),
        source_primitive_ids=tuple(source_ids),
        active_start=tuple(active_start),
        active_stop=tuple(active_stop),
        opacity_time_coeffs=opacity_time_coeffs,
        spatial_precision_uv=spatial_precision_uv,
        depth_affine_uv=depth_affine_uv,
        depth_reference_uvt=torch.cat((ma, depth0[:, None], depth_beta), dim=1).index_select(
            0, torch.tensor(source_rows, dtype=torch.long, device=ma.device)
        ).contiguous(),
    )
    atlas = rebin_projective_trace_cell_atlas_support_events(
        atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        uv_padding=uv_padding,
        trace_uv_padding=None if trace_uv_padding is None else trace_uv_padding[source_rows],
        depth_padding=depth_padding,
        root_epsilon=root_epsilon,
    )
    if stratify_visibility:
        atlas = stratify_projective_trace_cell_atlas_visibility_events(
            atlas,
            times,
            root_epsilon=root_epsilon,
        )
    if mark_visibility_fallback:
        atlas = mark_projective_trace_cell_visibility_fallbacks(
            atlas,
            times,
            depth_epsilon=depth_epsilon,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
        )
    return atlas


def _validate_projective_trace_cell_atlas_metadata(atlas: ProjectiveTraceCellTraceAtlas) -> None:
    trace_count = int(atlas.coeffs.shape[0])
    for name, values in (
        ("source_window_indices", atlas.source_window_indices),
        ("source_primitive_ids", atlas.source_primitive_ids),
        ("active_start", atlas.active_start),
        ("active_stop", atlas.active_stop),
    ):
        if len(values) != trace_count:
            raise ValueError(f"{name} must have one entry per cell trace")
    if atlas.opacity_time_coeffs is not None:
        if atlas.opacity_time_coeffs.shape != (trace_count, 3):
            raise ValueError("opacity_time_coeffs must have shape [N,3]")
        if atlas.opacity_time_coeffs.dtype != torch.float32:
            raise ValueError("opacity_time_coeffs must be float32")
        if atlas.opacity_time_coeffs.device != atlas.coeffs.device:
            raise ValueError("opacity_time_coeffs must be on the same device as coeffs")
        if not atlas.opacity_time_coeffs.is_contiguous():
            raise ValueError("opacity_time_coeffs must be contiguous")
    if atlas.spatial_precision_uv is not None:
        if atlas.spatial_precision_uv.shape != (trace_count, 3):
            raise ValueError("spatial_precision_uv must have shape [N,3]")
        if atlas.spatial_precision_uv.dtype != torch.float32:
            raise ValueError("spatial_precision_uv must be float32")
        if atlas.spatial_precision_uv.device != atlas.coeffs.device:
            raise ValueError("spatial_precision_uv must be on the same device as coeffs")
        if not atlas.spatial_precision_uv.is_contiguous():
            raise ValueError("spatial_precision_uv must be contiguous")
        q_uu = atlas.spatial_precision_uv[:, 0]
        q_uv = atlas.spatial_precision_uv[:, 1]
        q_vv = atlas.spatial_precision_uv[:, 2]
        if bool(torch.any((q_uu <= 0.0) | (q_vv <= 0.0) | (q_uu * q_vv - q_uv.square() <= 0.0)).item()):
            raise ValueError("spatial_precision_uv must be positive definite")
    if atlas.depth_affine_uv is not None:
        if atlas.depth_affine_uv.shape != (trace_count, 6):
            raise ValueError("depth_affine_uv must have shape [N,6]")
        if atlas.depth_affine_uv.dtype != torch.float32:
            raise ValueError("depth_affine_uv must be float32")
        if atlas.depth_affine_uv.device != atlas.coeffs.device:
            raise ValueError("depth_affine_uv must be on the same device as coeffs")
        if not atlas.depth_affine_uv.is_contiguous():
            raise ValueError("depth_affine_uv must be contiguous")

    if atlas.depth_reference_uvt is not None:
        value = atlas.depth_reference_uvt
        if value.shape != (trace_count, 7) or value.dtype != torch.float32:
            raise ValueError("depth_reference_uvt must be float32 [N,7]")
        if value.device != atlas.coeffs.device or not value.is_contiguous():
            raise ValueError("depth_reference_uvt must be contiguous on the coeffs device")


def _uvt_reference_depth(reference: Tensor, times: Tensor, u, v) -> Tensor:
    """Use the source-centered expression; expansion can change float32 ties."""
    delta = torch.stack(torch.broadcast_tensors(
        torch.as_tensor(u, dtype=reference.dtype, device=reference.device) - reference[:, 0:1],
        torch.as_tensor(v, dtype=reference.dtype, device=reference.device) - reference[:, 1:2],
        times.reshape(1, -1) - reference[:, 2:3],
    ), dim=-1)
    return reference[:, 3:4] + (reference[:, None, 4:] * delta).sum(dim=-1)


def _validate_uvt_reference_depth_coefficients(atlas: ProjectiveTraceCellTraceAtlas) -> None:
    reference = atlas.depth_reference_uvt
    if reference is None:
        return
    # Direct polynomial optimization must not silently keep old UVT ordering.
    # Source-world optimization rebuilds these two linked representations.
    slope = reference[:, 6] + reference[:, 4] * atlas.coeffs[:, 1] + reference[:, 5] * atlas.coeffs[:, 4]
    zero = torch.zeros_like(slope)
    expected = torch.stack((
        reference[:, 0] - atlas.coeffs[:, 1] * reference[:, 2], atlas.coeffs[:, 1], zero,
        reference[:, 1] - atlas.coeffs[:, 4] * reference[:, 2], atlas.coeffs[:, 4], zero,
        reference[:, 3] - slope * reference[:, 2], slope, zero,
    ), dim=-1)
    matches = torch.equal(expected.detach(), atlas.coeffs.detach())
    if atlas.depth_affine_uv is not None:
        spatial = torch.stack((reference[:, 4], zero, zero, reference[:, 5], zero, zero), dim=-1)
        matches = matches and torch.equal(spatial.detach(), atlas.depth_affine_uv.detach())
    if not matches:
        raise ValueError("UVT depth reference is stale; regenerate the UVT projection, or explicitly remove depth_reference_uvt when changing to polynomial depth")


def _cell_opacity_time_coeffs(atlas: ProjectiveTraceCellTraceAtlas) -> Tensor | None:
    coeffs = atlas.opacity_time_coeffs
    if coeffs is None:
        return None
    _validate_projective_trace_cell_atlas_metadata(atlas)
    return coeffs


def _cell_opacity_time_coeffs_or_zeros(atlas: ProjectiveTraceCellTraceAtlas) -> Tensor:
    coeffs = _cell_opacity_time_coeffs(atlas)
    if coeffs is not None:
        return coeffs
    return torch.zeros(
        (int(atlas.coeffs.shape[0]), 3),
        dtype=atlas.coeffs.dtype,
        device=atlas.coeffs.device,
    )


def _cell_spatial_precision_uv_or_isotropic(atlas: ProjectiveTraceCellTraceAtlas, *, sigma_px: float) -> Tensor:
    _validate_projective_trace_cell_atlas_metadata(atlas)
    if atlas.spatial_precision_uv is not None:
        return atlas.spatial_precision_uv
    inv_sigma2 = 1.0 / float(sigma_px * sigma_px)
    precision = torch.zeros((int(atlas.coeffs.shape[0]), 3), dtype=torch.float32, device=atlas.coeffs.device)
    precision[:, 0] = inv_sigma2
    precision[:, 2] = inv_sigma2
    return precision.contiguous()


def _cell_opacity_time_scale(atlas: ProjectiveTraceCellTraceAtlas, times: Tensor) -> Tensor | None:
    coeffs = _cell_opacity_time_coeffs(atlas)
    if coeffs is None:
        return None
    if times.ndim != 1:
        raise ValueError("times must have shape [S]")
    if times.dtype != torch.float32 or times.device != coeffs.device:
        raise ValueError("times must be float32 and on the same device as opacity_time_coeffs")
    t = times.reshape(1, -1)
    qv = coeffs[:, 0:1] + coeffs[:, 1:2] * t + coeffs[:, 2:3] * t.square()
    return torch.exp(-0.5 * qv)


def _index_select_cell_opacity_time_coeffs(atlas: ProjectiveTraceCellTraceAtlas, index: Tensor) -> Tensor | None:
    coeffs = _cell_opacity_time_coeffs(atlas)
    if coeffs is None:
        return None
    return coeffs.index_select(0, index)


def _index_select_cell_spatial_precision_uv(atlas: ProjectiveTraceCellTraceAtlas, index: Tensor) -> Tensor | None:
    precision = atlas.spatial_precision_uv
    if precision is None:
        return None
    _validate_projective_trace_cell_atlas_metadata(atlas)
    return precision.index_select(0, index)


def _cell_depth_affine_uv(atlas: ProjectiveTraceCellTraceAtlas) -> Tensor | None:
    coeffs = atlas.depth_affine_uv
    if coeffs is None:
        return None
    _validate_projective_trace_cell_atlas_metadata(atlas)
    return coeffs


def _index_select_cell_depth_affine_uv(atlas: ProjectiveTraceCellTraceAtlas, index: Tensor) -> Tensor | None:
    coeffs = _cell_depth_affine_uv(atlas)
    if coeffs is None:
        return None
    return coeffs.index_select(0, index)


def _cell_depth_affine_uv_or_zeros(atlas: ProjectiveTraceCellTraceAtlas) -> Tensor:
    coeffs = _cell_depth_affine_uv(atlas)
    if coeffs is not None:
        return coeffs
    return torch.zeros((int(atlas.coeffs.shape[0]), 6), dtype=torch.float32, device=atlas.coeffs.device)


def _cell_trace_quadratic_radius2(
    atlas: ProjectiveTraceCellTraceAtlas,
    trace_id: int,
    du: Tensor,
    dv: Tensor,
    *,
    sigma_px: float,
) -> Tensor:
    if atlas.spatial_precision_uv is None:
        return (du.square() + dv.square()) / float(sigma_px * sigma_px)
    q_uu = atlas.spatial_precision_uv[int(trace_id), 0].to(dtype=du.dtype)
    q_uv = atlas.spatial_precision_uv[int(trace_id), 1].to(dtype=du.dtype)
    q_vv = atlas.spatial_precision_uv[int(trace_id), 2].to(dtype=du.dtype)
    return q_uu * du.square() + 2.0 * q_uv * du * dv + q_vv * dv.square()


def eval_projective_trace_cell_depth_at_uv_torch(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    u: Tensor | float,
    v: Tensor | float,
) -> Tensor:
    """Evaluate the conditional depth model at screen coordinates.

    The base trace stores center depth in ``coeffs[:,6:9]``. Optional
    ``depth_affine_uv`` stores time-polynomial screen slopes:

    ``[du0, du1, du2, dv0, dv1, dv2]``.

    The evaluated model is

    ``z(u,v,t) = z_c(t) + z_u(t) * (u - u_c(t)) + z_v(t) * (v - v_c(t))``.

    This is a CPU/Torch compiler helper for visibility certificates. The
    current Metal interval renderer still sorts from cell depth metadata.
    """

    _validate_projective_trace_cell_atlas_metadata(atlas)
    if atlas.depth_reference_uvt is not None:
        _validate_uvt_reference_depth_coefficients(atlas)
        return _uvt_reference_depth(atlas.depth_reference_uvt, times, u, v)
    dense = eval_projective_trace_cell_torch(atlas.coeffs, times)
    center_u = dense[:, :, 0]
    center_v = dense[:, :, 1]
    center_depth = dense[:, :, 2]
    depth_affine_uv = _cell_depth_affine_uv(atlas)
    if depth_affine_uv is None:
        return center_depth

    u_tensor = torch.as_tensor(u, dtype=center_depth.dtype, device=center_depth.device)
    v_tensor = torch.as_tensor(v, dtype=center_depth.dtype, device=center_depth.device)
    t = times.to(dtype=center_depth.dtype, device=center_depth.device).reshape(1, -1)
    slope_u = depth_affine_uv[:, 0:1] + depth_affine_uv[:, 1:2] * t + depth_affine_uv[:, 2:3] * t.square()
    slope_v = depth_affine_uv[:, 3:4] + depth_affine_uv[:, 4:5] * t + depth_affine_uv[:, 5:6] * t.square()
    return center_depth + slope_u * (u_tensor - center_u) + slope_v * (v_tensor - center_v)


def _cell_has_nonzero_depth_affine_uv(atlas: ProjectiveTraceCellTraceAtlas) -> bool:
    if atlas.depth_reference_uvt is not None:
        return bool(torch.any(atlas.depth_reference_uvt[:, 4:6].detach() != 0).cpu().item())
    coeffs = _cell_depth_affine_uv(atlas)
    if coeffs is None:
        return False
    return bool(torch.any(coeffs.detach().abs() > 0.0).cpu().item())


def _validate_optional_tile_depth_domain(
    *,
    image_width: int | None,
    image_height: int | None,
    tile_size: int | None,
) -> bool:
    provided = (image_width is not None, image_height is not None, tile_size is not None)
    if any(provided) and not all(provided):
        raise ValueError("image_width, image_height, and tile_size must be provided together")
    if image_width is None or image_height is None or tile_size is None:
        return False
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    return True


def _cell_trace_depth_at_uv_sample(
    dense: Tensor,
    depth_affine_uv: Tensor | None,
    times: Tensor,
    *,
    trace_id: int,
    sample_index: int,
    u: float,
    v: float,
) -> float:
    center_depth = dense[int(trace_id), int(sample_index), 2]
    if depth_affine_uv is None:
        return float(center_depth.item())
    t = times[int(sample_index)].to(dtype=center_depth.dtype, device=center_depth.device)
    coeffs = depth_affine_uv[int(trace_id)].to(dtype=center_depth.dtype, device=center_depth.device)
    slope_u = coeffs[0] + coeffs[1] * t + coeffs[2] * t * t
    slope_v = coeffs[3] + coeffs[4] * t + coeffs[5] * t * t
    center_u = dense[int(trace_id), int(sample_index), 0]
    center_v = dense[int(trace_id), int(sample_index), 1]
    depth = center_depth + slope_u * (float(u) - center_u) + slope_v * (float(v) - center_v)
    return float(depth.item())


def _cell_trace_depth_line_for_sample(
    dense: Tensor,
    depth_affine_uv: Tensor | None,
    times: Tensor,
    *,
    trace_id: int,
    sample_index: int,
) -> tuple[float, float, float]:
    center_depth = dense[int(trace_id), int(sample_index), 2]
    if depth_affine_uv is None:
        return 0.0, 0.0, float(center_depth.item())
    t = times[int(sample_index)].to(dtype=center_depth.dtype, device=center_depth.device)
    coeffs = depth_affine_uv[int(trace_id)].to(dtype=center_depth.dtype, device=center_depth.device)
    slope_u = coeffs[0] + coeffs[1] * t + coeffs[2] * t * t
    slope_v = coeffs[3] + coeffs[4] * t + coeffs[5] * t * t
    center_u = dense[int(trace_id), int(sample_index), 0]
    center_v = dense[int(trace_id), int(sample_index), 1]
    line_0 = center_depth - slope_u * center_u - slope_v * center_v
    return float(slope_u.item()), float(slope_v.item()), float(line_0.item())


def _affine_line_range_for_tile(
    line_u: float,
    line_v: float,
    line_0: float,
    *,
    tile_u: int,
    tile_v: int,
    image_width: int,
    image_height: int,
    tile_size: int,
) -> tuple[float, float]:
    u0 = int(tile_u) * int(tile_size)
    v0 = int(tile_v) * int(tile_size)
    u1_exclusive = min(int(image_width), u0 + int(tile_size))
    v1_exclusive = min(int(image_height), v0 + int(tile_size))
    if u0 < 0 or v0 < 0 or u0 >= int(image_width) or v0 >= int(image_height):
        raise ValueError("cell tile coordinates are outside the image tile grid")
    if u1_exclusive <= u0 or v1_exclusive <= v0:
        raise ValueError("cell tile is empty")
    u_min = float(u0) + 0.5
    u_max = float(u1_exclusive) - 0.5
    v_min = float(v0) + 0.5
    v_max = float(v1_exclusive) - 0.5
    values = [
        float(line_u) * u + float(line_v) * v + float(line_0)
        for u, v in ((u_min, v_min), (u_min, v_max), (u_max, v_min), (u_max, v_max))
    ]
    return min(values), max(values)


def _cell_trace_depth_range_for_tile_sample(
    dense: Tensor,
    depth_affine_uv: Tensor | None,
    times: Tensor,
    *,
    trace_id: int,
    sample_index: int,
    tile_u: int,
    tile_v: int,
    image_width: int | None,
    image_height: int | None,
    tile_size: int | None,
) -> tuple[float, float]:
    if (
        depth_affine_uv is None
        or image_width is None
        or image_height is None
        or tile_size is None
    ):
        depth = float(dense[int(trace_id), int(sample_index), 2].item())
        return depth, depth
    u0 = int(tile_u) * int(tile_size)
    v0 = int(tile_v) * int(tile_size)
    u1_exclusive = min(int(image_width), u0 + int(tile_size))
    v1_exclusive = min(int(image_height), v0 + int(tile_size))
    if u0 < 0 or v0 < 0 or u0 >= int(image_width) or v0 >= int(image_height):
        raise ValueError("cell tile coordinates are outside the image tile grid")
    if u1_exclusive <= u0 or v1_exclusive <= v0:
        raise ValueError("cell tile is empty")
    u_min = float(u0) + 0.5
    u_max = float(u1_exclusive) - 0.5
    v_min = float(v0) + 0.5
    v_max = float(v1_exclusive) - 0.5
    depths = [
        _cell_trace_depth_at_uv_sample(
            dense,
            depth_affine_uv,
            times,
            trace_id=trace_id,
            sample_index=sample_index,
            u=u,
            v=v,
        )
        for u, v in ((u_min, v_min), (u_min, v_max), (u_max, v_min), (u_max, v_max))
    ]
    return min(depths), max(depths)


def _cell_atlas_has_nonzero_temporal_opacity(atlas: ProjectiveTraceCellTraceAtlas) -> bool:
    coeffs = _cell_opacity_time_coeffs(atlas)
    if coeffs is None:
        return False
    return bool(torch.any(coeffs.detach().abs() > 0.0).cpu().item())


def _require_no_temporal_opacity_for_cell_metal(atlas: ProjectiveTraceCellTraceAtlas) -> None:
    if _cell_atlas_has_nonzero_temporal_opacity(atlas):
        raise ValueError("projective cell Metal render does not yet support opacity_time_coeffs")


def _projective_trace_support_tiles(
    *,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    image_width: int,
    image_height: int,
    tile_size: int,
) -> tuple[int, int, int, int] | None:
    tile_cols = (image_width + tile_size - 1) // tile_size
    tile_rows = (image_height + tile_size - 1) // tile_size
    tile_u_min = int(math.floor(float(u_min) / float(tile_size)))
    tile_u_max = int(math.floor(float(u_max) / float(tile_size))) + 1
    tile_v_min = int(math.floor(float(v_min) / float(tile_size)))
    tile_v_max = int(math.floor(float(v_max) / float(tile_size))) + 1
    if tile_u_max <= 0 or tile_v_max <= 0 or tile_u_min >= tile_cols or tile_v_min >= tile_rows:
        return None
    return (
        max(0, tile_u_min),
        min(tile_cols, tile_u_max),
        max(0, tile_v_min),
        min(tile_rows, tile_v_max),
    )


def projective_trace_cell_atlas_coverage_report(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    uv_padding: float = 0.0,
    max_missing_examples: int = 16,
) -> ProjectiveTraceCellAtlasCoverageReport:
    """Check whether compiled tile cells still cover live cell-trace samples.

    Cell-trace coefficients are differentiable and may move during training,
    while tile-time membership is compiled metadata. This report is the cheap
    lifecycle guard: it detects when the current trace rows visit frame/tile
    pairs that are absent from the packed atlas cells.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if uv_padding < 0.0:
        raise ValueError("uv_padding must be non-negative")
    if max_missing_examples < 0:
        raise ValueError("max_missing_examples must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    times_cpu = times.detach().cpu().contiguous()
    dense = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
    frame_count = int(times_cpu.numel())
    trace_count = int(coeffs_cpu.shape[0])

    covered: set[tuple[int, int, int, int]] = set()
    for cell in atlas.cells:
        if cell.tile_u < 0 or cell.tile_v < 0:
            raise ValueError("cell tile coordinates must be non-negative")
        if cell.start >= cell.stop:
            raise ValueError("cell time ranges must be non-empty")
        start = max(0, int(cell.start))
        stop = min(frame_count, int(cell.stop))
        if start >= stop:
            continue
        for trace_id in cell.primitive_ids:
            if trace_id < 0 or trace_id >= trace_count:
                raise ValueError("cell trace id is outside the atlas coeff table")
            for sample_index in range(start, stop):
                covered.add((int(trace_id), sample_index, int(cell.tile_u), int(cell.tile_v)))

    checked_tile_pairs = 0
    missing_tile_pairs = 0
    invalid_active_samples = 0
    missing_examples: list[tuple[int, int, int, int]] = []
    for trace_id in range(trace_count):
        active_start = int(atlas.active_start[trace_id])
        active_stop = int(atlas.active_stop[trace_id])
        if active_start < 0 or active_stop > frame_count or active_start >= active_stop:
            invalid_active_samples += 1
        start = max(0, active_start)
        stop = min(frame_count, active_stop)
        if start >= stop:
            continue
        for sample_index in range(start, stop):
            u, v, _depth, valid_sign = dense[trace_id, sample_index]
            if float(valid_sign.item()) == 0.0:
                invalid_active_samples += 1
                continue
            tile_range = _projective_trace_support_tiles(
                u_min=float(u.item()) - float(uv_padding),
                u_max=float(u.item()) + float(uv_padding),
                v_min=float(v.item()) - float(uv_padding),
                v_max=float(v.item()) + float(uv_padding),
                image_width=image_width,
                image_height=image_height,
                tile_size=tile_size,
            )
            if tile_range is None:
                continue
            tile_u_min, tile_u_max, tile_v_min, tile_v_max = tile_range
            for tile_u in range(tile_u_min, tile_u_max):
                for tile_v in range(tile_v_min, tile_v_max):
                    checked_tile_pairs += 1
                    key = (trace_id, sample_index, tile_u, tile_v)
                    if key in covered:
                        continue
                    missing_tile_pairs += 1
                    if len(missing_examples) < max_missing_examples:
                        missing_examples.append(key)

    return ProjectiveTraceCellAtlasCoverageReport(
        stale=missing_tile_pairs > 0 or invalid_active_samples > 0,
        checked_tile_pairs=checked_tile_pairs,
        missing_tile_pairs=missing_tile_pairs,
        invalid_active_samples=invalid_active_samples,
        missing_examples=tuple(missing_examples),
    )


def projective_trace_cell_atlas_support_margin_report(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    uv_padding: float = 0.0,
    max_missing_examples: int = 16,
) -> ProjectiveTraceCellAtlasSupportMarginReport:
    """Quantify how far live support moved beyond compiled tile cells."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if uv_padding < 0.0:
        raise ValueError("uv_padding must be non-negative")
    if max_missing_examples < 0:
        raise ValueError("max_missing_examples must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    times_cpu = times.detach().cpu().contiguous()
    dense = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
    frame_count = int(times_cpu.numel())
    trace_count = int(coeffs_cpu.shape[0])

    covered_by_sample: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for cell in atlas.cells:
        if cell.tile_u < 0 or cell.tile_v < 0:
            raise ValueError("cell tile coordinates must be non-negative")
        if cell.start >= cell.stop:
            raise ValueError("cell time ranges must be non-empty")
        start = max(0, int(cell.start))
        stop = min(frame_count, int(cell.stop))
        if start >= stop:
            continue
        for trace_id in cell.primitive_ids:
            if trace_id < 0 or trace_id >= trace_count:
                raise ValueError("cell trace id is outside the atlas coeff table")
            for sample_index in range(start, stop):
                covered_by_sample.setdefault((int(trace_id), sample_index), set()).add(
                    (int(cell.tile_u), int(cell.tile_v))
                )

    checked_tile_pairs = 0
    missing_tile_pairs = 0
    invalid_active_samples = 0
    missing_without_covered_sample = 0
    boundary_slacks: list[float] = []
    overshoots: list[float] = []
    missing_examples: list[tuple[int, int, int, int, float]] = []

    for trace_id in range(trace_count):
        active_start = int(atlas.active_start[trace_id])
        active_stop = int(atlas.active_stop[trace_id])
        if active_start < 0 or active_stop > frame_count or active_start >= active_stop:
            invalid_active_samples += 1
        start = max(0, active_start)
        stop = min(frame_count, active_stop)
        if start >= stop:
            continue
        for sample_index in range(start, stop):
            u, v, _depth, valid_sign = dense[trace_id, sample_index]
            if float(valid_sign.item()) == 0.0:
                invalid_active_samples += 1
                continue
            u_min = float(u.item()) - float(uv_padding)
            u_max = float(u.item()) + float(uv_padding)
            v_min = float(v.item()) - float(uv_padding)
            v_max = float(v.item()) + float(uv_padding)
            tile_range = _projective_trace_support_tiles(
                u_min=u_min,
                u_max=u_max,
                v_min=v_min,
                v_max=v_max,
                image_width=image_width,
                image_height=image_height,
                tile_size=tile_size,
            )
            if tile_range is None:
                continue
            tile_u_min, tile_u_max, tile_v_min, tile_v_max = tile_range
            covered_tiles = covered_by_sample.get((trace_id, sample_index), set())
            if covered_tiles:
                covered_u = [tile_u for tile_u, _tile_v in covered_tiles]
                covered_v = [tile_v for _tile_u, tile_v in covered_tiles]
                covered_u_min = min(covered_u)
                covered_u_max = max(covered_u)
                covered_v_min = min(covered_v)
                covered_v_max = max(covered_v)
                boundary_slacks.append(
                    min(
                        u_min - float(covered_u_min * tile_size),
                        float((covered_u_max + 1) * tile_size) - u_max,
                        v_min - float(covered_v_min * tile_size),
                        float((covered_v_max + 1) * tile_size) - v_max,
                    )
                )
            else:
                covered_u_min = covered_u_max = covered_v_min = covered_v_max = 0
                boundary_slacks.append(-float(tile_size))
            for tile_u in range(tile_u_min, tile_u_max):
                for tile_v in range(tile_v_min, tile_v_max):
                    checked_tile_pairs += 1
                    if (tile_u, tile_v) in covered_tiles:
                        continue
                    missing_tile_pairs += 1
                    if not covered_tiles:
                        missing_without_covered_sample += 1
                        overshoot = float(tile_size)
                    else:
                        u_overshoot = 0.0
                        if tile_u < covered_u_min:
                            u_overshoot = float(covered_u_min * tile_size) - u_min
                        elif tile_u > covered_u_max:
                            u_overshoot = u_max - float((covered_u_max + 1) * tile_size)
                        v_overshoot = 0.0
                        if tile_v < covered_v_min:
                            v_overshoot = float(covered_v_min * tile_size) - v_min
                        elif tile_v > covered_v_max:
                            v_overshoot = v_max - float((covered_v_max + 1) * tile_size)
                        overshoot = max(0.0, u_overshoot, v_overshoot)
                    overshoots.append(float(overshoot))
                    if len(missing_examples) < max_missing_examples:
                        missing_examples.append((trace_id, sample_index, tile_u, tile_v, float(overshoot)))

    if overshoots:
        sorted_overshoots = sorted(overshoots)
        p95_index = min(len(sorted_overshoots) - 1, int(math.ceil(0.95 * len(sorted_overshoots))) - 1)
        max_overshoot = float(sorted_overshoots[-1])
        mean_overshoot = float(sum(sorted_overshoots) / len(sorted_overshoots))
        p95_overshoot = float(sorted_overshoots[p95_index])
    else:
        max_overshoot = 0.0
        mean_overshoot = 0.0
        p95_overshoot = 0.0
    if boundary_slacks:
        sorted_slacks = sorted(boundary_slacks)
        p05_index = min(len(sorted_slacks) - 1, max(0, int(math.floor(0.05 * len(sorted_slacks)))))
        min_slack = float(sorted_slacks[0])
        mean_slack = float(sum(sorted_slacks) / len(sorted_slacks))
        p05_slack = float(sorted_slacks[p05_index])
    else:
        min_slack = 0.0
        mean_slack = 0.0
        p05_slack = 0.0

    return ProjectiveTraceCellAtlasSupportMarginReport(
        stale=missing_tile_pairs > 0 or invalid_active_samples > 0,
        checked_tile_pairs=checked_tile_pairs,
        missing_tile_pairs=missing_tile_pairs,
        invalid_active_samples=invalid_active_samples,
        missing_without_covered_sample=missing_without_covered_sample,
        min_boundary_slack_px=min_slack,
        mean_boundary_slack_px=mean_slack,
        p05_boundary_slack_px=p05_slack,
        max_boundary_overshoot_px=max_overshoot,
        mean_boundary_overshoot_px=mean_overshoot,
        p95_boundary_overshoot_px=p95_overshoot,
        missing_examples=tuple(missing_examples),
    )


def projective_trace_cell_atlas_visibility_report(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    depth_epsilon: float = 1.0e-6,
    image_width: int | None = None,
    image_height: int | None = None,
    tile_size: int | None = None,
    mark_ambiguous_stale: bool = True,
    max_mismatch_examples: int = 16,
) -> ProjectiveTraceCellAtlasVisibilityReport:
    """Check whether compiled cell depth order matches live trace depths.

    Tile membership and order are static metadata during a single
    forward/backward pass. This report catches the complementary stale case to
    coverage: support may still be correct, but coefficient updates can move
    conditional depths enough that the stored front-to-back order is no longer
    the live per-sample order.
    """

    if depth_epsilon < 0.0:
        raise ValueError("depth_epsilon must be non-negative")
    if max_mismatch_examples < 0:
        raise ValueError("max_mismatch_examples must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)
    has_tile_depth_domain = _validate_optional_tile_depth_domain(
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    times_cpu = times.detach().cpu().contiguous()
    dense = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
    depth_affine_uv_cpu = (
        None
        if atlas.depth_affine_uv is None or not has_tile_depth_domain
        else atlas.depth_affine_uv.detach().cpu().contiguous()
    )
    frame_count = int(times_cpu.numel())
    trace_count = int(coeffs_cpu.shape[0])

    entries_by_key: dict[tuple[int, int, int], list[tuple[int, float, float, int]]] = {}
    fallback_keys: set[tuple[int, int, int]] = set()
    for cell_index, cell in enumerate(atlas.cells):
        if cell.tile_u < 0 or cell.tile_v < 0:
            raise ValueError("cell tile coordinates must be non-negative")
        if cell.start < 0 or cell.stop > frame_count or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for times")
        if len(cell.ordered_primitive_ids) != len(cell.depth_intervals):
            raise ValueError("cell ordered ids and depth intervals must match")
        for trace_id in cell.ordered_primitive_ids:
            if trace_id < 0 or trace_id >= trace_count:
                raise ValueError("cell trace id is outside the atlas coeff table")
        for sample_index in range(cell.start, cell.stop):
            key = (sample_index, int(cell.tile_u), int(cell.tile_v))
            if cell.fallback:
                fallback_keys.add(key)
            entries = entries_by_key.setdefault(key, [])
            for trace_id, depth_interval in zip(cell.ordered_primitive_ids, cell.depth_intervals):
                depth_min, depth_max = depth_interval
                entries.append((int(trace_id), 0.5 * (float(depth_min) + float(depth_max)), float(depth_min), cell_index))

    checked_tile_samples = 0
    order_mismatch_samples = 0
    ambiguous_depth_samples = 0
    invalid_depth_samples = 0
    mismatch_examples: list[tuple[int, int, int, int, int]] = []
    ambiguous_examples: list[tuple[int, int, int, int, int]] = []
    for (sample_index, tile_u, tile_v), entries in entries_by_key.items():
        entries.sort(key=lambda item: (item[1], item[2], item[0]))
        seen: set[int] = set()
        stored_order: list[int] = []
        for trace_id, _depth_mid, _depth_min, _cell_index in entries:
            if trace_id not in seen:
                seen.add(trace_id)
                stored_order.append(trace_id)

        live_depths: list[tuple[float, float, float, int]] = []
        stored_active: list[int] = []
        for trace_id in stored_order:
            active_start = int(atlas.active_start[trace_id])
            active_stop = int(atlas.active_stop[trace_id])
            if not (active_start <= sample_index < active_stop):
                continue
            valid_sign = dense[trace_id, sample_index, 3]
            if float(valid_sign.item()) == 0.0:
                invalid_depth_samples += 1
                continue
            stored_active.append(trace_id)
            depth_min, depth_max = _cell_trace_depth_range_for_tile_sample(
                dense,
                depth_affine_uv_cpu,
                times_cpu,
                trace_id=trace_id,
                sample_index=sample_index,
                tile_u=tile_u,
                tile_v=tile_v,
                image_width=image_width,
                image_height=image_height,
                tile_size=tile_size,
            )
            live_depths.append((0.5 * (depth_min + depth_max), depth_min, depth_max, trace_id))

        if len(live_depths) < 2:
            continue
        checked_tile_samples += 1
        sorted_ranges = sorted(live_depths, key=lambda item: (item[1], item[2], item[3]))
        for (_mid_a, _min_a, max_a, trace_a), (_mid_b, min_b, _max_b, trace_b) in zip(sorted_ranges, sorted_ranges[1:]):
            if max_a + float(depth_epsilon) >= min_b:
                ambiguous_depth_samples += 1
                if len(ambiguous_examples) < max_mismatch_examples:
                    ambiguous_examples.append((sample_index, tile_u, tile_v, trace_a, trace_b))
                break
        if (sample_index, tile_u, tile_v) in fallback_keys and not mark_ambiguous_stale:
            continue
        sorted_depths = sorted(live_depths, key=lambda item: (item[0], item[1], item[3]))
        live_order = tuple(trace_id for _mid, _min_depth, _max_depth, trace_id in sorted_depths)
        if tuple(stored_active) == live_order:
            continue
        order_mismatch_samples += 1
        if len(mismatch_examples) < max_mismatch_examples:
            mismatch_examples.append((sample_index, tile_u, tile_v, stored_active[0], live_order[0]))

    return ProjectiveTraceCellAtlasVisibilityReport(
        stale=(
            order_mismatch_samples > 0
            or invalid_depth_samples > 0
            or (mark_ambiguous_stale and ambiguous_depth_samples > 0)
        ),
        checked_tile_samples=checked_tile_samples,
        order_mismatch_samples=order_mismatch_samples,
        ambiguous_depth_samples=ambiguous_depth_samples,
        invalid_depth_samples=invalid_depth_samples,
        mismatch_examples=tuple(mismatch_examples),
        ambiguous_examples=tuple(ambiguous_examples),
    )


def projective_trace_cell_atlas_fallback_stats(
    atlas: ProjectiveTraceCellTraceAtlas,
) -> ProjectiveTraceCellAtlasFallbackStats:
    """Summarize how much of a compiled cell atlas requires fallback evaluation."""

    _validate_projective_trace_cell_atlas_metadata(atlas)

    total_tile_keys: set[tuple[int, int, int]] = set()
    fallback_tile_keys: set[tuple[int, int, int]] = set()
    total_trace_samples = 0
    fallback_trace_samples = 0
    reasons: set[str] = set()

    for cell in atlas.cells:
        trace_samples = (int(cell.stop) - int(cell.start)) * len(cell.ordered_primitive_ids)
        total_trace_samples += trace_samples
        if cell.fallback:
            fallback_trace_samples += trace_samples
            reasons.update(cell.fallback_reasons)
        for sample_index in range(int(cell.start), int(cell.stop)):
            key = (sample_index, int(cell.tile_u), int(cell.tile_v))
            total_tile_keys.add(key)
            if cell.fallback:
                fallback_tile_keys.add(key)

    total_tile_samples = len(total_tile_keys)
    fallback_tile_samples = len(fallback_tile_keys)
    fallback_fraction = (
        float(fallback_tile_samples) / float(total_tile_samples)
        if total_tile_samples > 0
        else 0.0
    )

    return ProjectiveTraceCellAtlasFallbackStats(
        total_cells=len(atlas.cells),
        fallback_cells=sum(1 for cell in atlas.cells if cell.fallback),
        total_tile_samples=total_tile_samples,
        fallback_tile_samples=fallback_tile_samples,
        total_trace_samples=total_trace_samples,
        fallback_trace_samples=fallback_trace_samples,
        fallback_fraction=fallback_fraction,
        fallback_reasons=tuple(sorted(reasons)),
    )


def projective_trace_cell_atlas_complexity_stats(
    atlas: ProjectiveTraceCellTraceAtlas,
) -> ProjectiveTraceCellAtlasComplexityStats:
    """Measure stored cell topology and fallback, before native interval union.

    ``interval_trace_entries`` counts retained cell entries, not packed slots;
    the native packer's ``tile_counts`` measures its coalesced work separately.
    """

    fallback = projective_trace_cell_atlas_fallback_stats(atlas)
    groups: dict[tuple[int, int, tuple[int, ...]], int] = {}
    interval_trace_entries = 0
    dense_trace_samples = 0
    for cell in atlas.cells:
        group_key = (int(cell.tile_u), int(cell.tile_v), tuple(sorted(cell.primitive_ids)))
        groups[group_key] = groups.get(group_key, 0) + 1
        entry_count = len(cell.ordered_primitive_ids)
        interval_trace_entries += entry_count
        dense_trace_samples += (int(cell.stop) - int(cell.start)) * entry_count

    max_cells_per_group = max(groups.values(), default=0)
    split_cells = sum(max(0, count - 1) for count in groups.values())
    ratio = (
        float(interval_trace_entries) / float(dense_trace_samples)
        if dense_trace_samples > 0
        else 0.0
    )

    return ProjectiveTraceCellAtlasComplexityStats(
        total_cells=len(atlas.cells),
        tile_active_set_groups=len(groups),
        visibility_stratum_split_cells=split_cells,
        max_cells_per_active_set_group=max_cells_per_group,
        interval_trace_entries=interval_trace_entries,
        dense_trace_samples=dense_trace_samples,
        interval_to_dense_trace_sample_ratio=ratio,
        fallback_cells=fallback.fallback_cells,
        fallback_fraction=fallback.fallback_fraction,
    )


def projective_trace_cell_atlas_budget_report(
    atlas: ProjectiveTraceCellTraceAtlas,
    *,
    max_interval_to_dense_trace_sample_ratio: float = 1.0,
    max_fallback_fraction: float = 0.20,
    max_cells_per_active_set_group: int = 16,
) -> ProjectiveTraceCellAtlasBudgetReport:
    """Check whether atlas interval/visibility complexity is still controlled."""

    if max_interval_to_dense_trace_sample_ratio < 0.0:
        raise ValueError("max_interval_to_dense_trace_sample_ratio must be non-negative")
    if max_fallback_fraction < 0.0:
        raise ValueError("max_fallback_fraction must be non-negative")
    if max_cells_per_active_set_group < 1:
        raise ValueError("max_cells_per_active_set_group must be positive")

    stats = projective_trace_cell_atlas_complexity_stats(atlas)
    failures: list[str] = []
    if stats.interval_to_dense_trace_sample_ratio > float(max_interval_to_dense_trace_sample_ratio):
        failures.append("interval_to_dense_trace_sample_ratio")
    if stats.fallback_fraction > float(max_fallback_fraction):
        failures.append("fallback_fraction")
    if stats.max_cells_per_active_set_group > int(max_cells_per_active_set_group):
        failures.append("max_cells_per_active_set_group")

    return ProjectiveTraceCellAtlasBudgetReport(
        within_budget=not failures,
        stats=stats,
        failures=tuple(failures),
        max_interval_to_dense_trace_sample_ratio=float(max_interval_to_dense_trace_sample_ratio),
        max_fallback_fraction=float(max_fallback_fraction),
        max_cells_per_active_set_group=int(max_cells_per_active_set_group),
    )


def _quadratic_roots_in_closed_interval(
    *,
    c0: float,
    c1: float,
    c2: float,
    t_min: float,
    t_max: float,
    eps: float,
) -> tuple[float, ...]:
    if t_max < t_min:
        return ()
    roots: list[float] = []
    if abs(c2) <= eps:
        if abs(c1) <= eps:
            return ()
        roots.append(-c0 / c1)
    else:
        discriminant = c1 * c1 - 4.0 * c2 * c0
        if discriminant < -eps:
            return ()
        sqrt_discriminant = math.sqrt(max(0.0, discriminant))
        denom = 2.0 * c2
        roots.extend(((-c1 - sqrt_discriminant) / denom, (-c1 + sqrt_discriminant) / denom))

    accepted: list[float] = []
    for root in sorted(roots):
        if root < t_min - eps or root > t_max + eps:
            continue
        clamped = min(max(root, t_min), t_max)
        if not any(abs(clamped - existing) <= eps for existing in accepted):
            accepted.append(clamped)
    return tuple(accepted)


def _projective_trace_cell_pair_depth_roots(
    coeffs_cpu: Tensor,
    *,
    trace_a: int,
    trace_b: int,
    t_min: float,
    t_max: float,
    eps: float,
) -> tuple[float, ...]:
    diff = coeffs_cpu[trace_a, 6:9] - coeffs_cpu[trace_b, 6:9]
    return _quadratic_roots_in_closed_interval(
        c0=float(diff[0].item()),
        c1=float(diff[1].item()),
        c2=float(diff[2].item()),
        t_min=t_min,
        t_max=t_max,
        eps=eps,
    )


def _axis_support_sides(uv_padding: float) -> tuple[tuple[str, float], ...]:
    if uv_padding == 0.0:
        return (("center", 0.0),)
    return (("min", -float(uv_padding)), ("max", float(uv_padding)))


def _cell_trace_axis_roots_for_boundary(
    coeffs_cpu: Tensor,
    *,
    trace_id: int,
    axis_offset: int,
    signed_padding: float,
    boundary: float,
    t_min: float,
    t_max: float,
    eps: float,
) -> tuple[float, ...]:
    coeff = coeffs_cpu[trace_id, axis_offset : axis_offset + 3]
    return _quadratic_roots_in_closed_interval(
        c0=float(coeff[0].item()) + float(signed_padding) - float(boundary),
        c1=float(coeff[1].item()),
        c2=float(coeff[2].item()),
        t_min=t_min,
        t_max=t_max,
        eps=eps,
    )


def _axis_tile_boundaries(*, image_extent: int, tile_size: int) -> tuple[float, ...]:
    boundaries = {0.0, float(image_extent)}
    boundary = float(tile_size)
    while boundary < float(image_extent):
        boundaries.add(boundary)
        boundary += float(tile_size)
    return tuple(sorted(boundaries))


def _quadratic_value_range_over_interval(
    coeff: Tensor,
    *,
    t_min: float,
    t_max: float,
) -> tuple[float, float]:
    values = [
        float((coeff[0] + coeff[1] * t_min + coeff[2] * t_min * t_min).item()),
        float((coeff[0] + coeff[1] * t_max + coeff[2] * t_max * t_max).item()),
    ]
    c2 = float(coeff[2].item())
    if c2 != 0.0:
        vertex = -float(coeff[1].item()) / (2.0 * c2)
        if t_min <= vertex <= t_max:
            values.append(float((coeff[0] + coeff[1] * vertex + coeff[2] * vertex * vertex).item()))
    return min(values), max(values)


def _add_sample_boundary_from_root(
    boundaries: set[int],
    times_cpu: Tensor,
    *,
    start: int,
    stop: int,
    root: float,
    eps: float,
) -> None:
    root_tensor = torch.tensor(float(root), dtype=times_cpu.dtype)
    left = int(torch.searchsorted(times_cpu, root_tensor, right=False).item())
    right = int(torch.searchsorted(times_cpu, root_tensor, right=True).item())
    boundary = left
    for candidate in (left, right - 1):
        if 0 <= candidate < int(times_cpu.numel()) and abs(float(times_cpu[candidate].item()) - float(root)) <= float(eps):
            boundary = candidate
            break
    if start < boundary < stop:
        boundaries.add(boundary)


def projective_trace_cell_support_event_report(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    uv_padding: float = 0.0,
    root_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellSupportEventReport:
    """Find continuous screen/tile support boundary events for cell traces."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if uv_padding < 0.0:
        raise ValueError("uv_padding must be non-negative")
    if root_epsilon < 0.0:
        raise ValueError("root_epsilon must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)

    times_cpu = times.detach().cpu().contiguous()
    if times_cpu.ndim != 1:
        raise ValueError("times must have shape [S]")
    if int(times_cpu.numel()) < 1:
        raise ValueError("times must not be empty")
    if int(times_cpu.numel()) > 1 and bool(torch.any(times_cpu[1:] < times_cpu[:-1]).item()):
        raise ValueError("times must be sorted in nondecreasing order")

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    frame_count = int(times_cpu.numel())
    events: list[ProjectiveTraceCellSupportEvent] = []
    split_times: list[float] = []
    axes = (
        ("u", 0, _axis_tile_boundaries(image_extent=image_width, tile_size=tile_size)),
        ("v", 3, _axis_tile_boundaries(image_extent=image_height, tile_size=tile_size)),
    )

    for trace_id in range(int(coeffs_cpu.shape[0])):
        active_start = int(atlas.active_start[trace_id])
        active_stop = int(atlas.active_stop[trace_id])
        if active_start < 0 or active_stop > frame_count or active_start >= active_stop:
            raise ValueError("active intervals must be valid for the supplied times")
        if active_stop - active_start < 2:
            continue
        t_min = float(times_cpu[active_start].item())
        t_max = float(times_cpu[active_stop - 1].item())
        for axis, axis_offset, boundaries in axes:
            for side, signed_padding in _axis_support_sides(float(uv_padding)):
                for boundary in boundaries:
                    roots = _cell_trace_axis_roots_for_boundary(
                        coeffs_cpu,
                        trace_id=trace_id,
                        axis_offset=axis_offset,
                        signed_padding=signed_padding,
                        boundary=boundary,
                        t_min=t_min,
                        t_max=t_max,
                        eps=float(root_epsilon),
                    )
                    for root in roots:
                        events.append(
                            ProjectiveTraceCellSupportEvent(
                                trace_id=trace_id,
                                axis=axis,
                                side=side,
                                boundary=float(boundary),
                                time=float(root),
                            )
                        )
                        if not any(abs(float(root) - existing) <= float(root_epsilon) for existing in split_times):
                            split_times.append(float(root))

    return ProjectiveTraceCellSupportEventReport(
        events=tuple(sorted(events, key=lambda event: (event.time, event.trace_id, event.axis, event.boundary, event.side))),
        split_times=tuple(sorted(split_times)),
    )


def projective_trace_cell_visibility_event_report(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    root_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellVisibilityEventReport:
    """Find continuous depth-order event times for cell-local trace pairs."""

    if root_epsilon < 0.0:
        raise ValueError("root_epsilon must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)

    times_cpu = times.detach().cpu().contiguous()
    if times_cpu.ndim != 1:
        raise ValueError("times must have shape [S]")
    if int(times_cpu.numel()) < 1:
        raise ValueError("times must not be empty")
    if int(times_cpu.numel()) > 1 and bool(torch.any(times_cpu[1:] < times_cpu[:-1]).item()):
        raise ValueError("times must be sorted in nondecreasing order")

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    trace_count = int(coeffs_cpu.shape[0])
    frame_count = int(times_cpu.numel())
    events: list[ProjectiveTraceCellVisibilityEvent] = []
    split_times: list[float] = []

    for cell_index, cell in enumerate(atlas.cells):
        if cell.start < 0 or cell.stop > frame_count or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for times")
        trace_ids = tuple(sorted(set(int(trace_id) for trace_id in cell.primitive_ids)))
        for trace_id in trace_ids:
            if trace_id < 0 or trace_id >= trace_count:
                raise ValueError("cell trace id is outside the atlas coeff table")
        for pair_i, trace_a in enumerate(trace_ids):
            for trace_b in trace_ids[pair_i + 1:]:
                overlap_start = max(int(cell.start), int(atlas.active_start[trace_a]), int(atlas.active_start[trace_b]))
                overlap_stop = min(int(cell.stop), int(atlas.active_stop[trace_a]), int(atlas.active_stop[trace_b]))
                if overlap_stop - overlap_start < 2:
                    continue
                t_min = float(times_cpu[overlap_start].item())
                t_max = float(times_cpu[overlap_stop - 1].item())
                roots = _projective_trace_cell_pair_depth_roots(
                    coeffs_cpu,
                    trace_a=trace_a,
                    trace_b=trace_b,
                    t_min=t_min,
                    t_max=t_max,
                    eps=float(root_epsilon),
                )
                for root in roots:
                    events.append(
                        ProjectiveTraceCellVisibilityEvent(
                            cell_index=cell_index,
                            tile_u=int(cell.tile_u),
                            tile_v=int(cell.tile_v),
                            trace_a=int(trace_a),
                            trace_b=int(trace_b),
                            time=float(root),
                        )
                    )
                    if not any(abs(float(root) - existing) <= float(root_epsilon) for existing in split_times):
                        split_times.append(float(root))

    return ProjectiveTraceCellVisibilityEventReport(
        events=tuple(sorted(events, key=lambda event: (event.time, event.cell_index, event.trace_a, event.trace_b))),
        split_times=tuple(sorted(split_times)),
    )


def projective_trace_cell_uv_visibility_event_report(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    depth_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellUVVisibilityEventReport:
    """Find in-tile depth-order boundaries induced by UV-varying depth.

    Time-root visibility events split the sensor-time axis. A nonzero
    ``depth_affine_uv`` can also make the pairwise order boundary an affine
    line inside a single image tile at a fixed sample. This report names those
    UV events so the compiler can distinguish "stable over tile" from "needs a
    spatial split or fallback" without hiding the reason inside a generic
    ambiguous-depth flag.
    """

    if depth_epsilon < 0.0:
        raise ValueError("depth_epsilon must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)
    _validate_optional_tile_depth_domain(
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )

    times_cpu = times.detach().cpu().contiguous()
    if times_cpu.ndim != 1:
        raise ValueError("times must have shape [S]")
    if int(times_cpu.numel()) < 1:
        raise ValueError("times must not be empty")
    if int(times_cpu.numel()) > 1 and bool(torch.any(times_cpu[1:] < times_cpu[:-1]).item()):
        raise ValueError("times must be sorted in nondecreasing order")

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    dense = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
    depth_affine_uv_cpu = None if atlas.depth_affine_uv is None else atlas.depth_affine_uv.detach().cpu().contiguous()
    trace_count = int(coeffs_cpu.shape[0])
    frame_count = int(times_cpu.numel())
    events: list[ProjectiveTraceCellUVVisibilityEvent] = []
    event_keys: set[tuple[int, int, int]] = set()

    for cell_index, cell in enumerate(atlas.cells):
        if cell.tile_u < 0 or cell.tile_v < 0:
            raise ValueError("cell tile coordinates must be non-negative")
        if cell.start < 0 or cell.stop > frame_count or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for times")
        trace_ids = tuple(sorted(set(int(trace_id) for trace_id in cell.primitive_ids)))
        for trace_id in trace_ids:
            if trace_id < 0 or trace_id >= trace_count:
                raise ValueError("cell trace id is outside the atlas coeff table")
        for pair_i, trace_a in enumerate(trace_ids):
            for trace_b in trace_ids[pair_i + 1:]:
                overlap_start = max(int(cell.start), int(atlas.active_start[trace_a]), int(atlas.active_start[trace_b]))
                overlap_stop = min(int(cell.stop), int(atlas.active_stop[trace_a]), int(atlas.active_stop[trace_b]))
                for sample_index in range(overlap_start, overlap_stop):
                    if float(dense[trace_a, sample_index, 3].item()) == 0.0:
                        continue
                    if float(dense[trace_b, sample_index, 3].item()) == 0.0:
                        continue
                    au, av, a0 = _cell_trace_depth_line_for_sample(
                        dense,
                        depth_affine_uv_cpu,
                        times_cpu,
                        trace_id=trace_a,
                        sample_index=sample_index,
                    )
                    bu, bv, b0 = _cell_trace_depth_line_for_sample(
                        dense,
                        depth_affine_uv_cpu,
                        times_cpu,
                        trace_id=trace_b,
                        sample_index=sample_index,
                    )
                    line_u = au - bu
                    line_v = av - bv
                    line_0 = a0 - b0
                    min_delta, max_delta = _affine_line_range_for_tile(
                        line_u,
                        line_v,
                        line_0,
                        tile_u=int(cell.tile_u),
                        tile_v=int(cell.tile_v),
                        image_width=int(image_width),
                        image_height=int(image_height),
                        tile_size=int(tile_size),
                    )
                    if min_delta > float(depth_epsilon) or max_delta < -float(depth_epsilon):
                        continue
                    event_keys.add((sample_index, int(cell.tile_u), int(cell.tile_v)))
                    events.append(
                        ProjectiveTraceCellUVVisibilityEvent(
                            cell_index=cell_index,
                            tile_u=int(cell.tile_u),
                            tile_v=int(cell.tile_v),
                            sample_index=sample_index,
                            trace_a=int(trace_a),
                            trace_b=int(trace_b),
                            time=float(times_cpu[sample_index].item()),
                            line_u=float(line_u),
                            line_v=float(line_v),
                            line_0=float(line_0),
                            min_delta=float(min_delta),
                            max_delta=float(max_delta),
                        )
                    )

    return ProjectiveTraceCellUVVisibilityEventReport(
        events=tuple(
            sorted(
                events,
                key=lambda event: (
                    event.time,
                    event.cell_index,
                    event.tile_v,
                    event.tile_u,
                    event.trace_a,
                    event.trace_b,
                ),
            )
        ),
        event_tile_samples=len(event_keys),
    )


def split_projective_trace_cell_atlas_uv_visibility_events(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    child_tile_size: int,
) -> ProjectiveTraceCellTraceAtlas:
    """Retile a cell atlas onto a finer grid for UV depth-order events.

    The first spatial split representation uses the existing cell schema:
    replace each parent tile cell by child tile cells on a finer global grid,
    then recompute per-child depth intervals and front-to-back order. A UV
    zero line that crossed the parent tile can become stable inside each child
    tile, avoiding fallback without adding an oblique-polygon renderer yet.
    Callers should run ``mark_projective_trace_cell_visibility_fallbacks`` on
    the returned atlas with ``tile_size=child_tile_size``; child tiles whose
    order boundary still crosses their footprint remain fallback candidates.
    """

    _validate_projective_trace_cell_atlas_metadata(atlas)
    _validate_optional_tile_depth_domain(
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )
    if child_tile_size <= 0:
        raise ValueError("child_tile_size must be positive")
    if child_tile_size > tile_size:
        raise ValueError("child_tile_size must be no larger than tile_size")
    if tile_size % child_tile_size != 0:
        raise ValueError("child_tile_size must evenly divide tile_size")
    if child_tile_size == tile_size:
        return atlas

    times_cpu = times.detach().cpu().contiguous()
    if times_cpu.ndim != 1:
        raise ValueError("times must have shape [S]")
    if int(times_cpu.numel()) < 1:
        raise ValueError("times must not be empty")
    if int(times_cpu.numel()) > 1 and bool(torch.any(times_cpu[1:] < times_cpu[:-1]).item()):
        raise ValueError("times must be sorted in nondecreasing order")

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    dense = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
    depth_affine_uv_cpu = None if atlas.depth_affine_uv is None else atlas.depth_affine_uv.detach().cpu().contiguous()
    frame_count = int(times_cpu.numel())
    trace_count = int(coeffs_cpu.shape[0])
    cells: list[ProjectiveTraceTileTimeCell] = []

    visibility_reasons = {"visibility_ambiguous_depth", "visibility_uv_depth_line"}

    for cell in atlas.cells:
        if cell.tile_u < 0 or cell.tile_v < 0:
            raise ValueError("cell tile coordinates must be non-negative")
        if cell.start < 0 or cell.stop > frame_count or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for times")
        trace_ids = tuple(sorted(set(int(trace_id) for trace_id in cell.primitive_ids)))
        for trace_id in trace_ids:
            if trace_id < 0 or trace_id >= trace_count:
                raise ValueError("cell trace id is outside the atlas coeff table")

        parent_u0 = int(cell.tile_u) * int(tile_size)
        parent_v0 = int(cell.tile_v) * int(tile_size)
        parent_u1 = min(int(image_width), parent_u0 + int(tile_size))
        parent_v1 = min(int(image_height), parent_v0 + int(tile_size))
        if parent_u0 < 0 or parent_v0 < 0 or parent_u0 >= int(image_width) or parent_v0 >= int(image_height):
            raise ValueError("cell tile coordinates are outside the image tile grid")
        if parent_u1 <= parent_u0 or parent_v1 <= parent_v0:
            raise ValueError("cell tile is empty")

        child_u0 = parent_u0 // int(child_tile_size)
        child_u1 = (parent_u1 - 1) // int(child_tile_size)
        child_v0 = parent_v0 // int(child_tile_size)
        child_v1 = (parent_v1 - 1) // int(child_tile_size)
        for child_v in range(child_v0, child_v1 + 1):
            for child_u in range(child_u0, child_u1 + 1):
                ordered_depths: list[tuple[float, float, int, tuple[float, float]]] = []
                for trace_id in trace_ids:
                    sample_start = max(int(cell.start), int(atlas.active_start[trace_id]))
                    sample_stop = min(int(cell.stop), int(atlas.active_stop[trace_id]))
                    if sample_start >= sample_stop:
                        continue
                    depth_min = math.inf
                    depth_max = -math.inf
                    for sample_index in range(sample_start, sample_stop):
                        if float(dense[trace_id, sample_index, 3].item()) == 0.0:
                            continue
                        sample_min, sample_max = _cell_trace_depth_range_for_tile_sample(
                            dense,
                            depth_affine_uv_cpu,
                            times_cpu,
                            trace_id=trace_id,
                            sample_index=sample_index,
                            tile_u=child_u,
                            tile_v=child_v,
                            image_width=image_width,
                            image_height=image_height,
                            tile_size=child_tile_size,
                        )
                        depth_min = min(depth_min, sample_min)
                        depth_max = max(depth_max, sample_max)
                    if not math.isfinite(depth_min) or not math.isfinite(depth_max):
                        continue
                    depth_mid = 0.5 * (depth_min + depth_max)
                    ordered_depths.append((depth_mid, depth_min, int(trace_id), (depth_min, depth_max)))
                if not ordered_depths:
                    continue

                ordered_depths.sort(key=lambda item: (item[0], item[1], item[2]))
                child_reasons = tuple(
                    reason
                    for reason in cell.fallback_reasons
                    if reason not in visibility_reasons
                )
                child_fallback = bool(cell.fallback and (child_reasons or not cell.fallback_reasons))
                cells.append(
                    ProjectiveTraceTileTimeCell(
                        tile_u=int(child_u),
                        tile_v=int(child_v),
                        start=int(cell.start),
                        stop=int(cell.stop),
                        primitive_ids=tuple(sorted(item[2] for item in ordered_depths)),
                        ordered_primitive_ids=tuple(item[2] for item in ordered_depths),
                        depth_intervals=tuple(item[3] for item in ordered_depths),
                        fallback=child_fallback,
                        fallback_reasons=child_reasons if child_fallback else (),
                    )
                )

    return replace(atlas, cells=sorted(cells, key=lambda cell: (cell.start, cell.stop, cell.tile_v, cell.tile_u)))


def _uv_visibility_child_tile_candidates(
    *,
    tile_size: int,
    min_child_tile_size: int,
) -> tuple[int, ...]:
    if min_child_tile_size <= 0:
        raise ValueError("min_child_tile_size must be positive")
    if min_child_tile_size > tile_size:
        raise ValueError("min_child_tile_size must be no larger than tile_size")
    return tuple(
        size
        for size in range(int(tile_size) - 1, int(min_child_tile_size) - 1, -1)
        if int(tile_size) % size == 0
    )


def adapt_projective_trace_cell_atlas_uv_visibility_events(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    min_child_tile_size: int = 1,
    max_residual_event_tile_samples: int = 0,
    depth_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellUVVisibilitySpatialSplitReport:
    """Choose a grid-refinement split for UV visibility events.

    The current cell atlas uses one global tile size per render pass. This
    policy therefore keeps the parent atlas when no UV order line crosses a
    parent tile, and otherwise tries child-grid retile sizes from largest to
    smallest. The first child grid whose residual UV event count is within the
    supplied budget is accepted. If no candidate clears the budget, the best
    candidate is returned with explicit fallback markings for unresolved child
    cells.
    """

    if max_residual_event_tile_samples < 0:
        raise ValueError("max_residual_event_tile_samples must be non-negative")
    parent_report = projective_trace_cell_uv_visibility_event_report(
        atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        depth_epsilon=depth_epsilon,
    )
    parent_marked = mark_projective_trace_cell_visibility_fallbacks(
        atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        depth_epsilon=depth_epsilon,
    )
    parent_stats = projective_trace_cell_atlas_fallback_stats(parent_marked)
    candidates = _uv_visibility_child_tile_candidates(
        tile_size=int(tile_size),
        min_child_tile_size=int(min_child_tile_size),
    )
    if parent_report.event_tile_samples == 0 or not candidates:
        return ProjectiveTraceCellUVVisibilitySpatialSplitReport(
            atlas=parent_marked,
            accepted=(parent_report.event_tile_samples <= int(max_residual_event_tile_samples)),
            split_attempted=False,
            input_tile_size=int(tile_size),
            output_tile_size=int(tile_size),
            candidate_tile_sizes=candidates,
            parent_cells=len(parent_marked.cells),
            parent_uv_events=len(parent_report.events),
            parent_uv_event_tile_samples=parent_report.event_tile_samples,
            parent_fallback_cells=parent_stats.fallback_cells,
            parent_fallback_fraction=parent_stats.fallback_fraction,
            residual_uv_events=len(parent_report.events),
            residual_uv_event_tile_samples=parent_report.event_tile_samples,
            output_cells=len(parent_marked.cells),
            fallback_cells=parent_stats.fallback_cells,
            fallback_fraction=parent_stats.fallback_fraction,
        )

    best: ProjectiveTraceCellUVVisibilitySpatialSplitReport | None = None
    for child_tile_size in candidates:
        split = split_projective_trace_cell_atlas_uv_visibility_events(
            atlas,
            times,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            child_tile_size=int(child_tile_size),
        )
        residual_report = projective_trace_cell_uv_visibility_event_report(
            split,
            times,
            image_width=image_width,
            image_height=image_height,
            tile_size=int(child_tile_size),
            depth_epsilon=depth_epsilon,
        )
        marked = mark_projective_trace_cell_visibility_fallbacks(
            split,
            times,
            image_width=image_width,
            image_height=image_height,
            tile_size=int(child_tile_size),
            depth_epsilon=depth_epsilon,
        )
        stats = projective_trace_cell_atlas_fallback_stats(marked)
        report = ProjectiveTraceCellUVVisibilitySpatialSplitReport(
            atlas=marked,
            accepted=(residual_report.event_tile_samples <= int(max_residual_event_tile_samples)),
            split_attempted=True,
            input_tile_size=int(tile_size),
            output_tile_size=int(child_tile_size),
            candidate_tile_sizes=candidates,
            parent_cells=len(parent_marked.cells),
            parent_uv_events=len(parent_report.events),
            parent_uv_event_tile_samples=parent_report.event_tile_samples,
            parent_fallback_cells=parent_stats.fallback_cells,
            parent_fallback_fraction=parent_stats.fallback_fraction,
            residual_uv_events=len(residual_report.events),
            residual_uv_event_tile_samples=residual_report.event_tile_samples,
            output_cells=len(marked.cells),
            fallback_cells=stats.fallback_cells,
            fallback_fraction=stats.fallback_fraction,
        )
        if best is None:
            best = report
        else:
            best_key = (
                best.residual_uv_event_tile_samples,
                best.fallback_fraction,
                best.output_cells,
            )
            report_key = (
                report.residual_uv_event_tile_samples,
                report.fallback_fraction,
                report.output_cells,
            )
            if report_key < best_key:
                best = report
        if report.accepted:
            return report

    if best is None:
        raise RuntimeError("UV visibility split policy produced no candidate")
    return best


def _merge_split_times(
    values: list[float],
    *,
    domain_min: float,
    domain_max: float,
    eps: float,
) -> tuple[float, ...]:
    accepted: list[float] = []
    for value in sorted(values):
        if value < domain_min - eps or value > domain_max + eps:
            continue
        clamped = min(max(float(value), domain_min), domain_max)
        if not any(abs(clamped - existing) <= eps for existing in accepted):
            accepted.append(clamped)
    return tuple(accepted)


def projective_trace_cell_sensor_time_event_partition(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    uv_padding: float = 0.0,
    include_support: bool = True,
    include_visibility: bool = True,
    extra_split_times: tuple[float, ...] | list[float] = (),
    root_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellSensorTimePartition:
    """Combine support, visibility, and caller-supplied sensor-time events.

    The returned intervals are continuous sensor-time intervals, not frame-index
    cells. They are the compiler target for finite exposure, rolling shutter,
    and smooth camera orbits: frame/sample indices can be derived later from
    these event-cell boundaries.
    """

    if not include_support and not include_visibility and not extra_split_times:
        raise ValueError("at least one event source or extra split time is required")
    if root_epsilon < 0.0:
        raise ValueError("root_epsilon must be non-negative")
    times_cpu = times.detach().cpu().contiguous()
    if times_cpu.ndim != 1:
        raise ValueError("times must have shape [S]")
    if int(times_cpu.numel()) < 1:
        raise ValueError("times must not be empty")
    if int(times_cpu.numel()) > 1 and bool(torch.any(times_cpu[1:] < times_cpu[:-1]).item()):
        raise ValueError("times must be sorted in nondecreasing order")

    domain_min = float(times_cpu[0].item())
    domain_max = float(times_cpu[-1].item())
    split_candidates: list[float] = [domain_min, domain_max, *(float(value) for value in extra_split_times)]
    support_events: tuple[ProjectiveTraceCellSupportEvent, ...] = ()
    visibility_events: tuple[ProjectiveTraceCellVisibilityEvent, ...] = ()

    if include_support:
        support_report = projective_trace_cell_support_event_report(
            atlas,
            times,
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
            uv_padding=uv_padding,
            root_epsilon=root_epsilon,
        )
        support_events = support_report.events
        split_candidates.extend(support_report.split_times)

    if include_visibility:
        visibility_report = projective_trace_cell_visibility_event_report(
            atlas,
            times,
            root_epsilon=root_epsilon,
        )
        visibility_events = visibility_report.events
        split_candidates.extend(visibility_report.split_times)

    split_times = _merge_split_times(
        split_candidates,
        domain_min=domain_min,
        domain_max=domain_max,
        eps=float(root_epsilon),
    )
    intervals = tuple(
        ProjectiveTraceCellSensorTimeInterval(start_time=float(start), stop_time=float(stop))
        for start, stop in zip(split_times, split_times[1:])
        if stop > start
    )
    return ProjectiveTraceCellSensorTimePartition(
        intervals=intervals,
        split_times=split_times,
        support_events=support_events,
        visibility_events=visibility_events,
    )


def projective_trace_cell_sensor_time_partition_quadrature(
    partition: ProjectiveTraceCellSensorTimePartition,
    *,
    exposure_start: float,
    exposure_stop: float,
    samples_per_interval: int = 1,
    row_index: int = -1,
    normalize_weights: bool = True,
) -> ProjectiveTraceCellSensorTimeQuadrature:
    """Lower a continuous sensor-time partition to weighted midpoint samples."""

    if exposure_stop <= exposure_start:
        raise ValueError("exposure_stop must be greater than exposure_start")
    if samples_per_interval < 1:
        raise ValueError("samples_per_interval must be positive")

    samples: list[ProjectiveTraceCellSensorTimeQuadratureSample] = []
    denominator = float(exposure_stop - exposure_start) if normalize_weights else 1.0
    for interval_index, interval in enumerate(partition.intervals):
        clipped_start = max(float(interval.start_time), float(exposure_start))
        clipped_stop = min(float(interval.stop_time), float(exposure_stop))
        if clipped_stop <= clipped_start:
            continue
        step = (clipped_stop - clipped_start) / float(samples_per_interval)
        weight = step / denominator
        for sample_index in range(int(samples_per_interval)):
            sample_start = clipped_start + float(sample_index) * step
            sample_stop = sample_start + step
            samples.append(
                ProjectiveTraceCellSensorTimeQuadratureSample(
                    interval_index=interval_index,
                    row_index=int(row_index),
                    start_time=float(sample_start),
                    stop_time=float(sample_stop),
                    time=float(0.5 * (sample_start + sample_stop)),
                    weight=float(weight),
                )
            )

    total_weight = float(sum(sample.weight for sample in samples))
    return ProjectiveTraceCellSensorTimeQuadrature(
        samples=tuple(samples),
        total_weight=total_weight,
    )


def projective_trace_cell_sensor_time_partition_rolling_quadrature(
    partition: ProjectiveTraceCellSensorTimePartition,
    *,
    row_count: int,
    frame_time: float,
    exposure_duration: float,
    readout_duration: float,
    samples_per_interval: int = 1,
    normalize_weights: bool = True,
) -> tuple[ProjectiveTraceCellSensorTimeQuadrature, ...]:
    """Lower a partition to per-row rolling-shutter exposure quadrature."""

    if row_count < 1:
        raise ValueError("row_count must be positive")
    if exposure_duration <= 0.0:
        raise ValueError("exposure_duration must be positive")
    if readout_duration < 0.0:
        raise ValueError("readout_duration must be non-negative")

    schedules: list[ProjectiveTraceCellSensorTimeQuadrature] = []
    denom = max(1, row_count - 1)
    for row_index in range(int(row_count)):
        row_offset = 0.0 if row_count == 1 else float(readout_duration) * float(row_index) / float(denom)
        row_start = float(frame_time) + row_offset
        row_stop = row_start + float(exposure_duration)
        schedules.append(
            projective_trace_cell_sensor_time_partition_quadrature(
                partition,
                exposure_start=row_start,
                exposure_stop=row_stop,
                samples_per_interval=samples_per_interval,
                row_index=row_index,
                normalize_weights=normalize_weights,
            )
        )
    return tuple(schedules)


def stratify_projective_trace_cell_atlas_visibility_events(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    root_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellTraceAtlas:
    """Split cell intervals at continuous depth-order roots.

    This is the continuous counterpart to the sampled stratum compiler below.
    It uses the cell-local depth polynomials to cut each tile-time cell at
    pairwise roots of ``z_i(t)-z_j(t)`` and at active-span boundaries, then
    stores the stable midpoint order for each resulting interval. Exact roots
    that land on a frame sample are isolated as singleton sample intervals, so
    later fallback marking can cover only the tie sample instead of the whole
    neighboring run.
    """

    if root_epsilon < 0.0:
        raise ValueError("root_epsilon must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    times_cpu = times.detach().cpu().contiguous()
    if times_cpu.ndim != 1:
        raise ValueError("times must have shape [S]")
    if int(times_cpu.numel()) < 1:
        raise ValueError("times must not be empty")
    if int(times_cpu.numel()) > 1 and bool(torch.any(times_cpu[1:] < times_cpu[:-1]).item()):
        raise ValueError("times must be sorted in nondecreasing order")

    dense = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
    frame_count = int(times_cpu.numel())
    trace_count = int(coeffs_cpu.shape[0])
    cells: list[ProjectiveTraceTileTimeCell] = []

    def _depth_at(trace_id: int, t_value: float) -> float:
        coeff = coeffs_cpu[trace_id, 6:9]
        return float((coeff[0] + coeff[1] * t_value + coeff[2] * t_value * t_value).item())

    def _add_boundary(boundaries: set[int], cell: ProjectiveTraceTileTimeCell, boundary: int) -> None:
        if int(cell.start) < boundary < int(cell.stop):
            boundaries.add(boundary)

    def _add_root_boundaries(boundaries: set[int], cell: ProjectiveTraceTileTimeCell, root: float) -> None:
        root_tensor = torch.tensor(float(root), dtype=times_cpu.dtype)
        left = int(torch.searchsorted(times_cpu, root_tensor, right=False).item())
        right = int(torch.searchsorted(times_cpu, root_tensor, right=True).item())
        exact_index: int | None = None
        for candidate in (left, right - 1):
            if 0 <= candidate < frame_count and abs(float(times_cpu[candidate].item()) - float(root)) <= float(root_epsilon):
                exact_index = candidate
                break
        if exact_index is not None:
            _add_boundary(boundaries, cell, exact_index)
            _add_boundary(boundaries, cell, exact_index + 1)
            return
        _add_boundary(boundaries, cell, left)

    for cell in atlas.cells:
        if cell.tile_u < 0 or cell.tile_v < 0:
            raise ValueError("cell tile coordinates must be non-negative")
        if cell.start < 0 or cell.stop > frame_count or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for times")
        trace_ids = tuple(sorted(set(int(trace_id) for trace_id in cell.primitive_ids)))
        for trace_id in trace_ids:
            if trace_id < 0 or trace_id >= trace_count:
                raise ValueError("cell trace id is outside the atlas coeff table")

        boundaries: set[int] = {int(cell.start), int(cell.stop)}
        for trace_id in trace_ids:
            _add_boundary(boundaries, cell, int(atlas.active_start[trace_id]))
            _add_boundary(boundaries, cell, int(atlas.active_stop[trace_id]))

        for pair_i, trace_a in enumerate(trace_ids):
            for trace_b in trace_ids[pair_i + 1:]:
                overlap_start = max(int(cell.start), int(atlas.active_start[trace_a]), int(atlas.active_start[trace_b]))
                overlap_stop = min(int(cell.stop), int(atlas.active_stop[trace_a]), int(atlas.active_stop[trace_b]))
                if overlap_stop - overlap_start < 2:
                    continue
                t_min = float(times_cpu[overlap_start].item())
                t_max = float(times_cpu[overlap_stop - 1].item())
                for root in _projective_trace_cell_pair_depth_roots(
                    coeffs_cpu,
                    trace_a=trace_a,
                    trace_b=trace_b,
                    t_min=t_min,
                    t_max=t_max,
                    eps=float(root_epsilon),
                ):
                    _add_root_boundaries(boundaries, cell, root)

        ordered_boundaries = sorted(boundaries)
        for start, stop in zip(ordered_boundaries, ordered_boundaries[1:]):
            if stop <= start:
                continue
            active_trace_ids = tuple(
                trace_id
                for trace_id in trace_ids
                if int(atlas.active_start[trace_id]) <= start and stop <= int(atlas.active_stop[trace_id])
            )
            if not active_trace_ids:
                continue
            t_mid = 0.5 * (float(times_cpu[start].item()) + float(times_cpu[stop - 1].item()))
            ordered = tuple(
                trace_id
                for _depth, trace_id in sorted(
                    (_depth_at(trace_id, t_mid), trace_id) for trace_id in active_trace_ids
                )
            )
            depth_intervals: list[tuple[float, float]] = []
            for trace_id in ordered:
                trace_depths = dense[trace_id, start:stop, 2]
                valid = dense[trace_id, start:stop, 3] != 0.0
                valid_depths = trace_depths[valid]
                if valid_depths.numel() == 0:
                    depth_intervals.append((math.inf, math.inf))
                    continue
                depth_intervals.append(
                    (
                        float(valid_depths.amin().item()),
                        float(valid_depths.amax().item()),
                    )
                )
            cells.append(
                ProjectiveTraceTileTimeCell(
                    tile_u=int(cell.tile_u),
                    tile_v=int(cell.tile_v),
                    start=start,
                    stop=stop,
                    primitive_ids=tuple(sorted(active_trace_ids)),
                    ordered_primitive_ids=ordered,
                    depth_intervals=tuple(depth_intervals),
                    fallback=bool(cell.fallback),
                    fallback_reasons=cell.fallback_reasons,
                )
            )

    return replace(atlas, cells=sorted(cells, key=lambda cell: (cell.start, cell.stop, cell.tile_v, cell.tile_u)))


def mark_projective_trace_cell_visibility_fallbacks(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    depth_epsilon: float = 1.0e-6,
    image_width: int | None = None,
    image_height: int | None = None,
    tile_size: int | None = None,
    fallback_reason: str = "visibility_ambiguous_depth",
) -> ProjectiveTraceCellTraceAtlas:
    """Mark only the sample runs whose depths cannot use a stable order.

    A bad sample must not force unrelated times in its parent cell to fallback.
    Existing fallback reasons remain in force over their original whole interval.
    Exactly zero spatial slope polynomials use scalar depth checks; they have
    no spatial order boundary to search for, even when scalar depths coincide.
    """

    if depth_epsilon < 0.0:
        raise ValueError("depth_epsilon must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)
    has_tile_depth_domain = _validate_optional_tile_depth_domain(
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    times_cpu = times.detach().cpu().contiguous()
    dense = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
    depth_affine_uv_cpu = (
        None
        if atlas.depth_affine_uv is None or not has_tile_depth_domain
        else atlas.depth_affine_uv.detach().cpu().contiguous()
    )
    if depth_affine_uv_cpu is not None and not bool(torch.any(depth_affine_uv_cpu != 0).item()):
        depth_affine_uv_cpu = None
    frame_count = int(times_cpu.numel())
    trace_count = int(coeffs_cpu.shape[0])
    fallback_reasons_by_cell: dict[int, dict[int, set[str]]] = {}

    def _mark_cell(cell_index: int, sample_index: int, reason: str) -> None:
        fallback_reasons_by_cell.setdefault(int(cell_index), {}).setdefault(int(sample_index), set()).add(str(reason))

    entries_by_key: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for cell_index, cell in enumerate(atlas.cells):
        if cell.tile_u < 0 or cell.tile_v < 0 or (
            has_tile_depth_domain and (
                cell.tile_u * int(tile_size) >= int(image_width)
                or cell.tile_v * int(tile_size) >= int(image_height)
            )
        ):
            raise ValueError("cell tile coordinates are outside the image tile grid")
        if cell.start < 0 or cell.stop > frame_count or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for times")
        for trace_id in cell.ordered_primitive_ids:
            if trace_id < 0 or trace_id >= trace_count:
                raise ValueError("cell trace id is outside the atlas coeff table")
        for sample_index in range(cell.start, cell.stop):
            entries = entries_by_key.setdefault((sample_index, int(cell.tile_u), int(cell.tile_v)), [])
            for trace_id in cell.ordered_primitive_ids:
                entries.append((int(trace_id), cell_index))

    for (sample_index, tile_u, tile_v), entries in entries_by_key.items():
        seen: set[int] = set()
        live_depths: list[tuple[float, float, int, int]] = []
        for trace_id, cell_index in entries:
            if trace_id in seen:
                continue
            seen.add(trace_id)
            valid_sign = dense[trace_id, sample_index, 3]
            if float(valid_sign.item()) == 0.0:
                _mark_cell(cell_index, sample_index, fallback_reason)
                continue
            depth_min, depth_max = _cell_trace_depth_range_for_tile_sample(
                dense,
                depth_affine_uv_cpu,
                times_cpu,
                trace_id=trace_id,
                sample_index=sample_index,
                tile_u=tile_u,
                tile_v=tile_v,
                image_width=image_width,
                image_height=image_height,
                tile_size=tile_size,
            )
            live_depths.append((depth_min, depth_max, trace_id, cell_index))
        if len(live_depths) < 2:
            continue
        live_depths.sort(key=lambda item: (item[0], item[1], item[2]))
        for (_min_a, max_a, _trace_a, cell_a), (min_b, _max_b, _trace_b, cell_b) in zip(live_depths, live_depths[1:]):
            if max_a + float(depth_epsilon) >= min_b:
                _mark_cell(cell_a, sample_index, fallback_reason)
                _mark_cell(cell_b, sample_index, fallback_reason)

    if depth_affine_uv_cpu is not None:
        uv_event_report = projective_trace_cell_uv_visibility_event_report(
            atlas,
            times,
            image_width=int(image_width),
            image_height=int(image_height),
            tile_size=int(tile_size),
            depth_epsilon=depth_epsilon,
        )
        for event in uv_event_report.events:
            _mark_cell(int(event.cell_index), int(event.sample_index), "visibility_uv_depth_line")

    if not fallback_reasons_by_cell:
        return atlas

    cells: list[ProjectiveTraceTileTimeCell] = []
    for cell_index, cell in enumerate(atlas.cells):
        if cell_index not in fallback_reasons_by_cell:
            cells.append(cell)
            continue
        marked_samples = fallback_reasons_by_cell[cell_index]
        boundaries = sorted({cell.start, cell.stop, *marked_samples, *(sample + 1 for sample in marked_samples)})
        segments: list[ProjectiveTraceTileTimeCell] = []
        for start, stop in zip(boundaries, boundaries[1:]):
            added_reasons = marked_samples.get(start, set())
            reasons = tuple(sorted(set(cell.fallback_reasons) | added_reasons))
            fallback = bool(cell.fallback or added_reasons)
            if segments and segments[-1].fallback == fallback and segments[-1].fallback_reasons == reasons:
                segments[-1] = replace(segments[-1], stop=stop)
            else:
                segments.append(replace(cell, start=start, stop=stop, fallback=fallback, fallback_reasons=reasons))
        cells.extend(segments)

    return replace(atlas, cells=cells)


def stratify_projective_trace_cell_atlas_visibility(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    depth_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellTraceAtlas:
    """Split cell time ranges into runs with stable live depth order.

    This is the first sampled visibility-stratum compiler for cell-local
    projective traces. It preserves the tensor payload and tile support, but
    replaces broad cells whose front-to-back order changes over time with
    smaller sample intervals whose compiled order matches live evaluated depth.
    Ambiguous near-ties remain ambiguous; the fallback marker handles those
    after stratification.
    """

    if depth_epsilon < 0.0:
        raise ValueError("depth_epsilon must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    times_cpu = times.detach().cpu().contiguous()
    dense = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
    frame_count = int(times_cpu.numel())
    trace_count = int(coeffs_cpu.shape[0])

    traces_by_key: dict[tuple[int, int, int], set[int]] = {}
    fallback_reasons_by_key: dict[tuple[int, int, int], set[str]] = {}
    for cell in atlas.cells:
        if cell.tile_u < 0 or cell.tile_v < 0:
            raise ValueError("cell tile coordinates must be non-negative")
        if cell.start < 0 or cell.stop > frame_count or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for times")
        for trace_id in cell.ordered_primitive_ids:
            if trace_id < 0 or trace_id >= trace_count:
                raise ValueError("cell trace id is outside the atlas coeff table")
        for sample_index in range(cell.start, cell.stop):
            key = (sample_index, int(cell.tile_u), int(cell.tile_v))
            traces = traces_by_key.setdefault(key, set())
            for trace_id in cell.ordered_primitive_ids:
                active_start = int(atlas.active_start[trace_id])
                active_stop = int(atlas.active_stop[trace_id])
                if active_start <= sample_index < active_stop:
                    traces.add(int(trace_id))
            if cell.fallback_reasons:
                fallback_reasons_by_key.setdefault(key, set()).update(cell.fallback_reasons)

    order_by_key: dict[tuple[int, int, int], tuple[tuple[int, ...], tuple[str, ...]]] = {}
    for key, trace_ids in traces_by_key.items():
        sample_index, _tile_u, _tile_v = key
        live_depths: list[tuple[float, int]] = []
        for trace_id in trace_ids:
            valid_sign = dense[trace_id, sample_index, 3]
            if float(valid_sign.item()) == 0.0:
                continue
            live_depths.append((float(dense[trace_id, sample_index, 2].item()), int(trace_id)))
        if not live_depths:
            continue
        live_depths.sort(key=lambda item: (item[0], item[1]))
        reasons = fallback_reasons_by_key.get(key, set())
        order_by_key[key] = (
            tuple(trace_id for _depth, trace_id in live_depths),
            tuple(sorted(reasons)),
        )

    cells: list[ProjectiveTraceTileTimeCell] = []
    tile_keys = sorted({(tile_u, tile_v) for _sample, tile_u, tile_v in order_by_key})
    for tile_u, tile_v in tile_keys:
        samples = sorted(sample for sample, key_u, key_v in order_by_key if key_u == tile_u and key_v == tile_v)
        run_start: int | None = None
        run_stop: int | None = None
        run_order: tuple[int, ...] = ()
        run_reasons: tuple[str, ...] = ()

        def _flush_run() -> None:
            nonlocal run_start, run_stop, run_order, run_reasons
            if run_start is None or run_stop is None or not run_order:
                return
            depth_intervals: list[tuple[float, float]] = []
            for trace_id in run_order:
                trace_depths = dense[trace_id, run_start:run_stop, 2]
                valid = dense[trace_id, run_start:run_stop, 3] != 0.0
                valid_depths = trace_depths[valid]
                if valid_depths.numel() == 0:
                    depth_intervals.append((math.inf, math.inf))
                    continue
                depth_intervals.append(
                    (
                        float(valid_depths.amin().item()),
                        float(valid_depths.amax().item()),
                    )
                )
            cells.append(
                ProjectiveTraceTileTimeCell(
                    tile_u=tile_u,
                    tile_v=tile_v,
                    start=run_start,
                    stop=run_stop,
                    primitive_ids=tuple(sorted(run_order)),
                    ordered_primitive_ids=run_order,
                    depth_intervals=tuple(depth_intervals),
                    fallback=bool(run_reasons),
                    fallback_reasons=run_reasons,
                )
            )
            run_start = None
            run_stop = None
            run_order = ()
            run_reasons = ()

        for sample_index in samples:
            order, reasons = order_by_key[(sample_index, tile_u, tile_v)]
            if (
                run_start is not None
                and run_stop == sample_index
                and run_order == order
                and run_reasons == reasons
            ):
                run_stop = sample_index + 1
                continue
            _flush_run()
            run_start = sample_index
            run_stop = sample_index + 1
            run_order = order
            run_reasons = reasons
        _flush_run()

    return replace(atlas, cells=sorted(cells, key=lambda cell: (cell.start, cell.stop, cell.tile_v, cell.tile_u)))


def rebin_projective_trace_cell_atlas_support_events(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    uv_padding: float = 0.0,
    trace_uv_padding: Tensor | None = None,
    depth_padding: float = 0.0,
    root_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellTraceAtlas:
    """Rebuild support cells with continuous tile-boundary split points.

    Optional [N,2] UV radii bound each trace separately; ``uv_padding`` remains
    a minimum on both axes. A wide trace must not inflate every other trace's
    tile list. The same radii drive event roots and interval support boxes.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if uv_padding < 0.0:
        raise ValueError("uv_padding must be non-negative")
    if depth_padding < 0.0:
        raise ValueError("depth_padding must be non-negative")
    if root_epsilon < 0.0:
        raise ValueError("root_epsilon must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    if trace_uv_padding is None:
        padding = [(float(uv_padding), float(uv_padding))] * int(coeffs_cpu.shape[0])
    else:
        if trace_uv_padding.shape != (int(coeffs_cpu.shape[0]), 2):
            raise ValueError("trace_uv_padding must have shape [N,2]")
        radii = trace_uv_padding.detach().cpu()
        if not bool(torch.all(torch.isfinite(radii) & (radii >= 0.0))):
            raise ValueError("trace_uv_padding must be finite and non-negative")
        padding = radii.clamp_min(float(uv_padding)).tolist()
    times_cpu = times.detach().cpu().contiguous()
    if times_cpu.ndim != 1:
        raise ValueError("times must have shape [S]")
    if int(times_cpu.numel()) < 1:
        raise ValueError("times must not be empty")
    if int(times_cpu.numel()) > 1 and bool(torch.any(times_cpu[1:] < times_cpu[:-1]).item()):
        raise ValueError("times must be sorted in nondecreasing order")

    frame_count = int(times_cpu.numel())
    axes = (
        (0, _axis_tile_boundaries(image_extent=image_width, tile_size=tile_size)),
        (3, _axis_tile_boundaries(image_extent=image_height, tile_size=tile_size)),
    )
    records: list[ProjectiveTraceTileTimeRecord] = []

    for trace_id in range(int(coeffs_cpu.shape[0])):
        padding_u, padding_v = padding[trace_id]
        active_start = int(atlas.active_start[trace_id])
        active_stop = int(atlas.active_stop[trace_id])
        if active_start < 0 or active_stop > frame_count or active_start >= active_stop:
            raise ValueError("active intervals must be valid for the supplied times")

        boundaries: set[int] = {active_start, active_stop}
        if active_stop - active_start >= 2:
            t_min = float(times_cpu[active_start].item())
            t_max = float(times_cpu[active_stop - 1].item())
            for (axis_offset, tile_boundaries), axis_padding in zip(axes, (padding_u, padding_v)):
                for _side, signed_padding in _axis_support_sides(axis_padding):
                    for boundary in tile_boundaries:
                        for root in _cell_trace_axis_roots_for_boundary(
                            coeffs_cpu,
                            trace_id=trace_id,
                            axis_offset=axis_offset,
                            signed_padding=signed_padding,
                            boundary=boundary,
                            t_min=t_min,
                            t_max=t_max,
                            eps=float(root_epsilon),
                        ):
                            _add_sample_boundary_from_root(
                                boundaries,
                                times_cpu,
                                start=active_start,
                                stop=active_stop,
                                root=root,
                                eps=float(root_epsilon),
                            )

        for start, stop in zip(sorted(boundaries), sorted(boundaries)[1:]):
            if stop <= start:
                continue
            t_min = float(times_cpu[start].item())
            t_max = float(times_cpu[stop - 1].item())
            u_min, u_max = _quadratic_value_range_over_interval(coeffs_cpu[trace_id, 0:3], t_min=t_min, t_max=t_max)
            v_min, v_max = _quadratic_value_range_over_interval(coeffs_cpu[trace_id, 3:6], t_min=t_min, t_max=t_max)
            depth_min, depth_max = _quadratic_value_range_over_interval(coeffs_cpu[trace_id, 6:9], t_min=t_min, t_max=t_max)
            tile_range = _projective_trace_support_tiles(
                u_min=u_min - padding_u,
                u_max=u_max + padding_u,
                v_min=v_min - padding_v,
                v_max=v_max + padding_v,
                image_width=image_width,
                image_height=image_height,
                tile_size=tile_size,
            )
            if tile_range is None:
                continue
            tile_u_min, tile_u_max, tile_v_min, tile_v_max = tile_range
            records.append(
                ProjectiveTraceTileTimeRecord(
                    primitive_id=trace_id,
                    window_index=int(atlas.source_window_indices[trace_id]),
                    start=start,
                    stop=stop,
                    tile_u_min=tile_u_min,
                    tile_u_max=tile_u_max,
                    tile_v_min=tile_v_min,
                    tile_v_max=tile_v_max,
                    depth_min=float(depth_min) - float(depth_padding),
                    depth_max=float(depth_max) + float(depth_padding),
                    fallback=False,
                    fallback_reason="",
                )
            )

    return replace(atlas, cells=assemble_projective_trace_tile_time_atlas(records))


def rebin_projective_trace_cell_atlas(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
) -> ProjectiveTraceCellTraceAtlas:
    """Rebuild tile-time cells for the current cell-trace coefficients.

    This is a support recompilation step, not a trace refit: it preserves the
    live coefficient/color/opacity tensors and only refreshes interval tile
    membership plus conservative depth intervals for the sampled frame times.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if uv_padding < 0.0:
        raise ValueError("uv_padding must be non-negative")
    if depth_padding < 0.0:
        raise ValueError("depth_padding must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    times_cpu = times.detach().cpu().contiguous()
    dense = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
    frame_count = int(times_cpu.numel())
    records: list[ProjectiveTraceTileTimeRecord] = []

    for trace_id in range(int(coeffs_cpu.shape[0])):
        active_start = int(atlas.active_start[trace_id])
        active_stop = int(atlas.active_stop[trace_id])
        if active_start < 0 or active_stop > frame_count or active_start >= active_stop:
            raise ValueError("active intervals must be valid for the supplied times")

        samples = dense[trace_id, active_start:active_stop]
        valid = samples[:, 3] != 0.0
        if not bool(torch.any(valid).item()):
            continue
        valid_samples = samples[valid]
        uv_min = valid_samples[:, :2].amin(dim=0)
        uv_max = valid_samples[:, :2].amax(dim=0)
        depth_min = float(valid_samples[:, 2].amin().item()) - float(depth_padding)
        depth_max = float(valid_samples[:, 2].amax().item()) + float(depth_padding)
        tile_range = _projective_trace_support_tiles(
            u_min=float(uv_min[0].item()) - float(uv_padding),
            u_max=float(uv_max[0].item()) + float(uv_padding),
            v_min=float(uv_min[1].item()) - float(uv_padding),
            v_max=float(uv_max[1].item()) + float(uv_padding),
            image_width=image_width,
            image_height=image_height,
            tile_size=tile_size,
        )
        if tile_range is None:
            continue
        tile_u_min, tile_u_max, tile_v_min, tile_v_max = tile_range
        records.append(
            ProjectiveTraceTileTimeRecord(
                primitive_id=trace_id,
                window_index=int(atlas.source_window_indices[trace_id]),
                start=active_start,
                stop=active_stop,
                tile_u_min=tile_u_min,
                tile_u_max=tile_u_max,
                tile_v_min=tile_v_min,
                tile_v_max=tile_v_max,
                depth_min=depth_min,
                depth_max=depth_max,
                fallback=False,
                fallback_reason="",
            )
        )

    return replace(atlas, cells=assemble_projective_trace_tile_time_atlas(records))


def _composite_ordered_projective_cell_tile(
    atlas: ProjectiveTraceCellTraceAtlas,
    dense: Tensor,
    opacity_time_scale: Tensor | None,
    sample_index: int,
    ordered_ids: tuple[int, ...],
    pixel_u: Tensor,
    pixel_v: Tensor,
    *,
    sigma_px: float,
    alpha_cutoff: float,
    transmittance_cutoff: float,
) -> Tensor:
    """Batch a tile's fixed order while preserving tile-wide early termination."""
    if not ordered_ids:
        return atlas.color.new_zeros((pixel_v.numel(), pixel_u.numel(), atlas.color.shape[1]))
    index = torch.tensor(ordered_ids, dtype=torch.long, device=atlas.color.device)
    centers = dense.index_select(0, index)[:, sample_index]
    du = pixel_u[None, None, :] - centers[:, 0, None, None]
    dv = pixel_v[None, :, None] - centers[:, 1, None, None]
    if atlas.spatial_precision_uv is None:
        radius2 = (du.square() + dv.square()) / float(sigma_px * sigma_px)
    else:
        q = atlas.spatial_precision_uv.index_select(0, index)
        radius2 = q[:, 0, None, None] * du.square() + 2.0 * q[:, 1, None, None] * du * dv + q[:, 2, None, None] * dv.square()
    opacity = atlas.opacity.index_select(0, index)
    if opacity_time_scale is not None:
        opacity = opacity * opacity_time_scale.index_select(0, index)[:, sample_index]
    alpha = (opacity[:, None, None] * torch.exp(-0.5 * radius2)).clamp(0.0, 1.0)
    if alpha_cutoff > 0.0:
        alpha = torch.where(alpha >= float(alpha_cutoff), alpha, torch.zeros_like(alpha))
    prefix = torch.cat((torch.ones_like(alpha[:1]), torch.cumprod(1.0 - alpha, dim=0)[:-1]), dim=0)
    if transmittance_cutoff > 0.0:
        # The scalar reference stops only once every pixel is below threshold.
        # Its first valid trace is always processed, even for thresholds >= 1.
        active = prefix.detach().amax(dim=(1, 2)) > float(transmittance_cutoff)
        active[0] = True
        prefix = prefix * active[:, None, None]
    weights = (prefix * alpha).flatten(1).transpose(0, 1)
    return (weights @ atlas.color.index_select(0, index)).reshape(pixel_v.numel(), pixel_u.numel(), -1)


def render_projective_trace_cell_atlas_reference(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    sigma_px: float,
    alpha_cutoff: float = 0.0,
    transmittance_cutoff: float = 0.0,
    allow_fallback_cells: bool = False,
    fallback_sort_live_depth: bool = True,
    fallback_tiles_only: bool = False,
) -> Tensor:
    """Reference renderer for packed cell-local polynomial traces.

    ``fallback_tiles_only`` leaves other tile samples zero. Selected tiles still
    collect every contributing cell, including cells not marked as fallback.
    """

    _check_projective_trace_render_inputs(atlas.coeffs, times, atlas.color, atlas.opacity)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if alpha_cutoff < 0.0:
        raise ValueError("alpha_cutoff must be non-negative")
    if transmittance_cutoff < 0.0:
        raise ValueError("transmittance_cutoff must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)
    _validate_uvt_reference_depth_coefficients(atlas)

    trace_count = int(atlas.coeffs.shape[0])
    dense = eval_projective_trace_cell_torch(atlas.coeffs, times)
    opacity_time_scale = _cell_opacity_time_scale(atlas, times)
    depth_affine_uv = _cell_depth_affine_uv(atlas)
    use_pixel_depth_fallback_sort = fallback_sort_live_depth and _cell_has_nonzero_depth_affine_uv(atlas)
    reference_depth = (
        None if atlas.depth_reference_uvt is None or use_pixel_depth_fallback_sort
        else _uvt_reference_depth(atlas.depth_reference_uvt, times, 0.0, 0.0)
    )
    # Sorting/validity decisions are discrete. Copy their exact float32 values
    # once instead of synchronizing MPS for every trace in every tile.
    valid_values = dense[:, :, 3].detach().cpu().tolist()
    depth_values = (dense[:, :, 2] if reference_depth is None else reference_depth).detach().cpu().tolist()
    def source_order_key(trace_id: int) -> tuple[int, int]:
        return (int(atlas.source_primitive_ids[trace_id]), int(trace_id))

    entries_by_key: dict[tuple[int, int, int], list[tuple[int, float, float]]] = {}
    fallback_keys: set[tuple[int, int, int]] = set()
    for cell in atlas.cells:
        if cell.tile_u < 0 or cell.tile_v < 0:
            raise ValueError("cell tile coordinates must be non-negative")
        if cell.start < 0 or cell.stop > int(times.numel()) or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for times")
        if len(cell.ordered_primitive_ids) != len(cell.depth_intervals):
            raise ValueError("cell ordered ids and depth intervals must match")
        if cell.fallback and not allow_fallback_cells:
            reason = ",".join(cell.fallback_reasons) if cell.fallback_reasons else "fallback"
            raise ValueError(f"cannot reference-render fallback cell: {reason}")
        for trace_id in cell.ordered_primitive_ids:
            if trace_id < 0 or trace_id >= trace_count:
                raise ValueError("cell trace id is outside the atlas coeff table")
        for sample_index in range(cell.start, cell.stop):
            key = (sample_index, cell.tile_u, cell.tile_v)
            if cell.fallback:
                fallback_keys.add(key)
            entries = entries_by_key.setdefault(key, [])
            for trace_id, depth_interval in zip(cell.ordered_primitive_ids, cell.depth_intervals):
                depth_min, depth_max = depth_interval
                entries.append((trace_id, 0.5 * (depth_min + depth_max), depth_min))

    ordered_by_key: dict[tuple[int, int, int], tuple[int, ...]] = {}
    for key, entries in entries_by_key.items():
        if fallback_tiles_only and key not in fallback_keys:
            continue
        if key in fallback_keys and fallback_sort_live_depth:
            sample_index, _tile_u, _tile_v = key

            def _live_depth_key(item: tuple[int, float, float]) -> tuple[float, int, int]:
                if valid_values[item[0]][sample_index] == 0.0:
                    return (math.inf, *source_order_key(item[0]))
                return (depth_values[item[0]][sample_index], *source_order_key(item[0]))

            entries.sort(key=_live_depth_key)
        else:
            entries.sort(key=lambda item: (item[1], item[2], item[0]))
        seen: set[int] = set()
        ordered: list[int] = []
        for trace_id, _depth_mid, _depth_min in entries:
            if trace_id not in seen:
                seen.add(trace_id)
                ordered.append(trace_id)
        ordered_by_key[key] = tuple(ordered)

    tile_cols = (image_width + tile_size - 1) // tile_size
    tile_rows = (image_height + tile_size - 1) // tile_size
    out = torch.zeros(
        (int(times.numel()), image_height, image_width, int(atlas.color.shape[1])),
        dtype=atlas.color.dtype,
        device=atlas.color.device,
    )

    for sample_index in range(int(times.numel())):
        for tile_v in range(tile_rows):
            v0 = tile_v * tile_size
            v1 = min(image_height, v0 + tile_size)
            pixel_v = torch.arange(v0, v1, dtype=atlas.color.dtype, device=atlas.color.device) + 0.5
            for tile_u in range(tile_cols):
                ordered_ids = ordered_by_key.get((sample_index, tile_u, tile_v), ())
                if not ordered_ids:
                    continue
                u0 = tile_u * tile_size
                u1 = min(image_width, u0 + tile_size)
                pixel_u = torch.arange(u0, u1, dtype=atlas.color.dtype, device=atlas.color.device) + 0.5
                du = pixel_u.reshape(1, -1)
                dv = pixel_v.reshape(-1, 1)
                if fallback_tiles_only and not use_pixel_depth_fallback_sort:
                    out[sample_index, v0:v1, u0:u1, :] = _composite_ordered_projective_cell_tile(
                        atlas, dense, opacity_time_scale, sample_index,
                        tuple(i for i in ordered_ids if valid_values[i][sample_index] != 0.0),
                        pixel_u, pixel_v, sigma_px=sigma_px, alpha_cutoff=alpha_cutoff,
                        transmittance_cutoff=transmittance_cutoff,
                    )
                    continue
                tile_rgb = out[sample_index, v0:v1, u0:u1, :]
                transmittance = torch.ones((v1 - v0, u1 - u0), dtype=atlas.color.dtype, device=atlas.color.device)

                if (sample_index, tile_u, tile_v) in fallback_keys and use_pixel_depth_fallback_sort:
                    pixel_v_values = pixel_v.detach().cpu().tolist()
                    pixel_u_values = pixel_u.detach().cpu().tolist()
                    for local_v, pixel_v_value in enumerate(pixel_v_values):
                        for local_u, pixel_u_value in enumerate(pixel_u_values):
                            source_depths = (
                                None if atlas.depth_reference_uvt is None else
                                _uvt_reference_depth(atlas.depth_reference_uvt, times[sample_index:sample_index+1], pixel_u_value, pixel_v_value)[:, 0].detach().cpu().tolist()
                            )
                            pixel_order = sorted(
                                ordered_ids,
                                key=lambda trace_id: (
                                    float(source_depths[trace_id]) if source_depths is not None else _cell_trace_depth_at_uv_sample(
                                        dense,
                                        depth_affine_uv,
                                        times,
                                        trace_id=int(trace_id),
                                        sample_index=sample_index,
                                        u=float(pixel_u_value),
                                        v=float(pixel_v_value),
                                    ),
                                    *source_order_key(trace_id),
                                ),
                            )
                            pixel_transmittance = torch.ones((), dtype=atlas.color.dtype, device=atlas.color.device)
                            for trace_id in pixel_order:
                                center_u = dense[trace_id, sample_index, 0]
                                center_v = dense[trace_id, sample_index, 1]
                                if valid_values[trace_id][sample_index] == 0.0:
                                    continue
                                du_centered = torch.as_tensor(
                                    float(pixel_u_value),
                                    dtype=atlas.color.dtype,
                                    device=atlas.color.device,
                                ) - center_u
                                dv_centered = torch.as_tensor(
                                    float(pixel_v_value),
                                    dtype=atlas.color.dtype,
                                    device=atlas.color.device,
                                ) - center_v
                                radius2 = _cell_trace_quadratic_radius2(
                                    atlas,
                                    int(trace_id),
                                    du_centered,
                                    dv_centered,
                                    sigma_px=float(sigma_px),
                                )
                                opacity_i = atlas.opacity[trace_id]
                                if opacity_time_scale is not None:
                                    opacity_i = opacity_i * opacity_time_scale[trace_id, sample_index]
                                alpha = (opacity_i * torch.exp(-0.5 * radius2)).clamp(0.0, 1.0)
                                if alpha_cutoff > 0.0 and bool(alpha < float(alpha_cutoff)):
                                    alpha = torch.zeros_like(alpha)
                                tile_rgb[local_v, local_u, :] += pixel_transmittance * alpha * atlas.color[trace_id]
                                pixel_transmittance = pixel_transmittance * (1.0 - alpha)
                                if transmittance_cutoff > 0.0 and bool(pixel_transmittance <= float(transmittance_cutoff)):
                                    break
                    continue

                for trace_id in ordered_ids:
                    center_u = dense[trace_id, sample_index, 0]
                    center_v = dense[trace_id, sample_index, 1]
                    if valid_values[trace_id][sample_index] == 0.0:
                        continue
                    du_centered = du - center_u
                    dv_centered = dv - center_v
                    radius2 = _cell_trace_quadratic_radius2(
                        atlas,
                        trace_id,
                        du_centered,
                        dv_centered,
                        sigma_px=float(sigma_px),
                    )
                    opacity_i = atlas.opacity[trace_id]
                    if opacity_time_scale is not None:
                        opacity_i = opacity_i * opacity_time_scale[trace_id, sample_index]
                    alpha = opacity_i * torch.exp(-0.5 * radius2)
                    alpha = alpha.clamp(0.0, 1.0)
                    if alpha_cutoff > 0.0:
                        alpha = torch.where(alpha >= float(alpha_cutoff), alpha, torch.zeros_like(alpha))
                    tile_rgb += transmittance.unsqueeze(-1) * alpha.unsqueeze(-1) * atlas.color[trace_id]
                    transmittance = transmittance * (1.0 - alpha)
                    if transmittance_cutoff > 0.0 and bool(torch.all(transmittance <= float(transmittance_cutoff)).item()):
                        break

    return out


def _render_projective_trace_cell_atlas_dense_samples_reference(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    sigma_px: float,
    alpha_cutoff: float = 0.0,
    transmittance_cutoff: float = 0.0,
    row_start: int = 0,
    row_stop: int | None = None,
) -> Tensor:
    """Dense live-depth reference render for continuous direct trace samples."""

    _check_projective_trace_render_inputs(atlas.coeffs, times, atlas.color, atlas.opacity)
    _validate_projective_trace_cell_atlas_metadata(atlas)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if alpha_cutoff < 0.0:
        raise ValueError("alpha_cutoff must be non-negative")
    if transmittance_cutoff < 0.0:
        raise ValueError("transmittance_cutoff must be non-negative")
    if row_stop is None:
        row_stop = image_height
    if row_start < 0 or row_stop > image_height or row_start >= row_stop:
        raise ValueError("row range must be non-empty and inside the image")

    trace_count = int(atlas.coeffs.shape[0])
    channel_count = int(atlas.color.shape[1])
    dense = eval_projective_trace_cell_torch(atlas.coeffs, times)
    opacity_time_scale = _cell_opacity_time_scale(atlas, times)
    out = torch.zeros(
        (int(times.numel()), row_stop - row_start, image_width, channel_count),
        dtype=atlas.color.dtype,
        device=atlas.color.device,
    )
    pixel_u = torch.arange(image_width, dtype=atlas.color.dtype, device=atlas.color.device) + 0.5
    pixel_v = torch.arange(row_start, row_stop, dtype=atlas.color.dtype, device=atlas.color.device) + 0.5
    du = pixel_u.reshape(1, -1)
    dv = pixel_v.reshape(-1, 1)

    for sample_index in range(int(times.numel())):
        active = [
            trace_id
            for trace_id in range(trace_count)
            if float(dense[trace_id, sample_index, 3].item()) != 0.0
        ]
        order = sorted(active, key=lambda trace_id: (float(dense[trace_id, sample_index, 2].item()), trace_id))
        transmittance = torch.ones(
            (row_stop - row_start, image_width),
            dtype=atlas.color.dtype,
            device=atlas.color.device,
        )
        for trace_id in order:
            center_u = dense[trace_id, sample_index, 0]
            center_v = dense[trace_id, sample_index, 1]
            du_centered = du - center_u
            dv_centered = dv - center_v
            radius2 = _cell_trace_quadratic_radius2(
                atlas,
                trace_id,
                du_centered,
                dv_centered,
                sigma_px=float(sigma_px),
            )
            opacity_i = atlas.opacity[trace_id]
            if opacity_time_scale is not None:
                opacity_i = opacity_i * opacity_time_scale[trace_id, sample_index]
            alpha = opacity_i * torch.exp(-0.5 * radius2)
            alpha = alpha.clamp(0.0, 1.0)
            if alpha_cutoff > 0.0:
                alpha = torch.where(alpha >= float(alpha_cutoff), alpha, torch.zeros_like(alpha))
            out[sample_index] += transmittance.unsqueeze(-1) * alpha.unsqueeze(-1) * atlas.color[trace_id]
            transmittance = transmittance * (1.0 - alpha)
            if transmittance_cutoff > 0.0 and bool(torch.all(transmittance <= float(transmittance_cutoff)).item()):
                break

    return out


def render_projective_trace_cell_atlas_quadrature_reference(
    atlas: ProjectiveTraceCellTraceAtlas,
    quadrature: ProjectiveTraceCellSensorTimeQuadrature,
    *,
    image_width: int,
    image_height: int,
    sigma_px: float,
    alpha_cutoff: float = 0.0,
    transmittance_cutoff: float = 0.0,
) -> Tensor:
    """Reference finite-exposure render from weighted continuous-time samples.

    This path treats the cell atlas as a direct sensor-time trace table and
    ignores integer frame cell ranges. It is the correctness oracle for
    exposure quadrature schedules before those schedules are lowered to a
    production tile kernel.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if alpha_cutoff < 0.0:
        raise ValueError("alpha_cutoff must be non-negative")
    if transmittance_cutoff < 0.0:
        raise ValueError("transmittance_cutoff must be non-negative")
    channel_count = int(atlas.color.shape[1])
    if not quadrature.samples:
        return torch.zeros((image_height, image_width, channel_count), dtype=atlas.color.dtype, device=atlas.color.device)

    times = torch.tensor(
        [sample.time for sample in quadrature.samples],
        dtype=torch.float32,
        device=atlas.coeffs.device,
    ).contiguous()
    rendered = _render_projective_trace_cell_atlas_dense_samples_reference(
        atlas,
        times,
        image_width=image_width,
        image_height=image_height,
        sigma_px=sigma_px,
        alpha_cutoff=alpha_cutoff,
        transmittance_cutoff=transmittance_cutoff,
    )
    weights = torch.tensor(
        [sample.weight for sample in quadrature.samples],
        dtype=atlas.color.dtype,
        device=atlas.color.device,
    ).reshape(-1, 1, 1, 1)
    return (rendered * weights).sum(dim=0)


def render_projective_trace_cell_atlas_rolling_quadrature_reference(
    atlas: ProjectiveTraceCellTraceAtlas,
    row_quadrature: tuple[ProjectiveTraceCellSensorTimeQuadrature, ...],
    *,
    image_width: int,
    image_height: int,
    sigma_px: float,
    alpha_cutoff: float = 0.0,
    transmittance_cutoff: float = 0.0,
) -> Tensor:
    """Reference rolling-shutter render from one quadrature schedule per row."""

    if len(row_quadrature) != image_height:
        raise ValueError("row_quadrature must contain one schedule per image row")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if alpha_cutoff < 0.0:
        raise ValueError("alpha_cutoff must be non-negative")
    if transmittance_cutoff < 0.0:
        raise ValueError("transmittance_cutoff must be non-negative")

    rows: list[Tensor] = []
    channel_count = int(atlas.color.shape[1])
    for row_index, quadrature in enumerate(row_quadrature):
        if not quadrature.samples:
            rows.append(torch.zeros((image_width, channel_count), dtype=atlas.color.dtype, device=atlas.color.device))
            continue
        times = torch.tensor(
            [sample.time for sample in quadrature.samples],
            dtype=torch.float32,
            device=atlas.coeffs.device,
        ).contiguous()
        rendered = _render_projective_trace_cell_atlas_dense_samples_reference(
            atlas,
            times,
            image_width=image_width,
            image_height=image_height,
            sigma_px=sigma_px,
            alpha_cutoff=alpha_cutoff,
            transmittance_cutoff=transmittance_cutoff,
            row_start=row_index,
            row_stop=row_index + 1,
        )
        weights = torch.tensor(
            [sample.weight for sample in quadrature.samples],
            dtype=atlas.color.dtype,
            device=atlas.color.device,
        ).reshape(-1, 1, 1, 1)
        rows.append((rendered * weights).sum(dim=0).squeeze(0))

    return torch.stack(rows, dim=0)


def _quadrature_sample_tensors(
    quadrature: ProjectiveTraceCellSensorTimeQuadrature,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    times = torch.tensor(
        [sample.time for sample in quadrature.samples],
        dtype=torch.float32,
        device=device,
    )
    weights = torch.tensor(
        [sample.weight for sample in quadrature.samples],
        dtype=dtype,
        device=device,
    )
    if int(times.numel()) == 0:
        return times.contiguous(), weights.contiguous()
    order = torch.argsort(times)
    return times.index_select(0, order).contiguous(), weights.index_select(0, order).contiguous()


def _sample_active_indices_for_trace(
    atlas: ProjectiveTraceCellTraceAtlas,
    trace_id: int,
    sample_times_cpu: Tensor,
    *,
    domain_times: Tensor | None,
    time_epsilon: float,
) -> tuple[int, int] | None:
    sample_count = int(sample_times_cpu.numel())
    if sample_count == 0:
        return None
    if domain_times is None:
        return (0, sample_count)

    domain_cpu = domain_times.detach().cpu().contiguous()
    if domain_cpu.ndim != 1:
        raise ValueError("domain_times must have shape [S]")
    if int(domain_cpu.numel()) < 1:
        raise ValueError("domain_times must not be empty")
    if int(domain_cpu.numel()) > 1 and bool(torch.any(domain_cpu[1:] < domain_cpu[:-1]).item()):
        raise ValueError("domain_times must be sorted in nondecreasing order")

    active_start = int(atlas.active_start[trace_id])
    active_stop = int(atlas.active_stop[trace_id])
    if active_start < 0 or active_stop > int(domain_cpu.numel()) or active_start >= active_stop:
        raise ValueError("active intervals must be valid for domain_times")
    start_time = float(domain_cpu[active_start].item())
    if active_stop < int(domain_cpu.numel()):
        stop_time = float(domain_cpu[active_stop].item())
        mask = (sample_times_cpu >= start_time - float(time_epsilon)) & (sample_times_cpu < stop_time - float(time_epsilon))
    else:
        stop_time = float(domain_cpu[active_stop - 1].item())
        mask = (sample_times_cpu >= start_time - float(time_epsilon)) & (sample_times_cpu <= stop_time + float(time_epsilon))

    active_indices = torch.nonzero(mask, as_tuple=False).flatten()
    if int(active_indices.numel()) == 0:
        return None
    return int(active_indices[0].item()), int(active_indices[-1].item()) + 1


def lower_projective_trace_cell_atlas_quadrature(
    atlas: ProjectiveTraceCellTraceAtlas,
    quadrature: ProjectiveTraceCellSensorTimeQuadrature,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    domain_times: Tensor | None = None,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    root_epsilon: float = 1.0e-6,
    depth_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellQuadratureLowering:
    """Lower continuous quadrature samples to a sample-indexed cell atlas.

    The existing interval Metal kernel consumes integer sample intervals. This
    helper makes that contract explicit for finite-exposure schedules: sample
    indices are quadrature samples, while trace coefficients remain raw
    sensor-time functions evaluated at the corresponding fractional times.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if uv_padding < 0.0:
        raise ValueError("uv_padding must be non-negative")
    if depth_padding < 0.0:
        raise ValueError("depth_padding must be non-negative")
    if root_epsilon < 0.0:
        raise ValueError("root_epsilon must be non-negative")
    if depth_epsilon < 0.0:
        raise ValueError("depth_epsilon must be non-negative")
    _validate_projective_trace_cell_atlas_metadata(atlas)

    sample_times, sample_weights = _quadrature_sample_tensors(
        quadrature,
        device=atlas.coeffs.device,
        dtype=atlas.color.dtype,
    )
    if int(sample_times.numel()) == 0:
        empty_indices = torch.empty((0,), dtype=torch.long, device=atlas.coeffs.device)
        return ProjectiveTraceCellQuadratureLowering(
            atlas=ProjectiveTraceCellTraceAtlas(
                coeffs=atlas.coeffs.index_select(0, empty_indices),
                opacity=atlas.opacity.index_select(0, empty_indices),
                color=atlas.color.index_select(0, empty_indices),
                cells=[],
                source_window_indices=(),
                source_primitive_ids=(),
                active_start=(),
                active_stop=(),
                opacity_time_coeffs=_index_select_cell_opacity_time_coeffs(atlas, empty_indices),
                spatial_precision_uv=_index_select_cell_spatial_precision_uv(atlas, empty_indices),
                depth_affine_uv=_index_select_cell_depth_affine_uv(atlas, empty_indices),
                depth_reference_uvt=(None if atlas.depth_reference_uvt is None else atlas.depth_reference_uvt.index_select(0, empty_indices)),
            ),
            times=sample_times,
            weights=sample_weights,
            source_trace_indices=(),
        )

    sample_times_cpu = sample_times.detach().cpu().contiguous()
    source_trace_indices: list[int] = []
    active_start: list[int] = []
    active_stop: list[int] = []
    for trace_id in range(int(atlas.coeffs.shape[0])):
        active_range = _sample_active_indices_for_trace(
            atlas,
            trace_id,
            sample_times_cpu,
            domain_times=domain_times,
            time_epsilon=float(root_epsilon),
        )
        if active_range is None:
            continue
        source_trace_indices.append(trace_id)
        active_start.append(active_range[0])
        active_stop.append(active_range[1])

    index = torch.tensor(source_trace_indices, dtype=torch.long, device=atlas.coeffs.device)
    base_atlas = ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.index_select(0, index),
        opacity=atlas.opacity.index_select(0, index),
        color=atlas.color.index_select(0, index),
        cells=[],
        source_window_indices=tuple(int(atlas.source_window_indices[trace_id]) for trace_id in source_trace_indices),
        source_primitive_ids=tuple(int(atlas.source_primitive_ids[trace_id]) for trace_id in source_trace_indices),
        active_start=tuple(active_start),
        active_stop=tuple(active_stop),
        opacity_time_coeffs=_index_select_cell_opacity_time_coeffs(atlas, index),
        spatial_precision_uv=_index_select_cell_spatial_precision_uv(atlas, index),
        depth_affine_uv=_index_select_cell_depth_affine_uv(atlas, index),
        depth_reference_uvt=(None if atlas.depth_reference_uvt is None else atlas.depth_reference_uvt.index_select(0, index)),
    )
    if not source_trace_indices:
        return ProjectiveTraceCellQuadratureLowering(
            atlas=base_atlas,
            times=sample_times,
            weights=sample_weights,
            source_trace_indices=(),
        )

    support_atlas = rebin_projective_trace_cell_atlas_support_events(
        base_atlas,
        sample_times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        uv_padding=uv_padding,
        depth_padding=depth_padding,
        root_epsilon=root_epsilon,
    )
    event_atlas = stratify_projective_trace_cell_atlas_visibility_events(
        support_atlas,
        sample_times,
        root_epsilon=root_epsilon,
    )
    sampled_atlas = stratify_projective_trace_cell_atlas_visibility(
        event_atlas,
        sample_times,
        depth_epsilon=depth_epsilon,
    )
    return ProjectiveTraceCellQuadratureLowering(
        atlas=sampled_atlas,
        times=sample_times,
        weights=sample_weights,
        source_trace_indices=tuple(source_trace_indices),
    )


def split_projective_trace_cell_atlas_fallback_cells(
    atlas: ProjectiveTraceCellTraceAtlas,
) -> tuple[ProjectiveTraceCellTraceAtlas, ProjectiveTraceCellTraceAtlas]:
    """Split an atlas into fast cells and cells requiring fallback evaluation."""

    fast_cells = [cell for cell in atlas.cells if not cell.fallback]
    fallback_cells = [cell for cell in atlas.cells if cell.fallback]
    return (
        replace(atlas, cells=fast_cells),
        replace(atlas, cells=fallback_cells),
    )


def projective_trace_cell_atlas_fallback_tile_sample_mask(
    atlas: ProjectiveTraceCellTraceAtlas,
    *,
    frames: int,
    image_width: int,
    image_height: int,
    tile_size: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a bool mask of fallback sample/tile regions.

    The mask shape is ``[frames, tile_rows, tile_cols]``. A true value means the
    whole tile/sample must be replaced by live-depth fallback compositing.
    """

    if frames <= 0:
        raise ValueError("frames must be positive")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    tile_cols = (int(image_width) + int(tile_size) - 1) // int(tile_size)
    tile_rows = (int(image_height) + int(tile_size) - 1) // int(tile_size)
    fallback_mask = torch.zeros((int(frames), tile_rows, tile_cols), dtype=torch.bool, device=device)

    for cell in atlas.cells:
        if cell.tile_u < 0 or cell.tile_u >= tile_cols or cell.tile_v < 0 or cell.tile_v >= tile_rows:
            raise ValueError("cell tile coordinates are outside the image tile grid")
        if cell.start < 0 or cell.stop > int(frames) or cell.start >= cell.stop:
            raise ValueError("cell time range is invalid for frames")
        if cell.fallback:
            fallback_mask[int(cell.start) : int(cell.stop), int(cell.tile_v), int(cell.tile_u)] = True

    return fallback_mask


def _projective_trace_cell_atlas_detached_cpu(atlas: ProjectiveTraceCellTraceAtlas) -> ProjectiveTraceCellTraceAtlas:
    return ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.detach().cpu().contiguous(),
        depth_reference_uvt=(None if atlas.depth_reference_uvt is None else atlas.depth_reference_uvt.detach().cpu().contiguous()),
        opacity=atlas.opacity.detach().cpu().contiguous(),
        color=atlas.color.detach().cpu().contiguous(),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        opacity_time_coeffs=(
            None
            if atlas.opacity_time_coeffs is None
            else atlas.opacity_time_coeffs.detach().cpu().contiguous()
        ),
        spatial_precision_uv=(
            None
            if atlas.spatial_precision_uv is None
            else atlas.spatial_precision_uv.detach().cpu().contiguous()
        ),
        depth_affine_uv=(
            None
            if atlas.depth_affine_uv is None
            else atlas.depth_affine_uv.detach().cpu().contiguous()
        ),
    )


def _projective_trace_cell_reference_samples_cpu(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    sigma_px: float,
    alpha_cutoff: float,
    transmittance_cutoff: float,
) -> Tensor:
    return render_projective_trace_cell_atlas_reference(
        _projective_trace_cell_atlas_detached_cpu(atlas),
        times.detach().cpu().contiguous(),
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        sigma_px=sigma_px,
        alpha_cutoff=alpha_cutoff,
        transmittance_cutoff=transmittance_cutoff,
        allow_fallback_cells=True,
        fallback_sort_live_depth=True,
    )


def _patch_projective_trace_fallback_samples(
    fast_samples: Tensor,
    fallback_samples: Tensor,
    fallback_mask: Tensor,
    *,
    tile_size: int,
) -> Tensor:
    if fast_samples.shape != fallback_samples.shape:
        raise ValueError("fast_samples and fallback_samples must have the same shape")
    if fast_samples.ndim != 4:
        raise ValueError("samples must have shape [frames,height,width,channels]")
    if fallback_mask.shape[0] != fast_samples.shape[0]:
        raise ValueError("fallback_mask frames must match samples")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not bool(torch.any(fallback_mask).item()):
        return fast_samples

    patched = fast_samples.clone()
    height = int(fast_samples.shape[1])
    width = int(fast_samples.shape[2])
    for sample_index, tile_v, tile_u in fallback_mask.detach().cpu().nonzero(as_tuple=False).tolist():
        v0 = int(tile_v) * int(tile_size)
        u0 = int(tile_u) * int(tile_size)
        v1 = min(height, v0 + int(tile_size))
        u1 = min(width, u0 + int(tile_size))
        patched[int(sample_index), v0:v1, u0:u1, :] = fallback_samples[int(sample_index), v0:v1, u0:u1, :]
    return patched


def _mark_projective_trace_lowering_fallbacks(
    lowering: ProjectiveTraceCellQuadratureLowering,
    *,
    depth_epsilon: float,
    enabled: bool,
) -> ProjectiveTraceCellQuadratureLowering:
    if not enabled:
        return lowering
    return replace(
        lowering,
        atlas=mark_projective_trace_cell_visibility_fallbacks(
            lowering.atlas,
            lowering.times,
            depth_epsilon=depth_epsilon,
        ),
    )


def _mark_projective_trace_rolling_lowering_fallbacks(
    lowering: ProjectiveTraceCellRollingQuadratureLowering,
    *,
    depth_epsilon: float,
    enabled: bool,
) -> ProjectiveTraceCellRollingQuadratureLowering:
    if not enabled:
        return lowering
    return replace(
        lowering,
        atlas=mark_projective_trace_cell_visibility_fallbacks(
            lowering.atlas,
            lowering.times,
            depth_epsilon=depth_epsilon,
        ),
    )


def render_projective_trace_cell_atlas_quadrature_interval_metal(
    atlas: ProjectiveTraceCellTraceAtlas,
    quadrature: ProjectiveTraceCellSensorTimeQuadrature,
    config,
    *,
    sigma_px: float,
    domain_times: Tensor | None = None,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    root_epsilon: float = 1.0e-6,
    depth_epsilon: float = 1.0e-6,
    allow_fallback_cells: bool = False,
) -> Tensor:
    """Render finite-exposure quadrature through the interval Metal kernel."""

    if atlas.coeffs.device.type != "mps":
        raise ValueError("projective quadrature Metal render requires MPS tensors")
    if int(config.tile_x) != int(config.tile_y):
        raise ValueError("projective quadrature lowering currently requires square tiles")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")

    lowering = lower_projective_trace_cell_atlas_quadrature(
        atlas,
        quadrature,
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        domain_times=domain_times,
        uv_padding=uv_padding,
        depth_padding=depth_padding,
        root_epsilon=root_epsilon,
        depth_epsilon=depth_epsilon,
    )
    if int(lowering.times.numel()) == 0:
        return torch.zeros((int(config.height), int(config.width), int(atlas.color.shape[1])), dtype=atlas.color.dtype, device=atlas.color.device)

    sample_config = replace(config, frames=int(lowering.times.numel()))
    rendered = render_projective_trace_cell_interval_atlas_metal(
        lowering.atlas,
        lowering.times,
        sample_config,
        sigma_px=sigma_px,
        allow_fallback_cells=allow_fallback_cells,
    )
    return (rendered * lowering.weights.reshape(-1, 1, 1, 1)).sum(dim=0)


def render_projective_trace_cell_atlas_quadrature_interval_mixed_metal(
    atlas: ProjectiveTraceCellTraceAtlas,
    quadrature: ProjectiveTraceCellSensorTimeQuadrature,
    config,
    *,
    sigma_px: float,
    domain_times: Tensor | None = None,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    root_epsilon: float = 1.0e-6,
    depth_epsilon: float = 1.0e-6,
    mark_visibility_fallbacks: bool = True,
) -> Tensor:
    """Render finite exposure with fast Metal samples plus fallback patches.

    Fallback is applied at whole tile/sample granularity. The patched region is
    rendered with live-depth reference compositing over the full active list, so
    fallback does not behave like an additive layer on top of the fast render.
    Fallback patches are detached CPU oracle values; gradients are preserved
    only through non-fallback fast regions.
    """

    if atlas.coeffs.device.type != "mps":
        raise ValueError("projective quadrature mixed Metal render requires MPS tensors")
    if int(config.tile_x) != int(config.tile_y):
        raise ValueError("projective quadrature lowering currently requires square tiles")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")

    lowering = lower_projective_trace_cell_atlas_quadrature(
        atlas,
        quadrature,
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        domain_times=domain_times,
        uv_padding=uv_padding,
        depth_padding=depth_padding,
        root_epsilon=root_epsilon,
        depth_epsilon=depth_epsilon,
    )
    lowering = _mark_projective_trace_lowering_fallbacks(
        lowering,
        depth_epsilon=depth_epsilon,
        enabled=mark_visibility_fallbacks,
    )
    if int(lowering.times.numel()) == 0:
        return torch.zeros((int(config.height), int(config.width), int(atlas.color.shape[1])), dtype=atlas.color.dtype, device=atlas.color.device)

    stats = projective_trace_cell_atlas_fallback_stats(lowering.atlas)
    if stats.fallback_cells == 0:
        sample_config = replace(config, frames=int(lowering.times.numel()))
        rendered = render_projective_trace_cell_interval_atlas_metal(
            lowering.atlas,
            lowering.times,
            sample_config,
            sigma_px=sigma_px,
            allow_fallback_cells=False,
        )
        return (rendered * lowering.weights.reshape(-1, 1, 1, 1)).sum(dim=0)

    sample_config = replace(config, frames=int(lowering.times.numel()))
    fast_atlas, _fallback_atlas = split_projective_trace_cell_atlas_fallback_cells(lowering.atlas)
    if fast_atlas.cells:
        fast_samples = render_projective_trace_cell_interval_atlas_metal(
            fast_atlas,
            lowering.times,
            sample_config,
            sigma_px=sigma_px,
            allow_fallback_cells=False,
        )
    else:
        fast_samples = torch.zeros(
            (int(lowering.times.numel()), int(config.height), int(config.width), int(atlas.color.shape[1])),
            dtype=atlas.color.dtype,
            device=atlas.color.device,
        )

    fallback_samples = _projective_trace_cell_reference_samples_cpu(
        lowering.atlas,
        lowering.times,
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        sigma_px=sigma_px,
        alpha_cutoff=float(config.alpha_threshold),
        transmittance_cutoff=float(config.transmittance_threshold),
    ).to(device=fast_samples.device)
    fallback_mask = projective_trace_cell_atlas_fallback_tile_sample_mask(
        lowering.atlas,
        frames=int(lowering.times.numel()),
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        device=fast_samples.device,
    )
    patched_samples = _patch_projective_trace_fallback_samples(
        fast_samples,
        fallback_samples,
        fallback_mask,
        tile_size=int(config.tile_x),
    )
    return (patched_samples * lowering.weights.reshape(-1, 1, 1, 1)).sum(dim=0)


def _rolling_quadrature_unique_samples(
    row_quadrature: tuple[ProjectiveTraceCellSensorTimeQuadrature, ...],
    *,
    row_count: int,
    device: torch.device,
    dtype: torch.dtype,
    time_epsilon: float,
) -> tuple[ProjectiveTraceCellSensorTimeQuadrature, Tensor]:
    unique_times: list[float] = []
    row_weights: list[list[float]] = []

    def _find_or_add_time(time_value: float) -> int:
        for sample_index, existing_time in enumerate(unique_times):
            if abs(float(time_value) - float(existing_time)) <= float(time_epsilon):
                return sample_index
        unique_times.append(float(time_value))
        row_weights.append([0.0 for _ in range(row_count)])
        return len(unique_times) - 1

    for row_index, quadrature in enumerate(row_quadrature):
        for sample in quadrature.samples:
            if int(sample.row_index) not in (-1, row_index):
                raise ValueError("quadrature sample row_index must match its schedule row or be -1")
            sample_index = _find_or_add_time(float(sample.time))
            row_weights[sample_index][row_index] += float(sample.weight)

    order = sorted(range(len(unique_times)), key=lambda index: unique_times[index])
    sorted_times = [unique_times[index] for index in order]
    sorted_weights = [row_weights[index] for index in order]
    samples = tuple(
        ProjectiveTraceCellSensorTimeQuadratureSample(
            interval_index=-1,
            row_index=-1,
            start_time=float(time_value),
            stop_time=float(time_value),
            time=float(time_value),
            weight=1.0,
        )
        for time_value in sorted_times
    )
    weight_tensor = torch.tensor(sorted_weights, dtype=dtype, device=device)
    return (
        ProjectiveTraceCellSensorTimeQuadrature(
            samples=samples,
            total_weight=float(len(samples)),
        ),
        weight_tensor.contiguous(),
    )


def lower_projective_trace_cell_atlas_rolling_quadrature(
    atlas: ProjectiveTraceCellTraceAtlas,
    row_quadrature: tuple[ProjectiveTraceCellSensorTimeQuadrature, ...],
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    domain_times: Tensor | None = None,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    root_epsilon: float = 1.0e-6,
    depth_epsilon: float = 1.0e-6,
) -> ProjectiveTraceCellRollingQuadratureLowering:
    """Lower per-row rolling schedules to one shared sample-index atlas."""

    if len(row_quadrature) != int(image_height):
        raise ValueError("row_quadrature must contain one schedule per image row")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if root_epsilon < 0.0:
        raise ValueError("root_epsilon must be non-negative")

    unique_quadrature, row_weights = _rolling_quadrature_unique_samples(
        row_quadrature,
        row_count=int(image_height),
        device=atlas.coeffs.device,
        dtype=atlas.color.dtype,
        time_epsilon=float(root_epsilon),
    )
    lowering = lower_projective_trace_cell_atlas_quadrature(
        atlas,
        unique_quadrature,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        domain_times=domain_times,
        uv_padding=uv_padding,
        depth_padding=depth_padding,
        root_epsilon=root_epsilon,
        depth_epsilon=depth_epsilon,
    )
    return ProjectiveTraceCellRollingQuadratureLowering(
        atlas=lowering.atlas,
        times=lowering.times,
        row_weights=row_weights,
        source_trace_indices=lowering.source_trace_indices,
    )


def render_projective_trace_cell_atlas_rolling_quadrature_batched_reference(
    atlas: ProjectiveTraceCellTraceAtlas,
    row_quadrature: tuple[ProjectiveTraceCellSensorTimeQuadrature, ...],
    *,
    image_width: int,
    image_height: int,
    tile_size: int,
    sigma_px: float,
    domain_times: Tensor | None = None,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    root_epsilon: float = 1.0e-6,
    depth_epsilon: float = 1.0e-6,
    alpha_cutoff: float = 0.0,
    transmittance_cutoff: float = 0.0,
) -> Tensor:
    """Reference rolling render using one shared sample-index lowering."""

    lowering = lower_projective_trace_cell_atlas_rolling_quadrature(
        atlas,
        row_quadrature,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        domain_times=domain_times,
        uv_padding=uv_padding,
        depth_padding=depth_padding,
        root_epsilon=root_epsilon,
        depth_epsilon=depth_epsilon,
    )
    if int(lowering.times.numel()) == 0:
        return torch.zeros((image_height, image_width, int(atlas.color.shape[1])), dtype=atlas.color.dtype, device=atlas.color.device)
    rendered = render_projective_trace_cell_atlas_reference(
        lowering.atlas,
        lowering.times,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        sigma_px=sigma_px,
        alpha_cutoff=alpha_cutoff,
        transmittance_cutoff=transmittance_cutoff,
    )
    return (rendered * lowering.row_weights.reshape(-1, image_height, 1, 1)).sum(dim=0)


def render_projective_trace_cell_atlas_rolling_quadrature_interval_metal(
    atlas: ProjectiveTraceCellTraceAtlas,
    row_quadrature: tuple[ProjectiveTraceCellSensorTimeQuadrature, ...],
    config,
    *,
    sigma_px: float,
    domain_times: Tensor | None = None,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    root_epsilon: float = 1.0e-6,
    depth_epsilon: float = 1.0e-6,
    allow_fallback_cells: bool = False,
) -> Tensor:
    """Render rolling-shutter row schedules through one batched Metal call."""

    if len(row_quadrature) != int(config.height):
        raise ValueError("row_quadrature must contain one schedule per image row")
    if atlas.coeffs.device.type != "mps":
        raise ValueError("projective rolling quadrature Metal render requires MPS tensors")
    if int(config.tile_x) != int(config.tile_y):
        raise ValueError("projective rolling quadrature lowering currently requires square tiles")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")

    lowering = lower_projective_trace_cell_atlas_rolling_quadrature(
        atlas,
        row_quadrature,
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        domain_times=domain_times,
        uv_padding=uv_padding,
        depth_padding=depth_padding,
        root_epsilon=root_epsilon,
        depth_epsilon=depth_epsilon,
    )
    if int(lowering.times.numel()) == 0:
        return torch.zeros((int(config.height), int(config.width), int(atlas.color.shape[1])), dtype=atlas.color.dtype, device=atlas.color.device)

    sample_config = replace(config, frames=int(lowering.times.numel()))
    return render_projective_trace_cell_interval_atlas_rows_metal(
        lowering.atlas,
        lowering.times,
        lowering.row_weights,
        sample_config,
        sigma_px=sigma_px,
        allow_fallback_cells=allow_fallback_cells,
    )


def render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal(
    atlas: ProjectiveTraceCellTraceAtlas,
    row_quadrature: tuple[ProjectiveTraceCellSensorTimeQuadrature, ...],
    config,
    *,
    sigma_px: float,
    domain_times: Tensor | None = None,
    uv_padding: float = 0.0,
    depth_padding: float = 0.0,
    root_epsilon: float = 1.0e-6,
    depth_epsilon: float = 1.0e-6,
    mark_visibility_fallbacks: bool = True,
) -> Tensor:
    """Render rolling shutter with row-weighted Metal and fallback patches."""

    if len(row_quadrature) != int(config.height):
        raise ValueError("row_quadrature must contain one schedule per image row")
    if atlas.coeffs.device.type != "mps":
        raise ValueError("projective rolling mixed Metal render requires MPS tensors")
    if int(config.tile_x) != int(config.tile_y):
        raise ValueError("projective rolling quadrature lowering currently requires square tiles")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")

    lowering = lower_projective_trace_cell_atlas_rolling_quadrature(
        atlas,
        row_quadrature,
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        domain_times=domain_times,
        uv_padding=uv_padding,
        depth_padding=depth_padding,
        root_epsilon=root_epsilon,
        depth_epsilon=depth_epsilon,
    )
    lowering = _mark_projective_trace_rolling_lowering_fallbacks(
        lowering,
        depth_epsilon=depth_epsilon,
        enabled=mark_visibility_fallbacks,
    )
    if int(lowering.times.numel()) == 0:
        return torch.zeros((int(config.height), int(config.width), int(atlas.color.shape[1])), dtype=atlas.color.dtype, device=atlas.color.device)

    stats = projective_trace_cell_atlas_fallback_stats(lowering.atlas)
    sample_config = replace(config, frames=int(lowering.times.numel()))
    if stats.fallback_cells == 0:
        return render_projective_trace_cell_interval_atlas_rows_metal(
            lowering.atlas,
            lowering.times,
            lowering.row_weights,
            sample_config,
            sigma_px=sigma_px,
            allow_fallback_cells=False,
        )

    fast_atlas, _fallback_atlas = split_projective_trace_cell_atlas_fallback_cells(lowering.atlas)
    if fast_atlas.cells:
        fast_samples = render_projective_trace_cell_interval_atlas_metal(
            fast_atlas,
            lowering.times,
            sample_config,
            sigma_px=sigma_px,
            allow_fallback_cells=False,
        )
    else:
        fast_samples = torch.zeros(
            (int(lowering.times.numel()), int(config.height), int(config.width), int(atlas.color.shape[1])),
            dtype=atlas.color.dtype,
            device=atlas.color.device,
        )

    fallback_samples = _projective_trace_cell_reference_samples_cpu(
        lowering.atlas,
        lowering.times,
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        sigma_px=sigma_px,
        alpha_cutoff=float(config.alpha_threshold),
        transmittance_cutoff=float(config.transmittance_threshold),
    ).to(device=fast_samples.device)
    fallback_mask = projective_trace_cell_atlas_fallback_tile_sample_mask(
        lowering.atlas,
        frames=int(lowering.times.numel()),
        image_width=int(config.width),
        image_height=int(config.height),
        tile_size=int(config.tile_x),
        device=fast_samples.device,
    )
    patched_samples = _patch_projective_trace_fallback_samples(
        fast_samples,
        fallback_samples,
        fallback_mask,
        tile_size=int(config.tile_x),
    )
    return (patched_samples * lowering.row_weights.reshape(-1, int(config.height), 1, 1)).sum(dim=0)


def has_projective_trace_tile_metal() -> bool:
    return hasattr(torch.ops.star_uvt_v0, "render_projective_trace_tiles")


def has_projective_trace_cell_metal() -> bool:
    return hasattr(torch.ops.star_uvt_v0, "render_projective_trace_cell_tiles")


def has_projective_trace_cell_interval_metal() -> bool:
    return hasattr(torch.ops.star_uvt_v0, "render_projective_trace_cell_interval_tiles")


def has_projective_trace_family_interval_metal() -> bool:
    return hasattr(torch.ops.star_uvt_v0, "render_projective_trace_family_interval_tiles")


def has_projective_trace_family_interval_backward_metal() -> bool:
    return hasattr(torch.ops.star_uvt_v0, "direct_projective_trace_family_interval_backward")


def has_projective_trace_cell_interval_rows_metal() -> bool:
    return hasattr(torch.ops.star_uvt_v0, "render_projective_trace_cell_interval_rows")


def has_projective_trace_cell_interval_backward_metal() -> bool:
    return hasattr(torch.ops.star_uvt_v0, "direct_projective_trace_cell_interval_backward")


def _require_peak_splat_projective_alpha(config) -> None:
    """Reject an alpha contract that the projective CPU oracles cannot mirror."""

    alpha_mode = getattr(config, "alpha_mode", "peak_splat")
    if alpha_mode != "peak_splat":
        raise ValueError(
            "projective-atlas rendering currently supports only alpha_mode='peak_splat'; "
            "use the validated core STAR-UVT q-UVT RGB direct_atomic path "
            "for beer_lambert"
        )


def _make_projective_interval_meta(config, device: torch.device, trace_count: int) -> tuple[Tensor, Tensor]:
    _require_peak_splat_projective_alpha(config)
    if int(config.height) <= 0 or int(config.width) <= 0 or int(config.frames) <= 0:
        raise ValueError("height, width, and frames must be positive")
    if int(config.tile_x) not in (8, 16):
        raise ValueError("tile_x must be 8 or 16")
    if int(config.tile_y) not in (8, 16):
        raise ValueError("tile_y must be 8 or 16")
    if int(config.tile_capacity) not in (32, 64, 128, 256):
        raise ValueError("tile_capacity must be 32, 64, 128, or 256")
    tiles_x = (int(config.width) + int(config.tile_x) - 1) // int(config.tile_x)
    tiles_y = (int(config.height) + int(config.tile_y) - 1) // int(config.tile_y)
    tile_count = tiles_x * tiles_y
    meta_i32 = torch.tensor(
        [
            int(config.height),
            int(config.width),
            int(config.frames),
            int(config.tile_x),
            int(config.tile_y),
            int(config.frames),
            tiles_x,
            tiles_y,
            1,
            tile_count,
            int(trace_count),
            int(config.tile_capacity),
            0,
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
            1.0e-8,
            float(config.max_alpha),
            float(
                {"peak_splat": 0, "beer_lambert": 1}[
                    getattr(config, "alpha_mode", "peak_splat")
                ]
            ),
        ],
        device=device,
        dtype=torch.float32,
    )
    return meta_i32, meta_f32


def _make_projective_family_interval_meta(
    config,
    device: torch.device,
    *,
    base_trace_count: int,
    q_count: int,
    basis_count: int,
) -> tuple[Tensor, Tensor]:
    meta_i32, meta_f32 = _make_projective_interval_meta(
        config,
        device,
        int(base_trace_count) * int(q_count),
    )
    meta_i32 = meta_i32.clone()
    meta_i32[12] = int(base_trace_count)
    meta_i32[13] = int(basis_count)
    return meta_i32, meta_f32


def render_projective_trace_tile_time_atlas_metal(
    cells: list[ProjectiveTraceTileTimeCell],
    coeffs: Tensor,
    times: Tensor,
    colors: Tensor,
    opacities: Tensor,
    config,
    *,
    sigma_px: float,
    allow_fallback_cells: bool = False,
) -> Tensor:
    """Render packed projective/rational atlas cells with the native Metal kernel."""

    _check_projective_trace_render_inputs(coeffs, times, colors, opacities)
    _require_peak_splat_projective_alpha(config)
    if coeffs.device.type != "mps":
        raise ValueError("projective atlas Metal render requires MPS tensors")
    if colors.shape[1] != 3:
        raise ValueError("projective atlas Metal render currently requires RGB colors")
    if int(times.numel()) != int(config.frames):
        raise ValueError("times must contain config.frames samples")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    primitive_count = int(coeffs.shape[0])
    for cell in cells:
        for primitive_id in cell.ordered_primitive_ids:
            if primitive_id < 0 or primitive_id >= primitive_count:
                raise ValueError("projective atlas Metal cells must use primitive ids in [0, N)")
    if not has_projective_trace_tile_metal():
        raise RuntimeError("star_uvt_v0.render_projective_trace_tiles custom op not found. Rebuild the extension.")

    from .rasterize import _make_meta, _runtime_validate

    _runtime_validate(config)
    bins = pack_projective_trace_tile_time_bins(
        cells,
        image_width=int(config.width),
        image_height=int(config.height),
        frames=int(config.frames),
        tile_x=int(config.tile_x),
        tile_y=int(config.tile_y),
        tile_t=int(config.tile_t),
        tile_capacity=int(config.tile_capacity),
        device=coeffs.device,
        allow_fallback_cells=allow_fallback_cells,
    )
    if bool(torch.any(bins.tile_overflow > 0).item()):
        raise ValueError("packed projective atlas tile capacity overflow")

    meta_i32, meta_f32 = _make_meta(config, coeffs.device, primitive_count)
    return torch.ops.star_uvt_v0.render_projective_trace_tiles(
        coeffs.contiguous(),
        times.contiguous(),
        opacities.contiguous(),
        colors.contiguous(),
        bins.tile_counts,
        bins.tile_primitive_ids,
        bins.tile_active_start,
        bins.tile_active_stop,
        meta_i32,
        meta_f32,
        float(sigma_px),
    )


def render_projective_trace_cell_atlas_metal(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    config,
    *,
    sigma_px: float,
    allow_fallback_cells: bool = False,
) -> Tensor:
    """Render packed cell-local polynomial traces with the native Metal kernel."""

    _check_projective_trace_render_inputs(atlas.coeffs, times, atlas.color, atlas.opacity)
    _require_peak_splat_projective_alpha(config)
    _require_no_temporal_opacity_for_cell_metal(atlas)
    if atlas.coeffs.device.type != "mps":
        raise ValueError("projective cell atlas Metal render requires MPS tensors")
    if atlas.color.shape[1] != 3:
        raise ValueError("projective cell atlas Metal render currently requires RGB colors")
    if int(times.numel()) != int(config.frames):
        raise ValueError("times must contain config.frames samples")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if not has_projective_trace_cell_metal():
        raise RuntimeError("star_uvt_v0.render_projective_trace_cell_tiles custom op not found. Rebuild the extension.")

    from .rasterize import _make_meta, _runtime_validate

    _runtime_validate(config)
    bins = pack_projective_trace_tile_time_bins(
        atlas.cells,
        image_width=int(config.width),
        image_height=int(config.height),
        frames=int(config.frames),
        tile_x=int(config.tile_x),
        tile_y=int(config.tile_y),
        tile_t=int(config.tile_t),
        tile_capacity=int(config.tile_capacity),
        device=atlas.coeffs.device,
        allow_fallback_cells=allow_fallback_cells,
    )
    if bool(torch.any(bins.tile_overflow > 0).item()):
        raise ValueError("packed projective cell atlas tile capacity overflow")

    meta_i32, meta_f32 = _make_meta(config, atlas.coeffs.device, int(atlas.coeffs.shape[0]))
    return torch.ops.star_uvt_v0.render_projective_trace_cell_tiles(
        atlas.coeffs.contiguous(),
        times.contiguous(),
        atlas.opacity.contiguous(),
        atlas.color.contiguous(),
        bins.tile_counts,
        bins.tile_primitive_ids,
        bins.tile_active_start,
        bins.tile_active_stop,
        meta_i32,
        meta_f32,
        float(sigma_px),
    )


def render_projective_trace_cell_interval_atlas_metal(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    config,
    *,
    sigma_px: float,
    allow_fallback_cells: bool = False,
) -> Tensor:
    """Render cell-local traces from interval-compressed spatial tile bins.

    Unlike ``render_projective_trace_cell_atlas_metal``, this path packs each
    accepted cell once into a spatial tile bin and relies on per-entry
    ``[active_start, active_stop)`` checks in the Metal kernel. It is the first
    hot-path shape that consumes the interval-compressed atlas object directly
    instead of re-expanding it into fixed temporal slabs.
    """

    _check_projective_trace_render_inputs(atlas.coeffs, times, atlas.color, atlas.opacity)
    if atlas.coeffs.device.type != "mps":
        raise ValueError("projective interval cell atlas Metal render requires MPS tensors")
    if atlas.color.shape[1] != 3:
        raise ValueError("projective interval cell atlas Metal render currently requires RGB colors")
    if int(times.numel()) != int(config.frames):
        raise ValueError("times must contain config.frames samples")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if not has_projective_trace_cell_interval_metal():
        raise RuntimeError("star_uvt_v0.render_projective_trace_cell_interval_tiles custom op not found. Rebuild the extension.")

    bins = pack_projective_trace_tile_time_bins(
        atlas.cells,
        image_width=int(config.width),
        image_height=int(config.height),
        frames=int(config.frames),
        tile_x=int(config.tile_x),
        tile_y=int(config.tile_y),
        tile_t=int(config.frames),
        tile_capacity=int(config.tile_capacity),
        device=atlas.coeffs.device,
        allow_fallback_cells=allow_fallback_cells,
    )
    if bool(torch.any(bins.tile_overflow > 0).item()):
        raise ValueError("packed projective interval atlas tile capacity overflow")

    meta_i32, meta_f32 = _make_projective_interval_meta(config, atlas.coeffs.device, int(atlas.coeffs.shape[0]))
    return torch.ops.star_uvt_v0.render_projective_trace_cell_interval_tiles(
        atlas.coeffs.contiguous(),
        times.contiguous(),
        atlas.opacity.contiguous(),
        _cell_opacity_time_coeffs_or_zeros(atlas).contiguous(),
        _cell_spatial_precision_uv_or_isotropic(atlas, sigma_px=float(sigma_px)).contiguous(),
        _cell_depth_affine_uv_or_zeros(atlas).contiguous(),
        atlas.color.contiguous(),
        bins.tile_counts,
        bins.tile_primitive_ids,
        bins.tile_active_start,
        bins.tile_active_stop,
        meta_i32,
        meta_f32,
        float(sigma_px),
    )


def render_projective_trace_family_interval_atlas_metal(
    cells: list[ProjectiveTraceTileTimeCell],
    family_coeffs: Tensor,
    q_basis: Tensor,
    times: Tensor,
    opacity: Tensor,
    opacity_time_coeffs: Tensor,
    spatial_precision_uv: Tensor,
    depth_affine_uv: Tensor,
    color: Tensor,
    config,
    *,
    sigma_px: float,
    allow_fallback_cells: bool = False,
) -> Tensor:
    """Render interval-compressed Q-family traces without materializing coeff rows.

    ``tile_trace_ids`` in ``cells`` are global ids ``q_index * N + trace_id``.
    The Metal kernel derives ``(q_index, trace_id)`` from that id, contracts
    ``family_coeffs[N,9,B]`` with ``q_basis[Q,B]`` in shader code, then uses
    the existing interval ordering/compositing semantics.
    """

    _check_projective_trace_family_inputs(family_coeffs, q_basis, times)
    if opacity.shape != (int(family_coeffs.shape[0]),):
        raise ValueError("opacity must have shape [N] matching family_coeffs")
    if opacity_time_coeffs.shape != (int(family_coeffs.shape[0]), 3):
        raise ValueError("opacity_time_coeffs must have shape [N,3] matching family_coeffs")
    if spatial_precision_uv.shape != (int(family_coeffs.shape[0]), 3):
        raise ValueError("spatial_precision_uv must have shape [N,3] matching family_coeffs")
    if depth_affine_uv.shape != (int(family_coeffs.shape[0]), 6):
        raise ValueError("depth_affine_uv must have shape [N,6] matching family_coeffs")
    if color.shape != (int(family_coeffs.shape[0]), 3):
        raise ValueError("color must have shape [N,3] matching family_coeffs")
    for name, tensor in (
        ("opacity", opacity),
        ("opacity_time_coeffs", opacity_time_coeffs),
        ("spatial_precision_uv", spatial_precision_uv),
        ("depth_affine_uv", depth_affine_uv),
        ("color", color),
    ):
        if tensor.dtype != torch.float32 or tensor.device != family_coeffs.device:
            raise ValueError(f"{name} must be float32 and on the same device as family_coeffs")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if family_coeffs.device.type != "mps":
        raise ValueError("projective family interval Metal render requires MPS tensors")
    if int(times.numel()) != int(config.frames):
        raise ValueError("times must contain config.frames samples")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    total_trace_count = int(family_coeffs.shape[0]) * int(q_basis.shape[0])
    for cell in cells:
        for primitive_id in cell.ordered_primitive_ids:
            if primitive_id < 0 or primitive_id >= total_trace_count:
                raise ValueError("family interval cells must use global ids in [0, Q*N)")
    if not has_projective_trace_family_interval_metal():
        raise RuntimeError(
            "star_uvt_v0.render_projective_trace_family_interval_tiles custom op not found. Rebuild the extension."
        )

    bins = pack_projective_trace_tile_time_bins(
        cells,
        image_width=int(config.width),
        image_height=int(config.height),
        frames=int(config.frames),
        tile_x=int(config.tile_x),
        tile_y=int(config.tile_y),
        tile_t=int(config.frames),
        tile_capacity=int(config.tile_capacity),
        device=family_coeffs.device,
        allow_fallback_cells=allow_fallback_cells,
    )
    if bool(torch.any(bins.tile_overflow > 0).item()):
        raise ValueError("packed projective family interval atlas tile capacity overflow")

    meta_i32, meta_f32 = _make_projective_family_interval_meta(
        config,
        family_coeffs.device,
        base_trace_count=int(family_coeffs.shape[0]),
        q_count=int(q_basis.shape[0]),
        basis_count=int(family_coeffs.shape[2]),
    )
    return torch.ops.star_uvt_v0.render_projective_trace_family_interval_tiles(
        family_coeffs.contiguous(),
        q_basis.contiguous(),
        times.contiguous(),
        opacity.contiguous(),
        opacity_time_coeffs.contiguous(),
        spatial_precision_uv.contiguous(),
        depth_affine_uv.contiguous(),
        color.contiguous(),
        bins.tile_counts,
        bins.tile_primitive_ids,
        bins.tile_active_start,
        bins.tile_active_stop,
        meta_i32,
        meta_f32,
        float(sigma_px),
    )


def direct_backward_projective_trace_family_interval_atlas_metal(
    cells: list[ProjectiveTraceTileTimeCell],
    family_coeffs: Tensor,
    q_basis: Tensor,
    times: Tensor,
    opacity: Tensor,
    opacity_time_coeffs: Tensor,
    spatial_precision_uv: Tensor,
    depth_affine_uv: Tensor,
    color: Tensor,
    grad_image: Tensor,
    config,
    *,
    sigma_px: float,
    allow_fallback_cells: bool = False,
) -> ProjectiveTraceFamilyAtlasGrad:
    """Direct VJP for interval-compressed Q-family traces.

    Visibility order and tile membership are treated as compiled constants,
    matching ``direct_backward_projective_trace_cell_interval_atlas_metal``.
    The kernel contracts the q-family coefficients in shader code and chains
    center-coefficient gradients back into ``family_coeffs`` and ``q_basis``.
    """

    _check_projective_trace_family_inputs(family_coeffs, q_basis, times)
    base_count = int(family_coeffs.shape[0])
    if opacity.shape != (base_count,):
        raise ValueError("opacity must have shape [N] matching family_coeffs")
    if opacity_time_coeffs.shape != (base_count, 3):
        raise ValueError("opacity_time_coeffs must have shape [N,3] matching family_coeffs")
    if spatial_precision_uv.shape != (base_count, 3):
        raise ValueError("spatial_precision_uv must have shape [N,3] matching family_coeffs")
    if depth_affine_uv.shape != (base_count, 6):
        raise ValueError("depth_affine_uv must have shape [N,6] matching family_coeffs")
    if color.shape != (base_count, 3):
        raise ValueError("color must have shape [N,3] matching family_coeffs")
    for name, tensor in (
        ("opacity", opacity),
        ("opacity_time_coeffs", opacity_time_coeffs),
        ("spatial_precision_uv", spatial_precision_uv),
        ("depth_affine_uv", depth_affine_uv),
        ("color", color),
    ):
        if tensor.dtype != torch.float32 or tensor.device != family_coeffs.device:
            raise ValueError(f"{name} must be float32 and on the same device as family_coeffs")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if family_coeffs.device.type != "mps":
        raise ValueError("projective family interval Metal backward requires MPS tensors")
    if grad_image.shape != (int(config.frames), int(config.height), int(config.width), 3):
        raise ValueError("grad_image must have shape [frames,height,width,3]")
    if grad_image.dtype != torch.float32 or grad_image.device != family_coeffs.device:
        raise ValueError("grad_image must be float32 and on the same device as family_coeffs")
    if not grad_image.is_contiguous():
        raise ValueError("grad_image must be contiguous")
    if int(times.numel()) != int(config.frames):
        raise ValueError("times must contain config.frames samples")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    total_trace_count = base_count * int(q_basis.shape[0])
    for cell in cells:
        for primitive_id in cell.ordered_primitive_ids:
            if primitive_id < 0 or primitive_id >= total_trace_count:
                raise ValueError("family interval cells must use global ids in [0, Q*N)")
    if not has_projective_trace_family_interval_backward_metal():
        raise RuntimeError(
            "star_uvt_v0.direct_projective_trace_family_interval_backward custom op not found. Rebuild the extension."
        )

    bins = pack_projective_trace_tile_time_bins(
        cells,
        image_width=int(config.width),
        image_height=int(config.height),
        frames=int(config.frames),
        tile_x=int(config.tile_x),
        tile_y=int(config.tile_y),
        tile_t=int(config.frames),
        tile_capacity=int(config.tile_capacity),
        device=family_coeffs.device,
        allow_fallback_cells=allow_fallback_cells,
    )
    if bool(torch.any(bins.tile_overflow > 0).item()):
        raise ValueError("packed projective family interval atlas tile capacity overflow")

    meta_i32, meta_f32 = _make_projective_family_interval_meta(
        config,
        family_coeffs.device,
        base_trace_count=base_count,
        q_count=int(q_basis.shape[0]),
        basis_count=int(family_coeffs.shape[2]),
    )
    (
        grad_family_coeffs,
        grad_q_basis,
        grad_opacity,
        grad_opacity_time_coeffs,
        grad_spatial_precision_uv,
        grad_color,
    ) = torch.ops.star_uvt_v0.direct_projective_trace_family_interval_backward(
        family_coeffs.contiguous(),
        q_basis.contiguous(),
        times.contiguous(),
        opacity.contiguous(),
        opacity_time_coeffs.contiguous(),
        spatial_precision_uv.contiguous(),
        depth_affine_uv.contiguous(),
        color.contiguous(),
        grad_image.contiguous(),
        bins.tile_counts,
        bins.tile_primitive_ids,
        bins.tile_active_start,
        bins.tile_active_stop,
        meta_i32,
        meta_f32,
        float(sigma_px),
    )
    return ProjectiveTraceFamilyAtlasGrad(
        grad_family_coeffs=grad_family_coeffs,
        grad_q_basis=grad_q_basis,
        grad_opacity=grad_opacity,
        grad_color=grad_color,
        grad_opacity_time_coeffs=grad_opacity_time_coeffs,
        grad_spatial_precision_uv=grad_spatial_precision_uv,
    )


def render_projective_trace_cell_interval_atlas_rows_metal(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    row_weights: Tensor,
    config,
    *,
    sigma_px: float,
    allow_fallback_cells: bool = False,
) -> Tensor:
    """Render a weighted row gather from interval-compressed cell traces.

    ``row_weights`` has shape ``[samples, height]``. The Metal kernel renders
    only the final image, skipping sample/row pairs whose weight is zero.
    """

    _check_projective_trace_render_inputs(atlas.coeffs, times, atlas.color, atlas.opacity)
    if atlas.coeffs.device.type != "mps":
        raise ValueError("projective interval row Metal render requires MPS tensors")
    if atlas.color.shape[1] != 3:
        raise ValueError("projective interval row Metal render currently requires RGB colors")
    if int(times.numel()) != int(config.frames):
        raise ValueError("times must contain config.frames samples")
    if row_weights.shape != (int(config.frames), int(config.height)):
        raise ValueError("row_weights must have shape [frames,height]")
    if row_weights.dtype != torch.float32 or row_weights.device != atlas.coeffs.device:
        raise ValueError("row_weights must be float32 and on the same device as atlas coeffs")
    if not row_weights.is_contiguous():
        raise ValueError("row_weights must be contiguous")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if not has_projective_trace_cell_interval_rows_metal():
        raise RuntimeError("star_uvt_v0.render_projective_trace_cell_interval_rows custom op not found. Rebuild the extension.")

    bins = pack_projective_trace_tile_time_bins(
        atlas.cells,
        image_width=int(config.width),
        image_height=int(config.height),
        frames=int(config.frames),
        tile_x=int(config.tile_x),
        tile_y=int(config.tile_y),
        tile_t=int(config.frames),
        tile_capacity=int(config.tile_capacity),
        device=atlas.coeffs.device,
        allow_fallback_cells=allow_fallback_cells,
    )
    if bool(torch.any(bins.tile_overflow > 0).item()):
        raise ValueError("packed projective interval row atlas tile capacity overflow")

    meta_i32, meta_f32 = _make_projective_interval_meta(config, atlas.coeffs.device, int(atlas.coeffs.shape[0]))
    return torch.ops.star_uvt_v0.render_projective_trace_cell_interval_rows(
        atlas.coeffs.contiguous(),
        times.contiguous(),
        atlas.opacity.contiguous(),
        _cell_opacity_time_coeffs_or_zeros(atlas).contiguous(),
        _cell_spatial_precision_uv_or_isotropic(atlas, sigma_px=float(sigma_px)).contiguous(),
        _cell_depth_affine_uv_or_zeros(atlas).contiguous(),
        atlas.color.contiguous(),
        bins.tile_counts,
        bins.tile_primitive_ids,
        bins.tile_active_start,
        bins.tile_active_stop,
        row_weights.contiguous(),
        meta_i32,
        meta_f32,
        float(sigma_px),
    )


def direct_backward_projective_trace_tile_time_atlas_metal(
    cells: list[ProjectiveTraceTileTimeCell],
    coeffs: Tensor,
    times: Tensor,
    colors: Tensor,
    opacities: Tensor,
    grad_image: Tensor,
    config,
    *,
    sigma_px: float,
    allow_fallback_cells: bool = False,
) -> ProjectiveTraceAtlasGrad:
    """Direct VJP for packed projective atlas cells through the native Metal op.

    The derivative follows the rendered projective footprint and alpha
    compositing while treating visibility order and tile membership as compiled
    constants.
    """

    _check_projective_trace_render_inputs(coeffs, times, colors, opacities)
    _require_peak_splat_projective_alpha(config)
    if coeffs.device.type != "mps":
        raise ValueError("projective atlas Metal backward requires MPS tensors")
    if colors.shape[1] != 3:
        raise ValueError("projective atlas Metal backward currently requires RGB colors")
    if grad_image.shape != (int(config.frames), int(config.height), int(config.width), 3):
        raise ValueError("grad_image must have shape [frames,height,width,3]")
    if grad_image.dtype != torch.float32 or grad_image.device != coeffs.device:
        raise ValueError("grad_image must be float32 and on the same device as coeffs")
    if int(times.numel()) != int(config.frames):
        raise ValueError("times must contain config.frames samples")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    primitive_count = int(coeffs.shape[0])
    for cell in cells:
        for primitive_id in cell.ordered_primitive_ids:
            if primitive_id < 0 or primitive_id >= primitive_count:
                raise ValueError("projective atlas Metal cells must use primitive ids in [0, N)")
    if not hasattr(torch.ops.star_uvt_v0, "direct_projective_trace_backward"):
        raise RuntimeError("star_uvt_v0.direct_projective_trace_backward custom op not found. Rebuild the extension.")

    from .rasterize import _make_meta, _runtime_validate

    _runtime_validate(config)
    bins = pack_projective_trace_tile_time_bins(
        cells,
        image_width=int(config.width),
        image_height=int(config.height),
        frames=int(config.frames),
        tile_x=int(config.tile_x),
        tile_y=int(config.tile_y),
        tile_t=int(config.tile_t),
        tile_capacity=int(config.tile_capacity),
        device=coeffs.device,
        allow_fallback_cells=allow_fallback_cells,
    )
    if bool(torch.any(bins.tile_overflow > 0).item()):
        raise ValueError("packed projective atlas tile capacity overflow")

    meta_i32, meta_f32 = _make_meta(config, coeffs.device, primitive_count)
    grad_coeffs, grad_opacity, grad_color = torch.ops.star_uvt_v0.direct_projective_trace_backward(
        coeffs.contiguous(),
        times.contiguous(),
        opacities.contiguous(),
        colors.contiguous(),
        grad_image.contiguous(),
        bins.tile_counts,
        bins.tile_primitive_ids,
        bins.tile_active_start,
        bins.tile_active_stop,
        meta_i32,
        meta_f32,
        float(sigma_px),
    )
    return ProjectiveTraceAtlasGrad(
        grad_coeffs=grad_coeffs,
        grad_opacity=grad_opacity,
        grad_color=grad_color,
    )


def direct_backward_projective_trace_cell_interval_atlas_metal(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: Tensor,
    grad_image: Tensor,
    config,
    *,
    sigma_px: float,
    allow_fallback_cells: bool = False,
) -> ProjectiveTraceAtlasGrad:
    """Direct VJP for interval-compressed cell-local projective traces.

    This consumes the same spatial tile bins as
    ``render_projective_trace_cell_interval_atlas_metal``. Visibility order and
    tile membership are treated as compiled atlas constants.
    """

    _check_projective_trace_render_inputs(atlas.coeffs, times, atlas.color, atlas.opacity)
    if atlas.coeffs.device.type != "mps":
        raise ValueError("projective interval cell atlas Metal backward requires MPS tensors")
    if atlas.color.shape[1] != 3:
        raise ValueError("projective interval cell atlas Metal backward currently requires RGB colors")
    if grad_image.shape != (int(config.frames), int(config.height), int(config.width), 3):
        raise ValueError("grad_image must have shape [frames,height,width,3]")
    if grad_image.dtype != torch.float32 or grad_image.device != atlas.coeffs.device:
        raise ValueError("grad_image must be float32 and on the same device as atlas coeffs")
    if int(times.numel()) != int(config.frames):
        raise ValueError("times must contain config.frames samples")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if not has_projective_trace_cell_interval_backward_metal():
        raise RuntimeError("star_uvt_v0.direct_projective_trace_cell_interval_backward custom op not found. Rebuild the extension.")

    bins = pack_projective_trace_tile_time_bins(
        atlas.cells,
        image_width=int(config.width),
        image_height=int(config.height),
        frames=int(config.frames),
        tile_x=int(config.tile_x),
        tile_y=int(config.tile_y),
        tile_t=int(config.frames),
        tile_capacity=int(config.tile_capacity),
        device=atlas.coeffs.device,
        allow_fallback_cells=allow_fallback_cells,
    )
    if bool(torch.any(bins.tile_overflow > 0).item()):
        raise ValueError("packed projective interval atlas tile capacity overflow")

    meta_i32, meta_f32 = _make_projective_interval_meta(config, atlas.coeffs.device, int(atlas.coeffs.shape[0]))
    grad_coeffs, grad_opacity, grad_opacity_time_coeffs, grad_spatial_precision_uv, grad_color = torch.ops.star_uvt_v0.direct_projective_trace_cell_interval_backward(
        atlas.coeffs.contiguous(),
        times.contiguous(),
        atlas.opacity.contiguous(),
        _cell_opacity_time_coeffs_or_zeros(atlas).contiguous(),
        _cell_spatial_precision_uv_or_isotropic(atlas, sigma_px=float(sigma_px)).contiguous(),
        _cell_depth_affine_uv_or_zeros(atlas).contiguous(),
        atlas.color.contiguous(),
        grad_image.contiguous(),
        bins.tile_counts,
        bins.tile_primitive_ids,
        bins.tile_active_start,
        bins.tile_active_stop,
        meta_i32,
        meta_f32,
        float(sigma_px),
    )
    return ProjectiveTraceAtlasGrad(
        grad_coeffs=grad_coeffs,
        grad_opacity=grad_opacity,
        grad_color=grad_color,
        grad_opacity_time_coeffs=grad_opacity_time_coeffs,
        grad_spatial_precision_uv=grad_spatial_precision_uv if atlas.spatial_precision_uv is not None else None,
    )


def projective_trace_windows_to_uvt_tubes(
    windows: list[ProjectiveTraceWindow],
    *,
    sigma_px: float,
    opacity: Tensor,
    color: Tensor,
    primitive_ids: Tensor | list[int] | tuple[int, ...] | None = None,
    temporal_precision: float = 0.0,
    require_accepted: bool = True,
) -> ProjectiveTraceUVTBridge:
    """Lower accepted affine projective charts into the existing STAR UVT contract."""

    if not windows:
        raise ValueError("windows must not be empty")
    if sigma_px <= 0.0:
        raise ValueError("sigma_px must be positive")
    if temporal_precision < 0.0:
        raise ValueError("temporal_precision must be non-negative")

    first_fit = windows[0].fit
    if first_fit.poly_coeffs.ndim != 3 or first_fit.poly_coeffs.shape[1] != 3:
        raise ValueError("window fit poly_coeffs must have shape [N,3,D]")
    primitive_count = int(first_fit.poly_coeffs.shape[0])
    if opacity.shape != (primitive_count,):
        raise ValueError("opacity must have shape [N]")
    if color.ndim != 2 or color.shape[0] != primitive_count:
        raise ValueError("color must have shape [N,C]")
    if opacity.dtype != torch.float32 or color.dtype != torch.float32:
        raise ValueError("opacity and color must be float32")
    if opacity.device != first_fit.poly_coeffs.device or color.device != first_fit.poly_coeffs.device:
        raise ValueError("opacity, color, and windows must be on the same device")

    ids = list(range(primitive_count)) if primitive_ids is None else _to_1d_int_list(primitive_ids, expected=primitive_count, name="primitive_ids")
    ma_parts: list[Tensor] = []
    q_parts: list[Tensor] = []
    depth0_parts: list[Tensor] = []
    depth_beta_parts: list[Tensor] = []
    opacity_parts: list[Tensor] = []
    color_parts: list[Tensor] = []
    source_window_indices: list[int] = []
    source_primitive_ids: list[int] = []
    active_start: list[int] = []
    active_stop: list[int] = []

    for window_index, window in enumerate(windows):
        if require_accepted and not window.accepted:
            raise ValueError(f"cannot lower unresolved projective trace window: {window.reason}")
        fit = window.fit
        if fit.degree != 1:
            raise ValueError("only degree-1 projective chart windows can be lowered to q_uvt tubes")
        if fit.poly_coeffs.shape != first_fit.poly_coeffs.shape:
            raise ValueError("all windows must have matching primitive dimensions")
        if fit.poly_coeffs.dtype != torch.float32 or fit.poly_coeffs.device != first_fit.poly_coeffs.device:
            raise ValueError("all windows must be float32 on the same device")

        coeffs = fit.poly_coeffs
        inv_scale = 1.0 / fit.time_scale.clamp_min(1.0e-12)
        center_uv = coeffs[:, :2, 0]
        velocity_uv = coeffs[:, :2, 1] * inv_scale
        center_t = fit.time_center.expand(primitive_count, 1)
        ma_parts.append(torch.cat((center_uv, center_t), dim=-1))

        lambda_u = torch.full((primitive_count,), 1.0 / float(sigma_px * sigma_px), dtype=torch.float32, device=coeffs.device)
        lambda_v = lambda_u.clone()
        lambda_t = torch.full_like(lambda_u, float(temporal_precision))
        zeros = torch.zeros_like(lambda_u)
        velocity_u = velocity_uv[:, 0]
        velocity_v = velocity_uv[:, 1]
        q_parts.append(
            torch.stack(
                (
                    lambda_u,
                    zeros,
                    -lambda_u * velocity_u,
                    lambda_v,
                    -lambda_v * velocity_v,
                    lambda_t + lambda_u * velocity_u.square() + lambda_v * velocity_v.square(),
                ),
                dim=-1,
            )
        )

        depth0_parts.append(coeffs[:, 2, 0])
        depth_beta = torch.zeros((primitive_count, 3), dtype=torch.float32, device=coeffs.device)
        depth_beta[:, 2] = coeffs[:, 2, 1] * inv_scale
        depth_beta_parts.append(depth_beta)
        opacity_parts.append(opacity)
        color_parts.append(color)
        source_window_indices.extend([window_index] * primitive_count)
        source_primitive_ids.extend(ids)
        active_start.extend([window.start] * primitive_count)
        active_stop.extend([window.stop] * primitive_count)

    return ProjectiveTraceUVTBridge(
        ma=torch.cat(ma_parts, dim=0).contiguous(),
        q_uvt=torch.cat(q_parts, dim=0).contiguous(),
        depth0=torch.cat(depth0_parts, dim=0).contiguous(),
        depth_beta=torch.cat(depth_beta_parts, dim=0).contiguous(),
        opacity=torch.cat(opacity_parts, dim=0).contiguous(),
        color=torch.cat(color_parts, dim=0).contiguous(),
        source_window_indices=tuple(source_window_indices),
        source_primitive_ids=tuple(source_primitive_ids),
        active_start=tuple(active_start),
        active_stop=tuple(active_stop),
    )


def render_projective_trace_uvt_bridge_reference(
    bridge: ProjectiveTraceUVTBridge,
    *,
    image_width: int,
    image_height: int,
    frame_times: Tensor,
    alpha_threshold: float = 0.0,
    transmittance_threshold: float = 0.0,
    background: tuple[float, ...] | None = None,
    max_alpha: float = 1.0,
    use_window_gates: bool = True,
) -> Tensor:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if frame_times.ndim != 1:
        raise ValueError("frame_times must have shape [S]")
    if frame_times.dtype != torch.float32:
        raise ValueError("frame_times must be float32")
    if frame_times.device != bridge.ma.device:
        raise ValueError("frame_times and bridge tensors must be on the same device")
    if alpha_threshold < 0.0:
        raise ValueError("alpha_threshold must be non-negative")
    if transmittance_threshold < 0.0:
        raise ValueError("transmittance_threshold must be non-negative")
    if max_alpha <= 0.0:
        raise ValueError("max_alpha must be positive")
    tube_count = int(bridge.ma.shape[0])
    if bridge.q_uvt.shape != (tube_count, 6):
        raise ValueError("bridge.q_uvt must have shape [M,6]")
    if bridge.depth0.shape != (tube_count,) or bridge.depth_beta.shape != (tube_count, 3):
        raise ValueError("bridge depth tensors have invalid shapes")
    if bridge.opacity.shape != (tube_count,) or bridge.color.ndim != 2 or bridge.color.shape[0] != tube_count:
        raise ValueError("bridge opacity/color tensors have invalid shapes")
    if len(bridge.active_start) != tube_count or len(bridge.active_stop) != tube_count:
        raise ValueError("bridge active windows must match tube count")

    device = bridge.ma.device
    channels = int(bridge.color.shape[1])
    if background is None:
        bg = torch.zeros((channels,), dtype=torch.float32, device=device)
    else:
        if len(background) != channels:
            raise ValueError("background must match color channels")
        bg = torch.tensor(background, dtype=torch.float32, device=device)
    out = torch.empty((int(frame_times.numel()), image_height, image_width, channels), dtype=torch.float32, device=device)
    active_start = torch.tensor(bridge.active_start, dtype=torch.int64, device=device)
    active_stop = torch.tensor(bridge.active_stop, dtype=torch.int64, device=device)

    for sample_index in range(int(frame_times.numel())):
        t = frame_times[sample_index]
        for y in range(image_height):
            for x in range(image_width):
                a = torch.tensor([x + 0.5, y + 0.5, t], dtype=torch.float32, device=device)
                delta = a.unsqueeze(0) - bridge.ma
                qv = (
                    bridge.q_uvt[:, 0] * delta[:, 0].square()
                    + 2.0 * bridge.q_uvt[:, 1] * delta[:, 0] * delta[:, 1]
                    + 2.0 * bridge.q_uvt[:, 2] * delta[:, 0] * delta[:, 2]
                    + bridge.q_uvt[:, 3] * delta[:, 1].square()
                    + 2.0 * bridge.q_uvt[:, 4] * delta[:, 1] * delta[:, 2]
                    + bridge.q_uvt[:, 5] * delta[:, 2].square()
                )
                alpha = torch.clamp(bridge.opacity * torch.exp(-0.5 * qv), max=float(max_alpha))
                if use_window_gates:
                    active_window = (active_start <= sample_index) & (sample_index < active_stop)
                    alpha = torch.where(active_window, alpha, torch.zeros_like(alpha))
                active = torch.nonzero(alpha >= float(alpha_threshold), as_tuple=False).flatten()
                if active.numel() == 0:
                    out[sample_index, y, x] = bg
                    continue
                depths = bridge.depth0 + (delta * bridge.depth_beta).sum(dim=-1)
                order = torch.argsort(depths.index_select(0, active), stable=True)
                accum = torch.zeros((channels,), dtype=torch.float32, device=device)
                transmittance = torch.tensor(1.0, dtype=torch.float32, device=device)
                for local_idx in order.tolist():
                    tube_id = int(active[local_idx])
                    alpha_i = alpha[tube_id]
                    accum = accum + transmittance * alpha_i * bridge.color[tube_id]
                    transmittance = transmittance * (1.0 - alpha_i)
                    if float(transmittance) <= transmittance_threshold:
                        break
                out[sample_index, y, x] = accum + transmittance * bg

    return out


def projective_trace_uvt_bridge_active_spans(
    bridge: ProjectiveTraceUVTBridge,
    *,
    frames: int,
) -> tuple[tuple[int, int], ...]:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if len(bridge.active_start) != len(bridge.active_stop):
        raise ValueError("bridge active windows must have matching lengths")
    boundaries = {0, int(frames)}
    for start, stop in zip(bridge.active_start, bridge.active_stop):
        if start < 0 or stop > frames or start >= stop:
            raise ValueError("bridge active windows must be valid frame intervals")
        boundaries.add(int(start))
        boundaries.add(int(stop))
    ordered = sorted(boundaries)
    return tuple((start, stop) for start, stop in zip(ordered, ordered[1:]) if start < stop)


def render_projective_trace_uvt_bridge_metal_gated(
    bridge: ProjectiveTraceUVTBridge,
    config,
) -> Tensor:
    """Render an interval-gated q-UVT bridge through the native gated Metal op."""

    from .rasterize import render_uvt_tubes_gated

    if bridge.ma.device.type != "mps":
        raise ValueError("interval-gated Metal bridge requires MPS tensors")
    if bridge.color.ndim != 2 or bridge.color.shape[1] != 3:
        raise ValueError("STAR UVT Metal bridge currently requires RGB color")
    tube_count = int(bridge.ma.shape[0])
    if len(bridge.active_start) != tube_count or len(bridge.active_stop) != tube_count:
        raise ValueError("bridge active windows must match tube count")

    active_start = torch.tensor(bridge.active_start, dtype=torch.int64, device=bridge.ma.device)
    active_stop = torch.tensor(bridge.active_stop, dtype=torch.int64, device=bridge.ma.device)
    return render_uvt_tubes_gated(
        bridge.ma,
        bridge.q_uvt,
        bridge.depth0,
        bridge.depth_beta,
        bridge.opacity,
        bridge.color,
        active_start.to(dtype=torch.int32),
        active_stop.to(dtype=torch.int32),
        config,
    )


def direct_backward_projective_trace_uvt_bridge_metal_gated(
    bridge: ProjectiveTraceUVTBridge,
    grad_image: Tensor,
    config,
) -> ProjectiveTraceUVTBridgeGrad:
    """Direct VJP for an interval-gated q-UVT bridge through native Metal."""

    from .rasterize import direct_atomic_backward_gated

    if bridge.ma.device.type != "mps":
        raise ValueError("interval-gated Metal bridge backward requires MPS tensors")
    if bridge.color.ndim != 2 or bridge.color.shape[1] != 3:
        raise ValueError("STAR UVT Metal bridge backward currently requires RGB color")
    tube_count = int(bridge.ma.shape[0])
    if len(bridge.active_start) != tube_count or len(bridge.active_stop) != tube_count:
        raise ValueError("bridge active windows must match tube count")
    if grad_image.shape != (config.frames, config.height, config.width, 3):
        raise ValueError("grad_image must have shape [frames,height,width,3]")
    if grad_image.dtype != torch.float32 or grad_image.device != bridge.ma.device:
        raise ValueError("grad_image must be float32 and on the same device as bridge tensors")

    active_start = torch.tensor(bridge.active_start, dtype=torch.int32, device=bridge.ma.device)
    active_stop = torch.tensor(bridge.active_stop, dtype=torch.int32, device=bridge.ma.device)
    grad_ma, grad_q, grad_opacity, grad_color, tile_unstable = direct_atomic_backward_gated(
        bridge.ma,
        bridge.q_uvt,
        bridge.depth0,
        bridge.depth_beta,
        bridge.opacity,
        bridge.color,
        grad_image,
        active_start,
        active_stop,
        config,
    )
    return ProjectiveTraceUVTBridgeGrad(
        grad_ma=grad_ma,
        grad_q_uvt=grad_q,
        grad_depth0=torch.zeros_like(bridge.depth0),
        grad_depth_beta=torch.zeros_like(bridge.depth_beta),
        grad_opacity=grad_opacity,
        grad_color=grad_color,
        tile_unstable=tile_unstable,
    )


def split_projective_trace_windows(
    coeffs: Tensor,
    times: Tensor,
    *,
    degree: int = 1,
    eps: float = 1.0e-6,
    max_residual_uv: float = 1.0,
    max_depth_residual: float = float("inf"),
    min_denominator_abs: float = 1.0e-6,
    min_valid_fraction: float = 1.0,
    min_samples: int | None = None,
    max_windows: int = 1024,
) -> list[ProjectiveTraceWindow]:
    _check_projective_trace_inputs(coeffs, times)
    if times.numel() == 0:
        raise ValueError("times must not be empty")
    if max_residual_uv < 0.0:
        raise ValueError("max_residual_uv must be non-negative")
    if max_depth_residual < 0.0:
        raise ValueError("max_depth_residual must be non-negative")
    if min_denominator_abs < 0.0:
        raise ValueError("min_denominator_abs must be non-negative")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in [0, 1]")
    if max_windows < 1:
        raise ValueError("max_windows must be positive")

    min_window_samples = max(degree + 1, int(min_samples or (degree + 1)))
    pending = [(0, int(times.numel()))]
    windows: list[ProjectiveTraceWindow] = []

    while pending:
        start, stop = pending.pop()
        local_times = times[start:stop].contiguous()
        fit = fit_projective_trace_polynomial(coeffs, local_times, degree=degree, eps=eps)
        domain_times = times[start : min(stop + 1, int(times.numel()))].contiguous()
        domain_denominator_has_root, domain_denominator_min_abs = _denominator_interval_certificate(coeffs, domain_times)
        fit = replace(
            fit,
            denominator_min_abs=domain_denominator_min_abs,
            denominator_has_root=domain_denominator_has_root,
        )

        uv_ok = bool(torch.all(fit.residual_max_uv <= float(max_residual_uv)).item())
        depth_ok = bool(torch.all(fit.residual_max_depth <= float(max_depth_residual)).item())
        denom_margin_ok = bool(torch.all(fit.denominator_min_abs >= float(min_denominator_abs)).item())
        denom_root_ok = bool(torch.all(~fit.denominator_has_root).item())
        denom_ok = denom_margin_ok and denom_root_ok
        valid_ok = bool(torch.all(fit.valid_fraction >= float(min_valid_fraction)).item())
        accepted = uv_ok and depth_ok and denom_ok and valid_ok

        reason_parts = []
        if not uv_ok:
            reason_parts.append("uv_residual")
        if not depth_ok:
            reason_parts.append("depth_residual")
        if not denom_margin_ok:
            reason_parts.append("denominator")
        if not denom_root_ok:
            reason_parts.append("denominator_boundary")
        if not valid_ok:
            reason_parts.append("invalid_samples")

        if accepted:
            windows.append(ProjectiveTraceWindow(start, stop, True, "accepted", fit))
            continue

        if (stop - start) <= min_window_samples:
            windows.append(ProjectiveTraceWindow(start, stop, False, ",".join(reason_parts), fit))
            continue

        if len(windows) + len(pending) + 2 > max_windows:
            windows.append(ProjectiveTraceWindow(start, stop, False, "max_windows", fit))
            continue

        mid = (start + stop) // 2
        pending.append((mid, stop))
        pending.append((start, mid))

    return sorted(windows, key=lambda window: window.start)
