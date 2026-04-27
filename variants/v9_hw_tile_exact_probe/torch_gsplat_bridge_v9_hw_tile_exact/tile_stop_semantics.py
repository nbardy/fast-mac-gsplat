from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


GaussianSplat = tuple[float, float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class TileStopGapCase:
    name: str
    height: int
    width: int
    tile_size: int
    splats: tuple[GaussianSplat, ...]
    note: str


def default_tile_stop_gap_cases() -> tuple[TileStopGapCase, ...]:
    """Tiny scenes where V8 candidate-prefix stop counts differ from V9 fullscreen draws."""
    return (
        TileStopGapCase(
            name="one_tile_skipped_candidate",
            height=16,
            width=16,
            tile_size=16,
            splats=(
                (8.0, 8.0, 0.035, 0.0, 0.035, 0.55, 1.0, 0.0, 0.0),
                (15.5, 15.5, 0.50, 0.0, 0.50, 0.75, 0.0, 1.0, 0.0),
                (8.0, 8.0, 0.035, 0.0, 0.035, 0.00, 0.0, 0.0, 1.0),
            ),
            note=(
                "Splat 1 is a real tile candidate but invisible at many pixels; splat 2 is skipped by V8 "
                "alpha-support binning but would still be counted by the fullscreen diagnostic draw list."
            ),
        ),
        TileStopGapCase(
            name="two_tile_clipping_gap",
            height=16,
            width=32,
            tile_size=16,
            splats=(
                (8.0, 8.0, 0.25, 0.0, 0.25, 0.80, 1.0, 0.0, 0.0),
                (24.0, 8.0, 0.25, 0.0, 0.25, 0.80, 0.0, 1.0, 0.0),
                (60.0, 60.0, 0.25, 0.0, 0.25, 0.80, 0.0, 0.0, 1.0),
            ),
            note=(
                "Each visible splat belongs to only one tile and splat 2 is off-screen; fullscreen "
                "instances count all three candidates in both tiles."
            ),
        ),
    )


def compare_v8_candidate_prefix_tile_stop_gap(
    cases: tuple[TileStopGapCase, ...] | None = None,
    *,
    alpha_threshold: float = 1.0 / 255.0,
    transmittance_threshold: float = 1.0e-4,
    max_alpha: float = 0.99,
    eps: float = 1.0e-8,
) -> list[dict[str, Any]]:
    """Compare V8 tile-bin candidate-prefix state with current V9 fullscreen diagnostic state.

    V8 backward consumes `tile_stop_counts` over the sorted per-tile candidate prefix. The committed
    V9 probe still draws every diagnostic splat fullscreen, so skipped/off-tile candidates are included
    unless the draw stream is replaced by clipped quads or tile-bin-fed records.
    """
    selected_cases = cases or default_tile_stop_gap_cases()
    reports: list[dict[str, Any]] = []
    for case in selected_cases:
        _validate_case(case)
        v8_tile_candidates = _v8_tile_candidates(
            case,
            alpha_threshold=alpha_threshold,
            eps=eps,
        )
        fullscreen_candidates = [list(range(len(case.splats))) for _ in range(_tile_count(case))]
        v8_stop_counts = _candidate_prefix_stop_counts(
            case,
            v8_tile_candidates,
            alpha_threshold=alpha_threshold,
            transmittance_threshold=transmittance_threshold,
            max_alpha=max_alpha,
        )
        fullscreen_stop_counts = _candidate_prefix_stop_counts(
            case,
            fullscreen_candidates,
            alpha_threshold=alpha_threshold,
            transmittance_threshold=transmittance_threshold,
            max_alpha=max_alpha,
        )
        gap_tile_ids = [
            idx for idx, (v8_count, v9_count) in enumerate(zip(v8_stop_counts, fullscreen_stop_counts))
            if v8_count != v9_count or v8_tile_candidates[idx] != fullscreen_candidates[idx]
        ]
        reports.append(
            {
                "case": case.name,
                "height": case.height,
                "width": case.width,
                "tile_size": case.tile_size,
                "tiles_x": _ceil_div(case.width, case.tile_size),
                "tiles_y": _ceil_div(case.height, case.tile_size),
                "splat_count": len(case.splats),
                "note": case.note,
                "v8_tile_candidates": v8_tile_candidates,
                "v8_candidate_prefix_tile_stop_counts": v8_stop_counts,
                "v9_fullscreen_draw_candidates": fullscreen_candidates,
                "v9_fullscreen_diagnostic_tile_stop_counts": fullscreen_stop_counts,
                "gap_tile_ids": gap_tile_ids,
                "parity_passed": not gap_tile_ids,
                "gate": "failed_as_expected_until_clipped_quads_or_tile_bins" if gap_tile_ids else "unexpected_parity",
            }
        )
    return reports


def assert_v8_candidate_prefix_gap_exposed() -> list[dict[str, Any]]:
    """Fail closed if the tiny cases no longer expose the V9/V8 state-contract gap."""
    reports = compare_v8_candidate_prefix_tile_stop_gap()
    unexpected = [report for report in reports if report["parity_passed"]]
    if unexpected:
        names = ", ".join(str(report["case"]) for report in unexpected)
        raise AssertionError(f"V9 fullscreen diagnostic unexpectedly matched V8 candidate-prefix state: {names}")
    return reports


def _validate_case(case: TileStopGapCase) -> None:
    if case.height <= 0 or case.width <= 0:
        raise ValueError("height and width must be positive")
    if case.tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not case.splats:
        raise ValueError("at least one splat is required")


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _tile_count(case: TileStopGapCase) -> int:
    return _ceil_div(case.width, case.tile_size) * _ceil_div(case.height, case.tile_size)


def _alpha_support_tau(opacity: float, *, alpha_threshold: float, eps: float) -> float | None:
    if opacity <= alpha_threshold:
        return None
    ratio = max(alpha_threshold / max(opacity, eps), eps)
    tau = -2.0 * math.log(ratio)
    if not math.isfinite(tau) or tau <= 0.0:
        return None
    return tau


def _snugbox(
    *,
    mean_x: float,
    mean_y: float,
    conic_x: float,
    conic_y: float,
    conic_z: float,
    tau: float,
    width: int,
    height: int,
    eps: float,
) -> tuple[int, int, int, int]:
    det = max(conic_x * conic_z - conic_y * conic_y, eps)
    half_x = math.sqrt(max(tau * conic_z / det, 0.0))
    half_y = math.sqrt(max(tau * conic_x / det, 0.0))
    x0 = max(0, math.floor(mean_x - half_x - 0.5))
    x1 = min(width - 1, math.ceil(mean_x + half_x - 0.5))
    y0 = max(0, math.floor(mean_y - half_y - 0.5))
    y1 = min(height - 1, math.ceil(mean_y + half_y - 0.5))
    return int(x0), int(y0), int(x1), int(y1)


def _ellipse_intersects_rect(
    *,
    mean_x: float,
    mean_y: float,
    conic_x: float,
    conic_y: float,
    conic_z: float,
    tau: float,
    rx0: float,
    ry0: float,
    rx1: float,
    ry1: float,
) -> bool:
    dx0 = rx0 - mean_x
    dx1 = rx1 - mean_x
    dy0 = ry0 - mean_y
    dy1 = ry1 - mean_y
    if rx0 <= mean_x <= rx1 and ry0 <= mean_y <= ry1:
        return True

    qmin = math.inf
    qmin = min(qmin, conic_x * dx0 * dx0 + 2.0 * conic_y * dx0 * dy0 + conic_z * dy0 * dy0)
    qmin = min(qmin, conic_x * dx0 * dx0 + 2.0 * conic_y * dx0 * dy1 + conic_z * dy1 * dy1)
    qmin = min(qmin, conic_x * dx1 * dx1 + 2.0 * conic_y * dx1 * dy0 + conic_z * dy0 * dy0)
    qmin = min(qmin, conic_x * dx1 * dx1 + 2.0 * conic_y * dx1 * dy1 + conic_z * dy1 * dy1)
    if conic_z > 1.0e-8:
        dy = min(max(-(conic_y / conic_z) * dx0, dy0), dy1)
        qmin = min(qmin, conic_x * dx0 * dx0 + 2.0 * conic_y * dx0 * dy + conic_z * dy * dy)
        dy = min(max(-(conic_y / conic_z) * dx1, dy0), dy1)
        qmin = min(qmin, conic_x * dx1 * dx1 + 2.0 * conic_y * dx1 * dy + conic_z * dy * dy)
    if conic_x > 1.0e-8:
        dx = min(max(-(conic_y / conic_x) * dy0, dx0), dx1)
        qmin = min(qmin, conic_x * dx * dx + 2.0 * conic_y * dx * dy0 + conic_z * dy0 * dy0)
        dx = min(max(-(conic_y / conic_x) * dy1, dx0), dx1)
        qmin = min(qmin, conic_x * dx * dx + 2.0 * conic_y * dx * dy1 + conic_z * dy1 * dy1)
    return qmin <= tau


def _v8_tile_candidates(
    case: TileStopGapCase,
    *,
    alpha_threshold: float,
    eps: float,
) -> list[list[int]]:
    tiles_x = _ceil_div(case.width, case.tile_size)
    tile_candidates: list[list[int]] = [[] for _ in range(_tile_count(case))]
    for splat_id, splat in enumerate(case.splats):
        mean_x, mean_y, conic_x, conic_y, conic_z, opacity, _, _, _ = splat
        tau = _alpha_support_tau(opacity, alpha_threshold=alpha_threshold, eps=eps)
        if tau is None:
            continue
        x0, y0, x1, y1 = _snugbox(
            mean_x=mean_x,
            mean_y=mean_y,
            conic_x=conic_x,
            conic_y=conic_y,
            conic_z=conic_z,
            tau=tau,
            width=case.width,
            height=case.height,
            eps=eps,
        )
        if x0 > x1 or y0 > y1:
            continue
        tx0 = x0 // case.tile_size
        tx1 = x1 // case.tile_size
        ty0 = y0 // case.tile_size
        ty1 = y1 // case.tile_size
        for ty in range(ty0, ty1 + 1):
            ry0 = float(ty * case.tile_size) + 0.5
            ry1 = min(float(case.height - 1) + 0.5, float((ty + 1) * case.tile_size - 1) + 0.5)
            for tx in range(tx0, tx1 + 1):
                rx0 = float(tx * case.tile_size) + 0.5
                rx1 = min(float(case.width - 1) + 0.5, float((tx + 1) * case.tile_size - 1) + 0.5)
                if _ellipse_intersects_rect(
                    mean_x=mean_x,
                    mean_y=mean_y,
                    conic_x=conic_x,
                    conic_y=conic_y,
                    conic_z=conic_z,
                    tau=tau,
                    rx0=rx0,
                    ry0=ry0,
                    rx1=rx1,
                    ry1=ry1,
                ):
                    tile_candidates[ty * tiles_x + tx].append(splat_id)
    return [sorted(candidates) for candidates in tile_candidates]


def _candidate_prefix_stop_counts(
    case: TileStopGapCase,
    tile_candidates: list[list[int]],
    *,
    alpha_threshold: float,
    transmittance_threshold: float,
    max_alpha: float,
) -> list[int]:
    tiles_x = _ceil_div(case.width, case.tile_size)
    stops: list[int] = []
    for tile_id, candidates in enumerate(tile_candidates):
        tile_x = tile_id % tiles_x
        tile_y = tile_id // tiles_x
        x_start = tile_x * case.tile_size
        y_start = tile_y * case.tile_size
        x_end = min(case.width, x_start + case.tile_size)
        y_end = min(case.height, y_start + case.tile_size)
        tile_stop = 0
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                transmittance = 1.0
                local_stop = 0
                for prefix, splat_id in enumerate(candidates, start=1):
                    if transmittance <= transmittance_threshold:
                        break
                    local_stop = prefix
                    alpha = _eval_alpha(
                        float(x) + 0.5,
                        float(y) + 0.5,
                        case.splats[splat_id],
                        alpha_threshold=alpha_threshold,
                        max_alpha=max_alpha,
                    )
                    if alpha <= 0.0:
                        continue
                    transmittance *= 1.0 - alpha
                tile_stop = max(tile_stop, local_stop)
        stops.append(tile_stop)
    return stops


def _eval_alpha(
    x: float,
    y: float,
    splat: GaussianSplat,
    *,
    alpha_threshold: float,
    max_alpha: float,
) -> float:
    mean_x, mean_y, conic_x, conic_y, conic_z, opacity, _, _, _ = splat
    dx = x - mean_x
    dy = y - mean_y
    power = -0.5 * (conic_x * dx * dx + 2.0 * conic_y * dx * dy + conic_z * dy * dy)
    if power > 0.0:
        return 0.0
    alpha = min(max_alpha, opacity * math.exp(power))
    return alpha if alpha >= alpha_threshold else 0.0
