from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRACE_BYTES_PER_VISIT = {
    "id_alpha_tbefore": 12,
    "id_alpha_tbefore_flags": 16,
    "id_alpha_tbefore_center_depth": 24,
}


def _median_ms(profile: dict[str, Any], key: str) -> float:
    return float(profile["timing_summary_ms"][key]["median_ms"])


def _mib(byte_count: float) -> float:
    return float(byte_count) / (1024.0 * 1024.0)


def _tile_config(profile_data: dict[str, Any]) -> dict[str, int]:
    config = profile_data["projective_rational"]["tile_config"]
    return {
        "tile_x": int(config["tile_x"]),
        "tile_y": int(config["tile_y"]),
        "tile_t": int(config["tile_t"]),
        "tile_capacity": int(config["tile_capacity"]),
    }


def _row(
    *,
    label: str,
    width: int,
    height: int,
    frames: int,
    base_width: int,
    base_height: int,
    base_frames: int,
    entry_count: int,
    visits: int,
    tile_capacity: int,
) -> dict[str, Any]:
    pixel_scale = (width * height * frames) / float(base_width * base_height * base_frames)
    projected_entries = int(round(entry_count * pixel_scale))
    projected_visits = int(round(visits * pixel_scale))
    per_pixel_count_bytes = projected_entries * 4
    sparse_offset_count_bytes = projected_entries * 8

    layouts = {}
    for name, bytes_per_visit in TRACE_BYTES_PER_VISIT.items():
        dense_bytes = projected_entries * tile_capacity * bytes_per_visit + per_pixel_count_bytes
        sparse_upper_bytes = projected_visits * bytes_per_visit + sparse_offset_count_bytes
        layouts[name] = {
            "bytes_per_visit": bytes_per_visit,
            "dense_slot_bytes": int(dense_bytes),
            "dense_slot_mib": _mib(dense_bytes),
            "sparse_upper_bound_bytes": int(sparse_upper_bytes),
            "sparse_upper_bound_mib": _mib(sparse_upper_bytes),
        }

    return {
        "label": label,
        "width": width,
        "height": height,
        "frames": frames,
        "linear_pixel_scale": pixel_scale,
        "projected_trace_pixels": projected_entries,
        "projected_tile_pixel_tube_visits": projected_visits,
        "layouts": layouts,
    }


def _parse_projection(value: str) -> tuple[str, int, int, int]:
    label, shape = value.split("=", 1) if "=" in value else (value, value)
    width_raw, height_raw, frames_raw = shape.lower().replace("x", ",").split(",")
    return label, int(width_raw), int(height_raw), int(frames_raw)


def plan(profile_json: Path, projections: list[str]) -> dict[str, Any]:
    data = json.loads(profile_json.read_text())
    meta = data["meta"]
    prt = data["projective_rational"]
    profile = prt["backward_profile"]
    tile = _tile_config(data)

    width = int(meta["width"])
    height = int(meta["height"])
    frames = int(meta["frames"])
    entry_count = int(profile["total_tile_count"]) * int(profile["tile_pixel_count"])
    visits = int(profile["tile_pixel_tube_visits"])
    backward_ms = _median_ms(profile, "backward_kernel_ms")
    compute_only_ms = _median_ms(profile, "compute_only_kernel_ms")
    replay_only_ms = _median_ms(profile, "replay_only_kernel_ms")
    derivative_math_ms = max(0.0, compute_only_ms - replay_only_ms)
    atomic_write_ms = max(0.0, backward_ms - compute_only_ms)
    optimistic_cached_backward_floor_ms = derivative_math_ms + atomic_write_ms

    rows = [
        _row(
            label="observed",
            width=width,
            height=height,
            frames=frames,
            base_width=width,
            base_height=height,
            base_frames=frames,
            entry_count=entry_count,
            visits=visits,
            tile_capacity=tile["tile_capacity"],
        )
    ]
    for projection in projections:
        label, projected_width, projected_height, projected_frames = _parse_projection(projection)
        rows.append(
            _row(
                label=label,
                width=projected_width,
                height=projected_height,
                frames=projected_frames,
                base_width=width,
                base_height=height,
                base_frames=frames,
                entry_count=entry_count,
                visits=visits,
                tile_capacity=tile["tile_capacity"],
            )
        )

    return {
        "name": "projective_rational_trace_cache_planner",
        "profile_json": str(profile_json),
        "note": "Trace-cache memory estimates are linear projections from measured tile-pixel-tube visits; sparse rows are upper bounds because not every binned tube survives alpha/transmittance replay.",
        "pass": True,
        "policy": meta.get("prt_tile_policy"),
        "support_alpha_threshold": meta.get("prt_support_alpha_threshold"),
        "tile_config_key": prt["tile_config_key"],
        "tile_config": tile,
        "observed": {
            "width": width,
            "height": height,
            "frames": frames,
            "trace_pixels": entry_count,
            "tile_pixel_tube_visits": visits,
            "active_tile_count": profile["active_tile_count"],
            "total_tile_count": profile["total_tile_count"],
            "mean_active_tile_count": profile["mean_active_tile_count"],
            "tile_count_percentiles": profile["tile_count_percentiles"],
        },
        "timing_ms": {
            "backward_kernel": backward_ms,
            "compute_only_kernel": compute_only_ms,
            "replay_only_kernel": replay_only_ms,
            "derivative_math_estimate": derivative_math_ms,
            "atomic_write_estimate": atomic_write_ms,
            "optimistic_cached_backward_floor": optimistic_cached_backward_floor_ms,
            "optimistic_replay_savings": replay_only_ms,
        },
        "projection_rows": rows,
        "read": (
            "Dense per-pixel slot traces are too large even on the 64x64x4 row. "
            "A compact sparse trace is plausible at this scale, but its memory grows linearly "
            "with visit count and still carries write/read overhead. A fused train-step path is "
            "the cleaner speed target if we can keep forward and backward adjacent."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument(
        "--projection",
        action="append",
        default=[],
        help="Projected shape as label=WIDTHxHEIGHTxFRAMES, e.g. 128x128x4.",
    )
    args = parser.parse_args()

    summary = plan(args.profile_json, args.projection)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
