from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


RESULT_GLOBS = (
    "depth_banded_homography_flow_atlas_valid_prt_tiled_compare_f1n_*.json",
    "depth_banded_homography_flow_atlas_valid_prt_tiled_compare_f1o_*.json",
)
UINT32_BYTES = 4
FLOAT32_BYTES = 4


def _mib(byte_count: int) -> float:
    return float(byte_count) / (1024.0 * 1024.0)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _int_at(data: dict[str, Any], *keys: str) -> int:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            raise KeyError(".".join(keys))
        value = value[key]
    return int(value)


def _float_at(data: dict[str, Any], *keys: str) -> float | None:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    return float(value)


def _tile_count(
    *,
    width: int,
    height: int,
    frames: int,
    tile_x: int,
    tile_y: int,
    tile_t: int,
    depth_bands: int = 1,
) -> int:
    return math.ceil(width / tile_x) * math.ceil(height / tile_y) * math.ceil(frames / tile_t) * depth_bands


def _timing_row(timing: dict[str, Any], key: str) -> dict[str, float | int | None]:
    item = timing.get(key, {})
    samples = item.get("samples_ms", [])
    return {
        "median_ms": None if "median_ms" not in item else float(item["median_ms"]),
        "min_ms": None if "min_ms" not in item else float(item["min_ms"]),
        "max_ms": None if "max_ms" not in item else float(item["max_ms"]),
        "sample_count": len(samples) if isinstance(samples, list) else None,
    }


def _metric_row(metrics: dict[str, Any], key: str) -> dict[str, float | None]:
    item = metrics.get(key, {})
    return {
        "psnr": None if "psnr" not in item else float(item["psnr"]),
        "mse": None if "mse" not in item else float(item["mse"]),
        "l1": None if "l1" not in item else float(item["l1"]),
        "max_abs": None if "max_abs" not in item else float(item["max_abs"]),
    }


def _budget_row(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    config = data["config"]
    atlas_tile = config["atlas_tile_config"]
    prt_tile = config["prt_tile_config"]
    width = _int_at(config, "target_size")
    height = width
    frames = _int_at(config, "frames")
    depth_bands = _int_at(config, "depth_bands")
    atlas_capacity = int(atlas_tile["tile_capacity"])
    max_pixel_candidates = _int_at(config, "atlas_max_pixel_candidates")
    required_pixel_candidates = depth_bands * atlas_capacity
    local_array_bytes = max_pixel_candidates * (UINT32_BYTES + FLOAT32_BYTES)
    local_array_with_count_bytes = local_array_bytes + UINT32_BYTES

    atlas_tile_count = _tile_count(
        width=width,
        height=height,
        frames=frames,
        tile_x=int(atlas_tile["tile_x"]),
        tile_y=int(atlas_tile["tile_y"]),
        tile_t=int(atlas_tile["tile_t"]),
        depth_bands=depth_bands,
    )
    prt_tile_count = _tile_count(
        width=width,
        height=height,
        frames=frames,
        tile_x=int(prt_tile["tile_x"]),
        tile_y=int(prt_tile["tile_y"]),
        tile_t=int(prt_tile["tile_t"]),
    )
    atlas_tile_bin_bytes = atlas_tile_count * (2 * UINT32_BYTES + atlas_capacity * UINT32_BYTES)
    prt_tile_bin_bytes = prt_tile_count * (3 * UINT32_BYTES + int(prt_tile["tile_capacity"]) * UINT32_BYTES)

    atlas = data["atlas_cached"]
    prt = data["valid_prt_tiled"]
    atlas_summary = atlas["tile_summary"]
    prt_summary = prt["tile_summary"]
    atlas_metrics = atlas["metrics"]
    prt_metrics = prt["metrics"]

    return {
        "source": str(path),
        "label": path.stem,
        "pass": bool(data.get("pass")),
        "config": {
            "seed": _int_at(config, "seed"),
            "target_size": width,
            "frames": frames,
            "tubes": _int_at(config, "tubes"),
            "depth_bands": depth_bands,
            "support_scale": float(config["support_scale"]),
            "atlas_tile_config": atlas_tile,
            "prt_tile_config": prt_tile,
        },
        "candidate_budget": {
            "max_pixel_candidates": max_pixel_candidates,
            "required_pixel_candidates": required_pixel_candidates,
            "headroom_candidates": max_pixel_candidates - required_pixel_candidates,
            "usage_fraction": required_pixel_candidates / float(max(max_pixel_candidates, 1)),
            "fits": required_pixel_candidates <= max_pixel_candidates,
        },
        "memory_estimate": {
            "per_pixel_local_array_bytes": local_array_bytes,
            "per_pixel_local_array_with_count_bytes": local_array_with_count_bytes,
            "atlas_tile_bin_bytes": atlas_tile_bin_bytes,
            "atlas_tile_bin_mib": _mib(atlas_tile_bin_bytes),
            "valid_prt_tile_bin_bytes": prt_tile_bin_bytes,
            "valid_prt_tile_bin_mib": _mib(prt_tile_bin_bytes),
            "notes": {
                "per_pixel_local_array": "candidate id uint32 plus depth float32 arrays sized by STAR_ATLAS_MAX_PIXEL_CANDIDATES; count register not included unless using *_with_count.",
                "tile_bin": "tile counts plus overflow arrays plus one uint32 id slot per tile-capacity entry; atlas tile count includes the depth-band dimension.",
            },
        },
        "tile_pairs": {
            "atlas_cached": int(atlas_summary["tile_pairs"]),
            "valid_prt_tiled": int(prt_summary["tile_pairs"]),
            "atlas_active_tile_count": int(atlas_summary["active_tile_count"]),
            "valid_prt_active_tile_count": int(prt_summary["active_tile_count"]),
            "atlas_max_tile_count": int(atlas_summary["max_tile_count"]),
            "valid_prt_max_tile_count": int(prt_summary["max_tile_count"]),
            "atlas_overflow_tile_count": int(atlas_summary["overflow_tile_count"]),
            "valid_prt_overflow_tile_count": int(prt_summary["overflow_tile_count"]),
        },
        "timing_ms": {
            "atlas_scan": _timing_row(atlas["timing_ms"], "atlas_scan"),
            "atlas_cached": _timing_row(atlas["timing_ms"], "atlas_cached"),
            "direct_dense_reference": _timing_row(atlas["timing_ms"], "direct_dense_reference"),
            "valid_prt_direct": _timing_row(prt["timing_ms"], "prt_direct"),
            "valid_prt_tiled": _timing_row(prt["timing_ms"], "prt_tiled"),
        },
        "psnr_rows": {
            "cached_atlas_vs_scan_atlas": _metric_row(atlas_metrics, "cached_atlas_vs_scan_atlas"),
            "cached_atlas_vs_direct_dense_reference": _metric_row(
                atlas_metrics,
                "cached_atlas_vs_direct_dense_reference",
            ),
            "prt_tiled_vs_prt_direct": _metric_row(prt_metrics, "prt_tiled_vs_prt_direct"),
            "prt_tiled_vs_direct_dense_reference": _metric_row(
                prt_metrics,
                "prt_tiled_vs_direct_dense_reference",
            ),
        },
        "speedups": data.get("speedups", {}),
    }


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_count": len(rows),
        "passing_count": sum(1 for row in rows if row["pass"]),
        "candidate_budget": {
            "max_pixel_candidates": sorted({row["candidate_budget"]["max_pixel_candidates"] for row in rows}),
            "required_pixel_candidates": sorted({row["candidate_budget"]["required_pixel_candidates"] for row in rows}),
            "all_fit": all(row["candidate_budget"]["fits"] for row in rows),
        },
        "memory_estimate": {
            "per_pixel_local_array_bytes": sorted(
                {row["memory_estimate"]["per_pixel_local_array_bytes"] for row in rows}
            ),
            "atlas_tile_bin_mib": _stats([row["memory_estimate"]["atlas_tile_bin_mib"] for row in rows]),
        },
        "tile_pairs": {
            "atlas_cached": _stats([float(row["tile_pairs"]["atlas_cached"]) for row in rows]),
            "valid_prt_tiled": _stats([float(row["tile_pairs"]["valid_prt_tiled"]) for row in rows]),
        },
        "timing_ms": {
            "atlas_cached_median": _stats(
                [
                    float(row["timing_ms"]["atlas_cached"]["median_ms"])
                    for row in rows
                    if row["timing_ms"]["atlas_cached"]["median_ms"] is not None
                ]
            ),
            "valid_prt_tiled_median": _stats(
                [
                    float(row["timing_ms"]["valid_prt_tiled"]["median_ms"])
                    for row in rows
                    if row["timing_ms"]["valid_prt_tiled"]["median_ms"] is not None
                ]
            ),
        },
        "psnr": {
            "cached_vs_direct_dense": _stats(
                [
                    float(row["psnr_rows"]["cached_atlas_vs_direct_dense_reference"]["psnr"])
                    for row in rows
                    if row["psnr_rows"]["cached_atlas_vs_direct_dense_reference"]["psnr"] is not None
                ]
            ),
            "prt_tiled_vs_direct_dense": _stats(
                [
                    float(row["psnr_rows"]["prt_tiled_vs_direct_dense_reference"]["psnr"])
                    for row in rows
                    if row["psnr_rows"]["prt_tiled_vs_direct_dense_reference"]["psnr"] is not None
                ]
            ),
        },
    }


def _default_inputs(results_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in RESULT_GLOBS:
        paths.extend(results_dir.glob(pattern))
    return sorted(set(paths))


def summarize(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("no input result files matched")
    rows = [_budget_row(path, _read_json(path)) for path in sorted(paths)]
    return {
        "name": "depth_banded_homography_flow_atlas_candidate_budget_summary",
        "note": (
            "Read-only accounting summary for F1n/F1o cached atlas result JSONs. "
            "Memory rows are estimates from recorded tile configs and candidate budgets; no renderer is run."
        ),
        "pass": all(row["candidate_budget"]["fits"] for row in rows),
        "summary": _summarize(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, help="F1n/F1o result JSON files")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory used for the default F1n/F1o result glob.",
    )
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    paths = args.inputs if args.inputs else _default_inputs(args.results_dir)
    report = summarize(paths)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
