from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "research_project/benchmarks/projective_rational_metal_forward_timing_probe.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt_prt.tile_config import (  # noqa: E402
    DEFAULT_PROJECTIVE_RATIONAL_TILE_CANDIDATES,
    ProjectiveRationalTileConfig,
    parse_projective_rational_tile_config,
    select_projective_rational_tile_summary,
)


def _parse_int_list(value: str) -> list[int]:
    out = [int(item) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("values must be positive")
    return out


def _parse_candidate(value: str) -> ProjectiveRationalTileConfig:
    try:
        return parse_projective_rational_tile_config(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _candidate_summary(
    candidate: ProjectiveRationalTileConfig,
    probe: dict[str, Any],
    returncode: int,
) -> dict[str, Any]:
    rows = list(probe.get("rows", []))
    overflow_tiles = sum(int(row.get("overflow_tile_count", 0)) for row in rows)
    max_tile_count = max((int(row.get("max_tile_count", 0)) for row in rows), default=0)
    max_error = max((float(row.get("max_abs_error_vs_direct", 0.0)) for row in rows), default=0.0)
    tiled_median_ms_sum = sum(float(row.get("tiled", {}).get("median_ms", float("inf"))) for row in rows)
    total_tile_pairs = sum(int(row.get("total_tile_pairs", 0)) for row in rows)
    active_tile_count = sum(int(row.get("active_tile_count", 0)) for row in rows)
    pass_gate = bool(probe.get("pass")) and returncode == 0
    capacity_overage = max(0, max_tile_count - int(candidate.tile_capacity))
    return {
        "candidate": candidate.as_dict(),
        "candidate_key": candidate.key,
        "pass": pass_gate,
        "returncode": returncode,
        "max_tile_count": max_tile_count,
        "overflow_tile_count": overflow_tiles,
        "max_abs_error_vs_direct": max_error,
        "tiled_median_ms_sum": tiled_median_ms_sum,
        "total_tile_pairs": total_tile_pairs,
        "active_tile_count": active_tile_count,
        "selection_score": [
            0 if pass_gate else 1,
            tiled_median_ms_sum,
            total_tile_pairs,
            active_tile_count,
            max_tile_count,
        ],
        "failure_score": [
            overflow_tiles,
            capacity_overage,
            max_error,
            tiled_median_ms_sum,
        ],
        "rows": rows,
    }


def _run_candidate(
    *,
    candidate: ProjectiveRationalTileConfig,
    tube_counts: list[int],
    frames: int,
    width: int,
    height: int,
    camera_motion_scale: float,
    warmups: int,
    repeats: int,
    seed: int,
    tmpdir: Path,
) -> dict[str, Any]:
    out_json = tmpdir / (
        f"prt_tile_sweep_{candidate.tile_x}x{candidate.tile_y}x"
        f"{candidate.tile_t}_cap{candidate.tile_capacity}.json"
    )
    env = os.environ.copy()
    env.update(candidate.as_env())
    cmd = [
        sys.executable,
        str(PROBE),
        "--tube-counts",
        ",".join(str(item) for item in tube_counts),
        "--frames",
        str(frames),
        "--width",
        str(width),
        "--height",
        str(height),
        "--camera-motion-scale",
        str(camera_motion_scale),
        "--tile-x",
        str(candidate.tile_x),
        "--tile-y",
        str(candidate.tile_y),
        "--tile-t",
        str(candidate.tile_t),
        "--tile-capacity",
        str(candidate.tile_capacity),
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
        "--seed",
        str(seed),
        "--out-json",
        str(out_json),
    ]
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    probe = json.loads(out_json.read_text()) if out_json.exists() else {"pass": False, "rows": []}
    summary = _candidate_summary(candidate, probe, result.returncode)
    if result.stderr:
        summary["stderr"] = result.stderr
    return summary


def run_sweep(
    *,
    candidates: list[ProjectiveRationalTileConfig],
    tube_counts: list[int],
    frames: int,
    width: int,
    height: int,
    camera_motion_scale: float,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("expected at least one tile candidate")
    with TemporaryDirectory() as raw_tmpdir:
        tmpdir = Path(raw_tmpdir)
        results = [
            _run_candidate(
                candidate=candidate,
                tube_counts=tube_counts,
                frames=frames,
                width=width,
                height=height,
                camera_motion_scale=camera_motion_scale,
                warmups=warmups,
                repeats=repeats,
                seed=seed,
                tmpdir=tmpdir,
            )
            for candidate in candidates
        ]
    selected, best_failed = select_projective_rational_tile_summary(results)
    return {
        "name": "projective_rational_tile_config_sweep",
        "note": "Each candidate runs in a fresh process because Metal shader tile constants are process-static.",
        "tube_counts": tube_counts,
        "frames": frames,
        "width": width,
        "height": height,
        "camera_motion_scale": camera_motion_scale,
        "warmups": warmups,
        "repeats": repeats,
        "seed": seed,
        "pass": selected is not None,
        "selected_candidate": selected,
        "best_failed_candidate": best_failed,
        "candidates": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=_parse_candidate,
        nargs="+",
        default=list(DEFAULT_PROJECTIVE_RATIONAL_TILE_CANDIDATES),
    )
    parser.add_argument("--tube-counts", type=_parse_int_list, default=[512])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--camera-motion-scale", type=float, default=1.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    summary = run_sweep(
        candidates=args.candidates,
        tube_counts=args.tube_counts,
        frames=args.frames,
        width=args.width,
        height=args.height,
        camera_motion_scale=args.camera_motion_scale,
        warmups=args.warmups,
        repeats=args.repeats,
        seed=args.seed,
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
