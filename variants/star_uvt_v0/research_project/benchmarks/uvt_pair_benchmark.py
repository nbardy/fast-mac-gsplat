from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    brute_force_render_uvt_tubes,
    make_gate0_scene,
    render_uvt_tubes,
    sliced_per_frame_pair_count,
)


DEFAULT_SCENES = (
    "single_static",
    "moving_diagonal",
    "two_non_crossing",
    "crossing_depth",
    "fast_screen_motion",
    "wide_temporal_support",
)


def benchmark_scene(scene: str) -> dict[str, object]:
    ma, q_uvt, depth0, depth_beta, opacity, color, config = make_gate0_scene(scene)
    cpu_started = time.perf_counter()
    reference = brute_force_render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config)
    cpu_ms = (time.perf_counter() - cpu_started) * 1000.0
    row: dict[str, object] = {
        "scene": scene,
        "frames": config.frames,
        "height": config.height,
        "width": config.width,
        "tube_count": int(ma.shape[0]),
        "cpu_brute_force_ms": cpu_ms,
        "summed_per_frame_tile_splat_pairs": sliced_per_frame_pair_count(ma, q_uvt, opacity, config),
    }
    if not torch.backends.mps.is_available():
        row["metal_skipped"] = "MPS is not available"
        return row
    result = render_uvt_tubes(
        ma.to("mps"),
        q_uvt.to("mps"),
        depth0.to("mps"),
        depth_beta.to("mps"),
        opacity.to("mps"),
        color.to("mps"),
        config,
        return_aux=True,
        reference=reference,
    )
    if result.stats is None:
        raise RuntimeError("expected Metal stats")
    row.update(result.stats.__dict__)
    return row


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    pair_ratios = [float(row["pair_ratio"]) for row in rows if "pair_ratio" in row]
    metal_times = [float(row["forward_wall_clock_ms"]) for row in rows if row.get("forward_wall_clock_ms") is not None]
    return {
        "scene_count": len(rows),
        "mean_pair_ratio": sum(pair_ratios) / max(len(pair_ratios), 1),
        "max_pair_ratio": max(pair_ratios) if pair_ratios else None,
        "mean_metal_forward_wall_clock_ms": sum(metal_times) / max(len(metal_times), 1),
        "max_metal_forward_wall_clock_ms": max(metal_times) if metal_times else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default=",".join(DEFAULT_SCENES))
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    rows = [benchmark_scene(scene) for scene in args.scenes.split(",") if scene]
    report = {"summary": summarize(rows), "rows": rows}
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
