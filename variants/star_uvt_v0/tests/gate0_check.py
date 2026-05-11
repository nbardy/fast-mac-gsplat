from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    brute_force_render_uvt_tubes,
    make_gate0_scene,
    render_uvt_tubes,
    sliced_per_frame_pair_count,
)


def run_scene(scene: str, *, cpu_only: bool) -> dict[str, object]:
    ma, q, depth0, depth_beta, opacity, color, config = make_gate0_scene(scene)
    ref = brute_force_render_uvt_tubes(ma, q, depth0, depth_beta, opacity, color, config)
    row: dict[str, object] = {
        "scene": scene,
        "cpu_reference_shape": list(ref.shape),
        "summed_per_frame_tile_splat_pairs": sliced_per_frame_pair_count(ma, q, opacity, config),
    }
    if cpu_only:
        return row
    if not torch.backends.mps.is_available():
        row["metal_skipped"] = "MPS is not available"
        return row
    result = render_uvt_tubes(
        ma.to("mps"),
        q.to("mps"),
        depth0.to("mps"),
        depth_beta.to("mps"),
        opacity.to("mps"),
        color.to("mps"),
        config,
        return_aux=True,
        reference=ref,
    )
    assert result.stats is not None
    row.update(result.stats.__dict__)
    if result.stats.max_rgb_error is None or result.stats.max_rgb_error > 2.0e-5:
        raise AssertionError(f"{scene}: max RGB error too high: {result.stats.max_rgb_error}")
    if result.stats.overflow_tile_count != 0:
        raise AssertionError(f"{scene}: expected zero overflow, got {result.stats.overflow_tile_count}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--scenes",
        default="single_static,moving_diagonal,two_non_crossing,crossing_depth,fast_screen_motion,wide_temporal_support",
    )
    args = parser.parse_args()

    rows = [run_scene(scene, cpu_only=args.cpu_only) for scene in args.scenes.split(",") if scene]
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
