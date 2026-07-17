#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DYNAWORLD = ROOT.parents[3]
WORLD_FOAM_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORLD_FOAM_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_FOAM_DIR))

from gate0_beam_toy import default_sites, linspace, make_boundaries  # noqa: E402
from torch_world_foam_lane2_fused_slab import PowerBoundaryConfig, count_power_boundary_events  # noqa: E402


def run_smoke() -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    sites = default_sites()
    boundaries = make_boundaries(sites)
    u_values = linspace(-1.0, 1.0, 17)
    boundary_f32 = torch.tensor(
        [[boundary.nx, boundary.nz, boundary.nt, boundary.b] for boundary in boundaries],
        dtype=torch.float32,
        device="mps",
    )
    boundary_u32 = torch.tensor(
        [[boundary.left, boundary.right, 0, 0] for boundary in boundaries],
        dtype=torch.int32,
        device="mps",
    )
    beam_f32 = torch.tensor(
        [[u, 0.0, 1.0, 0.25, 3.0] for u in u_values],
        dtype=torch.float32,
        device="mps",
    )
    beam_u32 = torch.tensor(
        [[idx, 0, 0, 0] for idx in range(len(u_values))],
        dtype=torch.int32,
        device="mps",
    )

    rows = []
    for velocity, expected in ((0.35, 149), (0.70, 151)):
        counts = count_power_boundary_events(
            boundary_f32,
            beam_f32,
            PowerBoundaryConfig(camera_velocity_x=velocity, invalid_epsilon=1.0e-7),
            boundary_u32=boundary_u32,
            beam_u32=beam_u32,
        )
        torch.mps.synchronize()
        counts_cpu = counts.cpu()
        total = int(counts_cpu[:, 2].sum().item())
        invalid = int(counts_cpu[:, 3].sum().item())
        flagged_rows = int((counts_cpu[:, 4] != 0).sum().item())
        rows.append(
            {
                "camera_velocity_x": velocity,
                "expected_boundary_event_sum": expected,
                "metal_boundary_event_sum": total,
                "invalid_denominator_count": invalid,
                "flagged_row_count": flagged_rows,
                "matches_cpu_fixture": total == expected and invalid == 0 and flagged_rows == 0,
            }
        )

    return {
        "benchmark": "world_foam_lane2_mps_power_boundary_smoke",
        "status": "ok" if all(row["matches_cpu_fixture"] for row in rows) else "failed",
        "boundary_count": len(boundaries),
        "beam_count": len(u_values),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke the World Foam Lane 2 MPS power-boundary count op.")
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
