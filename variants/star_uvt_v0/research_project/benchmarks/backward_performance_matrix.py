from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARKS = Path(__file__).resolve().parent
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig  # noqa: E402
from backward_performance_smoke import make_inputs, time_backward  # noqa: E402


CASES: dict[str, dict[str, int]] = {
    "smoke": {"tube_count": 16, "size": 32, "frames": 4, "iterations": 2, "warmup_iterations": 1, "seed": 11},
    "large_local": {"tube_count": 64, "size": 64, "frames": 8, "iterations": 1, "warmup_iterations": 1, "seed": 23},
}


def run_case(name: str, spec: dict[str, int]) -> dict[str, object]:
    config = UVTRenderConfig(height=spec["size"], width=spec["size"], frames=spec["frames"])
    inputs = make_inputs(spec["tube_count"], config, spec["seed"])
    dense = time_backward("dense", inputs, config, spec["iterations"], warmup_iterations=spec["warmup_iterations"])
    metal_tile = time_backward("metal_tile", inputs, config, spec["iterations"], warmup_iterations=spec["warmup_iterations"])
    if float(metal_tile["ma_grad_norm_last"]) <= 0.0:
        raise AssertionError(f"expected non-zero Metal tile gradient for {name}, got {metal_tile}")
    dense_ms = float(dense["mean_ms"])
    metal_ms = float(metal_tile["mean_ms"])
    return {
        "case": name,
        "tube_count": spec["tube_count"],
        "height": spec["size"],
        "width": spec["size"],
        "frames": spec["frames"],
        "iterations": spec["iterations"],
        "warmup_iterations": spec["warmup_iterations"],
        "dense": dense,
        "metal_tile": metal_tile,
        "dense_to_metal_mean_ratio": dense_ms / max(metal_ms, 1.0e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="smoke,large_local")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print(json.dumps({"metal_skipped": "MPS is not available"}, indent=2, sort_keys=True))
        return

    names = [name.strip() for name in args.cases.split(",") if name.strip()]
    rows = []
    for name in names:
        if name not in CASES:
            raise ValueError(f"unknown case {name!r}; expected one of {sorted(CASES)}")
        rows.append(run_case(name, CASES[name]))
    report = {"cases": rows}
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
