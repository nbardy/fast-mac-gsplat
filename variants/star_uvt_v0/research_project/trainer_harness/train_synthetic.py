from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .train import run_synthetic_fit, write_json
except ImportError:  # pragma: no cover - script execution fallback.
    from train import run_synthetic_fit, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="moving_diagonal")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter-pixels", type=float, default=0.75)
    parser.add_argument("--metal-check", action="store_true")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    row = run_synthetic_fit(
        scene=args.scene,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        jitter_pixels=args.jitter_pixels,
        metal_check=args.metal_check,
    )
    if args.out_json is not None:
        write_json(row, args.out_json)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

