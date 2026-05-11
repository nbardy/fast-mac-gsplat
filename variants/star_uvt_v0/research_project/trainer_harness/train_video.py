from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .train import fit_video_target, write_json
except ImportError:  # pragma: no cover - script execution fallback.
    from train import fit_video_target, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--tube-count", type=int, default=16)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    row = fit_video_target(
        args.video,
        tube_count=args.tube_count,
        target_size=args.target_size,
        max_frames=args.max_frames,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )
    if args.out_json is not None:
        write_json(row, args.out_json)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

