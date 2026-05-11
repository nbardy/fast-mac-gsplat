from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v13b_rgb_grad_handoff import estimate_rgb_grad_handoff_memory


def mib(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory accounting for the v13b RGB-gradient handoff scaffold."
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    args = parser.parse_args()

    estimate = estimate_rgb_grad_handoff_memory(
        batch=args.batch,
        height=args.height,
        width=args.width,
        feature_dim=args.feature_dim,
        dtype_bytes=args.dtype_bytes,
    )
    payload = estimate.as_dict()
    payload.update(
        {
            "current_dense_backward_input_mib": mib(estimate.current_dense_backward_input_bytes),
            "handoff_dense_backward_input_mib": mib(estimate.handoff_dense_backward_input_bytes),
            "avoided_mib": mib(estimate.avoided_bytes),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
