from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v9_hw_eval_parity.parity_v8 import run_parity_case  # noqa: E402


def main() -> None:
    if not torch.backends.mps.is_available():
        print("MPS unavailable; v8/v9 parity smoke skipped.")
        return

    row = run_parity_case(
        "tiny_single",
        height=16,
        width=16,
        gaussians=1,
        seed=0,
        warmup=1,
        iters=2,
        v9_direct=True,
    )
    print(json.dumps(row, indent=2, sort_keys=True))
    assert row["status"] == "ok", row
    assert row["comparable_to_v8"] is True, row
    assert row["rgb_max_abs_err"] <= 2.0e-5, row
    assert row["rgb_mean_abs_err"] <= 2.0e-6, row
    assert row["v8_median_ms"] > 0.0, row
    assert row["v9_median_ms"] > 0.0, row


if __name__ == "__main__":
    main()
