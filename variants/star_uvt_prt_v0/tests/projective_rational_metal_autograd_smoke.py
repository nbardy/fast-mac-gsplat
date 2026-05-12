from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.trainer_harness.projective_rational_metal_autograd_smoke import run_smoke  # noqa: E402


def main() -> None:
    summary = run_smoke(steps=4, lr=0.1, forward_mode="tiled", backward_mode="tile_pixel_atomic")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise AssertionError("projective rational Metal autograd smoke failed")


if __name__ == "__main__":
    main()
