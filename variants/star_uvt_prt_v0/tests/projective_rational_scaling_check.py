from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_projection_scaling_probe import run_probe  # noqa: E402


def main() -> None:
    summary = run_probe(frame_counts=[8, 16], tube_count=512, repeats=1, warmups=0, seed=17)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise AssertionError("projective rational scaling check failed")


if __name__ == "__main__":
    main()
