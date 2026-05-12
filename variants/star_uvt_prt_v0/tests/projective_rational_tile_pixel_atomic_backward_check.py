from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_tile_pixel_atomic_backward_check import run_check


def main() -> None:
    summary = run_check(abs_tol=5.0e-4, rel_tol=5.0e-2)
    print(summary)
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
