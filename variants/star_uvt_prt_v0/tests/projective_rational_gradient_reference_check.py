from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_gradient_reference_check import run_check  # noqa: E402


def main() -> None:
    summary = run_check(eps=1.0e-3, abs_tol=1.0e-4, rel_tol=5.0e-2)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise AssertionError("projective rational gradient reference check failed")


if __name__ == "__main__":
    main()
