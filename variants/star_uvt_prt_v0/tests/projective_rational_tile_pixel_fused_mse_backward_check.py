from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_tile_pixel_fused_mse_backward_check import run_check  # noqa: E402


if __name__ == "__main__":
    print(run_check(abs_tol=5.0e-4, rel_tol=5.0e-2, loss_tol=1.0e-5))
