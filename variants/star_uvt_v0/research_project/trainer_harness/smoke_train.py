from __future__ import annotations

import json

try:
    from .train import run_synthetic_fit
except ImportError:  # pragma: no cover - script execution fallback.
    from train import run_synthetic_fit


def main() -> None:
    row = run_synthetic_fit(scene="moving_diagonal", steps=25, lr=0.08, device="cpu", seed=3, jitter_pixels=0.70)
    if float(row["final_loss"]) >= float(row["initial_loss"]) * 0.75:
        raise AssertionError(f"expected loss drop, got {row}")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
