from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_v8_hw_train import RasterConfig, estimate_hardware_train_state, probe_hardware_train


def assert_raises(message_part: str, fn) -> None:
    try:
        fn()
    except ValueError as exc:
        if message_part not in str(exc):
            raise AssertionError(f"expected {message_part!r} in {str(exc)!r}") from exc
        return
    raise AssertionError(f"expected ValueError containing {message_part!r}")


def main() -> None:
    cfg = RasterConfig(height=2160, width=3840, tile_size=16)
    plan = estimate_hardware_train_state(cfg, batch_size=2)
    expected_tiles = 2 * ((2160 + 15) // 16) * ((3840 + 15) // 16)
    expected_pixels = 2 * 2160 * 3840
    assert plan.tile_stop_bytes == expected_tiles * 4
    assert plan.final_T_bytes == expected_pixels * 4
    assert plan.pixel_stop_bytes == expected_pixels * 4
    assert plan.requested_state_mode == "compute"
    assert plan.selected_state_mode == "compute"
    assert plan.requested_state_bytes == 0
    assert not plan.allocates_per_pixel_state_by_default

    tile_cfg = RasterConfig(
        height=64,
        width=80,
        tile_size=16,
        use_hardware_train=True,
        backward_state_mode="tile_stop",
        capture_stop_count=True,
    )
    tile_plan = estimate_hardware_train_state(tile_cfg, batch_size=3, selected_state_mode="compute")
    assert tile_plan.requested_state_bytes == tile_plan.tile_stop_bytes
    assert tile_plan.selected_state_bytes == 0

    pixel_cfg = RasterConfig(
        height=64,
        width=80,
        tile_size=16,
        use_hardware_train=True,
        backward_state_mode="pixel_stop",
        capture_stop_count=True,
        capture_pixel_stop=True,
    )
    pixel_plan = estimate_hardware_train_state(pixel_cfg, batch_size=1, selected_state_mode="compute")
    assert pixel_plan.requested_capture_modes == ("tile_stop", "pixel_stop")
    assert pixel_plan.requested_state_bytes == pixel_plan.tile_stop_bytes + pixel_plan.pixel_stop_bytes

    probe = probe_hardware_train(pixel_cfg, device=torch.device("mps"))
    assert probe.requested
    assert not probe.selected
    assert probe.requested_state_mode == "pixel_stop"
    assert probe.selected_state_mode == "compute"
    assert probe.fallback_reason

    assert_raises(
        "requires capture_final_T=True",
        lambda: estimate_hardware_train_state(
            RasterConfig(height=64, width=64, backward_state_mode="final_T"),
        ),
    )
    assert_raises(
        "mutually exclusive",
        lambda: estimate_hardware_train_state(
            RasterConfig(
                height=64,
                width=64,
                backward_state_mode="pixel_stop",
                capture_stop_count=True,
                capture_final_T=True,
                capture_pixel_stop=True,
            ),
        ),
    )
    assert_raises(
        "cannot request capture_final_T",
        lambda: estimate_hardware_train_state(
            RasterConfig(height=64, width=64, backward_state_mode="compute", capture_final_T=True),
        ),
    )
    print("state_plan_check: ok")


if __name__ == "__main__":
    main()
