from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .world_tube import (
        PinholeCameraMotion,
        make_pinhole_world_tube_demo,
        project_world_tubes_pinhole,
        project_world_tubes_pinhole_motion,
        project_world_tubes_pinhole_projective_motion,
    )
except ImportError:  # pragma: no cover - script execution fallback.
    from world_tube import (
        PinholeCameraMotion,
        make_pinhole_world_tube_demo,
        project_world_tubes_pinhole,
        project_world_tubes_pinhole_motion,
        project_world_tubes_pinhole_projective_motion,
    )


def max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().detach().cpu())


def main() -> None:
    batch, camera, config = make_pinhole_world_tube_demo()
    fixed = project_world_tubes_pinhole(batch, camera, config)
    zero_motion = PinholeCameraMotion(
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        fx_dot=0.0,
        fy_dot=0.0,
        cx_dot=0.0,
        cy_dot=0.0,
        world_to_camera=camera.world_to_camera,
        world_to_camera_dot=torch.zeros_like(camera.world_to_camera),
        chart_time=0.0,
    )
    moving_zero = project_world_tubes_pinhole_motion(batch, zero_motion, config)

    zero_deltas = [max_abs_delta(left, right) for left, right in zip(fixed, moving_zero, strict=True)]
    if max(zero_deltas) > 1.0e-5:
        raise AssertionError(f"zero camera motion mismatch: {zero_deltas}")

    translated_dot = torch.zeros_like(camera.world_to_camera)
    translated_dot[0, 3] = 0.15
    translated_dot[2, 3] = 0.05
    translated_motion = PinholeCameraMotion(
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        fx_dot=0.0,
        fy_dot=0.0,
        cx_dot=0.0,
        cy_dot=0.0,
        world_to_camera=camera.world_to_camera,
        world_to_camera_dot=translated_dot,
        chart_time=0.0,
    )
    moving_translated = project_world_tubes_pinhole_motion(batch, translated_motion, config)
    projective_translated = project_world_tubes_pinhole_projective_motion(batch, translated_motion, config)
    projective_deltas = [
        max_abs_delta(left, right) for left, right in zip(moving_translated, projective_translated, strict=True)
    ]
    if max(projective_deltas) > 1.0e-4:
        raise AssertionError(f"projective camera gauge mismatch: {projective_deltas}")

    q_delta = max_abs_delta(fixed[1], moving_translated[1])
    depth_beta_delta = max_abs_delta(fixed[3], moving_translated[3])
    if not torch.isfinite(moving_translated[1]).all():
        raise AssertionError("translated camera q_uvt contains non-finite values")
    if not torch.isfinite(moving_translated[3]).all():
        raise AssertionError("translated camera depth_beta contains non-finite values")
    if q_delta <= 0.0:
        raise AssertionError("translated camera motion did not change q_uvt")
    if depth_beta_delta <= 0.0:
        raise AssertionError("translated camera motion did not change depth_beta")

    print(
        json.dumps(
            {
                "zero_motion_max_abs_delta": max(zero_deltas),
                "zero_motion_component_deltas": zero_deltas,
                "translated_motion_q_delta": q_delta,
                "translated_motion_depth_beta_delta": depth_beta_delta,
                "projective_motion_component_deltas": projective_deltas,
                "projective_motion_max_abs_delta": max(projective_deltas),
                "projected_tube_count": int(fixed[0].shape[0]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
