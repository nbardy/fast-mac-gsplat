from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig  # noqa: E402

try:
    from .variable_camera_segments import project_piecewise_camera_time_segments
    from .world_tube import WorldTubeBatch
except ImportError:  # pragma: no cover - direct script execution fallback.
    from variable_camera_segments import project_piecewise_camera_time_segments
    from world_tube import WorldTubeBatch


def _make_batch() -> WorldTubeBatch:
    return WorldTubeBatch(
        x0=torch.tensor([[-0.08, -0.04, 2.0], [0.10, 0.06, 2.4]], dtype=torch.float32),
        velocity=torch.tensor([[0.01, 0.02, 0.03], [-0.02, 0.01, -0.02]], dtype=torch.float32),
        t0=torch.tensor([0.0, 0.0], dtype=torch.float32),
        precision_xy=torch.tensor([[80.0, 90.0], [70.0, 85.0]], dtype=torch.float32),
        lambda_t=torch.tensor([0.35, 0.40], dtype=torch.float32),
        opacity=torch.tensor([0.62, 0.50], dtype=torch.float32),
        color=torch.tensor([[0.8, 0.3, 0.2], [0.2, 0.7, 0.9]], dtype=torch.float32),
    )


def _make_cameras(frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    K_seq = torch.eye(3, dtype=torch.float32).repeat(frames, 1, 1)
    K_seq[:, 0, 0] = 60.0
    K_seq[:, 1, 1] = 58.0
    K_seq[:, 0, 2] = 8.0
    K_seq[:, 1, 2] = 8.0
    w2c_seq = torch.eye(4, dtype=torch.float32).repeat(frames, 1, 1)
    w2c_seq[:, 0, 3] = torch.linspace(0.0, -0.06, frames)
    w2c_seq[:, 1, 3] = torch.linspace(0.0, 0.03, frames)
    return K_seq, w2c_seq


def _assert_projected(projected: object, expected_tubes: int, expected_parent: list[int]) -> None:
    tensors = {
        "ma": projected.ma,
        "q_uvt": projected.q_uvt,
        "depth0": projected.depth0,
        "depth_beta": projected.depth_beta,
        "opacity": projected.opacity,
        "color": projected.color,
        "t_minmax": projected.t_minmax,
    }
    expected_shapes = {
        "ma": (expected_tubes, 3),
        "q_uvt": (expected_tubes, 6),
        "depth0": (expected_tubes,),
        "depth_beta": (expected_tubes, 3),
        "opacity": (expected_tubes,),
        "color": (expected_tubes, 3),
        "t_minmax": (expected_tubes, 2),
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise AssertionError(
                f"{name} shape mismatch: {tuple(tensor.shape)} != {expected_shapes[name]}"
            )
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} has non-finite values")
    if tuple(projected.parent_id.shape) != (expected_tubes,):
        raise AssertionError(f"parent_id shape mismatch: {tuple(projected.parent_id.shape)}")
    if projected.parent_id.tolist() != expected_parent:
        raise AssertionError(f"parent_id mismatch: {projected.parent_id.tolist()} != {expected_parent}")


def main() -> None:
    full_frames = 4
    config = UVTRenderConfig(height=16, width=16, frames=full_frames)
    batch = _make_batch()
    K_seq, w2c_seq = _make_cameras(full_frames)

    one_segment = project_piecewise_camera_time_segments(
        batch,
        K_seq,
        w2c_seq,
        config,
        full_frames=full_frames,
        frame_start=0,
        frames_per_segment=full_frames,
    )
    per_frame = project_piecewise_camera_time_segments(
        batch,
        K_seq,
        w2c_seq,
        config,
        full_frames=full_frames,
        frame_start=0,
        frames_per_segment=1,
    )

    _assert_projected(one_segment, expected_tubes=2, expected_parent=[0, 1])
    _assert_projected(per_frame, expected_tubes=8, expected_parent=[0, 1, 0, 1, 0, 1, 0, 1])
    if one_segment.diagnostics.segment_count != 2:
        raise AssertionError("full-window projection should use one segment")
    if per_frame.diagnostics.segment_count != 8:
        raise AssertionError("per-frame projection should use one segment per frame")

    print(
        json.dumps(
            {
                "full_window_tubes": int(one_segment.ma.shape[0]),
                "per_frame_tubes": int(per_frame.ma.shape[0]),
                "per_frame_parent_id": per_frame.parent_id.tolist(),
                "per_frame_t_minmax": per_frame.t_minmax.tolist(),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
