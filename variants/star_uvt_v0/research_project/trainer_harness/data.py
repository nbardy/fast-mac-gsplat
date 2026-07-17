from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor


def _find_dynaworld_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "train" / "sequence_data.py").exists():
            return parent
    return None


def load_video_target(
    video_path: Path,
    *,
    target_size: int,
    max_frames: int,
    device: torch.device | str = "cpu",
    start_seconds: float | None = None,
    fps: float | None = None,
    duration_seconds: float | None = None,
    image_crop_mode: str = "resize",
) -> Tensor:
    """Load a video target as [F,H,W,3] using Dynaworld's video loader."""

    dynaworld_root = _find_dynaworld_root()
    if dynaworld_root is None:
        raise FileNotFoundError("Could not find dynaworld/src/train/sequence_data.py from this variant")
    train_path = dynaworld_root / "src" / "train"
    if str(train_path) not in sys.path:
        sys.path.insert(0, str(train_path))
    from sequence_data import load_video_sequence, load_video_window_sequence

    if start_seconds is not None or fps is not None or duration_seconds is not None:
        if max_frames <= 0:
            raise ValueError("windowed video loading requires max_frames/frame_count to be positive")
        sequence = load_video_window_sequence(
            video_path,
            target_size=target_size,
            start_seconds=0.0 if start_seconds is None else float(start_seconds),
            duration_seconds=duration_seconds,
            fps=0.0 if fps is None else float(fps),
            frame_count=int(max_frames),
            image_crop_mode=image_crop_mode,
        ).to(device)
    else:
        sequence = load_video_sequence(
            video_path,
            target_size=target_size,
            max_frames=max_frames,
            image_crop_mode=image_crop_mode,
        ).to(device)
    return sequence.frames.permute(0, 2, 3, 1).contiguous()
