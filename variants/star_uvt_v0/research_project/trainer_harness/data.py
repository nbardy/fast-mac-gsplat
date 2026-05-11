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
) -> Tensor:
    """Load a video target as [F,H,W,3] using Dynaworld's video loader."""

    dynaworld_root = _find_dynaworld_root()
    if dynaworld_root is None:
        raise FileNotFoundError("Could not find dynaworld/src/train/sequence_data.py from this variant")
    train_path = dynaworld_root / "src" / "train"
    if str(train_path) not in sys.path:
        sys.path.insert(0, str(train_path))
    from sequence_data import load_video_sequence

    sequence = load_video_sequence(video_path, target_size=target_size, max_frames=max_frames).to(device)
    return sequence.frames.permute(0, 2, 3, 1).contiguous()

