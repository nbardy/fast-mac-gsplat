from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig, make_gate0_scene, render_uvt_tubes

try:
    from .data import load_video_target
    from .model import ScreenTimeTubeModel, dense_differentiable_render_uvt_tubes, render_model
except ImportError:  # pragma: no cover - script execution fallback.
    from data import load_video_target
    from model import ScreenTimeTubeModel, dense_differentiable_render_uvt_tubes, render_model


@dataclass(frozen=True)
class FitConfig:
    scene: str
    steps: int = 30
    lr: float = 0.08
    device: str = "cpu"
    seed: int = 0
    jitter_pixels: float = 0.75


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(value)


def make_scene_fit_target(
    scene: str,
    *,
    device: torch.device,
    seed: int,
    jitter_pixels: float,
) -> tuple[Tensor, ScreenTimeTubeModel, UVTRenderConfig]:
    ma, q_uvt, depth0, _depth_beta, opacity, color, config = make_gate0_scene(scene, device=device)
    depth_beta = torch.zeros((ma.shape[0], 3), dtype=torch.float32, device=device)
    target = dense_differentiable_render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config).detach()
    model = ScreenTimeTubeModel.from_uvt_tensors(
        ma,
        q_uvt,
        depth0,
        opacity,
        color,
        config,
        seed=seed,
        jitter_pixels=jitter_pixels,
    )
    return target, model, config


def fit_model(model: ScreenTimeTubeModel, target: Tensor, *, steps: int, lr: float) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for _step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image = render_model(model)
        loss = torch.mean((image - target).square())
        losses.append(float(loss.detach().cpu()))
        if _step == steps:
            break
        loss.backward()
        optimizer.step()
    return losses


def _metal_stats(model: ScreenTimeTubeModel) -> dict[str, float | int]:
    if not torch.backends.mps.is_available():
        return {}
    ma, q_uvt, depth0, depth_beta, opacity, color = model.tensors()
    result = render_uvt_tubes(
        ma.detach().to("mps"),
        q_uvt.detach().to("mps"),
        depth0.detach().to("mps"),
        depth_beta.detach().to("mps"),
        opacity.detach().to("mps"),
        color.detach().to("mps"),
        model.config,
        return_aux=True,
        reference="cpu",
    )
    if not hasattr(result, "stats") or result.stats is None:
        raise RuntimeError("Metal UVT render did not return stats")
    return {
        "metal_max_rgb_error": result.stats.max_rgb_error,
        "metal_mean_rgb_error": result.stats.mean_rgb_error,
        "metal_forward_wall_clock_ms": result.stats.forward_wall_clock_ms,
        "metal_pair_ratio": result.stats.pair_ratio,
        "metal_overflow_tile_count": result.stats.overflow_tile_count,
    }


def run_synthetic_fit(
    *,
    scene: str = "moving_diagonal",
    steps: int = 30,
    lr: float = 0.08,
    device: str = "cpu",
    seed: int = 0,
    jitter_pixels: float = 0.75,
    metal_check: bool = False,
) -> dict[str, object]:
    dev = resolve_device(device)
    target, model, _config = make_scene_fit_target(scene, device=dev, seed=seed, jitter_pixels=jitter_pixels)
    losses = fit_model(model, target, steps=steps, lr=lr)
    row: dict[str, object] = {
        "scene": scene,
        "device": str(dev),
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / max(losses[0], 1.0e-12),
    }
    if metal_check:
        row.update(_metal_stats(model))
    return row


def fit_video_target(
    video_path: Path,
    *,
    tube_count: int,
    target_size: int,
    max_frames: int,
    steps: int,
    lr: float,
    device: str,
    seed: int,
) -> dict[str, object]:
    dev = resolve_device(device)
    target = load_video_target(video_path, target_size=target_size, max_frames=max_frames, device=dev)
    config = UVTRenderConfig(height=int(target.shape[1]), width=int(target.shape[2]), frames=int(target.shape[0]))
    model = ScreenTimeTubeModel(tube_count, config, seed=seed, device=dev)
    losses = fit_model(model, target, steps=steps, lr=lr)
    return {
        "video_path": str(video_path),
        "device": str(dev),
        "tube_count": tube_count,
        "target_size": target_size,
        "frames": int(target.shape[0]),
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_ratio": losses[-1] / max(losses[0], 1.0e-12),
    }


def write_json(row: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
