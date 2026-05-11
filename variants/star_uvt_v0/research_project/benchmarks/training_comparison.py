from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import make_gate0_scene  # noqa: E402

try:
    from research_project.trainer_harness.model import dense_differentiable_render_uvt_tubes, render_model
    from research_project.trainer_harness.per_frame_baseline import PerFrameGaussianModel, render_per_frame_gaussians
    from research_project.trainer_harness.train import fit_model
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = Path(__file__).resolve().parents[1] / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from model import ScreenTimeTubeModel, dense_differentiable_render_uvt_tubes, render_model
    from per_frame_baseline import PerFrameGaussianModel, render_per_frame_gaussians
    from train import fit_model
else:
    from research_project.trainer_harness.model import ScreenTimeTubeModel


def _fit_per_frame(model: PerFrameGaussianModel, target: torch.Tensor, *, steps: int, lr: float) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image = render_per_frame_gaussians(model)
        loss = torch.mean((image - target).square())
        losses.append(float(loss.detach().cpu()))
        if step == steps:
            break
        loss.backward()
        optimizer.step()
    return losses


def run_training_comparison(
    *,
    scene: str,
    steps: int,
    lr: float,
    seed: int,
    jitter_pixels: float,
) -> dict[str, object]:
    ma, q_uvt, depth0, _depth_beta, opacity, color, config = make_gate0_scene(scene)
    depth_beta = torch.zeros((ma.shape[0], 3), dtype=torch.float32)
    target = dense_differentiable_render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config).detach()

    uvt_model = ScreenTimeTubeModel.from_uvt_tensors(
        ma,
        q_uvt,
        depth0,
        opacity,
        color,
        config,
        seed=seed,
        jitter_pixels=jitter_pixels,
    )
    per_frame_model = PerFrameGaussianModel.from_uvt_tensors(
        ma,
        q_uvt,
        depth0,
        opacity,
        color,
        config,
        seed=seed,
        jitter_pixels=jitter_pixels,
    )

    uvt_started = time.perf_counter()
    uvt_losses = fit_model(uvt_model, target, steps=steps, lr=lr)
    uvt_ms = (time.perf_counter() - uvt_started) * 1000.0
    per_frame_started = time.perf_counter()
    per_frame_losses = _fit_per_frame(per_frame_model, target, steps=steps, lr=lr)
    per_frame_ms = (time.perf_counter() - per_frame_started) * 1000.0

    with torch.no_grad():
        uvt_final_image = render_model(uvt_model)
        per_frame_final_image = render_per_frame_gaussians(per_frame_model)
        uvt_final_l1 = torch.mean((uvt_final_image - target).abs()).item()
        per_frame_final_l1 = torch.mean((per_frame_final_image - target).abs()).item()

    return {
        "scene": scene,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "jitter_pixels": jitter_pixels,
        "frames": config.frames,
        "height": config.height,
        "width": config.width,
        "tube_count": int(ma.shape[0]),
        "per_frame_splat_count": int(ma.shape[0] * config.frames),
        "uvt": {
            "initial_loss": uvt_losses[0],
            "final_loss": uvt_losses[-1],
            "loss_ratio": uvt_losses[-1] / max(uvt_losses[0], 1.0e-12),
            "final_l1": uvt_final_l1,
            "wall_clock_ms": uvt_ms,
        },
        "per_frame": {
            "initial_loss": per_frame_losses[0],
            "final_loss": per_frame_losses[-1],
            "loss_ratio": per_frame_losses[-1] / max(per_frame_losses[0], 1.0e-12),
            "final_l1": per_frame_final_l1,
            "wall_clock_ms": per_frame_ms,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="moving_diagonal")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--jitter-pixels", type=float, default=0.70)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    row = run_training_comparison(
        scene=args.scene,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        jitter_pixels=args.jitter_pixels,
    )
    if float(row["uvt"]["final_loss"]) >= float(row["uvt"]["initial_loss"]):
        raise AssertionError(f"UVT loss did not decrease: {row['uvt']}")
    if float(row["per_frame"]["final_loss"]) >= float(row["per_frame"]["initial_loss"]):
        raise AssertionError(f"per-frame loss did not decrease: {row['per_frame']}")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
