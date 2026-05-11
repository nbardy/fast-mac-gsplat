from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig, render_uvt_tubes  # noqa: E402

try:
    from research_project.benchmarks.video_fit_comparison import fit_uvt, mse_to_psnr, render_uvt_model
    from research_project.trainer_harness.data import load_video_target
    from research_project.trainer_harness.model import ScreenTimeTubeModel
except ImportError:  # pragma: no cover - direct script execution fallback.
    BENCHMARKS = Path(__file__).resolve().parent
    HARNESS = Path(__file__).resolve().parents[1] / "trainer_harness"
    for path in (BENCHMARKS, HARNESS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from data import load_video_target
    from model import ScreenTimeTubeModel
    from video_fit_comparison import fit_uvt, mse_to_psnr, render_uvt_model


def metal_stats(model: ScreenTimeTubeModel) -> tuple[torch.Tensor, dict[str, object]]:
    ma, q_uvt, depth0, depth_beta, opacity, color = model.tensors()
    result = render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, model.config, return_aux=True)
    if result.stats is None:
        raise AssertionError("Metal render did not return stats")
    return result.image, result.stats.__dict__


def apply_uvt_tile_env(config: UVTRenderConfig) -> None:
    os.environ["STAR_UVT_TILE_X"] = str(config.tile_x)
    os.environ["STAR_UVT_TILE_Y"] = str(config.tile_y)
    os.environ["STAR_UVT_TILE_T"] = str(config.tile_t)
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(config.tile_capacity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--tube-count", type=int, default=1792)
    parser.add_argument("--steps-before-split", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--uvt-spatial-precision", type=float, default=0.125)
    parser.add_argument("--uvt-temporal-precision", type=float, default=2.0)
    parser.add_argument("--uvt-opacity", type=float, default=0.7)
    parser.add_argument("--split-offset", type=float, default=0.0)
    parser.add_argument("--split-temporal-precision-scale", type=float, default=1.0)
    parser.add_argument("--split-opacity-scale", type=float, default=1.0)
    parser.add_argument("--split-depth-offset", type=float, default=0.0)
    parser.add_argument("--uvt-tile-t", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--uvt-tile-capacity", type=int, choices=(32, 64, 128, 256), default=128)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("mps")
    target = load_video_target(
        args.video_path,
        target_size=args.target_size,
        max_frames=args.max_frames,
        device=device,
    )
    config = UVTRenderConfig(
        height=int(target.shape[1]),
        width=int(target.shape[2]),
        frames=int(target.shape[0]),
        tile_t=args.uvt_tile_t,
        tile_capacity=args.uvt_tile_capacity,
    )
    apply_uvt_tile_env(config)
    model = ScreenTimeTubeModel.from_video_samples(
        target,
        config,
        tube_count=args.tube_count,
        seed=args.seed,
        spatial_precision=args.uvt_spatial_precision,
        temporal_precision=args.uvt_temporal_precision,
        opacity=args.uvt_opacity,
    )
    losses = fit_uvt(
        model,
        target,
        steps=args.steps_before_split,
        lr=args.lr,
        final_lr=None,
        final_lr_start_step=None,
        backend="metal_tile",
    )

    with torch.no_grad():
        pre_image, pre_stats = metal_stats(model)
        pre_loss = torch.mean((pre_image - target).square()).item()
        split_model = model.temporal_split(
            offset_frames=args.split_offset,
            temporal_precision_scale=args.split_temporal_precision_scale,
            opacity_scale=args.split_opacity_scale,
            depth_offset=args.split_depth_offset,
        )
        post_image, post_stats = metal_stats(split_model)
        post_loss = torch.mean((post_image - target).square()).item()
        split_diff = torch.mean((post_image - pre_image).square()).item()
        split_l1 = torch.mean((post_image - pre_image).abs()).item()

    row = {
        "video_path": str(args.video_path),
        "height": int(target.shape[1]),
        "width": int(target.shape[2]),
        "frames": int(target.shape[0]),
        "steps_before_split": args.steps_before_split,
        "pre_split_train_loss": losses[-1],
        "pre_split_render_loss": pre_loss,
        "pre_split_psnr": mse_to_psnr(pre_loss),
        "post_split_render_loss": post_loss,
        "post_split_psnr": mse_to_psnr(post_loss),
        "post_minus_pre_loss": post_loss - pre_loss,
        "post_vs_pre_mse": split_diff,
        "post_vs_pre_l1": split_l1,
        "pre_split_tube_count": int(model.tube_count),
        "post_split_tube_count": int(split_model.tube_count),
        "split": {
            "offset_frames": args.split_offset,
            "temporal_precision_scale": args.split_temporal_precision_scale,
            "opacity_scale": args.split_opacity_scale,
            "depth_offset": args.split_depth_offset,
        },
        "tile_t": args.uvt_tile_t,
        "tile_capacity": args.uvt_tile_capacity,
        "pre_metal_stats": pre_stats,
        "post_metal_stats": post_stats,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
