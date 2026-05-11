from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig, render_uvt_tubes  # noqa: E402

try:
    from research_project.trainer_harness.data import load_video_target
    from research_project.trainer_harness.model import ScreenTimeTubeModel, dense_differentiable_render_uvt_tubes
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = Path(__file__).resolve().parents[1] / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from data import load_video_target
    from model import ScreenTimeTubeModel, dense_differentiable_render_uvt_tubes


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def time_dense(model: ScreenTimeTubeModel, *, iterations: int, warmup_iterations: int) -> tuple[float, torch.Tensor]:
    ma, q_uvt, depth0, depth_beta, opacity, color = model.tensors()
    device = ma.device
    image = None
    with torch.no_grad():
        for _ in range(warmup_iterations):
            image = dense_differentiable_render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, model.config)
        synchronize(device)
        started_at = time.perf_counter()
        for _ in range(iterations):
            image = dense_differentiable_render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, model.config)
        synchronize(device)
    if image is None:
        raise AssertionError("dense render did not run")
    return (time.perf_counter() - started_at) * 1000.0 / float(iterations), image


def time_metal(
    model: ScreenTimeTubeModel,
    *,
    reference: torch.Tensor,
    iterations: int,
    warmup_iterations: int,
) -> tuple[float, dict[str, Any]]:
    if not torch.backends.mps.is_available():
        return 0.0, {"metal_skipped": "MPS is not available"}
    ma, q_uvt, depth0, depth_beta, opacity, color = model.tensors()
    for tensor in (ma, q_uvt, depth0, depth_beta, opacity, color):
        if tensor.device.type != "mps":
            raise ValueError("Metal timing requires model tensors on MPS")
    reference_cpu = reference.detach().cpu()
    with torch.no_grad():
        for _ in range(warmup_iterations):
            render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, model.config)
        torch.mps.synchronize()
        started_at = time.perf_counter()
        for _ in range(iterations):
            render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, model.config)
        torch.mps.synchronize()
        result = render_uvt_tubes(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            color,
            model.config,
            return_aux=True,
            reference=reference_cpu,
        )
    if result is None or not hasattr(result, "stats") or result.stats is None:
        raise AssertionError("Metal render did not return stats")
    return (time.perf_counter() - started_at) * 1000.0 / float(iterations), result.stats.__dict__


def run_case(
    *,
    video_path: Path,
    target_size: int,
    max_frames: int,
    tube_count: int,
    seed: int,
    spatial_precision: float,
    temporal_precision: float,
    opacity: float,
    iterations: int,
    warmup_iterations: int,
) -> dict[str, Any]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    target = load_video_target(video_path, target_size=target_size, max_frames=max_frames, device=device)
    config = UVTRenderConfig(height=int(target.shape[1]), width=int(target.shape[2]), frames=int(target.shape[0]))
    model = ScreenTimeTubeModel.from_video_samples(
        target,
        config,
        tube_count=tube_count,
        seed=seed,
        spatial_precision=spatial_precision,
        temporal_precision=temporal_precision,
        opacity=opacity,
    )
    dense_ms, dense_image = time_dense(model, iterations=iterations, warmup_iterations=warmup_iterations)
    metal_ms, metal_stats = time_metal(
        model,
        reference=dense_image,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )
    return {
        "target_size": target_size,
        "frames": max_frames,
        "tube_count": tube_count,
        "seed": seed,
        "spatial_precision": spatial_precision,
        "temporal_precision": temporal_precision,
        "opacity": opacity,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "device": str(device),
        "dense_render_ms": dense_ms,
        "metal_render_ms": metal_ms,
        "metal_stats": metal_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--tube-counts", default="224,448")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spatial-precision", type=float, default=0.25)
    parser.add_argument("--temporal-precision", type=float, default=0.5)
    parser.add_argument("--opacity", type=float, default=0.7)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    rows = [
        run_case(
            video_path=args.video_path,
            target_size=args.target_size,
            max_frames=args.max_frames,
            tube_count=int(tube_count.strip()),
            seed=args.seed,
            spatial_precision=args.spatial_precision,
            temporal_precision=args.temporal_precision,
            opacity=args.opacity,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
        )
        for tube_count in args.tube_counts.split(",")
        if tube_count.strip()
    ]
    report = {"rows": rows}
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
