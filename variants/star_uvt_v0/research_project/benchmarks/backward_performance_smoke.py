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

from torch_gsplat_bridge_star_uvt import UVTRenderConfig  # noqa: E402

try:
    from research_project.trainer_harness.model import dense_differentiable_render_uvt_tubes
    from research_project.trainer_harness.tile_metal_autograd import render_uvt_tubes_metal_tile_backward
except ImportError:  # pragma: no cover - direct script execution fallback.
    HARNESS = Path(__file__).resolve().parents[1] / "trainer_harness"
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    from model import dense_differentiable_render_uvt_tubes
    from tile_metal_autograd import render_uvt_tubes_metal_tile_backward


def make_inputs(tube_count: int, config: UVTRenderConfig, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    centers = torch.rand((tube_count, 2), generator=generator) * torch.tensor([config.width, config.height])
    ma = torch.cat((centers, torch.zeros((tube_count, 1))), dim=-1).to("mps")
    precision = torch.full((tube_count,), 0.16, dtype=torch.float32, device="mps")
    temporal = torch.full((tube_count,), 0.30, dtype=torch.float32, device="mps")
    q_uvt = torch.stack(
        (
            precision,
            torch.zeros_like(precision),
            torch.zeros_like(precision),
            precision,
            torch.zeros_like(precision),
            temporal,
        ),
        dim=-1,
    )
    depth0 = torch.linspace(0.7, 1.3, tube_count, dtype=torch.float32, device="mps")
    depth_beta = torch.zeros((tube_count, 3), dtype=torch.float32, device="mps")
    opacity = torch.full((tube_count,), 0.35, dtype=torch.float32, device="mps")
    color = torch.rand((tube_count, 3), generator=generator, dtype=torch.float32).to("mps").mul(0.8).add(0.1)
    return ma, q_uvt, depth0, depth_beta, opacity, color


def clone_for_grad(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().clone().requires_grad_(True) for value in inputs)


def time_backward(
    name: str,
    inputs: tuple[torch.Tensor, ...],
    config: UVTRenderConfig,
    iterations: int,
    *,
    warmup_iterations: int = 0,
) -> dict[str, object]:
    times = []
    grad_norms = []
    total_iterations = warmup_iterations + iterations
    for iteration in range(total_iterations):
        ma, q_uvt, depth0, depth_beta, opacity, color = clone_for_grad(inputs)
        torch.mps.synchronize()
        started_at = time.perf_counter()
        if name == "dense":
            image = dense_differentiable_render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config)
        elif name == "metal_tile":
            image = render_uvt_tubes_metal_tile_backward(ma, q_uvt, depth0, depth_beta, opacity, color, config)
        else:
            raise ValueError(f"unknown backend {name!r}")
        loss = image.square().mean()
        loss.backward()
        torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if iteration >= warmup_iterations:
            times.append(elapsed_ms)
        grad_norms.append(float(torch.linalg.vector_norm(ma.grad.detach()).cpu()))
    return {
        "backend": name,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "ma_grad_norm_last": grad_norms[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube-count", type=int, default=16)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        print(json.dumps({"metal_skipped": "MPS is not available"}, indent=2, sort_keys=True))
        return
    config = UVTRenderConfig(height=args.size, width=args.size, frames=args.frames)
    inputs = make_inputs(args.tube_count, config, args.seed)
    dense = time_backward("dense", inputs, config, args.iterations, warmup_iterations=args.warmup_iterations)
    metal_tile = time_backward("metal_tile", inputs, config, args.iterations, warmup_iterations=args.warmup_iterations)
    if float(metal_tile["ma_grad_norm_last"]) <= 0.0:
        raise AssertionError(f"expected non-zero Metal tile gradient, got {metal_tile}")
    print(
        json.dumps(
            {
                "tube_count": args.tube_count,
                "height": args.size,
                "width": args.size,
                "frames": args.frames,
                "dense": dense,
                "metal_tile": metal_tile,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
