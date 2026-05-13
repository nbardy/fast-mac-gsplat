from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_metal_forward_timing_probe import (  # noqa: E402
    _batch,
    _camera,
    _parse_int_list,
)
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    centered_frame_times,
    compile_projective_rational_tubes,
    fit_camera_path_polynomial,
)
from research_project.trainer_harness.projective_rational_metal_autograd import (  # noqa: E402
    render_projective_rational_tubes_metal_direct_serial_backward,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    ProjectiveRationalTileConfig,
    UVTRenderConfig,
    apply_projective_rational_tile_env,
    parse_projective_rational_tile_config,
    recommend_projective_rational_tile_config,
    render_projective_rational_tubes_tiled,
)


PARAM_NAMES = ("h_coeff", "lambda_uv", "lambda_t", "center_t", "opacity", "color")


def _sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _resolve_tile_config(args: argparse.Namespace) -> ProjectiveRationalTileConfig:
    if args.tile_config is None:
        return ProjectiveRationalTileConfig(args.tile_x, args.tile_y, args.tile_t, args.tile_capacity)
    if args.tile_config == "auto":
        return recommend_projective_rational_tile_config(
            tube_count=max(args.tube_counts),
            camera_motion_scale=args.camera_motion_scale,
        )
    return parse_projective_rational_tile_config(args.tile_config)


def _make_case(
    *,
    tube_count: int,
    frames: int,
    width: int,
    height: int,
    seed: int,
    camera_motion_scale: float,
    tile_config: ProjectiveRationalTileConfig,
    support_alpha_threshold: float | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, UVTRenderConfig, dict[str, int | float]]:
    times = centered_frame_times(frames)
    k_seq, w2c_seq = _camera(frames, width, height, times, motion_scale=camera_motion_scale)
    camera_path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=2, frame_times=times)
    projected = compile_projective_rational_tubes(_batch(tube_count, seed=seed), camera_path)
    config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        support_alpha_threshold=support_alpha_threshold,
        **tile_config.as_render_kwargs(),
    )

    params = {
        "h_coeff": projected.h_coeff.to("mps").detach().clone().requires_grad_(True),
        "lambda_uv": projected.lambda_uv.to("mps").detach().clone().requires_grad_(True),
        "lambda_t": projected.lambda_t.to("mps").detach().clone().requires_grad_(True),
        "center_t": projected.center_t.to("mps").detach().clone().requires_grad_(True),
        "opacity": projected.opacity.to("mps").detach().clone().requires_grad_(True),
        "color": projected.color.to("mps").detach().clone().requires_grad_(True),
    }
    with torch.no_grad():
        target_color = torch.roll(projected.color, shifts=1, dims=0).to("mps")
        target = render_projective_rational_tubes_tiled(
            projected.h_coeff.to("mps"),
            projected.lambda_uv.to("mps"),
            projected.lambda_t.to("mps"),
            projected.t0.to("mps") if hasattr(projected, "t0") else projected.center_t.to("mps"),
            projected.opacity.to("mps"),
            target_color,
            config,
        ).detach()
        aux = render_projective_rational_tubes_tiled(
            params["h_coeff"].detach(),
            params["lambda_uv"].detach(),
            params["lambda_t"].detach(),
            params["center_t"].detach(),
            params["opacity"].detach(),
            params["color"].detach(),
            config,
            return_aux=True,
        )
    tile_counts = aux.tile_counts.cpu()
    tile_overflow = aux.tile_overflow.cpu()
    stats = {
        "camera_fit_error": camera_path.fit_error,
        "active_tile_count": int((tile_counts > 0).sum().item()),
        "total_tile_pairs": int(torch.clamp(tile_counts, max=config.tile_capacity).sum().item()),
        "max_tile_count": int(tile_counts.max().item()),
        "overflow_tile_count": int((tile_overflow > 0).sum().item()),
    }
    return params, target, config, stats


def _step(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    config: UVTRenderConfig,
    *,
    forward_mode: str,
    backward_mode: str,
    lr: float,
) -> float:
    for value in params.values():
        value.grad = None
    image = render_projective_rational_tubes_metal_direct_serial_backward(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        config,
        forward_mode=forward_mode,
        backward_mode=backward_mode,
    )
    loss = (image - target).square().mean()
    loss.backward()
    with torch.no_grad():
        for value in params.values():
            value -= lr * value.grad
    return float(loss.detach().cpu())


def _time_case(
    *,
    tube_count: int,
    frames: int,
    width: int,
    height: int,
    tile_config: ProjectiveRationalTileConfig,
    warmups: int,
    repeats: int,
    seed: int,
    camera_motion_scale: float,
    forward_mode: str,
    backward_mode: str,
    lr: float,
    support_alpha_threshold: float | None,
) -> dict[str, Any]:
    params, target, config, tile_stats = _make_case(
        tube_count=tube_count,
        frames=frames,
        width=width,
        height=height,
        seed=seed,
        camera_motion_scale=camera_motion_scale,
        tile_config=tile_config,
        support_alpha_threshold=support_alpha_threshold,
    )
    losses: list[float] = []
    for _ in range(warmups):
        losses.append(_step(params, target, config, forward_mode=forward_mode, backward_mode=backward_mode, lr=lr))
        _sync()

    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        losses.append(_step(params, target, config, forward_mode=forward_mode, backward_mode=backward_mode, lr=lr))
        _sync()
        elapsed.append((time.perf_counter() - start) * 1000.0)

    return {
        "tube_count": tube_count,
        "frames": frames,
        "width": width,
        "height": height,
        "forward_mode": forward_mode,
        "backward_mode": backward_mode,
        "lr": lr,
        "support_alpha_threshold": support_alpha_threshold,
        "warmups": warmups,
        "repeats": repeats,
        "median_step_ms": statistics.median(elapsed),
        "min_step_ms": min(elapsed),
        "max_step_ms": max(elapsed),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "losses": losses,
        **tile_stats,
        "pass": tile_stats["overflow_tile_count"] == 0 and losses[-1] < losses[0],
    }


def run_probe(
    *,
    tube_counts: list[int],
    frames: int,
    width: int,
    height: int,
    tile_config: ProjectiveRationalTileConfig,
    warmups: int,
    repeats: int,
    seed: int,
    camera_motion_scale: float,
    forward_mode: str,
    backward_mode: str,
    lr: float,
    support_alpha_threshold: float | None,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the PRT train-step timing probe")
    rows = [
        _time_case(
            tube_count=tube_count,
            frames=frames,
            width=width,
            height=height,
            tile_config=tile_config,
            warmups=warmups,
            repeats=repeats,
            seed=seed + index,
            camera_motion_scale=camera_motion_scale,
            forward_mode=forward_mode,
            backward_mode=backward_mode,
            lr=lr,
            support_alpha_threshold=support_alpha_threshold,
        )
        for index, tube_count in enumerate(tube_counts)
    ]
    return {
        "name": "projective_rational_train_step_timing_probe",
        "note": "Diagnostic PRT forward+loss+backward+SGD timing; not a video-quality benchmark.",
        "camera_motion_scale": camera_motion_scale,
        "support_alpha_threshold": support_alpha_threshold,
        "tile_config_key": tile_config.key,
        "tile_config": tile_config.as_dict(),
        "pass": all(bool(row["pass"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube-counts", type=_parse_int_list, default=[16, 64])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--tile-x", type=int, default=8)
    parser.add_argument("--tile-y", type=int, default=8)
    parser.add_argument("--tile-t", type=int, default=2)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--tile-config", default="auto", help="'auto' or an explicit config like 4x4x2:256")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--camera-motion-scale", type=float, default=1.0)
    parser.add_argument("--support-alpha-threshold", type=float)
    parser.add_argument("--forward-mode", choices=("direct", "tiled"), default="tiled")
    parser.add_argument(
        "--backward-mode",
        choices=("direct_serial", "tile_pair_atomic", "tile_pixel_atomic"),
        default="tile_pixel_atomic",
    )
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    try:
        tile_config = _resolve_tile_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    apply_projective_rational_tile_env(tile_config)

    summary = run_probe(
        tube_counts=args.tube_counts,
        frames=args.frames,
        width=args.width,
        height=args.height,
        tile_config=tile_config,
        warmups=args.warmups,
        repeats=args.repeats,
        seed=args.seed,
        camera_motion_scale=args.camera_motion_scale,
        forward_mode=args.forward_mode,
        backward_mode=args.backward_mode,
        lr=args.lr,
        support_alpha_threshold=args.support_alpha_threshold,
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
