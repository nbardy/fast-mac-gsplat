from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, TypeVar

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_metal_forward_timing_probe import (  # noqa: E402
    _parse_int_list,
)
from research_project.benchmarks.projective_rational_train_step_timing_probe import (  # noqa: E402
    _make_case,
)
from research_project.trainer_harness.projective_rational_metal_autograd import (  # noqa: E402
    render_projective_rational_tubes_metal_direct_serial_backward,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    ProjectiveRationalTileConfig,
    apply_projective_rational_tile_env,
    parse_projective_rational_tile_config,
    recommend_projective_rational_tile_config,
)


T = TypeVar("T")


def _sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _timed(fn: Callable[[], T]) -> tuple[float, T]:
    start = time.perf_counter()
    result = fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0, result


def _sgd(params: dict[str, torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for value in params.values():
            value -= lr * value.grad


def _step_breakdown(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    config,
    *,
    forward_mode: str,
    backward_mode: str,
    lr: float,
) -> dict[str, float]:
    for value in params.values():
        value.grad = None

    step_start = time.perf_counter()
    forward_ms, image = _timed(
        lambda: render_projective_rational_tubes_metal_direct_serial_backward(
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
    )
    loss_ms, loss = _timed(lambda: (image - target).square().mean())
    backward_ms, _ = _timed(lambda: loss.backward())
    optimizer_ms, _ = _timed(lambda: _sgd(params, lr))
    wall_ms = (time.perf_counter() - step_start) * 1000.0

    return {
        "forward_ms": forward_ms,
        "loss_ms": loss_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "component_sum_ms": forward_ms + loss_ms + backward_ms + optimizer_ms,
        "wall_ms": wall_ms,
        "loss": float(loss.detach().cpu()),
    }


def _resolve_tile_config(args: argparse.Namespace) -> ProjectiveRationalTileConfig:
    if args.tile_config == "auto":
        return recommend_projective_rational_tile_config(
            tube_count=max(args.tube_counts),
            camera_motion_scale=args.camera_motion_scale,
        )
    return parse_projective_rational_tile_config(args.tile_config)


def _median(records: list[dict[str, float]], key: str) -> float:
    return statistics.median(record[key] for record in records)


def _run_case(
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
        losses.append(
            _step_breakdown(
                params,
                target,
                config,
                forward_mode=forward_mode,
                backward_mode=backward_mode,
                lr=lr,
            )["loss"]
        )

    records = [
        _step_breakdown(
            params,
            target,
            config,
            forward_mode=forward_mode,
            backward_mode=backward_mode,
            lr=lr,
        )
        for _ in range(repeats)
    ]
    losses.extend(record["loss"] for record in records)

    medians = {
        "median_forward_ms": _median(records, "forward_ms"),
        "median_loss_ms": _median(records, "loss_ms"),
        "median_backward_ms": _median(records, "backward_ms"),
        "median_optimizer_ms": _median(records, "optimizer_ms"),
        "median_component_sum_ms": _median(records, "component_sum_ms"),
        "median_wall_ms": _median(records, "wall_ms"),
    }
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
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "records": records,
        **medians,
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
        raise RuntimeError("MPS is required for the PRT train-step breakdown probe")
    rows = [
        _run_case(
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
        "name": "projective_rational_train_step_breakdown_probe",
        "note": "Diagnostic PRT train-step timing split into forward, loss, backward, and SGD segments.",
        "camera_motion_scale": camera_motion_scale,
        "support_alpha_threshold": support_alpha_threshold,
        "tile_config_key": tile_config.key,
        "tile_config": tile_config.as_dict(),
        "pass": all(bool(row["pass"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube-counts", type=_parse_int_list, default=[256])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
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
