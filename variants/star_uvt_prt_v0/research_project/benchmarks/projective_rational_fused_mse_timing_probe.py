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

from research_project.benchmarks.projective_rational_train_step_timing_probe import (  # noqa: E402
    PARAM_NAMES,
    _make_case,
    _parse_int_list,
    _sync,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    ProjectiveRationalTileConfig,
    apply_projective_rational_tile_env,
    parse_projective_rational_tile_config,
    projective_rational_tile_pixel_atomic_backward,
    projective_rational_tile_pixel_fused_mse_backward,
    recommend_projective_rational_train_speed_tile_policy,
    render_projective_rational_tubes_tiled,
)


def _grads_from_tuple(values: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
    return {name: grad.detach().cpu() for name, grad in zip(PARAM_NAMES, values[:6], strict=True)}


def _separate_step(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    config,
) -> tuple[float, dict[str, torch.Tensor]]:
    image = render_projective_rational_tubes_tiled(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        config,
    )
    loss = (image - target).square().mean()
    grad_image = 2.0 * (image - target) / float(image.numel())
    grads = projective_rational_tile_pixel_atomic_backward(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        grad_image,
        config,
    )
    return float(loss.detach().cpu()), _grads_from_tuple(grads)


def _fused_step(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    config,
) -> tuple[float, dict[str, torch.Tensor], int]:
    result = projective_rational_tile_pixel_fused_mse_backward(
        params["h_coeff"],
        params["lambda_uv"],
        params["lambda_t"],
        params["center_t"],
        params["opacity"],
        params["color"],
        target,
        config,
    )
    loss = result.loss_sum.detach().cpu()[0] / float(config.frames * config.height * config.width * 3)
    grads = (
        result.grad_h_coeff,
        result.grad_lambda_uv,
        result.grad_lambda_t,
        result.grad_center_t,
        result.grad_opacity,
        result.grad_color,
    )
    overflow_tile_count = int((result.tile_overflow.detach().cpu() > 0).sum().item())
    return float(loss), _grads_from_tuple(grads), overflow_tile_count


def _time_call(fn, *, warmups: int, repeats: int) -> tuple[list[float], Any]:
    last_result = None
    for _ in range(warmups):
        last_result = fn()
        _sync()
    rows = []
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = fn()
        _sync()
        rows.append((time.perf_counter() - start) * 1000.0)
    return rows, last_result


def _resolve_policy(args: argparse.Namespace) -> tuple[ProjectiveRationalTileConfig, float | None, str | None]:
    if args.prt_tile_policy == "train_speed":
        policy = recommend_projective_rational_train_speed_tile_policy(tube_count=max(args.tube_counts))
        return policy.tile_config, policy.support_alpha_threshold, policy.name
    if args.tile_config is not None:
        return parse_projective_rational_tile_config(args.tile_config), args.support_alpha_threshold, None
    return (
        ProjectiveRationalTileConfig(args.tile_x, args.tile_y, args.tile_t, args.tile_capacity),
        args.support_alpha_threshold,
        None,
    )


def _compare_grads(reference: dict[str, torch.Tensor], fused: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for name in PARAM_NAMES:
        diff = (fused[name] - reference[name]).abs()
        ref_abs = reference[name].abs()
        rel = diff / torch.clamp(ref_abs, min=1.0e-8)
        rows.append(
            {
                "param": name,
                "max_abs_error": float(diff.max().item()),
                "max_rel_error": float(rel.max().item()),
            }
        )
    return rows


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the fused MSE timing probe")
    tile_config, support_alpha_threshold, policy_name = _resolve_policy(args)
    apply_projective_rational_tile_env(tile_config)
    rows = []
    for index, tube_count in enumerate(args.tube_counts):
        params, target, config, tile_stats = _make_case(
            tube_count=tube_count,
            frames=args.frames,
            width=args.width,
            height=args.height,
            seed=args.seed + index,
            camera_motion_scale=args.camera_motion_scale,
            tile_config=tile_config,
            support_alpha_threshold=support_alpha_threshold,
        )
        separate_loss, separate_grads = _separate_step(params, target, config)
        fused_loss, fused_grads, fused_overflow = _fused_step(params, target, config)
        grad_rows = _compare_grads(separate_grads, fused_grads)
        separate_times, separate_last = _time_call(
            lambda: _separate_step(params, target, config),
            warmups=args.warmups,
            repeats=args.repeats,
        )
        fused_times, fused_last = _time_call(
            lambda: _fused_step(params, target, config),
            warmups=args.warmups,
            repeats=args.repeats,
        )
        separate_last_loss = float(separate_last[0])
        fused_last_loss = float(fused_last[0])
        separate_median = statistics.median(separate_times)
        fused_median = statistics.median(fused_times)
        rows.append(
            {
                "tube_count": tube_count,
                "frames": args.frames,
                "width": args.width,
                "height": args.height,
                "separate_loss": separate_loss,
                "fused_loss": fused_loss,
                "loss_abs_error": abs(fused_loss - separate_loss),
                "separate_last_loss": separate_last_loss,
                "fused_last_loss": fused_last_loss,
                "fused_overflow_tile_count": fused_overflow,
                "grad_rows": grad_rows,
                "max_grad_abs_error": max(row["max_abs_error"] for row in grad_rows),
                "max_grad_rel_error": max(row["max_rel_error"] for row in grad_rows),
                "separate_samples_ms": separate_times,
                "fused_samples_ms": fused_times,
                "separate_median_ms": separate_median,
                "fused_median_ms": fused_median,
                "fused_speedup": separate_median / fused_median if fused_median > 0.0 else 0.0,
                "pass": abs(fused_loss - separate_loss) <= args.loss_tol
                and fused_overflow == 0
                and all(row["max_abs_error"] <= args.abs_tol or row["max_rel_error"] <= args.rel_tol for row in grad_rows),
                **tile_stats,
            }
        )
    return {
        "name": "projective_rational_fused_mse_timing_probe",
        "note": "Compares separate tiled PRT render plus MSE grad plus tile-pixel backward against fused MSE backward.",
        "tile_config_key": tile_config.key,
        "tile_config": tile_config.as_dict(),
        "prt_tile_policy": policy_name,
        "support_alpha_threshold": support_alpha_threshold,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "abs_tol": args.abs_tol,
        "rel_tol": args.rel_tol,
        "loss_tol": args.loss_tol,
        "pass": all(bool(row["pass"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube-counts", type=_parse_int_list, default=[1024])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--tile-x", type=int, default=8)
    parser.add_argument("--tile-y", type=int, default=8)
    parser.add_argument("--tile-t", type=int, default=2)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--tile-config")
    parser.add_argument("--prt-tile-policy", choices=("train_speed",), default="train_speed")
    parser.add_argument("--support-alpha-threshold", type=float)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--camera-motion-scale", type=float, default=1.0)
    parser.add_argument("--abs-tol", type=float, default=5.0e-4)
    parser.add_argument("--rel-tol", type=float, default=5.0e-2)
    parser.add_argument("--loss-tol", type=float, default=1.0e-5)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    summary = run_probe(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
