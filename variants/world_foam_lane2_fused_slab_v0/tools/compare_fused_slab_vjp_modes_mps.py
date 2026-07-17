#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from gate4_moving_ray_slab_compiler import DEFAULT_CONFIG, SyntheticRayMotion, _load_config, load_powerfoam_training_data
from smoke_fused_slab_affine_realray_mps import _parse_int_list
from train_eval_fused_slab_mixed_mps import RESULTS_DIR, _run_one


VALID_VJP_MODES = (
    "reduce",
    "direct_atomic",
    "direct_atomic_grad_only",
    "direct_atomic_rgb_only",
    "direct_atomic_track",
    "direct_atomic_grad_only_ownerupdate",
    "fused_mse_rgb_only",
)


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modes:
        raise ValueError("expected at least one VJP mode")
    unknown = sorted(set(modes) - set(VALID_VJP_MODES))
    if unknown:
        raise ValueError(f"unknown VJP mode(s): {unknown}; expected one of {VALID_VJP_MODES}")
    return modes


def _median_ms(row: dict[str, Any], key: str) -> float | None:
    summary = row.get("step_summary", {}).get(key, {})
    if "median_s" not in summary:
        return None
    return float(summary["median_s"]) * 1000.0


def _scale(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or first <= 0.0:
        return None
    return float(last) / float(first)


def summarize_mode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    total_first = _median_ms(first, "total") if first else None
    total_last = _median_ms(last, "total") if last else None
    backward_first = _median_ms(first, "backward") if first else None
    backward_last = _median_ms(last, "backward") if last else None
    return {
        "status": "ok" if rows and all(row.get("status") == "ok" for row in rows) else "failed",
        "frame_counts": [int(row["frame_count"]) for row in rows],
        "total_median_ms_by_frame": {
            str(int(row["frame_count"])): _median_ms(row, "total") for row in rows
        },
        "render_median_ms_by_frame": {
            str(int(row["frame_count"])): _median_ms(row, "render") for row in rows
        },
        "backward_median_ms_by_frame": {
            str(int(row["frame_count"])): _median_ms(row, "backward") for row in rows
        },
        "optimizer_median_ms_by_frame": {
            str(int(row["frame_count"])): _median_ms(row, "optimizer") for row in rows
        },
        "train_psnr_by_frame": {
            str(int(row["frame_count"])): float(row["final_train_psnr"]) for row in rows
        },
        "heldout_psnr_by_frame": {
            str(int(row["frame_count"])): float(row["final_heldout_psnr"]) for row in rows
        },
        "total_median_scale_first_to_last": _scale(total_first, total_last),
        "backward_median_scale_first_to_last": _scale(backward_first, backward_last),
        "row_wall_s_by_frame": {
            str(int(row["frame_count"])): float(row.get("wall_timing", {}).get("total_run_s", 0.0)) for row in rows
        },
        "train_loop_s_by_frame": {
            str(int(row["frame_count"])): float(row.get("wall_timing", {}).get("train_loop_s", 0.0)) for row in rows
        },
    }


def run_compare(
    *,
    config_path: Path,
    frame_counts: tuple[int, ...],
    vjp_modes: tuple[str, ...],
    render_size: int,
    site_count: int,
    time_slabs: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    residual_depth_padding: float,
    synthetic_motion: SyntheticRayMotion,
    steps: int,
    warmup_steps: int,
    lr: float,
    vjp_reduce_chunk_size: int,
    alpha_aux_weight: float,
    depth_aux_weight: float,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    shared_load_start = time.perf_counter()
    shared_data_cfg = _load_config(config_path, max_frames=max(frame_counts), render_size=render_size)
    shared_training_data = load_powerfoam_training_data(shared_data_cfg, torch.device("cpu"))
    shared_load_s = time.perf_counter() - shared_load_start

    mode_payloads: list[dict[str, Any]] = []
    for mode in vjp_modes:
        mode_start = time.perf_counter()
        rows = [
            _run_one(
                config_path=config_path,
                frame_count=frame_count,
                render_size=render_size,
                site_count=site_count,
                time_slabs=time_slabs,
                near=near,
                far=far,
                density=density,
                invalid_epsilon=invalid_epsilon,
                transmittance_threshold=transmittance_threshold,
                residual_depth_padding=residual_depth_padding,
                synthetic_motion=synthetic_motion,
                steps=steps,
                warmup_steps=warmup_steps,
                lr=lr,
                vjp_reduce_chunk_size=vjp_reduce_chunk_size,
                vjp_mode=mode,
                alpha_aux_weight=alpha_aux_weight,
                depth_aux_weight=depth_aux_weight,
                cached_training_data=shared_training_data,
            )
            for frame_count in frame_counts
        ]
        mode_payloads.append(
            {
                "vjp_mode": mode,
                "status": "ok" if all(row.get("status") == "ok" for row in rows) else "failed",
                "mode_wall_s": float(time.perf_counter() - mode_start),
                "summary": summarize_mode_rows(rows),
                "rows": rows,
            }
        )

    return {
        "benchmark": "world_foam_lane2_fused_slab_vjp_mode_compare_mps",
        "status": "ok" if all(mode["status"] == "ok" for mode in mode_payloads) else "failed",
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "vjp_modes": list(vjp_modes),
        "render_size": int(render_size),
        "site_count": int(site_count),
        "time_slabs": int(time_slabs),
        "steps": int(steps),
        "warmup_steps": int(warmup_steps),
        "synthetic_motion": synthetic_motion.to_dict(),
        "shared_loaded_data": {
            "enabled": True,
            "max_frame_count": int(max(frame_counts)),
            "load_s": float(shared_load_s),
        },
        "modes": mode_payloads,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fused slab affine VJP modes in one MPS process.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,16")
    parser.add_argument(
        "--vjp-modes",
        default="direct_atomic_grad_only,direct_atomic_rgb_only,direct_atomic_track",
        help="Comma-separated VJP modes to run against one shared max-frame load.",
    )
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--density", type=float, default=10.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--residual-depth-padding", type=float, default=0.001)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--vjp-reduce-chunk-size", type=int, default=16)
    parser.add_argument("--alpha-aux-weight", type=float, default=0.0)
    parser.add_argument("--depth-aux-weight", type=float, default=0.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "fused_slab_vjp_mode_compare_mps.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_compare(
        config_path=args.config,
        frame_counts=_parse_int_list(args.frame_counts),
        vjp_modes=_parse_modes(args.vjp_modes),
        render_size=args.render_size,
        site_count=args.site_count,
        time_slabs=args.time_slabs,
        near=args.near,
        far=args.far,
        density=args.density,
        invalid_epsilon=args.invalid_epsilon,
        transmittance_threshold=args.transmittance_threshold,
        residual_depth_padding=args.residual_depth_padding,
        synthetic_motion=SyntheticRayMotion(
            origin_velocity=(args.origin_velocity_x, args.origin_velocity_y, args.origin_velocity_z),
            direction_velocity=(args.direction_velocity_x, args.direction_velocity_y, args.direction_velocity_z),
        ),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        vjp_reduce_chunk_size=args.vjp_reduce_chunk_size,
        alpha_aux_weight=args.alpha_aux_weight,
        depth_aux_weight=args.depth_aux_weight,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
