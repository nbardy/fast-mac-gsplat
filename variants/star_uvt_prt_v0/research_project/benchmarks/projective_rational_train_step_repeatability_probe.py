from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.projective_rational_train_step_timing_probe import (  # noqa: E402
    PARAM_NAMES,
    _make_case,
    _parse_int_list,
    _resolve_tile_config,
)
from research_project.trainer_harness.projective_rational_metal_autograd import (  # noqa: E402
    render_projective_rational_tubes_metal_direct_serial_backward,
)
from torch_gsplat_bridge_star_uvt_prt import apply_projective_rational_tile_env  # noqa: E402


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _gradient_row(
    *,
    tube_count: int,
    frames: int,
    width: int,
    height: int,
    seed: int,
    camera_motion_scale: float,
    tile_config,
    forward_mode: str,
    backward_mode: str,
) -> tuple[dict[str, torch.Tensor], float, dict[str, Any]]:
    params, target, config, tile_stats = _make_case(
        tube_count=tube_count,
        frames=frames,
        width=width,
        height=height,
        seed=seed,
        camera_motion_scale=camera_motion_scale,
        tile_config=tile_config,
    )
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
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    grads = {name: params[name].grad.detach().cpu() for name in PARAM_NAMES}
    return grads, float(loss.detach().cpu()), tile_stats


def run_probe(
    *,
    tube_counts: list[int],
    repeats: int,
    abs_tol: float,
    frames: int,
    width: int,
    height: int,
    seed: int,
    camera_motion_scale: float,
    tile_config,
    forward_mode: str,
    backward_mode: str,
) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the PRT train-step repeatability probe")
    if repeats < 2:
        raise ValueError("repeats must be at least 2")

    rows = []
    for index, tube_count in enumerate(tube_counts):
        case_seed = seed + index
        first_grads: dict[str, torch.Tensor] | None = None
        first_loss: float | None = None
        digests: list[dict[str, str]] = []
        max_grad_delta = 0.0
        max_loss_delta = 0.0
        tile_stats: dict[str, Any] = {}
        for _ in range(repeats):
            grads, loss, tile_stats = _gradient_row(
                tube_count=tube_count,
                frames=frames,
                width=width,
                height=height,
                seed=case_seed,
                camera_motion_scale=camera_motion_scale,
                tile_config=tile_config,
                forward_mode=forward_mode,
                backward_mode=backward_mode,
            )
            digests.append({name: _tensor_digest(grads[name]) for name in PARAM_NAMES})
            if first_grads is None:
                first_grads = grads
                first_loss = loss
                continue
            for name in PARAM_NAMES:
                delta = float((grads[name] - first_grads[name]).abs().max().item())
                max_grad_delta = max(max_grad_delta, delta)
            max_loss_delta = max(max_loss_delta, abs(loss - float(first_loss)))

        unique_digest_counts = {
            name: len({digest[name] for digest in digests})
            for name in PARAM_NAMES
        }
        rows.append(
            {
                "tube_count": tube_count,
                "frames": frames,
                "width": width,
                "height": height,
                "forward_mode": forward_mode,
                "backward_mode": backward_mode,
                "repeats": repeats,
                "max_grad_delta": max_grad_delta,
                "max_loss_delta": max_loss_delta,
                "unique_digest_counts": unique_digest_counts,
                **tile_stats,
                "pass": max_grad_delta <= abs_tol and max_loss_delta <= abs_tol and tile_stats["overflow_tile_count"] == 0,
            }
        )

    return {
        "name": "projective_rational_train_step_repeatability_probe",
        "note": "Same-state PRT backward repeatability check for loss and parameter gradients.",
        "camera_motion_scale": camera_motion_scale,
        "tile_config_key": tile_config.key,
        "tile_config": tile_config.as_dict(),
        "abs_tol": abs_tol,
        "pass": all(bool(row["pass"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube-counts", type=_parse_int_list, default=[16, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--abs-tol", type=float, default=1.0e-6)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--tile-x", type=int, default=8)
    parser.add_argument("--tile-y", type=int, default=8)
    parser.add_argument("--tile-t", type=int, default=2)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--tile-config", default="auto", help="'auto' or an explicit config like 4x4x2:256")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--camera-motion-scale", type=float, default=1.0)
    parser.add_argument("--forward-mode", choices=("direct", "tiled"), default="tiled")
    parser.add_argument("--backward-mode", choices=("direct_serial", "tile_pair_atomic"), default="tile_pair_atomic")
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    try:
        tile_config = _resolve_tile_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    apply_projective_rational_tile_env(tile_config)

    summary = run_probe(
        tube_counts=args.tube_counts,
        repeats=args.repeats,
        abs_tol=args.abs_tol,
        frames=args.frames,
        width=args.width,
        height=args.height,
        seed=args.seed,
        camera_motion_scale=args.camera_motion_scale,
        tile_config=tile_config,
        forward_mode=args.forward_mode,
        backward_mode=args.backward_mode,
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
