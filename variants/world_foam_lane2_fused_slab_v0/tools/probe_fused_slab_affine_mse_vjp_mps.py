#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    fused_slab_affine_num32_den16_autograd,
    fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only,
)


def _compare_case(
    *,
    name: str,
    row_index: torch.Tensor,
    row_offsets: torch.Tensor,
    depth_num: torch.Tensor,
    depth_den: torch.Tensor,
    sites: torch.Tensor,
    site_rgba_init: torch.Tensor,
    ray_coeff: torch.Tensor,
    frame_t: torch.Tensor,
    target_rgb: torch.Tensor,
    config: RealRayReplayConfig,
    time_slab_count: int,
    row_count: int,
    loss_atol: float,
    grad_atol: float,
) -> dict[str, Any]:
    site_rgba_ref = site_rgba_init.detach().clone().requires_grad_(True)
    ref_rgb, _, _ = fused_slab_affine_num32_den16_autograd(
        row_index,
        row_offsets,
        depth_num,
        depth_den,
        sites,
        site_rgba_ref,
        ray_coeff,
        frame_t,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
        vjp_mode="direct_atomic_rgb_only",
    )
    ref_loss = F.mse_loss(ref_rgb, target_rgb)
    ref_loss.backward()
    torch.mps.synchronize()
    ref_grad = site_rgba_ref.grad.detach()

    fused_loss, fused_grad = fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only(
        row_index,
        row_offsets,
        depth_num,
        depth_den,
        sites,
        site_rgba_init,
        ray_coeff,
        frame_t,
        target_rgb,
        config,
        time_slab_count=time_slab_count,
        row_count=row_count,
    )
    torch.mps.synchronize()

    loss_abs_diff = float((fused_loss.reshape(()) - ref_loss.detach()).abs().cpu().item())
    grad_abs_diff = (fused_grad - ref_grad).abs()
    grad_max_abs_diff = float(grad_abs_diff.max().cpu().item())
    grad_row_max_abs_diff = grad_abs_diff.max(dim=1).values.detach().cpu()
    touched_site_count = int((ref_grad.abs().sum(dim=1) > 1.0e-12).sum().cpu().item())
    return {
        "name": name,
        "status": "ok" if loss_abs_diff <= loss_atol and grad_max_abs_diff <= grad_atol else "failed",
        "frame_count": int(frame_t.numel()),
        "site_count": int(sites.shape[0]),
        "track_count": int(ray_coeff.shape[0]),
        "loss_abs_diff": loss_abs_diff,
        "grad_max_abs_diff": grad_max_abs_diff,
        "grad_row_max_abs_diff": [float(v) for v in grad_row_max_abs_diff.tolist()],
        "touched_site_count": touched_site_count,
        "reference_loss": float(ref_loss.detach().cpu().item()),
        "fused_loss": float(fused_loss.detach().cpu().item()),
        "reference_grad_abs_sum": float(ref_grad.abs().sum().cpu().item()),
        "fused_grad_abs_sum": float(fused_grad.abs().sum().cpu().item()),
    }


def _tiny_two_site_case(device: torch.device) -> dict[str, Any]:
    return {
        "name": "tiny_two_site_two_frame",
        "row_index": torch.tensor([0], device=device, dtype=torch.int32),
        "row_offsets": torch.tensor([0, 0], device=device, dtype=torch.int32),
        "depth_num": torch.empty((0, 2), device=device, dtype=torch.float32),
        "depth_den": torch.empty((0, 2), device=device, dtype=torch.float16),
        "sites": torch.tensor(
            [
                [0.0, 0.0, 3.0, 0.0, 0.0],
                [1.0, 0.0, 3.0, 1.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        ),
        "site_rgba_init": torch.tensor(
            [
                [0.80, 0.15, 0.25, 1.20],
                [0.10, 0.70, 0.35, 0.90],
            ],
            device=device,
            dtype=torch.float32,
        ),
        "ray_coeff": torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            device=device,
            dtype=torch.float32,
        ),
        "frame_t": torch.tensor([0.0, 1.0], device=device, dtype=torch.float32),
        "target_rgb": torch.tensor(
            [[[0.20, 0.30, 0.40], [0.25, 0.35, 0.45]]],
            device=device,
            dtype=torch.float32,
        ),
        "time_slab_count": 1,
        "row_count": 1,
    }


def _multi_site_case(device: torch.device, *, frame_count: int, site_count: int = 12) -> dict[str, Any]:
    site_ids = torch.arange(site_count, device=device, dtype=torch.float32)
    x = site_ids * 3.0
    sites = torch.stack(
        [
            x,
            torch.zeros_like(x),
            torch.full_like(x, 3.05),
            torch.full_like(x, 0.5),
            torch.zeros_like(x),
        ],
        dim=1,
    )
    site_rgba_init = torch.stack(
        [
            0.15 + 0.025 * site_ids,
            0.65 - 0.015 * site_ids,
            0.20 + 0.010 * site_ids,
            0.55 + 0.050 * torch.remainder(site_ids, 4),
        ],
        dim=1,
    )
    ray_coeff = torch.zeros((site_count, 12), device=device, dtype=torch.float32)
    ray_coeff[:, 0] = x
    ray_coeff[:, 8] = 1.0
    frame_t = torch.linspace(0.0, 1.0, frame_count, device=device, dtype=torch.float32)
    track_id = site_ids[:, None]
    frame_id = torch.arange(frame_count, device=device, dtype=torch.float32)[None, :]
    target_rgb = torch.stack(
        [
            0.10 + 0.015 * track_id + 0.010 * frame_id,
            0.20 + 0.010 * track_id - 0.005 * frame_id,
            0.30 + 0.007 * track_id + 0.003 * frame_id,
        ],
        dim=2,
    ).contiguous()
    return {
        "name": f"multi_site_{site_count}_sites_{frame_count}_frames",
        "row_index": torch.zeros((site_count,), device=device, dtype=torch.int32),
        "row_offsets": torch.tensor([0, 0], device=device, dtype=torch.int32),
        "depth_num": torch.empty((0, 2), device=device, dtype=torch.float32),
        "depth_den": torch.empty((0, 2), device=device, dtype=torch.float16),
        "sites": sites,
        "site_rgba_init": site_rgba_init,
        "ray_coeff": ray_coeff,
        "frame_t": frame_t,
        "target_rgb": target_rgb,
        "time_slab_count": 1,
        "row_count": 1,
    }


def run_probe(*, loss_atol: float = 1.0e-6, grad_atol: float = 1.0e-5) -> dict[str, Any]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    device = torch.device("mps")
    config = RealRayReplayConfig(near=0.1, far=6.0, invalid_epsilon=1.0e-6, transmittance_threshold=1.0e-4)
    cases = [_tiny_two_site_case(device), *[_multi_site_case(device, frame_count=f) for f in (2, 4, 8, 16)]]
    case_results = [
        _compare_case(
            config=config,
            loss_atol=loss_atol,
            grad_atol=grad_atol,
            **case,
        )
        for case in cases
    ]
    return {
        "benchmark": "world_foam_lane2_fused_slab_affine_mse_vjp_parity_mps",
        "status": "ok" if all(case["status"] == "ok" for case in case_results) else "failed",
        "loss_abs_diff": max(case["loss_abs_diff"] for case in case_results),
        "grad_max_abs_diff": max(case["grad_max_abs_diff"] for case in case_results),
        "loss_atol": float(loss_atol),
        "grad_atol": float(grad_atol),
        "cases": case_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe fused affine RGB-MSE VJP parity on MPS.")
    parser.add_argument("--loss-atol", type=float, default=1.0e-6)
    parser.add_argument("--grad-atol", type=float, default=1.0e-5)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_probe(loss_atol=args.loss_atol, grad_atol=args.grad_atol)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
