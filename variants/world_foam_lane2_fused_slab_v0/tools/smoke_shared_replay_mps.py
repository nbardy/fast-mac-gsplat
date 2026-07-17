#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DYNAWORLD = ROOT.parents[3]
WORLD_FOAM_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORLD_FOAM_DIR) not in sys.path:
    sys.path.insert(0, str(WORLD_FOAM_DIR))

from gate0_beam_toy import ToyConfig, default_sites, linspace, make_boundaries, slab_events  # noqa: E402
from gate0_shared_forward_backward import (  # noqa: E402
    DEFAULT_SITE_SIGNALS,
    backward_signal_gradients,
    build_shared_slab_cache,
    gradient_seed,
    make_boundary_lookup,
    render_ray_from_candidates,
    run as run_cpu_reference,
)
from torch_world_foam_lane2_fused_slab import PowerBoundaryConfig, shared_signal_replay  # noqa: E402


def candidate_masks(
    *,
    u_values: list[float],
    boundaries: tuple[Any, ...],
    config: ToyConfig,
) -> tuple[list[int], int]:
    boundary_index = {
        (boundary.left, boundary.right): idx
        for idx, boundary in enumerate(boundaries)
    }
    if len(boundary_index) > 31:
        raise ValueError("Gate 0.6 mask smoke supports at most 31 boundaries")
    masks: list[int] = []
    total_candidate_events = 0
    for u in u_values:
        events, invalid = slab_events(
            boundaries,
            u=u,
            t0=0.0,
            t1=1.0,
            near=config.near,
            far=config.far,
            camera_velocity_x=config.camera_velocity_x,
            invalid_epsilon=config.invalid_epsilon,
        )
        if invalid:
            raise ValueError(f"unexpected invalid denominators for u={u}: {invalid}")
        mask = 0
        for event in events:
            mask |= 1 << boundary_index[event]
        masks.append(mask)
        total_candidate_events += len(events)
    return masks, total_candidate_events


def cpu_outputs_and_grad_samples(
    *,
    frames: int,
    config: ToyConfig,
    u_values: list[float],
) -> tuple[list[list[float]], list[list[list[float]]], float, tuple[float, ...]]:
    sites = default_sites()
    boundaries = make_boundaries(sites)
    boundary_lookup = make_boundary_lookup(boundaries)
    slab_cache, _invalid = build_shared_slab_cache(
        u_values=u_values,
        boundaries=boundaries,
        config=config,
    )
    frame_times = linspace(0.0, 1.0, frames)
    outputs: list[list[float]] = []
    grad_samples: list[list[list[float]]] = []
    tapes = []
    for u_index, u in enumerate(u_values):
        output_row: list[float] = []
        grad_row: list[list[float]] = []
        for t_index, t in enumerate(frame_times):
            slab_index = min(int(math.floor(t * config.time_slabs)), config.time_slabs - 1)
            grad_output = gradient_seed(u_index, t_index)
            tape = render_ray_from_candidates(
                sites=sites,
                boundary_lookup=boundary_lookup,
                candidate_events=slab_cache[(u, slab_index)],
                site_signals=DEFAULT_SITE_SIGNALS,
                u=u,
                t=t,
                near=config.near,
                far=config.far,
                camera_velocity_x=config.camera_velocity_x,
                slab_index=slab_index,
                grad_output=grad_output,
            )
            sample_grad = [0.0 for _ in sites]
            for segment in tape.segments:
                sample_grad[segment.site_id] += grad_output * segment.length
            output_row.append(tape.output)
            grad_row.append(sample_grad)
            tapes.append(tape)
        outputs.append(output_row)
        grad_samples.append(grad_row)
    loss = sum(tape.output * tape.grad_output for tape in tapes)
    grads = backward_signal_gradients(tapes, site_count=len(sites))
    return outputs, grad_samples, loss, grads


def run_frame_count(*, frames: int, config: ToyConfig, timing_iters: int) -> dict[str, Any]:
    sites = default_sites()
    boundaries = make_boundaries(sites)
    u_values = linspace(-1.0, 1.0, config.u_samples)
    frame_times = linspace(0.0, 1.0, frames)
    masks, shared_forward_event_sum = candidate_masks(
        u_values=u_values,
        boundaries=boundaries,
        config=config,
    )

    device = torch.device("mps")
    boundary_f32 = torch.tensor(
        [[boundary.nx, boundary.nz, boundary.nt, boundary.b] for boundary in boundaries],
        dtype=torch.float32,
        device=device,
    )
    candidate_mask_u32 = torch.tensor(masks, dtype=torch.int32, device=device)
    sites_f32 = torch.tensor(
        [[site.x, site.z, site.t, site.weight] for site in sites],
        dtype=torch.float32,
        device=device,
    )
    site_signal_f32 = torch.tensor(DEFAULT_SITE_SIGNALS, dtype=torch.float32, device=device)
    beam_f32 = torch.tensor(
        [[u, 0.0, 1.0, config.near, config.far] for u in u_values],
        dtype=torch.float32,
        device=device,
    )
    frame_t_f32 = torch.tensor(frame_times, dtype=torch.float32, device=device)
    grad_output_f32 = torch.tensor(
        [[gradient_seed(u_index, t_index) for t_index in range(frames)] for u_index in range(len(u_values))],
        dtype=torch.float32,
        device=device,
    )

    mps_output, mps_grad_samples = shared_signal_replay(
        boundary_f32,
        candidate_mask_u32,
        sites_f32,
        site_signal_f32,
        beam_f32,
        frame_t_f32,
        grad_output_f32,
        PowerBoundaryConfig(camera_velocity_x=config.camera_velocity_x, invalid_epsilon=config.invalid_epsilon),
    )
    torch.mps.synchronize()
    started_at = time.perf_counter()
    for _ in range(timing_iters):
        timed_output, timed_grad_samples = shared_signal_replay(
            boundary_f32,
            candidate_mask_u32,
            sites_f32,
            site_signal_f32,
            beam_f32,
            frame_t_f32,
            grad_output_f32,
            PowerBoundaryConfig(camera_velocity_x=config.camera_velocity_x, invalid_epsilon=config.invalid_epsilon),
        )
    torch.mps.synchronize()
    # Keep references live until after synchronize so launch work is not elided.
    _timed_shape = (timed_output.shape, timed_grad_samples.shape)
    replay_wall_clock_ms = (time.perf_counter() - started_at) * 1000.0 / float(timing_iters)

    cpu_output, cpu_grad_samples, cpu_loss, cpu_grad = cpu_outputs_and_grad_samples(
        frames=frames,
        config=config,
        u_values=u_values,
    )
    cpu_output_t = torch.tensor(cpu_output, dtype=torch.float32)
    cpu_grad_sample_t = torch.tensor(cpu_grad_samples, dtype=torch.float32)
    mps_output_cpu = mps_output.cpu()
    mps_grad_samples_cpu = mps_grad_samples.cpu()
    mps_grad = mps_grad_samples_cpu.sum(dim=(0, 1))
    mps_loss = float((mps_output * grad_output_f32).sum().cpu().item())

    cpu_payload = run_cpu_reference(
        ToyConfig(
            frame_counts=(frames,),
            u_samples=config.u_samples,
            time_slabs=config.time_slabs,
            near=config.near,
            far=config.far,
            camera_velocity_x=config.camera_velocity_x,
            invalid_epsilon=config.invalid_epsilon,
        )
    )
    cpu_row = cpu_payload["rows"][0]
    return {
        "frames": frames,
        "beam_count": len(u_values),
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "shared_forward_boundary_scans": int(cpu_row["shared_forward_boundary_scans"]),
        "shared_backward_boundary_scans": int(cpu_row["shared_backward_boundary_scans"]),
        "direct_forward_boundary_scans": int(cpu_row["direct_forward_boundary_scans"]),
        "direct_backward_boundary_scans": int(cpu_row["direct_backward_boundary_scans"]),
        "shared_forward_backward_boundary_scan_ratio": float(cpu_row["shared_forward_backward_boundary_scan_ratio"]),
        "shared_forward_event_sum": shared_forward_event_sum,
        "mps_loss": mps_loss,
        "cpu_loss": cpu_loss,
        "loss_abs_error": abs(mps_loss - cpu_loss),
        "mps_shared_replay_wall_clock_ms": replay_wall_clock_ms,
        "timing_iters": timing_iters,
        "max_output_abs_error": float((mps_output_cpu - cpu_output_t).abs().max().item()),
        "signal_gradient_max_abs_error": float((mps_grad - torch.tensor(cpu_grad, dtype=torch.float32)).abs().max().item()),
        "grad_sample_max_abs_error": float((mps_grad_samples_cpu - cpu_grad_sample_t).abs().max().item()),
    }


def run_smoke(*, timing_iters: int) -> dict[str, Any]:
    if timing_iters <= 0:
        raise ValueError("timing_iters must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    config = ToyConfig(
        frame_counts=(2, 4, 8, 16),
        u_samples=17,
        time_slabs=1,
        near=0.25,
        far=3.0,
        camera_velocity_x=0.35,
        invalid_epsilon=1.0e-7,
    )
    rows = [run_frame_count(frames=frames, config=config, timing_iters=timing_iters) for frames in config.frame_counts]
    tolerance = 2.0e-5
    acceptance = {
        "outputs_match_cpu_reference": all(row["max_output_abs_error"] <= tolerance for row in rows),
        "grad_samples_match_cpu_reference": all(row["grad_sample_max_abs_error"] <= tolerance for row in rows),
        "signal_gradients_match_cpu_reference": all(row["signal_gradient_max_abs_error"] <= tolerance for row in rows),
        "loss_matches_cpu_reference": all(row["loss_abs_error"] <= tolerance for row in rows),
        "shared_forward_backward_scans_sublinear": all(
            row["shared_forward_backward_boundary_scan_ratio"] < 1.0 for row in rows
        ),
    }
    return {
        "benchmark": "world_foam_lane2_gate0_6_mps_shared_replay_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "device": "mps",
        "gate": "0.6",
        "gradient_scope": "mps_site_signal_only_fixed_segments_geometry_gradients_not_implemented",
        "tolerance": tolerance,
        "timing_iters": timing_iters,
        "acceptance": acceptance,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke the World Foam Lane 2 MPS shared replay op.")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--timing-iters", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke(timing_iters=args.timing_iters)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
