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

from gate4_moving_ray_slab_compiler import (  # noqa: E402
    DEFAULT_CONFIG,
    SyntheticRayMotion,
    _load_config,
    apply_synthetic_ray_motion,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)
from gate4_affine_slab_tape import build_gate4_affine_slab_tape  # noqa: E402
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    MAX_REALRAY_BOUNDARIES,
    RealRayReplayConfig,
    fused_slab_affine_coeff16_realray_rgba_depth_replay,
    fused_slab_affine_coeff_realray_rgba_depth_replay,
    fused_slab_affine_num32_den16_autograd,
    fused_slab_affine_num32_den16_realray_rgba_depth_replay,
    fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay,
    fused_slab_affine_num32_den16_vjp_direct_atomic,
    fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only,
    fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate,
    fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only,
    fused_slab_affine_num32_den16_vjp_direct_atomic_track,
    fused_slab_affine_num32_den16_vjp_reduce,
    fused_slab_affine_realray_rgba_depth_replay,
    realray_rgba_depth_replay,
)


RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"
VJP_SEED_MODES = ("rgb", "rgba-depth")


def _parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one frame count")
    return out


def _timed_mps_call(fn: Any, *, timing_iters: int) -> tuple[tuple[torch.Tensor, ...], float]:
    out = fn()
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(timing_iters):
        out = fn()
    torch.mps.synchronize()
    return out, (time.perf_counter() - start) * 1000.0 / float(timing_iters)


def _make_vjp_seed_tensors(
    *,
    mode: str,
    track_count: int,
    frame_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if mode not in VJP_SEED_MODES:
        raise ValueError(f"vjp seed mode must be one of {VJP_SEED_MODES}, got {mode!r}")
    if mode == "rgb":
        grad_rgb = torch.ones((track_count, frame_count, 3), dtype=torch.float32)
        grad_alpha = torch.zeros((track_count, frame_count), dtype=torch.float32)
        grad_depth = torch.zeros_like(grad_alpha)
    else:
        base = torch.arange(track_count * frame_count, dtype=torch.float32).reshape(track_count, frame_count)
        grad_alpha = 0.125 + torch.remainder(base, 7.0) * 0.025
        grad_depth = 0.05 + torch.remainder(base, 11.0) * 0.01
        grad_rgb = torch.stack(
            (
                0.75 + torch.remainder(base, 5.0) * 0.05,
                0.50 + torch.remainder(base, 3.0) * 0.04,
                0.25 + torch.remainder(base, 13.0) * 0.015,
            ),
            dim=-1,
        )
    summary = {
        "mode": mode,
        "rgb_abs_sum": float(grad_rgb.abs().sum().item()),
        "alpha_abs_sum": float(grad_alpha.abs().sum().item()),
        "depth_abs_sum": float(grad_depth.abs().sum().item()),
        "rgb_abs_max": float(grad_rgb.abs().max().item()),
        "alpha_abs_max": float(grad_alpha.abs().max().item()),
        "depth_abs_max": float(grad_depth.abs().max().item()),
    }
    return (
        grad_rgb.to(device=device).contiguous(),
        grad_alpha.to(device=device).contiguous(),
        grad_depth.to(device=device).contiguous(),
        summary,
    )


def _autograd_site_rgba_grad(
    *,
    vjp_mode: str,
    row_index_i32: torch.Tensor,
    row_offsets_i32: torch.Tensor,
    candidate_depth_num_f32: torch.Tensor,
    candidate_depth_den_f16: torch.Tensor,
    sites_f32: torch.Tensor,
    site_rgba_f32: torch.Tensor,
    ray_coeff_f32: torch.Tensor,
    frame_t_f32: torch.Tensor,
    grad_rgb_f32: torch.Tensor,
    grad_alpha_f32: torch.Tensor,
    grad_depth_f32: torch.Tensor,
    op_config: RealRayReplayConfig,
    time_slabs: int,
    row_count: int,
    reduce_chunk_size: int,
) -> torch.Tensor:
    site_rgba_leaf = site_rgba_f32.detach().clone().requires_grad_(True)
    rgb, alpha, depth = fused_slab_affine_num32_den16_autograd(
        row_index_i32,
        row_offsets_i32,
        candidate_depth_num_f32,
        candidate_depth_den_f16,
        sites_f32,
        site_rgba_leaf,
        ray_coeff_f32,
        frame_t_f32,
        op_config,
        time_slab_count=time_slabs,
        row_count=row_count,
        reduce_chunk_size=reduce_chunk_size,
        vjp_mode=vjp_mode,
    )
    loss = (rgb * grad_rgb_f32).sum() + (alpha * grad_alpha_f32).sum() + (depth * grad_depth_f32).sum()
    loss.backward()
    torch.mps.synchronize()
    grad = site_rgba_leaf.grad
    if grad is None:
        raise RuntimeError(f"autograd did not produce site_rgba grad for vjp_mode={vjp_mode!r}")
    return grad.detach().cpu()


def _storage_bytes(*tensors: torch.Tensor) -> int:
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))


def _build_affine_csr_bundle(
    *,
    boundaries: tuple[Any, ...],
    rays: torch.Tensor,
    frame_indices: torch.Tensor,
    frame_count: int,
    time_slabs: int,
    near: float,
    far: float,
    invalid_epsilon: float,
    residual_depth_padding: float,
    layout: str,
    tile_h: int,
    tile_w: int,
    candidate_order: str,
) -> dict[str, Any]:
    return build_gate4_affine_slab_tape(
        boundaries=boundaries,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        residual_depth_padding=residual_depth_padding,
        layout=layout,
        tile_h=tile_h,
        tile_w=tile_w,
        candidate_order=candidate_order,
    ).to_legacy_bundle()


def _profile_frame_count(
    *,
    frame_count: int,
    config_path: Path,
    render_size: int,
    site_count: int,
    time_slabs: int,
    layout: str,
    tile_h: int,
    tile_w: int,
    candidate_order: str,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    residual_depth_padding: float,
    synthetic_motion: SyntheticRayMotion,
    include_ownerupdate: bool,
    include_vjp: bool,
    vjp_seed_mode: str,
    vjp_reduce_chunk_size: int,
    timing_iters: int,
) -> dict[str, Any]:
    cfg = _load_config(config_path, max_frames=frame_count, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    loaded_frame_count = int(data["frame_count"])
    if loaded_frame_count != frame_count:
        raise ValueError(f"requested {frame_count} frames but loader returned {loaded_frame_count}")
    rays = apply_synthetic_ray_motion(
        rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        motion=synthetic_motion,
    )
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
    )
    boundaries = make_boundaries_4d(sites)
    tape = build_gate4_affine_slab_tape(
        boundaries=boundaries,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        residual_depth_padding=residual_depth_padding,
        layout=layout,
        tile_h=tile_h,
        tile_w=tile_w,
        candidate_order=candidate_order,
    )
    bundle = tape.to_legacy_bundle()

    device = torch.device("mps")
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    boundary_f32 = torch.tensor(
        [[boundary.nx, boundary.ny, boundary.nz, boundary.nt, boundary.b] for boundary in boundaries],
        dtype=torch.float32,
        device=device,
    )
    boundary_site_pairs_i32 = torch.tensor(
        [[boundary.left, boundary.right] for boundary in boundaries],
        dtype=torch.int32,
        device=device,
    )
    sites_f32 = torch.tensor(
        [[site.x, site.y, site.z, site.t, site.weight] for site in sites],
        dtype=torch.float32,
        device=device,
    )
    site_rgba_f32 = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)
    frame_t_f32 = bundle["frame_t"].to(device)
    frame_t_flat_f32 = frame_t_f32.repeat(int(bundle["track_count"])).contiguous()
    explicit_rays_f32 = bundle["explicit_rays"].to(device)
    ray_coeff_f32 = bundle["ray_coeff"].to(device)
    row_index_i32 = bundle["row_index"].to(device)
    row_offsets_i32 = bundle["row_offsets"].to(device)
    candidate_ids_i32 = bundle["candidate_ids"].to(device)
    candidate_depth_coeff_f32 = bundle["candidate_depth_coeffs"].to(device)
    candidate_depth_coeff_f16 = bundle["candidate_depth_coeffs"].to(device=device, dtype=torch.float16)
    candidate_depth_num_f32 = bundle["candidate_depth_coeffs"][:, :2].contiguous().to(device)
    candidate_depth_den_f16 = bundle["candidate_depth_coeffs"][:, 2:].contiguous().to(device=device, dtype=torch.float16)

    direct, direct_ms = _timed_mps_call(
        lambda: realray_rgba_depth_replay(
            boundary_f32,
            sites_f32,
            site_rgba_f32,
            explicit_rays_f32,
            frame_t_flat_f32,
            op_config,
        ),
        timing_iters=timing_iters,
    )
    fused, fused_ms = _timed_mps_call(
        lambda: fused_slab_affine_realray_rgba_depth_replay(
            boundary_f32,
            row_index_i32,
            row_offsets_i32,
            candidate_ids_i32,
            sites_f32,
            site_rgba_f32,
            ray_coeff_f32,
            frame_t_f32,
            op_config,
            time_slab_count=time_slabs,
            row_count=int(bundle["row_count"]),
        ),
        timing_iters=timing_iters,
    )
    coeff: tuple[torch.Tensor, ...] | None = None
    coeff_ms: float | None = None
    mixed: tuple[torch.Tensor, ...] | None = None
    mixed_ms: float | None = None
    ownerupdate: tuple[torch.Tensor, ...] | None = None
    ownerupdate_ms: float | None = None
    mixed_vjp: tuple[torch.Tensor, ...] | None = None
    mixed_vjp_ms: float | None = None
    mixed_vjp_direct: tuple[torch.Tensor, ...] | None = None
    mixed_vjp_direct_ms: float | None = None
    mixed_vjp_direct_grad_only: torch.Tensor | None = None
    mixed_vjp_direct_grad_only_ms: float | None = None
    mixed_vjp_direct_grad_only_ownerupdate: torch.Tensor | None = None
    mixed_vjp_direct_grad_only_ownerupdate_ms: float | None = None
    mixed_vjp_direct_rgb_only: torch.Tensor | None = None
    mixed_vjp_direct_rgb_only_ms: float | None = None
    mixed_vjp_direct_track: torch.Tensor | None = None
    mixed_vjp_direct_track_ms: float | None = None
    autograd_vjp_grads: dict[str, torch.Tensor] = {}
    coeff16: tuple[torch.Tensor, ...] | None = None
    coeff16_ms: float | None = None
    vjp_seed_summary: dict[str, Any] | None = None
    if layout == "per-track":
        coeff, coeff_ms = _timed_mps_call(
            lambda: fused_slab_affine_coeff_realray_rgba_depth_replay(
                row_index_i32,
                row_offsets_i32,
                candidate_depth_coeff_f32,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                op_config,
                time_slab_count=time_slabs,
                row_count=int(bundle["row_count"]),
            ),
            timing_iters=timing_iters,
        )
        mixed, mixed_ms = _timed_mps_call(
            lambda: fused_slab_affine_num32_den16_realray_rgba_depth_replay(
                row_index_i32,
                row_offsets_i32,
                candidate_depth_num_f32,
                candidate_depth_den_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                op_config,
                time_slab_count=time_slabs,
                row_count=int(bundle["row_count"]),
            ),
            timing_iters=timing_iters,
        )
        if include_ownerupdate:
            ownerupdate, ownerupdate_ms = _timed_mps_call(
                lambda: fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay(
                    row_index_i32,
                    row_offsets_i32,
                    candidate_ids_i32,
                    candidate_depth_num_f32,
                    candidate_depth_den_f16,
                    boundary_site_pairs_i32,
                    sites_f32,
                    site_rgba_f32,
                    ray_coeff_f32,
                    frame_t_f32,
                    op_config,
                    time_slab_count=time_slabs,
                    row_count=int(bundle["row_count"]),
                ),
                timing_iters=timing_iters,
            )
        if include_vjp:
            grad_rgb_f32, grad_alpha_f32, grad_depth_f32, vjp_seed_summary = _make_vjp_seed_tensors(
                mode=vjp_seed_mode,
                track_count=int(bundle["track_count"]),
                frame_count=frame_count,
                device=device,
            )
            mixed_vjp, mixed_vjp_ms = _timed_mps_call(
                lambda: fused_slab_affine_num32_den16_vjp_reduce(
                    row_index_i32,
                    row_offsets_i32,
                    candidate_depth_num_f32,
                    candidate_depth_den_f16,
                    sites_f32,
                    site_rgba_f32,
                    ray_coeff_f32,
                    frame_t_f32,
                    grad_rgb_f32,
                    grad_alpha_f32,
                    grad_depth_f32,
                    op_config,
                    time_slab_count=time_slabs,
                    row_count=int(bundle["row_count"]),
                    reduce_chunk_size=vjp_reduce_chunk_size,
                ),
                timing_iters=timing_iters,
            )
            mixed_vjp_direct, mixed_vjp_direct_ms = _timed_mps_call(
                lambda: fused_slab_affine_num32_den16_vjp_direct_atomic(
                    row_index_i32,
                    row_offsets_i32,
                    candidate_depth_num_f32,
                    candidate_depth_den_f16,
                    sites_f32,
                    site_rgba_f32,
                    ray_coeff_f32,
                    frame_t_f32,
                    grad_rgb_f32,
                    grad_alpha_f32,
                    grad_depth_f32,
                    op_config,
                    time_slab_count=time_slabs,
                    row_count=int(bundle["row_count"]),
                ),
                timing_iters=timing_iters,
            )
            mixed_vjp_direct_grad_only_out, mixed_vjp_direct_grad_only_ms = _timed_mps_call(
                lambda: (
                    fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only(
                        row_index_i32,
                        row_offsets_i32,
                        candidate_depth_num_f32,
                        candidate_depth_den_f16,
                        sites_f32,
                        site_rgba_f32,
                        ray_coeff_f32,
                        frame_t_f32,
                        grad_rgb_f32,
                        grad_alpha_f32,
                        grad_depth_f32,
                        op_config,
                        time_slab_count=time_slabs,
                        row_count=int(bundle["row_count"]),
                    ),
                ),
                timing_iters=timing_iters,
            )
            mixed_vjp_direct_grad_only = mixed_vjp_direct_grad_only_out[0]
            if include_ownerupdate:
                (
                    mixed_vjp_direct_grad_only_ownerupdate_out,
                    mixed_vjp_direct_grad_only_ownerupdate_ms,
                ) = _timed_mps_call(
                    lambda: (
                        fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate(
                            row_index_i32,
                            row_offsets_i32,
                            candidate_ids_i32,
                            candidate_depth_num_f32,
                            candidate_depth_den_f16,
                            boundary_site_pairs_i32,
                            sites_f32,
                            site_rgba_f32,
                            ray_coeff_f32,
                            frame_t_f32,
                            grad_rgb_f32,
                            grad_alpha_f32,
                            grad_depth_f32,
                            op_config,
                            time_slab_count=time_slabs,
                            row_count=int(bundle["row_count"]),
                        ),
                    ),
                    timing_iters=timing_iters,
                )
                mixed_vjp_direct_grad_only_ownerupdate = mixed_vjp_direct_grad_only_ownerupdate_out[0]
            mixed_vjp_direct_rgb_only_out, mixed_vjp_direct_rgb_only_ms = _timed_mps_call(
                lambda: (
                    fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only(
                        row_index_i32,
                        row_offsets_i32,
                        candidate_depth_num_f32,
                        candidate_depth_den_f16,
                        sites_f32,
                        site_rgba_f32,
                        ray_coeff_f32,
                        frame_t_f32,
                        grad_rgb_f32,
                        op_config,
                        time_slab_count=time_slabs,
                        row_count=int(bundle["row_count"]),
                    ),
                ),
                timing_iters=timing_iters,
            )
            mixed_vjp_direct_rgb_only = mixed_vjp_direct_rgb_only_out[0]
            mixed_vjp_direct_track_out, mixed_vjp_direct_track_ms = _timed_mps_call(
                lambda: (
                    fused_slab_affine_num32_den16_vjp_direct_atomic_track(
                        row_index_i32,
                        row_offsets_i32,
                        candidate_depth_num_f32,
                        candidate_depth_den_f16,
                        sites_f32,
                        site_rgba_f32,
                        ray_coeff_f32,
                        frame_t_f32,
                        grad_rgb_f32,
                        grad_alpha_f32,
                        grad_depth_f32,
                        op_config,
                        time_slab_count=time_slabs,
                        row_count=int(bundle["row_count"]),
                    ),
                ),
                timing_iters=timing_iters,
            )
            mixed_vjp_direct_track = mixed_vjp_direct_track_out[0]
            for autograd_mode in (
                "reduce",
                "direct_atomic",
                "direct_atomic_grad_only",
                "direct_atomic_rgb_only",
                "direct_atomic_track",
            ):
                autograd_vjp_grads[autograd_mode] = _autograd_site_rgba_grad(
                    vjp_mode=autograd_mode,
                    row_index_i32=row_index_i32,
                    row_offsets_i32=row_offsets_i32,
                    candidate_depth_num_f32=candidate_depth_num_f32,
                    candidate_depth_den_f16=candidate_depth_den_f16,
                    sites_f32=sites_f32,
                    site_rgba_f32=site_rgba_f32,
                    ray_coeff_f32=ray_coeff_f32,
                    frame_t_f32=frame_t_f32,
                    grad_rgb_f32=grad_rgb_f32,
                    grad_alpha_f32=grad_alpha_f32,
                    grad_depth_f32=grad_depth_f32,
                    op_config=op_config,
                    time_slabs=time_slabs,
                    row_count=int(bundle["row_count"]),
                    reduce_chunk_size=vjp_reduce_chunk_size,
                )
        coeff16, coeff16_ms = _timed_mps_call(
            lambda: fused_slab_affine_coeff16_realray_rgba_depth_replay(
                row_index_i32,
                row_offsets_i32,
                candidate_depth_coeff_f16,
                sites_f32,
                site_rgba_f32,
                ray_coeff_f32,
                frame_t_f32,
                op_config,
                time_slab_count=time_slabs,
                row_count=int(bundle["row_count"]),
            ),
            timing_iters=timing_iters,
        )
    direct_rgb, direct_alpha, direct_depth = (tensor.detach().cpu() for tensor in direct)
    fused_rgb, fused_alpha, fused_depth = (tensor.detach().cpu() for tensor in fused)
    if coeff is not None:
        coeff_rgb, coeff_alpha, coeff_depth = (tensor.detach().cpu() for tensor in coeff)
    else:
        coeff_rgb = coeff_alpha = coeff_depth = None
    if mixed is not None:
        mixed_rgb, mixed_alpha, mixed_depth = (tensor.detach().cpu() for tensor in mixed)
    else:
        mixed_rgb = mixed_alpha = mixed_depth = None
    if ownerupdate is not None:
        ownerupdate_rgb, ownerupdate_alpha, ownerupdate_depth = (tensor.detach().cpu() for tensor in ownerupdate)
    else:
        ownerupdate_rgb = ownerupdate_alpha = ownerupdate_depth = None
    if mixed_vjp is not None:
        mixed_vjp_rgb, mixed_vjp_alpha, mixed_vjp_depth, mixed_vjp_grad = (
            tensor.detach().cpu() for tensor in mixed_vjp
        )
    else:
        mixed_vjp_rgb = mixed_vjp_alpha = mixed_vjp_depth = mixed_vjp_grad = None
    if mixed_vjp_direct is not None:
        mixed_vjp_direct_rgb, mixed_vjp_direct_alpha, mixed_vjp_direct_depth, mixed_vjp_direct_grad = (
            tensor.detach().cpu() for tensor in mixed_vjp_direct
        )
    else:
        mixed_vjp_direct_rgb = mixed_vjp_direct_alpha = mixed_vjp_direct_depth = mixed_vjp_direct_grad = None
    mixed_vjp_direct_grad_only_cpu = (
        mixed_vjp_direct_grad_only.detach().cpu() if mixed_vjp_direct_grad_only is not None else None
    )
    mixed_vjp_direct_grad_only_ownerupdate_cpu = (
        mixed_vjp_direct_grad_only_ownerupdate.detach().cpu()
        if mixed_vjp_direct_grad_only_ownerupdate is not None
        else None
    )
    mixed_vjp_direct_rgb_only_cpu = (
        mixed_vjp_direct_rgb_only.detach().cpu() if mixed_vjp_direct_rgb_only is not None else None
    )
    mixed_vjp_direct_track_cpu = (
        mixed_vjp_direct_track.detach().cpu() if mixed_vjp_direct_track is not None else None
    )
    if coeff16 is not None:
        coeff16_rgb, coeff16_alpha, coeff16_depth = (tensor.detach().cpu() for tensor in coeff16)
    else:
        coeff16_rgb = coeff16_alpha = coeff16_depth = None
    track_count = int(bundle["track_count"])
    direct_rgb = direct_rgb.reshape(track_count, frame_count, 3)
    direct_alpha = direct_alpha.reshape(track_count, frame_count)
    direct_depth = direct_depth.reshape(track_count, frame_count)
    if mixed_vjp_rgb is not None and mixed_vjp_alpha is not None and mixed_vjp_depth is not None:
        mixed_vjp_rgb = mixed_vjp_rgb.reshape(track_count, frame_count, 3)
        mixed_vjp_alpha = mixed_vjp_alpha.reshape(track_count, frame_count)
        mixed_vjp_depth = mixed_vjp_depth.reshape(track_count, frame_count)
    if (
        mixed_vjp_direct_rgb is not None
        and mixed_vjp_direct_alpha is not None
        and mixed_vjp_direct_depth is not None
    ):
        mixed_vjp_direct_rgb = mixed_vjp_direct_rgb.reshape(track_count, frame_count, 3)
        mixed_vjp_direct_alpha = mixed_vjp_direct_alpha.reshape(track_count, frame_count)
        mixed_vjp_direct_depth = mixed_vjp_direct_depth.reshape(track_count, frame_count)
    coeff_errors = None
    coeff_timing = None
    if coeff_rgb is not None and coeff_alpha is not None and coeff_depth is not None and coeff_ms is not None:
        coeff_errors = {
            "rgb": float((coeff_rgb - direct_rgb).abs().max().item()),
            "alpha": float((coeff_alpha - direct_alpha).abs().max().item()),
            "depth": float((coeff_depth - direct_depth).abs().max().item()),
            "rgb_vs_id_csr": float((coeff_rgb - fused_rgb).abs().max().item()),
            "alpha_vs_id_csr": float((coeff_alpha - fused_alpha).abs().max().item()),
            "depth_vs_id_csr": float((coeff_depth - fused_depth).abs().max().item()),
        }
        coeff_timing = {
            "fused_slab_affine_coeff_csr": float(coeff_ms),
            "coeff_speedup_vs_explicit": float(direct_ms) / float(max(coeff_ms, 1.0e-9)),
            "coeff_speedup_vs_id_csr": float(fused_ms) / float(max(coeff_ms, 1.0e-9)),
        }
    mixed_errors = None
    mixed_timing = None
    if (
        mixed_rgb is not None
        and mixed_alpha is not None
        and mixed_depth is not None
        and mixed_ms is not None
    ):
        mixed_errors = {
            "rgb": float((mixed_rgb - direct_rgb).abs().max().item()),
            "alpha": float((mixed_alpha - direct_alpha).abs().max().item()),
            "depth": float((mixed_depth - direct_depth).abs().max().item()),
            "rgb_vs_id_csr": float((mixed_rgb - fused_rgb).abs().max().item()),
            "alpha_vs_id_csr": float((mixed_alpha - fused_alpha).abs().max().item()),
            "depth_vs_id_csr": float((mixed_depth - fused_depth).abs().max().item()),
            "rgb_vs_coeff32": float((mixed_rgb - coeff_rgb).abs().max().item()) if coeff_rgb is not None else 0.0,
            "alpha_vs_coeff32": float((mixed_alpha - coeff_alpha).abs().max().item())
            if coeff_alpha is not None
            else 0.0,
            "depth_vs_coeff32": float((mixed_depth - coeff_depth).abs().max().item())
            if coeff_depth is not None
            else 0.0,
        }
        mixed_timing = {
            "fused_slab_affine_num32_den16_csr": float(mixed_ms),
            "mixed_speedup_vs_explicit": float(direct_ms) / float(max(mixed_ms, 1.0e-9)),
            "mixed_speedup_vs_id_csr": float(fused_ms) / float(max(mixed_ms, 1.0e-9)),
            "mixed_speedup_vs_coeff32": float(coeff_ms or 0.0) / float(max(mixed_ms, 1.0e-9)),
        }
    ownerupdate_errors = None
    ownerupdate_timing = None
    if (
        ownerupdate_rgb is not None
        and ownerupdate_alpha is not None
        and ownerupdate_depth is not None
        and ownerupdate_ms is not None
    ):
        ownerupdate_errors = {
            "rgb": float((ownerupdate_rgb - direct_rgb).abs().max().item()),
            "alpha": float((ownerupdate_alpha - direct_alpha).abs().max().item()),
            "depth": float((ownerupdate_depth - direct_depth).abs().max().item()),
            "rgb_vs_mixed": float((ownerupdate_rgb - mixed_rgb).abs().max().item()) if mixed_rgb is not None else 0.0,
            "alpha_vs_mixed": float((ownerupdate_alpha - mixed_alpha).abs().max().item())
            if mixed_alpha is not None
            else 0.0,
            "depth_vs_mixed": float((ownerupdate_depth - mixed_depth).abs().max().item())
            if mixed_depth is not None
            else 0.0,
        }
        ownerupdate_timing = {
            "fused_slab_affine_num32_den16_ownerupdate_csr": float(ownerupdate_ms),
            "ownerupdate_speedup_vs_explicit": float(direct_ms) / float(max(ownerupdate_ms, 1.0e-9)),
            "ownerupdate_speedup_vs_mixed": float(mixed_ms or 0.0) / float(max(ownerupdate_ms, 1.0e-9)),
            "ownerupdate_speedup_vs_coeff32": float(coeff_ms or 0.0) / float(max(ownerupdate_ms, 1.0e-9)),
        }
    mixed_vjp_errors = None
    mixed_vjp_timing = None
    mixed_vjp_grad_summary = None
    mixed_vjp_direct_errors = None
    mixed_vjp_direct_timing = None
    mixed_vjp_direct_grad_summary = None
    mixed_vjp_direct_grad_only_errors = None
    mixed_vjp_direct_grad_only_timing = None
    mixed_vjp_direct_grad_only_summary = None
    mixed_vjp_direct_grad_only_ownerupdate_errors = None
    mixed_vjp_direct_grad_only_ownerupdate_timing = None
    mixed_vjp_direct_grad_only_ownerupdate_summary = None
    mixed_vjp_direct_rgb_only_errors = None
    mixed_vjp_direct_rgb_only_timing = None
    mixed_vjp_direct_rgb_only_summary = None
    mixed_vjp_direct_track_errors = None
    mixed_vjp_direct_track_timing = None
    mixed_vjp_direct_track_summary = None
    autograd_vjp_errors = None
    autograd_vjp_summaries = None
    if (
        mixed_vjp_rgb is not None
        and mixed_vjp_alpha is not None
        and mixed_vjp_depth is not None
        and mixed_vjp_grad is not None
        and mixed_vjp_ms is not None
    ):
        mixed_vjp_errors = {
            "rgb": float((mixed_vjp_rgb - direct_rgb).abs().max().item()),
            "alpha": float((mixed_vjp_alpha - direct_alpha).abs().max().item()),
            "depth": float((mixed_vjp_depth - direct_depth).abs().max().item()),
            "rgb_vs_mixed": float((mixed_vjp_rgb - mixed_rgb).abs().max().item()) if mixed_rgb is not None else 0.0,
            "alpha_vs_mixed": float((mixed_vjp_alpha - mixed_alpha).abs().max().item())
            if mixed_alpha is not None
            else 0.0,
            "depth_vs_mixed": float((mixed_vjp_depth - mixed_depth).abs().max().item())
            if mixed_depth is not None
            else 0.0,
        }
        mixed_vjp_timing = {
            "fused_slab_affine_num32_den16_vjp_reduce": float(mixed_vjp_ms),
            "mixed_vjp_speedup_vs_explicit": float(direct_ms) / float(max(mixed_vjp_ms, 1.0e-9)),
            "mixed_vjp_speedup_vs_mixed_forward": float(mixed_ms or 0.0) / float(max(mixed_vjp_ms, 1.0e-9)),
        }
        mixed_vjp_grad_summary = {
            "abs_sum": float(mixed_vjp_grad.abs().sum().item()),
            "abs_max": float(mixed_vjp_grad.abs().max().item()),
            "finite": bool(torch.isfinite(mixed_vjp_grad).all().item()),
        }
    if (
        mixed_vjp_direct_rgb is not None
        and mixed_vjp_direct_alpha is not None
        and mixed_vjp_direct_depth is not None
        and mixed_vjp_direct_grad is not None
        and mixed_vjp_direct_ms is not None
    ):
        mixed_vjp_direct_errors = {
            "rgb": float((mixed_vjp_direct_rgb - direct_rgb).abs().max().item()),
            "alpha": float((mixed_vjp_direct_alpha - direct_alpha).abs().max().item()),
            "depth": float((mixed_vjp_direct_depth - direct_depth).abs().max().item()),
            "rgb_vs_mixed": float((mixed_vjp_direct_rgb - mixed_rgb).abs().max().item())
            if mixed_rgb is not None
            else 0.0,
            "alpha_vs_mixed": float((mixed_vjp_direct_alpha - mixed_alpha).abs().max().item())
            if mixed_alpha is not None
            else 0.0,
            "depth_vs_mixed": float((mixed_vjp_direct_depth - mixed_depth).abs().max().item())
            if mixed_depth is not None
            else 0.0,
            "grad_vs_reduce": float((mixed_vjp_direct_grad - mixed_vjp_grad).abs().max().item())
            if mixed_vjp_grad is not None
            else 0.0,
            "grad_rel_vs_reduce": float(
                (mixed_vjp_direct_grad - mixed_vjp_grad).abs().max().item()
                / max(float(mixed_vjp_grad.abs().max().item()), 1.0e-9)
            )
            if mixed_vjp_grad is not None
            else 0.0,
        }
        mixed_vjp_direct_timing = {
            "fused_slab_affine_num32_den16_vjp_direct_atomic": float(mixed_vjp_direct_ms),
            "mixed_vjp_direct_speedup_vs_reduce": float(mixed_vjp_ms or 0.0)
            / float(max(mixed_vjp_direct_ms, 1.0e-9)),
            "mixed_vjp_direct_speedup_vs_explicit": float(direct_ms) / float(max(mixed_vjp_direct_ms, 1.0e-9)),
            "mixed_vjp_direct_speedup_vs_mixed_forward": float(mixed_ms or 0.0)
            / float(max(mixed_vjp_direct_ms, 1.0e-9)),
        }
        mixed_vjp_direct_grad_summary = {
            "abs_sum": float(mixed_vjp_direct_grad.abs().sum().item()),
            "abs_max": float(mixed_vjp_direct_grad.abs().max().item()),
            "finite": bool(torch.isfinite(mixed_vjp_direct_grad).all().item()),
        }
    if mixed_vjp_direct_grad_only_cpu is not None and mixed_vjp_direct_grad_only_ms is not None:
        mixed_vjp_direct_grad_only_errors = {
            "grad_vs_reduce": float((mixed_vjp_direct_grad_only_cpu - mixed_vjp_grad).abs().max().item())
            if mixed_vjp_grad is not None
            else 0.0,
            "grad_rel_vs_reduce": float(
                (mixed_vjp_direct_grad_only_cpu - mixed_vjp_grad).abs().max().item()
                / max(float(mixed_vjp_grad.abs().max().item()), 1.0e-9)
            )
            if mixed_vjp_grad is not None
            else 0.0,
            "grad_vs_direct_atomic": float(
                (mixed_vjp_direct_grad_only_cpu - mixed_vjp_direct_grad).abs().max().item()
            )
            if mixed_vjp_direct_grad is not None
            else 0.0,
        }
        mixed_vjp_direct_grad_only_timing = {
            "fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only": float(mixed_vjp_direct_grad_only_ms),
            "mixed_vjp_direct_grad_only_speedup_vs_reduce": float(mixed_vjp_ms or 0.0)
            / float(max(mixed_vjp_direct_grad_only_ms, 1.0e-9)),
            "mixed_vjp_direct_grad_only_speedup_vs_direct_atomic": float(mixed_vjp_direct_ms or 0.0)
            / float(max(mixed_vjp_direct_grad_only_ms, 1.0e-9)),
            "mixed_vjp_direct_grad_only_speedup_vs_mixed_forward": float(mixed_ms or 0.0)
            / float(max(mixed_vjp_direct_grad_only_ms, 1.0e-9)),
        }
        mixed_vjp_direct_grad_only_summary = {
            "abs_sum": float(mixed_vjp_direct_grad_only_cpu.abs().sum().item()),
            "abs_max": float(mixed_vjp_direct_grad_only_cpu.abs().max().item()),
            "finite": bool(torch.isfinite(mixed_vjp_direct_grad_only_cpu).all().item()),
        }
    if (
        mixed_vjp_direct_grad_only_ownerupdate_cpu is not None
        and mixed_vjp_direct_grad_only_ownerupdate_ms is not None
    ):
        mixed_vjp_direct_grad_only_ownerupdate_errors = {
            "grad_vs_reduce": float((mixed_vjp_direct_grad_only_ownerupdate_cpu - mixed_vjp_grad).abs().max().item())
            if mixed_vjp_grad is not None
            else 0.0,
            "grad_rel_vs_reduce": float(
                (mixed_vjp_direct_grad_only_ownerupdate_cpu - mixed_vjp_grad).abs().max().item()
                / max(float(mixed_vjp_grad.abs().max().item()), 1.0e-9)
            )
            if mixed_vjp_grad is not None
            else 0.0,
            "grad_vs_grad_only": float(
                (mixed_vjp_direct_grad_only_ownerupdate_cpu - mixed_vjp_direct_grad_only_cpu).abs().max().item()
            )
            if mixed_vjp_direct_grad_only_cpu is not None
            else 0.0,
        }
        mixed_vjp_direct_grad_only_ownerupdate_timing = {
            "fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate": float(
                mixed_vjp_direct_grad_only_ownerupdate_ms
            ),
            "mixed_vjp_direct_grad_only_ownerupdate_speedup_vs_reduce": float(mixed_vjp_ms or 0.0)
            / float(max(mixed_vjp_direct_grad_only_ownerupdate_ms, 1.0e-9)),
            "mixed_vjp_direct_grad_only_ownerupdate_speedup_vs_grad_only": float(
                mixed_vjp_direct_grad_only_ms or 0.0
            )
            / float(max(mixed_vjp_direct_grad_only_ownerupdate_ms, 1.0e-9)),
            "mixed_vjp_direct_grad_only_ownerupdate_speedup_vs_mixed_forward": float(mixed_ms or 0.0)
            / float(max(mixed_vjp_direct_grad_only_ownerupdate_ms, 1.0e-9)),
        }
        mixed_vjp_direct_grad_only_ownerupdate_summary = {
            "abs_sum": float(mixed_vjp_direct_grad_only_ownerupdate_cpu.abs().sum().item()),
            "abs_max": float(mixed_vjp_direct_grad_only_ownerupdate_cpu.abs().max().item()),
            "finite": bool(torch.isfinite(mixed_vjp_direct_grad_only_ownerupdate_cpu).all().item()),
        }
    if mixed_vjp_direct_rgb_only_cpu is not None and mixed_vjp_direct_rgb_only_ms is not None:
        mixed_vjp_direct_rgb_only_errors = {
            "grad_vs_reduce": float((mixed_vjp_direct_rgb_only_cpu - mixed_vjp_grad).abs().max().item())
            if mixed_vjp_grad is not None
            else 0.0,
            "grad_rel_vs_reduce": float(
                (mixed_vjp_direct_rgb_only_cpu - mixed_vjp_grad).abs().max().item()
                / max(float(mixed_vjp_grad.abs().max().item()), 1.0e-9)
            )
            if mixed_vjp_grad is not None
            else 0.0,
            "grad_vs_grad_only": float((mixed_vjp_direct_rgb_only_cpu - mixed_vjp_direct_grad_only_cpu).abs().max().item())
            if mixed_vjp_direct_grad_only_cpu is not None
            else 0.0,
        }
        mixed_vjp_direct_rgb_only_timing = {
            "fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only": float(mixed_vjp_direct_rgb_only_ms),
            "mixed_vjp_direct_rgb_only_speedup_vs_reduce": float(mixed_vjp_ms or 0.0)
            / float(max(mixed_vjp_direct_rgb_only_ms, 1.0e-9)),
            "mixed_vjp_direct_rgb_only_speedup_vs_grad_only": float(mixed_vjp_direct_grad_only_ms or 0.0)
            / float(max(mixed_vjp_direct_rgb_only_ms, 1.0e-9)),
            "mixed_vjp_direct_rgb_only_speedup_vs_mixed_forward": float(mixed_ms or 0.0)
            / float(max(mixed_vjp_direct_rgb_only_ms, 1.0e-9)),
        }
        mixed_vjp_direct_rgb_only_summary = {
            "abs_sum": float(mixed_vjp_direct_rgb_only_cpu.abs().sum().item()),
            "abs_max": float(mixed_vjp_direct_rgb_only_cpu.abs().max().item()),
            "finite": bool(torch.isfinite(mixed_vjp_direct_rgb_only_cpu).all().item()),
        }
    if mixed_vjp_direct_track_cpu is not None and mixed_vjp_direct_track_ms is not None:
        mixed_vjp_direct_track_errors = {
            "grad_vs_reduce": float((mixed_vjp_direct_track_cpu - mixed_vjp_grad).abs().max().item())
            if mixed_vjp_grad is not None
            else 0.0,
            "grad_rel_vs_reduce": float(
                (mixed_vjp_direct_track_cpu - mixed_vjp_grad).abs().max().item()
                / max(float(mixed_vjp_grad.abs().max().item()), 1.0e-9)
            )
            if mixed_vjp_grad is not None
            else 0.0,
            "grad_vs_grad_only": float((mixed_vjp_direct_track_cpu - mixed_vjp_direct_grad_only_cpu).abs().max().item())
            if mixed_vjp_direct_grad_only_cpu is not None
            else 0.0,
        }
        mixed_vjp_direct_track_timing = {
            "fused_slab_affine_num32_den16_vjp_direct_atomic_track": float(mixed_vjp_direct_track_ms),
            "mixed_vjp_direct_track_speedup_vs_reduce": float(mixed_vjp_ms or 0.0)
            / float(max(mixed_vjp_direct_track_ms, 1.0e-9)),
            "mixed_vjp_direct_track_speedup_vs_grad_only": float(mixed_vjp_direct_grad_only_ms or 0.0)
            / float(max(mixed_vjp_direct_track_ms, 1.0e-9)),
            "mixed_vjp_direct_track_speedup_vs_mixed_forward": float(mixed_ms or 0.0)
            / float(max(mixed_vjp_direct_track_ms, 1.0e-9)),
        }
        mixed_vjp_direct_track_summary = {
            "abs_sum": float(mixed_vjp_direct_track_cpu.abs().sum().item()),
            "abs_max": float(mixed_vjp_direct_track_cpu.abs().max().item()),
            "finite": bool(torch.isfinite(mixed_vjp_direct_track_cpu).all().item()),
        }
    if autograd_vjp_grads:
        autograd_reduce = autograd_vjp_grads["reduce"]
        autograd_vjp_errors = {}
        autograd_vjp_summaries = {}
        for mode, grad in sorted(autograd_vjp_grads.items()):
            diff = (grad - autograd_reduce).abs()
            rel = float(diff.max().item()) / max(float(autograd_reduce.abs().max().item()), 1.0e-9)
            autograd_vjp_errors[mode] = {
                "grad_vs_autograd_reduce": float(diff.max().item()),
                "grad_rel_vs_autograd_reduce": rel,
            }
            autograd_vjp_summaries[mode] = {
                "abs_sum": float(grad.abs().sum().item()),
                "abs_max": float(grad.abs().max().item()),
                "finite": bool(torch.isfinite(grad).all().item()),
            }
        if mixed_vjp_grad is not None:
            diff = (autograd_reduce - mixed_vjp_grad).abs()
            autograd_vjp_errors["reduce"]["grad_vs_raw_reduce"] = float(diff.max().item())
            autograd_vjp_errors["reduce"]["grad_rel_vs_raw_reduce"] = float(
                diff.max().item() / max(float(mixed_vjp_grad.abs().max().item()), 1.0e-9)
            )
    coeff16_errors = None
    coeff16_timing = None
    if (
        coeff16_rgb is not None
        and coeff16_alpha is not None
        and coeff16_depth is not None
        and coeff16_ms is not None
    ):
        coeff16_errors = {
            "rgb": float((coeff16_rgb - direct_rgb).abs().max().item()),
            "alpha": float((coeff16_alpha - direct_alpha).abs().max().item()),
            "depth": float((coeff16_depth - direct_depth).abs().max().item()),
            "rgb_vs_id_csr": float((coeff16_rgb - fused_rgb).abs().max().item()),
            "alpha_vs_id_csr": float((coeff16_alpha - fused_alpha).abs().max().item()),
            "depth_vs_id_csr": float((coeff16_depth - fused_depth).abs().max().item()),
            "rgb_vs_coeff32": float((coeff16_rgb - coeff_rgb).abs().max().item()) if coeff_rgb is not None else 0.0,
            "alpha_vs_coeff32": float((coeff16_alpha - coeff_alpha).abs().max().item())
            if coeff_alpha is not None
            else 0.0,
            "depth_vs_coeff32": float((coeff16_depth - coeff_depth).abs().max().item())
            if coeff_depth is not None
            else 0.0,
        }
        coeff16_timing = {
            "fused_slab_affine_coeff16_csr": float(coeff16_ms),
            "coeff16_speedup_vs_explicit": float(direct_ms) / float(max(coeff16_ms, 1.0e-9)),
            "coeff16_speedup_vs_id_csr": float(fused_ms) / float(max(coeff16_ms, 1.0e-9)),
            "coeff16_speedup_vs_coeff32": float(coeff_ms or 0.0) / float(max(coeff16_ms, 1.0e-9)),
        }
    candidate_storage_bytes = _storage_bytes(bundle["row_index"], bundle["row_offsets"], bundle["candidate_ids"])
    coeff_candidate_storage_bytes = _storage_bytes(
        bundle["row_index"],
        bundle["row_offsets"],
        bundle["candidate_depth_coeffs"],
    )
    mixed_candidate_storage_bytes = _storage_bytes(
        bundle["row_index"],
        bundle["row_offsets"],
        bundle["candidate_depth_coeffs"][:, :2].contiguous(),
        bundle["candidate_depth_coeffs"][:, 2:].contiguous().to(dtype=torch.float16),
    )
    ownerupdate_candidate_storage_bytes = _storage_bytes(
        bundle["row_index"],
        bundle["row_offsets"],
        bundle["candidate_ids"],
        bundle["candidate_depth_coeffs"][:, :2].contiguous(),
        bundle["candidate_depth_coeffs"][:, 2:].contiguous().to(dtype=torch.float16),
        torch.empty((len(boundaries), 2), dtype=torch.int32),
    )
    coeff16_candidate_storage_bytes = _storage_bytes(
        bundle["row_index"],
        bundle["row_offsets"],
        bundle["candidate_depth_coeffs"].to(dtype=torch.float16),
    )
    affine_ray_storage_bytes = _storage_bytes(bundle["ray_coeff"])
    explicit_ray_storage_bytes = _storage_bytes(bundle["explicit_rays"])
    return {
        "frames": frame_count,
        "render_size": render_size,
        "train_views": list(data["train_views"]),
        "track_count": track_count,
        "pixel_rays": int(track_count * frame_count),
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "time_slabs": time_slabs,
        "layout": layout,
        "candidate_order": str(bundle["candidate_order"]),
        "row_count": int(bundle["row_count"]),
        "tile_shape": bundle["tile_shape"],
        "tile_grid_shape": bundle["tile_grid_shape"],
        "candidate_count": int(bundle["candidate_count"]),
        "max_candidates_per_row": int(bundle["max_candidates_per_row"]),
        "avg_candidates_per_row": float(bundle["avg_candidates_per_row"]),
        "empty_row_count": int(bundle["empty_row_count"]),
        "per_frame_event_sum": int(bundle["per_frame_event_sum"]),
        "missing_sample_events": int(bundle["missing_sample_events"]),
        "extra_candidate_events": int(bundle["extra_candidate_events"]),
        "candidate_replay_iterations": int(bundle["candidate_replay_iterations"]),
        "candidate_depth_order": bundle["candidate_depth_order"],
        "direct_boundary_iterations": int(bundle["direct_boundary_iterations"]),
        "compiled_boundary_tests": int(bundle["compiled_boundary_tests"]),
        "compiled_boundary_test_ratio": float(bundle["compiled_boundary_tests"])
        / float(max(int(bundle["direct_boundary_iterations"]), 1)),
        "candidate_replay_iteration_ratio": float(bundle["candidate_replay_iterations"])
        / float(max(int(bundle["per_frame_event_sum"]), 1)),
        "candidate_storage_bytes": candidate_storage_bytes,
        "coeff_candidate_storage_bytes": coeff_candidate_storage_bytes,
        "mixed_candidate_storage_bytes": mixed_candidate_storage_bytes,
        "ownerupdate_candidate_storage_bytes": ownerupdate_candidate_storage_bytes,
        "coeff16_candidate_storage_bytes": coeff16_candidate_storage_bytes,
        "affine_ray_storage_bytes": affine_ray_storage_bytes,
        "explicit_ray_storage_bytes": explicit_ray_storage_bytes,
        "total_fused_storage_bytes": candidate_storage_bytes + affine_ray_storage_bytes,
        "total_coeff_fused_storage_bytes": coeff_candidate_storage_bytes + affine_ray_storage_bytes,
        "total_mixed_fused_storage_bytes": mixed_candidate_storage_bytes + affine_ray_storage_bytes,
        "total_ownerupdate_fused_storage_bytes": ownerupdate_candidate_storage_bytes + affine_ray_storage_bytes,
        "total_coeff16_fused_storage_bytes": coeff16_candidate_storage_bytes + affine_ray_storage_bytes,
        "fused_storage_vs_explicit_ray_ratio": float(candidate_storage_bytes + affine_ray_storage_bytes)
        / float(max(explicit_ray_storage_bytes, 1)),
        "coeff_fused_storage_vs_explicit_ray_ratio": float(coeff_candidate_storage_bytes + affine_ray_storage_bytes)
        / float(max(explicit_ray_storage_bytes, 1)),
        "mixed_fused_storage_vs_explicit_ray_ratio": float(mixed_candidate_storage_bytes + affine_ray_storage_bytes)
        / float(max(explicit_ray_storage_bytes, 1)),
        "ownerupdate_fused_storage_vs_explicit_ray_ratio": float(
            ownerupdate_candidate_storage_bytes + affine_ray_storage_bytes
        )
        / float(max(explicit_ray_storage_bytes, 1)),
        "coeff16_fused_storage_vs_explicit_ray_ratio": float(coeff16_candidate_storage_bytes + affine_ray_storage_bytes)
        / float(max(explicit_ray_storage_bytes, 1)),
        "max_rgb_abs_error_vs_explicit_realray": float((fused_rgb - direct_rgb).abs().max().item()),
        "max_alpha_abs_error_vs_explicit_realray": float((fused_alpha - direct_alpha).abs().max().item()),
        "max_depth_abs_error_vs_explicit_realray": float((fused_depth - direct_depth).abs().max().item()),
        "coeff_max_abs_error_vs_explicit_realray": coeff_errors,
        "mixed_max_abs_error_vs_explicit_realray": mixed_errors,
        "ownerupdate_max_abs_error_vs_explicit_realray": ownerupdate_errors,
        "mixed_vjp_max_abs_error_vs_explicit_realray": mixed_vjp_errors,
        "mixed_vjp_grad_summary": mixed_vjp_grad_summary,
        "mixed_vjp_direct_max_abs_error_vs_explicit_realray": mixed_vjp_direct_errors,
        "mixed_vjp_direct_grad_summary": mixed_vjp_direct_grad_summary,
        "mixed_vjp_direct_grad_only_error_vs_reduce": mixed_vjp_direct_grad_only_errors,
        "mixed_vjp_direct_grad_only_summary": mixed_vjp_direct_grad_only_summary,
        "mixed_vjp_direct_grad_only_ownerupdate_error_vs_reduce": mixed_vjp_direct_grad_only_ownerupdate_errors,
        "mixed_vjp_direct_grad_only_ownerupdate_summary": mixed_vjp_direct_grad_only_ownerupdate_summary,
        "mixed_vjp_direct_rgb_only_error_vs_reduce": mixed_vjp_direct_rgb_only_errors,
        "mixed_vjp_direct_rgb_only_summary": mixed_vjp_direct_rgb_only_summary,
        "mixed_vjp_direct_track_error_vs_reduce": mixed_vjp_direct_track_errors,
        "mixed_vjp_direct_track_summary": mixed_vjp_direct_track_summary,
        "autograd_vjp_error_vs_reduce": autograd_vjp_errors,
        "autograd_vjp_summary": autograd_vjp_summaries,
        "vjp_seed_summary": vjp_seed_summary,
        "coeff16_max_abs_error_vs_explicit_realray": coeff16_errors,
        "outputs_are_finite": bool(
            torch.isfinite(fused_rgb).all().item()
            and torch.isfinite(fused_alpha).all().item()
            and torch.isfinite(fused_depth).all().item()
            and (
                coeff_rgb is None
                or (
                    torch.isfinite(coeff_rgb).all().item()
                    and torch.isfinite(coeff_alpha).all().item()  # type: ignore[arg-type]
                    and torch.isfinite(coeff_depth).all().item()  # type: ignore[arg-type]
                )
            )
            and (
                mixed_rgb is None
                or (
                    torch.isfinite(mixed_rgb).all().item()
                    and torch.isfinite(mixed_alpha).all().item()  # type: ignore[arg-type]
                    and torch.isfinite(mixed_depth).all().item()  # type: ignore[arg-type]
                )
            )
            and (
                ownerupdate_rgb is None
                or (
                    torch.isfinite(ownerupdate_rgb).all().item()
                    and torch.isfinite(ownerupdate_alpha).all().item()  # type: ignore[arg-type]
                    and torch.isfinite(ownerupdate_depth).all().item()  # type: ignore[arg-type]
                )
            )
            and (
                mixed_vjp_rgb is None
                or (
                    torch.isfinite(mixed_vjp_rgb).all().item()
                    and torch.isfinite(mixed_vjp_alpha).all().item()  # type: ignore[arg-type]
                    and torch.isfinite(mixed_vjp_depth).all().item()  # type: ignore[arg-type]
                    and bool(mixed_vjp_grad_summary and mixed_vjp_grad_summary["finite"])
                )
            )
            and (
                mixed_vjp_direct_rgb is None
                or (
                    torch.isfinite(mixed_vjp_direct_rgb).all().item()
                    and torch.isfinite(mixed_vjp_direct_alpha).all().item()  # type: ignore[arg-type]
                    and torch.isfinite(mixed_vjp_direct_depth).all().item()  # type: ignore[arg-type]
                    and bool(mixed_vjp_direct_grad_summary and mixed_vjp_direct_grad_summary["finite"])
                )
            )
            and (
                mixed_vjp_direct_grad_only_cpu is None
                or bool(mixed_vjp_direct_grad_only_summary and mixed_vjp_direct_grad_only_summary["finite"])
            )
            and (
                mixed_vjp_direct_grad_only_ownerupdate_cpu is None
                or bool(
                    mixed_vjp_direct_grad_only_ownerupdate_summary
                    and mixed_vjp_direct_grad_only_ownerupdate_summary["finite"]
                )
            )
            and (
                mixed_vjp_direct_rgb_only_cpu is None
                or bool(mixed_vjp_direct_rgb_only_summary and mixed_vjp_direct_rgb_only_summary["finite"])
            )
            and (
                mixed_vjp_direct_track_cpu is None
                or bool(mixed_vjp_direct_track_summary and mixed_vjp_direct_track_summary["finite"])
            )
            and (
                coeff16_rgb is None
                or (
                    torch.isfinite(coeff16_rgb).all().item()
                    and torch.isfinite(coeff16_alpha).all().item()  # type: ignore[arg-type]
                    and torch.isfinite(coeff16_depth).all().item()  # type: ignore[arg-type]
                )
            )
        ),
        "linear_fit": {
            "max_origin_residual": float(bundle["max_origin_residual"]),
            "max_direction_residual": float(bundle["max_direction_residual"]),
            "residual_depth_padding": residual_depth_padding,
        },
        "timing_ms": {
            "explicit_per_frame_realray": float(direct_ms),
            "fused_slab_affine_csr": float(fused_ms),
            "speedup_vs_explicit": float(direct_ms) / float(max(fused_ms, 1.0e-9)),
            **(coeff_timing or {}),
            **(mixed_timing or {}),
            **(ownerupdate_timing or {}),
            **(mixed_vjp_timing or {}),
            **(mixed_vjp_direct_timing or {}),
            **(mixed_vjp_direct_grad_only_timing or {}),
            **(mixed_vjp_direct_grad_only_ownerupdate_timing or {}),
            **(mixed_vjp_direct_rgb_only_timing or {}),
            **(mixed_vjp_direct_track_timing or {}),
            **(coeff16_timing or {}),
        },
    }


def run_smoke(
    *,
    config_path: Path,
    frame_counts: tuple[int, ...],
    render_size: int,
    site_count: int,
    time_slabs: int,
    layout: str,
    tile_h: int,
    tile_w: int,
    candidate_order: str,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    residual_depth_padding: float,
    synthetic_motion: SyntheticRayMotion,
    include_ownerupdate: bool,
    include_vjp: bool,
    vjp_seed_mode: str,
    vjp_reduce_chunk_size: int,
    timing_iters: int,
) -> dict[str, Any]:
    if timing_iters <= 0:
        raise ValueError("timing_iters must be positive")
    if vjp_reduce_chunk_size <= 0:
        raise ValueError("vjp_reduce_chunk_size must be positive")
    if vjp_seed_mode not in VJP_SEED_MODES:
        raise ValueError(f"vjp_seed_mode must be one of {VJP_SEED_MODES}, got {vjp_seed_mode!r}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    rows = [
        _profile_frame_count(
            frame_count=frame_count,
            config_path=config_path,
            render_size=render_size,
            site_count=site_count,
            time_slabs=time_slabs,
            layout=layout,
            tile_h=tile_h,
            tile_w=tile_w,
            candidate_order=candidate_order,
            near=near,
            far=far,
            density=density,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
            residual_depth_padding=residual_depth_padding,
            synthetic_motion=synthetic_motion,
            include_ownerupdate=include_ownerupdate,
            include_vjp=include_vjp,
            vjp_seed_mode=vjp_seed_mode,
            vjp_reduce_chunk_size=vjp_reduce_chunk_size,
            timing_iters=timing_iters,
        )
        for frame_count in frame_counts
    ]
    tolerance = 5.0e-4
    depth_order_rows = [row["candidate_depth_order"] for row in rows]
    depth_order_diagnostics = {
        "checked_samples": int(sum(int(row["checked_samples"]) for row in depth_order_rows)),
        "valid_depth_values": int(sum(int(row["valid_depth_values"]) for row in depth_order_rows)),
        "adjacent_pairs": int(sum(int(row["adjacent_pairs"]) for row in depth_order_rows)),
        "adjacent_inversions": int(sum(int(row["adjacent_inversions"]) for row in depth_order_rows)),
        "samples_with_adjacent_inversions": int(
            sum(int(row["samples_with_adjacent_inversions"]) for row in depth_order_rows)
        ),
        "max_adjacent_inversions_per_sample": int(
            max(int(row["max_adjacent_inversions_per_sample"]) for row in depth_order_rows)
        ),
        "max_depth_drop": float(max(float(row["max_depth_drop"]) for row in depth_order_rows)),
        "ordered_append_safe": bool(all(bool(row["ordered_append_safe"]) for row in depth_order_rows)),
    }
    if depth_order_diagnostics["adjacent_pairs"]:
        depth_order_diagnostics["adjacent_inversion_rate"] = float(
            depth_order_diagnostics["adjacent_inversions"]
        ) / float(depth_order_diagnostics["adjacent_pairs"])
    else:
        depth_order_diagnostics["adjacent_inversion_rate"] = 0.0
    max_rgb_error = max(float(row["max_rgb_abs_error_vs_explicit_realray"]) for row in rows)
    max_alpha_error = max(float(row["max_alpha_abs_error_vs_explicit_realray"]) for row in rows)
    max_depth_error = max(float(row["max_depth_abs_error_vs_explicit_realray"]) for row in rows)
    coeff_rows = [row for row in rows if row["coeff_max_abs_error_vs_explicit_realray"] is not None]
    mixed_rows = [row for row in rows if row["mixed_max_abs_error_vs_explicit_realray"] is not None]
    ownerupdate_rows = [row for row in rows if row["ownerupdate_max_abs_error_vs_explicit_realray"] is not None]
    mixed_vjp_rows = [row for row in rows if row["mixed_vjp_max_abs_error_vs_explicit_realray"] is not None]
    mixed_vjp_direct_rows = [
        row for row in rows if row["mixed_vjp_direct_max_abs_error_vs_explicit_realray"] is not None
    ]
    mixed_vjp_direct_grad_only_rows = [
        row for row in rows if row["mixed_vjp_direct_grad_only_error_vs_reduce"] is not None
    ]
    mixed_vjp_direct_grad_only_ownerupdate_rows = [
        row for row in rows if row["mixed_vjp_direct_grad_only_ownerupdate_error_vs_reduce"] is not None
    ]
    mixed_vjp_direct_rgb_only_rows = [
        row for row in rows if row["mixed_vjp_direct_rgb_only_error_vs_reduce"] is not None
    ]
    mixed_vjp_direct_track_rows = [
        row for row in rows if row["mixed_vjp_direct_track_error_vs_reduce"] is not None
    ]
    coeff16_rows = [row for row in rows if row["coeff16_max_abs_error_vs_explicit_realray"] is not None]
    max_coeff_error = (
        max(
            max(float(row["coeff_max_abs_error_vs_explicit_realray"][key]) for key in ("rgb", "alpha", "depth"))
            for row in coeff_rows
        )
        if coeff_rows
        else None
    )
    max_mixed_error = (
        max(
            max(float(row["mixed_max_abs_error_vs_explicit_realray"][key]) for key in ("rgb", "alpha", "depth"))
            for row in mixed_rows
        )
        if mixed_rows
        else None
    )
    max_ownerupdate_error = (
        max(
            max(
                float(row["ownerupdate_max_abs_error_vs_explicit_realray"][key])
                for key in ("rgb", "alpha", "depth")
            )
            for row in ownerupdate_rows
        )
        if ownerupdate_rows
        else None
    )
    max_mixed_vjp_error = (
        max(
            max(float(row["mixed_vjp_max_abs_error_vs_explicit_realray"][key]) for key in ("rgb", "alpha", "depth"))
            for row in mixed_vjp_rows
        )
        if mixed_vjp_rows
        else None
    )
    max_mixed_vjp_direct_error = (
        max(
            max(
                float(row["mixed_vjp_direct_max_abs_error_vs_explicit_realray"][key])
                for key in ("rgb", "alpha", "depth")
            )
            for row in mixed_vjp_direct_rows
        )
        if mixed_vjp_direct_rows
        else None
    )
    max_mixed_vjp_direct_grad_delta = (
        max(float(row["mixed_vjp_direct_max_abs_error_vs_explicit_realray"]["grad_vs_reduce"]) for row in mixed_vjp_direct_rows)
        if mixed_vjp_direct_rows
        else None
    )
    max_mixed_vjp_direct_grad_rel_delta = (
        max(
            float(row["mixed_vjp_direct_max_abs_error_vs_explicit_realray"]["grad_rel_vs_reduce"])
            for row in mixed_vjp_direct_rows
        )
        if mixed_vjp_direct_rows
        else None
    )
    max_mixed_vjp_direct_grad_only_delta = (
        max(float(row["mixed_vjp_direct_grad_only_error_vs_reduce"]["grad_vs_reduce"]) for row in mixed_vjp_direct_grad_only_rows)
        if mixed_vjp_direct_grad_only_rows
        else None
    )
    max_mixed_vjp_direct_grad_only_rel_delta = (
        max(
            float(row["mixed_vjp_direct_grad_only_error_vs_reduce"]["grad_rel_vs_reduce"])
            for row in mixed_vjp_direct_grad_only_rows
        )
        if mixed_vjp_direct_grad_only_rows
        else None
    )
    max_mixed_vjp_direct_grad_only_ownerupdate_delta = (
        max(
            float(row["mixed_vjp_direct_grad_only_ownerupdate_error_vs_reduce"]["grad_vs_reduce"])
            for row in mixed_vjp_direct_grad_only_ownerupdate_rows
        )
        if mixed_vjp_direct_grad_only_ownerupdate_rows
        else None
    )
    max_mixed_vjp_direct_grad_only_ownerupdate_rel_delta = (
        max(
            float(row["mixed_vjp_direct_grad_only_ownerupdate_error_vs_reduce"]["grad_rel_vs_reduce"])
            for row in mixed_vjp_direct_grad_only_ownerupdate_rows
        )
        if mixed_vjp_direct_grad_only_ownerupdate_rows
        else None
    )
    max_mixed_vjp_direct_rgb_only_delta = (
        max(float(row["mixed_vjp_direct_rgb_only_error_vs_reduce"]["grad_vs_reduce"]) for row in mixed_vjp_direct_rgb_only_rows)
        if mixed_vjp_direct_rgb_only_rows
        else None
    )
    max_mixed_vjp_direct_rgb_only_rel_delta = (
        max(
            float(row["mixed_vjp_direct_rgb_only_error_vs_reduce"]["grad_rel_vs_reduce"])
            for row in mixed_vjp_direct_rgb_only_rows
        )
        if mixed_vjp_direct_rgb_only_rows
        else None
    )
    max_mixed_vjp_direct_track_delta = (
        max(float(row["mixed_vjp_direct_track_error_vs_reduce"]["grad_vs_reduce"]) for row in mixed_vjp_direct_track_rows)
        if mixed_vjp_direct_track_rows
        else None
    )
    max_mixed_vjp_direct_track_rel_delta = (
        max(
            float(row["mixed_vjp_direct_track_error_vs_reduce"]["grad_rel_vs_reduce"])
            for row in mixed_vjp_direct_track_rows
        )
        if mixed_vjp_direct_track_rows
        else None
    )
    autograd_vjp_rows = [row for row in rows if row["autograd_vjp_error_vs_reduce"] is not None]
    max_autograd_vjp_rel_delta_by_mode = {
        mode: max(
            float(row["autograd_vjp_error_vs_reduce"][mode]["grad_rel_vs_autograd_reduce"])
            for row in autograd_vjp_rows
        )
        for mode in ("direct_atomic", "direct_atomic_grad_only", "direct_atomic_rgb_only", "direct_atomic_track")
        if autograd_vjp_rows
    }
    max_autograd_vjp_raw_reduce_rel_delta = (
        max(
            float(row["autograd_vjp_error_vs_reduce"]["reduce"].get("grad_rel_vs_raw_reduce", 0.0))
            for row in autograd_vjp_rows
        )
        if autograd_vjp_rows
        else None
    )
    max_coeff16_error = (
        max(
            max(float(row["coeff16_max_abs_error_vs_explicit_realray"][key]) for key in ("rgb", "alpha", "depth"))
            for row in coeff16_rows
        )
        if coeff16_rows
        else None
    )
    coeff16_tolerance = 1.0e-3
    direct_grad_tolerance = 2.0e-3
    direct_grad_relative_tolerance = 1.0e-5

    def _within_grad_tolerance(delta: float | None, rel_delta: float | None) -> bool:
        return (
            delta is None
            or delta <= direct_grad_tolerance
            or (rel_delta is not None and rel_delta <= direct_grad_relative_tolerance)
        )

    rgb_only_expected_to_match_reduce = vjp_seed_mode == "rgb"
    rgb_only_within_grad_tolerance = _within_grad_tolerance(
        max_mixed_vjp_direct_rgb_only_delta,
        max_mixed_vjp_direct_rgb_only_rel_delta,
    )
    rgb_only_expected_behavior = (
        rgb_only_within_grad_tolerance
        if rgb_only_expected_to_match_reduce
        else max_mixed_vjp_direct_rgb_only_delta is not None and not rgb_only_within_grad_tolerance
    )
    autograd_rgb_only_expected_behavior = (
        max_autograd_vjp_rel_delta_by_mode.get("direct_atomic_rgb_only") is None
        or max_autograd_vjp_rel_delta_by_mode["direct_atomic_rgb_only"] <= direct_grad_relative_tolerance
    )
    autograd_general_modes_match = all(
        max_autograd_vjp_rel_delta_by_mode.get(mode, 0.0) <= direct_grad_relative_tolerance
        for mode in ("direct_atomic", "direct_atomic_grad_only", "direct_atomic_track")
    )
    ownerupdate_checked = include_ownerupdate
    ownerupdate_vjp_checked = include_ownerupdate and include_vjp
    ownerupdate_grad_within_tolerance = _within_grad_tolerance(
        max_mixed_vjp_direct_grad_only_ownerupdate_delta,
        max_mixed_vjp_direct_grad_only_ownerupdate_rel_delta,
    )
    ownerupdate_gradients_finite = all(
        row["mixed_vjp_direct_grad_only_ownerupdate_summary"] is not None
        and bool(row["mixed_vjp_direct_grad_only_ownerupdate_summary"]["finite"])
        for row in rows
    )
    ownerupdate_matches_explicit = max_ownerupdate_error is not None and max_ownerupdate_error <= tolerance

    acceptance = {
        "zero_missing_sample_events": all(int(row["missing_sample_events"]) == 0 for row in rows),
        "candidate_rows_under_metal_cap": all(
            int(row["max_candidates_per_row"]) <= MAX_REALRAY_BOUNDARIES for row in rows
        ),
        "matches_explicit_realray": max(max_rgb_error, max_alpha_error, max_depth_error) <= tolerance,
        "coeff_matches_explicit_realray": max_coeff_error is None or max_coeff_error <= tolerance,
        "mixed_matches_explicit_realray": max_mixed_error is None or max_mixed_error <= tolerance,
        "mixed_vjp_matches_explicit_realray": max_mixed_vjp_error is None or max_mixed_vjp_error <= tolerance,
        "mixed_vjp_direct_matches_explicit_realray": max_mixed_vjp_direct_error is None
        or max_mixed_vjp_direct_error <= tolerance,
        "mixed_vjp_direct_matches_reduce_grad": _within_grad_tolerance(
            max_mixed_vjp_direct_grad_delta,
            max_mixed_vjp_direct_grad_rel_delta,
        ),
        "mixed_vjp_direct_grad_only_matches_reduce_grad": _within_grad_tolerance(
            max_mixed_vjp_direct_grad_only_delta,
            max_mixed_vjp_direct_grad_only_rel_delta,
        ),
        "mixed_vjp_direct_rgb_only_has_expected_seed_behavior": rgb_only_expected_behavior,
        "mixed_vjp_direct_track_matches_reduce_grad": _within_grad_tolerance(
            max_mixed_vjp_direct_track_delta,
            max_mixed_vjp_direct_track_rel_delta,
        ),
        "autograd_vjp_modes_match_reduce": autograd_general_modes_match,
        "autograd_vjp_rgb_only_has_expected_seed_behavior": autograd_rgb_only_expected_behavior,
        "autograd_reduce_matches_raw_reduce": max_autograd_vjp_raw_reduce_rel_delta is None
        or max_autograd_vjp_raw_reduce_rel_delta <= direct_grad_relative_tolerance,
        "mixed_vjp_gradients_finite": all(
            row["mixed_vjp_grad_summary"] is None or bool(row["mixed_vjp_grad_summary"]["finite"]) for row in rows
        ),
        "mixed_vjp_direct_gradients_finite": all(
            row["mixed_vjp_direct_grad_summary"] is None
            or bool(row["mixed_vjp_direct_grad_summary"]["finite"])
            for row in rows
        ),
        "mixed_vjp_direct_grad_only_gradients_finite": all(
            row["mixed_vjp_direct_grad_only_summary"] is None
            or bool(row["mixed_vjp_direct_grad_only_summary"]["finite"])
            for row in rows
        ),
        "mixed_vjp_direct_rgb_only_gradients_finite": all(
            row["mixed_vjp_direct_rgb_only_summary"] is None
            or bool(row["mixed_vjp_direct_rgb_only_summary"]["finite"])
            for row in rows
        ),
        "mixed_vjp_direct_track_gradients_finite": all(
            row["mixed_vjp_direct_track_summary"] is None
            or bool(row["mixed_vjp_direct_track_summary"]["finite"])
            for row in rows
        ),
        "autograd_vjp_gradients_finite": all(
            row["autograd_vjp_summary"] is None
            or all(bool(summary["finite"]) for summary in row["autograd_vjp_summary"].values())
            for row in rows
        ),
        "outputs_are_finite": all(bool(row["outputs_are_finite"]) for row in rows),
        "affine_fit_exact": all(
            float(row["linear_fit"]["max_origin_residual"]) <= 1.0e-5
            and float(row["linear_fit"]["max_direction_residual"]) <= 1.0e-5
            for row in rows
        ),
        "moving_ray_tracks_present": synthetic_motion.active,
    }
    if ownerupdate_checked:
        acceptance["ownerupdate_matches_explicit_realray"] = ownerupdate_matches_explicit
    if ownerupdate_vjp_checked:
        acceptance["mixed_vjp_direct_grad_only_ownerupdate_matches_reduce_grad"] = ownerupdate_grad_within_tolerance
        acceptance["mixed_vjp_direct_grad_only_ownerupdate_gradients_finite"] = ownerupdate_gradients_finite
    return {
        "benchmark": "world_foam_lane2_fused_slab_affine_realray_mps_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "4_moving_ray_affine_tiled_slab_csr_fused_forward",
        "device": "mps",
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "time_slabs": time_slabs,
        "layout": layout,
        "candidate_order": candidate_order,
        "candidate_storage_format": f"{layout}_slab_csr_i32",
        "coeff_candidate_storage_format": "per_track_slab_csr_f32_depth_coefficients" if layout == "per-track" else None,
        "mixed_candidate_storage_format": "per_track_slab_csr_f32_num_f16_den" if layout == "per-track" else None,
        "ownerupdate_candidate_storage_format": "per_track_slab_csr_i32_boundary_id_f32_num_f16_den"
        if layout == "per-track"
        else None,
        "coeff16_candidate_storage_format": "per_track_slab_csr_f16_depth_coefficients" if layout == "per-track" else None,
        "ray_track_format": "affine_f32_12_origin_base_slope_direction_base_slope",
        "comparison_unit": "fused_moving_ray_csr_forward_vs_explicit_per_frame_realray_forward",
        "gradient_scope": f"mixed_num32_den16_site_rgba_vjp_{vjp_seed_mode}_seed"
        if include_vjp
        else "none_forward_only",
        "quality_claim": False,
        "training_claim": False,
        "include_ownerupdate": include_ownerupdate,
        "include_vjp": include_vjp,
        "vjp_seed_mode": vjp_seed_mode,
        "vjp_reduce_chunk_size": vjp_reduce_chunk_size,
        "synthetic_motion": synthetic_motion.to_dict(),
        "tolerance": tolerance,
        "coeff16_tolerance": coeff16_tolerance,
        "direct_grad_tolerance": direct_grad_tolerance,
        "direct_grad_relative_tolerance": direct_grad_relative_tolerance,
        "max_realray_boundaries": MAX_REALRAY_BOUNDARIES,
        "timing_iters": timing_iters,
        "rows": rows,
        "candidate_depth_order_diagnostics": depth_order_diagnostics,
        "max_errors": {
            "rgb": max_rgb_error,
            "alpha": max_alpha_error,
            "depth": max_depth_error,
        },
        "mixed_max_error": max_mixed_error,
        "mixed_vjp_diagnostics": {
            "max_error": max_mixed_vjp_error,
            "within_strict_tolerance": max_mixed_vjp_error is None or max_mixed_vjp_error <= tolerance,
        },
        "mixed_vjp_direct_diagnostics": {
            "max_error": max_mixed_vjp_direct_error,
            "max_grad_delta_vs_reduce": max_mixed_vjp_direct_grad_delta,
            "max_grad_rel_delta_vs_reduce": max_mixed_vjp_direct_grad_rel_delta,
            "within_strict_tolerance": max_mixed_vjp_direct_error is None
            or max_mixed_vjp_direct_error <= tolerance,
            "within_grad_tolerance": max_mixed_vjp_direct_grad_delta is None
            or max_mixed_vjp_direct_grad_delta <= direct_grad_tolerance
            or (
                max_mixed_vjp_direct_grad_rel_delta is not None
                and max_mixed_vjp_direct_grad_rel_delta <= direct_grad_relative_tolerance
            ),
        },
        "mixed_vjp_direct_grad_only_diagnostics": {
            "max_grad_delta_vs_reduce": max_mixed_vjp_direct_grad_only_delta,
            "max_grad_rel_delta_vs_reduce": max_mixed_vjp_direct_grad_only_rel_delta,
            "within_grad_tolerance": max_mixed_vjp_direct_grad_only_delta is None
            or max_mixed_vjp_direct_grad_only_delta <= direct_grad_tolerance
            or (
                max_mixed_vjp_direct_grad_only_rel_delta is not None
                and max_mixed_vjp_direct_grad_only_rel_delta <= direct_grad_relative_tolerance
            ),
        },
        "mixed_vjp_direct_grad_only_ownerupdate_diagnostics": {
            "checked": ownerupdate_vjp_checked,
            "max_grad_delta_vs_reduce": max_mixed_vjp_direct_grad_only_ownerupdate_delta,
            "max_grad_rel_delta_vs_reduce": max_mixed_vjp_direct_grad_only_ownerupdate_rel_delta,
            "within_grad_tolerance": ownerupdate_grad_within_tolerance if ownerupdate_vjp_checked else None,
        },
        "mixed_vjp_direct_rgb_only_diagnostics": {
            "max_grad_delta_vs_reduce": max_mixed_vjp_direct_rgb_only_delta,
            "max_grad_rel_delta_vs_reduce": max_mixed_vjp_direct_rgb_only_rel_delta,
            "expected_to_match_reduce": rgb_only_expected_to_match_reduce,
            "expected_divergence_from_reduce": not rgb_only_expected_to_match_reduce,
            "within_grad_tolerance": rgb_only_within_grad_tolerance,
            "has_expected_seed_behavior": rgb_only_expected_behavior,
        },
        "mixed_vjp_direct_track_diagnostics": {
            "max_grad_delta_vs_reduce": max_mixed_vjp_direct_track_delta,
            "max_grad_rel_delta_vs_reduce": max_mixed_vjp_direct_track_rel_delta,
            "within_grad_tolerance": max_mixed_vjp_direct_track_delta is None
            or max_mixed_vjp_direct_track_delta <= direct_grad_tolerance
            or (
                max_mixed_vjp_direct_track_rel_delta is not None
                and max_mixed_vjp_direct_track_rel_delta <= direct_grad_relative_tolerance
            ),
        },
        "autograd_vjp_diagnostics": {
            "max_grad_rel_delta_by_mode_vs_autograd_reduce": max_autograd_vjp_rel_delta_by_mode,
            "max_reduce_rel_delta_vs_raw_reduce": max_autograd_vjp_raw_reduce_rel_delta,
            "rgb_only_expected_to_match_reduce": True,
            "rgb_only_has_expected_seed_behavior": autograd_rgb_only_expected_behavior,
            "general_modes_match_reduce": autograd_general_modes_match,
        },
        "ownerupdate_diagnostics": {
            "checked": ownerupdate_checked,
            "max_error": max_ownerupdate_error,
            "within_strict_tolerance": ownerupdate_matches_explicit if ownerupdate_checked else None,
        },
        "coeff16_diagnostics": {
            "max_error": max_coeff16_error,
            "within_approx_tolerance": max_coeff16_error is None or max_coeff16_error <= coeff16_tolerance,
        },
        "acceptance": acceptance,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke fused tiled/slab affine real-ray replay on MPS.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8")
    parser.add_argument("--render-size", type=int, default=16)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=2)
    parser.add_argument("--layout", choices=("tiled", "per-track"), default="tiled")
    parser.add_argument("--candidate-order", choices=("boundary-id", "slab-mid-depth"), default="boundary-id")
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--residual-depth-padding", type=float, default=1.0e-4)
    parser.add_argument("--origin-velocity-x", type=float, default=0.08)
    parser.add_argument("--origin-velocity-y", type=float, default=0.0)
    parser.add_argument("--origin-velocity-z", type=float, default=0.02)
    parser.add_argument("--direction-velocity-x", type=float, default=0.02)
    parser.add_argument("--direction-velocity-y", type=float, default=0.0)
    parser.add_argument("--direction-velocity-z", type=float, default=0.0)
    parser.add_argument("--include-ownerupdate", action="store_true")
    parser.add_argument("--include-vjp", action="store_true")
    parser.add_argument("--vjp-seed-mode", choices=VJP_SEED_MODES, default="rgb")
    parser.add_argument("--vjp-reduce-chunk-size", type=int, default=4)
    parser.add_argument("--timing-iters", type=int, default=5)
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "gate4_fused_slab_affine_realray_mps_smoke.json")
    args = parser.parse_args(argv)
    if args.include_ownerupdate and args.layout != "per-track":
        parser.error("--include-ownerupdate requires --layout per-track; tiled layout does not run owner-update kernels")
    return args


def main() -> None:
    args = parse_args()
    payload = run_smoke(
        config_path=args.config,
        frame_counts=_parse_int_list(args.frame_counts),
        render_size=args.render_size,
        site_count=args.site_count,
        time_slabs=args.time_slabs,
        layout=args.layout,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        candidate_order=args.candidate_order,
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
        include_ownerupdate=bool(args.include_ownerupdate),
        include_vjp=bool(args.include_vjp),
        vjp_seed_mode=str(args.vjp_seed_mode),
        vjp_reduce_chunk_size=args.vjp_reduce_chunk_size,
        timing_iters=args.timing_iters,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
