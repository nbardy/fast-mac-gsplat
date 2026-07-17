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

from gate1_realray_per_sample_reference import (  # noqa: E402
    DEFAULT_CONFIG,
    _load_config,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
    render_samples,
)
from smoke_shared_realray_vjp_mps import _build_candidate_bundle, _make_gradients, _reshape_rgb  # noqa: E402
from torch_world_foam_lane2_fused_slab import (  # noqa: E402
    RealRayReplayConfig,
    shared_realray_rgba_depth_vjp_reduce,
    shared_realray_rgba_depth_vjp_reduce_csr,
)


REDUCTION_CHUNK_SIZE = 4


def _frame_times(frame_count: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [float(frame) / float(max(frame_count - 1, 1)) for frame in range(frame_count)],
        dtype=torch.float32,
        device=device,
    )


def _build_per_track_csr(bundle: dict[str, Any], *, time_slabs: int) -> dict[str, Any]:
    pixel_tracks = int(bundle["pixel_tracks"])
    offsets: list[int] = [0]
    ids: list[int] = []
    for track_id in range(pixel_tracks):
        for slab_id in range(time_slabs):
            candidates = list(bundle["candidate_sets"][(track_id, slab_id)])
            ids.extend(candidates)
            offsets.append(len(ids))
    return {
        "row_index": torch.arange(pixel_tracks, dtype=torch.int32),
        "row_offsets": torch.tensor(offsets, dtype=torch.int32),
        "candidate_ids": torch.tensor(ids, dtype=torch.int32),
        "row_count": pixel_tracks,
        "exact_candidate_sets": True,
        "superset_candidate_sets": True,
        "extra_candidate_refs_vs_per_track_slab": 0,
    }


def _build_tiled_csr(bundle: dict[str, Any], *, time_slabs: int, tile_h: int, tile_w: int) -> dict[str, Any]:
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError("tile_h and tile_w must be positive")
    view_count = int(bundle["view_count"])
    height = int(bundle["height"])
    width = int(bundle["width"])
    tiles_y = (height + tile_h - 1) // tile_h
    tiles_x = (width + tile_w - 1) // tile_w
    tile_count = view_count * tiles_y * tiles_x
    tile_sets: list[list[set[int]]] = [[set() for _ in range(time_slabs)] for _ in range(tile_count)]
    row_index: list[int] = []
    for view in range(view_count):
        for y in range(height):
            for x in range(width):
                track_id = view * height * width + y * width + x
                tile_y = y // tile_h
                tile_x = x // tile_w
                tile_id = view * tiles_y * tiles_x + tile_y * tiles_x + tile_x
                row_index.append(tile_id)
                for slab_id in range(time_slabs):
                    tile_sets[tile_id][slab_id].update(bundle["candidate_sets"][(track_id, slab_id)])

    offsets: list[int] = [0]
    ids: list[int] = []
    for tile_id in range(tile_count):
        for slab_id in range(time_slabs):
            candidates = sorted(tile_sets[tile_id][slab_id])
            ids.extend(candidates)
            offsets.append(len(ids))

    extra_refs = 0
    superset = True
    for track_id, tile_id in enumerate(row_index):
        for slab_id in range(time_slabs):
            track_candidates = set(bundle["candidate_sets"][(track_id, slab_id)])
            tile_candidates = tile_sets[tile_id][slab_id]
            superset = superset and track_candidates.issubset(tile_candidates)
            extra_refs += len(tile_candidates - track_candidates)

    return {
        "row_index": torch.tensor(row_index, dtype=torch.int32),
        "row_offsets": torch.tensor(offsets, dtype=torch.int32),
        "candidate_ids": torch.tensor(ids, dtype=torch.int32),
        "row_count": tile_count,
        "tile_shape": [tile_h, tile_w],
        "tile_grid_shape": [view_count, tiles_y, tiles_x],
        "exact_candidate_sets": False,
        "superset_candidate_sets": bool(superset),
        "extra_candidate_refs_vs_per_track_slab": int(extra_refs),
    }


def _csr_stats(layout: dict[str, Any], *, bundle: dict[str, Any], boundary_count: int, bitset_bytes: int) -> dict[str, Any]:
    offsets = layout["row_offsets"]
    ids = layout["candidate_ids"]
    row_index = layout["row_index"]
    counts = (offsets[1:] - offsets[:-1]).to(dtype=torch.int64)
    candidate_count = int(ids.numel())
    row_offset_bytes = int(offsets.numel() * offsets.element_size())
    candidate_id_bytes = int(ids.numel() * ids.element_size())
    row_index_bytes = int(row_index.numel() * row_index.element_size())
    total_bytes = row_offset_bytes + candidate_id_bytes + row_index_bytes
    if counts.numel() == 0:
        quantiles = {key: 0 for key in ("p50", "p90", "p95", "p99")}
    else:
        sorted_counts = torch.sort(counts).values
        def q(frac: float) -> int:
            index = min(int(math.ceil(frac * float(sorted_counts.numel())) - 1), sorted_counts.numel() - 1)
            return int(sorted_counts[max(index, 0)].item())
        quantiles = {"p50": q(0.50), "p90": q(0.90), "p95": q(0.95), "p99": q(0.99)}
    return {
        "row_count": int(layout["row_count"]),
        "row_index_shape": list(row_index.shape),
        "row_offsets_shape": list(offsets.shape),
        "candidate_ids_shape": list(ids.shape),
        "candidate_count": candidate_count,
        "max_candidates_per_row": int(counts.max().item()) if counts.numel() else 0,
        "avg_candidates_per_row": float(counts.to(dtype=torch.float32).mean().item()) if counts.numel() else 0.0,
        **quantiles,
        "empty_row_count": int((counts == 0).sum().item()) if counts.numel() else 0,
        "row_index_bytes": row_index_bytes,
        "row_offset_bytes": row_offset_bytes,
        "candidate_id_bytes": candidate_id_bytes,
        "storage_bytes": total_bytes,
        "storage_vs_bitset_ratio": float(total_bytes) / float(max(bitset_bytes, 1)),
        "candidate_iterations": int(int(bundle["pixel_rays"]) * candidate_count / max(int(layout["row_count"]), 1)),
        "exact_candidate_sets": bool(layout["exact_candidate_sets"]),
        "superset_candidate_sets": bool(layout["superset_candidate_sets"]),
        "extra_candidate_refs_vs_per_track_slab": int(layout["extra_candidate_refs_vs_per_track_slab"]),
        "indices_in_bounds": bool((ids >= 0).all().item() and (ids < boundary_count).all().item()) if ids.numel() else True,
    }


def _csr_valid(layout: dict[str, Any], *, boundary_count: int) -> dict[str, bool]:
    offsets = layout["row_offsets"]
    ids = layout["candidate_ids"]
    return {
        "offsets_monotonic": bool((offsets[1:] >= offsets[:-1]).all().item()),
        "last_offset_matches_index_count": bool(int(offsets[-1].item()) == int(ids.numel())),
        "indices_in_bounds": bool((ids >= 0).all().item() and (ids < boundary_count).all().item()) if ids.numel() else True,
    }


def _run_layout(
    *,
    layout: dict[str, Any],
    sites_f32: torch.Tensor,
    boundary_f32: torch.Tensor,
    site_rgba_f32: torch.Tensor,
    bundle: dict[str, Any],
    frame_count: int,
    config: RealRayReplayConfig,
    grad_rgb_f32: torch.Tensor,
    grad_alpha_f32: torch.Tensor,
    grad_depth_f32: torch.Tensor,
    timing_iters: int,
) -> dict[str, Any]:
    device = torch.device("mps")
    frame_t_f32 = _frame_times(frame_count, device)
    track_rays_f32 = bundle["track_rays"].to(device)
    row_index_i32 = layout["row_index"].to(device)
    offsets_i32 = layout["row_offsets"].to(device)
    ids_i32 = layout["candidate_ids"].to(device)
    output_rgb, output_alpha, output_depth, grad_site_rgba = shared_realray_rgba_depth_vjp_reduce_csr(
        boundary_f32,
        row_index_i32,
        offsets_i32,
        ids_i32,
        sites_f32,
        site_rgba_f32,
        track_rays_f32,
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config,
        time_slab_count=int(bundle["candidate_mask"].shape[0] // bundle["track_rays"].shape[0]),
        row_count=int(layout["row_count"]),
    )
    timed = (output_rgb, output_alpha, output_depth, grad_site_rgba)
    started_at = time.perf_counter()
    for _ in range(timing_iters):
        timed = shared_realray_rgba_depth_vjp_reduce_csr(
            boundary_f32,
            row_index_i32,
            offsets_i32,
            ids_i32,
            sites_f32,
            site_rgba_f32,
            track_rays_f32,
            frame_t_f32,
            grad_rgb_f32,
            grad_alpha_f32,
            grad_depth_f32,
            config,
            time_slab_count=int(bundle["candidate_mask"].shape[0] // bundle["track_rays"].shape[0]),
            row_count=int(layout["row_count"]),
        )
    torch.mps.synchronize()
    _timed_shapes = tuple(tensor.shape for tensor in timed)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0 / float(timing_iters)
    return {
        "rgb": output_rgb.cpu(),
        "alpha": output_alpha.cpu(),
        "depth": output_depth.cpu(),
        "grad_site_rgba": grad_site_rgba.cpu(),
        "mps_shared_realray_csr_reduced_vjp_wall_clock_ms": float(elapsed_ms),
    }


def _run_bitset(
    *,
    sites_f32: torch.Tensor,
    boundary_f32: torch.Tensor,
    site_rgba_f32: torch.Tensor,
    bundle: dict[str, Any],
    frame_count: int,
    config: RealRayReplayConfig,
    grad_rgb_f32: torch.Tensor,
    grad_alpha_f32: torch.Tensor,
    grad_depth_f32: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device("mps")
    frame_t_f32 = _frame_times(frame_count, device)
    return_values = shared_realray_rgba_depth_vjp_reduce(
        boundary_f32,
        bundle["candidate_mask"].to(device),
        sites_f32,
        site_rgba_f32,
        bundle["track_rays"].to(device),
        frame_t_f32,
        grad_rgb_f32,
        grad_alpha_f32,
        grad_depth_f32,
        config,
    )
    output_rgb, output_alpha, output_depth, grad_site_rgba = return_values
    return {
        "rgb": output_rgb.cpu(),
        "alpha": output_alpha.cpu(),
        "depth": output_depth.cpu(),
        "grad_site_rgba": grad_site_rgba.cpu(),
    }


def _split_summary(
    *,
    split: str,
    bundle: dict[str, Any],
    bitset: dict[str, Any],
    per_track: dict[str, Any],
    tiled: dict[str, Any],
    direct_cpu: dict[str, Any],
    per_track_layout: dict[str, Any],
    tiled_layout: dict[str, Any],
    boundary_count: int,
    time_slabs: int,
) -> dict[str, Any]:
    view_count = int(bundle["view_count"])
    frame_count = int(bitset["alpha"].shape[1])
    height = int(bundle["height"])
    width = int(bundle["width"])
    bitset_rgb_image = _reshape_rgb(
        bitset["rgb"],
        view_count=view_count,
        frame_count=frame_count,
        height=height,
        width=width,
    )
    bitset_bytes = int(bundle["candidate_mask"].numel() * bundle["candidate_mask"].element_size())
    per_track_stats = _csr_stats(per_track_layout, bundle=bundle, boundary_count=boundary_count, bitset_bytes=bitset_bytes)
    tiled_stats = _csr_stats(tiled_layout, bundle=bundle, boundary_count=boundary_count, bitset_bytes=bitset_bytes)
    return {
        "split": split,
        "rgb_shape": list(bitset_rgb_image.shape),
        "pixel_tracks": int(bundle["pixel_tracks"]),
        "pixel_rays": int(bundle["pixel_rays"]),
        "candidate_mask_shape": bundle["candidate_mask_shape"],
        "mask_word_count": int(bundle["word_count"]),
        "bitset_storage_bytes": bitset_bytes,
        "per_track_csr": per_track_stats,
        "tiled_csr": tiled_stats,
        "tile_shape": tiled_layout.get("tile_shape"),
        "tile_grid_shape": tiled_layout.get("tile_grid_shape"),
        "per_frame_event_sum": int(bundle["per_frame_event_sum"]),
        "shared_slab_event_sum": int(bundle["shared_slab_event_sum"]),
        "event_sharing_ratio": float(bundle["event_sharing_ratio"]),
        "missing_sample_events": int(bundle["missing_sample_events"]),
        "extra_candidate_events": int(bundle["extra_candidate_events"]),
        "max_candidates_per_slab": int(bundle["max_candidates_per_slab"]),
        "direct_forward_boundary_scans": int(bundle["direct_forward_boundary_scans"]),
        "shared_forward_boundary_scans": int(bundle["shared_forward_boundary_scans"]),
        "shared_forward_boundary_scan_ratio": float(bundle["shared_forward_boundary_scan_ratio"]),
        "per_track_max_csr_vs_bitset_rgb_abs_error": float((per_track["rgb"] - bitset["rgb"]).abs().max().item()),
        "per_track_max_csr_vs_bitset_alpha_abs_error": float((per_track["alpha"] - bitset["alpha"]).abs().max().item()),
        "per_track_max_csr_vs_bitset_depth_abs_error": float((per_track["depth"] - bitset["depth"]).abs().max().item()),
        "per_track_max_csr_vs_bitset_rgba_gradient_abs_error": float(
            (per_track["grad_site_rgba"] - bitset["grad_site_rgba"]).abs().max().item()
        ),
        "tiled_max_csr_vs_bitset_rgb_abs_error": float((tiled["rgb"] - bitset["rgb"]).abs().max().item()),
        "tiled_max_csr_vs_bitset_alpha_abs_error": float((tiled["alpha"] - bitset["alpha"]).abs().max().item()),
        "tiled_max_csr_vs_bitset_depth_abs_error": float((tiled["depth"] - bitset["depth"]).abs().max().item()),
        "tiled_max_csr_vs_bitset_rgba_gradient_abs_error": float(
            (tiled["grad_site_rgba"] - bitset["grad_site_rgba"]).abs().max().item()
        ),
        "max_bitset_rgb_abs_error_vs_cpu": float((bitset_rgb_image - direct_cpu["rgb"]).abs().max().item()),
        "per_track_mps_shared_realray_csr_reduced_vjp_wall_clock_ms": per_track[
            "mps_shared_realray_csr_reduced_vjp_wall_clock_ms"
        ],
        "tiled_mps_shared_realray_csr_reduced_vjp_wall_clock_ms": tiled[
            "mps_shared_realray_csr_reduced_vjp_wall_clock_ms"
        ],
        "partial_gradient_bytes": int(
            ((int(bundle["pixel_rays"]) + REDUCTION_CHUNK_SIZE - 1) // REDUCTION_CHUNK_SIZE)
            * int(bitset["grad_site_rgba"].shape[0])
            * 4
            * 4
        ),
        "reduced_gradient_bytes": int(bitset["grad_site_rgba"].numel() * bitset["grad_site_rgba"].element_size()),
        "per_track_csr_valid": _csr_valid(per_track_layout, boundary_count=boundary_count),
        "tiled_csr_valid": _csr_valid(tiled_layout, boundary_count=boundary_count),
        "time_slabs": time_slabs,
    }


def run_smoke(
    *,
    config_path: Path,
    max_frames: int | None,
    render_size: int | None,
    site_count: int,
    time_slabs: int,
    tile_h: int,
    tile_w: int,
    near: float,
    far: float,
    density: float,
    invalid_epsilon: float,
    transmittance_threshold: float,
    timing_iters: int,
) -> dict[str, Any]:
    if timing_iters <= 0:
        raise ValueError("timing_iters must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    cfg = _load_config(config_path, max_frames=max_frames, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    sample_rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    sample_frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    heldout_targets = data["heldout_targets"]
    heldout_rays = data["heldout_rays"]
    heldout_frame_indices = data["heldout_frame_indices"]
    if heldout_targets is None or heldout_rays is None or heldout_frame_indices is None:
        raise ValueError("CSR candidate storage smoke requires heldout targets, rays, and frame indices")

    frame_count = int(data["frame_count"])
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=sample_rays,
        frame_indices=sample_frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
    )
    boundaries = make_boundaries_4d(sites)
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    device = torch.device("mps")
    sites_f32 = torch.tensor([[site.x, site.y, site.z, site.t, site.weight] for site in sites], dtype=torch.float32, device=device)
    boundary_f32 = torch.tensor(
        [[boundary.nx, boundary.ny, boundary.nz, boundary.nt, boundary.b] for boundary in boundaries],
        dtype=torch.float32,
        device=device,
    )
    site_rgba_f32 = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)

    train_bundle = _build_candidate_bundle(
        boundaries=boundaries,
        rays=sample_rays,
        frame_indices=sample_frame_indices,
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        split="train",
    )
    heldout_bundle = _build_candidate_bundle(
        boundaries=boundaries,
        rays=heldout_rays.detach().cpu().to(dtype=torch.float32),
        frame_indices=heldout_frame_indices.detach().cpu().to(dtype=torch.long),
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        split="heldout",
    )

    cpu_train_render = render_samples(
        sites=sites,
        boundaries=boundaries,
        rays=sample_rays,
        frame_indices=sample_frame_indices,
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    cpu_heldout_render = render_samples(
        sites=sites,
        boundaries=boundaries,
        rays=heldout_rays.detach().cpu().to(dtype=torch.float32),
        frame_indices=heldout_frame_indices.detach().cpu().to(dtype=torch.long),
        frame_count=frame_count,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )

    train_grad_rgb, train_grad_alpha, train_grad_depth = _make_gradients(
        track_count=int(train_bundle["track_rays"].shape[0]),
        frame_count=frame_count,
    )
    heldout_grad_rgb, heldout_grad_alpha, heldout_grad_depth = _make_gradients(
        track_count=int(heldout_bundle["track_rays"].shape[0]),
        frame_count=frame_count,
    )
    train_grad_rgb_mps = train_grad_rgb.to(device)
    train_grad_alpha_mps = train_grad_alpha.to(device)
    train_grad_depth_mps = train_grad_depth.to(device)
    heldout_grad_rgb_mps = heldout_grad_rgb.to(device)
    heldout_grad_alpha_mps = heldout_grad_alpha.to(device)
    heldout_grad_depth_mps = heldout_grad_depth.to(device)

    train_bitset = _run_bitset(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=train_bundle,
        frame_count=frame_count,
        config=op_config,
        grad_rgb_f32=train_grad_rgb_mps,
        grad_alpha_f32=train_grad_alpha_mps,
        grad_depth_f32=train_grad_depth_mps,
    )
    heldout_bitset = _run_bitset(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=heldout_bundle,
        frame_count=frame_count,
        config=op_config,
        grad_rgb_f32=heldout_grad_rgb_mps,
        grad_alpha_f32=heldout_grad_alpha_mps,
        grad_depth_f32=heldout_grad_depth_mps,
    )
    train_per_track_layout = _build_per_track_csr(train_bundle, time_slabs=time_slabs)
    heldout_per_track_layout = _build_per_track_csr(heldout_bundle, time_slabs=time_slabs)
    train_tiled_layout = _build_tiled_csr(train_bundle, time_slabs=time_slabs, tile_h=tile_h, tile_w=tile_w)
    heldout_tiled_layout = _build_tiled_csr(heldout_bundle, time_slabs=time_slabs, tile_h=tile_h, tile_w=tile_w)

    train_per_track = _run_layout(
        layout=train_per_track_layout,
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=train_bundle,
        frame_count=frame_count,
        config=op_config,
        grad_rgb_f32=train_grad_rgb_mps,
        grad_alpha_f32=train_grad_alpha_mps,
        grad_depth_f32=train_grad_depth_mps,
        timing_iters=timing_iters,
    )
    train_tiled = _run_layout(
        layout=train_tiled_layout,
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=train_bundle,
        frame_count=frame_count,
        config=op_config,
        grad_rgb_f32=train_grad_rgb_mps,
        grad_alpha_f32=train_grad_alpha_mps,
        grad_depth_f32=train_grad_depth_mps,
        timing_iters=timing_iters,
    )
    heldout_per_track = _run_layout(
        layout=heldout_per_track_layout,
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=heldout_bundle,
        frame_count=frame_count,
        config=op_config,
        grad_rgb_f32=heldout_grad_rgb_mps,
        grad_alpha_f32=heldout_grad_alpha_mps,
        grad_depth_f32=heldout_grad_depth_mps,
        timing_iters=timing_iters,
    )
    heldout_tiled = _run_layout(
        layout=heldout_tiled_layout,
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=heldout_bundle,
        frame_count=frame_count,
        config=op_config,
        grad_rgb_f32=heldout_grad_rgb_mps,
        grad_alpha_f32=heldout_grad_alpha_mps,
        grad_depth_f32=heldout_grad_depth_mps,
        timing_iters=timing_iters,
    )

    train = _split_summary(
        split="train",
        bundle=train_bundle,
        bitset=train_bitset,
        per_track=train_per_track,
        tiled=train_tiled,
        direct_cpu=cpu_train_render,
        per_track_layout=train_per_track_layout,
        tiled_layout=train_tiled_layout,
        boundary_count=len(boundaries),
        time_slabs=time_slabs,
    )
    heldout = _split_summary(
        split="heldout",
        bundle=heldout_bundle,
        bitset=heldout_bitset,
        per_track=heldout_per_track,
        tiled=heldout_tiled,
        direct_cpu=cpu_heldout_render,
        per_track_layout=heldout_per_track_layout,
        tiled_layout=heldout_tiled_layout,
        boundary_count=len(boundaries),
        time_slabs=time_slabs,
    )
    tolerance = 5.0e-4
    max_forward_error = max(
        train["per_track_max_csr_vs_bitset_rgb_abs_error"],
        train["per_track_max_csr_vs_bitset_alpha_abs_error"],
        train["per_track_max_csr_vs_bitset_depth_abs_error"],
        train["tiled_max_csr_vs_bitset_rgb_abs_error"],
        train["tiled_max_csr_vs_bitset_alpha_abs_error"],
        train["tiled_max_csr_vs_bitset_depth_abs_error"],
        heldout["per_track_max_csr_vs_bitset_rgb_abs_error"],
        heldout["per_track_max_csr_vs_bitset_alpha_abs_error"],
        heldout["per_track_max_csr_vs_bitset_depth_abs_error"],
        heldout["tiled_max_csr_vs_bitset_rgb_abs_error"],
        heldout["tiled_max_csr_vs_bitset_alpha_abs_error"],
        heldout["tiled_max_csr_vs_bitset_depth_abs_error"],
    )
    max_cpu_error = max(train["max_bitset_rgb_abs_error_vs_cpu"], heldout["max_bitset_rgb_abs_error_vs_cpu"])
    max_grad_error = max(
        train["per_track_max_csr_vs_bitset_rgba_gradient_abs_error"],
        train["tiled_max_csr_vs_bitset_rgba_gradient_abs_error"],
        heldout["per_track_max_csr_vs_bitset_rgba_gradient_abs_error"],
        heldout["tiled_max_csr_vs_bitset_rgba_gradient_abs_error"],
    )
    acceptance = {
        "csr_offsets_monotonic": train["per_track_csr_valid"]["offsets_monotonic"]
        and train["tiled_csr_valid"]["offsets_monotonic"]
        and heldout["per_track_csr_valid"]["offsets_monotonic"]
        and heldout["tiled_csr_valid"]["offsets_monotonic"],
        "csr_last_offset_matches_index_count": train["per_track_csr_valid"]["last_offset_matches_index_count"]
        and train["tiled_csr_valid"]["last_offset_matches_index_count"]
        and heldout["per_track_csr_valid"]["last_offset_matches_index_count"]
        and heldout["tiled_csr_valid"]["last_offset_matches_index_count"],
        "csr_indices_in_bounds": train["per_track_csr_valid"]["indices_in_bounds"]
        and train["tiled_csr_valid"]["indices_in_bounds"]
        and heldout["per_track_csr_valid"]["indices_in_bounds"]
        and heldout["tiled_csr_valid"]["indices_in_bounds"],
        "csr_matches_bitset_candidates": train["per_track_csr"]["exact_candidate_sets"]
        and train["tiled_csr"]["superset_candidate_sets"]
        and heldout["per_track_csr"]["exact_candidate_sets"]
        and heldout["tiled_csr"]["superset_candidate_sets"],
        "csr_forward_matches_bitset": max_forward_error <= tolerance and max_grad_error <= tolerance,
        "csr_forward_matches_cpu": max_cpu_error <= tolerance,
        "csr_storage_ratio_below_bitset": train["tiled_csr"]["storage_vs_bitset_ratio"] < 1.0
        and heldout["tiled_csr"]["storage_vs_bitset_ratio"] < 1.0,
        "no_missing_sample_events": train["missing_sample_events"] == 0 and heldout["missing_sample_events"] == 0,
        "outputs_are_finite": bool(
            torch.isfinite(train_per_track["rgb"]).all().item()
            and torch.isfinite(train_tiled["rgb"]).all().item()
            and torch.isfinite(heldout_per_track["rgb"]).all().item()
            and torch.isfinite(heldout_tiled["rgb"]).all().item()
        ),
    }
    return {
        "benchmark": "world_foam_lane2_gate2f_mps_shared_realray_csr_candidate_storage_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "2F_realray_mps_shared_csr_candidate_storage",
        "device": "mps",
        "config_path": str(config_path),
        "sample_id": data["source_label"],
        "train_views": list(data["train_views"]),
        "heldout_views": list(data["heldout_views"]),
        "pose_source": data["pose_source"],
        "frame_counts": [frame_count],
        "frame_count": frame_count,
        "render_size": int(cfg["render"]["render_size"]),
        "time_slabs": time_slabs,
        "site_count": site_count,
        "boundary_count": len(boundaries),
        "near": float(near),
        "far": float(far),
        "density": float(density),
        "renderer_scope": "mps_real_camera_ray_4d_power_cell_time_slab_shared_forward_and_reduced_fixed_segment_vjp",
        "gradient_scope": "frozen_geometry_reduced_site_rgba_only_no_geometry_or_topology_gradients",
        "sharing_scope": "mps_real_camera_ray_time_slab_candidate_csr_storage",
        "quality_claim": False,
        "training_claim": False,
        "csr_candidate_storage_claim": True,
        "candidate_storage_format": "per_track_csr_and_tiled_csr_i32",
        "bitset_reference_storage_format": "int32_bitset_words",
        "csr_offset_dtype": "int32",
        "csr_index_dtype": "int32",
        "world_foam_renderer_status": "mps_shared_real_camera_ray_csr_candidate_storage_smoke_no_quality_claim",
        "tile_shape": [tile_h, tile_w],
        "tolerance": tolerance,
        "timing_iters": timing_iters,
        "train": train,
        "heldout": heldout,
        "acceptance": acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke World Foam shared real-ray CSR candidate storage.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--render-size", type=int)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--timing-iters", type=int, default=5)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke(
        config_path=args.config,
        max_frames=args.max_frames,
        render_size=args.render_size,
        site_count=args.site_count,
        time_slabs=args.time_slabs,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        near=args.near,
        far=args.far,
        density=args.density,
        invalid_epsilon=args.invalid_epsilon,
        transmittance_threshold=args.transmittance_threshold,
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
