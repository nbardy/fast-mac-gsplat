#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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
)
from smoke_shared_realray_csr_candidate_storage_mps import (  # noqa: E402
    _build_per_track_csr,
    _build_tiled_csr,
    _csr_stats,
    _csr_valid,
    _run_bitset,
    _run_layout,
)
from smoke_shared_realray_vjp_mps import _build_candidate_bundle, _make_gradients  # noqa: E402
from torch_world_foam_lane2_fused_slab import RealRayReplayConfig  # noqa: E402


def parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one integer")
    if sorted(out) != list(out):
        raise ValueError("frame counts must be sorted ascending")
    return out


def _scene_for_frame_count(
    *,
    config_path: Path,
    frame_count: int,
    render_size: int,
    site_count: int,
    near: float,
    far: float,
    density: float,
) -> dict[str, Any]:
    cfg = _load_config(config_path, max_frames=frame_count, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    sample_rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    sample_frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    heldout_targets = data["heldout_targets"]
    heldout_rays = data["heldout_rays"]
    heldout_frame_indices = data["heldout_frame_indices"]
    if heldout_targets is None or heldout_rays is None or heldout_frame_indices is None:
        raise ValueError("CSR scaling smoke requires heldout targets, rays, and frame indices")
    loaded_frame_count = int(data["frame_count"])
    if loaded_frame_count != frame_count:
        raise ValueError(f"requested {frame_count} frames but loader returned {loaded_frame_count}")
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
    return {
        "cfg": cfg,
        "data": data,
        "sites": sites,
        "boundaries": make_boundaries_4d(sites),
        "targets": targets,
        "sample_rays": sample_rays,
        "sample_frame_indices": sample_frame_indices,
        "heldout_rays": heldout_rays.detach().cpu().to(dtype=torch.float32),
        "heldout_frame_indices": heldout_frame_indices.detach().cpu().to(dtype=torch.long),
    }


def _layout_accounting(
    *,
    bundle: dict[str, Any],
    per_track_layout: dict[str, Any],
    tiled_layout: dict[str, Any],
    boundary_count: int,
    split: str,
) -> dict[str, Any]:
    bitset_bytes = int(bundle["candidate_mask"].numel() * bundle["candidate_mask"].element_size())
    per_track = _csr_stats(per_track_layout, bundle=bundle, boundary_count=boundary_count, bitset_bytes=bitset_bytes)
    tiled = _csr_stats(tiled_layout, bundle=bundle, boundary_count=boundary_count, bitset_bytes=bitset_bytes)
    direct_scans = int(bundle["direct_forward_boundary_scans"])
    return {
        "split": split,
        "pixel_tracks": int(bundle["pixel_tracks"]),
        "pixel_rays": int(bundle["pixel_rays"]),
        "candidate_mask_shape": bundle["candidate_mask_shape"],
        "mask_word_count": int(bundle["word_count"]),
        "bitset_storage_bytes": bitset_bytes,
        "per_track_csr": per_track,
        "tiled_csr": tiled,
        "per_track_csr_valid": _csr_valid(per_track_layout, boundary_count=boundary_count),
        "tiled_csr_valid": _csr_valid(tiled_layout, boundary_count=boundary_count),
        "tile_shape": tiled_layout.get("tile_shape"),
        "tile_grid_shape": tiled_layout.get("tile_grid_shape"),
        "per_frame_event_sum": int(bundle["per_frame_event_sum"]),
        "shared_slab_event_sum": int(bundle["shared_slab_event_sum"]),
        "event_sharing_ratio": float(bundle["event_sharing_ratio"]),
        "missing_sample_events": int(bundle["missing_sample_events"]),
        "extra_candidate_events": int(bundle["extra_candidate_events"]),
        "max_candidates_per_slab": int(bundle["max_candidates_per_slab"]),
        "direct_forward_boundary_scans": direct_scans,
        "shared_forward_boundary_scans": int(bundle["shared_forward_boundary_scans"]),
        "shared_forward_boundary_scan_ratio": float(bundle["shared_forward_boundary_scan_ratio"]),
        "tiled_candidate_iteration_vs_direct_scan_ratio": float(tiled["candidate_iterations"]) / float(max(direct_scans, 1)),
        "per_track_candidate_iteration_vs_direct_scan_ratio": float(per_track["candidate_iterations"])
        / float(max(direct_scans, 1)),
    }


def _run_tiled_mps_check(
    *,
    sites_f32: torch.Tensor,
    boundary_f32: torch.Tensor,
    site_rgba_f32: torch.Tensor,
    bundle: dict[str, Any],
    tiled_layout: dict[str, Any],
    frame_count: int,
    config: RealRayReplayConfig,
    timing_iters: int,
) -> dict[str, Any]:
    grad_rgb, grad_alpha, grad_depth = _make_gradients(
        track_count=int(bundle["track_rays"].shape[0]),
        frame_count=frame_count,
    )
    device = torch.device("mps")
    bitset = _run_bitset(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=bundle,
        frame_count=frame_count,
        config=config,
        grad_rgb_f32=grad_rgb.to(device),
        grad_alpha_f32=grad_alpha.to(device),
        grad_depth_f32=grad_depth.to(device),
    )
    tiled = _run_layout(
        layout=tiled_layout,
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=bundle,
        frame_count=frame_count,
        config=config,
        grad_rgb_f32=grad_rgb.to(device),
        grad_alpha_f32=grad_alpha.to(device),
        grad_depth_f32=grad_depth.to(device),
        timing_iters=timing_iters,
    )
    return {
        "tiled_max_csr_vs_bitset_rgb_abs_error": float((tiled["rgb"] - bitset["rgb"]).abs().max().item()),
        "tiled_max_csr_vs_bitset_alpha_abs_error": float((tiled["alpha"] - bitset["alpha"]).abs().max().item()),
        "tiled_max_csr_vs_bitset_depth_abs_error": float((tiled["depth"] - bitset["depth"]).abs().max().item()),
        "tiled_max_csr_vs_bitset_rgba_gradient_abs_error": float(
            (tiled["grad_site_rgba"] - bitset["grad_site_rgba"]).abs().max().item()
        ),
        "tiled_mps_shared_realray_csr_reduced_vjp_wall_clock_ms": tiled[
            "mps_shared_realray_csr_reduced_vjp_wall_clock_ms"
        ],
        "reduced_gradient_bytes": int(bitset["grad_site_rgba"].numel() * bitset["grad_site_rgba"].element_size()),
        "outputs_are_finite": bool(
            torch.isfinite(tiled["rgb"]).all().item()
            and torch.isfinite(tiled["alpha"]).all().item()
            and torch.isfinite(tiled["depth"]).all().item()
            and torch.isfinite(tiled["grad_site_rgba"]).all().item()
        ),
    }


def _profile_frame_count(
    *,
    config_path: Path,
    frame_count: int,
    render_size: int,
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
    scene = _scene_for_frame_count(
        config_path=config_path,
        frame_count=frame_count,
        render_size=render_size,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
    )
    data = scene["data"]
    sites = scene["sites"]
    boundaries = scene["boundaries"]
    train_bundle = _build_candidate_bundle(
        boundaries=boundaries,
        rays=scene["sample_rays"],
        frame_indices=scene["sample_frame_indices"],
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        split="train",
    )
    heldout_bundle = _build_candidate_bundle(
        boundaries=boundaries,
        rays=scene["heldout_rays"],
        frame_indices=scene["heldout_frame_indices"],
        frame_count=frame_count,
        time_slabs=time_slabs,
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        split="heldout",
    )
    train_per_track = _build_per_track_csr(train_bundle, time_slabs=time_slabs)
    train_tiled = _build_tiled_csr(train_bundle, time_slabs=time_slabs, tile_h=tile_h, tile_w=tile_w)
    heldout_per_track = _build_per_track_csr(heldout_bundle, time_slabs=time_slabs)
    heldout_tiled = _build_tiled_csr(heldout_bundle, time_slabs=time_slabs, tile_h=tile_h, tile_w=tile_w)
    boundary_count = len(boundaries)
    train = _layout_accounting(
        bundle=train_bundle,
        per_track_layout=train_per_track,
        tiled_layout=train_tiled,
        boundary_count=boundary_count,
        split="train",
    )
    heldout = _layout_accounting(
        bundle=heldout_bundle,
        per_track_layout=heldout_per_track,
        tiled_layout=heldout_tiled,
        boundary_count=boundary_count,
        split="heldout",
    )
    device = torch.device("mps")
    sites_f32 = torch.tensor([[site.x, site.y, site.z, site.t, site.weight] for site in sites], dtype=torch.float32, device=device)
    boundary_f32 = torch.tensor(
        [[boundary.nx, boundary.ny, boundary.nz, boundary.nt, boundary.b] for boundary in boundaries],
        dtype=torch.float32,
        device=device,
    )
    site_rgba_f32 = torch.tensor([site.rgba for site in sites], dtype=torch.float32, device=device)
    op_config = RealRayReplayConfig(
        near=near,
        far=far,
        invalid_epsilon=invalid_epsilon,
        transmittance_threshold=transmittance_threshold,
    )
    train_mps = _run_tiled_mps_check(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=train_bundle,
        tiled_layout=train_tiled,
        frame_count=frame_count,
        config=op_config,
        timing_iters=timing_iters,
    )
    heldout_mps = _run_tiled_mps_check(
        sites_f32=sites_f32,
        boundary_f32=boundary_f32,
        site_rgba_f32=site_rgba_f32,
        bundle=heldout_bundle,
        tiled_layout=heldout_tiled,
        frame_count=frame_count,
        config=op_config,
        timing_iters=timing_iters,
    )
    train.update(train_mps)
    heldout.update(heldout_mps)
    return {
        "frames": frame_count,
        "sample_id": data["source_label"],
        "train_views": list(data["train_views"]),
        "heldout_views": list(data["heldout_views"]),
        "pose_source": data["pose_source"],
        "render_size": int(scene["cfg"]["render"]["render_size"]),
        "time_slabs": time_slabs,
        "site_count": len(sites),
        "boundary_count": boundary_count,
        "train": train,
        "heldout": heldout,
    }


def _growth(first: dict[str, Any], last: dict[str, Any], *, split: str) -> dict[str, Any]:
    first_split = first[split]
    last_split = last[split]
    direct_growth = float(last_split["direct_forward_boundary_scans"]) / float(
        max(int(first_split["direct_forward_boundary_scans"]), 1)
    )
    shared_growth = float(last_split["shared_forward_boundary_scans"]) / float(
        max(int(first_split["shared_forward_boundary_scans"]), 1)
    )
    tiled_iter_growth = float(last_split["tiled_csr"]["candidate_iterations"]) / float(
        max(int(first_split["tiled_csr"]["candidate_iterations"]), 1)
    )
    bitset_storage_growth = float(last_split["bitset_storage_bytes"]) / float(max(int(first_split["bitset_storage_bytes"]), 1))
    tiled_storage_growth = float(last_split["tiled_csr"]["storage_bytes"]) / float(
        max(int(first_split["tiled_csr"]["storage_bytes"]), 1)
    )
    return {
        "split": split,
        "from_frames": int(first["frames"]),
        "to_frames": int(last["frames"]),
        "direct_scan_growth": direct_growth,
        "shared_scan_growth": shared_growth,
        "tiled_candidate_iteration_growth": tiled_iter_growth,
        "bitset_storage_growth": bitset_storage_growth,
        "tiled_csr_storage_growth": tiled_storage_growth,
        "shared_scans_sublinear_vs_direct": shared_growth < direct_growth,
        "tiled_candidate_iterations_sublinear_vs_direct": tiled_iter_growth < direct_growth,
    }


def run_benchmark(
    *,
    config_path: Path,
    frame_counts: tuple[int, ...],
    render_size: int,
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
    rows = [
        _profile_frame_count(
            config_path=config_path,
            frame_count=frame_count,
            render_size=render_size,
            site_count=site_count,
            time_slabs=time_slabs,
            tile_h=tile_h,
            tile_w=tile_w,
            near=near,
            far=far,
            density=density,
            invalid_epsilon=invalid_epsilon,
            transmittance_threshold=transmittance_threshold,
            timing_iters=timing_iters,
        )
        for frame_count in frame_counts
    ]
    tolerance = 5.0e-4
    mps_errors = [
        row[split][field]
        for row in rows
        for split in ("train", "heldout")
        for field in (
            "tiled_max_csr_vs_bitset_rgb_abs_error",
            "tiled_max_csr_vs_bitset_alpha_abs_error",
            "tiled_max_csr_vs_bitset_depth_abs_error",
            "tiled_max_csr_vs_bitset_rgba_gradient_abs_error",
        )
    ]
    growth = {
        "train": _growth(rows[0], rows[-1], split="train"),
        "heldout": _growth(rows[0], rows[-1], split="heldout"),
    }
    acceptance = {
        "all_rows_zero_missing": all(
            int(row[split]["missing_sample_events"]) == 0 for row in rows for split in ("train", "heldout")
        ),
        "all_csr_rows_valid": all(
            all(row[split][layout][key] for key in ("offsets_monotonic", "last_offset_matches_index_count", "indices_in_bounds"))
            for row in rows
            for split in ("train", "heldout")
            for layout in ("per_track_csr_valid", "tiled_csr_valid")
        ),
        "tiled_csr_matches_bitset_mps": max(mps_errors) <= tolerance,
        "tiled_csr_storage_below_bitset": all(
            float(row[split]["tiled_csr"]["storage_vs_bitset_ratio"]) < 1.0
            for row in rows
            for split in ("train", "heldout")
        ),
        "shared_scan_growth_sublinear": growth["train"]["shared_scans_sublinear_vs_direct"]
        and growth["heldout"]["shared_scans_sublinear_vs_direct"],
        "tiled_candidate_iterations_sublinear": growth["train"]["tiled_candidate_iterations_sublinear_vs_direct"]
        and growth["heldout"]["tiled_candidate_iterations_sublinear_vs_direct"],
        "outputs_are_finite": all(bool(row[split]["outputs_are_finite"]) for row in rows for split in ("train", "heldout")),
    }
    return {
        "benchmark": "world_foam_lane2_gate2g_mps_shared_realray_csr_scaling_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "2G_realray_mps_shared_csr_scaling",
        "device": "mps",
        "config_path": str(config_path),
        "frame_counts": list(frame_counts),
        "render_size": render_size,
        "time_slabs": time_slabs,
        "site_count": site_count,
        "near": near,
        "far": far,
        "density": density,
        "tile_shape": [tile_h, tile_w],
        "timing_iters": timing_iters,
        "comparison_unit": "mps_real_camera_ray_time_slab_shared_csr_scaling",
        "renderer_scope": "mps_real_camera_ray_4d_power_cell_time_slab_shared_forward_and_reduced_fixed_segment_vjp",
        "gradient_scope": "frozen_geometry_reduced_site_rgba_only_no_geometry_or_topology_gradients",
        "sharing_scope": "mps_real_camera_ray_time_slab_candidate_csr_storage_scaling",
        "quality_claim": False,
        "training_claim": False,
        "large_scale_claim": False,
        "rows": rows,
        "growth": growth,
        "max_tiled_csr_vs_bitset_mps_error": max(mps_errors),
        "tolerance": tolerance,
        "acceptance": acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke World Foam shared real-ray CSR scaling on MPS.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-counts", default="2,4,8")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--far", type=float, default=3.25)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--timing-iters", type=int, default=3)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(
        config_path=args.config,
        frame_counts=parse_int_list(args.frame_counts),
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
