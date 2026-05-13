from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.trainer_harness.projective_rational import (  # noqa: E402
    WorldTubeBatch,
    centered_frame_times,
    compile_projective_rational_tubes,
    direct_project_world_tubes,
    evaluate_projective_centers,
    fit_camera_path_polynomial,
    projection_matrices,
)


def _psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1.0e-12))


def _camera(
    frames: int,
    width: int,
    height: int,
    times: Tensor,
    *,
    pan_x: float,
    zoom: float,
    dolly_z: float,
) -> tuple[Tensor, Tensor]:
    k = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(frames, 1, 1)
    k[:, 0, 0] = 0.9 * float(width)
    k[:, 1, 1] = 0.9 * float(width)
    k[:, 0, 2] = 0.5 * float(width)
    k[:, 1, 2] = 0.5 * float(height)
    zoom_factor = 1.0 + float(zoom) * times
    if float(zoom_factor.min()) <= 0.0:
        raise ValueError("zoom stress makes focal length non-positive")
    k[:, 0, 0] *= zoom_factor
    k[:, 1, 1] *= zoom_factor

    w2c = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(frames, 1, 1)
    w2c[:, 0, 3] = -float(pan_x) * times
    w2c[:, 2, 3] = -float(dolly_z) * times
    return k, w2c


def _world_tubes(tube_count: int, *, seed: int, velocity_scale: float) -> WorldTubeBatch:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x0 = torch.empty((tube_count, 3), dtype=torch.float32)
    x0[:, 0] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-1.4, 1.4, generator=generator)
    x0[:, 1] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-0.9, 0.9, generator=generator)
    x0[:, 2] = torch.empty((tube_count,), dtype=torch.float32).uniform_(3.8, 8.2, generator=generator)
    velocity = torch.empty((tube_count, 3), dtype=torch.float32)
    velocity[:, 0] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    velocity[:, 1] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-1.0, 1.0, generator=generator)
    velocity[:, 2] = torch.empty((tube_count,), dtype=torch.float32).uniform_(-0.5, 0.5, generator=generator)
    velocity = velocity * float(velocity_scale)
    return WorldTubeBatch(
        x0=x0,
        velocity=velocity,
        t0=torch.zeros((tube_count,), dtype=torch.float32),
        precision_xy=torch.empty((tube_count, 2), dtype=torch.float32).uniform_(36.0, 72.0, generator=generator),
        lambda_t=torch.empty((tube_count,), dtype=torch.float32).uniform_(0.03, 0.08, generator=generator),
        opacity=torch.empty((tube_count,), dtype=torch.float32).uniform_(0.35, 0.75, generator=generator),
        color=torch.empty((tube_count, 3), dtype=torch.float32).uniform_(0.05, 0.95, generator=generator),
    )


def _fit_poly(values: Tensor, times: Tensor, *, degree: int) -> tuple[Tensor, Tensor]:
    frames = int(values.shape[0])
    vand = torch.stack([times.pow(k) for k in range(degree + 1)], dim=-1)
    solution = torch.linalg.lstsq(vand.to(torch.float64), values.reshape(frames, -1).to(torch.float64)).solution
    coeff = solution.to(torch.float32).reshape(degree + 1, *values.shape[1:])
    recon = torch.einsum("fd,d...->f...", vand.to(torch.float32), coeff)
    return coeff, recon


def _project_world_points(p_seq: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
    ones = torch.ones((int(points.shape[0]), 1), dtype=torch.float32, device=points.device)
    hom = torch.cat((points, ones), dim=-1)
    h = torch.einsum("frc,nc->fnr", p_seq, hom)
    z = h[..., 2].clamp_min(1.0e-6)
    return torch.stack((h[..., 0] / z, h[..., 1] / z), dim=-1), z


def _depth_bands(ref_depth: Tensor, *, bands: int) -> tuple[Tensor, Tensor]:
    inv_depth = 1.0 / ref_depth.clamp_min(1.0e-6)
    lo = torch.quantile(inv_depth, 0.05)
    hi = torch.quantile(inv_depth, 0.95)
    band_inv = torch.linspace(float(lo), float(hi), bands, dtype=torch.float32)
    assignments = (inv_depth.view(-1, 1) - band_inv.view(1, -1)).abs().argmin(dim=1)
    return 1.0 / band_inv.clamp_min(1.0e-6), assignments


def _homography_flow_centers(
    ref_uv: Tensor,
    band_depths: Tensor,
    assignments: Tensor,
    k_seq: Tensor,
    w2c_seq: Tensor,
    *,
    ref_frame: int,
) -> Tensor:
    frames = int(k_seq.shape[0])
    tube_count = int(ref_uv.shape[0])
    p_seq = projection_matrices(k_seq, w2c_seq)
    k_ref_inv = torch.linalg.inv(k_seq[ref_frame])
    c2w_ref = torch.linalg.inv(w2c_seq[ref_frame])
    uv1 = torch.cat((ref_uv, torch.ones((tube_count, 1), dtype=torch.float32)), dim=-1)
    centers = torch.empty((frames, tube_count, 2), dtype=torch.float32)
    for band_id, depth in enumerate(band_depths.tolist()):
        mask = assignments == band_id
        if not bool(mask.any()):
            continue
        rays = uv1[mask] @ k_ref_inv.T
        cam_points = rays * float(depth)
        world_h = torch.cat((cam_points, torch.ones((int(cam_points.shape[0]), 1), dtype=torch.float32)), dim=-1) @ c2w_ref.T
        band_centers, _ = _project_world_points(p_seq, world_h[:, :3])
        centers[:, mask] = band_centers
    return centers


def _compile_prt_centers(
    batch: WorldTubeBatch,
    k_seq: Tensor,
    w2c_seq: Tensor,
    times: Tensor,
    *,
    degree: int,
) -> tuple[Tensor, Tensor, Tensor, float]:
    path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=degree, frame_times=times)
    projected = compile_projective_rational_tubes(batch, path)
    centers, depth = evaluate_projective_centers(projected, times)
    return centers, depth, projected.lambda_uv, path.fit_error


def _segmented_centers(
    batch: WorldTubeBatch,
    k_seq: Tensor,
    w2c_seq: Tensor,
    times: Tensor,
    *,
    segments: int,
    degree: int,
) -> tuple[Tensor, Tensor, list[float]]:
    frames = int(times.numel())
    centers = torch.empty((frames, int(batch.x0.shape[0]), 2), dtype=torch.float32)
    depth = torch.empty((frames, int(batch.x0.shape[0])), dtype=torch.float32)
    fit_errors = []
    for segment in range(segments):
        start = (frames * segment) // segments
        end = (frames * (segment + 1)) // segments
        seg_times = times[start:end]
        path = fit_camera_path_polynomial(k_seq[start:end], w2c_seq[start:end], degree=degree, frame_times=seg_times)
        projected = compile_projective_rational_tubes(batch, path)
        seg_centers, seg_depth = evaluate_projective_centers(projected, seg_times)
        centers[start:end] = seg_centers
        depth[start:end] = seg_depth
        fit_errors.append(path.fit_error)
    return centers, depth, fit_errors


def _center_metrics(approx: Tensor, reference: Tensor) -> dict[str, float]:
    err = (approx - reference).norm(dim=-1)
    return {
        "mean_px": float(err.mean().detach().cpu()),
        "p95_px": float(torch.quantile(err.flatten(), 0.95).detach().cpu()),
        "max_px": float(err.max().detach().cpu()),
    }


def _per_tube_max_error(approx: Tensor, reference: Tensor) -> Tensor:
    return (approx - reference).norm(dim=-1).max(dim=0).values


def _image_metrics(image: Tensor, reference: Tensor) -> dict[str, float]:
    diff = image - reference
    mse = float(diff.square().mean().detach().cpu())
    return {
        "mse": mse,
        "psnr": _psnr(mse),
        "l1": float(diff.abs().mean().detach().cpu()),
        "max_abs": float(diff.abs().max().detach().cpu()),
    }


def _render_from_centers(
    batch: WorldTubeBatch,
    centers: Tensor,
    depth: Tensor,
    lambda_uv: Tensor,
    times: Tensor,
    *,
    height: int,
    width: int,
    alpha_threshold: float,
) -> Tensor:
    y = torch.arange(height, dtype=torch.float32, device=centers.device) + 0.5
    x = torch.arange(width, dtype=torch.float32, device=centers.device) + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    frames = int(times.numel())
    image = torch.empty((frames, height, width, 3), dtype=torch.float32, device=centers.device)
    for frame in range(frames):
        order = torch.argsort(depth[frame].detach(), stable=True)
        accum = torch.zeros((height, width, 3), dtype=torch.float32, device=centers.device)
        trans = torch.ones((height, width, 1), dtype=torch.float32, device=centers.device)
        dt = times[frame].to(centers.device) - batch.t0.to(centers.device)
        for tube in order.tolist():
            lambda_uu, lambda_uv_cross, lambda_vv = lambda_uv[tube].to(centers.device)
            du = xx - centers[frame, tube, 0]
            dv = yy - centers[frame, tube, 1]
            spatial = lambda_uu * du.square() + 2.0 * lambda_uv_cross * du * dv + lambda_vv * dv.square()
            temporal = batch.lambda_t[tube].to(centers.device) * dt[tube].square()
            alpha = (batch.opacity[tube].to(centers.device) * torch.exp(-0.5 * (spatial + temporal))).clamp(max=0.99)
            alpha = torch.where(alpha >= alpha_threshold, alpha, torch.zeros_like(alpha))
            alpha3 = alpha.unsqueeze(-1)
            accum = accum + trans * alpha3 * batch.color[tube].to(centers.device).view(1, 1, 3)
            trans = trans * (1.0 - alpha3)
        image[frame] = accum
    return image


def _min_eigenvalue(lambda_uv: Tensor) -> Tensor:
    a = lambda_uv[:, 0]
    b = lambda_uv[:, 1]
    c = lambda_uv[:, 2]
    return (0.5 * (a + c - ((a - c).square() + 4.0 * b.square()).sqrt())).clamp_min(1.0e-6)


def _estimate_tile_pairs(
    centers: Tensor,
    batch: WorldTubeBatch,
    lambda_uv: Tensor,
    times: Tensor,
    *,
    width: int,
    height: int,
    tile_x: int,
    tile_y: int,
    tile_t: int,
    alpha_threshold: float,
    clamp_to_image: bool,
    tube_mask: Tensor | None = None,
) -> dict[str, int]:
    frames = int(times.numel())
    tube_count = int(centers.shape[1])
    tube_ids = range(tube_count)
    if tube_mask is not None:
        if tuple(tube_mask.shape) != (tube_count,):
            raise ValueError(f"tube_mask must have shape ({tube_count},)")
        tube_ids = [int(v) for v in torch.nonzero(tube_mask.detach().cpu(), as_tuple=False).flatten().tolist()]
    min_eig = _min_eigenvalue(lambda_uv)
    support_tau = 2.0 * torch.log((batch.opacity / float(alpha_threshold)).clamp_min(1.0 + 1.0e-6))
    total = 0
    active = 0
    max_pairs_per_tube_tile = 0
    for tube in tube_ids:
        for start in range(0, frames, tile_t):
            end = min(frames, start + tile_t)
            frame_slice = slice(start, end)
            dt = times[frame_slice] - batch.t0[tube]
            budget = support_tau[tube] - batch.lambda_t[tube] * dt.square()
            mask = budget > 0.0
            if not bool(mask.any()):
                continue
            frame_centers = centers[frame_slice, tube][mask]
            radius = (budget[mask] / min_eig[tube]).sqrt()
            min_u = float((frame_centers[:, 0] - radius).min())
            max_u = float((frame_centers[:, 0] + radius).max())
            min_v = float((frame_centers[:, 1] - radius).min())
            max_v = float((frame_centers[:, 1] + radius).max())
            tile_min_x = math.floor(min_u / tile_x)
            tile_max_x = math.floor(max_u / tile_x)
            tile_min_y = math.floor(min_v / tile_y)
            tile_max_y = math.floor(max_v / tile_y)
            if clamp_to_image:
                tile_min_x = max(0, min(tile_min_x, math.ceil(width / tile_x) - 1))
                tile_max_x = max(0, min(tile_max_x, math.ceil(width / tile_x) - 1))
                tile_min_y = max(0, min(tile_min_y, math.ceil(height / tile_y) - 1))
                tile_max_y = max(0, min(tile_max_y, math.ceil(height / tile_y) - 1))
            pairs = max(0, tile_max_x - tile_min_x + 1) * max(0, tile_max_y - tile_min_y + 1)
            if pairs > 0:
                active += 1
                total += pairs
                max_pairs_per_tube_tile = max(max_pairs_per_tube_tile, pairs)
    return {
        "total_tile_pairs": int(total),
        "active_tube_time_tiles": int(active),
        "max_pairs_per_tube_time_tile": int(max_pairs_per_tube_tile),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    times = centered_frame_times(args.frames)
    k_seq, w2c_seq = _camera(
        args.frames,
        args.target_size,
        args.target_size,
        times,
        pan_x=args.pan_x,
        zoom=args.zoom,
        dolly_z=args.dolly_z,
    )
    batch = _world_tubes(args.tubes, seed=args.seed, velocity_scale=args.velocity_scale)

    direct_centers, direct_depth = direct_project_world_tubes(batch, k_seq, w2c_seq, times)
    prt_centers, prt_depth, lambda_uv, prt_fit_error = _compile_prt_centers(batch, k_seq, w2c_seq, times, degree=args.prt_degree)
    first_centers, first_depth, _, first_fit_error = _compile_prt_centers(batch, k_seq, w2c_seq, times, degree=1)
    segmented_centers, segmented_depth, segmented_fit_errors = _segmented_centers(
        batch,
        k_seq,
        w2c_seq,
        times,
        segments=args.segments,
        degree=1,
    )

    ref_frame = args.frames // 2
    band_depths, assignments = _depth_bands(direct_depth[ref_frame], bands=args.depth_bands)
    flow_centers = _homography_flow_centers(direct_centers[ref_frame], band_depths, assignments, k_seq, w2c_seq, ref_frame=ref_frame)
    residual = direct_centers - flow_centers
    _, residual_recon = _fit_poly(residual, times, degree=args.residual_degree)
    gauge_centers = flow_centers + residual_recon
    per_tube_max = _per_tube_max_error(gauge_centers, direct_centers)
    fallback_mask = per_tube_max > args.fallback_max_px
    hybrid_centers = torch.where(fallback_mask.view(1, -1, 1), prt_centers, gauge_centers)

    with torch.no_grad():
        reference_image = _render_from_centers(
            batch,
            direct_centers,
            direct_depth,
            lambda_uv,
            times,
            height=args.target_size,
            width=args.target_size,
            alpha_threshold=args.alpha_threshold,
        )
        gauge_image = _render_from_centers(
            batch,
            gauge_centers,
            direct_depth,
            lambda_uv,
            times,
            height=args.target_size,
            width=args.target_size,
            alpha_threshold=args.alpha_threshold,
        )
        first_image = _render_from_centers(
            batch,
            first_centers,
            first_depth,
            lambda_uv,
            times,
            height=args.target_size,
            width=args.target_size,
            alpha_threshold=args.alpha_threshold,
        )
        segmented_image = _render_from_centers(
            batch,
            segmented_centers,
            segmented_depth,
            lambda_uv,
            times,
            height=args.target_size,
            width=args.target_size,
            alpha_threshold=args.alpha_threshold,
        )
        prt_image = _render_from_centers(
            batch,
            prt_centers,
            prt_depth,
            lambda_uv,
            times,
            height=args.target_size,
            width=args.target_size,
            alpha_threshold=args.alpha_threshold,
        )
        hybrid_image = _render_from_centers(
            batch,
            hybrid_centers,
            direct_depth,
            lambda_uv,
            times,
            height=args.target_size,
            width=args.target_size,
            alpha_threshold=args.alpha_threshold,
        )

    gauge_metrics = _center_metrics(gauge_centers, direct_centers)
    first_metrics = _center_metrics(first_centers, direct_centers)
    segmented_metrics = _center_metrics(segmented_centers, direct_centers)
    prt_metrics = _center_metrics(prt_centers, direct_centers)
    residual_tile_pairs = _estimate_tile_pairs(
        residual_recon,
        batch,
        lambda_uv,
        times,
        width=args.target_size,
        height=args.target_size,
        tile_x=args.tile_x,
        tile_y=args.tile_y,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        clamp_to_image=False,
    )
    segmented_tile_pairs = _estimate_tile_pairs(
        segmented_centers,
        batch,
        lambda_uv,
        times,
        width=args.target_size,
        height=args.target_size,
        tile_x=args.tile_x,
        tile_y=args.tile_y,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        clamp_to_image=True,
    )
    first_tile_pairs = _estimate_tile_pairs(
        first_centers,
        batch,
        lambda_uv,
        times,
        width=args.target_size,
        height=args.target_size,
        tile_x=args.tile_x,
        tile_y=args.tile_y,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        clamp_to_image=True,
    )
    gauge_nonfallback_tile_pairs = _estimate_tile_pairs(
        residual_recon,
        batch,
        lambda_uv,
        times,
        width=args.target_size,
        height=args.target_size,
        tile_x=args.tile_x,
        tile_y=args.tile_y,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        clamp_to_image=False,
        tube_mask=~fallback_mask,
    )
    prt_fallback_tile_pairs = _estimate_tile_pairs(
        prt_centers,
        batch,
        lambda_uv,
        times,
        width=args.target_size,
        height=args.target_size,
        tile_x=args.tile_x,
        tile_y=args.tile_y,
        tile_t=args.tile_t,
        alpha_threshold=args.alpha_threshold,
        clamp_to_image=True,
        tube_mask=fallback_mask,
    )
    hybrid_tile_pairs = {
        "gauge_residual_tile_pairs": gauge_nonfallback_tile_pairs["total_tile_pairs"],
        "prt_fallback_tile_pairs": prt_fallback_tile_pairs["total_tile_pairs"],
        "total_tile_pairs": gauge_nonfallback_tile_pairs["total_tile_pairs"] + prt_fallback_tile_pairs["total_tile_pairs"],
        "fallback_tube_count": int(fallback_mask.sum().item()),
    }
    p95_ratio = gauge_metrics["p95_px"] / max(first_metrics["p95_px"], 1.0e-9)
    hybrid_metrics = _center_metrics(hybrid_centers, direct_centers)
    hybrid_image_metrics = _image_metrics(hybrid_image, reference_image)
    kill_criteria = {
        "center_residual_p95_clearly_below_projective_first_order": p95_ratio <= args.p95_ratio_gate,
        "center_residual_max_at_most_1px": gauge_metrics["max_px"] <= 1.0,
        "flow_sheared_tile_pairs_below_segmented_f4": residual_tile_pairs["total_tile_pairs"] < segmented_tile_pairs["total_tile_pairs"],
        "rendered_tubes_not_more_than_n": int(gauge_centers.shape[1]) <= args.tubes,
        "render_psnr_at_least_50db": _image_metrics(gauge_image, reference_image)["psnr"] >= 50.0,
    }
    hybrid_criteria = {
        "center_residual_max_at_most_fallback_threshold": hybrid_metrics["max_px"] <= args.fallback_max_px,
        "center_residual_p95_clearly_below_projective_first_order": hybrid_metrics["p95_px"] / max(first_metrics["p95_px"], 1.0e-9) <= args.p95_ratio_gate,
        "hybrid_tile_pairs_below_segmented_f4": hybrid_tile_pairs["total_tile_pairs"] < segmented_tile_pairs["total_tile_pairs"],
        "rendered_tubes_not_more_than_n": int(hybrid_centers.shape[1]) <= args.tubes,
        "render_psnr_at_least_50db": hybrid_image_metrics["psnr"] >= 50.0,
    }
    return {
        "name": "depth_banded_homography_flow_residual_probe",
        "note": (
            "Projection/render-only falsifier for depth-banded homography-flow gauge residual tubes. "
            "The homography flow is compiler state; rendered tube count remains N. Dense rendering uses exact "
            "direct depth for the gauge row to isolate center residual before any Metal/backward work."
        ),
        "config": {
            "target_size": args.target_size,
            "max_frames": args.frames,
            "tubes": args.tubes,
            "seed": args.seed,
            "fallback_max_px": args.fallback_max_px,
            "pan_x": args.pan_x,
            "zoom": args.zoom,
            "dolly_z": args.dolly_z,
            "depth_bands": args.depth_bands,
            "residual_degree": args.residual_degree,
            "segments": args.segments,
            "prt_degree": args.prt_degree,
            "velocity_scale": args.velocity_scale,
            "tile_x": args.tile_x,
            "tile_y": args.tile_y,
            "tile_t": args.tile_t,
        },
        "depth_bands": {
            "depths": [float(v) for v in band_depths.detach().cpu().tolist()],
            "counts": [int((assignments == band).sum().item()) for band in range(args.depth_bands)],
        },
        "homography_flow_residual": {
            "rendered_tubes": args.tubes,
            "center_metrics_vs_direct": gauge_metrics,
            "image_metrics_vs_direct": _image_metrics(gauge_image, reference_image),
            "flow_sheared_tile_pair_estimate": residual_tile_pairs,
            "fallback_outliers": {
                "max_px_threshold": args.fallback_max_px,
                "tube_count": int(fallback_mask.sum().item()),
                "fraction": float(fallback_mask.float().mean().item()),
                "largest_outlier_max_px": float(per_tube_max[fallback_mask].max().item()) if bool(fallback_mask.any()) else 0.0,
            },
        },
        "hybrid_gauge_with_prt_fallback": {
            "rendered_tubes": args.tubes,
            "fallback_tubes": int(fallback_mask.sum().item()),
            "gauge_residual_tubes": int((~fallback_mask).sum().item()),
            "center_metrics_vs_direct": hybrid_metrics,
            "image_metrics_vs_direct": hybrid_image_metrics,
            "tile_pair_estimate": hybrid_tile_pairs,
            "criteria": hybrid_criteria,
            "pass": all(bool(value) for value in hybrid_criteria.values()),
        },
        "projective_first_order": {
            "rendered_tubes": args.tubes,
            "camera_fit_error": first_fit_error,
            "center_metrics_vs_direct": first_metrics,
            "image_metrics_vs_direct": _image_metrics(first_image, reference_image),
            "tile_pair_estimate": first_tile_pairs,
        },
        "segmented_f4": {
            "rendered_tubes": args.tubes * args.segments,
            "camera_fit_errors": segmented_fit_errors,
            "center_metrics_vs_direct": segmented_metrics,
            "image_metrics_vs_direct": _image_metrics(segmented_image, reference_image),
            "tile_pair_estimate": segmented_tile_pairs,
        },
        "projective_rational": {
            "rendered_tubes": args.tubes,
            "camera_fit_error": prt_fit_error,
            "center_metrics_vs_direct": prt_metrics,
            "image_metrics_vs_direct": _image_metrics(prt_image, reference_image),
        },
        "kill_criteria": kill_criteria,
        "deferred_criteria": {
            "actual_flow_sheared_render_time": "deferred until there is a flow-sheared tiled renderer; this probe reports tile-pair estimates only",
        },
        "pass": all(bool(value) for value in kill_criteria.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--tubes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pan-x", type=float, default=0.06)
    parser.add_argument("--zoom", type=float, default=0.02)
    parser.add_argument("--dolly-z", type=float, default=0.08)
    parser.add_argument("--depth-bands", type=int, default=4)
    parser.add_argument("--residual-degree", type=int, default=1)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--prt-degree", type=int, default=2)
    parser.add_argument("--velocity-scale", type=float, default=0.0)
    parser.add_argument("--tile-x", type=int, default=8)
    parser.add_argument("--tile-y", type=int, default=8)
    parser.add_argument("--tile-t", type=int, default=4)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--p95-ratio-gate", type=float, default=0.75)
    parser.add_argument("--fallback-max-px", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = run_probe(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
