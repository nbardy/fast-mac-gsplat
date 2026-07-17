from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from multicam_heldout_compare import (
    DEFAULT_BASELINE_CONFIG,
    FreeDynamic3DGS,
    ProjectedTubeSequence,
    RenderedSequence,
    SplatRenderConfig,
    UVTRenderConfig,
    WorldTubeModel,
    apply_uvt_tile_env,
    camera_from_K_w2c,
    config_data_for_run,
    initialize_material_points_from_first_frame,
    initialize_world_tubes_from_train_views,
    load_config_file,
    load_multicam_video_bundle,
    prefix_metrics,
    project_world_tube_sequence,
    projected_regularization,
    render_gaussian_frame,
    render_projected_sequence,
    resolve_device,
    resolve_dynaworld_path,
    resolve_variant_path,
    robust_l1,
    select_K_for_view_time,
    select_view_K,
    select_view_w2c,
    select_w2c_for_view_time,
    serialize_config_value,
    synchronize_device,
    world_tube_metal_stats,
    write_json,
)
from research_project.trainer_harness.tile_metal_autograd import _reduce_sample_bundle
from research_project.trainer_harness.variable_camera_segments import project_piecewise_camera_time_segments
from research_project.trainer_harness.world_tube import (
    PinholeCameraMotion,
    project_world_tubes_pinhole_motion,
    project_world_tubes_pinhole_projective_motion,
)
from torch_gsplat_bridge_star_uvt import stable_backward_samples


def timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    synchronize_device(device)
    started = time.perf_counter()
    value = fn()
    synchronize_device(device)
    return value, time.perf_counter() - started


def summarize(rows: list[dict[str, float]], *, skip_keys: set[str] | None = None) -> dict[str, dict[str, float | int]]:
    skipped = set() if skip_keys is None else skip_keys
    keys = sorted({key for row in rows for key in row if key not in skipped})
    out: dict[str, dict[str, float | int]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        out[key] = {
            "count": len(values),
            "mean_s": statistics.fmean(values),
            "min_s": min(values),
            "max_s": max(values),
            "total_s": sum(values),
        }
    mean_total = float(out.get("total", {}).get("mean_s", 0.0))
    if mean_total > 0.0:
        for value in out.values():
            value["mean_pct_of_total"] = float(value["mean_s"]) / mean_total
    return out


def summarize_scalars(rows: list[dict[str, float]], key: str) -> dict[str, float | int]:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def build_world_tube_model(bundle, args: argparse.Namespace, device: torch.device) -> WorldTubeModel:
    init_x0, init_color, init_t0 = initialize_world_tubes_from_train_views(
        bundle,
        tube_count=args.uvt_tubes,
        init_depth=args.init_depth,
        seed=args.seed,
        init_views=args.uvt_init_views,
        init_sampling=getattr(args, "uvt_init_sampling", "random"),
        init_frames=getattr(args, "uvt_init_frames", "first"),
        init_frame_indices=None,
    )
    return WorldTubeModel(
        init_x0=init_x0,
        init_color=init_color,
        init_t0=init_t0,
        frames=bundle.frame_count,
        init_precision_xy=args.uvt_init_precision_xy,
        init_lambda_t=args.uvt_init_lambda_t,
        init_opacity=args.uvt_init_opacity,
        min_precision_xy=args.uvt_min_precision_xy,
        min_lambda_t=args.uvt_min_lambda_t,
        velocity_reg_weight=args.uvt_velocity_reg,
        depth_velocity_reg_weight=args.uvt_depth_velocity_reg,
        position_reg_weight=args.uvt_position_reg,
    ).to(device)


def build_splat_model(bundle, args: argparse.Namespace, device: torch.device) -> tuple[FreeDynamic3DGS, SplatRenderConfig]:
    init_xyz, init_rgb = initialize_material_points_from_first_frame(
        video=bundle.train_frames[0].permute(0, 2, 3, 1).contiguous(),
        K=bundle.train_K[0],
        num_elements=args.splat_count,
        init_depth=args.init_depth,
    )
    model = FreeDynamic3DGS(
        init_xyz=init_xyz,
        init_rgb=init_rgb,
        num_frames=bundle.frame_count,
        splat_mode="per_frame",
        init_scale=args.splat_init_scale,
        scale_init_log_jitter=0.0,
        init_alpha_logit=0.0,
        init_xyz_noise=0.001,
        init_quat_noise=0.0,
        log_scale_min=-12.0,
        log_scale_max=4.0,
    ).to(device)
    _, _, _, height, width = bundle.train_frames.shape
    return model, SplatRenderConfig(
        height=height,
        width=width,
        renderer=args.splat_renderer,
        tile_size=16 if args.splat_renderer == "fast_mac" else 8,
        bound_scale=3.0,
        alpha_threshold=1.0 / 255.0,
        near_plane=1.0e-3,
        camera_projection="legacy_pinhole",
    )


def one_frame_uvt_config(config: UVTRenderConfig) -> UVTRenderConfig:
    return UVTRenderConfig(
        height=config.height,
        width=config.width,
        frames=1,
        tile_x=config.tile_x,
        tile_y=config.tile_y,
        tile_t=1,
        tile_capacity=config.tile_capacity,
        alpha_threshold=config.alpha_threshold,
        transmittance_threshold=config.transmittance_threshold,
        background=config.background,
        max_alpha=config.max_alpha,
    )


def centered_time(frame: float, frames: int) -> float:
    return float(frame) - 0.5 * float(frames - 1)


def global_to_local_time(global_t: float, *, full_frames: int, config: UVTRenderConfig, frame_start: int) -> float:
    offset = float(frame_start) - 0.5 * float(full_frames - 1) + 0.5 * float(config.frames - 1)
    return float(global_t) - offset


def select_view_K_sequence(K: torch.Tensor, *, view: int, frames: int, view_count: int) -> torch.Tensor:
    if K.ndim == 3:
        return K[view].unsqueeze(0).expand(frames, -1, -1).contiguous()
    if K.ndim == 4:
        return K[view, :frames].contiguous()
    raise ValueError(f"Expected K with shape [V,3,3] or [V,T,3,3], got {tuple(K.shape)}")


def select_view_w2c_sequence(w2c: torch.Tensor, *, view: int, frames: int) -> torch.Tensor:
    if w2c.ndim != 4:
        raise ValueError(f"Expected w2c with shape [V,T,4,4], got {tuple(w2c.shape)}")
    return w2c[view, :frames].contiguous()


def apply_synthetic_camera_motion(args: argparse.Namespace, K_seq: torch.Tensor, w2c_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pan_x = float(getattr(args, "uvt_synthetic_pan_x", 0.0))
    pan_y = float(getattr(args, "uvt_synthetic_pan_y", 0.0))
    dolly_z = float(getattr(args, "uvt_synthetic_dolly_z", 0.0))
    zoom = float(getattr(args, "uvt_synthetic_zoom", 0.0))
    principal_x = float(getattr(args, "uvt_synthetic_principal_x", 0.0))
    principal_y = float(getattr(args, "uvt_synthetic_principal_y", 0.0))
    if not any((pan_x, pan_y, dolly_z, zoom, principal_x, principal_y)):
        return K_seq, w2c_seq
    frames = int(K_seq.shape[0])
    if frames == 1:
        phase = torch.zeros((1,), dtype=K_seq.dtype, device=K_seq.device)
    else:
        phase = torch.linspace(-0.5, 0.5, frames, dtype=K_seq.dtype, device=K_seq.device)
    moved_K = K_seq.clone()
    moved_w2c = w2c_seq.clone()
    moved_w2c[:, 0, 3] = moved_w2c[:, 0, 3] + pan_x * phase
    moved_w2c[:, 1, 3] = moved_w2c[:, 1, 3] + pan_y * phase
    moved_w2c[:, 2, 3] = moved_w2c[:, 2, 3] + dolly_z * phase
    moved_K[:, 0, 0] = moved_K[:, 0, 0] * (1.0 + zoom * phase)
    moved_K[:, 1, 1] = moved_K[:, 1, 1] * (1.0 + zoom * phase)
    moved_K[:, 0, 2] = moved_K[:, 0, 2] + principal_x * phase
    moved_K[:, 1, 2] = moved_K[:, 1, 2] + principal_y * phase
    return moved_K, moved_w2c


def camera_sequences_for_view(bundle, args: argparse.Namespace, *, view: int, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    K_seq = select_view_K_sequence(bundle.train_K, view=view, frames=frames, view_count=int(bundle.train_frames.shape[0]))
    w2c_seq = select_view_w2c_sequence(bundle.train_w2c, view=view, frames=frames)
    return apply_synthetic_camera_motion(args, K_seq, w2c_seq)


def project_world_tube_sequence_dynamic_first_order(
    *,
    model: WorldTubeModel,
    K_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
    config: UVTRenderConfig,
    full_frames: int,
    frame_start: int,
) -> ProjectedTubeSequence:
    window_mid_frame = float(frame_start) + 0.5 * float(config.frames - 1)
    mid_index = int(round(window_mid_frame))
    mid_index = max(0, min(int(full_frames) - 1, mid_index))
    prev_index = max(0, mid_index - 1)
    next_index = min(int(full_frames) - 1, mid_index + 1)
    if next_index == prev_index:
        K_dot = torch.zeros_like(K_seq[mid_index])
        w2c_dot = torch.zeros_like(w2c_seq[mid_index])
    else:
        dt = float(next_index - prev_index)
        K_dot = (K_seq[next_index] - K_seq[prev_index]) / dt
        w2c_dot = (w2c_seq[next_index] - w2c_seq[prev_index]) / dt
    K_mid = K_seq[mid_index]
    chart_global_t = centered_time(window_mid_frame, int(full_frames))
    camera = PinholeCameraMotion(
        fx=float(K_mid[0, 0].detach().cpu()),
        fy=float(K_mid[1, 1].detach().cpu()),
        cx=float(K_mid[0, 2].detach().cpu()),
        cy=float(K_mid[1, 2].detach().cpu()),
        fx_dot=float(K_dot[0, 0].detach().cpu()),
        fy_dot=float(K_dot[1, 1].detach().cpu()),
        cx_dot=float(K_dot[0, 2].detach().cpu()),
        cy_dot=float(K_dot[1, 2].detach().cpu()),
        world_to_camera=w2c_seq[mid_index].to(dtype=torch.float32),
        world_to_camera_dot=w2c_dot.to(dtype=torch.float32),
        chart_time=chart_global_t,
    )
    ma, q_uvt, depth0, depth_beta, opacity, color = project_world_tubes_pinhole_motion(model.batch(), camera, config)
    local_t = global_to_local_time(chart_global_t, full_frames=int(full_frames), config=config, frame_start=frame_start)
    ma = torch.cat((ma[:, :2], torch.full_like(ma[:, 2:3], local_t)), dim=-1).contiguous()
    return ProjectedTubeSequence(ma=ma, q_uvt=q_uvt, depth0=depth0, depth_beta=depth_beta, opacity=opacity, color=color)


def project_world_tube_sequence_projective_first_order(
    *,
    model: WorldTubeModel,
    K_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
    config: UVTRenderConfig,
    full_frames: int,
    frame_start: int,
) -> ProjectedTubeSequence:
    window_mid_frame = float(frame_start) + 0.5 * float(config.frames - 1)
    mid_index = int(round(window_mid_frame))
    mid_index = max(0, min(int(full_frames) - 1, mid_index))
    prev_index = max(0, mid_index - 1)
    next_index = min(int(full_frames) - 1, mid_index + 1)
    if next_index == prev_index:
        K_dot = torch.zeros_like(K_seq[mid_index])
        w2c_dot = torch.zeros_like(w2c_seq[mid_index])
    else:
        dt = float(next_index - prev_index)
        K_dot = (K_seq[next_index] - K_seq[prev_index]) / dt
        w2c_dot = (w2c_seq[next_index] - w2c_seq[prev_index]) / dt
    K_mid = K_seq[mid_index]
    chart_global_t = centered_time(window_mid_frame, int(full_frames))
    camera = PinholeCameraMotion(
        fx=float(K_mid[0, 0].detach().cpu()),
        fy=float(K_mid[1, 1].detach().cpu()),
        cx=float(K_mid[0, 2].detach().cpu()),
        cy=float(K_mid[1, 2].detach().cpu()),
        fx_dot=float(K_dot[0, 0].detach().cpu()),
        fy_dot=float(K_dot[1, 1].detach().cpu()),
        cx_dot=float(K_dot[0, 2].detach().cpu()),
        cy_dot=float(K_dot[1, 2].detach().cpu()),
        world_to_camera=w2c_seq[mid_index].to(dtype=torch.float32),
        world_to_camera_dot=w2c_dot.to(dtype=torch.float32),
        chart_time=chart_global_t,
    )
    ma, q_uvt, depth0, depth_beta, opacity, color = project_world_tubes_pinhole_projective_motion(model.batch(), camera, config)
    local_t = global_to_local_time(chart_global_t, full_frames=int(full_frames), config=config, frame_start=frame_start)
    ma = torch.cat((ma[:, :2], torch.full_like(ma[:, 2:3], local_t)), dim=-1).contiguous()
    return ProjectedTubeSequence(ma=ma, q_uvt=q_uvt, depth0=depth0, depth_beta=depth_beta, opacity=opacity, color=color)


def project_world_tube_sequence_segmented_camera(
    *,
    model: WorldTubeModel,
    K_seq: torch.Tensor,
    w2c_seq: torch.Tensor,
    config: UVTRenderConfig,
    full_frames: int,
    frame_start: int,
    frames_per_segment: int,
) -> tuple[ProjectedTubeSequence, dict[str, float]]:
    segments = project_piecewise_camera_time_segments(
        model.batch(),
        K_seq,
        w2c_seq,
        config,
        full_frames=full_frames,
        frame_start=frame_start,
        frames_per_segment=frames_per_segment,
    )
    projected = ProjectedTubeSequence(
        ma=segments.ma,
        q_uvt=segments.q_uvt,
        depth0=segments.depth0,
        depth_beta=segments.depth_beta,
        opacity=segments.opacity,
        color=segments.color,
    )
    diagnostics = {
        "projected_tube_count": float(segments.diagnostics.segment_count),
        "mean_segments_per_tube": float(segments.diagnostics.mean_segments_per_tube),
        "temporal_chunk_count": float(segments.diagnostics.temporal_chunk_count),
    }
    return projected, diagnostics


def project_world_tube_sequence_per_frame_camera(
    *,
    model: WorldTubeModel,
    bundle,
    view: int,
    frame_start: int,
    config: UVTRenderConfig,
    full_frames: int,
    K_seq: torch.Tensor | None = None,
    w2c_seq: torch.Tensor | None = None,
) -> list[ProjectedTubeSequence]:
    frame_config = one_frame_uvt_config(config)
    view_count = int(bundle.train_frames.shape[0])
    return [
        project_world_tube_sequence(
            model,
            (
                K_seq[frame_start + local_frame]
                if K_seq is not None
                else select_K_for_view_time(bundle.train_K, view=view, t=frame_start + local_frame, view_count=view_count)
            ),
            (
                w2c_seq[frame_start + local_frame]
                if w2c_seq is not None
                else select_w2c_for_view_time(bundle.train_w2c, view=view, t=frame_start + local_frame)
            ),
            frame_config,
            full_frames=full_frames,
            frame_start=frame_start + local_frame,
        )
        for local_frame in range(config.frames)
    ]


def render_projected_sequence_per_frame_camera(
    *,
    projected_frames: list[ProjectedTubeSequence],
    config: UVTRenderConfig,
    backend: str,
    reduction_mode: str,
    sample_emission_mode: str,
) -> RenderedSequence:
    frame_config = one_frame_uvt_config(config)
    images = [
        render_projected_sequence(
            projected,
            frame_config,
            backend=backend,
            reduction_mode=reduction_mode,
            sample_emission_mode=sample_emission_mode,
        ).rgb[0]
        for projected in projected_frames
    ]
    rgb = torch.stack(images, dim=0).contiguous()
    alpha = torch.ones((config.frames, config.height, config.width), dtype=rgb.dtype, device=rgb.device)
    return RenderedSequence(rgb=rgb, alpha=alpha)


def projected_regularization_per_frame_camera(
    *,
    projected_frames: list[ProjectedTubeSequence],
    config: UVTRenderConfig,
    args: argparse.Namespace,
) -> torch.Tensor:
    frame_config = one_frame_uvt_config(config)
    regs = [
        projected_regularization(
            projected,
            frame_config,
            tile_load_weight=args.uvt_tile_load_reg,
            tile_load_target=args.uvt_tile_load_target,
            depth_slope_weight=args.uvt_depth_slope_reg,
            depth_margin_weight=args.uvt_depth_margin_reg,
            depth_margin=args.uvt_depth_margin,
        )[0]
        for projected in projected_frames
    ]
    return torch.stack(regs).mean()


def star_step(
    *,
    model: WorldTubeModel,
    optimizer: torch.optim.Optimizer,
    bundle,
    args: argparse.Namespace,
    render_config: UVTRenderConfig,
    window_config: UVTRenderConfig,
    step: int,
    device: torch.device,
) -> dict[str, float]:
    view_count, frames, _, _, _ = bundle.train_frames.shape
    view = step % view_count
    frame_start = 0
    config = render_config
    full_frames = None
    if args.uvt_loss_scope == "temporal_window":
        max_start = frames - args.uvt_window_frames
        frame_start = (step * args.frame_stride_for_probe) % (max_start + 1)
        config = window_config
        full_frames = frames
    camera_sequence_mode = getattr(args, "uvt_camera_sequence_mode", "static_view")
    if camera_sequence_mode == "per_frame_loop":
        full_frames = frames
    elif camera_sequence_mode in {"dynamic_first_order", "projective_first_order", "segmented"}:
        full_frames = frames
    elif camera_sequence_mode != "static_view":
        raise ValueError(
            "uvt_camera_sequence_mode must be one of: static_view, dynamic_first_order, "
            "projective_first_order, segmented, per_frame_loop"
        )

    phases: dict[str, float] = {}
    step_started = time.perf_counter()
    _, phases["zero_grad"] = timed(device, lambda: optimizer.zero_grad(set_to_none=True))
    if camera_sequence_mode == "static_view":
        K_seq, w2c_seq = camera_sequences_for_view(bundle, args, view=view, frames=frames)
        projected, phases["project"] = timed(
            device,
            lambda: project_world_tube_sequence(
                model,
                K_seq[0],
                w2c_seq[0],
                config,
                full_frames=full_frames,
                frame_start=frame_start,
            ),
        )
        rendered, phases["render"] = timed(
            device,
            lambda: render_projected_sequence(
                projected,
                config,
                backend=args.uvt_render_backend,
                reduction_mode=args.uvt_reduction_mode,
                sample_emission_mode=args.uvt_sample_emission_mode,
            ),
        )
        projection_diagnostics: dict[str, float] = {"projected_tube_count": float(projected.ma.shape[0])}
    elif camera_sequence_mode == "dynamic_first_order":
        K_seq, w2c_seq = camera_sequences_for_view(bundle, args, view=view, frames=frames)
        projected, phases["project"] = timed(
            device,
            lambda: project_world_tube_sequence_dynamic_first_order(
                model=model,
                K_seq=K_seq,
                w2c_seq=w2c_seq,
                config=config,
                full_frames=frames,
                frame_start=frame_start,
            ),
        )
        rendered, phases["render"] = timed(
            device,
            lambda: render_projected_sequence(
                projected,
                config,
                backend=args.uvt_render_backend,
                reduction_mode=args.uvt_reduction_mode,
                sample_emission_mode=args.uvt_sample_emission_mode,
            ),
        )
        projection_diagnostics = {"projected_tube_count": float(projected.ma.shape[0])}
    elif camera_sequence_mode == "projective_first_order":
        K_seq, w2c_seq = camera_sequences_for_view(bundle, args, view=view, frames=frames)
        projected, phases["project"] = timed(
            device,
            lambda: project_world_tube_sequence_projective_first_order(
                model=model,
                K_seq=K_seq,
                w2c_seq=w2c_seq,
                config=config,
                full_frames=frames,
                frame_start=frame_start,
            ),
        )
        rendered, phases["render"] = timed(
            device,
            lambda: render_projected_sequence(
                projected,
                config,
                backend=args.uvt_render_backend,
                reduction_mode=args.uvt_reduction_mode,
                sample_emission_mode=args.uvt_sample_emission_mode,
            ),
        )
        projection_diagnostics = {"projected_tube_count": float(projected.ma.shape[0])}
    elif camera_sequence_mode == "segmented":
        K_seq, w2c_seq = camera_sequences_for_view(bundle, args, view=view, frames=frames)
        segment_projection_diagnostics: dict[str, float] = {}

        def project_segmented() -> ProjectedTubeSequence:
            projected_value, diagnostics = project_world_tube_sequence_segmented_camera(
                model=model,
                K_seq=K_seq,
                w2c_seq=w2c_seq,
                config=config,
                full_frames=frames,
                frame_start=frame_start,
                frames_per_segment=args.uvt_segment_frames,
            )
            segment_projection_diagnostics.update(diagnostics)
            return projected_value

        projected, phases["project"] = timed(device, project_segmented)
        rendered, phases["render"] = timed(
            device,
            lambda: render_projected_sequence(
                projected,
                config,
                backend=args.uvt_render_backend,
                reduction_mode=args.uvt_reduction_mode,
                sample_emission_mode=args.uvt_sample_emission_mode,
            ),
        )
        projection_diagnostics = segment_projection_diagnostics
    else:
        K_seq, w2c_seq = camera_sequences_for_view(bundle, args, view=view, frames=frames)
        projected, phases["project"] = timed(
            device,
            lambda: project_world_tube_sequence_per_frame_camera(
                model=model,
                bundle=bundle,
                view=view,
                frame_start=frame_start,
                config=config,
                full_frames=frames,
                K_seq=K_seq,
                w2c_seq=w2c_seq,
            ),
        )
        rendered, phases["render"] = timed(
            device,
            lambda: render_projected_sequence_per_frame_camera(
                projected_frames=projected,
                config=config,
                backend=args.uvt_render_backend,
                reduction_mode=args.uvt_reduction_mode,
                sample_emission_mode=args.uvt_sample_emission_mode,
            ),
        )
        projection_diagnostics = {
            "projected_tube_count": float(sum(int(item.ma.shape[0]) for item in projected)),
            "mean_segments_per_tube": float(config.frames),
            "temporal_chunk_count": float(config.frames),
        }

    def compute_loss() -> torch.Tensor:
        if args.uvt_loss_scope == "sampled_frame":
            frame = step % frames
            target = bundle.train_frames[view, frame].permute(1, 2, 0)
            recon_loss = robust_l1(rendered.rgb[frame] - target)
        elif args.uvt_loss_scope == "view_sequence":
            target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
            recon_loss = robust_l1(rendered.rgb - target)
        else:
            target = bundle.train_frames[view, frame_start : frame_start + args.uvt_window_frames].permute(0, 2, 3, 1).contiguous()
            recon_loss = robust_l1(rendered.rgb - target)
        if camera_sequence_mode != "per_frame_loop":
            projected_reg, _ = projected_regularization(
                projected,
                config,
                tile_load_weight=args.uvt_tile_load_reg,
                tile_load_target=args.uvt_tile_load_target,
                depth_slope_weight=args.uvt_depth_slope_reg,
                depth_margin_weight=args.uvt_depth_margin_reg,
                depth_margin=args.uvt_depth_margin,
            )
        else:
            projected_reg = projected_regularization_per_frame_camera(projected_frames=projected, config=config, args=args)
        return recon_loss + model.regularization() + projected_reg

    loss, phases["loss"] = timed(device, compute_loss)
    _, phases["backward"] = timed(device, loss.backward)
    _, phases["optimizer"] = timed(
        device,
        lambda: (torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0), optimizer.step()),
    )
    synchronize_device(device)
    phases["total"] = time.perf_counter() - step_started
    phases["loss_value"] = float(loss.detach().cpu())
    phases.update(projection_diagnostics)
    return phases


def splat_step(
    *,
    model: FreeDynamic3DGS,
    render_cfg: SplatRenderConfig,
    optimizer: torch.optim.Optimizer,
    bundle,
    args: argparse.Namespace,
    step: int,
    device: torch.device,
) -> dict[str, float]:
    view_count, frames, _, height, width = bundle.train_frames.shape
    view = step % view_count
    frame = (step * 7) % frames
    phases: dict[str, float] = {}
    step_started = time.perf_counter()
    _, phases["zero_grad"] = timed(device, lambda: optimizer.zero_grad(set_to_none=True))

    splat_loss_scope = getattr(args, "splat_loss_scope", "sampled_frame")
    if splat_loss_scope == "sampled_frame":
        camera = camera_from_K_w2c(
            select_K_for_view_time(bundle.train_K, view=view, t=frame, view_count=view_count),
            select_w2c_for_view_time(bundle.train_w2c, view=view, t=frame),
        )
        splats, phases["frame_model"] = timed(device, lambda: model.frame(frame))
        image, phases["render"] = timed(
            device,
            lambda: render_gaussian_frame(
                splats,
                camera,
                height=height,
                width=width,
                mode=render_cfg.renderer,
                tile_size=render_cfg.tile_size,
                bound_scale=render_cfg.bound_scale,
                alpha_threshold=render_cfg.alpha_threshold,
                near_plane=render_cfg.near_plane,
                camera_projection=render_cfg.camera_projection,
            ).permute(1, 2, 0),
        )
    elif splat_loss_scope == "view_sequence":
        cameras = [
            camera_from_K_w2c(
                select_K_for_view_time(bundle.train_K, view=view, t=t, view_count=view_count),
                select_w2c_for_view_time(bundle.train_w2c, view=view, t=t),
            )
            for t in range(frames)
        ]
        splats_by_frame, phases["frame_model"] = timed(device, lambda: [model.frame(t) for t in range(frames)])

        def render_sequence() -> torch.Tensor:
            images = [
                render_gaussian_frame(
                    splats_by_frame[t],
                    cameras[t],
                    height=height,
                    width=width,
                    mode=render_cfg.renderer,
                    tile_size=render_cfg.tile_size,
                    bound_scale=render_cfg.bound_scale,
                    alpha_threshold=render_cfg.alpha_threshold,
                    near_plane=render_cfg.near_plane,
                    camera_projection=render_cfg.camera_projection,
                ).permute(1, 2, 0)
                for t in range(frames)
            ]
            return torch.stack(images, dim=0).contiguous()

        image, phases["render"] = timed(device, render_sequence)
    else:
        raise ValueError("splat_loss_scope must be one of: sampled_frame, view_sequence")

    def compute_loss() -> torch.Tensor:
        if splat_loss_scope == "sampled_frame":
            target = bundle.train_frames[view, frame].permute(1, 2, 0)
        else:
            target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
        loss = robust_l1(image - target)
        return loss + 1.0e-4 * model.scale_loss() + 1.0e-3 * model.temporal_smoothness_loss()

    loss, phases["loss"] = timed(device, compute_loss)
    _, phases["backward"] = timed(device, loss.backward)
    _, phases["optimizer"] = timed(
        device,
        lambda: (torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0), optimizer.step()),
    )
    synchronize_device(device)
    phases["total"] = time.perf_counter() - step_started
    phases["loss_value"] = float(loss.detach().cpu())
    return phases


def star_backward_microbreakdown(
    *,
    model: WorldTubeModel,
    bundle,
    args: argparse.Namespace,
    render_config: UVTRenderConfig,
    window_config: UVTRenderConfig,
    device: torch.device,
) -> dict[str, Any]:
    if args.uvt_render_backend != "metal_tile":
        return {"skipped": "STAR backward microbreakdown currently targets the metal_tile backend."}
    if getattr(args, "uvt_camera_sequence_mode", "static_view") != "static_view":
        return {"skipped": "STAR backward microbreakdown currently targets static-view sequence projection."}
    if args.uvt_sample_emission_mode != "atomic_append" or args.uvt_reduction_mode != "index_add":
        return {
            "skipped": (
                "STAR backward microbreakdown currently reports the default stable sample-emission path only; "
                f"active path is sample_emission_mode={args.uvt_sample_emission_mode!r}, "
                f"reduction_mode={args.uvt_reduction_mode!r}."
            )
        }

    frames = int(bundle.train_frames.shape[1])
    config = window_config if args.uvt_loss_scope == "temporal_window" else render_config
    full_frames = frames if args.uvt_loss_scope == "temporal_window" else None
    frame_start = 0
    projected, project_s = timed(
        device,
        lambda: project_world_tube_sequence(
            model,
            select_view_K(bundle.train_K, 0),
            select_view_w2c(bundle.train_w2c, 0),
            config,
            full_frames=full_frames,
            frame_start=frame_start,
        ),
    )
    grad_image = torch.ones((config.frames, config.height, config.width, 3), dtype=torch.float32, device=device)

    def sample_backward() -> tuple[torch.Tensor, ...]:
        return stable_backward_samples(
            projected.ma.detach(),
            projected.q_uvt.detach(),
            projected.depth0.detach(),
            projected.depth_beta.detach(),
            projected.opacity.detach(),
            projected.color.detach(),
            grad_image,
            config,
        )

    samples, sample_s = timed(device, sample_backward)
    ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, tile_unstable = samples
    tube_count = int(projected.ma.shape[0])
    (grad_ma, grad_q, grad_opacity, grad_color), reduce_bundle_s = timed(
        device,
        lambda: _reduce_sample_bundle(
            ids,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        ),
    )
    zero_depth0 = torch.zeros_like(projected.depth0)
    zero_depth_beta = torch.zeros_like(projected.depth_beta)
    model.zero_grad(set_to_none=True)
    _, projection_vjp_s = timed(
        device,
        lambda: torch.autograd.backward(
            (
                projected.ma,
                projected.q_uvt,
                projected.depth0,
                projected.depth_beta,
                projected.opacity,
                projected.color,
            ),
            (
                grad_ma,
                grad_q,
                zero_depth0,
                zero_depth_beta,
                grad_opacity,
                grad_color,
            ),
        ),
    )
    tile_count = ((config.width + config.tile_x - 1) // config.tile_x) * (
        (config.height + config.tile_y - 1) // config.tile_y
    ) * ((config.frames + config.tile_t - 1) // config.tile_t)
    allocated_sample_slots = tile_count * config.tile_x * config.tile_y * config.tile_t * config.tile_capacity
    sample_count = int(ids.numel())
    return {
        "project_s": project_s,
        "sample_backward_s": sample_s,
        "reduce_total_s": reduce_bundle_s,
        "reduce_bundle_s": reduce_bundle_s,
        "projection_vjp_s": projection_vjp_s,
        "sample_plus_reduce_s": sample_s + reduce_bundle_s,
        "allocated_sample_slot_count": int(allocated_sample_slots),
        "sample_count": sample_count,
        "compact_sample_fraction": float(sample_count) / float(max(allocated_sample_slots, 1)),
        "valid_sample_count": int(((ids >= 0) & (ids < tube_count)).sum().detach().cpu()),
        "unstable_tile_fraction": float(tile_unstable.float().mean().detach().cpu()),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    if args.uvt_render_backend == "metal_tile" and device.type != "mps":
        raise ValueError("--uvt-render-backend=metal_tile requires device=mps")
    torch.manual_seed(args.seed)
    config = load_config_file(resolve_dynaworld_path(args.baseline_config))
    data_cfg = config_data_for_run(config, target_size=args.target_size, max_frames=args.max_frames)
    bundle = load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=dict(config["camera"]),
        target_size=args.target_size,
        device=device,
    )
    _, frames, _, height, width = bundle.train_frames.shape
    if args.uvt_loss_scope == "temporal_window" and args.uvt_window_frames > frames:
        raise ValueError(f"--uvt-window-frames={args.uvt_window_frames} exceeds loaded frame count {frames}")

    render_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        tile_x=args.uvt_tile_x,
        tile_y=args.uvt_tile_y,
        tile_t=args.uvt_tile_t,
        tile_capacity=args.uvt_tile_capacity,
    )
    apply_uvt_tile_env(render_config)
    window_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=args.uvt_window_frames if args.uvt_loss_scope == "temporal_window" else frames,
        tile_x=render_config.tile_x,
        tile_y=render_config.tile_y,
        tile_t=render_config.tile_t,
        tile_capacity=render_config.tile_capacity,
        alpha_threshold=render_config.alpha_threshold,
        transmittance_threshold=render_config.transmittance_threshold,
        background=render_config.background,
        max_alpha=render_config.max_alpha,
    )

    skip_star = bool(getattr(args, "skip_star", False))
    skip_dynamic = bool(getattr(args, "skip_dynamic", False))
    star_model = None if skip_star else build_world_tube_model(bundle, args, device)
    star_optimizer = None if star_model is None else torch.optim.Adam(star_model.parameters(), lr=args.uvt_lr)
    splat_model = None
    splat_render_cfg = None
    splat_optimizer = None
    if not skip_dynamic:
        splat_model, splat_render_cfg = build_splat_model(bundle, args, device)
        splat_optimizer = torch.optim.Adam(splat_model.parameters(), lr=args.splat_lr)

    star_rows = []
    splat_rows = []
    for step in range(args.warmup_steps + args.steps):
        star_row = None
        if star_model is not None and star_optimizer is not None:
            star_row = star_step(
                model=star_model,
                optimizer=star_optimizer,
                bundle=bundle,
                args=args,
                render_config=render_config,
                window_config=window_config,
                step=step,
                device=device,
            )
        splat_row = None
        if splat_model is not None and splat_render_cfg is not None and splat_optimizer is not None:
            splat_row = splat_step(
                model=splat_model,
                render_cfg=splat_render_cfg,
                optimizer=splat_optimizer,
                bundle=bundle,
                args=args,
                step=step,
                device=device,
            )
        if step >= args.warmup_steps:
            if star_row is not None:
                star_rows.append(star_row)
            if splat_row is not None:
                splat_rows.append(splat_row)

    synthetic_camera_motion_active = any(
        float(getattr(args, name, 0.0))
        for name in (
            "uvt_synthetic_pan_x",
            "uvt_synthetic_pan_y",
            "uvt_synthetic_dolly_z",
            "uvt_synthetic_zoom",
            "uvt_synthetic_principal_x",
            "uvt_synthetic_principal_y",
        )
    )
    star_stats = (
        world_tube_metal_stats(star_model, bundle, camera_projection="legacy_pinhole", render_config=render_config)
        if (
            star_model is not None
            and args.uvt_render_backend == "metal_tile"
            and args.uvt_camera_sequence_mode == "static_view"
            and not synthetic_camera_motion_active
        )
        else None
    )
    if star_model is None:
        star_backward_breakdown = {"skipped": "STAR timing disabled by skip_star."}
    elif getattr(args, "skip_backward_microbreakdown", False):
        star_backward_breakdown = {"skipped": "Disabled by --skip-backward-microbreakdown."}
    else:
        star_backward_breakdown = star_backward_microbreakdown(
            model=star_model,
            bundle=bundle,
            args=args,
            render_config=render_config,
            window_config=window_config,
            device=device,
        )
    return {
        "meta": {
            "baseline_config": str(resolve_dynaworld_path(args.baseline_config)),
            "target_size": args.target_size,
            "max_frames": args.max_frames,
            "loaded_frame_count": frames,
            "loaded_train_frame_shape": list(bundle.train_frames.shape),
            "device": str(device),
            "train_cameras": bundle.train_camera_names,
            "heldout_cameras": bundle.heldout_camera_names,
            "pose_source": bundle.pose_source,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "uvt_reduction_mode": args.uvt_reduction_mode,
            "uvt_sample_emission_mode": args.uvt_sample_emission_mode,
            "uvt_camera_sequence_mode": args.uvt_camera_sequence_mode,
            "uvt_segment_frames": args.uvt_segment_frames,
            "synthetic_camera_motion": {
                "pan_x": args.uvt_synthetic_pan_x,
                "pan_y": args.uvt_synthetic_pan_y,
                "dolly_z": args.uvt_synthetic_dolly_z,
                "zoom": args.uvt_synthetic_zoom,
                "principal_x": args.uvt_synthetic_principal_x,
                "principal_y": args.uvt_synthetic_principal_y,
                "active": synthetic_camera_motion_active,
            },
            "config_data": serialize_config_value(data_cfg),
            "note": "Per-step timing only; not quality evidence.",
        },
        "star_uvt": {
            "skipped": skip_star,
            "tube_count": args.uvt_tubes,
            "render_backend": args.uvt_render_backend,
            "reduction_mode": args.uvt_reduction_mode,
            "sample_emission_mode": args.uvt_sample_emission_mode,
            "camera_sequence_mode": args.uvt_camera_sequence_mode,
            "loss_scope": args.uvt_loss_scope,
            "window_frames": args.uvt_window_frames if args.uvt_loss_scope == "temporal_window" else None,
            "lr": args.uvt_lr,
            "tile_x": render_config.tile_x,
            "tile_y": render_config.tile_y,
            "tile_t": render_config.tile_t,
            "tile_capacity": render_config.tile_capacity,
            "tile_load_reg": args.uvt_tile_load_reg,
            "tile_load_target": args.uvt_tile_load_target,
            "depth_slope_reg": args.uvt_depth_slope_reg,
            "rows": star_rows,
            "summary": summarize(
                star_rows,
                skip_keys={
                    "loss_value",
                    "projected_tube_count",
                    "mean_segments_per_tube",
                    "temporal_chunk_count",
                },
            ),
            "loss_value_summary": summarize_scalars(star_rows, "loss_value"),
            "projected_tube_count_summary": summarize_scalars(star_rows, "projected_tube_count"),
            "mean_segments_per_tube_summary": summarize_scalars(star_rows, "mean_segments_per_tube"),
            "temporal_chunk_count_summary": summarize_scalars(star_rows, "temporal_chunk_count"),
            "metal_stats_after_probe": star_stats,
            "backward_microbreakdown": star_backward_breakdown,
        },
        "free_dynamic_splats": {
            "skipped": skip_dynamic,
            "splat_count": args.splat_count,
            "renderer": args.splat_renderer,
            "loss_scope": getattr(args, "splat_loss_scope", "sampled_frame"),
            "lr": args.splat_lr,
            "rows": splat_rows,
            "summary": summarize(splat_rows, skip_keys={"loss_value"}),
            "loss_value_summary": summarize_scalars(splat_rows, "loss_value"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--frame-stride-for-probe", type=int, default=3)
    parser.add_argument("--uvt-tubes", type=int, default=256)
    parser.add_argument("--uvt-lr", type=float, default=0.03)
    parser.add_argument("--uvt-render-backend", choices=("dense", "metal_tile"), default="metal_tile")
    parser.add_argument(
        "--uvt-camera-sequence-mode",
        choices=("static_view", "dynamic_first_order", "projective_first_order", "segmented", "per_frame_loop"),
        default="static_view",
    )
    parser.add_argument("--uvt-segment-frames", type=int, default=4)
    parser.add_argument("--uvt-synthetic-pan-x", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-pan-y", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-dolly-z", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-zoom", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-principal-x", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-principal-y", type=float, default=0.0)
    parser.add_argument(
        "--uvt-reduction-mode",
        choices=(
            "index_add",
            "sorted_cpu",
            "key_sort_scan_metal",
            "key_sort_compensated_scan_metal",
            "key_sort_segmented_metal",
        ),
        default="index_add",
    )
    parser.add_argument(
        "--uvt-sample-emission-mode",
        choices=(
            "atomic_append",
            "with_keys",
            "tile_pair",
            "tile_pair_compensated",
            "tile_pair_grouped",
            "tile_pair_parallel",
            "tile_pair_scanline",
            "tile_pair_sharedsort",
            "tile_pair_target_bounds",
            "tile_pair_suffix",
            "direct_atomic",
            "direct_fixedpoint",
            "direct_split_fixedpoint",
            "direct_serial",
            "tile_pair_atomic",
            "tile_pair_fixedpoint",
            "tile_pair_reduced",
            "tile_pair_reduced_parallel",
            "tile_pair_suffix_reduced",
        ),
        default="direct_atomic",
    )
    parser.add_argument("--uvt-loss-scope", choices=("sampled_frame", "view_sequence", "temporal_window"), default="temporal_window")
    parser.add_argument("--uvt-window-frames", type=int, default=4)
    parser.add_argument("--uvt-init-precision-xy", type=float, default=30.0)
    parser.add_argument("--uvt-init-lambda-t", type=float, default=0.35)
    parser.add_argument("--uvt-init-opacity", type=float, default=0.35)
    parser.add_argument("--uvt-min-precision-xy", type=float, default=1.0e-5)
    parser.add_argument("--uvt-min-lambda-t", type=float, default=1.0e-5)
    parser.add_argument("--uvt-velocity-reg", type=float, default=1.0e-4)
    parser.add_argument("--uvt-depth-velocity-reg", type=float, default=0.0)
    parser.add_argument("--uvt-position-reg", type=float, default=1.0e-6)
    parser.add_argument("--uvt-tile-load-reg", type=float, default=0.001)
    parser.add_argument("--uvt-tile-load-target", type=float, default=7000.0)
    parser.add_argument("--uvt-depth-slope-reg", type=float, default=0.05)
    parser.add_argument("--uvt-depth-margin-reg", type=float, default=0.0)
    parser.add_argument("--uvt-depth-margin", type=float, default=0.05)
    parser.add_argument("--uvt-tile-x", type=int, default=8)
    parser.add_argument("--uvt-tile-y", type=int, default=8)
    parser.add_argument("--uvt-tile-t", type=int, default=1)
    parser.add_argument("--uvt-tile-capacity", type=int, default=256)
    parser.add_argument("--uvt-init-views", choices=("first", "all_train"), default="first")
    parser.add_argument("--uvt-init-sampling", choices=("random", "grid"), default="random")
    parser.add_argument("--uvt-init-frames", choices=("first", "all", "fit"), default="first")
    parser.add_argument("--skip-backward-microbreakdown", action="store_true")
    parser.add_argument("--skip-star", action="store_true")
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument("--splat-count", type=int, default=2048)
    parser.add_argument("--splat-lr", type=float, default=0.002)
    parser.add_argument("--splat-renderer", choices=("dense", "fast_mac"), default="fast_mac")
    parser.add_argument("--splat-loss-scope", choices=("sampled_frame", "view_sequence"), default="sampled_frame")
    parser.add_argument("--splat-init-scale", type=float, default=0.035)
    parser.add_argument("--init-depth", type=float, default=2.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("research_project/benchmarks/results/multicam_train_step_timing_probe_mps_256_16f_current_default.json"),
    )
    args = parser.parse_args()
    report = run_probe(args)
    out_path = resolve_variant_path(args.out_json)
    write_json(out_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote multicam train-step timing probe to {out_path}")


if __name__ == "__main__":
    main()
