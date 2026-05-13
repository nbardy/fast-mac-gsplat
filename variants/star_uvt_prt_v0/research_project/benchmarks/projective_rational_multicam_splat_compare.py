from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_dynaworld_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "train" / "multicam_video_data.py").exists():
            return parent
    raise FileNotFoundError("could not find dynaworld root from PRT variant")


DYNAWORLD_ROOT = _find_dynaworld_root()
TRAIN_SRC = DYNAWORLD_ROOT / "src" / "train"
GAUGE_EXPERIMENTS = DYNAWORLD_ROOT / "research_experiments" / "gauge_fields"
for _path in (TRAIN_SRC, GAUGE_EXPERIMENTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from config_utils import load_config_file  # noqa: E402
from multicam_video_data import load_multicam_video_bundle  # noqa: E402
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    CameraPathPolynomial,
    WorldTubeBatch,
    centered_frame_times,
    compile_projective_rational_tubes,
    fit_camera_path_polynomial,
)
from research_project.trainer_harness.projective_rational_metal_autograd import (  # noqa: E402
    render_projective_rational_tubes_metal_direct_serial_backward,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    apply_projective_rational_tile_env,
    parse_projective_rational_tile_config,
    projective_rational_tile_pixel_fused_mse_backward,
    recommend_projective_rational_train_speed_tile_policy,
    recommend_projective_rational_tile_config,
    render_projective_rational_tubes_tiled,
)
from train_splat_baseline import (  # noqa: E402
    FreeDynamic3DGS,
    SplatRenderConfig,
    camera_from_K_w2c,
    initialize_material_points_from_first_frame,
    render_gaussian_frame,
    render_splat_sequence,
)


DEFAULT_BASELINE_CONFIG = (
    DYNAWORLD_ROOT
    / "src"
    / "train_configs"
    / "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alphaab_alpha1_128.jsonc"
)


def _resolve_dynaworld_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return DYNAWORLD_ROOT / value


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _logit(value: Tensor) -> Tensor:
    return torch.logit(value.clamp(1.0e-4, 1.0 - 1.0e-4))


def _softplus_inverse(value: Tensor) -> Tensor:
    return torch.log(torch.expm1(value.clamp_min(1.0e-6)))


def _psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1.0e-12))


def _metrics(image: Tensor, target: Tensor) -> dict[str, float]:
    diff = image - target
    mse = float(diff.square().mean().detach().cpu())
    return {
        "mse": mse,
        "psnr": _psnr(mse),
        "l1": float(diff.abs().mean().detach().cpu()),
    }


def _aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: sum(float(row[key]) for row in rows) / float(len(rows))
        for key in sorted(rows[0])
    }


def _select_k(K: Tensor, *, view: int, frame: int, view_count: int) -> Tensor:
    if K.ndim == 4:
        return K[view, frame]
    if K.ndim == 3:
        if int(K.shape[0]) == view_count:
            return K[view]
        return K[frame]
    return K


def _select_w2c(w2c: Tensor, *, view: int, frame: int) -> Tensor:
    if w2c.ndim == 4:
        return w2c[view, frame]
    if w2c.ndim == 3:
        return w2c[frame]
    return w2c


def _camera_sequence(K: Tensor, w2c: Tensor, *, view: int, frames: int, view_count: int) -> tuple[Tensor, Tensor]:
    k_seq = torch.stack([_select_k(K, view=view, frame=frame, view_count=view_count) for frame in range(frames)], dim=0)
    w2c_seq = torch.stack([_select_w2c(w2c, view=view, frame=frame) for frame in range(frames)], dim=0)
    return k_seq.to(dtype=torch.float32).contiguous(), w2c_seq.to(dtype=torch.float32).contiguous()


def _camera_path_to_device(path: CameraPathPolynomial, device: torch.device) -> CameraPathPolynomial:
    return CameraPathPolynomial(
        p_coeff=path.p_coeff.to(device),
        frame_times=path.frame_times.to(device),
        fit_error=path.fit_error,
    )


def _unproject_pixels(
    *,
    u: Tensor,
    v: Tensor,
    depth: float,
    K: Tensor,
    w2c: Tensor,
) -> Tensor:
    x_cam = (u - K[0, 2]) / K[0, 0].clamp_min(1.0e-6) * float(depth)
    y_cam = (v - K[1, 2]) / K[1, 1].clamp_min(1.0e-6) * float(depth)
    z_cam = torch.full_like(x_cam, float(depth))
    points_cam = torch.stack((x_cam, y_cam, z_cam, torch.ones_like(x_cam)), dim=-1)
    c2w = torch.linalg.inv(w2c)
    return (points_cam @ c2w.T)[:, :3]


class MulticamPRTWorldTubeModel(torch.nn.Module):
    def __init__(
        self,
        *,
        bundle,
        tube_count: int,
        init_depth: float,
        init_precision_xy: float,
        init_lambda_t: float,
        init_opacity: float,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        train_frames = bundle.train_frames
        view_count, frames, _channels, height, width = train_frames.shape
        sample_ids = torch.randint(view_count * frames * height * width, (tube_count,), generator=generator)
        view_ids = sample_ids // (frames * height * width)
        rem = sample_ids % (frames * height * width)
        frame_ids = rem // (height * width)
        rem = rem % (height * width)
        y_ids = rem // width
        x_ids = rem % width

        world_points = []
        colors = []
        for view, frame, y, x in zip(view_ids.tolist(), frame_ids.tolist(), y_ids.tolist(), x_ids.tolist()):
            K = _select_k(bundle.train_K, view=view, frame=frame, view_count=view_count)
            w2c = _select_w2c(bundle.train_w2c, view=view, frame=frame)
            point = _unproject_pixels(
                u=torch.tensor([float(x) + 0.5], dtype=torch.float32, device=device),
                v=torch.tensor([float(y) + 0.5], dtype=torch.float32, device=device),
                depth=init_depth,
                K=K.to(device),
                w2c=w2c.to(device),
            )[0]
            world_points.append(point)
            colors.append(train_frames[view, frame, :, y, x].to(device))

        times = centered_frame_times(frames, device=device)
        x0 = torch.stack(world_points, dim=0)
        color = torch.stack(colors, dim=0).clamp(0.02, 0.98)
        velocity = torch.zeros_like(x0)
        velocity = velocity + 1.0e-4 * torch.randn(velocity.shape, generator=generator).to(device)

        self.x0 = torch.nn.Parameter(x0)
        self.velocity = torch.nn.Parameter(velocity)
        self.raw_lambda_t = torch.nn.Parameter(
            _softplus_inverse(torch.full((tube_count,), float(init_lambda_t), dtype=torch.float32, device=device))
        )
        self.raw_opacity = torch.nn.Parameter(
            _logit(torch.full((tube_count,), float(init_opacity), dtype=torch.float32, device=device) / 0.99)
        )
        self.raw_color = torch.nn.Parameter(_logit(color))
        self.register_buffer("t0", times[frame_ids.to(device)].to(dtype=torch.float32))
        self.register_buffer(
            "precision_xy",
            torch.full((tube_count, 2), float(init_precision_xy), dtype=torch.float32, device=device),
        )

    def batch(self) -> WorldTubeBatch:
        return WorldTubeBatch(
            x0=self.x0,
            velocity=self.velocity,
            t0=self.t0,
            precision_xy=self.precision_xy,
            lambda_t=torch.nn.functional.softplus(self.raw_lambda_t).clamp_min(1.0e-5),
            opacity=0.99 * torch.sigmoid(self.raw_opacity),
            color=torch.sigmoid(self.raw_color),
        )


def _compile_detached_footprint(model: MulticamPRTWorldTubeModel, camera_path: CameraPathPolynomial):
    projected = compile_projective_rational_tubes(model.batch(), camera_path)
    return type(projected)(
        h_coeff=projected.h_coeff,
        lambda_uv=projected.lambda_uv.detach().contiguous(),
        lambda_t=projected.lambda_t,
        center_t=projected.center_t,
        depth_coeff=projected.depth_coeff,
        opacity=projected.opacity,
        color=projected.color,
        camera_fit_error=projected.camera_fit_error,
    )


def _render_prt_train(model: MulticamPRTWorldTubeModel, camera_path: CameraPathPolynomial, config: UVTRenderConfig) -> Tensor:
    projected = _compile_detached_footprint(model, camera_path)
    return render_projective_rational_tubes_metal_direct_serial_backward(
        projected.h_coeff,
        projected.lambda_uv,
        projected.lambda_t,
        projected.center_t,
        projected.opacity,
        projected.color,
        config,
        forward_mode="tiled",
        backward_mode="tile_pixel_atomic",
    )


def _fused_mse_prt_train_step(
    *,
    model: MulticamPRTWorldTubeModel,
    camera_path: CameraPathPolynomial,
    config: UVTRenderConfig,
    target: Tensor,
) -> float:
    projected = _compile_detached_footprint(model, camera_path)
    result = projective_rational_tile_pixel_fused_mse_backward(
        projected.h_coeff,
        projected.lambda_uv,
        projected.lambda_t,
        projected.center_t,
        projected.opacity,
        projected.color,
        target,
        config,
    )
    if int((result.tile_overflow.detach().cpu() > 0).sum().item()) > 0:
        raise RuntimeError("fused MSE PRT train step overflowed tile capacity")
    backward_pairs = (
        (projected.h_coeff, result.grad_h_coeff),
        (projected.lambda_t, result.grad_lambda_t),
        (projected.center_t, result.grad_center_t),
        (projected.opacity, result.grad_opacity),
        (projected.color, result.grad_color),
    )
    tensors = [tensor for tensor, _grad in backward_pairs if tensor.requires_grad]
    grads = [grad for tensor, grad in backward_pairs if tensor.requires_grad]
    torch.autograd.backward(tensors, grads)
    return float((result.loss_sum.detach().cpu()[0] / float(target.numel())).item())


def _render_prt_eval(model: MulticamPRTWorldTubeModel, camera_path: CameraPathPolynomial, config: UVTRenderConfig):
    projected = _compile_detached_footprint(model, camera_path)
    return _render_prt_projected_eval(projected, config)


def _render_prt_projected_eval(projected, config: UVTRenderConfig):
    return render_projective_rational_tubes_tiled(
        projected.h_coeff,
        projected.lambda_uv,
        projected.lambda_t,
        projected.center_t,
        projected.opacity,
        projected.color,
        config,
        return_aux=True,
    )


def _time_call(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    _sync(device)
    started = time.perf_counter()
    result = fn()
    _sync(device)
    return result, time.perf_counter() - started


def _time_repeated(
    device: torch.device,
    fn: Callable[[], Any],
    *,
    warmups: int,
    repeats: int,
) -> tuple[Any, list[float]]:
    if warmups < 0:
        raise ValueError("render_warmups must be non-negative")
    if repeats <= 0:
        raise ValueError("render_repeats must be positive")
    result = None
    for _ in range(warmups):
        result, _elapsed = _time_call(device, fn)
    samples = []
    for _ in range(repeats):
        result, elapsed = _time_call(device, fn)
        samples.append(elapsed)
    return result, samples


def _summarize_seconds(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_s": samples,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "max_s": max(samples),
    }


def _finite_loss_pair(train_report: dict[str, Any]) -> bool:
    return math.isfinite(float(train_report["initial_loss"])) and math.isfinite(float(train_report["final_loss"]))


def _fit_prt(
    *,
    model: MulticamPRTWorldTubeModel,
    bundle,
    train_camera_paths: list[CameraPathPolynomial],
    config: UVTRenderConfig,
    steps: int,
    lr: float,
    loss_mode: str,
    train_mode: str,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    if loss_mode not in {"sampled_frame", "sequence"}:
        raise ValueError("loss_mode must be one of: sampled_frame, sequence")
    if train_mode not in {"separate", "fused_mse"}:
        raise ValueError("train_mode must be one of: separate, fused_mse")
    if train_mode == "fused_mse" and loss_mode != "sequence":
        raise ValueError("fused_mse train mode currently requires --prt-loss-mode sequence")
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    started = time.perf_counter()
    for step in range(steps + 1):
        view = int(torch.randint(0, bundle.train_view_count, (1,), generator=generator, device=device).item())
        frame = int(torch.randint(0, bundle.frame_count, (1,), generator=generator, device=device).item())
        optimizer.zero_grad(set_to_none=True)
        target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
        if train_mode == "fused_mse":
            loss_value = _fused_mse_prt_train_step(
                model=model,
                camera_path=train_camera_paths[view],
                config=config,
                target=target,
            )
            losses.append({"step": step, "view": view, "frame": frame, "loss": loss_value})
            if step == steps:
                break
        else:
            image = _render_prt_train(model, train_camera_paths[view], config)
            if loss_mode == "sampled_frame":
                loss = (image[frame] - target[frame]).square().mean()
            else:
                loss = (image - target).square().mean()
            losses.append({"step": step, "view": view, "frame": frame, "loss": float(loss.detach().cpu())})
            if step == steps:
                break
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _sync(device)
    return {
        "steps": steps,
        "loss_mode": loss_mode,
        "train_mode": train_mode,
        "train_loop_elapsed_s": time.perf_counter() - started,
        "losses": losses,
        "initial_loss": losses[0]["loss"],
        "final_loss": losses[-1]["loss"],
        "sampled_loss_decreased": losses[-1]["loss"] < losses[0]["loss"],
    }


def _fit_splats(
    *,
    bundle,
    splat_count: int,
    steps: int,
    lr: float,
    init_depth: float,
    init_scale: float,
    renderer: str,
    camera_projection: str,
    device: torch.device,
    seed: int,
) -> tuple[FreeDynamic3DGS, SplatRenderConfig, dict[str, Any]]:
    torch.manual_seed(seed)
    view_count, frames, _channels, height, width = bundle.train_frames.shape
    train_video = bundle.train_frames
    init_xyz, init_rgb = initialize_material_points_from_first_frame(
        video=train_video[0].permute(0, 2, 3, 1).contiguous(),
        K=bundle.train_K[0],
        num_elements=splat_count,
        init_depth=init_depth,
    )
    model = FreeDynamic3DGS(
        init_xyz=init_xyz,
        init_rgb=init_rgb,
        num_frames=frames,
        splat_mode="per_frame",
        init_scale=init_scale,
        scale_init_log_jitter=0.0,
        init_alpha_logit=0.0,
        init_xyz_noise=0.001,
        init_quat_noise=0.0,
        log_scale_min=-12.0,
        log_scale_max=4.0,
    ).to(device)
    render_cfg = SplatRenderConfig(
        height=height,
        width=width,
        renderer=renderer,
        tile_size=16 if renderer == "fast_mac" else 8,
        bound_scale=3.0,
        alpha_threshold=1.0 / 255.0,
        near_plane=1.0e-3,
        camera_projection="camera_model" if camera_projection == "dataset_lens" else "legacy_pinhole",
    )
    generator = torch.Generator(device=device).manual_seed(seed + 991)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    started = time.perf_counter()
    for step in range(steps + 1):
        view = int(torch.randint(0, view_count, (1,), generator=generator, device=device).item())
        frame = int(torch.randint(0, frames, (1,), generator=generator, device=device).item())
        optimizer.zero_grad(set_to_none=True)
        camera = camera_from_K_w2c(
            _select_k(bundle.train_K, view=view, frame=frame, view_count=view_count),
            _select_w2c(bundle.train_w2c, view=view, frame=frame),
        )
        image = render_gaussian_frame(
            model.frame(frame),
            camera,
            height=height,
            width=width,
            mode=render_cfg.renderer,
            tile_size=render_cfg.tile_size,
            bound_scale=render_cfg.bound_scale,
            alpha_threshold=render_cfg.alpha_threshold,
            near_plane=render_cfg.near_plane,
            camera_projection=render_cfg.camera_projection,
        ).permute(1, 2, 0)
        target = bundle.train_frames[view, frame].permute(1, 2, 0).contiguous()
        loss = (image - target).square().mean()
        loss = loss + 1.0e-4 * model.scale_loss() + 1.0e-3 * model.temporal_smoothness_loss()
        losses.append({"step": step, "view": view, "frame": frame, "loss": float(loss.detach().cpu())})
        if step == steps:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _sync(device)
    return (
        model,
        render_cfg,
        {
            "steps": steps,
            "train_loop_elapsed_s": time.perf_counter() - started,
            "losses": losses,
            "initial_loss": losses[0]["loss"],
            "final_loss": losses[-1]["loss"],
            "sampled_loss_decreased": losses[-1]["loss"] < losses[0]["loss"],
        },
    )


@torch.no_grad()
def _eval_prt(
    *,
    model: MulticamPRTWorldTubeModel,
    bundle,
    train_camera_paths: list[CameraPathPolynomial],
    heldout_camera_paths: list[CameraPathPolynomial],
    config: UVTRenderConfig,
    device: torch.device,
    render_warmups: int,
    render_repeats: int,
    cache_compiled: bool,
) -> dict[str, Any]:
    train_rows = []
    train_times = []
    compile_times = []
    max_tile_count = 0
    overflow_tile_count = 0
    for view, camera_path in enumerate(train_camera_paths):
        if cache_compiled:
            projected, compile_elapsed = _time_call(
                device,
                lambda camera_path=camera_path: _compile_detached_footprint(model, camera_path),
            )
            compile_times.append(compile_elapsed)
            render_fn = lambda projected=projected: _render_prt_projected_eval(projected, config)
        else:
            render_fn = lambda camera_path=camera_path: _render_prt_eval(model, camera_path, config)
        aux, elapsed = _time_repeated(
            device,
            render_fn,
            warmups=render_warmups,
            repeats=render_repeats,
        )
        train_times.extend(elapsed)
        max_tile_count = max(max_tile_count, int(aux.tile_counts.max().detach().cpu()))
        overflow_tile_count += int((aux.tile_overflow > 0).sum().detach().cpu())
        target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
        train_rows.append(_metrics(aux.image, target))

    heldout_rows = []
    heldout_times = []
    if bundle.heldout_frames is not None:
        for view, camera_path in enumerate(heldout_camera_paths):
            if cache_compiled:
                projected, compile_elapsed = _time_call(
                    device,
                    lambda camera_path=camera_path: _compile_detached_footprint(model, camera_path),
                )
                compile_times.append(compile_elapsed)
                render_fn = lambda projected=projected: _render_prt_projected_eval(projected, config)
            else:
                render_fn = lambda camera_path=camera_path: _render_prt_eval(model, camera_path, config)
            aux, elapsed = _time_repeated(
                device,
                render_fn,
                warmups=render_warmups,
                repeats=render_repeats,
            )
            heldout_times.extend(elapsed)
            max_tile_count = max(max_tile_count, int(aux.tile_counts.max().detach().cpu()))
            overflow_tile_count += int((aux.tile_overflow > 0).sum().detach().cpu())
            target = bundle.heldout_frames[view].permute(0, 2, 3, 1).contiguous()
            heldout_rows.append(_metrics(aux.image, target))

    metrics = _aggregate_metric_rows(train_rows)
    if heldout_rows:
        metrics.update({f"heldout_{key}": value for key, value in _aggregate_metric_rows(heldout_rows).items()})
    return {
        "metrics": metrics,
        "train_render_seconds": _summarize_seconds(train_times),
        "heldout_render_seconds": None if not heldout_times else _summarize_seconds(heldout_times),
        "cache_compiled": cache_compiled,
        "compile_seconds": None if not compile_times else _summarize_seconds(compile_times),
        "max_tile_count": max_tile_count,
        "overflow_tile_count": overflow_tile_count,
    }


@torch.no_grad()
def _eval_splats(
    *,
    model: FreeDynamic3DGS,
    render_cfg: SplatRenderConfig,
    bundle,
    camera_projection: str,
    device: torch.device,
    render_warmups: int,
    render_repeats: int,
) -> dict[str, Any]:
    train_rows = []
    train_times = []
    for view in range(bundle.train_view_count):
        cameras = [
            camera_from_K_w2c(
                _select_k(bundle.train_K, view=view, frame=frame, view_count=bundle.train_view_count),
                _select_w2c(bundle.train_w2c, view=view, frame=frame),
            )
            for frame in range(bundle.frame_count)
        ]
        rendered, elapsed = _time_repeated(
            device,
            lambda cameras=cameras: render_splat_sequence(model, cameras, render_cfg),
            warmups=render_warmups,
            repeats=render_repeats,
        )
        train_times.extend(elapsed)
        target = bundle.train_frames[view].permute(0, 2, 3, 1).contiguous()
        train_rows.append(_metrics(rendered["rgb"], target))

    heldout_rows = []
    heldout_times = []
    if bundle.heldout_frames is not None and bundle.heldout_K is not None and bundle.heldout_w2c is not None:
        for view in range(bundle.heldout_view_count):
            cameras = [
                camera_from_K_w2c(
                    _select_k(bundle.heldout_K, view=view, frame=frame, view_count=bundle.heldout_view_count),
                    _select_w2c(bundle.heldout_w2c, view=view, frame=frame),
                )
                for frame in range(bundle.frame_count)
            ]
            rendered, elapsed = _time_repeated(
                device,
                lambda cameras=cameras: render_splat_sequence(model, cameras, render_cfg),
                warmups=render_warmups,
                repeats=render_repeats,
            )
            heldout_times.extend(elapsed)
            target = bundle.heldout_frames[view].permute(0, 2, 3, 1).contiguous()
            heldout_rows.append(_metrics(rendered["rgb"], target))

    metrics = _aggregate_metric_rows(train_rows)
    if heldout_rows:
        metrics.update({f"heldout_{key}": value for key, value in _aggregate_metric_rows(heldout_rows).items()})
    return {
        "metrics": metrics,
        "train_render_seconds": _summarize_seconds(train_times),
        "heldout_render_seconds": None if not heldout_times else _summarize_seconds(heldout_times),
        "camera_projection": camera_projection,
    }


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "mps":
        raise ValueError("projective rational multicam compare currently requires --device=mps")
    torch.manual_seed(args.seed)

    config = load_config_file(_resolve_dynaworld_path(args.baseline_config))
    data_cfg = dict(config["data"])
    camera_cfg = dict(config["camera"])
    if data_cfg.get("multicam_manifest") is not None:
        data_cfg["multicam_manifest"] = str(_resolve_dynaworld_path(data_cfg["multicam_manifest"]))
    data_cfg["max_frames"] = int(args.max_frames)
    bundle = load_multicam_video_bundle(
        data_cfg=data_cfg,
        camera_cfg=camera_cfg,
        target_size=int(args.target_size),
        device=device,
    )
    _views, frames, _channels, height, width = bundle.train_frames.shape
    frame_times_cpu = centered_frame_times(frames, device="cpu")

    train_camera_paths = []
    train_fit_errors = []
    for view in range(bundle.train_view_count):
        k_seq, w2c_seq = _camera_sequence(bundle.train_K.detach().cpu(), bundle.train_w2c.detach().cpu(), view=view, frames=frames, view_count=bundle.train_view_count)
        path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=args.camera_poly_degree, frame_times=frame_times_cpu)
        train_fit_errors.append(path.fit_error)
        train_camera_paths.append(_camera_path_to_device(path, device))

    heldout_camera_paths = []
    heldout_fit_errors = []
    if bundle.heldout_K is not None and bundle.heldout_w2c is not None:
        for view in range(bundle.heldout_view_count):
            k_seq, w2c_seq = _camera_sequence(bundle.heldout_K.detach().cpu(), bundle.heldout_w2c.detach().cpu(), view=view, frames=frames, view_count=bundle.heldout_view_count)
            path = fit_camera_path_polynomial(k_seq, w2c_seq, degree=args.camera_poly_degree, frame_times=frame_times_cpu)
            heldout_fit_errors.append(path.fit_error)
            heldout_camera_paths.append(_camera_path_to_device(path, device))

    prt_support_alpha_threshold = args.prt_support_alpha_threshold
    if args.prt_tile_policy == "train_speed":
        if args.tile_config != "auto":
            raise ValueError("--prt-tile-policy train_speed requires --tile-config auto")
        tile_policy = recommend_projective_rational_train_speed_tile_policy(tube_count=args.prt_tubes)
        tile_config = tile_policy.tile_config
        if prt_support_alpha_threshold is None:
            prt_support_alpha_threshold = tile_policy.support_alpha_threshold
        prt_tile_policy = tile_policy.name
    else:
        tile_config = (
            recommend_projective_rational_tile_config(tube_count=args.prt_tubes)
            if args.tile_config == "auto"
            else parse_projective_rational_tile_config(args.tile_config)
        )
        prt_tile_policy = "generic_auto" if args.tile_config == "auto" else "explicit_tile_config"
    prt_eval_support_alpha_threshold = (
        prt_support_alpha_threshold
        if args.prt_eval_support_alpha_threshold is None
        else args.prt_eval_support_alpha_threshold
    )
    apply_projective_rational_tile_env(tile_config)
    prt_train_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        alpha_threshold=args.prt_alpha_threshold,
        support_alpha_threshold=prt_support_alpha_threshold,
        background=(1.0, 1.0, 1.0),
        **tile_config.as_render_kwargs(),
    )
    prt_eval_config = UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        alpha_threshold=args.prt_alpha_threshold,
        support_alpha_threshold=prt_eval_support_alpha_threshold,
        background=(1.0, 1.0, 1.0),
        **tile_config.as_render_kwargs(),
    )
    prt_model = MulticamPRTWorldTubeModel(
        bundle=bundle,
        tube_count=args.prt_tubes,
        init_depth=args.init_depth,
        init_precision_xy=args.prt_init_precision_xy,
        init_lambda_t=args.prt_init_lambda_t,
        init_opacity=args.prt_init_opacity,
        seed=args.seed,
        device=device,
    ).to(device)
    prt_train = _fit_prt(
        model=prt_model,
        bundle=bundle,
        train_camera_paths=train_camera_paths,
        config=prt_train_config,
        steps=args.steps,
        lr=args.prt_lr,
        loss_mode=args.prt_loss_mode,
        train_mode=args.prt_train_mode,
        device=device,
        seed=args.seed + 17,
    )
    prt_eval = _eval_prt(
        model=prt_model,
        bundle=bundle,
        train_camera_paths=train_camera_paths,
        heldout_camera_paths=heldout_camera_paths,
        config=prt_eval_config,
        device=device,
        render_warmups=args.render_warmups,
        render_repeats=args.render_repeats,
        cache_compiled=args.prt_eval_cache_compiled,
    )

    splat_model, splat_render_cfg, splat_train = _fit_splats(
        bundle=bundle,
        splat_count=args.splat_count,
        steps=args.steps,
        lr=args.splat_lr,
        init_depth=args.init_depth,
        init_scale=args.splat_init_scale,
        renderer=args.splat_renderer,
        camera_projection=args.splat_camera_projection,
        device=device,
        seed=args.seed,
    )
    splat_eval = _eval_splats(
        model=splat_model,
        render_cfg=splat_render_cfg,
        bundle=bundle,
        camera_projection=args.splat_camera_projection,
        device=device,
        render_warmups=args.render_warmups,
        render_repeats=args.render_repeats,
    )

    return {
        "name": "projective_rational_multicam_splat_compare",
        "note": (
            "Small real-multicam gate. PRT trains world tubes with the projective-rational tiled Metal path. "
            "The baseline is the existing FreeDynamic3DGS per-frame direct-splat model."
        ),
        "pass": (
            _finite_loss_pair(prt_train)
            and _finite_loss_pair(splat_train)
            and int(prt_eval["overflow_tile_count"]) == 0
        ),
        "meta": {
            "baseline_config": str(_resolve_dynaworld_path(args.baseline_config)),
            "target_size": args.target_size,
            "max_frames": args.max_frames,
            "frames": frames,
            "height": height,
            "width": width,
            "device": str(device),
            "seed": args.seed,
            "steps": args.steps,
            "render_warmups": args.render_warmups,
            "render_repeats": args.render_repeats,
            "prt_eval_cache_compiled": args.prt_eval_cache_compiled,
            "train_cameras": bundle.train_camera_names,
            "heldout_cameras": bundle.heldout_camera_names,
            "pose_source": bundle.pose_source,
            "sample_id": None if bundle.metadata is None else bundle.metadata.get("sample_id"),
            "camera_poly_degree": args.camera_poly_degree,
            "prt_alpha_threshold": args.prt_alpha_threshold,
            "prt_support_alpha_threshold": prt_support_alpha_threshold,
            "prt_train_support_alpha_threshold": prt_support_alpha_threshold,
            "prt_eval_support_alpha_threshold": prt_eval_support_alpha_threshold,
            "prt_tile_policy": prt_tile_policy,
            "train_camera_fit_errors": train_fit_errors,
            "heldout_camera_fit_errors": heldout_fit_errors,
        },
        "projective_rational": {
            "tube_count": args.prt_tubes,
            "parameter_count": sum(parameter.numel() for parameter in prt_model.parameters()),
            "tile_config_key": tile_config.key,
            "tile_config": tile_config.as_dict(),
            "train_support_alpha_threshold": prt_support_alpha_threshold,
            "eval_support_alpha_threshold": prt_eval_support_alpha_threshold,
            "lr": args.prt_lr,
            "loss_mode": args.prt_loss_mode,
            "train_mode": args.prt_train_mode,
            "init_depth": args.init_depth,
            "init_precision_xy": args.prt_init_precision_xy,
            "init_lambda_t": args.prt_init_lambda_t,
            "init_opacity": args.prt_init_opacity,
            "train": prt_train,
            "eval": prt_eval,
        },
        "free_dynamic_splats": {
            "splat_count": args.splat_count,
            "parameter_count": sum(parameter.numel() for parameter in splat_model.parameters()),
            "renderer": args.splat_renderer,
            "camera_projection": args.splat_camera_projection,
            "lr": args.splat_lr,
            "init_depth": args.init_depth,
            "init_scale": args.splat_init_scale,
            "train": splat_train,
            "eval": splat_eval,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--camera-poly-degree", type=int, default=1)
    parser.add_argument("--tile-config", default="auto", help="'auto' or an explicit config like 8x8x2:128")
    parser.add_argument(
        "--prt-tile-policy",
        choices=("generic", "train_speed"),
        default="generic",
        help="generic preserves the fail-closed auto selector; train_speed opts into measured support-pruned 1024 policy.",
    )
    parser.add_argument("--prt-tubes", type=int, default=128)
    parser.add_argument("--prt-lr", type=float, default=0.02)
    parser.add_argument("--prt-loss-mode", choices=("sampled_frame", "sequence"), default="sampled_frame")
    parser.add_argument("--prt-train-mode", choices=("separate", "fused_mse"), default="separate")
    parser.add_argument("--prt-init-precision-xy", type=float, default=36.0)
    parser.add_argument("--prt-init-lambda-t", type=float, default=0.25)
    parser.add_argument("--prt-init-opacity", type=float, default=0.35)
    parser.add_argument("--prt-alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--prt-support-alpha-threshold", type=float)
    parser.add_argument("--prt-eval-support-alpha-threshold", type=float)
    parser.add_argument("--splat-count", type=int, default=128)
    parser.add_argument("--splat-lr", type=float, default=0.002)
    parser.add_argument("--splat-renderer", choices=("dense", "fast_mac"), default="dense")
    parser.add_argument("--splat-camera-projection", choices=("legacy_pinhole", "dataset_lens"), default="legacy_pinhole")
    parser.add_argument("--splat-init-scale", type=float, default=0.035)
    parser.add_argument("--init-depth", type=float, default=0.5)
    parser.add_argument("--render-warmups", type=int, default=0)
    parser.add_argument("--render-repeats", type=int, default=1)
    parser.add_argument(
        "--prt-eval-cache-compiled",
        action="store_true",
        help="Precompile PRT eval footprints once per camera and time only tiled rasterization.",
    )
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = run_compare(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
