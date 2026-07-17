from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch import Tensor


SCRIPT_PATH = Path(__file__).resolve()
VARIANT_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]
for _path in (VARIANT_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


WORKER1_DENSE_CANDIDATES = (
    "torch_gsplat_bridge_star_prt",
    "research_project.trainer_harness.curve_tube",
    "research_project.trainer_harness.star_prt_dense",
    "research_project.trainer_harness.star_prt_curve_dense",
    "research_project.trainer_harness.projective_rational",
    "research_project.dense_prt",
    "star_prt_dense",
    "dense_prt",
)

WORKER1_DENSE_FUNCTIONS = (
    "dense_render_compiled_curve_tubes",
    "dense_render_projective_rational_curves",
    "dense_render_projective_rational_tubes",
    "render_projective_rational_curves_dense",
    "render_dense_projective_rational",
)


@dataclass(frozen=True)
class Scene:
    x0: Tensor
    velocity: Tensor
    sigma_px: Tensor
    opacity: Tensor
    color: Tensor


@dataclass(frozen=True)
class ProjectedCurves:
    h_coeff: Tensor
    sigma_px: Tensor
    opacity: Tensor
    color: Tensor
    camera_fit_error: float
    polynomial_degree: int


@dataclass(frozen=True)
class DenseRendererAdapter:
    info: dict[str, Any]
    config_cls: Any | None = None
    render_compiled_fn: Any | None = None

    def render(self, centers: Tensor, depth: Tensor, scene: Scene, *, height: int, width: int) -> Tensor:
        if self.config_cls is None or self.render_compiled_fn is None:
            return render_dense_gaussian(centers, depth, scene, height=height, width=width)
        config = self.config_cls(height=height, width=width, frames=int(centers.shape[0]))
        curve_uv_depth = torch.cat((centers, depth.unsqueeze(-1)), dim=-1).to(torch.float32).contiguous()
        lambda_uv = scene_lambda_uv(scene)
        lambda_t = torch.zeros((int(scene.x0.shape[0]),), dtype=torch.float32)
        center_t = torch.zeros((int(scene.x0.shape[0]),), dtype=torch.float32)
        return self.render_compiled_fn(
            curve_uv_depth,
            lambda_uv,
            lambda_t,
            center_t,
            scene.opacity.contiguous(),
            scene.color.contiguous(),
            config,
        )


def centered_times(frames: int) -> Tensor:
    values = torch.arange(frames, dtype=torch.float32)
    return values - 0.5 * float(frames - 1)


def load_dense_renderer() -> DenseRendererAdapter:
    import_errors: dict[str, str] = {}
    for name in WORKER1_DENSE_CANDIDATES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            import_errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        for fn_name in WORKER1_DENSE_FUNCTIONS:
            fn = getattr(module, fn_name, None)
            if callable(fn):
                signature = inspect.signature(fn)
                config_cls = getattr(module, "PRTRenderConfig", None)
                if config_cls is None:
                    config_cls = getattr(module, "CurveTubeRenderConfig", None)
                if (
                    fn_name == "dense_render_compiled_curve_tubes"
                    and config_cls is not None
                    and len(signature.parameters) >= 7
                ):
                    return DenseRendererAdapter(
                        info={
                            "available": True,
                            "module": name,
                            "function": fn_name,
                            "signature": str(signature),
                            "used": True,
                        },
                        config_cls=config_cls,
                        render_compiled_fn=fn,
                    )
                return DenseRendererAdapter(info={
                    "available": True,
                    "module": name,
                    "function": fn_name,
                    "signature": str(signature),
                    "used": False,
                    "todo": (
                        "TODO(worker1): wire this benchmark's ProjectedCurves adapter "
                        "to the dense module once the STAR-PRT curve API is finalized."
                    ),
                })
        return DenseRendererAdapter(info={
            "available": True,
            "module": name,
            "function": None,
            "used": False,
            "todo": (
                "TODO(worker1): module imported, but no known dense-render function "
                f"was found. Expected one of {list(WORKER1_DENSE_FUNCTIONS)}."
            ),
        })
    return DenseRendererAdapter(info={
        "available": False,
        "module": None,
        "function": None,
        "used": False,
        "candidates": list(WORKER1_DENSE_CANDIDATES),
        "todo": (
            "TODO(worker1): add a dense STAR-PRT curve module under one of the "
            "candidate import paths, or update WORKER1_DENSE_CANDIDATES here."
        ),
        "import_errors": import_errors,
    })


def make_scene(primitives: int, *, seed: int) -> Scene:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.rand((primitives,), generator=generator) - 0.5) * 3.2
    y = (torch.rand((primitives,), generator=generator) - 0.5) * 2.2
    z = 3.2 + torch.rand((primitives,), generator=generator) * 3.8
    x0 = torch.stack((x, y, z), dim=-1).to(torch.float32)

    velocity = torch.randn((primitives, 3), generator=generator, dtype=torch.float32)
    velocity[:, 0] *= 0.035
    velocity[:, 1] *= 0.025
    velocity[:, 2] *= 0.018

    sigma_px = 1.1 + torch.rand((primitives, 2), generator=generator, dtype=torch.float32) * 1.8
    opacity = 0.18 + torch.rand((primitives,), generator=generator, dtype=torch.float32) * 0.52
    color = torch.rand((primitives, 3), generator=generator, dtype=torch.float32).mul(0.85).add(0.08)
    return Scene(x0=x0, velocity=velocity, sigma_px=sigma_px, opacity=opacity, color=color)


def make_camera_path(times: Tensor, *, height: int, width: int) -> tuple[Tensor, Tensor]:
    frames = int(times.numel())
    scale = times / times.abs().max().clamp_min(1.0)
    k_seq = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(frames, 1, 1)
    focal = 0.88 * float(min(height, width))
    zoom = 1.0 + 0.035 * scale + 0.018 * scale.square()
    k_seq[:, 0, 0] = focal * zoom
    k_seq[:, 1, 1] = focal * (1.0 - 0.012 * scale)
    k_seq[:, 0, 2] = 0.5 * float(width) + 0.9 * scale
    k_seq[:, 1, 2] = 0.5 * float(height) - 0.6 * scale.square()

    w2c_seq = torch.eye(4, dtype=torch.float32).view(1, 4, 4).repeat(frames, 1, 1)
    yaw = 0.055 * torch.sin(1.25 * scale) + 0.018 * scale
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    w2c_seq[:, 0, 0] = cos_y
    w2c_seq[:, 0, 2] = sin_y
    w2c_seq[:, 2, 0] = -sin_y
    w2c_seq[:, 2, 2] = cos_y
    w2c_seq[:, 0, 3] = 0.18 * scale + 0.035 * scale.square()
    w2c_seq[:, 1, 3] = -0.05 * torch.sin(1.7 * scale)
    w2c_seq[:, 2, 3] = -0.16 * scale.square()
    return k_seq, w2c_seq


def projection_matrices(k_seq: Tensor, w2c_seq: Tensor) -> Tensor:
    return torch.bmm(k_seq, w2c_seq[:, :3, :])


def project_points(p_matrix: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
    ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
    hom = torch.cat((points, ones), dim=-1)
    if p_matrix.ndim == 2:
        h = hom @ p_matrix.T
    elif hom.ndim == 3:
        h = torch.einsum("frc,fnc->fnr", p_matrix, hom)
    else:
        h = torch.einsum("nrc,nc->nr", p_matrix, hom)
    depth = h[..., 2]
    safe_depth = depth.clamp_min(1.0e-6)
    centers = torch.stack((h[..., 0] / safe_depth, h[..., 1] / safe_depth), dim=-1)
    return centers, depth


def direct_project(scene: Scene, p_seq: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
    points = scene.x0.view(1, -1, 3) + scene.velocity.view(1, -1, 3) * times.view(-1, 1, 1)
    return project_points(p_seq, points)


def static_affine_star_project(scene: Scene, p_seq: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
    ref_idx = int(torch.argmin(times.abs()).item())
    ref_t = times[ref_idx]
    p_ref = p_seq[ref_idx]
    eps = torch.tensor(1.0e-2, dtype=torch.float32)
    ref_points = scene.x0 + scene.velocity * ref_t
    centers_ref, depth_ref = project_points(p_ref, ref_points)
    centers_plus, depth_plus = project_points(p_ref, ref_points + scene.velocity * eps)
    center_slope = (centers_plus - centers_ref) / eps
    depth_slope = (depth_plus - depth_ref) / eps
    dt = (times - ref_t).view(-1, 1, 1)
    centers = centers_ref.view(1, -1, 2) + center_slope.view(1, -1, 2) * dt
    depth = depth_ref.view(1, -1) + depth_slope.view(1, -1) * (times - ref_t).view(-1, 1)
    return centers, depth


def first_order_projective_affine(direct_centers: Tensor, direct_depth: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
    frames = int(times.numel())
    ref_idx = int(torch.argmin(times.abs()).item())
    left = max(0, ref_idx - 1)
    right = min(frames - 1, ref_idx + 1)
    denom = (times[right] - times[left]).clamp_min(1.0e-6)
    center_slope = (direct_centers[right] - direct_centers[left]) / denom
    depth_slope = (direct_depth[right] - direct_depth[left]) / denom
    dt = (times - times[ref_idx]).view(-1, 1, 1)
    centers = direct_centers[ref_idx].view(1, -1, 2) + center_slope.view(1, -1, 2) * dt
    depth = direct_depth[ref_idx].view(1, -1) + depth_slope.view(1, -1) * (times - times[ref_idx]).view(-1, 1)
    return centers, depth


def segmented_affine_project(direct_centers: Tensor, direct_depth: Tensor, times: Tensor, *, segment_frames: int) -> tuple[Tensor, Tensor]:
    frames, primitives, _ = direct_centers.shape
    centers = torch.empty_like(direct_centers)
    depth = torch.empty_like(direct_depth)
    stride = max(1, int(segment_frames) - 1)
    anchors = list(range(0, frames, stride))
    if anchors[-1] != frames - 1:
        anchors.append(frames - 1)
    for start, end in zip(anchors[:-1], anchors[1:]):
        denom = (times[end] - times[start]).clamp_min(1.0e-6)
        for frame in range(start, end + 1):
            alpha = ((times[frame] - times[start]) / denom).clamp(0.0, 1.0)
            centers[frame] = torch.lerp(direct_centers[start], direct_centers[end], alpha)
            depth[frame] = torch.lerp(direct_depth[start], direct_depth[end], alpha)
    if frames == 1:
        centers[0] = direct_centers[0]
        depth[0] = direct_depth[0]
    return centers.view(frames, primitives, 2), depth.view(frames, primitives)


def fit_projection_polynomial(p_seq: Tensor, times: Tensor, *, degree: int) -> tuple[Tensor, float]:
    frames = int(times.numel())
    degree = min(int(degree), frames - 1)
    vandermonde = torch.stack([times.pow(k) for k in range(degree + 1)], dim=-1)
    solution = torch.linalg.lstsq(vandermonde.to(torch.float64), p_seq.reshape(frames, 12).to(torch.float64)).solution
    coeff = solution.to(torch.float32).reshape(degree + 1, 3, 4)
    recon = torch.einsum("fd,drc->frc", vandermonde, coeff)
    denom = p_seq.reshape(frames, -1).norm(dim=1).clamp_min(1.0e-8)
    fit_error = float(((recon - p_seq).reshape(frames, -1).norm(dim=1) / denom).max().detach().cpu())
    return coeff, fit_error


def compile_projective_curves(scene: Scene, p_coeff: Tensor, *, camera_fit_error: float) -> ProjectedCurves:
    degree = int(p_coeff.shape[0]) - 1
    h_coeff = scene.x0.new_zeros((int(scene.x0.shape[0]), degree + 2, 3))
    x_h = torch.cat((scene.x0, torch.ones((int(scene.x0.shape[0]), 1), dtype=torch.float32)), dim=-1)
    v_h = torch.cat((scene.velocity, torch.zeros((int(scene.x0.shape[0]), 1), dtype=torch.float32)), dim=-1)
    for power in range(degree + 2):
        if power <= degree:
            h_coeff[:, power] = h_coeff[:, power] + torch.einsum("rc,nc->nr", p_coeff[power], x_h)
        if 0 <= power - 1 <= degree:
            h_coeff[:, power] = h_coeff[:, power] + torch.einsum("rc,nc->nr", p_coeff[power - 1], v_h)
    return ProjectedCurves(
        h_coeff=h_coeff,
        sigma_px=scene.sigma_px,
        opacity=scene.opacity,
        color=scene.color,
        camera_fit_error=camera_fit_error,
        polynomial_degree=degree,
    )


def evaluate_projective_curves(projected: ProjectedCurves, times: Tensor) -> tuple[Tensor, Tensor]:
    powers = torch.stack([times.pow(k) for k in range(int(projected.h_coeff.shape[1]))], dim=-1)
    h = torch.einsum("fd,ndc->fnc", powers, projected.h_coeff)
    depth = h[..., 2]
    safe_depth = depth.clamp_min(1.0e-6)
    centers = torch.stack((h[..., 0] / safe_depth, h[..., 1] / safe_depth), dim=-1)
    return centers, depth


def render_dense_gaussian(
    centers: Tensor,
    depth: Tensor,
    scene: Scene,
    *,
    height: int,
    width: int,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    alpha_threshold: float = 1.0 / 255.0,
) -> Tensor:
    frames = int(centers.shape[0])
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) + 0.5,
        torch.arange(width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    image = torch.empty((frames, height, width, 3), dtype=torch.float32)
    bg = torch.tensor(background, dtype=torch.float32).view(1, 1, 3)
    for frame in range(frames):
        valid = torch.isfinite(centers[frame]).all(dim=-1) & torch.isfinite(depth[frame]) & (depth[frame] > 1.0e-5)
        order = torch.argsort(depth[frame].where(valid, torch.full_like(depth[frame], float("inf"))), stable=True)
        accum = torch.zeros((height, width, 3), dtype=torch.float32)
        transmittance = torch.ones((height, width, 1), dtype=torch.float32)
        for primitive_id in order.tolist():
            if not bool(valid[primitive_id]):
                continue
            sigma_x = scene.sigma_px[primitive_id, 0].clamp_min(0.35)
            sigma_y = scene.sigma_px[primitive_id, 1].clamp_min(0.35)
            du = (xx - centers[frame, primitive_id, 0]) / sigma_x
            dv = (yy - centers[frame, primitive_id, 1]) / sigma_y
            alpha = scene.opacity[primitive_id] * torch.exp(-0.5 * (du.square() + dv.square()))
            alpha = torch.where(alpha >= alpha_threshold, alpha.clamp(max=0.995), torch.zeros_like(alpha))
            alpha3 = alpha.unsqueeze(-1)
            accum = accum + transmittance * alpha3 * scene.color[primitive_id].view(1, 1, 3)
            transmittance = transmittance * (1.0 - alpha3)
        image[frame] = accum + transmittance * bg
    return image.clamp(0.0, 1.0)


def scene_lambda_uv(scene: Scene) -> Tensor:
    inv_sigma2 = scene.sigma_px.clamp_min(0.35).square().reciprocal()
    return torch.stack(
        (
            inv_sigma2[:, 0],
            torch.zeros_like(inv_sigma2[:, 0]),
            inv_sigma2[:, 1],
        ),
        dim=-1,
    ).to(torch.float32).contiguous()


def projected_count(centers: Tensor, depth: Tensor, scene: Scene, *, height: int, width: int) -> int:
    support = 3.0 * scene.sigma_px.max(dim=1).values.view(1, -1)
    valid = torch.isfinite(centers).all(dim=-1) & torch.isfinite(depth) & (depth > 1.0e-5)
    in_bounds = (
        (centers[..., 0] >= -support)
        & (centers[..., 0] <= float(width) + support)
        & (centers[..., 1] >= -support)
        & (centers[..., 1] <= float(height) + support)
    )
    return int((valid & in_bounds).sum().item())


def residual_stats(centers: Tensor, reference: Tensor, depth: Tensor, reference_depth: Tensor) -> dict[str, float]:
    mask = (
        torch.isfinite(centers).all(dim=-1)
        & torch.isfinite(reference).all(dim=-1)
        & torch.isfinite(depth)
        & torch.isfinite(reference_depth)
        & (depth > 1.0e-5)
        & (reference_depth > 1.0e-5)
    )
    if not bool(mask.any()):
        return {"mean_px": float("inf"), "rms_px": float("inf"), "p95_px": float("inf"), "max_px": float("inf")}
    error = (centers - reference).norm(dim=-1)[mask]
    return {
        "mean_px": float(error.mean().detach().cpu()),
        "rms_px": float(torch.sqrt(error.square().mean()).detach().cpu()),
        "p95_px": float(torch.quantile(error, 0.95).detach().cpu()),
        "max_px": float(error.max().detach().cpu()),
    }


def rgb_metrics(image: Tensor, reference: Tensor) -> dict[str, float]:
    diff = image - reference
    mse = float(diff.square().mean().detach().cpu())
    psnr = -10.0 * math.log10(max(mse, 1.0e-12))
    return {
        "mse": mse,
        "psnr": psnr,
        "l1": float(diff.abs().mean().detach().cpu()),
    }


def timed_projection(fn) -> tuple[tuple[Tensor, Tensor], float]:
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def timed_render(fn) -> tuple[Tensor, float]:
    start = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def method_row(
    *,
    name: str,
    centers: Tensor,
    depth: Tensor,
    image: Tensor,
    reference_centers: Tensor,
    reference_depth: Tensor,
    reference_image: Tensor,
    scene: Scene,
    height: int,
    width: int,
    projection_ms: float,
    render_ms: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "center_residual_px": residual_stats(centers, reference_centers, depth, reference_depth),
        "rgb": rgb_metrics(image, reference_image),
        "projected_primitive_count": projected_count(centers, depth, scene, height=height, width=width),
        "elapsed_ms": {
            "projection": projection_ms,
            "render": render_ms,
            "total": projection_ms + render_ms,
        },
    }
    if extra:
        row.update(extra)
    return row


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(max(1, int(args.torch_threads)))
    scene = make_scene(int(args.primitives), seed=int(args.seed))
    times = centered_times(int(args.frames))
    k_seq, w2c_seq = make_camera_path(times, height=int(args.height), width=int(args.width))
    p_seq = projection_matrices(k_seq, w2c_seq)
    dense_renderer = load_dense_renderer()

    (reference_centers, reference_depth), reference_projection_ms = timed_projection(lambda: direct_project(scene, p_seq, times))
    reference_image, reference_render_ms = timed_render(
        lambda: dense_renderer.render(
            reference_centers,
            reference_depth,
            scene,
            height=int(args.height),
            width=int(args.width),
        )
    )

    methods: dict[str, Any] = {}

    (centers, depth), projection_ms = timed_projection(lambda: static_affine_star_project(scene, p_seq, times))
    image, render_ms = timed_render(
        lambda: dense_renderer.render(centers, depth, scene, height=int(args.height), width=int(args.width))
    )
    methods["static_affine_star"] = method_row(
        name="static_affine_star",
        centers=centers,
        depth=depth,
        image=image,
        reference_centers=reference_centers,
        reference_depth=reference_depth,
        reference_image=reference_image,
        scene=scene,
        height=int(args.height),
        width=int(args.width),
        projection_ms=projection_ms,
        render_ms=render_ms,
        extra={"description": "Reference-frame STAR-style affine screen tube; camera motion is ignored."},
    )

    (centers, depth), projection_ms = timed_projection(
        lambda: first_order_projective_affine(reference_centers, reference_depth, times)
    )
    image, render_ms = timed_render(
        lambda: dense_renderer.render(centers, depth, scene, height=int(args.height), width=int(args.width))
    )
    methods["projective_first_order_affine"] = method_row(
        name="projective_first_order_affine",
        centers=centers,
        depth=depth,
        image=image,
        reference_centers=reference_centers,
        reference_depth=reference_depth,
        reference_image=reference_image,
        scene=scene,
        height=int(args.height),
        width=int(args.width),
        projection_ms=projection_ms,
        render_ms=render_ms,
        extra={"description": "Single first-order Taylor approximation around the center frame."},
    )

    (centers, depth), projection_ms = timed_projection(
        lambda: segmented_affine_project(
            reference_centers,
            reference_depth,
            times,
            segment_frames=int(args.segment_frames),
        )
    )
    image, render_ms = timed_render(
        lambda: dense_renderer.render(centers, depth, scene, height=int(args.height), width=int(args.width))
    )
    methods["segmented_affine"] = method_row(
        name="segmented_affine",
        centers=centers,
        depth=depth,
        image=image,
        reference_centers=reference_centers,
        reference_depth=reference_depth,
        reference_image=reference_image,
        scene=scene,
        height=int(args.height),
        width=int(args.width),
        projection_ms=projection_ms,
        render_ms=render_ms,
        extra={
            "segment_frames": int(args.segment_frames),
            "description": "Piecewise affine center/depth interpolation between exact segment anchors.",
        },
    )

    def _projective_curve_projection() -> tuple[Tensor, Tensor, ProjectedCurves]:
        p_coeff, fit_error = fit_projection_polynomial(p_seq, times, degree=int(args.curve_degree))
        projected = compile_projective_curves(scene, p_coeff, camera_fit_error=fit_error)
        curve_centers, curve_depth = evaluate_projective_curves(projected, times)
        return curve_centers, curve_depth, projected

    start = time.perf_counter()
    centers, depth, projected = _projective_curve_projection()
    projection_ms = (time.perf_counter() - start) * 1000.0
    image, render_ms = timed_render(
        lambda: dense_renderer.render(centers, depth, scene, height=int(args.height), width=int(args.width))
    )
    methods["projective_rational_curve_dense"] = method_row(
        name="projective_rational_curve_dense",
        centers=centers,
        depth=depth,
        image=image,
        reference_centers=reference_centers,
        reference_depth=reference_depth,
        reference_image=reference_image,
        scene=scene,
        height=int(args.height),
        width=int(args.width),
        projection_ms=projection_ms,
        render_ms=render_ms,
        extra={
            "camera_fit_error": projected.camera_fit_error,
            "polynomial_degree": projected.polynomial_degree,
            "description": "Compiled homogeneous polynomial curve evaluated as projective rational centers, rendered by the selected CPU dense path.",
        },
    )

    finite_methods = [
        bool(math.isfinite(method["rgb"]["mse"]))
        and bool(math.isfinite(method["center_residual_px"]["max_px"]))
        and method["projected_primitive_count"] > 0
        for method in methods.values()
    ]
    result = {
        "benchmark": "star_prt_curve_comparison",
        "device": "cpu",
        "config": {
            "height": int(args.height),
            "width": int(args.width),
            "frames": int(args.frames),
            "primitives": int(args.primitives),
            "seed": int(args.seed),
            "curve_degree": int(args.curve_degree),
            "segment_frames": int(args.segment_frames),
            "torch_threads": int(args.torch_threads),
        },
        "worker1_dense_module": dense_renderer.info,
        "reference": {
            "name": "per_frame_direct_projection_reference",
            "center_residual_px": {"mean_px": 0.0, "rms_px": 0.0, "p95_px": 0.0, "max_px": 0.0},
            "rgb": {"mse": 0.0, "psnr": 120.0, "l1": 0.0},
            "projected_primitive_count": projected_count(
                reference_centers,
                reference_depth,
                scene,
                height=int(args.height),
                width=int(args.width),
            ),
            "elapsed_ms": {
                "projection": reference_projection_ms,
                "render": reference_render_ms,
                "total": reference_projection_ms + reference_render_ms,
            },
        },
        "methods": methods,
        "pass": bool(all(finite_methods)),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU STAR-PRT curve-vs-affine comparison benchmark.")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--primitives", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--curve-degree", type=int, default=3)
    parser.add_argument("--segment-frames", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
