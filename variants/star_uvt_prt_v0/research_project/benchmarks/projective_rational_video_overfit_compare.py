from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import cv2
import torch
from PIL import Image
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.trainer_harness.projective_rational_metal_autograd import (  # noqa: E402
    render_projective_rational_tubes_metal_direct_serial_backward,
)
from torch_gsplat_bridge_star_uvt_prt import (  # noqa: E402
    UVTRenderConfig,
    apply_projective_rational_tile_env,
    recommend_projective_rational_tile_config,
    render_projective_rational_tubes_tiled,
)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _logit(value: Tensor) -> Tensor:
    return torch.logit(value.clamp(1.0e-4, 1.0 - 1.0e-4))


def _softplus_inverse(value: Tensor) -> Tensor:
    return torch.log(torch.expm1(value.clamp_min(1.0e-6)))


def _centered_times(frames: int, *, device: torch.device) -> Tensor:
    values = torch.arange(frames, dtype=torch.float32, device=device)
    return values - 0.5 * float(frames - 1)


def load_video_target(video_path: Path, *, target_size: int, max_frames: int, device: torch.device) -> Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    frames: list[Tensor] = []
    while len(frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        scale = float(target_size) / float(min(height, width))
        resized = cv2.resize(
            frame_rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        y0 = max(0, (resized.shape[0] - target_size) // 2)
        x0 = max(0, (resized.shape[1] - target_size) // 2)
        crop = resized[y0 : y0 + target_size, x0 : x0 + target_size]
        if crop.shape[0] != target_size or crop.shape[1] != target_size:
            crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        frames.append(torch.from_numpy(crop).to(torch.float32).div(255.0))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    return torch.stack(frames, dim=0).to(device).contiguous()


class ScreenPRTModel(torch.nn.Module):
    def __init__(
        self,
        target: Tensor,
        *,
        tube_count: int,
        h_terms: int,
        spatial_precision: float,
        temporal_precision: float,
        opacity: float,
        seed: int,
    ) -> None:
        super().__init__()
        if h_terms <= 0 or h_terms > 8:
            raise ValueError("h_terms must be in [1, 8]")
        frames, height, width, _ = target.shape
        generator = torch.Generator(device="cpu").manual_seed(seed)
        flat_count = frames * height * width
        sample_ids = torch.randint(flat_count, (tube_count,), generator=generator)
        frame_ids = sample_ids // (height * width)
        rem = sample_ids % (height * width)
        y_ids = rem // width
        x_ids = rem % width
        colors = target.detach().cpu()[frame_ids, y_ids, x_ids].to(target.device)
        times = _centered_times(frames, device=target.device)

        h_coeff = torch.zeros((tube_count, h_terms, 3), dtype=torch.float32, device=target.device)
        h_coeff[:, 0, 0] = x_ids.to(target.device, dtype=torch.float32) + 0.5
        h_coeff[:, 0, 1] = y_ids.to(target.device, dtype=torch.float32) + 0.5
        h_coeff[:, 0, 2] = 1.0
        if h_terms > 1:
            h_coeff[:, 1, :2] = 0.01 * torch.randn((tube_count, 2), generator=generator).to(target.device)
        self.h_coeff = torch.nn.Parameter(h_coeff)
        self.center_t = torch.nn.Parameter(times[frame_ids.to(target.device)].clone())
        self.raw_precision_uv = torch.nn.Parameter(
            _softplus_inverse(torch.full((tube_count, 2), float(spatial_precision), dtype=torch.float32, device=target.device))
        )
        self.raw_lambda_t = torch.nn.Parameter(
            _softplus_inverse(torch.full((tube_count,), float(temporal_precision), dtype=torch.float32, device=target.device))
        )
        self.raw_opacity = torch.nn.Parameter(
            _logit(torch.full((tube_count,), float(opacity), dtype=torch.float32, device=target.device))
        )
        self.raw_color = torch.nn.Parameter(_logit(colors))

    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        precision_uv = torch.nn.functional.softplus(self.raw_precision_uv).clamp_min(1.0e-5)
        lambda_uv = torch.stack(
            (
                precision_uv[:, 0],
                torch.zeros_like(precision_uv[:, 0]),
                precision_uv[:, 1],
            ),
            dim=-1,
        )
        lambda_t = torch.nn.functional.softplus(self.raw_lambda_t).clamp_min(1.0e-5)
        opacity = 0.99 * torch.sigmoid(self.raw_opacity)
        color = torch.sigmoid(self.raw_color)
        return self.h_coeff, lambda_uv, lambda_t, self.center_t, opacity, color

    def render(self, config: UVTRenderConfig) -> Tensor:
        return render_projective_rational_tubes_metal_direct_serial_backward(*self.tensors(), config)


class PerFrameScreenGaussianModel(torch.nn.Module):
    def __init__(
        self,
        target: Tensor,
        *,
        splats_per_frame: int,
        spatial_precision: float,
        opacity: float,
        seed: int,
    ) -> None:
        super().__init__()
        frames, height, width, _ = target.shape
        generator = torch.Generator(device="cpu").manual_seed(seed)
        centers = torch.empty((frames, splats_per_frame, 2), dtype=torch.float32, device=target.device)
        colors = torch.empty((frames, splats_per_frame, 3), dtype=torch.float32, device=target.device)
        for frame in range(frames):
            sample_ids = torch.randint(height * width, (splats_per_frame,), generator=generator)
            y_ids = sample_ids // width
            x_ids = sample_ids % width
            centers[frame, :, 0] = x_ids.to(target.device, dtype=torch.float32) + 0.5
            centers[frame, :, 1] = y_ids.to(target.device, dtype=torch.float32) + 0.5
            colors[frame] = target.detach().cpu()[frame, y_ids, x_ids].to(target.device)
        self.center_uv = torch.nn.Parameter(centers)
        self.raw_precision_uv = torch.nn.Parameter(
            _softplus_inverse(
                torch.full((frames, splats_per_frame, 2), float(spatial_precision), dtype=torch.float32, device=target.device)
            )
        )
        self.raw_opacity = torch.nn.Parameter(
            _logit(torch.full((frames, splats_per_frame), float(opacity), dtype=torch.float32, device=target.device))
        )
        self.raw_color = torch.nn.Parameter(_logit(colors))

    def render_loop(self, *, height: int, width: int, max_alpha: float = 0.99) -> Tensor:
        frames, splats, _ = self.center_uv.shape
        y = torch.arange(height, dtype=torch.float32, device=self.center_uv.device) + 0.5
        x = torch.arange(width, dtype=torch.float32, device=self.center_uv.device) + 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        precision = torch.nn.functional.softplus(self.raw_precision_uv).clamp_min(1.0e-5)
        opacity = (0.99 * torch.sigmoid(self.raw_opacity)).clamp(max=max_alpha)
        color = torch.sigmoid(self.raw_color)
        rows = []
        for frame in range(frames):
            accum = torch.zeros((height, width, 3), dtype=torch.float32, device=self.center_uv.device)
            trans = torch.ones((height, width, 1), dtype=torch.float32, device=self.center_uv.device)
            for splat in range(splats):
                du = xx - self.center_uv[frame, splat, 0]
                dv = yy - self.center_uv[frame, splat, 1]
                exponent = 0.5 * (
                    precision[frame, splat, 0] * du.square() + precision[frame, splat, 1] * dv.square()
                )
                alpha = (opacity[frame, splat] * torch.exp(-exponent)).clamp(max=max_alpha)
                alpha3 = alpha.unsqueeze(-1)
                accum = accum + trans * alpha3 * color[frame, splat].view(1, 1, 3)
                trans = trans * (1.0 - alpha3)
            rows.append(accum)
        return torch.stack(rows, dim=0).contiguous()

    def render_dense_vectorized(self, *, height: int, width: int, max_alpha: float = 0.99) -> Tensor:
        frames = int(self.center_uv.shape[0])
        y = torch.arange(height, dtype=torch.float32, device=self.center_uv.device) + 0.5
        x = torch.arange(width, dtype=torch.float32, device=self.center_uv.device) + 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        precision = torch.nn.functional.softplus(self.raw_precision_uv).clamp_min(1.0e-5)
        opacity = (0.99 * torch.sigmoid(self.raw_opacity)).clamp(max=max_alpha)
        color = torch.sigmoid(self.raw_color)

        du = xx.view(1, 1, height, width) - self.center_uv[:, :, 0].view(frames, -1, 1, 1)
        dv = yy.view(1, 1, height, width) - self.center_uv[:, :, 1].view(frames, -1, 1, 1)
        exponent = 0.5 * (
            precision[:, :, 0].view(frames, -1, 1, 1) * du.square()
            + precision[:, :, 1].view(frames, -1, 1, 1) * dv.square()
        )
        alpha = (opacity.view(frames, -1, 1, 1) * torch.exp(-exponent)).clamp(max=max_alpha)
        trans_inclusive = torch.cumprod(1.0 - alpha, dim=1)
        trans = torch.cat((torch.ones_like(trans_inclusive[:, :1]), trans_inclusive[:, :-1]), dim=1)
        weights = (trans * alpha).unsqueeze(-1)
        return (weights * color.view(frames, -1, 1, 1, 3)).sum(dim=1).contiguous()

    def render(self, *, height: int, width: int, mode: str, max_alpha: float = 0.99) -> Tensor:
        if mode == "loop":
            return self.render_loop(height=height, width=width, max_alpha=max_alpha)
        if mode == "dense_vectorized":
            return self.render_dense_vectorized(height=height, width=width, max_alpha=max_alpha)
        raise ValueError("baseline render mode must be 'loop' or 'dense_vectorized'")


def mse_to_psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1.0e-12))


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def fit_model(
    model: torch.nn.Module,
    target: Tensor,
    render_fn,
    *,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[list[float], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    started = time.perf_counter()
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        image = render_fn()
        loss = (image - target).square().mean()
        losses.append(float(loss.detach().cpu()))
        if step == steps:
            break
        loss.backward()
        optimizer.step()
        _sync(device)
    _sync(device)
    return losses, (time.perf_counter() - started) * 1000.0


def timed_render(render_fn, *, device: torch.device, repeats: int, warmups: int) -> tuple[Tensor, list[float]]:
    samples = []
    image = None
    with torch.no_grad():
        for _ in range(warmups):
            image = render_fn()
            _sync(device)
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter()
            image = render_fn()
            _sync(device)
            samples.append((time.perf_counter() - started) * 1000.0)
    if image is None:
        raise AssertionError("render repeat count must be positive")
    return image, samples


def summarize_ms(samples: list[float]) -> dict[str, Any]:
    return {
        "samples": samples,
        "min": min(samples),
        "median": statistics.median(samples),
        "max": max(samples),
    }


def metrics(image: Tensor, target: Tensor) -> dict[str, float]:
    mse = float((image - target).square().mean().detach().cpu())
    l1 = float((image - target).abs().mean().detach().cpu())
    return {"mse": mse, "l1": l1, "psnr": mse_to_psnr(mse)}


def _as_uint8(frame: Tensor) -> Image.Image:
    array = frame.detach().cpu().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def write_contact_sheet(path: Path, target: Tensor, prt: Tensor, baseline: Tensor | None) -> None:
    frame_count = min(int(target.shape[0]), 4)
    rows = [target[:frame_count], prt[:frame_count]]
    if baseline is not None:
        rows.append(baseline[:frame_count])
    height = int(target.shape[1])
    width = int(target.shape[2])
    gutter = 2
    sheet = Image.new(
        "RGB",
        (frame_count * width + (frame_count - 1) * gutter, len(rows) * height + (len(rows) - 1) * gutter),
        (0, 0, 0),
    )
    for row_idx, row in enumerate(rows):
        for frame_idx in range(frame_count):
            sheet.paste(_as_uint8(row[frame_idx]), (frame_idx * (width + gutter), row_idx * (height + gutter)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "mps":
        raise ValueError("projective rational video overfit benchmark currently requires --device=mps")
    torch.manual_seed(args.seed)
    target = load_video_target(args.video_path, target_size=args.target_size, max_frames=args.max_frames, device=device)
    frames, height, width, _ = target.shape
    tile_config = recommend_projective_rational_tile_config(tube_count=args.tube_count)
    apply_projective_rational_tile_env(tile_config)
    config = UVTRenderConfig(height=height, width=width, frames=frames, **tile_config.as_render_kwargs())

    prt_model = ScreenPRTModel(
        target,
        tube_count=args.tube_count,
        h_terms=args.h_terms,
        spatial_precision=args.spatial_precision,
        temporal_precision=args.temporal_precision,
        opacity=args.opacity,
        seed=args.seed,
    ).to(device)
    prt_losses, prt_train_ms = fit_model(
        prt_model,
        target,
        lambda: prt_model.render(config),
        steps=args.steps,
        lr=args.lr,
        device=device,
    )
    prt_image, prt_render_samples = timed_render(
        lambda: render_projective_rational_tubes_tiled(*prt_model.tensors(), config),
        device=device,
        repeats=args.render_repeats,
        warmups=args.render_warmups,
    )
    aux = render_projective_rational_tubes_tiled(*prt_model.tensors(), config, return_aux=True)

    baseline_model = None
    baseline_losses = None
    baseline_train_ms = None
    baseline_image = None
    baseline_render_samples = None
    if not args.skip_baseline:
        baseline_model = PerFrameScreenGaussianModel(
            target,
            splats_per_frame=args.per_frame_splats,
            spatial_precision=args.spatial_precision,
            opacity=args.opacity,
            seed=args.seed,
        ).to(device)
        baseline_eval_render_mode = args.baseline_eval_render_mode or args.baseline_render_mode
        baseline_train_render = lambda: baseline_model.render(
            height=height,
            width=width,
            mode=args.baseline_render_mode,
        )
        baseline_eval_render = lambda: baseline_model.render(
            height=height,
            width=width,
            mode=baseline_eval_render_mode,
        )
        baseline_losses, baseline_train_ms = fit_model(
            baseline_model,
            target,
            baseline_train_render,
            steps=args.steps,
            lr=args.baseline_lr,
            device=device,
        )
        baseline_image, baseline_render_samples = timed_render(
            baseline_eval_render,
            device=device,
            repeats=args.render_repeats,
            warmups=args.render_warmups,
        )

    if args.contact_sheet is not None:
        write_contact_sheet(args.contact_sheet, target, prt_image, baseline_image)

    return {
        "name": "projective_rational_video_overfit_compare",
        "note": (
            "Single-video overfit diagnostic. PRT is screen-time projective-rational tubes using the Metal "
            "tile_pixel_atomic training backward. The baseline is a simple per-frame screen Gaussian model, "
            "not full 3DGS."
        ),
        "video_path": str(args.video_path),
        "frames": frames,
        "height": height,
        "width": width,
        "steps": args.steps,
        "device": str(device),
        "seed": args.seed,
        "render_warmups": args.render_warmups,
        "tile_config_key": tile_config.key,
        "tile_config": tile_config.as_dict(),
        "prt": {
            "tube_count": args.tube_count,
            "h_terms": args.h_terms,
            "parameter_count": parameter_count(prt_model),
            "lr": args.lr,
            "initial_loss": prt_losses[0],
            "final_loss": prt_losses[-1],
            "loss_ratio": prt_losses[-1] / max(prt_losses[0], 1.0e-12),
            "losses": prt_losses,
            "train_wall_ms": prt_train_ms,
            "render_benchmark_ms": summarize_ms(prt_render_samples),
            "metrics": metrics(prt_image, target),
            "max_tile_count": int(aux.tile_counts.max().detach().cpu()),
            "overflow_tile_count": int((aux.tile_overflow > 0).sum().detach().cpu()),
            "active_tile_count": int((aux.tile_counts > 0).sum().detach().cpu()),
        },
        "per_frame_screen_gaussian": None
        if baseline_model is None or baseline_losses is None or baseline_train_ms is None or baseline_image is None
        else {
            "splats_per_frame": args.per_frame_splats,
            "total_splats": args.per_frame_splats * frames,
            "parameter_count": parameter_count(baseline_model),
            "train_render_mode": args.baseline_render_mode,
            "eval_render_mode": args.baseline_eval_render_mode or args.baseline_render_mode,
            "lr": args.baseline_lr,
            "initial_loss": baseline_losses[0],
            "final_loss": baseline_losses[-1],
            "loss_ratio": baseline_losses[-1] / max(baseline_losses[0], 1.0e-12),
            "losses": baseline_losses,
            "train_wall_ms": baseline_train_ms,
            "render_benchmark_ms": summarize_ms([] if baseline_render_samples is None else baseline_render_samples),
            "metrics": metrics(baseline_image, target),
        },
        "pass": prt_losses[-1] < prt_losses[0] and int((aux.tile_overflow > 0).sum().detach().cpu()) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tube-count", type=int, default=128)
    parser.add_argument("--h-terms", type=int, default=3)
    parser.add_argument("--spatial-precision", type=float, default=0.08)
    parser.add_argument("--temporal-precision", type=float, default=0.3)
    parser.add_argument("--opacity", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--per-frame-splats", type=int, default=32)
    parser.add_argument("--baseline-lr", type=float, default=0.03)
    parser.add_argument("--baseline-render-mode", choices=("loop", "dense_vectorized"), default="loop")
    parser.add_argument("--baseline-eval-render-mode", choices=("loop", "dense_vectorized"))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--render-warmups", type=int, default=1)
    parser.add_argument("--render-repeats", type=int, default=3)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
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
