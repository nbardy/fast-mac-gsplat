from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from PIL import Image
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_project.benchmarks.depth_banded_homography_flow_atlas_residual_probe import (  # noqa: E402
    _forward_homography_centers,
    _homography_matrices,
    _inverse_homography_atlas_targets,
)
from research_project.benchmarks.depth_banded_homography_flow_atlas_tiled_render_probe import (  # noqa: E402
    _build_atlas_tile_sets,
)
from research_project.benchmarks.depth_banded_homography_flow_residual_probe import (  # noqa: E402
    _camera,
    _compile_prt_centers,
    _depth_bands,
    _fit_poly,
    _image_metrics,
    _render_from_centers,
    _world_tubes,
)
from research_project.trainer_harness.projective_rational import (  # noqa: E402
    WorldTubeBatch,
    centered_frame_times,
    direct_project_world_tubes,
)


def _psnr(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1.0e-12))


def _logit(value: Tensor) -> Tensor:
    return torch.logit(value.clamp(1.0e-4, 1.0 - 1.0e-4))


def _softplus_inverse(value: Tensor) -> Tensor:
    return torch.log(torch.expm1(value.clamp_min(1.0e-6)))


class DirectAtlasResidualModel(torch.nn.Module):
    def __init__(
        self,
        *,
        atlas_ref_uv: Tensor,
        residual_coeff: Tensor,
        homographies: Tensor,
        assignments: Tensor,
        depth: Tensor,
        times: Tensor,
        lambda_uv: Tensor,
        lambda_t: Tensor,
        opacity: Tensor,
        color: Tensor,
        init_seed: int,
        color_noise: float,
        geometry_noise_px: float,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(init_seed)
        ref_noise = geometry_noise_px * torch.randn(atlas_ref_uv.shape, generator=generator)
        coeff_noise = geometry_noise_px * 0.05 * torch.randn(residual_coeff.shape, generator=generator)
        color_init = (torch.rand(color.shape, generator=generator) * float(color_noise) + color * (1.0 - float(color_noise))).clamp(
            1.0e-4,
            1.0 - 1.0e-4,
        )
        self.atlas_ref_uv = torch.nn.Parameter(atlas_ref_uv + ref_noise)
        self.residual_coeff = torch.nn.Parameter(residual_coeff + coeff_noise)
        self.raw_precision_uv = torch.nn.Parameter(_softplus_inverse(lambda_uv[:, [0, 2]].clamp_min(1.0e-5)))
        self.raw_lambda_t = torch.nn.Parameter(_softplus_inverse(lambda_t.clamp_min(1.0e-5)))
        self.center_t = torch.nn.Parameter(torch.zeros_like(lambda_t))
        self.raw_opacity = torch.nn.Parameter(_logit(opacity.clamp(1.0e-4, 0.99)))
        self.raw_color = torch.nn.Parameter(_logit(color_init))
        self.register_buffer("homographies", homographies)
        self.register_buffer("assignments", assignments)
        self.register_buffer("depth", depth)
        self.register_buffer("times", times)

    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        degree = int(self.residual_coeff.shape[0]) - 1
        vand = torch.stack([self.times.pow(k) for k in range(degree + 1)], dim=-1)
        residual = torch.einsum("fd,dnc->fnc", vand, self.residual_coeff)
        atlas_centers = self.atlas_ref_uv.view(1, -1, 2) + residual
        warped_centers = _forward_homography_centers(atlas_centers, self.homographies, self.assignments)
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
        opacity = (0.99 * torch.sigmoid(self.raw_opacity)).clamp(max=0.99)
        color = torch.sigmoid(self.raw_color)
        return atlas_centers, warped_centers, lambda_uv, lambda_t, self.center_t, opacity, color

    def render(self, *, height: int, width: int, alpha_threshold: float) -> tuple[Tensor, WorldTubeBatch, Tensor, Tensor]:
        atlas_centers, warped_centers, lambda_uv, lambda_t, center_t, opacity, color = self.tensors()
        tube_count = int(color.shape[0])
        batch = WorldTubeBatch(
            x0=torch.zeros((tube_count, 3), dtype=torch.float32),
            velocity=torch.zeros((tube_count, 3), dtype=torch.float32),
            t0=center_t,
            precision_xy=lambda_uv[:, [0, 2]],
            lambda_t=lambda_t,
            opacity=opacity,
            color=color,
        )
        image = _render_from_centers(
            batch,
            warped_centers,
            self.depth,
            lambda_uv,
            self.times,
            height=height,
            width=width,
            alpha_threshold=alpha_threshold,
        )
        return image, batch, lambda_uv, atlas_centers


def _contact_sheet(path: Path, *, target: Tensor, initial: Tensor, final: Tensor) -> None:
    rows: list[Image.Image] = []
    for frame in range(int(target.shape[0])):
        cells = []
        for image in (target[frame], initial[frame], final[frame]):
            array = image.detach().clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()
            cells.append(Image.fromarray(array, mode="RGB"))
        row = Image.new("RGB", (sum(cell.width for cell in cells), cells[0].height))
        x = 0
        for cell in cells:
            row.paste(cell, (x, 0))
            x += cell.width
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
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
    reference_batch = _world_tubes(args.tubes, seed=args.seed, velocity_scale=args.velocity_scale)
    direct_centers, direct_depth = direct_project_world_tubes(reference_batch, k_seq, w2c_seq, times)
    _, _, lambda_uv, _ = _compile_prt_centers(reference_batch, k_seq, w2c_seq, times, degree=args.prt_degree)
    target = _render_from_centers(
        reference_batch,
        direct_centers,
        direct_depth,
        lambda_uv,
        times,
        height=args.target_size,
        width=args.target_size,
        alpha_threshold=args.alpha_threshold,
    ).detach()

    ref_frame = args.frames // 2
    ref_uv = direct_centers[ref_frame].detach()
    band_depths, assignments = _depth_bands(direct_depth[ref_frame], bands=args.depth_bands)
    homographies = _homography_matrices(band_depths, k_seq, w2c_seq, ref_frame=ref_frame)
    atlas_targets = _inverse_homography_atlas_targets(direct_centers, homographies, assignments)
    atlas_residual_targets = atlas_targets - ref_uv.view(1, -1, 2)
    residual_coeff, _ = _fit_poly(atlas_residual_targets, times, degree=args.residual_degree)

    model = DirectAtlasResidualModel(
        atlas_ref_uv=ref_uv.detach(),
        residual_coeff=residual_coeff.detach(),
        homographies=homographies.detach(),
        assignments=assignments.detach(),
        depth=direct_depth.detach(),
        times=times.detach(),
        lambda_uv=lambda_uv.detach(),
        lambda_t=reference_batch.lambda_t.detach(),
        opacity=reference_batch.opacity.detach(),
        color=reference_batch.color.detach(),
        init_seed=args.seed + 1009,
        color_noise=args.color_noise,
        geometry_noise_px=args.geometry_noise_px,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    initial_image, _, _, _ = model.render(
        height=args.target_size,
        width=args.target_size,
        alpha_threshold=args.alpha_threshold,
    )
    initial_loss = torch.nn.functional.mse_loss(initial_image, target)
    losses = [{"step": 0, "loss": float(initial_loss.detach().cpu())}]
    final_image = initial_image
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        final_image, _, _, _ = model.render(
            height=args.target_size,
            width=args.target_size,
            alpha_threshold=args.alpha_threshold,
        )
        loss = torch.nn.functional.mse_loss(final_image, target)
        loss.backward()
        optimizer.step()
        if step == args.steps or step % max(1, args.log_every) == 0:
            losses.append({"step": int(step), "loss": float(loss.detach().cpu())})

    with torch.no_grad():
        final_image, final_batch, final_lambda_uv, final_atlas_centers = model.render(
            height=args.target_size,
            width=args.target_size,
            alpha_threshold=args.alpha_threshold,
        )
        tile_sets, tile_stats = _build_atlas_tile_sets(
            final_atlas_centers.detach(),
            assignments,
            final_batch,
            final_lambda_uv.detach(),
            times,
            width=args.target_size,
            height=args.target_size,
            tile_size=args.tile_size,
            tile_t=args.tile_t,
            alpha_threshold=args.alpha_threshold,
            support_scale=args.support_scale,
        )
        tile_counts = [len(tile) for tile in tile_sets]
        overflow_tile_count = sum(1 for count in tile_counts if count > args.tile_capacity)
        final_loss = torch.nn.functional.mse_loss(final_image, target)
        initial_metrics = _image_metrics(initial_image.detach(), target)
        final_metrics = _image_metrics(final_image.detach(), target)

    if args.contact_sheet is not None:
        _contact_sheet(args.contact_sheet, target=target, initial=initial_image.detach(), final=final_image.detach())

    loss_drop_fraction = 1.0 - float(final_loss.detach().cpu()) / max(float(initial_loss.detach().cpu()), 1.0e-12)
    finite = bool(torch.isfinite(final_image).all().item()) and all(math.isfinite(row["loss"]) for row in losses)
    passed = (
        finite
        and loss_drop_fraction >= args.min_loss_drop_fraction
        and final_metrics["psnr"] >= args.min_psnr
        and overflow_tile_count == 0
    )
    return {
        "name": "depth_banded_homography_flow_atlas_cpu_overfit_gate",
        "note": "CPU/direct differentiable atlas-residual overfit gate; not a Metal backward or real heldout proof.",
        "pass": passed,
        "config": {
            "seed": args.seed,
            "target_size": args.target_size,
            "frames": args.frames,
            "tubes": args.tubes,
            "steps": args.steps,
            "depth_bands": args.depth_bands,
            "residual_degree": args.residual_degree,
            "tile_size": args.tile_size,
            "tile_t": args.tile_t,
            "tile_capacity": args.tile_capacity,
            "support_scale": args.support_scale,
            "lr": args.lr,
            "color_noise": args.color_noise,
            "geometry_noise_px": args.geometry_noise_px,
        },
        "metrics": {
            "initial_loss": float(initial_loss.detach().cpu()),
            "final_loss": float(final_loss.detach().cpu()),
            "loss_drop_fraction": loss_drop_fraction,
            "initial_psnr": _psnr(float(initial_loss.detach().cpu())),
            "final_psnr": _psnr(float(final_loss.detach().cpu())),
            "initial_image_metrics": initial_metrics,
            "final_image_metrics": final_metrics,
        },
        "tile_summary": {
            **tile_stats,
            "max_tile_count": max(tile_counts) if tile_counts else 0,
            "overflow_tile_count": int(overflow_tile_count),
            "tile_capacity": args.tile_capacity,
        },
        "losses": losses,
        "contact_sheet": None if args.contact_sheet is None else str(args.contact_sheet),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--tubes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--depth-bands", type=int, default=4)
    parser.add_argument("--residual-degree", type=int, default=3)
    parser.add_argument("--prt-degree", type=int, default=2)
    parser.add_argument("--velocity-scale", type=float, default=0.01)
    parser.add_argument("--pan-x", type=float, default=0.35)
    parser.add_argument("--zoom", type=float, default=0.015)
    parser.add_argument("--dolly-z", type=float, default=0.08)
    parser.add_argument("--alpha-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--support-scale", type=float, default=1.4)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--tile-t", type=int, default=4)
    parser.add_argument("--tile-capacity", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--color-noise", type=float, default=0.85)
    parser.add_argument("--geometry-noise-px", type=float, default=0.35)
    parser.add_argument("--min-loss-drop-fraction", type=float, default=0.5)
    parser.add_argument("--min-psnr", type=float, default=20.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    args = parser.parse_args()

    report = run_gate(args)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
