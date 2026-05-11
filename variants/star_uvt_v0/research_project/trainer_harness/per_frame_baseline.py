from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torch_gsplat_bridge_star_uvt import UVTRenderConfig

try:
    from .model import _inv_softplus, _logit
except ImportError:  # pragma: no cover - script execution fallback.
    from model import _inv_softplus, _logit


class PerFrameGaussianModel(nn.Module):
    """Independent per-frame 2D Gaussian baseline for STAR-UVT comparisons."""

    def __init__(
        self,
        frames: int,
        splats_per_frame: int,
        config: UVTRenderConfig,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        min_precision: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if frames != config.frames:
            raise ValueError("frames must match config.frames")
        if splats_per_frame <= 0:
            raise ValueError("splats_per_frame must be positive")
        self.config = config
        self.frames = int(frames)
        self.splats_per_frame = int(splats_per_frame)
        self.min_precision = float(min_precision)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        center_u = torch.rand((frames, splats_per_frame), generator=generator) * float(config.width)
        center_v = torch.rand((frames, splats_per_frame), generator=generator) * float(config.height)
        center_uv = torch.stack((center_u, center_v), dim=-1)
        precision = torch.full((frames, splats_per_frame, 2), 0.25, dtype=torch.float32)
        opacity = torch.full((frames, splats_per_frame), 0.35, dtype=torch.float32)
        color = torch.rand((frames, splats_per_frame, 3), generator=generator).mul(0.6).add(0.2)
        depth = torch.linspace(0.8, 1.2, splats_per_frame, dtype=torch.float32).view(1, splats_per_frame)
        depth = depth.expand(frames, splats_per_frame).contiguous()

        dev = torch.device(device)
        self.center_uv = nn.Parameter(center_uv.to(dev))
        self.raw_precision = nn.Parameter(_inv_softplus(precision - self.min_precision).to(dev))
        self.raw_opacity = nn.Parameter(_logit(opacity / 0.99).to(dev))
        self.raw_color = nn.Parameter(_logit(color).to(dev))
        self.depth = nn.Parameter(depth.to(dev))

    @classmethod
    def from_uvt_tensors(
        cls,
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        opacity: Tensor,
        color: Tensor,
        config: UVTRenderConfig,
        *,
        seed: int = 0,
        jitter_pixels: float = 0.0,
        min_precision: float = 1.0e-4,
    ) -> "PerFrameGaussianModel":
        model = cls(
            config.frames,
            int(ma.shape[0]),
            config,
            seed=seed,
            device=ma.device,
            min_precision=min_precision,
        )
        frames = torch.arange(config.frames, dtype=torch.float32, device=ma.device) - 0.5 * float(config.frames - 1)
        q2 = torch.stack(
            (
                torch.stack((q_uvt[:, 0], q_uvt[:, 1]), dim=-1),
                torch.stack((q_uvt[:, 1], q_uvt[:, 3]), dim=-1),
            ),
            dim=-2,
        )
        inv_q2 = torch.linalg.inv(q2 + torch.eye(2, dtype=torch.float32, device=ma.device).unsqueeze(0) * 1.0e-6)
        cross = q_uvt[:, [2, 4]]
        center_velocity = -(inv_q2 @ cross.unsqueeze(-1)).squeeze(-1)
        dt = frames.view(-1, 1) - ma[:, 2].view(1, -1)
        shifts = dt.unsqueeze(-1) * center_velocity.unsqueeze(0)
        center_uv = ma[:, :2].unsqueeze(0) + shifts
        precision = torch.stack((q_uvt[:, 0], q_uvt[:, 3]), dim=-1).clamp_min(min_precision * 2.0)
        precision = precision.unsqueeze(0).expand(config.frames, -1, -1).contiguous()
        generator = torch.Generator(device="cpu").manual_seed(seed + 31)

        with torch.no_grad():
            model.center_uv.copy_(center_uv)
            model.raw_precision.copy_(_inv_softplus(precision - min_precision))
            model.depth.copy_(depth0.view(1, -1).expand(config.frames, -1))
            model.raw_opacity.copy_(_logit(opacity.clamp(0.0, 0.98).view(1, -1).expand(config.frames, -1) / 0.99))
            model.raw_color.copy_(_logit(color.clamp(1.0e-5, 1.0 - 1.0e-5).view(1, -1, 3).expand(config.frames, -1, -1)))
            if jitter_pixels > 0.0:
                noise = torch.randn(model.center_uv.shape, generator=generator, device="cpu").to(ma.device)
                model.center_uv.add_(noise * float(jitter_pixels))
        return model

    @classmethod
    def from_video_samples(
        cls,
        target: Tensor,
        config: UVTRenderConfig,
        *,
        splats_per_frame: int,
        seed: int = 0,
        spatial_precision: float = 0.25,
        opacity: float = 0.35,
        sample_mode: str = "random",
        min_precision: float = 1.0e-4,
    ) -> "PerFrameGaussianModel":
        if target.shape != (config.frames, config.height, config.width, 3):
            raise ValueError(
                "target must have shape "
                f"({config.frames}, {config.height}, {config.width}, 3), got {tuple(target.shape)}"
            )
        model = cls(
            config.frames,
            splats_per_frame,
            config,
            seed=seed,
            device=target.device,
            min_precision=min_precision,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed + 101)
        if sample_mode == "random":
            y = torch.randint(
                0,
                int(config.height),
                (config.frames, splats_per_frame),
                generator=generator,
                dtype=torch.long,
            )
            x = torch.randint(
                0,
                int(config.width),
                (config.frames, splats_per_frame),
                generator=generator,
                dtype=torch.long,
            )
        elif sample_mode == "stratified":
            cols = max(1, int(round((splats_per_frame * float(config.width) / float(config.height)) ** 0.5)))
            while cols * max(1, (splats_per_frame + cols - 1) // cols) < splats_per_frame:
                cols += 1
            rows = max(1, (splats_per_frame + cols - 1) // cols)
            local = torch.arange(splats_per_frame, dtype=torch.float32)
            y = torch.empty((config.frames, splats_per_frame), dtype=torch.long)
            x = torch.empty((config.frames, splats_per_frame), dtype=torch.long)
            for frame in range(int(config.frames)):
                jitter = torch.rand((splats_per_frame, 2), generator=generator) - 0.5
                cx = ((local.remainder(cols) + 0.5 + jitter[:, 0] * 0.35) / float(cols)) * float(config.width)
                cy = (((local // cols) + 0.5 + jitter[:, 1] * 0.35) / float(rows)) * float(config.height)
                x[frame] = cx.floor().clamp(0, int(config.width) - 1).to(torch.long)
                y[frame] = cy.floor().clamp(0, int(config.height) - 1).to(torch.long)
        else:
            raise ValueError("sample_mode must be one of: random, stratified")

        target_cpu = target.detach().cpu()
        frame_ids = torch.arange(config.frames, dtype=torch.long).view(config.frames, 1)
        color = target_cpu[frame_ids, y, x].clamp(1.0e-5, 1.0 - 1.0e-5).to(target.device)
        center_uv = torch.stack((x.float() + 0.5, y.float() + 0.5), dim=-1).to(target.device)
        precision = torch.full(
            (config.frames, splats_per_frame, 2),
            float(spatial_precision),
            dtype=torch.float32,
            device=target.device,
        ).clamp_min(min_precision * 2.0)
        opacity_tensor = torch.full(
            (config.frames, splats_per_frame),
            float(opacity),
            dtype=torch.float32,
            device=target.device,
        )
        depth = torch.linspace(0.8, 1.2, splats_per_frame, dtype=torch.float32, device=target.device)
        depth = depth.view(1, splats_per_frame).expand(config.frames, splats_per_frame)

        with torch.no_grad():
            model.center_uv.copy_(center_uv)
            model.raw_precision.copy_(_inv_softplus(precision - min_precision))
            model.raw_opacity.copy_(_logit(opacity_tensor.clamp(0.0, 0.98) / 0.99))
            model.raw_color.copy_(_logit(color))
            model.depth.copy_(depth)
        return model

    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        precision = F.softplus(self.raw_precision) + self.min_precision
        opacity = torch.sigmoid(self.raw_opacity) * 0.99
        color = torch.sigmoid(self.raw_color)
        return self.center_uv, precision, self.depth, opacity, color


def render_per_frame_gaussians(model: PerFrameGaussianModel) -> Tensor:
    center_uv, precision, depth, opacity, color = model.tensors()
    config = model.config
    device = center_uv.device
    y = torch.arange(config.height, dtype=torch.float32, device=device) + 0.5
    x = torch.arange(config.width, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).view(1, config.height, config.width, 1, 2)
    delta = grid - center_uv.view(config.frames, 1, 1, model.splats_per_frame, 2)
    qv = precision[:, None, None, :, 0] * delta[..., 0].square() + precision[:, None, None, :, 1] * delta[..., 1].square()
    alpha = torch.clamp(opacity[:, None, None, :] * torch.exp(-0.5 * qv), max=config.max_alpha)
    background = torch.tensor(config.background, dtype=torch.float32, device=device).view(1, 1, 1, 3)
    frames = []
    for frame in range(config.frames):
        accum = torch.zeros((config.height, config.width, 3), dtype=torch.float32, device=device)
        transmittance = torch.ones((config.height, config.width, 1), dtype=torch.float32, device=device)
        order = torch.argsort(depth[frame].detach(), stable=True).detach().cpu().tolist()
        for splat_id in order:
            alpha_i = alpha[frame, :, :, splat_id].unsqueeze(-1)
            accum = accum + transmittance * alpha_i * color[frame, splat_id].view(1, 1, 3)
            transmittance = transmittance * (1.0 - alpha_i)
        frames.append(accum + transmittance * background.squeeze(0))
    return torch.stack(frames, dim=0)


def _ensure_fast_mac_v6_on_path() -> None:
    for parent in Path(__file__).resolve().parents:
        variant_dir = parent / "variants" / "v6_refined"
        if (variant_dir / "torch_gsplat_bridge_v6").exists():
            variant = str(variant_dir)
            if variant not in sys.path:
                sys.path.insert(0, variant)
            return
    raise FileNotFoundError("Could not find fast-mac v6_refined variant for per-frame fast baseline")


def render_per_frame_gaussians_fast_mac(model: PerFrameGaussianModel, *, max_fast_pairs: int = 2048) -> Tensor:
    _ensure_fast_mac_v6_on_path()
    from torch_gsplat_bridge_v6 import RasterConfig, rasterize_projected_gaussians

    center_uv, precision, depth, opacity, color = model.tensors()
    if center_uv.device.type != "mps":
        raise ValueError("per-frame fast_mac renderer requires MPS tensors")
    config = model.config
    zeros = torch.zeros_like(precision[..., 0])
    conics = torch.stack((precision[..., 0], zeros, precision[..., 1]), dim=-1)
    return rasterize_projected_gaussians(
        center_uv.float(),
        conics.float(),
        color.float(),
        opacity.float(),
        depth.float(),
        RasterConfig(
            height=int(config.height),
            width=int(config.width),
            tile_size=16,
            max_fast_pairs=int(max_fast_pairs),
            alpha_threshold=float(config.alpha_threshold),
            transmittance_threshold=float(config.transmittance_threshold),
            background=tuple(float(value) for value in config.background),
            batch_strategy="flatten",
        ),
    ).clamp(0.0, 1.0)
