from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torch_gsplat_bridge_star_uvt import UVTRenderConfig


def _logit(value: Tensor) -> Tensor:
    clamped = value.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(clamped) - torch.log1p(-clamped)


def _inv_softplus(value: Tensor) -> Tensor:
    clamped = value.clamp_min(1.0e-8)
    return clamped + torch.log(-torch.expm1(-clamped))


def _quadratic(q_uvt: Tensor, delta: Tensor) -> Tensor:
    return (
        q_uvt[..., 0] * delta[..., 0] * delta[..., 0]
        + 2.0 * q_uvt[..., 1] * delta[..., 0] * delta[..., 1]
        + 2.0 * q_uvt[..., 2] * delta[..., 0] * delta[..., 2]
        + q_uvt[..., 3] * delta[..., 1] * delta[..., 1]
        + 2.0 * q_uvt[..., 4] * delta[..., 1] * delta[..., 2]
        + q_uvt[..., 5] * delta[..., 2] * delta[..., 2]
    )


def _block_match_velocity(
    target_cpu: Tensor,
    frame_ids: Tensor,
    x: Tensor,
    y: Tensor,
    *,
    search_radius: int,
    patch_radius: int,
    min_improvement_ratio: float | None = None,
) -> Tensor:
    if int(target_cpu.shape[0]) < 2 or search_radius <= 0:
        return torch.zeros((int(frame_ids.numel()), 2), dtype=torch.float32)
    offsets = [(dx, dy) for dy in range(-search_radius, search_radius + 1) for dx in range(-search_radius, search_radius + 1)]
    patch_offsets = [(dx, dy) for dy in range(-patch_radius, patch_radius + 1) for dx in range(-patch_radius, patch_radius + 1)]
    height = int(target_cpu.shape[1])
    width = int(target_cpu.shape[2])
    velocities = torch.zeros((int(frame_ids.numel()), 2), dtype=torch.float32)
    for idx in range(int(frame_ids.numel())):
        frame = int(frame_ids[idx])
        src_x = int(x[idx])
        src_y = int(y[idx])
        if frame < int(target_cpu.shape[0]) - 1:
            dst_frame = frame + 1
            direction = 1.0
        else:
            dst_frame = frame - 1
            direction = -1.0
        best_error = float("inf")
        zero_error = None
        best_dx = 0
        best_dy = 0
        for dx, dy in offsets:
            error = 0.0
            for patch_dx, patch_dy in patch_offsets:
                sx = min(max(src_x + patch_dx, 0), width - 1)
                sy = min(max(src_y + patch_dy, 0), height - 1)
                tx = min(max(src_x + dx + patch_dx, 0), width - 1)
                ty = min(max(src_y + dy + patch_dy, 0), height - 1)
                delta = target_cpu[frame, sy, sx] - target_cpu[dst_frame, ty, tx]
                error += float(delta.square().mean().item())
            if dx == 0 and dy == 0:
                zero_error = error
            if error < best_error:
                best_error = error
                best_dx = dx
                best_dy = dy
        if min_improvement_ratio is not None:
            if zero_error is None:
                raise AssertionError("zero-motion error was not evaluated")
            if best_error >= zero_error * float(min_improvement_ratio):
                best_dx = 0
                best_dy = 0
        velocities[idx, 0] = direction * float(best_dx)
        velocities[idx, 1] = direction * float(best_dy)
    return velocities


def make_uvt_grid(config: UVTRenderConfig, device: torch.device | str) -> Tensor:
    dev = torch.device(device)
    frames = torch.arange(config.frames, dtype=torch.float32, device=dev)
    t = frames - 0.5 * float(config.frames - 1)
    y = torch.arange(config.height, dtype=torch.float32, device=dev) + 0.5
    x = torch.arange(config.width, dtype=torch.float32, device=dev) + 0.5
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    return torch.stack((xx, yy, tt), dim=-1).contiguous()


class ScreenTimeTubeModel(nn.Module):
    """Projected UVT tube parameters for the research trainer harness."""

    def __init__(
        self,
        tube_count: int,
        config: UVTRenderConfig,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        min_precision: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if tube_count <= 0:
            raise ValueError("tube_count must be positive")
        self.config = config
        self.tube_count = int(tube_count)
        self.min_precision = float(min_precision)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        center_u = torch.rand((tube_count,), generator=generator) * float(config.width)
        center_v = torch.rand((tube_count,), generator=generator) * float(config.height)
        center_uv = torch.stack((center_u, center_v), dim=-1)
        center_t = torch.zeros((tube_count, 1), dtype=torch.float32)
        velocity_uv = torch.randn((tube_count, 2), generator=generator) * 0.25
        precision = torch.full((tube_count, 3), 0.25, dtype=torch.float32)
        opacity = torch.full((tube_count,), 0.35, dtype=torch.float32)
        color = torch.rand((tube_count, 3), generator=generator).mul(0.6).add(0.2)
        depth0 = torch.linspace(0.8, 1.2, tube_count, dtype=torch.float32)

        dev = torch.device(device)
        self.center_uv = nn.Parameter(center_uv.to(dev))
        self.center_t = nn.Parameter(center_t.to(dev))
        self.velocity_uv = nn.Parameter(velocity_uv.to(dev))
        self.raw_precision = nn.Parameter(_inv_softplus(precision - self.min_precision).to(dev))
        self.raw_opacity = nn.Parameter(_logit(opacity / 0.99).to(dev))
        self.raw_color = nn.Parameter(_logit(color).to(dev))
        self.depth0 = nn.Parameter(depth0.to(dev))

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
    ) -> "ScreenTimeTubeModel":
        model = cls(
            int(ma.shape[0]),
            config,
            seed=seed,
            device=ma.device,
            min_precision=min_precision,
        )
        lambda_u = q_uvt[:, 0].clamp_min(min_precision * 2.0)
        lambda_v = q_uvt[:, 3].clamp_min(min_precision * 2.0)
        velocity_u = -q_uvt[:, 2] / lambda_u
        velocity_v = -q_uvt[:, 4] / lambda_v
        lambda_t = (q_uvt[:, 5] - lambda_u * velocity_u.square() - lambda_v * velocity_v.square()).clamp_min(
            min_precision * 2.0
        )
        precision = torch.stack((lambda_u, lambda_v, lambda_t), dim=-1)
        generator = torch.Generator(device="cpu").manual_seed(seed + 17)

        with torch.no_grad():
            model.center_uv.copy_(ma[:, :2])
            model.center_t.copy_(ma[:, 2:3])
            model.velocity_uv.copy_(torch.stack((velocity_u, velocity_v), dim=-1))
            model.raw_precision.copy_(_inv_softplus(precision - min_precision))
            model.depth0.copy_(depth0)
            model.raw_opacity.copy_(_logit(opacity.clamp(0.0, 0.98) / 0.99))
            model.raw_color.copy_(_logit(color))
            if jitter_pixels > 0.0:
                center_noise = torch.randn(model.center_uv.shape, generator=generator, device="cpu").to(ma.device)
                velocity_noise = torch.randn(model.velocity_uv.shape, generator=generator, device="cpu").to(ma.device)
                color_noise = torch.randn(model.raw_color.shape, generator=generator, device="cpu").to(ma.device)
                model.center_uv.add_(center_noise * float(jitter_pixels))
                model.velocity_uv.add_(velocity_noise * float(jitter_pixels) * 0.10)
                model.raw_color.add_(color_noise * 0.05)
        return model

    @classmethod
    def from_video_samples(
        cls,
        target: Tensor,
        config: UVTRenderConfig,
        *,
        tube_count: int,
        seed: int = 0,
        spatial_precision: float = 0.25,
        temporal_precision: float = 0.25,
        opacity: float = 0.35,
        sample_mode: str = "random",
        velocity_init: str = "zero",
        velocity_search_radius: int = 4,
        velocity_patch_radius: int = 1,
        velocity_min_improvement_ratio: float = 0.9,
        min_precision: float = 1.0e-4,
    ) -> "ScreenTimeTubeModel":
        """Initialize projected tubes from actual target pixels and frame times."""

        if target.shape != (config.frames, config.height, config.width, 3):
            raise ValueError(
                "target must have shape "
                f"({config.frames}, {config.height}, {config.width}, 3), got {tuple(target.shape)}"
            )
        model = cls(
            tube_count,
            config,
            seed=seed,
            device=target.device,
            min_precision=min_precision,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if sample_mode == "random":
            frame_ids = torch.arange(tube_count, dtype=torch.long) % int(config.frames)
            frame_ids = frame_ids[torch.randperm(tube_count, generator=generator)]
            y = torch.randint(0, int(config.height), (tube_count,), generator=generator, dtype=torch.long)
            x = torch.randint(0, int(config.width), (tube_count,), generator=generator, dtype=torch.long)
        elif sample_mode == "stratified":
            frame_ids = torch.arange(tube_count, dtype=torch.long) % int(config.frames)
            x = torch.empty((tube_count,), dtype=torch.long)
            y = torch.empty((tube_count,), dtype=torch.long)
            for frame in range(int(config.frames)):
                positions = torch.nonzero(frame_ids == frame, as_tuple=False).flatten()
                count = int(positions.numel())
                if count == 0:
                    continue
                cols = max(1, int(round((count * float(config.width) / float(config.height)) ** 0.5)))
                while cols * max(1, (count + cols - 1) // cols) < count:
                    cols += 1
                rows = max(1, (count + cols - 1) // cols)
                local = torch.arange(count, dtype=torch.float32)
                jitter = torch.rand((count, 2), generator=generator) - 0.5
                cx = ((local.remainder(cols) + 0.5 + jitter[:, 0] * 0.35) / float(cols)) * float(config.width)
                cy = (((local // cols) + 0.5 + jitter[:, 1] * 0.35) / float(rows)) * float(config.height)
                x[positions] = cx.floor().clamp(0, int(config.width) - 1).to(torch.long)
                y[positions] = cy.floor().clamp(0, int(config.height) - 1).to(torch.long)
            order = torch.randperm(tube_count, generator=generator)
            frame_ids = frame_ids[order]
            x = x[order]
            y = y[order]
        elif sample_mode == "temporal_quarters":
            pieces = min(4, int(config.frames))
            base_count = (tube_count + pieces - 1) // pieces
            base_x = torch.randint(0, int(config.width), (base_count,), generator=generator, dtype=torch.long)
            base_y = torch.randint(0, int(config.height), (base_count,), generator=generator, dtype=torch.long)
            tube_ids = torch.arange(tube_count, dtype=torch.long)
            piece_ids = (tube_ids // base_count).clamp_max(pieces - 1)
            base_ids = tube_ids.remainder(base_count)
            starts = torch.div(piece_ids * int(config.frames), pieces, rounding_mode="floor")
            ends = torch.div((piece_ids + 1) * int(config.frames), pieces, rounding_mode="floor").clamp_min(starts + 1)
            span = (ends - starts).clamp_min(1)
            frame_offsets = torch.floor(torch.rand((tube_count,), generator=generator) * span.float()).to(torch.long)
            frame_ids = starts + frame_offsets
            x = base_x[base_ids]
            y = base_y[base_ids]
            order = torch.randperm(tube_count, generator=generator)
            frame_ids = frame_ids[order]
            x = x[order]
            y = y[order]
        else:
            raise ValueError("sample_mode must be one of: random, stratified, temporal_quarters")
        target_cpu = target.detach().cpu()
        if velocity_init == "zero":
            velocity_uv = torch.zeros((tube_count, 2), dtype=torch.float32, device=target.device)
        elif velocity_init in ("block_match", "block_match_gated"):
            velocity_uv = _block_match_velocity(
                target_cpu,
                frame_ids,
                x,
                y,
                search_radius=int(velocity_search_radius),
                patch_radius=int(velocity_patch_radius),
                min_improvement_ratio=None
                if velocity_init == "block_match"
                else float(velocity_min_improvement_ratio),
            ).to(target.device)
        else:
            raise ValueError("velocity_init must be one of: zero, block_match, block_match_gated")
        color = target_cpu[frame_ids, y, x].clamp(1.0e-5, 1.0 - 1.0e-5).to(target.device)
        center_uv = torch.stack((x.float() + 0.5, y.float() + 0.5), dim=-1).to(target.device)
        center_t = (frame_ids.float() - 0.5 * float(config.frames - 1)).view(tube_count, 1).to(target.device)
        precision = torch.tensor(
            [float(spatial_precision), float(spatial_precision), float(temporal_precision)],
            dtype=torch.float32,
            device=target.device,
        )
        precision = precision.clamp_min(min_precision * 2.0).view(1, 3).expand(tube_count, 3)
        opacity_tensor = torch.full((tube_count,), float(opacity), dtype=torch.float32, device=target.device)
        depth0 = torch.linspace(0.8, 1.2, tube_count, dtype=torch.float32, device=target.device)

        with torch.no_grad():
            model.center_uv.copy_(center_uv)
            model.center_t.copy_(center_t)
            model.velocity_uv.copy_(velocity_uv)
            model.raw_precision.copy_(_inv_softplus(precision - min_precision))
            model.raw_opacity.copy_(_logit(opacity_tensor.clamp(0.0, 0.98) / 0.99))
            model.raw_color.copy_(_logit(color))
            model.depth0.copy_(depth0)
        return model

    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        precision = F.softplus(self.raw_precision) + self.min_precision
        lambda_u = precision[:, 0]
        lambda_v = precision[:, 1]
        lambda_t = precision[:, 2]
        velocity_u = self.velocity_uv[:, 0]
        velocity_v = self.velocity_uv[:, 1]
        zeros = torch.zeros_like(lambda_u)
        q_uvt = torch.stack(
            (
                lambda_u,
                zeros,
                -lambda_u * velocity_u,
                lambda_v,
                -lambda_v * velocity_v,
                lambda_t + lambda_u * velocity_u.square() + lambda_v * velocity_v.square(),
            ),
            dim=-1,
        )
        ma = torch.cat((self.center_uv, self.center_t), dim=-1)
        depth_beta = torch.zeros((self.tube_count, 3), dtype=torch.float32, device=ma.device)
        opacity = torch.sigmoid(self.raw_opacity) * 0.99
        color = torch.sigmoid(self.raw_color)
        return ma, q_uvt, self.depth0, depth_beta, opacity, color

    def temporal_split(
        self,
        *,
        offset_frames: float,
        temporal_precision_scale: float = 2.0,
        opacity_scale: float = 1.0,
        depth_offset: float = 1.0e-4,
    ) -> "ScreenTimeTubeModel":
        """Clone each learned tube into two temporal children for split/refine tests."""

        if offset_frames < 0.0:
            raise ValueError("offset_frames must be non-negative")
        if temporal_precision_scale <= 0.0:
            raise ValueError("temporal_precision_scale must be positive")
        if opacity_scale <= 0.0:
            raise ValueError("opacity_scale must be positive")
        if depth_offset < 0.0:
            raise ValueError("depth_offset must be non-negative")
        child = ScreenTimeTubeModel(
            self.tube_count * 2,
            self.config,
            seed=0,
            device=self.center_uv.device,
            min_precision=self.min_precision,
        )
        with torch.no_grad():
            offsets = torch.tensor(
                [-float(offset_frames), float(offset_frames)],
                dtype=self.center_t.dtype,
                device=self.center_t.device,
            ).view(1, 2, 1)
            min_t = -0.5 * float(self.config.frames - 1)
            max_t = 0.5 * float(self.config.frames - 1)
            center_t = (self.center_t.view(self.tube_count, 1, 1) + offsets).reshape(-1, 1).clamp(min_t, max_t)
            precision = F.softplus(self.raw_precision) + self.min_precision
            child_precision = precision.repeat_interleave(2, dim=0)
            child_precision[:, 2].mul_(float(temporal_precision_scale))
            opacity = torch.sigmoid(self.raw_opacity) * 0.99
            child_opacity = 1.0 - torch.sqrt((1.0 - opacity).clamp_min(1.0e-6))
            child_opacity = (child_opacity * float(opacity_scale)).clamp(0.0, 0.98)
            depth_offsets = torch.tensor(
                [-float(depth_offset), float(depth_offset)],
                dtype=self.depth0.dtype,
                device=self.depth0.device,
            )

            child.center_uv.copy_(self.center_uv.repeat_interleave(2, dim=0))
            child.center_t.copy_(center_t)
            child.velocity_uv.copy_(self.velocity_uv.repeat_interleave(2, dim=0))
            child.raw_precision.copy_(_inv_softplus(child_precision - self.min_precision))
            child.raw_opacity.copy_(_logit(child_opacity.repeat_interleave(2) / 0.99))
            child.raw_color.copy_(self.raw_color.repeat_interleave(2, dim=0))
            child.depth0.copy_(self.depth0.repeat_interleave(2) + depth_offsets.repeat(self.tube_count))
        return child


def dense_differentiable_render_uvt_tubes(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: UVTRenderConfig,
) -> Tensor:
    del depth_beta
    grid = make_uvt_grid(config, ma.device)
    delta = grid.unsqueeze(3) - ma.view(1, 1, 1, -1, 3)
    qv = _quadratic(q_uvt.view(1, 1, 1, -1, 6), delta)
    alpha = torch.clamp(opacity.view(1, 1, 1, -1) * torch.exp(-0.5 * qv), max=config.max_alpha)
    order = torch.argsort(depth0.detach(), stable=True).detach().cpu().tolist()

    background = torch.tensor(config.background, dtype=torch.float32, device=ma.device).view(1, 1, 1, 3)
    accum = torch.zeros((config.frames, config.height, config.width, 3), dtype=torch.float32, device=ma.device)
    transmittance = torch.ones((config.frames, config.height, config.width, 1), dtype=torch.float32, device=ma.device)
    for tube_id in order:
        alpha_i = alpha[..., tube_id].unsqueeze(-1)
        accum = accum + transmittance * alpha_i * color[tube_id].view(1, 1, 1, 3)
        transmittance = transmittance * (1.0 - alpha_i)
    return accum + transmittance * background


def render_model(model: ScreenTimeTubeModel) -> Tensor:
    ma, q_uvt, depth0, depth_beta, opacity, color = model.tensors()
    return dense_differentiable_render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, model.config)
