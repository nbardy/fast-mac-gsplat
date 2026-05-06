from __future__ import annotations

import torch


def reference_features_and_alpha(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    height: int,
    width: int,
    feature_background: torch.Tensor,
    alpha_threshold: float = 1.0 / 255.0,
    transmittance_threshold: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch front-to-back blend using the v5 pixel-center convention."""
    feature_dim = colors.shape[-1]
    device = means2d.device
    dtype = means2d.dtype

    order = torch.argsort(depths.detach(), stable=True)
    means2d_s = means2d[order]
    conics_s = conics[order]
    colors_s = colors[order]
    opacities_s = opacities[order]

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([xx, yy], dim=-1)

    out_features = torch.zeros(height, width, feature_dim, device=device, dtype=dtype)
    out_alpha = torch.zeros(height, width, device=device, dtype=dtype)
    transmittance = torch.ones(height, width, device=device, dtype=dtype)

    for i in range(means2d_s.shape[0]):
        active = transmittance > transmittance_threshold
        if not active.any():
            break
        d = pixels - means2d_s[i]
        a, b, c = conics_s[i, 0], conics_s[i, 1], conics_s[i, 2]
        power = -0.5 * (a * d[..., 0].square() + 2.0 * b * d[..., 0] * d[..., 1] + c * d[..., 1].square())
        raw_alpha = opacities_s[i] * torch.exp(power)
        alpha = raw_alpha.clamp(max=0.99)
        alpha = torch.where(
            (power <= 0.0) & (alpha >= alpha_threshold) & active,
            alpha,
            torch.zeros_like(alpha),
        )

        contrib = transmittance * alpha
        out_features = out_features + contrib.unsqueeze(-1) * colors_s[i]
        out_alpha = out_alpha + contrib
        transmittance = transmittance * (1.0 - alpha)

    out_features = out_features + transmittance.unsqueeze(-1) * feature_background
    return out_features, out_alpha
