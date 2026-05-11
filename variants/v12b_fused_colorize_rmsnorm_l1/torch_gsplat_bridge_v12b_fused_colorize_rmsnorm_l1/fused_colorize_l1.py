from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


Activation = Literal["sigmoid", "identity"]
NormMode = Literal["rms", "none"]
Reduction = Literal["mean", "sum"]


@dataclass(frozen=True)
class FusedColorizeL1Output:
    loss: Tensor
    rgb: Tensor
    composed_rgb: Tensor
    logits: Tensor
    inv_rms: Tensor | None


@dataclass(frozen=True)
class ManualFusedColorizeL1Grads:
    loss: Tensor
    grad_features: Tensor
    grad_alpha: Tensor
    grad_color_weight: Tensor
    grad_color_bias: Tensor
    grad_rms_gamma: Tensor | None


def _check_activation(value: str) -> Activation:
    if value not in {"sigmoid", "identity"}:
        raise ValueError(f"activation must be 'sigmoid' or 'identity', got {value!r}")
    return value  # type: ignore[return-value]


def _check_norm(value: str) -> NormMode:
    if value not in {"rms", "none"}:
        raise ValueError(f"norm must be 'rms' or 'none', got {value!r}")
    return value  # type: ignore[return-value]


def _check_reduction(value: str) -> Reduction:
    if value not in {"mean", "sum"}:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {value!r}")
    return value  # type: ignore[return-value]


def _as_bhw3(value: Tensor, *, batch: int, height: int, width: int, name: str) -> Tensor:
    if value.ndim == 1 and int(value.shape[0]) == 3:
        value = value.view(1, 1, 1, 3)
    elif value.ndim == 4 and int(value.shape[-1]) == 3:
        pass
    elif value.ndim == 4 and int(value.shape[1]) == 3:
        value = value.permute(0, 2, 3, 1)
    else:
        raise ValueError(
            f"{name} must have shape [3], [B,H,W,3], or [B,3,H,W]; got {tuple(value.shape)}"
        )
    try:
        return torch.broadcast_to(value, (batch, height, width, 3))
    except RuntimeError as exc:
        raise ValueError(
            f"{name} with shape {tuple(value.shape)} cannot broadcast to {(batch, height, width, 3)}"
        ) from exc


def _validate_inputs(
    features_bhwf: Tensor,
    alpha_bhw: Tensor,
    target_rgb: Tensor,
    background_rgb: Tensor,
    color_weight_3f: Tensor,
    color_bias_3: Tensor,
    rms_gamma_f: Tensor | None,
    *,
    norm: NormMode,
) -> tuple[Tensor, Tensor]:
    if features_bhwf.ndim != 4:
        raise ValueError(f"features_bhwf must have shape [B,H,W,F], got {tuple(features_bhwf.shape)}")
    if alpha_bhw.shape != features_bhwf.shape[:3]:
        raise ValueError(
            f"alpha_bhw must have shape {tuple(features_bhwf.shape[:3])}, got {tuple(alpha_bhw.shape)}"
        )
    feature_dim = int(features_bhwf.shape[-1])
    if color_weight_3f.shape != (3, feature_dim):
        raise ValueError(f"color_weight_3f must have shape {(3, feature_dim)}, got {tuple(color_weight_3f.shape)}")
    if color_bias_3.shape != (3,):
        raise ValueError(f"color_bias_3 must have shape (3,), got {tuple(color_bias_3.shape)}")
    if norm == "rms":
        if rms_gamma_f is None:
            raise ValueError("rms_gamma_f is required when norm='rms'")
        if rms_gamma_f.shape != (feature_dim,):
            raise ValueError(f"rms_gamma_f must have shape {(feature_dim,)}, got {tuple(rms_gamma_f.shape)}")
    batch, height, width, _ = features_bhwf.shape
    target_bhw3 = _as_bhw3(target_rgb, batch=batch, height=height, width=width, name="target_rgb")
    background_bhw3 = _as_bhw3(background_rgb, batch=batch, height=height, width=width, name="background_rgb")
    return target_bhw3, background_bhw3


def fused_rmsnorm_colorize_alpha_l1_loss(
    features_bhwf: Tensor,
    alpha_bhw: Tensor,
    target_rgb: Tensor,
    background_rgb: Tensor,
    color_weight_3f: Tensor,
    color_bias_3: Tensor,
    rms_gamma_f: Tensor | None,
    *,
    eps: float = 1.0e-6,
    activation: Activation = "sigmoid",
    norm: NormMode = "rms",
    reduction: Reduction = "mean",
) -> FusedColorizeL1Output:
    """Reference fused RMSNorm + 1x1 colorize + alpha-compose + L1 loss.

    This is intentionally a PyTorch reference, not the final Metal fast path.
    It fixes the v12b shape contract and gives the Metal kernel an exact parity
    target while leaving stable fast-mac variants untouched.
    """

    activation = _check_activation(activation)
    norm = _check_norm(norm)
    reduction = _check_reduction(reduction)
    target_bhw3, background_bhw3 = _validate_inputs(
        features_bhwf,
        alpha_bhw,
        target_rgb,
        background_rgb,
        color_weight_3f,
        color_bias_3,
        rms_gamma_f,
        norm=norm,
    )

    if norm == "rms":
        inv_rms = torch.rsqrt(features_bhwf.square().mean(dim=-1, keepdim=True) + float(eps))
        prepared = features_bhwf * inv_rms * rms_gamma_f.view(1, 1, 1, -1)
    else:
        inv_rms = None
        prepared = features_bhwf

    logits = torch.einsum("bhwf,cf->bhwc", prepared, color_weight_3f) + color_bias_3.view(1, 1, 1, 3)
    rgb = torch.sigmoid(logits) if activation == "sigmoid" else logits
    alpha = alpha_bhw.unsqueeze(-1)
    composed = alpha * rgb + (1.0 - alpha) * background_bhw3
    delta_abs = (composed - target_bhw3).abs()
    loss = delta_abs.mean() if reduction == "mean" else delta_abs.sum()
    return FusedColorizeL1Output(loss=loss, rgb=rgb, composed_rgb=composed, logits=logits, inv_rms=inv_rms)


def manual_fused_rmsnorm_colorize_alpha_l1_grads(
    features_bhwf: Tensor,
    alpha_bhw: Tensor,
    target_rgb: Tensor,
    background_rgb: Tensor,
    color_weight_3f: Tensor,
    color_bias_3: Tensor,
    rms_gamma_f: Tensor | None,
    *,
    eps: float = 1.0e-6,
    activation: Activation = "sigmoid",
    norm: NormMode = "rms",
    reduction: Reduction = "mean",
) -> ManualFusedColorizeL1Grads:
    """Closed-form image-space gradients for the v12b fused reference."""

    activation = _check_activation(activation)
    norm = _check_norm(norm)
    reduction = _check_reduction(reduction)
    target_bhw3, background_bhw3 = _validate_inputs(
        features_bhwf,
        alpha_bhw,
        target_rgb,
        background_rgb,
        color_weight_3f,
        color_bias_3,
        rms_gamma_f,
        norm=norm,
    )

    batch, height, width, feature_dim = features_bhwf.shape
    x = features_bhwf.reshape(-1, feature_dim)
    alpha = alpha_bhw.reshape(-1, 1)
    target = target_bhw3.reshape(-1, 3)
    background = background_bhw3.reshape(-1, 3)

    if norm == "rms":
        assert rms_gamma_f is not None
        inv = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + float(eps))
        y = x * inv * rms_gamma_f.view(1, -1)
    else:
        inv = None
        y = x

    logits = y @ color_weight_3f.t() + color_bias_3.view(1, 3)
    rgb = torch.sigmoid(logits) if activation == "sigmoid" else logits
    prediction = alpha * rgb + (1.0 - alpha) * background
    delta = prediction - target
    loss = delta.abs().mean() if reduction == "mean" else delta.abs().sum()
    scale = 1.0 / float(delta.numel()) if reduction == "mean" else 1.0

    grad_prediction = torch.sign(delta) * scale
    grad_rgb = grad_prediction * alpha
    grad_logits = grad_rgb * rgb * (1.0 - rgb) if activation == "sigmoid" else grad_rgb
    grad_alpha = (grad_prediction * (rgb - background)).sum(dim=-1).reshape(batch, height, width)
    grad_color_weight = grad_logits.t() @ y
    grad_color_bias = grad_logits.sum(dim=0)
    grad_y = grad_logits @ color_weight_3f

    if norm == "rms":
        assert inv is not None and rms_gamma_f is not None
        u = grad_y * rms_gamma_f.view(1, -1)
        dot = (u * x).sum(dim=-1, keepdim=True)
        grad_x = inv * u - x * (inv * inv * inv / float(feature_dim)) * dot
        grad_gamma = (grad_y * x * inv).sum(dim=0)
    else:
        grad_x = grad_y
        grad_gamma = None

    return ManualFusedColorizeL1Grads(
        loss=loss,
        grad_features=grad_x.reshape_as(features_bhwf),
        grad_alpha=grad_alpha,
        grad_color_weight=grad_color_weight,
        grad_color_bias=grad_color_bias,
        grad_rms_gamma=grad_gamma,
    )
