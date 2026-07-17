from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

try:
    from . import _C  # noqa: F401
except Exception:
    _C = None
from .rasterize import UVTRenderConfig, _depth_at, _quadratic, _runtime_validate


@dataclass(frozen=True)
class UVTFeatureRenderResult:
    feature_image: Tensor
    alpha: Tensor
    tile_counts: Tensor
    tile_overflow: Tensor
    tile_unstable: Tensor
    tile_tube_ids: Tensor | None = None
    tile_depths: Tensor | None = None


@dataclass(frozen=True)
class UVTFeatureSparsePixelsRenderResult:
    feature_values: Tensor
    alpha_values: Tensor
    tile_counts: Tensor
    tile_overflow: Tensor
    tile_unstable: Tensor
    tile_tube_ids: Tensor
    tile_depths: Tensor


@dataclass(frozen=True)
class UVTFeatureAlphaRenderResult:
    alpha: Tensor
    tile_counts: Tensor
    tile_overflow: Tensor
    tile_unstable: Tensor
    tile_tube_ids: Tensor
    tile_depths: Tensor


@dataclass(frozen=True)
class UVTFeatureBinResult:
    tile_counts: Tensor
    tile_overflow: Tensor
    tile_unstable: Tensor
    tile_tube_ids: Tensor
    tile_depths: Tensor


@dataclass(frozen=True)
class UVTFeatureAutogradResult:
    feature_image: Tensor
    alpha: Tensor


@dataclass(frozen=True)
class UVTLinearSigmoidMSEBackwardResult:
    grad_ma: Tensor
    grad_q_uvt: Tensor
    grad_opacity: Tensor
    grad_feature: Tensor
    grad_color_weight: Tensor
    grad_color_bias: Tensor
    tile_unstable: Tensor


@dataclass(frozen=True)
class UVTLogitHandoffPrepResult:
    grad_logits_thw3: Tensor
    grad_alpha: Tensor


@dataclass(frozen=True)
class UVTLogitHandoffBackwardResult:
    grad_ma: Tensor
    grad_q_uvt: Tensor
    grad_opacity: Tensor
    grad_feature: Tensor
    tile_unstable: Tensor


@dataclass(frozen=True)
class UVTHiddenSparseSigmoidMSEBackwardResult:
    grad_ma: Tensor
    grad_q_uvt: Tensor
    grad_opacity: Tensor
    grad_feature: Tensor
    loss: Tensor
    tile_unstable: Tensor


@dataclass(frozen=True)
class UVTHiddenTargetAreaForwardSumsResult:
    pred_sums: Tensor
    tile_unstable: Tensor


@dataclass(frozen=True)
class UVTHiddenTargetAreaBackwardResult:
    grad_ma: Tensor
    grad_q_uvt: Tensor
    grad_opacity: Tensor
    grad_feature: Tensor
    tile_unstable: Tensor
    grad_hidden_weight: Tensor | None = None
    grad_hidden_bias: Tensor | None = None
    grad_output_weight: Tensor | None = None
    grad_output_bias: Tensor | None = None


_TARGET_AREA_BACKWARD_MODE_BITS = {
    "target_area_star_only": 0,
    "target_area_skip_feature_grad": 1,
    "target_area_feature_grad_only": 2,
    "target_area_recompute_only": 3,
    "target_area_traversal_only": 7,
    "target_area_hidden_forward_only": 11,
    "target_area_hidden_preact_only": 19,
    "target_area_star_only_rowmajor_wt": 32,
    "target_area_recompute_only_rowmajor_wt": 35,
    "target_area_star_only_vec4_wt": 64,
    "target_area_recompute_only_vec4_wt": 67,
    "target_area_colorizer_grad_only": 144,
    "target_area_colorizer_vec4_wt": 192,
    "target_area_colorizer_simdreduce_grad_only": 400,
    "target_area_colorizer_simdreduce_vec4_wt": 448,
}


def _target_area_backward_mode_bits(backward_mode: str) -> int:
    try:
        return _TARGET_AREA_BACKWARD_MODE_BITS[backward_mode]
    except KeyError as exc:
        choices = ", ".join(sorted(_TARGET_AREA_BACKWARD_MODE_BITS))
        raise ValueError(f"unknown target-area backward_mode {backward_mode!r}; expected one of: {choices}") from exc


def _make_feature_meta(
    config: UVTRenderConfig,
    device: torch.device,
    tube_count: int,
    feature_dim: int,
    *,
    feature_backward_mode: int = 0,
) -> tuple[Tensor, Tensor]:
    _runtime_validate(config)
    if feature_dim <= 0 or feature_dim > 128:
        raise ValueError("feature_dim must be in 1..128")
    tiles_x = (config.width + config.tile_x - 1) // config.tile_x
    tiles_y = (config.height + config.tile_y - 1) // config.tile_y
    tiles_t = (config.frames + config.tile_t - 1) // config.tile_t
    tile_count = tiles_x * tiles_y * tiles_t
    meta_i32 = torch.tensor(
        [
            config.height,
            config.width,
            config.frames,
            config.tile_x,
            config.tile_y,
            config.tile_t,
            tiles_x,
            tiles_y,
            tiles_t,
            tile_count,
            tube_count,
            config.tile_capacity,
            feature_dim,
            int(feature_backward_mode),
        ],
        device=device,
        dtype=torch.int32,
    )
    meta_f32 = torch.tensor(
        [
            float(config.alpha_threshold),
            float(config.transmittance_threshold),
            0.0,
            0.0,
            0.0,
            1.0e-8,
            float(config.max_alpha),
        ],
        device=device,
        dtype=torch.float32,
    )
    return meta_i32, meta_f32


def _check_feature_inputs(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    *,
    require_mps: bool,
) -> None:
    if ma.ndim != 2 or ma.shape[-1] != 3:
        raise ValueError("ma must have shape [N,3]")
    if q_uvt.shape != (ma.shape[0], 6):
        raise ValueError("q_uvt must have shape [N,6]")
    if depth0.shape != (ma.shape[0],):
        raise ValueError("depth0 must have shape [N]")
    if depth_beta.shape != (ma.shape[0], 3):
        raise ValueError("depth_beta must have shape [N,3]")
    if opacity.shape != (ma.shape[0],):
        raise ValueError("opacity must have shape [N]")
    if feature.ndim != 2 or feature.shape[0] != ma.shape[0]:
        raise ValueError("feature must have shape [N,F]")
    if feature.shape[1] <= 0 or feature.shape[1] > 128:
        raise ValueError("feature dimension must be in 1..128")
    for name, tensor in {
        "ma": ma,
        "q_uvt": q_uvt,
        "depth0": depth0,
        "depth_beta": depth_beta,
        "opacity": opacity,
        "feature": feature,
    }.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != ma.device:
            raise ValueError(f"{name} must be on the same device as ma")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if require_mps and ma.device.type != "mps":
        raise ValueError("Metal STAR-UVT feature render requires MPS tensors")


def _frame_time(frame: int, frames: int) -> float:
    return float(frame) - 0.5 * float(frames - 1)


_BACKWARD_MODE_BITS = {
    "direct_atomic": 0,
    "gradcache": 1,
    "direct_atomic_skip_feature_grad": 2,
    "gradcache_skip_feature_grad": 3,
    "gradcache_reduce_feature_grad": 5,
    "gradcache_reduce_feature_grad_vec4": 17,
    "fused_first3_sigmoid_mse": 8,
    "direct_atomic_feature_grad_only": 32,
    "gradcache_feature_grad_only": 33,
    "gradcache_feature_grad_only_reduce": 37,
    "gradcache_feature_grad_only_reduce_vec4": 49,
}
_CACHED_BIN_BACKWARD_MODES = {
    "direct_atomic_cached_bins": "direct_atomic",
    "gradcache_cached_bins": "gradcache",
    "gradcache_reduce_feature_grad_cached_bins": "gradcache_reduce_feature_grad",
    "gradcache_reduce_feature_grad_vec4_cached_bins": "gradcache_reduce_feature_grad_vec4",
}
_LOGIT_HANDOFF_MODE_BITS = {
    "logit_handoff": 0,
    "logit_handoff_reduce": 4,
    "logit_handoff_reduce_vec4": 16,
}
_HIDDEN_SIGMOID_MSE_MODE_BITS = {
    "hidden_sigmoid_mse_star_only": 0,
    "hidden_sigmoid_mse_star_only_reduce_vec4": 16,
}


def _kernel_backward_mode(backward_mode: str) -> str:
    return _CACHED_BIN_BACKWARD_MODES.get(backward_mode, backward_mode)


def _uses_cached_bins(backward_mode: str) -> bool:
    return backward_mode in _CACHED_BIN_BACKWARD_MODES


def _feature_backward_mode_bits(backward_mode: str, feature_dim: int) -> int:
    kernel_mode = _kernel_backward_mode(backward_mode)
    if kernel_mode not in _BACKWARD_MODE_BITS:
        expected = "', '".join((*_BACKWARD_MODE_BITS, *_CACHED_BIN_BACKWARD_MODES))
        raise ValueError(f"backward_mode must be one of: '{expected}'")
    if kernel_mode == "fused_first3_sigmoid_mse" and not (3 <= feature_dim <= 64):
        raise ValueError("fused_first3_sigmoid_mse requires 3 <= feature_dim <= 64")
    return _BACKWARD_MODE_BITS[kernel_mode]


def _logit_handoff_mode_bits(backward_mode: str) -> int:
    if backward_mode not in _LOGIT_HANDOFF_MODE_BITS:
        expected = "', '".join(_LOGIT_HANDOFF_MODE_BITS)
        raise ValueError(f"backward_mode must be one of: '{expected}'")
    return _LOGIT_HANDOFF_MODE_BITS[backward_mode]


def _hidden_sigmoid_mse_mode_bits(backward_mode: str) -> int:
    if backward_mode not in _HIDDEN_SIGMOID_MSE_MODE_BITS:
        expected = "', '".join(_HIDDEN_SIGMOID_MSE_MODE_BITS)
        raise ValueError(f"backward_mode must be one of: '{expected}'")
    return _HIDDEN_SIGMOID_MSE_MODE_BITS[backward_mode]


def brute_force_render_uvt_feature_tubes(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    config: UVTRenderConfig,
) -> tuple[Tensor, Tensor]:
    _runtime_validate(config)
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=False)
    device = ma.device
    feature_dim = int(feature.shape[1])
    out = torch.zeros((config.frames, feature_dim, config.height, config.width), dtype=torch.float32, device=device)
    alpha_out = torch.zeros((config.frames, config.height, config.width), dtype=torch.float32, device=device)
    for f in range(config.frames):
        t = _frame_time(f, config.frames)
        for y in range(config.height):
            for x in range(config.width):
                a = torch.tensor([x + 0.5, y + 0.5, t], dtype=torch.float32, device=device)
                d = a.unsqueeze(0) - ma
                qv = _quadratic(q_uvt, d)
                alpha = torch.clamp(opacity * torch.exp(-0.5 * qv), max=config.max_alpha)
                active = torch.nonzero(alpha >= config.alpha_threshold, as_tuple=False).flatten()
                if active.numel() == 0:
                    continue
                depths = _depth_at(
                    ma.index_select(0, active),
                    depth0.index_select(0, active),
                    depth_beta.index_select(0, active),
                    a,
                )
                order = torch.argsort(depths, stable=True)
                transmittance = torch.tensor(1.0, dtype=torch.float32, device=device)
                for local_idx in order.tolist():
                    tube_id = int(active[local_idx])
                    ai = alpha[tube_id]
                    out[f, :, y, x] = out[f, :, y, x] + transmittance * ai * feature[tube_id]
                    transmittance = transmittance * (1.0 - ai)
                    if float(transmittance.detach()) <= config.transmittance_threshold:
                        break
                alpha_out[f, y, x] = 1.0 - transmittance
    return out.contiguous(), alpha_out.contiguous()


def render_uvt_feature_tubes(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    config: UVTRenderConfig,
    *,
    return_bins: bool = False,
) -> UVTFeatureRenderResult:
    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    meta_i32, meta_f32 = _make_feature_meta(config, ma.device, ma.shape[0], int(feature.shape[1]))
    if return_bins:
        (
            out_feature,
            alpha,
            tile_counts,
            tile_overflow,
            tile_unstable,
            tile_tube_ids,
            tile_depths,
        ) = torch.ops.star_uvt_v0.render_features_with_bins(
            ma, q_uvt, depth0, depth_beta, opacity, feature, meta_i32, meta_f32
        )
    else:
        out_feature, alpha, tile_counts, tile_overflow, tile_unstable = torch.ops.star_uvt_v0.render_features(
            ma, q_uvt, depth0, depth_beta, opacity, feature, meta_i32, meta_f32
        )
        tile_tube_ids = None
        tile_depths = None
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return UVTFeatureRenderResult(
        feature_image=out_feature.permute(0, 3, 1, 2).contiguous(),
        alpha=alpha.contiguous(),
        tile_counts=tile_counts,
        tile_overflow=tile_overflow,
        tile_unstable=tile_unstable,
        tile_tube_ids=tile_tube_ids,
        tile_depths=tile_depths,
    )


def direct_atomic_feature_backward(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    grad_feature_image: Tensor,
    grad_alpha: Tensor,
    config: UVTRenderConfig,
    *,
    backward_mode: str = "direct_atomic",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    grad_feature_image = grad_feature_image.contiguous()
    grad_alpha = grad_alpha.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    if grad_feature_image.shape != (config.frames, feature.shape[1], config.height, config.width):
        raise ValueError("grad_feature_image must have shape [frames,feature_dim,height,width]")
    if grad_alpha.shape != (config.frames, config.height, config.width):
        raise ValueError("grad_alpha must have shape [frames,height,width]")
    if grad_feature_image.dtype != torch.float32 or grad_feature_image.device != ma.device:
        raise ValueError("grad_feature_image must be float32 and on the same device as ma")
    if grad_alpha.dtype != torch.float32 or grad_alpha.device != ma.device:
        raise ValueError("grad_alpha must be float32 and on the same device as ma")
    feature_backward_mode = _feature_backward_mode_bits(backward_mode, int(feature.shape[1]))
    meta_i32, meta_f32 = _make_feature_meta(
        config,
        ma.device,
        ma.shape[0],
        int(feature.shape[1]),
        feature_backward_mode=feature_backward_mode,
    )
    grad_feature_thwf = grad_feature_image.permute(0, 2, 3, 1).contiguous()
    grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable = torch.ops.star_uvt_v0.direct_atomic_feature_backward(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        grad_feature_thwf,
        grad_alpha,
        meta_i32,
        meta_f32,
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable


def direct_atomic_feature_backward_cached_bins(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    grad_feature_image: Tensor,
    grad_alpha: Tensor,
    tile_counts: Tensor,
    tile_tube_ids: Tensor,
    tile_depths: Tensor,
    tile_unstable: Tensor,
    config: UVTRenderConfig,
    *,
    backward_mode: str = "direct_atomic",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    grad_feature_image = grad_feature_image.contiguous()
    grad_alpha = grad_alpha.contiguous()
    tile_counts = tile_counts.contiguous()
    tile_tube_ids = tile_tube_ids.contiguous()
    tile_depths = tile_depths.contiguous()
    tile_unstable = tile_unstable.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    if grad_feature_image.shape != (config.frames, feature.shape[1], config.height, config.width):
        raise ValueError("grad_feature_image must have shape [frames,feature_dim,height,width]")
    if grad_alpha.shape != (config.frames, config.height, config.width):
        raise ValueError("grad_alpha must have shape [frames,height,width]")
    if grad_feature_image.dtype != torch.float32 or grad_feature_image.device != ma.device:
        raise ValueError("grad_feature_image must be float32 and on the same device as ma")
    if grad_alpha.dtype != torch.float32 or grad_alpha.device != ma.device:
        raise ValueError("grad_alpha must be float32 and on the same device as ma")
    for name, tensor, dtype in (
        ("tile_counts", tile_counts, torch.int32),
        ("tile_tube_ids", tile_tube_ids, torch.int32),
        ("tile_depths", tile_depths, torch.float32),
        ("tile_unstable", tile_unstable, torch.int32),
    ):
        if tensor.dtype != dtype or tensor.device != ma.device:
            raise ValueError(f"{name} must be {dtype} and on the same device as ma")
    feature_backward_mode = _feature_backward_mode_bits(backward_mode, int(feature.shape[1]))
    meta_i32, meta_f32 = _make_feature_meta(
        config,
        ma.device,
        ma.shape[0],
        int(feature.shape[1]),
        feature_backward_mode=feature_backward_mode,
    )
    grad_feature_thwf = grad_feature_image.permute(0, 2, 3, 1).contiguous()
    grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable_out = torch.ops.star_uvt_v0.direct_atomic_feature_backward_with_bins(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        grad_feature_thwf,
        grad_alpha,
        tile_counts,
        tile_tube_ids,
        tile_depths,
        tile_unstable,
        meta_i32,
        meta_f32,
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable_out


def direct_atomic_feature_sparse_pixels_backward_cached_bins(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    pixel_ids: Tensor,
    grad_feature_values: Tensor,
    grad_alpha_values: Tensor,
    tile_counts: Tensor,
    tile_tube_ids: Tensor,
    tile_depths: Tensor,
    tile_unstable: Tensor,
    config: UVTRenderConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    pixel_ids = pixel_ids.contiguous()
    grad_feature_values = grad_feature_values.contiguous()
    grad_alpha_values = grad_alpha_values.contiguous()
    tile_counts = tile_counts.contiguous()
    tile_tube_ids = tile_tube_ids.contiguous()
    tile_depths = tile_depths.contiguous()
    tile_unstable = tile_unstable.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    sparse_count = int(pixel_ids.shape[0])
    if pixel_ids.ndim != 1:
        raise ValueError("pixel_ids must have shape [M]")
    if grad_feature_values.shape != (sparse_count, int(feature.shape[1])):
        raise ValueError("grad_feature_values must have shape [M,feature_dim]")
    if grad_alpha_values.shape != (sparse_count,):
        raise ValueError("grad_alpha_values must have shape [M]")
    for name, tensor, dtype in (
        ("pixel_ids", pixel_ids, torch.int32),
        ("grad_feature_values", grad_feature_values, torch.float32),
        ("grad_alpha_values", grad_alpha_values, torch.float32),
        ("tile_counts", tile_counts, torch.int32),
        ("tile_tube_ids", tile_tube_ids, torch.int32),
        ("tile_depths", tile_depths, torch.float32),
        ("tile_unstable", tile_unstable, torch.int32),
    ):
        if tensor.dtype != dtype or tensor.device != ma.device:
            raise ValueError(f"{name} must be {dtype} and on the same device as ma")
    meta_i32, meta_f32 = _make_feature_meta(config, ma.device, ma.shape[0], int(feature.shape[1]))
    grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable_out = (
        torch.ops.star_uvt_v0.direct_atomic_feature_sparse_pixels_backward_with_bins(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            feature,
            pixel_ids,
            grad_feature_values,
            grad_alpha_values,
            tile_counts,
            tile_tube_ids,
            tile_depths,
            tile_unstable,
            meta_i32,
            meta_f32,
        )
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable_out


def render_uvt_feature_sparse_pixels_with_bins(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    pixel_ids: Tensor,
    config: UVTRenderConfig,
) -> UVTFeatureSparsePixelsRenderResult:
    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    pixel_ids = pixel_ids.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    if pixel_ids.ndim != 1:
        raise ValueError("pixel_ids must have shape [M]")
    if pixel_ids.dtype != torch.int32 or pixel_ids.device != ma.device:
        raise ValueError("pixel_ids must be int32 and on the same device as ma")
    meta_i32, meta_f32 = _make_feature_meta(config, ma.device, ma.shape[0], int(feature.shape[1]))
    (
        feature_values,
        alpha_values,
        tile_counts,
        tile_overflow,
        tile_unstable,
        tile_tube_ids,
        tile_depths,
    ) = torch.ops.star_uvt_v0.render_feature_sparse_pixels_with_bins(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        pixel_ids,
        meta_i32,
        meta_f32,
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return UVTFeatureSparsePixelsRenderResult(
        feature_values=feature_values,
        alpha_values=alpha_values,
        tile_counts=tile_counts,
        tile_overflow=tile_overflow,
        tile_unstable=tile_unstable,
        tile_tube_ids=tile_tube_ids,
        tile_depths=tile_depths,
    )


def render_uvt_feature_alpha_all_pixels_with_bins(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    config: UVTRenderConfig,
) -> UVTFeatureAlphaRenderResult:
    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    if ma.device.type != "mps":
        raise ValueError("render_uvt_feature_alpha_all_pixels_with_bins requires MPS tensors")
    if ma.dtype != torch.float32 or ma.dim() != 2 or ma.shape[1] != 3:
        raise ValueError("ma must be float32 with shape [N,3]")
    if q_uvt.dtype != torch.float32 or q_uvt.device != ma.device or q_uvt.shape != (ma.shape[0], 6):
        raise ValueError("q_uvt must be float32 on ma.device with shape [N,6]")
    if depth0.dtype != torch.float32 or depth0.device != ma.device or depth0.shape != (ma.shape[0],):
        raise ValueError("depth0 must be float32 on ma.device with shape [N]")
    if depth_beta.dtype != torch.float32 or depth_beta.device != ma.device or depth_beta.shape != (ma.shape[0], 3):
        raise ValueError("depth_beta must be float32 on ma.device with shape [N,3]")
    if opacity.dtype != torch.float32 or opacity.device != ma.device or opacity.shape != (ma.shape[0],):
        raise ValueError("opacity must be float32 on ma.device with shape [N]")

    total_pixels = int(config.frames * config.height * config.width)
    if total_pixels <= 0:
        raise ValueError("config must describe at least one pixel")
    if total_pixels > 2_147_483_647:
        raise ValueError("alpha all-pixels render requires total pixel count to fit int32")

    dummy_feature = torch.zeros((ma.shape[0], 1), dtype=torch.float32, device=ma.device)
    pixel_ids = torch.arange(total_pixels, dtype=torch.int32, device=ma.device)
    render = render_uvt_feature_sparse_pixels_with_bins(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        dummy_feature,
        pixel_ids,
        config,
    )
    return UVTFeatureAlphaRenderResult(
        alpha=render.alpha_values.reshape(config.frames, config.height, config.width).contiguous(),
        tile_counts=render.tile_counts,
        tile_overflow=render.tile_overflow,
        tile_unstable=render.tile_unstable,
        tile_tube_ids=render.tile_tube_ids,
        tile_depths=render.tile_depths,
    )


def bin_uvt_feature_tubes(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    config: UVTRenderConfig,
    *,
    feature_dim: int,
) -> UVTFeatureBinResult:
    _runtime_validate(config)
    if int(feature_dim) <= 0 or int(feature_dim) > 128:
        raise ValueError("feature_dim must be in 1..128")
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    if ma.device.type != "mps":
        raise ValueError("bin_uvt_feature_tubes requires MPS tensors")
    if ma.dtype != torch.float32 or ma.dim() != 2 or ma.shape[1] != 3:
        raise ValueError("ma must be float32 with shape [N,3]")
    if q_uvt.dtype != torch.float32 or q_uvt.device != ma.device or q_uvt.shape != (ma.shape[0], 6):
        raise ValueError("q_uvt must be float32 on ma.device with shape [N,6]")
    if depth0.dtype != torch.float32 or depth0.device != ma.device or depth0.shape != (ma.shape[0],):
        raise ValueError("depth0 must be float32 on ma.device with shape [N]")
    if depth_beta.dtype != torch.float32 or depth_beta.device != ma.device or depth_beta.shape != (ma.shape[0], 3):
        raise ValueError("depth_beta must be float32 on ma.device with shape [N,3]")
    if opacity.dtype != torch.float32 or opacity.device != ma.device or opacity.shape != (ma.shape[0],):
        raise ValueError("opacity must be float32 on ma.device with shape [N]")
    meta_i32, meta_f32 = _make_feature_meta(config, ma.device, ma.shape[0], int(feature_dim))
    tile_counts, tile_overflow, tile_unstable, tile_tube_ids, tile_depths = torch.ops.star_uvt_v0.bin_feature_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        meta_i32,
        meta_f32,
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return UVTFeatureBinResult(
        tile_counts=tile_counts,
        tile_overflow=tile_overflow,
        tile_unstable=tile_unstable,
        tile_tube_ids=tile_tube_ids,
        tile_depths=tile_depths,
    )


def direct_linear_sigmoid_mse_backward(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    target_rgb: Tensor,
    color_weight: Tensor,
    color_bias: Tensor,
    config: UVTRenderConfig,
    *,
    compute_colorizer_grad: bool = True,
) -> UVTLinearSigmoidMSEBackwardResult:
    """Direct STAR feature backward for a linear 1x1 sigmoid colorizer + mean MSE.

    This is a benchmark/prototype handoff surface. It covers
    `alpha * sigmoid(W @ feature_image + b)` with zero background and returns
    explicit colorizer parameter gradients. It does not support hidden
    colorizers, pre-norm, view-conditioning, or non-MSE losses.
    """

    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    target_rgb = target_rgb.contiguous()
    color_weight = color_weight.contiguous()
    color_bias = color_bias.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    feature_dim = int(feature.shape[1])
    if feature_dim > 64:
        raise ValueError("direct_linear_sigmoid_mse_backward requires feature_dim <= 64")
    if target_rgb.shape != (config.frames, 3, config.height, config.width):
        raise ValueError("target_rgb must have shape [frames,3,height,width]")
    if color_weight.shape != (3, feature_dim):
        raise ValueError("color_weight must have shape [3,feature_dim]")
    if color_bias.shape != (3,):
        raise ValueError("color_bias must have shape [3]")
    for name, tensor in {
        "target_rgb": target_rgb,
        "color_weight": color_weight,
        "color_bias": color_bias,
    }.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != ma.device:
            raise ValueError(f"{name} must be on the same device as ma")
    feature_backward_mode = 0 if compute_colorizer_grad else 16
    meta_i32, meta_f32 = _make_feature_meta(
        config,
        ma.device,
        ma.shape[0],
        feature_dim,
        feature_backward_mode=feature_backward_mode,
    )
    target_thw3 = target_rgb.permute(0, 2, 3, 1).contiguous()
    (
        grad_ma,
        grad_q,
        grad_opacity,
        grad_feature,
        grad_color_weight,
        grad_color_bias,
        tile_unstable,
    ) = torch.ops.star_uvt_v0.direct_atomic_feature_linear_sigmoid_mse_backward(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        target_thw3,
        color_weight,
        color_bias,
        meta_i32,
        meta_f32,
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return UVTLinearSigmoidMSEBackwardResult(
        grad_ma=grad_ma,
        grad_q_uvt=grad_q,
        grad_opacity=grad_opacity,
        grad_feature=grad_feature,
        grad_color_weight=grad_color_weight,
        grad_color_bias=grad_color_bias,
        tile_unstable=tile_unstable,
    )


def direct_hidden_sigmoid_mse_backward(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    target_rgb: Tensor,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    config: UVTRenderConfig,
    *,
    backward_mode: str = "hidden_sigmoid_mse_star_only_reduce_vec4",
) -> UVTLogitHandoffBackwardResult:
    """Fused STAR feature backward for hidden FeatureToColor sigmoid MSE.

    This benchmark-only path computes a no-pre-norm
    `Conv1x1 -> GELU -> Conv1x1 -> sigmoid -> alpha-compose -> mean MSE` VJP
    inside the STAR reverse traversal. It returns STAR parameter gradients only;
    colorizer parameter gradients are intentionally omitted.
    """

    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    target_rgb = target_rgb.contiguous()
    hidden_weight = hidden_weight.contiguous()
    hidden_bias = hidden_bias.contiguous()
    output_weight = output_weight.contiguous()
    output_bias = output_bias.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    feature_dim = int(feature.shape[1])
    hidden_dim = int(hidden_weight.shape[0]) if hidden_weight.dim() == 2 else 0
    if feature_dim > 64:
        raise ValueError("direct_hidden_sigmoid_mse_backward requires feature_dim <= 64")
    if hidden_dim <= 0 or hidden_dim > 64:
        raise ValueError("direct_hidden_sigmoid_mse_backward requires hidden_dim in 1..64")
    if target_rgb.shape != (config.frames, 3, config.height, config.width):
        raise ValueError("target_rgb must have shape [frames,3,height,width]")
    if hidden_weight.shape != (hidden_dim, feature_dim):
        raise ValueError("hidden_weight must have shape [hidden_dim,feature_dim]")
    if hidden_bias.shape != (hidden_dim,):
        raise ValueError("hidden_bias must have shape [hidden_dim]")
    if output_weight.shape != (3, hidden_dim):
        raise ValueError("output_weight must have shape [3,hidden_dim]")
    if output_bias.shape != (3,):
        raise ValueError("output_bias must have shape [3]")
    for name, tensor in {
        "target_rgb": target_rgb,
        "hidden_weight": hidden_weight,
        "hidden_bias": hidden_bias,
        "output_weight": output_weight,
        "output_bias": output_bias,
    }.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != ma.device:
            raise ValueError(f"{name} must be on the same device as ma")
    feature_backward_mode = _hidden_sigmoid_mse_mode_bits(backward_mode)
    meta_i32, meta_f32 = _make_feature_meta(
        config,
        ma.device,
        ma.shape[0],
        feature_dim,
        feature_backward_mode=feature_backward_mode,
    )
    target_thw3 = target_rgb.permute(0, 2, 3, 1).contiguous()
    grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable = torch.ops.star_uvt_v0.direct_atomic_feature_hidden_sigmoid_mse_backward(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        target_thw3,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        meta_i32,
        meta_f32,
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return UVTLogitHandoffBackwardResult(
        grad_ma=grad_ma,
        grad_q_uvt=grad_q,
        grad_opacity=grad_opacity,
        grad_feature=grad_feature,
        tile_unstable=tile_unstable,
    )


def direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    pixel_ids: Tensor,
    target_rgb_values: Tensor,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    tile_counts: Tensor,
    tile_tube_ids: Tensor,
    tile_depths: Tensor,
    tile_unstable: Tensor,
    config: UVTRenderConfig,
    *,
    total_loss_elems: int | None = None,
) -> UVTHiddenSparseSigmoidMSEBackwardResult:
    """Fused sparse-pixel hidden FeatureToColor sigmoid-MSE STAR backward.

    This benchmark-only path computes a no-pre-norm hidden RGB loss VJP for only
    `pixel_ids`, using cached STAR tile bins. It returns STAR parameter gradients
    and the sparse mean RGB MSE loss; colorizer parameter gradients are omitted.
    `total_loss_elems` lets chunked trainer calls normalize against the full
    step loss denominator instead of the current chunk only.
    """

    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    pixel_ids = pixel_ids.contiguous()
    target_rgb_values = target_rgb_values.contiguous()
    hidden_weight = hidden_weight.contiguous()
    hidden_bias = hidden_bias.contiguous()
    output_weight = output_weight.contiguous()
    output_bias = output_bias.contiguous()
    tile_counts = tile_counts.contiguous()
    tile_tube_ids = tile_tube_ids.contiguous()
    tile_depths = tile_depths.contiguous()
    tile_unstable = tile_unstable.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    feature_dim = int(feature.shape[1])
    hidden_dim = int(hidden_weight.shape[0]) if hidden_weight.dim() == 2 else 0
    sparse_count = int(pixel_ids.shape[0])
    if feature_dim > 64:
        raise ValueError("direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins requires feature_dim <= 64")
    if hidden_dim <= 0 or hidden_dim > 64:
        raise ValueError("direct_hidden_sigmoid_mse_sparse_pixels_backward_cached_bins requires hidden_dim in 1..64")
    if pixel_ids.ndim != 1:
        raise ValueError("pixel_ids must have shape [M]")
    if target_rgb_values.shape != (sparse_count, 3):
        raise ValueError("target_rgb_values must have shape [M,3]")
    if total_loss_elems is None:
        total_loss_elems = max(sparse_count * 3, 1)
    if int(total_loss_elems) <= 0:
        raise ValueError("total_loss_elems must be positive")
    if hidden_weight.shape != (hidden_dim, feature_dim):
        raise ValueError("hidden_weight must have shape [hidden_dim,feature_dim]")
    if hidden_bias.shape != (hidden_dim,):
        raise ValueError("hidden_bias must have shape [hidden_dim]")
    if output_weight.shape != (3, hidden_dim):
        raise ValueError("output_weight must have shape [3,hidden_dim]")
    if output_bias.shape != (3,):
        raise ValueError("output_bias must have shape [3]")
    for name, tensor, dtype in (
        ("pixel_ids", pixel_ids, torch.int32),
        ("target_rgb_values", target_rgb_values, torch.float32),
        ("hidden_weight", hidden_weight, torch.float32),
        ("hidden_bias", hidden_bias, torch.float32),
        ("output_weight", output_weight, torch.float32),
        ("output_bias", output_bias, torch.float32),
        ("tile_counts", tile_counts, torch.int32),
        ("tile_tube_ids", tile_tube_ids, torch.int32),
        ("tile_depths", tile_depths, torch.float32),
        ("tile_unstable", tile_unstable, torch.int32),
    ):
        if tensor.dtype != dtype or tensor.device != ma.device:
            raise ValueError(f"{name} must be {dtype} and on the same device as ma")
    meta_i32, meta_f32 = _make_feature_meta(
        config,
        ma.device,
        ma.shape[0],
        feature_dim,
        feature_backward_mode=int(total_loss_elems),
    )
    grad_ma, grad_q, grad_opacity, grad_feature, loss, tile_unstable_out = (
        torch.ops.star_uvt_v0.direct_atomic_feature_sparse_hidden_sigmoid_mse_backward_with_bins(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            feature,
            pixel_ids,
            target_rgb_values,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
            tile_counts,
            tile_tube_ids,
            tile_depths,
            tile_unstable,
            meta_i32,
            meta_f32,
        )
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return UVTHiddenSparseSigmoidMSEBackwardResult(
        grad_ma=grad_ma,
        grad_q_uvt=grad_q,
        grad_opacity=grad_opacity,
        grad_feature=grad_feature,
        loss=loss,
        tile_unstable=tile_unstable_out,
    )


def sparse_hidden_sigmoid_target_area_forward_sums_cached_bins(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    pixel_ids: Tensor,
    cell_ids: Tensor,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    tile_counts: Tensor,
    tile_tube_ids: Tensor,
    tile_depths: Tensor,
    tile_unstable: Tensor,
    config: UVTRenderConfig,
    *,
    cell_count: int,
) -> UVTHiddenTargetAreaForwardSumsResult:
    """Accumulate hidden FeatureToColor RGB sums per target-area cell."""

    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    pixel_ids = pixel_ids.contiguous()
    cell_ids = cell_ids.contiguous()
    hidden_weight = hidden_weight.contiguous()
    hidden_bias = hidden_bias.contiguous()
    output_weight = output_weight.contiguous()
    output_bias = output_bias.contiguous()
    tile_counts = tile_counts.contiguous()
    tile_tube_ids = tile_tube_ids.contiguous()
    tile_depths = tile_depths.contiguous()
    tile_unstable = tile_unstable.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    feature_dim = int(feature.shape[1])
    hidden_dim = int(hidden_weight.shape[0]) if hidden_weight.dim() == 2 else 0
    if feature_dim > 64:
        raise ValueError("target-area hidden VJP requires feature_dim <= 64")
    if hidden_dim <= 0 or hidden_dim > 64:
        raise ValueError("target-area hidden VJP requires hidden_dim in 1..64")
    if int(cell_count) <= 0:
        raise ValueError("cell_count must be positive")
    if pixel_ids.ndim != 1 or cell_ids.shape != pixel_ids.shape:
        raise ValueError("pixel_ids and cell_ids must have matching shape [M]")
    if hidden_weight.shape != (hidden_dim, feature_dim):
        raise ValueError("hidden_weight must have shape [hidden_dim,feature_dim]")
    if hidden_bias.shape != (hidden_dim,):
        raise ValueError("hidden_bias must have shape [hidden_dim]")
    if output_weight.shape != (3, hidden_dim):
        raise ValueError("output_weight must have shape [3,hidden_dim]")
    if output_bias.shape != (3,):
        raise ValueError("output_bias must have shape [3]")
    for name, tensor, dtype in (
        ("pixel_ids", pixel_ids, torch.int32),
        ("cell_ids", cell_ids, torch.int32),
        ("hidden_weight", hidden_weight, torch.float32),
        ("hidden_bias", hidden_bias, torch.float32),
        ("output_weight", output_weight, torch.float32),
        ("output_bias", output_bias, torch.float32),
        ("tile_counts", tile_counts, torch.int32),
        ("tile_tube_ids", tile_tube_ids, torch.int32),
        ("tile_depths", tile_depths, torch.float32),
        ("tile_unstable", tile_unstable, torch.int32),
    ):
        if tensor.dtype != dtype or tensor.device != ma.device:
            raise ValueError(f"{name} must be {dtype} and on the same device as ma")
    meta_i32, meta_f32 = _make_feature_meta(config, ma.device, ma.shape[0], feature_dim)
    pred_sums, tile_unstable_out = torch.ops.star_uvt_v0.sparse_hidden_sigmoid_target_area_forward_sums_with_bins(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        pixel_ids,
        cell_ids,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        tile_counts,
        tile_tube_ids,
        tile_depths,
        tile_unstable,
        meta_i32,
        meta_f32,
        int(cell_count),
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return UVTHiddenTargetAreaForwardSumsResult(pred_sums=pred_sums, tile_unstable=tile_unstable_out)


def direct_hidden_sigmoid_target_area_backward_cached_bins(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    pixel_ids: Tensor,
    cell_ids: Tensor,
    cell_grad_rgb: Tensor,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
    tile_counts: Tensor,
    tile_tube_ids: Tensor,
    tile_depths: Tensor,
    tile_unstable: Tensor,
    config: UVTRenderConfig,
    *,
    backward_mode: str = "target_area_star_only",
) -> UVTHiddenTargetAreaBackwardResult:
    """Backpropagate target-area cell RGB gradients through hidden FeatureToColor."""

    _runtime_validate(config)
    mode_bits = _target_area_backward_mode_bits(backward_mode)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    pixel_ids = pixel_ids.contiguous()
    cell_ids = cell_ids.contiguous()
    cell_grad_rgb = cell_grad_rgb.contiguous()
    hidden_weight = hidden_weight.contiguous()
    hidden_bias = hidden_bias.contiguous()
    output_weight = output_weight.contiguous()
    output_bias = output_bias.contiguous()
    tile_counts = tile_counts.contiguous()
    tile_tube_ids = tile_tube_ids.contiguous()
    tile_depths = tile_depths.contiguous()
    tile_unstable = tile_unstable.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    feature_dim = int(feature.shape[1])
    hidden_dim = int(hidden_weight.shape[0]) if hidden_weight.dim() == 2 else 0
    if feature_dim > 64:
        raise ValueError("target-area hidden VJP requires feature_dim <= 64")
    if hidden_dim <= 0 or hidden_dim > 64:
        raise ValueError("target-area hidden VJP requires hidden_dim in 1..64")
    if pixel_ids.ndim != 1 or cell_ids.shape != pixel_ids.shape:
        raise ValueError("pixel_ids and cell_ids must have matching shape [M]")
    if cell_grad_rgb.dim() != 2 or int(cell_grad_rgb.shape[1]) != 3:
        raise ValueError("cell_grad_rgb must have shape [C,3]")
    if hidden_weight.shape != (hidden_dim, feature_dim):
        raise ValueError("hidden_weight must have shape [hidden_dim,feature_dim]")
    if hidden_bias.shape != (hidden_dim,):
        raise ValueError("hidden_bias must have shape [hidden_dim]")
    if output_weight.shape != (3, hidden_dim):
        raise ValueError("output_weight must have shape [3,hidden_dim]")
    if output_bias.shape != (3,):
        raise ValueError("output_bias must have shape [3]")
    for name, tensor, dtype in (
        ("pixel_ids", pixel_ids, torch.int32),
        ("cell_ids", cell_ids, torch.int32),
        ("cell_grad_rgb", cell_grad_rgb, torch.float32),
        ("hidden_weight", hidden_weight, torch.float32),
        ("hidden_bias", hidden_bias, torch.float32),
        ("output_weight", output_weight, torch.float32),
        ("output_bias", output_bias, torch.float32),
        ("tile_counts", tile_counts, torch.int32),
        ("tile_tube_ids", tile_tube_ids, torch.int32),
        ("tile_depths", tile_depths, torch.float32),
        ("tile_unstable", tile_unstable, torch.int32),
    ):
        if tensor.dtype != dtype or tensor.device != ma.device:
            raise ValueError(f"{name} must be {dtype} and on the same device as ma")
    meta_i32, meta_f32 = _make_feature_meta(config, ma.device, ma.shape[0], feature_dim)
    (
        grad_ma,
        grad_q,
        grad_opacity,
        grad_feature,
        tile_unstable_out,
        grad_hidden_weight,
        grad_hidden_bias,
        grad_output_weight,
        grad_output_bias,
    ) = (
        torch.ops.star_uvt_v0.direct_atomic_feature_sparse_hidden_target_area_backward_with_bins(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            feature,
            pixel_ids,
            cell_ids,
            cell_grad_rgb,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
            tile_counts,
            tile_tube_ids,
            tile_depths,
            tile_unstable,
            meta_i32,
            meta_f32,
            int(mode_bits),
        )
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return UVTHiddenTargetAreaBackwardResult(
        grad_ma=grad_ma,
        grad_q_uvt=grad_q,
        grad_opacity=grad_opacity,
        grad_feature=grad_feature,
        tile_unstable=tile_unstable_out,
        grad_hidden_weight=grad_hidden_weight,
        grad_hidden_bias=grad_hidden_bias,
        grad_output_weight=grad_output_weight,
        grad_output_bias=grad_output_bias,
    )


def linear_sigmoid_mse_logit_handoff_prep(
    feature_image: Tensor,
    alpha: Tensor,
    target_rgb: Tensor,
    color_weight: Tensor,
    color_bias: Tensor,
    config: UVTRenderConfig,
) -> UVTLogitHandoffPrepResult:
    """Compute image-space logit handoff gradients for a linear sigmoid MSE.

    This is a benchmark-only prep surface for `direct_logit_handoff_backward`.
    It consumes rendered feature images in `[T,F,H,W]` layout and emits
    `grad_logits` in `[T,H,W,3]` layout so the reverse traversal can consume it
    without a Python-side permute/copy.
    """

    _runtime_validate(config)
    feature_image = feature_image.contiguous()
    alpha = alpha.contiguous()
    target_rgb = target_rgb.contiguous()
    color_weight = color_weight.contiguous()
    color_bias = color_bias.contiguous()
    feature_dim = int(feature_image.shape[1]) if feature_image.dim() >= 2 else 0
    if feature_dim <= 0 or feature_dim > 64:
        raise ValueError("linear_sigmoid_mse_logit_handoff_prep requires feature_dim in 1..64")
    if feature_image.shape != (config.frames, feature_dim, config.height, config.width):
        raise ValueError("feature_image must have shape [frames,feature_dim,height,width]")
    if alpha.shape != (config.frames, config.height, config.width):
        raise ValueError("alpha must have shape [frames,height,width]")
    if target_rgb.shape != (config.frames, 3, config.height, config.width):
        raise ValueError("target_rgb must have shape [frames,3,height,width]")
    if color_weight.shape != (3, feature_dim):
        raise ValueError("color_weight must have shape [3,feature_dim]")
    if color_bias.shape != (3,):
        raise ValueError("color_bias must have shape [3]")
    for name, tensor in {
        "feature_image": feature_image,
        "alpha": alpha,
        "target_rgb": target_rgb,
        "color_weight": color_weight,
        "color_bias": color_bias,
    }.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != feature_image.device:
            raise ValueError(f"{name} must be on the same device as feature_image")
    if feature_image.device.type != "mps":
        raise ValueError("linear_sigmoid_mse_logit_handoff_prep requires MPS tensors")
    grad_logits_thw3, grad_alpha = torch.ops.star_uvt_v0.linear_sigmoid_mse_handoff_prep(
        feature_image,
        alpha,
        target_rgb,
        color_weight,
        color_bias,
    )
    if feature_image.device.type == "mps":
        torch.mps.synchronize()
    return UVTLogitHandoffPrepResult(grad_logits_thw3=grad_logits_thw3, grad_alpha=grad_alpha)


def direct_logit_handoff_backward(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    grad_logits: Tensor,
    grad_alpha: Tensor,
    color_weight: Tensor,
    config: UVTRenderConfig,
    *,
    backward_mode: str = "logit_handoff",
    grad_logits_layout: str = "tchw",
) -> UVTLogitHandoffBackwardResult:
    """Direct STAR feature backward from image-space colorizer gradients.

    The caller owns colorizer/loss differentiation and passes
    `grad_logits = d loss / d logits` plus `grad_alpha = d loss / d alpha`.
    The Metal kernel only forms `W^T @ grad_logits` per pixel and applies the
    STAR feature-tube reverse traversal. It returns no colorizer parameter
    gradients.
    """

    _runtime_validate(config)
    ma = ma.contiguous()
    q_uvt = q_uvt.contiguous()
    depth0 = depth0.contiguous()
    depth_beta = depth_beta.contiguous()
    opacity = opacity.contiguous()
    feature = feature.contiguous()
    grad_logits = grad_logits.contiguous()
    grad_alpha = grad_alpha.contiguous()
    color_weight = color_weight.contiguous()
    _check_feature_inputs(ma, q_uvt, depth0, depth_beta, opacity, feature, require_mps=True)
    feature_dim = int(feature.shape[1])
    if feature_dim > 64:
        raise ValueError("direct_logit_handoff_backward requires feature_dim <= 64")
    if grad_logits_layout == "tchw":
        if grad_logits.shape != (config.frames, 3, config.height, config.width):
            raise ValueError("grad_logits must have shape [frames,3,height,width]")
    elif grad_logits_layout == "thw3":
        if grad_logits.shape != (config.frames, config.height, config.width, 3):
            raise ValueError("grad_logits must have shape [frames,height,width,3]")
    else:
        raise ValueError("grad_logits_layout must be 'tchw' or 'thw3'")
    if grad_alpha.shape != (config.frames, config.height, config.width):
        raise ValueError("grad_alpha must have shape [frames,height,width]")
    if color_weight.shape != (3, feature_dim):
        raise ValueError("color_weight must have shape [3,feature_dim]")
    for name, tensor in {
        "grad_logits": grad_logits,
        "grad_alpha": grad_alpha,
        "color_weight": color_weight,
    }.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != ma.device:
            raise ValueError(f"{name} must be on the same device as ma")
    feature_backward_mode = _logit_handoff_mode_bits(backward_mode)
    meta_i32, meta_f32 = _make_feature_meta(
        config,
        ma.device,
        ma.shape[0],
        feature_dim,
        feature_backward_mode=feature_backward_mode,
    )
    grad_logits_thw3 = grad_logits if grad_logits_layout == "thw3" else grad_logits.permute(0, 2, 3, 1).contiguous()
    grad_ma, grad_q, grad_opacity, grad_feature, tile_unstable = torch.ops.star_uvt_v0.direct_atomic_feature_logit_handoff_backward(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        grad_logits_thw3,
        grad_alpha,
        color_weight,
        meta_i32,
        meta_f32,
    )
    if ma.device.type == "mps":
        torch.mps.synchronize()
    return UVTLogitHandoffBackwardResult(
        grad_ma=grad_ma,
        grad_q_uvt=grad_q,
        grad_opacity=grad_opacity,
        grad_feature=grad_feature,
        tile_unstable=tile_unstable,
    )


def shift_ma_for_frame_chunk(
    ma: Tensor,
    *,
    global_frames: int,
    frame_start: int,
    chunk_frames: int,
) -> Tensor:
    """Shift tube centers so a local frame chunk uses full-clip time coordinates."""

    if global_frames <= 0 or chunk_frames <= 0:
        raise ValueError("global_frames and chunk_frames must be positive")
    if frame_start < 0 or frame_start + chunk_frames > global_frames:
        raise ValueError("frame chunk must be inside the global frame range")
    offset = float(frame_start) - 0.5 * float(global_frames - 1) + 0.5 * float(chunk_frames - 1)
    delta = torch.tensor((0.0, 0.0, offset), dtype=ma.dtype, device=ma.device).view(1, 3)
    return ma - delta


def chunked_uvt_config(config: UVTRenderConfig, *, chunk_frames: int) -> UVTRenderConfig:
    if chunk_frames <= 0 or chunk_frames > config.frames:
        raise ValueError("chunk_frames must be in 1..config.frames")
    return replace(config, frames=int(chunk_frames))


class _DirectAtomicFeatureRender(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        depth_beta: Tensor,
        opacity: Tensor,
        feature: Tensor,
        config: UVTRenderConfig,
        backward_mode: str,
    ) -> tuple[Tensor, Tensor]:
        ctx.config = config
        ctx.backward_mode = backward_mode
        ctx.cached_bins = _uses_cached_bins(backward_mode)
        ctx.kernel_backward_mode = _kernel_backward_mode(backward_mode)
        ma_c = ma.contiguous()
        q_c = q_uvt.contiguous()
        depth0_c = depth0.contiguous()
        depth_beta_c = depth_beta.contiguous()
        opacity_c = opacity.contiguous()
        feature_c = feature.contiguous()
        result = render_uvt_feature_tubes(
            ma_c,
            q_c,
            depth0_c,
            depth_beta_c,
            opacity_c,
            feature_c,
            config,
            return_bins=ctx.cached_bins,
        )
        if ctx.cached_bins:
            if result.tile_tube_ids is None or result.tile_depths is None:
                raise RuntimeError("cached-bin feature render did not return tile bins")
            ctx.save_for_backward(
                ma_c,
                q_c,
                depth0_c,
                depth_beta_c,
                opacity_c,
                feature_c,
                result.tile_counts,
                result.tile_tube_ids,
                result.tile_depths,
                result.tile_unstable,
            )
        else:
            ctx.save_for_backward(ma_c, q_c, depth0_c, depth_beta_c, opacity_c, feature_c)
        return result.feature_image, result.alpha

    @staticmethod
    def backward(ctx, grad_feature_image: Tensor, grad_alpha: Tensor) -> tuple[Tensor | None, ...]:
        if ctx.cached_bins:
            ma, q_uvt, depth0, depth_beta, opacity, feature, tile_counts, tile_tube_ids, tile_depths, tile_unstable = ctx.saved_tensors
            grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = direct_atomic_feature_backward_cached_bins(
                ma,
                q_uvt,
                depth0,
                depth_beta,
                opacity,
                feature,
                grad_feature_image.contiguous(),
                grad_alpha.contiguous(),
                tile_counts,
                tile_tube_ids,
                tile_depths,
                tile_unstable,
                ctx.config,
                backward_mode=ctx.kernel_backward_mode,
            )
        else:
            ma, q_uvt, depth0, depth_beta, opacity, feature = ctx.saved_tensors
            grad_ma, grad_q, grad_opacity, grad_feature, _tile_unstable = direct_atomic_feature_backward(
                ma,
                q_uvt,
                depth0,
                depth_beta,
                opacity,
                feature,
                grad_feature_image.contiguous(),
                grad_alpha.contiguous(),
                ctx.config,
                backward_mode=ctx.kernel_backward_mode,
            )
        return grad_ma, grad_q, None, None, grad_opacity, grad_feature, None, None


def render_uvt_feature_tubes_autograd(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    config: UVTRenderConfig,
    *,
    backward_mode: str = "direct_atomic",
) -> UVTFeatureAutogradResult:
    """Render feature tubes with direct Metal backward for train-loop use.

    Depth inputs are used for compositing order but intentionally receive no
    gradient, matching the current RGB direct-backward contract.
    """

    feature_image, alpha = _DirectAtomicFeatureRender.apply(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        config,
        backward_mode,
    )
    return UVTFeatureAutogradResult(feature_image=feature_image, alpha=alpha)


def render_uvt_feature_tubes_autograd_frame_chunk(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    config: UVTRenderConfig,
    *,
    frame_start: int,
    chunk_frames: int,
    backward_mode: str = "direct_atomic",
) -> UVTFeatureAutogradResult:
    ma_chunk = shift_ma_for_frame_chunk(
        ma,
        global_frames=config.frames,
        frame_start=frame_start,
        chunk_frames=chunk_frames,
    )
    return render_uvt_feature_tubes_autograd(
        ma_chunk,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        chunked_uvt_config(config, chunk_frames=chunk_frames),
        backward_mode=backward_mode,
    )
