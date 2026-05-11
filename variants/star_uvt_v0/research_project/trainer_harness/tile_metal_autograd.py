from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_gsplat_bridge_star_uvt import UVTRenderConfig, render_uvt_tubes, stable_backward_samples  # noqa: E402


def _reduce_samples(ids: Tensor, samples: Tensor, tube_count: int, trailing: int | None) -> Tensor:
    flat_ids = ids.reshape(-1)
    if trailing is None:
        sample_rows = samples.reshape(-1)
    else:
        sample_rows = samples.reshape(-1, trailing)
    if sample_rows.shape[0] != flat_ids.shape[0]:
        shared = min(int(sample_rows.shape[0]), int(flat_ids.shape[0]))
        flat_ids = flat_ids[:shared]
        sample_rows = sample_rows[:shared]
    valid_positions = torch.nonzero((flat_ids >= 0) & (flat_ids < tube_count), as_tuple=False).flatten()
    valid_ids = flat_ids.index_select(0, valid_positions).to(torch.int64)
    valid_samples = sample_rows.index_select(0, valid_positions)
    if trailing is None:
        out = torch.zeros((tube_count,), dtype=torch.float32, device=samples.device)
        out.index_add_(0, valid_ids, valid_samples)
        return out
    out = torch.zeros((tube_count, trailing), dtype=torch.float32, device=samples.device)
    out.index_add_(0, valid_ids, valid_samples)
    return out


def _reduce_sample_bundle(
    ids: Tensor,
    grad_ma_samples: Tensor,
    grad_q_samples: Tensor,
    grad_opacity_samples: Tensor,
    grad_color_samples: Tensor,
    tube_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    flat_ids = ids.reshape(-1).to(torch.int64)
    grad_ma_rows = grad_ma_samples.reshape(-1, 3)
    grad_q_rows = grad_q_samples.reshape(-1, 6)
    grad_opacity_rows = grad_opacity_samples.reshape(-1, 1)
    grad_color_rows = grad_color_samples.reshape(-1, 3)
    if not (
        flat_ids.shape[0]
        == grad_ma_rows.shape[0]
        == grad_q_rows.shape[0]
        == grad_opacity_rows.shape[0]
        == grad_color_rows.shape[0]
    ):
        raise ValueError("compact sample ids and gradient rows must have the same length")
    sample_rows = torch.cat(
        (
            grad_ma_rows,
            grad_q_rows,
            grad_opacity_rows,
            grad_color_rows,
        ),
        dim=-1,
    )
    out = torch.zeros((tube_count, 13), dtype=torch.float32, device=grad_ma_samples.device)
    out.index_add_(0, flat_ids, sample_rows)
    return out[:, :3], out[:, 3:9], out[:, 9], out[:, 10:13]


class _MetalTileBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ma: Tensor,
        q_uvt: Tensor,
        depth0: Tensor,
        depth_beta: Tensor,
        opacity: Tensor,
        color: Tensor,
        config: UVTRenderConfig,
    ) -> Tensor:
        ctx.config = config
        ctx.tube_count = int(ma.shape[0])
        ctx.save_for_backward(ma, q_uvt, depth0, depth_beta, opacity, color)
        return render_uvt_tubes(ma, q_uvt, depth0, depth_beta, opacity, color, config)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        ma, q_uvt, depth0, depth_beta, opacity, color = ctx.saved_tensors
        ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, _tile_unstable = stable_backward_samples(
            ma.detach(),
            q_uvt.detach(),
            depth0.detach(),
            depth_beta.detach(),
            opacity.detach(),
            color.detach(),
            grad_output.contiguous(),
            ctx.config,
        )
        tube_count = ctx.tube_count
        grad_ma, grad_q, grad_opacity, grad_color = _reduce_sample_bundle(
            ids,
            grad_ma_samples,
            grad_q_samples,
            grad_opacity_samples,
            grad_color_samples,
            tube_count,
        )
        grad_depth0 = torch.zeros_like(depth0)
        grad_depth_beta = torch.zeros_like(depth_beta)
        return grad_ma, grad_q, grad_depth0, grad_depth_beta, grad_opacity, grad_color, None


def render_uvt_tubes_metal_tile_backward(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    color: Tensor,
    config: UVTRenderConfig,
) -> Tensor:
    """Use Metal forward and Metal per-sample backward with MPS index-add reduction."""

    return _MetalTileBackward.apply(ma, q_uvt, depth0, depth_beta, opacity, color, config)
