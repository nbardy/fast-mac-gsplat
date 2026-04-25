#include "v9_cuda_contract.cuh"

namespace v9cuda {

namespace {

__global__ void tile_backward_replay_kernel() {
  // Skeleton contract:
  //
  // Replay each pixel's processed prefix in reverse. Use out_alpha/last_ids at
  // first because they make parity against gsplat/Graphdeco straightforward.
  //
  // Per pixel:
  //   T_cur = 1 - out_alpha[pixel]
  //   gT = dot(grad_rgb[pixel], background)
  //
  // For sorted references in reverse until this pixel's last_id:
  //   recompute alpha, raw, power, dx, dy
  //   denom = max(1 - alpha, eps)
  //   T_prev = T_cur / denom
  //   dot_c = dot(grad_rgb, color)
  //   d_alpha = T_prev * (dot_c - gT)
  //   d_color = grad_rgb * (T_prev * alpha)
  //   d_raw = d_alpha * clamp_gate * visible_gate
  //   d_power = d_raw * raw
  //   d_conic = d_power * [-0.5*dx^2, -dx*dy, -0.5*dy^2]
  //   d_mean = d_power * [a*dx + b*dy, b*dx + c*dy]
  //   d_opacity = d_raw * raw / max(opacity, eps)
  //   gT = alpha * dot_c + (1 - alpha) * gT
  //   T_cur = T_prev
  //
  // Reduction boundary:
  //   Baseline A: warp reduction then global atomicAdd.
  //   Baseline B: full 16x16 block reduction, then one atomic per
  //               splat/tile/component.
  //   Deferred partial reduce is a later Nsight-driven ablation.
}

}  // namespace

void tile_backward_replay_cuda(
    at::Tensor means2d,
    at::Tensor conics,
    at::Tensor colors,
    at::Tensor opacities,
    at::Tensor sorted_keys,
    at::Tensor sorted_gaussian_ids,
    at::Tensor tile_offsets,
    at::Tensor grad_rgb,
    at::Tensor out_alpha,
    at::Tensor last_ids,
    at::Tensor grad_means2d,
    at::Tensor grad_conics,
    at::Tensor grad_colors,
    at::Tensor grad_opacities,
    RasterParams params) {
  TORCH_CHECK(means2d.is_cuda() && conics.is_cuda(), "projected inputs must be CUDA tensors");
  TORCH_CHECK(colors.is_cuda() && opacities.is_cuda(), "splat attributes must be CUDA tensors");
  TORCH_CHECK(sorted_keys.is_cuda() && sorted_gaussian_ids.is_cuda(),
              "sorted intersection buffers must be CUDA tensors");
  TORCH_CHECK(tile_offsets.is_cuda(), "tile_offsets must be a CUDA tensor");
  TORCH_CHECK(grad_rgb.is_cuda() && out_alpha.is_cuda() && last_ids.is_cuda(),
              "replay inputs must be CUDA tensors");
  TORCH_CHECK(grad_means2d.is_cuda() && grad_conics.is_cuda() &&
                  grad_colors.is_cuda() && grad_opacities.is_cuda(),
              "gradient outputs must be CUDA tensors");
  (void)params;
  TORCH_CHECK(false,
              "tile_backward_replay_cuda is a V9 CUDA scaffold. Implement "
              "reverse replay and reduction variants on a CUDA host.");
}

}  // namespace v9cuda

