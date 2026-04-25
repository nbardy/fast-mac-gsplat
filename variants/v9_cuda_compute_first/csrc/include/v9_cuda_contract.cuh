#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

namespace v9cuda {

constexpr int kTileSize = 16;
constexpr int kThreadsPerTile = kTileSize * kTileSize;
constexpr int kChunkSize = 256;

struct RasterParams {
  int batch;
  int cameras;
  int gaussians;
  int height;
  int width;
  int tile_width;
  int tile_height;
  float alpha_threshold;
  float transmittance_threshold;
  float max_alpha;
};

// Exact front-to-back contract shared by CUDA forward and backward:
//
//   alpha = min(max_alpha, opacity * exp(-0.5 * q))
//   C += T * alpha * color
//   T *= (1 - alpha)
//   stop when T <= transmittance_threshold
//
// Backward must replay the identical processed prefix in reverse. The first
// scaffold uses render_alpha + last_id because it is easy to validate against
// gsplat/Graphdeco. A later V8-style tile_stop_count path can trade state
// memory for recompute after parity is proven.

void project_count_fused_cuda(
    at::Tensor means3d,
    at::Tensor covars,
    at::Tensor opacities,
    at::Tensor cameras,
    at::Tensor means2d,
    at::Tensor conics,
    at::Tensor depths,
    at::Tensor tiles_per_gauss,
    at::Tensor flags,
    RasterParams params);

void emit_pairs_cuda(
    at::Tensor means2d,
    at::Tensor conics,
    at::Tensor depths,
    at::Tensor opacities,
    at::Tensor pair_offsets,
    at::Tensor isect_keys,
    at::Tensor gaussian_ids,
    at::Tensor overflow,
    RasterParams params);

void tile_forward_train_cuda(
    at::Tensor means2d,
    at::Tensor conics,
    at::Tensor colors,
    at::Tensor opacities,
    at::Tensor sorted_keys,
    at::Tensor sorted_gaussian_ids,
    at::Tensor tile_offsets,
    at::Tensor out_rgb,
    at::Tensor out_alpha,
    at::Tensor last_ids,
    RasterParams params);

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
    RasterParams params);

}  // namespace v9cuda

