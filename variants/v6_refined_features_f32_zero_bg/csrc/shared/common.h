#pragma once

#include <torch/extension.h>
#include <tuple>

namespace gsplat {

struct ParsedMeta {
  int height;
  int width;
  int tiles_y;
  int tiles_x;
  int tile_size;
  int gaussians;
  int tile_count;
  int max_fast_pairs;
  int batch_size;
  int gaussians_per_batch;
  int tiles_per_image;
  int feature_dim;
  int stop_count_mode;
  int stop_count_dense_threshold;
  int reserved0;

  float alpha_threshold;
  float transmittance_threshold;
  float eps;
  float max_alpha;
};

inline ParsedMeta parse_meta(const torch::Tensor& meta_i32, const torch::Tensor& meta_f32) {
  TORCH_CHECK(meta_i32.numel() >= 15, "meta_i32 must have at least 15 values");
  TORCH_CHECK(meta_f32.numel() >= 4, "meta_f32 must have at least 4 values");
  auto mi = meta_i32.cpu();
  auto mf = meta_f32.cpu();
  auto* ip = mi.data_ptr<int32_t>();
  auto* fp = mf.data_ptr<float>();
  ParsedMeta out;
  out.height = ip[0];
  out.width = ip[1];
  out.tiles_y = ip[2];
  out.tiles_x = ip[3];
  out.tile_size = ip[4];
  out.gaussians = ip[5];
  out.tile_count = ip[6];
  out.max_fast_pairs = ip[7];
  out.batch_size = ip[8];
  out.gaussians_per_batch = ip[9];
  out.tiles_per_image = ip[10];
  out.feature_dim = ip[11];
  out.stop_count_mode = ip[12];
  out.stop_count_dense_threshold = ip[13];
  out.reserved0 = ip[14];
  out.alpha_threshold = fp[0];
  out.transmittance_threshold = fp[1];
  out.eps = fp[2];
  out.max_alpha = fp[3];
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_bin(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor> metal_render_fast_forward_eval(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_render_fast_forward_state(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    torch::Tensor& binned_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_fast_backward_saved(
    const torch::Tensor& grad_features,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids,
    const torch::Tensor& tile_stop_counts);

std::tuple<torch::Tensor, torch::Tensor> metal_render_active_forward_eval(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_render_active_forward_state(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    torch::Tensor& binned_ids,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_active_backward_saved(
    const torch::Tensor& grad_features,
    const torch::Tensor& grad_alpha,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids,
    const torch::Tensor& tile_stop_counts);

std::tuple<torch::Tensor, torch::Tensor> metal_render_overflow_forward(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& overflow_tile_ids,
    const torch::Tensor& overflow_tile_offsets,
    const torch::Tensor& overflow_sorted_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_overflow_backward(
    const torch::Tensor& grad_features_tiles,
    const torch::Tensor& grad_alpha_tiles,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& overflow_tile_ids,
    const torch::Tensor& overflow_tile_offsets,
    const torch::Tensor& overflow_sorted_ids);

}  // namespace gsplat
