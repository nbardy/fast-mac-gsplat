#pragma once

#include <torch/extension.h>
#include <tuple>
#include <vector>

namespace gsplat {

struct MetaView {
  int32_t height;
  int32_t width;
  int32_t tiles_y;
  int32_t tiles_x;
  int32_t tile_size;
  int32_t gaussians;
  int32_t tile_count;
  int32_t max_tile_pairs;

  float alpha_threshold;
  float transmittance_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float eps;
  float max_alpha;
};

inline MetaView parse_meta(const torch::Tensor& meta_i32, const torch::Tensor& meta_f32) {
  auto i = meta_i32.cpu();
  auto f = meta_f32.cpu();
  auto ip = i.data_ptr<int32_t>();
  auto fp = f.data_ptr<float>();
  return MetaView{
      ip[0], ip[1], ip[2], ip[3], ip[4], ip[5], ip[6], ip[7],
      fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], fp[6],
  };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_forward(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids);

}  // namespace gsplat
