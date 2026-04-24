#pragma once

#include <torch/extension.h>
#include <tuple>

namespace gsplat {

constexpr int32_t kMaxFrontK = 8;

struct MetaView {
  int32_t height;
  int32_t width;
  int32_t batch_size;
  int32_t gaussians_per_batch;
  int32_t front_k;
  float alpha_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float eps;
};

inline MetaView parse_meta(const torch::Tensor& meta_i32, const torch::Tensor& meta_f32) {
  auto i = meta_i32.cpu();
  auto f = meta_f32.cpu();
  auto ip = i.data_ptr<int32_t>();
  auto fp = f.data_ptr<float>();
  return MetaView{
      ip[0], ip[1], ip[2], ip[3], ip[4],
      fp[0], fp[1], fp[2], fp[3], fp[4],
  };
}

using ForwardOut = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
using BackwardOut = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

ForwardOut metal_forward(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

BackwardOut metal_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& out_image,
    const torch::Tensor& front_ids,
    const torch::Tensor& front_raw_alpha,
    const torch::Tensor& front_count,
    const torch::Tensor& overflow_mask);

}  // namespace gsplat
