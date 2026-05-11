#pragma once

#include <torch/extension.h>

#include <tuple>

namespace star_uvt {

struct ParsedMeta {
  int height;
  int width;
  int frames;
  int tile_x;
  int tile_y;
  int tile_t;
  int tiles_x;
  int tiles_y;
  int tiles_t;
  int tile_count;
  int tube_count;
  int tile_capacity;
  int reserved0;
  int reserved1;

  float alpha_threshold;
  float transmittance_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float eps;
  float max_alpha;
};

inline ParsedMeta parse_meta(const torch::Tensor& meta_i32, const torch::Tensor& meta_f32) {
  auto mi = meta_i32.cpu();
  auto mf = meta_f32.cpu();
  auto* ip = mi.data_ptr<int32_t>();
  auto* fp = mf.data_ptr<float>();

  ParsedMeta out;
  out.height = ip[0];
  out.width = ip[1];
  out.frames = ip[2];
  out.tile_x = ip[3];
  out.tile_y = ip[4];
  out.tile_t = ip[5];
  out.tiles_x = ip[6];
  out.tiles_y = ip[7];
  out.tiles_t = ip[8];
  out.tile_count = ip[9];
  out.tube_count = ip[10];
  out.tile_capacity = ip[11];
  out.reserved0 = ip[12];
  out.reserved1 = ip[13];

  out.alpha_threshold = fp[0];
  out.transmittance_threshold = fp[1];
  out.bg_r = fp[2];
  out.bg_g = fp[3];
  out.bg_b = fp[4];
  out.eps = fp[5];
  out.max_alpha = fp[6];
  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_uvt(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_simple_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_stable_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

}  // namespace star_uvt
