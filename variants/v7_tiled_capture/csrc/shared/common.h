#pragma once

#include <torch/extension.h>
#include <tuple>

namespace gsplat {

struct MetaView {
  int32_t height;
  int32_t width;
  int32_t batch_size;
  int32_t gaussians_per_batch;
  int32_t front_k;
  int32_t tile_size;
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
      ip[0], ip[1], ip[2], ip[3], ip[4], ip[5],
      fp[0], fp[1], fp[2], fp[3], fp[4],
  };
}

// forward returns:
//   out_device         [B,H,W,3]   float32 on the caller device
//   packed_gaussians   [B,G,12]    float32 CPU aux for backward
//   out_cpu            [B,H,W,3]   float32 CPU exact forward image for fallback replay
//   front_ids          [B,H,W,K]   uint16  CPU saved common-path ids
//   front_raw_alpha    [B,H,W,K]   float32 CPU saved raw alpha
//   front_meta         [B,H,W]     uint8   CPU packed count/overflow
//   tile_offsets       [B,T+1]     int32   CPU global offsets into tile_ids
//   tile_ids           [R]         uint16  CPU flattened tile-local contributor ids
using ForwardOut = std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

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
    const torch::Tensor& packed_gaussians,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& out_cpu,
    const torch::Tensor& front_ids,
    const torch::Tensor& front_raw_alpha,
    const torch::Tensor& front_meta,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_ids);

}  // namespace gsplat
