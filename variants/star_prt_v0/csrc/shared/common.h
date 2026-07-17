#pragma once

#include <torch/extension.h>

namespace star_prt {

using PRTBackwardOutputs = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                      torch::Tensor>;

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
  int h_terms;
  int reserved0;

  float alpha_threshold;
  float transmittance_threshold;
  float bg_r;
  float bg_g;
  float bg_b;
  float depth_epsilon;
  float max_alpha;
  float reserved_f0;
};

inline ParsedMeta parse_meta(const torch::Tensor& meta_i32, const torch::Tensor& meta_f32) {
  TORCH_CHECK(meta_i32.dtype() == torch::kInt32, "meta_i32 must be int32");
  TORCH_CHECK(meta_f32.dtype() == torch::kFloat32, "meta_f32 must be float32");
  TORCH_CHECK(meta_i32.numel() >= 14, "meta_i32 must contain at least 14 values");
  TORCH_CHECK(meta_f32.numel() >= 8, "meta_f32 must contain at least 8 values");

  auto mi = meta_i32.cpu();
  auto mf = meta_f32.cpu();
  const auto* ip = mi.data_ptr<int32_t>();
  const auto* fp = mf.data_ptr<float>();

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
  out.h_terms = ip[12];
  out.reserved0 = ip[13];

  out.alpha_threshold = fp[0];
  out.transmittance_threshold = fp[1];
  out.bg_r = fp[2];
  out.bg_g = fp[3];
  out.bg_b = fp[4];
  out.depth_epsilon = fp[5];
  out.max_alpha = fp[6];
  out.reserved_f0 = fp[7];
  return out;
}

inline void check_f32_contiguous(const char* name, const torch::Tensor& tensor, const torch::Device& device) {
  TORCH_CHECK(tensor.dtype() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK(tensor.device() == device, name, " must be on the same device as the first tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_projective_rational_contract(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  const ParsedMeta meta = parse_meta(meta_i32, meta_f32);
  TORCH_CHECK(h_coeff.dim() == 3 && h_coeff.size(2) == 3, "h_coeff must have shape [N,H,3]");
  TORCH_CHECK(h_coeff.size(0) == meta.tube_count, "h_coeff N must match meta tube_count");
  TORCH_CHECK(h_coeff.size(1) == meta.h_terms, "h_coeff H must match meta h_terms");
  const auto device = h_coeff.device();
  check_f32_contiguous("h_coeff", h_coeff, device);
  TORCH_CHECK(lambda_uv.sizes() == torch::IntArrayRef({meta.tube_count, 3}), "lambda_uv must have shape [N,3]");
  TORCH_CHECK(lambda_t.sizes() == torch::IntArrayRef({meta.tube_count}), "lambda_t must have shape [N]");
  TORCH_CHECK(center_t.sizes() == torch::IntArrayRef({meta.tube_count}), "center_t must have shape [N]");
  TORCH_CHECK(opacity.sizes() == torch::IntArrayRef({meta.tube_count}), "opacity must have shape [N]");
  TORCH_CHECK(color.sizes() == torch::IntArrayRef({meta.tube_count, 3}), "color must have shape [N,3]");
  check_f32_contiguous("lambda_uv", lambda_uv, device);
  check_f32_contiguous("lambda_t", lambda_t, device);
  check_f32_contiguous("center_t", center_t, device);
  check_f32_contiguous("opacity", opacity, device);
  check_f32_contiguous("color", color, device);
}

inline void check_compiled_curve_contract(
    const torch::Tensor& curve_uv_depth,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  const ParsedMeta meta = parse_meta(meta_i32, meta_f32);
  TORCH_CHECK(curve_uv_depth.dim() == 3 && curve_uv_depth.size(2) == 3,
              "curve_uv_depth must have shape [F,N,3]");
  TORCH_CHECK(curve_uv_depth.size(0) == meta.frames, "curve_uv_depth F must match meta frames");
  TORCH_CHECK(curve_uv_depth.size(1) == meta.tube_count, "curve_uv_depth N must match meta tube_count");
  const auto device = curve_uv_depth.device();
  check_f32_contiguous("curve_uv_depth", curve_uv_depth, device);
  TORCH_CHECK(lambda_uv.sizes() == torch::IntArrayRef({meta.tube_count, 3}), "lambda_uv must have shape [N,3]");
  TORCH_CHECK(lambda_t.sizes() == torch::IntArrayRef({meta.tube_count}), "lambda_t must have shape [N]");
  TORCH_CHECK(center_t.sizes() == torch::IntArrayRef({meta.tube_count}), "center_t must have shape [N]");
  TORCH_CHECK(opacity.sizes() == torch::IntArrayRef({meta.tube_count}), "opacity must have shape [N]");
  TORCH_CHECK(color.sizes() == torch::IntArrayRef({meta.tube_count, 3}), "color must have shape [N,3]");
  check_f32_contiguous("lambda_uv", lambda_uv, device);
  check_f32_contiguous("lambda_t", lambda_t, device);
  check_f32_contiguous("center_t", center_t, device);
  check_f32_contiguous("opacity", opacity, device);
  check_f32_contiguous("color", color, device);
}

inline void check_projective_rational_backward_contract(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  const ParsedMeta meta = parse_meta(meta_i32, meta_f32);
  check_projective_rational_contract(h_coeff, lambda_uv, lambda_t, center_t, opacity, color, meta_i32, meta_f32);
  TORCH_CHECK(grad_image.sizes() == torch::IntArrayRef({meta.frames, meta.height, meta.width, 3}),
              "grad_image must have shape [F,H,W,3]");
  check_f32_contiguous("grad_image", grad_image, h_coeff.device());
}

torch::Tensor metal_render_projective_rational_tubes(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

torch::Tensor metal_render_compiled_curve_tubes(
    const torch::Tensor& curve_uv_depth,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

PRTBackwardOutputs metal_compact_backward_projective_rational_tubes(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

}  // namespace star_prt
