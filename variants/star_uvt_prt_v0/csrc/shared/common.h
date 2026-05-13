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
  float support_alpha_threshold;
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
  out.support_alpha_threshold = fp[7];
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

torch::Tensor metal_render_projective_rational_direct(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_projective_rational_tiled(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_bin_inverse_homography_atlas_residual_tiles(
    const torch::Tensor& atlas_ref_uv,
    const torch::Tensor& atlas_residual_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& band_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_render_inverse_homography_atlas_residual_tiles(
    const torch::Tensor& atlas_ref_uv,
    const torch::Tensor& atlas_residual_coeff,
    const torch::Tensor& homographies,
    const torch::Tensor& inv_homographies,
    const torch::Tensor& depth,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& band_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_profile_projective_rational_tiled(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_projective_rational_direct_serial_backward(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_projective_rational_tile_pair_atomic_backward(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_projective_rational_tile_pixel_atomic_backward(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
metal_projective_rational_tile_pixel_fused_mse_backward(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& target_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
metal_projective_rational_tile_pixel_fused_mse_train_used_backward(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& target_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
metal_profile_projective_rational_tile_pixel_atomic_backward(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_stable_backward_samples_with_keys(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_backward_samples_compensated(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_target_bounds_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_suffix_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_parallel_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_grouped_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_sharedsort_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_scanline_backward_samples(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_atomic_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_fixedpoint_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_atomic_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_fixedpoint_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_split_fixedpoint_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_serial_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_reduce_sample_bundle_scan(
    const torch::Tensor& ids,
    const torch::Tensor& grad_ma_samples,
    const torch::Tensor& grad_q_samples,
    const torch::Tensor& grad_opacity_samples,
    const torch::Tensor& grad_color_samples,
    int64_t tube_count);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_reduce_sample_bundle_scan_compensated(
    const torch::Tensor& ids,
    const torch::Tensor& grad_ma_samples,
    const torch::Tensor& grad_q_samples,
    const torch::Tensor& grad_opacity_samples,
    const torch::Tensor& grad_color_samples,
    int64_t tube_count);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_reduce_sample_bundle_sorted_segments(
    const torch::Tensor& ids,
    const torch::Tensor& grad_ma_samples,
    const torch::Tensor& grad_q_samples,
    const torch::Tensor& grad_opacity_samples,
    const torch::Tensor& grad_color_samples,
    int64_t tube_count);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_reduced_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_reduced_parallel_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_tile_pair_suffix_reduced_backward(
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
