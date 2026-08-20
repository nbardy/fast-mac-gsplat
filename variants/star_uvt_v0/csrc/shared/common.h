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
  float alpha_mode;
};

inline ParsedMeta parse_meta(const torch::Tensor& meta_i32, const torch::Tensor& meta_f32) {
  TORCH_CHECK(meta_i32.scalar_type() == torch::kInt32, "meta_i32 must be int32");
  TORCH_CHECK(meta_f32.scalar_type() == torch::kFloat32, "meta_f32 must be float32");
  TORCH_CHECK(meta_i32.numel() >= 14, "meta_i32 must contain at least 14 values");
  TORCH_CHECK(
      meta_f32.numel() >= 8,
      "meta_f32 must contain alpha-mode metadata; rebuild/update the STAR-UVT wrapper");
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
  out.alpha_mode = fp[7];
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_uvt_gated(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& active_start,
    const torch::Tensor& active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

torch::Tensor metal_render_projective_trace_tiles(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_primitive_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_projective_trace_backward(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_primitive_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px);

torch::Tensor metal_render_projective_trace_cell_tiles(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px);

torch::Tensor metal_render_projective_trace_cell_interval_tiles(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_time_coeffs,
    const torch::Tensor& spatial_precision_uv,
    const torch::Tensor& depth_affine_uv,
    const torch::Tensor& color,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px);

torch::Tensor metal_render_projective_trace_family_interval_tiles(
    const torch::Tensor& family_coeffs,
    const torch::Tensor& q_basis,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_time_coeffs,
    const torch::Tensor& spatial_precision_uv,
    const torch::Tensor& depth_affine_uv,
    const torch::Tensor& color,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_projective_trace_family_interval_backward(
    const torch::Tensor& family_coeffs,
    const torch::Tensor& q_basis,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_time_coeffs,
    const torch::Tensor& spatial_precision_uv,
    const torch::Tensor& depth_affine_uv,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px);

torch::Tensor metal_render_projective_trace_cell_interval_rows(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_time_coeffs,
    const torch::Tensor& spatial_precision_uv,
    const torch::Tensor& depth_affine_uv,
    const torch::Tensor& color,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& row_weights,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_projective_trace_cell_interval_backward(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    const torch::Tensor& opacity,
    const torch::Tensor& opacity_time_coeffs,
    const torch::Tensor& spatial_precision_uv,
    const torch::Tensor& depth_affine_uv,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_trace_ids,
    const torch::Tensor& tile_active_start,
    const torch::Tensor& tile_active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    double sigma_px);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_uvt_features(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_render_uvt_features_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_bin_feature_tubes(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_atomic_backward_gated(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& active_start,
    const torch::Tensor& active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_atomic_feature_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& grad_feature_image,
    const torch::Tensor& grad_alpha_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> metal_direct_atomic_feature_backward_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& grad_feature_image,
    const torch::Tensor& grad_alpha_image,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_tube_ids,
    const torch::Tensor& tile_depths,
    const torch::Tensor& tile_unstable,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_direct_atomic_feature_sparse_pixels_backward_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& grad_feature_values,
    const torch::Tensor& grad_alpha_values,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_tube_ids,
    const torch::Tensor& tile_depths,
    const torch::Tensor& tile_unstable,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_direct_atomic_feature_sparse_hidden_sigmoid_mse_backward_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& target_rgb_values,
    const torch::Tensor& hidden_weight,
    const torch::Tensor& hidden_bias,
    const torch::Tensor& output_weight,
    const torch::Tensor& output_bias,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_tube_ids,
    const torch::Tensor& tile_depths,
    const torch::Tensor& tile_unstable,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor>
metal_sparse_hidden_sigmoid_target_area_forward_sums_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& cell_ids,
    const torch::Tensor& hidden_weight,
    const torch::Tensor& hidden_bias,
    const torch::Tensor& output_weight,
    const torch::Tensor& output_bias,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_tube_ids,
    const torch::Tensor& tile_depths,
    const torch::Tensor& tile_unstable,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    int64_t cell_count);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
metal_direct_atomic_feature_sparse_hidden_target_area_backward_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& cell_ids,
    const torch::Tensor& cell_grad_rgb,
    const torch::Tensor& hidden_weight,
    const torch::Tensor& hidden_bias,
    const torch::Tensor& output_weight,
    const torch::Tensor& output_bias,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_tube_ids,
    const torch::Tensor& tile_depths,
    const torch::Tensor& tile_unstable,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    int64_t mode_bits);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_render_feature_sparse_pixels_with_bins(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_direct_atomic_feature_linear_sigmoid_mse_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& target_rgb,
    const torch::Tensor& color_weight,
    const torch::Tensor& color_bias,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor> metal_linear_sigmoid_mse_handoff_prep(
    const torch::Tensor& feature_image,
    const torch::Tensor& alpha,
    const torch::Tensor& target_rgb,
    const torch::Tensor& color_weight,
    const torch::Tensor& color_bias);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_direct_atomic_feature_hidden_sigmoid_mse_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& target_rgb,
    const torch::Tensor& hidden_weight,
    const torch::Tensor& hidden_bias,
    const torch::Tensor& output_weight,
    const torch::Tensor& output_bias,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
metal_direct_atomic_feature_logit_handoff_backward(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& grad_logits,
    const torch::Tensor& grad_alpha_image,
    const torch::Tensor& color_weight,
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

torch::Tensor metal_projective_trace_eval(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    double eps);

torch::Tensor metal_projective_trace_family_eval(
    const torch::Tensor& family_coeffs,
    const torch::Tensor& q_basis,
    const torch::Tensor& times,
    double eps);

std::tuple<torch::Tensor, torch::Tensor> metal_projective_trace_family_backward(
    const torch::Tensor& family_coeffs,
    const torch::Tensor& q_basis,
    const torch::Tensor& times,
    const torch::Tensor& grad_out,
    double eps);

}  // namespace star_uvt
