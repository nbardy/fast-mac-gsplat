#include <torch/extension.h>

#include "shared/common.h"

namespace star_uvt {
namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_render_uvt(ma, q_uvt, depth0, depth_beta, opacity, color, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.render: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_gated_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& active_start,
    const torch::Tensor& active_stop,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_render_uvt_gated(ma, q_uvt, depth0, depth_beta, opacity, color, active_start, active_stop, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.render_gated: no backend available for device ", ma.device());
}

torch::Tensor projective_trace_eval_dispatch(
    const torch::Tensor& coeffs,
    const torch::Tensor& times,
    double eps) {
#if defined(__APPLE__)
  if (coeffs.device().is_mps()) {
    return metal_projective_trace_eval(coeffs, times, eps);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.projective_trace_eval: no backend available for device ", coeffs.device());
}

torch::Tensor projective_trace_family_eval_dispatch(
    const torch::Tensor& family_coeffs,
    const torch::Tensor& q_basis,
    const torch::Tensor& times,
    double eps) {
#if defined(__APPLE__)
  if (family_coeffs.device().is_mps()) {
    return metal_projective_trace_family_eval(family_coeffs, q_basis, times, eps);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.projective_trace_family_eval: no backend available for device ", family_coeffs.device());
}

std::tuple<torch::Tensor, torch::Tensor> projective_trace_family_backward_dispatch(
    const torch::Tensor& family_coeffs,
    const torch::Tensor& q_basis,
    const torch::Tensor& times,
    const torch::Tensor& grad_out,
    double eps) {
#if defined(__APPLE__)
  if (family_coeffs.device().is_mps()) {
    return metal_projective_trace_family_backward(family_coeffs, q_basis, times, grad_out, eps);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.projective_trace_family_backward: no backend available for device ", family_coeffs.device());
}

torch::Tensor render_projective_trace_tiles_dispatch(
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
    double sigma_px) {
#if defined(__APPLE__)
  if (coeffs.device().is_mps()) {
    return metal_render_projective_trace_tiles(
        coeffs,
        times,
        opacity,
        color,
        tile_counts,
        tile_primitive_ids,
        tile_active_start,
        tile_active_stop,
        meta_i32,
        meta_f32,
        sigma_px);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.render_projective_trace_tiles: no backend available for device ", coeffs.device());
}

torch::Tensor render_projective_trace_cell_tiles_dispatch(
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
    double sigma_px) {
#if defined(__APPLE__)
  if (coeffs.device().is_mps()) {
    return metal_render_projective_trace_cell_tiles(
        coeffs,
        times,
        opacity,
        color,
        tile_counts,
        tile_trace_ids,
        tile_active_start,
        tile_active_stop,
        meta_i32,
        meta_f32,
        sigma_px);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.render_projective_trace_cell_tiles: no backend available for device ", coeffs.device());
}

torch::Tensor render_projective_trace_cell_interval_tiles_dispatch(
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
    double sigma_px) {
#if defined(__APPLE__)
  if (coeffs.device().is_mps()) {
    return metal_render_projective_trace_cell_interval_tiles(
        coeffs,
        times,
        opacity,
        opacity_time_coeffs,
        spatial_precision_uv,
        depth_affine_uv,
        color,
        tile_counts,
        tile_trace_ids,
        tile_active_start,
        tile_active_stop,
        meta_i32,
        meta_f32,
        sigma_px);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.render_projective_trace_cell_interval_tiles: no backend available for device ", coeffs.device());
}

torch::Tensor render_projective_trace_family_interval_tiles_dispatch(
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
    double sigma_px) {
#if defined(__APPLE__)
  if (family_coeffs.device().is_mps()) {
    return metal_render_projective_trace_family_interval_tiles(
        family_coeffs,
        q_basis,
        times,
        opacity,
        opacity_time_coeffs,
        spatial_precision_uv,
        depth_affine_uv,
        color,
        tile_counts,
        tile_trace_ids,
        tile_active_start,
        tile_active_stop,
        meta_i32,
        meta_f32,
        sigma_px);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.render_projective_trace_family_interval_tiles: no backend available for device ", family_coeffs.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_projective_trace_family_interval_backward_dispatch(
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
    double sigma_px) {
#if defined(__APPLE__)
  if (family_coeffs.device().is_mps()) {
    return metal_direct_projective_trace_family_interval_backward(
        family_coeffs,
        q_basis,
        times,
        opacity,
        opacity_time_coeffs,
        spatial_precision_uv,
        depth_affine_uv,
        color,
        grad_image,
        tile_counts,
        tile_trace_ids,
        tile_active_start,
        tile_active_stop,
        meta_i32,
        meta_f32,
        sigma_px);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_projective_trace_family_interval_backward: no backend available for device ", family_coeffs.device());
}

torch::Tensor render_projective_trace_cell_interval_rows_dispatch(
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
    double sigma_px) {
#if defined(__APPLE__)
  if (coeffs.device().is_mps()) {
    return metal_render_projective_trace_cell_interval_rows(
        coeffs,
        times,
        opacity,
        opacity_time_coeffs,
        spatial_precision_uv,
        depth_affine_uv,
        color,
        tile_counts,
        tile_trace_ids,
        tile_active_start,
        tile_active_stop,
        row_weights,
        meta_i32,
        meta_f32,
        sigma_px);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.render_projective_trace_cell_interval_rows: no backend available for device ", coeffs.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> direct_projective_trace_backward_dispatch(
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
    double sigma_px) {
#if defined(__APPLE__)
  if (coeffs.device().is_mps()) {
    return metal_direct_projective_trace_backward(
        coeffs,
        times,
        opacity,
        color,
        grad_image,
        tile_counts,
        tile_primitive_ids,
        tile_active_start,
        tile_active_stop,
        meta_i32,
        meta_f32,
        sigma_px);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_projective_trace_backward: no backend available for device ", coeffs.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_projective_trace_cell_interval_backward_dispatch(
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
    double sigma_px) {
#if defined(__APPLE__)
  if (coeffs.device().is_mps()) {
    return metal_direct_projective_trace_cell_interval_backward(
        coeffs,
        times,
        opacity,
        opacity_time_coeffs,
        spatial_precision_uv,
        depth_affine_uv,
        color,
        grad_image,
        tile_counts,
        tile_trace_ids,
        tile_active_start,
        tile_active_stop,
        meta_i32,
        meta_f32,
        sigma_px);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_projective_trace_cell_interval_backward: no backend available for device ", coeffs.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_features_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_render_uvt_features(ma, q_uvt, depth0, depth_beta, opacity, feature, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.render_features: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_features_with_bins_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_render_uvt_features_with_bins(ma, q_uvt, depth0, depth_beta, opacity, feature, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.render_features_with_bins: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> bin_feature_tubes_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_bin_feature_tubes(ma, q_uvt, depth0, depth_beta, opacity, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.bin_feature_tubes: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> simple_backward_samples_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_simple_backward_samples(ma, q_uvt, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.simple_backward_samples: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> stable_backward_samples_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_stable_backward_samples(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.stable_backward_samples: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> stable_backward_samples_with_keys_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_stable_backward_samples_with_keys(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.stable_backward_samples_with_keys: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_backward_samples_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_backward_samples(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_backward_samples: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_backward_samples_compensated_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_backward_samples_compensated(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_backward_samples_compensated: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_target_bounds_backward_samples_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_target_bounds_backward_samples(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_target_bounds_backward_samples: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_suffix_backward_samples_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_suffix_backward_samples(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_suffix_backward_samples: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_parallel_backward_samples_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_parallel_backward_samples(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_parallel_backward_samples: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_grouped_backward_samples_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_grouped_backward_samples(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_grouped_backward_samples: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_sharedsort_backward_samples_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_sharedsort_backward_samples(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_sharedsort_backward_samples: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_scanline_backward_samples_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_scanline_backward_samples(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_scanline_backward_samples: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_atomic_backward_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_atomic_backward(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_atomic_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_atomic_backward_gated_dispatch(
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
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_atomic_backward_gated(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, active_start, active_stop, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_atomic_backward_gated: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_atomic_feature_backward_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& grad_feature_image,
    const torch::Tensor& grad_alpha_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_atomic_feature_backward(
        ma, q_uvt, depth0, depth_beta, opacity, feature, grad_feature_image, grad_alpha_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_atomic_feature_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_atomic_feature_backward_with_bins_dispatch(
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
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_atomic_feature_backward_with_bins(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        grad_feature_image,
        grad_alpha_image,
        tile_counts,
        tile_tube_ids,
        tile_depths,
        tile_unstable,
        meta_i32,
        meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_atomic_feature_backward_with_bins: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
direct_atomic_feature_sparse_pixels_backward_with_bins_dispatch(
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
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_atomic_feature_sparse_pixels_backward_with_bins(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        pixel_ids,
        grad_feature_values,
        grad_alpha_values,
        tile_counts,
        tile_tube_ids,
        tile_depths,
        tile_unstable,
        meta_i32,
        meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_atomic_feature_sparse_pixels_backward_with_bins: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
direct_atomic_feature_sparse_hidden_sigmoid_mse_backward_with_bins_dispatch(
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
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_atomic_feature_sparse_hidden_sigmoid_mse_backward_with_bins(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        pixel_ids,
        target_rgb_values,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        tile_counts,
        tile_tube_ids,
        tile_depths,
        tile_unstable,
        meta_i32,
        meta_f32);
  }
#endif
  TORCH_CHECK(
      false,
      "star_uvt_v0.direct_atomic_feature_sparse_hidden_sigmoid_mse_backward_with_bins: no backend available for device ",
      ma.device());
}

std::tuple<torch::Tensor, torch::Tensor>
sparse_hidden_sigmoid_target_area_forward_sums_with_bins_dispatch(
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
    int64_t cell_count) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_sparse_hidden_sigmoid_target_area_forward_sums_with_bins(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        pixel_ids,
        cell_ids,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        tile_counts,
        tile_tube_ids,
        tile_depths,
        tile_unstable,
        meta_i32,
        meta_f32,
        cell_count);
  }
#endif
  TORCH_CHECK(
      false,
      "star_uvt_v0.sparse_hidden_sigmoid_target_area_forward_sums_with_bins: no backend available for device ",
      ma.device());
}

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
direct_atomic_feature_sparse_hidden_target_area_backward_with_bins_dispatch(
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
    int64_t mode_bits) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_atomic_feature_sparse_hidden_target_area_backward_with_bins(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        pixel_ids,
        cell_ids,
        cell_grad_rgb,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        tile_counts,
        tile_tube_ids,
        tile_depths,
        tile_unstable,
        meta_i32,
        meta_f32,
        mode_bits);
  }
#endif
  TORCH_CHECK(
      false,
      "star_uvt_v0.direct_atomic_feature_sparse_hidden_target_area_backward_with_bins: no backend available for device ",
      ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
render_feature_sparse_pixels_with_bins_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& feature,
    const torch::Tensor& pixel_ids,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_render_feature_sparse_pixels_with_bins(
        ma, q_uvt, depth0, depth_beta, opacity, feature, pixel_ids, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.render_feature_sparse_pixels_with_bins: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
direct_atomic_feature_linear_sigmoid_mse_backward_dispatch(
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
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_atomic_feature_linear_sigmoid_mse_backward(
        ma, q_uvt, depth0, depth_beta, opacity, feature, target_rgb, color_weight, color_bias, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_atomic_feature_linear_sigmoid_mse_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor>
linear_sigmoid_mse_handoff_prep_dispatch(
    const torch::Tensor& feature_image,
    const torch::Tensor& alpha,
    const torch::Tensor& target_rgb,
    const torch::Tensor& color_weight,
    const torch::Tensor& color_bias) {
#if defined(__APPLE__)
  if (feature_image.device().is_mps()) {
    return metal_linear_sigmoid_mse_handoff_prep(feature_image, alpha, target_rgb, color_weight, color_bias);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.linear_sigmoid_mse_handoff_prep: no backend available for device ", feature_image.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
direct_atomic_feature_hidden_sigmoid_mse_backward_dispatch(
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
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_atomic_feature_hidden_sigmoid_mse_backward(
        ma, q_uvt, depth0, depth_beta, opacity, feature, target_rgb, hidden_weight, hidden_bias, output_weight,
        output_bias, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_atomic_feature_hidden_sigmoid_mse_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
direct_atomic_feature_logit_handoff_backward_dispatch(
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
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_atomic_feature_logit_handoff_backward(
        ma, q_uvt, depth0, depth_beta, opacity, feature, grad_logits, grad_alpha_image, color_weight, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_atomic_feature_logit_handoff_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_fixedpoint_backward_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_fixedpoint_backward(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_fixedpoint_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_atomic_backward_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_atomic_backward(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_atomic_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_fixedpoint_backward_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_fixedpoint_backward(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_fixedpoint_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_split_fixedpoint_backward_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_split_fixedpoint_backward(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_split_fixedpoint_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> direct_serial_backward_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_direct_serial_backward(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.direct_serial_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> reduce_sample_bundle_scan_dispatch(
    const torch::Tensor& ids,
    const torch::Tensor& grad_ma_samples,
    const torch::Tensor& grad_q_samples,
    const torch::Tensor& grad_opacity_samples,
    const torch::Tensor& grad_color_samples,
    int64_t tube_count) {
#if defined(__APPLE__)
  if (ids.device().is_mps()) {
    return metal_reduce_sample_bundle_scan(ids, grad_ma_samples, grad_q_samples, grad_opacity_samples, grad_color_samples, tube_count);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.reduce_sample_bundle_scan: no backend available for device ", ids.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> reduce_sample_bundle_scan_compensated_dispatch(
    const torch::Tensor& ids,
    const torch::Tensor& grad_ma_samples,
    const torch::Tensor& grad_q_samples,
    const torch::Tensor& grad_opacity_samples,
    const torch::Tensor& grad_color_samples,
    int64_t tube_count) {
#if defined(__APPLE__)
  if (ids.device().is_mps()) {
    return metal_reduce_sample_bundle_scan_compensated(
        ids,
        grad_ma_samples,
        grad_q_samples,
        grad_opacity_samples,
        grad_color_samples,
        tube_count);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.reduce_sample_bundle_scan_compensated: no backend available for device ", ids.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> reduce_sample_bundle_sorted_segments_dispatch(
    const torch::Tensor& ids,
    const torch::Tensor& grad_ma_samples,
    const torch::Tensor& grad_q_samples,
    const torch::Tensor& grad_opacity_samples,
    const torch::Tensor& grad_color_samples,
    int64_t tube_count) {
#if defined(__APPLE__)
  if (ids.device().is_mps()) {
    return metal_reduce_sample_bundle_sorted_segments(
        ids,
        grad_ma_samples,
        grad_q_samples,
        grad_opacity_samples,
        grad_color_samples,
        tube_count);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.reduce_sample_bundle_sorted_segments: no backend available for device ", ids.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_reduced_backward_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_reduced_backward(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_reduced_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_reduced_parallel_backward_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_reduced_parallel_backward(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_reduced_parallel_backward: no backend available for device ", ma.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tile_pair_suffix_reduced_backward_dispatch(
    const torch::Tensor& ma,
    const torch::Tensor& q_uvt,
    const torch::Tensor& depth0,
    const torch::Tensor& depth_beta,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (ma.device().is_mps()) {
    return metal_tile_pair_suffix_reduced_backward(ma, q_uvt, depth0, depth_beta, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_v0.tile_pair_suffix_reduced_backward: no backend available for device ", ma.device());
}

}  // namespace
}  // namespace star_uvt

TORCH_LIBRARY(star_uvt_v0, m) {
  m.def("render(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("render_gated(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor active_start, Tensor active_stop, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("projective_trace_eval(Tensor coeffs, Tensor times, float eps) -> Tensor");
  m.def("projective_trace_family_eval(Tensor family_coeffs, Tensor q_basis, Tensor times, float eps) -> Tensor");
  m.def("projective_trace_family_backward(Tensor family_coeffs, Tensor q_basis, Tensor times, Tensor grad_out, float eps) -> (Tensor, Tensor)");
  m.def("render_projective_trace_tiles(Tensor coeffs, Tensor times, Tensor opacity, Tensor color, Tensor tile_counts, Tensor tile_primitive_ids, Tensor tile_active_start, Tensor tile_active_stop, Tensor meta_i32, Tensor meta_f32, float sigma_px) -> Tensor");
  m.def("render_projective_trace_cell_tiles(Tensor coeffs, Tensor times, Tensor opacity, Tensor color, Tensor tile_counts, Tensor tile_trace_ids, Tensor tile_active_start, Tensor tile_active_stop, Tensor meta_i32, Tensor meta_f32, float sigma_px) -> Tensor");
  m.def("render_projective_trace_cell_interval_tiles(Tensor coeffs, Tensor times, Tensor opacity, Tensor opacity_time_coeffs, Tensor spatial_precision_uv, Tensor depth_affine_uv, Tensor color, Tensor tile_counts, Tensor tile_trace_ids, Tensor tile_active_start, Tensor tile_active_stop, Tensor meta_i32, Tensor meta_f32, float sigma_px) -> Tensor");
  m.def("render_projective_trace_family_interval_tiles(Tensor family_coeffs, Tensor q_basis, Tensor times, Tensor opacity, Tensor opacity_time_coeffs, Tensor spatial_precision_uv, Tensor depth_affine_uv, Tensor color, Tensor tile_counts, Tensor tile_trace_ids, Tensor tile_active_start, Tensor tile_active_stop, Tensor meta_i32, Tensor meta_f32, float sigma_px) -> Tensor");
  m.def("direct_projective_trace_family_interval_backward(Tensor family_coeffs, Tensor q_basis, Tensor times, Tensor opacity, Tensor opacity_time_coeffs, Tensor spatial_precision_uv, Tensor depth_affine_uv, Tensor color, Tensor grad_image, Tensor tile_counts, Tensor tile_trace_ids, Tensor tile_active_start, Tensor tile_active_stop, Tensor meta_i32, Tensor meta_f32, float sigma_px) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("render_projective_trace_cell_interval_rows(Tensor coeffs, Tensor times, Tensor opacity, Tensor opacity_time_coeffs, Tensor spatial_precision_uv, Tensor depth_affine_uv, Tensor color, Tensor tile_counts, Tensor tile_trace_ids, Tensor tile_active_start, Tensor tile_active_stop, Tensor row_weights, Tensor meta_i32, Tensor meta_f32, float sigma_px) -> Tensor");
  m.def("direct_projective_trace_backward(Tensor coeffs, Tensor times, Tensor opacity, Tensor color, Tensor grad_image, Tensor tile_counts, Tensor tile_primitive_ids, Tensor tile_active_start, Tensor tile_active_stop, Tensor meta_i32, Tensor meta_f32, float sigma_px) -> (Tensor, Tensor, Tensor)");
  m.def("direct_projective_trace_cell_interval_backward(Tensor coeffs, Tensor times, Tensor opacity, Tensor opacity_time_coeffs, Tensor spatial_precision_uv, Tensor depth_affine_uv, Tensor color, Tensor grad_image, Tensor tile_counts, Tensor tile_trace_ids, Tensor tile_active_start, Tensor tile_active_stop, Tensor meta_i32, Tensor meta_f32, float sigma_px) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("render_features(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("render_features_with_bins(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("bin_feature_tubes(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("simple_backward_samples(Tensor ma, Tensor q_uvt, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("stable_backward_samples(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("stable_backward_samples_with_keys(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_backward_samples(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_backward_samples_compensated(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_target_bounds_backward_samples(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_suffix_backward_samples(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_parallel_backward_samples(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_grouped_backward_samples(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_sharedsort_backward_samples(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_scanline_backward_samples(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_atomic_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_atomic_backward_gated(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor active_start, Tensor active_stop, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_atomic_feature_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor grad_feature_image, Tensor grad_alpha_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_atomic_feature_backward_with_bins(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor grad_feature_image, Tensor grad_alpha_image, Tensor tile_counts, Tensor tile_tube_ids, Tensor tile_depths, Tensor tile_unstable, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_atomic_feature_sparse_pixels_backward_with_bins(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor pixel_ids, Tensor grad_feature_values, Tensor grad_alpha_values, Tensor tile_counts, Tensor tile_tube_ids, Tensor tile_depths, Tensor tile_unstable, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_atomic_feature_sparse_hidden_sigmoid_mse_backward_with_bins(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor pixel_ids, Tensor target_rgb_values, Tensor hidden_weight, Tensor hidden_bias, Tensor output_weight, Tensor output_bias, Tensor tile_counts, Tensor tile_tube_ids, Tensor tile_depths, Tensor tile_unstable, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("sparse_hidden_sigmoid_target_area_forward_sums_with_bins(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor pixel_ids, Tensor cell_ids, Tensor hidden_weight, Tensor hidden_bias, Tensor output_weight, Tensor output_bias, Tensor tile_counts, Tensor tile_tube_ids, Tensor tile_depths, Tensor tile_unstable, Tensor meta_i32, Tensor meta_f32, int cell_count) -> (Tensor, Tensor)");
  m.def("direct_atomic_feature_sparse_hidden_target_area_backward_with_bins(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor pixel_ids, Tensor cell_ids, Tensor cell_grad_rgb, Tensor hidden_weight, Tensor hidden_bias, Tensor output_weight, Tensor output_bias, Tensor tile_counts, Tensor tile_tube_ids, Tensor tile_depths, Tensor tile_unstable, Tensor meta_i32, Tensor meta_f32, int mode_bits) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("render_feature_sparse_pixels_with_bins(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor pixel_ids, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_atomic_feature_linear_sigmoid_mse_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor target_rgb, Tensor color_weight, Tensor color_bias, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("linear_sigmoid_mse_handoff_prep(Tensor feature_image, Tensor alpha, Tensor target_rgb, Tensor color_weight, Tensor color_bias) -> (Tensor, Tensor)");
  m.def("direct_atomic_feature_hidden_sigmoid_mse_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor target_rgb, Tensor hidden_weight, Tensor hidden_bias, Tensor output_weight, Tensor output_bias, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_atomic_feature_logit_handoff_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor feature, Tensor grad_logits, Tensor grad_alpha_image, Tensor color_weight, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_fixedpoint_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_atomic_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_fixedpoint_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_split_fixedpoint_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("direct_serial_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("reduce_sample_bundle_scan(Tensor ids, Tensor grad_ma_samples, Tensor grad_q_samples, Tensor grad_opacity_samples, Tensor grad_color_samples, int tube_count) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("reduce_sample_bundle_scan_compensated(Tensor ids, Tensor grad_ma_samples, Tensor grad_q_samples, Tensor grad_opacity_samples, Tensor grad_color_samples, int tube_count) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("reduce_sample_bundle_sorted_segments(Tensor ids, Tensor grad_ma_samples, Tensor grad_q_samples, Tensor grad_opacity_samples, Tensor grad_color_samples, int tube_count) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_reduced_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_reduced_parallel_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("tile_pair_suffix_reduced_backward(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(star_uvt_v0, CompositeExplicitAutograd, m) {
  m.impl("render", star_uvt::render_dispatch);
  m.impl("render_gated", star_uvt::render_gated_dispatch);
  m.impl("projective_trace_eval", star_uvt::projective_trace_eval_dispatch);
  m.impl("projective_trace_family_eval", star_uvt::projective_trace_family_eval_dispatch);
  m.impl("projective_trace_family_backward", star_uvt::projective_trace_family_backward_dispatch);
  m.impl("render_projective_trace_tiles", star_uvt::render_projective_trace_tiles_dispatch);
  m.impl("render_projective_trace_cell_tiles", star_uvt::render_projective_trace_cell_tiles_dispatch);
  m.impl("render_projective_trace_cell_interval_tiles", star_uvt::render_projective_trace_cell_interval_tiles_dispatch);
  m.impl("render_projective_trace_family_interval_tiles", star_uvt::render_projective_trace_family_interval_tiles_dispatch);
  m.impl("direct_projective_trace_family_interval_backward", star_uvt::direct_projective_trace_family_interval_backward_dispatch);
  m.impl("render_projective_trace_cell_interval_rows", star_uvt::render_projective_trace_cell_interval_rows_dispatch);
  m.impl("direct_projective_trace_backward", star_uvt::direct_projective_trace_backward_dispatch);
  m.impl("direct_projective_trace_cell_interval_backward", star_uvt::direct_projective_trace_cell_interval_backward_dispatch);
  m.impl("render_features", star_uvt::render_features_dispatch);
  m.impl("render_features_with_bins", star_uvt::render_features_with_bins_dispatch);
  m.impl("bin_feature_tubes", star_uvt::bin_feature_tubes_dispatch);
  m.impl("simple_backward_samples", star_uvt::simple_backward_samples_dispatch);
  m.impl("stable_backward_samples", star_uvt::stable_backward_samples_dispatch);
  m.impl("stable_backward_samples_with_keys", star_uvt::stable_backward_samples_with_keys_dispatch);
  m.impl("tile_pair_backward_samples", star_uvt::tile_pair_backward_samples_dispatch);
  m.impl("tile_pair_backward_samples_compensated", star_uvt::tile_pair_backward_samples_compensated_dispatch);
  m.impl("tile_pair_target_bounds_backward_samples", star_uvt::tile_pair_target_bounds_backward_samples_dispatch);
  m.impl("tile_pair_suffix_backward_samples", star_uvt::tile_pair_suffix_backward_samples_dispatch);
  m.impl("tile_pair_parallel_backward_samples", star_uvt::tile_pair_parallel_backward_samples_dispatch);
  m.impl("tile_pair_grouped_backward_samples", star_uvt::tile_pair_grouped_backward_samples_dispatch);
  m.impl("tile_pair_sharedsort_backward_samples", star_uvt::tile_pair_sharedsort_backward_samples_dispatch);
  m.impl("tile_pair_scanline_backward_samples", star_uvt::tile_pair_scanline_backward_samples_dispatch);
  m.impl("direct_atomic_backward", star_uvt::direct_atomic_backward_dispatch);
  m.impl("direct_atomic_backward_gated", star_uvt::direct_atomic_backward_gated_dispatch);
  m.impl("direct_atomic_feature_backward", star_uvt::direct_atomic_feature_backward_dispatch);
  m.impl("direct_atomic_feature_backward_with_bins", star_uvt::direct_atomic_feature_backward_with_bins_dispatch);
  m.impl("direct_atomic_feature_sparse_pixels_backward_with_bins", star_uvt::direct_atomic_feature_sparse_pixels_backward_with_bins_dispatch);
  m.impl(
      "direct_atomic_feature_sparse_hidden_sigmoid_mse_backward_with_bins",
      star_uvt::direct_atomic_feature_sparse_hidden_sigmoid_mse_backward_with_bins_dispatch);
  m.impl(
      "sparse_hidden_sigmoid_target_area_forward_sums_with_bins",
      star_uvt::sparse_hidden_sigmoid_target_area_forward_sums_with_bins_dispatch);
  m.impl(
      "direct_atomic_feature_sparse_hidden_target_area_backward_with_bins",
      star_uvt::direct_atomic_feature_sparse_hidden_target_area_backward_with_bins_dispatch);
  m.impl("render_feature_sparse_pixels_with_bins", star_uvt::render_feature_sparse_pixels_with_bins_dispatch);
  m.impl("direct_atomic_feature_linear_sigmoid_mse_backward", star_uvt::direct_atomic_feature_linear_sigmoid_mse_backward_dispatch);
  m.impl("linear_sigmoid_mse_handoff_prep", star_uvt::linear_sigmoid_mse_handoff_prep_dispatch);
  m.impl("direct_atomic_feature_hidden_sigmoid_mse_backward", star_uvt::direct_atomic_feature_hidden_sigmoid_mse_backward_dispatch);
  m.impl("direct_atomic_feature_logit_handoff_backward", star_uvt::direct_atomic_feature_logit_handoff_backward_dispatch);
  m.impl("direct_fixedpoint_backward", star_uvt::direct_fixedpoint_backward_dispatch);
  m.impl("tile_pair_atomic_backward", star_uvt::tile_pair_atomic_backward_dispatch);
  m.impl("tile_pair_fixedpoint_backward", star_uvt::tile_pair_fixedpoint_backward_dispatch);
  m.impl("direct_split_fixedpoint_backward", star_uvt::direct_split_fixedpoint_backward_dispatch);
  m.impl("direct_serial_backward", star_uvt::direct_serial_backward_dispatch);
  m.impl("reduce_sample_bundle_scan", star_uvt::reduce_sample_bundle_scan_dispatch);
  m.impl("reduce_sample_bundle_scan_compensated", star_uvt::reduce_sample_bundle_scan_compensated_dispatch);
  m.impl("reduce_sample_bundle_sorted_segments", star_uvt::reduce_sample_bundle_sorted_segments_dispatch);
  m.impl("tile_pair_reduced_backward", star_uvt::tile_pair_reduced_backward_dispatch);
  m.impl("tile_pair_reduced_parallel_backward", star_uvt::tile_pair_reduced_parallel_backward_dispatch);
  m.impl("tile_pair_suffix_reduced_backward", star_uvt::tile_pair_suffix_reduced_backward_dispatch);
}
