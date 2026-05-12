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
  TORCH_CHECK(false, "star_uvt_prt_v0.render: no backend available for device ", ma.device());
}

torch::Tensor render_projective_rational_direct_dispatch(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (h_coeff.device().is_mps()) {
    return metal_render_projective_rational_direct(
        h_coeff, lambda_uv, lambda_t, center_t, opacity, color, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_prt_v0.render_projective_rational_direct: no backend available for device ",
              h_coeff.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_projective_rational_tiled_dispatch(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (h_coeff.device().is_mps()) {
    return metal_render_projective_rational_tiled(
        h_coeff, lambda_uv, lambda_t, center_t, opacity, color, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_prt_v0.render_projective_rational_tiled: no backend available for device ",
              h_coeff.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
projective_rational_direct_serial_backward_dispatch(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (h_coeff.device().is_mps()) {
    return metal_projective_rational_direct_serial_backward(
        h_coeff, lambda_uv, lambda_t, center_t, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "star_uvt_prt_v0.projective_rational_direct_serial_backward: no backend available for device ",
              h_coeff.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.simple_backward_samples: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.stable_backward_samples: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.stable_backward_samples_with_keys: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_backward_samples: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_backward_samples_compensated: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_target_bounds_backward_samples: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_suffix_backward_samples: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_parallel_backward_samples: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_grouped_backward_samples: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_sharedsort_backward_samples: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_scanline_backward_samples: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.direct_atomic_backward: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.direct_fixedpoint_backward: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_atomic_backward: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_fixedpoint_backward: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.direct_split_fixedpoint_backward: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.direct_serial_backward: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.reduce_sample_bundle_scan: no backend available for device ", ids.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.reduce_sample_bundle_scan_compensated: no backend available for device ", ids.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.reduce_sample_bundle_sorted_segments: no backend available for device ", ids.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_reduced_backward: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_reduced_parallel_backward: no backend available for device ", ma.device());
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
  TORCH_CHECK(false, "star_uvt_prt_v0.tile_pair_suffix_reduced_backward: no backend available for device ", ma.device());
}

}  // namespace
}  // namespace star_uvt

TORCH_LIBRARY(star_uvt_prt_v0, m) {
  m.def("render(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("render_projective_rational_direct(Tensor h_coeff, Tensor lambda_uv, Tensor lambda_t, Tensor center_t, Tensor opacity, Tensor color, Tensor meta_i32, Tensor meta_f32) -> Tensor");
  m.def("render_projective_rational_tiled(Tensor h_coeff, Tensor lambda_uv, Tensor lambda_t, Tensor center_t, Tensor opacity, Tensor color, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("projective_rational_direct_serial_backward(Tensor h_coeff, Tensor lambda_uv, Tensor lambda_t, Tensor center_t, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
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

TORCH_LIBRARY_IMPL(star_uvt_prt_v0, CompositeExplicitAutograd, m) {
  m.impl("render", star_uvt::render_dispatch);
  m.impl("render_projective_rational_direct", star_uvt::render_projective_rational_direct_dispatch);
  m.impl("render_projective_rational_tiled", star_uvt::render_projective_rational_tiled_dispatch);
  m.impl("projective_rational_direct_serial_backward", star_uvt::projective_rational_direct_serial_backward_dispatch);
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
