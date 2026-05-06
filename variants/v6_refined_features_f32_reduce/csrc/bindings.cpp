#include <torch/extension.h>
#include "shared/common.h"

namespace gsplat {
namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_bin(means2d, conics, colors, opacities, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v6_refined_features_f32_reduce.bin: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor> render_fast_forward_eval_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_fast_forward_eval(means2d, conics, colors, opacities, meta_i32, meta_f32, tile_counts, tile_offsets, binned_ids);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v6_refined_features_f32_reduce.render_fast_forward_eval: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> render_fast_forward_state_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    torch::Tensor binned_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_fast_forward_state(means2d, conics, colors, opacities, meta_i32, meta_f32, binned_ids, tile_counts, tile_offsets);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v6_refined_features_f32_reduce.render_fast_forward_state: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_fast_backward_saved_dispatch(
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
    const torch::Tensor& tile_stop_counts) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_fast_backward_saved(
        grad_features, grad_alpha, means2d, conics, colors, opacities, meta_i32, meta_f32, tile_counts, tile_offsets, binned_ids, tile_stop_counts);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v6_refined_features_f32_reduce.render_fast_backward_saved: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor> render_active_forward_eval_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_active_forward_eval(
        means2d, conics, colors, opacities, meta_i32, meta_f32, active_tile_ids, tile_counts, tile_offsets, binned_ids);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v6_refined_features_f32_reduce.render_active_forward_eval: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> render_active_forward_state_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    torch::Tensor binned_ids,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_active_forward_state(
        means2d, conics, colors, opacities, meta_i32, meta_f32, binned_ids, active_tile_ids, tile_counts, tile_offsets);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v6_refined_features_f32_reduce.render_active_forward_state: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_active_backward_saved_dispatch(
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
    const torch::Tensor& tile_stop_counts) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_active_backward_saved(
        grad_features,
        grad_alpha,
        means2d,
        conics,
        colors,
        opacities,
        meta_i32,
        meta_f32,
        active_tile_ids,
        tile_counts,
        tile_offsets,
        binned_ids,
        tile_stop_counts);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v6_refined_features_f32_reduce.render_active_backward_saved: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor> render_overflow_forward_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& overflow_tile_ids,
    const torch::Tensor& overflow_tile_offsets,
    const torch::Tensor& overflow_sorted_ids) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_overflow_forward(
        means2d, conics, colors, opacities, meta_i32, meta_f32, overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v6_refined_features_f32_reduce.render_overflow_forward: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_overflow_backward_dispatch(
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
    const torch::Tensor& overflow_sorted_ids) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_overflow_backward(
        grad_features_tiles, grad_alpha_tiles, means2d, conics, colors, opacities, meta_i32, meta_f32, overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v6_refined_features_f32_reduce.render_overflow_backward: no backend available for device ", means2d.device());
}

}  // namespace
}  // namespace gsplat

TORCH_LIBRARY(gsplat_metal_v6_refined_features_f32_reduce, m) {
  m.def("bin(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor)");
  m.def("render_fast_forward_eval(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids) -> (Tensor, Tensor)");
  m.def("render_fast_forward_state(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor(a!) binned_ids, Tensor tile_counts, Tensor tile_offsets) -> (Tensor, Tensor, Tensor)");
  m.def("render_fast_backward_saved(Tensor grad_features, Tensor grad_alpha, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids, Tensor tile_stop_counts) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("render_active_forward_eval(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor active_tile_ids, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids) -> (Tensor, Tensor)");
  m.def("render_active_forward_state(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor(a!) binned_ids, Tensor active_tile_ids, Tensor tile_counts, Tensor tile_offsets) -> (Tensor, Tensor, Tensor)");
  m.def("render_active_backward_saved(Tensor grad_features, Tensor grad_alpha, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor active_tile_ids, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids, Tensor tile_stop_counts) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("render_overflow_forward(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor overflow_tile_ids, Tensor overflow_tile_offsets, Tensor overflow_sorted_ids) -> (Tensor, Tensor)");
  m.def("render_overflow_backward(Tensor grad_features_tiles, Tensor grad_alpha_tiles, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor overflow_tile_ids, Tensor overflow_tile_offsets, Tensor overflow_sorted_ids) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gsplat_metal_v6_refined_features_f32_reduce, CompositeExplicitAutograd, m) {
  m.impl("bin", gsplat::bin_dispatch);
  m.impl("render_fast_forward_eval", gsplat::render_fast_forward_eval_dispatch);
  m.impl("render_fast_forward_state", gsplat::render_fast_forward_state_dispatch);
  m.impl("render_fast_backward_saved", gsplat::render_fast_backward_saved_dispatch);
  m.impl("render_active_forward_eval", gsplat::render_active_forward_eval_dispatch);
  m.impl("render_active_forward_state", gsplat::render_active_forward_state_dispatch);
  m.impl("render_active_backward_saved", gsplat::render_active_backward_saved_dispatch);
  m.impl("render_overflow_forward", gsplat::render_overflow_forward_dispatch);
  m.impl("render_overflow_backward", gsplat::render_overflow_backward_dispatch);
}
