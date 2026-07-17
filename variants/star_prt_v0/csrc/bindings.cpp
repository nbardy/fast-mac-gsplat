#include <torch/extension.h>

#include "shared/common.h"

namespace star_prt {
namespace {

torch::Tensor render_projective_rational_tubes_dispatch(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_projective_rational_contract(h_coeff, lambda_uv, lambda_t, center_t, opacity, color, meta_i32, meta_f32);
#if defined(__APPLE__)
  if (h_coeff.device().is_mps()) {
    return metal_render_projective_rational_tubes(
        h_coeff, lambda_uv, lambda_t, center_t, opacity, color, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false,
              "star_prt_v0.render_projective_rational_tubes has no compiled backend for device ",
              h_coeff.device(),
              "; use torch_gsplat_bridge_star_prt.render_projective_rational_tubes(..., backend='dense')");
}

torch::Tensor render_compiled_curve_tubes_dispatch(
    const torch::Tensor& curve_uv_depth,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_compiled_curve_contract(curve_uv_depth, lambda_uv, lambda_t, center_t, opacity, color, meta_i32, meta_f32);
#if defined(__APPLE__)
  if (curve_uv_depth.device().is_mps()) {
    return metal_render_compiled_curve_tubes(
        curve_uv_depth, lambda_uv, lambda_t, center_t, opacity, color, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false,
              "star_prt_v0.render_compiled_curve_tubes has no compiled backend for device ",
              curve_uv_depth.device(),
              "; use torch_gsplat_bridge_star_prt.render_compiled_curve_tubes(..., backend='dense')");
}

PRTBackwardOutputs compact_backward_projective_rational_tubes_dispatch(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& grad_image,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_projective_rational_backward_contract(
      h_coeff, lambda_uv, lambda_t, center_t, opacity, color, grad_image, meta_i32, meta_f32);
#if defined(__APPLE__)
  if (h_coeff.device().is_mps()) {
    return metal_compact_backward_projective_rational_tubes(
        h_coeff, lambda_uv, lambda_t, center_t, opacity, color, grad_image, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false,
              "star_prt_v0.compact_backward_projective_rational_tubes has no compiled backend for device ",
              h_coeff.device(),
              "; compact backward is scaffolded but not implemented");
}

}  // namespace
}  // namespace star_prt

TORCH_LIBRARY(star_prt_v0, m) {
  m.def("render_projective_rational_tubes(Tensor h_coeff, Tensor lambda_uv, Tensor lambda_t, Tensor center_t, Tensor opacity, Tensor color, Tensor meta_i32, Tensor meta_f32) -> Tensor");
  m.def("render_compiled_curve_tubes(Tensor curve_uv_depth, Tensor lambda_uv, Tensor lambda_t, Tensor center_t, Tensor opacity, Tensor color, Tensor meta_i32, Tensor meta_f32) -> Tensor");
  m.def("compact_backward_projective_rational_tubes(Tensor h_coeff, Tensor lambda_uv, Tensor lambda_t, Tensor center_t, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(star_prt_v0, CompositeExplicitAutograd, m) {
  m.impl("render_projective_rational_tubes", star_prt::render_projective_rational_tubes_dispatch);
  m.impl("render_compiled_curve_tubes", star_prt::render_compiled_curve_tubes_dispatch);
  m.impl("compact_backward_projective_rational_tubes", star_prt::compact_backward_projective_rational_tubes_dispatch);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
