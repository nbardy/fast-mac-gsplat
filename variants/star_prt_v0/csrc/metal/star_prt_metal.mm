#include "shared/common.h"

namespace star_prt {

torch::Tensor metal_render_projective_rational_tubes(
    const torch::Tensor& h_coeff,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_projective_rational_contract(h_coeff, lambda_uv, lambda_t, center_t, opacity, color, meta_i32, meta_f32);
  TORCH_CHECK(false,
              "star_prt_v0 Metal render_projective_rational_tubes is a scaffold only; "
              "use the Python dense fallback until the Metal kernels are implemented");
}

torch::Tensor metal_render_compiled_curve_tubes(
    const torch::Tensor& curve_uv_depth,
    const torch::Tensor& lambda_uv,
    const torch::Tensor& lambda_t,
    const torch::Tensor& center_t,
    const torch::Tensor& opacity,
    const torch::Tensor& color,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
  check_compiled_curve_contract(curve_uv_depth, lambda_uv, lambda_t, center_t, opacity, color, meta_i32, meta_f32);
  TORCH_CHECK(false,
              "star_prt_v0 Metal render_compiled_curve_tubes is a scaffold only; "
              "use the Python dense fallback until the Metal kernels are implemented");
}

PRTBackwardOutputs metal_compact_backward_projective_rational_tubes(
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
  TORCH_CHECK(false,
              "star_prt_v0 Metal compact_backward_projective_rational_tubes is a scaffold only; "
              "no compact PRT backward parity is implemented yet");
}

}  // namespace star_prt
