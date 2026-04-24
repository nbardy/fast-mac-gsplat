#include <torch/extension.h>
#include "shared/common.h"

namespace gsplat {
namespace {

ForwardOut forward_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_forward(means2d, conics, colors, opacities, depths, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v71.forward: no backend available for device ", means2d.device());
}

BackwardOut backward_dispatch(
    const torch::Tensor& grad_out,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& depths,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& out_image,
    const torch::Tensor& front_ids,
    const torch::Tensor& front_raw_alpha,
    const torch::Tensor& front_count,
    const torch::Tensor& overflow_mask) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_backward(
        grad_out,
        means2d,
        conics,
        colors,
        opacities,
        depths,
        meta_i32,
        meta_f32,
        out_image,
        front_ids,
        front_raw_alpha,
        front_count,
        overflow_mask);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v71.backward: no backend available for device ", means2d.device());
}

}  // namespace
}  // namespace gsplat

TORCH_LIBRARY(gsplat_metal_v71, m) {
  m.def("forward(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor depths, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("backward(Tensor grad_out, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor depths, Tensor meta_i32, Tensor meta_f32, Tensor out_image, Tensor front_ids, Tensor front_raw_alpha, Tensor front_count, Tensor overflow_mask) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gsplat_metal_v71, CompositeExplicitAutograd, m) {
  m.impl("forward", gsplat::forward_dispatch);
  m.impl("backward", gsplat::backward_dispatch);
}
