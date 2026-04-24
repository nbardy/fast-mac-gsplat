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
  TORCH_CHECK(false, "gsplat_metal_v7.forward: no backend available for device ", means2d.device());
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
    const torch::Tensor& aux) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_backward(grad_out, means2d, conics, colors, opacities, depths, meta_i32, meta_f32, aux);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v7.backward: no backend available for device ", means2d.device());
}

}  // namespace
}  // namespace gsplat

TORCH_LIBRARY(gsplat_metal_v7, m) {
  m.def("forward(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor depths, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor)");
  m.def("backward(Tensor grad_out, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor depths, Tensor meta_i32, Tensor meta_f32, Tensor aux) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gsplat_metal_v7, CompositeExplicitAutograd, m) {
  m.impl("forward", gsplat::forward_dispatch);
  m.impl("backward", gsplat::backward_dispatch);
}
