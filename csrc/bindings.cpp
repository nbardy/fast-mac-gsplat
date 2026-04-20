#include <torch/extension.h>
#include "shared/common.h"

namespace gsplat {
namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_forward(means2d, conics, colors, opacities, meta_i32, meta_f32);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_fast.forward: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> backward_dispatch(
    const torch::Tensor& grad_out,
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
    return metal_backward(grad_out, means2d, conics, colors, opacities, meta_i32, meta_f32, tile_counts, tile_offsets, binned_ids);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_fast.backward: no backend available for device ", means2d.device());
}

}  // namespace
}  // namespace gsplat

TORCH_LIBRARY(gsplat_metal_fast, m) {
  m.def("forward(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("backward(Tensor grad_out, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gsplat_metal_fast, CompositeExplicitAutograd, m) {
  m.impl("forward", gsplat::forward_dispatch);
  m.impl("backward", gsplat::backward_dispatch);
}
