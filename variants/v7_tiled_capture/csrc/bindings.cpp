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
  TORCH_CHECK(false, "gsplat_metal_v72.forward: no backend available for device ", means2d.device());
}

BackwardOut backward_dispatch(
    const torch::Tensor& grad_out,
    const torch::Tensor& packed_gaussians,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& out_cpu,
    const torch::Tensor& front_ids,
    const torch::Tensor& front_raw_alpha,
    const torch::Tensor& front_meta,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_ids) {
#if defined(__APPLE__)
  if (grad_out.device().is_mps()) {
    return metal_backward(
        grad_out,
        packed_gaussians,
        meta_i32,
        meta_f32,
        out_cpu,
        front_ids,
        front_raw_alpha,
        front_meta,
        tile_offsets,
        tile_ids);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v72.backward: no backend available for device ", grad_out.device());
}

}  // namespace
}  // namespace gsplat

TORCH_LIBRARY(gsplat_metal_v72, m) {
  m.def("forward(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor depths, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("backward(Tensor grad_out, Tensor packed_gaussians, Tensor meta_i32, Tensor meta_f32, Tensor out_cpu, Tensor front_ids, Tensor front_raw_alpha, Tensor front_meta, Tensor tile_offsets, Tensor tile_ids) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gsplat_metal_v72, CompositeExplicitAutograd, m) {
  m.impl("forward", gsplat::forward_dispatch);
  m.impl("backward", gsplat::backward_dispatch);
}
