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

}  // namespace
}  // namespace star_uvt

TORCH_LIBRARY(star_uvt_v0, m) {
  m.def("render(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("simple_backward_samples(Tensor ma, Tensor q_uvt, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("stable_backward_samples(Tensor ma, Tensor q_uvt, Tensor depth0, Tensor depth_beta, Tensor opacity, Tensor color, Tensor grad_image, Tensor meta_i32, Tensor meta_f32) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(star_uvt_v0, CompositeExplicitAutograd, m) {
  m.impl("render", star_uvt::render_dispatch);
  m.impl("simple_backward_samples", star_uvt::simple_backward_samples_dispatch);
  m.impl("stable_backward_samples", star_uvt::stable_backward_samples_dispatch);
}
