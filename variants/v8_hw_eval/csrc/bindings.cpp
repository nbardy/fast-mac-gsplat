
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "shared/common.h"

namespace py = pybind11;

namespace gsplat {

py::dict metal_probe_hardware_eval(bool compile_render_pipeline);

#if !defined(__APPLE__)
py::dict metal_probe_hardware_eval(bool compile_render_pipeline) {
  py::dict out;
  out["native_probe_available"] = true;
  out["compile_render_pipeline_requested"] = compile_render_pipeline;
  out["metal_available"] = false;
  out["metal_device_name"] = "";
  out["command_queue_created"] = false;
  out["render_pipeline_source_available"] = false;
  out["render_pipeline_compile_attempted"] = false;
  out["render_pipeline_library_compiled"] = py::none();
  out["render_pipeline_ready"] = py::none();
  out["render_pipeline_error"] = "Metal is only available on Apple platforms";
  out["no_cpu_readback"] = true;
  out["cpu_readback_path_present"] = false;
  out["render_pipeline_uses_cpu_readback"] = false;
  out["render_to_mps_interop"] = false;
  out["render_to_mps_interop_reason"] = "MPS/Torch texture interop is not implemented";
  out["imageblock_support_known"] = false;
  out["imageblock_supported"] = py::none();
  out["imageblock_support_reason"] = "not probed on non-Apple builds";
  out["raster_order_group_support_known"] = false;
  out["raster_order_group_supported"] = py::none();
  out["raster_order_group_support_reason"] = "not probed on non-Apple builds";
  return out;
}
#endif

namespace {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bin_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_bin(means2d, conics, colors, opacities, meta_i32, meta_f32, meta_host_i32, meta_host_f32);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v8_hw_eval.bin: no backend available for device ", means2d.device());
}

torch::Tensor render_active_forward_eval_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_active_forward_eval(means2d, conics, colors, opacities, meta_i32, meta_f32, meta_host_i32, meta_host_f32, active_tile_ids, tile_counts, tile_offsets, binned_ids);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v8_hw_eval.render_active_forward_eval: no backend available for device ", means2d.device());
}

torch::Tensor render_fast_forward_eval_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_fast_forward_eval(means2d, conics, colors, opacities, meta_i32, meta_f32, meta_host_i32, meta_host_f32, tile_counts, tile_offsets, binned_ids);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v8_hw_eval.render_fast_forward_eval: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor> render_fast_forward_state_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    torch::Tensor binned_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_fast_forward_state(means2d, conics, colors, opacities, meta_i32, meta_f32, meta_host_i32, meta_host_f32, binned_ids, tile_counts, tile_offsets);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v8_hw_eval.render_fast_forward_state: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_fast_backward_saved_dispatch(
    const torch::Tensor& grad_out,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids,
    const torch::Tensor& tile_stop_counts) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_fast_backward_saved(
        grad_out, means2d, conics, colors, opacities, meta_i32, meta_f32, meta_host_i32, meta_host_f32, tile_counts, tile_offsets, binned_ids, tile_stop_counts);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v8_hw_eval.render_fast_backward_saved: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor> render_active_forward_state_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    torch::Tensor binned_ids,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_active_forward_state(means2d, conics, colors, opacities, meta_i32, meta_f32, meta_host_i32, meta_host_f32, binned_ids, active_tile_ids, tile_counts, tile_offsets);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v8_hw_eval.render_active_forward_state: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_active_backward_saved_dispatch(
    const torch::Tensor& grad_out,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& active_tile_ids,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& binned_ids,
    const torch::Tensor& tile_stop_counts) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_active_backward_saved(
        grad_out, means2d, conics, colors, opacities, meta_i32, meta_f32, meta_host_i32, meta_host_f32, active_tile_ids, tile_counts, tile_offsets, binned_ids, tile_stop_counts);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v8_hw_eval.render_active_backward_saved: no backend available for device ", means2d.device());
}

torch::Tensor render_overflow_forward_dispatch(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& overflow_tile_ids,
    const torch::Tensor& overflow_tile_offsets,
    const torch::Tensor& overflow_sorted_ids) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_overflow_forward(
        means2d, conics, colors, opacities, meta_i32, meta_f32, meta_host_i32, meta_host_f32, overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v8_hw_eval.render_overflow_forward: no backend available for device ", means2d.device());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render_overflow_backward_dispatch(
    const torch::Tensor& grad_tiles,
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& meta_i32,
    const torch::Tensor& meta_f32,
    const torch::Tensor& meta_host_i32,
    const torch::Tensor& meta_host_f32,
    const torch::Tensor& overflow_tile_ids,
    const torch::Tensor& overflow_tile_offsets,
    const torch::Tensor& overflow_sorted_ids) {
#if defined(__APPLE__)
  if (means2d.device().is_mps()) {
    return metal_render_overflow_backward(
        grad_tiles, means2d, conics, colors, opacities, meta_i32, meta_f32, meta_host_i32, meta_host_f32, overflow_tile_ids, overflow_tile_offsets, overflow_sorted_ids);
  }
#endif
  TORCH_CHECK(false, "gsplat_metal_v8_hw_eval.render_overflow_backward: no backend available for device ", means2d.device());
}

}  // namespace
}  // namespace gsplat

TORCH_LIBRARY(gsplat_metal_v8_hw_eval, m) {
  m.def("bin(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor meta_host_i32, Tensor meta_host_f32) -> (Tensor, Tensor, Tensor)");
  m.def("render_fast_forward_eval(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor meta_host_i32, Tensor meta_host_f32, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids) -> Tensor");
  m.def("render_fast_forward_state(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor meta_host_i32, Tensor meta_host_f32, Tensor(a!) binned_ids, Tensor tile_counts, Tensor tile_offsets) -> (Tensor, Tensor)");
  m.def("render_fast_backward_saved(Tensor grad_out, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor meta_host_i32, Tensor meta_host_f32, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids, Tensor tile_stop_counts) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("render_active_forward_eval(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor meta_host_i32, Tensor meta_host_f32, Tensor active_tile_ids, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids) -> Tensor");
  m.def("render_active_forward_state(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor meta_host_i32, Tensor meta_host_f32, Tensor(a!) binned_ids, Tensor active_tile_ids, Tensor tile_counts, Tensor tile_offsets) -> (Tensor, Tensor)");
  m.def("render_active_backward_saved(Tensor grad_out, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor meta_host_i32, Tensor meta_host_f32, Tensor active_tile_ids, Tensor tile_counts, Tensor tile_offsets, Tensor binned_ids, Tensor tile_stop_counts) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("render_overflow_forward(Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor meta_host_i32, Tensor meta_host_f32, Tensor overflow_tile_ids, Tensor overflow_tile_offsets, Tensor overflow_sorted_ids) -> Tensor");
  m.def("render_overflow_backward(Tensor grad_tiles, Tensor means2d, Tensor conics, Tensor colors, Tensor opacities, Tensor meta_i32, Tensor meta_f32, Tensor meta_host_i32, Tensor meta_host_f32, Tensor overflow_tile_ids, Tensor overflow_tile_offsets, Tensor overflow_sorted_ids) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gsplat_metal_v8_hw_eval, CompositeExplicitAutograd, m) {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "probe_hardware_eval_native",
      &gsplat::metal_probe_hardware_eval,
      py::arg("compile_render_pipeline") = true,
      "Probe tensor-free Metal hardware-eval prerequisites.");
}
