#include "v9_cuda_contract.cuh"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

v9cuda::RasterParams make_params(
    int batch,
    int cameras,
    int gaussians,
    int height,
    int width,
    float alpha_threshold,
    float transmittance_threshold,
    float max_alpha) {
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  TORCH_CHECK(gaussians >= 0, "gaussians must be non-negative");
  v9cuda::RasterParams params;
  params.batch = batch;
  params.cameras = cameras;
  params.gaussians = gaussians;
  params.height = height;
  params.width = width;
  params.tile_width = (width + v9cuda::kTileSize - 1) / v9cuda::kTileSize;
  params.tile_height = (height + v9cuda::kTileSize - 1) / v9cuda::kTileSize;
  params.alpha_threshold = alpha_threshold;
  params.transmittance_threshold = transmittance_threshold;
  params.max_alpha = max_alpha;
  return params;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("make_params", &make_params, "Construct V9 CUDA raster params");
  m.def("project_count_fused", &v9cuda::project_count_fused_cuda,
        "Project 3D Gaussians, build conics, and count tile intersections");
  m.def("emit_pairs", &v9cuda::emit_pairs_cuda,
        "Emit [image|tile|depth] keys and Gaussian ids into fixed-capacity buffers");
  m.def("tile_forward_train", &v9cuda::tile_forward_train_cuda,
        "Exact one-block-per-tile front-to-back C/T forward");
  m.def("tile_backward_replay", &v9cuda::tile_backward_replay_cuda,
        "Exact reverse replay backward with block/warp reduction boundary");
}

