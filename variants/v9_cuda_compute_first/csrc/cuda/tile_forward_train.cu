#include "v9_cuda_contract.cuh"

namespace v9cuda {

namespace {

__global__ void tile_forward_train_kernel() {
  // Skeleton contract:
  //
  // grid = [batch*camera, tile_y, tile_x]
  // block = [16, 16, 1] = one pixel lane per tile pixel
  //
  // Shared memory per CHUNK=256 sorted references:
  //   sh_ids[256]
  //   sh_mean[256]
  //   sh_conic[256]
  //   sh_opacity[256]
  //   sh_color[256]
  //
  // For each pixel lane:
  //   C = 0
  //   T = 1
  //   last_id = -1
  //
  // For each sorted tile chunk:
  //   all lanes participate in load/sync, even if a lane has stopped.
  //   for splat in chunk:
  //     dx, dy = pixel_center - mean2d
  //     q = a*dx*dx + 2*b*dx*dy + c*dy*dy
  //     power = -0.5*q
  //     raw = opacity * exp(power)
  //     alpha = min(max_alpha, raw)
  //     visible = power <= 0 && alpha >= alpha_threshold
  //     if visible and !done:
  //       C += T * alpha * color
  //       T *= (1 - alpha)
  //       last_id = global_ref_index
  //       done = T <= transmittance_threshold
  //
  // Outputs:
  //   out_rgb[pixel] = C + T * background
  //   out_alpha[pixel] = 1 - T
  //   last_ids[pixel] = last processed sorted reference index
  //
  // Barrier rule:
  //   Use __syncthreads_count(done) for tile-wide early exit. Per-pixel stop
  //   gates arithmetic only; it must not skip block barriers.
}

}  // namespace

void tile_forward_train_cuda(
    at::Tensor means2d,
    at::Tensor conics,
    at::Tensor colors,
    at::Tensor opacities,
    at::Tensor sorted_keys,
    at::Tensor sorted_gaussian_ids,
    at::Tensor tile_offsets,
    at::Tensor out_rgb,
    at::Tensor out_alpha,
    at::Tensor last_ids,
    RasterParams params) {
  TORCH_CHECK(means2d.is_cuda() && conics.is_cuda(), "projected inputs must be CUDA tensors");
  TORCH_CHECK(colors.is_cuda() && opacities.is_cuda(), "splat attributes must be CUDA tensors");
  TORCH_CHECK(sorted_keys.is_cuda() && sorted_gaussian_ids.is_cuda(),
              "sorted intersection buffers must be CUDA tensors");
  TORCH_CHECK(tile_offsets.is_cuda(), "tile_offsets must be a CUDA tensor");
  TORCH_CHECK(out_rgb.is_cuda() && out_alpha.is_cuda() && last_ids.is_cuda(),
              "forward outputs must be CUDA tensors");
  (void)params;
  TORCH_CHECK(false,
              "tile_forward_train_cuda is a V9 CUDA scaffold. Implement exact "
              "C/T recurrence and parity tests on a CUDA host.");
}

}  // namespace v9cuda

