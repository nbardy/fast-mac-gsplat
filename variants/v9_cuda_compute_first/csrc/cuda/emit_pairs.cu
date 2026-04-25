#include "v9_cuda_contract.cuh"

namespace v9cuda {

namespace {

__global__ void emit_pairs_kernel() {
  // Skeleton contract:
  //
  // One thread or small thread group revisits one Gaussian support rectangle.
  // pair_offsets comes from CUB DeviceScan::ExclusiveSum over tiles_per_gauss.
  //
  // For each touched tile:
  //   key = image_id << (tile_bits + depth_bits)
  //       | tile_id  << depth_bits
  //       | depth_key
  //   gaussian_ids[offset + k] = gaussian_id
  //
  // Depth ordering:
  //   Use monotonic float-to-uint transform for positive camera-space depth.
  //   If equal-depth stability matters, encode a deterministic tie-break through
  //   the value or a wider composite key. Do not depend on unstable same-key
  //   radix-sort ordering.
  //
  // Capacity:
  //   The fixed-capacity buffers are intentionally explicit. Overflow sets a
  //   device flag and returns; Python/launcher grows capacity outside timing.
}

}  // namespace

void emit_pairs_cuda(
    at::Tensor means2d,
    at::Tensor conics,
    at::Tensor depths,
    at::Tensor opacities,
    at::Tensor pair_offsets,
    at::Tensor isect_keys,
    at::Tensor gaussian_ids,
    at::Tensor overflow,
    RasterParams params) {
  TORCH_CHECK(means2d.is_cuda() && conics.is_cuda() && depths.is_cuda(),
              "projected inputs must be CUDA tensors");
  TORCH_CHECK(opacities.is_cuda(), "opacities must be a CUDA tensor");
  TORCH_CHECK(pair_offsets.is_cuda(), "pair_offsets must be a CUDA tensor");
  TORCH_CHECK(isect_keys.is_cuda() && gaussian_ids.is_cuda(),
              "pair outputs must be CUDA tensors");
  TORCH_CHECK(overflow.is_cuda(), "overflow must be a CUDA tensor");
  (void)params;
  TORCH_CHECK(false,
              "emit_pairs_cuda is a V9 CUDA scaffold. Implement after CUB scan "
              "capacity policy is wired.");
}

}  // namespace v9cuda

