#include "v9_cuda_contract.cuh"

namespace v9cuda {

namespace {

__global__ void project_count_fused_kernel() {
  // Skeleton contract:
  //
  // One thread handles one [batch, camera, gaussian].
  //
  // Inputs:
  //   means3d, covariance or quat/scale, opacity, camera matrices/intrinsics.
  //
  // Outputs:
  //   means2d: float2 [B,C,N]
  //   conics: float3 [B,C,N] = inverse 2D covariance (a,b,c)
  //   depths: float [B,C,N]
  //   tiles_per_gauss: int32 [B,C,N]
  //   flags: clipped/overflow/debug bits
  //
  // Math:
  //   mu_c = R * mu_w + t
  //   mean2d = (fx*x/z + cx, fy*y/z + cy)
  //   J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
  //   Sigma2d = J * R * Sigma3d * R^T * J^T + eps2d * I
  //   conic = inverse_2x2(Sigma2d)
  //   tau = -2 * log(alpha_threshold / max(opacity_eff, eps))
  //   support = conservative ellipse AABB and optional ellipse-vs-tile test
  //
  // Capacity boundary:
  //   This kernel only counts. CUB DeviceScan owns prefix offsets.
}

}  // namespace

void project_count_fused_cuda(
    at::Tensor means3d,
    at::Tensor covars,
    at::Tensor opacities,
    at::Tensor cameras,
    at::Tensor means2d,
    at::Tensor conics,
    at::Tensor depths,
    at::Tensor tiles_per_gauss,
    at::Tensor flags,
    RasterParams params) {
  TORCH_CHECK(means3d.is_cuda(), "means3d must be a CUDA tensor");
  TORCH_CHECK(covars.is_cuda(), "covars must be a CUDA tensor");
  TORCH_CHECK(opacities.is_cuda(), "opacities must be a CUDA tensor");
  TORCH_CHECK(cameras.is_cuda(), "cameras must be a CUDA tensor");
  TORCH_CHECK(means2d.is_cuda() && conics.is_cuda() && depths.is_cuda(),
              "projection outputs must be CUDA tensors");
  TORCH_CHECK(tiles_per_gauss.is_cuda() && flags.is_cuda(),
              "count outputs must be CUDA tensors");
  (void)params;
  TORCH_CHECK(false,
              "project_count_fused_cuda is a V9 CUDA scaffold. Implement on a "
              "CUDA host after validating projection parity.");
}

}  // namespace v9cuda

