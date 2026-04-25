# V9 Plan C Execution Notes

Date: 2026-04-25

Scope: Direction C CUDA only. No Metal Direction A/B files were edited.

## Environment Detection

Local command result:

```json
{
  "machine": "arm64",
  "nvcc": null,
  "nvidia_smi": null,
  "platform": "macOS-15.5-arm64-arm-64bit-Mach-O",
  "python": "3.14.0",
  "torch_cuda_available": false,
  "torch_cuda_built": false,
  "torch_cuda_device_count": 0,
  "torch_mps_available": true,
  "torch_version": "2.11.0"
}
```

Conclusion: this Mac workspace cannot compile or run CUDA. There is no `nvcc`,
no `nvidia-smi`, and the local PyTorch build has no CUDA backend. CUDA work here
must stay source-level until moved to an NVIDIA host.

## Scaffold Status

Created `variants/v9_cuda_compute_first/` as a CUDA compute-first scaffold:

- `pyproject.toml`
- `setup.py`
- `torch_gsplat_bridge_v9_cuda_compute_first/environment.py`
- `tests/environment_check.py`
- `csrc/include/v9_cuda_contract.cuh`
- `csrc/bindings.cpp`
- `csrc/cuda/project_count_fused.cu`
- `csrc/cuda/emit_pairs.cu`
- `csrc/cuda/tile_forward_train.cu`
- `csrc/cuda/tile_backward_replay.cu`
- `README.md`

The package import path is Mac-safe and exposes `cuda_environment()`. The native
extension build is intentionally guarded on CUDA-enabled PyTorch, `CUDA_HOME`,
and `nvcc`. Runtime smoke tests additionally require
`torch.cuda.is_available()`. On this host, `python3 setup.py build_ext
--inplace` should fail with a clear CUDA-only error instead of producing
confusing compiler output.

No CUDA benchmark file was produced because there is no CUDA runtime. No fake
speed numbers were recorded.

## Encoded Kernel Contract

The scaffold encodes the Plan C pipeline in source comments:

```text
project_count_fused
  -> CUB DeviceScan::ExclusiveSum
emit_pairs
  -> CUB DeviceRadixSort::SortPairs / segmented sort
encode_tile_ranges
tile_forward_train
tile_backward_replay
project_backward
```

Default CUDA work shape:

```text
tile_size = 16
threads_per_block = 256
one block per tile per image/camera
one thread per pixel
CHUNK = 256 sorted splat references staged through shared memory
```

Forward math:

```text
q = a*dx^2 + 2*b*dx*dy + c*dy^2
power = -0.5*q
raw = opacity * exp(power)
alpha = min(max_alpha, raw)
visible = power <= 0 and alpha >= alpha_threshold

C += T * alpha * color
T *= 1 - alpha
stop when T <= transmittance_threshold
```

Backward replay math:

```text
T_cur = 1 - out_alpha[pixel]
gT = dot(grad_rgb[pixel], background)

for processed splats in reverse:
  denom = max(1 - alpha, eps)
  T_prev = T_cur / denom
  dot_c = dot(grad_rgb, color)
  d_alpha = T_prev * (dot_c - gT)
  d_color = grad_rgb * (T_prev * alpha)
  d_raw = d_alpha * clamp_gate * visible_gate
  d_power = d_raw * raw
  d_conic = d_power * [-0.5*dx^2, -dx*dy, -0.5*dy^2]
  d_mean = d_power * [a*dx + b*dy, b*dx + c*dy]
  d_opacity = d_raw * raw / max(opacity, eps)
  gT = alpha * dot_c + (1 - alpha) * gT
  T_cur = T_prev
```

Reduction boundary:

```text
baseline A: warp reduction then global atomicAdd
baseline B: full 16x16 block reduction, then one atomic per splat/tile/component
later: deferred partial reduce only if Nsight shows atomics dominate
```

## CUDA Host Next Steps

1. Use an NVIDIA Linux or Windows host with CUDA-enabled PyTorch, `nvcc`, and
   `nvidia-smi`.
2. Build the scaffold:

   ```bash
   cd variants/v9_cuda_compute_first
   python setup.py build_ext --inplace
   ```

3. Replace scaffold `TORCH_CHECK(false, ...)` bodies with tiny correctness-first
   kernels:

   ```text
   project_count_fused: one Gaussian -> mean2d/conic/depth/count
   emit_pairs: fixed small capacity -> keys + ids
   sort/ranges: CUB radix sort + tile offsets
   tile_forward_train: single tile, single image, C/T parity
   tile_backward_replay: finite-difference or V8/gsplat gradient parity
   ```

4. First validation matrix:

   ```text
   16x16: 1 splat, 2 overlapping splats, 16 ordered splats
   64x64: sparse grid, overlap stack, depth ties
   256x256: random projected scenes with fixed seed
   ```

5. First benchmark counters:

   ```text
   projection/count ms
   scan ms
   emit ms
   sort ms
   range encode ms
   forward ms
   backward replay ms
   n_isects
   max refs/tile
   p50/p90/p99 refs/tile
   overflow flag
   atomics issued estimate
   ```

6. Only after parity:

   ```text
   remove hot-path CPU shape reads with fixed/cached capacity
   compare warp vs full-block gradient reduction
   add tile_stop_count recompute ablation
   capture fixed-capacity CUDA Graph
   consider cp.async on SM80+ if splat batch loads are limiting
   ```

## Non-Goals For This Scaffold

- No Vulkan or hardware raster implementation yet.
- No fake CUDA benchmark on MPS.
- No per-pixel front-K or linked-list state.
- No custom global sort replacing CUB.
- No CUDA Dynamic Parallelism or TMA-first design.
