# V9 CUDA Compute-First Scaffold

Date: 2026-04-25

This variant is a Mac-safe source scaffold for Direction C. It does not try to
build or run CUDA kernels on Apple Silicon. The local package only exposes an
environment probe; native extension builds fail clearly when CUDA toolkit
support is absent, and runtime smoke tests still require a visible CUDA device.

## Local Environment Result

This workspace currently reports:

```text
machine: arm64
platform: macOS-15.5-arm64-arm-64bit-Mach-O
nvcc: not found
nvidia-smi: not found
torch CUDA built: false
torch CUDA available: false
torch MPS available: true
```

So no CUDA smoke or benchmark was run here.

## Kernel Contract

The CUDA path should be compute-first:

```text
project_count_fused
  -> CUB DeviceScan::ExclusiveSum
emit_pairs
  -> CUB DeviceRadixSort::SortPairs or segmented radix sort
encode_tile_ranges
tile_forward_train
tile_backward_replay
project_backward
```

Exact forward recurrence:

```text
alpha = min(max_alpha, opacity * exp(-0.5 * q))
C += T * alpha * color
T *= (1 - alpha)
stop when T <= transmittance_threshold
```

Backward must replay the same processed prefix in reverse. The first CUDA host
implementation should save `out_alpha` and `last_ids`, then add a V8-style
`tile_stop_count` recompute ablation after parity is proven.

## Files

- `csrc/cuda/project_count_fused.cu`: fused 3D projection, conic construction,
  opacity-aware support, and tile count contract.
- `csrc/cuda/emit_pairs.cu`: fixed-capacity pair emission after CUB scan.
- `csrc/cuda/tile_forward_train.cu`: exact one-block-per-16x16-tile C/T
  forward contract.
- `csrc/cuda/tile_backward_replay.cu`: exact reverse replay and reduction
  contract.
- `csrc/include/v9_cuda_contract.cuh`: shared parameter/state contract.
- `tests/environment_check.py`: local CUDA availability probe.

## CUDA Host Next Steps

1. Build with a CUDA-enabled PyTorch and `nvcc`:

   ```bash
   cd variants/v9_cuda_compute_first
   python setup.py build_ext --inplace
   ```

2. Replace scaffold `TORCH_CHECK(false, ...)` bodies with tiny kernels in this
   order: `project_count_fused`, CUB scan driver, `emit_pairs`, sort/ranges,
   `tile_forward_train`, `tile_backward_replay`.
3. Validate against V8/gsplat on tiny single- and multi-splat cases before any
   speed work.
4. Benchmark stage timing and memory counters only after correctness gates pass.
