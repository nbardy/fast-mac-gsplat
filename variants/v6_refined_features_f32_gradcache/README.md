# torch-metal-gsplat-v6-refined-features

Torch-native projected 2D Gaussian rasterizer for Apple Silicon / MPS with a
Metal hot path that composites arbitrary per-splat feature channels.

This is an isolated feature-channel namespace fork derived from
`variants/v6_refined_features_f32_reduce`, created to test direct fast-backward
`grad_features` caching without mutating the stable `v6_refined_features`
baseline or the current reduction fork.

This fork now carries the v6_refined active-tile scheduling surface for
arbitrary feature channels plus accumulated alpha. Direct tiles remain the
default; use `active_policy="auto"` or `"on"` for sparse-screen / overflow-tail
probes, matching the RGB v6_refined branch's caution around active scheduling.

## Feature Contract

- Package: `torch_gsplat_bridge_v6_refined_features_f32_gradcache`
- Custom-op namespace: `torch.ops.gsplat_metal_v6_refined_features_f32_gradcache`
- Input `colors` tensor is feature data with shape `[G,F]` or `[B,G,F]`.
- Output is `(features, accumulated_alpha)`.
- `features` shape is `[H,W,F]` or `[B,H,W,F]`.
- `accumulated_alpha` shape is `[H,W]` or `[B,H,W]`, where `0` means no splat
  coverage and `1` means fully opaque.
- `F` is runtime inferred from `colors.shape[-1]`.
- `RasterConfig.background` may be length 1, which broadcasts to every feature
  channel, or exactly length `F`.
- The default runtime cap is `GSP_FEATURE_CAP=64`. Set a larger cap before
  import if a run needs `F > 64`.

## Inherited from v5_features

- **Batchwise rendering**: accepts `[B,G,2/3/F]` inputs and renders `[B,H,W,F]`
- **Auto batch chunking**: `batch_strategy=auto|flatten|serial`
- **Inference-only fast path**: no sorted-ID writeback when gradients are not needed
- **Training fast path**: writes sorted IDs back into `binned_ids` and saves per-tile stop counts for backward
- **Active-tile fast path**: optional v6_refined-style active-tile eval/train
  kernels for arbitrary `F`, with alpha output and `grad_alpha` backward
- **Runtime-specialized ablations** via env before import:
  - `GSP_TILE_SIZE=8|16|32`
  - `GSP_CHUNK=32|64|128`
  - `GSP_FAST_CAP=1024|2048|4096`
  - `GSP_FEATURE_CAP=64` by default; raise for larger feature tensors

## Build

```bash
python3 setup.py build_ext --inplace
```

## Quick Check

```bash
python3 tests/feature_contract_check.py
python3 tests/alpha_output_check.py
python3 tests/reference_check.py
```

`feature_contract_check.py` covers the fork-specific contract:

- shapes for `F in {1,3,4,8,16,32,64}`
- F=3 forward parity against original v5
- `dL/dfeatures` against a CPU Torch reference for `F in {3,8,32}`
- 100-iteration no-NaN smoke at `F=32`

`alpha_output_check.py` covers the accumulated-alpha contract:

- forward shape/value checks for empty, one-splat, and two-splat pixels
- alpha-only loss gradients on means2d, conics, and opacities with zero color grad
- alpha parity against an explicit synthetic all-ones feature channel
- combined feature+alpha backward linearity
- F=3 alpha parity against original v5 transmittance

## Depth Sorting Contract

By default, `RasterConfig(inputs_sorted_by_depth=False)` stably sorts splats by
nondecreasing `depths` inside this fork, gathers `means2d` / `conics` /
`colors` / `opacities` into that order, and unsorts input gradients in backward.

Set `inputs_sorted_by_depth=True` only when the caller has already applied that
same per-batch stable depth order to every per-splat input tensor. Under that
explicit contract this fork skips the internal `argsort`, gather, and backward
unsort. Passing unsorted tensors with this flag changes compositing order and
gradients.

## Benchmarks

```bash
python3 benchmarks/benchmark_matrix.py --dry-run --include-stable-baseline --backward
python3 benchmarks/benchmark_matrix.py --height 128 --width 128 --gaussians 1024 --batch-sizes 1 --feature-dim 32 --cases medium_sigma_3_8 --warmup 1 --iters 2 --timeout-s 30 --backward
```

## Notes

- input API is projected 2D splats, not full 3D camera projection
- depth gradients are zero; sort order is piecewise constant
- overflow tiles fall back to a slower path
- `auto` batch mode chunks large batches to cap total launched tiles / gaussians
- v6_refined_features_f32_gradcache caches `grad_features[pix, :]` in a
  per-thread `float[32]` for direct fast backward when `F <= 32`. It passed
  correctness gates and is an opt-in timing candidate: it helped
  `256px/B16/F32`, lost `256px/B32/F32`, had mixed `512px/B16/F32` synthetic
  rows, and won the latest same-session 256px trainer fixed-render gate.
- Active tiles are not a global win. Dense-screen cases can be slower because
  they pay active-output initialization and sparse launch overhead. Use the
  profile fields (`selected_use_active_tiles`, `active_tile_fraction`,
  `overflow_tile_count`, stop-ratio stats) before promoting active mode.
- Keep original `v5_features` as the stable F-channel baseline; use this fork
  when callers need a separate namespace for v6-refined feature experiments.
