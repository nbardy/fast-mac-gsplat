# torch-metal-gsplat-v6-feature-lookup-experiment

Experimental fork of `variants/v6_refined_features` for testing whether F32
feature splatting memory can be reduced by rasterizing compact channels and
doing the F-dimensional feature reconstruction afterward.

This is a prototype namespace only. It is not wired into Dynaworld trainer
dispatch and it does not modify `v5`, `v5_features`, `v6`, or
`v6_refined_features`.

## Namespace

- Package: `torch_gsplat_bridge_v6_feature_lookup_experiment`
- Custom-op namespace: `torch.ops.gsplat_metal_v6_feature_lookup_experiment`
- Kernel source: `csrc/metal/gsplat_v6_feature_lookup_experiment_kernels.metal`

## Implemented Prototype

The buildable Metal path is copied from `v6_refined_features` and still
accumulates a dense runtime channel count. The experimental API changes the
meaning of those channels:

```python
from torch_gsplat_bridge_v6_feature_lookup_experiment import (
    RasterConfig,
    rasterize_projected_gaussians_feature_lookup,
)

result = rasterize_projected_gaussians_feature_lookup(
    means2d,
    conics,
    feature_weights,  # [G,K] or [B,G,K], compact coefficients
    feature_lookup,   # [K,F], reconstructs full features
    opacities,
    depths,
    RasterConfig(height=H, width=W, background=(0.0,)),
)

features = result.features  # [H,W,F] or [B,H,W,F]
alpha = result.alpha        # [H,W] or [B,H,W]
compact = result.compact    # [H,W,K] or [B,H,W,K]
```

Forward math:

```text
compact = splat(feature_weights, zero compact background)
features = compact @ feature_lookup + (1 - alpha) * full_feature_background
```

`RasterConfig.background` is interpreted in reconstructed feature space for this
API, so it must have length `1` or output `F`. The compact Metal pass always
uses zero background.

There is also an ID/weight-shaped skeleton:

```python
rasterize_projected_gaussians_feature_ids(
    means2d,
    conics,
    feature_ids,        # [G,L] or [B,G,L]
    feature_id_weights, # [G,L] or [B,G,L]
    feature_lookup,     # [K,F]
    opacities,
    depths,
    config,
)
```

That helper currently densifies sparse IDs into `[G,K]` or `[B,G,K]` with
`scatter_add_` before calling the compact-coefficient path. It is useful for
trainer/API experiments, not a proof of sparse in-kernel memory savings.

## Feasibility Read

Feasible without new compositing math:

- Splat K compact channels instead of F full channels when each splat can be
  represented as learned coefficients in a shared basis table.
- Reconstruct full features after the rasterizer with `compact @ lookup`.
- Backpropagate through the lookup and compact rasterizer using existing
  autograd: gradients entering F features multiply by `lookup.T` before the
  Metal rasterizer backward sees them.

Not implemented in this fork:

- A true sparse ID kernel that never materializes dense `[G,K]` coefficients.
- Per-splat variable-length ID lists in Metal.
- Backward gradients for sparse ID weights and lookup entries inside a custom
  fused Metal op.

## Required Kernel/API Work For True Sparse IDs

The current v6 feature kernels assume a dense per-splat feature tensor:

- `colors` is `[BG,F]` and `add_weighted_features` loops over every channel.
- Backward computes `dot_pixel_features(...)` over every channel and atomically
  accumulates `g_colors[g, f]`.
- `meta_i32.feature_dim` drives output allocation, gradient shape checks, and
  background packing.

A real sparse ID variant needs new custom ops and kernels with signatures closer
to:

```text
feature_ids: int32[BG,L]
feature_weights: float32[BG,L]
lookup: float32[K,F]
```

The forward kernel can either:

1. Accumulate compact per-pixel coefficients `[H,W,K]`, then launch/fuse a
   reconstruction pass to `[H,W,F]`.
2. Accumulate directly into `[H,W,F]` by gathering `lookup[id, :]` per ID, which
   saves per-splat storage but still pays F-channel output writes in the hot
   loop.

Path 1 is the cleaner memory experiment because rasterizer intermediates,
backward input gradients, and `g_feature_weights` scale with K or L rather than
F. It does require a new backward contract:

- `grad_features -> grad_compact = grad_features @ lookup.T`
- lookup gradient from all pixels: `compact.T @ grad_features`
- sparse weight gradients: for each splat ID, accumulate the compact-channel
  gradient at that ID
- alpha/background gradients remain as in `v6_refined_features`

## Build

Use the dynaworld root as the uv project, matching the repo guide:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

## Quick Check

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/tests/feature_lookup_parity_check.py
```

This checks two contracts:

- compact-basis rendering matches direct full-feature rendering when
  `features = feature_weights @ feature_lookup`, including gradients for means,
  conics, compact weights, lookup table, and opacity
- the current ID/weight skeleton matches explicitly densified compact
  coefficients

Observed on 2026-05-07:

```text
features max_abs=8.9406967e-08
alpha max_abs=0
loss max_abs=0
grad_means max_abs=5.8207661e-11
grad_conics max_abs=1.4901161e-08
grad_weights max_abs=6.0535967e-09
grad_lookup max_abs=2.5611371e-09
grad_opacities max_abs=1.8626451e-09
feature lookup direct parity: ok
id_skeleton feature max_abs=0
id_skeleton alpha max_abs=0
feature id skeleton parity: ok
```

## Bounded Benchmark

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/benchmarks/benchmark_lookup_basis.py \
  --height 128 --width 128 --batch-size 16 --gaussians 2048 \
  --feature-dim 32 --compact-dims 4,8,16 --warmup 1 --iters 2 \
  --no-overflow-fallback
```

The benchmark compares:

- `direct`: build `full_features = feature_weights @ lookup`, then rasterize F
  channels
- `lookup`: rasterize K compact channels, then reconstruct F channels after the
  rasterizer

The benchmark reports both allocation after synchronized backward and a
background-sampled peak during the measured forward/backward window. The sampled
peak is better than the after-backward value, but it is still not an Xcode/Metal
hardware memory capture.

Observed local rows:

| Shape | K | direct mean ms | lookup mean ms | direct current bytes | lookup current bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128px B4/G2048/F32 | 4 | 49.9 | 18.9 | 10749184 | 10782464 |
| 128px B4/G2048/F32 | 8 | 42.6 | 22.6 | 11929856 | 13142272 |
| 128px B4/G2048/F32 | 16 | 41.7 | 30.8 | 12486912 | 15534336 |
| 128px B16/G2048/F32 | 4 | 162.2 | 84.0 | 42992896 | 43127552 |
| 128px B16/G2048/F32 | 8 | 164.6 | 100.3 | 44569856 | 49419520 |
| 128px B16/G2048/F32 | 16 | 112.6 | 75.4 | 110631168 | 60299520 |

Read: compact lookup is a real timing candidate on this bounded synthetic
shape, but sampled memory is not a monotonic win. The final `[B,H,W,F]` tensor
still exists after reconstruction.

Sampled-peak follow-up:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/benchmarks/benchmark_lookup_basis.py \
  --height 128 --width 128 --batch-size 16 --gaussians 8192 \
  --feature-dim 32 --compact-dims 4,8,16 --warmup 1 --iters 2 \
  --seed 14 --no-overflow-fallback --memory-sample-interval-ms 0.25
```

| Shape | K | direct mean ms | lookup mean ms | direct sampled peak | lookup sampled peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128px B16/G2048/F32 | 4 | 158.8 | 81.0 | 152014080 | 117284096 |
| 128px B16/G2048/F32 | 8 | 159.8 | 96.5 | 154112256 | 124100864 |
| 128px B16/G2048/F32 | 16 | 112.6 | 75.4 | 151361024 | 167093504 |
| 128px B16/G8192/F32 | 4 | 225.9 | 96.9 | 244678400 | 166563584 |
| 128px B16/G8192/F32 | 8 | 179.6 | 117.9 | 278758400 | 210605056 |
| 128px B16/G8192/F32 | 16 | 184.1 | 155.6 | 282427392 | 198021120 |

Read: at the more relevant `G=8192` row, compact lookup is faster and lowers
sampled current allocation for K=4/8/16. At smaller `G=2048`, K=16 was faster
but had higher sampled current allocation than direct. Treat lookup as promising
for large-G compact features, not as a proven general memory fix.
