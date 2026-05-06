# v6_refined_features_f32_stage Engineering Notes

This is an experimental fork of `variants/v6_refined_features_f32_reduce`.
It stages per-chunk feature/color values in threadgroup memory to test whether
reusing `colors[g, f]` helps F32 feature splatting.

Do not point production configs at this fork. The first benchmark pass regressed
fwd+bwd total versus `v6_refined_features_f32_reduce`.

## Namespace

- Python package: `torch_gsplat_bridge_v6_refined_features_f32_stage`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features_f32_stage`
- Metal source: `csrc/metal/gsplat_v6_refined_features_f32_stage_kernels.metal`
- output API: `(features, accumulated_alpha)`

## What Changed Versus f32_reduce

- `load_chunk_params(...)` also stages `colors[g, f]` into
  `threadgroup float sh_colors[GSP_CHUNK * GSP_FEATURE_CAP]`.
- Forward and backward dot-product paths read staged feature/color chunks.
- `gsplat_metal.mm` has a fail-fast estimated threadgroup-memory guard so large
  `GSP_CHUNK` / `GSP_FEATURE_CAP` / `GSP_FAST_CAP` combinations do not compile
  an oversized shader blindly.

This fork keeps the f32_reduce color-gradient atomic reduction. It does not add
lookup-table feature IDs or trainer wiring.

## Verified Gates

Run from the Dynaworld root after building the extension:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_stage/tests/feature_contract_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_stage/tests/alpha_output_check.py
```

Observed result: feature contract passed for F3/F8/F32 active off/on, and alpha
tests A-F passed.

## Benchmark Snapshot

Local MPS, `512x512`, `B=16`, `G=8192`, `F=32`,
`case=medium_sigma_3_8`, `batch_strategy=flatten`, `active_policy=off`.

| Variant | iters | Forward ms | Backward ms | Total mean ms |
| --- | ---: | ---: | ---: | ---: |
| `v6_refined_features_f32_reduce` | 5 | 180.2 | 535.3 | 715.4 |
| `v6_refined_features_f32_stage` | 3 | 183.2 | 755.8 | 938.9 |

Forward-only eval also lost versus the accumulation fork and did not justify
the larger threadgroup footprint:

| Variant | iters | Forward mean ms |
| --- | ---: | ---: |
| `v6_refined_features_f32_reduce` | 5 | 202.6 |
| `v6_refined_features_f32_stage` | 5 | 276.7 |

Interpretation: staging feature chunks increases threadgroup-memory pressure and
does not improve the train path. Keep this fork as a negative result.
