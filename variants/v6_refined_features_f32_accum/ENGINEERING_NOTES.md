# v6_refined_features_f32_accum Engineering Notes

This is an experimental fork of `variants/v6_refined_features_f32_reduce`.
It tests a direct fast-forward optimization: for `F <= 32`, each pixel thread
accumulates its feature vector in thread-local storage and writes the dense
output once at the end.

Do not point production configs at this fork yet. Under the corrected
trainer-like `GSP_CHUNK=64,GSP_FAST_CAP=2048` cap it beats `f32_reduce` on
synthetic fwd+bwd pressure rows, but the 256px fixed-render trainer gate still
favors `f32_reduce`.

## Namespace

- Python package: `torch_gsplat_bridge_v6_refined_features_f32_accum`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features_f32_accum`
- Metal source: `csrc/metal/gsplat_v6_refined_features_f32_accum_kernels.metal`
- output API: `(features, accumulated_alpha)`

## What Changed Versus f32_reduce

- Adds `GSP_LOCAL_ACCUM_CAP=32`.
- Direct fast eval/state kernels use `thread float feature_accum[32]` when
  `feature_dim <= 32`.
- The fast forward path accumulates `sum(T * alpha * color)` locally and writes
  `acc + T * background` once per output channel.
- Active and overflow paths are inherited unchanged.
- Backward is inherited from f32_reduce.

This fork was initially most plausible for render/eval/video generation where
no backward runs. With the corrected cap, it is also a training-timing
candidate, but not the current trainer pick.

## Verified Gates

Run from the Dynaworld root after building the extension:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_accum/tests/feature_contract_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_accum/tests/alpha_output_check.py
```

Observed result: feature contract passed for F3/F8/F32 active off/on, and alpha
tests A-F passed.

## Benchmark Snapshot

Legacy local MPS, `512x512`, `B=16`, `G=8192`, `F=32`,
`case=medium_sigma_3_8`, `batch_strategy=flatten`, `active_policy=off`.
These rows used the old fallback-pressure cap and should not be used for the
primary throughput comparison.

| Variant | iters | Forward ms | Backward ms | Total mean ms |
| --- | ---: | ---: | ---: | ---: |
| `v6_refined_features_f32_reduce` | 5 | 180.2 | 535.3 | 715.4 |
| `v6_refined_features_f32_accum` | 5 | 168.9 | 610.1 | 779.1 |

Forward-only eval:

| Variant | iters | Forward mean ms | Forward median ms |
| --- | ---: | ---: | ---: |
| `v6_refined_features_f32_reduce` | 5 | 202.6 | 202.6 |
| `v6_refined_features_f32_accum` | 5 | 168.0 | 165.4 |

Corrected cap rows, `GSP_CHUNK=64,GSP_FAST_CAP=2048`:

| Shape | Variant | colors trainable | Forward ms | Backward ms | Total mean ms |
| --- | --- | --- | ---: | ---: | ---: |
| 128px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 26.1 | 67.4 | 93.5 |
| 128px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 24.3 | 66.9 | 91.2 |
| 512px B16/G8192/F32 | `v6_refined_features_f32_reduce` | true | 123.0 | 326.2 | 449.2 |
| 512px B16/G8192/F32 | `v6_refined_features_f32_accum` | true | 77.8 | 310.3 | 388.1 |
| 512px B16/G8192/F32 | `v6_refined_features_f32_reduce` | false | 87.9 | 179.8 | 267.7 |
| 512px B16/G8192/F32 | `v6_refined_features_f32_accum` | false | 78.4 | 178.3 | 256.7 |

Interpretation: local accumulation reduces direct fast-forward output traffic
and, once the fallback cap is fixed, also improves synthetic fwd+bwd total.

Dynaworld fixed-render gate, 256px multicam F32, `seed=0`, `warmup=2`,
`iters=4`:

| Variant | colors trainable | Raster fwd ms | Backward ms | Total median ms |
| --- | --- | ---: | ---: | ---: |
| `v6_refined_features_f32_reduce` | true | 67.7 | 511.5 | 658.9 |
| `v6_refined_features_f32_accum` | true | 73.7 | 511.8 | 665.5 |
| `v6_refined_features_f32_reduce` | false | 68.7 | 457.7 | 608.9 |
| `v6_refined_features_f32_accum` | false | 76.9 | 453.3 | 609.4 |

Interpretation: the accumulation fork is useful but not the trainer winner in
this gate. Keep `f32_reduce` as the better opt-in trainer candidate until a
larger or render-only gate says otherwise.

F64 synthetic pressure rows, `GSP_CHUNK=64,GSP_FAST_CAP=2048,GSP_FEATURE_CAP=64`,
`G=8192`, `case=medium_sigma_3_8`, `warmup=1`, `iters=2`:

| Shape | Stable total | `f32_reduce` total | `f32_accum` total |
| --- | ---: | ---: | ---: |
| 128px B16/F64 | 678.4ms | 340.6ms | 365.6ms |
| 256px B16/F64 | 2021.6ms | 1224.9ms | 964.6ms |
| 512px B4/F64 | 754.0ms | 412.7ms | 620.3ms |
| 512px B8/F64 | 1614.0ms | 890.8ms | 837.6ms |

Important caveat: this fork's `GSP_LOCAL_ACCUM_CAP` is still `32`, so F64 does
not actually take the local-accumulation branch yet. The F64 wins are useful
measured evidence for this compiled fork, but a true F64 accumulation follow-up
needs a separate cap-64 or two-block fork.

The cheap forward parity gate in
`src/benchmarks/fixed_render_variant_parity.py` measured exact 256px
feature/alpha/RGB/loss parity against stable `v6_refined_features` on the same
seeded trainer train and heldout render paths (`max_abs=0.0`, loss diff `0.0`).
The 128px `--check-gradients` gate also matched decoded sequence gradients
within `4.07e-10` on the train target and `9.60e-10` on the heldout target,
with exact colorize parameter gradients. This is still a pre-flight correctness
gate, not heldout-quality training parity.
Fork-local `feature_contract_check.py` now includes F64 color-gradient parity;
`F=64 feature grad active_policy=off` measured `max_abs=1.16e-10`.
