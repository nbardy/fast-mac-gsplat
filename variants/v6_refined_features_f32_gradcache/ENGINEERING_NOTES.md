# v6_refined_features_f32_gradcache Engineering Notes

This is an experimental fork of `variants/v6_refined_features_f32_reduce`. It
exists so the stable feature-splatting baseline and current reduction fork stay
untouched while we test one specific optimization: caching each pixel thread's
F32 `grad_features[pix, :]` vector during direct fast backward.

Do not point production configs at this fork by default. Promote it only after a
trainer smoke, full phase trace, and heldout-quality parity check.

## Namespace

- Python package: `torch_gsplat_bridge_v6_refined_features_f32_gradcache`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features_f32_gradcache`
- Metal source: `csrc/metal/gsplat_v6_refined_features_f32_gradcache_kernels.metal`
- output API: `(features, accumulated_alpha)`

The Dynaworld trainer can select this fork with
`render.fast_mac.feature_variant = "v6_refined_features_f32_gradcache"`, but no
checked-in training config uses it by default.

## What Changed Versus f32_reduce

- Adds `GSP_GRAD_CACHE_CAP=32`.
- In direct fast backward, non-F3 feature paths with `feature_dim <= 32` load
  `grad_features[pix, :]` into `thread float grad_cache[32]` once per pixel
  thread.
- `dot_pixel_bg`, `dot_pixel_features`, and color-gradient reduction can reuse
  that cached vector instead of reloading the same dense image gradient for
  every Gaussian.
- `F=3` and `F>32` stay on the inherited `f32_reduce` path.

## Result

This fork built and passed local correctness gates:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_refined_features_f32_gradcache
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_gradcache/tests/feature_contract_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_gradcache/tests/alpha_output_check.py
```

The benchmark result is shape-dependent, but this fork is now a live opt-in
trainer-timing candidate:

| Shape | `f32_reduce` | `f32_accum` | `f32_gradcache` | Artifact |
| --- | ---: | ---: | ---: | --- |
| 256px B16/G8192/F32 | 337.1ms | 314.3ms | 273.7ms | `2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl` |
| 256px B32/G8192/F32 | 585.0ms | 593.8ms | 609.2ms | `2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl` |
| 512px B16/G8192/F32 early | 527.0ms | 596.4ms | 597.2ms | `2026-05-07_512_f32_b16_gradcache_matrix.jsonl` |
| 512px B16/G8192/F32 confirm | 423.3ms | 412.9ms | 391.3ms | `2026-05-07_512_f32_b16_gradcache_confirm_matrix.jsonl` |

Read: caching the full F32 gradient vector can help a moderate 256px/B16 row,
but private/thread pressure loses at B32 and one early 512px row. A later
512px/B16 confirm and the trainer fixed-render gate favored this fork. Keep it
explicitly opt-in until a heldout-quality parity run proves the training outcome.
- The no-overflow backward path avoids cloning dense grad tensors unless
  overflow zeroing is actually needed.
- The benchmark script adds `--freeze-colors` to isolate color-gradient cost.
- `benchmarks/benchmark_matrix.py` is now the safe sequential runner for small,
  timeout-bounded matrices and stable-baseline comparison. It also supports
  fork-only `--freeze-colors` sweeps and skips stable rows in that mode rather
  than mutating the stable benchmark script. Its defaults are
  `GSP_CHUNK=64,GSP_FAST_CAP=2048`, matching the trainer-like fast cap; lower
  caps should be treated as fallback-pressure stress tests.

This fork does not include feature staging or lookup-table feature IDs. Those
should be separate forks if tested.

## Verified Gates

Run from the Dynaworld root after building the extension:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_gradcache/tests/feature_contract_check.py
```

Observed result:

```text
shape contract active_policy=off: ok
F=3 v5 parity active_policy=off max_abs=0
shape contract active_policy=on: ok
F=3 v5 parity active_policy=on max_abs=0
F=3 feature grad active_policy=off max_abs=1.8626451e-09
F=8 feature grad active_policy=off max_abs=9.3132257e-10
F=32 feature grad active_policy=off max_abs=2.3283064e-10
F=32 feature grad active_policy=on max_abs=2.3283064e-10
F=32 no-NaN smoke active_policy=off: ok
F=32 no-NaN smoke active_policy=on: ok
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_gradcache/tests/alpha_output_check.py
```

Observed result:

```text
Test A passed.
Test B passed.
Test C passed.
Test D passed.
Test E passed.
Test F passed.
```

## Benchmark Snapshot

Local MPS, `GSP_CHUNK=64,GSP_FAST_CAP=2048`, `G=8192`, `F=32`,
`case=medium_sigma_3_8`:

| Shape | `f32_reduce` | `f32_accum` | `f32_gradcache` | Read |
| --- | ---: | ---: | ---: | --- |
| 256px B16 | 337.1ms | 314.3ms | 273.7ms | gradcache wins |
| 256px B32 | 585.0ms | 593.8ms | 609.2ms | gradcache loses |
| 512px B16 early | 527.0ms | 596.4ms | 597.2ms | gradcache loses |
| 512px B16 confirm | 423.3ms | 412.9ms | 391.3ms | gradcache wins |

Use `64/2048` rows for primary throughput comparisons. Use older `32/512` rows
only to understand fallback/cap pressure. A 128px/B16/G8192/F32 profile
confirmed the mechanism: `cap=512` overflowed `970/1024` tiles and took about
`833ms` forward, while `cap=2048` overflowed `0` tiles and took about
`23-24ms` forward. Chunk size was not the cause.

The trainer-context timing gate is the seeded fixed-render graph mode in
`src/benchmarks/trainer_phase_benchmark.py`. At 256px, `seed=0`, `warmup=2`,
`iters=4`, a same-session rerun measured:

| Variant | Total mean ms | Total median ms | Raster fwd ms | Autograd backward total ms |
| --- | ---: | ---: | ---: | ---: |
| stable `v6_refined_features` | 1133.8 | 1122.8 | 89.7 | 888.6 |
| `v6_refined_features_f32_reduce` | 1165.6 | 1113.1 | 87.5 | 923.5 |
| `v6_refined_features_f32_accum` | 1060.0 | 1073.4 | 93.8 | 807.6 |
| `v6_refined_features_f32_gradcache` | 1008.0 | 966.4 | 82.6 | 765.6 |

This supports keeping `f32_gradcache` available for follow-up timing runs, but
still does not replace a heldout-quality parity gate.

F64 synthetic pressure rows, `GSP_CHUNK=64,GSP_FAST_CAP=2048,GSP_FEATURE_CAP=64`,
`G=8192`, `case=medium_sigma_3_8`, `warmup=1`, `iters=2`:

| Shape | Stable total | `f32_gradcache` total | `f32_accum` total |
| --- | ---: | ---: | ---: |
| 128px B16/F64 | 678.4ms | 340.6ms | 365.6ms |
| 256px B16/F64 | 2021.6ms | 1224.9ms | 964.6ms |
| 512px B4/F64 | 754.0ms | 412.7ms | 620.3ms |
| 512px B8/F64 | 1614.0ms | 890.8ms | 837.6ms |

Read: this fork remains the better choice at `512px/B4/F64`, while `f32_accum`
wins at `256px/B16/F64` and narrowly at `512px/B8/F64`. Benchmark the target
shape before picking a fork for F64.

The cheap forward parity gate in
`src/benchmarks/fixed_render_variant_parity.py` measured exact 256px
feature/alpha/RGB/loss parity against stable `v6_refined_features` on the same
seeded trainer train and heldout render paths (`max_abs=0.0`, loss diff `0.0`).
The 128px `--check-gradients` gate also matched decoded sequence gradients on
train and heldout targets within `8.15e-10`, with exact colorize parameter
gradients. This is still a pre-flight correctness gate, not heldout-quality
training parity.

Fork-local `feature_contract_check.py` now includes F64 color-gradient parity;
`F=64 feature grad active_policy=off` measured `max_abs=1.16e-10`.

## Metal References

- https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups
- https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)
- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth
- https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon
- https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy
