# v6_refined_features_f32_block4 Engineering Notes

This is an experimental fork of `variants/v6_refined_features_f32_reduce`. It
exists so the stable feature-splatting baseline and current reduction fork stay
untouched while we test one specific optimization: fusing the generic feature
dot-product and color-gradient reduction in 4-channel blocks.

Do not point production configs at this fork by default. Promote it only after a
trainer smoke, full phase trace, and heldout-quality parity check.

## Namespace

- Python package: `torch_gsplat_bridge_v6_refined_features_f32_block4`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features_f32_block4`
- Metal source: `csrc/metal/gsplat_v6_refined_features_f32_block4_kernels.metal`
- output API: `(features, accumulated_alpha)`

The Dynaworld trainer dispatcher does not expose this fork. Keep it
benchmark-only unless a future result justifies adding an explicit opt-in route.

## What Changed Versus f32_reduce

- Adds `reduce_atomic_add_feature_grads_and_dot_block4(...)`.
- Direct fast backward generic feature path computes `dot(grad_features, color)`
  and `g_colors` reduction from the same per-channel loads.
- The helper processes feature channels in blocks of 4 without keeping a full
  F32 private cache across all Gaussians.
- `F=3` and skipped-color-gradient paths stay on the inherited `f32_reduce`
  behavior.
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
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_block4/tests/feature_contract_check.py
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
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_block4/tests/alpha_output_check.py
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

| Shape | `f32_reduce` | `f32_accum` | `f32_gradcache` | `f32_block4` | Read | Artifact |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 256px B16 | 332.8ms | 326.0ms | 326.4ms | 377.9ms | loses | `2026-05-07_256_f32_b16_b32_block4_matrix.jsonl` |
| 256px B32 | 656.7ms | 647.2ms | 620.8ms | 692.8ms | loses | `2026-05-07_256_f32_b16_b32_block4_matrix.jsonl` |
| 512px B16 | 437.6ms | 393.1ms | 369.7ms | 547.3ms | loses badly | `2026-05-07_512_f32_b16_block4_matrix.jsonl` |

Read: reusing the per-channel `grad_features` loads for both the dot product and
color-gradient reduction looked plausible, but the extra helper/control work
increased backward time. Do not promote this fork or wire it into trainer
dispatch from the current evidence.

## Metal References

- https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups
- https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)
- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth
- https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon
- https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy
