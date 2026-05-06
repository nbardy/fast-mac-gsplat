# v6_refined_features_f32_reduce Engineering Notes

This is an experimental fork of `variants/v6_refined_features`. It exists so
the stable feature-splatting baseline stays untouched while we test one specific
optimization: reducing high-contention F-channel color-gradient atomics.

Do not point production configs at this fork by default. Promote it only after a
trainer smoke, full phase trace, and heldout-quality parity check.

## Namespace

- Python package: `torch_gsplat_bridge_v6_refined_features_f32_reduce`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features_f32_reduce`
- Metal source: `csrc/metal/gsplat_v6_refined_features_f32_reduce_kernels.metal`
- output API: `(features, accumulated_alpha)`

The Dynaworld trainer can select this fork with
`render.fast_mac.feature_variant = "v6_refined_features_f32_reduce"`, but no
checked-in training config uses it by default.

## What Changed Versus Stable v6_refined_features

- `F == 3` backward uses a `float3` path for feature/color gradient arithmetic.
- Generic `F` backward uses `reduce_atomic_add_feature_grads(...)`:
  `simd_sum` first, threadgroup aggregation second, one global atomic per
  Gaussian/channel/threadgroup last.
- Python backward checks `ctx.needs_input_grad[2]`; if colors/features do not
  need gradients, it marks metadata `reserved0` so Metal skips `g_colors`
  atomics.
- C++ backward now also avoids allocating a full `g_colors` gradient tensor
  when `reserved0` requests skipped color gradients. It returns an empty tensor
  to Python and passes a 1-float placeholder buffer to the kernel.
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
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/tests/feature_contract_check.py
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
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/tests/alpha_output_check.py
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

Local MPS, same-session comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_reduce/benchmarks/benchmark_mps.py \
  --height 512 --width 512 --gaussians 8192 --batch-size 16 --feature-dim 32 \
  --case medium_sigma_3_8 --warmup 1 --iters 3 --backward \
  --batch-strategy flatten --active-policy off --json
```

| Variant | iters | Forward ms | Backward ms | Total mean ms |
| --- | ---: | ---: | ---: | ---: |
| stable `v6_refined_features` | 3 | 216.6 | 1712.1 | 1928.7 |
| fork `v6_refined_features_f32_reduce` | 5 | 180.2 | 535.3 | 715.4 |
| fork `v6_refined_features_f32_reduce`, post allocation cleanup | 5 | 123.0 | 446.0 | 568.9 |
| fork `v6_refined_features_f32_reduce`, colors frozen | 5 | 119.7 | 242.5 | 362.2 |

The fork is much faster in backward on this F32/high-batch microbenchmark and
is the current best training-raster variant. The table above used an older
`GSP_CHUNK=32,GSP_FAST_CAP=512` pressure setting. With the trainer-like
`64/2048` cap, the same 512px/B16/G8192/F32 pressure probe measured:

| Variant | colors trainable | Forward ms | Backward ms | Total mean ms |
| --- | --- | ---: | ---: | ---: |
| stable `v6_refined_features` | true | 88.2 | 913.7 | 1001.9 |
| fork `v6_refined_features_f32_reduce` | true | 123.0 | 326.2 | 449.2 |
| fork `v6_refined_features_f32_reduce` | false | 87.9 | 179.8 | 267.7 |

Use the `64/2048` rows for primary throughput comparisons. Use the older
`32/512` rows only to understand fallback/cap pressure. A 128px/B16/G8192/F32
profile confirmed the mechanism: `cap=512` overflowed `970/1024` tiles and took
about `833ms` forward, while `cap=2048` overflowed `0` tiles and took about
`23-24ms` forward. Chunk size was not the cause in that profile pair.

128px multicam trainer smoke, checked-in DeepView train cameras `0006/0014`,
heldout `0005`, `16` frames, `8192` splats, cached V-JEPA features,
`warmup=1`, `iters=2`:

| Variant | Total mean ms | Raster fwd ms | Autograd backward total ms |
| --- | ---: | ---: | ---: |
| current config, `v5_features` | 439.9 | 25.8 | 245.3 |
| opt-in `v6_refined_features_f32_reduce` | 403.5 | 22.0 | 241.4 |

This is a runtime-valid trainer trace and is directionally faster, but it is
not a promotion gate. A warmed detached raster-backward probe on the same shape
was slower for the fork (`81.5ms` vs `74.0ms`), so promotion still needs a
larger phase trace and heldout-quality parity.

At 256px on the same split, the fork reduced the warmed detached raster-backward
probe (`116.6ms` vs stable feature `151.0ms`) but did not win the full trainer
step reliably. A warm2/iters4 phase run measured stable feature median
`980.3ms` versus fork median `1147.8ms`. Treat this as non-promotion evidence:
the fork is a useful kernel experiment, not the default trainer renderer yet.

The cleaner trainer-context timing is the seeded fixed-render graph mode in
`src/benchmarks/trainer_phase_benchmark.py`. At 256px, `seed=0`, `warmup=2`,
`iters=4`, it measured stable feature median `673.3ms`, fork median `658.9ms`,
and fork with colors frozen `608.9ms`. That supports the fork for timing and
especially frozen-feature/camera-only follow-ups, but still does not replace a
heldout-quality parity gate.

F64 synthetic pressure rows, `GSP_CHUNK=64,GSP_FAST_CAP=2048,GSP_FEATURE_CAP=64`,
`G=8192`, `case=medium_sigma_3_8`, `warmup=1`, `iters=2`:

| Shape | Stable total | `f32_reduce` total | `f32_accum` total |
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
