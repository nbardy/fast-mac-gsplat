# v6_refined_features_f64_accum64 Engineering Notes

This is an experimental fork of `variants/v6_refined_features_f32_accum`.
It tests a direct fast-forward optimization: for `F <= 64`, each pixel thread
accumulates its feature vector in thread-local storage and writes the dense
output once at the end.

Do not point production configs at this fork yet. The stable
`variants/v6_refined_features` baseline and the earlier `f32_reduce` /
`f32_accum` forks are untouched.

## Namespace

- Python package: `torch_gsplat_bridge_v6_refined_features_f64_accum64`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features_f64_accum64`
- Metal source: `csrc/metal/gsplat_v6_refined_features_f64_accum64_kernels.metal`
- output API: `(features, accumulated_alpha)`

## What Changed Versus f32_accum

- Raises `GSP_LOCAL_ACCUM_CAP` from `32` to `64`.
- Direct fast eval/state kernels use `thread float feature_accum[64]` when
  `feature_dim <= 64`.
- The fast forward path accumulates `sum(T * alpha * color)` locally and writes
  `acc + T * background` once per output channel.
- Active and overflow paths are inherited unchanged.
- Backward is inherited from f32_reduce.

This fork is a deliberate pressure test for true F64 local accumulation. The
risk is private/thread storage pressure from 64 floats per pixel thread; compare
against both `f32_reduce` and `f32_accum` on the same shape before keeping it.

## Verified Gates

Run from the Dynaworld root after building the extension:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f64_accum64/tests/feature_contract_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f64_accum64/tests/alpha_output_check.py
```

Observed result: feature contract passed for F3/F8/F32 active off/on, and alpha
tests A-F passed.

## Benchmark Snapshot

This fork should be compared only against same-session `f32_reduce` and
`f32_accum` rows. Older copied `f32_accum` rows do not prove anything about this
cap-64 variant.

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

This cap-64 fork was built and re-tested directly:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_refined_features_f64_accum64
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f64_accum64/tests/feature_contract_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f64_accum64/tests/alpha_output_check.py
```

Both tests passed, including `F=64 feature grad active_policy=off max_abs=1.16e-10`
and alpha tests A-F.

## Cap-64 Benchmark Result

The cap-64 private accumulator is not a promotion candidate:

| Shape | `f32_reduce` | `f32_accum` | `f64_accum64` | Artifact |
| --- | ---: | ---: | ---: | --- |
| 256px B16/G8192/F64 warm2/iters3 | 654.5ms | 646.4ms | 682.4ms | `2026-05-07_256_f64_b16_accum64_confirm_matrix.jsonl` |
| 512px B8/G8192/F64 warm1/iters2 | 542.7ms | 567.9ms | 567.0ms | `2026-05-07_512_f64_b8_accum64_matrix.jsonl` |

Read: the 64-float private accumulator adds enough forward pressure that it
does not beat existing forks. If this idea is revisited, use a two-block
32-channel design or another lower-private-memory strategy.
