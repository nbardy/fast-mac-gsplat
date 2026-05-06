# v6_refined_features_f32_zero_bg Engineering Notes

This is an experimental fork copied from
`variants/v6_refined_features_f32_reduce`. It exists so the stable
`v6_refined_features` baseline and the existing `f32_reduce` timing reference
stay untouched while we test one narrow optimization: skipping background-tail
feature writes when the configured feature background is exactly zero.

Do not point production configs at this fork by default. Promote it only after
a trainer fixed-render trace, full camera-swap parity where relevant, and
heldout-quality W&B parity.

## Namespace

- Python package: `torch_gsplat_bridge_v6_refined_features_f32_zero_bg`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features_f32_zero_bg`
- Metal source: `csrc/metal/gsplat_v6_refined_features_f32_zero_bg_kernels.metal`
- output API: `(features, accumulated_alpha)`

The Dynaworld trainer can select this fork with
`render.fast_mac.feature_variant = "v6_refined_features_f32_zero_bg"`, but no
checked-in training config uses it by default.

## What Changed

Inherited from `f32_reduce`:

- F32 feature/color gradient atomics use the reduction helper path.
- `F == 3` backward uses the reduced feature-gradient path rather than the
  older generic per-channel atomics.
- Python/C++ can skip allocating and writing `g_colors` when feature/color
  gradients are not required.
- The benchmark scripts and active-tile scheduling surface are inherited.

Added in this fork:

- Python marks exactly-zero feature backgrounds in metadata bit `2`.
- Existing skip-color-gradient metadata remains bit `1`.
- Fast and active forward kernels check the zero-background bit and skip only
  `add_background_tail(...)` when the feature background is exactly zero.
- The kernels still explicitly initialize valid output pixels with
  `zero_pixel(...)` and write background for invalid/empty pixels.

Rejected during this fork:

- Replacing `torch::empty` + explicit Metal init with `torch::zeros` and
  relying on implicit zero state before shader `+=` writes. That shortcut
  caused repeated-call parity failures and was removed. Do not revive it
  without a repeated-call parity gate.

This fork does not include feature staging, lookup-table feature IDs, fixed-bin
IDs, or trainer microbatching. Those remain separate experiments.

## Verified Gates

Run from the Dynaworld root after building the extension:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_zero_bg/tests/feature_contract_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_zero_bg/tests/alpha_output_check.py
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
F=64 feature grad active_policy=off max_abs=1.1641532e-10
F=32 feature grad active_policy=on max_abs=2.3283064e-10
F=32 no-NaN smoke active_policy=off: ok
F=32 no-NaN smoke active_policy=on: ok
Test A passed.
Test B passed.
Test C passed.
Test D passed.
Test E passed.
Test F passed.
```

A direct parent-vs-fork MPS parity probe at `B4/G512/H64/W64/F32` reported zero
feature/alpha output diff for active off/on. The largest gradient diff in the
latest rerun was `2.98e-08`.

## Benchmark Snapshot

Local MPS, `GSP_CHUNK=64`, `GSP_FAST_CAP=2048`,
`B=16,G=8192,F=32`, `case=medium_sigma_3_8`, fwd+bwd:

| Shape | Active policy | Variant | Forward ms | Backward ms | Total mean ms | Artifact |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 256px | off | `f32_reduce` | 98.0 | 287.9 | 386.0 | `2026-05-07_256_f32_b16_g8192_zero_bg_reduce_active_off.json` |
| 256px | off | `f32_zero_bg` | 89.4 | 271.7 | 361.1 | `2026-05-07_256_f32_b16_g8192_zero_bg_active_off.json` |
| 256px | on | `f32_reduce` | 140.5 | 331.5 | 472.0 | `2026-05-07_256_f32_b16_g8192_zero_bg_reduce_active_on.json` |
| 256px | on | `f32_zero_bg` | 139.8 | 314.8 | 454.6 | `2026-05-07_256_f32_b16_g8192_zero_bg_active_on.json` |
| 512px | off | `f32_reduce` | 133.3 | 477.4 | 610.7 | `2026-05-07_512_f32_b16_g8192_zero_bg_reduce_active_off.json` |
| 512px | off | `f32_zero_bg` | 127.1 | 429.6 | 556.7 | `2026-05-07_512_f32_b16_g8192_zero_bg_active_off.json` |
| 512px | on | `f32_reduce` | 293.3 | 520.1 | 813.4 | `2026-05-07_512_f32_b16_g8192_zero_bg_reduce_active_on.json` |
| 512px | on | `f32_zero_bg` | 289.7 | 472.2 | 761.9 | `2026-05-07_512_f32_b16_g8192_zero_bg_active_on.json` |

Read: this fork is a bounded local timing win over `f32_reduce` on these rows.
Because the intended shader change is a forward tail-write skip, the backward
delta should be treated as same-session timing evidence rather than a proven
mechanism. This is not a baseline replacement.

## Metal References

- https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups
- https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes
- https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)
- https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth
- https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon
- https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy
