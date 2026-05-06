# v6_refined_features_f32_fixedbin Engineering Notes

This is an experimental fork of `variants/v6_refined_features_f32_gradcache`.
It keeps the stable `v6_refined_features` baseline and the earlier optimization
forks untouched while testing one host/kernel change: fixed-cap per-tile bin
storage for no-overflow fast-path rows.

Do not point production configs at this fork by default. Promote it only after a
full train/heldout-quality parity run.

## Namespace

- Python package: `torch_gsplat_bridge_v6_refined_features_f32_fixedbin`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features_f32_fixedbin`
- Metal source: `csrc/metal/gsplat_v6_refined_features_f32_fixedbin_kernels.metal`
- output API: `(features, accumulated_alpha)`

Dynaworld can select this fork with
`render.fast_mac.feature_variant = "v6_refined_features_f32_fixedbin"`, but no
checked-in training config uses it by default.

## What Changed

- Inherits the `f32_gradcache` direct-fast-backward path.
- Replaces the exact-length bin allocation path:
  `count_tiles -> cumsum -> cat -> clone -> final_offset.item() -> empty(N)`.
- Allocates `binned_ids` as `tile_count * max_fast_pairs` and fills
  `tile_offsets[t] = t * max_fast_pairs` with a tiny Metal init kernel.
- `emit_binned_ids` writes only inside the tile's fixed span and drops writes
  beyond `max_fast_pairs` instead of corrupting the next tile.
- Python raises on any tile overflow. This fork is intentionally no-overflow
  only; use `f32_reduce`/`f32_accum`/`f32_gradcache` when overflow fallback is
  required.

The tradeoff is explicit: remove a host/MPS sync and exact-size allocation at
the cost of a larger fixed ID buffer. At `512px/B16/tile16/cap2048`, the fixed
buffer is about `128 MiB` of int32 IDs.

## Correctness Gates

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_refined_features_f32_fixedbin
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_fixedbin/tests/feature_contract_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f32_fixedbin/tests/alpha_output_check.py
```

Observed:

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

Trainer fixed-render parity versus stable `v6_refined_features`:

| Gate | max feature diff | max alpha diff | max RGB diff | max sequence grad diff | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| 256px train forward | 0.0 | 0.0 | 0.0 | n/a | `multicam256_v6_vs_f32_fixedbin_fixed_render_parity_seed0.json` |
| 256px heldout forward | 0.0 | 0.0 | 0.0 | n/a | `multicam256_heldout_v6_vs_f32_fixedbin_fixed_render_parity_seed0.json` |
| 128px train backward | 0.0 | 0.0 | 0.0 | 8.15e-10 | `multicam128_train_v6_vs_f32_fixedbin_fixed_render_grad_parity_seed0.json` |
| 128px heldout backward | 0.0 | 0.0 | 0.0 | 1.14e-09 | `multicam128_heldout_v6_vs_f32_fixedbin_fixed_render_grad_parity_seed0.json` |

## Benchmarks

Local MPS, `GSP_CHUNK=64,GSP_FAST_CAP=2048,GSP_FEATURE_CAP=64`,
`G=8192`, `case=medium_sigma_3_8`, `warmup=1`, `iters=2`:

| Shape | `f32_reduce` | `f32_accum` | `f32_gradcache` | `f32_fixedbin` | Read | Artifact |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 128px B16/F32 | 213.2ms | n/a | 197.1ms | 176.7ms | fixedbin wins | `2026-05-07_128_f32_b16_fixedbin_smoke_matrix.jsonl` |
| 256px B16/F32 | 493.2ms | 520.0ms | 497.3ms | 523.4ms | fixedbin loses | `2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 256px B32/F32 | 1003.4ms | 1048.0ms | 968.9ms | 996.6ms | fixedbin beats reduce, loses gradcache | `2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 512px B16/F32 | 855.4ms | 706.5ms | 716.4ms | 501.8ms | fixedbin wins target row | `2026-05-07_512_f32_b16_fixedbin_matrix.jsonl` |
| 256px B16/F64 | 779.8ms | 726.0ms | n/a | 718.1ms | fixedbin slight win | `2026-05-07_256_f64_b16_fixedbin_matrix.jsonl` |

Seeded 256px trainer fixed-render graph, `warmup=2`, `iters=4`, same local
window:

| Variant | total mean ms | total median ms | raster fwd ms | backward ms | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 725.4 | 718.8 | 69.4 | 572.2 | `multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json` |
| `v6_refined_features_f32_accum` | 696.9 | 691.5 | 76.3 | 537.2 | `multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json` |
| `v6_refined_features_f32_gradcache` | 814.7 | 828.0 | 79.3 | 638.4 | `multicam256_f32_f32_gradcache_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json` |
| `v6_refined_features_f32_fixedbin` | 696.1 | 687.7 | 68.8 | 544.4 | `multicam256_f32_f32_fixedbin_fixed_render_seed0_warm2_iters4.json` |

Read: fixedbin is a real target-row synthetic win and ties `f32_accum` on the
256px trainer fixed-render graph in the latest local window. It is not a default
until a real train/heldout-quality run confirms the no-overflow path behaves
well over optimizer time.
