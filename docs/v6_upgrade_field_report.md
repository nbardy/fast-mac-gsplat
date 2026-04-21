# v6 Upgrade Field Report

Date: 2026-04-21

## Summary

The chief-scientist `torch_metal_gsplat_v6_upgrade.tar.gz` bundle is now
preserved as:

```text
source_artifacts/torch_metal_gsplat_v6_upgrade.tar.gz
variants/v6_upgrade/
```

The uploaded v7 hardware bundle had the same SHA-256 as the source artifact
already preserved in this repo, so the existing locally fixed `variants/v7`
was not overwritten.

## Integration

`variants/v6_upgrade` is kept alongside `variants/v6` rather than replacing it.
The current `variants/v6` branch already had later local changes, including the
tighter active-policy behavior and pair-budget chunking, so replacing it with
the raw upgrade handoff would have been a regression.

The full benchmark runner now supports:

```text
v6_upgrade_direct
v6_upgrade_auto
```

These rows are subprocess-isolated. That matters because the handoff still uses
the Python package name `torch_gsplat_bridge_v6` and the Torch op namespace
`gsplat_metal_v6`, which would collide with `variants/v6` in one Python process.

## Fixes Applied

The same saturated-backward barrier bug found in v5 also existed in v6 and the
v6 upgrade handoff. The affected kernels looped over per-pixel `end_i` while
using `threadgroup_barrier()`.

Patched in both `variants/v6` and `variants/v6_upgrade`:

```text
tile_fast_backward_saved:     reverse loop uses uniform stop_count
tile_active_backward_saved:   reverse loop uses uniform stop_count
tile_overflow_backward:       reverse loop uses uniform count
per-pixel early stop:         remains as global_i < end_i inside the loop
```

This keeps barrier control uniform across the threadgroup while preserving the
early-stop math for each pixel.

## Validation

Builds:

```bash
cd variants/v6
python setup.py build_ext --inplace

cd ../v6_upgrade
python setup.py build_ext --inplace
```

The v6-upgrade build succeeded with two existing unused-variable warnings in
`gsplat_metal.mm`.

Reference checks:

```bash
cd variants/v6
python tests/reference_check.py

cd ../v6_upgrade
python tests/reference_check.py
```

Both pass tiny B=1/B=2 checks and the added saturated 64-splat check. The
saturated checks cover direct and active paths.

Representative saturated max errors:

```text
image:          2.086162567138672e-07
means grad:     2.3283064365386963e-10
conics grad:    1.1920928955078125e-07
colors grad:    3.725290298461914e-09
opacities grad: 9.313225746154785e-10
```

## Smoke Benchmark

Command:

```bash
python benchmarks/benchmark_full_matrix.py \
  --resolutions 128x128 \
  --splats 64 \
  --batch-sizes 1 \
  --distributions microbench_uniform_random,clustered_hot_tiles \
  --renderers v6_direct,v6_upgrade_direct,v6_upgrade_auto,v7_hardware \
  --modes forward,forward_backward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 90 \
  --output-md benchmarks/full_rasterizer_benchmark_v6_upgrade_smoke.md
```

Result: all 16 cells completed with `status=ok`.

Small smoke winners:

| Distribution | Mode | Winner | Mean ms |
|---|---|---|---:|
| uniform random | forward | v6_direct | 4.296 |
| uniform random | forward+backward | v6_direct | 6.133 |
| clustered hot tiles | forward | v6_direct | 4.231 |
| clustered hot tiles | forward+backward | v7_hardware | 9.323 |

The smoke is only an integration check. It is too small and too short to decide
default renderer policy.

## Current Recommendation

Keep `variants/v6` as the current v6 baseline. Use `variants/v6_upgrade` as a
preserved source handoff and benchmark target. Do not overwrite the existing v6
line with the upgrade handoff unless a larger benchmark shows that the upgrade
beats the locally evolved v6 branch.

## Deep Matrix Follow-Up

A full v6-family matrix was run after this smoke:

```text
docs/v6_upgrade_deep_benchmark_report.md
benchmarks/full_rasterizer_benchmark_v6_upgrade_deep.md
benchmarks/full_rasterizer_benchmark_v6_upgrade_deep.jsonl
```

That run covered 960 cells across four resolutions, three splat counts, B=1/B=4,
five projected-splat distributions, forward and forward+backward modes, and the
four v6-family renderers. All cells completed with `status=ok`.

The deep result keeps the same recommendation. Local `v6_direct` is still the
best default, but `v6_upgrade_direct` is competitive in larger
forward+backward workloads, especially at 64k splats.
