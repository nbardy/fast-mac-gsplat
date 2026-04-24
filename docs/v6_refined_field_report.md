# v6 Refined Field Report

Date: 2026-04-22

## Summary

The `torch_metal_gsplat_v6_refined.tar.gz` bundle is preserved as:

```text
source_artifacts/torch_metal_gsplat_v6_refined.tar.gz
variants/v6_refined/
```

`variants/v6_refined` is kept beside `variants/v6` and
`variants/v6_upgrade`; it does not replace either baseline.

## Integration

The root matrix benchmark now supports:

```text
v6_refined_direct
v6_refined_auto
```

Like `v6_upgrade`, the refined handoff imports as `torch_gsplat_bridge_v6`.
The root benchmark runs each renderer row in a subprocess, so these variants can
be compared in one matrix without same-process import collisions.

## Local Fixes Applied

The raw refined archive had the older saturated-backward loop form in three
Metal kernels: the reverse loop was controlled by per-pixel `end_i` while using
`threadgroup_barrier()`.

The integrated `variants/v6_refined` applies the same local fix already used in
`variants/v6` and `variants/v6_upgrade`:

```text
tile_fast_backward_saved:     reverse loop uses uniform stop_count
tile_active_backward_saved:   reverse loop uses uniform stop_count
tile_overflow_backward:       reverse loop uses uniform count
per-pixel early stop:         remains as global_i < end_i inside the loop
```

The refined `__init__.py` also exports `ProjectedGaussianRasterizer` to keep the
v6-family API surface stable.

After this integration fix, the checked-in runtime sources match
`variants/v6_upgrade`; the remaining differences are package metadata and
handoff documentation. Treat small timing differences between v6-upgrade and
v6-refined as measurement noise unless a larger shuffled matrix shows otherwise.

## Validation

Build:

```bash
cd variants/v6_refined
python3 setup.py build_ext --inplace
```

Reference check:

```bash
cd variants/v6_refined
python3 tests/reference_check.py
```

The reference check passed B=1/B=2 direct/active/auto cases and saturated
64-splat backward cases. Representative max errors:

```text
small image:       5.960464477539063e-08
small gradients:   <= 1.862645149230957e-09
saturated image:   2.086162567138672e-07
saturated conics:  1.1920928955078125e-07
```

## Smoke Matrix

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 128x128 \
  --splats 64 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers v6_direct,v6_auto,v6_upgrade_direct,v6_upgrade_auto,v6_refined_direct,v6_refined_auto \
  --modes forward,forward_backward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 90 \
  --accuracy \
  --accuracy-max-work-items 5000000 \
  --output-md benchmarks/full_rasterizer_benchmark_v6_refined_smoke.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v6_refined_smoke.jsonl
```

Result: all 120 cells completed with `status=ok`; all 120 cells produced dense
Torch reference accuracy measurements.

Worst measured dense-reference errors across the smoke:

```text
image max abs error: 8.982419967651367e-05
grad max abs error:  0.017425060272216797
```

The larger gradient error appears in the overflow-adversarial backward rows for
all v6-family renderers in this smoke, so it is not unique to v6-refined.

Small smoke winners:

| B | Distribution | Mode | Winner | Mean ms |
|---:|---|---|---|---:|
| 1 | clustered hot tiles | forward | v6_direct | 5.572 |
| 1 | clustered hot tiles | forward+backward | v6_direct | 7.688 |
| 1 | layered depth | forward | v6_direct | 4.480 |
| 1 | layered depth | forward+backward | v6_direct | 7.421 |
| 1 | uniform random | forward | v6_direct | 4.936 |
| 1 | uniform random | forward+backward | v6_direct | 7.092 |
| 1 | overflow adversarial | forward | v6_direct | 5.517 |
| 1 | overflow adversarial | forward+backward | v6_refined_direct | 8.746 |
| 1 | sparse screen | forward | v6_upgrade_auto | 5.436 |
| 1 | sparse screen | forward+backward | v6_direct | 7.900 |
| 4 | clustered hot tiles | forward | v6_upgrade_direct | 5.732 |
| 4 | clustered hot tiles | forward+backward | v6_direct | 8.789 |
| 4 | layered depth | forward | v6_refined_auto | 5.706 |
| 4 | layered depth | forward+backward | v6_upgrade_auto | 9.112 |
| 4 | uniform random | forward | v6_refined_auto | 5.617 |
| 4 | uniform random | forward+backward | v6_direct | 8.412 |
| 4 | overflow adversarial | forward | v6_refined_auto | 6.151 |
| 4 | overflow adversarial | forward+backward | v6_direct | 10.183 |
| 4 | sparse screen | forward | v6_upgrade_auto | 6.223 |
| 4 | sparse screen | forward+backward | v6_upgrade_auto | 8.373 |

In this small smoke, local `v6_direct` remains the strongest overall baseline,
winning 11 of 20 workload groups. `v6_refined_auto` wins three B=4 forward-only
groups, and `v6_refined_direct` wins the B=1 overflow forward+backward group.
This is enough to keep v6-refined in the matrix, but not enough to change the
default recommendation without a larger run.
