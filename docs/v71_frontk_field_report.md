# v7.1 Front-K Field Report

Date: 2026-04-22

## Summary

The `torch_metal_gsplat_v71_frontk_handoff.tar.gz` handoff is preserved as:

```text
source_artifacts/torch_metal_gsplat_v71_frontk_handoff.tar.gz
variants/v7_frontk/
```

Archive SHA-256:

```text
1b1837545fc5e50530823eae2998c2fa7bec035e046a5c78654d30c638fcc93e
```

The root benchmark runner now supports:

```text
v7_frontk_k2
v7_frontk_k4
v7_frontk_k8
```

## Local Fixes Applied

```text
Metal compile: replaced program-scope constexpr with a macro.
Fragment shader: removed duplicate [[position]] argument.
Draw order: restored far-to-near draw order for source-over blending parity.
Ellipse bounds: restored full pixel-space support bounds.
Tests/benchmark: added direct script sys.path setup.
v7.1 sample benchmark: disabled debug-state collection for MPS timing because it recomputes front-K in Python.
```

## Correctness

Bundled Python reference test:

```bash
cd variants/v7_frontk
python3 tests/reference_exactness.py
```

Result: all reference cases passed. Max image error was `0.0`; max gradient
errors were below `6e-8` in the clamp-overflow case.

MPS smoke against dense Torch reference:

```text
image max error:      5.960464477539063e-08
means grad max error: 2.3283064365386963e-10
conics grad max error: 4.656612873077393e-10
colors grad max error: 1.862645149230957e-09
opacity grad max error: 2.7939677238464355e-09
```

Conclusion: v7.1 fixes the major v7-finished backward correctness issue on the
small MPS smoke. Large target accuracy is skipped because dense Torch reference
would be too large.

## 512x512 / 6k

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 512x512 \
  --splats 6000 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers v7_frontk_k2,v7_frontk_k4,v7_frontk_k8 \
  --modes forward,forward_backward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 240 \
  --output-md benchmarks/full_rasterizer_benchmark_v71_frontk_512_6k.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v71_frontk_512_6k.jsonl
```

`front_k=2` was fastest in every forward+backward cell. B=1 comparison:

| Distribution | Pre-v7.1 Exact Best | Best ms | v7.1 k2 ms | Delta vs Exact Best | v7 Finished ms | v7.1 Bwd vs v7 Finished Bwd |
|---|---|---:|---:|---:|---:|---:|
| microbench_uniform_random | `v6_upgrade_direct` | 38.898 | 55.680 | +43.1% | 40.364 | +21.0% |
| sparse_screen | `v6_direct` | 27.760 | 43.600 | +57.1% | 37.432 | -59.9% |
| clustered_hot_tiles | `v6_refined_auto` | 95.208 | 54.997 | -42.2% | 51.370 | -43.4% |
| layered_depth | `v5_batched` | 28.228 | 56.749 | +101.0% | 41.687 | -16.4% |
| overflow_adversarial | `v6_upgrade_auto` | 634.802 | 587.924 | -7.4% | 155.760 | +395.9% |

Negative deltas mean v7.1 is faster. `v7_finished_hardware` is listed only as a
speed reference because its backward gradients are not correct.

For B=4, v7.1 k2 was faster than the pre-v7.1 exact best in clustered-hot-tiles
only: `202.522 ms` vs `611.650 ms` (`-66.9%`). It regressed microbench,
sparse-screen, layered-depth, and overflow-adversarial by `+54.1%` to `+441.2%`.

## 4096x4096 / 64k

Warm steady-state v7.1 command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 4096x4096 \
  --splats 65536 \
  --batch-sizes 1 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers v7_frontk_k2 \
  --modes forward_backward \
  --warmup 1 \
  --iters 1 \
  --timeout-sec 360 \
  --output-md benchmarks/full_rasterizer_benchmark_v71_frontk_4k64k_steady.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v71_frontk_4k64k_steady.jsonl
```

| Distribution | Pre-v7.1 Exact Best | Best ms | v7.1 Total ms | v7.1 Fwd ms | v7.1 Bwd ms | Delta vs Exact Best |
|---|---|---:|---:|---:|---:|---:|
| microbench_uniform_random | `v6_upgrade_direct` | 62.657 | 35852.168 | 17110.397 | 18741.771 | +57119.7% |
| sparse_screen | `v6_direct` | 52.268 | 21808.049 | 20033.602 | 1774.447 | +41623.5% |
| clustered_hot_tiles | `v6_upgrade_direct` | 176.944 | 23069.772 | 22319.137 | 750.635 | +12937.9% |
| layered_depth | `v6_upgrade_direct` | 74.644 | 24497.635 | 19704.240 | 4793.395 | +32719.3% |
| overflow_adversarial | `v6_auto` | 1089.448 | 27008.610 | 20388.068 | 6620.542 | +2379.1% |

Against `v7_finished_hardware` on the two available 4K backward probes:

| Distribution | v7 Finished Total ms | v7.1 Total ms | Total Delta | v7 Finished Bwd ms | v7.1 Bwd ms | Bwd Delta |
|---|---:|---:|---:|---:|---:|---:|
| microbench_uniform_random | 21440.399 | 35852.168 | +67.2% | 21185.617 | 18741.771 | -11.5% |
| clustered_hot_tiles | 20330.785 | 23069.772 | +13.5% | 20118.694 | 750.635 | -96.3% |

## Recommendation

v7.1 is a useful correctness proof for hardware backward, but it is not the
fastest training path.

At `512x512 / 6k`, v7.1 k2 is worth keeping as an exact contender for
clustered-hot-tiles and B=1 overflow-adversarial. At `4096x4096 / 64k`, it is
not viable as the default because the front-K capture pass dominates forward
time at roughly `17-22 s` per pass.

Ask for a v7.2 hardware handoff that keeps the corrected backward math but does
not require a full per-pixel, all-splats front-K capture pass. The target should
be a tile/bin-derived front-K state, or another sparse visibility state produced
while rasterizing, so 4K forward does not become the bottleneck.
