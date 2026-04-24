# Agent Notes: Apr 22 Performance Metrics

Date: 2026-04-22

## Scope

These notes compare fastest renderer paths for two target regimes:

- `512x512`, `6000` splats
- `4096x4096`, `65536` splats

The tables use `v6_direct` as the baseline. Positive percentages mean the
selected fastest path is faster than `v6_direct`:

```text
faster % = (v6_direct_ms / best_ms - 1) * 100
```

`v7_hardware` is forward-only/eval-oriented here. It is not recommended as a
training path because backward is orders of magnitude slower at large sizes.

## Source Reports

```text
benchmarks/full_rasterizer_benchmark_target_512_6k.md
benchmarks/full_rasterizer_benchmark_target_512_6k.jsonl
benchmarks/full_rasterizer_benchmark_target_512_6k_v7_forward.md
benchmarks/full_rasterizer_benchmark_target_512_6k_v7_forward.jsonl
benchmarks/full_rasterizer_benchmark_v6_upgrade_deep.md
benchmarks/full_rasterizer_benchmark_v6_upgrade_deep.jsonl
benchmarks/full_rasterizer_benchmark_target_4k64k_v5.md
benchmarks/full_rasterizer_benchmark_target_4k64k_v5.jsonl
benchmarks/full_rasterizer_benchmark_target_4k64k_v7_forward.md
benchmarks/full_rasterizer_benchmark_target_4k64k_v7_forward.jsonl
benchmarks/full_rasterizer_benchmark_v7_finished_512_6k.md
benchmarks/full_rasterizer_benchmark_v7_finished_512_6k.jsonl
benchmarks/full_rasterizer_benchmark_v7_finished_4k64k_backward_probe.md
benchmarks/full_rasterizer_benchmark_v7_finished_4k64k_backward_probe.jsonl
benchmarks/full_rasterizer_benchmark_v71_frontk_512_6k.md
benchmarks/full_rasterizer_benchmark_v71_frontk_512_6k.jsonl
benchmarks/full_rasterizer_benchmark_v71_frontk_4k64k_steady.md
benchmarks/full_rasterizer_benchmark_v71_frontk_4k64k_steady.jsonl
benchmarks/full_rasterizer_benchmark_v72_tiled_512_6k.md
benchmarks/full_rasterizer_benchmark_v72_tiled_512_6k.jsonl
benchmarks/full_rasterizer_benchmark_v72_tiled_4k64k.md
benchmarks/full_rasterizer_benchmark_v72_tiled_4k64k.jsonl
```

## 512x512 / 6k Forward

| B | Distribution | Fastest | Best ms | v6_direct ms | Faster vs v6_direct |
|---:|---|---|---:|---:|---:|
| 1 | clustered_hot_tiles | `v7_hardware` | 10.593 | 231.549 | +2085.8% |
| 1 | layered_depth | `v5_batched` | 6.254 | 68.583 | +996.7% |
| 1 | microbench_uniform_random | `v7_hardware` | 7.677 | 77.502 | +909.5% |
| 1 | overflow_adversarial | `v7_hardware` | 24.690 | 3588.763 | +14435.3% |
| 1 | sparse_screen | `v7_hardware` | 8.030 | 25.706 | +220.1% |
| 4 | clustered_hot_tiles | `v7_hardware` | 21.338 | 598.346 | +2704.2% |
| 4 | layered_depth | `v5_batched` | 10.752 | 17.222 | +60.2% |
| 4 | microbench_uniform_random | `v7_hardware` | 15.051 | 24.428 | +62.3% |
| 4 | overflow_adversarial | `v7_hardware` | 82.020 | 10712.930 | +12961.3% |
| 4 | sparse_screen | `v6_direct` | 15.010 | 15.010 | +0.0% |

## 512x512 / 6k Forward+Backward

| B | Distribution | Fastest | Best ms | v6_direct ms | Faster vs v6_direct |
|---:|---|---|---:|---:|---:|
| 1 | clustered_hot_tiles | `v6_refined_auto` | 95.208 | 278.464 | +192.5% |
| 1 | layered_depth | `v5_batched` | 28.228 | 79.674 | +182.2% |
| 1 | microbench_uniform_random | `v6_upgrade_direct` | 38.898 | 64.735 | +66.4% |
| 1 | overflow_adversarial | `v6_upgrade_auto` | 634.802 | 4566.481 | +619.4% |
| 1 | sparse_screen | `v6_direct` | 27.760 | 27.760 | +0.0% |
| 4 | clustered_hot_tiles | `v6_refined_auto` | 611.650 | 727.797 | +19.0% |
| 4 | layered_depth | `v5_batched` | 37.673 | 105.530 | +180.1% |
| 4 | microbench_uniform_random | `v6_refined_auto` | 30.562 | 70.282 | +130.0% |
| 4 | overflow_adversarial | `v6_refined_auto` | 1503.726 | 2336.763 | +55.4% |
| 4 | sparse_screen | `v6_refined_direct` | 38.187 | 165.936 | +334.5% |

## 4096x4096 / 64k Forward

| B | Distribution | Fastest | Best ms | v6_direct ms | Faster vs v6_direct |
|---:|---|---|---:|---:|---:|
| 1 | clustered_hot_tiles | `v6_upgrade_direct` | 127.332 | 131.833 | +3.5% |
| 1 | layered_depth | `v5_batched` | 12.409 | 14.572 | +17.4% |
| 1 | microbench_uniform_random | `v6_direct` | 11.752 | 11.752 | +0.0% |
| 1 | overflow_adversarial | `v7_hardware` | 492.732 | 585.299 | +18.8% |
| 1 | sparse_screen | `v6_direct` | 10.650 | 10.650 | +0.0% |
| 4 | clustered_hot_tiles | `v7_hardware` | 404.342 | 1381.861 | +241.8% |
| 4 | layered_depth | `v6_direct` | 39.768 | 39.768 | +0.0% |
| 4 | microbench_uniform_random | `v6_direct` | 33.943 | 33.943 | +0.0% |
| 4 | overflow_adversarial | `v7_hardware` | 2029.351 | 5253.447 | +158.9% |
| 4 | sparse_screen | `v6_direct` | 25.037 | 25.037 | +0.0% |

## 4096x4096 / 64k Forward+Backward

| B | Distribution | Fastest | Best ms | v6_direct ms | Faster vs v6_direct |
|---:|---|---|---:|---:|---:|
| 1 | clustered_hot_tiles | `v6_upgrade_direct` | 176.944 | 184.195 | +4.1% |
| 1 | layered_depth | `v6_upgrade_direct` | 74.644 | 75.928 | +1.7% |
| 1 | microbench_uniform_random | `v6_upgrade_direct` | 62.657 | 69.749 | +11.3% |
| 1 | overflow_adversarial | `v6_auto` | 1089.448 | 1468.462 | +34.8% |
| 1 | sparse_screen | `v6_direct` | 52.268 | 52.268 | +0.0% |
| 4 | clustered_hot_tiles | `v6_direct` | 684.783 | 684.783 | +0.0% |
| 4 | layered_depth | `v6_auto` | 269.955 | 271.098 | +0.4% |
| 4 | microbench_uniform_random | `v6_auto` | 229.856 | 231.469 | +0.7% |
| 4 | overflow_adversarial | `v5_batched` | 4725.260 | 8928.235 | +88.9% |
| 4 | sparse_screen | `v6_upgrade_direct` | 188.677 | 197.872 | +4.9% |

## Interpretation

At `512x512 / 6k`, fastest path selection changes frequently. For forward-only
evaluation, `v7_hardware` wins most cases. For forward+backward training,
`v6_refined_auto`, `v6_upgrade_*`, `v5_batched`, and `v6_direct` each win
different distributions.

At `4096x4096 / 64k`, fastest training paths are more stable. `v6_direct`,
`v6_auto`, and `v6_upgrade_direct` cover most forward+backward cases. The large
B=4 overflow-adversarial training case is the exception: `v5_batched` is fastest
there.

## v7.1 Front-K Update

The v7.1 front-K handoff adds a hardware backward path and passes the small MPS
forward/backward exactness smoke after local Metal compile/order/bounds fixes.
It is not forward-only; it has backward, and the gradient path is materially
more correct than `v7_finished_hardware`.

At `512x512 / 6k`, `front_k=2` was fastest in every v7.1 cell. For B=1
forward+backward:

| Distribution | Previous Exact Best | Best ms | v7.1 k2 ms | Delta |
|---|---|---:|---:|---:|
| clustered_hot_tiles | `v6_refined_auto` | 95.208 | 54.997 | -42.2% |
| layered_depth | `v5_batched` | 28.228 | 56.749 | +101.0% |
| microbench_uniform_random | `v6_upgrade_direct` | 38.898 | 55.680 | +43.1% |
| overflow_adversarial | `v6_upgrade_auto` | 634.802 | 587.924 | -7.4% |
| sparse_screen | `v6_direct` | 27.760 | 43.600 | +57.1% |

At `4096x4096 / 64k`, v7.1 is not viable as the fastest training path. The
front-K capture pass dominates forward time:

| Distribution | Previous Exact Best | Best ms | v7.1 k2 ms | Delta |
|---|---|---:|---:|---:|
| clustered_hot_tiles | `v6_upgrade_direct` | 176.944 | 23069.772 | +12937.9% |
| layered_depth | `v6_upgrade_direct` | 74.644 | 24497.635 | +32719.3% |
| microbench_uniform_random | `v6_upgrade_direct` | 62.657 | 35852.168 | +57119.7% |
| overflow_adversarial | `v6_auto` | 1089.448 | 27008.610 | +2379.1% |
| sparse_screen | `v6_direct` | 52.268 | 21808.049 | +41623.5% |

Compared with `v7_finished_hardware` on the two measured 4K backward probes,
v7.1 improves backward kernel time but loses end-to-end time because capture
moved into forward:

| Distribution | Total Delta vs v7 Finished | Backward Delta vs v7 Finished |
|---|---:|---:|
| microbench_uniform_random | +67.2% | -11.5% |
| clustered_hot_tiles | +13.5% | -96.3% |

Conclusion: ask for a new v7.2-style handoff with the v7.1 backward math, but
with sparse/tile-derived front-K state instead of a full per-pixel all-splats
capture pass.

## v7.2 Tiled-Capture Update

The v7.2 tiled-capture handoff does fix the main v7.1 scaling flaw. It uses
tile-local candidate lists for front-K capture and overflow replay instead of
scanning all splats per pixel. After the same Metal compile/order/bounds fixes,
the small MPS exactness smoke passed:

```text
image max error:      8.940696716308594e-08
grad max error:       1.4901161193847656e-07 in root harness smoke
```

At `512x512 / 6k`, v7.2 is often the fastest exact hardware-backward contender:

| B | Distribution | Previous Exact Best ms | v7.1 Best ms | v7.2 Best ms | v7.2 Delta vs Previous Exact Best |
|---:|---|---:|---:|---:|---:|
| 1 | clustered_hot_tiles | 95.208 | 54.997 | 27.995 | -70.6% |
| 1 | layered_depth | 28.228 | 56.749 | 29.897 | +5.9% |
| 1 | microbench_uniform_random | 38.898 | 55.680 | 23.074 | -40.7% |
| 1 | overflow_adversarial | 634.802 | 587.924 | 570.893 | -10.1% |
| 1 | sparse_screen | 27.760 | 43.600 | 22.533 | -18.8% |
| 4 | clustered_hot_tiles | 611.650 | 202.522 | 92.275 | -84.9% |
| 4 | layered_depth | 37.673 | 203.862 | 97.617 | +159.1% |
| 4 | microbench_uniform_random | 30.562 | 162.835 | 48.531 | +58.8% |
| 4 | overflow_adversarial | 1503.726 | 2316.551 | 2300.913 | +53.0% |
| 4 | sparse_screen | 38.187 | 149.303 | 37.702 | -1.3% |

At `4096x4096 / 64k`, v7.2 removes the v7.1 cliff but is still slower than
the established training paths:

| Distribution | Previous Exact Best ms | v7.1 k2 ms | v7.2 Best ms | v7.2 Delta vs Previous Exact Best | v7.2 Delta vs v7.1 |
|---|---:|---:|---:|---:|---:|
| clustered_hot_tiles | 176.944 | 23069.772 | 459.409 | +159.6% | -98.0% |
| layered_depth | 74.644 | 24497.635 | 487.923 | +553.7% | -98.0% |
| microbench_uniform_random | 62.657 | 35852.168 | 382.507 | +510.5% | -98.9% |
| overflow_adversarial | 1089.448 | 27008.610 | 6525.487 | +499.0% | -75.8% |
| sparse_screen | 52.268 | 21808.049 | 314.068 | +500.9% | -98.6% |

Conclusion: v7.2 is the right direction and should replace v7.1 for further
hardware-backward experiments. It should not replace v5/v6-family defaults for
4K training yet. The next ask should be v7.3 with GPU-side binning/saved-state
production and no CPU round trip for forward image/front-K state.

Recommended defaults:

- General training default: `v6_direct`
- 512x512 / 6k B=4 training: try `v6_refined_auto`
- 4K / 64k normal training: use `v6_direct` or `v6_upgrade_direct`
- 4K / 64k overflow-heavy B=4 training: use `v5_batched`
- Forward-only/eval: include `v7_hardware` in selection, but avoid it for backward
- v7.1 front-K: superseded by v7.2 for hardware-backward experiments
- v7.2 tiled-capture: useful at 512x512 / 6k and as the next research branch, not the 4K default

## v7.3 Hybrid V5-Style Branch

Created `variants/v7_hybrid_v5style` from v7.2 as a separate branch for the
"v5 shape plus v7.2 hardware raster" experiment. This first pass keeps the v7.2
hardware raster and tiled front-K kernels, then adds v5-style batch strategy
scaffolding:

- renderer names: `v73_hybrid_k2`, `v73_hybrid_k4`, `v73_hybrid_k8`
- optional suffixes: `_t32`, `_serial`, `_flatten`, `_auto`
- runtime guardrails: positive dimensions, `front_k <= 8`, exact overflow replay required
- batch chunking: `auto`, `serial`, or `flatten` using tile/G launch limits

This is not expected to fix the `4096x4096 / 64k / B=1` gap by itself. The 4K
gap comes from v7.2 still doing CPU tile bin construction and CPU saved-state
round trips. The next real speed fixes are:

- MPS/GPU tile binning from the v5 design
- no CPU `out_cpu`/front-state readback in eval
- GPU-resident saved state for training, or a v5/v6 compute train path
- overflow-tile-only fallback rather than broad per-pixel state traffic

Plan file: `variants/v7_hybrid_v5style/docs/v73_hybrid_v5style_plan.md`.

Smoke results:

| Case | v7.2 k2 | v7.3 hybrid k2 | Delta |
|---|---:|---:|---:|
| `512x512 / 6k / B=4` uniform F+B | 52.203 ms | 49.134 ms | -5.9% |
| `512x512 / 6k / B=4` clustered F+B | 92.820 ms | 94.950 ms | +2.3% |
| `4096x4096 / 64k / B=1` uniform F+B | 412.732 ms | 451.550 ms | +9.4% |

The 4K smoke also measured `v5_batched` at 63.273 ms, making v7.3 hybrid
613.6% slower than v5 in that cell. The hybrid wrapper is useful scaffolding,
not the 4K solution.

### v7.3 Train/Eval Routing Fix

Applied the two practical key fixes:

- no-grad/eval calls use `gsplat_metal_v73.forward_eval`, which skips front-K
  capture, CPU tile-bin construction, and backward saved-state allocation
- gradient/training calls default to `train_backend="auto"`, which routes to
  the sibling v5 compute training path when available; `_hwtrain` forces the
  old v7.2-style hardware backward route

"v5/v6-style compute train path" means the training renderer is not the
fixed-function hardware raster pipeline. It is the MPS/Metal compute-kernel
shape from v5/v6: tile binning, tile-local compute rendering, compact saved
tile state, and backward from that saved state.

New training probes:

| Case | v5 | v7.2 k2 | v7.3 hybrid | v7.3 `_hwtrain` |
|---|---:|---:|---:|---:|
| `512x512 / 6k / B=4` uniform F+B | 23.135 ms | 48.242 ms | 34.381 ms | 48.579 ms |
| `512x512 / 6k / B=4` clustered F+B | 141.302 ms | 95.728 ms | 91.297 ms | 95.394 ms |
| `4096x4096 / 64k / B=1` uniform F+B | 73.681 ms | 438.782 ms | 76.464 ms | 436.552 ms |

4K training result: v7.3 default is 82.6% faster than v7.2 and within 3.8% of
direct v5.

New forward-only probes:

| Case | v5 | v7.2 k2 | v7.3 hardware eval |
|---|---:|---:|---:|
| `512x512 / 6k / B=1` uniform forward | 7.615 ms | 11.430 ms | 7.861 ms |
| `4096x4096 / 64k / B=1` uniform forward | 11.781 ms | 183.679 ms | 133.129 ms |

Forward-only result: skipping capture helps v7.3 hardware eval, but 4K forward
is still much slower than v5 because the hardware path still reads the Metal
texture back before returning a torch MPS tensor.
