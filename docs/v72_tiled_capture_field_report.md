# v7.2 Tiled Capture Field Report

Date: 2026-04-23

## Summary

The `torch_metal_gsplat_v72_tiled_capture.tar.gz` handoff is preserved as:

```text
source_artifacts/torch_metal_gsplat_v72_tiled_capture.tar.gz
variants/v7_tiled_capture/
```

Archive SHA-256:

```text
59e4bd0b258e9cf3d9df911664cdbc79576e0e873fb1fae7e1061ce394a7e78e
```

The root benchmark runner now supports:

```text
v72_tiled_k2
v72_tiled_k4
v72_tiled_k8
v72_tiled_k2_t32  # optional tile-size suffix form
```

## Local Fixes Applied

```text
Metal compile: replaced program-scope constexpr with a macro.
Fragment shader: removed duplicate [[position]] argument.
Draw order: restored far-to-near draw order for source-over blending parity.
Ellipse bounds: restored full pixel-space support bounds.
CPU/Python tile bins: matched support bounds so tiled capture stays conservative.
```

## Correctness

Bundled tiled-reference test:

```bash
cd variants/v7_tiled_capture
python3 tests/reference_tiled_exactness.py
```

Result: all tiled reference cases passed. Max reported gradient error was below
`2e-6`.

Small MPS smoke against dense Torch reference:

```text
image max error:      8.940696716308594e-08
means grad max error: 1.7462298274040222e-10
conics grad max error: 1.1641532182693481e-09
colors grad max error: 9.313225746154785e-10
opacity grad max error: 1.862645149230957e-09
```

Conclusion: v7.2 passes the same small hardware correctness bar as v7.1 after
local fixes.

## 512x512 / 6k

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 512x512 \
  --splats 6000 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers v72_tiled_k2,v72_tiled_k4,v72_tiled_k8 \
  --modes forward,forward_backward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 240 \
  --output-md benchmarks/full_rasterizer_benchmark_v72_tiled_512_6k.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v72_tiled_512_6k.jsonl
```

B=1 forward+backward:

| Distribution | Previous Exact Best | Best ms | v7.1 Best ms | v7.2 Best | v7.2 ms | Delta vs Exact Best | Delta vs v7.1 |
|---|---|---:|---:|---|---:|---:|---:|
| microbench_uniform_random | `v6_upgrade_direct` | 38.898 | 55.680 | `v72_tiled_k4` | 23.074 | -40.7% | -58.6% |
| sparse_screen | `v6_direct` | 27.760 | 43.600 | `v72_tiled_k8` | 22.533 | -18.8% | -48.3% |
| clustered_hot_tiles | `v6_refined_auto` | 95.208 | 54.997 | `v72_tiled_k2` | 27.995 | -70.6% | -49.1% |
| layered_depth | `v5_batched` | 28.228 | 56.749 | `v72_tiled_k2` | 29.897 | +5.9% | -47.3% |
| overflow_adversarial | `v6_upgrade_auto` | 634.802 | 587.924 | `v72_tiled_k4` | 570.893 | -10.1% | -2.9% |

B=4 forward+backward:

| Distribution | Previous Exact Best | Best ms | v7.1 Best ms | v7.2 Best | v7.2 ms | Delta vs Exact Best | Delta vs v7.1 |
|---|---|---:|---:|---|---:|---:|---:|
| microbench_uniform_random | `v6_refined_auto` | 30.562 | 162.835 | `v72_tiled_k2` | 48.531 | +58.8% | -70.2% |
| sparse_screen | `v6_refined_direct` | 38.187 | 149.303 | `v72_tiled_k2` | 37.702 | -1.3% | -74.7% |
| clustered_hot_tiles | `v6_refined_auto` | 611.650 | 202.522 | `v72_tiled_k2` | 92.275 | -84.9% | -54.4% |
| layered_depth | `v5_batched` | 37.673 | 203.862 | `v72_tiled_k2` | 97.617 | +159.1% | -52.1% |
| overflow_adversarial | `v6_refined_auto` | 1503.726 | 2316.551 | `v72_tiled_k4` | 2300.913 | +53.0% | -0.7% |

## 4096x4096 / 64k

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 4096x4096 \
  --splats 65536 \
  --batch-sizes 1 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers v5_batched,v72_tiled_k2,v72_tiled_k4,v72_tiled_k8 \
  --modes forward_backward \
  --warmup 1 \
  --iters 1 \
  --timeout-sec 360 \
  --output-md benchmarks/full_rasterizer_benchmark_v72_tiled_4k64k.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v72_tiled_4k64k.jsonl
```

| Distribution | Previous Exact Best ms | v5 ms | v7.1 k2 ms | v7.2 Best | v7.2 ms | Delta vs Previous Exact Best | Delta vs v7.1 |
|---|---:|---:|---:|---|---:|---:|---:|
| microbench_uniform_random | 62.657 | 78.218 | 35852.168 | `v72_tiled_k2` | 382.507 | +510.5% | -98.9% |
| sparse_screen | 52.268 | 69.576 | 21808.049 | `v72_tiled_k2` | 314.068 | +500.9% | -98.6% |
| clustered_hot_tiles | 176.944 | 208.881 | 23069.772 | `v72_tiled_k2` | 459.409 | +159.6% | -98.0% |
| layered_depth | 74.644 | 89.923 | 24497.635 | `v72_tiled_k2` | 487.923 | +553.7% | -98.0% |
| overflow_adversarial | 1089.448 | 1207.543 | 27008.610 | `v72_tiled_k2` | 6525.487 | +499.0% | -75.8% |

## Interpretation

v7.2 fixes the v7.1 scaling bug. At 4K / 64k, normal cases dropped from
`21-36 s` to `0.31-0.49 s`, a roughly `98%` end-to-end reduction versus v7.1.

It is still not the fastest 4K training path. `v5_batched` and the v6-family
training paths remain faster because v7.2 still pays for:

- CPU packing and CPU tile-bin construction every forward pass
- CPU copies of the forward image and saved front state
- one full-pixel front-K pass and one full-pixel backward pass, even though each
  pixel now scans a local tile bin
- high overflow replay cost in adversarial scenes where many splats share the
  same central tiles

## Recommendation

Promote v7.2 over v7.1 for hardware-backward experiments. Do not promote it as
the default 4K / 64k training path yet.

The next useful ask is v7.3: keep v7.2's binned capture/replay and corrected
backward math, but move tile binning and saved-state production fully onto the
GPU and avoid CPU round trips for `out_cpu`, `front_ids`, `front_raw_alpha`, and
`front_meta`.
