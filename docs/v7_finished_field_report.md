# v7 Finished Hardware Field Report

Date: 2026-04-22

## Summary

The `torch_metal_gsplat_v7_hardware_finished.tar.gz` handoff is preserved as:

```text
source_artifacts/torch_metal_gsplat_v7_hardware_finished.tar.gz
variants/v7_finished/
```

The root benchmark runner now supports:

```text
v7_finished_hardware
```

This variant is kept beside `variants/v7`; it does not replace the current local
v7 hardware variant.

## Local Fixes Applied

The raw archive needed local fixes before it could be tested safely:

```text
fragment shader: removed duplicate [[position]] argument that prevented runtime Metal compilation
draw order: restored far-to-near draw order for source-over blending parity
ellipse bounds: restored current v7 bounds without the new half-pixel shift
B>1 uniform: restored per-batch u.gaussians after buffer offsetting
tests/benchmark: restored direct script entrypoint and JSON benchmark support
```

Without the draw-order and bounds fixes, the tiny reference check produced
roughly `9e-2` image max error. With those fixes, forward image parity matches
the current local v7 small reference.

## Correctness

Command:

```bash
cd variants/v7_finished
python3 setup.py build_ext --inplace
python3 tests/reference_check.py
```

Result after local forward fixes:

```text
image max error:      5.960464477539063e-08
means grad max error: 0.0004210318438708782
conics grad max error: 9.313225746154785e-10
colors grad max error: 0.034550316631793976
opacities grad error:  1.862645149230957e-09
```

Conclusion: forward image parity is restored, but the new backward formulas are
not correct. The current local `variants/v7` reference check has gradients near
`1e-9`, while `v7_finished` has materially wrong `means` and `colors` gradients.
Do not use `v7_finished_hardware` for training.

## 128x128 / 64-Splat Smoke

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 128x128 \
  --splats 64 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers v7_hardware,v7_finished_hardware \
  --modes forward,forward_backward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 120 \
  --accuracy \
  --accuracy-max-work-items 5000000 \
  --output-md benchmarks/full_rasterizer_benchmark_v7_finished_smoke.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v7_finished_smoke.jsonl
```

Result: all 40 cells completed with `status=ok`.

Accuracy:

```text
old v7 max image error:       0.0029408037662506104
old v7 max grad error:        9.5367431640625e-06
v7_finished max image error:  0.0029408037662506104
v7_finished max grad error:   0.29313182830810547
```

The new v7 won 10 of 20 workload groups in this tiny smoke, but the gradient
error makes forward+backward wins unusable for training.

## 512x512 / 6k Comparison

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 512x512 \
  --splats 6000 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers v7_hardware,v7_finished_hardware \
  --modes forward,forward_backward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 240 \
  --accuracy \
  --accuracy-max-work-items 7000000 \
  --output-md benchmarks/full_rasterizer_benchmark_v7_finished_512_6k.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v7_finished_512_6k.jsonl
```

The dense-reference accuracy columns are skipped here because the cells exceed
the accuracy work-item limit. Use the small smoke for correctness.

| B | Distribution | Mode | Old v7 ms | v7 Finished ms | Delta |
|---:|---|---|---:|---:|---:|
| 1 | clustered_hot_tiles | forward | 11.718 | 14.825 | +26.5% |
| 1 | clustered_hot_tiles | forward+backward | 40.668 | 51.370 | +26.3% |
| 1 | layered_depth | forward | 10.911 | 10.186 | -6.6% |
| 1 | layered_depth | forward+backward | 52.821 | 41.687 | -21.1% |
| 1 | microbench_uniform_random | forward | 9.257 | 9.858 | +6.5% |
| 1 | microbench_uniform_random | forward+backward | 39.857 | 40.364 | +1.3% |
| 1 | overflow_adversarial | forward | 33.110 | 23.841 | -28.0% |
| 1 | overflow_adversarial | forward+backward | 101.090 | 155.760 | +54.1% |
| 1 | sparse_screen | forward | 8.477 | 7.343 | -13.4% |
| 1 | sparse_screen | forward+backward | 37.217 | 37.432 | +0.6% |
| 4 | clustered_hot_tiles | forward | 20.277 | 21.024 | +3.7% |
| 4 | clustered_hot_tiles | forward+backward | 143.562 | 146.479 | +2.0% |
| 4 | layered_depth | forward | 15.468 | 16.672 | +7.8% |
| 4 | layered_depth | forward+backward | 135.338 | 134.874 | -0.3% |
| 4 | microbench_uniform_random | forward | 30.351 | 32.297 | +6.4% |
| 4 | microbench_uniform_random | forward+backward | 219.626 | 226.186 | +3.0% |
| 4 | overflow_adversarial | forward | 87.101 | 79.498 | -8.7% |
| 4 | overflow_adversarial | forward+backward | 378.549 | 369.843 | -2.3% |
| 4 | sparse_screen | forward | 22.357 | 11.117 | -50.3% |
| 4 | sparse_screen | forward+backward | 129.311 | 128.022 | -1.0% |

Negative delta means `v7_finished_hardware` is faster than old `v7_hardware`.
At 512x512 / 6k, finished v7 is mixed: it has some useful forward wins, but no
clear stable speedup and backward remains incorrect.

## 4096x4096 / 64k Forward

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 4096x4096 \
  --splats 65536 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers v7_hardware,v7_finished_hardware \
  --modes forward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 240 \
  --output-md benchmarks/full_rasterizer_benchmark_v7_finished_4k64k_forward.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v7_finished_4k64k_forward.jsonl
```

| B | Distribution | Old v7 ms | v7 Finished ms | Delta |
|---:|---|---:|---:|---:|
| 1 | clustered_hot_tiles | 117.265 | 117.198 | -0.1% |
| 1 | layered_depth | 133.826 | 117.702 | -12.0% |
| 1 | microbench_uniform_random | 210.819 | 143.662 | -31.9% |
| 1 | overflow_adversarial | 389.739 | 367.198 | -5.8% |
| 1 | sparse_screen | 111.345 | 114.497 | +2.8% |
| 4 | clustered_hot_tiles | 348.813 | 345.868 | -0.8% |
| 4 | layered_depth | 367.598 | 366.397 | -0.3% |
| 4 | microbench_uniform_random | 509.926 | 455.647 | -10.6% |
| 4 | overflow_adversarial | 1605.064 | 1643.673 | +2.4% |
| 4 | sparse_screen | 356.121 | 365.236 | +2.6% |

At 4K / 64k forward-only, finished v7 improves 7 of 10 rows, especially B=1
uniform random. It regresses sparse-screen and B=4 overflow slightly.

## 4096x4096 / 64k Backward Probe

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 4096x4096 \
  --splats 65536 \
  --batch-sizes 1 \
  --distributions microbench_uniform_random,clustered_hot_tiles \
  --renderers v7_finished_hardware \
  --modes forward_backward \
  --warmup 0 \
  --iters 1 \
  --timeout-sec 240 \
  --output-md benchmarks/full_rasterizer_benchmark_v7_finished_4k64k_backward_probe.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v7_finished_4k64k_backward_probe.jsonl
```

| Distribution | Total ms | Forward ms | Backward ms |
|---|---:|---:|---:|
| microbench_uniform_random | 21440.399 | 254.783 | 21185.617 |
| clustered_hot_tiles | 20330.785 | 212.092 | 20118.694 |

This is still around 20 seconds per backward pass at B=1 / 64k. It is the same
order of magnitude as old v7 and remains far slower than v6-family training
paths.

## Recommendation

Do not promote `v7_finished_hardware` to a training path.

Use it only as a forward/eval experiment:

- promising at 4K / 64k forward for B=1 uniform and layered cases
- mixed at 512x512 / 6k
- gradients are wrong
- large backward remains around 20 seconds per pass

The next chief-scientist ask should be a true v7 backward redesign, not a small
formula tweak. The current finished handoff does not solve the training blocker.
