# Full Rasterizer Benchmark

Generated: 2026-04-24 15:12:02

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `120.0`
- accuracy checks: `True`
- accuracy max work items: `20000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_eval_fallback | 2.671 |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_direct | 3.907 |
| 16x16 | 1 | 4 | microbench_uniform_random | forward | v8_direct | 2.978 |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | 3.535 |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_train_fallback | 2.558 |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | 3.712 |
| 16x16 | 1 | 32 | microbench_uniform_random | forward | v8_hw_eval_fallback | 2.317 |
| 16x16 | 1 | 32 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | 4.168 |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_train_fallback | 2.700 |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | 4.093 |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | v8_hw_eval_fallback | 2.792 |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | 3.751 |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward | v8_direct | 3.745 |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_direct | 4.531 |
| 32x32 | 1 | 32 | microbench_uniform_random | forward | v8_hw_train_fallback | 2.527 |
| 32x32 | 1 | 32 | microbench_uniform_random | forward_backward | v8_direct | 4.211 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 16x16 | 1 | 4 | microbench_uniform_random | forward | v8_direct | ok | 2.978 | 2.978 | 2.977 | 0.000 | ok | 5.960e-08 |  | +0.0% |  |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward | v8_hw_eval_fallback | ok | 3.629 | 3.629 | 3.629 | 0.000 | ok | 5.960e-08 |  | +21.9% |  |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward | v8_hw_train_fallback | ok | 3.847 | 3.847 | 3.847 | 0.000 | ok | 5.960e-08 |  | +29.2% |  |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | ok | 3.535 | 3.535 | 2.618 | 0.918 | ok | 5.960e-08 | 2.980e-08 | +0.0% |  |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v8_hw_eval_fallback | ok | 4.033 | 4.033 | 3.273 | 0.759 | ok | 5.960e-08 | 2.980e-08 | +14.1% |  |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | ok | 4.132 | 4.132 | 3.077 | 1.055 | ok | 5.960e-08 | 2.980e-08 | +16.9% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v8_direct | ok | 3.860 | 3.860 | 3.860 | 0.000 | ok | 5.960e-08 |  | +44.5% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_eval_fallback | ok | 2.671 | 2.671 | 2.670 | 0.000 | ok | 5.960e-08 |  | +0.0% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_train_fallback | ok | 4.078 | 4.078 | 4.078 | 0.000 | ok | 5.960e-08 |  | +52.7% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_direct | ok | 3.907 | 3.907 | 2.798 | 1.110 | ok | 5.960e-08 | 5.960e-08 | +0.0% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | ok | 4.284 | 4.284 | 3.310 | 0.974 | ok | 5.960e-08 | 5.960e-08 | +9.6% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | ok | 4.243 | 4.243 | 3.328 | 0.915 | ok | 5.960e-08 | 5.960e-08 | +8.6% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward | v8_direct | ok | 3.080 | 3.080 | 3.080 | 0.000 | ok | 1.192e-07 |  | +32.9% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward | v8_hw_eval_fallback | ok | 2.317 | 2.317 | 2.317 | 0.000 | ok | 1.192e-07 |  | +0.0% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward | v8_hw_train_fallback | ok | 3.531 | 3.531 | 3.531 | 0.000 | ok | 1.192e-07 |  | +52.4% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward_backward | v8_direct | ok | 5.363 | 5.363 | 4.140 | 1.223 | ok | 1.192e-07 | 5.960e-08 | +28.7% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward_backward | v8_hw_eval_fallback | ok | 6.827 | 6.827 | 5.054 | 1.774 | ok | 1.192e-07 | 5.960e-08 | +63.8% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | ok | 4.168 | 4.168 | 3.150 | 1.018 | ok | 1.192e-07 | 5.960e-08 | +0.0% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward | v8_direct | ok | 3.019 | 3.019 | 3.018 | 0.000 | ok | 6.402e-05 |  | +18.0% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_eval_fallback | ok | 3.839 | 3.839 | 3.839 | 0.000 | ok | 6.402e-05 |  | +50.1% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_train_fallback | ok | 2.558 | 2.558 | 2.558 | 0.000 | ok | 6.402e-05 |  | +0.0% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_direct | ok | 5.739 | 5.739 | 3.470 | 2.269 | ok | 6.402e-05 | 2.307e-04 | +54.6% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | ok | 3.712 | 3.712 | 2.808 | 0.904 | ok | 6.402e-05 | 2.307e-04 | +0.0% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | ok | 3.940 | 3.940 | 2.969 | 0.970 | ok | 6.402e-05 | 2.307e-04 | +6.1% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | v8_direct | ok | 3.217 | 3.217 | 3.217 | 0.000 | ok | 4.470e-08 |  | +15.2% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | v8_hw_eval_fallback | ok | 2.792 | 2.792 | 2.792 | 0.000 | ok | 4.470e-08 |  | +0.0% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | v8_hw_train_fallback | ok | 3.567 | 3.567 | 3.567 | 0.000 | ok | 4.470e-08 |  | +27.7% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | ok | 3.751 | 3.751 | 2.895 | 0.856 | ok | 4.470e-08 | 3.725e-09 | +0.0% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v8_hw_eval_fallback | ok | 3.960 | 3.960 | 2.999 | 0.961 | ok | 4.470e-08 | 3.725e-09 | +5.6% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | ok | 3.852 | 3.852 | 2.914 | 0.938 | ok | 4.470e-08 | 3.725e-09 | +2.7% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | v8_direct | ok | 3.625 | 3.625 | 3.620 | 0.005 | ok | 5.960e-08 |  | +34.3% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_eval_fallback | ok | 3.534 | 3.534 | 3.534 | 0.000 | ok | 5.960e-08 |  | +30.9% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_train_fallback | ok | 2.700 | 2.700 | 2.700 | 0.000 | ok | 5.960e-08 |  | +0.0% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_direct | ok | 4.877 | 4.877 | 3.710 | 1.167 | ok | 5.960e-08 | 5.960e-08 | +19.1% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | ok | 6.068 | 6.068 | 4.900 | 1.168 | ok | 5.960e-08 | 5.960e-08 | +48.2% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | ok | 4.093 | 4.093 | 3.268 | 0.826 | ok | 5.960e-08 | 5.960e-08 | +0.0% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward | v8_direct | ok | 4.659 | 4.659 | 4.658 | 0.000 | ok | 1.192e-07 |  | +84.4% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward | v8_hw_eval_fallback | ok | 4.771 | 4.771 | 4.771 | 0.000 | ok | 1.192e-07 |  | +88.8% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward | v8_hw_train_fallback | ok | 2.527 | 2.527 | 2.527 | 0.000 | ok | 1.192e-07 |  | +0.0% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward_backward | v8_direct | ok | 4.211 | 4.211 | 3.127 | 1.084 | ok | 1.192e-07 | 1.490e-08 | +0.0% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward_backward | v8_hw_eval_fallback | ok | 4.383 | 4.383 | 3.386 | 0.997 | ok | 1.192e-07 | 1.490e-08 | +4.1% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | ok | 5.046 | 5.046 | 3.997 | 1.048 | ok | 1.192e-07 | 1.490e-08 | +19.8% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward | v8_direct | ok | 3.745 | 3.745 | 3.745 | 0.000 | ok | 6.127e-05 |  | +0.0% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_eval_fallback | ok | 3.909 | 3.909 | 3.909 | 0.000 | ok | 6.127e-05 |  | +4.4% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_train_fallback | ok | 3.963 | 3.963 | 3.963 | 0.001 | ok | 6.127e-05 |  | +5.8% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_direct | ok | 4.531 | 4.531 | 3.297 | 1.234 | ok | 6.127e-05 | 5.615e-05 | +0.0% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | ok | 4.825 | 4.825 | 3.745 | 1.080 | ok | 6.127e-05 | 5.621e-05 | +6.5% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | ok | 4.904 | 4.904 | 3.794 | 1.110 | ok | 6.127e-05 | 5.621e-05 | +8.2% |  |  |  |  |
