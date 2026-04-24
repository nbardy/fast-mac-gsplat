# Full Rasterizer Benchmark

Generated: 2026-04-24 15:41:42

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
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_eval_fallback | 2.720 |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | 3.900 |
| 16x16 | 1 | 4 | microbench_uniform_random | forward | v8_direct | 2.861 |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | 3.936 |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_train_fallback | 2.608 |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_direct | 4.010 |
| 16x16 | 1 | 32 | microbench_uniform_random | forward | v8_hw_eval_fallback | 3.121 |
| 16x16 | 1 | 32 | microbench_uniform_random | forward_backward | v8_direct | 3.656 |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_train_fallback | 2.909 |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | 3.547 |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | v8_hw_train_fallback | 3.467 |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | 4.167 |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_eval_fallback | 2.576 |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | 3.710 |
| 32x32 | 1 | 32 | microbench_uniform_random | forward | v8_hw_eval_fallback | 2.254 |
| 32x32 | 1 | 32 | microbench_uniform_random | forward_backward | v8_direct | 3.447 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 16x16 | 1 | 4 | microbench_uniform_random | forward | v8_direct | ok | 2.861 | 2.861 | 2.861 | 0.000 | ok | 5.960e-08 |  | +0.0% |  |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward | v8_hw_eval_fallback | ok | 4.097 | 4.097 | 4.097 | 0.000 | ok | 5.960e-08 |  | +43.2% |  |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward | v8_hw_train_fallback | ok | 3.235 | 3.235 | 3.234 | 0.001 | ok | 5.960e-08 |  | +13.0% |  |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | ok | 4.036 | 4.036 | 3.127 | 0.909 | ok | 5.960e-08 | 2.980e-08 | +2.5% |  |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v8_hw_eval_fallback | ok | 5.710 | 5.710 | 3.725 | 1.986 | ok | 5.960e-08 | 2.980e-08 | +45.1% |  |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | ok | 3.936 | 3.936 | 3.052 | 0.884 | ok | 5.960e-08 | 2.980e-08 | +0.0% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v8_direct | ok | 3.331 | 3.331 | 3.330 | 0.000 | ok | 5.960e-08 |  | +22.5% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_eval_fallback | ok | 2.720 | 2.720 | 2.720 | 0.000 | ok | 5.960e-08 |  | +0.0% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_train_fallback | ok | 2.916 | 2.916 | 2.916 | 0.000 | ok | 5.960e-08 |  | +7.2% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_direct | ok | 4.340 | 4.340 | 3.424 | 0.916 | ok | 5.960e-08 | 5.960e-08 | +11.3% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | ok | 3.900 | 3.900 | 3.134 | 0.766 | ok | 5.960e-08 | 5.960e-08 | +0.0% |  |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | ok | 4.807 | 4.807 | 3.854 | 0.953 | ok | 5.960e-08 | 5.960e-08 | +23.2% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward | v8_direct | ok | 6.181 | 6.181 | 6.181 | 0.000 | ok | 1.192e-07 |  | +98.1% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward | v8_hw_eval_fallback | ok | 3.121 | 3.121 | 3.120 | 0.000 | ok | 1.192e-07 |  | +0.0% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward | v8_hw_train_fallback | ok | 3.422 | 3.422 | 3.422 | 0.001 | ok | 1.192e-07 |  | +9.7% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward_backward | v8_direct | ok | 3.656 | 3.656 | 2.569 | 1.087 | ok | 1.192e-07 | 5.960e-08 | +0.0% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward_backward | v8_hw_eval_fallback | ok | 4.880 | 4.880 | 3.759 | 1.121 | ok | 1.192e-07 | 5.960e-08 | +33.5% |  |  |  |  |
| 16x16 | 1 | 32 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | ok | 3.970 | 3.970 | 2.935 | 1.035 | ok | 1.192e-07 | 5.960e-08 | +8.6% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward | v8_direct | ok | 3.540 | 3.540 | 3.540 | 0.000 | ok | 6.402e-05 |  | +35.7% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_eval_fallback | ok | 2.888 | 2.888 | 2.888 | 0.000 | ok | 6.402e-05 |  | +10.7% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_train_fallback | ok | 2.608 | 2.608 | 2.608 | 0.000 | ok | 6.402e-05 |  | +0.0% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_direct | ok | 4.010 | 4.010 | 2.905 | 1.105 | ok | 6.402e-05 | 2.307e-04 | +0.0% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | ok | 6.104 | 6.104 | 5.052 | 1.052 | ok | 6.402e-05 | 2.307e-04 | +52.2% |  |  |  |  |
| 16x16 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | ok | 4.264 | 4.264 | 3.188 | 1.076 | ok | 6.402e-05 | 2.307e-04 | +6.3% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | v8_direct | ok | 4.196 | 4.196 | 4.196 | 0.000 | ok | 4.470e-08 |  | +21.0% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | v8_hw_eval_fallback | ok | 3.727 | 3.727 | 3.727 | 0.000 | ok | 4.470e-08 |  | +7.5% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | v8_hw_train_fallback | ok | 3.467 | 3.467 | 3.467 | 0.000 | ok | 4.470e-08 |  | +0.0% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | ok | 5.449 | 5.449 | 4.525 | 0.924 | ok | 4.470e-08 | 3.725e-09 | +30.8% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v8_hw_eval_fallback | ok | 6.040 | 6.040 | 3.184 | 2.856 | ok | 4.470e-08 | 3.725e-09 | +45.0% |  |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | ok | 4.167 | 4.167 | 3.186 | 0.981 | ok | 4.470e-08 | 3.725e-09 | +0.0% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | v8_direct | ok | 4.242 | 4.242 | 4.241 | 0.001 | ok | 5.960e-08 |  | +45.8% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_eval_fallback | ok | 3.137 | 3.137 | 3.137 | 0.000 | ok | 5.960e-08 |  | +7.8% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | v8_hw_train_fallback | ok | 2.909 | 2.909 | 2.909 | 0.000 | ok | 5.960e-08 |  | +0.0% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_direct | ok | 4.632 | 4.632 | 3.589 | 1.042 | ok | 5.960e-08 | 5.960e-08 | +30.6% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | ok | 5.323 | 5.323 | 4.332 | 0.990 | ok | 5.960e-08 | 5.960e-08 | +50.1% |  |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | ok | 3.547 | 3.547 | 2.752 | 0.795 | ok | 5.960e-08 | 5.960e-08 | +0.0% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward | v8_direct | ok | 4.127 | 4.127 | 4.127 | 0.000 | ok | 1.192e-07 |  | +83.1% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward | v8_hw_eval_fallback | ok | 2.254 | 2.254 | 2.254 | 0.000 | ok | 1.192e-07 |  | +0.0% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward | v8_hw_train_fallback | ok | 2.734 | 2.734 | 2.734 | 0.000 | ok | 1.192e-07 |  | +21.3% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward_backward | v8_direct | ok | 3.447 | 3.447 | 2.472 | 0.975 | ok | 1.192e-07 | 1.490e-08 | +0.0% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward_backward | v8_hw_eval_fallback | ok | 5.459 | 5.459 | 3.874 | 1.585 | ok | 1.192e-07 | 1.490e-08 | +58.4% |  |  |  |  |
| 32x32 | 1 | 32 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | ok | 4.157 | 4.157 | 3.161 | 0.996 | ok | 1.192e-07 | 1.490e-08 | +20.6% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward | v8_direct | ok | 4.628 | 4.628 | 4.628 | 0.000 | ok | 6.127e-05 |  | +79.7% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_eval_fallback | ok | 2.576 | 2.576 | 2.576 | 0.000 | ok | 6.127e-05 |  | +0.0% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward | v8_hw_train_fallback | ok | 3.842 | 3.842 | 3.842 | 0.000 | ok | 6.127e-05 |  | +49.1% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_direct | ok | 4.390 | 4.390 | 2.869 | 1.521 | ok | 6.127e-05 | 5.621e-05 | +18.3% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | ok | 4.124 | 4.124 | 3.401 | 0.723 | ok | 6.127e-05 | 5.615e-05 | +11.1% |  |  |  |  |
| 32x32 | 1 | 32 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | ok | 3.710 | 3.710 | 2.612 | 1.099 | ok | 6.127e-05 | 5.615e-05 | +0.0% |  |  |  |  |
