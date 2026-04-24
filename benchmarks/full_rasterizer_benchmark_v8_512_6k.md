# Full Rasterizer Benchmark

Generated: 2026-04-24 14:28:14

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `2`
- iters: `5`
- seed: `0`
- timeout seconds per cell: `180.0`
- accuracy checks: `False`
- accuracy max work items: `16000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_direct | 18.577 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v8_direct | 23.782 |
| 512x512 | 1 | 6000 | layered_depth | forward | v8_direct | 4.095 |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v8_direct | 6.110 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v8_direct | 3.320 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v8_direct | 7.178 |
| 512x512 | 1 | 6000 | sparse_screen | forward | v8_direct | 4.538 |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v8_direct | 5.604 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v8_direct | 101.382 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v8_direct | 86.209 |
| 512x512 | 4 | 6000 | layered_depth | forward | v8_direct | 12.007 |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v6_direct | 22.305 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v8_direct | 3.939 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v8_direct | 18.832 |
| 512x512 | 4 | 6000 | sparse_screen | forward | v6_direct | 7.261 |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v8_direct | 13.910 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_direct | ok | 5.544 | 5.781 | 5.544 | 0.000 |  |  |  | +67.0% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v8_direct | ok | 3.320 | 3.082 | 3.320 | 0.000 |  |  |  | +0.0% |  | -40.1% |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_direct | ok | 9.663 | 8.600 | 5.635 | 4.028 |  |  |  | +34.6% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v8_direct | ok | 7.178 | 6.072 | 3.660 | 3.518 |  |  |  | +0.0% |  | -25.7% |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v6_direct | ok | 5.591 | 5.042 | 5.591 | 0.000 |  |  |  | +23.2% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v8_direct | ok | 4.538 | 4.485 | 4.538 | 0.000 |  |  |  | +0.0% |  | -18.8% |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v6_direct | ok | 8.520 | 8.676 | 5.580 | 2.940 |  |  |  | +52.0% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v8_direct | ok | 5.604 | 5.589 | 2.976 | 2.627 |  |  |  | +0.0% |  | -34.2% |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_direct | ok | 18.577 | 18.304 | 18.577 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v8_direct | ok | 19.183 | 18.433 | 19.183 | 0.000 |  |  |  | +3.3% |  | +3.3% |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_direct | ok | 51.289 | 57.085 | 28.128 | 23.161 |  |  |  | +115.7% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v8_direct | ok | 23.782 | 23.313 | 16.102 | 7.680 |  |  |  | +0.0% |  | -53.6% |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v6_direct | ok | 6.997 | 7.972 | 6.996 | 0.000 |  |  |  | +70.9% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v8_direct | ok | 4.095 | 5.021 | 4.094 | 0.000 |  |  |  | +0.0% |  | -41.5% |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v6_direct | ok | 9.586 | 9.881 | 5.993 | 3.593 |  |  |  | +56.9% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v8_direct | ok | 6.110 | 6.391 | 3.331 | 2.778 |  |  |  | +0.0% |  | -36.3% |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_direct | ok | 8.263 | 7.069 | 8.263 | 0.000 |  |  |  | +109.8% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v8_direct | ok | 3.939 | 4.133 | 3.939 | 0.000 |  |  |  | +0.0% |  | -52.3% |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_direct | ok | 21.706 | 21.749 | 7.357 | 14.349 |  |  |  | +15.3% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v8_direct | ok | 18.832 | 18.813 | 5.437 | 13.396 |  |  |  | +0.0% |  | -13.2% |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v6_direct | ok | 7.261 | 5.211 | 7.261 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v8_direct | ok | 8.023 | 7.842 | 8.023 | 0.000 |  |  |  | +10.5% |  | +10.5% |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v6_direct | ok | 27.593 | 30.859 | 9.737 | 17.856 |  |  |  | +98.4% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v8_direct | ok | 13.910 | 13.897 | 4.090 | 9.820 |  |  |  | +0.0% |  | -49.6% |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_direct | ok | 108.252 | 114.192 | 108.252 | 0.001 |  |  |  | +6.8% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v8_direct | ok | 101.382 | 100.041 | 101.381 | 0.000 |  |  |  | +0.0% |  | -6.3% |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_direct | ok | 109.206 | 109.562 | 77.696 | 31.509 |  |  |  | +26.7% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v8_direct | ok | 86.209 | 88.113 | 60.215 | 25.994 |  |  |  | +0.0% |  | -21.1% |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v6_direct | ok | 13.510 | 13.387 | 13.510 | 0.000 |  |  |  | +12.5% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v8_direct | ok | 12.007 | 11.995 | 12.006 | 0.000 |  |  |  | +0.0% |  | -11.1% |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v6_direct | ok | 22.305 | 22.280 | 7.869 | 14.436 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v8_direct | ok | 24.259 | 22.903 | 6.632 | 17.627 |  |  |  | +8.8% |  | +8.8% |  |  |
