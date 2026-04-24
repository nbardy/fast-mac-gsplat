# Full Rasterizer Benchmark

Generated: 2026-04-23 10:42:56

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `240.0`
- accuracy checks: `False`
- accuracy max work items: `16000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v72_tiled_k2 | 10.232 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k2 | 27.995 |
| 512x512 | 1 | 6000 | layered_depth | forward | v72_tiled_k2 | 8.888 |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v72_tiled_k2 | 29.897 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v72_tiled_k2 | 10.050 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v72_tiled_k4 | 23.074 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v72_tiled_k8 | 31.141 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v72_tiled_k4 | 570.893 |
| 512x512 | 1 | 6000 | sparse_screen | forward | v72_tiled_k4 | 9.645 |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v72_tiled_k8 | 22.533 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v72_tiled_k2 | 25.525 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k2 | 92.275 |
| 512x512 | 4 | 6000 | layered_depth | forward | v72_tiled_k2 | 22.024 |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v72_tiled_k2 | 97.617 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v72_tiled_k2 | 22.543 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v72_tiled_k2 | 48.531 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v72_tiled_k2 | 103.255 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v72_tiled_k4 | 2300.913 |
| 512x512 | 4 | 6000 | sparse_screen | forward | v72_tiled_k2 | 20.495 |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v72_tiled_k2 | 37.702 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v72_tiled_k2 | ok | 10.050 | 10.050 | 10.050 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v72_tiled_k4 | ok | 10.562 | 10.562 | 10.562 | 0.000 |  |  |  | +5.1% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v72_tiled_k8 | ok | 12.980 | 12.980 | 12.980 | 0.000 |  |  |  | +29.2% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v72_tiled_k2 | ok | 24.211 | 24.211 | 9.538 | 14.673 |  |  |  | +4.9% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v72_tiled_k4 | ok | 23.074 | 23.074 | 8.001 | 15.073 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v72_tiled_k8 | ok | 25.310 | 25.310 | 9.238 | 16.072 |  |  |  | +9.7% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v72_tiled_k2 | ok | 9.870 | 9.870 | 9.869 | 0.000 |  |  |  | +2.3% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v72_tiled_k4 | ok | 9.645 | 9.645 | 9.645 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v72_tiled_k8 | ok | 14.026 | 14.026 | 14.026 | 0.000 |  |  |  | +45.4% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v72_tiled_k2 | ok | 25.171 | 25.171 | 9.592 | 15.580 |  |  |  | +11.7% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v72_tiled_k4 | ok | 26.900 | 26.900 | 9.852 | 17.048 |  |  |  | +19.4% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v72_tiled_k8 | ok | 22.533 | 22.533 | 9.893 | 12.640 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v72_tiled_k2 | ok | 10.232 | 10.232 | 10.232 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v72_tiled_k4 | ok | 12.504 | 12.504 | 12.504 | 0.000 |  |  |  | +22.2% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v72_tiled_k8 | ok | 18.948 | 18.948 | 18.947 | 0.000 |  |  |  | +85.2% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k2 | ok | 27.995 | 27.995 | 8.188 | 19.807 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k4 | ok | 38.897 | 38.897 | 10.199 | 28.697 |  |  |  | +38.9% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k8 | ok | 38.805 | 38.805 | 10.577 | 28.229 |  |  |  | +38.6% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v72_tiled_k2 | ok | 8.888 | 8.888 | 8.888 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v72_tiled_k4 | ok | 11.694 | 11.694 | 11.694 | 0.000 |  |  |  | +31.6% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v72_tiled_k8 | ok | 12.882 | 12.882 | 12.882 | 0.000 |  |  |  | +44.9% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v72_tiled_k2 | ok | 29.897 | 29.897 | 8.422 | 21.474 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v72_tiled_k4 | ok | 30.791 | 30.791 | 7.299 | 23.492 |  |  |  | +3.0% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v72_tiled_k8 | ok | 42.108 | 42.108 | 9.173 | 32.935 |  |  |  | +40.8% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v72_tiled_k2 | ok | 31.848 | 31.848 | 31.848 | 0.000 |  |  |  | +2.3% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v72_tiled_k4 | ok | 41.415 | 41.415 | 41.415 | 0.000 |  |  |  | +33.0% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v72_tiled_k8 | ok | 31.141 | 31.141 | 31.141 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v72_tiled_k2 | ok | 586.949 | 586.949 | 43.289 | 543.659 |  |  |  | +2.8% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v72_tiled_k4 | ok | 570.893 | 570.893 | 32.991 | 537.902 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v72_tiled_k8 | ok | 572.373 | 572.373 | 31.650 | 540.723 |  |  |  | +0.3% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v72_tiled_k2 | ok | 22.543 | 22.543 | 22.543 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v72_tiled_k4 | ok | 25.151 | 25.151 | 25.151 | 0.000 |  |  |  | +11.6% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v72_tiled_k8 | ok | 30.464 | 30.464 | 30.463 | 0.000 |  |  |  | +35.1% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v72_tiled_k2 | ok | 48.531 | 48.531 | 13.713 | 34.819 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v72_tiled_k4 | ok | 52.041 | 52.041 | 16.341 | 35.700 |  |  |  | +7.2% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v72_tiled_k8 | ok | 56.946 | 56.946 | 19.330 | 37.616 |  |  |  | +17.3% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v72_tiled_k2 | ok | 20.495 | 20.495 | 20.495 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v72_tiled_k4 | ok | 22.781 | 22.781 | 22.781 | 0.000 |  |  |  | +11.2% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v72_tiled_k8 | ok | 28.194 | 28.194 | 28.194 | 0.000 |  |  |  | +37.6% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v72_tiled_k2 | ok | 37.702 | 37.702 | 12.114 | 25.589 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v72_tiled_k4 | ok | 51.454 | 51.454 | 14.700 | 36.754 |  |  |  | +36.5% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v72_tiled_k8 | ok | 46.295 | 46.295 | 17.523 | 28.772 |  |  |  | +22.8% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v72_tiled_k2 | ok | 25.525 | 25.525 | 25.525 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v72_tiled_k4 | ok | 28.747 | 28.747 | 28.747 | 0.000 |  |  |  | +12.6% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v72_tiled_k8 | ok | 32.318 | 32.318 | 32.318 | 0.000 |  |  |  | +26.6% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k2 | ok | 92.275 | 92.275 | 19.999 | 72.275 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k4 | ok | 95.265 | 95.265 | 22.354 | 72.911 |  |  |  | +3.2% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k8 | ok | 99.960 | 99.960 | 23.933 | 76.028 |  |  |  | +8.3% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v72_tiled_k2 | ok | 22.024 | 22.024 | 22.023 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v72_tiled_k4 | ok | 35.394 | 35.394 | 35.394 | 0.000 |  |  |  | +60.7% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v72_tiled_k8 | ok | 37.940 | 37.940 | 37.940 | 0.000 |  |  |  | +72.3% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v72_tiled_k2 | ok | 97.617 | 97.617 | 16.063 | 81.554 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v72_tiled_k4 | ok | 99.798 | 99.798 | 18.621 | 81.177 |  |  |  | +2.2% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v72_tiled_k8 | ok | 105.445 | 105.445 | 19.927 | 85.518 |  |  |  | +8.0% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v72_tiled_k2 | ok | 103.255 | 103.255 | 103.255 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v72_tiled_k4 | ok | 105.757 | 105.757 | 105.757 | 0.000 |  |  |  | +2.4% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v72_tiled_k8 | ok | 109.741 | 109.741 | 109.741 | 0.000 |  |  |  | +6.3% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v72_tiled_k2 | ok | 2304.202 | 2304.202 | 131.880 | 2172.323 |  |  |  | +0.1% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v72_tiled_k4 | ok | 2300.913 | 2300.913 | 131.925 | 2168.988 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v72_tiled_k8 | ok | 2310.560 | 2310.560 | 135.859 | 2174.701 |  |  |  | +0.4% |  |  |  |  |
