# Full Rasterizer Benchmark

Generated: 2026-04-22 19:37:16

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
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v7_frontk_k2 | 31.270 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v7_frontk_k2 | 54.997 |
| 512x512 | 1 | 6000 | layered_depth | forward | v7_frontk_k2 | 29.002 |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v7_frontk_k2 | 56.749 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v7_frontk_k2 | 19.327 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v7_frontk_k2 | 55.680 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v7_frontk_k4 | 35.194 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v7_frontk_k2 | 587.924 |
| 512x512 | 1 | 6000 | sparse_screen | forward | v7_frontk_k2 | 30.907 |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v7_frontk_k2 | 43.600 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v7_frontk_k2 | 112.696 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v7_frontk_k2 | 202.522 |
| 512x512 | 4 | 6000 | layered_depth | forward | v7_frontk_k2 | 97.672 |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v7_frontk_k2 | 203.862 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v7_frontk_k2 | 55.647 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v7_frontk_k2 | 162.835 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v7_frontk_k2 | 124.142 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v7_frontk_k2 | 2316.551 |
| 512x512 | 4 | 6000 | sparse_screen | forward | v7_frontk_k2 | 108.280 |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v7_frontk_k2 | 149.303 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v7_frontk_k2 | ok | 19.327 | 19.327 | 19.326 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v7_frontk_k4 | ok | 27.051 | 27.051 | 27.051 | 0.000 |  |  |  | +40.0% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v7_frontk_k8 | ok | 36.563 | 36.563 | 36.562 | 0.000 |  |  |  | +89.2% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v7_frontk_k2 | ok | 55.680 | 55.680 | 15.620 | 40.060 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v7_frontk_k4 | ok | 70.215 | 70.215 | 22.939 | 47.276 |  |  |  | +26.1% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v7_frontk_k8 | ok | 62.279 | 62.279 | 25.359 | 36.920 |  |  |  | +11.9% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v7_frontk_k2 | ok | 30.907 | 30.907 | 30.906 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v7_frontk_k4 | ok | 30.941 | 30.941 | 30.941 | 0.000 |  |  |  | +0.1% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v7_frontk_k8 | ok | 34.148 | 34.148 | 34.148 | 0.000 |  |  |  | +10.5% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v7_frontk_k2 | ok | 43.600 | 43.600 | 30.487 | 13.114 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v7_frontk_k4 | ok | 47.797 | 47.797 | 33.413 | 14.384 |  |  |  | +9.6% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v7_frontk_k8 | ok | 60.310 | 60.310 | 41.711 | 18.599 |  |  |  | +38.3% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v7_frontk_k2 | ok | 31.270 | 31.270 | 31.270 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v7_frontk_k4 | ok | 33.518 | 33.518 | 33.517 | 0.000 |  |  |  | +7.2% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v7_frontk_k8 | ok | 35.091 | 35.091 | 35.091 | 0.000 |  |  |  | +12.2% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v7_frontk_k2 | ok | 54.997 | 54.997 | 30.869 | 24.128 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v7_frontk_k4 | ok | 57.580 | 57.580 | 32.938 | 24.642 |  |  |  | +4.7% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v7_frontk_k8 | ok | 59.783 | 59.783 | 32.852 | 26.931 |  |  |  | +8.7% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v7_frontk_k2 | ok | 29.002 | 29.002 | 29.002 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v7_frontk_k4 | ok | 29.181 | 29.181 | 29.181 | 0.000 |  |  |  | +0.6% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v7_frontk_k8 | ok | 31.829 | 31.829 | 31.829 | 0.000 |  |  |  | +9.7% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v7_frontk_k2 | ok | 56.749 | 56.749 | 27.373 | 29.376 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v7_frontk_k4 | ok | 57.602 | 57.602 | 28.143 | 29.459 |  |  |  | +1.5% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v7_frontk_k8 | ok | 76.546 | 76.546 | 39.914 | 36.632 |  |  |  | +34.9% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v7_frontk_k2 | ok | 35.439 | 35.439 | 35.439 | 0.000 |  |  |  | +0.7% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v7_frontk_k4 | ok | 35.194 | 35.194 | 35.194 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v7_frontk_k8 | ok | 36.346 | 36.346 | 36.345 | 0.000 |  |  |  | +3.3% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v7_frontk_k2 | ok | 587.924 | 587.924 | 37.564 | 550.360 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v7_frontk_k4 | ok | 591.175 | 591.175 | 39.642 | 551.534 |  |  |  | +0.6% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v7_frontk_k8 | ok | 593.230 | 593.230 | 40.344 | 552.886 |  |  |  | +0.9% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v7_frontk_k2 | ok | 55.647 | 55.647 | 55.647 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v7_frontk_k4 | ok | 60.587 | 60.587 | 60.587 | 0.000 |  |  |  | +8.9% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v7_frontk_k8 | ok | 92.895 | 92.895 | 92.895 | 0.000 |  |  |  | +66.9% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v7_frontk_k2 | ok | 162.835 | 162.835 | 42.770 | 120.065 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v7_frontk_k4 | ok | 189.053 | 189.053 | 59.766 | 129.287 |  |  |  | +16.1% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v7_frontk_k8 | ok | 224.842 | 224.842 | 89.094 | 135.747 |  |  |  | +38.1% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v7_frontk_k2 | ok | 108.280 | 108.280 | 108.279 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v7_frontk_k4 | ok | 112.217 | 112.217 | 112.217 | 0.000 |  |  |  | +3.6% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v7_frontk_k8 | ok | 119.503 | 119.503 | 119.503 | 0.000 |  |  |  | +10.4% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v7_frontk_k2 | ok | 149.303 | 149.303 | 108.144 | 41.159 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v7_frontk_k4 | ok | 155.311 | 155.311 | 112.123 | 43.189 |  |  |  | +4.0% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v7_frontk_k8 | ok | 176.955 | 176.955 | 119.971 | 56.985 |  |  |  | +18.5% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v7_frontk_k2 | ok | 112.696 | 112.696 | 112.695 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v7_frontk_k4 | ok | 119.172 | 119.172 | 119.171 | 0.000 |  |  |  | +5.7% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v7_frontk_k8 | ok | 121.593 | 121.593 | 121.593 | 0.000 |  |  |  | +7.9% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v7_frontk_k2 | ok | 202.522 | 202.522 | 114.506 | 88.016 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v7_frontk_k4 | ok | 208.177 | 208.177 | 116.181 | 91.996 |  |  |  | +2.8% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v7_frontk_k8 | ok | 221.053 | 221.053 | 120.201 | 100.852 |  |  |  | +9.1% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v7_frontk_k2 | ok | 97.672 | 97.672 | 97.672 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v7_frontk_k4 | ok | 104.616 | 104.616 | 104.616 | 0.000 |  |  |  | +7.1% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v7_frontk_k8 | ok | 110.519 | 110.519 | 110.519 | 0.000 |  |  |  | +13.2% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v7_frontk_k2 | ok | 203.862 | 203.862 | 99.343 | 104.519 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v7_frontk_k4 | ok | 213.978 | 213.978 | 103.514 | 110.464 |  |  |  | +5.0% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v7_frontk_k8 | ok | 223.233 | 223.233 | 109.265 | 113.967 |  |  |  | +9.5% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v7_frontk_k2 | ok | 124.142 | 124.142 | 124.142 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v7_frontk_k4 | ok | 129.906 | 129.906 | 129.905 | 0.000 |  |  |  | +4.6% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v7_frontk_k8 | ok | 134.319 | 134.319 | 134.318 | 0.000 |  |  |  | +8.2% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v7_frontk_k2 | ok | 2316.551 | 2316.551 | 152.154 | 2164.397 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v7_frontk_k4 | ok | 2335.064 | 2335.064 | 154.075 | 2180.989 |  |  |  | +0.8% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v7_frontk_k8 | ok | 2388.797 | 2388.797 | 156.204 | 2232.594 |  |  |  | +3.1% |  |  |  |  |
