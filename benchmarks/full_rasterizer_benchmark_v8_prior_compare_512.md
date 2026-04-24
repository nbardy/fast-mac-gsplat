# Full Rasterizer Benchmark

Generated: 2026-04-24 14:30:37

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
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_direct | 18.471 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v8_direct | 28.207 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v8_direct | 4.659 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v8_direct | 5.254 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v8_direct | 106.281 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_direct | 86.929 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v8_direct | 4.257 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v8_direct | 20.305 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_direct | ok | 5.907 | 5.122 | 5.907 | 0.000 |  |  |  | +26.8% |  |  | -31.0% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 8.186 | 7.402 | 8.186 | 0.000 |  |  |  | +75.7% |  | +38.6% | -4.4% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_refined_direct | ok | 8.564 | 8.249 | 8.563 | 0.000 |  |  |  | +83.8% |  | +45.0% |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v8_direct | ok | 4.659 | 4.499 | 4.659 | 0.000 |  |  |  | +0.0% |  | -21.1% | -45.6% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_direct | ok | 8.722 | 8.337 | 5.447 | 3.275 |  |  |  | +66.0% |  |  | -36.3% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 9.246 | 9.310 | 5.823 | 3.422 |  |  |  | +76.0% |  | +6.0% | -32.5% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_refined_direct | ok | 13.695 | 11.522 | 9.128 | 4.566 |  |  |  | +160.7% |  | +57.0% |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v8_direct | ok | 5.254 | 5.121 | 2.607 | 2.647 |  |  |  | +0.0% |  | -39.8% | -61.6% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_direct | ok | 18.471 | 18.287 | 18.471 | 0.000 |  |  |  | +0.0% |  |  | -14.3% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 21.707 | 21.379 | 21.706 | 0.000 |  |  |  | +17.5% |  | +17.5% | +0.7% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_refined_direct | ok | 21.557 | 22.180 | 21.557 | 0.000 |  |  |  | +16.7% |  | +16.7% |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v8_direct | ok | 21.419 | 18.595 | 21.419 | 0.000 |  |  |  | +16.0% |  | +16.0% | -0.6% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_direct | ok | 33.462 | 34.242 | 21.460 | 12.002 |  |  |  | +18.6% |  |  | -22.7% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 38.069 | 38.620 | 24.304 | 13.765 |  |  |  | +35.0% |  | +13.8% | -12.0% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_refined_direct | ok | 43.276 | 38.276 | 27.505 | 15.771 |  |  |  | +53.4% |  | +29.3% |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v8_direct | ok | 28.207 | 26.546 | 18.787 | 9.421 |  |  |  | +0.0% |  | -15.7% | -34.8% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_direct | ok | 15.373 | 15.567 | 15.372 | 0.000 |  |  |  | +261.1% |  |  | +140.1% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 6.662 | 6.170 | 6.662 | 0.000 |  |  |  | +56.5% |  | -56.7% | +4.1% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_refined_direct | ok | 6.401 | 6.150 | 6.401 | 0.000 |  |  |  | +50.4% |  | -58.4% |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v8_direct | ok | 4.257 | 4.043 | 4.257 | 0.000 |  |  |  | +0.0% |  | -72.3% | -33.5% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_direct | ok | 25.070 | 24.524 | 9.041 | 16.029 |  |  |  | +23.5% |  |  | +11.7% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 23.220 | 23.293 | 8.135 | 15.085 |  |  |  | +14.4% |  | -7.4% | +3.4% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_refined_direct | ok | 22.450 | 22.605 | 7.756 | 14.693 |  |  |  | +10.6% |  | -10.5% |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v8_direct | ok | 20.305 | 20.808 | 5.861 | 14.445 |  |  |  | +0.0% |  | -19.0% | -9.6% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_direct | ok | 128.064 | 132.695 | 128.063 | 0.001 |  |  |  | +20.5% |  |  | -10.1% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 125.366 | 126.297 | 125.366 | 0.000 |  |  |  | +18.0% |  | -2.1% | -12.0% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_refined_direct | ok | 142.455 | 144.714 | 142.454 | 0.001 |  |  |  | +34.0% |  | +11.2% |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v8_direct | ok | 106.281 | 106.147 | 106.280 | 0.000 |  |  |  | +0.0% |  | -17.0% | -25.4% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_direct | ok | 86.929 | 85.441 | 61.557 | 25.372 |  |  |  | +0.0% |  |  | -23.9% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 112.591 | 114.548 | 81.949 | 30.642 |  |  |  | +29.5% |  | +29.5% | -1.4% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_refined_direct | ok | 114.160 | 116.778 | 80.527 | 33.633 |  |  |  | +31.3% |  | +31.3% |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v8_direct | ok | 93.183 | 93.474 | 69.060 | 24.123 |  |  |  | +7.2% |  | +7.2% | -18.4% |  |
