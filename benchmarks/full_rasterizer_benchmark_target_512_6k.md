# Full Rasterizer Benchmark

Generated: 2026-04-22 12:25:05

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
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_refined_direct | 81.258 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_refined_auto | 95.208 |
| 512x512 | 1 | 6000 | layered_depth | forward | v5_batched | 6.254 |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v5_batched | 28.228 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_refined_direct | 12.594 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_upgrade_direct | 38.898 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v6_auto | 3111.863 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v6_upgrade_auto | 634.802 |
| 512x512 | 1 | 6000 | sparse_screen | forward | v6_auto | 22.226 |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v6_direct | 27.760 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_upgrade_auto | 583.891 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_refined_auto | 611.650 |
| 512x512 | 4 | 6000 | layered_depth | forward | v5_batched | 10.752 |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v5_batched | 37.673 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_direct | 24.428 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_refined_auto | 30.562 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v6_refined_auto | 1669.491 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v6_refined_auto | 1503.726 |
| 512x512 | 4 | 6000 | sparse_screen | forward | v6_direct | 15.010 |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v6_refined_direct | 38.187 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v5_batched | ok | 36.166 | 36.166 | 36.164 | 0.001 |  |  |  | +187.2% |  | -53.3% | +187.2% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_direct | ok | 77.502 | 77.502 | 77.498 | 0.003 |  |  |  | +515.4% |  |  | +515.4% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_auto | ok | 52.959 | 52.959 | 52.957 | 0.002 |  |  |  | +320.5% |  | -31.7% | +320.5% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 74.970 | 74.970 | 74.969 | 0.002 |  |  |  | +495.3% |  | -3.3% | +495.3% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 18.102 | 18.102 | 18.101 | 0.001 |  |  |  | +43.7% |  | -76.6% | +43.7% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_refined_direct | ok | 12.594 | 12.594 | 12.593 | 0.001 |  |  |  | +0.0% |  | -83.8% |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v6_refined_auto | ok | 17.042 | 17.042 | 17.041 | 0.001 |  |  |  | +35.3% |  | -78.0% | +35.3% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v5_batched | ok | 50.304 | 50.304 | 26.216 | 24.088 |  |  |  | +29.3% |  | -22.3% | -38.2% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_direct | ok | 64.735 | 64.735 | 40.899 | 23.836 |  |  |  | +66.4% |  |  | -20.5% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_auto | ok | 58.262 | 58.262 | 46.796 | 11.466 |  |  |  | +49.8% |  | -10.0% | -28.5% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 38.898 | 38.898 | 18.390 | 20.508 |  |  |  | +0.0% |  | -39.9% | -52.2% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 59.361 | 59.361 | 42.083 | 17.278 |  |  |  | +52.6% |  | -8.3% | -27.1% |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_refined_direct | ok | 81.432 | 81.432 | 59.440 | 21.991 |  |  |  | +109.3% |  | +25.8% |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v6_refined_auto | ok | 200.329 | 200.329 | 156.844 | 43.485 |  |  |  | +415.0% |  | +209.5% | +146.0% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v5_batched | ok | 34.593 | 34.593 | 34.590 | 0.003 |  |  |  | +55.6% |  | +34.6% | +47.7% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v6_direct | ok | 25.706 | 25.706 | 25.704 | 0.001 |  |  |  | +15.7% |  |  | +9.8% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v6_auto | ok | 22.226 | 22.226 | 22.225 | 0.001 |  |  |  | +0.0% |  | -13.5% | -5.1% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v6_upgrade_direct | ok | 41.967 | 41.967 | 41.966 | 0.002 |  |  |  | +88.8% |  | +63.3% | +79.2% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v6_upgrade_auto | ok | 119.756 | 119.756 | 119.755 | 0.001 |  |  |  | +438.8% |  | +365.9% | +411.4% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v6_refined_direct | ok | 23.417 | 23.417 | 23.415 | 0.002 |  |  |  | +5.4% |  | -8.9% |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v6_refined_auto | ok | 40.095 | 40.095 | 40.092 | 0.003 |  |  |  | +80.4% |  | +56.0% | +71.2% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v5_batched | ok | 33.235 | 33.235 | 21.605 | 11.629 |  |  |  | +19.7% |  | +19.7% | +5.6% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v6_direct | ok | 27.760 | 27.760 | 18.058 | 9.702 |  |  |  | +0.0% |  |  | -11.8% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v6_auto | ok | 37.378 | 37.378 | 24.120 | 13.259 |  |  |  | +34.7% |  | +34.7% | +18.8% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 42.765 | 42.765 | 33.850 | 8.915 |  |  |  | +54.1% |  | +54.1% | +35.9% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 37.109 | 37.109 | 27.623 | 9.487 |  |  |  | +33.7% |  | +33.7% | +17.9% |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v6_refined_direct | ok | 31.471 | 31.471 | 22.203 | 9.268 |  |  |  | +13.4% |  | +13.4% |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v6_refined_auto | ok | 45.965 | 45.965 | 33.816 | 12.149 |  |  |  | +65.6% |  | +65.6% | +46.1% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v5_batched | ok | 172.243 | 172.243 | 172.242 | 0.001 |  |  |  | +112.0% |  | -25.6% | +112.0% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_direct | ok | 231.549 | 231.549 | 231.548 | 0.002 |  |  |  | +185.0% |  |  | +185.0% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_auto | ok | 191.801 | 191.801 | 191.800 | 0.002 |  |  |  | +136.0% |  | -17.2% | +136.0% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 185.598 | 185.598 | 185.596 | 0.002 |  |  |  | +128.4% |  | -19.8% | +128.4% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 223.296 | 223.296 | 223.294 | 0.002 |  |  |  | +174.8% |  | -3.6% | +174.8% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_refined_direct | ok | 81.258 | 81.258 | 81.258 | 0.001 |  |  |  | +0.0% |  | -64.9% |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v6_refined_auto | ok | 81.594 | 81.594 | 81.593 | 0.001 |  |  |  | +0.4% |  | -64.8% | +0.4% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v5_batched | ok | 443.084 | 443.084 | 393.796 | 49.288 |  |  |  | +365.4% |  | +59.1% | +59.3% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_direct | ok | 278.464 | 278.464 | 258.009 | 20.455 |  |  |  | +192.5% |  |  | +0.1% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_auto | ok | 569.052 | 569.052 | 525.460 | 43.592 |  |  |  | +497.7% |  | +104.4% | +104.7% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 283.785 | 283.785 | 259.822 | 23.963 |  |  |  | +198.1% |  | +1.9% | +2.1% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 355.673 | 355.673 | 315.841 | 39.831 |  |  |  | +273.6% |  | +27.7% | +27.9% |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_refined_direct | ok | 278.059 | 278.059 | 243.897 | 34.162 |  |  |  | +192.1% |  | -0.1% |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v6_refined_auto | ok | 95.208 | 95.208 | 78.812 | 16.395 |  |  |  | +0.0% |  | -65.8% | -65.8% |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v5_batched | ok | 6.254 | 6.254 | 6.253 | 0.001 |  |  |  | +0.0% |  | -90.9% | -59.5% |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v6_direct | ok | 68.583 | 68.583 | 68.582 | 0.002 |  |  |  | +996.7% |  |  | +344.6% |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v6_auto | ok | 51.230 | 51.230 | 51.228 | 0.002 |  |  |  | +719.2% |  | -25.3% | +232.1% |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v6_upgrade_direct | ok | 57.982 | 57.982 | 57.981 | 0.001 |  |  |  | +827.2% |  | -15.5% | +275.9% |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v6_upgrade_auto | ok | 32.472 | 32.472 | 32.471 | 0.001 |  |  |  | +419.2% |  | -52.7% | +110.5% |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v6_refined_direct | ok | 15.426 | 15.426 | 15.426 | 0.001 |  |  |  | +146.7% |  | -77.5% |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v6_refined_auto | ok | 29.825 | 29.825 | 29.824 | 0.002 |  |  |  | +376.9% |  | -56.5% | +93.3% |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v5_batched | ok | 28.228 | 28.228 | 16.251 | 11.977 |  |  |  | +0.0% |  | -64.6% | -4.2% |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v6_direct | ok | 79.674 | 79.674 | 70.045 | 9.629 |  |  |  | +182.2% |  |  | +170.5% |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v6_auto | ok | 176.379 | 176.379 | 147.676 | 28.703 |  |  |  | +524.8% |  | +121.4% | +498.9% |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v6_upgrade_direct | ok | 66.153 | 66.153 | 53.327 | 12.826 |  |  |  | +134.4% |  | -17.0% | +124.6% |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v6_upgrade_auto | ok | 62.151 | 62.151 | 45.599 | 16.552 |  |  |  | +120.2% |  | -22.0% | +111.0% |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v6_refined_direct | ok | 29.452 | 29.452 | 17.610 | 11.842 |  |  |  | +4.3% |  | -63.0% |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v6_refined_auto | ok | 54.509 | 54.509 | 40.937 | 13.572 |  |  |  | +93.1% |  | -31.6% | +85.1% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v5_batched | ok | 4662.012 | 4662.012 | 4661.997 | 0.016 |  |  |  | +49.8% |  | +29.9% | -14.0% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v6_direct | ok | 3588.763 | 3588.763 | 3588.762 | 0.001 |  |  |  | +15.3% |  |  | -33.8% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v6_auto | ok | 3111.863 | 3111.863 | 3111.861 | 0.001 |  |  |  | +0.0% |  | -13.3% | -42.6% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v6_upgrade_direct | ok | 3386.239 | 3386.239 | 3386.230 | 0.009 |  |  |  | +8.8% |  | -5.6% | -37.5% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v6_upgrade_auto | ok | 6496.055 | 6496.055 | 6496.026 | 0.029 |  |  |  | +108.8% |  | +81.0% | +19.8% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v6_refined_direct | ok | 5420.513 | 5420.513 | 5420.502 | 0.011 |  |  |  | +74.2% |  | +51.0% |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v6_refined_auto | ok | 4477.730 | 4477.730 | 4477.720 | 0.010 |  |  |  | +43.9% |  | +24.8% | -17.4% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v5_batched | ok | 5433.776 | 5433.776 | 4910.342 | 523.435 |  |  |  | +756.0% |  | +19.0% | +54.0% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v6_direct | ok | 4566.481 | 4566.481 | 4423.568 | 142.912 |  |  |  | +619.4% |  |  | +29.5% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v6_auto | ok | 4899.493 | 4899.493 | 4601.436 | 298.057 |  |  |  | +671.8% |  | +7.3% | +38.9% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 3946.802 | 3946.802 | 3712.665 | 234.137 |  |  |  | +521.7% |  | -13.6% | +11.9% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 634.802 | 634.802 | 555.550 | 79.252 |  |  |  | +0.0% |  | -86.1% | -82.0% |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v6_refined_direct | ok | 3527.333 | 3527.333 | 3366.045 | 161.288 |  |  |  | +455.7% |  | -22.8% |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v6_refined_auto | ok | 4216.091 | 4216.091 | 4048.977 | 167.114 |  |  |  | +564.2% |  | -7.7% | +19.5% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v5_batched | ok | 46.283 | 46.283 | 46.281 | 0.002 |  |  |  | +89.5% |  | +89.5% | -33.6% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_direct | ok | 24.428 | 24.428 | 24.427 | 0.001 |  |  |  | +0.0% |  |  | -64.9% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_auto | ok | 39.227 | 39.227 | 39.226 | 0.001 |  |  |  | +60.6% |  | +60.6% | -43.7% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 57.277 | 57.277 | 57.275 | 0.002 |  |  |  | +134.5% |  | +134.5% | -17.8% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 45.791 | 45.791 | 45.790 | 0.001 |  |  |  | +87.5% |  | +87.5% | -34.3% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_refined_direct | ok | 69.684 | 69.684 | 69.683 | 0.002 |  |  |  | +185.3% |  | +185.3% |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v6_refined_auto | ok | 32.805 | 32.805 | 32.803 | 0.001 |  |  |  | +34.3% |  | +34.3% | -52.9% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v5_batched | ok | 223.393 | 223.393 | 162.705 | 60.688 |  |  |  | +631.0% |  | +217.9% | +182.7% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_direct | ok | 70.282 | 70.282 | 45.072 | 25.210 |  |  |  | +130.0% |  |  | -11.1% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_auto | ok | 311.299 | 311.299 | 179.198 | 132.101 |  |  |  | +918.6% |  | +342.9% | +293.9% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 85.708 | 85.708 | 44.290 | 41.418 |  |  |  | +180.4% |  | +21.9% | +8.5% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 172.250 | 172.250 | 92.951 | 79.299 |  |  |  | +463.6% |  | +145.1% | +118.0% |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_refined_direct | ok | 79.021 | 79.021 | 49.750 | 29.271 |  |  |  | +158.6% |  | +12.4% |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v6_refined_auto | ok | 30.562 | 30.562 | 15.137 | 15.425 |  |  |  | +0.0% |  | -56.5% | -61.3% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v5_batched | ok | 421.050 | 421.050 | 421.049 | 0.001 |  |  |  | +2705.2% |  | +2705.2% | +618.4% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v6_direct | ok | 15.010 | 15.010 | 15.009 | 0.001 |  |  |  | +0.0% |  |  | -74.4% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v6_auto | ok | 25.075 | 25.075 | 25.074 | 0.001 |  |  |  | +67.1% |  | +67.1% | -57.2% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v6_upgrade_direct | ok | 81.464 | 81.464 | 81.462 | 0.002 |  |  |  | +442.7% |  | +442.7% | +39.0% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v6_upgrade_auto | ok | 50.378 | 50.378 | 50.376 | 0.001 |  |  |  | +235.6% |  | +235.6% | -14.0% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v6_refined_direct | ok | 58.612 | 58.612 | 58.610 | 0.001 |  |  |  | +290.5% |  | +290.5% |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v6_refined_auto | ok | 52.932 | 52.932 | 52.930 | 0.002 |  |  |  | +252.6% |  | +252.6% | -9.7% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v5_batched | ok | 74.587 | 74.587 | 41.420 | 33.167 |  |  |  | +95.3% |  | -55.1% | +95.3% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v6_direct | ok | 165.936 | 165.936 | 109.226 | 56.710 |  |  |  | +334.5% |  |  | +334.5% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v6_auto | ok | 149.340 | 149.340 | 108.616 | 40.724 |  |  |  | +291.1% |  | -10.0% | +291.1% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 51.102 | 51.102 | 31.670 | 19.432 |  |  |  | +33.8% |  | -69.2% | +33.8% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 63.115 | 63.115 | 33.853 | 29.262 |  |  |  | +65.3% |  | -62.0% | +65.3% |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v6_refined_direct | ok | 38.187 | 38.187 | 21.367 | 16.820 |  |  |  | +0.0% |  | -77.0% |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v6_refined_auto | ok | 70.930 | 70.930 | 50.804 | 20.126 |  |  |  | +85.7% |  | -57.3% | +85.7% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v5_batched | ok | 613.284 | 613.284 | 613.282 | 0.001 |  |  |  | +5.0% |  | +2.5% | -6.0% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_direct | ok | 598.346 | 598.346 | 598.344 | 0.002 |  |  |  | +2.5% |  |  | -8.3% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_auto | ok | 609.569 | 609.569 | 609.567 | 0.002 |  |  |  | +4.4% |  | +1.9% | -6.6% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 673.940 | 673.940 | 673.938 | 0.001 |  |  |  | +15.4% |  | +12.6% | +3.3% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 583.891 | 583.891 | 583.890 | 0.001 |  |  |  | +0.0% |  | -2.4% | -10.5% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_refined_direct | ok | 652.553 | 652.553 | 652.552 | 0.001 |  |  |  | +11.8% |  | +9.1% |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v6_refined_auto | ok | 627.593 | 627.593 | 627.591 | 0.002 |  |  |  | +7.5% |  | +4.9% | -3.8% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v5_batched | ok | 817.972 | 817.972 | 725.001 | 92.971 |  |  |  | +33.7% |  | +12.4% | +26.2% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_direct | ok | 727.797 | 727.797 | 687.875 | 39.921 |  |  |  | +19.0% |  |  | +12.3% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_auto | ok | 855.412 | 855.412 | 809.738 | 45.674 |  |  |  | +39.9% |  | +17.5% | +32.0% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 805.619 | 805.619 | 750.579 | 55.040 |  |  |  | +31.7% |  | +10.7% | +24.3% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 897.185 | 897.185 | 840.130 | 57.054 |  |  |  | +46.7% |  | +23.3% | +38.4% |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_refined_direct | ok | 648.248 | 648.248 | 593.858 | 54.390 |  |  |  | +6.0% |  | -10.9% |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v6_refined_auto | ok | 611.650 | 611.650 | 580.896 | 30.754 |  |  |  | +0.0% |  | -16.0% | -5.6% |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v5_batched | ok | 10.752 | 10.752 | 10.751 | 0.000 |  |  |  | +0.0% |  | -37.6% | -53.1% |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v6_direct | ok | 17.222 | 17.222 | 17.221 | 0.001 |  |  |  | +60.2% |  |  | -24.8% |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v6_auto | ok | 23.429 | 23.429 | 23.427 | 0.002 |  |  |  | +117.9% |  | +36.0% | +2.3% |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v6_upgrade_direct | ok | 29.438 | 29.438 | 29.438 | 0.001 |  |  |  | +173.8% |  | +70.9% | +28.5% |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v6_upgrade_auto | ok | 30.486 | 30.486 | 30.486 | 0.001 |  |  |  | +183.5% |  | +77.0% | +33.1% |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v6_refined_direct | ok | 22.906 | 22.906 | 22.906 | 0.001 |  |  |  | +113.0% |  | +33.0% |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v6_refined_auto | ok | 33.042 | 33.042 | 33.040 | 0.001 |  |  |  | +207.3% |  | +91.9% | +44.2% |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v5_batched | ok | 37.673 | 37.673 | 14.429 | 23.244 |  |  |  | +0.0% |  | -64.3% | -73.5% |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v6_direct | ok | 105.530 | 105.530 | 71.179 | 34.351 |  |  |  | +180.1% |  |  | -25.7% |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v6_auto | ok | 150.621 | 150.621 | 115.828 | 34.794 |  |  |  | +299.8% |  | +42.7% | +6.0% |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v6_upgrade_direct | ok | 101.680 | 101.680 | 75.483 | 26.197 |  |  |  | +169.9% |  | -3.6% | -28.4% |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v6_upgrade_auto | ok | 81.005 | 81.005 | 53.527 | 27.478 |  |  |  | +115.0% |  | -23.2% | -43.0% |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v6_refined_direct | ok | 142.076 | 142.076 | 102.768 | 39.309 |  |  |  | +277.1% |  | +34.6% |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v6_refined_auto | ok | 103.476 | 103.476 | 80.699 | 22.776 |  |  |  | +174.7% |  | -1.9% | -27.2% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v5_batched | ok | 13232.613 | 13232.613 | 13232.591 | 0.022 |  |  |  | +692.6% |  | +23.5% | +195.2% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v6_direct | ok | 10712.930 | 10712.930 | 10712.921 | 0.009 |  |  |  | +541.7% |  |  | +139.0% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v6_auto | ok | 12426.512 | 12426.512 | 12426.502 | 0.010 |  |  |  | +644.3% |  | +16.0% | +177.2% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v6_upgrade_direct | ok | 11162.230 | 11162.230 | 11162.216 | 0.014 |  |  |  | +568.6% |  | +4.2% | +149.0% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v6_upgrade_auto | ok | 1826.983 | 1826.983 | 1826.982 | 0.001 |  |  |  | +9.4% |  | -82.9% | -59.2% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v6_refined_direct | ok | 4482.506 | 4482.506 | 4482.500 | 0.006 |  |  |  | +168.5% |  | -58.2% |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v6_refined_auto | ok | 1669.491 | 1669.491 | 1669.490 | 0.001 |  |  |  | +0.0% |  | -84.4% | -62.8% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v5_batched | ok | 1686.457 | 1686.457 | 1377.927 | 308.530 |  |  |  | +12.2% |  | -27.8% | -88.5% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v6_direct | ok | 2336.763 | 2336.763 | 2113.666 | 223.097 |  |  |  | +55.4% |  |  | -84.1% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v6_auto | ok | 1604.261 | 1604.261 | 1356.604 | 247.657 |  |  |  | +6.7% |  | -31.3% | -89.1% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 9935.023 | 9935.023 | 9698.209 | 236.814 |  |  |  | +560.7% |  | +325.2% | -32.4% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 11528.975 | 11528.975 | 11178.727 | 350.248 |  |  |  | +666.7% |  | +393.4% | -21.6% |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v6_refined_direct | ok | 14701.365 | 14701.365 | 14265.642 | 435.723 |  |  |  | +877.7% |  | +529.1% |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v6_refined_auto | ok | 1503.726 | 1503.726 | 1282.881 | 220.845 |  |  |  | +0.0% |  | -35.6% | -89.8% |  |
