# Full Rasterizer Benchmark

Generated: 2026-04-22 00:49:16

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `90.0`
- accuracy checks: `True`
- accuracy max work items: `5000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_direct | 5.572 |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_direct | 7.688 |
| 128x128 | 1 | 64 | layered_depth | forward | v6_direct | 4.480 |
| 128x128 | 1 | 64 | layered_depth | forward_backward | v6_direct | 7.421 |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_direct | 4.936 |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_direct | 7.092 |
| 128x128 | 1 | 64 | overflow_adversarial | forward | v6_direct | 5.517 |
| 128x128 | 1 | 64 | overflow_adversarial | forward_backward | v6_refined_direct | 8.746 |
| 128x128 | 1 | 64 | sparse_screen | forward | v6_upgrade_auto | 5.436 |
| 128x128 | 1 | 64 | sparse_screen | forward_backward | v6_direct | 7.900 |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward | v6_upgrade_direct | 5.732 |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward_backward | v6_direct | 8.789 |
| 128x128 | 4 | 64 | layered_depth | forward | v6_refined_auto | 5.706 |
| 128x128 | 4 | 64 | layered_depth | forward_backward | v6_upgrade_auto | 9.112 |
| 128x128 | 4 | 64 | microbench_uniform_random | forward | v6_refined_auto | 5.617 |
| 128x128 | 4 | 64 | microbench_uniform_random | forward_backward | v6_direct | 8.412 |
| 128x128 | 4 | 64 | overflow_adversarial | forward | v6_refined_auto | 6.151 |
| 128x128 | 4 | 64 | overflow_adversarial | forward_backward | v6_direct | 10.183 |
| 128x128 | 4 | 64 | sparse_screen | forward | v6_upgrade_auto | 6.223 |
| 128x128 | 4 | 64 | sparse_screen | forward_backward | v6_upgrade_auto | 8.373 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_direct | ok | 4.936 | 4.936 | 4.936 | 0.000 | ok | 1.192e-07 |  | +0.0% |  |  | -4.5% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_auto | ok | 6.649 | 6.649 | 6.648 | 0.000 | ok | 1.192e-07 |  | +34.7% |  | +34.7% | +28.7% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 8.452 | 8.452 | 8.452 | 0.000 | ok | 1.192e-07 |  | +71.2% |  | +71.2% | +63.6% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 6.081 | 6.081 | 6.080 | 0.000 | ok | 1.192e-07 |  | +23.2% |  | +23.2% | +17.7% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_refined_direct | ok | 5.166 | 5.166 | 5.166 | 0.000 | ok | 1.192e-07 |  | +4.7% |  | +4.7% |  |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_refined_auto | ok | 6.819 | 6.819 | 6.819 | 0.000 | ok | 1.192e-07 |  | +38.1% |  | +38.1% | +32.0% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_direct | ok | 7.092 | 7.092 | 5.374 | 1.718 | ok | 1.192e-07 | 3.725e-09 | +0.0% |  |  | -15.9% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_auto | ok | 8.674 | 8.674 | 7.296 | 1.378 | ok | 1.192e-07 | 3.725e-09 | +22.3% |  | +22.3% | +2.9% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 9.173 | 9.173 | 6.976 | 2.197 | ok | 1.192e-07 | 3.725e-09 | +29.3% |  | +29.3% | +8.8% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 8.232 | 8.232 | 6.897 | 1.334 | ok | 1.192e-07 | 3.725e-09 | +16.1% |  | +16.1% | -2.3% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_refined_direct | ok | 8.429 | 8.429 | 6.983 | 1.446 | ok | 1.192e-07 | 3.725e-09 | +18.9% |  | +18.9% |  |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_refined_auto | ok | 8.980 | 8.980 | 7.375 | 1.604 | ok | 1.192e-07 | 3.725e-09 | +26.6% |  | +26.6% | +6.5% |  |
| 128x128 | 1 | 64 | sparse_screen | forward | v6_direct | ok | 6.035 | 6.035 | 6.035 | 0.000 | ok | 1.788e-07 |  | +11.0% |  |  | -7.1% |  |
| 128x128 | 1 | 64 | sparse_screen | forward | v6_auto | ok | 5.867 | 5.867 | 5.866 | 0.000 | ok | 1.788e-07 |  | +7.9% |  | -2.8% | -9.7% |  |
| 128x128 | 1 | 64 | sparse_screen | forward | v6_upgrade_direct | ok | 5.946 | 5.946 | 5.946 | 0.000 | ok | 1.788e-07 |  | +9.4% |  | -1.5% | -8.5% |  |
| 128x128 | 1 | 64 | sparse_screen | forward | v6_upgrade_auto | ok | 5.436 | 5.436 | 5.435 | 0.000 | ok | 1.788e-07 |  | +0.0% |  | -9.9% | -16.4% |  |
| 128x128 | 1 | 64 | sparse_screen | forward | v6_refined_direct | ok | 6.499 | 6.499 | 6.499 | 0.000 | ok | 1.788e-07 |  | +19.6% |  | +7.7% |  |  |
| 128x128 | 1 | 64 | sparse_screen | forward | v6_refined_auto | ok | 7.129 | 7.129 | 7.129 | 0.000 | ok | 1.788e-07 |  | +31.2% |  | +18.1% | +9.7% |  |
| 128x128 | 1 | 64 | sparse_screen | forward_backward | v6_direct | ok | 7.900 | 7.900 | 6.514 | 1.386 | ok | 1.788e-07 | 9.313e-10 | +0.0% |  |  | -15.0% |  |
| 128x128 | 1 | 64 | sparse_screen | forward_backward | v6_auto | ok | 8.453 | 8.453 | 7.107 | 1.346 | ok | 1.788e-07 | 1.863e-09 | +7.0% |  | +7.0% | -9.0% |  |
| 128x128 | 1 | 64 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 8.629 | 8.629 | 7.121 | 1.508 | ok | 1.788e-07 | 1.863e-09 | +9.2% |  | +9.2% | -7.1% |  |
| 128x128 | 1 | 64 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 8.674 | 8.674 | 6.822 | 1.852 | ok | 1.788e-07 | 9.313e-10 | +9.8% |  | +9.8% | -6.6% |  |
| 128x128 | 1 | 64 | sparse_screen | forward_backward | v6_refined_direct | ok | 9.291 | 9.291 | 7.865 | 1.427 | ok | 1.788e-07 | 1.863e-09 | +17.6% |  | +17.6% |  |  |
| 128x128 | 1 | 64 | sparse_screen | forward_backward | v6_refined_auto | ok | 9.255 | 9.255 | 7.176 | 2.080 | ok | 1.788e-07 | 1.863e-09 | +17.2% |  | +17.2% | -0.4% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_direct | ok | 5.572 | 5.572 | 5.572 | 0.000 | ok | 6.986e-05 |  | +0.0% |  |  | -5.1% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_auto | ok | 6.919 | 6.919 | 6.919 | 0.000 | ok | 6.986e-05 |  | +24.2% |  | +24.2% | +17.9% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 6.895 | 6.895 | 6.894 | 0.000 | ok | 6.986e-05 |  | +23.7% |  | +23.7% | +17.5% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 8.662 | 8.662 | 8.662 | 0.000 | ok | 6.986e-05 |  | +55.5% |  | +55.5% | +47.6% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_refined_direct | ok | 5.870 | 5.870 | 5.870 | 0.000 | ok | 6.986e-05 |  | +5.3% |  | +5.3% |  |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_refined_auto | ok | 7.438 | 7.438 | 7.437 | 0.000 | ok | 6.986e-05 |  | +33.5% |  | +33.5% | +26.7% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_direct | ok | 7.688 | 7.688 | 5.885 | 1.803 | ok | 6.986e-05 | 1.757e-05 | +0.0% |  |  | -23.4% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_auto | ok | 9.825 | 9.825 | 7.168 | 2.657 | ok | 6.986e-05 | 1.758e-05 | +27.8% |  | +27.8% | -2.1% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 8.238 | 8.238 | 6.419 | 1.819 | ok | 6.986e-05 | 1.757e-05 | +7.1% |  | +7.1% | -17.9% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 11.525 | 11.525 | 9.622 | 1.902 | ok | 6.986e-05 | 1.758e-05 | +49.9% |  | +49.9% | +14.9% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_refined_direct | ok | 10.033 | 10.033 | 8.389 | 1.644 | ok | 6.986e-05 | 1.757e-05 | +30.5% |  | +30.5% |  |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_refined_auto | ok | 11.121 | 11.121 | 9.445 | 1.676 | ok | 6.986e-05 | 1.758e-05 | +44.6% |  | +44.6% | +10.8% |  |
| 128x128 | 1 | 64 | layered_depth | forward | v6_direct | ok | 4.480 | 4.480 | 4.480 | 0.000 | ok | 4.679e-05 |  | +0.0% |  |  | -26.3% |  |
| 128x128 | 1 | 64 | layered_depth | forward | v6_auto | ok | 6.168 | 6.168 | 6.168 | 0.000 | ok | 4.679e-05 |  | +37.7% |  | +37.7% | +1.4% |  |
| 128x128 | 1 | 64 | layered_depth | forward | v6_upgrade_direct | ok | 6.915 | 6.915 | 6.915 | 0.000 | ok | 4.679e-05 |  | +54.4% |  | +54.4% | +13.7% |  |
| 128x128 | 1 | 64 | layered_depth | forward | v6_upgrade_auto | ok | 5.412 | 5.412 | 5.412 | 0.000 | ok | 4.679e-05 |  | +20.8% |  | +20.8% | -11.0% |  |
| 128x128 | 1 | 64 | layered_depth | forward | v6_refined_direct | ok | 6.082 | 6.082 | 6.081 | 0.000 | ok | 4.679e-05 |  | +35.8% |  | +35.8% |  |  |
| 128x128 | 1 | 64 | layered_depth | forward | v6_refined_auto | ok | 6.235 | 6.235 | 6.235 | 0.000 | ok | 4.679e-05 |  | +39.2% |  | +39.2% | +2.5% |  |
| 128x128 | 1 | 64 | layered_depth | forward_backward | v6_direct | ok | 7.421 | 7.421 | 5.654 | 1.767 | ok | 4.679e-05 | 1.587e-06 | +0.0% |  |  | -10.3% |  |
| 128x128 | 1 | 64 | layered_depth | forward_backward | v6_auto | ok | 9.535 | 9.535 | 7.858 | 1.676 | ok | 4.679e-05 | 1.583e-06 | +28.5% |  | +28.5% | +15.2% |  |
| 128x128 | 1 | 64 | layered_depth | forward_backward | v6_upgrade_direct | ok | 8.951 | 8.951 | 7.655 | 1.296 | ok | 4.679e-05 | 1.583e-06 | +20.6% |  | +20.6% | +8.1% |  |
| 128x128 | 1 | 64 | layered_depth | forward_backward | v6_upgrade_auto | ok | 8.616 | 8.616 | 7.228 | 1.388 | ok | 4.679e-05 | 1.580e-06 | +16.1% |  | +16.1% | +4.1% |  |
| 128x128 | 1 | 64 | layered_depth | forward_backward | v6_refined_direct | ok | 8.277 | 8.277 | 6.004 | 2.273 | ok | 4.679e-05 | 1.580e-06 | +11.5% |  | +11.5% |  |  |
| 128x128 | 1 | 64 | layered_depth | forward_backward | v6_refined_auto | ok | 9.287 | 9.287 | 7.481 | 1.806 | ok | 4.679e-05 | 1.580e-06 | +25.1% |  | +25.1% | +12.2% |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward | v6_direct | ok | 5.517 | 5.517 | 5.517 | 0.000 | ok | 8.363e-05 |  | +0.0% |  |  | -6.6% |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward | v6_auto | ok | 6.682 | 6.682 | 6.681 | 0.000 | ok | 8.363e-05 |  | +21.1% |  | +21.1% | +13.1% |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward | v6_upgrade_direct | ok | 6.941 | 6.941 | 6.941 | 0.000 | ok | 8.363e-05 |  | +25.8% |  | +25.8% | +17.5% |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward | v6_upgrade_auto | ok | 6.311 | 6.311 | 6.311 | 0.000 | ok | 8.363e-05 |  | +14.4% |  | +14.4% | +6.9% |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward | v6_refined_direct | ok | 5.906 | 5.906 | 5.906 | 0.000 | ok | 8.363e-05 |  | +7.0% |  | +7.0% |  |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward | v6_refined_auto | ok | 7.591 | 7.591 | 7.590 | 0.000 | ok | 8.363e-05 |  | +37.6% |  | +37.6% | +28.5% |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward_backward | v6_direct | ok | 9.019 | 9.019 | 7.143 | 1.876 | ok | 8.363e-05 | 1.743e-02 | +3.1% |  |  | +3.1% |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward_backward | v6_auto | ok | 10.631 | 10.631 | 8.250 | 2.381 | ok | 8.363e-05 | 1.742e-02 | +21.6% |  | +17.9% | +21.6% |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 10.542 | 10.542 | 8.284 | 2.258 | ok | 8.363e-05 | 1.742e-02 | +20.5% |  | +16.9% | +20.5% |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 9.669 | 9.669 | 7.523 | 2.146 | ok | 8.363e-05 | 1.742e-02 | +10.6% |  | +7.2% | +10.6% |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward_backward | v6_refined_direct | ok | 8.746 | 8.746 | 6.878 | 1.869 | ok | 8.363e-05 | 1.742e-02 | +0.0% |  | -3.0% |  |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward_backward | v6_refined_auto | ok | 8.961 | 8.961 | 7.023 | 1.937 | ok | 8.363e-05 | 1.742e-02 | +2.5% |  | -0.6% | +2.5% |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward | v6_direct | ok | 6.292 | 6.292 | 6.291 | 0.000 | ok | 1.192e-07 |  | +12.0% |  |  | +0.4% |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward | v6_auto | ok | 7.160 | 7.160 | 7.160 | 0.000 | ok | 1.192e-07 |  | +27.5% |  | +13.8% | +14.3% |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 6.296 | 6.296 | 6.296 | 0.000 | ok | 1.192e-07 |  | +12.1% |  | +0.1% | +0.5% |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 6.192 | 6.192 | 6.192 | 0.000 | ok | 1.192e-07 |  | +10.2% |  | -1.6% | -1.1% |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward | v6_refined_direct | ok | 6.264 | 6.264 | 6.264 | 0.000 | ok | 1.192e-07 |  | +11.5% |  | -0.4% |  |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward | v6_refined_auto | ok | 5.617 | 5.617 | 5.617 | 0.000 | ok | 1.192e-07 |  | +0.0% |  | -10.7% | -10.3% |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward_backward | v6_direct | ok | 8.412 | 8.412 | 6.629 | 1.783 | ok | 1.192e-07 | 1.863e-09 | +0.0% |  |  | -7.1% |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward_backward | v6_auto | ok | 10.481 | 10.481 | 8.668 | 1.812 | ok | 1.192e-07 | 1.863e-09 | +24.6% |  | +24.6% | +15.7% |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 9.729 | 9.729 | 7.529 | 2.200 | ok | 1.192e-07 | 2.794e-09 | +15.7% |  | +15.7% | +7.4% |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 8.468 | 8.468 | 6.038 | 2.430 | ok | 1.192e-07 | 2.794e-09 | +0.7% |  | +0.7% | -6.5% |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward_backward | v6_refined_direct | ok | 9.055 | 9.055 | 6.972 | 2.084 | ok | 1.192e-07 | 1.863e-09 | +7.6% |  | +7.6% |  |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward_backward | v6_refined_auto | ok | 9.079 | 9.079 | 6.634 | 2.444 | ok | 1.192e-07 | 1.863e-09 | +7.9% |  | +7.9% | +0.3% |  |
| 128x128 | 4 | 64 | sparse_screen | forward | v6_direct | ok | 6.650 | 6.650 | 6.650 | 0.000 | ok | 1.788e-07 |  | +6.8% |  |  | -6.1% |  |
| 128x128 | 4 | 64 | sparse_screen | forward | v6_auto | ok | 6.520 | 6.520 | 6.520 | 0.000 | ok | 1.788e-07 |  | +4.8% |  | -1.9% | -7.9% |  |
| 128x128 | 4 | 64 | sparse_screen | forward | v6_upgrade_direct | ok | 6.739 | 6.739 | 6.739 | 0.000 | ok | 1.788e-07 |  | +8.3% |  | +1.3% | -4.8% |  |
| 128x128 | 4 | 64 | sparse_screen | forward | v6_upgrade_auto | ok | 6.223 | 6.223 | 6.223 | 0.000 | ok | 1.788e-07 |  | +0.0% |  | -6.4% | -12.1% |  |
| 128x128 | 4 | 64 | sparse_screen | forward | v6_refined_direct | ok | 7.080 | 7.080 | 7.079 | 0.000 | ok | 1.788e-07 |  | +13.8% |  | +6.5% |  |  |
| 128x128 | 4 | 64 | sparse_screen | forward | v6_refined_auto | ok | 6.925 | 6.925 | 6.924 | 0.000 | ok | 1.788e-07 |  | +11.3% |  | +4.1% | -2.2% |  |
| 128x128 | 4 | 64 | sparse_screen | forward_backward | v6_direct | ok | 9.090 | 9.090 | 6.229 | 2.860 | ok | 1.788e-07 | 4.657e-10 | +8.6% |  |  | -3.3% |  |
| 128x128 | 4 | 64 | sparse_screen | forward_backward | v6_auto | ok | 8.985 | 8.985 | 7.061 | 1.924 | ok | 1.788e-07 | 4.657e-10 | +7.3% |  | -1.2% | -4.4% |  |
| 128x128 | 4 | 64 | sparse_screen | forward_backward | v6_upgrade_direct | ok | 9.419 | 9.419 | 7.623 | 1.796 | ok | 1.788e-07 | 4.657e-10 | +12.5% |  | +3.6% | +0.2% |  |
| 128x128 | 4 | 64 | sparse_screen | forward_backward | v6_upgrade_auto | ok | 8.373 | 8.373 | 6.331 | 2.042 | ok | 1.788e-07 | 4.657e-10 | +0.0% |  | -7.9% | -10.9% |  |
| 128x128 | 4 | 64 | sparse_screen | forward_backward | v6_refined_direct | ok | 9.400 | 9.400 | 7.360 | 2.041 | ok | 1.788e-07 | 4.657e-10 | +12.3% |  | +3.4% |  |  |
| 128x128 | 4 | 64 | sparse_screen | forward_backward | v6_refined_auto | ok | 9.451 | 9.451 | 7.606 | 1.845 | ok | 1.788e-07 | 2.328e-10 | +12.9% |  | +4.0% | +0.5% |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward | v6_direct | ok | 6.262 | 6.262 | 6.262 | 0.000 | ok | 8.792e-05 |  | +9.3% |  |  | -1.2% |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward | v6_auto | ok | 6.491 | 6.491 | 6.491 | 0.000 | ok | 8.792e-05 |  | +13.2% |  | +3.7% | +2.4% |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 5.732 | 5.732 | 5.729 | 0.003 | ok | 8.792e-05 |  | +0.0% |  | -8.5% | -9.5% |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 8.565 | 8.565 | 8.565 | 0.000 | ok | 8.792e-05 |  | +49.4% |  | +36.8% | +35.2% |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward | v6_refined_direct | ok | 6.336 | 6.336 | 6.335 | 0.000 | ok | 8.792e-05 |  | +10.5% |  | +1.2% |  |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward | v6_refined_auto | ok | 8.495 | 8.495 | 8.495 | 0.000 | ok | 8.792e-05 |  | +48.2% |  | +35.7% | +34.1% |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward_backward | v6_direct | ok | 8.789 | 8.789 | 5.732 | 3.057 | ok | 8.792e-05 | 4.396e-06 | +0.0% |  |  | -8.1% |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward_backward | v6_auto | ok | 8.828 | 8.828 | 6.546 | 2.282 | ok | 8.792e-05 | 4.396e-06 | +0.4% |  | +0.4% | -7.7% |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 9.765 | 9.765 | 7.145 | 2.620 | ok | 8.792e-05 | 4.392e-06 | +11.1% |  | +11.1% | +2.1% |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 12.511 | 12.511 | 10.620 | 1.891 | ok | 8.792e-05 | 4.392e-06 | +42.3% |  | +42.3% | +30.8% |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward_backward | v6_refined_direct | ok | 9.563 | 9.563 | 6.780 | 2.783 | ok | 8.792e-05 | 4.392e-06 | +8.8% |  | +8.8% |  |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward_backward | v6_refined_auto | ok | 13.188 | 13.188 | 11.314 | 1.874 | ok | 8.792e-05 | 4.396e-06 | +50.1% |  | +50.1% | +37.9% |  |
| 128x128 | 4 | 64 | layered_depth | forward | v6_direct | ok | 5.740 | 5.740 | 5.740 | 0.000 | ok | 4.679e-05 |  | +0.6% |  |  | +0.2% |  |
| 128x128 | 4 | 64 | layered_depth | forward | v6_auto | ok | 6.485 | 6.485 | 6.485 | 0.000 | ok | 4.679e-05 |  | +13.6% |  | +13.0% | +13.2% |  |
| 128x128 | 4 | 64 | layered_depth | forward | v6_upgrade_direct | ok | 7.241 | 7.241 | 7.241 | 0.000 | ok | 4.679e-05 |  | +26.9% |  | +26.2% | +26.4% |  |
| 128x128 | 4 | 64 | layered_depth | forward | v6_upgrade_auto | ok | 7.256 | 7.256 | 7.256 | 0.000 | ok | 4.679e-05 |  | +27.2% |  | +26.4% | +26.6% |  |
| 128x128 | 4 | 64 | layered_depth | forward | v6_refined_direct | ok | 5.730 | 5.730 | 5.730 | 0.000 | ok | 4.679e-05 |  | +0.4% |  | -0.2% |  |  |
| 128x128 | 4 | 64 | layered_depth | forward | v6_refined_auto | ok | 5.706 | 5.706 | 5.706 | 0.000 | ok | 4.679e-05 |  | +0.0% |  | -0.6% | -0.4% |  |
| 128x128 | 4 | 64 | layered_depth | forward_backward | v6_direct | ok | 9.386 | 9.386 | 7.159 | 2.227 | ok | 4.679e-05 | 3.967e-07 | +3.0% |  |  | +1.4% |  |
| 128x128 | 4 | 64 | layered_depth | forward_backward | v6_auto | ok | 10.623 | 10.623 | 8.395 | 2.228 | ok | 4.679e-05 | 3.967e-07 | +16.6% |  | +13.2% | +14.8% |  |
| 128x128 | 4 | 64 | layered_depth | forward_backward | v6_upgrade_direct | ok | 10.059 | 10.059 | 6.968 | 3.091 | ok | 4.679e-05 | 3.958e-07 | +10.4% |  | +7.2% | +8.7% |  |
| 128x128 | 4 | 64 | layered_depth | forward_backward | v6_upgrade_auto | ok | 9.112 | 9.112 | 6.681 | 2.431 | ok | 4.679e-05 | 3.967e-07 | +0.0% |  | -2.9% | -1.5% |  |
| 128x128 | 4 | 64 | layered_depth | forward_backward | v6_refined_direct | ok | 9.254 | 9.254 | 6.985 | 2.268 | ok | 4.679e-05 | 3.958e-07 | +1.6% |  | -1.4% |  |  |
| 128x128 | 4 | 64 | layered_depth | forward_backward | v6_refined_auto | ok | 11.118 | 11.118 | 8.190 | 2.929 | ok | 4.679e-05 | 3.967e-07 | +22.0% |  | +18.5% | +20.2% |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward | v6_direct | ok | 7.296 | 7.296 | 7.296 | 0.000 | ok | 8.982e-05 |  | +18.6% |  |  | +3.0% |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward | v6_auto | ok | 8.558 | 8.558 | 8.558 | 0.000 | ok | 8.982e-05 |  | +39.1% |  | +17.3% | +20.8% |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward | v6_upgrade_direct | ok | 8.508 | 8.508 | 8.508 | 0.000 | ok | 8.982e-05 |  | +38.3% |  | +16.6% | +20.1% |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward | v6_upgrade_auto | ok | 7.697 | 7.697 | 7.696 | 0.000 | ok | 8.982e-05 |  | +25.1% |  | +5.5% | +8.7% |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward | v6_refined_direct | ok | 7.082 | 7.082 | 7.081 | 0.000 | ok | 8.982e-05 |  | +15.1% |  | -2.9% |  |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward | v6_refined_auto | ok | 6.151 | 6.151 | 6.151 | 0.000 | ok | 8.982e-05 |  | +0.0% |  | -15.7% | -13.1% |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward_backward | v6_direct | ok | 10.183 | 10.183 | 6.782 | 3.401 | ok | 8.982e-05 | 7.833e-03 | +0.0% |  |  | -13.1% |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward_backward | v6_auto | ok | 11.077 | 11.077 | 7.586 | 3.492 | ok | 8.982e-05 | 7.833e-03 | +8.8% |  | +8.8% | -5.4% |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward_backward | v6_upgrade_direct | ok | 11.161 | 11.161 | 7.418 | 3.742 | ok | 8.982e-05 | 7.833e-03 | +9.6% |  | +9.6% | -4.7% |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward_backward | v6_upgrade_auto | ok | 13.135 | 13.135 | 9.512 | 3.622 | ok | 8.982e-05 | 7.833e-03 | +29.0% |  | +29.0% | +12.1% |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward_backward | v6_refined_direct | ok | 11.714 | 11.714 | 8.422 | 3.292 | ok | 8.982e-05 | 7.833e-03 | +15.0% |  | +15.0% |  |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward_backward | v6_refined_auto | ok | 12.019 | 12.019 | 9.081 | 2.938 | ok | 8.982e-05 | 7.833e-03 | +18.0% |  | +18.0% | +2.6% |  |
