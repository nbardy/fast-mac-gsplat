# Full Rasterizer Benchmark

Generated: 2026-04-24 14:27:10

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `3`
- seed: `0`
- timeout seconds per cell: `120.0`
- accuracy checks: `True`
- accuracy max work items: `20000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v8_direct | 3.633 |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v6_direct | 8.200 |
| 16x16 | 1 | 4 | microbench_uniform_random | forward | torch_direct | 2.090 |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v6_direct | 8.067 |
| 16x16 | 1 | 64 | clustered_hot_tiles | forward | v8_direct | 2.591 |
| 16x16 | 1 | 64 | clustered_hot_tiles | forward_backward | v8_direct | 5.069 |
| 16x16 | 1 | 64 | microbench_uniform_random | forward | v8_direct | 3.886 |
| 16x16 | 1 | 64 | microbench_uniform_random | forward_backward | v8_direct | 4.544 |
| 16x16 | 2 | 4 | clustered_hot_tiles | forward | v8_direct | 2.741 |
| 16x16 | 2 | 4 | clustered_hot_tiles | forward_backward | v8_direct | 4.029 |
| 16x16 | 2 | 4 | microbench_uniform_random | forward | v8_direct | 2.800 |
| 16x16 | 2 | 4 | microbench_uniform_random | forward_backward | v8_direct | 4.063 |
| 16x16 | 2 | 64 | clustered_hot_tiles | forward | v8_direct | 2.941 |
| 16x16 | 2 | 64 | clustered_hot_tiles | forward_backward | v8_direct | 5.243 |
| 16x16 | 2 | 64 | microbench_uniform_random | forward | v8_direct | 2.856 |
| 16x16 | 2 | 64 | microbench_uniform_random | forward_backward | v8_direct | 4.706 |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | torch_direct | 2.562 |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_direct | 3.864 |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | torch_direct | 2.318 |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | 5.231 |
| 32x32 | 1 | 64 | clustered_hot_tiles | forward | v8_direct | 3.315 |
| 32x32 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_direct | 6.348 |
| 32x32 | 1 | 64 | microbench_uniform_random | forward | v8_direct | 2.807 |
| 32x32 | 1 | 64 | microbench_uniform_random | forward_backward | v8_direct | 4.903 |
| 32x32 | 2 | 4 | clustered_hot_tiles | forward | v8_direct | 2.460 |
| 32x32 | 2 | 4 | clustered_hot_tiles | forward_backward | v8_direct | 4.851 |
| 32x32 | 2 | 4 | microbench_uniform_random | forward | v8_direct | 2.774 |
| 32x32 | 2 | 4 | microbench_uniform_random | forward_backward | v8_direct | 3.897 |
| 32x32 | 2 | 64 | clustered_hot_tiles | forward | v8_direct | 3.459 |
| 32x32 | 2 | 64 | clustered_hot_tiles | forward_backward | v8_direct | 4.095 |
| 32x32 | 2 | 64 | microbench_uniform_random | forward | v8_direct | 2.704 |
| 32x32 | 2 | 64 | microbench_uniform_random | forward_backward | v8_direct | 4.342 |
| 64x64 | 1 | 4 | clustered_hot_tiles | forward | v8_direct | 4.050 |
| 64x64 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_direct | 5.108 |
| 64x64 | 1 | 4 | microbench_uniform_random | forward | v8_direct | 2.493 |
| 64x64 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | 3.724 |
| 64x64 | 1 | 64 | clustered_hot_tiles | forward | v8_direct | 2.608 |
| 64x64 | 1 | 64 | clustered_hot_tiles | forward_backward | v8_direct | 5.532 |
| 64x64 | 1 | 64 | microbench_uniform_random | forward | v8_direct | 4.400 |
| 64x64 | 1 | 64 | microbench_uniform_random | forward_backward | v8_direct | 4.091 |
| 64x64 | 2 | 4 | clustered_hot_tiles | forward | v8_direct | 3.493 |
| 64x64 | 2 | 4 | clustered_hot_tiles | forward_backward | v8_direct | 4.958 |
| 64x64 | 2 | 4 | microbench_uniform_random | forward | v8_direct | 3.052 |
| 64x64 | 2 | 4 | microbench_uniform_random | forward_backward | v8_direct | 4.789 |
| 64x64 | 2 | 64 | clustered_hot_tiles | forward | v8_direct | 2.673 |
| 64x64 | 2 | 64 | clustered_hot_tiles | forward_backward | v8_direct | 5.765 |
| 64x64 | 2 | 64 | microbench_uniform_random | forward | v8_direct | 3.361 |
| 64x64 | 2 | 64 | microbench_uniform_random | forward_backward | v8_direct | 4.027 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 16x16 | 1 | 4 | microbench_uniform_random | forward | torch_direct | ok | 2.090 | 1.926 | 2.090 | 0.000 | reference | 0.000e+00 |  | +0.0% |  | -66.5% |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward | v6_direct | ok | 6.240 | 3.798 | 6.240 | 0.000 | ok | 5.960e-08 |  | +198.6% | -66.5% |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward | v8_direct | ok | 3.436 | 2.648 | 3.435 | 0.000 | ok | 5.960e-08 |  | +64.4% | -39.2% | -44.9% |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | torch_direct | ok | 10.696 | 11.760 | 2.690 | 8.007 | reference | 0.000e+00 | 0.000e+00 | +32.6% |  | +32.6% |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v6_direct | ok | 8.067 | 6.150 | 6.569 | 1.498 | ok | 5.960e-08 | 2.980e-08 | +0.0% | +32.6% |  |  |  |
| 16x16 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | ok | 8.158 | 8.869 | 5.855 | 2.303 | ok | 5.960e-08 | 2.980e-08 | +1.1% | +31.1% | +1.1% |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | torch_direct | ok | 4.102 | 2.139 | 4.101 | 0.001 | reference | 0.000e+00 |  | +12.9% |  | -67.9% |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v6_direct | ok | 12.764 | 10.614 | 12.764 | 0.000 | ok | 5.960e-08 |  | +251.3% | -67.9% |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward | v8_direct | ok | 3.633 | 2.885 | 3.633 | 0.000 | ok | 5.960e-08 |  | +0.0% | +12.9% | -71.5% |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | torch_direct | ok | 8.546 | 7.789 | 3.219 | 5.327 | reference | 0.000e+00 | 0.000e+00 | +4.2% |  | +4.2% |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v6_direct | ok | 8.200 | 8.739 | 6.905 | 1.295 | ok | 5.960e-08 | 5.960e-08 | +0.0% | +4.2% |  |  |  |
| 16x16 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_direct | ok | 8.937 | 8.035 | 6.794 | 2.143 | ok | 5.960e-08 | 5.960e-08 | +9.0% | -4.4% | +9.0% |  |  |
| 16x16 | 2 | 4 | microbench_uniform_random | forward | torch_direct | ok | 4.595 | 4.038 | 4.595 | 0.000 | reference | 0.000e+00 |  | +64.1% |  | -28.3% |  |  |
| 16x16 | 2 | 4 | microbench_uniform_random | forward | v6_direct | ok | 6.408 | 6.312 | 6.408 | 0.000 | ok | 5.960e-08 |  | +128.9% | -28.3% |  |  |  |
| 16x16 | 2 | 4 | microbench_uniform_random | forward | v8_direct | ok | 2.800 | 2.569 | 2.799 | 0.000 | ok | 5.960e-08 |  | +0.0% | +64.1% | -56.3% |  |  |
| 16x16 | 2 | 4 | microbench_uniform_random | forward_backward | torch_direct | ok | 14.892 | 15.764 | 5.623 | 9.269 | reference | 0.000e+00 | 0.000e+00 | +266.5% |  | +100.7% |  |  |
| 16x16 | 2 | 4 | microbench_uniform_random | forward_backward | v6_direct | ok | 7.419 | 5.773 | 6.071 | 1.347 | ok | 5.960e-08 | 2.980e-08 | +82.6% | +100.7% |  |  |  |
| 16x16 | 2 | 4 | microbench_uniform_random | forward_backward | v8_direct | ok | 4.063 | 3.737 | 2.908 | 1.156 | ok | 5.960e-08 | 2.980e-08 | +0.0% | +266.5% | -45.2% |  |  |
| 16x16 | 2 | 4 | clustered_hot_tiles | forward | torch_direct | ok | 4.168 | 3.799 | 4.168 | 0.000 | reference | 0.000e+00 |  | +52.1% |  | -14.7% |  |  |
| 16x16 | 2 | 4 | clustered_hot_tiles | forward | v6_direct | ok | 4.886 | 4.389 | 4.886 | 0.000 | ok | 5.960e-08 |  | +78.3% | -14.7% |  |  |  |
| 16x16 | 2 | 4 | clustered_hot_tiles | forward | v8_direct | ok | 2.741 | 2.679 | 2.741 | 0.000 | ok | 5.960e-08 |  | +0.0% | +52.1% | -43.9% |  |  |
| 16x16 | 2 | 4 | clustered_hot_tiles | forward_backward | torch_direct | ok | 10.674 | 12.022 | 3.984 | 6.690 | reference | 0.000e+00 | 0.000e+00 | +164.9% |  | +44.1% |  |  |
| 16x16 | 2 | 4 | clustered_hot_tiles | forward_backward | v6_direct | ok | 7.410 | 6.286 | 5.943 | 1.467 | ok | 5.960e-08 | 5.960e-08 | +83.9% | +44.1% |  |  |  |
| 16x16 | 2 | 4 | clustered_hot_tiles | forward_backward | v8_direct | ok | 4.029 | 4.112 | 3.183 | 0.847 | ok | 5.960e-08 | 5.960e-08 | +0.0% | +164.9% | -45.6% |  |  |
| 16x16 | 1 | 64 | microbench_uniform_random | forward | torch_direct | ok | 13.850 | 15.351 | 13.850 | 0.001 | reference | 0.000e+00 |  | +256.4% |  | +207.9% |  |  |
| 16x16 | 1 | 64 | microbench_uniform_random | forward | v6_direct | ok | 4.498 | 4.147 | 4.498 | 0.000 | ok | 7.775e-05 |  | +15.7% | +207.9% |  |  |  |
| 16x16 | 1 | 64 | microbench_uniform_random | forward | v8_direct | ok | 3.886 | 3.685 | 3.886 | 0.001 | ok | 7.775e-05 |  | +0.0% | +256.4% | -13.6% |  |  |
| 16x16 | 1 | 64 | microbench_uniform_random | forward_backward | torch_direct | ok | 37.086 | 36.713 | 7.919 | 29.168 | reference | 0.000e+00 | 0.000e+00 | +716.1% |  | +373.9% |  |  |
| 16x16 | 1 | 64 | microbench_uniform_random | forward_backward | v6_direct | ok | 7.826 | 7.532 | 6.358 | 1.468 | ok | 7.775e-05 | 1.957e-04 | +72.2% | +373.9% |  |  |  |
| 16x16 | 1 | 64 | microbench_uniform_random | forward_backward | v8_direct | ok | 4.544 | 4.266 | 3.356 | 1.189 | ok | 7.775e-05 | 1.957e-04 | +0.0% | +716.1% | -41.9% |  |  |
| 16x16 | 1 | 64 | clustered_hot_tiles | forward | torch_direct | ok | 10.554 | 10.406 | 10.553 | 0.000 | reference | 0.000e+00 |  | +307.3% |  | +78.0% |  |  |
| 16x16 | 1 | 64 | clustered_hot_tiles | forward | v6_direct | ok | 5.930 | 6.386 | 5.929 | 0.000 | ok | 8.100e-05 |  | +128.8% | +78.0% |  |  |  |
| 16x16 | 1 | 64 | clustered_hot_tiles | forward | v8_direct | ok | 2.591 | 2.479 | 2.591 | 0.000 | ok | 8.100e-05 |  | +0.0% | +307.3% | -56.3% |  |  |
| 16x16 | 1 | 64 | clustered_hot_tiles | forward_backward | torch_direct | ok | 36.680 | 37.714 | 7.990 | 28.691 | reference | 0.000e+00 | 0.000e+00 | +623.6% |  | +326.7% |  |  |
| 16x16 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_direct | ok | 8.596 | 9.194 | 6.810 | 1.786 | ok | 8.100e-05 | 6.082e-04 | +69.6% | +326.7% |  |  |  |
| 16x16 | 1 | 64 | clustered_hot_tiles | forward_backward | v8_direct | ok | 5.069 | 4.527 | 3.110 | 1.959 | ok | 8.100e-05 | 6.082e-04 | +0.0% | +623.6% | -41.0% |  |  |
| 16x16 | 2 | 64 | microbench_uniform_random | forward | torch_direct | ok | 12.704 | 12.763 | 12.703 | 0.000 | reference | 0.000e+00 |  | +344.9% |  | +134.3% |  |  |
| 16x16 | 2 | 64 | microbench_uniform_random | forward | v6_direct | ok | 5.421 | 5.269 | 5.421 | 0.000 | ok | 7.775e-05 |  | +89.8% | +134.3% |  |  |  |
| 16x16 | 2 | 64 | microbench_uniform_random | forward | v8_direct | ok | 2.856 | 2.704 | 2.855 | 0.000 | ok | 7.775e-05 |  | +0.0% | +344.9% | -47.3% |  |  |
| 16x16 | 2 | 64 | microbench_uniform_random | forward_backward | torch_direct | ok | 69.238 | 69.948 | 14.378 | 54.860 | reference | 0.000e+00 | 0.000e+00 | +1371.1% |  | +893.5% |  |  |
| 16x16 | 2 | 64 | microbench_uniform_random | forward_backward | v6_direct | ok | 6.969 | 6.507 | 5.371 | 1.598 | ok | 7.775e-05 | 2.601e-04 | +48.1% | +893.5% |  |  |  |
| 16x16 | 2 | 64 | microbench_uniform_random | forward_backward | v8_direct | ok | 4.706 | 4.589 | 3.581 | 1.126 | ok | 7.775e-05 | 2.601e-04 | +0.0% | +1371.1% | -32.5% |  |  |
| 16x16 | 2 | 64 | clustered_hot_tiles | forward | torch_direct | ok | 15.068 | 12.605 | 15.068 | 0.000 | reference | 0.000e+00 |  | +412.4% |  | +195.6% |  |  |
| 16x16 | 2 | 64 | clustered_hot_tiles | forward | v6_direct | ok | 5.097 | 4.304 | 5.097 | 0.000 | ok | 8.100e-05 |  | +73.3% | +195.6% |  |  |  |
| 16x16 | 2 | 64 | clustered_hot_tiles | forward | v8_direct | ok | 2.941 | 2.633 | 2.941 | 0.000 | ok | 8.100e-05 |  | +0.0% | +412.4% | -42.3% |  |  |
| 16x16 | 2 | 64 | clustered_hot_tiles | forward_backward | torch_direct | ok | 67.176 | 67.947 | 14.562 | 52.614 | reference | 0.000e+00 | 0.000e+00 | +1181.3% |  | +819.4% |  |  |
| 16x16 | 2 | 64 | clustered_hot_tiles | forward_backward | v6_direct | ok | 7.307 | 5.756 | 5.753 | 1.554 | ok | 8.100e-05 | 3.041e-04 | +39.4% | +819.4% |  |  |  |
| 16x16 | 2 | 64 | clustered_hot_tiles | forward_backward | v8_direct | ok | 5.243 | 5.292 | 3.614 | 1.629 | ok | 8.100e-05 | 3.041e-04 | +0.0% | +1181.3% | -28.2% |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | torch_direct | ok | 2.318 | 2.282 | 2.318 | 0.000 | reference | 0.000e+00 |  | +0.0% |  | -60.4% |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | v6_direct | ok | 5.860 | 5.492 | 5.859 | 0.000 | ok | 4.470e-08 |  | +152.8% | -60.4% |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward | v8_direct | ok | 3.400 | 2.786 | 3.400 | 0.000 | ok | 4.470e-08 |  | +46.7% | -31.8% | -42.0% |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | torch_direct | ok | 8.692 | 7.958 | 2.566 | 6.126 | reference | 0.000e+00 | 0.000e+00 | +66.2% |  | +23.9% |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v6_direct | ok | 7.015 | 6.701 | 5.393 | 1.621 | ok | 4.470e-08 | 3.725e-09 | +34.1% | +23.9% |  |  |  |
| 32x32 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | ok | 5.231 | 4.504 | 4.106 | 1.125 | ok | 4.470e-08 | 3.725e-09 | +0.0% | +66.2% | -25.4% |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | torch_direct | ok | 2.562 | 2.518 | 2.562 | 0.001 | reference | 0.000e+00 |  | +0.0% |  | -52.5% |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | v6_direct | ok | 5.398 | 4.303 | 5.398 | 0.000 | ok | 5.960e-08 |  | +110.7% | -52.5% |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward | v8_direct | ok | 3.091 | 2.850 | 3.091 | 0.000 | ok | 5.960e-08 |  | +20.6% | -17.1% | -42.7% |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | torch_direct | ok | 6.835 | 6.152 | 2.977 | 3.858 | reference | 0.000e+00 | 0.000e+00 | +76.9% |  | -25.4% |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v6_direct | ok | 9.157 | 9.805 | 7.504 | 1.652 | ok | 5.960e-08 | 5.960e-08 | +137.0% | -25.4% |  |  |  |
| 32x32 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_direct | ok | 3.864 | 3.858 | 3.077 | 0.787 | ok | 5.960e-08 | 5.960e-08 | +0.0% | +76.9% | -57.8% |  |  |
| 32x32 | 2 | 4 | microbench_uniform_random | forward | torch_direct | ok | 4.447 | 4.025 | 4.447 | 0.000 | reference | 0.000e+00 |  | +60.3% |  | -25.4% |  |  |
| 32x32 | 2 | 4 | microbench_uniform_random | forward | v6_direct | ok | 5.960 | 4.799 | 5.960 | 0.000 | ok | 4.470e-08 |  | +114.8% | -25.4% |  |  |  |
| 32x32 | 2 | 4 | microbench_uniform_random | forward | v8_direct | ok | 2.774 | 3.017 | 2.774 | 0.000 | ok | 4.470e-08 |  | +0.0% | +60.3% | -53.5% |  |  |
| 32x32 | 2 | 4 | microbench_uniform_random | forward_backward | torch_direct | ok | 12.595 | 12.761 | 4.621 | 7.973 | reference | 0.000e+00 | 0.000e+00 | +223.2% |  | +47.5% |  |  |
| 32x32 | 2 | 4 | microbench_uniform_random | forward_backward | v6_direct | ok | 8.540 | 9.197 | 7.104 | 1.437 | ok | 4.470e-08 | 7.451e-09 | +119.2% | +47.5% |  |  |  |
| 32x32 | 2 | 4 | microbench_uniform_random | forward_backward | v8_direct | ok | 3.897 | 3.250 | 2.993 | 0.904 | ok | 4.470e-08 | 7.451e-09 | +0.0% | +223.2% | -54.4% |  |  |
| 32x32 | 2 | 4 | clustered_hot_tiles | forward | torch_direct | ok | 4.366 | 4.702 | 4.365 | 0.001 | reference | 0.000e+00 |  | +77.4% |  | -11.0% |  |  |
| 32x32 | 2 | 4 | clustered_hot_tiles | forward | v6_direct | ok | 4.904 | 4.567 | 4.904 | 0.000 | ok | 5.960e-08 |  | +99.3% | -11.0% |  |  |  |
| 32x32 | 2 | 4 | clustered_hot_tiles | forward | v8_direct | ok | 2.460 | 2.218 | 2.460 | 0.000 | ok | 5.960e-08 |  | +0.0% | +77.4% | -49.8% |  |  |
| 32x32 | 2 | 4 | clustered_hot_tiles | forward_backward | torch_direct | ok | 11.270 | 11.041 | 4.940 | 6.330 | reference | 0.000e+00 | 0.000e+00 | +132.3% |  | +45.0% |  |  |
| 32x32 | 2 | 4 | clustered_hot_tiles | forward_backward | v6_direct | ok | 7.772 | 7.864 | 5.775 | 1.997 | ok | 5.960e-08 | 1.192e-07 | +60.2% | +45.0% |  |  |  |
| 32x32 | 2 | 4 | clustered_hot_tiles | forward_backward | v8_direct | ok | 4.851 | 3.951 | 4.021 | 0.830 | ok | 5.960e-08 | 1.192e-07 | +0.0% | +132.3% | -37.6% |  |  |
| 32x32 | 1 | 64 | microbench_uniform_random | forward | torch_direct | ok | 11.409 | 11.300 | 11.409 | 0.000 | reference | 0.000e+00 |  | +306.5% |  | +13.2% |  |  |
| 32x32 | 1 | 64 | microbench_uniform_random | forward | v6_direct | ok | 10.079 | 7.081 | 10.079 | 0.001 | ok | 1.192e-07 |  | +259.1% | +13.2% |  |  |  |
| 32x32 | 1 | 64 | microbench_uniform_random | forward | v8_direct | ok | 2.807 | 2.411 | 2.806 | 0.000 | ok | 1.192e-07 |  | +0.0% | +306.5% | -72.2% |  |  |
| 32x32 | 1 | 64 | microbench_uniform_random | forward_backward | torch_direct | ok | 37.520 | 38.096 | 8.525 | 28.995 | reference | 0.000e+00 | 0.000e+00 | +665.2% |  | +499.0% |  |  |
| 32x32 | 1 | 64 | microbench_uniform_random | forward_backward | v6_direct | ok | 6.264 | 5.659 | 4.787 | 1.477 | ok | 1.192e-07 | 5.960e-08 | +27.8% | +499.0% |  |  |  |
| 32x32 | 1 | 64 | microbench_uniform_random | forward_backward | v8_direct | ok | 4.903 | 4.076 | 3.756 | 1.147 | ok | 1.192e-07 | 5.960e-08 | +0.0% | +665.2% | -21.7% |  |  |
| 32x32 | 1 | 64 | clustered_hot_tiles | forward | torch_direct | ok | 10.542 | 11.804 | 10.541 | 0.000 | reference | 0.000e+00 |  | +218.0% |  | +42.8% |  |  |
| 32x32 | 1 | 64 | clustered_hot_tiles | forward | v6_direct | ok | 7.383 | 7.418 | 7.383 | 0.000 | ok | 8.059e-05 |  | +122.7% | +42.8% |  |  |  |
| 32x32 | 1 | 64 | clustered_hot_tiles | forward | v8_direct | ok | 3.315 | 3.226 | 3.314 | 0.000 | ok | 8.059e-05 |  | +0.0% | +218.0% | -55.1% |  |  |
| 32x32 | 1 | 64 | clustered_hot_tiles | forward_backward | torch_direct | ok | 36.226 | 36.427 | 8.170 | 28.056 | reference | 0.000e+00 | 0.000e+00 | +470.7% |  | +470.7% |  |  |
| 32x32 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_direct | ok | 6.348 | 6.094 | 4.667 | 1.682 | ok | 8.059e-05 | 2.657e-04 | +0.0% | +470.7% |  |  |  |
| 32x32 | 1 | 64 | clustered_hot_tiles | forward_backward | v8_direct | ok | 7.481 | 8.772 | 4.766 | 2.715 | ok | 8.059e-05 | 2.657e-04 | +17.8% | +384.2% | +17.8% |  |  |
| 32x32 | 2 | 64 | microbench_uniform_random | forward | torch_direct | ok | 13.460 | 13.849 | 13.459 | 0.000 | reference | 0.000e+00 |  | +397.9% |  | +99.1% |  |  |
| 32x32 | 2 | 64 | microbench_uniform_random | forward | v6_direct | ok | 6.761 | 5.773 | 6.761 | 0.000 | ok | 1.788e-07 |  | +150.1% | +99.1% |  |  |  |
| 32x32 | 2 | 64 | microbench_uniform_random | forward | v8_direct | ok | 2.704 | 2.818 | 2.703 | 0.000 | ok | 1.788e-07 |  | +0.0% | +397.9% | -60.0% |  |  |
| 32x32 | 2 | 64 | microbench_uniform_random | forward_backward | torch_direct | ok | 69.118 | 68.677 | 16.035 | 53.083 | reference | 0.000e+00 | 0.000e+00 | +1491.7% |  | +860.2% |  |  |
| 32x32 | 2 | 64 | microbench_uniform_random | forward_backward | v6_direct | ok | 7.198 | 6.069 | 5.404 | 1.795 | ok | 1.788e-07 | 2.980e-08 | +65.8% | +860.2% |  |  |  |
| 32x32 | 2 | 64 | microbench_uniform_random | forward_backward | v8_direct | ok | 4.342 | 4.246 | 3.173 | 1.169 | ok | 1.788e-07 | 2.980e-08 | +0.0% | +1491.7% | -39.7% |  |  |
| 32x32 | 2 | 64 | clustered_hot_tiles | forward | torch_direct | ok | 13.418 | 12.290 | 13.418 | 0.000 | reference | 0.000e+00 |  | +287.9% |  | +75.5% |  |  |
| 32x32 | 2 | 64 | clustered_hot_tiles | forward | v6_direct | ok | 7.647 | 9.018 | 7.646 | 0.000 | ok | 8.059e-05 |  | +121.1% | +75.5% |  |  |  |
| 32x32 | 2 | 64 | clustered_hot_tiles | forward | v8_direct | ok | 3.459 | 2.684 | 3.459 | 0.000 | ok | 8.059e-05 |  | +0.0% | +287.9% | -54.8% |  |  |
| 32x32 | 2 | 64 | clustered_hot_tiles | forward_backward | torch_direct | ok | 67.475 | 67.269 | 14.532 | 52.943 | reference | 0.000e+00 | 0.000e+00 | +1547.7% |  | +713.3% |  |  |
| 32x32 | 2 | 64 | clustered_hot_tiles | forward_backward | v6_direct | ok | 8.296 | 9.150 | 6.657 | 1.639 | ok | 8.059e-05 | 1.329e-04 | +102.6% | +713.3% |  |  |  |
| 32x32 | 2 | 64 | clustered_hot_tiles | forward_backward | v8_direct | ok | 4.095 | 4.089 | 3.000 | 1.095 | ok | 8.059e-05 | 1.329e-04 | +0.0% | +1547.7% | -50.6% |  |  |
| 64x64 | 1 | 4 | microbench_uniform_random | forward | torch_direct | ok | 3.787 | 2.353 | 3.786 | 0.001 | reference | 0.000e+00 |  | +51.9% |  | -46.5% |  |  |
| 64x64 | 1 | 4 | microbench_uniform_random | forward | v6_direct | ok | 7.084 | 7.342 | 7.084 | 0.000 | ok | 5.960e-08 |  | +184.2% | -46.5% |  |  |  |
| 64x64 | 1 | 4 | microbench_uniform_random | forward | v8_direct | ok | 2.493 | 2.702 | 2.493 | 0.000 | ok | 5.960e-08 |  | +0.0% | +51.9% | -64.8% |  |  |
| 64x64 | 1 | 4 | microbench_uniform_random | forward_backward | torch_direct | ok | 7.549 | 7.477 | 3.306 | 4.243 | reference | 0.000e+00 | 0.000e+00 | +102.7% |  | -17.0% |  |  |
| 64x64 | 1 | 4 | microbench_uniform_random | forward_backward | v6_direct | ok | 9.095 | 9.532 | 7.852 | 1.243 | ok | 5.960e-08 | 9.313e-10 | +144.2% | -17.0% |  |  |  |
| 64x64 | 1 | 4 | microbench_uniform_random | forward_backward | v8_direct | ok | 3.724 | 3.664 | 2.739 | 0.985 | ok | 5.960e-08 | 9.313e-10 | +0.0% | +102.7% | -59.0% |  |  |
| 64x64 | 1 | 4 | clustered_hot_tiles | forward | torch_direct | ok | 4.150 | 2.861 | 4.149 | 0.001 | reference | 0.000e+00 |  | +2.5% |  | -9.4% |  |  |
| 64x64 | 1 | 4 | clustered_hot_tiles | forward | v6_direct | ok | 4.581 | 4.231 | 4.581 | 0.000 | ok | 5.960e-08 |  | +13.1% | -9.4% |  |  |  |
| 64x64 | 1 | 4 | clustered_hot_tiles | forward | v8_direct | ok | 4.050 | 3.163 | 4.049 | 0.000 | ok | 5.960e-08 |  | +0.0% | +2.5% | -11.6% |  |  |
| 64x64 | 1 | 4 | clustered_hot_tiles | forward_backward | torch_direct | ok | 8.174 | 7.686 | 3.772 | 4.402 | reference | 0.000e+00 | 0.000e+00 | +60.0% |  | -7.8% |  |  |
| 64x64 | 1 | 4 | clustered_hot_tiles | forward_backward | v6_direct | ok | 8.867 | 8.325 | 6.726 | 2.142 | ok | 5.960e-08 | 2.980e-08 | +73.6% | -7.8% |  |  |  |
| 64x64 | 1 | 4 | clustered_hot_tiles | forward_backward | v8_direct | ok | 5.108 | 4.886 | 4.255 | 0.853 | ok | 5.960e-08 | 3.725e-09 | +0.0% | +60.0% | -42.4% |  |  |
| 64x64 | 2 | 4 | microbench_uniform_random | forward | torch_direct | ok | 4.776 | 4.210 | 4.776 | 0.000 | reference | 0.000e+00 |  | +56.5% |  | -16.1% |  |  |
| 64x64 | 2 | 4 | microbench_uniform_random | forward | v6_direct | ok | 5.690 | 4.435 | 5.690 | 0.000 | ok | 5.960e-08 |  | +86.4% | -16.1% |  |  |  |
| 64x64 | 2 | 4 | microbench_uniform_random | forward | v8_direct | ok | 3.052 | 3.250 | 3.052 | 0.000 | ok | 5.960e-08 |  | +0.0% | +56.5% | -46.4% |  |  |
| 64x64 | 2 | 4 | microbench_uniform_random | forward_backward | torch_direct | ok | 12.758 | 12.331 | 6.116 | 6.642 | reference | 0.000e+00 | 0.000e+00 | +166.4% |  | +74.3% |  |  |
| 64x64 | 2 | 4 | microbench_uniform_random | forward_backward | v6_direct | ok | 7.320 | 6.435 | 5.810 | 1.511 | ok | 5.960e-08 | 1.863e-09 | +52.9% | +74.3% |  |  |  |
| 64x64 | 2 | 4 | microbench_uniform_random | forward_backward | v8_direct | ok | 4.789 | 4.812 | 3.035 | 1.754 | ok | 5.960e-08 | 1.863e-09 | +0.0% | +166.4% | -34.6% |  |  |
| 64x64 | 2 | 4 | clustered_hot_tiles | forward | torch_direct | ok | 5.371 | 4.421 | 5.371 | 0.000 | reference | 0.000e+00 |  | +53.8% |  | -25.3% |  |  |
| 64x64 | 2 | 4 | clustered_hot_tiles | forward | v6_direct | ok | 7.191 | 7.478 | 7.191 | 0.000 | ok | 5.960e-08 |  | +105.9% | -25.3% |  |  |  |
| 64x64 | 2 | 4 | clustered_hot_tiles | forward | v8_direct | ok | 3.493 | 3.823 | 3.493 | 0.000 | ok | 5.960e-08 |  | +0.0% | +53.8% | -51.4% |  |  |
| 64x64 | 2 | 4 | clustered_hot_tiles | forward_backward | torch_direct | ok | 12.790 | 14.782 | 4.169 | 8.621 | reference | 0.000e+00 | 0.000e+00 | +158.0% |  | +68.4% |  |  |
| 64x64 | 2 | 4 | clustered_hot_tiles | forward_backward | v6_direct | ok | 7.593 | 7.815 | 5.972 | 1.622 | ok | 5.960e-08 | 5.960e-08 | +53.2% | +68.4% |  |  |  |
| 64x64 | 2 | 4 | clustered_hot_tiles | forward_backward | v8_direct | ok | 4.958 | 4.034 | 3.986 | 0.972 | ok | 5.960e-08 | 2.980e-08 | +0.0% | +158.0% | -34.7% |  |  |
| 64x64 | 1 | 64 | microbench_uniform_random | forward | torch_direct | ok | 15.149 | 15.287 | 15.148 | 0.000 | reference | 0.000e+00 |  | +244.3% |  | +203.2% |  |  |
| 64x64 | 1 | 64 | microbench_uniform_random | forward | v6_direct | ok | 4.996 | 4.441 | 4.996 | 0.000 | ok | 1.788e-07 |  | +13.6% | +203.2% |  |  |  |
| 64x64 | 1 | 64 | microbench_uniform_random | forward | v8_direct | ok | 4.400 | 3.630 | 4.400 | 0.000 | ok | 1.788e-07 |  | +0.0% | +244.3% | -11.9% |  |  |
| 64x64 | 1 | 64 | microbench_uniform_random | forward_backward | torch_direct | ok | 38.718 | 38.610 | 8.946 | 29.772 | reference | 0.000e+00 | 0.000e+00 | +846.5% |  | +314.3% |  |  |
| 64x64 | 1 | 64 | microbench_uniform_random | forward_backward | v6_direct | ok | 9.346 | 10.627 | 7.557 | 1.789 | ok | 1.788e-07 | 1.490e-08 | +128.5% | +314.3% |  |  |  |
| 64x64 | 1 | 64 | microbench_uniform_random | forward_backward | v8_direct | ok | 4.091 | 4.035 | 2.888 | 1.203 | ok | 1.788e-07 | 1.490e-08 | +0.0% | +846.5% | -56.2% |  |  |
| 64x64 | 1 | 64 | clustered_hot_tiles | forward | torch_direct | ok | 10.635 | 11.231 | 10.634 | 0.000 | reference | 0.000e+00 |  | +307.8% |  | +72.5% |  |  |
| 64x64 | 1 | 64 | clustered_hot_tiles | forward | v6_direct | ok | 6.166 | 4.124 | 6.166 | 0.000 | ok | 7.558e-05 |  | +136.4% | +72.5% |  |  |  |
| 64x64 | 1 | 64 | clustered_hot_tiles | forward | v8_direct | ok | 2.608 | 2.551 | 2.608 | 0.000 | ok | 7.558e-05 |  | +0.0% | +307.8% | -57.7% |  |  |
| 64x64 | 1 | 64 | clustered_hot_tiles | forward_backward | torch_direct | ok | 37.015 | 36.923 | 7.892 | 29.123 | reference | 0.000e+00 | 0.000e+00 | +569.2% |  | +360.4% |  |  |
| 64x64 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_direct | ok | 8.039 | 6.963 | 5.598 | 2.441 | ok | 7.558e-05 | 6.849e-05 | +45.3% | +360.4% |  |  |  |
| 64x64 | 1 | 64 | clustered_hot_tiles | forward_backward | v8_direct | ok | 5.532 | 4.546 | 3.285 | 2.247 | ok | 7.558e-05 | 6.843e-05 | +0.0% | +569.2% | -31.2% |  |  |
| 64x64 | 2 | 64 | microbench_uniform_random | forward | torch_direct | ok | 12.625 | 12.515 | 12.624 | 0.000 | reference | 0.000e+00 |  | +275.7% |  | +110.6% |  |  |
| 64x64 | 2 | 64 | microbench_uniform_random | forward | v6_direct | ok | 5.995 | 6.224 | 5.995 | 0.000 | ok | 1.788e-07 |  | +78.4% | +110.6% |  |  |  |
| 64x64 | 2 | 64 | microbench_uniform_random | forward | v8_direct | ok | 3.361 | 3.140 | 3.361 | 0.000 | ok | 1.788e-07 |  | +0.0% | +275.7% | -43.9% |  |  |
| 64x64 | 2 | 64 | microbench_uniform_random | forward_backward | torch_direct | ok | 71.334 | 70.797 | 14.761 | 56.573 | reference | 0.000e+00 | 0.000e+00 | +1671.3% |  | +709.3% |  |  |
| 64x64 | 2 | 64 | microbench_uniform_random | forward_backward | v6_direct | ok | 8.815 | 9.799 | 7.143 | 1.671 | ok | 1.788e-07 | 7.451e-09 | +118.9% | +709.3% |  |  |  |
| 64x64 | 2 | 64 | microbench_uniform_random | forward_backward | v8_direct | ok | 4.027 | 3.999 | 2.891 | 1.136 | ok | 1.788e-07 | 7.451e-09 | +0.0% | +1671.3% | -54.3% |  |  |
| 64x64 | 2 | 64 | clustered_hot_tiles | forward | torch_direct | ok | 18.136 | 18.415 | 18.135 | 0.001 | reference | 0.000e+00 |  | +578.4% |  | +223.6% |  |  |
| 64x64 | 2 | 64 | clustered_hot_tiles | forward | v6_direct | ok | 5.604 | 5.789 | 5.604 | 0.000 | ok | 7.558e-05 |  | +109.6% | +223.6% |  |  |  |
| 64x64 | 2 | 64 | clustered_hot_tiles | forward | v8_direct | ok | 2.673 | 2.740 | 2.673 | 0.000 | ok | 7.558e-05 |  | +0.0% | +578.4% | -52.3% |  |  |
| 64x64 | 2 | 64 | clustered_hot_tiles | forward_backward | torch_direct | ok | 72.472 | 73.400 | 15.827 | 56.645 | reference | 0.000e+00 | 0.000e+00 | +1157.2% |  | +875.2% |  |  |
| 64x64 | 2 | 64 | clustered_hot_tiles | forward_backward | v6_direct | ok | 7.431 | 6.779 | 5.006 | 2.425 | ok | 7.558e-05 | 3.424e-05 | +28.9% | +875.2% |  |  |  |
| 64x64 | 2 | 64 | clustered_hot_tiles | forward_backward | v8_direct | ok | 5.765 | 5.352 | 4.278 | 1.487 | ok | 7.558e-05 | 3.424e-05 | +0.0% | +1157.2% | -22.4% |  |  |
