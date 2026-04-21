# Full Rasterizer Benchmark

Generated: 2026-04-21 20:24:35

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` is time delta, so negative means faster than v6 direct.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `90.0`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_direct | 4.231 |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v7_hardware | 9.323 |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_direct | 4.296 |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_direct | 6.133 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Slower Than Best | % vs Torch | % vs v6 Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_direct | ok | 4.296 | 4.296 | 4.296 | 0.000 | +0.0% |  |  |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 6.817 | 6.817 | 6.817 | 0.000 | +58.7% |  | +58.7% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v6_upgrade_auto | ok | 4.733 | 4.733 | 4.733 | 0.000 | +10.2% |  | +10.2% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v7_hardware | ok | 5.939 | 5.939 | 5.938 | 0.000 | +38.2% |  | +38.2% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_direct | ok | 6.133 | 6.133 | 4.656 | 1.478 | +0.0% |  |  |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 7.233 | 7.233 | 5.789 | 1.444 | +17.9% |  | +17.9% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v6_upgrade_auto | ok | 6.732 | 6.732 | 5.349 | 1.383 | +9.8% |  | +9.8% |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v7_hardware | ok | 8.014 | 8.014 | 4.028 | 3.986 | +30.7% |  | +30.7% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_direct | ok | 4.231 | 4.231 | 4.231 | 0.000 | +0.0% |  |  |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_upgrade_direct | ok | 8.129 | 8.129 | 8.128 | 0.000 | +92.1% |  | +92.1% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v6_upgrade_auto | ok | 5.624 | 5.624 | 5.624 | 0.000 | +32.9% |  | +32.9% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v7_hardware | ok | 6.911 | 6.911 | 6.906 | 0.004 | +63.3% |  | +63.3% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_direct | ok | 10.458 | 10.458 | 8.100 | 2.358 | +12.2% |  |  |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_upgrade_direct | ok | 10.582 | 10.582 | 8.527 | 2.055 | +13.5% |  | +1.2% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v6_upgrade_auto | ok | 15.095 | 15.095 | 12.680 | 2.414 | +61.9% |  | +44.3% |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v7_hardware | ok | 9.323 | 9.323 | 5.233 | 4.090 | +0.0% |  | -10.9% |  |
