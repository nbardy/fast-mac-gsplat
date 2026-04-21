# Full Rasterizer Benchmark

Generated: 2026-04-21 21:03:57

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` is time delta, so negative means faster than v6 direct.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `240.0`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v7_hardware | 129.820 |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | 178.227 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_direct | 31.301 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_direct | 68.039 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Slower Than Best | % vs Torch | % vs v6 Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 31.301 | 31.301 | 31.301 | 0.000 | +0.0% |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v7_hardware | ok | 149.561 | 149.561 | 149.560 | 0.001 | +377.8% |  | +377.8% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 68.039 | 68.039 | 14.796 | 53.243 | +0.0% |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v7_hardware | ok | 21574.243 | 21574.243 | 180.205 | 21394.038 | +31608.8% |  | +31608.8% |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v6_direct | ok | 253.653 | 253.653 | 253.651 | 0.002 | +95.4% |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v7_hardware | ok | 129.820 | 129.820 | 129.820 | 0.000 | +0.0% |  | -48.8% |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v6_direct | ok | 178.227 | 178.227 | 129.322 | 48.905 | +0.0% |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v7_hardware | ok | 20054.494 | 20054.494 | 147.290 | 19907.204 | +11152.2% |  | +11152.2% |  |
