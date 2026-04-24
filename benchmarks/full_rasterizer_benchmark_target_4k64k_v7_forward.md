# Full Rasterizer Benchmark

Generated: 2026-04-22 12:26:34

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
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v7_hardware | 138.797 |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v7_hardware | 130.956 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v7_hardware | 143.574 |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v7_hardware | 492.732 |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v7_hardware | 115.937 |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v7_hardware | 404.342 |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v7_hardware | 437.674 |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v7_hardware | 536.821 |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v7_hardware | 2029.351 |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v7_hardware | 355.607 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v7_hardware | ok | 143.574 | 143.574 | 143.573 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v7_hardware | ok | 115.937 | 115.937 | 115.936 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v7_hardware | ok | 138.797 | 138.797 | 138.796 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v7_hardware | ok | 130.956 | 130.956 | 130.955 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v7_hardware | ok | 492.732 | 492.732 | 492.730 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v7_hardware | ok | 536.821 | 536.821 | 536.820 | 0.002 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v7_hardware | ok | 355.607 | 355.607 | 355.605 | 0.002 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v7_hardware | ok | 404.342 | 404.342 | 404.340 | 0.002 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v7_hardware | ok | 437.674 | 437.674 | 437.673 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v7_hardware | ok | 2029.351 | 2029.351 | 2029.350 | 0.001 |  |  |  | +0.0% |  |  |  |  |
