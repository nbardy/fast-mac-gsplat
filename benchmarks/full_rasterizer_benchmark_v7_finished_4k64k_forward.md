# Full Rasterizer Benchmark

Generated: 2026-04-22 14:14:08

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
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v7_finished_hardware | 117.198 |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v7_finished_hardware | 117.702 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v7_finished_hardware | 143.662 |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v7_finished_hardware | 367.198 |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v7_hardware | 111.345 |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v7_finished_hardware | 345.868 |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v7_finished_hardware | 366.397 |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v7_finished_hardware | 455.647 |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v7_hardware | 1605.064 |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v7_hardware | 356.121 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v7_hardware | ok | 210.819 | 210.819 | 210.818 | 0.001 |  |  |  | +46.7% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v7_finished_hardware | ok | 143.662 | 143.662 | 143.661 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v7_hardware | ok | 111.345 | 111.345 | 111.345 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v7_finished_hardware | ok | 114.497 | 114.497 | 114.496 | 0.001 |  |  |  | +2.8% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v7_hardware | ok | 117.265 | 117.265 | 117.264 | 0.001 |  |  |  | +0.1% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v7_finished_hardware | ok | 117.198 | 117.198 | 117.198 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v7_hardware | ok | 133.826 | 133.826 | 133.825 | 0.001 |  |  |  | +13.7% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v7_finished_hardware | ok | 117.702 | 117.702 | 117.701 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v7_hardware | ok | 389.739 | 389.739 | 389.737 | 0.001 |  |  |  | +6.1% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v7_finished_hardware | ok | 367.198 | 367.198 | 367.196 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v7_hardware | ok | 509.926 | 509.926 | 509.925 | 0.001 |  |  |  | +11.9% |  |  |  |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v7_finished_hardware | ok | 455.647 | 455.647 | 455.645 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v7_hardware | ok | 356.121 | 356.121 | 356.118 | 0.002 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v7_finished_hardware | ok | 365.236 | 365.236 | 365.235 | 0.001 |  |  |  | +2.6% |  |  |  |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v7_hardware | ok | 348.813 | 348.813 | 348.811 | 0.001 |  |  |  | +0.9% |  |  |  |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v7_finished_hardware | ok | 345.868 | 345.868 | 345.867 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v7_hardware | ok | 367.598 | 367.598 | 367.597 | 0.001 |  |  |  | +0.3% |  |  |  |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v7_finished_hardware | ok | 366.397 | 366.397 | 366.396 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v7_hardware | ok | 1605.064 | 1605.064 | 1605.063 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v7_finished_hardware | ok | 1643.673 | 1643.673 | 1643.671 | 0.002 |  |  |  | +2.4% |  |  |  |  |
