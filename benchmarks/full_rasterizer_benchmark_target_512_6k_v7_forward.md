# Full Rasterizer Benchmark

Generated: 2026-04-22 12:26:05

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
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v7_hardware | 10.593 |
| 512x512 | 1 | 6000 | layered_depth | forward | v7_hardware | 8.191 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v7_hardware | 7.677 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v7_hardware | 24.690 |
| 512x512 | 1 | 6000 | sparse_screen | forward | v7_hardware | 8.030 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v7_hardware | 21.338 |
| 512x512 | 4 | 6000 | layered_depth | forward | v7_hardware | 14.851 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v7_hardware | 15.051 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v7_hardware | 82.020 |
| 512x512 | 4 | 6000 | sparse_screen | forward | v7_hardware | 16.227 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v7_hardware | ok | 7.677 | 7.677 | 7.676 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | sparse_screen | forward | v7_hardware | ok | 8.030 | 8.030 | 8.030 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v7_hardware | ok | 10.593 | 10.593 | 10.593 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | layered_depth | forward | v7_hardware | ok | 8.191 | 8.191 | 8.190 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v7_hardware | ok | 24.690 | 24.690 | 24.690 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v7_hardware | ok | 15.051 | 15.051 | 15.050 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | sparse_screen | forward | v7_hardware | ok | 16.227 | 16.227 | 16.226 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v7_hardware | ok | 21.338 | 21.338 | 21.337 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | layered_depth | forward | v7_hardware | ok | 14.851 | 14.851 | 14.850 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v7_hardware | ok | 82.020 | 82.020 | 82.020 | 0.000 |  |  |  | +0.0% |  |  |  |  |
