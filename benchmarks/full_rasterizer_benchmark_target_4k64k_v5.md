# Full Rasterizer Benchmark

Generated: 2026-04-22 12:28:38

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `120.0`
- accuracy checks: `False`
- accuracy max work items: `16000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v5_batched | 206.448 |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v5_batched | 200.971 |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v5_batched | 12.409 |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v5_batched | 89.723 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v5_batched | 15.632 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v5_batched | 73.156 |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v5_batched | 1090.540 |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v5_batched | 1140.667 |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v5_batched | 14.833 |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v5_batched | 57.834 |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v5_batched | 823.940 |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward_backward | v5_batched | 749.037 |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v5_batched | 44.504 |
| 4096x4096 | 4 | 65536 | layered_depth | forward_backward | v5_batched | 317.495 |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v5_batched | 35.903 |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward_backward | v5_batched | 255.847 |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v5_batched | 3566.565 |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward_backward | v5_batched | 4725.260 |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v5_batched | 34.467 |
| 4096x4096 | 4 | 65536 | sparse_screen | forward_backward | v5_batched | 214.883 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v5_batched | ok | 15.632 | 15.632 | 15.631 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v5_batched | ok | 73.156 | 73.156 | 14.078 | 59.078 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward | v5_batched | ok | 14.833 | 14.833 | 14.832 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v5_batched | ok | 57.834 | 57.834 | 9.408 | 48.426 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward | v5_batched | ok | 206.448 | 206.448 | 206.447 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v5_batched | ok | 200.971 | 200.971 | 134.712 | 66.259 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward | v5_batched | ok | 12.409 | 12.409 | 12.409 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v5_batched | ok | 89.723 | 89.723 | 15.933 | 73.790 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward | v5_batched | ok | 1090.540 | 1090.540 | 1090.539 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v5_batched | ok | 1140.667 | 1140.667 | 602.445 | 538.222 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward | v5_batched | ok | 35.903 | 35.903 | 35.902 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | microbench_uniform_random | forward_backward | v5_batched | ok | 255.847 | 255.847 | 38.801 | 217.045 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward | v5_batched | ok | 34.467 | 34.467 | 34.466 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | sparse_screen | forward_backward | v5_batched | ok | 214.883 | 214.883 | 27.094 | 187.789 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward | v5_batched | ok | 823.940 | 823.940 | 823.939 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | clustered_hot_tiles | forward_backward | v5_batched | ok | 749.037 | 749.037 | 502.349 | 246.688 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward | v5_batched | ok | 44.504 | 44.504 | 44.503 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | layered_depth | forward_backward | v5_batched | ok | 317.495 | 317.495 | 45.813 | 271.682 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward | v5_batched | ok | 3566.565 | 3566.565 | 3566.565 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 4 | 65536 | overflow_adversarial | forward_backward | v5_batched | ok | 4725.260 | 4725.260 | 2585.492 | 2139.768 |  |  |  | +0.0% |  |  |  |  |
