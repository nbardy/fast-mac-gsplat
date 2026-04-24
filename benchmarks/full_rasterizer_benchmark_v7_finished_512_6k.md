# Full Rasterizer Benchmark

Generated: 2026-04-22 14:13:19

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `240.0`
- accuracy checks: `True`
- accuracy max work items: `7000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v7_hardware | 11.718 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v7_hardware | 40.668 |
| 512x512 | 1 | 6000 | layered_depth | forward | v7_finished_hardware | 10.186 |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v7_finished_hardware | 41.687 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v7_hardware | 9.257 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v7_hardware | 39.857 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v7_finished_hardware | 23.841 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v7_hardware | 101.090 |
| 512x512 | 1 | 6000 | sparse_screen | forward | v7_finished_hardware | 7.343 |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v7_hardware | 37.217 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v7_hardware | 20.277 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v7_hardware | 143.562 |
| 512x512 | 4 | 6000 | layered_depth | forward | v7_hardware | 15.468 |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v7_finished_hardware | 134.874 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v7_hardware | 30.351 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v7_hardware | 219.626 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v7_finished_hardware | 79.498 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v7_finished_hardware | 369.843 |
| 512x512 | 4 | 6000 | sparse_screen | forward | v7_finished_hardware | 11.117 |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v7_finished_hardware | 128.022 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v7_hardware | ok | 9.257 | 9.257 | 9.257 | 0.000 | skipped |  |  | +0.0% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v7_finished_hardware | ok | 9.858 | 9.858 | 9.858 | 0.000 | skipped |  |  | +6.5% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v7_hardware | ok | 39.857 | 39.857 | 6.114 | 33.743 | skipped |  |  | +0.0% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v7_finished_hardware | ok | 40.364 | 40.364 | 7.255 | 33.109 | skipped |  |  | +1.3% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | sparse_screen | forward | v7_hardware | ok | 8.477 | 8.477 | 8.476 | 0.001 | skipped |  |  | +15.4% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | sparse_screen | forward | v7_finished_hardware | ok | 7.343 | 7.343 | 7.342 | 0.000 | skipped |  |  | +0.0% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v7_hardware | ok | 37.217 | 37.217 | 4.810 | 32.407 | skipped |  |  | +0.0% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | sparse_screen | forward_backward | v7_finished_hardware | ok | 37.432 | 37.432 | 4.756 | 32.676 | skipped |  |  | +0.6% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v7_hardware | ok | 11.718 | 11.718 | 11.717 | 0.001 | skipped |  |  | +0.0% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v7_finished_hardware | ok | 14.825 | 14.825 | 14.824 | 0.000 | skipped |  |  | +26.5% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v7_hardware | ok | 40.668 | 40.668 | 6.788 | 33.881 | skipped |  |  | +0.0% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v7_finished_hardware | ok | 51.370 | 51.370 | 8.773 | 42.597 | skipped |  |  | +26.3% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | layered_depth | forward | v7_hardware | ok | 10.911 | 10.911 | 10.911 | 0.001 | skipped |  |  | +7.1% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | layered_depth | forward | v7_finished_hardware | ok | 10.186 | 10.186 | 10.186 | 0.000 | skipped |  |  | +0.0% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v7_hardware | ok | 52.821 | 52.821 | 6.038 | 46.783 | skipped |  |  | +26.7% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | layered_depth | forward_backward | v7_finished_hardware | ok | 41.687 | 41.687 | 6.534 | 35.153 | skipped |  |  | +0.0% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v7_hardware | ok | 33.110 | 33.110 | 33.109 | 0.000 | skipped |  |  | +38.9% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward | v7_finished_hardware | ok | 23.841 | 23.841 | 23.841 | 0.000 | skipped |  |  | +0.0% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v7_hardware | ok | 101.090 | 101.090 | 26.276 | 74.814 | skipped |  |  | +0.0% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 1 | 6000 | overflow_adversarial | forward_backward | v7_finished_hardware | ok | 155.760 | 155.760 | 44.776 | 110.984 | skipped |  |  | +54.1% |  |  |  | work_items 1572864000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v7_hardware | ok | 30.351 | 30.351 | 30.350 | 0.001 | skipped |  |  | +0.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward | v7_finished_hardware | ok | 32.297 | 32.297 | 32.296 | 0.001 | skipped |  |  | +6.4% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v7_hardware | ok | 219.626 | 219.626 | 37.416 | 182.210 | skipped |  |  | +0.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v7_finished_hardware | ok | 226.186 | 226.186 | 38.156 | 188.031 | skipped |  |  | +3.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | sparse_screen | forward | v7_hardware | ok | 22.357 | 22.357 | 22.356 | 0.001 | skipped |  |  | +101.1% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | sparse_screen | forward | v7_finished_hardware | ok | 11.117 | 11.117 | 11.116 | 0.000 | skipped |  |  | +0.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v7_hardware | ok | 129.311 | 129.311 | 10.120 | 119.192 | skipped |  |  | +1.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | sparse_screen | forward_backward | v7_finished_hardware | ok | 128.022 | 128.022 | 9.529 | 118.493 | skipped |  |  | +0.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v7_hardware | ok | 20.277 | 20.277 | 20.276 | 0.001 | skipped |  |  | +0.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward | v7_finished_hardware | ok | 21.024 | 21.024 | 21.024 | 0.000 | skipped |  |  | +3.7% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v7_hardware | ok | 143.562 | 143.562 | 16.632 | 126.930 | skipped |  |  | +0.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v7_finished_hardware | ok | 146.479 | 146.479 | 17.634 | 128.845 | skipped |  |  | +2.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | layered_depth | forward | v7_hardware | ok | 15.468 | 15.468 | 15.468 | 0.001 | skipped |  |  | +0.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | layered_depth | forward | v7_finished_hardware | ok | 16.672 | 16.672 | 16.671 | 0.001 | skipped |  |  | +7.8% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v7_hardware | ok | 135.338 | 135.338 | 11.390 | 123.948 | skipped |  |  | +0.3% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | layered_depth | forward_backward | v7_finished_hardware | ok | 134.874 | 134.874 | 11.225 | 123.649 | skipped |  |  | +0.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v7_hardware | ok | 87.101 | 87.101 | 87.100 | 0.001 | skipped |  |  | +9.6% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward | v7_finished_hardware | ok | 79.498 | 79.498 | 79.498 | 0.000 | skipped |  |  | +0.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v7_hardware | ok | 378.549 | 378.549 | 93.285 | 285.264 | skipped |  |  | +2.4% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
| 512x512 | 4 | 6000 | overflow_adversarial | forward_backward | v7_finished_hardware | ok | 369.843 | 369.843 | 86.468 | 283.374 | skipped |  |  | +0.0% |  |  |  | work_items 6291456000 exceeds accuracy limit 7000000 |
