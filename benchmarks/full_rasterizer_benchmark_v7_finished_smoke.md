# Full Rasterizer Benchmark

Generated: 2026-04-22 14:11:55

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `120.0`
- accuracy checks: `True`
- accuracy max work items: `5000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v7_finished_hardware | 4.519 |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v7_finished_hardware | 9.686 |
| 128x128 | 1 | 64 | layered_depth | forward | v7_hardware | 6.109 |
| 128x128 | 1 | 64 | layered_depth | forward_backward | v7_hardware | 9.731 |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v7_hardware | 5.018 |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v7_hardware | 11.672 |
| 128x128 | 1 | 64 | overflow_adversarial | forward | v7_finished_hardware | 4.262 |
| 128x128 | 1 | 64 | overflow_adversarial | forward_backward | v7_hardware | 9.298 |
| 128x128 | 1 | 64 | sparse_screen | forward | v7_finished_hardware | 6.003 |
| 128x128 | 1 | 64 | sparse_screen | forward_backward | v7_finished_hardware | 8.565 |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward | v7_finished_hardware | 5.787 |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward_backward | v7_hardware | 11.520 |
| 128x128 | 4 | 64 | layered_depth | forward | v7_finished_hardware | 5.820 |
| 128x128 | 4 | 64 | layered_depth | forward_backward | v7_finished_hardware | 11.643 |
| 128x128 | 4 | 64 | microbench_uniform_random | forward | v7_finished_hardware | 5.871 |
| 128x128 | 4 | 64 | microbench_uniform_random | forward_backward | v7_hardware | 11.053 |
| 128x128 | 4 | 64 | overflow_adversarial | forward | v7_finished_hardware | 7.575 |
| 128x128 | 4 | 64 | overflow_adversarial | forward_backward | v7_hardware | 14.486 |
| 128x128 | 4 | 64 | sparse_screen | forward | v7_hardware | 5.346 |
| 128x128 | 4 | 64 | sparse_screen | forward_backward | v7_hardware | 11.249 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v7_hardware | ok | 5.018 | 5.018 | 5.018 | 0.000 | ok | 1.788e-07 |  | +0.0% |  |  |  |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward | v7_finished_hardware | ok | 5.837 | 5.837 | 5.836 | 0.000 | ok | 1.788e-07 |  | +16.3% |  |  |  |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v7_hardware | ok | 11.672 | 11.672 | 6.163 | 5.509 | ok | 1.788e-07 | 7.451e-09 | +0.0% |  |  |  |  |
| 128x128 | 1 | 64 | microbench_uniform_random | forward_backward | v7_finished_hardware | ok | 12.558 | 12.558 | 5.497 | 7.061 | ok | 1.788e-07 | 1.032e-02 | +7.6% |  |  |  |  |
| 128x128 | 1 | 64 | sparse_screen | forward | v7_hardware | ok | 6.378 | 6.378 | 6.377 | 0.001 | ok | 2.384e-07 |  | +6.2% |  |  |  |  |
| 128x128 | 1 | 64 | sparse_screen | forward | v7_finished_hardware | ok | 6.003 | 6.003 | 6.003 | 0.000 | ok | 2.384e-07 |  | +0.0% |  |  |  |  |
| 128x128 | 1 | 64 | sparse_screen | forward_backward | v7_hardware | ok | 9.544 | 9.544 | 4.451 | 5.093 | ok | 2.384e-07 | 2.794e-09 | +11.4% |  |  |  |  |
| 128x128 | 1 | 64 | sparse_screen | forward_backward | v7_finished_hardware | ok | 8.565 | 8.565 | 3.496 | 5.069 | ok | 2.384e-07 | 6.173e-03 | +0.0% |  |  |  |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v7_hardware | ok | 5.912 | 5.912 | 5.912 | 0.000 | ok | 4.768e-07 |  | +30.8% |  |  |  |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward | v7_finished_hardware | ok | 4.519 | 4.519 | 4.518 | 0.000 | ok | 4.768e-07 |  | +0.0% |  |  |  |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v7_hardware | ok | 11.096 | 11.096 | 6.304 | 4.793 | ok | 4.768e-07 | 4.470e-08 | +14.6% |  |  |  |  |
| 128x128 | 1 | 64 | clustered_hot_tiles | forward_backward | v7_finished_hardware | ok | 9.686 | 9.686 | 5.613 | 4.073 | ok | 4.768e-07 | 2.479e-02 | +0.0% |  |  |  |  |
| 128x128 | 1 | 64 | layered_depth | forward | v7_hardware | ok | 6.109 | 6.109 | 6.108 | 0.000 | ok | 4.172e-07 |  | +0.0% |  |  |  |  |
| 128x128 | 1 | 64 | layered_depth | forward | v7_finished_hardware | ok | 6.406 | 6.406 | 6.406 | 0.000 | ok | 4.172e-07 |  | +4.9% |  |  |  |  |
| 128x128 | 1 | 64 | layered_depth | forward_backward | v7_hardware | ok | 9.731 | 9.731 | 5.000 | 4.731 | ok | 4.172e-07 | 1.192e-07 | +0.0% |  |  |  |  |
| 128x128 | 1 | 64 | layered_depth | forward_backward | v7_finished_hardware | ok | 11.057 | 11.057 | 5.759 | 5.297 | ok | 4.172e-07 | 3.955e-02 | +13.6% |  |  |  |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward | v7_hardware | ok | 5.238 | 5.238 | 5.238 | 0.000 | ok | 5.364e-07 |  | +22.9% |  |  |  |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward | v7_finished_hardware | ok | 4.262 | 4.262 | 4.262 | 0.000 | ok | 5.364e-07 |  | +0.0% |  |  |  |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward_backward | v7_hardware | ok | 9.298 | 9.298 | 4.611 | 4.687 | ok | 5.364e-07 | 7.629e-06 | +0.0% |  |  |  |  |
| 128x128 | 1 | 64 | overflow_adversarial | forward_backward | v7_finished_hardware | ok | 9.399 | 9.399 | 3.659 | 5.740 | ok | 5.364e-07 | 2.931e-01 | +1.1% |  |  |  |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward | v7_hardware | ok | 6.612 | 6.612 | 6.612 | 0.000 | ok | 2.941e-03 |  | +12.6% |  |  |  |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward | v7_finished_hardware | ok | 5.871 | 5.871 | 5.871 | 0.000 | ok | 2.941e-03 |  | +0.0% |  |  |  |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward_backward | v7_hardware | ok | 11.053 | 11.053 | 5.300 | 5.754 | ok | 2.941e-03 | 1.877e-07 | +0.0% |  |  |  |  |
| 128x128 | 4 | 64 | microbench_uniform_random | forward_backward | v7_finished_hardware | ok | 11.350 | 11.350 | 5.632 | 5.717 | ok | 2.941e-03 | 3.913e-03 | +2.7% |  |  |  |  |
| 128x128 | 4 | 64 | sparse_screen | forward | v7_hardware | ok | 5.346 | 5.346 | 5.346 | 0.000 | ok | 2.384e-07 |  | +0.0% |  |  |  |  |
| 128x128 | 4 | 64 | sparse_screen | forward | v7_finished_hardware | ok | 7.840 | 7.840 | 7.840 | 0.000 | ok | 2.384e-07 |  | +46.6% |  |  |  |  |
| 128x128 | 4 | 64 | sparse_screen | forward_backward | v7_hardware | ok | 11.249 | 11.249 | 6.327 | 4.922 | ok | 2.384e-07 | 1.397e-09 | +0.0% |  |  |  |  |
| 128x128 | 4 | 64 | sparse_screen | forward_backward | v7_finished_hardware | ok | 13.609 | 13.609 | 6.010 | 7.599 | ok | 2.384e-07 | 1.945e-03 | +21.0% |  |  |  |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward | v7_hardware | ok | 6.896 | 6.896 | 6.896 | 0.000 | ok | 1.133e-03 |  | +19.2% |  |  |  |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward | v7_finished_hardware | ok | 5.787 | 5.787 | 5.786 | 0.000 | ok | 1.133e-03 |  | +0.0% |  |  |  |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward_backward | v7_hardware | ok | 11.520 | 11.520 | 5.113 | 6.407 | ok | 1.133e-03 | 2.543e-07 | +0.0% |  |  |  |  |
| 128x128 | 4 | 64 | clustered_hot_tiles | forward_backward | v7_finished_hardware | ok | 12.196 | 12.196 | 6.180 | 6.015 | ok | 1.133e-03 | 9.186e-03 | +5.9% |  |  |  |  |
| 128x128 | 4 | 64 | layered_depth | forward | v7_hardware | ok | 5.866 | 5.866 | 5.865 | 0.000 | ok | 7.808e-04 |  | +0.8% |  |  |  |  |
| 128x128 | 4 | 64 | layered_depth | forward | v7_finished_hardware | ok | 5.820 | 5.820 | 5.819 | 0.001 | ok | 7.808e-04 |  | +0.0% |  |  |  |  |
| 128x128 | 4 | 64 | layered_depth | forward_backward | v7_hardware | ok | 12.227 | 12.227 | 4.894 | 7.333 | ok | 7.808e-04 | 2.035e-07 | +5.0% |  |  |  |  |
| 128x128 | 4 | 64 | layered_depth | forward_backward | v7_finished_hardware | ok | 11.643 | 11.643 | 5.375 | 6.268 | ok | 7.808e-04 | 1.108e-02 | +0.0% |  |  |  |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward | v7_hardware | ok | 7.912 | 7.912 | 7.912 | 0.000 | ok | 5.960e-07 |  | +4.4% |  |  |  |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward | v7_finished_hardware | ok | 7.575 | 7.575 | 7.575 | 0.000 | ok | 5.960e-07 |  | +0.0% |  |  |  |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward_backward | v7_hardware | ok | 14.486 | 14.486 | 7.311 | 7.175 | ok | 5.960e-07 | 9.537e-06 | +0.0% |  |  |  |  |
| 128x128 | 4 | 64 | overflow_adversarial | forward_backward | v7_finished_hardware | ok | 15.897 | 15.897 | 7.213 | 8.684 | ok | 5.960e-07 | 8.565e-02 | +9.7% |  |  |  |  |
