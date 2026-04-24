# Full Rasterizer Benchmark

Generated: 2026-04-22 19:39:57

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `0`
- iters: `1`
- seed: `0`
- timeout seconds per cell: `240.0`
- accuracy checks: `False`
- accuracy max work items: `16000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v5_batched | 1002.262 |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v5_batched | 309.373 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v5_batched | 1026.534 |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v5_batched | 4560.586 |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v5_batched | 232.940 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v5_batched | ok | 1026.534 | 1026.534 | 896.993 | 129.541 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v7_frontk_k2 | ok | 34644.980 | 34644.980 | 16859.023 | 17785.957 |  |  |  | +3274.9% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v5_batched | ok | 232.940 | 232.940 | 132.488 | 100.452 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v7_frontk_k2 | ok | 18664.232 | 18664.232 | 17096.678 | 1567.554 |  |  |  | +7912.5% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v5_batched | ok | 1002.262 | 1002.262 | 869.318 | 132.944 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v7_frontk_k2 | ok | 18255.114 | 18255.114 | 17489.539 | 765.575 |  |  |  | +1721.4% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v5_batched | ok | 309.373 | 309.373 | 156.759 | 152.615 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v7_frontk_k2 | ok | 19464.501 | 19464.501 | 15796.887 | 3667.615 |  |  |  | +6191.6% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v5_batched | ok | 4560.586 | 4560.586 | 3912.344 | 648.241 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v7_frontk_k2 | ok | 24528.832 | 24528.832 | 17880.515 | 6648.318 |  |  |  | +437.8% |  |  |  |  |
