# Full Rasterizer Benchmark

Generated: 2026-04-22 19:44:43

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `1`
- seed: `0`
- timeout seconds per cell: `360.0`
- accuracy checks: `False`
- accuracy max work items: `16000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v7_frontk_k2 | 23069.772 |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v7_frontk_k2 | 24497.635 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v7_frontk_k2 | 35852.168 |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v7_frontk_k2 | 27008.610 |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v7_frontk_k2 | 21808.049 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v7_frontk_k2 | ok | 35852.168 | 35852.168 | 17110.397 | 18741.771 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v7_frontk_k2 | ok | 21808.049 | 21808.049 | 20033.602 | 1774.447 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v7_frontk_k2 | ok | 23069.772 | 23069.772 | 22319.137 | 750.635 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v7_frontk_k2 | ok | 24497.635 | 24497.635 | 19704.240 | 4793.395 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v7_frontk_k2 | ok | 27008.610 | 27008.610 | 20388.068 | 6620.542 |  |  |  | +0.0% |  |  |  |  |
