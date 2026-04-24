# Full Rasterizer Benchmark

Generated: 2026-04-23 11:55:14

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
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v73_hybrid_k2 | 91.297 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v5_batched | 23.135 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v5_batched | ok | 23.135 | 23.135 | 8.692 | 14.443 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v72_tiled_k2 | ok | 48.242 | 48.242 | 13.848 | 34.394 |  |  |  | +108.5% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v73_hybrid_k2 | ok | 34.381 | 34.381 | 11.019 | 23.363 |  |  |  | +48.6% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v73_hybrid_k2_hwtrain | ok | 48.579 | 48.579 | 14.573 | 34.006 |  |  |  | +110.0% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v5_batched | ok | 141.302 | 141.302 | 101.612 | 39.690 |  |  |  | +54.8% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k2 | ok | 95.728 | 95.728 | 20.882 | 74.846 |  |  |  | +4.9% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v73_hybrid_k2 | ok | 91.297 | 91.297 | 62.807 | 28.490 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v73_hybrid_k2_hwtrain | ok | 95.394 | 95.394 | 20.949 | 74.445 |  |  |  | +4.5% |  |  |  |  |
