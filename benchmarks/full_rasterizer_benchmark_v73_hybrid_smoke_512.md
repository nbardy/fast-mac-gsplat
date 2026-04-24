# Full Rasterizer Benchmark

Generated: 2026-04-23 10:56:50

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `180.0`
- accuracy checks: `False`
- accuracy max work items: `16000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k2 | 92.820 |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v73_hybrid_k2 | 49.134 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v72_tiled_k2 | ok | 52.203 | 52.203 | 15.936 | 36.267 |  |  |  | +6.2% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v73_hybrid_k2 | ok | 49.134 | 49.134 | 14.395 | 34.739 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | microbench_uniform_random | forward_backward | v73_hybrid_k2_serial | ok | 91.018 | 91.018 | 29.858 | 61.160 |  |  |  | +85.2% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v72_tiled_k2 | ok | 92.820 | 92.820 | 20.784 | 72.036 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v73_hybrid_k2 | ok | 94.950 | 94.950 | 20.947 | 74.003 |  |  |  | +2.3% |  |  |  |  |
| 512x512 | 4 | 6000 | clustered_hot_tiles | forward_backward | v73_hybrid_k2_serial | ok | 130.854 | 130.854 | 40.807 | 90.046 |  |  |  | +41.0% |  |  |  |  |
