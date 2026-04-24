# Full Rasterizer Benchmark

Generated: 2026-04-23 11:55:46

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `1`
- seed: `0`
- timeout seconds per cell: `300.0`
- accuracy checks: `False`
- accuracy max work items: `16000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v5_batched | 73.681 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v5_batched | ok | 73.681 | 73.681 | 14.456 | 59.225 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v72_tiled_k2 | ok | 438.782 | 438.782 | 185.998 | 252.784 |  |  |  | +495.5% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v73_hybrid_k2 | ok | 76.464 | 76.464 | 17.605 | 58.859 |  |  |  | +3.8% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v73_hybrid_k2_hwtrain | ok | 436.552 | 436.552 | 170.097 | 266.455 |  |  |  | +492.5% |  |  |  |  |
