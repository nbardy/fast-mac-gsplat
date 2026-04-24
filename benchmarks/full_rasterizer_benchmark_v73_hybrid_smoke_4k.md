# Full Rasterizer Benchmark

Generated: 2026-04-23 10:57:06

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `1`
- seed: `0`
- timeout seconds per cell: `240.0`
- accuracy checks: `False`
- accuracy max work items: `16000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v5_batched | 63.273 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v5_batched | ok | 63.273 | 63.273 | 11.277 | 51.996 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v72_tiled_k2 | ok | 412.732 | 412.732 | 171.483 | 241.250 |  |  |  | +552.3% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v73_hybrid_k2 | ok | 451.550 | 451.550 | 208.310 | 243.240 |  |  |  | +613.6% |  |  |  |  |
