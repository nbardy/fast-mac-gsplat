# Full Rasterizer Benchmark

Generated: 2026-04-24 14:28:48

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `2`
- seed: `0`
- timeout seconds per cell: `300.0`
- accuracy checks: `False`
- accuracy max work items: `16000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v8_direct | 12.142 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v8_direct | 67.579 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 12.432 | 12.432 | 12.431 | 0.000 |  |  |  | +2.4% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v8_direct | ok | 12.142 | 12.142 | 12.142 | 0.000 |  |  |  | +0.0% |  | -2.3% |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 70.668 | 70.668 | 14.972 | 55.696 |  |  |  | +4.6% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v8_direct | ok | 67.579 | 67.579 | 12.265 | 55.314 |  |  |  | +0.0% |  | -4.4% |  |  |
