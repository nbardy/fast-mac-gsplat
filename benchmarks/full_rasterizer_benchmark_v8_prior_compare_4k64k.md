# Full Rasterizer Benchmark

Generated: 2026-04-24 14:31:10

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
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_direct | 15.157 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v8_direct | 65.754 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_direct | ok | 16.273 | 16.273 | 16.273 | 0.000 |  |  |  | +7.4% |  |  | -6.0% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_upgrade_direct | ok | 15.157 | 15.157 | 15.157 | 0.000 |  |  |  | +0.0% |  | -6.9% | -12.4% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v6_refined_direct | ok | 17.310 | 17.310 | 17.310 | 0.000 |  |  |  | +14.2% |  | +6.4% |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v8_direct | ok | 15.825 | 15.825 | 15.824 | 0.000 |  |  |  | +4.4% |  | -2.8% | -8.6% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_direct | ok | 72.988 | 72.988 | 15.334 | 57.654 |  |  |  | +11.0% |  |  | +3.0% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_upgrade_direct | ok | 70.545 | 70.545 | 14.707 | 55.837 |  |  |  | +7.3% |  | -3.3% | -0.5% |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v6_refined_direct | ok | 70.888 | 70.888 | 14.966 | 55.922 |  |  |  | +7.8% |  | -2.9% |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v8_direct | ok | 65.754 | 65.754 | 11.839 | 53.915 |  |  |  | +0.0% |  | -9.9% | -7.2% |  |
