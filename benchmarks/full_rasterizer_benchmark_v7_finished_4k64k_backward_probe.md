# Full Rasterizer Benchmark

Generated: 2026-04-22 14:15:23

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
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v7_finished_hardware | 20330.785 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v7_finished_hardware | 21440.399 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v7_finished_hardware | ok | 21440.399 | 21440.399 | 254.783 | 21185.617 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v7_finished_hardware | ok | 20330.785 | 20330.785 | 212.092 | 20118.694 |  |  |  | +0.0% |  |  |  |  |
