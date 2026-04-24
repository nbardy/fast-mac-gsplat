# Full Rasterizer Benchmark

Generated: 2026-04-23 11:56:09

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
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v5_batched | 7.615 |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward | v5_batched | 8.128 |
| 4096x4096 | 1 | 6000 | microbench_uniform_random | forward | v5_batched | 6.559 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v5_batched | 11.781 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v5_batched | ok | 7.615 | 7.615 | 7.615 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v72_tiled_k2 | ok | 11.430 | 11.430 | 11.429 | 0.000 |  |  |  | +50.1% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v73_hybrid_k2 | ok | 7.861 | 7.861 | 7.860 | 0.000 |  |  |  | +3.2% |  |  |  |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward | v5_batched | ok | 8.128 | 8.128 | 8.128 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward | v72_tiled_k2 | ok | 19.125 | 19.125 | 19.125 | 0.000 |  |  |  | +135.3% |  |  |  |  |
| 512x512 | 1 | 65536 | microbench_uniform_random | forward | v73_hybrid_k2 | ok | 13.677 | 13.677 | 13.677 | 0.000 |  |  |  | +68.3% |  |  |  |  |
| 4096x4096 | 1 | 6000 | microbench_uniform_random | forward | v5_batched | ok | 6.559 | 6.559 | 6.559 | 0.001 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 6000 | microbench_uniform_random | forward | v72_tiled_k2 | ok | 142.265 | 142.265 | 142.264 | 0.000 |  |  |  | +2068.9% |  |  |  |  |
| 4096x4096 | 1 | 6000 | microbench_uniform_random | forward | v73_hybrid_k2 | ok | 107.664 | 107.664 | 107.664 | 0.000 |  |  |  | +1541.4% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v5_batched | ok | 11.781 | 11.781 | 11.780 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v72_tiled_k2 | ok | 183.679 | 183.679 | 183.678 | 0.000 |  |  |  | +1459.1% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward | v73_hybrid_k2 | ok | 133.129 | 133.129 | 133.128 | 0.000 |  |  |  | +1030.0% |  |  |  |  |
