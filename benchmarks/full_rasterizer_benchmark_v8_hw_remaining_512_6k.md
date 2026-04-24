# Full Rasterizer Benchmark

Generated: 2026-04-24 15:42:32

Lower milliseconds are better. `% vs torch` is speedup over direct Torch when Torch was small enough to run. `% vs v6 direct` and `% vs v6 refined direct` are time deltas, so negative means faster than that baseline.

## Settings

- warmup: `1`
- iters: `3`
- seed: `0`
- timeout seconds per cell: `120.0`
- accuracy checks: `False`
- accuracy max work items: `16000000`

## Winners

| Resolution | B | Splats | Distribution | Mode | Best Renderer | Mean ms |
|---|---:|---:|---|---|---|---:|
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v8_direct | 17.836 |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v8_direct | 25.268 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v8_direct | 5.918 |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v8_direct | 9.072 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v8_direct | ok | 5.918 | 4.654 | 5.918 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v8_hw_eval_fallback | ok | 6.465 | 6.719 | 6.464 | 0.000 |  |  |  | +9.2% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward | v8_hw_train_fallback | ok | 6.412 | 5.294 | 6.412 | 0.000 |  |  |  | +8.4% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v8_direct | ok | 9.072 | 7.022 | 3.883 | 5.189 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v8_hw_eval_fallback | ok | 13.379 | 13.993 | 5.074 | 8.305 |  |  |  | +47.5% |  |  |  |  |
| 512x512 | 1 | 6000 | microbench_uniform_random | forward_backward | v8_hw_train_fallback | ok | 12.704 | 10.354 | 6.708 | 5.997 |  |  |  | +40.0% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v8_direct | ok | 17.836 | 17.169 | 17.835 | 0.000 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v8_hw_eval_fallback | ok | 34.457 | 34.857 | 34.457 | 0.000 |  |  |  | +93.2% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward | v8_hw_train_fallback | ok | 19.337 | 16.594 | 19.337 | 0.000 |  |  |  | +8.4% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v8_direct | ok | 25.268 | 25.346 | 17.387 | 7.880 |  |  |  | +0.0% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v8_hw_eval_fallback | ok | 32.468 | 32.303 | 21.097 | 11.371 |  |  |  | +28.5% |  |  |  |  |
| 512x512 | 1 | 6000 | clustered_hot_tiles | forward_backward | v8_hw_train_fallback | ok | 30.397 | 31.519 | 21.176 | 9.221 |  |  |  | +20.3% |  |  |  |  |
