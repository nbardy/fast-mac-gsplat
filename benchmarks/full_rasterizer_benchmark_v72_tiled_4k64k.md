# Full Rasterizer Benchmark

Generated: 2026-04-23 10:44:41

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
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v5_batched | 208.881 |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v5_batched | 89.923 |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v5_batched | 78.218 |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v5_batched | 1207.543 |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v5_batched | 69.576 |

## Full Results

| Resolution | B | Splats | Distribution | Mode | Renderer | Status | Mean ms | Median ms | Fwd ms | Bwd ms | Accuracy | Image Max Err | Grad Max Err | Slower Than Best | % vs Torch | % vs v6 Direct | % vs v6 Refined Direct | Notes |
|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v5_batched | ok | 78.218 | 78.218 | 15.204 | 63.013 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v72_tiled_k2 | ok | 382.507 | 382.507 | 163.335 | 219.173 |  |  |  | +389.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v72_tiled_k4 | ok | 477.450 | 477.450 | 183.626 | 293.824 |  |  |  | +510.4% |  |  |  |  |
| 4096x4096 | 1 | 65536 | microbench_uniform_random | forward_backward | v72_tiled_k8 | ok | 633.445 | 633.445 | 238.999 | 394.446 |  |  |  | +709.8% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v5_batched | ok | 69.576 | 69.576 | 12.613 | 56.963 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v72_tiled_k2 | ok | 314.068 | 314.068 | 129.787 | 184.281 |  |  |  | +351.4% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v72_tiled_k4 | ok | 393.736 | 393.736 | 161.741 | 231.995 |  |  |  | +465.9% |  |  |  |  |
| 4096x4096 | 1 | 65536 | sparse_screen | forward_backward | v72_tiled_k8 | ok | 580.359 | 580.359 | 224.723 | 355.636 |  |  |  | +734.1% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v5_batched | ok | 208.881 | 208.881 | 141.538 | 67.342 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v72_tiled_k2 | ok | 459.409 | 459.409 | 137.734 | 321.675 |  |  |  | +119.9% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v72_tiled_k4 | ok | 494.852 | 494.852 | 159.442 | 335.410 |  |  |  | +136.9% |  |  |  |  |
| 4096x4096 | 1 | 65536 | clustered_hot_tiles | forward_backward | v72_tiled_k8 | ok | 706.538 | 706.538 | 215.408 | 491.130 |  |  |  | +238.2% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v5_batched | ok | 89.923 | 89.923 | 17.466 | 72.457 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v72_tiled_k2 | ok | 487.923 | 487.923 | 145.090 | 342.833 |  |  |  | +442.6% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v72_tiled_k4 | ok | 517.739 | 517.739 | 161.507 | 356.231 |  |  |  | +475.8% |  |  |  |  |
| 4096x4096 | 1 | 65536 | layered_depth | forward_backward | v72_tiled_k8 | ok | 715.296 | 715.296 | 210.416 | 504.880 |  |  |  | +695.5% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v5_batched | ok | 1207.543 | 1207.543 | 654.844 | 552.699 |  |  |  | +0.0% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v72_tiled_k2 | ok | 6525.487 | 6525.487 | 499.664 | 6025.823 |  |  |  | +440.4% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v72_tiled_k4 | ok | 6557.856 | 6557.856 | 525.197 | 6032.660 |  |  |  | +443.1% |  |  |  |  |
| 4096x4096 | 1 | 65536 | overflow_adversarial | forward_backward | v72_tiled_k8 | ok | 7045.124 | 7045.124 | 696.896 | 6348.228 |  |  |  | +483.4% |  |  |  |  |
