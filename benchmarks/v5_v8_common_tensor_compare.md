# V5/V8 Common-Tensor Benchmark

Generated: 2026-04-27 15:00:37 +07

V5 presorted receives the same generated Gaussian set after a stable depth sort and sets `inputs_sorted_by_depth=True`; the depth-sort time is intentionally excluded to measure the renderer-side win from avoiding redundant sort/unsort.

## Settings

- git HEAD: `d789179`
- python: `3.14.0`
- torch: `2.11.0`
- platform: `macOS-15.5-arm64-arm-64bit-Mach-O`
- warmup: `3`
- iters: `15`
- renderers: `v5_default,v5_presorted,v8_direct`
- modes: `forward,forward_backward`
- noise threshold: `5.0%`
- command: `/opt/homebrew/opt/python@3.14/bin/python3.14 benchmarks/v5_v8_common_tensor_compare.py --resolutions 512x512 --splats 6000 --batch-sizes 1,4 --distributions microbench_uniform_random --warmup 3 --iters 15 --include-accuracy-case --accuracy-resolution 64x64 --accuracy-splats 128 --accuracy-batch-size 2 --accuracy-distribution layered_depth --accuracy-seed 17 --output-md benchmarks/v5_v8_common_tensor_compare.md --output-jsonl benchmarks/v5_v8_common_tensor_compare.jsonl`

## V5 Default vs Presorted Parity

| Case | Status | Image Max Err | Grad Max Err | Max Err | Threshold |
|---|---|---:|---:|---:|---:|
| 64x64_B2_G128_layered_depth_seed17 | ok | 0.000e+00 | 5.960e-08 | 5.960e-08 | 1.000e-05 |

## V5 Presorted Verdict

| Case | Mode | V5 Default ms | V5 Presorted ms | Delta | Speedup | Verdict |
|---|---|---:|---:|---:|---:|---|
| 512x512_B1_G6000_microbench_uniform_random_seed0 | forward | 11.388 | 11.799 | +3.6% | -3.5% | noisy_flat |
| 512x512_B1_G6000_microbench_uniform_random_seed0 | forward_backward | 13.882 | 14.636 | +5.4% | -5.2% | slower |
| 512x512_B4_G6000_microbench_uniform_random_seed0 | forward | 8.372 | 8.824 | +5.4% | -5.1% | slower |
| 512x512_B4_G6000_microbench_uniform_random_seed0 | forward_backward | 31.505 | 36.301 | +15.2% | -13.2% | slower |
| 64x64_B2_G128_layered_depth_seed17 | forward | 12.998 | 3.132 | -75.9% | +314.9% | faster |
| 64x64_B2_G128_layered_depth_seed17 | forward_backward | 9.580 | 7.140 | -25.5% | +34.2% | faster |

## Timing Results

| Case | Mode | Renderer | Status | Median ms | Mean ms | Fwd Median ms | Bwd Median ms | Stddev ms | Delta vs V5 Default | Best | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 512x512_B1_G6000_microbench_uniform_random_seed0 | forward | v5_default | ok | 11.388 | 12.204 | 11.388 | 0.000 | 1.827 |  | v8_direct |  |
| 512x512_B1_G6000_microbench_uniform_random_seed0 | forward | v5_presorted | ok | 11.799 | 12.840 | 11.799 | 0.000 | 3.293 | +3.6% | v8_direct | presort excluded |
| 512x512_B1_G6000_microbench_uniform_random_seed0 | forward | v8_direct | ok | 8.435 | 8.659 | 8.435 | 0.000 | 1.107 | -25.9% | v8_direct |  |
| 512x512_B1_G6000_microbench_uniform_random_seed0 | forward_backward | v5_default | ok | 13.882 | 17.184 | 7.719 | 6.491 | 5.290 |  | v8_direct |  |
| 512x512_B1_G6000_microbench_uniform_random_seed0 | forward_backward | v5_presorted | ok | 14.636 | 15.297 | 8.705 | 6.334 | 2.582 | +5.4% | v8_direct | presort excluded |
| 512x512_B1_G6000_microbench_uniform_random_seed0 | forward_backward | v8_direct | ok | 9.700 | 9.628 | 5.070 | 4.306 | 0.725 | -30.1% | v8_direct |  |
| 512x512_B4_G6000_microbench_uniform_random_seed0 | forward | v5_default | ok | 8.372 | 8.638 | 8.372 | 0.000 | 1.439 |  | v8_direct |  |
| 512x512_B4_G6000_microbench_uniform_random_seed0 | forward | v5_presorted | ok | 8.824 | 9.616 | 8.824 | 0.000 | 2.919 | +5.4% | v8_direct | presort excluded |
| 512x512_B4_G6000_microbench_uniform_random_seed0 | forward | v8_direct | ok | 7.092 | 7.182 | 7.092 | 0.000 | 0.667 | -15.3% | v8_direct |  |
| 512x512_B4_G6000_microbench_uniform_random_seed0 | forward_backward | v5_default | ok | 31.505 | 61.810 | 17.516 | 17.964 | 75.690 |  | v8_direct |  |
| 512x512_B4_G6000_microbench_uniform_random_seed0 | forward_backward | v5_presorted | ok | 36.301 | 38.957 | 13.756 | 23.002 | 10.066 | +15.2% | v8_direct | presort excluded |
| 512x512_B4_G6000_microbench_uniform_random_seed0 | forward_backward | v8_direct | ok | 26.637 | 30.415 | 9.587 | 16.847 | 10.694 | -15.5% | v8_direct |  |
| 64x64_B2_G128_layered_depth_seed17 | forward | v5_default | ok | 12.998 | 14.926 | 12.998 | 0.000 | 6.228 |  | v5_presorted |  |
| 64x64_B2_G128_layered_depth_seed17 | forward | v5_presorted | ok | 3.132 | 5.126 | 3.132 | 0.000 | 3.204 | -75.9% | v5_presorted | presort excluded |
| 64x64_B2_G128_layered_depth_seed17 | forward | v8_direct | ok | 5.413 | 5.789 | 5.413 | 0.000 | 1.357 | -58.4% | v5_presorted |  |
| 64x64_B2_G128_layered_depth_seed17 | forward_backward | v5_default | ok | 9.580 | 14.245 | 6.651 | 3.116 | 9.302 |  | v8_direct |  |
| 64x64_B2_G128_layered_depth_seed17 | forward_backward | v5_presorted | ok | 7.140 | 8.496 | 5.382 | 2.110 | 3.316 | -25.5% | v8_direct | presort excluded |
| 64x64_B2_G128_layered_depth_seed17 | forward_backward | v8_direct | ok | 6.291 | 7.168 | 4.227 | 2.307 | 2.132 | -34.3% | v8_direct |  |
