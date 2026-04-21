# v6 Field Report

Date: 2026-04-21

## Summary

v6 is a real source handoff, not just pseudocode. It builds on Apple Silicon,
imports as `torch_gsplat_bridge_v6`, passes the small forward/backward reference
check, and is now wired into the shared benchmark scripts.

The current result is not a replacement for v5. On the synthetic workloads tested
here, v6's batch scheduling and active-tile machinery add forward overhead that
is larger than its backward savings. v5 remains the faster recommended path for
the existing benchmark distributions.

## Integration Work

- extracted `torch_metal_gsplat_v6.tar.gz` to `variants/v6/`
- preserved the source bundle as `source_artifacts/torch_metal_gsplat_v6.tar.gz`
- built the extension with `python setup.py build_ext --inplace`
- patched direct script execution by adding the variant root to `sys.path` in:
  - `variants/v6/tests/reference_check.py`
  - `variants/v6/benchmarks/benchmark_mps.py`
- added v6 to `benchmarks/compare_v2_v3.py`
- added v6 to Dynaworld's renderer stack benchmark selector
- added v6 benchmark ablation flags:
  - `--active-tiles` / `--no-active-tiles`
  - `--sort-active-tiles` / `--no-sort-active-tiles`
- extended the v6 matrix benchmark to sweep active-tile and active-sort modes

## Correctness

Command:

```bash
cd variants/v6
python setup.py build_ext --inplace
python tests/reference_check.py
```

Result:

```text
B=1 image max error: 5.960464477539063e-08
B=1 means grad max error: 2.4010660126805305e-10
B=1 conics grad max error: 9.313225746154785e-10
B=1 colors grad max error: 9.313225746154785e-10
B=1 opacities grad max error: 1.862645149230957e-09
B=2 image max error: 5.960464477539063e-08
B=2 means grad max error: 3.710738383233547e-10
B=2 conics grad max error: 1.862645149230957e-09
B=2 colors grad max error: 9.313225746154785e-10
B=2 opacities grad max error: 1.862645149230957e-09
```

## 1024x512 / 2048 Splats / B=1

Shared v2/v3/v5/v6 benchmark, `warmup=1`, `iters=5`.

Forward:

| Case | v2 | v3 | v5 | v6 |
|---|---:|---:|---:|---:|
| sparse sigma 1-5 | 4.012 ms | 4.669 ms | 4.066 ms | 5.579 ms |
| medium sigma 3-8 | 4.431 ms | 5.021 ms | 4.685 ms | 7.737 ms |

Forward+backward:

| Case | v2 | v3 | v5 | v6 |
|---|---:|---:|---:|---:|
| sparse sigma 1-5 | 12.905 ms | 7.728 ms | 11.355 ms | 8.562 ms |
| medium sigma 3-8 | 6.946 ms | 5.634 ms | 6.339 ms | 7.979 ms |

v6 can be competitive on sparse backward, but it is not the B=1 winner.

## 4096x4096 / 65536 Splats Per Image / B=4

This matches the scientist-proposed batch-native stress shape.

v5:

```text
forward_backward mean_ms=248.405 median_ms=246.670
fwd_ms=38.699 bwd_ms=209.707
```

v6 default active-tile scheduling:

```text
forward_backward mean_ms=354.122 median_ms=309.023
fwd_ms=116.762 bwd_ms=237.361
```

v6 best ablated mode found here, with active compaction and active sort disabled:

```text
forward_backward mean_ms=301.263 median_ms=301.098
fwd_ms=111.459 bwd_ms=189.804
```

Interpretation: v6 improves backward in the best ablation, but the forward path is
roughly 3x slower than v5, so total time loses.

Profile for this scene:

```text
tiles=262144
active_tile_count=262065
total_pairs=2140310
mean_pairs_per_tile=8.1646
p95_pairs_per_tile=13
max_pairs_per_tile=24
overflow_tile_count=0
mean_stop_ratio=0.9997
p95_stop_ratio=1.0
dense_active_tile_count=0
```

The scene is nearly all active light tiles. Active-tile compaction cannot remove
much work, count sorting has little useful structure to exploit, and adaptive
stop counts do not prune because the stop ratio is effectively 1.0.

## 4096x4096 / 65536 Total Splats / B=4

Dynaworld stack benchmark with `G=16384` per image, `warmup=1`, `iters=3`,
forward+backward, renderers `v5,v6`.

| Case | v5 | v6 |
|---|---:|---:|
| sparse sigma 1-5 | 168.412 ms | 223.853 ms |
| medium sigma 3-8 | 194.507 ms | 238.707 ms |

v6 is also behind v5 at the fixed-total 64k splat workload.

## Current Model

v6 is a useful experimental branch, but the new scheduling ideas only help if
they remove enough tile work to repay their overhead. The current synthetic
uniform scenes do not satisfy that condition:

- almost every tile is active at 4K B=4
- no overflow tiles appear
- no dense active tiles appear under the default threshold
- stop counts replay almost the full tile list
- active tile sorting adds CPU/GPU scheduling overhead without reducing kernel work

The biggest concrete regression is the v6 forward path. Even with active
compaction disabled, v6 forward is about 109-111 ms where v5 is about 34-39 ms
on the same B=4 4K workload. Backward can be modestly better, but not enough to
pay for that forward cost.

## Next Tests

- try clustered or sparse-screen scenes where active-tile compaction actually
  removes many tiles
- try overflow or high-occupancy scenes where count scheduling may reduce tail
  latency
- profile v6 forward kernel against v5 to isolate the added cost from active
  indirection, stop-count bookkeeping, or kernel structure
- consider keeping v6 as a research branch until its forward path is reduced
  back near v5
