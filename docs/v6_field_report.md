# v6 Field Report

Date: 2026-04-21

## Summary

v6 is a real source handoff, not just pseudocode. It builds on Apple Silicon,
imports as `torch_gsplat_bridge_v6`, passes the small forward/backward reference
check, and is now wired into the shared benchmark scripts.

Follow-up testing found and fixed the obvious forward regression: the
`use_active_tiles=False` path still routed through the active-tile kernels and
prefilled the full output with background. v6 now exposes its existing v5-style
direct tile kernels and uses them when active-tile scheduling is disabled. On the
current uniform synthetic workload, the direct-tile v6 path is the faster default.

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
- follow-up fix: exposed v6 direct tile forward/backward kernels through the
  Python bindings and made active-tile scheduling opt-in

## Correctness

Command:

```bash
cd variants/v6
python setup.py build_ext --inplace
python tests/reference_check.py
```

Result:

```text
B=1 direct image max error: 5.960464477539063e-08
B=1 direct means grad max error: 2.4010660126805305e-10
B=1 direct conics grad max error: 9.313225746154785e-10
B=1 direct colors grad max error: 9.313225746154785e-10
B=1 direct opacities grad max error: 1.862645149230957e-09
B=2 direct image max error: 5.960464477539063e-08
B=2 direct means grad max error: 3.710738383233547e-10
B=2 direct conics grad max error: 1.862645149230957e-09
B=2 direct colors grad max error: 9.313225746154785e-10
B=2 direct opacities grad max error: 1.862645149230957e-09
B=1 active image max error: 5.960464477539063e-08
B=1 active means grad max error: 2.4010660126805305e-10
B=1 active conics grad max error: 9.313225746154785e-10
B=1 active colors grad max error: 9.313225746154785e-10
B=1 active opacities grad max error: 1.862645149230957e-09
B=2 active image max error: 5.960464477539063e-08
B=2 active means grad max error: 3.710738383233547e-10
B=2 active conics grad max error: 1.862645149230957e-09
B=2 active colors grad max error: 9.313225746154785e-10
B=2 active opacities grad max error: 1.862645149230957e-09
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

Fair same-process alternating run, shared input distribution, `warmup=2`,
`iters=7`, no concurrent GPU benchmark jobs.

Forward:

| Renderer | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| v5 | 34.198 ms | 34.168 ms | 33.447 ms | 35.348 ms |
| v6 active | 112.878 ms | 112.565 ms | 109.986 ms | 116.020 ms |
| v6 direct | 38.014 ms | 34.121 ms | 33.874 ms | 60.298 ms |

Forward+backward:

| Renderer | Total Mean | Total Median | Forward Mean | Backward Mean |
|---|---:|---:|---:|---:|
| v5 | 250.827 ms | 250.705 ms | 37.988 ms | 212.839 ms |
| v6 active | 309.647 ms | 310.049 ms | 116.164 ms | 193.484 ms |
| v6 direct | 232.186 ms | 231.956 ms | 38.189 ms | 193.997 ms |

Standalone v6 default after the direct-kernel default change, `warmup=2`,
`iters=5`:

```text
forward_backward mean_ms=229.489 median_ms=228.060
fwd_ms=37.539 bwd_ms=191.950 active=False
```

Earlier pre-fix measurement, kept here because it explains the bug:

```text
v5: mean 248.405 ms, median 246.670 ms, fwd 38.699 ms, bwd 209.707 ms
v6 active/default before default change: mean 354.122 ms, median 309.023 ms, fwd 116.762 ms, bwd 237.361 ms
v6 no-active before direct-kernel fix: mean 301.263 ms, median 301.098 ms, fwd 111.459 ms, bwd 189.804 ms
```

Interpretation: the 3x forward regression was not just machine noise. It came
from v6 still using the active-tile rendering path even when active scheduling
was disabled. That path prefilled the entire `[B,H,W,3]` output with background
and then rendered active tiles. On B=4 4K RGB float output, that is about 805 MB
of extra writes before the real tile render. The direct-tile path avoids that and
recovers v5-like forward speed.

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
stop counts do not prune because the stop ratio is effectively 1.0. For this
shape, active scheduling should stay off.

## 4096x4096 / 65536 Total Splats / B=4

Dynaworld stack benchmark with `G=16384` per image, `warmup=1`, `iters=3`,
forward+backward, renderers `v5,v6`. These are pre-fix numbers with v6 still
using the active path by default; rerun after the direct-kernel default change
before using them for a headline table.

| Case | v5 | v6 |
|---|---:|---:|
| sparse sigma 1-5 | 168.412 ms | 223.853 ms |
| medium sigma 3-8 | 194.507 ms | 238.707 ms |

These fixed-total numbers are superseded for v6 default comparisons.

## Current Model

v6 direct mode is now the preferred default for current uniform workloads. The
active scheduling ideas should be treated as an opt-in experimental mode that
only helps if they remove enough tile work to repay their overhead. The current
synthetic uniform scenes do not satisfy that condition:

- almost every tile is active at 4K B=4
- no overflow tiles appear
- no dense active tiles appear under the default threshold
- stop counts replay almost the full tile list
- active tile sorting adds CPU/GPU scheduling overhead without reducing kernel work
- active forward also pays a full-output background fill, which is expensive at
  B=4 4K

The direct v6 path is now faster than v5 on total forward+backward for the
measured B=4 4K case because forward matches v5 and backward is about 19 ms
faster.

## v6.1 Scenario Patch

The synthetic random benchmark is now explicitly treated as
`microbench_uniform_random`: useful for catching fixed overhead regressions, but
not representative of sparse-screen, clustered, occlusion-heavy, or trace-backed
training scenes.

v6.1 adds:

- `active_policy=off|on|auto`
- optional pair-budget training chunks via `max_pairs_per_launch`
- `real_trace` replay support for saved `means2d/conics/colors/opacities/depths`
- new synthetic cases:
  - `sparse_screen`
  - `clustered_hot_tiles`
  - `layered_depth`
  - `overflow_adversarial`
- matrix benchmark sweeps for active policy and pair-budget settings

Pair-budget chunking is intentionally disabled by default with
`max_pairs_per_launch=0`. It needs a pre-binning pass to measure pair load, so it
is a useful experiment knob but not a free default.

The scientist-suggested `active_tile_fraction < 0.45` auto threshold was too
eager in local field tests. Sparse/layered cases around 15-29% active tiles were
mixed or slower with active scheduling, while the clearly active-friendly
clustered overflow case had about 5% active tiles plus overflow. The v6.1 auto
gate is therefore tighter:

```text
active when active_tile_fraction < 0.10
or overflow_tile_count > 0
or max_pairs_per_tile > 2 * max_fast_pairs
but reject active when active_tile_fraction > 0.75 and overflow_tile_count == 0
```

This keeps the saturated uniform microbench on the direct path, rejects ordinary
sparse-screen cases where the active path has not proven reliable, and still
activates for clustered overflow stress.

Noisy local policy sweep, `1024x512`, `B=4`, `G=4096`, `warmup=1`, `iters=3`,
forward+backward. The machine had unrelated long-running Python jobs active, so
these are pattern checks, not headline numbers.

| Case | Active Fraction | Overflow Tiles | Best Observed Pattern |
|---|---:|---:|---|
| `microbench_uniform_random` | 1.00 | 0 | direct/off best; auto resolves direct |
| `sparse_screen` | ~0.15 | 0 | mixed; active not reliable enough for auto |
| `clustered_hot_tiles` | ~0.05 | ~38-40 | auto resolves active; often improves over direct |
| `layered_depth` | ~0.28 | 0 | mixed; stop-count pruning helps but active scheduling is not clearly worth it |

One 4K rerun under the same noisy machine state showed v5 also much slower than
the previous quiet-machine baseline, so the absolute 4K numbers from that pass
should not replace the earlier field report. The relative behavior still held:
forced active scheduling caused a large forward penalty on saturated random
screens, while `active_policy=auto` rejected it.

## Saturated Backward Barrier Fix

The v5 training-parity failure exposed a Metal correctness trap that also
existed in v6: backward kernels looped over per-pixel `end_i` while using
`threadgroup_barrier()`. In crowded or high-opacity tiles, different pixels can
stop at different splat indices, which means the threadgroup can execute
divergent barrier counts and silently corrupt gradients.

Patched v6 kernels:

```text
tile_fast_backward_saved
tile_active_backward_saved
tile_overflow_backward
```

The reverse chunk loops now use uniform tile-level bounds:

```text
fast/direct: stop_count
active:      stop_count
overflow:    count
```

The per-pixel early-stop behavior remains as `global_i < end_i` inside the
loop. The v6 reference check now includes saturated 64-splat cases for both
direct and active paths.

## Next Tests

- rerun the fixed-total 64k stack benchmark with the new v6 direct default
- collect real projected traces from Dynaworld training batches and replay them
  with `--case real_trace --trace-file ...`
- rerun clustered, sparse-screen, layered-depth, and overflow-adversarial sweeps
  on a quiet machine
- try overflow or high-occupancy scenes where count scheduling may reduce tail
  latency
- if active scheduling is kept, avoid full-output background prefill when most
  tiles are active
