# V9 Full Backward Compute-Replay Smoke

Date: 2026-04-25

Variant:

```text
variants/v9_hw_tile_exact_probe
```

Command:

```text
python3 benchmarks/benchmark_full_backward.py --height 512 --width 512 --gaussians 4096 --warmup 2 --iters 5 --jsonl ../../benchmarks/v9_full_backward_compute_replay_smoke.jsonl
```

Backend:

```text
v8_hw_eval_compute_replay
```

This is the V9 full-backward base using proven V8/V8-hw-eval compute replay.
It is a correct training path, not hardware-raster state replay yet.

## Timing

Apple M4, B=1, 512x512, 4,096 projected splats:

| Metric | Median ms | Mean ms |
| --- | ---: | ---: |
| Forward | 3.092 | 2.953 |
| Backward | 2.134 | 2.834 |
| Forward + backward | 5.235 | 5.787 |

## Correctness

The paired correctness check:

```text
python3 tests/full_backward_check.py
```

Observed:

```text
image max error: 5.960464477539063e-08
means grad max error: 1.964508555829525e-10
conics grad max error: 1.862645149230957e-09
colors grad max error: 9.313225746154785e-10
opacities grad max error: 3.725290298461914e-09
```
