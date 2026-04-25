# V9 Tile Exact Backward-State Smoke

Date: 2026-04-25

Variant:

```text
variants/v9_hw_tile_exact_probe
```

Command:

```text
python3 benchmarks/benchmark_interop.py --sizes 64x64,512x512,1080x1920 --warmup 3 --iters 20 --paths direct,exact --jsonl ../../benchmarks/v9_hw_tile_exact_backward_state_smoke.jsonl
```

## Correctness Gate

```text
python3 tests/interop_check.py
```

Observed:

```text
tile_exact_overlap_max_abs_err=0.0
tile_exact_overlap_tile_stop_counts=[2, 2, 2, 2]
render_to_mps_tensor_max_abs_err=0.0
```

The exact path emits:

- direct Torch/MPS `RGBA32F` render target;
- MPS `int32` `tile_stop_counts`, one value per tile;
- MPS `float32[tiles,4]` debug reports.

This is still a toy fullscreen two-splat path. It validates visible-fragment
stop-state plumbing, not full V8 candidate-prefix state for invisible splats.

## Timing

Apple M4, Python 3.14, `torch.mps.synchronize()` after each timed op.

| Size | Path | Median ms | Mean ms | Min ms | Max ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 64x64 | direct RGBA32F render | 0.778 | 0.899 | 0.422 | 1.998 |
| 512x512 | direct RGBA32F render | 1.160 | 1.397 | 0.420 | 3.507 |
| 1080x1920 | direct RGBA32F render | 3.817 | 6.354 | 1.728 | 33.087 |
| 64x64 | exact imageblock + atomic stop | 1.407 | 1.846 | 0.716 | 4.984 |
| 512x512 | exact imageblock + atomic stop | 1.267 | 1.619 | 1.012 | 5.825 |
| 1080x1920 | exact imageblock + atomic stop | 4.792 | 4.781 | 3.817 | 6.019 |

## Interpretation

The new state plumbing is not exploding memory and did not crash. The exact path
keeps the 48 B/sample imageblock state and returns only compact per-tile stop
metadata globally.

The direct render baseline was noisy at 1080p in this run, so median is more
useful than mean. The exact path was about 1.26x the direct constant-render
median at 1080p. That is acceptable for a state smoke, but it is not a claim
that hardware Gaussian training is fast yet.

The important new constraint is from Metal itself: tile shaders rejected
shader-declared threadgroup scratch arrays. For tile-level stop, the viable
primitive is fragment-side atomic max into a buffer, or a separate compute pass.
