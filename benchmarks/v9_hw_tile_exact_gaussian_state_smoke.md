# V9 HW Tile Exact Gaussian State Smoke

Date: 2026-04-25

Variant:

```text
variants/v9_hw_tile_exact_probe
```

Command:

```bash
python3 benchmarks/benchmark_interop.py --sizes 32x32,512x512,1080x1920 --warmup 3 --iters 20 --paths direct,exact,gaussian --jsonl ../../benchmarks/v9_hw_tile_exact_gaussian_state_smoke.jsonl
```

Apple M4, Python 3.14, Torch MPS. Each sample synchronizes with
`torch.mps.synchronize()` after the op.

| Path | Resolution | Median ms | Mean ms | Min ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct RGBA32F render | 32x32 | 0.551 | 0.696 | 0.425 | 3.342 |
| direct RGBA32F render | 512x512 | 0.652 | 0.818 | 0.537 | 3.958 |
| direct RGBA32F render | 1080x1920 | 1.826 | 2.179 | 1.479 | 4.731 |
| exact imageblock two-splat overlap | 32x32 | 0.581 | 0.799 | 0.430 | 4.238 |
| exact imageblock two-splat overlap | 512x512 | 1.212 | 1.708 | 0.812 | 5.762 |
| exact imageblock two-splat overlap | 1080x1920 | 4.519 | 5.928 | 2.930 | 11.451 |
| exact imageblock four-Gaussian diagnostic | 32x32 | 0.873 | 1.400 | 0.506 | 6.243 |
| exact imageblock four-Gaussian diagnostic | 512x512 | 1.693 | 2.469 | 1.399 | 7.351 |
| exact imageblock four-Gaussian diagnostic | 1080x1920 | 8.610 | 8.560 | 7.085 | 10.615 |

Correctness smoke:

```text
tile_exact_gaussian_max_abs_err=1.1920928955078125e-07
tile_exact_gaussian_tile_stop_counts=[4, 4, 4, 4]
```

Interpretation:

- The Gaussian fragment/imageblock recurrence is accurate against the dense CPU
  scalar reference for this fixed ordered diagnostic.
- The path is not a training-speed win. At 1080p it is about 4.7x the direct
  constant render median and about 1.9x the exact two-splat toy median.
- This still uses fullscreen diagnostic draws, not clipped projected quads or
  V8 tile-bin ingestion. The next speed gate is reducing fragment work, not
  adding more per-pixel imageblock state.
