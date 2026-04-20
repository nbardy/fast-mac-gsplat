# Direct Torch Reference Comparison

Date: 2026-04-20

## What this baseline is

`benchmarks/compare_v2_v3.py --include-torch-reference` now includes a direct
Torch reference renderer. It sorts splats by detached depth, then loops over
splats and applies vectorized Torch tensor operations over all pixels.

This is a useful small-scene baseline because it keeps the math in plain Torch,
but it is not a viable 4K / 65,536-splat renderer. At that size, the direct
reference would do roughly:

```text
4096 * 4096 * 65536 = 1,099,511,627,776 pixel-splat evaluations
```

The benchmark therefore skips the Torch reference when `height * width *
gaussians` exceeds `--torch-max-work-items`.

## Command

```bash
uv run python benchmarks/compare_v2_v3.py \
  --height 128 --width 128 --gaussians 512 \
  --warmup 1 --iters 3 \
  --include-torch-reference --tile-stats

uv run python benchmarks/compare_v2_v3.py \
  --height 128 --width 128 --gaussians 512 \
  --warmup 1 --iters 3 --backward \
  --include-torch-reference --tile-stats
```

## Results

Forward:

| Case | v2 fastpath | v3 candidate | Torch direct | Torch / v3 |
| --- | ---: | ---: | ---: | ---: |
| sigma 1-5 px | `4.654 ms` | `7.443 ms` | `163.244 ms` | `21.9x` |
| sigma 3-8 px | `4.666 ms` | `6.114 ms` | `162.016 ms` | `26.5x` |

Forward + backward:

| Case | v2 fastpath | v3 candidate | Torch direct | Torch / v3 |
| --- | ---: | ---: | ---: | ---: |
| sigma 1-5 px | `7.972 ms` | `8.940 ms` | `828.056 ms` | `92.6x` |
| sigma 3-8 px | `10.850 ms` | `11.928 ms` | `866.864 ms` | `72.7x` |

Tile stats:

| Case | Total tile pairs | Max tile count | Mean tile count | Overflow tiles |
| --- | ---: | ---: | ---: | ---: |
| sigma 1-5 px | `1,847` | `42` | `28.859` | `0` |
| sigma 3-8 px | `3,566` | `85` | `55.719` | `0` |

## Interpretation

The direct Torch reference is already much slower at 128x128 / 512 splats:
roughly `22-27x` slower than v3 forward and `73-93x` slower than v3
forward+backward.

At this small size, v2 is still faster than v3 because v3 pays extra overhead
for 256-thread tile groups, parameter staging, and tile-local reductions. The
4K / 65,536-splat benchmark is the more relevant regime for v3, where v3 wins
the training path.

The next clean benchmark improvement is to add randomized renderer order and
multi-seed median reporting, then rerun after all other heavy local jobs are
stopped.
