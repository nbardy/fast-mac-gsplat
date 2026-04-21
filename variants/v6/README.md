
# torch-metal-gsplat-v6

Torch-native projected 2D Gaussian rasterizer for Apple Silicon / MPS with a Metal hot path.

## What changes in v6

- **Batch-first renderer**: keeps native `[B,G,...] -> [B,H,W,3]` API.
- **Optional active-tile scheduling**: compacts fast tiles across the batch and dispatches only active fast tiles.
- **Count-sorted active tiles**: active fast tiles are optionally sorted by tile occupancy before render/backward.
- **No unconditional grad clone**: backward consumes the original contiguous grad image.
- **Adaptive stop counts**: training can save per-tile stop counts `always`, `never`, or `adaptive` (dense tiles only).
- **Overflow fallback** remains available for pathological tiles.

## Build

```bash
python setup.py build_ext --inplace
```

## Quick check

```bash
python tests/reference_check.py
```

## Benchmarks

```bash
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --case medium_sigma_3_8 --backward --profile
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case medium_sigma_3_8 --backward --profile
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case medium_sigma_3_8 --backward --profile --active-tiles
python benchmarks/benchmark_matrix.py --height 4096 --width 4096 --gaussians 65536 --batch-sizes 1,2,4,8 --warmup 1 --iters 3 --backward --shuffle-order
```

See `../../docs/v6_field_report.md` for the first Mac build, correctness, and
benchmark report. In the current uniform synthetic workloads, v6 is fastest with
active-tile scheduling disabled; enable active tiles for sparse-screen or
overflow-heavy experiments where compaction may pay for itself.

## Notes

- input API is projected 2D splats, not full 3D camera projection
- depth gradients are zero; sort order is piecewise constant
- overflow tiles fall back to a slower exact path
- v6 is intended as the **batch-focused** branch; v3 remains the best measured B=1 training path until v6 batch-backward wins clearly
