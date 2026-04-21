
# torch-metal-gsplat-v6

Torch-native projected 2D Gaussian rasterizer for Apple Silicon / MPS with a Metal hot path.

## What changes in v6

- **Batch-first renderer**: keeps native `[B,G,...] -> [B,H,W,3]` API.
- **Optional active-tile scheduling**: compacts fast tiles across the batch and dispatches only active fast tiles.
- **Active policy gate**: `active_policy=off|on|auto`, with `auto` currently conservative because active scheduling still has measurable fixed overhead.
- **Count-sorted active tiles**: active fast tiles are optionally sorted by tile occupancy before render/backward.
- **No unconditional grad clone**: backward consumes the original contiguous grad image.
- **Adaptive stop counts**: training can save per-tile stop counts `always`, `never`, or `adaptive` (dense tiles only).
- **Optional pair-budget chunks**: `max_pairs_per_launch > 0` can split training batches by measured tile-pair load.
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
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case microbench_uniform_random --backward --profile --active-policy auto
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case microbench_uniform_random --backward --profile --max-pairs-per-launch 1500000
python benchmarks/benchmark_matrix.py --height 4096 --width 4096 --gaussians 65536 --batch-sizes 1,2,4,8 --warmup 1 --iters 3 --backward --shuffle-order
```

Benchmark cases:

- `microbench_uniform_random`: saturated uniform projected splats; useful for fixed-overhead regressions, not a real scene proxy
- `sparse_screen`: a few occupied screen regions with many empty tiles
- `clustered_hot_tiles`: hot clustered regions that exercise overflow and active scheduling
- `layered_depth`: overlapping regions with several depth bands
- `overflow_adversarial`: concentrated large splats for overflow stress
- `real_trace`: replays dumped `means2d/conics/colors/opacities/depths` tensors from `--trace-file`

See `../../docs/v6_field_report.md` for the first Mac build, correctness, and
benchmark report. In the current uniform synthetic workloads, v6 is fastest with
active-tile scheduling disabled; enable active tiles for clustered or
overflow-heavy experiments where compaction may pay for itself. Pair-budget
chunking is intentionally disabled by default because it adds a pre-binning pass;
sweep `--max-pairs-per-launch` before making it a default.

## Notes

- input API is projected 2D splats, not full 3D camera projection
- depth gradients are zero; sort order is piecewise constant
- overflow tiles fall back to a slower exact path
- v6 is intended as the **batch-focused** branch; v3 remains the best measured B=1 training path until v6 batch-backward wins clearly
