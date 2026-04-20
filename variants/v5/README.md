# torch-metal-gsplat-v5

Torch-native projected 2D Gaussian rasterizer for Apple Silicon / MPS with a Metal hot path.

## New in v5

- **Batchwise rendering**: accepts `[B,G,2/3]` inputs and renders `[B,H,W,3]`
- **Auto batch chunking**: `batch_strategy=auto|flatten|serial`
- **Inference-only fast path**: no sorted-ID writeback when gradients are not needed
- **Training fast path**: writes sorted IDs back into `binned_ids` and saves per-tile stop counts for backward
- **Runtime-specialized ablations** via env before import:
  - `GSP_TILE_SIZE=8|16|32`
  - `GSP_CHUNK=32|64|128`
  - `GSP_FAST_CAP=1024|2048|4096`

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
python benchmarks/benchmark_matrix.py --height 4096 --width 4096 --gaussians 65536 --batch-sizes 1,2,4 --warmup 1 --iters 3 --backward
```

## Notes

- input API is projected 2D splats, not full 3D camera projection
- depth gradients are zero; sort order is piecewise constant
- overflow tiles fall back to a slower path
- `auto` batch mode chunks large batches to cap total launched tiles / gaussians
