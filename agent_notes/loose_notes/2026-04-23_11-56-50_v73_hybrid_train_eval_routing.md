# 2026-04-23 v7.3 Hybrid Train/Eval Routing

User asked to apply the key fixes from the v5-style v7.2 hybrid plan and asked
what "v5/v6-style compute train path" means.

Implemented in `variants/v7_hybrid_v5style`:

- added `gsplat_metal_v73.forward_eval`, a hardware-raster eval op that skips
  tiled front-K capture, CPU tile-bin construction, and backward saved-state
  allocation
- added `RasterConfig.train_backend = auto | v5_compute | hardware`
- default training route now tries the sibling v5 compute rasterizer when
  gradients are required, preserving the v7.2-style hardware backward behind
  benchmark suffix `_hwtrain`
- benchmark parser now accepts `_trainv5` / `_v5train` / `_compute` and
  `_hwtrain` suffixes for v7.3 hybrid renderers

Meaning of "v5/v6-style compute train path":

- not Metal fixed-function triangle rasterization
- MPS/Metal compute kernels bin splats into tiles, render pixels per tile, save
  compact tile state, then backward reuses that state
- this avoids the v7.2 training problem: hardware texture output plus CPU
  front-state/readback round trips

Key checks:

- `python3 -m py_compile benchmarks/benchmark_full_matrix.py variants/v7_hybrid_v5style/torch_gsplat_bridge_v73/rasterize.py`
- `python3 setup.py build_ext --inplace` in `variants/v7_hybrid_v5style`
- `python3 tests/reference_tiled_exactness.py`
- 16x16 root harness forward: image max error `1.788e-7`
- 16x16 root harness F+B: grad max error `2.98e-8`

Benchmarks:

- `benchmarks/full_rasterizer_benchmark_v73_hybrid_trainroute_512.md`
- `benchmarks/full_rasterizer_benchmark_v73_hybrid_trainroute_4k.md`
- `benchmarks/full_rasterizer_benchmark_v73_hybrid_evalroute.md`

Most important numbers:

| Case | Best relevant result |
|---|---:|
| 4K/64K B=1 uniform F+B v5 | 73.681 ms |
| 4K/64K B=1 uniform F+B v72 k2 | 438.782 ms |
| 4K/64K B=1 uniform F+B v73 hybrid default | 76.464 ms |
| 4K/64K B=1 uniform F+B v73 `_hwtrain` | 436.552 ms |
| 4K/64K B=1 uniform forward v5 | 11.781 ms |
| 4K/64K B=1 uniform forward v72 k2 | 183.679 ms |
| 4K/64K B=1 uniform forward v73 hardware eval | 133.129 ms |

Interpretation:

- Training route fix worked: v73 default is now within 3.8% of v5 at the 4K/64K
  uniform F+B probe and 82.6% faster than v72.
- Hardware eval no-capture fix helped versus v72, but hardware eval is still far
  slower than v5 at 4K because the output still round-trips through a Metal
  texture readback before becoming a torch MPS tensor.
