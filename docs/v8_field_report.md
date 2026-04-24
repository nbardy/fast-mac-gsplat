# v8 Field Report

Date: 2026-04-24

## Summary

`variants/v8` is now a runnable v6-derived training baseline under its own
package and Torch custom-op namespace:

```text
Python package: torch_gsplat_bridge_v8
Torch ops:      gsplat_metal_v8
Metal source:   gsplat_v8_kernels.metal
Benchmark name: v8_direct
```

The direct tile math is intentionally unchanged from v6. The first real v8
optimization is host-side metadata: Python now builds a CPU metadata tensor and
an MPS metadata tensor. Metal kernels still receive the MPS metadata, but the
Objective-C++ bridge parses CPU metadata instead of copying tiny MPS metadata
tensors back to CPU on every op.

This removes one known hot-path synchronization source without changing the
forward/backward equations, binning math, saved sorted IDs, stop counts, or
tile-level gradient reductions.

## Files

- `variants/v8/torch_gsplat_bridge_v8/rasterize.py`
- `variants/v8/csrc/bindings.cpp`
- `variants/v8/csrc/shared/common.h`
- `variants/v8/csrc/metal/gsplat_metal.mm`
- `variants/v8/csrc/metal/gsplat_v8_kernels.metal`
- `benchmarks/benchmark_full_matrix.py`

## Low-Level Metal Research

The hardware raster path should stay separate from `v8_direct`.

Relevant Apple features for a later `v8_hw_experiment`:

- programmable blending: useful for same-pixel color/transmittance update in a
  render pass, but not enough for Gaussian-gradient reduction;
- imageblocks and tile shaders: useful for on-chip tile state, but explicit
  imageblock limits make unbounded per-pixel history/front-K a bad fit;
- raster order groups: useful for ordered per-pixel fragment updates, but they
  serialize overlapping fragments and do not solve backward suffix state;
- mesh shaders: possible later GPU-driven quad generation/culling, not needed
  for v8 direct;
- floating-point atomics: available on newer Apple GPU families, but still need
  quad/simd/tile reduction before atomics.

Primary sources:

- Apple Metal Feature Set Tables:
  `https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf`
- Apple Tile Shading Tech Talk:
  `https://developer.apple.com/videos/play/tech-talks/604/`
- Apple Raster Order Groups Tech Talk:
  `https://developer.apple.com/videos/play/tech-talks/605/`
- Apple Modern Rendering with Metal:
  `https://developer.apple.com/videos/play/wwdc2019/601/`
- PyTorch MPS docs:
  `https://docs.pytorch.org/docs/stable/mps.html`

Research context only:

- `https://arxiv.org/abs/2505.18764` reports differentiable hardware
  rasterization speedups using programmable blending and hybrid reductions on an
  RTX4080. This supports the direction but is not proof for Apple Metal/MPS.

Detailed shader pseudocode and two implementation plans are in
`docs/v8_hw_tile_raster_plans.md`.

## Correctness

Command:

```bash
cd variants/v8
python3 setup.py build_ext --inplace
python3 tests/reference_check.py
```

Result: pass.

Worst local reference-check errors:

| Case | Max image err | Worst grad err |
|---|---:|---:|
| direct B=1/B=2 small | `5.96e-08` | `1.86e-09` |
| saturated direct | `2.09e-07` | `1.19e-07` |
| active B=1/B=2 small | `5.96e-08` | `1.86e-09` |
| saturated active | `2.09e-07` | `1.19e-07` |

Shared matrix smoke:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 16x16,32x32,64x64 \
  --splats 4,64 \
  --batch-sizes 1,2 \
  --distributions microbench_uniform_random,clustered_hot_tiles \
  --renderers torch_direct,v6_direct,v8_direct \
  --modes forward,forward_backward \
  --accuracy \
  --accuracy-max-work-items 20000000 \
  --warmup 1 --iters 3 \
  --output-md benchmarks/full_rasterizer_benchmark_v8_smoke.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v8_smoke.jsonl
```

Result: 144 rows, 0 status errors, 0 v8 accuracy errors.

Worst v8 matrix-smoke errors:

```text
image max abs error: 8.100271e-05
grad max abs error:  6.082072e-04
```

## 512 / 6K Results

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 512x512 \
  --splats 6000 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth \
  --renderers v6_direct,v8_direct \
  --modes forward,forward_backward \
  --warmup 2 --iters 5 \
  --timeout-sec 180 \
  --output-md benchmarks/full_rasterizer_benchmark_v8_512_6k.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v8_512_6k.jsonl
```

Result: 32 rows, 0 errors. v8 beat v6 direct in 13 of 16 comparable cells.
Median v8 delta versus v6 direct was `-23.4%`.

Notable cells:

| Case | v6 mean ms | v8 mean ms | Delta |
|---|---:|---:|---:|
| B=1 uniform forward | 5.544 | 3.320 | -40.1% |
| B=1 uniform F+B | 9.663 | 7.178 | -25.7% |
| B=1 clustered F+B | 51.289 | 23.782 | -53.6% |
| B=4 uniform forward | 8.263 | 3.939 | -52.3% |
| B=4 uniform F+B | 21.706 | 18.832 | -13.2% |
| B=4 clustered F+B | 109.206 | 86.209 | -21.1% |

Observed losses:

| Case | v6 mean ms | v8 mean ms | Delta |
|---|---:|---:|---:|
| B=1 clustered forward | 18.577 | 19.183 | +3.3% |
| B=4 sparse forward | 7.261 | 8.023 | +10.5% |
| B=4 layered F+B | 22.305 | 24.259 | +8.8% |

## Prior-Variant Compare

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 512x512 \
  --splats 6000 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,clustered_hot_tiles \
  --renderers v6_direct,v6_upgrade_direct,v6_refined_direct,v8_direct \
  --modes forward,forward_backward \
  --warmup 2 --iters 5 \
  --timeout-sec 180 \
  --output-md benchmarks/full_rasterizer_benchmark_v8_prior_compare_512.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v8_prior_compare_512.jsonl
```

Result: v8 won 6 of 8 cells. v6 direct still won:

- B=1 clustered forward-only: v6 direct 18.471 ms, v8 21.419 ms
- B=4 clustered forward+backward: v6 direct 86.929 ms, v8 93.183 ms

## 4K / 64K Results

Command:

```bash
python3 benchmarks/benchmark_full_matrix.py \
  --resolutions 4096x4096 \
  --splats 65536 \
  --batch-sizes 1 \
  --distributions microbench_uniform_random \
  --renderers v6_direct,v6_upgrade_direct,v6_refined_direct,v8_direct \
  --modes forward,forward_backward \
  --warmup 1 --iters 2 \
  --timeout-sec 300 \
  --output-md benchmarks/full_rasterizer_benchmark_v8_prior_compare_4k64k.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v8_prior_compare_4k64k.jsonl
```

Result:

| Mode | Best | v8 status |
|---|---|---|
| forward | v6-upgrade direct, 15.157 ms | v8 was 15.825 ms, +4.4% vs best |
| forward+backward | v8 direct, 65.754 ms | best measured row |

4K/64K F+B comparison:

| Renderer | Mean ms | Forward ms | Backward ms | Delta vs v8 |
|---|---:|---:|---:|---:|
| v6 direct | 72.988 | 15.334 | 57.654 | +11.0% |
| v6-upgrade direct | 70.545 | 14.707 | 55.837 | +7.3% |
| v6-refined direct | 70.888 | 14.966 | 55.922 | +7.8% |
| v8 direct | 65.754 | 11.839 | 53.915 | 0.0% |

## Interpretation

The host-metadata split is a valid v8-base improvement. It does not change math,
but it removes one bridge-side MPS-to-CPU read in every Metal op. The effect is
small but real at 4K F+B and larger in launch-heavy 512 workloads.

This is not yet a finished v8:

- `metal_bin` still reads `tile_offsets[-1].item()` to size `binned_ids`;
- overflow fallback still uses Python `nonzero`/`.tolist()`/per-tile `.item()`;
- overflow patching still uses full-image clone/scatter;
- active policy still makes CPU decisions;
- hardware raster is not part of this path.

## Next Loop

1. Add an optional fixed/cached `pair_capacity` bin path that avoids
   `tile_offsets[-1].item()` when capacity is known.
2. Add a device capacity flag so fixed-capacity binning fails closed instead of
   writing out of bounds.
3. Move overflow-tile compaction to GPU.
4. Replace full-image overflow clone/scatter with tile-local patch kernels.
5. Re-run the same 512/6K and 4K/64K gates after each change.
6. Keep hardware work in `v8_hw_experiment`, not `v8_direct`.

Current decision: `v8_direct` is a better training base than v6 direct for the
measured forward+backward gates, but it is not yet better than every prior
forward-only row.
