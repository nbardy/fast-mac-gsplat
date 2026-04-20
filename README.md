# fast-mac-gsplat

Torch-first, Metal-backed Gaussian rasterizer fast path for Apple Silicon.

This is the recommended high-performance Mac path when you do not need Taichi.
It is a pure Metal/Torch extension built for speed. If you need a
Taichi-compatible renderer instead, use
[`taichi-gsplat-differentiable-render-mac`](https://github.com/nbardy/taichi-gsplat-differentiable-render-mac).

## Design goals

- Keep the Python training loop in **PyTorch**.
- Use **Metal kernels** for the renderer hot path.
- Minimize memory during training by saving only compact tile bins, **not** dense `G x H x W` activations.
- Keep backward differentiable via **recompute**, not via storing per-pixel alpha / transmittance graphs.

## Fast path pipeline

1. Python wrapper sorts Gaussians by depth **outside** the custom op.
2. `count_tiles` computes SnugBox-style bounds and exact tile intersections.
3. `tile_counts -> tile_offsets` uses `torch.cumsum` on MPS.
4. `emit_binned_ids` writes sorted-rank Gaussian IDs into per-tile bins via atomic cursors.
5. `tile_sort_render_forward` loads one tile's bin into threadgroup memory, bitonic-sorts the IDs, and immediately renders the tile.
6. Forward stores the sorted tile-local ID order in `binned_ids`.
7. `tile_sort_render_backward` reuses that saved order, recomputes the forward alpha chain, then reverse-scans it for gradients.

## Saved activations

Forward returns:

- `image`: `[H, W, 3]`
- `tile_counts`: `[T]`
- `tile_offsets`: `[T + 1]`
- `binned_ids`: `[N]`

That is the entire saved state for backward, plus the sorted inputs. `binned_ids` is written back in sorted tile-local order so backward does not repeat the tile bitonic sort. There is no saved dense `alpha`, `weights`, `dx`, or `power` volume.

## Differentiability

Gradients are implemented for:

- `means2d`
- `conics`
- `colors`
- `opacities`

Depth is treated as a piecewise-constant ordering key, so gradients w.r.t. `depths` are zero.

## Important caveat

This fast path uses a compile-time threadgroup list cap of **4096 splats per tile**. If a tile exceeds that count, the op throws and you should:

- increase `max_tile_pairs` only if you also raise the shader compile-time cap,
- reduce splat density,
- or add a slower overflow fallback (for example a global pair-sort path).

## Why this is faster than the earlier bridge

- No MLX in the hot path.
- No global pair sort on the common path.
- Tile bins are grouped once, then sorted **locally** in threadgroup memory.
- Backward recomputes local alpha instead of storing huge activations.

## Build

Build on macOS with Xcode command line tools and a matching PyTorch version:

```bash
python setup.py build_ext --inplace
```

Because this extension uses **internal PyTorch MPS headers**, engineering should pin the PyTorch version used for the Mac backend.

## Local Validation

This handoff is not just a stub. On the local Apple Silicon machine, after small packaging and schema fixes, the extension:

- builds with `python setup.py build_ext --inplace`
- imports and registers `torch.ops.gsplat_metal_fast`
- renders a tiny MPS scene through the Torch API
- runs backward and returns gradients for means, conics, colors, and opacities
- matches a direct Torch CPU reference for a 16x16 / 4-splat forward and backward check at about `1e-8` absolute error

A synthetic projected 4096x4096 / 65,536-splat smoke also ran locally:

- forward-only, warmed, 3 timed iterations: about `13.3 ms` mean forward
- forward + backward, warmed, 1 timed iteration: about `9.9 ms` forward and `99.6 ms` backward

Those timings use already-projected 2D splats with small random radii and should be treated as smoke-test numbers, not a full scene-quality benchmark.

After sorted tile IDs were saved for backward, synchronized direct-op smoke tests measured:

- 4096x4096 / 65,536 splats / sigma 1-5 px: forward `9.9 ms`, backward `31.4 ms`
- 4096x4096 / 65,536 splats / sigma 3-8 px: forward `15.5 ms`, backward `93.4 ms`
- 1024x1024 / 65,536 splats / sigma 1-5 px: forward `6.36 ms`, backward `30.0 ms`

## Variants

`variants/v3/` contains the newer Torch+Metal v3 handoff. It is kept side by
side with the validated v2 fastpath so we can benchmark and audit the algorithm
changes without losing the known-good baseline.

v3 changes the tile kernel shape to 256 threads, stages Gaussian parameters in
threadgroup memory, adds an overflow-tile fallback, and reduces backward global
atomics with tile-local SIMD/threadgroup reductions.

Build and validate v3 directly:

```bash
cd variants/v3
uv run python setup.py build_ext --inplace
uv run python tests/reference_check.py
```

Compare v2 and v3 from the repository root:

```bash
uv run python benchmarks/compare_v2_v3.py --height 4096 --width 4096 --gaussians 65536 --warmup 2 --iters 5
uv run python benchmarks/compare_v2_v3.py --height 4096 --width 4096 --gaussians 65536 --warmup 2 --iters 5 --backward
```

Latest local comparison:

| Case | v2 forward | v3 forward | v3 / v2 | v3 faster than v2 |
| --- | ---: | ---: | ---: | ---: |
| 4096x4096, 65,536 splats, sigma 1-5 px | `15.506 ms` | `12.410 ms` | `0.800x` | `+25%` |
| 4096x4096, 65,536 splats, sigma 3-8 px | `24.935 ms` | `13.702 ms` | `0.550x` | `+82%` |

| Case | v2 forward+backward | v3 forward+backward | v3 / v2 | v3 faster than v2 |
| --- | ---: | ---: | ---: | ---: |
| 4096x4096, 65,536 splats, sigma 1-5 px | `70.654 ms` | `47.872 ms` | `0.678x` | `+48%` |
| 4096x4096, 65,536 splats, sigma 3-8 px | `134.162 ms` | `60.738 ms` | `0.453x` | `+121%` |

See `docs/chief_scientist_field_report.md` for field notes on the v2 fixes,
backward bottleneck, and v3 status. See `docs/v3_saved_order_ablation.md` for
the saved-order backward ablation.

## Direct Torch vs Taichi vs fast-mac

The table below compares three Mac paths:

- **Direct Torch baseline**: simple depth-sorted Torch tensor math. Useful for
  correctness and tiny scenes, not a high-resolution renderer.
- **Taichi Mac**:
  [`taichi-gsplat-differentiable-render-mac`](https://github.com/nbardy/taichi-gsplat-differentiable-render-mac),
  useful when the surrounding renderer stack needs Taichi compatibility.
- **fast-mac**: this repo's pure Metal/Torch path. v2 is lower overhead on tiny
  scenes; v3 is the stronger large-scene/backward path.

These are local Apple Silicon synthetic projected-2D Gaussian timings from the
Dynaworld stack benchmark. `% faster` uses
`(baseline_ms / renderer_ms - 1) * 100`. `Best fast-mac` means the faster of v2
and v3 for that row. Taichi uses native batch for `B > 1`; v2/v3 are currently
single-image APIs looped over the batch.

Small and bootstrap-scale comparisons:

| Case | Direct Torch | Taichi | Best fast-mac | Taichi faster than Torch | Best faster than Torch | Best faster than Taichi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x64, B=4, G=128, sigma 1-5, forward | `126.618 ms` | `14.759 ms` | `12.861 ms` (v2) | `+758%` | `+885%` | `+15%` |
| 64x64, B=4, G=128, sigma 3-8, forward | `145.510 ms` | `17.993 ms` | `15.916 ms` (v2) | `+709%` | `+814%` | `+13%` |
| 64x64, B=4, G=128, sigma 1-5, fwd+bwd | `685.026 ms` | `35.858 ms` | `21.535 ms` (v2) | `+1810%` | `+3081%` | `+67%` |
| 64x64, B=4, G=128, sigma 3-8, fwd+bwd | `593.915 ms` | `36.881 ms` | `33.096 ms` (v2) | `+1510%` | `+1695%` | `+11%` |
| 128x128, B=4, G=128, sigma 1-5, forward | `83.627 ms` | `11.961 ms` | `11.050 ms` (v2) | `+599%` | `+657%` | `+8%` |
| 128x128, B=4, G=128, sigma 3-8, forward | `97.898 ms` | `16.517 ms` | `11.517 ms` (v2) | `+493%` | `+750%` | `+43%` |
| 128x128, B=4, G=128, sigma 1-5, fwd+bwd | `702.222 ms` | `47.335 ms` | `24.254 ms` (v2) | `+1384%` | `+2795%` | `+95%` |
| 128x128, B=4, G=128, sigma 3-8, fwd+bwd | `717.170 ms` | `66.641 ms` | `28.428 ms` (v2) | `+976%` | `+2423%` | `+134%` |
| 128x128, B=1, G=512, sigma 1-5, forward | `107.720 ms` | `18.254 ms` | `4.007 ms` (v2) | `+490%` | `+2588%` | `+356%` |
| 128x128, B=1, G=512, sigma 3-8, forward | `113.577 ms` | `14.365 ms` | `3.680 ms` (v2) | `+691%` | `+2986%` | `+290%` |
| 128x128, B=1, G=512, sigma 1-5, fwd+bwd | `612.363 ms` | `44.268 ms` | `8.578 ms` (v3) | `+1283%` | `+7039%` | `+416%` |
| 128x128, B=1, G=512, sigma 3-8, fwd+bwd | `636.652 ms` | `43.138 ms` | `6.236 ms` (v2) | `+1376%` | `+10109%` | `+592%` |

Large no-Torch stress comparison:

| Case | Taichi | v2 | v3 | Winner | Winner faster than Taichi |
| --- | ---: | ---: | ---: | --- | ---: |
| 1024x1024, B=1, G=65,536, sigma 1-5, forward | `54.323 ms` | `14.410 ms` | `9.717 ms` | v3 | `+459%` |
| 1024x1024, B=1, G=65,536, sigma 3-8, forward | `63.798 ms` | `16.919 ms` | `18.117 ms` | v2 | `+277%` |
| 1024x1024, B=1, G=65,536, sigma 1-5, fwd+bwd | `352.500 ms` | `41.148 ms` | `20.238 ms` | v3 | `+1642%` |
| 1024x1024, B=1, G=65,536, sigma 3-8, fwd+bwd | `1081.152 ms` | `119.657 ms` | `39.510 ms` | v3 | `+2636%` |

Taichi did not beat the fastest fast-mac variant in these measurements. It did
beat v3 alone in a few small batched cases, but v2 was faster in those same
rows. The practical split is: use v2 for low-res bootstrap and smoke tests, use
v3 for larger scenes and backward-heavy training, and use Taichi when Taichi
compatibility matters more than maximum raster throughput.

Direct Torch is skipped at large sizes because the direct reference performs
work proportional to `height * width * gaussians`. Dense vectorization would
materialize impractical activation volumes; the looped Torch reference avoids
that allocation but is not a meaningful renderer at 1024x1024 or 4K.

## Direct Torch Reference

`benchmarks/compare_v2_v3.py --include-torch-reference` can also time a direct
Torch reference renderer. This reference is intentionally simple: it loops over
depth-sorted splats and uses Torch tensor ops over the image for each splat. It
is useful as a correctness and small-scene speed baseline, but it is not a
viable 4K / 65,536-splat renderer.

The benchmark skips the Torch reference when `height * width * gaussians`
exceeds `--torch-max-work-items`. A 4096x4096 / 65,536-splat scene is about
`1.1e12` pixel-splat evaluations, so comparing that directly to Torch is not a
meaningful use of the baseline.

Local 128x128 / 512-splat comparison, `--warmup 1 --iters 3`:

| Case | v2 forward | v3 forward | Torch forward |
| --- | ---: | ---: | ---: |
| sigma 1-5 px | `4.654 ms` | `7.443 ms` | `163.244 ms` |
| sigma 3-8 px | `4.666 ms` | `6.114 ms` | `162.016 ms` |

| Case | v2 forward+backward | v3 forward+backward | Torch forward+backward |
| --- | ---: | ---: | ---: |
| sigma 1-5 px | `7.972 ms` | `8.940 ms` | `828.056 ms` |
| sigma 3-8 px | `10.850 ms` | `11.928 ms` | `866.864 ms` |

At this tiny scale v2 beats v3 because v3 pays extra staging/reduction overhead.
At 4K / 65,536 splats, v3 is the faster training path.
