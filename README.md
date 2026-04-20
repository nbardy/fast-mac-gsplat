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

| Case | v2 forward | v3 forward | v3 / v2 |
| --- | ---: | ---: | ---: |
| 4096x4096, 65,536 splats, sigma 1-5 px | `15.506 ms` | `12.410 ms` | `0.800x` |
| 4096x4096, 65,536 splats, sigma 3-8 px | `24.935 ms` | `13.702 ms` | `0.550x` |

| Case | v2 forward+backward | v3 forward+backward | v3 / v2 |
| --- | ---: | ---: | ---: |
| 4096x4096, 65,536 splats, sigma 1-5 px | `70.654 ms` | `47.872 ms` | `0.678x` |
| 4096x4096, 65,536 splats, sigma 3-8 px | `134.162 ms` | `60.738 ms` | `0.453x` |

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
- **fast-mac v3**: this repo's recommended pure Metal/Torch fast path.

These are local Apple Silicon synthetic projected-2D Gaussian timings. The
1024x1024 rows use identical sparse sigma 1-5 px input for Taichi and fast-mac.
The 4K rows use `benchmarks/compare_v2_v3.py --height 4096 --width 4096
--gaussians 65536 --warmup 2 --iters 5`.

Forward:

| Case | Direct Torch baseline | Taichi Mac | fast-mac v3 | Notes |
| --- | ---: | ---: | ---: | --- |
| 128x128, 512 splats, sigma 1-5 px | `163.244 ms` | `9.812 ms` | `7.443 ms` | Torch baseline is feasible here. |
| 1024x1024, 65,536 splats, sigma 1-5 px | skipped | `27.630 ms` | `7.519 ms` | About `6.9e10` Torch pixel-splat evals. |
| 4096x4096, 65,536 splats, sigma 1-5 px | skipped | unsupported at 16x16 tiles | `12.410 ms` | About `1.1e12` Torch pixel-splat evals. |
| 4096x4096, 65,536 splats, sigma 3-8 px | skipped | unsupported at 16x16 tiles | `13.702 ms` | Medium-radius stress case. |

Forward + backward:

| Case | Direct Torch baseline | Taichi Mac | fast-mac v3 | Notes |
| --- | ---: | ---: | ---: | --- |
| 128x128, 512 splats, sigma 1-5 px | `828.056 ms` | `18.400 ms` | `8.940 ms` | Torch baseline is feasible here. |
| 1024x1024, 65,536 splats, sigma 1-5 px | skipped | `284.805 ms` | `17.704 ms` | fast-mac is about `16.1x` faster than Taichi here. |
| 4096x4096, 65,536 splats, sigma 1-5 px | skipped | unsupported at 16x16 tiles | `47.872 ms` | Recommended training-scale path. |
| 4096x4096, 65,536 splats, sigma 3-8 px | skipped | unsupported at 16x16 tiles | `60.738 ms` | Medium-radius stress case. |

Direct Torch is skipped at large sizes because the direct reference performs
work proportional to `height * width * gaussians`. Dense vectorization would
materialize impractical activation volumes; the looped Torch reference avoids
that allocation but is not a meaningful renderer at 1024x1024 or 4K.

The Taichi 4K rows are marked unsupported for the current 16x16-tile path
because 4096x4096 produces 65,536 tiles and the fork currently packs the tile ID
into a 16-bit key range. Running 4K through Taichi would need a wider tile key
or a different tile size, which is a separate benchmark regime.

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
