# fast-mac-gsplat

Torch-first, Metal-backed Gaussian rasterizer fast path for Apple Silicon.

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
uv run python benchmarks/compare_v2_v3.py --height 4096 --width 4096 --gaussians 65536 --warmup 1 --iters 3
uv run python benchmarks/compare_v2_v3.py --height 4096 --width 4096 --gaussians 65536 --warmup 1 --iters 3 --backward
```

Latest local comparison:

| Case | v2 forward | v3 forward | v3 / v2 |
| --- | ---: | ---: | ---: |
| 4096x4096, 65,536 splats, sigma 1-5 px | `13.563 ms` | `8.805 ms` | `0.649x` |
| 4096x4096, 65,536 splats, sigma 3-8 px | `19.964 ms` | `13.876 ms` | `0.695x` |

| Case | v2 forward+backward | v3 forward+backward | v3 / v2 |
| --- | ---: | ---: | ---: |
| 4096x4096, 65,536 splats, sigma 1-5 px | `74.045 ms` | `57.253 ms` | `0.773x` |
| 4096x4096, 65,536 splats, sigma 3-8 px | `137.559 ms` | `68.198 ms` | `0.496x` |

See `docs/chief_scientist_field_report.md` for field notes on the v2 fixes,
backward bottleneck, and v3 status.
