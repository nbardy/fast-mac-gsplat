# Engineering notes for v3

## What changed from the earlier fastpath

- uses 256-thread tile groups instead of 1024-thread groups
- compiles around a smaller shared-ID cap (`GSP_FAST_CAP = 2048`) to improve occupancy
- keeps the fast path exact for normal tiles and adds a slow overflow path instead of hard-failing
- stages Gaussian parameters into threadgroup memory in chunks to cut repeated device-memory reads
- keeps forward exact front-to-back alpha compositing
- keeps backward low-memory by recomputing alpha/transmittance instead of storing dense activations
- reduces backward global atomics by doing per-Gaussian reductions inside each tile threadgroup
- treats Torch `[G,3]` tensors as packed `float*`, not `float3*`

## Why keep Torch ops for sort / scan

The wrapper still uses Torch MPS for:

- stable global depth sort (`torch.argsort(stable=True)`)
- prefix sum (`torch.cumsum`)
- rare overflow-tile ID sorting

That is deliberate. It keeps the hot raster kernels in Metal while using robust library primitives for global sort/scan.

## Expected fast path

1. Python sorts by detached depth.
2. Metal counts tile intersections with SnugBox + exact rectangle/ellipse test.
3. Torch computes tile offsets.
4. Metal emits compact tile bins of sorted-rank IDs.
5. Fast tiles local-sort in threadgroup memory and render.
6. Overflow tiles go through the slower overflow path.
7. Backward recomputes and does Gaussian-centric gradient reduction.

## Caveats

- still a projected 2D renderer, not a full 3D projection frontend
- still depends on internal PyTorch MPS Metal headers
- depth gradients remain zero because order is treated as piecewise constant
- this source bundle should be compiled and profiled on the target Mac before calling it production-ready

## Local validation after import

The extracted bundle needed one packaging patch before it could build in-place:
`setup.py` now passes extension source paths relative to `variants/v3/` while
keeping the include directory absolute.

Validation run:

```bash
cd variants/v3
uv run python setup.py build_ext --inplace
uv run python tests/reference_check.py
```

The reference check passed:

- image max error: `5.960464477539063e-08`
- means grad max error: `2.4010660126805305e-10`
- conics grad max error: `9.313225746154785e-10`
- colors grad max error: `9.313225746154785e-10`
- opacities grad max error: `1.862645149230957e-09`

Small v2/v3 parity check at 128x128 / 512 splats produced max image diff `0.0`.

## Local 4K benchmark comparison

Run from the repository root:

```bash
uv run python benchmarks/compare_v2_v3.py --height 4096 --width 4096 --gaussians 65536 --warmup 2 --iters 5
uv run python benchmarks/compare_v2_v3.py --height 4096 --width 4096 --gaussians 65536 --warmup 2 --iters 5 --backward
```

Forward only:

| Case | v2 fastpath | v3 candidate | v3 / v2 |
| --- | ---: | ---: | ---: |
| sparse sigma 1-5 px | `15.506 ms` | `12.410 ms` | `0.800x` |
| medium sigma 3-8 px | `24.935 ms` | `13.702 ms` | `0.550x` |

Forward + backward:

| Case | v2 fastpath | v3 candidate | v3 / v2 |
| --- | ---: | ---: | ---: |
| sparse sigma 1-5 px | `70.654 ms` | `47.872 ms` | `0.678x` |
| medium sigma 3-8 px | `134.162 ms` | `60.738 ms` | `0.453x` |

These are projected-2D synthetic raster hot-path tests, not full scene-training
benchmarks. They are still useful because both versions render the same inputs
through the Torch wrapper and synchronize MPS at timing boundaries.

## Saved-order ablation

Fast forward now writes the sorted tile-local ID order back into `binned_ids`.
Fast backward reuses that order and skips its duplicate local bitonic sort.

The ablation at `../../docs/v3_saved_order_ablation.md` measured:

- no unconditional `grad_out` clone: small/noisy, `+0.8%` sparse and `-2.8%`
  medium forward+backward
- saved sorted IDs on top: `-9.5%` sparse and `-4.9%` medium
  forward+backward versus clone-only
- both changes together: `-8.7%` sparse and `-7.5%` medium
  forward+backward versus the v3 baseline measured in the same session
