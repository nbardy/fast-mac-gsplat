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
uv run python benchmarks/compare_v2_v3.py --height 4096 --width 4096 --gaussians 65536 --warmup 1 --iters 3
uv run python benchmarks/compare_v2_v3.py --height 4096 --width 4096 --gaussians 65536 --warmup 1 --iters 3 --backward
```

Forward only:

| Case | v2 fastpath | v3 candidate | v3 / v2 |
| --- | ---: | ---: | ---: |
| sparse sigma 1-5 px | `13.563 ms` | `8.805 ms` | `0.649x` |
| medium sigma 3-8 px | `19.964 ms` | `13.876 ms` | `0.695x` |

Forward + backward:

| Case | v2 fastpath | v3 candidate | v3 / v2 |
| --- | ---: | ---: | ---: |
| sparse sigma 1-5 px | `74.045 ms` | `57.253 ms` | `0.773x` |
| medium sigma 3-8 px | `137.559 ms` | `68.198 ms` | `0.496x` |

These are projected-2D synthetic raster hot-path tests, not full scene-training
benchmarks. They are still useful because both versions render the same inputs
through the Torch wrapper and synchronize MPS at timing boundaries.

## Immediate follow-up

v3 fast backward currently sorts tile-local IDs again. The validated v2 path
already proved that writing the forward sorted order back into `binned_ids` lets
backward skip that duplicate sort. Port that optimization into v3 before judging
the remaining backward cost.
