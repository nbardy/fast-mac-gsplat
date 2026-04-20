# Field Report for the Torch+Metal Fast Path

Audience: chief scientist / next shader author.

## Short version

The previous Torch+Metal fastpath is now a real working baseline, not just a
source handoff. We fixed packaging, custom-op registration, a critical Metal
layout bug, and backward ordering overhead. It builds locally, imports as a
Torch extension, renders on MPS tensors, runs backward, and matches a small CPU
reference at roughly `1e-8` absolute error.

The main field lesson is that backward is not slow for one single reason. The
duplicate local sort mattered and was worth fixing, but the remaining cost is
mostly the recompute reverse scan plus heavy global atomic accumulation under
overlap. In sparse projected cases backward can get into the roughly 3x-forward
range. In denser tile-overlap cases it can still be 6x or more.

The v3 handoff is directionally right: it keeps the layout fix, moves to
256-thread tile groups, stages Gaussian parameters in threadgroup memory, and
uses tile-local reductions before global atomic writes. In local synthetic
4096x4096 / 65,536-splat tests, v3 was faster than the validated v2 fastpath in
both forward and forward+backward.

## What had to be fixed in the previous shader

### Build and registration

The extension was close, but not drop-in. We had to make the setup/build path
usable from the checked-out repo and align the C++ custom-op schemas with the
actual C++ return types. Torch custom-op schemas declared as fixed tuple returns
must return fixed C++ tuples, not `std::vector<torch::Tensor>`, or import aborts
with a schema mismatch.

This was not a conceptual shader problem, but it blocked all real testing.

### Torch `[G,3]` layout vs Metal `float3`

The biggest correctness bug was treating tightly packed Torch `[G,3]` tensors as
`device float3*` in Metal. Torch stores rows as 3 contiguous `float32` values
with 12-byte row stride. Metal `float3` is aligned like a 16-byte vector. That
means `float3*` indexing reads index 0 correctly and corrupts later splats.

The durable fix is to treat `conics` and `colors` as flat `device float*` and
load triples manually with `idx * 3 + channel`. Both the validated v2 path and
the v3 candidate now do this.

### Tile bounds and pixel convention

The tile counter needed to be consistent about pixel centers. The working path
uses `x + 0.5, y + 0.5` sampling and SnugBox bounds expressed in that same
coordinate convention. This matters because tile counts drive allocation size;
an off-by-one in support bounds is both a correctness problem and a performance
problem.

### Backward repeated the local sort

The first working backward pass loaded each tile bin and sorted again. That was
correct but wasteful. We changed forward to write tile-local IDs back into
`binned_ids` in sorted order, then backward reuses that saved sorted order.

That improvement moved the validated v2 path from "forward is great, backward
is brutal" toward a usable training kernel. It did not fully solve backward.

## What the field timings say

These are projected-2D synthetic splat timings on local Apple Silicon. They are
not full 3D scene-quality benchmarks, but they are useful stress tests for the
raster hot path.

Validated v2 after saving sorted tile IDs:

| Case | Forward | Backward | Backward / forward |
| --- | ---: | ---: | ---: |
| 4096x4096, 65,536 splats, sigma 1-5 px | `9.9 ms` | `31.4 ms` | `3.2x` |
| 4096x4096, 65,536 splats, sigma 3-8 px | `15.5 ms` | `93.4 ms` | `6.0x` |
| 1024x1024, 65,536 splats, sigma 1-5 px | `6.36 ms` | `30.0 ms` | `4.7x` |

The new v3 candidate compared against v2 through
`benchmarks/compare_v2_v3.py`, same synthetic inputs, wrapper-level timings:

| Case | v2 forward | v3 forward | v3 / v2 |
| --- | ---: | ---: | ---: |
| 4096x4096, 65,536 splats, sigma 1-5 px | `13.563 ms` | `8.805 ms` | `0.649x` |
| 4096x4096, 65,536 splats, sigma 3-8 px | `19.964 ms` | `13.876 ms` | `0.695x` |

| Case | v2 forward+backward | v3 forward+backward | v3 / v2 |
| --- | ---: | ---: | ---: |
| 4096x4096, 65,536 splats, sigma 1-5 px | `74.045 ms` | `57.253 ms` | `0.773x` |
| 4096x4096, 65,536 splats, sigma 3-8 px | `137.559 ms` | `68.198 ms` | `0.496x` |

v3 also matched v2 output on a small 128x128 / 512-splat comparison with max
image difference `0.0`.

## Current backward model

Observed fact: backward remains much slower than forward when tile overlap rises.

Current belief: the dominant costs are:

1. recomputing the alpha/transmittance chain per tile pixel,
2. reverse-scanning the same local list for gradients,
3. global atomic accumulation into per-Gaussian gradients,
4. repeated parameter loads when many pixels in a tile touch the same splats.

The "duplicate sort" branch is weakened, not invalidated. It was real in v2 and
worth fixing. But v3 currently still bitonic-sorts IDs in fast backward and
still wins in the tested cases, because its parameter staging and tile-local
SIMD/threadgroup reduction reduce bandwidth and global atomic pressure.

Falsification test: save the v3 forward sorted order the way v2 now does, remove
the v3 backward bitonic sort, and rerun the 4K / 64K sparse and medium cases. If
that moves only a few percent, atomics/recompute remain dominant. If it moves
20%+, sorting still deserves more attention.

## About 16x16 tiles

The 16x16 specialization is not obviously a flaw. It maps exactly to 256 pixels,
which matches v3's 256-thread group design: one thread per tile pixel. That is a
reasonable Apple-GPU shape and simpler than supporting arbitrary runtime tile
sizes.

It is not a free hyperparameter today. Changing it means changing shader compile
constants, threadgroup memory sizing, the local sort/reduction structure, and
the wrapper validation. The right next step is not "make tile size runtime"; it
is to benchmark a small number of compiled variants, probably 8x8, 16x16, and
32x32, against real projected splat distributions.

## Tensor/device contract

The fast kernels need MPS tensors. CPU tensors are useful for small reference
checks, but the Metal custom ops operate through PyTorch's MPS/Metal path. The
Python wrapper keeps autograd in Torch and dispatches the hot path through the
extension.

## v3 status after integration

We extracted the v3 bundle into `variants/v3/` and kept the original tarball in
`source_artifacts/torch_metal_gsplat_v3.tar.gz`.

Local validation:

- patched `variants/v3/setup.py` so extension sources are relative to setup.py
- built v3 with `uv run python setup.py build_ext --inplace`
- ran `uv run python tests/reference_check.py`
- result: forward and gradients matched the CPU reference around `1e-9` to
  `1e-8`
- ran v2/v3 side-by-side 4K / 64K benchmark comparisons

Remaining caveats:

- v3 is still a projected-2D rasterizer, not a full 3D camera projection stack
- overflow fallback exists but was not stress-tested with deliberately
  pathological tile overflows
- all speed numbers above are synthetic raster hot-path numbers, not training
  throughput numbers

## Postscript: saved-order patch

We ported the v2 saved-sorted-order trick into v3 after the first field report:
fast forward now writes the tile-local sorted ID order back into `binned_ids`,
and fast backward skips its duplicate local bitonic sort.

In the same 4096x4096 / 65,536-splat synthetic cases, with `--warmup 2 --iters
5`, v3 forward+backward moved:

| Case | v3 before | no grad clone only | no clone + saved IDs |
| --- | ---: | ---: | ---: |
| sigma 1-5 px | `52.460 ms` | `52.904 ms` | `47.872 ms` |
| sigma 3-8 px | `65.687 ms` | `63.878 ms` | `60.738 ms` |

The no-clone change is small/noisy. The saved-ID change is the real win:
`-9.5%` sparse and `-4.9%` medium versus clone-only, or `-8.7%` sparse and
`-7.5%` medium versus the v3 baseline from the same session.

Correctness stayed intact: the v3 reference check still matches around `1e-9`
to `1e-8`, and a small v2/v3 MPS image parity check stayed at max diff `0.0`.

## Recommended next kernel pass

1. Add counters to report max tile count, mean tile count, total tile pairs, and
   overflow tile count per benchmark case.
2. Stress-test overflow fallback with a dense center-cluster scene.
3. Run 8x8 / 16x16 / 32x32 compiled variants only after the counters show that
   tile occupancy, not atomics, is the next dominant issue.
4. Keep the low-memory recompute backward unless training memory says otherwise;
   saving dense per-pixel activations would likely trade the current bottleneck
   for an immediate memory wall.
