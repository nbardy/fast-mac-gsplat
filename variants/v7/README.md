# torch-metal-gsplat-v7-hardware

Experimental **Torch custom-op + Metal graphics/compute** handoff for a hardware-forward Gaussian rasterizer.

What this package tries to do:
- keep the **Torch ABI** and autograd boundary,
- move the **forward** path into a real Metal **render pipeline**,
- keep the **backward** path in Metal **compute** so the gradient math stays custom,
- use projected 2D Gaussian inputs `[G,2/3]` or `[B,G,2/3]`.

This is intentionally a **clean-sheet v7 source handoff**. It is not derived from the v2/v3/v5 compute-only renderer implementation.

## Forward path

1. Torch sorts projected splats by depth.
2. Objective-C++ creates a Metal render pass and draws **instanced ellipse impostor quads**.
3. Vertex shader expands a screen-space quad from the projected Gaussian's snug bounding box.
4. Fragment shader evaluates the exact Gaussian alpha footprint.
5. Standard alpha blending composites the image.

## Training path

Training currently uses the same hardware-forward image path and a compute replay backward over the sorted projected splats.
This keeps the Torch ABI simple while the forward gets closer to true hardware graphics support.

## Important status note

This package now builds and runs on the local Apple Silicon toolchain. The
initial handoff needed several engineering fixes:

- add direct-script import fixes and the missing reference-test entrypoint
- remove a duplicate Metal `[[position]]` fragment argument that blocked shader compilation
- draw sorted splats far-to-near for hardware source-over blending
- expand quad bounds so edge pixels match the dense reference
- fix backward gradient formulas for color and means

After those fixes, `tests/reference_check.py` matches the small dense reference:

```text
image max error: 5.960464477539063e-08
means grad max error: 2.473825588822365e-10
conics grad max error: 9.313225746154785e-10
colors grad max error: 9.313225746154785e-10
opacities grad max error: 1.862645149230957e-09
```

The architecture is still experimental. v7 currently copies MPS tensors through
CPU/shared buffers and reads the render texture back to Torch, so it is not a
drop-in replacement for the compute renderers yet.

## Files
- `torch_gsplat_bridge_v7/rasterize.py` — Python API / autograd wrapper
- `csrc/bindings.cpp` — Torch op registration
- `csrc/metal/gsplat_hardware.mm` — ObjC++ bridge
- `csrc/metal/gsplat_v7_hardware.metal` — Metal vertex/fragment/compute shaders
- `csrc/shared/common.h` — shared ABI definitions

## Suggested next steps on a Mac
1. Profile forward-only at larger resolutions where hardware blending may matter.
2. Remove CPU round trips by wiring MPS/Metal resources directly.
3. Revisit backward; the current replay path remains compute/atomic-heavy.
4. Compare against v6 direct with `../../benchmarks/benchmark_full_matrix.py`.
