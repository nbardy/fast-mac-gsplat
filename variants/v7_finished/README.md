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

This package is the **best source handoff I can write here**, but it still needs real Apple-machine validation.
The render-pipeline code is substantially more concrete than the old Torch/Metal scaffold, but the PyTorch MPS tensor <-> Metal texture/buffer bridge is still the main integration risk.

## Files
- `torch_gsplat_bridge_v7/rasterize.py` — Python API / autograd wrapper
- `csrc/bindings.cpp` — Torch op registration
- `csrc/metal/gsplat_hardware.mm` — ObjC++ bridge
- `csrc/metal/gsplat_v7_hardware.metal` — Metal vertex/fragment/compute shaders
- `csrc/shared/common.h` — shared ABI definitions

## Suggested next steps on a Mac
1. Build the extension.
2. Validate B=1 forward parity against v3 or a dense CPU reference.
3. Validate small backward finite-difference checks.
4. Only then profile whether the hardware-forward path wins enough to justify deeper investment.
