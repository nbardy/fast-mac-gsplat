# torch-metal-gsplat-v9-project3d-train

Experimental fork of v5 that keeps the projected-2D rasterizer API and adds a
training-ready pinhole 3D projection API.

## APIs

- `rasterize_projected_gaussians(...)`: copied v5 projected-2D API.
- `project_pinhole_gaussians(...)`: Metal projection from 3D Gaussian params
  to `means2d/conics/opacities/depths`, with a per-splat Metal VJP for
  training.
- `rasterize_pinhole_gaussians(...)`: convenience path for
  `3D Gaussian + pinhole camera -> Metal projection -> Metal raster`.

The new projection path supports pinhole only:

```text
means3d/scales/quats/opacities/colors + fx/fy/cx/cy + camera_to_world
  -> projected 2D splat packet
  -> existing v5-style tile rasterizer
```

## Build

```bash
python setup.py build_ext --inplace
```

## Benchmark

From the Dynaworld repo root:

```bash
python src/benchmarks/fast_mac_project3d_benchmark.py --build-v9
```

Default cases include a small smoke case and a 128px / 8192-splat case.

## Training Backward

The backward path stays staged instead of becoming one giant raster/projection
kernel:

```text
raster backward -> dmeans2d/dconics/dopacity/dcolors
project backward -> dmeans3d/dscales/dquats/dopacities/dcamera
```

Projection backward is a one-thread-per-splat Metal VJP. Per-splat gradients
are written directly; camera/intrinsic gradients use batch-local atomics.

## Caveats

- Non-pinhole lens models are intentionally out of scope for this fork.
