# torch-metal-gsplat-v8-project3d

Experimental fork of v5 that keeps the projected-2D rasterizer API and adds a
forward-only pinhole 3D projection API.

## APIs

- `rasterize_projected_gaussians(...)`: copied v5 projected-2D API.
- `project_pinhole_gaussians(...)`: Metal forward projection from 3D Gaussian
  params to `means2d/conics/opacities/depths`.
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
python src/benchmarks/fast_mac_project3d_benchmark.py --build-v8
```

Default cases include a small smoke case and a 128px / 8192-splat case.

## Caveats

- 3D projection backward is not implemented.
- Raster backward is still the copied v5 projected-2D backward. Gradients can
  flow to `colors` through `rasterize_pinhole_gaussians`, but not back through
  `means3d`, `scales`, `quats`, `opacities`, or camera parameters.
- Non-pinhole lens models are intentionally out of scope for this fork.
