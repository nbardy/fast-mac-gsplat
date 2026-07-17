# spacetime_v0

Gate 0 starter scaffold for **spacetime Gaussian rasterization** — a Metal
renderer for 4D world-space Gaussian primitives with a sensor-spacetime
tile-time index.

This variant is not buildable yet. It collects the handoff docs and the
starter source files in one place so the implementation can begin.

## Lineage and relationship to `star_uvt_v0`

`star_uvt_v0` is the screen-time tube (UVT) Gate 0 renderer:

- input is already-projected `ScreenTimeTube` packets in `(u, v, t)`;
- no world/camera projection inside the kernel;
- tile-time bin in `(u, v, t)` then composite.

`spacetime_v0` is a **superset**:

- input is 4D world-space Gaussians `{mu: float4, Q: float4x4, color, alpha}`;
- **adds pass 0**: project each 4D Gaussian into a 3D sensor-time Gaussian
  by integrating along ray depth `s` (closed-form, see `docs/handoff.md`
  section 3);
- reuses the same tile-time `(tile_x, tile_y, time_bin)` index, count/fill,
  and render passes that `star_uvt_v0` already prototypes.

The two variants will eventually share most of passes 1-5. Pass 0 (world
projection) is the new GPU work.

## Files

```text
docs/handoff.md
  Research engineer handoff. 4D Gaussian representation, camera lift,
  closed-form ray-depth integration to a 3D sensor-time footprint,
  tile-time AABB, GPU pipeline, milestones, acceptance tests, risks.

reference/st_raster_ref.py
  CPU reference implementation. Orthographic camera, project_gaussian_affine,
  build_tile_time_index, both compositing modes, demo with two
  constant-velocity 4D Gaussians. Run with `python st_raster_ref.py`;
  writes frames to `./st_out` if Pillow is available.

csrc/metal/st_gaussian_raster.metal
  Starter Metal kernels: clear_uints, count_cell_overlaps,
  fill_cell_overlaps, render_order_independent, render_sorted_alpha.
  ProjectedGaussian and RasterConfig structs match the host skeleton.
  Pass 0 (project 4D -> 3D sensor-time) is NOT implemented here yet;
  these kernels assume projected input.

host_swift/STMetalHostSkeleton.swift
  Swift host-side build guide. Sets up command queue, pipeline states,
  buffers, CPU prefix-scan prototype, dispatches count/fill/render.
  Not a drop-in finished renderer; use as a structural reference when
  wiring into the project.
```

## Milestone plan (mirrors `docs/handoff.md` section 9)

| Milestone | Scope |
|---|---|
| A: CPU reference | Orthographic projection + tile-time bin + OI compositing. Already covered by `reference/st_raster_ref.py`. |
| B: Metal prototype | Upload CPU-projected gaussians, GPU count/fill/render, CPU prefix scan. `csrc/metal/` covers the kernels; `host_swift/` covers structure. |
| C: Full GPU indexing | GPU prefix scan, per-cell depth sort or bucket compositing, overflow handling. |
| D: Perspective + local affine | Per-macro-tile projection (this is the actual pass 0 GPU kernel), exact fallback for high-curvature tiles. |
| E: Training integration | Differentiable approximation, per-pixel primitive IDs/weights, gradient kernels for `mu`, `Q`, `logAmp`, color. |

## Acceptance smoke targets (math)

From `docs/handoff.md` section 8:

1. A constant-velocity 3D Gaussian equals one tilted 4D Gaussian
   (`reference/st_raster_ref.make_constant_velocity_gaussian` constructs
   this; render parity against equivalent 3D path is the test).
2. Projected footprint of an orthographic primitive matches numerical
   line integration along the ray-depth axis.
3. Integrated mass is invariant under shifting the ray parameter.
4. Sensor-time AABB contains all samples with Mahalanobis `d^2 <= R^2`.

## Status

Not started. Files were dropped here as starter material on 2026-05-12.
