# star_uvt_prt_v0

Focused fork for STAR-UVT moving-camera support. This is still the STAR-UVT
sublinear rasterization lane: the only change is that moving camera paths compile
world tubes into a stronger sensor-time footprint instead of forcing affine
`ma/q_uvt` or falling back to per-frame projection.

Source notes:

```text
research_experiments/camera_movement_aware_worldtubes.md
research_experiments/camera_movement_aware_worldtubes_more_notes.md
research_experiments/camera_movement_aware_worldtubes_final.md
```

The goal is to keep learned state as world-space tubes, then compile each camera path into projective rational sensor-time tubes:

```text
world tube + camera path -> h_coeff(t) -> STAR-UVT rasterizer
```

This is not a production renderer yet. The first milestone is a Python
projection compiler and audit that proves the moving-camera center trajectory
before touching Metal.

## Direction Routing

The two original directions stay in scope, but with different jobs:

```text
Direction 1: world spacetime object + camera-ray pullback/compiler
  immediate implementation here as Projective Rational Tubes v0.

Direction 2: gauge atlas primitives
  later generalization after PRT proves the moving-camera compiler path.
```

Do not turn this into a learned camera-locked screen-tube cache. `h_coeff` is a
compiled render-time artifact from world state plus a camera path, not persistent
learned scene state.

## Current Files

```text
research_project/trainer_harness/projective_rational.py
  Camera path polynomial fit, world-tube to homogeneous image-curve compiler,
  direct projection audit helpers, and dense PRT reference renderer.

research_project/benchmarks/projective_rational_projection_audit.py
  Gate A-style audit: static camera parity, forward camera motion curvature,
  zoom/intrinsics motion, camera-cut residual detection, dense render finite
  smoke, and a curvature-selective hybrid diagnostic.

research_project/benchmarks/projective_rational_projection_scaling_probe.py
  Projection-only scaling probe across frame counts. Gates on moving-camera
  projection accuracy and affine-break detection; CPU timings are diagnostic,
  not a Metal rasterizer speed claim.

csrc/ and torch_gsplat_bridge_star_uvt_prt/
  Forked STAR-UVT extension namespace. The current PRT additions are a
  forward-only direct Metal reference op and a first tiled PRT bin-and-render
  path.

research_project/PROGRESS.md
  Compact gate checklist and current status.

tests/projective_rational_gate_check.py
  Lightweight gate wrapper for the audit.

tests/projective_rational_scaling_check.py
  Lightweight gate wrapper for the projection scaling probe.

tests/projective_rational_direct_render_check.py
  Dense-vs-CPU-vs-Metal parity check for the direct PRT forward op.

tests/projective_rational_tiled_render_check.py
  Dense-vs-tiled-Metal parity check for the first tiled PRT forward path.

research_project/benchmarks/projective_rational_metal_forward_timing_probe.py
  Diagnostic Metal timing and tile-load probe for direct PRT vs tiled PRT.
```

## New Idea Added In This Fork

Use a **curvature-selective hybrid compiler** as a bridge between current affine
UVT and full projective rational tubes:

```text
if affine center residual <= threshold:
    emit old affine UVT tube
else:
    emit projective rational tube
```

This keeps the cheap current raster path for low-curvature tubes and spends the
new PRT path only where moving-camera perspective actually breaks affine UVT.
The audit logs this as `curvature_selective`.

Next idea to test after the first Metal forward: **residual-certified footprint
inflation**. When the camera polynomial fit is close but not exact, estimate a
per-window pixel residual from a small exact-projection probe and inflate the
support bound just enough to cover it. This gives a measured blur/speed tradeoff
before splitting a camera window.

## Run

From this directory:

```bash
python3 research_project/benchmarks/projective_rational_projection_audit.py
python3 research_project/benchmarks/projective_rational_projection_audit.py --out-json research_project/benchmarks/results/projective_rational_projection_audit.json
python3 research_project/benchmarks/projective_rational_projection_scaling_probe.py --out-json research_project/benchmarks/results/projective_rational_projection_scaling_probe.json
python3 tests/projective_rational_gate_check.py
python3 tests/projective_rational_scaling_check.py
python3 setup.py build_ext --inplace
python3 tests/projective_rational_direct_render_check.py --out-json research_project/benchmarks/results/projective_rational_direct_render_check.json
python3 tests/projective_rational_tiled_render_check.py --out-json research_project/benchmarks/results/projective_rational_tiled_render_check.json
python3 research_project/benchmarks/projective_rational_metal_forward_timing_probe.py --out-json research_project/benchmarks/results/projective_rational_metal_forward_timing_probe.json
STAR_UVT_TILE_CAPACITY=256 python3 research_project/benchmarks/projective_rational_metal_forward_timing_probe.py --tube-counts 256 --tile-capacity 256 --out-json research_project/benchmarks/results/projective_rational_metal_forward_timing_probe_cap256_256t.json
```

Expected first gate:

```text
static_affine_parity.pass = true
moving_camera_projective.pass = true
zoom_projective.pass = true
camera_cut_detection.pass = true
dense_render_smoke.pass = true
curvature_selective.pass = true
projective_rational_projection_scaling_probe.pass = true
projective_rational_direct_render_check.pass = true
projective_rational_tiled_render_check.pass = true
projective_rational_metal_forward_timing_probe.pass = true
projective_rational_metal_forward_timing_probe_cap256_256t.pass = true
```
