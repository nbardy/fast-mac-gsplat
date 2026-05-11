# Phase 2: WorldTube To ScreenTimeTube

## Goal

Introduce the real STAR-GS normal form while keeping the renderer contract
unchanged.

## Representation

```text
x(t) = x0 + v * (t - t0)
opacity_time = exp(-0.5 * lambda_t * (t - t0)^2)
spatial_precision = A
```

The 4D precision block implied by the tube is:

```text
Lambda_xx = A
Lambda_xt = -A * v
Lambda_tt = dot(v, A * v) + lambda_t
```

## Gate

World-space tubes must project to the same `ma`, `q_uvt`, `depth0`,
`depth_beta`, `opacity`, and `color` tensors already accepted by Gate 0.

## Current Slice

Phase 2a implements an orthographic/fronto-parallel projection scaffold:

```bash
python3 research_project/trainer_harness/world_projection_smoke.py
```

This is intentionally weaker than the final gate. It proves the tensor contract
from `WorldTubeBatch` to `ScreenTimeTube`.

Phase 2b implements a pinhole-camera projection scaffold:

```bash
python3 research_project/trainer_harness/pinhole_projection_smoke.py
```

It uses `world_to_camera`, `fx`, `fy`, `cx`, and `cy`, then linearizes the
fronto-parallel tube plane into screen-time center, velocity, precision, and
depth tensors. It still does not cover full anisotropic 3D covariance
integration or typed GFlow `CameraSpec` plumbing.

Phase 2c adds a Dynaworld `CameraSpec` adapter smoke:

```bash
python3 research_project/trainer_harness/camera_spec_projection_smoke.py
```

This covers the current `dynaworld/src/train/camera.py` pinhole `CameraSpec`
shape and convention, but not distorted camera models.
