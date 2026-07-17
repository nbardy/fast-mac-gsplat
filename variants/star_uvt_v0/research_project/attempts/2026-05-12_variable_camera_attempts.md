# STAR-UVT Variable-Camera Attempts - 2026-05-12

## Reason

The UVT rasterizer does not break under a moving camera. It accepts
screen-time quadratics and does not care whether the motion came from object
velocity, camera velocity, intrinsics, or a future camera gauge.

The break is the current projection normal form. Today the sequence path emits
one affine/quadratic screen-time tube from one fixed `K` and one fixed `w2c`.
For tensors with a time dimension, the current multicam path selects
`K[view, 0]` and `w2c[view, 0]` for the whole sequence. The measured
`per_frame_loop` is only a negative control because it calls the one-frame
projection/render path `F` times and destroys the sequence-level amortization.

## Attempt 1: Dynamic First-Order UVT Projection

Add the missing camera-motion derivative while keeping one projected tube per
world tube:

```text
WorldTubeBatch [N], K(tau), dK/dt, w2c(tau), d(w2c)/dt
  -> ma [N,3], q_uvt [N,6], depth0 [N], depth_beta [N,3]
```

At chart time `tau`:

```text
x_tau = x0 + velocity * (tau - t0)
y_tau = R_tau x_tau + T_tau
y_dot = R_tau velocity + R_dot x_tau + T_dot
```

For pinhole projection:

```text
u_dot =
    fx_dot * y_x / y_z
  + fx * (y_dot_x * y_z - y_x * y_dot_z) / y_z^2
  + cx_dot

v_dot =
    fy_dot * y_y / y_z
  + fy * (y_dot_y * y_z - y_y * y_dot_z) / y_z^2
  + cy_dot
```

Then pack the existing UVT normal form:

```text
q_uvt =
  (lambda_u,
   lambda_uv,
   -(lambda_u * u_dot + lambda_uv * v_dot),
   lambda_v,
   -(lambda_uv * u_dot + lambda_v * v_dot),
   lambda_t + lambda_u*u_dot^2 + 2*lambda_uv*u_dot*v_dot + lambda_v*v_dot^2)
```

Depth uses:

```text
depth0 = y_z
depth_beta_t = y_dot_z
```

Expected behavior:

- With zero camera derivatives, this reduces to the current fixed-camera
  pinhole projection.
- It is the fastest possible moving-camera smoke because it does not increase
  primitive count.
- It will fail on wide temporal support plus strong camera curvature because a
  single affine UVT chart cannot represent curved screen-time motion.

## Attempt 2: Piecewise Camera-Time UVT Segments

Use the production-shaped representation: flatten each world tube into a small
number of local camera-time UVT charts.

```text
WorldTubeBatch [N], K_seq [T,3,3], w2c_seq [T,4,4]
  -> ProjectedTubeSegments [M]

ProjectedTubeSegments:
  ma           [M,3]
  q_uvt        [M,6]
  depth0       [M]
  depth_beta   [M,3]
  opacity      [M]
  color        [M,3]
  parent_id    [M]
  t_minmax     [M,2]
```

The first implementation can use fixed chunk sizes, for example
`frames_per_segment=1,2,4,8,full`, with midpoint camera and finite-difference
camera motion. Adaptive splitting can come after we have aligned speed and
quality numbers.

Important limitation: the current Metal renderer does not hard-clamp tube
validity with `t_minmax`. Until shader support exists, each segment must either
have sufficiently local temporal support or use opacity/time weighting to avoid
duplicated energy. Benchmark reports must state this explicitly.

Expected behavior:

- `frames_per_segment=full` should be close to Attempt 1 or current static
  STAR for static cameras.
- `frames_per_segment=1` approaches the negative-control per-frame projection
  shape and should be treated as the upper-cost bound.
- Useful production behavior is in the middle: `M` grows much slower than
  `N * frame_count` for smooth camera paths.

## Attempt 3: Projective Camera-Time Gauge

This is now implemented as the newest/final first-order path. It is the cleaner
theory and should be listed first in current comparisons:

```text
h(t) = P(t) X(t)
p(t) = (h_x / h_z, h_y / h_z)
h_dot = P_dot X + P X_dot
```

It makes `K(t)` and `w2c(t)` native, and the UVT chart comes from
`h_0, h_dot_0`. For pinhole cameras it should numerically match Attempt 1,
but it is the better normal form to keep as the final code path because the
camera and intrinsics motion enter through one homogeneous derivative.

## Aligned Benchmark Matrix

The benchmark needs the same resolution, frame count, optimizer steps, tube
count, and direct backward mode across rows.

Initial fixed-step matrix:

```text
target_size: 128
frames:      8, 16, 32
steps:       8 measured + 2 warmup
device:      mps when available
backward:    direct_atomic
rows:
  star_static_view
  star_dynamic_first_order
  star_segmented_frames_per_segment_4
  star_segmented_frames_per_segment_1
  star_per_frame_loop_negative_control
  dynamic_gsplats_view_sequence_baseline
```

Scale row:

```text
target_size: 256
frames:      32 if memory allows
```

Quality smoke:

```text
target_size: 128
frames:      8
steps:       100 fixed steps
metrics:     train PSNR, heldout PSNR, train loop wall time, eval render time
rows:
  star_static_view
  star_dynamic_first_order
  star_segmented_frames_per_segment_4
  dynamic_gsplats_baseline
```

If `PSGD` was meant as a specific optimizer/gradient metric rather than PSNR,
add it explicitly in the result table. The default quality metric in this lane
is PSNR, with higher better.

## Win Criteria

A row is promising only if:

```text
1. It avoids the old per-sample gradient workspace.
2. It stays much closer to static-view STAR than to per-frame loop timing.
3. Its segment count is reported as M/N and M/(N*F).
4. It does not lose obvious PSNR against static-view STAR in the same-step
   smoke.
5. It states whether the test uses a synthetic moving camera path, a real
   time-varying dataset camera path, or the current fixed-camera data.
```

The likely near-term answer is:

```text
Attempt 1 gives the cheap derivative-correct smoke.
Attempt 2 is the production-shaped representation.
Attempt 3 is the future theory if segment counts grow too fast.
```

## Implementation Status

Implemented Attempt 1:

- `trainer_harness/world_tube.py`
  - `PinholeCameraMotion`
  - `project_world_tubes_pinhole_motion`
- `trainer_harness/pinhole_motion_projection_smoke.py`

Implemented Attempt 2 plumbing:

- `trainer_harness/variable_camera_segments.py`
  - `ProjectedVariableCameraSegments`
  - `project_piecewise_camera_time_segments`
- `trainer_harness/variable_camera_segments_smoke.py`

Implemented Attempt 3 final/projective path:

- `trainer_harness/world_tube.py`
  - `project_world_tubes_pinhole_projective_motion`
- `benchmarks/multicam_train_step_timing_probe.py`
  - `project_world_tube_sequence_projective_first_order`
  - `--uvt-camera-sequence-mode projective_first_order`
- `benchmarks/variable_camera_attempt_compare.py`
  - `projective_first_order` render/reference row
- `benchmarks/multicam_heldout_compare.py`
  - same-step trained PSNR support for `static_view`,
    `dynamic_first_order`, `projective_first_order`, and `segmented`

Integrated the runnable attempts into:

- `benchmarks/multicam_train_step_timing_probe.py`
- `benchmarks/variable_camera_attempt_compare.py`
- `dynaworld/research_experiments/world_foam_lane2/fixed_step_speed_compare.py`

Smoke validation:

```bash
python3 dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/pinhole_motion_projection_smoke.py
python3 dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/variable_camera_segments_smoke.py
python3 -m py_compile \
  dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/world_tube.py \
  dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/variable_camera_segments.py \
  dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/multicam_train_step_timing_probe.py \
  dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/variable_camera_attempt_compare.py
```

The first-order smoke shows zero-motion parity with fixed-camera projection to
`7.45e-09` max absolute delta, then nonzero finite `q_uvt` and `depth_beta`
changes under synthetic camera translation. The segmented smoke verifies
flattened segment shape, `parent_id`, and `t_minmax` bookkeeping.

The projective smoke compares Attempt 3 against Attempt 1 under a translated
moving pinhole camera and matches to `9.54e-07` max absolute delta.

## Current Sorted Implementations

Current comparison order:

1. **Attempt 3 / final: projective first-order gauge.** Same UVT tensor
   contract, one projected tube per world tube, homogeneous `P(t)X(t)` normal
   form. This is the clean path to promote for first-order moving cameras.
2. **Attempt 1: direct dynamic first-order derivative.** Same runtime shape and
   same pinhole answer as Attempt 3, but expressed in coordinate derivatives
   (`R_dot`, `T_dot`, `fx_dot`, `cx_dot`, etc.).
3. **Attempt 2: segmented camera-time UVT charts.** Correct production shape
   for curved camera motion, but still not production-quality until the Metal
   renderer enforces hard `t_min/t_max` or a real temporal partition of unity.

Static view remains the fixed-camera control. Per-frame loop remains the
negative control. Free dynamic GSplats remains the baseline comparison.

## Fixed-Step Speed Results

Result directory:

```text
dynaworld/research_experiments/world_foam_lane2/results/variable_camera_fixed_step_modes_2026_05_12/
```

Command shape:

```bash
python3 dynaworld/research_experiments/world_foam_lane2/fixed_step_speed_compare.py \
  --device mps \
  --cases 128x8,128x16,128x32,256x32 \
  --steps 8 \
  --warmup-steps 2 \
  --skip-world-foam \
  --skip-dynamic \
  --uvt-render-backend metal_tile \
  --uvt-sample-emission-mode direct_atomic \
  --uvt-camera-sequence-mode <mode> \
  --uvt-synthetic-pan-x 0.06 \
  --uvt-synthetic-zoom 0.02
```

All rows loaded the requested frame count. The synthetic moving camera is used
because the current DeepView validation clip has no measured per-frame camera
pose delta in `w2c`; this is a projection stress test, not proof on real
time-varying camera capture.

Mean fixed-step time:

| Case | Loaded Frames | Static View | Attempt 1 Dynamic First-Order | Attempt 2 Segmented f4 | Per-Frame Loop | Attempt 1 Speedup vs Loop | Attempt 2 Speedup vs Loop |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128px_8f | 8 | 0.035218s | 0.165506s | 0.058258s | 0.174077s | 1.05x | 2.99x |
| 128px_16f | 16 | 0.040079s | 0.206074s | 0.083112s | 0.305583s | 1.48x | 3.68x |
| 128px_32f | 32 | 0.040512s | 0.039181s | 0.080923s | 0.592606s | 15.12x | 7.32x |
| 256px_32f | 32 | 0.536398s | 0.082692s | 0.169595s | 0.661185s | 8.00x | 3.90x |

Render-only phase time:

| Case | Static View | Attempt 1 Dynamic First-Order | Attempt 2 Segmented f4 | Per-Frame Loop | Attempt 1 Speedup vs Loop | Attempt 2 Speedup vs Loop |
|---|---:|---:|---:|---:|---:|---:|
| 128px_8f | 0.003105s | 0.011731s | 0.009360s | 0.019597s | 1.67x | 2.09x |
| 128px_16f | 0.003616s | 0.011132s | 0.013095s | 0.031384s | 2.82x | 2.40x |
| 128px_32f | 0.003379s | 0.003262s | 0.005811s | 0.062991s | 19.31x | 10.84x |
| 256px_32f | 0.038829s | 0.005685s | 0.011525s | 0.084533s | 14.87x | 7.33x |

Projected primitive counts:

| Mode | 8f | 16f | 32f |
|---|---:|---:|---:|
| Static view | 256 | 256 | 256 |
| Attempt 1 dynamic first-order | 256 | 256 | 256 |
| Attempt 2 segmented f4 | 512 | 1024 | 2048 |
| Per-frame loop | 2048 | 4096 | 8192 |

Read: Attempt 1 keeps the same primitive count as static STAR. Attempt 2 with
fixed 4-frame chunks grows as `N * ceil(F/4)`, so it is still much smaller than
the per-frame loop but no longer flat. The per-frame loop is the explicit
negative control and should not be the production variable-camera path.

Timing caveat: the 8f/16f dynamic-first-order fixed-step rows show MPS/run-order
noise: they are slower than the 32f row even though the primitive count is
unchanged. Do not overfit those two numbers. The 32f and 256px/32f rows, plus
the render-only accuracy probe below, are the cleaner speed signal for Attempt
1.

## Render-Accuracy Probe

Result directory:

```text
dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/results/variable_camera_attempt_accuracy_2026_05_12/
```

This probe compares render output to the per-frame moving-camera loop under the
same untrained `WorldTubeModel` and synthetic camera motion. It is projection
accuracy against a reference path, not trained reconstruction PSNR. Higher PSNR
is better. If `PSGD` was intended as a different metric, it has not been added
yet.

| Case | Loaded Frames | Static PSNR vs Ref | Attempt 1 PSNR vs Ref | Attempt 2 f4 PSNR vs Ref | Attempt 1 Render Speedup vs Loop | Attempt 2 f4 Render Speedup vs Loop |
|---|---:|---:|---:|---:|---:|---:|
| 128px_8f | 8 | 39.02 | 54.72 | 17.36 | 5.07x | 0.48x |
| 128px_16f | 16 | 42.51 | 63.90 | 16.65 | 8.63x | 0.94x |
| 256px_32f | 16 | 42.56 | 64.00 | 16.77 | 8.57x | 2.44x |

The `max_frames=32` accuracy rows currently load 16 frames from the default
multicam validation config, so they are useful for speed/accuracy orientation
but not a true 32-frame accuracy matrix.

## 2026-05-13 Aligned Three-Attempt Results

Speed result directory:

```text
dynaworld/research_experiments/world_foam_lane2/results/variable_camera_three_attempts_speed_2026_05_13/
```

Trained PSNR result directory:

```text
dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/results/variable_camera_three_attempts_psnr_2026_05_13/
```

That first PSNR run used `view_sequence` and hit nonfinite STAR losses before
100 steps, so it is retained only as failed evidence. The accepted all-100-step
PSNR table is:

```text
dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/results/variable_camera_three_attempts_psnr_lr00003_2026_05_13/
```

Synthetic moving-camera reference PSNR result directory:

```text
dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/results/variable_camera_three_attempts_reference_psnr_2026_05_13/
```

The speed matrix uses synthetic camera motion (`pan_x=0.06`, `zoom=0.02`) so
the projection paths exercise moving camera coefficients. The trained PSNR
matrix uses the real fixed-camera multicam targets with no synthetic camera
motion so the GSplat comparison is fair. The reference-PSNR matrix is separate:
it compares each STAR projection to the synthetic per-frame moving-camera loop.

Fixed-step speed versus free dynamic GSplats:

| Case | Frames | Attempt 3 Projective | Attempt 1 Dynamic | Attempt 2 Segmented f4 | GSplat Baseline | Projective vs GSplat | Dynamic vs GSplat | Segmented vs GSplat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128px_8f | 8 | 0.038s | 0.037s | 0.056s | 0.160s | 4.3x | 4.3x | 2.9x |
| 128px_16f | 16 | 0.045s | 0.045s | 0.058s | 0.339s | 7.6x | 7.5x | 5.8x |
| 128px_32f | 32 | 0.041s | 0.052s | 0.098s | 0.559s | 13.5x | 10.7x | 5.7x |
| 256px_32f | 32 | 0.089s | 0.082s | 0.131s | 0.647s | 7.3x | 7.9x | 5.0x |

Render phase only:

| Case | Attempt 3 Projective | Attempt 1 Dynamic | Attempt 2 Segmented f4 | GSplat Baseline |
|---|---:|---:|---:|---:|
| 128px_8f | 0.0032s | 0.0032s | 0.0089s | 0.0753s |
| 128px_16f | 0.0038s | 0.0045s | 0.0093s | 0.1691s |
| 128px_32f | 0.0036s | 0.0042s | 0.0072s | 0.2521s |
| 256px_32f | 0.0057s | 0.0056s | 0.0112s | 0.2775s |

Same-step trained reconstruction PSNR, 100 optimizer steps, STAR `lr=0.0003`:

| Case | Method | Train PSNR | Heldout PSNR | Train Loop | Eval Render |
|---|---|---:|---:|---:|---:|
| 128px_8f | Attempt 3 Projective | 8.515 | 8.185 | 3.533s | 0.026s |
| 128px_8f | GSplat Baseline | 9.560 | 7.755 | 2.353s | 0.176s |
| 128px_8f | Attempt 1 Dynamic | 8.515 | 8.185 | 3.348s | 0.022s |
| 128px_8f | Attempt 2 Segmented f4 | 8.141 | 7.866 | 3.070s | 0.087s |
| 128px_16f | Attempt 3 Projective | 7.801 | 7.606 | 3.589s | 0.029s |
| 128px_16f | GSplat Baseline | 9.178 | 7.615 | 2.381s | 0.342s |
| 128px_16f | Attempt 1 Dynamic | 7.801 | 7.606 | 3.292s | 0.025s |
| 128px_16f | Attempt 2 Segmented f4 | 8.036 | 7.826 | 3.054s | 0.080s |
| 128px_32f | Attempt 3 Projective | 7.491 | 7.326 | 3.225s | 0.024s |
| 128px_32f | GSplat Baseline | 8.932 | 7.533 | 2.606s | 0.739s |
| 128px_32f | Attempt 1 Dynamic | 7.491 | 7.326 | 3.210s | 0.024s |
| 128px_32f | Attempt 2 Segmented f4 | 7.834 | 7.687 | 2.647s | 0.086s |
| 256px_32f | Attempt 3 Projective | 7.447 | 7.287 | 3.517s | 0.024s |
| 256px_32f | GSplat Baseline | 8.323 | 7.182 | 2.301s | 0.427s |
| 256px_32f | Attempt 1 Dynamic | 7.447 | 7.287 | 3.474s | 0.023s |
| 256px_32f | Attempt 2 Segmented f4 | 7.776 | 7.635 | 3.274s | 0.101s |

Synthetic moving-camera projection PSNR against the per-frame loop reference:

| Case | Loaded Frames | Attempt 3 Projective | Attempt 1 Dynamic | Attempt 2 Segmented f4 | Attempt 2 Segmented f1 |
|---|---:|---:|---:|---:|---:|
| 128px_8f | 8 | 54.73 | 54.72 | 17.36 | 17.17 |
| 128px_16f | 16 | 63.92 | 63.90 | 16.65 | 13.33 |
| 128px_32f | 32 | 72.69 | 72.66 | 16.33 | 12.17 |
| 256px_32f | 32 | 72.64 | 72.64 | 16.46 | 12.26 |

## Current Conclusion

Attempt 3 is the final first-order implementation to keep first in docs and
tables. It is mathematically the projective camera-time gauge version of
Attempt 1, and the smoke plus benchmark rows show it matches Attempt 1 for the
pinhole path while keeping the cleaner `h(t)=P(t)X(t)` formulation.

Attempt 1 remains useful as the coordinate-derivative check. It should stay as
an implementation cross-check and fallback, but not as the conceptual final.

Attempt 2 remains the production-shaped answer for strongly curved camera
motion, but not the current quality winner. The segment representation and
`parent_id/t_minmax` plumbing are in place, but the Metal shader still does not
hard-clamp temporal validity. Until that lands, the soft segmented path has poor
synthetic reference PSNR and should not be promoted over the first-order paths.

The next real gate is not more first-order math. It is:

```text
Add hard t_min/t_max segment validity in Metal.
Rerun Attempt 2 reference PSNR.
Then run a real variable-camera dataset, not only synthetic camera motion.
```

## 2026-05-13 Implicit-Camera STAR Fork

The simple non-feature implicit-camera dynamic-GS baseline to fork is:

```text
dynaworld/src/train_configs/local_mac_compare_free_linear_time_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc
```

That baseline is the cleanest match because it removes video features and token
decoding from the splat side: one direct Gaussian bank, linear xyz velocity in
normalized video time, fixed RGB, fixed scale/rotation, and learned implicit
camera heads.

Added STAR fork:

```text
dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/star_uvt_implicit_camera_baseline.py
```

Projection fix required for this: `world_tube.py` now preserves tensor-valued
camera intrinsics in pinhole/projective projection instead of converting
`fx/fy/cx/cy` to Python floats. Without that, camera-token gradients through
intrinsics would be silently cut.

What matches the GS baseline:

- direct learnable primitive bank
- linear xyz motion in normalized video time
- fixed RGB over time
- same `global_camera_token`
- same `path_camera_token + path_time_proj`
- same `GlobalCameraHead` / `PathCameraHead`
- same camera regularizer weights
- same reconstruction-loss parser

What is STAR-specific and not a perfect primitive equivalence:

- one UVT world tube replaces per-frame 3DGS raster calls
- opacity is fixed per tube; temporal support comes from `lambda_t`
- spatial footprint is fronto-parallel `precision_xy`, not full 3DGS
  anisotropic scale/rotation

Smoke artifacts:

| Artifact | Backend | Shape | Tubes | Steps | Final PSNR | Camera token grad |
|---|---|---:|---:|---:|---:|---|
| `star_uvt_implicit_camera_smoke_2026_05_13.json` | dense CPU | 32px_2f | 16 | 2 | 6.34 | nonzero global/path |
| `star_uvt_implicit_camera_metal_smoke_2026_05_13.json` | Metal tile | 32px_2f | 16 | 2 | 6.34 | nonzero global/path |
| `star_uvt_implicit_camera_64x8_smoke_2026_05_13.json` | Metal tile | 64px_8f | 256 | 3 | 5.99 | nonzero global/path |

Interpretation: the learned-camera path now exists for STAR UVT and gradients
reach both camera tokens through the projective UVT renderer. This is not yet a
full speed/PSNR comparison against the original 8192-splat 128px/16f GS
baseline; it is the implementation/smoke gate proving that the same implicit
camera baseline can be run in STAR.
