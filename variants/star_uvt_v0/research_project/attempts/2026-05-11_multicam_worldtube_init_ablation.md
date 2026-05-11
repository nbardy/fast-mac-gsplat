# 2026-05-11 Multicam Worldtube Init Ablation

Question: after the single-video overfit path showed a speed and fit signal,
can the same support/initialization ideas improve the DeepView multicam
train/heldout comparison?

Fixed comparison contract:

```text
dataset:       DeepView 03_Dog goodset
train cameras: camera_0006, camera_0014
heldout:       camera_0005
frames:        16
target size:   128x128
device:        mps
budget:        60 seconds per model
STAR backend:  metal_tile
splat backend: fast_mac
reference:     V-JEPA F32 alpha 1/128, heldout PSNR 13.6248
```

## Projection Audit

`worldtube_projection_audit.py` was added to inspect how initialized
worldtubes project into the train and heldout cameras before optimization.

The broad old init (`init_precision_xy=30`, `init_lambda_t=0.35`,
`init_opacity=0.35`) projects much wider than the single-video screen-space
settings in the anchor view:

```text
old 256-tube init:
  anchor lambda_u ~= 0.0389
  anchor lambda_v ~= 0.0219
  support radius ~= 15.2px x 20.3px
```

Narrowing to `init_precision_xy=96` gets closer to the single-video support:

```text
precision96 256-tube init:
  anchor lambda_u ~= 0.1246
  anchor lambda_v ~= 0.0701
  support radius ~= 8.5px x 11.3px
```

That geometric match did not improve the optimization result.

## Result Matrix

Old first-train-view init remains the best STAR-UVT multicam point so far:

```text
256 tubes, first train view, precision_xy 30, lambda_t 0.35, opacity 0.35:
  STAR train PSNR   10.681523323059082
  STAR heldout PSNR 10.493327140808105
  STAR steps        126
  STAR render eval  1.1473245000233874s

  splat train PSNR   20.192928314208984
  splat heldout PSNR 10.865671157836914
  splat steps        2729
  splat render eval  0.8316092919849325s
```

Single-video temporal/support transfer was negative:

```text
512 tubes, first train view, precision_xy 30, lambda_t 1.0, opacity 0.7:
  STAR train PSNR   8.91626262664795
  STAR heldout PSNR 8.603602409362793
  STAR steps        123
  STAR render eval  0.7689992079976946s

  splat heldout PSNR 10.857621192932129
```

Narrower spatial support was also negative:

```text
256 tubes, first train view, precision_xy 96, lambda_t 0.35, opacity 0.35:
  STAR train PSNR   8.877586841583252
  STAR heldout PSNR 8.753138542175293
  STAR steps        148
  STAR render eval  0.5526651249965653s

  splat heldout PSNR 10.824089050292969
```

Initializing from both train views did not help:

```text
256 tubes, all train views, precision_xy 30, lambda_t 0.35, opacity 0.35:
  STAR train PSNR   10.403272151947021
  STAR heldout PSNR 10.397439002990723
  STAR steps        130
  STAR render eval  0.7631473329965957s

  splat train PSNR   18.640210151672363
  splat heldout PSNR 10.863123893737793
  splat steps        2380
  splat render eval  0.6360267080017366s
```

## Read

The multicam issue is not solved by copying the single-video support recipe,
making the initialized projected blobs narrower, or covering both train cameras
at init time. The current STAR-UVT worldtube harness is still underfitting the
train views, and the direct splat baseline remains better on heldout under the
same local 60-second budget.

## Camera Projection Parity

The DeepView goodset cameras report `opencv_fisheye` lens metadata, while the
current STAR-UVT harness and local direct-splat baseline both use legacy
pinhole projection. The V-JEPA F32 reference config also has
`camera_projection: legacy_pinhole`, so this is not a simple explanation for
STAR-UVT losing to the local direct-splat baseline. It is still a physical
camera-model mismatch large enough to block a clean larger comparison.

`camera_projection_parity_audit.py` measures the pixel shift between the
DeepView fisheye projection and the pinhole approximation on the goodset
cameras:

```text
128px:
  camera_0006 mean 8.0625px, p95 19.6123px, max 25.3002px
  camera_0014 mean 8.1064px, p95 19.6325px, max 25.6131px
  camera_0005 mean 8.1447px, p95 19.6736px, max 25.6720px

256px:
  camera_0006 mean 16.2805px, p95 39.5764px, max 51.0328px
  camera_0014 mean 16.3690px, p95 39.6174px, max 51.6612px
  camera_0005 mean 16.4461px, p95 39.6998px, max 51.7793px
```

The next useful multicam work is camera/model parity and optimization
mechanics, not another blind capacity increase. The loaded DeepView bundle
reports `deepview_models_relative_opencv_fisheye`, while this harness currently
uses the pinhole-style comparison path inherited from the baseline config.
Until that decision is resolved, the 128px multicam result should stay a
research signal rather than a promotion gate.
