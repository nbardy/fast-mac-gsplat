# 2026-05-10 Gate 0 And Trainer Harness Attempt

## What Changed

- Added Gate 0 source in `variants/star_uvt_v0/`.
- Kept the first Metal implementation projected-only.
- Added this research-project folder to track phases, progress, attempts, and
  UVT-specific learnings.
- Added a dense differentiable trainer harness for projected
  `ScreenTimeTube` fitting.

## Important Limits

- The default trainer harness uses a dense PyTorch renderer as the gradient
  reference.
- Small true-Metal backward probes and an autograd bridge exist, but they are
  not production trainer evidence.
- The harness currently fits projected screen-time tubes; world-space projection
  is covered by smokes, not by a production trainer.
- Video loading is scaffolded and smoke-tested as a target source, but real
  video quality is not claimed yet.

## Verified Commands

```bash
python3 tests/gate0_check.py --cpu-only
python3 tests/gate0_check.py
python3 research_project/trainer_harness/smoke_train.py
python3 research_project/trainer_harness/train_synthetic.py \
  --scene moving_diagonal --steps 25 --lr 0.08 --seed 3 \
  --jitter-pixels 0.70 --metal-check
python3 research_project/trainer_harness/train_video.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 4 --target-size 16 --max-frames 2 --steps 1 --lr 0.02 --device cpu
python3 research_project/trainer_harness/world_projection_smoke.py
python3 research_project/trainer_harness/pinhole_projection_smoke.py
python3 research_project/trainer_harness/camera_spec_projection_smoke.py
python3 research_project/trainer_harness/gradient_probe.py
python3 research_project/trainer_harness/metal_autograd_smoke.py
python3 research_project/trainer_harness/simple_metal_backward_smoke.py
python3 research_project/trainer_harness/stable_metal_backward_smoke.py
python3 research_project/trainer_harness/unstable_metal_backward_smoke.py
python3 research_project/trainer_harness/tile_metal_autograd_smoke.py
python3 research_project/benchmarks/uvt_pair_benchmark.py
python3 research_project/benchmarks/backward_performance_smoke.py
python3 research_project/benchmarks/backward_performance_matrix.py
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 8 --per-frame-splats 8 --target-size 32 --max-frames 4 \
  --steps 8 --lr 0.04 --device cpu --seed 5 \
  --out-json research_project/benchmarks/results/video_fit_comparison_fixture.json \
  --contact-sheet research_project/benchmarks/results/video_fit_comparison_fixture.png
python3 research_project/benchmarks/training_comparison.py
```

The `--metal-check` synthetic trainer run reported final loss
`1.7662021491560154e-05`, Metal-vs-brute max RGB error
`5.960464477539063e-08`, and zero overflow.

The orthographic world projection smoke reported 2 projected tubes,
Metal-vs-brute max RGB error `5.960464477539063e-08`, and zero overflow.

The pinhole world projection smoke reported 2 projected tubes, used W2C plus
intrinsics, and reported Metal-vs-brute max RGB error
`5.960464477539063e-08` with zero overflow.

The `CameraSpec` adapter smoke used Dynaworld's `make_default_camera`, converted
it into the local `PinholeCamera`, and rendered through the same Metal parity
path with Metal-vs-brute max RGB error `5.960464477539063e-08` and zero
overflow.

The dense-backward gradient probe reported finite, nonzero gradients for center,
velocity, precision, opacity, and color.

The hybrid autograd smoke uses Metal forward and dense PyTorch backward. This
does not claim true Metal backward. It reports finite gradients for `ma`,
`q_uvt`, `opacity`, and `color`; depth gradients remain zero because depth
ordering is detached in the dense reference renderer.

The simplified Metal backward probe compares a single-tube per-sample Metal
gradient kernel against dense autograd. It does not cover sorted-tile
compositing or unstable-depth fallback. Current max gradient errors:
`1.9073486328125e-06` for color, `1.7881393432617188e-07` for `ma`, and `0.0`
for `q_uvt` and opacity.

The stable sorted-tile Metal backward probe reuses the Metal bin/sort tile path
and compares sorted alpha-compositing gradients against dense autograd on an
overlapping two-tube scene. Current max gradient errors: `3.4332275390625e-05`
for opacity, `1.1444091796875e-05` for color and `q_uvt`, and
`4.76837158203125e-06` for `ma`.

The unstable fallback Metal backward probe uses deterministic per-sample depth
ordering on `crossing_depth`. Current max gradient errors:
`3.4332275390625e-05` for `q_uvt`, `3.0517578125e-05` for opacity,
`1.33514404296875e-05` for color, and `5.9604644775390625e-06` for `ma`.

The tile-backward autograd bridge uses Metal forward, Metal per-sample
backward, and MPS `index_add_` reduction by tube id. The `crossing_depth` smoke
reports finite nonzero gradients for `ma`, `q_uvt`, opacity, and color.

The bounded backward performance smoke compares dense MPS backward against the
Metal tile-backward autograd bridge on a synthetic 16-tube, 32x32x4 case. With 1
warmup iteration and 2 measured iterations it reported dense mean
`16.18629200675059 ms` and Metal tile-backward mean `36.01418749894947 ms`, so
the dense path is faster at tiny scale.

The bounded large-scene backward matrix adds a `large_local` case with 64 tubes,
64x64 resolution, and 8 frames. With 1 warmup iteration and 1 measured iteration
it reported dense mean `104.00879199733026 ms`, Metal tile-backward mean
`73.45674998941831 ms`, and dense-to-Metal mean ratio `1.4159187823081347`.
This is still local bounded evidence, not promotion evidence.

The renderer pair benchmark reported the UVT pair count against the sliced
per-frame tile-splat baseline for all six tiny scenes: mean pair ratio `0.5`,
max pair ratio `0.5`, zero overflow, and max Metal-vs-brute RGB error
`5.960464477539063e-08`.

The tiny training comparison fits UVT tubes and an independent per-frame
Gaussian baseline against the same deterministic target. Current `moving_diagonal`
result: UVT loss `4.5250795665197074e-04 -> 1.7662021491560154e-05`;
per-frame loss `3.663150127977133e-04 -> 1.1087417988164816e-05`. It proves the
training comparison harness, not production quality.

The real-video fixture comparison fits UVT tubes and the simple per-frame
Gaussian baseline to 4 frames from `test_video_small_128_4fps.mp4` at 32x32 and
writes `research_project/benchmarks/results/video_fit_comparison_fixture.png`.
Current result: UVT loss `0.3166208863258362 -> 0.2973604202270508`, final L1
`0.5045824646949768`, 104 parameters, wall-clock `1050.151959003415 ms`;
per-frame loss `0.31666192412376404 -> 0.2972259521484375`, final L1
`0.5054242014884949`, 288 parameters, wall-clock `52.544709003996104 ms`.

Phase 5 promotion audit result: do not integrate into production GFlow/FasterGS
yet. The required current FasterGS same-video/same-resolution/same-split
comparison and held-out-camera evidence do not exist for this lane.

## Failed Or Deferred Work

- No production-scale backward performance benchmark was attempted in this
  slice.
- No current FasterGS video-quality or held-out-camera comparison was attempted.
- Production integration is intentionally deferred by the Phase 5 audit.
- Gradients through the discrete depth order remain out of scope.
- No production GFlow trainer integration was attempted.
- No production typed GFlow `CameraSpec` integration was attempted beyond the
  local projection smoke.
- Distorted camera models were not attempted.
