# STAR-UVT PRT v0 Progress

Last updated: 2026-05-13

## Checklist

- [x] Gate A0: fork created as `star_uvt_prt_v0`, separate from `star_uvt_v0`.
- [x] Gate A1: Python camera-path polynomial fit and world-tube PRT compiler.
- [x] Gate A2: static camera affine parity audit.
- [x] Gate A3: moving-camera dolly audit where affine UVT shows curvature error and PRT stays exact to the fitted camera path.
- [x] Gate A4: zoom/intrinsics audit.
- [x] Gate A5: camera-cut residual diagnostic rejects a bad smooth polynomial fit.
- [x] Gate A6: dense PRT render finite smoke.
- [x] Gate A7: curvature-selective hybrid diagnostic.
- [x] Gate A8: projection-only frame scaling probe across 8/16/32 frames.
- [x] Gate B0: direct Metal PRT forward API and dense-vs-Metal parity.
- [x] Gate B1: tiled Metal PRT bin-and-render parity on a tiny scene.
- [x] Gate B2: diagnostic tiled PRT timing and tile-load scaling against direct PRT.
- [x] Gate B3: capacity-256 256-tube overflow clearance smoke.
- [x] Gate B4: support tightening and tile-shape sweep.
- [x] Gate B5a: production PRT tile config selector and process-static env contract.
- [x] Gate B5b: timing launch integration applies selector before first Metal shader call.
- [x] Gate B5c: train-step smoke applies selector before first PRT Metal render.
- [x] Gate C0: CPU PRT autograd-vs-finite-difference gradient reference.
- [x] Gate C1: direct-serial Metal PRT backward parity against C0.
- [x] Gate C1b: tiled-forward PRT autograd train-step smoke using direct-serial backward.
- [x] Gate C2: tiled tile-pair atomic PRT backward parity and train-step smoke.
- [x] Gate C3a: 16/64-tube PRT train-step timing against direct-serial backward.
- [x] Gate C3b: numeric repeatability check for tiled atomic PRT backward.
- [x] Gate C3d: selector-recommended 256/512-tube PRT train-step timing.
- [x] Gate C3e: train-step breakdown isolates backward as the scale bottleneck.
- [x] Gate C4: tile-pixel atomic PRT backward removes the repeated target-slot recompute.
- [x] Gate C4b: moving-camera stress timing clears tile-pixel default promotion.
- [x] Gate C5: local single-video screen-PRT overfit compare.
- [x] Gate C5b: warmed direct-screen dense eval baseline and same-wall PRT overfit slice.
- [x] Gate C5c: 128px fuller-res single-video overfit slice.
- [x] Gate D0: synthetic world-camera forward probe against exact per-frame projection.
- [x] Gate D1: synthetic world-camera train/holdout compare against dense per-frame projection.
- [x] Gate D1b: 128px synthetic world-camera train/holdout scaling row.
- [x] Gate D2: real multicam PRT world-tube compare against direct dynamic splats.
- [ ] Gate C3c: decide whether bitwise deterministic gradients are required for PRT training.
- [ ] Gate D: variable-camera timing against `static_view`, `per_frame_loop`, segmented, and direct splats.
- [ ] Gate E: heldout/novel-camera sanity with world-state-only learned parameters.

## Current Read

This fork is not a separate thesis from STAR-UVT. It is the moving-camera
completion path for STAR-UVT's sublinear goal: keep one world-space primitive
covering many frames, compile the camera path into a compact sensor-time tube,
and keep rasterization in `(u, v, t)`.

The first runnable proof is projection-only plus dense reference. It does not
claim Metal speed or training quality yet. The audit proves the core center-curve
reason to fork:

```text
moving camera affine center error: 2.0354537963867188 px
moving camera PRT center error:    3.814697265625e-06 px
```

The projection-only scaling probe keeps the same conclusion across longer
windows:

```text
8 frames:  affine error 0.5121097564697266 px, PRT error 3.0517578125e-05 px
16 frames: affine error 1.8731575012207031 px, PRT error 3.0517578125e-05 px
32 frames: affine error 10.037454605102539 px, PRT error 3.0517578125e-05 px
```

The timing numbers in that probe are CPU diagnostics only. They do not prove
sublinear rendering speed; the next speed claim has to come from a tiled Metal
PRT path with dense-vs-Metal parity.

Gate B0 now has a compiled extension namespace and direct Metal PRT forward op:

```text
dense vs CPU brute PRT max error:   0.0
dense vs direct Metal PRT max error: 1.7881393432617188e-07
```

This is deliberately a direct per-pixel Metal reference op. It proves the
compiled API and rational-center semantics, but it is not the sublinear tiled
rasterizer yet. Gate B1 is where the actual STAR-UVT speed path starts.

Gate B1 now has a tiled Metal PRT bin-and-render path:

```text
dense vs tiled Metal PRT max error: 5.960464477539063e-08
active tiles: 24
max tile count: 2
overflow tiles: 0
```

The first tiled path uses sample-level depth selection inside each tile. That is
the conservative correctness-first version; Gate B2 still has to measure tile
load, timing, and whether stable depth shortcuts are worth adding.

Gate B2 timing probe:

```text
16 tubes:  direct median 2.165500001865439 ms, tiled median 3.4470835016691126 ms, ratio 1.5918187479564403
64 tubes:  direct median 6.6889790032291785 ms, tiled median 5.393875493609812 ms, ratio 0.8063824824395261
128 tubes: direct median 16.94479199795751 ms, tiled median 15.051396498165559 ms, ratio 0.8882609181617469
```

The same setup at 256 tubes fails the default-capacity gate with `15` overflow
tiles and max error `0.21615934371948242`. That is the next concrete scaling
blocker: either raise/segment capacity, tighten support bounds, or split camera
windows before claiming useful-scale speed. The no-overflow timing rows are
diagnostic and noisy; they are not a training-speed claim.

Gate B3 capacity-256 smoke:

```text
256 tubes, tile_capacity=256: direct median 17.852812503406312 ms, tiled median 15.45362549222773 ms, ratio 0.8656129385372294
max tile count: 152
overflow tiles: 0
max error vs direct: 5.364418029785156e-07
```

This shows the 256-tube failure was a capacity overflow, not a PRT binning math
failure. It does not solve default-capacity scaling; the next useful work is to
tighten support bounds, split camera windows, or make capacity selection part of
the cost model.

A 512-tube capacity-256 probe fails:

```text
512 tubes, tile_capacity=256: max tile count 313, overflow tiles 16, max error 0.2973073422908783
```

That bounds the current capacity-only fix: capacity 256 clears the 256-tube
smoke, but larger scenes need support tightening, camera-window splitting, or
adaptive capacity before the tiled PRT path is useful at scale.

Gate B4 support tightening and tile-shape probe:

The PRT binner now spends the opacity-derived support budget per frame:
`spatial_budget = support_tau - lambda_t * tau_t^2`. This tightens spatial tile
bounds near the temporal support edge without changing the PRT representation or
sample shader.

```text
8x8x2, cap128, 16/64/128 tubes: pass, max tile counts 13/38/87, overflow 0
8x8x2, cap256, 256 tubes: pass, max tile count 150, overflow 0
8x8x2, cap256, 512 tubes: fail, max tile count 296, overflow tiles 16
8x8x1, cap256, 512 tubes: fail, max tile count 302, overflow tiles 30
4x4x2, cap256, 512 tubes: pass, max tile count 238, overflow 0, max error 7.152557373046875e-07
```

Read: support tightening helps slightly but does not remove the 512-tube hotspot.
Temporal tile splitting alone does not fix it. Spatial tile splitting does clear
the 512-tube cap-256 gate in this smoke, so the next practical rasterizer path is
an explicit tile-shape/capacity cost model rather than more global capacity bumps.

The first explicit tile-config sweep runs each candidate in a fresh Python
process because the Metal tile constants are process-static. On the 512-tube
stress case it selected `4x4x2:256`:

```text
8x8x2:128: fail, max tile count 303, overflow tiles 33
8x8x2:256: fail, max tile count 303, overflow tiles 16
8x8x2:512: pass, max tile count 303, overflow 0, tiled/direct ratio 0.8633058849041223
4x4x2:128: fail, max tile count 238, overflow tiles 80
4x4x2:256: pass, max tile count 238, overflow 0, tiled/direct ratio 0.5604430452008048
4x4x2:512: pass, max tile count 238, overflow 0, tiled/direct ratio 0.6103707983493051
```

Read: capacity and tile shape have to be chosen together. `4x4` fixes the
hotspot shape, but only cap 256 keeps all active tile lists valid. `8x8` with
cap 512 is valid but slower on this smoke, so shape splitting beats a pure
capacity bump here.

A small cap-256 occupancy matrix narrows the hotspot:

```text
8x8x2:256, 384 tubes, 64x48x8:  pass, max tile count 221, overflow 0
8x8x2:256, 512 tubes, 64x48x8:  fail, max tile count 296, overflow 16
8x8x2:256, 512 tubes, 128x96x8: pass, max tile count 227, overflow 0
8x8x2:256, 512 tubes, 64x48x16: fail, max tile count 303, overflow 23
```

Read: the default 512-tube failure is a spatial tile-density hotspot, not a
pure temporal-window problem. Doubling spatial resolution creates more spatial
tiles and clears the overflow; doubling frames does not. This supports the
existing `4x4x2:256` selector choice for 512 tubes.

At 1024 tubes, the selected configuration moves to `4x4x2:512`:

```text
8x8x2:512: fail, max tile count 609, overflow tiles 16
4x4x2:256: fail, max tile count 496, overflow tiles 80
4x4x2:512: pass, max tile count 496, overflow 0, tiled/direct ratio 0.6402926779318678
```

Read: the current cost model needs both a shape ladder and a capacity ladder.
`4x4` remains the right shape in this synthetic hotspot, but the capacity tier
must rise from 256 to 512 by 1024 tubes.

A stronger moving-camera stress (`--camera-motion-scale 3.0`) at 512 tubes also
selects `4x4x2:512`:

```text
8x8x2:512: pass, max tile count 346, overflow 0, tiled/direct ratio 1.210377738086172
4x4x2:256: fail, max tile count 276, overflow tiles 3
4x4x2:512: pass, max tile count 276, overflow 0, tiled/direct ratio 0.7517214978728403
```

Read: stronger camera motion pushes the 512-tube case over the cap-256 edge.
The 4x4 shape still controls the hotspot better than 8x8, but the selected
capacity has to rise to 512 under stronger camera motion.

Gate B5 now has a production-facing selector module:
`torch_gsplat_bridge_star_uvt_prt.tile_config`. It exposes typed tile configs,
env export, the verified heuristic selector, and the shared sweep-summary
selector used by `projective_rational_tile_config_sweep.py`.

The current verified heuristic is deliberately narrow:

```text
<=128 tubes, normal motion: 8x8x2:128
<=256 tubes, normal motion: 8x8x2:256
<=512 tubes, normal motion: 4x4x2:256
<=512 tubes, motion scale >= 3: 4x4x2:512
<=1024 tubes: 4x4x2:512
>1024 tubes: fail closed unless allow_unverified=True
```

The env contract is still process-static from the caller's point of view: set
`STAR_UVT_TILE_X/Y/T/CAPACITY` before the first Metal shader call and construct
`UVTRenderConfig` from the same values. The selector does not make tile constants
runtime-switchable inside an already-warmed process.

The forward timing probe now accepts `--tile-config auto` or an explicit key like
`--tile-config 4x4x2:256`, applies the corresponding env vars before rendering,
and records `tile_config` / `tile_config_key` in the output JSON. Manual
`--tile-x/y/t/capacity` flags still work and are also mirrored into env by the
probe, so timing launches no longer need separate shell env flags.

Tiny Metal smoke for the auto path:

```text
python3 research_project/benchmarks/projective_rational_metal_forward_timing_probe.py --tube-counts 16 --tile-config auto --warmups 0 --repeats 1 --out-json research_project/benchmarks/results/projective_rational_metal_forward_timing_probe_auto_smoke_16t.json
tile_config_key: 8x8x2:128
max abs error vs direct: 5.960464477539062e-07
max tile count: 13
overflow tiles: 0
```

Gate B5c now has a tiny train-step smoke that applies the selector before the
first PRT Metal render. It selects `8x8x2:128` for the two-tube smoke and passes
that config into `UVTRenderConfig`, so the selector contract is exercised on an
actual autograd path, not just timing probes.

Gate C0 establishes the gradient target before writing Metal backward kernels:

```text
python3 research_project/benchmarks/projective_rational_gradient_reference_check.py --out-json research_project/benchmarks/results/projective_rational_gradient_reference_check.json
pass: true
max abs error: 2.7381349354982376e-05
max rel error: 0.02855873424253072
checked params: h_coeff, lambda_uv, lambda_t, center_t, opacity, color
```

Gate C1 adds the first Metal backward parity check:

```text
python3 research_project/benchmarks/projective_rational_direct_serial_backward_check.py --out-json research_project/benchmarks/results/projective_rational_direct_serial_backward_check.json
pass: true
max abs error: 5.960464477539063e-08
max rel error: 1.0218293027719483e-05
checked params: h_coeff, lambda_uv, lambda_t, center_t, opacity, color
```

This is a direct-serial Metal reference kernel. It proves the PRT gradient math
and binding boundary against the C0 CPU target, but it is intentionally not the
fast tiled training path. Gate C2 still has to move these derivatives into the
tile/sample path before we can call PRT training a speed path.

Gate C1b adds the first train-step autograd bridge:

```text
python3 research_project/trainer_harness/projective_rational_metal_autograd_smoke.py --out-json research_project/benchmarks/results/projective_rational_metal_autograd_smoke.json
pass: true
forward_mode: tiled
tile_config_key: 8x8x2:128
initial_loss: 0.0008642825414426625
final_loss: 0.0008352987351827323
```

This originally used tiled Metal PRT forward and direct-serial Metal PRT
backward. Gate C2 replaces that smoke with the tiled tile-pair atomic backward.

Gate C2 adds the first tiled PRT backward kernel:

```text
python3 research_project/benchmarks/projective_rational_tile_pair_atomic_backward_check.py --out-json research_project/benchmarks/results/projective_rational_tile_pair_atomic_backward_check.json
pass: true
max abs error: 5.4016709327697754e-08
max rel error: 4.970375357515877e-06
tile_unstable_count: 2
```

The autograd train-step smoke now runs `forward_mode=tiled` and
`backward_mode=tile_pair_atomic`:

```text
python3 research_project/trainer_harness/projective_rational_metal_autograd_smoke.py --backward-mode tile_pair_atomic --out-json research_project/benchmarks/results/projective_rational_metal_autograd_smoke.json
pass: true
initial_loss: 0.0008642825414426625
final_loss: 0.0008352987351827323
```

Read: PRT now has a tiled backward path wired through autograd. It is still an
atomic tile-pair path and only checked on the tiny smoke, so the next gate is
non-tiny train-step timing and repeatability, not more direct-serial parity.

Gate C3a times a small but non-tiny train-step path:

```text
python3 research_project/benchmarks/projective_rational_train_step_timing_probe.py --tube-counts 16,64 --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_train_step_timing_probe_16_64_tiled_atomic_smoke.json
python3 research_project/benchmarks/projective_rational_train_step_timing_probe.py --tube-counts 16,64 --backward-mode direct_serial --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_train_step_timing_probe_16_64_direct_serial_smoke.json
```

Result:

```text
16 tubes: tile-pair atomic 17.468499994720332 ms, direct serial 303.265750000719 ms, speedup 17.36072073116626x
64 tubes: tile-pair atomic 73.10462500026915 ms, direct serial 2398.661458006245 ms, speedup 32.81135028047013x
```

Read: tiled atomic backward changes the training path from correctness-only to
meaningfully faster on the diagnostic scene. It is still not a video-quality or
determinism claim; C3b must check repeatability before scaling this path.

Gate C3b checks same-state repeatability:

```text
python3 research_project/benchmarks/projective_rational_train_step_repeatability_probe.py --tube-counts 16,64 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_train_step_repeatability_probe_16_64_tiled_atomic.json
```

Result:

```text
16 tubes: pass, max_grad_delta 3.7834979593753815e-10, max_loss_delta 0.0
64 tubes: pass, max_grad_delta 2.3283064365386963e-10, max_loss_delta 0.0
unique gradient digests: 3/3 for every checked parameter
```

Read: tiled atomic PRT backward is numerically repeatable at this scale but not
bitwise deterministic. That is acceptable for a first train-speed path, but C3c
keeps the bitwise-determinism decision explicit before promotion.

Gate C3d scales the same tiled atomic train-step probe to the selector-recommended
256/512-tube cases:

```text
python3 research_project/benchmarks/projective_rational_train_step_timing_probe.py --tube-counts 256 --tile-config auto --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_train_step_timing_probe_256_auto_tiled_atomic.json
python3 research_project/benchmarks/projective_rational_train_step_timing_probe.py --tube-counts 512 --tile-config auto --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_train_step_timing_probe_512_auto_tiled_atomic.json
```

Result:

```text
256 tubes, 8x8x2:256: pass, median step 1535.383541995543 ms, max tile count 150, overflow 0
512 tubes, 4x4x2:256: pass, median step 2989.2938749981113 ms, max tile count 238, overflow 0
```

Read: the selector-recommended train path now stays finite, overflow-free, and
loss-decreasing at 256/512 tubes on the synthetic diagnostic scene. That is a
correctness and scaling-validity result, not the speed win. The current tiled
atomic training path is still seconds per step at this size, so the rasterizer
needs a real train-speed pass before it can support the original STAR-UVT speed
claim.

Gate C3e splits the same train-step timing into forward, loss, backward, and
optimizer segments:

```text
python3 research_project/benchmarks/projective_rational_train_step_breakdown_probe.py --tube-counts 256 --tile-config auto --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_train_step_breakdown_probe_256_auto_tiled_atomic.json
python3 research_project/benchmarks/projective_rational_train_step_breakdown_probe.py --tube-counts 512 --tile-config auto --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_train_step_breakdown_probe_512_auto_tiled_atomic.json
```

Result:

```text
256 tubes: median forward 9.60899998608511 ms, backward 1409.865624998929 ms, wall 1420.4636249924079 ms
512 tubes: median forward 21.143167003174312 ms, backward 3032.4306250113295 ms, wall 3055.680666991975 ms
```

Read: forward PRT rasterization is not the train-step bottleneck in this probe.
The current tile-pair atomic backward recomputes the per-pixel ordered sequence
once per target slot, so the next speed path should replace that kernel before
spending time on forward rasterizer micro-optimizations.

Gate C4 adds `projective_rational_tile_pixel_atomic_backward`, a tiled backward
kernel that computes each tile pixel's ordered sequence once and accumulates
gradients for all active tubes from that pass.

Validation:

```text
python3 tests/projective_rational_tile_pixel_atomic_backward_check.py
python3 research_project/benchmarks/projective_rational_tile_pixel_atomic_backward_check.py --out-json research_project/benchmarks/results/projective_rational_tile_pixel_atomic_backward_check.json
python3 research_project/trainer_harness/projective_rational_metal_autograd_smoke.py --backward-mode tile_pixel_atomic --out-json research_project/benchmarks/results/projective_rational_metal_autograd_smoke_tile_pixel_atomic.json
```

Result:

```text
tile-pixel atomic parity: pass, max abs 7.450580596923828e-08, max rel 1.1374921996321063e-05
train smoke: pass, initial loss 0.0008642825414426625, final loss 0.0008352987351827323
```

Timing against the old tile-pair atomic path:

```text
256 tubes: old median step 1535.383541995543 ms; tile-pixel median step 38.774833010393195 ms
512 tubes: old median step 2989.2938749981113 ms; tile-pixel median step 48.69275000237394 ms
```

Breakdown with tile-pixel atomic:

```text
256 tubes: median forward 8.881457993993536 ms, backward 15.592499999911524 ms, wall 26.357708004070446 ms
512 tubes: median forward 20.76037500228267 ms, backward 27.39641700463835 ms, wall 48.84575000323821 ms
```

Repeatability:

```text
256 tubes: pass, max_grad_delta 1.6880221664905548e-09, max_loss_delta 0.0
512 tubes: pass, max_grad_delta 3.205059329047799e-09, max_loss_delta 0.0
unique gradient digests: 3/3 for every checked parameter
```

Read: this is the first PRT training-speed result that matches the intended
shape of the STAR-UVT rasterizer. It is still synthetic and atomic, so it is
not a held-out-video quality claim and not bitwise deterministic. But the
seconds-per-step blocker from C3e is gone on the 256/512 diagnostic cases.

Gate C4b checks the new default candidate under stronger camera motion:

```text
python3 research_project/benchmarks/projective_rational_train_step_timing_probe.py --tube-counts 512 --tile-config auto --camera-motion-scale 3.0 --backward-mode tile_pixel_atomic --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_train_step_timing_probe_512_motion3_auto_tile_pixel_atomic.json
python3 research_project/benchmarks/projective_rational_train_step_breakdown_probe.py --tube-counts 512 --tile-config auto --camera-motion-scale 3.0 --backward-mode tile_pixel_atomic --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_train_step_breakdown_probe_512_motion3_auto_tile_pixel_atomic.json
```

Result:

```text
512 tubes, motion scale 3.0, selected 4x4x2:512: pass, median step 63.434208001126535 ms, max tile count 276, overflow 0
breakdown: forward 27.24974999728147 ms, backward 34.7171250032261 ms, wall 62.473084006342106 ms
```

Read: the stronger moving-camera stress no longer forces a seconds-per-step
backward path. `tile_pixel_atomic` is now the default PRT training backward for
the research harness and timing probes, while direct-serial and tile-pair modes
remain available for parity and regression checks.

Gate C5 adds a local single-video overfit benchmark:
`research_project/benchmarks/projective_rational_video_overfit_compare.py`.
This is intentionally narrower than the full world-camera harness: it optimizes
screen-time PRT tubes with the Metal `tile_pixel_atomic` backward and compares
against a simple per-frame screen Gaussian baseline. The baseline is not full
3DGS.

Smoke:

```text
python3 research_project/benchmarks/projective_rational_video_overfit_compare.py ../../../../../tests/fixtures/lalaland.mp4 --target-size 32 --max-frames 3 --steps 2 --tube-count 32 --per-frame-splats 16 --render-repeats 1 --out-json research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_32_3f_2step_smoke.json --contact-sheet research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_32_3f_2step_smoke.png
```

Local overfit comparison:

```text
python3 research_project/benchmarks/projective_rational_video_overfit_compare.py ../../../../../tests/fixtures/lalaland.mp4 --target-size 64 --max-frames 4 --steps 20 --tube-count 128 --per-frame-splats 32 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_64_4f_20step_128prt_32pf.json --contact-sheet research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_64_4f_20step_128prt_32pf.png
```

Result:

```text
PRT: pass, PSNR 22.707577326192375 dB, final MSE 0.0053609563037753105, median render 6.408583998563699 ms, train wall 308.9734999957727 ms
per-frame screen Gaussian: PSNR 15.421344281483602 dB, final MSE 0.02869892120361328, median render 16.718792001483962 ms, train wall 1824.5809170039138 ms
PRT tile load: max tile count 66, overflow 0
```

Same-step 200-step run:

```text
python3 research_project/benchmarks/projective_rational_video_overfit_compare.py ../../../../../tests/fixtures/lalaland.mp4 --target-size 64 --max-frames 4 --steps 200 --tube-count 128 --per-frame-splats 32 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_64_4f_200step_128prt_32pf.json --contact-sheet research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_64_4f_200step_128prt_32pf.png
```

Result:

```text
PRT: pass, PSNR 26.43693188933276 dB, final MSE 0.002271468983963132, median render 8.317833009641618 ms, train wall 4279.6319170010975 ms
per-frame screen Gaussian: PSNR 23.767788461727967 dB, final MSE 0.004199727904051542, median render 17.888916001538746 ms, train wall 18312.233666991233 ms
PRT tile load: max tile count 61, overflow 0
```

Comparable-parameter 200-step run with 64 splats per frame:

```text
python3 research_project/benchmarks/projective_rational_video_overfit_compare.py ../../../../../tests/fixtures/lalaland.mp4 --target-size 64 --max-frames 4 --steps 200 --tube-count 128 --per-frame-splats 64 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_64_4f_200step_128prt_64pf.json --contact-sheet research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_64_4f_200step_128prt_64pf.png
```

Result:

```text
PRT: pass, parameters 2176, PSNR 26.477061063327035 dB, final MSE 0.00225057709030807, median render 7.809999995515682 ms, train wall 5338.531999994302 ms
per-frame screen Gaussian: parameters 2048, PSNR 25.085034375204227 dB, final MSE 0.003100962843745947, median render 22.821167003712617 ms, train wall 32772.46066600492 ms
PRT tile load: max tile count 60, overflow 0
```

Gate C5b strengthens the local direct-screen baseline without claiming a full
3DGS comparison. The baseline still trains through the original per-splat loop
because MPS backward through dense vectorized alpha compositing is slower, but
it can now evaluate render speed through a warmed dense vectorized path.

Same-step, comparable-parameter, warmed dense-eval run:

```text
python3 research_project/benchmarks/projective_rational_video_overfit_compare.py ../../../../../tests/fixtures/lalaland.mp4 --target-size 64 --max-frames 4 --steps 200 --tube-count 128 --per-frame-splats 64 --baseline-eval-render-mode dense_vectorized --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_64_4f_200step_128prt_64pf_loop_train_dense_eval_warm.json --contact-sheet research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_64_4f_200step_128prt_64pf_loop_train_dense_eval_warm.png
```

Result:

```text
PRT: parameters 2176, PSNR 26.419920016248945 dB, final MSE 0.0022803840693086386, train wall 4528.415291992133 ms, warmed render median 8.75945900043007 ms, max tile count 61, overflow 0
direct-screen baseline: parameters 2048, PSNR 25.085034375204227 dB, final MSE 0.003100962843745947, train wall 33081.05508299195 ms, warmed dense-eval render median 7.0026249886723235 ms
```

Same-wall PRT-only run against that baseline's training budget:

```text
python3 research_project/benchmarks/projective_rational_video_overfit_compare.py ../../../../../tests/fixtures/lalaland.mp4 --target-size 64 --max-frames 4 --steps 1500 --tube-count 128 --per-frame-splats 64 --skip-baseline --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_64_4f_1500step_128prt_same_wall_as_64pf_baseline.json --contact-sheet research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_64_4f_1500step_128prt_same_wall_as_64pf_baseline.png
```

Result:

```text
PRT: parameters 2176, PSNR 28.7287778393939 dB, final MSE 0.0013400537427514791, train wall 36633.79866699688 ms, warmed render median 7.757499988656491 ms, max tile count 51, overflow 0
```

Read: at the same 200 optimizer steps, PRT beats the direct-screen baseline on
quality and training wall time, while the dense direct-screen eval path is
slightly faster to render at this tiny scale. At roughly the same training
wall-clock budget, PRT reaches 28.7287778393939 dB against the baseline's
25.085034375204227 dB. This is still a local screen-space overfit bridge, not a
world-camera or heldout proof.

Gate C5c repeats the local overfit comparison at 128px with comparable
parameters: 256 PRT tubes (`4352` params) against 128 per-frame screen splats
(`4096` params). The baseline still trains through the loop path and evaluates
with warmed dense vectorized rendering. This is still screen-space overfit, not
full 3DGS or world-camera heldout.

Same-step 200-step run:

```text
python3 research_project/benchmarks/projective_rational_video_overfit_compare.py ../../../../../tests/fixtures/lalaland.mp4 --target-size 128 --max-frames 4 --steps 200 --tube-count 256 --per-frame-splats 128 --baseline-eval-render-mode dense_vectorized --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_128_4f_200step_256prt_128pf_loop_train_dense_eval_warm.json --contact-sheet research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_128_4f_200step_256prt_128pf_loop_train_dense_eval_warm.png
```

Result:

```text
PRT: parameters 4352, PSNR 26.84948999585133 dB, final MSE 0.002065622713416815, train wall 5563.788790997933 ms, warmed render median 11.02358300704509 ms, max tile count 60, overflow 0
direct-screen baseline: parameters 4096, PSNR 25.49720083433901 dB, final MSE 0.002820200053974986, train wall 48931.89037499542 ms, warmed dense-eval render median 23.29466700030025 ms
```

Same-wall PRT-only run against that baseline's training budget:

```text
python3 research_project/benchmarks/projective_rational_video_overfit_compare.py ../../../../../tests/fixtures/lalaland.mp4 --target-size 128 --max-frames 4 --steps 1750 --tube-count 256 --per-frame-splats 128 --skip-baseline --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_128_4f_1750step_256prt_same_wall_as_128pf_baseline.json --contact-sheet research_project/benchmarks/results/projective_rational_video_overfit_compare_lalaland_128_4f_1750step_256prt_same_wall_as_128pf_baseline.png
```

Result:

```text
PRT: parameters 4352, PSNR 29.328427511305453 dB, final MSE 0.001167232170701027, train wall 54290.77204100031 ms, warmed render median 10.575874999631196 ms, max tile count 59, overflow 0
```

Read: unlike the 64px tiny case, PRT is faster than the warmed dense
direct-screen eval path at 128px while also training much faster and reaching
higher same-step quality. Same-wall PRT reaches 29.328427511305453 dB. The
visual contact sheets remain blurry because this is a low-primitive screen-space
overfit, but the fuller-resolution row preserves the speed/quality direction.

Gate D0 adds a synthetic world-camera forward probe:
`research_project/benchmarks/projective_rational_world_camera_forward_probe.py`.
It renders the same moving-camera world tubes through an exact per-frame dense
projection reference, a static-camera PRT path, and the moving-camera PRT path.
This is forward-only and synthetic; it is the bridge before a full train/heldout
world-camera harness.

Smoke:

```text
python3 research_project/benchmarks/projective_rational_world_camera_forward_probe.py --frames 4 --height 32 --width 32 --tube-count 32 --camera-motion-scale 3.0 --warmups 0 --repeats 1 --out-json research_project/benchmarks/results/projective_rational_world_camera_forward_probe_32_4f_32t_smoke.json
```

64px moving-camera rows:

```text
python3 research_project/benchmarks/projective_rational_world_camera_forward_probe.py --frames 4 --height 64 --width 64 --tube-count 64 --camera-motion-scale 3.0 --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_world_camera_forward_probe_64_4f_64t_motion3_warm.json
python3 research_project/benchmarks/projective_rational_world_camera_forward_probe.py --frames 4 --height 64 --width 64 --tube-count 128 --camera-motion-scale 3.0 --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_world_camera_forward_probe_64_4f_128t_motion3_warm.json
```

Result:

```text
64 tubes: PRT PSNR vs exact per-frame reference 120.0 dB, static-camera PSNR 33.70920953512132 dB, PRT median render 2.8497079911176115 ms, exact per-frame dense reference 45.558124998933636 ms, max tile count 39, overflow 0
128 tubes: PRT PSNR vs exact per-frame reference 120.0 dB, static-camera PSNR 32.477733403488244 dB, PRT median render 4.573249985696748 ms, exact per-frame dense reference 51.37370800366625 ms, max tile count 79, overflow 0
```

Read: the moving-camera PRT compiler fixes the static-camera error in this
synthetic forward probe and preserves the tiled STAR-UVT-style speed shape.
This still is not a direct-splat training or heldout comparison.

Gate D1 adds a synthetic world-camera train/holdout comparison:
`research_project/benchmarks/projective_rational_world_camera_train_compare.py`.
It learns world-state tube parameters and evaluates both the training camera
sequence and a shifted holdout camera sequence. The baseline is exact dense
per-frame projection of the same trainable world tubes, not full 3DGS or direct
splats. For this first gate, the compiled footprint `lambda_uv` is detached so
the result measures the camera-path training loop through world position,
velocity, temporal precision, opacity, and color.

Smoke:

```text
python3 research_project/benchmarks/projective_rational_world_camera_train_compare.py --steps 3 --render-repeats 1 --render-warmups 0 --out-json research_project/benchmarks/results/projective_rational_world_camera_train_compare_32_4f_32t_3step_smoke.json
```

64px rows:

```text
python3 research_project/benchmarks/projective_rational_world_camera_train_compare.py --height 64 --width 64 --tube-count 64 --steps 20 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_world_camera_train_compare_64_4f_64t_20step.json
python3 research_project/benchmarks/projective_rational_world_camera_train_compare.py --height 64 --width 64 --tube-count 64 --steps 20 --tile-config 8x8x2:128 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_world_camera_train_compare_64_4f_64t_20step_8x8x2_cap128.json
python3 research_project/benchmarks/projective_rational_world_camera_train_compare.py --height 64 --width 64 --tube-count 128 --steps 20 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_world_camera_train_compare_64_4f_128t_20step_auto.json
python3 research_project/benchmarks/projective_rational_world_camera_train_compare.py --height 64 --width 64 --tube-count 128 --steps 20 --tile-config 8x8x2:128 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_world_camera_train_compare_64_4f_128t_20step_8x8x2_cap128.json
```

Result:

```text
32px/32 tubes, auto: PRT train/holdout PSNR 36.1888/36.0396 dB, train wall 271 ms, render 15.7/16.8 ms; dense train/holdout PSNR 36.1877/36.0387 dB, train wall 582 ms, render 43.3/43.9 ms
64px/64 tubes, auto 4x4x2:512: PRT 38.2043/38.1243 dB, train wall 728 ms, render 29.1/30.2 ms; dense 38.2039/38.1241 dB, train wall 4683 ms, render 77.0/76.8 ms
64px/64 tubes, 8x8x2:128: PRT 38.2053/38.1246 dB, train wall 970 ms, render 27.7/28.4 ms; dense 38.2039/38.1241 dB, train wall 4758 ms, render 76.5/74.1 ms
64px/128 tubes, auto 4x4x2:512: PRT 40.6451/40.5645 dB, train wall 1607 ms, render 48.4/49.9 ms; dense 40.6374/40.5579 dB, train wall 7561 ms, render 101.7/100.0 ms
64px/128 tubes, 8x8x2:128: PRT 40.6390/40.5600 dB, train wall 1393 ms, render 48.7/57.3 ms; dense 40.6374/40.5579 dB, train wall 7390 ms, render 81.7/102.0 ms
```

Read: D1 proves the trainable world-state path can optimize through the tiled
PRT camera compiler and evaluate train plus holdout camera sequences. Quality
matches the exact dense per-frame projection baseline within run noise while
training and rendering materially faster in these synthetic rows. This is still
not a full 3DGS/direct-splat comparison and not a real-video heldout result.

Gate D1b repeats the world-camera train/holdout comparison at 128px for the
128-tube case, with both the conservative auto tile config and an explicit
`8x8x2:128` row:

```text
python3 research_project/benchmarks/projective_rational_world_camera_train_compare.py --height 128 --width 128 --tube-count 128 --steps 20 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_world_camera_train_compare_128_4f_128t_20step_auto.json
python3 research_project/benchmarks/projective_rational_world_camera_train_compare.py --height 128 --width 128 --tube-count 128 --steps 20 --tile-config 8x8x2:128 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_world_camera_train_compare_128_4f_128t_20step_8x8x2_cap128.json
```

Result:

```text
128px/128 tubes, auto 4x4x2:512: PRT train/holdout PSNR 40.6525/40.5855 dB, train wall 1345 ms, render 32.8/27.6 ms, max tile count 54, overflow 0; dense 40.6488/40.5832 dB, train wall 5578 ms, render 71.4/70.1 ms
128px/128 tubes, 8x8x2:128: PRT train/holdout PSNR 40.6494/40.5794 dB, train wall 1521 ms, render 32.5/27.6 ms, max tile count 69, overflow 0; dense 40.6488/40.5832 dB, train wall 5457 ms, render 72.0/71.7 ms
```

Read: the synthetic heldout-camera speed/quality result survives the 128px
scale-up. Auto `4x4x2:512` is slightly faster on train wall here, so there is no
selector change from this row. This is still exact dense projection of the same
world tubes, not full direct splats.

Gate D2 adds the first real multicam direct-splat comparison:
`research_project/benchmarks/projective_rational_multicam_splat_compare.py`.
It uses the DeepView dog local multicam config with train cameras `camera_0006`
and `camera_0014`, and heldout camera `camera_0005`. The PRT path trains
world tubes through the moving-camera tiled Metal rasterizer. The baseline is
Dynaworld's `FreeDynamic3DGS` direct per-frame dynamic splat model rendered
through the dense splat path in these rows.

The initial draft trained PRT against the full train-camera sequence each step
while direct splats sampled one frame per step. That was unfair, so the script
now defaults PRT to `--prt-loss-mode sampled_frame`; all D2 numbers below use
that corrected mode. A one-step random-view sampled loss can increase when the
sample changes, so the benchmark records `sampled_loss_decreased` separately
but gates pass/fail on finite losses plus no PRT tile overflow.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 32 --max-frames 2 --steps 1 --prt-tubes 16 --splat-count 16 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_32_2f_16t_16s_1step_smoke.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --splat-count 128 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_20step.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 128 --splat-count 128 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_200step.json
```

Result:

```text
32px/2f/1 step: PRT 176 params, train/heldout PSNR 4.0547/4.1095 dB, train wall 0.161 s, render 10.745/8.725 ms, max tile count 8, overflow 0; direct splats 448 params, train/heldout PSNR 3.7585/3.8652 dB, train wall 0.970 s, render 5.472/2.719 ms
64px/4f/20 steps: PRT 1408 params, train/heldout PSNR 6.9801/6.2738 dB, train wall 1.101 s, render 42.998/42.114 ms, max tile count 37, overflow 0; direct splats 7168 params, train/heldout PSNR 4.0628/4.1142 dB, train wall 2.125 s, render 25.996/22.241 ms
64px/4f/200 steps: PRT 1408 params, train/heldout PSNR 12.7012/9.5443 dB, train wall 15.565 s, render 64.103/69.896 ms, max tile count 35, overflow 0; direct splats 7168 params, train/heldout PSNR 4.4404/4.3374 dB, train wall 6.437 s, render 29.469/26.951 ms
```

Read: this is finally the direct-splat baseline gate on a real multicam bundle.
PRT wins early quality and uses fewer parameters here, but the dense direct
splat renderer is still faster in all D2 render rows and the 200-step direct
baseline trains faster. The absolute PSNR is low, so this is a harness and
directional signal, not a tuned result. The next useful work is better PRT
initialization/loss weighting, a same-wall row, and then a stronger direct
splat render baseline before making a broad speed claim.

Read: this is the first actual video-overfit result for the PRT fork. It is a
good local sanity check for the rasterizer and optimizer path, but it is not yet
the requested full comparison against direct splats or world-camera heldout.

The new idea added in this fork is the curvature-selective hybrid compiler:
low-curvature tubes can stay on the old affine UVT path, while only high-curvature
moving-camera tubes use PRT. That is meant to preserve STAR-UVT's cheap path
instead of forcing every tube through the more expensive rational shader.

The next idea queued after tiled Metal forward is residual-certified footprint
inflation: use a small exact-projection probe to estimate the camera polynomial's
pixel residual and inflate only the PRT support bound needed to cover it. This
should be measured before splitting a camera window.

## Next Gates

1. Decide whether PRT training needs bitwise deterministic gradients or only numeric repeatability.
2. Add tile-load scaling scenes that stress moving-camera curvature beyond the synthetic `camera_motion_scale` knob.
3. Add timing flags for `--uvt-camera-sequence-mode projective_rational`.
4. Decide whether stable depth shortcuts are worth adding or whether sample-level ordering is the right first training path.
5. Add same-wall and stronger-baseline rows for the real multicam direct-splat harness.
