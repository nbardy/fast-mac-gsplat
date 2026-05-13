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
- [x] Gate D2b: corrected-depth fast-mac direct-splat and same-wall rows.
- [x] Gate D2c: repeated render timing probe for PRT tile sizes.
- [x] Gate D2d: PRT forward phase profile separates camera-compiler cost from Metal raster cost.
- [x] Gate D2e: cached-compiled PRT eval timing in the direct-splat compare harness.
- [x] Gate D2f: analytic 2x2 compiler inverse removes the tiny-matrix `torch.linalg.inv` bottleneck.
- [x] Gate D2g: real-multicam PRT train-step breakdown after compiler inverse fix.
- [x] Gate D2h: internal phase timing for `projective_rational_tile_pixel_atomic_backward`.
- [x] Gate D2i: exact `tile_t=1` presorted backward shortcut and D2 timing rows.
- [x] Gate D2j: cached direct-splat compare for `tile_t=1` and 128-tube selector update.
- [x] Gate D2k: 256-tube cached direct-splat compare for `tile_t=1` and selector update.
- [x] Gate D2l: 512-tube capacity correction and `tile_t=1` train/render tradeoff.
- [x] Gate D2m: 1024-tube real-D2 overflow check and fail-closed selector ceiling.
- [x] Gate D2n: rejected 2x2 spatial tile capacity probe for 1024 tubes.
- [x] Gate D2o: 1024-tube alpha-threshold support-shrink sweep.
- [x] Gate D2p: support-only alpha threshold for 1024-tube capacity.
- [x] Gate D2q: 1024-tube support-pruned `tile_t=1` train-speed comparison.
- [x] Gate D2r: lower 1024-tube `tile_t=1` support-pruning cutoff.
- [x] Gate D2s: explicit 1024 train-speed tile policy API.
- [x] Gate D2t: support-aware backward phase profile for the 1024 train-speed policy.
- [x] Gate D2u: reject existing PRT `tile_pair_atomic` as the lower-contention shortcut.
- [x] Gate D2v: profile PRT backward compute-only vs atomic writes.
- [x] Gate D2w: selected 1024 PRT backward replay workload shape.
- [x] Gate D2x: isolate alpha/order replay as the PRT backward cost center.
- [x] Gate D2y: trace-cache memory viability planner.
- [x] Gate D2z: fused MSE train-step backward parity smoke.
- [x] Gate D3a: fused MSE timing on selected 1024 train-speed row.
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
<=128 tubes, normal motion: 8x8x1:128
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

Tiny Metal smoke for the original auto path before D2j selected `tile_t=1` for
128-tube normal-motion cases:

```text
python3 research_project/benchmarks/projective_rational_metal_forward_timing_probe.py --tube-counts 16 --tile-config auto --warmups 0 --repeats 1 --out-json research_project/benchmarks/results/projective_rational_metal_forward_timing_probe_auto_smoke_16t.json
tile_config_key: 8x8x2:128
max abs error vs direct: 5.960464477539062e-07
max tile count: 13
overflow tiles: 0
```

Gate B5c now has a tiny train-step smoke that applies the selector before the
first PRT Metal render. It originally selected `8x8x2:128`; after D2j, live
selector calls at this 128-and-under tier select `8x8x1:128`. The smoke still
passes that selected config into `UVTRenderConfig`, so the selector contract is
exercised on an actual autograd path, not just timing probes.

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
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 32 --max-frames 2 --steps 1 --prt-tubes 16 --splat-count 16 --init-depth 2.0 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_32_2f_16t_16s_1step_smoke.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --splat-count 128 --init-depth 2.0 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_20step.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 128 --splat-count 128 --init-depth 2.0 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_200step.json
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

Gate D2b fixes the main setup weakness in D2: the `--init-depth 2.0` value was
bad for this DeepView dog bundle. The direct-splat baseline's own gauge-field
default is `0.5`, and the cheap 20-step sweep showed that `0.5` is materially
better for both PRT and direct splats. The harness default is now `0.5`.

Corrected-depth commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_20step_depth0p5_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_200step_depth0p5_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_72step_depth0p5_prt_samewall_fastmacsplat200.json
```

Result:

```text
20 steps, depth 0.5: PRT train/heldout PSNR 14.9283/14.3764 dB, train wall 1.485 s, render 55.564/62.950 ms, max tile count 99, overflow 0; fast-mac direct splats 8.6144/7.9594 dB, train wall 0.582 s, render 28.179/23.775 ms
200 steps, depth 0.5: PRT train/heldout PSNR 16.1345/13.4165 dB, train wall 16.903 s, render 71.362/72.718 ms, max tile count 55, overflow 0; fast-mac direct splats 13.2901/11.2841 dB, train wall 5.313 s, render 35.620/29.694 ms
72-step PRT same-wall row, depth 0.5: PRT train/heldout PSNR 16.1552/14.1770 dB, train wall 5.285 s, render 46.328/53.304 ms, max tile count 79, overflow 0. This matches the 200-step fast-mac direct-splat train budget at 5.313 s while keeping higher train and heldout PSNR.
```

Read: after correcting depth, the quality result is no longer a low-PSNR
curiosity. At matched train wall-clock, PRT reaches 16.1552/14.1770 dB with
1408 parameters while fast-mac direct splats reach 13.2901/11.2841 dB with
7168 parameters. The speed warning remains: direct splats still render faster
than PRT in this harness, roughly 30-36 ms versus 46-73 ms depending on row.
The next rasterizer work should target render latency, not just optimization
quality.

Gate D2c adds explicit render warmup/repeat controls to
`projective_rational_multicam_splat_compare.py`:
`--render-warmups` and `--render-repeats`. This was necessary because the
single-sample D2b render timings were too noisy for tile selection.

Repeated timing commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --render-warmups 1 --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_20step_depth0p5_tile8x8x2_cap128_repeat5_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:128 --render-warmups 1 --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_20step_depth0p5_tile4x4x2_cap128_repeat5_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --tile-config 16x16x2:128 --render-warmups 1 --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_20step_depth0p5_tile16x16x2_cap128_repeat5_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --render-warmups 1 --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_72step_depth0p5_tile8x8x2_cap128_repeat5_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --tile-config 16x16x2:128 --render-warmups 1 --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_72step_depth0p5_tile16x16x2_cap128_repeat5_fastmacsplat.json
```

Result:

```text
20-step repeated timing: 8x8x2 PRT render median 56.611/53.476 ms train/heldout; 4x4x2 55.645/61.982 ms; 16x16x2 51.682/44.100 ms. Direct fast-mac splat render medians stay near 24-31 ms.
72-step repeated timing: 8x8x2 PRT train/heldout PSNR 16.1264/14.1754 dB, train wall 6.071 s, render 63.360/50.699 ms; 16x16x2 PRT 16.1554/14.2860 dB, train wall 7.292 s, render 62.826/66.183 ms. Direct fast-mac splat render medians stay near 29-32 ms.
```

Read: no selector change. A larger `16x16x2:128` tile looks faster at the
20-step checkpoint, but it is not a robust win at the 72-step same-wall point
and it slows training. This points away from a pure tile-size fix. The next
rasterizer question is kernel-level: separate tile assignment, sort/fill,
shade/blend, and backward timing instead of only sweeping tile geometry.

Gate D2d adds a profile-only op,
`profile_projective_rational_tubes_tiled`, and
`research_project/benchmarks/projective_rational_multicam_phase_profile.py`.
The production render op is unchanged. The profile op returns normal image and
tile diagnostics plus phase timings for allocation, clear, bin, render, and
total. The benchmark also times `_compile_detached_footprint` separately, since
the earlier D2 "render" timers included camera-path coefficient compilation on
every measured call.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_phase_profile.py --target-size 32 --max-frames 2 --steps 1 --prt-tubes 16 --init-depth 0.5 --render-warmups 0 --render-repeats 1 --out-json research_project/benchmarks/results/projective_rational_multicam_phase_profile_32_2f_16t_1step_smoke.json
python3 research_project/benchmarks/projective_rational_multicam_phase_profile.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --init-depth 0.5 --tile-config 8x8x2:128 --render-warmups 1 --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_phase_profile_64_4f_128t_20step_depth0p5_tile8x8x2_cap128_repeat5.json
python3 research_project/benchmarks/projective_rational_multicam_phase_profile.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --init-depth 0.5 --tile-config 16x16x2:128 --render-warmups 1 --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_phase_profile_64_4f_128t_20step_depth0p5_tile16x16x2_cap128_repeat5.json
python3 research_project/benchmarks/projective_rational_multicam_phase_profile.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --init-depth 0.5 --tile-config 8x8x2:128 --render-warmups 1 --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_phase_profile_64_4f_128t_72step_depth0p5_tile8x8x2_cap128_repeat5.json
python3 research_project/benchmarks/projective_rational_multicam_phase_profile.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --init-depth 0.5 --tile-config 16x16x2:128 --render-warmups 1 --render-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_phase_profile_64_4f_128t_72step_depth0p5_tile16x16x2_cap128_repeat5.json
```

Result:

```text
20-step 8x8x2: compile median 29.523/35.872 ms train/heldout; profiled Metal total 8.677/14.457 ms; render_tiles share 94.0%/95.1%; max tile count 96/95; overflow 0
20-step 16x16x2: compile median 37.290/42.546 ms; profiled Metal total 16.184/8.493 ms; render_tiles share 97.4%/95.7%; max tile count 99/99; overflow 0
72-step 8x8x2: compile median 45.928/40.542 ms; profiled Metal total 9.455/10.302 ms; render_tiles share 93.5%/93.9%; max tile count 69/78; overflow 0
72-step 16x16x2: compile median 39.154/32.989 ms; profiled Metal total 5.693/13.272 ms; render_tiles share 92.8%/96.3%; max tile count 74/79; overflow 0
```

Read: the D2 render slowdown is not primarily tile assignment. Clear plus bin is
sub-millisecond in the 64px/4-frame/128-tube rows, while the tiled Metal op is
dominated by the render/shade kernel. More importantly, the full D2 eval timing
was paying tens of milliseconds to compile world tubes into camera-space PRT
coefficients for each camera render. The next speed pass should therefore split
the problem: cache or fuse the camera compiler where the camera path is fixed,
and optimize `render_projective_rational_tiles` if working inside the Metal
rasterizer. More tile-size sweeping alone is unlikely to close the direct-splat
render gap.

Gate D2e adds `--prt-eval-cache-compiled` to
`projective_rational_multicam_splat_compare.py`. With this flag, the benchmark
compiles the PRT footprint once per eval camera after training and times only
the tiled rasterizer in the render-repeat loop. The output records the one-time
compile timing separately as `projective_rational.eval.compile_seconds`.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 32 --max-frames 2 --steps 1 --prt-tubes 16 --splat-count 16 --splat-renderer fast_mac --init-depth 0.5 --render-warmups 0 --render-repeats 1 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_32_2f_16t_16s_1step_depth0p5_cachedprt_fastmacsplat_smoke.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_20step_depth0p5_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_72step_depth0p5_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_200step_depth0p5_cachedprt_fastmacsplat.json
```

Result:

```text
20 steps cached PRT before D2f: train/heldout PSNR 14.9275/14.3763 dB, train wall 1.303 s, raster render 10.038/10.708 ms, compile median 39.079 ms, max tile 99, overflow 0; fast-mac direct splats 8.6144/7.9594 dB, train wall 0.604 s, render 20.490/21.080 ms.
72 steps cached PRT before D2f: 16.1283/14.1625 dB, train wall 4.816 s, raster render 4.474/4.972 ms, compile median 35.419 ms, max tile 77, overflow 0; same-run 72-step fast-mac direct splats 9.8891/8.9967 dB, train wall 1.719 s, render 22.934/30.506 ms.
200 steps cached PRT before D2f: 16.7678/14.1060 dB, train wall 14.329 s, raster render 6.987/8.300 ms, compile median 67.059 ms, max tile 56, overflow 0; fast-mac direct splats 13.2900/11.2843 dB, train wall 4.511 s, render 22.337/22.597 ms.
```

Read: when the camera-space PRT footprint is cached, the actual tiled PRT
rasterizer is faster than the fast-mac direct-splat eval path in these D2 rows.
That rescues the original rasterizer speed thesis, but with an important
condition: the current end-to-end eval path is only fast if it does not recompile
world tubes for every timed render call. The next practical speed task is to
make this cache/fuse boundary first-class for camera-path playback and for
camera-edit bake, then decide whether `render_projective_rational_tiles` itself
needs optimization.

Gate D2f replaces the compiler's batched `torch.linalg.inv` over tiny 2x2
covariance matrices with the explicit 2x2 inverse formula. The old and new
lambda-UV compiler outputs matched exactly in a CPU and MPS parity check on the
projection-audit scene:

```text
cpu max abs lambda_uv diff vs old torch.linalg.inv path: 0.0
mps max abs lambda_uv diff vs old torch.linalg.inv path: 0.0
```

Validation:

```text
python3 tests/projective_rational_gate_check.py
python3 tests/projective_rational_tiled_render_check.py
```

Result after rerunning the same cached D2e JSONs:

```text
20 steps cached PRT after D2f: train/heldout PSNR 14.9276/14.3762 dB, train wall 0.631 s, raster render 5.027/5.400 ms, compile median 2.153 ms, max tile 99, overflow 0; fast-mac direct splats 8.6144/7.9594 dB, train wall 0.612 s, render 24.310/24.930 ms.
72 steps cached PRT after D2f: 16.0546/14.1108 dB, train wall 2.259 s, raster render 10.349/11.802 ms, compile median 3.322 ms, max tile 79, overflow 0; same-run fast-mac direct splats 9.8891/8.9967 dB, train wall 1.980 s, render 29.472/32.655 ms.
200 steps cached PRT after D2f: 17.0640/13.8394 dB, train wall 5.755 s, raster render 6.421/7.333 ms, compile median 3.195 ms, max tile 55, overflow 0; fast-mac direct splats 13.2901/11.2841 dB, train wall 4.526 s, render 20.627/20.103 ms.
```

Read: the compiler bottleneck was mostly a bad primitive choice, not an
unavoidable camera-compiler cost. With the analytic inverse, cached PRT eval
still renders faster than fast-mac direct splats, and PRT training wall-clock is
now close to the direct-splat baseline while keeping materially higher train and
heldout PSNR in the D2 rows. The next speed question moves back to training:
whether compile can be reused or simplified inside repeated train steps, and
whether backward phase timing shows another tiny-kernel bottleneck.

Gate D2g adds `projective_rational_multicam_train_breakdown.py`, a diagnostic
sync-boundary train-step profiler for the same real multicam D2 PRT path. It
times sampling, zero-grad, PRT compile, tiled forward, loss, backward, optimizer,
and full step wall time. These rows are diagnostic because each segment
synchronizes; they are not a replacement for the normal unsplit train wall.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 32 --max-frames 2 --steps 1 --prt-tubes 16 --init-depth 0.5 --render-warmups 0 --render-repeats 1 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_32_2f_16t_1step_smoke.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --init-depth 0.5 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_64_4f_128t_20step_depth0p5.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --init-depth 0.5 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_64_4f_128t_72step_depth0p5.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 128 --init-depth 0.5 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_64_4f_128t_200step_depth0p5.json
```

Result:

```text
20-step median diagnostic step: total 21.768 ms, backward 12.659 ms (58.2%), forward 5.229 ms (24.0%), compile 1.753 ms (8.1%); eval PSNR 14.9276/14.3763 dB, cached render 4.886/4.634 ms.
72-step median diagnostic step: total 32.353 ms, backward 18.298 ms (56.6%), forward 7.041 ms (21.8%), compile 3.839 ms (11.9%); eval PSNR 16.1156/14.1326 dB, cached render 11.608/12.147 ms.
200-step median diagnostic step: total 25.792 ms, backward 14.561 ms (56.5%), forward 6.122 ms (23.7%), compile 2.532 ms (9.8%); eval PSNR 16.2801/13.4639 dB, cached render 6.107/6.682 ms.
```

Read: after D2f, compiler cost is no longer the train-step bottleneck. The real
D2 PRT training path is now dominated by backward, then forward rasterization.
The next speed pass should profile or split `projective_rational_tile_pixel_atomic_backward`
instead of further optimizing the camera compiler.

Gate D2h adds a profiled sibling of `projective_rational_tile_pixel_atomic_backward`
that runs the same clear, bin, gradient-clear, and backward kernels while
synchronizing after each phase. It returns the same gradients plus tile counts,
overflow, unstable tiles, and a CPU timing tensor. The normal autograd path is
unchanged.

Validation:

```text
python3 tests/projective_rational_tile_pixel_atomic_backward_check.py
python3 tests/projective_rational_metal_autograd_smoke.py
```

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 32 --max-frames 2 --steps 1 --prt-tubes 16 --init-depth 0.5 --profile-warmups 0 --profile-repeats 1 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_32_2f_16t_1step_smoke.json
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --init-depth 0.5 --profile-warmups 1 --profile-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_128t_20step_depth0p5.json
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --init-depth 0.5 --profile-warmups 1 --profile-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_128t_72step_depth0p5.json
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 128 --init-depth 0.5 --profile-warmups 1 --profile-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_128t_200step_depth0p5.json
```

Result:

```text
20-step profiled backward: total 9.087 ms, backward kernel 8.514 ms (93.7%), bin 0.258 ms, clear tiles 0.164 ms, clear grads 0.168 ms, max tile 96, overflow 0.
72-step profiled backward: total 14.869 ms, backward kernel 13.968 ms (93.9%), bin 0.451 ms, clear tiles 0.225 ms, clear grads 0.227 ms, max tile 70, overflow 0.
200-step profiled backward: total 8.655 ms, backward kernel 7.678 ms (88.7%), bin 0.532 ms, clear tiles 0.226 ms, clear grads 0.252 ms, max tile 46, overflow 0.
```

Read: D2g's backward bottleneck is not hiding in tile allocation, clearing, or
binning. On the real D2 rows, the pixel-atomic backward shader itself accounts
for roughly 89-94% of profiled backward wall. The next useful speed work is
inside `projective_rational_tile_pixel_atomic_backward`: reduce per-pixel
sample ordering/recomputation, or test a different accumulation structure, not
more host-side phase splitting.

Gate D2i adds the first inner-kernel shortcut: when `STAR_TILE_T == 1`, each
tile covers exactly one frame, and PRT depth is independent of pixel position.
The binned tile-depth sort is therefore already the exact per-pixel sample
order, so `projective_rational_tile_pixel_atomic_backward` can skip the
O(count^2) per-pixel reselect loop and replay the sorted tile IDs directly.

Validation:

```text
python3 tests/projective_rational_tile_pixel_atomic_backward_check.py
STAR_UVT_TILE_T=1 python3 tests/projective_rational_tile_pixel_atomic_backward_check.py
python3 tests/projective_rational_metal_autograd_smoke.py
```

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --init-depth 0.5 --tile-config 8x8x1:128 --profile-warmups 1 --profile-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_128t_72step_depth0p5_tile8x8x1_before_presort.json
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --init-depth 0.5 --tile-config 8x8x1:128 --profile-warmups 1 --profile-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_128t_20step_depth0p5_tile8x8x1_presorted.json
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --init-depth 0.5 --tile-config 8x8x1:128 --profile-warmups 1 --profile-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_128t_72step_depth0p5_tile8x8x1_presorted.json
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 128 --init-depth 0.5 --tile-config 8x8x1:128 --profile-warmups 1 --profile-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_128t_200step_depth0p5_tile8x8x1_presorted.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --init-depth 0.5 --tile-config 8x8x1:128 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_64_4f_128t_20step_depth0p5_tile8x8x1_presorted.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --init-depth 0.5 --tile-config 8x8x1:128 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_64_4f_128t_72step_depth0p5_tile8x8x1_presorted.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 128 --init-depth 0.5 --tile-config 8x8x1:128 --render-warmups 1 --render-repeats 3 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_64_4f_128t_200step_depth0p5_tile8x8x1_presorted.json
```

Result:

```text
72-step tile8x8x1 before shortcut: profiled backward total 14.669 ms, backward kernel 12.692 ms, bin 0.960 ms, max tile 62, overflow 0.
20-step tile8x8x1 after shortcut: profiled backward total 13.743 ms, backward kernel 12.168 ms, bin 0.882 ms; train step 25.416 ms, backward 10.991 ms, forward 10.086 ms, PSNR 14.9277/14.3763 dB.
72-step tile8x8x1 after shortcut: profiled backward total 8.148 ms, backward kernel 6.896 ms, bin 0.764 ms; train step 30.107 ms, backward 13.116 ms, forward 8.987 ms, PSNR 16.1678/14.1779 dB.
200-step tile8x8x1 after shortcut: profiled backward total 5.926 ms, backward kernel 4.423 ms, bin 0.851 ms; train step 23.841 ms, backward 10.609 ms, forward 6.871 ms, PSNR 16.5397/14.0461 dB.
```

Read: the diagnostic sync-boundary profiler says the presorted `tile_t=1` path
is a real inner-kernel win once supports shrink. Against the default D2g
`8x8x2:128` train-breakdown rows, it improves the 72-step median train step from
32.353 ms to 30.107 ms and the 200-step median train step from 25.792 ms to
23.841 ms. It is still worse in the 20-step diagnostic breakdown because
`tile_t=1` doubles tile/bin and forward work while early supports are broad.

Tried and rejected in this gate: extending the fast path to all `count == 1`
tiles under `tile_t=2`. It remained numerically valid but made the default D2
train-breakdown rows slower, so that change was dropped before commit.

Gate D2j reruns the full cached direct-splat comparison with
`--tile-config 8x8x1:128`. This checks the actual comparison harness instead of
only the diagnostic train-breakdown profiler: same 20/72/200 step counts, same
fast-mac direct-splat baseline, cached PRT eval render timing, and the same real
D2 multicam train/heldout split.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --tile-config 8x8x1:128 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_20step_depth0p5_tile8x8x1_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --tile-config 8x8x1:128 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_72step_depth0p5_tile8x8x1_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 128 --splat-count 128 --splat-renderer fast_mac --init-depth 0.5 --tile-config 8x8x1:128 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_128t_128s_200step_depth0p5_tile8x8x1_cachedprt_fastmacsplat.json
```

Result:

```text
20-step tile8x8x1 cached compare: PRT PSNR 14.9283/14.3765 dB, train wall 0.403 s, cached render 4.236/3.958 ms, compile 1.448 ms, max tile 90, overflow 0; fast-mac direct splats 8.6144/7.9594 dB, train wall 1.377 s, render 18.971/20.297 ms.
72-step tile8x8x1 cached compare: PRT PSNR 16.1872/14.1126 dB, train wall 2.173 s, cached render 8.725/8.975 ms, compile 2.414 ms, max tile 69, overflow 0; fast-mac direct splats 9.8891/8.9967 dB, train wall 1.727 s, render 24.248/24.431 ms.
200-step tile8x8x1 cached compare: PRT PSNR 16.7240/13.6577 dB, train wall 4.961 s, cached render 5.559/6.123 ms, compile 2.513 ms, max tile 51, overflow 0; fast-mac direct splats 13.2900/11.2843 dB, train wall 5.153 s, render 19.644/20.600 ms.
```

Read: on the actual direct-splat comparison harness, `8x8x1:128` is faster than
the previous `8x8x2:128` PRT rows for normal train wall and cached eval render
at 20/72/200 steps, while preserving the train/heldout quality lead over the
fast-mac direct-splat baseline. This is enough to update the verified selector
for `tube_count <= 128` normal-motion PRT rows to `8x8x1:128`. Higher tube
counts stay on the older `tile_t=2` heuristic until measured.

Gate D2k repeats the cached direct-splat comparison at 256 PRT tubes and 256
direct splats. This checks whether the exact `tile_t=1` presorted backward path
still pays after doubling tube count, using the same real D2 DeepView dog
multicam train/heldout split and cached PRT eval timing.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 256 --splat-count 256 --splat-renderer fast_mac --init-depth 0.5 --tile-config 8x8x2:256 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_256t_256s_20step_depth0p5_tile8x8x2_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 256 --splat-count 256 --splat-renderer fast_mac --init-depth 0.5 --tile-config 8x8x1:256 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_256t_256s_20step_depth0p5_tile8x8x1_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 256 --splat-count 256 --splat-renderer fast_mac --init-depth 0.5 --tile-config 8x8x2:256 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_256t_256s_72step_depth0p5_tile8x8x2_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 256 --splat-count 256 --splat-renderer fast_mac --init-depth 0.5 --tile-config 8x8x1:256 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_256t_256s_72step_depth0p5_tile8x8x1_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 256 --splat-count 256 --splat-renderer fast_mac --init-depth 0.5 --tile-config 8x8x2:256 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_256t_256s_200step_depth0p5_tile8x8x2_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 256 --splat-count 256 --splat-renderer fast_mac --init-depth 0.5 --tile-config 8x8x1:256 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_256t_256s_200step_depth0p5_tile8x8x1_cachedprt_fastmacsplat.json
```

Result:

```text
20-step tile8x8x2: PRT PSNR 14.3690/14.1503 dB, train wall 1.791 s, cached render 16.888/16.336 ms, compile 2.480 ms, max tile 200, overflow 0; fast-mac direct splats 12.2818/10.1738 dB, train wall 0.382 s, render 15.431/17.741 ms.
20-step tile8x8x1: PRT PSNR 14.3948/14.1748 dB, train wall 0.866 s, cached render 14.365/15.077 ms, compile 4.004 ms, max tile 183, overflow 0; fast-mac direct splats 12.2818/10.1738 dB, train wall 0.592 s, render 28.808/19.661 ms.
72-step tile8x8x2: PRT PSNR 15.6980/14.2926 dB, train wall 4.425 s, cached render 7.099/7.602 ms, compile 2.609 ms, max tile 139, overflow 0; fast-mac direct splats 13.4828/11.1775 dB, train wall 2.543 s, render 30.719/31.510 ms.
72-step tile8x8x1: PRT PSNR 15.8483/14.3294 dB, train wall 3.009 s, cached render 8.987/10.240 ms, compile 2.056 ms, max tile 127, overflow 0; fast-mac direct splats 13.4828/11.1775 dB, train wall 1.919 s, render 31.584/29.509 ms.
200-step tile8x8x2: PRT PSNR 17.2357/13.6158 dB, train wall 9.373 s, cached render 13.058/10.298 ms, compile 4.150 ms, max tile 99, overflow 0; fast-mac direct splats 16.2209/12.2689 dB, train wall 5.550 s, render 29.345/32.799 ms.
200-step tile8x8x1: PRT PSNR 17.3049/13.9043 dB, train wall 4.826 s, cached render 9.399/12.411 ms, compile 2.957 ms, max tile 87, overflow 0; fast-mac direct splats 16.2209/12.2689 dB, train wall 4.288 s, render 32.563/34.390 ms.
```

Read: `8x8x1:256` is the better normal-motion training policy at 256 tubes.
It cuts PRT train wall at every checked step count and slightly improves both
train and heldout PSNR. Cached eval render is mixed by view and step count
because `tile_t=1` doubles the temporal tile grid, but the PRT rows remain
substantially faster than direct dynamic splats in the 72/200-step render
checks. I updated the verified selector for `tube_count <= 256` normal-motion
PRT rows to `8x8x1:256`; 512+ tubes stay on the older heuristic until measured.

Gate D2l measures the 512-tube selector tier on the same cached real-D2
direct-splat comparison. The nominal 512 selector from older synthetic timing
rows, `4x4x2:256`, is not valid on this real multicam row: it overflows at
20 steps. The gate therefore checks the capacity correction first, then compares
the valid `tile_t=2` and `tile_t=1` capacity-512 policies.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 512 --splat-count 512 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:256 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_512t_512s_20step_depth0p5_tile4x4x2_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 512 --splat-count 512 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:256 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_512t_512s_20step_depth0p5_tile4x4x1cap256_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 512 --splat-count 512 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_512t_512s_20step_depth0p5_tile4x4x2cap512_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 512 --splat-count 512 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_512t_512s_20step_depth0p5_tile4x4x1cap512_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 512 --splat-count 512 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_512t_512s_72step_depth0p5_tile4x4x2cap512_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 512 --splat-count 512 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_512t_512s_72step_depth0p5_tile4x4x1cap512_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 512 --splat-count 512 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_512t_512s_200step_depth0p5_tile4x4x2cap512_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 512 --splat-count 512 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --render-warmups 1 --render-repeats 5 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_512t_512s_200step_depth0p5_tile4x4x1cap512_cachedprt_fastmacsplat.json
```

Result:

```text
20-step tile4x4x2:256: pass false, max tile 403, overflow 772, PRT PSNR 13.9205/14.1479 dB, train wall 3.722 s, cached render 36.542/36.224 ms; direct splats 15.1559/11.4593 dB, train wall 1.018 s, render 31.707/33.061 ms.
20-step tile4x4x1:256: pass false, max tile 370, overflow 1123, PRT PSNR 13.6709/14.1018 dB, train wall 2.678 s, cached render 55.127/54.062 ms; direct splats 15.1559/11.4593 dB, train wall 1.109 s, render 25.237/30.853 ms.
20-step tile4x4x2:512: pass true, max tile 409, overflow 0, PRT PSNR 13.6582/13.9660 dB, train wall 5.793 s, cached render 55.417/51.798 ms; direct splats 15.1559/11.4593 dB, train wall 1.031 s, render 23.607/23.158 ms.
20-step tile4x4x1:512: pass true, max tile 382, overflow 0, PRT PSNR 13.6736/13.9672 dB, train wall 3.334 s, cached render 71.067/69.025 ms; direct splats 15.1559/11.4593 dB, train wall 1.060 s, render 25.278/31.018 ms.
72-step tile4x4x2:512: pass true, max tile 296, overflow 0, PRT PSNR 15.6340/14.3135 dB, train wall 14.528 s, cached render 32.762/36.956 ms; direct splats 15.3450/12.0576 dB, train wall 2.068 s, render 23.690/32.892 ms.
72-step tile4x4x1:512: pass true, max tile 275, overflow 0, PRT PSNR 15.5902/14.3548 dB, train wall 7.406 s, cached render 38.740/42.865 ms; direct splats 15.3450/12.0576 dB, train wall 2.055 s, render 22.158/21.454 ms.
200-step tile4x4x2:512: pass true, max tile 175, overflow 0, PRT PSNR 16.9642/14.1056 dB, train wall 23.148 s, cached render 11.189/13.588 ms; direct splats 17.7169/12.5350 dB, train wall 5.134 s, render 24.834/31.812 ms.
200-step tile4x4x1:512: pass true, max tile 181, overflow 0, PRT PSNR 16.6867/14.2889 dB, train wall 12.996 s, cached render 15.707/21.905 ms; direct splats 17.7169/12.5350 dB, train wall 5.550 s, render 28.778/21.441 ms.
```

Read: the immediate correction is capacity, not `tile_t`. The current
`4x4x2:256` selector is invalid for the real 512-tube D2 row, so the verified
generic selector must move to `4x4x2:512`. The `4x4x1:512` path is a real
train-speed option, cutting PRT train wall from 14.528 s to 7.406 s at 72 steps
and from 23.148 s to 12.996 s at 200 steps, with slightly better heldout PSNR.
But it is slower to render than `4x4x2:512` at every measured step count and
has lower train PSNR at 72/200 steps. Since the same selector currently feeds
training and eval/playback-style probes, the conservative default is
`4x4x2:512`; `4x4x1:512` should become an explicit train-speed or split-policy
choice if we want that optimization.

Gate D2m tests the remaining advertised 1024-tube selector tier on the same
real D2 multicam compare. This is only a 20-step validity probe; it is not a
quality or speed matrix. Both capacity-512 variants overflow, so 1024 tubes is
not verified on this real row.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x1cap512_cachedprt_fastmacsplat.json
```

Result:

```text
20-step tile4x4x2:512: pass false, max tile 835, overflow 841, PRT PSNR 13.4417/14.1006 dB, train wall 15.216 s, cached render 101.945/114.930 ms.
20-step tile4x4x1:512: pass false, max tile 784, overflow 1260, PRT PSNR 13.2017/14.1768 dB, train wall 6.126 s, cached render 135.066/134.430 ms.
```

Read: neither `tile_t=2` nor `tile_t=1` can make 1024 tubes valid with the
current 4x4 spatial tiles and max capacity 512. The selector now fails closed
above 512 tubes unless `allow_unverified=True` is explicit. To re-enable 1024,
we need a real capacity strategy: residual-certified footprint tightening,
camera-window segmentation, smaller effective support, or a different
accumulation/bucketing path.

Gate D2n temporarily allowed 2x2 spatial tiles in Python and C++ validation,
smoked the tile-pixel backward path, and then tested whether smaller spatial
tiles solve the 1024-tube capacity failure. The validator change was not kept:
2x2 did not solve overflow and made render substantially slower.

Validation during the temporary patch:

```text
STAR_UVT_TILE_X=2 STAR_UVT_TILE_Y=2 STAR_UVT_TILE_T=2 STAR_UVT_TILE_CAPACITY=512 python3 tests/projective_rational_tile_pixel_atomic_backward_check.py
```

Result:

```text
2x2x2 backward smoke: pass true, max abs error 1.1176e-07, profile overflow tile count 0.
20-step tile2x2x2:512: pass false, max tile 816, overflow 3007, PRT PSNR 13.3239/13.9631 dB, train wall 18.646 s, cached render 279.280/299.570 ms.
20-step tile2x2x1:512: pass false, max tile 783, overflow 4527, PRT PSNR 13.3459/14.1926 dB, train wall 12.485 s, cached render 399.184/404.706 ms.
```

Read: smaller spatial tiles are not the 1024 capacity strategy. They lower the
peak count only marginally versus 4x4 (`835` to `816` for `tile_t=2`, `784` to
`783` for `tile_t=1`), increase the number of overflowed tiles, and are far
slower to render. The useful next path is not more spatial subdivision; it is
support shrinkage, camera-window segmentation, or an accumulation path that
does not require all overlapping tubes to fit in one fixed tile list.

Gate D2o adds a benchmark-only `--prt-alpha-threshold` knob to the D2 compare
harness and sweeps support shrinkage for the 1024-tube `4x4x2:512` row. The
selector remains fail-closed above 512 tubes; this gate is evidence about a
possible capacity strategy, not a default policy.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.01568627450980392 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_alpha4over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.03137254901960784 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_alpha8over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.06274509803921569 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_alpha16over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.09411764705882353 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_alpha24over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.10980392156862745 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_alpha28over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.11764705882352941 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_alpha30over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.12549019607843137 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_alpha32over255_cachedprt_fastmacsplat.json
```

Result:

```text
1/255:  pass false, max tile 835, overflow 841, PRT PSNR 13.4417/14.1006 dB, train wall 15.216 s, cached render 101.945/114.930 ms.
4/255:  pass false, max tile 780, overflow 493, PRT PSNR 13.1986/14.3016 dB, train wall 11.621 s, cached render 73.673/67.319 ms.
8/255:  pass false, max tile 747, overflow 340, PRT PSNR 13.4973/14.3389 dB, train wall 9.227 s, cached render 62.210/54.136 ms.
16/255: pass false, max tile 676, overflow 148, PRT PSNR 13.6989/14.3288 dB, train wall 6.741 s, cached render 47.186/38.484 ms.
24/255: pass false, max tile 556, overflow 37, PRT PSNR 13.8650/14.4107 dB, train wall 4.828 s, cached render 36.561/31.872 ms.
28/255: pass false, max tile 524, overflow 3, PRT PSNR 14.0692/14.1969 dB, train wall 4.018 s, cached render 32.691/27.306 ms.
30/255: pass true, max tile 484, overflow 0, PRT PSNR 14.0641/14.3129 dB, train wall 3.584 s, cached render 29.703/25.487 ms.
32/255: pass true, max tile 459, overflow 0, PRT PSNR 14.0899/14.0990 dB, train wall 3.326 s, cached render 28.022/24.801 ms.
```

Read: threshold-only support shrinkage can clear 1024 tubes, but only at an
aggressive cutoff around `30/255`. The result is encouraging as a capacity
direction because render time drops by roughly 4x versus the default-threshold
overflow row, but it changes the renderer's alpha semantics and should not
re-enable the automatic 1024 selector by itself. The next useful test is a
separate support-threshold or support-margin parameter that shrinks binning
without changing final alpha compositing, then checks parity/PSNR.

Gate D2p adds that separate support-only threshold. `UVTRenderConfig` now has
`support_alpha_threshold`; it defaults to `alpha_threshold`, so default behavior
is unchanged. The PRT bin kernel uses `support_alpha_threshold` only for tile
support bounds, while final alpha compositing still uses `alpha_threshold`.
The D2 compare harness exposes this as `--prt-support-alpha-threshold`.

Validation:

```text
python3 tests/projective_rational_tiled_render_check.py
python3 tests/projective_rational_tile_pixel_atomic_backward_check.py
python3 tests/projective_rational_metal_autograd_smoke.py
python3 -m py_compile research_project/benchmarks/projective_rational_multicam_splat_compare.py
```

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.11764705882352941 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_supportalpha30over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.12549019607843137 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_supportalpha32over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.13333333333333333 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_supportalpha34over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.13333333333333333 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_72step_depth0p5_tile4x4x2cap512_supportalpha34over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.1568627450980392 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x2cap512_supportalpha40over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.1568627450980392 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_72step_depth0p5_tile4x4x2cap512_supportalpha40over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x2:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.1568627450980392 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_200step_depth0p5_tile4x4x2cap512_supportalpha40over255_cachedprt_fastmacsplat.json
```

Result:

```text
20-step support 30/255: pass false, max tile 527, overflow 5, PRT PSNR 13.8867/14.3840 dB, train wall 4.342 s, cached render 21.492/20.796 ms.
20-step support 32/255: pass false, max tile 525, overflow 2, PRT PSNR 13.9234/14.3465 dB, train wall 3.185 s, cached render 19.004/18.380 ms.
20-step support 34/255: pass true, max tile 473, overflow 0, PRT PSNR 13.8645/14.2535 dB, train wall 3.015 s, cached render 17.993/17.380 ms.
72-step support 34/255: pass false, max tile 524, overflow 5, PRT PSNR 14.9836/14.0353 dB, train wall 11.357 s, cached render 16.570/16.588 ms.
20-step support 40/255: pass true, max tile 424, overflow 0, PRT PSNR 14.1372/14.2095 dB, train wall 2.268 s, cached render 14.779/13.233 ms.
72-step support 40/255: pass true, max tile 469, overflow 0, PRT PSNR 15.1541/14.1680 dB, train wall 8.821 s, cached render 14.065/14.100 ms.
200-step support 40/255: pass true, max tile 390, overflow 0, PRT PSNR 16.2110/13.9342 dB, train wall 22.892 s, cached render 12.958/12.441 ms; direct splats PSNR 18.1577/12.5995 dB, train wall 5.115 s, render 32.453/45.681 ms.
```

Read: support-only pruning works without changing the final alpha cutoff, but
the passing threshold has to be stronger than the render-alpha sweep because
the renderer still evaluates lower-alpha tails once a tube is binned. A support
threshold of `40/255` passes the 20/72/200 1024-tube rows and gives fast PRT
renders; the 200-step row keeps a heldout quality lead over direct splats
(`13.9342` vs `12.5995` dB) while underfitting train relative to the direct
splat baseline. This is a real 1024 capacity path, but it is not ready to make
the generic auto selector support 1024 until we define the policy surface:
default render fidelity, train-speed mode, support pruning schedule, or
capacity fallback.

Gate D2q compares the same 1024-tube support-pruned row with `tile_t=1`.
This tests whether the exact presorted backward shortcut remains useful in the
1024 support-pruned regime.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.1568627450980392 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x1cap512_supportalpha40over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.1568627450980392 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_72step_depth0p5_tile4x4x1cap512_supportalpha40over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.1568627450980392 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_200step_depth0p5_tile4x4x1cap512_supportalpha40over255_cachedprt_fastmacsplat.json
```

Result:

```text
20-step tile4x4x2:512 support 40/255: pass true, max tile 424, overflow 0, PRT PSNR 14.1372/14.2095 dB, train wall 2.268 s, cached render 14.779/13.233 ms.
20-step tile4x4x1:512 support 40/255: pass true, max tile 365, overflow 0, PRT PSNR 14.0347/14.4656 dB, train wall 1.629 s, cached render 13.493/13.675 ms.
72-step tile4x4x2:512 support 40/255: pass true, max tile 469, overflow 0, PRT PSNR 15.1541/14.1680 dB, train wall 8.821 s, cached render 14.065/14.100 ms.
72-step tile4x4x1:512 support 40/255: pass true, max tile 404, overflow 0, PRT PSNR 14.9909/14.2664 dB, train wall 2.815 s, cached render 14.292/13.553 ms.
200-step tile4x4x2:512 support 40/255: pass true, max tile 390, overflow 0, PRT PSNR 16.2110/13.9342 dB, train wall 22.892 s, cached render 12.958/12.441 ms.
200-step tile4x4x1:512 support 40/255: pass true, max tile 328, overflow 0, PRT PSNR 16.5509/13.7676 dB, train wall 8.875 s, cached render 18.117/14.311 ms; direct splats PSNR 18.1577/12.5995 dB, train wall 5.578 s, render 24.810/25.918 ms.
```

Read: `tile_t=1` is the better 1024 support-pruned training policy, cutting
PRT train wall by 28% at 20 steps, 68% at 72 steps, and 61% at 200 steps while
keeping zero overflow and a heldout PSNR lead over direct splats. It is not a
pure render win: at 200 steps, `tile_t=2` renders faster and has better heldout
PSNR, while `tile_t=1` has better train PSNR and much lower train wall. This
reinforces that PRT needs an explicit policy surface instead of one generic
selector: train-speed can prefer `tile_t=1` plus support pruning, while
fidelity/playback can prefer the more conservative `tile_t=2` support-pruned
path until a schedule or split policy is chosen.

Gate D2r lowers the `tile_t=1` support-only threshold to find the actual 1024
capacity cutoff instead of carrying forward the conservative `40/255` row.

Commands:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.10980392156862745 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_72step_depth0p5_tile4x4x1cap512_supportalpha28over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.11764705882352941 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_72step_depth0p5_tile4x4x1cap512_supportalpha30over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.12549019607843137 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_72step_depth0p5_tile4x4x1cap512_supportalpha32over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.13333333333333333 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_72step_depth0p5_tile4x4x1cap512_supportalpha34over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.1411764705882353 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_72step_depth0p5_tile4x4x1cap512_supportalpha36over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.12549019607843137 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_depth0p5_tile4x4x1cap512_supportalpha32over255_cachedprt_fastmacsplat.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-alpha-threshold 0.00392156862745098 --prt-support-alpha-threshold 0.12549019607843137 --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_200step_depth0p5_tile4x4x1cap512_supportalpha32over255_cachedprt_fastmacsplat.json
```

Result:

```text
72-step support 28/255: pass false, max tile 522, overflow 5, PRT PSNR 15.0506/14.0894 dB, train wall 4.033 s, cached render 16.602/17.824 ms.
72-step support 30/255: pass false, max tile 516, overflow 1, PRT PSNR 15.1236/14.1822 dB, train wall 3.759 s, cached render 15.624/15.379 ms.
72-step support 32/255: pass true, max tile 493, overflow 0, PRT PSNR 15.0573/14.3772 dB, train wall 3.447 s, cached render 13.597/13.530 ms.
72-step support 34/255: pass true, max tile 470, overflow 0, PRT PSNR 15.1768/14.0598 dB, train wall 2.819 s, cached render 12.169/12.355 ms.
72-step support 36/255: pass true, max tile 440, overflow 0, PRT PSNR 15.1518/14.1611 dB, train wall 3.260 s, cached render 13.417/14.079 ms.
72-step support 40/255: pass true, max tile 404, overflow 0, PRT PSNR 14.9909/14.2664 dB, train wall 2.815 s, cached render 14.292/13.553 ms.
20-step support 32/255: pass true, max tile 459, overflow 0, PRT PSNR 13.8584/14.4636 dB, train wall 1.081 s, cached render 17.384/16.484 ms.
200-step support 32/255: pass true, max tile 387, overflow 0, PRT PSNR 16.2663/13.9667 dB, train wall 8.887 s, cached render 14.342/14.038 ms; direct splats PSNR 18.1577/12.5995 dB, train wall 5.825 s.
```

Read: `32/255` is the lowest measured `tile_t=1` 1024 support threshold that
clears the 72-step capacity gate; `30/255` still overflows. This makes
`32/255` the current train-speed policy candidate, not `40/255`: it preserves
more support, improves heldout PSNR at 72 and 200 steps, and still passes
20/72/200 with zero overflow. The tradeoff is thinner capacity margin at
72 steps (`493/512`) and slower 20-step render, so a conservative render/eval
policy may still prefer a higher threshold or a margin rule.

Gate D2s turns the D2r row into an explicit opt-in policy without weakening the
generic selector. `recommend_projective_rational_tile_config(tube_count=1024)`
still fails closed. The new
`recommend_projective_rational_train_speed_tile_policy(tube_count=1024)`
returns policy `train_speed_support32_1024`, tile `4x4x1:512`, and support
alpha `32/255`. The D2 compare harness exposes this with
`--prt-tile-policy train_speed`; explicit `--tile-config` remains a separate
manual path.

Validation:

```text
python3 tests/projective_rational_tile_config_check.py
python3 -m py_compile torch_gsplat_bridge_star_uvt_prt/tile_config.py torch_gsplat_bridge_star_uvt_prt/__init__.py research_project/benchmarks/projective_rational_multicam_splat_compare.py
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 32 --max-frames 2 --steps 1 --prt-tubes 1024 --splat-count 16 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --render-warmups 0 --render-repeats 1 --prt-eval-cache-compiled --out-json /tmp/prt_train_speed_policy_smoke.json
```

Smoke read: pass true, policy `train_speed_support32_1024`, support
`0.12549019607843137`, tile `4x4x1:512`, max tile 474, overflow 0.

Gate D2t wires the backward phase profiler to the same explicit train-speed
policy and support threshold, then profiles the selected 1024 policy. This
prevents the profile harness from accidentally measuring the old overflowing
generic selector path.

Validation:

```text
python3 -m py_compile research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --prt-tile-policy train_speed --profile-warmups 1 --profile-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_1024t_20step_train_speed_support32.json
```

Result:

```text
pass true, policy train_speed_support32_1024, support 32/255, tile 4x4x1:512, max tile 460, overflow 0, grad finite true.
median total 19.517 ms:
  alloc tiles 0.009 ms
  clear tiles 0.243 ms
  bin tubes 0.518 ms
  alloc grads 0.012 ms
  clear grads 0.262 ms
  backward kernel 18.483 ms
```

Read: for the selected 1024 train-speed policy, the remaining backward time is
inside `projective_rational_tile_pixel_atomic_backward`, not binning or setup.
The backward kernel is about 95% of the profiled total, while binning is about
2.7%. The next speed work should target lower-atomic/two-pass accumulation
inside the backward kernel; reducing tile setup will not move this row much.

Gate D2u tests whether the existing PRT `tile_pair_atomic` backward is already
the lower-contention alternative. The train-step timing and breakdown probes now
accept `--support-alpha-threshold` so the synthetic case can match the selected
1024 support-pruned policy surface.

Validation:

```text
python3 -m py_compile research_project/benchmarks/projective_rational_train_step_timing_probe.py research_project/benchmarks/projective_rational_train_step_breakdown_probe.py
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_train_step_breakdown_probe.py --tube-counts 1024 --frames 4 --width 64 --height 64 --tile-config 4x4x1:512 --support-alpha-threshold 0.12549019607843137 --warmups 1 --repeats 5 --forward-mode tiled --backward-mode tile_pixel_atomic --out-json research_project/benchmarks/results/projective_rational_train_step_breakdown_probe_1024_support32_tile_pixel_atomic.json
python3 research_project/benchmarks/projective_rational_train_step_breakdown_probe.py --tube-counts 1024 --frames 4 --width 64 --height 64 --tile-config 4x4x1:512 --support-alpha-threshold 0.12549019607843137 --warmups 1 --repeats 5 --forward-mode tiled --backward-mode tile_pair_atomic --out-json research_project/benchmarks/results/projective_rational_train_step_breakdown_probe_1024_support32_tile_pair_atomic.json
```

Result:

```text
tile_pixel_atomic: pass true, max tile 266, overflow 0, loss 0.00509760 -> 0.00509594, median forward 17.464 ms, backward 5.906 ms, wall 24.161 ms.
tile_pair_atomic:  pass true, max tile 266, overflow 0, loss 0.00509760 -> 0.00509594, median forward 17.524 ms, backward 2066.803 ms, wall 2086.299 ms.
```

Read: the existing `tile_pair_atomic` PRT backward is numerically viable on this
case but unusably slow. It is roughly 350x slower in the measured backward
segment, so the next lower-contention attempt should not be a direct switch to
`tile_pair_atomic`. It needs a new PRT-specific two-pass/reduction design that
avoids both pixel-level atomic contention and per-tube serial replay.

Gate D2v adds a profile-only `projective_rational_tile_pixel_compute_only_backward`
kernel. It runs the same per-pixel ordering, alpha replay, and local gradient
math as `projective_rational_tile_pixel_atomic_backward`, but writes a per-pixel
debug scalar instead of atomically accumulating gradients. This is not a
training path; it isolates arithmetic/replay cost from atomic-write cost.

Validation:

```text
python3 -m py_compile torch_gsplat_bridge_star_uvt_prt/rasterize.py research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py
python3 setup.py build_ext --inplace
python3 tests/projective_rational_tile_pixel_atomic_backward_check.py
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --prt-tile-policy train_speed --profile-warmups 1 --profile-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_1024t_20step_train_speed_support32_computeonly.json
```

Result:

```text
pass true, policy train_speed_support32_1024, support 32/255, tile 4x4x1:512, max tile 459, overflow 0, grad finite true.
median total 19.168 ms:
  alloc tiles 0.009 ms
  clear tiles 0.219 ms
  bin tubes 0.501 ms
  alloc grads 0.011 ms
  clear grads 0.210 ms
  backward kernel 18.216 ms
  compute-only kernel 17.541 ms
```

Read: atomics are not the main remaining cost. The compute-only kernel is 96%
of the full backward kernel time, leaving only about `0.676 ms` as the measured
atomic-write ceiling in this profile. A two-pass accumulation rewrite will not
move the selected 1024 policy much unless it also reduces the per-pixel replay
and local gradient arithmetic. The next kernel idea should focus on reusing
per-pixel alpha/order state or reducing repeated `eval_prt_h` / `exp` work, not
just changing the final accumulation destination.

Gate D2w adds tile-occupancy workload counters to the backward phase profiler.
This quantifies how much replay work the selected 1024 train-speed policy asks
the backward kernel to do.

Validation:

```text
python3 -m py_compile research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --prt-tile-policy train_speed --profile-warmups 1 --profile-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_1024t_20step_train_speed_support32_workload.json
```

Result:

```text
pass true, policy train_speed_support32_1024, support 32/255, tile 4x4x1:512, max tile 458, overflow 0.
active tiles: 1024 / 1024
mean active tile count: 140.970
tile-count percentiles: p50 111, p90 286, p95 335, p99 422
tile pixel count: 16
total tile-tube pairs: 144353
tile-pixel-tube visits: 2309648
median backward kernel: 18.213 ms
median compute-only kernel: 17.719 ms
visits per compute-only ms: 130348
```

Read: the selected 1024 backward row is broad replay work, not a tiny hotspot.
Every tile is active, the p99 active tile holds 422 tubes, and even this
64x64x4 profile replays about 2.31M tile-pixel-tube visits. This reinforces the
D2v result: accumulation rewrites alone are unlikely to matter. The next speed
attempt should reduce replay itself, either by caching per-pixel order/alpha
state from forward for backward, fusing the training forward/backward pass, or
making the support policy/density policy reduce visits without corrupting the
render alpha semantics.

Gate D2x adds a profile-only `projective_rational_tile_pixel_replay_only_backward`
kernel. It keeps tile load, depth sort, PRT sample ordering, and alpha /
transmittance replay, then writes one debug scalar per pixel. It skips reverse
blend propagation, local PRT derivative math, and all gradient writes. This
separates "can we avoid replay?" from "can we tune derivative math?"

Validation:

```text
python3 -m py_compile torch_gsplat_bridge_star_uvt_prt/rasterize.py research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py
python3 setup.py build_ext --inplace
python3 tests/projective_rational_tile_pixel_atomic_backward_check.py
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --prt-tile-policy train_speed --profile-warmups 1 --profile-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_1024t_20step_train_speed_support32_replayonly.json
```

Result:

```text
pass true, policy train_speed_support32_1024, support 32/255, tile 4x4x1:512, max tile 458, overflow 0.
active tiles: 1024 / 1024
tile-pixel-tube visits: 2311072
median total: 19.050 ms
median backward kernel: 18.116 ms
median compute-only kernel: 17.574 ms
median replay-only kernel: 17.036 ms
replay / compute-only: 0.969
compute-only minus replay-only: 0.538 ms
backward minus compute-only: 0.542 ms
```

Read: the selected 1024 backward cost is alpha/order replay, not derivative
math. Replay-only is about 97% of compute-only time. The reverse local-gradient
work and the atomic gradient writes are each only about half a millisecond in
this row. The next serious speed experiment should cache the forward
compositing trace or fuse forward and backward during training so backward can
consume per-pixel order, alpha, and transmittance state instead of replaying the
whole PRT shader.

Gate D2y adds `projective_rational_trace_cache_planner.py`, a benchmark-side
planner that converts the measured D2x tile-pixel-tube visits into trace memory
budgets. It does not claim an exact future implementation cost; sparse trace
rows are upper bounds because not every binned tube survives alpha and
transmittance replay.

Validation:

```text
python3 -m py_compile research_project/benchmarks/projective_rational_trace_cache_planner.py
python3 research_project/benchmarks/projective_rational_trace_cache_planner.py --profile-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_64_4f_1024t_20step_train_speed_support32_replayonly.json --projection 128x128x4 --projection 256x256x4 --projection 256x256x16 --out-json research_project/benchmarks/results/projective_rational_trace_cache_planner_64_4f_1024t_train_speed_support32.json
```

Result for the minimal `id + alpha + t_before` trace layout:

```text
observed 64x64x4:   dense 96.1 MiB, sparse upper bound 26.6 MiB, visits 2.31M
projected 128x128x4: dense 384.2 MiB, sparse upper bound 106.3 MiB, visits 9.24M
projected 256x256x4: dense 1537.0 MiB, sparse upper bound 425.2 MiB, visits 36.98M
projected 256x256x16: dense 6148.0 MiB, sparse upper bound 1700.7 MiB, visits 147.91M
```

The D2x timing ceiling is:

```text
replay-only kernel: 17.036 ms
optimistic cached backward floor: 1.080 ms
```

Read: a dense per-pixel slot trace is already too large for the tiny 64x64x4
row and becomes absurd at fuller resolution. A compact sparse trace may be
viable for small training/eval probes, but it still has substantial write/read
bandwidth and scales linearly with visit count. The cleaner implementation bet
is a fused training path that keeps forward compositing and backward adjacent
without materializing a generic full-frame trace.

Gate D2z adds a research-only
`projective_rational_tile_pixel_fused_mse_backward` op. It bins once, performs
the tiled PRT forward compositing inside the same per-pixel kernel, computes
MSE target gradients, and immediately runs the reverse compositing and local
PRT derivative path. This is the first concrete fused train-step kernel smoke;
it is not yet wired into the trainer timing loop.

Validation:

```text
python3 -m py_compile torch_gsplat_bridge_star_uvt_prt/rasterize.py torch_gsplat_bridge_star_uvt_prt/__init__.py research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py tests/projective_rational_tile_pixel_fused_mse_backward_check.py
python3 setup.py build_ext --inplace
python3 tests/projective_rational_tile_pixel_fused_mse_backward_check.py
python3 research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py --out-json research_project/benchmarks/results/projective_rational_tile_pixel_fused_mse_backward_check.json
```

Result:

```text
pass true
reference loss: 0.2951766551
fused loss: 0.2951766849
loss abs error: 2.98e-08
max grad abs error: 9.31e-10
max grad rel error: 1.47e-07
overflow tiles: 0
```

Read: the fused train-step direction is now a real checked kernel path, not only
a planner conclusion. The next gate should time this fused MSE path on the
selected 1024 train-speed row against the current `render -> loss -> backward`
step and only then decide how to expose it in the trainer harness.

Gate D3a adds `projective_rational_fused_mse_timing_probe.py` and times the
selected 1024 train-speed row against the separate manual path:
`tiled render -> MSE grad image -> tile-pixel backward`. This deliberately
excludes optimizer updates so the row isolates the fused kernel decision.

Validation:

```text
python3 -m py_compile research_project/benchmarks/projective_rational_fused_mse_timing_probe.py
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_fused_mse_timing_probe.py --tube-counts 1024 --frames 4 --width 64 --height 64 --prt-tile-policy train_speed --warmups 1 --repeats 5 --out-json research_project/benchmarks/results/projective_rational_fused_mse_timing_probe_64_4f_1024t_train_speed_support32.json
```

Result:

```text
pass true, policy train_speed_support32_1024, support 32/255, tile 4x4x1:512
max tile 266, overflow 0, fused overflow 0
loss abs error: 4.66e-10
max grad abs error: 2.33e-10
max grad rel error: 1.97e-04
separate median: 25.466 ms
fused median: 6.990 ms
fused speedup: 3.64x
```

Read: D3a is the first strong speed result after the backward investigation.
The fused MSE path removes the replay duplicate exactly where D2x/D2y predicted
and cuts the selected 1024 training kernel slice by about 72.5%. The next step
is to put this behind an explicit research-harness mode and measure full train
wall, PSNR, and render timing against the existing non-fused path.

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
5. Split train-speed and render-speed tile policy if the 512-tube `tile_t=1` train win should become default for training only.
6. Decide the policy surface for 1024 support-only pruning: default fidelity mode, explicit train-speed mode, support schedule, or capacity fallback; `tile_t=1` with support `32/255` is the current measured train-speed choice.
7. Wire the fused MSE PRT path into an explicit research-harness train mode and measure full train wall, PSNR, and render timing against the non-fused path.
8. Keep accumulation-only and derivative-math rewrites behind fused-train-step work unless a new profile changes the cost split.
9. Promote the cached or fused camera-compiler path from benchmark flag to the intended playback and bake contract.
