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
- [x] Gate D3b: fused MSE full multicam train/eval row.
- [x] Gate D3c: fused sequence 72/200-step 1024 multicam rows.
- [x] Gate D3d: fused sequence 128px 1024 multicam scaling rows.
- [x] Gate D3e: fused sequence 128px 8-frame 1024 multicam scaling rows.
- [x] Gate D3f: explicit 2048-tube/2048-splat 128px 8-frame capacity rows.
- [x] Gate D3g: 2048-tube support-threshold dial and rejected global policy promotion.
- [x] Gate D3h: 256px x 8-frame 2048-tube scaling rows.
- [x] Gate D3i: split train/eval support-threshold probe for 256px 2048-tube rows.
- [x] Gate D3j: same-wall 256px PRT/splat row with multi-support eval on one PRT checkpoint.
- [x] Gate D3k: `tile_t=1` forward presorted-order shortcut with parity and selected timing rows.
- [x] Gate D3l: 2048-tube 256px train-wall profile after the forward shortcut.
- [x] Gate D3m: fused-MSE `tile_t=1` threadgroup presort train-kernel shortcut.
- [x] Gate D3n: fused-MSE replay bookkeeping cleanup after threadgroup presort.
- [x] Gate D3o: 2048-tube support-schedule boundary sweep after replay cleanup.
- [x] Gate D3p: spend train72/eval64 wall savings on a 190-step PRT same-wall row.
- [x] Gate D3q: exact 200-step PRT vs 200-step splat comparison on train72/eval64.
- [x] Gate D3r: fused-MSE tile-level loss reduction micro-kernel cleanup.
- [x] Gate D3s: reject fused-MSE `h_terms == 3` specialization after timing regression.
- [x] Gate D3t: 195-step same-wall schedule boundary after D3r.
- [x] Gate D3u: reject fused-MSE opacity exp reuse after timing no-op.
- [x] Gate D3v: current-code 190-step same-wall rerun after D3r.
- [x] Gate D3w: reject tile-slot threadgroup gradient reductions.
- [x] Gate D3x: current D3r backward phase profile and trace-cache viability read.
- [x] Gate D3y: train-used-gradient fused-MSE kernel that skips unused gradient families.
- [x] Gate D3z: 240-step D3y same-wall boundary row rejects spending the whole train-wall margin.
- [x] Gate D4a: 205/210/220-step D3y boundary sweep rejects replacing the accepted 200-step row.
- [x] Gate F0: depth-banded homography-flow gauge residual-tube projection/render falsifier.
- [x] Gate F0b: depth-banded residual robustness rows for object velocity and harder camera motion.
- [x] Gate F0c: quantify PRT-fallback outliers for the hard-camera/object-motion residual row.
- [x] Gate F0d: four-seed hybrid fallback robustness sweep.
- [x] Gate F0e: tile-policy estimate sweep for the hybrid fallback row.
- [x] Gate F0f: residual-coordinate culling control against ordinary image-space culling.
- [x] Gate F0g: stricter reference-atlas-plus-residual culling control.
- [x] Gate F0h: inverse-homography atlas-residual representation probe.
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

Gate D3b wires the fused path into the real multicam compare harness as
`--prt-train-mode fused_mse`. It currently requires `--prt-loss-mode sequence`
so the fused op can compute the exact full-sequence MSE without faking
sampled-frame targets.

Validation:

```text
python3 -m py_compile research_project/benchmarks/projective_rational_multicam_splat_compare.py
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 32 --max-frames 2 --steps 1 --prt-tubes 16 --splat-count 16 --splat-renderer fast_mac --init-depth 0.5 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 0 --render-repeats 1 --prt-eval-cache-compiled --out-json /tmp/prt_multicam_fused_mse_smoke.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode separate --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_sequence_train_speed_support32_separate.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_20step_sequence_train_speed_support32_fused_mse.json
```

Result:

```text
separate sequence PRT: pass true, train wall 1.393 s, loss 0.053215 -> 0.036405, PSNR 14.3438 / heldout 14.3879 dB, render 17.39 / 17.05 ms, max tile 467, overflow 0.
fused_mse sequence PRT: pass true, train wall 0.590 s, loss 0.053215 -> 0.036177, PSNR 14.3413 / heldout 14.3494 dB, render 17.94 / 17.77 ms, max tile 466, overflow 0.
PRT train-wall speedup: 2.36x.
Fast-mac direct splat baseline in the same rows: PSNR 15.1065 / heldout 11.8419 dB, render about 25.95-28.31 ms.
```

Read: the fused path survives the real multicam harness and keeps the same
quality surface on this 20-step sequence row while cutting PRT train wall by
more than half. This does not make the overall representation "done"; it moves
the training-speed bottleneck enough that the next comparison should use the
fused mode by default for sequence-loss PRT rows and return to quality/capacity
questions rather than backward replay.

Gate D3c runs longer fused-sequence rows with the selected 1024 train-speed
policy and the same 1024 fast-mac direct-splat baseline.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_72step_sequence_train_speed_support32_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 64 --max-frames 4 --steps 200 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_64_4f_1024t_1024s_200step_sequence_train_speed_support32_fused_mse.json
```

Result:

```text
20-step fused sequence:  PRT train wall 0.590 s, PSNR 14.3413 / heldout 14.3494 dB, render 17.94 / 17.77 ms, max tile 466, overflow 0.
72-step fused sequence:  PRT train wall 2.188 s, PSNR 15.5323 / heldout 14.3798 dB, render 14.05 / 14.41 ms, max tile 451, overflow 0.
200-step fused sequence: PRT train wall 5.996 s, PSNR 18.2240 / heldout 13.9701 dB, render 18.52 / 15.70 ms, max tile 321, overflow 0.

200-step direct splat baseline: train wall 5.813 s, PSNR 18.1577 / heldout 12.5995 dB, render 34.23 ms.
```

Read: this is the cleanest current 1024-row story. With fused sequence training,
PRT reaches comparable train wall to fast-mac direct splats by 200 steps, slightly
beats direct splats on train PSNR, keeps the heldout PSNR lead, and renders
about 2x faster on train cameras. The remaining gap is no longer "backward is
obviously too slow"; it is deciding whether this sequence-loss fused path is the
right default and then scaling resolution/frame count without losing capacity
margin.

Gate D3d scales the fused sequence path from the 64x64x4 gate to 128px while
keeping the same 4-frame window, 1024 PRT tubes, 1024 direct splats, and
train-speed support policy.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 4 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_4f_1024t_1024s_20step_sequence_train_speed_support32_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 4 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_4f_1024t_1024s_72step_sequence_train_speed_support32_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 4 --steps 200 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_4f_1024t_1024s_200step_sequence_train_speed_support32_fused_mse.json
```

Result:

```text
20-step 128px PRT:  train wall 1.013 s, PSNR 14.1146 / heldout 13.8874 dB, render 38.26 / 36.43 ms, max tile 402, overflow 0.
20-step 128px splat: train wall 1.176 s, PSNR 14.6958 / heldout 11.6551 dB, render 29.79 / 31.35 ms.

72-step 128px PRT:  train wall 3.024 s, PSNR 15.6588 / heldout 13.8785 dB, render 26.56 / 27.04 ms, max tile 360, overflow 0.
72-step 128px splat: train wall 1.779 s, PSNR 15.0143 / heldout 12.2189 dB, render 34.60 / 22.92 ms.

200-step 128px PRT:  train wall 6.479 s, PSNR 17.9687 / heldout 13.3233 dB, render 18.01 / 17.96 ms, max tile 275, overflow 0.
200-step 128px splat: train wall 5.720 s, PSNR 17.6388 / heldout 12.3287 dB, render 32.63 / 33.45 ms.
```

Read: the fused 1024 PRT path scales to 128px without capacity failures. It
keeps a heldout PSNR lead at all measured step counts and beats direct splats on
train PSNR by 72/200 steps. The speed story is mixed but useful: PRT train wall
is faster at 20 steps, slower by 72/200, and PRT render becomes faster again by
200 steps as support shrinks during training. The next scaling gate should vary
frame count or tube count, not return to 64px.

Gate D3e doubles the temporal window to 8 frames at 128px while keeping 1024 PRT
tubes, 1024 direct splats, fused sequence loss, and the train-speed support
policy.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 20 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_1024t_1024s_20step_sequence_train_speed_support32_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 72 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_1024t_1024s_72step_sequence_train_speed_support32_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 200 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_1024t_1024s_200step_sequence_train_speed_support32_fused_mse.json
```

Result:

```text
20-step 128px 8f PRT:  train wall 1.091 s, PSNR 14.3718 / heldout 13.6643 dB, render 42.28 / 40.87 ms, max tile 240, overflow 0.
20-step 128px 8f splat: train wall 0.656 s, PSNR 14.4082 / heldout 11.5027 dB, render 64.40 / 71.79 ms.

72-step 128px 8f PRT:  train wall 2.631 s, PSNR 15.8625 / heldout 13.7627 dB, render 26.43 / 27.59 ms, max tile 204, overflow 0.
72-step 128px 8f splat: train wall 1.939 s, PSNR 14.6331 / heldout 12.0490 dB, render 69.27 / 70.67 ms.

200-step 128px 8f PRT:  train wall 6.811 s, PSNR 18.1388 / heldout 13.0431 dB, render 20.01 / 18.70 ms, max tile 159, overflow 0.
200-step 128px 8f splat: train wall 5.947 s, PSNR 16.2365 / heldout 12.3544 dB, render 55.85 / 68.03 ms.
```

Read: the 8-frame scaling row is stronger than the 4-frame 128px row for the
sublinear thesis. PRT has no overflow, keeps a large render-speed lead at every
step count, and beats direct splats on train and heldout PSNR by 72/200 steps.
The cost is train wall: PRT is still slower than direct splats by about 14.5%
at 200 steps, but the render and heldout wins now clearly survive the longer
window.

Gate D3f probes whether simply doubling capacity improves the 128px x 8f row.
Because the train-speed policy is intentionally fail-closed above 1024 tubes,
this row uses an explicit `4x4x1:512` tile config with the same support
threshold, `32/255`, and compares 2048 PRT tubes against 2048 direct splats.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 20 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.12549019607843137 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_2048t_2048s_20step_sequence_tile4x4x1cap512_support32_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 72 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.12549019607843137 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_2048t_2048s_72step_sequence_tile4x4x1cap512_support32_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.12549019607843137 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_2048t_2048s_200step_sequence_tile4x4x1cap512_support32_fused_mse.json
```

Result:

```text
20-step 2048 PRT:  train wall 2.328 s, PSNR 13.9021 / heldout 13.8044 dB, render 94.25 / 89.22 ms, max tile 429, overflow 0.
20-step 2048 splat: train wall 1.279 s, PSNR 14.0393 / heldout 11.6181 dB, render 72.32 / 85.29 ms.

72-step 2048 PRT:  train wall 6.596 s, PSNR 15.3537 / heldout 13.8939 dB, render 63.98 / 65.42 ms, max tile 429, overflow 0.
72-step 2048 splat: train wall 1.331 s, PSNR 14.3046 / heldout 12.2788 dB, render 51.52 / 59.64 ms.

200-step 2048 PRT:  train wall 13.416 s, PSNR 17.5875 / heldout 13.4285 dB, render 34.16 / 36.79 ms, max tile 355, overflow 0.
200-step 2048 splat: train wall 4.984 s, PSNR 15.8656 / heldout 12.4923 dB, render 72.13 / 78.93 ms.
```

Read: 2048 PRT capacity does not beat the current 1024-tube operating point.
It passes with zero overflow and still beats 2048 direct splats on heldout PSNR,
but it is much slower to train and render until 200 steps, and even at 200 steps
it gives lower train PSNR than the 1024-tube 200-step row. The useful conclusion
is that the next PRT work is not "add more tubes"; it is reducing tile pressure
or improving initialization/capacity use so a larger tube set does not make the
rasterizer pay for mostly redundant support.

Gate D3g sweeps tighter support thresholds for the same 2048-tube 128px x 8f
row. The goal is to find whether 2048 is bad because of capacity itself or
because support `32/255` makes every tile carry too many mostly redundant tubes.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 72 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_2048t_2048s_72step_sequence_tile4x4x1cap512_support48_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 72 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2196078431372549 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_2048t_2048s_72step_sequence_tile4x4x1cap512_support56_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 72 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_2048t_2048s_72step_sequence_tile4x4x1cap512_support64_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_2048t_2048s_200step_sequence_tile4x4x1cap512_support48_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2196078431372549 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_2048t_2048s_200step_sequence_tile4x4x1cap512_support56_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_2048t_2048s_200step_sequence_tile4x4x1cap512_support64_fused_mse.json
```

Result:

```text
72-step support48:  PRT train wall 2.708 s, PSNR 15.8818 / heldout 13.5903 dB, render 26.14 / 26.61 ms, max tile 252, overflow 0.
72-step support56:  PRT train wall 2.251 s, PSNR 16.5722 / heldout 13.3961 dB, render 17.70 / 16.66 ms, max tile 201, overflow 0.
72-step support64:  PRT train wall 2.033 s, PSNR 17.0477 / heldout 13.2416 dB, render 17.07 / 17.54 ms, max tile 136, overflow 0.

200-step support48: PRT train wall 6.195 s, PSNR 18.1871 / heldout 13.0447 dB, render 18.76 / 17.43 ms, max tile 233, overflow 0.
200-step support56: PRT train wall 5.535 s, PSNR 18.3705 / heldout 12.7277 dB, render 16.17 / 15.77 ms, max tile 201, overflow 0.
200-step support64: PRT train wall 5.637 s, PSNR 18.4041 / heldout 12.7391 dB, render 13.79 / 14.67 ms, max tile 126, overflow 0.
```

Read: support threshold is the 2048 dial. `48/255` is the balanced current row:
at 200 steps it slightly beats the 1024-tube 200-step train PSNR, matches the
1024 heldout PSNR, and is faster to train and render. `64/255` is the overfit
speed row: it reaches 18.4041 dB train PSNR and 13.8 ms train render, but gives
up heldout PSNR. This makes the previous D3f negative result more specific:
2048 was not bad because the representation lacks capacity; it was bad because
support `32/255` made the raster workload too dense.

Rejected promotion: a global 2048 train-speed policy is not safe yet because
the current selector only sees `tube_count` and `camera_motion_scale`, not
target resolution or tube density. This smoke intentionally failed after trying
to promote support `48/255` as the global 2048 train-speed policy:

```text
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 32 --max-frames 2 --steps 1 --prt-tubes 2048 --splat-count 16 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 0 --render-repeats 1 --prt-eval-cache-compiled --out-json /tmp/prt_multicam_2048_train_speed_policy_smoke.json
RuntimeError: fused MSE PRT train step overflowed tile capacity
```

So the 2048 rows stay explicit for now. A real selector needs a density-aware
signature, or the harness must keep requiring explicit tile config plus support
threshold for 2048+ tube experiments.

Gate D3h scales the 2048 PRT rows to 256px while keeping the same 8-frame
multicam sample and 2048 direct-splat baseline. It tests whether the support
threshold dial still gives a useful speed/quality row at a fuller local
resolution.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 72 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_72step_sequence_tile4x4x1cap512_support48_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 72 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_72step_sequence_tile4x4x1cap512_support64_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_200step_sequence_tile4x4x1cap512_support64_fused_mse.json
```

Result:

```text
72-step support48 PRT:  train wall 7.413 s, PSNR 15.6489 / heldout 13.6551 dB, render 89.59 / 98.29 ms, max tile 257, overflow 0.
72-step support48 splat: train wall 2.174 s, PSNR 13.9928 / heldout 12.2259 dB, render 59.17 / 63.08 ms.

72-step support64 PRT:  train wall 3.740 s, PSNR 16.5587 / heldout 13.1365 dB, render 28.69 / 32.45 ms, max tile 130, overflow 0.
72-step support64 splat: train wall 1.944 s, PSNR 13.9928 / heldout 12.2259 dB, render 70.97 / 64.96 ms.

200-step support64 PRT:  train wall 8.106 s, PSNR 18.0244 / heldout 12.7822 dB, render 17.65 / 23.51 ms, max tile 115, overflow 0.
200-step support64 splat: train wall 6.434 s, PSNR 15.6135 / heldout 12.4150 dB, render 66.20 / 79.22 ms.
```

Read: the 256px row keeps the overfit and render-speed story but exposes the
training-speed gap. Support `64/255` is the useful 256px setting: it beats direct
splats on train and heldout PSNR and renders 2-4x faster, but the fused PRT train
loop is still slower than direct splats at 72/200 steps. Support `48/255` is too
dense at 256px for speed despite the stronger heldout PSNR. This points the next
engineering target back at train-step cost and support scheduling, not basic
render correctness.

Gate D3i adds a split train/eval support-threshold flag so the fast 256px
support `64/255` train row can be evaluated with looser support. This tests
whether heldout PSNR can be recovered at render/eval time without paying the
full support `48/255` train cost.

Validation:

```text
python3 setup.py build_ext --inplace
python3 -m py_compile research_project/benchmarks/projective_rational_multicam_splat_compare.py
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 72 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-eval-support-alpha-threshold 0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_72step_sequence_tile4x4x1cap512_train64_eval48_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 72 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-eval-support-alpha-threshold 0.2196078431372549 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_72step_sequence_tile4x4x1cap512_train64_eval56_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-eval-support-alpha-threshold 0.2196078431372549 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_200step_sequence_tile4x4x1cap512_train64_eval56_fused_mse.json
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-eval-support-alpha-threshold 0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_200step_sequence_tile4x4x1cap512_train64_eval48_fused_mse.json
```

Result:

```text
72-step train64/eval48 PRT:  train wall 3.753 s, PSNR 14.7623 / heldout 13.4184 dB, render 81.11 / 87.42 ms, max tile 229, overflow 0.
72-step train64/eval48 splat: train wall 1.944 s, PSNR 13.9928 / heldout 12.2259 dB, render 75.75 / 69.52 ms.

72-step train64/eval56 PRT:  train wall 3.802 s, PSNR 15.5118 / heldout 13.2977 dB, render 48.80 / 58.40 ms, max tile 185, overflow 0.
72-step train64/eval56 splat: train wall 1.760 s, PSNR 13.9928 / heldout 12.2259 dB, render 76.46 / 61.42 ms.

200-step train64/eval56 PRT: train wall 8.042 s, PSNR 16.2980 / heldout 13.0979 dB, render 30.75 / 42.15 ms, max tile 168, overflow 0.
200-step train64/eval56 splat: train wall 6.022 s, PSNR 15.6136 / heldout 12.4152 dB, render 80.34 / 89.75 ms.

200-step train64/eval48 PRT: train wall 7.732 s, PSNR 14.9961 / heldout 13.1349 dB, render 55.16 / 73.11 ms, max tile 221, overflow 0.
200-step train64/eval48 splat: train wall 5.734 s, PSNR 15.6136 / heldout 12.4151 dB, render 78.87 / 83.14 ms.
```

Read: split support is useful diagnostically but should not become the next
default. Looser eval support recovers only a small amount of heldout PSNR while
it destroys overfit PSNR and, at eval48, most of the render-speed story. Eval56
is the only plausible compromise: it keeps a speed win over direct splats and
raises heldout versus eval64, but it still gives up much of the support64 train
PSNR. The better direction is support scheduling or residual-certified support
inflation, not a static train/eval mismatch.

Benchmark caveat: these rows are same data, same train/heldout camera split,
same 8-frame window, and same nominal primitive count, so they are useful for
the current overfit question. They are not yet a broad representation claim:
PRT has far fewer learned parameters than per-frame splats, PRT fused sequence
loss is not the same objective as sampled-frame splat training, and cached
compiled PRT render timing is fair for repeated fixed camera paths but less fair
for constantly changing camera edits.

Rasterizer read: the fast path is tiled PRT with support pruning, cached eval
compile, and fused sequence MSE. The direct PRT reference path is not the speed
story. Compiler overhead is mostly under control in cached playback rows; the
remaining render cost is inside the Metal shade/blend path, especially
per-pixel sample-order replay. Backward and fused MSE already have a `tile_t=1`
presorted-order shortcut, so the next rasterizer experiment should test the same
shortcut in forward render with parity and cached-eval timing rows.

Gate D3j adds separate `--prt-steps` / `--splat-steps` controls and
`--prt-extra-eval-support-alpha-thresholds`, then runs the 256px x 8-frame
same-wall row. The tracked row trains one PRT support64 checkpoint for 135 steps
and one 2048-splat baseline for 200 steps; the same PRT checkpoint is evaluated
at eval support56 primary, plus support64 and support48 extras.

Validation:

```text
python3 setup.py build_ext --inplace
python3 -m py_compile research_project/benchmarks/projective_rational_multicam_splat_compare.py
python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 135 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-eval-support-alpha-threshold 0.2196078431372549 --prt-extra-eval-support-alpha-thresholds 0.25098039215686274,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt135_splat200_train64_eval56_extra64_48_fused_mse.json
```

Result:

```text
same-wall primary eval56 PRT: train wall 6.039 s, PSNR 15.9708 / heldout 13.2449 dB, render 37.95 / 52.00 ms, max tile 158, overflow 0.
same-wall 200-step splat:    train wall 6.135 s, PSNR 15.6135 / heldout 12.4150 dB, render 73.98 / 92.68 ms.

same checkpoint eval64 PRT:  PSNR 17.3970 / heldout 13.0040 dB, render 20.91 / 28.88 ms, max tile 114, overflow 0.
same checkpoint eval48 PRT:  PSNR 14.8970 / heldout 13.3000 dB, render 64.87 / 85.88 ms, max tile 209, overflow 0.
```

Read: this is the cleanest current answer to the same-wall overfit question.
At approximately equal train wall, PRT eval56 beats the direct-splat baseline on
train PSNR, heldout PSNR, and render speed. Eval64 is the fastest/overfit
render and still beats splat heldout; eval48 gives the best heldout but loses
train PSNR and most of the render-speed margin. Eval56 remains the compromise
point for 256px support64 training.

Gate D3k applies the existing `tile_t=1` presorted-order shortcut to the PRT
forward render kernel. The tiled binning pass already sorts by tile depth; for a
single-frame time tile, forward no longer reselects rational sample order for
every pixel. The old `tile_t>1` path stays on per-sample rational order.

Validation:

```text
python3 setup.py build_ext --inplace
STAR_UVT_TILE_T=1 python3 tests/projective_rational_tiled_render_check.py --tile-t 1 --out-json research_project/benchmarks/results/projective_rational_tiled_render_check_tilet1_forward_shortcut.json
python3 tests/projective_rational_tiled_render_check.py --tile-t 2 --out-json research_project/benchmarks/results/projective_rational_tiled_render_check_tilet2_forward_shortcut_regression.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 135 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-eval-support-alpha-threshold 0.2196078431372549 --prt-extra-eval-support-alpha-thresholds 0.25098039215686274,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt135_splat200_train64_eval56_extra64_48_forward_shortcut_fused_mse.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 128 --max-frames 8 --steps 200 --prt-tubes 1024 --splat-count 1024 --splat-renderer fast_mac --init-depth 0.5 --prt-tile-policy train_speed --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_128_8f_1024t_1024s_200step_sequence_train_speed_support32_forward_shortcut_fused_mse.json
```

Result:

```text
tile_t=1 parity: pass, max abs error 5.9605e-08, overflow 0.
tile_t=2 regression parity: pass, max abs error 5.9605e-08, overflow 0.

256px same-wall eval56 before shortcut: render 37.95 / 52.00 ms, PSNR 15.9708 / heldout 13.2449 dB.
256px same-wall eval56 after shortcut:  render 7.62 / 8.81 ms, PSNR 15.9965 / heldout 13.2398 dB, max tile 170, overflow 0.
256px same checkpoint eval64 after shortcut: render 5.81 / 7.74 ms, PSNR 17.6354 / heldout 12.8901 dB.
256px same checkpoint eval48 after shortcut: render 8.92 / 11.64 ms, PSNR 14.8920 / heldout 13.3283 dB.

128px 1024 support32 before shortcut: render 20.01 / 18.70 ms, PSNR 18.1388 / heldout 13.0431 dB.
128px 1024 support32 after shortcut:  render 3.67 / 3.86 ms, PSNR 18.2056 / heldout 13.0560 dB, max tile 136, overflow 0.
```

Read: this is the first strong renderer-side speed win in the moving-camera PRT
fork. The same-wall 256px row now renders roughly 5-6x faster than its previous
PRT eval path and roughly 10x faster than the direct-splat baseline, while
keeping the same quality story. The 128px selected 1024 row shows the same
pattern. This does not solve train wall by itself; it cleans up playback/bake
render and makes the sublinear rasterizer story much stronger.

Gate D3l profiles the remaining train-wall problem after D3k. The goal is to
separate playback render speed from the fused train-step cost on the current
2048-tube, 256px, 8-frame, support64 row.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_train_step_breakdown_probe.py --tube-counts 2048 --frames 8 --width 256 --height 256 --tile-config 4x4x1:512 --support-alpha-threshold 0.25098039215686274 --forward-mode tiled --backward-mode tile_pixel_atomic --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_train_step_breakdown_probe_2048_256_8f_support64_tile4x4x1_forward_shortcut.json
python3 research_project/benchmarks/projective_rational_fused_mse_timing_probe.py --tube-counts 2048 --frames 8 --width 256 --height 256 --tile-config 4x4x1:512 --support-alpha-threshold 0.25098039215686274 --warmups 1 --repeats 3 --out-json research_project/benchmarks/results/projective_rational_fused_mse_timing_probe_2048_256_8f_support64_tile4x4x1_forward_shortcut.json
python3 -m py_compile research_project/benchmarks/projective_rational_fused_mse_timing_probe.py research_project/benchmarks/projective_rational_multicam_train_breakdown.py
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 256 --max-frames 8 --steps 20 --prt-tubes 2048 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 1 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_256_8f_2048t_20step_support64_fused_mse_forward_shortcut.json
```

Result:

```text
Projected separate train-step breakdown: forward 3.27 ms, loss 1.17 ms, backward 21.48 ms, optimizer 0.30 ms, total 26.04 ms, max tile 137, overflow 0.
Projected fused-MSE timing: separate 22.91 ms, fused 19.04 ms, speedup 1.20x, max grad abs error 3.73e-09, max grad rel error 0.00493, overflow 0.
Real multicam fused-MSE breakdown: median step 59.82 ms, fused MSE 49.59 ms, compile 4.04 ms, projected autograd backward 2.50 ms, clip 1.13 ms, optimizer 1.99 ms.
Real multicam share of median step: fused MSE 82.9%, compile 6.7%, projected autograd 4.2%, optimizer 3.3%, clip 1.9%.
Real multicam short row: 20 steps, loss 0.06337 -> 0.03240, eval PSNR 14.8884 / heldout 13.4261 dB, eval render 8.86 / 7.50 ms, max tile 119, overflow 0.
```

Read: D3k did its job for playback and bake render. The remaining 256px
training wall is not camera compilation, not the world-tube autograd chain, and
not optimizer overhead. It is the fused MSE Metal kernel itself, dominated by
the same tile/sample replay and gradient accumulation work. The next train-speed
work should therefore attack the fused train kernel directly: accumulation-only
specialization, derivative-math simplification, trace/replay reuse, or a lower
support schedule. Promoting cached compiler paths is still the right playback
contract, but it will not close the training wall by itself.

Gate D3m applies the same `tile_t=1` presorted-order idea to the fused MSE train
kernel. Before this change, each pixel thread copied and sorted its tile's tube
list. The new fused kernel sorts the tile list once in threadgroup memory and
then each pixel replays that sorted order. The optimization is deliberately
narrow: it changes the fused MSE kernel dispatch to use `STAR_THREADS` per
threadgroup and leaves the non-fused tile-pixel backward path unchanged.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py --out-json research_project/benchmarks/results/projective_rational_tile_pixel_fused_mse_backward_check_threadgroup_presort.json
python3 research_project/benchmarks/projective_rational_fused_mse_timing_probe.py --tube-counts 2048 --frames 8 --width 256 --height 256 --tile-config 4x4x1:512 --support-alpha-threshold 0.25098039215686274 --warmups 1 --repeats 5 --out-json research_project/benchmarks/results/projective_rational_fused_mse_timing_probe_2048_256_8f_support64_tile4x4x1_threadgroup_presort.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 256 --max-frames 8 --steps 20 --prt-tubes 2048 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 1 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_256_8f_2048t_20step_support64_fused_mse_threadgroup_presort.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 135 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-eval-support-alpha-threshold 0.2196078431372549 --prt-extra-eval-support-alpha-thresholds 0.25098039215686274,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt135_splat200_train64_eval56_extra64_48_threadgroup_presort_fused_mse.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 170 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-eval-support-alpha-threshold 0.2196078431372549 --prt-extra-eval-support-alpha-thresholds 0.25098039215686274,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt170_splat200_train64_eval56_extra64_48_threadgroup_presort_fused_mse.json
```

Result:

```text
Fused MSE parity: pass, loss abs error 2.98e-08, max grad abs error 1.86e-09, overflow 0.
Projected fused MSE before D3m: 19.04 ms fused, 22.91 ms separate, 1.20x speedup over separate.
Projected fused MSE after D3m:  13.95 ms fused, 23.81 ms separate, 1.71x speedup over separate.

Real 20-step breakdown before D3m: median step 59.82 ms, fused MSE 49.59 ms, train loop 1.597 s.
Real 20-step breakdown after D3m:  median step 53.06 ms, fused MSE 44.98 ms, train loop 1.374 s.

135-step under-wall row after D3m: PRT wall 5.624 s vs splat wall 7.109 s.
135-step eval56 PRT: PSNR 15.7336 / heldout 13.2276 dB, render 9.35 / 11.46 ms, max tile 175, overflow 0.
135-step splat:      PSNR 15.6135 / heldout 12.4150 dB, render 62.97 / 87.65 ms.

170-step calibrated row after D3m: PRT wall 6.855 s vs splat wall 6.165 s.
170-step eval56 PRT: PSNR 16.2002 / heldout 13.2881 dB, render 8.46 / 9.64 ms, max tile 164, overflow 0.
170-step splat:      PSNR 15.6135 / heldout 12.4150 dB, render 73.93 / 84.65 ms.
170-step eval64 PRT: PSNR 17.8774 / heldout 12.9558 dB, render 6.63 / 8.34 ms, max tile 118, overflow 0.
170-step eval48 PRT: PSNR 15.0470 / heldout 13.3993 dB, render 10.54 / 13.76 ms, max tile 211, overflow 0.
```

Read: D3m is a real but partial train-speed win. It removes the most obvious
per-pixel redundant sort from fused MSE and improves the same 2048/256px train
path without changing the quality story. The remaining train wall is still
inside the fused MSE kernel, so the next train-speed ideas should be gradient
accumulation structure and derivative simplification, not camera compilation.
For comparison reporting, use the D3m bracket: 135 PRT steps is clearly under
the splat wall and still wins PSNR/render speed; 170 PRT steps is slightly over
the splat wall and widens the PSNR margin.

Gate D3n cleans up fused-MSE replay bookkeeping after D3m. The `tile_t=1` fused
kernel was already using a threadgroup-sorted tile list, but it still copied
that sorted list into per-thread replay arrays and pre-cleared per-candidate
bookkeeping arrays. D3n replays directly from the shared sorted IDs for
`tile_t=1` and initializes only visited candidates.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py --out-json research_project/benchmarks/results/projective_rational_tile_pixel_fused_mse_backward_check_replay_bookkeeping_cleanup.json
python3 -m py_compile research_project/benchmarks/projective_rational_fused_mse_timing_probe.py research_project/benchmarks/projective_rational_multicam_train_breakdown.py research_project/benchmarks/projective_rational_multicam_splat_compare.py
python3 research_project/benchmarks/projective_rational_fused_mse_timing_probe.py --tube-counts 2048 --frames 8 --width 256 --height 256 --tile-config 4x4x1:512 --support-alpha-threshold 0.25098039215686274 --warmups 1 --repeats 5 --out-json research_project/benchmarks/results/projective_rational_fused_mse_timing_probe_2048_256_8f_support64_tile4x4x1_replay_bookkeeping_cleanup.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 256 --max-frames 8 --steps 20 --prt-tubes 2048 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 1 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_256_8f_2048t_20step_support64_fused_mse_replay_bookkeeping_cleanup.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 135 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-eval-support-alpha-threshold 0.2196078431372549 --prt-extra-eval-support-alpha-thresholds 0.25098039215686274,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt135_splat200_train64_eval56_extra64_48_replay_bookkeeping_cleanup_fused_mse.json
```

Result:

```text
Fused MSE parity: pass, loss abs error 2.98e-08, max grad abs error 1.86e-09, overflow 0.
Projected fused MSE after D3m: 13.95 ms fused, 23.81 ms separate, 1.71x speedup over separate.
Projected fused MSE after D3n: 12.21 ms fused, 24.77 ms separate, 2.03x speedup over separate.

Real 20-step breakdown after D3m: median step 53.06 ms, fused MSE 44.98 ms, train loop 1.374 s.
Real 20-step breakdown after D3n: median step 53.40 ms, fused MSE 41.08 ms, train loop 1.689 s.

135-step under-wall row after D3n: PRT wall 5.670 s vs splat wall 7.350 s.
135-step eval56 PRT: PSNR 16.0302 / heldout 13.2225 dB, render 8.57 / 9.42 ms, max tile 163, overflow 0.
135-step splat:      PSNR 15.6136 / heldout 12.4152 dB, render 79.82 / 87.23 ms.
135-step eval64 PRT: PSNR 17.4955 / heldout 12.8992 dB, render 7.35 / 8.65 ms, max tile 120, overflow 0.
135-step eval48 PRT: PSNR 14.9446 / heldout 13.3054 dB, render 9.80 / 12.34 ms, max tile 218, overflow 0.
```

Read: D3n is worth keeping as a small fused-kernel cleanup. The projected fused
kernel timing improved again, and the real fused-MSE segment improved, but total
train-step wall is now dominated by noise and remaining non-fused overheads
around the same 53 ms median. The under-wall same-wall comparison remains strong:
PRT stays below splat training wall and still wins train PSNR, heldout PSNR, and
render speed. The next train-speed work needs a larger structural change to
gradient accumulation or support scheduling, not more replay bookkeeping cleanup.

Gate D3o tests support scheduling as the next train-speed lever. The question is
whether training with a tighter support threshold can cut train wall while an
eval support setting still beats direct splats on overfit PSNR, heldout PSNR,
and render speed. An initial train80 run failed before training because the
extension had been cleaned after commit; rebuilding with
`python3 setup.py build_ext --inplace` fixed the harness state.

Validation:

```text
python3 setup.py build_ext --inplace
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 135 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.3137254901960784 --prt-eval-support-alpha-threshold 0.2196078431372549 --prt-extra-eval-support-alpha-thresholds 0.3137254901960784,0.25098039215686274,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt135_splat200_train80_eval56_extra80_64_48_replay_cleanup_fused_mse.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 135 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2823529411764706 --prt-eval-support-alpha-threshold 0.2196078431372549 --prt-extra-eval-support-alpha-thresholds 0.2823529411764706,0.25098039215686274,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt135_splat200_train72_eval56_extra72_64_48_replay_cleanup_fused_mse.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 135 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2980392156862745 --prt-eval-support-alpha-threshold 0.25098039215686274 --prt-extra-eval-support-alpha-thresholds 0.2980392156862745,0.2196078431372549,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt135_splat200_train76_eval64_extra76_56_48_replay_cleanup_fused_mse.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 135 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2901960784313726 --prt-eval-support-alpha-threshold 0.25098039215686274 --prt-extra-eval-support-alpha-thresholds 0.2901960784313726,0.2196078431372549,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt135_splat200_train74_eval64_extra74_56_48_replay_cleanup_fused_mse.json
```

Result:

```text
Baseline D3n train64/eval56: wall 5.670 s, PSNR 16.0302 / heldout 13.2225 dB, render 8.57 / 9.42 ms.
Baseline D3n train64/eval64: PSNR 17.4955 / heldout 12.8992 dB, render 7.35 / 8.65 ms.

train72/eval64: wall 4.914 s vs splat 7.043 s; PSNR 15.7230 / heldout 13.1291 dB; render 8.09 / 10.58 ms; max tile 120; overflow 0.
train72/eval56: PSNR 14.5950 / heldout 13.1588 dB; render 9.91 / 10.56 ms.

train74/eval64: wall 4.304 s vs splat 7.033 s; PSNR 15.0366 / heldout 13.0986 dB; render 7.23 / 12.48 ms; max tile 119; overflow 0.
train76/eval64: wall 4.376 s vs splat 6.549 s; PSNR 14.6046 / heldout 13.1583 dB; render 10.16 / 13.49 ms; max tile 119; overflow 0.
train80/eval56: wall 4.431 s vs splat 6.609 s; PSNR 13.8306 / heldout 12.9219 dB; render 14.92 / 19.20 ms; max tile 138; overflow 0.
```

Read: support scheduling is real but sharp. Train72/eval64 is the best new
under-wall compromise: it trains about 13% faster than train64 while still
beating splats on train PSNR, heldout PSNR, and render speed. Train74/eval64 and
tighter support settings keep the heldout win and speed but lose train PSNR to
splats, so they are not the overfit-focused default. For "beat splats at same or
less wall" reporting, keep train64/eval56 as the strongest overfit row and add
train72/eval64 as the faster under-wall schedule candidate.

Gate D3p spends the D3o train72/eval64 wall savings on more PRT optimization
steps instead of stopping at 135 steps. The first command failed before training
because the local extension had been cleaned again; rebuilding with
`python3 setup.py build_ext --inplace` restored the registered fused-MSE op.

Validation:

```text
python3 setup.py build_ext --inplace
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 190 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2823529411764706 --prt-eval-support-alpha-threshold 0.25098039215686274 --prt-extra-eval-support-alpha-thresholds 0.2823529411764706,0.2196078431372549,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt190_splat200_train72_eval64_extra72_56_48_replay_cleanup_fused_mse.json
```

Result:

```text
190-step train72/eval64: PRT wall 6.311 s vs splat wall 7.141 s.
190-step eval64 PRT: PSNR 15.7453 / heldout 13.2515 dB, render 9.29 / 9.80 ms, max tile 98, overflow 0.
190-step splat:      PSNR 15.6135 / heldout 12.4150 dB, render 78.56 / 96.82 ms.
190-step eval72 PRT: PSNR 17.6293 / heldout 12.7018 dB, render 8.02 / 9.24 ms, max tile 66, overflow 0.
190-step eval56 PRT: PSNR 14.3695 / heldout 13.3011 dB, render 14.75 / 17.45 ms, max tile 143, overflow 0.
190-step eval48 PRT: PSNR 13.8370 / heldout 13.2531 dB, render 15.37 / 17.43 ms, max tile 186, overflow 0.
```

Read: D3p is the cleanest same-wall overfit comparison so far. With the faster
train72 schedule, PRT can run 190 optimization steps in less wall time than 200
direct-splat steps while still beating splats on train PSNR, heldout PSNR, and
render speed. The quality margin is smaller than the strongest train64/eval56
overfit row, but the timing is cleaner: 6.31 s PRT vs 7.14 s splat and about
8-10x faster eval rendering. Use this row for "same-to-same train budget" and
the D3n train64/eval56 row for "best PRT overfit quality under splat wall."

Gate D3q runs the exact same step count requested by the user: 200 PRT fused-MSE
steps versus 200 direct-splat steps on the D3o train72/eval64 schedule.

Validation:

```text
python3 setup.py build_ext --inplace
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 200 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2823529411764706 --prt-eval-support-alpha-threshold 0.25098039215686274 --prt-extra-eval-support-alpha-thresholds 0.2823529411764706,0.2196078431372549,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samesteps_prt200_splat200_train72_eval64_extra72_56_48_replay_cleanup_fused_mse.json
```

Result:

```text
200-step train72/eval64: PRT wall 6.222 s vs splat wall 6.067 s.
200-step eval64 PRT: PSNR 16.0683 / heldout 13.2186 dB, render 12.37 / 18.94 ms, max tile 110, overflow 0.
200-step splat:      PSNR 15.6136 / heldout 12.4151 dB, render 66.30 / 83.07 ms.
200-step eval72 PRT: PSNR 18.0461 / heldout 12.7662 dB, render 9.08 / 16.69 ms, max tile 77, overflow 0.
200-step eval56 PRT: PSNR 14.5067 / heldout 13.3301 dB, render 17.40 / 19.78 ms, max tile 155, overflow 0.
200-step eval48 PRT: PSNR 13.9265 / heldout 13.2562 dB, render 16.40 / 20.26 ms, max tile 197, overflow 0.
```

Read: exact equal-step training is not a clean wall-clock win in this single
run. PRT is about 2.6% slower on train loop wall, while still winning train
PSNR, heldout PSNR, and render speed. Report this row as "same steps: quality
and render win, train wall near-tie/slightly slower." Keep D3p for a strict
under-wall row and D3n train64/eval56 for strongest overfit quality under the
splat wall.

Gate D3r reduces fused-MSE loss accumulation from one global atomic per pixel to
one threadgroup reduction plus one global atomic per tile. This does not change
the gradient path.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py --out-json research_project/benchmarks/results/projective_rational_tile_pixel_fused_mse_backward_check_loss_threadgroup_reduce.json
python3 research_project/benchmarks/projective_rational_fused_mse_timing_probe.py --tube-counts 2048 --frames 8 --width 256 --height 256 --tile-config 4x4x1:512 --support-alpha-threshold 0.25098039215686274 --warmups 1 --repeats 5 --out-json research_project/benchmarks/results/projective_rational_fused_mse_timing_probe_2048_256_8f_support64_tile4x4x1_loss_threadgroup_reduce.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 256 --max-frames 8 --steps 20 --prt-tubes 2048 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 1 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_256_8f_2048t_20step_support64_fused_mse_loss_threadgroup_reduce.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 200 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2823529411764706 --prt-eval-support-alpha-threshold 0.25098039215686274 --prt-extra-eval-support-alpha-thresholds 0.2823529411764706,0.2196078431372549,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samesteps_prt200_splat200_train72_eval64_extra72_56_48_loss_threadgroup_reduce_fused_mse.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 200 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2823529411764706 --prt-eval-support-alpha-threshold 0.25098039215686274 --prt-extra-eval-support-alpha-thresholds 0.2823529411764706,0.2196078431372549,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samesteps_prt200_splat200_train72_eval64_extra72_56_48_loss_threadgroup_reduce_fused_mse_repeat2.json
```

Result:

```text
Parity: pass, loss abs error 0, max grad abs error 1.86e-09, overflow 0.
Projected fused MSE: 12.21 ms -> 10.76 ms, separate 21.78 ms, speedup 2.02x.
Real 20-step breakdown: median step 50.84 ms, fused MSE 41.46 ms, train loop 1.301 s.

200-step loss-reduce run 1: PRT wall 6.742 s vs splat 6.969 s; PSNR 15.8585 / heldout 13.1293 dB; render 8.52 / 12.15 ms.
200-step loss-reduce run 2: PRT wall 6.794 s vs splat 6.743 s; PSNR 16.1061 / heldout 13.2066 dB; render 7.30 / 8.39 ms.
```

Read: D3r is worth keeping as a small parity-safe cleanup, but it is not a
robust exact-step train-wall unlock by itself. The projected fused kernel gets
faster, and the 20-step diagnostic total improves, but the full 200-step
comparison is still a noise-band near tie: one paired run is under splat wall
and the repeat is slightly over. The honest report is now "same steps: PRT wins
quality and render speed; train wall is essentially tied and needs one more
structural fused-kernel improvement for a decisive exact-step wall-clock win."

Gate D3s tested the low-risk idea of specializing the fused-MSE kernel for the
current `h_terms == 3` camera-polynomial path. The attempted code used explicit
three-term `h`/`dh_dtau` evaluation and explicit three-term `grad_h_coeff`
atomics inside the fused kernel. It passed parity but was slower, so the code
change was reverted and only the rejection artifacts are kept.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py --out-json research_project/benchmarks/results/projective_rational_tile_pixel_fused_mse_backward_check_hterms3_specialized.json
python3 research_project/benchmarks/projective_rational_fused_mse_timing_probe.py --tube-counts 2048 --frames 8 --width 256 --height 256 --tile-config 4x4x1:512 --support-alpha-threshold 0.25098039215686274 --warmups 1 --repeats 5 --out-json research_project/benchmarks/results/projective_rational_fused_mse_timing_probe_2048_256_8f_support64_tile4x4x1_hterms3_specialized.json
```

Result:

```text
Parity: pass, loss abs error 0, max grad abs error 1.86e-09, overflow 0.
Projected fused MSE with h_terms==3 specialization: 15.64 ms.
Previous D3r projected fused MSE: 10.76 ms.
```

Read: reject this micro-specialization. The generic loop is apparently not the
current fused-MSE bottleneck, or the branch/extra helper shape makes register
pressure worse. Do not retry h-terms specialization unless it is coupled to a
larger replay-cache or gradient-accumulation rewrite with fresh profiling.

Gate D3t tests whether D3r leaves enough headroom to spend 195 PRT steps under
the paired 200-step direct-splat wall on the train72/eval64 support schedule.

Validation:

```text
python3 setup.py build_ext --inplace
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 195 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2823529411764706 --prt-eval-support-alpha-threshold 0.25098039215686274 --prt-extra-eval-support-alpha-thresholds 0.2823529411764706,0.2196078431372549,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt195_splat200_train72_eval64_extra72_56_48_loss_threadgroup_reduce_fused_mse.json
```

Result:

```text
195-step train72/eval64: PRT wall 6.638 s vs splat wall 7.081 s.
195-step eval64 PRT: PSNR 15.6399 / heldout 13.2195 dB, render 8.44 / 13.19 ms, max tile 116, overflow 0.
195-step splat:      PSNR 15.6136 / heldout 12.4152 dB, render 71.57 / 95.65 ms.
195-step eval72 PRT: PSNR 17.8086 / heldout 13.0072 dB, render 6.39 / 11.38 ms, max tile 81, overflow 0.
195-step eval56 PRT: PSNR 14.3863 / heldout 13.2086 dB, render 13.48 / 18.60 ms, max tile 170, overflow 0.
195-step eval48 PRT: PSNR 13.8160 / heldout 13.1270 dB, render 15.17 / 20.09 ms, max tile 214, overflow 0.
```

Read: 195 steps still fits under the paired splat wall and keeps the PSNR/render
wins, but it does not improve the quality row over D3p's 190-step result. Keep
D3p as the cleaner same-budget recommendation and D3t as the boundary point
showing the train72/eval64 schedule can spend up to roughly 195 steps before
becoming an exact-step noise-band result.

Gate D3u tested a tiny replay-reuse idea in the fused-MSE backward pass: reuse
the stored differentiable alpha for the opacity gradient as
`d_alpha * alpha / opacity` instead of recomputing `exp(-0.5 q)`. The source
change passed parity but did not improve timing, so it was reverted and only
the rejection artifacts are kept.

Validation:

```text
python3 setup.py build_ext --inplace
python3 research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py --out-json research_project/benchmarks/results/projective_rational_tile_pixel_fused_mse_backward_check_opacity_exp_reuse.json
python3 research_project/benchmarks/projective_rational_fused_mse_timing_probe.py --tube-counts 2048 --frames 8 --width 256 --height 256 --tile-config 4x4x1:512 --support-alpha-threshold 0.25098039215686274 --warmups 1 --repeats 5 --out-json research_project/benchmarks/results/projective_rational_fused_mse_timing_probe_2048_256_8f_support64_tile4x4x1_opacity_exp_reuse.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 256 --max-frames 8 --steps 20 --prt-tubes 2048 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 1 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_256_8f_2048t_20step_support64_fused_mse_opacity_exp_reuse.json
```

Result:

```text
Parity: pass, loss abs error 0, max grad abs error 1.86e-09, overflow 0.
Projected fused MSE with opacity exp reuse: 10.92 ms.
Previous D3r projected fused MSE: 10.76 ms.
20-step breakdown: median step 50.70 ms, fused MSE 41.31 ms, train loop 1.395 s.
Previous D3r 20-step breakdown: median step 50.84 ms, fused MSE 41.46 ms, train loop 1.301 s.
```

Read: reject the source change. It removes an `exp` but adds a divide, and the
focused timing probe was slightly slower while the real 20-step breakdown stayed
noise-band. Keep this as evidence that isolated scalar algebra inside the replay
loop is too small; the next speed attempt should change accumulation or replay
structure rather than swapping one scalar op.

Gate D3v reran the clean 190-step same-wall comparison on the current accepted
D3r loss-reduction kernel. The first attempt failed before training because the
local extension had been cleaned after the prior commit; `python3 setup.py
build_ext --inplace` restored op registration and the rerun completed.

Validation:

```text
python3 setup.py build_ext --inplace
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 190 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2823529411764706 --prt-eval-support-alpha-threshold 0.25098039215686274 --prt-extra-eval-support-alpha-thresholds 0.2823529411764706,0.2196078431372549,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt190_splat200_train72_eval64_extra72_56_48_loss_threadgroup_reduce_fused_mse.json
```

Result:

```text
Current-code 190-step train72/eval64: PRT wall 6.581 s vs splat wall 7.083 s.
Current-code 190-step eval64 PRT: PSNR 15.6459 / heldout 13.2360 dB, render 9.17 / 12.43 ms, max tile 109, overflow 0.
Current-code 190-step splat:      PSNR 15.6136 / heldout 12.4152 dB, render 79.69 / 90.91 ms.
Current-code 190-step eval72 PRT: PSNR 17.4682 / heldout 12.9143 dB, render 8.47 / 11.11 ms, max tile 76, overflow 0.
Current-code 190-step eval56 PRT: PSNR 14.5416 / heldout 13.2947 dB, render 11.89 / 16.69 ms, max tile 158, overflow 0.
Current-code 190-step eval48 PRT: PSNR 14.0778 / heldout 13.2606 dB, render 14.63 / 20.38 ms, max tile 202, overflow 0.
```

Read: this supersedes D3p as the current-code same-budget row. It is slightly
lower quality than the older replay-cleanup-only D3p row, but it preserves the
important claim on the accepted kernel: 190 PRT steps fit under the wall-clock
of a paired 200-step direct-splat run while still winning train PSNR, heldout
PSNR, and render speed. For heldout-only selection in this row, eval56 is best;
for train-overfit selection, eval72 is best; eval64 remains the balanced row.

Gate D3w tested tile-slot gradient reductions in the fused-MSE backward path for
the `STAR_TILE_T == 1` training configuration. The first variant reduced all
per-slot gradients across the tile and emitted one atomic set per slot. The
second reduced only color gradients and left the shape/opacity/camera gradients
on the existing per-pixel atomics. Both passed parity, but both were slower than
the accepted D3r kernel, so the source changes were reverted and only rejection
artifacts are kept.

Validation:

```text
python3 setup.py build_ext --inplace
STAR_UVT_TILE_X=4 STAR_UVT_TILE_Y=4 STAR_UVT_TILE_T=1 STAR_UVT_TILE_CAPACITY=512 python3 research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py --out-json research_project/benchmarks/results/projective_rational_tile_pixel_fused_mse_backward_check_tile_slot_reduce_tile4x4x1.json
python3 research_project/benchmarks/projective_rational_fused_mse_timing_probe.py --tube-counts 2048 --frames 8 --width 256 --height 256 --tile-config 4x4x1:512 --support-alpha-threshold 0.25098039215686274 --warmups 1 --repeats 5 --out-json research_project/benchmarks/results/projective_rational_fused_mse_timing_probe_2048_256_8f_support64_tile4x4x1_tile_slot_reduce.json
STAR_UVT_TILE_X=4 STAR_UVT_TILE_Y=4 STAR_UVT_TILE_T=1 STAR_UVT_TILE_CAPACITY=512 python3 research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py --out-json research_project/benchmarks/results/projective_rational_tile_pixel_fused_mse_backward_check_tile_slot_color_reduce_tile4x4x1.json
python3 research_project/benchmarks/projective_rational_fused_mse_timing_probe.py --tube-counts 2048 --frames 8 --width 256 --height 256 --tile-config 4x4x1:512 --support-alpha-threshold 0.25098039215686274 --warmups 1 --repeats 5 --out-json research_project/benchmarks/results/projective_rational_fused_mse_timing_probe_2048_256_8f_support64_tile4x4x1_tile_slot_color_reduce.json
```

Result:

```text
Full tile-slot reduction parity: pass, loss abs error 2.98e-08, max grad abs error 1.86e-09.
Full tile-slot reduction projected fused MSE: 18.58 ms.

Color-only tile-slot reduction parity: pass, loss abs error 2.98e-08, max grad abs error 9.31e-10.
Color-only tile-slot reduction projected fused MSE: 14.04 ms.

Previous accepted D3r projected fused MSE: 10.76 ms.
```

Read: reject threadgroup slot reductions for the current 4x4x1 training tile.
The barrier cost dominates the saved global atomics at this tile size. The next
train-speed attempt should avoid per-slot threadgroup reductions and instead
look for fewer replay passes, cheaper ordering/support, or a separate compact
sample-gradient path with an actually cheap reducer.

Gate D3x reran the backward phase profile on the current accepted D3r
loss-reduction kernel and added `--prt-train-mode` to the profile CLI so it can
exercise the same fused-MSE path as the real comparison rows.

Validation:

```text
python3 research_project/benchmarks/projective_rational_multicam_backward_phase_profile.py --target-size 256 --max-frames 8 --steps 20 --prt-tubes 2048 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --profile-warmups 1 --profile-repeats 5 --out-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_256_8f_2048t_20step_support64_tile4x4x1_current_d3r.json
python3 research_project/benchmarks/projective_rational_trace_cache_planner.py --profile-json research_project/benchmarks/results/projective_rational_multicam_backward_phase_profile_256_8f_2048t_20step_support64_tile4x4x1_current_d3r.json --out-json research_project/benchmarks/results/projective_rational_trace_cache_planner_256_8f_2048t_support64_tile4x4x1_current_d3r.json --projection current=256x256x8 --projection halfres16=128x128x16 --projection full512_16=512x512x16
```

Result:

```text
Current D3r profile: active tiles 32768, tile-pixel-tube visits 15376848, max tile count 119, p50/p90/p95/p99 tile counts 23/63/75/94, overflow 0.
Diagnostic profile medians: total 40.79 ms, bin tubes 4.65 ms, backward kernel 35.60 ms, compute-only kernel 12.90 ms, replay-only kernel 9.87 ms.
Fused trainer row in same run: median step 52.81 ms, fused MSE 41.44 ms, train loop 1.838 s, loss 0.06337 -> 0.03225.

Trace-cache planner current 256x256x8: sparse id+alpha+t_before upper bound 179.97 MiB; dense slots 3074 MiB.
Trace-cache planner full512x16 projection: sparse id+alpha+t_before upper bound 1439.79 MiB; dense slots 24592 MiB.
Timing decomposition: derivative math estimate 3.03 ms, replay-only 9.87 ms, atomic write estimate 22.71 ms, optimistic cached-backward floor 25.73 ms.
```

Read: D3x says a naive replay trace cache is not the next clean unlock. Dense
traces are far too large, compact sparse traces are plausible only at current
scale and become multi-GiB at fuller rows, and even a perfect replay cache
leaves a roughly 25.7 ms backward floor because gradient atomics dominate. The
next structural train-speed idea should either change the gradient write shape
without per-slot tile barriers, reduce the number of visited tile-pixel-tube
pairs, or fuse a cheaper sample-gradient accumulation path instead of caching
all replay state.

Gate D3y adds the explicit train-used-gradient fused-MSE op from the
three-agent review. It returns the same result shape as the full fused op but
the Metal kernel skips `lambda_uv` and `center_t` gradient math/writes, because
the current train path does not train `lambda_uv` and `center_t` is effectively
fixed by the world-tube `t0`. Unlike D3s/D3u, this changes the write set;
unlike D3w, it does not add per-slot threadgroup reductions or barriers.

Validation:

```text
python3 setup.py build_ext --inplace
STAR_UVT_TILE_X=4 STAR_UVT_TILE_Y=4 STAR_UVT_TILE_T=1 STAR_UVT_TILE_CAPACITY=512 python3 research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py --fused-mode train_used --abs-tol 5e-9 --rel-tol 0 --loss-tol 1e-7 --out-json research_project/benchmarks/results/projective_rational_tile_pixel_fused_mse_backward_check_train_used_tile4x4x1.json
STAR_UVT_TILE_X=4 STAR_UVT_TILE_Y=4 STAR_UVT_TILE_T=1 STAR_UVT_TILE_CAPACITY=512 python3 research_project/benchmarks/projective_rational_tile_pixel_fused_mse_backward_check.py --fused-mode full --out-json research_project/benchmarks/results/projective_rational_tile_pixel_fused_mse_backward_check_full_after_train_used_tile4x4x1.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 256 --max-frames 8 --steps 20 --prt-tubes 2048 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse --render-warmups 1 --render-repeats 1 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_256_8f_2048t_20step_support64_fused_mse_d3y_baseline_rerun.json
python3 research_project/benchmarks/projective_rational_multicam_train_breakdown.py --target-size 256 --max-frames 8 --steps 20 --prt-tubes 2048 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.25098039215686274 --prt-loss-mode sequence --prt-train-mode fused_mse_train_used --render-warmups 1 --render-repeats 1 --out-json research_project/benchmarks/results/projective_rational_multicam_train_breakdown_256_8f_2048t_20step_support64_fused_mse_train_used_d3y.json
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 200 --prt-steps 200 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2823529411764706 --prt-eval-support-alpha-threshold 0.25098039215686274 --prt-extra-eval-support-alpha-thresholds 0.2823529411764706,0.2196078431372549,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse_train_used --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samesteps_prt200_splat200_train72_eval64_extra72_56_48_fused_mse_train_used_d3y.json
```

Result:

```text
Train-used parity: pass, loss abs error 0, max used-grad abs error 1.86e-09, overflow 0. Checked h_coeff, lambda_t, opacity, color; skipped lambda_uv and center_t by design.
Full fused op after sibling addition: pass, loss abs error 2.98e-08, max grad abs error 3.73e-09, overflow 0.

20-step baseline rerun: median fused MSE 41.33 ms, median step 50.29 ms, train loop 1.324 s.
20-step train-used D3y: median fused MSE 33.92 ms, median step 42.05 ms, train loop 1.120 s.
Improvement: fused MSE 17.9%, step 16.4%; loss still decreased, overflow 0.

Exact 200-step D3y row: PRT wall 5.622 s vs splat wall 6.851 s.
D3y eval64 PRT: PSNR 16.0183 / heldout 13.1600 dB, render 7.22 / 8.40 ms, max tile 112, overflow 0.
D3y splat:      PSNR 15.6132 / heldout 12.4150 dB, render 69.50 / 67.23 ms.
D3y eval72 PRT: PSNR 18.0062 / heldout 12.7301 dB, render 5.90 / 7.29 ms, max tile 69, overflow 0.
D3y eval56 PRT: PSNR 14.5761 / heldout 13.2414 dB, render 9.96 / 11.04 ms, max tile 152, overflow 0.
D3y eval48 PRT: PSNR 13.9711 / heldout 13.2702 dB, render 13.50 / 14.59 ms, max tile 190, overflow 0.
```

Read: D3y is the first accepted train-speed change that turns exact same-step
training into a clean wall-clock win, not just a noise-band tie. The 200-step
PRT row is about 18% faster than the paired 200-step direct-splat row while
keeping the train PSNR, heldout PSNR, and render-speed wins. The tradeoff is
explicit: D3y is a training op for the current model where `lambda_uv` and
`center_t` are not train-used parameter families. Keep the full fused op for
diagnostics or future runs that train those parameters.

Gate D3z tests whether the D3y train-wall margin should be spent on more PRT
steps. The row compares 240 D3y PRT steps against the same paired 200-step
direct-splat baseline.

Command:

```text
STAR_UVT_TILE_T=1 python3 research_project/benchmarks/projective_rational_multicam_splat_compare.py --target-size 256 --max-frames 8 --steps 240 --prt-steps 240 --splat-steps 200 --prt-tubes 2048 --splat-count 2048 --splat-renderer fast_mac --init-depth 0.5 --tile-config 4x4x1:512 --prt-support-alpha-threshold 0.2823529411764706 --prt-eval-support-alpha-threshold 0.25098039215686274 --prt-extra-eval-support-alpha-thresholds 0.2823529411764706,0.2196078431372549,0.18823529411764706 --prt-loss-mode sequence --prt-train-mode fused_mse_train_used --render-warmups 1 --render-repeats 3 --prt-eval-cache-compiled --out-json research_project/benchmarks/results/projective_rational_multicam_splat_compare_256_8f_2048t_2048s_samewall_prt240_splat200_train72_eval64_extra72_56_48_fused_mse_train_used_d3z.json
```

Result:

```text
D3z 240-step PRT wall: 7.647 s.
D3z paired 200-step splat wall: 6.718 s.

D3z eval64 PRT: PSNR 16.0449 / heldout 13.1298 dB, render 8.49 / 11.37 ms, max tile 100, overflow 0.
D3z splat:      PSNR 15.6135 / heldout 12.4150 dB, render 77.91 / 117.99 ms.
D3z eval72 PRT: PSNR 18.4502 / heldout 12.8720 dB, render 7.06 / 8.87 ms, max tile 68, overflow 0.
D3z eval56 PRT: PSNR 14.6264 / heldout 13.3259 dB, render 11.92 / 13.10 ms, max tile 139, overflow 0.
D3z eval48 PRT: PSNR 14.0088 / heldout 13.3049 dB, render 13.66 / 15.43 ms, max tile 180, overflow 0.
```

Read: D3z rejects spending the full D3y wall-clock margin on 240 PRT steps.
The extra 40 steps push PRT over the paired splat train wall by about 0.93 s,
while the balanced eval64 source PSNR moves only from D3y's 16.0183 dB to
16.0449 dB and heldout slightly drops. D3y remains the clean accepted row:
200-vs-200 exact steps, faster train wall, higher train/heldout PSNR, and much
faster render. If we spend the wall margin at all, it should be a smaller
schedule search around roughly 210-225 steps or a different optimizer/support
schedule, not a blind 240-step run.

Gate D4a runs that smaller schedule search at 205/210/220 PRT steps against
paired 200-step fast-mac direct-splat baselines. All rows use the same D3y
train-used fused-MSE op, train72/eval64 support schedule, 2048 PRT tubes,
2048 direct splats, 256px, 8 frames, and cached compiled PRT eval.

Result:

```text
D3y accepted row: PRT 200 steps 5.622 s vs splat 200 steps 6.851 s; eval64 PSNR 16.0183 / heldout 13.1600 dB; render 7.22 / 8.40 ms.

D4a 220-step row: PRT 6.210 s vs splat 5.888 s; delta +0.321 s. Eval64 PSNR 16.0298 / heldout 13.2719 dB; render 7.88 / 13.04 ms; max tile 110; overflow 0.
D4b 210-step row: PRT 5.945 s vs splat 5.834 s; delta +0.111 s. Eval64 PSNR 15.9993 / heldout 13.2027 dB; render 11.25 / 14.15 ms; max tile 118; overflow 0.
D4c 205-step row: PRT 5.669 s vs splat 5.842 s; delta -0.173 s. Eval64 PSNR 15.9945 / heldout 13.0611 dB; render 7.83 / 10.53 ms; max tile 115; overflow 0.
```

Read: do not replace D3y with a step-spend row. Under the faster current
paired-splat wall, 205 steps is the only under-wall point, and it is lower
quality than D3y on both balanced train PSNR and heldout PSNR. 210 and 220 keep
the PSNR/render win over splats, but both miss the paired train wall. The useful
next path is not blind extra steps; it is either a better optimizer/support
schedule or the next representation/rasterizer branch.

Gate F0 implements the first projection/render-only falsifier for the separate
depth-banded homography-flow gauge residual-tube idea. The script keeps the
learned object as `N` world tubes, compiles four representative depth-band
homography flows from the render camera path, assigns each tube by reference
depth, and stores/renders only a low-degree residual center path:

```text
p_i(tau) = F_b(u_i0, v_i0, tau) + r_i(tau)
```

The dense render uses exact direct depth for the gauge row so this gate isolates
center residual before any Metal/backward work.

Command:

```text
python3 research_project/benchmarks/depth_banded_homography_flow_residual_probe.py --out-json research_project/benchmarks/results/depth_banded_homography_flow_residual_probe_f0_128_32f_256t_degree1.json
python3 research_project/benchmarks/depth_banded_homography_flow_residual_probe.py --residual-degree 2 --out-json research_project/benchmarks/results/depth_banded_homography_flow_residual_probe_f0_128_32f_256t_degree2.json
```

Result:

```text
F0 degree1: pass false. Center p95 0.6911 px vs projective_first_order 3.6731 px; max 2.6097 px; render PSNR 50.34 dB; flow-sheared tile pairs 16575 vs segmented_f4 18455; rendered tubes 256 vs segmented_f4 1024.
F0 degree2: pass true.  Center p95 0.1419 px vs projective_first_order 3.6731 px; max 0.6357 px; render PSNR 64.55 dB; flow-sheared tile pairs 16586 vs segmented_f4 18455; rendered tubes 256 vs segmented_f4 1024.
Segmented_f4: center p95 0.2147 px; render PSNR 56.80 dB; rendered tubes 1024.
PRT degree2: center p95 1.08e-05 px; render PSNR 120 dB; rendered tubes 256.
```

Read: F0 does not replace PRT as the exact fallback; PRT is still the correctness
anchor. It does show the gauge-residual branch is worth keeping alive. Degree 1
fails the max-residual gate despite beating projective-first-order on p95 and
PSNR. Degree 2 passes the implemented residual, PSNR, tile-estimate, and
rendered-tube gates, and beats segmented f4 on p95, render PSNR, tile-pair
estimate, and rendered-tube count. The actual flow-sheared render-time gate is
deferred until that renderer exists. Next work should turn the degree-2
residual upper-bound into an actual flow-sheared tile renderer or test
robustness with object velocity and harder camera paths.

Gate F0b runs that robustness check:

```text
python3 research_project/benchmarks/depth_banded_homography_flow_residual_probe.py --residual-degree 2 --velocity-scale 0.01 --out-json research_project/benchmarks/results/depth_banded_homography_flow_residual_probe_f0_128_32f_256t_degree2_velocity001.json
python3 research_project/benchmarks/depth_banded_homography_flow_residual_probe.py --residual-degree 2 --pan-x 0.09 --zoom 0.025 --dolly-z 0.12 --out-json research_project/benchmarks/results/depth_banded_homography_flow_residual_probe_f0_128_32f_256t_degree2_hardcam.json
python3 research_project/benchmarks/depth_banded_homography_flow_residual_probe.py --residual-degree 3 --pan-x 0.09 --zoom 0.025 --dolly-z 0.12 --out-json research_project/benchmarks/results/depth_banded_homography_flow_residual_probe_f0_128_32f_256t_degree3_hardcam.json
python3 research_project/benchmarks/depth_banded_homography_flow_residual_probe.py --residual-degree 3 --velocity-scale 0.01 --pan-x 0.09 --zoom 0.025 --dolly-z 0.12 --out-json research_project/benchmarks/results/depth_banded_homography_flow_residual_probe_f0_128_32f_256t_degree3_hardcam_velocity001.json
```

Result:

```text
Velocity 0.01, degree2: pass true. p95 0.2157 px, max 0.9358 px, PSNR 59.79 dB, tile pairs 16533 vs segmented 18529.
Hard camera, degree2: pass false. p95 0.5392 px, max 2.8076 px, PSNR 53.81 dB, tile pairs 16634 vs segmented 19620.
Hard camera, degree3: pass true. p95 0.1536 px, max 0.8581 px, PSNR 62.05 dB, tile pairs 16635 vs segmented 19620.
Hard camera + velocity 0.01, degree3: pass false. p95 0.1993 px, max 1.2722 px, PSNR 58.54 dB, tile pairs 16586 vs segmented 19659.
```

Read: F0b gives a clean boundary. Degree-2 residual is robust to mild object
motion under the original stress. Harder camera motion needs degree 3 to pass
the max-residual gate. Combining hard camera motion with object motion breaks
the strict max-residual gate even at degree 3, though p95, PSNR, and tile-pair
estimates remain good. That argues for a hybrid policy: gauge-residual for low
residual/common background tubes, PRT fallback or window split for high-max
outliers and moving objects under hard camera motion.

Gate F0c extends the probe with per-tube max-residual outlier counts and an
upper-bound hybrid row that replaces only outlier tubes with PRT centers.

Command:

```text
python3 research_project/benchmarks/depth_banded_homography_flow_residual_probe.py --residual-degree 3 --velocity-scale 0.01 --pan-x 0.09 --zoom 0.025 --dolly-z 0.12 --out-json research_project/benchmarks/results/depth_banded_homography_flow_residual_probe_f0_hybrid_128_32f_256t_degree3_hardcam_velocity001.json
```

Result:

```text
Pure degree3 gauge, hard camera + velocity 0.01: pass false. p95 0.1993 px, max 1.2722 px, PSNR 58.54 dB, tile pairs 16586.
Outliers above 1px max: 6 / 256 tubes, 2.34%.
Hybrid gauge + PRT fallback: pass true. p95 0.1695 px, max 0.9978 px, PSNR 59.81 dB, tile pairs 16668 = 15952 gauge residual + 716 PRT fallback.
Segmented_f4 reference: p95 0.4347 px, PSNR 51.45 dB, tile pairs 19659, rendered tubes 1024.
```

Read: the fallback policy is plausible. The hard camera/object-motion failure is
not a broad collapse; it is a small outlier tail. Replacing 6 high-residual
tubes with PRT centers clears the max-residual gate while keeping the tile-pair
estimate below segmented_f4 and the rendered tube count at N. This is still an
upper-bound probe, not a renderer claim: the next real implementation would be
a hybrid flow-sheared tiled renderer plus PRT fallback path.

Gate F0d checks whether that small fallback tail is stable across seeds. It
wraps the F0c hard-camera/object-motion configuration in a four-seed sweep:

```text
python3 research_project/benchmarks/depth_banded_homography_flow_hybrid_seed_sweep.py --out-json research_project/benchmarks/results/depth_banded_homography_flow_hybrid_seed_sweep_f0d_hardcam_velocity001_4seeds.json
```

Result:

```text
Seeds: 17, 23, 31, 47.
Pure gauge pass count: 1 / 4.
Hybrid pass count: 4 / 4.
Fallback tubes: min 0, median 3, max 6 out of 256.
Hybrid max residual: min 0.8084 px, median 0.9545 px, max 0.9978 px.
Hybrid p95 residual: min 0.1695 px, median 0.1826 px, max 0.2007 px.
Hybrid PSNR: min 59.81 dB, median 60.70 dB, max 61.30 dB.
Hybrid tile-pair ratio vs segmented_f4: min 0.830, median 0.847, max 0.848.
```

Read: F0d strengthens the hybrid policy read. Pure gauge is too brittle under
hard camera plus object motion, but a small PRT fallback tail clears all four
sampled seeds while still estimating fewer tile pairs than segmented_f4. The
next implementation step should be an actual flow-sheared tiled forward path
with fallback routing, not more dense upper-bound probes.

Gate F0e keeps the F0d seed set fixed and sweeps estimated spatial tile size for
the hybrid fallback upper bound:

```text
python3 research_project/benchmarks/depth_banded_homography_flow_hybrid_seed_sweep.py --tile-x 4 --tile-y 4 --out-json research_project/benchmarks/results/depth_banded_homography_flow_hybrid_seed_sweep_f0e_hardcam_velocity001_4seeds_tile4.json
python3 research_project/benchmarks/depth_banded_homography_flow_hybrid_seed_sweep.py --tile-x 16 --tile-y 16 --out-json research_project/benchmarks/results/depth_banded_homography_flow_hybrid_seed_sweep_f0e_hardcam_velocity001_4seeds_tile16.json
```

The default F0d row is the matching 8x8 tile estimate.

Result:

```text
4x4 tiles:  pass true, hybrid pass 4 / 4, tile-pair ratio vs segmented_f4 min 0.810, median 0.815, max 0.824.
8x8 tiles:  pass true, hybrid pass 4 / 4, tile-pair ratio vs segmented_f4 min 0.830, median 0.847, max 0.848.
16x16 tiles: pass true, hybrid pass 4 / 4, tile-pair ratio vs segmented_f4 min 0.880, median 0.895, max 0.908.
Shared quality across tile estimates: fallback tubes min 0, median 3, max 6; hybrid max residual min 0.8084 px, median 0.9545 px, max 0.9978 px; hybrid p95 min 0.1695 px, median 0.1826 px, max 0.2007 px; hybrid PSNR min 59.81 dB, median 60.70 dB, max 61.30 dB.
```

Read: the residual/fallback decision is independent of the estimated tile size,
as expected. On this hard-camera/object-motion stress, smaller 4x4 spatial tiles
give the lowest tile-pair estimate against segmented_f4. That does not prove
4x4 is the runtime winner, because a real Metal flow-sheared renderer will pay
tile bookkeeping and dispatch costs. It does say the first implementation should
make tile shape explicit and measure 4x4 against 8x8 in the actual renderer,
rather than assuming the older 8x8 UVT tile shape is still best.

Gate F0f checks whether F0e's tile-pair win comes from the residual-coordinate
flow-sheared frame or from ordinary image-space center culling:

```text
python3 research_project/benchmarks/depth_banded_homography_flow_culling_control.py --out-json research_project/benchmarks/results/depth_banded_homography_flow_culling_control_f0f_hardcam_velocity001_4seeds_tiles4_8_16.json
```

Result:

```text
4x4 residual-coordinate hybrid ratio vs segmented_f4: min 0.810, median 0.815, max 0.824.
4x4 image-space hybrid control ratio vs segmented_f4: min 0.998, median 0.999, max 1.001.
4x4 residual-coordinate savings vs image-space control: min 9975, median 10631, max 10893 tile pairs.

8x8 residual-coordinate hybrid ratio vs segmented_f4: min 0.830, median 0.847, max 0.848.
8x8 image-space hybrid control ratio vs segmented_f4: min 0.998, median 0.999, max 1.002.
8x8 residual-coordinate savings vs image-space control: min 2953, median 3047.5, max 3344 tile pairs.

16x16 residual-coordinate hybrid ratio vs segmented_f4: min 0.880, median 0.895, max 0.908.
16x16 image-space hybrid control ratio vs segmented_f4: min 0.999, median 1.000, max 1.002.
16x16 residual-coordinate savings vs image-space control: min 764, median 872, max 986 tile pairs.
```

Read: the win is specifically the flow-sheared residual coordinate system, not
generic screen-space culling. Ordinary image-space culling is approximately
segmented_f4 cost on this stress, while residual-coordinate culling keeps the
hybrid below segmented_f4 for every seed/tile row. This makes the next renderer
target sharper: implement residual-coordinate tile assignment plus flow-sheared
screen evaluation, with a PRT fallback list for high-residual tubes.

Gate F0g makes that culling target stricter. Raw residual-only bins are compact,
but a renderer still needs an absolute output location. The culling-control
script now also estimates `reference_atlas + residual` bins before applying the
depth-band homography warp:

```text
python3 research_project/benchmarks/depth_banded_homography_flow_culling_control.py --out-json research_project/benchmarks/results/depth_banded_homography_flow_culling_control_f0g_reference_atlas_hardcam_velocity001_4seeds_tiles4_8_16.json
```

Result:

```text
4x4 reference-atlas hybrid ratio vs segmented_f4: min 0.818, median 0.823, max 0.828.
4x4 image-space hybrid control ratio vs segmented_f4: min 0.998, median 0.999, max 1.001.
4x4 reference-atlas savings vs image-space control: min 9739, median 10198.5, max 10412 tile pairs.

8x8 reference-atlas hybrid ratio vs segmented_f4: min 0.849, median 0.852, max 0.863.
8x8 image-space hybrid control ratio vs segmented_f4: min 0.998, median 0.999, max 1.002.
8x8 reference-atlas savings vs image-space control: min 2654, median 2934, max 2988 tile pairs.

16x16 reference-atlas hybrid ratio vs segmented_f4: min 0.887, median 0.893, max 0.897.
16x16 image-space hybrid control ratio vs segmented_f4: min 0.999, median 1.000, max 1.002.
16x16 reference-atlas savings vs image-space control: min 856, median 885, max 930 tile pairs.
```

Read: the stricter reference-atlas estimate keeps the culling win alive. It is
slightly weaker than raw residual-only bins for 4x4/8x8, but still far below
ordinary image-space culling and segmented_f4 on every row. That prevents the
wrong implementation target: do not build a raw residual-only rasterizer. Build
a reference-atlas-plus-residual tile path, then evaluate the depth-band
homography warp into screen space.

Gate F0h fixes the remaining representation mismatch. Earlier F0 rows fit the
residual as a screen-space delta. A renderer-facing atlas path should instead
invert each depth-band homography, fit the residual in reference-atlas
coordinates, then warp `reference_uv + residual_atlas(t)` forward.

Command:

```text
python3 research_project/benchmarks/depth_banded_homography_flow_atlas_residual_probe.py --out-json research_project/benchmarks/results/depth_banded_homography_flow_atlas_residual_probe_f0h_hardcam_velocity001_4seeds_tiles4_8_16.json
```

Result:

```text
All 12 seed/tile rows pass with zero fallback tubes.
Atlas-residual hybrid center max: min 0.3049 px, median 0.4606 px, max 0.5828 px.
Atlas-residual hybrid PSNR: min 73.98 dB, median 74.98 dB, max 77.15 dB.

4x4 atlas ratio vs segmented_f4: min 0.817, median 0.819, max 0.828.
4x4 image-space control ratio vs segmented_f4: min 0.999, median 0.999, max 1.001.
4x4 atlas savings vs image-space control: min 9926, median 10304.5, max 10550 tile pairs.

8x8 atlas ratio vs segmented_f4: min 0.850, median 0.854, max 0.855.
8x8 image-space control ratio vs segmented_f4: min 0.998, median 1.000, max 1.001.
8x8 atlas savings vs image-space control: min 2811, median 2910.5, max 2981 tile pairs.

16x16 atlas ratio vs segmented_f4: min 0.886, median 0.894, max 0.903.
16x16 image-space control ratio vs segmented_f4: min 0.999, median 1.000, max 1.001.
16x16 atlas savings vs image-space control: min 802, median 879.5, max 934 tile pairs.
```

Read: F0h is the current best representation target for the flow-sheared path.
Fitting residuals in atlas coordinates is cleaner than adding screen-space
residuals after the homography: the hard-camera/object-motion row no longer
needs any PRT fallback under these four seeds, while preserving the 4x4 culling
advantage. The fallback route still matters for tougher rows and as a safety
valve, but the first Metal renderer should implement inverse-homography
atlas-residual tubes rather than the earlier screen-additive residual variant.

The separate representation idea tested by F0 is
depth-banded homography-flow gauge residual tubes. Compile a small bank of
camera-induced depth-band flows from `K_seq,w2c_seq`, let each world tube store
only a low-degree residual,

```text
p_i(tau) = F_b(u_i0, v_i0, tau) + r_i(tau),
```

and route low-residual tubes back through the cheap affine UVT path while
falling back to PRT for high residual/high curvature tubes. This preserves the
world-object contract because `F_b` is render-time compiler state, not learned
camera-specific scene state. First gate should be projection/render only:
128px, 32 frames, 256 tubes, 4 depth bands, residual degree 1 first, compared
against per-frame reference, projective first-order, segmented_f4, and PRT.
Kill it before Metal/backward if it needs more than `N` rendered tubes, render
PSNR versus per-frame reference is below 50 dB, center residual max stays above
1 px, or tile-pair estimates are not below segmented_f4.

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
7. Do not globally promote 2048 by tube count alone: support `48/255` is the balanced 128px x 8f row, while 256px prefers `64/255` for speed; the selector needs target-size or density context before 2048 can become `--prt-tile-policy train_speed`.
8. Treat static split train/eval support as a diagnostic, but keep support scheduling alive: D3o shows train72/eval64 is a faster under-wall candidate while train74+ is too tight for overfit PSNR; D3p and current-code D3v show train72/eval64 can spend that wall saving on 190 PRT steps and still finish under one 200-step splat wall; D3t shows 195 steps is still under wall but not a quality improvement; D3q/D3r show exact 200-vs-200 steps was a quality/render win but a train-wall near tie before D3y; D3z shows 240 D3y PRT steps overspend the wall margin for little balanced-quality gain; D4a shows 205 is the only under-wall point in the 205/210/220 sweep and does not beat D3y quality.
9. Move gradient accumulation structure, derivative-math simplification, and trace/replay reuse to the front of the train-speed queue; D3m/D3n remove redundant fused-kernel sorting and replay bookkeeping, D3r removes loss atomics, D3s/D3u reject isolated scalar-loop micro-specializations, D3w rejects per-slot threadgroup reductions at 4x4x1, D3x rejects naive replay caching, and D3y shows write-set pruning can turn exact 200-vs-200 into a train-wall win.
10. Promote the cached or fused camera-compiler path from benchmark flag to the intended playback and bake contract.
11. Keep playback/bake speed and training speed as separate claims: D3k supports the sublinear render story, D3y is the first current-code exact-step train-wall win for the current fixed-`lambda_uv`/fixed-`center_t` model, and D3z/D4a are boundary results showing that more steps must still fit the train-wall budget and improve quality before replacing D3y.
12. Continue depth-banded homography-flow gauge residual tubes after F0-F0h: degree-2 residual passed the clean projection/render falsifier and mild object-motion row, hard camera needs degree 3, and hard camera plus object motion originally needed a small PRT fallback/window-split tail under screen-additive residuals. F0h is now the better renderer target: inverse-homography atlas residuals pass all four hard-camera/object-motion seeds with no fallback tubes, high PSNR, and the same 4x4 culling advantage. The current implementation is still a dense upper-bound with exact direct depth, not a Metal flow-sheared renderer or training path.
