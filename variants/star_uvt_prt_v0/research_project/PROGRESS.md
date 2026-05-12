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
- [ ] Gate B5: production PRT tile config selection and integration.
- [ ] Gate C: PRT backward parity.
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
8x8x2:512: pass, max tile count 303, overflow 0, tiled/direct ratio 0.8764672143637684
4x4x2:128: fail, max tile count 238, overflow tiles 80
4x4x2:256: pass, max tile count 238, overflow 0, tiled/direct ratio 0.5893798248457015
```

Read: capacity and tile shape have to be chosen together. `4x4` fixes the
hotspot shape, but only cap 256 keeps all active tile lists valid. `8x8` with
cap 512 is valid but slower on this smoke, so shape splitting beats a pure
capacity bump here.

The new idea added in this fork is the curvature-selective hybrid compiler:
low-curvature tubes can stay on the old affine UVT path, while only high-curvature
moving-camera tubes use PRT. That is meant to preserve STAR-UVT's cheap path
instead of forcing every tube through the more expensive rational shader.

The next idea queued after tiled Metal forward is residual-certified footprint
inflation: use a small exact-projection probe to estimate the camera polynomial's
pixel residual and inflate only the PRT support bound needed to cover it. This
should be measured before splitting a camera window.

## Next Gates

1. Add a production PRT tile config selector instead of requiring manual env flags.
2. Add tile-load scaling scenes that stress moving-camera curvature.
3. Decide whether stable depth shortcuts are worth adding or whether sample-level ordering is the right first training path.
4. Add timing flags for `--uvt-camera-sequence-mode projective_rational`.
5. Only after forward timing, add direct-atomic exploratory backward plus a deterministic reporting fallback.
