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
- [ ] Gate B2: tiled PRT timing and tile-load scaling against direct PRT.
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

The new idea added in this fork is the curvature-selective hybrid compiler:
low-curvature tubes can stay on the old affine UVT path, while only high-curvature
moving-camera tubes use PRT. That is meant to preserve STAR-UVT's cheap path
instead of forcing every tube through the more expensive rational shader.

The next idea queued after tiled Metal forward is residual-certified footprint
inflation: use a small exact-projection probe to estimate the camera polynomial's
pixel residual and inflate only the PRT support bound needed to cover it. This
should be measured before splitting a camera window.

## Next Gates

1. Add a timing probe for tiled PRT vs direct PRT and affine STAR-UVT.
2. Add tile-load scaling scenes that stress moving-camera curvature.
3. Decide whether stable depth shortcuts are worth adding or whether sample-level ordering is the right first training path.
4. Add timing flags for `--uvt-camera-sequence-mode projective_rational`.
5. Only after forward timing, add direct-atomic exploratory backward plus a deterministic reporting fallback.
