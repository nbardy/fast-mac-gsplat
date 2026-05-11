# Phase 3: Backward Owner

## Goal

Decide whether training gradients should be owned by a Metal backward path or
by a higher-level projected renderer for early experiments.

## Candidate Gates

- Dense PyTorch backward remains the correctness reference.
- Metal backward must pass small-scene gradient parity before benchmarking.
- Any backward timing claim must include a side-by-side baseline and a parity
  check, not an isolated timing row.

## Non-Goals

Do not promote the UVT path into the default renderer because one tiny forward
smoke is fast. This phase needs full-matrix evidence.

## Current Slice

Phase 3a keeps dense PyTorch as the gradient owner and checks that projected
tube parameters receive usable gradients:

```bash
python3 research_project/trainer_harness/gradient_probe.py
```

This does not implement Metal backward. It establishes the local correctness
reference that a future Metal backward must match.

Phase 3b adds a hybrid autograd bridge:

```bash
python3 research_project/trainer_harness/metal_autograd_smoke.py
```

The bridge uses Metal forward and dense PyTorch backward. This is useful for
trainer plumbing and gradient-reference experiments, but it is not the true
Metal backward owner required before training-speed claims.

Phase 3c adds a simplified true-Metal backward probe:

```bash
python3 research_project/trainer_harness/simple_metal_backward_smoke.py
```

It computes per-sample gradients in Metal and reduces them in PyTorch, then
compares against dense autograd for a single tube. This still does not cover the
full sorted-tile, alpha-compositing, unstable-depth fallback backward.

Phase 3d adds a stable sorted-tile compositing backward probe:

```bash
python3 research_project/trainer_harness/stable_metal_backward_smoke.py
```

It reuses the Metal bin/sort tile path and computes per-sample compositing
gradients in Metal for stable tiles. It still does not cover unstable per-sample
depth fallback, gradients through the discrete depth order, or a production
backward reduction kernel.

Phase 3e extends the same probe to unstable fallback ordering:

```bash
python3 research_project/trainer_harness/unstable_metal_backward_smoke.py
```

This covers deterministic per-sample depth ordering on a crossing-depth scene.
It still leaves production reduction and performance work open.

Phase 3f wires the Metal backward probe into an autograd bridge with on-device
MPS reduction:

```bash
python3 research_project/trainer_harness/tile_metal_autograd_smoke.py
```

This is now a usable prototype gradient owner for small research smokes. It
still needs large-scene performance validation before any training-speed claim.

Phase 3g adds a bounded backward performance smoke:

```bash
python3 research_project/benchmarks/backward_performance_smoke.py
```

This checks that the prototype gradient owner runs and reports timing on a
small synthetic case. It is not promotion evidence.

Current result with 1 warmup iteration and 2 measured iterations on a 16-tube,
32x32x4 synthetic case: dense MPS mean `16.18629200675059 ms`; Metal
tile-backward mean `36.01418749894947 ms`. The dense path is faster at tiny
scale.

Phase 3h adds a bounded large-scene timing matrix:

```bash
python3 research_project/benchmarks/backward_performance_matrix.py
```

Current `large_local` result with 64 tubes, 64x64 resolution, 8 frames, 1 warmup
iteration, and 1 measured iteration: dense MPS mean `104.00879199733026 ms`;
Metal tile-backward mean `73.45674998941831 ms`; dense-to-Metal mean ratio
`1.4159187823081347`. Treat this only as local bounded timing evidence.
