# STAR-PRT v0 Design Note

STAR-PRT is a comparison fork for the variable-camera problem. It does not
replace `star_uvt_v0`; it keeps the same compact tube spirit while changing the
screen trajectory contract.

## Compared Lanes

- Current static STAR UVT stores a fixed screen-space center, UVT precision,
  affine temporal motion, depth slope, opacity, and color. It is efficient, but
  camera motion is only represented through the chosen projection/initialization
  path.
- Projective STAR UVT keeps the existing UVT tensor contract and improves how
  world tubes are projected into that affine screen tube. It is the lowest-risk
  path because the renderer and backward API remain unchanged.
- STAR-PRT stores homogeneous image-curve coefficients `h(t) = (u*w, v*w, w)`.
  The screen center is evaluated as `(h_x / h_z, h_y / h_z)`, so moving cameras
  and moving world points can stay projective instead of being collapsed into a
  single affine Taylor approximation.
- Compiled curve tubes are the pragmatic comparison baseline. They sample exact
  per-frame centers and depth, then fit or store a compact screen curve. This
  tests whether curve storage is enough before committing to full rational Metal
  kernels.

## Why This Fork Exists

The existing projective STAR UVT path is useful because it preserves the current
renderer. The limitation is that the renderer still sees a first-order screen
tube. STAR-PRT tests the next representation step: keep compact tubes, but let
their projected center be nonlinear under camera motion.

The immediate benchmark is CPU-only:

```bash
uv run python variants/star_prt_v0/research_project/benchmarks/star_prt_curve_comparison.py
```

It compares direct per-frame projection, static STAR-style affine projection,
first-order projective affine projection, segmented affine projection, and the
new homogeneous projective-rational curve.

## Static STAR Promotion Blocker

Static STAR promotion still depends on deterministic compact backward. The
named policy added in `star_uvt_v0` is:

```text
deterministic_compact:
  sample_emission_mode = tile_pair
  reduction_mode = key_sort_scan_metal
```

That is the current zero-pruned tile-pair quality/promotion gate. The suffix
and segmented reducer path remains available as `deterministic_suffix_segmented`
for comparison because it is repeatable and compact, but it is not the current
promotion contract.

The gate script checks that `deterministic_compact` matches the deterministic
compact promotion contract:

```bash
uv run python variants/star_uvt_v0/research_project/benchmarks/deterministic_compact_promotion_gate.py \
  --policy deterministic_compact \
  --require-deterministic \
  --require-compact \
  --require-promotion-contract
```

This is a policy and repeatability gate, not a claim that STAR-PRT Metal
forward/backward is done.

## Implemented vs Placeholder

Implemented in `star_prt_v0`:

- The bridge package imports as `torch_gsplat_bridge_star_prt`.
- The dense/reference module imports as
  `research_project.trainer_harness.curve_tube`.
- Dense PyTorch rendering works through `render_projective_rational_tubes` and
  `render_compiled_curve_tubes` with `backend="dense"` or `"auto"`.
- The C++/Metal scaffold reserves the intended op namespace and kernel names.

Still placeholder:

- Metal rendering and compact backward are scaffolds until real binning,
  sorting, compositing, and gradient kernels land.
- STAR-PRT compact backward is not implemented or promoted. The existing
  STAR UVT static promotion path is pinned to the deterministic compact policy
  above.
- No STAR-PRT training-quality or Metal-performance claim is made from the
  CPU-only curve smoke.

## Tensor Shape Contract

World-tube inputs:

```text
x0 [N,3]
velocity [N,3]
t0 [N]
precision_xy [N,2]
lambda_t [N]
opacity [N]
color [N,3]
```

Compiled projective-rational state:

```text
h_coeff [N,H,3]
lambda_uv [N,3]
lambda_t [N]
center_t [N]
depth_coeff [N,H]
opacity [N]
color [N,3]
```

For a static camera polynomial of degree 0, `H = 2`: one constant homogeneous
term and one tube-velocity term. Dense harness center/depth samples use
`centers [N,F,2]` and `depth [N,F]`. The bridge package's compiled curve tensor
is `curve_uv_depth [F,N,3]`, and renders return `image [F,H_px,W_px,3]`.

## Static Compact Backward Promotion Gates

1. CPU contract gate: import both the bridge package and dense module, validate
   the tensor shapes above, prove static projection residual is below `1e-5`,
   and compare dense harness rendering against bridge dense rendering.
2. Gradient reference gate: compare CPU autograd with finite differences for
   `h_coeff`, `lambda_uv`, `lambda_t`, `center_t`, `opacity`, and `color`.
3. Metal parity gate: compare compact backward against the existing static
   backward on the same sorted tile workload for loss, image, and every
   trainable gradient.
4. Determinism gate: decide bitwise identity versus numeric repeatability before
   promotion, then record the actual max absolute and relative gradient deltas.
5. Performance gate: time backward-only, allocation/binning/replay, and total
   train step separately after warmup; require a consistent win on the same
   workload.
6. Quality gate: fixed-step and fixed-wall training must not regress train or
   heldout PSNR, and support-pruning or static train/eval splits must clear
   quality before becoming default.
7. Rollout gate: keep compact backward behind an explicit flag or tile policy
   until the parity, determinism, performance, and quality gates are all green.
