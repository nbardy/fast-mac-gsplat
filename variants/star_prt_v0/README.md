# star_prt_v0

Lightweight scaffold for the STAR Projective Rational Tube (PRT) lane.

This variant is intentionally separate from `star_uvt_v0`:

- Python package: `torch_gsplat_bridge_star_prt`
- Torch op namespace: `star_prt_v0`
- Distribution name: `torch-gsplat-star-prt-v0`

The working path today is a slow dense PyTorch fallback. The C++ and Metal files
define the intended API and tensor contract, but their compiled render entrypoints
raise until real Metal binning, sorting, compositing, and backward kernels are
implemented.

## Tensor Contract

`render_projective_rational_tubes` accepts:

```text
h_coeff: [N,H,3] float32
  Homogeneous image-curve coefficients for h(tau) = (x*w, y*w, w).
  tau = frame_time - center_t, with frame_time = frame - 0.5 * (frames - 1).

lambda_uv: [N,3] float32
  Symmetric screen precision terms (uu, uv, vv).

lambda_t: [N] float32
center_t: [N] float32
opacity: [N] float32
color: [N,3] float32
```

`render_compiled_curve_tubes` accepts the same tube attributes, but receives the
already compiled curve tensor directly:

```text
curve_uv_depth: [F,N,3] float32
  Per-frame (u, v, depth) samples.
```

Both renderers return:

```text
image: [F,H,W,3] float32
```

## Current API

```python
from torch_gsplat_bridge_star_prt import PRTRenderConfig, render_projective_rational_tubes

config = PRTRenderConfig(height=32, width=32, frames=4)
image = render_projective_rational_tubes(
    h_coeff,
    lambda_uv,
    lambda_t,
    center_t,
    opacity,
    color,
    config,
)
```

`backend="auto"` and `backend="dense"` both use the dense fallback. `backend="metal"`
routes through `torch.ops.star_prt_v0` and currently raises a scaffold error.
`metal_compact_backward_projective_rational_tubes` likewise registers the compact
backward API contract but intentionally raises until the backward kernel exists.

## Local Smokes

```bash
uv run python variants/star_prt_v0/research_project/trainer_harness/curve_tube_smoke.py
uv run python variants/star_prt_v0/research_project/benchmarks/star_prt_curve_comparison.py
uv run python variants/star_prt_v0/tests/test_curve_tube_smoke.py
```

The representation and STAR UVT comparison are summarized in
`docs/curve_tube_design.md`.

## Placeholder Kernel Names

The Metal skeleton reserves these names:

```text
star_prt_bin_projective_rational_tubes
star_prt_render_projective_rational_tubes
star_prt_render_compiled_curve_tubes
star_prt_compact_backward_projective_rational_tubes
```

No Metal parity is claimed in this variant yet.
