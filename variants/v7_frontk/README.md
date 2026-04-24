# torch-metal-gsplat-v71-frontk

Experimental **Torch custom-op + Metal graphics/compute** handoff for a **real v7.1** design:

- exact hardware-rendered forward image,
- saved **front-K visibility state** per pixel,
- exact compute backward over the saved front layers,
- exact compute replay fallback **only on overflow pixels**.

This is the concrete follow-up to the failed v7 handoff. The goal is not “slightly tune replay backward.” The goal is to **replace** the backward design so training work scales with `pixels * K` on the common path instead of `pixels * gaussians` everywhere.

## What changed versus the old v7 handoff

The old package returned only the final image and forced backward to replay all sorted gaussians for every pixel. That made backward both slow and gradient-fragile.

This v7.1 handoff changes the ABI and the algorithm:

1. **Forward render pass** still draws instanced Gaussian impostors through a Metal render pipeline.
2. A **capture compute pass** records the first `K` visible Gaussian ids and raw alphas for each pixel and flags overflow pixels.
3. Backward runs two kernels:
   - **front-K exact backward** for pixels with `<= K` contributors,
   - **overflow replay backward** only for flagged pixels.

## Saved forward state

Per pixel we store:

- `front_ids[k]` — sorted gaussian index for the `k`-th visible contributor
- `front_raw_alpha[k]` — unclamped `opacity * exp(power)` for the same contributor
- `front_count` — number of captured contributors, clamped to `K`
- `overflow_mask` — `1` when the pixel had more than `K` visible contributors

That is enough to run an exact reverse recursion for all non-overflow pixels.

## Backward math

For a non-overflow pixel with visible layers `i = 0..n-1`:

- `alpha_i = min(0.99, raw_i)`
- `T_{i-1} = Π_{j<i} (1 - alpha_j)`
- `S_n = bg`
- `S_i = alpha_i * c_i + (1 - alpha_i) * S_{i+1}`

Then:

- `∂L/∂c_i = grad_out * (T_{i-1} * alpha_i)`
- `∂L/∂alpha_i = dot(grad_out, T_{i-1} * (c_i - S_{i+1}))`

Chain rule to Gaussian parameters uses:

- `raw_i = opacity_i * exp(power_i)`
- `∂raw/∂opacity = exp(power)`
- `∂raw/∂power = raw`
- `∂power/∂mean_x = a * dx + b * dy`
- `∂power/∂mean_y = b * dx + c * dy`
- `∂power/∂a = -0.5 * dx^2`
- `∂power/∂b = -dx * dy`
- `∂power/∂c = -0.5 * dy^2`

## Important status note

This is still a **source handoff**, not a performance-validated Mac release. I cannot validate the Objective-C++ / Metal path in this environment. What *is* validated here is the math: the Python reference implementation included in this package matches dense autograd reference gradients to numerical precision on small CPU cases.

## Files

- `torch_gsplat_bridge_v71/rasterize.py` — Python API and autograd wrapper
- `torch_gsplat_bridge_v71/reference.py` — pure-Python front-K reference path used for validation
- `csrc/bindings.cpp` — Torch op registration
- `csrc/metal/gsplat_v71.mm` — Objective-C++ bridge
- `csrc/metal/gsplat_v71.metal` — Metal render and compute shaders
- `csrc/shared/common.h` — shared ABI definitions
- `docs/v71_kernel_design.xml` — structured design notebook with four code/correct loops
- `tests/reference_exactness.py` — CPU exactness checks for images and gradients

## Suggested next steps on a Mac

1. Build the extension.
2. Run `tests/reference_exactness.py` once on CPU/reference to confirm the math baseline.
3. Run a tiny MPS parity check against the reference path.
4. Measure overflow rate for `K=2` and `K=4` on your real scenes.
5. Benchmark front-K backward against v3 and v5 before deciding whether to deepen the hardware branch.
