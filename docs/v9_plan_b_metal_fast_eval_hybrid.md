# V9 Plan B: Metal Fast Eval / Hybrid Path

Date: 2026-04-25

Scope: Direction B only. This plan uses the working V9 hardware-raster forward
pieces as a fast eval, preview, and possible hybrid path. It does not claim V8
training parity until the explicit state contract is solved.

## 1. Precise Goal

Build the best practical Metal fast-eval path from the pieces that already work:

- projected screen-space Gaussian inputs already resident on MPS;
- instanced hardware quads;
- fragment Gaussian evaluation;
- fixed-function premultiplied source-over blending;
- direct buffer-backed Torch/MPS output;
- `RGBA16F` and `RGBA32F` output options;
- optional stable depth ordering wrapper.

The target product is a fast renderer that is useful for preview, validation,
dataset rendering, interactive eval, and possibly approximate training
experiments. Promotion to a forward+backward training kernel requires a separate
state contract: exact `C`, `T`, `final_T`, and stopped prefix behavior matching
V8.

## 2. Non-Goals

This direction should not pretend to be the V8 replacement by default.

Non-goals:

- no ICB execute path in shared code or benchmarks;
- no claim of exact multi-splat V8 parity from fixed-function blending alone;
- no exact backward unless per-pixel state and replay semantics match V8;
- no CPU staging or CPU readback in the hot path;
- no "RGBA then slice" bandwidth story;
- no broad refactor of V8 projection, binning, or backward code unless it is
  used as an explicit hybrid fallback.

## 3. What This Path Is For

Good use cases:

- interactive preview where image quality tolerance is more important than exact
  training gradients;
- eval-only rendering on MPS without CPU staging;
- camera sweep previews;
- visual debugging of projected splats;
- fast approximate image losses if the training loop accepts gradient mismatch;
- hybrid forward display path while V8 remains the gradient source.

Possible but gated use cases:

- approximate training with straight-through or V8-recompute backward;
- forward+V8-backward hybrid only if the loss owner accepts that backward is not
  the derivative of the hardware forward image;
- exact training only after a side-state path produces the same V8 state that
  backward needs.

Not suitable yet:

- exact multi-splat image parity;
- exact differentiable training;
- scenes where small ordering differences or alpha precision changes dominate
  the loss;
- any path that depends on ICB execution stability.

## 4. Current Evidence

Working pieces from the V9 exploration:

- Direct Metal render output into Torch/MPS tensor storage works.
- Gaussian fragment eval into `RGBA32F` works.
- Gaussian fragment eval into `RGBA16F` works.
- Constant direct targets validate for `RGBA32F`, `RGBA16F`, `R32F`, and
  `RG32F`.
- Stable depth sorting wrapper works and is deterministic.
- Single-splat black-background parity against V8 is essentially exact.
- Multi-splat parity against V8 fails.

Representative measured results:

| Case | Median |
|---|---:|
| 512x512 fixed eval 6K RGBA32F | 1.659 ms |
| 1080x1920 fixed eval 6K RGBA32F | 1.846 ms |
| 4096x4096 fixed eval 6K RGBA32F | 4.976 ms |
| 4096x4096 fixed eval 64K RGBA32F | 6.123 ms |
| 512x512 fixed eval 6K RGBA16F | 1.679 ms |
| 1080x1920 fixed eval 6K RGBA16F | 1.739 ms |
| 4096x4096 fixed eval 6K RGBA16F | 1.863 ms |
| 4096x4096 fixed eval 64K RGBA16F | 4.958 ms |

Representative parity results:

| Case | Resolution | Splats | Max RGB Error |
|---|---:|---:|---:|
| `tiny_single` | 16x16 | 1 | about `1.49e-08` |
| `grid_ordered` | 16x16 | 16 | about `9.68e-02` |
| `overlap_ordered` | 16x16 | 16 | about `1.24e-01` |
| `overlap_ordered` | 64x64 | 16 | about `5.80e-02` |

Interpretation: this is a strong fast eval path, not a solved training path.

## 5. Math Contract

### 5.1 V8 Exact Forward Contract

For a pixel `x`, sorted splats `i = 0..N-1`, color `c_i`, opacity `o_i`, and
screen-space conic matrix:

```text
Q_i = [[a_i, b_i],
       [b_i, c_i]]
d_i = x - mean_i
power_i = -0.5 * d_i^T Q_i d_i
alpha_i = min(0.99, o_i * exp(power_i))
alpha_i = 0 if power_i > 0 or alpha_i < alpha_threshold
```

V8 uses explicit front-to-back transmittance:

```text
C_0 = 0
T_0 = 1

for i in front_to_back_order:
    if T_i <= transmittance_threshold:
        stop
    C_{i+1} = C_i + T_i * alpha_i * c_i
    T_{i+1} = T_i * (1 - alpha_i)

out_rgb = C_N + T_N * background
final_T = T_N
stop_count = number of processed splats for that pixel
```

Backward depends on the same recurrence. The minimum exact state is either:

- saved `final_T` plus enough prefix information to replay the same stopped
  splats, or
- a deterministic recompute path that regenerates the exact same prefix.

### 5.2 Current Metal Fixed-Function Contract

The current V9 fragment shader emits premultiplied source color:

```text
S_i.rgb = alpha_i * c_i
S_i.a   = alpha_i
```

The render pipeline uses:

```text
sourceRGBFactor      = one
destinationRGBFactor = oneMinusSourceAlpha
sourceAlphaFactor    = one
destinationAlphaFactor = oneMinusSourceAlpha
```

For a destination `D`, one blended fragment updates:

```text
D' = S_i + (1 - alpha_i) * D
```

If fragments are submitted in order `0..N-1` over a black target, the final RGB
is:

```text
D_N.rgb = sum_i alpha_i * c_i * product_{j > i}(1 - alpha_j)
```

That is the standard "new source over old destination" recurrence. It matches
V8's front-to-back color only under narrow conditions:

```text
V8 order:             near -> far
hardware blend order: far  -> near
background:           black, or composed with equivalent alpha after the pass
alpha set:            identical per pixel
early stop:           disabled or mathematically irrelevant
precision:            close enough for chosen output format
per-pixel order:      validated on the target GPU/API path
```

It also trivially matches one-splat black-background cases.

### 5.3 Where They Diverge

The current fast path diverges from V8 when:

- it submits the same ascending V8 order into fixed source-over blending instead
  of the reverse painter order needed for color equivalence;
- hardware primitive/fragment order differs from the assumed order;
- equal-depth stable ordering differs;
- V8 stops early at `T <= 1e-4` but hardware blending keeps blending farther
  fragments;
- the training path needs `final_T`, `stop_count`, or a stopped prefix;
- background composition is not black and alpha semantics are not applied
  exactly;
- output precision is `RGBA16F` and the loss requires `RGBA32F`-level accuracy;
- Gaussian support bounds or alpha discard thresholds differ.

The main accuracy bug is therefore semantic, not just numeric. We are missing
the explicit ordered `C/T/stop` contract.

## 6. Output Plane Choices

Metal buffer-backed textures require 256-byte aligned rows. Current direct
width multiples:

| Format | Bytes / Pixel | Torch Shape | Direct Width Multiple | Gaussian Eval |
|---|---:|---|---:|---|
| `RGBA32F` | 16 | `[H,W,4] float32` | 16 | yes |
| `RGBA16F` | 8 | `[H,W,4] float16` | 32 | yes |
| `R32F` | 4 | `[H,W] float32` or `[H,W,1]` | 64 | constant only today |
| `RG32F` | 8 | `[H,W,2] float32` | 32 | constant only today |

Full-frame store sizes:

| Resolution | RGBA32F | RGBA16F / RG32F | R32F |
|---:|---:|---:|---:|
| 512x512 | 4.19 MB | 2.10 MB | 1.05 MB |
| 1080x1920 | 33.18 MB | 16.59 MB | 8.29 MB |
| 4096x4096 | 268.44 MB | 134.22 MB | 67.11 MB |

Format guidance:

- `RGBA16F` is the best current fast-eval candidate. It halves output storage
  and validated with one-splat max error `0.00048828125`.
- `RGBA32F` remains the reference eval format and the safer loss input.
- `R32F` should be used for side-state such as `final_T` or alpha-only output,
  not for RGB color.
- `RG32F` could hold two side-state values, for example `final_T` and
  normalized stop count, but integer stop count should ideally be a real `uint`
  compute buffer rather than a float render target.
- Returning `RGBA32F` and slicing does not save bandwidth. The render pass still
  stores all four channels.

## 7. Current Kernel / Render-Pass Shape

Current fixed eval render path:

```text
Python:
    validate MPS tensors:
        means2d [G,2] float32
        conics  [G,3] float32
        colors  [G,3] float32
        opacities [G] float32
    make contiguous views
    call native render_gaussian_eval_format(format, ..., direct=True)

Native:
    create output tensor with dtype/channels for format
    get MTLBuffer for inputs and output tensor storage
    create buffer-backed MTLTexture if row alignment allows direct path
    otherwise optionally use private texture plus GPU blit
    encode render pass:
        clear color target to (0,0,0,0)
        bind gaussian eval PSO
        bind input buffers at vertex indices 0..3
        bind GaussianEvalParams at vertex index 4 and fragment index 0
        draw triangle strip vertexCount=4 instanceCount=G

Vertex shader per instance:
    read mean, conic, color, opacity
    compute support threshold tau from alpha_threshold / opacity
    derive axis-aligned conservative bounds hx, hy from conic inverse terms
    clamp quad bounds to image
    emit one of four quad corners in NDC
    pass mean, conic, color, opacity to fragment

Fragment shader:
    d = pixel_position - mean
    power = -0.5 * (a*d.x*d.x + 2*b*d.x*d.y + c*d.y*d.y)
    discard if power > 0
    alpha = min(0.99, opacity * exp(power))
    discard if alpha < 1/255
    return premultiplied float4(color * alpha, alpha)

Fixed blend:
    dst = src + (1 - src.a) * dst
```

This is compact and fast because it pushes coverage to raster hardware and only
executes the fragment shader on covered quad pixels.

## 8. Sorted Eval Wrapper Role

The sorted wrapper is useful but limited.

What it does:

- computes a stable `torch.argsort(depths.detach(), stable=True)` on MPS;
- gathers `means2d`, `conics`, `colors`, and `opacities`;
- submits a deterministic order to the same render path;
- supports descending order for reverse-order probes.

What it does not do:

- it does not change the fixed source-over recurrence;
- it does not produce `final_T`;
- it does not produce `stop_count`;
- it does not expose which splats contributed before early stop;
- it does not create a backward replay contract;
- it does not prove hardware per-pixel blending order is a documented training
  invariant.

Required next probe for this direction:

```text
Compare three orders against V8:
    1. current input order
    2. V8 ascending depth order
    3. reverse V8 depth order

Expected:
    reverse V8 depth order should be the only candidate that can match fixed
    source-over color for multi-splat black-background cases, assuming identical
    alpha and no early-stop effects.
```

If reverse-order color still fails substantially, the reason is likely coverage,
alpha threshold, per-pixel order, or a mismatch between V8's pixel math and the
fragment shader.

## 9. Next Feasible Improvements

### B1. Reverse-Order Painter Eval

Add an explicit wrapper mode:

```text
front_to_back = stable_argsort(depths)
draw_order = reverse(front_to_back)
render fixed source-over in draw_order
```

This is the cheapest possible attempt to close image parity for color only.
It still does not solve backward state.

Promotion gate:

- multi-splat black-background max RGB error below tolerance at 64x64, 512x512,
  1080p, and 4K;
- compare `RGBA32F` first;
- only then test `RGBA16F`.

### B2. RGBA16F Default Eval Output

Make `RGBA16F` the default for preview/eval when the caller opts in.

Promotion gate:

- image-space tolerance against `RGBA32F` on representative scenes;
- no unexpected banding or alpha artifacts;
- loss consumers explicitly accept `float16` or own conversion to `float32`.

### B3. Alpha / Final-T Side Plane

The fixed alpha blend accumulates:

```text
A_out = 1 - product_i(1 - alpha_i)
```

If the blended fragment set is identical to V8 and early stop is ignored, then:

```text
final_T_approx = 1 - A_out
```

This can be useful for preview diagnostics, but it is not enough for exact
backward. It lacks the stopped prefix and per-splat prefix transmittance.

Possible implementation:

- use output alpha from `RGBA32F` or `RGBA16F` for approximate `final_T`;
- or emit an `R32F` side target if the color target can be split;
- validate against V8 `final_T` on parity cases.

Risks:

- `RGBA16F` alpha quantization can disturb very small `T`;
- hardware alpha includes all blended fragments, not V8 early-stop prefix;
- if draw order or fragment set differs, `1 - alpha` is not V8 `T`.

### B4. Stop-Count Assist Pass

To support exact backward, we need `stop_count` or equivalent prefix metadata.
Fixed-function blending cannot compute it directly.

Possible compute assist:

```text
kernel stop_count_assist(pixel):
    T = 1
    count = 0
    for splat in V8 front_to_back tile segment:
        alpha = eval_alpha(pixel, splat)
        if alpha >= threshold:
            T *= (1 - alpha)
            count += 1
            if T <= transmittance_threshold:
                break
    final_T[pixel] = T
    stop_count[pixel] = count
```

This starts to look like V8 forward without color accumulation. It may still be
useful if:

- color comes from hardware raster quickly;
- side-state compute is cheaper than full V8 color+state;
- exact backward can replay the same prefix.

But it can also erase the performance win. The benchmark must include this pass
before making any training claim.

### B5. Tile / Imageblock State Variant

Tile/imageblock should not be used as decoration. It is worth revisiting only if
it explicitly implements V8 state:

```text
per pixel in imageblock:
    C.rgb float or half
    T float
    stop_count uint
```

This is closer to Direction A than Direction B. For Direction B, imageblocks are
only a fallback if fixed eval plus side-state cannot produce useful contracts.

Known pressure from probes:

- `half4_baseline`: 24 B/sample, 6 KB per 16x16 tile;
- `ct_fp32`: 32 B/sample, 8 KB per 16x16 tile;
- `ct_stop_flags_fp32_u32x2`: 48 B/sample, 12 KB per 16x16 tile.

At 4K, full-frame `C/T/stop/flags` equivalent is roughly 398 MB. This must beat
V8 on wall time, not just look elegant.

## 10. Backward Feasibility

### Exact Backward

Exact backward is not feasible from fixed eval alone.

Requirements:

- same sorted per-pixel splat sequence as V8;
- same alpha threshold and clamp behavior;
- same early stop;
- `final_T` and stopped prefix, or deterministic replay that reaches the same
  stop point;
- backward reductions to splat parameters with atomics or tile reductions.

The hardware color target by itself is insufficient.

### Approximate Backward: V8 Recompute

Hybrid option:

```text
forward:
    image = hardware_fast_eval(inputs)

backward:
    recompute V8 forward state from same inputs
    run V8 backward
    return V8 gradients
```

This gives useful gradients if the hardware forward image is close to V8, but
the gradient is the derivative of the V8 recomputed image, not necessarily the
displayed hardware image.

This is acceptable only as an explicitly approximate training mode.

### Approximate Backward: Straight-Through

Straight-through option:

```text
forward:
    image_hw = hardware_fast_eval(inputs)

backward:
    pretend image_hw came from V8 or from simplified alpha recurrence
```

This is risky. It can optimize the wrong objective if image mismatch is large.
Use only for experiments with gradient checks and ablation results.

### Approximate Backward: Fixed Source-Over Replay

One could derive gradients for the fixed source-over recurrence:

```text
D_{k+1} = S_k + (1 - alpha_k) * D_k
```

But exact replay still needs the per-pixel ordered fragment list. Without a
tile/bin list and deterministic replay, it is not recoverable from the final
image alone.

### Finite Difference

Finite difference is not a serious training path. It is too slow by orders of
magnitude and should only be used for tiny correctness tests.

## 11. Backward Math If State Is Solved

For exact V8 recurrence:

```text
C_{i+1} = C_i + T_i * alpha_i * c_i
T_{i+1} = T_i * (1 - alpha_i)
out = C_N + T_N * background
```

Reverse pass for one pixel:

```text
g_C_N += g_out
g_T_N += dot(g_out, background)

for i in reversed(processed_prefix):
    # C_{i+1}
    g_color_i += T_i * alpha_i * g_C_{i+1}
    g_alpha_i += T_i * dot(g_C_{i+1}, color_i)
    g_T_i     += alpha_i * dot(g_C_{i+1}, color_i)
    g_C_i     += g_C_{i+1}

    # T_{i+1}
    g_alpha_i += -T_i * g_T_{i+1}
    g_T_i     += (1 - alpha_i) * g_T_{i+1}
```

Alpha parameter derivatives before clamp/discard:

```text
alpha = opacity * exp(power)
power = -0.5 * d^T Q d
d = x - mean

d opacity += g_alpha * exp(power)
d power   += g_alpha * opacity * exp(power)
d mean    += d power * Q * d
d Q       += d power * (-0.5 * outer(d, d))
```

Clamp, threshold, and discard branches must match V8 exactly. Contributions
discarded by threshold should not receive gradient unless an approximate
surrogate is explicitly selected.

## 12. Memory / Performance Expectations

Baseline output storage at 4K:

- `RGBA32F`: 268 MB per frame;
- `RGBA16F`: 134 MB per frame;
- `R32F` side plane: 67 MB per frame;
- `RGBA16F + R32F final_T`: about 201 MB;
- `RGBA16F + R32F final_T + U32 stop_count`: about 268 MB before any tile lists.

Performance expectations:

- Current `RGBA16F` fast eval is promising: 4K/64K median about `4.958 ms`.
- Current `RGBA32F` 4K/64K median is about `6.123 ms`.
- A side-state compute pass can easily consume the entire win if it loops all
  relevant splats per pixel.
- The only worthwhile hybrid is one where hardware color remains fast and state
  assist is much cheaper than full V8 forward, or where approximate training is
  acceptable.

Rule of thumb:

```text
preview path:
    optimize for image and latency

training path:
    optimize for exact state contract first
    only then optimize latency
```

## 13. Validation Plan

### B0. Preserve Safe Baseline

- Build and run existing output-plane smoke tests.
- Keep ICB execute disabled.
- Keep CPU readback only in validation.

Pass condition:

- `RGBA32F` Gaussian max error remains `0.0` on one-splat validation;
- `RGBA16F` Gaussian max error remains around half precision quantization.

### B1. Reverse-Order Color Parity

Add parity rows:

- input order;
- V8 ascending depth order;
- reverse V8 depth order.

Use:

- black background first;
- `RGBA32F` first;
- small overlapping cases first;
- then 512x512/6K, 1080p/6K, 4K/64K.

Pass condition:

- reverse-order `RGBA32F` color error is within agreed tolerance on multi-splat
  cases.

Fail condition:

- reverse order still has large errors. Then fixed-function eval remains
  approximate preview only.

### B2. RGBA16F Tolerance Gate

Compare `RGBA16F` against `RGBA32F` and V8 reference on representative scenes.

Pass condition:

- max/mean error is acceptable for preview/eval;
- no loss instability if used for approximate training.

### B3. Final-T Side-State Gate

Validate:

```text
T_from_alpha = 1 - output_alpha
```

against V8 `final_T` on cases without early-stop sensitivity.

Pass condition:

- `T_from_alpha` matches when color matches and early stop is irrelevant.

Fail condition:

- alpha is not a usable proxy. Then side-state must be compute-owned.

### B4. Stop-Count Assist Gate

Implement a compute assist only if B1 succeeds.

Pass condition:

- `final_T` and `stop_count` match V8;
- added pass still leaves total wall time competitive.

Fail condition:

- assist is as expensive as V8 forward. Then do not use Direction B for exact
  training.

### B5. Hybrid Backward Gate

Compare gradients:

- V8 baseline forward+backward;
- hardware forward + V8 recompute backward;
- hardware forward + side-state replay backward, if implemented.

Pass condition:

- image error and gradient error are both acceptable for the intended mode;
- benchmark includes full forward+backward wall time and memory.

## 14. Exact Pseudocode

### 14.1 Fast Eval Forward

```python
def metal_fast_eval(
    means2d, conics, colors, opacities, depths,
    height, width,
    output_format="rgba16f",
    order_mode="input",        # input, v8_front_to_back, reverse_v8
    direct=True,
):
    assert means2d.device.type == "mps"
    assert conics.device.type == "mps"
    assert colors.device.type == "mps"
    assert opacities.device.type == "mps"

    if order_mode == "input":
        m, q, c, o = means2d, conics, colors, opacities
    else:
        perm = torch.argsort(depths.detach(), stable=True)
        if order_mode == "reverse_v8":
            perm = torch.flip(perm, dims=[0])
        m = means2d.index_select(0, perm)
        q = conics.index_select(0, perm)
        c = colors.index_select(0, perm)
        o = opacities.index_select(0, perm)

    out = render_gaussian_eval_format(
        output_format,
        m.contiguous(),
        q.contiguous(),
        c.contiguous(),
        o.contiguous(),
        height,
        width,
        direct=direct,
    )
    return out
```

### 14.2 Render Shader Pseudocode

```metal
vertex quad_vs(instance_id g, vertex_id v):
    mean = means2d[g]
    Q = conics[g]
    opacity = opacities[g]

    if opacity <= alpha_threshold:
        emit_offscreen()

    tau = -2 * log(alpha_threshold / max(opacity, eps))
    det = max(Q.a * Q.c - Q.b * Q.b, eps)
    hx = sqrt(max(tau * Q.c / det, 0))
    hy = sqrt(max(tau * Q.a / det, 0))

    bounds = clamp([mean.x - hx, mean.y - hy,
                    mean.x + hx, mean.y + hy], image_bounds)
    corner_px = select_quad_corner(bounds, v)
    position = pixel_to_ndc(corner_px)
    pass mean, Q, color, opacity

fragment gaussian_fs(stage_in in):
    d = pixel_position - in.mean
    power = -0.5 * (Q.a*d.x*d.x + 2*Q.b*d.x*d.y + Q.c*d.y*d.y)
    if power > 0:
        discard
    alpha = min(0.99, opacity * exp(power))
    if alpha < alpha_threshold:
        discard
    return float4(color * alpha, alpha)

blend:
    dst = src + (1 - src.a) * dst
```

### 14.3 Optional Final-T From Alpha

```python
def approximate_final_T_from_hardware_alpha(out_rgba):
    # Valid only if output alpha represents the same blended fragment set.
    alpha_accum = out_rgba[..., 3].float()
    return 1.0 - alpha_accum
```

### 14.4 Compute Assist For Exact State

```python
def compute_v8_state_assist(tile_segments, sorted_splats, height, width):
    # One compute thread or SIMD lane group owns one pixel, or one block owns a
    # tile. This must use the same tile segments and sorted IDs as V8.
    for pixel in pixels:
        T = 1.0
        stop_count = 0
        for splat_id in tile_segments[pixel.tile]:
            alpha = eval_alpha(pixel, sorted_splats[splat_id])
            if alpha >= alpha_threshold:
                T = T * (1.0 - alpha)
                stop_count += 1
                if T <= transmittance_threshold:
                    break
        final_T[pixel] = T
        stop[pixel] = stop_count
```

### 14.5 Hybrid Backward With V8 Recompute

```python
class HardwareForwardV8Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, depths, camera_meta):
        image_hw = metal_fast_eval(
            means2d, conics, colors, opacities, depths,
            camera_meta.height,
            camera_meta.width,
            output_format=camera_meta.eval_format,
            order_mode=camera_meta.order_mode,
        )
        ctx.save_for_backward(means2d, conics, colors, opacities, depths)
        ctx.camera_meta = camera_meta
        return image_hw

    @staticmethod
    def backward(ctx, grad_image):
        means2d, conics, colors, opacities, depths = ctx.saved_tensors

        # Explicitly approximate: gradients come from V8 recompute, not from
        # differentiating the hardware render pass.
        image_v8, state_v8 = v8_forward_recompute(
            means2d, conics, colors, opacities, depths, ctx.camera_meta
        )
        grads = v8_backward(grad_image.float(), state_v8)
        return grads.means2d, grads.conics, grads.colors, grads.opacities, None, None
```

### 14.6 Exact Backward If State Assist Is Solved

```python
def exact_backward_with_state(grad_image, sorted_inputs, final_T, stop_count, tile_segments):
    zero global gradients

    for tile in tiles:
        for pixel in tile.pixels:
            prefix = tile_segments[tile][:stop_count[pixel]]

            # Recompute forward prefix values or load compact saved prefix.
            C_prefix, T_prefix, alpha_prefix = replay_forward_prefix(
                pixel, sorted_inputs, prefix
            )

            g_C = grad_image[pixel]
            g_T = dot(grad_image[pixel], background)

            for k in reversed(range(len(prefix))):
                splat = sorted_inputs[prefix[k]]
                T_i = T_prefix[k]
                alpha_i = alpha_prefix[k]
                color_i = splat.color

                g_color = T_i * alpha_i * g_C
                g_alpha = T_i * dot(g_C, color_i) - T_i * g_T
                g_T = alpha_i * dot(g_C, color_i) + (1 - alpha_i) * g_T

                g_params = alpha_to_param_grads(g_alpha, pixel, splat)

                reduce_or_atomic_add(splat.grad_color, g_color)
                reduce_or_atomic_add(splat.grad_params, g_params)
```

## 15. Decision Gates

Direction B should be promoted only as far as the evidence supports:

| Gate | Requirement | Outcome If Failed |
|---|---|---|
| B1 color | reverse-order `RGBA32F` matches V8 color tolerance | eval-only approximate preview |
| B2 precision | `RGBA16F` accepted by image/loss tolerance | keep `RGBA32F` for reference |
| B3 final_T | alpha-derived or side-pass `final_T` matches V8 | no training-state claim |
| B4 stop | `stop_count`/prefix matches V8 | no exact backward |
| B5 full perf | forward+state+backward beats or complements V8 | keep V8 as training base |

The strongest near-term product is a fast eval renderer. The weakest part is
training correctness: without `final_T` and stopped prefix, backward is either
approximate or delegated to V8 recompute.

## 16. Recommended Next Implementation Order

1. Add reverse-depth sorted eval as a wrapper around the output-planes variant.
2. Extend the parity harness to include input, ascending, and reverse orders.
3. Benchmark `RGBA32F` and `RGBA16F` only after color parity is understood.
4. Validate `final_T ~= 1 - alpha` on the cases where color matches.
5. Decide whether stop-count assist is worth implementing.
6. Only after state matches V8, design a backward replay kernel.

If step 2 fails badly, stop trying to make Direction B exact. Keep it as the
fast preview/eval path and put exact training work into the programmable
tile/imageblock or compute direction.
