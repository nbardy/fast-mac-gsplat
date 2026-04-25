# V9 Three Direction Plan Synthesis

Date: 2026-04-25

This is the top-level map for the three post-reflection directions. The full
kernel-level plans are:

- `docs/v9_plan_a_metal_tile_imageblock_exact.md`
- `docs/v9_plan_b_metal_fast_eval_hybrid.md`
- `docs/v9_plan_c_cuda_compute_first.md`

The shared premise is now clear: direct output interop is solved enough. The
unsolved problem is exact multi-splat ordered compositing plus backward state.

## Execution Update

The first implementation pass after these plans changed the status:

- Direction B now has sorted/reverse-order wrappers in
  `variants/v9_hw_output_planes_probe`.
- Direction A now has `variants/v9_hw_tile_exact_probe`, a minimal 16x16
  imageblock exact `C/T` semantic probe.
- Direction C now has `variants/v9_cuda_compute_first`, a CUDA source scaffold
  that reports this Mac has no CUDA runtime and fails native build clearly.

These updates improve the outlook for both Metal eval and Metal exact-forward
research, but they do not create an exact training path yet.

## Direction Summary

| Direction | Best Use | Accuracy Target | Backward Target | Main Risk |
|---|---|---|---|---|
| A: Metal tile/imageblock exact | Best-of-both-worlds Metal training attempt | Exact V8 multi-splat `C/T/stop` | V8-compatible replay first, render-assisted later | Fragment/imageblock ordering and stop metadata may erase the forward win. |
| B: Metal fast eval/hybrid | Preview/eval renderer and possible approximate training | Approximate unless reverse-order color parity passes | None by default; V8 recompute only as approximate hybrid | Fixed blending still does not expose `final_T` or stopped prefix. |
| C: CUDA compute-first | Portable exact CUDA training path | Exact gsplat/V8/Graphdeco math | Exact tile replay with reduced atomics | Heavy-overlap pixels still have serial ordered alpha recurrence. |

## Recommended Priority

1. **Keep B as the near-term product path.**
   It already has working direct MPS output and `RGBA16F` Gaussian eval. Promote
   it only as fast eval/preview unless side-state gates pass.

2. **Use A as the risky Metal research path.**
   It is the only Metal path that can plausibly become exact hardware-assisted
   training, because it attacks explicit programmable per-pixel `C/T/stop`
   state. It should be killed quickly if imageblock ordering or stop metadata
   cannot match V8 on tiny overlap tests.

3. **Use C as the serious CUDA plan.**
   CUDA should be compute-first: CUB scan/sort, direct tensor writes, one block
   per tile, exact replay backward, and warp/block reductions before atomics.
   Vulkan hardware raster can be a later branch, not the first CUDA port.

## Shared Math Contract

All exact paths must implement the same scalar recurrence. For pixel center
`p = (x + 0.5, y + 0.5)`:

```text
dx, dy  = p - mean2d_i
q_i     = a_i*dx^2 + 2*b_i*dx*dy + c_i*dy^2
power_i = -0.5 * q_i
raw_i   = opacity_i * exp(power_i)
alpha_i = min(max_alpha, raw_i)
visible = power_i <= 0 and alpha_i >= alpha_threshold

C_0 = 0
T_0 = 1

for i in stable sorted front-to-back candidates:
  if T_i <= transmittance_threshold:
    stop
  if visible:
    C_{i+1} = C_i + T_i * alpha_i * color_i
    T_{i+1} = T_i * (1 - alpha_i)
  else:
    C_{i+1} = C_i
    T_{i+1} = T_i

out = C_M + T_M * background
final_T = T_M
```

Backward must replay the same prefix in reverse:

```text
T_cur = T_final
gT = dot(grad_out, background)

for i in reverse(processed_prefix):
  denom  = max(1 - alpha_i, eps)
  T_prev = T_cur / denom
  dot_c  = dot(grad_out, color_i)

  g_alpha = T_prev * (dot_c - gT)
  g_color = grad_out * (T_prev * alpha_i)

  clamp_gate = 1 if raw_i < max_alpha else 0
  g_raw   = g_alpha * clamp_gate
  g_power = g_raw * raw_i

  grad_a       += g_power * (-0.5 * dx^2)
  grad_b       += g_power * (-1.0 * dx * dy)
  grad_c       += g_power * (-0.5 * dy^2)
  grad_mean_x  += g_power * (a*dx + b*dy)
  grad_mean_y  += g_power * (b*dx + c*dy)
  grad_opacity += g_raw * raw_i / max(opacity, eps)

  gT = alpha_i * dot_c + (1 - alpha_i) * gT
  T_cur = T_prev
```

Any path that cannot provide this contract should be marked eval-only or
approximate.

## Direction A: Metal Tile/Imageblock Exact

Core hypothesis:

```text
V8 visibility bins
  -> Metal render/tile/imageblock exact forward
  -> minimal saved state
  -> V8-compatible compute backward replay
```

Required primitives:

- direct buffer-backed Torch/MPS render target;
- tile shader init and flush;
- explicit-layout imageblock `PixelState`;
- fragment shader updates imageblock state with blending disabled;
- optional ROG only after tiny exact tests;
- V8 compute bin/sort path as the first sorted-ingestion producer;
- V8 compute backward as the first exact backward.

First state layout:

```text
PixelState:
  float4 c_t       // C.rgb, T
  uint observed_i  // debug or candidate ordinal
  uint flags       // stopped, overflow/debug
```

Measured imageblock cost on M4:

| Layout | 16x16 Tile | 32x32 Tile |
|---|---:|---:|
| `ct_fp32` | 8,192 B | 32,768 B |
| `ct_stop_flags_fp32_u32x2` | 12,288 B | 49,152 B |

Critical gate:

```text
tiny overlapping multi-splat forward image == V8 within tolerance
no CPU wait/readback
no fixed-function blending
no ICB execute
```

Why this could work:

- It attacks the real blocker: explicit ordered `C/T` state.
- It reuses V8 visibility and backward instead of rewriting all training logic.

Why it may fail:

- Fragment-driven raster does not naturally see invisible candidates, but V8
  prefix/stop behavior can depend on candidate prefix semantics.
- ROG/imageblock ordering may serialize hot pixels.
- Compute stop-count assist may cost as much as V8 forward.

Direction A should be implemented as a correctness-first spike, not a big
mainline rewrite.

### A Execution Result

`variants/v9_hw_tile_exact_probe` proves the missing low-level primitive:

```text
tile clear -> ordered fragment [[imageblock_data]] C/T update
  -> tile resolve -> direct Torch/MPS output
```

The smoke uses two full-screen constant-alpha fragments:

```text
splat 0: alpha=0.25, red
splat 1: alpha=0.50, green
expected: float4(0.25, 0.375, 0.0, 0.375)
```

Observed on Apple M4:

```text
tile_exact_overlap_max_abs_err = 0.0
tile_exact_imageblock_sample_length = 48 B
tile_exact_imageblock_memory_16x16 = 12,288 B
tile_exact_imageblock_memory_32x32 = 49,152 B
```

The path is fail-closed to 16x16. The exact 32x32 pipeline compiles but render
encoder creation failed on M4, likely from tile-memory pressure. The next A gate
is 2-4 projected Gaussian quads in stable order, still without V8 tile bins.

## Direction B: Metal Fast Eval / Hybrid

Core hypothesis:

```text
projected MPS tensors
  -> instanced Gaussian quads
  -> fragment Gaussian alpha
  -> fixed source-over blending
  -> direct RGBA16F/RGBA32F Torch/MPS output
```

This is the path that already has useful speed.

Current median results:

| Case | RGBA32F | RGBA16F |
|---|---:|---:|
| 512x512 Gaussian 6K | 1.659 ms | 1.679 ms |
| 1080p Gaussian 6K | 1.846 ms | 1.739 ms |
| 4096x4096 Gaussian 6K | 4.976 ms | 1.863 ms |
| 4096x4096 Gaussian 64K | 6.123 ms | 4.958 ms |

Immediate improvements:

1. Test reverse-order on realistic projected scenes.
2. Make `RGBA16F` the default fast-preview candidate only after image error
   gates pass.
3. Test alpha-derived `final_T = 1 - out_alpha` only on cases where color
   parity passes.
4. Stop trying to make B exact if reverse-order color parity fails outside
   controlled black-background overlap stacks.

What B can honestly support:

- preview;
- interactive eval;
- dataset rendering with accepted approximation;
- approximate training experiments with explicit caveats.

What B cannot claim yet:

- exact V8 multi-splat parity;
- exact backward;
- a true forward+backward speedup.

The key warning is that fixed-function source-over blending computes a different
ordered recurrence unless the submit order, background, alpha gates, early stop,
and hardware fragment order all line up. Even when color matches, it still does
not produce `final_T`, `stop_count`, or a stopped prefix for backward.

### B Execution Result

`variants/v9_hw_output_planes_probe` now exports:

```text
render_gaussian_eval_format_sorted(...)
render_gaussian_eval_rgba_sorted(...)
render_gaussian_eval_rgba16_sorted(...)
```

Controlled black-background overlap stacks show that reverse/depth-descending
submission is the fixed-function color-parity candidate:

| Case | Format | Order | Max RGB Error vs V8 |
|---|---|---|---:|
| 16x32 G=2 | RGBA32F | input / ascending | 0.25 |
| 16x32 G=2 | RGBA32F | descending | 9.31e-10 |
| 16x32 G=16 | RGBA32F | input / ascending | 0.385184 |
| 16x32 G=16 | RGBA32F | descending | 1.19e-07 |
| 64x64 G=16 | RGBA32F | descending | 1.19e-07 |
| 64x64 G=16 | RGBA16F | descending | 0.001294 |

This is a real fast-eval finding: for black-background color, fixed
source-over can match V8 if submitted in reverse painter order. It is still not
an exact training path because `final_T`, `stop_count`, and the stopped prefix
are absent.

## Direction C: CUDA Compute-First

Core hypothesis:

```text
fused 3D projection/count
  -> opacity-aware tile intersection
  -> fixed-capacity CUB scan/sort
  -> one CUDA block per 16x16 tile
  -> exact C/T forward
  -> compact state
  -> exact backward replay with warp/block reductions
```

Required CUDA primitives:

- CUB `DeviceScan` and `DeviceRadixSort` or segmented radix sort;
- `cooperative_groups` or warp shuffles for reductions;
- shared memory batches of IDs, means, conics, colors, opacities;
- direct Torch CUDA tensor pointers;
- CUDA Graphs for fixed-shape launch replay after the baseline is correct;
- optional `cp.async` / TMA only after profiling.

Default tile work shape:

```text
tile_size = 16
threads_per_block = 256
one CUDA block per tile per image
one thread per pixel
shared batch size = 256 splat refs
```

Default backward atomics strategy:

```text
per tile:
  each pixel computes local gradient partials
  reduce by warp
  reduce across warps in shared memory
  one or few lanes issue global atomicAdd per splat/component
```

CUDA is the cleanest exact-training path because it can own all math in compute
and avoid Metal render-pass semantics. The main risk is still mathematical:
ordered alpha compositing is serial per pixel in heavy overlap. CUDA can reduce
projection, sort, launch, memory, and atomic overhead, but it cannot make the
per-pixel recurrence associative without changing the model.

### C Execution Result

`variants/v9_cuda_compute_first` is a source-level scaffold only on this Mac:

```text
nvcc: missing
nvidia-smi: missing
PyTorch CUDA built: false
torch.cuda.is_available(): false
MPS available: true
```

The package exposes `cuda_environment()`, includes CUDA skeletons for
`project_count_fused`, `emit_pairs`, `tile_forward_train`, and
`tile_backward_replay`, and intentionally fails native build with a clear
CUDA-only message on this host. No CUDA benchmark exists yet because no CUDA
runtime is available locally.

## Cross-Direction Kill Gates

Kill any "exact training" claim if:

- multi-splat image parity fails on tiny overlap tests;
- gradient parity fails against V8/reference;
- the path requires per-pixel front-K/full-history storage by default;
- a CPU wait/readback appears in the timed path;
- full-prefix or stop-state recompute erases the forward gain;
- global atomics happen per pixel per splat per component;
- ICB or cross-API indirect execution is required before correctness is proven.

## Next Implementation Order

If the next session is Metal-focused:

1. Direction B: test descending/reverse order on realistic projected scenes,
   including cases with nontrivial support bounds and early-stop opportunities.
2. Direction B: decide whether `RGBA16F` is acceptable on real eval images.
3. Direction A: replace the two fixed full-screen fragments with 2-4 projected
   Gaussian quads in stable order and compare to a CPU/V8 `C/T` reference.
4. Direction A: feed exact imageblock forward from GPU-resident V8 tile bins
   only after the Gaussian-quad exact probe passes.
5. Direction A: pair exact forward with V8 compute backward only after forward
   parity and state capture pass.

If the next session is CUDA-focused:

1. Move `variants/v9_cuda_compute_first` to a CUDA host.
2. Build the scaffold and replace `TORCH_CHECK(false, ...)` stubs in order:
   `project_count_fused`, `emit_pairs`, CUB sort/ranges,
   `tile_forward_train`, then `tile_backward_replay`.
3. Add fixed-capacity scan/sort buffers and no hot-path `.item()` allocation.
4. Compare exact tile forward/backward to gsplat/Graphdeco/V8.
5. Only then consider Vulkan hardware raster interop.

## Decision

There are three valid paths, but they should not be mixed up:

- **B is the fastest near-term usable renderer.**
- **A is the only Metal path that could become exact hardware-assisted
  training.**
- **C is the best long-term CUDA training architecture.**

The next technical question for Metal is not "can hardware raster be fast?" It
can. The question is whether programmable tile/imageblock state can reproduce
V8's ordered `C/T/stop` contract without costing as much as V8 compute.
