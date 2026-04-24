# v8/v9 Hardware Tile Raster Plans

Date: 2026-04-24

This note expands the lower-level hardware raster direction. It is not the
current `v8_direct` implementation. It is the design target for a separate
`v8_hw_experiment` or `v9_tile_hw` branch that tries to combine the best current
compute path with Metal render-pipeline features:

- v8/v6 tile visibility and sorted IDs;
- imageblocks / tile shaders for tile-local per-pixel state;
- programmable blending or raster order groups for ordered per-pixel updates;
- v8 backward math and tile-level gradient reductions;
- optional heavy-tile segment descriptors only for actual heavy work.

The goal is a 2-3x forward+backward speedup hypothesis, not a guarantee. The
branch must prove the speedup against `v8_direct` and must fail closed when
Metal interop or early-stop state is not exact enough.

Primary Metal references already captured in `docs/v8_field_report.md`:

- Apple Metal Feature Set Tables:
  `https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf`
- Apple Tile Shading Tech Talk:
  `https://developer.apple.com/videos/play/tech-talks/604/`
- Apple Raster Order Groups Tech Talk:
  `https://developer.apple.com/videos/play/tech-talks/605/`
- Apple Modern Rendering with Metal:
  `https://developer.apple.com/videos/play/wwdc2019/601/`
- PyTorch MPS docs:
  `https://docs.pytorch.org/docs/stable/mps.html`

## Non-Negotiable Math Contract

Forward:

```text
T_0 = 1
C_0 = 0

alpha_i = min(max_alpha, opacity_i * exp(power_i))
visible_i = power_i <= 0 && alpha_i >= alpha_threshold

C_{i+1} = C_i + T_i * alpha_i * color_i
T_{i+1} = T_i * (1 - alpha_i)

out = C_M + T_M * background
```

`M` is the exact processed prefix after early stop. Hardware raster can only be a
training path if backward sees the same `M`.

Backward:

```text
S_after_i = normalized color behind splat i, ending at background
dcolor_i = grad_out * T_i * alpha_i
dalpha_i = T_i * dot(grad_out, color_i - S_after_i)
```

The scalar v8/v6 form is:

```text
gT_i = dot(grad_out, S_after_i)
g_alpha_i = T_i * (dot(grad_out, color_i) - gT_i)
gT_{i-1} = alpha_i * dot(grad_out, color_i) + (1 - alpha_i) * gT_i
```

Any hardware path must preserve:

- stable depth order;
- alpha clamp and visibility gates;
- early-stop semantics;
- tile-level reduction before global gradient atomics.

## Shader Pair A: Hardware Forward / Eval

This is the first hardware target. It is useful even if training continues to
use `v8_direct`.

### Pseudocode A1: Tile Shader + Imageblock Forward

```text
render_encoder_begin
  bind sorted splat draw stream
  bind output texture or MPS-compatible output storage
  bind imageblock layout:
    pixel.C_rgb: fp32 or fp16x3 if error allows
    pixel.T: fp32
    pixel.stop_count: uint16 or uint32 optional
    pixel.flags: uint32 optional

tile_init_shader(tile_id):
  for each pixel lane in tile:
    imageblock.C[pixel] = 0
    imageblock.T[pixel] = 1
    imageblock.stop_count[pixel] = 0
    imageblock.flags[pixel] = 0

fragment_shader(splat_id, pixel):
  if imageblock.flags[pixel].stopped:
    return

  m, conic, opacity, color = load_splat(splat_id)
  power, alpha = eval_gaussian(pixel, m, conic, opacity)
  if not visible(power, alpha):
    return

  T = imageblock.T[pixel]
  imageblock.C[pixel] += T * alpha * color
  T_new = T * (1 - alpha)
  imageblock.T[pixel] = T_new
  imageblock.stop_count[pixel] += 1

  if T_new <= transmittance_threshold:
    imageblock.flags[pixel].stopped = 1

tile_flush_shader(tile_id):
  for each pixel lane in tile:
    out[pixel] = imageblock.C[pixel] + imageblock.T[pixel] * background
render_encoder_end
```

Low-level options:

- Use raster order groups if programmable blending/imageblock updates need
  ordered same-pixel access under overlapping fragments.
- Use imageblocks for `C/T` to avoid global-memory read/write per fragment.
- Store `stop_count` only if training parity requires it. For eval-only, omit it.
- Prefer fp32 `T`; fp16 `C` is an ablation only after image error is measured.
- Do not store front-K arrays in imageblock. Imageblock memory is limited and
  front-K grows with pixels and overlap.

Atomics:

- None in the forward/eval shader.
- Ordered same-pixel update should be handled by raster order groups or the
  render pass ordering model, not global atomics.

### Pseudocode A2: GPU-Driven Hardware Eval Stream

This variant keeps v8 compute visibility and uses hardware only for raster work.

```text
v8_count_tile_refs(...)
v8_prefix_counts(...)
v8_emit_tile_refs(...)
v8_stats_tiles(...)

if hardware_eval_supported and no_cpu_readback:
  build_indirect_draws_from_tile_bins(...)
  hardware_tile_forward_eval(...)
else:
  v8_tile_forward_eval(...)
```

Key rule: hardware eval consumes GPU-resident visibility. It does not rebuild
bins on CPU.

## Kernel / Algo Breakdown A

### A-Kernel 1: `hw_tile_forward_eval`

Purpose: produce image output using render-pipeline fragment generation and
tile-local compositing.

Inputs:

```text
sorted gaussian IDs or draw instances
means2d, conics, colors, opacities
tile metadata / viewport
background and thresholds
```

Outputs:

```text
out image, ideally directly MPS/Torch-visible
optional final_T image
optional stop_count image or tile stop summary
```

Work decomposition:

```text
one render tile maps to one screen tile or hardware tile region
fragment per covered pixel per splat quad
tile shader initializes and flushes imageblock state
```

Memory:

```text
imageblock:
  C: 3 floats or packed half/fp32 candidate
  T: 1 float
  flags/stop: optional 4 bytes

global:
  output RGB
  optional state only when training capture is enabled
```

Atomics:

```text
forward eval: none
training capture: none unless writing compact counters
```

Hardware optimization checklist:

- keep the render pass single-encoder if imageblock state must persist;
- avoid CPU texture readback;
- avoid separate CPU draw construction;
- use indirect draws only after GPU command generation is stable;
- align tile size with imageblock pressure, not just v8's 16x16 compute tile;
- profile raster order serialization on clustered overlap.

### A-Kernel 2: `hw_eval_to_mps_output`

Purpose: prove output interop. This is a gate, not an optimization.

Options:

```text
Option 1: render into an MTLTexture backed by storage that can be exposed as an
          MPS/Torch tensor without CPU staging.

Option 2: render into texture, then run a GPU blit/compute copy into the MPS
          tensor storage on the same command stream.

Option 3: fail the hardware path and use v8 compute output.
```

Rejected:

```text
waitUntilCompleted()
CPU texture readback
CPU-side image copy into Torch
private queue that races PyTorch MPS scheduling
```

Promotion gate:

```text
hardware forward output == v8 output within tolerance
no CPU wait/readback in the timed path
4K/64K forward beats v8_direct forward by enough to justify complexity
```

## Shader Pair B: Hardware-Assisted Training / Backward

This is the harder path. It should only start after Shader Pair A proves
GPU-native output.

### Pseudocode B1: Training Forward With Minimal State

```text
v8_count_tile_refs(...)
v8_prefix_counts(...)
v8_emit_tile_refs(...)

render_encoder_begin
tile_init_shader(tile):
  C = 0
  T = 1
  stop_index_or_count = 0
  stopped = false

fragment_shader(splat_id, pixel):
  if stopped:
    return

  alpha = eval_alpha(splat_id, pixel)
  if alpha invisible:
    return

  C += T * alpha * color[splat_id]
  T = T * (1 - alpha)
  stop_index_or_count += 1

  if T <= threshold:
    stopped = true

tile_flush_shader(tile):
  out[pixel] = C + T * bg
  if training_state_enabled:
    write final_T[pixel] or compact stop metadata
render_encoder_end
```

State options:

| State | Cost | Why use it |
|---|---:|---|
| tile stop count only | tiny | v8-compatible, but backward recomputes per-pixel end |
| per-pixel final_T | 67 MB per 4K B=1 fp32 | may remove one backward replay |
| per-pixel stop index | 67 MB per 4K B=1 i32 | exact early-stop index, expensive |
| front-K/history | unbounded | reject as default |

Default recommendation: tile stop count plus recompute, same as v8, until a
measured ablation proves per-pixel state wins.

### Pseudocode B2: Reverse Backward With Tile Reduction

```text
for each tile workgroup:
  load sorted IDs for tile
  stop_count = saved tile stop count

  // Reconstruct per-pixel final T and end_i.
  for pixel lane:
    T = 1
    end_i = 0
    for i in 0..stop_count-1:
      alpha = eval_alpha(id[i], pixel)
      if visible:
        T *= (1 - alpha)
        end_i = i + 1
        if T <= threshold:
          break

  gT = dot(grad_out[pixel], background)

  for i in reverse(stop_count):
    if i >= end_i:
      continue

    alpha, raw, power, d = eval_alpha(id[i], pixel)
    if not visible:
      continue

    T_prev = T / (1 - alpha)
    dot_color = dot(grad_out[pixel], color[id[i]])
    g_alpha = T_prev * (dot_color - gT)
    g_color = grad_out[pixel] * (T_prev * alpha)

    local_grad[pixel] = geometry_grad(g_alpha, raw, power, d)

    reduce local_grad across tile pixels for this splat
    if lane owns reduced value:
      atomic_add global grad arrays once per splat/tile/component

    gT = alpha * dot_color + (1 - alpha) * gT
    T = T_prev
```

Atomics:

```text
after tile reduction, per splat/tile:
  atomic_add grad_means.x
  atomic_add grad_means.y
  atomic_add grad_conics.a
  atomic_add grad_conics.b
  atomic_add grad_conics.c
  atomic_add grad_colors.r
  atomic_add grad_colors.g
  atomic_add grad_colors.b
  atomic_add grad_opacity
```

Rejected:

```text
per-fragment/per-pixel global atomics
front-K history as default state
global sort over fixed capacity
launch over heavy capacity instead of actual heavy tile count
```

## Kernel / Algo Breakdown B

### B-Kernel 1: `hw_train_forward_state`

Purpose: hardware forward plus just enough training state.

Inputs:

```text
same as hw_tile_forward_eval
saved sorted IDs from v8 binning
state mode: tile_stop | final_T | pixel_stop | debug
```

Outputs:

```text
out image
tile_stop_counts or compact stop metadata
optional final_T
optional per-pixel stop index
```

Work decomposition:

```text
fragment work handles coverage and C/T update
tile shader handles state init/flush
compute fallback handles unsupported devices or interop failure
```

Atomic policy:

- no gradient atomics in forward;
- if compact state uses counters, counters must be bounded and overflow-flagged;
- capacity overflow must set a device flag and route to compute fallback.

Low-level optimization:

- store minimal state first;
- use imageblock only for per-pixel `C/T/flags`;
- keep state writes linear and coalesced at tile flush;
- use pixel stop index only for heavy/clustered ablations;
- keep early-stop branch coherent by testing clustered and uniform separately.

### B-Kernel 2: `hw_or_compute_tile_backward_saved`

Purpose: exact gradients with v8 math.

Two allowed implementations:

```text
Option B2a: compute workgroup per tile
  Uses v8's proven threadgroup reductions and atomics.
  Hardware forward only provides optional final_T/stop metadata.

Option B2b: render/fragment-assisted backward
  Re-rasterizes splat quads in reverse or replays tile bins.
  Must reduce in quad/simd/tile groups before global atomics.
```

Default recommendation: start with B2a. It is boring but gives a clean test of
whether hardware forward helps training after interop/state costs.

Atomic details:

- Use relaxed floating-point atomics for final global gradient adds.
- Emit atomics after tile reduction, not per pixel.
- If profiling shows atomics dominate, test deferred partials:

```text
partial_grad_buffer[tile_ref, splat_id] = 9 floats
sort/group partials by splat_id
reduce partials in a second pass
```

This is an ablation only because the partial buffer can exceed atomic savings.

## Heavy Tile Segmentation Add-On

Use only when `actual_heavy_tile_count > 0`.

Descriptor:

```text
D = (C_seg, T_seg)
combine(A, B).C = A.C + A.T * B.C
combine(A, B).T = A.T * B.T
```

Backward scalar suffix:

```text
H_after = dot(grad_out, S_after)
H_before_chunk = dot(grad_out, C_seg) + T_seg * H_after
```

Important early-stop rule:

```text
segment descriptors are exact for full compositing;
with v8 early stop, the segment path must know or recompute the exact stopped
prefix under the true incoming T.
```

Kernel list:

```text
heavy_count_tiles
heavy_compact_tile_ids
heavy_segment_desc
heavy_segment_scan
heavy_backward_tile_reduce
```

Memory rules:

- allocate by actual heavy tile count;
- no default heavy buffers;
- no capacity launches;
- no RGB suffix buffer until scalar suffix-dot loses in a benchmark.

## Detailed Plan 1: Conservative Hardware Eval Then Compute Backward

Goal: get a real hardware-forward number without risking training correctness.

Stage 1: `v8_hw_eval_probe`

```text
fork variants/v8 -> variants/v8_hw_experiment
keep v8 binning and direct compute train path
add render pipeline only for no-grad/eval
render splat quads using existing sorted depth order
write output without CPU readback
fall back to v8 compute if interop is not GPU-native
```

Deliverables:

- image parity against v8 direct;
- forward timing for 512/6K and 4K/64K;
- proof that output reaches Torch/MPS without CPU staging.

Stop if:

- output requires `waitUntilCompleted()` in the hot path;
- output requires CPU texture readback;
- hardware forward is not at least 25% faster than v8 forward on a 4K case.

Stage 2: imageblock C/T

```text
add imageblock C/T state
add tile shader init/flush
test fp32 C/T first
test fp16 C only as an explicit accuracy ablation
```

Stage 3: shared visibility

```text
hardware eval consumes v8 tile bins
no CPU bin rebuild
no per-frame CPU draw list construction if indirect draw path is feasible
```

Promotion gate:

```text
forward-only:
  v8_hw_eval beats v8_direct by >=25% on 4K/64K uniform or clustered
  image max error inside existing tolerance
  no CPU readback
```

This plan can ship as eval-only even if backward remains compute.

## Detailed Plan 2: Training Hardware-Tile Path

Goal: prove or kill the 2-3x forward+backward hypothesis.

Stage 1: hardware forward + compute backward

```text
use hardware forward output
save tile_stop_counts or final_T according to mode
run v8 compute backward from saved sorted IDs
compare against v8_direct gradients
```

State modes to test:

```text
mode=tile_stop     tiny state, recompute per-pixel final T in backward
mode=final_T       +67 MB per 4K B=1, may save replay work
mode=pixel_stop    +67 MB per 4K B=1, exact stop index
mode=debug_history rejected except tiny validation scenes
```

Stage 2: backward-state ablation

Benchmark:

```text
512/6K B=1,B=4: uniform, sparse, clustered, layered
4K/64K B=1: uniform, clustered
overflow-adversarial with capacity flags
```

Record:

```text
forward ms
backward ms
state bytes
image/grad max error
tile refs
stop ratio
heavy tile count
atomic count estimate
CPU wait/readback count
```

Stage 3: tile-reduced backward

```text
keep one workgroup per tile/chunk
reduce 256 pixel lanes per splat before global atomics
test deferred partial reduction only if atomic counters dominate
```

Stage 4: heavy-tile segmentation

```text
enable only for actual heavy tiles
use scalar suffix-dot
respect exact early-stop prefix
launch over compacted heavy tile IDs
```

Promotion gate:

```text
training:
  4K/64K B=1 forward+backward <= 0.5 * v8_direct
  512/6K B=4 forward+backward <= 0.7 * v8_direct
  no correctness regression beyond current tolerance
  no CPU readback/wait in hot path
  memory overhead < 25% of v8_direct for default mode
```

Stop if:

- per-pixel state becomes required for all scenes;
- raster order serialization erases forward speedup;
- backward still rebuilds all useful state after hardware forward;
- MPS/Torch interop forces CPU staging;
- global atomics happen per fragment/pixel.

## Current Recommendation

Do not fold this into `v8_direct`. Keep `v8_direct` as the clean compute training
baseline. Start `v8_hw_experiment` with Plan 1. Only begin Plan 2 after Plan 1
proves GPU-native hardware output and a meaningful forward win.

If Plan 2 cannot beat `v8_direct` forward+backward, keep the hardware path as an
eval-only renderer and continue optimizing compute training with fixed-capacity
binning, GPU overflow compaction, and heavy-tile segmentation.
