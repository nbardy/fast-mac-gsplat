# V9 Plan A: Exact Metal Tile/Imageblock Training Path

Date: 2026-04-25

Scope: Direction A after the V9 hardware raster reflection. This is a kernel
plan only. It does not claim that hardware raster training is already viable.

The plan is deliberately narrow: use Metal tile shaders, imageblocks, direct
Torch/MPS render targets, and optionally raster order groups to implement the
same ordered Gaussian alpha-compositing contract as V8. Fixed-function blending
is not part of the training path because it cannot expose the `C/T/stop` state
needed by backward.

## 1. Precise Goal

Build a fail-closed V9 Metal training experiment that proves or kills this
hypothesis:

```text
GPU-resident v8 visibility metadata
  -> Metal render/tile/imageblock exact forward
  -> minimal saved state
  -> v8-compatible backward replay
  -> no CPU staging, no private queue synchronization, no per-pixel atomics
```

The first successful version only needs to beat V8 forward while keeping V8
backward exact. It does not have to make backward a render pass. A render/tile
backward should only be attempted after the forward path proves multi-splat
parity and a real 4K speedup.

Promotion target:

```text
512x512 / 6K / B=1 and B=4:
  image and gradients within current V8 tolerance
  forward+backward <= 0.70x best V8 direct path

4096x4096 / 64K / B=1:
  image and gradients within current V8 tolerance
  forward+backward <= 0.50x best V8 direct path
  no CPU wait/readback in timed path
```

## 2. Non-Goals

- Do not use fixed-function source-over blending for training. It is fast but
  does not expose `T`, exact early stop, or backward prefix state.
- Do not use ICB execution in the shared path. ICB allocation is allowed for
  probes; execution previously crashed in AGX and must stay isolated.
- Do not save front-K or full per-pixel splat history as the default state.
- Do not build per-tile draw lists on the CPU.
- Do not read render targets, saved state, counters, or bin lists back to CPU in
  the timed path.
- Do not claim a training win from single-splat parity. The blocking case is
  overlapping multi-splat compositing.
- Do not fuse 3D projection in this direction. Direction A starts from V8-style
  projected tensors: `means2d`, `conics`, `colors`, `opacities`, `depths`.
  Projection fusion can be a later upstream producer if the exact 2D renderer
  survives the gates.

## 3. Existing Evidence To Respect

Working:

- Direct buffer-backed Torch/MPS render target works for RGBA32F and RGBA16F.
- Gaussian fragment eval works.
- RGBA16F output is a strong eval candidate.
- Tile/imageblock pipelines compile on the tested Apple M4.
- Raster order groups are reported as supported on the tested Apple M4.
- V8 compute backward already has the right recurrence and tile-reduced atomics.

Failing or incomplete:

- Fixed-function blending fails multi-splat V8 parity.
- Stable depth sort alone does not fix parity.
- Tile/imageblock probes have not yet proven a fragment + imageblock ordered
  compositing loop.
- ICB execute is fenced off after an AGX crash.
- Current tile execution probe proved dispatch and layout, not full init,
  fragment update, and flush semantics.

The useful conclusion is that output interop is mostly solved. The remaining
hard parts are ordered same-pixel updates, exact early-stop semantics, and
backward-compatible state.

## 4. Math Contract

For one pixel center:

```text
p = (x + 0.5, y + 0.5)
d_i = p - mean_i
Q_i = [[a_i, b_i],
       [b_i, c_i]]

power_i = -0.5 * (a_i * dx_i^2 + 2 * b_i * dx_i * dy_i + c_i * dy_i^2)
raw_i   = opacity_i * exp(power_i)
alpha_i = min(max_alpha, raw_i)

visible_i = (power_i <= 0) && (alpha_i >= alpha_threshold)
```

Forward, in stable ascending-depth order:

```text
C_0 = (0, 0, 0)
T_0 = 1

for i in sorted candidates:
  if T_i <= transmittance_threshold:
    stop before candidate i

  if visible_i:
    C_{i+1} = C_i + T_i * alpha_i * color_i
    T_{i+1} = T_i * (1 - alpha_i)
  else:
    C_{i+1} = C_i
    T_{i+1} = T_i

out = C_M + T_M * background
final_T = T_M
```

Important: V8's `tile_stop_counts[tile]` is not the number of visible
contributions. It is a conservative tile-local candidate prefix length. In the
current V8 state kernel, each valid pixel updates its local stop index before
calling `eval_alpha`, so invisible candidate splats still advance the prefix
while the pixel is alive. A fragment-only hardware path that increments a count
only when a fragment exists does not reproduce this value.

Backward starts from `grad_out = dL/dout_rgb` and replays the exact same visible
prefix. V8's default safe form recomputes `T_final` and each pixel's `end_i`
from sorted tile IDs:

```text
T_cur = T_final
gT    = dot(grad_out, background)

for i in reverse(processed prefix):
  denom  = max(1 - alpha_i, eps)
  T_prev = T_cur / denom
  dot_gc = dot(grad_out, color_i)

  g_alpha_i = T_prev * (dot_gc - gT)
  g_color_i = grad_out * (T_prev * alpha_i)

  clamp_gate = 1 if raw_i < max_alpha else 0
  g_raw_i    = g_alpha_i * clamp_gate
  g_power_i  = g_raw_i * raw_i

  g_a_i       = g_power_i * (-0.5 * dx_i^2)
  g_b_i       = g_power_i * (-dx_i * dy_i)
  g_c_i       = g_power_i * (-0.5 * dy_i^2)
  g_mean_x_i  = g_power_i * (a_i * dx_i + b_i * dy_i)
  g_mean_y_i  = g_power_i * (b_i * dx_i + c_i * dy_i)
  g_opacity_i = g_raw_i * raw_i / max(opacity_i, eps)

  gT    = alpha_i * dot_gc + (1 - alpha_i) * gT
  T_cur = T_prev
```

This recurrence is the non-negotiable training contract. Hardware forward can
only replace V8 forward if backward sees the same sorted candidates, gates,
`T_final`, and stopped prefix behavior.

## 5. Architecture Overview

Recommended starting architecture:

```text
Python/autograd wrapper
  stable depth sort per batch, same as V8
  projected tensors stay on MPS

V8 visibility compute
  count_tiles
  scan tile_counts
  emit_binned_ids
  optional per-tile sort or sorted writeback

V9 Metal render pass
  direct buffer-backed Torch/MPS output texture
  tile shader init: imageblock C/T/stopped state
  fragment shader: ordered Gaussian update into imageblock
  tile shader flush: output image plus optional state

Backward
  default: V8 compute backward replay from sorted tile bins
  optional state modes: tile_count, final_T, pixel_stop
  future only: render/tile-assisted backward if forward wins
```

The first exact training attempt should not make hardware responsible for
backward reductions. The risk is too high and V8 already has a correct
workgroup-per-tile backward.

## 6. Metal API Mapping

Required native pieces:

```text
MTLRenderPipelineDescriptor
  vertexFunction = v9_tile_ref_quad_vs or v9_global_quad_vs
  fragmentFunction = v9_exact_composite_fs
  colorAttachments[0].pixelFormat = RGBA32F or RGBA16F
  colorAttachments[0].blendingEnabled = NO
  supportIndirectCommandBuffers = YES only in isolated ICB harnesses

MTLTileRenderPipelineDescriptor
  tileFunction = v9_exact_init_tile or v9_exact_flush_tile
  threadgroupSizeMatchesTileSize = YES
  maxTotalThreadsPerThreadgroup = tile_size * tile_size
  colorAttachments[0].pixelFormat = output format

MTLRenderPassDescriptor
  colorAttachment[0].texture = direct texture over Torch MPS output buffer
  loadAction = Clear or DontCare depending on init path
  storeAction = Store

MTLRenderCommandEncoder
  setTileBuffer / setTileBytes for metadata and output-state buffers
  setVertexBuffer for draw records and projected tensors
  setFragmentBuffer for projected tensors and metadata
  dispatchThreadsPerTile(tile_size, tile_size, 1) for init
  drawPrimitives(... instanceCount=draw_record_count) for splat/tile refs
  dispatchThreadsPerTile(tile_size, tile_size, 1) for flush
```

Explicit-layout imageblocks should be accessed through:

```text
threadgroup_imageblock PixelState* ptr =
  imageblock_data.data(coord, 0, imageblock_data_rate::color)
```

The current probes already found that `imageblock<T>.read()` is not available
for this explicit-layout mode. Any implementation plan that depends on generic
`read()`/`write()` helpers needs to be rewritten or killed at compile time.

Direct output requirements:

```text
RGBA32F: bytes_per_pixel = 16, width multiple for 256-byte row alignment = 16
RGBA16F: bytes_per_pixel = 8,  width multiple for 256-byte row alignment = 32
R32F:    bytes_per_pixel = 4,  width multiple for 256-byte row alignment = 64
RG32F:   bytes_per_pixel = 8,  width multiple for 256-byte row alignment = 32
```

The render work must be encoded on the Torch MPS dispatch queue and current MPS
command buffer, matching the successful V9 direct-output probes. A private
queue or `waitUntilCompleted` disqualifies the path.

## 7. Per-Pixel Imageblock State

Measured imageblock layouts on Apple M4:

| Layout | Logical State | Measured Sample | 16x16 Tile | 32x32 Tile |
|---|---:|---:|---:|---:|
| `half4_baseline` | 8 B | 24 B/sample | 6,144 B | 24,576 B |
| `ct_fp32` | 16 B | 32 B/sample | 8,192 B | 32,768 B |
| `ct_stop_fp32_u32` | 20 B | 48 B/sample | 12,288 B | 49,152 B |
| `ct_stop_flags_fp32_u32x2` | 24 B | 48 B/sample | 12,288 B | 49,152 B |

Recommended first exact layout:

```text
struct PixelState {
  float4 c_t;        // C.r, C.g, C.b, T
  uint observed_i;   // debug or optional max local candidate ordinal
  uint flags;        // bit0 stopped, bit1 overflow/debug
};
```

Measured cost: 48 B/sample, 12 KiB per 16x16 tile, 48 KiB per 32x32 tile.

For a first correctness gate, use 16x16 tiles even though the probe observed a
32x32 footprint in one execution path. 16x16 matches V8 tile bins and keeps the
logical state smaller. If Metal reports or schedules the imageblock as 32x32,
that must be treated as an occupancy risk and measured explicitly.

Do not put per-splat history in imageblock. A 16x16 tile with `K=4` history of
`uint id + float alpha + float T` would add roughly:

```text
256 pixels * 4 entries * 12 B = 12,288 B logical
```

before alignment. This doubles useful state and still overflows on real
clustered scenes.

## 8. Global State Options

Default exact state:

```text
binned_ids       uint32, scene dependent
tile_offsets     int32, tile_count + 1
tile_counts      uint32, tile_count
tile_stop_counts int32, optional, default may equal tile_counts
out_rgb or out_rgba
```

State modes:

| Mode | Extra Global State | Correctness | Expected Use |
|---|---:|---|---|
| `tile_count_full` | none beyond tile counts | exact, slowest backward | first exact gate |
| `tile_stop_compute` | 4 B/tile | exact if produced by V8-like compute | default if it wins |
| `final_T` | 4 B/pixel | exact only if generated with same gates | ablation |
| `pixel_stop` | 4 B/pixel | exact only if local candidate index is correct | ablation |
| `front_K` | unbounded | exact only if K covers prefix | reject as default |

The fragment path cannot safely produce V8 `tile_stop_counts` by counting only
fragments. Fragment generation skips pixels outside the splat quad, but V8's
candidate prefix advances through invisible candidates. If we need exact
`tile_stop_counts`, use a V8-like compute postpass or use `tile_count_full`.

## 9. Memory And Bandwidth Estimates

### Pixel State

Assume B=1.

| Item | 1080x1920 | 4096x4096 |
|---|---:|---:|
| Pixels | 2,073,600 | 16,777,216 |
| 16x16 tiles | 8,160 | 65,536 |
| 32x32 tiles | 2,040 | 16,384 |
| `tile_stop_counts` i32 | 0.031 MiB | 0.25 MiB |
| `final_T` fp32 | 7.91 MiB | 64.00 MiB |
| `pixel_stop` i32 | 7.91 MiB | 64.00 MiB |
| `final_T + pixel_stop` | 15.82 MiB | 128.00 MiB |
| RGB fp32 output | 23.73 MiB | 192.00 MiB |
| RGBA32F render target | 31.64 MiB | 256.00 MiB |
| RGBA16F render target | 15.82 MiB | 128.00 MiB |

### Imageblock Pressure

Imageblock storage is transient tile-local memory, not a global allocation. The
full-frame-equivalent numbers are still useful for pressure comparison:

| Layout | 1080p Equivalent | 4096 Equivalent |
|---|---:|---:|
| `half4_baseline`, 24 B/sample | 47.8 MiB | 384 MiB |
| `ct_fp32`, 32 B/sample | 63.8 MiB | 512 MiB |
| `ct_stop_flags`, 48 B/sample | 95.6 MiB | 768 MiB |

At 4096x4096, a 48 KiB 32x32 tile state can reduce occupancy enough to erase
the raster win. This is why the first exact attempt should test both:

```text
16x16, 48 B/sample, V8-aligned
32x32, 48 B/sample, fewer tile dispatches but higher per-active-tile pressure
```

### Visibility List

`binned_ids` cost:

```text
4 bytes * total_tile_references
```

Every 100M tile references is about 381 MiB. Direction A must not duplicate this
list for hardware and compute. The hardware path consumes the same V8 visibility
metadata.

### Bandwidth Red Flags

Do not add:

```text
RGBA32F render target + RGB32 output copy + final_T + pixel_stop
```

at 4096, because that is:

```text
256 MiB + 192 MiB + 64 MiB + 64 MiB = 576 MiB
```

before visibility lists, gradients, and input tensors. If exact forward needs
both RGBA32F and per-pixel state to beat V8, the design is probably losing.

## 10. Work Responsibilities

### Compute Visibility Kernels

Reuse or adapt V8:

```text
count_tiles:
  one thread per Gaussian
  compute alpha-support bbox
  ellipse-vs-tile test
  atomic add tile_counts

scan_tile_counts:
  prefix sum on GPU
  produce tile_offsets and total_pairs

emit_binned_ids:
  one thread per Gaussian
  write Gaussian IDs into tile bins

sort_tile_bins:
  either V8 bitonic inside tile forward/state
  or a separate GPU pass that writes sorted binned IDs before render
```

The first Direction A prototype may reuse V8's current allocation path. A
promotable version must move toward fixed-capacity or cached high-watermark
buffers so `total_pairs` does not force CPU synchronization every frame.

### Tile Shader Init

One tile shader dispatch over each render tile:

```text
threadgroup/tile lane maps to one pixel inside tile
state.C = 0
state.T = 1
state.flags = 0
state.observed_i = 0
```

This stage proves imageblock init and barrier semantics.

### Vertex Shader

Two possible ingestion modes:

1. Global sorted splat draw:

```text
instance_id = global sorted Gaussian ID
emit Gaussian bbox quad
```

This is simplest for fragment/imageblock correctness, but it does not consume
tile bins and cannot produce V8 tile-local stop prefixes.

2. Tile-ref draw stream:

```text
DrawRec {
  uint tile_id;
  uint local_i;    // index in sorted binned_ids for that tile
  uint splat_id;
  uint flags;
}

instance_id = tile reference record
emit quad clipped to tile rect
```

This is the serious path. It can be encoded as one instanced draw over a
GPU-built record buffer. It avoids per-tile CPU draw calls. Cross-tile primitive
order does not matter because records are clipped to one tile; same-tile records
must be sorted by `local_i`.

### Fragment Shader

Fragment responsibilities:

```text
load PixelState for current pixel from imageblock
if stopped: return
load splat params
eval power/raw_alpha/alpha
if not visible: return
state.C += state.T * alpha * color
state.T *= (1 - alpha)
if state.T <= threshold: state.flags |= stopped
state.observed_i = max(state.observed_i, local_i + 1) for debug only
write PixelState back
```

This requires ordered same-pixel updates. Without ROG or an equivalent ordered
imageblock access guarantee, overlapping fragments can race or reorder and the
math is wrong.

### Tile Shader Flush

One tile shader dispatch after draws:

```text
load PixelState
write output RGB/RGBA:
  out = C + T * background
optional write final_T[pixel] = T
optional write pixel_stop[pixel] = observed_i
optional write debug flags
```

Do not rely on `observed_i` as V8's exact prefix unless the draw stream emits
tile-covering records for every candidate/pixel or a separate compute pass
validates equivalence.

### Backward Kernel

Default: V8 compute backward replay:

```text
one 16x16 tile per threadgroup
256 lanes, one pixel per lane
8 simdgroups of 32 lanes
load sorted binned IDs
recompute T_final and end_i
reverse replay
reduce 256 pixel partials per splat
one global atomic add per splat/tile/component
```

This keeps the hardest atomic problem in the proven compute kernel family.

## 11. Sorted Splat Ingestion Strategy

### Strategy 0: Global Sorted Draw, Correctness Probe

Use stable sorted projected tensors, draw one instanced quad per Gaussian, and
let hardware raster generate fragments. This tests:

- fragment Gaussian eval;
- imageblock `C/T` update;
- ROG ordering;
- direct output flush.

It does not test tile-bin ingestion and is not a complete training path.

Kill it if multi-splat parity fails against V8 on tiny overlap cases.

### Strategy 1: V8 Bins Plus Full Backward Prefix

Use V8 visibility for backward but hardware forward for output:

```text
hardware forward:
  global sorted draw or tile-ref draw

backward:
  stop_count = tile_counts[tile]
  replay full sorted tile bin
```

This is exact and avoids relying on fragment-derived stop metadata. It may be
slower than V8 backward on dense scenes because it disables early-stop savings.
It is still the cleanest first training parity gate.

### Strategy 2: GPU-Built Tile-Ref Draw Stream

Create a GPU record per tile reference:

```text
for each tile:
  for local_i in sorted tile bin:
    rec = {tile_id, local_i, splat_id}
```

Then issue one draw:

```text
drawPrimitives(type=triangle, vertexStart=0, vertexCount=6,
               instanceCount=record_count)
```

The vertex shader reads `rec[instance_id]`, computes the clipped quad for that
splat within that tile, and passes `tile_id`, `local_i`, and `splat_id` to the
fragment shader.

To avoid CPU synchronization:

- first prototype may use known `binned_ids.size(0)` from the existing V8 API;
- promotable version uses fixed-capacity records plus a valid flag, or a
  GPU-generated indirect draw argument buffer;
- ICB execution is not required for this step and should not be used until the
  separate ICB harness is stable.

This strategy gives the fragment shader the tile-local ordinal needed for debug
and state ablations. It still does not solve V8 `tile_stop_counts` unless every
alive pixel observes every candidate ordinal, including invisible candidates.

### Strategy 3: Compute Stop-Count Postpass

If early-stop savings are necessary, run a small V8-like compute state pass:

```text
input: sorted binned_ids, tile_counts
output: tile_stop_counts
work: same candidate prefix logic as V8, optionally without RGB writes
```

This adds compute work but may be cheaper than per-pixel final state and keeps
backward exact.

## 12. Atomics Strategy

No global atomics in forward.

Backward default:

```text
for each tile and each splat in reverse prefix:
  each pixel lane computes local gradient partials
  simdgroup reduce partials
  threadgroup reduce 8 simdgroups
  one lane issues global atomics:
    grad_mean_x
    grad_mean_y
    grad_conic_a
    grad_conic_b
    grad_conic_c
    grad_color_r
    grad_color_g
    grad_color_b
    grad_opacity
```

That is 9 global float atomics per contributing splat/tile after reducing 256
pixels. This is acceptable as the default because it matches V8's proven shape.

Rejected:

```text
per-fragment global atomics
per-pixel global atomics
fragment shader atomic gradient accumulation
unbounded partial-gradient buffers
```

Optional high-contention mode:

```text
tile_backward_replay
  -> write partial_grad[tile_ref] = {splat_id, 9 floats}
  -> segmented reduce by splat_id
  -> final write to grad tensors
```

Only test this when counters show global atomics dominate clustered or overflow
scenes. For normal uniform scenes, it probably adds bandwidth and loses.

## 13. Exact Forward Pseudocode

### A-Forward-0: V8 Visibility

```text
kernel count_tiles(gid):
  if gid >= G: return
  tau = -2 * log(alpha_threshold / max(opacity[gid], eps))
  if opacity[gid] <= alpha_threshold or tau <= 0:
    bbox[gid] = empty
    return

  q = conics[gid]
  det = max(q.a * q.c - q.b * q.b, eps)
  half_x = sqrt(max(tau * q.c / det, 0))
  half_y = sqrt(max(tau * q.a / det, 0))
  bbox = clipped(mean +/- half)

  for tile in bbox tiles:
    if ellipse_intersects_tile(mean, q, tau, tile):
      atomic_add(tile_counts[tile], 1)

scan tile_counts -> tile_offsets

kernel emit_binned_ids(gid):
  repeat tile support test
  idx = atomic_add(tile_cursor[tile], 1)
  binned_ids[idx] = gid

kernel sort_or_prepare_bins(tile):
  load tile bin
  stable sort by already-depth-sorted gid or explicit depth key
  write sorted IDs back
```

### A-Forward-1: Tile Init

```text
tile_kernel exact_init_tile(imageblock PixelState ib, tid2):
  if tid2 outside imageblock size: return
  state_ptr = ib.data(tid2, 0, imageblock_data_rate::color)
  state_ptr->c_t = float4(0, 0, 0, 1)
  state_ptr->observed_i = 0
  state_ptr->flags = 0
```

### A-Forward-2: Tile-Ref Vertex

```text
vertex exact_tile_ref_quad_vs(vertex_id, instance_id):
  rec = draw_records[instance_id]
  if rec.invalid:
    emit degenerate vertex

  g = rec.splat_id
  tile_rect = tile_id_to_pixel_rect(rec.tile_id)
  bbox = gaussian_alpha_bbox(mean[g], conic[g], opacity[g])
  quad = intersect(bbox, tile_rect)
  if quad empty:
    emit degenerate vertex

  pos = select one of 6 quad vertices
  out.position = pixel_to_clip(pos)
  out.pixel_pos = pos
  out.splat_id = g
  out.tile_id = rec.tile_id
  out.local_i = rec.local_i
```

### A-Forward-3: Ordered Fragment Update

```text
fragment exact_composite_fs(varyings):
  pixel = floor(varyings.pixel_pos)
  state_ptr = imageblock.data(pixel, 0, imageblock_data_rate::color)
  state = *state_ptr

  if state.flags & STOPPED:
    return

  g = varyings.splat_id
  p = pixel + 0.5
  d = p - mean[g]
  power = -0.5 * (a*dx*dx + 2*b*dx*dy + c*dy*dy)
  if power > 0:
    return

  raw_alpha = opacity[g] * exp(power)
  alpha = min(max_alpha, raw_alpha)
  if alpha < alpha_threshold:
    return

  C = state.c_t.rgb
  T = state.c_t.a
  C = C + T * alpha * color[g]
  T = T * (1 - alpha)

  state.c_t = float4(C, T)
  state.observed_i = max(state.observed_i, varyings.local_i + 1)
  if T <= transmittance_threshold:
    state.flags |= STOPPED

  *state_ptr = state
```

Required invariant: same-pixel fragments must observe stable local order. If
ROG/imageblock ordering cannot enforce this, the path is invalid for training.

### A-Forward-4: Tile Flush

```text
tile_kernel exact_flush_tile(imageblock PixelState ib, tid2):
  pixel = tile_origin + tid2
  if pixel outside image: return

  state_ptr = ib.data(tid2, 0, imageblock_data_rate::color)
  state = *state_ptr
  rgb = state.c_t.rgb + state.c_t.a * background

  out[pixel] = rgb or rgba(rgb, state.c_t.a)

  if write_final_T:
    final_T[pixel] = state.c_t.a

  if write_pixel_stop_debug:
    pixel_stop[pixel] = state.observed_i

  if write_debug_flags:
    debug_flags[pixel] = state.flags
```

## 14. Exact Backward Pseudocode

Default backward remains compute. It may consume optional `final_T`, but the
first exact version should recompute to avoid trusting unproven state.

```text
kernel exact_backward_replay(tile_id, tid):
  count = tile_counts[tile_id]
  if count == 0: return
  if count > max_fast_pairs: route overflow exact path

  stop_count =
    if state_mode == tile_stop_compute:
      min(count, tile_stop_counts[tile_id])
    else:
      count

  load sorted binned_ids[tile_offsets[tile_id] : +stop_count]

  pixel = tile_pixel(tile_id, tid)
  valid = pixel inside image
  p = pixel + 0.5
  go = valid ? grad_out[pixel] : 0

  // Reconstruct exact final T and per-pixel end index.
  T_final = 1
  end_i = stop_count

  for chunk_start in 0..stop_count step CHUNK:
    stage chunk params into threadgroup memory
    barrier
    alive_total = reduce_alive(valid && T_final > threshold)
    if alive_total == 0: break

    if valid && T_final > threshold:
      for local in chunk:
        g = shared_ids[chunk_start + local]
        alpha = eval_alpha(pixel, g)
        if visible(alpha):
          T_final *= (1 - alpha)
          if T_final <= threshold:
            end_i = chunk_start + local + 1
            break
    barrier

  T_cur = T_final
  gT = dot(go, background)

  for chunk_end in reverse_chunks(stop_count, CHUNK):
    stage chunk params into threadgroup memory
    barrier

    for local in reverse(chunk):
      global_i = chunk_start + local
      g = shared_ids[global_i]

      local_grad = zero

      if valid && global_i < end_i:
        alpha, raw_alpha, power, d = eval_alpha(pixel, g)
        if visible(alpha):
          denom = max(1 - alpha, eps)
          T_prev = T_cur / denom
          dot_gc = dot(go, color[g])

          g_alpha = T_prev * (dot_gc - gT)
          local_grad.color = go * (T_prev * alpha)

          gate = raw_alpha < max_alpha ? 1 : 0
          g_raw = g_alpha * gate
          g_power = g_raw * raw_alpha

          local_grad.conic_a = g_power * (-0.5 * dx * dx)
          local_grad.conic_b = g_power * (-dx * dy)
          local_grad.conic_c = g_power * (-0.5 * dy * dy)

          g_dx = g_power * (-(a * dx + b * dy))
          g_dy = g_power * (-(b * dx + c * dy))
          local_grad.mean_x = -g_dx
          local_grad.mean_y = -g_dy

          local_grad.opacity = g_raw * raw_alpha / max(opacity[g], eps)

          gT = alpha * dot_gc + (1 - alpha) * gT
          T_cur = T_prev

      simd_reduce local_grad
      threadgroup_reduce 8 simdgroups
      if owner lane:
        atomic_add global gradients for g

    barrier
```

Uniform-barrier rule: `end_i` is per-pixel and must only gate math inside the
loop. It must not change the number of barriers executed by different lanes.

## 15. Render-Assisted Backward Is A Later Branch

If forward wins and V8 compute backward becomes the bottleneck, a render/tile
backward can be explored. It must keep the same reverse recurrence and
tile-reduced atomics. The only plausible shape is:

```text
tile shader or compute shader per tile
  load sorted tile IDs
  reconstruct T_final/end_i
  reverse loop IDs
  reduce 256 pixel lanes per splat
  global atomics once per splat/tile/component
```

A fragment backward that issues atomics per fragment is a kill condition. It
will be too slow and too nondeterministic under clustered overlap.

## 16. Accuracy Bugs To Watch

Known or likely failure modes:

1. Fragment order differs from V8 stable sorted order.
2. ROG only orders some resources, not the imageblock state being updated.
3. Fragment-derived `observed_i` counts visible fragments, not V8 candidate
   prefixes.
4. Bbox clipping causes pixels outside the quad to skip invisible candidate
   ordinals, breaking any attempt to derive `tile_stop_counts`.
5. Alpha clamp derivative differs at `raw_alpha == max_alpha`.
6. Background is applied twice if output stores `C` and later wrapper composites.
7. Half precision output hides small parity errors in RGB.
8. Row alignment silently forces fallback or blit path.
9. Private command queue introduces races with PyTorch MPS tensors.
10. A validation run accidentally uses fixed-function blending and reports a
    false pass on single-splat cases.

The primary accuracy target is overlapping multi-splat parity, not constant
color or one-Gaussian checks.

## 17. Risk List

### Ordering

Gaussian alpha compositing is non-commutative. If two fragments for the same
pixel update `C/T` out of sorted order, the image and gradients are wrong.
ROG/imageblock ordering must be proven with an adversarial two-splat overlap
test where swapping order changes output.

### Raster Order Groups

ROG is a correctness dependency, not a speed feature. It can serialize hot
pixels and erase the forward win on clustered scenes.

### Imageblock Limits

Measured useful state is 48 B/sample. At 32x32 this is 48 KiB per active tile.
That can lower occupancy enough that V8 compute wins.

### Fragment + Imageblock Semantics

Current probes compiled tile functions. They did not prove that the final
fragment function can safely read/write the same explicit-layout imageblock
across init/draw/flush stages. This gets its own compile and runtime gate.

### ICB

ICB execution crashed previously. Direction A does not need ICB for the first
tile-ref instanced draw. If indirect execution is revisited, it must be in a
separate harness with Metal validation enabled and `supportIndirectCommandBuffers`
set on the pipeline descriptor.

### Row Alignment

Direct buffer-backed render targets require 256-byte row alignment. Widths that
do not satisfy the format multiple must use a GPU copy fallback or padded
output. CPU staging is not allowed.

### Command Queue Interop

The render pass must be encoded on Torch's MPS queue/command buffer. A private
Metal queue can force synchronization or race with PyTorch's view of tensor
storage.

### Memory Explosion

`final_T + pixel_stop` costs 128 MiB at 4096 B=1 and 512 MiB at B=4. Front-K is
not a default training state.

### Occupancy

Tile shaders plus 48 B/sample imageblock state may have lower occupancy than
V8's compute threadgroup memory path. Measure clustered and layered scenes, not
only uniform sparse scenes.

### Validation Hazards

Metal validation should be enabled for isolated crash probes, but timings with
validation enabled are not performance data. ICB and ROG tests must be separated
from safe benchmark paths.

## 18. Milestone Gates

### Gate A0: Fragment/Imageblock Compile Gate

Deliverable:

```text
minimal render pass:
  init tile shader writes C/T
  fragment updates C/T for a constant alpha
  flush tile shader writes direct output
```

Pass:

- builds on target macOS/M4;
- no ICB;
- direct MPS output validates;
- two overlapping draws produce order-sensitive expected values.

Kill:

- fragment cannot access/update required imageblock state;
- ROG cannot order same-pixel updates;
- implementation requires CPU readback or private queue wait.

### Gate A1: Global Sorted Exact Forward

Deliverable:

```text
stable sorted splat draw
imageblock C/T forward
V8 parity harness for single and overlapping multi-splat cases
```

Pass:

- max RGB error within V8 tolerance on overlap cases;
- `final_T` debug matches compute recompute on tiny scenes;
- no fixed-function blending.

Kill:

- multi-splat errors remain like fixed blending;
- output only passes when overlap is absent;
- ROG serialization makes 512/6K slower than V8 forward by more than 25%.

### Gate A2: V8 Bin Consumer

Deliverable:

```text
GPU-built DrawRec buffer from tile_counts/tile_offsets/binned_ids
one instanced draw over tile refs
quads clipped to tile rect
```

Pass:

- no CPU per-tile draw loop;
- same image as global sorted path;
- no duplicate visibility list.

Kill:

- needs ICB execution before basic correctness;
- record buffer bandwidth dominates;
- tile clipping breaks parity.

### Gate A3: Training Exact With Compute Backward

Deliverable:

```text
hardware exact forward
V8 compute backward with stop_count = tile_counts or compute stop counts
gradient parity tests
```

Pass:

- gradients match V8 reference tolerance;
- 512/6K B=1,B=4 forward+backward not slower than V8 by more than 10%;
- 4K/64K B=1 shows a credible path to beating V8.

Kill:

- exact gradients require per-pixel history by default;
- full-prefix backward erases all forward gains;
- memory overhead exceeds 25% of V8 default state.

### Gate A4: Stop-State Ablation

Test:

```text
tile_count_full
tile_stop_compute
final_T
pixel_stop
final_T + pixel_stop
```

Pass:

- one mode improves forward+backward without breaking parity;
- default extra global state remains small, ideally `tile_stop_counts`.

Kill:

- only per-pixel state wins and memory overhead is unacceptable;
- fragment-derived stop metadata is not exact;
- compute stop-count postpass costs as much as V8 forward.

### Gate A5: Promotion Matrix

Benchmark:

```text
512x512 / 6K / B=1, B=4
1080x1920 / 6K / B=1
4096x4096 / 64K / B=1

distributions:
  uniform
  sparse
  clustered hot tiles
  layered depth
  overflow adversarial

modes:
  forward
  forward+backward
  image max error
  gradient max error
  total refs
  max/p95 refs per tile
  stop ratio
  atomic estimate
  CPU wait/readback count
```

Promote only if it beats best V8 on the target cells and does not fail the
sparse/clustered/layered cases.

## 19. Kill Criteria Summary

Kill Direction A as a training path if any of these are true:

- exact imageblock forward cannot match V8 multi-splat parity;
- ordered fragment updates require serialization that removes the forward win;
- backward must save front-K or per-pixel history by default;
- hardware forward plus exact backward is not faster than V8 direct on 4K/64K;
- GPU-built draw stream requires unsafe ICB execution;
- row alignment or tensor ownership forces CPU staging;
- command queue interop cannot be made safe with Torch MPS;
- global atomics happen per fragment or per pixel;
- memory overhead exceeds the output savings from hardware rasterization.

If killed for training, keep the V9 hardware path as eval/preview only:

```text
v9_hw_output_planes_probe
  direct RGBA16F output
  fragment Gaussian eval
  no backward claim
```

and continue training optimization in the V8 compute family.

## 20. Strongest And Weakest Parts Of This Direction

Strongest:

- It attacks the real blocker: explicit ordered `C/T` state instead of
  fixed-function blending.
- It reuses V8 visibility and backward math, avoiding a full rewrite of the
  only proven training path.
- It has clear fail gates before the project spends time on risky ICB or
  render-assisted backward work.

Weakest:

- Fragment/imageblock/ROG ordering may serialize clustered pixels enough that
  V8 compute remains faster.
- Exact stop metadata is awkward in a fragment-driven pipeline because invisible
  candidates do not generate fragments.
- The serious tile-ref draw stream can become bandwidth-heavy, and a promotable
  no-CPU-sync version eventually needs indirect draw arguments or another
  GPU-resident dispatch mechanism.

The practical first implementation should therefore be small: prove exact
multi-splat imageblock forward, then pair it with V8 compute backward. Anything
more ambitious before that is likely to repeat the old hardware-backward
failure mode.
