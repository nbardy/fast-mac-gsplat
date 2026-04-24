# v8 Critical Theory Notes

Date: 2026-04-24

This note is the critical pass on the v8 direction. It answers three questions:

1. Do any existing shaders actually beat the v6-family compute path?
2. Is there analytical math that unifies or leans out backward?
3. What work is still likely to move the number, and what should be stopped?

## Verdict

No complete existing shader in this repo is a better 4K training renderer than
the v6 direct compute path.

The pieces worth keeping are narrower:

- v6 direct remains the default training core.
- v7 hardware raster can be useful for forward/eval experiments, especially if
  the output can stay GPU-resident.
- v7.2 tiled front-state is worth borrowing as an idea, not as an implementation.
- v7.4 SGV-Seg contains useful associative compositing math for heavy tiles, but
  its implementation shape is not acceptable for the default path.
- v6-upgrade/refined variants are worth targeted ablations, not a blind rewrite.

The best v8 is therefore a disciplined v6 fork:

```text
v6 tile compute core
  + fixed/cached pair capacity
  + no hot-path CPU readback
  + GPU overflow compaction
  + tile-local gradient reductions
  + optional heavy-tile segmentation only when actual heavy tiles exist
  + optional hardware eval only after no-readback texture/MPS interop
```

## Existing Shader Audit

| Line | What is useful | Why it does not replace v6 direct |
|---|---|---|
| v3 saved order | Forward writes sorted tile IDs; backward avoids re-sort. | Already imported into v5/v6. |
| v5 batched | Train/eval split, batch API, saved sorted IDs, overflow fallback. | v6 keeps the core and improves batch/backward behavior. |
| v6 direct | 16x16 tiles, 256 threads, chunked loads, saved IDs, stop counts, simdgroup reductions before atomics. | This is the current baseline to beat. |
| v6 active | Can help sparse/clustered/overflow tails. | Full-output prefill and policy mistakes can dominate dense 4K. |
| v6-upgrade/refined | Some 64k/backward-heavy cells are interesting. | Deep matrix did not broadly beat local v6. |
| v7 hardware | Forward-only clustered cases can win. | Training backward needs ordered alpha state; current hardware backward is too slow or wrong. |
| v7.1 front-K | Correctness proof for hardware backward state. | Per-pixel capture scales badly at 4K. |
| v7.2 tiled capture | Better local tile-bin idea; wins some smaller clustered cases. | 4K path still carries too much state/CPU cost. |
| v7.4 SGV-Seg | Associative segment descriptor is analytically useful. | Global fixed-capacity sort, serial scan, capacity launches, default heavy buffers, and per-pixel atomics are fatal. |

Hardware raster forward does not fundamentally break backward. The problem is
that fixed-function blending normally returns only the final color. Exact
training backward needs the ordered compositing prefix that was actually
processed, the early-stop point, and a way to accumulate gradients by Gaussian.
Once compute kernels have to rebuild that state, hardware forward no longer owns
the training pipeline. It can still be an eval consumer of the same visibility
metadata.

## Forward Contract

For one pixel, process visible splats in depth order until either the list ends
or transmittance crosses the early-stop threshold.

```text
T_0 = 1
C_0 = 0

d_i = p - mean_i
power_i = -0.5 * (a_i*dx_i^2 + 2*b_i*dx_i*dy_i + c_i*dy_i^2)
raw_i = opacity_i * exp(power_i)
alpha_i = min(max_alpha, raw_i)

visible_i = power_i <= 0 && alpha_i >= alpha_threshold

C_{i+1} = C_i + T_i * alpha_i * color_i
T_{i+1} = T_i * (1 - alpha_i)

out = C_M + T_M * background
```

`M` is the processed prefix length, not necessarily the full tile list. Any
segmented or hardware-assisted backward must respect this exact `M`.

## Backward Contract

Backward can be written cleanly with suffix color:

```text
S_after_i = normalized color behind splat i, ending at background

dcolor_i = grad_out * T_i * alpha_i
dalpha_i = T_i * dot(grad_out, color_i - S_after_i)
```

Geometry derivatives:

```text
draw_i = dalpha_i if raw_i < max_alpha else 0
dpower_i = draw_i * raw_i

dopacity_i = draw_i * exp(power_i)
da_i = dpower_i * (-0.5 * dx_i^2)
db_i = dpower_i * (-dx_i * dy_i)
dc_i = dpower_i * (-0.5 * dy_i^2)

dmean_x_i = dpower_i * (a_i*dx_i + b_i*dy_i)
dmean_y_i = dpower_i * (b_i*dx_i + c_i*dy_i)
```

The positive mean sign follows from `dx = pixel_x - mean_x`.

## Why v6 Backward Is Already The Right Math

v6 does not explicitly store `S_after_i`, but it implements the same recurrence
with a scalar:

```text
gT_i = dot(grad_out, S_after_i)
T_cur = T_{i+1}
T_prev = T_cur / (1 - alpha_i)

g_alpha_i = T_prev * (dot(grad_out, color_i) - gT_i)
g_color_i = grad_out * (T_prev * alpha_i)

gT_{i-1} = alpha_i * dot(grad_out, color_i) + (1 - alpha_i) * gT_i
T_cur = T_prev
```

So the v7.4 explicit suffix-color view and the v6 `gT` reverse recurrence are
equivalent for the processed prefix. The default v8 direct path should keep the
v6 scalar form unless Metal codegen proves a vector suffix form is faster.

The cleanup worth doing is structural, not mathematical:

- factor the reverse-composite helper once and share it across direct, active,
  overflow, and heavy paths;
- keep visibility gates and clamp gates identical across paths;
- prefer `dopacity = draw * exp(power)` when `exp(power)` is already available;
- keep the `raw < max_alpha` gradient gate strict;
- keep early-stop semantics identical to v6.

## Heavy-Tile Segmentation Math

The useful v7.4 idea is an associative compositing descriptor for a segment:

```text
D = (C_seg, T_seg)

combine(A, B).C = A.C + A.T * B.C
combine(A, B).T = A.T * B.T
```

This permits prefix/suffix scans over chunks:

```text
T_i = prefix_T_chunk * local_T_i
dalpha_i = T_i * dot(grad_out, color_i - S_after_i)
```

But storing suffix color as RGB is more than backward needs. Backward only needs:

```text
h_after_i = dot(grad_out, S_after_i)
```

For chunk descriptors:

```text
H_before_chunk = dot(grad_out, C_seg) + T_seg * H_after_chunk
```

A heavy path can therefore store or compute a scalar suffix-dot per chunk/pixel
instead of a 3-float suffix color. Pair that with `prefix_T` and `T_seg`.

Important limitation: the descriptor is exact for full compositing, but v6 uses
early stop. With early stop, the stop point depends on absolute incoming
transmittance. A segment descriptor computed as if incoming `T=1` is not enough
to decide where a globally prefixed chunk should stop. A correct heavy path must
either:

- carry the exact per-pixel stop point from forward, or
- recompute/find the stop point under the true incoming prefix `T`, then truncate
  suffix/prefix math at that point.

If it does not do this, gradients before the stop can see colors from splats that
forward skipped.

## Workgroup And Primitive Rules

Keep the v6 work decomposition for the direct path:

```text
one workgroup per tile
tile = 16x16 pixels
threads = 256
simdgroups = 8 on typical Apple GPU execution
chunk size = 32 or 64 splats, benchmarked but fixed per kernel variant
```

Core primitives:

- count tile references on GPU with ellipse-vs-rect rejection;
- prefix-sum counts without hot-path CPU synchronization;
- emit compact tile-local sorted IDs;
- train forward writes sorted IDs and tile stop counts;
- eval forward skips saved state;
- backward replays the stopped prefix and walks it in reverse;
- reduce all pixel lanes in the tile before global gradient atomics;
- compact overflow/heavy tile IDs on GPU and launch over actual compacted counts.

Do not switch to per-pixel global atomics. The right default accumulation shape
is still:

```text
per splat per tile after tile reduction:
  2 mean atomics
  3 conic atomics
  3 color atomics
  1 opacity atomic
```

Deferred gradient partials are only an ablation if profiling shows atomics are
the limiting cost. They add a large `[tile_ref, gaussian_id, gradients]` buffer
and another grouping/reduction pass, so memory traffic can erase the win.

## Memory And Bandwidth Reality

Full-image tensors are expensive at 4K:

```text
RGB fp32 image, B=1: 4096*4096*3*4 = 201 MB
RGB fp32 image, B=4: 805 MB
one fp32 scalar per pixel, B=1: 67 MB
one i32 stop index per pixel, B=1: 67 MB
```

That means:

- a full-image clone for overflow patching is a major bandwidth bug;
- saving `final_T` by default is not free;
- saving a per-pixel stop index by default is not free;
- per-pixel front-K state can explode quickly;
- heavy buffers must be allocated from actual heavy tile count, not capacity.

For direct v8, the lean state should stay:

```text
binned_ids          int32, total tile refs
tile_counts         int32, num_tiles
tile_offsets        int32, num_tiles + 1
tile_stop_counts    int32 or uint16, num_tiles
optional counters   small device buffer
```

Per-pixel `final_T` and per-pixel stop index can be benchmarked as explicit
ablation switches, but should not be default state.

## Stop List

Stop a v8 branch or keep it experimental if it requires any of these in the
timed training path:

- CPU `.item()`, `.tolist()`, `nonzero` scheduling, or Python per-tile loops;
- command-buffer wait/readback for normal forward/backward;
- CPU-owned hardware visibility bins;
- global fixed-capacity sort over mostly empty capacity;
- launch over capacity instead of actual compacted work;
- default heavy buffers when heavy count is zero;
- per-pixel global gradient atomics;
- full-image clone/scatter for overflow;
- silent capacity overflow;
- front-K or history buffers proportional to `pixels * splats`.

## Next Work That Can Actually Move The Number

1. Build a fixed/cached-capacity v8 direct path and remove the timed
   `tile_offsets[-1].item()` allocation decision.
2. Add device stats: total refs, max refs, p95 refs, stop ratio, overflow count,
   heavy count, active fraction, and capacity flags.
3. Move overflow compaction fully to GPU. No Python tile loops.
4. Patch overflow tiles in place or through a tile-local output kernel. No
   full-image clone.
5. Keep active scheduling behind a conservative policy. It must win sparse or
   clustered cases and stay off dense uniform 4K.
6. Add heavy segmentation only after v8 direct matches v6. Launch over actual
   heavy tiles and preserve v6-style tile reductions.
7. Test scalar suffix-dot heavy backward before any RGB suffix buffer.
8. Keep hardware raster as eval-only until it can consume GPU visibility and
   return GPU-native MPS output without staging.

## Promotion Gate

Do not call v8 better until it beats v6 direct on the boring cases first:

```text
512x512 / 6k, B=1 and B=4, forward and forward+backward
4096x4096 / 64k, B=1, forward and forward+backward
4096x4096 / 64k, B=4, forward and forward+backward if memory allows
uniform, sparse, clustered, layered, overflow-adversarial, and at least one real trace
```

Report:

```text
median ms
p95 ms
forward/backward split
launch count
allocation bytes
total tile refs
p95/max tile refs
stop ratio
overflow/heavy counts
capacity flags
image/gradient max error on checked cells
```

The highest-probability path is not a clever new hardware-backward shader. It is
making the v6-shaped path brutally clean, then adding one measured escape hatch
at a time.
