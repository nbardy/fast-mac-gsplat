# Hardware Rasterizer + Fast Backward Handoff

Last updated: 2026-04-24

This document is the current handoff for the fast Mac gsplat rasterizer line. It
summarizes the variants, what improved, what regressed, and the current goal:
merge the useful hardware rasterizer work with the fast compute backward path
without reintroducing CPU round trips.

## Current Answer

The practical renderer today is a hybrid:

- use v5/v6-style MPS compute for training and backward
- keep v7-style hardware rasterization as an eval/forward research path
- do not use pure v7 hardware backward as the 4K training default yet

The strongest proof is the 4K/64K uniform forward+backward probe:

| Renderer | Total ms | Meaning |
|---|---:|---|
| `v5_batched` | 73.681 | current compute training reference |
| `v72_tiled_k2` | 438.782 | hardware backward after tiled capture fix, still too much CPU/state cost |
| `v73_hybrid_k2` | 76.464 | hybrid route, training uses v5 compute |
| `v73_hybrid_k2_hwtrain` | 436.552 | forced hardware train route, similar to v7.2 |

So v7.3 hybrid is 82.6% faster than v7.2 in that cell and within 3.8% of v5.
That is not because hardware backward became fast. It is because training was
routed back to the fast compute path while preserving hardware eval experiments.

## New v6-Best Kernel Plan

The next useful implementation target is a new v6-family compute renderer, not a
pure hardware-backward renderer. The design goal is a single GPU-resident
training pipeline that keeps the measured v6 direct strengths, removes remaining
CPU synchronization from binning/overflow/active scheduling, and leaves the v7
hardware path as an eval consumer of the same visibility metadata.

Working name: `v6_best` or `v6x`.

Follow-up design note: `docs/v8_kernel_design_iteration.md` is the current
kernel-design notebook for turning these ideas into a disciplined v6-derived v8
candidate. It records the math contract, old-kernel lessons, v7.4 SGV-Seg
findings, and staged promotion gates.

Critical follow-up: `docs/v8_critical_theory_notes.md` records the shader audit,
backward-math equivalence, memory/atomic stop list, and the stricter answer on
why hardware raster is eval-only until GPU-native output interop is solved.

### Success Criteria

Promote the new kernel only if it satisfies all of these:

- same small-scene image and gradient tolerance as current v6 direct and active
  reference checks
- no CPU readback in the forward/backward hot path except an explicit slow-path
  capacity grow or diagnostic mode
- 4K/64K B=1 uniform forward+backward beats the best current v6-family row, not
  just older hardware rows
- 4K/64K B=4 uniform stays on the direct path and does not reintroduce a full
  output prefill or clone
- sparse, clustered, layered, and overflow-adversarial cases either win or route
  to the known safer path with bounded regression
- hardware eval, if enabled, consumes the same GPU bin metadata and does not own
  a separate CPU visibility system

### System Architecture

The training architecture should stay tile-first:

```text
projected tensors [B,G,*]
  -> stable depth order per image
  -> flatten to [B*G,*]
  -> GPU tile support/bin pass
  -> GPU prefix/compact metadata
  -> direct tile render state path by default
  -> optional active tile path only when GPU tile stats justify it
  -> overflow tile path only for tiles above fast cap
  -> backward replay from sorted tile IDs and stop counts
  -> optional deferred gradient reduction for high-atomic-contention scenes
```

The current v6 line already has the right core shape:

- `GSP_TILE_SIZE=16`, so one default threadgroup covers one 16x16 tile
- `GSP_THREADS=256`, one thread per pixel
- Metal simdgroup width is 32 lanes, so the default tile threadgroup contains 8
  simdgroups
- `GSP_CHUNK=64` splats are staged at a time into threadgroup memory
- `GSP_FAST_CAP=2048` sorted tile IDs fit in threadgroup memory for normal tiles
- forward training writes sorted IDs back into `binned_ids` and saves one
  `tile_stop_counts[tile]`
- backward replays the stopped prefix and uses simdgroup reductions before
  global atomics

The new version should keep that execution model and fix the remaining system
costs around it.

### Data Model

Inputs stay fp32 and flattened after per-image depth sort:

| Tensor | Shape | Notes |
|---|---:|---|
| `means2d` | `[B*G,2]` | pixel-space mean after stable depth sort |
| `conics` | `[B*G,3]` | packed symmetric inverse covariance `[a,b,c]` |
| `colors` | `[B*G,3]` | RGB, fp32 for training accuracy |
| `opacities` | `[B*G]` | opacity multiplier |
| `meta_i32` | fixed | dimensions, tile counts, batch shape, caps, policy knobs |
| `meta_f32` | fixed | alpha threshold, transmittance threshold, background, eps |

Visibility metadata should be device-resident:

| Tensor | Shape | Producer | Consumer |
|---|---:|---|---|
| `tile_counts` | `[B*tiles_per_image]` | count kernel | prefix, policy, render |
| `tile_offsets` | `[tile_count+1]` | scan | emit, render |
| `binned_ids` | `[pair_capacity]` | emit | render/backward |
| `total_pairs` | scalar/device | scan | guard only |
| `active_tile_ids` | `[active_capacity]` | compact | active render/backward |
| `overflow_tile_ids` | `[overflow_capacity]` | compact | overflow render/backward |
| `overflow_offsets` | `[overflow_count+1]` | scan | overflow replay |
| `tile_stop_counts` | `[tile_count]` | train forward | backward |

The important change from current v6 is avoiding CPU shape discovery in the hot
path. Today, `tile_offsets[-1].item()`, overflow `.tolist()`, and active-policy
`.item()` calls can synchronize with the CPU. A best v6 should use one of these
allocation strategies:

1. Fixed-capacity pair buffers from `RasterConfig.max_total_pairs` for benchmark
   and training loops.
2. Cached high-watermark buffers that grow only on an explicit slow path.
3. A two-call API where the first call returns device stats and a separate
   capacity negotiation step is outside the timed training loop.

Do not allocate `binned_ids` from a CPU-read `total_pairs` inside every forward
pass if the goal is the best 4K training kernel.

### Forward Math

For pixel center `p = (x + 0.5, y + 0.5)` and Gaussian `g`:

```text
d        = p - mean_g
Q_g      = [[a, b], [b, c]]
power    = -0.5 * d^T Q_g d
raw_a    = opacity_g * exp(power)
alpha    = min(max_alpha, raw_a)
visible  = power <= 0 and alpha >= alpha_threshold

w        = T * alpha
rgb     += w * color_g
T       *= 1 - alpha
stop     when T <= transmittance_threshold

out_rgb  = rgb + T * background
```

Tile support uses the same conservative threshold:

```text
tau      = -2 * log(alpha_threshold / max(opacity_g, eps))
det      = max(a*c - b*b, eps)
half_x   = sqrt(max(tau * c / det, 0))
half_y   = sqrt(max(tau * a / det, 0))
bbox     = clipped [mean_x +/- half_x, mean_y +/- half_y]
```

Then each candidate tile performs an ellipse-vs-rect test before incrementing
the tile count or emitting a pair. This keeps bins sparse without relying on a
loose bounding box alone.

### Backward Math

Backward should continue to recompute alpha/transmittance rather than save
per-pixel front state. For a pixel gradient `go = dL/dout_rgb`, first replay the
tile prefix forward to recover `T_final` and the per-pixel `end_i`.

Initialize:

```text
T_cur = T_final
gT    = dot(go, background)
```

For splats in reverse depth order:

```text
denom     = max(1 - alpha, eps)
T_prev    = T_cur / denom
dot_gc    = dot(go, color_g)
g_alpha   = T_prev * (dot_gc - gT)
g_color  += go * (T_prev * alpha)

gate      = 1 if raw_alpha < max_alpha else 0
g_raw     = g_alpha * gate
g_power   = g_raw * raw_alpha

g_a      += g_power * (-0.5) * dx^2
g_b      += g_power * (-1.0) * dx * dy
g_c      += g_power * (-0.5) * dy^2

g_dx      = g_power * -(a*dx + b*dy)
g_dy      = g_power * -(b*dx + c*dy)
g_mean_x += -g_dx
g_mean_y += -g_dy
g_opacity+= g_raw * raw_alpha / max(opacity, eps)

gT        = alpha * dot_gc + (1 - alpha) * gT
T_cur     = T_prev
```

The reverse loops must remain uniform at the threadgroup level. Per-pixel
`end_i` can only gate contributions inside the loop. It must not change the
number of `threadgroup_barrier()` calls executed by different lanes.

### Kernel Primitives

The core kernel set should be explicit and small:

| Kernel | Workgroup | Main GPU primitive | Notes |
|---|---:|---|---|
| `bin_count_tiles_v6x` | 256 threads over splats | relaxed atomic add to `tile_counts` | writes bbox/tau scratch only if needed by emit |
| `scan_tile_counts_v6x` | library or custom scan | prefix sum | returns device `total_pairs`, no CPU sync in steady state |
| `bin_emit_pairs_v6x` | 256 threads over splats | relaxed atomic cursor per tile | writes `binned_ids` within fixed capacity |
| `compact_tile_lists_v6x` | 256 threads over tiles | prefix/compact | emits active and overflow tile IDs plus stats |
| `tile_forward_state_v6x` | one 16x16 tile per threadgroup | tile-local sort, shared chunk loads, simdgroup reductions | direct default |
| `tile_forward_eval_v6x` | one tile per threadgroup | same but no stop-count state | eval compute fallback |
| `tile_backward_replay_v6x` | one tile per threadgroup | reverse replay, simdgroup reduce, global atomics | direct default |
| `tile_active_forward_state_v6x` | one active tile per threadgroup | sparse tile queue | only for clearly sparse/overflow scenes |
| `tile_active_backward_v6x` | one active tile per threadgroup | sparse tile queue | shares direct math |
| `tile_overflow_forward_v6x` | one overflow tile per threadgroup, streaming chunks | no full-image clone | exact replay for over-cap tiles |
| `tile_overflow_backward_v6x` | one overflow tile per threadgroup | same backward recurrence | optional deferred gradient mode |
| `reduce_grad_partials_v6x` | splat-major segments | segmented reduction | only when atomics become the bottleneck |

Metal naming note: the code uses `simdgroup`; this is the Apple GPU equivalent
of the CUDA "warp" level for this design. With `GSP_TILE_SIZE=16`, every
threadgroup has 8 simdgroups of 32 lanes.

### Workgroup Plan

Default direct path:

```text
grid.x = B * tiles_y * tiles_x
threads_per_threadgroup = tile_size * tile_size = 256
lane tid maps to pixel:
  px = tid % tile_size
  py = tid / tile_size
```

Within each tile:

1. Load up to `count` tile IDs into `threadgroup uint shared_ids`.
2. Sort IDs by depth order. Current bitonic sort is acceptable for the first
   version because forward writes the sorted order back for backward.
3. Process splats in `GSP_CHUNK=64` chunks.
4. Stage means, conics, colors, and opacities into threadgroup memory.
5. Each lane evaluates its own pixel and maintains `T`.
6. Use `simd_sum`/`simd_max` plus a small threadgroup array to reduce alive
   lanes and tile stop counts across 8 simdgroups.
7. Save `tile_stop_counts[tile] = max(pixel_stop_i)`.

Active path:

```text
grid.x = active_tile_count
tg_id = active_tile_ids[active_idx]
```

Use it only when a device policy says enough work is skipped. The current
evidence says active scheduling is a loss on saturated uniform 4K scenes because
almost every tile is active and the output prefill costs hundreds of megabytes.

Overflow path:

```text
grid.x = overflow_tile_count
tg_id = overflow_tile_ids[local_overflow_idx]
stream sorted overflow IDs in chunks
```

Overflow handling must not build segments through CPU loops. The tile compaction
step should create overflow IDs and offsets on GPU. A slow CPU fallback can
exist for correctness debugging, but it should not be the benchmark path.

### Atomics Strategy

Current v6 does the right first-level reduction: each pixel computes local
partials, each simdgroup reduces them, the threadgroup reduces the 8 simdgroup
partials, and only one lane issues global atomics per splat/tile for:

```text
means2d:   2 atomic adds
conics:    3 atomic adds
colors:    3 atomic adds
opacities: 1 atomic add
```

That is 9 global atomic adds per contributing splat per tile, after reducing
across the 256 pixels in that tile. Keep this as the default because it avoids a
large partial-gradient buffer.

Add a second mode only for high-contention scenes:

```text
tile_backward_replay_v6x
  -> write compact partials [tile_ref, splat_id, 9 floats]
  -> reduce_grad_partials_v6x groups by splat_id
  -> one final write per splat
```

Use a device policy for this mode. It is likely useful when:

- the same Gaussian contributes to many tiles
- overflow or clustered scenes produce atomic hot spots
- measured total tile pairs are high enough that partial-buffer bandwidth is
  cheaper than global atomic contention

Do not use deferred reduction for the common light-tile uniform case until a
benchmark proves it wins; it adds memory traffic.

### Memory Bandwidth Budget

The fastest path is the one that avoids unnecessary full-image traffic. At 4K:

```text
4096 * 4096 * 3 * fp32 = about 201 MB per image
B=4 output             = about 805 MB
```

Therefore:

- no active-path full output background prefill when most tiles are active
- no `out_fast.clone()` except for overflow tiles that actually need exact
  replacement
- no CPU texture or tensor readback in training or eval hot paths
- no per-pixel saved front-K state for v6 training
- save sorted tile IDs and one stop count per tile instead
- keep overflow tile images as `[overflow_tiles, tile_size, tile_size, 3]`, not
  another full image

Threadgroup memory budget for the default kernel is reasonable:

```text
shared_ids  = 2048 * 4                 = 8192 bytes
chunk params= 64 * (float2+3+3+1)*4    = 2304 bytes
partials    = small 8-simdgroup arrays
```

Even if `GSP_FAST_CAP` grows to 4096, the ID buffer is about 16 KB. The real
memory danger is full-image traffic and large global partial-gradient buffers,
not the current per-tile shared memory.

### Device Policy

The new auto policy should be computed on GPU and read by kernels, not by CPU
`.item()` calls in the steady state. Required stats:

```text
active_tile_count
overflow_tile_count
total_pairs
max_pairs_per_tile
p95_pairs_per_tile or approximate histogram bands
estimated_output_prefill_bytes
estimated_atomic_refs
```

Initial routing:

- direct path when `active_tile_fraction > 0.75` and no overflow
- active path when `active_tile_fraction < 0.10`
- active path when overflow exists and compact replay avoids enough work
- deferred gradient reduction only when estimated atomic contention is high
- overflow exact path only for `tile_counts > max_fast_pairs`

The policy should be conservative. v6 direct is already strong, and previous
active auto modes caused large regressions when the heuristic was too eager.

### Implementation Stages

1. **v6x document and counters.**
   Add kernel timing and device stats for tile counts, total pairs, active
   fraction, overflow count, max/p95 pairs, stop ratio, and atomic-mode choice.

2. **No-CPU-sync binning API.**
   Replace per-forward CPU `total_pairs` shape discovery with a fixed-capacity
   or high-watermark `binned_ids` buffer. Overflow capacity failure should
   return a device flag and trigger an explicit slow grow/retry outside the
   benchmarked step.

3. **GPU active and overflow compaction.**
   Move active tile list creation and overflow segment creation out of Python
   loops. This removes `.tolist()` and per-tile CPU slicing from the training
   path.

4. **Direct state kernel cleanup.**
   Keep one tile per threadgroup and the current saved-order/stop-count design,
   but split reusable helpers so direct, active, and overflow variants share one
   math implementation. Preserve uniform barrier control.

5. **Overflow exact path without full-image clone.**
   Composite overflow tile outputs directly into the main output with a tile
   scatter kernel, or render overflow tiles into the main output in-place after
   the direct background path. Do not clone the whole 4K image for a few tiles.

6. **Deferred gradient reduction experiment.**
   Add the partial-gradient path behind a runtime policy. Benchmark it only on
   clustered and overflow-heavy scenes first.

7. **Hardware eval consumer.**
   Let hardware eval consume `tile_counts`, `tile_offsets`, and `binned_ids`
   from the v6x visibility pass if texture-to-MPS output interop is solved. Do
   not let hardware eval rebuild bins on CPU.

8. **Promotion matrix.**
   Compare `v5_batched`, `v6_direct`, `v6_auto`, `v6_upgrade_direct`,
   `v6_refined_direct`, and `v6x` across the existing benchmark distributions
   before changing defaults.

### Pseudocode: Forward Train Kernel

```text
kernel tile_forward_state_v6x(tile_id, tid):
  count = tile_counts[tile_id]
  if count == 0:
    write_background_for_pixel()
    if tid == 0: stop_counts[tile_id] = 0
    return

  if count > max_fast_pairs:
    mark_overflow_background_or_skip()
    if tid == 0: stop_counts[tile_id] = 0
    return

  load binned_ids[tile_offsets[tile_id] : +count] into shared_ids
  bitonic_sort(shared_ids)
  write sorted shared_ids back to binned_ids

  p = pixel_center(tile_id, tid)
  T = 1
  rgb = 0
  local_stop = 0

  for chunk in shared_ids by GSP_CHUNK:
    stage means/conics/colors/opacities in threadgroup memory
    if no live lanes in tile: break
    for splat in chunk:
      local_stop = splat_index + 1
      alpha = eval_alpha(p, splat)
      if alpha visible:
        rgb += T * alpha * color
        T *= 1 - alpha
        if T <= threshold: break

  stop_counts[tile_id] = threadgroup_max(local_stop)
  out[pixel] = rgb + T * background
```

### Pseudocode: Backward Replay Kernel

```text
kernel tile_backward_replay_v6x(tile_id, tid):
  stop_count = min(tile_counts[tile_id], tile_stop_counts[tile_id])
  if stop_count == 0 or stop_count > max_fast_pairs:
    return

  load sorted binned prefix into shared_ids
  p = pixel_center(tile_id, tid)
  go = grad_out[pixel]

  T_final = replay_forward_transmittance(shared_ids, stop_count, p)
  end_i = per_pixel_stop_index

  T_cur = T_final
  gT = dot(go, background)

  for chunk in reverse_chunks(shared_ids, stop_count):
    stage params
    for splat in reverse(chunk):
      local partials = 0
      if pixel valid and splat_index < end_i and alpha visible:
        compute reverse recurrence partials
      reduce partials across simdgroup and threadgroup
      lane 0 atomically accumulates 9 gradient values for this splat
```

### Open Design Questions

- Can the root op expose or reuse a persistent MPS buffer for `binned_ids`
  without fighting Torch tensor ownership?
- Is a fixed `max_total_pairs` acceptable for the training caller, with a grow
  retry when a scene exceeds it?
- Does a global pair sort by `(tile_id, depth_id)` beat per-tile bitonic sort on
  Apple GPUs, or does it add too much memory traffic?
- What threshold makes deferred gradient reduction beat direct atomics for real
  Dynaworld traces?
- Can hardware render output be copied into a torch MPS tensor without CPU
  staging? If not, v6x compute remains the training/eval default at 4K.

## Version Map

| Version | Renderer names | Main idea | What worked | What did not work |
|---|---|---|---|---|
| v2 root | `v2_fastpath` | older single-image Metal compute fast path | useful low-overhead baseline | no native batching, superseded for most work |
| v3 | `v3_candidate` | stronger single-image compute path with saved-order ideas | good B=1 large-scene training baseline | not the batch/default path |
| v5 | `v5_batched` | native batched compute renderer with eval/train split | fast training, compact tile state, batch support, overflow fallback | not always best at B=1, active/sparse policies live elsewhere |
| v6 | `v6_direct`, `v6_auto` | v5-style compute with direct and active-tile modes | current broad training baseline, especially B>1 | active scheduling can be slower unless it removes enough tile work |
| v6 upgrade | `v6_upgrade_direct`, `v6_upgrade_auto` | chief-scientist v6 upgrade handoff | sometimes fastest at 4K/64K normal training | not a universal replacement for local v6 |
| v6 refined | `v6_refined_direct`, `v6_refined_auto` | refined v6 handoff kept side by side | useful contender at 512/6K, especially some B=4 cells | small timing differences vs v6 upgrade often look like noise |
| v7 original | `v7_hardware` | Metal render pipeline forward plus replay backward | proved hardware raster can be wired through Torch | 4K backward was around 20 seconds, not viable |
| v7 finished | `v7_finished_hardware` | finished hardware handoff | some forward wins | backward gradients were wrong, do not use for training |
| v7.1 front-K | `v7_frontk_k2`, `v7_frontk_k4`, `v7_frontk_k8` | save per-pixel front-K state for hardware backward | fixed major hardware-backward correctness issue | front-K capture became `pixels x splats`; 4K total was 21-36 s |
| v7.2 tiled capture | `v72_tiled_k2`, `v72_tiled_k4`, `v72_tiled_k8` | capture front-K from tile bins instead of all splats | fixed the v7.1 scaling cliff, often strong at 512/6K | still slow at 4K due to CPU binning, CPU saved state, texture readback |
| v7.3 hybrid | `v73_hybrid_k2`, `v73_hybrid_k2_hwtrain` | hardware eval plus compute training route | 4K training is back near v5 speed | hardware eval still reads texture back to CPU-backed path before MPS tensor |

There is no separate v7.4 source handoff in this repo as of this note. The
download named `torch_metal_gsplat_v72_tiled_capture (1).tar.gz` matched the
stored v7.2 archive byte-for-byte.

## What Got Better

v5 made the compute renderer usable for real training:

- native `[B,G,...]` batch input
- eval/train split
- MPS tile bins and compact saved state
- overflow fallback
- batch chunking

v6 improved the operational training baseline:

- direct tile kernels became the preferred default for uniform workloads
- active-tile scheduling became an explicit policy, not an assumed win
- saturated-backward barrier bugs were fixed in the v6 family

v7.1 improved correctness of the hardware-backward idea:

- front-K saved state made gradients materially more correct than `v7_finished`
- small exactness smokes passed after compile/order/bounds fixes

v7.2 improved the main hardware scaling failure:

- replaced full-scene `capture_front_k` with `capture_front_k_binned`
- overflow replay became tile-bin local
- 4K/64K normal cases dropped from roughly 21-36 s in v7.1 to 0.31-0.49 s in v7.2

v7.3 improved product shape:

- no-grad/eval uses `forward_eval`, skipping front-K capture and training aux state
- training defaults to v5 compute via `train_backend="auto"`
- `_hwtrain` remains available to test the hardware backward route directly

## What Got Worse Or Stayed Bad

Hardware backward is still not the fast path.

v7 and v7 finished failed for training in different ways:

- original v7 had huge 4K backward time
- v7 finished had incorrect backward gradients

v7.1 fixed correctness but accidentally moved the bottleneck:

- front-K capture scanned all splats for every pixel
- at 4K/64K that dominated end-to-end runtime

v7.2 fixed the `pixels x splats` capture shape but kept major CPU costs:

- CPU packing and CPU tile-bin construction every forward pass
- CPU copies of the forward image and saved front state
- full-pixel front-K and backward passes, even with tile-local bins
- heavy overflow replay in adversarial central-tile scenes

v7.3 avoids those costs for training only by routing training to v5 compute. For
forward/eval it still has the important remaining hardware blocker:

- the hardware render output is a Metal texture that is read back before the API
  returns a torch MPS tensor

This is why v7.3 hardware eval improved over v7.2 but still lost badly at 4K:

| Case | v5 | v7.2 k2 | v7.3 hardware eval |
|---|---:|---:|---:|
| `512x512 / 6k / B=1` uniform forward | 7.615 ms | 11.430 ms | 7.861 ms |
| `4096x4096 / 64k / B=1` uniform forward | 11.781 ms | 183.679 ms | 133.129 ms |

## Best Current Defaults

Use these defaults unless a benchmark says otherwise:

| Workload | Default |
|---|---|
| general training | `v6_direct` or the v7.3 hybrid default route |
| 512/6K exploratory training | include `v72_tiled_k*` and `v73_hybrid_k*` in the matrix |
| 4K/64K normal training | v5/v6 compute family, not pure hardware backward |
| 4K/64K overflow-heavy B=4 training | `v5_batched` has been the strongest measured path |
| forward-only hardware experiments | `v73_hybrid_k2` |
| hardware-backward experiments | `v73_hybrid_k2_hwtrain` or `v72_tiled_k2`, not production defaults |

## Goal Architecture

The target is not "hardware rasterizer replaces everything." The target is:

1. Hardware rasterizer for forward/eval when it can return GPU-resident output.
2. Compute training/backward path using v5/v6 tile bins and saved tile state.
3. Shared tile metadata so eval and training do not maintain separate visibility systems.
4. No CPU tile-bin construction in the hot path.
5. No CPU front-state or texture readback in the hot path.

In other words, merge the hardware rasterizer and fast backward by sharing the
same GPU-resident bin/state infrastructure, not by forcing hardware raster to do
all training work.

## Next Engineering Steps

1. Remove hardware eval output readback.
   The key open question is whether the Metal render target can be exposed or
   copied into a torch MPS tensor without CPU staging. If not, hardware eval will
   remain a research path at 4K.

2. Move v7.2 tile-bin construction onto MPS/GPU.
   The v5 bin path already has the right shape: count tile refs on device,
   prefix-sum offsets on device, emit binned IDs on device.

3. Share bin metadata between hardware eval and compute train.
   Avoid maintaining two separate sorting/binning systems. The ideal split is
   one GPU visibility pass feeding either hardware eval or compute train.

4. Keep `_hwtrain` as a diagnostic renderer.
   It is useful for measuring hardware backward progress, but should not be the
   default path until it beats the compute train route on both speed and accuracy.

5. Benchmark with paired modes.
   Every future handoff should report at least:

   - `512x512 / 6k / B=1,4`
   - `4096x4096 / 64k / B=1`
   - distributions: uniform, sparse screen, clustered hot tiles, layered depth,
     overflow adversarial
   - modes: forward and forward+backward
   - renderers: current v5/v6 best, v72 tiled, v73 hybrid default, v73 `_hwtrain`

## How To Read Future Results

Use this checklist before promoting a hardware variant:

- Does image accuracy match dense reference on small MPS tests?
- Do gradients match dense reference on small MPS tests?
- Does 4K/64K forward+backward beat the compute route, not just older hardware?
- Is the timing win still present across sparse, clustered, layered, and overflow cases?
- Did the implementation remove CPU staging, or did it only move work between CPU stages?
- Is the fast path actually used under `torch.is_grad_enabled()`?

If the answer to the CPU-staging question is no, expect 4K hardware results to
look better than v7.2 but still worse than v5/v6.

## Key Files

| File | Purpose |
|---|---|
| `benchmarks/benchmark_full_matrix.py` | shared benchmark harness and renderer names |
| `docs/agent_notes_apr_22_performance_metrics.md` | main performance tables and deltas |
| `docs/v72_tiled_capture_field_report.md` | v7.2 tiled-capture validation and results |
| `variants/v7_hybrid_v5style/docs/v73_hybrid_v5style_plan.md` | v7.3 hybrid plan and routing notes |
| `variants/v7_hybrid_v5style/torch_gsplat_bridge_v73/rasterize.py` | train/eval routing implementation |
| `variants/v7_hybrid_v5style/csrc/metal/gsplat_sparse.mm` | v7.3 hardware eval and hardware train bridge |
