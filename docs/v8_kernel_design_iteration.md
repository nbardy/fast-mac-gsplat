# v8 Kernel Design Iteration

Date: 2026-04-24

This note is the current design notebook for a v8 renderer. It is intentionally
not a victory claim. The measured result so far is that the v6 direct compute
path is still the strongest practical training baseline, while the v7 hardware
line and the v7.4 SGV-Seg handoff expose ideas but not a faster implementation.
The companion note `docs/v8_critical_theory_notes.md` records the critical
shader audit, backward-math equivalence, heavy-segmentation cautions, and stop
list.
The first implementation report is `docs/v8_field_report.md`; it records the
v6-derived `variants/v8` baseline, host-metadata split, correctness checks, and
initial benchmark gates.
The hardware tile raster design is in `docs/v8_hw_tile_raster_plans.md`; keep it
separate from `v8_direct` until GPU-native output interop and backward state are
proven.

The v8 goal is therefore narrow:

```text
Start from the proven v6 tile-compute architecture.
Keep exact forward/backward math.
Remove known hot-path sync and memory mistakes.
Only add heavy-tile segmentation if it launches over real heavy work.
Benchmark every step against v6 direct before calling it a successor.
```

## Position

The fastest credible v8 is not a hardware-raster-first renderer. It is a
v6-style compute training renderer that can share GPU visibility with a future
hardware eval path. Hardware rasterization can come later if it can consume the
same visibility and return a torch MPS tensor without CPU staging.

Training needs more than a final color. It needs the ordered alpha-compositing
function and enough state to replay derivatives. The state that has actually
worked is compact tile state:

- sorted tile-local Gaussian IDs
- tile offsets and counts
- tile stop counts
- compact overflow or heavy-tile lists when needed

The state that has not worked at 4K is per-pixel front-K or CPU-resident
hardware saved state.

## Lessons From Prior Variants

### v3

The v3 saved-order ablation is still one of the cleanest signals in the repo.
Saving forward sorted tile IDs made forward a little slower but forward+backward
faster:

| Case | Saved-order F+B delta | Backward estimate delta |
|---|---:|---:|
| 4K / 64K sparse sigma 1-5 | -8.7% | -13.5% |
| 4K / 64K medium sigma 3-8 | -7.5% | -12.5% |

Interpretation: do not sort again in backward if forward already had the
correct tile-local order.

### v5

v5 made the system usable in training loops:

- batched `[B,G,*]` API
- eval/train split
- saved sorted IDs in train mode
- overflow fallback
- batch chunking

It also showed that batch-general kernels can cost B=1 performance. v8 should
keep the batched API, but it should consider B=1 and B>1 specialization only if
benchmarks show indexing overhead is real.

### v6

v6 direct is the baseline to beat.

The main v6 findings:

- 16x16 tiles are the best tested default.
- One thread per pixel gives a natural 256-thread workgroup.
- With Metal simdgroup width 32, each tile workgroup has 8 simdgroups.
- Direct path beats active path on saturated 4K scenes because active scheduling
  removes almost no work and can add a full-output background prefill.
- Active scheduling should be conservative and measured, not a default.
- Backward barriers must use uniform tile-level loop bounds. Per-pixel early
  stop can only gate arithmetic inside those loops.

### v7 Hardware

The hardware render pipeline proved integration is possible but not fast enough
for training:

- original hardware backward was around tens of seconds at 4K
- front-K fixed some correctness but created a `pixels x splats` capture cliff
- tiled capture fixed the capture shape but kept CPU bins, CPU state, and
  texture readback
- hybrid v7.3 became practical only by routing training back to compute

Interpretation: hardware forward is not the training state. At best it is an
eval output path after shared GPU visibility and no-readback tensor return are
solved.

### v7.4 SGV-Seg

The v7.4 handoff contains a useful mathematical idea: associative chunk
descriptors for heavy tiles. Its implementation should not be used as the v8
base.

Measured local quick probes after disposable build fixes:

| Case | Renderer | Forward | Forward+backward |
|---|---:|---:|---:|
| 512x512 / 6K uniform | v6 direct | 8.82 ms | 16.68 ms |
| 512x512 / 6K uniform | v7.4 direct | 14.83 ms | 27.47 ms |
| 4K / 64K uniform | v6 direct | 89.25 ms | 163.90 ms |
| 4K / 64K uniform | v7.4 direct | 160.82 ms | 297.60 ms |

Obvious v7.4 implementation problems:

- macOS build packaging was broken as shipped
- the Python `_C` import path was broken for a pure `TORCH_LIBRARY` extension
- global bitonic sort over fixed capacity is too expensive
- the scan/classify pass is serial on one GPU thread
- backward issues per-pixel global atomics instead of tile-reduced atomics
- heavy buffers are allocated even when direct mode is used
- heavy kernels launch over `heavy_tile_capacity`, not actual heavy count
- capacity counters are not enforced by the Python wrapper
- the bridge waits synchronously on a private Metal command buffer

The v8 rule is: harvest the segmented math, not the implementation.

## Mathematical Contract

The renderer computes front-to-back alpha compositing for each pixel. Inputs are
already sorted by stable depth within each image.

For pixel center:

```text
p = (x + 0.5, y + 0.5)
d_i = p - mean_i
Q_i = [[a_i, b_i], [b_i, c_i]]
power_i = -0.5 * d_i^T Q_i d_i
raw_i = opacity_i * exp(power_i)
alpha_i = min(max_alpha, raw_i)
```

A splat contributes only when:

```text
power_i <= 0
alpha_i >= alpha_threshold
```

Forward recurrence:

```text
C_0 = 0
T_0 = 1
C_{i+1} = C_i + T_i * alpha_i * color_i
T_{i+1} = T_i * (1 - alpha_i)
out = C_N + T_N * background
```

Early stop is an optimization, not a different function:

```text
stop when T_i <= transmittance_threshold
```

The backward recurrence can be written with the suffix color behind splat `i`:

```text
S_after_i = color behind splat i, including background
dL/dcolor_i = grad_out * T_i * alpha_i
dL/dalpha_i = T_i * dot(grad_out, color_i - S_after_i)
```

Then:

```text
d_raw_i = d_alpha_i if raw_i < max_alpha else 0
d_power_i = d_raw_i * raw_i

d_opacity_i = d_raw_i * raw_i / max(opacity_i, eps)
d_a_i = d_power_i * (-0.5) * dx^2
d_b_i = d_power_i * (-1.0) * dx * dy
d_c_i = d_power_i * (-0.5) * dy^2

d_mean_x_i = d_power_i * (a_i * dx + b_i * dy)
d_mean_y_i = d_power_i * (b_i * dx + c_i * dy)
```

This is equivalent to the v6 implementation form that tracks `T_cur` and `gT`
in reverse. v8 should keep whichever form generates better Metal code, but the
math contract above is the invariant.

## Tile Support Math

Tile binning uses alpha-threshold support, not raw bounding boxes alone.

Given:

```text
alpha_threshold <= opacity * exp(-0.5 * q)
q <= tau
tau = -2 * log(alpha_threshold / max(opacity, eps))
det = max(a*c - b*b, eps)
```

Conservative half extents:

```text
half_x = sqrt(max(tau * c / det, 0))
half_y = sqrt(max(tau * a / det, 0))
```

Then perform an ellipse-vs-tile-rect test before incrementing or emitting a
tile reference. This avoids bloating bins from a loose axis-aligned box.

## v8 Architecture

The v8 baseline should be a fork of v6, not v7.4:

```text
Python depth sort
  -> flatten [B,G,*] to [B*G,*]
  -> GPU count tile refs
  -> GPU prefix offsets
  -> GPU emit tile refs
  -> direct tile forward train/eval
  -> saved sorted tile IDs + tile stop counts
  -> direct tile backward replay
  -> optional compact overflow/heavy path
```

State saved for backward:

| State | Purpose |
|---|---|
| `tile_counts` | per global tile ref count |
| `tile_offsets` | tile ref segment boundaries |
| `binned_ids` | sorted tile-local IDs after train forward |
| `tile_stop_counts` | max stopped prefix needed for backward |
| `overflow_tile_ids` | only tiles above fast cap |
| `overflow_offsets` | compact overflow segment boundaries |
| `overflow_sorted_ids` | sorted IDs for overflow replay |
| `stats` | counters used to validate capacities and policy choices |

Do not save:

- front-K tensors
- full per-pixel alpha history
- full per-pixel contributor lists
- fixed-capacity heavy tile buffers in direct mode
- full-image clones for overflow replacement

## Kernel Set

### Required Kernels

| Kernel | Work shape | Contract |
|---|---|---|
| `v8_count_tile_refs` | one thread per Gaussian | compute support and atomic-add tile counts |
| `v8_emit_tile_refs` | one thread per Gaussian | emit IDs into tile segments using tile cursors |
| `v8_tile_forward_eval` | one workgroup per tile | direct eval output, no saved stop count |
| `v8_tile_forward_state` | one workgroup per tile | direct train output, write sorted IDs and stop count |
| `v8_tile_backward_saved` | one workgroup per tile | replay stopped prefix, reduce gradients by tile, global atomic once per splat/tile |
| `v8_stats_tiles` | one thread per tile | compute active count, overflow count, max refs, histogram bands |
| `v8_compact_overflow_tiles` | one thread per tile | compact overflow IDs and offsets |

### Optional Kernels

| Kernel | When allowed |
|---|---|
| `v8_tile_heavy_segment_desc` | only if actual heavy tile count is nonzero |
| `v8_tile_heavy_segment_scan` | only over actual heavy tile count |
| `v8_tile_heavy_backward` | only if segmented path is selected and capacity counters pass |
| `v8_reduce_grad_partials` | only if deferred reduction beats tile-level atomics in a targeted ablation |
| `v8_hardware_eval_copy` | only after no-readback texture-to-MPS interop is proven |

## Workgroup Design

Default tile:

```text
tile_size = 16
threads = 16 * 16 = 256
simdgroups = 256 / 32 = 8
thread tid maps to:
  px = tid % 16
  py = tid / 16
```

Per tile forward:

```text
load IDs into threadgroup memory
sort tile-local IDs if not already sorted
write sorted order back in train mode
for chunks of 64 splats:
  load means/conics/colors/opacities into threadgroup memory
  each lane evaluates one pixel
  reduce alive lanes across simdgroups
  break only on tile-uniform no-live condition
write output pixel
write max stop count in train mode
```

Per tile backward:

```text
load sorted stopped prefix
forward replay to recover per-pixel T_final and end_i
reverse replay over uniform tile-level loop bounds
each lane computes local derivative contribution
simd_sum within each simdgroup
reduce 8 simdgroup partials in threadgroup memory
lane 0 atomically adds one reduced gradient per splat/tile
```

The important atomic rule:

```text
bad:  one global atomic per pixel/splat/component
good: one global atomic per tile/splat/component after reducing 256 pixels
```

For each contributing splat, the reduced global writes are:

```text
mean:     2 atomic adds
conic:    3 atomic adds
color:    3 atomic adds
opacity:  1 atomic add
total:    9 atomic adds per tile/splat
```

This is the v6 advantage that v7.4 gave up.

## Heavy Tile Segmentation

Segmented heavy-tile math is the one v7.4 idea worth retaining.

For a chunk evaluated with incoming transmittance `T=1`:

```text
D = (C_seg, T_seg)
```

Composition is associative:

```text
combine(A, B).C = A.C + A.T * B.C
combine(A, B).T = A.T * B.T
```

For heavy tile chunks:

```text
prefix_T[chunk,pixel] = transmittance entering chunk
suffix_C[chunk,pixel] = color behind chunk, including later chunks and background
```

Then each chunk backward can run independently:

```text
T_i = prefix_T[chunk,pixel] * local_T_i
dalpha_i = T_i * dot(grad_out, color_i - S_after_i)
```

But the implementation rules are strict:

- launch over actual `heavy_tile_count`, not capacity
- allocate heavy buffers only if `heavy_tile_count > 0`
- enforce `heavy_tile_count <= heavy_tile_capacity`
- enforce `chunk_count <= max_chunks`
- use tile-level reduction before atomics inside heavy backward if possible
- route uniform light-tile scenes away from segmentation

The first v8 should not enable segmentation by default. It should be an ablation
for overflow-adversarial and clustered-heavy scenes only.

## Memory Budget

At 4K:

```text
pixels = 4096 * 4096 = 16,777,216
RGB fp32 output = pixels * 3 * 4 = 201,326,592 bytes = 201 MB
B=4 output = about 805 MB
```

Common state at 4K/64K uniform from local v7.4 direct probe:

```text
tile refs = 642,580
max refs/tile = 27
mean refs/tile = 9.80
```

Approximate compact state:

```text
binned_ids int32        = 642,580 * 4 ~= 2.6 MB
tile_counts int32       = 65,536 * 4 ~= 0.3 MB
tile_offsets int32      = 65,537 * 4 ~= 0.3 MB
tile_stop_counts int32  = 65,536 * 4 ~= 0.3 MB
```

This is why compact tile state is acceptable.

Bad default heavy state, using v7.4 defaults:

```text
heavy_tile_capacity = 4096
max_chunks = 64
tile_pixels = 256

chunk_desc [cap,chunks,pixels,4] fp32 ~= 1.07 GB
suffix_C   [cap,chunks,pixels,4] fp32 ~= 1.07 GB
prefix_T   [cap,chunks,pixels]   fp32 ~= 0.27 GB
total heavy aux ~= 2.4 GB
```

This is unacceptable as a default. v8 must allocate heavy state based on actual
heavy count, or require the caller to pass a small explicit capacity and fail
closed when exceeded.

## CPU Sync Rules

Current v6 still has some CPU shape discovery and Python fallback pieces. v8
should remove them only when doing so does not make the kernel slower.

Allowed:

- CPU reads during explicit diagnostics or slow capacity growth
- CPU benchmark logging after synchronization
- Python depth sort if replacing it is not the current bottleneck

Not allowed in hot path:

- per-forward `.item()` to size `binned_ids`
- `.tolist()` loops to gather overflow segments
- CPU tile-bin construction
- texture readback
- command-buffer wait inside custom op unless required for correctness

Preferred allocation options:

1. Fixed `pair_capacity` and `overflow_capacity` from `RasterConfig`.
2. Cached high-watermark buffers owned by a module object.
3. Explicit two-step prepare/run API for long training loops.

The fixed-capacity version is simplest to benchmark.

## Active Policy

Do not make active scheduling aspirational. It must earn its cost.

Initial policy:

```text
direct if active_tile_fraction > 0.75 and overflow_count == 0
direct if p95_refs_per_tile < dense_threshold and overflow_count == 0
active only if active_tile_fraction < 0.10
active if overflow_count > 0 and compact path avoids enough work
heavy segmentation only if max_refs_per_tile > max_fast_pairs or targeted threshold
```

The policy must be backed by stats:

```text
total_pairs
active_tile_count
overflow_tile_count
max_refs_per_tile
p50/p90/p95/p99 refs per tile or histogram bands
mean_stop_ratio
p95_stop_ratio
heavy_tile_count
capacity_overflow flags
```

If stats require CPU readback in the measured path, log them in benchmark mode
only and keep the production path fixed-policy until a GPU policy is implemented.

## Hardware Forward Path

Hardware forward is not banned. It is just not the first v8 training target.

Hardware forward can be reintroduced if all of this is true:

- visibility is built once on GPU and shared with compute backward
- hardware forward obeys the same depth order and alpha equation
- render output stays GPU-resident as a torch MPS tensor
- no CPU draw-call construction in the hot path
- compute backward consumes the same tile IDs and stop state

If any of those fail, hardware belongs in eval experiments, not training.

## Implementation Sequence

### Stage 1: v8 Direct Baseline

Fork v6 into `variants/v8` and keep the public API close to v6.

Changes:

- add explicit capacity config fields:
  - `pair_capacity`
  - `overflow_tile_capacity`
  - `stats_enabled`
- add capacity counters
- keep v6 direct tile forward/backward math
- keep tile-level gradient reduction
- skip active and heavy segmentation initially

Promotion gate:

```text
v8_direct within +/- 3% of v6_direct on:
  512x512 / 6K / B=1,4
  4K / 64K / B=1
```

If Stage 1 cannot match v6, stop and profile. Do not add complexity.

### Stage 2: GPU Overflow Compaction

Replace Python overflow segment gathering with GPU compaction.

Promotion gate:

```text
overflow-adversarial cases improve
uniform no-overflow cases do not regress more than 2%
```

### Stage 3: No Full-Image Clone For Overflow

Current overflow replacement can clone or scatter full images depending on path.
v8 should update only overflow tiles.

Promotion gate:

```text
clustered/overflow scenes improve
no extra full-image write is visible in 4K memory traffic
```

### Stage 4: Heavy Segmented Ablation

Implement v7.4's associative chunk math inside the v6 tile architecture.

Rules:

- heavy buffer shape is `[actual_heavy_tiles, chunks, tile_pixels, ...]`
- launch grid uses actual heavy count
- counters fail closed
- heavy backward reduces within tile/chunk before global atomics

Promotion gate:

```text
must beat v8 direct on overflow-adversarial or clustered-heavy scenes
must not be enabled for uniform light-tile scenes
```

### Stage 5: Deferred Gradient Reduction Ablation

Only test if Xcode profiling shows global atomics dominate.

Design:

```text
tile backward writes compact partials:
  [tile_ref, gaussian_id, 9 floats]
sort/group by gaussian_id or emit into fixed splat buckets
reduce to final gradients
```

Risk:

```text
partial buffer traffic can exceed atomic cost
```

Promotion gate:

```text
wins on clustered-heavy traces
does not regress uniform 4K by more than 3%
```

### Stage 6: Hardware Eval Consumer

Only after compute training is stable:

```text
v8 visibility -> hardware eval output -> MPS tensor without CPU staging
```

This is eval-only unless exact training parity is proven.

## Benchmark Matrix

Every v8 stage must compare against:

- `v3_candidate` for B=1 single-image training where applicable
- `v5_batched`
- `v6_direct`
- `v6_auto`
- `v6_upgrade_direct`
- `v6_refined_direct`
- v7.4 direct only as a cautionary comparison

Minimum workloads:

```text
512x512 / 6K / B=1,4
1024x512 / 2K / B=1,4
4096x4096 / 64K / B=1
4096x4096 / 64K / B=4 where memory permits
```

Distributions:

```text
microbench_uniform_random
sparse_screen
clustered_hot_tiles
layered_depth
overflow_adversarial
real_trace
```

Modes:

```text
forward
forward_backward
accuracy on small cells
stats logging on all cells
```

## Stop Conditions

Stop an implementation branch immediately if:

- it needs global bitonic sort over fixed multi-million ref capacity
- it allocates heavy buffers when heavy count is zero
- it launches over capacity instead of actual compacted count
- it gives up tile-level reduction before atomics
- it uses per-pixel front-K as required training state
- it silently accepts capacity overflow
- it requires CPU readback for output or bins in the measured hot path

These are not theoretical concerns; each one has already hurt a prior variant.

## Current Recommendation

Do not write a new hardware-first kernel. Write v8 as a disciplined v6 fork:

```text
v8_direct first.
No segmentation until direct matches v6.
No hardware until output interop is solved.
No default active path until stats prove it.
No promotion without the full matrix.
```

The best chance of beating prior kernels is not a bigger kernel. It is removing
specific measured costs while preserving the v6 execution shape that already
works.
