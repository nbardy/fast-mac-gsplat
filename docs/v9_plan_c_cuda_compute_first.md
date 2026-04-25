# V9 Plan C: CUDA Compute-First Rasterizer

Date: 2026-04-25

Direction C is the CUDA route. It is intentionally compute-first. The Metal V9
work showed that direct GPU-resident output can be very fast, but fixed-function
hardware blending does not expose the ordered `C/T/stop` state needed for
training. CUDA should use its strongest native tools first: CUB scan/sort,
cooperative groups, shared memory, warp/block reductions, direct Torch CUDA
tensor writes, and CUDA Graphs. A Vulkan hardware-raster branch remains useful,
but it should be separate and later.

## Goal

Build an accurate CUDA V9 forward+backward rasterizer that matches the V8/gsplat
3DGS math contract while improving launch, allocation, sort, and backward
atomic behavior.

The first promoted CUDA path should:

- match V8/gsplat/Graphdeco image and gradient tolerances on small and medium
  correctness cases;
- use fused 3D projection and opacity-aware tile intersection;
- avoid CPU `.item()` shape reads in the steady-state training loop;
- use CUB scan/radix sort for tile-reference construction;
- run one CUDA block per 16x16 tile for forward and backward;
- save compact forward state for exact backward replay;
- reduce gradients inside warp/block before global atomics;
- write directly to Torch CUDA tensors;
- benchmark stage timings and memory counters against V8, gsplat, and
  Graphdeco-style baselines.

## Non-Goals

- Do not start with Vulkan, Direct3D, mesh shaders, fragment interlock, or
  hardware rasterization. CUDA cannot directly launch NVIDIA fixed-function
  raster units from a CUDA kernel.
- Do not use fixed-function alpha blending for training correctness.
- Do not save per-pixel contributor lists or front-K histories as the default
  backward state.
- Do not allocate pair, heavy, or partial buffers from a CPU-read count every
  iteration.
- Do not replace a proven CUB sort/scan with a custom global bitonic sort.
- Do not enable heavy-tile segmentation or deferred reduction until a measured
  baseline identifies the bottleneck.
- Do not claim a forward-only eval win as a training win.

## Evidence From The Local Work

The Mac results are a useful warning even though CUDA hardware is different:

| Path | Signal |
|---|---|
| V8 compute | Best local training baseline; 4K/64K uniform was `12.142 ms` forward and `67.579 ms` forward+backward. |
| V9 direct Metal output | Solved GPU-resident output; 4K direct constant render dropped from `15.947 ms` blit to `4.683 ms` direct. |
| V9 fixed eval | Fast but not parity; single splat matches, multi-splat errors reach about `0.124`. |
| V9 RGBA16F | Best eval-format candidate; 4K/64K Gaussian median `6.123 ms` RGBA32F vs `4.958 ms` RGBA16F. |
| Tile/imageblock probes | Feature shape is promising, but compile/layout probes did not implement ordered `C/T/stop`. |
| ICB execute | Unsafe in the Metal branch after AGX crash; do not make indirect graphics execution a CUDA assumption. |

The key conclusion: output bandwidth matters, but the real training contract is
ordered front-to-back compositing plus enough saved state for exact backward.

## Full CUDA Pipeline

```text
inputs:
  means3d, quats/scales or cov3d, colors, opacities, cameras

stage 1:
  project_count_fused
    -> means2d, conics2d, depths, radii/support, tiles_per_gauss

stage 2:
  CUB DeviceScan
    -> pair_offsets, total_pairs_device

stage 3:
  emit_pairs
    -> isect_keys, gaussian_ids

stage 4:
  CUB DeviceRadixSort or DeviceSegmentedRadixSort
    -> sorted_keys, sorted_gaussian_ids

stage 5:
  encode_tile_ranges
    -> tile_offsets / tile_ranges

stage 6:
  tile_forward_train or tile_forward_eval
    -> RGB output, alpha/final_T or tile_stop_counts, last_ids optional

stage 7:
  tile_backward_replay
    -> grad_means2d, grad_conics, grad_colors, grad_opacities

stage 8:
  project_backward
    -> grad_means3d, grad_cov/quats/scales, optional camera grads

optional:
  deferred_partial_reduce
  heavy_tile_segmented_forward_backward
  CUDA Graph replay
```

The scan/sort boundary should remain explicit. Fusing projection, count, emit,
sort, and raster into one giant kernel would make capacity handling, sorting,
and validation harder while not removing the core need for ordered tile lists.

## Math Contract

### 3D Projection To 2D Conic

For a Gaussian with world mean `mu_w` and covariance `Sigma_w`, camera transform
`R, t`, and intrinsics `fx, fy, cx, cy`:

```text
mu_c = R * mu_w + t
x = mu_c.x
y = mu_c.y
z = mu_c.z

u = fx * x / z + cx
v = fy * y / z + cy
mean2d = (u, v)
depth = z
```

The projection Jacobian at the mean is:

```text
J = [ fx/z      0   -fx*x/(z*z)
        0    fy/z   -fy*y/(z*z) ]
```

Camera-space covariance:

```text
Sigma_c = R * Sigma_w * R^T
Sigma_2d = J * Sigma_c * J^T
```

Add a small blur/regularization term if matching the chosen reference requires
it:

```text
Sigma_2d_regularized = Sigma_2d + eps_2d * I
```

Invert the 2D covariance to get the conic matrix:

```text
det = max(Sigma_xx * Sigma_yy - Sigma_xy^2, eps)
a =  Sigma_yy / det
b = -Sigma_xy / det
c =  Sigma_xx / det
Q = [a b; b c]
```

If using antialiasing compensation, keep it explicit:

```text
opacity_eff = opacity * compensation
```

Do not hide compensation inside a renderer-specific constant. It must be
validated against the selected reference.

### Opacity-Aware Tile Support

For pixel offset `d = p - mean2d`:

```text
q = a*dx^2 + 2*b*dx*dy + c*dy^2
power = -0.5 * q
raw_alpha = opacity_eff * exp(power)
alpha = min(max_alpha, raw_alpha)
visible = (power <= 0) and (alpha >= alpha_threshold)
```

Tile support should be bounded by the alpha threshold:

```text
alpha_threshold <= opacity_eff * exp(-0.5*q)
q <= tau
tau = -2 * log(alpha_threshold / max(opacity_eff, eps))
```

Conservative axis-aligned support from `Q`:

```text
det_q = max(a*c - b*b, eps)
half_x = sqrt(max(tau * c / det_q, 0))
half_y = sqrt(max(tau * a / det_q, 0))
```

The emit pass should still do an ellipse-vs-tile or tighter rect test before
writing references. Loose boxes inflate sort cost and backward work.

### Forward Recurrence

For each pixel, sorted front-to-back by depth and stable tie-break:

```text
C_0 = 0
T_0 = 1

for i in visible sorted splats:
  C_{i+1} = C_i + T_i * alpha_i * color_i
  T_{i+1} = T_i * (1 - alpha_i)
  stop when T_{i+1} <= transmittance_threshold

out = C_M + T_M * background
```

This is the contract the Metal fixed-function path failed to expose. CUDA must
own the recurrence in shader code.

### Backward Reverse Recurrence

Let `g = dL/dout`. Initialize the suffix gradient through final transmittance:

```text
T_cur = T_final
gT_next = dot(g, background)
```

Replay processed splats in reverse order. For splat `i`, recompute
`alpha_i`, `raw_alpha_i`, and local geometry:

```text
denom = max(1 - alpha_i, eps)
T_prev = T_cur / denom
dot_c = dot(g, color_i)

d_alpha = T_prev * (dot_c - gT_next)
d_color = g * (T_prev * alpha_i)

clamp_gate = 1 if raw_alpha_i < max_alpha else 0
visible_gate = 1 if power_i <= 0 and alpha_i >= alpha_threshold else 0
d_raw = d_alpha * clamp_gate * visible_gate
d_power = d_raw * raw_alpha_i

d_opacity_eff = d_raw * raw_alpha_i / max(opacity_eff, eps)

d_a = d_power * (-0.5) * dx^2
d_b = d_power * (-1.0) * dx * dy
d_c = d_power * (-0.5) * dy^2

d_mean_x = d_power * (a*dx + b*dy)
d_mean_y = d_power * (b*dx + c*dy)

gT_prev = alpha_i * dot_c + (1 - alpha_i) * gT_next
T_cur = T_prev
gT_next = gT_prev
```

Projection backward then maps `d_mean2d`, `d_conic`, `d_depth`, and optional
compensation gradients back to `means3d`, covariance, quats/scales, and camera
parameters. Keep this in a separate kernel at first; fusing it into raster
backward would obscure correctness and increase register pressure.

## Kernel Primitives

### CUB

Use CUB/CCCL as the default:

- `cub::DeviceScan::ExclusiveSum` over `tiles_per_gauss`.
- `cub::DeviceRadixSort::SortPairs` with `DoubleBuffer` for global sort.
- `cub::DeviceSegmentedRadixSort::SortPairs` as an ablation for per-image
  segments when toolkit support is available.
- `cub::BlockReduce` or `cub::WarpReduce` for gradient reductions if
  cooperative-groups/manual shuffle codegen is worse.
- `cub::BlockScan` only for heavy-tile segmentation or deferred partial
  compaction.

Sort key layout:

```text
uint64 key =
  image_id << (tile_bits + depth_bits)
  | tile_id << depth_bits
  | depth_key
```

For ascending front-to-back depth, use a monotonic float-to-uint transform for
positive camera-space depths, or pack a quantized depth key after confirming
ordering tolerances. Stable equal-depth order needs a deterministic tie-break:

```text
secondary = gaussian_id or sorted_input_id
```

If depth ties must be exact and common, use a wider composite sort strategy or
sort by `(tile, depth, id)` encoded across key/value. Do not depend on unstable
same-key ordering.

### Cooperative Groups And Warp Intrinsics

Default block:

```text
tile_size = 16
threads_per_block = 256
warps_per_block = 8
thread tid -> pixel (tid % 16, tid / 16)
```

Use:

- `cooperative_groups::thread_block` for explicit block sync.
- `cooperative_groups::tiled_partition<32>` for warp reductions.
- `__ballot_sync`, `__any_sync`, and `__all_sync` for active/visible masks.
- `__shfl_down_sync` or `cg::reduce` for component reductions.
- `__match_any_sync` for warp-aggregated atomics when duplicate Gaussian IDs
  appear inside a warp.
- `__syncthreads_count(done)` for tile-wide early exit, with all block threads
  participating.

Barrier rule: per-pixel early stop may gate arithmetic, but all threads must
take the same barriers and loop structure.

### Shared Memory Layout

Baseline shared memory per tile batch:

```text
extern __shared__ uint8_t smem[];

int32  sh_ids[CHUNK];          // 256 * 4  = 1 KB
float2 sh_mean[CHUNK];         // 256 * 8  = 2 KB
float3 sh_conic[CHUNK];        // 256 * 12 = 3 KB
float  sh_opacity[CHUNK];      // 256 * 4  = 1 KB
float3 sh_color[CHUNK];        // 256 * 12 = 3 KB
float  sh_depth[CHUNK];        // optional, 1 KB
float  sh_warp_partials[8][K]; // gradients, < 2 KB typical
```

Expected total: about 10-16 KB for 3-channel training. Prefer SoA or
alignment-padded vector loads if Nsight shows poor coalescing.

### Async Copy And TMA

Baseline should use normal coalesced global loads into shared memory.

SM80+ ablation:

```text
cuda::memcpy_async or cp.async
double-buffer sh_* chunks
compute chunk k while prefetching chunk k+1
```

Use only after profiling shows splat-parameter loads are a limiter. Small,
irregular, sorted splat chunks may not benefit enough to justify complexity.

SM90+ TMA/thread-block-cluster branch:

- not part of the baseline;
- consider only for large structured tile queues, heavy-tile segmentation, or
  output/state plane movement;
- reject if setup and shape rigidity exceed measured savings.

### CUDA Graphs

After fixed capacities and stable tensor addresses are in place, capture:

```text
project_count
CUB scan
emit_pairs
CUB sort
encode_tile_ranges
forward
backward
project_backward
optional optimizer-side reductions
```

Use graph node updates for pointers/scalars when topology is stable. Capacity
growth, different sort temp storage, or different optional kernels may require
recapture. Graphs reduce launch overhead; they do not solve pair traffic.

## Memory Layout

Training inputs/projected state:

```text
means3d:       float3 or SoA x/y/z
cov/quatscale: chosen reference-compatible representation
colors:        float3 for training, half only as eval ablation
opacities:     float
means2d:       float2 [B,C,N]
conics:        float3 [B,C,N]
depths:        float [B,C,N]
radii/support: int or float support metadata
```

Intersection buffers:

```text
tiles_per_gauss: int32 [B*C*N]
pair_offsets:    int32 [B*C*N + 1]
isect_keys:      uint64 [pair_capacity]
gaussian_ids:    int32 [pair_capacity]
sort double buffers for keys and ids
tile_offsets:    int32 [B*C*num_tiles + 1]
```

Output/state options:

```text
render_rgb:    float32 [B,C,H,W,3]
render_alpha:  float32 [B,C,H,W] or final_T equivalent
last_ids:      int32 [B,C,H,W]
tile_stop:     int32 [B,C,num_tiles] optional V8-style compact state
```

Default first implementation should keep `render_alpha + last_ids` because it
matches common CUDA rasterizer contracts and simplifies backward. Then add a
`tile_stop` recompute ablation to reduce memory.

## Bandwidth And Capacity Estimates

Use decimal MB for transfer intuition and note MiB where useful.

### 1080p, B=1

```text
pixels = 1920 * 1080 = 2,073,600
tiles  = ceil(1920/16) * ceil(1080/16) = 120 * 68 = 8,160

RGB fp32 output        = 24,883,200 B  = 24.9 MB
RGBA fp32 output       = 33,177,600 B  = 33.2 MB
alpha/final_T fp32     =  8,294,400 B  =  8.3 MB
last_id int32          =  8,294,400 B  =  8.3 MB
alpha + last_id state  = 16,588,800 B  = 16.6 MB
tile_stop int32        =     32,640 B  =  0.03 MB
tile_offsets int32     =     32,644 B  =  0.03 MB
```

For a 6K-splat 1080p scene, a reasonable first fixed pair capacity is
`131,072` references:

```text
pair_capacity          = 131,072
raw key+id bytes       = pair_capacity * (8 + 4) = 1.57 MB
double-buffer key+id   = about 3.15 MB
CUB temp budget        = profile-dependent, provision 4-16 MB initially
```

If real scenes exceed this, grow capacity outside the measured step and log the
overflow distribution.

### 4096x4096, B=1, 64K Splats

```text
pixels = 4096 * 4096 = 16,777,216
tiles  = 256 * 256 = 65,536

RGB fp32 output        = 201,326,592 B = 201.3 MB = 192 MiB
RGBA fp32 output       = 268,435,456 B = 268.4 MB = 256 MiB
alpha/final_T fp32     =  67,108,864 B =  67.1 MB =  64 MiB
last_id int32          =  67,108,864 B =  67.1 MB =  64 MiB
alpha + last_id state  = 134,217,728 B = 134.2 MB = 128 MiB
tile_stop int32        =     262,144 B =   0.3 MB
tile_offsets int32     =     262,148 B =   0.3 MB
```

Local V8/V7.4 notes reported a 4K/64K uniform scene with:

```text
tile_refs      = 642,580
max_refs/tile  = 27
mean_refs/tile = 9.80
```

First 4K/64K capacity:

```text
pair_capacity          = 1,048,576
raw key+id bytes       = 12.58 MB
double-buffer key+id   = 25.17 MB
CUB temp budget        = profile-dependent, provision 32-96 MB initially
binned id state actual = 642,580 * 4 = 2.57 MB
```

The state cliff is not the tile pair list. It is full-frame per-pixel saved
state. `alpha + last_id` costs 128 MiB at 4K B=1; `tile_stop` costs only about
0.5 MB including offsets, but requires backward recompute and must be proven.

### Traffic Implications

Forward minimum stores at 4K B=1:

```text
RGB output only                  ~= 201 MB
RGB + alpha + last_id            ~= 335 MB
RGBA output + alpha + last_id    ~= 403 MB
```

Sort traffic is several passes over pair buffers. At 1M capacity, even one full
read+write over double-buffered keys/ids is roughly 50 MB; radix sort can touch
that multiple times depending on bit range. Tight tile support and bit-range
sorting matter.

Backward traffic is dominated by replay reads and gradient atomics in heavy
overlap scenes. The default mitigation is block-level reduction before global
atomics, not larger saved histories.

## Atomics Strategy

### Default: Tile/Block Reduction Before Global Atomics

For each splat reference in a tile, 256 pixel threads compute local gradient
partials. Reduce inside the block, then issue one global atomic per gradient
component for the tile/splat pair:

```text
mean2d:   2 atomicAdd
conic:    3 atomicAdd
color:    3 atomicAdd
opacity:  1 atomicAdd
total:    9 fp32 atomics per splat/tile
```

This preserves the V6/V8 advantage:

```text
bad:  pixel * splat * component global atomics
good: tile * splat * component global atomics after 256-lane reduction
```

### Warp-Aggregated Alternative

Use warp reductions first:

```text
warp = tiled_partition<32>(block)
partial = warp_reduce(local_grad)
lane0 atomicAdd(global_grad, partial)
```

This is simpler and may beat full-block reduction if the extra block
synchronization costs more than the saved atomics. It should be the baseline
comparison.

### Full-Block Reduction Alternative

Eight warp leaders write partials to shared memory. The first warp reduces the
eight partials and one lane issues atomics:

```text
if lane == 0:
  sh_partials[warp_id][component] = warp_sum
block.sync()
if warp_id == 0:
  block_sum = reduce 8 warp partials
  if lane == 0:
    atomicAdd(...)
block.sync()
```

This is the likely best path for clustered hot tiles where many pixels touch the
same splat.

### Deferred Partial Reduction

When global atomics dominate, write partials instead:

```text
partial_keys:  gaussian_id [num_tile_splat_refs]
partial_vals:  9 floats    [num_tile_splat_refs]
CUB sort/group by gaussian_id
reduce_by_key into final gradients
```

Use only if Nsight shows atomic stalls dominate. It adds substantial memory
traffic:

```text
partial value bytes = refs * 9 * 4
4K measured refs 642,580 -> about 23.1 MB values + keys
1M capacity -> about 36 MB values + keys
```

Deferred reduction is attractive for adversarial clustered scenes, but it can
lose on normal scenes.

## Pseudocode

### Kernel 1: `project_count_fused`

```cuda
kernel project_count_fused(
    means3d, cov_or_quat_scale, opacity, camera,
    means2d, conics, depths, radii, tiles_per_gauss, flags,
    H, W, tile_size):

  g = global_thread_id
  if g >= B*C*N:
    return

  mu_c = transform_point(camera[g.image], means3d[g])
  if mu_c.z <= near_plane:
    tiles_per_gauss[g] = 0
    flags[g] = CLIPPED
    return

  Sigma_c = transform_covariance(camera[g.image], cov_or_quat_scale[g])
  J = projection_jacobian(mu_c, camera[g.image].intrinsics)
  Sigma_2d = J * Sigma_c * J^T + eps_2d * I
  conic = inverse_2x2(Sigma_2d)

  mean2d = project(mu_c)
  opacity_eff = opacity[g] * compensation_if_enabled(Sigma_2d)
  tau = -2 * log(alpha_threshold / max(opacity_eff, eps))
  if tau <= 0:
    tiles_per_gauss[g] = 0
    return

  bounds = conservative_tile_bounds(mean2d, conic, tau, H, W, tile_size)
  count = 0
  for ty in bounds.y0..bounds.y1:
    for tx in bounds.x0..bounds.x1:
      if ellipse_intersects_tile(mean2d, conic, tau, tx, ty, tile_size):
        count += 1

  means2d[g] = mean2d
  conics[g] = conic
  depths[g] = mu_c.z
  radii[g] = bounds_or_radius
  tiles_per_gauss[g] = count
```

### Stage 2: CUB Scan

```cpp
cub::DeviceScan::ExclusiveSum(
    temp_storage,
    temp_bytes,
    tiles_per_gauss,
    pair_offsets,
    B*C*N + 1,
    stream);
```

Implementation detail: write `pair_offsets[B*C*N]` or `total_pairs_device`
without synchronizing to CPU. A small device-side overflow kernel checks:

```text
total_pairs_device <= pair_capacity
```

If false, set a device flag and fail closed or trigger an out-of-band grow/retry
path.

### Kernel 3: `emit_pairs`

```cuda
kernel emit_pairs(
    means2d, conics, depths, opacity, pair_offsets,
    isect_keys, gaussian_ids, overflow_flag,
    H, W, tile_size, pair_capacity):

  g = global_thread_id
  if g >= B*C*N:
    return

  start = pair_offsets[g]
  end = pair_offsets[g + 1]
  if end > pair_capacity:
    overflow_flag = 1
    return

  local = 0
  bounds = recompute_or_load_bounds(g)
  for ty in bounds.y0..bounds.y1:
    for tx in bounds.x0..bounds.x1:
      if ellipse_intersects_tile(...):
        tile_id = image_tile_base(g.image) + ty * tiles_x + tx
        depth_key = depth_to_sort_key(depths[g])
        key = pack_key(g.image, tile_id, depth_key)
        isect_keys[start + local] = key
        gaussian_ids[start + local] = g
        local += 1
```

No atomic is needed because the scan gives each Gaussian a private output
range.

### Stage 4: CUB Sort

```cpp
cub::DeviceRadixSort::SortPairs(
    temp_storage,
    temp_bytes,
    keys_in,
    keys_out,
    ids_in,
    ids_out,
    total_pairs_or_capacity_for_graph,
    begin_bit,
    end_bit,
    stream);
```

For no-hot-sync steady state, either:

- sort the active `total_pairs` using device-accessible count if wrapper/API
  permits without CPU sync; or
- sort a fixed high-watermark capacity and kill it if wasted traffic is too
  high.

The preferred implementation is a fixed-capacity wrapper with a cached
high-watermark and no CPU read in normal training. The slow grow/retry path can
synchronize outside the measured step.

### Kernel 5: `encode_tile_ranges`

```cuda
kernel encode_tile_ranges(sorted_keys, total_pairs, tile_offsets, num_tiles):

  i = global_thread_id

  if i == 0:
    fill tile_offsets with invalid sentinel or use previous clear kernel

  if i >= total_pairs:
    return

  tile = unpack_tile(sorted_keys[i])
  prev_tile = (i == 0) ? INVALID : unpack_tile(sorted_keys[i - 1])
  next_tile = (i + 1 == total_pairs) ? INVALID : unpack_tile(sorted_keys[i + 1])

  if i == 0 or tile != prev_tile:
    tile_offsets[tile] = i
  if i + 1 == total_pairs or tile != next_tile:
    tile_offsets[tile + 1] = i + 1
```

Alternative: produce `ranges[tile] = {start,end}` to avoid sentinel fill cost.
Benchmark both on large empty-tile scenes.

### Kernel 6: `tile_forward_train`

```cuda
kernel tile_forward_train(
    sorted_ids, tile_offsets,
    means2d, conics, colors, opacities,
    output_rgb, output_alpha, last_ids_or_tile_stop,
    H, W, background):

  tile = blockIdx
  tid = threadIdx.x
  pixel = tile_pixel(tile, tid)
  valid_pixel = pixel inside image

  start = tile_offsets[tile]
  end = tile_offsets[tile + 1]

  C = float3(0)
  T = 1
  done = !valid_pixel
  last_ref = start - 1
  processed_count = 0

  for base in range(start, end, CHUNK):
    all_done = __syncthreads_count(done)
    if all_done == blockDim.x:
      break

    j = base + tid
    if j < end and tid < CHUNK:
      gid = sorted_ids[j]
      sh_ids[tid] = gid
      sh_mean[tid] = means2d[gid]
      sh_conic[tid] = conics[gid]
      sh_opacity[tid] = opacities[gid]
      sh_color[tid] = colors[gid]
    block.sync()

    chunk_n = min(CHUNK, end - base)
    for k in 0..chunk_n-1:
      if !done:
        alpha = eval_alpha(pixel, sh_mean[k], sh_conic[k], sh_opacity[k])
        if alpha >= alpha_threshold:
          C += T * alpha * sh_color[k]
          T_next = T * (1 - alpha)
          last_ref = base + k
          processed_count = base + k - start + 1
          T = T_next
          if T <= transmittance_threshold:
            done = true
    block.sync()

  if valid_pixel:
    output_rgb[pixel] = C + T * background
    output_alpha[pixel] = 1 - T
    last_ids[pixel] = last_ref

  tile_processed_max = block_reduce_max(processed_count)
  if tid == 0 and using_tile_stop:
    tile_stop[tile] = tile_processed_max
```

Keep eval and train variants separate:

- eval can skip saved state and optionally write half output;
- train writes fp32 accumulators and backward state.

### Kernel 7: `tile_backward_replay`

```cuda
kernel tile_backward_replay(
    sorted_ids, tile_offsets,
    means2d, conics, colors, opacities,
    grad_output, output_alpha_or_final_T, last_ids_or_tile_stop,
    grad_means2d, grad_conics, grad_colors, grad_opacities,
    H, W, background):

  tile = blockIdx
  tid = threadIdx.x
  pixel = tile_pixel(tile, tid)
  valid_pixel = pixel inside image

  start = tile_offsets[tile]
  end = tile_offsets[tile + 1]
  stop_end = choose_stop_end(start, end, last_ids[pixel], tile_stop[tile])

  // Recompute or recover T_final. With output_alpha: T_final = 1 - alpha.
  // With tile_stop only: replay forward to get per-pixel T_final and end_ref.
  if using_output_alpha:
    T_cur = 1 - output_alpha[pixel]
    end_ref = last_ids[pixel]
  else:
    replay forward over uniform tile_stop to compute T_cur and end_ref

  g = valid_pixel ? grad_output[pixel] : float3(0)
  gT_next = valid_pixel ? dot(g, background) : 0

  for base in reverse_chunks(start, stop_end, CHUNK):
    load chunk params into shared for all threads
    block.sync()

    for k in reverse(chunk_n-1..0):
      ref = base + k
      gid = sh_ids[k]

      local_grad = zero_grad_pack()
      participates = valid_pixel && ref <= end_ref

      if participates:
        alpha, raw, power, dx, dy = eval_alpha_full(pixel, sh_mean[k], ...)
        if alpha contributed:
          T_prev = T_cur / max(1 - alpha, eps)
          local_grad = reverse_step_grad(T_prev, gT_next, g, sh_color[k], ...)
          gT_next = alpha * dot(g, sh_color[k]) + (1 - alpha) * gT_next
          T_cur = T_prev

      // Reduce local_grad over 256 pixels for this gid.
      warp_sum = warp_reduce(local_grad)
      if lane == 0:
        sh_partials[warp_id] = warp_sum
      block.sync()

      if warp_id == 0:
        block_sum = reduce_8_warp_partials(sh_partials)
        if lane == 0:
          atomicAdd(grad_means2d[gid], block_sum.mean2d)
          atomicAdd(grad_conics[gid], block_sum.conic)
          atomicAdd(grad_colors[gid], block_sum.color)
          atomicAdd(grad_opacities[gid], block_sum.opacity)
      block.sync()
```

All barriers are tile-uniform. `participates` only gates arithmetic.

### Kernel 8: `project_backward`

```cuda
kernel project_backward(
    grad_means2d, grad_conics, grad_depths,
    means3d, cov_or_quat_scale, cameras,
    grad_means3d, grad_cov_or_quat_scale, grad_cameras):

  g = global_thread_id
  if g >= B*C*N:
    return

  Recompute mu_c, J, Sigma_c, Sigma_2d, inverse terms.
  Backprop conic inverse:
    dSigma_2d = -Q^T * dQ * Q^T
  Backprop regularization/compensation if enabled.
  Backprop Sigma_2d = J * Sigma_c * J^T.
  Backprop J and projection mean into mu_c.
  Backprop camera transform into mean3d and optional camera params.
  Backprop covariance into cov/quaternion/scale representation.
```

Start by matching a reference implementation. Optimize this kernel only after
the raster forward/backward stage is correct and profiled.

### Optional Kernel 9: `deferred_partial_reduce`

```cuda
tile_backward_partial:
  write partial_keys[m] = gid
  write partial_vals[m] = reduced 9-float gradient

CUB sort partial_keys/partial_vals by gid

reduce_partials_by_key:
  one block per run or segmented block reduction
  write final gradients
```

Kill this path if partial-buffer bandwidth exceeds atomic-stall savings.

## Accuracy Risks

- Depth ordering must match the reference, including equal-depth tie-breaks.
- Alpha clamp and alpha threshold gates must match forward and backward.
- Antialiasing compensation must be explicit and validated.
- Early stop must not alter the mathematical function beyond the chosen
  threshold tolerance.
- Per-pixel `last_id` and tile-level `tile_stop` are not equivalent unless
  backward recomputes per-pixel end refs correctly.
- Half output is eval-only until gradient tolerances prove otherwise.
- Projection backward is easy to mismatch because covariance conventions differ
  across V8, gsplat, and Graphdeco.

## Validation Plan

Correctness references:

- V8 projected-input path for local math parity.
- `gsplat` CUDA for full 3D projection/raster parity.
- Graphdeco diff-gaussian-rasterization for architecture and simple reference
  behavior.
- Torch autograd finite differences on tiny scenes where feasible.

Required tests:

```text
single splat, black background
two overlapping splats with known order
equal-depth stable tie-break
alpha below threshold
alpha clamp at max_alpha
near-plane cull
large covariance tile support
small covariance one-pixel support
early stop with high opacity stack
batch/image indexing
capacity overflow fail-closed
```

Metrics:

```text
image max_abs, mean_abs
grad mean2d max_abs, mean_abs
grad conic/cov max_abs, mean_abs
grad color max_abs, mean_abs
grad opacity max_abs, mean_abs
grad 3D mean/cov/quaternion/scale where enabled
```

Initial tolerances should be conservative:

```text
fp32 forward image max_abs <= reference tolerance already accepted by gsplat/V8
fp32 gradient max_abs close to existing V8 small-case tolerances
half eval output separately bounded by image-quality threshold
```

Do not relax tolerances to hide ordering mistakes. Multi-splat errors like the
Metal fixed-eval `~0.124` case are semantic failures, not acceptable numeric
drift.

## Benchmark Plan

Compare against:

- current local V8 where available;
- upstream `gsplat`;
- Graphdeco CUDA rasterizer;
- naive Torch only for small cases.

Stage timings:

```text
project_count_fused
CUB scan
emit_pairs
CUB sort
encode_tile_ranges
forward raster
backward raster
project_backward
deferred reduction if enabled
total forward
total forward+backward
```

Scene matrix:

```text
512x512 / 6K / B=1,4
1080x1920 / 6K / B=1,4
4096x4096 / 64K / B=1
4096x4096 / 64K / B=4 where memory allows

microbench_uniform_random
sparse_screen
clustered_hot_tiles
layered_depth
overflow_adversarial
real_trace if available
```

Counters:

```text
total_pairs
pair_capacity
capacity_overflow
active_tile_count
tile refs p50/p90/p95/p99/max
mean and p95 stop ratio
number of splat/tile gradient reductions
estimated global atomics
actual global atomic stall counters from Nsight
shared bytes per block
registers per thread
occupancy
CUB temp bytes
allocated/reserved CUDA memory
```

Promotion speed gates should be relative, not absolute:

- first baseline must be within 10% of upstream `gsplat` on forward+backward
  before custom optimizations are judged;
- no-hot-sync capacity path should beat a CPU-sized allocation path in steady
  training loops;
- block-reduced backward must beat warp-only atomics on clustered hot tiles
  without regressing uniform 4K by more than 3%;
- `tile_stop` recompute must save meaningful memory and not regress
  forward+backward by more than 5% on normal scenes;
- CUDA Graph capture must reduce launch overhead on small/medium cases without
  making capacity changes fragile.

## Milestones

### Milestone 0: Baseline Measurement

Run upstream/reference CUDA paths on the exact matrix. Record split timings,
pair counts, state bytes, and gradient tolerances.

Gate:

```text
reference numbers are reproducible
test harness can compare images and gradients
Nsight profile workflow exists
```

Kill criteria:

```text
no reliable CUDA test machine
no ability to compare against gsplat/Graphdeco outputs
```

### Milestone 1: Fused Projection + Count

Implement `project_count_fused` and fixed-capacity overflow flags.

Gate:

```text
projected means/conics/depths match reference
tile counts match reference or are conservatively larger with measured overhead
no CPU shape read in steady path
```

Kill criteria:

```text
tile count inflation causes >20% pair growth on normal scenes
capacity overflow is silently ignored
projection backward cannot be matched within tolerance
```

### Milestone 2: Pair Emit + CUB Sort + Ranges

Build sorted tile lists with CUB.

Gate:

```text
sorted tile ranges match reference order
pair capacity handles 1080p/6K and 4K/64K target cases
sort bit ranges are minimized
```

Kill criteria:

```text
custom sort replaces CUB without a measured reason
fixed-capacity sort over mostly empty buffers dominates runtime
equal-depth ordering remains nondeterministic
```

### Milestone 3: Accurate Forward

Implement train/eval tile forward.

Gate:

```text
single and multi-splat image parity passes
early stop matches reference threshold behavior
forward split timings are competitive with reference CUDA
```

Kill criteria:

```text
multi-splat errors indicate ordering/state mismatch
barrier divergence appears under early stop
state stores exceed planned memory without a clear win
```

### Milestone 4: Backward Replay

Implement reverse replay with warp and block reduction variants.

Gate:

```text
gradients match reference on all small tests
forward+backward competitive with gsplat/Graphdeco
global atomics reduced to tile/splat/component level
```

Kill criteria:

```text
per-pixel global atomics are required for speed or simplicity
register pressure cuts occupancy enough to lose badly
gradient mismatch comes from saved-state ambiguity
```

### Milestone 5: Memory/Launch Optimizations

Test `tile_stop`, CUDA Graphs, and `cp.async` separately.

Gate:

```text
each optimization has isolated before/after numbers
no optimization changes math outputs
fallback path remains simple and correct
```

Kill criteria:

```text
cp.async/TMA adds complexity without profiler-backed load savings
CUDA Graph capture cannot handle capacity updates cleanly
tile_stop recompute loses more time than memory it saves
```

### Milestone 6: Heavy/Clustered Ablations

Only after baseline is strong, test heavy-tile segmentation and deferred
partials.

Gate:

```text
wins on overflow_adversarial or clustered_hot_tiles
disabled for normal uniform/light scenes
allocates by actual heavy count, not capacity
```

Kill criteria:

```text
heavy buffers allocate when heavy count is zero
launches over capacity instead of compacted count
partial reduction traffic exceeds atomic serialization savings
```

## Optional Vulkan Hardware-Raster Branch

This branch is not Direction C's first implementation. It is a later
best-of-both-worlds experiment.

Viable architecture:

```text
CUDA:
  project/count/sort or prepare draw stream

Vulkan:
  vertex/mesh shader expands splats
  fragment shader evaluates alpha
  fragment interlock or ROAA provides ordered programmable same-pixel state
  external-memory images/buffers hold RGB, T/final_T, stop metadata

CUDA:
  imports external memory
  waits/signals external semaphores
  runs backward replay or projection backward
```

Required primitives:

- `VK_EXT_fragment_shader_interlock` for fragment critical sections;
- `VK_EXT_rasterization_order_attachment_access` plus local read where
  available;
- Vulkan external memory imported with CUDA external-memory APIs;
- CUDA external semaphores for GPU-side synchronization.

Why it is not first:

- it is cross-API, not CUDA-only;
- graphics primitive order and interlock behavior must still match the training
  depth order;
- interlock/ROAA can serialize exactly the hot pixels where splats overlap;
- backward still needs compact state and gradient reductions;
- debugging synchronization and image layout issues can consume the whole
  experiment before a compute baseline exists.

When to start it:

```text
compute V9 forward+backward has baseline numbers
direct external-memory smoke writes a Torch-visible CUDA buffer/image
Vulkan forward can produce exact C/T/stop on tiny multi-splat tests
```

Kill it if:

```text
forward is not at least 25% faster than compute on parity-shaped 4K cases
state transfer/cross-API synchronization eats the forward win
training gradients require recomputing all useful state in CUDA anyway
```

## Strongest Hypothesis

The most credible CUDA V9 win is not hardware rasterization. It is a disciplined
compute rasterizer with:

```text
fused projection/count
fixed-capacity no-hot-sync pair buffers
CUB radix sort with tight opacity-aware tile support
one block per 16x16 tile
exact C/T front-to-back math
compact state
block-reduced backward atomics
CUDA Graph replay
```

The weakest part is the same dependency every exact Gaussian rasterizer has:
per-pixel alpha compositing is order-dependent and serial along the splat list.
CUDA can reduce overhead around that dependency, but it cannot make the
recurrence associative per splat without changing the math. Heavy-tile
segmentation and deferred reductions are therefore ablations, not defaults.
