# V9 CUDA Hardware Rasterization Notes

Date: 2026-04-25

Scope: research and architecture notes only. No Metal variants or Python
packages were modified.

## Executive Position

The best CUDA path is not a direct port of the Metal hardware-raster attempt.
CUDA does not expose NVIDIA fixed-function rasterization units as CUDA kernels.
If we want true hardware rasterization, the practical route is a Vulkan or
Direct3D graphics pipeline plus CUDA external-memory/semaphore interop. That can
be valuable for eval and for a separate differentiable hardware-raster branch,
but it is not the first CUDA training kernel to build.

The strongest CUDA training path is a tile-based CUDA compute rasterizer that
keeps the V8/V9 lessons:

```text
fused 3D projection -> opacity-aware tile intersection -> CUB scan/sort
  -> one CUDA block per 16x16 tile
  -> direct writes to torch CUDA tensors
  -> forward saves compact alpha/order state
  -> backward replays per-pixel ordered compositing
  -> warp/block reductions before global atomics
```

CUDA gives us stronger primitives than Metal for this compute route:

- CUB device scans and radix sorts instead of custom scan/sort glue.
- `cooperative_groups` and warp intrinsics for clean warp/block reductions.
- Direct torch CUDA tensor writes with no texture row-alignment problem.
- `cp.async` on SM80+ and TMA/cluster DSM on SM90+ as later load/staging
  ablations.
- CUDA Graphs for repeated fixed-shape training loops.

The optional graphics route should be a separate branch:

```text
CUDA or Vulkan bin/sort -> Vulkan mesh/vertex rasterization
  -> fragment shader interlock or rasterization-order attachment access
  -> programmable per-pixel blending/state
  -> CUDA external memory import/export for torch-visible tensors
```

That route may be real on RTX hardware, but it is cross-API and order-sensitive.
It should not block the CUDA compute baseline.

## Existing Evidence To Carry Over

Local Mac evidence says:

- V6/V8-style compute training is still the best known 4K training base.
- Hardware forward can be fast only when output stays GPU-resident.
- Hardware forward does not fundamentally break backward, but fixed-function
  blending gives only final color. Training needs processed order, stop point,
  and gradient accumulation by Gaussian.
- Per-pixel front-K/history explodes at 4K.
- Tile-level reductions before global atomics are essential.
- Heavy-tile segmentation is useful math, not a default implementation.
- ICB/indirect render command execution was crash-prone on Metal and should not
  be treated as a portability assumption.

CUDA reference code confirms the same shape:

- Graphdeco `diff-gaussian-rasterization` uses projection/preprocess,
  tile/depth key emission, CUB radix sort, one block per tile, and reverse
  backward with atomics.
- `gsplat` uses fused projection, opacity-aware tile intersection, CUB radix
  sort or segmented radix sort, one block per tile, shared batches, output alpha,
  `last_ids`, and warp reductions before atomics.
- `gsplat` still has a CPU shape read in its wrapper for `n_isects =
  cum_tiles_per_gauss[-1].item<int64_t>()`; a "best V9 CUDA" should remove that
  from the hot path with fixed or cached capacity.

## Metal Idea To CUDA Mapping

| Metal V9 idea | CUDA replacement | Assessment |
|---|---|---|
| Direct MPS render target over tensor storage | Direct CUDA kernel writes to torch CUDA tensor data pointers | Easier on CUDA. No renderable texture format or 256-byte row-alignment gate for the compute path. |
| Tile/imageblock per-pixel state | `__shared__` memory per CUDA block, optionally dynamic shared memory | Use for splat parameter chunks and reductions. Do not store unbounded per-pixel histories. |
| Raster order groups | Sorted tile lists in CUDA compute; Vulkan `VK_EXT_fragment_shader_interlock` or `VK_EXT_rasterization_order_attachment_access` for graphics route | CUDA compute should avoid per-pixel ordering locks. Graphics route can use fragment interlock/ROAA, but sorting still matters. |
| ICB / indirect draws | CUDA Graphs for repeated kernels, persistent work queues for dynamic tiles, Vulkan indirect draw only in graphics branch | CUDA Graphs reduce launch overhead; they do not solve sort/bin memory traffic. Dynamic parallelism is risky for hot path. |
| Output format sweep | CUDA output planes: RGB fp32, alpha/final_T, optional half output for eval | CUDA can write exactly the tensor layout we want. Use half only after error checks. |
| Hardware forward plus compute backward | CUDA compute forward/backward first; optional Vulkan forward with CUDA backward later | Compute baseline is much easier to validate. |

## Math Contract

The CUDA kernel must keep the V8/V9 compositing contract.

For pixel center `p = (x + 0.5, y + 0.5)`:

```text
d_i      = p - mean2d_i
q_i      = a_i*dx_i^2 + 2*b_i*dx_i*dy_i + c_i*dy_i^2
power_i  = -0.5 * q_i
raw_i    = opacity_i * exp(power_i)
alpha_i  = min(max_alpha, raw_i)
visible  = q_i >= 0 and alpha_i >= alpha_threshold

C_0 = 0
T_0 = 1
C_{i+1} = C_i + T_i * alpha_i * color_i
T_{i+1} = T_i * (1 - alpha_i)
stop when T <= transmittance_threshold
out = C_M + T_M * background
```

Backward can use the scalar suffix-dot recurrence used by V8:

```text
T_cur = T_final
gT = dot(grad_out, background)

for i in reverse(processed_prefix):
  denom   = max(1 - alpha_i, eps)
  T_prev  = T_cur / denom
  dot_c   = dot(grad_out, color_i)
  g_alpha = T_prev * (dot_c - gT)
  g_color = grad_out * (T_prev * alpha_i)

  gate    = 1 if raw_i < max_alpha else 0
  g_raw   = g_alpha * gate
  g_power = g_raw * raw_i

  grad_conic_a += g_power * (-0.5) * dx^2
  grad_conic_b += g_power * (-1.0) * dx * dy
  grad_conic_c += g_power * (-0.5) * dy^2
  grad_mean_x  += g_power * (a*dx + b*dy)
  grad_mean_y  += g_power * (b*dx + c*dy)
  grad_opacity += g_raw * raw_i / max(opacity, eps)

  gT = alpha_i * dot_c + (1 - alpha_i) * gT
  T_cur = T_prev
```

Important: CUDA warp/block barriers must remain uniform. Per-pixel `end_i` or
`last_id` can gate arithmetic, not barrier participation.

## Recommended CUDA Architecture

### Stage 0: Reference Baseline

Start from `gsplat`-style CUDA, not from scratch:

```text
projection_ewa_3dgs_fused
intersect_tile with AccuTile/SNUGBOX support
CUB DeviceScan over tiles_per_gauss
CUB DeviceRadixSort or DeviceSegmentedRadixSort over isect_ids
rasterize_to_pixels_3dgs_fwd
rasterize_to_pixels_3dgs_bwd
```

Baseline target:

- match `gsplat` or Graphdeco image/gradient tolerance;
- get timing splits for projection, intersect, scan, sort, offset encode,
  forward raster, backward raster, projection backward;
- log `n_isects`, p50/p90/p99 tile refs, max tile refs, stop ratio, and atomics
  estimate.

### Stage 1: No Hot-Path CPU Shape Read

The first serious CUDA improvement should remove per-iteration CPU shape
discovery:

```text
input fixed capacities:
  pair_capacity
  tile_capacity
  heavy_tile_capacity

kernel project_count:
  writes tiles_per_gauss and overflow flags

CUB scan:
  writes offsets into fixed buffers

kernel emit:
  if total_pairs > pair_capacity:
    set device overflow flag and return
  else:
    emit isect_ids and flatten_ids
```

Capacity failure should be a slow grow/retry outside the benchmarked training
step. In steady-state training, do not call `.item()` just to allocate the next
buffer.

### Stage 2: Tile Forward With Compact State

One block per tile, one thread per pixel:

```text
threads = dim3(16, 16, 1)
grid    = dim3(num_images, tile_height, tile_width)

shared:
  id_batch[256]
  mean_opacity_batch[256]
  conic_batch[256]
  optional color_batch[256 * C]

for each tile:
  range = tile_offsets[tile_id]..tile_offsets[tile_id + 1]
  T = 1
  C = 0
  last_id = -1

  for batch of 256 sorted refs:
    if __syncthreads_count(done) == 256: break
    load one splat per thread into shared
    __syncthreads()
    for splat in batch:
      if not done:
        alpha = eval_alpha(pixel, splat)
        next_T = T * (1 - alpha)
        if visible and next_T > threshold:
          C += T * alpha * color
          T = next_T
          last_id = global_ref_index
        else if next_T <= threshold:
          done = true
          break

  write render_rgb[pixel]
  write render_alpha[pixel] = 1 - T
  write last_ids[pixel] = last_id
```

CUDA should store per-pixel `render_alpha` and `last_ids` initially, matching
the established CUDA ecosystem. A leaner V8-style `tile_stop_counts` variant is
an ablation:

- `render_alpha + last_ids`: larger state, simpler backward, standard gsplat.
- `tile_stop_counts` only: smaller state, recomputes per-pixel final T in
  backward, closer to Mac V8.

At 4096x4096:

```text
RGB fp32 output      = 201,326,592 bytes
RGBA fp32 output     = 268,435,456 bytes
one fp32 alpha plane =  67,108,864 bytes
one i32 last_id      =  67,108,864 bytes
B=4 RGB fp32 output  = 805,306,368 bytes
```

### Stage 3: Backward With Block-Level Reduction

The key upgrade over naive CUDA rasterizers is reducing across the whole 16x16
tile before global atomics where possible.

Baseline `gsplat` reduces at warp level and issues one atomic per warp/key. A
V9 CUDA branch should test a full-block reduction for each splat:

```text
for tile:
  load sorted range
  for pixel:
    T_final = 1 - alpha[pixel]
    end_ref = last_ids[pixel]

  for batches from back to front:
    load splat params/colors into shared
    for splat in reverse(batch):
      local_grad = 0
      if pixel participates:
        compute reverse recurrence local partials

      warp_reduce local_grad
      write one partial per warp to shared
      __syncthreads()
      first warp reduces warp partials
      lane 0 atomicAdd global grads
      __syncthreads()
```

Atomic policy:

```text
default:
  9 float atomics per splat/tile after block reduction

fallback:
  warp-reduced atomics if block reduction costs too much synchronization

ablation:
  deferred partial buffer [tile_ref, gaussian_id, 9 floats]
  CUB sort/reduce by gaussian_id
```

Deferred partials should be used only if profiling shows global atomics dominate
clustered/heavy scenes. It can create more memory traffic than it saves.

### Stage 4: Fused Projection To Tile Intersection

`gsplat` already has fused projection kernels. The next step is not fusing all
the way to final raster in one mega-kernel because scan/sort separates stages.
The practical fusion is:

```text
projection_count_kernel:
  transform mean/covariance/quat-scale
  compute conic/radius/depth/compensation
  compute opacity-aware tile count
  write projected params and tiles_per_gauss
```

Then:

```text
CUB scan
emit_intersections_kernel
CUB sort
offset_encode_kernel
raster kernels
```

This saves a read/write pass if the current pipeline projects first and counts
later. Keep the projected tensors if backward projection needs them.

### Stage 5: Heavy Tile Segmentation

Use only for actual heavy tiles:

```text
heavy_tiles = compact(tile_ref_count > heavy_threshold)
for each heavy tile:
  split sorted refs into chunks
  per pixel, per chunk:
    D = (C_seg, T_seg)

segment combine:
  combine(A, B).C = A.C + A.T * B.C
  combine(A, B).T = A.T * B.T
```

Backward can use scalar suffix-dot:

```text
H_before_chunk = dot(grad_out, C_seg) + T_seg * H_after_chunk
```

Risk: early stop depends on incoming `T`, so segment descriptors alone are not
enough unless the exact stopped prefix is carried or recomputed. Do not enable
this for normal light tiles.

## Optional Graphics/Hardware Raster Branch

Pure CUDA does not launch fixed-function rasterizer work. The viable NVIDIA
hardware raster branch is Vulkan/Direct3D plus CUDA interop:

```text
CUDA:
  project/count/sort or prepare draw stream

Vulkan:
  mesh shader or vertex shader expands splats
  fragment shader evaluates alpha
  fragment interlock or ROAA protects per-pixel programmable blend
  output image and state images live in external memory

CUDA:
  imports output/state memory
  backward uses compute tile replay or graphics-produced state
```

Required graphics features:

- `VK_EXT_fragment_shader_interlock`: per-pixel critical sections for shader
  loads/stores and programmable blending.
- `VK_EXT_rasterization_order_attachment_access` plus
  `VK_KHR_dynamic_rendering_local_read`: framebuffer fetch with guaranteed
  rasterization-order attachment access.
- Vulkan external memory and semaphores imported into CUDA with
  `cudaImportExternalMemory`, `cudaExternalMemoryGetMappedBuffer`,
  `cudaImportExternalSemaphore`, `cudaWaitExternalSemaphoresAsync`, and
  `cudaSignalExternalSemaphoresAsync`.

Why this is not first:

- It is not a CUDA-only kernel.
- It needs graphics-device setup and cross-API synchronization.
- Correct blending still requires sorted primitive submission.
- Fragment interlock/ROAA serializes hot pixels.
- Training still needs backward state and gradient reduction.

Why it remains interesting:

- NVIDIA's Vulkan Gaussian splatting sample already explores mesh/vertex shader
  splat rendering and GPU radix sort.
- Recent hardware-raster 3DGS papers report meaningful full-pipeline wins with
  programmable blending and half formats.
- This could be a best-of-both-worlds eval path or a future training branch if
  state capture stays compact.

## Bottlenecks And Failure Modes

| Bottleneck | Why it matters | Mitigation |
|---|---|---|
| Sort/bin memory traffic | Tile intersections can dominate at high splat counts. | Opacity-aware tight tile intersection; segmented sort when image segments are independent; fixed/cached capacity. |
| CPU shape sync | `.item()` on total intersections stalls the stream. | Fixed capacity or high-watermark buffers; slow grow/retry path. |
| Global atomics in backward | Many tiles can update the same Gaussian gradients. | Warp/block reduction before atomics; profile deferred partials only for hot scenes. |
| Shared-memory pressure | Large tile size or many color channels lowers occupancy. | Keep 16x16 default; specialize channel counts; use dynamic shared memory with occupancy checks. |
| Register pressure | Backward recurrence stores color, conic, T, suffix state, many grads. | Split channel specializations; use `__launch_bounds__`; inspect ptxas register report. |
| Alpha compositing dependency | Per-pixel splat order is serial. | Use early stop; chunking; heavy segmentation only for heavy tiles. |
| Per-pixel state blowup | `alpha + last_id` costs 134 MB at 4K B=1. | Benchmark `tile_stop_counts` recompute variant. |
| Output bandwidth | RGB fp32 is 201 MB at 4K B=1 before state. | Eval half output only after error checks; training likely keeps fp32 accumulators. |
| Dynamic parallelism | Device-side launches carry runtime overhead and constraints. | Prefer compacted work queues or CUDA Graphs. |
| Vulkan interop | Cross-API synchronization bugs are hard to debug. | Keep it separate; build minimal import/export smoke before 3DGS. |

## Concrete CUDA V9 Plan

1. **Baseline import study.**
   Benchmark `gsplat` and Graphdeco-style CUDA on identical synthetic cases:
   512/6K, 1080p/6K, 4K/64K, B=1/B=4 when memory allows.

2. **Fixed-capacity intersection buffers.**
   Implement a no-hot-sync wrapper with device flags for capacity overflow.
   Compare against `gsplat`'s CPU-sized `n_isects` path.

3. **Projection-count fusion.**
   Fuse projection and tile-count for 3DGS EWA projection. Keep projected
   outputs needed by backward.

4. **Block-reduced backward.**
   Replace warp-only gradient atomics with block-level reduction per splat.
   Benchmark against `gsplat` on uniform and clustered scenes.

5. **Tile-stop-count ablation.**
   Store `tile_stop_counts` instead of `last_ids + alpha` and recompute final T
   per pixel. This ports the Mac V8 state idea to CUDA and can save 134 MB at
   4K B=1.

6. **Async load ablation.**
   Add SM80 `cp.async` / `cuda::memcpy_async` staging for splat parameter
   batches. Keep a normal shared-load kernel as baseline.

7. **Heavy-tile segmentation ablation.**
   Compact actual heavy tiles, allocate buffers from actual heavy count, and
   test scalar suffix-dot backward. Keep disabled by default.

8. **CUDA Graph capture.**
   For fixed shapes/capacities, capture projection, scan, emit, sort, offsets,
   forward, backward, and optimizer-side reductions into one reusable graph.

9. **Optional Vulkan hardware branch.**
   Only after compute baseline is quantified, build a minimal Vulkan/CUDA
   external-memory smoke: render programmable-blend splats into an imported
   CUDA buffer and verify no CPU copy. Then test fragment interlock/ROAA.

## Validation And Benchmark Matrix

Correctness:

```text
image max_abs / mean_abs vs reference
grad max_abs / mean_abs for mean2d/conic/color/opacity
depth-order stability with equal depths
alpha clamp gate
alpha threshold gate
early-stop parity
capacity overflow fail-closed behavior
```

Performance:

```text
projection ms
tile count ms
scan ms
emit ms
sort ms
offset encode ms
forward raster ms
backward raster ms
projection backward ms
total forward
total forward+backward
```

Scene cases:

```text
uniform random
sparse screen
clustered hot tiles
layered depth
overflow/heavy adversarial
real trace if available
```

Counters:

```text
n_isects
tile_count p50/p90/p99/max
active tile fraction
stop ratio
estimated backward atomics
state bytes
temporary sort bytes
CUDA memory allocated/reserved
occupancy, register count, shared bytes per block
Nsight Compute memory throughput and atomic stalls
```

## Open Questions

- Does block-level reduction beat `gsplat`'s warp-level atomic strategy on RTX
  4090/4080/H100, or does extra synchronization lose?
- Is `tile_stop_counts` recompute faster than storing `alpha + last_id` on CUDA,
  or does extra replay dominate?
- Does segmented radix sort beat global radix sort for the batch/image layouts
  we care about?
- Does `cp.async` help when each batch loads small splat structs with reuse
  across 256 pixels, or does the current shared load already hide latency?
- Can CUDA Graphs handle the dynamic capacity/grow path cleanly enough for
  training loops?
- Is a Vulkan programmable-blend branch worth maintaining if compute V9 already
  matches or beats it on forward+backward?

## Sources

- NVIDIA CUDA C++ Programming Guide:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Cooperative Groups:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
- CUDA Graphs:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- CUDA API interoperability:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html
- CUDA async copies and TMA:
  https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/async-copies.html
- PTX ISA:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CCCL/CUB:
  https://nvidia.github.io/cccl/cub/
- CUTLASS:
  https://docs.nvidia.com/cutlass/latest/overview.html
- gsplat docs:
  https://docs.gsplat.studio/main/apis/rasterization.html
- gsplat source:
  https://github.com/nerfstudio-project/gsplat
- Graphdeco differential Gaussian rasterization:
  https://github.com/graphdeco-inria/diff-gaussian-rasterization
- NVIDIA Vulkan Gaussian splatting sample note:
  https://developer.nvidia.com/blog/real-time-gpu-accelerated-gaussian-splatting-with-nvidia-designworks-sample-vk_gaussian_splatting/
- Khronos Vulkan ROAA sample:
  https://docs.vulkan.org/samples/latest/samples/extensions/rasterization_order_attachment_access/README.html
- `VK_EXT_fragment_shader_interlock`:
  https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_fragment_shader_interlock.html
- Efficient Differentiable Hardware Rasterization for 3D Gaussian Splatting:
  https://arxiv.org/abs/2505.18764
- FlashGS:
  https://arxiv.org/abs/2408.07967
