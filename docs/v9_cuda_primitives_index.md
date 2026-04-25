# V9 CUDA Primitives Index

Date: 2026-04-25

This is the low-level primitive catalog for a CUDA port of the V8/V9 gsplat
rasterizer work. It records feature names, architecture assumptions, sources,
and why each primitive matters.

## Architecture Assumptions

| Target tier | Minimum | What it unlocks | Use in this project |
|---|---:|---|---|
| Baseline CUDA training | SM70+ preferred | modern independent thread scheduling, half atomics on SM70+, common RTX support | Main CUDA compute rasterizer. |
| Ampere path | SM80+ | `cp.async`, hardware-accelerated barriers, strong CUDA Graph support | Async shared-memory staging ablation. |
| Hopper/Ada/Blackwell high-end path | SM90+ for Hopper features | thread block clusters, distributed shared memory, TMA, vector `float2/float4 atomicAdd` | Heavy-tile and bandwidth experiments only. |
| Graphics hardware-raster path | Vulkan extension/device dependent | fragment interlock, ROAA, mesh shaders, external memory/semaphores | Separate branch, not pure CUDA. |

## Primitive Map

| Primitive | Minimum / availability | Source | Why it matters | Risk |
|---|---:|---|---|---|
| `cooperative_groups::thread_block` | CUDA 9+ | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html | Clean block sync/reduce structure for one block per tile. | Low. |
| `cooperative_groups::tiled_partition<32>` | CUDA 9+ | same as above | Warp-sized subgroups for reductions and warp-level early-out checks. | Low. |
| `cooperative_groups::reduce` | CUDA 9+ | same as above | Existing `gsplat` uses it for warp sums before atomic adds. | Low. |
| `__shfl_down_sync`, `__shfl_xor_sync` | CUDA 9+ sync variants | https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/ | Fast register-level reductions without shared memory. | Must use correct masks after divergence. |
| `__ballot_sync`, `__any_sync`, `__all_sync` | CUDA 9+ sync variants | same as above | Warp alive masks, visibility masks, and compaction within a warp. | Mask bugs can silently corrupt reductions. |
| `__match_any_sync` | CUDA 9+ | same as above | Warp-aggregated atomics by Gaussian id or tile id. | Helps only when duplicate keys occur inside a warp. |
| `__syncwarp` | CUDA 9+ | same as above | Explicit warp barrier on Volta+ independent thread scheduling. | Do not assume implicit warp lockstep. |
| `__syncthreads_count(predicate)` | CUDA C++ | CUDA Programming Guide | Tile-wide early-out: if all 256 pixel lanes are done, stop batch loop. | Block-wide only; all live threads must participate. |
| CUB `DeviceScan::ExclusiveSum` / `InclusiveSum` | CCCL/CUB | https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceScan.html | Prefix sums for tile refs and compacted work queues. | Temp storage must be cached, not allocated every frame. |
| CUB `DeviceRadixSort::SortPairs` | CCCL/CUB | https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html | Sort `[image|tile|depth] -> gaussian_id` intersections. | O(N+P) temp in common API unless DoubleBuffer path is used. |
| CUB `DeviceSegmentedRadixSort::SortPairs` | CCCL, first listed for CUDA Toolkit 12.3 in docs | https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceSegmentedRadixSort.html | Sort each image/batch segment separately, avoiding high bits and possibly reducing work. | Newer API; verify toolkit availability on target. |
| CUB `BlockReduce` | CCCL/CUB | https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockReduce.html | Full 16x16 tile reduction before global gradient atomics. | Extra sync can lose vs warp reductions. |
| CUB `WarpReduce` | CCCL/CUB | https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpReduce.html | Drop-in warp reductions if cooperative groups codegen is worse. | Similar to manual shuffles. |
| CUB `BlockScan` | CCCL/CUB | https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockScan.html | Heavy-tile segment prefix/suffix scans inside a block. | Only useful for actual heavy tiles. |
| `atomicAdd(float*)` | CC 2.x+ | CUDA C++ Programming Guide atomics | Final gradient accumulation. | Contention can dominate clustered scenes. |
| `atomicAdd(double*)` | CC 6.x+ | CUDA C++ Programming Guide atomics | Debug/high precision only. | Slower, usually not default. |
| `atomicAdd(__half*)` | CC 7.x+ | CUDA C++ Programming Guide atomics | Eval/state half ablations. | Training gradients likely need fp32. |
| `atomicAdd(float2*)`, `atomicAdd(float4*)` | CC 9.x+, global memory only | CUDA C++ Programming Guide atomics | Possible gradient vector updates on Hopper+. | Atomicity is per element, not whole vector. Not a correctness shortcut. |
| Warp-aggregated atomics by key | Kepler+ concept, modern sync variants preferred | https://developer.nvidia.com/blog/voting-and-shuffling-optimize-atomic-operations/ | Reduce multiple lanes updating same Gaussian to one atomic per warp/key. | Distribution-dependent. |
| `cuda::memcpy_async` | SM80+ for async path, fallback otherwise | https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/async-copies.html | Async global-to-shared staging of splat batches. | Needs alignment; small irregular loads may not benefit. |
| PTX `cp.async.*.shared.global` | PTX ISA 7.0, SM80+ | https://docs.nvidia.com/cuda/parallel-thread-execution/ | Lower-level control of async copies and wait groups. | More fragile than C++ APIs. |
| `cuda::barrier`, `cuda::pipeline` | SM80+ hardware acceleration for block scope | CUDA C++ Programming Guide async sections | Producer/consumer staging of splat chunks. | Increases complexity; only use after baseline profiling. |
| TMA `cuda::ptx::cp_async_bulk_tensor` | CC 9.0+ | CUDA async copies guide | Bulk tensor copies for multidimensional global/shared tiles. | Likely overkill for 256-splat chunks; useful for large planes/tile queues. |
| `cuTensorMapEncodeTiled` | CC 9.0+ TMA setup | CUDA async copies guide | Host-created tensor map for TMA multidimensional copies. | Host setup and shape rigidity. |
| Thread block clusters | CC 9.0+ | CUDA C++ Programming Guide | Co-schedule blocks on a GPU Processing Cluster. | Not needed for one block per tile baseline. |
| Distributed shared memory | CC 9.0+ | CUDA C++ Programming Guide | Blocks in a cluster can read/write/atomic remote shared memory. | Interesting for heavy tiles split across blocks; high complexity. |
| `cluster.sync()` | CC 9.0+ cluster API | CUDA C++ Programming Guide | Required for safe DSM access. | Cluster-level synchronization can limit occupancy. |
| CUDA Graphs `cudaGraphLaunch` | CUDA 10+ family, current guide | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html | Collapse repeated fixed-shape training pipeline launch overhead. | Graph update/capacity changes need careful design. |
| CUDA Dynamic Parallelism CDP2 | CUDA 12+ default, CC 9+ only CDP2 | https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/dynamic-parallelism.html | Device-side launch for irregular heavy work. | Hot-path overhead and runtime limits make it a poor first choice. |
| CUDA external memory | CUDA interop | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html | Import Vulkan/D3D buffers/images into CUDA without copies. | Cross-API lifetime and layout hazards. |
| CUDA external semaphores | CUDA interop | same as above | GPU-side ordering between Vulkan/D3D and CUDA. | Wait-before-signal and ownership mistakes can deadlock or corrupt. |
| CUDA surface objects | CUDA arrays / surfaces | CUDA C++ Programming Guide texture/surface memory | Possible image writes if graphics interop maps arrays. | Compute path can just use linear tensor pointers. |
| CUTLASS/CuTe layouts | Library, Volta through Blackwell support | https://docs.nvidia.com/cutlass/latest/overview.html | Useful reference for tiling, async copy, TMA, warp specialization patterns. | Tensor cores do not directly fit alpha compositing. |

## Existing CUDA Rasterizer References

### Graphdeco Diff Gaussian Rasterization

URL: https://github.com/graphdeco-inria/diff-gaussian-rasterization

Useful architecture:

```text
preprocessCUDA:
  project 3D Gaussian to 2D conic/radius/depth
  compute tiles_touched

CUB InclusiveSum:
  point_offsets from tiles_touched

duplicateWithKeys:
  emit 64-bit key = tile_id | depth_bits
  emit gaussian index

CUB DeviceRadixSort::SortPairs:
  sort intersections by tile then depth

identifyTileRanges:
  convert sorted keys to per-tile ranges

renderCUDA forward:
  one block per tile
  shared batches
  one thread per pixel
  front-to-back alpha compositing
  save final transmittance/last contributor state

renderCUDA backward:
  reverse tile traversal
  per-pixel atomics to Gaussian gradients
```

What to borrow:

- key packing and CUB sort pipeline;
- one block per tile;
- projection/preprocess split;
- simple correctness reference.

What to improve:

- reduce backward atomics at tile/block level;
- avoid CPU-sized allocation in training loops;
- tighter tile intersection when conics/opacities are known.

### gsplat

URL: https://github.com/nerfstudio-project/gsplat

Docs: https://docs.gsplat.studio/main/apis/rasterization.html

Useful architecture:

```text
projection_ewa_3dgs_fused_fwd:
  B*C*N fused camera projection
  opacity-aware radius/conic/depth output

intersect_tile:
  optional AccuTile/SNUGBOX ellipse intersection
  first pass counts tiles per Gaussian
  cumsum counts
  second pass emits isect_ids and flatten_ids

radix_sort_double_buffer:
  CUB DeviceRadixSort::SortPairs with DoubleBuffer

segmented_radix_sort_double_buffer:
  CUB DeviceSegmentedRadixSort::SortPairs with DoubleBuffer

rasterize_to_pixels_3dgs_fwd:
  grid = [image, tile_y, tile_x]
  threads = [tile_size, tile_size]
  dynamic shared memory for ids/conics/means
  output render_colors, render_alphas, last_ids

rasterize_to_pixels_3dgs_bwd:
  reverse traversal
  cooperative_groups tiled_partition<32>
  warp reductions
  gpuAtomicAdd to gradients
```

What to borrow:

- AccuTile/SNUGBOX opacity-aware tile culling;
- DoubleBuffer CUB sort;
- `render_alpha + last_ids` state contract;
- fused projection and packed modes;
- warp reduction helpers.

What to improve:

- no hot-path `cum_tiles_per_gauss[-1].item<int64_t>()`;
- full-block reduction ablation before global atomics;
- V8-style `tile_stop_counts` recompute ablation;
- fixed-shape CUDA Graph capture.

## Replacement Table For Metal-Specific Features

| Metal feature | CUDA/native replacement | Graphics replacement |
|---|---|---|
| Imageblock memory | CUDA `__shared__` memory | Vulkan tile/local read, framebuffer fetch, subpasses |
| Tile shader dispatch | One CUDA block per tile | Vulkan render pass tile/local attachment path |
| Raster order groups | Sorted tile list and serial per-pixel loop | `VK_EXT_fragment_shader_interlock` or ROAA |
| ICB | CUDA Graphs, persistent kernels, work queues | Vulkan indirect draws / mesh shader task generation |
| Direct MPS tensor render target | Direct torch CUDA tensor pointer writes | Vulkan external memory imported into CUDA |
| Metal `simdgroup` | CUDA warp, `thread_block_tile<32>` | Vulkan subgroup |
| Metal relaxed atomics | CUDA `atomicAdd`, warp-aggregated atomics | Fragment shader atomics/interlock, but avoid for training if possible |
| Metal buffer-backed texture row alignment | Not needed for CUDA linear tensors | Still relevant for image interop and graphics attachments |

## Workgroup Choices

Default:

```text
tile_size = 16
threads_per_block = 16 * 16 = 256
warps_per_block = 8
one pixel per thread
one block per tile per image
```

Why keep it:

- matches Graphdeco, gsplat, and local Mac V8 evidence;
- gives one warp per 32 pixels and 8 warps per tile;
- enough pixel lanes for reductions before atomics;
- shared memory for 256 ids + 256 mean/opacities + 256 conics is modest.

Shared memory estimates for 3-channel baseline:

```text
id_batch[256]           int32  =  1 KB
mean_opacity[256]       vec3   =  3 KB
conic[256]              vec3   =  3 KB
color[256 * 3]          float  =  3 KB
warp/block partials            < 2 KB
typical total                  ~10-16 KB
```

This leaves occupancy headroom on most RTX-class GPUs. Register pressure is more
likely than shared memory to be the backward limiter.

## Atomic Strategy

Default training backward:

```text
for each splat in a tile:
  each pixel thread computes local partials
  reduce inside warp
  reduce across 8 warps in shared memory
  one lane issues global atomicAdd for:
    mean x/y
    conic a/b/c
    color r/g/b
    opacity
```

That is 9 fp32 atomics per splat/tile after tile reduction.

Do not use:

```text
per pixel per splat per component global atomicAdd
```

Deferred partial path:

```text
tile backward writes [gaussian_id, 9 grad floats] per tile/splat
CUB radix sort or run-length group by gaussian_id
reduce partials to final gradients
```

Use only when:

- Nsight Compute shows atomic contention dominates;
- cluster/heavy scenes reuse the same Gaussian across many tiles;
- added partial-buffer bandwidth is lower than atomic serialization cost.

## Memory Layout Recommendations

Input/projected:

```text
means2d:    float2 or two float planes
conics:     float3 packed as struct-of-arrays if coalescing improves
colors:     float3/fp32 for training, half/uint8 only for eval ablations
opacities:  float
depths:     float, bitcast for radix key
```

Intersections:

```text
isect_ids:  int64 [image bits | tile bits | depth bits]
flatten_ids:int32 [n_isects]
offsets:    int32 [I, tile_h, tile_w + sentinel strategy]
```

Output/state options:

| State | 4K B=1 cost | Notes |
|---|---:|---|
| RGB fp32 | 201 MB | Training output. |
| RGBA fp32 | 268 MB | Avoid unless needed for API compatibility. |
| alpha/final_T fp32 | 67 MB | Standard CUDA backward helper. |
| last_id i32 | 67 MB | Standard CUDA backward helper. |
| tile_stop_counts i32 | ~0.3 MB for 4096^2/16 tiles | V8-style compact state, requires recompute. |
| front-K/history | unbounded | Reject as default. |

## Feature-Specific Notes

### CUB Sort Bit Ranges

Use bit ranges aggressively. If key layout is:

```text
[image bits | tile bits | depth bits]
```

then `begin_bit=0`, `end_bit=32 + tile_n_bits + image_n_bits` is enough. For
segmented per-image sorting, sort only the lower `32 + tile_n_bits` bits.

### `float2` / `float4` Atomics

CUDA supports vector `atomicAdd` only on CC 9.x+ and only for global memory
addresses. The operation is atomic per element, not as a single vector
transaction. Use it as a possible instruction-count optimization, not a
correctness primitive.

### TMA

TMA is attractive for large regular multidimensional transfers, but splat
parameter chunks are small and irregular after sorting. First use normal
coalesced loads and then `cp.async`. Consider TMA only for:

- large output/state plane copies;
- structured tile queues;
- heavy-tile chunk descriptor movement;
- Hopper-specific branch after baseline wins.

### Thread Block Clusters / DSM

Clusters and distributed shared memory can help if one heavy tile must be split
across multiple blocks while sharing a tile-local state. That is a late-stage
heavy-tile experiment. It is not needed for the baseline 16x16 tile kernel.

### CUDA Graphs

Use when shapes and capacities are fixed:

```text
capture:
  projection_count
  scan
  emit
  sort
  offsets
  forward
  backward
  projection_backward

replay:
  cudaGraphLaunch(graphExec, stream)
```

Graph node updates can handle pointer/parameter changes if topology is stable.
Capacity growth or a different number of kernels may require recapture.

### Dynamic Parallelism

CUDA dynamic parallelism can launch child kernels from device code, but CDP2 has
runtime overhead and resource limits. It is not a good default for "one heavy
tile launches a child kernel" until a work-queue implementation has been tested
and failed.

## Hardware Raster References For Separate Branch

`VK_EXT_fragment_shader_interlock`:

- Provides fragment shader critical sections for overlapping pixels.
- Useful for per-pixel data structures and programmable blending.
- Still depends on API primitive order.

`VK_EXT_rasterization_order_attachment_access`:

- Guarantees framebuffer fetch sees prior same-pixel writes in rasterization
  order.
- Does not make blending order-independent.
- Sorted primitive submission is still required for correct alpha compositing.

CUDA interop:

- Import graphics memory with `cudaImportExternalMemory`.
- Map buffers with `cudaExternalMemoryGetMappedBuffer`.
- Import and wait/signal semaphores with `cudaImportExternalSemaphore`,
  `cudaWaitExternalSemaphoresAsync`, and `cudaSignalExternalSemaphoresAsync`.

This branch should start with a minimal constant-output image interop smoke
before any 3DGS shader.

## Dead Ends

- Per-pixel linked lists for training: unbounded memory, global atomics, pointer
  chasing.
- Per-pixel front-K as default: repeats the known Metal memory cliff.
- Global fixed-capacity sort over mostly empty capacity: high memory traffic.
- Dynamic parallelism per tile: launch/runtime overhead likely dominates.
- TMA-first design: too much complexity before proving memory movement is the
  limit.
- Vulkan/CUDA hardware raster as the first CUDA training base: too many
  variables before a clean compute baseline.

## Source URLs

- CUDA C++ Programming Guide:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Cooperative Groups:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
- CUDA Graphs:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- CUDA API interoperability:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html
- CUDA async copies/TMA:
  https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/async-copies.html
- PTX ISA:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUB/CCCL:
  https://nvidia.github.io/cccl/cub/
- CUTLASS:
  https://docs.nvidia.com/cutlass/latest/overview.html
- CUDA warp-level primitives blog:
  https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
- Warp-aggregated atomics by key:
  https://developer.nvidia.com/blog/voting-and-shuffling-optimize-atomic-operations/
- gsplat:
  https://github.com/nerfstudio-project/gsplat
- gsplat rasterization docs:
  https://docs.gsplat.studio/main/apis/rasterization.html
- Graphdeco diff-gaussian-rasterization:
  https://github.com/graphdeco-inria/diff-gaussian-rasterization
- NVIDIA Vulkan Gaussian splatting sample article:
  https://developer.nvidia.com/blog/real-time-gpu-accelerated-gaussian-splatting-with-nvidia-designworks-sample-vk_gaussian_splatting/
- Vulkan ROAA sample:
  https://docs.vulkan.org/samples/latest/samples/extensions/rasterization_order_attachment_access/README.html
- `VK_EXT_fragment_shader_interlock`:
  https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_fragment_shader_interlock.html
