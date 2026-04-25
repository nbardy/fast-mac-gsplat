# V9 Hardware Raster Reflection Handoff

Date: 2026-04-25

This note is a context checkpoint, not a next-plan. It records what we tried,
what worked, what failed, and what is still ambiguous after the V8/V9 hardware
raster exploration.

## Short Version

The real win was not a finished hardware rasterizer. The win was reducing the
unknowns:

- Metal render-pass output can land directly in Torch/MPS tensor storage without
  CPU staging.
- Screen-space Gaussian eval through a fragment shader works and can be fast.
- `RGBA16F` output is the strongest current eval-format candidate.
- Fixed-function blending is not a v8-compatible multi-splat renderer.
- Tile/imageblock and ROG features are available enough to keep investigating,
  but we have not yet implemented the programmable ordered `C/T/stop` state that
  backward needs.
- ICB execution is fenced off after an AGX crash. Allocation probes are fine;
  execute is not part of any safe path.

## What We Built

### `variants/v9_hw_interop_probe`

Purpose: prove lower-level output interop.

Working:

- Metal render pass writes to a Torch MPS tensor through a buffer-backed
  `RGBA32Float` texture.
- Private texture plus GPU blit fallback also works.
- Native op returns the tensor without CPU readback or `waitUntilCompleted`.
- Tile/imageblock pipeline compiles.
- Raster order groups are supported on the tested Apple M4.
- ICB allocation probe succeeds.

Main benchmark result:

| Resolution | Blit Median | Direct Median |
|---:|---:|---:|
| 512x512 | 0.720 ms | 0.358 ms |
| 1080x1920 | 3.500 ms | 1.090 ms |
| 4096x4096 | 15.947 ms | 4.683 ms |

Conclusion: output interop is no longer the main blocker.

### `variants/v9_hw_fixed_eval_probe`

Purpose: render actual screen-space Gaussian quads from MPS tensors.

Working:

- Instanced render pipeline reads `means2d`, conics, colors, and opacities from
  MPS tensor buffers.
- Fragment shader evaluates Gaussian alpha.
- Hardware source-over blending writes premultiplied RGBA.
- Direct Torch/MPS output remains GPU-native.

Important benchmark results:

| Resolution | Splats | Median ms |
|---:|---:|---:|
| 512x512 | 6000 | 1.672 |
| 1080x1920 | 6000 | 1.630 |
| 4096x4096 | 65536 | 12.378 |

Conclusion: fast enough to keep studying, but not a parity result.

### `variants/v9_hw_eval_parity_probe`

Purpose: compare V9 fixed eval against V8 forward eval on identical projected
tensors.

Working:

- The harness sends the same projected tensors to both V8 and V9.
- Single-splat black-background rows match to about `1.49e-08` max RGB error.
- The harness now fails clearly if `v8_hw_eval` native ops are not built.

Failing:

- Multi-splat rows do not match V8.

Representative errors:

| Case | Resolution | Splats | Max RGB Error |
|---|---:|---:|---:|
| `grid_ordered` | 16x16 | 16 | 0.0967607 |
| `grid_ordered` | 64x64 | 16 | 0.00247034 |
| `overlap_ordered` | 16x16 | 16 | 0.123525 |
| `overlap_ordered` | 64x64 | 16 | 0.0579901 |

Conclusion: fixed-function blending can match the trivial case, but it does not
provide the ordered front-to-back transmittance contract that V8 implements.

### `variants/v9_hw_sorted_eval_probe`

Purpose: test whether stable depth submission order closes the parity gap.

Working:

- `render_gaussian_eval_rgba_sorted(...)` uses stable
  `torch.argsort(depths.detach(), stable=True)` on MPS.
- Tests prove order changes output and repeated sorted calls are deterministic.
- Default sort convention matches the V8 Python wrapper's ascending-depth sort.

Did not help enough:

- Sorting controls draw submission order, but the render path still uses
  fixed-function source-over blending.
- It still does not expose `T`, `final_T`, `stop_count`, or a processed prefix
  for backward.

Conclusion: useful diagnostic scaffolding, not a parity solution.

### `variants/v9_hw_output_planes_probe`

Purpose: reduce output bandwidth and test lower-precision render targets.

Working:

- Direct constant render targets validate for `RGBA32F`, `RGBA16F`, `R32F`, and
  `RG32F`.
- Gaussian eval validates for `RGBA32F` and `RGBA16F`.
- `RGBA16F` Gaussian one-splat max abs error is `0.00048828125`, matching half
  precision quantization expectations.

Key median timings:

| Case | RGBA32F | RGBA16F |
|---|---:|---:|
| 512x512 Gaussian 6K | 1.659 ms | 1.679 ms |
| 1080x1920 Gaussian 6K | 1.846 ms | 1.739 ms |
| 4096x4096 Gaussian 6K | 4.976 ms | 1.863 ms |
| 4096x4096 Gaussian 64K | 6.123 ms | 4.958 ms |

Conclusion: `RGBA16F` is the best current eval-output candidate if image
tolerance accepts it. Returning RGBA32F and slicing channels is not a bandwidth
optimization because the full RGBA32F target is still stored.

### `variants/v9_hw_tile_state_probe`

Purpose: measure tile/imageblock state feasibility.

Working:

- Tile/imageblock pipelines compile.
- Native layout probes report sample length and per-tile memory.

Important measurements on Apple M4:

| Layout | Bytes / Sample | 16x16 Tile | 32x32 Tile |
|---|---:|---:|---:|
| `half4_baseline` | 24 | 6,144 B | 24,576 B |
| `ct_fp32` | 32 | 8,192 B | 32,768 B |
| `ct_stop_flags_fp32_u32x2` | 48 | 12,288 B | 49,152 B |

Did not help yet:

- We have not implemented a full programmable per-pixel ordered compositing
  pass using imageblock memory.
- Useful training state is expensive. `C/T/stop/flags` doubles the measured
  baseline and reaches about 49 KB per 32x32 tile.
- Imageblock memory solves storage locality only. It does not by itself solve
  sorted splat ingestion, exact early stop, gradient reduction, or backward
  replay.

Conclusion: tile/imageblock is still the right Metal feature for a serious
hardware-training attempt, but only if used to implement explicit V8 math. The
compile/layout probes alone do not make the renderer correct or faster.

### `variants/v9_hw_draw_formats_probe`

Purpose: test direct render target formats and ICB execution.

Working:

- Direct `RGBA32F`, `RGBA16F`, `R32F`, and `RG32F` constant render targets
  validate.
- ICB allocation succeeds.

Failed:

- Minimal ICB execution crashed Python inside
  `AGX::RenderContext::executeCommandsInBufferCommon`.
- The path is now fail-closed in both Python and native code.

Likely bug:

- The render pipeline descriptor probably missed
  `supportIndirectCommandBuffers = YES`. That is documented in
  `docs/v9_hw_icb_crash_handoff.md`.

Conclusion: ICB should stay out of the hot path. If revisited, it needs a
separate crash harness with Metal validation enabled, not shared benchmark code.

## Accuracy Bugs / Semantic Gaps

The main accuracy problem is not a numerical precision bug. It is a semantic
gap between fixed-function raster blending and V8's math.

V8's forward contract is:

```text
C += T * alpha_i * color_i
T *= (1 - alpha_i)
stop when T <= transmittance_threshold
```

V8 also sorts inputs by depth, bins by tile, sorts tile-local IDs, and can save
or recompute the stopped prefix for backward.

Current V9 fixed eval:

- emits Gaussian quads as render primitives;
- lets fixed-function source-over blending combine fragments;
- outputs premultiplied RGBA;
- does not expose final `T`;
- does not expose the first stopped index or processed prefix;
- does not provide a v8-compatible backward contract.

This explains the observed behavior:

- single splat matches V8 because order and transmittance recurrence are
  trivial;
- multi-splat scenes diverge because draw/fragment behavior is not the same
  explicit ordered recurrence V8 uses;
- depth-mismatch rows can sometimes numerically match on one machine, but that
  is an observed hardware behavior, not a documented V8 ordering guarantee.

## What We Tried That Did Not Move The Main Blocker

- **Fixed-function hardware blending:** fast, but does not expose training
  state.
- **Stable MPS sorting wrapper:** makes order deterministic, but does not create
  `T`, `final_T`, or stop metadata.
- **RGBA output slicing idea:** rejected because it does not reduce render-target
  stores.
- **ICB execute path:** crashed before the fail-closed guard.
- **Tile/imageblock compile probes:** proved feature availability and memory
  shape, but did not yet implement exact V8 compositing.
- **ROG support probe:** useful signal, but ROG alone does not solve sorted
  ingestion, transmittance state, or backward reductions.

## Could Tile/Imageblock Still Be Worth Trying?

Yes, but only under a narrower definition.

Worth trying:

- a tile/imageblock forward pass that explicitly owns per-pixel `C` and `T`;
- `stop_count` or `last_id` capture for backward replay;
- small tile sizes chosen from measured imageblock pressure;
- exact comparison against V8 multi-splat cases before any benchmark claim;
- optional `RGBA16F` output only after correctness is established.

Probably not worth trying:

- using imageblock as a decorative replacement for a normal render target;
- storing long per-pixel histories or front-K lists at 4K;
- assuming ROG/fixed blending can replace V8's ordered recurrence;
- enabling ICB execution inside shared benchmark code before the isolated crash
  harness proves it safe.

## CUDA Reflection

CUDA should not copy the Metal hardware-raster path directly. CUDA does not
expose fixed-function raster hardware inside CUDA kernels. The strongest CUDA
path is compute-first:

```text
fused 3D projection
opacity-aware tile intersection
CUB scan/radix sort
one CUDA block per 16x16 tile
direct writes to Torch CUDA tensors
compact forward state
backward replay with warp/block reductions before global atomics
```

Hardware raster on NVIDIA likely means a separate Vulkan or Direct3D graphics
interop branch using fragment interlock / rasterization-order attachment access
plus CUDA external memory. That may be useful later, but it should not block a
CUDA compute V9 baseline.

## Current Best Base

- Best training/backward base: `variants/v8`.
- Best Metal eval-output base: `variants/v9_hw_output_planes_probe`.
- Best parity diagnostic: `variants/v9_hw_eval_parity_probe`.
- Best CUDA direction: compute-first notes in
  `docs/v9_cuda_hardware_rasterization_notes.md`.

The next planning session should start from this premise: output interop is
solved enough; multi-splat ordered compositing and backward state are not.
