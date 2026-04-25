# V9 Parallel Exploration Results

## Safe Benchmark Results

These are safe-path results after the ICB execute path was fenced off. ICB
execution is not included.

### Fixed Eval Gaussian Direct

This path renders instanced screen-space Gaussian quads from MPS input tensors
into a direct buffer-backed `RGBA32Float` MPS tensor. It is eval-only and not
v8-equivalent yet: no depth sort, no batching, no exact front-to-back
transmittance, and no backward state.

| Resolution | Splats | Median ms | Mean ms |
|---:|---:|---:|---:|
| 512x512 | 1 | 0.516 | 0.496 |
| 1080x1920 | 1 | 0.876 | 1.449 |
| 512x512 | 512 | 0.472 | 0.507 |
| 1080x1920 | 512 | 0.943 | 0.946 |
| 512x512 | 6000 | 1.672 | 2.416 |
| 1080x1920 | 6000 | 1.630 | 1.824 |
| 4096x4096 | 65536 | 12.378 | 12.524 |

The 4K/64K number is close to the old v8 4K/64K uniform forward mean
(`12.142 ms`), but this is not a parity benchmark. The v9 fixed-eval probe uses
small sparse quads and hardware source-over blending in input order. Treat it as
proof that render-shader tensor inputs and direct output work, not as a win yet.

### Direct Constant Render Baseline

| Resolution | Format | Median ms | Mean ms |
|---:|---|---:|---:|
| 512x512 | RGBA32F | 0.257 | 0.261 |
| 1080x1920 | RGBA32F | 0.814 | 0.859 |
| 4096x4096 | RGBA32F | 3.470 | 3.799 |

### Direct Output Format Sweep

All formats below create direct buffer-backed render targets over Torch MPS
tensors and validate by CPU readback in the test path.

| Resolution | Format | Median ms | Mean ms | Width Multiple |
|---:|---|---:|---:|---:|
| 512x512 | RGBA32F | 0.493 | 1.129 | 16 |
| 1080x1920 | RGBA32F | 1.134 | 1.483 | 16 |
| 4096x4096 | RGBA32F | 4.737 | 5.601 | 16 |
| 512x512 | RGBA16F | 0.395 | 0.952 | 32 |
| 1080x1920 | RGBA16F | 0.922 | 1.300 | 32 |
| 4096x4096 | RGBA16F | 2.763 | 3.437 | 32 |
| 512x512 | R32F | 0.392 | 0.947 | 64 |
| 1080x1920 | R32F | 0.604 | 0.620 | 64 |
| 4096x4096 | R32F | 2.044 | 2.624 | 64 |
| 512x512 | RG32F | 0.345 | 0.852 | 32 |
| 1080x1920 | RG32F | 0.793 | 0.800 | 32 |
| 4096x4096 | RG32F | 3.333 | 3.735 | 32 |

The format sweep says output bandwidth still matters. If accuracy allows it,
RGBA16F or split R/RG planes are worth exploring. There is no true direct RGB
Torch tensor render target in this probe.

## Second Parallel Round Results

### V8 Parity Harness

`variants/v9_hw_eval_parity_probe` now has a dedicated v8 comparison harness:

- `tests/parity_v8_smoke.py`
- `benchmarks/benchmark_parity_v8.py`
- `benchmarks/v9_hw_eval_parity_smoke.md`
- `benchmarks/v9_hw_eval_parity_smoke_rerun.md`

The harness passes the same projected `means2d`, conics, colors, opacities, and
depths to v8 forward eval and v9 fixed eval, then compares v9 premultiplied RGB
against v8 RGB on a black background.

Key result: single-splat rows match; multi-splat rows do not.

| Case | Resolution | Splats | Comparable | Max RGB Error | V8 Median ms | V9 Median ms |
|---|---:|---:|---|---:|---:|---:|
| `tiny_single` | 16x16 | 1 | yes | `1.49e-08` | 4.952 | 0.521 |
| `tiny_single` | 64x64 | 1 | yes | `1.49e-08` | 2.893 | 0.358 |
| `grid_ordered` | 16x16 | 16 | no | `9.68e-02` | 4.043 | 0.519 |
| `grid_ordered` | 64x64 | 16 | no | `2.47e-03` | 4.123 | 0.427 |
| `overlap_ordered` | 16x16 | 16 | no | `1.24e-01` | 15.151 | 0.427 |
| `overlap_ordered` | 64x64 | 16 | no | `5.80e-02` | 4.341 | 0.389 |

The v9 timings are encouraging for eval, but they are not promotable parity
wins yet. The multi-splat rows are diagnostics: fixed-function source-over
blending does not expose v8's ordered front-to-back transmittance contract,
final `T`, or stop point.

### Sorted Eval Wrapper

`variants/v9_hw_sorted_eval_probe` adds
`render_gaussian_eval_rgba_sorted(...)`. It uses stable
`torch.argsort(depths.detach(), stable=True)` on MPS tensors and submits
reordered splats to the existing renderer. Tests show order changes output and
default ascending-depth order matches manual reordering.

This is useful test scaffolding, not a parity solution. It controls submission
order, but the renderer still uses fixed hardware blending and still lacks
v8-compatible transmittance and stop metadata.

### RGBA16F Output Plane

`variants/v9_hw_output_planes_probe` adds Gaussian eval output for
`RGBA16Float` alongside `RGBA32Float`. One-splat validation gives:

| Output | Max Abs Error |
|---|---:|
| RGBA32F Gaussian | 0.0 |
| RGBA16F Gaussian | 0.00048828125 |

Key median timings:

| Case | RGBA32F | RGBA16F |
|---|---:|---:|
| 512x512 Gaussian 6K | 1.659 ms | 1.679 ms |
| 1080x1920 Gaussian 6K | 1.846 ms | 1.739 ms |
| 4096x4096 Gaussian 6K | 4.976 ms | 1.863 ms |
| 4096x4096 Gaussian 64K | 6.123 ms | 4.958 ms |

`RGBA16F` is the strongest current eval-output candidate if image-quality
tolerance accepts half precision. Returning RGBA32F and slicing channels is not
a bandwidth optimization because the render pass still stores the full RGBA32F
target.

### CUDA Research Track

The CUDA research notes are:

- `docs/v9_cuda_hardware_rasterization_notes.md`
- `docs/v9_cuda_primitives_index.md`

The recommendation is compute-first CUDA, not graphics-raster-first CUDA:
fused projection, opacity-aware tile intersection, CUB scan/radix sort, one CUDA
block per 16x16 tile, direct Torch CUDA output, compact forward state, and
backward replay with warp/block reductions before global atomics. Vulkan
hardware raster with fragment interlock or rasterization-order attachment access
should be a separate branch after the compute baseline is measured.

## Third Parallel Round Results

### Direction B: Reverse-Order Output-Planes Eval

`variants/v9_hw_output_planes_probe` now has format-aware sorted wrappers:

- `render_gaussian_eval_format_sorted(...)`
- `render_gaussian_eval_rgba_sorted(...)`
- `render_gaussian_eval_rgba16_sorted(...)`

The sorted parity diagnostic confirms the fixed-function source-over rule:
black-background V8 color matches only when the hardware path submits splats in
reverse/depth-descending painter order.

| Case | Format | Order | Max RGB Error vs V8 |
|---|---|---|---:|
| 16x32 G=2 | RGBA32F | input / ascending | 0.25 |
| 16x32 G=2 | RGBA32F | descending | `9.31e-10` |
| 16x32 G=16 | RGBA32F | input / ascending | 0.385184 |
| 16x32 G=16 | RGBA32F | descending | `1.19e-07` |
| 64x64 G=16 | RGBA32F | descending | `1.19e-07` |
| 64x64 G=16 | RGBA16F | descending | 0.001294 |

This is a real color-only eval result, not an exact training result. The path
still lacks `final_T`, `stop_count`, stopped-prefix metadata, and backward.

### Direction A: Exact Imageblock C/T Spike

`variants/v9_hw_tile_exact_probe` adds a minimal 16x16-tile exact imageblock
semantic probe. It clears explicit imageblock state, updates it from two
ordered fragments with fixed blending disabled, and resolves `float4(C.rgb,T)`
to a direct Torch/MPS render target.

Observed on Apple M4:

```text
tile_exact_overlap_max_abs_err = 0.0
tile_exact_imageblock_sample_length = 48 B
tile_exact_imageblock_memory_16x16 = 12,288 B
tile_exact_imageblock_memory_32x32 = 49,152 B
```

This proves the missing primitive:

```text
tile clear -> ordered fragment [[imageblock_data]] C/T update
  -> tile resolve -> direct Torch/MPS output
```

It does not yet prove Gaussian quads, V8 tile-bin ingestion, exact stop-count
semantics, backward, or performance. The exact 32x32 state path compiles but
failed render encoder creation on M4, so the API is fail-closed to 16x16.

### Direction C: CUDA Scaffold

`variants/v9_cuda_compute_first` is now a source-level CUDA scaffold. The local
Mac environment reports:

```text
nvcc: missing
nvidia-smi: missing
PyTorch CUDA built: false
torch.cuda.is_available(): false
MPS available: true
```

The scaffold exposes environment checks and CUDA skeletons for
`project_count_fused`, `emit_pairs`, `tile_forward_train`, and
`tile_backward_replay`. Native build intentionally fails on this host with a
clear CUDA-only message, so no fake CUDA benchmark was recorded.

## ICB Crash State

`render_constant_rgba_direct_icb` is disabled/fail-closed in both Python and
native code after a pre-patch crash in `AGX executeCommandsInBufferCommon`.

Claude's handoff in `docs/v9_hw_icb_crash_handoff.md` identifies the most likely
bug: the render pipeline descriptor did not set
`supportIndirectCommandBuffers = YES`. That is a plausible root cause, but ICB
execution should stay out of the shared benchmark path until it is isolated in a
separate crash harness with Metal validation enabled.

## Recommendation

Use `v9_hw_output_planes_probe` as the next eval starting point because it
inherits direct tensor output and adds the useful `RGBA16F` candidate. Do not
promote it as a v8 replacement yet.

Next required Metal work is a real parity-shaped multi-splat path:

```text
v9 fixed eval:
  sorted projected inputs
  same alpha threshold
  comparable conics/radius distribution
  same output target policy

compare against:
  v8 forward-only image
  image max error
  wall time at 512/6K, 1080p/6K, 4K/64K
```

The updated evidence is sharper:

- fixed-function blending can match V8 **color** in controlled black-background
  cases if splats are submitted in reverse painter order;
- programmable imageblock state can implement exact `C/T` updates at 16x16;
- neither path currently produces exact backward state.

Next Metal work should therefore split cleanly:

```text
B: test reverse-order eval on realistic scenes and decide RGBA16F tolerance
A: replace fixed fullscreen fragments with projected Gaussian quads, then feed
   the exact imageblock path from GPU-resident V8 tile bins
```

Only after `final_T` plus stopped-prefix metadata are solved should any
hardware path claim exact forward+backward training.
