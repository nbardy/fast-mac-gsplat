# V9 HW Output Planes Probe Notes

Date: 2026-04-25

Scope: `variants/v9_hw_output_planes_probe`

## What Changed

- Added format metadata for direct buffer-backed render targets:
  `RGBA32Float`, `RGBA16Float`, `R32Float`, and `RG32Float`.
- Added runnable Gaussian eval output for `RGBA16Float` alongside the existing
  `RGBA32Float` path.
- Kept ICB execution untouched. This probe still uses normal render command
  encoding only.
- Added Python validation helpers and benchmark paths for:
  `formats`, `gaussian-direct-rgba32f`, `gaussian-direct-rgba16f`,
  `gaussian-blit-rgba32f`, and `gaussian-blit-rgba16f`.

## Validation

`python3 tests/interop_check.py` passes on Apple M4.

Validation readback happens after the native op returns. The native render ops
do not call `getBytes`, `waitUntilCompleted`, or CPU-stage GPU output.

The one-splat validation result:

| Output | Max Abs Error |
|---|---:|
| RGBA32F Gaussian | 0.0 |
| RGBA16F Gaussian | 0.00048828125 |

The RGBA16F error is half-precision quantization of the expected
premultiplied RGBA center value.

## Row Alignment

Metal buffer-backed textures require 256-byte aligned rows. For contiguous Torch
MPS tensors in this probe, the width multiples are:

| Format | Bytes/Pixel | Direct Width Multiple |
|---|---:|---:|
| RGBA32F | 16 | 16 |
| RGBA16F | 8 | 32 |
| R32F | 4 | 64 |
| RG32F | 8 | 32 |

Unaligned direct paths fail closed before encoding the render pass. A private
texture plus GPU blit can be used as a fallback, but it adds a full-frame copy.

## Returning RGBA Then Slicing

Returning RGBA32F and slicing channels is not an output-bandwidth reduction. The
render pass still stores the full RGBA32F target, and a materialized slice adds
another GPU memory operation. Returning RGBA16F is acceptable only if downstream
code can consume `torch.float16` or explicitly owns the conversion. If fewer
channels are the goal, mainline should use a real R/RG render target or a packed
format, not an RGBA target followed by slicing.

## Benchmark Summary

Full results:

- `benchmarks/v9_hw_output_planes_rgba16_direct_formats_gaussian_6000.jsonl`
- `benchmarks/v9_hw_output_planes_rgba16_gaussian_4k64k.jsonl`
- `benchmarks/v9_hw_output_planes_rgba16_direct_formats_gaussian_6000.md`

Key medians:

| Case | RGBA32F | RGBA16F |
|---|---:|---:|
| 512x512 constant | 0.728 ms | 0.391 ms |
| 1080x1920 constant | 1.267 ms | 0.994 ms |
| 4096x4096 constant | 3.175 ms | 2.969 ms |
| 512x512 Gaussian 6K | 1.659 ms | 1.679 ms |
| 1080x1920 Gaussian 6K | 1.846 ms | 1.739 ms |
| 4096x4096 Gaussian 6K | 4.976 ms | 1.863 ms |
| 4096x4096 Gaussian 64K | 6.123 ms | 4.958 ms |

## Recommendation

Promote RGBA16F as the next mainline output-format candidate for fixed eval, but
gate it behind an image-error comparison against RGBA32F on parity-shaped
inputs. The path is runnable and removes half the render-target storage
bandwidth without adding a copy.

Split R/RG planes remain useful for downstream consumers that do not need full
RGBA, but Gaussian eval would need shader/API work to emit meaningful separate
planes. This probe only validates R/RG direct targets for constant renders.
