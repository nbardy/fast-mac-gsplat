# V9 HW Output Planes RGBA16F Sweep

Date: 2026-04-25

Variant: `variants/v9_hw_output_planes_probe`

Device: Apple M4

Command:

```bash
python3 benchmarks/benchmark_interop.py \
  --sizes 512x512,1080x1920,4096x4096 \
  --warmup 5 \
  --iters 20 \
  --paths formats,gaussian-direct-rgba32f,gaussian-direct-rgba16f \
  --formats rgba32f,rgba16f,r32f,rg32f \
  --gaussians 6000 \
  --jsonl ../../benchmarks/v9_hw_output_planes_rgba16_direct_formats_gaussian_6000.jsonl
```

## Direct Constant Render

All rows are direct buffer-backed render targets over Torch MPS tensor storage.

| Resolution | Format | Median ms | Mean ms | Width Multiple |
|---:|---|---:|---:|---:|
| 512x512 | RGBA32F | 0.728 | 0.744 | 16 |
| 512x512 | RGBA16F | 0.391 | 0.418 | 32 |
| 512x512 | R32F | 0.521 | 0.689 | 64 |
| 512x512 | RG32F | 0.453 | 0.463 | 32 |
| 1080x1920 | RGBA32F | 1.267 | 1.383 | 16 |
| 1080x1920 | RGBA16F | 0.994 | 1.277 | 32 |
| 1080x1920 | R32F | 0.889 | 0.968 | 64 |
| 1080x1920 | RG32F | 1.105 | 1.451 | 32 |
| 4096x4096 | RGBA32F | 3.175 | 3.546 | 16 |
| 4096x4096 | RGBA16F | 2.969 | 3.345 | 32 |
| 4096x4096 | R32F | 2.103 | 2.524 | 64 |
| 4096x4096 | RG32F | 3.283 | 3.661 | 32 |

## Gaussian Direct Render, 6K Splats

This is the fixed-eval probe path: instanced screen-space Gaussian quads,
hardware source-over blending, no ICB execution.

| Resolution | Format | Splats | Median ms | Mean ms |
|---:|---|---:|---:|---:|
| 512x512 | RGBA32F | 6000 | 1.659 | 1.888 |
| 512x512 | RGBA16F | 6000 | 1.679 | 2.184 |
| 1080x1920 | RGBA32F | 6000 | 1.846 | 2.282 |
| 1080x1920 | RGBA16F | 6000 | 1.739 | 3.268 |
| 4096x4096 | RGBA32F | 6000 | 4.976 | 5.618 |
| 4096x4096 | RGBA16F | 6000 | 1.863 | 2.574 |

## Focused 4K/64K Gaussian Direct

Command:

```bash
python3 benchmarks/benchmark_interop.py \
  --sizes 4096x4096 \
  --warmup 3 \
  --iters 10 \
  --paths gaussian-direct-rgba32f,gaussian-direct-rgba16f \
  --gaussians 65536 \
  --jsonl ../../benchmarks/v9_hw_output_planes_rgba16_gaussian_4k64k.jsonl
```

| Resolution | Format | Splats | Median ms | Mean ms |
|---:|---|---:|---:|---:|
| 4096x4096 | RGBA32F | 65536 | 6.123 | 6.807 |
| 4096x4096 | RGBA16F | 65536 | 4.958 | 5.856 |

## Recommendation

Use RGBA16F as the next mainline output-format candidate for fixed eval if the
image-quality tolerance accepts half precision. It is runnable for Gaussian eval
with no CPU staging and gives the strongest 4K Gaussian win in this probe.

Do not return RGBA32F and slice channels to reduce bandwidth. That still pays
the full RGBA32F render-target store, and any materialized slice adds another
GPU memory operation. If consumers truly need fewer channels, use a real R/RG
target or a dedicated packed target.
