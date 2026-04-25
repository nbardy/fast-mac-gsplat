# V9 HW Draw Format and ICB Notes

## ICB execution status

Treat render ICB execution as unsafe in `variants/v9_hw_draw_formats_probe`.

A minimal one-command render ICB execute path reached
`-[AGXG16GFamilyRenderContext executeCommandsInBuffer:withRange:]` and crashed
with `EXC_BAD_ACCESS` inside Apple's
`AGX executeCommandsInBufferCommon`. A later non-inherited variant avoided the
process crash but produced only the render-pass clear color, not the draw.

Current policy:

- keep ICB allocation probing only;
- do not execute `render_constant_rgba_direct_icb`;
- do not include ICB in default tests or benchmarks;
- revisit ICB only in a separate isolated harness, outside the Torch/MPS command
  buffer path, with a full Metal validation/debug capture.

Probable cause: the minimal CPU-encoded ICB state/resource contract was
incomplete or not valid for this render pipeline/command-buffer context. The
driver crash is still a hard stop for this variant; the safe engineering answer
is to use direct draw calls until a separate ICB harness proves correctness.

## Output format findings

Direct buffer-backed render targets over Torch MPS buffers validated for:

| Format | Torch shape/dtype | Bytes/pixel | Width multiple |
|---|---|---:|---:|
| `RGBA32Float` | `[H, W, 4] float32` | 16 | 16 |
| `RGBA16Float` | `[H, W, 4] float16` | 8 | 32 |
| `R32Float` | `[H, W] float32` | 4 | 64 |
| `RG32Float` | `[H, W, 2] float32` | 8 | 32 |

The common constraint is 256-byte row alignment for
`newTextureWithDescriptor:offset:bytesPerRow:` on the Torch MPS buffer.

There is no practical packed 3-channel direct render target for a contiguous
Torch tensor. Metal renderable formats are 1, 2, or 4 channel here, not
`RGB32Float`. Returning `[H, W, 4]` and ignoring alpha remains the simplest
mainline path. Lower-channel outputs only help if the next kernel can consume
one or two channels directly.

## Draw-stream recommendation

For the next mainline kernel:

1. Use direct render-to-MPS `RGBA32Float` first. It is proven and compatible
   with the existing RGB consumer by ignoring alpha.
2. Keep `RGBA16Float` as the first bandwidth-reduction experiment if acceptable
   numerically.
3. Use ordinary instanced draw calls for Gaussian quads before introducing ICB.
4. Avoid CPU draw loops for large Gaussian counts except as a debugging path.
5. Defer GPU-generated draw args/ICB until the fixed direct draw kernel is
   correct and benchmarked.
