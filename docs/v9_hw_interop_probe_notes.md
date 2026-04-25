# V9 Hardware Interop Probe Notes

## Why this exists

The v8 hardware plans were blocked by a lower-level requirement: a Metal render
pass must produce a Torch/MPS tensor without CPU staging. Until that is true,
tile shaders, imageblocks, raster order groups, and indirect command buffers
cannot become a real fast path for training or eval.

`variants/v9_hw_interop_probe` is the smallest test of that requirement.

## Implemented primitive

```text
out = torch.empty([H, W, 4], device="mps", dtype=float32)
texture = RGBA32Float render target

encode on PyTorch MPS command queue:
  preferred:
    make texture view over out.storage().MTLBuffer
    render fullscreen triangle directly into tensor storage
  fallback:
    render fullscreen triangle into private texture
    blit texture -> out.storage().MTLBuffer

return out
```

The native op does not call `waitUntilCompleted`, `getBytes`, `[buffer contents]`,
or create CPU tensors. Tests may read the result back after the op returns to
validate correctness.

## Advanced Metal feature status

Tested on Apple M4:

| Feature | V9 status | What it means |
|---|---:|---|
| Render pipeline | implemented | A real render pass writes an RGBA32F target. |
| Torch/MPS tensor buffer access | implemented | The op obtains the tensor backing `MTLBuffer`; direct render-target texture view works for aligned rows, with GPU blit fallback. |
| Tile/imageblock | compile probe passed | Minimal tile imageblock pipeline compiles; sample length is 24 B and 16x16 imageblock memory is 6144 B. |
| Raster order groups | device probe passed | `areRasterOrderGroupsSupported == true`; shader use is still pending. |
| ICB | allocation probe passed | The variant creates a one-command draw ICB, but does not execute it. |
| Render-to-MPS validation | passed | Blit path 9x7 and direct path 16x16 validate with max abs error `0.0`. |

## Interop Bandwidth Numbers

These timings measure constant render into a newly allocated Torch MPS tensor.
They are not Gaussian timings; they isolate the lower-level interop cost.

| Resolution | Blit Median | Direct Median | Notes |
|---:|---:|---:|---|
| 64x64 | 0.598 ms | 0.473 ms | Fixed overhead dominated. |
| 512x512 | 0.720 ms | 0.358 ms | Direct avoids the copy and wins. |
| 1080x1920 | 3.499 ms | 1.090 ms | Direct removes most of the copy cost. |
| 2160x3840 | 14.317 ms | 2.274 ms | The copy dominates the blit path. |
| 4096x4096 | 15.947 ms | 4.683 ms | Direct keeps 4K interop below v8-scale full render cost. |

The direct result changes the plan: a serious hardware eval path should render
directly into the MPS tensor's buffer-backed texture whenever `width * 16` is
valid for Metal row alignment. The blit path is useful as a fallback/probe, but
it is too expensive to be the default at 4K.

## Next Kernel Plan

### V9 Fixed Eval

Use the direct render-to-MPS tensor path, replacing the constant fragment shader
with Gaussian quad rendering.

```text
input:
  projected means2d/conics/colors/opacities/depths on MPS
  precomputed or GPU-generated draw instances

vertex:
  one quad per Gaussian
  expand mean/radius to screen-space triangle pair
  pass gaussian id and local pixel coordinates

fragment:
  q = conic quadratic at pixel
  alpha = opacity * exp(-0.5 * q)
  discard alpha < threshold
  output premultiplied RGB and alpha

render:
  depth order must match compute baseline
  blend approximates C += T * alpha * color and T *= (1 - alpha)
```

Kill gate: fixed blending must match eval tolerance and beat v8 compute after
including render setup and GPU blit.

### V9 Tile/Imageblock Eval

After fixed eval proves the interop path, move compositing into tile-local state:

```text
imageblock per pixel:
  float3 C
  float T
  uint stop

tile init:
  C = background
  T = 1
  stop = 0

fragment/update:
  if stop: skip
  alpha = gaussian contribution
  C += T * alpha * color
  T *= 1 - alpha
  if T < threshold: stop = 1

tile flush:
  write C to color attachment or output texture
  optional write final_T/stop_count state for backward
```

Key risks:

- imageblock memory grows with per-pixel state and tile dimensions;
- same-pixel ordering may require ROG and serialize hot pixels;
- training still needs exact enough saved state for backward.

### V9 ICB

ICB should wait until eval works. It can reduce CPU draw overhead once the
per-Gaussian draw format is settled, but it does not solve output interop or
backward state capture.
