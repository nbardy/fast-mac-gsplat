# torch-metal-gsplat-v9-hw-output-planes

V9 fixed-eval starts with the missing gate from the hardware rasterizer plans:
render-pass output must land in Torch/MPS-visible storage without CPU staging.

This variant keeps the inherited interop probes and adds the first small
eval-only Gaussian render path:

1. allocate a Torch MPS tensor `[H,W,4] float32`;
2. preferably render a full-screen triangle directly into a buffer-backed
   `RGBA32Float` or `RGBA16Float` texture over the tensor storage;
3. optionally render into a private `RGBA32Float` texture and GPU-blit that
   texture into the MPS tensor's backing `MTLBuffer`;
4. return the tensor without `waitUntilCompleted`, `getBytes`, or CPU staging.
5. render simple screen-space Gaussian splats from MPS input tensors into
   RGBA32F or RGBA16F MPS output tensors.

The Python validation reads the output back to CPU only after the op returns, so
that the test can compare values. The native op itself does not read GPU data on
the CPU.

## Build

```bash
python setup.py build_ext --inplace
```

## Check

```bash
python tests/interop_check.py
```

## Benchmark

```bash
python3 benchmarks/benchmark_interop.py \
  --sizes 512x512,1080x1920,4096x4096 \
  --warmup 5 \
  --iters 20 \
  --paths formats,gaussian-direct-rgba32f,gaussian-direct-rgba16f \
  --formats rgba32f,rgba16f,r32f,rg32f \
  --gaussians 6000
```

## Eval API

```python
render_gaussian_eval_rgba(
    means2d,   # MPS float32 [G,2], pixel-space x/y
    conics,   # MPS float32 [G,3], packed a,b,c inverse covariance
    colors,   # MPS float32 [G,3]
    opacities,  # MPS float32 [G]
    height,
    width,
    direct=True,
)  # -> MPS float32 [H,W,4], premultiplied RGB plus alpha

render_gaussian_eval_format(
    "rgba16f",
    means2d,
    conics,
    colors,
    opacities,
    height,
    width,
    direct=True,
)  # -> MPS float16 [H,W,4], premultiplied RGB plus alpha
```

## Current Scope

- Render pipeline: compiled and executed.
- Torch/MPS interop: implemented for direct `RGBA32Float`, `RGBA16Float`,
  `R32Float`, and `RG32Float` render target output into tensor storage when row
  bytes are 256-byte aligned; GPU blit remains a fallback probe for RGBA
  Gaussian paths.
- Fixed eval Gaussian path: implemented as instanced screen-space quads. It
  reads `means2d`, `conics`, `colors`, and `opacities` directly from MPS tensor
  buffers, evaluates the Gaussian in the fragment shader, and writes
  premultiplied RGBA through hardware source-over blending. RGBA32F and RGBA16F
  Gaussian output formats both compile, run, and validate.
- Limitations: no backward pass, no tile/imageblock path, no depth sort, no
  v8 parity target, no batching, and direct output still requires aligned rows.
  Multiple Gaussians blend in input order.
- Row alignment: direct buffer-backed texture rows must be 256-byte aligned.
  Width multiples are 16 pixels for RGBA32F, 32 for RGBA16F, 64 for R32F, and
  32 for RG32F.
- Tile/imageblock: compile probe only.
- Raster order groups: device feature probe only.
- ICB: allocation probe only.

The next useful step is to wire RGBA16F into the parity-shaped fixed-eval route
and compare image error against the RGBA32F output before making it the default.
