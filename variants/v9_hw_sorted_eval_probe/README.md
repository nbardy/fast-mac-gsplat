# torch-metal-gsplat-v9-hw-sorted-eval

V9 fixed-eval starts with the missing gate from the hardware rasterizer plans:
render-pass output must land in Torch/MPS-visible storage without CPU staging.

This variant keeps the inherited interop probes and adds the first small
eval-only Gaussian render path:

1. allocate a Torch MPS tensor `[H,W,4] float32`;
2. preferably render a full-screen triangle directly into a buffer-backed
   `RGBA32Float` texture over the tensor storage;
3. optionally render into a private `RGBA32Float` texture and GPU-blit that
   texture into the MPS tensor's backing `MTLBuffer`;
4. return the tensor without `waitUntilCompleted`, `getBytes`, or CPU staging.
5. render simple screen-space Gaussian splats from MPS input tensors into the
   same RGBA32F MPS output tensor.
6. optionally depth-sort Gaussian inputs on MPS before submitting the same
   fixed-function render path.

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
python benchmarks/benchmark_interop.py --sizes 64x64,512x512,1080x1920 --warmup 3 --iters 10 --paths blit,direct,gaussian-direct
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

render_gaussian_eval_rgba_sorted(
    means2d,
    conics,
    colors,
    opacities,
    depths,  # MPS float32 [G]
    height,
    width,
    direct=True,
    descending=False,
)  # -> MPS float32 [H,W,4], after stable depth sort
```

`render_gaussian_eval_rgba_sorted(..., descending=False)` follows the v8
wrapper convention: it uses `torch.argsort(depths.detach(), stable=True)`, so
lower numeric depths are submitted first and equal-depth splats keep input
order. `descending=True` submits higher numeric depths first. The sort and
gathers run in Torch on MPS tensors; the native op still sees ordinary sorted
`means2d`, `conics`, `colors`, and `opacities`.

## Current Scope

- Render pipeline: compiled and executed.
- Torch/MPS interop: implemented for direct `RGBA32Float` render target output
  into tensor storage when `width * 16` is 256-byte aligned; GPU blit remains a
  fallback probe.
- Fixed eval Gaussian path: implemented as instanced screen-space quads. It
  reads `means2d`, `conics`, `colors`, and `opacities` directly from MPS tensor
  buffers, evaluates the Gaussian in the fragment shader, and writes
  premultiplied RGBA through hardware source-over blending.
- Sorted eval wrapper: implemented in Python/Torch with stable depth order,
  matching v8's ascending-depth permutation by default.
- Limitations: no backward pass, no tile/imageblock path, no exact v8 parity
  target, no batching, and direct output still requires aligned rows. Multiple
  Gaussians still blend through fixed hardware source-over in submitted order.
  Sorting alone does not recover v8's exact front-to-back transmittance math or
  output alpha semantics.
- Tile/imageblock: compile probe only.
- Raster order groups: device feature probe only.
- ICB: allocation probe only.

The next useful step is to compare this source-over fixed-eval path against the
v8 compute reference for sorted inputs, then decide whether exact parity needs
programmable per-pixel state via tile/imageblock/raster-order-group work.
