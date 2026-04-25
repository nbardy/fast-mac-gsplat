# torch-metal-gsplat-v9-hw-eval-parity

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

V8 parity/benchmark harness:

```bash
cd variants/v9_hw_eval_parity_probe
(cd ../v8_hw_eval && python3 setup.py build_ext --inplace)
python3 setup.py build_ext --inplace
python3 benchmarks/benchmark_parity_v8.py \
  --sizes 16x16,64x64 \
  --gaussians 1,16 \
  --cases tiny_single,grid_ordered,overlap_ordered \
  --warmup 2 \
  --iters 5 \
  --jsonl ../../benchmarks/v9_hw_eval_parity_smoke.jsonl \
  --markdown ../../benchmarks/v9_hw_eval_parity_smoke.md
```

Tiny parity smoke:

```bash
python3 tests/parity_v8_smoke.py
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
```

## Current Scope

- Render pipeline: compiled and executed.
- Torch/MPS interop: implemented for direct `RGBA32Float` render target output
  into tensor storage when `width * 16` is 256-byte aligned; GPU blit remains a
  fallback probe.
- Fixed eval Gaussian path: implemented as instanced screen-space quads. It
  reads `means2d`, `conics`, `colors`, and `opacities` directly from MPS tensor
  buffers, evaluates the Gaussian in the fragment shader, and writes
  premultiplied RGBA through hardware source-over blending.
- Limitations: no backward pass, no tile/imageblock path, no depth sort, no
  batching, no transmittance early termination, and direct output still
  requires aligned rows. Multi-splat hardware blend order is not currently a
  v8-equivalent ordering contract.
- Tile/imageblock: compile probe only.
- Raster order groups: device feature probe only.
- ICB: allocation probe only.

The parity harness compares v9's premultiplied RGB output to v8 forward eval
only under explicit assumptions: batch size 1, black background, projected
pixel-space inputs, and a single Gaussian. Multi-splat rows are diagnostic:
they use the same projected tensors, but current v9 does not reproduce v8's
depth/tile ordering. The optional `depth_mismatch` case is an order diagnostic,
not a parity claim.
