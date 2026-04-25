# torch-metal-gsplat-v9-hw-interop

V9 starts with the missing gate from the hardware rasterizer plans: render-pass
output must land in Torch/MPS-visible storage without CPU staging.

This variant is intentionally not a Gaussian rasterizer yet. It proves the
interop primitive that imageblocks, ROG, and ICB would depend on:

1. allocate a Torch MPS tensor `[H,W,4] float32`;
2. preferably render a full-screen triangle directly into a buffer-backed
   `RGBA32Float` texture over the tensor storage;
3. optionally render into a private `RGBA32Float` texture and GPU-blit that
   texture into the MPS tensor's backing `MTLBuffer`;
4. return the tensor without `waitUntilCompleted`, `getBytes`, or CPU staging.

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
python benchmarks/benchmark_interop.py --sizes 64x64,512x512,1080x1920 --warmup 3 --iters 10 --paths blit,direct
```

## Current Scope

- Render pipeline: compiled and executed.
- Torch/MPS interop: implemented for direct `RGBA32Float` render target output
  into tensor storage when `width * 16` is 256-byte aligned; GPU blit remains a
  fallback probe.
- Tile/imageblock: compile probe only.
- Raster order groups: device feature probe only.
- ICB: allocation probe only.

The next useful step is a fixed eval renderer that writes Gaussian color through
the direct render-to-MPS path, still without imageblocks. Tile/imageblock state
should come after this interop path is stable under benchmark load.
