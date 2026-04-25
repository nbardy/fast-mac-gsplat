# torch-metal-gsplat-v9-hw-tile-state

V9 tile-state starts from the working render-pass interop gate and probes the
next question: whether Metal tile shaders/imageblocks can hold per-pixel
Gaussian compositing state (`C`, `T`, stop metadata) in the render pass.

This variant is intentionally not a Gaussian rasterizer yet. It proves and
measures the primitives that imageblocks, ROG, and later ICB would depend on:

1. allocate a Torch MPS tensor `[H,W,4] float32`;
2. preferably render a full-screen triangle directly into a buffer-backed
   `RGBA32Float` texture over the tensor storage;
3. optionally render into a private `RGBA32Float` texture and GPU-blit that
   texture into the MPS tensor's backing `MTLBuffer`;
4. compile several tile/imageblock layouts for `C/T/stop`;
5. dispatch a tile shader inside a render pass whose color attachment is the
   direct Torch MPS target;
6. return tensors without native `waitUntilCompleted`, `getBytes`, or CPU
   staging.

The Python validation reads the output back to CPU only after the op returns, so
that the test can compare values. The native op itself does not read GPU data on
the CPU.

## Build

```bash
python3 setup.py build_ext --inplace
```

## Check

```bash
python3 tests/interop_check.py
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
- Tile/imageblock: compiles four layouts and reports exact
  `imageblockSampleLength` plus 8x8/16x16/32x32 memory.
- Tile execution: dispatches `dispatchThreadsPerTile` in a render pass on the
  direct MPS target and returns an MPS tile-report tensor.
- Raster order groups: device feature probe only.
- ICB: allocation probe only.

On Apple M4, `C/T + stop_count + flags` compiles with a 48 B imageblock sample:
12 KiB for a 16x16 tile and 48 KiB for a 32x32 tile. A 32x32 execution dispatch
reports a 32x32 imageblock footprint.

The exact init/update/flush shader sequence is still risky. Metal treats these
structs as explicit-layout imageblocks: `imageblock.read()` is unavailable, and
the probe uses `imageblock.data(coord, ..., imageblock_data_rate::color)` for
state writes. A prior same-shader readback attempt after an imageblock barrier
returned zero, so this variant proves compile plus dispatch, not a complete
tile-local compositing loop yet.
