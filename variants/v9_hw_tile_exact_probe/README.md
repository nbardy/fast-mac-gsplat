# torch-metal-gsplat-v9-hw-tile-exact

V9 tile-state starts from the working render-pass interop gate and probes the
next question: whether Metal tile shaders/imageblocks can hold per-pixel
Gaussian compositing state (`C`, `T`, stop metadata) in the render pass.

This variant is intentionally still a probe, not a full Gaussian rasterizer. It
proves and measures the primitives that imageblocks, ROG, and later ICB would
depend on:

1. allocate a Torch MPS tensor `[H,W,4] float32`;
2. preferably render a full-screen triangle directly into a buffer-backed
   `RGBA32Float` texture over the tensor storage;
3. optionally render into a private `RGBA32Float` texture and GPU-blit that
   texture into the MPS tensor's backing `MTLBuffer`;
4. compile several tile/imageblock layouts for `C/T/stop`;
5. dispatch a tile shader inside a render pass whose color attachment is the
   direct Torch MPS target;
6. run the minimal exact imageblock overlap path:
   `tile clear -> ordered fragment C/T update + atomic tile stop -> tile report -> tile resolve`;
7. run an ordered Gaussian fragment path through the same exact imageblock
   `C/T` state and atomic tile stop-count output;
8. return tensors without native `waitUntilCompleted`, `getBytes`, or CPU
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
python3 tests/full_backward_check.py
```

## Benchmark

```bash
python3 benchmarks/benchmark_interop.py --sizes 64x64,512x512,1080x1920 --warmup 3 --iters 10 --paths blit,direct,exact,gaussian
python3 benchmarks/benchmark_full_backward.py --height 512 --width 512 --gaussians 4096 --warmup 2 --iters 5
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
- Tile exact overlap: validates two ordered constant-alpha splats using
  explicit imageblock `C/T` state and blending disabled. Expected output is
  `float4(0.25, 0.375, 0.0, 0.375)` and the smoke test reports
  `tile_exact_overlap_max_abs_err=0.0`.
- Tile exact Gaussian: validates four ordered Gaussian splats using the same
  explicit imageblock `C/T` path, direct MPS output, and GPU-written tile
  stop-count tensor. The smoke test compares the render output against a CPU
  Gaussian reference.
- Backward-state gate: the exact overlap path now returns GPU-written MPS
  `tile_stop_counts` (`int32`, one value per tile) plus debug `tile_reports`.
  The 32x32 / 16x16-tile smoke reports `tile_stop_counts=[2, 2, 2, 2]`.
- Full backward base: `rasterize_projected_gaussians_full_backward` is wired and
  gradient-checked through the built V8/V8-hw-eval compute replay backend. This
  is the correct training fallback while the hardware-raster state producer is
  still a probe.
- Raster order groups: device feature probe only.
- ICB: allocation probe only.

On Apple M4, `C/T + stop_count + flags` compiles with a 48 B imageblock sample:
12 KiB for a 16x16 tile and 48 KiB for a 32x32 tile. A 32x32 execution dispatch
reports a 32x32 imageblock footprint.

The exact init/update/report/flush shader sequence now works for a 16x16-tile
constant-overlap case. Metal treats tile kernels as explicit-layout
imageblocks, so clear uses `imageblock.data(...)`; fragment update and tile
resolve use `[[imageblock_data]]` structs with `[[raster_order_group(0)]]`
members. The 32x32 exact-overlap path compiles but failed render-encoder
creation on Apple M4 with the 48 B/sample state, so the API is intentionally
fail-closed to 16x16.

Tile shaders cannot declare their own threadgroup scratch arrays on this path,
so tile-level stop is updated by fragment-side atomic max into an MPS buffer.
That matches the eventual V8-shaped state tensor better than a tile-local toy
reduction, but it still only counts visible fragments in this probe.

This still does not prove V8 parity. Next work is replacing the fullscreen
Gaussian diagnostic draw with clipped projected Gaussian quads, then feeding the
same path from GPU-resident V8 tile bins.

Full backward is currently done by compute replay, not hardware-raster replay.
That is deliberate: the hardware path cannot own training gradients until it
emits the same sorted bins and candidate-prefix stop counts that V8 backward
consumes.
