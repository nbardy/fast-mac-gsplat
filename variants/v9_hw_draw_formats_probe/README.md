# torch-metal-gsplat-v9-hw-draw-formats

V9 starts with the missing gate from the hardware rasterizer plans: render-pass
output must land in Torch/MPS-visible storage without CPU staging.

This variant is intentionally not a Gaussian rasterizer yet. It proves the
interop primitive that imageblocks and ROG would depend on:

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
python benchmarks/benchmark_interop.py --sizes 64x64,512x512,1080x1920 --warmup 3 --iters 10 --paths blit,direct,formats
```

Do not benchmark `--paths icb`. The benchmark now reports it as skipped, because
minimal ICB execution crashed inside Apple's AGX driver on this machine.

## Current Scope

- Render pipeline: compiled and executed.
- Torch/MPS interop: implemented for direct buffer-backed render target output
  into tensor storage for `RGBA32Float`, `RGBA16Float`, `R32Float`, and
  `RG32Float` when each row is 256-byte aligned; GPU blit remains a fallback
  probe for `RGBA32Float`.
- Output row alignment:
  - `RGBA32Float`: width multiple of 16.
  - `RGBA16Float` / `RG32Float`: width multiple of 32.
  - `R32Float`: width multiple of 64.
- Three-channel direct output: no native `RGB32Float` render target/Torch
  contiguous tensor layout was found. Use RGBA and ignore alpha, or use split
  lower-channel attachments/tensors only when the downstream math can consume
  them directly.
- Tile/imageblock: compile probe only.
- Raster order groups: device feature probe only.
- ICB: allocation probe only. Execution is disabled/fail-closed after a minimal
  execute path crashed with `EXC_BAD_ACCESS` in
  `AGX executeCommandsInBufferCommon`.

The next useful step is a fixed eval renderer that writes Gaussian color through
the direct `RGBA32Float` render-to-MPS path, still without imageblocks or ICB.
If bandwidth is the bottleneck and precision allows it, `RGBA16Float` is the
first lower-bandwidth target to test in the mainline kernel.
