# v8_project3d engineering notes

This fork starts from v5 and adds a Metal pinhole projection op. The projection
forward runs in Metal. The training path is hybrid: raster backward uses the
copied v5 Metal kernels, then projection gradients are propagated by
rematerializing the pinhole projection in Torch autograd.
The inherited notes below describe the copied projected-2D rasterizer.

This revision folds in the actual v3 field result and adds the next major feature: batched rendering.

## Main changes

1. **Train / eval split**
   - eval forward does not write back sorted IDs
   - train forward writes sorted IDs back into `binned_ids` and returns per-tile stop counts

2. **True batched API**
   - inputs may be `[G,...]` or `[B,G,...]`
   - outputs are `[H,W,3]` or `[B,H,W,3]`
   - Metal path flattens batch into one launch
   - wrapper can chunk large batches automatically

3. **Auto batch chunking**
   - `batch_strategy=auto|flatten|serial`
   - `auto` uses launch limits on tiles and gaussians

4. **Shared fast path behavior**
   - training path reuses saved sorted tile order in backward
   - backward only replays the deepest per-tile prefix touched by any pixel

## Expected wins

- training keeps the v3 saved-sort gain
- eval should recover the small forward regression caused by sorted-ID writeback
- batchwise rendering should improve throughput for small/medium B while preserving a serial fallback for memory pressure

## What to test first

1. small CPU reference for B=1 and B=2
2. eval-vs-train forward timing at B=1
3. B=1 / B=2 / B=4 / B=8 throughput with `auto`, `flatten`, `serial`
4. overflow stress with clustered large splats
5. 8/16/32 tile ablations only after occupancy / pair stats say tile shape matters

## Project3D Training Status

`rasterize_pinhole_gaussians(...)` now supports gradients through:

- `means3d`
- `scales`
- `quats`
- `opacities`
- `colors`
- `fx/fy/cx/cy`
- `camera_to_world`

The projection backward is intentionally implemented as a Torch
rematerialization first, not a hand-derived Metal kernel. This makes the path
usable for forward/backward benchmarking and training experiments while keeping
the pure-Metal backward decision dependent on measured bottlenecks.
