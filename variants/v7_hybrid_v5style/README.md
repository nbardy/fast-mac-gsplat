# torch-metal-gsplat-v73-hybrid-v5style

This is a v7.3 hybrid branch copied from the v7.2 tiled-capture handoff. It
keeps the v7.2 hardware raster/capture path, then starts moving the Python API
and launch policy toward the v5 shape.

Current routing:

- no-grad/eval calls use `gsplat_metal_v73.forward_eval`, a hardware-raster
  path that skips front-K capture and backward saved-state construction
- gradient/training calls default to `train_backend="auto"`, which uses the
  sibling v5 compute training path when available
- set `train_backend="hardware"` or use benchmark suffix `_hwtrain` to force
  the v7.2-style hardware backward path

The inherited v7.2 source fixes the **main v7.1 forward bottleneck**:

- old v7.1 capture pass: **one thread per pixel, each pixel scans all splats**
- new v7.2 capture pass: **one thread per pixel, each pixel scans only the bin for its screen tile**

The core change is sparse visibility plumbing:

1. Depth-sorted projected Gaussians are binned into screen tiles on the CPU side during forward packaging.
2. `capture_front_k_binned` uses those tile bins instead of scanning the full scene.
3. `backward_overflow_replay_binned` also replays only the local tile bin, not all splats.
4. Saved front state is kept on **CPU tensors** for autograd context so forward does not copy it back to MPS only to copy it to CPU again in backward.
5. Saved-state memory is reduced with:
   - `front_ids`: `uint16` instead of `int32`
   - `front_meta`: packed `uint8` instead of separate `int32 front_count` + `uint8 overflow_mask`

## What this does and does not claim

What changed:

- fixes the `pixels × splats` capture shape that made big scenes explode
- keeps the v7.1 front-K backward math and overflow split
- adds exact sparse tile-bin replay for overflow pixels
- cuts saved-state bandwidth and aux tensor traffic

What this does **not** claim:

- I did **not** validate the Metal path on Apple hardware in this environment
- CPU packaging and CPU-built tile bins are still a real cost center
- this is a source handoff meant to replace the obviously bad `pixels × splats` pass, not a fully benchmarked winner

## New internal ABI

Forward now returns:

- `out` on the caller device
- `packed_gaussians` on CPU
- `out_cpu` exact forward image on CPU
- `front_ids` on CPU, shape `[B,H,W,K]`, dtype `uint16`
- `front_raw_alpha` on CPU, shape `[B,H,W,K]`, dtype `float32`
- `front_meta` on CPU, shape `[B,H,W]`, dtype `uint8`
- `tile_offsets` on CPU, shape `[B,T+1]`, dtype `int32`
- `tile_ids` on CPU, shape `[R]`, dtype `uint16`

`front_meta` layout:

- low 4 bits: captured count
- high bit: overflow flag

## Why this fixes the bad scaling

Old v7.1 capture complexity:

- `O(B * H * W * G)`

New v7.2 capture complexity:

- `O(total_tile_refs + sum_pixels bin_size(tile(pixel)))`

If bins stay sparse, the common path behaves like:

- `pixels × average_bin_size`

instead of

- `pixels × all_splats`

That is the specific bottleneck you called out.

## Remaining next-step work

This branch now adds v5-style `batch_strategy` launch chunking around the
hardware path and routes training through v5 compute by default. The next
hardware-specific performance upgrade is still to remove the output texture
readback and expose the hardware raster output as a GPU-resident torch tensor.

See `docs/v73_hybrid_v5style_plan.md` for the staged plan.

## Files to look at

- `csrc/metal/gsplat_v73_sparse.metal`
- `csrc/metal/gsplat_sparse.mm`
- `torch_gsplat_bridge_v73/reference.py`
- `docs/v73_sparse_capture_notebook.xml`
- `docs/v73_validation.md`
