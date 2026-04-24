# torch-metal-gsplat-v72-tiled-capture

This is a follow-on source handoff that fixes the **main v7.1 forward bottleneck**:

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

The next obvious upgrade, if the Metal path validates, is to move tile binning off the CPU and into a proper GPU binning path so the source handoff stops paying for CPU packaging.

## Files to look at

- `csrc/metal/gsplat_v72_sparse.metal`
- `csrc/metal/gsplat_sparse.mm`
- `torch_gsplat_bridge_v72/reference.py`
- `docs/v72_sparse_capture_notebook.xml`
- `docs/v72_validation.md`
