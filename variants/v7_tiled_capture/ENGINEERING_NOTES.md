# Engineering notes for v7.2 tiled capture

## What was wrong in v7.1

The earlier v7.1 handoff did improve the *backward common path*, but it built front-K state with a pass that effectively did:

- launch `H*W` threads
- each thread loop over all `G` Gaussians

That reintroduced the exact scaling wall the redesign was supposed to remove.

## What changed here

### 1) Sparse tile bins
Gaussians are already depth sorted before forward. We preserve that order and append each Gaussian index into every tile touched by its alpha-threshold support box.

That gives each tile a contributor list in the exact same front-to-back order that the full scan would have seen.

### 2) Binned capture
`capture_front_k_binned` now scans the tile list for the current pixel's tile, not the full scene.

### 3) Binned overflow replay
`backward_overflow_replay_binned` also scans only the local tile list.

### 4) CPU aux state
The old v7.1 handoff copied saved front-K tensors back into MPS tensors in forward and then copied them back to CPU-backed Metal buffers in backward. This handoff keeps those aux tensors on CPU.

### 5) Packed metadata
`front_count` and `overflow_mask` are packed into one `uint8` tensor called `front_meta`.

## Why tile bins are exact here

The bin membership test uses the same threshold-support box math as the render quad generation:

- same alpha threshold
- same determinant clamp
- same half-width / half-height formulas

This means the tile list is a conservative exact candidate list for the thresholded visibility test.

Inside the tile-scoped kernels we still run the full pixel-local visibility test before accepting a contributor. So tile bins change the candidate set, not the visibility criterion.

## Known limits

- ids are `uint16`, so this handoff assumes `G <= 65536` per batch
- tile offset storage is `int32`; adversarial huge-support scenes can overflow total ref capacity
- tile bins are still built on CPU in this draft
- no Apple-hardware validation was possible here

## Practical implication

This version is the right answer to the specific criticism:

> keep v7.1's corrected backward math, but do not build front-K by scanning all splats per pixel

That is exactly what changed.
