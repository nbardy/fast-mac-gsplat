# torch-metal-gsplat-v3

A Torch-first, Metal-hot-path projected 2D Gaussian rasterizer for Apple Silicon.

## Design goals

- keep the Python training loop and autograd in Torch
- move the raster hot path onto Metal / MPS
- avoid dense `[G,H,W]` activations
- use exact front-to-back alpha compositing
- keep backward low-memory by recomputation
- use a fast tile-local path for normal tiles and a slower overflow path for pathological tiles

## Current architecture

1. Python wrapper stable-sorts splats by detached depth on MPS.
2. Metal bins sorted-rank IDs into screen tiles using opacity-aware SnugBox + exact tile/ellipse intersection.
3. `torch.cumsum` on MPS builds tile offsets.
4. Metal emits compact tile bins of 32-bit sorted-rank IDs.
5. Fast path:
   - tiles with `count <= max_fast_pairs` are locally bitonic-sorted in threadgroup memory
   - chunks of Gaussian parameters are staged into threadgroup memory
   - forward uses exact front-to-back alpha blending
   - backward recomputes local alpha and uses a Gaussian-centric reduction to cut atomics
6. Overflow path:
   - wrapper extracts only the overflow tile segments
   - per-tile ID sort is done with Torch MPS ops for those rare tiles
   - dedicated Metal kernels render and backprop only overflow tiles

## Important caveats

- projected 2D Gaussian interface only; 3D camera projection lives outside this package
- gradients w.r.t. depth ordering are zero by design
- this uses internal PyTorch MPS Metal APIs, so PyTorch version pinning matters
- the v3 rewrite is a best-effort source handoff and should be compiled/validated on the target Mac

## Why this version exists

This v3 package incorporates lessons from earlier iterations:

- do not treat Torch `[G,3]` tensors as `float3*` in Metal; load packed triples manually
- sort depths once in Torch, then sort tile bins by sorted-rank IDs rather than fatter keys
- use 256 threads per tile to match 16x16 pixel tiles and leave more occupancy than 1024-thread groups
- use threadgroup parameter staging to attack bandwidth at 4K
- use recompute backward with Gaussian-centric reduction to cut global atomics
- keep an overflow path instead of forcing a giant fast-tile cap everywhere

## Local status

This extracted handoff now builds locally after a small `setup.py` path fix and
passes `tests/reference_check.py` on MPS. In the repository-level
`benchmarks/compare_v2_v3.py` script, it beat the validated v2 fastpath on
4096x4096 / 65,536 projected synthetic splats:

- forward: `12.410 ms` vs `15.506 ms` on sparse sigma 1-5 px
- forward: `13.702 ms` vs `24.935 ms` on medium sigma 3-8 px
- forward+backward: `47.872 ms` vs `70.654 ms` on sparse sigma 1-5 px
- forward+backward: `60.738 ms` vs `134.162 ms` on medium sigma 3-8 px

The fast forward path now writes its tile-local sorted order back into
`binned_ids`, so fast backward can skip its duplicate local bitonic sort. See
`../../docs/v3_saved_order_ablation.md` for the measured ablation.
