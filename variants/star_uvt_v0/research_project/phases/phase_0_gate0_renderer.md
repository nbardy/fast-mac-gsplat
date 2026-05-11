# Phase 0: Projected UVT Renderer

## Goal

Prove that already-projected `ScreenTimeTube` data can be binned into UVT tiles
and rendered in Metal with parity against a brute-force reference.

## In Scope

- Fixed-capacity per-tile buffers.
- Atomic tile append for tube IDs and center depths.
- Local tile sort for stable tiles.
- Deterministic per-sample fallback for unstable depth-order tiles.
- Pair-count reporting against a summed per-frame tile-splat baseline.

## Out of Scope

- Camera projection.
- HexGaussian or world-space expansion.
- Backward pass.
- Production trainer integration.

## Gate

`python3 tests/gate0_check.py` must pass with zero overflow and explicit
max/mean RGB parity stats on all tiny scenes.

