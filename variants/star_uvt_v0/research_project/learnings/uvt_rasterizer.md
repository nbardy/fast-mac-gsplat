# UVT Rasterizer Learnings

## Keep The First Primitive Normal

The useful first object is a projected `ScreenTimeTube`, with the later
world-space `WorldTube` projection feeding the same renderer tensors. Starting
with HexGaussian would mix projection math and tile renderer risk too early.

## Separate Gradient Ownership From Forward Parity

Gate 0 only proves forward compositing and tile-pair behavior. Training can
start with a dense differentiable PyTorch renderer while Metal remains the
forward parity and stats path.

## Track Unstable Tiles Explicitly

Center-depth tile sorting is only valid when all pairwise depth orderings remain
stable over the tile. Unstable tiles need deterministic per-sample depth order
until a stronger ordering proof or split strategy exists.

## Warm Up Before Timing Backward

MPS and Metal first-use costs can dominate tiny backward measurements. Backward
benchmarks should report warmup iterations separately from measured iterations,
and tiny-scene timing must not be used as promotion evidence.
