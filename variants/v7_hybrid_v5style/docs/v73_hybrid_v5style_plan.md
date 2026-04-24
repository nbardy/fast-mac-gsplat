# v7.3 Hybrid V5-Style Plan

Goal: keep the v7.2 hardware rasterization path, but recover the fast-system
shape that makes v5 competitive: GPU-resident binning, bounded batch launches,
saved state for training, and fallback work only where overflow actually occurs.

## Stage 0: Current Copy

Done in `variants/v7_hybrid_v5style`:

- copied v7.2 tiled capture into a separate v7.3 namespace
- preserved the v7.2 hardware raster plus binned front-K capture kernels
- added v5-style `batch_strategy` and batch launch limits around the hardware path
- added benchmark names `v73_hybrid_k2`, `v73_hybrid_k4`, `v73_hybrid_k8`, with optional tile and batch strategy suffixes
- added `forward_eval`, a hardware-raster eval op that skips capture/state
- added `train_backend=auto|v5_compute|hardware`; default training uses v5 compute when available

This stage is now a practical hybrid for training: default F+B uses the v5
compute path, while `_hwtrain` keeps the v7.2-style hardware-backward path for
experiments.

Smoke results after this stage:

| Case | v7.2 k2 | v7.3 hybrid k2 | Delta |
|---|---:|---:|---:|
| `512x512 / 6k / B=4` uniform F+B | 52.203 ms | 49.134 ms | -5.9% |
| `512x512 / 6k / B=4` clustered F+B | 92.820 ms | 94.950 ms | +2.3% |
| `4096x4096 / 64k / B=1` uniform F+B | 412.732 ms | 451.550 ms | +9.4% |

The same 4K smoke measured `v5_batched` at 63.273 ms, so v7.3 hybrid was
613.6% slower than v5 in that cell. That confirms the next fix must remove CPU
binning/readback rather than just tuning the wrapper.

After adding train/eval routing:

| Case | v5 | v7.2 k2 | v7.3 hybrid | v7.3 `_hwtrain` |
|---|---:|---:|---:|---:|
| `512x512 / 6k / B=4` uniform F+B | 23.135 ms | 48.242 ms | 34.381 ms | 48.579 ms |
| `512x512 / 6k / B=4` clustered F+B | 141.302 ms | 95.728 ms | 91.297 ms | 95.394 ms |
| `4096x4096 / 64k / B=1` uniform F+B | 73.681 ms | 438.782 ms | 76.464 ms | 436.552 ms |

The 4K training probe is now within 3.8% of v5 and 82.6% faster than v7.2.

## Stage 1: Hardware Eval Path

Use v7.2 hardware rasterization as the forward/eval path when gradients are not
needed. This stage is partially done: `forward_eval` skips front-K capture, tile
bin construction, and saved-state allocation.

Required fixes:

- skip front-K capture entirely for no-grad eval: done
- avoid CPU `out_cpu` creation in eval: still open; texture readback remains
- keep `front_k` only for training mode: done for eval routing

Expected impact:

- meaningful forward-only speedup versus v7.2 hardware capture
- still slower than v5 at 4K until output readback is removed

## Stage 2: GPU Binning

Replace the CPU tile bin builder with a v5-style MPS-resident bin path:

- count tile references on MPS
- prefix-sum offsets on MPS
- emit binned IDs on MPS
- pass those bins to v7.2 capture/overflow kernels without rebuilding them on CPU

The current v7.2 code copies projected Gaussian records to CPU, builds bins on
CPU, then uploads bins back to Metal buffers. That is the main reason v7.2 still
loses at 4K after removing the old `pixels x splats` scan.

## Stage 3: GPU Saved State

Training cannot be fast while forward saves `out_cpu`, `front_ids`,
`front_raw_alpha`, and `front_meta` as CPU tensors. The practical fix is now in
place: default training routes to v5 compute when available.

Two viable designs:

- expose GPU-resident saved-state tensors and let backward consume them directly
- use the v5 compute training path for backward-bearing calls, reserving hardware raster for eval: done

If PyTorch/MPS interop makes Metal texture/buffer ownership awkward, the second
option is probably the pragmatic fastest path.

## Stage 4: V5-Style Stop Counts

v5 saves per-tile stop counts instead of replaying every possible contributor.
The v7.2 front-K state is exact, but at high resolution it scales with pixels.

Next training kernel should prefer:

- per-tile sorted IDs
- per-tile stop counts
- replay only the stopped prefix in backward
- keep front-K only where it demonstrably reduces work

## Stage 5: Overflow-Only Replay

v7.2 already distinguishes common front-K pixels from overflow pixels, but the
big-scene path still pays broad per-pixel state costs. The v5 shape is better:

- fast path handles normal tiles
- overflow tile list is compact
- overflow forward/backward only run on those tiles
- gradients for overflow tiles are zeroed from the fast path before exact replay

## Recommendation

Yes, hardware rasterization can still get there for forward/eval. For training,
it only starts beating v5/v6 if the next version removes CPU binning and CPU
saved-state traffic. If that is not feasible, the best hybrid is explicit:

- v7.2-style hardware raster for no-grad/eval forward
- v5/v6-style compute path for training/backward
- shared GPU binning and tile metadata between both paths
