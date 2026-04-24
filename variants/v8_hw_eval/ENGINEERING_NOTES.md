
# v8 engineering notes

v8 keeps the v5 batch-native API and pushes three concrete changes into source:

1. **Active-tile compaction**
   - build `active_tile_ids = nonzero((tile_counts > 0) & (tile_counts <= max_fast_pairs))`
   - optionally sort active fast tiles by tile count before dispatch
   - Metal forward/backward kernels dispatch only those active fast tiles

2. **No unconditional grad clone**
   - backward no longer clones the full `[B,H,W,3]` grad image just to zero overflow tiles
   - fast backward consumes `grad_out.contiguous()` directly
   - overflow backward gathers only the overflow tile images from the original grad image

3. **Adaptive stop counts**
   - `stop_count_mode=always|never|adaptive`
   - `adaptive` only records stop counts for tiles with `tile_count >= stop_count_dense_threshold`
   - light tiles use `tile_stop_counts[tile_id] = tile_counts[tile_id]`, so backward replays the exact full tile without extra forward bookkeeping

## Benchmark asks for engineering

Run on a quiet machine and report median/min/p95 in addition to mean.

1. `B=1,2,4,8`, 4K, 65536 splats, sparse / medium / overflow_stress
2. compare `stop_count_mode=always` vs `adaptive` vs `never`
3. compare `sort_active_tiles_by_count=True/False`
4. compare `batch_strategy=auto/flatten/serial`
5. profile with Xcode GPU counters for memory bandwidth and atomic stalls

## Success criterion

v8 does **not** need to beat v3 at `B=1`.

It wins if it materially improves **batched forward+backward per frame** while preserving the v5 reference-check envelope.
