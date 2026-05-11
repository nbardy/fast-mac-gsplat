# v13a_temporal_recompute_state

Opt-in fast-mac feature shader fork for testing exact backward-time
recomputation of fixedbin raster metadata on top of the v11 feature
grad-cache, zero-background, hostmeta, fixedbin variant.

## Lineage

- Copied from `variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`.
- Inherits the v8 host-side metadata split:
  `meta_i32/meta_f32` stay on MPS for shader args, while
  `meta_host_i32/meta_host_f32` are parsed by the bridge for allocation,
  validation, and dispatch sizes.
- Inherits fixed-capacity no-overflow binning: `binned_ids` is allocated as
  `tile_count * max_fast_pairs`, and the fork raises if any tile exceeds the
  cap.
- Python package: `torch_gsplat_bridge_v13a_temporal_recompute_state`.
- Custom op namespace: `torch.ops.gsplat_metal_v13a_temporal_recompute_state`.
- Metal source: `csrc/metal/gsplat_v13a_temporal_recompute_state_kernels.metal`.

## Intended Optimization

The fork keeps the parent grad-cache backward path for `F <= 32`, where each
pixel thread caches `grad_features[pix, :]`.

New in v13a: `RasterConfig.backward_state_strategy` accepts:

- `"save"`: inherited v11 behavior. Forward saves `active_tile_ids`,
  `tile_counts`, `tile_offsets`, `binned_ids`, and `tile_stop_counts` for
  backward.
- `"recompute"`: forward saves empty placeholders for those tile-state tensors.
  Backward reruns `bin` plus the selected forward-state kernel to reconstruct
  sorted `binned_ids` and `tile_stop_counts`, then calls the inherited saved-state
  backward kernel unchanged.

This reduces long-lived autograd saved state, especially the fixedbin
`binned_ids = tile_count * max_fast_pairs` tensor. It costs an extra bin pass
and an extra forward-state render during backward, so it is a memory valve, not
a speed path.

Inherited from v9: for forward fast and active tiles, Python sets metadata bit `2` when
`RasterConfig.background` is exactly zero for all active feature channels. The
Metal kernels then skip only the final `add_background_tail(...)` call. Empty or
invalid pixels still use the inherited explicit background/zero initialization.

Metadata bit `1` remains reserved for skipping color-gradient allocation in
backward, so backward checks must mask bit `1` rather than treating any reserved
bit as skip-color.

## Build

From the Dynaworld root:

```bash
( cd third_party/fast-mac-gsplat/variants/v13a_temporal_recompute_state
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

This fork is selectable through the shared Dynaworld fast-mac renderer as
`fast_mac.feature_variant="v13a_temporal_recompute_state"`. Existing configs
are unchanged because the default `backward_state_strategy` is `"save"`.

## Benchmark Switch

The direct benchmark exposes the strategy:

```bash
GSP_FAST_CAP=4096 GSP_FEATURE_CAP=64 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v13a_temporal_recompute_state/benchmarks/benchmark_mps.py \
  --height 128 --width 128 --gaussians 8192 --batch-size 2 \
  --feature-dim 32 --case medium_sigma_3_8 --backward --alpha-loss \
  --batch-strategy flatten --backward-state-strategy recompute --json
```
