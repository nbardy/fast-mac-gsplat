# v9_features_gradcache_zero_bg

Opt-in fast-mac feature shader fork for testing two feature-channel optimizations
without mutating stable variants.

## Lineage

- Copied from `variants/v6_refined_features_f32_gradcache`.
- Ports only the zero-feature-background tail-skip behavior from
  `variants/v6_refined_features_f32_zero_bg`.
- Python package: `torch_gsplat_bridge_v9_features_gradcache_zero_bg`.
- Custom op namespace: `torch.ops.gsplat_metal_v9_features_gradcache_zero_bg`.
- Metal source: `csrc/metal/gsplat_v9_features_gradcache_zero_bg_kernels.metal`.

## Intended Optimization

The fork keeps the parent grad-cache backward path for `F <= 32`, where each
pixel thread caches `grad_features[pix, :]`.

For forward fast and active tiles, Python sets metadata bit `2` when
`RasterConfig.background` is exactly zero for all active feature channels. The
Metal kernels then skip only the final `add_background_tail(...)` call. Empty or
invalid pixels still use the inherited explicit background/zero initialization.

Metadata bit `1` remains reserved for skipping color-gradient allocation in
backward, so backward checks must mask bit `1` rather than treating any reserved
bit as skip-color.

## Build

From the Dynaworld root:

```bash
( cd third_party/fast-mac-gsplat/variants/v9_features_gradcache_zero_bg
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

This fork is not wired into checked-in trainer configs here.
