# v10_features_gradcache_zero_bg_hostmeta

Opt-in fast-mac feature shader fork for testing the v8 host-side metadata
bridge path on top of the v9 feature grad-cache plus zero-background variant.

## Lineage

- Copied from `variants/v9_features_gradcache_zero_bg`.
- Ports only the v8 host-side metadata split where applicable:
  `meta_i32/meta_f32` stay on MPS for shader args, while
  `meta_host_i32/meta_host_f32` are parsed by the bridge for allocation,
  validation, and dispatch sizes.
- Python package: `torch_gsplat_bridge_v10_features_gradcache_zero_bg_hostmeta`.
- Custom op namespace: `torch.ops.gsplat_metal_v10_features_gradcache_zero_bg_hostmeta`.
- Metal source: `csrc/metal/gsplat_v10_features_gradcache_zero_bg_hostmeta_kernels.metal`.

## Intended Optimization

The fork keeps the parent grad-cache backward path for `F <= 32`, where each
pixel thread caches `grad_features[pix, :]`.

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
( cd third_party/fast-mac-gsplat/variants/v10_features_gradcache_zero_bg_hostmeta
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

This fork is not wired into checked-in trainer configs here.
