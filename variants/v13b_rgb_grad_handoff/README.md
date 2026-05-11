# v13b_rgb_grad_handoff

Opt-in fast-mac feature shader fork for testing an RGB-gradient handoff
boundary on top of the v11 F32 grad-cache, zero-background, host-metadata, and
fixedbin feature rasterizer.

## Lineage

- Copied from `variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`.
- Inherits the v8 host-side metadata split:
  `meta_i32/meta_f32` stay on MPS for shader args, while
  `meta_host_i32/meta_host_f32` are parsed by the bridge for allocation,
  validation, and dispatch sizes.
- Inherits fixed-capacity no-overflow binning: `binned_ids` is allocated as
  `tile_count * max_fast_pairs`, and the
  fork raises if any tile exceeds the cap.
- Python package: `torch_gsplat_bridge_v13b_rgb_grad_handoff`.
- Custom op namespace: `torch.ops.gsplat_metal_v13b_rgb_grad_handoff`.
- Metal source: `csrc/metal/gsplat_v13b_rgb_grad_handoff_kernels.metal`.

## Intended Optimization

The normal raster API is currently v11-compatible. It still accepts dense
`grad_features[B,H,W,F]` and `grad_alpha[B,H,W]` in backward, so it is runnable
as a renamed independent fork.

The v13b-specific addition is a scaffolded low-level op:

```python
rgb_grad_handoff_backward(...)
```

The intended handoff path is:

1. raster forward-state produces `out_features[B,H,W,F]` and `out_alpha[B,H,W]`
2. an RGB image-space objective produces `grad_composed_rgb[B,H,W,3]`
3. the missing Metal kernel computes sigmoid-linear colorizer VJP per pixel
4. the same kernel streams feature/alpha gradients directly into reverse raster
   accumulation without allocating `grad_features[B,H,W,F]`

The C++ op is registered but intentionally raises until the Metal kernel is
implemented. The helper `estimate_rgb_grad_handoff_memory(...)` and
`benchmarks/rgb_grad_handoff_accounting.py` provide the bandwidth accounting for
the missing kernel boundary.

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
( cd third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

This fork is selectable through the shared Dynaworld fast-mac renderer as
`fast_mac.feature_variant="v13b_rgb_grad_handoff"`. That selects the
v11-compatible raster API only; it does not enable the scaffolded RGB-gradient
handoff kernel.

## Accounting Prototype

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
  python third_party/fast-mac-gsplat/variants/v13b_rgb_grad_handoff/benchmarks/rgb_grad_handoff_accounting.py \
  --batch 16 --height 256 --width 256 --feature-dim 32
```
