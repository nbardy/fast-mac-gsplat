# v12c_fused_raster_color_loss_backward

Opt-in fast-mac feature shader fork for prototyping fused raster/color/loss
backward. The inherited raster API remains intact, and the new prototype path
adds a fast-tile no-overflow Metal op that computes linear-sigmoid colorize,
RGB alpha composition, and mean-MSE pixel gradients inside raster backward
without materializing a dense `grad_features[B,H,W,F]` buffer.

## Lineage

- Copied from `variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`.
- Inherits the v8 host-side metadata split:
  `meta_i32/meta_f32` stay on MPS for shader args, while
  `meta_host_i32/meta_host_f32` are parsed by the bridge for allocation,
  validation, and dispatch sizes.
- Inherits fixed-capacity no-overflow binning: `binned_ids` is allocated as
  `tile_count * max_fast_pairs`, and the fork raises if any tile exceeds the
  cap.
- Python package: `torch_gsplat_bridge_v12c_fused_raster_color_loss_backward`.
- Custom op namespace: `torch.ops.gsplat_metal_v12c_fused_raster_color_loss_backward`.
- Metal source: `csrc/metal/gsplat_v12c_fused_raster_color_loss_backward_kernels.metal`.

## Intended Optimization

The fork keeps the parent grad-cache backward path for `F <= 32`, where each
pixel thread caches `grad_features[pix, :]`.

The new prototype path is exposed as:

```python
fused_linear_sigmoid_mse_backward(...)
```

It supports only:

- MPS tensors
- fast-tile path with `active_policy="off"`
- no overflow (`enable_overflow_fallback=False`)
- `feature_dim <= 32`
- colorizer `rgb = sigmoid(W @ features + b)` with `W.shape == [3,F]`
- mean MSE reconstruction after `alpha * rgb + (1 - alpha) * background_rgb`

It returns explicit Gaussian and colorizer parameter gradients for parity and
benchmarking. It is not wired into trainer autograd or checked-in configs.

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
( cd third_party/fast-mac-gsplat/variants/v12c_fused_raster_color_loss_backward
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

This fork is not wired into checked-in trainer configs here.

## Prototype Check

After building, run:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/v12c_fused_raster_color_loss_backward \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
  python third_party/fast-mac-gsplat/variants/v12c_fused_raster_color_loss_backward/tests/fused_linear_sigmoid_mse_check.py
```
