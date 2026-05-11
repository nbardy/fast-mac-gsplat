# v12a_fused_colorize_l1_no_norm

Opt-in fast-mac feature shader fork for testing a fused no-pre-norm
`Conv2d(F,3,1x1) + sigmoid + alpha compose + L1` gradient producer on top of
the v11 feature grad-cache, host-metadata, zero-background, fixedbin variant.

## Lineage

- Copied from `variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`.
- Inherits v11 raster behavior: grad-cache backward for F32, zero-background
  tail skip, host-side metadata parsing, and fixed-capacity no-overflow binning.
- Python package: `torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm`.
- Custom op namespace: `torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm`.
- Metal source: `csrc/metal/gsplat_v12a_fused_colorize_l1_no_norm_kernels.metal`.

## Intended Optimization

The rasterizer path is intentionally inherited. The new v12a API is
`fused_no_norm_l1_grad(...)`, which accepts:

```text
features: [N,H,W,F]
alpha: [N,H,W]
target_rgb/background_rgb: [N,3,H,W]
weight: [3,F] or [3,F,1,1]
bias: [3]
```

and returns:

```text
loss_per_image, grad_features, grad_alpha, grad_weight, grad_bias
```

The gradients match `mean_n(mean_cyx(abs(pred-target)))`, where
`pred = alpha * sigmoid(weight @ features + bias) + (1-alpha) * background`.
This is a prototype gradient producer, not a trainer-facing autograd Function.
It still writes dense `grad_features`; full raster+loss fusion is a separate
future kernel.

The fixedbin change removes the exact-length `tile_offsets[-1].item()` bin
allocation sync, but uses a larger fixed ID buffer. It is intentionally
no-overflow only; increase `GSP_FAST_CAP` or use v10/v9 if a row overflows.

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
( cd third_party/fast-mac-gsplat/variants/v12a_fused_colorize_l1_no_norm
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

This fork is not wired into checked-in trainer configs here.

## Local Checks

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/v12a_fused_colorize_l1_no_norm \
  .venv/bin/python third_party/fast-mac-gsplat/variants/v12a_fused_colorize_l1_no_norm/tests/fused_colorize_l1_check.py

PYTHONPATH=third_party/fast-mac-gsplat/variants/v12a_fused_colorize_l1_no_norm \
  .venv/bin/python third_party/fast-mac-gsplat/variants/v12a_fused_colorize_l1_no_norm/benchmarks/benchmark_fused_colorize_l1.py \
  --images 4 --height 128 --width 128 --feature-dim 32 --check
```
