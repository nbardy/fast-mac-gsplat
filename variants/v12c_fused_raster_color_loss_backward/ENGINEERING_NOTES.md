# v12c_fused_raster_color_loss_backward Engineering Notes

This variant is a copied fork of
`variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`. It keeps that
fork's opt-in F32 `grad_features[pix, :]` direct-backward cache,
zero-feature-background tail skip, host metadata split, and fixed-capacity
no-overflow binning.

The v12c-specific addition is a narrow prototype fused backward path for:

```text
raster features/alpha -> linear 1x1 sigmoid colorize -> RGB alpha compose -> mean MSE
```

The Metal kernel computes the pixel colorize/compose/MSE VJP locally and feeds
that thread-local feature/alpha gradient directly into the reverse raster
contributor loop. It returns explicit gradients for Gaussian screen-space
inputs, per-splat features, opacities, and linear colorizer parameters. It is a
parity/timing scaffold, not trainer integration.

The namespace is intentionally unique:

- Python package: `torch_gsplat_bridge_v12c_fused_raster_color_loss_backward`
- custom op namespace: `torch.ops.gsplat_metal_v12c_fused_raster_color_loss_backward`
- Metal source: `csrc/metal/gsplat_v12c_fused_raster_color_loss_backward_kernels.metal`

Behavioral delta versus the parent gradcache fork:

- Adds `render_fast_backward_linear_sigmoid_mse(...)` to the custom op surface.
- Adds `fused_linear_sigmoid_mse_backward(...)` to the Python package.
- Adds `tests/fused_linear_sigmoid_mse_check.py` for MPS parity against the
  inherited unfused raster + PyTorch colorize/compose/MSE autograd path.
- Leaves active-tile and overflow fused paths unsupported; callers must use
  `active_policy="off"` and `enable_overflow_fallback=False`.
- Leaves hidden colorizers, LayerNorm, view conditioning, L1, and DSSIM
  unsupported in the fused prototype.

No trainer configs were edited for this fork. Treat it as an opt-in shader
candidate until a focused smoke/parity gate and target-shape timing run justify
using it.
