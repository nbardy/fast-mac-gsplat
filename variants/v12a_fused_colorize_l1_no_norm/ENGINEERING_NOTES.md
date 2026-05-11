# v12a_fused_colorize_l1_no_norm Engineering Notes

This variant is a copied fork of
`variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`. It keeps that
fork's opt-in F32 `grad_features[pix, :]` direct-backward cache,
zero-feature-background tail skip, host-side metadata split, and fixed-capacity
no-overflow binning.

The namespace is intentionally unique:

- Python package: `torch_gsplat_bridge_v12a_fused_colorize_l1_no_norm`
- custom op namespace: `torch.ops.gsplat_metal_v12a_fused_colorize_l1_no_norm`
- Metal source: `csrc/metal/gsplat_v12a_fused_colorize_l1_no_norm_kernels.metal`

Behavioral delta versus the parent gradcache fork:

- Raster shader math is inherited from v11.
- New op: `fused_no_norm_l1_grad(features, alpha, target_rgb, background_rgb,
  weight, bias)`.
- The new op is a prototype gradient producer for exactly:
  no-pre-norm, no-hidden, no-view-conditioning, sigmoid colorizer, alpha RGB
  compose, and L1 reconstruction.
- It returns `loss_per_image`, `grad_features`, `grad_alpha`, `grad_weight`,
  and `grad_bias`. These gradients are scaled for the total loss
  `loss_per_image.mean()`.
- It does not call raster backward and is not wired into trainer autograd yet.
  The intended next boundary is a custom objective autograd function that feeds
  `grad_features` and `grad_alpha` into the existing raster backward.

No trainer configs were edited for this fork. Treat it as an opt-in shader
candidate until a focused smoke/parity gate and target-shape timing run justify
using it.
