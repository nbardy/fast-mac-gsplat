# v9_features_gradcache_zero_bg Engineering Notes

This variant is a copied fork of
`variants/v6_refined_features_f32_gradcache`. It keeps that fork's opt-in
F32 `grad_features[pix, :]` direct-backward cache and adds only the
zero-feature-background tail skip from
`variants/v6_refined_features_f32_zero_bg`.

The namespace is intentionally unique:

- Python package: `torch_gsplat_bridge_v9_features_gradcache_zero_bg`
- custom op namespace: `torch.ops.gsplat_metal_v9_features_gradcache_zero_bg`
- Metal source: `csrc/metal/gsplat_v9_features_gradcache_zero_bg_kernels.metal`

Behavioral delta versus the parent gradcache fork:

- `_make_meta(...)` sets reserved bit `2` when the configured feature background
  is exactly zero across the active feature dimension.
- Fast and active forward kernels skip the final `add_background_tail(...)` only
  under that bit.
- Backward skip-color-gradient checks mask reserved bit `1`, preserving the
  zero-background bit for forward metadata.
- Overflow forward is left as inherited; this matches the source zero-bg fork's
  scoped fast/active tail-skip behavior.

No trainer configs were edited for this fork. Treat it as an opt-in shader
candidate until a focused smoke/parity gate and target-shape timing run justify
using it.
