# v10_features_gradcache_zero_bg_hostmeta Engineering Notes

This variant is a copied fork of
`variants/v9_features_gradcache_zero_bg`. It keeps that fork's opt-in F32
`grad_features[pix, :]` direct-backward cache and zero-feature-background
tail skip, then ports the v8 host-side metadata split.

The namespace is intentionally unique:

- Python package: `torch_gsplat_bridge_v10_features_gradcache_zero_bg_hostmeta`
- custom op namespace: `torch.ops.gsplat_metal_v10_features_gradcache_zero_bg_hostmeta`
- Metal source: `csrc/metal/gsplat_v10_features_gradcache_zero_bg_hostmeta_kernels.metal`

Behavioral delta versus the parent gradcache fork:

- `_make_meta(...)` now returns GPU metadata plus CPU `meta_host_i32` and
  `meta_host_f32` tensors.
- Torch op schemas and C++/Metal entrypoints accept the host metadata pair.
- `parse_meta(...)` now reads CPU metadata directly instead of calling
  `.cpu()` on MPS metadata inside the bridge.
- Shader math is left inherited from v9; `meta_i32/meta_f32` are still the
  tensors passed to Metal kernels.
- The inherited `tile_offsets[-1].item()` allocation sync in `metal_bin(...)`
  is intentionally unchanged; this port only targets metadata parsing.

No trainer configs were edited for this fork. Treat it as an opt-in shader
candidate until a focused smoke/parity gate and target-shape timing run justify
using it.
