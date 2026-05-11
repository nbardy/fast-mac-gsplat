# v13a_temporal_recompute_state Engineering Notes

This variant is a copied fork of
`variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`. It keeps that
fork's opt-in F32 `grad_features[pix, :]` direct-backward cache,
zero-feature-background tail skip, v8 host-side metadata split, and
fixed-capacity binning.

The namespace is intentionally unique:

- Python package: `torch_gsplat_bridge_v13a_temporal_recompute_state`
- custom op namespace: `torch.ops.gsplat_metal_v13a_temporal_recompute_state`
- Metal source: `csrc/metal/gsplat_v13a_temporal_recompute_state_kernels.metal`

Behavioral delta versus the parent gradcache fork:

- `RasterConfig.backward_state_strategy` chooses `"save"` or `"recompute"`.
- `"save"` keeps the copied v11 autograd state contract.
- `"recompute"` saves empty placeholders for tile metadata in forward, then
  reruns `bin` and `render_*_forward_state` in backward to recreate
  `active_tile_ids`, `tile_counts`, `tile_offsets`, `binned_ids`, and
  `tile_stop_counts`.
- The C++ and Metal kernels are namespace-renamed but behavior-identical to
  v11; recompute is implemented at the Python autograd boundary.
- Python still raises on fixedbin overflow, so this fork remains no-overflow
  only.

No trainer configs were edited for this fork. The shared Dynaworld fast-mac
renderer can select it with
`fast_mac.feature_variant="v13a_temporal_recompute_state"` and can choose
`fast_mac.backward_state_strategy="save" | "recompute"`. Treat `"recompute"`
as a memory-valve candidate until a target-shape memory trace justifies using
it.
