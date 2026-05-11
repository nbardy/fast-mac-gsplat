# v13b_rgb_grad_handoff Engineering Notes

This variant is a copied fork of
`variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`. It keeps that
fork's opt-in F32 `grad_features[pix, :]` direct-backward cache,
zero-feature-background tail skip, v8 host-side metadata split, and
fixed-capacity no-overflow binning.

The namespace is intentionally unique:

- Python package: `torch_gsplat_bridge_v13b_rgb_grad_handoff`
- custom op namespace: `torch.ops.gsplat_metal_v13b_rgb_grad_handoff`
- Metal source: `csrc/metal/gsplat_v13b_rgb_grad_handoff_kernels.metal`

Behavioral delta versus the parent gradcache fork:

- Normal raster APIs are renamed but intentionally v11-compatible.
- Adds `rgb_grad_handoff_backward(...)` as the target low-level API for a future
  RGB-gradient handoff kernel.
- Registers `render_fast_backward_rgb_grad_handoff(...)` under the v13b custom
  op namespace, but the implementation raises until the Metal kernel exists.
- Adds `estimate_rgb_grad_handoff_memory(...)` and
  `benchmarks/rgb_grad_handoff_accounting.py` for target-shape bandwidth
  accounting.

Missing kernel sketch:

```text
for each fast-path pixel:
  load feature[f], alpha, grad_rgb[3], background_rgb[3]
  logits[c] = bias[c] + sum_f weight[c,f] * feature[f]
  rgb[c] = sigmoid(logits[c])
  composed[c] = alpha * rgb[c] + (1 - alpha) * background_rgb[c]
  g_alpha = sum_c grad_rgb[c] * (rgb[c] - background_rgb[c])
  for f:
    g_feature_f = sum_c grad_rgb[c] * alpha * rgb[c] * (1-rgb[c]) * weight[c,f]
  optionally accumulate g_weight/g_bias from the same local values
  feed g_feature_f and g_alpha into the inherited reverse contributor loop
```

For `B=16,H=W=256,F=32,float32`, the current dense backward input is
`grad_features` plus `grad_alpha` = 132 MiB. The handoff input is
`grad_rgb` = 12 MiB. The target removes 120 MiB, or about 90.9%, of dense
per-pixel backward-input traffic before counting allocator pressure.

No trainer configs were edited for this fork. The shared Dynaworld fast-mac
renderer can select the renamed v11-compatible raster API with
`fast_mac.feature_variant="v13b_rgb_grad_handoff"`. The actual RGB-gradient
handoff op remains a scaffold until the Metal streaming VJP kernel is written.
