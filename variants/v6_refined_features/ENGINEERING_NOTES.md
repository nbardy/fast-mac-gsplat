# v6_refined_features engineering notes

This fork currently preserves the tested `v5_features` F-channel and
accumulated-alpha contract under a separate package/custom-op namespace:

- Python package: `torch_gsplat_bridge_v6_refined_features`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features`
- output API: `(features, accumulated_alpha)`

It is intentionally isolated so the Dynaworld trainer can select it with
`render.fast_mac.feature_variant = "v6_refined_features"` while keeping
`v5_features` as the stable baseline.

## Current limitation

This is not yet a full v6_refined feature port. The v6_refined RGB branch has
active-tile scheduling and adaptive stop-count behavior that are still RGB-3
specific. Porting those kernels to arbitrary `F` remains the next performance
task. Until that lands, use this fork as a namespace and trainer-integration
base, not as evidence that F32 has inherited v6_refined's high-res speedups.

## Verified gates

- shape contract for `F in {1,3,4,8,16,32,64}`
- F=3 parity against RGB v5
- F=32 feature-gradient check against the Torch reference
- accumulated-alpha forward/backward checks
- reference image/gradient checks
- F32 trainer smoke through alpha-aware composition, colorize, and PCA media
