# v6_refined_features engineering notes

This fork preserves the tested `v5_features` F-channel and accumulated-alpha
contract under a separate package/custom-op namespace while carrying the
v6_refined active-tile scheduling surface:

- Python package: `torch_gsplat_bridge_v6_refined_features`
- custom op namespace: `torch.ops.gsplat_metal_v6_refined_features`
- output API: `(features, accumulated_alpha)`

It is intentionally isolated so the Dynaworld trainer can select it with
`render.fast_mac.feature_variant = "v6_refined_features"` while keeping
`v5_features` as the stable baseline.

## Active-tile status

The feature fork now has active-tile eval/train kernels for arbitrary `F`,
accumulated alpha, and alpha-gradient backward. It also carries the v6_refined
Python selection policy and adaptive stop-count metadata:

- default: `active_policy="off"`
- sparse/overflow probe: `active_policy="auto"`
- forced ablation: `active_policy="on"`

Do not promote active mode globally from a smoke result. Dense-screen F32 can be
slower because active mode initializes full feature/alpha outputs and adds a
sparse-launch path. Use the benchmark profile fields before selecting it.

## Verified gates

- shape contract for `F in {1,3,4,8,16,32,64}`
- F=3 parity against RGB v5
- F=32 feature-gradient check against the Torch reference
- active-policy F=3 parity against RGB v5
- active-policy F=32 feature-gradient parity against the direct path
- accumulated-alpha forward/backward checks
- active-policy F32 feature+alpha gradient parity against the direct path
- reference image/gradient checks
- F32 trainer smoke through alpha-aware composition, colorize, and PCA media
