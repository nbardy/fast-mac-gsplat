# v12b_fused_colorize_rmsnorm_l1

Opt-in fast-mac feature shader scaffold for testing a fused
RMSNorm + 1x1 colorize + alpha-compose + L1 backward path for F-channel
feature splatting.

## Lineage

- Copied from `variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`.
- Ports only the v8 host-side metadata split where applicable:
  `meta_i32/meta_f32` stay on MPS for shader args, while
  `meta_host_i32/meta_host_f32` are parsed by the bridge for allocation,
  validation, and dispatch sizes.
- Ports the existing `v6_refined_features_f32_fixedbin` no-overflow binning
  idea: `binned_ids` is allocated as `tile_count * max_fast_pairs`, and the
  fork raises if any tile exceeds the cap.
- Python package: `torch_gsplat_bridge_v12b_fused_colorize_rmsnorm_l1`.
- Custom op namespace: `torch.ops.gsplat_metal_v12b_fused_colorize_rmsnorm_l1`.
- Metal source: `csrc/metal/gsplat_v12b_fused_colorize_rmsnorm_l1_kernels.metal`.

## Current Status

This directory currently contains:

- a renamed v11 rasterizer baseline, so the variant can build/import without
  colliding with v11;
- a pure PyTorch reference implementation in
  `torch_gsplat_bridge_v12b_fused_colorize_rmsnorm_l1/fused_colorize_l1.py`;
- a closed-form gradient checker in
  `tests/fused_colorize_l1_reference_check.py`.

The fused Metal colorize/L1 backward kernel is not implemented yet. The
reference module fixes the shape contract and exact formulas for that kernel.

See:

```text
research_notes/fast_mac_v12b_fused_colorize_rmsnorm_l1.md
```

## Inherited Rasterizer Optimization

The fork keeps the parent grad-cache backward path for `F <= 32`, where each
pixel thread caches `grad_features[pix, :]`.

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
( cd third_party/fast-mac-gsplat/variants/v12b_fused_colorize_rmsnorm_l1
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

This fork is not wired into checked-in trainer configs here.

## Reference Test

The formula reference is CPU-only and does not require building the extension:

```bash
.venv/bin/python third_party/fast-mac-gsplat/variants/v12b_fused_colorize_rmsnorm_l1/tests/fused_colorize_l1_reference_check.py
```
