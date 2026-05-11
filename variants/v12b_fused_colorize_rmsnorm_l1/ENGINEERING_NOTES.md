# v12b_fused_colorize_rmsnorm_l1 Engineering Notes

This variant is a copied fork of
`variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`. It keeps that
fork's opt-in F32 `grad_features[pix, :]` direct-backward cache,
zero-feature-background tail skip, v8 host-side metadata split, and
fixed-capacity binning scaffold.

The namespace is intentionally unique:

- Python package: `torch_gsplat_bridge_v12b_fused_colorize_rmsnorm_l1`
- custom op namespace: `torch.ops.gsplat_metal_v12b_fused_colorize_rmsnorm_l1`
- Metal source: `csrc/metal/gsplat_v12b_fused_colorize_rmsnorm_l1_kernels.metal`

Behavioral delta versus the copied v11 rasterizer:

- Added a pure PyTorch fused color/loss reference at
  `torch_gsplat_bridge_v12b_fused_colorize_rmsnorm_l1/fused_colorize_l1.py`.
- Added `tests/fused_colorize_l1_reference_check.py` to check the closed-form
  RMSNorm + 1x1 + alpha-compose + L1 gradients against PyTorch autograd.
- No fused Metal colorize/L1 backward kernel exists yet. The current C++/Metal
  rasterizer is the renamed v11 scaffold.

Inherited v11 behavior:

- `_make_meta(...)` now returns GPU metadata plus CPU `meta_host_i32` and
  `meta_host_f32` tensors.
- Torch op schemas and C++/Metal entrypoints accept the host metadata pair.
- `parse_meta(...)` now reads CPU metadata directly instead of calling
  `.cpu()` on MPS metadata inside the bridge.
- Shader math is left inherited from v9; `meta_i32/meta_f32` are still the
  tensors passed to Metal kernels.
- `metal_bin(...)` allocates fixed `tile_count * max_fast_pairs` ID storage,
  initializes offsets on GPU, and drops writes past each tile cap. Python raises
  on any overflow, so this fork is no-overflow only.

No trainer configs were edited for this fork. Treat it as an opt-in design and
reference scaffold until a focused Metal parity gate and target-shape timing run
justify wiring it into renderer dispatch.
