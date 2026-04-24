# v8 Hardware Eval Scaffold Notes

Date: 2026-04-24

Worker 1 implemented Plan 1 as `variants/v8_hw_eval/`. The referenced
`RTK.md` from `AGENTS.md` was not present in this repository, so this scaffold
follows the user-provided ownership and isolation constraints.

## Current Behavior

- Python package: `torch_gsplat_bridge_v8_hw_eval`
- Native torch namespace: `gsplat_metal_v8_hw_eval`
- Default execution: inherited v8 compute bin/render/backward path
- Hardware eval flags:
  - `use_hardware_eval`
  - `hardware_eval_policy=off|fallback|require`
  - `emit_final_T`
  - `emit_stop_count`
- Probe: `probe_hardware_eval(config)` returns a structured
  `HardwareEvalStatus`.
- Native probe: `probe_hardware_eval_capabilities()` calls
  `probe_hardware_eval_native` when the extension is built. It compiles a
  tensor-free minimal render pipeline and reports capabilities without touching
  tensor data or entering the timed raster path.

The probe intentionally reports `supported=False`. With
`hardware_eval_policy="fallback"`, requested hardware eval fails closed to the
inherited v8 compute path. With `hardware_eval_policy="require"`, it raises a
clear error instead of pretending a hardware render pass exists.

## Current Local Probe Result

On the current Apple M4 test machine, the tensor-free probe reports:

- Metal device and command queue available.
- Minimal RGBA32F render pipeline source available.
- Minimal render library compiles.
- Minimal render pipeline state is ready.
- No CPU readback in the probe.
- Hardware eval still not selected because render output is not wired to
  Torch/MPS-visible storage.
- Imageblock support and raster-order-group support are intentionally reported
  as unknown until an exact tile/imageblock compositing shader exists and is
  validated.

The current selection is therefore still `v8_compute_fallback`, with missing
prerequisite `render_to_mps_interop` and unknown prerequisites
`imageblock_support` and `raster_order_group_support`.

## Concrete Gap To Real Hardware Eval

This scaffold does not yet implement the render-pipeline portion of
`docs/v8_hw_tile_raster_plans.md`. Missing pieces:

- MPS/Torch-visible Metal render target or texture interop for direct image
  output without CPU readback.
- GPU-resident draw stream or indirect draw command generation from v8 tile
  bins.
- Tile shader/imageblock layout for per-pixel `C`, `T`, optional stop count,
  and flags.
- Ordered same-pixel fragment updates via raster order groups or an equivalent
  render-pass ordering guarantee.
- Exact optional `final_T` and `stop_count` capture needed to validate parity
  and future training use.
- Validation that hardware tile size, imageblock pressure, early stop, alpha
  clamp, and stable depth order match the v8 math contract.

Until those are solved, timed paths should benchmark the v8 compute fallback,
not a partial hardware path.
