# v8 hardware-training scaffold notes

Date: 2026-04-24

Scope: `variants/v8_hw_train/` implements Plan 2 as a fallback-safe scaffold
based on `variants/v8`. It does not implement a Metal render-pipeline forward
or render-assisted backward.

## Current Behavior

- Python package: `torch_gsplat_bridge_v8_hw_train`
- Native torch namespace: `gsplat_metal_v8_hw_train`
- Default path: cloned v8 compute forward/backward.
- `RasterConfig` exposes `use_hardware_train`, `hardware_train_policy`,
  `capture_final_T`, `capture_stop_count`, `capture_pixel_stop`, and
  `backward_state_mode`.
- `probe_hardware_train(config)` reports unsupported hardware training without
  touching tensor data.
- `use_hardware_train=True, hardware_train_policy="fallback"` warns and routes
  to v8 compute.
- `use_hardware_train=True, hardware_train_policy="strict"` raises before
  attempting a non-existent hardware training path.
- `estimate_hardware_train_state(config, batch_size=B)` plans state memory
  without allocation:
  - `tile_stop`: `B * ceil(H/tile) * ceil(W/tile) * 4` bytes, int32.
  - `final_T`: `B * H * W * 4` bytes, fp32.
  - `pixel_stop`: `B * H * W * 4` bytes, int32.
- `profile_projected_gaussians` reports requested/selected state modes,
  requested/selected state bytes, per-buffer bytes, state shapes, and the exact
  fallback reason. Unsupported hardware training selects `compute` and reports
  zero selected hardware-state bytes.
- State-mode validation fails closed:
  - `tile_stop` requires `capture_stop_count=True`.
  - `final_T` requires `capture_stop_count=True` and `capture_final_T=True`.
  - `pixel_stop` requires `capture_stop_count=True` and
    `capture_pixel_stop=True`.
  - `final_T` and `pixel_stop` captures are mutually exclusive in this scaffold.

## Training-Grade Hardware Gaps

The hardware path cannot be promoted until these are solved:

- GPU-native output interop: hardware render output must land in Torch/MPS
  visible storage or be copied by GPU work on the same scheduling path. CPU
  texture readback, CPU staging, and `waitUntilCompleted()` in the timed path
  are disqualifying.
- Current exact blocker: the older v7 render experiments stage through CPU
  textures/buffers and `waitUntilCompleted()`. v8's training backward consumes
  GPU-resident sorted tile bins and the exact forward processed prefix `M`.
  Without a GPU-native render output plus exact `M` state capture, a hardware
  forward would either race PyTorch MPS scheduling or feed backward a different
  prefix than the one used for compositing.
- Stable ordered compositing: fragment/imageblock updates must preserve v8's
  stable depth order, alpha clamp, visibility gates, and early-stop threshold
  for every pixel.
- Exact processed prefix `M`: backward must see the exact forward prefix after
  early stop. Tile-level stop counts alone are only valid if backward recomputes
  each pixel's true end index and final transmittance under the same gates.
- State capture modes: `tile_stop`, `final_T`, and `pixel_stop` need concrete
  GPU buffers, overflow flags, and memory accounting. `final_T` and
  `pixel_stop` are about 67 MB each at 4K B=1 with fp32/i32 state.
- Backward math: gradients must use v8's reverse recurrence with
  `S_after_i`/`gT`, reconstruct `T_prev`, and reduce per tile before global
  gradient atomics. Per-pixel or per-fragment global atomics are not acceptable.
- Overflow/heavy tiles: capacity overflow must fail closed to compute or route
  through exact compacted heavy-tile work. Heavy segmentation must respect the
  true early-stop prefix for the incoming transmittance.
- Profiling evidence: promotion needs image and gradient parity against v8 plus
  forward+backward timing wins without additional CPU readback/wait overhead.
