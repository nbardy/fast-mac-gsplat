# world_foam_lane2_fused_slab_v0

Isolated Lane 2 World Foam Metal scaffold.

This directory is intentionally disjoint from production fast-mac variants. It
does not provide a production trainer hook, viewer hook, or config integration.
The current value is a small, inspectable shader ABI for screen-time beam
events and 2D+time power-boundary event counts, plus local Torch/MPS bridge
  sources for real-ray forward, VJP, reduced-VJP, CSR candidate storage, CSR
  frozen-geometry autograd, and frozen-geometry training/eval smokes.

## Current native build boundary (2026-08-15)

`bindings.cpp` is now sealed at exactly 133 unique schemas with 133 matching
`CompositeExplicitAutograd` implementations. The retained
`_C.cpython-311-darwin.so` predates the latest 30 registrations and exposes
only 103, so it is a diagnostic stale binary and must not be used for G4/G6.
This is a build-state mismatch, not a missing registration in current source.

`native_build_contract.py` is the standard-library source of truth used by
both `setup.py` and the repository verifiers. It pins the two translation
units, runtime-compiled Metal sources, headers/Python ABI files, the 133-schema
name/signature digests, and the 30 memory-light/full-geometry schemas added
after the retained binary. `setup.py` fails before compilation if that contract
drifts and declares the header/Metal inputs as build dependencies.

On an operator-approved quiet host, force the correct CPython 3.11 rebuild:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  UV_CACHE_DIR=/private/tmp/uv-cache uv run \
    --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
    python setup.py build_ext --inplace --force )
```

Then attest and independently verify the build without launching Metal:

```bash
cd /Users/nicholasbardy/git/gsplats_browser/dynaworld
PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/attest_worldfoam_fused_slab_build.py \
  --write-receipt
PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_worldfoam_native_variant_imports.py
```

Acceptance requires the exact active-interpreter path
`torch_world_foam_lane2_fused_slab/_C.cpython-311-darwin.so`, exact source and
binary hashes, CPython 3.11/Darwin architecture, Torch and compiler identities,
all 133 source signatures, all 133 dispatcher signatures, and a kernel for
every schema. The receipt scope is build and registration only: it does not
claim Metal execution, numerical parity, memory fit, speed, or paper quality.

## Source Inventory

The complete source surface is:

- `csrc/shared/world_foam_lane2_types.h`: shared host/Metal structs, buffer
  indices, event-axis constants, grid flags, and count flags.
- `csrc/metal/world_foam_lane2_event_count.metal`: `wf2_count_screen_time_beam_events`,
  a one-thread-per-beam kernel that counts strict `u`, `v`, and `t` tile-boundary
  crossings for a screen-time segment. It can optionally append
  `WF2BoundaryEvent` records when `WF2_GRID_FLAG_WRITE_EVENTS` is set.
- `csrc/metal/world_foam_lane2_power_boundary.metal`: `wf2_count_power_boundary_events`,
  a one-thread-per-beam-slab kernel that counts depth-overlapping shared-metric
  power boundaries under the current toy camera model.
- `csrc/metal/world_foam_lane2_power_boundary_tensor.metal`:
  `wf2_count_power_boundary_events_tensor`, a tensor-buffer variant of the same
  power-boundary count kernel for the first Torch/MPS runtime bridge.
- `csrc/bindings.cpp` and `csrc/metal/world_foam_lane2_metal.mm`: isolated
  Torch custom-op bridge source for
  `world_foam_lane2_fused_slab_v0.count_power_boundary_events` and
  `world_foam_lane2_fused_slab_v0.shared_signal_replay` /
  `world_foam_lane2_fused_slab_v0.shared_rgb_replay` /
  `world_foam_lane2_fused_slab_v0.shared_rgba_depth_replay` /
  `world_foam_lane2_fused_slab_v0.shared_rgba_depth_vjp` /
  `world_foam_lane2_fused_slab_v0.realray_rgba_depth_replay` /
  `world_foam_lane2_fused_slab_v0.shared_realray_rgba_depth_replay` /
  `world_foam_lane2_fused_slab_v0.shared_realray_rgba_depth_vjp`.
- `torch_world_foam_lane2_fused_slab/ops.py`: thin Python tensor validator/wrapper.
- `setup.py`: local extension build recipe matching the STAR-UVT bridge shape.
- `native_build_contract.py`: exact translation-unit, dependency, and
  133-schema source contract consumed before every build.
- `tools/static_validate.py`: local static validator for the shared ABI,
  power-boundary CPU fixture, and Metal source compilation.
- `tools/smoke_power_boundary_mps.py`: first MPS count-only runtime smoke.
- `tools/smoke_shared_replay_mps.py`: Gate 0.6 MPS shared scalar signal and
  site-signal-gradient replay smoke.
- `tools/smoke_rgb_strip_mps.py`: Gate 0.7 toy RGB strip smoke built from one
  shared-RGB replay launch.
- `tools/smoke_composite_strip_mps.py`: Gate 0.8 toy RGB/alpha/depth composite
  strip smoke built from one shared RGBA-depth replay launch.
- `tools/smoke_composite_vjp_mps.py`: Gate 0.9 fixed-segment site-RGBA VJP
  smoke for the toy compositor.
- `tools/smoke_composite_vjp_slab_mask_mps.py`: Gate 0.95 slab-indexed
  candidate-mask smoke for the same fixed-segment VJP.
- `tools/smoke_full_frame_vjp_mps.py`: Gate 1 toy full-frame-shaped
  RGB/alpha/depth VJP smoke using the existing `u/t` replay op over `H*W`
  flattened beams.
- `tools/smoke_realray_replay_mps.py`: Gate 1C true camera-ray forward smoke
  using flattened `[origin, direction]` rays, 4D sites, and 4D power
  boundaries.
- `tools/smoke_shared_realray_replay_mps.py`: Gate 2B true camera-ray shared
  forward smoke using bitset candidates per `(pixel track, time slab)`.
- `tools/smoke_shared_realray_vjp_mps.py`: Gate 2C true camera-ray shared
  fixed-segment site-RGBA/density VJP smoke.
- `tools/smoke_shared_realray_vjp_reduce_mps.py`: Gate 2D reduced true
  camera-ray VJP smoke with frozen-geometry site-RGBA autograd parity.
- `tools/smoke_shared_realray_csr_candidate_storage_mps.py`: Gate 2F
  per-track and tiled CSR candidate-storage smoke for the shared real-ray
  reduced VJP path.
- `tools/smoke_shared_realray_csr_scaling_mps.py`: Gate 2G tiled CSR
  frame-scaling smoke for the shared real-ray reduced VJP path.
- `tools/smoke_shared_realray_autograd_overfit_mps.py`: Gate 2E
  teacher-target frozen-geometry site-RGBA autograd overfit smoke.
- `tools/smoke_shared_realray_real_target_train_mps.py`: Gate 3
  real-target frozen-geometry site-RGBA training smoke.
- `tools/train_eval_shared_realray_csr_mps.py`: Gate 3 CSR 256px/16f
  same-split train/eval artifact using tiled CSR frozen-geometry
  site-RGBA/density autograd.

There are no production benchmark harnesses, production trainer hooks, or
viewer hooks in this variant.

## What Exists

The scaffold implements isolated event-counting and replay/compositing gates:

- screen-time beam ABI: `WF2ScreenTimeBeam`, `WF2GridConfig`,
  `WF2BeamEventCount`, and optional `WF2BoundaryEvent`;
- strict interior tile-boundary crossing counts in `u`, `v`, and `t`;
- optional event-record emission with axis, boundary index, segment parameter
  `s`, interpolated `uvt`, and direction flag;
- power-boundary ABI: `WF2PowerBoundary3D`, `WF2PowerBeamSlab`,
  `WF2PowerBoundaryConfig`, and `WF2PowerBoundaryCount`;
- power-boundary event counts matching the Gate 0 beam toy's fixed camera-path
  model;
- MPS custom-op source paths for tensor-buffer power-boundary counts, toy
  shared replay/compositing/VJP smokes, true-ray per-sample forward rendering,
  true-ray shared forward rendering, true-ray shared fixed-segment VJP, reduced
  site-RGBA VJP, CSR candidate storage, CSR frozen-geometry autograd, a tiny
  real-target training smoke, and a same-split 256px/16f fixed-geometry
  train/eval artifact.

`WF2ScreenTimeBeam` is a line segment in screen-time coordinates:

```text
start_uvt = (u_px, v_px, t_frame_centered)
end_uvt   = (u_px, v_px, t_frame_centered)
radius_px = optional expansion hint for later binning work
payload_id = caller-owned beam/source id
```

`WF2GridConfig` describes the screen-time tile lattice. The first kernel counts
strict crossings of interior tile boundaries:

```text
u boundaries: k * tile_size_u, k in [1, tile_count_u - 1]
v boundaries: k * tile_size_v, k in [1, tile_count_v - 1]
t boundaries: k * tile_size_t, k in [1, tile_count_t - 1]
```

For each beam, `WF2BeamEventCount` reports the per-axis counts and total. When
`WF2_GRID_FLAG_WRITE_EVENTS` is set, the kernel also appends `WF2BoundaryEvent`
records with the crossing axis, boundary index, segment parameter `s`, and
interpolated `uvt`.

`WF2PowerBoundary3D` stores a power-cell boundary plane in `(x,z,t)`:

```text
nx * x + nz * z + nt * t + b = 0
```

`WF2PowerBeamSlab` stores a fixed `u` beam slab over `[t0,t1]` and depth range
`[near_depth, far_depth]`. The power-boundary kernel substitutes the toy camera
path:

```text
x(t) = u_center + camera_velocity_x * t
z(t) = s
```

and counts boundaries whose depth interval overlaps the slab depth range. This
is the Metal ABI analog of
`dynaworld/research_experiments/world_foam_lane2/gate0_beam_toy.py`.

## Static Validation

From the repository root:

```bash
python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/static_validate.py
```

Or from this variant directory:

```bash
python3 tools/static_validate.py
```

The validator proves only static and host-side facts:

1. the shared header is valid C++17 and exposes the expected struct fields;
2. compiles and runs a C++ power-boundary ABI probe that reproduces the CPU
   Gate 0 slab counts for `camera_velocity_x=0.35` and `0.7`;
3. compiles `world_foam_lane2_event_count.metal`,
   `world_foam_lane2_power_boundary.metal`, and
   `world_foam_lane2_power_boundary_tensor.metal` with
   `xcrun -sdk macosx metal` when the Xcode Metal compiler is available.

It does not dispatch either Metal kernel. It does not verify GPU buffer binding,
thread-grid sizing, output counts, event payloads, overflow behavior, MPS/Torch
interop, numerical parity against a GPU readback, or trainer/render quality.

There is no production build target yet. The local bridge source is intentionally
count-only and must be built from this directory:

```bash
python3 setup.py build_ext --inplace
```

It has not been promoted into any parent package or Dynaworld config.

## Next Host Bridge Pattern

The first executable path follows the STAR-UVT bridge shape, not the larger
PowerFoam package:

- local `setup.py` with one `CppExtension`;
- one `csrc/bindings.cpp` namespace that fails loudly unless tensors are on MPS;
- one Objective-C++ bridge that reads the local `.metal` source files, builds a
  `DynamicMetalShaderLibrary`, caches kernels, binds buffers, dispatches one
  thread per beam/slab, and reads count buffers back through Torch tensors;
- a thin Python wrapper that validates tensor dtypes, shapes, and packed config
  values before calling `torch.ops.world_foam_lane2_fused_slab_v0.*`.

The local `xcrun metal` compiler is currently unavailable in this environment,
which blocks offline shader compilation checks. That does not by itself block a
runtime MPS bridge, because the STAR-UVT pattern compiles source strings through
PyTorch's dynamic Metal shader library at runtime. The current count-only bridge
has been built and smoke-tested against the Gate 0 CPU slab-count fixture.

Current runtime smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_power_boundary_mps.py \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_mps_power_boundary_smoke.json
```

Result: MPS returns `149` and `151` power-boundary events for the two CPU Gate 0
velocities, matching the expected fixture counts with zero invalid rows.

## Gate 0.6 Shared Replay Boundary

This variant now contains a narrow Gate 0.6 MPS shared-replay op:

- `csrc/metal/world_foam_lane2_shared_replay_tensor.metal` implements
  `wf2_shared_signal_replay_tensor`, one thread per `(beam, frame)` ray.
- `torch_world_foam_lane2_fused_slab.shared_signal_replay` launches it and returns
  `output_f32 [M,T]` plus `grad_samples_f32 [M,T,S]`.
- `tools/smoke_shared_replay_mps.py` compares MPS outputs, per-ray site-gradient
  samples, reduced site gradients, loss, and scan accounting against
  `gate0_shared_forward_backward.py`.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_shared_replay_mps.py \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_6_mps_shared_replay_smoke.json
```

The saved 16-frame row matches the CPU reference within float32 tolerance:
`max_output_abs_error=1.1920928955078125e-07`,
`signal_gradient_max_abs_error=7.62939453125e-06`, and
`shared_forward_backward_boundary_scan_ratio=0.03125`. The same row reports
`mps_shared_replay_wall_clock_ms=1.0247604001051513` over 20 timed launches.

This remains a replay-op proof only. It does not imply tile sorting,
compositing, site-position gradients, trainer integration, or image-quality
evidence.

## Gate 0.7 RGB Strip Boundary

This variant now also contains a toy RGB strip smoke:

- `tools/smoke_rgb_strip_mps.py` uses `shared_rgb_replay`, one thread per
  `(beam, frame)` ray with three RGB channels.
- It writes an image-shaped `16 x 17 x 3` output and a PPM proof image.
- It compares MPS RGB output, site-RGB signal gradients, and loss against the
  CPU shared-segment reference.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_rgb_strip_mps.py \
  --timing-iters 20 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_7_mps_rgb_strip_smoke.json \
  --ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate0_7_mps_rgb_strip.ppm
```

The saved row reports `max_rgb_abs_error=4.76837158203125e-07`,
`color_gradient_max_abs_error=4.57763671875e-05`,
`loss_abs_error=1.0043974384643661e-05`, and
`mps_rgb_strip_wall_clock_ms=0.8528229001967702` over 20 timed iterations.

This is still not a full renderer. The separate Gate 0.8 path adds toy forward
alpha/depth compositing, but there are still no geometry gradients, trainer
hook, or heldout-camera metric.

## Gate 0.8 Composite Strip Boundary

This variant now contains a forward-only compositor smoke:

- `tools/smoke_composite_strip_mps.py` uses `shared_rgba_depth_replay`, one
  thread per `(beam, frame)` ray.
- It writes RGB `[16,17,3]`, alpha `[16,17]`, and expected-depth `[16,17]`
  strips plus a PPM proof image.
- It compares MPS RGB, alpha, and depth outputs against a CPU shared-segment
  compositor reference.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_composite_strip_mps.py \
  --timing-iters 20 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_8_mps_composite_strip_smoke.json \
  --ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate0_8_mps_composite_strip.ppm
```

The saved row reports `max_rgb_abs_error=1.7881393432617188e-07`,
`max_alpha_abs_error=1.7881393432617188e-07`,
`max_depth_abs_error=3.5762786865234375e-07`, and
`mps_composite_wall_clock_ms=0.8724833500309614` over 20 timed iterations.

This is still not a production renderer. It has no geometry gradients,
density-gradient training hook, real-video camera/image formation, or
heldout-camera metric.

## Gate 0.9 Composite VJP Boundary

This variant now contains a narrow fixed-segment compositor VJP smoke:

- `tools/smoke_composite_vjp_mps.py` uses `shared_rgba_depth_vjp`, one thread
  per `(beam, frame)` ray.
- It emits RGB, alpha, depth, and per-ray per-site RGBA gradient samples.
- It compares MPS outputs and reduced site-RGBA gradients against CPU autograd,
  then checks CPU autograd against finite differences.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_composite_vjp_mps.py \
  --timing-iters 20 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_9_mps_composite_vjp_smoke.json
```

The saved row reports `max_rgba_gradient_abs_error=1.9073486328125e-06`,
`finite_difference_max_abs_error=0.0003147125244140625`, and
`mps_composite_vjp_wall_clock_ms=1.3421896001091227` over 20 timed iterations.

This is still not a general backward pass. Boundary cuts, segment owners,
sorting, topology, site positions, site weights, camera projection, and
real-video image formation remain fixed or absent.

## Gate 0.95 Slab-Indexed Mask Boundary

The shared replay kernels now accept `candidate_mask_u32` as row-major
`[beam, slab]`; a length-`beam_count` tensor is still the `time_slabs=1`
case. This removes the earlier single-slab mask limitation without claiming
CSR storage or a full-frame renderer.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_composite_vjp_slab_mask_mps.py \
  --timing-iters 20 \
  --time-slabs 1,2,4 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_95_mps_composite_vjp_slab_mask_smoke.json
```

The saved rows report:

- `time_slabs=1`: `total_candidates=149`, scan ratio `0.0625`,
  `mps_composite_vjp_wall_clock_ms=0.6196104499395005`;
- `time_slabs=2`: `total_candidates=292`, scan ratio `0.125`,
  `mps_composite_vjp_wall_clock_ms=0.975545800247346`;
- `time_slabs=4`: `total_candidates=576`, scan ratio `0.25`,
  `mps_composite_vjp_wall_clock_ms=1.1342895497364225`;
- all rows have `max_rgba_gradient_abs_error=1.9073486328125e-06`,
  `finite_difference_max_abs_error=0.0003147125244140625`, and
  `segment_overflow_count=0`.

This is still toy strip output with int32 bitmask candidates. It is capped at
31 boundaries and 32 sites and does not implement CSR candidates, full-frame
image formation, geometry/topology gradients, trainer integration, or heldout
quality.

## Gate 1A Toy Full-Frame-Shaped VJP Boundary

This variant now contains a toy full-frame-shaped fixed-segment VJP smoke:

- `tools/smoke_full_frame_vjp_mps.py` uses `shared_rgba_depth_vjp`, one thread
  per flattened `(pixel, frame)` ray.
- It packs an `H x W` toy image as `H*W` beam rows, uses row-major
  `[beam, slab]` candidate masks, and reshapes RGB/alpha/depth outputs to
  `[T,H,W,...]`.
- It adds synthetic vertical variation by offsetting the existing `u` coordinate
  per row. This is still the current `x,z,t` toy cell path, not true camera-ray
  `(u,v,t)` projection.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_full_frame_vjp_mps.py \
  --height 8 --width 9 --frames 8 --time-slabs 2 --timing-iters 10 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate1_mps_full_frame_vjp_smoke.json \
  --ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate1_mps_full_frame_vjp.ppm
```

The saved row reports `rgb_shape=[8,8,9,3]`,
`alpha_shape=[8,8,9]`, `depth_shape=[8,8,9]`,
`max_rgba_gradient_abs_error=7.62939453125e-06`,
`finite_difference_max_abs_error=0.001068115234375`, and
`mps_full_frame_vjp_wall_clock_ms=2.1561333000136074` over 10 timed launches.
It also reports `shared_forward_boundary_scan_ratio=0.25` for `time_slabs=2`
with `candidate_mask_shape=[72,2]`.

This is still not a real full-frame renderer. The JSON explicitly marks
`world_foam_renderer_status=toy_full_frame_image_shape_only_existing_u_t_replay_op_no_true_u_v_t_camera_rays`.
There is no CSR candidate storage, real camera ray consumption, geometry or
topology gradient, trainer hook, or heldout-camera metric.

## Gate 1C Real-Ray Forward Boundary

This variant now contains a true camera-ray forward smoke:

- `realray_rgba_depth_replay` consumes `rays_f32 [R,6]` as world/model-space
  origin plus direction and `frame_t_f32 [R]` as normalized time.
- `boundary_f32` widens to `[B,5] = nx,ny,nz,nt,b`.
- `sites_f32` widens to `[S,5] = x,y,z,t,weight`.
- The Metal kernel solves each boundary cut along the actual ray:
  `s = -(dot(n_xyz, origin) + nt*t + b) / dot(n_xyz, direction)`.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_realray_replay_mps.py \
  --timing-iters 10 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate1_mps_realray_replay_smoke.json \
  --train-ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate1_mps_realray_train.ppm \
  --heldout-ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate1_mps_realray_heldout.ppm
```

The saved row matches the CPU real-ray reference within float32 tolerance:
train max RGB/alpha/depth errors are `3.5762786865234375e-07`,
`4.172325134277344e-07`, and `3.5762786865234375e-07`; heldout max
RGB/alpha/depth errors are `2.384185791015625e-07`,
`3.5762786865234375e-07`, and `3.5762786865234375e-07`.

This is still a linear per-sample forward baseline. It does not implement
temporal sharing, backward gradients, CSR candidates, trainer integration, or
heldout-quality promotion.

## Gate 2B Shared Real-Ray Forward Boundary

This variant now contains a true camera-ray shared forward smoke:

- `shared_realray_rgba_depth_replay` consumes static pixel-track rays
  `track_rays_f32 [K,6]`, frame times `frame_t_f32 [T]`, 4D power boundaries,
  and bitset candidates per `(track, time_slab)`.
- `candidate_mask_i32 [K*time_slabs,W]` uses `W=ceil(B/32)` int32-backed words,
  so the current 66-boundary real-ray scene does not regress to the toy
  31-boundary mask cap.
- The Metal kernel solves the same true-ray 4D crossing equation as Gate 1C,
  but scans only the slab candidate bitset for each pixel track and frame.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_shared_realray_replay_mps.py \
  --max-frames 2 \
  --render-size 32 \
  --time-slabs 1 \
  --timing-iters 10 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate2_mps_shared_realray_forward_smoke.json \
  --train-ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate2_mps_shared_realray_train.ppm \
  --heldout-ppm-out dynaworld/research_experiments/world_foam_lane2/results/gate2_mps_shared_realray_heldout.ppm
```

The saved row matches the direct CPU real-ray compositor within float32
tolerance. Train max RGB/alpha/depth errors are
`3.5762786865234375e-07`, `4.172325134277344e-07`, and
`3.5762786865234375e-07`; heldout max RGB/alpha/depth errors are
`2.384185791015625e-07`, `3.5762786865234375e-07`, and
`3.5762786865234375e-07`. For the two-frame gate, train direct/shared boundary
scans are `270336` / `135168` and heldout direct/shared scans are `135168` /
`67584`.

This is still forward-only. It does not implement real-ray backward gradients,
autograd, trainer integration, CSR candidates, or heldout-quality promotion.

## Gate 2C Shared Real-Ray VJP Boundary

This variant now contains a true camera-ray fixed-segment VJP smoke:

- `shared_realray_rgba_depth_vjp` uses the same bitset candidate layout as
  Gate 2B.
- It emits `grad_samples_rgba [K,T,S,4]`, where the final channel is density.
- It differentiates RGB, alpha, and expected-depth outputs only through fixed
  segment owners and lengths.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_shared_realray_vjp_mps.py \
  --max-frames 2 \
  --render-size 16 \
  --time-slabs 1 \
  --timing-iters 5 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate2c_mps_shared_realray_vjp_smoke.json
```

The saved row matches the CPU fixed-segment VJP reference. Train max
RGB/alpha/depth errors are `2.980232238769531e-07`,
`4.172325134277344e-07`, and `3.5762786865234375e-07`; heldout max
RGB/alpha/depth errors are `1.7881393432617188e-07`,
`2.980232238769531e-07`, and `2.384185791015625e-07`. Train and heldout max
RGBA-gradient errors are `4.842877388000488e-08` and
`4.470348358154297e-08`.

This is still a smoke VJP, not a trainer path. It materializes
`[K,T,S,4]` gradient samples and does not differentiate boundary cuts, owner
selection, sorting, masks, site positions, site weights, rays, camera geometry,
or topology.

## Gate 2D Shared Real-Ray Reduced VJP Boundary

This variant now contains a reduced true camera-ray fixed-segment VJP smoke:

- `shared_realray_rgba_depth_vjp_reduce` uses the same bitset candidate layout
  as Gate 2B and Gate 2C.
- It returns `grad_site_rgba [S,4]`, where the final channel is density.
- It does not allocate the Gate 2C `[K,T,S,4]` sample-gradient tensor inside
  the reduced op.
- Internally it materializes chunk partials shaped `[chunk_count,S,4]` and then
  finalizes them into `[S,4]`.
- `shared_realray_rgba_depth_autograd` wraps the forward op with a custom
  PyTorch backward that calls the reduced VJP and returns gradients for
  `site_rgba_f32` only.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_shared_realray_vjp_reduce_mps.py \
  --max-frames 2 \
  --render-size 16 \
  --time-slabs 1 \
  --timing-iters 5 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate2d_mps_shared_realray_reduced_vjp_smoke.json
```

The saved row matches both the CPU reduced VJP reference and the sum of the
Gate 2C materialized MPS oracle. Train and heldout reduced gradient shapes are
`[12,4]`; train and heldout partial gradient shapes are `[256,12,4]` and
`[128,12,4]`; train and heldout oracle sample-gradient shapes are
`[512,2,12,4]` and `[256,2,12,4]`. The partial gradient float count is `0.25x`
the Gate 2C oracle gradient-float count. Train and heldout max reduced
RGBA-gradient errors are `3.0517578125e-05` and `7.62939453125e-06`. Train and
heldout max autograd RGBA-gradient errors are `0.0`; train and heldout
autograd loss absolute errors are also `0.0`. Train and heldout reduced VJP
wall times are `4.916041599062737 ms` and `3.7805999992997386 ms`.

This is a correctness boundary, not the final fast trainer path. The first
chunked reducer avoids float atomics and races while reducing gradient storage,
but it still does not differentiate boundary cuts, owner selection, sorting,
masks, site positions, site weights, rays, camera geometry, or topology.

## Gate 2F CSR Candidate Storage Boundary

This variant now contains a CSR candidate-storage smoke for the shared real-ray
reduced VJP path:

- `shared_realray_rgba_depth_vjp_reduce_csr` consumes CSR candidate rows instead
  of the Gate 2D bitset mask.
- Per-track CSR is an exact parity layout for each `(pixel track, time slab)`.
- Tiled CSR is a candidate superset layout that stores one row per spatial tile
  and time slab.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_shared_realray_csr_candidate_storage_mps.py \
  --max-frames 2 \
  --render-size 16 \
  --time-slabs 1 \
  --tile-h 8 \
  --tile-w 8 \
  --timing-iters 5 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate2f_mps_shared_realray_csr_candidate_storage_smoke.json
```

The saved row reports zero CSR-vs-bitset RGB/alpha/depth and reduced-gradient
error for both per-track and tiled CSR. Tiled CSR storage is below the bitset
reference in this tiny smoke: train `3956 / 6144 = 0.6438802083333334x`,
heldout `1788 / 3072 = 0.58203125x`. Exact per-track CSR is larger than the
bitset at this scale (`14.584635416666666x` train), so it remains a parity
oracle, not the target storage layout.

This is still a storage-format smoke. It does not prove large-scale memory
behavior, geometry/topology gradients, trainer integration, or heldout quality.

## Gate 2G CSR Frame-Scaling Boundary

This variant now also contains a tiled CSR frame-scaling smoke:

- `smoke_shared_realray_csr_scaling_mps.py` runs the shared real-ray reduced
  VJP path at frame counts `2,4,8`.
- It keeps one tiled CSR row per spatial tile and time slab.
- It reports direct boundary-scan growth, shared candidate-build scan growth,
  tiled CSR candidate-iteration growth, storage growth, and MPS parity against
  the bitset reduced-VJP oracle.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_shared_realray_csr_scaling_mps.py \
  --frame-counts 2,4,8 \
  --render-size 32 \
  --time-slabs 1 \
  --tile-h 8 \
  --tile-w 8 \
  --timing-iters 3 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate2g_mps_shared_realray_csr_scaling_smoke.json
```

The saved row is `status=ok`. From 2 to 8 frames, direct scans grow `4.0x`
while shared candidate-build scans grow `1.0x`; tiled CSR candidate iterations
grow `3.9218009478672986x` on train and `3.915492957746479x` on heldout. At
8 frames, tiled CSR storage is `0.6080729166666666x` the train bitset storage
and `0.5651041666666666x` the heldout bitset storage. Tiled CSR MPS output and
reduced-gradient errors versus the bitset oracle are `0.0`.

This remains fixed-geometry scaling evidence, not a full trainer or heldout
quality comparison.

## Gate 2E Frozen-Geometry Autograd Overfit Boundary

This variant also contains a teacher-target parameter-update smoke:

- `smoke_shared_realray_autograd_overfit_mps.py` uses the same shared real-ray
  candidate layout as Gate 2D.
- It renders a teacher target with fixed 4D sites, fixed boundaries, and the
  original site RGBA/density values.
- It starts from a perturbed site-RGBA tensor and optimizes only that tensor
  through `shared_realray_rgba_depth_autograd`.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_shared_realray_autograd_overfit_mps.py \
  --max-frames 2 \
  --render-size 16 \
  --time-slabs 1 \
  --steps 25 \
  --lr 0.05 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate2e_mps_shared_realray_autograd_overfit_smoke.json
```

The saved row uses real train camera rays from DeepView `03_Dog`, two frames at
16px, one time slab, 12 fixed 4D sites, and 66 fixed boundaries. Loss drops
from `0.004183291457593441` to `0.0001105795890907757`; the first-step
gradient abs sum is `0.09477987885475159`; and the max site-RGBA parameter
update is `0.9012540578842163`.

This is still not a real-target trainer. It proves that PyTorch can update
frozen-geometry site RGBA/density through the shared real-ray Metal path, but
it does not optimize site positions, weights, boundary topology, ray/camera
geometry, sorting, or ownership.

## Gate 3 Frozen-Geometry Real-Target Training Boundary

This variant now contains a tiny real-target training smoke:

- `smoke_shared_realray_real_target_train_mps.py` uses the same shared real-ray
  candidate layout as Gate 2D and Gate 2E.
- It optimizes only `site_rgba_f32` against actual train RGB frames.
- It keeps 4D sites, weights, boundaries, ray geometry, sorting, masks, and
  ownership fixed.

Current smoke:

```bash
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_shared_realray_real_target_train_mps.py \
  --max-frames 2 \
  --render-size 16 \
  --time-slabs 1 \
  --steps 40 \
  --lr 0.03 \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate3_mps_shared_realray_real_target_train_smoke.json
```

The saved row uses real train camera rays from DeepView `03_Dog`, two frames at
16px, one time slab, 12 fixed 4D sites, and 66 fixed boundaries. Train RGB MSE
drops from `0.04258020222187042` to `0.025183267891407013`; train PSNR
improves from `13.707922803417098` to `15.988879146126294`; and the max
site-RGBA parameter update is `1.1732829809188843`.

This is still a tiny frozen-geometry smoke. It is not a full trainer, not a
heldout-quality baseline, and not a replacement for geometry/topology
gradients or a large-scale CSR/tiled candidate-storage benchmark.

## Buffer Contract

The Metal kernel currently expects:

```text
buffer(0): device const WF2ScreenTimeBeam*
buffer(1): constant WF2GridConfig&
buffer(2): device WF2BeamEventCount*
buffer(3): device atomic_uint* global_event_count
buffer(4): device WF2BoundaryEvent*
```

Dispatch one thread per beam. `global_event_count` should be zeroed by the
caller before dispatch. If event writing is disabled, the global counter still
accumulates the total number of counted boundary crossings.

The power-boundary kernel expects:

```text
buffer(0): device const WF2PowerBoundary3D*
buffer(1): device const WF2PowerBeamSlab*
buffer(2): constant WF2PowerBoundaryConfig&
buffer(3): device WF2PowerBoundaryCount*
```

Dispatch one thread per beam slab. It reports `boundary_event_count`,
`invalid_denominator_count`, and flags invalid beams or near-zero denominators.

The tensor bridge kernel expects:

```text
buffer(0): device const float* boundary_f32  # [B,4]: nx,nz,nt,b
buffer(1): device const uint*  boundary_u32  # [B,4]: left,right,0,0
buffer(2): device const float* beam_f32      # [M,5]: u,t0,t1,near,far
buffer(3): device const uint*  beam_u32      # [M,4]: payload,flags,0,0
buffer(4): device const int*   config_i32    # [2]: boundary_count,beam_count
buffer(5): device const float* config_f32    # [2]: camera_velocity_x,invalid_epsilon
buffer(6): device uint*        counts_u32    # [M,8]: WF2PowerBoundaryCount fields
```

The shared replay tensor kernel expects:

```text
buffer(0):  device const float* boundary_f32       # [B,4]: nx,nz,nt,b
buffer(1):  device const uint*  candidate_mask_u32 # [M]: int32-backed boundary-bit mask per beam, bits 0..30
buffer(2):  device const float* sites_f32          # [S,4]: x,z,t,weight
buffer(3):  device const float* site_signal_f32    # [S]
buffer(4):  device const float* beam_f32           # [M,5]: u,t0,t1,near,far
buffer(5):  device const float* frame_t_f32        # [T]
buffer(6):  device const float* grad_output_f32    # [M,T]
buffer(7):  device const int*   config_i32         # [5]: B,M,S,T,time_slabs
buffer(8):  device const float* config_f32         # [1]: camera_velocity_x
buffer(9):  device float*       output_f32         # [M,T]
buffer(10): device float*       grad_sample_f32    # [M,T,S]
```

The shared RGB replay tensor kernel uses the same buffers, replacing the scalar
signal and gradient payloads with RGB payloads:

```text
buffer(3):  device const float* site_rgb_f32        # [S,3]
buffer(6):  device const float* grad_output_rgb_f32 # [M,T,3]
buffer(9):  device float*       output_rgb_f32      # [M,T,3]
buffer(10): device float*       grad_sample_rgb_f32 # [M,T,S,3]
```

The shared RGBA/depth replay tensor kernel uses the same candidate and geometry
buffers, replacing gradient payloads with forward compositor outputs:

```text
buffer(3):  device const float* site_rgba_f32   # [S,4]: r,g,b,density
buffer(6):  device const int*   config_i32      # [5]: B,M,S,T,time_slabs
buffer(7):  device const float* config_f32      # [1]: camera_velocity_x
buffer(8):  device float*       output_rgb_f32  # [M,T,3]
buffer(9):  device float*       output_alpha_f32 # [M,T]
buffer(10): device float*       output_depth_f32 # [M,T]
```

The shared RGBA/depth VJP tensor kernel adds upstream gradients and per-ray
per-site gradient samples:

```text
buffer(6):  device const float* grad_rgb_f32       # [M,T,3]
buffer(7):  device const float* grad_alpha_f32     # [M,T]
buffer(8):  device const float* grad_depth_f32     # [M,T]
buffer(14): device float*       grad_sample_rgba_f32 # [M,T,S,4]
```

The real-ray forward tensor kernel uses a separate, non-shared per-sample ABI:

```text
buffer(0): device const float* boundary_f32     # [B,5]: nx,ny,nz,nt,b
buffer(1): device const float* sites_f32        # [S,5]: x,y,z,t,weight
buffer(2): device const float* site_rgba_f32    # [S,4]: r,g,b,density
buffer(3): device const float* rays_f32         # [R,6]: ox,oy,oz,dx,dy,dz
buffer(4): device const float* frame_t_f32      # [R]
buffer(5): device const int*   config_i32       # [3]: B,R,S
buffer(6): device const float* config_f32       # [4]: near,far,invalid_eps,transmittance_threshold
buffer(7): device float*       output_rgb_f32   # [R,3]
buffer(8): device float*       output_alpha_f32 # [R]
buffer(9): device float*       output_depth_f32 # [R]
```

The shared real-ray forward tensor kernel uses a bitset candidate ABI:

```text
buffer(0):  device const float* boundary_f32       # [B,5]: nx,ny,nz,nt,b
buffer(1):  device const uint*  candidate_mask_u32 # [K*time_slabs,W]: bitset words
buffer(2):  device const float* sites_f32          # [S,5]: x,y,z,t,weight
buffer(3):  device const float* site_rgba_f32      # [S,4]: r,g,b,density
buffer(4):  device const float* track_rays_f32     # [K,6]: ox,oy,oz,dx,dy,dz
buffer(5):  device const float* frame_t_f32        # [T]
buffer(6):  device const int*   config_i32         # [6]: B,K,S,T,time_slabs,W
buffer(7):  device const float* config_f32         # [4]: near,far,invalid_eps,transmittance_threshold
buffer(8):  device float*       output_rgb_f32     # [K,T,3]
buffer(9):  device float*       output_alpha_f32   # [K,T]
buffer(10): device float*       output_depth_f32   # [K,T]
```

The shared real-ray VJP tensor kernel adds upstream gradients and per-ray
per-site gradient samples:

```text
buffer(6):  device const float* grad_rgb_f32          # [K,T,3]
buffer(7):  device const float* grad_alpha_f32        # [K,T]
buffer(8):  device const float* grad_depth_f32        # [K,T]
buffer(9):  device const int*   config_i32            # [6]: B,K,S,T,time_slabs,W
buffer(10): device const float* config_f32            # [4]: near,far,invalid_eps,transmittance_threshold
buffer(14): device float*       grad_sample_rgba_f32  # [K,T,S,4]
```

The public shared real-ray reduced VJP op uses the same inputs and returns a
site-gradient tensor directly:

```text
buffer(6):  device const float* grad_rgb_f32          # [K,T,3]
buffer(7):  device const float* grad_alpha_f32        # [K,T]
buffer(8):  device const float* grad_depth_f32        # [K,T]
buffer(9):  device const int*   config_i32            # [6]: B,K,S,T,time_slabs,W
buffer(10): device const float* config_f32            # [4]: near,far,invalid_eps,transmittance_threshold
buffer(11): device float*       grad_site_rgba_f32    # [S,4]
```

Internally, the current implementation launches a partial reducer and finalizer:

```text
partial_reduce buffer(11): device float* partial_grad_site_rgba_f32 # [chunk_count,S,4]
finalize_reduce buffer(0): device const float* partial_grad_site_rgba_f32 # [chunk_count,S,4]
finalize_reduce buffer(2): device float* grad_site_rgba_f32 # [S,4]
```

The CSR reduced VJP op keeps the same forward and reduced-gradient outputs but
replaces bitset words with explicit CSR candidate rows:

```text
buffer(0):  device const float* boundary_f32              # [B,5]: nx,ny,nz,nt,b
buffer(1):  device const int*   row_index_i32             # [K]: maps pixel track to CSR row
buffer(2):  device const int*   candidate_row_offsets_i32 # [row_count*time_slabs+1]
buffer(3):  device const int*   candidate_boundary_ids_i32 # [candidate_count]
buffer(4):  device const float* sites_f32                 # [S,5]: x,y,z,t,weight
buffer(5):  device const float* site_rgba_f32             # [S,4]: r,g,b,density
buffer(6):  device const float* track_rays_f32            # [K,6]: ox,oy,oz,dx,dy,dz
buffer(7):  device const float* frame_t_f32               # [T]
buffer(8):  device const float* grad_rgb_f32              # [K,T,3]
buffer(9):  device const float* grad_alpha_f32            # [K,T]
buffer(10): device const float* grad_depth_f32            # [K,T]
buffer(11): device const int*   config_i32                # [7]: B,K,S,T,time_slabs,row_count,candidate_count
buffer(12): device const float* config_f32                # [4]: near,far,invalid_eps,transmittance_threshold
buffer(13): device float*       output_rgb_f32            # [K,T,3]
buffer(14): device float*       output_alpha_f32          # [K,T]
buffer(15): device float*       output_depth_f32          # [K,T]
buffer(16): device float*       partial_grad_site_rgba_f32 # [chunk_count,S,4]
```

The public autograd wrapper does not add a new Metal ABI. It calls the shared
real-ray forward op in `forward()` and the reduced VJP op in `backward()`, then
returns a gradient only for `site_rgba_f32`.

## Missing For GPU Execution

- GPU readback parity harness for the Gate 0 power-boundary fixture and
  hand-authored screen-time beam fixtures beyond the current two-velocity count
  smoke, Gate 0.6 shared-replay smoke, Gate 0.7 toy RGB strip, Gate 0.8 toy
  composite strip, Gate 0.9 fixed-segment VJP, and Gate 0.95 slab-indexed
  mask VJP, toy full-frame-shaped VJP, true-ray per-sample forward, and
  true-ray shared forward, reduced fixed-segment VJP, CSR candidate-storage,
  and CSR frame-scaling smokes;
- world-foam primitive projection into `WF2ScreenTimeBeam`;
- radius expansion semantics for event counts and future tile spans;
- true `u/v/t` camera-ray tile-span emission, sorting, general geometry and
  topology gradients, and ownership of trainer backward propagation;
- integration with fast-mac package loading, Dynaworld training configs, viewer
  formats, and benchmark matrices.

Until those exist, this is not an executable full renderer variant. The bridge
source now launches count, shared scalar replay, shared RGB strip replay,
forward-only composite strip, fixed-segment site-RGBA VJP, and slab-indexed
mask VJP kernels, plus true-ray per-sample, shared-forward, fixed-segment
shared-real-ray VJP, reduced site-gradient VJP, CSR candidate-storage and
frame-scaling reduced VJP smokes, frozen-geometry site-RGBA autograd, and a
tiny real-target training smoke, but not a production trainer or general
backward pass.

## Comparison Boundary

Compare this lane against STAR-UVT and dynamic splat baselines only after it has
a full train/eval path, not just the current 16px/2f frozen-geometry training
smoke. The first fair comparison should use the same video, frame count,
resolution, camera split, train schedule, and held-out-camera metric set.

The comparison should answer three separate questions:

- event/scaffold parity: does the Metal path match CPU toy counts and fixture
  event records exactly?
- renderer/trainer viability: can it run end-to-end without host readback loops
  or per-frame CPU scheduling bottlenecks?
- model quality and cost: on the same data, does it beat or complement STAR-UVT
  and the current dynamic FasterGS/dynamic-splat baseline on held-out views,
  wall-clock, peak memory, and artifact stability?

Do not promote this lane based on shader compilation, ABI checks, or source-view
training quality alone.
