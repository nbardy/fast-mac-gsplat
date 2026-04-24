# v8 Hardware Raster Theory Remaining Work

Date: 2026-04-24

Scope: audit the advanced Metal rasterization path after the v8 hardware eval
and train scaffolds. This is a theory and implementation note only; it does not
change the renderer.

## Verdict

The advanced Metal path is still an eval-first experiment, not a training
replacement for `v8_direct`.

The current evidence is internally consistent:

- `v7_finished` proves instanced Gaussian quads and fixed-function blending can
  produce a hardware forward image, but it packs inputs on CPU, waits on a
  private command buffer, reads the texture back, and its backward shape is not
  training-grade.
- `v7_frontk` makes hardware-backward state more correct, but saves per-pixel
  front-K state and scans too broadly.
- `v7_tiled_capture` improves front-K capture by using tile bins, but still
  builds bins on CPU, returns CPU saved state, and uses broad per-pixel gradient
  atomics.
- `v7_hybrid_v5style` becomes practical for training by routing training back
  to v5 compute; hardware eval still has texture readback.
- `v8_direct` has the right training math shape: GPU tile bins, sorted tile IDs,
  per-tile stop counts, reverse replay, and tile-reduced global atomics.

The hardware path is worth pursuing only if it shares v8 GPU visibility and
returns GPU-resident Torch/MPS output. If render output or saved state crosses
CPU memory in the timed path, keep it out of training and use it only as a
diagnostic renderer.

## Metal Feature Requirements

| Feature | Actually Needed? | Why | Main Risk |
|---|---:|---|---|
| Render pipeline | Yes for hardware eval | Generates covered pixels from Gaussian quads and uses raster hardware for fragment generation. The v7 line already uses this shape. | Fixed-function blending hides intermediate transmittance and stop state, so it is not enough for training. |
| Tile shaders | Needed for the serious eval/training attempt, not for the first interop probe | Initialize and flush tile-local state such as `C`, `T`, flags, and optional stop metadata. | Feature availability and API plumbing; tile shader work must be encoded in a render pass, not the current v8 compute-kernel launcher. |
| Imageblocks | Yes if replacing global per-fragment state traffic | Keep per-pixel `C/T/stopped` on chip while fragments for a tile run. This is the only plausible way hardware forward beats compute without global read/write per fragment. | Imageblock memory is small; do not put front-K/history in it. 16x16 fp32 `C/T` is 4 KiB before flags; 32x32 is 16 KiB before flags. |
| Raster order groups or equivalent ordered access | Required when programmable fragment writes update same-pixel state | Gaussian alpha compositing is ordered and non-commutative. Overlapping same-pixel fragments must observe stable depth order. | Correctness feature, not a speed feature. Clustered overlap can serialize and erase the forward win. |
| Indirect draw / ICB | Needed after output interop is proven | Hardware eval must consume GPU-resident v8 tile bins or a GPU-built draw stream. CPU draw-list construction repeats the v7 system bottleneck. | ICB generation can become its own kernel/encoding bottleneck. Do not start here before proving GPU-native output. |
| MPS/Torch interop | Mandatory first gate | Render output and any saved state must land in Torch-visible MPS storage or be copied by GPU work on the same scheduling stream. | This likely needs lower-level PyTorch/Metal integration than current `DynamicMetalShaderLibrary` compute launches. `waitUntilCompleted`, `getBytes`, `[buf contents]`, and CPU tensor staging disqualify the path. |

Fixed-function blending is useful for a narrow eval probe, but not enough for
exact training. Imageblock/ROG state is the real hardware-raster experiment.

## Math Invariants

For a pixel center `p = (x + 0.5, y + 0.5)`, v8 processes visible splats in
stable depth order:

```text
d_i       = p - mean_i
power_i   = -0.5 * (a_i*dx_i^2 + 2*b_i*dx_i*dy_i + c_i*dy_i^2)
raw_i     = opacity_i * exp(power_i)
alpha_i   = min(max_alpha, raw_i)
visible_i = power_i <= 0 && alpha_i >= alpha_threshold

C_0 = 0
T_0 = 1
C_{i+1} = C_i + T_i * alpha_i * color_i
T_{i+1} = T_i * (1 - alpha_i)
stop when T <= transmittance_threshold
out = C_M + T_M * background
```

`M` is the exact processed prefix for that pixel. It is not necessarily the tile
list length and not necessarily the per-tile max stop count.

Backward must see the same prefix and gates:

```text
T_cur = T_final
gT    = dot(grad_out, background)

for i in reverse(M):
  T_prev  = T_cur / max(1 - alpha_i, eps)
  dot_gc  = dot(grad_out, color_i)
  g_alpha = T_prev * (dot_gc - gT)
  g_color = grad_out * (T_prev * alpha_i)

  gate    = 1 if raw_i < max_alpha else 0
  g_raw   = g_alpha * gate
  g_power = g_raw * raw_i

  da      = g_power * (-0.5 * dx_i^2)
  db      = g_power * (-dx_i * dy_i)
  dc      = g_power * (-0.5 * dy_i^2)
  dmean_x = g_power * (a_i*dx_i + b_i*dy_i)
  dmean_y = g_power * (b_i*dx_i + c_i*dy_i)
  dopacity = g_raw * raw_i / max(opacity_i, eps)

  gT      = alpha_i * dot_gc + (1 - alpha_i) * gT
  T_cur   = T_prev
```

The hardware path threatens these invariants in specific ways:

- draw or fragment order can diverge from v8's stable sorted IDs;
- fixed blending produces final color but not exact `T_final`, `M`, or per-pixel
  stop index;
- fixed blending does not early-stop, so later fragments may affect alpha/color
  unless the shader owns `T/stopped` state;
- tile stop count is only a conservative per-tile maximum. Backward must still
  recompute per-pixel `end_i` and `T_final`;
- front-K state is only exact when `K` covers the true processed prefix or an
  exact overflow path runs;
- suffix-from-final-color formulas are fragile when early stop, alpha clamp, and
  visibility gates differ even slightly;
- per-fragment global gradient atomics are mathematically accumulative but
  operationally unacceptable. Gradients must be reduced across tile pixels
  before global atomics.

## Minimal Implementation Sequence

1. **Interop probe, no Gaussian complexity.**
   Render a known texture/pattern into Torch-visible MPS output or GPU-copy it
   into an MPS tensor on the same stream.

   Kill gate: any `waitUntilCompleted`, `getBytes`, `[buf contents]`, CPU tensor
   staging, or private queue synchronization in the timed path.

2. **Fixed-blend hardware eval, separate from training.**
   Rebuild the v7-style render pipeline only as an eval probe. Use sorted inputs
   and compare image parity against `v8_direct`.

   Kill gate: no image parity, CPU output staging remains, or 4K/64K forward is
   not at least 25% faster than v8 compute forward.

3. **Shared GPU visibility and draw stream.**
   Hardware eval must consume v8 `tile_counts`, `tile_offsets`, and `binned_ids`.
   Add GPU-generated indirect draws/ICB only after the output interop gate is
   passed.

   Kill gate: CPU tile-bin construction, CPU per-tile loops, or per-frame CPU
   draw-list construction.

4. **Imageblock/ROG C/T forward.**
   Add tile shader init/flush and imageblock `C/T/stopped` state. Start fp32.
   Capture optional `final_T` or `pixel_stop` only behind explicit state modes.

   Kill gate: ordered fragment serialization removes the forward win, imageblock
   pressure lowers occupancy too far, or fp32 parity fails.

5. **Training mode: hardware forward + v8 compute backward.**
   Default state is `tile_stop` plus recompute in backward. Test `final_T` and
   `pixel_stop` as ablations, not defaults.

   Kill gate: gradients differ beyond current v8 tolerances, default memory
   overhead exceeds 25% of v8 state, or forward+backward does not beat v8 direct
   on 4K/64K B=1 and 512/6K B=4.

6. **Only then consider render-assisted backward.**
   Keep v8's reverse recurrence and tile reductions. Try deferred partial
   gradients only if profiling proves global atomics dominate.

   Kill gate: per-pixel/per-fragment global atomics, capacity launches, silent
   overflow, or any CPU readback in the hot path.

## State Memory Estimates

Units are MiB. `4K` here means the repo benchmark size `4096x4096`; UHD
`3840x2160` is about half these 4K pixel-state numbers. Tile counts assume
16x16 tiles. Visibility list memory is scene-dependent: `binned_ids` costs
4 bytes per tile reference, so every 100M refs is about 381 MiB.

| State mode | 512 B=1/B=4 | 1080p B=1/B=4 | 4096 B=1/B=4 |
|---|---:|---:|---:|
| `tile_stop_counts` i32 | 0.004 / 0.016 | 0.031 / 0.125 | 0.25 / 1.00 |
| `final_T` fp32 | 1.00 / 4.00 | 7.91 / 31.64 | 64.00 / 256.00 |
| `pixel_stop` i32 | 1.00 / 4.00 | 7.91 / 31.64 | 64.00 / 256.00 |
| `final_T + pixel_stop` | 2.00 / 8.00 | 15.82 / 63.28 | 128.00 / 512.00 |
| RGB fp32 output tensor | 3.00 / 12.00 | 23.73 / 94.92 | 192.00 / 768.00 |
| RGBA32F render target | 4.00 / 16.00 | 31.64 / 126.56 | 256.00 / 1024.00 |
| v7 front-K, K=2 | 3.25 / 13.00 | 25.71 / 102.83 | 208.00 / 832.00 |
| v7 front-K, K=4 | 6.25 / 25.00 | 49.44 / 197.75 | 400.00 / 1600.00 |

The default training state should remain `tile_stop_counts` plus sorted
`binned_ids`. Per-pixel `final_T` and `pixel_stop` are plausible ablations.
Front-K/history should remain a debug or tiny-scene validation mode.

## Expected Bottlenecks

- **Atomics:** v8's acceptable default is 9 global float atomics per
  contributing splat/tile after reducing 256 pixel lanes. v7-style per-pixel
  atomics are a stop condition.
- **Bandwidth:** full-image traffic dominates quickly: RGB fp32 output is
  192 MiB at 4096 B=1, and RGBA32F render target is 256 MiB. CPU readback doubles
  down on the worst traffic.
- **Imageblock pressure:** fp32 `C/T` is 16 bytes per pixel before flags. A
  16x16 tile is manageable; 32x32 can push occupancy down, especially with stop
  metadata.
- **Ordered fragment serialization:** ROG/imageblock correctness may serialize
  clustered hot pixels. This must be profiled on clustered and layered traces,
  not only uniform scenes.
- **CPU waits and staging:** v7's `waitUntilCompleted`, `getBytes`, CPU bin
  construction, CPU saved state, and CPU gradient buffer extraction are exactly
  the system costs v8 must avoid.
- **Command queue interop:** a private Metal command queue can race or force
  synchronization with PyTorch MPS. Hardware render work must be ordered with
  Torch's MPS stream or an equivalent shared-event path.
- **Draw construction:** per-splat or per-tile CPU draw work can dominate before
  raster work starts. ICB helps only if generated from GPU metadata.

## What Can Be Done Now

Within the current repo and custom-op shape:

- keep `variants/v8_hw_eval` and `variants/v8_hw_train` as fail-closed
  scaffolds;
- strengthen v8 compute training: fixed/cached pair capacity, GPU overflow
  compaction, in-place overflow patching, counters, and heavy-tile ablations;
- add benchmark-only counters for total refs, max/p95 refs, stop ratio, overflow
  count, heavy count, atomic estimates, and CPU wait/readback count;
- write standalone hardware eval probes that may use CPU staging outside the
  timed path for validation, but do not promote them;
- implement training state ablation APIs in compute first (`tile_stop`,
  `final_T`, `pixel_stop`) to quantify backward replay savings before involving
  render encoders.

## What Likely Needs Lower-Level Integration

These are not well served by the current v8 `DynamicMetalShaderLibrary` compute
launcher alone:

- creating render pipelines with tile shaders, imageblocks, and ROG state;
- encoding render passes on the same command stream used by PyTorch MPS;
- creating or borrowing MTLTextures/MTLBuffers that back Torch MPS tensors;
- GPU blit/compute copy from a render texture into a Torch tensor without CPU
  staging;
- GPU-generated indirect draw or ICB execution from v8 visibility metadata;
- persistent capacity buffers whose ownership is shared cleanly with autograd.

If those hooks are not available without private queues and CPU waits, the
hardware path should remain eval-only and v8 compute should remain the training
default.

## Files Changed

- `docs/v8_hw_theory_remaining_work.md`

## Key Conclusions

- The first hard gate is not imageblock math; it is GPU-native Torch/MPS render
  output interop.
- Fixed-function blending can be an eval probe, but exact training needs
  ordered `C/T/stopped` state or a compute backward that reconstructs the exact
  forward prefix.
- `tile_stop_counts` remains the only acceptable default training state.
  `final_T`, `pixel_stop`, and front-K are memory-expensive ablations.
- Hardware training should be killed quickly if it reintroduces CPU staging,
  per-pixel atomics, unbounded per-pixel history, or capacity-sized launches.
