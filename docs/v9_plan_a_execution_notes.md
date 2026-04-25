# V9 Plan A Execution Notes

Date: 2026-04-25

Scope: Direction A, smallest Metal tile/imageblock exact-forward spike.

Variant:

```text
variants/v9_hw_tile_exact_probe
```

This is not full training, not Gaussian parity, and not backward. It is the
smallest runtime proof that a Metal render pass can:

1. initialize explicit imageblock per-pixel state;
2. update that state from ordered fragments with fixed-function blending
   disabled;
3. resolve the state into a direct Torch/MPS tensor render target;
4. avoid native CPU readback or ICB execution.

## What Was Implemented

The variant was forked from `v9_hw_tile_state_probe` and adds one isolated API:

```python
run_tile_exact_overlap_probe(height=32, width=32, tile_size=16)
```

Native path:

```text
render pass over direct buffer-backed RGBA32F Torch/MPS target
  tile pipeline: v9_exact_clear_tile
    PixelState.c_t = (0, 0, 0, 1)
    PixelState.stop_count = 0
    PixelState.flags = 0

  render pipeline: v9_exact_update_fs
    draw 2 fullscreen instances
    blending disabled
    color write mask disabled
    fragment input/output uses [[imageblock_data]]
    PixelState members use [[raster_order_group(0)]]

  tile pipeline: v9_exact_resolve_tile
    output float4(C.rgb, T) to the render target
```

The two fragments are fixed constant-alpha splats:

```text
splat 0: alpha = 0.25, color = red
splat 1: alpha = 0.50, color = green

C0 = 0
T0 = 1

C1 = C0 + T0 * 0.25 * red   = (0.25, 0, 0)
T1 = T0 * (1 - 0.25)        = 0.75

C2 = C1 + T1 * 0.50 * green = (0.25, 0.375, 0)
T2 = T1 * (1 - 0.50)        = 0.375
```

Expected render target:

```text
float4(0.25, 0.375, 0.0, 0.375)
```

This deliberately tests the missing primitive, not bbox generation or Gaussian
evaluation.

## Runtime Result

On Apple M4:

```text
tile_exact_overlap_probe_available = true
tile_exact_imageblock_sample_length = 48 B
tile_exact_imageblock_memory_16x16 = 12,288 B
tile_exact_imageblock_memory_32x32 = 49,152 B
tile_exact_overlap_max_abs_err = 0.0
```

That means explicit `C/T/stop/flags` imageblock state can survive:

```text
tile init -> fragment imageblock update -> tile resolve
```

inside one render pass and land in a Torch/MPS tensor.

## 32x32 Tile Finding

The same exact pipeline compiles for 32x32 dimensions and reports 49,152 B of
imageblock memory, but runtime render encoder creation failed on Apple M4:

```text
RuntimeError: failed to create tile exact render command encoder
```

The working API is therefore fail-closed to 16x16 tiles. The likely issue is
tile-memory pressure from 48 B/sample state plus the RGBA32F color attachment.
This must be measured more carefully before trying 32x32 again.

## What This Proves

Proved:

- explicit-layout tile clear using `imageblock<T, imageblock_layout_explicit>`;
- fragment update using `[[imageblock_data]]`;
- `[[raster_order_group(0)]]` state members compile and run;
- fixed-function blending can be disabled while fragment code performs exact
  `C/T` recurrence;
- tile resolve can emit final state to the direct MPS output target;
- no ICB execute is needed;
- no native CPU readback is used.

Not proved:

- Gaussian ellipse bbox generation in the exact path;
- tile-ref draw stream ingestion from V8 bins;
- stable depth-ordered multi-splat V8 parity;
- V8 `tile_stop_counts` semantics;
- final_T or pixel_stop output for backward;
- backward gradients;
- performance at real scene sizes.

## Accuracy Implication

This removes one major unknown: Metal fragment/imageblock update semantics can
represent the exact scalar recurrence for an ordered two-fragment overlap:

```text
C += T * alpha * color
T *= 1 - alpha
```

The remaining accuracy bug is now higher level. The exact path must ensure that
the fragments arrive in the same stable front-to-back candidate order as V8, and
that invisible candidates and early-stop prefix state match V8's backward
contract.

In particular, a fragment-only path still does not naturally account for V8's
tile-local candidate prefix semantics, because pixels that receive no fragment
for a candidate will not observe that candidate ordinal.

## Next Kill Gate

Next A gate:

```text
A1: replace the two fixed fullscreen splats with 2-4 projected Gaussian quads
    in stable input order, still 16x16 tiles, still no tile bins.
```

Pass condition:

```text
tiny overlapping Gaussian output == CPU/V8-style C/T reference within 1e-5
```

Fail condition:

```text
fragment order is nondeterministic, ROG serializes too heavily, or exact output
requires CPU staging / ICB execute / 32x32 tile state.
```

After A1, the next real training gate is not another imageblock compile test.
It is a GPU-resident tile-ref draw stream:

```text
V8 sorted bins -> GPU draw records -> clipped Gaussian quads -> exact imageblock C/T
```

Backward should remain V8 compute replay until forward parity and state capture
are proven.

## Commands Run

```text
python3 -m py_compile torch_gsplat_bridge_v9_hw_tile_exact/interop.py tests/interop_check.py
python3 setup.py build_ext --inplace
python3 tests/interop_check.py
```

Reference syntax checked against Apple's official imageblock OIT sample:

```text
https://developer.apple.com/documentation/metal/implementing-order-independent-transparency-with-image-blocks
```
