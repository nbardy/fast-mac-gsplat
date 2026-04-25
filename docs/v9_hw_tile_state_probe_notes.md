# V9 HW Tile State Probe Notes

Scope: `variants/v9_hw_tile_state_probe`.

This probe extends the direct Torch/MPS render-target interop path with a small
Metal tile shader/imageblock experiment. It is not a Gaussian renderer yet.

## Results on Apple M4

`probe_hw_interop(compile_pipelines=True, compile_advanced=True)` reports:

| Layout | Logical state | Sample length | 16x16 imageblock | 32x32 imageblock |
| --- | ---: | ---: | ---: | ---: |
| `half4_baseline` | 8 B | 24 B | 6,144 B | 24,576 B |
| `ct_fp32` | 16 B | 32 B | 8,192 B | 32,768 B |
| `ct_stop_fp32_u32` | 20 B | 48 B | 12,288 B | 49,152 B |
| `ct_stop_flags_fp32_u32x2` | 24 B | 48 B | 12,288 B | 49,152 B |

The previous 24 B sample result was the `half4` baseline. The useful Gaussian
state is more expensive:

- `float4 c_t`: `C.rgb` plus transmittance `T`;
- `uint stop_count`: accepted contribution count, or backward stop index;
- `uint flags`: stopped bit and overflow/debug bits.

Adding one `uint` to `float4` moves the sample from 32 B to 48 B on this M4.
Adding a second `uint` is free under the measured alignment, so keeping both
`stop_count` and `flags` is reasonable if this layout is used.

## 4K State Cost

For 3840x2160:

- pixels: 8,294,400;
- 16x16 tiles: 32,400;
- 32x32 tiles: 8,160;
- final backward state (`final_T: float32`, `stop_count: uint32`): 66,355,200 B;
- output `RGBA32F`: 132,710,400 B.

Full-frame-equivalent imageblock pressure for `ct_stop_flags_fp32_u32x2`:

- 16x16 accounting: 12,288 B/tile * 32,400 = 398,131,200 B;
- 32x32 accounting: 49,152 B/tile * 8,160 = 401,080,320 B.

This is not a global allocation; imageblock storage is transient tile-local
memory. The full-frame number is only a pressure/comparison estimate.

## Execution Probe

`run_tile_state_execution_probe(32, 32, 32)`:

- creates a Torch MPS `[H,W,4] float32` target;
- creates an `RGBA32Float` texture view over the tensor buffer;
- begins a render pass with that direct texture as color attachment;
- dispatches a tile function with `dispatchThreadsPerTile`;
- writes one `float4` report per tile to a Torch MPS report tensor;
- returns both tensors without native CPU readback.

The test validates the render target clear and the tile report after the op
returns by copying tensors to CPU in Python.

## API Limitations Found

Metal classifies these structs as explicit-layout imageblocks. Consequences:

- `imageblock<T>.read()` is not available; the compile error is:
  `no member named 'read' in 'metal::imageblock<..., imageblock_layout_explicit>'`.
- The tile shader must use `imageblock.data(coord, index, imageblock_data_rate::color)`.
- A 16x16 dispatch on this Apple M4 reported a 32x32 imageblock footprint; the
  runnable probe therefore uses a 32x32 dispatch and a pipeline compiled with
  `maxTotalThreadsPerThreadgroup = 1024`.
- A same-shader experiment that wrote through `data(coord)`, barriered with
  `mem_threadgroup_imageblock`, and then read `data(0,0)` reported zero. The
  current execution probe reports the lane's written state and imageblock
  dimensions, so it proves dispatch but not a complete init/flush readback path.

## Layout Plan

For eval-only:

```text
imageblock per pixel:
  float4 c_t   // C.rgb, T
```

This is 32 B/sample measured on M4.

For eval plus backward metadata:

```text
imageblock per pixel:
  float4 c_t       // C.rgb, T
  uint stop_count  // accepted contributions or stop index
  uint flags       // stopped bit, overflow/debug bits

global optional per pixel:
  float final_T
  uint stop_count
```

This is 48 B/sample measured on M4. Do not store front-K or per-splat history in
imageblock; keep backward capture to bounded global state or recompute.

## Assessment

Promising:

- tile/imageblock pipelines compile on Apple M4;
- imageblock memory is queryable and deterministic;
- tile shader dispatch can run inside the direct MPS render pass.

Risky:

- useful `C/T/stop` state costs 48 B/sample, roughly 2x the half4 baseline;
- actual execution reports a 32x32 tile footprint, so per-active-tile state is
  48 KiB for the recommended layout;
- exact tile init/update/flush semantics still need a working read/flush shader
  or a fragment/imageblock pair before Gaussian compositing can be considered
  de-risked.
