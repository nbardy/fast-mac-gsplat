# V9 Parallel Exploration Plan

## Forks

The working interop base is `variants/v9_hw_interop_probe`. It proves direct
Metal render output into Torch/MPS tensor storage through a buffer-backed
`RGBA32Float` render target.

Three isolated forks explore the next choices:

| Fork | Purpose | Kill Gate |
|---|---|---|
| `variants/v9_hw_fixed_eval_probe` | First real eval renderer using Gaussian tensor inputs and direct MPS render output. | Cannot beat v8 forward once draw setup and output format are included. |
| `variants/v9_hw_tile_state_probe` | Tile/imageblock state for `C/T/stop` and later backward capture. | Imageblock memory, ordering, or API constraints make exact compositing slower than v8 compute. |
| `variants/v9_hw_draw_formats_probe` | Output format, row alignment, ICB, and draw-stream strategy. | We cannot avoid RGBA bandwidth/copies or cannot generate draw work cheaply enough. |

## Current Hard Facts

- Direct render-to-MPS tensor storage works on Apple M4.
- Blit fallback works but is too expensive at high resolution.
- Tile/imageblock compile probe works on Apple M4.
- Raster order groups are reported supported on Apple M4.
- ICB allocation works; execution is still unproven.

## Next Mainline Candidate

The next serious version should be fixed eval before tile/imageblock training:

```text
output:
  direct buffer-backed RGBA32F MPS tensor

draw:
  one instanced quad path if possible
  CPU draw loop only as a measurement baseline, not as final design

fragment:
  q = conic quadratic
  alpha = opacity * exp(-0.5 * q)
  discard if alpha < threshold
  emit premultiplied color/alpha

blend:
  start with approximate front-to-back fixed blending
  only promote if it beats v8 and accuracy is acceptable for eval
```

Training remains a separate gate because backward needs stable depth order,
prefix transmittance, stop/final-T state, and reduced gradient accumulation.
