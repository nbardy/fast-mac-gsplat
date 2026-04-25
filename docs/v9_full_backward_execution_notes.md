# V9 Full Backward Execution Notes

Date: 2026-04-25

Variant:

```text
variants/v9_hw_tile_exact_probe
```

## Status

Full backward is wired and tested through:

```python
rasterize_projected_gaussians_full_backward(...)
ProjectedGaussianRasterizerFullBackward(...)
```

Current backend:

```text
v8_hw_eval_compute_replay
```

That means full Gaussian forward/backward is correct and trainable today, but
the gradient owner is still the proven V8 compute replay backend. The V9
hardware-raster path is not promoted to training backward until it emits
V8-equivalent real-Gaussian state.

## Why This Is The Correct First Full-Backward Base

V8 backward consumes:

```text
sorted binned_ids
tile_counts
tile_offsets
tile_stop_counts
grad_out
means2d/conics/colors/opacities
```

Then it recomputes per-pixel transmittance over the stopped prefix and replays
the alpha chain backward. It already has:

- exact gradients for means, conics, colors, and opacities;
- active-tile and direct-tile backward variants;
- overflow backward fallback;
- tile-level stop-count state to avoid storing full per-pixel history.

The V9 tile/imageblock probe can now emit a visible-fragment
`tile_stop_counts` tensor, but it does not yet produce real-Gaussian sorted bins
or V8 candidate-prefix stop counts. Using it for training gradients now would be
wrong in skipped/invisible-candidate cases.

## API

```python
from torch_gsplat_bridge_v9_hw_tile_exact import (
    make_full_backward_config,
    probe_full_backward,
    rasterize_projected_gaussians_full_backward,
)

status = probe_full_backward()
cfg = make_full_backward_config(height=H, width=W, tile_size=16)
out = rasterize_projected_gaussians_full_backward(
    means2d, conics, colors, opacities, depths, cfg
)
loss = out.square().mean()
loss.backward()
```

`probe_full_backward()` is intentionally explicit:

```text
available = true
backend = v8_hw_eval_compute_replay
exact_forward = true
exact_backward = true
hardware_forward_state = false
uses_v8_compute_replay = true
```

If `variants/v8` is not built, the wrapper falls back to the built
`variants/v8_hw_eval` namespace. That is what this local run used.

## Correctness Run

Command:

```text
python3 tests/full_backward_check.py
```

Observed against a dense PyTorch reference:

```text
image max error: 5.960464477539063e-08
means grad max error: 2.3283064365386963e-10
conics grad max error: 1.862645149230957e-09
colors grad max error: 9.313225746154785e-10
opacities grad max error: 1.862645149230957e-09
```

Also verified inherited V8-hw-eval reference tests:

```text
python3 variants/v8_hw_eval/tests/reference_check.py
```

Those pass direct, active, B=1, B=2, and saturated overlap gradient checks.

## Smoke Timing

Command:

```text
python3 benchmarks/benchmark_full_backward.py --height 512 --width 512 --gaussians 4096 --warmup 2 --iters 5 --jsonl ../../benchmarks/v9_full_backward_compute_replay_smoke.jsonl
```

Apple M4, B=1, 512x512, 4,096 projected splats:

| Metric | Median ms | Mean ms |
| --- | ---: | ---: |
| Forward | 3.092 | 2.953 |
| Backward | 2.134 | 2.834 |
| Forward + backward | 5.235 | 5.787 |

Refresh command after the Gaussian hardware-state probe landed:

```text
python3 benchmarks/benchmark_full_backward.py --height 512 --width 512 --gaussians 4096 --warmup 3 --iters 20 --jsonl ../../benchmarks/v9_full_backward_compute_replay_refresh.jsonl
```

Refresh result:

| Metric | Median ms | Mean ms |
| --- | ---: | ---: |
| Forward | 12.458 | 16.203 |
| Backward | 9.667 | 9.911 |
| Forward + backward | 22.077 | 26.114 |

The refresh confirms the backend still runs, but the timing is much slower than
the earlier smoke. Treat this as a benchmark stability warning until rerun in an
isolated process/device state with direct V8-vs-V9 common tensors.

## Next Hardware-Backward Gate

The next kernel work is not new gradient math. It is state parity:

```text
V8 sorted bins
  -> GPU draw records / tile refs
  -> exact hardware C/T forward
  -> tile_stop_counts matching V8 candidate-prefix semantics
  -> V8 backward replay
```

Kill condition:

```text
hardware tile_stop_counts != V8 tile_stop_counts on invisible/skipped candidates
```

Pass condition:

```text
same image, same tile_stop_counts, same gradients as V8 on tiny Gaussian scenes
```

Once that passes, the compute replay backend can be kept as the backward kernel
while hardware raster owns forward/state production.
