
# torch-metal-gsplat-v8-hw-train

Fallback-safe hardware-assisted training scaffold for the v8 Apple Silicon/MPS
projected Gaussian rasterizer.

This variant intentionally keeps the v8 compute renderer/backward as the
default and exact fallback. The package name is
`torch_gsplat_bridge_v8_hw_train`; native custom ops are registered under
`gsplat_metal_v8_hw_train`.

## Hardware Training Status

Plan 2 controls are exposed on `RasterConfig`:

- `use_hardware_train`
- `hardware_train_policy="fallback"|"strict"`
- `capture_final_T`
- `capture_stop_count`
- `capture_pixel_stop`
- `backward_state_mode="compute"|"tile_stop"|"final_T"|"pixel_stop"`

The real hardware render pass, GPU-native Torch/MPS output interop, and exact
training state capture are not implemented in this scaffold. When
`use_hardware_train=True`, `hardware_train_policy="fallback"` emits a clear
runtime warning and uses the cloned v8 compute path. With
`hardware_train_policy="strict"`, the same unsupported condition raises instead.

Use `probe_hardware_train(config)` and `estimate_hardware_train_state(config)`
to inspect the decision and state-memory plan without touching tensor data or
adding CPU readback to a timed path. The state planner reports exact byte counts
for `tile_stop` int32, `final_T` fp32, and `pixel_stop` int32 state. It does not
allocate per-pixel state buffers.

State-mode validation is intentionally strict:

- `tile_stop` requires `capture_stop_count=True`
- `final_T` requires `capture_stop_count=True` and `capture_final_T=True`
- `pixel_stop` requires `capture_stop_count=True` and `capture_pixel_stop=True`
- `final_T` and `pixel_stop` captures are mutually exclusive in this scaffold

Probe without building or running a raster:

```bash
python3 - <<'PY'
from torch_gsplat_bridge_v8_hw_train import RasterConfig, estimate_hardware_train_state, probe_hardware_train
cfg = RasterConfig(16, 16, use_hardware_train=True, backward_state_mode="pixel_stop", capture_pixel_stop=True)
print(probe_hardware_train(cfg))
print(estimate_hardware_train_state(cfg, batch_size=1))
PY
```

## Inherited v8 Behavior

- **Batch-first renderer**: keeps native `[B,G,...] -> [B,H,W,3]` API.
- **Host-side metadata parsing**: kernels still receive GPU metadata, but the
  Objective-C++ bridge parses CPU metadata so tiny MPS metadata tensors are not
  copied back to CPU inside every op.
- **Optional active-tile scheduling**: compacts fast tiles across the batch and dispatches only active fast tiles.
- **Active policy gate**: `active_policy=off|on|auto`, with `auto` currently conservative because active scheduling still has measurable fixed overhead.
- **Count-sorted active tiles**: active fast tiles are optionally sorted by tile occupancy before render/backward.
- **No unconditional grad clone**: backward consumes the original contiguous grad image.
- **Adaptive stop counts**: training can save per-tile stop counts `always`, `never`, or `adaptive` (dense tiles only).
- **Optional pair-budget chunks**: `max_pairs_per_launch > 0` can split training batches by measured tile-pair load.
- **Overflow fallback** remains available for pathological tiles.

## Build

```bash
python setup.py build_ext --inplace
```

## Quick check

```bash
python tests/reference_check.py
```

## Benchmarks

```bash
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --case medium_sigma_3_8 --backward --profile
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case medium_sigma_3_8 --backward --profile
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case microbench_uniform_random --backward --profile --active-policy auto
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case microbench_uniform_random --backward --profile --max-pairs-per-launch 1500000
python benchmarks/benchmark_matrix.py --height 4096 --width 4096 --gaussians 65536 --batch-sizes 1,2,4,8 --warmup 1 --iters 3 --backward --shuffle-order
```

Benchmark cases:

- `microbench_uniform_random`: saturated uniform projected splats; useful for fixed-overhead regressions, not a real scene proxy
- `sparse_screen`: a few occupied screen regions with many empty tiles
- `clustered_hot_tiles`: hot clustered regions that exercise overflow and active scheduling
- `layered_depth`: overlapping regions with several depth bands
- `overflow_adversarial`: concentrated large splats for overflow stress
- `real_trace`: replays dumped `means2d/conics/colors/opacities/depths` tensors from `--trace-file`

See `../../docs/v8_field_report.md` for the first Mac build, correctness, and
benchmark report. In the current 4K/64K uniform forward+backward gate, v8 direct
beats v6 direct, v6-upgrade direct, and v6-refined direct. Active scheduling and
overflow fallback remain experimental policy paths; keep direct mode as the base
until GPU overflow compaction lands.

## Notes

- input API is projected 2D splats, not full 3D camera projection
- depth gradients are zero; sort order is piecewise constant
- overflow tiles fall back to a slower exact path
- this scaffold does not create a Metal render pass, imageblock layout,
  MPS-visible output texture, exact final_T/pixel_stop buffers, or a
  hardware-assisted backward; see `../../docs/v8_hw_train_notes.md`
