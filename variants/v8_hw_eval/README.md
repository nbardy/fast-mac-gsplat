
# torch-metal-gsplat-v8-hw-eval

Fallback-safe hardware-eval scaffold cloned from v8. The package is
`torch_gsplat_bridge_v8_hw_eval` and its native torch namespace is
`gsplat_metal_v8_hw_eval`.

## What changes from v8

- **Exact v8 compute fallback by default**: existing v8 bin/render/backward
  kernels remain the only executed path unless a future hardware eval path is
  fully implemented.
- **Hardware-eval config surface**: `RasterConfig` exposes
  `use_hardware_eval`, `hardware_eval_policy=off|fallback|require`,
  `emit_final_T`, and `emit_stop_count`.
- **Fail-closed status/probe**: `probe_hardware_eval(config)` reports that
  imageblock/tile-shader eval is currently unsupported, with the concrete
  reason. `hardware_eval_policy="require"` raises instead of silently falling
  back.
- **Isolated namespace**: the copied native ops register as
  `gsplat_metal_v8_hw_eval` so this experiment can coexist with v8 and other
  workers' variants.

## Inherited v8 behavior

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

Probe without building or running a raster:

```bash
python3 - <<'PY'
from torch_gsplat_bridge_v8_hw_eval import RasterConfig, probe_hardware_eval
cfg = RasterConfig(16, 16, use_hardware_eval=True, emit_final_T=True, emit_stop_count=True)
print(probe_hardware_eval(cfg).as_dict())
PY
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
  raster-order update, indirect draw stream, MPS-visible output texture, or
  final_T/stop_count capture; see `../../docs/v8_hw_eval_notes.md`
