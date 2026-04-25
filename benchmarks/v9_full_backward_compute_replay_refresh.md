# V9 Full Backward Compute-Replay Refresh

Date: 2026-04-25

Variant:

```text
variants/v9_hw_tile_exact_probe
```

Command:

```bash
python3 benchmarks/benchmark_full_backward.py --height 512 --width 512 --gaussians 4096 --warmup 3 --iters 20 --jsonl ../../benchmarks/v9_full_backward_compute_replay_refresh.jsonl
```

Backend:

```text
v8_hw_eval_compute_replay
```

Apple M4, B=1, 512x512, 4,096 projected splats:

| Metric | Median ms | Mean ms |
| --- | ---: | ---: |
| Forward | 12.458 | 16.203 |
| Backward | 9.667 | 9.911 |
| Forward + backward | 22.077 | 26.114 |

Interpretation:

- This confirms the V9 full-backward wrapper still runs through the V8-hw-eval
  compute replay backend after the Gaussian hardware-state probe changes.
- It is materially slower than the earlier smoke in
  `v9_full_backward_compute_replay_smoke.md` on the same nominal case
  (`5.235 ms` total median). Treat this as current-run MPS/system-load noise or
  a benchmark stability warning until isolated with a process/device reset and
  direct V8-vs-V9 common-tensor comparison.
- This is not evidence that hardware raster backward got faster. The hardware
  raster path still owns only diagnostic forward/state probes.
