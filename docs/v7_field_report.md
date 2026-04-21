# v7 Hardware Field Report

Date: 2026-04-21

## Summary

v7 is a hardware-forward source handoff: Torch custom op at the Python boundary,
Metal render pipeline for forward, and Metal compute replay for backward. It is
now imported as `variants/v7` and preserved as
`source_artifacts/torch_metal_gsplat_v7_hardware.tar.gz`.

It builds and passes the small dense-reference check after local fixes, but it
is not yet a replacement for v6 direct. The current implementation copies MPS
tensors to CPU/shared Metal buffers and reads the render texture back to Torch,
so the render-pipeline win is fighting heavy bridge overhead.

## Fixes Applied

The handoff did not run cleanly as delivered. The local fixes were:

- add direct-execution import path fixes to `tests/reference_check.py` and
  `benchmarks/benchmark_v7.py`
- add the missing `if __name__ == "__main__"` entrypoint to the reference check
- remove duplicate Metal `[[position]]` declaration in the fragment shader
- draw the already depth-sorted buffer far-to-near so standard source-over
  hardware blending matches the compute renderer's front-to-back alpha equation
- expand quad bounds to `[0, W] x [0, H]` pixel-center support instead of
  shrinking by `0.5` and clamping to `W-1/H-1`
- fix backward formulas:
  - `dL/dcolor = grad * weight`, not `grad * weight / alpha`
  - `dL/dmean` sign was reversed
- improve the standalone benchmark with warmup, fixed inputs, median, and JSON
  output

## Correctness

Command:

```bash
cd variants/v7
python setup.py build_ext --inplace
python tests/reference_check.py
```

Result:

```text
image max error: 5.960464477539063e-08
means grad max error: 2.473825588822365e-10
conics grad max error: 9.313225746154785e-10
colors grad max error: 9.313225746154785e-10
opacities grad max error: 1.862645149230957e-09
```

The check is intentionally small (`B=1`, `G=4`, `16x16`). It proves the corrected
math path is plausible, not that large scenes are production-ready.

## Smoke Performance

Tiny smoke, `128x128`, `G=64`, `B=1`, `warmup=1`, `iters=2`, subprocess matrix:

| Distribution | Mode | Torch | v6 direct | v7 hardware |
|---|---|---:|---:|---:|
| uniform random | forward | 207.970 ms | 10.436 ms | 25.654 ms |
| uniform random | forward+backward | 712.614 ms | 10.941 ms | 13.248 ms |
| clustered hot tiles | forward | 271.543 ms | 71.682 ms | 36.989 ms |
| clustered hot tiles | forward+backward | 358.383 ms | 13.973 ms | 15.422 ms |

Interpretation:

- v7 can beat v6 direct in clustered tiny forward-only cases, likely because
  hardware blending avoids some compute-path tile overhead.
- v7 still loses to v6 direct in forward+backward, but in this rerun the gap was
  much smaller than the first smoke.
- These numbers include subprocess and first-use effects enough that they should
  be treated as a smoke, not a headline benchmark.

## 4K / 64K Large-Splat Check

Command:

```bash
python benchmarks/benchmark_full_matrix.py \
  --resolutions 4096x4096 \
  --splats 65536 \
  --batch-sizes 1 \
  --distributions microbench_uniform_random,clustered_hot_tiles \
  --renderers v6_direct,v7_hardware \
  --modes forward,forward_backward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 240 \
  --output-md benchmarks/full_rasterizer_benchmark_v7_4k64k.md
```

Result: all 8 cells completed with `status=ok`.

| Distribution | Mode | v6 direct | v7 hardware | Result |
|---|---|---:|---:|---|
| uniform random | forward | 31.301 ms | 149.561 ms | v7 slower by 377.8% |
| uniform random | forward+backward | 68.039 ms | 21574.243 ms | v7 not viable |
| clustered hot tiles | forward | 253.653 ms | 129.820 ms | v7 faster by 48.8% |
| clustered hot tiles | forward+backward | 178.227 ms | 20054.494 ms | v7 not viable |

Interpretation:

- The hardware forward path can win in a clustered forward-only case.
- It loses badly on uniform forward, where the compute path is much better.
- The current v7 backward is not viable for training at 4K/64K. The measured
  backward component was about `19.9-21.4 s`.
- This reinforces the current model: v7 is a useful render-pipeline experiment,
  but it is not the training renderer until CPU/shared-buffer round trips and
  backward architecture are replaced.

## Benchmark Integration

Added:

- `benchmarks/benchmark_full_matrix.py`
- optional v7 row in `benchmarks/compare_v2_v3.py` via `--include-v7`

Example full run:

```bash
python benchmarks/benchmark_full_matrix.py \
  --resolutions 512x512,1024x512,1920x1080,4096x4096 \
  --splats 512,2048,65536 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers torch_direct,v2_fastpath,v3_candidate,v5_batched,v6_direct,v6_auto,v7_hardware \
  --modes forward,forward_backward \
  --output-md benchmarks/full_rasterizer_benchmark.md
```

The matrix runner executes each cell in a subprocess, records skips/errors/timeouts,
and writes Markdown plus optional JSONL. This is deliberate because experimental
variants like v7 can fail or take too long without corrupting the rest of the
report.

## Current Model

v7 is valuable because it proves the render-pipeline path can be wired through a
Torch custom op and made numerically consistent on a small case. It is not yet
the fast training path. The next serious v7 work is not another shader tweak; it
is removing CPU round trips and then re-evaluating whether hardware blending wins
at high resolution.
