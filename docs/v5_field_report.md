# v5 Field Report

## Context

The v5 source handoff arrived as `torch_metal_gsplat_v5.tar.gz`. The stated
goal was to move beyond the single-image v2/v3 path and test a dual-mode,
batched Torch+Metal rasterizer:

- `[G,...]` or `[B,G,...]` projected splat inputs.
- `[H,W,3]` or `[B,H,W,3]` output.
- eval forward skips sorted-ID writeback.
- training forward keeps the saved-sort backward win.
- `batch_strategy=auto|flatten|serial`.

I extracted the bundle into `variants/v5/`, preserved the original archive in
`source_artifacts/`, built it on the local Apple Silicon machine, added it to
the existing v2/v3 benchmark harness, and ran correctness plus throughput
checks.

## Integration Work

Files added or changed:

- `variants/v5/`: extracted v5 source handoff.
- `source_artifacts/torch_metal_gsplat_v5.tar.gz`: original handoff archive.
- `benchmarks/compare_v2_v3.py`: now imports v5, prints median timing, and can
  report v5 tile/profile stats next to v3.
- `variants/v5/tests/reference_check.py`: fixed local import path so it runs
  directly from the v5 directory.
- `variants/v5/benchmarks/benchmark_mps.py`: fixed local import path and changed
  synthetic conics from `1 / sigma` to `1 / sigma^2`.

The conic fix matters. The renderer interprets `conics` as inverse covariance
coefficients. The v2/v3 comparison benchmark already generated `1 / sigma^2`,
but the v5 standalone benchmark generated `1 / sigma`. Without that fix, the
case names looked comparable while the actual splat footprint was not.

## Build And Correctness

Build command:

```bash
cd variants/v5
python setup.py build_ext --inplace
```

Result:

- Build succeeded.
- Only warning seen before cleanup: unused local variable `meta` in the bridge.
- The extension imports and registers the v5 Torch op.

Reference check:

```bash
python tests/reference_check.py
```

Result:

- B=1 image max error: `5.96e-08`
- B=1 means grad max error: `2.40e-10`
- B=1 conics grad max error: `9.31e-10`
- B=1 colors grad max error: `9.31e-10`
- B=1 opacities grad max error: `1.86e-09`
- B=2 image max error: `5.96e-08`
- B=2 means grad max error: `3.71e-10`
- B=2 conics grad max error: `1.86e-09`
- B=2 colors grad max error: `9.31e-10`
- B=2 opacities grad max error: `1.86e-09`

This validates both the scalar and batched API shape on small scenes against
the CPU reference.

## Benchmark Method Notes

All numbers below are local Apple Silicon MPS timings with explicit
`torch.mps.synchronize()` timing boundaries.

For v2/v3/v5 comparisons, the shared harness generates projected 2D splats with
the same seed and case definitions:

- `sparse_sigma_1_5`: random isotropic sigma in `[1, 5]` px.
- `medium_sigma_3_8`: random isotropic sigma in `[3, 8]` px.

Mean and median are both recorded because the MPS stack occasionally produces
large outliers even after warmup. Median is often the better "steady" number,
but mean reflects real wall-clock noise.

## B=1 Comparison Against v2/v3

Command shape:

```bash
python benchmarks/compare_v2_v3.py --height 4096 --width 4096 --gaussians 65536 --warmup 2 --iters 5
python benchmarks/compare_v2_v3.py --height 4096 --width 4096 --gaussians 65536 --warmup 2 --iters 5 --backward
```

Forward-only, 4096x4096, 65,536 splats:

| Case | v2 mean | v3 mean | v5 mean | Winner |
| --- | ---: | ---: | ---: | --- |
| sigma 1-5 | `15.643 ms` | `13.675 ms` | `20.110 ms` | v3 |
| sigma 3-8 | `23.758 ms` | `13.350 ms` | `11.322 ms` | v5 |

Forward+backward, 4096x4096, 65,536 splats:

| Case | v2 mean | v3 mean | v5 mean | Winner |
| --- | ---: | ---: | ---: | --- |
| sigma 1-5 | `72.585 ms` | `48.684 ms` | `61.780 ms` | v3 |
| sigma 3-8 | `135.596 ms` | `60.360 ms` | `73.231 ms` | v3 |

Interpretation:

- v5 is not a universal replacement for v3 at B=1.
- v5 does show the expected eval-forward advantage in the 4K medium case:
  `11.322 ms` vs v3 `13.350 ms`, about `18%` faster.
- v3 remains the better B=1 training/backward path in these tests:
  about `27%` faster than v5 in sparse forward+backward and about `21%` faster
  in medium forward+backward.

## Native v5 Batch Results

v5's new feature is true batch support, so I also timed its standalone benchmark.

4096x4096, 65,536 splats, medium sigma 3-8:

| Mode | Batch | Strategy | Mean total | Median total | Forward | Backward |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| forward | 1 | flatten | `16.691 ms` | `16.903 ms` | `16.691 ms` | n/a |
| forward | 4 | flatten | `32.576 ms` | `32.596 ms` | `32.576 ms` | n/a |
| fwd+bwd | 1 | auto | `64.997 ms` | `63.971 ms` | `11.996 ms` | `53.001 ms` |
| fwd+bwd | 4 | auto | `297.749 ms` | `245.932 ms` | `37.496 ms` | `260.254 ms` |
| fwd+bwd | 4 | flatten | `274.373 ms` | `246.163 ms` | `37.025 ms` | `237.348 ms` |
| fwd+bwd | 4 | serial | `277.897 ms` | `277.859 ms` | `66.673 ms` | `211.224 ms` |

The cleanest v5 batch result is forward throughput:

- B=1 forward mean: `16.691 ms/frame`.
- B=4 forward mean: `32.576 ms total`, or `8.144 ms/frame`.
- That is about `2.05x` better per-frame throughput at B=4.

Backward batching helped less and was noisier. At B=4, flatten/auto improve
forward launch efficiency, while backward still pays heavy memory traffic and
gradient accumulation cost.

1024x1024, 65,536 splats, medium sigma 3-8:

| Mode | Batch | Strategy | Mean total | Median total | Per-frame mean |
| --- | ---: | --- | ---: | ---: | ---: |
| forward | 1 | flatten | `13.461 ms` | `10.804 ms` | `13.461 ms` |
| forward | 4 | flatten | `21.740 ms` | `21.754 ms` | `5.435 ms` |
| fwd+bwd | 1 | flatten | `28.111 ms` | `27.884 ms` | `28.111 ms` |
| fwd+bwd | 4 | flatten | `99.180 ms` | `98.751 ms` | `24.795 ms` |

At 1024, native batch gives roughly `2.48x` better forward per-frame throughput
by mean. Forward+backward improves only about `13%` per frame by mean.

## Tile/Chunk Ablation

Command shape:

```bash
GSP_TILE_SIZE=8  GSP_CHUNK=32  GSP_FAST_CAP=1024 python benchmarks/benchmark_mps.py ...
GSP_TILE_SIZE=16 GSP_CHUNK=64  GSP_FAST_CAP=2048 python benchmarks/benchmark_mps.py ...
GSP_TILE_SIZE=32 GSP_CHUNK=128 GSP_FAST_CAP=2048 python benchmarks/benchmark_mps.py ...
```

1024x1024, 65,536 splats, B=1, medium sigma 3-8, forward+backward:

| Tile | Chunk | Cap | Mean total | Forward | Backward |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 32 | 1024 | `31.853 ms` | `10.324 ms` | `21.529 ms` |
| 16 | 64 | 2048 | `28.259 ms` | `8.579 ms` | `19.680 ms` |
| 32 | 128 | 2048 | `52.407 ms` | `13.051 ms` | `39.356 ms` |

Tile 16 remains the best tested default. It was about `13%` faster than tile 8
and about `85%` faster than tile 32 in this probe. I do not see evidence yet
that v5 should move away from 16x16 tiles for the default path.

## Current Model

Observed facts:

- v5 builds and passes B=1/B=2 forward/backward reference checks.
- v5's B=1 forward can beat v3 in at least one 4K case.
- v5's B=1 backward does not beat v3 in the measured cases.
- v5's native batch improves forward per-frame throughput clearly.
- v5's native batch improves backward per-frame throughput only modestly and
  with more variance.

Inference:

- v5 is a successful architectural expansion, not a clean speed replacement.
- Its strongest near-term value is train-loop shape: batched API, eval/train
  split, strategy knobs, and built-in profiling.
- For B=1 high-resolution training, v3 should stay the default until v5
  backward is improved.
- For eval or multi-frame forward rendering, v5 is promising now.

Speculation to test:

- v5 backward may be losing to v3 through extra batch-general indexing,
  stop-count bookkeeping, or less favorable compiler specialization.
- The backward wall is still mostly bandwidth/atomic pressure, not local sort.
- Batching exposes forward launch amortization more than backward because
  backward writes gradients into global buffers and replays the alpha chain.

## Recommended Next Tests

1. Compare v3 vs v5 on the same 4K scene with gradients disabled and enabled,
   isolating only the sorted-ID writeback cost.
2. Add a v5 B=1 "single specialization" compile path if the batch-general kernel
   is causing B=1 backward overhead.
3. Profile backward atomics and memory bandwidth with Xcode GPU tools before
   changing the algorithm.
4. Re-run B=4 forward+backward with no other GPU workloads; the measured B=4
   backward numbers had enough variance that mean and median diverged.
5. Test real projected distributions from Dynaworld, not only uniform synthetic
   splats, because tile occupancy and early alpha stop counts will decide which
   path wins in training.

## Bottom Line

v5 is real: it compiles, passes small CPU-reference checks, and adds the missing
batched API shape. The measured speed story is narrower than the handoff hope:
v5 is not yet the best B=1 training renderer, but it is the right branch for
batched training-loop integration and it already shows strong forward
per-frame throughput gains at B=4.
