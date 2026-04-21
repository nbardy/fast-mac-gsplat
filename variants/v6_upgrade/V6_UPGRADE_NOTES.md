# v6 Upgrade Notes

This source handoff folds in two classes of changes:

1. **Renderer path fixes / policy**
   - Exposes and uses the existing direct tile kernels when active scheduling is off.
   - Makes direct mode the default (`active_policy='off'`).
   - Adds `active_policy='auto'|'on'|'off'` and preserves `use_active_tiles` as a legacy explicit override.
   - Keeps overflow fallback and saved-sort backward reuse.

2. **Benchmark harness expansion**
   - Adds more representative synthetic cases: `sparse_screen`, `clustered_hot_tiles`, `layered_depth`, `overflow_adversarial`, `temporal_adjacent`.
   - Adds `real_trace` replay from dumped projected splat tensors.
   - Adds matrix sweeps over active policies and overrides.
   - Adds a same-process compare harness across v6 modes.

## Suggested engineer commands

```bash
cd torch_metal_gsplat_v6_upgrade
python setup.py build_ext --inplace
python tests/reference_check.py
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case uniform_random --backward --profile
python benchmarks/benchmark_mps.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case sparse_screen --backward --profile --active-policy auto
python benchmarks/benchmark_compare.py --height 4096 --width 4096 --gaussians 65536 --batch-size 4 --case uniform_random --backward --shuffle-order
python benchmarks/benchmark_matrix.py --height 4096 --width 4096 --gaussians 65536 --batch-sizes 1,2,4 --cases uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial,temporal_adjacent --active-policies off,auto,on --warmup 1 --iters 3 --backward --shuffle-order
```

## Expected trace file keys for `case=real_trace`
- `means2d`: `[B,G,2]` or `[G,2]`
- `conics`: `[B,G,3]` or `[G,3]`
- `colors`: `[B,G,3]` or `[G,3]`
- `opacities`: `[B,G]` or `[G]`
- `depths`: `[B,G]` or `[G]`
