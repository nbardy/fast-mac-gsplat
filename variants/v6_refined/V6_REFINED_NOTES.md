
# v6 refined notes

This handoff rolls together:
- the richer v6 benchmark harness
- direct-kernel default for `use_active_tiles=False`
- active-policy control (`off|on|auto`)
- trace replay benchmark support
- same-process compare harness

Recommended benchmark ladder:
1. `uniform_random` for fixed-overhead regressions
2. `sparse_screen`, `clustered_hot_tiles`, `layered_depth`, `overflow_adversarial` for scheduling stress
3. `real_trace` for product decisions

Recommended default for current workloads:
- `active_policy='off'`
- `use_active_tiles=None`

Recommended compare command:
```bash
python benchmarks/benchmark_compare.py   --height 4096 --width 4096 --gaussians 65536   --batch-size 4 --case medium_sigma_3_8 --backward --shuffle-order
```
