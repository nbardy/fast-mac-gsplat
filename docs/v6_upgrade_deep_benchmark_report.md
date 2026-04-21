# v6 Upgrade Deep Benchmark Report

Date: 2026-04-21

## Summary

The v6-upgrade handoff is competitive in pockets, but it does not replace the
locally evolved `variants/v6` branch as the default renderer.

Deep matrix result:

- cells: `960`
- status: `960 ok`, `0 timeout`, `0 error`
- settings: `warmup=1`, `iters=2`, `timeout-sec=120`
- resolutions: `512x512`, `1024x512`, `1920x1080`, `4096x4096`
- splats: `512`, `2048`, `65536`
- batch sizes: `1`, `4`
- distributions:
  - `microbench_uniform_random`
  - `sparse_screen`
  - `clustered_hot_tiles`
  - `layered_depth`
  - `overflow_adversarial`
- renderers:
  - `v6_direct`
  - `v6_auto`
  - `v6_upgrade_direct`
  - `v6_upgrade_auto`
- modes:
  - `forward`
  - `forward_backward`

Raw outputs:

```text
benchmarks/full_rasterizer_benchmark_v6_upgrade_deep.md
benchmarks/full_rasterizer_benchmark_v6_upgrade_deep.jsonl
```

## Winner Counts

There are 240 unique workload groups. Each group compares four renderers.

| Renderer | Wins | Share |
|---|---:|---:|
| `v6_direct` | 117 | 48.8% |
| `v6_upgrade_direct` | 67 | 27.9% |
| `v6_auto` | 29 | 12.1% |
| `v6_upgrade_auto` | 27 | 11.2% |

Grouped by family:

| Family | Wins | Share |
|---|---:|---:|
| local v6 | 146 | 60.8% |
| v6-upgrade | 94 | 39.2% |

By mode:

| Mode | v6 direct | v6 auto | upgrade direct | upgrade auto |
|---|---:|---:|---:|---:|
| forward | 67 | 12 | 30 | 11 |
| forward+backward | 50 | 17 | 37 | 16 |

The upgrade is more interesting for training than pure eval: `v6_upgrade_direct`
won 37 forward+backward groups versus 30 forward-only groups.

## Distribution Behavior

Winner counts by projected-splat distribution:

| Distribution | v6 direct | v6 auto | upgrade direct | upgrade auto |
|---|---:|---:|---:|---:|
| `microbench_uniform_random` | 19 | 8 | 10 | 11 |
| `sparse_screen` | 29 | 5 | 12 | 2 |
| `clustered_hot_tiles` | 25 | 5 | 15 | 3 |
| `layered_depth` | 22 | 7 | 16 | 3 |
| `overflow_adversarial` | 22 | 4 | 14 | 8 |

Interpretation:

- `v6_direct` is still the best default across scene types.
- `v6_upgrade_direct` is most competitive in backward-heavy and 64k cases.
- `v6_upgrade_auto` has real wins, but they are not stable enough to make it
  the default policy.
- `sparse_screen` was not a reliable active-scheduling win. The current active
  paths pay enough fixed overhead that sparse-looking synthetic scenes can still
  lose.

## Direct-vs-Direct Delta

This compares `v6_upgrade_direct` to `v6_direct` for every workload.
Negative means the upgrade was faster.

| Slice | Mean Delta | Median Delta | Upgrade Wins |
|---|---:|---:|---:|
| all workloads | `+3.8%` | `+1.3%` | 101 / 240 |
| forward | `+5.6%` | `+4.2%` | 43 / 120 |
| forward+backward | `+1.9%` | `+0.3%` | 58 / 120 |
| 512 splats | `+5.6%` | `+5.6%` | 26 / 80 |
| 2048 splats | `+5.3%` | `+1.5%` | 33 / 80 |
| 65536 splats | `+0.4%` | `-0.5%` | 42 / 80 |
| B=1 | `+5.0%` | `+2.6%` | 51 / 120 |
| B=4 | `+2.5%` | `+0.7%` | 50 / 120 |

The important nuance: the upgrade is not broadly faster, but it gets more
competitive as the workload gets larger. At 64k splats the median direct-vs-
direct result was slightly in favor of the upgrade.

## 4K / 64K Results

These are the headline stress cases. Lower is better.

### B=1

| Distribution | Mode | v6 direct | v6 auto | upgrade direct | upgrade auto | Winner |
|---|---|---:|---:|---:|---:|---|
| `microbench_uniform_random` | forward | 11.752 | 12.915 | 12.065 | 12.462 | v6 direct |
| `microbench_uniform_random` | forward+backward | 69.749 | 62.798 | 62.657 | 68.348 | upgrade direct |
| `sparse_screen` | forward | 10.650 | 12.605 | 11.928 | 27.063 | v6 direct |
| `sparse_screen` | forward+backward | 52.268 | 59.363 | 63.181 | 69.354 | v6 direct |
| `clustered_hot_tiles` | forward | 131.833 | 144.147 | 127.332 | 146.567 | upgrade direct |
| `clustered_hot_tiles` | forward+backward | 184.195 | 192.942 | 176.944 | 195.778 | upgrade direct |
| `layered_depth` | forward | 14.572 | 14.627 | 16.256 | 31.177 | v6 direct |
| `layered_depth` | forward+backward | 75.928 | 77.624 | 74.644 | 88.455 | upgrade direct |
| `overflow_adversarial` | forward | 585.299 | 601.332 | 1289.412 | 628.754 | v6 direct |
| `overflow_adversarial` | forward+backward | 1468.462 | 1089.448 | 1154.462 | 1514.028 | v6 auto |

### B=4

| Distribution | Mode | v6 direct | v6 auto | upgrade direct | upgrade auto | Winner |
|---|---|---:|---:|---:|---:|---|
| `microbench_uniform_random` | forward | 33.943 | 35.467 | 34.292 | 34.555 | v6 direct |
| `microbench_uniform_random` | forward+backward | 231.469 | 229.856 | 230.142 | 229.991 | v6 auto |
| `sparse_screen` | forward | 25.037 | 28.029 | 27.424 | 86.221 | v6 direct |
| `sparse_screen` | forward+backward | 197.872 | 191.498 | 188.677 | 252.630 | upgrade direct |
| `clustered_hot_tiles` | forward | 1381.861 | 1006.525 | 1075.467 | 819.863 | upgrade auto |
| `clustered_hot_tiles` | forward+backward | 684.783 | 754.457 | 925.505 | 1070.714 | v6 direct |
| `layered_depth` | forward | 39.768 | 40.326 | 39.836 | 102.303 | v6 direct |
| `layered_depth` | forward+backward | 271.098 | 269.955 | 270.532 | 331.298 | v6 auto |
| `overflow_adversarial` | forward | 5253.447 | 4358.096 | 4519.327 | 4674.138 | v6 auto |
| `overflow_adversarial` | forward+backward | 8928.235 | 8367.094 | 7555.238 | 8054.878 | upgrade direct |

## Field Interpretation

The v6-upgrade should stay in the repo. It is not dead code: it wins 39% of the
full matrix and is especially close on 64k forward+backward workloads. But the
local v6 branch is still the safer default because it wins more total cases,
wins more 4K cases, and avoids large regressions in common sparse/direct cases.

The active-policy story is still not solved. Auto modes can win in specific
clustered or overflow-heavy cells, but they also create large regressions. A
future auto policy should be based on measured tile statistics and should be
validated against this deep matrix, not just the small smoke.

## Caveats

- This is a broad local benchmark, not a final perf paper number.
- `warmup=1` and `iters=2` keep runtime practical but leave noise.
- Each cell runs in its own subprocess, so this is good for isolation and bad
  for minimizing first-use effects.
- Forward-only and forward+backward paths are not the same code path because
  the renderers have eval/train splits. A forward-only number can be slower than
  the forward component of a training run in pathological cases.

## Recommendation

Keep `variants/v6` as the default batch-training renderer. Keep
`variants/v6_upgrade` as a preserved candidate for larger backward-heavy scenes.
The next useful experiment is not replacing v6 with v6-upgrade wholesale; it is
isolating the 64k cases where upgrade direct wins and porting only the specific
mechanism if it survives a targeted ablation.
