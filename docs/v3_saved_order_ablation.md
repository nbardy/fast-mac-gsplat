# v3 Saved-Order Ablation

Date: 2026-04-20

## Goal

Measure two small v3 backward optimizations independently:

1. avoid cloning the full 4K RGB gradient image when no overflow tiles exist,
2. save the forward tile-local sorted ID order and reuse it in fast backward.

The test case is synthetic projected 2D splats on MPS:

```bash
uv run python benchmarks/compare_v2_v3.py \
  --height 4096 --width 4096 --gaussians 65536 \
  --warmup 2 --iters 5 --backward
```

Forward-only checks used the same command without `--backward`.

## Tile occupancy

The ablation cases had no overflow tiles, so the clone optimization tested the
common fast path.

| Case | Total tile pairs | Max tile count | Mean tile count | Nonzero tiles | Overflow tiles |
| --- | ---: | ---: | ---: | ---: | ---: |
| sigma 1-5 px | `273,547` | `15` | `4.174` | `64,485` | `0` |
| sigma 3-8 px | `552,285` | `25` | `8.427` | `65,520` | `0` |

## Forward+Backward Ablation

| Variant | Sparse sigma 1-5 px | Delta | Medium sigma 3-8 px | Delta |
| --- | ---: | ---: | ---: | ---: |
| v3 baseline | `52.460 ms` | - | `65.687 ms` | - |
| no unconditional grad clone | `52.904 ms` | `+0.444 ms` / `+0.8%` | `63.878 ms` | `-1.809 ms` / `-2.8%` |
| clone fix + saved sorted IDs | `47.872 ms` | `-4.588 ms` / `-8.7%` | `60.738 ms` | `-4.949 ms` / `-7.5%` |

The clone-only result is small and noisy. The saved sorted ID change is the real
measured win.

## Estimated Backward-Only Cost

Subtracting separate forward-only timings gives a rough backward-only estimate.
These estimates are useful for direction but inherit timing noise from separate
runs.

| Variant | Sparse backward estimate | Delta | Medium backward estimate | Delta |
| --- | ---: | ---: | ---: | ---: |
| v3 baseline | `40.980 ms` | - | `53.760 ms` | - |
| no unconditional grad clone | `41.424 ms` | `+0.444 ms` / `+1.1%` | `51.951 ms` | `-1.809 ms` / `-3.4%` |
| clone fix + saved sorted IDs | `35.462 ms` | `-5.518 ms` / `-13.5%` | `47.036 ms` | `-6.724 ms` / `-12.5%` |

## Forward Cost

Saving sorted IDs adds a global writeback in forward. In this run, forward-only
time rose:

| Case | v3 baseline forward | saved-ID forward | Delta |
| --- | ---: | ---: | ---: |
| sigma 1-5 px | `11.480 ms` | `12.410 ms` | `+0.930 ms` / `+8.1%` |
| sigma 3-8 px | `11.927 ms` | `13.702 ms` | `+1.775 ms` / `+14.9%` |

Even with that forward writeback cost, the end-to-end forward+backward step was
faster because backward no longer repeats the local bitonic sort.

## Correctness Checks

After both changes:

```bash
cd variants/v3
uv run python tests/reference_check.py
```

Result:

```text
image max error: 5.960464477539063e-08
means grad max error: 2.4010660126805305e-10
conics grad max error: 9.313225746154785e-10
colors grad max error: 9.313225746154785e-10
opacities grad max error: 1.862645149230957e-09
```

Small v2/v3 MPS image parity at 128x128 / 512 splats remained exact:

```text
v2_v3 image max 0.0 mean 0.0
```

## Conclusion

Keep both changes. The no-clone change is harmless and helps the medium case a
little, but the main speedup is saved sorted tile IDs. The likely reason is that
v3 fast backward was paying the full local bitonic-sort cost again for every
tile even though forward already had the correct sorted order.
