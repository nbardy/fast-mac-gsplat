# Gate 0 Acceptance

Gate 0 proves only projected screen-time tube rendering.

Accepted scope:

- `ScreenTimeTube` arrays are the only renderer input.
- Metal kernels clear fixed tile buffers, bin tubes, locally sort tiles, render,
  and report tile stats.
- CPU brute force is the parity reference.
- Per-frame pair count is computed by slicing the projected UVT tubes, not by
  calling a 3DGS renderer.
- Forward wall-clock time is synchronized and reported for Metal smoke runs.

Out of scope:

- camera/world projection;
- WorldTube parameter optimization;
- HexGaussian precision residuals;
- backward;
- production trainer integration.

Tiny scenes:

```text
single_static
moving_diagonal
two_non_crossing
crossing_depth
fast_screen_motion
wide_temporal_support
```

Initial command:

```bash
python3 setup.py build_ext --inplace
python3 tests/gate0_check.py
```

Current source also includes a separate projected trainer harness under
`research_project/trainer_harness/`. That harness is not part of Gate 0
acceptance.
