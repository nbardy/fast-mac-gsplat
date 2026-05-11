# star_uvt_v0

Gate 0 STAR-GS projected UVT tube renderer.

This variant is deliberately narrow:

- input is already-projected `ScreenTimeTube` data;
- no world/camera projection;
- no HexGaussian;
- no backward;
- no production trainer integration;
- fixed-capacity per-tile buffers instead of global scan/radix sort.

The purpose is to answer the first renderer question:

```text
Can a Metal UVT tile renderer match brute-force screen-time compositing, and
does its tile-tube pair count beat summed per-frame tile-splat pairs?
```

## Inputs

```text
ma:         [N, 3]  // u, v, t
q_uvt:      [N, 6]  // symmetric precision: uu, uv, ut, vv, vt, tt
depth0:     [N]
depth_beta: [N, 3]
opacity:    [N]
color:      [N, 3]
```

The renderer time coordinate is:

```text
t_hat = frame_index - 0.5 * (frames - 1)
```

## Default Gate 0 Constants

```text
STAR_UVT_TILE_X = 8
STAR_UVT_TILE_Y = 8
STAR_UVT_TILE_T = 2
STAR_UVT_TILE_CAPACITY = 128
```

These can be overridden with environment variables before the extension is
loaded. The Python `UVTRenderConfig` must match the runtime shader constants.

## Build

```bash
python3 setup.py build_ext --inplace
```

## Smoke

CPU-only reference and stats:

```bash
python3 tests/gate0_check.py --cpu-only
```

Metal parity, when MPS is available and the extension is built:

```bash
python3 tests/gate0_check.py
```

Projected trainer harness smoke:

```bash
python3 research_project/trainer_harness/smoke_train.py
python3 research_project/trainer_harness/train_synthetic.py --scene moving_diagonal --steps 25 --metal-check
python3 research_project/trainer_harness/world_projection_smoke.py
python3 research_project/trainer_harness/pinhole_projection_smoke.py
python3 research_project/trainer_harness/camera_spec_projection_smoke.py
python3 research_project/trainer_harness/gradient_probe.py
python3 research_project/trainer_harness/metal_autograd_smoke.py
python3 research_project/trainer_harness/simple_metal_backward_smoke.py
python3 research_project/trainer_harness/stable_metal_backward_smoke.py
python3 research_project/trainer_harness/unstable_metal_backward_smoke.py
python3 research_project/trainer_harness/tile_metal_autograd_smoke.py
python3 research_project/benchmarks/uvt_pair_benchmark.py
python3 research_project/benchmarks/backward_performance_smoke.py
python3 research_project/benchmarks/training_comparison.py
```

## Acceptance

Gate 0 is accepted only when:

- CPU brute-force parity passes on tiny deterministic scenes;
- stable-only scenes match to explicit max/mean RGB tolerances;
- unstable scenes match after deterministic per-sample fallback is enabled;
- UVT pair count and per-frame pair count are reported;
- pair ratio and effective pair ratio are reported;
- stable/unstable tile fractions are reported;
- overflow count is reported;
- forward wall-clock time is reported for Metal smoke runs;
- no camera projection or HexGaussian projection is included.

The research-project folder records the next phases:

```text
research_project/README.md
research_project/PROGRESS.md
research_project/phases/
research_project/trainer_harness/
```
