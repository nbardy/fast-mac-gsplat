# Phase 4: Video And MVP Integration

## Goal

Connect the UVT representation to real video fitting only after the projected
trainer and projection gates are proven.

## Boundaries

- Reuse Dynaworld video loaders for frame tensors where practical.
- Keep this lane separate from the main FasterGS MVP until quality and held-out
  behavior justify integration.
- Compare against the current dynamic FasterGS path, not against a toy-only
  target.

## Gate

The first useful integration result is a side-by-side report with:

- same source video;
- same frame count and resolution;
- same train/held-out camera split when cameras exist;
- renderer timing;
- reconstruction loss;
- visual contact sheet or equivalent proof.

## Current Slice

Phase 4a is a renderer-side pair and timing report:

```bash
python3 research_project/benchmarks/uvt_pair_benchmark.py
```

It compares UVT tile-tube pairs against the sliced per-frame tile-splat baseline
on the same deterministic tiny scenes. This is useful for renderer iteration,
but it does not satisfy the full video/training integration gate above.

Phase 4b is a tiny training comparison:

```bash
python3 research_project/benchmarks/training_comparison.py
```

It trains UVT tubes and a simple independent per-frame Gaussian baseline on the
same deterministic target. This proves the comparison harness, but it is still
not a production FasterGS benchmark or held-out video-quality result.

Phase 4c is a real-video fixture comparison:

```bash
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 8 --per-frame-splats 8 --target-size 32 --max-frames 4 \
  --steps 8 --lr 0.04 --device cpu --seed 5 \
  --out-json research_project/benchmarks/results/video_fit_comparison_fixture.json \
  --contact-sheet research_project/benchmarks/results/video_fit_comparison_fixture.png
```

It reuses the Dynaworld video loader, fits both local research representations
to the same fixture target, and writes a contact sheet. It still does not
satisfy the production integration gate because the baseline is not current
FasterGS and the fixture has no held-out-camera split.
