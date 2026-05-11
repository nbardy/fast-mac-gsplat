# Phase 1: Projected Trainer Harness

## Goal

Fit projected `ScreenTimeTube` parameters against small video tensors without
requiring a Metal backward pass.

## Strategy

- Use a dense differentiable PyTorch renderer as the gradient owner.
- In Phase 1, use the Metal path only for parity/stats after a fit; backward
  ownership is handled separately in Phase 3.
- Start with deterministic tiny synthetic targets.
- Add real video frame loading only as a target source, not as a camera model.

## Gate

The smoke harness must show a deterministic loss drop on CPU:

```bash
python3 research_project/trainer_harness/smoke_train.py
```

The output must include initial/final loss, step count, scene name, and device.

Video loader plumbing is checked separately:

```bash
python3 research_project/trainer_harness/train_video.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 4 --target-size 16 --max-frames 2 --steps 1 --lr 0.02 --device cpu
```

That command proves target loading and optimization plumbing only. It is not a
video-quality result.
