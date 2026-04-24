# v7.1 engineering notes

## Why the old v7 failed

The previous handoff had two separate problems:

1. **Architecture** — backward replayed all gaussians for every pixel, which destroys the point of hardware-forward rendering during training.
2. **Local math bugs** — the replay kernel used `go * T` for color gradients instead of `go * T * alpha`, and mean gradients had the wrong sign.

The result matched the field report: image parity looked okay, but gradient parity regressed badly and large-scene backward was unusably slow.

## What this v7.1 handoff does

- keeps the graphics forward image path,
- adds a saved front-K visibility state,
- uses exact reverse recursion for the common path,
- keeps replay only as a sparse overflow fallback,
- preserves a clean Torch autograd boundary.

## Design limits in this source handoff

- `front_k` is runtime-configurable but capped at `8` in the Metal kernel source (`kMaxFrontK = 8`).
- The capture pass still scans sorted splats per pixel. It stops as soon as it proves overflow (`K+1` hits), but it is not yet a fully graphics-native front-layer peel.
- The bridge still stages tensors through CPU buffers. That is not the end-state performance design; it is a clean handoff boundary.
- Depth gradients remain zero, same as the old handoff, because depth is only used for sorting before the custom op call.

## Why this is still meaningfully better

Even with the same high-level bridge, the new backward work changes from:

- **all pixels × all gaussians**

to:

- **all non-overflow pixels × K**
- **overflow pixels × all gaussians**

That is the first version that has a plausible training story when overflow is sparse.

## Validation included here

The package contains a pure-Python manual backward path that implements the exact same front-K / overflow split. The included tests compare it against dense autograd reference and hit near-machine precision on small cases, including overflow and alpha-clamp cases.
