# Phase 5: Promotion Decision

## Goal

Decide whether STAR-UVT should remain an isolated research variant or be wired
into the production GFlow/FasterGS MVP path.

## Required Evidence

- A real video/training comparison against the current dynamic FasterGS path.
- Same source video, resolution, frame count, and train/held-out split.
- Reconstruction loss and renderer timing.
- Visual proof, such as a contact sheet or viewer capture.
- Clear failure modes, including unstable ordering, tile overflow, and backward
  mismatch if Metal backward is introduced.

## Current Decision

2026-05-10: keep STAR-UVT isolated in `variants/star_uvt_v0/`. Do not wire it
into the production GFlow/FasterGS MVP path yet.

## Evidence In Hand

- Gate 0 forward Metal parity on six deterministic projected UVT scenes.
- Orthographic, pinhole, and Dynaworld `CameraSpec` projection smokes.
- Dense-backward, hybrid autograd, true-Metal backward parity, unstable fallback,
  and tile-backward autograd smokes.
- Bounded backward timing on tiny and `large_local` synthetic cases.
- Synthetic UVT-vs-per-frame training comparison.
- Fixture-video UVT-vs-per-frame comparison with contact-sheet proof.

## Missing For Promotion

- Current dynamic FasterGS baseline on the same source video.
- Same resolution, frame count, and train/held-out split.
- Held-out-camera metrics where cameras exist.
- Visual proof against the production baseline, not just the local per-frame
  research baseline.
- Production-scale backward timing and memory evidence.
- A decision on gradients through discrete depth-order changes.

The current phase is complete as a negative promotion decision: the work remains
valuable research infrastructure, but it should not block or replace the MVP
path.
