# 2026-04-24 Hardware Rasterizer Fast Backward Handoff

Wrote a durable handoff for the v2-v7.3 renderer lineage after switching to
GPT-5.5. The user asked for one document defining all versions, what worked and
did not, what is getting better or worse, and the goal for merging hardware
rasterization with fast backward.

Added:

- `docs/hardware_rasterizer_fast_backward_handoff.md`

Also linked it from the README benchmark/docs paragraph.

The new handoff records the current working model:

- training should use v5/v6-style compute paths today
- hardware rasterization remains useful for forward/eval research
- v7.3 hybrid is the current practical shape because it routes training to v5
  compute and keeps hardware eval available
- the next real blocker is returning hardware raster output as a GPU-resident
  torch MPS tensor without CPU staging

Key numbers captured in the handoff:

- 4K/64K B=1 uniform F+B:
  - `v5_batched`: 73.681 ms
  - `v72_tiled_k2`: 438.782 ms
  - `v73_hybrid_k2`: 76.464 ms
  - `v73_hybrid_k2_hwtrain`: 436.552 ms
- 4K/64K B=1 uniform forward:
  - `v5_batched`: 11.781 ms
  - `v72_tiled_k2`: 183.679 ms
  - `v73_hybrid_k2`: 133.129 ms

Interpretation:

- v7.2 fixed the v7.1 `pixels x splats` capture cliff but remained CPU/state
  bound.
- v7.3 hybrid restored training speed by using compute backward, not by making
  hardware backward fast.
- Hardware eval still loses at 4K until texture/readback interop is solved.
