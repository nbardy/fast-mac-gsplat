# V9 HW Eval Parity Probe Notes

The v9 fixed-eval path is not a full v8 replacement yet. It renders
screen-space Gaussians as instanced hardware quads into an `RGBA32Float` MPS
tensor and uses hardware source-over blending. The RGB channels are
premultiplied by alpha, so they are comparable to v8 RGB only when v8 uses a
black background.

The parity harness in
`variants/v9_hw_eval_parity_probe/benchmarks/benchmark_parity_v8.py` generates
one identical set of projected tensors, passes those tensors to v9 fixed eval
and v8 forward eval, then compares v9 `[..., :3]` against v8 RGB after both
native ops have returned. CPU readback is used only for validation and
reporting.

The harness requires both native extensions to be built:

```bash
(cd variants/v8_hw_eval && python3 setup.py build_ext --inplace)
(cd variants/v9_hw_eval_parity_probe && python3 setup.py build_ext --inplace)
```

If the v8 extension is missing, the harness now raises a direct error before
calling `torch.ops.gsplat_metal_v8_hw_eval`.

Current assumptions for rows marked `comparable_to_v8=true`:

- batch size is 1;
- inputs are already projected pixel-space `means2d` and conics;
- v8 background is `(0, 0, 0)`;
- the case has one Gaussian, so v8 depth and tile ordering are no-ops;
- v9 direct output is used only for widths where `width * 16` is 256-byte
  aligned.

Current v9 limitations called out in each row:

- eval-only, no backward;
- no batching;
- no depth sort or v8-equivalent multi-splat ordering;
- no tile/imageblock path;
- no transmittance early termination;
- no non-black background composition;
- direct render target output requires aligned rows.

The important blocker for real v8 equivalence is ordering and state: v8 sorts
by depth, bins by tile, sorts tile-local ids, and can stop once transmittance is
low. The current v9 path blends instanced quads through fixed-function hardware
blending, and the observed multi-splat output does not match v8's ordered
source-over contract. It can match single-splat black-background cases, but it
is not v8-equivalent for multi-splat scenes yet.

The `depth_mismatch` rows in the smoke report may show low numeric error on
this hardware. They are still marked `comparable_to_v8=false` because they are
probing the observed hardware blend order, not a documented or v8-compatible
ordering guarantee.
