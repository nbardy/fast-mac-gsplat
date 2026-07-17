# STAR-PRT v0 Research Notes

This fork is the compact STAR PRT/curve comparison lane. The first benchmark is
CPU-only and deliberately avoids Metal so it can run as a smoke gate while the
dense curve module and compact backward work are still moving in parallel.

## Benchmark

Run from the fast-mac-gsplat repo root:

```bash
uv run python variants/star_prt_v0/research_project/benchmarks/star_prt_curve_comparison.py \
  --out-json variants/star_prt_v0/research_project/benchmarks/results/star_prt_curve_comparison_smoke.json
```

The JSON compares:

- `per_frame_direct_projection_reference`: exact per-frame projection and dense
  CPU RGB render.
- `static_affine_star`: reference-frame STAR-style affine screen tube that
  ignores camera motion.
- `projective_first_order_affine`: one first-order projective Taylor
  approximation around the center frame.
- `segmented_affine`: piecewise affine interpolation between direct segment
  anchors.
- `projective_rational_curve_dense`: compiled homogeneous polynomial curve,
  evaluated as projective rational centers and rendered by the local dense CPU
  fallback.

The script also probes for Worker 1's future dense STAR-PRT module through
defensive candidate imports. Missing imports are reported in JSON as TODOs
instead of making the smoke fail.
