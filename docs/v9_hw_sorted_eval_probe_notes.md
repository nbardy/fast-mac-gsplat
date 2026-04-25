# V9 HW Sorted Eval Probe Notes

Scope: `variants/v9_hw_sorted_eval_probe`.

This probe adds `render_gaussian_eval_rgba_sorted(...)`, a Python/Torch wrapper
around the existing fixed-eval Metal render path. It computes a stable
permutation from `depths.detach()` on MPS, gathers `means2d`, `conics`,
`colors`, and `opacities`, then calls the native source-over renderer.

Sort convention:

- `descending=False` is the default and matches the v8 wrapper convention:
  `torch.argsort(depths.detach(), stable=True)`.
- Lower numeric depths are submitted first.
- Equal-depth splats keep input order.
- `descending=True` submits higher numeric depths first.

This is useful for order-controlled eval probes, but sorting is not enough to
unlock v8 parity. Remaining gaps:

- fixed hardware source-over blending updates color in submitted draw order,
  while v8 computes explicit front-to-back `C/T` transmittance;
- RGBA output alpha is fixed-function accumulated alpha, not v8's training
  metadata or final transmittance contract;
- no batching API;
- no backward path;
- direct render output still requires 256-byte-aligned rows;
- no programmable per-pixel tile/imageblock state or raster-order-group path.

If depths use the common convention where smaller is closer, exact painter-style
source-over compositing would submit farther splats first. That can be probed
with `descending=True` when larger depth means farther, but it still does not
restore exact v8 stop thresholds or transmittance metadata.
