# v5 Dynaworld Train Parity Report

## Summary

Dynaworld's current `fast_mac` trainer adapter is using the v5 renderer:

```text
src/train/renderers/fast_mac.py
third_party/fast-mac-gsplat/variants/v5/torch_gsplat_bridge_v5
torch.ops.gsplat_metal_v5
```

On Dynaworld's tiny known-good training baseline, the original v5 handoff was
much faster than the Torch dense rasterizer but did not train to the same result.
Taichi did train to the same result on the same control. This made the issue a
training-parity problem, not a throughput problem.

This has now been traced to a real v5 Metal backward bug under saturated /
crowded splat conditions, not to a Dynaworld API mismatch. The failing backward
path used a per-pixel early-stop loop bound around `threadgroup_barrier()`, so
different pixels in the same tile could execute different numbers of barriers.
The fix is to keep the reverse chunk loop uniform across the tile and keep the
per-pixel early-stop test inside the loop.

Fix commit:

```text
971db71 Fix v5 saturated backward barriers
```

After that fix, the tiny Dynaworld training control no longer diverges:

```text
old v5 report: Eval/Loss ~= 0.176579
post-fix run:   Eval/Loss ~= 0.070275
```

The remaining gap to the old dense/Taichi numbers is small enough that it should
be retested under a fully pinned seed/runtime before calling it a renderer
semantic difference.

## Repo State

Original failing Dynaworld parent:

```text
e390f24
```

Original failing fast-mac-gsplat:

```text
8c4679d
```

Fix fast-mac-gsplat:

```text
971db71
```

## Baseline Under Test

Config:

```text
src/train_configs/local_mac_overfit_prebaked_camera.jsonc
```

Control settings:

- video variant: `small_32_2fps`
- frames: 23 prebaked DUSt3R frames
- render size: 32x32
- model size: 32
- splats: 128 tokens x 4 gaussians/token = 512
- train window: all frames per optimizer step
- loss: old tiny-baseline `l1_mse`
  - `l1_weight=1.0`
  - `mse_weight=0.2`
- near plane: `0.0001`
- seed: `20260421`
- run length: 100 optimizer steps
- W&B disabled

The only intended variable was renderer:

- `dense`: PyTorch dense reference renderer
- `taichi`: Taichi Metal renderer
- `fast_mac`: v5 Metal renderer via Dynaworld adapter

For `fast_mac`, the adapter path is:

```text
render_gaussian_frames(...)
  -> render_fast_mac_3dgs_batch(...)
  -> project_for_fast_mac_batch(...)
  -> torch_gsplat_bridge_v5.rasterize_projected_gaussians(...)
```

The v5 config used the trainer defaults:

```text
tile_size=16
max_fast_pairs=2048
alpha_threshold=1/255
transmittance_threshold=1e-4
background=(1, 1, 1)
enable_overflow_fallback=True
batch_strategy=flatten
```

## 100-Step Results

These are the original failing results before the v5 backward fix.

```text
dense    step=  1 loss=0.232605 render_mean=0.7098
dense    step= 10 loss=0.118925 render_mean=0.5343
dense    step= 25 loss=0.089385 render_mean=0.5519
dense    step= 50 loss=0.076461 render_mean=0.5630
dense    step=100 loss=0.065573 render_mean=0.5458
RESULT dense    train_loss=0.065573 eval_loss=0.065807 eval_l1=0.063866 eval_mse=0.009707 eval_ssim=0.535537 render_mean=0.5608 white=0.0000 seconds=85.16 it_s=1.17

taichi   step=  1 loss=0.235207 render_mean=0.7105
taichi   step= 10 loss=0.118555 render_mean=0.5559
taichi   step= 25 loss=0.105950 render_mean=0.5110
taichi   step= 50 loss=0.078382 render_mean=0.5603
taichi   step=100 loss=0.066641 render_mean=0.5604
RESULT taichi   train_loss=0.066641 eval_loss=0.066723 eval_l1=0.064770 eval_mse=0.009763 eval_ssim=0.508788 render_mean=0.5424 white=0.0000 seconds=45.28 it_s=2.21

fast_mac step=  1 loss=0.235207 render_mean=0.7105
fast_mac step= 10 loss=0.117463 render_mean=0.5367
fast_mac step= 25 loss=0.227202 render_mean=0.5190
fast_mac step= 50 loss=0.408617 render_mean=0.8261
fast_mac step=100 loss=0.174565 render_mean=0.6525
RESULT fast_mac train_loss=0.174565 eval_loss=0.176579 eval_l1=0.166798 eval_mse=0.048908 eval_ssim=0.221364 render_mean=0.6535 white=0.0498 seconds=18.43 it_s=5.43
```

Post-fix rerun of the same `fast_mac` control reached:

```text
FINAL_METRICS Loss=0.069294 Eval/Loss=0.070275 Eval/L1=0.068052 Eval/MSE=0.011115 Eval/SSIM=0.478317 Eval/PSNR=19.541011
```

## Interpretation

Dense and Taichi both reproduce the tiny baseline:

- dense eval loss: `0.065807`
- Taichi eval loss: `0.066723`

v5 fast-mac does not:

- v5 eval loss: `0.176579`
- v5 white-pixel fraction: `0.0498`
- v5 is fastest, but training behavior diverges after initially improving.

That interpretation applies to the original v5 handoff. After the saturated
backward fix, v5 reached `Eval/Loss=0.070275` on the same control, so the
stronger current interpretation is:

- the Dynaworld v5 adapter API was basically correct
- v5 forward was already close to Taichi on the tiny case
- v5 backward was corrupted when pixels saturated/stopped at different depths
- crowded / high-opacity projected scenes were the missing test case

The first-step losses for Taichi and v5 are identical to six significant
figures:

```text
taichi   step=1 loss=0.235207
fast_mac step=1 loss=0.235207
```

That strongly suggests the initial projected-input forward path is close to
Taichi for this case. The divergence appears during optimization, so the first
suspect is v5 training/backward parity or a train-time semantic mismatch in the
v5 autograd path.

## Adapter Semantics To Match

Dynaworld projection sorts splats by camera-space z before handing projected
splats to renderer adapters:

```text
src/train/renderers/common.py
project_gaussians_2d_batch(...)
```

Both Taichi and v5 then receive monotonic rank depths preserving that sorted
order. This intentionally avoids depending on camera z gradients through sort:

```text
depths = arange(G) / (G - 1)
```

Dense compositing simply uses the sorted tensor order and does not call a
renderer-side sort.

All three paths use white background and alpha clamp/max of `0.99`.

## Root Cause

The relevant v5 kernels were:

```text
variants/v5/csrc/metal/gsplat_v5_kernels.metal
tile_fast_backward_saved(...)
tile_overflow_backward(...)
```

Before the fix, each pixel/thread computed its own `end_i` from forward
transmittance saturation and then used that value as the loop bound for the
reverse chunk loop. That loop contains `threadgroup_barrier()`.

This is invalid for a tiled cooperative kernel. Threads in the same threadgroup
must reach the same barriers in the same order. In saturated/crowded scenes,
some pixels stopped earlier than others, so the tile could execute divergent
barrier control. The result was not an obvious crash; it was wrong gradients.

The fix is:

```text
fast path backward:     loop over uniform tile stop_count
overflow path backward: loop over uniform tile count
per-pixel contribution: guard with global_i < end_i inside the loop
```

This preserves the forward early-stop math while keeping barrier control uniform.

## Validation After Fix

The bundled v5 reference check now includes a saturated many-splat case in
addition to the original tiny B=1/B=2 checks.

Command:

```bash
cd third_party/fast-mac-gsplat/variants/v5
python tests/reference_check.py
```

Representative post-fix output:

```text
B=1 image max error: 5.960464477539063e-08
B=1 means grad max error: 2.4010660126805305e-10
B=1 conics grad max error: 9.313225746154785e-10
B=1 colors grad max error: 9.313225746154785e-10
B=1 opacities grad max error: 1.862645149230957e-09
B=2 image max error: 5.960464477539063e-08
B=2 means grad max error: 3.710738383233547e-10
B=2 conics grad max error: 1.862645149230957e-09
B=2 colors grad max error: 9.313225746154785e-10
B=2 opacities grad max error: 1.862645149230957e-09
saturated image max error: 2.086162567138672e-07
saturated means grad max error: 2.3283064365386963e-10
saturated conics grad max error: 1.1920928955078125e-07
saturated colors grad max error: 3.725290298461914e-09
saturated opacities grad max error: 9.313225746154785e-10
eval overflow disabled raises: ok
```

Dynaworld adapter parity against Taichi was also checked on spread and crowded
projected 3D cases. In the crowded case, the post-fix v5-vs-Taichi differences
were near numeric noise:

```text
CASE crowded B 2 G 64
loss taichi/fast 0.7882294058799744 0.7882294058799744
image max=2.98023e-07
xyz max=2.23517e-08
scales max=5.32717e-07
quats max=7.12462e-08
opacity max=2.79397e-09
rgb max=1.86265e-09
```

## Remaining Checks

The fix resolves the known parity break, but these checks are still worth keeping
before making v5 the default training renderer everywhere:

- run the 100-step control with fully pinned model/data seed and compare dense,
  Taichi, and v5 in one process if possible
- replay real Dynaworld projected traces at higher splat counts
- keep saturated/crowded gradient parity in CI or at least in the local release
  checklist
- separately profile whether v5 or v3 should be the default for B=1 vs native
  batch training

## Current Product Decision

The original "do not use v5 for training" conclusion is no longer supported by
the evidence. v5 should be treated as a viable training renderer candidate after
`971db71`, with one caveat: promote it by benchmark tier, not by assumption.

Use:

- `dense` as the canonical tiny correctness baseline
- `taichi` as the compatibility baseline
- `fast_mac`/v5 as the native batch Metal candidate, now requiring larger trace
  replay and long-run stability checks rather than basic tiny-parity debugging
