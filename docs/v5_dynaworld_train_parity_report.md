# v5 Dynaworld Train Parity Report

## Summary

Dynaworld's current `fast_mac` trainer adapter is using the v5 renderer:

```text
src/train/renderers/fast_mac.py
third_party/fast-mac-gsplat/variants/v5/torch_gsplat_bridge_v5
torch.ops.gsplat_metal_v5
```

On Dynaworld's tiny known-good training baseline, v5 is much faster than the
Torch dense rasterizer but does not train to the same result. Taichi does train
to the same result on the same control. This makes the current v5 issue a
training-parity problem, not a throughput problem.

## Repo State

Dynaworld parent:

```text
e390f24
```

fast-mac-gsplat:

```text
8c4679d
```

The trainer-side v5 adapter is local/untracked in this checkout:

```text
src/train/renderers/fast_mac.py
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

## Interpretation

Dense and Taichi both reproduce the tiny baseline:

- dense eval loss: `0.065807`
- Taichi eval loss: `0.066723`

v5 fast-mac does not:

- v5 eval loss: `0.176579`
- v5 white-pixel fraction: `0.0498`
- v5 is fastest, but training behavior diverges after initially improving.

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

## What The Shader Engineer Should Check First

1. Compare v5 vs Taichi projected-2D forward and backward on the exact tiny
   baseline tensors at step 0 and after 10 optimizer steps.

2. Compare gradients for:

   ```text
   means2d
   conics
   colors
   opacities
   ```

   Report max abs diff, p99 abs diff, p99 relative diff, and any NaN/Inf counts.

3. Add/verify a tiny train-equivalence smoke:

   ```text
   B=23, H=W=32, G=512, seed=20260421, 100 optimizer steps
   expected v5 eval_loss ~= 0.066 if equivalent to Taichi/dense
   current v5 eval_loss ~= 0.177
   ```

4. Inspect v5 backward semantics around:

   - saved sorted IDs in `binned_ids`
   - `tile_stop_counts`
   - overflow tile forward/backward composition
   - unsorting gradients back through `perm`
   - the treatment of pixels whose forward transmittance saturated/stopped

5. Confirm that v5's pixel sampling convention and Gaussian exponent convention
   match Taichi for both forward and backward. Forward parity at step 1 suggests
   this is probably not the main issue, but it should still be locked down.

6. Run the v5 path with overflow fallback disabled or with a very high
   `max_fast_pairs` on this tiny case, if possible, to rule out a mixed fast
   path / overflow path gradient mismatch.

## Current Product Decision

Do not make v5 the default Dynaworld training renderer yet.

Use:

- `dense` as the canonical tiny correctness baseline
- `taichi` as the currently viable faster baseline candidate
- `fast_mac`/v5 as a renderer-under-test until the tiny train-parity control
  matches dense/Taichi

