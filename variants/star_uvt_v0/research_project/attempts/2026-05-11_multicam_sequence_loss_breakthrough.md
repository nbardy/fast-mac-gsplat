# 2026-05-11 Multicam Sequence-Loss Breakthrough

Question: is STAR-UVT underfitting multicam because each training step renders
the full 16-frame tube sequence but only backprops one sampled frame?

## Change

`multicam_heldout_compare.py` now has:

```text
--uvt-loss-scope sampled_frame | view_sequence
```

The old behavior is `sampled_frame`. The new `view_sequence` mode still samples
one train camera per step, but applies the reconstruction loss to all rendered
frames for that view. This matches the actual STAR-UVT render granularity:
`render_world_tube_sequence(...)` already produces all `T` frames.

## Smoke

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-tubes 8 --uvt-lr 0.03 \
  --uvt-loss-scope view_sequence --splat-count 8 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_viewseq_loss_smoke_16_2f_1s
```

The smoke passed and wrote `loss_scope: view_sequence` into the report.

## 128px Pilot

Command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 128 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope view_sequence \
  --uvt-init-views first --uvt-init-precision-xy 30.0 \
  --uvt-init-lambda-t 0.35 --uvt-init-opacity 0.35 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_128_16f_60s_uvt256_viewseq_loss_oldinit
```

Result:

```text
STAR-UVT, 256 tubes, old init, view_sequence loss:
  train PSNR        15.593097686767578
  heldout PSNR      13.423128128051758
  steps             135
  train loop        60.03055345802568s
  eval render       0.6185550000227522s

Direct splats, 2048, fast_mac:
  train PSNR        20.368911743164062
  heldout PSNR      10.850052833557129
  steps             2840
  train loop        60.010647957999026s
  eval render       0.40161637499113567s

V-JEPA F32 reference row:
  train PSNR        19.4875
  heldout PSNR      13.6248
  train loop        18m00s
  resolution        256px
```

## Read

This is the first strong multicam STAR-UVT result. At the same local
60-second/128px/16-frame budget, STAR-UVT now beats the direct splat heldout
baseline by `2.573075294494629` dB. It is also close to the 256px V-JEPA F32
heldout reference despite using a shorter 60-second local budget and 128px
frames, so it warrants a 256px follow-up.

The root cause was an optimization mismatch, not the rasterizer at 128px:
sampled-frame loss discarded most of each full-sequence render. The next
comparison should keep `view_sequence` as the default STAR-UVT multicam loss
scope.

## 256px Bounded Follow-Up

Command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope view_sequence \
  --uvt-init-views first --uvt-init-precision-xy 30.0 \
  --uvt-init-lambda-t 0.35 --uvt-init-opacity 0.35 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_viewseq_loss_oldinit
```

Result:

```text
STAR-UVT, 256 tubes, old init, view_sequence loss:
  train PSNR        10.676740646362305
  heldout PSNR      10.409326553344727
  steps             42
  train loop        61.391295250010444s
  eval render       1.7832725839980412s

Direct splats, 2048, fast_mac:
  train PSNR        17.51949119567871
  heldout PSNR      10.730738639831543
  steps             2225
  train loop        60.021517666988075s
  eval render       0.8745809590036515s
```

Read: the 256px run is step-starved at a 60-second budget. STAR-UVT does not
yet beat direct splats at 256px, and it is far below the V-JEPA F32 256px
heldout PSNR `13.6248`. A fair 256px promotion run needs either a longer STAR
budget, fewer/lighter rendered frames per optimizer step, or rasterizer and
training-throughput work.

## 256px Temporal-Window Follow-Up

Command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-init-views first --uvt-init-precision-xy 30.0 \
  --uvt-init-lambda-t 0.35 --uvt-init-opacity 0.35 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_oldinit
```

Result:

```text
STAR-UVT, 256 tubes, old init, temporal_window loss, 4-frame windows:
  train PSNR        12.275136947631836
  heldout PSNR      11.813445091247559
  steps             157
  train loop        60.151555499993265s
  eval render       1.3796198749914765s

Direct splats, 2048, fast_mac:
  train PSNR        19.916349411010742
  heldout PSNR      10.738123893737793
  steps             2959
  train loop        60.007742791989585s
  eval render       0.38876454101409763s
```

Read: temporal windows restore a same-budget 256px heldout win over direct
splats by improving STAR step count from 42 to 157. This does not clear the
promotion bar: STAR remains below the V-JEPA F32 heldout PSNR `13.6248`, and
the current STAR-UVT multicam eval render is about `3.55x` slower than the
paired `fast_mac` direct-splat render. The next speed work should profile and
optimize the rasterizer/training loop rather than only adding representation
capacity.

## Render Timing Follow-Up

Two probes now separate fixed overhead from trained-model behavior.

Initialized timing probe:

```bash
python3 research_project/benchmarks/multicam_render_timing_probe.py \
  --target-size 256 --max-frames 16 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --iterations 1 --warmup-iterations 1 \
  --out-json research_project/benchmarks/results/multicam_render_timing_probe_mps_256_16f_uvt256_splat2048_stats.json
```

Result: initialized STAR full projection+render totaled
`0.1808375830296427s` across the three eval sequences versus direct splats
`0.22867970800143667s`. STAR render-only was only
`0.027068290975876153s`; projection-only was `0.1553649159905035s`. Initial
Metal stats showed pair ratio about `0.83-0.86`, zero overflow, and max tile
count `33-35`.

Trained temporal-window rerun:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-init-views first --uvt-init-precision-xy 30.0 \
  --uvt-init-lambda-t 0.35 --uvt-init-opacity 0.35 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_oldinit_timing_fields
```

Result: STAR train PSNR `11.885068416595459`, heldout PSNR
`11.320009231567383`, 159 steps, render-only eval `1.3073187510017306s`.
Direct splats train PSNR `20.24591064453125`, heldout PSNR
`10.723106384277344`, 3138 steps, render-only eval `0.4013093340327032s`.

Read: STAR still wins heldout against direct splats in this rerun, but trained
STAR render-only is about `3.26x` slower. Because initialized STAR is faster
than initialized splats, the speed regression is likely learned-support or
tile-load behavior. Future trained reports now include STAR Metal stats so the
next rerun can check pair ratio, unstable-tile fraction, overflow, and max tile
count directly.

## Metal Stats and Support-Floor Follow-Up

Trained stats rerun:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-init-views first --uvt-init-precision-xy 30.0 \
  --uvt-init-lambda-t 0.35 --uvt-init-opacity 0.35 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_oldinit_timing_stats
```

Result: STAR heldout PSNR `11.136907577514648` versus direct splats
`10.723885536193848`, but STAR render-only eval was
`1.1644907489826437s` versus splats `0.5643962080066558s`. The STAR Metal
stats explain the slowdown: pair ratio `2.98-3.78`, unstable-tile fraction
`1.0`, max tile count `174-222`, and overflow on `8155-8192` of 8192 UVT
tiles.

Support-floor experiment:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-init-views first --uvt-init-precision-xy 30.0 \
  --uvt-init-lambda-t 0.35 --uvt-init-opacity 0.35 \
  --uvt-min-precision-xy 30.0 --uvt-min-lambda-t 0.35 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_support_floor30_lam035
```

Result: negative. STAR heldout PSNR fell to `9.554825782775879`, below direct
splats at `10.727174758911133`. STAR render-only eval was still
`1.2437176250386983s` versus splats `0.3679181660118047s`. Pair ratio remained
`2.34-4.53`, unstable-tile fraction stayed `1.0`, and two of three eval views
still had many overflowed tiles.

Read: the speed pathology is trained tile-load and order instability. A hard
minimum precision floor is too blunt: it hurts quality and does not restore UVT
compactness. The next experiment should penalize or constrain unstable
depth-order behavior and total tile load directly.

## Velocity Regularization Follow-Up

Command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-init-views first --uvt-init-precision-xy 30.0 \
  --uvt-init-lambda-t 0.35 --uvt-init-opacity 0.35 \
  --uvt-velocity-reg 0.01 --uvt-depth-velocity-reg 0.1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_velreg001_z01
```

Result: mixed quality, negative speed. STAR heldout PSNR improved to
`11.486005783081055`, above direct splats at `10.724501609802246`, but
render-only eval remained slow: STAR `1.179607957979897s` versus splats
`0.4023879590095021s`. STAR Metal stats stayed pathological: pair ratio
`2.50-3.68`, unstable-tile fraction `1.0`, overflow on all 8192 UVT tiles, and
max tile count `171-220`.

Read: penalizing velocity is not enough to restore UVT compactness. The next
candidate should constrain tile load or order instability more directly,
possibly with a differentiable projected support/tile-load proxy rather than
only parameter-level regularization.

## Projected Tile-Load Regularization

Implementation:

- Added a projected sequence split inside `multicam_heldout_compare.py`, so the
  training step can reuse the same `ma`, `q_uvt`, `depth_beta`, and `opacity`
  tensors for both render and regularization.
- Added `--uvt-tile-load-reg`, `--uvt-tile-load-target`, and
  `--uvt-depth-slope-reg`.
- The tile-load proxy mirrors the Metal binning bound by using the diagonal of
  the inverse packed UVT precision and the opacity threshold support radius.
  With a positive target, the loss penalizes only excess projected support.

Strong 20-second probe:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.02 --uvt-tile-load-target 450 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_temporal_window4_tileload002_target450
```

Result: the mechanism works but was too strong. STAR heldout PSNR was
`9.005831718444824` versus direct splats `9.248841285705566`, but STAR
render-only eval was `0.27568720804993063s` versus splats
`0.31377941701794043s`. Metal stats had pair ratio `0.92-0.93`, zero overflow,
and max tile count `35-36`.

Same-budget no-reg control:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_temporal_window4_noreg_instrumented
```

Result: STAR quality won but speed failed. STAR heldout PSNR was
`10.790498733520508` versus direct splats `9.37923812866211`, but STAR
render-only eval was `1.1133767089922912s` versus splats
`0.25056229197070934s`. The logged tile-load proxy jumped from `580` at step 1
to `1797` at step 10, `3266` at step 20, and `66910` at step 40. Final Metal
stats had pair ratio `2.50-3.55`, unstable-tile fraction `1.0`, and overflow
on `619-3880` tiles.

Soft 20-second probe:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.005 --uvt-tile-load-target 1500 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_temporal_window4_tileload0005_target1500
```

Result: first good tradeoff. STAR heldout PSNR `10.76546859741211` stayed near
the no-reg control while beating direct splats `9.407424926757812`. STAR
render-only eval dropped to `0.2995397500053514s`, close to direct splats
`0.25954554200870916s`. Pair ratio dropped to `1.08-1.13`, max tile count
`53-63`, and overflow was zero.

Tile-load-only 60-second run:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.005 --uvt-tile-load-target 1500 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0005_target1500
```

Result: first promoted 256px multicam speed/quality point. STAR heldout PSNR
`11.419742584228516` beat direct splats `10.707969665527344`. STAR render-only
eval was `0.2937038330419455s`, faster than direct splats at
`0.47125908301677555s`. Metal stats: pair ratio `1.06-1.10`, max tile count
`58-59`, zero overflow. Active tiles remained mostly unstable
(`0.96-0.98` unstable fraction), motivating the depth-slope follow-up.

Depth-slope follow-up:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.005 --uvt-tile-load-target 1500 \
  --uvt-depth-slope-reg 0.05 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0005_target1500_depthslope005
```

Result: new best 256px multicam speed/quality point. STAR heldout PSNR
`11.877435684204102` beat direct splats `10.717645645141602`. STAR render-only
eval was `0.2507540419755969s`, faster than direct splats at
`0.26824004197260365s`. Metal stats: pair ratio `0.98-1.00`, max tile count
`52-54`, zero overflow, and unstable-tile fraction `0.93-0.96`. This improves
both heldout quality and render speed over tile-load-only.

Stronger slope follow-up:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.005 --uvt-tile-load-target 1500 \
  --uvt-depth-slope-reg 0.2 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0005_target1500_depthslope02
```

Result: negative at the longer budget. STAR heldout PSNR fell to
`11.32148551940918`, and STAR render-only eval rose to
`0.4422728330246173s`, slightly slower than direct splats at
`0.41102154200780205s`. Keep `0.05`, not `0.2`, as the current slope weight.

Depth-margin diagnostic:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.005 --uvt-tile-load-target 1500 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-depth-margin-reg 0.01 --uvt-depth-margin 0.05 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_temporal_window4_tileload0005_target1500_depthslope005_depthmargin001_m005
```

Result: not promoted. STAR heldout PSNR was `11.522964477539062`, above the
20-second direct-splat baseline at `9.3114652633667`, but pair ratio worsened
to `1.08-1.12`, max tile count rose to `59-65`, and unstable-tile fraction
stayed `0.97-0.99`. The regularizer quickly reduced the logged
`depth_margin_proxy` from `1.0` at step 1 to about `0.0075-0.016`, but that did
not translate into stable Metal tiles. Do not promote depth-margin reg yet.

Tile-t=1 rasterizer diagnostic:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.005 --uvt-tile-load-target 1500 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0005_target1500_depthslope005_tilet1
```

Result: useful rasterizer evidence, not a quality promotion. At 20 seconds,
`--uvt-tile-t 1` reached STAR heldout PSNR `11.10805892944336` with render-only
eval `0.16470333401230164s`, pair ratio `1.85-1.88`, max tile count `46-50`,
zero overflow, and zero unstable tiles. The paired 20-second direct-splat run
had heldout PSNR `9.486968994140625` and render-only eval
`0.3029780409706291s`. At 60 seconds, STAR heldout PSNR was
`11.079706192016602` versus direct splats `10.713973045349121`; STAR render-only
eval was `0.1884017909760587s` versus splats `0.4057717919931747s`. Metal stats
again showed zero unstable tiles and zero overflow, with pair ratio
`1.72-1.75` and max tile count `42-45`. Shrinking temporal tile depth therefore
removes the expensive unstable path and improves render-only time, but it loses
about `0.80` heldout PSNR versus the then-current `tile_t=2` 60-second best
(`11.877435684204102`). This rejected the strict old tile-load setting, not
`tile_t=1` as a whole.

Relaxed tile-t=1 promotion:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 3000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target3000_depthslope005_tilet1
```

Result: promoted at this point as the bounded 256px multicam speed/quality
default.
The diagnostic no-tile-load 20-second run showed why some support pressure is
still needed: STAR heldout rose to `11.421504020690918`, but max tile count
hit `142-181` and overflowed `6604-14322` tiles. Relaxing rather than removing
the pressure worked. At 20 seconds, `--uvt-tile-load-reg 0.001
--uvt-tile-load-target 3000 --uvt-tile-t 1` reached STAR heldout PSNR
`11.720141410827637`, render-only eval `0.18447483397903852s`, pair ratio
`2.75-2.81`, max tile count `82-85`, zero overflow, and zero unstable tiles.
At 60 seconds, STAR reached train PSNR `12.820858001708984`, heldout PSNR
`12.002521514892578`, and 246 steps. Direct splats reached train PSNR
`19.738224029541016`, heldout PSNR `10.725918769836426`, and 2833 steps. STAR
render-only eval was `0.19579870899906382s` versus splats
`0.43779041699599475s`. Metal stats had pair ratio `2.27-2.56`, max tile count
`64-72`, zero overflow, and zero unstable tiles. This beats the prior `tile_t=2`
best on heldout PSNR and render-only time, while still trailing the V-JEPA F32
reference heldout PSNR `13.6248`.

384-tube capacity probe:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 384 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 3000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt384_temporal_window4_tileload0001_target3000_depthslope005_tilet1
```

Result: negative, no 60-second escalation. STAR heldout PSNR fell to
`11.04841136932373` versus the 256-tube 20-second relaxed `tile_t=1` value
`11.720141410827637`; train PSNR was `11.771284103393555`, and render-only eval
was `0.29677454198827036s`. Metal stats still had zero overflow and zero
unstable tiles, but max tile count rose to `110-123` and pair ratio rose to
`3.13-3.26`. More tubes under the same budget reduce step throughput and push
the default tile capacity, so capacity alone is not the next lever.

256px view-sequence retry:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope view_sequence \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 3000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_viewseq_tileload0001_target3000_depthslope005_tilet1
```

Result: negative, no 60-second escalation. STAR completed only 26 steps in 20
seconds, reached train PSNR `9.258035659790039`, and heldout PSNR
`8.97468376159668`. The paired direct-splat baseline reached heldout PSNR
`9.148686408996582`. Metal stats were still stable with zero overflow and zero
unstable tiles, but full-sequence loss remains too expensive at 256px. Keep
temporal-window training as the 256px default.

LR 0.05 probe:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 3000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_lr005_temporal_window4_tileload0001_target3000_depthslope005_tilet1
```

Result: negative, no 60-second escalation. STAR heldout PSNR fell to
`11.359124183654785` versus the LR `0.03` 20-second relaxed `tile_t=1` value
`11.720141410827637`; train PSNR fell to `11.424661636352539`. Metal stats
still had zero overflow and zero unstable tiles, but pair ratio rose to
`3.32-3.43`. At this pre-bundled-reducer point, keep LR `0.03` for the 256px
recipe.

LR 0.02 probe:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.02 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 3000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_lr002_temporal_window4_tileload0001_target3000_depthslope005_tilet1
```

Result: negative, no 60-second escalation. STAR heldout PSNR fell to
`10.951751708984375`, train PSNR fell to `11.512050151824951`, and render-only
eval was `0.16975908196764067s`. The rasterizer remained stable with zero
overflow and zero unstable tiles. Together with the LR `0.05` rejection, this
kept LR `0.03` as the then-current 256px optimizer setting. The later
bundled-reducer retune supersedes this for full 60-second runs.

Depth-slope no-op check:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 3000 \
  --uvt-tile-t 1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_temporal_window4_tileload0001_target3000_tilet1_nodepthslope
```

Result: do not sweep depth-slope further under `tile_t=1`. The logged
`depth_slope_proxy` was exactly `0.0`, matching the projection math:
`depth_beta` is temporal-only and a one-frame tile has zero temporal
half-extent. The no-depth-slope run reached heldout PSNR
`11.638251304626465`, close to but not above the saved 20-second report with
the historical `--uvt-depth-slope-reg 0.05` flag (`11.720141410827637`).

Tile-load target 5000 promotion:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 5000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target5000_depthslope005_tilet1
```

Result: promoted at this point as the bounded 256px multicam speed/quality
default.
At 20 seconds, target `5000` reached heldout PSNR `11.750381469726562`, just
above target `3000` at `11.720141410827637`, with zero overflow, max tile count
`103-116`, and one eval view showing only `0.00055` unstable-tile fraction. At
60 seconds, STAR reached train PSNR `12.996597290039062`, heldout PSNR
`12.157893180847168`, and 254 steps. Direct splats reached train PSNR
`17.520974159240723`, heldout PSNR `10.75090217590332`, and 2209 steps. STAR
render-only eval was `0.21316362501238473s` versus splats
`0.4603952500037849s`. Metal stats had pair ratio `3.33-3.62`, max tile count
`98-115`, zero overflow, and zero unstable tiles. This beats target `3000` on
heldout PSNR while staying below tile capacity, but it is closer to the cap and
still below the V-JEPA F32 reference heldout PSNR `13.6248`.

Tile-load target 7000 boundary:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target7000_depthslope005_tilet1
```

Result: rejected as the default despite slightly better PSNR, because the 60s
run crosses the tile-capacity guardrail. At 20 seconds, target `7000` reached
heldout PSNR `11.859132766723633`, zero overflow, zero unstable tiles, and max
tile count `114-119`. At 60 seconds, STAR heldout PSNR rose to
`12.210857391357422`, but max tile count reached `123-137` and one eval view
overflowed `499` tiles. STAR render-only eval was still fast at
`0.20270062497002073s` versus splats `0.3378250010428019s`, but the default
tile capacity is no longer respected. Keep target `5000` as the safe current
default; target `7000` is a useful signal that quality still increases with
support if the rasterizer capacity or binning can handle it.

Tile-load target 7000 with capacity 256:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256
```

Result: promoted as the local quality default with a memory caveat. Raising the
tile capacity removed the target-`7000` overflow and improved heldout PSNR to
`12.388733863830566`. STAR train PSNR was `12.911274433135986`, STAR steps were
175, and STAR render-only eval was `0.20699129099375568s` versus direct splats
at `0.29931937501532957s`. Direct splats reached heldout PSNR
`10.748902320861816`. Metal stats had pair ratio `3.50-3.82`, max tile count
`114-127`, zero overflow, and zero unstable tiles. The cost is doubled Metal
buffer memory (`16.97MB` to `33.75MB`) and lower STAR step throughput than
cap-128 target `5000`, so keep the memory caveat attached to this default.

Tile-load target 9000 with capacity 256:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 9000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_temporal_window4_tileload0001_target9000_depthslope005_tilet1_cap256
```

Result: rejected, no 60-second escalation. STAR completed only 58 steps in 20
seconds, heldout PSNR fell to `10.476442337036133`, and train PSNR fell to
`10.994823932647705`. Metal stats had zero overflow and zero unstable tiles,
but max tile count was still high at `119-126`. More support under cap-256
slowed optimization enough to underfit, so target `7000` remains the support
setting.

Cap-256 short-budget check:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256
```

Result: cap-256 is not the right cheap gate. At 20 seconds, cap-256 target
`7000` completed only 60 steps, reached train PSNR `11.145819187164307`, and
heldout PSNR `10.551692962646484`. Cap-128 target `7000` at the same 20-second
budget completed 82 steps and reached heldout PSNR `11.859132766723633`. Both
had zero overflow and zero unstable tiles. Keep cap-256 as the 60-second quality
default, but use cap-128 for short exploratory probes unless the probe is about
longer-budget overflow.

Tile-load target 6000 midpoint:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 6000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_uvt256_temporal_window4_tileload0001_target6000_depthslope005_tilet1
```

Result: rejected, no 60-second escalation. STAR heldout PSNR was
`11.611959457397461`, below target `5000` at `11.750381469726562`; train PSNR
was `12.329296112060547`, render-only eval was `0.20948595702066086s`, and max
tile count rose to `111-127`. It had zero overflow and zero unstable tiles, but
it was already at the default capacity edge without improving quality.

Train-step timing probe for the current cap-256 default:

```bash
python3 research_project/benchmarks/multicam_train_step_timing_probe.py \
  --device mps --steps 8 --warmup-steps 2 \
  --out-json research_project/benchmarks/results/multicam_train_step_timing_probe_mps_256_16f_current_default.json
```

Initial result: the remaining train-time speed blocker was not the forward
renderer by itself. STAR-UVT averaged `0.3036720311138197s` per profiled train
step versus `0.013559588376665488s` for direct fast-mac splats. STAR forward
render took only `0.002225083371740766s` for the 4-frame temporal window, while
STAR backward took `0.23889267686899984s` and worldtube projection took
`0.05870089036034187s`.

The deeper microbreakdown showed a specific backward waste: the stable backward
bridge emitted `67108864` fixed sample slots for a 4-frame window, but only
`458991` were valid. MPS reductions over the mostly empty buffer took
`0.1513007489265874s`. I patched the stable backward Metal kernel to write
gradient samples through a device counter and slice the returned tensors before
the MPS `index_add_` reductions. Stable backward, unstable backward, Gate 0
forward parity, and tile-autograd smokes still passed.

After the compact-output patch, the same timing command reports STAR-UVT at
`0.1739891823817743s` per train step. Backward dropped to
`0.09416389050602447s`; its microbreakdown is `0.007699042034801096s` for Metal
sample generation, `0.0325404170434922s` for reductions over `491831` compact
samples, and `0.0021938749705441296s` for projection VJP. Worldtube projection
forward is now the other large cost at `0.0704665103694424s`. This answers the
rasterizer question more concretely: compacting backward samples helps a lot,
and the next speed work should target projection plus the remaining compact
backward cost, not a blind forward path rewrite.

## Compact-Backward 60-Second Quality Rerun

Command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_backward
```

Result:

```text
STAR-UVT, compact backward, 256 tubes:
  train PSNR        13.66211748123169
  heldout PSNR      12.700817108154297
  steps             294
  train loop        60.14397466601804s
  eval render-only  0.30119724897667766s
  heldout render    0.07833108300110325s

Direct splats, 2048, fast_mac:
  train PSNR        17.441619396209717
  heldout PSNR      10.724161148071289
  steps             2254
  train loop        60.021229375037365s
  eval render-only  0.24550599994836375s
  heldout render    0.059047749964520335s
```

Metal stats stayed stable: zero overflow and zero unstable tiles on all train
and heldout eval views. The cost is support growth: pair ratio rose to
`3.78-4.16`, and max tile count reached `123-135`.

Read: the compact-output patch is a real trainer improvement. It increased the
same-budget STAR step count from `175` to `294`, raised heldout PSNR from
`12.388733863830566` to `12.700817108154297`, and widened the direct-splat
heldout gap. But the trained render speed claim did not survive the extra
optimization: STAR render-only is now slower than direct splats in this rerun.
That means the next speed work is not a generic "make the rasterizer faster"
task. Forward raster is small in the step profile; the open work is projection,
compact backward/reduction, and support control so the trained model keeps the
UVT compactness that made the initialized and earlier cap-256 reports fast.

## Closed-Form Projection Patch

Change:

`project_world_tubes_pinhole(...)` no longer materializes a per-tube
`[2, 3] @ [3, 2]` projection Jacobian, diagonal world covariance, and batched
`torch.linalg.inv`. It computes the 2x2 screen covariance entries and inverse
in closed form. This preserves the same pinhole linearization but removes a lot
of small MPS tensor work from every STAR optimizer step.

Validation:

```text
old-formula equivalence:
  ma max abs       0.0
  q_uvt max abs    2.9802322387695312e-08
  depth0 max abs   0.0

non-identity camera equivalence:
  ma max abs          1.52587890625e-05
  q_uvt max abs       1.1920928955078125e-07
  depth0 max abs      0.0
  depth_beta max abs  0.0

pinhole_projection_smoke.py:
  max RGB error    5.960464477539063e-08
  overflow tiles   0

camera_spec_projection_smoke.py:
  max RGB error    5.960464477539063e-08
  overflow tiles   0
```

Timing probe:

```bash
python3 research_project/benchmarks/multicam_train_step_timing_probe.py \
  --device mps --steps 8 --warmup-steps 2 \
  --out-json research_project/benchmarks/results/multicam_train_step_timing_probe_mps_256_16f_projection_closedform.json
```

Result:

```text
STAR mean step        0.10214632287534187s
STAR projection       0.0023792186329956166s
STAR render           0.0040640363658894785s
STAR backward         0.08975485964037944s
Direct splat step     0.019570161501178518s

Backward microbreakdown:
  sample generation   0.015150916995480657s
  compact reductions  0.030503582034725696s
  projection VJP      0.002249667013529688s
```

Read: projection was the right next target. It dropped from about `70ms` in the
compact-backward profile to about `2.4ms`. The bottleneck is now the Metal
sample generation plus MPS reductions, not projection or forward raster.

## Closed-Form Projection 60-Second Quality Rerun

Command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.03 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_projection_closedform
```

Result:

```text
STAR-UVT, closed-form projection, 256 tubes:
  train PSNR        14.423624038696289
  heldout PSNR      12.778368949890137
  steps             410
  train loop        60.095966540975496s
  eval render-only  0.08381970797199756s
  heldout render    0.021688583015929908s

Direct splats, 2048, fast_mac:
  train PSNR        20.28785991668701
  heldout PSNR      10.700675964355469
  steps             3179
  train loop        60.00549187499564s
  eval render-only  0.36649591801688075s
  heldout render    0.07554204197367653s
```

Metal stats stayed clean: pair ratio `3.02-3.23`, max tile count `103-119`,
zero overflow, and zero unstable tiles.

Read: this restores the STAR speed claim against the paired direct-splat
baseline and slightly improves heldout PSNR over the compact-only rerun
(`12.778368949890137` versus `12.700817108154297`). It still does not beat the
V-JEPA F32 heldout reference `13.6248`. The next useful work is either quality
under this faster step path or reducing the remaining compact backward/reduction
cost.

## Bundled Compact Reduction

Change:

`tile_metal_autograd.py` now bundles the compact `ma`, `q_uvt`, opacity, and
color sample gradients into one 13-channel tensor before the MPS `index_add_`.
This keeps the compact-output contract from the Metal backward patch but avoids
four separate reductions over the same tube ids.

Timing probe:

```bash
python3 research_project/benchmarks/multicam_train_step_timing_probe.py \
  --device mps --steps 8 --warmup-steps 2 \
  --out-json research_project/benchmarks/results/multicam_train_step_timing_probe_mps_256_16f_projection_closedform_compact_bundle_reduce.json
```

Result:

```text
STAR mean step        0.06733408838044852s
STAR projection       0.001725447982607875s
STAR render           0.0024019792545004748s
STAR backward         0.05865076563350158s
Direct splat step     0.015081312507390976s

Backward microbreakdown:
  sample generation   0.010251250001601875s
  bundled reduction   0.008818959002383053s
  projection VJP      0.0027470840141177177s
```

Read: bundled reduction is a real hot-path win. Compared with the
closed-form-only profile, compact reduction dropped from
`0.030503582034725696s` to `0.008818959002383053s`, and STAR mean train step
dropped from `0.10214632287534187s` to `0.06733408838044852s`. Forward render
is now faster than direct splats in the timing probe (`0.0024019792545004748s`
versus `0.005343109376553912s`). The remaining speed blocker is STAR backward,
not projection or forward raster alone.

## LR Stability After Faster Backward

The faster reducer changed the full-budget optimizer stability boundary.

LR `0.03` full rerun:

```text
first NaN loss      step 180
last finite loss    step 170, loss 0.18437513709068298
final heldout PSNR  7.12491512298584
```

LR `0.02` looked useful in the 20-second bracket:

```text
LR 0.01, 20s: heldout PSNR 12.534915924072266, 270 STAR steps
LR 0.02, 20s: heldout PSNR 13.122428894042969, 302 STAR steps
```

But the 60-second escalation also collapsed:

```text
last finite loss    step 190, loss 0.12414514273405075
first NaN loss      step 200
final heldout PSNR  7.12491512298584
final tile stats    pair ratio 0.0, max tile count 0, active pairs 0
```

LR `0.015` was the midpoint check:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.015 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr0015
```

Result:

```text
STAR-UVT, bundled reducer, LR 0.015:
  train PSNR        16.549213409423828
  heldout PSNR      13.005823135375977
  steps             879
  train loop        60.035872791020665s
  eval render-only  0.03621062601450831s
  heldout render    0.010658667015377432s

Direct splats, 2048, fast_mac:
  train PSNR        20.19438934326172
  heldout PSNR      10.703802108764648
  steps             3195
  train loop        60.02963354199892s
  eval render-only  0.513283250038512s
```

Read: LR `0.015` is stable but not promoted. It completes slightly more STAR
steps and renders faster than LR `0.01`, but heldout PSNR is lower
(`13.005823135375977` versus `13.20147705078125`) and pair load is larger
(`1.80-2.30` pair ratio, max tile count `76-83`). Keep LR `0.01`.

Tile-load target `9000` at LR `0.01`:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 9000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target9000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result:

```text
STAR-UVT, bundled reducer, LR 0.01, target 9000:
  train PSNR        17.124666213989258
  heldout PSNR      12.860218048095703
  steps             790
  train loop        60.04627462499775s
  eval render-only  0.037336542969569564s
  heldout render    0.01084641698980704s

Direct splats, 2048, fast_mac:
  train PSNR        20.510637283325195
  heldout PSNR      10.703763961791992
  steps             3258
  train loop        60.00277195795206s
  eval render-only  0.3647003750083968s
```

Read: target `9000` is stable but not promoted. It raises STAR train PSNR, but
heldout PSNR falls below target `7000` (`12.860218048095703` versus
`13.20147705078125`) and STAR completes fewer steps. Metal stats are clean:
pair ratio `1.55-2.15`, max tile count `69-74`, zero overflow, and zero
unstable tiles. The next quality lever should not be simply relaxing projected
support.

The stable 60-second setting is LR `0.01`:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result:

```text
STAR-UVT, bundled reducer, LR 0.01:
  train PSNR        16.730055809020996
  heldout PSNR      13.20147705078125
  steps             849
  train loop        60.011288209003396s
  eval render-only  0.046138084086123854s
  heldout render    0.014686542039271444s

Direct splats, 2048, fast_mac:
  train PSNR        19.02044677734375
  heldout PSNR      10.722965240478516
  steps             2568
  train loop        60.00262387498515s
  eval render-only  0.29644233302678913s
  heldout render    0.08152529201470315s
```

Metal stats stayed clean: pair ratio `1.59-2.08`, max tile count `65-70`,
zero overflow, and zero unstable tiles.

Read: this is the current best legacy-pinhole local 256px/16-frame/60-second
artifact. STAR-UVT now beats the same-time direct-splat heldout PSNR by
`2.4785118103027344` dB and renders much faster. It still trails the V-JEPA F32
heldout reference by `0.4233229492187494` dB. The next quality work should
start from LR `0.01`; LR `0.02` and `0.03` are too hot under the current faster
backward path.

Dataset-lens projection diagnostic:

The DeepView goodset cameras are `opencv_fisheye`. The projection audit showed
that treating them as legacy pinhole cameras moves a projected grid by roughly
`16px` mean and `51px` max at 256px, so the multicam runner now has an opt-in
STAR-only camera path:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_dataset_lens_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result:

```text
STAR-UVT, dataset lens:
  train PSNR        16.58480167388916
  heldout PSNR      13.496740341186523
  steps             845
  train loop        60.01322920899838s
  eval render-only  0.0413991259993054s
  heldout render    0.01195199997164309s

Direct splats, 2048, fast_mac:
  train PSNR        19.735459327697754
  heldout PSNR      10.734033584594727
  steps             2789
  train loop        60.00262387498515s
  eval render-only  0.4126907510217279s
  heldout render    0.074937375029549s
```

Metal stats stayed clean: pair ratio `2.10-2.18`, max tile count `70-78`, zero
overflow, and zero unstable tiles. The result is not a replacement for the
legacy-pinhole V-JEPA comparison row, but it is a strong diagnostic: true
dataset-lens projection lifts STAR by another `0.29526329040527344` heldout
PSNR over the legacy-pinhole STAR row and remains about `10x` faster than the
paired direct-splat render-only eval. The current speed blocker is not forward
raster; the quality blocker includes camera-model parity, and training speed is
still dominated by STAR backward/sample reduction.

Lens-aware direct-splat baseline and first V-JEPA crossing:

The direct-splat renderer already supports `CameraSpec` lens projection through
Dynaworld's `render.camera_projection='camera_model'` path, but the multicam
comparison harness had been constructing only pinhole cameras for the direct
splat baseline. I added `--splat-camera-projection dataset_lens` so the
baseline can use the same DeepView `opencv_fisheye` lens contract as STAR while
leaving the legacy rows unchanged.

20-second gate:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_both_dataset_lens_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result: STAR heldout PSNR `13.600945472717285`, direct splats
`8.922689437866211`; STAR render-only eval `0.043396833061706275s`, direct
splats `0.5898302079876885s`.

60-second escalation:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result:

```text
STAR-UVT, dataset lens:
  train PSNR        16.240556716918945
  heldout PSNR      13.632997512817383
  steps             886
  train loop        60.01523466600338s
  eval render-only  0.04134708392666653s
  heldout render    0.012246291968040168s

Direct splats, 2048, fast_mac, dataset lens:
  train PSNR        17.265151500701904
  heldout PSNR      11.188531875610352
  steps             2438
  train loop        60.001816874952056s
  eval render-only  0.4395427079871297s
  heldout render    0.07924108300358057s
```

Metal stats stayed clean: pair ratio `2.21-2.33`, max tile count `68-74`, zero
overflow, and zero unstable tiles. This is the first local same-split STAR
heldout crossing over the V-JEPA F32 reference (`13.632997512817383` versus
`13.6248`). The margin is only `0.00819751281738238` dB and the run is still a
256px/16-frame local harness result, so promotion is not done. But the speed
thesis is clear in this row: STAR is about `10.6x` faster than lens-aware
direct splats on render-only eval and more than `2.4` PSNR better on heldout.

Seed-1 repeat:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed1_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result: STAR heldout PSNR `12.9697904586792`, direct splats
`11.243672370910645`; STAR render-only eval `0.04029629105934873s`, direct
splats `0.5776280419668183s`; STAR steps `790`, direct splat steps `2532`.
Metal stats stayed clean: pair ratio `2.13-2.37`, max tile count `69-73`, zero
overflow, and zero unstable tiles. This repeat keeps the speed story intact but
does not reproduce V-JEPA parity. Treat the seed-0 crossing as a promising
measured artifact, not a robust conclusion.

Grid-init seed-1 repeat:

I added `--uvt-init-sampling random|grid` to separate initialization pixel
coverage from the training seed. The default `random` mode keeps the old
behavior. The new `grid` mode samples deterministic image-grid pixels from the
chosen initialization views.

Smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --seed 1 \
  --uvt-tubes 8 --uvt-lr 0.01 \
  --uvt-render-backend dense --uvt-camera-projection dataset_lens \
  --uvt-init-sampling grid \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --splat-count 8 --splat-renderer dense --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_grid_init_smoke_16_2f_1s
```

The smoke passed and wrote `init_sampling: grid` into the STAR report.

60-second repeat:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-sampling grid \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed1_gridinit_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result:

```text
STAR-UVT, dataset lens, seed 1, grid init:
  train PSNR        16.25625467300415
  heldout PSNR      13.179410934448242
  steps             932
  train loop        60.037565333012026s
  eval render-only  0.035952959035057575s
  heldout render    0.01105145801557228s

Direct splats, 2048, fast_mac, dataset lens:
  train PSNR        17.340192794799805
  heldout PSNR      11.206615447998047
  steps             2405
  train loop        60.01556845800951s
  eval render-only  0.5049242499517277s
  heldout render    0.08374808396911249s
```

Metal stats stayed clean: pair ratio `2.22-2.38`, max tile count `71-75`, zero
overflow, and zero unstable tiles.

Read: deterministic grid init improves seed 1 over random init
(`13.179410934448242` versus `12.9697904586792`) and keeps STAR much faster
than direct splats. It still misses the V-JEPA F32 heldout reference
`13.6248`, so the quality blocker is not just random pixel coverage at init.
The next quality gate should test a deterministic train-view/window schedule or
a stronger multi-view/motion-aware initialization before touching the
rasterizer.

All-train grid-init escalation:

20-second gate:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_both_dataset_lens_seed1_alltrain_gridinit_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result: STAR heldout PSNR `13.527872085571289`, direct splats
`9.138171195983887`; STAR render-only eval `0.04275804205099121s`, direct
splats `0.3338368329568766s`. Metal stats were clean: pair ratio `2.64-2.77`,
max tile count `85-93`, zero overflow, and zero unstable tiles.

60-second escalation:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed1_alltrain_gridinit_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result:

```text
STAR-UVT, dataset lens, seed 1, all-train grid init:
  train PSNR        15.971568584442139
  heldout PSNR      13.52819538116455
  steps             851
  train loop        60.06032329198206s
  eval render-only  0.11616870801663026s
  heldout render    0.034273458004463464s

Direct splats, 2048, fast_mac, dataset lens:
  train PSNR        16.335904121398926
  heldout PSNR      11.074682235717773
  steps             2092
  train loop        60.0236113750143s
  eval render-only  0.5125780410016887s
  heldout render    0.09727545798523352s
```

Metal stats stayed clean: pair ratio `2.42-2.52`, max tile count `78-81`, zero
overflow, and zero unstable tiles.

Read: all-train grid init is a real quality improvement for seed 1. It recovers
about `0.56` dB over seed-1 random first-view init and about `0.35` dB over
seed-1 first-view grid init, while staying faster than direct splats. It still
misses V-JEPA `13.6248` by about `0.10` dB and renders slower than first-view
grid, so promotion is still blocked. The next quality step should keep
multi-view coverage but reduce stochastic training variance or support growth.

Deterministic train-schedule probe:

I added `--uvt-train-schedule random|cycle`. The default `random` mode keeps
the old behavior. The new `cycle` mode deterministically cycles train views and
temporal-window starts for STAR-UVT.

Smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --seed 1 \
  --uvt-tubes 8 --uvt-lr 0.01 \
  --uvt-render-backend dense --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --uvt-train-schedule cycle \
  --splat-count 8 --splat-renderer dense --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_cycle_schedule_smoke_16_2f_1s
```

The smoke passed and wrote `train_schedule: cycle` into the STAR report.

20-second gate:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid \
  --uvt-train-schedule cycle \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_both_dataset_lens_seed1_alltrain_gridinit_cycle_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result: STAR heldout PSNR `13.28015422821045`, direct splats
`9.227516174316406`; STAR render-only eval `0.038111125002615154s`, direct
splats `0.3976892919745296s`; STAR steps `312`, direct splat steps `960`.
Metal stats stayed clean: pair ratio `2.47-2.65`, max tile count `74-79`, zero
overflow, and zero unstable tiles.

Read: this is negative. The same all-train grid init with random train samples
reached heldout PSNR `13.527872085571289` at the 20-second gate. Do not
escalate deterministic cycle scheduling to 60 seconds without a better schedule
design.

All-train grid LR `0.015` probe:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid \
  --uvt-tubes 256 --uvt-lr 0.015 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_both_dataset_lens_seed1_alltrain_gridinit_lr0015_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result: STAR heldout PSNR `13.287956237792969`, direct splats
`8.6444730758667`; STAR render-only eval `0.04351187701104209s`, direct splats
`0.6308000839781016s`; STAR steps `310`, direct splat steps `717`. Metal stats
stayed clean: pair ratio `2.64-2.74`, max tile count `77-82`, zero overflow,
and zero unstable tiles.

Read: this is negative. LR `0.015` is worse than the LR `0.01` all-train grid
20-second gate (`13.287956237792969` versus `13.527872085571289`), so do not
escalate it to 60 seconds.

All-train grid tile-load target `5000` probe:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 20 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 5000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_both_dataset_lens_seed1_alltrain_gridinit_temporal_window4_tileload0001_target5000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result: STAR heldout PSNR `13.459638595581055`, direct splats
`9.227531433105469`; STAR render-only eval `0.0383758339448832s`, direct
splats `0.28163870907155797s`; STAR steps `308`, direct splat steps `960`.
Metal stats stayed clean and more compact than target `7000`: pair ratio
`2.26-2.36`, max tile count `74-75`, zero overflow, and zero unstable tiles.

Read: target `5000` improves compactness and keeps render fast, but it loses
heldout quality versus target `7000` at the same 20-second all-train grid gate
(`13.459638595581055` versus `13.527872085571289`). Keep target `7000` as the
quality setting.

## Time-Distributed All-Frames Init

Change:

```text
--uvt-init-frames first | all
```

The previous multi-view initializer could sample from all train cameras, but it
still sampled colors from frame 0 while initializing every tube at `t0 = 0`,
the centered sequence time. `--uvt-init-frames all` splits the tube budget
across train-view/frame groups, samples colors from the source frame, and sets
each tube's `t0` to `frame - 0.5 * (frames - 1)`.

Smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --seed 1 \
  --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_time_init_smoke_16_2f_1s
```

The smoke passed and wrote `init_frames: all` into the report.

Seed-1 20-second gate:

```text
STAR-UVT:      heldout 13.768306732177734, train 15.55235481262207,
               steps 346, render-only 0.04166583297774196s
Direct splats: heldout 8.622618675231934, train 9.99921178817749,
               steps 710, render-only 0.3382836260134354s
```

Seed-1 30-second gate:

```text
STAR-UVT:      heldout 13.726262092590332, train 16.17089080810547,
               steps 530, render-only 0.041071167041081935s
Direct splats: heldout 9.915854454040527, train 13.178874015808105,
               steps 1287, render-only 0.4364799159229733s
```

Seed-1 60-second escalation:

```text
STAR-UVT:      heldout 13.564573287963867, train 16.77016544342041,
               steps 785, render-only 0.08123612502822652s
Direct splats: heldout 11.048260688781738, train 16.125881671905518,
               steps 2045, render-only 0.42233533307444304s
```

Seed-0 20-second repeat:

```text
STAR-UVT:      heldout 13.769630432128906, train 15.741607666015625,
               steps 365, render-only 0.03941354202106595s
Direct splats: heldout 8.91901969909668, train 10.778040409088135,
               steps 838, render-only 0.29203037498518825s
```

Seed-2 20-second repeat:

```text
STAR-UVT:      heldout 13.764396667480469, train 15.683393478393555,
               steps 333, render-only 0.041590416978579015s
Direct splats: heldout 8.697443962097168, train 10.178499221801758,
               steps 732, render-only 0.46188579098088667s
```

Seed-0 10-second bracket:

```text
STAR-UVT:      heldout 12.681236267089844, train 13.3593430519104,
               steps 176, render-only 0.04378599900519475s
Direct splats: heldout 7.715583801269531, train 8.220246315002441,
               steps 416, render-only 0.28413499996531755s
```

Seed-0 15-second bracket:

```text
STAR-UVT:      heldout 13.669918060302734, train 14.98948049545288,
               steps 261, render-only 0.04481595807010308s
Direct splats: heldout 8.104644775390625, train 8.896464824676514,
               steps 530, render-only 0.38280241598840803s
```

Seed-0 30-second repeat:

```text
STAR-UVT:      heldout 13.600011825561523, train 16.258267402648926,
               steps 509, render-only 0.04163579299347475s
Direct splats: heldout 9.411006927490234, train 12.076716423034668,
               steps 1057, render-only 0.2929964159266092s
```

Read: this is the strongest quality mechanism found in this pass. It changes
the result from "seed-1 all-train grid still misses V-JEPA" to "seed-1
all-frames crosses V-JEPA at 20 and 30 seconds." With the seed-0 20-second
and seed-2 20-second repeats, the 20-second all-frames recipe now crosses the
V-JEPA F32 reference `13.6248` on all three tested seeds. It is not a blanket
longer-budget promotion: seed 0 at 30 seconds misses V-JEPA by about `0.0248`
dB, and seed 1 at 60 seconds falls below the 20/30-second results. The next
comparison should treat heldout-camera early stopping as a real STAR-UVT
requirement. The seed-0 10/15/20/30-second curve says the current sweet spot is
roughly 15-20 seconds, with 20 seconds still the best tested point. This result
does not point to a forward rasterizer rewrite; the all-frames rows keep STAR
render-only eval substantially faster than direct splats.

## Checkpoint-Curve Instrumentation

Change:

```text
--uvt-checkpoint-every-steps N
```

When `N > 0`, the STAR trainer stores small CPU snapshots of the worldtube
state every `N` steps and once at the final step if needed. After training, the
benchmark evaluates those snapshots and writes:

```text
star_uvt.checkpoint_curve.rows
star_uvt.checkpoint_curve.best_by_heldout_psnr
```

`--skip-splats` is also wired for STAR-only checkpoint diagnostics. When set,
the direct dynamic splat baseline is not trained and the report writes:

```text
free_dynamic_splats: null
```

Smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --seed 3 \
  --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --uvt-checkpoint-every-steps 1 \
  --splat-count 8 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_checkpoint_curve_smoke_16_2f_1s
```

Result: the smoke passed and wrote two checkpoint rows. Read: this is now the
right tool for diagnosing the 20-second peak inside longer STAR runs. It is a
research diagnostic; heldout-selected checkpoint numbers should be labeled as
selected and should not replace final-checkpoint comparisons when claiming an
unbiased heldout result.

Skip-splats smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --seed 4 \
  --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --uvt-checkpoint-every-steps 1 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_skip_splats_smoke_16_2f_1s
```

Result: the smoke passed and wrote `free_dynamic_splats: null` plus STAR
checkpoint rows. Read: this is the right mode for long STAR-only schedule
diagnostics once the paired direct-splat baseline has already been measured.

Seed-0 30-second diagnostic:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 30 --device mps --seed 0 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 50 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_30s_both_dataset_lens_seed0_alltrain_gridinit_allframes_checkpoint50_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result:

```text
STAR final:  step 489, train 16.346084594726562,
             heldout 13.518027305603027, render-only 0.04052966699237004s
STAR best:   step 300, elapsed 18.632994499988854s,
             train 15.358994007110596,
             heldout 13.730653762817383, render-only 0.0463867480866611s
Direct splats final:
             heldout 9.841208457946777, render-only 0.310775084013585s
```

Read: the checkpoint curve confirms the earlier separate 10/15/20/30-second
bracket. The STAR model is best around the 18-20 second region, then train PSNR
keeps climbing while heldout PSNR drops. This is exactly the failure mode a
mid-run selector or schedule change should address.

Seed-1 60-second STAR-only diagnostic:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed1_alltrain_gridinit_allframes_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result:

```text
STAR final:  step 1019, train 17.187273025512695,
             heldout 13.354101181030273, render-only 0.04292666696710512s
STAR best:   step 300, elapsed 17.160626166965812s,
             train 15.61048173904419,
             heldout 13.75400447845459, render-only 0.036524292023386806s
Checkpoint rows:
             step 100 heldout 10.69461441040039
             step 200 heldout 13.170500755310059
             step 300 heldout 13.75400447845459
             step 400 heldout 13.74360466003418
             step 500 heldout 13.627070426940918
             step 600 heldout 13.620699882507324
             step 700 heldout 13.605888366699219
             step 800 heldout 13.470739364624023
             step 900 heldout 13.400154113769531
             step 1000 heldout 13.319355010986328
```

Metal stats stayed clean: pair ratio `2.61-3.00`, max tile count `71-80`, zero
overflow, and zero unstable tiles. Read: seed 1 repeats the same shape as the
seed-0 checkpoint curve. The useful checkpoint is around step 300 / 17-19
seconds; extending fixed-budget training to 60 seconds increases train PSNR but
hurts heldout-camera PSNR. The next work should be a schedule or selector that
preserves this mid-run generalization point, not a forward-raster rewrite.

Seed-2 40-second STAR-only diagnostic:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 40 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_40s_star_only_dataset_lens_seed2_alltrain_gridinit_allframes_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
```

Result:

```text
STAR final:  step 629, train 16.31698513031006,
             heldout 13.631431579589844, render-only 0.0434984170133248s
STAR best:   step 500, elapsed 32.050383166992106s,
             train 16.440028190612793,
             heldout 13.988276481628418, render-only 0.037308208004105836s
Checkpoint rows:
             step 100 heldout 10.585294723510742
             step 200 heldout 13.141227722167969
             step 300 heldout 13.574493408203125
             step 400 heldout 13.837061882019043
             step 500 heldout 13.988276481628418
             step 600 heldout 13.693096160888672
             step 629 heldout 13.631431579589844
```

Metal stats stayed clean: pair ratio `3.14-3.41`, max tile count `75-84`, zero
overflow, and zero unstable tiles. Read: seed 2 confirms that long fixed-budget
training can pass through a much better heldout checkpoint and then decay, but
the peak is later than seed 0 and seed 1. A hard step-300 cutoff would throw
away the best seed-2 model. The next schedule should monitor a validation curve
or use a smoother training objective, not assume one global stop time.

## LR-Decay Schedule Bracket

Change:

```text
--uvt-lr-decay-step S
--uvt-lr-decay-factor F
```

Default behavior is unchanged. When `S > 0`, STAR training uses the base
`--uvt-lr` until step `S`, then multiplies it by `F`. Logs include the active
`lr`.

CPU smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 2 --max-steps 10 \
  --device cpu --seed 6 \
  --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --uvt-lr-decay-step 5 --uvt-lr-decay-factor 0.5 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_lr_decay_log_smoke_16_2f_2s
```

Result: the smoke wrote LR `0.03` at step 1 and LR `0.015` at step 10,
confirming the schedule applies after the threshold.

Seed-1 decay factor `0.2`:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 300 --uvt-lr-decay-factor 0.2 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed1_alltrain_gridinit_allframes_lrdecay300x02_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final:  step 784, train 16.584238052368164,
             heldout 13.643348693847656, render-only 0.11957950098440051s
STAR best:   step 600, elapsed 35.903643500001635,
             train 16.29133701324463,
             heldout 13.743600845336914, render-only 0.044522624055389315s
```

Seed-1 decay factor `0.05`:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 300 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed1_alltrain_gridinit_allframes_lrdecay300x005_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final:  step 1003, train 16.25966167449951,
             heldout 13.692363739013672, render-only 0.04503662494244054s
STAR best:   step 500, elapsed 29.56072895799298,
             train 15.92283582687378,
             heldout 13.71772575378418, render-only 0.03478599904337898s
```

Read: LR decay after step 300 is a useful but incomplete schedule lever.
Both decayed runs improve the seed-1 final checkpoint over the no-decay final
heldout PSNR `13.354101181030273`, and factor `0.05` keeps render speed clean.
However, neither schedule recovers the best no-decay selected checkpoint
`13.75400447845459`, and seed 2 already showed that the peak can happen much
later than step 300. The next schedule gate should be validation-shaped or
adaptive rather than another hard-coded single decay step.

Seed-2 decay after the observed later peak:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final:  step 756, train 17.018577575683594,
             heldout 13.81359577178955, render-only 0.08080591692123562s
STAR best:   step 500, elapsed 31.662656374974176,
             train 16.483168601989746,
             heldout 13.909360885620117, render-only 0.06991625000955537s
Checkpoint rows:
             step 300 heldout 13.67954158782959
             step 400 heldout 13.859981536865234
             step 500 heldout 13.909360885620117
             step 600 heldout 13.850353240966797
             step 700 heldout 13.831350326538086
             step 756 heldout 13.81359577178955
```

Metal stats stayed clean: pair ratio `3.00-3.13`, max tile count `75-80`, zero
overflow, and zero unstable tiles. Read: decaying after the per-seed observed
peak improves the final seed-2 model versus the earlier 40-second no-decay
final, but it still trails the selected step-500 checkpoint. This reinforces
the same conclusion as seed 1: LR decay is useful, but the clean rule is
validation-shaped selection or an adaptive schedule, not a fixed global decay
step.

Paired seed-2 schedule comparison:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final:
  step             996
  train PSNR       17.022375106811523
  heldout PSNR     13.84060287475586
  render-only      0.043013749993406236s

STAR selected best:
  step             800
  elapsed          48.05842208303511s
  heldout PSNR     13.873014450073242
  render-only      0.032829875010065734s

Direct splats:
  steps            2326
  train PSNR       17.103739738464355
  heldout PSNR     11.156550407409668
  render-only      0.3524511669529602s
```

Metal stats stayed clean: pair ratio `2.88-3.04`, max tile count `76-81`, zero
overflow, and zero unstable tiles. Read: this is the clean paired schedule row.
STAR beats direct dynamic splats by about `2.68` heldout PSNR and renders about
`8.2x` faster at eval. It also clears the V-JEPA F32 heldout reference
`13.6248` at the final checkpoint. The caveat remains that this is a local
harness row and selected-checkpoint diagnostics still show the schedule is not
fully preserving the best possible heldout point.

## Selected-Checkpoint Report Path

Change:

```text
--uvt-select-checkpoint best_heldout
```

When enabled, the benchmark loads the checkpoint named by
`checkpoint_curve.best_by_heldout_psnr`, evaluates it into a separate
`star_uvt_selected` report section, writes selected train/heldout media, and
then restores the final model state before writing the normal final-checkpoint
report. This is intentionally labeled:

```text
uses_heldout_for_selection: true
```

Smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --seed 7 \
  --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint best_heldout \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_selected_checkpoint_smoke_16_2f_1s
```

Result: the smoke passed. The report wrote `star_uvt_selected` with selected
step `2`, and the output directory contains:

```text
star_uvt_selected_train_view0_preview.png
star_uvt_selected_train_view0_side_by_side.mp4
star_uvt_selected_heldout_view0_preview.png
star_uvt_selected_heldout_view0_side_by_side.mp4
```

Read: future long runs can now emit final-checkpoint, checkpoint-curve, and
selected-checkpoint artifacts in one report. The selected section is useful for
diagnosing schedules, but should remain clearly separated from unbiased test
claims because it uses heldout-camera PSNR for selection.

Paired seed-1 selected-checkpoint MPS artifact:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 300 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint best_heldout \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed1_alltrain_gridinit_allframes_lrdecay300x005_selectbest_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final:
  step             1031
  train PSNR       16.322839736938477
  heldout PSNR     13.758882522583008
  render-only      0.04524075100198388s

STAR selected:
  step             300
  elapsed          18.1000763750053s
  train PSNR       15.563595294952393
  heldout PSNR     13.818532943725586
  render-only      0.03815650095930323s

Direct splats:
  steps            2271
  train PSNR       16.899147033691406
  heldout PSNR     11.15761947631836
  render-only      0.3847669999813661s
```

Final and selected STAR Metal stats stayed clean: zero overflow and zero
unstable tiles. Read: this confirms the selected-checkpoint report path on the
real MPS recipe and gives a paired seed-1 artifact with final, selected, and
direct-splat media. It also shows MPS run-to-run variation versus the earlier
STAR-only schedule bracket, so exact selected rows should be tied to the saved
report path rather than generalized from memory.

Paired seed-2 selected-checkpoint MPS artifact:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint best_heldout \
  --splat-renderer fast_mac --splat-count 2048 --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_selectbest_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final:
  step             1012
  train PSNR       17.101717948913574
  heldout PSNR     13.873907089233398
  render-only      0.042239958012942225s

STAR selected:
  step             600
  elapsed          36.40041800000472s
  train PSNR       16.882763862609863
  heldout PSNR     13.915654182434082
  render-only      0.034174208994954824s

Direct splats:
  steps            2178
  train PSNR       16.65969181060791
  heldout PSNR     11.085673332214355
  render-only      0.40090283303288743s
```

Final and selected STAR Metal stats stayed clean: pair ratio `2.74-3.07`, max
tile count `74-78`, zero overflow, and zero unstable tiles. Read: this row
answers the immediate speed concern. Current STAR-UVT selected render-only is
not slower than direct dynamic splats; it is about `11.7x` faster in the saved
report. Earlier "STAR render is slower" rows were trained tile-load/support
pathologies that the `tile_t=1`, cap-256, lens-aware, all-frames init recipe
has already moved past. The next blocker is schedule selection and unbiased
validation, with rasterizer work limited to hardening, lower variance timing,
and future production integration.

## Train-Plateau Non-Heldout Selector

Change:

```text
--uvt-select-checkpoint first_train_psnr_plateau
--uvt-select-train-psnr-plateau-delta 0.5
```

This selects the first checkpoint whose train PSNR gain from the previous
checkpoint is at or below the configured delta. It writes the normal
`star_uvt_selected` report section, but labels it:

```text
uses_heldout_for_selection: false
selection_metric: eval_psnr_gain_from_previous_checkpoint
```

CPU smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --seed 7 \
  --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_train_psnr_plateau \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_plateau_smoke_16_2f_1s
```

The smoke passed and wrote selected train/heldout preview PNGs and MP4s.

Seed-2 MPS STAR-only diagnostic:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_plateau \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_trainplateau050_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final:
  step             955
  train PSNR       17.085532188415527
  heldout PSNR     13.786846160888672
  render-only      0.05564970901468769s

STAR train-plateau selected:
  step             400
  elapsed          26.757762582972646s
  train PSNR       16.005226135253906
  heldout PSNR     13.83452320098877
  render-only      0.03962891804985702s
  selected gain    0.457211971282959

Heldout-best checkpoint in same curve:
  step             500
  train PSNR       16.497983932495117
  heldout PSNR     13.94494915008545
```

Final and selected STAR Metal stats stayed clean: zero overflow and zero
unstable tiles. Read: the train-plateau selector is a real non-heldout
improvement over final-checkpoint reporting and clears the V-JEPA F32 heldout
reference on seed 2. It is not the final answer, because it leaves about
`0.11` dB versus the heldout-best checkpoint in the same curve.

Seed-1 repeat:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 300 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_plateau \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed1_alltrain_gridinit_allframes_lrdecay300x005_trainplateau050_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final:
  step             1040
  train PSNR       16.341201782226562
  heldout PSNR     13.790154457092285
  render-only      0.0415437490446493s

STAR train-plateau selected:
  step             400
  elapsed          24.412976625026204s
  train PSNR       15.838202476501465
  heldout PSNR     13.771395683288574
  render-only      0.051538082014303654s
  selected gain    0.40828561782836914

Heldout-best checkpoint in same curve:
  step             700
  train PSNR       16.135772705078125
  heldout PSNR     13.826448440551758
```

Final and selected STAR Metal stats stayed clean: zero overflow and zero
unstable tiles. Read: the train-plateau rule clears V-JEPA on seed 1 too, but
it is slightly worse than the final checkpoint and misses the heldout-best
checkpoint by about `0.055` dB. The rule is therefore a useful non-heldout
baseline selector, not the final schedule solution.

## Train-Camera Balance Selector Diagnostic

Change:

```text
--uvt-select-checkpoint best_min_train_view_psnr
```

Checkpoint rows now include:

```text
train_view_eval_psnr
train_min_view_eval_psnr
train_view_eval_psnr_gap
```

CPU smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --seed 7 \
  --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint best_min_train_view_psnr \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_view_selector_smoke_16_2f_1s
```

The smoke passed and wrote per-train-view PSNR fields plus selected media.

Seed-2 MPS STAR-only diagnostic:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint best_min_train_view_psnr \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_mintrainview_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final / selected:
  step             1000
  train PSNR       17.001721382141113
  min train-view   16.572290420532227
  heldout PSNR     13.835639953613281
  render-only      0.04325254098512232s

Heldout-best checkpoint in same curve:
  step             600
  train PSNR       16.79953670501709
  min train-view   16.24464988708496
  heldout PSNR     13.904929161071777
```

Read: rejected as a schedule rule. The minimum train-view PSNR kept improving
late, so the selector picked the final checkpoint and missed the heldout peak
by about `0.069` dB. The per-view fields are useful diagnostic evidence, but
plain `best_min_train_view_psnr` should not be the next selector to tune.

## True Train-Camera Dev Split

Change:

```text
--uvt-optimizer-train-views first_only
--uvt-select-checkpoint best_train_view_psnr
--uvt-select-train-view-index 1
```

This restricts STAR optimizer samples to train camera index `0` and selects
checkpoints by train camera index `1`, leaving the real heldout camera untouched.

CPU smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --seed 7 \
  --uvt-camera-projection dataset_lens \
  --uvt-init-views first --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --uvt-optimizer-train-views first_only \
  --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint best_train_view_psnr \
  --uvt-select-train-view-index 1 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_view_dev_selector_smoke_16_2f_1s
```

The smoke passed and wrote `optimizer_train_view_indices: [0]`,
`selection_metric: train_view_1_eval_psnr`, and selected media.

Seed-2 MPS STAR-only diagnostic:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views first --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-optimizer-train-views first_only \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint best_train_view_psnr \
  --uvt-select-train-view-index 1 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed2_firsttrain_devview1_select_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final / selected by dev train view:
  step             972
  train PSNR       15.337390899658203
  dev-view PSNR    10.769975662231445
  heldout PSNR     12.647516250610352
  render-only      0.03078908397583291s

Heldout-best checkpoint in same curve:
  step             800
  train PSNR       15.301321506500244
  heldout PSNR     12.661633491516113
```

Read: rejected. The selector contract is clean, but optimizing only one of two
train cameras loses too much multiview signal. It falls below the V-JEPA F32
heldout reference and below the all-train STAR recipe by more than a dB. Future
unbiased selection should use a lighter validation subset or smoothed train
curve, not full camera removal.

## Seed-0 LR-Decay Schedule Repeat

Command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 0 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 300 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint best_heldout \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed0_alltrain_gridinit_allframes_lrdecay300x005_selectbest_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Result:

```text
STAR final:
  step             970
  train PSNR       16.02366304397583
  heldout PSNR     13.81613826751709
  render-only      0.040288584015797824s

STAR heldout-selected:
  step             600
  elapsed          37.35523024998838s
  train PSNR       15.836221694946289
  heldout PSNR     13.87098217010498
  render-only      0.04048433306161314s

Heldout-best row:
  step             600
  train-view PSNR  [16.141557693481445, 15.530885696411133]
  heldout PSNR     13.87098217010498

Best min-train-view row:
  step             700
  min train-view   15.730052947998047
  heldout PSNR     13.86395263671875
```

Final Metal stats were clean: pair ratio `3.626-3.862`, max tile count
`94-102`, zero overflow, and zero unstable tiles. Selected Metal stats were
also clean: pair ratio `3.749-4.026`, max tile count `94-102`, zero overflow,
and zero unstable tiles. Selected train and heldout preview PNGs plus
side-by-side MP4s were written.

Read: this seed-0 repeat confirms the LR-decay schedule direction. The recent
tuned rows now have V-JEPA-crossing final checkpoints on seeds 0, 1, and 2,
and the render-only time remains about `0.04s`. It does not solve unbiased
checkpoint selection because the selected row uses the heldout camera, and it
does not add a paired direct-splat row because this run used `--skip-splats`.

## Train-Plateau Patience-2 Selector

Change:

```text
--uvt-select-train-psnr-plateau-patience 2
```

This keeps the existing `first_train_psnr_plateau` selector but requires two
consecutive checkpoint-to-checkpoint train-PSNR gains at or below the plateau
delta before selecting. Default patience remains `1`, so the old behavior is
unchanged unless the new flag is set.

CPU smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_train_psnr_plateau \
  --uvt-select-train-psnr-plateau-patience 2 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_plateau_patience2_smoke_16_2f_1s
```

The smoke passed and wrote `select_train_psnr_plateau_patience: 2`,
`uses_heldout_for_selection: false`, and selected media.

Seed-2 MPS command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_plateau \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-plateau-patience 2 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_trainplateau050_patience2_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Seed-2 result:

```text
STAR final:
  step             990
  train PSNR       17.03642463684082
  heldout PSNR     13.805965423583984
  render-only      0.04240429098717868s

STAR patience-2 selected:
  step             500
  elapsed          30.836721166968346s
  train PSNR       16.383162021636963
  heldout PSNR     13.84100341796875
  render-only      0.03880170703632757s
  uses heldout     false
  selected gain    0.4284195899963379

Heldout-best checkpoint in same curve:
  step             600
  train PSNR       16.8155574798584
  heldout PSNR     13.855948448181152
```

Selected Metal stats stayed clean: pair ratio `3.263-3.385`, max tile count
`80-88`, zero overflow, and zero unstable tiles.

Seed-1 MPS command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 300 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_plateau \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-plateau-patience 2 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed1_alltrain_gridinit_allframes_lrdecay300x005_trainplateau050_patience2_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Seed-1 result:

```text
STAR final:
  step             1016
  train PSNR       16.223440170288086
  heldout PSNR     13.688732147216797
  render-only      0.045003665960393846s

STAR patience-2 selected:
  step             500
  elapsed          28.439446334028617s
  train PSNR       15.940573692321777
  heldout PSNR     13.704198837280273
  render-only      0.04046700103208423s
  uses heldout     false
  selected gain    0.12256145477294922

Heldout-best checkpoint in same curve:
  step             400
  train PSNR       15.818012237548828
  heldout PSNR     13.726997375488281
```

Selected Metal stats stayed clean: pair ratio `3.602-3.937`, max tile count
`91-103`, zero overflow, and zero unstable tiles.

Read: patience `2` is a candidate, not the answer. It moved seed 2 closer to
the heldout-best checkpoint and selected a better checkpoint than final without
heldout-camera selection. But on seed 1 it overshot the heldout-best checkpoint
by one interval, and the whole seed-1 repeat was lower than the previous
patience-1 seed-1 run. Both selected rows clear V-JEPA, but the next selector
should use a smoother train curve or a cheap validation-rendered subset rather
than only increasing plateau patience.

## Train-Gain-Drop Selector

Retrospective read over the saved checkpoint curves showed a better train-only
rule than raw plateau patience: after train PSNR gain has entered the low-gain
region, select the previous checkpoint when the next train-PSNR gain drops.
This preserves the no-heldout-selection contract while trying to catch the
point just before diminishing returns dominate.

Change:

```text
--uvt-select-checkpoint first_train_psnr_gain_drop
--uvt-select-train-psnr-plateau-delta 0.5
--uvt-select-train-psnr-gain-drop 0.02
```

CPU smoke:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_gain_drop_smoke_16_2f_1s
```

The smoke passed and wrote `selector: first_train_psnr_gain_drop`,
`selection_metric: eval_psnr_gain_drop_after_low_gain`, and
`uses_heldout_for_selection: false`.

Seed-2 MPS command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_traingain_drop002_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Seed-2 result:

```text
STAR final:
  step             884
  train PSNR       17.028188705444336
  heldout PSNR     13.872002601623535
  render-only      0.054093708051368594s

STAR gain-drop selected:
  step             500
  elapsed          32.4258956250269s
  train PSNR       16.531704902648926
  heldout PSNR     13.904694557189941
  render-only      0.039125834009610116s
  uses heldout     false
  selected gain    0.5023479461669922
  next gain        0.3444557189941406
  gain drop        0.15789222717285156

Heldout-best checkpoint in same curve:
  step             500
  heldout PSNR     13.904694557189941
```

Selected Metal stats stayed clean: pair ratio `3.118-3.270`, max tile count
`79-86`, zero overflow, and zero unstable tiles.

Seed-1 MPS command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 300 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed1_alltrain_gridinit_allframes_lrdecay300x005_traingain_drop002_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Seed-1 result:

```text
STAR final:
  step             1005
  train PSNR       16.174781799316406
  heldout PSNR     13.683605194091797
  render-only      0.06090208300156519s

STAR gain-drop selected:
  step             400
  elapsed          22.553510166995693s
  train PSNR       15.725313186645508
  heldout PSNR     13.721198081970215
  render-only      0.04510591697180644s
  uses heldout     false
  selected gain    0.32379674911499023
  next gain        0.09837532043457031
  gain drop        0.22542142868041992

Heldout-best checkpoint in same curve:
  step             500
  heldout PSNR     13.735674858093262
```

Selected Metal stats stayed clean: pair ratio `3.670-4.036`, max tile count
`92-104`, zero overflow, and zero unstable tiles.

Read: this is the best non-heldout selector candidate so far. It exactly
selected the heldout-best row on the new seed-2 curve, improved over the final
checkpoint on seed 1, and missed seed 1's heldout-best row by only about
`0.0145` dB. Both selected rows clear the V-JEPA F32 reference and keep the
render-speed story intact. The follow-up below adds seed-0 confirmation and a
paired direct-splat report.

Seed-0 confirmation command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 0 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 300 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_star_only_dataset_lens_seed0_alltrain_gridinit_allframes_lrdecay300x005_traingain_drop002_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Seed-0 result:

```text
STAR final:
  step             924
  train PSNR       16.046878337860107
  heldout PSNR     13.903148651123047
  render-only      0.043030957982409745s

STAR gain-drop selected:
  step             400
  elapsed          26.377487833960913s
  train PSNR       15.63296127319336
  heldout PSNR     13.901209831237793
  render-only      0.04819570906693116s
  uses heldout     false
  selected gain    0.3193788528442383
  next gain        0.13135051727294922
  gain drop        0.18802833557128906

Heldout-best checkpoint in same curve:
  step             500
  heldout PSNR     13.915482521057129
```

Paired seed-2 direct-splat command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --splat-renderer fast_mac --splat-count 2048 \
  --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_traingain_drop002_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Paired seed-2 result:

```text
STAR final:
  step             870
  train PSNR       16.955312728881836
  heldout PSNR     13.835267066955566
  render-only      0.04184024897404015s

STAR gain-drop selected:
  step             400
  elapsed          28.700299500022084s
  train PSNR       16.00412082672119
  heldout PSNR     13.888997077941895
  render-only      0.04640516696963459s
  uses heldout     false

Heldout-best checkpoint in same curve:
  step             500
  heldout PSNR     13.902454376220703

Direct dynamic splats:
  steps            2508
  train PSNR       17.396635055541992
  heldout PSNR     11.190529823303223
  render-only      0.9052186670596711s
```

Read update: seed 0 confirms the selector clears V-JEPA without heldout-camera
selection, and the paired seed-2 report confirms the speed/quality comparison
against direct dynamic splats. The paired selected STAR row is about `2.70` dB
better on heldout PSNR and about `19.5x` faster by synchronized render-only
timing. It missed the paired run's heldout-best checkpoint by only about
`0.0135` dB. This shifts the next work away from a forward-rasterizer rewrite:
the current blocker is whether to freeze this train-only selector or collect
one more paired repeat before scaling.

Paired seed-1 direct-splat repeat command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps --seed 1 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 300 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --splat-renderer fast_mac --splat-count 2048 \
  --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed1_alltrain_gridinit_allframes_lrdecay300x005_traingain_drop002_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

Paired seed-1 result:

```text
STAR final:
  step             908
  train PSNR       16.143315315246582
  heldout PSNR     13.881017684936523
  render-only      0.042270958016160876s

STAR gain-drop selected:
  step             400
  elapsed          26.221386374963913s
  train PSNR       15.686367511749268
  heldout PSNR     13.879861831665039
  render-only      0.0712511669844389s
  uses heldout     false

Heldout-best checkpoint in same curve:
  step             500
  heldout PSNR     13.894041061401367

Direct dynamic splats:
  steps            2403
  train PSNR       17.342755794525146
  heldout PSNR     11.199346542358398
  render-only      0.7236963339382783s
```

Read update: the weaker seed-1 paired repeat confirms the selector strongly
enough to use it as the current reporting selector. Selected STAR is about
`2.68` dB better than direct dynamic splats on heldout PSNR and about `10.2x`
faster by synchronized render-only timing. The selected render timing is
noisier than seed 2, but final STAR render-only in the same run is
`0.042270958016160876s`, so the speed direction is still clear. Next work
should scale or increase resolution with gain-drop frozen, not revisit the
forward rasterizer first.

512px same-budget scale probe command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 512 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --splat-renderer fast_mac --splat-count 2048 \
  --splat-camera-projection dataset_lens \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_512_16f_60s_both_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_traingain_drop002_checkpoint100_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

512px result:

```text
STAR final / selected:
  step             70
  train PSNR       9.331888198852539
  heldout PSNR     9.205381393432617
  train loop       68.71951654099996s
  final render     0.5322581669999522s
  selected render  0.09585146000000577s
  uses heldout     false
  selector detail  no_gain_drop_after_low_gain_before_final_checkpoint

Direct dynamic splats:
  steps            2095
  train PSNR       16.229832649230957
  heldout PSNR     10.980579376220703
  train loop       60.02396495900001s
  render-only      0.4002393330000018s
```

Read update: this rejects the naive full-resolution bump. STAR is too
step-starved at 512px with the same 256-tube 60-second recipe and loses heldout
PSNR by about `1.78` dB to direct dynamic splats. The selected STAR re-eval
render is still faster, but quality is not close, and the selector had no
useful curve to act on because there was only one checkpoint row. The next
512px path needs a changed scale strategy: longer STAR budget, multiscale or
crop/window training, or train-step throughput work.

512px window-1 scale-strategy probe command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 512 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 1 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_512_16f_60s_star_only_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_traingain_drop002_checkpoint100_temporal_window1_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle
```

512px window-1 STAR-only result:

```text
STAR final:
  step             1061
  train PSNR       16.179537296295166
  heldout PSNR     13.718976974487305
  train loop       60.01436449999994s
  render-only      0.11756491700009519s

STAR gain-drop selected:
  step             500
  elapsed          29.38946591699994s
  train PSNR       15.535942554473877
  heldout PSNR     13.551651000976562
  render-only      0.09932737500002986s
  uses heldout     false
  next gain        0.3253297805786133
  gain drop        0.21209287643432617

Heldout-best checkpoint in same curve:
  step             800
  train PSNR       16.030282497406006
  heldout PSNR     13.740259170532227
  render-only      0.09801204100017458s

Saved 512px direct dynamic splats:
  steps            2095
  train PSNR       16.229832649230957
  heldout PSNR     10.980579376220703
  render-only      0.4002393330000018s
```

Read update: `--uvt-window-frames 1` rescues 512px step throughput and quality.
STAR final is about `2.74` dB above the saved paired 512px direct-splat row and
renders about `3.4x` faster by render-only timing. It also clears the 256px
V-JEPA F32 reference `13.6248`, but the STAR-only frozen gain-drop selector
underselects step `500` and lands below that reference. The next check is a
formal paired rerun so this scale-strategy result does not rely on a saved
direct-splat row.

Formal paired 512px window-1 result:

```text
STAR final:
  step             1188
  train PSNR       16.308331966400146
  heldout PSNR     13.701825141906738
  train loop       60.02716958400015s
  render-only      0.11522445899981903s

STAR gain-drop selected:
  step             600
  elapsed          30.223125584000172s
  train PSNR       15.948731899261475
  heldout PSNR     13.678083419799805
  render-only      0.11207666699988295s
  uses heldout     false

Heldout-best checkpoint in same curve:
  step             900
  train PSNR       16.191404819488525
  heldout PSNR     13.729055404663086
  render-only      0.11145145899990894s

Direct dynamic splats:
  steps            1856
  train PSNR       15.943255424499512
  heldout PSNR     10.760580062866211
  train loop       60.019441415999836s
  render-only      0.5262394170001699s
```

Formal paired read: STAR final is about `2.94` dB above direct dynamic splats
and about `4.6x` faster by render-only timing. The non-heldout selected row also
clears the V-JEPA F32 reference and misses heldout-best by about `0.051` dB.
The scale blocker is now windowing/scale policy plus a better selector, not an
immediate forward-rasterizer rewrite.

512px stricter gain-drop selector diagnostic command:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 512 --max-frames 16 --train-seconds 60 --device mps --seed 2 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 1 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-plateau-delta 0.1 \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_512_16f_60s_star_only_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_traingain_drop002_checkpoint100_temporal_window1_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lowgain010
```

512px stricter gain-drop result:

```text
STAR final:
  step             1126
  train PSNR       16.33893585205078
  heldout PSNR     13.706064224243164
  render-only      0.11138250099975266s

STAR gain-drop selected:
  step             800
  elapsed          42.32113100000015s
  train PSNR       16.217055797576904
  heldout PSNR     13.677860260009766
  render-only      0.11746345799997471s
  uses heldout     false

Heldout-best checkpoint in same curve:
  step             1100
  train PSNR       16.33059310913086
  heldout PSNR     13.707569122314453
  render-only      0.11781229200005328s
```

Read update: lowering the low-gain threshold from `0.5` to `0.1` delayed
selection from the early shoulder to step `800`, but selected heldout PSNR was
still essentially the same as the formal paired step-600 selected row. On
512px/window-1, final/best-train reporting is currently the cleaner
non-heldout choice; another plain threshold tweak is unlikely to be useful.

Seed-1 512px/window-1 paired repeat and LR-decay bracket:

```text
Paired seed-1, LR decay step 300:
  STAR final:
    step           873
    train PSNR     15.402153015136719
    heldout PSNR   13.494588851928711
    render-only    0.3712127500000406s

  STAR gain-drop selected:
    step           400
    heldout PSNR   13.415751457214355
    render-only    0.1480234160001146s
    uses heldout   false

  Heldout-best:
    step           800
    heldout PSNR   13.498263359069824

  Direct dynamic splats:
    steps          1587
    train PSNR     14.744291305541992
    heldout PSNR   10.3926362991333
    render-only    0.4046094999998786s

STAR-only seed-1, LR decay step 500:
  STAR final:
    step           963
    train PSNR     16.17401695251465
    heldout PSNR   13.576004028320312
    render-only    0.34020658299959905s

  STAR gain-drop selected:
    step           600
    heldout PSNR   13.557477951049805
    render-only    0.13215666599990072s

  Heldout-best:
    step           700
    heldout PSNR   13.58298397064209

STAR-only seed-1, LR decay step 700:
  STAR final:
    step           730
    train PSNR     16.080385208129883
    heldout PSNR   13.397587776184082
    render-only    0.4030705410000337s

  STAR gain-drop selected:
    step           600
    heldout PSNR   13.414778709411621
    render-only    0.14510950000021694s

  Heldout-best:
    step           400
    heldout PSNR   13.490400314331055
```

Read update: seed 1 confirms that 512px/window-1 is a large direct-splat win
but not yet a seed-stable V-JEPA crossing. The seed-1 paired row beats direct
splats by about `3.10` dB heldout PSNR, and the best later-decay STAR-only row
beats the same direct-splat baseline by about `3.18` dB. But the best seed-1
heldout value found here, `13.58298397064209`, still misses V-JEPA F32
`13.6248` by about `0.042` dB. Step `700` decay is negative. Next work should
change capacity, support/window policy, or multiscale/crop training rather
than keep nudging LR-decay or gain-threshold scalars.

512px window-1 tube-capacity bracket:

```text
STAR-only seed-1, 384 tubes:
  final step       928
  final train PSNR 16.37885093688965
  final heldout    13.640532493591309
  selected step    600
  selected heldout 13.572845458984375
  selected render  0.10984833200018329s
  heldout-best     13.640532493591309
  max tile / pair  89 / 2.579706206704229
  overflow/unstable 0 / 0.0

STAR-only seed-2, 384 tubes:
  final step       946
  final train PSNR 16.32470178604126
  final heldout    13.4086275100708
  selected step    600
  selected heldout 13.339370727539062
  selected render  0.12307370799999262s
  heldout-best     13.415517807006836 at step 800
  max tile / pair  96 / 2.63295604941921
  overflow/unstable 0 / 0.0

STAR-only seed-1, 320 tubes:
  final step       732
  final train PSNR 16.15735626220703
  final heldout    13.682265281677246
  selected step    600
  selected heldout 13.637592315673828
  selected render  0.10943916599944714s
  heldout-best     13.769192695617676 at step 400
  checkpoint render 0.10670079200053806s final / 0.11296041700006754s heldout-best
  max tile / pair  92 / 2.777442764740763
  overflow/unstable 0 / 0.0

STAR-only seed-2, 320 tubes:
  final step       837
  final train PSNR 16.250001907348633
  final heldout    13.637543678283691
  selected step    600
  selected heldout 13.598714828491211
  selected render  0.10498941599962563s
  heldout-best     13.637543678283691
  checkpoint render 0.10442929100008769s
  max tile / pair  95 / 2.9231726580824278
  overflow/unstable 0 / 0.0

STAR-only seed-0, 320 tubes, LR decay step 500:
  final step       950
  final train PSNR 16.564763069152832
  final heldout    13.542135238647461
  final render     0.12215708199983055s
  selected step    600
  selected heldout 13.437091827392578
  selected render  0.11039912500018545s
  heldout-best     13.70832633972168 at step 400
  checkpoint render 0.13102466600003027s heldout-best
  max tile / pair  91 / 2.7181041505538044
  overflow/unstable 0 / 0.0

STAR-only seed-0, 256 tubes, LR decay step 500:
  final step       940
  final train PSNR 16.32259750366211
  final heldout    13.580879211425781
  final render     0.10690020900028685s
  selected step    600
  selected heldout 13.636795043945312
  selected render  0.11646866799992495s
  heldout-best     13.636795043945312 at step 600
  checkpoint render 0.11571520699999382s heldout-best
  max tile / pair  72 / 2.472526244148732
  overflow/unstable 0 / 0.0

STAR-only seed-0, 320 tubes, LR decay step 400:
  final step       994
  final train PSNR 16.223084449768066
  final heldout    13.612069129943848
  final render     0.1571206259995961s
  selected step    500
  selected heldout 13.586051940917969
  selected render  0.12204341700044097s
  heldout-best     13.652498245239258 at step 400
  checkpoint render 0.1324513740000839s heldout-best
  max tile / pair  95 / 3.0365741671511706
  overflow/unstable 0 / 0.0
```

Read update: the speed story survives the 512px capacity bracket. Clean
checkpoint render-only timing stays near `0.104-0.132s`, and all runs have zero
overflow and zero unstable tiles. Capacity is quality-sensitive rather than a
rasterizer blocker: 384 tubes rescues seed 1 but breaks seed 2, while 320 tubes
has useful heldout-best peaks but not robust non-heldout selection. Seed 0 is
the clearest example: 320 tubes peaks at `13.70832633972168`, above the 256-tube
peak, but the selector misses and lands at `13.437091827392578`; 256 tubes
selects its peak and clears V-JEPA at `13.636795043945312`. Moving 320 decay to
step `400` is not the fix. Next work should compare 256-vs-320 under a better
scale-aware selector or try support/window/multiscale policy, not start with a
Metal rasterizer rewrite.

Balanced train-view plateau selector diagnostic:

The new selector `first_balanced_train_psnr_plateau` keeps the train-PSNR
plateau idea but requires the selected previous checkpoint to have a bounded
train-camera PSNR gap. It is intended to avoid selecting a checkpoint that has
only overfit one train camera.

CPU smoke commands:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_balanced_train_psnr_plateau \
  --uvt-select-train-view-gap-max 1.0 --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_balanced_train_plateau_smoke_16_2f_1s

python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_balanced_train_psnr_plateau --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_balanced_train_plateau_default_smoke_16_2f_1s
```

Both smokes passed. The default smoke wrote
`selector: first_balanced_train_psnr_plateau`,
`select_train_view_gap_max: 1.2`, and `uses_heldout_for_selection: false`.

MPS seed-0 320-tube gap-1.0 output:
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_512_16f_60s_star_only_dataset_lens_seed0_alltrain_gridinit_allframes_lrdecay500x005_balanced_plateau_gap100_checkpoint100_temporal_window1_tileload0001_target7000_depthslope005_tilet1_cap256_tubes320_compact_bundle`

```text
STAR final:
  step             969
  train PSNR       16.323806762695312
  heldout PSNR     13.594148635864258
  render-only      0.15156229099920893s

STAR balanced selected:
  step             400
  train PSNR       15.122135639190674
  heldout PSNR     13.62360954284668
  render-only      0.12536841700148216s
  train-view gap   0.22287845611572266
  next gain        0.2531619071960449
  gap max          1.0

Heldout-best checkpoint:
  step             900
  train PSNR       16.25763988494873
  heldout PSNR     13.625420570373535
  render-only      0.11949833400103671s
```

MPS seed-1 320-tube gap-1.0 output:
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_512_16f_60s_star_only_dataset_lens_seed1_alltrain_gridinit_allframes_lrdecay500x005_balanced_plateau_gap100_checkpoint100_temporal_window1_tileload0001_target7000_depthslope005_tilet1_cap256_tubes320_compact_bundle`

```text
STAR final:
  step             922
  train PSNR       16.298762798309326
  heldout PSNR     13.597005844116211
  render-only      0.13014579099944967s

STAR balanced selected:
  step             900
  train PSNR       16.27601432800293
  heldout PSNR     13.600866317749023
  render-only      0.10447470900089684s
  train-view gap   0.8217201232910156
  next gain        0.022748470306396484
  gap max          1.0

Heldout-best checkpoint:
  step             400
  train PSNR       15.27962875366211
  heldout PSNR     13.698472023010254
  render-only      0.12727266699948814s
  train-view gap   1.1557254791259766
```

MPS seed-1 320-tube gap-1.2 output:
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_512_16f_60s_star_only_dataset_lens_seed1_alltrain_gridinit_allframes_lrdecay500x005_balanced_plateau_gap120_checkpoint100_temporal_window1_tileload0001_target7000_depthslope005_tilet1_cap256_tubes320_compact_bundle`

```text
STAR final:
  step             1000
  train PSNR       16.400321006774902
  heldout PSNR     13.662494659423828
  render-only      0.12017454200031352s

STAR balanced selected:
  step             500
  train PSNR       15.45426082611084
  heldout PSNR     13.442426681518555
  render-only      0.12534225000126753s
  train-view gap   0.7513523101806641
  next gain        0.49742698669433594
  gap max          1.2

Heldout-best checkpoint:
  step             800
  train PSNR       16.24009418487549
  heldout PSNR     13.675647735595703
  render-only      0.10657049999917945s
```

Read update: reject this as the current 512px selector. Gap max `1.0` picked
the intended seed-0 shoulder, but the selected heldout PSNR was still just
under V-JEPA F32 and only barely below the run's heldout-best checkpoint. On
seed 1, the same threshold missed the step-400 heldout peak because that peak's
train-view gap was `1.1557254791259766`, then selected a late step-900
checkpoint. Relaxing the threshold to `1.2` did not rescue the rule; it selected
a bad step-500 shoulder while final and heldout-best were much better. Keep the
selector code as a diagnostic, but do not keep threshold-tuning this path next.
The next 512px work should use a lightweight validation-rendered subset,
support/window policy, or multiscale/crop training rather than a Metal
rasterizer rewrite.

128px same-step single-video overfit paired reference:

The previous 128px paired artifact only used 50 steps. To answer the direct
same-step question, this run trains both the STAR-UVT single-video model and
the independent per-frame splat baseline for 200 steps on the same 16 frames.

```bash
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 1792 --per-frame-splats 64 \
  --target-size 128 --max-frames 16 --steps 200 \
  --lr 0.12 --per-frame-lr 0.32 --device mps --seed 5 \
  --uvt-init-mode video_samples \
  --uvt-spatial-precision 0.125 --uvt-temporal-precision 0.5 \
  --uvt-opacity 0.7 --uvt-sample-mode random \
  --uvt-render-backend metal_tile \
  --out-json research_project/benchmarks/results/video_fit_single_overfit_128_16f_200steps_1792uvt_lr012_s0125_64pf_lr032_metal_tile.json \
  --contact-sheet research_project/benchmarks/results/video_fit_single_overfit_128_16f_200steps_1792uvt_lr012_s0125_64pf_lr032_metal_tile.png
```

Result:

```text
STAR-UVT:
  steps          200
  tubes          1792
  train time     20.749791541000377s
  PSNR           22.31398344039917
  L1             0.04967977851629257
  render         3.108167000391404ms
  params         23296

Per-frame splats:
  steps          200
  splats/frame   64
  train time     687.8440475830003s
  PSNR           20.627903938293457
  L1             0.06224499270319939
  render         143.26100000016595ms
  params         9216

Delta:
  PSNR           +1.686079502105713 dB for STAR-UVT
  train speed    33.14944375339196x faster for STAR-UVT
  render speed   46.09179622012763x faster for STAR-UVT
```

Read update: this closes the missing same-step 128px overfit baseline. UVT is
not just faster at a lower step count; at equal 200 steps it is higher PSNR and
much faster than the independent per-frame splat baseline. This supports the
single-video speed thesis, but it remains an overfit fixture result. It does
not replace the multicam heldout selector/scale-policy work.

128px same-step temporal-support bracket:

These runs reuse the same 1792-tube, LR `0.12`, spatial precision `0.125`,
opacity `0.7`, 200-step Metal setup and skip the already measured slow
per-frame baseline.

```text
UVT temporal precision 1.0:
  steps          200
  PSNR           22.817583084106445
  L1             0.047857724130153656
  train time     22.077392624999447s
  render         4.774042000462941ms

UVT temporal precision 2.0:
  steps          200
  PSNR           23.130309581756592
  L1             0.04626224562525749
  train time     17.787110125000254s
  render         1.5611250000802102ms

UVT temporal precision 4.0:
  steps          200
  PSNR           22.765743732452393
  L1             0.04845142364501953
  train time     14.655789750000622s
  render         1.7657499993219972ms
```

Read update: at equal 200 UVT steps, narrower temporal support keeps helping up
to temporal precision `2.0`, then over-constrains at `4.0`. The `t=2.0` row is
now the best same-step 128px overfit point: versus the saved 64-splats/frame
200-step baseline it is `+2.5024056434631348` dB, `38.670927584589315x`
faster to train, and `91.76779565557226x` faster to render.

128px same-step temporal-support LR bracket:

```text
UVT temporal precision 2.0, LR 0.08:
  steps          200
  PSNR           22.904906272888184
  L1             0.04748460650444031
  train time     17.100815124999826s
  render         2.619916000185185ms

UVT temporal precision 2.0, LR 0.16:
  steps          200
  PSNR           22.98252582550049
  L1             0.04686432331800461
  train time     20.488064832999953s
  render         1.3112910000927513ms
```

Read update: LR `0.12` remains the current 128px equal-step setting. Both
nearby LR probes trail the saved `t=2.0`, LR `0.12` row at PSNR
`23.130309581756592`.

512px seed-0 320-tube window-2 policy check:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 512 --max-frames 16 --train-seconds 60 --device mps --seed 0 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 320 --uvt-lr 0.01 \
  --uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05 \
  --uvt-loss-scope temporal_window --uvt-window-frames 2 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_512_16f_60s_star_only_dataset_lens_seed0_alltrain_gridinit_allframes_lrdecay500x005_traingain_drop002_checkpoint100_temporal_window2_tileload0001_target7000_depthslope005_tilet1_cap256_tubes320_compact_bundle
```

Result:

```text
STAR final:
  step             536
  train PSNR       16.30258846282959
  heldout PSNR     13.391561508178711
  render-only      0.33027225000114413s

STAR gain-drop selected:
  step             500
  heldout PSNR     13.422961235046387
  render-only      0.10342058300011558s
  uses heldout     false

Heldout-best checkpoint:
  step             400
  train PSNR       15.661608695983887
  heldout PSNR     13.558874130249023
  render-only      0.10766366600000765s

Metal selected:
  max tile         84-89
  max pair ratio   2.8609017561213266
  overflow         0
  unstable         0.0
```

Read update: reject window 2 for this 512px seed-0 320-tube branch. It cuts
step throughput from the window-1 run's `950` steps to `536`, does not recover
the selected checkpoint, and lowers the heldout-best peak from
`13.70832633972168` to `13.558874130249023`. This is not a rasterizer problem:
Metal stats are clean. The next 512px branch should try a different
support/scale policy or validation-shaped selection signal, not a simple
window-2 escalation.

512px seed-0 320-tube hard LR-drop policy check:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 512 --max-frames 16 --train-seconds 60 --device mps --seed 0 \
  --uvt-render-backend metal_tile --uvt-camera-projection dataset_lens \
  --uvt-init-views all_train --uvt-init-sampling grid --uvt-init-frames all \
  --uvt-tubes 320 --uvt-lr 0.01 \
  --uvt-lr-decay-step 400 --uvt-lr-decay-factor 0.005 \
  --uvt-loss-scope temporal_window --uvt-window-frames 1 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --uvt-checkpoint-every-steps 100 \
  --uvt-select-checkpoint first_train_psnr_gain_drop \
  --uvt-select-train-psnr-plateau-delta 0.5 \
  --uvt-select-train-psnr-gain-drop 0.02 \
  --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_512_16f_60s_star_only_dataset_lens_seed0_alltrain_gridinit_allframes_lrdecay400x0005_traingain_drop002_checkpoint100_temporal_window1_tileload0001_target7000_depthslope005_tilet1_cap256_tubes320_compact_bundle
```

Result:

```text
STAR final:
  step             1000
  train PSNR       15.591412544250488
  heldout PSNR     13.599885940551758
  render-only      0.12993799899868463s

STAR gain-drop selected:
  step             1000
  heldout PSNR     13.599885940551758
  render-only      0.13297337400126708s
  uses heldout     false
  selector detail  no_gain_drop_after_low_gain_before_final_checkpoint

Heldout-best checkpoint:
  step             1000
  train PSNR       15.591412544250488
  heldout PSNR     13.599885940551758
  render-only      0.13298595900141663s

Metal selected:
  max tile         103
  max pair ratio   3.2112812143984426
  overflow         0
  unstable         0.0
```

Read update: reject hard scalar decay as the 512px seed-0 320-tube fix. A
`400 -> 0.005x` drop is stable and clean, but it lands below both the earlier
320-tube heldout-best peak `13.70832633972168` and the softer step-400 decay
heldout-best `13.652498245239258`. The render result remains fast and clean,
so this again points to support/scale policy or validation-shaped selection,
not a first-priority rasterizer rewrite.

512px seed-0 320-tube temporal-floor support check:

The 128px same-step overfit bracket showed narrower temporal support helps, so
this checks whether a simple `lambda_t` floor transfers to the multicam 512px
seed-0 320-tube branch.

```text
STAR-only seed-0, 320 tubes, min lambda_t 0.7:
  final step       1205
  final heldout    13.514815330505371
  selected step    500
  selected heldout 13.336383819580078
  selected render  0.10821341699738696s
  heldout-best     step 1205, 13.514815330505371
  max tile / pair  76 / 3.7183004841728744
  overflow/unstable 0 / 0.0

STAR-only seed-0, 320 tubes, min lambda_t 2.0:
  final step       1191
  final heldout    13.086738586425781
  selected step    600
  selected heldout 12.77337646484375
  selected render  0.11402837600053317s
  heldout-best     step 1191, 13.086738586425781
  max tile / pair  65 / 4.578496029044701
  overflow/unstable 0 / 0.0
```

Read update: reject a simple temporal floor as the 512px support fix. It keeps
the renderer clean but lowers quality versus the no-floor 320-tube branch
(heldout-best `13.70832633972168`, selected `13.437091827392578`). The
single-video temporal-support lever does not transfer to multicam as a global
minimum `lambda_t`.

Train-view gap-collapse selector:

Post-hoc on saved 512px/320-tube curves, a train-view gap collapse threshold
looked like it could separate the seed-0/seed-1 shoulders from seed 2:

```text
threshold 0.7:
  seed 0, 320 tubes -> step 400, heldout 13.70832633972168
  seed 1, 320 tubes -> step 400, heldout 13.769192695617676
  seed 2, 320 tubes -> fallback final step 837, heldout 13.637543678283691
```

So the selector was wired as
`--uvt-select-checkpoint first_train_view_gap_collapse`, using
`--uvt-select-train-view-gap-collapse`, and CPU-smoked at
`multicam_heldout_compare_train_view_gap_collapse_smoke_16_2f_1s`.

Live seed-0 MPS result:

```text
STAR final:
  step             978
  train PSNR       16.591779708862305
  heldout PSNR     13.70932674407959
  render-only      0.1661941240017768s

STAR gap-collapse selected:
  step             400
  train PSNR       15.269580364227295
  heldout PSNR     13.490143775939941
  render-only      0.1287422500026878s
  uses heldout     false
  next gap         0.12047004699707031 at step 500

Heldout-best checkpoint:
  step             900
  train PSNR       16.548688888549805
  heldout PSNR     13.739107131958008
  render-only      0.10841429100037203s

Metal selected:
  max tile         111
  max pair ratio   3.2891966678158213
  overflow         0
  unstable         0.0
```

Read update: reject train-view gap collapse as the current selector. It is a
reasonable diagnostic to keep, but this live repeat shows the train-camera gap
can collapse too early; selected fell below V-JEPA while final and heldout-best
cleared it. Another train-only scalar threshold is not the next path.

Train-camera temporal dev-frame selector:

The next attempt made a lightweight validation-shaped split inside the train
cameras instead of dropping a whole train camera. New flags:
`--uvt-validation-frame-stride`, `--uvt-validation-frame-offset`,
`--uvt-select-checkpoint best_train_dev_frame_psnr`, and
`--uvt-init-frames fit`. The fit-init CPU smoke
`multicam_heldout_compare_train_dev_frame_selector_fitinit_smoke_16_2f_1s`
passed and confirmed optimizer frame `[0]`, dev frame `[1]`, selected metric
`train_dev_frame_eval_psnr`, and `uses_heldout_for_selection: false`.

Real 512px seed-0, 320-tube fit-init run:

```text
optimizer frames  [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15]
dev frames        [1, 5, 9, 13]
selected step     1059
selected heldout  13.461018562316895
selected dev PSNR 16.582809448242188
selected render   0.10677025099903403s
heldout-best      step 300, 13.579765319824219
max tile / pair   88 / 2.3111431115765724
overflow/unstable 0 / 0.0
```

All-init control with the same optimizer/dev frame split:

```text
selected step     1125
selected heldout  13.39246654510498
selected dev PSNR 15.492460250854492
selected render   0.11622983400047815s
heldout-best      step 600, 13.449368476867676
max tile / pair   101 / 2.853536779149101
overflow/unstable 0 / 0.0
```

Read update: reject this train-camera temporal-dev selector lane. The clean
split loses too much quality, and the all-init control still selects the final
checkpoint by same-camera dev PSNR while true heldout is already falling. This
is no longer a promising cheap validation subset; next 512px work should change
multiscale/crop training or support/window policy.

512px free init-lambda support check:

The hard `min_lambda_t` floor was negative, so this tests a softer support
change: `--uvt-init-lambda-t 2.0` with the default tiny `min_lambda_t`, allowing
optimization to relax temporal support.

```text
STAR selected:
  step             600
  train PSNR       14.649134159088135
  heldout PSNR     13.213919639587402
  render-only      0.10834887499913748s
  uses heldout     false

STAR final:
  step             1209
  train PSNR       15.284734725952148
  heldout PSNR     13.40640640258789
  render-only      0.1142721670003084s

Heldout-best:
  step             1100
  heldout PSNR     13.415067672729492

Metal selected:
  max tile         76
  max pair ratio   4.07200551573087
  overflow         0
  unstable         0.0
```

Read update: reject narrow temporal initialization as the 512px support fix.
It keeps Metal clean and render speed intact, but quality is far below the
earlier no-floor 320-tube heldout-best peak `13.70832633972168`. The overfit
temporal-support win does not transfer through either a hard floor or an
init-only temporal precision bias.

512px bounded sequence-consistency support check:

This adds an occasional loss on multiple frames from the same rendered train
sequence:

```text
--uvt-sequence-consistency-every-steps
--uvt-sequence-consistency-frames
--uvt-sequence-consistency-weight
```

The CPU smoke passed, but the full 16-frame consistency backward failed on MPS
before useful training:

```text
RuntimeError: Invalid buffer size: 12.00 GiB
```

Four-frame consistency every 20 steps was too heavy:

```text
STAR selected/final:
  step             340
  train PSNR       14.951775550842285
  heldout PSNR     13.58269214630127
  render-only      0.11685991700323939s
  uses heldout     false

Metal selected:
  max tile         103
  max pair ratio   3.2703078231742175
  overflow         0
  unstable         0.000335693359375
```

Four-frame consistency every 50 steps recovered throughput and clean Metal
stats, but did not beat the current no-consistency branch:

```text
STAR selected:
  step             600
  train PSNR       16.08782720565796
  heldout PSNR     13.619542121887207
  render-only      0.11323120699853462s
  uses heldout     false

STAR final / heldout-best:
  step             666
  train PSNR       16.194210529327393
  heldout PSNR     13.626453399658203
  render-only      0.1246469170000637s

Metal selected:
  max tile         89
  max pair ratio   2.7667986055505245
  overflow         0
  unstable         0.0
```

Read update: reject sequence consistency as the current 512px support fix. The
every-50 result only roughly ties the V-JEPA row at final/heldout-best and the
non-heldout selected checkpoint is still below it; both bounded variants remain
below the no-consistency 320-tube heldout-best peak `13.70832633972168`. This
is still a support/window or multiscale-training problem before it is a
rasterizer rewrite.

512px multiscale auxiliary loss bracket:

This hook adds a coarse reconstruction term without a second render:

```text
--uvt-multiscale-loss-weight
--uvt-multiscale-loss-factor
```

The CPU smoke passed and logs `multiscale_loss` plus `multiscale_term`.

```text
Seed 0, factor 4, weight 0.25:
  selected step       500
  selected heldout    13.591398239135742
  selected train      15.623985290527344
  selected render     0.11842049999904702s
  final heldout       13.656238555908203
  heldout-best        step 1100, 13.656238555908203
  max tile / pair     101 / 3.0611386619566683
  overflow/unstable   0 / 0.0

Seed 1, factor 4, weight 0.25:
  selected step       600
  selected heldout    13.418802261352539
  selected train      15.749229907989502
  selected render     0.1141331260005245s
  final heldout       13.384246826171875
  heldout-best        step 400, 13.520487785339355
  max tile / pair     102 / 2.9002174273613615
  overflow/unstable   0 / 0.0
```

Seed 1, factor 4, weight 0.05:

```text
STAR selected:
  step             600
  train PSNR       15.75159215927124
  heldout PSNR     13.568358421325684
  render-only      0.11502208300044003s
  uses heldout     false

STAR final:
  step             1037
  train PSNR       16.221471786499023
  heldout PSNR     13.560102462768555
  render-only      0.15729729200029396s

Heldout-best:
  step             400
  heldout PSNR     13.603869438171387

Metal selected:
  max tile         100
  max pair ratio   2.893910082667028
  overflow         0
  unstable         0.0
```

Read update: simple global factor-4 multiscale loss is mixed-to-negative, not
a default. Seed 0 says coarse support weighting can help final quality, but the
gain-drop selector still fires too early. Seed 1 rejects weight `0.25`, and
lowering to `0.05` only partially recovers while remaining worse than the saved
no-multiscale 320-tube row: selected/final/heldout-best `13.637592315673828` /
`13.682265281677246` / `13.769192695617676`. Keep the hook for targeted
crop/scale experiments, but do not promote this global auxiliary branch.

512px deterministic crop auxiliary loss bracket:

This hook adds local full-resolution pressure from the existing render:

```text
--uvt-crop-loss-weight
--uvt-crop-loss-size
```

The crop origin cycles over a deterministic 3x3 grid. The CPU smoke passed and
logs `crop_loss` plus `crop_term`.

```text
Seed 1, crop size 256, weight 0.25:
  selected/final step 578
  train PSNR          15.60933542251587
  heldout PSNR        13.566254615783691
  render-only         0.17345320899767103s
  uses heldout        false
  max tile / pair     104 / 3.0012101814828935
  overflow/unstable   0 / 0.0

Seed 1, crop size 128, weight 0.25:
  selected step       500
  selected train      14.819037437438965
  selected heldout    13.48839282989502
  selected render     0.16554591700150922s
  final heldout       13.565434455871582
  heldout-best        step 400, 13.591614723205566
  max tile / pair     101 / 3.2960348470935785
  overflow/unstable   0 / 0.0
```

Read update: reject deterministic crop loss as the 512px support fix. It is
clean in Metal, but it starves the optimizer and stays well below the saved
no-crop seed-1 row. Local crop weighting alone is not the missing mechanism.

512px deterministic cycle train-schedule revisit:

The earlier 256px 20-second cycle schedule gate was negative, but that does
not transfer to the tuned 512px 320-tube/window-1 recipe. I reran seeds 0 and 1
with the no-aux recipe and changed only:

```text
--uvt-train-schedule cycle
```

Seed 0:

```text
out dir              research_project/benchmarks/results/mcam512_s0_t320_cycle
steps                1130
selected step         600
selected train        15.494235038757324
selected heldout      13.798948287963867
selected render       0.10962479100089695s
heldout-only render   0.0361380829999689s
final heldout         13.692935943603516
heldout-best          step 500, 13.841632843017578
max tile / pair       92 / 2.835304099610527
overflow/unstable     0 / 0.0
```

Seed 1:

```text
out dir              research_project/benchmarks/results/mcam512_s1_t320_cycle
steps                1120
selected step         600
selected train        15.35140609741211
selected heldout      13.915006637573242
selected render       0.11339320700062672s
heldout-only render   0.038770208000642015s
final heldout         13.743698120117188
heldout-best          step 500, 13.91877555847168
max tile / pair       94 / 2.835488909234995
overflow/unstable     0 / 0.0
```

Read update before seed 2: this was the first positive 512px branch after the
sequence, multiscale, and crop auxiliary losses. The selected checkpoints use
the non-heldout train-gain-drop selector, clear the V-JEPA F32 reference
`13.6248` on seeds 0 and 1, and keep render-only eval around `0.11s` across
the three eval sequences.

Seed-2 follow-up:

```text
out dir              research_project/benchmarks/results/mcam512_s2_t320_cycle
steps                1591
selected step         400
selected train        14.109616756439209
selected heldout      13.023938179016113
selected render       0.14805312499993306s
heldout-only render   0.048837874999662745s
final heldout         7.100964546203613
heldout-best          step 300, 13.442176818847656
max tile / pair       94 / 3.439398660382039
overflow/unstable     0 / 0.0 at selected
```

This run becomes non-finite after step `480`: reconstruction remains finite, but
model/projection regularizers become non-finite and the final Metal report has
zero active tiles. The lower-LR stability bracket is finite but underfits:

```text
out dir              research_project/benchmarks/results/mcam512_s2_t320_cycle_lr005
lr                   0.005
steps                463
selected step         463
selected train        13.940347671508789
selected heldout      13.037338256835938
selected render       0.17735508299847424s
heldout-best          step 463, 13.037338256835938
max tile / pair       112 / 3.4973726765940043
overflow/unstable     0 / 0.0
```

Read update after seed 2: reject plain cycle as the seed-robust 512px default.
It is a useful signal that sampling order matters, but the next branch should
try shuffled or phase-randomized coverage and add a non-finite guard, not a
rasterizer rewrite.

Shuffled-cycle follow-up:

I added `--uvt-train-schedule shuffled_cycle`. The schedule precomputes shuffled
cycle orders for view/frame pairs or view/window pairs, so it preserves full
coverage without the fixed phase that broke plain cycle on seed 2. The CPU smoke
`multicam_heldout_compare_shuffled_cycle_smoke_16_2f_1s` passed and reported
`train_schedule: shuffled_cycle`.

Full seed-2 result:

```text
out dir              research_project/benchmarks/results/mcam512_s2_t320_shuffled_cycle
steps                920
stopped_reason       null
selected step         600
selected train        16.362467765808105
selected heldout      13.574305534362793
selected render       0.13692954100042698s
heldout-only render   0.042509167000389425s
final heldout         13.566455841064453
heldout-best          step 300, 13.6640625
max tile / pair       93 / 2.8858366120208703
overflow/unstable     0 / 0.0
```

This rescues the seed-2 collapse from plain cycle, but it does not yet clear
V-JEPA with the current non-heldout selector. The heldout-best checkpoint does
clear it, which means the quality exists briefly and the problem is still
schedule/selection stability.

Earlier LR decay is negative:

```text
out dir              research_project/benchmarks/results/mcam512_s2_t320_shuffled_cycle_lrdecay300
selected step         400
selected train        14.864503383636475
selected heldout      13.564926147460938
selected render       0.17289004099984595s
final/heldout-best    step 689, 13.578947067260742
max tile / pair       106 / 3.220804280059469
overflow/unstable     0 / 0.0
```

I also checked a balanced train-plateau selector (`delta=0.6`, `gap=1.7`) after
the saved rows suggested it would pick better shoulders. The live seed-2 rerun
did not reproduce the original shuffled curve:

```text
out dir              research_project/benchmarks/results/mcam512_s2_t320_shuffled_cycle_balplateau_d06_gap17
steps                456
selected step         400
selected train        15.691062927246094
selected heldout      13.52459716796875
selected render       0.18055795800137275s
final/heldout-best    step 456, 13.557467460632324
max tile / pair       98 / 3.1285079310153523
overflow/unstable     0 / 0.0
```

Read update: do not promote `shuffled_cycle` plus balanced plateau yet. It is
the right mechanism family, but the live rerun says the curve is not stable
enough. The next iteration should make coverage/schedule deterministic under
less MPS contention or use fixed-step matched reruns before touching the
rasterizer.

Fixed-step rerun:

I reran the same seed-2 shuffled-cycle recipe with `--max-steps 300` and
`--train-seconds 999`, so the step cap rather than wall clock stopped training.

```text
out dir              research_project/benchmarks/results/mcam512_s2_t320_shuffled_cycle_fixed300
steps                300
stopped_reason       null
train loop           27.671377624999877s
final/train PSNR     14.725461959838867
final/heldout PSNR   13.527697563171387
checkpoint render    0.15980995899735717s
final render         0.48209829200095555s
max tile / pair      108 / 3.5619259781486448
overflow/unstable    0 / 0.0
```

Read update: this confirms the earlier step-300 heldout-best `13.6640625` was
not reproducible enough to target with a selector. The branch is clean in Metal,
but quality is unstable; do not promote this schedule or start rasterizer work
from it.

Reshuffled-cycle schedule:

I added `--uvt-train-schedule reshuffled_cycle`. It preserves full coverage like
`shuffled_cycle`, but reshuffles the coverage order every epoch instead of
repeating one shuffled order forever. The CPU smoke
`multicam_heldout_compare_reshuffled_cycle_smoke_16_2f_1s` passed with
`train_schedule: reshuffled_cycle`, `steps: 2`, and `stopped_reason: null`.

First seed-2 fixed-300 check:

```text
out dir              research_project/benchmarks/results/mcam512_s2_t320_reshuffled_cycle_fixed300
steps                300
stopped_reason       null
train loop           34.773467208999136s
final/train PSNR     14.792136669158936
final/heldout PSNR   13.639530181884766
checkpoint render    0.19894387400017877s
max tile / pair      99 / 3.305289860131723
overflow/unstable    0 / 0.0
```

This beats the fixed shuffled step-300 rerun (`13.527697563171387`) and clears
the V-JEPA reference, but it is still only one seed. I then ran fixed 600 steps
with the normal non-heldout gain-drop selector:

```text
Seed 0:
  out dir             research_project/benchmarks/results/mcam512_s0_t320_reshuffled_cycle_fixed600_gain
  selected step       600
  selected heldout    13.639025688171387
  selected train      16.0900239944458
  selected render     0.19431933400119306s
  heldout-best        step 600, 13.639025688171387
  max tile / pair     92 / 2.7868366656300876
  overflow/unstable   0 / 0.0

Seed 1:
  out dir             research_project/benchmarks/results/mcam512_s1_t320_reshuffled_cycle_fixed600_gain
  selected step       600
  selected heldout    13.894740104675293
  selected train      15.092887878417969
  selected render     0.18023745800019242s
  heldout-best        step 600, 13.894740104675293
  max tile / pair     98 / 2.6724916167562047
  overflow/unstable   0 / 0.0

Seed 2:
  out dir             research_project/benchmarks/results/mcam512_s2_t320_reshuffled_cycle_fixed600_gain
  selected step       500
  selected heldout    13.700183868408203
  selected train      15.877279281616211
  selected render     0.1726733739978954s
  final heldout       13.660613059997559
  heldout-best        step 500, 13.700183868408203
  max tile / pair     91 / 2.9009679371370676
  overflow/unstable   0 / 0.0
```

Read update: `reshuffled_cycle` is the first robust-floor candidate. It clears
the V-JEPA F32 reference on all three seeds without heldout selection and keeps
Metal clean. It is not a pure replacement for plain cycle because seed 0 drops
from the plain-cycle selected `13.798948287963867` to `13.639025688171387`.
The next schedule policy should try to keep reshuffled's seed-2 stability while
recovering the seed-0/seed-1 cycle strength.

Phase-rotated-cycle schedule:

I added `--uvt-train-schedule phase_rotated_cycle`. It keeps the ordered cycle
pairs but rotates the start point each coverage epoch. The CPU smoke
`multicam_heldout_compare_phase_rotated_cycle_smoke_16_2f_1s` passed with
`train_schedule: phase_rotated_cycle`, `steps: 2`, and `stopped_reason: null`.

Seed 2:

```text
out dir              research_project/benchmarks/results/mcam512_s2_t320_phase_rotated_cycle_fixed600_gain
selected step         600
selected heldout      13.706971168518066
selected train        15.267481803894043
selected render       0.11264262600161601s
heldout-best          step 600, 13.706971168518066
max tile / pair       91 / 2.94270269147189
overflow/unstable     0 / 0.0
```

Seed 0:

```text
out dir              research_project/benchmarks/results/mcam512_s0_t320_phase_rotated_cycle_fixed600_gain
selected step         600
selected heldout      13.602667808532715
selected train        15.431066513061523
selected render       0.1473905829989235s
heldout-best          step 600, 13.602667808532715
max tile / pair       95 / 2.9729977723513414
overflow/unstable     0 / 0.0
```

Read update: reject `phase_rotated_cycle` for now. It is faster and slightly
better than reshuffled on seed 2, but it falls below the V-JEPA reference and
below reshuffled on seed 0. I did not spend seed 1 because the branch already
fails the robust-floor test.

View-shuffled-cycle schedule:

I added `--uvt-train-schedule view_shuffled_cycle`. It keeps frame/window order
temporal, like cycle, but shuffles the train-camera order inside each
frame/window slot. The CPU smoke
`multicam_heldout_compare_view_shuffled_cycle_smoke_16_2f_1s` passed with
`train_schedule: view_shuffled_cycle`, `steps: 2`, and `stopped_reason: null`.

```text
Seed 0:
  out dir             research_project/benchmarks/results/mcam512_s0_t320_view_shuffled_cycle_fixed600_gain
  selected step       600
  selected heldout    13.639522552490234
  selected train      15.180933952331543
  selected render     0.17508845800330164s
  heldout-best        step 600, 13.639522552490234
  max tile / pair     96 / 3.1756918117808755
  overflow/unstable   0 / 0.0

Seed 1:
  out dir             research_project/benchmarks/results/mcam512_s1_t320_view_shuffled_cycle_fixed600_gain
  selected step       500
  selected heldout    13.7864990234375
  selected train      14.93717908859253
  selected render     0.155762375004997s
  final/heldout-best  step 600, 13.812097549438477
  max tile / pair     97 / 3.113578796795887
  overflow/unstable   0 / 0.0029296875

Seed 2:
  out dir             research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_gain
  selected step       500
  selected heldout    13.788138389587402
  selected train      14.891727924346924
  selected render     0.14073404100054177s
  final/heldout-best  step 600, 13.793721199035645
  max tile / pair     96 / 3.3630823644462753
  overflow/unstable   0 / 0.0
```

Read update: this is the best current robust-floor schedule. It improves seed 2
over reshuffled while keeping seed 0 at the same floor, and all selected
checkpoints clear V-JEPA without heldout selection. The remaining issue is
selector policy: fixed step `600` is better than the selected step `500` on
seeds 1 and 2.

View-shuffled best-train selector:

I did not need a new selector for the fixed-600 report. The existing
`best_train_psnr` selector picks the best train-PSNR checkpoint without using
heldout. The CPU smoke passed:

```text
out dir          research_project/benchmarks/results/multicam_heldout_compare_view_shuffled_besttrain_selector_smoke_16_2f_1s
train schedule   view_shuffled_cycle
selector         best_train_psnr
selected step    2
uses heldout     false
```

On the saved 512px view-shuffled curves, `best_train_psnr` selects step `600` on
all three seeds:

```text
Seed 0 best_train    step 600, heldout 13.639522552490234, render 0.1782556249963818s
Seed 1 best_train    step 600, heldout 13.812097549438477, render 0.1593077910001739s
Seed 2 best_train    step 600, heldout 13.793721199035645, render 0.1350984589989821s
```

Read update: the current recipe to carry forward is `view_shuffled_cycle` plus
`--uvt-select-checkpoint best_train_psnr`. It remains non-heldout and avoids the
early gain-drop trigger on seeds 1 and 2.

Non-finite guard follow-up:

The trainer now checks scalar loss at the existing log cadence, records
`stopped_reason` / `stopped_step`, serializes non-finite scalar log fields as
`null`, and restores the last checkpointed finite state before final eval. I
first tried restoring every step, but that made MPS training too slow, so the
restore point is now tied to the existing checkpoint cadence.

Normal CPU smoke:

```text
out dir          research_project/benchmarks/results/multicam_heldout_compare_nonfinite_guard_smoke3_16_2f_1s
stopped_reason   null
steps            2
```

Forced non-finite CPU smoke:

```text
command delta    --uvt-depth-slope-reg nan
out dir          research_project/benchmarks/results/multicam_heldout_compare_nonfinite_guard_forced_smoke_16_2f_1s
stopped_reason   nonfinite_loss
stopped_step     1
steps            0
log loss         null
log projected    null
```

This does not fix seed-2 cycle quality. It makes failed branches explicit and
keeps the final eval tied to a finite state.
