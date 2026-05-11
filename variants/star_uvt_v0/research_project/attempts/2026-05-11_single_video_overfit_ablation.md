# 2026-05-11 Single-Video Overfit Ablation

Question: with heldout cameras set aside, can STAR-UVT overfit one video at the
same optimizer-step count as a per-frame splat baseline, and is the speed thesis
showing up?

Dataset and fixed contract:

```text
video:        dynaworld/test_data/test_video_small_128_4fps.mp4
frames:       16
target size:  32x32
device:       mps
steps:        200
seed:         7
```

## Script Changes

`research_project/benchmarks/video_fit_comparison.py` now reports final MSE,
PSNR, and render time. It also supports UVT-only ablations via
`--skip-per-frame` and supports a data-sampled UVT initializer via
`--uvt-init-mode video_samples`. The later 64px follow-up added
`--uvt-sample-mode stratified` and `--per-frame-lr` so UVT and per-frame splats
can use separately tuned learning rates in the same comparison. The 128px
follow-up added an optional color/opacity-only refinement tail via
`--uvt-appearance-refine-steps` and `--uvt-appearance-lr`. It also added a
naive block-match velocity initializer via `--uvt-velocity-init block_match` for
motion-aware init tests.

`research_project/trainer_harness/model.py` now has
`ScreenTimeTubeModel.from_video_samples(...)`, which initializes tube centers
from target pixels and frame times. It supports both random and stratified
sample placement.

## Results

Baseline fixed-step paired runs:

```text
64 UVT, lr 0.04:
  PSNR 21.764323711395264
  train 9.535820624994813s
  render 6.768666004063562ms

64 splats/frame, lr 0.04:
  PSNR 25.14953851699829
  train 61.06520704101422s
  render 40.68333297618665ms
```

Data-sampled UVT init was not enough at the original LR:

```text
64 UVT, video_samples, lr 0.04:
  spatial 0.25, temporal 0.25, opacity 0.35 -> PSNR 21.970112323760986
  spatial 0.50, temporal 2.00, opacity 0.70 -> PSNR 21.633059978485107
```

Learning rate and capacity helped. Data-sampled init became useful once LR was
raised:

```text
64 UVT,  lr 0.08 -> PSNR 22.59230375289917
128 UVT, lr 0.04 -> PSNR 22.782671451568604
128 UVT, lr 0.08 -> PSNR 23.555450439453125
128 UVT, lr 0.12 -> PSNR 23.806335926055908
128 UVT, lr 0.16 -> PSNR 23.937160968780518
128 UVT, video_samples, lr 0.12 -> PSNR 23.99590015411377
128 UVT, video_samples, lr 0.16 -> PSNR 24.188532829284668
128 UVT, video_samples, lr 0.24 -> PSNR 24.315695762634277
128 UVT, video_samples, lr 0.32 -> PSNR 24.338133335113525
128 UVT, video_samples, lr 0.32, temporal 0.5, opacity 0.7 -> PSNR 24.639911651611328
192 UVT, video_samples, lr 0.32, temporal 0.5, opacity 0.7 -> PSNR 25.805532932281494
224 UVT, video_samples, lr 0.32, temporal 0.5, opacity 0.7 -> PSNR 26.46265983581543
240 UVT, video_samples, lr 0.32, temporal 0.5, opacity 0.7 -> PSNR 26.36221408843994
224 UVT, video_samples, lr 0.48, temporal 0.5, opacity 0.7 -> PSNR 25.933196544647217
224 UVT, video_samples, lr 0.48 -> 0.16 at step 100 -> PSNR 25.79216480255127
224 UVT, video_samples, lr 0.32 -> 0.16 at step 150 -> PSNR 26.341335773468018
256 UVT, lr 0.12 -> PSNR 24.836182594299316
256 UVT, lr 0.16 -> PSNR 24.366016387939453
```

Convergence bracket for the tuned 224-tube recipe:

```text
224 UVT, lr 0.32, 200 steps -> PSNR 26.46265983581543, train 66.62527337501524s
224 UVT, lr 0.32, 320 steps -> PSNR 27.038772106170654, train 107.8203659580031s
224 UVT, lr 0.32, 340 steps -> PSNR 27.101047039031982, train 114.65805949998321s
224 UVT, lr 0.32, 400 steps -> PSNR 27.22731113433838, train 131.91997808398446s
64 splats/frame, lr 0.32, 200 steps -> PSNR 27.248921394348145, train 118.61497245798819s
```

Best self-contained paired run so far:

```text
command:
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 224 --per-frame-splats 64 --target-size 32 --max-frames 16 \
  --steps 200 --lr 0.32 --device mps --seed 7 \
  --uvt-init-mode video_samples --uvt-spatial-precision 0.25 \
  --uvt-temporal-precision 0.5 --uvt-opacity 0.7 \
  --out-json research_project/benchmarks/results/video_fit_single_overfit_32_16f_200steps_224uvt_64pf_tuned_lr032.json \
  --contact-sheet research_project/benchmarks/results/video_fit_single_overfit_32_16f_200steps_224uvt_64pf_tuned_lr032.png

STAR-UVT:
  PSNR 26.46265983581543
  L1 0.03167424723505974
  train 66.62527337501524s
  render 31.05658298591152ms
  params 2912

per-frame splats:
  PSNR 27.248921394348145
  L1 0.030112704262137413
  train 118.61497245798819s
  render 102.62579101254232ms
  params 9216
```

## Read

STAR-UVT has a real small-scale speed signal with the tuned 224-tube recipe:
it is within `0.7862615585327148` dB of the tuned 64-splats/frame baseline,
while training about `1.78x` faster and rendering about `3.30x` faster in this
small overfit harness. The useful knobs were higher LR, video-sampled init,
moderately local temporal precision (`0.5`), higher initial opacity (`0.7`),
and capacity around 224 tubes. 240 tubes was slightly worse, and the older
256-tube non-tuned run lost the speed advantage.

The convergence bracket is stronger than the fixed-200-step comparison: 340
STAR-UVT steps fit to PSNR `27.101047039031982` in
`114.65805949998321s`, just under the tuned per-frame baseline's
`118.61497245798819s`, while keeping render time near `32ms`. At 400 steps,
STAR-UVT essentially ties the per-frame PSNR (`27.22731113433838` versus
`27.248921394348145`) but takes `131.91997808398446s`, so it is no longer a
train-time win.

Simple staged LR did not help. Both high-to-low (`0.48 -> 0.16` at step 100)
and late decay (`0.32 -> 0.16` at step 150) trailed the constant `0.32`
200-step baseline.

## 64px Transfer

The tuned 32px recipe does not transfer directly to 64px:

```text
224 UVT, 64x64, 16 frames, 200 steps:
  PSNR 23.345627784729004
  train 254.14723570799106s
  dense render 81.55500001157634ms

448 UVT, 64x64, 16 frames, 100 steps:
  PSNR 23.94777774810791
  train 350.3017887909955s
  dense render 161.26137500395998ms
```

Capacity helps, but dense training becomes too slow to use as the main
iteration loop.

Metal tile-backward changes the 32px result substantially:

```text
224 UVT, 32x32, 16 frames, 200 Metal steps:
  PSNR 25.75181007385254
  train 14.252110375004122s
  render 1.2917090207338333ms

224 UVT, 32x32, 16 frames, 800 Metal steps:
  PSNR 27.42915630340576
  train 54.56761045800522s
  render 1.6061250062193722ms
```

The 800-step Metal run beats the tuned 64-splats/frame baseline PSNR
`27.248921394348145` while taking less than half of its train time and rendering
much faster.

At 64px, Metal tile-backward makes iteration practical, but quality is not
solved yet:

```text
224 UVT, 64x64, 16 frames, 800 Metal steps:
  PSNR 24.250736236572266
  train 128.09551870898576s
  render 5.836124997586012ms

224 UVT, 64x64, 16 frames, 1600 Metal steps:
  PSNR 24.356164932250977
  train 278.4675174159929s
  render 1.7624159809201956ms

448 UVT, 64x64, 16 frames, 800 Metal steps:
  PSNR 23.577630519866943
  train 115.49633204200654s
  render 2.4364170094486326ms
```

Doubling the 224-tube run from 800 to 1600 Metal steps only gained about
`0.10542869567871094` dB, so longer training alone is not the 64px answer.
448 tubes was worse than 224 tubes at the same LR, so larger capacity alone is
not enough either. A reducer-shape bug in the Metal tile-backward bridge was
fixed while testing this path: sample buffers are row-normalized, ids and
samples are truncated to their shared length, invalid tube ids are filtered,
and reduction gathers both ids and samples from one explicit position list
before `index_add_`.

Forward-only timing on video-initialized tensors shows the Metal path is still
promising:

```text
224 UVT, 64x64, 16 frames:
  dense forward 155.59799999270277ms
  Metal forward 47.28181932781202ms
  pair ratio 0.7782855868910935
  overflow tiles 0

448 UVT, 64x64, 16 frames:
  dense forward 309.5513886655681ms
  Metal forward 125.89868066910033ms
  pair ratio 0.7766065388951522
  overflow tiles 0
```

Current read: larger-resolution progress needs the Metal tile backward or
another efficient backward route. Dense-backward sweeps are now the bottleneck
for iteration speed, but the 64px quality gap is now a representation/training
problem rather than just a renderer-speed problem.

## 64px LR and Baseline Follow-Up

Stratified video-sample initialization was worse than random sampling:

```text
224 UVT, 64x64, 16 frames, 800 Metal steps, stratified samples:
  PSNR 23.263163566589355
  train 108.5340038750146s
  render 1.4634999970439821ms
```

The earlier 448-tube loss was mostly an LR issue. Lowering LR made 448 tubes
the current 64px winner:

```text
448 UVT, 64x64, 16 frames, 800 Metal steps, LR 0.16:
  PSNR 24.879634380340576
  train 107.97213766700588s
  render 1.817332988139242ms

448 UVT, 64x64, 16 frames, 800 Metal steps, LR 0.24:
  PSNR 25.096933841705322
  train 104.36115416602115s
  render 1.6158749931491911ms

448 UVT, 64x64, 16 frames, 800 Metal steps, LR 0.28:
  PSNR 24.401702880859375
  train 114.05838112501078s
  render 2.907874993979931ms
```

Same-step 64px comparison against the local per-frame splat baseline:

```text
448 UVT, LR 0.24, 200 Metal steps:
  PSNR 23.8846492767334
  train 19.297393749991897s
  render 2.980333985760808ms
  params 5824

64 splats/frame, LR 0.32, 200 steps:
  PSNR 23.97939920425415
  train 211.76061187498271s
  render 95.37416699458845ms
  params 9216
```

The 200-step UVT result is within `0.09474992752075195` dB of the splat
baseline while training about `10.97x` faster and rendering about `32x` faster.

Same-wall-clock read: 448 UVT at LR `0.24` for 800 Metal steps reached PSNR
`25.096933841705322` in `104.36115416602115s`, already beating the 200-step
splat baseline while taking less than half its train time. At 1600 Metal steps
it reached PSNR `25.285780429840088` in `229.8335517499945s`, still rendering
in `2.0606659818440676ms`. That is slightly longer than the 200-step splat
runtime, but it is `1.3063812255859375` dB higher PSNR and about `46x` faster
at render time.

## 128px Transfer

The 64px recipe transfers in speed, but the quality needs resolution-aware
capacity and support tuning:

```text
448 UVT, 128x128, 16 frames, 200 Metal steps, LR 0.24, spatial 0.25:
  PSNR 20.90794801712036
  train 55.213978125015274s
  render 2.2840409947093576ms

896 UVT, 128x128, 16 frames, 200 Metal steps, LR 0.16, spatial 0.25:
  PSNR 21.883294582366943
  train 54.04171474999748s
  render 6.051458010915667ms

896 UVT, 128x128, 16 frames, 200 Metal steps, LR 0.24, spatial 0.25:
  PSNR 21.376972198486328
  train 52.18457691700314s
  render 1.521624973975122ms

1792 UVT, 128x128, 16 frames, 200 Metal steps, LR 0.12, spatial 0.25:
  PSNR 22.033333778381348
  train 60.05925966697396s
  render 1.7228329961653799ms

1792 UVT, 128x128, 16 frames, 200 Metal steps, LR 0.16, spatial 0.25:
  PSNR 21.683576107025146
  train 50.42433258402161s
  render 1.3567499991040677ms

1792 UVT, 128x128, 16 frames, 200 Metal steps, LR 0.12, spatial 0.125:
  PSNR 22.2884202003479
  train 51.93527437499142s
  render 1.6798329888843ms

1792 UVT, 128x128, 16 frames, 200 Metal steps, LR 0.12, spatial 0.0625:
  PSNR 21.938321590423584
  train 54.058288625004934s
  render 1.3862499909009784ms

1792 UVT, 128x128, 16 frames, 400 Metal steps, LR 0.12, spatial 0.125:
  PSNR 22.23587989807129
  train 109.82694870900013s
  render 1.4550419873557985ms

1792 UVT, 128x128, 16 frames, 200 + 200 appearance-only steps:
  LR 0.12, appearance LR 0.04, spatial 0.125
  PSNR 22.21776008605957
  train 119.35245308399317s
  render 1.6747920017223805ms

1792 UVT, 128x128, 16 frames, 400 Metal steps, LR 0.12 -> 0.02 at step 200:
  PSNR 22.72578239440918
  train 126.49729370800196s
  render 2.073166018817574ms

1792 UVT, 128x128, 16 frames, 400 Metal steps, LR 0.12 -> 0.04 at step 200:
  PSNR 22.809326648712158
  train 134.12330012500752s
  render 2.7211669948883355ms

1792 UVT, 128x128, 16 frames, 400 Metal steps, LR 0.12 -> 0.06 at step 200:
  PSNR 21.763882637023926
  train 113.54914891699445s
  render 2.128332998836413ms

1792 UVT, 128x128, 16 frames, block-match velocity init:
  LR 0.12 -> 0.04 at step 200, spatial 0.125, temporal 0.5
  PSNR 21.428205966949463
  train 135.5116336660285s
  render 2.4871669884305447ms

1792 UVT, 128x128, 16 frames, temporal precision 1.0:
  LR 0.12 -> 0.04 at step 200, spatial 0.125
  PSNR 23.209903240203857
  train 131.20218429199304s
  render 3.2060420198831707ms

1792 UVT, 128x128, 16 frames, temporal precision 2.0:
  LR 0.12 -> 0.04 at step 200, spatial 0.125
  PSNR 23.207027912139893
  train 137.8384064999991s
  render 6.680000020423904ms
```

Current 128px best is 1792 tubes, LR `0.12 -> 0.04` at step 200, spatial
precision `0.125`, temporal precision `1.0`, 400 Metal steps. Constant LR for
400 steps was worse, and the appearance-only refinement tail was also worse, so
the useful mechanisms were whole-model LR decay and shorter temporal support.
Naive block-match velocity initialization was actively harmful.

Bounded 128px per-frame reference:

```text
1792 UVT, LR 0.12, spatial 0.125, 50 Metal steps:
  PSNR 20.928823947906494
  train 13.987883624999085s
  render 2.2979159839451313ms

64 splats/frame, LR 0.32, 50 steps:
  PSNR 19.460207223892212
  train 175.3501634580025s
  render 161.16950000287034ms
```

This bounded 128px comparison says UVT is still ahead of the local per-frame
splat baseline at equal steps while training about `12.54x` faster and
rendering about `70x` faster. It does not yet prove a full 200-step 128px
splat comparison, because that dense baseline would be much slower.

The immediate blocker is not just the rasterizer. Rasterizer work matters for
the speed thesis at higher tube counts, but the representation/training setup
still needs to close the remaining single-video PSNR gap. The next useful work
is a representation mechanism: piecewise temporal splitting or a better
motion-aware initializer. The naive block matcher did not help. Color/opacity
only refinement did not help in this harness; staged whole-model LR decay did.
