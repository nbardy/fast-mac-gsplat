# STAR-UVT Benchmarks

This folder holds small side-by-side reports for the UVT research lane.

Current runnable benchmark:

```bash
python3 research_project/benchmarks/uvt_pair_benchmark.py
python3 research_project/benchmarks/backward_performance_smoke.py
python3 research_project/benchmarks/backward_performance_matrix.py
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --out-json research_project/benchmarks/results/video_fit_comparison_fixture.json \
  --contact-sheet research_project/benchmarks/results/video_fit_comparison_fixture.png
python3 research_project/benchmarks/training_comparison.py
python3 research_project/benchmarks/uvt_forward_speed_probe.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --target-size 64 --max-frames 16 --tube-counts 224,448 \
  --spatial-precision 0.25 --temporal-precision 0.5 --opacity 0.7 \
  --out-json research_project/benchmarks/results/uvt_forward_speed_probe_64_16f_224_448_tuned_v2.json
python3 research_project/benchmarks/multicam_render_timing_probe.py \
  --target-size 256 --max-frames 16 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-json research_project/benchmarks/results/multicam_render_timing_probe_mps_256_16f_uvt256_splat2048_stats.json
python3 research_project/benchmarks/multicam_train_step_timing_probe.py \
  --device mps --steps 8 --warmup-steps 2 \
  --out-json research_project/benchmarks/results/multicam_train_step_timing_probe_mps_256_16f_projection_closedform_compact_bundle_reduce.json
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 128 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 \
  --uvt-loss-scope view_sequence \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_128_16f_60s_uvt256_viewseq_loss_oldinit
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 256 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 --uvt-lr 0.01 \
  --uvt-loss-scope temporal_window --uvt-window-frames 4 \
  --uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000 \
  --uvt-depth-slope-reg 0.05 \
  --uvt-tile-t 1 --uvt-tile-capacity 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001
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
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_checkpoint_curve_smoke_16_2f_1s
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_skip_splats_smoke_16_2f_1s
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint best_heldout --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_selected_checkpoint_smoke_16_2f_1s
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_train_psnr_plateau --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_plateau_smoke_16_2f_1s
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_train_psnr_plateau \
  --uvt-select-train-psnr-plateau-patience 2 --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_plateau_patience2_smoke_16_2f_1s
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_train_psnr_gain_drop --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_gain_drop_smoke_16_2f_1s
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint best_min_train_view_psnr --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_view_selector_smoke_16_2f_1s
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_train_view_gap_collapse --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_view_gap_collapse_smoke_16_2f_1s
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_balanced_train_psnr_plateau --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_balanced_train_plateau_default_smoke_16_2f_1s
python3 research_project/benchmarks/camera_projection_parity_audit.py
```

- `uvt_pair_benchmark.py` compares the Metal UVT tile-tube pair count against
  the sliced per-frame tile-splat pair baseline on the deterministic Gate 0
  scenes.
- `training_comparison.py` fits UVT tubes and a simple independent per-frame
  Gaussian baseline to the same deterministic tiny target.
- `backward_performance_smoke.py` compares dense MPS backward against the Metal
  tile-backward autograd bridge on a bounded synthetic case.
- `backward_performance_matrix.py` runs the same comparison on a tiny smoke and
  a bounded larger local case.
- `video_fit_comparison.py` reuses the Dynaworld video loader, fits UVT and the
  local per-frame Gaussian baseline to the same fixture frames, and writes a
  contact sheet. The per-frame baseline defaults to the old random init, but
  `--per-frame-init-mode video_samples` enables a stronger pixel-sampled direct
  baseline with explicit `--per-frame-spatial-precision`, `--per-frame-opacity`,
  and `--per-frame-sample-mode`.
- `multicam_heldout_compare.py` reuses the Dynaworld multicam validation loader
  and the current V-JEPA F32 baseline split, then trains STAR-UVT worldtubes
  and a free dynamic 3DGS splat baseline for the same wall-clock budget before
  reporting train and held-out PSNR, L1, MSE, SSIM, render time, and preview
  media. For STAR-UVT multicam, `--uvt-loss-scope view_sequence` is the current
  128px setting because the renderer already produces the full sequence. At
  256px and short budgets, `--uvt-loss-scope temporal_window` with
  `--uvt-window-frames 4` improves step throughput. New reports also include
  synchronized render-only timing fields alongside the older aggregate eval
  elapsed field.
- `multicam_heldout_compare.py --uvt-checkpoint-every-steps N` stores small
  STAR worldtube snapshots during training and evaluates them after training.
  The report adds `checkpoint_curve.rows` and
  `checkpoint_curve.best_by_heldout_psnr`. This is a diagnostic for
  overtraining and schedule work, not a replacement for unbiased heldout
  reporting.
- `multicam_heldout_compare.py --skip-splats` skips the direct dynamic splat
  baseline and writes `free_dynamic_splats: null`. Use it for STAR-only
  checkpoint curves after a paired direct-splat row already exists.
- `multicam_heldout_compare.py --uvt-lr-decay-step S
  --uvt-lr-decay-factor F` keeps the default constant LR unless `S > 0`, then
  multiplies the STAR optimizer LR by `F` after `S` completed steps. Logs
  include the active `lr`.
- `multicam_heldout_compare.py --uvt-select-checkpoint best_heldout` loads the
  best STAR checkpoint from `checkpoint_curve.best_by_heldout_psnr`, evaluates
  it into a separate `star_uvt_selected` report section, and writes selected
  train/heldout media. This requires checkpoint reporting and is explicitly a
  heldout-selected diagnostic, not an unbiased test metric.
- `multicam_heldout_compare.py --uvt-select-checkpoint
  first_train_psnr_plateau` selects the first checkpoint whose train PSNR gain
  from the previous checkpoint is at or below
  `--uvt-select-train-psnr-plateau-delta` for
  `--uvt-select-train-psnr-plateau-patience` consecutive checkpoint intervals.
  The selected report is labeled `uses_heldout_for_selection: false`. This is
  a schedule diagnostic, not a final validation rule.
- `multicam_heldout_compare.py --uvt-select-checkpoint
  first_train_psnr_gain_drop` selects the previous checkpoint after train PSNR
  gain has already entered the low-gain region and the next gain drops by at
  least `--uvt-select-train-psnr-gain-drop`. It is a non-heldout schedule
  diagnostic; selected reports are labeled `uses_heldout_for_selection: false`.
- `multicam_heldout_compare.py --uvt-select-checkpoint
  best_min_train_view_psnr` selects the checkpoint with the best worst-case
  train-camera PSNR. Checkpoint rows include `train_view_eval_psnr`,
  `train_min_view_eval_psnr`, and `train_view_eval_psnr_gap`.
- `multicam_heldout_compare.py --uvt-select-checkpoint
  first_train_view_gap_collapse` selects the previous checkpoint before
  `train_view_eval_psnr_gap` falls below
  `--uvt-select-train-view-gap-collapse`. This is a non-heldout diagnostic and
  selected reports are labeled `uses_heldout_for_selection: false`.
- `multicam_heldout_compare.py --uvt-optimizer-train-views first_only
  --uvt-select-checkpoint best_train_view_psnr --uvt-select-train-view-index 1`
  trains STAR on the first train camera only and selects by the second train
  camera. This is a true train-camera dev split diagnostic.
- `uvt_forward_speed_probe.py` separates dense forward-render timing from the
  custom Metal UVT forward path for video-initialized projected tubes.
- `multicam_render_timing_probe.py` times initialized multicam STAR worldtubes
  and direct splats without a full training run. For STAR it splits full
  projection+render, projection-only, and render-only timings, and records
  Metal tile stats when using `--uvt-render-backend metal_tile`.
- `multicam_train_step_timing_probe.py` times the current multicam train step
  for STAR-UVT and fast-mac direct splats, with separate projection, render,
  loss, backward, and optimizer columns. This is timing evidence only, not a
  quality metric.
- `camera_projection_parity_audit.py` measures the pixel shift between
  DeepView `opencv_fisheye` projection and the current legacy pinhole
  approximation on the goodset cameras.

These are research iteration benchmarks, not full video-quality benchmarks.

Latest bounded backward smoke, 2026-05-10: 16 tubes, 32x32x4, 1 warmup
iteration, 2 measured iterations; dense MPS mean `16.18629200675059 ms`, Metal
tile-backward mean `36.01418749894947 ms`.

Latest bounded large local case, 2026-05-10: 64 tubes, 64x64x8, 1 warmup
iteration, 1 measured iteration; dense MPS mean `104.00879199733026 ms`, Metal
tile-backward mean `73.45674998941831 ms`.

Latest video fixture comparison, 2026-05-10: 4 frames from
`test_video_small_128_4fps.mp4` at 32x32, 8 optimization steps. UVT loss
`0.3166208863258362 -> 0.2973604202270508`; per-frame loss
`0.31666192412376404 -> 0.2972259521484375`. Contact sheet:
`research_project/benchmarks/results/video_fit_comparison_fixture.png`.

Latest fixed-step single-video overfit, 2026-05-11: 16 frames from
`test_video_small_128_4fps.mp4` at 32x32, 200 optimization steps on MPS.
With 32 UVT tubes versus 32 splats per frame, UVT reached PSNR
`20.808022022247314` in `4196.840165997855 ms` with render time
`5.274666997138411 ms`; per-frame splats reached PSNR `23.24096441268921` in
`26972.653915989213 ms` with render time `29.452292015776038 ms`. With 64 UVT
tubes versus 64 splats per frame, UVT reached PSNR `21.764323711395264` in
`9535.820624994813 ms` with render time `6.768666004063562 ms`; per-frame
splats reached PSNR `25.14953851699829` in `61065.20704101422 ms` with render
time `40.68333297618665 ms`. UVT is faster at this small overfit scale but
fits worse.

Follow-up overfit ablation, 2026-05-11: higher LR and capacity helped.
Data-sampled UVT initialization helped only after LR was raised. The best
self-contained paired run so far uses 224 UVT tubes versus 64 splats per frame
at LR `0.32`, temporal precision `0.5`, and opacity `0.7`, for 200 steps:
STAR-UVT reached PSNR `26.46265983581543` in `66625.27337501524 ms` with render
time `31.05658298591152 ms`; per-frame splats reached PSNR
`27.248921394348145` in `118614.97245798819 ms` with render time
`102.62579101254232 ms`. The best 128-tube UVT-only point used video-sampled
init, temporal precision `0.5`, opacity `0.7`, and LR `0.32`, reaching PSNR
`24.639911651611328`. A 240-tube run was slightly worse than 224 tubes at PSNR
`26.36221408843994`. A convergence bracket on the tuned 224-tube recipe found
that 340 UVT steps reached PSNR `27.101047039031982` in
`114658.05949998321 ms`, just under the tuned per-frame 200-step runtime, and
400 UVT steps reached PSNR `27.22731113433838` in `131919.97808398446 ms`,
essentially tying the per-frame PSNR but losing the train-time edge. Simple
staged LR did not help: `0.48 -> 0.16` at step 100 reached PSNR
`25.79216480255127`, and `0.32 -> 0.16` at step 150 reached PSNR
`26.341335773468018`, both below constant LR `0.32`.

64px transfer check, 2026-05-11: the tuned 224-tube recipe at 64x64, 16 frames,
200 steps reached PSNR `23.345627784729004` in `254147.23570799106 ms`, with
dense render time `81.55500001157634 ms`. A same-purpose 448-tube/100-step run
improved to PSNR `23.94777774810791` but took `350301.7887909955 ms`, with
dense render time `161.26137500395998 ms`. Capacity helps, but the dense
training path scales too slowly for larger sweeps.

Metal tile-backward overfit, 2026-05-11: the single-video benchmark can now use
`--uvt-render-backend metal_tile`. At 32x32, 16 frames, the tuned 224-tube
recipe reached PSNR `27.42915630340576` after 800 Metal steps in
`54567.61045800522 ms`, with render time `1.6061250062193722 ms`; this beats
the tuned 64-splats/frame baseline PSNR `27.248921394348145` while taking less
than half the train time. At 64x64, 224 tubes and 800 Metal steps reached PSNR
`24.250736236572266` in `128095.51870898576 ms`, render
`5.836124997586012 ms`; 1600 Metal steps only improved this to PSNR
`24.356164932250977` in `278467.5174159929 ms`, render
`1.7624159809201956 ms`. 448 tubes at the same 800-step settings was worse at
PSNR `23.577630519866943`. A Metal reducer shape bug was fixed in this pass by
row-normalizing sample buffers, filtering invalid tube ids, and gathering ids
and samples from a shared explicit position list before reduction.

64px follow-up, 2026-05-11: stratified video-sample init was worse than random
sampling, reaching PSNR `23.263163566589355` for 224 tubes, 800 Metal steps.
The previous 448-tube failure was mostly LR: 448 tubes at LR `0.16` reached
PSNR `24.879634380340576`, LR `0.24` reached PSNR `25.096933841705322`, and
LR `0.28` fell to PSNR `24.401702880859375`. Current 64px local UVT baseline is
448 tubes, random video-sampled init, LR `0.24`, temporal precision `0.5`,
opacity `0.7`, Metal tile-backward. In the same-step 64px comparison, 448 UVT
at 200 Metal steps reached PSNR `23.8846492767334` in `19297.393749991897 ms`,
render `2.980333985760808 ms`; the 64-splats/frame baseline at 200 steps
reached PSNR `23.97939920425415` in `211760.61187498271 ms`, render
`95.37416699458845 ms`. At 800 Metal steps, UVT reached PSNR
`25.096933841705322` in `104361.15416602115 ms`, already beating the splat
baseline in less than half the train time. At 1600 Metal steps, UVT reached
PSNR `25.285780429840088` in `229833.5517499945 ms`, render
`2.0606659818440676 ms`.

Forward speed probe, 2026-05-11: at 64x64, 16 frames, video-sampled tuned
initialization, external synchronized timing was dense `155.59799999270277 ms`
versus Metal `47.28181932781202 ms` for 224 tubes, and dense
`309.5513886655681 ms` versus Metal `125.89868066910033 ms` for 448 tubes.
Both cases had zero overflow and pair ratio about `0.78`. The Metal forward
path helps, and the Metal tile-backward overfit path is now the right local
iteration loop. Dense PyTorch backward is too slow for larger 64px sweeps.

Latest multicam heldout smoke, 2026-05-11: DeepView goodset train cameras
`camera_0006` and `camera_0014`, heldout camera `camera_0005`, 32x32, 2 frames,
1 second per model on CPU. STAR-UVT reached heldout PSNR `15.151495933532715`;
free dynamic splats reached heldout PSNR `4.180948257446289`. This only proves
the comparison path and media/report outputs.

Latest MPS/Metal multicam pilot, 2026-05-11: same split, 64x64, 4 frames, 5
seconds per model. STAR-UVT with 64 tubes and Metal tile backward reached
heldout PSNR `11.774863243103027`; free dynamic splats with 256 fast-mac splats
reached heldout PSNR `4.894941329956055`. This is still a small pilot, not a
full-resolution promotion result.

Latest 128px/16-frame MPS/Metal pilot, 2026-05-11: same split, 60 seconds per
model. STAR-UVT with 256 tubes and Metal tile backward reached train PSNR
`10.681523323059082` and heldout PSNR `10.493327140808105` in 126 steps. Free
dynamic splats with 2048 fast-mac splats reached train PSNR
`20.192928314208984` and heldout PSNR `10.865671157836914` in 2729 steps. Both
are below the V-JEPA F32 alpha `1/128` reference heldout PSNR `13.6248`.

128px multicam transfer check, 2026-05-11: exposing
`--uvt-init-precision-xy`, `--uvt-init-lambda-t`, and `--uvt-init-opacity`
showed that the single-video temporal-support win does not transfer directly to
the current worldtube multicam harness. With 512 tubes, init lambda_t `1.0`,
and init opacity `0.7`, STAR-UVT reached only train PSNR
`8.91626262664795` and heldout PSNR `8.603602409362793` in 123 steps, while
the same-time direct splat baseline reached heldout PSNR
`10.857621192932129`. This is worse than the previous 256-tube old-init
STAR-UVT heldout PSNR `10.493327140808105`.

128px multicam worldtube init ablation, 2026-05-11: projection audit showed
the old `init_precision_xy=30` support is broader than the single-video
screen-space recipe, while `init_precision_xy=96` gets closer in the anchor
view. The narrower init was still worse, reaching train PSNR
`8.877586841583252` and heldout PSNR `8.753138542175293` in the 60-second
pilot. Initializing from all train views with the old broad support reached
train PSNR `10.403272151947021` and heldout PSNR `10.397439002990723`, also
below the old first-view init. Direct splats in that all-train run reached
heldout PSNR `10.863123893737793`. The current multicam blocker is not solved
by narrower support or all-train-view initialization.

Camera projection parity audit, 2026-05-11: the DeepView goodset cameras are
`opencv_fisheye` with radial coefficients around `0.098, -0.018`. Compared to
the current legacy pinhole approximation, the measured grid shift is about
`8.06-8.14px` mean and `25.30-25.67px` max at 128px, and about `16.28-16.45px`
mean and `51.03-51.78px` max at 256px. This is large enough that camera-model
parity is now a first-class blocker before a bigger multicam STAR-UVT run.

Multicam view-sequence loss, 2026-05-11: `--uvt-loss-scope view_sequence`
changes STAR-UVT training from backpropagating one sampled frame to using all
frames in the already-rendered train view sequence. At 128x128, 16 frames, 60
seconds per model, 256 tubes with the old broad init reached train PSNR
`15.593097686767578` and heldout PSNR `13.423128128051758` in 135 steps. The
paired 2048-splat fast-mac baseline reached train PSNR `20.368911743164062`
and heldout PSNR `10.850052833557129` in 2840 steps. This is the first
multicam STAR-UVT heldout win over direct splats and is close to the V-JEPA F32
256px/18-minute reference heldout PSNR `13.6248`.

256px bounded view-sequence pilot, 2026-05-11: the same 256-tube recipe at
256x256, 16 frames, and 60 seconds per model did not preserve the 128px win.
STAR-UVT reached only 42 steps, train PSNR `10.676740646362305`, heldout PSNR
`10.409326553344727`, and eval render `1.7832725839980412s`. Direct splats
reached train PSNR `17.51949119567871`, heldout PSNR `10.730738639831543`, and
eval render `0.8745809590036515s` in 2225 steps. This is a step-throughput
blocker for the 256px comparison, not a reversal of the 128px sequence-loss
result.

256px temporal-window pilot, 2026-05-11: using
`--uvt-loss-scope temporal_window --uvt-window-frames 4` at the same
256x256/16-frame/60-second budget improved STAR step count to 157 and recovered
the direct-splat heldout win. STAR-UVT reached train PSNR
`12.275136947631836`, heldout PSNR `11.813445091247559`, and eval render
`1.3796198749914765s`. Direct splats reached train PSNR
`19.916349411010742`, heldout PSNR `10.738123893737793`, and eval render
`0.38876454101409763s` in 2959 steps. The representation win is real enough to
keep tuning, but the current STAR-UVT multicam render is about `3.55x` slower
than the paired `fast_mac` splat render and still trails the V-JEPA F32 heldout
PSNR `13.6248`.

256px render timing split, 2026-05-11: initialized-model timing at the same
256px/16-frame shape found STAR full projection+render totaled
`0.1808375830296427s` across the three eval sequences versus direct splats
`0.22867970800143667s`. STAR render-only totaled
`0.027068290975876153s`, while projection-only totaled
`0.1553649159905035s`. Initialized STAR Metal stats had pair ratio about
`0.83-0.86`, zero overflow, and max tile count `33-35`. A trained 60-second
temporal-window rerun with render-only timing fields told the opposite story:
STAR heldout PSNR `11.320009231567383` still beat direct splats
`10.723106384277344`, but STAR render-only eval time was
`1.3073187510017306s` versus direct splats `0.4013093340327032s`. The speed
blocker is therefore in the trained STAR path; the next diagnostic is trained
Metal tile stats and support profiles, not another initialized-only timing
probe.

Trained Metal stats and support-floor check, 2026-05-11: the trained
temporal-window model lost UVT compactness. STAR Metal stats showed pair ratio
`2.98-3.78`, unstable-tile fraction `1.0`, max tile count `174-222`, and
overflow on `8155-8192` of 8192 UVT tiles. An opt-in support-floor experiment
added `--uvt-min-precision-xy` and `--uvt-min-lambda-t`, then tested floors
equal to the old init values (`30.0`, `0.35`). That was negative: STAR heldout
PSNR fell to `9.554825782775879` versus direct splats `10.727174758911133`,
and render-only eval was still `1.2437176250386983s` versus splats
`0.3679181660118047s`. Pair ratio remained `2.34-4.53`, every active tile was
unstable, and two of three views still overflowed many tiles. The next speed
lever is depth-order/tile-load regularization, not only preserving spatial or
temporal precision floors.

Velocity regularization check, 2026-05-11: `--uvt-velocity-reg`,
`--uvt-depth-velocity-reg`, and `--uvt-position-reg` expose the STAR worldtube
regularization weights. A 256px temporal-window run with velocity reg `0.01`
and depth-velocity reg `0.1` improved STAR heldout PSNR to
`11.486005783081055` versus direct splats `10.724501609802246`, but it did not
solve speed. STAR render-only eval was `1.179607957979897s` versus direct
splats `0.4023879590095021s`; pair ratio stayed `2.50-3.68`, all active tiles
were unstable, all 8192 UVT tiles overflowed, and max tile count remained
`171-220`.

Projected tile-load regularization check, 2026-05-11:
`--uvt-tile-load-reg`, `--uvt-tile-load-target`, and `--uvt-depth-slope-reg`
expose differentiable projected support and depth-slope proxies in the sampled
train view or temporal window. A strong 20-second tile-load probe (`0.02`,
target `450`) proved the speed mechanism by dropping pair ratio below `1.0`
with zero overflow, but it underfit heldout. The softer 60-second tile-load
setting (`--uvt-tile-load-reg 0.005 --uvt-tile-load-target 1500`) restored both
heldout and speed. Adding light depth-slope pressure was the first 256px
multicam speed/quality point: `--uvt-depth-slope-reg 0.05` reached STAR heldout
PSNR `11.877435684204102` versus direct splats `10.717645645141602`, STAR
render-only eval `0.2507540419755969s` versus splats
`0.26824004197260365s`, pair ratio `0.98-1.00`, max tile count `52-54`, and
zero overflow. Stronger depth-slope reg `0.2` was negative at 60 seconds
(heldout `11.32148551940918`, render-only `0.4422728330246173s`). Active tiles
are still mostly order-unstable, so the next quality/speed gate is
order-stability or per-sample fallback cost, not raw overflow.

Projected depth-margin diagnostic, 2026-05-11: `--uvt-depth-margin-reg` and
`--uvt-depth-margin` expose a pairwise center-depth separation proxy for nearby
projected tubes. A 20-second probe on top of the current best tile-load plus
depth-slope recipe used `--uvt-depth-margin-reg 0.01 --uvt-depth-margin 0.05`.
It was not promoted: STAR heldout PSNR was `11.522964477539062`, but pair ratio
worsened to `1.08-1.12`, max tile count rose to `59-65`, and unstable-tile
fraction stayed `0.97-0.99`. Center-depth separation alone is not the
order-stability fix.

Tile-shape diagnostic, 2026-05-11: `multicam_heldout_compare.py` now exposes
`--uvt-tile-x`, `--uvt-tile-y`, `--uvt-tile-t`, and `--uvt-tile-capacity` so
Metal shader tile shape is explicit in benchmark reports. Setting
`--uvt-tile-t 1` on top of the current 256px recipe eliminated unstable tiles
and improved render-only time, but over-compressing with the old tile-load
setting lost quality. The strict 60-second run reached STAR
heldout PSNR `11.079706192016602` versus direct splats
`10.713973045349121`, with STAR render-only eval `0.1884017909760587s` versus
splats `0.4057717919931747s`. Metal stats showed pair ratio `1.72-1.75`, max
tile count `42-45`, zero overflow, and zero unstable tiles. Relaxing the
tile-load pressure to `--uvt-tile-load-reg 0.001 --uvt-tile-load-target 3000`
made `tile_t=1` the first 256px stable default. At 60 seconds, STAR reached heldout PSNR
`12.002521514892578` versus direct splats `10.725918769836426`, and render-only
eval `0.19579870899906382s` versus splats `0.43779041699599475s`. Metal stats
showed pair ratio `2.27-2.56`, max tile count `64-72`, zero overflow, and zero
unstable tiles. Relaxing target further to `5000` improved quality: STAR
heldout PSNR `12.157893180847168` versus direct splats `10.75090217590332`,
render-only eval `0.21316362501238473s` versus splats
`0.4603952500037849s`, pair ratio `3.33-3.62`, max tile count `98-115`, zero
overflow, and zero unstable tiles.

Target-`7000` boundary check, 2026-05-11: allowing more support improved PSNR
but crossed the default tile-capacity guardrail. At 20 seconds, STAR heldout
PSNR was `11.859132766723633`, with zero overflow and max tile count `114-119`.
At 60 seconds, heldout rose to `12.210857391357422`, but max tile count reached
`123-137` and one eval view overflowed `499` tiles. Raising tile capacity to
`256` removed that clipping and is the current quality default: heldout PSNR
`12.388733863830566` versus splats `10.748902320861816`, render-only eval
`0.20699129099375568s` versus splats `0.29931937501532957s`, max tile count
`114-127`, zero overflow, zero unstable tiles, and Metal buffer memory doubled
from `16.97MB` to `33.75MB`.

Target-`9000` cap-256 check, 2026-05-11: increasing support past target `7000`
was negative at 20 seconds even with tile capacity `256`. STAR completed only
58 steps, heldout PSNR fell to `10.476442337036133`, and train PSNR fell to
`10.994823932647705`. There was no overflow, but max tile count stayed
`119-126`, near the active cap edge, and the optimization underfit. Keep target
`7000` for cap-256.

Cap-256 short-budget check, 2026-05-11: the cap-256 target-`7000` recipe is a
60-second quality default, not a 20-second gate. At 20 seconds, cap-256 target
`7000` completed only 60 steps and reached heldout PSNR `10.551692962646484`;
the cap-128 target-`7000` 20-second probe completed 82 steps and reached
`11.859132766723633`. Use cap-128 for cheap gates unless the experiment
specifically targets longer-budget overflow.

Target-`6000` midpoint check, 2026-05-11: the midpoint between safe target
`5000` and overflowing target `7000` was negative at the 20-second gate. STAR
heldout PSNR was `11.611959457397461`, below target `5000` at
`11.750381469726562`, and max tile count rose to `111-127`. It did not overflow,
but it was too close to capacity without a quality gain, so no 60-second
escalation was run.

384-tube capacity check, 2026-05-11: raising STAR capacity under the relaxed
`tile_t=1` default was negative at 20 seconds. With 384 tubes, STAR heldout
PSNR was `11.04841136932373`, below the 256-tube 20-second result
`11.720141410827637`; render-only eval slowed to `0.29677454198827036s`. The
rasterizer stayed stable with zero overflow and zero unstable tiles, but max
tile count rose to `110-123`, near the default tile capacity `128`. Do not run
the 384-tube 60-second escalation until capacity or initialization changes.

256px view-sequence retry, 2026-05-11: applying the relaxed `tile_t=1` recipe
to `--uvt-loss-scope view_sequence` was negative at 20 seconds. STAR completed
only 26 steps, reached heldout PSNR `8.97468376159668`, and lost to the paired
direct splat heldout PSNR `9.148686408996582`. The Metal path stayed stable
with zero overflow and zero unstable tiles, but full-sequence training is still
too slow at 256px. Keep temporal-window training as the 256px default.

LR check, 2026-05-11: raising the relaxed `tile_t=1` recipe from LR `0.03` to
LR `0.05` was negative at the 20-second gate. STAR heldout PSNR fell to
`11.359124183654785`, below the LR `0.03` 20-second result
`11.720141410827637`, and train PSNR fell to `11.424661636352539`. The
rasterizer stayed stable with zero overflow and zero unstable tiles, but pair
ratio rose to `3.32-3.43`. LR `0.02` was also negative: heldout PSNR was
`10.951751708984375` and train PSNR was `11.512050151824951`. For that
pre-bundled-reducer recipe, keep LR `0.03`; the later faster reducer retunes
the full-budget default to LR `0.01`.

Depth-slope no-op check, 2026-05-11: under the current `tile_t=1` pinhole
projection, `depth_slope_proxy` logs as exactly `0.0` because `depth_beta` is
temporal-only and the temporal tile half-extent is zero. A no-depth-slope 20s
parity run reached heldout PSNR `11.638251304626465`, close to the
`--uvt-depth-slope-reg 0.05` 20s value `11.720141410827637`. Keep the saved
best command as-is for artifact continuity, but do not spend more sweeps on
depth-slope weight while `tile_t=1` and the current depth model are unchanged.

Train-step timing probe and compact backward patch, 2026-05-11: the first
profile on the current 256px/16-frame cap-256 default showed STAR-UVT at
`0.3036720311138197s` per profiled train step versus `0.013559588376665488s`
for paired `fast_mac` direct splats. The backward microbreakdown showed why:
the stable backward bridge emitted a fixed `67108864` sample slots for a
4-frame window while only `458991` were valid, and MPS reductions over the
mostly empty buffer took `0.1513007489265874s`.

The compact-output patch writes stable-backward samples through a device
counter and slices to the written prefix before reduction. Stable and unstable
backward smokes still pass. The same profile now shows STAR-UVT at
`0.1739891823817743s` per train step. STAR backward dropped to
`0.09416389050602447s`; its microbreakdown is `0.007699042034801096s` for
Metal sample generation, `0.0325404170434922s` for reductions over `491831`
compact samples, and `0.0021938749705441296s` for projection VJP. Worldtube
projection forward is now the other large cost at `0.0704665103694424s`.
Forward raster remains small relative to those costs.

Compact-backward 60-second quality rerun, 2026-05-11: with the same cap-256
quality command and compact stable-backward outputs, STAR completed `294` steps
instead of the previous `175`, reached train PSNR `13.66211748123169`, and
reached heldout PSNR `12.700817108154297`. The paired direct fast-mac splat
baseline completed `2254` steps, reached train PSNR `17.441619396209717`, and
reached heldout PSNR `10.724161148071289`. This strengthens the same-time
heldout win over direct splats but still trails the V-JEPA F32 heldout
reference `13.6248`. It also reverses the previous render-only speed claim:
STAR render-only eval was `0.30119724897667766s` versus direct splats
`0.24550599994836375s`. STAR Metal stats had zero overflow and zero unstable
tiles, but pair ratio rose to `3.78-4.16` and max tile count reached `123-135`.
The speed next step is projection, compact reduction, and trained support
control, not a blind forward-raster rewrite.

Closed-form projection patch, 2026-05-11: `project_world_tubes_pinhole(...)`
now computes the 2x2 screen covariance and inverse explicitly instead of using
batched matrix multiplications plus `torch.linalg.inv`. A formula-equivalence
probe reported `ma` max diff `0.0`, `q_uvt` max diff
`2.9802322387695312e-08`, and `depth0` max diff `0.0`; a non-identity camera
probe reported `ma` max diff `1.52587890625e-05`, `q_uvt` max diff
`1.1920928955078125e-07`, and zero `depth0` / `depth_beta` diff. Pinhole and
`CameraSpec` projection smokes still pass. The train-step timing probe output
`multicam_train_step_timing_probe_mps_256_16f_projection_closedform.json`
shows STAR mean step `0.10214632287534187s`, projection
`0.0023792186329956166s`, render `0.0040640363658894785s`, and backward
`0.08975485964037944s`. Projection is no longer the dominant cost; compact
backward/reduction is.

Closed-form projection 60-second quality rerun, 2026-05-11: the same cap-256
quality command now completed `410` STAR steps, reached train PSNR
`14.423624038696289`, and reached heldout PSNR `12.778368949890137`. The
paired direct fast-mac splat baseline completed `3179` steps, reached train
PSNR `20.28785991668701`, and reached heldout PSNR `10.700675964355469`. STAR
render-only eval was `0.08381970797199756s` versus direct splats
`0.36649591801688075s`, and heldout render-only was `0.021688583015929908s`
versus `0.07554204197367653s`. STAR Metal stats stayed stable: pair ratio
`3.02-3.23`, max tile count `103-119`, zero overflow, and zero unstable tiles.
This restores the speed claim versus direct splats, but STAR still trails the
V-JEPA F32 heldout reference `13.6248`.

Bundled compact reducer and LR retune, 2026-05-11: the compact-output backward
still reduced `ma`, `q_uvt`, opacity, and color separately on MPS. Bundling the
13 gradient channels into one compact `index_add_` reduced the train-step probe
from STAR mean step `0.10214632287534187s` to `0.06733408838044852s`; backward
dropped from `0.08975485964037944s` to `0.05865076563350158s`, and compact
reduction dropped from `0.030503582034725696s` to
`0.008818959002383053s`. Render is now a small forward phase at
`0.0024019792545004748s` versus direct splats `0.005343109376553912s`; the
dominant remaining phase is STAR backward, not projection or forward raster.

The faster reducer changed the effective optimizer stability boundary. LR
`0.03` collapsed at step `180` in the first full rerun. A 20-second bracket
made LR `0.02` look promising (`13.122428894042969` heldout versus
`12.534915924072266` for LR `0.01`), but the 60-second LR `0.02` escalation was
also unstable: step `190` still had finite loss with `tile_load_proxy: NaN`,
step `200` was the first NaN loss, final heldout PSNR was only
`7.12491512298584`, and final Metal stats had zero active tile pairs. The
midpoint LR `0.015` was stable but worse than LR `0.01`: STAR completed `879`
steps, reached train PSNR `16.549213409423828`, heldout PSNR
`13.005823135375977`, and render-only eval `0.03621062601450831s`. Metal stats
were clean but larger than LR `0.01`: pair ratio `1.80-2.30`, max tile count
`76-83`, zero overflow, and zero unstable tiles. LR `0.01` is the current
stable cap-256 full-run setting. A support-relaxation check at LR `0.01` with
tile-load target `9000` was also negative. It stayed stable and clean, with
pair ratio `1.55-2.15`, max tile count `69-74`, zero overflow, and zero
unstable tiles, but it completed only `790` STAR steps and fell to heldout PSNR
`12.860218048095703` despite higher train PSNR `17.124666213989258`. This
rejects raw target relaxation as the next quality lever. At target `7000`, the
current best 60-second run reached STAR train PSNR `16.730055809020996`,
heldout PSNR
`13.20147705078125`, and `849` steps. The paired direct fast-mac splat baseline
reached train PSNR `19.02044677734375`, heldout PSNR
`10.722965240478516`, and `2568` steps. STAR render-only eval was
`0.046138084086123854s` versus splats `0.29644233302678913s`, and heldout
render-only was `0.014686542039271444s` versus `0.08152529201470315s`. STAR
Metal stats stayed stable: pair ratio `1.59-2.08`, max tile count `65-70`,
zero overflow, and zero unstable tiles. This is the current best legacy-pinhole
60-second local artifact, still `0.4233229492187494` PSNR below the V-JEPA F32
heldout reference `13.6248`.

Dataset-lens STAR projection diagnostic, 2026-05-11: DeepView goodset cameras
are `opencv_fisheye`, and the camera-projection audit measured the legacy
pinhole approximation at roughly `16px` mean and `51px` max grid shift at
256px. `--uvt-camera-projection dataset_lens` keeps the direct-splat baseline
unchanged but projects STAR worldtubes through the dataset lens model. It is
therefore a camera-model diagnostic, not a replacement for the legacy-pinhole
comparison row.

The 60-second dataset-lens run reached:

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

Metal stats stayed clean: pair ratio `2.10-2.18`, max tile count `70-78`,
zero overflow, and zero unstable tiles. Read: the fisheye-aware STAR path gives
another `0.29526329040527344` heldout PSNR over the legacy-pinhole STAR run and
is still much faster than direct splats. The measured 60-second render-only
time is slightly faster than the legacy-pinhole STAR row despite higher tile
pressure, so this does not point to a generic forward-raster rewrite. It points
to camera-contract clarity and remaining backward/sample reduction work.

Lens-aware direct-splat baseline, 2026-05-11: the benchmark now also accepts
`--splat-camera-projection dataset_lens`. For direct splats this builds
`CameraSpec` values with the DeepView `opencv_fisheye` lens and uses
`render.camera_projection='camera_model'` before the projected Gaussians enter
fast-mac. That makes the baseline slower than legacy pinhole, but gives the
camera-contract comparison we needed.

The 20-second gate reached STAR heldout PSNR `13.600945472717285` versus
direct splats `8.922689437866211`; STAR render-only eval was
`0.043396833061706275s` versus direct splats `0.5898302079876885s`.

The 60-second lens-aware row reached:

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

Metal stats stayed clean: pair ratio `2.21-2.33`, max tile count `68-74`,
zero overflow, and zero unstable tiles. Read: this is the current cleanest
local comparison. STAR beats lens-aware direct splats by `2.4444656372070312`
heldout PSNR and about `10.6x` render-only eval speed. It also clears the
V-JEPA F32 heldout reference `13.6248` by `0.00819751281738238` dB. The V-JEPA
margin is too small to call promotion done, but it is the first local
same-split V-JEPA heldout crossing.

Seed-1 repeat did not reproduce the crossing. Under the same 60-second
lens-aware settings, STAR reached heldout PSNR `12.9697904586792`, direct
splats reached `11.243672370910645`, STAR render-only eval was
`0.04029629105934873s`, and direct splats render-only eval was
`0.5776280419668183s`. Metal stats stayed clean with pair ratio `2.13-2.37`,
max tile count `69-73`, zero overflow, and zero unstable tiles. Read: speed is
robust; V-JEPA parity is not robust yet.

Grid-init seed-1 repeat, 2026-05-11: `--uvt-init-sampling grid` makes the
worldtube initializer sample deterministic image-grid pixels instead of random
pixels. The 60-second lens-aware seed-1 row reached STAR heldout PSNR
`13.179410934448242` versus direct splats `11.206615447998047`; STAR render-only
eval was `0.035952959035057575s` versus direct splats `0.5049242499517277s`.
Metal stats stayed clean with pair ratio `2.22-2.38`, max tile count `71-75`,
zero overflow, and zero unstable tiles. Read: grid init improves the seed-1
miss by about `0.21` dB but still does not reproduce the V-JEPA crossing, so
the next quality gate should be deterministic schedule or stronger multi-view
initialization, not plain grid coverage alone.

All-train grid-init repeat, 2026-05-11: combining `--uvt-init-views all_train`
with `--uvt-init-sampling grid` is the best seed-1 repeatability result so far.
The 20-second gate reached STAR heldout PSNR `13.527872085571289` versus direct
splats `9.138171195983887`. The 60-second escalation reached STAR heldout PSNR
`13.52819538116455` versus direct splats `11.074682235717773`; STAR render-only
eval was `0.11616870801663026s` versus direct splats `0.5125780410016887s`.
Metal stats stayed clean with pair ratio `2.42-2.52`, max tile count `78-81`,
zero overflow, and zero unstable tiles. Read: all-train grid init recovers
about `0.56` dB over seed-1 random first-view init but still misses V-JEPA
`13.6248`; the next quality lever should be train schedule or motion-aware
multi-view init.

Deterministic train-schedule probe, 2026-05-11: `--uvt-train-schedule cycle`
cycles STAR train views and temporal-window starts instead of sampling them.
The CPU smoke wrote `train_schedule: cycle` into the report. The 20-second
all-train grid cycle gate reached STAR heldout PSNR `13.28015422821045` versus
direct splats `9.227516174316406`; STAR render-only eval was
`0.038111125002615154s` versus direct splats `0.3976892919745296s`. Metal stats
stayed clean with pair ratio `2.47-2.65`, max tile count `74-79`, zero
overflow, and zero unstable tiles. Read: cycle scheduling is worse than the
same all-train grid init with random training samples at 20 seconds, so do not
escalate it as-is.

All-train grid LR probe, 2026-05-11: LR `0.015` is negative for the current
all-train grid seed-1 setting. The 20-second gate reached STAR heldout PSNR
`13.287956237792969` versus `13.527872085571289` for LR `0.01`; STAR
render-only eval was `0.04351187701104209s`, Metal stats were clean, and pair
ratio was `2.64-2.74`. Do not escalate LR `0.015` to 60 seconds.

All-train grid compactness probe, 2026-05-11: lowering tile-load target from
`7000` to `5000` improves compactness but loses heldout quality. The 20-second
target-`5000` gate reached STAR heldout PSNR `13.459638595581055` versus
`13.527872085571289` for target `7000`; STAR render-only eval was
`0.0383758339448832s`, and pair ratio fell to `2.26-2.36`. Keep target `7000`
as the quality setting.

Time-distributed all-frames init, 2026-05-11: `--uvt-init-frames all` splits
the tube budget across train views and source frames, samples each tube color
from that frame, and initializes `t0` to the centered frame time. This fixes
the old multi-view init mismatch where all tubes inherited frame-0 appearance
while sitting at sequence-center time. The CPU smoke
`multicam_heldout_compare_time_init_smoke_16_2f_1s` wrote `init_frames: all`
into the report.

Seed-1 all-train grid all-frames results:

```text
20s: STAR heldout 13.768306732177734, train 15.55235481262207,
     steps 346, render-only 0.04166583297774196s;
     direct splats heldout 8.622618675231934, render-only 0.3382836260134354s.

30s: STAR heldout 13.726262092590332, train 16.17089080810547,
     steps 530, render-only 0.041071167041081935s;
     direct splats heldout 9.915854454040527, render-only 0.4364799159229733s.

60s: STAR heldout 13.564573287963867, train 16.77016544342041,
     steps 785, render-only 0.08123612502822652s;
     direct splats heldout 11.048260688781738, render-only 0.42233533307444304s.
```

Read: the 20s and 30s seed-1 rows clear the V-JEPA F32 heldout reference
`13.6248`, while the 60s row falls back under it. This makes the old
same-budget assumption suspect: for STAR-UVT, longer local optimization can
increase train PSNR while degrading heldout-camera PSNR.

Seed-0 repeat, 2026-05-11: the 20-second all-train grid all-frames row reached
STAR heldout PSNR `13.769630432128906`, train PSNR `15.741607666015625`,
steps `365`, and render-only eval `0.03941354202106595s`. Direct splats reached
heldout PSNR `8.91901969909668` and render-only eval `0.29203037498518825s`.
Metal stats were clean with pair ratio `3.63-3.69`, max tile count `94-102`,
zero overflow, and zero unstable tiles. The 30-second seed-0 row was worse:
STAR heldout PSNR `13.600011825561523`, direct splats `9.411006927490234`, and
STAR render-only eval `0.04163579299347475s` versus splats
`0.2929964159266092s`.

Seed-2 repeat, 2026-05-11: the 20-second all-train grid all-frames row reached
STAR heldout PSNR `13.764396667480469`, train PSNR `15.683393478393555`,
steps `333`, and render-only eval `0.041590416978579015s`. Direct splats
reached heldout PSNR `8.697443962097168` and render-only eval
`0.46188579098088667s`. Metal stats were clean with pair ratio `3.61-3.80`,
max tile count `83-91`, zero overflow, and zero unstable tiles.

Seed-0 shorter-budget bracket, 2026-05-11: the same all-train grid all-frames
recipe reached STAR heldout PSNR `12.681236267089844` at 10 seconds and
`13.669918060302734` at 15 seconds. The 10-second row completed `176` STAR
steps with render-only eval `0.04378599900519475s`; the 15-second row completed
`261` STAR steps with render-only eval `0.04481595807010308s`. Direct splats
reached heldout PSNR `7.715583801269531` at 10 seconds and
`8.104644775390625` at 15 seconds. Read: 15 seconds already clears V-JEPA on
seed 0, but 20 seconds is materially better; the useful region is not below
10 seconds.

Read: all-frames init is the right representation lever. The 20-second recipe
now crosses V-JEPA on all three tested seeds, but 30-second seed 0 and
60-second seed 1 show that longer local training can hurt heldout. The
shorter-budget bracket says the current sweet spot is roughly 15-20 seconds,
with 20 seconds still the best tested point. The next gate should be an
explicit early-stop/heldout selector or a schedule that preserves the 20-second
generalization point. It should not start with a forward rasterizer rewrite.

Checkpoint-curve smoke, 2026-05-11: `--uvt-checkpoint-every-steps 1` passed on
the 16px/2-frame CPU smoke. The report included two checkpoint rows and
`checkpoint_curve.best_by_heldout_psnr`, proving the JSON/report path can carry
an opt-in STAR checkpoint curve.

Skip-splats checkpoint smoke, 2026-05-11: `--skip-splats` passed on the same
16px/2-frame CPU shape and wrote `free_dynamic_splats: null`. This keeps
longer STAR-only checkpoint diagnostics from retraining a direct-splat baseline
that is not being remeasured.

Seed-0 30-second checkpoint curve, 2026-05-11: rerunning the all-train grid
all-frames recipe with `--uvt-checkpoint-every-steps 50` confirmed the mid-run
peak. The final STAR checkpoint reached heldout PSNR `13.518027305603027` at
step `489`, below V-JEPA. The best checkpoint was step `300`, elapsed
`18.632994499988854s`, with train PSNR `15.358994007110596`, heldout PSNR
`13.730653762817383`, and render-only eval `0.0463867480866611s`. Later
checkpoints declined: step `350` heldout `13.652517318725586`, step `400`
heldout `13.619556427001953`, and step `450` heldout `13.595906257629395`.
Read: a longer final-run miss can contain a V-JEPA-crossing STAR checkpoint;
the next schedule work should preserve the 18-20s region rather than extending
fixed-budget training.

Seed-1 60-second STAR-only checkpoint curve, 2026-05-11: rerunning the same
all-train grid all-frames recipe with `--skip-splats` and
`--uvt-checkpoint-every-steps 100` showed the same mid-run peak without paying
for another direct-splat train. The final STAR checkpoint reached heldout PSNR
`13.354101181030273` at step `1019`, while the best checkpoint was step `300`,
elapsed `17.160626166965812s`, with train PSNR `15.61048173904419`, heldout
PSNR `13.75400447845459`, and render-only eval `0.036524292023386806s`. The
curve then declined: step `400` heldout `13.74360466003418`, step `500`
heldout `13.627070426940918`, step `600` heldout `13.620699882507324`, and
step `1000` heldout `13.319355010986328`. Read: the seed-1 long run contains a
strong V-JEPA-crossing STAR checkpoint, but the final 60-second checkpoint is
worse than V-JEPA. This reinforces schedule/early-stop as the next gate.

Seed-2 40-second STAR-only checkpoint curve, 2026-05-11: the same recipe with
`--seed 2`, `--train-seconds 40`, `--skip-splats`, and checkpoints every 100
steps peaked later. The final STAR checkpoint reached heldout PSNR
`13.631431579589844` at step `629`, barely above the V-JEPA F32 heldout
reference, while the best checkpoint was step `500`, elapsed
`32.050383166992106s`, with train PSNR `16.440028190612793`, heldout PSNR
`13.988276481628418`, and render-only eval `0.037308208004105836s`. The row
then declined by step `600` to heldout `13.693096160888672`. Metal stats stayed
clean with pair ratio `3.14-3.41`, max tile count `75-84`, zero overflow, and
zero unstable tiles. Read: seed 2 confirms the broader overtraining pattern,
but not a fixed step-300 cutoff. The next gate should preserve the best
validation-shaped checkpoint, not blindly stop at one wall-clock.

Seed-1 LR-decay schedule bracket, 2026-05-11: adding
`--uvt-lr-decay-step 300` is a partial fix for the long-run seed-1 decay. With
decay factor `0.2`, the final 60-second checkpoint reached heldout PSNR
`13.643348693847656` at step `784`, improving over the no-decay final
`13.354101181030273`; its best checkpoint was step `600` at heldout
`13.743600845336914`. Final aggregate render-only eval was noisy and slow at
`0.11957950098440051s`, though checkpoint render-only rows stayed around
`0.044-0.051s`. With decay factor `0.05`, the final checkpoint improved to
heldout PSNR `13.692363739013672` at step `1003`, with render-only eval
`0.04503662494244054s`, and the best checkpoint was step `500` at heldout
`13.71772575378418`. Metal stats stayed clean in both rows: zero overflow and
zero unstable tiles. Read: LR decay after the observed seed-1 peak preserves a
V-JEPA-crossing final checkpoint, especially at factor `0.05`, but it still
does not match the best selected checkpoint. This makes schedule work useful,
not solved.

Seed-2 later-peak LR-decay check, 2026-05-11: using the seed-2 checkpoint curve
to decay after the later step-500 peak also helps the final checkpoint. With
`--uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05`, the 60-second run
finished at step `756`, train PSNR `17.018577575683594`, heldout PSNR
`13.81359577178955`, and aggregate render-only eval
`0.08080591692123562s`. The best checkpoint was step `500`, elapsed
`31.662656374974176s`, with heldout PSNR `13.909360885620117`. Metal stats
were clean: pair ratio `3.00-3.13`, max tile count `75-80`, zero overflow, and
zero unstable tiles. Read: decaying at the observed per-seed peak improves the
final model, but it still does not preserve the selected best checkpoint.

Paired seed-2 LR-decay comparison, 2026-05-11: rerunning the seed-2
`--uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05` setting with the
lens-aware 2048-splat `fast_mac` baseline gives the current clean paired
schedule row. STAR final heldout PSNR was `13.84060287475586` at step `996`;
direct dynamic splats reached heldout PSNR `11.156550407409668` at step
`2326`. STAR render-only eval was `0.043013749993406236s` versus direct splats
`0.3524511669529602s`. STAR's selected best checkpoint was step `800`, elapsed
`48.05842208303511s`, heldout PSNR `13.873014450073242`, and render-only eval
`0.032829875010065734s`. Metal stats stayed clean with pair ratio `2.88-3.04`,
max tile count `76-81`, zero overflow, and zero unstable tiles. Read: this is
a same-budget STAR win over both the paired direct splats and the V-JEPA F32
heldout row, but it remains a local harness row and selected-checkpoint quality
is still better than the final checkpoint.

Selected-checkpoint report smoke, 2026-05-11: `--uvt-select-checkpoint
best_heldout` passed on the 16px/2-frame CPU smoke with `--skip-splats`. The
report wrote `star_uvt_selected.selector: best_heldout`,
`uses_heldout_for_selection: true`, and selected train/heldout preview PNGs and
MP4s. Read: this gives future schedule experiments a first-class selected-model
artifact while keeping final-checkpoint metrics separate.

Paired seed-1 selected-checkpoint MPS artifact, 2026-05-11: rerunning the
seed-1 `--uvt-lr-decay-step 300 --uvt-lr-decay-factor 0.05` schedule with
`--uvt-select-checkpoint best_heldout` and the lens-aware direct-splat baseline
wrote final, selected, and direct media in one report. STAR final heldout PSNR
was `13.758882522583008` at step `1031`; STAR selected heldout PSNR was
`13.818532943725586` at step `300`; direct dynamic splats reached heldout PSNR
`11.15761947631836` at step `2271`. STAR final render-only eval was
`0.04524075100198388s`, STAR selected render-only eval was
`0.03815650095930323s`, and direct splats render-only eval was
`0.3847669999813661s`. Final and selected STAR Metal stats were clean, with
zero overflow and zero unstable tiles. Read: the selected-artifact path works
on the real MPS recipe; the exact selected step and PSNR vary across MPS reruns,
so report the saved artifact path with each row.

Paired seed-2 selected-checkpoint MPS artifact, 2026-05-11: rerunning the
seed-2 `--uvt-lr-decay-step 500 --uvt-lr-decay-factor 0.05` schedule with
`--uvt-select-checkpoint best_heldout` and the lens-aware direct-splat baseline
wrote final, selected, and direct media in one report:

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
tile count `74-78`, zero overflow, and zero unstable tiles. Read: the latest
selected STAR artifact is not render-slower than direct dynamic splats. It is
about `11.7x` faster by synchronized render-only eval while also clearing the
local V-JEPA F32 heldout row, but the selected metric is still heldout-selected
and must stay separate from unbiased test claims.

Seed-0 LR-decay schedule repeat, 2026-05-11: the STAR-only run used the
all-train grid all-frames recipe, `--uvt-lr-decay-step 300`,
`--uvt-lr-decay-factor 0.05`, `--uvt-select-checkpoint best_heldout`, and
`--skip-splats`:

```text
STAR final:
  step             970
  train PSNR       16.02366304397583
  heldout PSNR     13.81613826751709
  render-only      0.040288584015797824s

STAR selected:
  step             600
  elapsed          37.35523024998838s
  train PSNR       15.836221694946289
  heldout PSNR     13.87098217010498
  render-only      0.04048433306161314s
```

Final and selected Metal stats stayed clean: max tile count `94-102`, pair
ratio `3.63-4.03`, zero overflow, and zero unstable tiles. Read: the tuned
LR-decay schedule now has V-JEPA-crossing final checkpoints on seeds 0, 1, and
2, but this seed-0 row skipped direct splats and the selected checkpoint uses
heldout-camera selection. Treat it as schedule evidence, not the final unbiased
selector.

Train-plateau selected-checkpoint diagnostic, 2026-05-11:
`--uvt-select-checkpoint first_train_psnr_plateau` selects the first checkpoint
where train PSNR gain from the previous checkpoint is at or below
`--uvt-select-train-psnr-plateau-delta`. The CPU smoke
`multicam_heldout_compare_train_plateau_smoke_16_2f_1s` passed and wrote
`uses_heldout_for_selection: false`.

The first real seed-2 MPS artifact used the same tuned STAR recipe as the
selected-checkpoint rows, with `--skip-splats` and plateau delta `0.5`:

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
  uses heldout     false

Heldout-best checkpoint in same curve:
  step             500
  heldout PSNR     13.94494915008545
```

Final and selected STAR Metal stats stayed clean: zero overflow and zero
unstable tiles. Read: a train-only plateau rule improves over the final
checkpoint and clears the V-JEPA F32 heldout reference without heldout-camera
selection. It does not fully recover the heldout-best checkpoint on this run,
so the next selector work should tune or replace the plateau rule rather than
claiming unbiased selection solved.

The seed-1 repeat used the analogous `300 -> 0.05x` LR-decay schedule with
`--skip-splats` and plateau delta `0.5`:

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
  uses heldout     false

Heldout-best checkpoint in same curve:
  step             700
  heldout PSNR     13.826448440551758
```

Read: the same non-heldout selector also clears V-JEPA on seed 1, but it is
slightly worse than the final checkpoint and misses the heldout-best checkpoint
by about `0.055` dB. This strengthens the narrow claim that the train-plateau
rule is V-JEPA-crossing on two tested seeds, while weakening any claim that it
is the right final early-stop rule.

Train-plateau patience-2 diagnostic, 2026-05-11:
`--uvt-select-train-psnr-plateau-patience 2` requires two consecutive
checkpoint-to-checkpoint train-PSNR gains at or below the plateau delta. The
CPU smoke `multicam_heldout_compare_train_plateau_patience2_smoke_16_2f_1s`
passed and wrote `select_train_psnr_plateau_patience: 2`.

Seed-2 MPS STAR-only result:

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
  plateau run      2

Heldout-best checkpoint in same curve:
  step             600
  heldout PSNR     13.855948448181152
```

Seed-1 MPS STAR-only result:

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
  plateau run      2

Heldout-best checkpoint in same curve:
  step             400
  heldout PSNR     13.726997375488281
```

Read: patience `2` is useful but not solved. It improved seed 2 versus final
and missed the heldout-best checkpoint by only about `0.015` dB, but it moved
seed 1 past that run's heldout peak. Both selected rows clear the V-JEPA F32
reference without heldout-camera selection and kept clean selected Metal stats:
zero overflow and zero unstable tiles. The next selector should use a smoother
train curve or a lightweight rendered validation subset, not just a larger
plateau patience.

Train-gain-drop selected-checkpoint diagnostic, 2026-05-11:
`--uvt-select-checkpoint first_train_psnr_gain_drop` selects the previous
checkpoint after train PSNR gain has entered the low-gain region and the next
gain falls by at least `--uvt-select-train-psnr-gain-drop`. The CPU smoke
`multicam_heldout_compare_train_gain_drop_smoke_16_2f_1s` passed and wrote
`uses_heldout_for_selection: false`.

Seed-2 MPS STAR-only result:

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
  next gain        0.3444557189941406
  gain drop        0.15789222717285156

Heldout-best checkpoint in same curve:
  step             500
  heldout PSNR     13.904694557189941
```

Seed-1 MPS STAR-only result:

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
  next gain        0.09837532043457031
  gain drop        0.22542142868041992

Heldout-best checkpoint in same curve:
  step             500
  heldout PSNR     13.735674858093262
```

Seed-0 MPS STAR-only result:

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

Paired seed-2 report against direct dynamic splats:

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

Paired seed-1 report against direct dynamic splats:

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

Read: this is the best non-heldout selector candidate so far. It exactly
selected the heldout-best checkpoint on one seed-2 STAR-only run, missed the
seed-0 and seed-1 heldout-best checkpoints by about `0.014` dB, and stayed
above the V-JEPA F32 heldout reference on all three STAR-only repeats. The
paired seed-2 report keeps the same conclusion against direct dynamic splats:
selected STAR beats direct splats by about `2.70` dB heldout PSNR and renders
about `19.5x` faster by synchronized render-only timing. The paired seed-1
repeat is weaker but confirms the selector: selected STAR beats direct splats
by about `2.68` dB and renders about `10.2x` faster by synchronized render-only
timing. Selected Metal stats stayed clean: zero overflow and zero unstable
tiles. Freeze gain-drop as the current reporting selector for the next
scale/full-resolution probe; do not treat it as a production default.

512px same-budget scale probe, 2026-05-11: same seed-2 recipe, same 60-second
local budget, gain-drop frozen.

```text
STAR final / selected:
  step             70
  train PSNR       9.331888198852539
  heldout PSNR     9.205381393432617
  train loop       68.71951654099996s
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

Read: the naive 512px scale-up is rejected. STAR remains faster in selected
render-only timing, but it is too step-starved at the same local budget and
loses heldout PSNR by about `1.78` dB to direct splats. Do not report this as
full-resolution parity. The next 512px attempt needs a changed scale strategy:
longer STAR budget, multiscale training, crop/window training, or train-step
throughput work.

512px window-1 scale-strategy probe, 2026-05-11: same seed-2 recipe and
60-second local budget as the rejected 512px row, but with
`--uvt-window-frames 1`. The first STAR-only pass showed the signal; this is
the formal paired rerun against direct dynamic splats.

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
  render-only      0.5262394170001699s
```

Read: window-1 rescues the 512px scale strategy. The final STAR checkpoint is
about `2.94` dB above the paired 512px direct-splat row and renders about
`4.6x` faster by render-only timing. The non-heldout selected row also clears
the 256px V-JEPA F32 reference `13.6248`, though it still misses the run's
heldout-best checkpoint by about `0.051` dB. This makes the next 512px work
scale-aware checkpoint selection or multiscale/window policy, not a
first-priority rasterizer rewrite.

512px stricter gain-drop selector diagnostic, 2026-05-11:
`--uvt-select-train-psnr-plateau-delta 0.1` with the same STAR-only
512px/window-1 recipe.

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

Read: lowering the gain threshold delays selection, but does not improve the
selected metric. It still clears the V-JEPA F32 reference, yet final/best-train
is better on this 512px/window-1 curve. Do not spend the next run on another
plain threshold tweak.

512px seed-1 window-1 repeat and LR-decay bracket, 2026-05-11:

```text
Paired seed-1, LR decay step 300:
  STAR final       step 873, heldout PSNR 13.494588851928711, render 0.3712127500000406s
  STAR selected    step 400, heldout PSNR 13.415751457214355, render 0.1480234160001146s
  STAR heldout-best step 800, heldout PSNR 13.498263359069824
  Direct splats    step 1587, heldout PSNR 10.3926362991333, render 0.4046094999998786s

STAR-only seed-1, LR decay step 500:
  STAR final       step 963, heldout PSNR 13.576004028320312, render 0.34020658299959905s
  STAR selected    step 600, heldout PSNR 13.557477951049805, render 0.13215666599990072s
  STAR heldout-best step 700, heldout PSNR 13.58298397064209

STAR-only seed-1, LR decay step 700:
  STAR final       step 730, heldout PSNR 13.397587776184082, render 0.4030705410000337s
  STAR selected    step 600, heldout PSNR 13.414778709411621, render 0.14510950000021694s
  STAR heldout-best step 400, heldout PSNR 13.490400314331055
```

Read: seed 1 repeats the strong direct-splat win, but not the V-JEPA crossing.
The best tested seed-1 512px/window-1 row is LR decay step `500`, with
heldout-best PSNR `13.58298397064209`, still below the V-JEPA F32 reference
`13.6248`. Step `700` is worse, so the next 512px seed-robustness branch should
change capacity, support/window policy, or multiscale/crop training rather than
keep nudging the LR-decay step.

512px window-1 tube-capacity bracket, 2026-05-11:

```text
STAR-only seed-1, 384 tubes:
  STAR final       step 928, heldout PSNR 13.640532493591309, checkpoint render 0.10645100000010643s
  STAR selected    step 600, heldout PSNR 13.572845458984375, render 0.10984833200018329s

STAR-only seed-2, 384 tubes:
  STAR final       step 946, heldout PSNR 13.4086275100708, checkpoint render 0.12059308300013072s
  STAR selected    step 600, heldout PSNR 13.339370727539062, render 0.12307370799999262s

STAR-only seed-1, 320 tubes:
  STAR final       step 732, heldout PSNR 13.682265281677246, checkpoint render 0.10670079200053806s
  STAR selected    step 600, heldout PSNR 13.637592315673828, render 0.10943916599944714s
  STAR heldout-best step 400, heldout PSNR 13.769192695617676, render 0.11296041700006754s

STAR-only seed-2, 320 tubes:
  STAR final       step 837, heldout PSNR 13.637543678283691, checkpoint render 0.10442929100008769s
  STAR selected    step 600, heldout PSNR 13.598714828491211, render 0.10498941599962563s

STAR-only seed-0, 320 tubes, LR decay step 500:
  STAR final       step 950, heldout PSNR 13.542135238647461, render 0.12215708199983055s
  STAR selected    step 600, heldout PSNR 13.437091827392578, render 0.11039912500018545s
  STAR heldout-best step 400, heldout PSNR 13.70832633972168, render 0.13102466600003027s

STAR-only seed-0, 256 tubes, LR decay step 500:
  STAR final       step 940, heldout PSNR 13.580879211425781, render 0.10690020900028685s
  STAR selected    step 600, heldout PSNR 13.636795043945312, render 0.11646866799992495s
  STAR heldout-best step 600, heldout PSNR 13.636795043945312, render 0.11571520699999382s

STAR-only seed-0, 320 tubes, LR decay step 400:
  STAR final       step 994, heldout PSNR 13.612069129943848, render 0.1571206259995961s
  STAR selected    step 500, heldout PSNR 13.586051940917969, render 0.12204341700044097s
  STAR heldout-best step 400, heldout PSNR 13.652498245239258, render 0.1324513740000839s
```

Read: 384 tubes is rejected as a default because it fixes seed 1 but breaks
seed 2. The 320-tube middle point is promising because seed 1 clears V-JEPA on
final and selected rows and seed 2 clears on final, but it still underperforms
the earlier 256-tube seed-2 512px row. Seed 0 makes the tradeoff sharper: 320
tubes gives a better heldout-best peak than 256 tubes, but the non-heldout
selector misses it badly; 256 tubes has the cleaner current selected row.
Moving 320-tube seed0 decay from step `500` to `400` does not fix the selector
miss or preserve the peak. All capacity runs kept zero overflow and zero
unstable tiles, with checkpoint render-only timing around `0.104-0.132s`; the
next 512px issue is seed-robust scale policy/selection, not a forward rasterizer
rewrite.

512px seed-0 320-tube window-2 policy check, 2026-05-11:

```text
STAR-only seed-0, 320 tubes, window 2:
  STAR final       step 536, heldout PSNR 13.391561508178711, render 0.33027225000114413s
  STAR selected    step 500, heldout PSNR 13.422961235046387, render 0.10342058300011558s
  STAR heldout-best step 400, heldout PSNR 13.558874130249023, render 0.10766366600000765s
  max tile / pair  89 / 2.8609017561213266
  overflow/unstable 0 / 0.0
```

Read: window 2 is rejected for this 512px seed-0 320-tube branch. It fixes
neither selected quality nor oracle peak quality; it cuts step throughput from
the window-1 run's `950` steps to `536` and falls well below the window-1
heldout-best peak `13.70832633972168`. The clean Metal stats reinforce that
the miss is support/window policy and training throughput, not a rasterizer
failure.

512px seed-0 320-tube hard LR-drop policy check, 2026-05-11:

```text
STAR-only seed-0, 320 tubes, window 1, decay 400 -> 0.005x:
  STAR final       step 1000, heldout PSNR 13.599885940551758, render 0.12993799899868463s
  STAR selected    step 1000, heldout PSNR 13.599885940551758, render 0.13297337400126708s
  STAR heldout-best step 1000, heldout PSNR 13.599885940551758, render 0.13298595900141663s
  max tile / pair  103 / 3.2112812143984426
  overflow/unstable 0 / 0.0
```

Read: hard scalar decay is rejected too. It avoids a catastrophic final
collapse, but it does not preserve the useful seed-0 320-tube shoulder; it
lands below the earlier window-1 heldout-best `13.70832633972168` and below the
softer step-400 decay run's heldout-best `13.652498245239258`. The render path
is still not the blocker: the selected row is clean and stays in the same
`0.10-0.13s` render-only band as the other healthy 512px STAR rows.

512px seed-0 320-tube temporal-floor support check, 2026-05-11:

```text
STAR-only seed-0, 320 tubes, min lambda_t 0.7:
  STAR final       step 1205, heldout PSNR 13.514815330505371
  STAR selected    step 500, heldout PSNR 13.336383819580078, render 0.10821341699738696s
  STAR heldout-best step 1205, heldout PSNR 13.514815330505371
  max tile / pair  76 / 3.7183004841728744
  overflow/unstable 0 / 0.0

STAR-only seed-0, 320 tubes, min lambda_t 2.0:
  STAR final       step 1191, heldout PSNR 13.086738586425781
  STAR selected    step 600, heldout PSNR 12.77337646484375, render 0.11402837600053317s
  STAR heldout-best step 1191, heldout PSNR 13.086738586425781
  max tile / pair  65 / 4.578496029044701
  overflow/unstable 0 / 0.0
```

Read: reject a simple temporal precision floor for the 512px multicam branch.
It made the per-frame overfit harness better, but here it lowers the seed-0
320-tube curve while preserving clean Metal stats. The next support-policy
experiment needs a different mechanism, not just a `lambda_t` floor.

512px train-view gap-collapse selector diagnostic, 2026-05-11:

Post-hoc on the saved 320-tube checkpoint curves, threshold `0.7` looked useful:
it picked seed-0 step `400` at heldout PSNR `13.70832633972168`, seed-1 step
`400` at `13.769192695617676`, and fell back to seed-2 final step `837` at
`13.637543678283691`. It was therefore wired as
`--uvt-select-checkpoint first_train_view_gap_collapse` with CPU smoke
`multicam_heldout_compare_train_view_gap_collapse_smoke_16_2f_1s`.

The live seed-0 rerun rejects it as the current selector:

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

Read: keep gap collapse as a diagnostic, not a reporting selector. It can pick
the right checkpoint on one saved curve, but the live repeat collapses the
train-camera gap before the true heldout shoulder. This reinforces that 512px
selection needs a validation-shaped signal, not another train-only scalar
threshold.

512px train-camera temporal dev-frame selector diagnostic, 2026-05-11:

The harness now supports a lighter validation-shaped selector without removing
a whole train camera:

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --uvt-validation-frame-stride 4 \
  --uvt-validation-frame-offset 1 \
  --uvt-select-checkpoint best_train_dev_frame_psnr
```

This excludes frames `[1, 5, 9, 13]` from optimizer sampling and adds
`train_fit_frame_eval_psnr` / `train_dev_frame_eval_psnr` to checkpoint rows.
`--uvt-init-frames fit` is available when the dev frames should also be
excluded from tube initialization. The CPU smoke
`multicam_heldout_compare_train_dev_frame_selector_fitinit_smoke_16_2f_1s`
passed.

Real seed-0 512px/320-tube results:

```text
Clean fit-init temporal dev split:
  optimizer frames [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15]
  dev frames       [1, 5, 9, 13]
  selected step    1059
  selected heldout 13.461018562316895
  selected dev     16.582809448242188
  selected render  0.10677025099903403s
  heldout-best     step 300, 13.579765319824219
  max tile / pair  88 / 2.3111431115765724
  overflow/unstable 0 / 0.0

All-init control with the same optimizer/dev split:
  selected step    1125
  selected heldout 13.39246654510498
  selected dev     15.492460250854492
  selected render  0.11622983400047815s
  heldout-best     step 600, 13.449368476867676
  max tile / pair  101 / 2.853536779149101
  overflow/unstable 0 / 0.0
```

Read: reject train-camera temporal dev frames as the current 512px selector.
The clean split lowers representation quality below the earlier no-dev 320-tube
oracle peak `13.70832633972168`, and the all-init control shows same-camera
dev-frame PSNR is still monotonic enough to select final while true heldout
falls. This closes the cheap train-camera validation-subset lane; next work
should move to multiscale/crop training or a different support/window policy.

512px free init-lambda support check, 2026-05-11:

The 128px overfit harness liked narrower temporal support, while the 512px
multicam `min_lambda_t` floors were negative. This run checks the softer
version: start narrow with `--uvt-init-lambda-t 2.0`, but keep
`--uvt-min-lambda-t` at the default so optimization can relax support.

```text
STAR-only seed-0, 320 tubes, init lambda_t 2.0:
  selected step    600
  selected heldout 13.213919639587402
  selected train   14.649134159088135
  selected render  0.10834887499913748s
  final step       1209
  final heldout    13.40640640258789
  heldout-best     step 1100, 13.415067672729492
  max tile / pair  76 / 4.07200551573087
  overflow/unstable 0 / 0.0
```

Read: reject narrow temporal initialization as the 512px support fix. It
renders cleanly but is far below the no-floor 320-tube branch whose heldout-best
peak was `13.70832633972168`. The overfit temporal-support win does not
transfer to multicam as a hard support floor or as an initialization-only bias.

512px bounded sequence-consistency support check, 2026-05-11:

The hard temporal floors and softer init-lambda check both failed, so this hook
tests a different support pressure: occasional multi-frame consistency loss from
the same rendered train sequence.

The full 16-frame consistency backward is not usable on the local MPS path yet;
it failed before useful training with:

```text
RuntimeError: Invalid buffer size: 12.00 GiB
```

Bounded four-frame consistency every 20 steps:

```text
STAR-only seed-0, 320 tubes, seq consistency 4 frames every 20 steps:
  selected/final step 340
  selected heldout    13.58269214630127
  selected train      14.951775550842285
  selected render     0.11685991700323939s
  max tile / pair     103 / 3.2703078231742175
  overflow/unstable   0 / 0.000335693359375
```

Bounded four-frame consistency every 50 steps:

```text
STAR-only seed-0, 320 tubes, seq consistency 4 frames every 50 steps:
  selected step       600
  selected heldout    13.619542121887207
  selected train      16.08782720565796
  selected render     0.11323120699853462s
  final step          666
  final heldout       13.626453399658203
  heldout-best        step 666, 13.626453399658203
  max tile / pair     89 / 2.7667986055505245
  overflow/unstable   0 / 0.0
```

Read: reject bounded sequence consistency as the current 512px fix. Every 20
steps starves the optimizer. Every 50 steps keeps Metal clean and roughly ties
the V-JEPA reference only at final/heldout-best, but the non-heldout selected
checkpoint is still a hair below V-JEPA and the whole branch remains below the
no-consistency 320-tube oracle peak `13.70832633972168`. The speed evidence
still points away from a forward-raster rewrite: the selected render-only time
is `0.11323120699853462s` across the three eval sequences, and the Metal stats
show zero overflow and zero unstable tiles.

512px multiscale auxiliary loss bracket, 2026-05-11:

The multiscale hook adds a downsampled reconstruction term without adding a
second render:

```text
--uvt-multiscale-loss-weight
--uvt-multiscale-loss-factor
```

The CPU smoke `multicam_heldout_compare_multiscale_smoke_16_4f_1s` passed and
logged `multiscale_loss` / `multiscale_term`. The first full bracket used
factor `4`, weight `0.25` on the same 512px, seed-0/1, 320-tube, window-1
recipe.

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

Seed 1, factor 4, lower weight 0.05:

```text
Seed 1, factor 4, weight 0.05:
  selected step       600
  selected heldout    13.568358421325684
  selected train      15.75159215927124
  selected render     0.11502208300044003s
  final heldout       13.560102462768555
  heldout-best        step 400, 13.603869438171387
  max tile / pair     100 / 2.893910082667028
  overflow/unstable   0 / 0.0
```

Read: keep the hook, but reject simple global factor-4 multiscale loss as the
512px default. Weight `0.25` fixes seed-0 final enough to clear the V-JEPA
reference, but still leaves the gain-drop selector too early. More importantly,
it damages seed 1 badly versus the saved no-multiscale 320-tube row:
selected/final/heldout-best `13.637592315673828` / `13.682265281677246` /
`13.769192695617676`. Lowering the weight to `0.05` only partially recovers
seed 1 and still remains below V-JEPA and below no-multiscale. If this lane
continues, use a targeted crop/scale policy rather than another near-zero
global auxiliary.

512px deterministic crop auxiliary loss bracket, 2026-05-11:

The crop hook adds local full-resolution weighting without another render:

```text
--uvt-crop-loss-weight
--uvt-crop-loss-size
```

It cycles a deterministic 3x3 crop grid over the already-rendered train output.
The CPU smoke `multicam_heldout_compare_crop_smoke_16_4f_1s` passed and logged
`crop_loss` / `crop_term`. The full 512px checks targeted the failing seed-1,
320-tube, window-1 branch.

```text
Seed 1, crop size 256, weight 0.25:
  selected/final step 578
  selected heldout    13.566254615783691
  selected train      15.60933542251587
  selected render     0.17345320899767103s
  heldout-best        step 578, 13.566254615783691
  max tile / pair     104 / 3.0012101814828935
  overflow/unstable   0 / 0.0

Seed 1, crop size 128, weight 0.25:
  selected step       500
  selected heldout    13.48839282989502
  selected train      14.819037437438965
  selected render     0.16554591700150922s
  final heldout       13.565434455871582
  heldout-best        step 400, 13.591614723205566
  max tile / pair     101 / 3.2960348470935785
  overflow/unstable   0 / 0.0
```

Read: reject deterministic crop loss as the current 512px support fix. Both
crop sizes leave Metal clean but roughly halve the step count and stay below
the saved no-crop seed-1 row. This branch is not a raster capacity issue; it is
an optimization/support policy issue.

512px deterministic cycle-schedule revisit, 2026-05-11: the old 256px
20-second cycle probe was negative, but the 512px no-aux 320-tube/window-1
recipe benefits from deterministic train scheduling. This check changes only:

```text
--uvt-train-schedule cycle
```

Seed 0 result:

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

Seed 1 result:

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

Seed 2 rejects the plain cycle rule:

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
overflow/unstable     0 / 0.0 at selected; final model produced zero active tiles
```

The seed-2 LR `0.01` run becomes non-finite after step `480`: reconstruction
remains finite, but model/projection regularizers become non-finite and the
final Metal report has zero active tiles. A lower-LR stability bracket stayed
finite but underfit:

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

Read: deterministic cycle is useful evidence that training-sample order matters,
but it is not a seed-robust 512px default. It improves the non-heldout selected
checkpoint on seeds 0 and 1 while staying fast, then fails seed 2. The next
schedule branch should preserve coverage while breaking the fixed cycle phase,
for example shuffled or phase-randomized window order, and should add a
non-finite guard for honest reports. This remains an optimizer/schedule issue,
not a rasterizer rewrite.

Non-finite guard, 2026-05-11: the STAR trainer now checks scalar loss at the
existing log cadence, records `stopped_reason` / `stopped_step`, writes
non-finite scalar log fields as `null`, and restores the last checkpointed
finite state before final eval. The normal smoke passed:

```text
out dir          research_project/benchmarks/results/multicam_heldout_compare_nonfinite_guard_smoke3_16_2f_1s
stopped_reason   null
steps            2
```

The forced non-finite smoke uses `--uvt-depth-slope-reg nan` only to exercise
the guard path:

```text
out dir          research_project/benchmarks/results/multicam_heldout_compare_nonfinite_guard_forced_smoke_16_2f_1s
stopped_reason   nonfinite_loss
stopped_step     1
steps            0
log loss         null
log projected    null
```

Read: future unstable schedule or support branches should fail as explicit
non-finite training stops instead of silently evaluating a corrupted zero-tile
final model.

512px shuffled-cycle follow-up, 2026-05-11: the new schedule keeps the same
no-aux 320-tube/window-1 recipe and changes only:

```text
--uvt-train-schedule shuffled_cycle
```

The CPU smoke `multicam_heldout_compare_shuffled_cycle_smoke_16_2f_1s` passed
and reported `train_schedule: shuffled_cycle`. The seed-2 full run rescued the
plain-cycle non-finite failure but did not make the current non-heldout selector
strong enough:

```text
out dir              research_project/benchmarks/results/mcam512_s2_t320_shuffled_cycle
steps                920
stopped_reason       null
selected step         600
selected train        16.362467765808105
selected heldout      13.574305534362793
selected render       0.13692954100042698s
final heldout         13.566455841064453
heldout-best          step 300, 13.6640625
max tile / pair       93 / 2.8858366120208703
overflow/unstable     0 / 0.0
```

An earlier LR drop at step `300` is rejected:

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

The saved rows suggested a possible non-heldout selector:
`first_balanced_train_psnr_plateau` with plateau delta `0.6` and max train-view
gap `1.7` would have selected strong earlier shoulders on seed 0 cycle, seed 1
cycle, and the original seed-2 shuffled row. The live seed-2 rerun did not
reproduce that shoulder:

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

The exact-step check also failed to reproduce the original step-300 shoulder:

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

Read: shuffled coverage is the right kind of next lever because it fixes the
seed-2 stability failure without touching the renderer. It is not yet a default:
selected quality remains below the V-JEPA reference and below the saved random
seed-2 branch, while the original heldout-best shoulder is not reproducible in
the exact-step rerun. Work next on schedule determinism, runtime variance, and
selector stability; do not start a rasterizer rewrite from this evidence.

512px reshuffled-cycle follow-up, 2026-05-11: `reshuffled_cycle` keeps the same
full-coverage contract but generates a fresh deterministic shuffle each coverage
epoch instead of repeating one shuffled order. The new option is:

```text
--uvt-train-schedule reshuffled_cycle
```

The CPU smoke `multicam_heldout_compare_reshuffled_cycle_smoke_16_2f_1s` passed
with `train_schedule: reshuffled_cycle`, `steps: 2`, and
`stopped_reason: null`.

The first 512px check compared directly against the failed shuffled fixed-300
target:

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

The fixed-600 three-seed check uses the same no-aux 320-tube/window-1 recipe,
`--max-steps 600`, and the non-heldout `first_train_psnr_gain_drop` selector:

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

Read: this is the first 512px STAR-UVT schedule variant whose non-heldout
selected checkpoint clears the V-JEPA F32 reference `13.6248` on seeds 0, 1,
and 2. It is a robust-floor candidate rather than a pure quality win: seed 2 is
better than random and fixed shuffled, seed 1 stays strong, but seed 0 gives up
the much higher plain-cycle result (`13.798948287963867` selected,
`13.841632843017578` heldout-best). Keep iterating schedule/selector policy; do
not move to a rasterizer rewrite from this result.

512px phase-rotated-cycle follow-up, 2026-05-11: `phase_rotated_cycle` keeps the
ordered cycle pairs but rotates the start point each coverage epoch. The new
option is:

```text
--uvt-train-schedule phase_rotated_cycle
```

The CPU smoke `multicam_heldout_compare_phase_rotated_cycle_smoke_16_2f_1s`
passed with `train_schedule: phase_rotated_cycle`, `steps: 2`, and
`stopped_reason: null`.

Seed 2 looked promising:

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

Seed 0 rejects it as a robust-floor replacement:

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

Read: reject this branch for now. It recovers the fast render timing and seed-2
stability, but it falls below V-JEPA on seed 0 and below the reshuffled robust
floor. Do not run seed 1 unless a later schedule policy explains the seed-0
drop.

512px view-shuffled-cycle follow-up, 2026-05-11: `view_shuffled_cycle` keeps
frames/windows in temporal cycle order, but shuffles the train-camera order
inside each frame/window slot. The new option is:

```text
--uvt-train-schedule view_shuffled_cycle
```

The CPU smoke `multicam_heldout_compare_view_shuffled_cycle_smoke_16_2f_1s`
passed with `train_schedule: view_shuffled_cycle`, `steps: 2`, and
`stopped_reason: null`.

The fixed-600 three-seed check uses the same no-aux 320-tube/window-1 recipe and
the non-heldout `first_train_psnr_gain_drop` selector:

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

Read: this is the best current robust-floor schedule. It keeps seed 0 at the
same floor as reshuffled, improves seed 2 materially, and stays above the V-JEPA
F32 reference on all three non-heldout selected checkpoints. It still does not
recover the plain-cycle seed-0 peak, and the gain-drop selector fires early on
seeds 1 and 2 even though fixed step `600` is better. Next work should tune the
selector or treat fixed-600 as the reporting checkpoint for this schedule.

The existing non-heldout `best_train_psnr` selector gives that fixed-600 report
without adding another selector. The CPU smoke
`multicam_heldout_compare_view_shuffled_besttrain_selector_smoke_16_2f_1s`
passed:

```text
train_schedule       view_shuffled_cycle
selector             best_train_psnr
selected step        2
uses heldout         false
```

Applied post-hoc to the saved 512px view-shuffled checkpoint curves, it selects
step `600` on all three seeds:

```text
Seed 0 best_train    step 600, heldout 13.639522552490234, render 0.1782556249963818s
Seed 1 best_train    step 600, heldout 13.812097549438477, render 0.1593077910001739s
Seed 2 best_train    step 600, heldout 13.793721199035645, render 0.1350984589989821s
```

Read: for `view_shuffled_cycle`, report `--uvt-select-checkpoint
best_train_psnr` until a better selector is proven. The rule is still
non-heldout, and it avoids the early gain-drop selection on seeds 1 and 2.

Train-camera balance selector diagnostic, 2026-05-11: checkpoint rows now
include `train_view_eval_psnr`, `train_min_view_eval_psnr`, and
`train_view_eval_psnr_gap`. `--uvt-select-checkpoint best_min_train_view_psnr`
selects the checkpoint with the best worst-case train-camera PSNR and labels
the selected report `uses_heldout_for_selection: false`. The CPU smoke
`multicam_heldout_compare_train_view_selector_smoke_16_2f_1s` passed.

The seed-2 MPS diagnostic used the same tuned STAR-only recipe with
`--skip-splats`:

```text
STAR final / selected by min train-view PSNR:
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

Read: this selector is rejected as a schedule rule. The worst-case train-camera
PSNR was still monotonic enough to pick the final checkpoint, so it missed the
heldout peak by about `0.069` dB. Keep the per-train-camera fields for
analysis, but do not spend more runs on plain `best_min_train_view_psnr`.

True train-camera dev split diagnostic, 2026-05-11:
`--uvt-optimizer-train-views first_only` restricts STAR optimizer samples to
train camera index `0`, while `--uvt-select-checkpoint best_train_view_psnr
--uvt-select-train-view-index 1` selects by the other train camera. The CPU
smoke `multicam_heldout_compare_train_view_dev_selector_smoke_16_2f_1s` passed
and wrote `optimizer_train_view_indices: [0]` plus selected media.

The seed-2 MPS STAR-only diagnostic used first-view initialization to avoid
initializing from the dev camera:

```text
STAR final / selected by dev train view:
  step             972
  optimizer views  [0]
  train PSNR       15.337390899658203
  dev-view PSNR    10.769975662231445
  heldout PSNR     12.647516250610352
  render-only      0.03078908397583291s

Heldout-best checkpoint in same curve:
  step             800
  heldout PSNR     12.661633491516113
```

Read: rejected as a quality path. This is the cleanest non-heldout selector
contract so far, but removing one of two train cameras is too damaging: true
heldout falls well below V-JEPA and far below the all-train recipe. Future
selector work should not throw away a whole train camera.

Balanced train-plateau selector diagnostic, 2026-05-11:
`--uvt-select-checkpoint first_balanced_train_psnr_plateau` combines the train
PSNR low-gain shoulder with a maximum train-camera PSNR gap from the previous
checkpoint. The explicit-gap CPU smoke
`multicam_heldout_compare_balanced_train_plateau_smoke_16_2f_1s` passed with
`--uvt-select-train-view-gap-max 1.0`, and the default-gap CPU smoke
`multicam_heldout_compare_balanced_train_plateau_default_smoke_16_2f_1s`
passed with the default gap max `1.2`.

```text
STAR-only seed-0, 320 tubes, gap max 1.0:
  selected step    400
  selected train   15.122135639190674
  selected heldout 13.62360954284668
  selected render  0.12536841700148216s
  final heldout    13.594148635864258
  heldout-best     step 900, 13.625420570373535

STAR-only seed-1, 320 tubes, gap max 1.0:
  selected step    900
  selected heldout 13.600866317749023
  final heldout    13.597005844116211
  heldout-best     step 400, 13.698472023010254

STAR-only seed-1, 320 tubes, gap max 1.2:
  selected step    500
  selected train   15.45426082611084
  selected heldout 13.442426681518555
  selected render  0.12534225000126753s
  final heldout    13.662494659423828
  heldout-best     step 800, 13.675647735595703
```

Read: reject this as the current 512px selector. Gap max `1.0` picked the
intended seed-0 shoulder and landed just under the V-JEPA reference, but it
missed the seed-1 oracle peak because the step-400 train-camera gap was
`1.1557254791259766`. Relaxing to gap max `1.2` did not fix that seed; it
selected a bad step-500 shoulder while final and heldout-best were much better.
Keep the selector as a diagnostic, but do not keep tuning this threshold path
as the next scale rule.

Latest 128px single-video transfer, 2026-05-11: the 64px winner transfers in
speed but needs resolution-aware capacity/support. At 128x128, 16 frames, 200
Metal steps, 448 tubes at LR `0.24` reached PSNR `20.90794801712036`; 896
tubes at LR `0.16` reached `21.883294582366943`; 1792 tubes at LR `0.12`,
spatial precision `0.125`, reached PSNR `22.2884202003479` in
`51935.27437499142 ms`, render `1.6798329888843 ms`. The same setting at 400
constant-LR steps did not improve eval PSNR (`22.23587989807129`), and a
200-step plus 200-step appearance-only tail reached only PSNR
`22.21776008605957`. Whole-model LR decay worked: LR `0.12 -> 0.02` at step
200 reached PSNR `22.72578239440918`; LR `0.12 -> 0.04` reached the current
baseline PSNR `22.809326648712158` in `134123.30012500752 ms`, render
`2.7211669948883355 ms`; LR `0.12 -> 0.06` fell to PSNR
`21.763882637023926`. Naive block-match velocity init was harmful at PSNR
`21.428205966949463`. Narrower temporal support helped: temporal precision
`1.0` reached the current 128px best PSNR `23.209903240203857`, and temporal
precision `2.0` was essentially tied but slightly worse at `23.207027912139893`.
A bounded 50-step 128px paired run found 1792 UVT at PSNR
`20.928823947906494` in `13987.883624999085 ms`, render `2.2979159839451313 ms`;
64 splats/frame reached PSNR `19.460207223892212` in `175350.1634580025 ms`,
render `161.16950000287034 ms`.
A full same-step 200-step paired run now closes that baseline gap: 1792 UVT at
LR `0.12`, spatial precision `0.125`, temporal precision `0.5`, and Metal tile
backward reached PSNR `22.31398344039917` in `20749.791541000377 ms`, render
`3.108167000391404 ms`; 64 splats/frame at LR `0.32` reached PSNR
`20.627903938293457` in `687844.0475830003 ms`, render
`143.26100000016595 ms`. That is a `1.686079502105713` dB UVT win at the same
200 steps, about `33.15x` faster training, and about `46.09x` faster rendering.
The same-step temporal-support bracket improves the UVT row without rerunning
the slow per-frame baseline: temporal precision `1.0` reached PSNR
`22.817583084106445` in `22077.392624999447 ms`, render
`4.774042000462941 ms`; temporal precision `2.0` reached the best equal-step
PSNR `23.130309581756592` in `17787.110125000254 ms`, render
`1.5611250000802102 ms`; temporal precision `4.0` regressed to
`22.765743732452393`. Relative to the saved per-frame 200-step row, the
`t=2.0` UVT row is `+2.5024056434631348` dB, about `38.67x` faster to train,
and about `91.77x` faster to render. A media rerun of the same recipe wrote
`research_project/benchmarks/results/video_fit_single_overfit_128_16f_200steps_1792uvt_lr012_s0125_t20_uvtonly_sheet_metal_tile.png`
and reached PSNR `23.138446807861328` in `17548.05325000052 ms`, render
`1.0632920020725578 ms`.
A same-step LR bracket around the `t=2.0` setting rejects nearby LR changes:
LR `0.08` reached PSNR `22.904906272888184`, and LR `0.16` reached
`22.98252582550049`. Keep LR `0.12` for the current 128px equal-step overfit
recipe.

128px 400-step headroom check, 2026-05-11: keeping the same 1792-tube, LR
`0.12`, spatial precision `0.125`, temporal precision `2.0`, opacity `0.7`,
Metal tile-backward recipe but training STAR for 400 steps reached PSNR
`23.569955825805664` in `31383.009374996618 ms`, render
`1.2202500001876615 ms`. It emits
`research_project/benchmarks/results/video_fit_single_overfit_128_16f_400steps_1792uvt_lr012_s0125_t20_uvtonly_sheet_metal_tile.png`.
Against the saved 64-splats/frame 200-step baseline, this is
`+2.942051887512207` dB, about `21.92x` faster to train, and about
`117.40x` faster to render. Use the 200-step row for equal-step claims and the
400-step row as the cap-128/tile-t-2 speed reference.

128px tile-shape and capacity check, 2026-05-11: the local overfit scripts now
accept `--uvt-tile-t` and `--uvt-tile-capacity`, and record them in JSON. Keeping
the 1792-tube, LR `0.12`, spatial precision `0.125`, temporal precision `2.0`,
opacity `0.7` recipe but setting `--uvt-tile-t 1 --uvt-tile-capacity 128`
improves the cap-128 local quality recipe. The later synchronized repeat-timing
rows below supersede the first one-shot render timings for speed claims.
Capacity `256` is a quality-mode
tradeoff. With `tile_t=2` it reached PSNR `24.08522367477417` but rendered in
`62.1839169980376 ms`; with `tile_t=1` it keeps the quality at PSNR
`24.083971977233887` in `77384.35312500224 ms`, render
`5.169167001440655 ms`. The `tile_t=1`, cap-256 row is still about `8.89x`
faster to train and `27.71x` faster to render than the direct splat baseline,
but the cap-128 row remains the default speed recipe. At equal 200 steps,
`tile_t=1`, cap-256 reaches PSNR `23.22518825531006` in
`39326.092333001725 ms`, render `11.385666999558453 ms`: only
`0.04694938659667969` dB above cap-128, while much slower. Against the saved
per-frame splat baseline it is still `+2.5972843170166016` dB, `17.49x` faster
to train, and `12.58x` faster to render, so use it only as a same-step quality
mode.

`tile_t=1` cap-128 LR bracket, 2026-05-11: rerunning the 1792-tube 400-step
recipe at nearby learning rates gives a tiny quality gain but weakens the speed
story. LR `0.10` reaches PSNR `23.786139488220215` in `42992.40170899793 ms`,
render `2.3423329985234886 ms`; LR `0.11` reaches PSNR `23.796110153198242` in
`62273.226834000525 ms`, render `5.030708998674527 ms`; LR `0.14` regresses to
PSNR `23.553497791290283` in `52348.505624999234 ms`, render
`4.206790999887744 ms`. Keep LR `0.12` as the cap-128 speed recipe; if the
budget allows roughly `5 ms` render, the `tile_t=1`, cap-256 quality row gives
far more PSNR. The equal-step row also stays LR `0.12`: a 200-step LR `0.11`
check reaches only PSNR `23.128459453582764` in `53446.34366600076 ms`, render
`19.63295899986406 ms`, worse and much slower than the 200-step LR `0.12` row.

Synchronized repeat render timing, 2026-05-11: the benchmark now accepts
`--render-benchmark-repeats`, synchronizes MPS around each final render, and
stores min/median/max plus the sample list under `uvt.render_benchmark_ms`.
It also records the same structure for `per_frame.render_benchmark_ms` when the
direct baseline is included. The current `tile_t=1`, cap-128 recipe at equal
200 steps has a paired 20-repeat render benchmark against the 64-splats/frame
baseline:

- STAR: PSNR `23.189358711242676`, train `32827.123874998506 ms`, render median
  `7.475833499483997 ms`, min `5.896291000681231 ms`, max
  `23.557167001854395 ms`.
- Direct splats: PSNR `20.627903938293457`, train `1177423.645084 ms`, render
  median `203.9981664984225 ms`, min `172.3443330010923 ms`, max
  `398.43345799818053 ms`.

The paired same-step read is `+2.5614547729492188` dB, `35.86740189507551x`
faster training, and `27.287681903630283x` faster median render for STAR. The
400-step STAR-only repeat-timed row reaches PSNR `23.745369911193848` in
`81651.38491700054 ms`, render median `10.333166999771493 ms`, min
`5.961792001471622 ms`, max `15.938207998260623 ms`, over 20 repeats. Use these
repeat-timed rows for conservative speed claims; older render timings in this
section are one-shot historical samples.

Cap-256 quality-mode repeat timing, 2026-05-11: the 1792-tube, LR `0.12`,
spatial precision `0.125`, temporal precision `2.0`, `tile_t=1`, cap-256
400-step row reaches PSNR `24.085018634796143` in `38323.750625000685 ms`, with
render median `4.634708000594401 ms`, min `4.513209001743235 ms`, and max
`5.108583001856459 ms` over 20 repeats. Against the paired 64-splats/frame
baseline above, that is `+3.4571146965026855` dB, `30.723079705980084x` faster
to train, and `44.01532231852789x` faster to render by median. This is the
current 128px single-video quality mode; keep cap-128 for the equal-step speed
claim because the 200-step cap-256 row only adds `0.04694938659667969` dB while
taking longer.

Cap-256 quality-mode seed robustness, 2026-05-11: the same recipe reaches PSNR
`24.0570068359375` at seed `0`, `24.085018634796143` at seed `5`, and
`24.107441902160645` at seed `13`. The three-seed mean is
`24.083155790964764`, span `0.05043506622314453`, and population stdev
`0.020632120857024965`. Mean train time is `51.418234347000784 s`, and mean
median render is `5.657340167090297 ms`; versus the paired direct baseline, that
is `+3.4552518526713065` dB, `22.89895131633743x` faster train, and
`36.059024289385015x` faster median render. Artifact suffixes:
`...cap256_renderbench20_metal_tile`, `...cap256_seed0_renderbench20_metal_tile`,
and `...cap256_seed13_renderbench20_metal_tile`.

Native 256px single-video scale gate, 2026-05-11: this uses
`test_video_small.mp4` at 256x256 rather than upsampling the 128px fixture. A
scaled 7168-tube, cap-256 200-step attempt was interrupted after about 14 minutes
without writing an artifact, so the first bounded gate used 50 steps. Results:

- 7168 tubes, cap `256`, 50 steps: PSNR `22.230050563812256`, train
  `146.55860279099943 s`, median render `12.478374999773223 ms`.
- 3584 tubes, cap `256`, 50 steps: PSNR `21.195032596588135`, train
  `161.72111600000062 s`, median render `12.958499999513151 ms`.
- 7168 tubes, cap `128`, 50 steps: PSNR `22.362003326416016`, train
  `22.07102679200034 s`, median render `11.440250000305241 ms`.
- 7168 tubes, cap `64`, 50 steps: PSNR `16.391620635986328`, train
  `38.039112208000006 s`, median render `8.02179199672537 ms`.

Read: cap `128` is the useful 256px boundary. It is faster and slightly better
than cap `256`; cap `64` is too lossy, and lower tube count does not buy speed.
The 7168-tube cap-128 recipe then reaches PSNR `24.46974277496338` in
`70.66461358399829 s` at 200 steps, with median render
`18.966646002809284 ms`, and PSNR `25.1381516456604` in
`168.99901500000124 s` at 400 steps, with median render
`26.702728999225656 ms`. This is the current native-256px single-video recipe.

Native 256px single-video seed robustness, 2026-05-11: the 400-step cap-128
recipe reaches PSNR `25.044105052947998` at seed `0`, `25.1381516456604` at
seed `5`, and `25.099973678588867` at seed `13`. The three-seed mean is
`25.09407679239909`, span `0.09404659271240234`, and population stdev
`0.038620118679728956`. MPS timing varies more than quality: train times are
`168.99901500000124 s`, `267.3640800409994 s`, and `236.97501191700212 s`;
render medians are `26.702728999225656 ms`, `19.233874998462852 ms`, and
`32.02449999844248 ms`. Treat this as quality-robustness evidence, not a precise
speed aggregate.

Per-frame video-init baseline feasibility, 2026-05-11: the benchmark now accepts
`--per-frame-init-mode video_samples`, `--per-frame-spatial-precision`,
`--per-frame-opacity`, and `--per-frame-sample-mode`. The smoke
`research_project/benchmarks/results/video_fit_per_frame_video_init_smoke_16_2f_1step.json`
passes and writes the new init fields. On native 256px, the 5-step paired probe
with 64 video-initialized splats/frame reaches direct PSNR
`6.486777663230896` in `76.71433137500208 s`, median render
`779.2045410024002 ms`; STAR reaches PSNR `13.252005577087402` in
`1.7836201249992882 s`, median render `6.664707998425001 ms`. This is
`+6.765227913856506` dB for STAR, `43.010465232910896x` faster train, and
`116.91503081403435x` faster median render at the feasibility point. A random
per-frame init row was even weaker at PSNR `5.105733871459961`, so video init is
the fairer direct baseline, but the Python per-frame renderer is too slow for a
full 200-step 256px same-step baseline.

Equal-step tube-count retune, 2026-05-11: the 1792-tube row is not the fastest
equal-step speed/quality point. With synchronized 20-repeat render timing, 896
tubes at LR `0.16` reaches PSNR `22.211849689483643` in
`30988.440125001944 ms`, render median `7.579749999422347 ms`; 1344 tubes at LR
`0.14` reaches PSNR `22.728750705718994` in `17953.880875000323 ms`, render
median `4.324979001467 ms`; 1600 tubes at LR `0.13` reaches PSNR
`23.01387310028076` in `18926.709583000047 ms`, render median
`4.298770498280646 ms`; 1728 tubes at LR `0.125` reaches PSNR
`23.199284076690674` in `19664.952208000614 ms`, render median
`4.426229001182946 ms`. Against the paired direct baseline, the 1728-tube row is
`+2.571380138397217` dB, `59.87421848932689x` faster to train, and
`46.08847993267007x` faster to render by median, so use it for the equal-step
speed recipe. The 1728-tube 400-step row reaches PSNR `23.601515293121338` in
`33423.33199999848 ms`, render median `4.425312499733991 ms`, which is
`0.14385461807250977` dB below the 1792-tube 400-step repeat-timed row; keep
1792 tubes for the cap-128 400-step quality recipe.

1728-tube equal-step seed robustness, 2026-05-11: the equal-step speed recipe is
not a seed-5 fluke. Seed `0` reaches PSNR `23.242146968841553` in
`32942.09425000008 ms`, render median `8.675395998579916 ms`; seed `5` reaches
PSNR `23.199284076690674` in `19664.952208000614 ms`, render median
`4.426229001182946 ms`; seed `13` reaches PSNR `23.195884227752686` in
`36460.29487499982 ms`, render median `4.701228997873841 ms`. The three-seed
PSNR mean is `23.212438424428303`, span `0.04626274108886719`, population stdev
`0.021052916687334368`. Timing is noisier than quality, but the three-seed mean
is train `29.689113777666837 s`, median render `5.934284665878901 ms`; versus
the paired direct baseline, that is `+2.584534486134846` dB,
`39.65843015394074x` faster train, and `34.37620167960197x` faster median
render.

`tile_t=1` cap-128 seed robustness, 2026-05-11: the 400-step default is stable
enough to treat as a recipe, not a seed-5 fluke. Seed `0` reaches PSNR
`23.78368377685547` in `71685.35045899989 ms`, render `10.91291599732358 ms`;
seed `5` reaches `23.715169429779053` in `33695.67583299795 ms`, render
`2.245167001092341 ms`; seed `13` reaches `23.625149726867676` in
`65515.39149999735 ms`, render `9.041584002261516 ms`. The PSNR mean is
`23.708000977834065`, span `0.15853404998779297`, population stdev
`0.06491944382000768`. The one-shot render timing samples are noisy across
seeds, so quote seed `5` for matched speed comparisons and the three-seed set
for quality robustness.

Temporal-quarter init rejection, 2026-05-11: `--uvt-sample-mode
temporal_quarters` reuses spatial sites across four temporal pieces at the same
1792-tube count. The CPU smoke
`research_project/benchmarks/results/video_fit_temporal_quarters_smoke_32_4f_2steps.json`
passed with decreasing loss, but the 128px/16f/400-step run reached only PSNR
`23.275623321533203` in `32351.469666999037 ms`, render
`1.253209000424249 ms`. That is `-0.29433250427246094` dB versus the current
random-sampled 400-step recipe, so reject this init mode as the next overfit
quality mechanism. Artifact:
`research_project/benchmarks/results/video_fit_single_overfit_128_16f_400steps_1792uvt_lr012_s0125_t20_temporalquarters_metal_tile.png`.

Temporal split/refine rejection, 2026-05-11: the benchmark now accepts
`--uvt-temporal-split-step`, `--uvt-temporal-split-offset`,
`--uvt-temporal-split-precision-scale`, and
`--uvt-temporal-split-opacity-scale`. The CPU split smokes
`research_project/benchmarks/results/video_fit_temporal_split_smoke_32_4f_4steps.json`
and
`research_project/benchmarks/results/video_fit_temporal_split_opacity_smoke_32_4f_4steps.json`
passed, but the 128px/16f/400-step split/refine checks reject the mechanism:
offset `0.5`, precision scale `2.0`, opacity scale `1.0` reached PSNR
`19.583353996276855`; offset `0.25`, scale `1.0`, opacity scale `1.0` reached
`18.187309503555298`; offset `0.25`, scale `1.0`, opacity scale `2.0` improved
to `21.271286010742188` but still lost `2.2986698150634766` dB to the unsplit
400-step recipe and rendered slower (`7.774166999297449 ms`). The split handoff
is damaging: the post-split initial loss jumps to `0.0137-0.0201` after
pre-split losses around `0.0049`.

Render-preserving duplicate split rejection, 2026-05-11: allowing
`--uvt-temporal-split-offset 0.0` and exposing
`--uvt-temporal-split-depth-offset` tested whether the split failure was from
temporal/depth displacement. It was not. Offset `0.0`, precision scale `1.0`,
opacity scale `1.0`, old depth offset `1e-4` reached PSNR
`19.849724769592285`, and the true zero-depth case reached only
`19.785715341567993`. The zero-depth row took `182225.27079200154 ms`, rendered
in `26.041582998004742 ms`, and still jumped from pre-split loss
`0.004865488037467003` to post-split initial loss `0.02108645997941494`. This
rejects the current duplicate split as a render-preserving split boundary in
the 128px Metal path.

Duplicate-split tile-fallback diagnosis, 2026-05-11: the boundary probe
`research_project/benchmarks/results/video_fit_split_boundary_probe_128_16f_1792_split200_preserve_depth0.json`
confirms why the duplicate split is slow and lossy. At step 200 before the
split, the current 1792-tube recipe renders at PSNR `23.139398097991943`, loss
`0.004853557329624891`, forward time `9.447625001484994 ms`, stable tile
fraction `1.0`, max tile count `207`, and overflow tile count `598`. After a
zero-offset, zero-depth duplicate split to 3584 tubes, the same render drops to
PSNR `17.03884720802307`, loss `0.019774947315454483`, forward time
`110.73370899975998 ms`, stable tile fraction `0.0`, unstable tile fraction
`1.0`, max tile count `398`, and overflow tile count `1898`. The normal
STAR-UVT 1792-tube recipe remains fast; the rejected split/refine path is
creating a Metal tile-capacity/fallback problem.

`tile_t=1` split follow-up, 2026-05-11: changing the temporal tile span improves
the base render but does not save duplicate split/refine. The boundary probe
`research_project/benchmarks/results/video_fit_split_boundary_probe_128_16f_1792_split200_preserve_depth0_tilet1_cap128.json`
has pre-split PSNR `23.180255889892578`, forward time `4.090958998858696 ms`,
stable tile fraction `1.0`, max tile count `184`, and overflow tile count `764`.
After zero-offset duplicate split to 3584 tubes it drops to PSNR
`18.50017786026001`, forward time `85.22179200008395 ms`, unstable tile fraction
`1.0`, max tile count `354`, and overflow tile count `3613`. A controlled
896-to-1792 scheduled split under `tile_t=1`, offset `0.25`, precision scale
`1.0`, opacity scale `2.0`, reached only PSNR `22.63478994369507` in
`41211.88500000062 ms`, render `1.6186249995371327 ms`; the same scheduled split
under `tile_t=2` reached PSNR `22.610313892364502`. Keep split/refine rejected
and use unsplit `tile_t=1`, cap-128 for the current local recipe.

Block-match velocity-init rejection, 2026-05-11: raw block-match velocity init
was retested under the current 128px/16f/400-step recipe and reached PSNR
`22.520790100097656`, `1.0491657257080078` dB below zero velocity. The harness
now also accepts `--uvt-velocity-init block_match_gated` with
`--uvt-velocity-min-improvement-ratio`; the CPU smoke
`research_project/benchmarks/results/video_fit_block_match_gated_smoke_32_4f_2steps.json`
passed, but the 128px runs still lose. Ratio `0.9` reached PSNR
`23.01954984664917`; stricter ratio `0.5` reached `23.172695636749268`, still
`0.3972601890563965` dB below zero velocity and slower to train. Keep
`--uvt-velocity-init zero` for the current overfit recipe.

800-step headroom rejection, 2026-05-11: the current 128px recipe does not
benefit from simply training longer. Constant LR `0.12` for 800 steps reached
only PSNR `22.233996391296387` in `133191.6974579981 ms`, render
`11.50029200289282 ms`. A staged tail, LR `0.12 -> 0.04` at step 400, recovered
to PSNR `23.460845947265625` in `109592.67349999936 ms`, render
`3.71179099965957 ms`, but still missed the 400-step recipe by
`0.10910987854003906` dB. The staged 800-step row still beats the saved
64-splats/frame 200-step baseline by `2.832942008972168` dB and is about
`6.28x` faster to train, but it does not beat the older `tile_t=2` 400-step row
and is superseded by the `tile_t=1`, cap-128 400-step recipe above.
