# STAR-UVT Research Project

This folder is the clean work area for the STAR-GS UVT rasterizer lane. It is
kept inside `variants/star_uvt_v0/` so the work stays opt-in and does not touch
the stable fast-mac variants.

## Layout

```text
research_project/
├── PROGRESS.md
├── attempts/
├── benchmarks/
├── learnings/
├── phases/
└── trainer_harness/
```

- `phases/` records the staged plan and gate criteria.
- `trainer_harness/` is the first training scaffold for projected
  `ScreenTimeTube` fitting.
- `benchmarks/` contains side-by-side renderer reports.
- `attempts/` records what was tried, including failed or limited attempts.
- `learnings/` stores durable takeaways for this specific UVT lane.

## Runnable Checks

From `variants/star_uvt_v0/`:

```bash
python3 tests/gate0_check.py --cpu-only
python3 tests/gate0_check.py
python3 research_project/trainer_harness/smoke_train.py
python3 research_project/trainer_harness/train_synthetic.py --scene moving_diagonal --steps 25 --metal-check
python3 research_project/trainer_harness/world_projection_smoke.py
python3 research_project/trainer_harness/pinhole_projection_smoke.py
python3 research_project/trainer_harness/camera_spec_projection_smoke.py
python3 research_project/trainer_harness/gradient_probe.py
python3 research_project/trainer_harness/metal_autograd_smoke.py
python3 research_project/trainer_harness/simple_metal_backward_smoke.py
python3 research_project/trainer_harness/stable_metal_backward_smoke.py
python3 research_project/trainer_harness/unstable_metal_backward_smoke.py
python3 research_project/trainer_harness/tile_metal_autograd_smoke.py
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
  --uvt-select-checkpoint first_train_psnr_plateau \
  --uvt-select-train-psnr-plateau-patience 2 --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_plateau_patience2_smoke_16_2f_1s
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 16 --max-frames 2 --train-seconds 1 --max-steps 2 \
  --device cpu --uvt-checkpoint-every-steps 1 \
  --uvt-select-checkpoint first_train_psnr_gain_drop --skip-splats \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_train_gain_drop_smoke_16_2f_1s
python3 research_project/benchmarks/camera_projection_parity_audit.py
```

The trainer harness remains a research surface. It now includes orthographic,
pinhole, and Dynaworld `CameraSpec` projection smokes plus small Metal backward
probes plus a local multicam heldout comparison harness. The current multicam
STAR-UVT path has heldout wins over direct splats in bounded 128px and 256px
temporal-window pilots. The current stable 256px `tile_t=1`
tile-load-regularized run avoids Metal unstable tiles and, after compact
backward, closed-form pinhole projection, bundled compact reduction, and LR
retuning, reaches heldout PSNR `13.20147705078125` versus direct splats
`10.722965240478516`. STAR render-only eval is `0.046138084086123854s` versus
direct splats `0.29644233302678913s`. The opt-in
`--uvt-camera-projection dataset_lens` diagnostic uses the DeepView
`opencv_fisheye` model for STAR and lifts the same 60-second recipe to heldout
PSNR `13.496740341186523`, with STAR render-only eval
`0.0413991259993054s` versus direct splats `0.4126907510217279s`. The
follow-up `--splat-camera-projection dataset_lens` row gives direct splats the
same fisheye camera contract and reaches STAR heldout PSNR
`13.632997512817383` versus lens-aware direct splats `11.188531875610352` and
the V-JEPA F32 reference `13.6248`. STAR render-only eval is
`0.04134708392666653s` versus direct splats `0.4395427079871297s`. This is the
first local V-JEPA heldout crossing, but by only about `0.0082` dB, so it still
does not claim HexGaussian projection, production integration, or full-scale
promotion. A seed-1 repeat under the same lens-aware contract fell to heldout
PSNR `12.9697904586792`; deterministic grid init improved seed 1 to
`13.179410934448242`, and all-train grid init improved seed 1 to
`13.52819538116455`, but still missed the V-JEPA reference. A deterministic
cycle training schedule was negative at the 20-second gate.
`--uvt-init-frames all` is the strongest follow-up: it initializes tubes from
all train frames and sets their `t0` to the centered source-frame time instead
of putting every initialized tube at sequence-center time. On seed 1, this
all-train grid all-frames recipe reached heldout PSNR `13.768306732177734` at
20 seconds and `13.726262092590332` at 30 seconds, clearing the V-JEPA F32
reference `13.6248`; the 60-second seed-1 row fell to `13.564573287963867`.
A 20-second seed-0 repeat also cleared V-JEPA at heldout PSNR
`13.769630432128906`, while the 30-second seed-0 repeat fell to
`13.600011825561523`. A 20-second seed-2 repeat also cleared V-JEPA at
`13.764396667480469`. Repeatability is now specifically an early-stop/schedule
question: the 20-second all-frames recipe crossed V-JEPA on all three tested
seeds. A seed-0 shorter-budget bracket reached `12.681236267089844` at
10 seconds and `13.669918060302734` at 15 seconds, so the useful window is
roughly 15-20 seconds before longer budgets can degrade heldout PSNR. The
current train-step timing probe shows the remaining 256px local train-time
speed blocker is STAR backward, not projection or forward raster alone, and the
latest all-frames rows keep render-only STAR faster than direct splats.
`--uvt-checkpoint-every-steps` is available as an opt-in diagnostic for longer
STAR runs; it reports `checkpoint_curve.best_by_heldout_psnr` without changing
the default final-checkpoint comparison. `--skip-splats` is available for
STAR-only checkpoint diagnostics once the paired direct-splat row is already
known. In the seed-0 30-second diagnostic, the best checkpoint was step `300`
at elapsed `18.632994499988854s` with heldout PSNR `13.730653762817383`, while
the final checkpoint fell to `13.518027305603027`. In the STAR-only seed-1
60-second diagnostic, the best checkpoint was also step `300`, elapsed
`17.160626166965812s`, with heldout PSNR `13.75400447845459`; the final step
`1019` fell to `13.354101181030273`. The STAR-only seed-2 40-second diagnostic
peaked later, at step `500`, elapsed `32.050383166992106s`, with heldout PSNR
`13.988276481628418`; the final step `629` fell to `13.631431579589844`. The
current blocker is therefore schedule/early-stop behavior, not forward render
speed, and the stop rule should be validation-shaped rather than a fixed
step-300 cutoff. The first opt-in schedule hook is
`--uvt-lr-decay-step/--uvt-lr-decay-factor`. On seed 1, decaying LR after step
`300` improved the 60-second final checkpoint from the no-decay heldout PSNR
`13.354101181030273` to `13.643348693847656` with factor `0.2`, and
`13.692363739013672` with factor `0.05`. The factor-`0.05` row kept render-only
eval at `0.04503662494244054s`, so schedule work is the right next lane, but
this is still below the best heldout-selected checkpoint. On seed 2, decaying
after the later step-500 peak with factor `0.05` finished at heldout PSNR
`13.81359577178955`, with best checkpoint `13.909360885620117`. Schedule decay
is useful, but a validation-shaped selector is still the cleaner rule. The
paired seed-2 schedule comparison reached STAR final heldout PSNR
`13.84060287475586` versus direct splats `11.156550407409668`, with STAR
render-only eval `0.043013749993406236s` versus direct splats
`0.3524511669529602s`. `--uvt-select-checkpoint best_heldout` now writes a
separate `star_uvt_selected` diagnostic section and selected-checkpoint media;
it is explicitly labeled as heldout-selected, not unbiased. A paired seed-1
selected-checkpoint MPS artifact reached STAR final heldout PSNR
`13.758882522583008`, STAR selected heldout PSNR `13.818532943725586`, and
direct splats heldout PSNR `11.15761947631836`. The paired seed-2 selected
artifact is stronger: STAR final heldout PSNR `13.873907089233398`, STAR
selected heldout PSNR `13.915654182434082`, direct splats heldout PSNR
`11.085673332214355`, selected STAR render-only eval
`0.034174208994954824s`, and direct-splat render-only eval
`0.40090283303288743s`. A seed-0 STAR-only `300 -> 0.05x` LR-decay repeat
kept the final checkpoint above V-JEPA too, with final heldout PSNR
`13.81613826751709` and heldout-selected diagnostic PSNR
`13.87098217010498` at step `600`; that confirms the schedule direction across
seeds 0, 1, and 2, but it skipped direct splats and the selected row uses
heldout selection. `--uvt-select-checkpoint first_train_psnr_plateau` is
now available as a non-heldout schedule diagnostic; the first seed-2 MPS run
selected step `400` with heldout PSNR `13.83452320098877` and
`uses_heldout_for_selection: false`, versus heldout-best step `500` at
`13.94494915008545`. The seed-1 repeat also selected step `400` without
heldout selection and reached heldout PSNR `13.771395683288574`, while final
was `13.790154457092285` and heldout-best was `13.826448440551758`.
`--uvt-select-train-psnr-plateau-patience 2` is now wired and smoke-tested; it
selected step `500` without heldout on seed 2, reaching heldout PSNR
`13.84100341796875` versus heldout-best `13.855948448181152`, but the seed-1
repeat selected step `500` at `13.704198837280273` while heldout-best was step
`400` at `13.726997375488281`. It remains a candidate, not a solved selector.
`--uvt-select-checkpoint first_train_psnr_gain_drop` is the best current
non-heldout selector candidate: on STAR-only seeds 0, 1, and 2 it selected
heldout PSNR `13.901209831237793`, `13.721198081970215`, and
`13.904694557189941`, respectively, without heldout-camera selection. The
paired seed-2 gain-drop report selected step `400` at heldout PSNR
`13.888997077941895` versus lens-aware direct splats at
`11.190529823303223`; selected STAR render-only eval was
`0.04640516696963459s` versus direct splats `0.9052186670596711s`. A paired
seed-1 repeat selected step `400` at heldout PSNR `13.879861831665039` versus
direct splats at `11.199346542358398`; selected STAR render-only eval was
`0.0712511669844389s` versus direct splats `0.7236963339382783s`. Gain-drop is
now the current reporting selector for the next scale/full-resolution probe,
but it remains a research rule rather than a production default.
The first 512px same-budget probe rejects a naive scale-up with the same
256-tube recipe: STAR completed only `70` steps and reached heldout PSNR
`9.205381393432617`, while direct splats completed `2095` steps and reached
`10.980579376220703`. Selected STAR re-eval render-only timing was still fast
at `0.09585146000000577s` versus direct splats `0.4002393330000018s`, but the
quality gap makes 512px a training-throughput/scale-strategy problem.
A formal 512px paired follow-up changed the STAR training window to one frame
and rescued the scale-up: STAR completed `1188` steps in the same local budget,
finished at heldout PSNR `13.701825141906738`, and rendered in
`0.11522445899981903s`. Direct dynamic splats reached heldout PSNR
`10.760580062866211` and rendered in `0.5262394170001699s`. The non-heldout
gain-drop selector chose step `600`, heldout PSNR `13.678083419799805`, and
render-only `0.11207666699988295s`; the heldout-best checkpoint was step `900`
at `13.729055404663086`. The next 512px work is scale-aware selection and
windowing strategy, not a forward-rasterizer rewrite first. A stricter
gain-drop threshold (`0.1` instead of `0.5`) delayed selection to step `800`,
but selected heldout PSNR stayed at `13.677860260009766` while final reached
`13.706064224243164`, so lower gain-drop threshold alone is not the selector
fix.
A seed-1 512px/window-1 repeat keeps the direct-splat win but not the V-JEPA
crossing. The paired seed-1 row with LR decay at step `300` reached STAR final
heldout PSNR `13.494588851928711` versus direct splats
`10.3926362991333`. Later STAR-only decay at step `500` improved final heldout
to `13.576004028320312`, while step `700` regressed to
`13.397587776184082`; both remain below the V-JEPA F32 reference `13.6248`.
A tube-capacity bracket changed the read but did not solve it. At 384 tubes,
seed 1 cleared V-JEPA at final heldout PSNR `13.640532493591309`, but seed 2
regressed to `13.4086275100708`; 384 is not the new default. At 320 tubes, seed
1 reached final heldout PSNR `13.682265281677246`, non-heldout selected PSNR
`13.637592315673828`, and heldout-best PSNR `13.769192695617676`, while seed 2
reached final/best heldout PSNR `13.637543678283691` and selected PSNR
`13.598714828491211`. Seed 0 shows why this is not a simple capacity promotion:
320 tubes reached heldout-best PSNR `13.70832633972168`, but the current
non-heldout selector fell to `13.437091827392578`; the matching 256-tube seed-0
run selected the heldout-best step and reached `13.636795043945312`. Moving the
320-tube seed-0 decay earlier to step `400` did not preserve the peak. The clean
checkpoint render-only timings stayed around `0.104-0.132s` with zero overflow
and zero unstable tiles, so 512px is still a scale-policy/selection problem, not
a first-priority rasterizer rewrite.
Checkpoint rows now include per-train-camera PSNR fields; the plain
`best_min_train_view_psnr` selector was rejected on seed 2 because it selected
the final checkpoint instead of the heldout peak.
`--uvt-optimizer-train-views first_only` plus
`--uvt-select-checkpoint best_train_view_psnr` was also rejected: the seed-2
one-camera dev split selected final step `972` with true heldout PSNR
`12.647516250610352`, well below the all-train recipe.

The combined `first_balanced_train_psnr_plateau` selector is now wired and
smoke-tested, but rejected as the current 512px rule. On 320 tubes, seed 0 with
gap max `1.0` selected step `400` at heldout PSNR `13.62360954284668`, but seed
1 gap max `1.0` missed the step-400 oracle peak and selected
`13.600866317749023`; gap max `1.2` then selected a bad step-500 shoulder at
`13.442426681518555` while final and heldout-best were much better. Keep it as
a diagnostic only.
A 512px seed-0 320-tube window-2 check is also negative: changing only
`--uvt-window-frames 1` to `2` reduced selected heldout PSNR to
`13.422961235046387` and heldout-best to `13.558874130249023`, versus the
window-1 seed-0 320-tube heldout-best peak `13.70832633972168`.
A hard LR-drop check is negative too: changing the same window-1 seed-0
320-tube branch to `--uvt-lr-decay-step 400 --uvt-lr-decay-factor 0.005`
completed `1000` steps but landed final/selected/heldout-best at the same
checkpoint, heldout PSNR `13.599885940551758`, selected render-only
`0.13297337400126708s`, max tile `103`, max pair ratio `3.2112812143984426`,
zero overflow, and zero unstable tiles. This keeps the speed story intact but
rejects hard scalar decay as the 512px seed-robustness fix.

The 128px single-video overfit lane also has a stronger equal-step result now.
Keeping the saved 64-splats/frame 200-step baseline fixed, the current
equal-step STAR-UVT speed recipe is 1728 tubes, LR `0.125`, spatial precision
`0.125`, temporal precision `2.0`, opacity `0.7`, Metal tile-backward with
`tile_t=1` and tile capacity `128`. A synchronized 20-repeat render benchmark
at equal 200 steps reaches STAR PSNR `23.199284076690674` in
`19.664952208000614s`, median render `4.426229001182946ms`; direct splats reach
PSNR `20.627903938293457` in `1177.423645084s`, median render
`203.9981664984225ms`. STAR is `+2.571380138397217` dB,
`59.87421848932689x` faster to train, and `46.08847993267007x` faster to render
by median. The 1728-tube equal-step recipe is quality-stable across seeds `0`,
`5`, and `13`: PSNR mean `23.212438424428303`, span `0.04626274108886719`.
Using the three-seed timing mean, it is still `39.65843015394074x` faster to
train and `34.37620167960197x` faster to render than direct splats. At 400
steps, the cap-128 quality recipe remains 1792 tubes, LR `0.12`; it reaches PSNR
`23.745369911193848`, train `81.65138491700054s`, median render
`10.333166999771493ms`, so it is the current 128px local cap-128 quality recipe.
The older `tile_t=2` 400-step row remains useful as a speed reference:
PSNR `23.569955825805664` in `31.383009374996618s`, render
`1.2202500001876615ms`. Raising tile capacity to `256` improves quality to
PSNR `24.085018634796143` with `tile_t=1`, train `38.323750625000685s`, median
render `4.634708000594401ms` at 400 steps. At equal 200 steps, cap `256` reaches
only PSNR `23.22518825531006` and renders in `11.385666999558453ms`, so cap
`128` remains the clean equal-step speed comparison. The cap-256 quality mode is
also stable across seeds `0`, `5`, and `13`: PSNR mean
`24.083155790964764`, span `0.05043506622314453`, train mean
`51.418234347000784s`, and mean median render `5.657340167090297ms`.
An LR bracket under `tile_t=1`, cap `128` does not change the default: LR `0.10`
reaches PSNR `23.786139488220215`, LR `0.11` reaches `23.796110153198242`, and
LR `0.14` regresses to `23.553497791290283`. The small lower-LR PSNR gain costs
enough time that cap `256` is the cleaner quality mode. A 200-step LR `0.11`
check also regresses to PSNR `23.128459453582764` and render
`19.63295899986406ms`, so the equal-step row stays LR `0.12`.
The cap `128` 400-step quality is not seed-fragile: seeds `0`, `5`, and `13`
reach PSNR `23.78368377685547`, `23.715169429779053`, and
`23.625149726867676` respectively, mean `23.708000977834065` with span
`0.15853404998779297`. Use the seed-5 row for matched timing claims and the
three-seed set for quality robustness.
The first native 256px single-video overfit gate uses `test_video_small.mp4`
instead of upscaling the 128px fixture. Scaling the 128px quality recipe to
7168 tubes shows that tile capacity is the real scale boundary: cap `256` is too
slow, half tube count is worse and not cheaper, and cap `64` breaks quality. The
current 256px recipe is 7168 tubes, LR `0.12`, spatial precision `0.125`,
temporal precision `2.0`, opacity `0.7`, `tile_t=1`, cap `128`. It reaches PSNR
`24.46974277496338` in `70.66461358399829s` at 200 steps and PSNR
`25.1381516456604` in `168.99901500000124s` at 400 steps. Median render is
`18.966646002809284ms` at 200 steps and `26.702728999225656ms` at 400 steps.
The 400-step 256px recipe is quality-stable across seeds `0`, `5`, and `13`:
PSNRs are `25.044105052947998`, `25.1381516456604`, and
`25.099973678588867`, with mean `25.09407679239909` and span
`0.09404659271240234`. MPS timing is noisier than quality, so use those rows as
robustness evidence rather than exact speed averages.
The local per-frame baseline now has an opt-in video-sampled initializer via
`--per-frame-init-mode video_samples`. A 256px 5-step feasibility probe with
64 splats/frame confirms that the Python per-frame baseline is not a practical
full same-step target at this resolution: video-initialized direct splats reach
PSNR `6.486777663230896` in `76.71433137500208s`, with median render
`779.2045410024002ms`; STAR reaches PSNR `13.252005577087402` in
`1.7836201249992882s`, median render `6.664707998425001ms`. This says the next
strong baseline should use a faster direct-splat renderer, not a 200-step
extension of this Python per-frame harness.
An explicit temporal-piece init, `--uvt-sample-mode temporal_quarters`, is
rejected for this recipe: at the same 400 steps it reached PSNR
`23.275623321533203`, losing `0.29433250427246094` dB to random sampling.
The current temporal split/refine operator is also rejected: splitting the
1792-tube model at step 200 into 3584 temporal children loses badly at 400
steps. The best split variant tested used offset `0.25`, precision scale `1.0`,
and child opacity scale `2.0`, but reached only PSNR `21.271286010742188`.
Even the render-preserving duplicate split is rejected: offset `0.0`, precision
scale `1.0`, and zero depth offset reached only PSNR `19.785715341567993` and
still jumped split-boundary loss from `0.004865488037467003` to
`0.02108645997941494`. A targeted split-boundary probe confirms this is a
Metal tile fallback problem in the rejected split path: the 1792-tube pre-split
render is PSNR `23.139398097991943`, forward `9.447625001484994ms`, stable tile
fraction `1.0`; the 3584-tube duplicate render drops to PSNR
`17.03884720802307`, forward `110.73370899975998ms`, stable tile fraction
`0.0`, unstable tile fraction `1.0`, and overflow tile count `1898`.
The `tile_t=1` boundary probe improves the pre-split render to PSNR
`23.180255889892578`, forward `4.090958998858696ms`, but the duplicate split
still drops to PSNR `18.50017786026001`, forward `85.22179200008395ms`, with
unstable tile fraction `1.0`. A controlled 896-to-1792 scheduled split under
`tile_t=1` reaches only PSNR `22.63478994369507`, so split/refine remains
rejected for the current local recipe.
Motion-aware block-match init is also rejected under the current recipe. Raw
block match reaches PSNR `22.520790100097656`; gated block match improves that
to `23.172695636749268` at best, still below the zero-velocity recipe.
Longer training is not the next quality lever either: 800 constant-LR steps
fall to PSNR `22.233996391296387`, and 800 steps with LR `0.12 -> 0.04` at
step 400 reaches `23.460845947265625`, still below the 400-step recipe.
That overfit support lever does not transfer to 512px multicam as a global
temporal floor: `--uvt-min-lambda-t 0.7` fell to heldout-best
`13.514815330505371`, and `--uvt-min-lambda-t 2.0` fell to
`13.086738586425781`, both with clean Metal stats.
`--uvt-select-checkpoint first_train_view_gap_collapse` is also wired and
smoke-tested, but rejected as the current 512px selector. A post-hoc threshold
`0.7` looked promising on saved 320-tube curves, but the live seed-0 MPS run
selected step `400` at heldout PSNR `13.490143775939941`, while final was
`13.70932674407959` and heldout-best was step `900` at
`13.739107131958008`.
The first lightweight validation-rendered subset is now tested and rejected:
`--uvt-validation-frame-stride 4 --uvt-validation-frame-offset 1` holds out
train-camera frames `[1, 5, 9, 13]`, and
`--uvt-select-checkpoint best_train_dev_frame_psnr` selects by those frames
without using the true heldout camera. A clean fit-init run excludes those
frames from initialization too via `--uvt-init-frames fit`, but selected the
final step `1059` at true heldout PSNR `13.461018562316895`; its heldout-best
checkpoint was only `13.579765319824219`. The leaky all-init control also
selected final and reached only `13.39246654510498` true heldout. This rejects
train-camera temporal dev frames as the next 512px selector lane.
The free temporal-support init check is also negative: changing only
`--uvt-init-lambda-t` to `2.0` without a `min_lambda_t` floor selected step
`600` at heldout PSNR `13.213919639587402`; final was `13.40640640258789`,
and heldout-best was `13.415067672729492`. Metal stayed clean, but this is far
below the earlier no-floor 320-tube peak, so the overfit temporal-support win
does not transfer as either a hard floor or a narrower initialization bias.
Bounded sequence consistency is now tested and rejected as the next 512px fix.
The full 16-frame consistency backward hit an MPS `12.00 GiB` invalid-buffer
failure; four-frame consistency every 20 steps was step-starved at `340` steps
in 60 seconds and heldout PSNR `13.58269214630127`. Four-frame consistency every
50 steps was clean and faster, selecting step `600` at true heldout PSNR
`13.619542121887207` with `0.11323120699853462s` render-only time, and ending at
heldout-best `13.626453399658203`. That is only a near-tie with the V-JEPA row
and below the no-consistency 320-tube oracle peak, so current work should stay
on support/window or multiscale policy before rasterizer work.
The first multiscale auxiliary loss bracket is mixed and not a default. The
hook adds `--uvt-multiscale-loss-weight` and
`--uvt-multiscale-loss-factor`, reusing the existing render and adding a
downsampled reconstruction term. At factor `4`, weight `0.25`, seed 0 improved
final/heldout-best to `13.656238555908203`, but the non-heldout gain-drop
selector fired early at `13.591398239135742`. Seed 1 rejects the setting:
the no-multiscale 320-tube row had selected/final/heldout-best
`13.637592315673828` / `13.682265281677246` / `13.769192695617676`, while the
multiscale run fell to `13.418802261352539` / `13.384246826171875` /
`13.520487785339355`. Lowering the same global factor-4 auxiliary to weight
`0.05` improved seed 1 but still stayed below no-multiscale, at selected /
final / heldout-best `13.568358421325684` / `13.560102462768555` /
`13.603869438171387`. Keep the hook, but next try a more selective scale/crop
policy rather than promoting simple global factor-4 multiscale loss.
The first deterministic crop-loss branch is also negative. The hook adds
`--uvt-crop-loss-weight` and `--uvt-crop-loss-size`, cycling through a 3x3 grid
of full-resolution train crops from the existing render. On seed 1, crop size
`256` at weight `0.25` reached only `578` steps and heldout PSNR
`13.566254615783691`; crop size `128` at the same weight reached `584` steps,
selected `13.48839282989502`, final `13.565434455871582`, and heldout-best
`13.591614723205566`. Metal stayed clean, but the branch is slower and worse
than no-crop, so deterministic crop loss is not the 512px support fix.
A 512px deterministic cycle-schedule revisit is mixed. Keeping the no-aux
320-tube/window-1 recipe and changing only `--uvt-train-schedule cycle`, seed 0
selected step `600` at heldout PSNR `13.798948287963867` and seed 1 selected
step `600` at `13.915006637573242`, both without using heldout for selection.
Their heldout-best checkpoints are step `500` at `13.841632843017578` and
`13.91877555847168`, and selected render-only eval stays fast at
`0.10962479100089695s` and `0.11339320700062672s` across the three eval
sequences. Seed 2 rejects the rule: LR `0.01` selected only
`13.023938179016113`, heldout-best was `13.442176818847656`, and the run became
non-finite after step `480`; LR `0.005` stayed finite but underfit at
`13.037338256835938`. Current read: cycle proves sampling order matters, but do
not promote plain cycle as the 512px default. Try a shuffled/phase-randomized
coverage schedule or a stability guard before rasterizer work.
The trainer now includes that stability guard at the existing log cadence:
non-finite loss records `stopped_reason` / `stopped_step`, writes non-finite
scalar log fields as `null`, and restores the last checkpointed finite state
before final eval. The normal CPU smoke
`multicam_heldout_compare_nonfinite_guard_smoke3_16_2f_1s` passed, and the
forced CPU smoke `multicam_heldout_compare_nonfinite_guard_forced_smoke_16_2f_1s`
stopped at step `1` with `stopped_reason: nonfinite_loss`.

The follow-up `shuffled_cycle` schedule is now wired and smoke-tested. It keeps
the same renderer and no-aux 320-tube/window-1 capacity, but shuffles coverage
within deterministic cycles. On seed 2, `mcam512_s2_t320_shuffled_cycle` stayed
finite and selected step `600` at heldout PSNR `13.574305534362793`, render-only
`0.13692954100042698s`; its heldout-best checkpoint was step `300` at
`13.6640625`. That fixes the plain-cycle collapse but does not yet give a
non-heldout selected checkpoint that clears V-JEPA. Earlier LR decay at step
`300` is rejected (`13.564926147460938` selected, `13.578947067260742`
heldout-best), and a live balanced-plateau selector rerun did not reproduce the
saved curve's better shoulder (`13.52459716796875` selected,
`13.557467460632324` heldout-best). Current read: this is still a quality and
selection-stability problem, not a rasterizer-first problem. The fixed-step
check `mcam512_s2_t320_shuffled_cycle_fixed300` confirms that: exact step `300`
landed at heldout PSNR `13.527697563171387`, so the earlier heldout-best
`13.6640625` shoulder is not reproducible enough to promote.
The new `reshuffled_cycle` schedule keeps full coverage but reshuffles each
epoch. It is smoke-tested and is the first three-seed 512px schedule candidate
whose non-heldout selected checkpoints all clear the V-JEPA F32 reference:
fixed-600 seed 0 selected/final `13.639025688171387`, seed 1
`13.894740104675293`, and seed 2 selected `13.700183868408203` at step `500`
with final `13.660613059997559`. Treat this as a robust-floor candidate, not a
pure upgrade: it rescues seed 2, but seed 0 is substantially weaker than plain
cycle.
The follow-up `phase_rotated_cycle` schedule is also wired and smoke-tested, but
is rejected for now. It was good on seed 2 (`13.706971168518066` selected,
`0.11264262600161601s` render-only) but failed seed 0 at `13.602667808532715`,
below the V-JEPA reference and below reshuffled.
The latest `view_shuffled_cycle` schedule keeps temporal frame/window order and
shuffles only the train-camera order inside each slot. It is now the strongest
robust-floor candidate: selected heldout PSNR is seed 0 `13.639522552490234`,
seed 1 `13.7864990234375`, seed 2 `13.788138389587402`, all without heldout
selection. Fixed-600 final checkpoints are better on seeds 1 and 2
(`13.812097549438477` and `13.793721199035645`), so the remaining issue is the
selector firing early on this schedule.
The existing `best_train_psnr` selector fixes that reporting issue for
`view_shuffled_cycle`: the smoke
`multicam_heldout_compare_view_shuffled_besttrain_selector_smoke_16_2f_1s`
passed, and on the saved three-seed 512px curves it selects step `600` for all
three seeds without heldout selection: `13.639522552490234`,
`13.812097549438477`, and `13.793721199035645`.
