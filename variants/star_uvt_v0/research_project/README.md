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

Current orientation note:

- `attempts/2026-05-12_star_uvt_state_review.md` answers the current
  forward-speed versus training-backward status: sparse UVT forward is measured,
  but fast deterministic sublinear training backward is still open.
- `attempts/2026-05-12_variable_camera_attempts.md` records the moving-camera
  projection split: Attempt 1 is dynamic first-order UVT, Attempt 2 is
  piecewise camera-time UVT segments, and the per-frame loop remains only a
  negative control.

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
python3 research_project/benchmarks/projective_atlas_scaling_probe.py \
  --out-json research_project/benchmarks/results/projective_atlas_scaling_probe.json
python3 research_project/benchmarks/projective_atlas_scaling_probe.py \
  --run-metal --iterations 1 --warmup-iterations 1 \
  --out-json research_project/benchmarks/results/projective_atlas_scaling_probe_interval_metal.json
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
  --uvt-tile-t 1 --uvt-tile-capacity 128 \
  --out-json research_project/benchmarks/results/uvt_forward_speed_probe_64_16f_224_448_tuned_v2.json
python3 research_project/benchmarks/uvt_backward_breakdown_probe.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small.mp4 \
  --target-size 256 --max-frames 16 --tube-count 7168 \
  --uvt-tile-t 1 --uvt-tile-capacity 128 \
  --out-json research_project/benchmarks/results/uvt_backward_breakdown_probe_256_16f_7168_s0125_t20_tilet1_cap128.json
python3 research_project/benchmarks/uvt_train_step_timing_probe.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small.mp4 \
  --target-size 256 --max-frames 16 --tube-count 7168 \
  --uvt-tile-t 1 --uvt-tile-capacity 128 --steps 30 --warmup-steps 2 \
  --out-json research_project/benchmarks/results/uvt_train_step_timing_probe_256_16f_7168_s0125_t20_tilet1_cap128_30steps.json
python3 research_project/benchmarks/uvt_train_step_timing_probe.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small.mp4 \
  --target-size 256 --max-frames 16 --tube-count 7168 \
  --uvt-tile-t 1 --uvt-tile-capacity 128 --steps 20 --warmup-steps 2 \
  --sample-count-every 5 --tile-load-reg 0.003 --tile-load-target 60 \
  --out-json research_project/benchmarks/results/uvt_train_step_timing_probe_256_16f_7168_s0125_t20_tilet1_cap128_20steps_samplecount_tileloadreg0003_target60.json
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small.mp4 \
  --target-size 256 --max-frames 16 --tube-count 7168 --steps 50 \
  --lr 0.12 --device mps --uvt-init-mode video_samples \
  --uvt-spatial-precision 0.125 --uvt-temporal-precision 2.0 --uvt-opacity 0.7 \
  --uvt-render-backend metal_tile --uvt-tile-t 1 --uvt-tile-capacity 128 \
  --uvt-tile-load-reg 0.003 --uvt-tile-load-target 60 \
  --render-benchmark-repeats 10 --skip-per-frame \
  --out-json research_project/benchmarks/results/video_fit_single_overfit_256_16f_50steps_7168uvt_lr012_s0125_t20_tilet1_cap128_tileloadreg0003_target60_uvtonly_renderbench10_metal_tile.json \
  --contact-sheet research_project/benchmarks/results/video_fit_single_overfit_256_16f_50steps_7168uvt_lr012_s0125_t20_tilet1_cap128_tileloadreg0003_target60_uvtonly_renderbench10_metal_tile.png
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small.mp4 \
  --target-size 256 --max-frames 16 --tube-count 7168 --steps 200 \
  --lr 0.12 --device mps --uvt-init-mode video_samples \
  --uvt-spatial-precision 0.125 --uvt-temporal-precision 2.0 --uvt-opacity 0.7 \
  --uvt-render-backend metal_tile --uvt-tile-t 1 --uvt-tile-capacity 128 \
  --uvt-tile-load-reg 0.003 --uvt-tile-load-target 60 \
  --render-benchmark-repeats 10 --skip-per-frame \
  --out-json research_project/benchmarks/results/video_fit_single_overfit_256_16f_200steps_7168uvt_lr012_s0125_t20_tilet1_cap128_tileloadreg0003_target60_uvtonly_renderbench10_metal_tile.json \
  --contact-sheet research_project/benchmarks/results/video_fit_single_overfit_256_16f_200steps_7168uvt_lr012_s0125_t20_tilet1_cap128_tileloadreg0003_target60_uvtonly_renderbench10_metal_tile.png
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
latest all-frames rows keep render-only STAR faster than direct splats. A
native-256px raw-forward tile-shape probe now shows that `tile_t=4` can make the
raw Metal forward path sublinear versus the sliced per-frame pair count
(`0.6304844910456923` pair ratio at 7168 tubes), but the matching 50-step
trainer row is slower and slightly worse than `tile_t=1`. Treat this as a
backward/reduction-wrapper blocker, not as a missing raw-forward rasterizer.
The focused backward probe confirms why: `tile_t=1`, `tile_t=2`, and `tile_t=4`
all emit `5,342,341` compact backward samples on the same 256px/16-frame
initialized scene, so the backward path is still per-pixel sample based rather
than tile-pair compact. The 30-step phase probe is backward dominated, with
median backward `391.97191700077383ms` for `tile_t=1` and
`358.5281044979638ms` for `tile_t=4`. A follow-up train-step sample-count hook
shows that initialized-scene count is only the starting snapshot: during a
20-step run, `tile_t=1` grows from `8,076,866` to `23,232,213` compact samples,
and `tile_t=4` grows from `8,076,861` to `22,333,671`. The speed blocker is
therefore not just a fixed reducer overhead; optimization can expand the
per-pixel backward sample volume itself. An opt-in tile-load regularizer in the
same timing probe can contain that growth: weight `0.003`, target `60` cuts
median `tile_t=1` total/backward from `454.1237500052375ms` /
`414.0798125008587ms` to `268.05322949803667ms` / `258.73991700063925ms`, with
final recon loss `0.012283111922442913`; weight `0.01` cuts sample rows further
but worsens final recon loss to `0.015173783525824547`. Treat this as a
speed/quality bracket, not as the final sublinear backward design. The paired
sample-row versus tile-pair diagnostic makes the missing backward target
concrete: the unregularized 20-step `tile_t=1` row has about `12.90x` to
`17.02x` more compact backward sample rows than raw-forward UVT tile/tube pairs
at the sampled points; the `0.003` tile-load row still has about `12.90x` to
`16.00x` more sample rows than tile pairs, though it sharply reduces overflow.
The real sublinear backward should work in tile-pair space or avoid emitting
those per-pixel rows. The 50-step
single-video overfit transfer is consistent: unregularized `tile_t=1` reaches
PSNR `22.363874912261963` in `25.242409541999223s`, median render
`9.427915996639058ms`; weight `0.003` reaches PSNR `21.97382688522339` in
`18.74674650000088s`, median render `5.818312503834022ms`; weight `0.01`
reaches PSNR `21.31075143814087` in `10.924605249994784s`, median render
`5.032062501413748ms`. At 200 steps, weight `0.003` reaches PSNR
`23.976197242736816` in `42.51604850000149s`, median render
`5.74274999962654ms`; the unregularized UVT-only 200-step row reaches PSNR
`24.46974277496338` in `70.66461358399829s`, median render
`18.966646002809284ms`. The support knob still helps at 200 steps, but it is a
quality/speed tradeoff.
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
extension of this Python per-frame harness. The benchmark now has that faster
path as `--per-frame-render-backend fast_mac`, using the existing `v6_refined`
projected-Gaussian autograd renderer. With the same fair video-init settings as
the old dense row, the 5-step direct baseline reaches PSNR `6.473215818405151`
in `0.4040724170008616s`, median render `5.363958000089042ms`, so the bottleneck
is removed. The corrected 200-step same-step row reaches STAR PSNR
`24.47240114212036` in `64.06204816699756s`, median render
`16.813333500977024ms`, while fast direct splats reach PSNR
`20.513088703155518` in `5.162235333002172s`, median render
`6.129583500296576ms`. A same-wall-clock direct-only run with `2500` fast direct
steps reaches PSNR `21.23270273208618` in `69.8356277089988s`, median render
`8.872333499311935ms`. This preserves a large 256px STAR quality lead at equal
steps and comparable train time, but it also means STAR is not a render-speed
win against the fast direct-splat baseline at this setting.
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
`13.812097549438477`, and `13.793721199035645`. A live paired seed-0 rerun
weakens that selector claim: `mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct`
selects step `600` at heldout PSNR `13.597569465637207`, below the V-JEPA F32
reference, while the heldout-best checkpoint in the same curve is step `400` at
`13.696317672729492`. Direct splats in that same run reach only heldout PSNR
`8.24771499633789`, so the remaining issue is selector robustness, not the
direct-splat baseline.
The balanced selector is not the fix: the paired seed-0 rerun
`mcam512_s0_t320_view_shuffled_cycle_fixed600_balanced_d03_gap165_paired_direct`
selects step `400` at heldout PSNR `13.495000839233398`, and even its
heldout-best checkpoint is only `13.591666221618652`. Treat the next blocker as
saved-curve/live-rerun variance and determinism before more selector tuning.
A STAR-only rerun confirms that framing:
`mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_staronly_rerun1` selects
step `600` at heldout PSNR `13.886017799377441` with zero overflow/unstable
tiles, and `mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_staronly_rerun2`
selects step `600` at `13.815839767456055`. The paired best-train rerun
`mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct_rerun2`
also clears V-JEPA at `13.730086326599121`, while direct splats stay at
`8.247519493103027`. The runner trains STAR before the direct-splat branch, so
the earlier paired negatives are not explained by direct splats running first.
That paired matrix is now measured and seed 2 is the blocker: seed 0 reaches
`13.730086326599121`, seed 1 reaches `13.756308555603027`, but seed 2 reaches
only `13.593206405639648`, and a seed-2 repeat reaches
`13.608675003051758`. Direct splats stay near `8.25` heldout PSNR. Current read:
STAR beats direct splats decisively, but this is not a robust V-JEPA replacement
until seed 2 clears.
The seed-2 STAR-only repeat
`mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_staronly_rerun1` does
clear, selecting step `600` at heldout PSNR `13.788966178894043`. That moves the
blocker from "seed 2 cannot fit" to live variance/determinism under the current
MPS recipe: two paired seed-2 rows miss V-JEPA, while the STAR-only row clears,
and STAR still trains before the direct-splat branch. Do not escalate to
full-resolution until the paired 512px seed-2 miss is explained or controlled.
The env-captured follow-up preserves the same read. `multicam_heldout_compare.py`
now records argv/env/torch metadata and supports `--torch-deterministic
{off,warn,error}`. The real `warn` control was too slow to be useful on MPS, but
the practical controlled launch with `PYTHONHASHSEED=0` and MPS fallback off
missed again in paired mode:
`mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct_envcapture_rerun3`
selected heldout PSNR `13.54917049407959`, while the matching STAR-only run
`mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_staronly_envcapture_rerun2`
cleared at `13.783736228942871`. Treat this as recipe/MPS fragility, not a
full-resolution green light.
`multicam_star_repeatability_probe.py` then removed the direct-splat branch
entirely. The CPU smoke repeated exactly (`0.0` PSNR span and `0.0` state delta),
but the real seed-2 MPS two-repeat artifact
`mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_repeatability_envcapture_2x.json`
landed at selected heldout PSNR `13.818974494934082` and `13.716343879699707`
with different state digests and final state max delta `1.6650149822235107`.
That isolates the blocker to MPS/custom-backward repeatability inside STAR
training, not the direct-splat branch.
`uvt_gradient_repeatability_probe.py` localizes the source further. On the real
512px seed-2 fixed first window, three identical calls produce three unique
`stable_backward_samples` digests, three unique fixed-sample `index_add_`
reduction digests, and three unique full-autograd gradient digests while scalar
loss span stays `0.0`. The full one-step gradient max delta is only
`4.423782229423523e-09`, but the fixed-bundle reduction max delta reaches
`0.3125` in `grad_q`; over hundreds of Adam steps this is enough to explain the
observed state drift.
The follow-up sorted-reduction diagnostic keeps default training on MPS
`index_add_`, but adds an opt-in `sorted_cpu` reducer to the tile-backward bridge
and exposes it through `uvt_gradient_repeatability_probe.py`. On the 512px
seed-2 fixed window,
`mcam512_s2_t320_view_shuffled_cycle_gradient_repeatability_step0_sortedcpu_autograd.json`
has three unique raw sample digests and three unique default `index_add_`
digests, but one unique fixed `sorted_cpu` digest, one unique canonical generated
sample reduction digest, and one unique full-autograd gradient digest. This
controls one-step gradient drift for the fixed window, so the next target is a
device-resident deterministic reducer or kernel-side sample ordering fix, not a
CPU reducer as the training design.
The first on-device diagnostic reducer now exists too:
`reduce_sample_bundle_scan` is a custom Metal scan reducer, exposed as
`scan_metal` and `sort_scan_metal` modes. It proves fixed-bundle reduction can be
stable on MPS, but the real 512px fixed-window probes
`mcam512_s2_t320_view_shuffled_cycle_gradient_repeatability_step0_scanmetal_autograd.json`
and
`mcam512_s2_t320_view_shuffled_cycle_gradient_repeatability_step0_sortscanmetal_autograd.json`
still have three unique generated-reduction digests and three unique autograd
digests. Read: on-device float32 reduction wrappers are not enough at real scale;
we need kernel-side deterministic sample keys/order or the tile-pair/per-tube
VJP path.
That kernel-side key diagnostic now exists. `stable_backward_samples_with_keys`
emits deterministic per-sample keys, and `key_sort_scan_metal` sorts by
`(tube_id, key)` before the custom Metal scan. The tiny MPS artifact
`uvt_gradient_repeatability_probe_mps_smoke_16_2f_t16_keysortscan_autograd.json`
has one generated keyed-reduction digest and one full-autograd digest. The real
512px fixed-window artifact
`mcam512_s2_t320_view_shuffled_cycle_gradient_repeatability_step0_keysortscan_autograd.json`
still has three raw sample digests and three default `index_add_` digests, but
one generated keyed-reduction digest and one full-autograd digest with zero
gradient delta. This controls the current one-step MPS drift; it is not the
final speed design, because backward still emits `484697` per-pixel sample rows
on that fixed window instead of operating directly in tile-pair/per-tube space.
The real multicam train harness also accepts the same opt-in path via
`--uvt-reduction-mode key_sort_scan_metal --uvt-sample-emission-mode with_keys`;
the 1-step MPS smoke
`multicam_heldout_compare_keysortscan_smoke_16_2f_1s` passed and records those
modes in both `run_meta.json` and `comparison_report.json`. The same flags are
wired into `multicam_star_repeatability_probe.py`; the 2-repeat keyed MPS smoke
`multicam_star_repeatability_probe_keysortscan_smoke_16_2f_1step.json` reports
`final_state_max_abs: 0.0`. The real 512px seed-2 training repeats now clear at
20, 100, and 600 steps with the same keyed path. The 600-step artifact
`mcam512_s2_t320_view_shuffled_cycle_keysortscan_repeatability_600steps.json`
has final state max delta `0.0`, final train/heldout PSNR span `0.0`, final
train PSNR `15.208253383636475`, and final heldout PSNR
`13.75709342956543` in both repeats. This closes the training-repeatability
gate for the current per-pixel sample path, but it does not close the speed
claim; backward still needs a true tile-pair/per-tube VJP to be sublinear.

The timing probes now expose the same keyed path too. The bounded 32px/16f/224
tube comparison shows why this is not the final rasterizer: `index_add` backward
breakdown has median sample+reduce `6.305208502453752` ms, while
`key_sort_scan_metal` has `21.90893750230316` ms on the same `135565` sample
rows. In the actual train-step timing probe, median backward rises from
`34.564916997624096` ms to `77.67279200197663` ms. Median compact samples are
`188522` versus `11833` UVT tile-tube pairs, about `15.93` sample rows per tile
pair.

The first tile-pair backward emitter is now implemented and exposed as
`--uvt-sample-emission-mode tile_pair`. It passes parity against the keyed
per-pixel backward on stable tiny scenes: `4159 -> 128` rows at 16px/2f with
max reduced-gradient delta `7.62939453125e-05`, and `135565 -> 10556` rows at
32px/16f/224 tubes with max `grad_q` delta `0.00030517578125`. That proves the
right row space. It is not yet the speed win: the 32px breakdown records median
sample+reduce `23.357353999017505` ms, and the actual train-step probe records
median backward `76.03254100104095` ms. That roughly matches keyed per-pixel
backward (`77.67279200197663` ms) but still loses to default
`index_add` (`34.564916997624096` ms). Treat this kernel as a diagnostic
tile-pair VJP scaffold; the next useful speed work is optimized per-tile
accumulation or fused reduction.

The next diagnostic speed path is direct atomic accumulation, exposed as
`--uvt-sample-emission-mode direct_atomic`. It bypasses compact sample rows and
the Python/MPS reducer, accumulating directly into per-tube gradients. The 32px
parity row reduces `135565` per-pixel rows to `224` direct tube-gradient rows
with max `grad_q` delta `0.0003662109375` and zero unstable tiles. It is the
first train-step speed win: at 32px/16f/224 tubes, median backward is
`8.311833000334445` ms versus default `index_add` at
`34.564916997624096` ms, keyed at `77.67279200197663` ms, and naive tile-pair
at `76.03254100104095` ms. At 256px/16f/7168 tubes, the 5-step row records
median backward `41.306667000753805` ms and total step `52.94633399898885` ms,
versus the older default 30-step row at median backward
`391.97191700077383` ms and total `403.4594580043631` ms. The flag is also now
accepted by the multicam comparison and repeatability CLIs. The tiny fixed-step
repeatability smoke
`multicam_star_repeatability_probe_directatomic_smoke_16_2f_2steps_fixedbudget.json`
runs two 16px/2-frame/16-tube MPS repeats for two steps each with final state
max delta `0.0`. The first bounded useful-size gate,
`mcam32_s2_t224_directatomic_repeatability_20steps.json`, runs two
32px/16-frame/224-tube MPS repeats for 20 steps each. It is not bit-exact:
final state max delta is `0.0001089535653591156` and state digests differ, even
though train/heldout PSNR spans are only about `1e-06` and `2e-06`. Do not
promote this as the deterministic reporting path yet: global float atomics can
accumulate in variable order. The same-step single-video overfit sanity check
does clear: the 256px/16-frame/7168-tube, 50-step tile-load row reaches PSNR
`21.97383165359497` with direct atomic versus `21.97382688522339` in the earlier
default-backward artifact, while UVT fit wall time drops from
`18746.74650000088` ms to `5612.386834000063` ms. The 200-step same-recipe row
also clears: direct atomic reaches PSNR `23.977155685424805` versus
`23.976197242736816`, while fit wall time drops from `42516.04850000149` ms to
`16431.616750000103` ms. Next gate is a small multicam row, with keyed reduction
kept as the deterministic reporting path.

That small multicam row now exists as
`mcam32_s2_t224_directatomic_paired20/comparison_report.json`. At
32px/16f/20 steps, STAR direct-atomic reaches train/heldout PSNR
`8.433213710784912` / `8.312164306640625` versus direct splats at
`8.406636238098145` / `8.236161231994629`, with train loop `1.6225384999997914`
s versus `2.838062375005393` s and render-only eval `0.03705504100798862` s
versus `0.5165199170005508` s. Treat this only as a smoke-scale paired row; the
256px same-to-same gate below is the meaningful multicam read.

The 256px same-to-same multicam gate now exists too. In
`mcam256_s0_t256_directatomic_fixed365_paired/comparison_report.json`, STAR
direct atomic runs the same `365` STAR steps as the prior seed-0 default row and
reaches heldout PSNR `13.968801498413086` in `16.86269975000323` s; the prior
default row was `13.769630432128906` in `20.019632791983895` s. In
`mcam256_s0_t256_directatomic_20s_paired/comparison_report.json`, the same
20-second budget reaches STAR heldout PSNR `13.885225296020508` in `367` steps,
while direct splats reach `8.864513397216797` in `812` steps. This is a real
256px paired win, and the three-seed direct-atomic 20-second read now clears
the V-JEPA F32 heldout reference on every seed: seed 0 reaches
STAR/direct-splat heldout PSNR `13.885225296020508` / `8.864513397216797`,
seed 1 reaches `13.736623764038086` / `9.101115226745605`, and seed 2 reaches
`13.848220825195312` / `9.198732376098633`. Against the prior default-backward
STAR rows, direct atomic is mixed rather than a pure quality win: seed 0 and
seed 2 improve, seed 1 is about `0.032` dB lower. Multicam wall-clock still has
overhead beyond the raw backward kernel, so keep keyed reduction as the
deterministic reporting path.

The 512px direct-atomic fixed-600 paired matrix is now measured too. It keeps
the same 512px/16-frame, 320-tube, `view_shuffled_cycle`, fixed-600,
`best_train_psnr` recipe as the prior paired rows and changes only the STAR
backward path to `direct_atomic`/`index_add`. STAR heldout PSNR is seed 0
`13.669089317321777`, seed 1 `13.904035568237305`, and seed 2
`13.819857597351074`, versus direct splats at `8.247410774230957`,
`8.256668090820312`, and `8.267980575561523`. The STAR floor is
`+0.04428931732177688` dB above the V-JEPA F32 heldout reference. Compared with
the nearest prior paired default-backward matrix, direct atomic fixes the seed-2
miss and raises the matrix mean, but it is not a pure upgrade over every saved
STAR row because seed 0 is below the strongest paired seed-0 rerun. This is the
best current 512px paired matrix, but deterministic repeatability remains open.

That repeatability risk is now measured and remains a blocker:
`mcam512_s2_t320_view_shuffled_cycle_directatomic_repeatability_600steps.json`
runs two same-process seed-2, 512px, fixed-600 direct-atomic repeats. The final
state max/mean abs delta is `1.431039810180664` / `0.11831939475876944`, final
train PSNR span is `0.1379227638244629`, and final heldout PSNR span is
`0.03300189971923828`. Both repeats clear V-JEPA, but direct atomic is not an
exact-repeat reporting path at 512px.

Shorter probes localize the failure: at 20 steps the state digests already
differ but PSNR spans are `0.0`; at 100 steps state max abs delta has grown to
`0.03318440169095993`, with heldout PSNR span `0.000308990478515625`. The drift
then compounds into visible 600-step variation.

The deterministic tile-pair route is now measured in the same repeatability
harness. `tile_pair + key_sort_scan_metal` is exact at 32px and at 512px for
20, 100, and 600 steps. The 512px 600-step artifact
`mcam512_s2_t320_view_shuffled_cycle_tilepair_keysortscan_repeatability_600steps.json`
has final state max/mean abs delta `0.0`, train/heldout PSNR span `0.0`, train
loops `112.29406904200005s` and `85.4813433330055s`, and heldout PSNR
`13.569375991821289`. That is faster than the deterministic per-pixel keyed
path (`221.05617970800085s` / `206.5364632500059s`) but slower than direct
atomic (`~22.8s`) and lower quality than both keyed (`13.75709342956543`) and
direct atomic (`13.713482856750488` / `13.746484756469727`). Treat tile-pair as
the first exact compact-row training path, not yet the reporting path; its
gradient parity/quality gap needs to be fixed before optimizing it as the
promotion rasterizer.

The local gradient gap is now measured and is small. The gradient probe can
compare two backward modes on the same fixed window and after optional pretrain.
For keyed per-pixel versus tile-pair on the 512px seed-2 window, max
model-gradient delta is `6.51925802230835e-09` at initialization,
`1.2777745723724365e-06` after 600 direct-atomic pretrain steps, and
`7.320195436477661e-07` after 600 tile-pair pretrain steps. Each comparison has
loss delta `0.0` and exact repeatability inside both modes. The tile-pair
quality gap is therefore not a gross local VJP mismatch; the next question is
trajectory sensitivity from deterministic floating-point accumulation order.

The matched trajectory replay confirms that read. At 100 steps, tile-pair and
keyed per-pixel have already diverged in state space (`0.02207188308238983` max
abs), but train/heldout PSNR deltas are still only
`-0.00025272369384765625` / `-0.00014972686767578125`. At 600 steps, keyed
heldout is `13.75709342956543`, tile-pair heldout is `13.569375991821289`, and
the final state max/mean abs delta is `1.417811632156372` /
`0.12496386851583208`. The matched checkpoint max state delta jumps between
step 100 and 200, before LR decay. This points to accumulation-order trajectory
drift as the next implementation target.

The Kahan-only accumulation test is now rejected. `tile_pair_compensated` is
wired as an opt-in diagnostic mode and passed the tiny MPS runtime smoke
`uvt_backward_breakdown_probe_tilepair_compensated_smoke_16_2f_16t_tilet1.json`.
On the 512px seed-2 100-step matched replay, however, compensated tile-pair has
state max/mean abs delta `0.040996529161930084` /
`0.00013770289037243595` versus keyed per-pixel. Plain tile-pair at the same
gate was `0.02207188308238983` / `0.00013591208698926494`. The PSNR deltas are
still tiny, but state tracking is worse, so the next useful rasterizer
experiment is an ordered deterministic compact-row reducer rather than
compensated summation alone.

The compensated final keyed scan is rejected too. The new
`key_sort_compensated_scan_metal` reducer passed its tiny MPS smoke, but the
512px seed-2 100-step replay has state max/mean abs delta
`0.03426568582653999` / `0.00013610214602002607`, worse than plain tile-pair's
`0.02207188308238983` / `0.00013591208698926494`. The remaining issue is not
fixed by compensation around the existing compact sums.

The LR bracket narrows the failure further. Reducing LR from `0.01` to `0.005`
cuts the keyed-vs-tile-pair state max delta at step 200 from
`0.5959800481796265` to `0.029318034648895264`, so optimizer step size affects
how quickly the compact path separates. It is not enough as a recipe: at 600
steps, LR `0.005` gives tile-pair heldout PSNR `13.314628601074219`, while LR
`0.0075` gives `13.562674522399902`; neither beats the stronger deterministic
keyed LR `0.01` heldout row `13.75709342956543` or clears the promotion bar.
Keep LR tuning as diagnosis, not the next rasterizer fix.

The scanline compact-row split is rejected as well. `tile_pair_scanline` passed
the tiny MPS smoke
`uvt_backward_breakdown_probe_tilepair_scanline_smoke_16_2f_16t_tilet1.json`,
but its 512px seed-2 100-step replay
`mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_scanline_100steps.json`
has state max/mean abs delta `0.02909490466117859` /
`0.00013122651705219012`, worse than plain tile-pair's `0.02207188308238983`
max at the same gate. It is also slower than keyed in that short replay
(`26.565228542000114s` versus `19.360704333004833s`). Do not run a 600-step
scanline row unless another patch changes the compact row structure or reducer.

The current deterministic candidate is zero-row-pruned `tile_pair`. The kernels
now keep rows invalid when the accumulated row gradient is exactly zero. This
cuts useless reducer work and changes the long-horizon read: the 512px seed-2
600-step replay
`mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_zero_prune_600steps.json`
has pruned tile-pair heldout PSNR `13.866263389587402` versus keyed
`13.75709342956543`, with train loops `146.85938100000203s` versus
`222.79236249999667s`. The repeatability artifact
`mcam512_s2_t320_view_shuffled_cycle_tilepair_zero_prune_repeatability_600steps.json`
is exact, final state max delta `0.0`. The paired fixed-600 seeds 0/1/2 rows
reach STAR heldout PSNR `13.728736877441406`, `13.71193790435791`, and
`13.866263389587402` against direct splats `8.247447967529297`,
`8.256773948669434`, and `8.26813793182373`. Follow-up repeatability artifacts
for seeds 0 and 1 are exact too:
`mcam512_s0_t320_view_shuffled_cycle_tilepair_zero_prune_repeatability_600steps.json`
and `mcam512_s1_t320_view_shuffled_cycle_tilepair_zero_prune_repeatability_600steps.json`
both have final state max delta `0.0`. This clears the existing V-JEPA F32
heldout reference on all three seeds with exact same-process repeatability, but
still needs a freshly matched V-JEPA/full-resolution baseline before promotion.
The immediate baseline refresh found a stronger local V-JEPA/TokenGS-style row:
run `24absic1` in
`outputs/run_logs/20260508_144522_fast512_tokenbudget_train.log` is a multires
64/128/256/512 schedule, not a pure 512px run, and logs heldout PSNR `13.9870`.
So the pruned tile-pair STAR matrix beats direct splats and the older `13.6248`
F32 reference, but it does not beat the stronger multires V-JEPA row yet.

The same refresh also produced a manual pure-512 V-JEPA F32 baseline at
`outputs/multicam_relative_pose/full_relpose_features_F32_512_v6refined_goodset_train0006_0014_holdout0005_manual250_nomedia/`.
It ran the pure-512 v6refined goodset config for 250 steps with W&B/media
logging disabled, saved `checkpoint_final.pt`, and writes the condensed result
to `manual_probe_summary.json`: loop `4382.390926374996` s, mean train PSNR
`14.9983`, heldout PSNR `13.5727`. This makes the boundary cleaner: STAR now
beats direct splats, the older `13.6248` row, and this pure-512 manual row, but
not the stronger multires `13.9870` row.

The fixed-tube UVT scale probe now gives the rasterizer answer separately from
the quality answer. With `7168` tubes and 16 frames, forward UVT tile-tube pairs
stay nearly flat at `451838`, `516990`, `531638`, and `539762` for
128/256/512/1024 targets while output-buffer memory grows `64x`; artifacts are
`uvt_forward_speed_probe_*_16f_7168_s0125_t20_tilet1_cap128_metalonly_scale.json`.
The matching zero-pruned tile-pair backward rows stay roughly flat too:
`207324`, `211374`, and `213642` rows for 256/512/1024, with allocated tile-pair
slots growing `2097152 -> 8388608 -> 33554432`. Read: STAR-UVT now has real
sparse/sublinear rasterizer evidence at fixed tube count, but not a solved
full-training speed or stronger-baseline PSNR win.

A 512px direct-atomic fixed-tube breakdown now pins the speed target for the
deterministic rasterizer work. The artifact
`uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_directatomic_scale.json`
uses the same 7168-tube, 16-frame, `tile_t=1`, cap-128 setup and emits only
`7168` direct per-tube gradient rows, with sample+reduce median
`35.70937499171123` ms. The matching deterministic zero-pruned tile-pair row
emits `211374` rows from `8388608` allocated slots and takes
`129.20112499705283` ms. So direct atomic is about `3.6x` faster on this probe,
but it remains nondeterministic at 512px training scale. The next speed work is
a deterministic fused tile-pair/per-tube backward, not another broad
schedule-only sweep.

The first deterministic direct-output prototype is negative. `direct_serial`
is now wired as a diagnostic sample-emission mode: one thread per tube loops
its tile support in fixed order and writes final per-tube gradients directly.
The parity smoke
`uvt_direct_serial_backward_parity_smoke_16_2f_16t.json` matches direct atomic
within small float-order tolerance, but the 512px fixed-tube timing row
`uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_directserial_scale_1it.json`
takes `327.0019999908982` ms sample+reduce median. That is slower than
zero-pruned tile-pair (`129.20112499705283` ms) and direct atomic
(`35.70937499171123` ms), so serial per-tube scanning is not the missing
deterministic fast path.

The first deterministic target-support skip is also negative.
`tile_pair_target_bounds` keeps the deterministic tile-slot output format but
skips pixels outside the target tube's analytic support. The tiny parity smoke
`uvt_tile_pair_target_bounds_parity_smoke_16_2f_16t_tilet1.json` matches plain
`tile_pair` exactly after reduction, but the 512px fixed-tube timing row
`uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_target_bounds_scale.json`
takes `132.46404100209475` ms sample+reduce median, slightly slower than
zero-pruned tile-pair's `129.20112499705283` ms. So support-bounds skipping is
not the missing speed path either.

The suffix-composite tile-pair variant is the first deterministic speed
improvement, but not a quality improvement. `tile_pair_suffix` keeps the same
deterministic tile-slot row contract and computes the target gradient from a
prefix transmittance plus suffix color composite. It is exact-repeatable at the
20-step smoke and improves the 512px fixed-tube backward median to
`112.1208749973448` ms, versus zero-pruned tile-pair's
`129.20112499705283` ms. The full 512px seed-2 600-step row
`mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_suffix_staronly`
trains faster (`88.92246554100711` s), but heldout PSNR is only
`13.808026313781738`, below zero-pruned tile-pair's
`13.866263389587402` and below the `13.9870` multires V-JEPA row. Treat it as
a speed candidate, not a quality promotion.

A same-wall-clock suffix retry uses the speed margin for 750 steps:
`mcam512_s2_t320_view_shuffled_cycle_timebudget750_besttrain_tilepair_suffix_staronly`
takes `128.04329337499803` s and reaches heldout PSNR `13.85195541381836`.
That closes most of the 600-step suffix gap but still misses the zero-pruned
seed-2 row (`13.866263389587402`) while running longer than its
`113.04000200000155` s loop. Do not promote suffix as the quality branch without
a separate quality change.

The key-sort segmented reducer is a fixed-probe win but a full-train rejection.
`key_sort_segmented_metal` reuses the stable sample key sort, then reduces each
tube's contiguous sorted segment instead of scanning the whole row buffer. The
512px suffix backward probe improves sample+reduce median to
`98.4361659939168` ms, and the 512px step-0 gradient compare matches
`key_sort_scan_metal` with max abs `0.0`. However the matched 600-step suffix
training row takes `171.39199266598735` s at the same heldout PSNR
`13.808026313781738`, much slower than suffix scan's `88.92246554100711` s.
Treat this as a diagnostic reducer, not the training-speed fix.

The direct-reduced suffix path is exact but not a trainer-speed win.
`tile_pair_suffix_reduced_backward` and
`--uvt-sample-emission-mode tile_pair_suffix_reduced` keep the suffix sample
math and reduce per tube in Metal tile-major/slot order, avoiding the explicit
Python/MPS key sort and gather. Tiny and 512px parity checks match suffix plus
`key_sort_scan_metal` exactly (`0.0` max abs across all gradient tensors). The
512px fixed backward row
`uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_suffix_reduced_scale.json`
takes `99.05412500665989` ms sample+reduce median, but the matched 600-step
training row
`mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_suffix_reduced_staronly`
takes `118.60049016600533` s at the same heldout PSNR
`13.808026313781738`, slower than suffix scan's `88.92246554100711` s and
slower/lower-quality than zero-pruned tile-pair. Treat this as a useful
negative result: avoiding the explicit sort/gather is not enough when the
replacement is a per-tube support scan.

The plain zero-pruned direct-reduced path is also exact, but still not promoted.
`tile_pair_reduced_backward` and
`--uvt-sample-emission-mode tile_pair_reduced` match `tile_pair +
key_sort_scan_metal` exactly on the 512px parity artifact
`uvt_tile_pair_reduced_backward_parity_512_16f_7168t_tilet1_cap128.json`
(`0.0` max abs, `63051` valid reference rows). Its isolated fixed rows are the
best deterministic quality-preserving timings so far: cap-128
`40.33462500956375` ms sample+reduce and cap-256 `46.68695799773559` ms. The
full 600-step cap-256 STAR-only gate
`mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_reduced_staronly`
recovers zero-pruned quality exactly at train/heldout PSNR
`15.34240198135376` / `13.866263389587402`, but the train loop is
`132.13478325000324` s. Treat it as evidence that the sort/gather can be removed
without changing math, not as a solved training rasterizer; the per-tube support
scan is still too expensive once training expands support.

The direct fixed-point atomic path is fast and exact-repeatable, but not
trainable enough to promote. `direct_fixedpoint_backward` accumulates direct
per-tube gradients into integer atomic buffers before converting back to float.
The fixed 512px row is much closer to direct atomic than the deterministic
tile-pair family: default scale (`1e6`, from the pre-metadata artifact command)
takes `47.18237501219846` ms sample+reduce median, and the `scale1e4` retry
takes `42.020083012175746` ms. The 20-step repeatability artifact is exact
(`final_state_0_1.max_abs=0.0`). But default scale goes nonfinite at step `160`
and selects only step `100` at heldout PSNR `7.480576515197754`; scale `1e4`
reaches heldout PSNR `10.693564414978027` at 200 steps, then the 600-step retry
goes nonfinite at step `350` and selects step `300` at heldout PSNR
`10.548453330993652`. Treat fixed-point atomics as a useful speed diagnostic,
not as the current UVT training rasterizer.

A tile-pair fixed-point retry is rejected for the same reason. The
`tile_pair_fixedpoint_backward` path quantizes only after deterministic
tile-slot accumulation and is exact-repeatable in the 20-step gate. Its fixed
512px rows are also plausible speed diagnostics: cap-128 takes
`64.06379198597278` ms sample+reduce, and a cap-256 warm rerun takes
`43.17741599516012` ms. But the 200-step training gates fail hard. Default
scale (`1e6`) goes nonfinite at step `100` and selects only train/heldout PSNR
`7.657606363296509` / `7.620186805725098`; `STAR_UVT_FIXEDPOINT_SCALE=10000`
goes nonfinite at step `180`, selects step `100`, and reaches only
`7.5389885902404785` / `7.422435760498047`. Do not promote tile-pair
fixed-point without a different stability mechanism.

The tile-pair float-atomic retry is also only a diagnostic. `tile_pair_atomic`
keeps the zero-pruned per-slot accumulation, then atomically adds each nonzero
tile-slot sum directly into global tube gradients. It avoids fixed-point
quantization and matches zero-pruned tile-pair closely at 512px
(`9.1552734375e-05` max abs on the largest gradient tensor), but its fixed
timing is only `93.08808401692659` ms sample+reduce. The 20-step repeatability
gate is not exact either: final state max abs delta is
`8.440017700195312e-05`, only slightly better than direct atomic's
`0.00010570883750915527`. Do not spend a 600-step quality run on this branch
under the current deterministic-promotion bar.

A split fixed-point retry is also rejected. `direct_split_fixedpoint_backward`
adds coarse and fine integer accumulators (`100` and `1000000` scales) to test
whether the single-scale path was failing from overflow/quantization. It is
exact-repeatable and the 512px fixed row lands at `53.01575000339653` ms
sample+reduce median, but the 200-step training gate goes nonfinite at step
`120` and selects step `100` at heldout PSNR `7.471946716308594`. This is worse
than the `scale1e4` single fixed-point retry, so do not pursue the fixed-point
family without a different stability mechanism.

The tile-pair-parallel reducer is also a speed rejection. `tile_pair_parallel`
keeps the compact tile-slot row contract, but reduces the tile's pixels in a
fixed threadgroup tree instead of one serial row thread. The tiny parity artifact
is exact-repeatable and matches plain `tile_pair` after reduction within max abs
`1.52587890625e-05`, but the 512px fixed row
`uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_parallel_scale.json`
takes `141.02554200508166` ms sample+reduce median. That is slower than
zero-pruned tile-pair (`129.20112499705283` ms) and suffix
(`112.1208749973448` ms), so no 600-step quality run was launched.

The tile-pair-grouped reducer is the first deterministic fixed-backward speed
branch below suffix, but it is not yet the quality branch. `tile_pair_grouped`
sorts once per tile and loops the tile-local slots inside that group. The 512px
fixed row
`uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_grouped_scale.json`
takes `90.38029100338463` ms sample+reduce median with `211374` compact rows
and unstable fraction `0.0`, beating suffix (`112.1208749973448` ms) and
zero-pruned tile-pair (`129.20112499705283` ms). The 100-step keyed-vs-grouped
trajectory is nearly identical and grouped is exact-repeatable, but the 600-step
STAR-only row
`mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_grouped_staronly`
lands at heldout PSNR `13.74386978149414` in `100.48031524999533` s. That is
faster than zero-pruned seed 2 but below zero-pruned quality
(`13.866263389587402`), below suffix quality (`13.808026313781738`), and slower
than suffix's `88.92246554100711` s loop. Treat it as the current best
deterministic speed probe, not a promoted training default.

Grouped's quality miss is also not explained by a gross local VJP mismatch. The
512px step-0 keyed-vs-grouped gradient gate
`mcam512_s2_t320_view_shuffled_cycle_gradient_step0_withkeys_vs_tilepair_grouped.json`
reports loss delta `0.0` and max gradient delta `6.51925802230835e-09`. After
an internal 600-step grouped pretrain, the artifact
`mcam512_s2_t320_view_shuffled_cycle_gradient_step600_grouped_pretrain_withkeys_vs_tilepair_grouped.json`
reports loss delta `0.0` and max gradient delta `1.3746321201324463e-06`, with
deterministic autograd digests in both modes. So grouped should be treated like
suffix: its lower 600-step heldout PSNR is trajectory/optimizer sensitivity, not
an obvious local backward bug.

`tile_pair_sharedsort` is implemented as a quality-preserving but small speed
variant. It sorts once per tile, then accumulates each tile slot serially over
pixels like zero-pruned `tile_pair`. The tiny parity artifact
`uvt_tile_pair_sharedsort_backward_parity_smoke_16_2f_16t_tilet1.json` matches
plain tile-pair ids and keys exactly, with reduced-gradient max abs
`3.814697265625e-06`. Current-build fixed 512px rows show only a modest speed
gain: cap-128 sharedsort is `52.655457984656096` ms sample+reduce versus a
current cap-128 tile-pair rerun at `56.643062496732455` ms; cap-256 sharedsort is
`64.27791700116359` ms versus a current cap-256 tile-pair rerun at
`69.4723329943372` ms. The 100-step keyed-vs-sharedsort trajectory stays close
(`-0.00014972686767578125` heldout delta), but the 600-step STAR-only row
`mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_sharedsort_staronly`
lands at train/heldout PSNR `15.138561725616455` / `13.569375991821289` in
`163.4644198330061` s. So sharedsort is useful evidence, not the fast
sublinear-quality UVT backward solution.

The same-step local overfit pull now has an evaluated 200-step direct-splat row.
The paired artifact
`mcam512_s2_t320_view_shuffled_cycle_fixed200_besttrain_tilepair_zero_prune_paired`
and summary
`mcam512_same_step_overfit_summary_2026_05_12.json` show direct splats at 200
steps reach train/heldout PSNR `6.988864898681641` / `6.835783004760742` in
`6.4328010419994825` s, while zero-pruned STAR reaches
`12.694841861724854` / `12.390332221984863` in `37.24962037500518` s and
renders faster (`0.13297945899830665` s versus `0.6536745410121512` s). At 600
steps, direct splats remain far behind (`9.27243709564209` /
`8.26813793182373`), so this direct-splat baseline is no longer the hard local
comparator. The hard questions are keeping zero-pruned quality while recovering
suffix/grouped speed, and beating the stronger multires V-JEPA row.

Suffix's quality miss is not explained by a gross local VJP mismatch. The 512px
step-0 keyed-vs-suffix gradient gate reports max model-gradient delta
`6.05359673500061e-09`; after 600 suffix-pretrain steps the max delta is
`2.3096799850463867e-07`. This points to trajectory/optimizer sensitivity, not
an obvious wrong-gradient bug.

The first larger-resolution quality probe rejects naive 1024px escalation. The
STAR-only artifact
`mcam1024_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_zero_prune_staronly`
keeps the 512px zero-pruned recipe and reaches heldout PSNR
`13.766105651855469` in `209.39192004200595` s, with render-only eval
`0.8590989580116002` s and max trained pair ratio `5.134140396727249`. This is
below the 512px seed-2 row `13.866263389587402` and below the stronger multires
V-JEPA row `13.9870`.

A stricter support-control repeat is also a quality rejection. The artifact
`mcam1024_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_zero_prune_staronly_tileload003`
raises tile-load weight to `0.003` at the same target `7000`. It reduces max
pair ratio `5.134140396727249 -> 4.722782922009996` and loop time
`209.39192004200595 -> 192.36781579200033` s, but heldout PSNR drops to
`13.61436939239502`. Treat stronger tile-load pressure as a speed/compactness
knob, not the next quality path.

The matching longer-training check is also a rejection. The 512px seed-2
artifact
`mcam512_s2_t320_view_shuffled_cycle_fixed1000_besttrain_tilepair_zero_prune_staronly`
extends the zero-pruned recipe to `1000` steps. The non-heldout selector picks
final step `1000` with train PSNR `15.832406044006348`, heldout PSNR
`13.866610527038574`, and loop `336.69192270799977` s; the heldout-best
checkpoint is step `700` at `13.875921249389648`. Since the 600-step paired row
was already `13.866263389587402` heldout in `113.04000200000155` s, extra steps
do not close the `13.9870` multires V-JEPA gap.

The first schedule variant after that is also rejected. The new
`epoch_view_shuffled_cycle` option shuffles train views once per full temporal
epoch instead of once per window. Its smoke passed, but the 512px seed-2
artifact
`mcam512_s2_t320_epoch_view_shuffled_cycle_fixed600_besttrain_tilepair_zero_prune_staronly`
selects step `600` at heldout PSNR `13.707905769348145`; heldout-best is step
`400` at `13.862957954406738`. That misses the current view-shuffled seed-2
zero-pruned row (`13.866263389587402`), so do not expand this branch to seeds
0/1 unless another schedule change explains the miss.

Retrying plain `cycle` under the same current zero-pruned tile-pair path is
also negative. The artifact
`mcam512_s2_t320_cycle_fixed600_besttrain_tilepair_zero_prune_staronly` stays
finite, but selected/heldout-best is only step `600` at heldout PSNR
`13.701606750488281` in `132.23203912499594` s. So the fixed rasterizer does
not rescue the old cycle schedule.

The first static/dynamic tube initialization probe is also rejected as a simple
quality fix. `multicam_heldout_compare.py` now exposes opt-in
`--uvt-static-tube-fraction` and `--uvt-static-init-lambda-t`; the split is a
capacity bias, not a semantic classifier. The smoke
`multicam_heldout_compare_static_dynamic_init_smoke_16_2f_1s` passed. The 512px
seed-2 artifact
`mcam512_s2_t320_view_shuffled_cycle_static025_lamt002_fixed600_besttrain_tilepair_zero_prune_staronly`
uses `80` static and `240` dynamic tubes, selects step `600`, and reaches
heldout PSNR `13.644302368164062` in `136.58938849999686` s, with render-only
eval `0.13702220900449902` s, max pair ratio `3.2617686604907052`, max tile
`93`, zero overflow, and zero unstable tiles. This misses the current
deterministic seed-2 row (`13.866263389587402`) and the `13.9870` multires
V-JEPA row, so do not expand this branch without changing more than
initialization.

Adding a static-only velocity penalty does not rescue that branch. The new
`--uvt-static-velocity-reg` flag regularizes only the static slice. The smoke
`multicam_heldout_compare_static_dynamic_velreg01_smoke_16_2f_1s` passed. The
512px seed-2 artifact
`mcam512_s2_t320_view_shuffled_cycle_static025_lamt002_velreg01_fixed600_besttrain_tilepair_zero_prune_staronly`
keeps `80` static and `240` dynamic tubes, adds static velocity weight `0.1`,
and selects step `600` at heldout PSNR `13.454254150390625`; heldout-best is
step `500` at `13.526884078979492`. Loop time rises to `160.88216154198744` s
and render-only eval to `0.2113907500024652` s. Metal remains clean, but the
quality is worse than the init-only split, so do not expand this branch.
