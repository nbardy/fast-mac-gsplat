# STAR-UVT Progress

Last updated: 2026-05-11

## Checklist

- [x] Gate 0: projected UVT Metal forward renderer with fixed tile buffers.
- [x] Gate 0: CPU brute-force reference and Metal parity smoke.
- [x] Gate 0: all planned tiny scenes represented in the smoke harness.
- [x] Gate 1: clean projected `ScreenTimeTube` trainer harness scaffold.
- [x] Gate 1: CPU smoke proves loss decreases on a deterministic tiny scene.
- [x] Gate 1: real video frame loader smoke on a repo fixture.
- [x] Gate 2a: orthographic `WorldTube` to `ScreenTimeTube` projection smoke.
- [x] Gate 2b: pinhole-camera `WorldTube` to `ScreenTimeTube` projection smoke.
- [x] Gate 2c: Dynaworld `CameraSpec` adapter smoke.
- [x] Gate 3a: dense PyTorch backward gradient probe for projected tubes.
- [x] Gate 3b: hybrid Metal-forward/dense-backward autograd smoke.
- [x] Gate 3c: simplified single-tube Metal backward parity probe.
- [x] Gate 3d: stable sorted-tile Metal backward parity probe.
- [x] Gate 3e: unstable fallback Metal backward parity probe.
- [x] Gate 3f: Metal tile-backward autograd bridge with MPS reduction.
- [x] Gate 3g: bounded backward performance smoke.
- [x] Gate 3h: bounded large-scene backward performance benchmark.
- [x] Gate 4a: side-by-side UVT-vs-sliced-per-frame renderer pair benchmark.
- [x] Gate 4b: side-by-side tiny training benchmark against per-frame splat training.
- [x] Gate 4c: real-video fixture fit comparison with contact-sheet proof.
- [x] Gate 5: promotion audit completed; do not integrate yet.
- [x] Gate 6a: multicam heldout comparison harness against V-JEPA F32 split and direct splats.
- [x] Gate 6b: tiny CPU smoke for report and media output plumbing.
- [x] Gate 6c: small MPS/Metal same-time pilot at 64px and 4 frames.
- [x] Gate 6d: 128px/16-frame same-time pilot.
- [x] Gate 6e: fixed-step single-video overfit comparison.
- [x] Gate 6f: single-video overfit LR/capacity/init ablation.
- [x] Gate 6g: 64px transfer and forward-speed probe.
- [x] Gate 6h: single-video Metal tile-backward overfit path.
- [x] Gate 6i: 128px single-video transfer probe.
- [x] Gate 6j: 128px multicam worldtube init ablation.
- [x] Gate 6k: DeepView fisheye versus pinhole projection-magnitude audit.
- [x] Gate 6l: 128px multicam view-sequence loss breakthrough.
- [x] Gate 6m: 256px/16-frame bounded 60-second view-sequence pilot.
- [x] Gate 6n: 256px/16-frame temporal-window throughput pilot.
- [x] Gate 6o: synchronized render-only timing fields for future multicam reports.
- [x] Gate 6p: initialized-model multicam render timing probe with STAR projection/render split.
- [x] Gate 6q: trained 256px temporal-window rerun with render-only timing fields.
- [x] Gate 6r: trained STAR metal tile stats rerun identified pair-ratio and overflow blow-up.
- [x] Gate 6s: opt-in precision-floor support experiment.
- [x] Gate 6t: opt-in velocity/depth-velocity regularization experiment.
- [x] Gate 6w: current 256px multicam train-step timing breakdown.
- [x] Gate 6x: compact STAR backward sample outputs before MPS reduction.
- [x] Gate 6y: compact-backward 60-second quality rerun.
- [x] Gate 6z: closed-form pinhole projection and 60-second quality rerun.
- [x] Gate 6aa: bundled compact reduction and LR stability bracket.
- [x] Gate 6ab: opt-in dataset-lens STAR projection diagnostic.
- [x] Gate 6ac: lens-aware direct-splat baseline and first V-JEPA heldout crossing.
- [x] Gate 6ad: deterministic grid init repeatability check.
- [x] Gate 6ae: all-train grid init repeatability escalation.
- [x] Gate 6af: deterministic train schedule probe.
- [x] Gate 6ag: all-train grid LR `0.015` probe.
- [x] Gate 6ah: all-train grid tile-load target `5000` probe.
- [x] Gate 6ai: time-distributed all-frames initialization smoke and seed-1 curve.
- [x] Gate 6aj: seed-0 and seed-2 repeat for all-frames initialization.
- [x] Gate 6ak: document current speed-versus-quality decision.
- [x] Gate 6al: seed-0 shorter-budget curve around the 20-second peak.
- [x] Gate 6am: opt-in STAR checkpoint-curve reporting smoke.
- [x] Gate 6an: 30-second seed-0 checkpoint curve confirms the mid-run peak.
- [x] Gate 6ao: STAR-only `--skip-splats` checkpoint diagnostic smoke.
- [x] Gate 6ap: 60-second seed-1 checkpoint curve confirms long-run decay.
- [x] Gate 6aq: 40-second seed-2 checkpoint curve shows seed-dependent peak.
- [x] Gate 6ar: opt-in STAR LR-decay schedule and seed-1 schedule bracket.
- [x] Gate 6as: seed-2 later-peak LR-decay schedule check.
- [x] Gate 6at: paired seed-2 LR-decay row against direct dynamic splats.
- [x] Gate 6au: opt-in selected-checkpoint report and media smoke.
- [x] Gate 6av: paired seed-1 selected-checkpoint MPS artifact.
- [x] Gate 6aw: paired seed-2 selected-checkpoint MPS artifact.
- [x] Gate 6ax: train-curve plateau selector smoke and seed-2 MPS artifact.
- [x] Gate 6ay: train-camera balance selector diagnostic and rejection.
- [x] Gate 6az: true train-camera dev split selector diagnostic and rejection.
- [x] Gate 6ba: seed-0 LR-decay schedule repeat.
- [x] Gate 6bb: train-plateau patience-2 selector diagnostic.
- [x] Gate 6bc: train-gain-drop selector diagnostic.
- [x] Gate 6bd: paired train-gain-drop report against direct dynamic splats.
- [x] Gate 6be: second paired train-gain-drop report on weaker seed 1.
- [x] Gate 6bf: 512px same-budget gain-drop scale probe and rejection.
- [x] Gate 6bg: 512px window-1 scale-strategy probe and selector warning.
- [x] Gate 6bh: 512px stricter gain-drop selector diagnostic.
- [x] Gate 6bi: 512px seed-1 window-1 repeat and LR-decay bracket.
- [x] Gate 6bj: 512px window-1 tube-capacity bracket.
- [x] Gate 6bk: 512px seed-0 256-vs-320 capacity and decay-400 check.
- [x] Gate 6bl: balanced train-plateau selector diagnostic and rejection.
- [x] Gate 6bm: 128px same-step 200-step single-video overfit comparison.
- [x] Gate 6bn: 512px seed-0 320-tube window-2 policy check.
- [x] Gate 6bo: 512px seed-0 320-tube hard LR-drop policy check.
- [x] Gate 6bp: 128px same-step temporal-support bracket.
- [x] Gate 6bq: 128px same-step temporal-support LR bracket.
- [x] Gate 6br: 512px seed-0 320-tube temporal-floor support check.
- [x] Gate 6bs: train-view gap-collapse selector diagnostic and rejection.
- [x] Gate 6bt: train-camera temporal dev-frame selector diagnostic and rejection.
- [x] Gate 6bu: 512px seed-0 320-tube free init-lambda support check.
- [x] Gate 6bv: 512px bounded sequence-consistency support check and rejection.
- [x] Gate 6bw: 512px multiscale auxiliary loss bracket.
- [x] Gate 6bx: 512px deterministic crop auxiliary loss bracket.
- [x] Gate 6by: 512px deterministic cycle train-schedule revisit and rejection as seed-robust default.
- [x] Gate 6bz: 512px shuffled-cycle schedule and non-finite guard follow-up.
- [x] Gate 6ca: 512px reshuffled-cycle robust-floor schedule.
- [x] Gate 6cb: 512px phase-rotated-cycle rejection.
- [x] Gate 6cc: 512px view-shuffled-cycle schedule and best-train reporting selector.
- [x] Gate 6cd: 128px single-video 400-step overfit headroom check.
- [x] Gate 6ce: 128px temporal-quarter piecewise init rejection.
- [x] Gate 6cf: 128px temporal split/refine operator rejection.
- [x] Gate 6cg: 128px block-match and gated block-match velocity-init rejection.
- [x] Gate 6ch: 128px render-preserving duplicate split rejection.
- [x] Gate 6ci: 128px 800-step overfit headroom rejection.
- [x] Gate 6cj: 128px duplicate-split Metal tile-fallback diagnosis.
- [x] Gate 6ck: 128px tile-shape and tile-capacity overfit check.
- [x] Gate 6cl: 128px scheduled split under current final count rejection.
- [x] Gate 6cm: 128px `tile_t=1` LR bracket and default-speed rejection.
- [x] Gate 6cn: 128px `tile_t=1` cap-128 three-seed robustness check.
- [x] Gate 6co: 128px synchronized repeat render timing for current recipe.
- [x] Gate 6cp: 128px paired synchronized render timing against direct splats.
- [x] Gate 6cq: 128px equal-step tube-count speed/quality retune.
- [x] Gate 6cr: 128px 1728-tube equal-step three-seed robustness check.
- [x] Gate 6cs: 128px cap-256 quality-mode synchronized repeat timing.
- [x] Gate 6ct: 128px cap-256 quality-mode three-seed robustness check.
- [x] Gate 6cu: 256px single-video cap/tube scale gate.
- [x] Gate 6cv: 256px single-video 400-step three-seed robustness check.
- [x] Gate 6cw: 256px per-frame video-init baseline feasibility probe.
- [ ] Gate 6u: 256px/16-frame longer-budget comparison against the 18-minute V-JEPA F32 row.
- [ ] Gate 6v: camera-model parity decision for DeepView fisheye versus current pinhole harness.

## Current State

The renderer can answer the Gate 0 question for already-projected UVT tubes.
The trainer harness can fit projected screen-time tube parameters against tiny
synthetic and fixture-video targets with a dense differentiable PyTorch
renderer, then check learned tensors with the Metal forward path. A prototype
Metal tile-backward autograd bridge exists for small MPS smokes.

The current trainer is a research scaffold, not a production path. It exists to
separate representation and optimization questions from the future production
renderer question. The Metal tile-backward path is now good enough for local
single-video overfit iteration, but it still needs hardening before promotion.
The multicam comparison harness is the first path that compares STAR-UVT to the
current V-JEPA F32 train/heldout split and to direct dynamic splats under the
same local wall-clock budget.

The latest multicam read is that 256px STAR-UVT has crossed both the
lens-aware direct dynamic splat baseline and the 256px V-JEPA F32 reference
under the current gain-drop reporting selector, but 512px is not a solved
seed-robust recipe yet. A naive 512px/window-4 bump was step-starved and
rejected; a 512px/window-1 run recovered step throughput and quality on seed 2,
while the seed-1 repeat still beats direct splats but does not clear the V-JEPA
F32 reference. A 512px tube-capacity bracket shows that 320 tubes improves the
weaker seed-1 result and clears V-JEPA on final/selected rows, but seed 2 still
prefers the 256-tube recipe and seed 0 exposes a selector/schedule failure:
320 tubes has a better heldout-best peak than 256 tubes, but the current
non-heldout selector misses that peak. The current gain-drop selector is still
conservative at 512px. The compact
backward-sample patch cut local STAR train-step time, the closed-form 2x2
pinhole projection removed projection as a major phase, and the bundled compact
reducer cut MPS gradient reduction again. The current stable cap-256 60-second
artifact uses LR `0.01`: STAR heldout PSNR
`13.20147705078125` versus direct splats `10.722965240478516`, with STAR
render-only eval `0.046138084086123854s` versus splats
`0.29644233302678913s`. An opt-in dataset-lens STAR diagnostic, using the
DeepView `opencv_fisheye` camera model instead of the legacy pinhole
approximation, lifts the 60-second STAR heldout PSNR to
`13.496740341186523` and remains fast: STAR render-only eval
`0.0413991259993054s` versus direct splats `0.4126907510217279s` in the paired
run. This is not a full apples-to-apples V-JEPA replacement because the V-JEPA
reference config and direct-splat baseline still use the legacy pinhole path,
but it makes camera-model parity a first-class quality lever. The follow-up
run that also gives direct splats the same `dataset_lens` camera contract is
the current clean comparison row: STAR heldout PSNR
`13.632997512817383`, direct splats `11.188531875610352`, V-JEPA F32 reference
`13.6248`; STAR render-only eval `0.04134708392666653s` versus direct splats
`0.4395427079871297s`. This is the first measured local STAR crossing over the
V-JEPA heldout row, but only by about `0.0082` dB, so treat it as a gate to
repeat/scale rather than a production promotion. A seed-1 repeat did not
reproduce the crossing: STAR heldout PSNR fell to `12.9697904586792` while
direct splats reached `11.243672370910645`. A deterministic grid-init seed-1
repeat improved STAR to heldout PSNR `13.179410934448242` versus direct splats
`11.206615447998047`, and initializing from all train views with grid sampling
improved seed 1 again to heldout PSNR `13.52819538116455` versus direct splats
`11.074682235717773`, but still did not recover the V-JEPA crossing. A
deterministic cycle train schedule was negative at the 20-second gate, reaching
heldout PSNR `13.28015422821045` versus `13.527872085571289` for the same
all-train grid init with the random train schedule. LR `0.02` and `0.03` are
too hot at the full 60-second budget after the speedups, and LR `0.015` is
also negative for the all-train grid setting at the 20-second gate. Tightening
tile-load target from `7000` to `5000` improves compactness and speed but loses
heldout quality.

The best current quality lever is time-distributed initialization:
`--uvt-init-frames all` splits the initial tubes across train-view/frame groups
and initializes each tube's `t0` to the centered source-frame time. This fixes
the older mismatch where multi-view initialization sampled frame-0 colors while
placing every tube at sequence-center time. At 20 seconds, this recipe crosses
the V-JEPA F32 heldout reference `13.6248` on all three tested seeds: seed 0
reached `13.769630432128906`, seed 1 reached `13.768306732177734`, and seed 2
reached `13.764396667480469`. These rows keep STAR render-only eval around
`0.04s` versus direct splats at `0.292-0.462s`. The 30-second rows are mixed:
seed 1 reached `13.726262092590332`, but seed 0 reached
`13.600011825561523`. A seed-0 shorter-budget bracket reached `12.681236267089844`
at 10 seconds and `13.669918060302734` at 15 seconds, so the current sweet spot
is not simply the shortest run; it ramps into the 15-20 second region and then
can decay. The 60-second seed-1 run fell to `13.564573287963867`, so longer
training is not automatically better and can hurt heldout-camera PSNR. The
current blocker is early-stop/schedule behavior and more repeat evidence, not
forward rasterizer speed.

The benchmark now has opt-in STAR checkpoint-curve reporting via
`--uvt-checkpoint-every-steps N`. It stores small worldtube state snapshots
during STAR training, evaluates them after training, and reports
`checkpoint_curve.rows` plus `checkpoint_curve.best_by_heldout_psnr`.
`--skip-splats` is also wired for STAR-only checkpoint diagnostics when the
direct-splat baseline is already known. The CPU smokes
`multicam_heldout_compare_checkpoint_curve_smoke_16_2f_1s` and
`multicam_heldout_compare_skip_splats_smoke_16_2f_1s` passed, with the
skip-splats report writing `free_dynamic_splats: null`. This is a diagnostic
selector for the current research lane; do not treat heldout-selected numbers
as a final unbiased test metric. A real 30-second seed-0 checkpoint run with
checkpoints every 50 steps confirms the shape: best heldout was step `300` at
elapsed `18.632994499988854s`, with heldout PSNR `13.730653762817383`; the
final step `489` fell to `13.518027305603027`. A STAR-only 60-second seed-1
diagnostic with checkpoints every 100 steps shows the same peak: best heldout
was step `300` at elapsed `17.160626166965812s`, with heldout PSNR
`13.75400447845459`; the final step `1019` fell to `13.354101181030273`.
A STAR-only 40-second seed-2 diagnostic did not peak at step 300; it peaked at
step `500`, elapsed `32.050383166992106s`, with heldout PSNR
`13.988276481628418`, then fell to final step `629` heldout PSNR
`13.631431579589844`. That is direct evidence for a mid-run selector or
schedule gate, but the gate should track validation shape rather than hard-code
a fixed 20-second or 300-step cutoff. An opt-in LR decay schedule is now wired
via `--uvt-lr-decay-step` and `--uvt-lr-decay-factor`. On seed 1, decaying
from LR `0.01` after step `300` improves the final 60-second checkpoint versus
the no-decay final. Factor `0.2` finished at heldout PSNR
`13.643348693847656`, and factor `0.05` finished at heldout PSNR
`13.692363739013672` with STAR render-only eval `0.04503662494244054s`. This
is a partial schedule win: it preserves a V-JEPA-crossing final checkpoint on
seed 1, but it still does not match the best selected checkpoint quality. On
seed 2, decaying after the later step-500 peak with factor `0.05` finished the
60-second run at heldout PSNR `13.81359577178955`, with best checkpoint
`13.909360885620117` at step `500`, zero overflow, and zero unstable tiles.
That again supports schedule work, but still leaves validation selection ahead
of a hard-coded decay step. The paired seed-2 schedule row against lens-aware
direct dynamic splats is now the clean comparison artifact: STAR final heldout
PSNR `13.84060287475586` versus direct splats `11.156550407409668`, STAR
render-only eval `0.043013749993406236s` versus direct splats
`0.3524511669529602s`, and STAR best checkpoint `13.873014450073242` at step
`800`. This is a same-budget local STAR win over direct splats and V-JEPA, but
the selected-checkpoint diagnostic still shows better quality is available.
`--uvt-select-checkpoint best_heldout` is now available as an explicit
diagnostic artifact path: it loads the best checkpoint from
`checkpoint_curve.best_by_heldout_psnr`, writes a separate `star_uvt_selected`
report section, and saves selected train/heldout media. The CPU smoke
`multicam_heldout_compare_selected_checkpoint_smoke_16_2f_1s` passed and wrote
the selected media. Because this selector uses the heldout camera, it is
deliberately labeled with `uses_heldout_for_selection: true` and should not be
reported as an unbiased test metric.

The first real MPS selected-checkpoint artifact is the paired seed-1
`300 -> 0.05x` LR-decay run with direct dynamic splats. STAR final heldout PSNR
was `13.758882522583008` versus direct splats `11.15761947631836`; STAR
selected heldout PSNR was `13.818532943725586` at step `300`. Final STAR
render-only eval was `0.04524075100198388s`, selected STAR render-only eval
was `0.03815650095930323s`, and direct splats render-only eval was
`0.3847669999813661s`. This confirms the selected-artifact path on the real
MPS recipe, while also showing some MPS run-to-run variation versus the earlier
STAR-only seed-1 schedule bracket.

The second real MPS selected-checkpoint artifact is the paired seed-2
`500 -> 0.05x` LR-decay run with direct dynamic splats. STAR final heldout PSNR
was `13.873907089233398` versus direct splats `11.085673332214355`; STAR
selected heldout PSNR was `13.915654182434082` at step `600`. Final STAR
render-only eval was `0.042239958012942225s`, selected STAR render-only eval
was `0.034174208994954824s`, and direct splats render-only eval was
`0.40090283303288743s`. Final and selected STAR Metal stats stayed clean:
pair ratio about `2.74-3.07`, max tile count `74-78`, zero overflow, and zero
unstable tiles. This is the current clearest answer to the speed question:
the latest selected STAR artifact is not render-slower than direct dynamic
splats; it renders about `11.7x` faster by the synchronized render-only field.
The remaining blocker is schedule/selection and unbiased validation, not a
blind forward-rasterizer rewrite.

The seed-0 STAR-only LR-decay repeat used the same all-train grid all-frames
recipe with `300 -> 0.05x` decay and `--skip-splats`. The final checkpoint
reached heldout PSNR `13.81613826751709` at step `970`; the heldout-selected
diagnostic checkpoint reached `13.87098217010498` at step `600`, elapsed
`37.35523024998838s`, with selected render-only eval
`0.04048433306161314s`. Final and selected Metal stats stayed clean: max tile
count `94-102`, pair ratio `3.63-4.03`, zero overflow, and zero unstable tiles.
This makes the tuned LR-decay schedule V-JEPA-crossing on seeds 0, 1, and 2 in
recent local rows, but the selected seed-0 artifact still uses heldout for
selection and the run skipped direct splats, so it is schedule evidence rather
than an unbiased comparison row.

A non-heldout train-curve selector is now wired as
`--uvt-select-checkpoint first_train_psnr_plateau`. It selects the first
checkpoint whose train PSNR gain from the previous checkpoint is at or below
`--uvt-select-train-psnr-plateau-delta` for
`--uvt-select-train-psnr-plateau-patience` consecutive checkpoint intervals
and records `uses_heldout_for_selection: false`. The CPU smokes
`multicam_heldout_compare_train_plateau_smoke_16_2f_1s` and
`multicam_heldout_compare_train_plateau_patience2_smoke_16_2f_1s` passed. A
real seed-2 STAR-only MPS run with delta `0.5` and patience `1` selected step
`400`, elapsed
`26.757762582972646s`, heldout PSNR `13.83452320098877`, and render-only eval
`0.03962891804985702s`; the final checkpoint reached heldout PSNR
`13.786846160888672`, while the heldout-best checkpoint in the same curve was
step `500` at `13.94494915008545`. This is useful because it clears the V-JEPA
F32 heldout reference without peeking at the heldout camera, but it is not yet
the solved selector: it leaves about `0.11` dB versus the heldout-selected best
checkpoint on this run. A seed-1 STAR-only repeat also selected step `400`
without heldout selection, reaching heldout PSNR `13.771395683288574`; the
final checkpoint was slightly better at `13.790154457092285`, and the
heldout-best checkpoint was step `700` at `13.826448440551758`. The current
read is that train-plateau selection is a useful non-heldout diagnostic and
stays above V-JEPA on seeds 1 and 2, but it is not a reliable improvement over
the final checkpoint yet.

The patience-2 variant selected step `500` on both real MPS repeats. On seed 2,
it selected heldout PSNR `13.84100341796875`, improved over that run's final
`13.805965423583984`, and missed the heldout-best step `600` value
`13.855948448181152` by only about `0.015` dB. On seed 1, it selected heldout
PSNR `13.704198837280273`, improved over final `13.688732147216797`, but
missed that run's heldout-best step `400` value `13.726997375488281`. Both
selected rows keep `uses_heldout_for_selection: false` and clear the V-JEPA
F32 heldout reference, but the mixed seed-1 result means patience `2` is only a
candidate selector, not the final rule.

The newer `first_train_psnr_gain_drop` selector picks the previous checkpoint
when train-PSNR gain has already fallen under the low-gain threshold and then
drops by at least `--uvt-select-train-psnr-gain-drop`. It is still non-heldout:
selected reports keep `uses_heldout_for_selection: false`. The CPU smoke
`multicam_heldout_compare_train_gain_drop_smoke_16_2f_1s` passed. In STAR-only
MPS repeats, the selector cleared the V-JEPA F32 heldout reference on seeds 0,
1, and 2. Seed 0 selected step `400`, heldout PSNR
`13.901209831237793`, versus heldout-best step `500` at
`13.915482521057129` and final `13.903148651123047`. Seed 1 selected step
`400`, heldout PSNR `13.721198081970215`, versus heldout-best step `500` at
`13.735674858093262` and final `13.683605194091797`. Seed 2 selected step
`500`, exactly matching heldout-best in that curve at heldout PSNR
`13.904694557189941` versus final `13.872002601623535`. Selected Metal stats
were clean on all three STAR-only runs: zero overflow and zero unstable tiles.

The paired seed-2 gain-drop report against lens-aware direct dynamic splats is
the clean first comparison artifact. In that run STAR final reached heldout
PSNR `13.835267066955566`, while the gain-drop selected checkpoint chose step
`400`, heldout PSNR `13.888997077941895`, train PSNR
`16.00412082672119`, and render-only eval `0.04640516696963459s` without
heldout-camera selection. The direct dynamic splat baseline reached heldout
PSNR `11.190529823303223`, train PSNR `17.396635055541992`, and render-only
eval `0.9052186670596711s`. The selected STAR row missed the run's
heldout-best step `500` by about `0.0135` dB and rendered about `19.5x` faster
than direct dynamic splats by the synchronized render-only field.

A second paired run on weaker seed 1 confirms the comparison. STAR final
reached heldout PSNR `13.881017684936523`; the gain-drop selected checkpoint
again chose step `400`, heldout PSNR `13.879861831665039`, train PSNR
`15.686367511749268`, and render-only eval `0.0712511669844389s` without
heldout-camera selection. Direct dynamic splats reached heldout PSNR
`11.199346542358398`, train PSNR `17.342755794525146`, and render-only eval
`0.7236963339382783s`. The selected STAR row missed heldout-best step `500`
by about `0.0142` dB and rendered about `10.2x` faster than direct dynamic
splats despite a noisier selected render timing. This is now enough to freeze
gain-drop as the current reporting selector for the next scale/full-resolution
probe. It is still a research rule, not a production default.

The 512px/window-1 capacity bracket tested tube counts above the original
256-tube setting under the same 60-second STAR-only protocol. Increasing to
384 tubes rescued the weaker seed-1 final checkpoint to heldout PSNR
`13.640532493591309`, but regressed seed 2 to final heldout PSNR
`13.4086275100708`, so it is rejected as a default. The 320-tube setting is a
mixed middle point: seed 1 reached final heldout PSNR `13.682265281677246`,
non-heldout selected PSNR `13.637592315673828`, and heldout-best PSNR
`13.769192695617676`; seed 2 reached final/best heldout PSNR
`13.637543678283691`, while the non-heldout selected row was slightly under the
V-JEPA F32 reference at `13.598714828491211`. Seed 0 sharpened the read: 320
tubes reached a better heldout-best peak (`13.70832633972168` at step `400`)
than 256 tubes (`13.636795043945312` at step `600`), but the current non-heldout
selector picked step `600` for 320 tubes and fell to `13.437091827392578`, while
256 tubes selected the heldout-best step and cleared V-JEPA at
`13.636795043945312`. Moving the 320-tube seed-0 LR decay earlier to step `400`
did not fix this: final heldout PSNR was `13.612069129943848`, selected was
`13.586051940917969`, and heldout-best fell to `13.652498245239258`. The clean
checkpoint render-only timings stayed fast (`0.104-0.132s`) with zero overflow
and zero unstable tiles, so the capacity bracket does not point to a
forward-rasterizer rewrite. It points to seed-robust scale policy and selection:
320 tubes has useful oracle peaks, 384 is not a default, and 256 tubes remains
the cleaner current selector path on seeds 0 and 2.

The first 512px same-budget scale probe rejects a naive resolution bump with
the same 256-tube, 60-second recipe. On seed 2, STAR only completed `70` steps
in `68.71951654099996s`, the gain-drop selector fell back to the final
checkpoint, and STAR heldout PSNR was only `9.205381393432617`. Direct dynamic
splats completed `2095` steps in `60.02396495900001s` and reached heldout PSNR
`10.980579376220703`. The selected STAR re-eval render-only timing was
`0.09585146000000577s` versus direct splats `0.4002393330000018s`, so the
speed story is not dead, but quality is nowhere near competitive. The 512px
blocker is train-step throughput and scale strategy, not checkpoint selection;
the next 512px attempt should change budget, multiscale/crop strategy, or
training throughput before trying to claim full-resolution parity.

A follow-up 512px scale-strategy probe changed only the temporal training
window to `--uvt-window-frames 1`, then reran as a formal paired comparison
against direct dynamic splats. This rescued the 512px run: STAR completed
`1188` steps in `60.02716958400015s`, reached final heldout PSNR
`13.701825141906738`, final train PSNR `16.308331966400146`, and final
render-only eval `0.11522445899981903s`. Direct dynamic splats completed
`1856` steps in `60.019441415999836s`, reached heldout PSNR
`10.760580062866211`, train PSNR `15.943255424499512`, and render-only eval
`0.5262394170001699s`. The non-heldout gain-drop selector chose step `600`,
heldout PSNR `13.678083419799805`, train PSNR `15.948731899261475`, and
render-only eval `0.11207666699988295s`; the heldout-best checkpoint in the
same curve was step `900` at `13.729055404663086`. Metal stats stayed clean:
zero overflow, final pair ratio `2.337-2.667`, max tile count `56-66`, and
only `0.00235` max unstable-tile fraction. The scale-up blocker is therefore
windowing/scale strategy plus a better selector, not an immediate
forward-rasterizer rewrite.

A stricter 512px STAR-only selector diagnostic lowered
`--uvt-select-train-psnr-plateau-delta` from `0.5` to `0.1`. It delayed
gain-drop selection from the early shoulder to step `800`, but did not improve
selected heldout PSNR: selected was `13.677860260009766`, while final was
`13.706064224243164` and heldout-best was step `1100` at
`13.707569122314453`. This keeps the row above the V-JEPA F32 reference, but
it rejects "just lower the gain threshold" as the 512px selector fix. For
512px/window-1, final or best-train reporting is currently cleaner than the
gain-drop selector until a real scale-aware rule is designed.

The seed-1 512px/window-1 repeat is a useful limitation. The formal paired
seed-1 row used the previous seed-1 schedule (`--uvt-lr-decay-step 300`) and
reached STAR final heldout PSNR `13.494588851928711`, selected heldout PSNR
`13.415751457214355`, and direct dynamic splat heldout PSNR
`10.3926362991333`. Two STAR-only schedule probes then bracketed later decay:
step `500` improved the final row to heldout PSNR `13.576004028320312` with
heldout-best step `700` at `13.58298397064209`, while step `700` regressed to
final heldout PSNR `13.397587776184082`. Thus 512px/window-1 is repeatably a
large win over direct splats on seed 1, but the V-JEPA crossing is not
seed-stable yet. The next 512px quality branch should change representation
capacity, support/window policy, or multiscale/crop training, not keep nudging
LR-decay or gain-threshold scalars.

Checkpoint rows now also include `train_view_eval_psnr`,
`train_min_view_eval_psnr`, and `train_view_eval_psnr_gap`, with an opt-in
`--uvt-select-checkpoint best_min_train_view_psnr` diagnostic. The CPU smoke
`multicam_heldout_compare_train_view_selector_smoke_16_2f_1s` passed. The real
seed-2 MPS run rejected this selector as a schedule rule: it selected the final
step `1000`, heldout PSNR `13.835639953613281`, because the minimum train-view
PSNR kept improving monotonically. The heldout-best checkpoint in the same
curve was step `600`, heldout PSNR `13.904929161071777`. Keep the per-view
checkpoint fields for diagnosis, but do not spend more runs on plain
`best_min_train_view_psnr` as the non-heldout selector.

A cleaner train-camera dev split is also wired via
`--uvt-optimizer-train-views first_only` plus
`--uvt-select-checkpoint best_train_view_psnr --uvt-select-train-view-index 1`.
The CPU smoke `multicam_heldout_compare_train_view_dev_selector_smoke_16_2f_1s`
passed and confirmed `optimizer_train_view_indices: [0]` with selected metric
`train_view_1_eval_psnr`. The real seed-2 MPS run rejected this as a path to
the current quality target: training only on camera `0006` and selecting on
train camera `0014` selected the final step `972`, true heldout PSNR
`12.647516250610352`; the best true heldout checkpoint in that curve was step
`800` at `12.661633491516113`. This is below the V-JEPA F32 heldout reference
and far below the all-train recipe. Use this result to avoid one-camera
optimization as the next selector lane; future unbiased selection needs a
lighter validation signal without throwing away one of the two train cameras.

The combined train-plateau plus train-camera-balance selector is also wired as
`--uvt-select-checkpoint first_balanced_train_psnr_plateau`, with CPU smokes for
both explicit gap max `1.0` and default gap max `1.2`. It is useful as a
diagnostic, but the live 512px/320-tube repeats reject it as the current
selector. On seed 0 with gap max `1.0`, it selected step `400`, heldout PSNR
`13.62360954284668`, just under the V-JEPA F32 reference, while heldout-best was
step `900` at `13.625420570373535`. On seed 1 with gap max `1.0`, it selected
step `900`, heldout PSNR `13.600866317749023`, while heldout-best was step
`400` at `13.698472023010254`. Relaxing to gap max `1.2` on seed 1 selected a
bad step `500` shoulder at heldout PSNR `13.442426681518555`, while final was
`13.662494659423828` and heldout-best was step `800` at
`13.675647735595703`. Do not keep tuning this selector threshold next; the
512px issue remains scale-aware support/window policy or a validation-shaped
selection signal, not a forward-rasterizer rewrite.

A 512px seed-0 320-tube window-2 policy check is also negative. It changed only
`--uvt-window-frames` from `1` to `2` against the same 60-second STAR-only
recipe. STAR completed `536` steps, selected step `500`, and reached selected
heldout PSNR `13.422961235046387`; final was `13.391561508178711`, and
heldout-best was step `400` at `13.558874130249023`. The matching window-1
seed-0 320-tube run had a much stronger heldout-best peak
`13.70832633972168`, even though its selector missed. Metal stayed clean
(`0` overflow, `0.0` unstable tiles, selected max tile `84-89`), so window-2 is
not the support/window fix for 512px seed robustness.

A hard LR-drop check on the same 512px seed-0 320-tube window-1 branch is also
negative. It moved the decay to step `400` and reduced the factor to `0.005`.
The run completed `1000` steps, but final, selected, and heldout-best all
landed on step `1000` with heldout PSNR `13.599885940551758`, below the earlier
window-1 seed-0 320-tube heldout-best peak `13.70832633972168` and below the
softer step-400 decay run's heldout-best `13.652498245239258`. Render remained
fast and clean: selected render-only eval was `0.13297337400126708s`, max tile
count was `103`, max pair ratio was `3.2112812143984426`, overflow was `0`, and
unstable tile fraction was `0.0`. This rejects "freeze the shoulder with a
harder scalar decay" as the next 512px fix.

The temporal-floor support check is negative too. On the same seed-0 320-tube
512px branch, `--uvt-min-lambda-t 0.7` reached final/heldout-best PSNR
`13.514815330505371` and selected `13.336383819580078`; `--uvt-min-lambda-t
2.0` fell further to final/heldout-best `13.086738586425781` and selected
`12.77337646484375`. Both runs kept zero overflow and zero unstable tiles, so
the overfit temporal-support win does not transfer as a simple multicam
temporal floor.

The train-view gap-collapse selector is wired as
`--uvt-select-checkpoint first_train_view_gap_collapse`. It selects the
previous checkpoint before `train_view_eval_psnr_gap` falls below
`--uvt-select-train-view-gap-collapse` and labels selected reports as
`uses_heldout_for_selection: false`. A CPU smoke passed, and a post-hoc
threshold `0.7` read looked promising on the saved 320-tube curves: it picked
the heldout-best checkpoint on seed 0 and seed 1 and fell back to final on
seed 2. The live seed-0 MPS rerun rejects it as the current selector, though:
it selected step `400`, heldout PSNR `13.490143775939941`, while final reached
`13.70932674407959` and heldout-best was step `900` at
`13.739107131958008`. Keep the selector as a diagnostic only; the train-view
gap collapse is not stable enough across repeats.

The train-camera temporal dev-frame selector is wired as
`--uvt-validation-frame-stride/--uvt-validation-frame-offset` plus
`--uvt-select-checkpoint best_train_dev_frame_psnr`. Checkpoint rows now report
`train_fit_frame_eval_psnr` and `train_dev_frame_eval_psnr`, and
`--uvt-init-frames fit` can exclude dev frames from initialization as well as
optimizer sampling. The CPU fit-init smoke passed, but the real 512px seed-0
320-tube checks reject this selector lane. The clean fit-init run selected the
final step `1059` by dev-frame PSNR and reached true heldout PSNR
`13.461018562316895`, while its heldout-best checkpoint was step `300` at
`13.579765319824219`. The all-init control also selected final, heldout PSNR
`13.39246654510498`, while heldout-best was step `600` at
`13.449368476867676`. Both stayed below the earlier no-dev 320-tube oracle
peak `13.70832633972168`, so train-camera temporal dev frames are not the next
512px selector path.

The free init-lambda support check is negative. It changed only
`--uvt-init-lambda-t` to `2.0` on the same 512px seed-0 320-tube window-1
branch, without setting a `min_lambda_t` floor. The selected non-heldout
gain-drop checkpoint was step `600`, true heldout PSNR `13.213919639587402`;
final was step `1209` at `13.40640640258789`, and heldout-best was step
`1100` at `13.415067672729492`. Metal stayed clean with max tile `76`, max
pair ratio `4.07200551573087`, zero overflow, and zero unstable tiles, but
quality is far below the no-dev no-floor 320-tube peak. Initializing narrower
temporal support is not the 512px fix.

The bounded sequence-consistency support check is also not the current 512px
fix. The hook is wired as `--uvt-sequence-consistency-every-steps`,
`--uvt-sequence-consistency-frames`, and `--uvt-sequence-consistency-weight`,
and the CPU smoke passed. Full 16-frame consistency failed before useful
training on MPS with `RuntimeError: Invalid buffer size: 12.00 GiB`, so the
real checks used four consistency frames. Every 20 steps was too expensive:
only `340` steps in 60 seconds, heldout PSNR `13.58269214630127`, selected
render `0.11685991700323939s`, and a tiny heldout unstable-tile fraction.
Every 50 steps recovered throughput to `666` steps and clean Metal stats, with
selected non-heldout checkpoint step `600`, true heldout PSNR
`13.619542121887207`, selected render `0.11323120699853462s`, and final /
heldout-best step `666` at `13.626453399658203`. That only roughly matches the
V-JEPA row at final and remains below the no-consistency 320-tube heldout-best
peak `13.70832633972168`; the bottleneck still reads as support/window or
training policy, not a first-priority rasterizer rewrite.

The multiscale auxiliary loss hook is wired and smoke-tested as
`--uvt-multiscale-loss-weight` plus `--uvt-multiscale-loss-factor`. It reuses
the existing render and adds a downsampled reconstruction loss, so it tests
scale/support weighting without changing the rasterizer. The first full 512px
bracket used factor `4`, weight `0.25`. Seed 0 improved final/heldout-best to
PSNR `13.656238555908203` with clean Metal stats, above the V-JEPA reference,
but the current gain-drop selector fired early at step `500`, heldout PSNR
`13.591398239135742`. Seed 1 rejects this as a default: no-multiscale seed 1
had selected/final/heldout-best `13.637592315673828` / `13.682265281677246` /
`13.769192695617676`, while multiscale fell to selected/final/heldout-best
`13.418802261352539` / `13.384246826171875` / `13.520487785339355`.
The lower-weight seed-1 check at factor `4`, weight `0.05` improved over
weight `0.25` but still did not recover the no-multiscale branch: selected /
final / heldout-best were `13.568358421325684` / `13.560102462768555` /
`13.603869438171387`, with clean Metal stats. Conclusion: simple global
factor-4 multiscale loss is not the 512px fix. Keep the hook for targeted
crop/scale tests, but do not spend the next pass on another near-zero global
weight unless there is a sharper selector or crop hypothesis.

The deterministic crop auxiliary hook is wired and smoke-tested as
`--uvt-crop-loss-weight` plus `--uvt-crop-loss-size`. It cycles through a 3x3
crop grid from the already-rendered train output, so it tests local full-res
weighting without changing the rasterizer. On the failing 512px seed-1 branch,
crop size `256`, weight `0.25` reached only `578` steps in 60 seconds and
selected/final/heldout-best all landed at heldout PSNR `13.566254615783691`.
Crop size `128`, weight `0.25` similarly reached only `584` steps; selected was
step `500`, heldout PSNR `13.48839282989502`, final `13.565434455871582`, and
heldout-best step `400` at `13.591614723205566`. Both stayed clean in Metal but
fell well below the no-crop seed-1 row, so deterministic crop loss is not the
512px support fix.

The 512px deterministic cycle train-schedule revisit is mixed and rejected as a
seed-robust default. It keeps the no-aux 320-tube/window-1 recipe and changes
only `--uvt-train-schedule cycle`. Seed 0 selected step `600` by non-heldout
train-gain-drop at heldout PSNR `13.798948287963867`, with heldout-best step
`500` at `13.841632843017578`. Seed 1 also selected step `600` without using
heldout, at heldout PSNR `13.915006637573242`, with heldout-best step `500` at
`13.91877555847168`. Those rows clear the V-JEPA F32 reference `13.6248`, beat
the saved random-schedule selected checkpoints, and render in
`0.10962479100089695-0.11339320700062672s` across the three eval sequences.
Seed 2 rejects the rule: LR `0.01` selected step `400` at heldout PSNR
`13.023938179016113`, heldout-best was only `13.442176818847656`, and the run
became non-finite after step `480`. A seed-2 LR `0.005` stability bracket stayed
finite but underfit, selecting/finaling at heldout PSNR `13.037338256835938`.
Cycle is still useful evidence that sampling order matters, but the next
512px branch needs a shuffled/phase-randomized coverage schedule or a stability
guard, not a rasterizer rewrite.

The trainer now has a lightweight non-finite guard for this class of failure.
It checks the scalar loss at the existing log cadence, records
`stopped_reason` / `stopped_step`, writes non-finite scalar log fields as
`null`, and restores the last checkpointed finite state before final eval. The
normal CPU smoke
`multicam_heldout_compare_nonfinite_guard_smoke3_16_2f_1s` passed with
`stopped_reason: null`; the forced CPU smoke
`multicam_heldout_compare_nonfinite_guard_forced_smoke_16_2f_1s` passed with
`stopped_reason: nonfinite_loss` at step `1` and evaluated the restored finite
state. This makes future bad branches honest without changing the renderer.

The `shuffled_cycle` train schedule is wired and smoke-tested. It cycles through
a deterministic shuffled coverage order for view/frame or view/window pairs,
keeping the same no-aux 320-tube/window-1 raster path. On the failing seed 2,
`mcam512_s2_t320_shuffled_cycle` stayed finite for `920` steps and selected
step `600` at heldout PSNR `13.574305534362793`, render-only
`0.13692954100042698s`; heldout-best was step `300` at `13.6640625`. That
rescues the plain-cycle collapse and clears V-JEPA only as a heldout-best
oracle, not as the current non-heldout selected checkpoint. Moving the LR drop
earlier to step `300` is worse: selected `13.564926147460938`, heldout-best
`13.578947067260742`. A post-hoc balanced train-plateau selector looks
interesting on saved curves, but the live shuffled seed-2 rerun with
`delta=0.6`, `gap=1.7` did not reproduce the better step-300 shoulder: selected
`13.52459716796875`, heldout-best `13.557467460632324`. Current read: fix
sampling/selection stability before rasterizer work. A fixed-step rerun stopping
exactly at `300` steps also failed to reproduce the original shoulder:
`mcam512_s2_t320_shuffled_cycle_fixed300` reached heldout PSNR
`13.527697563171387` with clean Metal stats. Treat the original step-300
heldout-best as a noisy non-promotable observation, not a reliable selector
target.

The `reshuffled_cycle` schedule is now wired and smoke-tested. It preserves full
coverage but reshuffles the coverage order each epoch instead of repeating one
fixed shuffled order forever. The fixed-600 three-seed check is the first
seed-robust 512px schedule candidate: seed 0 selected/final heldout PSNR
`13.639025688171387`, seed 1 `13.894740104675293`, and seed 2 selected
`13.700183868408203` at step `500` with final `13.660613059997559`. All three
selected checkpoints are non-heldout selections, finite, clean in Metal, and
above the V-JEPA F32 reference `13.6248`. The tradeoff is that seed 0 is much
worse than the plain-cycle seed-0 result, so this is a robust-floor candidate,
not an outright best-quality default.

The `phase_rotated_cycle` branch is wired and smoke-tested, but rejected as the
next default. It keeps cycle's ordered coverage while rotating the start point
each epoch. Seed 2 fixed-600 was finite and slightly beat reshuffled at selected
heldout PSNR `13.706971168518066` with fast render-only `0.11264262600161601s`,
but seed 0 fell to `13.602667808532715`, below the V-JEPA reference and below
reshuffled. Do not spend seed 1 on this branch unless a later policy explains
the seed-0 drop.

The `view_shuffled_cycle` branch is wired and smoke-tested. It keeps frames or
windows in temporal cycle order, but shuffles the train-camera order inside each
frame/window slot. This is the strongest seed-2 schedule so far: fixed-600 seed
2 selected step `500` at heldout PSNR `13.788138389587402`, final/heldout-best
step `600` at `13.793721199035645`, with render-only `0.14073404100054177s`.
The full three-seed selected matrix clears V-JEPA without heldout selection:
seed 0 `13.639522552490234`, seed 1 `13.7864990234375`, seed 2
`13.788138389587402`. The fixed-600 final checkpoints are better on seeds 1 and
2 (`13.812097549438477` and `13.793721199035645`), so the current gain-drop
selector is too eager for this schedule. Treat this as the best current
robust-floor schedule, with selector work still open.

The existing `best_train_psnr` selector is the right current reporting selector
for `view_shuffled_cycle`. The CPU smoke
`multicam_heldout_compare_view_shuffled_besttrain_selector_smoke_16_2f_1s`
passed with `selector: best_train_psnr`, `uses_heldout_for_selection: false`.
On the saved 512px view-shuffled curves it selects step `600` on all three
seeds and matches the heldout-best checkpoint without using heldout: seed 0
`13.639522552490234`, seed 1 `13.812097549438477`, seed 2
`13.793721199035645`. Next reports for this schedule should use
`--uvt-select-checkpoint best_train_psnr`, not gain-drop.

## Latest Evidence

```bash
python3 tests/gate0_check.py
```

Passes six tiny scenes:

```text
single_static
moving_diagonal
two_non_crossing
crossing_depth
fast_screen_motion
wide_temporal_support
```

Metal parity is at or below `5.97e-08` max RGB error, with zero overflow.

```bash
python3 research_project/trainer_harness/smoke_train.py
```

Passes on CPU with `moving_diagonal`: initial loss
`4.5250795665197074e-04`, final loss `1.7662021491560154e-05`.

```bash
python3 research_project/trainer_harness/train_video.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 4 --target-size 16 --max-frames 2 --steps 1 --lr 0.02 --device cpu
```

Passes as a loader/plumbing smoke: 2 frames loaded, initial loss
`0.29026854038238525`, final loss `0.2883112132549286`.

```bash
python3 research_project/trainer_harness/world_projection_smoke.py
```

Added as the Phase 2a projection smoke. It covers an orthographic
`WorldTubeBatch` conversion into the exact Gate 0 tensor contract; it does not
cover perspective camera projection yet.

Current result: 2 projected tubes, Metal-vs-brute max RGB error
`5.960464477539063e-08`, zero overflow.

```bash
python3 research_project/trainer_harness/camera_spec_projection_smoke.py
```

Added as the Phase 2c adapter smoke. It builds a Dynaworld `CameraSpec`,
converts it to the local `PinholeCamera`, and projects world tubes through the
same Gate 0 tensor contract.

Current result: 2 projected tubes, Metal-vs-brute max RGB error
`5.960464477539063e-08`, zero overflow.

```bash
python3 research_project/trainer_harness/pinhole_projection_smoke.py
```

Added as the Phase 2b projection smoke. It covers a pinhole camera with
`world_to_camera` and intrinsics, then converts the local fronto-parallel
`WorldTubeBatch` into the exact Gate 0 tensor contract.

Current result: 2 projected tubes, Metal-vs-brute max RGB error
`5.960464477539063e-08`, zero overflow.

```bash
python3 research_project/trainer_harness/gradient_probe.py
```

Added as the Phase 3a dense-backward smoke. It verifies finite, nonzero
gradients for projected tube center, velocity, precision, opacity, and color
parameters.

```bash
python3 research_project/trainer_harness/metal_autograd_smoke.py
```

Added as the Phase 3b hybrid autograd smoke. It uses the Metal renderer for the
forward image and dense PyTorch as the backward reference. This is not a true
Metal backward kernel.

Current result: MPS smoke passes with finite gradients for `ma`, `q_uvt`,
`opacity`, and `color`. `depth0` and `depth_beta` gradients are zero because the
dense reference renderer uses detached depth ordering.

```bash
python3 research_project/trainer_harness/simple_metal_backward_smoke.py
```

Added as the Phase 3c simplified true-Metal backward probe. It compares
per-sample Metal gradients against dense autograd for the single-tube,
no-compositing case. It is not the full sorted-tile compositing backward owner.

Current result: max absolute gradient errors are `1.9073486328125e-06` for
color, `1.7881393432617188e-07` for `ma`, and `0.0` for `q_uvt` and opacity.

```bash
python3 research_project/trainer_harness/stable_metal_backward_smoke.py
```

Added as the Phase 3d stable sorted-tile Metal backward probe. It reuses the
Metal bin/sort tile path, computes sorted alpha-compositing gradient
contributions in Metal, reduces them by tube id in Python, and compares against
a matching dense autograd reference. It does not cover unstable fallback or
depth-order gradients.

Current result: max absolute gradient errors are `3.4332275390625e-05` for
opacity, `1.1444091796875e-05` for color and `q_uvt`, and
`4.76837158203125e-06` for `ma`.

```bash
python3 research_project/trainer_harness/unstable_metal_backward_smoke.py
```

Added as the Phase 3e unstable fallback Metal backward probe. It uses the
deterministic per-sample depth ordering path on the `crossing_depth` scene and
compares gradients against a matching dense autograd reference. It still reduces
sample contributions in Python and does not include gradients through the
discrete depth order.

Current result: 4 unstable tiles, max absolute gradient errors are
`3.4332275390625e-05` for `q_uvt`, `3.0517578125e-05` for opacity,
`1.33514404296875e-05` for color, and `5.9604644775390625e-06` for `ma`.

```bash
python3 research_project/trainer_harness/tile_metal_autograd_smoke.py
```

Added as the Phase 3f autograd bridge smoke. It uses Metal forward, Metal
per-sample backward, and MPS `index_add_` reduction by tube id on
`crossing_depth`.

Current result: finite nonzero gradients for `ma`, `q_uvt`, opacity, and color;
depth gradients remain zero because the depth order is discrete/detached.

```bash
python3 research_project/benchmarks/backward_performance_smoke.py
```

Added as the Phase 3g bounded performance smoke. It compares dense MPS backward
against the Metal tile-backward autograd bridge on a synthetic 16-tube,
32x32x4 case. This is a smoke-scale timing check, not a promotion benchmark.

Current result with one warmup iteration and 2 measured iterations: dense MPS
mean `16.18629200675059 ms`; Metal tile-backward mean
`36.01418749894947 ms`. On this tiny case the dense path is faster.

```bash
python3 research_project/benchmarks/backward_performance_matrix.py
```

Added as the Phase 3h bounded large-scene timing benchmark. It runs the same
timed dense-vs-Metal comparison on `smoke` and `large_local`; the latter uses 64
tubes, 64x64 resolution, 8 frames, 1 warmup iteration, and 1 measured iteration.
This is still local bounded evidence, not production-scale training evidence.

Current `large_local` result: dense MPS mean `104.00879199733026 ms`; Metal
tile-backward mean `73.45674998941831 ms`; dense-to-Metal mean ratio
`1.4159187823081347`.

```bash
python3 research_project/benchmarks/uvt_pair_benchmark.py
```

Added as the Phase 4a renderer benchmark. It reports UVT tile-tube pairs,
sliced per-frame tile-splat pairs, pair ratios, CPU brute-force timing, and
Metal forward timing for the six tiny scenes. This is not the full per-frame
training comparison yet.

Current result: 6 scenes, mean pair ratio `0.5`, max pair ratio `0.5`, zero
overflow, and max Metal-vs-brute RGB error `5.960464477539063e-08`.

```bash
python3 research_project/benchmarks/training_comparison.py
```

Added as the Phase 4b tiny training comparison. It fits UVT tubes and an
independent per-frame Gaussian baseline to the same deterministic target and
reports loss, L1, parameter count, and wall-clock timing. This is still not a
full FasterGS/video-quality benchmark.

Current result on `moving_diagonal`, 25 CPU steps: UVT loss
`4.5250795665197074e-04 -> 1.7662021491560154e-05`; per-frame loss
`3.663150127977133e-04 -> 1.1087417988164816e-05`.

```bash
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 8 --per-frame-splats 8 --target-size 32 --max-frames 4 \
  --steps 8 --lr 0.04 --device cpu --seed 5 \
  --out-json research_project/benchmarks/results/video_fit_comparison_fixture.json \
  --contact-sheet research_project/benchmarks/results/video_fit_comparison_fixture.png
```

Added as the Phase 4c real-video fixture comparison. It reuses the Dynaworld
video loader, fits projected UVT tubes and the simple per-frame Gaussian
baseline to the same 4-frame 32x32 target, and writes a contact sheet. It is not
a current FasterGS comparison and has no held-out-camera split.

Current result: UVT loss `0.3166208863258362 -> 0.2973604202270508`, final L1
`0.5045824646949768`, 104 parameters, wall-clock `1050.151959003415 ms`;
per-frame loss `0.31666192412376404 -> 0.2972259521484375`, final L1
`0.5054242014884949`, 288 parameters, wall-clock `52.544709003996104 ms`.

```bash
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 64 --per-frame-splats 64 --target-size 32 --max-frames 16 \
  --steps 200 --lr 0.04 --device mps --seed 7 \
  --out-json research_project/benchmarks/results/video_fit_single_overfit_32_16f_200steps_64cap.json \
  --contact-sheet research_project/benchmarks/results/video_fit_single_overfit_32_16f_200steps_64cap.png
```

Added as the Gate 6e fixed-step single-video overfit check. Current 64-capacity
result: UVT reached PSNR `21.764323711395264`, L1 `0.05626721307635307`, and
render time `6.768666004063562 ms` after 200 steps in
`9535.820624994813 ms`. Per-frame splats reached PSNR `25.14953851699829`, L1
`0.04000972956418991`, and render time `40.68333297618665 ms` after the same
200 steps in `61065.20704101422 ms`. The lower 32-capacity run showed the same
shape: UVT PSNR `20.808022022247314` versus per-frame splats
`23.24096441268921`, with UVT faster. This is a speed signal for small
single-video overfit, not a quality win.

```bash
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 128 --per-frame-splats 64 --target-size 32 --max-frames 16 \
  --steps 200 --lr 0.12 --device mps --seed 7 --uvt-init-mode random \
  --out-json research_project/benchmarks/results/video_fit_single_overfit_32_16f_200steps_128uvt_64pf_lr012.json \
  --contact-sheet research_project/benchmarks/results/video_fit_single_overfit_32_16f_200steps_128uvt_64pf_lr012.png
```

Added as the Gate 6f single-video overfit ablation. Increasing LR and capacity
helped. Data-sampled UVT init helped only after LR was raised. The best paired
run so far is 224 UVT tubes, 64 splats/frame, LR `0.32`, temporal precision
`0.5`, opacity `0.7`: STAR-UVT reached PSNR `26.46265983581543`, L1
`0.03167424723505974`, train time `66625.27337501524 ms`, and render time
`31.05658298591152 ms`; per-frame splats reached PSNR `27.248921394348145`, L1
`0.030112704262137413`, train time `118614.97245798819 ms`, and render time
`102.62579101254232 ms`. The best 128-tube UVT-only point used video-sampled
init, temporal precision `0.5`, opacity `0.7`, and LR `0.32`, reaching PSNR
`24.639911651611328`. A 240-tube run was slightly worse than 224 tubes at PSNR
`26.36221408843994`. A convergence bracket on the tuned 224-tube recipe found
that 340 UVT steps reached PSNR `27.101047039031982` in
`114658.05949998321 ms`, just under the tuned per-frame baseline runtime, and
400 UVT steps reached PSNR `27.22731113433838` in `131919.97808398446 ms`,
essentially tying the per-frame PSNR but losing the train-time edge. Simple
staged LR did not help: `0.48 -> 0.16` at step 100 reached PSNR
`25.79216480255127`, and `0.32 -> 0.16` at step 150 reached PSNR
`26.341335773468018`, both below constant LR `0.32`.

```bash
python3 research_project/benchmarks/uvt_forward_speed_probe.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --target-size 64 --max-frames 16 --tube-counts 224,448 \
  --spatial-precision 0.25 --temporal-precision 0.5 --opacity 0.7 \
  --out-json research_project/benchmarks/results/uvt_forward_speed_probe_64_16f_224_448_tuned_v2.json
```

Added as the Gate 6g 64px transfer and forward-speed probe. The tuned 224-tube
recipe at 64x64, 16 frames, 200 steps reached PSNR `23.345627784729004` in
`254147.23570799106 ms`; 448 tubes for 100 steps improved to PSNR
`23.94777774810791` but took `350301.7887909955 ms`. Metal forward timing on
video-initialized tensors was faster than dense forward: 224 tubes dense
`155.59799999270277 ms` versus Metal `47.28181932781202 ms`; 448 tubes dense
`309.5513886655681 ms` versus Metal `125.89868066910033 ms`. Both Metal cases
had zero overflow and pair ratio about `0.78`. The speed path exists at forward
render time. Dense-backward training is too slow for larger sweeps; the Metal
tile-backward path below is now the local iteration path.

```bash
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 224 --per-frame-splats 64 --target-size 32 --max-frames 16 \
  --steps 800 --lr 0.32 --device mps --seed 7 \
  --uvt-init-mode video_samples --uvt-spatial-precision 0.25 \
  --uvt-temporal-precision 0.5 --uvt-opacity 0.7 \
  --uvt-render-backend metal_tile --skip-per-frame \
  --out-json research_project/benchmarks/results/video_fit_uvt_ablate_32_16f_800steps_224cap_tuned_lr032_metal_tile.json
```

Added as the Gate 6h Metal tile-backward overfit path. At 32x32, 16 frames, the
tuned 224-tube recipe reached PSNR `27.42915630340576` after 800 Metal steps in
`54567.61045800522 ms`, render `1.6061250062193722 ms`. That beats the tuned
64-splats/frame baseline PSNR `27.248921394348145` while taking less than half
the train time. At 64x64, 224 tubes and 800 Metal steps reached PSNR
`24.250736236572266` in `128095.51870898576 ms`, render
`5.836124997586012 ms`; 1600 Metal steps only improved this to PSNR
`24.356164932250977` in `278467.5174159929 ms`, render
`1.7624159809201956 ms`. 448 tubes at the same 800-step settings was worse at
PSNR `23.577630519866943`. Lowering the 448-tube LR fixed that capacity result:
LR `0.16` reached PSNR `24.879634380340576`, LR `0.24` reached the current
64px best PSNR `25.096933841705322`, and LR `0.28` fell to
`24.401702880859375`. A same-step 64px comparison found 448 UVT at 200 Metal
steps within `0.09474992752075195` dB of 64 splats/frame at 200 steps, while
training `19.297393749991897s` versus `211.76061187498271s` and rendering
`2.980333985760808ms` versus `95.37416699458845ms`. At 800 Metal steps, 448
UVT already beat that splat baseline PSNR in `104.36115416602115s`; at 1600
steps it reached PSNR `25.285780429840088` in `229.8335517499945s`. The Metal
tile-backward reducer was made shape-aware so flattened sample buffers and id
buffers reduce safely, and it now gathers valid ids and samples from the same
explicit position list before `index_add_`.

```bash
python3 research_project/benchmarks/video_fit_comparison.py \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/test_data/test_video_small_128_4fps.mp4 \
  --tube-count 1792 --target-size 128 --max-frames 16 \
  --steps 400 --lr 0.12 --uvt-final-lr 0.04 \
  --uvt-final-lr-start-step 200 --device mps --seed 7 \
  --uvt-init-mode video_samples --uvt-spatial-precision 0.125 \
  --uvt-temporal-precision 0.5 --uvt-opacity 0.7 \
  --uvt-render-backend metal_tile --skip-per-frame \
  --out-json research_project/benchmarks/results/video_fit_uvt_ablate_128_16f_400steps_1792cap_lr012_to004_step200_s0125_metal_tile.json
```

Added as Gate 6i 128px single-video transfer. The 64px recipe transfers in
speed but needs resolution-aware capacity, support, and LR schedule. The first
1792-tube, LR `0.12`, spatial precision `0.125`, 200-step run reached PSNR
`22.2884202003479`, train `51.93527437499142s`, render
`1.6798329888843ms`; 400 constant-LR steps did not improve eval PSNR
(`22.23587989807129`). Color/opacity-only refinement after 200 steps also did
not help (`22.21776008605957`). Whole-model LR decay did help: LR
`0.12 -> 0.02` at step 200 reached `22.72578239440918`, LR `0.12 -> 0.04`
reached `22.809326648712158`, and LR `0.12 -> 0.06` fell to
`21.763882637023926`. Naive block-match velocity init was harmful
(`21.428205966949463`). Narrower temporal support then set the current 128px
best: temporal precision `1.0` reached PSNR `23.209903240203857`, while
temporal precision `2.0` was slightly lower at `23.207027912139893`. A bounded
50-step 128px paired run found UVT at PSNR `20.928823947906494` in
`13.987883624999085s`, render `2.2979159839451313ms`; 64 splats/frame reached
PSNR `19.460207223892212` in `175.3501634580025s`, render
`161.16950000287034ms`. This supports the speed story, but it is not yet a full
128px/200-step splat comparison.

Gate 6bp closes the 128px same-step temporal-support bracket without rerunning
the slow per-frame baseline. Reusing the same 128px/16-frame/200-step setup,
1792 UVT tubes at temporal precision `1.0` reached PSNR
`22.817583084106445` in `22.077392624999447s`, render
`4.774042000462941ms`; temporal precision `2.0` improved to PSNR
`23.130309581756592` in `17.787110125000254s`, render
`1.5611250000802102ms`; temporal precision `4.0` regressed to
`22.765743732452393`. Against the saved 64-splats/frame 200-step baseline,
the `t=2.0` row is `+2.5024056434631348` dB, `38.670927584589315x` faster to
train, and `91.76779565557226x` faster to render. A media rerun of the same
recipe emitted a target/STAR contact sheet at
`research_project/benchmarks/results/video_fit_single_overfit_128_16f_200steps_1792uvt_lr012_s0125_t20_uvtonly_sheet_metal_tile.png`
and reached PSNR `23.138446807861328` in `17.54805325000052s`, render
`1.0632920020725578ms`.
Gate 6bq then bracketed LR around that `t=2.0` setting. LR `0.08` reached PSNR
`22.904906272888184`, and LR `0.16` reached `22.98252582550049`, so LR `0.12`
remains the current 128px equal-step overfit setting.
Gate 6cd checks local quality headroom beyond the equal-step comparison. Keeping
the same 1792-tube, LR `0.12`, spatial precision `0.125`, temporal precision
`2.0`, opacity `0.7`, Metal tile-backward recipe but training STAR for 400
steps reached PSNR `23.569955825805664` in `31.383009374996618s`, render
`1.2202500001876615ms`, with contact sheet
`research_project/benchmarks/results/video_fit_single_overfit_128_16f_400steps_1792uvt_lr012_s0125_t20_uvtonly_sheet_metal_tile.png`.
Against the saved 64-splats/frame 200-step baseline, this is
`+2.942051887512207` dB, `21.917721126228813x` faster to train, and
`117.40299117241051x` faster to render.
Gate 6ce tests a more explicit temporal-piece init instead of raw capacity:
`--uvt-sample-mode temporal_quarters` reuses spatial sites across four temporal
quarters at the same 1792-tube count. It is rejected for the current 128px
recipe: the 400-step run reached PSNR `23.275623321533203` in
`32.35146966699904s`, render `1.253209000424249ms`, losing
`0.29433250427246094` dB to the random-sampled 400-step recipe while taking
slightly longer. Artifact:
`research_project/benchmarks/results/video_fit_single_overfit_128_16f_400steps_1792uvt_lr012_s0125_t20_temporalquarters_metal_tile.png`.
Gate 6cf tests a real split/refine operator: train 1792 tubes to step 200,
split each learned tube into two temporal children, then continue to step 400.
This is rejected in the current form. The aggressive split, offset `0.5` and
temporal precision scale `2.0`, reached PSNR `19.583353996276855`; the gentler
split, offset `0.25` and scale `1.0`, reached only `18.187309503555298`; adding
child opacity scale `2.0` improved to `21.271286010742188`, still
`2.2986698150634766` dB below the unsplit 400-step recipe and slower. The
logged post-split initial losses jump from pre-split losses around `0.0049` to
`0.0137-0.0201`, so the current split/refine handoff damages the representation
before the optimizer can recover it.
Gate 6ch checks whether the split failure was caused by temporal/depth
displacement rather than duplication itself. Even with offset `0.0`, precision
scale `1.0`, and opacity-conserving children, the split is rejected: the old
tiny depth offset row reached PSNR `19.849724769592285`, and zero depth offset
reached only `19.785715341567993`. The zero-depth row was also slow
(`182.22527079200154s`) and rendered in `26.041582998004742ms`. Its split
boundary still jumped from pre-split loss `0.004865488037467003` to post-split
initial loss `0.02108645997941494`, so the current duplicate split does not
preserve the Metal render at 128px.
Gate 6cg retests motion-aware initialization under the current best 128px
recipe. Raw block-match velocity init reached PSNR `22.520790100097656`,
`1.0491657257080078` dB below the zero-velocity recipe. A gated variant is now
available as `--uvt-velocity-init block_match_gated`, requiring a patch-error
improvement over zero motion, but it also loses: improvement ratio `0.9`
reached `23.01954984664917`, and stricter ratio `0.5` reached
`23.172695636749268`, still `0.3972601890563965` dB below zero velocity and
slower to train. Keep zero velocity for the current 128px overfit recipe.
Gate 6ci checks whether the current recipe simply needs a longer local budget.
It does not. Constant LR `0.12` for 800 steps regressed to PSNR
`22.233996391296387`, `1.3359594345092773` dB below the 400-step recipe. A
softer 800-step tail, LR `0.12 -> 0.04` at step 400, recovered to
`23.460845947265625` but still missed the 400-step recipe by
`0.10910987854003906` dB while taking `109.59267349999936s` and rendering in
`3.71179099965957ms`. Keep 400 steps as the local quality budget; Gate 6ck
below supersedes the older `tile_t=2` raster setting with `tile_t=1`.
Gate 6cj isolates the duplicate-split failure at the render boundary. The
probe artifact
`research_project/benchmarks/results/video_fit_split_boundary_probe_128_16f_1792_split200_preserve_depth0.json`
trains the current 1792-tube recipe to step 200, renders once, duplicates with
offset `0.0`, precision scale `1.0`, opacity scale `1.0`, and zero depth offset,
then renders once again. The pre-split render is PSNR `23.139398097991943`,
loss `0.004853557329624891`, forward time `9.447625001484994ms`, stable tile
fraction `1.0`, max tile count `207`, and overflow tile count `598`. The
post-split render drops to PSNR `17.03884720802307`, loss
`0.019774947315454483`, forward time `110.73370899975998ms`, stable tile
fraction `0.0`, unstable tile fraction `1.0`, max tile count `398`, and
overflow tile count `1898`. So the base STAR-UVT speed story is intact, but the
current duplicate split/refine path is a raster-capacity/fallback problem.
Gate 6ck exposes `--uvt-tile-t` and `--uvt-tile-capacity` in the local overfit
scripts and reruns the current 1792-tube recipe. The best speed/quality tradeoff
is now `tile_t=1`, capacity `128`. Gate 6co below supersedes the first one-shot
render timings with synchronized repeat timings. Capacity `256` is now a
quality-mode row with `tile_t=1`: at 200 steps
it reaches PSNR `23.22518825531006`, train `39.326092333001725s`, render
`11.385666999558453ms`; at 400 steps it reaches PSNR `24.083971977233887`,
train `77.38435312500224s`, render `5.169167001440655ms`. Keep cap-128 for the
equal-step speed claim and cap-256 for quality mode.
Gate 6cl retests split/refine under the improved tile shape and under the
current final tube count. Full duplicate split with `tile_t=1` still fails at
the boundary: pre-split PSNR `23.180255889892578`, forward
`4.090958998858696ms`, stable tile fraction `1.0`; post-split PSNR
`18.50017786026001`, forward `85.22179200008395ms`, unstable tile fraction
`1.0`, overflow tile count `3613`. A controlled 896-to-1792 scheduled split
under `tile_t=1` reaches only PSNR `22.63478994369507`, and the same schedule
under `tile_t=2` reaches `22.610313892364502`. Keep split/refine rejected.
Gate 6cm brackets LR under the new `tile_t=1`, cap-128 recipe. LR `0.10`
reaches PSNR `23.786139488220215`, train `42.99240170899793s`, render
`2.3423329985234886ms`; LR `0.11` reaches PSNR `23.796110153198242`, train
`62.273226834000525s`, render `5.030708998674527ms`; LR `0.14` regresses to
PSNR `23.553497791290283`, train `52.348505624999234s`, render
`4.206790999887744ms`. The lower-LR rows buy only about `0.07-0.08` dB over
LR `0.12` while losing much of the speed advantage, and cap-256 quality mode
dominates them if `5ms` render is acceptable. Keep LR `0.12` for the cap-128
speed recipe. The equal-step row stays LR `0.12` too: a 200-step LR `0.11`
check regressed to PSNR `23.128459453582764`, train `53.44634366600076s`,
render `19.63295899986406ms`.
Gate 6cn checks whether the cap-128 default is a seed-5 artifact. It is not, at
least for 128px single-video overfit quality. The 400-step `tile_t=1`, cap-128,
LR `0.12` recipe reaches PSNR `23.78368377685547` at seed `0`,
`23.715169429779053` at seed `5`, and `23.625149726867676` at seed `13`: mean
`23.708000977834065`, span `0.15853404998779297`, population stdev
`0.06491944382000768`. The one-shot render timing samples are noisy across
seeds, so keep the speed claim anchored to the matched seed-5 same-step row and
use the three-seed set only as quality robustness evidence.
Gate 6co adds `--render-benchmark-repeats`, synchronizes MPS around each final
render timing, and reruns the current cap-128 recipe with 20 render repeats. At
equal 200 steps the repeat-timed row reaches PSNR `23.185200691223145`, train
`38.34856141600176s`, render median `8.913395500712795ms`, min
`5.644625001878012ms`, max `25.10791600070661ms`. Against the saved direct
64-splats/frame row, this is `+2.5572967529296875` dB, `17.936632358156274x`
faster to train, and `16.072550577241806x` faster to render by median. At 400
steps the repeat-timed row reaches PSNR `23.745369911193848`, train
`81.65138491700054s`, render median `10.333166999771493ms`, min
`5.961792001471622ms`, max `15.938207998260623ms`. Use these repeat-timed rows
for conservative speed claims; older rows without repeat timing are one-shot
historical samples.
Gate 6cp applies the same synchronized 20-repeat timing to the direct
64-splats/frame paired baseline. The paired artifact
`research_project/benchmarks/results/video_fit_single_overfit_128_16f_200steps_1792uvt_lr012_s0125_t20_tilet1_cap128_64pf_lr032_paired_renderbench20_metal_tile.json`
uses 200 steps for both models, STAR LR `0.12`, direct LR `0.32`, and seed `5`.
STAR reaches PSNR `23.189358711242676`, train `32.827123874998506s`, render
median `7.475833499483997ms`; direct splats reach PSNR `20.627903938293457`,
train `1177.423645084s`, render median `203.9981664984225ms`. The paired
same-step read is therefore `+2.5614547729492188` dB, `35.86740189507551x`
faster training, and `27.287681903630283x` faster median render for STAR.
Gate 6cq retunes tube count under the repeat-timed equal-step setup. 896 tubes
at LR `0.16` reaches PSNR `22.211849689483643`, train `30.988440125001944s`,
render median `7.579749999422347ms`; 1344 tubes at LR `0.14` reaches
`22.728750705718994`, train `17.953880875000323s`, render median
`4.324979001467ms`; 1600 tubes at LR `0.13` reaches `23.01387310028076`, train
`18.926709583000047s`, render median `4.298770498280646ms`; 1728 tubes at LR
`0.125` reaches `23.199284076690674`, train `19.664952208000614s`, render
median `4.426229001182946ms`. Against the paired direct baseline, the 1728-tube
row is `+2.571380138397217` dB, `59.87421848932689x` faster to train, and
`46.08847993267007x` faster to render by median, so use 1728 tubes as the
equal-step speed recipe. The matching 1728-tube 400-step row reaches only PSNR
`23.601515293121338`, `0.14385461807250977` dB below the 1792-tube repeat-timed
400-step row, so keep 1792 tubes for the 400-step cap-128 quality recipe.
Gate 6cr checks whether the 1728-tube equal-step speed recipe is seed-fragile.
It is stable on quality: seed `0` reaches PSNR `23.242146968841553`, seed `5`
reaches `23.199284076690674`, and seed `13` reaches `23.195884227752686`.
The three-seed mean is `23.212438424428303`, span `0.04626274108886719`, and
population stdev `0.021052916687334368`. Timing remains noisy, but the
three-seed mean is train `29.689113777666837s`, median render
`5.934284665878901ms`; versus the paired direct baseline this is
`+2.584534486134846` dB, `39.65843015394074x` faster train, and
`34.37620167960197x` faster median render.
Gate 6cs repeat-times the 1792-tube cap-256 400-step quality mode. It reaches
PSNR `24.085018634796143`, train `38.323750625000685s`, render median
`4.634708000594401ms`, min `4.513209001743235ms`, max `5.108583001856459ms`.
Against the paired direct baseline this is `+3.4571146965026855` dB,
`30.723079705980084x` faster train, and `44.01532231852789x` faster median
render. It also beats the 1792-tube cap-128 400-step repeat-timed quality row by
`0.3396487236022949` dB in this run, so the cap-256 row is the current 128px
single-video quality mode.
Gate 6ct checks whether that cap-256 quality mode is seed-fragile. It is stable
on quality: seed `0` reaches PSNR `24.0570068359375`, seed `5` reaches
`24.085018634796143`, and seed `13` reaches `24.107441902160645`. The three-seed
mean is `24.083155790964764`, span `0.05043506622314453`, and population stdev
`0.020632120857024965`. Timing is noisier: the three-seed mean train time is
`51.418234347000784s` and mean median render is `5.657340167090297ms`; versus
the paired direct baseline this is still `+3.4552518526713065` dB,
`22.89895131633743x` faster train, and `36.059024289385015x` faster median
render.
Gate 6cu moves the same single-video overfit harness to native 256px
`test_video_small.mp4`. The scaled 7168-tube, cap-256 200-step attempt was
interrupted after about 14 minutes without an artifact, so the bounded cap gate
starts at 50 steps. At 50 steps, 7168 tubes with cap `256` reaches PSNR
`22.230050563812256` in `146.55860279099943s`, median render
`12.478374999773223ms`; halving to 3584 tubes is worse and not cheaper, reaching
PSNR `21.195032596588135` in `161.72111600000062s`. The useful boundary is cap
`128`: 7168 tubes, cap `128` reaches PSNR `22.362003326416016` in
`22.07102679200034s`, median render `11.440250000305241ms`, while cap `64`
collapses to PSNR `16.391620635986328`. With cap `128`, the 200-step row reaches
PSNR `24.46974277496338` in `70.66461358399829s`, median render
`18.966646002809284ms`, and the 400-step headroom row reaches PSNR
`25.1381516456604` in `168.99901500000124s`, median render
`26.702728999225656ms`. Use 7168 tubes and cap `128` as the current 256px
single-video recipe; do not use cap `256` at 256px unless a new profiler explains
the fixed backward cost.
Gate 6cv checks whether the 256px 400-step cap-128 recipe is seed-fragile. It is
stable on quality: seed `0` reaches PSNR `25.044105052947998`, seed `5` reaches
`25.1381516456604`, and seed `13` reaches `25.099973678588867`. The three-seed
mean is `25.09407679239909`, span `0.09404659271240234`, and population stdev
`0.038620118679728956`. Timing is noisy on MPS: train times are
`168.99901500000124s`, `267.3640800409994s`, and `236.97501191700212s`; median
render samples are `26.702728999225656ms`, `19.233874998462852ms`, and
`32.02449999844248ms`. Use the three-seed set for quality robustness, not as a
precise speed average.
Gate 6cw adds a video-sampled initializer for the per-frame Gaussian baseline so
the 256px direct-splat check is not only random-init. The tiny CPU smoke
`research_project/benchmarks/results/video_fit_per_frame_video_init_smoke_16_2f_1step.json`
passes and records the new per-frame init fields. On native 256px,
5-step paired direct feasibility is still prohibitive: STAR reaches PSNR
`13.252005577087402` in `1.7836201249992882s`, median render
`6.664707998425001ms`; video-initialized 64-splats/frame reaches PSNR
`6.486777663230896` in `76.71433137500208s`, median render
`779.2045410024002ms`. STAR is `+6.765227913856506` dB, `43.010465232910896x`
faster to train, and `116.91503081403435x` faster to render at this 5-step
feasibility point. Do not spend a full 200-step 256px per-frame run in this
Python baseline harness unless the baseline renderer is replaced or heavily
optimized.

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 32 --max-frames 2 --train-seconds 1 --device cpu \
  --uvt-tubes 16 --splat-count 16 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_smoke
```

Added as the Gate 6a/6b multicam heldout comparison smoke. It loads the same
DeepView goodset config used by the V-JEPA F32 alpha `1/128` baseline row:
train cameras `camera_0006` and `camera_0014`, heldout camera `camera_0005`.
It trains STAR-UVT worldtubes and a free dynamic 3DGS splat baseline for the
same wall-clock budget, writes a JSON report, and saves train/heldout preview
media.

Current smoke result: STAR-UVT heldout PSNR `15.151495933532715`; free dynamic
splats heldout PSNR `4.180948257446289`. This result is intentionally tiny
and should not be used as a quality claim.

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 64 --max-frames 4 --train-seconds 5 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 64 \
  --splat-renderer fast_mac --splat-count 256 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_64_4f_5s
```

Added as the Gate 6c MPS/Metal same-time pilot. Current result: STAR-UVT
heldout PSNR `11.774863243103027`, train PSNR `12.554058074951172`, 42 steps in
`5.014684582973132` seconds, eval render elapsed
`0.18138195801293477` seconds. Free dynamic splats heldout PSNR
`4.894941329956055`, train PSNR `5.227722406387329`, 198 steps in
`5.013484333001543` seconds, eval render elapsed `0.06884004198946059`
seconds.

The current comparison uses the baseline config's pinhole render convention, but
the loaded DeepView bundle reports `pose_source:
deepview_models_relative_opencv_fisheye`. Treat every Gate 6 result as a
research comparison until the fisheye-versus-pinhole parity choice is resolved.

```bash
python3 research_project/benchmarks/multicam_heldout_compare.py \
  --target-size 128 --max-frames 16 --train-seconds 60 --device mps \
  --uvt-render-backend metal_tile --uvt-tubes 256 \
  --splat-renderer fast_mac --splat-count 2048 \
  --out-dir research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_128_16f_60s
```

Added as the Gate 6d fuller pilot. Current result: STAR-UVT heldout PSNR
`10.493327140808105`, train PSNR `10.681523323059082`, 126 steps in
`60.3600161249924` seconds, eval render elapsed `1.1473245000233874` seconds.
Free dynamic splats heldout PSNR `10.865671157836914`, train PSNR
`20.192928314208984`, 2729 steps in `60.00598812501994` seconds, eval render
elapsed `0.8316092919849325` seconds. Both are below the V-JEPA F32 alpha
`1/128` reference heldout PSNR `13.6248`, so this is not a win yet.

Follow-up 128px multicam transfer check exposed UVT worldtube init knobs and
tested 512 tubes with `init_lambda_t=1.0` and `init_opacity=0.7` under the same
60-second-per-model budget. Result: STAR-UVT train PSNR `8.91626262664795`,
heldout PSNR `8.603602409362793`, 123 steps in `60.298264083015965s`, eval
render elapsed `0.7689992079976946s`; direct splats heldout PSNR
`10.857621192932129`. This is worse than the old 256-tube/init pilot, so the
single-video temporal-support gain does not transfer directly to the current
worldtube multicam setup.

Further 128px multicam init ablation found two more negative results. Narrowing
worldtube init support to `init_precision_xy=96` matched the single-video
screen-space support more closely in the anchor projection audit, but train
PSNR fell to `8.877586841583252` and heldout PSNR fell to
`8.753138542175293`. Initializing from both train cameras with the old broad
support (`--uvt-init-views all_train`) reached train PSNR
`10.403272151947021` and heldout PSNR `10.397439002990723`, still below the
old first-view init and below direct splats at heldout PSNR
`10.863123893737793`. Current read: multicam STAR-UVT is underfitting and
needs camera/model parity or optimization changes before a larger 256px run.

DeepView camera projection audit found that the goodset cameras are not
near-pinhole at these render sizes. Compared with the current legacy pinhole
approximation, the `opencv_fisheye` metadata shifts the sampled projection grid
by about `8.06-8.14px` mean and `25.30-25.67px` max at 128px, and by about
`16.28-16.45px` mean and `51.03-51.78px` max at 256px. The V-JEPA reference
config also uses `camera_projection: legacy_pinhole`, so this does not by
itself explain STAR-UVT losing to the local direct splat baseline. It does mean
a clean 256px promotion comparison needs an explicit camera-model decision
before spending the larger budget.

Changing the STAR-UVT multicam loss from one sampled frame to the whole rendered
view sequence is the first major multicam win. With the same 128px/16-frame/60s
contract and the old 256-tube init, `--uvt-loss-scope view_sequence` reached
train PSNR `15.593097686767578` and heldout PSNR `13.423128128051758` in 135
steps, beating direct dynamic splats at heldout PSNR `10.850052833557129`. It
is also close to the 256px V-JEPA F32 reference heldout PSNR `13.6248`, though
that reference used an 18-minute train loop and larger render size. Current
read: sampled-frame loss was wasting the full-sequence STAR render, and
view-sequence loss should be the multicam default for the next 256px pilot.

The first 256px/16-frame/60s view-sequence pilot did not transfer the 128px win
at the same short budget. STAR-UVT reached only 42 steps, train PSNR
`10.676740646362305`, and heldout PSNR `10.409326553344727`; direct splats
reached train PSNR `17.51949119567871` and heldout PSNR
`10.730738639831543` in 2225 steps, while the 256px V-JEPA reference heldout is
`13.6248`. Current read: 256px needs either a longer STAR budget or
rasterizer/training-throughput work before it is a fair promotion run.

A 256px/16-frame temporal-window follow-up reduces the rendered frames per
STAR optimizer step with `--uvt-loss-scope temporal_window` and
`--uvt-window-frames 4`. Under the same 60-second local budget, STAR-UVT
reached 157 steps, train PSNR `12.275136947631836`, heldout PSNR
`11.813445091247559`, and eval render `1.3796198749914765s`. Direct splats
reached train PSNR
`19.916349411010742`, heldout PSNR `10.738123893737793`, and eval render
`0.38876454101409763s` in 2959 steps. This restores the 256px same-time
heldout win over direct splats, but it is still below the V-JEPA F32 heldout
PSNR `13.6248` and renders about `3.55x` slower than the paired `fast_mac`
direct-splat baseline.

The multicam report now also writes synchronized render-only timing fields:
`eval_render_only_elapsed_s`, `eval_render_mean_sequence_s`,
`eval_render_max_sequence_s`, per-train/heldout render-only totals, and
`eval_render_sequence_count`. The CPU smoke
`multicam_heldout_compare_timing_fields_smoke_16_2f_1s` verified those fields
for both STAR-UVT and direct splats.

The initialized-model 256px timing probe does not show an inherent STAR
render-speed loss. With 256 initialized tubes versus 2048 initialized fast-mac
splats at 256px/16f, STAR full projection+render totaled
`0.1808375830296427s` across the three eval sequences, while direct splats
totaled `0.22867970800143667s`. STAR render-only was only
`0.027068290975876153s`, and projection-only was `0.1553649159905035s`. The
initialized STAR Metal stats had pair ratio about `0.83-0.86`, zero overflow,
and max tile count `33-35`.

The trained 256px temporal-window rerun still shows the real speed blocker. With
the same 60-second budget and new timing fields, STAR reached train PSNR
`11.885068416595459`, heldout PSNR `11.320009231567383`, and 159 steps; direct
splats reached train PSNR `20.24591064453125`, heldout PSNR
`10.723106384277344`, and 3138 steps. STAR render-only eval time was
`1.3073187510017306s` across three sequences versus direct splats
`0.4013093340327032s`, about `3.26x` slower. Since the initialized probe is
fast, the next rasterizer gate should capture trained Metal tile stats and
support profiles to explain whether optimization is expanding tube coverage,
triggering fallback, or otherwise increasing tile load.

The trained Metal stats rerun diagnosed the slowdown: the learned STAR model
overflowed almost every 256px UVT tile and made every active tile unstable. The
default temporal-window run had pair ratio `2.98-3.78`, unstable-tile fraction
`1.0`, max tile count `174-222`, and overflow count `8155-8192` out of the
8192 UVT tiles. This means the trained STAR path is no longer exploiting UVT
compactness; it is worse than the sliced per-frame pair count and is stuck in
the unstable/overflow path.

An opt-in precision-floor experiment added `--uvt-min-precision-xy` and
`--uvt-min-lambda-t`. Setting those floors to the old compact init values
(`30.0` and `0.35`) was a negative result: STAR heldout PSNR fell to
`9.554825782775879`, below direct splats at `10.727174758911133`, while
render-only eval was still `1.2437176250386983s` versus splats
`0.3679181660118047s`. The floors reduced some overflow but did not fix the
core issue: pair ratio remained `2.34-4.53`, unstable-tile fraction stayed
`1.0`, and two of three eval views still overflowed many tiles. The next lever
should target depth-order instability and tile-load regularization, not just a
hard lower bound on spatial/temporal precision.

An opt-in velocity regularization experiment added `--uvt-velocity-reg`,
`--uvt-depth-velocity-reg`, and `--uvt-position-reg`. Testing velocity reg
`0.01` plus depth-velocity reg `0.1` was mixed for quality but negative for
speed. STAR heldout PSNR rose to `11.486005783081055`, above direct splats at
`10.724501609802246`, but render-only eval stayed slow at
`1.179607957979897s` versus splats `0.4023879590095021s`. Metal stats still
showed pair ratio `2.50-3.68`, unstable-tile fraction `1.0`, overflow on all
8192 UVT tiles, and max tile count `171-220`. Velocity regularization alone is
therefore not the compactness fix, though it did not hurt heldout quality.

An opt-in projected tile-load regularization experiment added
`--uvt-tile-load-reg`, `--uvt-tile-load-target`, and `--uvt-depth-slope-reg`.
The strong 20-second tile-load setting (`0.02`, target `450`) proved the
mechanism: pair ratio dropped below `1.0`, overflow went to zero, and STAR
render-only became slightly faster than splats, but heldout PSNR fell below the
direct-splat baseline. The softer setting (`--uvt-tile-load-reg 0.005
--uvt-tile-load-target 1500`) was the first good multicam speed/quality point.
Adding a light projected depth-slope penalty improved it further. The previous
best 256px/16f/60s temporal-window recipe was `--uvt-tile-load-reg 0.005
--uvt-tile-load-target 1500 --uvt-depth-slope-reg 0.05`: STAR reached train
PSNR `12.651652812957764`, heldout PSNR `11.877435684204102`, and 237 steps;
direct splats reached train PSNR `19.71877384185791`, heldout PSNR
`10.717645645141602`, and 2827 steps. STAR render-only eval was
`0.2507540419755969s` versus splats `0.26824004197260365s`, with zero overflow,
pair ratio `0.98-1.00`, and max tile count `52-54`. This restores the intended
STAR speed claim for the bounded 256px multicam comparison while keeping the
direct-splat heldout win. The remaining issue is that active tiles are still
mostly order-unstable (`0.93-0.96` unstable fraction), and STAR remains below
the V-JEPA F32 reference heldout PSNR `13.6248`. A stronger depth-slope penalty
(`0.2`) was negative at 60 seconds: heldout fell to `11.32148551940918` and
render-only rose to `0.4422728330246173s`.

A projected depth-margin proxy was added as an opt-in diagnostic
(`--uvt-depth-margin-reg`, `--uvt-depth-margin`) to penalize nearby projected
tubes whose center depths are too close. The first 20-second probe on top of
the current tile-load plus depth-slope recipe (`--uvt-depth-margin-reg 0.01
--uvt-depth-margin 0.05`) was not promoted: STAR heldout PSNR was
`11.522964477539062`, but pair ratio rose to `1.08-1.12` and unstable-tile
fraction stayed `0.97-0.99`. The proxy separated some center depths, but did
not reduce the renderer's unstable path. Keep the current default without
depth-margin regularization.

An explicit tile-shape diagnostic added `--uvt-tile-x`, `--uvt-tile-y`,
`--uvt-tile-t`, and `--uvt-tile-capacity` to the multicam harness. Testing
`--uvt-tile-t 1` on top of the current 256px recipe eliminated unstable Metal
tiles entirely, but it traded away too much heldout quality to become the
default. At 20 seconds, STAR heldout PSNR was `11.10805892944336` with
render-only eval `0.16470333401230164s`, zero unstable tiles, and pair ratio
`1.85-1.88`. At 60 seconds, STAR heldout PSNR was `11.079706192016602` versus
direct splats `10.713973045349121`, with render-only eval `0.1884017909760587s`
versus splats `0.4057717919931747s`, zero unstable tiles, and pair ratio
`1.72-1.75`. This confirms rasterizer tile shape is a useful speed/stability
knob. A relaxed tile-load setting promoted it into the 256px default:
`--uvt-tile-t 1 --uvt-tile-load-reg 0.001 --uvt-tile-load-target 3000
--uvt-depth-slope-reg 0.05`. At 20 seconds, this reached STAR heldout PSNR
`11.720141410827637`, render-only eval `0.18447483397903852s`, zero overflow,
zero unstable tiles, and pair ratio `2.75-2.81`. At 60 seconds, STAR reached
train PSNR `12.820858001708984`, heldout PSNR `12.002521514892578`, and 246
steps; direct splats reached train PSNR `19.738224029541016`, heldout PSNR
`10.725918769836426`, and 2833 steps. STAR render-only eval was
`0.19579870899906382s` versus splats `0.43779041699599475s`, with pair ratio
`2.27-2.56`, max tile count `64-72`, zero overflow, and zero unstable tiles.
This is now the best bounded 256px multicam speed/quality point, though it still
trails the V-JEPA F32 reference heldout PSNR `13.6248`.

Relaxing tile-load target further to `5000` improved the 256px result without
overflow and is the new current default. At 20 seconds, STAR heldout PSNR was
`11.750381469726562`, slightly above the target-`3000` 20-second result
`11.720141410827637`, with zero overflow, max tile count `103-116`, and only a
tiny unstable fraction on one eval view (`0.00055`). At 60 seconds, STAR reached
train PSNR `12.996597290039062`, heldout PSNR `12.157893180847168`, and 254
steps; direct splats reached train PSNR `17.520974159240723`, heldout PSNR
`10.75090217590332`, and 2209 steps. STAR render-only eval was
`0.21316362501238473s` versus splats `0.4603952500037849s`, with pair ratio
`3.33-3.62`, max tile count `98-115`, zero overflow, and zero unstable tiles.
This is the new best bounded 256px multicam speed/quality point, still below
the V-JEPA F32 reference heldout PSNR `13.6248`.

A target-`7000` probe found the tile-capacity boundary. At 20 seconds, STAR
heldout PSNR rose again to `11.859132766723633`, with zero overflow, zero
unstable tiles, and max tile count `114-119`. At 60 seconds, STAR heldout PSNR
rose to `12.210857391357422`, but max tile count reached `123-137` and one eval
view overflowed `499` tiles. Render-only eval stayed fast at
`0.20270062497002073s` versus splats `0.3378250010428019s`, but this crosses
the current stability/capacity guardrail. Keep target `5000` as the default
until tile capacity or binning changes; target `7000` is evidence that quality
still increases with support, but the default rasterizer cap starts failing.

Raising tile capacity to `256` removed that target-`7000` clipping and produced
the new best bounded 256px multicam quality point. With `--uvt-tile-capacity
256`, target `7000`, and the same 60-second budget, STAR reached train PSNR
`12.911274433135986`, heldout PSNR `12.388733863830566`, and 175 steps; direct
splats reached train PSNR `17.89496374130249`, heldout PSNR
`10.748902320861816`, and 2301 steps. STAR render-only eval was
`0.20699129099375568s` versus splats `0.29931937501532957s`, with pair ratio
`3.50-3.82`, max tile count `114-127`, zero overflow, and zero unstable tiles.
The tradeoff is doubled Metal buffer memory (`16.97MB -> 33.75MB`) and fewer
STAR steps than cap-128 target `5000`; use this as the quality default on local
MPS, and keep the memory caveat attached.

A target-`9000` cap-256 probe was negative at the 20-second gate. STAR completed
only 58 steps, heldout PSNR fell to `10.476442337036133`, and train PSNR fell
to `10.994823932647705`. Metal stats still had zero overflow and zero unstable
tiles, with max tile count `119-126`, but the larger support slowed optimization
enough to underfit badly. Do not escalate target `9000`; target `7000` is the
current cap-256 support setting.

A 20-second check of the cap-256 target-`7000` quality default shows that the
cap-256 recipe should not be used as the short-budget gate. At 20 seconds,
cap-256 target `7000` completed only 60 steps and reached heldout PSNR
`10.551692962646484`, versus cap-128 target `7000` at 82 steps and heldout PSNR
`11.859132766723633`. Both had zero overflow and zero unstable tiles. The
cap-256 setting is the 60-second quality default because it fixes cap-128
overflow at longer training, but cap-128 remains the cheaper short-budget probe.

A target-`6000` midpoint did not give a safer improvement. At 20 seconds, STAR
heldout PSNR fell to `11.611959457397461`, below target `5000` at
`11.750381469726562`, while max tile count rose to `111-127`, essentially the
default capacity edge. It had zero overflow and zero unstable tiles, but the
quality and tile-count tradeoff was worse than target `5000`; no 60-second
escalation was run.

A 384-tube capacity probe on the relaxed `tile_t=1` recipe was negative at the
20-second gate, so it was not escalated to a 60-second run. STAR heldout PSNR
fell to `11.04841136932373` from the 256-tube 20-second value
`11.720141410827637`, train PSNR fell to `11.771284103393555`, and render-only
eval slowed to `0.29677454198827036s`. Metal stats stayed stable with zero
overflow and zero unstable tiles, but max tile count rose to `110-123`, close
to the default tile capacity `128`, and pair ratio rose to `3.13-3.26`.
Capacity alone is therefore not the next 256px lever under the current tile
capacity; it underfits per wall-clock and approaches the rasterizer cap.

A 256px `view_sequence` retry on the relaxed `tile_t=1` recipe was also
negative at the 20-second gate. STAR completed only 26 steps, heldout PSNR fell
to `8.97468376159668`, and train PSNR was `9.258035659790039`; the paired
direct-splat baseline reached heldout PSNR `9.148686408996582`. Metal stats
remained healthy with zero overflow and zero unstable tiles, but the full
sequence loss is still too slow at 256px. Keep `--uvt-loss-scope
temporal_window --uvt-window-frames 4` as the 256px default.

An LR `0.05` probe on the current relaxed `tile_t=1` recipe was negative at 20
seconds. STAR heldout PSNR fell to `11.359124183654785` from the LR `0.03`
20-second value `11.720141410827637`, train PSNR fell to
`11.424661636352539`, and pair ratio rose to `3.32-3.43`. The Metal path still
had zero overflow and zero unstable tiles, but the higher LR degraded the fit.
An LR `0.02` probe was also negative: heldout fell to `10.951751708984375`,
train PSNR fell to `11.512050151824951`, and render-only eval was
`0.16975908196764067s`. At this pre-bundled-reducer point, keep
`--uvt-lr 0.03` as the 256px default. The later bundled-reducer retune
supersedes this for full 60-second runs.

A no-depth-slope parity check confirmed that `depth_slope_proxy` is exactly
`0.0` under the current `tile_t=1` pinhole setup because projected
`depth_beta` only has a temporal component and `tile_t=1` has zero temporal
half-extent. The no-slope 20-second report reached heldout PSNR
`11.638251304626465`, close to but slightly below the same-scope report with
the historical `--uvt-depth-slope-reg 0.05` flag (`11.720141410827637`). The
flag is harmless in the current best artifact, but depth-slope tuning is not a
useful `tile_t=1` lever unless the depth model changes.

Train-step timing probe and compact backward patch, 2026-05-11:
`multicam_train_step_timing_probe.py` profiles the current 256px/16-frame
cap-256 recipe without claiming quality. The first profile found STAR-UVT at
`0.3036720311138197s` per train step versus `0.013559588376665488s` for paired
`fast_mac` direct splats. The expensive part was the STAR backward bridge:
`stable_backward_samples` produced a fixed `67108864` sample slots for a
4-frame window, but only `458991` were valid; MPS reductions over the mostly
empty buffer took `0.1513007489265874s`.

The compact-output patch changes the Metal stable-backward kernel to write
gradient samples through a device counter, then slices the returned tensors
before MPS `index_add_`. Backward parity still passes. The same 8-step profile
now shows STAR-UVT at `0.1739891823817743s` per train step, with backward down
to `0.09416389050602447s`. The microbreakdown is `0.007699042034801096s` for
Metal sample generation, `0.0325404170434922s` for reductions over `491831`
compact samples, and `0.0021938749705441296s` for projection VJP. Worldtube
projection forward is now the other large cost at `0.0704665103694424s` per
step. Direct splats in the same rerun averaged `0.01830059887288371s` per
sampled-frame step. The next speed work should target projection and the
remaining compact backward cost; forward raster alone is not the blocker.

Compact-backward 60-second quality rerun, 2026-05-11:

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

Result: compact backward improves the actual same-time quality run but does not
settle the speed claim. STAR completed `294` steps, up from the previous
cap-256 60-second run's `175`, reached train PSNR `13.66211748123169`, and
reached heldout PSNR `12.700817108154297`. Direct fast-mac splats completed
`2254` steps, reached train PSNR `17.441619396209717`, and reached heldout PSNR
`10.724161148071289`. This is a stronger STAR heldout win over direct splats,
but it is still below the V-JEPA F32 heldout reference `13.6248`.

The render-speed read is now negative for this trained support: STAR render-only
eval was `0.30119724897667766s` versus direct splats
`0.24550599994836375s`, and heldout render-only was `0.07833108300110325s`
versus `0.059047749964520335s`. Metal stats were healthy on stability and
capacity, with zero overflow and zero unstable tiles, but pair ratio rose to
`3.78-4.16` and max tile count reached `123-135`. The compact backward patch is
a real trainer-throughput and quality win; the remaining speed work should
control trained support and optimize projection/reduction before rewriting the
forward raster path.

Closed-form projection patch and 60-second quality rerun, 2026-05-11:

`project_world_tubes_pinhole(...)` no longer builds an `[N,2,2]` covariance
with batched matrix multiplications and `torch.linalg.inv`. It now computes the
same 2x2 screen covariance entries and inverse explicitly. A local equivalence
probe against the old formula showed `ma` max diff `0.0`, `q_uvt` max diff
`2.9802322387695312e-08`, and `depth0` max diff `0.0`. A non-identity camera
equivalence probe showed `ma` max diff `1.52587890625e-05`, `q_uvt` max diff
`1.1920928955078125e-07`, and zero `depth0` / `depth_beta` diff. The pinhole
and `CameraSpec` projection smokes still pass.

The timing probe with output
`research_project/benchmarks/results/multicam_train_step_timing_probe_mps_256_16f_projection_closedform.json`
shows STAR mean train step dropped again to `0.10214632287534187s`. Projection
forward fell from `0.0704665103694424s` to `0.0023792186329956166s`; render was
`0.0040640363658894785s`, backward was `0.08975485964037944s`, and direct
splats were `0.019570161501178518s` per sampled-frame step. The backward
microbreakdown is now the dominant STAR cost: sample generation
`0.015150916995480657s`, compact reductions `0.030503582034725696s`, and
projection VJP `0.002249667013529688s`.

The paired 60-second comparison with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_projection_closedform`
completed `410` STAR steps. STAR reached train PSNR `14.423624038696289` and
heldout PSNR `12.778368949890137`; direct fast-mac splats completed `3179`
steps, reached train PSNR `20.28785991668701`, and reached heldout PSNR
`10.700675964355469`. STAR render-only eval was `0.08381970797199756s` versus
direct splats `0.36649591801688075s`; heldout render-only was
`0.021688583015929908s` versus `0.07554204197367653s`. Metal stats stayed
healthy: pair ratio `3.02-3.23`, max tile count `103-119`, zero overflow, and
zero unstable tiles.

Read: this validates the speed thesis much better than the compact-only rerun.
The current 256px STAR point is faster than paired direct splats and much better
on heldout PSNR, but it still trails the V-JEPA F32 heldout reference `13.6248`
by about `0.85` dB. The next experiment should target quality under the now
faster projection path or reduce the remaining compact backward/reduction cost.

Bundled compact reduction and LR retune, 2026-05-11:

The compact-output backward still reduced `ma`, `q_uvt`, opacity, and color in
four separate MPS passes. Bundling those sample gradients into one 13-channel
compact `index_add_` cut the timing probe to STAR mean step
`0.06733408838044852s`, with projection `0.001725447982607875s`, render
`0.0024019792545004748s`, and backward `0.05865076563350158s`. The compact
reduction component dropped from `0.030503582034725696s` to
`0.008818959002383053s`. Forward render is now smaller than direct splats in
the same timing probe (`0.0024019792545004748s` versus
`0.005343109376553912s`); the remaining speed blocker is backward.

The faster path changes the LR boundary. LR `0.03` collapsed by step `180`.
LR `0.02` looked better than LR `0.01` at 20 seconds, but the 60-second
escalation had finite loss with `tile_load_proxy: NaN` at step `190`, first
NaN loss at step `200`, final heldout PSNR `7.12491512298584`, and zero active
tile pairs. The midpoint LR `0.015` stayed stable for 60 seconds but did not
improve quality: STAR completed `879` steps, reached heldout PSNR
`13.005823135375977`, and had slightly larger pair load than LR `0.01`
(`1.80-2.30` pair ratio, max tile count `76-83`). LR `0.01` remains the
current stable full-run setting. Relaxing the tile-load target from `7000` to
`9000` at LR `0.01` was also negative: it stayed stable and improved train
PSNR to `17.124666213989258`, but heldout fell to `12.860218048095703` with
`790` steps and no overflow. The missing V-JEPA gap is therefore not solved by
simple LR midpointing or raw support relaxation. The paired 60-second
comparison with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001`
completed `849` STAR steps, reached train PSNR `16.730055809020996`, and
reached heldout PSNR `13.20147705078125`; direct splats completed `2568` steps,
reached train PSNR `19.02044677734375`, and reached heldout PSNR
`10.722965240478516`. STAR render-only eval was `0.046138084086123854s` versus
direct splats `0.29644233302678913s`; heldout render-only was
`0.014686542039271444s` versus `0.08152529201470315s`. Metal stats stayed
healthy: pair ratio `1.59-2.08`, max tile count `65-70`, zero overflow, and
zero unstable tiles. This is the current best legacy-pinhole local 60-second
artifact and is about `0.42` dB below the V-JEPA F32 heldout reference.

Dataset-lens STAR projection diagnostic, 2026-05-11:

DeepView goodset cameras are `opencv_fisheye`; the camera audit measured the
legacy pinhole approximation at about `16.28-16.45px` mean grid shift and about
`51px` max shift at 256px. The multicam harness now has an opt-in
`--uvt-camera-projection dataset_lens` mode that projects STAR worldtubes
through Dynaworld `CameraSpec`/`project_points_camera` and uses the returned
pixel Jacobian for local screen-time covariance and velocity. The default
remains `legacy_pinhole` for the existing apples-to-apples rows.

The 20-second gate was positive: dataset-lens STAR reached heldout PSNR
`13.381278991699219` versus the legacy 20-second STAR row
`12.534915924072266`. The full 60-second run with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_uvt256_dataset_lens_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001`
completed `845` STAR steps, reached train PSNR `16.58480167388916`, and
reached heldout PSNR `13.496740341186523`; paired direct splats completed
`2789` steps, reached train PSNR `19.735459327697754`, and reached heldout PSNR
`10.734033584594727`. STAR render-only eval was `0.0413991259993054s` versus
direct splats `0.4126907510217279s`; heldout render-only was
`0.01195199997164309s` versus `0.074937375029549s`. Metal stats stayed clean:
pair ratio `2.10-2.18`, max tile count `70-78`, zero overflow, and zero
unstable tiles.

Read: dataset-lens projection improves quality without invalidating the speed
story. The trained model has more tile pressure than the legacy-pinhole row,
but the measured 60-second render-only time is not worse than the previous
legacy best. STAR remains about `0.128059658813476` dB below the V-JEPA F32
heldout reference `13.6248`. The next question is camera-contract parity and
quality, not a generic forward-raster rewrite.

Lens-aware direct-splat baseline and V-JEPA crossing, 2026-05-11:

The harness now has a separate `--splat-camera-projection dataset_lens` flag.
For direct splats this creates `CameraSpec` values with the DeepView
`opencv_fisheye` model and lets Dynaworld's Gaussian renderer use
`render.camera_projection='camera_model'` before handing projected Gaussians to
fast-mac. This keeps legacy rows stable while enabling a fairer camera-contract
comparison.

The 20-second gate was already strong: STAR heldout PSNR
`13.600945472717285`, direct splats `8.922689437866211`, STAR render-only eval
`0.043396833061706275s`, direct splats `0.5898302079876885s`.

The 60-second run with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001`
completed `886` STAR steps, reached train PSNR `16.240556716918945`, and
reached heldout PSNR `13.632997512817383`; lens-aware direct splats completed
`2438` steps, reached train PSNR `17.265151500701904`, and reached heldout
PSNR `11.188531875610352`. STAR render-only eval was
`0.04134708392666653s` versus direct splats `0.4395427079871297s`; heldout
render-only was `0.012246291968040168s` versus `0.07924108300358057s`. Metal
stats stayed clean: pair ratio `2.21-2.33`, max tile count `68-74`, zero
overflow, and zero unstable tiles.

Read: this is the cleanest current local result. It beats direct splats by
`2.4444656372070312` heldout PSNR and about `10.6x` render-only eval speed,
and it clears the V-JEPA F32 heldout reference by `0.00819751281738238` dB.
Because the margin over V-JEPA is tiny and V-JEPA had an 18-minute training
budget, this is not a promotion claim by itself. It does say the STAR speed
thesis is alive and the next work should repeat/scale the lens-aware row, not
start with a forward-raster rewrite.

Seed-1 repeat:

The repeat run with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed1_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001`
did not reproduce the V-JEPA crossing. STAR completed `790` steps, reached
train PSNR `16.289902687072754`, and reached heldout PSNR
`12.9697904586792`; lens-aware direct splats completed `2532` steps, reached
train PSNR `18.113491535186768`, and reached heldout PSNR
`11.243672370910645`. STAR render-only eval was `0.04029629105934873s` versus
direct splats `0.5776280419668183s`. Metal stats were still clean: pair ratio
`2.13-2.37`, max tile count `69-73`, zero overflow, and zero unstable tiles.

Read: the speed thesis still holds, but the V-JEPA crossing is not robust
across seeds yet. The next quality gate should address initialization or
repeatability before claiming V-JEPA parity.

Deterministic grid-init repeat:

The harness now has `--uvt-init-sampling random|grid`. The `grid` mode samples
initialization pixels from a deterministic image grid instead of random pixels
from the selected initialization views.

The seed-1 grid run with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed1_gridinit_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001`
completed `932` STAR steps, reached train PSNR `16.25625467300415`, and
reached heldout PSNR `13.179410934448242`; lens-aware direct splats completed
`2405` steps, reached train PSNR `17.340192794799805`, and reached heldout PSNR
`11.206615447998047`. STAR render-only eval was `0.035952959035057575s`
versus direct splats `0.5049242499517277s`; heldout render-only was
`0.01105145801557228s` versus `0.08374808396911249s`. Metal stats stayed clean:
pair ratio `2.22-2.38`, max tile count `71-75`, zero overflow, and zero
unstable tiles.

Read: grid init improves the seed-1 miss by about `0.20962047576904297` dB
over random init (`13.179410934448242` versus `12.9697904586792`) and keeps the
render-speed win. It is not enough to reproduce V-JEPA parity. The next quality
gate should not be plain deterministic pixel coverage alone; test a more
deterministic training schedule or a stronger multi-view/motion-aware init.

All-train grid-init escalation:

The 20-second all-train grid gate with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_both_dataset_lens_seed1_alltrain_gridinit_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001`
reached STAR heldout PSNR `13.527872085571289` versus direct splats
`9.138171195983887`; STAR render-only eval was `0.04275804205099121s` versus
direct splats `0.3338368329568766s`. That was strong enough to escalate.

The 60-second all-train grid run with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_60s_both_dataset_lens_seed1_alltrain_gridinit_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001`
completed `851` STAR steps, reached train PSNR `15.971568584442139`, and
reached heldout PSNR `13.52819538116455`; lens-aware direct splats completed
`2092` steps, reached train PSNR `16.335904121398926`, and reached heldout PSNR
`11.074682235717773`. STAR render-only eval was `0.11616870801663026s` versus
direct splats `0.5125780410016887s`; heldout render-only was
`0.034273458004463464s` versus `0.09727545798523352s`. Metal stats stayed
clean: pair ratio `2.42-2.52`, max tile count `78-81`, zero overflow, and zero
unstable tiles.

Read: multi-view grid init is the best seed-1 repeatability lever so far,
recovering about `0.5584049224853516` dB over seed-1 random first-view init and
about `0.3487844467163086` dB over seed-1 first-view grid init. It still misses
the V-JEPA F32 heldout reference by about `0.09660461883544993` dB, and it is
slower to render than first-view grid because the learned support is larger.
This points to better multi-view initialization or deterministic train scheduling
as the next quality lever, not a forward-raster rewrite.

Deterministic train-schedule probe:

The harness now has `--uvt-train-schedule random|cycle`. The `cycle` mode
keeps STAR initialization unchanged but cycles train views and temporal-window
starts deterministically instead of sampling both randomly.

The tiny CPU smoke with output
`research_project/benchmarks/results/multicam_heldout_compare_cycle_schedule_smoke_16_2f_1s`
passed and wrote `train_schedule: cycle` into the STAR report.

The 20-second all-train grid cycle run with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_both_dataset_lens_seed1_alltrain_gridinit_cycle_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001`
completed `312` STAR steps, reached train PSNR `15.107933044433594`, and
reached heldout PSNR `13.28015422821045`; lens-aware direct splats completed
`960` steps and reached heldout PSNR `9.227516174316406`. STAR render-only
eval was `0.038111125002615154s` versus direct splats `0.3976892919745296s`.
Metal stats stayed clean: pair ratio `2.47-2.65`, max tile count `74-79`, zero
overflow, and zero unstable tiles.

Read: deterministic cycling is negative versus the same 20-second all-train
grid run with the random train schedule (`13.28015422821045` versus
`13.527872085571289`). Do not escalate cycle scheduling to 60 seconds as-is.

All-train grid LR `0.015` probe:

The 20-second all-train grid LR `0.015` run with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_both_dataset_lens_seed1_alltrain_gridinit_lr0015_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle`
completed `310` STAR steps, reached train PSNR `15.169015884399414`, and
reached heldout PSNR `13.287956237792969`; lens-aware direct splats completed
`717` steps and reached heldout PSNR `8.6444730758667`. STAR render-only eval
was `0.04351187701104209s` versus direct splats `0.6308000839781016s`. Metal
stats stayed clean: pair ratio `2.64-2.74`, max tile count `77-82`, zero
overflow, and zero unstable tiles.

Read: LR `0.015` is negative versus LR `0.01` for the same 20-second all-train
grid setting (`13.287956237792969` versus `13.527872085571289`). Do not
escalate it to 60 seconds.

All-train grid tile-load target `5000` probe:

The 20-second all-train grid target-`5000` run with output
`research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_both_dataset_lens_seed1_alltrain_gridinit_temporal_window4_tileload0001_target5000_depthslope005_tilet1_cap256_compact_bundle_lr001`
completed `308` STAR steps, reached train PSNR `15.119296073913574`, and
reached heldout PSNR `13.459638595581055`; lens-aware direct splats completed
`960` steps and reached heldout PSNR `9.227531433105469`. STAR render-only
eval was `0.0383758339448832s` versus direct splats `0.28163870907155797s`.
Metal stats stayed clean and more compact than target `7000`: pair ratio
`2.26-2.36`, max tile count `74-75`, zero overflow, and zero unstable tiles.

Read: target `5000` improves compactness but loses quality versus target `7000`
for the same 20-second all-train grid setting (`13.459638595581055` versus
`13.527872085571289`). Keep target `7000` as the quality setting.

## Promotion Decision

2026-05-11 decision: keep STAR-UVT isolated in
`variants/star_uvt_v0/`. Do not wire it into the production GFlow/FasterGS MVP
path yet.

The research lane now has forward parity, projection smokes, backward parity
smokes, bounded local backward timing, synthetic training comparison,
fixture-video contact sheet proof, and a same-time multicam heldout comparison
harness. The original 128px/16-frame pilot and follow-up init ablations did not
beat direct splats, but view-sequence loss now gives STAR-UVT a heldout win over
direct splats at the same 60-second local budget, and temporal-window training
extends that direct-splat heldout win to 256px. A projected tile-load proxy now
also gives one bounded 256px setting that is both higher heldout PSNR and faster
render-only than the paired `fast_mac` direct-splat baseline before the compact
backward rerun. The compact rerun improved steps and heldout quality but gave
up the render-only speed win because the trained support grew. The closed-form
projection patch then restored the speed win and lifted STAR to heldout PSNR
`12.778368949890137`, but the larger comparison is still blocked by two issues:
STAR-UVT remains below the V-JEPA F32 reference and fisheye-vs-pinhole camera
parity is unresolved. The initialized timing probe showed the renderer itself
can be fast; the default learned model destroyed UVT compactness by overflowing
tiles and making active tiles unstable. The simple precision-floor experiment
did not repair that, and a high velocity/depth-velocity penalty also left all
active tiles unstable and overflowing. The tile-load plus light depth-slope
proxy fixes overflow, and the relaxed `tile_t=1` tile-shape recipe now removes
the unstable path while improving heldout PSNR over the previous `tile_t=2`
best. The bundled-reducer LR `0.01` run lifts the current local 60-second
heldout PSNR to `13.20147705078125`, still below the V-JEPA F32 reference. The
dataset-lens STAR diagnostic lifts the same 60-second recipe to heldout PSNR
`13.496740341186523` and preserves the render-speed win, but it is an explicit
camera-model diagnostic rather than a replacement for the legacy-pinhole V-JEPA
comparison row. The lens-aware direct-splat comparison then reaches STAR
heldout PSNR `13.632997512817383`, direct-splat heldout PSNR
`11.188531875610352`, and STAR render-only eval `0.04134708392666653s` versus
direct splats `0.4395427079871297s`. That is the first local STAR heldout
crossing over the V-JEPA F32 row, but the margin is only about `0.0082` dB and
the run is still a 256px/16-frame local harness result. A seed-1 repeat under
the same lens-aware contract fell to heldout PSNR `12.9697904586792`; grid init
improved seed 1 to `13.179410934448242`, and all-train grid init improved it to
`13.52819538116455`, but still missed V-JEPA parity. Time-distributed
all-frames init fixes that at the 15-20 second gate, and checkpoint diagnostics
now show the 60-second seed-1 run contains a selected step-300 peak at heldout
PSNR `13.75400447845459` before decaying to final heldout PSNR
`13.354101181030273`. Repeatability is now primarily an early-stop/schedule
blocker, not a forward-rasterizer blocker. The early 256px deterministic cycle
schedule was negative at the 20-second gate. The later 512px 320-tube window-1
revisit is positive on seeds 0 and 1, with non-heldout selected heldout PSNR
`13.798948287963867` and `13.915006637573242` and render-only eval
`0.109-0.113s` across the three eval sequences, but seed 2 rejects cycle with a
non-finite LR `0.01` run and an underfit LR `0.005` stability bracket. The
single-video overfit lane shows a clear local speed win: at 64px UVT beats the
local per-frame splat baseline by PSNR in less than half the train time, and
the 128px same-step comparison now favors UVT too. The first paired
128px/16f/200-step row reached UVT PSNR `22.31398344039917`, and the follow-up
temporal-support bracket improved the UVT-only equal-step row to PSNR
`23.130309581756592` at temporal precision `2.0`, train
`17.787110125000254s`, render `1.5611250000802102ms`. A 400-step STAR-only
headroom check on the same `t=2.0` recipe improves local overfit quality to
PSNR `23.569955825805664` in `31.383009374996618s`, render
`1.2202500001876615ms`. The saved
64-splats/frame 200-step baseline reached PSNR `20.627903938293457` in
`687.8440475830003s`, render `143.26100000016595ms`. It still lacks the
evidence required for promotion: a
full-resolution same-split heldout win against the strong V-JEPA row,
camera-model parity, and production-scale integration.
