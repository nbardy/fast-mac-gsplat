# 512px View-Shuffled Cycle Report

Date: 2026-05-12

## Purpose

This note freezes the current 512px STAR-UVT scale read into one inspectable
artifact. It answers the narrow question: under the current local DeepView
goodset split, does the best 512px STAR recipe now clear the V-JEPA F32 heldout
reference robustly enough to guide the next experiment?

## Inputs

Common setup:

- target size: `512`
- frames: `16`
- train cameras: `camera_0006`, `camera_0014`
- heldout camera: `camera_0005`
- pose/lens source: `deepview_models_relative_opencv_fisheye`
- STAR camera projection: `dataset_lens`
- STAR tubes: `320`
- STAR loss scope: temporal window, `--uvt-window-frames 1`
- tile controls: `--uvt-tile-t 1 --uvt-tile-capacity 256`
- support controls: `--uvt-tile-load-reg 0.001 --uvt-tile-load-target 7000`
- depth control: `--uvt-depth-slope-reg 0.05`
- schedule: `view_shuffled_cycle`
- reporting selector: `best_train_psnr`
- heldout used for selection: no

The V-JEPA reference remains the existing F32 alpha `1/128` baseline row:
train PSNR `19.4875`, heldout PSNR `13.6248`, wall clock `18m00s` train loop
and `18m22s` W&B runtime, from `dynaworld/BASELINES.md`.

Important caveat: the three current `view_shuffled_cycle` rows are STAR-only
artifacts with `skip_splats: true`. The paired 512px direct-splat reference is
from earlier window-1 rows under the same dataset/lens contract, not rerun inside
these three STAR-only directories.

## Current STAR Rows

| seed | artifact | selected step | train PSNR | heldout PSNR | elapsed s | render-only s | heldout-render-only s | max pair ratio | max tile | overflow | unstable |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `mcam512_s0_t320_view_shuffled_cycle_fixed600_gain` | 600 | `15.180933952331543` | `13.639522552490234` | `65.4390769160018` | `0.1782556249963818` | `0.0736958749985206` | `3.1756918117808755` | `96` | `0` | `0.0` |
| 1 | `mcam512_s1_t320_view_shuffled_cycle_fixed600_gain` | 600 | `15.310863018035889` | `13.812097549438477` | `74.23046191699905` | `0.1593077910001739` | `0.05322945800071466` | `2.990017481602631` | `95` | `0` | `0.0` |
| 2 | `mcam512_s2_t320_view_shuffled_cycle_fixed600_gain` | 600 | `15.333081245422363` | `13.793721199035645` | `64.00552208299996` | `0.1350984589989821` | `0.040353208998567425` | `3.1764845728748274` | `94` | `0` | `0.0` |

Aggregate selected heldout PSNR:

- mean: `13.748447100321451`
- min: `13.639522552490234`
- max: `13.812097549438477`
- span: `0.172574996948242`
- margin above V-JEPA F32 heldout reference at the weakest seed: `+0.014722552490234`

Aggregate selected render-only time:

- mean: `0.157553958331846s`
- min: `0.1350984589989821s`
- max: `0.1782556249963818s`

## Paired Seed-0 Rerun

After this summary was created, the weakest seed was rerun with paired direct
splats enabled and `--uvt-select-checkpoint best_train_psnr`:

```text
out dir:
  research_project/benchmarks/results/mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct

STAR selected:
  selector            best_train_psnr
  uses heldout        false
  step                600
  train PSNR          15.349876880645752
  heldout PSNR        13.597569465637207
  render-only         0.11119404100463726s
  overflow/unstable   0 / 0.0

STAR heldout-best in same curve:
  step                400
  heldout PSNR        13.696317672729492

Direct splats:
  renderer            fast_mac
  camera projection   dataset_lens
  steps               600
  train loop          20.043970749997243s
  heldout PSNR        8.24771499633789
  render-only         1.2847208340026555s
```

This is a useful negative result. It confirms the same-run direct baseline is
far below STAR, but it also shows `best_train_psnr` is not a locked selector:
on this live rerun, the selected STAR checkpoint misses the V-JEPA F32 heldout
reference `13.6248` by about `0.0272` dB, while the heldout-best checkpoint in
the same curve clears it. The earlier three-seed table should therefore be read
as saved-curve evidence, not a solved robust-floor claim.

## Balanced Selector Rerun

A post-hoc train-plateau plus train-view-gap selector looked promising on the
first paired seed-0 curve, but a live rerun rejected it:

```text
out dir:
  research_project/benchmarks/results/mcam512_s0_t320_view_shuffled_cycle_fixed600_balanced_d03_gap165_paired_direct

STAR selected:
  selector            first_balanced_train_psnr_plateau
  uses heldout        false
  step                400
  train PSNR          14.490061283111572
  heldout PSNR        13.495000839233398
  render-only         0.12350416600384051s
  train-view gap      1.401881217956543
  next train gain     0.19243812561035156
  overflow/unstable   0 / 0.0

STAR heldout-best in same curve:
  step                600
  heldout PSNR        13.591666221618652

Direct splats:
  renderer            fast_mac
  camera projection   dataset_lens
  steps               600
  train loop          11.901573375005682s
  heldout PSNR        8.247457504272461
  render-only         0.2415786669953377s
```

This rejects the balanced selector as the next reporting rule. It selected the
intended early shoulder, but the live curve shifted down: the selected checkpoint
misses V-JEPA by about `0.1298` dB, and even the heldout-best checkpoint misses
by about `0.0331` dB. The direct-splat gap remains large, so the blocker is now
curve reproducibility/variance plus selector robustness, not same-run direct
comparison.

## STAR-Only Replicates

To isolate whether paired direct-splat work was changing STAR behavior, the same
seed-0 `best_train_psnr` command was rerun twice with `--skip-splats`:

```text
out dir:
  research_project/benchmarks/results/mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_staronly_rerun1

STAR selected:
  selector            best_train_psnr
  uses heldout        false
  step                600
  train PSNR          15.153077602386475
  heldout PSNR        13.886017799377441
  render-only         0.10877087500557536s
  overflow/unstable   0 / 0.0

STAR heldout-best in same curve:
  step                600
  heldout PSNR        13.886017799377441

out dir:
  research_project/benchmarks/results/mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_staronly_rerun2

STAR selected:
  selector            best_train_psnr
  uses heldout        false
  step                600
  train PSNR          15.127953052520752
  heldout PSNR        13.815839767456055
  render-only         0.1103979579929728s
  overflow/unstable   0 / 0.0

STAR heldout-best in same curve:
  step                600
  heldout PSNR        13.815839767456055
```

Both STAR-only best-train reruns are positive. They clear the V-JEPA F32
reference by about `0.2612` dB and `0.1910` dB. The script trains and evaluates
STAR before entering the direct-splat branch, so the paired negative rows are not
explained by direct-splat training happening before STAR in the same process.

## Paired Best-Train Rerun

After the STAR-only replicates, the paired direct-splat best-train run was rerun:

```text
out dir:
  research_project/benchmarks/results/mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct_rerun2

STAR selected:
  selector            best_train_psnr
  uses heldout        false
  step                600
  train PSNR          15.380860328674316
  heldout PSNR        13.730086326599121
  render-only         0.11077054099587258s
  overflow/unstable   0 / 0.0

STAR heldout-best in same curve:
  step                600
  heldout PSNR        13.730086326599121

Direct splats:
  renderer            fast_mac
  camera projection   dataset_lens
  steps               600
  train loop          17.158791125002608s
  heldout PSNR        8.247519493103027
  render-only         0.30742666700825794s
```

This paired rerun also clears V-JEPA and keeps the direct-splat baseline near the
prior `8.247` heldout PSNR rows. The more accurate read is not "paired direct
breaks STAR." It is that the same nominal STAR recipe has meaningful MPS/live
run spread, while the direct-splat baseline is stable and far below STAR.

Current seed-0 selected heldout spread:

| artifact | skip splats | selector | selected step | selected heldout | heldout-best | direct heldout |
|---|---:|---|---:|---:|---:|---:|
| `mcam512_s0_t320_view_shuffled_cycle_fixed600_gain` | yes | `first_train_psnr_gain_drop` | 600 | `13.639522552490234` | `13.639522552490234` | n/a |
| `mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct` | no | `best_train_psnr` | 600 | `13.597569465637207` | `13.696317672729492` | `8.24771499633789` |
| `mcam512_s0_t320_view_shuffled_cycle_fixed600_balanced_d03_gap165_paired_direct` | no | `first_balanced_train_psnr_plateau` | 400 | `13.495000839233398` | `13.591666221618652` | `8.247457504272461` |
| `mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_staronly_rerun1` | yes | `best_train_psnr` | 600 | `13.886017799377441` | `13.886017799377441` | n/a |
| `mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_staronly_rerun2` | yes | `best_train_psnr` | 600 | `13.815839767456055` | `13.815839767456055` | n/a |
| `mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct_rerun2` | no | `best_train_psnr` | 600 | `13.730086326599121` | `13.730086326599121` | `8.247519493103027` |

Across the four `best_train_psnr` rows, selected heldout spans
`13.597569465637207` to `13.886017799377441` (`0.2884483337402344` dB). Three
of those four clear V-JEPA; the one miss is only `0.0272` dB below. This is
promising but not robust enough to call the 512px recipe locked.

## Paired Three-Seed Matrix

The current paired best-train family was then completed for seeds 1 and 2, with a
second seed-2 repeat because the first seed-2 row missed V-JEPA:

| artifact | seed | STAR selected heldout | STAR heldout-best | direct heldout | STAR render-only s | direct render-only s | max pair ratio | max tile | overflow | unstable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct_rerun2` | 0 | `13.730086326599121` | `13.730086326599121` | `8.247519493103027` | `0.11077054099587258` | `0.30742666700825794` | `2.9228518191265813` | `89` | `0` | `0.0` |
| `mcam512_s1_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct` | 1 | `13.756308555603027` | `13.756308555603027` | `8.256725311279297` | `0.13106466799217742` | `0.3415996670082677` | `3.13789110175505` | `102` | `0` | `0.0` |
| `mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct` | 2 | `13.593206405639648` | `13.593206405639648` | `8.26798152923584` | `0.14224566600751132` | `0.3077655410015723` | `3.2401051693473404` | `90` | `0` | `0.0` |
| `mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct_rerun2` | 2 | `13.608675003051758` | `13.608675003051758` | `8.267892837524414` | `0.21863812499213964` | `0.34421083399502095` | `3.205300641697287` | `96` | `0` | `0.0` |
| `mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct_envcapture_rerun3` | 2 | `13.54917049407959` | `13.54917049407959` | `8.267990112304688` | `0.1580982910018065` | `2.203050998992694` | `3.092638929134037` | `94` | `0` | `0.0` |

Primary three-seed selected heldout PSNR is mean `13.6932004292806`, min
`13.593206405639648`, span `0.1631021499633789`. Against V-JEPA F32 heldout
`13.6248`, seed 0 is `+0.10528632659912063`, seed 1 is
`+0.13150855560302688`, and seed 2 is `-0.03159359436035203`. The seed-2 repeat
is also below V-JEPA by `0.016124996948242654`; the env-captured seed-2 repeat is
below by `0.07562950592040968`. Direct splat quality is stable and far behind;
the latest env-captured direct row is `8.267990112304688` heldout PSNR. Its
render timing is much slower than the earlier direct rows, so do not use that row
as a speed summary.

This is the current scale read: STAR is decisively better than direct splats, but
the paired 512px view-shuffled recipe is still not a robust V-JEPA replacement
because seed 2 misses three paired rows.

## Seed-2 STAR-Only Check

The next isolation run repeated seed 2 with the same `best_train_psnr` reporting
rule but `--skip-splats`:

```text
out dir:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_staronly_rerun1

STAR selected:
  selector            best_train_psnr
  uses heldout        false
  step                600
  train PSNR          15.40644359588623
  heldout PSNR        13.788966178894043
  render-only         0.16255258299497655s
  overflow/unstable   0 / 0.0

STAR heldout-best in same curve:
  step                600
  heldout PSNR        13.788966178894043
```

This clears the V-JEPA F32 reference by `0.1641661788940425` dB. The three
current seed-2 best-train rows now span `13.593206405639648` to
`13.788966178894043` (`0.19575977325439453` dB): two paired rows miss V-JEPA,
while the STAR-only row clears it. Because the runner trains STAR before the
direct-splat branch, this is still best read as live MPS/run variance under a
fragile recipe, not as direct-splat training causally degrading STAR.

## Controlled Seed-2 A/B

I added run metadata capture to `multicam_heldout_compare.py`: new reports now
record `argv`, `cwd`, selected environment variables, PyTorch version, MPS/CUDA
availability, and whether deterministic algorithms are enabled. The CPU smoke
`multicam_heldout_compare_deterministic_metadata_smoke_16_2f_1s` passed with
`--torch-deterministic warn` and confirmed those fields are written.

The first real MPS audit tried the paired seed-2 command with
`--torch-deterministic warn`, `PYTHONHASHSEED=0`, and
`PYTORCH_ENABLE_MPS_FALLBACK=0`. That is not a practical control for this path:
after more than four minutes it had produced only `run_meta.json` and no
`comparison_report.json`, so it was stopped. Treat the partial directory
`mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct_deterministic_warn_rerun3`
as a metadata/audit artifact, not a benchmark row.

The practical controlled launch kept `PYTHONHASHSEED=0` and
`PYTORCH_ENABLE_MPS_FALLBACK=0`, but left PyTorch deterministic algorithms off.
It produced a third paired seed-2 miss:

```text
out dir:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_paired_direct_envcapture_rerun3

STAR selected:
  selector            best_train_psnr
  uses heldout        false
  step                600
  train PSNR          15.214183807373047
  heldout PSNR        13.54917049407959
  render-only         0.1580982910018065s
  overflow/unstable   0 / 0.0

Direct splats:
  heldout PSNR        8.267990112304688
  render-only         2.203050998992694s
```

The matching controlled STAR-only launch clears V-JEPA again:

```text
out dir:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_staronly_envcapture_rerun2

STAR selected:
  selector            best_train_psnr
  uses heldout        false
  step                600
  train PSNR          15.27522897720337
  heldout PSNR        13.783736228942871
  render-only         0.15025695900112623s
  overflow/unstable   0 / 0.0
```

Read: the same launch controls still split paired seed 2 (`13.54917049407959`)
from STAR-only seed 2 (`13.783736228942871`). Since STAR trains and evaluates
before the direct-splat branch, this still should not be read as direct-splat
training corrupting STAR. It is stronger evidence that the current MPS training
path or recipe is fragile enough that one report row is not a safe selector.

## Same-Process Repeatability Probe

To remove the direct-splat branch entirely, I added
`research_project/benchmarks/multicam_star_repeatability_probe.py`. It runs the
same STAR training call more than once in one Python process with the same seed,
then records selected/final metrics, state digests, and parameter deltas.

The tiny CPU smoke is exactly repeatable:

```text
out json:
  research_project/benchmarks/results/multicam_star_repeatability_probe_cpu_smoke_16_2f_2steps.json

selected heldout span   0.0
selected train span     0.0
final state max delta   0.0
```

The real seed-2 MPS repeatability probe is not exact:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_repeatability_envcapture_2x.json

launch controls:
  PYTHONHASHSEED               0
  PYTORCH_ENABLE_MPS_FALLBACK  0
  torch deterministic mode     off

Repeat 1:
  selected step       600
  selected train      15.260942935943604
  selected heldout    13.818974494934082
  render-only         0.16086037399509223s
  state digest        1338c3f5308a173f3eb3ffa7b96aa320f62f79551c48590cdf1c38783843af43

Repeat 2:
  selected step       600
  selected train      15.102227687835693
  selected heldout    13.716343879699707
  render-only         0.16524062499956926s
  state digest        5d112ccfa49fd72cb90e3691034789481e9ae651fa9ffc44fd71f4a16f1b1f0f

Deltas:
  selected heldout span  0.102630615234375
  selected train span    0.15871524810791016
  final state max abs    1.6650149822235107
  final state mean abs   0.12425407596996853
```

Both repeats clear the V-JEPA reference and have zero overflow/unstable tiles.
That means the branch is capable, but not exact-repeatable on the current MPS
training path. This directly explains why a single paired or STAR-only report row
is not a sufficient selector. The next implementation question is no longer
"does direct-splat training change STAR?" It is "which MPS/custom backward or
reduction operation creates this state drift, and do we need a deterministic
diagnostic path or a more stable backward?"

## Fixed-Step Gradient Repeatability Probe

I then added `research_project/benchmarks/uvt_gradient_repeatability_probe.py` to
localize the drift before optimizer amplification. For a fixed model state and a
fixed training window it repeats three phases:

1. `stable_backward_samples(...)` with identical projected tensors and identical
   `grad_output`;
2. `_reduce_sample_bundle(...)`, which uses MPS `index_add_`, on one fixed sample
   bundle;
3. full autograd gradient from the same model state and fixed loss.

The tiny MPS smoke already shows non-bitwise repeatability while preserving the
same scalar loss:

```text
out json:
  research_project/benchmarks/results/uvt_gradient_repeatability_probe_mps_smoke_16_2f_t16.json

sample digests       3 unique
reduction digests    3 unique
autograd digests     3 unique
loss span            0.0
sample max delta     12.0
reduction max delta  1.1920928955078125e-07
autograd max delta   1.7462298274040222e-10
```

The real 512px seed-2 fixed-window artifact shows the same pattern at useful
scale:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_repeatability_step0_envcapture.json

fixed window:
  view        0
  frame start 0
  loss        0.3558986485004425

sample generation:
  sample rows       484697
  digests           3 unique
  ids different     450118 / 447827 vs first
  max id delta      215
  max grad_q delta  199.8830108642578 / 186.3970184326172

fixed-bundle reduction:
  digests           3 unique
  max grad_q delta  0.21875 / 0.3125

full autograd:
  digests           3 unique
  loss span         0.0
  max grad delta    4.423782229423523e-09 / 3.958120942115784e-09
```

Read: the sample stream is not stable in position-wise row order/content, and
even reducing a fixed sample bundle through MPS `index_add_` is not bitwise
stable. One-step
parameter gradient differences are tiny, but hundreds of Adam steps are enough
to amplify that into different STAR states and different heldout PSNR. This
points to a deterministic diagnostic reduction path or kernel-side ordering fix
as the next engineering target.

## Sorted-CPU Reduction Diagnostic

I added an opt-in `sorted_cpu` reduction mode to
`research_project/trainer_harness/tile_metal_autograd.py` and exposed it in
`uvt_gradient_repeatability_probe.py`. This does not change the default trainer
path: default STAR training still uses MPS `index_add_`. The new path copies a
sample bundle to CPU, sorts by tube id, and reduces in a fixed order so we can
separate "sample rows arrive in different order" from "the final per-tube
gradient is actually different."

Tiny MPS smoke with default autograd plus diagnostic reduction:

```text
out json:
  research_project/benchmarks/results/uvt_gradient_repeatability_probe_mps_smoke_16_2f_t16_sortedcpu_diag.json

sample digests                         1 unique
default fixed index_add digests        3 unique
diagnostic fixed sorted_cpu digests    1 unique
diagnostic generated sorted digests    1 unique
autograd digests                       3 unique
```

Tiny MPS smoke with `--autograd-reduction-mode sorted_cpu`:

```text
out json:
  research_project/benchmarks/results/uvt_gradient_repeatability_probe_mps_smoke_16_2f_t16_sortedcpu_autograd.json

sample digests                         3 unique
default fixed index_add digests        3 unique
diagnostic fixed sorted_cpu digests    1 unique
diagnostic generated sorted digests    1 unique
autograd digests                       1 unique
loss span                              0.0
```

The real 512px seed-2 fixed-window run matches the useful result:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_repeatability_step0_sortedcpu_autograd.json

sample digests                         3 unique
default fixed index_add digests        3 unique
diagnostic fixed sorted_cpu digests    1 unique
diagnostic generated sorted digests    1 unique
autograd digests                       1 unique
loss span                              0.0

sample max grad_q delta                204.0955810546875
default fixed index_add grad_q delta   0.375
sorted_cpu fixed grad_q delta          0.0
sorted_cpu generated grad_q delta      0.0
autograd max grad delta                0.0
```

Read: the raw sample rows can vary, but after canonical deterministic reduction
the generated sample bundles produce the same per-tube gradients, and full
autograd becomes bitwise stable on the fixed window. This strongly implicates
reduction ordering as the practical one-step gradient drift source. It is still
not a training-speed solution because CPU sorted reduction is a diagnostic slow
path; the needed implementation is a device-resident deterministic reducer or
kernel-side sample ordering.

## On-Device Scan Reduction Diagnostic

I then added a custom Metal `reduce_sample_bundle_scan` op plus two diagnostic
reducer modes:

- `scan_metal`: scan compact sample rows in their current row order;
- `sort_scan_metal`: MPS stable-sort compact rows by tube id, then scan.

The direct op smoke matches CPU accumulation on a hand-built bundle exactly
(`max_abs 0.0`), and the autograd bridge produces finite gradients. The tiny
MPS probes also look good:

```text
scan_metal tiny smoke:
  out json:
    research_project/benchmarks/results/uvt_gradient_repeatability_probe_mps_smoke_16_2f_t16_scanmetal_autograd.json
  sample digests               2 unique
  default index_add digests    3 unique
  fixed scan digests           1 unique
  generated scan digests       1 unique
  autograd digests             1 unique

sort_scan_metal tiny smoke:
  out json:
    research_project/benchmarks/results/uvt_gradient_repeatability_probe_mps_smoke_16_2f_t16_sortscanmetal_autograd.json
  sample digests               3 unique
  default index_add digests    3 unique
  fixed sorted-scan digests    1 unique
  generated sorted-scan        1 unique
  autograd digests             1 unique
```

But both on-device scan reducers fail the real 512px fixed-window gate:

```text
scan_metal 512px:
  out json:
    research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_repeatability_step0_scanmetal_autograd.json
  sample digests               3 unique
  fixed scan digests           1 unique
  generated scan digests       3 unique
  autograd digests             3 unique
  generated grad_q max delta   0.65625
  autograd max grad delta      4.889443516731262e-09

sort_scan_metal 512px:
  out json:
    research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_repeatability_step0_sortscanmetal_autograd.json
  sample digests               3 unique
  fixed sorted-scan digests    1 unique
  generated sorted-scan        3 unique
  autograd digests             3 unique
  generated grad_q max delta   0.1875
  autograd max grad delta      3.958120942115784e-09
```

Read: fixed-bundle reduction can be stable on MPS, but real-scale generated
sample rows still carry atomic append ordering into float32 accumulation. MPS
sort-by-id plus a float32 scan is not enough to reproduce the CPU sorted-float64
result. The next implementation target should be kernel-side deterministic
sample keys/order or a true tile-pair/per-tube VJP, not another reducer wrapper.

## Keyed Sample Emission Follow-Up

I then added keyed sample emission to the same Metal backward kernel. The new
`stable_backward_samples_with_keys` path emits one deterministic key per compact
sample row. The `key_sort_scan_metal` reducer sorts by `(tube_id, key)` before
the custom Metal scan reducer, so atomic append row order no longer controls
the per-tube accumulation order.

Tiny keyed smoke:

```text
out json:
  research_project/benchmarks/results/uvt_gradient_repeatability_probe_mps_smoke_16_2f_t16_keysortscan_autograd.json

sample digests                         1 unique
default index_add digests              3 unique
fixed keyed-scan digests               1 unique
generated keyed-scan digests           1 unique
autograd digests                       1 unique
autograd loss span                     0.0
```

Real 512px seed-2 fixed-window keyed gate:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_repeatability_step0_keysortscan_autograd.json

sample rows                            484697
sample digests                         3 unique
default index_add digests              3 unique
fixed keyed-scan digests               1 unique
generated keyed-scan digests           1 unique
autograd digests                       1 unique
generated keyed-scan max delta         0.0
autograd max grad delta                0.0
autograd loss span                     0.0
```

This is the first on-device path that controls the real 512px fixed-window
gradient drift without a CPU sorted reducer. The keyed path also clears the
same-process 512px training-repeat gate:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_keysortscan_repeatability_600steps.json

max steps                              600
repeats                                2
final train PSNR                       15.208253383636475 / 15.208253383636475
final heldout PSNR                     13.75709342956543 / 13.75709342956543
final train PSNR span                  0.0
final heldout PSNR span                0.0
final state max delta                  0.0
final state digest prefix              0c333d6937b7 / 0c333d6937b7
last train-log elapsed                 220.76867083300021s / 206.23561612500635s
```

The shorter 20-step and 100-step seed-2 keyed repeats also report final state
max delta `0.0` and final train/heldout PSNR span `0.0`. This closes the
repeatability gate for the current per-pixel sample path, but it does not solve
the sublinear backward claim: the fixed window still emits `484697` per-pixel
sample rows. The next speed gate is still tile-pair or per-tube VJP.

I then exposed the same `--uvt-reduction-mode` and
`--uvt-sample-emission-mode` switches in the backward timing probes. The bounded
32px/16f/224-tube comparison shows the keyed path is the slow correctness
fallback, not the final rasterizer:

```text
backward breakdown artifacts:
  research_project/benchmarks/results/uvt_backward_breakdown_probe_32_16f_224t_tilet1_cap128_indexadd_compare.json
  research_project/benchmarks/results/uvt_backward_breakdown_probe_32_16f_224t_tilet1_cap128_keysortscan_compare.json

sample rows                            135565
index_add sample+reduce median         6.305208502453752 ms
key_sort_scan sample+reduce median     21.90893750230316 ms
keyed / index_add sample+reduce        3.4747364014650777x
keyed / index_add reduce phase         5.267273899020175x

train-step artifacts:
  research_project/benchmarks/results/uvt_train_step_timing_probe_32_16f_224t_tilet1_cap128_indexadd_compare.json
  research_project/benchmarks/results/uvt_train_step_timing_probe_32_16f_224t_tilet1_cap128_keysortscan_compare.json

index_add backward median              34.564916997624096 ms
key_sort_scan backward median          77.67279200197663 ms
index_add total-step median            39.248250002856366 ms
key_sort_scan total-step median        83.17829100269591 ms
median compact samples                 188522
median UVT tile-tube pairs             11833
sample rows per tile pair              15.93
```

Read: the deterministic keyed path is useful for stable reporting, but any
speed claim must bypass compact per-pixel sample materialization.

I also exposed the keyed path through `multicam_heldout_compare.py` as
`--uvt-reduction-mode key_sort_scan_metal` plus
`--uvt-sample-emission-mode with_keys`. The 1-step MPS smoke
`multicam_heldout_compare_keysortscan_smoke_16_2f_1s` passed, with those modes
recorded in `run_meta.json` and `comparison_report.json`. That makes the next
training-repeat gate runnable without editing the harness again. I also wired
the same flags into `multicam_star_repeatability_probe.py`; the 2-repeat keyed
MPS smoke `multicam_star_repeatability_probe_keysortscan_smoke_16_2f_1step.json`
reports `final_state_max_abs: 0.0`.

## Tile-Pair Backward Diagnostic

The first diagnostic tile-pair emitter is now wired as
`tile_pair_backward_samples` and exposed through
`--uvt-sample-emission-mode tile_pair`. It passes the small stable parity gate:

```text
16px/2f/16 tubes:
  per-pixel rows        4159
  tile-pair rows        128
  max reduced delta     7.62939453125e-05

32px/16f/224 tubes:
  per-pixel rows        135565
  tile-pair rows        10556
  max grad_q delta      0.00030517578125
  unstable tiles        0.0
```

The timing gate is mixed and should not be promoted as a speed win:

```text
32px breakdown tile_pair sample+reduce median       23.357353999017505 ms
32px train-step tile_pair backward median           76.03254100104095 ms
32px train-step tile_pair sample_to_uvt_pair_ratio  1.0
prior keyed per-pixel backward median               77.67279200197663 ms
prior default index_add backward median             34.564916997624096 ms
```

Read: the row-space thesis is now implemented and verified, but the naive
one-thread-per-tile-pair recompute kernel is still not the fast backward. The
next speed work should optimize per-tile accumulation or fuse reduction, not
rerun 512px quality as if this closed the rasterizer claim.

## Direct Atomic Backward Speed Diagnostic

I then added a direct-atomic backward mode:

```text
flag:
  --uvt-sample-emission-mode direct_atomic
```

This path skips sample-row materialization and the Python/MPS reduction bundle.
The Metal kernel accumulates gradients directly into per-tube buffers with
global float atomics. Parity against the keyed per-pixel reduction is within
small float tolerance on stable cases:

```text
16px/2f/16 tubes:
  per-pixel rows          4159
  direct tube-grad rows   16
  max color delta         0.000019073486328125
  max ma delta            0.0000057220458984375
  max opacity delta       0.000019073486328125
  max grad_q delta        0.0001068115234375
  unstable tiles          0.0

32px/16f/224 tubes:
  per-pixel rows          135565
  direct tube-grad rows   224
  max color delta         0.00005340576171875
  max ma delta            0.00003337860107421875
  max opacity delta       0.000091552734375
  max grad_q delta        0.0003662109375
  unstable tiles          0.0
```

This is the first train-step speed win:

```text
32px/16f/224 tubes:
  direct_atomic sample+reduce median       4.366166002000682 ms
  direct_atomic train-step backward median 8.311833000334445 ms
  default index_add backward median        34.564916997624096 ms
  keyed per-pixel backward median          77.67279200197663 ms
  naive tile-pair backward median          76.03254100104095 ms

256px/16f/7168 tubes:
  direct_atomic 5-step backward median     41.306667000753805 ms
  direct_atomic 5-step total median        52.94633399898885 ms
  older default 30-step backward median    391.97191700077383 ms
  older default 30-step total median       403.4594580043631 ms
```

Read: direct atomic proves a fast per-tube-gradient path exists, but it is not
the deterministic reporting path yet. Because it uses global `atomic_float`
accumulation, order can vary.

I wired `direct_atomic` into `multicam_heldout_compare.py` and
`multicam_star_repeatability_probe.py` so the fast path can run through the real
STAR training loop. The valid tiny fixed-step smoke is:

```text
out json:
  research_project/benchmarks/results/multicam_star_repeatability_probe_directatomic_smoke_16_2f_2steps_fixedbudget.json

setup:
  target size             16
  frames                  2
  tubes                   16
  device                  mps
  repeats                 2
  max steps               2
  sample emission mode    direct_atomic

result:
  repeat steps            2 / 2
  final state max delta   0.0
  final train PSNR span   0.0
  final heldout PSNR span 0.0
```

This clears only the tiny wiring/repeatability smoke. The next gate is
useful-scale same-process repeatability and a same-step single-video overfit
comparison before rerunning expensive 512px quality rows.

The first bounded useful-size repeatability gate is:

```text
out json:
  research_project/benchmarks/results/mcam32_s2_t224_directatomic_repeatability_20steps.json

setup:
  target size             32
  frames                  16
  tubes                   224
  device                  mps
  repeats                 2
  max steps               20
  sample emission mode    direct_atomic

result:
  repeat steps            20 / 20
  final state max delta   0.0001089535653591156
  final train PSNR span   0.00000095367431640625
  final heldout PSNR span 0.0000019073486328125
```

Read: the drift is small enough that quality may be fine, but the state digests
differ. Direct atomic is not the bit-exact reporting path at useful size.

The same-step single-video overfit checks are positive:

```text
out json:
  research_project/benchmarks/results/video_fit_single_overfit_256_16f_50steps_7168uvt_lr012_s0125_t20_tilet1_cap128_tileloadreg0003_target60_directatomic_uvtonly_renderbench10_metal_tile.json

matched prior default-backward row:
  research_project/benchmarks/results/video_fit_single_overfit_256_16f_50steps_7168uvt_lr012_s0125_t20_tilet1_cap128_tileloadreg0003_target60_uvtonly_renderbench10_metal_tile.json

setup:
  target size             256
  frames                  16
  tubes                   7168
  steps                   50
  lr                      0.12
  tile load reg/target    0.003 / 60
  sample emission mode    direct_atomic

result:
  direct_atomic PSNR      21.97383165359497
  default-backward PSNR   21.97382688522339
  direct_atomic loss      0.009008552879095078
  default-backward loss   0.009008551016449928
  direct_atomic fit time  5612.386834000063 ms
  default fit time        18746.74650000088 ms
```

Read: the fast path preserves this 50-step overfit result and materially cuts
fit wall time. It still should not replace the deterministic keyed path for
exact-repeat reporting.

The matching 200-step row also clears:

```text
out json:
  research_project/benchmarks/results/video_fit_single_overfit_256_16f_200steps_7168uvt_lr012_s0125_t20_tilet1_cap128_tileloadreg0003_target60_directatomic_uvtonly_renderbench10_metal_tile.json

matched prior default-backward row:
  research_project/benchmarks/results/video_fit_single_overfit_256_16f_200steps_7168uvt_lr012_s0125_t20_tilet1_cap128_tileloadreg0003_target60_uvtonly_renderbench10_metal_tile.json

result:
  direct_atomic PSNR      23.977155685424805
  default-backward PSNR   23.976197242736816
  direct_atomic loss      0.0053431205451488495
  default-backward loss   0.005344003438949585
  direct_atomic fit time  16431.616750000103 ms
  default fit time        42516.04850000149 ms
```

Read: 200-step same-to-same overfit preserves quality and cuts UVT fit wall
time by about `2.59x`. The next question is whether the same direct-atomic path
preserves the multicam train/heldout read.

The first smoke-scale paired multicam row is:

```text
out dir:
  research_project/benchmarks/results/mcam32_s2_t224_directatomic_paired20

setup:
  target size             32
  frames                  16
  STAR tubes              224
  direct splats           2048
  steps                   20
  sample emission mode    direct_atomic

STAR direct atomic:
  train PSNR              8.433213710784912
  heldout PSNR            8.312164306640625
  train loop              1.6225384999997914 s
  render-only eval        0.03705504100798862 s
  max pair ratio          3.1038594969644406
  max tile count          116
  max unstable fraction   0.375

Direct splats:
  train PSNR              8.406636238098145
  heldout PSNR            8.236161231994629
  train loop              2.838062375005393 s
  render-only eval        0.5165199170005508 s
```

Read: this proves the direct-atomic mode runs in the paired multicam harness and
does not obviously hurt the smoke-scale train/heldout read. It is not a
meaningful quality claim because the row is only 32px and 20 steps.

The 256px paired rows are stronger:

```text
fixed-step out dir:
  research_project/benchmarks/results/mcam256_s0_t256_directatomic_fixed365_paired

matched prior default-backward row:
  research_project/benchmarks/results/multicam_heldout_compare_mps_pilot_256_16f_20s_both_dataset_lens_seed0_alltrain_gridinit_allframes_temporal_window4_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle_lr001

STAR direct atomic, 365 steps:
  train PSNR              15.637612342834473
  heldout PSNR            13.968801498413086
  train loop              16.86269975000323 s
  render-only eval        0.056642417002876755 s
  max pair ratio          3.8648713315677004
  max tile count          101
  max unstable fraction   0.0

Prior STAR default backward, 365 steps:
  train PSNR              15.741607666015625
  heldout PSNR            13.769630432128906
  train loop              20.019632791983895 s
  render-only eval        0.03941354202106595 s

Direct splats, same 365-step cap:
  train PSNR              7.935251951217651
  heldout PSNR            7.530378818511963
  train loop              9.370884083997225 s
  render-only eval        0.4438640829976066 s
```

The same-wall-clock row is:

```text
seed 0 out dir:
  research_project/benchmarks/results/mcam256_s0_t256_directatomic_20s_paired

STAR direct atomic, 20 seconds:
  steps                   367
  train PSNR              15.691498756408691
  heldout PSNR            13.885225296020508
  train loop              20.02054599999974 s
  render-only eval        0.06333295900549274 s

Direct splats, 20 seconds:
  steps                   812
  train PSNR              10.62636947631836
  heldout PSNR            8.864513397216797
  train loop              20.004992750000383 s
  render-only eval        0.37127641700499225 s
```

The same 20-second direct-atomic recipe now has a three-seed paired read:

| seed | artifact | STAR steps | STAR heldout | direct steps | direct heldout | STAR render-only s | direct render-only s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `mcam256_s0_t256_directatomic_20s_paired` | `367` | `13.885225296020508` | `812` | `8.864513397216797` | `0.06333295900549274` | `0.37127641700499225` |
| 1 | `mcam256_s1_t256_directatomic_20s_paired` | `487` | `13.736623764038086` | `906` | `9.101115226745605` | `0.05331295800715452` | `0.34283045700431103` |
| 2 | `mcam256_s2_t256_directatomic_20s_paired` | `514` | `13.848220825195312` | `943` | `9.198732376098633` | `0.05993591599690262` | `0.2928721249991213` |

Read: the 256px direct-atomic rows clear direct splats by a large margin on all
three seeds and clear the V-JEPA F32 heldout reference `13.6248` on all three.
Relative to the prior default-backward STAR rows, direct atomic is mixed:
seed 0 improves by about `0.116` dB, seed 1 loses about `0.032` dB, and seed 2
improves by about `0.084` dB. The same-wall-clock STAR step count is not a
simple raw-kernel-speed story (`367`, `487`, `514` direct-atomic steps versus
`365`, `346`, `333` prior default steps), so this does not mean the whole
multicam trainer is now raw-kernel-speed-bound. It does mean the fast backward
path is quality-safe enough for a 512px probe, with keyed reduction still
reserved for exact-repeat reporting.

That 512px direct-atomic probe is now a three-seed matrix:

```text
out dirs:
  research_project/benchmarks/results/mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_directatomic_paired
  research_project/benchmarks/results/mcam512_s1_t320_view_shuffled_cycle_fixed600_besttrain_directatomic_paired
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_directatomic_paired

STAR direct atomic, fixed 600 steps:
  seed 0 heldout PSNR     13.669089317321777
  seed 1 heldout PSNR     13.904035568237305
  seed 2 heldout PSNR     13.819857597351074
  min / mean / max        13.669089317321777 / 13.797660827636719 / 13.904035568237305
  weakest V-JEPA margin   +0.04428931732177688 dB
  train loops             30.19863641700067 s / 23.72993458300334 s / 21.192358166001213 s
  render-only eval        0.16537191598763457 s / 0.13686329199845204 s / 0.11917258299945388 s
  overflow/unstable       all 0 / 0.0

Direct splats in the same runs:
  heldout PSNR            8.247410774230957 / 8.256668090820312 / 8.267980575561523
  train loops             17.67805879200023 s / 18.272382041999663 s / 14.366463208003552 s
```

Read: the 512px direct-atomic matrix clears V-JEPA on all three seeds and fixes
the known weak paired seed-2 blocker. It also raises the matrix floor versus the
nearest prior paired default-backward rows, which were `13.730086326599121`,
`13.756308555603027`, and `13.608675003051758`. It is not a pure upgrade over
every historical STAR row: seed 0 is lower than the strongest paired seed-0
rerun and below the two STAR-only seed-0 repeats. It still does not close
deterministic repeatability.

The direct-atomic repeatability check rejects promotion:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_directatomic_repeatability_600steps.json

repeat 0:
  train loop              22.75719408299483 s
  train PSNR              15.35076904296875
  heldout PSNR            13.713482856750488
  digest                  89e536511d0c8ca527ed50476a3d185654b3c481168c0ce7fd622f0bf4ae6ea8

repeat 1:
  train loop              22.819027249999635 s
  train PSNR              15.212846279144287
  heldout PSNR            13.746484756469727
  digest                  44afd4bb88f0e25fe3727dbd0abd72a78b68242ff16895101a4f5e8f5f9f79db

deltas:
  final state max abs     1.431039810180664
  final state mean abs    0.11831939475876944
  train PSNR span         0.1379227638244629
  heldout PSNR span       0.03300189971923828
```

Both repeats still clear V-JEPA, but the state drift is far beyond a reporting
tolerance. Direct atomic is a fast exploratory path, not the exact-repeat path.

Shorter repeatability probes show how the drift grows:

```text
20-step out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_directatomic_repeatability_20steps.json
  final state max abs     0.00010570883750915527
  final state mean abs    4.9138592917838e-7
  train PSNR span         0.0
  heldout PSNR span       0.0

100-step out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_directatomic_repeatability_100steps.json
  final state max abs     0.03318440169095993
  final state mean abs    0.00016630965858764415
  train PSNR span         0.0000209808349609375
  heldout PSNR span       0.000308990478515625
```

Read: direct-atomic drift is present immediately, grows by 100 steps, and is
quality-visible by 600 steps. The next renderer/trainer work should be a
repeatable fused/tile-pair backward, not another direct-atomic quality sweep.

## Tile-Pair Deterministic Gate

I then wired `tile_pair` into the multicam repeatability and heldout harness
validation so it can run with `key_sort_scan_metal`. The small smoke is exact:

```text
out json:
  research_project/benchmarks/results/mcam32_s2_t224_tilepair_keysortscan_repeatability_20steps.json

final state max abs       0.0
train PSNR span           0.0
heldout PSNR span         0.0
```

The useful 512px seed-2 gates are also exact at 20, 100, and 600 steps:

```text
20-step out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_keysortscan_repeatability_20steps.json
  train loops             2.834733083000174 s / 1.3231617080018623 s
  heldout PSNR            7.95670223236084 / 7.95670223236084
  final state max abs     0.0

100-step out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_keysortscan_repeatability_100steps.json
  train loops             13.83672304200445 s / 11.261568125002668 s
  heldout PSNR            9.824247360229492 / 9.824247360229492
  final state max abs     0.0

600-step out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_keysortscan_repeatability_600steps.json
  train loops             112.29406904200005 s / 85.4813433330055 s
  train PSNR              15.138561725616455 / 15.138561725616455
  heldout PSNR            13.569375991821289 / 13.569375991821289
  final state max abs     0.0
```

This is a real deterministic compact-row path, but it does not close promotion.
At 600 steps it is about `2x` faster than the existing deterministic per-pixel
keyed path (`221.05617970800085s` / `206.5364632500059s`), but still about
`4-5x` slower than direct atomic (`22.75719408299483s` /
`22.819027249999635s`). More importantly, its 600-step heldout PSNR
`13.569375991821289` is below the per-pixel keyed row
`13.75709342956543` and below both direct-atomic repeats
`13.713482856750488` / `13.746484756469727`.

Read: tile-pair has the right repeatability shape and a better row count, but
its gradient path is not quality-equivalent enough to be the reporting path.
The next implementation slice is tile-pair gradient parity/quality first, then
speed; direct atomic remains the fast exploratory path.

## Tile-Pair Gradient Localization

I then extended `uvt_gradient_repeatability_probe.py` so one run can compare two
autograd backward modes on the same fixed window, and optionally pretrain before
the comparison. The local gradient deltas are tiny:

```text
initial state:
  out json              research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_step0_withkeys_vs_tilepair.json
  loss delta            0.0
  max gradient delta    6.51925802230835e-09

600-step direct-atomic-pretrained state:
  out json              research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_step600_directatomic_pretrain_withkeys_vs_tilepair.json
  pretrain loop         23.07617775000108s
  comparison loss       0.12468379735946655
  loss delta            0.0
  max gradient delta    1.2777745723724365e-06

600-step tile-pair-pretrained state:
  out json              research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_step600_tilepair_pretrain_withkeys_vs_tilepair.json
  pretrain loop         135.12079912499757s
  comparison loss       0.11894877254962921
  loss delta            0.0
  max gradient delta    7.320195436477661e-07
```

Read: the lower 600-step heldout PSNR from the tile-pair training path is not
explained by a gross missing term in the local tile-pair VJP. The most likely
current explanation is optimizer trajectory sensitivity from tiny deterministic
floating-point sum-order differences. The next useful debug is a matched
keyed-vs-tile-pair trajectory replay or an accumulation experiment that reduces
sum-order drift without returning to per-pixel rows.

## Matched Trajectory Replay

I added `multicam_star_mode_compare_probe.py` to train the deterministic
per-pixel keyed path and the deterministic tile-pair path from the same nominal
setup, then compare matching checkpoints.

The 100-step replay shows state divergence before it is visible in metrics:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_100steps.json

train loops:
  keyed per-pixel        18.06653083299898s
  tile-pair              14.569895291002467s

final state delta:
  max abs                0.02207188308238983
  mean abs               0.00013591208698926494

final PSNR delta, tile-pair minus keyed:
  train                  -0.00025272369384765625
  heldout                -0.00014972686767578125
```

The 600-step replay reproduces the full quality gap in a single artifact:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_600steps.json

train loops:
  keyed per-pixel        220.26230666699848s
  tile-pair              171.74044720900565s

final keyed:
  train PSNR             15.208253383636475
  heldout PSNR           13.75709342956543

final tile-pair:
  train PSNR             15.138561725616455
  heldout PSNR           13.569375991821289

final delta, tile-pair minus keyed:
  train PSNR             -0.06969165802001953
  heldout PSNR           -0.18771743774414062
  state max abs          1.417811632156372
  state mean abs         0.12496386851583208
```

Matched checkpoint state divergence:

| step | state max abs | state mean abs | keyed elapsed s | tile-pair elapsed s |
|---:|---:|---:|---:|---:|
| 100 | `0.02207188308238983` | `0.00013591208698926494` | `18.08824708299653` | `23.5531806670042` |
| 200 | `0.5959800481796265` | `0.011760711829577173` | `50.11988695799664` | `58.057103292005195` |
| 300 | `1.0160683393478394` | `0.05342666847365243` | `90.49734808300127` | `87.3491251670057` |
| 400 | `1.0968774557113647` | `0.09600993309702192` | `132.03449170800013` | `115.02822808400379` |
| 500 | `1.3656842708587646` | `0.12601606845855712` | `178.26661516699824` | `141.85255929200503` |
| 600 | `1.417811632156372` | `0.12496386851583208` | `219.96248229200137` | `171.7128001250021` |

Read: the divergence becomes large between steps 100 and 200, well before the
LR decay at step 500. Tile-pair can be faster in this matched replay, but its
trajectory is not quality-equivalent. The next useful implementation experiment
is not another metric sweep; it is an accumulation-order experiment, such as
compensated per-tile summation or an ordered tile-pair reduction variant that
tests whether the deterministic compact row path can track the keyed trajectory
more closely.

## Compensated Tile-Pair Probe

I added an opt-in `tile_pair_compensated` mode that uses compensated summation
inside each tile-pair row while keeping the existing plain `tile_pair` path
unchanged. The new mode is wired through the Metal op, Python bridge, autograd
harness, and benchmark CLIs. Runtime shader compilation passed:

```text
smoke:
  research_project/benchmarks/results/uvt_backward_breakdown_probe_tilepair_compensated_smoke_16_2f_16t_tilet1.json

sample backward median:
  317.8678330004914 ms

reduce median:
  147.1505410008831 ms

valid sample rows:
  128 / 128
```

The 100-step keyed-vs-compensated replay is:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_compensated_100steps.json

train loops:
  keyed per-pixel        24.060547749999387s
  tile-pair compensated  20.05895091700222s

final state delta:
  max abs                0.040996529161930084
  mean abs               0.00013770289037243595

final PSNR delta, compensated minus keyed:
  train                  -0.0002446174621582031
  heldout                +0.00005626678466796875
```

Matched checkpoint state divergence:

| step | plain tile-pair max abs | compensated max abs |
|---:|---:|---:|
| 20 | `0.00010204315185546875` | `0.0000699758529663086` |
| 40 | `0.0034194663166999817` | `0.0034178420901298523` |
| 60 | `0.0026456117630004883` | `0.0027609169483184814` |
| 80 | `0.004040230996906757` | `0.0024684304371476173` |
| 100 | `0.02207188308238983` | `0.040996529161930084` |

Read: compensated summation is not the fix. It preserves the tiny 100-step PSNR
tie, but it does not track the keyed trajectory better; the final state max
delta is worse than plain tile-pair at step 100. Do not spend a 600-step run on
this variant unless another change also alters ordering or optimization
dynamics.

## Compensated Final-Reducer Probe

I added a second diagnostic variant, `key_sort_compensated_scan_metal`, that
keeps the sample emission unchanged and only changes the final per-tube Metal
scan to use compensated summation. Runtime smoke passed:

```text
smoke:
  research_project/benchmarks/results/uvt_backward_breakdown_probe_keysort_compensated_tilepair_smoke_16_2f_16t_tilet1.json

reduction mode:
  key_sort_compensated_scan_metal

sample emission:
  tile_pair
```

The 100-step matched replay is:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_keysort_compensated_100steps.json

train loops:
  keyed per-pixel        20.748569541996403s
  tile-pair              14.94696783299878s

final state delta:
  max abs                0.03426568582653999
  mean abs               0.00013610214602002607

final PSNR delta, tile-pair minus keyed:
  train                  -0.000010013580322265625
  heldout                -0.00027942657470703125
```

Step-100 state max deltas across the small accumulation variants:

| variant | step-100 state max abs | heldout delta vs keyed |
|---|---:|---:|
| plain tile-pair + keyed scan | `0.02207188308238983` | `-0.00014972686767578125` |
| tile-pair + compensated keyed scan | `0.03426568582653999` | `-0.00027942657470703125` |
| compensated tile-pair + keyed scan | `0.040996529161930084` | `+0.00005626678466796875` |

Read: the final compensated scan is also not the fix. It passes the smoke but
does not improve the trajectory relative to plain tile-pair, so there is no
reason to run the 600-step row for this variant. The remaining useful
rasterizer work needs to change the compact emission/reduction structure, not
just add compensation around the existing sums.

## LR Stability Bracket

I tested whether the tile-pair trajectory gap is mostly an optimizer-step-size
issue by rerunning the matched keyed-vs-tile-pair probe at lower LR. This is a
minimal schedule bracket, not a broad quality sweep.

At LR `0.005`, the 200-step checkpoint is much closer than the LR `0.01` run:

```text
out json:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_lr0005_200steps.json

LR 0.005, step 200:
  state max abs          0.029318034648895264
  state mean abs         0.00015659207503111768
  heldout delta          -0.00011730194091796875

LR 0.01, step 200 from the 600-step replay:
  state max abs          0.5959800481796265
  state mean abs         0.011760711829577173
```

The 600-step LR bracket is:

| LR | keyed heldout | tile-pair heldout | tile-pair minus keyed | final state max abs | step-200 state max abs |
|---:|---:|---:|---:|---:|---:|
| `0.01` | `13.75709342956543` | `13.569375991821289` | `-0.18771743774414062` | `1.417811632156372` | `0.5959800481796265` |
| `0.0075` | `13.636750221252441` | `13.562674522399902` | `-0.07407569885253906` | `1.1469531059265137` | `0.11033141613006592` |
| `0.005` | `13.182788848876953` | `13.314628601074219` | `+0.13183975219726562` | `0.7858446836471558` | `0.029318034648895264` |

Read: LR controls how fast the trajectories separate, but LR tuning alone does
not produce a promotion recipe. LR `0.005` makes tile-pair relatively stable
and even better than keyed at the same underfit setting, but absolute heldout
PSNR drops to `13.314628601074219`. LR `0.0075` recovers keyed quality but still
leaves tile-pair at `13.562674522399902`, essentially the same absolute
tile-pair quality as LR `0.01` and still below V-JEPA. The next fix should
change the deterministic compact backward/reduction, not just the LR.

I then tested a structural compact-emission variant, `tile_pair_scanline`, that
splits each `(tile, tube slot)` backward row into `(tile, tube slot, local
scanline)` rows. The goal was to preserve more keyed-like accumulation order
without falling all the way back to per-pixel sample rows. The tiny MPS smoke
passes:

```text
artifact:
  research_project/benchmarks/results/uvt_backward_breakdown_probe_tilepair_scanline_smoke_16_2f_16t_tilet1.json

sample emission mode:
  tile_pair_scanline

allocated sample slots:
  8192

valid sample rows:
  1024

sample+reduce median:
  919.125583001005 ms
```

The 100-step keyed-vs-scanline replay is:

```text
artifact:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_scanline_100steps.json

train loop:
  keyed per-pixel        19.360704333004833s
  tile-pair scanline     26.565228542000114s

final state delta:
  max abs                0.02909490466117859
  mean abs               0.00013122651705219012

final PSNR delta, scanline minus keyed:
  train                  -0.0002732276916503906
  heldout                -0.0000896453857421875
```

Read: scanline emission is also rejected as the next promotion path. It keeps
PSNR tied at 100 steps, but it tracks keyed state worse than plain tile-pair at
the same gate (`0.02909490466117859` versus `0.02207188308238983` max abs) and
is slower than keyed in the short replay because it expands rows. Do not run the
600-step scanline row unless a later patch changes the row structure or reducer
again.

The next change was smaller but more important: keep `tile_pair`'s row
structure, but leave rows invalid when the accumulated gradient is exactly
zero. This prunes support-bounds rows that do not actually contribute to the
backward pass.

Tiny and 32px checks:

```text
tiny smoke:
  artifact              research_project/benchmarks/results/uvt_backward_breakdown_probe_tilepair_zero_prune_smoke_warm_16_2f_16t_tilet1.json
  valid rows            117
  sample+reduce median  9.835853998083621 ms

keyed comparison smoke:
  artifact              research_project/benchmarks/results/uvt_gradient_repeatability_probe_tilepair_zero_prune_vs_keyed_smoke_16_2f_t16.json
  comparison max abs    0.0

32px timing:
  artifact              research_project/benchmarks/results/uvt_backward_breakdown_probe_32_16f_224t_tilet1_cap128_tilepair_zero_prune_compare.json
  valid rows            4802
  prior tile-pair rows  10556
  sample+reduce median  21.83349999540951 ms
  prior median          23.357353999017505 ms

32px train-step:
  artifact              research_project/benchmarks/results/uvt_train_step_timing_probe_32_16f_224t_tilet1_cap128_tilepair_zero_prune_compare.json
  backward median       65.07550049718702 ms
  prior tile-pair       76.03254100104095 ms
```

The 512px matched replay is now positive:

```text
100-step artifact:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_zero_prune_100steps.json

100-step final state max/mean:
  0.021925896406173706 / 0.00012713783196107085

600-step artifact:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_zero_prune_600steps.json

train loop:
  keyed per-pixel        222.79236249999667s
  tile-pair pruned       146.85938100000203s

final heldout PSNR:
  keyed per-pixel        13.75709342956543
  tile-pair pruned       13.866263389587402

final heldout delta:
  +0.10916996002197266
```

The pruned tile-pair repeatability gate is exact:

```text
artifact:
  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_zero_prune_repeatability_600steps.json

loops:
  105.77447474999644s
  77.70333404199482s

final train/heldout PSNR:
  15.34240198135376 / 13.866263389587402

final state max/mean delta:
  0.0 / 0.0
```

Follow-up repeatability now covers all paired seeds:

| seed | artifact | loop s, repeat 0 | loop s, repeat 1 | heldout PSNR | final state max/mean delta |
|---:|---|---:|---:|---:|---:|
| 0 | `mcam512_s0_t320_view_shuffled_cycle_tilepair_zero_prune_repeatability_600steps.json` | `106.25342641599855` | `77.79185762500128` | `13.728736877441406` | `0.0 / 0.0` |
| 1 | `mcam512_s1_t320_view_shuffled_cycle_tilepair_zero_prune_repeatability_600steps.json` | `147.09395920800307` | `81.63904133300093` | `13.71193790435791` | `0.0 / 0.0` |
| 2 | `mcam512_s2_t320_view_shuffled_cycle_tilepair_zero_prune_repeatability_600steps.json` | `105.77447474999644` | `77.70333404199482` | `13.866263389587402` | `0.0 / 0.0` |

The paired three-seed matrix is the strongest deterministic STAR read so far:

| seed | artifact | STAR selected heldout | direct heldout | STAR loop s | direct loop s | STAR render-only s | direct render-only s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `mcam512_s0_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_zero_prune_paired` | `13.728736877441406` | `8.247447967529297` | `107.09030087500287` | `17.26219120799942` | `0.10847329099487979` | `0.5315814570058137` |
| 1 | `mcam512_s1_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_zero_prune_paired` | `13.71193790435791` | `8.256773948669434` | `133.76722862500174` | `17.80944529199769` | `0.12546491600369336` | `0.5987954170050216` |
| 2 | `mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_zero_prune_paired` | `13.866263389587402` | `8.26813793182373` | `113.04000200000155` | `17.273487166996347` | `0.11670004200277617` | `0.32718483400094556` |

Aggregate selected heldout PSNR:

- STAR min/mean/max: `13.71193790435791` / `13.76897939046224` / `13.866263389587402`
- direct splats min/mean/max: `8.247447967529297` / `8.257453282674154` / `8.26813793182373`
- weakest STAR margin above the V-JEPA F32 heldout reference `13.6248`: `+0.08713790435791066`

Read: zero-row pruning is not just a timing cleanup at 512px. It makes
tile-pair the current deterministic candidate: exact on all three seed repeats,
faster than keyed per-pixel, faster at render-only eval than direct splats, and
above the existing V-JEPA F32 heldout reference on all three paired seeds. It is
still not the final production claim: the V-JEPA row is still the existing
reference rather than a freshly rerun full-resolution matched row.

## Earlier Direct-Splat Context

The closest paired 512px lens-aware direct-splat references are:

| seed | artifact | STAR final heldout | STAR selected heldout | direct heldout | STAR selected render-only s | direct render-only s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `multicam_heldout_compare_mps_pilot_512_16f_60s_both_dataset_lens_seed1_alltrain_gridinit_allframes_lrdecay300x005_traingain_drop002_checkpoint100_temporal_window1_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle` | `13.494588851928711` | `13.415751457214355` | `10.3926362991333` | `0.1480234160001146` | `0.4046094999998786` |
| 2 | `multicam_heldout_compare_mps_pilot_512_16f_60s_both_dataset_lens_seed2_alltrain_gridinit_allframes_lrdecay500x005_traingain_drop002_checkpoint100_temporal_window1_tileload0001_target7000_depthslope005_tilet1_cap256_compact_bundle` | `13.701825141906738` | `13.678083419799805` | `10.760580062866211` | `0.11207666699988295` | `0.5262394170001699` |

These paired rows are not the current best STAR schedule, but they show the
direct-splat baseline is far below STAR on the same 512px dataset/lens contract
and slower at render-only eval.

## Read

The current 512px STAR result is a real improvement over the prior 512px
schedule churn, and STAR decisively beats direct splats in the paired matrix.
Still, this is not a robust V-JEPA replacement: seed 2 misses the V-JEPA F32
heldout reference three times under paired best-train runs, even though two
STAR-only seed-2 repeats clear it.
The saved `view_shuffled_cycle` curves show all three seeds clearing the V-JEPA
F32 heldout reference under a non-heldout `best_train_psnr` read, with clean
Metal stats. The paired seed-0 rerun shows that this selector can still choose a
checkpoint below V-JEPA even when the heldout-best checkpoint in the same curve
would clear it. The balanced-selector rerun goes further: the same seed and
nominal recipe can produce a curve whose heldout-best checkpoint is also below
V-JEPA. The later STAR-only reruns and paired seed-0/seed-1 best-train rows land
above V-JEPA, but paired seed 2 misses three times and the seed-2 STAR-only rows
clear.

This still is not a production promotion:

- the V-JEPA row is the existing 256px/16f F32 reference, not a freshly rerun
  full-resolution V-JEPA row;
- the original three `view_shuffled_cycle` rows are STAR-only;
- paired seed 2 misses V-JEPA by `0.0316`, `0.0161`, and `0.0756` dB on three
  live rows, while STAR-only seed 2 clears by `0.1642` and `0.1589` dB;
- the balanced selector remains rejected;
- the main weak link is live run variance under the 512px recipe, not render
  stability or direct-splat quality;
- render speed is good and clean, but training scale policy and selector behavior
  still need a stricter reporting gate before claiming a solved recipe.

## Next Gate

Use this recipe as the current 512px STAR schedule baseline, but do not promote
`best_train_psnr` as solved and do not use the balanced selector. The next clean
experiment should be one of:

1. implement an ordered tile-pair reduction variant, or another deterministic
   compact-row reducer only if the pruned tile-pair matrix fails the next
   repeatability or scale gate;
2. run a fresh V-JEPA/full-resolution matched row before calling this a final
   promotion;
3. keep keyed per-pixel as the correctness fallback and direct atomic as the
   fast exploratory path;
4. test larger/full-resolution only after preserving the pruned tile-pair
   repeatability gate;
5. run another 512px quality row only if it answers repeatability, scale, or a
   matched-baseline question.

## Rasterizer Scale And Stronger Baseline Boundary

After the step-back review, I checked the local V-JEPA evidence before launching
another long baseline. The existing multires fast token-budget run
`24absic1` is stronger than the older 256px alpha-threshold reference used above:

```text
log        outputs/run_logs/20260508_144522_fast512_tokenbudget_train.log
checkpoint outputs/multicam_relative_pose/full_relpose_features_F32_multires64_128_256_512_tokenbudget_world4_fast_alpha1_128_relpose_outputinit012_goodset_train0006_0014_holdout0005/checkpoint_final.pt
wall       250 tqdm steps, about 1507s
train      17.1348 / 17.4790 PSNR
heldout    13.9870 PSNR
```

This is not a pure 512px rerun: the checkpoint config has model/render base size
`256` and a multires 64/128/256/512 schedule. It is still the strongest local
same-goodset V-JEPA/TokenGS-style row found in this pass. Therefore the pruned
tile-pair STAR matrix beats direct splats and the older `13.6248` F32 reference,
but it does **not** beat the stronger `13.9870` multires V-JEPA row yet.

I also ran the missing pure-512 V-JEPA F32 refresh manually with no W&B/media
logging. It used the local pure-512 v6refined goodset config for 250 steps,
seed 0, and saved:

```text
out dir     outputs/multicam_relative_pose/full_relpose_features_F32_512_v6refined_goodset_train0006_0014_holdout0005_manual250_nomedia
checkpoint  checkpoint_final.pt
summary     manual_probe_summary.json
loop        4382.390926374996s / 73.0398487729166m
train       15.6574 / 14.3392 PSNR
train mean  14.9983 PSNR
heldout     13.5727 PSNR
```

The original `manual_probe_report.json` contains all 250 step rows and the
checkpoint path but missed the metrics payload key; `manual_probe_summary.json`
records the final metrics from the terminal `validation_video_payload` print.
Read: the deterministic zero-pruned STAR matrix beats this pure-512 manual
baseline on heldout PSNR and is much faster in this local loop comparison, but
the stronger multires row above is still the row to beat for a serious
V-JEPA/TokenGS replacement claim.

The rasterizer scale probe keeps `7168` tubes, 16 frames,
`spatial_precision=0.125`, `temporal_precision=2.0`, `tile_t=1`, and cap `128`.
Forward fixed-tube artifacts:

| target | artifact | UVT pairs | Metal buffer bytes | render ms |
|---:|---|---:|---:|---:|
| 128 | `uvt_forward_speed_probe_128_16f_7168_s0125_t20_tilet1_cap128_metalonly_scale.json` | `451838` | `4243456` | `4.248926999935065` |
| 256 | `uvt_forward_speed_probe_256_16f_7168_s0125_t20_tilet1_cap128_metalonly_scale.json` | `516990` | `16973824` | `4.722833375126356` |
| 512 | `uvt_forward_speed_probe_512_16f_7168_s0125_t20_tilet1_cap128_metalonly_scale.json` | `531638` | `67895296` | `11.279520875177695` |
| 1024 | `uvt_forward_speed_probe_1024_16f_7168_s0125_t20_tilet1_cap128_metalonly_scale.json` | `539762` | `271581184` | `7.2551334000309` |

The active pair count stays almost flat while pixels and output-buffer memory
grow `64x` from 128 to 1024. Timing is noisy, but the pair-count behavior is the
evidence we were missing for sparse forward scaling.

Zero-pruned tile-pair backward fixed-tube artifacts:

| target | artifact | rows | valid rows | allocated slots | sample+reduce ms |
|---:|---|---:|---:|---:|---:|
| 256 | `uvt_backward_breakdown_probe_256_16f_7168_s0125_t20_tilet1_cap128_tilepair_zero_prune_scale_rerun.json` | `207324` | `207324` | `2097152` | `256.09837500815047` |
| 512 | `uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_zero_prune_scale.json` | `211374` | `211374` | `8388608` | `129.20112499705283` |
| 1024 | `uvt_backward_breakdown_probe_1024_16f_7168_s0125_t20_tilet1_cap128_tilepair_zero_prune_scale.json` | `213642` | `186513` | `33554432` | `125.9666664955148` |

Read: yes, we now have fixed-tube sparse/sublinear UVT rasterizer evidence in
forward pair counts and compact backward row counts. No, this is not yet a full
fast training win. The 256px tile occupancy case is still inefficient, the full
train loop is slower than direct splats, and STAR quality has not beaten the
stronger multires V-JEPA row.

## 1024px Larger-Resolution Probe

After preserving the zero-pruned tile-pair repeatability gate, I ran one
STAR-only 1024px seed-2 probe with the same deterministic 512px recipe:

```text
artifact      research_project/benchmarks/results/mcam1024_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_zero_prune_staronly
target        1024
seed          2
tubes         320
steps         600
selector      best_train_psnr
reduction     key_sort_scan_metal
emission      tile_pair
train loop    209.39192004200595s
train PSNR    15.243183135986328
heldout PSNR  13.766105651855469
render-only   0.8590989580116002s
heldout-only  0.28504458300449187s
max pair      5.134140396727249
max tile      110
overflow      0
unstable      0.0
```

The checkpoint curve climbs monotonically on both train and heldout PSNR:
heldout goes `9.1142 -> 11.8340 -> 13.0892 -> 13.3349 -> 13.6549 -> 13.7661`
from steps 100 to 600. That means the run is not failing because the selector
missed an earlier heldout peak. It is simply not enough.

Read: this rejects naive fuller-resolution escalation as the next answer. The
1024px run is below the 512px seed-2 zero-pruned paired row
(`13.866263389587402` heldout, `113.04000200000155` s loop) and still below the
stronger multires V-JEPA row `13.9870`. It also shows trained support growth:
the fixed-tube scale probe had roughly flat pairs, but the trained 1024px model
reaches max pair ratio `5.13`, so quality training can expand support enough to
erase the clean raw scaling story. Next work should control support growth or
change the representation/schedule; do not spend another run on a plain
resolution bump.

I then tested whether stronger support control at 1024px fixes that failure by
raising only the tile-load penalty from `0.001` to `0.003`, keeping target
`7000`:

```text
artifact      research_project/benchmarks/results/mcam1024_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_zero_prune_staronly_tileload003
train loop    192.36781579200033s
train PSNR    14.983142852783203
heldout PSNR  13.61436939239502
render-only   0.8722000830020988s
heldout-only  0.3928669590022764s
max pair      4.722782922009996
max tile      92
final proxy   12826.658203125
overflow      0
unstable      0.0
```

Against the `0.001/7000` row, this cuts loop time by `17.02410425000562` s,
reduces max pair ratio by `0.4113574747172528`, and reduces final tile-load
proxy by about `4683.78`. But heldout PSNR drops by `0.15173625946044922`.
Read: stricter support control does what it is supposed to do mechanically, but
it is not the quality lever. Keep it as a speed/compactness knob; do not use
stronger tile-load pressure as the next route to the `13.9870` multires V-JEPA
gap.

## 512px 1000-Step Step-Budget Probe

I also tested whether the deterministic 512px seed-2 row was simply
undertrained by extending the same zero-pruned recipe to 1000 steps:

```text
artifact      research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed1000_besttrain_tilepair_zero_prune_staronly
target        512
seed          2
tubes         320
steps         1000
selector      best_train_psnr
reduction     key_sort_scan_metal
emission      tile_pair
train loop    336.69192270799977s
selected step 1000
train PSNR    15.832406044006348
heldout PSNR  13.866610527038574
render-only   0.1672017909877468s
heldout-only  0.05512412499228958s
max pair      2.740664448035156
max tile      89
overflow      0
unstable      0.0
```

The heldout curve peaks at step `700`, not at the selected final step:
`9.8242 -> 12.3903 -> 13.4141 -> 13.6017 -> 13.8418 -> 13.8663 -> 13.8759 -> 13.8738 -> 13.8659 -> 13.8666`
from steps 100 through 1000. Read: longer training alone is rejected. Compared
with the 600-step paired seed-2 row (`13.866263389587402` heldout,
`113.04000200000155` s loop), the heldout-best improvement is only
`0.009657859802246094` dB and still does not close the `13.9870` multires
V-JEPA gap.

## Epoch-View-Shuffled Schedule Probe

I added one schedule variant to test whether preserving more cycle structure
while adding epoch-level view variation helps the hard seed-2 case:

```text
--uvt-train-schedule epoch_view_shuffled_cycle
```

The variant keeps temporal windows in cycle order and shuffles train-view order
once per full temporal epoch, instead of per window as `view_shuffled_cycle`
does. The smoke
`research_project/benchmarks/results/multicam_heldout_compare_epoch_view_shuffled_cycle_smoke_16_2f_1s`
passed.

```text
artifact      research_project/benchmarks/results/mcam512_s2_t320_epoch_view_shuffled_cycle_fixed600_besttrain_tilepair_zero_prune_staronly
target        512
seed          2
tubes         320
steps         600
selector      best_train_psnr
reduction     key_sort_scan_metal
emission      tile_pair
train loop    107.1718466670136s
selected step 600
train PSNR    15.319983005523682
heldout PSNR  13.707905769348145
render-only   0.13141591699968558s
heldout-only  0.047126584002398886s
heldout-best  step 400, 13.862957954406738
max pair      3.06181592423155
max tile      96
overflow      0
unstable      0.0
```

The heldout curve is `9.8282 -> 12.2871 -> 13.5111 -> 13.8630 -> 13.5658 -> 13.7079`
from steps 100 through 600. Read: this branch is rejected after seed 2. It
misses the current view-shuffled seed-2 zero-pruned row (`13.866263389587402`)
even by heldout-best checkpoint and remains below the `13.9870` multires V-JEPA
row. Keep the schedule hook as diagnostic code; do not spend seeds 0/1 on it
without a new mechanism.

## Cycle Retry Under Zero-Pruned Tile-Pair

Because the older cycle rejection predated the current zero-pruned tile-pair
candidate, I reran the plain `cycle` schedule once under the current path:

```text
artifact      research_project/benchmarks/results/mcam512_s2_t320_cycle_fixed600_besttrain_tilepair_zero_prune_staronly
target        512
seed          2
tubes         320
steps         600
selector      best_train_psnr
reduction     key_sort_scan_metal
emission      tile_pair
train loop    132.23203912499594s
selected step 600
train PSNR    15.018131732940674
heldout PSNR  13.701606750488281
render-only   0.1128737090039067s
heldout-best  step 600, 13.701606750488281
max pair      2.9901379768184593
max tile      97
overflow      0
unstable      0.0
```

The run stays finite and Metal-clean, so the old non-finite failure is gone
under the current path. The quality is still worse than both
`epoch_view_shuffled_cycle` and the current `view_shuffled_cycle` zero-pruned
seed-2 row. Read: plain cycle is still rejected; do not spend seeds 0/1 on this
retry.

## Static/Dynamic Tube Initialization Probe

I added an opt-in capacity split to initialization:

```text
--uvt-static-tube-fraction 0.25
--uvt-static-init-lambda-t 0.02
```

This is not a semantic static/dynamic classifier. The dynamic pool keeps the
existing `all_train` / `grid` / `all` initialization, while the static pool is
first-frame initialized with lower temporal precision so it starts broader in
time. The smoke
`research_project/benchmarks/results/multicam_heldout_compare_static_dynamic_init_smoke_16_2f_1s`
passed and reported `12` dynamic tubes and `4` static tubes.

Full seed-2 probe:

```text
artifact      research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_static025_lamt002_fixed600_besttrain_tilepair_zero_prune_staronly
target        512
seed          2
tubes         320
static tubes  80
dynamic tubes 240
steps         600
selector      best_train_psnr
reduction     key_sort_scan_metal
emission      tile_pair
train loop    136.58938849999686s
selected step 600
train PSNR    15.551663875579834
heldout PSNR  13.644302368164062
render-only   0.13702220900449902s
heldout-only  0.043864250008482486s
heldout-best  step 600, 13.644302368164062
max pair      3.2617686604907052
max tile      93
overflow      0
unstable      0.0
```

The heldout curve is `11.2966 -> 12.2846 -> 12.8155 -> 13.4511 -> 13.4936 -> 13.6443`
from steps 100 through 600. Read: capacity-split initialization alone is
rejected. It is worse than the current deterministic seed-2 row
(`13.866263389587402`) and still far below the `13.9870` multires V-JEPA row.
Do not spend seeds 0/1 on this exact branch.

## Static-Velocity Regularized Capacity Split

I then added a stronger static-bias knob:

```text
--uvt-static-velocity-reg 0.1
```

This regularizes only the static slice, which is concatenated after the dynamic
tubes. The smoke
`research_project/benchmarks/results/multicam_heldout_compare_static_dynamic_velreg01_smoke_16_2f_1s`
passed and recorded `static_velocity_reg: 0.1`.

Full seed-2 probe:

```text
artifact      research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_static025_lamt002_velreg01_fixed600_besttrain_tilepair_zero_prune_staronly
target        512
seed          2
tubes         320
static tubes  80
dynamic tubes 240
static velreg 0.1
steps         600
selector      best_train_psnr
reduction     key_sort_scan_metal
emission      tile_pair
train loop    160.88216154198744s
selected step 600
train PSNR    15.640942573547363
heldout PSNR  13.454254150390625
render-only   0.2113907500024652s
heldout-only  0.04544458299642429s
heldout-best  step 500, 13.526884078979492
max pair      3.3733146746200675
max tile      95
overflow      0
unstable      0.0
```

The heldout curve is `11.3644 -> 12.5185 -> 13.1813 -> 13.1994 -> 13.5269 -> 13.4543`
from steps 100 through 600. Read: stronger static-motion bias improves train
PSNR relative to init-only split, but worsens heldout and runtime. It is
rejected and should not be expanded to seeds 0/1.

## 512px Direct-Atomic Backward Contrast

After the step-back review, I ran the missing fixed-tube 512px direct-atomic
backward breakdown:

```text
artifact      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_directatomic_scale.json
target        512
frames        16
tubes         7168
tile_t        1
cap           128
emission      direct_atomic
reduction     index_add
sample rows   7168
allocated     7168
sample median 35.70449999824632 ms
reduce median 0.004874993464909494 ms
total median  35.70937499171123 ms
unstable      0.0
```

The matching deterministic zero-pruned tile-pair scale row at 512px emits
`211374` compact rows from `8388608` allocated slots and reports sample+reduce
median `129.20112499705283` ms. Read: direct atomic is about `3.6x` faster in
this fixed-tube backward probe and emits about `29.5x` fewer gradient rows, but
the 512px fixed-600 repeatability artifact already rejects it as the exact
reporting path. The next rasterizer work should aim for a deterministic
fused tile-pair/per-tube VJP that approaches this direct-atomic row count and
timing without losing repeatability.

## Direct-Serial Deterministic Prototype

I added a diagnostic `direct_serial_backward` Metal op and exposed it through:

```text
torch_gsplat_bridge_star_uvt.direct_serial_backward
--uvt-sample-emission-mode direct_serial
```

The kernel is a transposed tile-pair emitter. It launches one deterministic
thread per tube, loops that tube's tile bounds in fixed `(tz, ty, tx, f, y, x)`
order, reconstructs each tile's sorted local tube order, and writes final
per-tube gradients directly. It uses no gradient atomics and no sample-row
reducer.

Tiny parity smoke:

```text
artifact      research_project/benchmarks/results/uvt_direct_serial_backward_parity_smoke_16_2f_16t.json
reference     direct_atomic_backward
candidate     direct_serial_backward
target        16
frames        2
tubes         16
max q delta   4.57763671875e-05
max opacity   3.0517578125e-05
max color     1.1444091796875e-05
max ma        4.291534423828125e-06
unstable      0.0 / 0.0
```

Tiny timing smoke:

```text
artifact      research_project/benchmarks/results/uvt_backward_breakdown_probe_directserial_smoke_16_2f_16t_tilet1.json
target        16
frames        2
tubes         16
sample rows   16
total median  8.373916993150488 ms
```

Fixed 512px timing:

```text
artifact      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_directserial_scale_1it.json
target        512
frames        16
tubes         7168
sample rows   7168
sample median 326.9928749941755 ms
reduce median 0.00912499672267586 ms
total median  327.0019999908982 ms
```

Read: the direct-serial prototype validates a deterministic no-row-buffer
shape, but it is slower than deterministic zero-pruned tile-pair
(`129.20112499705283` ms) and much slower than direct atomic
(`35.70937499171123` ms). The missing path is not one serial thread per tube;
it needs a parallel deterministic segmented or tile-local reduction.

## Tile-Pair Target-Bounds Skip Rejection

I added a diagnostic `tile_pair_target_bounds_backward_samples` Metal op and
exposed it through:

```text
torch_gsplat_bridge_star_uvt.tile_pair_target_bounds_backward_samples
--uvt-sample-emission-mode tile_pair_target_bounds
```

The kernel keeps the deterministic tile-slot row contract from `tile_pair`, but
skips `(f, y, x)` samples outside the target tube's analytic support before
running the expensive per-pixel backward scan. This tested whether the
zero-row-pruned tile-pair path was still wasting most of its time on pixels
that cannot contribute to the target row.

Tiny parity smoke:

```text
artifact       research_project/benchmarks/results/uvt_tile_pair_target_bounds_parity_smoke_16_2f_16t_tilet1.json
reference      tile_pair_backward_samples
candidate      tile_pair_target_bounds_backward_samples
target         16
frames         2
tubes          16
reference rows 117
candidate rows 117
max deltas     0.0 for ma/q/opacity/color after reduction
unstable       0.0 / 0.0
```

Fixed 512px timing:

```text
artifact      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_target_bounds_scale.json
target        512
frames        16
tubes         7168
sample rows   211374
allocated     8388608
sample median 109.98058300174307 ms
reduce median 22.483458000351675 ms
total median  132.46404100209475 ms
unstable      0.0
```

Read: the support skip is correct, but it is not faster at the fixed 512px
scale. It is slightly slower than zero-pruned tile-pair
(`129.20112499705283` ms), so reject it as the next rasterizer path.

## Tile-Pair Suffix-Composite Speed Probe

I added a diagnostic `tile_pair_suffix_backward_samples` Metal op and exposed
it through:

```text
torch_gsplat_bridge_star_uvt.tile_pair_suffix_backward_samples
--uvt-sample-emission-mode tile_pair_suffix
```

The kernel keeps the deterministic tile-slot row contract from `tile_pair`, but
does not allocate per-pixel forward arrays or scan backward through stored
state. For each target tube row it computes prefix transmittance up to the
target, then computes the post-target suffix color composite to recover
`dL/dT_after_target`.

Tiny parity smoke:

```text
artifact        research_project/benchmarks/results/uvt_tile_pair_suffix_parity_smoke_16_2f_16t_tilet1.json
reference       tile_pair_backward_samples
candidate       tile_pair_suffix_backward_samples
target          16
frames          2
tubes           16
reference rows  117
candidate rows  117
max q delta     1.52587890625e-05
max opacity     3.814697265625e-06
max ma          1.9073486328125e-06
max color       0.0
unstable        0.0 / 0.0
```

Fixed 512px timing:

```text
artifact      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_suffix_scale.json
target        512
frames        16
tubes         7168
sample rows   211374
allocated     8388608
sample median 89.43087499937974 ms
reduce median 22.68999999796506 ms
total median  112.1208749973448 ms
unstable      0.0
```

Repeatability and 100-step replay:

```text
repeatability artifact research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_suffix_repeatability_20steps.json
repeatability          final_state_max_abs 0.0
mode-compare artifact  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_suffix_100steps.json
100-step elapsed       keyed 20.499955874998705 s, suffix 11.745555834000697 s
100-step state delta   max 0.05802058428525925, mean 0.0001735614226033379
100-step PSNR deltas   train -0.0003037452697753906, heldout +0.00011444091796875
```

Full 512px seed-2 600-step row:

```text
artifact        research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_suffix_staronly
selected step   600
train PSNR      15.211486339569092
heldout PSNR    13.808026313781738
train loop      88.92246554100711 s
render-only     0.11502037501486484 s
max pair ratio  3.0513058467255063
max tile        90
overflow        0
unstable        0.0
```

Read: suffix-composite is a real deterministic speed improvement over
zero-pruned tile-pair (`112.1208749973448` ms versus `129.20112499705283` ms
in the fixed backward probe; `88.92246554100711` s versus
`113.04000200000155` s in the full 600-step loop). But it is not a quality
promotion: heldout PSNR is `13.808026313781738`, below zero-pruned tile-pair's
`13.866263389587402` and below the `13.9870` multires V-JEPA row. Keep it as
the current deterministic speed candidate, not a seed-expansion quality branch.

Same-wall-clock suffix retry:

```text
artifact        research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_timebudget750_besttrain_tilepair_suffix_staronly
selected step   750
train PSNR      15.51722240447998
heldout PSNR    13.85195541381836
train loop      128.04329337499803 s
render-only     0.12207708400092088 s
max pair ratio  2.9437261710302622
max tile        90
overflow        0
unstable        0.0
```

Read: the extra suffix steps recover most of the 600-step quality gap, but still
do not clear the zero-pruned seed-2 row (`13.866263389587402` heldout) and take
longer than that row's `113.04000200000155` s loop. This rejects
same-wall-clock compensation as the missing quality fix for suffix.

Zero-pruned-time suffix retry:

```text
artifact        research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_timebudget113s_besttrain_tilepair_suffix_staronly
selected step   666
train PSNR      15.376543521881104
heldout PSNR    13.829670906066895
train loop      113.05004737499985 s
render-only     0.11196500100777484 s
max pair ratio  3.0018892148819702
max tile        91
overflow        0
unstable        0.0
```

Read: this is the clean same-time comparison against the zero-pruned 600-step
row (`113.04000200000155` s). Suffix gets slightly better overfit/train PSNR
than zero-pruned at the same wall time (`15.376543521881104` versus
`15.34240198135376`), but still loses heldout (`13.829670906066895` versus
`13.866263389587402`). That makes suffix a useful same-time overfit branch, not
the active heldout-quality branch.

## Key-Sort Segmented Reducer Probe

I added a deterministic segmented reducer:

```text
torch_gsplat_bridge_star_uvt.reduce_sample_bundle_sorted_segments
--uvt-reduction-mode key_sort_segmented_metal
```

The mode preserves the existing stable `(tube_id, key)` sort, then the Metal
kernel binary-searches each tube's contiguous sorted segment and sums only that
range.

Fixed 512px backward rows:

```text
artifact      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_keysort_segmented_scale_rerun5.json
mode          tile_pair + key_sort_segmented_metal
sample median 112.28783300612122 ms
reduce median 4.514999993261881 ms
total median  116.8028329993831 ms

artifact      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_suffix_keysort_segmented_scale.json
mode          tile_pair_suffix + key_sort_segmented_metal
sample median 93.16520800348371 ms
reduce median 5.270957990433089 ms
total median  98.4361659939168 ms
```

Correctness and repeatability checks:

```text
smoke          research_project/benchmarks/results/uvt_gradient_repeatability_probe_keysort_segmented_suffix_smoke_16_2f_t16.json
smoke read     sample/reduction/diagnostic/autograd digests all unique count 1
512 gradient   research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_step0_keysort_segmented_vs_scan_tilepair_suffix.json
512 read       comparison max abs versus key_sort_scan_metal 0.0
20-step repeat research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_suffix_keysort_segmented_repeatability_20steps.json
repeat read    final_state_max_abs 0.0, final train/heldout PSNR spans 0.0
```

Matched full 512px seed-2 suffix row:

```text
artifact        research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_suffix_keysort_segmented_staronly
selected step   600
train PSNR      15.211486339569092
heldout PSNR    13.808026313781738
train loop      171.39199266598735 s
render-only     0.171773458016105 s
max pair ratio  3.0513058467255063
max tile        90
overflow        0
unstable        0.0
```

Read: the segmented reducer proves the lower-level scan is removable from the
fixed backward probe, especially for suffix (`98.4361659939168` ms versus
`112.1208749973448` ms). It does not transfer to full training: the matched
600-step loop is much slower than suffix scan (`171.39199266598735` s versus
`88.92246554100711` s) with identical selected heldout PSNR. Keep it as a
diagnostic rejection. The next speed implementation must fuse or localize the
deterministic reduction instead of sorting and gathering rows as a separate
per-step path.

## Direct-Reduced Suffix Probe

I added a direct-reduced suffix backward path:

```text
torch_gsplat_bridge_star_uvt.tile_pair_suffix_reduced_backward
--uvt-sample-emission-mode tile_pair_suffix_reduced
```

The path runs the suffix tile-pair sample kernel, then reduces each tube inside
Metal by scanning that tube's analytic tile bounds in deterministic
tile-major/slot order. It avoids the explicit Python/MPS `argsort` and
`index_select` used by the keyed reducer path.

Correctness checks:

```text
tiny parity     16px, 2 frames, 16 tubes
tiny read       max abs 0.0 for ma, q, opacity, and color
512 parity      512px, 16 frames, 7168 tubes
512 valid rows  211374
512 read        max abs 0.0 for ma, q, opacity, and color
```

Fixed 512px backward row:

```text
artifact      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_suffix_reduced_scale.json
mode          tile_pair_suffix_reduced
sample median 99.04662500775885 ms
reduce median 0.007499998901039362 ms
total median  99.05412500665989 ms
```

Repeatability check:

```text
artifact     research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_suffix_reduced_repeatability_20steps.json
state read   final_state_0_1.max_abs 0.0, mean_abs 0.0
PSNR read    final train/heldout PSNR spans 0.0
loop times   2.0861335420049727 s, 1.4533952500059968 s
```

Matched full 512px seed-2 suffix row:

```text
artifact        research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_suffix_reduced_staronly
selected step   600
train PSNR      15.211486339569092
heldout PSNR    13.808026313781738
train loop      118.60049016600533 s
render-only     0.14151216701429803 s
max pair ratio  3.0513058467255063
max tile        90
overflow        0
unstable        0.0
```

Read: direct-reduced suffix preserves suffix math exactly and removes the
explicit sort/gather overhead, but it is not the trainer-speed fix. The fixed
backward row is near segmented suffix (`99.05412500665989` ms versus
`98.4361659939168` ms), yet the full 600-step loop is slower than suffix scan
(`118.60049016600533` s versus `88.92246554100711` s) and slower/lower-quality
than zero-pruned tile-pair (`113.04000200000155` s,
`13.866263389587402` heldout). The per-tube support scan is still too much
work.

## Direct Fixed-Point Atomic Rejection

I added `direct_fixedpoint_backward` as the deterministic version of the fast
direct-atomic shape: per-tube gradients are accumulated into `atomic_int`
fixed-point buffers, then converted back to float.

Fixed 512px backward rows:

```text
artifact        research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_directfixedpoint_scale.json
scale           default 1e6, inferred from the pre-metadata artifact command
sample          47.169625002425164 ms
reduce          0.012750009773299098 ms
sample+reduce   47.18237501219846 ms
valid rows      7168
unstable        0.0

artifact        research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_directfixedpoint_scale1e4.json
scale           1e4, inferred from artifact name/command
sample          42.01241700502578 ms
reduce          0.007666007149964571 ms
sample+reduce   42.020083012175746 ms
valid rows      7168
unstable        0.0
```

Repeatability check:

```text
artifact     research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_directfixedpoint_repeatability_20steps.json
state read   final_state_0_1.max_abs 0.0, mean_abs 0.0
PSNR read    final train/heldout PSNR spans 0.0
```

Training gates:

```text
artifact        research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_directfixedpoint_staronly
scale           default 1e6
stopped         nonfinite_loss at step 160
selected step   100
train PSNR      7.606770038604736
heldout PSNR    7.480576515197754
train loop      7.627481291987351 s

artifact        research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed200_besttrain_directfixedpoint_scale1e4_staronly
scale           1e4
selected step   200
train PSNR      9.961743831634521
heldout PSNR    10.693564414978027
train loop      7.114760875003412 s

artifact        research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_directfixedpoint_scale1e4_staronly
scale           1e4
stopped         nonfinite_loss at step 350
selected step   300
train PSNR      10.356242179870605
heldout PSNR    10.548453330993652
train loop      12.131991665999522 s
```

Read: fixed-point atomics hit the speed shape we wanted better than any
deterministic tile-pair path so far (`42-47` ms versus direct atomic's `35.7`
ms), and they are exact-repeatable in same-process probes. They are still not
the usable UVT training rasterizer because the quantized gradient path either
goes nonfinite or stalls far below the normal deterministic rows.

## Direct Split-Fixed-Point Rejection

I added `direct_split_fixedpoint_backward` to test whether the single-scale
fixed-point branch failed because one integer scale had to choose between range
and precision. The new branch accumulates a coarse integer sum and a fine
residual sum for every gradient component.

Parity and fixed backward:

```text
tiny parity     research_project/benchmarks/results/uvt_direct_split_fixedpoint_backward_parity_smoke_16_2f_16t.json
tiny repeat     max_abs 0.0 for ma/q/opacity/color
tiny vs serial  ma 8.702278137207031e-06, q 3.814697265625e-05,
                opacity 1.2874603271484375e-05, color 3.4332275390625e-05

512 parity      research_project/benchmarks/results/uvt_direct_split_fixedpoint_backward_parity_512_16f_7168t_tilet1_cap128.json
512 repeat      max_abs 0.0 for ma/q/opacity/color
512 vs serial   ma 6.341934204101562e-05, q 0.000762939453125,
                opacity 0.0004119873046875, color 0.000102996826171875

512 timing      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_directsplitfixedpoint_scale100_1e6.json
sample          53.006874994025566 ms
reduce          0.008875009370967746 ms
sample+reduce   53.01575000339653 ms
valid rows      7168
unstable        0.0
```

Training gates:

```text
repeatability   research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_directsplitfixedpoint_repeatability_20steps.json
state read      final_state_0_1.max_abs 0.0, mean_abs 0.0
PSNR read       final train/heldout PSNR spans 0.0
final PSNR      train 7.875271797180176, heldout 7.874758720397949

artifact        research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed200_besttrain_directsplitfixedpoint_staronly
stopped         nonfinite_loss at step 120
selected step   100
train PSNR      7.602231979370117
heldout PSNR    7.471946716308594
train loop      8.518700500004343 s
```

Read: split fixed-point keeps exact repeatability and a useful fixed-backward
speed, but it makes the training failure worse than the lower-scale single
fixed-point retry. The fixed-point family should stay rejected unless the next
idea changes optimizer stability rather than only accumulator range/precision.

## Suffix Gradient Diagnosis

I extended `uvt_gradient_repeatability_probe.py` so the local gradient gate can
run `tile_pair_target_bounds` and `tile_pair_suffix`.

Step-0 keyed-vs-suffix:

```text
artifact          research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_step0_withkeys_vs_tilepair_suffix.json
primary           with_keys + key_sort_scan_metal
candidate         tile_pair_suffix + key_sort_scan_metal
max grad delta    6.05359673500061e-09
loss delta        0.0
autograd unique   1 / 1
```

After 600 suffix-pretrain steps:

```text
artifact          research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_step600_suffix_pretrain_withkeys_vs_tilepair_suffix.json
pretrain          600 steps, tile_pair_suffix + key_sort_scan_metal
primary           with_keys + key_sort_scan_metal
candidate         tile_pair_suffix + key_sort_scan_metal
max grad delta    2.3096799850463867e-07
loss delta        0.0
autograd unique   1 / 1
```

Read: suffix's lower 600-step heldout PSNR is not from an obvious local VJP
mistake. The remaining explanation is trajectory/optimizer sensitivity under
the alternate deterministic accumulation path.

## Tile-Pair-Parallel Rejection

I added `tile_pair_parallel_backward_samples` as the next deterministic speed
attempt. It keeps the current compact `(tile, tube slot)` row contract but uses
one Metal threadgroup per row to reduce all pixels in the tile with a fixed tree.

```text
tiny smoke      research_project/benchmarks/results/uvt_backward_breakdown_probe_tilepair_parallel_smoke_16_2f_16t_tilet1.json
rows            117
unstable        0.0
sample+reduce   8.675312994455453 ms

tiny parity     research_project/benchmarks/results/uvt_tile_pair_parallel_backward_parity_smoke_16_2f_16t_tilet1.json
repeat delta    0.0 max_abs for raw samples and reduced gradients
vs tile_pair    max reduced delta 1.52587890625e-05

512 timing      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_parallel_scale.json
sample          119.39916700066533 ms
reduce          21.62637500441633 ms
sample+reduce   141.02554200508166 ms
valid rows      211374
unstable        0.0
```

Read: the branch is deterministic and close to the existing tile-pair gradient,
but it fails the fixed-512 speed gate. It is slower than zero-pruned tile-pair
(`129.20112499705283` ms) and suffix (`112.1208749973448` ms), so I did not run
a 600-step train. The next rasterizer branch needs a different reduction shape,
not another one-threadgroup-per-tile-slot recompute variant.

## Tile-Pair-Grouped Speed Candidate

I added `tile_pair_grouped_backward_samples` as a more plausible tile-local
deterministic speed branch. It launches one threadgroup per tile, sorts the tile
once, and loops tile-local tube slots inside that group while preserving the
same compact `(tile, tube slot)` row keys.

```text
tiny smoke      research_project/benchmarks/results/uvt_backward_breakdown_probe_tilepair_grouped_smoke_16_2f_16t_tilet1.json
rows            117
unstable        0.0

tiny parity     research_project/benchmarks/results/uvt_tile_pair_grouped_backward_parity_smoke_16_2f_16t_tilet1.json
repeat delta    0.0 max_abs for raw samples and reduced gradients
vs tile_pair    max reduced delta 1.52587890625e-05

512 timing      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_grouped_scale.json
sample          68.54158300848212 ms
reduce          21.838707994902506 ms
sample+reduce   90.38029100338463 ms
valid rows      211374
allocated rows  8388608
unstable        0.0
```

This is the first deterministic fixed-backward speed probe below suffix
(`112.1208749973448` ms) and zero-pruned tile-pair (`129.20112499705283` ms).
It still does not approach nondeterministic direct atomic (`35.70937499171123`
ms), but it clears the "different enough to train" speed gate.

The short trajectory and repeatability gates looked clean:

```text
compare artifact  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_grouped_100steps.json
keyed loop        19.755125249997946 s
grouped loop      11.625079500008724 s
state max/mean    0.019117549061775208 / 0.00011119506691881854
train PSNR delta  -0.00024509429931640625
heldout delta     -0.00007534027099609375

repeat artifact   research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_grouped_repeatability_100steps.json
state max/mean    0.0 / 0.0
PSNR spans        train 0.0, heldout 0.0
```

The full train gate is why this is not promoted:

```text
artifact        research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_grouped_staronly
selected step   600
selector        best_train_psnr, no heldout selection
train PSNR      15.220745086669922
heldout PSNR    13.74386978149414
train loop      100.48031524999533 s
render-only     0.11086025099211838 s selected, 0.12215687600837555 s final
metal           max tile 87/89/91, overflow 0, unstable 0.0
```

Read: grouped is a real speed-shape candidate, but it loses the 600-step quality
gate. It is faster than zero-pruned seed 2 (`100.48s` versus `113.04s`) but
below zero-pruned heldout (`13.7439` versus `13.8663`), below suffix heldout
(`13.8080`), and slower than suffix's `88.92s` loop. Do not expand grouped to
seeds 0/1 until a quality-preserving change explains the 600-step miss.

## Grouped Gradient Diagnosis

I ran the same local gradient gate for grouped that cleared the suffix branch.

Step-0 keyed-vs-grouped:

```text
artifact          research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_step0_withkeys_vs_tilepair_grouped.json
primary           with_keys + key_sort_scan_metal
candidate         tile_pair_grouped + key_sort_scan_metal
max grad delta    6.51925802230835e-09
loss delta        0.0
autograd unique   1 / 1
```

After 600 grouped-pretrain steps:

```text
artifact          research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_gradient_step600_grouped_pretrain_withkeys_vs_tilepair_grouped.json
pretrain          600 steps, tile_pair_grouped + key_sort_scan_metal
pretrain loop     105.11219408399484 s
primary           with_keys + key_sort_scan_metal
candidate         tile_pair_grouped + key_sort_scan_metal
max grad delta    1.3746321201324463e-06
loss delta        0.0
autograd unique   1 / 1
```

Read: grouped's lower 600-step heldout PSNR is not from an obvious local VJP
mistake. This points to trajectory/optimizer sensitivity from the alternate
deterministic accumulation path, just like suffix. The next grouped work should
change the optimization trajectory or preserve zero-pruned accumulation order;
do not spend another pass simply expanding grouped seeds.

## Same-Step Overfit Pull

I added one missing paired artifact so the 200-step direct-splat baseline is an
evaluated row rather than an inference from loss logs:

```text
artifact  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed200_besttrain_tilepair_zero_prune_paired
steps     200
STAR      tile_pair + key_sort_scan_metal
splats    fast_mac, 2048 splats, dataset_lens camera projection
```

The durable machine-readable summary is:
`research_project/benchmarks/results/mcam512_same_step_overfit_summary_2026_05_12.json`.

| Row | Steps | Train loop s | Train PSNR | Heldout PSNR | Render-only s |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct splats | 200 | `6.4328010419994825` | `6.988864898681641` | `6.835783004760742` | `0.6536745410121512` |
| STAR zero-pruned | 200 | `37.24962037500518` | `12.694841861724854` | `12.390332221984863` | `0.13297945899830665` |
| STAR suffix | 200 | `26.980708625007537` | `12.701470851898193` | `12.364795684814453` | `0.12207016699539963` |
| STAR grouped | 200 | `27.96819470799528` | `12.674538612365723` | `12.37235164642334` | `0.12086216700845398` |
| direct splats | 600 | `17.273487166996347` | `9.27243709564209` | `8.26813793182373` | `0.32718483400094556` |
| STAR zero-pruned | 600 | `113.04000200000155` | `15.34240198135376` | `13.866263389587402` | `0.11449558299500495` |
| STAR suffix | 600 | `88.90582837500551` | `15.211486339569092` | `13.808026313781738` | `0.1141706669877749` |
| STAR grouped | 600 | `100.46536491700681` | `15.220745086669922` | `13.74386978149414` | `0.11197879099927377` |

Read: for the local 512px/16-frame seed-2 overfit question, STAR is already far
ahead of the direct dynamic splat baseline by 200 equal steps and renders much
faster in this harness. The remaining STAR problem is not "can it beat these
direct splats"; it is whether the rasterizer can keep zero-pruned quality while
recovering suffix/grouped speed, and whether any STAR row can beat the stronger
multires V-JEPA row.

## Tile-Pair Sharedsort Probe

I added `tile_pair_sharedsort` as a narrow rasterizer probe: each tile sorts its
tube slots once in threadgroup memory, then preserves the zero-pruned
tile-pair-style serial pixel accumulation for each compact `(tile, slot)` row.
The intent was to test whether we could remove repeated sort work without
changing the accumulation semantics that made zero-pruned the current quality
branch.

Validation passed at smoke scale:

```text
smoke artifact  research_project/benchmarks/results/uvt_backward_breakdown_probe_tilepair_sharedsort_smoke_16_2f_16t_tilet1.json
parity artifact research_project/benchmarks/results/uvt_tile_pair_sharedsort_backward_parity_smoke_16_2f_16t_tilet1.json
rows            117 / 117
ids/keys delta  0 / 0
reduced delta   3.814697265625e-06 max_abs
```

Current-build 512px timing shows a small win, not a breakthrough:

```text
cap-128 sharedsort  52.655457984656096 ms sample+reduce, 63051 rows
cap-128 tile-pair   56.643062496732455 ms sample+reduce, 63051 rows
cap-256 sharedsort  64.27791700116359 ms sample+reduce
cap-256 tile-pair   69.4723329943372 ms sample+reduce
```

The short trajectory check stayed close:

```text
artifact       research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_sharedsort_100steps.json
state max/mean 0.02207188308238983 / 0.00013591208698926494
train delta    -0.00025272369384765625
heldout delta  -0.00014972686767578125
```

The full STAR-only gate did not promote it:

```text
artifact       research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_sharedsort_staronly
selected step  600
train PSNR     15.138561725616455
heldout PSNR   13.569375991821289
train loop     163.4644198330061 s
render-only    0.15410962598980404 s selected
```

Read: sharedsort is useful evidence because it preserves tile-pair semantics at
smoke scale and gives a small current-build speed win. It is not the fast UVT
training rasterizer we want. Do not spend the next pass on sharedsort seed
expansion; the next rasterizer step needs a larger speed move while preserving
or recovering the zero-pruned 600-step quality.

## Plain Tile-Pair Direct-Reduced Probe

I tested the already-wired plain direct-reduced tile-pair path:

```text
torch_gsplat_bridge_star_uvt.tile_pair_reduced_backward
--uvt-sample-emission-mode tile_pair_reduced
```

This is the zero-pruned tile-pair sample kernel plus a Metal reducer that scans
each tube's analytic tile bounds in deterministic tile-major/slot order. It
bypasses the explicit Python/MPS key sort and gather while keeping the
zero-pruned accumulation math.

Correctness and repeatability looked clean:

```text
parity artifact  research_project/benchmarks/results/uvt_tile_pair_reduced_backward_parity_512_16f_7168t_tilet1_cap128.json
reference        tile_pair + key_sort_scan_metal
candidate        tile_pair_reduced_backward
valid rows       63051
max grad delta   0.0
unstable delta   0.0

repeat artifact  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_reduced_repeatability_20steps.json
state max_abs    0.0
```

The fixed 512px timing rows were the strongest deterministic
quality-preserving timings so far:

```text
cap-128 artifact research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_reduced_scale.json
cap-128 total    40.33462500956375 ms sample+reduce

cap-256 artifact research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap256_tilepair_reduced_scale.json
cap-256 total    46.68695799773559 ms sample+reduce
```

The full trainer loop is the rejection:

```text
artifact       research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_reduced_staronly
selected step  600
train PSNR     15.34240198135376
heldout PSNR   13.866263389587402
train loop     132.13478325000324 s
render-only    0.12436912400880828 s final
```

Read: this is an important negative/partial. It proves the zero-pruned quality
branch can bypass the explicit sort/gather and remain exact, but the per-tube
support scan grows too much during real training. Do not promote or seed-expand
`tile_pair_reduced` as-is; the next direct-reduced attempt needs a tile-local
or otherwise non-per-tube-scan reducer.

## Tile-Pair Fixed-Point Atomic Probe

I added a tile-slot version of the fixed-point idea:

```text
torch_gsplat_bridge_star_uvt.tile_pair_fixedpoint_backward
--uvt-sample-emission-mode tile_pair_fixedpoint
```

The intent was to keep deterministic integer atomics while quantizing only once
per `(tile, tube slot)` sum, instead of quantizing every direct per-pixel/tube
contribution.

The implementation compiled and the small gates worked:

```text
tiny timing     research_project/benchmarks/results/uvt_backward_breakdown_probe_tilepair_fixedpoint_smoke_16_2f_16t_tilet1.json
tiny total      4.338770508184098 ms sample+reduce

512 parity      research_project/benchmarks/results/uvt_tile_pair_fixedpoint_backward_parity_512_16f_7168t_tilet1_cap128.json
max grad delta  6.103515625e-05
valid rows      63051

repeatability   research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_fixedpoint_repeatability_20steps.json
state max_abs   0.0
```

Fixed-size 512px timing was plausible but noisy:

```text
cap-128 artifact research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_fixedpoint_scale.json
cap-128 total    64.06379198597278 ms sample+reduce

cap-256 artifact research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap256_tilepair_fixedpoint_scale_rerun_warm3.json
cap-256 total    43.17741599516012 ms sample+reduce
```

The training gates reject it:

```text
artifact       research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed200_besttrain_tilepair_fixedpoint_staronly
scale          1e6
stopped        nonfinite_loss at step 100
selected step  0
train PSNR     7.657606363296509
heldout PSNR   7.620186805725098

artifact       research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_fixed200_besttrain_tilepair_fixedpoint_scale1e4_staronly
scale          10000
stopped        nonfinite_loss at step 180
selected step  100
train PSNR     7.5389885902404785
heldout PSNR   7.422435760498047
```

Read: tile-pair fixed-point is deterministic and has a plausible isolated speed
shape, but it is not trainable under the current optimizer/scale choices. This
does not solve the fast UVT training rasterizer; keep it as a rejected
fixed-point diagnostic unless a separate stability plan changes the training
behavior.

## Tile-Pair Float-Atomic Probe

I added the float-atomic version of the tile-slot direct-gradient idea:

```text
torch_gsplat_bridge_star_uvt.tile_pair_atomic_backward
--uvt-sample-emission-mode tile_pair_atomic
```

This keeps the zero-pruned per-slot accumulation math, then atomically adds one
nonzero tile-slot gradient into the global tube gradients. It is the direct
test of whether reducing atomic count from per-pixel/tube contributions to
per-tile-slot sums makes the direct-atomic idea stable enough without
fixed-point quantization.

The implementation compiled and the tiny smoke passed:

```text
tiny timing     research_project/benchmarks/results/uvt_backward_breakdown_probe_tilepair_atomic_smoke_16_2f_16t_tilet1.json
tiny total      4.192124994006008 ms sample+reduce
```

The 512px parity check is close to zero-pruned tile-pair, but not exact because
the final cross-tile accumulation is still float-atomic:

```text
parity artifact research_project/benchmarks/results/uvt_tile_pair_atomic_backward_parity_512_16f_7168t_tilet1_cap128.json
reference       tile_pair + key_sort_scan_metal
candidate       tile_pair_atomic_backward
valid rows      211374
grad_ma max     1.0013580322265625e-05
grad_q max      9.1552734375e-05
opacity max     6.103515625e-05
color max       1.52587890625e-05
unstable delta  0
```

The fixed 512px timing and repeatability gates reject promotion:

```text
timing artifact research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_atomic_scale.json
total median    93.08808401692659 ms sample+reduce

repeat artifact research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_atomic_repeatability_20steps.json
state max_abs   8.440017700195312e-05
state mean_abs  1.8583021821021768e-07
```

Read: tile-pair float atomics reduce direct-atomic drift slightly
(`0.00008440017700195312` versus direct atomic's 20-step
`0.00010570883750915527`) but still miss exact repeatability, and the fixed
timing is not better than grouped. Do not run a 600-step quality row unless the
acceptance bar changes to bounded nondeterministic drift; under the current
deterministic-promotion bar this branch is diagnostic-only.

## Direct-Reduced Slot-Count Micro-Optimization Rejection

I tested a small exact optimization to the direct-reduced tile-pair reducer:
pass `tile_counts` into `reduce_tile_pair_bounds_scan` and scan only the live
slots for each candidate tile instead of all `STAR_TILE_CAPACITY` slots.

The math stayed exact:

```text
parity artifact research_project/benchmarks/results/uvt_tile_pair_reduced_backward_parity_512_16f_7168t_tilet1_cap128_slotcount.json
candidate       tile_pair_reduced_backward_slotcount
reference       tile_pair + key_sort_scan_metal
max grad delta  0.0
unstable delta  0
```

The timing rejected it:

```text
cap-128 artifact research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_reduced_slotcount_scale.json
cap-128 total    105.39762600092217 ms sample+reduce

cap-256 artifact research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap256_tilepair_reduced_slotcount_scale.json
cap-256 total    116.2034579901956 ms sample+reduce
```

That is much worse than the previous direct-reduced fixed rows
(`40.33462500956375` ms cap-128, `46.68695799773559` ms cap-256), so I reverted
the code change. Read: loading `tile_counts` inside the per-tube support-scan
loops is more expensive than scanning empty capacity slots. Do not repeat this
micro-optimization.

## Parallel Direct-Reduced Per-Tube Scan Rejection

I added a separate opt-in reducer branch:

```text
torch_gsplat_bridge_star_uvt.tile_pair_reduced_parallel_backward
--uvt-sample-emission-mode tile_pair_reduced_parallel
```

It preserves the existing direct-reduced setup, but changes the final
`reduce_tile_pair_bounds_scan` stage to `reduce_tile_pair_bounds_scan_parallel`:
one Metal threadgroup is launched per tube, each lane scans a strided subset of
that tube's analytic tile/support rows, and the group reduces in a fixed tree.
The serial `tile_pair_reduced` path remains unchanged.

The tiny smoke and parity gates pass:

```text
timing artifact research_project/benchmarks/results/uvt_backward_breakdown_probe_tilepair_reduced_parallel_smoke_16_2f_16t_tilet1.json
total median    4.83741700736573 ms sample+reduce
rows            16 direct tube-gradient rows
unstable        0.0

parity artifact research_project/benchmarks/results/uvt_tile_pair_reduced_parallel_backward_parity_smoke_16_2f_16t_tilet1.json
reference       tile_pair_reduced_backward
candidate       tile_pair_reduced_parallel_backward
max grad delta  0.0
unstable delta  0
```

The fixed 512px speed gate rejects it:

```text
artifact        research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_reduced_parallel_scale.json
total median    151.81316700181924 ms sample+reduce
sample median   151.80191700346768 ms
rows            7168 direct tube-gradient rows
unstable        0.0
```

This is slower than serial direct-reduced (`40.33462500956375` ms), grouped
(`90.38029100338463` ms), suffix (`112.1208749973448` ms), and the older
zero-pruned fixed row (`129.20112499705283` ms). Read: threadgroup-parallelizing
the per-tube support scan is the wrong reduction shape; the launch/barrier cost
dominates. Do not run a 20-step repeatability or 600-step quality row for this
branch unless a later patch changes the reducer shape again.

## Plain Tile-Pair + Index-Add Reducer Check

I tested an existing composition rather than adding a new kernel:

```text
--uvt-sample-emission-mode tile_pair
--uvt-reduction-mode index_add
```

The purpose was to separate key-sort overhead from tile-pair sample-emission
cost. This path is not deterministic under the current bar, but it is useful as
a speed diagnostic because it avoids key sort, gather, and scan.

The fixed 512px row is:

```text
artifact      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_indexadd_scale.json
sample median 113.21250000037253 ms
reduce median 4.114499999559484 ms
total median  117.32699999993201 ms sample+reduce
rows          211374 valid tile-pair rows
unstable      0.0
```

Read: sort removal helps only a little. Plain tile-pair + index-add beats the
old plain tile-pair + keyed scan row (`129.20112499705283` ms) by removing
roughly the reducer overhead, but it is still slower than grouped
(`90.38029100338463` ms) and suffix (`112.1208749973448` ms). The sample
emission itself is the dominant cost. Do not spend repeatability or quality
runs on this path unless the acceptance bar changes away from deterministic
promotion or the emission kernel changes.

## Grouped Tile-Pair + Index-Add Reducer Check

I also tested the best deterministic speed-shape emission kernel with the cheap
MPS reducer:

```text
--uvt-sample-emission-mode tile_pair_grouped
--uvt-reduction-mode index_add
```

The fixed 512px timing row is:

```text
artifact      research_project/benchmarks/results/uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_tilepair_grouped_indexadd_scale.json
sample median 72.39775000198279 ms
reduce median 4.092707997187972 ms
total median  76.49045799917076 ms sample+reduce
rows          211374 valid compact rows
unstable      0.0
```

This is the useful lower-bound measurement: compared with grouped +
`key_sort_scan_metal` (`90.38029100338463` ms), the sort/gather reducer costs
about `14` ms on this window. That does not make it promotable, because the
repeatability row fails at length.

```text
20-step artifact   research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_grouped_indexadd_repeatability_20steps.json
20-step state      max 2.4080276489257812e-05, mean 1.0973382943900235e-07
20-step PSNR span  train 0.0, heldout 0.0

100-step artifact  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_grouped_indexadd_repeatability_100steps.json
100-step state     max 0.017633624374866486, mean 0.0000700580161979555
100-step PSNR span train 0.0000095367431640625, heldout 0.00003910064697265625

600-step artifact  research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_tilepair_grouped_indexadd_repeatability_600steps.json
600-step state     max 1.4393987655639648, mean 0.11432085718427386
600-step PSNR span train 0.2850780487060547, heldout 0.08693218231201172
600-step heldout   13.708361625671387 vs 13.621429443359375
```

Read: grouped + index-add is faster than grouped keyed scan, but it is not
deterministic enough for STAR reporting and drifts worse than direct atomic by
600 steps. Keep the row as a speed lower bound only; do not spend a quality run
on it.

## Grouped Tile-Pair + Compensated Keyed Scan

Before writing another kernel, I tested whether the existing compensated final
scan helps the grouped trajectory:

```text
--uvt-sample-emission-mode tile_pair_grouped
--reduction-mode key_sort_compensated_scan_metal
```

The 100-step matched replay is:

```text
artifact      research_project/benchmarks/results/mcam512_s2_t320_view_shuffled_cycle_mode_compare_keyed_vs_tilepair_grouped_keysort_compensated_100steps.json
state max     0.03128485009074211
state mean    0.0001310603622446901
train delta   -0.000007152557373046875 dB grouped minus keyed
heldout delta -0.00026798248291015625 dB grouped minus keyed
train loop    keyed 21.543105542004923 s, grouped 11.899838832992828 s
```

This is worse than grouped with the normal keyed scan at the same gate:
`0.019117549061775208` / `0.00011119506691881854` final state max/mean abs
delta. Read: final-scan compensation is not the missing grouped quality fix.
Do not run a 600-step compensated grouped row.

## 512px Direct-Reduced Train-Step Timing Bracket

The fixed backward row made `tile_pair_reduced` look promising, but the full
600-step row was slower than zero-pruned. I added short 512px train-step timing
rows to separate early-step reducer speed from support growth.

Unregularized zero-pruned:

```text
artifact       research_project/benchmarks/results/uvt_train_step_timing_probe_512_16f_7168_s0125_t20_tilet1_cap256_tilepair_keysortscan_20steps.json
mode           tile_pair + key_sort_scan_metal
total median   659.4660414993996 ms
backward med   640.212916994642 ms
max total      3066.0262500023236 ms
sample rows    251099 -> 1301037
tile proxy     66.37944793701172 -> 507.6557312011719
```

Unregularized direct-reduced:

```text
artifact       research_project/benchmarks/results/uvt_train_step_timing_probe_512_16f_7168_s0125_t20_tilet1_cap256_tilepair_reduced_20steps.json
mode           tile_pair_reduced
total median   581.4878124947427 ms
backward med   560.606311999436 ms
max total      3004.9175830063177 ms
tile proxy     66.37944793701172 -> 507.6448669433594
final recon    0.028355862945318222
```

Read: direct-reduced is faster in the short loop, but both paths are dominated
by support growth. The target `7000` regularizer setting is inactive on this
proxy scale.

I then bracketed direct-reduced with real tile-load targets:

```text
artifact       research_project/benchmarks/results/uvt_train_step_timing_probe_512_16f_7168_s0125_t20_tilet1_cap256_tilepair_reduced_20steps_tileloadreg0003_target200.json
target         200
total median   557.4091669986956 ms
backward med   540.076229000988 ms
max total      2695.660625002347 ms
tile proxy max 481.34552001953125
final recon    0.02667274884879589

artifact       research_project/benchmarks/results/uvt_train_step_timing_probe_512_16f_7168_s0125_t20_tilet1_cap256_tilepair_reduced_20steps_tileloadreg0003_target100.json
target         100
total median   527.294957995764 ms
backward med   509.83279200590914 ms
max total      2156.30816599878 ms
tile proxy max 397.0201721191406
final recon    0.02127799764275551
```

Read: target `100`, weight `0.003` is a real 512px single-video speed-control
lead. It is not a promoted recipe yet because this is only a 20-step timing
probe; the next fair gate is a same-step 50/200-step overfit row with render
timing.

## 512px Same-Step Single-Video Overfit

I ran the fair overfit gate for the target-100 direct-reduced branch:

```text
artifact      research_project/benchmarks/results/video_fit_single_overfit_512_16f_50steps_7168uvt_lr012_s0125_t20_tilet1_cap256_tileloadreg0003_target100_tilepair_reduced_uvtonly_renderbench10_metal_tile.json
steps         50
PSNR          19.731093645095825
MSE           0.010638750158250332
wall clock    79.00857495800301 s
render median 29.479833006917033 ms
tile proxy    274.5934143066406

artifact      research_project/benchmarks/results/video_fit_single_overfit_512_16f_200steps_7168uvt_lr012_s0125_t20_tilet1_cap256_tileloadreg0003_target100_tilepair_reduced_uvtonly_renderbench10_metal_tile.json
steps         200
PSNR          22.267677783966064
MSE           0.00593242421746254
wall clock    265.34595041599823 s
render median 15.4583960029413 ms
tile proxy    252.0620880126953
```

Then I ran a matched per-frame splat baseline at 512px:

```text
artifact          research_project/benchmarks/results/video_fit_single_overfit_512_16f_200steps_448pf_videoinit_s0125_op07_strat_fastmac_lr032_skipuvt_renderbench10.json
steps             200
splats/frame      448
total splats      7168
init              video_samples, stratified, precision 0.125, opacity 0.7
backend           fast_mac
fast cap          2048
LR                0.32
PSNR              25.22531270980835
MSE               0.003002400975674391
wall clock        14.454584291990614 s
render median     15.65556249988731 ms
```

I first tried `--per-frame-fast-max-pairs 4096`, but the run failed before
training because the current `v6_refined` fast-mac build is compiled with cap
`2048`. The successful row uses that compiled cap.

Read: on 512px single-video overfit, controlled direct-reduced UVT loses to the
same-total-splat per-frame baseline on quality and training speed, and only ties
render speed. Do not promote target-100 direct-reduced as the overfit recipe.
The next overfit work needs a representation or optimization quality change,
not another speed-only reducer toggle.

## 512px Appearance-Refine Check

I tested whether the direct-reduced quality miss is mostly appearance rather
than geometry/support:

```text
artifact      research_project/benchmarks/results/video_fit_single_overfit_512_16f_50steps_app50_7168uvt_lr012_s0125_t20_tilet1_cap256_tileloadreg0003_target100_tilepair_reduced_uvtonly_renderbench10_metal_tile.json
main steps    50
app steps     50
app LR        0.04
PSNR          20.100538730621338
app loss      0.010638750158250332 -> 0.009771162644028664
wall clock    159.13381716699223 s
render median 22.466125003120396 ms
tile proxy    275.2648010253906
```

Plain 50-step target-100 direct-reduced UVT was PSNR `19.731093645095825` in
`79.00857495800301` s. So `50` appearance-only steps add only about
`+0.3694` dB while doubling wall time. Read: the 512px gap is not primarily
color/opacity refinement. The next quality mechanism needs to change geometry,
support, or motion.

## 512px Staged-LR Overfit Check

I tested whether the earlier 128px staged-LR improvement transfers to this 512px
target-100 direct-reduced branch:

```text
artifact      research_project/benchmarks/results/video_fit_single_overfit_512_16f_200steps_lr012_to004_step100_7168uvt_s0125_t20_tilet1_cap256_tileloadreg0003_target100_tilepair_reduced_uvtonly_renderbench10_metal_tile.json
steps         200
LR schedule   0.12 -> 0.04 at step 100
PSNR          21.792545318603516
MSE           0.006618285086005926
wall clock    294.75141954098945 s
render median 22.178207997058053 ms
tile proxy    257.7630310058594
```

This is worse than the constant-LR 200-step UVT row (`22.267677783966064` PSNR,
`265.34595041599823` s, `15.4583960029413` ms render median) and remains far
behind the matched per-frame splat baseline (`25.22531270980835` PSNR,
`14.454584291990614` s, `15.65556249988731` ms render median). Read: staged LR
does not rescue the 512px target-100 direct-reduced branch.

## 512px Stratified-Init Fairness Check

The winning per-frame splat baseline uses stratified video samples, while the
target-100 UVT rows above used random video samples. I ran a UVT-only control
that changes only the UVT sample placement:

```text
artifact      research_project/benchmarks/results/video_fit_single_overfit_512_16f_200steps_7168uvt_lr012_s0125_t20_op07_strat_tilet1_cap256_tileloadreg0003_target100_tilepair_reduced_uvtonly_renderbench10_metal_tile.json
steps         200
sample mode   stratified
PSNR          22.290687561035156
MSE           0.005901076830923557
wall clock    298.500004083995 s
render median 22.068042002501898 ms
tile proxy    251.14137268066406
```

This is only `+0.0230` dB over the random UVT row (`22.267677783966064` PSNR)
and remains about `2.9346` dB below the matched per-frame stratified baseline
(`25.22531270980835` PSNR). Read: random-versus-stratified initialization is not
the main explanation for the 512px overfit gap.
