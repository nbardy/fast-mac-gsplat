# STAR-UVT State Review

Date: 2026-05-12

## Question

Step back and answer:

- What are we doing?
- What is happening?
- What do we need or want?
- Did we ever get a fast sublinear UVT rasterizer?

This note uses existing local artifacts only. No new training was launched for
this review.

## Short Answer

We have a real sparse UVT forward renderer. We do not yet have a promoted fast,
deterministic, quality-preserving UVT training rasterizer.

The forward/render side is not the main blocker anymore. The training-side
backward path is the blocker. The fastest useful backward path is direct float
atomics, but it is not exact-repeatable at 512px/600 steps. The exact
deterministic paths are slower or change the optimization trajectory enough to
lose quality.

## What We Are Doing

The current research lane is testing whether STAR-UVT/worldtube primitives can
replace or beat sliced dynamic splats on local video fitting and multicam
heldout evaluation, while preserving the speed thesis:

1. Represent video as moving screen/world tubes rather than independent
   per-frame splats.
2. Render through UVT tiles, ideally with fewer tile/tube pairs than summed
   per-frame tile/splat pairs.
3. Train against the same local DeepView split and compare against direct
   dynamic splats and the V-JEPA F32 baseline.
4. Promote only if the path is fast, high quality, and repeatable.

The current clean comparison setup is the DeepView goodset:

- train cameras: `camera_0006`, `camera_0014`
- heldout camera: `camera_0005`
- frames: `16`
- lens/projection: `deepview_models_relative_opencv_fisheye` with
  `dataset_lens`
- key 512px STAR recipe: `320` tubes, temporal window `1`,
  `view_shuffled_cycle`, fixed `600` steps or matched time budgets

## What Is Happening

STAR is clearly better than direct dynamic splats on this local split, but the
best fast branch and the best deterministic branch are different branches.

### Same-step overfit

Artifact:
`research_project/benchmarks/results/mcam512_same_step_overfit_summary_2026_05_12.json`

At 200 steps:

| row | train loop s | train PSNR | heldout PSNR | render-only s |
|---|---:|---:|---:|---:|
| direct splats fast-mac | `6.4328010419994825` | `6.988864898681641` | `6.835783004760742` | `0.6536745410121512` |
| STAR zero-pruned tile-pair | `37.24962037500518` | `12.694841861724854` | `12.390332221984863` | `0.13297945899830665` |
| STAR suffix | `26.980708625007537` | `12.701470851898193` | `12.364795684814453` | `0.12207016699539963` |
| STAR grouped | `27.96819470799528` | `12.674538612365723` | `12.37235164642334` | `0.12086216700845398` |

Read: STAR overfits this single-video/local view task much better at equal
steps and renders faster, but training still costs more wall time than direct
splats.

### Fast branch

Artifacts:

- `mcam512_s{0,1,2}_t320_view_shuffled_cycle_fixed600_besttrain_directatomic_paired/`
- `mcam512_s2_t320_view_shuffled_cycle_directatomic_repeatability_600steps.json`
- `uvt_backward_breakdown_probe_512_16f_7168_s0125_t20_tilet1_cap128_directatomic_scale.json`

The 512px direct-atomic matrix is the strongest fast branch:

| seed | STAR train loop s | STAR heldout | direct splat heldout | STAR render s | direct render s |
|---:|---:|---:|---:|---:|---:|
| 0 | `30.19863641700067` | `13.669089317321777` | `8.247410774230957` | `0.11950616599642672` | `0.719359834001807` |
| 1 | `23.72993458300334` | `13.904035568237305` | `8.256668090820312` | `0.11106141699565342` | `0.33759995800210163` |
| 2 | `21.192358166001213` | `13.819857597351074` | `8.267980575561523` | `0.11514749999332707` | `0.3579017909942195` |

This clears the V-JEPA F32 heldout reference `13.6248` on all three seeds.
The fixed 512px backward probe is also fast: direct atomics reduce to tube
gradients with median sample+reduce `35.70937499171123` ms.

But direct atomics are not promotable yet. The 512px/600-step same-process
repeatability probe has final state max/mean abs delta
`1.431039810180664` / `0.11831939475876944`, and final heldout span
`0.03300189971923828`. That is useful for exploration, not a deterministic
reporting path.

### Deterministic branch

Artifacts:

- `mcam512_s2_t320_view_shuffled_cycle_keysortscan_repeatability_600steps.json`
- `mcam512_s2_t320_view_shuffled_cycle_tilepair_zero_prune_repeatability_600steps.json`
- `mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_zero_prune_paired/`

The keyed per-pixel path is exact-repeatable at 512px/600 steps:

- final state max abs delta: `0.0`
- final heldout PSNR span: `0.0`
- heldout PSNR: `13.75709342956543`
- train loops: `221.05617970800085` s and `206.5364632500059` s

That proves the MPS drift can be controlled, but it is far too slow.

The zero-pruned tile-pair path is the current exact deterministic quality
reference:

- selected step: `600`
- train loop: `113.04000200000155` s
- train/heldout PSNR: `15.34240198135376` / `13.866263389587402`
- direct splat heldout PSNR in the paired row: `8.26813793182373`
- render-only: `0.11670004200277617` s
- repeatability final state max abs delta: `0.0`

This is good quality and exact repeatability, but still not the fast training
path we wanted.

### Faster deterministic probes

Artifacts:

- `mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_grouped_staronly/`
- `mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_suffix_staronly/`
- `mcam512_s2_t320_view_shuffled_cycle_timebudget113s_besttrain_tilepair_suffix_staronly/`
- `mcam512_s2_t320_view_shuffled_cycle_fixed600_besttrain_tilepair_reduced_staronly/`

Current deterministic branch read:

| mode | train loop s | selected step | train PSNR | heldout PSNR | render-only s |
|---|---:|---:|---:|---:|---:|
| zero-pruned tile-pair | `113.04000200000155` | `600` | `15.34240198135376` | `13.866263389587402` | `0.11670004200277617` |
| grouped | `100.48031524999533` | `600` | `15.220745086669922` | `13.74386978149414` | `0.11086025099211838` |
| suffix fixed-600 | `88.92246554100711` | `600` | `15.211486339569092` | `13.808026313781738` | `0.11502037501486484` |
| suffix same `113s` time | `113.05004737499985` | `666` | `15.376543521881104` | `13.829670906066895` | `0.11196500100777484` |
| direct-reduced | `132.13478325000324` | `600` | `15.34240198135376` | `13.866263389587402` | `0.2910830830078339` |

Read: suffix is a useful same-time overfit branch: at the same `113s` budget it
beats zero-pruned train PSNR by about `+0.0341` dB. It loses heldout by about
`-0.0366` dB, so it is not the quality/reporting branch.

Direct-reduced preserves zero-pruned quality in the saved STAR-only run, and
its fixed 512px isolated sample+reduce timing is promising at
`40.33462500956375` ms, but the full trainer loop is worse than zero-pruned.
That makes it a partial/negative result, not a solved path.

Continuation update: the obvious per-tube parallel version is also negative.
`tile_pair_reduced_parallel_backward` matches serial direct-reduced exactly at
tiny smoke scale, but the fixed 512px timing is `151.81316700181924` ms
sample+reduce, worse than serial direct-reduced, grouped, suffix, and
zero-pruned fixed rows. Do not spend a repeatability or quality run on that
branch.

Continuation update: removing the keyed reducer from grouped tile-pair is a
useful speed lower bound but not a training answer. `tile_pair_grouped +
index_add` reaches `76.49045799917076` ms sample+reduce on the fixed 512px
window, faster than grouped keyed scan at `90.38029100338463` ms. The 600-step
repeatability row rejects it: final state max/mean abs delta
`1.4393987655639648` / `0.11432085718427386`, train PSNR span
`0.2850780487060547`, and heldout PSNR span `0.08693218231201172`. Do not spend
a quality row on this branch.

## Did We Get A Fast Sublinear UVT Rasterizer?

For forward rendering: yes, at least as a measured research prototype.

Artifact:
`uvt_forward_speed_probe_256_16f_7168_s0125_t20_tilet4_cap128_metalonly.json`

At 256px/16 frames/7168 tubes with `tile_t=4`:

- Metal render median: `6.742891700014297` ms
- UVT tile/tube pair ratio vs summed per-frame tile/splat pairs:
  `0.6304844910456923`
- UVT pairs: `175641`
- summed per-frame pairs: `278581`
- overflow: `0`
- unstable tile fraction: `0.0`

That is the sublinear forward evidence.

For training/backward: no, not yet.

The trainer does not automatically inherit the forward win. Earlier probes
showed the backward path can still emit per-pixel compact sample rows; changing
tile shape can reduce raw forward pairs without reducing backward rows. The
current exact deterministic training branches are still slower than the fast
direct-atomic branch, and the direct-atomic branch is not repeatable enough for
promotion.

So the honest status is:

```text
fast sparse forward UVT rasterizer: yes
fast deterministic sublinear UVT training rasterizer: no
```

## What We Need

The next promoted path has to satisfy all of these:

1. Same 512px/16-frame goodset split and same train/heldout camera contract.
2. Repeatability at 512px/600 steps: final state max abs delta `0.0` or a
   deliberately accepted, measured tolerance with no quality drift.
3. Heldout quality at least comparable to the current deterministic
   zero-pruned row, not just direct splats.
4. Train-loop speed closer to direct atomic than keyed/zero-pruned paths.
5. Render timing remains clean and faster than direct dynamic splats.
6. Zero overflow and near-zero unstable tile fraction.

The implementation target is not another broad quality sweep. It is a
quality-preserving deterministic backward that keeps tile-pair/per-tube
compactness while avoiding direct-atomic nondeterminism and avoiding
per-pixel-row materialization.

Good next kernel directions:

- ordered or compensated tile-pair accumulation that preserves the keyed path's
  trajectory better;
- a different deterministic direct-reduced shape that avoids both per-step
  sorting and per-tube support scanning;
- a trajectory replay gate that compares the new path against keyed or
  zero-pruned over 100/200/600 steps before spending more full benchmark time.

Branches to keep separate:

- direct atomic: fast exploration branch, not reporting/promotion;
- keyed per-pixel: exact correctness fallback, too slow;
- zero-pruned tile-pair: current deterministic quality reference;
- suffix/grouped: speed/overfit probes, not heldout-quality references;
- grouped/index-add: speed lower bound, rejected by 600-step repeatability;
- fixedpoint/float tile-pair atomics: rejected diagnostic probes.

## Current Decision

Do not claim "we solved STAR-UVT speed." We solved enough forward/render speed
to move the blocker. The blocker is now deterministic training backward.

The next useful action is to work on a deterministic compact backward variant
with a tight gate:

```text
tiny parity -> 512px fixed-window parity/timing -> 20-step repeatability
  -> 100-step trajectory replay -> 600-step repeatability/quality
```

Only after that should we rerun a broad 512px or fuller-resolution quality
matrix.
