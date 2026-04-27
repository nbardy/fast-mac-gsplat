# 2026-04-26 Fast-Mac Variant Audit

Scope: `/Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat`

This is the current map of the renderer variants, what each one proves, what it
does not prove, and what is still on the table. It is intentionally practical:
which code path should be used, which variants should stay as evidence, and
what kernel work remains before hardware rasterization can become a training
path.

## Current Answer

The repo has three separate tracks that should not be collapsed:

| Track | Current best | Why |
|---|---|---|
| Training/backward today | `variants/v8` as the best measured fast-mac training base; `variants/v5` as the currently integrated Dynaworld adapter path | V8 keeps the proven v6-style compute math and removes one bridge-side sync. V5 is stable, batched, and is what Dynaworld currently routes through. |
| Hardware eval / render-pass research | `variants/v9_hw_output_planes_probe` and `variants/v9_hw_tile_exact_probe` | Output interop works. RGBA16F eval is promising. Exact imageblock `C/T` state now works in a probe, but it is not V8 parity yet. |
| Historical hardware-backward line | `variants/v7_tiled_capture` and `variants/v7_hybrid_v5style` as evidence only | V7.2 fixed the worst front-K capture shape, but 4K training is still state/CPU bound. V7.3 becomes practical only by routing training back to compute. |

The important status:

- Hardware raster output-to-MPS interop is no longer the main unknown.
- Fixed-function hardware blending is not a V8-compatible multi-splat renderer.
- Tile/imageblock `C/T` state is now plausible, but still needs real projected Gaussian quads fed from V8 bins.
- Full training gradients should stay on V8-style compute replay until the hardware path emits V8-equivalent sorted bins and candidate-prefix stop counts.
- The immediate local dirty change is the v5 `inputs_sorted_by_depth` path, which skips forward depth sort and backward batch unsort when the caller already supplies sorted inputs. The Dynaworld parent adapter now defaults this on because it generates monotonic `_rank_depths(...)` for fast-mac inputs.

## Current Decision Table

| Use case | Use | Avoid | Reason |
|---|---|---|---|
| Dynaworld training right now | `variants/v5` through `src/train/renderers/fast_mac.py`, with `inputs_sorted_by_depth=True` because the parent generates rank-sorted depths | V7 hardware train paths | V5 is the integrated path and the presorted change removes the sort/unsort cost without changing math. |
| Fast-mac training baseline to develop next | `variants/v8` / `v8_direct` | Rewriting around V7 hardware backward | V8 has the best measured 4K/64K forward+backward row and keeps exact compute replay. |
| Large matrix comparison | `v6_direct`, `v6_upgrade_direct`, `v6_refined_direct`, `v8_direct`, `v5_batched` | Only one smoke benchmark | V6 family still wins pockets. Keep matrix evidence before promoting. |
| Eval preview / render-pass speed exploration | `v9_hw_output_planes_probe` | Claiming it as training parity | RGBA16F eval can be fast, but fixed blending lacks `T`, stop count, and exact backward state. |
| Hardware forward state exploration | `v9_hw_tile_exact_probe` | ICB execution in shared benchmarks | Exact imageblock C/T plus tile stop count is the useful next primitive. ICB crashed before and remains fenced. |
| CUDA future | `v9_cuda_compute_first` as source scaffold | Porting Metal graphics-raster assumptions directly | CUDA should be compute-first: fused projection, tile count/scan/sort, block-per-tile forward/backward. |

## Variant Audit

| Variant | What it is | Pros | Cons / risks | Status / next |
|---|---|---|---|---|
| root `torch_gsplat_bridge_fast` / `v2_fastpath` | Older single-image compute fast path | Low-overhead baseline; useful for historical comparison | Single-image only; older architecture; not the current training target | Keep only as benchmark baseline. |
| `variants/v3` / `v3_candidate` | Single-image tile compute renderer with saved forward sorted order | Strong measured B=1 large-scene training path; saved sorted IDs made F+B faster | No native batch API; not Dynaworld's current adapter | Preserve as B=1 reference and source of saved-order lesson. |
| `variants/v5` / `v5_batched` | Batched projected-2D renderer with eval/train split, overflow fallback, saved sorted IDs | Current Dynaworld path; stable batched API; saturated-backward barrier bug fixed; new presorted option removes redundant sort/unsort | B=1 is not always fastest; still has CPU-visible sort work unless caller marks inputs sorted; older than v8 metadata split | Finish by committing presorted path, re-run Dynaworld timing, and keep as stable production adapter until v8 is wired. |
| `variants/v6` / `v6_direct`, `v6_auto` | Batch-focused v5 successor with direct and active-tile paths | Direct path is a strong baseline; active path can help some sparse/overflow cases; correct tile/backward shape | Active scheduling can regress badly on saturated 4K because fixed overhead and output prefill dominate | Keep in matrix; default active policy should stay conservative. |
| `variants/v6_upgrade` | Preserved v6-upgrade handoff | Wins meaningful 64K/backward-heavy pockets; 960-cell matrix completed cleanly | Does not broadly beat local v6; active policy still unstable | Do not replace local v6 wholesale. Port only mechanisms that survive targeted ablation. |
| `variants/v6_refined` | Preserved refined v6 handoff with local saturated-backward safety fix | Useful additional v6-family comparator; same stable API | Mostly overlaps `v6_upgrade` after local fixes; timing differences may be noise | Keep in benchmark matrix, not a default. |
| `variants/v7` / `v7_hardware` | First Metal graphics render-pipeline forward plus compute replay backward | Proves Torch custom op plus render-pipeline forward can match tiny reference | CPU/shared-buffer staging and render texture readback; 4K backward is not viable | Historical proof only. Do not use for training. |
| `variants/v7_finished` | Finished v7 handoff preserved beside local v7 | Forward can be benchmarked; some forward wins | Gradients are materially wrong in local checks | Do not use for training. Keep as source handoff evidence. |
| `variants/v7_frontk` | V7.1 hardware forward plus per-pixel front-K state | Correct small MPS/reference gradients; proves exact front-K backward idea | Capture scales like pixels x splats and is tens of seconds at 4K | Superseded by tiled capture for hardware-backward experiments. |
| `variants/v7_tiled_capture` | V7.2 tile-bin front-K capture and overflow replay | Fixes the worst V7.1 capture shape; exact small checks pass; some 512/6K wins | Still CPU/state/readback bound; 4K training loses badly to compute | Preserve as evidence for tiled state, not default training. |
| `variants/v7_hybrid_v5style` | V7.3 hybrid: hardware eval path plus v5-style compute training route | Practical shape for the V7 line; training speed recovers by using compute | Hardware eval still has readback/state costs; forced hardware train remains slow | Keep as lineage proof that hybrid beats pure hardware training until state is GPU-resident. |
| `variants/v8` / `v8_direct` | V6-derived compute renderer with host-side metadata split | Best measured 4K/64K uniform forward+backward row; exact training math unchanged; good 512/6K wins | Still reads `tile_offsets[-1].item()` for pair sizing; overflow path uses CPU decisions/full-image patching; active policy still CPU-side | Best next training base. Build `v8x` around fixed/cached pair capacity, GPU overflow compaction, tile-local overflow patching, and device-side policy. |
| `variants/v8_hw_eval` | Fail-closed hardware eval scaffold | Safe API surface for requesting hardware eval and falling back to V8 compute | No real render output/state; reports unsupported by design | Mostly superseded by V9 interop probes. Keep as fallback/scaffold reference. |
| `variants/v8_hw_train` | Fail-closed hardware-training state scaffold | Useful state-mode API and memory accounting for `tile_stop`, `final_T`, `pixel_stop` | No hardware forward state, no real backward interop | Keep as design reference for state contracts. |
| `variants/v8_project3d` | V5-style rasterizer plus pinhole 3D projection ops | Adds Metal pinhole projection and code/engineering notes indicate projection VJP support | README caveat appears stale against code/engineering notes; needs current reference/benchmark audit | Re-verify before use. Decide whether it is superseded by `v9_project3d_train`. |
| `variants/v9_project3d_train` | V5-style rasterizer plus training-ready pinhole 3D projection VJP | Staged raster backward -> projection backward; gradients through means/scales/quats/opacities/colors/camera/intrinsics | Pinhole-only; separate from the V8/V9 hardware-raster track | Candidate for 3D training integration, but should be benchmarked separately from renderer-kernel work. |
| `variants/v9_hw_interop_probe` | Direct render-pass output to Torch/MPS tensor | Proves buffer-backed render target output without native CPU staging; direct path much faster than blit at high res | Not a Gaussian rasterizer | Completed primitive. Use as ancestor evidence only. |
| `variants/v9_hw_fixed_eval_probe` | First Gaussian eval render pipeline over MPS input tensors | Fast screen-space Gaussian quads; direct MPS output | No depth sort, no batching, no exact V8 transmittance, no backward | Superseded by parity/sorted/output-plane probes. |
| `variants/v9_hw_eval_parity_probe` | V9 fixed eval vs V8 comparison harness | Single-splat parity works; multi-splat failures are documented | Multi-splat rows fail because fixed blending is not V8 math | Keep as parity diagnostic. |
| `variants/v9_hw_sorted_eval_probe` | Stable MPS depth-sort wrapper for fixed eval | Deterministic submit order; proves order sensitivity | Sorting alone does not create `T`, stop metadata, or exact backward state | Diagnostic only. |
| `variants/v9_hw_output_planes_probe` | Output format and RGBA16F/R/RG plane probe plus Gaussian eval | Best Metal eval-output base; RGBA16F is promising for bandwidth; sorted format wrappers exist | Eval-only; row alignment constraints; fixed blending remains non-parity | Keep as eval path. Do not promote to training without side-state. |
| `variants/v9_hw_draw_formats_probe` | Direct render target format and ICB probe | Direct format sweep is useful; documents row alignment and no true RGB32F target | ICB execution crashed AGX and is fail-closed | Keep ICB out of shared paths. Revisit only in isolated validation harness. |
| `variants/v9_hw_tile_state_probe` | Tile/imageblock layout and dispatch probe | Measures imageblock memory; tile dispatch over direct MPS target works | Earlier exact init/update/flush was not complete | Superseded by `v9_hw_tile_exact_probe` for exact C/T work. |
| `variants/v9_hw_tile_exact_probe` | Exact imageblock C/T semantic probe plus V8 compute-replay full-backward wrapper | Ordered C/T update works for constant overlap; Gaussian imageblock probe matches CPU at ~1e-7; GPU-written tile stop counts work; full backward is available via V8 compute replay | Gaussian probe still uses diagnostic fullscreen fragment evaluation; no V8 tile-bin ingestion; no real hardware-owned training state yet | Main hardware-forward-state workbench. Next step is clipped projected Gaussian quads from GPU-resident V8 tile bins. |
| `variants/v9_cuda_compute_first` | CUDA source scaffold | Correct strategic direction for CUDA: compute-first tile raster, direct Torch CUDA tensors, compact state, backward replay | No CUDA toolchain/device on this Mac; no native benchmark | Keep as scaffold until CUDA host is available. |

## What Is Actually On The Table

### 1. Finish the v5 sort/unsort fix

Current dirty v5 work adds `RasterConfig.inputs_sorted_by_depth`. If true, v5:

- skips `torch.argsort(depths)` in forward/eval/profile paths;
- skips the batch gather after sort;
- returns gradients directly in backward instead of `_unsort_batched`;
- has a new reference check proving identical image and gradients versus the default sort path on already-sorted inputs.

This is the right fix for the "batch unsort is killing speed" question when the
caller already sorted and saved the order. It is not a generic replacement for
sorting arbitrary unsorted inputs.

Remaining work:

- Keep the Dynaworld parent default enabled only while `project_for_fast_mac(...)` and `project_for_fast_mac_batch(...)` continue to use monotonic `_rank_depths(...)`.
- Re-run the known render smoke and a training microbench with instrumentation for `argsort=0` and `unsort=0`.
- Commit the submodule change first, then commit the Dynaworld parent pointer/config change.

### 2. Promote or integrate the V8 training base

`v8_direct` should be the next fast-mac training base if we are developing the
renderer itself rather than only stabilizing the current Dynaworld adapter.

Useful completed evidence:

- 512/6K matrix: V8 beat V6 direct in 13 of 16 comparable cells.
- 4K/64K uniform F+B: V8 direct was the best measured row at 65.754 ms.
- Math stays V6-compatible: sorted bins, tile stop counts, recompute backward,
  tile-local reductions.

Remaining V8 work:

- Add fixed/cached `pair_capacity` so the hot path does not allocate from `tile_offsets[-1].item()`.
- Add device-side overflow/capacity flags so fixed buffers fail closed.
- Move overflow compaction off CPU.
- Replace full-image overflow clone/scatter with tile-local patch kernels.
- Keep active scheduling conservative and device-stat-driven.

### 3. Turn V9 tile exact from probe into hardware forward state

The V9 tile-exact line is the only current hardware path that can plausibly
become training-relevant. The reason is explicit programmable `C/T` state, not
fixed-function blending.

Next kernel sequence:

```text
V8 sorted bins / tile ranges
  -> GPU draw records or per-tile projected Gaussian refs
  -> clipped projected Gaussian quads
  -> 16x16 imageblock C/T update in stable order
  -> tile_stop_counts matching V8 candidate-prefix semantics
  -> direct Torch/MPS output
  -> V8 compute replay backward
```

Pass gate:

```text
same image, same tile_stop_counts, same gradients as V8
on tiny overlapping Gaussian scenes with invisible/skipped candidates
```

Kill gate:

```text
hardware tile_stop_counts only count visible fragments and diverge from
the V8 candidate prefix needed by backward
```

Only after that gate passes should speed matter.

### 4. Keep V9 output planes as eval-only

`v9_hw_output_planes_probe` is useful if we need a preview/eval renderer:

- direct MPS render targets work;
- RGBA16F can be much faster at high resolution;
- reverse-order sorted eval can match black-background color in controlled cases.

It still lacks:

- exact `T`;
- stop count / processed prefix;
- batching;
- backward contract;
- stable V8-compatible multi-splat semantics.

So it should not become the training renderer unless paired with separate exact
side-state.

### 5. Clean up Project3D status

There are two Project3D variants:

- `v8_project3d`: README says forward-only, but engineering notes and code show a projection backward path.
- `v9_project3d_train`: README and code both frame it as training-ready pinhole projection plus V5-style raster.

This needs a focused audit before use:

- run reference checks for both;
- decide whether `v8_project3d` is stale/duplicate;
- route Dynaworld only to the one with verified projection gradients.

## What Is Left To Complete

Short-term:

- Finish and commit v5 presorted path plus the Dynaworld parent default.
- Re-run sort/unsort timing after the parent renderer passes sorted inputs.
- Add an agent note or README line that `inputs_sorted_by_depth=True` is a contract, not a hint.
- Decide whether Dynaworld should stay on v5 for stability or move to v8 after a common-tensor training benchmark.

Medium-term:

- Build `v8x`: fixed pair capacity, device capacity flag, GPU overflow compaction, tile-local overflow patching.
- Run the same 512/6K and 4K/64K matrix after each kernel change.
- Keep v6-upgrade/refined only as ablation sources unless they beat v8 in a fresh matrix.

Hardware path:

- In `v9_hw_tile_exact_probe`, replace fullscreen Gaussian diagnostic fragments with clipped projected Gaussian quads.
- Feed that path from V8 sorted bins/tile ranges, not CPU-built draw lists.
- Match V8 image and `tile_stop_counts` on scenes with invisible candidates and early stop.
- Keep backward as V8 compute replay until hardware forward state is exact.
- Do not re-enable ICB execution in shared benchmark code.

Project3D path:

- Verify which Project3D fork is authoritative.
- Add current reference/benchmark notes.
- Keep projection/raster integration separate from hardware-raster promotion.

## Final Code Goal

The final renderer should not be "pure hardware raster" or "just another probe."
The goal is a single training-grade fast path with explicit fallbacks:

```text
3D or projected inputs
  -> stable depth order, optionally supplied by caller
  -> GPU-resident tile counts / offsets / binned IDs
  -> fixed or cached pair-capacity buffers in the steady state
  -> exact tile C/T forward
  -> compact stop/capacity state
  -> V8-style compute replay backward with tile reductions
  -> optional hardware eval output path when it is faster and parity-shaped
```

Required properties:

- no CPU readback or CPU shape discovery in the timed training hot path;
- no duplicate sort/unsort when the caller already owns sorted order;
- exact image and gradient parity with the current compute renderer on small
  dense-reference cases;
- 4K/64K B=1 and 512/6K B=4 speed wins against `v8_direct`;
- clear fail-closed fallback for overflow/capacity/hardware unsupported cases;
- one Dynaworld adapter API that can select stable compute, v8x, or hardware
  eval without changing the trainer loop.

Until that exists, the practical split is:

- use compute for training;
- use V9 hardware probes for eval/state research;
- only promote hardware forward when it produces the same state V8 backward
  already consumes.
