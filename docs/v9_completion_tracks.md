# V9 Completion Tracks

## Active Rule

ICB allocation may be probed, but ICB execution stays disabled in shared
variants. The pre-patch crash was likely caused by a pipeline that did not set
`supportIndirectCommandBuffers = YES`, but that fix belongs in a separate
validation harness with Metal validation enabled.

## Current Tracks

| Fork / doc | Goal | Status | Outcome |
|---|---|---|---|
| `variants/v9_hw_eval_parity_probe` | Compare fixed eval against v8 forward on identical projected tensors. | Complete | Single-splat black-background rows match v8 within ~`1.5e-8`; multi-splat rows do not match because fixed hardware blending is not a v8 ordering/state contract. |
| `variants/v9_hw_sorted_eval_probe` | Add depth-sorted fixed eval wrapper and order tests. | Complete | Stable MPS sort and deterministic submit order work, but sorting alone does not recover v8 transmittance or stop-state semantics. |
| `variants/v9_hw_output_planes_probe` | Reduce output bandwidth via lower precision or split planes. | Complete | `RGBA16F` Gaussian eval is runnable and validates with expected half precision error; 4K/64K median improved from `6.123 ms` (`RGBA32F`) to `4.958 ms` (`RGBA16F`) in this probe. |
| `docs/v9_cuda_hardware_rasterization_notes.md` | Map Metal V9 lessons to CUDA. | Complete | CUDA should start as compute-first tile rasterization using CUB scan/sort, direct Torch CUDA tensor writes, compact forward state, and backward replay reductions; graphics hardware raster belongs in a separate Vulkan interop branch. |

## Current Decision

The best Metal eval base is now `v9_hw_output_planes_probe`: it carries direct
MPS output, Gaussian tensor inputs, and the `RGBA16F` output candidate. It is
still not a v8 replacement because multi-splat parity fails.

For training/backward, `variants/v8` remains the baseline. Hardware raster
forward can be paired with backward only after it exposes the same ordered
prefix, final transmittance, and stop metadata that v8 backward expects, either
through programmable tile/imageblock state or a compute fallback that recomputes
the exact prefix.

## Promotion Criteria

Promote a v9 mainline only if:

- direct MPS render output remains stable;
- v9 fixed eval beats v8 forward on at least one realistic parity-shaped case;
- image error is explained and bounded;
- output format choice does not add a hidden full-frame copy;
- ICB execution remains out of the hot path.

Current status: the first and fourth criteria are met for the safe probes; the
second and third are not met for multi-splat scenes.
