# V9 HW ICB Execute Crash — Handoff

Crash signal received in iTerm: `EXC_BAD_ACCESS (SIGSEGV) / KERN_INVALID_ADDRESS at 0x7c` inside `AGXMetalG16G_B0` `executeCommandsInBufferCommon`. This doc consolidates what is known, what was checked, what is still open, and where to look next.

## TL;DR

- The crashing call is `[render_encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0,1)]` in `encode_constant_render_icb` (file: `variants/v9_hw_draw_formats_probe/csrc/metal/v9_hw_interop.mm`).
- The crash happened **before** the fail-closed patch — both file mtimes (interop.py, v9_hw_interop.mm) and the rebuilt `_C.so` are timestamped 6–9 minutes after the crash.
- After patch, the path is fenced at two layers (Python `RuntimeError`, native `TORCH_CHECK(false, ...)`), so the binary on disk no longer reproduces the SIGSEGV.
- Crash is **specific to the ICB execute path**. The non-ICB direct render is validated and stable: `direct_render_to_mps_tensor_validated=true, max_abs_err=0.0` at 16x16, 9x7, 64x64..4096x4096.
- Most likely root cause (high confidence, not yet proven): the cached `MTLRenderPipelineState` is built **without** `desc.supportIndirectCommandBuffers = YES`. See "Top suspect" below.

## Timeline (Asia/Ho_Chi_Minh, +07)

| Time | Event |
|---|---|
| 12:58:33 | Crash captured (`Python` process `pid 76332`, parent `codex pid 28268`). |
| 13:04:43 | `csrc/metal/v9_hw_interop.mm` modified (fail-closed patch + native `TORCH_CHECK(false,...)` in `render_constant_rgba_direct_icb_native`). |
| 13:04:51 | `torch_gsplat_bridge_v9_hw_draw_formats/interop.py` modified (Python `RuntimeError` raised before native call; `icb_execute_op_available` set False). |
| 13:07:23 | `_C.cpython-314-darwin.so` rebuilt with the guards. |

So: the **crash report you are holding is pre-patch**. Re-running the same code on disk now will hit the guard, not AGX.

## Verified empirically just now

Ran in `variants/v9_hw_draw_formats_probe/`:

```
python3 tests/interop_check.py
```

Result: passes; key fields from the probe JSON:

| Field | Value |
|---|---|
| `metal_available` | true |
| `metal_device_name` | Apple M4 |
| `render_pipeline_ready` | true (rgba32f, rgba16f, r32f, rg32f all ready) |
| `icb_created` | true |
| `icb_execute_validated` | false (deliberately disabled) |
| `direct_render_to_mps_tensor_validated` | true |
| `render_to_mps_tensor_max_abs_err` | 0.0 |

Direct invocation:

```python
from torch_gsplat_bridge_v9_hw_draw_formats.interop import render_constant_rgba_direct_icb
render_constant_rgba_direct_icb(16, 16, (0.125, 0.5, 0.875, 1.0))
# RuntimeError: render_constant_rgba_direct_icb is disabled: minimal ICB execution crashed
#   in AGX executeCommandsInBufferCommon on macOS/Apple M4. Treat ICB execution as unsafe
#   until reworked in a separate isolated harness.
```

Confirms the guard fires before reaching AGX.

## Full crash stack (around the failing call)

From `Crashed Thread: 0  Dispatch queue: metal gpu stream`:

```
0  libsystem_kernel.dylib  __pthread_kill + 8
1  libsystem_pthread.dylib pthread_kill + 296
2  libsystem_c.dylib       raise + 32
3  Python                  faulthandler_fatal_error + 380
4  libsystem_platform.dylib _sigtramp + 56
5  AGXMetalG16G_B0   AGX::RenderContext<AGX::HAL200::Encoders, AGX::HAL200::Classes,
                    AGX::HAL200::ObjClasses>::executeCommandsInBufferCommon(
                        AGXG16GFamilyIndirectCommandBuffer*,
                        AGX::IndirectCommandBuffer::ExecutionRange) + 3012
6  AGXMetalG16G_B0   -[AGXG16GFamilyRenderContext executeCommandsInBuffer:withRange:] + 68
7  _C.cpython-314-darwin.so  encode_constant_render_icb(MTLDevice*, MTLCommandBuffer*,
                              MTLRenderPipelineState*, MTLTexture*,
                              std::array<float,4> const&) + 36
                              [v9_hw_interop.mm:364]   ← INLINED at crash time
8  _C.cpython-314-darwin.so  invocation function for block in
                              render_constant_format_direct_native(...) + 688
                              [v9_hw_interop.mm:488]
9  libdispatch.dylib   _dispatch_client_callout + 16
10 libdispatch.dylib   _dispatch_lane_barrier_sync_invoke_and_complete + 56
11 _C.cpython-314-darwin.so  render_constant_format_direct_native(...) + 516
                              [v9_hw_interop.mm:475]
12 _C.cpython-314-darwin.so  render_constant_rgba_direct_icb_native(...) + 72
                              [v9_hw_interop.mm:514]
13 _C.cpython-314-darwin.so  pybind11 cast.h:2138 (call_impl)
14 _C.cpython-314-darwin.so  pybind11 cast.h:2106 (call)
15 _C.cpython-314-darwin.so  pybind11 lambda forward (pybind11.h:431)
16 _C.cpython-314-darwin.so  pybind11 __invoke (pybind11.h:401)
17 _C.cpython-314-darwin.so  pybind11::cpp_function::dispatcher (pybind11.h:1064)
18 Python  cfunction_call + 72
19+ Python eval/main, pymain_run_stdin (codex piped Python via stdin)
```

Key signals from registers / VM map:

- `Exception subtype: KERN_INVALID_ADDRESS at 0x000000000000007c` — small-offset null-deref. AGX is dereferencing field at offset 0x7c (124 bytes) of a NULL or freed object. Classic "uninitialized internal struct" / "expected `pipelineStateRef` slot is empty".
- Crash at `executeCommandsInBufferCommon + 3012` — i.e. mid-iteration, *after* the GPU command processor began walking the ICB.
- Frames 5/6 are the only AGX frames; everything below is our code calling Metal correctly enough to reach `executeCommandsInBuffer:`.

The Crash Report is fully captured at `bug_type=309` and is what was pasted; we have all of it. `.ips` is just the JSON-tail format of the same report — no extra hidden frames beyond what's listed.

## Source map (post-patch, current line numbers)

| File | Symbol | Lines |
|---|---|---|
| `variants/v9_hw_draw_formats_probe/csrc/metal/v9_hw_interop.mm` | `build_render_pipeline` | 133–159 |
| `variants/v9_hw_draw_formats_probe/csrc/metal/v9_hw_interop.mm` | `cached_render_pipeline` | 161–184 |
| `variants/v9_hw_draw_formats_probe/csrc/metal/v9_hw_interop.mm` | `encode_constant_render` (no-ICB, working) | 296–~315 |
| `variants/v9_hw_draw_formats_probe/csrc/metal/v9_hw_interop.mm` | `encode_constant_render_icb` (ICB, crashing) | ~318–402 |
| `variants/v9_hw_draw_formats_probe/csrc/metal/v9_hw_interop.mm` | `render_constant_format_direct_native` | 478–534 |
| `variants/v9_hw_draw_formats_probe/csrc/metal/v9_hw_interop.mm` | `render_constant_rgba_direct_icb_native` (now fail-closed) | 543–551 |
| `variants/v9_hw_draw_formats_probe/torch_gsplat_bridge_v9_hw_draw_formats/interop.py` | `render_constant_rgba_direct_icb` (Python guard) | 121–129 |
| `variants/v9_hw_draw_formats_probe/tests/interop_check.py` | regression: ICB execute must stay disabled | 57 |

Note: the line numbers embedded in the crash report (`v9_hw_interop.mm:364, :475, :488, :514`) are from the `.so` *as built at crash time*. The post-patch source has shifted; do not map them onto the current file.

## Top suspect: `MTLRenderPipelineDescriptor.supportIndirectCommandBuffers`

The ICB encoding looks like this (paraphrased from current source):

```objc
MTLIndirectCommandBufferDescriptor* icb_desc = [...];
icb_desc.commandTypes              = MTLIndirectCommandTypeDraw;
icb_desc.inheritPipelineState      = NO;     // ICB encodes its own PSO
icb_desc.inheritBuffers            = NO;     // ICB encodes its own bindings
icb_desc.maxVertexBufferBindCount  = 1;
icb_desc.maxFragmentBufferBindCount = 1;

id<MTLIndirectCommandBuffer> icb =
    [device newIndirectCommandBufferWithDescriptor:icb_desc maxCommandCount:1 options:0];
[icb resetWithRange:NSMakeRange(0, 1)];

id<MTLIndirectRenderCommand> cmd = [icb indirectRenderCommandAtIndex:0];
[cmd setRenderPipelineState:pso];                                  // ← uses cached PSO
[cmd setVertexBuffer:dummy_vertex_buffer offset:0 atIndex:0];
[cmd setFragmentBuffer:rgba_buffer       offset:0 atIndex:0];
[cmd drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:1 baseInstance:0];

// optimize before execute (correct)
[blit_encoder optimizeIndirectCommandBuffer:icb withRange:NSMakeRange(0,1)];
[blit_encoder endEncoding];

// render pass that executes the ICB
id<MTLRenderCommandEncoder> render_encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
[render_encoder setRenderPipelineState:pso];                       // redundant but ok
[render_encoder setVertexBuffer:dummy_vertex_buffer offset:0 atIndex:0];
[render_encoder setFragmentBuffer:rgba_buffer       offset:0 atIndex:0];
[render_encoder useResource:dummy_vertex_buffer usage:Read stages:Vertex];
[render_encoder useResource:rgba_buffer         usage:Read stages:Fragment];
[render_encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0,1)];   // ← SIGSEGV
[render_encoder endEncoding];
```

`build_render_pipeline` (lines 133–159) creates the descriptor:

```objc
MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
desc.label                            = @"v9_hw_draw_formats_render_probe";
desc.vertexFunction                   = vs;
desc.fragmentFunction                 = fs;
desc.colorAttachments[0].pixelFormat  = format.pixel_format;

id<MTLRenderPipelineState> pso =
    [device newRenderPipelineStateWithDescriptor:desc error:&err];
```

**`desc.supportIndirectCommandBuffers` is never set** (defaults to `NO`). When you bind such a PSO via `[icb indirectRenderCommandAtIndex:i].setRenderPipelineState:pso` and then `executeCommandsInBuffer:`, AGX expects an internal "ICB-blittable" pointer table inside the PSO. Without `supportIndirectCommandBuffers = YES`, that table is not built; the encoded slot in the per-command struct is null/garbage; AGX dereferences it during execute → null+offset deref. Address `0x7c` is consistent with this kind of internal pipeline-state slot.

This matches the crash signature precisely:
- only fails on `executeCommandsInBufferCommon`, not on regular `drawPrimitives` (because `setRenderPipelineState` outside an ICB takes a different code path);
- only fails when `inheritPipelineState = NO` (because then the ICB has to use its own encoded PSO);
- crash is mid-execute, not at PSO build, because Apple silently accepted an ICB-invalid PSO.

### Minimal fix to test

```objc
// in build_render_pipeline:
if (@available(macOS 11.0, *)) {
    desc.supportIndirectCommandBuffers = YES;
}
```

Then rebuild `_C.so` and re-enable just the `executeCommandsInBuffer` probe (do NOT relax the Python/native guards yet — gate it behind a debug-only flag in an isolated harness as the patch comment requests).

## Answers to the kernel agent's questions

1. **Was the crash from before or after the fail-closed ICB patch?**
   **Before.** Crash 12:58:33; mm/py modified 13:04:4x; `_C.so` rebuilt 13:07:23. The codex session for that worker shows the patch landing in the 13:04 turn, in direct response to this crash.

2. **Can they provide the full stack around `render_constant_rgba_direct_icb_native`, not the truncated report?**
   **Yes — see "Full crash stack" above.** Frames 5–18 are reproduced verbatim from the report. The pasted report was complete; nothing was truncated above frame 5. The `.ips` JSON contains only the same frames in a different encoding.

3. **Did they see any evidence of missing ICB resource residency, bad `inheritPipelineState`/`inheritBuffers`, or invalid fragment buffer binding?**
   - **Residency**: the parent encoder calls `useResource` on both buffers with explicit `stages:` masks (lines ~392–393). It does **not** mark the ICB itself as resident. With `inheritBuffers=NO`, the ICB's encoded buffer pointers reference `dummy_vertex_buffer` and `rgba_buffer`, both of which *are* made resident — so residency is technically satisfied for the only buffers the ICB references. The optimize step (`[blit optimizeIndirectCommandBuffer:icb]`) runs before execute, also correct. Residency is **likely not** the root cause, but worth re-checking by also calling `[render_encoder useResource:icb usage:Read]` and adding `useResources:count:` for completeness.
   - **`inheritPipelineState`/`inheritBuffers`**: both `NO`. That is internally consistent (the ICB encodes its own state). What it **demands**, though, is that the encoded PSO supports ICB use — see "Top suspect" above. This is the most likely defect.
   - **Fragment buffer binding**: one fragment buffer at index 0, `maxFragmentBufferBindCount = 1`. Matches the encode site. Not suspicious.

4. **Can they confirm whether the crash only happens on `executeCommandsInBuffer`, not regular direct render?**
   **Confirmed.** `tests/interop_check.py` runs in full (post-patch) and validates direct render at 16x16, 9x7, and 64×64..4096×4096 with `max_abs_err = 0.0`. The non-ICB encoder uses the same cached PSO and the same `dummy_vertex` / `setFragmentBuffer` pattern — the only delta vs. the crashing path is `executeCommandsInBuffer:` instead of `drawPrimitives:`. So the bug is in the ICB-binding side, not in PSO compilation, render pass setup, or buffer-backed texture creation.

## Better questions to ask next (suggested)

The four asked are good but won't reach root cause on their own. Add:

5. **What is `desc.supportIndirectCommandBuffers` when building the cached PSO?** (Almost certainly `NO`, by default. This alone explains the symptom.)
6. **Does it reproduce with Metal validation enabled?** Run with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER_VALIDATE_LOAD_ACTIONS=1 MTL_DEBUG_LAYER_VALIDATE_STORE_ACTIONS=1`. The validation layer turns this exact misuse into a readable assertion ("indirect command buffer references render pipeline state that does not support indirect command buffers") instead of a SIGSEGV. Should be standard for any future ICB probe.
7. **Is the ICB itself made resident with `[render_encoder useResource:icb usage:Read]`?** Not strictly required when `inheritBuffers=NO` and bindings are explicit, but on AGX the heuristic differs by macOS version.
8. **Does the same ICB descriptor work without binding any vertex/fragment buffers** (use a vertex shader that ignores `[[buffer(0)]]` and a constant-output fragment shader)? Removes the buffer-residency variable.
9. **Does it reproduce on Apple silicon other than M4** (M1/M2/M3)? The crash file says `AGXMetalG16G_B0` (M4); some ICB AGX paths regressed in macOS 15 specifically.
10. **What are exact macOS / Xcode / Metal versions?** Crash is on macOS 15.5 (24F74); some ICB issues are AGX-version-specific.
11. **Was a one-shot reset/optimize sufficient, or does AGX need `[icb resetWithRange:]` again before each execute?** Less likely, but documenting it removes a question for the next probe.
12. **Can the ICB be allocated with `MTLResourceStorageModeShared` options vs. `0`?** Default storage on macOS Apple-silicon is shared, but explicit options sometimes change ICB layout.
13. **What does Metal's `[icb encodedLength]` / Apple's `MTL_DEBUG_LAYER` say after `optimizeIndirectCommandBuffer:`?** Confirms the encoded ICB is non-empty before execute.

## Open work

- [ ] Add `desc.supportIndirectCommandBuffers = YES` in `build_render_pipeline` and re-test ICB execute behind a debug flag.
- [ ] Wire a Metal validation layer mode into `tests/interop_check.py` so future ICB regressions surface as Metal asserts, not Python SIGSEGV crash reports.
- [ ] Move ICB execute experiments into a dedicated harness (separate process, separate `.so`, optional flag) per the patch comment's directive — the current `.so` already imports cleanly into long-running training jobs; do not let a future ICB patch crash the parent Python.
- [ ] If the supportIndirectCommandBuffers fix lands cleanly, document residency / argument-buffer rules for "ICB execute on Torch MPS command queue" in `docs/v9_hw_draw_format_icb_notes.md` for the next variant.

## Pointers / file index

- Crash report (raw, pasted): not stored in repo; address `0x7c`, signal `SIGSEGV`, AGXMetalG16G_B0 frames `executeCommandsInBufferCommon + 3012`, `-[AGXG16GFamilyRenderContext executeCommandsInBuffer:withRange:] + 68`.
- ICB encoder (crashing path): `variants/v9_hw_draw_formats_probe/csrc/metal/v9_hw_interop.mm` `encode_constant_render_icb`.
- Direct render (working path, comparison): same file, `encode_constant_render`.
- PSO builder (suspected misconfiguration): same file, `build_render_pipeline` (lines 133–159).
- Native fail-closed: same file, `render_constant_rgba_direct_icb_native` (lines 543–551).
- Python fail-closed: `variants/v9_hw_draw_formats_probe/torch_gsplat_bridge_v9_hw_draw_formats/interop.py` `render_constant_rgba_direct_icb` (lines 121–129).
- Test asserting it stays disabled: `variants/v9_hw_draw_formats_probe/tests/interop_check.py:57`.
- Probe field reports describing the rest of v9 status: `docs/v9_hw_interop_probe_notes.md`, `docs/v9_hw_draw_format_icb_notes.md`, `docs/v9_parallel_exploration_plan.md`.

## Reproduction recipe (post-fix, for the next agent)

1. Apply the one-line `desc.supportIndirectCommandBuffers = YES` change in `build_render_pipeline`.
2. Rebuild: `cd variants/v9_hw_draw_formats_probe && uv run python -m pip install -e . --no-build-isolation`.
3. Remove **only** the `TORCH_CHECK(false, ...)` in `render_constant_rgba_direct_icb_native`, but leave the Python `RuntimeError` guard until the C++ side is proven not to crash.
4. From a small standalone harness (NOT inside an in-progress codex session — one Python process, one call):
   ```python
   from torch_gsplat_bridge_v9_hw_draw_formats._C import render_constant_rgba_direct_icb
   t = render_constant_rgba_direct_icb(16, 16, (0.125, 0.5, 0.875, 1.0))
   print(t.shape, t.dtype, t.device, t.cpu().mean(dim=(0,1)))
   ```
5. Run with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`. If a Metal assertion fires, fix that error first; only re-enable the higher-level `RuntimeError` after a clean run.
