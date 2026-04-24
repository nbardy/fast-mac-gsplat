# fast-mac-gsplat

Torch-first, Metal-backed Gaussian rasterizer fast path for Apple Silicon.

Bringing World Model Research to your mac

## Benchmarks

  | Splats | Mode | Torch | v5 | % Increase |
  |---:|---|---:|---:|---:|
  | 512 | Forward | 751.724 ms | 3.014 ms | +24,840% |
  | 512 | Forward+backward | 5109.671 ms | 7.838 ms | +65,094% |
  | 1024 | Forward | 1712.214 ms | 4.355 ms | +39,219% |
  | 1024 | Forward+backward | OOM | 11.373 ms | ♾️ %|
  | 64k | Forward | 94992.427 ms | 9.644 ms | +984,907% |
  | 64k | Forward+backward | OOM | 27.457 ms | ♾️ % |

For broad sweeps across renderers, resolutions, splat counts, and projected
splat distributions:

```bash
python benchmarks/benchmark_full_matrix.py \
  --resolutions 512x512,1024x512,1920x1080,4096x4096 \
  --splats 512,2048,65536 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers torch_direct,v2_fastpath,v3_candidate,v5_batched,v6_direct,v6_auto,v6_upgrade_direct,v6_upgrade_auto,v6_refined_direct,v6_refined_auto,v8_direct,v7_hardware,v7_finished_hardware,v7_frontk_k2,v72_tiled_k2 \
  --modes forward,forward_backward \
  --output-md benchmarks/full_rasterizer_benchmark.md
```

The full benchmark writes a Markdown report and marks Torch rows as skipped
when the dense reference would be too large.
Add `--accuracy` to compare image and gradient outputs against the dense Torch
reference for cells under `--accuracy-max-work-items`; larger cells keep speed
timing and mark accuracy as skipped.
See `docs/v6_refined_field_report.md` for the first v6-refined smoke matrix.
See `docs/v72_tiled_capture_field_report.md` for the v7.2 tiled-capture results.
See `docs/hardware_rasterizer_fast_backward_handoff.md` for the current
handoff on merging hardware rasterization with fast compute backward.
See `docs/v8_kernel_design_iteration.md` for the current v8 kernel design
notebook and promotion plan.
See `docs/v8_critical_theory_notes.md` for the critical shader audit,
backward-math equivalence notes, and v8 stop list.
See `docs/v8_field_report.md` for the first v8 implementation, correctness
checks, and benchmark results.
See `docs/v8_hw_tile_raster_plans.md` for the hardware tile/imageblock shader
design and two implementation plans.
See `docs/v8_hw_eval_notes.md` and `docs/v8_hw_train_notes.md` for the first
fallback-safe implementations of those two plans.
See `docs/v8_hw_theory_remaining_work.md` for the post-scaffold audit of the
remaining Metal render-pipeline, imageblock, ROG, ICB, and Torch/MPS interop
work.

## Variants

| Variant | Use When | Status |
|---|---|---|
| `v2_fastpath` | You need the older low-overhead single-image compute baseline for comparisons. | Kept as a benchmark baseline. |
| `v3_candidate` | You want the strongest measured B=1 large-scene training path. | Good single-image training baseline. |
| `v5_batched` | You want native `[B,G,...]` batch rendering and the simpler batched compute path. | Viable training candidate after the saturated-backward fix. |
| `v6_direct` | You want the current default batch-focused training path for B>1. | Preferred batch training baseline. |
| `v6_auto` | You want v6 to enable active-tile scheduling only for sparse, clustered, or overflow-heavy scenes. | Experimental policy mode; benchmark against direct. |
| `v6_upgrade_direct` | You want to test the chief-scientist v6-upgrade handoff without replacing local v6. | Preserved source handoff and benchmark target. |
| `v6_upgrade_auto` | You want to test the v6-upgrade active-policy path. | Preserved source handoff and benchmark target. |
| `v6_refined_direct` | You want to test the v6-refined handoff beside local v6 and v6-upgrade. | Preserved source handoff with local saturated-backward safety fix. |
| `v6_refined_auto` | You want to test the v6-refined active-policy path. | Preserved source handoff with local saturated-backward safety fix. |
| `v8_direct` | You want the current v6-derived v8 training baseline with host-side metadata parsing. | Best measured 4K/64K uniform forward+backward row so far; not best for every forward-only case. |
| `v8_hw_eval_fallback` | You want to exercise the Plan 1 hardware-eval API surface while preserving exact v8 compute behavior. | Implemented as a fail-closed scaffold; real imageblock/render-pipeline eval is still pending. |
| `v8_hw_train_fallback` | You want to exercise the Plan 2 hardware-training state/capture API surface while preserving exact v8 compute behavior. | Implemented as a fail-closed scaffold; real hardware forward state and backward interop are still pending. |
| `v7_hardware` | You want to experiment with Metal render-pipeline forward rasterization. | Not recommended for training yet; backward is too slow at 4K/64K. |
| `v7_finished_hardware` | You want to test the finished v7 hardware handoff beside local v7. | Preserved source handoff; forward can be benchmarked, but backward gradients currently fail dense-reference checks. |
| `v7_frontk_k2` / `v7_frontk_k4` / `v7_frontk_k8` | You want to test the v7.1 front-K hardware backward handoff with different saved-per-pixel depths. | Preserved source handoff with local Metal compile/order/bounds fixes; exactness smoke passes, but 4K capture is too slow for the default training path. |
| `v72_tiled_k2` / `v72_tiled_k4` / `v72_tiled_k8` | You want to test the v7.2 tiled-capture hardware backward handoff. Add `_t32` or another `_tN` suffix to change tile size from the default 16. | Preserved source handoff with local Metal compile/order/bounds fixes; exactness smoke passes and full benchmarks are tracked in the field report. |

The current `variants/v6` branch already includes later local engineering fixes
on top of the original v6 line. `variants/v6_upgrade` is preserved separately so
the source handoff can be tested without overwriting the current v6 baseline.
`variants/v6_refined` is preserved the same way for the refined handoff.

## API

The renderer variants use the same high-level API shape:

```python
from torch_gsplat_bridge_vX import RasterConfig, ProjectedGaussianRasterizer, rasterize_projected_gaussians

config = RasterConfig(height=height, width=width, background=(1.0, 1.0, 1.0))
image = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, config)
```

Inputs are projected 2D splats on MPS tensors:

- `means2d`: pixel-space centers
- `conics`: packed inverse covariance terms `[xx, xy, yy]`
- `colors`: RGB
- `opacities`: alpha scale
- `depths`: used for stable sort, with zero depth-order gradients

Shape support depends on the variant:

| Variant | Import Package | Input Shape | Output Shape |
|---|---|---|---|
| root v2 | `torch_gsplat_bridge_fast` | `[G,2/3]`, `[G]` | `[H,W,3]` |
| v3 | `torch_gsplat_bridge_v3` | `[G,2/3]`, `[G]` | `[H,W,3]` |
| v5 | `torch_gsplat_bridge_v5` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |
| v6 | `torch_gsplat_bridge_v6` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |
| v6-upgrade | `torch_gsplat_bridge_v6` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |
| v6-refined | `torch_gsplat_bridge_v6` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |
| v8 | `torch_gsplat_bridge_v8` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |
| v8-hw-eval | `torch_gsplat_bridge_v8_hw_eval` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |
| v8-hw-train | `torch_gsplat_bridge_v8_hw_train` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |
| v7 | `torch_gsplat_bridge_v7` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |
| v7-finished | `torch_gsplat_bridge_v7` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |
| v7.1-frontK | `torch_gsplat_bridge_v71` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |
| v7.2-tiled-capture | `torch_gsplat_bridge_v72` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |

The call pattern is intentionally stable across variants. The `RasterConfig`
fields are not identical:

- v2/v3 are single-image only. Loop outside the renderer if you need batches.
- v5 adds native batching, eval/train split, overflow fallback, and batch chunking.
- v6 keeps the v5 API shape and adds active-tile policy and stop-count controls.
- v8 keeps v6 direct math and adds host-side metadata parsing under a separate
  `torch_gsplat_bridge_v8` package.
- v8-hw-eval adds `use_hardware_eval`, `hardware_eval_policy`, `emit_final_T`,
  and `emit_stop_count`; it currently reports unsupported hardware eval and
  falls back to exact v8 compute unless `hardware_eval_policy="require"`.
- v8-hw-train adds `use_hardware_train`, `hardware_train_policy`, capture flags,
  and `backward_state_mode`; it currently reports unsupported hardware training
  and falls back to exact v8 compute unless `hardware_train_policy="strict"`.
- v6-upgrade and v6-refined also import as `torch_gsplat_bridge_v6`, so install them in a
  separate environment from `variants/v6` if you need to compare package APIs.
- v7 and v7-finished use the same call shape, but have a smaller config and are experimental.
- v7.1-frontK adds `front_k` to choose how many visible contributors are saved per pixel before the backward replay fallback.
- v7.2-tiled-capture adds `front_k` and `tile_size`; benchmark names use `v72_tiled_k2` and optional tile suffixes like `v72_tiled_k2_t32`.

Install the current stable batched package directly from GitHub:

```bash
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v5"
```

Other installable variants:

```bash
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v3"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v6"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v6_upgrade"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v6_refined"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v8"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v8_hw_eval"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v8_hw_train"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v7"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v7_finished"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v7_frontk"
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v7_tiled_capture"
```

Use projected 2D splats on MPS tensors:

```python
import torch
from torch_gsplat_bridge_v5 import RasterConfig, rasterize_projected_gaussians

device = torch.device("mps")
batch_size, gaussians = 1, 1024
height, width = 1024, 512

means2d = torch.rand((batch_size, gaussians, 2), device=device)
means2d[..., 0] *= width
means2d[..., 1] *= height

sigma = torch.full((batch_size, gaussians), 4.0, device=device)
inv_var = 1.0 / (sigma * sigma)
conics = torch.stack([inv_var, torch.zeros_like(inv_var), inv_var], dim=-1)

colors = torch.rand((batch_size, gaussians, 3), device=device)
opacities = torch.full((batch_size, gaussians), 0.5, device=device)
depths = torch.rand((batch_size, gaussians), device=device)

config = RasterConfig(height=height, width=width, background=(1.0, 1.0, 1.0))
image = rasterize_projected_gaussians(means2d, conics, colors, opacities, depths, config)

loss = image.square().mean()
loss.backward()
```

Inputs may be `[G, ...]` for one image or `[B, G, ...]` for batched rendering. The output is `[H, W, 3]` or `[B, H, W, 3]`.

## Design goals

- Keep the Python training loop in **PyTorch**.
- Use **Metal kernels** for the renderer hot path.
- Minimize memory during training by saving only compact tile bins, **not** dense `G x H x W` activations.
- Keep backward differentiable via **recompute**, not via storing per-pixel alpha / transmittance graphs.
