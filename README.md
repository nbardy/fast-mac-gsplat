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
  --renderers torch_direct,v2_fastpath,v3_candidate,v5_batched,v6_direct,v6_auto,v6_upgrade_direct,v6_upgrade_auto,v7_hardware \
  --modes forward,forward_backward \
  --output-md benchmarks/full_rasterizer_benchmark.md
```

The full benchmark writes a Markdown report and marks Torch rows as skipped
when the dense reference would be too large.

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
| `v7_hardware` | You want to experiment with Metal render-pipeline forward rasterization. | Not recommended for training yet; backward is too slow at 4K/64K. |

The current `variants/v6` branch already includes later local engineering fixes
on top of the original v6 line. `variants/v6_upgrade` is preserved separately so
the source handoff can be tested without overwriting the current v6 baseline.

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
| v7 | `torch_gsplat_bridge_v7` | `[G,2/3]` or `[B,G,2/3]`, `[G]` or `[B,G]` | `[H,W,3]` or `[B,H,W,3]` |

The call pattern is intentionally stable across variants. The `RasterConfig`
fields are not identical:

- v2/v3 are single-image only. Loop outside the renderer if you need batches.
- v5 adds native batching, eval/train split, overflow fallback, and batch chunking.
- v6 keeps the v5 API shape and adds active-tile policy and stop-count controls.
- v6-upgrade also imports as `torch_gsplat_bridge_v6`, so install it in a
  separate environment from `variants/v6` if you need to compare package APIs.
- v7 uses the same call shape, but has a smaller config and is experimental.

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
python -m pip install "git+https://github.com/nbardy/fast-mac-gsplat.git#subdirectory=variants/v7"
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
